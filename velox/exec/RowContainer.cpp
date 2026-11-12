/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/exec/RowContainer.h"

#include "velox/common/memory/RawVector.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/ContainerRowSerde.h"
#include "velox/exec/Operator.h"
#include "velox/type/FloatingPointUtil.h"

namespace facebook::velox::exec {
namespace {
template <TypeKind Kind>
int32_t kindSize() {
  return sizeof(typename KindToFlatVector<Kind>::HashRowType);
}

int32_t typeKindSize(TypeKind kind) {
  if (kind == TypeKind::UNKNOWN) {
    return sizeof(UnknownValue);
  }

  return VELOX_DYNAMIC_TYPE_DISPATCH(kindSize, kind);
}

// The number of flags (bits) per accumulator, one for null and one for
// initialized.
constexpr int32_t kNumAccumulatorFlags = 2;

// Returns the larger of alignments 'a' and 'b', which must be powers of 2,
// so the larger is a multiple of the smaller.
int32_t combineAlignments(int32_t a, int32_t b) {
  VELOX_CHECK_EQ(__builtin_popcount(a), 1, "Alignment can only be power of 2");
  VELOX_CHECK_EQ(__builtin_popcount(b), 1, "Alignment can only be power of 2");
  return std::max(a, b);
}
} // namespace

Accumulator::Accumulator(Aggregate* aggregate, TypePtr spillType)
    : isFixedSize_{aggregate->isFixedSize()},
      fixedSize_{aggregate->accumulatorFixedWidthSize()},
      usesExternalMemory_{aggregate->accumulatorUsesExternalMemory()},
      alignment_{aggregate->accumulatorAlignmentSize()},
      spillType_{std::move(spillType)},
      spillExtractFunction_{
          [aggregate](folly::Range<char**> groups, VectorPtr& result) {
            aggregate->extractAccumulators(
                groups.data(), groups.size(), &result);
          }},
      destroyFunction_{[aggregate](folly::Range<char**> groups) {
        aggregate->destroy(groups);
      }} {
  VELOX_CHECK_NOT_NULL(aggregate);
}

Accumulator::Accumulator(
    bool isFixedSize,
    int32_t fixedSize,
    bool usesExternalMemory,
    int32_t alignment,
    TypePtr spillType,
    std::function<void(folly::Range<char**> groups, VectorPtr& result)>
        spillExtractFunction,
    std::function<void(folly::Range<char**> groups)> destroyFunction)
    : isFixedSize_{isFixedSize},
      fixedSize_{fixedSize},
      usesExternalMemory_{usesExternalMemory},
      alignment_{alignment},
      spillType_{std::move(spillType)},
      spillExtractFunction_{std::move(spillExtractFunction)},
      destroyFunction_{std::move(destroyFunction)} {
  VELOX_CHECK(
      !usesExternalMemory_ || destroyFunction_ != nullptr,
      "An accumulator that uses external memory requires a destroy function.");
}

bool Accumulator::isFixedSize() const {
  return isFixedSize_;
}

int32_t Accumulator::fixedWidthSize() const {
  return fixedSize_;
}

bool Accumulator::usesExternalMemory() const {
  return usesExternalMemory_;
}

int32_t Accumulator::alignment() const {
  return alignment_;
}

void Accumulator::destroy(folly::Range<char**> groups) {
  if (destroyFunction_) {
    destroyFunction_(groups);
  }
}

const TypePtr& Accumulator::spillType() const {
  return spillType_;
}

void Accumulator::extractForSpill(
    folly::Range<char**> groups,
    VectorPtr& result) const {
  spillExtractFunction_(groups, result);
}

// static
RowColumnStats RowContainer::columnStats(
    std::span<RowContainer* const> containers,
    column_index_t columnIndex) {
  std::vector<RowColumnStats> stats;
  stats.reserve(containers.size());
  for (const auto* container : containers) {
    stats.push_back(container->columnStats(columnIndex));
  }
  return RowColumnStats::merge(stats);
}

// static
RowColumnStats RowColumnStats::merge(
    const std::vector<RowColumnStats>& statsList) {
  RowColumnStats mergedStats;
  for (const auto& stats : statsList) {
    mergedStats.nullCount_ += stats.nullCount_;
    mergedStats.sumBytes_ += stats.sumBytes_;
    mergedStats.maxBytes_ = std::max(mergedStats.maxBytes_, stats.maxBytes_);
  }
  return mergedStats;
}

std::string RowContainerIterator::toString() const {
  return fmt::format(
      "[allocationIndex:{} rowOffset:{} rowNumber:{}]",
      allocationIndex,
      rowOffset,
      rowNumber);
}

RowContainer::Layout::Layout(
    const std::vector<TypePtr>& keyTypes,
    const std::vector<Accumulator>& accumulators,
    const std::vector<TypePtr>& dependentTypes,
    const Options& options) {
  VELOX_CHECK(
      accumulators.empty() || dependentTypes.empty(),
      "A row container cannot have both accumulators and dependent columns");
  // The offsets of the keys, accumulators and dependent columns, and the bit
  // offsets of their null flags. A non-nullable key has an entry in
  // 'nullOffsets', which is not used.
  std::vector<int32_t> offsets;
  std::vector<int32_t> nullOffsets;
  int32_t offset = 0;
  int32_t flagOffset = 0;
  bool isVariableWidth = false;
  for (auto& type : keyTypes) {
    typeKinds.push_back(type->kind());
    types.push_back(type);
    offsets.push_back(offset);
    offset += typeKindSize(type->kind());
    nullOffsets.push_back(flagOffset);
    isVariableWidth |= !type->isFixedWidth();
    if (options.nullableKeys) {
      ++flagOffset;
    }
  }
  // Make offset at least sizeof pointer so that there is space for a
  // free list next pointer below the bit at 'freeFlagOffset'.
  offset = std::max<int32_t>(offset, sizeof(void*));
  flagsByteOffset = offset;
  if (!accumulators.empty()) {
    // This moves flagOffset to the start of the next byte.
    // This is to guarantee the null and initialized bits for an aggregate
    // always appear in the same byte.
    flagOffset = (flagOffset + 7) & -8;
  }
  for (const auto& accumulator : accumulators) {
    // Null bit.
    nullOffsets.push_back(flagOffset);
    // Increment for two bits: null bit and following initialized bit.
    flagOffset += kNumAccumulatorFlags;
    isVariableWidth |= !accumulator.isFixedSize();
    usesExternalMemory |= accumulator.usesExternalMemory();
    alignment = combineAlignments(accumulator.alignment(), alignment);
  }
  for (auto& type : dependentTypes) {
    types.push_back(type);
    typeKinds.push_back(type->kind());
    nullOffsets.push_back(flagOffset);
    ++flagOffset;
    isVariableWidth |= !type->isFixedWidth();
  }
  if (options.hasProbedFlag) {
    probedFlagOffset = flagOffset + flagsByteOffset * 8;
    ++flagOffset;
  }
  // Free flag.
  freeFlagOffset = flagOffset + flagsByteOffset * 8;
  ++flagOffset;
  // Add 1 to the last null offset to get the number of bits.
  flagBytes = bits::nbytes(flagOffset);
  // Fixup 'nullOffsets' to be the bit number from the start of the row.
  for (auto& nullOffset : nullOffsets) {
    nullOffset += flagsByteOffset * 8;
  }
  offset += flagBytes;
  for (const auto& accumulator : accumulators) {
    // Accumulator offset must be aligned by their alignment size.
    offset = bits::roundUp(offset, accumulator.alignment());
    offsets.push_back(offset);
    offset += accumulator.fixedWidthSize();
  }
  for (auto& type : dependentTypes) {
    offsets.push_back(offset);
    offset += typeKindSize(type->kind());
  }
  if (options.rawDependentSize > 0) {
    alignment = combineAlignments(alignof(int64_t), alignment);
    offset = bits::roundUp(offset, alignof(int64_t));
    rawDependentSize = options.rawDependentSize;
    rawDependentOffset = offset;
    offset += options.rawDependentSize;
  }
  if (isVariableWidth) {
    rowSizeOffset = offset;
    offset += sizeof(uint32_t);
  }
  if (options.hasNext) {
    nextOffset = offset;
    offset += sizeof(void*);
  }
  if (options.hasCountFlag) {
    countOffset = offset;
    offset += sizeof(int32_t);
  }
  fixedRowSize = bits::roundUp(offset, alignment);
  normalizedKeySize = options.hasNormalizedKey
      ? bits::roundUp(sizeof(normalized_key_t), alignment)
      : 0;
  for (auto i = 0; i < offsets.size(); ++i) {
    rowColumns.emplace_back(
        offsets[i],
        (options.nullableKeys || i >= keyTypes.size())
            ? nullOffsets[i]
            : RowColumn::kNotNullOffset);
  }
}

RowContainer::RowContainer(
    const std::vector<TypePtr>& keyTypes,
    const std::vector<Accumulator>& accumulators,
    const std::vector<TypePtr>& dependentTypes,
    memory::MemoryPool* pool,
    const Options& options)
    : keyTypes_(keyTypes),
      nullableKeys_(options.nullableKeys),
      useListRowIndex_(options.useListRowIndex),
      layout_(keyTypes, accumulators, dependentTypes, options),
      stringAllocator_(std::make_unique<HashStringAllocator>(pool)),
      accumulators_(accumulators),
      rowColumnsStats_(layout_.types.size()),
      normalizedKeySize_(layout_.normalizedKeySize),
      rows_(pool),
      rowPointers_(StlAllocator<char*>(stringAllocator_.get())) {}

RowContainer::~RowContainer() {
  clear();
}

char* RowContainer::newRow() {
  VELOX_DCHECK(mutable_, "Can't add row into an immutable row container");
  ++numRows_;
  char* row;
  if (firstFreeRow_) {
    row = firstFreeRow_;
    VELOX_CHECK(bits::isBitSet(row, layout_.freeFlagOffset));
    firstFreeRow_ = nextFree(row);
    --numFreeRows_;
  } else {
    row = rows_.allocateFixed(
              layout_.fixedRowSize + normalizedKeySize_, layout_.alignment) +
        normalizedKeySize_;
    if (normalizedKeySize_) {
      ++numRowsWithNormalizedKey_;
    }

    if (useListRowIndex_) {
      rowPointers_.push_back(row);
    }
  }
  return initializeRow(row, false /* reuse */);
}

void RowContainer::setAllNull(char* row) {
  VELOX_CHECK(!bits::isBitSet(row, layout_.freeFlagOffset));
  removeOrUpdateRowColumnStats(row, /*setToNull=*/true);
  for (const auto& column : layout_.rowColumns) {
    row[column.nullByte()] |= column.nullMask();
  }
}

char* RowContainer::initializeRow(char* row, bool reuse) {
  if (reuse) {
    VELOX_CHECK_EQ(layout_.nextOffset, 0);
    removeOrUpdateRowColumnStats(row, /*setToNull=*/false);
    const std::span<char*> rows(&row, 1);
    freeVariableWidthValues(rows);
    freeAggregates(rows);
  }
  if (layout_.rowSizeOffset != 0) {
    // Zeroes the string views so that a column not stored reads as an empty
    // value and freeing the row does not hit uninitialized data. The fastest
    // way is to set the whole row to 0.
    ::memset(row, 0, layout_.fixedRowSize);
  }
  // Clears the null, initialized, probed and free flags.
  ::memset(row + layout_.flagsByteOffset, 0, layout_.flagBytes);
  if (layout_.rowSizeOffset) {
    variableRowSize(row) = 0;
  }
  bits::clearBit(row, layout_.freeFlagOffset);
  if (layout_.nextOffset) {
    *reinterpret_cast<char**>(row + layout_.nextOffset) = nullptr;
  }
  if (layout_.countOffset) {
    countRef(row) = 1;
  }
  return row;
}

void RowContainer::removeOrUpdateRowColumnStats(
    const char* row,
    bool setToNull) {
  for (auto i = 0; i < layout_.types.size(); i++) {
    const auto column = columnAt(i);
    if (setToNull && column.nullMask() == 0) {
      // A non-nullable key keeps its value.
      continue;
    }
    auto& stats = rowColumnsStats_[i];
    if (isNullAt(row, column)) {
      stats.removeNull();
    } else if (!layout_.types[i]->isFixedWidth()) {
      stats.removeValue(variableSizeAt(row, i));
    }
    if (setToNull) {
      stats.addNull();
    }
  }
}

void RowContainer::eraseRows(std::span<char*> rows) {
  // Sets the free flags before freeing any memory, so that a row listed twice
  // throws instead of having its memory freed twice.
  for (auto* row : rows) {
    VELOX_CHECK(
        !bits::isBitSet(row, layout_.freeFlagOffset), "Double free of row");
    removeOrUpdateRowColumnStats(row, /*setToNull=*/false);
    bits::setBit(row, layout_.freeFlagOffset);
  }
  freeRowsExtraMemory(rows);
  for (auto* row : rows) {
    nextFree(row) = firstFreeRow_;
    firstFreeRow_ = row;
  }
  numFreeRows_ += rows.size();
}

int32_t RowContainer::filterOwnedRows(RowSpan rows, std::span<char*> result)
    const {
  VELOX_CHECK_GE(result.size(), rows.size());
  raw_vector<folly::Range<char*>> ranges(stringAllocator_->pool());
  ranges.resize(rows_.numRanges());
  for (auto i = 0; i < rows_.numRanges(); ++i) {
    ranges[i] = rows_.rangeAt(i);
  }
  std::sort(
      ranges.begin(), ranges.end(), [](const auto& left, const auto& right) {
        return left.data() < right.data();
      });
  raw_vector<uint64_t> starts(stringAllocator_->pool());
  raw_vector<uint64_t> sizes(stringAllocator_->pool());
  starts.reserve(ranges.size());
  sizes.reserve(ranges.size());
  for (const auto& range : ranges) {
    starts.push_back(reinterpret_cast<uintptr_t>(range.data()));
    sizes.push_back(range.size());
  }
  int32_t numRows = 0;
  for (auto* row : rows) {
    const auto address = reinterpret_cast<uintptr_t>(row);
    // The last range starting at or below 'address' is the only one that can
    // hold it.
    const auto it = std::upper_bound(starts.begin(), starts.end(), address);
    if (it == starts.begin()) {
      continue;
    }
    const auto index = it - starts.begin() - 1;
    if (address < starts[index] + sizes[index]) {
      result[numRows++] = row;
    }
  }
  return numRows;
}

template <typename ViewType>
void RowContainer::freeVariableWidthValues(
    column_index_t columnIndex,
    RowSpan rows) {
  static_assert(
      std::is_same_v<ViewType, StringView> ||
      std::is_same_v<ViewType, std::string_view>);

  const auto column = columnAt(columnIndex);
  for (auto row : rows) {
    if (isNullAt(row, column)) {
      continue;
    }

    auto& view = valueAt<ViewType>(row, column.offset());
    if constexpr (std::is_same_v<ViewType, StringView>) {
      if (view.isInline()) {
        continue;
      }
    } else {
      if (view.empty()) {
        continue;
      }
    }
    stringAllocator_->free(HashStringAllocator::headerOf(view.data()));
  }
}

void RowContainer::freeVariableWidthValues(RowSpan rows) {
  for (auto i = 0; i < layout_.types.size(); ++i) {
    switch (layout_.typeKinds[i]) {
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY: {
        freeVariableWidthValues<StringView>(i, rows);
        break;
      }
      case TypeKind::ROW:
      case TypeKind::ARRAY:
      case TypeKind::MAP: {
        freeVariableWidthValues<std::string_view>(i, rows);
        break;
      }
      default:;
    }
  }
}

void RowContainer::freeAggregates(std::span<char*> rows) {
  for (auto& accumulator : accumulators_) {
    accumulator.destroy(folly::Range<char**>(rows.data(), rows.size()));
  }
}

void RowContainer::freeRowsExtraMemory(std::span<char*> rows) {
  freeVariableWidthValues(rows);
  freeAggregates(rows);
  numRows_ -= rows.size();
}

void RowContainer::updateColumnStats(
    const char* row,
    column_index_t columnIndex) {
  auto& columnStats = rowColumnsStats_[columnIndex];
  if (isNullAt(row, layout_.rowColumns[columnIndex])) {
    columnStats.addNull();
  } else if (!layout_.types[columnIndex]->isFixedWidth()) {
    columnStats.addValue(variableSizeAt(row, columnIndex));
  }
}

void RowContainer::store(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    char* row,
    column_index_t columnIndex) {
  VELOX_DCHECK_LT(columnIndex, layout_.types.size());
  const bool isKey = columnIndex < keyTypes_.size();
  if (isKey && !nullableKeys_) {
    VELOX_DYNAMIC_TYPE_DISPATCH(
        storeNoNulls,
        layout_.typeKinds[columnIndex],
        decoded,
        rowIndex,
        isKey,
        row,
        layout_.rowColumns[columnIndex].offset());
  } else {
    auto rowColumn = layout_.rowColumns[columnIndex];
    VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
        storeWithNulls,
        layout_.typeKinds[columnIndex],
        decoded,
        rowIndex,
        isKey,
        row,
        rowColumn.offset(),
        rowColumn.nullByte(),
        rowColumn.nullMask(),
        columnIndex);
  }
  updateColumnStats(row, columnIndex);
}

void RowContainer::store(
    const DecodedVector& decoded,
    RowSpan rows,
    column_index_t columnIndex) {
  VELOX_DCHECK_LT(columnIndex, layout_.types.size());
  VELOX_CHECK_GE(decoded.size(), rows.size());
  const bool isKey = columnIndex < keyTypes_.size();
  if ((isKey && !nullableKeys_) || !decoded.mayHaveNulls()) {
    VELOX_DYNAMIC_TYPE_DISPATCH(
        storeNoNullsBatch,
        layout_.typeKinds[columnIndex],
        decoded,
        rows,
        isKey,
        layout_.rowColumns[columnIndex].offset(),
        columnIndex);
  } else {
    const auto& rowColumn = layout_.rowColumns[columnIndex];
    VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
        storeWithNullsBatch,
        layout_.typeKinds[columnIndex],
        decoded,
        rows,
        isKey,
        rowColumn.offset(),
        rowColumn.nullByte(),
        rowColumn.nullMask(),
        columnIndex);
  }
}

HashStringAllocator::InputStream RowContainer::prepareRead(
    const char* row,
    int32_t offset) {
  const auto& view = reinterpret_cast<const std::string_view*>(row + offset);
  // We set 'stream' to range over the ranges that start at the Header
  // immediately below the first character in the std::string_view.
  return HashStringAllocator::InputStream(
      HashStringAllocator::headerOf(view->data()));
}

int32_t RowContainer::variableSizeAt(
    const char* row,
    column_index_t columnIndex) const {
  const auto& rowColumn = layout_.rowColumns[columnIndex];

  if (isNullAt(row, rowColumn)) {
    return 0;
  }

  const auto typeKind = layout_.typeKinds[columnIndex];
  if (is_string_kind(typeKind)) {
    return reinterpret_cast<const StringView*>(row + rowColumn.offset())
        ->size();
  } else {
    return reinterpret_cast<const std::string_view*>(row + rowColumn.offset())
        ->size();
  }
}

int32_t RowContainer::fixedSizeAt(column_index_t columnIndex) const {
  return typeKindSize(layout_.typeKinds[columnIndex]);
}

int32_t RowContainer::extractVariableSizeAt(
    const char* row,
    column_index_t columnIndex,
    char* output) const {
  const auto rowColumn = layout_.rowColumns[columnIndex];

  // 4 bytes for size + N bytes for data.
  if (isNullAt(row, rowColumn)) {
    ::memset(output, 0, 4);
    return 4;
  }

  const auto typeKind = layout_.typeKinds[columnIndex];
  if (is_string_kind(typeKind)) {
    const auto value = valueAt<StringView>(row, rowColumn.offset());
    const auto size = value.size();
    ::memcpy(output, &size, 4);

    if (value.isInline() ||
        reinterpret_cast<const HashStringAllocator::Header*>(value.data())[-1]
                .size() >= value.size()) {
      ::memcpy(output + 4, value.data(), size);
    } else {
      HashStringAllocator::InputStream stream(
          HashStringAllocator::headerOf(value.data()));
      stream.ByteInputStream::readBytes(output + 4, size);
    }
    return 4 + size;
  }

  const auto value = valueAt<std::string_view>(row, rowColumn.offset());
  const auto size = value.size();

  auto stream = prepareRead(row, rowColumn.offset());

  ::memcpy(output, &size, 4);
  stream.ByteInputStream::readBytes(output + 4, size);

  return 4 + size;
}

int32_t RowContainer::storeVariableSizeAt(
    const char* data,
    char* row,
    column_index_t columnIndex) {
  const auto typeKind = layout_.typeKinds[columnIndex];
  const auto rowColumn = layout_.rowColumns[columnIndex];

  // First 4 bytes is the size of the data.
  const auto size = *reinterpret_cast<const int32_t*>(data);

  if (is_string_kind(typeKind)) {
    if (size > 0) {
      stringAllocator_->copyMultipart(
          StringView(data + 4, size), row, rowColumn.offset());
    } else {
      valueAt<StringView>(row, rowColumn.offset()) = StringView();
    }
  } else {
    if (size > 0) {
      ByteOutputStream stream(stringAllocator_.get(), false, false);
      const auto position = stringAllocator_->newWrite(stream);
      stream.appendStringView(std::string_view(data + 4, size));
      stringAllocator_->finishWrite(stream, 0);
      valueAt<std::string_view>(row, rowColumn.offset()) =
          std::string_view(reinterpret_cast<char*>(position.position), size);
    } else {
      valueAt<std::string_view>(row, rowColumn.offset()) = std::string_view();
    }
  }

  return 4 + size;
}

void RowContainer::checkSerializable() const {
  VELOX_CHECK(
      accumulators_.empty(), "Rows with accumulators cannot be serialized");
  VELOX_CHECK_EQ(
      layout_.rawDependentSize,
      0,
      "Rows with a raw dependent cannot be serialized");
}

void RowContainer::extractSerializedRows(RowSpan rows, const VectorPtr& result)
    const {
  checkSerializable();

  // The format of the extracted row is: null bytes followed by keys and
  // dependent columns. Fixed-width columns are serialized into fixed number of
  // bytes (see typeKindSize). Variable-width columns are serialized as 4 bytes
  // of size followed by that many bytes.

  // First, calculate total number of bytes needed to serialize all rows.

  size_t fixedWidthRowSize = 0;
  bool hasVariableWidth = false;
  for (auto i = 0; i < layout_.types.size(); ++i) {
    const auto& type = layout_.types[i];
    if (type->isFixedWidth()) {
      fixedWidthRowSize += typeKindSize(type->kind());
    } else {
      hasVariableWidth = true;
    }
  }

  size_t totalBytes =
      layout_.flagBytes * rows.size() + fixedWidthRowSize * rows.size();
  if (hasVariableWidth) {
    for (const char* row : rows) {
      for (auto i = 0; i < layout_.types.size(); ++i) {
        const auto& type = layout_.types[i];
        if (!type->isFixedWidth()) {
          // 4 bytes for size + N bytes for data.
          totalBytes += 4 + variableSizeAt(row, i);
        }
      }
    }
  }

  // Allocate sufficient buffer.
  auto* flatResult = result->as<FlatVector<StringView>>();
  flatResult->resize(rows.size());
  auto* rawBuffer = flatResult->getRawStringBufferWithSpace(totalBytes, true);

  // Write serialized data.
  size_t totalWritten = 0;
  for (auto i = 0; i < rows.size(); ++i) {
    auto* row = rows[i];
    size_t offset = 0;

    // Copy nulls and other flags.
    ::memcpy(
        rawBuffer + offset, row + layout_.flagsByteOffset, layout_.flagBytes);
    offset += layout_.flagBytes;

    // Copy values.
    for (auto j = 0; j < layout_.types.size(); ++j) {
      const auto& type = layout_.types[j];
      if (type->isFixedWidth()) {
        const auto size = typeKindSize(type->kind());
        ::memcpy(
            rawBuffer + offset, row + layout_.rowColumns[j].offset(), size);
        offset += size;
      } else {
        auto size = extractVariableSizeAt(row, j, rawBuffer + offset);
        offset += size;
      }
    }

    flatResult->setNoCopy(i, StringView(rawBuffer, offset));
    rawBuffer += offset;
    totalWritten += offset;
  }

  VELOX_CHECK_EQ(totalWritten, totalBytes);
}

void RowContainer::storeSerializedRow(
    const FlatVector<StringView>& vector,
    vector_size_t index,
    char* row) {
  checkSerializable();
  VELOX_CHECK(!vector.isNullAt(index));
  const auto serialized = vector.valueAt(index);
  size_t offset = 0;

  ::memcpy(row + layout_.flagsByteOffset, serialized.data(), layout_.flagBytes);
  offset += layout_.flagBytes;

  RowSizeTracker tracker(row[layout_.rowSizeOffset], *stringAllocator_);
  for (auto i = 0; i < layout_.types.size(); ++i) {
    const auto& type = layout_.types[i];
    if (type->isFixedWidth()) {
      const auto size = typeKindSize(type->kind());
      ::memcpy(
          row + layout_.rowColumns[i].offset(),
          serialized.data() + offset,
          size);
      offset += size;
    } else {
      const auto size = storeVariableSizeAt(serialized.data() + offset, row, i);
      offset += size;
    }
    updateColumnStats(row, i);
  }
}

void RowContainer::extractString(
    StringView value,
    FlatVector<StringView>* values,
    vector_size_t index) {
  if (value.isInline() ||
      reinterpret_cast<const HashStringAllocator::Header*>(value.data())[-1]
              .size() >= value.size()) {
    // The string is inline or all in one piece out of line.
    values->set(index, value);
    return;
  }
  auto rawBuffer = values->getRawStringBufferWithSpace(value.size());
  HashStringAllocator::InputStream stream(
      HashStringAllocator::headerOf(value.data()));
  stream.ByteInputStream::readBytes(rawBuffer, value.size());
  values->setNoCopy(index, StringView(rawBuffer, value.size()));
}

void RowContainer::storeComplexType(
    const DecodedVector& decoded,
    vector_size_t index,
    bool isKey,
    char* row,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask) {
  if (decoded.isNullAt(index)) {
    VELOX_DCHECK(nullMask);
    row[nullByte] |= nullMask;
    return;
  }
  RowSizeTracker tracker(row[layout_.rowSizeOffset], *stringAllocator_);
  ByteOutputStream stream(stringAllocator_.get(), false, false);
  auto position = stringAllocator_->newWrite(stream);
  ContainerRowSerdeOptions options{.isKey = isKey};
  ContainerRowSerde::serialize(
      *decoded.base(), decoded.index(index), stream, options);
  stringAllocator_->finishWrite(stream, 0);

  valueAt<std::string_view>(row, offset) = std::string_view(
      reinterpret_cast<char*>(position.position), stream.size());
}

// static
int32_t RowContainer::compareStringAsc(
    StringView left,
    const DecodedVector& decoded,
    vector_size_t index) {
  auto right = decoded.valueAt<StringView>(index);
  // Decides from the prefix without contiguousString(), which reads the
  // allocator header of an out-of-line 'left'.
  if (auto result = left.comparePrefix(right)) {
    return *result;
  }
  std::string storage;
  return HashStringAllocator::contiguousString(left, storage).compare(right);
}

// static
int32_t RowContainer::compareStringAsc(StringView left, StringView right) {
  if (auto result = left.comparePrefix(right)) {
    return *result;
  }
  std::string leftStorage;
  std::string rightStorage;
  return HashStringAllocator::contiguousString(left, leftStorage)
      .compare(HashStringAllocator::contiguousString(right, rightStorage));
}

int RowContainer::compareComplexType(
    const char* row,
    int32_t offset,
    const DecodedVector& decoded,
    vector_size_t index,
    CompareFlags flags) const {
  VELOX_DCHECK(flags.nullAsValue(), "not supported null handling mode");

  auto stream = prepareRead(row, offset);
  return ContainerRowSerde::compare(stream, decoded, index, flags);
}

int32_t RowContainer::compareComplexType(
    const char* left,
    const char* right,
    const Type* type,
    int32_t leftOffset,
    int32_t rightOffset,
    CompareFlags flags) const {
  VELOX_DCHECK(flags.nullAsValue(), "not supported null handling mode");

  auto leftStream = prepareRead(left, leftOffset);
  auto rightStream = prepareRead(right, rightOffset);
  return ContainerRowSerde::compare(leftStream, rightStream, type, flags);
}

int32_t RowContainer::compareComplexType(
    const char* left,
    const char* right,
    const Type* type,
    int32_t offset,
    CompareFlags flags) const {
  return compareComplexType(left, right, type, offset, offset, flags);
}

template <bool typeProvidesCustomComparison, TypeKind Kind>
void RowContainer::hashTyped(
    const Type* type,
    RowColumn column,
    bool nullable,
    RowSpan rows,
    bool mix,
    uint64_t* result) const {
  using T = typename KindToFlatVector<Kind>::HashRowType;

  auto offset = column.offset();
  std::string storage;
  auto numRows = rows.size();

  for (int32_t i = 0; i < numRows; ++i) {
    char* row = rows[i];
    if (nullable && isNullAt(row, column)) {
      result[i] = mix ? bits::hashMix(result[i], BaseVector::kNullHash)
                      : BaseVector::kNullHash;
    } else {
      uint64_t hash;
      if constexpr (is_string_kind(Kind)) {
        hash =
            folly::hasher<StringView>()(HashStringAllocator::contiguousString(
                valueAt<StringView>(row, offset), storage));
      } else if constexpr (is_nested_kind(Kind)) {
        auto in = prepareRead(row, offset);
        hash = ContainerRowSerde::hash(in, type);
      } else if constexpr (typeProvidesCustomComparison) {
        hash = static_cast<const CanProvideCustomComparisonType<Kind>*>(type)
                   ->hash(valueAt<T>(row, offset));
      } else if constexpr (std::is_floating_point_v<T>) {
        hash = util::floating_point::NaNAwareHash<T>()(valueAt<T>(row, offset));
      } else {
        hash = folly::hasher<T>()(valueAt<T>(row, offset));
      }
      result[i] = mix ? bits::hashMix(result[i], hash) : hash;
    }
  }
}

void RowContainer::hash(
    column_index_t columnIndex,
    RowSpan rows,
    bool mix,
    uint64_t* result) const {
  if (layout_.typeKinds[columnIndex] == TypeKind::UNKNOWN) {
    for (auto i = 0; i < rows.size(); ++i) {
      result[i] = mix ? bits::hashMix(result[i], BaseVector::kNullHash)
                      : BaseVector::kNullHash;
    }
    return;
  }

  bool nullable = columnIndex >= keyTypes_.size() || nullableKeys_;

  const auto& type = layout_.types[columnIndex];

  if (type->providesCustomComparison()) {
    VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
        hashTyped,
        true,
        layout_.typeKinds[columnIndex],
        type.get(),
        columnAt(columnIndex),
        nullable,
        rows,
        mix,
        result);
  } else {
    VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
        hashTyped,
        false,
        layout_.typeKinds[columnIndex],
        type.get(),
        columnAt(columnIndex),
        nullable,
        rows,
        mix,
        result);
  }
}

void RowContainer::hashKeys(RowSpan rows, std::span<uint64_t> result) const {
  VELOX_CHECK_GE(result.size(), rows.size());
  for (column_index_t i = 0; i < keyTypes_.size(); ++i) {
    hash(i, rows, /*mix=*/i > 0, result.data());
  }
}

void RowContainer::clear() {
  if (layout_.usesExternalMemory) {
    constexpr int32_t kBatch = 1000;
    std::vector<char*> rows(kBatch);
    RowContainerIterator iter;
    while (auto numRows = listRows(&iter, std::span(rows))) {
      freeRowsExtraMemory({rows.data(), static_cast<size_t>(numRows)});
    }
  }

  rows_.clear();
  rowPointers_.clear();
  rowPointers_.shrink_to_fit();
  stringAllocator_->clear();
  numRows_ = 0;
  numRowsWithNormalizedKey_ = 0;
  normalizedKeySize_ = layout_.normalizedKeySize;
  numFreeRows_ = 0;
  firstFreeRow_ = nullptr;

  rowColumnsStats_.assign(layout_.types.size(), RowColumnStats{});
}

void RowContainer::setProbedFlag(RowSpan rows) {
  const int32_t probedFlagByte = layout_.probedFlagOffset / 8;
  const uint8_t probedFlagMask = uint8_t{1} << (layout_.probedFlagOffset & 7);
  for (char* row : rows) {
    // Row may be null in case of a FULL join.
    if (row) {
      uint8_t* flagsByte = reinterpret_cast<uint8_t*>(row) + probedFlagByte;
      if ((*flagsByte & probedFlagMask) == 0) {
        *flagsByte |= probedFlagMask;
      }
    }
  }
}

bool RowContainer::testAndSetProbedFlag(char* row) {
  VELOX_DCHECK_NOT_NULL(row);
  const int32_t probedFlagByte = layout_.probedFlagOffset / 8;
  const uint8_t probedFlagMask = uint8_t{1} << (layout_.probedFlagOffset & 7);
  uint8_t* flagsByte = reinterpret_cast<uint8_t*>(row) + probedFlagByte;
  const bool wasSet = (*flagsByte & probedFlagMask) != 0;
  if (!wasSet) {
    *flagsByte |= probedFlagMask;
  }
  return !wasSet;
}

void RowContainer::extractProbedFlags(
    RowSpan rows,
    bool setNullForNullKeysRow,
    bool setNullForNonProbedRow,
    const VectorPtr& result) const {
  auto* flatResult = result->asChecked<FlatVector<bool>>();
  flatResult->resize(rows.size());
  flatResult->clearAllNulls();

  auto* rawValues = flatResult->mutableRawValues<uint64_t>();
  for (auto i = 0; i < rows.size(); ++i) {
    // Check if this row has null keys.
    bool nullResult = false;
    if (setNullForNullKeysRow && nullableKeys_) {
      for (auto c = 0; c < keyTypes_.size(); ++c) {
        if (isNullAt(rows[i], columnAt(c))) {
          nullResult = true;
          break;
        }
      }
    }

    if (nullResult) {
      flatResult->setNull(i, true);
    } else {
      const bool probed = bits::isBitSet(rows[i], layout_.probedFlagOffset);
      if (setNullForNonProbedRow && !probed) {
        flatResult->setNull(i, true);
      } else {
        bits::setBit(rawValues, i, probed);
      }
    }
  }
}

std::optional<int64_t> RowContainer::estimateRowSize() const {
  if (numRows_ == 0) {
    return std::nullopt;
  }
  int64_t freeBytes = rows_.freeBytes() + layout_.fixedRowSize * numFreeRows_;
  int64_t usedSize = rows_.allocatedBytes() - freeBytes +
      usedVariableWidthBytes() - rowPointers_.capacity() * sizeof(char*);
  int64_t rowSize = usedSize / numRows_;
  VELOX_CHECK_GT(
      rowSize, 0, "Estimated row size of the RowContainer must be positive.");
  return rowSize;
}

int64_t RowContainer::maxGrowthBytes(
    vector_size_t numRows,
    int64_t variableWidthBytes) const {
  // Small containers can grow in smaller units but for spilling the practical
  // minimum increment is a huge page.
  constexpr int32_t kAllocUnit = memory::AllocationTraits::kHugePageSize;
  // A row taken from the allocation runs also takes its normalized key.
  const int64_t newRowSize = layout_.fixedRowSize + normalizedKeySize_;
  const int64_t numFreeRows = numFreeRows_ + rows_.freeBytes() / newRowSize;
  const int64_t needRows = std::max<int64_t>(0, numRows - numFreeRows);
  int64_t needBytes =
      std::max<int64_t>(0, variableWidthBytes - stringAllocator_->freeSpace());
  return bits::roundUp(needRows * newRowSize, kAllocUnit) +
      bits::roundUp(needBytes, kAllocUnit);
}

void RowContainer::checkConsistency() const {
  constexpr int32_t kBatch = 1000;
  std::vector<char*> rows(kBatch);
  RowContainerIterator iter;
  int64_t numAllocatedRows = 0;
  std::vector<RowColumnStats> expectedStats(layout_.types.size());
  // Scans the allocations, which skips erased rows, unlike the row index.
  while (auto numRows =
             listRows<ProbedFilter::kAll>(&iter, std::span(rows), kUnlimited)) {
    for (auto i = 0; i < numRows; ++i) {
      char* row = rows[i];
      VELOX_CHECK(!bits::isBitSet(row, layout_.freeFlagOffset));
      ++numAllocatedRows;
      for (auto column = 0; column < rowColumnsStats_.size(); ++column) {
        if (isNullAt(row, columnAt(column))) {
          expectedStats[column].addNull();
        } else if (!layout_.types[column]->isFixedWidth()) {
          expectedStats[column].addValue(variableSizeAt(row, column));
        }
      }
    }
  }

  uint64_t numFree = 0;
  for (auto* free = firstFreeRow_; free != nullptr; free = nextFree(free)) {
    ++numFree;
    VELOX_CHECK(bits::isBitSet(free, layout_.freeFlagOffset));
  }
  VELOX_CHECK_EQ(numFree, numFreeRows_);
  VELOX_CHECK_EQ(numAllocatedRows, numRows_);

  VELOX_CHECK_EQ(layout_.types.size(), rowColumnsStats_.size());
  for (auto column = 0; column < rowColumnsStats_.size(); ++column) {
    const auto& stored = rowColumnsStats_[column];
    const auto& expected = expectedStats[column];
    VELOX_CHECK_EQ(expected.nullCount(), stored.nullCount());
    VELOX_CHECK_EQ(expected.sumBytes(), stored.sumBytes());
    VELOX_CHECK_GE(stored.maxBytes(), expected.maxBytes());
  }
}

void RowContainer::skip(RowContainerIterator& iter, int32_t numRows) const {
  VELOX_DCHECK(accumulators_.empty(), "Used in join only");
  VELOX_DCHECK_LE(0, numRows);
  if (!iter.endOfRun) {
    // Set to first row.
    VELOX_DCHECK_EQ(0, iter.rowNumber);
    VELOX_DCHECK_EQ(0, iter.allocationIndex);
    iter.normalizedKeysLeft = numRowsWithNormalizedKey_;
    iter.normalizedKeySize = layout_.normalizedKeySize;
    auto range = rows_.rangeAt(0);
    iter.rowBegin = range.data();
    iter.endOfRun = iter.rowBegin + range.size();
  }
  if (iter.rowNumber + numRows >= numRows_) {
    iter.rowNumber = numRows_;
    iter.rowBegin = nullptr;
    return;
  }
  int32_t rowSize = layout_.fixedRowSize +
      (iter.normalizedKeysLeft > 0 ? layout_.normalizedKeySize : 0);
  auto toSkip = numRows;
  if (iter.normalizedKeysLeft && iter.normalizedKeysLeft < numRows) {
    toSkip -= iter.normalizedKeysLeft;
    skip(iter, iter.normalizedKeysLeft);
    rowSize = layout_.fixedRowSize;
  }
  while (toSkip) {
    if (iter.rowBegin &&
        toSkip * rowSize <= (iter.endOfRun - iter.rowBegin) - rowSize) {
      iter.rowBegin += toSkip * rowSize;
      break;
    }
    int32_t rowsInRun = (iter.endOfRun - iter.rowBegin) / rowSize;
    toSkip -= rowsInRun;
    ++iter.allocationIndex;
    auto range = rows_.rangeAt(iter.allocationIndex);
    iter.endOfRun = range.data() + range.size();
    iter.rowBegin = range.data();
  }
  if (iter.normalizedKeysLeft) {
    iter.normalizedKeysLeft -= numRows;
  }
  iter.rowNumber += numRows;
}

std::unique_ptr<RowPartitions> RowContainer::createRowPartitions(
    memory::MemoryPool& pool) {
  VELOX_CHECK(
      mutable_, "Can only create RowPartitions once from a row container");
  mutable_ = false;
  return std::make_unique<RowPartitions>(numRows_, pool);
}

int32_t RowContainer::listPartitionRows(
    RowContainerIterator& iter,
    uint8_t partition,
    const RowPartitions& rowPartitions,
    std::span<char*> rows) const {
  const auto maxRows = static_cast<int32_t>(rows.size());
  VELOX_CHECK(
      !mutable_, "Can't list partition rows from a mutable row container");
  VELOX_CHECK_EQ(
      rowPartitions.size(), numRows_, "All rows must have a partition");
  if (numRows_ == 0) {
    return 0;
  }
  const auto partitionNumberVector =
      xsimd::batch<uint8_t>::broadcast(partition);
  const auto& allocation = rowPartitions.allocation();
  int32_t numResults = 0;
  while (numResults < maxRows && iter.rowNumber < numRows_) {
    constexpr int32_t kBatch = xsimd::batch<uint8_t>::size;
    // Start at multiple of kBatch.
    auto startRow = iter.rowNumber / kBatch * kBatch;
    // Ignore the possible hits at or below iter.rowNumber.
    uint32_t firstBatchMask = ~bits::lowMask(iter.rowNumber - startRow);
    int32_t runIndex;
    int32_t offsetInRun;
    VELOX_CHECK_LT(startRow, numRows_);
    allocation.findRun(startRow, &runIndex, &offsetInRun);
    auto run = allocation.runAt(runIndex);
    auto runEnd = run.numBytes();
    auto runBytes = run.data<uint8_t>();
    for (; offsetInRun < runEnd; offsetInRun += kBatch) {
      auto bits =
          simd::toBitMask(
              partitionNumberVector ==
              xsimd::batch<uint8_t>::load_unaligned(runBytes + offsetInRun)) &
          firstBatchMask;
      firstBatchMask = ~0;
      bool atEnd = false;
      if (startRow + kBatch >= numRows_) {
        // Clear bits that are for rows past numRows_ - 1.
        bits &= bits::lowMask(numRows_ - startRow);
        atEnd = true;
      }
      while (bits) {
        const int32_t hit = __builtin_ctz(bits);
        const auto distance = hit + startRow - iter.rowNumber;
        skip(iter, distance);
        rows[numResults++] = iter.currentRow();
        if (numResults == maxRows) {
          skip(iter, 1);
          return numResults;
        }
        // Clear last set bit in 'bits'.
        bits &= bits - 1;
      }
      startRow += kBatch;
      // The last batch of 32 bytes may have been partly filled. If so, we
      // could have skipped past end.
      if (atEnd) {
        iter.rowNumber = numRows_;
        return numResults;
      }

      if (iter.rowNumber != startRow) {
        skip(iter, startRow - iter.rowNumber);
      }
    }
  }
  return numResults;
}

std::string RowContainer::toString() const {
  std::stringstream out;
  out << "Keys: ";
  for (auto i = 0; i < keyTypes_.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << keyTypes_[i]->toString();
  }

  if (layout_.types.size() > keyTypes_.size()) {
    out << " Dependents: ";
    for (auto i = keyTypes_.size(); i < layout_.types.size(); ++i) {
      if (i > keyTypes_.size()) {
        out << ", ";
      }
      out << layout_.types[i]->toString();
    }
  }

  if (!accumulators_.empty()) {
    out << " Num accumulators: " << accumulators_.size();
  }

  out << " Num rows: " << numRows_;
  return out.str();
}

std::string RowContainer::toString(const char* row) const {
  auto types = layout_.types;
  auto rowType = ROW(std::move(types));
  auto vector =
      BaseVector::create<RowVector>(rowType, 1, stringAllocator_->pool());

  for (auto i = 0; i < rowType->size(); ++i) {
    extractColumn(&row, 1, i, vector->childAt(i));
  }

  return vector->toString(0);
}

RowPartitions::RowPartitions(int32_t numRows, memory::MemoryPool& pool)
    : capacity_(numRows) {
  const auto numPages = memory::AllocationTraits::numPages(capacity_);
  if (numPages > 0) {
    pool.allocateNonContiguous(numPages, allocation_);
  }
}

void RowPartitions::appendPartitions(std::span<const uint8_t> partitions) {
  int32_t toAdd = partitions.size();
  int index = 0;
  VELOX_CHECK_LE(size_ + toAdd, capacity_);
  while (toAdd) {
    int32_t run;
    int32_t offset;
    allocation_.findRun(size_, &run, &offset);
    auto runSize = allocation_.runAt(run).numBytes();
    auto copySize = std::min<int32_t>(toAdd, runSize - offset);
    ::memcpy(
        allocation_.runAt(run).data<uint8_t>() + offset,
        &partitions[index],
        copySize);
    size_ += copySize;
    index += copySize;
    toAdd -= copySize;
    // Zero out to the next multiple of SIMD width for asan/valgrind.
    if (!toAdd) {
      bits::padToAlignment(
          allocation_.runAt(run).data<uint8_t>(),
          runSize,
          offset + copySize,
          xsimd::batch<uint8_t>::size);
    }
  }
}

RowComparator::RowComparator(
    const RowTypePtr& rowType,
    const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys,
    const std::vector<core::SortOrder>& sortingOrders,
    RowContainer* rowContainer)
    : rowContainer_(rowContainer) {
  const auto numKeys = sortingKeys.size();
  for (auto i = 0; i < numKeys; ++i) {
    const auto channel = exprToChannel(sortingKeys[i].get(), rowType);
    VELOX_USER_CHECK_NE(
        channel,
        kConstantChannel,
        "RowComparator doesn't allow constant comparison keys");
    keyInfo_.push_back(std::make_pair(channel, sortingOrders[i]));
  }
}

int32_t RowComparator::compare(const char* lhs, const char* rhs) {
  if (lhs == rhs) {
    return 0;
  }
  for (auto& key : keyInfo_) {
    if (auto result = rowContainer_->compare(
            lhs,
            rhs,
            key.first,
            {key.second.isNullsFirst(), key.second.isAscending(), false})) {
      return result;
    }
  }
  return 0;
}

bool RowComparator::operator()(const char* lhs, const char* rhs) {
  return compare(lhs, rhs) < 0;
}

int32_t RowComparator::compare(
    const std::vector<DecodedVector>& decodedVectors,
    vector_size_t index,
    const char* other) {
  for (auto& key : keyInfo_) {
    if (auto result = rowContainer_->compare(
            other,
            rowContainer_->columnAt(key.first),
            decodedVectors[key.first],
            index,
            {key.second.isNullsFirst(), key.second.isAscending(), false})) {
      return -result;
    }
  }
  return 0;
}

bool RowComparator::operator()(
    const std::vector<DecodedVector>& decodedVectors,
    vector_size_t index,
    const char* other) {
  return compare(decodedVectors, index, other) < 0;
}
} // namespace facebook::velox::exec
