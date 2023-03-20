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

#include "velox/exec/ContainerRowSerde.h"

namespace facebook::velox::exec {
namespace {
template <TypeKind Kind>
static int32_t kindSize() {
  return sizeof(typename KindToFlatVector<Kind>::HashRowType);
}

static int32_t typeKindSize(TypeKind kind) {
  if (kind == TypeKind::UNKNOWN) {
    return sizeof(UnknownValue);
  }

  return VELOX_DYNAMIC_TYPE_DISPATCH(kindSize, kind);
}

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
__attribute__((__no_sanitize__("thread")))
#endif
#endif
inline void
setBit(char* bits, uint32_t idx) {
  auto bitsAs8Bit = reinterpret_cast<uint8_t*>(bits);
  bitsAs8Bit[idx / 8] |= (1 << (idx % 8));
}
} // namespace

RowContainer::RowContainer(
    const std::vector<TypePtr>& keyTypes,
    bool nullableKeys,
    const std::vector<std::unique_ptr<Aggregate>>& aggregates,
    const std::vector<TypePtr>& dependentTypes,
    bool hasNext,
    bool isJoinBuild,
    bool hasProbedFlag,
    bool hasNormalizedKeys,
    memory::MemoryPool* pool,
    const RowSerde& serde)
    : keyTypes_(keyTypes),
      nullableKeys_(nullableKeys),
      aggregates_(aggregates),
      isJoinBuild_(isJoinBuild),
      hasNormalizedKeys_(hasNormalizedKeys),
      rows_(pool),
      stringAllocator_(pool),
      serde_(serde) {
  // Compute the layout of the payload row.  The row has keys, null
  // flags, accumulators, dependent fields. All fields are fixed
  // width. If variable width data is referenced, this is done with
  // StringView that inlines or points to the data.  The number of
  // bytes used by each key is determined by keyTypes[i].  Null flags
  // are one bit per field. If nullableKeys is true there is a null
  // flag for each key. A null bit for each accumulator and dependent
  // field follows.  If hasProbedFlag is true, there is an extra bit
  // to track if the row has been selected by a hash join probe. This
  // is followed by a free bit which is set if the row is in a free
  // list. The accumulators come next, with size given by
  // Aggregate::accumulatorFixedWidthSize(). Dependent fields follow.
  // These are non-key columns for hash join or order by. If there are variable
  // length columns or accumulators, i.e. ones that allocate extra space, this
  // space is tracked by a uint32_t after the dependent columns. If this is a
  // hash join build side, the pointer to the next row with the same key is
  // after the optional row size.
  //
  // In most cases, rows are prefixed with a normalized_key_t at index
  // -1, 8 bytes below the pointer. This space is reserved for a 64
  // bit unique digest of the keys for speeding up comparison. This
  // space is reserved for the rows that are inserted before the
  // cardinality grows too large for packing all in 64
  // bits. 'numRowsWithNormalizedKey_' gives the number of rows with
  // the extra field.
  int32_t offset = 0;
  int32_t nullOffset = 0;
  bool isVariableWidth = false;
  for (auto& type : keyTypes_) {
    typeKinds_.push_back(type->kind());
    types_.push_back(type);
    offsets_.push_back(offset);
    offset += typeKindSize(type->kind());
    nullOffsets_.push_back(nullOffset);
    isVariableWidth |= !type->isFixedWidth();
    if (nullableKeys) {
      ++nullOffset;
    }
  }
  // Make offset at least sizeof pointer so that there is space for a
  // free list next pointer below the bit at 'freeFlagOffset_'.
  offset = std::max<int32_t>(offset, sizeof(void*));
  int32_t firstAggregate = offsets_.size();
  int32_t firstAggregateOffset = offset;
  for (auto& aggregate : aggregates) {
    nullOffsets_.push_back(nullOffset);
    ++nullOffset;
    isVariableWidth |= !aggregate->isFixedSize();
    usesExternalMemory_ |= aggregate->accumulatorUsesExternalMemory();
    alignment_ = aggregate->combineAlignment(alignment_);
    aggregate->setAllocator(&stringAllocator_);
  }
  for (auto& type : dependentTypes) {
    types_.push_back(type);
    typeKinds_.push_back(type->kind());
    nullOffsets_.push_back(nullOffset);
    ++nullOffset;
    isVariableWidth |= !type->isFixedWidth();
  }
  if (hasProbedFlag) {
    nullOffsets_.push_back(nullOffset);
    probedFlagOffset_ = nullOffset + firstAggregateOffset * 8;
    ++nullOffset;
  }
  // Free flag.
  nullOffsets_.push_back(nullOffset);
  freeFlagOffset_ = nullOffset + firstAggregateOffset * 8;
  ++nullOffset;
  // Fixup nullOffsets_ to be the bit number from the start of the row.
  for (int32_t i = 0; i < nullOffsets_.size(); ++i) {
    nullOffsets_[i] += firstAggregateOffset * 8;
  }
  int32_t nullBytes = bits::nbytes(nullOffsets_.size());
  offset += nullBytes;
  for (auto& aggregate : aggregates) {
    // Accumulator offset must be aligned by their alignment size.
    offset = bits::roundUp(offset, aggregate->accumulatorAlignmentSize());
    offsets_.push_back(offset);
    offset += aggregate->accumulatorFixedWidthSize();
  }
  for (auto& type : dependentTypes) {
    offsets_.push_back(offset);
    offset += typeKindSize(type->kind());
  }
  if (isVariableWidth) {
    rowSizeOffset_ = offset;
    offset += sizeof(uint32_t);
  }
  if (hasNext) {
    nextOffset_ = offset;
    offset += sizeof(void*);
  }
  fixedRowSize_ = bits::roundUp(offset, alignment_);
  for (int i = 0; i < aggregates_.size(); ++i) {
    nullOffset = nullOffsets_[i + firstAggregate];
    aggregates_[i]->setOffsets(
        offsets_[i + firstAggregate],
        nullByte(nullOffset),
        nullMask(nullOffset),
        rowSizeOffset_);
  }
  // A distinct hash table has no aggregates and if the hash table has
  // no nulls, it may be that there are no null flags.
  if (!nullOffsets_.empty()) {
    // All flags like free and probed flags and null flags for keys and non-keys
    // start as 0.
    initialNulls_.resize(nullBytes, 0x0);
    // Aggregates are null on a new row.
    auto aggregateNullOffset = nullableKeys ? keyTypes.size() : 0;
    for (int32_t i = 0; i < aggregates_.size(); ++i) {
      bits::setBit(initialNulls_.data(), i + aggregateNullOffset);
    }
  }
  originalNormalizedKeySize_ = hasNormalizedKeys_
      ? bits::roundUp(sizeof(normalized_key_t), alignment_)
      : 0;
  normalizedKeySize_ = originalNormalizedKeySize_;
  for (auto i = 0; i < offsets_.size(); ++i) {
    rowColumns_.emplace_back(
        offsets_[i],
        (nullableKeys_ || i >= keyTypes_.size()) ? nullOffsets_[i]
                                                 : RowColumn::kNotNullOffset);
  }
}

char* RowContainer::newRow() {
  char* row;
  VELOX_DCHECK(
      !partitions_, "Rows may not be added after partitions() has been called");
  ++numRows_;
  if (firstFreeRow_) {
    row = firstFreeRow_;
    VELOX_CHECK(bits::isBitSet(row, freeFlagOffset_));
    firstFreeRow_ = nextFree(row);
    --numFreeRows_;
  } else {
    row = rows_.allocateFixed(fixedRowSize_ + normalizedKeySize_, alignment_) +
        normalizedKeySize_;
    if (normalizedKeySize_) {
      ++numRowsWithNormalizedKey_;
    }
  }
  return initializeRow(row, false /* reuse */);
}

char* RowContainer::initializeRow(char* row, bool reuse) {
  if (reuse) {
    auto rows = folly::Range<char**>(&row, 1);
    freeVariableWidthFields(rows);
    freeAggregates(rows);
  }

  if (!nullOffsets_.empty()) {
    memcpy(
        row + nullByte(nullOffsets_[0]),
        initialNulls_.data(),
        initialNulls_.size());
  }
  if (rowSizeOffset_) {
    variableRowSize(row) = 0;
  }
  bits::clearBit(row, freeFlagOffset_);
  return row;
}

void RowContainer::eraseRows(folly::Range<char**> rows) {
  freeVariableWidthFields(rows);
  freeAggregates(rows);
  numRows_ -= rows.size();
  for (auto* row : rows) {
    VELOX_CHECK(!bits::isBitSet(row, freeFlagOffset_), "Double free of row");
    bits::setBit(row, freeFlagOffset_);
    nextFree(row) = firstFreeRow_;
    firstFreeRow_ = row;
  }
  numFreeRows_ += rows.size();
}

void RowContainer::freeVariableWidthFields(folly::Range<char**> rows) {
  for (auto i = 0; i < types_.size(); ++i) {
    switch (typeKinds_[i]) {
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY:
      case TypeKind::ROW:
      case TypeKind::ARRAY:
      case TypeKind::MAP: {
        auto column = columnAt(i);
        for (auto row : rows) {
          if (!isNullAt(row, column.nullByte(), column.nullMask())) {
            StringView view = valueAt<StringView>(row, column.offset());
            if (!view.isInline()) {
              stringAllocator_.free(HashStringAllocator::headerOf(view.data()));
            }
          }
        }
      } break;
      default:;
    }
  }
}

void RowContainer::checkConsistency() {
  constexpr int32_t kBatch = 1000;
  std::vector<char*> rows(kBatch);

  RowContainerIterator iter;
  int64_t allocatedRows = 0;
  for (;;) {
    int64_t numRows = listRows(&iter, kBatch, rows.data());
    if (!numRows) {
      break;
    }
    for (auto i = 0; i < numRows; ++i) {
      auto row = rows[i];
      VELOX_CHECK(!bits::isBitSet(row, freeFlagOffset_));
      ++allocatedRows;
    }
  }

  size_t numFree = 0;
  for (auto free = firstFreeRow_; free; free = nextFree(free)) {
    ++numFree;
    VELOX_CHECK(bits::isBitSet(free, freeFlagOffset_));
  }
  VELOX_CHECK_EQ(numFree, numFreeRows_);
  VELOX_CHECK_EQ(allocatedRows, numRows_);
}

void RowContainer::freeAggregates(folly::Range<char**> rows) {
  for (auto& aggregate : aggregates_) {
    aggregate->destroy(rows);
  }
}

void RowContainer::store(
    const DecodedVector& decoded,
    vector_size_t index,
    char* row,
    int32_t column) {
  auto numKeys = keyTypes_.size();
  if (column < numKeys && !nullableKeys_) {
    VELOX_DYNAMIC_TYPE_DISPATCH(
        storeNoNulls,
        typeKinds_[column],
        decoded,
        index,
        row,
        offsets_[column]);
  } else {
    VELOX_DCHECK(column < keyTypes_.size() || aggregates_.empty());
    auto rowColumn = rowColumns_[column];
    VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
        storeWithNulls,
        typeKinds_[column],
        decoded,
        index,
        row,
        rowColumn.offset(),
        rowColumn.nullByte(),
        rowColumn.nullMask());
  }
}

void RowContainer::prepareRead(
    const char* row,
    int32_t offset,
    ByteStream& stream) {
  auto view = reinterpret_cast<const StringView*>(row + offset);
  if (view->isInline()) {
    stream.setRange(ByteRange{
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(view->data())),
        static_cast<int32_t>(view->size()),
        0});
    return;
  }
  // We set 'stream' to range over the ranges that start at the Header
  // immediately below the first character in the StringView.
  HashStringAllocator::prepareRead(
      HashStringAllocator::headerOf(view->data()), stream);
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
  BufferPtr buffer = values->getBufferWithSpace(value.size());
  auto start = buffer->size();
  buffer->setSize(start + value.size());
  ByteStream stream;
  HashStringAllocator::prepareRead(
      HashStringAllocator::headerOf(value.data()), stream);
  stream.readBytes(buffer->asMutable<char>() + start, value.size());
  values->setNoCopy(
      index, StringView(buffer->as<char>() + start, value.size()));
}

void RowContainer::storeComplexType(
    const DecodedVector& decoded,
    vector_size_t index,
    char* row,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask) {
  if (decoded.isNullAt(index)) {
    VELOX_DCHECK(nullMask);
    row[nullByte] |= nullMask;
    return;
  }
  RowSizeTracker tracker(row[rowSizeOffset_], stringAllocator_);
  ByteStream stream(&stringAllocator_, false, false);
  auto position = stringAllocator_.newWrite(stream);
  serde_.serialize(*decoded.base(), decoded.index(index), stream);
  stringAllocator_.finishWrite(stream, 0);
  valueAt<StringView>(row, offset) =
      StringView(reinterpret_cast<char*>(position.position), stream.size());
}

//   static
int32_t RowContainer::compareStringAsc(
    StringView left,
    const DecodedVector& decoded,
    vector_size_t index) {
  std::string storage;
  return HashStringAllocator::contiguousString(left, storage)
      .compare(decoded.valueAt<StringView>(index));
}

// static
int RowContainer::compareComplexType(
    const char* row,
    int32_t offset,
    const DecodedVector& decoded,
    vector_size_t index,
    CompareFlags flags) {
  VELOX_DCHECK(!flags.stopAtNull, "not supported compare flag");

  ByteStream stream;
  prepareRead(row, offset, stream);
  return serde_.compare(stream, decoded, index, flags);
}

int32_t RowContainer::compareStringAsc(StringView left, StringView right) {
  std::string leftStorage;
  std::string rightStorage;
  return HashStringAllocator::contiguousString(left, leftStorage)
      .compare(HashStringAllocator::contiguousString(right, rightStorage));
}

int32_t RowContainer::compareComplexType(
    const char* left,
    const char* right,
    const Type* type,
    int32_t offset,
    CompareFlags flags) {
  VELOX_DCHECK(!flags.stopAtNull, "not supported compare flag");

  ByteStream leftStream;
  ByteStream rightStream;
  prepareRead(left, offset, leftStream);
  prepareRead(right, offset, rightStream);
  return serde_.compare(leftStream, rightStream, type, flags);
}

template <TypeKind Kind>
void RowContainer::hashTyped(
    const Type* type,
    RowColumn column,
    bool nullable,
    folly::Range<char**> rows,
    bool mix,
    uint64_t* result) {
  using T = typename KindToFlatVector<Kind>::HashRowType;
  auto nullByte = column.nullByte();
  auto nullMask = column.nullMask();
  auto offset = column.offset();
  std::string storage;
  auto numRows = rows.size();
  for (int32_t i = 0; i < numRows; ++i) {
    char* row = rows[i];
    if (nullable && isNullAt(row, nullByte, nullMask)) {
      result[i] = mix ? bits::hashMix(result[i], BaseVector::kNullHash)
                      : BaseVector::kNullHash;
    } else {
      uint64_t hash;
      if (Kind == TypeKind::VARCHAR || Kind == TypeKind::VARBINARY) {
        hash =
            folly::hasher<StringView>()(HashStringAllocator::contiguousString(
                valueAt<StringView>(row, offset), storage));
      } else if (
          Kind == TypeKind::ROW || Kind == TypeKind::ARRAY ||
          Kind == TypeKind::MAP) {
        ByteStream in;
        prepareRead(row, offset, in);
        hash = serde_.hash(in, type);
      } else {
        hash = folly::hasher<T>()(valueAt<T>(row, offset));
      }
      result[i] = mix ? bits::hashMix(result[i], hash) : hash;
    }
  }
}

void RowContainer::hash(
    int32_t column,
    folly::Range<char**> rows,
    bool mix,
    uint64_t* result) {
  bool nullable = column >= keyTypes_.size() || nullableKeys_;
  VELOX_DYNAMIC_TYPE_DISPATCH(
      hashTyped,
      typeKinds_[column],
      types_[column].get(),
      columnAt(column),
      nullable,
      rows,
      mix,
      result);
}

void RowContainer::clear() {
  if (usesExternalMemory_) {
    constexpr int32_t kBatch = 1000;
    std::vector<char*> rows(kBatch);

    RowContainerIterator iter;
    for (;;) {
      int64_t numRows = listRows(&iter, kBatch, rows.data());
      if (!numRows) {
        break;
      }
      auto rowsData = folly::Range<char**>(rows.data(), numRows);
      freeAggregates(rowsData);
    }
  }
  rows_.clear();
  stringAllocator_.clear();
  numRows_ = 0;
  numRowsWithNormalizedKey_ = 0;
  normalizedKeySize_ = originalNormalizedKeySize_;
  numFreeRows_ = 0;
  firstFreeRow_ = nullptr;
}

void RowContainer::setProbedFlag(char** rows, int32_t numRows) {
  for (auto i = 0; i < numRows; i++) {
    // Row may be null in case of a FULL join.
    if (rows[i]) {
      setBit(rows[i], probedFlagOffset_);
    }
  }
}

void RowContainer::extractProbedFlags(
    const char* FOLLY_NONNULL const* FOLLY_NONNULL rows,
    int32_t numRows,
    bool setNullForNullKeysRow,
    bool setNullForNonProbedRow,
    const VectorPtr& result) {
  result->resize(numRows);
  result->clearAllNulls();
  auto flatResult = result->as<FlatVector<bool>>();
  auto* rawValues = flatResult->mutableRawValues<uint64_t>();
  for (auto i = 0; i < numRows; ++i) {
    // Check if this row has null keys.
    bool nullResult = false;
    if (setNullForNullKeysRow && nullableKeys_) {
      for (auto c = 0; c < keyTypes_.size(); ++c) {
        bool isNull =
            isNullAt(rows[i], columnAt(c).nullByte(), columnAt(c).nullMask());
        if (isNull) {
          nullResult = true;
          break;
        }
      }
    }

    if (nullResult) {
      flatResult->setNull(i, true);
    } else {
      bool probed = bits::isBitSet(rows[i], probedFlagOffset_);
      if (setNullForNonProbedRow && !probed) {
        flatResult->setNull(i, true);
      } else {
        bits::setBit(rawValues, i, probed);
      }
    }
  }
}

int64_t RowContainer::sizeIncrement(
    vector_size_t numRows,
    int64_t variableLengthBytes) const {
  constexpr int32_t kAllocUnit =
      AllocationPool::kMinPages * memory::AllocationTraits::kPageSize;
  int32_t needRows = std::max<int64_t>(0, numRows - numFreeRows_);
  int64_t needBytes =
      std::min<int64_t>(0, variableLengthBytes - stringAllocator_.freeSpace());
  return bits::roundUp(needRows * fixedRowSize_, kAllocUnit) +
      bits::roundUp(needBytes, kAllocUnit);
}

void RowContainer::skip(RowContainerIterator& iter, int32_t numRows) {
  VELOX_DCHECK(aggregates_.empty(), "Used in join only");
  VELOX_DCHECK_LE(0, numRows);
  if (!iter.endOfRun) {
    // Set to first row.
    VELOX_DCHECK_EQ(0, iter.rowNumber);
    VELOX_DCHECK_EQ(0, iter.allocationIndex);
    iter.normalizedKeysLeft = numRowsWithNormalizedKey_;
    iter.normalizedKeySize = originalNormalizedKeySize_;
    auto run = rows_.allocationAt(0)->runAt(0);
    iter.rowBegin = run.data<char>();
    iter.endOfRun = iter.rowBegin + run.numBytes();
  }
  if (iter.rowNumber + numRows >= numRows_) {
    iter.rowNumber = numRows_;
    iter.rowBegin = nullptr;
    return;
  }
  int32_t rowSize = fixedRowSize_ +
      (iter.normalizedKeysLeft > 0 ? originalNormalizedKeySize_ : 0);
  auto toSkip = numRows;
  if (iter.normalizedKeysLeft && iter.normalizedKeysLeft < numRows) {
    toSkip -= iter.normalizedKeysLeft;
    skip(iter, iter.normalizedKeysLeft);
    rowSize = fixedRowSize_;
  }
  while (toSkip) {
    if (iter.rowBegin &&
        toSkip * rowSize <= (iter.endOfRun - iter.rowBegin) - rowSize) {
      iter.rowBegin += toSkip * rowSize;
      break;
    }
    int32_t rowsInRun = (iter.endOfRun - iter.rowBegin) / rowSize;
    toSkip -= rowsInRun;
    auto numRuns = rows_.allocationAt(iter.allocationIndex)->numRuns();
    if (iter.runIndex >= numRuns - 1) {
      ++iter.allocationIndex;
      iter.runIndex = 0;
    } else {
      ++iter.runIndex;
    }
    auto run = rows_.allocationAt(iter.allocationIndex)->runAt(iter.runIndex);
    if (iter.allocationIndex == rows_.numSmallAllocations() - 1 &&
        iter.runIndex ==
            rows_.allocationAt(iter.allocationIndex)->numRuns() - 1) {
      iter.endOfRun = run.data<char>() + rows_.currentOffset();
    } else {
      iter.endOfRun = run.data<char>() + run.numBytes();
    }
    iter.rowBegin = run.data<char>();
  }
  if (iter.normalizedKeysLeft) {
    iter.normalizedKeysLeft -= numRows;
  }
  iter.rowNumber += numRows;
}

RowPartitions& RowContainer::partitions() {
  if (!partitions_) {
    partitions_ = std::make_unique<RowPartitions>(numRows_, *rows_.pool());
  }
  return *partitions_;
}

int32_t RowContainer::listPartitionRows(
    RowContainerIterator& iter,
    uint8_t partition,
    int32_t maxRows,
    char** result) {
  if (!numRows_) {
    return 0;
  }
  VELOX_CHECK(
      partitions_, "partitions() must be called before listPartitionRows()");
  VELOX_CHECK_EQ(
      partitions_->size(), numRows_, "All rows must have a partition");
  auto partitionNumberVector = xsimd::batch<uint8_t>::broadcast(partition);
  auto& allocation = partitions_->allocation();
  auto numRuns = allocation.numRuns();
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
        int32_t hit = __builtin_ctz(bits);
        auto distance = hit + startRow - iter.rowNumber;
        skip(iter, distance);
        result[numResults++] = iter.currentRow();
        if (numResults == maxRows) {
          skip(iter, 1);
          return numResults;
        }
        // Clear last set bit in 'bits'.
        bits &= bits - 1;
      }
      startRow += kBatch;
      // The last batch of 32 bytes may have been partly filled. If so, we could
      // have skipped past end.
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

RowPartitions::RowPartitions(int32_t numRows, memory::MemoryPool& pool)
    : capacity_(numRows) {
  auto numPages =
      bits::roundUp(capacity_, memory::AllocationTraits::kPageSize) /
      memory::AllocationTraits::kPageSize;
  pool.allocateNonContiguous(numPages, allocation_);
}

void RowPartitions::appendPartitions(folly::Range<const uint8_t*> partitions) {
  int32_t toAdd = partitions.size();
  int index = 0;
  VELOX_CHECK_LE(size_ + toAdd, capacity_);
  while (toAdd) {
    int32_t run;
    int32_t offset;
    allocation_.findRun(size_, &run, &offset);
    auto runSize = allocation_.runAt(run).numBytes();
    auto copySize = std::min<int32_t>(toAdd, runSize - offset);
    memcpy(
        allocation_.runAt(run).data<uint8_t>() + offset,
        &partitions[index],
        copySize);
    size_ += copySize;
    index += copySize;
    toAdd -= copySize;
    // Zero out to the next multiple of SIMD width for asan/valgring.
    if (!toAdd) {
      bits::padToAlignment(
          allocation_.runAt(run).data<uint8_t>(),
          runSize,
          offset + copySize,
          xsimd::batch<uint8_t>::size);
    }
  }
}

} // namespace facebook::velox::exec
