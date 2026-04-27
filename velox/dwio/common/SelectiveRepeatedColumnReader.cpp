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

#include "velox/dwio/common/SelectiveRepeatedColumnReader.h"

#include "velox/dwio/common/BufferUtil.h"
#include "velox/dwio/common/SelectiveColumnReaderInternal.h"

namespace facebook::velox::dwio::common {

namespace {

int sumLengths(
    const int32_t* lengths,
    const uint64_t* nulls,
    int first,
    int last) {
  int sum = 0;
  if (!nulls) {
    for (auto i = first; i < last; ++i) {
      sum += lengths[i];
    }
  } else if (last - first < 64) {
    bits::forEachSetBit(nulls, first, last, [&](int i) { sum += lengths[i]; });
  } else {
    xsimd::batch<int32_t> sums{};
    static_assert(sums.size <= 64);
    auto submask = bits::lowMask(sums.size);
    bits::forEachWord(first, last, [&](int i, uint64_t mask) {
      mask &= nulls[i];
      for (int j = 0; j < 64 && mask; j += sums.size) {
        if (auto m = (mask >> j) & submask) {
          auto selected = simd::fromBitMask<int32_t>(m);
          sums += simd::maskLoad(&lengths[i * 64 + j], selected);
        }
      }
    });
    sum = xsimd::reduce_add(sums);
  }
  return sum;
}

void prepareResult(
    VectorPtr& result,
    const TypePtr& type,
    vector_size_t size,
    memory::MemoryPool* pool) {
  if (!(result &&
        ((type->kind() == TypeKind::ARRAY &&
          result->encoding() == VectorEncoding::Simple::ARRAY) ||
         (type->kind() == TypeKind::MAP &&
          result->encoding() == VectorEncoding::Simple::MAP)) &&
        result.use_count() == 1)) {
    VLOG(1) << "Reallocating result " << type->kind() << " vector of size "
            << size;
    result = BaseVector::create(type, size, pool);
    return;
  }
  result->resetDataDependentFlags(nullptr);
  result->resize(size);
  // Nulls are handled in getValues calls.  Offsets and sizes are handled in
  // makeOffsetsAndSizes.  Child vectors are handled in child column readers.
}

} // namespace

void SelectiveRepeatedColumnReader::ensureAllLengthsBuffer(vector_size_t size) {
  if (!allLengthsHolder_ ||
      allLengthsHolder_->capacity() < size * sizeof(vector_size_t)) {
    allLengthsHolder_ = allocateIndices(size, pool_);
    allLengths_ = allLengthsHolder_->asMutable<vector_size_t>();
  }
}

void SelectiveRepeatedColumnReader::makeNestedRowSet(
    const RowSet& rows,
    int32_t maxRow) {
  ensureAllLengthsBuffer(maxRow + 1);
  auto* nulls = nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
  // Reads the lengths, leaves an uninitialized gap for a null
  // map/list. Reading these checks the null mask.
  readLengths(allLengths_, maxRow + 1, nulls);

  vector_size_t nestedLength;
  if (nestedRowsAllSelected_) {
    nestedLength = sumLengths(allLengths_, nulls, 0, maxRow + 1);
    childTargetReadOffset_ += nestedLength;
    nestedRows_ = RowSet(iota(nestedLength, nestedRowsHolder_), nestedLength);
    return;
  }

  nestedLength = 0;
  for (auto row : rows) {
    if (!nulls || !bits::isBitNull(nulls, row)) {
      nestedLength += prunedLengthAt(row);
    }
  }
  nestedRowsHolder_.resize(nestedLength);

  vector_size_t currentRow = 0;
  vector_size_t nestedRow = 0;
  vector_size_t nestedOffset = 0;
  for (auto rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
    const auto row = rows[rowIndex];
    // Add up the lengths of non-null rows skipped since the last
    // non-null.
    nestedOffset += sumLengths(allLengths_, nulls, currentRow, row);
    currentRow = row + 1;
    if (nulls && bits::isBitNull(nulls, row)) {
      continue;
    }
    const auto lengthAtRow = prunedLengthAt(row);
    std::iota(
        nestedRowsHolder_.data() + nestedRow,
        nestedRowsHolder_.data() + nestedRow + lengthAtRow,
        nestedOffset);
    nestedRow += lengthAtRow;
    nestedOffset += allLengths_[row];
  }
  nestedOffset += sumLengths(allLengths_, nulls, currentRow, maxRow + 1);
  childTargetReadOffset_ += nestedOffset;
  nestedRows_ = nestedRowsHolder_;
}

void SelectiveRepeatedColumnReader::makeOffsetsAndSizes(
    const RowSet& rows,
    ArrayVectorBase& result) {
  auto* rawOffsets =
      result.mutableOffsets(rows.size())->asMutable<vector_size_t>();
  auto* rawSizes = result.mutableSizes(rows.size())->asMutable<vector_size_t>();
  auto* nulls = nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
  numValues_ = rows.size();
  vector_size_t currentOffset = 0;
  if (nestedRowsAllSelected_ && rows.size() == outputRows().size()) {
    if (nulls) {
      for (int i = 0; i < rows.size(); ++i) {
        VELOX_DCHECK_EQ(i, rows[i]);
        rawOffsets[i] = currentOffset;
        if (bits::isBitNull(nulls, i)) {
          rawSizes[i] = 0;
          anyNulls_ = true;
        } else {
          rawSizes[i] = allLengths_[i];
          currentOffset += allLengths_[i];
        }
      }
    } else {
      for (int i = 0; i < rows.size(); ++i) {
        VELOX_DCHECK_EQ(i, rows[i]);
        rawOffsets[i] = currentOffset;
        rawSizes[i] = allLengths_[i];
        currentOffset += allLengths_[i];
      }
    }
    return;
  }
  vector_size_t currentRow = 0;
  vector_size_t nestedRowIndex = 0;
  for (int i = 0; i < rows.size(); ++i) {
    const auto row = rows[i];
    currentOffset += sumLengths(allLengths_, nulls, currentRow, row);
    currentRow = row + 1;
    nestedRowIndex =
        advanceNestedRows(nestedRows_, nestedRowIndex, currentOffset);
    rawOffsets[i] = nestedRowIndex;
    if (nulls && bits::isBitNull(nulls, row)) {
      rawSizes[i] = 0;
      anyNulls_ = true;
    } else {
      currentOffset += allLengths_[row];
      const auto newNestedRowIndex =
          advanceNestedRows(nestedRows_, nestedRowIndex, currentOffset);
      rawSizes[i] = newNestedRowIndex - nestedRowIndex;
      nestedRowIndex = newNestedRowIndex;
    }
  }
}

RowSet SelectiveRepeatedColumnReader::applyFilter(const RowSet& rows) {
  if (!scanSpec_->filter()) {
    return rows;
  }
  switch (scanSpec_->filter()->kind()) {
    case velox::common::FilterKind::kIsNull:
      filterNulls<int32_t>(rows, true, false);
      break;
    case velox::common::FilterKind::kIsNotNull:
      filterNulls<int32_t>(rows, false, false);
      break;
    default:
      VELOX_UNSUPPORTED(
          "Unsupported filter for column {}, only IS NULL and IS NOT NULL are supported: {}",
          scanSpec_->fieldName(),
          scanSpec_->filter()->toString());
  }
  return outputRows_;
}

void SelectiveRepeatedColumnReader::getExtractionSizeValues(
    const RowSet& rows,
    VectorPtr* result) {
  VELOX_DCHECK_NOT_NULL(result);
  FlatVector<int64_t>* flatResult = nullptr;
  if (*result && result->get()->type()->isBigint()) {
    flatResult = result->get()->asFlatVector<int64_t>();
  }
  if (!flatResult || !flatResult->values()) {
    *result = std::make_shared<FlatVector<int64_t>>(
        pool_,
        BIGINT(),
        nullptr,
        rows.size(),
        AlignedBuffer::allocate<int64_t>(rows.size(), pool_),
        std::vector<BufferPtr>{});
    flatResult = result->get()->asFlatVector<int64_t>();
  } else {
    flatResult->resize(static_cast<vector_size_t>(rows.size()));
  }
  auto* sizesData = flatResult->mutableRawValues();
  auto* nulls = nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
  for (vector_size_t i = 0; i < static_cast<vector_size_t>(rows.size()); ++i) {
    sizesData[i] =
        (nulls && bits::isBitNull(nulls, rows[i])) ? 0 : allLengths_[rows[i]];
  }
  setComplexNulls(rows, *result);
}

SelectiveListColumnReader::SelectiveListColumnReader(
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    FormatParams& params,
    velox::common::ScanSpec& scanSpec)
    : SelectiveRepeatedColumnReader(requestedType, params, scanSpec, fileType) {
  VELOX_CHECK(
      scanSpec.extractionType() ==
              velox::common::ScanSpec::ExtractionType::kNone ||
          scanSpec.extractionType() ==
              velox::common::ScanSpec::ExtractionType::kSize,
      "Array column reader only supports kNone and kSize extraction, got: {}",
      static_cast<int>(scanSpec.extractionType()));
}

uint64_t SelectiveListColumnReader::skip(uint64_t numValues) {
  numValues = formatData_->skipNulls(numValues);
  std::array<int32_t, kBufferSize> buffer{};
  uint64_t childElements = 0;
  uint64_t lengthsRead = 0;
  while (lengthsRead < numValues) {
    uint64_t chunk =
        std::min(numValues - lengthsRead, static_cast<uint64_t>(kBufferSize));
    readLengths(buffer.data(), static_cast<int32_t>(chunk), nullptr);
    for (size_t i = 0; i < chunk; ++i) {
      childElements += static_cast<size_t>(buffer[i]);
    }
    lengthsRead += chunk;
  }
  if (child_) {
    child_->seekTo(
        child_->readOffset() + static_cast<int64_t>(childElements), false);
  }
  childTargetReadOffset_ += static_cast<int64_t>(childElements);
  return numValues;
}

void SelectiveListColumnReader::read(
    int64_t offset,
    const RowSet& rows,
    const uint64_t* incomingNulls) {
  // Catch up if the child is behind the length stream.
  if (child_) {
    child_->seekTo(childTargetReadOffset_, false);
  }
  prepareRead<char>(offset, rows, incomingNulls);
  auto activeRows = applyFilter(rows);
  nestedRowsAllSelected_ = activeRows.size() == rows.back() + 1 &&
      scanSpec_->maxArrayElementsCount() ==
          std::numeric_limits<vector_size_t>::max();
  makeNestedRowSet(activeRows, rows.back());
  // When deltaUpdate is set, treat extractionType as kNone so all child
  // streams are read.  The extraction transform is applied after the
  // delta update.
  if (scanSpec_->extractionType() ==
          velox::common::ScanSpec::ExtractionType::kSize &&
      !scanSpec_->deltaUpdate()) {
    // Size extraction: only need offsets/sizes, skip child stream.
    if (child_ && !nestedRows_.empty()) {
      child_->seekTo(child_->readOffset() + nestedRows_.back() + 1, false);
    }
  } else if (child_ && !nestedRows_.empty()) {
    child_->readWithTiming(child_->readOffset(), nestedRows_, nullptr);
    nestedRowsAllSelected_ = nestedRowsAllSelected_ &&
        nestedRows_.size() == child_->outputRows().size();
    nestedRows_ = child_->outputRows();
  }
  numValues_ = activeRows.size();
  readOffset_ = offset + rows.back() + 1;
}

void SelectiveListColumnReader::getValues(
    const RowSet& rows,
    VectorPtr* result) {
  VELOX_DCHECK_NOT_NULL(result);

  // When deltaUpdate is set, treat extractionType as kNone so the reader
  // produces the full array.  The extraction transform is applied after
  // the delta update.
  if (scanSpec_->extractionType() ==
          velox::common::ScanSpec::ExtractionType::kSize &&
      !scanSpec_->deltaUpdate()) {
    getExtractionSizeValues(rows, result);
    return;
  }

  prepareResult(*result, requestedType_, rows.size(), pool_);
  auto* resultArray = result->get()->asUnchecked<ArrayVector>();
  makeOffsetsAndSizes(rows, *resultArray);
  setComplexNulls(rows, *result);
  if (child_ && !nestedRows_.empty()) {
    auto& elements = resultArray->elements();
    prepareStructResult(requestedType_->childAt(0), &elements);
    child_->getValues(nestedRows_, &elements);
  }
}

uint64_t SelectiveMapColumnReaderBase::skip(uint64_t numValues) {
  numValues = formatData_->skipNulls(numValues);
  if (keyReader_ || elementReader_) {
    std::array<int32_t, kBufferSize> buffer;
    uint64_t childElements{0};
    uint64_t lengthsRead{0};
    while (lengthsRead < numValues) {
      const uint64_t chunk =
          std::min(numValues - lengthsRead, static_cast<uint64_t>(kBufferSize));
      readLengths(buffer.data(), chunk, nullptr);
      for (size_t i = 0; i < chunk; ++i) {
        childElements += buffer[i];
      }
      lengthsRead += chunk;
    }

    if (keyReader_) {
      keyReader_->seekTo(keyReader_->readOffset() + childElements, false);
    }
    if (elementReader_) {
      elementReader_->seekTo(
          elementReader_->readOffset() + childElements, false);
    }
    childTargetReadOffset_ += childElements;
  } else {
    VELOX_FAIL("repeated reader with no children");
  }
  return numValues;
}

void SelectiveMapColumnReaderBase::read(
    int64_t offset,
    const RowSet& rows,
    const uint64_t* incomingNulls) {
  // When deltaUpdate is set, treat extractionType as kNone so all streams
  // are read.  The extraction transform is applied after the delta update.
  const auto extractionType = scanSpec_->deltaUpdate()
      ? velox::common::ScanSpec::ExtractionType::kNone
      : scanSpec_->extractionType();

  // Catch up if child readers are behind the length stream.
  if (keyReader_) {
    keyReader_->seekTo(childTargetReadOffset_, false);
  }
  if (elementReader_) {
    elementReader_->seekTo(childTargetReadOffset_, false);
  }

  prepareRead<char>(offset, rows, incomingNulls);
  const auto activeRows = applyFilter(rows);
  nestedRowsAllSelected_ = activeRows.size() == rows.back() + 1;
  VELOX_CHECK_EQ(
      scanSpec_->maxArrayElementsCount(),
      std::numeric_limits<vector_size_t>::max());
  makeNestedRowSet(activeRows, rows.back());

  if (extractionType == velox::common::ScanSpec::ExtractionType::kSize) {
    // Size extraction: only need offsets/sizes, skip both key and value
    // streams.  Advance children past the nested rows without reading.
    if (keyReader_ && !nestedRows_.empty()) {
      keyReader_->seekTo(
          keyReader_->readOffset() + nestedRows_.back() + 1, false);
    }
    if (elementReader_ && !nestedRows_.empty()) {
      elementReader_->seekTo(
          elementReader_->readOffset() + nestedRows_.back() + 1, false);
    }
  } else if (extractionType == velox::common::ScanSpec::ExtractionType::kKeys) {
    // Keys extraction: read only keys, skip values.
    if (keyReader_ && !nestedRows_.empty()) {
      keyReader_->readWithTiming(
          keyReader_->readOffset(), nestedRows_, nullptr);
      nestedRowsAllSelected_ = nestedRowsAllSelected_ &&
          nestedRows_.size() == keyReader_->outputRows().size();
      nestedRows_ = keyReader_->outputRows();
    }
    if (elementReader_ && !nestedRows_.empty()) {
      elementReader_->seekTo(
          elementReader_->readOffset() + nestedRows_.back() + 1, false);
    }
  } else if (
      extractionType == velox::common::ScanSpec::ExtractionType::kValues) {
    // Values extraction: read only values, skip keys.
    if (keyReader_ && !nestedRows_.empty()) {
      keyReader_->seekTo(
          keyReader_->readOffset() + nestedRows_.back() + 1, false);
    }
    if (elementReader_ && !nestedRows_.empty()) {
      elementReader_->readWithTiming(
          elementReader_->readOffset(), nestedRows_, nullptr);
      nestedRowsAllSelected_ = nestedRowsAllSelected_ &&
          nestedRows_.size() == elementReader_->outputRows().size();
      nestedRows_ = elementReader_->outputRows();
    }
  } else {
    // Normal read: read both keys and values.
    VELOX_CHECK_EQ(
        static_cast<int>(extractionType),
        static_cast<int>(velox::common::ScanSpec::ExtractionType::kNone));
    if (keyReader_ && elementReader_ && !nestedRows_.empty()) {
      keyReader_->readWithTiming(
          keyReader_->readOffset(), nestedRows_, nullptr);
      nestedRowsAllSelected_ = nestedRowsAllSelected_ &&
          nestedRows_.size() == keyReader_->outputRows().size();
      nestedRows_ = keyReader_->outputRows();
      if (!nestedRows_.empty()) {
        elementReader_->readWithTiming(
            elementReader_->readOffset(), nestedRows_, nullptr);
        nestedRowsAllSelected_ = nestedRowsAllSelected_ &&
            nestedRows_.size() == elementReader_->outputRows().size();
        nestedRows_ = elementReader_->outputRows();
      }
    }
  }
  numValues_ = activeRows.size();
  readOffset_ = offset + rows.back() + 1;
}

SelectiveMapColumnReader::SelectiveMapColumnReader(
    const TypePtr& requestedType,
    const TypeWithIdPtr& fileType,
    FormatParams& params,
    ScanSpec& scanSpec)
    : SelectiveMapColumnReaderBase(requestedType, params, scanSpec, fileType) {
  VELOX_CHECK(!scanSpec_->isFlatMapAsStruct());
  // We should not need this anymore.  Is there a safe way to find out if there
  // is any prod usages that forget to set up the map children in scan spec?
  // This should be only possible when user bypasses the connector interface and
  // create file readers directly.
  if (scanSpec_->children().empty()) {
    scanSpec_->getOrCreateChild(ScanSpec::kMapKeysFieldName);
    scanSpec_->getOrCreateChild(ScanSpec::kMapValuesFieldName);
  }
  scanSpec_->children()[0]->setProjectOut(true);
  scanSpec_->children()[1]->setProjectOut(true);
}

void SelectiveMapColumnReaderBase::getExtractionValues(
    const RowSet& rows,
    VectorPtr* result) {
  VELOX_DCHECK_NOT_NULL(result);
  const auto extractionType = scanSpec_->extractionType();
  VELOX_DCHECK_NE(
      static_cast<int>(extractionType),
      static_cast<int>(velox::common::ScanSpec::ExtractionType::kNone));

  // When deltaUpdate is set, treat extractionType as kNone so the reader
  // produces the full map.  The extraction transform is applied after
  // the delta update.
  if (extractionType == velox::common::ScanSpec::ExtractionType::kSize &&
      !scanSpec_->deltaUpdate()) {
    getExtractionSizeValues(rows, result);
    return;
  }

  // kKeys or kValues: compute offsets/sizes via a reusable MapVector,
  // then read elements and construct the output ArrayVector.
  prepareResult(
      extractionOffsetsTemp_,
      requestedType_,
      static_cast<vector_size_t>(rows.size()),
      pool_);
  auto* tempMap = extractionOffsetsTemp_->asUnchecked<MapVector>();
  makeOffsetsAndSizes(rows, *tempMap);
  setComplexNulls(rows, extractionOffsetsTemp_);

  // Extract elements from the existing result to reuse across batches.
  VectorPtr elements;
  if (*result && result->get()->encoding() == VectorEncoding::Simple::ARRAY) {
    elements = result->get()->asUnchecked<ArrayVector>()->elements();
  }
  if (extractionType == velox::common::ScanSpec::ExtractionType::kKeys) {
    if (!nestedRows_.empty()) {
      keyReader_->getValues(nestedRows_, &elements);
    }
  } else {
    if (!nestedRows_.empty()) {
      prepareStructResult(requestedType_->childAt(1), &elements);
      elementReader_->getValues(nestedRows_, &elements);
    }
  }
  auto elemType = elements
      ? elements->type()
      : requestedType_->childAt(
            extractionType == velox::common::ScanSpec::ExtractionType::kKeys
                ? 0
                : 1);
  *result = std::make_shared<ArrayVector>(
      pool_,
      ARRAY(elemType),
      tempMap->nulls(),
      rows.size(),
      tempMap->offsets(),
      tempMap->sizes(),
      elements);
}

void SelectiveMapColumnReader::getValues(
    const RowSet& rows,
    VectorPtr* result) {
  VELOX_DCHECK_NOT_NULL(result);
  const auto extractionType = scanSpec_->extractionType();

  // When deltaUpdate is set, treat extractionType as kNone so the reader
  // produces the full map.  The extraction transform is applied after
  // the delta update.
  if (extractionType != velox::common::ScanSpec::ExtractionType::kNone &&
      !scanSpec_->deltaUpdate()) {
    getExtractionValues(rows, result);
    return;
  }

  // Normal path: produce MapVector.  If the result has a non-MAP type
  // (e.g., from a previous extraction transform), prepareResult will
  // replace it with a fresh MapVector.
  prepareResult(*result, requestedType_, rows.size(), pool_);
  auto* resultMap = result->get()->asUnchecked<MapVector>();
  makeOffsetsAndSizes(rows, *resultMap);
  setComplexNulls(rows, *result);
  VELOX_CHECK(
      keyReader_ && elementReader_,
      "keyReader_ and elementReaer_ must exist in "
      "SelectiveMapColumnReader::getValues");
  if (!nestedRows_.empty()) {
    keyReader_->getValues(nestedRows_, &resultMap->mapKeys());
    auto& values = resultMap->mapValues();
    prepareStructResult(requestedType_->childAt(1), &values);
    elementReader_->getValues(nestedRows_, &values);
  }
}

SelectiveMapAsStructColumnReader::SelectiveMapAsStructColumnReader(
    const TypePtr& requestedType,
    const TypeWithIdPtr& fileType,
    FormatParams& params,
    ScanSpec& scanSpec)
    : SelectiveMapColumnReaderBase(requestedType, params, scanSpec, fileType) {
  VELOX_CHECK(scanSpec_->isFlatMapAsStruct() && requestedType_->isMap());
  VELOX_CHECK_EQ(
      static_cast<int>(scanSpec.extractionType()),
      static_cast<int>(velox::common::ScanSpec::ExtractionType::kNone),
      "Flat map as struct reader does not support extraction pushdown");
  mapScanSpec_.addMapKeyFieldRecursively(*requestedType_->childAt(0));
  mapScanSpec_.addMapValueFieldRecursively(*requestedType_->childAt(1));
  column_index_t maxChannel = 0;
  for (auto& childSpec : scanSpec_->children()) {
    auto field = folly::tryTo<int64_t>(childSpec->fieldName());
    VELOX_CHECK(
        field.hasValue(),
        "Fail to parse field name: {}",
        childSpec->fieldName());
    keyToIndex_[*field] = childSpec->channel();
    maxChannel = std::max(maxChannel, childSpec->channel());
  }
  copyRanges_.resize(maxChannel + 1);
}

void SelectiveMapAsStructColumnReader::getValues(
    const RowSet& rows,
    VectorPtr* result) {
  VELOX_CHECK_NOT_NULL(*result);
  VELOX_CHECK(
      result->get()->type()->isRow(),
      "Expect ROW, got {}",
      result->get()->type()->toString());
  BaseVector::prepareForReuse(*result, rows.size());
  auto* resultRow = result->get()->asChecked<RowVector>();
  setComplexNulls(rows, *result);
  for (auto& child : resultRow->children()) {
    bits::fillBits(child->mutableRawNulls(), 0, rows.size(), bits::kNull);
  }
  numValues_ = rows.size();
  if (nestedRows_.empty()) {
    return;
  }
  keyReader_->getValues(nestedRows_, &mapKeys_);
  prepareStructResult(requestedType_->childAt(1), &mapValues_);
  elementReader_->getValues(nestedRows_, &mapValues_);
  decodedKeys_.decode(*mapKeys_);
  for (auto& ranges : copyRanges_) {
    ranges.clear();
  }
  switch (mapKeys_->type()->kind()) {
    case TypeKind::TINYINT:
      makeCopyRanges<int8_t>(rows);
      break;
    case TypeKind::SMALLINT:
      makeCopyRanges<int16_t>(rows);
      break;
    case TypeKind::INTEGER:
      makeCopyRanges<int32_t>(rows);
      break;
    case TypeKind::BIGINT:
      makeCopyRanges<int64_t>(rows);
      break;
    default:
      VELOX_UNSUPPORTED(
          "Unsupported key type: {}", mapKeys_->type()->toString());
  }
  for (column_index_t i = 0; i < resultRow->childrenSize(); ++i) {
    resultRow->childAt(i)->copyRanges(mapValues_.get(), copyRanges_[i]);
  }
}

template <typename T>
void SelectiveMapAsStructColumnReader::makeCopyRanges(const RowSet& rows) {
  auto* nulls = nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
  for (vector_size_t i = 0,
                     currentOffset = 0,
                     currentRow = 0,
                     nestedRowIndex = 0;
       i < rows.size();
       ++i) {
    const auto row = rows[i];
    if (nulls && bits::isBitNull(nulls, row)) {
      anyNulls_ = true;
      continue;
    }
    currentOffset += sumLengths(allLengths_, nulls, currentRow, row);
    currentRow = row + 1;
    nestedRowIndex =
        advanceNestedRows(nestedRows_, nestedRowIndex, currentOffset);
    currentOffset += allLengths_[row];
    const auto newNestedRowIndex =
        advanceNestedRows(nestedRows_, nestedRowIndex, currentOffset);
    for (auto j = nestedRowIndex; j < newNestedRowIndex; ++j) {
      VELOX_CHECK(!decodedKeys_.isNullAt(j));
      auto it = keyToIndex_.find(decodedKeys_.valueAt<T>(j));
      if (it == keyToIndex_.end()) {
        continue;
      }
      copyRanges_[it->second].push_back({
          .sourceIndex = j,
          .targetIndex = i,
          .count = 1,
      });
    }
    nestedRowIndex = newNestedRowIndex;
  }
}

} // namespace facebook::velox::dwio::common
