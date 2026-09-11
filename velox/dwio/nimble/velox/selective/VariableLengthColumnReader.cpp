/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/nimble/velox/selective/VariableLengthColumnReader.h"

#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/velox/selective/ColumnReader.h"
#include "velox/vector/DictionaryVector.h"

namespace facebook::nimble {

namespace {

bool estimateMaterializedSizeImpl(
    velox::dwio::common::FormatData& formatData,
    const std::vector<velox::dwio::common::SelectiveColumnReader*>& children,
    size_t& byteSize,
    size_t& rowCount) {
  auto lengthsRowCount =
      formatData.as<NimbleData>().nullsDecoder()->estimateRowCount();
  if (!lengthsRowCount.has_value()) {
    return false;
  }
  rowCount = *lengthsRowCount;
  size_t rowSize = 8;
  for (auto* child : children) {
    if (!child) {
      continue;
    }
    size_t childByteSize, childRowCount;
    if (!child->estimateMaterializedSize(childByteSize, childRowCount)) {
      return false;
    }
    if (rowCount > 0) {
      rowSize += childByteSize / rowCount;
    }
  }
  byteSize = rowSize * rowCount;
  return true;
}

void tryReuseArrayVectorBase(
    velox::ArrayVectorBase& vector,
    velox::vector_size_t size,
    velox::BufferPtr& offsets,
    velox::BufferPtr& sizes) {
  if (vector.offsets() && vector.offsets()->isMutable()) {
    offsets = vector.mutableOffsets(size);
  }
  if (vector.sizes() && vector.sizes()->isMutable()) {
    sizes = vector.mutableSizes(size);
  }
}

velox::VectorPtr tryReuseWritableVector(const velox::VectorPtr& vector) {
  if (vector && vector->encoding() != velox::VectorEncoding::Simple::CONSTANT &&
      vector->encoding() != velox::VectorEncoding::Simple::DICTIONARY &&
      vector.use_count() == 1) {
    vector->resize(0);
    return vector;
  }
  return nullptr;
}

void tryReuseArrayVector(
    velox::ArrayVector& vector,
    velox::vector_size_t size,
    velox::BufferPtr& offsets,
    velox::BufferPtr& sizes,
    velox::VectorPtr& elements) {
  tryReuseArrayVectorBase(vector, size, offsets, sizes);
  elements = tryReuseWritableVector(vector.elements());
}

void tryReuseMapVector(
    velox::MapVector& vector,
    velox::vector_size_t size,
    velox::BufferPtr& offsets,
    velox::BufferPtr& sizes,
    velox::VectorPtr& keys,
    velox::VectorPtr& values) {
  tryReuseArrayVectorBase(vector, size, offsets, sizes);
  keys = tryReuseWritableVector(vector.mapKeys());
  values = tryReuseWritableVector(vector.mapValues());
}

velox::DictionaryVector<velox::ComplexType>* prepareDictionaryArrayResult(
    velox::VectorPtr& result,
    const velox::TypePtr& type,
    velox::vector_size_t size,
    velox::memory::MemoryPool* const pool) {
  velox::VectorPtr alphabet;
  velox::BufferPtr nulls, indices, sizes, offsets;
  velox::VectorPtr elements;
  auto dictionaryVector = result
      ? result->as<velox::DictionaryVector<velox::ComplexType>>()
      : nullptr;
  if (result && result.unique()) {
    if (result->nulls() && result->nulls()->isMutable()) {
      nulls = result->mutableNulls(size);
    }
    if (dictionaryVector) {
      if (dictionaryVector->indices() &&
          dictionaryVector->indices()->isMutable()) {
        indices = dictionaryVector->mutableIndices(size);
      }
      if (dictionaryVector->valueVector() &&
          dictionaryVector->valueVector().unique()) {
        alphabet = dictionaryVector->valueVector();
        auto arrayVector = alphabet->as<velox::ArrayVector>();
        if (arrayVector) {
          tryReuseArrayVector(*arrayVector, size, offsets, sizes, elements);
        }
      }
    } else if (auto arrayVector = result->as<velox::ArrayVector>()) {
      tryReuseArrayVector(*arrayVector, size, offsets, sizes, elements);
    }
  }

  if (!nulls) {
    nulls = velox::AlignedBuffer::allocate<uint64_t>(
        velox::bits::nwords(size), pool);
  }

  if (!offsets) {
    offsets = velox::AlignedBuffer::allocate<velox::vector_size_t>(size, pool);
  }
  if (!sizes) {
    sizes = velox::AlignedBuffer::allocate<velox::vector_size_t>(size, pool);
  }

  alphabet = std::make_shared<velox::ArrayVector>(
      pool,
      type,
      nulls,
      size,
      std::move(offsets),
      std::move(sizes),
      /* elements */ elements);

  // Then prepare the dictionary vector.
  if (!indices) {
    indices = velox::AlignedBuffer::allocate<velox::vector_size_t>(size, pool);
  }

  auto rawIndices = indices->asMutable<velox::vector_size_t>();
  std::fill(rawIndices, rawIndices + size, 0);
  alphabet->resize(0);

  result = velox::BaseVector::wrapInDictionary(
      /* nulls */ nullptr,
      /* indices */ std::move(indices),
      /* size */ 0,
      /* values */ std::move(alphabet));
  dictionaryVector = result->as<velox::DictionaryVector<velox::ComplexType>>();
  dictionaryVector->resize(size);

  return dictionaryVector;
}

velox::DictionaryVector<velox::ComplexType>* prepareDictionaryMapResult(
    velox::VectorPtr& result,
    const velox::TypePtr& type,
    velox::vector_size_t size,
    velox::memory::MemoryPool* const pool) {
  velox::VectorPtr alphabet;
  velox::BufferPtr nulls, indices, sizes, offsets;
  velox::VectorPtr keys;
  velox::VectorPtr elements;
  auto dictionaryVector = result
      ? result->as<velox::DictionaryVector<velox::ComplexType>>()
      : nullptr;
  if (result && result.unique()) {
    if (result->nulls() && result->nulls()->isMutable()) {
      nulls = result->mutableNulls(size);
    }
    if (dictionaryVector) {
      if (dictionaryVector->indices() &&
          dictionaryVector->indices()->isMutable()) {
        indices = dictionaryVector->mutableIndices(size);
      }
      if (dictionaryVector->valueVector() &&
          dictionaryVector->valueVector().unique()) {
        alphabet = dictionaryVector->valueVector();
        auto mapVector = alphabet->as<velox::MapVector>();
        if (mapVector) {
          tryReuseMapVector(*mapVector, size, offsets, sizes, keys, elements);
        }
      }
    } else if (auto mapVector = result->as<velox::MapVector>()) {
      tryReuseMapVector(*mapVector, size, offsets, sizes, keys, elements);
    }
  }

  if (!nulls) {
    nulls = velox::AlignedBuffer::allocate<uint64_t>(
        velox::bits::nwords(size), pool);
  }

  if (!offsets) {
    offsets = velox::AlignedBuffer::allocate<velox::vector_size_t>(size, pool);
  }
  if (!sizes) {
    sizes = velox::AlignedBuffer::allocate<velox::vector_size_t>(size, pool);
  }

  alphabet = std::make_shared<velox::MapVector>(
      pool,
      type,
      nulls,
      size,
      std::move(offsets),
      std::move(sizes),
      keys,
      elements);

  // Then prepare the dictionary vector.
  if (!indices) {
    indices = velox::AlignedBuffer::allocate<velox::vector_size_t>(size, pool);
  }

  auto rawIndices = indices->asMutable<velox::vector_size_t>();
  std::fill(rawIndices, rawIndices + size, 0);
  alphabet->resize(0);

  result = velox::BaseVector::wrapInDictionary(
      /* nulls */ nullptr,
      /* indices */ std::move(indices),
      /* size */ 0,
      /* values */ std::move(alphabet));
  dictionaryVector = result->as<velox::DictionaryVector<velox::ComplexType>>();
  dictionaryVector->resize(size);

  return dictionaryVector;
}

// There are 2 more efficient alternative ways than the current naive
// implementation if needed:
//
// 1) Append the cached run value to the end and fix all the 0 indices
// 2) More intrusive api for child readers to materialize into a certain
//    position of the output vector.
//
// Prefer 2) over 1) due to producing a more cache friendly dictionary layout
// and for being defensive against bad UDF assumptions.  We probably won't do 2)
// for the sole purpose of deduplicated file types though, but if we ever do it
// for other reasons, we can leverage it here.
void appendToLastRun(
    const velox::BaseVector& lastRun,
    velox::dwio::common::SelectiveColumnReader& child,
    velox::RowSet nestedRows,
    velox::VectorPtr& output) {
  velox::VectorPtr temp;
  child.getValues(nestedRows, &temp);
  if (!output) {
    output = velox::BaseVector::create(
        temp->type(), temp->size() + lastRun.size(), temp->pool());
  } else {
    output->resize(temp->size() + lastRun.size());
  }
  output->copy(&lastRun, 0, 0, lastRun.size());
  output->copy(temp.get(), lastRun.size(), 0, temp->size());
}

void loadLastRun(
    const velox::BaseVector& input,
    vector_size_t size,
    velox::VectorPtr& output) {
  if (!output) {
    output = velox::BaseVector::create(input.type(), size, input.pool());
  } else {
    output->resize(size);
  }
  output->copy(&input, 0, input.size() - size, size);
}

} // namespace

ListColumnReader::ListColumnReader(
    const velox::TypePtr& requestedType,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
    NimbleParams& params,
    velox::common::ScanSpec& scanSpec)
    : SelectiveListColumnReader{requestedType, fileType, params, scanSpec} {
  auto& childType = requestedType_->childAt(0);
  auto& childFileType = fileType_->childAt(0);
  if (scanSpec_->children().empty()) {
    scanSpec.getOrCreateChild(
        velox::common::Subfield(
            velox::common::ScanSpec::kArrayElementsFieldName));
  }
  scanSpec_->children()[0]->setProjectOut(true);

  // For kSize extraction we only need the length stream, so skip creating
  // the element reader entirely.  This avoids registering its streams and
  // reduces IO.  When deltaUpdate is set, we still need the element reader
  // because delta updates operate on the full array.
  if (scanSpec.extractionType() ==
          velox::common::ScanSpec::ExtractionType::kSize &&
      !scanSpec.deltaUpdate()) {
    return;
  }

  auto childParams =
      params.makeChildParams(params.nimbleType()->asArray().elements());
  child_ = buildColumnReader(
      childType,
      childFileType,
      childParams,
      *scanSpec_->children()[0],
      /*isRoot=*/false);
  children_ = {child_.get()};
}

void ListColumnReader::readLengths(
    int32_t* lengths,
    int32_t numLengths,
    const uint64_t* nulls) {
  // Relies on the strong assumption about readNulls are called with exactly the
  // same row count.
  VELOX_CHECK_LE(
      numLengths * sizeof(int32_t),
      formatData().as<NimbleData>().getPreloadedValues()->capacity());
  std::memcpy(
      lengths,
      formatData().as<NimbleData>().getPreloadedValues()->as<int32_t>(),
      numLengths * sizeof(int32_t));
}

uint64_t ListColumnReader::skip(uint64_t numValues) {
  auto nullsBuffer = velox::AlignedBuffer::allocate<uint64_t>(
      velox::bits::nwords(numValues), pool_);
  auto nulls = nullsBuffer->as<uint64_t>();
  formatData_->readNulls(numValues, nullptr, nullsBuffer);
  if (child_) {
    nimble::Vector<int32_t> buffer{pool_, numValues};
    uint64_t childElements = 0;
    readLengths(buffer.data(), numValues, nullptr);
    for (size_t i = 0; i < numValues; ++i) {
      if (velox::bits::isBitSet(nulls, i)) {
        childElements += static_cast<size_t>(buffer[i]);
      }
    }
    child_->seekTo(child_->readOffset() + childElements, false);
    childTargetReadOffset_ += childElements;
  }
  return velox::bits::countNonNulls(nulls, 0, numValues);
}

bool ListColumnReader::estimateMaterializedSize(
    size_t& byteSize,
    size_t& rowCount) const {
  return estimateMaterializedSizeImpl(
      formatData(), children(), byteSize, rowCount);
}

namespace {

/// Returns true if the MAP extraction type needs the key child reader.
/// When deltaUpdate is set, all child readers are needed regardless of
/// ExtractionType because delta updates (e.g., MAP_CONCAT) operate on the
/// full map.
bool needsKeyReader(
    velox::common::ScanSpec::ExtractionType extractionType,
    bool hasDeltaUpdate) {
  return hasDeltaUpdate ||
      extractionType == velox::common::ScanSpec::ExtractionType::kNone ||
      extractionType == velox::common::ScanSpec::ExtractionType::kKeys;
}

/// Returns true if the MAP extraction type needs the value child reader.
/// When deltaUpdate is set, all child readers are needed regardless of
/// ExtractionType because delta updates (e.g., MAP_CONCAT) operate on the
/// full map.
bool needsElementReader(
    velox::common::ScanSpec::ExtractionType extractionType,
    bool hasDeltaUpdate) {
  return hasDeltaUpdate ||
      extractionType == velox::common::ScanSpec::ExtractionType::kNone ||
      extractionType == velox::common::ScanSpec::ExtractionType::kValues;
}

void makeMapChildrenReaders(
    const velox::dwio::common::TypeWithId& fileType,
    const velox::Type& requestedType,
    NimbleParams& params,
    const velox::common::ScanSpec& scanSpec,
    velox::common::ScanSpec::ExtractionType extractionType,
    bool hasDeltaUpdate,
    std::unique_ptr<velox::dwio::common::SelectiveColumnReader>& keyReader,
    std::unique_ptr<velox::dwio::common::SelectiveColumnReader>&
        elementReader) {
  auto& keyType = requestedType.childAt(0);
  auto& valueType = requestedType.childAt(1);
  auto& fileKeyType = fileType.childAt(0);
  auto& fileValueType = fileType.childAt(1);
  auto& mapSchemaType = params.nimbleType()->asMap();
  // Skip creating child readers that extraction pushdown doesn't need.
  // This avoids registering their streams and reduces IO.
  if (needsKeyReader(extractionType, hasDeltaUpdate)) {
    auto keyParams = params.makeChildParams(mapSchemaType.keys());
    keyReader = buildColumnReader(
        keyType,
        fileKeyType,
        keyParams,
        *scanSpec.children()[0],
        /*isRoot=*/false);
  }
  if (needsElementReader(extractionType, hasDeltaUpdate)) {
    auto valueParams = params.makeChildParams(mapSchemaType.values());
    elementReader = buildColumnReader(
        valueType,
        fileValueType,
        valueParams,
        *scanSpec.children()[1],
        /*isRoot=*/false);
  }
}

template <typename ReadLengths>
uint64_t skipMapChildren(
    uint64_t numValues,
    velox::memory::MemoryPool* memoryPool,
    velox::dwio::common::FormatData& formatData,
    velox::dwio::common::SelectiveColumnReader* keyReader,
    velox::dwio::common::SelectiveColumnReader* elementReader,
    int64_t& childTargetReadOffset,
    ReadLengths&& readLengths) {
  auto nullsBuffer = velox::AlignedBuffer::allocate<uint64_t>(
      velox::bits::nwords(numValues), memoryPool);
  auto nulls = nullsBuffer->as<uint64_t>();
  formatData.readNulls(numValues, nullptr, nullsBuffer);
  if (!keyReader && !elementReader) {
    return velox::bits::countNonNulls(
        nulls, 0, static_cast<uint32_t>(numValues));
  }
  nimble::Vector<int32_t> buffer{memoryPool, numValues};
  uint64_t childElements = 0;
  readLengths(buffer.data(), numValues, nullptr);
  for (size_t i = 0; i < numValues; ++i) {
    if (!velox::bits::isBitNull(nulls, i)) {
      childElements += buffer[i];
    }
  }
  if (keyReader) {
    keyReader->seekTo(keyReader->readOffset() + childElements, false);
  }
  if (elementReader) {
    elementReader->seekTo(elementReader->readOffset() + childElements, false);
  }
  childTargetReadOffset += childElements;
  return velox::bits::countNonNulls(nulls, 0, numValues);
}

} // namespace

MapColumnReader::MapColumnReader(
    const velox::TypePtr& requestedType,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
    NimbleParams& params,
    velox::common::ScanSpec& scanSpec)
    : SelectiveMapColumnReader{requestedType, fileType, params, scanSpec} {
  makeMapChildrenReaders(
      *fileType_,
      *requestedType_,
      params,
      *scanSpec_,
      scanSpec.extractionType(),
      scanSpec.deltaUpdate() != nullptr,
      keyReader_,
      elementReader_);
  if (keyReader_) {
    children_.push_back(keyReader_.get());
  }
  if (elementReader_) {
    children_.push_back(elementReader_.get());
  }
}

void MapColumnReader::readLengths(
    int32_t* lengths,
    int32_t numLengths,
    const uint64_t* nulls) {
  // Relies on the strong assumption about readNulls are called with exactly the
  // same row count.
  VELOX_CHECK_LE(
      numLengths * sizeof(int32_t),
      formatData().as<NimbleData>().getPreloadedValues()->capacity());
  std::memcpy(
      lengths,
      formatData().as<NimbleData>().getPreloadedValues()->as<int32_t>(),
      numLengths * sizeof(int32_t));
}

uint64_t MapColumnReader::skip(uint64_t numValues) {
  return skipMapChildren(
      numValues,
      pool_,
      *formatData_,
      keyReader_.get(),
      elementReader_.get(),
      childTargetReadOffset_,
      [this](int32_t* lengths, int32_t numLengths, const uint64_t* nulls) {
        readLengths(lengths, numLengths, nulls);
      });
}

bool MapColumnReader::estimateMaterializedSize(
    size_t& byteSize,
    size_t& rowCount) const {
  return estimateMaterializedSizeImpl(
      formatData(), children(), byteSize, rowCount);
}

MapAsStructColumnReader::MapAsStructColumnReader(
    const velox::TypePtr& requestedType,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
    NimbleParams& params,
    velox::common::ScanSpec& scanSpec)
    : SelectiveMapAsStructColumnReader{
          requestedType,
          fileType,
          params,
          scanSpec} {
  // MapAsStruct never uses extraction pushdown (asserted in base class),
  // so always create both readers.
  makeMapChildrenReaders(
      *fileType_,
      *requestedType_,
      params,
      mapScanSpec_,
      velox::common::ScanSpec::ExtractionType::kNone,
      /*hasDeltaUpdate=*/false,
      keyReader_,
      elementReader_);
  children_ = {keyReader_.get(), elementReader_.get()};
}

void MapAsStructColumnReader::readLengths(
    int32_t* lengths,
    int32_t numLengths,
    const uint64_t* nulls) {
  // Relies on the strong assumption about readNulls are called with exactly the
  // same row count.
  NIMBLE_CHECK_LE(
      numLengths * sizeof(int32_t),
      formatData().as<NimbleData>().getPreloadedValues()->capacity());
  std::memcpy(
      lengths,
      formatData().as<NimbleData>().getPreloadedValues()->as<int32_t>(),
      numLengths * sizeof(int32_t));
}

uint64_t MapAsStructColumnReader::skip(uint64_t numValues) {
  return skipMapChildren(
      numValues,
      pool_,
      *formatData_,
      keyReader_.get(),
      elementReader_.get(),
      childTargetReadOffset_,
      [this](int32_t* lengths, int32_t numLengths, const uint64_t* nulls) {
        readLengths(lengths, numLengths, nulls);
      });
}

DeduplicatedReadHelper::DeduplicatedReadHelper(
    velox::dwio::common::SelectiveRepeatedColumnReader* columnReader,
    bool deduplicatedLength)
    : columnReader_{columnReader},
      deduplicatedLength_{deduplicatedLength},
      lengthsDecoder_{
          columnReader_->formatData().as<NimbleData>().makeLengthDecoder()} {}

// Returns the total/unfiltered alphabet size in the read range.
// NOTE: since reader->next() calls can pass in arbitrary row counts, we can
// start the current read range in the middle of the last dictionary run from
// the last read. In this case, we would return the size of the alphabet that
// *needs to be read* in the current range.
// This method also gets the below state arrays ready with the same cardinality
//   * runStartRows
//   * allLengths
// NOTE: read alphabet size differs from total alphabet size only when we can
// directly copy the loaded last run from last range.
size_t DeduplicatedReadHelper::prepareDeduplicatedStates(
    int32_t numLengths,
    const uint64_t* nulls) {
  // We need the children to just read the alphabet elements.
  // So we can't wait until getValues to dedupe.

  // Read the offsets that is already decoded with the nulls.
  auto& preloadedOffsetsBuffer =
      columnReader_->formatData().as<NimbleData>().getPreloadedValues();
  // Relies on the strong assumption about readNulls are called with exactly the
  // same row count.
  VELOX_CHECK_LE(
      numLengths * sizeof(int32_t), preloadedOffsetsBuffer->capacity());
  VELOX_CHECK(preloadedOffsetsBuffer->isMutable());
  auto tmpOffsets = preloadedOffsetsBuffer->asMutable<vector_size_t>();

  // The dedupe can happen in place (so maybe the same can be done for lengths),
  // because we are always writing into ranges we already scanned.
  // A few ways to improve this
  // 1) use actual arena for reallocations, and use different allocation sizes
  //    for the pre and post dedupe buffers
  // 2) consider doing a buffered load, so the load cost is closer to the
  // buffer size, unless it's a really bad data shape for this encoding.
  size_t alphabetSize = 0;
  // Not currently worried about overflowing edge cases as that won't fit into
  // memory.
  vector_size_t currentOffset = -1;
  vector_size_t currentLength = -1;
  if (deduplicatedLength_) {
    for (int i = 0; i < numLengths; ++i) {
      if (nulls && velox::bits::isBitNull(nulls, i)) {
        continue;
      }
      if (currentOffset == -1) {
        startFromLastRun_ = tmpOffsets[i] == latestOffset_;
      } else if (tmpOffsets[i] == currentOffset) {
        continue;
      }
      currentOffset = tmpOffsets[i];
      tmpOffsets[alphabetSize++] = i;
    }
  } else {
    columnReader_->ensureAllLengthsBuffer(numLengths);
    // Nulls in the read range is the correct nulls to use for scattering, since
    // the lengths stream is not nullable.
    lengthsDecoder_->nextIndices(columnReader_->allLengths_, numLengths, nulls);
    for (int i = 0; i < numLengths; ++i) {
      if (nulls && velox::bits::isBitNull(nulls, i)) {
        continue;
      }
      if (currentOffset == -1) {
        startFromLastRun_ = tmpOffsets[i] == latestOffset_ &&
            columnReader_->allLengths_[i] == latestLength_;
      } else if (tmpOffsets[i] == currentOffset) {
        if (columnReader_->allLengths_[i] == currentLength) {
          continue;
        }
        VELOX_CHECK_EQ(
            currentLength, 0, "None empty shared prefix is not allowed");
      }
      currentOffset = tmpOffsets[i];
      currentLength = columnReader_->allLengths_[i];
      tmpOffsets[alphabetSize++] = i;
    }
  }
  copyLastRun_ = startFromLastRun_ && loadLastRun_;
  if (currentOffset < 0) {
    VELOX_CHECK_EQ(alphabetSize, 0);
    return 0;
  }
  latestOffset_ = currentOffset;
  latestLength_ = currentLength;

  // Now convert the total alphabet size to the alphabet size that needs to be
  // read for this read range. We want all state arrays to have sizes matching
  // the read alphabet size.
  alphabetSize -= copyLastRun_;
  // Include last run in the current read range if needed but not loaded
  // previously. The read alphabet size already accounts for this.
  const bool readFromLastRun = startFromLastRun_ && !loadLastRun_;

  if (!runStartRowsHolder_ ||
      runStartRowsHolder_->capacity() < alphabetSize * sizeof(vector_size_t)) {
    runStartRowsHolder_ =
        velox::allocateIndices(alphabetSize, columnReader_->pool_);
    runStartRows_ = runStartRowsHolder_->asMutable<vector_size_t>();
  }

  // We don't need the content of the offsets (which are actually the indices in
  // the file) as we will just extract the run start indices and then have to
  // reconstruct the compact indices from the result dictionary vector anyway.
  // The deduped run start rows contains the continued run (if applicable), so
  // we need to
  // * either exclude it for conformity with other state arrays
  // * or special case access to runStartRows_ with copyLastRun_ for downstream
  // access.
  std::memcpy(
      runStartRows_, tmpOffsets + copyLastRun_, alphabetSize * sizeof(int32_t));

  if (deduplicatedLength_) {
    columnReader_->ensureAllLengthsBuffer(alphabetSize);
    // If we need to read from last run, that length needs to be backfilled from
    // the last read range.
    // NOTE: if we need to read from last run, we have at least one alphabet.
    if (readFromLastRun) {
      lengthAt(0) = lastRunLength_;
    }

    // The bulk reader code is written as if we will read nulls for the alphabet
    // from lengths stream, but on the write side there is no such information.
    // We will proceed to simplify.
    lengthsDecoder_->nextIndices(
        columnReader_->allLengths_ + readFromLastRun,
        alphabetSize - readFromLastRun,
        nullptr);
  } else {
    if (readFromLastRun) {
      lengthAt(0) = lastRunLength_;
    }
    for (size_t i = readFromLastRun; i < alphabetSize; ++i) {
      lengthAt(i) = columnReader_->allLengths_[runStartRows_[i]];
    }
  }

  return alphabetSize;
}

// Compiles the list of selected dictionary indices in the current read range
// and returns the total length. Only includes dictionary runs to be read, just
// like prepareDeduplicatedStates.
vector_size_t DeduplicatedReadHelper::getSelectedAlphabetLength(
    velox::RowSet rows,
    size_t alphabetSize) {
  vector_size_t totalLength = 0;
  size_t currIndex = 0;
  selectedIndices_.clear();

  // Check if the last range run to copy is indeed selected.
  // In this case we don't need to know the exact non-null start
  // of the last range run, and just let nulls take care of correctness.
  // We miss an optimization but largely simplify the rest of the logic.
  if (!rows.empty() && alphabetSize > 0) {
    copyLastRun_ = copyLastRun_ && rows.front() < runStartRows_[0];
  }

  vector_size_t rowIndex = 0;
  for (; currIndex < alphabetSize; ++currIndex) {
    // Skip all the rows already covered by the previous index.
    // NOTE: runStartRows_[0] can be larger than the first non-null
    // row position.
    while (rowIndex < rows.size() &&
           rows[rowIndex] < runStartRows_[currIndex]) {
      ++rowIndex;
    }

    if (rowIndex >= rows.size()) {
      break;
    }

    // Non-contiguous row index can skip over multiple runs.
    if (currIndex < alphabetSize - 1 &&
        rows[rowIndex] >= runStartRows_[currIndex + 1]) {
      continue;
    }

    VELOX_DCHECK_GE(rows[rowIndex], runStartRows_[currIndex]);
    selectedIndices_.push_back(currIndex);
    totalLength += columnReader_->prunedLengthAt(currIndex);
  }

  return totalLength;
}

void DeduplicatedReadHelper::makeNestedRowSet(
    const velox::RowSet& rows,
    const uint64_t* nulls,
    int32_t maxRow) {
  auto alphabetSize = prepareDeduplicatedStates(maxRow + 1, nulls);
  auto selectedAlphabetLength = getSelectedAlphabetLength(rows, alphabetSize);
  const bool skipLastRun = !startFromLastRun_ && !loadLastRun_;
  const bool nestedRowsAllSelected =
      (columnReader_->scanSpec_->maxArrayElementsCount() ==
       std::numeric_limits<vector_size_t>::max()) &&
      velox::dwio::common::isDense(selectedIndices_) && !skipLastRun;
  vector_size_t* nestedRowsData = nullptr;
  if (nestedRowsAllSelected) {
    columnReader_->nestedRows_ = velox::RowSet(
        velox::iota(selectedAlphabetLength, columnReader_->nestedRowsHolder_),
        selectedAlphabetLength);
  } else {
    auto& nestedRowsHolder = columnReader_->nestedRowsHolder_;
    nestedRowsHolder.resize(selectedAlphabetLength);
    columnReader_->nestedRows_ = nestedRowsHolder;
    nestedRowsData = nestedRowsHolder.data();
  }
  if (alphabetSize == 0 && !copyLastRun_) {
    return;
  }

  vector_size_t nestedOffset = 0;
  vector_size_t nestedRow = 0;
  vector_size_t currIndex = 0;
  vector_size_t selectedIndex = 0;

  // if last run is not read, and we don't need to read it, we should
  // advance with last run length.
  if (skipLastRun) {
    nestedOffset += lastRunLength_;
    skippedLastRunlength_ = lastRunLength_;
  } else {
    skippedLastRunlength_ = 0;
  }

  for (; currIndex < alphabetSize && runStartRows_[currIndex] <= maxRow;
       ++currIndex) {
    if (!nestedRowsAllSelected && selectedIndex < selectedIndices_.size() &&
        currIndex == selectedIndices_[selectedIndex]) {
      ++selectedIndex;
      auto lengthAtRow = columnReader_->prunedLengthAt(currIndex);
      std::iota(
          nestedRowsData + nestedRow,
          nestedRowsData + nestedRow + lengthAtRow,
          nestedOffset);
      nestedRow += lengthAtRow;
    }
    nestedOffset += lengthAt(currIndex);
  }
  if (!nestedRowsAllSelected) {
    VELOX_CHECK_EQ(nestedRow, selectedAlphabetLength);
  }

  loadLastRun_ = true;
  lastRunLoaded_ = false;
  if (alphabetSize > 0) {
    lastRunLength_ = lengthAt(alphabetSize - 1);
    // The last run is not selected when the batch was fully filtered (empty
    // rows) or the run starts past the last selected row. The rows.empty() term
    // also guards rows.back(), which is out-of-bounds on an empty RowSet.
    const bool lastRunNotSelected =
        rows.empty() || runStartRows_[alphabetSize - 1] > rows.back();
    if (lastRunNotSelected) {
      // Leave the last run unread in child, in case we need it for the next
      // read. This is as if we skipped the last section manually. Alternatively
      // would be to always read the last run and populate the cache (but have
      // to take pruning into account).
      nestedOffset -= lastRunLength_;
      loadLastRun_ = false;
    }
  }
  columnReader_->childTargetReadOffset_ += nestedOffset;
}

// NOTE: this can actually be called in prepareRead, causing us to call
// prepareDeduplicatedStates in succession.
void DeduplicatedReadHelper::skip(uint64_t numValues, size_t alphabetSize) {
  // If read alphabet size is 0, we didn't even reach a new run, the skip is
  // no-op for the dictionary run states.
  if (alphabetSize > 0) {
    uint64_t childElements = 0;
    for (size_t i = 0; i < alphabetSize - 1; ++i) {
      childElements += lengthAt(i);
    }

    if (!startFromLastRun_ && !loadLastRun_) {
      childElements += lastRunLength_;
      VLOG(1)
          << "SEEK: Non contiguous but don't need cache. Bumping child offset by "
          << lastRunLength_;
    }
    columnReader_->childTargetReadOffset_ += childElements;
    auto lastRowLength = lengthAt(alphabetSize - 1);

    lastRunLength_ = lastRowLength;
    loadLastRun_ = false;
  }
}

// Make the alphabet according to the prepared read range. We might
// end up needing a subset of the prepared alphabet, but we won't compact
// them further for now.
//
// Note that although alphabet itself is unfiltered, its children is still
// subject to filtering due to subfield pruning or any key filter on the map.
void DeduplicatedReadHelper::makeAlphabetOffsetsAndSizes(
    velox::ArrayVectorBase& alphabet,
    vector_size_t lastRunLength) {
  const vector_size_t returnAlphabetSize =
      selectedIndices_.size() + copyLastRun_;
  alphabet.resize(returnAlphabetSize);
  auto* rawOffsets =
      alphabet.mutableOffsets(returnAlphabetSize)->asMutable<vector_size_t>();
  auto* rawSizes =
      alphabet.mutableSizes(returnAlphabetSize)->asMutable<vector_size_t>();
  if (copyLastRun_) {
    rawSizes[0] = lastRunLength;
    rawOffsets[0] = 0;
  } else {
    lastRunLength = 0;
  }
  vector_size_t currentOffset = skippedLastRunlength_;
  vector_size_t nestedRowIndex = 0;
  if (velox::dwio::common::isDense(selectedIndices_) &&
      velox::dwio::common::isDense(columnReader_->nestedRows_)) {
    for (vector_size_t i = 0; i < selectedIndices_.size(); ++i) {
      rawOffsets[i + copyLastRun_] = lastRunLength + nestedRowIndex;
      currentOffset += lengthAt(i);
      auto newNestedRowIndex = std::min<vector_size_t>(
          columnReader_->nestedRows_.size(), currentOffset);
      rawSizes[i + copyLastRun_] = newNestedRowIndex - nestedRowIndex;
      nestedRowIndex = newNestedRowIndex;
    }
  } else {
    for (vector_size_t i = 0, j = 0; i < selectedIndices_.size(); ++i) {
      rawOffsets[i + copyLastRun_] = lastRunLength + nestedRowIndex;
      while (j <= selectedIndices_[i]) {
        currentOffset += lengthAt(j++);
      }
      auto newNestedRowIndex = columnReader_->advanceNestedRows(
          columnReader_->nestedRows_, nestedRowIndex, currentOffset);
      rawSizes[i + copyLastRun_] = newNestedRowIndex - nestedRowIndex;
      nestedRowIndex = newNestedRowIndex;
    }
  }
}

void DeduplicatedReadHelper::populateSelectedIndices(
    const uint64_t* nulls,
    velox::RowSet rows,
    velox::vector_size_t* indices) {
  vector_size_t currSelectedIdx = 0;
  int i = 0;
  if (copyLastRun_) {
    // synonymous to nestedRows_.empty()
    // TODO: we can return const encoding in the future.
    auto lastRunEndRowIndex = selectedIndices_.empty()
        ? rows.back() + 1
        : runStartRows_[selectedIndices_[0]];
    while (i < rows.size() && rows[i] < lastRunEndRowIndex) {
      indices[i] = 0;
      ++i;
    }
  }

  for (; i < rows.size(); ++i) {
    if (!nulls || velox::bits::isBitSet(nulls, rows[i])) {
      // Since we might return fewer rows than we prepared, need to
      // skip to the indices for the prepared row range.
      while (currSelectedIdx + 1 < selectedIndices_.size() &&
             rows[i] >= runStartRows_[selectedIndices_[currSelectedIdx + 1]]) {
        ++currSelectedIdx;
      }
      // Use the selected indices. All alphabet entries
      // are populated for selected indices, however.
      indices[i] = currSelectedIdx + copyLastRun_;
    }
  }
}

DeduplicatedArrayColumnReader::DeduplicatedArrayColumnReader(
    const velox::TypePtr& requestedType,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
    NimbleParams& params,
    velox::common::ScanSpec& scanSpec)
    : SelectiveListColumnReader{requestedType, fileType, params, scanSpec},
      deduplicatedReadHelper_{this, true} {
  auto& childType = requestedType_->childAt(0);
  auto& childFileType = fileType_->childAt(0);
  if (scanSpec_->children().empty()) {
    scanSpec.getOrCreateChild(
        velox::common::Subfield(
            velox::common::ScanSpec::kArrayElementsFieldName));
  }
  scanSpec_->children()[0]->setProjectOut(true);

  // For kSize extraction we only need the length stream, so skip creating
  // the element reader entirely.  This avoids registering its streams and
  // reduces IO.
  if (scanSpec.extractionType() ==
      velox::common::ScanSpec::ExtractionType::kSize) {
    return;
  }

  auto childParams = params.makeChildParams(
      params.nimbleType()->asArrayWithOffsets().elements());
  child_ = buildColumnReader(
      childType,
      childFileType,
      childParams,
      *scanSpec_->children()[0],
      /*isRoot=*/false);
  children_ = {child_.get()};
}

void DeduplicatedArrayColumnReader::readLengths(
    int32_t* /* lengths */,
    int32_t /* numLengths */,
    const uint64_t* /* nulls */) {
  // Instead of copying into a buffer, we want to instead reuse the preloaded
  // values buffer for some in place dedupe.
  VELOX_UNREACHABLE(
      fmt::format(
          "node {}, should not call readLengths in class {}",
          fileType().id(),
          typeid(*this).name()));
}

void DeduplicatedArrayColumnReader::makeNestedRowSet(
    const velox::RowSet& rows,
    int32_t maxRow) {
  if (readsNullsOnly()) {
    // Nulls-only read (IS NULL): skip building the deduplicated states,
    // matching the map reader. This avoids advancing the deduplicated decoders
    // through the desynced skipNulls() seek path. Skipping is safe because the
    // selected rows are all null; getValues() emits the null output directly
    // (its readsNullsOnly() branch) without touching this unbuilt state.
    nestedRows_ = {};
    return;
  }
  deduplicatedReadHelper_.makeNestedRowSet(
      rows,
      nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr,
      maxRow);
}

uint64_t DeduplicatedArrayColumnReader::skip(uint64_t numValues) {
  VLOG(1) << "Skipping " << numValues << " values";
  auto nullsBuffer = velox::AlignedBuffer::allocate<uint64_t>(
      velox::bits::nwords(numValues), pool_);
  auto nulls = nullsBuffer->as<uint64_t>();
  formatData_->readNulls(numValues, nullptr, nullsBuffer);
  if (child_) {
    auto alphabetSize =
        deduplicatedReadHelper_.prepareDeduplicatedStates(numValues, nulls);
    deduplicatedReadHelper_.skip(numValues, alphabetSize);
    child_->seekTo(childTargetReadOffset_, false);
  }
  return velox::bits::countNonNulls(nulls, 0, numValues);
}

void DeduplicatedArrayColumnReader::read(
    int64_t offset,
    const velox::RowSet& rows,
    const uint64_t* incomingNulls) {
  if (!readsNullsOnly() && deduplicatedReadHelper_.loadLastRun_ &&
      !deduplicatedReadHelper_.lastRunLoaded_) {
    velox::VectorPtr result;
    getValues({}, &result);
  }
  SelectiveListColumnReader::read(offset, rows, incomingNulls);
}

// Returns filtered alphabet based on the read row set,
// and then wraps indices for the final filtered row set.
void DeduplicatedArrayColumnReader::getValues(
    const velox::RowSet& rows,
    velox::VectorPtr* result) {
  VELOX_DCHECK_NOT_NULL(result);

  if (scanSpec_->extractionType() ==
      velox::common::ScanSpec::ExtractionType::kSize) {
    getExtractionSizeValues(rows, result);
    return;
  }

  if (readsNullsOnly()) {
    // Nulls-only read: makeNestedRowSet left the deduplicated state unbuilt, so
    // emit a null-structured array directly instead of materializing and
    // masking a dictionary vector. The selected rows are all null (IS NULL), so
    // no element content is needed.
    *result = velox::BaseVector::create(requestedType_, rows.size(), pool_);
    setComplexNulls(rows, *result);
    return;
  }

  deduplicatedReadHelper_.lastRunLoaded_ = true;
  // TODO: looks like elements of the alphabet is not properly
  // referenced, and thus we are already in the elements of the
  // next batch in lazy vector scenario.
  auto dictionaryVector =
      prepareDictionaryArrayResult(*result, requestedType_, rows.size(), pool_);
  if (!rows.empty()) {
    setComplexNulls(rows, *result);
  }
  auto indices =
      dictionaryVector->mutableIndices(rows.size())->asMutable<vector_size_t>();
  auto* alphabetArray =
      dictionaryVector->valueVector()->asUnchecked<velox::ArrayVector>();

  deduplicatedReadHelper_.makeAlphabetOffsetsAndSizes(
      *alphabetArray, lastRunValue_ ? lastRunValue_->size() : 0);
  if (child_ && (!nestedRows_.empty() || copyLastRun())) {
    auto& elements = alphabetArray->elements();
    prepareStructResult(requestedType_->childAt(0), &elements);

    // Condensed branch for when we only need the cached run, and nothing from
    // the current read range.
    if (nestedRows_.empty()) {
      VELOX_CHECK_NOT_NULL(lastRunValue_);
      elements = velox::BaseVector::copy(*lastRunValue_);
    } else {
      if (copyLastRun()) {
        appendToLastRun(*lastRunValue_, *child_, nestedRows_, elements);
      } else {
        child_->getValues(nestedRows_, &elements);
      }
      if (deduplicatedReadHelper_.loadLastRun_) {
        VELOX_CHECK_NOT_NULL(elements);
        auto lastAlphabetLength =
            alphabetArray->sizeAt(alphabetArray->size() - 1);
        VELOX_CHECK_GE(elements->size(), lastAlphabetLength);
        loadLastRun(*elements, lastAlphabetLength, lastRunValue_);
      }
    }
  }
  if (lastRunValue_ && deduplicatedReadHelper_.lastRunLength_ == 0) {
    lastRunValue_->resize(0);
  }

  if (!rows.empty()) {
    auto* nulls =
        nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
    deduplicatedReadHelper_.populateSelectedIndices(nulls, rows, indices);
  }
}

DeduplicatedMapColumnReader::DeduplicatedMapColumnReader(
    const velox::TypePtr& requestedType,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
    NimbleParams& params,
    velox::common::ScanSpec& scanSpec)
    : SelectiveMapColumnReader{requestedType, fileType, params, scanSpec},
      deduplicatedReadHelper_{this, false} {
  auto& keyType = requestedType_->childAt(0);
  auto& valueType = requestedType_->childAt(1);
  auto& fileKeyType = fileType_->childAt(0);
  auto& fileValueType = fileType_->childAt(1);

  const auto extractionType = scanSpec.extractionType();
  const bool hasDeltaUpdate = scanSpec.deltaUpdate() != nullptr;
  auto& mapSchemaType = params.nimbleType()->asSlidingWindowMap();
  // Skip creating child readers that extraction pushdown doesn't need.
  // This avoids registering their streams and reduces IO.
  if (needsKeyReader(extractionType, hasDeltaUpdate)) {
    auto keyParams = params.makeChildParams(mapSchemaType.keys());
    keyReader_ = buildColumnReader(
        keyType,
        fileKeyType,
        keyParams,
        *scanSpec_->children()[0],
        /*isRoot=*/false);
  }
  if (needsElementReader(extractionType, hasDeltaUpdate)) {
    auto valueParams = params.makeChildParams(mapSchemaType.values());
    elementReader_ = buildColumnReader(
        valueType,
        fileValueType,
        valueParams,
        *scanSpec_->children()[1],
        /*isRoot=*/false);
  }
  if (keyReader_) {
    children_.push_back(keyReader_.get());
  }
  if (elementReader_) {
    children_.push_back(elementReader_.get());
  }
}

void DeduplicatedMapColumnReader::readLengths(
    int32_t* /* lengths */,
    int32_t /* numLengths */,
    const uint64_t* /* nulls */) {
  // Instead of copying into a buffer, we want to instead reuse the preloaded
  // values buffer for some in place dedupe.
  VELOX_UNREACHABLE(
      fmt::format(
          "node {}, should not call readLengths in class {}",
          fileType().id(),
          typeid(*this).name()));
}

void DeduplicatedMapColumnReader::makeNestedRowSet(
    const velox::RowSet& rows,
    int32_t maxRow) {
  if (readsNullsOnly()) {
    // Nulls-only read (IS NULL): skip building the deduplicated states. Their
    // construction (prepareDeduplicatedStates) decodes the per-row lengths,
    // which desync from the offsets after a skipNulls() seek and trip the
    // shared-prefix check. Skipping is safe because the selected rows are all
    // null; getValues() emits the null output directly (its readsNullsOnly()
    // branch) without touching this unbuilt state.
    nestedRows_ = {};
    return;
  }
  deduplicatedReadHelper_.makeNestedRowSet(
      rows,
      nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr,
      maxRow);
}

uint64_t DeduplicatedMapColumnReader::skip(uint64_t numValues) {
  VLOG(1) << "Skipping " << numValues << " values";
  auto nullsBuffer = velox::AlignedBuffer::allocate<uint64_t>(
      velox::bits::nwords(numValues), pool_);
  auto nulls = nullsBuffer->as<uint64_t>();
  formatData_->readNulls(numValues, nullptr, nullsBuffer);
  if (keyReader_ || elementReader_) {
    auto alphabetSize =
        deduplicatedReadHelper_.prepareDeduplicatedStates(numValues, nulls);
    deduplicatedReadHelper_.skip(numValues, alphabetSize);
    if (keyReader_) {
      keyReader_->seekTo(childTargetReadOffset_, false);
    }
    if (elementReader_) {
      elementReader_->seekTo(childTargetReadOffset_, false);
    }
  }
  return velox::bits::countNonNulls(nulls, 0, numValues);
}

void DeduplicatedMapColumnReader::read(
    int64_t offset,
    const velox::RowSet& rows,
    const uint64_t* incomingNulls) {
  if (!readsNullsOnly() && deduplicatedReadHelper_.loadLastRun_ &&
      !deduplicatedReadHelper_.lastRunLoaded_) {
    velox::VectorPtr result;
    getValues({}, &result);
  }
  SelectiveMapColumnReader::read(offset, rows, incomingNulls);
}

// Returns filtered alphabet based on the read row set,
// and then wraps indices for the final filtered row set.
void DeduplicatedMapColumnReader::getValues(
    const velox::RowSet& rows,
    velox::VectorPtr* result) {
  VELOX_DCHECK_NOT_NULL(result);
  const auto extractionType = scanSpec_->extractionType();

  if (extractionType != velox::common::ScanSpec::ExtractionType::kNone) {
    getExtractionValues(rows, result);
    return;
  }

  if (readsNullsOnly()) {
    // Nulls-only read: makeNestedRowSet left the deduplicated state unbuilt, so
    // emit a null-structured map directly instead of materializing and masking
    // a dictionary vector. The selected rows are all null (IS NULL), so no
    // element content is needed.
    *result = velox::BaseVector::create(requestedType_, rows.size(), pool_);
    setComplexNulls(rows, *result);
    return;
  }

  deduplicatedReadHelper_.lastRunLoaded_ = true;
  // TODO: looks like elements of the alphabet is not properly
  // referenced, and thus we are already in the elements of the
  // next batch in lazy vector scenario.
  auto dictionaryVector =
      prepareDictionaryMapResult(*result, requestedType_, rows.size(), pool_);
  if (!rows.empty()) {
    setComplexNulls(rows, *result);
  }
  auto indices =
      dictionaryVector->mutableIndices(rows.size())->asMutable<vector_size_t>();
  auto* alphabetMap =
      dictionaryVector->valueVector()->asUnchecked<velox::MapVector>();

  deduplicatedReadHelper_.makeAlphabetOffsetsAndSizes(
      *alphabetMap, lastRunKeys_ ? lastRunKeys_->size() : 0);
  if ((keyReader_ || elementReader_) &&
      (!nestedRows_.empty() || copyLastRun())) {
    auto& keys = alphabetMap->mapKeys();
    auto& elements = alphabetMap->mapValues();
    prepareStructResult(requestedType_->childAt(0), &keys);
    prepareStructResult(requestedType_->childAt(1), &elements);

    // Condensed branch for when we only need the cached run, and nothing from
    // the current read range.
    if (nestedRows_.empty()) {
      VELOX_CHECK_NOT_NULL(lastRunKeys_);
      keys = velox::BaseVector::copy(*lastRunKeys_);
      elements = velox::BaseVector::copy(*lastRunValues_);
    } else {
      if (copyLastRun()) {
        appendToLastRun(*lastRunKeys_, *keyReader_, nestedRows_, keys);
        appendToLastRun(
            *lastRunValues_, *elementReader_, nestedRows_, elements);
      } else {
        keyReader_->getValues(nestedRows_, &keys);
        elementReader_->getValues(nestedRows_, &elements);
      }
      if (deduplicatedReadHelper_.loadLastRun_) {
        VELOX_CHECK_NOT_NULL(keys);
        VELOX_CHECK_NOT_NULL(elements);
        auto lastAlphabetLength = alphabetMap->sizeAt(alphabetMap->size() - 1);
        VELOX_CHECK_GE(keys->size(), lastAlphabetLength);
        VELOX_CHECK_GE(elements->size(), lastAlphabetLength);
        loadLastRun(*keys, lastAlphabetLength, lastRunKeys_);
        loadLastRun(*elements, lastAlphabetLength, lastRunValues_);
      }
    }
  }
  if (lastRunKeys_ && deduplicatedReadHelper_.lastRunLength_ == 0) {
    lastRunKeys_->resize(0);
    lastRunValues_->resize(0);
  }

  if (!rows.empty()) {
    auto* nulls =
        nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
    deduplicatedReadHelper_.populateSelectedIndices(nulls, rows, indices);
  }
}
} // namespace facebook::nimble
