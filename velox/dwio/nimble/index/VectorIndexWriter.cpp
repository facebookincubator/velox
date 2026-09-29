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

#include "velox/dwio/nimble/index/VectorIndexWriter.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <span>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <flatbuffers/flatbuffers.h>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/VectorIndexUtility.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/VectorIndexGenerated.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::nimble::index {

namespace {

// Bounds graph fanout to prevent pathological memory and build costs.
constexpr uint32_t kMaxHnswNumConnections{512};

constexpr int kHnswConstructionSearchDepth{200};

constexpr uint8_t kMaxPqBits{24};

// Clamps the requested partition count to the available training data.
uint32_t calculateNumPartitions(
    uint64_t numVectors,
    uint32_t requestedNumPartitions) {
  const auto desiredNumPartitions = requestedNumPartitions > 0
      ? static_cast<uint64_t>(requestedNumPartitions)
      : std::max<uint64_t>(
            1,
            static_cast<uint64_t>(std::sqrt(static_cast<double>(numVectors))));
  const auto minVectorsPerPartition = static_cast<uint64_t>(
      faiss::ClusteringParameters{}.min_points_per_centroid);
  const auto maximumNumPartitions =
      std::max<uint64_t>(1, numVectors / minVectorsPerPartition);
  return static_cast<uint32_t>(
      std::min(desiredNumPartitions, maximumNumPartitions));
}

// Collects FAISS serialization output in a contiguous pool-tracked buffer.
struct SerializedIndexWriter : public faiss::IOWriter {
  SerializedIndexWriter(
      velox::memory::MemoryPool* pool,
      size_t maxSerializedSize)
      : pool{pool}, maxSerializedSize{maxSerializedSize} {
    NIMBLE_CHECK_NOT_NULL(pool);
    NIMBLE_CHECK_GT(maxSerializedSize, 0);
  }

  // Appends one FAISS write and returns the number of items consumed.
  size_t operator()(const void* source, size_t itemSize, size_t numItems)
      override {
    // FAISS emits zero-item writes for absent optional data such as direct
    // maps.
    if (numItems == 0) {
      return 0;
    }
    NIMBLE_CHECK_GT(itemSize, 0, "FAISS write item size must be positive");
    NIMBLE_CHECK_NOT_NULL(source, "FAISS write buffer must not be null");
    NIMBLE_CHECK_LE(
        numItems,
        std::numeric_limits<size_t>::max() / itemSize,
        "FAISS write size exceeds the platform limit");
    const auto numBytes = itemSize * numItems;
    NIMBLE_USER_CHECK_LE(
        numBytes,
        maxSerializedSize - serializedSize,
        "Serialized vector index exceeds the configured limit");
    const auto requiredSize = serializedSize + numBytes;
    ensureCapacity(requiredSize);
    serializedData->setSize(requiredSize);
    std::memcpy(
        serializedData->asMutable<char>() + serializedSize, source, numBytes);
    serializedSize = requiredSize;
    return numItems;
  }

  // Grows storage geometrically without exceeding the configured limit.
  void ensureCapacity(size_t requiredSize) {
    const auto currentCapacity =
        serializedData == nullptr ? 0 : serializedData->capacity();
    if (currentCapacity >= requiredSize) {
      return;
    }
    const auto doubledCapacity = currentCapacity > maxSerializedSize / 2
        ? maxSerializedSize
        : currentCapacity * 2;
    const auto newCapacity = std::max(requiredSize, doubledCapacity);
    if (serializedData == nullptr) {
      serializedData = velox::AlignedBuffer::allocate<char>(newCapacity, pool);
    } else {
      velox::AlignedBuffer::reallocate<char>(&serializedData, newCapacity);
    }
  }

  // Accounts serialized bytes against the writer pool.
  velox::memory::MemoryPool* const pool;

  // Prevents serialization from exceeding the caller's configured bound.
  const size_t maxSerializedSize;

  // Owns all bytes emitted by faiss::write_index().
  velox::BufferPtr serializedData;

  // Tracks initialized bytes separately from allocated capacity.
  size_t serializedSize{0};
};

// Constructs the configured index and transfers IVF quantizer ownership.
std::unique_ptr<faiss::Index> createFaissIndex(
    const VectorIndexConfig& config,
    uint32_t numPartitions) {
  const auto dimensions = static_cast<int>(config.dimensions);
  const auto faissMetric = toFaissMetric(config.metric);

  if (config.indexType == VectorIndexType::kHnswSq8) {
    auto index = std::make_unique<faiss::IndexHNSWSQ>(
        dimensions,
        faiss::ScalarQuantizer::QT_8bit,
        static_cast<int>(config.hnswNumConnections),
        faissMetric);
    index->hnsw.efConstruction = kHnswConstructionSearchDepth;
    return index;
  }

  auto quantizer = std::make_unique<faiss::IndexFlat>(dimensions, faissMetric);
  std::unique_ptr<faiss::IndexIVF> index;
  switch (config.indexType) {
    case VectorIndexType::kIvfFlat:
      index = std::make_unique<faiss::IndexIVFFlat>(
          quantizer.get(), dimensions, numPartitions, faissMetric);
      break;
    case VectorIndexType::kIvfSq8:
      index = std::make_unique<faiss::IndexIVFScalarQuantizer>(
          quantizer.get(),
          dimensions,
          numPartitions,
          faiss::ScalarQuantizer::QT_8bit,
          faissMetric);
      break;
    case VectorIndexType::kIvfPq:
      index = std::make_unique<faiss::IndexIVFPQ>(
          quantizer.get(),
          dimensions,
          numPartitions,
          static_cast<size_t>(config.pqSubQuantizers),
          static_cast<size_t>(config.pqBits),
          faissMetric);
      break;
    case VectorIndexType::kIvfRaBitQ:
      index = std::make_unique<faiss::IndexIVFRaBitQ>(
          quantizer.get(), dimensions, numPartitions, faissMetric);
      break;
    case VectorIndexType::kHnswSq8:
      NIMBLE_UNREACHABLE("HNSW index must be created before IVF dispatch");
    default:
      NIMBLE_UNREACHABLE(
          "Unsupported vector index type: {}",
          static_cast<int>(config.indexType));
  }

  NIMBLE_CHECK_NOT_NULL(
      index.get(), "IVF index construction produced no index");
  index->own_fields = true;
  (void)quantizer.release();
  index->cp.spherical = config.metric == VectorDistanceMetric::kCosine;
  return index;
}

// Validates one index configuration and resolves its input column.
velox::column_index_t validateConfig(
    const VectorIndexConfig& config,
    const velox::RowTypePtr& inputType) {
  NIMBLE_USER_CHECK(
      !config.columnName.empty(), "Vector index column name must not be empty");
  NIMBLE_USER_CHECK_GT(
      config.dimensions, 0, "Vector dimensions must be positive");
  NIMBLE_USER_CHECK_LE(
      config.dimensions,
      static_cast<uint32_t>(std::numeric_limits<int>::max()),
      "Vector dimensions exceed the FAISS limit");
  NIMBLE_USER_CHECK_LE(
      config.numPartitions,
      static_cast<uint32_t>(std::numeric_limits<int>::max()),
      "Number of partitions exceeds the FAISS limit");
  NIMBLE_USER_CHECK_GT(
      config.maxBufferedVectorSizeBytes,
      0,
      "Vector buffer size limit must be positive");
  NIMBLE_USER_CHECK_LE(
      config.maxBufferedVectorSizeBytes,
      kMaxVectorIndexSizeBytes,
      "Vector buffer size limit exceeds the supported maximum");
  NIMBLE_USER_CHECK_GT(
      config.maxIndexSizeBytes, 0, "Vector index size limit must be positive");
  NIMBLE_USER_CHECK_LE(
      config.maxIndexSizeBytes,
      kMaxVectorIndexSizeBytes,
      "Vector index size limit exceeds the supported maximum");

  if (config.indexType == VectorIndexType::kIvfPq) {
    NIMBLE_USER_CHECK_GT(
        config.pqSubQuantizers,
        0,
        "Number of PQ sub-quantizers must be positive");
    NIMBLE_USER_CHECK_EQ(
        config.dimensions % config.pqSubQuantizers,
        0,
        "Vector dimensions must be divisible by the number of PQ "
        "sub-quantizers");
    NIMBLE_USER_CHECK_GE(
        config.pqBits, 1, "PQ bits per code must be at least one");
    NIMBLE_USER_CHECK_LE(
        config.pqBits, kMaxPqBits, "PQ bits per code exceed the FAISS limit");
  }

  if (config.indexType == VectorIndexType::kHnswSq8) {
    NIMBLE_USER_CHECK_GT(
        config.hnswNumConnections,
        0,
        "Number of HNSW connections must be positive");
    NIMBLE_USER_CHECK_LE(
        config.hnswNumConnections,
        kMaxHnswNumConnections,
        "Number of HNSW connections is too large");
  }

  const auto columnIndex = inputType->getChildIdx(config.columnName);
  const auto& columnType = inputType->childAt(columnIndex);
  NIMBLE_USER_CHECK_EQ(
      static_cast<int>(columnType->kind()),
      static_cast<int>(velox::TypeKind::ARRAY),
      "Vector index column must be an array: {}",
      columnType->toString());
  NIMBLE_USER_CHECK_EQ(
      static_cast<int>(columnType->childAt(0)->kind()),
      static_cast<int>(velox::TypeKind::REAL),
      "Vector index elements must be REAL: {}",
      columnType->toString());
  return columnIndex;
}

// Rejects top-level nulls before any column accumulator is mutated.
void validateTopLevelRows(const velox::RowVector& input, uint64_t fileRow) {
  if (!input.mayHaveNulls()) {
    return;
  }
  for (velox::vector_size_t row = 0; row < input.size(); ++row) {
    NIMBLE_USER_CHECK(
        !input.isNullAt(row),
        "Input contains a null top-level row: {}",
        fileRow + static_cast<uint64_t>(row));
  }
}

} // namespace

std::unique_ptr<VectorIndexWriter> VectorIndexWriter::create(
    std::span<const VectorIndexConfig> configs,
    const velox::RowTypePtr& inputType,
    velox::memory::MemoryPool* pool) {
  NIMBLE_USER_CHECK(!configs.empty(), "Vector index configs must not be empty");
  NIMBLE_USER_CHECK_NOT_NULL(inputType, "Input type must not be null");
  NIMBLE_USER_CHECK_NOT_NULL(pool, "Memory pool must not be null");

  std::set<std::string> columnNames;
  std::vector<Accumulator> accumulators;
  accumulators.reserve(configs.size());
  for (const auto& config : configs) {
    NIMBLE_USER_CHECK(
        columnNames.emplace(config.columnName).second,
        "Duplicate vector index column: {}",
        config.columnName);
    accumulators.emplace_back(
        Accumulator{
            .config = config,
            .columnIndex = validateConfig(config, inputType),
            .vectors = nullptr,
            .numVectors = 0,
        });
  }

  return std::unique_ptr<VectorIndexWriter>(
      new VectorIndexWriter(std::move(accumulators), pool));
}

VectorIndexWriter::VectorIndexWriter(
    std::vector<Accumulator> accumulators,
    velox::memory::MemoryPool* pool)
    : pool_{pool}, accumulators_{std::move(accumulators)} {
  NIMBLE_CHECK_NOT_NULL(pool_, "Memory pool must not be null");
  NIMBLE_CHECK(
      !accumulators_.empty(), "Vector index configs must not be empty");
}

VectorIndexWriter::~VectorIndexWriter() = default;

namespace {

// Returns the decoded array base after checked type conversion.
const velox::ArrayVector& decodedArrayVector(
    const velox::DecodedVector& decodedArrays) {
  const auto* baseVector = decodedArrays.base();
  if (baseVector == nullptr) {
    NIMBLE_FAIL("Decoded vector base must not be null");
  }
  return *baseVector->asChecked<velox::ArrayVector>();
}

// Keeps decoded array and element mappings alive across validation and append.
struct DecodedVectorColumn {
  explicit DecodedVectorColumn(const velox::VectorPtr& column)
      : decodedArrays{*column},
        arrayVector{decodedArrayVector(decodedArrays)},
        decodedElements{*arrayVector.elements()} {}

  velox::DecodedVector decodedArrays;
  const velox::ArrayVector& arrayVector;
  velox::DecodedVector decodedElements;
};

// Validates one decoded vector column and returns its new value count.
size_t validateVectors(
    velox::vector_size_t numRows,
    uint64_t fileRowOffset,
    const VectorIndexConfig& config,
    const DecodedVectorColumn& input) {
  for (velox::vector_size_t rowOffset = 0; rowOffset < numRows; ++rowOffset) {
    const auto row = fileRowOffset + static_cast<uint64_t>(rowOffset);
    if (input.decodedArrays.mayHaveNulls()) {
      NIMBLE_USER_CHECK(
          !input.decodedArrays.isNullAt(rowOffset),
          "Vector index column contains a null row: {}",
          row);
    }
    const auto arrayRow = input.decodedArrays.index(rowOffset);
    const auto offset = input.arrayVector.offsetAt(arrayRow);
    const auto size = input.arrayVector.sizeAt(arrayRow);
    NIMBLE_USER_CHECK_EQ(
        static_cast<uint32_t>(size),
        config.dimensions,
        "Vector dimension mismatch at row: {}",
        row);
    if (input.decodedElements.mayHaveNulls()) {
      for (velox::vector_size_t i = 0; i < size; ++i) {
        NIMBLE_USER_CHECK(
            !input.decodedElements.isNullAt(offset + i),
            "Vector index column contains a null element at row: {}",
            row);
      }
    }
  }

  const auto numVectors = fileRowOffset + static_cast<uint64_t>(numRows);
  NIMBLE_USER_CHECK_LE(
      numVectors,
      static_cast<uint64_t>(std::numeric_limits<faiss::idx_t>::max()),
      "Number of vectors exceeds the FAISS limit");
  NIMBLE_USER_CHECK_LE(
      numVectors,
      std::numeric_limits<size_t>::max() / config.dimensions,
      "Vector buffer size exceeds the platform limit");
  return static_cast<size_t>(numVectors * config.dimensions);
}

// Appends one validated vector column to its pool-tracked buffer.
void appendVectors(
    velox::vector_size_t numRows,
    uint64_t fileRowOffset,
    const VectorIndexConfig& config,
    const DecodedVectorColumn& input,
    velox::Buffer& output) {
  auto* outputData = output.asMutable<float>();
  NIMBLE_CHECK_NOT_NULL(outputData);
  const std::span<float> outputValues{
      outputData, output.size() / sizeof(float)};
  const auto outputStart =
      static_cast<size_t>(fileRowOffset) * config.dimensions;
  for (velox::vector_size_t rowOffset = 0; rowOffset < numRows; ++rowOffset) {
    const auto arrayRow = input.decodedArrays.index(rowOffset);
    const auto offset = input.arrayVector.offsetAt(arrayRow);
    const auto outputOffset =
        outputStart + static_cast<size_t>(rowOffset) * config.dimensions;
    const auto outputVector =
        outputValues.subspan(outputOffset, config.dimensions);
    if (input.decodedElements.isIdentityMapping()) {
      const auto* contiguousInputData = input.decodedElements.data<float>();
      NIMBLE_CHECK_NOT_NULL(contiguousInputData);
      const std::span<const float> inputValues{
          contiguousInputData,
          static_cast<size_t>(input.arrayVector.elements()->size())};
      std::copy_n(
          inputValues.begin() + offset,
          config.dimensions,
          outputVector.begin());
      continue;
    }
    // Dictionary and constant encodings require decoded value access.
    for (velox::vector_size_t i = 0;
         i < static_cast<velox::vector_size_t>(config.dimensions);
         ++i) {
      outputVector[i] = input.decodedElements.valueAt<float>(offset + i);
    }
  }
}

} // namespace

void VectorIndexWriter::write(const velox::VectorPtr& input) {
  NIMBLE_CHECK(!closed_, "VectorIndexWriter has been closed");
  NIMBLE_USER_CHECK_NOT_NULL(input, "Input vector must not be null");
  if (input->size() == 0) {
    return;
  }

  const auto* rowVector = input->asChecked<velox::RowVector>();
  const auto fileRowOffset = accumulators_.front().numVectors;
  validateTopLevelRows(*rowVector, fileRowOffset);

  std::vector<std::unique_ptr<DecodedVectorColumn>> decodedColumns;
  decodedColumns.reserve(accumulators_.size());
  std::vector<size_t> numValues;
  numValues.reserve(accumulators_.size());
  for (const auto& accumulator : accumulators_) {
    auto decodedColumn = std::make_unique<DecodedVectorColumn>(
        rowVector->childAt(accumulator.columnIndex));
    numValues.emplace_back(validateVectors(
        rowVector->size(),
        accumulator.numVectors,
        accumulator.config,
        *decodedColumn));
    decodedColumns.emplace_back(std::move(decodedColumn));
  }
  for (size_t i = 0; i < accumulators_.size(); ++i) {
    ensureVectorCapacity(accumulators_.at(i), numValues.at(i));
  }
  for (size_t i = 0; i < accumulators_.size(); ++i) {
    auto& accumulator = accumulators_.at(i);
    NIMBLE_CHECK_NOT_NULL(accumulator.vectors.get());
    appendVectors(
        rowVector->size(),
        accumulator.numVectors,
        accumulator.config,
        *decodedColumns.at(i),
        *accumulator.vectors);
    accumulator.numVectors += static_cast<uint64_t>(rowVector->size());
  }
}

void VectorIndexWriter::ensureVectorCapacity(
    Accumulator& accumulator,
    size_t numValues) const {
  const auto maxValues = static_cast<size_t>(
      accumulator.config.maxBufferedVectorSizeBytes / sizeof(float));
  NIMBLE_USER_CHECK_LE(
      numValues,
      maxValues,
      "Buffered vector data exceeds the configured limit for column: {}",
      accumulator.config.columnName);

  const auto currentCapacity = accumulator.vectors == nullptr
      ? 0
      : accumulator.vectors->capacity() / sizeof(float);
  if (currentCapacity < numValues) {
    const auto doubledCapacity =
        currentCapacity > maxValues / 2 ? maxValues : currentCapacity * 2;
    const auto newCapacity = std::max(numValues, doubledCapacity);
    if (accumulator.vectors == nullptr) {
      accumulator.vectors =
          velox::AlignedBuffer::allocate<float>(newCapacity, pool_);
    } else {
      velox::AlignedBuffer::reallocate<float>(
          &accumulator.vectors, newCapacity);
    }
  }
  NIMBLE_CHECK_NOT_NULL(accumulator.vectors.get());
  accumulator.vectors->setSize(numValues * sizeof(float));
}

VectorIndexWriter::SerializedIndex VectorIndexWriter::buildIndex(
    Accumulator& accumulator) const {
  const auto& config = accumulator.config;
  NIMBLE_CHECK_GT(accumulator.numVectors, 0, "No vectors to index");
  NIMBLE_USER_CHECK_NOT_NULL(
      accumulator.vectors.get(), "Vector data must be buffered before close");

  auto* vectorData = accumulator.vectors->asMutable<float>();
  NIMBLE_CHECK_NOT_NULL(vectorData);
  normalizeVectors(
      config.metric, accumulator.numVectors, config.dimensions, vectorData);

  const auto numPartitions = config.indexType == VectorIndexType::kHnswSq8
      ? 0
      : calculateNumPartitions(accumulator.numVectors, config.numPartitions);
  auto ownedIndex = createFaissIndex(config, numPartitions);
  NIMBLE_CHECK_NOT_NULL(ownedIndex.get());
  // @lint-ignore NULLSAFECLANG nullable-dereference
  auto& faissIndex = *ownedIndex;

  const auto faissNumVectors =
      static_cast<faiss::idx_t>(accumulator.numVectors);
  // Some FAISS implementations are initialized as trained, while others must
  // learn their quantization state from the input vectors.
  if (!faissIndex.is_trained) {
    faissIndex.train(faissNumVectors, vectorData);
  }
  faissIndex.add(faissNumVectors, vectorData);

  SerializedIndexWriter writer{
      pool_, static_cast<size_t>(config.maxIndexSizeBytes)};
  faiss::write_index(&faissIndex, &writer);
  return SerializedIndex{
      .serializedData = std::move(writer.serializedData),
      .serializedSize = writer.serializedSize,
      .numPartitions = numPartitions,
  };
}

std::string VectorIndexWriter::serializeDirectory(
    std::span<const PersistedIndex> indexes) const {
  NIMBLE_CHECK(!indexes.empty(), "Persisted vector indexes must not be empty");
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<serialization::VectorIndex>> descriptors;
  descriptors.reserve(indexes.size());
  for (const auto& index : indexes) {
    const auto& accumulator = accumulators_[index.accumulatorIndex];
    const auto& config = accumulator.config;
    const auto configOffset = serialization::CreateVectorIndexMeta(
        builder,
        builder.CreateString(config.columnName),
        config.dimensions,
        toSerializedMetric(config.metric),
        toSerializedIndexType(config.indexType),
        index.numPartitions,
        accumulator.numVectors);
    const auto sectionOffset = serialization::CreateMetadataSection(
        builder,
        index.indexSection.offset(),
        index.indexSection.size(),
        static_cast<serialization::CompressionType>(
            index.indexSection.compressionType()),
        index.indexSection.uncompressedSize().value());
    descriptors.emplace_back(
        serialization::CreateVectorIndex(builder, configOffset, sectionOffset));
  }

  builder.Finish(
      serialization::CreateVectorIndexDirectory(
          builder, builder.CreateVector(descriptors)));

  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize(),
  };
}

void VectorIndexWriter::close(
    const CreateMetadataSectionFn& createMetadataFn,
    const WriteOptionalSectionFn& writeMetadataFn) {
  NIMBLE_CHECK(!closed_, "close() already called");
  closed_ = true;

  if (accumulators_.front().numVectors == 0) {
    for (const auto& accumulator : accumulators_) {
      NIMBLE_CHECK_EQ(accumulator.numVectors, 0);
    }
    return;
  }

  std::vector<PersistedIndex> persistedIndexes;
  persistedIndexes.reserve(accumulators_.size());
  for (size_t i = 0; i < accumulators_.size(); ++i) {
    auto& accumulator = accumulators_[i];
    NIMBLE_CHECK_GT(accumulator.numVectors, 0);
    auto serializedIndex = buildIndex(accumulator);
    accumulator.vectors.reset();

    NIMBLE_CHECK_NOT_NULL(serializedIndex.serializedData.get());
    NIMBLE_USER_CHECK_LE(
        serializedIndex.serializedSize,
        accumulator.config.maxIndexSizeBytes,
        "Serialized vector index exceeds the configured limit for column: {}",
        accumulator.config.columnName);

    const std::string_view serializedData{
        serializedIndex.serializedData->as<char>(),
        serializedIndex.serializedSize};
    auto indexSection = createMetadataFn(serializedData);
    NIMBLE_CHECK_GT(
        indexSection.size(), 0, "Persisted vector index must not be empty");
    NIMBLE_CHECK(
        indexSection.uncompressedSize().has_value(),
        "Persisted vector index must record its uncompressed size");
    NIMBLE_CHECK_EQ(
        indexSection.uncompressedSize().value(),
        serializedIndex.serializedSize,
        "Persisted vector index uncompressed size is incorrect");

    persistedIndexes.emplace_back(
        PersistedIndex{
            .accumulatorIndex = i,
            .numPartitions = serializedIndex.numPartitions,
            .indexSection = std::move(indexSection),
        });
  }

  NIMBLE_CHECK_EQ(persistedIndexes.size(), accumulators_.size());
  writeMetadataFn(
      std::string(kVectorIndexSection), serializeDirectory(persistedIndexes));
}

} // namespace facebook::nimble::index
