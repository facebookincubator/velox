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

#include "velox/dwio/nimble/index/VectorIndex.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <variant>

#include <faiss/Index.h>
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
#include <folly/Synchronized.h>
#include <folly/synchronization/CallOnce.h>

#include "velox/common/io/Options.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/VectorIndexUtility.h"
#include "velox/dwio/nimble/tablet/MetadataInput.h"
#include "velox/dwio/nimble/tablet/VectorIndexGenerated.h"

namespace facebook::nimble::index {

namespace {

// Adapts an in-memory metadata section to the FAISS streaming reader API.
struct VectorIndexReader : public faiss::IOReader {
  // Points into a MetadataBuffer retained by the caller during deserialization.
  const uint8_t* const serializedData;

  // Bounds all reads requested by FAISS.
  const size_t dataBytes;

  // Tracks the next unread byte.
  size_t position{0};

  VectorIndexReader(const uint8_t* serializedData, size_t dataBytes)
      : serializedData{serializedData}, dataBytes{dataBytes} {
    NIMBLE_CHECK_NOT_NULL(serializedData);
    NIMBLE_CHECK_GT(dataBytes, 0);
  }

  size_t operator()(void* destination, size_t itemSize, size_t numItems)
      override {
    // FAISS reads zero items for absent optional data such as direct maps.
    if (numItems == 0) {
      return 0;
    }

    NIMBLE_CHECK_GT(itemSize, 0, "FAISS read item size must be positive");
    NIMBLE_CHECK_NOT_NULL(destination);
    NIMBLE_CHECK_LE(position, dataBytes);
    const auto numItemsToRead =
        std::min(numItems, (dataBytes - position) / itemSize);
    const auto numBytesToRead = numItemsToRead * itemSize;
    if (numBytesToRead == 0) {
      // Returning zero reports EOF so FAISS rejects a truncated index.
      return 0;
    }
    std::memcpy(destination, serializedData + position, numBytesToRead);
    position += numBytesToRead;
    return numItemsToRead;
  }
};

// Deserializes one FAISS index while its metadata buffer remains alive.
std::unique_ptr<faiss::Index> readFaissIndex(std::string_view serializedIndex) {
  NIMBLE_CHECK_FILE(
      !serializedIndex.empty(), "FAISS index data must not be empty");
  VectorIndexReader reader(
      reinterpret_cast<const uint8_t*>(serializedIndex.data()),
      serializedIndex.size());
  try {
    return std::unique_ptr<faiss::Index>(faiss::read_index(&reader));
  } catch (const std::exception& error) {
    NIMBLE_FILE_FAIL("Failed to deserialize FAISS index: {}", error.what());
  }
}

// Verifies that the serialized implementation agrees with Nimble metadata.
bool checkIndexType(const faiss::Index& index, VectorIndexType indexType) {
  switch (indexType) {
    case VectorIndexType::kIvfFlat:
      return dynamic_cast<const faiss::IndexIVFFlat*>(&index) != nullptr;
    case VectorIndexType::kIvfSq8:
      return dynamic_cast<const faiss::IndexIVFScalarQuantizer*>(&index) !=
          nullptr;
    case VectorIndexType::kIvfPq:
      return dynamic_cast<const faiss::IndexIVFPQ*>(&index) != nullptr;
    case VectorIndexType::kIvfRaBitQ:
      return dynamic_cast<const faiss::IndexIVFRaBitQ*>(&index) != nullptr;
    case VectorIndexType::kHnswSq8:
      return dynamic_cast<const faiss::IndexHNSWSQ*>(&index) != nullptr;
    default:
      NIMBLE_UNREACHABLE(
          "Unknown vector index type: {}", static_cast<int>(indexType));
  }
}

// Copies a query and normalizes it when FAISS uses inner product for cosine.
std::vector<float> prepareQueryVector(
    const std::vector<float>& queryVector,
    VectorDistanceMetric metric) {
  auto preparedQuery = queryVector;
  normalizeVectors(
      metric,
      /*numVectors=*/1,
      static_cast<uint32_t>(preparedQuery.size()),
      preparedQuery.data());
  return preparedQuery;
}

// Serializes HNSW searches because FAISS updates process-global statistics.
folly::Synchronized<std::monostate>& hnswSearchGuard() {
  static auto* guard = new folly::Synchronized<std::monostate>();
  return *guard;
}

// VectorIndex::search accepts one query vector per call.
constexpr faiss::idx_t kNumQueriesPerSearch{1};

// Runs one search without mutating shared request parameters or IVF statistics.
void searchFaissIndex(
    const faiss::Index& index,
    VectorIndexType indexType,
    const VectorIndex::SearchConfig& config,
    const float* queryVector,
    faiss::idx_t maxNumNeighbors,
    float* scores,
    faiss::idx_t* labels) {
  if (indexType == VectorIndexType::kHnswSq8) {
    NIMBLE_CHECK_NOT_NULL(
        dynamic_cast<const faiss::IndexHNSW*>(&index),
        "FAISS index metadata requires an HNSW index");
    NIMBLE_USER_CHECK_GT(
        config.hnswSearchDepth, 0, "HNSW search depth must be positive");
    NIMBLE_USER_CHECK_LE(
        config.hnswSearchDepth,
        static_cast<uint32_t>(std::numeric_limits<int>::max()),
        "HNSW search depth exceeds the FAISS limit");
    faiss::SearchParametersHNSW searchParameters;
    searchParameters.efSearch = static_cast<int>(config.hnswSearchDepth);
    // FAISS updates process-global HNSW statistics during search.
    const auto hnswSearchLock = hnswSearchGuard().wlock();
    index.search(
        kNumQueriesPerSearch,
        queryVector,
        maxNumNeighbors,
        scores,
        labels,
        &searchParameters);
    return;
  }

  const auto* ivfIndex = dynamic_cast<const faiss::IndexIVF*>(&index);
  NIMBLE_CHECK_NOT_NULL(ivfIndex, "FAISS index metadata requires an IVF index");
  NIMBLE_CHECK_NOT_NULL(ivfIndex->quantizer);
  NIMBLE_CHECK_NOT_NULL(
      dynamic_cast<const faiss::IndexFlat*>(ivfIndex->quantizer),
      "FAISS IVF index requires a flat quantizer");
  NIMBLE_USER_CHECK_GT(
      config.numProbes, 0, "Number of probed partitions must be positive");
  faiss::SearchParametersIVF searchParameters;
  const auto numProbes = static_cast<faiss::idx_t>(
      std::min(ivfIndex->nlist, static_cast<size_t>(config.numProbes)));
  NIMBLE_CHECK_GT(numProbes, 0);
  searchParameters.nprobe = static_cast<size_t>(numProbes);
  std::vector<float> centroidScores(static_cast<size_t>(numProbes));
  std::vector<faiss::idx_t> partitionLabels(static_cast<size_t>(numProbes));

  // IVF search first finds the nearest coarse partitions, then scans vectors
  // assigned to those partitions. Keeping the stages explicit lets the second
  // pass use request-local statistics instead of FAISS's process-global state.
  ivfIndex->quantizer->search(
      kNumQueriesPerSearch,
      queryVector,
      numProbes,
      centroidScores.data(),
      partitionLabels.data(),
      searchParameters.quantizer_params);

  // Pass request-local statistics to avoid FAISS's process-global
  // indexIVF_stats, which is not safe for concurrent searches.
  //
  // TODO: Aggregate per-search FAISS statistics in VectorIndex for
  // observability.
  faiss::IndexIVFStats searchStats;
  ivfIndex->search_preassigned(
      kNumQueriesPerSearch,
      queryVector,
      maxNumNeighbors,
      partitionLabels.data(),
      centroidScores.data(),
      scores,
      labels,
      /*store_pairs=*/false,
      &searchParameters,
      &searchStats);
}

// Validates metadata before allocating or deserializing a FAISS index.
void validateVectorIndexMetadata(const VectorIndex::Metadata& metadata) {
  NIMBLE_CHECK_FILE(
      !metadata.columnName.empty(),
      "VectorIndex column name must not be empty");
  NIMBLE_CHECK_FILE_GT(
      metadata.dimensions, 0, "VectorIndex dimensions must be positive");
  NIMBLE_CHECK_FILE_LE(
      metadata.dimensions,
      static_cast<uint32_t>(std::numeric_limits<int>::max()),
      "VectorIndex dimensions exceed the FAISS limit");
  NIMBLE_CHECK_FILE_GT(
      metadata.numVectors, 0, "VectorIndex vector count must be positive");
  NIMBLE_CHECK_FILE_LE(
      metadata.numVectors,
      static_cast<uint64_t>(std::numeric_limits<faiss::idx_t>::max()),
      "VectorIndex vector count exceeds the FAISS limit");
}

// Parses and validates one vector index descriptor's logical metadata.
VectorIndex::Metadata parseVectorIndexMetadata(
    const serialization::VectorIndexMeta& config) {
  NIMBLE_CHECK_FILE_NOT_NULL(
      config.column_name(), "VectorIndex column name is missing");
  const auto* configTable =
      reinterpret_cast<const flatbuffers::Table*>(&config);
  const auto serializedMetric = configTable->GetField<int8_t>(
      serialization::VectorIndexMeta::VT_METRIC,
      static_cast<int8_t>(serialization::VectorDistanceMetric_L2));
  const auto serializedIndexType = configTable->GetField<int8_t>(
      serialization::VectorIndexMeta::VT_INDEX_TYPE,
      static_cast<int8_t>(serialization::VectorIndexType_IVF_FLAT));

  VectorIndex::Metadata metadata{
      .columnName = config.column_name()->str(),
      .dimensions = config.dimensions(),
      .metric = fromSerializedMetric(serializedMetric),
      .indexType = fromSerializedIndexType(serializedIndexType),
      .numVectors = config.num_vectors(),
  };
  validateVectorIndexMetadata(metadata);
  return metadata;
}

// Parses and validates one vector index data section.
MetadataSection parseVectorIndexSection(
    const serialization::MetadataSection& section) {
  NIMBLE_CHECK_FILE_GT(
      section.size(), 0, "VectorIndex index data must not be empty");
  NIMBLE_CHECK_FILE_LE(
      section.size(),
      kMaxVectorIndexSizeBytes,
      "VectorIndex index data exceeds the supported limit");
  const auto uncompressedSize = section.uncompressed_size();
  NIMBLE_CHECK_FILE_GT(
      uncompressedSize,
      0,
      "VectorIndex uncompressed data size must be recorded");
  NIMBLE_CHECK_FILE_LE(
      uncompressedSize,
      kMaxVectorIndexSizeBytes,
      "VectorIndex uncompressed data exceeds the supported limit");

  const auto* sectionTable =
      reinterpret_cast<const flatbuffers::Table*>(&section);
  const auto serializedCompressionType = sectionTable->GetField<uint8_t>(
      serialization::MetadataSection::VT_COMPRESSION_TYPE,
      static_cast<uint8_t>(serialization::CompressionType_Uncompressed));
  NIMBLE_CHECK_FILE_LE(
      serializedCompressionType,
      static_cast<uint8_t>(serialization::CompressionType_MAX),
      "VectorIndex compression type is invalid");
  const auto compressionType =
      static_cast<CompressionType>(serializedCompressionType);
  if (compressionType == CompressionType::Uncompressed) {
    NIMBLE_CHECK_FILE_EQ(
        uncompressedSize,
        section.size(),
        "VectorIndex uncompressed data size must equal its section size");
  } else {
    NIMBLE_CHECK_FILE_GE(
        uncompressedSize,
        section.size(),
        "VectorIndex uncompressed data must not be smaller than its section");
  }

  return MetadataSection{
      section.offset(),
      section.size(),
      compressionType,
      uncompressedSize,
  };
}

} // namespace

// Owns the state required to create the index input on first use.
struct VectorIndexDirectory::InputState {
  explicit InputState(const IndexLookup::Options& sourceOptions)
      : ioOptions{[&] {
          NIMBLE_CHECK_NOT_NULL(sourceOptions.ioOptions);
          return std::make_shared<velox::io::ReaderOptions>(
              *sourceOptions.ioOptions);
        }()},
        options{
            .file = sourceOptions.file,
            .ioOptions = ioOptions.get(),
            .fileHandle = sourceOptions.fileHandle,
            .cache = sourceOptions.cache,
            .pinIndex = sourceOptions.pinIndex,
            .preloadIndex = sourceOptions.preloadIndex,
        } {
    NIMBLE_CHECK_NOT_NULL(options.file);
  }

  std::shared_ptr<MetadataInput> input() {
    folly::call_once(initializeOnce, [this] {
      options.validate();
      metadataInput = createIndexMetadataInput(options);
    });
    NIMBLE_CHECK_NOT_NULL(metadataInput);
    return metadataInput;
  }

  // Keeps the ReaderOptions referenced by options alive.
  std::shared_ptr<velox::io::ReaderOptions> ioOptions;

  // Preserves index-specific statistics and cache behavior.
  IndexLookup::Options options;

  // Serializes construction while allowing concurrent reads afterward.
  folly::once_flag initializeOnce;

  // Remains null until the first vector index is materialized.
  std::shared_ptr<MetadataInput> metadataInput;
};

VectorIndexDirectory VectorIndexDirectory::create(
    Section directorySection,
    const IndexLookup::Options& options) {
  const auto directoryData = directorySection.content();
  NIMBLE_CHECK_FILE(
      !directoryData.empty(), "Vector index directory must not be empty");
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(directoryData.data()),
      directoryData.size());
  NIMBLE_CHECK_FILE(
      serialization::VerifyVectorIndexDirectoryBuffer(verifier),
      "Invalid VectorIndexDirectory FlatBuffer");

  const auto* root = flatbuffers::GetRoot<serialization::VectorIndexDirectory>(
      directoryData.data());
  NIMBLE_CHECK_FILE_NOT_NULL(root, "VectorIndexDirectory root is missing");
  const auto* descriptors = root->indices();
  NIMBLE_CHECK_FILE_NOT_NULL(
      descriptors, "VectorIndexDirectory indices are missing");
  NIMBLE_CHECK_FILE_GT(
      descriptors->size(), 0, "VectorIndexDirectory must contain an index");

  folly::F14FastMap<std::string, Entry> entries;
  entries.reserve(descriptors->size());
  for (const auto* descriptor : *descriptors) {
    NIMBLE_CHECK_FILE_NOT_NULL(descriptor, "VectorIndex descriptor is missing");
    const auto* config = descriptor->config();
    NIMBLE_CHECK_FILE_NOT_NULL(config, "VectorIndex config is missing");
    auto metadata = parseVectorIndexMetadata(*config);
    NIMBLE_CHECK_FILE(
        !entries.contains(metadata.columnName),
        "VectorIndex column name is duplicated: {}",
        metadata.columnName);
    const auto* section = descriptor->index_data();
    NIMBLE_CHECK_FILE_NOT_NULL(
        section, "VectorIndex index data section is missing");
    const auto columnName = metadata.columnName;
    entries.emplace(
        columnName,
        Entry{
            .metadata = std::move(metadata),
            .indexSection = parseVectorIndexSection(*section),
        });
  }

  return VectorIndexDirectory{
      std::move(entries), std::make_shared<InputState>(options)};
}

VectorIndexDirectory::VectorIndexDirectory(
    folly::F14FastMap<std::string, Entry> entries,
    std::shared_ptr<InputState> inputState)
    : entries_{std::move(entries)}, inputState_{std::move(inputState)} {
  NIMBLE_CHECK_NOT_NULL(inputState_);
}

size_t VectorIndexDirectory::numIndexes() const {
  return entries_.size();
}

bool VectorIndexDirectory::contains(std::string_view columnName) const {
  return entries_.contains(columnName);
}

std::shared_ptr<const VectorIndex> VectorIndexDirectory::load(
    std::string_view columnName) const {
  const auto entry = entries_.find(columnName);
  NIMBLE_USER_CHECK(
      entry != entries_.end(),
      "Vector index column does not exist: {}",
      columnName);

  const auto indexData =
      inputState_->input()->load({&entry->second.indexSection, 1});
  NIMBLE_CHECK_EQ(indexData.size(), 1);
  NIMBLE_CHECK_NOT_NULL(indexData.front().get());
  return VectorIndex::create(
      entry->second.metadata, indexData.front()->content());
}

std::shared_ptr<const VectorIndex> VectorIndex::create(
    Metadata metadata,
    std::string_view serializedIndex) {
  validateVectorIndexMetadata(metadata);
  return std::shared_ptr<const VectorIndex>(
      new VectorIndex(std::move(metadata), serializedIndex));
}

VectorIndex::VectorIndex(Metadata metadata, std::string_view serializedIndex)
    : columnName_{std::move(metadata.columnName)},
      dimensions_{metadata.dimensions},
      metric_{metadata.metric},
      indexType_{metadata.indexType},
      numVectors_{metadata.numVectors},
      faissIndex_{readFaissIndex(serializedIndex)} {
  NIMBLE_CHECK_FILE_EQ(
      faissIndex_->d,
      static_cast<int>(dimensions_),
      "FAISS index dimensions disagree with its metadata");
  NIMBLE_CHECK_FILE_EQ(
      faissIndex_->ntotal,
      static_cast<faiss::idx_t>(numVectors_),
      "FAISS vector count disagrees with its metadata");
  NIMBLE_CHECK_FILE_EQ(
      static_cast<int>(faissIndex_->metric_type),
      static_cast<int>(toFaissMetric(metric_)),
      "FAISS index metric disagrees with its metadata");
  NIMBLE_CHECK_FILE(
      checkIndexType(*faissIndex_, indexType_),
      "FAISS index type disagrees with its metadata");
}

VectorIndex::~VectorIndex() = default;

std::vector<VectorIndex::SearchResult> VectorIndex::search(
    const SearchConfig& config) const {
  NIMBLE_USER_CHECK_EQ(
      config.queryVector.size(),
      static_cast<size_t>(dimensions_),
      "Query vector dimensions do not match the index");
  NIMBLE_USER_CHECK_GT(
      config.numNeighbors, 0, "Number of neighbors must be positive");

  const auto queryVector = prepareQueryVector(config.queryVector, metric_);

  const auto maxNumNeighbors = static_cast<faiss::idx_t>(
      std::min<uint64_t>(config.numNeighbors, numVectors_));
  std::vector<float> scores(static_cast<size_t>(maxNumNeighbors));
  std::vector<faiss::idx_t> labels(static_cast<size_t>(maxNumNeighbors));

  searchFaissIndex(
      *faissIndex_,
      indexType_,
      config,
      queryVector.data(),
      maxNumNeighbors,
      scores.data(),
      labels.data());

  std::vector<SearchResult> results;
  results.reserve(static_cast<size_t>(maxNumNeighbors));
  for (faiss::idx_t i = 0; i < maxNumNeighbors; ++i) {
    if (labels[i] == -1) {
      // FAISS uses -1 for unfilled slots when the probed partitions contain
      // fewer candidates than requested. All later slots are also unfilled.
      break;
    }
    NIMBLE_CHECK_LT(
        static_cast<uint64_t>(labels[i]),
        numVectors_,
        "FAISS returned an out-of-range row ID");
    results.push_back(
        SearchResult{
            .rowId = labels[i],
            .score = scores[i],
        });
  }

  return results;
}

const std::string& VectorIndex::columnName() const {
  return columnName_;
}

uint32_t VectorIndex::dimensions() const {
  return dimensions_;
}

VectorDistanceMetric VectorIndex::metric() const {
  return metric_;
}

VectorIndexType VectorIndex::indexType() const {
  return indexType_;
}

uint64_t VectorIndex::numVectors() const {
  return numVectors_;
}

} // namespace facebook::nimble::index
