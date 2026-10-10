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
#include <variant>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/zerocopy_io.h>
#include <faiss/index_io.h>
#include <flatbuffers/flatbuffers.h>
#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
#include <folly/synchronization/CallOnce.h>
#include <omp.h>

#include "velox/common/Casts.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/io/Options.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/VectorIndexUtility.h"
#include "velox/dwio/nimble/tablet/MetadataInput.h"
#include "velox/dwio/nimble/tablet/VectorIndexGenerated.h"

namespace facebook::nimble::index {

namespace {

// Mode 3 assigns whole queries, rather than query-list pairs, to OpenMP
// workers. This preserves each query's ordered list scan for max_codes, but a
// single-query batch remains single-threaded.
constexpr int kIvfParallelModeQueries{3};

// Deserializes one FAISS index from caller-owned zero-copy storage.
std::unique_ptr<faiss::Index> readFaissIndex(std::string_view serializedIndex) {
  NIMBLE_CHECK_FILE(
      !serializedIndex.empty(), "FAISS index data must not be empty");
  // Upstream FAISS takes a mutable pointer but does not modify the input.
  auto* serializedData = const_cast<uint8_t*>(
      reinterpret_cast<const uint8_t*>(serializedIndex.data()));
  faiss::ZeroCopyIOReader reader(serializedData, serializedIndex.size());
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

// Normalizes a copy for cosine and otherwise uses the caller's storage.
const float* prepareQueryVectors(
    faiss::idx_t numQueries,
    const std::vector<float>& queryVectors,
    VectorDistanceMetric metric,
    uint32_t dimensions,
    std::vector<float>& normalizedQueries) {
  if (metric != VectorDistanceMetric::kCosine) {
    return queryVectors.data();
  }
  // Preserve the caller's query vectors while normalizing the search input.
  normalizedQueries = queryVectors;
  // FAISS implements cosine similarity as inner product over unit vectors.
  normalizeVectors(
      metric,
      /*numVectors=*/static_cast<uint64_t>(numQueries),
      dimensions,
      normalizedQueries.data());
  return normalizedQueries.data();
}

// Serializes HNSW searches because FAISS updates process-global statistics.
folly::Synchronized<std::monostate>& hnswSearchGuard() {
  static auto* guard = new folly::Synchronized<std::monostate>();
  return *guard;
}

// Adapts one Nimble row selection to FAISS.
class FaissRowSelection {
 public:
  // Validates the selection and borrows bitmap data without copying it.
  FaissRowSelection(
      const VectorIndex::SearchConfig::RowSelection& rowSelection,
      uint64_t numVectors) {
    if (const auto bitmap = rowSelection.bitmap()) {
      const auto expectedBitmapBytes = velox::bits::nbytes(numVectors);
      NIMBLE_USER_CHECK_EQ(
          bitmap->size(),
          expectedBitmapBytes,
          "Row selection bitmap size does not match the index");
      selector_.emplace<faiss::IDSelectorBitmap>(
          bitmap->size(), bitmap->data());
      return;
    }

    const auto range = rowSelection.range();
    if (!range.has_value()) {
      return;
    }
    NIMBLE_USER_CHECK_LE(
        range->startRow, range->endRow, "Row selection range must be ordered");
    NIMBLE_USER_CHECK_LE(
        range->endRow,
        numVectors,
        "Row selection range exceeds the index row count");
    selector_.emplace<faiss::IDSelectorRange>(range->startRow, range->endRow);
  }

  // Applies the adapted selector to FAISS search parameters.
  void applyTo(faiss::SearchParameters& searchParameters) {
    if (auto* bitmap = std::get_if<faiss::IDSelectorBitmap>(&selector_)) {
      searchParameters.sel = bitmap;
    } else if (auto* range = std::get_if<faiss::IDSelectorRange>(&selector_)) {
      searchParameters.sel = range;
    }
    // FAISS interprets a null selector as selecting every indexed row.
  }

 private:
  // Owns one adapted selector and borrows bitmap storage when applicable.
  std::variant<std::monostate, faiss::IDSelectorBitmap, faiss::IDSelectorRange>
      selector_;
};

// Searches an HNSW index with request-local parameters.
void searchHnswIndex(
    const faiss::IndexHNSW& index,
    faiss::idx_t numQueries,
    uint32_t hnswSearchDepth,
    const float* queryVectors,
    faiss::idx_t maxNumNeighbors,
    float* scores,
    faiss::idx_t* labels,
    FaissRowSelection& rowSelection) {
  NIMBLE_USER_CHECK_GT(
      hnswSearchDepth, 0, "HNSW search depth must be positive");
  const auto effectiveSearchDepth =
      std::min<uint64_t>(hnswSearchDepth, static_cast<uint64_t>(index.ntotal));
  NIMBLE_USER_CHECK_LE(
      effectiveSearchDepth,
      static_cast<uint64_t>(std::numeric_limits<int>::max()),
      "HNSW search depth exceeds the FAISS limit");
  NIMBLE_USER_CHECK_LE(
      maxNumNeighbors,
      static_cast<faiss::idx_t>(std::numeric_limits<int>::max()),
      "Number of neighbors exceeds the HNSW limit");
  faiss::SearchParametersHNSW searchParameters;
  searchParameters.efSearch = static_cast<int>(effectiveSearchDepth);
  rowSelection.applyTo(searchParameters);
  // FAISS updates process-global HNSW statistics during search.
  const auto hnswSearchLock = hnswSearchGuard().wlock();
  index.search(
      numQueries,
      queryVectors,
      maxNumNeighbors,
      scores,
      labels,
      &searchParameters);
}

// Searches an IVF index without updating FAISS process-global statistics.
void searchIvfIndex(
    const faiss::IndexIVF& index,
    faiss::idx_t numQueries,
    uint32_t requestedNumProbes,
    uint64_t maxCodes,
    bool ensureTopKFull,
    const float* queryVectors,
    faiss::idx_t maxNumNeighbors,
    float* scores,
    faiss::idx_t* labels,
    FaissRowSelection& rowSelection) {
  const auto* quantizer =
      velox::checkedPointerCast<const faiss::IndexFlat>(index.quantizer);
  NIMBLE_USER_CHECK_GT(
      requestedNumProbes, 0, "Number of probed partitions must be positive");
  faiss::SearchParametersIVF searchParameters;
  rowSelection.applyTo(searchParameters);
  NIMBLE_USER_CHECK_LE(
      maxCodes,
      static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
      "Maximum scanned codes exceeds the FAISS limit");
  searchParameters.max_codes = static_cast<size_t>(maxCodes);
  searchParameters.ensure_topk_full = ensureTopKFull;
  const auto numProbes = static_cast<faiss::idx_t>(
      std::min(index.nlist, static_cast<size_t>(requestedNumProbes)));
  NIMBLE_CHECK_GT(numProbes, 0);
  searchParameters.nprobe = static_cast<size_t>(numProbes);
  const auto numPartitionAssignments =
      static_cast<size_t>(numQueries) * static_cast<size_t>(numProbes);
  std::vector<float> centroidScores(numPartitionAssignments);
  std::vector<faiss::idx_t> partitionLabels(numPartitionAssignments);

  // IVF search first finds the nearest coarse partitions, then scans vectors
  // assigned to those partitions. Keeping the stages explicit lets the second
  // pass use request-local statistics instead of FAISS's process-global state.
  quantizer->search(
      numQueries,
      queryVectors,
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
  index.search_preassigned(
      numQueries,
      queryVectors,
      maxNumNeighbors,
      partitionLabels.data(),
      centroidScores.data(),
      scores,
      labels,
      /*store_pairs=*/false,
      &searchParameters,
      &searchStats);
}

// Returns whether the runtime index type belongs to the IVF family.
bool isIvfIndexType(VectorIndexType indexType) {
  switch (indexType) {
    case VectorIndexType::kHnswSq8:
      return false;
    case VectorIndexType::kIvfFlat:
    case VectorIndexType::kIvfSq8:
    case VectorIndexType::kIvfPq:
    case VectorIndexType::kIvfRaBitQ:
      return true;
  }
  NIMBLE_UNREACHABLE(
      "Unsupported vector index type: {}", static_cast<int>(indexType));
}

// Dispatches a batch using the selected index-specific options.
void searchFaissIndex(
    const faiss::Index& index,
    VectorIndexType indexType,
    faiss::idx_t numQueries,
    const VectorIndex::SearchOptions& searchOptions,
    const float* queryVectors,
    faiss::idx_t maxNumNeighbors,
    float* scores,
    faiss::idx_t* labels,
    FaissRowSelection& rowSelection) {
  if (isIvfIndexType(indexType)) {
    NIMBLE_USER_CHECK(
        searchOptions.kind() == VectorIndex::SearchOptions::Kind::kIvf,
        "IVF index requires IVF search options");
    const auto ivfSearchOptions =
        velox::checkedPointerCast<const VectorIndex::IvfSearchOptions>(
            &searchOptions);
    const auto* ivfIndex =
        velox::checkedPointerCast<const faiss::IndexIVF>(&index);
    searchIvfIndex(
        *ivfIndex,
        numQueries,
        ivfSearchOptions->numProbes,
        ivfSearchOptions->maxCodes,
        ivfSearchOptions->ensureTopKFull,
        queryVectors,
        maxNumNeighbors,
        scores,
        labels,
        rowSelection);
    return;
  }

  NIMBLE_USER_CHECK(
      searchOptions.kind() == VectorIndex::SearchOptions::Kind::kHnsw,
      "HNSW index requires HNSW search options");
  const auto hnswSearchOptions =
      velox::checkedPointerCast<const VectorIndex::HnswSearchOptions>(
          &searchOptions);
  const auto* hnswIndex =
      velox::checkedPointerCast<const faiss::IndexHNSW>(&index);
  searchHnswIndex(
      *hnswIndex,
      numQueries,
      hnswSearchOptions->searchDepth,
      queryVectors,
      maxNumNeighbors,
      scores,
      labels,
      rowSelection);
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

VectorIndex::SearchConfig::RowSelection
VectorIndex::SearchConfig::RowSelection::all() {
  return RowSelection{AllRows{}};
}

VectorIndex::SearchConfig::RowSelection
VectorIndex::SearchConfig::RowSelection::fromBitmap(
    std::span<const uint8_t> bitmap) {
  return RowSelection{bitmap};
}

VectorIndex::SearchConfig::RowSelection
VectorIndex::SearchConfig::RowSelection::fromRange(RowRange range) {
  return RowSelection{range};
}

std::optional<std::span<const uint8_t>>
VectorIndex::SearchConfig::RowSelection::bitmap() const {
  if (const auto* bitmap = std::get_if<std::span<const uint8_t>>(&selection_)) {
    return *bitmap;
  }
  return std::nullopt;
}

std::optional<RowRange> VectorIndex::SearchConfig::RowSelection::range() const {
  if (const auto* range = std::get_if<RowRange>(&selection_)) {
    return *range;
  }
  return std::nullopt;
}

VectorIndex::SearchConfig::RowSelection::RowSelection(Selection selection)
    : selection_{std::move(selection)} {}

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
  const auto serializedIndex = indexData.front()->content();
  return VectorIndex::create(
      entry->second.metadata, serializedIndex, indexData.front());
}

std::shared_ptr<const VectorIndex> VectorIndex::create(
    Metadata metadata,
    std::string_view serializedIndex,
    std::shared_ptr<const void> indexData) {
  validateVectorIndexMetadata(metadata);
  NIMBLE_CHECK_NOT_NULL(indexData);
  return std::shared_ptr<const VectorIndex>(new VectorIndex(
      std::move(metadata), serializedIndex, std::move(indexData)));
}

VectorIndex::VectorIndex(
    Metadata metadata,
    std::string_view serializedIndex,
    std::shared_ptr<const void> indexData)
    : columnName_{std::move(metadata.columnName)},
      dimensions_{metadata.dimensions},
      metric_{metadata.metric},
      indexType_{metadata.indexType},
      numVectors_{metadata.numVectors},
      indexData_{std::move(indexData)},
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
  if (isIvfIndexType(indexType_)) {
    auto* ivfIndex =
        velox::checkedPointerCast<faiss::IndexIVF>(faissIndex_.get());
    ivfIndex->parallel_mode = kIvfParallelModeQueries;
  }
}

VectorIndex::~VectorIndex() = default;

VectorIndex::SearchResults::SearchResults(
    std::vector<SearchResult> results,
    std::vector<size_t> resultOffsets)
    : results_{std::move(results)}, resultOffsets_{std::move(resultOffsets)} {
  NIMBLE_CHECK_GT(
      resultOffsets_.size(), 1, "Search result must contain query offsets");
  NIMBLE_CHECK_EQ(
      resultOffsets_.front(), 0, "Search result must start at offset zero");
  NIMBLE_CHECK_EQ(
      resultOffsets_.back(),
      results_.size(),
      "Search result final offset must match the result count");
  for (size_t queryIndex = 0; queryIndex + 1 < resultOffsets_.size();
       ++queryIndex) {
    NIMBLE_CHECK_LE(
        resultOffsets_[queryIndex],
        resultOffsets_[queryIndex + 1],
        "Search result offsets must be monotonic");
  }
}

size_t VectorIndex::SearchResults::numQueries() const {
  return resultOffsets_.size() - 1;
}

size_t VectorIndex::SearchResults::totalNumResults() const {
  return results_.size();
}

folly::Range<const VectorIndex::SearchResult*>
VectorIndex::SearchResults::results(size_t queryIndex) const& {
  NIMBLE_CHECK_LT(queryIndex, numQueries());
  return {
      results_.data() + resultOffsets_[queryIndex],
      results_.data() + resultOffsets_[queryIndex + 1],
  };
}

VectorIndex::SearchResults VectorIndex::search(
    const SearchConfig& config) const {
  NIMBLE_USER_CHECK(
      !config.queryVectors.empty(), "Query vector batch must not be empty");
  NIMBLE_USER_CHECK_GT(
      config.numQueries, 0, "Number of query vectors must be positive");
  const auto expectedQueryElements =
      static_cast<size_t>(config.numQueries) * dimensions_;
  NIMBLE_USER_CHECK_EQ(
      config.queryVectors.size(),
      expectedQueryElements,
      "Query vector batch size does not match the number of queries and index dimensions");
  NIMBLE_USER_CHECK_NOT_NULL(
      config.searchOptions, "Search options must be set");
  NIMBLE_USER_CHECK_GT(
      config.numNeighbors, 0, "Number of neighbors must be positive");
  NIMBLE_USER_CHECK_GT(
      config.numSearchThreads, 0, "Number of search threads must be positive");
  NIMBLE_USER_CHECK_LE(
      config.numSearchThreads,
      static_cast<uint32_t>(std::numeric_limits<int>::max()),
      "Number of search threads exceeds the OpenMP limit");
  const auto prevNumThreads = omp_get_max_threads();
  // omp_set_num_threads updates the calling OpenMP task's nthreads-var ICV,
  // so concurrent searches on other driver threads retain their own budgets.
  omp_set_num_threads(static_cast<int>(config.numSearchThreads));
  SCOPE_EXIT {
    omp_set_num_threads(prevNumThreads);
  };
  const auto numQueries = static_cast<faiss::idx_t>(config.numQueries);

  FaissRowSelection rowSelection{config.rowSelection, numVectors_};

  std::vector<float> normalizedQueries;
  const auto* queryData = prepareQueryVectors(
      numQueries, config.queryVectors, metric_, dimensions_, normalizedQueries);

  const auto maxNumNeighbors = static_cast<faiss::idx_t>(
      std::min<uint64_t>(config.numNeighbors, numVectors_));
  NIMBLE_CHECK_GT(maxNumNeighbors, 0);
  const auto numResultSlots = numQueries * static_cast<size_t>(maxNumNeighbors);
  std::vector<float> scores(numResultSlots);
  std::vector<faiss::idx_t> labels(numResultSlots);

  searchFaissIndex(
      *faissIndex_,
      indexType_,
      static_cast<faiss::idx_t>(numQueries),
      *config.searchOptions,
      queryData,
      maxNumNeighbors,
      scores.data(),
      labels.data(),
      rowSelection);

  std::vector<size_t> resultOffsets(numQueries + 1, 0);
  for (size_t queryIndex = 0; queryIndex < numQueries; ++queryIndex) {
    const auto resultOffset = queryIndex * static_cast<size_t>(maxNumNeighbors);
    size_t numQueryResults{0};
    // FAISS uses -1 for unfilled slots when the probed partitions contain
    // fewer candidates than requested. All later slots are also unfilled.
    while (numQueryResults < static_cast<size_t>(maxNumNeighbors) &&
           labels[resultOffset + numQueryResults] != -1) {
      ++numQueryResults;
    }
    resultOffsets[queryIndex + 1] = resultOffsets[queryIndex] + numQueryResults;
  }

  std::vector<SearchResult> results;
  results.reserve(resultOffsets.back());
  for (size_t queryIndex = 0; queryIndex < numQueries; ++queryIndex) {
    const auto resultOffset = queryIndex * static_cast<size_t>(maxNumNeighbors);
    const auto numQueryResults =
        resultOffsets[queryIndex + 1] - resultOffsets[queryIndex];
    for (size_t resultIndex = 0; resultIndex < numQueryResults; ++resultIndex) {
      const auto flatResultIndex = resultOffset + resultIndex;
      NIMBLE_CHECK_FILE_GE(
          labels[flatResultIndex], 0, "FAISS returned a negative row ID");
      NIMBLE_CHECK_FILE_LT(
          static_cast<uint64_t>(labels[flatResultIndex]),
          numVectors_,
          "FAISS returned an out-of-range row ID");
      results.push_back({
          .rowId = labels[flatResultIndex],
          .score = scores[flatResultIndex],
      });
    }
  }

  auto searchResults =
      SearchResults{std::move(results), std::move(resultOffsets)};
  NIMBLE_CHECK_EQ(searchResults.numQueries(), config.numQueries);
  return searchResults;
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
