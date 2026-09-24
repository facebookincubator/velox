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

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/utils/random.h>
#include <fmt/format.h>
#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <omp.h>

#include "fb_velox/common/Profiler.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/file/File.h"
#include "velox/common/io/Options.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/VectorIndex.h"
#include "velox/dwio/nimble/index/VectorIndexUtility.h"
#include "velox/dwio/nimble/index/VectorIndexWriter.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

DEFINE_uint32(
    vector_index_num_vectors,
    50'000,
    "Number of vectors in the generated index.");
DEFINE_uint32(
    vector_index_dimensions,
    1'024,
    "Number of float dimensions in each vector.");
DEFINE_uint32(
    vector_index_num_partitions,
    223,
    "Number of IVF partitions requested while building the index.");
DEFINE_string(
    vector_index_type,
    "ivf_rabitq",
    "Index type to benchmark: ivf_flat, ivf_sq8, ivf_pq, ivf_rabitq, or "
    "hnsw_sq8.");
DEFINE_string(
    vector_index_metric,
    "cosine",
    "Distance metric to benchmark: l2, cosine, or dot_product.");
DEFINE_uint32(
    vector_index_num_queries,
    100,
    "Number of queries searched per batch.");
DEFINE_uint32(
    vector_index_num_neighbors,
    50,
    "Number of nearest neighbors returned by each search.");
DEFINE_uint32(
    vector_index_num_probes,
    0,
    "Number of IVF partitions probed by each search; zero probes all "
    "partitions.");
DEFINE_uint32(
    vector_index_hnsw_search_depth,
    32,
    "HNSW candidate-list size used while traversing the graph.");
DEFINE_int64(
    vector_index_data_seed,
    1'234,
    "Seed used to generate deterministic synthetic vectors.");
DEFINE_uint32(
    vector_index_benchmark_seconds,
    60,
    "Minimum duration of the measured workload.");
DEFINE_uint32(
    vector_index_num_threads,
    1,
    "Number of OpenMP threads available to FAISS.");
DEFINE_string(
    vector_index_workload,
    "all",
    "Workload to run: write, load, reload_and_search, search, or all.");

namespace facebook::nimble::index {
namespace {

constexpr std::string_view kColumnName{"embedding"};
constexpr double kNanosPerSecond{1'000'000'000};
constexpr double kNanosPerMicrosecond{1'000};
constexpr uint64_t kFnv1a64OffsetBasis{14'695'981'039'346'656'037ULL};
constexpr uint64_t kFnv1a64Prime{1'099'511'628'211ULL};

// Configures one deterministic vector-index benchmark run.
struct BenchmarkOptions {
  // Sets the number of indexed vectors.
  uint32_t numVectors;
  // Sets the number of floating-point values in each vector.
  uint32_t dimensions;
  // Sets the number of IVF partitions created by the writer.
  uint32_t numPartitions;
  // Selects the index implementation under test.
  VectorIndexType indexType;
  // Selects the distance function used for build and search.
  VectorDistanceMetric metric;
  // Sets the number of held-out query vectors.
  uint32_t numQueries;
  // Sets the requested top-K result count.
  uint32_t numNeighbors;
  // Sets the IVF probe count; zero means all partitions.
  uint32_t numProbes;
  // Sets the HNSW candidate-list size used while traversing the graph.
  uint32_t hnswSearchDepth;
  // Selects the deterministic synthetic-data stream.
  int64_t dataSeed;
  // Sets the minimum measured duration of each workload.
  uint32_t benchmarkSeconds;
  // Selects the load, reload-and-search, search, or combined workload.
  std::string workload;
};

// Accumulates CPU, wall, and per-operation timing for one workload.
struct TimedResult {
  // Stores aggregate CPU and wall time in nanoseconds.
  velox::CpuWallTiming timing;
  // Counts completed load or search-batch operations.
  uint64_t numOperations{0};
  // Counts individual queries completed across all search batches.
  uint64_t numQueries{0};
  // Stores wall time in nanoseconds for each completed operation.
  std::vector<uint64_t> operationWallNanos;
};

// Stores nearest-rank wall-time percentiles in microseconds.
struct WallPercentiles {
  double p50Micros{0};
  double p90Micros{0};
  double p99Micros{0};
};

// Captures search quality and a deterministic result identity.
struct RecallResult {
  // Measures the fraction of exact top-K neighbors returned.
  double recallAtK{0};
  // Measures the fraction of queries containing the exact nearest neighbor.
  double oneRecallAtK{0};
  // Detects result changes between otherwise comparable benchmark runs.
  uint64_t resultChecksum{0};
};

// Owns the deterministic indexed vectors and held-out queries.
struct SyntheticData {
  // Number of vectors in the corpus.
  uint32_t numVectors;
  // Number of floating-point values in each vector.
  uint32_t dimensions;
  // Indexed vectors in row-major order.
  std::vector<float> corpus;
  // Held-out query vectors in row-major order.
  std::vector<float> queries;
};

// Owns the serialized index payload and its directory section.
struct WrittenIndex {
  std::string indexData;
  std::string directoryData;
};

VectorIndexType parseIndexType(std::string_view name) {
  if (name == "ivf_flat") {
    return VectorIndexType::kIvfFlat;
  }
  if (name == "ivf_sq8") {
    return VectorIndexType::kIvfSq8;
  }
  if (name == "ivf_pq") {
    return VectorIndexType::kIvfPq;
  }
  if (name == "ivf_rabitq") {
    return VectorIndexType::kIvfRaBitQ;
  }
  if (name == "hnsw_sq8") {
    return VectorIndexType::kHnswSq8;
  }
  VELOX_USER_FAIL("Unsupported vector index type: {}", name);
}

VectorDistanceMetric parseMetric(std::string_view name) {
  if (name == "l2") {
    return VectorDistanceMetric::kL2;
  }
  if (name == "cosine") {
    return VectorDistanceMetric::kCosine;
  }
  if (name == "dot_product") {
    return VectorDistanceMetric::kDotProduct;
  }
  VELOX_USER_FAIL("Unsupported vector index metric: {}", name);
}

double nanosPerOperation(uint64_t nanos, uint64_t numOperations) {
  VELOX_CHECK_GT(numOperations, 0);
  return static_cast<double>(nanos) / static_cast<double>(numOperations);
}

double operationsPerSecond(uint64_t numOperations, uint64_t wallNanos) {
  VELOX_CHECK_GT(numOperations, 0);
  return static_cast<double>(numOperations) * kNanosPerSecond /
      static_cast<double>(std::max<uint64_t>(1, wallNanos));
}

WallPercentiles wallPercentilesMicros(const std::vector<uint64_t>& samples) {
  VELOX_CHECK(!samples.empty());
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const auto percentile = [&sorted](double quantile) {
    const auto rank = static_cast<size_t>(
        std::ceil(quantile * static_cast<double>(sorted.size())));
    return static_cast<double>(sorted[std::max<size_t>(1, rank) - 1]) /
        kNanosPerMicrosecond;
  };
  return {
      .p50Micros = percentile(0.50),
      .p90Micros = percentile(0.90),
      .p99Micros = percentile(0.99),
  };
}

template <typename Operation>
TimedResult runForDuration(
    uint32_t benchmarkSeconds,
    uint32_t numQueriesPerOperation,
    Operation&& operation) {
  TimedResult result;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{benchmarkSeconds};
  result.operationWallNanos.reserve(1'024);
  {
    velox::ProcessCpuWallTimer timer{result.timing};
    do {
      const auto start = std::chrono::steady_clock::now();
      operation(result.numOperations);
      result.operationWallNanos.push_back(
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - start)
                  .count()));
      ++result.numOperations;
    } while (std::chrono::steady_clock::now() < deadline);
  }
  result.numQueries = result.numOperations * numQueriesPerOperation;
  return result;
}

// Drives deterministic vector-index write, load, search, and quality workloads.
class VectorIndexBenchmark {
 public:
  explicit VectorIndexBenchmark(BenchmarkOptions options)
      : options_{std::move(options)},
        rootPool_{velox::memory::memoryManager()->addRootPool(
            "VectorIndexBenchmark")},
        writerPool_{rootPool_->addLeafChild("writer")},
        readerPool_{rootPool_->addLeafChild("reader")} {
    auto syntheticData = makeSyntheticData();
    searchConfigs_ = makeSearchConfigs(syntheticData.queries);
    input_ = makeInput(syntheticData.corpus);
    index_ = writeIndex(input_);
    directory_.emplace(createIndexDirectory());
    recall_ = validateAndMeasureRecall(syntheticData);
  }

  // Executes and reports each selected measured workload.
  void run() const {
    if (shouldRun("write")) {
      printOperationResult(
          "Vector index build + serialize + destroy",
          runForDuration(
              options_.benchmarkSeconds, 0, [this](uint64_t /* operation */) {
                const auto written = writeIndex(input_);
                folly::doNotOptimizeAway(written.indexData.data());
                folly::doNotOptimizeAway(written.directoryData.data());
              }));
    }
    if (shouldRun("load")) {
      printOperationResult(
          "Resident load + destroy",
          runForDuration(
              options_.benchmarkSeconds, 0, [this](uint64_t /* operation */) {
                const auto index = directory_->load(kColumnName);
                folly::doNotOptimizeAway(index.get());
              }));
    }
    if (shouldRun("reload_and_search")) {
      printQueryResult(
          "Resident E2E (reload + sequential query batch + destroy)",
          runForDuration(
              options_.benchmarkSeconds,
              options_.numQueries,
              [this](uint64_t /* operation */) {
                const auto index = directory_->load(kColumnName);
                searchAll(*index);
              }));
    }
    if (shouldRun("search")) {
      const auto index = directory_->load(kColumnName);
      printQueryResult(
          "Same-object warm search",
          runForDuration(
              options_.benchmarkSeconds,
              options_.numQueries,
              [this, &index](uint64_t /* operation */) { searchAll(*index); }));
    }
    fmt::print(
        "Quality: recall@{}={:.4f}, 1-recall@{}={:.4f}, checksum={}\n",
        effectiveNumNeighbors(),
        recall_.recallAtK,
        effectiveNumNeighbors(),
        recall_.oneRecallAtK,
        recall_.resultChecksum);
  }

  // Prints the complete configuration and serialized index size.
  void printConfiguration() const {
    const auto indexParameters = options_.indexType == VectorIndexType::kHnswSq8
        ? fmt::format("hnsw_search_depth={}", options_.hnswSearchDepth)
        : fmt::format(
              "partitions={}, probes={}",
              options_.numPartitions,
              options_.numProbes == 0 ? "all"
                                      : fmt::to_string(options_.numProbes));
    fmt::print(
        "Nimble vector index: type={}, vectors={}, dimensions={}, "
        "metric={}, {}, queries={}, neighbors={}, data=faiss_smooth, seed={}, "
        "serialized={} B, workload={}, omp_threads={}\n",
        FLAGS_vector_index_type,
        options_.numVectors,
        options_.dimensions,
        FLAGS_vector_index_metric,
        indexParameters,
        options_.numQueries,
        effectiveNumNeighbors(),
        options_.dataSeed,
        index_.indexData.size(),
        options_.workload,
        FLAGS_vector_index_num_threads);
  }

 private:
  uint32_t effectiveNumNeighbors() const {
    return std::min(options_.numNeighbors, options_.numVectors);
  }

  bool shouldRun(std::string_view workload) const {
    return options_.workload == "all" || options_.workload == workload;
  }

  void printOperationResult(std::string_view name, const TimedResult& result)
      const {
    const auto percentiles = wallPercentilesMicros(result.operationWallNanos);
    fmt::print(
        "{}: operations={}, operations/s={:.2f}, "
        "cpu/operation={:.2f} us, wall/operation={:.2f} us, "
        "wall_p50={:.2f} us, wall_p90={:.2f} us, wall_p99={:.2f} us\n",
        name,
        result.numOperations,
        operationsPerSecond(result.numOperations, result.timing.wallNanos),
        nanosPerOperation(result.timing.cpuNanos, result.numOperations) /
            kNanosPerMicrosecond,
        nanosPerOperation(result.timing.wallNanos, result.numOperations) /
            kNanosPerMicrosecond,
        percentiles.p50Micros,
        percentiles.p90Micros,
        percentiles.p99Micros);
  }

  void printQueryResult(std::string_view name, const TimedResult& result)
      const {
    const auto percentiles = wallPercentilesMicros(result.operationWallNanos);
    fmt::print(
        "{}: batches={}, queries={}, qps={:.2f}, cpu/query={:.2f} us, "
        "wall/query={:.2f} us, batch_wall_p50={:.2f} us, "
        "batch_wall_p90={:.2f} us, batch_wall_p99={:.2f} us\n",
        name,
        result.numOperations,
        result.numQueries,
        operationsPerSecond(result.numQueries, result.timing.wallNanos),
        nanosPerOperation(result.timing.cpuNanos, result.numQueries) /
            kNanosPerMicrosecond,
        nanosPerOperation(result.timing.wallNanos, result.numQueries) /
            kNanosPerMicrosecond,
        percentiles.p50Micros,
        percentiles.p90Micros,
        percentiles.p99Micros);
  }

  SyntheticData makeSyntheticData() const {
    const auto corpusElements =
        static_cast<size_t>(options_.numVectors) * options_.dimensions;
    const auto queryElements =
        static_cast<size_t>(options_.numQueries) * options_.dimensions;
    std::vector<float> values(corpusElements + queryElements);
    faiss::rand_smooth_vectors(
        static_cast<size_t>(options_.numVectors) + options_.numQueries,
        options_.dimensions,
        values.data(),
        options_.dataSeed);
    const auto queryBegin = values.begin() +
        static_cast<std::vector<float>::difference_type>(corpusElements);
    SyntheticData data{
        .numVectors = options_.numVectors,
        .dimensions = options_.dimensions,
        .corpus = {values.begin(), queryBegin},
        .queries = {queryBegin, values.end()},
    };
    normalizeVectors(
        options_.metric, data.numVectors, data.dimensions, data.corpus.data());
    normalizeVectors(
        options_.metric,
        options_.numQueries,
        data.dimensions,
        data.queries.data());
    return data;
  }

  velox::RowVectorPtr makeInput(const std::vector<float>& vectors) const {
    VELOX_CHECK_EQ(vectors.size() % options_.dimensions, 0);
    const auto numVectors =
        static_cast<velox::vector_size_t>(vectors.size() / options_.dimensions);
    const auto numElements = static_cast<velox::vector_size_t>(vectors.size());
    auto elements = velox::BaseVector::create<velox::FlatVector<float>>(
        velox::REAL(), numElements, writerPool_.get());
    for (size_t i = 0; i < vectors.size(); ++i) {
      elements->set(static_cast<velox::vector_size_t>(i), vectors[i]);
    }
    auto offsets = velox::allocateOffsets(numVectors, writerPool_.get());
    auto sizes = velox::allocateSizes(numVectors, writerPool_.get());
    auto* rawOffsets = offsets->asMutable<velox::vector_size_t>();
    auto* rawSizes = sizes->asMutable<velox::vector_size_t>();
    VELOX_CHECK_NOT_NULL(rawOffsets);
    VELOX_CHECK_NOT_NULL(rawSizes);
    for (velox::vector_size_t row = 0; row < numVectors; ++row) {
      rawOffsets[row] =
          row * static_cast<velox::vector_size_t>(options_.dimensions);
      rawSizes[row] = static_cast<velox::vector_size_t>(options_.dimensions);
    }
    auto embeddings = std::make_shared<velox::ArrayVector>(
        writerPool_.get(),
        velox::ARRAY(velox::REAL()),
        nullptr,
        numVectors,
        std::move(offsets),
        std::move(sizes),
        std::move(elements));
    return std::make_shared<velox::RowVector>(
        writerPool_.get(),
        velox::ROW(std::string{kColumnName}, velox::ARRAY(velox::REAL())),
        nullptr,
        numVectors,
        std::vector<velox::VectorPtr>{std::move(embeddings)});
  }

  std::vector<VectorIndex::SearchConfig> makeSearchConfigs(
      const std::vector<float>& queries) const {
    VELOX_CHECK_EQ(
        queries.size(),
        static_cast<size_t>(options_.numQueries) * options_.dimensions);
    std::vector<VectorIndex::SearchConfig> searchConfigs;
    searchConfigs.reserve(options_.numQueries);
    for (uint32_t query = 0; query < options_.numQueries; ++query) {
      const auto queryBegin =
          queries.begin() +
          static_cast<std::ptrdiff_t>(
              static_cast<size_t>(query) * options_.dimensions);
      searchConfigs.push_back({
          .queryVector = std::vector<float>(
              queryBegin,
              queryBegin + static_cast<std::ptrdiff_t>(options_.dimensions)),
          .numNeighbors = effectiveNumNeighbors(),
          .numProbes = options_.numProbes == 0
              ? std::numeric_limits<uint32_t>::max()
              : options_.numProbes,
          .hnswSearchDepth = options_.hnswSearchDepth,
      });
    }
    return searchConfigs;
  }

  void searchAll(const VectorIndex& index) const {
    for (const auto& searchConfig : searchConfigs_) {
      const auto searchResults = index.search(searchConfig);
      folly::doNotOptimizeAway(searchResults.data());
    }
  }

  RecallResult validateAndMeasureRecall(const SyntheticData& data) const {
    const auto index = directory_->load(kColumnName);
    VELOX_CHECK_EQ(index->dimensions(), options_.dimensions);
    VELOX_CHECK_EQ(index->numVectors(), options_.numVectors);
    VELOX_CHECK(!searchConfigs_.empty());

    const auto numNeighbors = effectiveNumNeighbors();
    faiss::IndexFlat groundTruth{
        options_.dimensions, toFaissMetric(options_.metric)};
    groundTruth.add(options_.numVectors, data.corpus.data());
    std::vector<float> exactScores(
        static_cast<size_t>(options_.numQueries) * numNeighbors);
    std::vector<faiss::idx_t> exactLabels(
        static_cast<size_t>(options_.numQueries) * numNeighbors);
    groundTruth.search(
        options_.numQueries,
        data.queries.data(),
        numNeighbors,
        exactScores.data(),
        exactLabels.data());

    uint64_t numMatches{0};
    uint64_t numTopOneMatches{0};
    uint64_t checksum{kFnv1a64OffsetBasis};
    for (uint32_t query = 0; query < options_.numQueries; ++query) {
      const auto results = index->search(searchConfigs_.at(query));
      const auto exactBegin = exactLabels.begin() +
          static_cast<std::ptrdiff_t>(
                                  static_cast<size_t>(query) * numNeighbors);
      const auto exactEnd =
          exactBegin + static_cast<std::ptrdiff_t>(numNeighbors);
      if (std::any_of(
              results.begin(),
              results.end(),
              [exactTopOne = *exactBegin](const auto& result) {
                return result.rowId == exactTopOne;
              })) {
        ++numTopOneMatches;
      }
      for (const auto& result : results) {
        if (std::find(exactBegin, exactEnd, result.rowId) != exactEnd) {
          ++numMatches;
        }
        checksum ^= static_cast<uint64_t>(result.rowId) + 1;
        checksum *= kFnv1a64Prime;
      }
    }
    return {
        .recallAtK = static_cast<double>(numMatches) /
            (static_cast<double>(options_.numQueries) * numNeighbors),
        .oneRecallAtK =
            static_cast<double>(numTopOneMatches) / options_.numQueries,
        .resultChecksum = checksum,
    };
  }

  WrittenIndex writeIndex(const velox::RowVectorPtr& input) const {
    const VectorIndexConfig config{
        .columnName = std::string{kColumnName},
        .dimensions = options_.dimensions,
        .metric = options_.metric,
        .indexType = options_.indexType,
        .numPartitions = options_.numPartitions,
    };
    const std::array configs{config};
    auto writer = VectorIndexWriter::create(
        configs,
        std::static_pointer_cast<const velox::RowType>(input->type()),
        writerPool_.get());
    NIMBLE_CHECK_NOT_NULL(writer, "Vector index writer factory returned null");
    writer->write(input);

    WrittenIndex written;
    const CreateMetadataSectionFn createMetadataFn =
        [&written](std::string_view content) {
          const auto offset = written.indexData.size();
          written.indexData.append(content);
          return MetadataSection{
              offset,
              static_cast<uint32_t>(content.size()),
              CompressionType::Uncompressed,
              static_cast<uint32_t>(content.size()),
          };
        };
    const WriteOptionalSectionFn writeMetadataFn =
        [&written](const std::string& name, std::string_view content) {
          VELOX_CHECK_EQ(name, kVectorIndexSection);
          VELOX_CHECK(
              written.directoryData.empty(),
              "Vector-index directory must be written exactly once");
          written.directoryData.assign(content);
        };
    writer->close(createMetadataFn, writeMetadataFn);
    VELOX_CHECK(!written.indexData.empty());
    VELOX_CHECK(!written.directoryData.empty());
    return written;
  }

  VectorIndexDirectory createIndexDirectory() const {
    auto directoryBuffer = MetadataBuffer::decompress(
        index_.directoryData, CompressionType::Uncompressed, readerPool_.get());
    velox::io::ReaderOptions ioOptions{readerPool_.get()};
    ioOptions.setIndexIoStats(std::make_shared<velox::io::IoStatistics>());
    return VectorIndexDirectory::create(
        Section{MetadataBuffer{std::move(directoryBuffer)}},
        IndexLookup::Options{
            .file = std::make_shared<velox::InMemoryReadFile>(index_.indexData),
            .ioOptions = &ioOptions,
        });
  }

  // Holds immutable workload parameters.
  const BenchmarkOptions options_;
  // Owns the benchmark memory hierarchy.
  const std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  // Accounts for vectors and serialized index construction.
  const std::shared_ptr<velox::memory::MemoryPool> writerPool_;
  // Accounts for directory metadata and serialized-index I/O buffers.
  const std::shared_ptr<velox::memory::MemoryPool> readerPool_;
  // Stores one request for each held-out query.
  std::vector<VectorIndex::SearchConfig> searchConfigs_;
  // Owns the input reused by write workloads.
  velox::RowVectorPtr input_;
  // Owns the in-memory representation of the generated file sections.
  WrittenIndex index_;
  // Owns the directory reused by resident-load workloads.
  std::optional<VectorIndexDirectory> directory_;
  // Stores quality measured before timed workloads begin.
  RecallResult recall_;
};

// Runs the selected workload and captures its CPU and heap profiles.
void runBenchmark(const VectorIndexBenchmark& benchmark) {
  facebook::fb_velox::common::ScopedProfiler::Result profileResult;
  {
    facebook::fb_velox::common::ScopedProfiler profiler{&profileResult};
    benchmark.run();
  }
  fmt::print(
      "CPU profile result: {}\n"
      "Heap profile result: {}\n",
      profileResult.cpuScubaUrl.empty() ? "none" : profileResult.cpuScubaUrl,
      profileResult.heapScubaUrl.empty() ? "none" : profileResult.heapScubaUrl);
}

} // namespace
} // namespace facebook::nimble::index

int main(int argc, char** argv) {
  const folly::Init init{&argc, &argv};
  VELOX_USER_CHECK_GT(FLAGS_vector_index_num_vectors, 0);
  VELOX_USER_CHECK_GT(FLAGS_vector_index_dimensions, 0);
  VELOX_USER_CHECK_GT(FLAGS_vector_index_num_queries, 0);
  VELOX_USER_CHECK_GT(FLAGS_vector_index_num_neighbors, 0);
  VELOX_USER_CHECK_GT(FLAGS_vector_index_hnsw_search_depth, 0);
  VELOX_USER_CHECK_GT(FLAGS_vector_index_benchmark_seconds, 0);
  VELOX_USER_CHECK_GT(FLAGS_vector_index_num_threads, 0);
  VELOX_USER_CHECK(
      FLAGS_vector_index_workload == "all" ||
          FLAGS_vector_index_workload == "write" ||
          FLAGS_vector_index_workload == "load" ||
          FLAGS_vector_index_workload == "reload_and_search" ||
          FLAGS_vector_index_workload == "search",
      "Unsupported reader workload: {}",
      FLAGS_vector_index_workload);
  VELOX_USER_CHECK_LE(
      FLAGS_vector_index_num_threads,
      static_cast<uint32_t>(std::numeric_limits<int>::max()),
      "The OpenMP thread count is too large");
  VELOX_USER_CHECK_LE(
      static_cast<uint64_t>(std::max(
          FLAGS_vector_index_num_vectors, FLAGS_vector_index_num_queries)) *
          FLAGS_vector_index_dimensions,
      static_cast<uint64_t>(
          std::numeric_limits<facebook::velox::vector_size_t>::max()),
      "The generated elements vector is too large");
  facebook::velox::memory::MemoryManager::initialize({});
  omp_set_num_threads(static_cast<int>(FLAGS_vector_index_num_threads));

  facebook::nimble::index::VectorIndexBenchmark benchmark{{
      .numVectors = FLAGS_vector_index_num_vectors,
      .dimensions = FLAGS_vector_index_dimensions,
      .numPartitions = FLAGS_vector_index_num_partitions,
      .indexType =
          facebook::nimble::index::parseIndexType(FLAGS_vector_index_type),
      .metric = facebook::nimble::index::parseMetric(FLAGS_vector_index_metric),
      .numQueries = FLAGS_vector_index_num_queries,
      .numNeighbors = FLAGS_vector_index_num_neighbors,
      .numProbes = FLAGS_vector_index_num_probes,
      .hnswSearchDepth = FLAGS_vector_index_hnsw_search_depth,
      .dataSeed = FLAGS_vector_index_data_seed,
      .benchmarkSeconds = FLAGS_vector_index_benchmark_seconds,
      .workload = FLAGS_vector_index_workload,
  }};
  benchmark.printConfiguration();
  facebook::nimble::index::runBenchmark(benchmark);
}
