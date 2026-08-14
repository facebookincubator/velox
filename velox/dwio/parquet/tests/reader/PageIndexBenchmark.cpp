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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <thread>
#include <vector>

#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include "velox/common/base/Fs.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/DirectBufferedInput.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;

DEFINE_uint64(
    page_index_cloud_request_latency_us,
    8'000,
    "Fixed latency charged to each simulated cloud request.");
DEFINE_uint64(
    page_index_cloud_bandwidth_mib_per_sec,
    250,
    "Serialized transfer bandwidth for simulated cloud requests.");
DEFINE_int32(
    page_index_cloud_prefetch_row_groups,
    3,
    "Number of row groups to prefetch in simulated cloud scans.");
DEFINE_int32(
    page_index_cloud_iterations,
    5,
    "Number of alternating measured scans per simulated cloud mode.");

namespace {

constexpr vector_size_t kRowsPerRowGroup = 100'000;
constexpr vector_size_t kNumRowGroups = 4;
constexpr vector_size_t kNumMatchingRowsPerGroup = 1'000;
constexpr vector_size_t kPayloadBytes = 256;
constexpr uint32_t kNumMeasuredRuns = 9;
constexpr uint64_t kFooterReadBytes = 1ULL << 20;

struct CloudStorageProfile {
  uint64_t requestLatencyMicros{8'000};
  uint64_t bandwidthBytesPerSecond{250ULL << 20};
  uint64_t naturalReadSize{8ULL << 20};
};

struct CloudReadStats {
  std::atomic<uint64_t> numRequests{0};
  std::atomic<uint64_t> numReads{0};
  std::atomic<uint64_t> numBatchReads{0};
  std::atomic<uint64_t> numRanges{0};
  std::atomic<uint64_t> bytesRead{0};
  std::atomic<uint64_t> injectedDelayMicros{0};
};

// Models a serialized cloud request at the ReadFile boundary. One preadv call
// pays one round trip, regardless of the number of ranges in the batch.
class SimulatedCloudReadFile final : public ReadFile {
 public:
  SimulatedCloudReadFile(
      std::shared_ptr<ReadFile> delegate,
      CloudStorageProfile profile,
      std::shared_ptr<CloudReadStats> stats)
      : delegate_(std::move(delegate)),
        profile_(profile),
        stats_(std::move(stats)),
        name_("simulated-cloud:" + delegate_->getName()) {
    VELOX_CHECK_GT(profile_.bandwidthBytesPerSecond, 0);
  }

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buffer,
      const FileIoContext& context = {}) const override {
    recordRequest(length, 1, false);
    return delegate_->pread(offset, length, buffer, context);
  }

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context = {}) const override {
    const auto bytes = std::accumulate(
        buffers.begin(),
        buffers.end(),
        uint64_t{0},
        [](auto total, auto range) { return total + range.size(); });
    recordRequest(bytes, buffers.size(), true);
    return delegate_->preadv(offset, buffers, context);
  }

  uint64_t preadv(
      folly::Range<const common::Region*> regions,
      folly::Range<folly::IOBuf*> buffers,
      const FileIoContext& context = {}) const override {
    const auto bytes = std::accumulate(
        regions.begin(),
        regions.end(),
        uint64_t{0},
        [](auto total, auto region) { return total + region.length; });
    recordRequest(bytes, regions.size(), true);
    return delegate_->preadv(regions, buffers, context);
  }

  uint64_t preadv(
      folly::Range<const common::Region*> regions,
      folly::Range<const folly::Range<char*>*> buffers,
      const FileIoContext& context = {}) const override {
    const auto bytes = std::accumulate(
        regions.begin(),
        regions.end(),
        uint64_t{0},
        [](auto total, auto region) { return total + region.length; });
    recordRequest(bytes, regions.size(), true);
    return delegate_->preadv(regions, buffers, context);
  }

  bool directIo(uint64_t& alignment) const override {
    return delegate_->directIo(alignment);
  }

  bool shouldCoalesce() const override {
    return true;
  }

  uint64_t size() const override {
    return delegate_->size();
  }

  uint64_t memoryUsage() const override {
    return sizeof(*this) + delegate_->memoryUsage();
  }

  std::string getName() const override {
    return name_;
  }

  uint64_t getNaturalReadSize() const override {
    return profile_.naturalReadSize;
  }

 private:
  void recordRequest(uint64_t bytes, uint64_t numRanges, bool batch) const {
    const auto transferMicros = static_cast<uint64_t>(std::ceil(
        static_cast<long double>(bytes) * 1'000'000 /
        profile_.bandwidthBytesPerSecond));
    const auto delayMicros = profile_.requestLatencyMicros + transferMicros;
    ++stats_->numRequests;
    if (batch) {
      ++stats_->numBatchReads;
    } else {
      ++stats_->numReads;
    }
    stats_->numRanges += numRanges;
    stats_->bytesRead += bytes;
    stats_->injectedDelayMicros += delayMicros;
    bytesRead_ += bytes;
    std::this_thread::sleep_for(std::chrono::microseconds(delayMicros));
  }

  const std::shared_ptr<ReadFile> delegate_;
  const CloudStorageProfile profile_;
  const std::shared_ptr<CloudReadStats> stats_;
  const std::string name_;
};

struct ScanResult {
  uint64_t elapsedMicros{0};
  uint64_t outputRows{0};
  uint64_t dataBytesRead{0};
  uint64_t metadataBytesRead{0};
  int64_t pageIndexBytesRead{0};
  int64_t dataBytesPlanned{0};
  int64_t dataBytesAvoided{0};
  int64_t pagesSkipped{0};
  int64_t pagesRetained{0};
  uint64_t cloudRequests{0};
  uint64_t cloudReads{0};
  uint64_t cloudBatchReads{0};
  uint64_t cloudRanges{0};
  uint64_t cloudBytesRead{0};
  uint64_t injectedDelayMicros{0};
};

uint64_t median(std::vector<uint64_t> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

class PageIndexBenchmark {
 public:
  explicit PageIndexBenchmark(
      std::optional<std::string> persistentFilePath = std::nullopt) {
    rootPool_ = memory::memoryManager()->addRootPool("PageIndexBenchmark");
    leafPool_ = rootPool_->addLeafChild("PageIndexBenchmark");
    if (persistentFilePath.has_value()) {
      filePath_ = std::move(*persistentFilePath);
      std::filesystem::remove(filePath_);
    } else {
      tempDirectory_ = common::testutil::TempDirectoryPath::create();
      filePath_ = tempDirectory_->getPath() + "/page-index-benchmark.parquet";
    }
    writeInput();
  }

  ScanResult scan(
      bool pageIndexEnabled,
      int64_t filterUpperBound,
      uint64_t expectedOutputRows,
      const CloudStorageProfile* cloudProfile = nullptr,
      int32_t prefetchRowGroups =
          io::ReaderOptions::kDefaultPrefetchRowGroups) {
    auto dataIoStats = std::make_shared<io::IoStatistics>();
    auto metadataIoStats = std::make_shared<io::IoStatistics>();
    dwio::common::ReaderOptions readerOptions{leafPool_.get()};
    readerOptions.setDataIoStats(dataIoStats);
    readerOptions.setMetadataIoStats(metadataIoStats);
    readerOptions.setFilePreloadThreshold(0);
    readerOptions.setPrefetchRowGroups(prefetchRowGroups);
    auto parquetOptions = std::make_shared<ParquetReaderOptions>();
    parquetOptions->footerSpeculativeIoSize = kFooterReadBytes;
    parquetOptions->setFilterColumnIndexEnabled(pageIndexEnabled);
    readerOptions.setFormatSpecificOptions(std::move(parquetOptions));

    std::shared_ptr<CloudReadStats> cloudReadStats;
    std::unique_ptr<BufferedInput> input;
    if (cloudProfile != nullptr) {
      cloudReadStats = std::make_shared<CloudReadStats>();
      auto readFile = std::make_shared<SimulatedCloudReadFile>(
          std::make_shared<LocalReadFile>(filePath_),
          *cloudProfile,
          cloudReadStats);
      auto tracker = std::make_shared<cache::ScanTracker>(
          "PageIndexBenchmark", nullptr, readerOptions.loadQuantum());
      input = std::make_unique<DirectBufferedInput>(
          std::move(readFile),
          MetricsLog::voidLog(),
          StringIdLease(fileIds(), filePath_),
          std::move(tracker),
          StringIdLease(fileIds(), "PageIndexBenchmark"),
          dataIoStats,
          nullptr,
          nullptr,
          readerOptions);
    } else {
      input = std::make_unique<BufferedInput>(
          std::make_shared<LocalReadFile>(filePath_),
          *leafPool_,
          MetricsLog::voidLog(),
          dataIoStats.get());
    }
    auto reader = std::make_unique<ParquetReader>(
        std::move(input), std::move(readerOptions));

    const auto initialDataBytesRead = dataIoStats->rawBytesRead();
    const auto initialMetadataBytesRead = metadataIoStats->rawBytesRead();
    const auto initialCloudRequests =
        cloudReadStats ? cloudReadStats->numRequests.load() : 0;
    const auto initialCloudReads =
        cloudReadStats ? cloudReadStats->numReads.load() : 0;
    const auto initialCloudBatchReads =
        cloudReadStats ? cloudReadStats->numBatchReads.load() : 0;
    const auto initialCloudRanges =
        cloudReadStats ? cloudReadStats->numRanges.load() : 0;
    const auto initialCloudBytesRead =
        cloudReadStats ? cloudReadStats->bytesRead.load() : 0;
    const auto initialInjectedDelayMicros =
        cloudReadStats ? cloudReadStats->injectedDelayMicros.load() : 0;

    auto scanSpec = std::make_shared<common::ScanSpec>("");
    scanSpec->addAllChildFields(*rowType_);
    scanSpec->getOrCreateChild(common::Subfield("id"))
        ->setFilter(
            std::make_unique<common::BigintRange>(
                std::numeric_limits<int64_t>::min(), filterUpperBound, false));
    RowReaderOptions rowReaderOptions;
    rowReaderOptions.select(
        std::make_shared<ColumnSelector>(rowType_, rowType_->names()));
    rowReaderOptions.setScanSpec(scanSpec);

    const auto start = std::chrono::steady_clock::now();
    auto rowReader = reader->createRowReader(rowReaderOptions);
    auto result = BaseVector::create(rowType_, 1, leafPool_.get());
    uint64_t outputRows{0};
    while (rowReader->next(10'000, result) > 0) {
      auto* rowVector = result->asUnchecked<RowVector>();
      for (auto column = 0; column < rowVector->childrenSize(); ++column) {
        rowVector->childAt(column)->loadedVector();
      }
      outputRows += result->size();
    }
    const auto elapsedMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start)
            .count();

    RuntimeStatistics runtimeStats;
    rowReader->updateRuntimeStats(runtimeStats);
    VELOX_CHECK_EQ(outputRows, expectedOutputRows);
    return {
        static_cast<uint64_t>(elapsedMicros),
        outputRows,
        dataIoStats->rawBytesRead() - initialDataBytesRead,
        metadataIoStats->rawBytesRead() - initialMetadataBytesRead,
        runtimeStats.pageIndexBytesRead,
        runtimeStats.pageIndexDataBytesPlanned,
        runtimeStats.pageIndexDataBytesAvoided,
        runtimeStats.pageIndexPagesSkipped,
        runtimeStats.pageIndexPagesRetained,
        cloudReadStats
            ? cloudReadStats->numRequests.load() - initialCloudRequests
            : 0,
        cloudReadStats ? cloudReadStats->numReads.load() - initialCloudReads
                       : 0,
        cloudReadStats
            ? cloudReadStats->numBatchReads.load() - initialCloudBatchReads
            : 0,
        cloudReadStats ? cloudReadStats->numRanges.load() - initialCloudRanges
                       : 0,
        cloudReadStats
            ? cloudReadStats->bytesRead.load() - initialCloudBytesRead
            : 0,
        cloudReadStats ? cloudReadStats->injectedDelayMicros.load() -
                initialInjectedDelayMicros
                       : 0,
    };
  }

  uint64_t fileSize() const {
    return fs::file_size(filePath_);
  }

 private:
  void writeInput() {
    test::VectorMaker vectorMaker{leafPool_.get()};
    const auto numRows = kRowsPerRowGroup * kNumRowGroups;
    auto ids = vectorMaker.flatVector<int64_t>(
        numRows, [](auto row) { return row % kRowsPerRowGroup; });
    const std::string payload(kPayloadBytes, 'x');
    auto payloads = vectorMaker.flatVector<std::string>(
        numRows, [&](auto /*row*/) { return payload; });
    auto data = vectorMaker.rowVector({"id", "payload"}, {ids, payloads});
    rowType_ = asRowType(data->type());

    auto localWriteFile =
        std::make_unique<LocalWriteFile>(filePath_, true, false);
    auto sink =
        std::make_unique<WriteFileSink>(std::move(localWriteFile), filePath_);
    WriterOptions writerOptions;
    writerOptions.memoryPool = rootPool_.get();
    writerOptions.compressionKind = common::CompressionKind_NONE;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<DefaultFlushPolicy>(
          kRowsPerRowGroup, 512ULL << 20);
    };
    auto parquetOptions = std::make_shared<ParquetWriterOptions>();
    parquetOptions->enableWritePageIndex = true;
    parquetOptions->enableDictionary = false;
    parquetOptions->dataPageSize = 64 << 10;
    parquetOptions->batchSize = 10'000;
    writerOptions.formatSpecificOptions = std::move(parquetOptions);
    parquet::Writer writer(std::move(sink), writerOptions, rowType_);
    writer.write(data);
    writer.close();
  }

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
  std::shared_ptr<common::testutil::TempDirectoryPath> tempDirectory_;
  std::string filePath_;
  RowTypePtr rowType_;
};

void printResult(
    std::string_view label,
    const ScanResult& representative,
    const std::vector<uint64_t>& elapsedMicros) {
  std::cout << label << ": median_us=" << median(elapsedMicros)
            << " output_rows=" << representative.outputRows
            << " data_bytes_read=" << representative.dataBytesRead
            << " metadata_bytes_read=" << representative.metadataBytesRead
            << " page_index_bytes=" << representative.pageIndexBytesRead
            << " planned_data_bytes=" << representative.dataBytesPlanned
            << " avoided_data_bytes=" << representative.dataBytesAvoided
            << " pages_skipped=" << representative.pagesSkipped
            << " pages_retained=" << representative.pagesRetained
            << " cloud_requests=" << representative.cloudRequests
            << " cloud_reads=" << representative.cloudReads
            << " cloud_batch_reads=" << representative.cloudBatchReads
            << " cloud_ranges=" << representative.cloudRanges
            << " cloud_bytes_read=" << representative.cloudBytesRead
            << " injected_delay_us=" << representative.injectedDelayMicros
            << '\n';
}

void runScenario(
    PageIndexBenchmark& benchmark,
    std::string_view label,
    int64_t filterUpperBound,
    uint64_t expectedOutputRows,
    uint32_t numMeasuredRuns = kNumMeasuredRuns,
    const CloudStorageProfile* cloudProfile = nullptr,
    int32_t prefetchRowGroups = io::ReaderOptions::kDefaultPrefetchRowGroups) {
  benchmark.scan(
      false,
      filterUpperBound,
      expectedOutputRows,
      cloudProfile,
      prefetchRowGroups);
  benchmark.scan(
      true,
      filterUpperBound,
      expectedOutputRows,
      cloudProfile,
      prefetchRowGroups);
  std::vector<uint64_t> disabledTimes;
  std::vector<uint64_t> enabledTimes;
  ScanResult disabledResult;
  ScanResult enabledResult;
  for (uint32_t iteration = 0; iteration < numMeasuredRuns; ++iteration) {
    if (iteration % 2 == 0) {
      disabledResult = benchmark.scan(
          false,
          filterUpperBound,
          expectedOutputRows,
          cloudProfile,
          prefetchRowGroups);
      enabledResult = benchmark.scan(
          true,
          filterUpperBound,
          expectedOutputRows,
          cloudProfile,
          prefetchRowGroups);
    } else {
      enabledResult = benchmark.scan(
          true,
          filterUpperBound,
          expectedOutputRows,
          cloudProfile,
          prefetchRowGroups);
      disabledResult = benchmark.scan(
          false,
          filterUpperBound,
          expectedOutputRows,
          cloudProfile,
          prefetchRowGroups);
    }
    disabledTimes.push_back(disabledResult.elapsedMicros);
    enabledTimes.push_back(enabledResult.elapsedMicros);
  }

  printResult(std::string(label) + "_disabled", disabledResult, disabledTimes);
  printResult(std::string(label) + "_enabled", enabledResult, enabledTimes);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  const auto* persistentFilePath =
      std::getenv("VELOX_PAGE_INDEX_BENCHMARK_FILE");
  PageIndexBenchmark benchmark(
      persistentFilePath ? std::make_optional<std::string>(persistentFilePath)
                         : std::nullopt);
  std::cout << "file_bytes=" << benchmark.fileSize() << '\n';
  runScenario(
      benchmark,
      "local_selective",
      kNumMatchingRowsPerGroup - 1,
      kNumRowGroups * kNumMatchingRowsPerGroup);
  runScenario(
      benchmark,
      "local_match_all",
      kRowsPerRowGroup - 1,
      kNumRowGroups * kRowsPerRowGroup);

  VELOX_CHECK_GT(FLAGS_page_index_cloud_bandwidth_mib_per_sec, 0);
  VELOX_CHECK_GE(FLAGS_page_index_cloud_prefetch_row_groups, 0);
  VELOX_CHECK_GT(FLAGS_page_index_cloud_iterations, 0);
  const CloudStorageProfile cloudProfile{
      .requestLatencyMicros = FLAGS_page_index_cloud_request_latency_us,
      .bandwidthBytesPerSecond = FLAGS_page_index_cloud_bandwidth_mib_per_sec
          << 20,
  };
  std::cout << "cloud_request_latency_us=" << cloudProfile.requestLatencyMicros
            << " cloud_bandwidth_bytes_per_second="
            << cloudProfile.bandwidthBytesPerSecond
            << " cloud_prefetch_row_groups="
            << FLAGS_page_index_cloud_prefetch_row_groups
            << " cloud_iterations=" << FLAGS_page_index_cloud_iterations
            << '\n';
  runScenario(
      benchmark,
      "cloud_selective",
      kNumMatchingRowsPerGroup - 1,
      kNumRowGroups * kNumMatchingRowsPerGroup,
      FLAGS_page_index_cloud_iterations,
      &cloudProfile,
      FLAGS_page_index_cloud_prefetch_row_groups);
  runScenario(
      benchmark,
      "cloud_match_all",
      kRowsPerRowGroup - 1,
      kNumRowGroups * kRowsPerRowGroup,
      FLAGS_page_index_cloud_iterations,
      &cloudProfile,
      FLAGS_page_index_cloud_prefetch_row_groups);
  return 0;
}
