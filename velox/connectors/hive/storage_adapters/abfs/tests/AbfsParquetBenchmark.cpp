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

#include <folly/init/Init.h>
#include <folly/ScopeGuard.h>
#include <folly/executors/IOThreadPoolExecutor.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileProperties.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/storage_adapters/abfs/RegisterAbfsFileSystem.h"
#include "velox/connectors/hive/storage_adapters/abfs/tests/AzuriteServer.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/OperatorStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/PortUtil.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::filesystems {
namespace {

using namespace std::chrono;
using common::testutil::TempFilePath;
using exec::test::AssertQueryBuilder;
using exec::test::HiveConnectorTestBase;
using exec::test::PlanBuilder;
using exec::test::kHiveConnectorId;
using namespace facebook::velox::test;

constexpr int64_t kSeed = 1'000'000'007;
constexpr uint32_t kRows = 48'000;
constexpr uint32_t kRowsPerRowGroup = 12'000;
constexpr uint32_t kCopies = 8;
constexpr uint32_t kRounds = 3;

struct ProcessSample {
  uint64_t threads{0};
  uint64_t rssKb{0};
};

class StdoutSilencer {
 public:
  StdoutSilencer() : saved_(dup(STDOUT_FILENO)) {
    VELOX_CHECK_GE(saved_, 0);
    const auto sink = open("/dev/null", O_WRONLY);
    VELOX_CHECK_GE(sink, 0);
    std::fflush(stdout);
    VELOX_CHECK_EQ(dup2(sink, STDOUT_FILENO), STDOUT_FILENO);
    close(sink);
  }

  ~StdoutSilencer() {
    std::fflush(stdout);
    VELOX_CHECK_EQ(dup2(saved_, STDOUT_FILENO), STDOUT_FILENO);
    close(saved_);
  }

 private:
  int saved_;
};

ProcessSample sampleProcess() {
  std::ifstream status("/proc/self/status");
  ProcessSample sample;
  std::string line;
  while (std::getline(status, line)) {
    const auto separator = line.find(':');
    if (separator == std::string::npos) {
      continue;
    }
    const auto key = line.substr(0, separator);
    if (key != "Threads" && key != "VmRSS") {
      continue;
    }
    uint64_t value;
    std::string unit;
    std::istringstream values(line.substr(separator + 1));
    if (!(values >> value)) {
      continue;
    }
    if (key == "Threads") {
      sample.threads = value;
    } else if (!(values >> unit) || unit == "kB") {
      sample.rssKb = value;
    }
  }
  return sample;
}

class ProcessSampler {
 public:
  ProcessSampler() : initial_(sampleProcess()), peak_(initial_) {
    sampling_.store(true, std::memory_order_relaxed);
    sampler_ = std::thread([this] {
      while (sampling_.load(std::memory_order_relaxed)) {
        const auto current = sampleProcess();
        peak_.threads = std::max(peak_.threads, current.threads);
        peak_.rssKb = std::max(peak_.rssKb, current.rssKb);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
  }

  void stop() {
    sampling_.store(false, std::memory_order_relaxed);
    if (sampler_.joinable()) {
      sampler_.join();
    }
  }

  ~ProcessSampler() {
    stop();
  }

  const ProcessSample& initial() const {
    return initial_;
  }

  const ProcessSample& peak() const {
    return peak_;
  }

 private:
  ProcessSample initial_;
  ProcessSample peak_;
  std::atomic<bool> sampling_{false};
  std::thread sampler_;
};

int64_t rowValue(uint64_t row) {
  return kSeed + static_cast<int64_t>(row) * 1'000'003;
}

int64_t rowPayload(uint64_t row) {
  const auto value = rowValue(row);
  return (value << 7) ^ (value >> 3) ^ 0x5a5a5a5a5a5a5a5aLL;
}

uint64_t rowChecksum(uint64_t row) {
  return static_cast<uint64_t>(rowValue(row)) * 0x100000001b3ULL ^
      static_cast<uint64_t>(rowPayload(row));
}

uint64_t fixtureChecksum() {
  uint64_t checksum = 0;
  for (uint64_t row = 0; row < kRows; ++row) {
    checksum += rowChecksum(row);
  }
  return checksum;
}

struct CpuSample {
  uint64_t userMicros{0};
  uint64_t systemMicros{0};
};

CpuSample sampleCpu(int who) {
  struct rusage usage;
  VELOX_CHECK_EQ(getrusage(who, &usage), 0);
  return {
      static_cast<uint64_t>(usage.ru_utime.tv_sec) * 1'000'000 +
          usage.ru_utime.tv_usec,
      static_cast<uint64_t>(usage.ru_stime.tv_sec) * 1'000'000 +
          usage.ru_stime.tv_usec};
}

struct ScanStats {
  uint64_t tableScanInputBytes{0};
  uint64_t tableScanRawInputBytes{0};
  std::optional<RuntimeMetric> storageReadBytes;
  std::optional<RuntimeMetric> overreadBytes;
};

ScanStats scanStats(const std::shared_ptr<exec::Task>& task) {
  ScanStats result;
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      if (op.operatorType == "TableScan") {
        result.tableScanInputBytes += op.inputBytes;
        result.tableScanRawInputBytes += op.rawInputBytes;
        for (const auto& [name, metric] : op.runtimeStats) {
          if (name == "storageReadBytes") {
            if (!result.storageReadBytes) {
              result.storageReadBytes = metric;
            } else {
              result.storageReadBytes->sum += metric.sum;
              result.storageReadBytes->count += metric.count;
            }
          } else if (name == "overreadBytes") {
            if (!result.overreadBytes) {
              result.overreadBytes = metric;
            } else {
              result.overreadBytes->sum += metric.sum;
              result.overreadBytes->count += metric.count;
            }
          }
        }
      }
    }
  }
  return result;
}

class BenchmarkHarness : public HiveConnectorTestBase {
 public:
  void TestBody() override {}

  void initialize() {
    SetUp();
  }

  void shutdown() {
    TearDown();
  }

  void run(const std::shared_ptr<AzuriteServer>& server) {
    const auto rowType = ROW({"id", "payload"}, {BIGINT(), BIGINT()});
    const auto fixture = TempFilePath::create();
    writeFixture(fixture->getPath(), rowType);
    server->addFile(fixture->getPath());

    parquet::registerParquetReaderFactory();
    auto cleanupReader = folly::makeGuard(
        [] { parquet::unregisterParquetReaderFactory(); });
    auto cleanupHiveConnector = folly::makeGuard([&] {
      connector::ConnectorRegistry::global().erase(kHiveConnectorId);
    });

    const auto fileSize = fixture->fileSize();
    const auto expectedChecksum = fixtureChecksum() * kCopies;
    std::cout << "# limitation=synchronous ABFS over local Azurite only; no real Azure service, native async, fibers, or io_uring claim\n";
    std::cout << "# fixture_schema=id:BIGINT,payload:BIGINT; seed=" << kSeed
          << "; id=seed+row*1000003; payload=(id<<7)^(id>>3)^5a5a5a5a5a5a5a5a\n";
    std::cout << "# fixture_rows=" << kRows
              << "; row_groups=" << (kRows / kRowsPerRowGroup)
              << "; file_size_bytes=" << fileSize
              << "; data_checksum=" << fixtureChecksum() << "\n";
    std::cout << "# query_drivers=" << kCopies
              << "; storage_read_operations=Velox IO operations, not proven HTTP request counts or wire bytes\n";
    std::cout << "# blob_range_downloads_derived=storageReadBytes.count when available; derived from the current synchronous ABFS code path, not an observed HTTP/server count\n";
    std::cout << "# http_request_count_observed=unavailable; current ABFS/Azurite helper exposes no safe counter\n";
    std::cout << "# cache_rounds=cold-cache; AsyncDataCache cleared before every measured round; fileProperties avoids GetProperties in the measured path\n";
    std::cout << "# sampler_interval_ms=5; sampled_peak_rss_kb=peak sampled VmRSS, not VmHWM\n";
    std::cout << "io_executor_threads,round,elapsed_wall_us,process_user_us,process_system_us,storage_read_bytes,storage_read_operations,blob_range_downloads_derived,overread_bytes,http_request_count_observed,table_scan_input_bytes,table_scan_raw_input_bytes,initial_threads,final_threads,sampled_peak_threads,sampled_peak_rss_kb,row_count,checksum\n";

    for (const auto executorThreads : {1U, 2U, 8U}) {
      connector::ConnectorRegistry::global().erase(kHiveConnectorId);
      ioExecutor_.reset();
      ioExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(
          executorThreads);
      resetHiveConnector(server->hiveConfig({
          {connector::hive::FileConfig::kFilePreloadThreshold, "0"},
          {connector::hive::HiveConfig::kEnableFileHandleCache, "false"},
          {"fs.azure.account.auth.type.test.dfs.core.windows.net", "SharedKey"}
      }));
      for (uint32_t round = 1; round <= kRounds; ++round) {
        if (auto* cache = cache::AsyncDataCache::getInstance()) {
          cache->clear();
        }
        const auto roundSplits = makeSplits(server->fileURI(), fileSize);
        ProcessSampler sampler;
        const auto& initial = sampler.initial();

        const auto cpuBefore = sampleCpu(RUSAGE_SELF);
        const auto start = steady_clock::now();
        std::shared_ptr<exec::Task> task;
        auto result = AssertQueryBuilder(
                           PlanBuilder(pool_.get())
                               .tableScan(rowType)
                               .planNode(),
                           duckDbQueryRunner_)
                          .maxDrivers(kCopies)
                          .splits(roundSplits)
                          .copyResults(pool_.get(), task);
        const auto elapsed = duration_cast<microseconds>(
                                 steady_clock::now() - start)
                                 .count();
        const auto cpuAfter = sampleCpu(RUSAGE_SELF);
        sampler.stop();
        const auto final = sampleProcess();

        const auto checksum = checksumResult(result);
        VELOX_CHECK_EQ(
            result->size(), static_cast<vector_size_t>(kRows * kCopies));
        VELOX_CHECK_EQ(checksum, expectedChecksum);
        const auto stats = scanStats(task);
        std::cout << executorThreads << ',' << round << ',' << elapsed << ','
                  << cpuAfter.userMicros - cpuBefore.userMicros << ','
                  << cpuAfter.systemMicros - cpuBefore.systemMicros
                  << ','
                  << (stats.storageReadBytes
                          ? std::to_string(stats.storageReadBytes->sum)
                          : "unavailable")
                  << ','
                  << (stats.storageReadBytes
                          ? std::to_string(stats.storageReadBytes->count)
                          : "unavailable")
                  << ','
                  << (stats.storageReadBytes
                          ? std::to_string(stats.storageReadBytes->count)
                          : "unavailable")
                  << ','
                  << (stats.overreadBytes
                          ? std::to_string(stats.overreadBytes->sum)
                          : "unavailable")
                  << ",unavailable," << stats.tableScanInputBytes << ','
                  << stats.tableScanRawInputBytes << ','
                  << initial.threads << ',' << final.threads << ','
                  << sampler.peak().threads << ',' << sampler.peak().rssKb << ','
                  << result->size() << ',' << checksum << '\n';
      }
    }
  }

 private:
  static std::vector<std::shared_ptr<connector::ConnectorSplit>> makeSplits(
      const std::string& path,
      uint64_t fileSize) {
    std::vector<std::shared_ptr<connector::ConnectorSplit>> splits;
    for (uint32_t i = 0; i < kCopies; ++i) {
      splits.push_back(
          connector::hive::HiveConnectorSplitBuilder(path)
              .connectorId(kHiveConnectorId)
              .fileFormat(dwio::common::FileFormat::PARQUET)
              .start(0)
              .length(fileSize)
              .cacheable(false)
              .fileProperties(FileProperties{
                  .fileSize = static_cast<int64_t>(fileSize)})
              .build());
    }
    return splits;
  }

  static uint64_t checksumResult(const RowVectorPtr& result) {
    uint64_t checksum = 0;
    const auto* ids = result->childAt(0)->as<FlatVector<int64_t>>();
    const auto* payloads = result->childAt(1)->as<FlatVector<int64_t>>();
    for (vector_size_t row = 0; row < result->size(); ++row) {
      checksum += static_cast<uint64_t>(ids->valueAt(row)) *
              0x100000001b3ULL ^
          static_cast<uint64_t>(payloads->valueAt(row));
    }
    return checksum;
  }

  void writeFixture(const std::string& path, const RowTypePtr& rowType) {
    auto pool = rootPool_->addAggregateChild("AbfsParquetBenchmark.Writer");
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::make_unique<LocalWriteFile>(path, true, false), path);
    dwio::common::WriterOptions writerOptions;
    writerOptions.memoryPool = pool.get();
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<parquet::DefaultFlushPolicy>(
          kRowsPerRowGroup, std::numeric_limits<int64_t>::max());
    };
    writerOptions.formatSpecificOptions =
        std::make_shared<parquet::ParquetWriterOptions>();
    parquet::Writer writer(std::move(sink), writerOptions, rowType);
    for (uint64_t firstRow = 0; firstRow < kRows;
         firstRow += kRowsPerRowGroup) {
      writer.write(makeRowVector(
          {"id", "payload"},
          {makeFlatVector<int64_t>(kRowsPerRowGroup, [firstRow](auto row) {
             return rowValue(firstRow + row);
           }),
           makeFlatVector<int64_t>(kRowsPerRowGroup, [firstRow](auto row) {
             return rowPayload(firstRow + row);
           })}));
    }
    writer.close();
  }
};

} // namespace
} // namespace facebook::velox::filesystems

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::exec::test::OperatorTestBase::SetUpTestCase();
  facebook::velox::filesystems::registerAbfsFileSystem();
  auto port = facebook::velox::exec::test::getFreePort();
  std::shared_ptr<facebook::velox::filesystems::AzuriteServer> server;
  {
    facebook::velox::filesystems::StdoutSilencer silence;
    server = std::make_shared<facebook::velox::filesystems::AzuriteServer>(
        port);
    server->start();
  }
  auto stopServer = folly::makeGuard([&] { server->stop(); });

  facebook::velox::filesystems::BenchmarkHarness benchmark;
  benchmark.initialize();
  auto shutdown = folly::makeGuard([&] { benchmark.shutdown(); });
  benchmark.run(server);
  return 0;
}
