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

/// Measures the per-split cost of file column statistics without and with
/// ReaderOptions::nimbleLazyFileStats(), on a wide file: 324 columns, 344
/// schema nodes and 10K rows. Each split does what a scan does before it reads
/// rows: opens the reader, prunes on one BIGINT column and estimates the row
/// size of the 4 columns it reads.
///
/// Run:
///   buck2 run @//mode/opt \
///     fbcode//velox/dwio/nimble/selective/tests:lazy_file_stats_benchmark
///
/// CPU us per split (opt mode, fastest of 5 runs of 1,000 splits) and memory
/// pool allocations per split:
///  mode   default allocator  --use_mmap_allocator  allocations
///  eager  237                418                   292
///  lazy   120                311                   37

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/time/Timer.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

DEFINE_int32(num_splits, 1'000, "Splits per run.");
DEFINE_bool(use_mmap_allocator, false, "Allocate with MmapAllocator.");

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

constexpr vector_size_t kNumRows{10'000};
constexpr int kNumRuns{5};

// Columns of each type. The ARRAY and MAP columns add 19 child nodes, for 1 +
// 324 + 19 = 344 nodes.
constexpr int kNumBigints{150};
constexpr int kNumDoubles{60};
constexpr int kNumVarchars{90};
constexpr int kNumBooleans{10};
constexpr int kNumArrays{9};
constexpr int kNumMaps{5};

// Cost of one run of FLAGS_num_splits splits.
struct SplitCost {
  uint64_t cpuNanos;
  uint64_t numAllocs;
};

// Returns the run with less CPU time.
SplitCost faster(const SplitCost& lhs, const SplitCost& rhs) {
  return lhs.cpuNanos <= rhs.cpuNanos ? lhs : rhs;
}

// Prints the per-split averages of 'cost'.
void print(const char* mode, const SplitCost& cost) {
  fmt::print(
      "{:<8}{:>16.1f}{:>16.1f}\n",
      mode,
      static_cast<double>(cost.cpuNanos) / FLAGS_num_splits / 1'000,
      static_cast<double>(cost.numAllocs) / FLAGS_num_splits);
}

class LazyFileStatsBenchmark : public velox::test::VectorTestBase {
 public:
  LazyFileStatsBenchmark() {
    registerSelectiveNimbleReaderFactory();
  }

  ~LazyFileStatsBenchmark() {
    unregisterSelectiveNimbleReaderFactory();
  }

  // Prints the fastest of kNumRuns runs in each mode.
  void run() {
    const auto table = makeTable();
    const auto file = test::createNimbleFile(*pool()->parent(), table);
    fmt::print(
        "{} columns, {} rows, {:.1f} MB; {} splits per run\n",
        table->type()->size(),
        kNumRows,
        static_cast<double>(file.size()) / (1 << 20),
        FLAGS_num_splits);
    const auto readFile = std::make_shared<InMemoryReadFile>(file);
    SplitCost eager{UINT64_MAX, 0};
    SplitCost lazy{UINT64_MAX, 0};
    for (int run = 0; run < kNumRuns; ++run) {
      eager = faster(eager, runSplits(readFile, /*lazyFileStats=*/false));
      lazy = faster(lazy, runSplits(readFile, /*lazyFileStats=*/true));
    }
    fmt::print("{:<8}{:>16}{:>16}\n", "mode", "CPU us/split", "allocs/split");
    print("eager", eager);
    print("lazy", lazy);
  }

 private:
  RowVectorPtr makeTable() {
    std::vector<std::string> names;
    std::vector<VectorPtr> columns;
    const auto add = [&](const std::string& name, VectorPtr column) {
      names.push_back(name);
      columns.push_back(std::move(column));
    };
    for (int i = 0; i < kNumBigints; ++i) {
      add(fmt::format("bigint_{}", i),
          makeFlatVector<int64_t>(
              kNumRows, [i](auto row) { return row * 7 % 1'000 + i; }));
    }
    for (int i = 0; i < kNumDoubles; ++i) {
      add(fmt::format("double_{}", i),
          makeFlatVector<double>(
              kNumRows, [i](auto row) { return row * 0.25 + i; }));
    }
    for (int i = 0; i < kNumVarchars; ++i) {
      add(fmt::format("varchar_{}", i),
          makeFlatVector<std::string>(kNumRows, [i](auto row) {
            return fmt::format("value_{}_{}", i, row % 97);
          }));
    }
    for (int i = 0; i < kNumBooleans; ++i) {
      add(fmt::format("boolean_{}", i),
          makeFlatVector<bool>(
              kNumRows, [i](auto row) { return (row + i) % 3 == 0; }));
    }
    for (int i = 0; i < kNumArrays; ++i) {
      add(fmt::format("array_{}", i),
          makeArrayVector<int64_t>(
              kNumRows,
              [](auto row) { return row % 4; },
              [i](auto index) { return index + i; }));
    }
    for (int i = 0; i < kNumMaps; ++i) {
      add(fmt::format("map_{}", i),
          makeMapVector<int64_t, double>(
              kNumRows,
              [](auto row) { return row % 3; },
              [](auto index) { return index % 10; },
              [i](auto index) { return index * 0.5 + i; }));
    }
    return makeRowVector(names, columns);
  }

  SplitCost runSplits(
      const std::shared_ptr<ReadFile>& readFile,
      bool lazyFileStats) {
    const auto numAllocsBefore = pool()->stats().numAllocs;
    uint64_t cpuNanos{0};
    {
      NanosecondCPUTimer timer(&cpuNanos);
      for (int split = 0; split < FLAGS_num_splits; ++split) {
        runSplit(readFile, lazyFileStats);
      }
    }
    return {cpuNanos, pool()->stats().numAllocs - numAllocsBefore};
  }

  void runSplit(const std::shared_ptr<ReadFile>& readFile, bool lazyFileStats) {
    dwio::common::ReaderOptions readerOptions(pool());
    readerOptions.setDataIoStats(std::make_shared<io::IoStatistics>());
    readerOptions.setMetadataIoStats(std::make_shared<io::IoStatistics>());
    readerOptions.setScanSpec(scanSpec_);
    readerOptions.setFilePreloadThreshold(0);
    readerOptions.setNimbleLazyFileStats(lazyFileStats);
    auto reader =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE)
            ->createReader(
                std::make_unique<dwio::common::BufferedInput>(
                    readFile, *pool()),
                readerOptions);

    const auto filterNode = reader->typeWithId()->childByName("bigint_0")->id();
    NIMBLE_CHECK(
        reader->columnStatistics(filterNode) != nullptr, "Missing stats");

    dwio::common::RowReaderOptions rowReaderOptions;
    rowReaderOptions.setScanSpec(scanSpec_);
    rowReaderOptions.setEagerFirstStripeLoad(false);
    const auto rowReader = reader->createRowReader(rowReaderOptions);
    NIMBLE_CHECK(
        rowReader->estimatedRowSize().has_value(), "Missing row size estimate");
  }

  const std::shared_ptr<common::ScanSpec> scanSpec_{[] {
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*ROW({
        {"bigint_0", BIGINT()},
        {"varchar_0", VARCHAR()},
        {"varchar_1", VARCHAR()},
        {"double_0", DOUBLE()},
    }));
    return scanSpec;
  }()};
};

} // namespace
} // namespace facebook::nimble

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::Options options;
  if (FLAGS_use_mmap_allocator) {
    options.allocatorCapacity = 8L << 30;
    options.arbitratorCapacity = options.allocatorCapacity;
    options.useMmapAllocator = true;
  }
  facebook::velox::memory::initializeMemoryManager(options);
  facebook::nimble::LazyFileStatsBenchmark benchmark;
  benchmark.run();
  return 0;
}
