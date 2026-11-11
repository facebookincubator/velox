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

/// Measures null filters on scalar columns through the Velox selective reader.
/// IS NULL, and IS NOT NULL on an unprojected column, need only the nulls, so
/// the reader skips decoding the values. IS NOT NULL on a projected column
/// needs the values and is the control.
///
/// Schema: ROW(c0: INTEGER, c1: <type under test>). c1 carries the filter and
/// has either uniform random values with random nulls, or one constant value
/// and no nulls. c0 has no filter and is never loaded, so only c1 is decoded.
/// Files use the default encoding selection.
///
/// Run:
///   buck2 run @//mode/opt \
///     fbcode//velox/dwio/nimble/common/selective/tests:nulls_only_read_benchmark
///
/// c1 read CPU in ms (opt mode, 10M rows, nimbleScanFastPath on in both
/// builds), before -> after reading only the nulls. Each cell is the median of
/// 6 alternating runs per build, and each run takes the fastest of kNumRuns
/// reads. The percentages are the share of null rows. Value chunks are still
/// decompressed, so compressed columns gain less.
///                IS NULL      IS NOT NULL,    IS NOT NULL,
///                             unprojected     projected (control)
///  bool   10%    29 -> 9      29 -> 11        35 -> 37
///  bool   50%    74 -> 15     71 -> 7         68 -> 72
///  bool   90%    44 -> 25     28 -> 6         27 -> 28
///  int32  10%    66 -> 35     50 -> 37        51 -> 51
///  int32  50%   104 -> 30     30 -> 21        31 -> 31
///  int32  90%    56 -> 29     14 -> 9         15 -> 15
///  int64  10%    68 -> 35     50 -> 36        51 -> 51
///  int64  50%   106 -> 30     30 -> 21        30 -> 31
///  int64  90%    57 -> 29     14 -> 9         15 -> 15
///  double 10%    51 -> 33     46 -> 35        48 -> 48
///  double 50%    87 -> 29     27 -> 20        28 -> 29
///  double 90%    47 -> 29     13 -> 9         14 -> 14

#include <random>

#include <fmt/format.h>
#include <folly/init/Init.h>

#include "velox/common/time/Timer.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

constexpr vector_size_t kNumRows{10'000'000};
constexpr int kBatchSize{1'024};
constexpr int kNumRuns{10};

// A null filter on c1, and whether c1 is projected.
struct Query {
  std::string_view name;
  bool isNull;
  bool projectC1;
};

// Times of one read.
struct ReadTimes {
  // Wall time of the next() loop.
  uint64_t readNanos;

  // CPU time of the c1 read, from the reader's decoding stats.
  uint64_t c1Nanos;
};

// Returns a uniform random value of type 'T'.
template <typename T>
T randomValue(std::mt19937& rng) {
  if constexpr (std::is_same_v<T, bool>) {
    return std::bernoulli_distribution{}(rng);
  } else if constexpr (std::is_floating_point_v<T>) {
    return std::uniform_real_distribution<T>{-1e6, 1e6}(rng);
  } else {
    return std::uniform_int_distribution<T>{0, 999'999}(rng);
  }
}

// c1 is node 2: the root is node 0 and c0 is node 1.
uint64_t c1DecodeNanos(const dwio::common::RuntimeStats& stats) {
  const auto node = stats.columnStats.find(2);
  NIMBLE_CHECK(node != stats.columnStats.end(), "Missing c1 stats");
  const auto format = node->second.find(dwio::common::FileFormat::NIMBLE);
  NIMBLE_CHECK(
      format != node->second.end() && format->second.decodingStats.has_value(),
      "Missing c1 decoding stats");
  return format->second.decodingStats->decodeCPUTimeNanos.sum();
}

// Writes one file per c1 type and null mix, and prints each query's fastest
// read times on it.
class NullsOnlyReadBenchmark : public velox::test::VectorTestBase {
 public:
  NullsOnlyReadBenchmark() {
    registerSelectiveNimbleReaderFactory();
  }

  ~NullsOnlyReadBenchmark() {
    unregisterSelectiveNimbleReaderFactory();
  }

  void run() {
    runType<bool>("bool", BOOLEAN());
    runType<int32_t>("int32", INTEGER());
    runType<int64_t>("int64", BIGINT());
    runType<double>("double", DOUBLE());
  }

 private:
  template <typename T>
  void runType(std::string_view name, const TypePtr& type) {
    const auto rowType = ROW({{"c0", INTEGER()}, {"c1", type}});
    for (const int nullPercent : {10, 50, 90}) {
      runQueries(
          name,
          fmt::format("{}% nulls", nullPercent),
          rowType,
          makeNullableColumn<T>(nullPercent));
    }
    runQueries(
        name,
        "constant",
        rowType,
        makeFlatVector<T>(kNumRows, [](auto /*row*/) { return T{1}; }));
  }

  // Runs each query kNumRuns times on a file with 'c1' and prints its fastest
  // read times.
  void runQueries(
      std::string_view typeName,
      std::string_view columnName,
      const TypePtr& rowType,
      const VectorPtr& c1) {
    constexpr Query kQueries[] = {
        {"IS NULL", true, true},
        {"IS NOT NULL, unprojected", false, false},
        {"IS NOT NULL, projected", false, true},
    };
    const auto file = writeFile(c1);
    for (const auto& query : kQueries) {
      ReadTimes best{UINT64_MAX, UINT64_MAX};
      for (int run = 0; run < kNumRuns; ++run) {
        const auto times = read(file, rowType, query);
        best.readNanos = std::min(best.readNanos, times.readNanos);
        best.c1Nanos = std::min(best.c1Nanos, times.c1Nanos);
      }
      fmt::print(
          "{:<7}{:<11}{:<26}read={:>6.1f}ms  c1_cpu={:>6.1f}ms\n",
          typeName,
          columnName,
          query.name,
          static_cast<double>(best.readNanos) / 1e6,
          static_cast<double>(best.c1Nanos) / 1e6);
    }
  }

  // Uniform random values with 'nullPercent' percent of the rows null at
  // random.
  template <typename T>
  VectorPtr makeNullableColumn(int nullPercent) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> percent(0, 99);
    return makeFlatVector<T>(
        kNumRows,
        [&](auto /*row*/) { return randomValue<T>(rng); },
        [&](auto /*row*/) { return percent(rng) < nullPercent; });
  }

  // Writes c0, the row number, and 'c1' to an in-memory Nimble file.
  std::string writeFile(const VectorPtr& c1) {
    return test::createNimbleFile(
        *pool()->parent(),
        makeRowVector(
            {makeFlatVector<int32_t>(kNumRows, [](auto row) { return row; }),
             c1}));
  }

  // Reads 'file' once with 'query' and returns the read times.
  ReadTimes
  read(const std::string& file, const TypePtr& rowType, const Query& query) {
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    auto* c1 = scanSpec->childByName("c1");
    if (query.isNull) {
      c1->setFilter(std::make_unique<common::IsNull>());
    } else {
      c1->setFilter(std::make_unique<common::IsNotNull>());
    }
    TypePtr outputType = rowType;
    if (!query.projectC1) {
      c1->setProjectOut(false);
      c1->setChannel(common::ScanSpec::kNoChannel);
      outputType = ROW("c0", INTEGER());
    }

    dwio::common::ReaderOptions readerOptions(pool());
    readerOptions.setScanSpec(scanSpec);
    auto reader =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE)
            ->createReader(
                std::make_unique<dwio::common::BufferedInput>(
                    std::make_shared<InMemoryReadFile>(file), *pool()),
                readerOptions);

    // The stripe loads in the timed loop rather than in createRowReader.
    // Nulls-only reads need nimbleScanFastPath and the non-legacy encodings,
    // which stringDecoderZeroCopy selects.
    dwio::common::RowReaderOptions rowReaderOptions;
    rowReaderOptions.setScanSpec(scanSpec);
    rowReaderOptions.setRequestedType(asRowType(rowType));
    rowReaderOptions.setEagerFirstStripeLoad(false);
    rowReaderOptions.setNimbleScanFastPath(true);
    rowReaderOptions.setCollectColumnStats(true);
    auto rowReader = reader->createRowReader(rowReaderOptions);

    auto result = BaseVector::create(outputType, 0, pool());
    uint64_t readNanos{0};
    {
      NanosecondWallTimer timer(&readNanos);
      while (rowReader->next(kBatchSize, result) > 0) {
      }
    }
    dwio::common::RuntimeStats stats;
    rowReader->updateRuntimeStats(stats);
    return {readNanos, c1DecodeNanos(stats)};
  }
};

} // namespace
} // namespace facebook::nimble

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::initializeMemoryManager(
      facebook::velox::memory::MemoryManager::Options{});
  facebook::nimble::NullsOnlyReadBenchmark().run();
  return 0;
}
