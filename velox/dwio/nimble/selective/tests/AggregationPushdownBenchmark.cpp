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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/common/file/tests/TestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitBoundaries.h"
#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/vector/AggregationHook.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

constexpr vector_size_t kBatchSize{4'096};
constexpr int64_t kCardinality{1'000};
// Values below this pass the filter-only workload's filter: half the rows.
constexpr int64_t kNumPassingValues{kCardinality / 2};
const auto kInputType = ROW({"value", "row_id"}, BIGINT());
const auto kFilterOnlyOutputType = ROW("row_id", BIGINT());

// How 'value' is stored.
enum class Layout : uint8_t {
  // Encoded with the encoding under test.
  kPlain,
  // Dictionary encoded, with the encoding under test for the indices.
  kDictionary,
};

// What the timed read does with 'value'.
enum class Workload : uint8_t {
  // SUM through a hook into a single group.
  kGlobalSum,
  // SUM through a hook with one group per row.
  kPerRowSum,
  // Loads every row.
  kProjection,
  // Filters on 'value' without projecting it.
  kFilterOnly,
};

struct BenchmarkConfig {
  EncodingType encodingType;
  Layout layout;
  Workload workload;
  vector_size_t numRows;
  bool scanFastPath;
};

struct GroupState {
  int64_t sum{0};
  uint8_t nulls{0};
};

struct Readers {
  std::unique_ptr<dwio::common::Reader> reader;
  std::unique_ptr<dwio::common::RowReader> rowReader;
  VectorPtr result;
};

// Returns a layout of 'encodingType' for 'numBits'-bit values. The writer
// silently re-selects the encoding when a layout fails to encode, so the layout
// must be complete: BlockBitPacking needs its three child slots, and
// SubIntSplit needs sections that cover every bit.
EncodingLayout makeLayout(EncodingType encodingType, int numBits) {
  if (encodingType == EncodingType::BlockBitPacking) {
    return EncodingLayout{
        encodingType,
        {},
        CompressionType::Uncompressed,
        {std::nullopt, std::nullopt, std::nullopt}};
  }
  if (encodingType == EncodingType::SubIntSplit) {
    const std::vector<subintsplit::SectionPlan> sections{
        {.bitStart = 0, .bitEnd = 15},
        {.bitStart = 16, .bitEnd = numBits - 1},
    };
    return EncodingLayout{
        encodingType,
        EncodingLayout::Config{subintsplit::makePreserveSplitConfig(sections)},
        CompressionType::Uncompressed,
        std::vector<std::optional<const EncodingLayout>>(sections.size())};
  }
  return EncodingLayout{encodingType, {}, CompressionType::Uncompressed};
}

int64_t valueAt(vector_size_t row) {
  return (static_cast<int64_t>(row) * 9'973) % kCardinality;
}

bool isSum(Workload workload) {
  return workload == Workload::kGlobalSum || workload == Workload::kPerRowSum;
}

// Returns what a read of 'config' returns: the SUM for the SUM workloads and
// the number of output rows for the others.
uint64_t expectedResult(const BenchmarkConfig& config) {
  uint64_t result{0};
  for (vector_size_t row = 0; row < config.numRows; ++row) {
    if (isSum(config.workload)) {
      result += valueAt(row);
    } else if (
        config.workload == Workload::kProjection ||
        valueAt(row) < kNumPassingValues) {
      ++result;
    }
  }
  return result;
}

class AggregationPushdownBenchmark : public velox::test::VectorTestBase {
 public:
  AggregationPushdownBenchmark() {
    registerSelectiveNimbleReaderFactory();
  }

  ~AggregationPushdownBenchmark() {
    unregisterSelectiveNimbleReaderFactory();
  }

  Readers createReaders(const BenchmarkConfig& config) {
    const bool filterOnly = config.workload == Workload::kFilterOnly;
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*kInputType);
    if (filterOnly) {
      auto* valueSpec = scanSpec->childByName("value");
      valueSpec->setFilter(
          std::make_unique<common::BigintRange>(
              0, kNumPassingValues - 1, /*nullAllowed=*/false));
      valueSpec->setProjectOut(false);
      valueSpec->setChannel(common::ScanSpec::kNoChannel);
      scanSpec->childByName("row_id")->setChannel(0);
    }

    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions readerOptions(pool());
    readerOptions.setScanSpec(scanSpec);
    Readers readers;
    readers.reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(
            std::make_shared<InMemoryReadFile>(file(config)), *pool()),
        readerOptions);
    dwio::common::RowReaderOptions rowReaderOptions;
    rowReaderOptions.setScanSpec(scanSpec);
    rowReaderOptions.setRequestedType(kInputType);
    rowReaderOptions.setEagerFirstStripeLoad(true);
    rowReaderOptions.setStringDecoderZeroCopy(true);
    rowReaderOptions.setNimbleScanFastPath(config.scanFastPath);
    readers.rowReader = readers.reader->createRowReader(rowReaderOptions);
    readers.result = BaseVector::create(
        filterOnly ? kFilterOnlyOutputType : kInputType, 0, pool());
    return readers;
  }

  // Reads every row of 'readers' for 'workload'. Returns the SUM for the SUM
  // workloads and the number of output rows for the others.
  uint64_t read(Readers& readers, Workload workload) {
    if (isSum(workload)) {
      return sum(readers, workload);
    }
    uint64_t numOutputRows{0};
    while (readers.rowReader->next(kBatchSize, readers.result) > 0) {
      numOutputRows += readers.result->size();
      // Filter-only output leaves 'row_id' lazy, so only filter evaluation and
      // row selection are timed.
      if (workload == Workload::kProjection) {
        folly::doNotOptimizeAway(readers.result->asUnchecked<RowVector>()
                                     ->childAt(0)
                                     ->loadedVector());
      }
    }
    return numOutputRows;
  }

 private:
  const std::string& file(const BenchmarkConfig& config) {
    auto& file = files_[std::make_tuple(
        config.encodingType, config.layout, config.numRows)];
    if (!file.empty()) {
      return file;
    }

    auto input = makeRowVector(
        {"value", "row_id"},
        {makeFlatVector<int64_t>(config.numRows, valueAt),
         makeFlatVector<int64_t>(config.numRows, folly::identity)});
    // Dictionary indices are 32-bit; 'value' is 64-bit.
    auto layout = makeLayout(config.encodingType, /*numBits=*/64);
    if (config.layout == Layout::kDictionary) {
      layout = EncodingLayout{
          EncodingType::Dictionary,
          {},
          CompressionType::Uncompressed,
          {std::nullopt, makeLayout(config.encodingType, /*numBits=*/32)}};
    }
    WriterOptions writerOptions;
    writerOptions.encodingLayoutTree.emplace(
        Kind::Row,
        std::unordered_map<
            EncodingLayoutTree::StreamIdentifier,
            EncodingLayout>{},
        "",
        std::vector<EncodingLayoutTree>{
            EncodingLayoutTree{Kind::Scalar, {{0, std::move(layout)}}, "value"},
            EncodingLayoutTree{Kind::Scalar, {}, "row_id"},
        });
    file = test::createNimbleFile(*pool()->parent(), input, writerOptions);
    return file;
  }

  uint64_t sum(Readers& readers, Workload workload) {
    const bool global = workload == Workload::kGlobalSum;
    std::vector<GroupState> states(global ? 1 : kBatchSize);
    std::vector<char*> groups(kBatchSize);
    for (vector_size_t row = 0; row < kBatchSize; ++row) {
      groups[row] = reinterpret_cast<char*>(&states[global ? 0 : row]);
    }
    uint64_t numNulls{0};
    aggregate::SumHook<int64_t, false> hook{
        offsetof(GroupState, sum),
        offsetof(GroupState, nulls),
        1,
        groups.data(),
        &numNulls};
    std::vector<vector_size_t> rowNumbers(kBatchSize);
    std::iota(rowNumbers.begin(), rowNumbers.end(), 0);

    while (readers.rowReader->next(kBatchSize, readers.result) > 0) {
      auto* lazy = readers.result->asUnchecked<RowVector>()
                       ->childAt(0)
                       ->asUnchecked<LazyVector>();
      lazy->load(RowSet(rowNumbers.data(), readers.result->size()), &hook);
    }

    uint64_t sum{0};
    for (const auto& state : states) {
      sum += state.sum;
    }
    return sum;
  }

  std::map<std::tuple<EncodingType, Layout, vector_size_t>, std::string> files_;
};

AggregationPushdownBenchmark& benchmarkState() {
  static AggregationPushdownBenchmark benchmark;
  return benchmark;
}

// Times only the read. Writing the file, creating the readers and computing
// the expected result stay outside the timed region.
void runBenchmark(unsigned iterations, const BenchmarkConfig& config) {
  folly::BenchmarkSuspender suspender;
  const auto expected = expectedResult(config);
  for (unsigned iteration = 0; iteration < iterations; ++iteration) {
    auto readers = benchmarkState().createReaders(config);
    suspender.dismiss();
    const auto result = benchmarkState().read(readers, config.workload);
    folly::doNotOptimizeAway(result);
    suspender.rehire();
    VELOX_CHECK_EQ(result, expected);
  }
}

// Registers 'workload' on a 'layout' column of 'numRows' rows encoded with
// 'encodingType': once with scanFastPath off, then once with it on, timed
// relative to the first.
#define REGISTER_ENCODING_BENCHMARKS(                                        \
    prefix, encodingType, name, layout, workload, numRows)                   \
  BENCHMARK_NAMED_PARAM(                                                     \
      runBenchmark,                                                          \
      prefix##_##name,                                                       \
      BenchmarkConfig{                                                       \
          encodingType, layout, workload, numRows, /*scanFastPath=*/false}); \
  BENCHMARK_RELATIVE_NAMED_PARAM(                                            \
      runBenchmark,                                                          \
      prefix##_##name##_ScanFastPath,                                        \
      BenchmarkConfig{                                                       \
          encodingType, layout, workload, numRows, /*scanFastPath=*/true})

// Registers 'workload' on a 'layout' column of 'numRows' rows once per
// encoding under test.
#define REGISTER_BENCHMARKS(name, layout, workload, numRows)                \
  REGISTER_ENCODING_BENCHMARKS(                                             \
      FBW, EncodingType::FixedBitWidth, name, layout, workload, numRows);   \
  REGISTER_ENCODING_BENCHMARKS(                                             \
      BBP, EncodingType::BlockBitPacking, name, layout, workload, numRows); \
  REGISTER_ENCODING_BENCHMARKS(                                             \
      SIS, EncodingType::SubIntSplit, name, layout, workload, numRows)

REGISTER_BENCHMARKS(
    1M_GlobalSum,
    Layout::kPlain,
    Workload::kGlobalSum,
    1'000'000);
REGISTER_BENCHMARKS(
    1M_PerRowSum,
    Layout::kPlain,
    Workload::kPerRowSum,
    1'000'000);
REGISTER_BENCHMARKS(
    1M_Projection,
    Layout::kPlain,
    Workload::kProjection,
    1'000'000);
REGISTER_BENCHMARKS(
    1M_DictionaryProjection,
    Layout::kDictionary,
    Workload::kProjection,
    1'000'000);
REGISTER_BENCHMARKS(
    1M_FilterOnly,
    Layout::kPlain,
    Workload::kFilterOnly,
    1'000'000);
BENCHMARK_DRAW_LINE();

REGISTER_BENCHMARKS(
    100K_GlobalSum,
    Layout::kPlain,
    Workload::kGlobalSum,
    100'000);
REGISTER_BENCHMARKS(
    100K_PerRowSum,
    Layout::kPlain,
    Workload::kPerRowSum,
    100'000);
REGISTER_BENCHMARKS(
    100K_Projection,
    Layout::kPlain,
    Workload::kProjection,
    100'000);
REGISTER_BENCHMARKS(
    100K_DictionaryProjection,
    Layout::kDictionary,
    Workload::kProjection,
    100'000);
REGISTER_BENCHMARKS(
    100K_FilterOnly,
    Layout::kPlain,
    Workload::kFilterOnly,
    100'000);

#undef REGISTER_BENCHMARKS
#undef REGISTER_ENCODING_BENCHMARKS

} // namespace
} // namespace facebook::nimble

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::initializeMemoryManager(
      facebook::velox::memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}
