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
#include <string>
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
#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/vector/AggregationHook.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

constexpr vector_size_t kBatchSize{4'096};
const auto kInputType = ROW("value", BIGINT());

enum class Grouping : uint8_t {
  kGlobal,
  kPerRow,
};

struct GroupState {
  int64_t sum{0};
  uint8_t nulls{0};
};

class AggregationPushdownBenchmark : public velox::test::VectorTestBase {
 public:
  AggregationPushdownBenchmark() {
    registerSelectiveNimbleReaderFactory();
  }

  ~AggregationPushdownBenchmark() {
    unregisterSelectiveNimbleReaderFactory();
  }

  const std::string& file(EncodingType encodingType, vector_size_t numRows) {
    auto key = std::make_pair(encodingType, numRows);
    auto& file = files_[key];
    if (!file.empty()) {
      return file;
    }

    auto input = makeRowVector(
        {"value"}, {makeFlatVector<int64_t>(numRows, [](auto row) {
          return (static_cast<int64_t>(row) * 9'973) % 1'000;
        })});
    EncodingLayout layout{encodingType, {}, CompressionType::Uncompressed};
    WriterOptions writerOptions;
    writerOptions.encodingLayoutTree.emplace(
        Kind::Row,
        std::unordered_map<
            EncodingLayoutTree::StreamIdentifier,
            EncodingLayout>{},
        "",
        std::vector<EncodingLayoutTree>{EncodingLayoutTree{
            Kind::Scalar, {{0, std::move(layout)}}, "value"}});
    file = test::createNimbleFile(*pool()->parent(), input, writerOptions);
    return file;
  }

  uint64_t
  run(EncodingType encodingType, Grouping grouping, vector_size_t numRows) {
    const auto numGroups = grouping == Grouping::kGlobal ? 1 : kBatchSize;
    std::vector<GroupState> states(numGroups);
    std::vector<char*> groups(kBatchSize);
    for (vector_size_t row = 0; row < kBatchSize; ++row) {
      const auto group = grouping == Grouping::kGlobal ? 0 : row;
      groups[row] = reinterpret_cast<char*>(&states[group]);
    }
    uint64_t numNulls{0};
    aggregate::SumHook<int64_t, false> hook{
        offsetof(GroupState, sum),
        offsetof(GroupState, nulls),
        1,
        groups.data(),
        &numNulls};

    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*kInputType);

    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions readerOptions(pool());
    readerOptions.setScanSpec(scanSpec);
    auto reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(
            std::make_shared<InMemoryReadFile>(file(encodingType, numRows)),
            *pool()),
        readerOptions);
    dwio::common::RowReaderOptions rowReaderOptions;
    rowReaderOptions.setScanSpec(scanSpec);
    rowReaderOptions.setRequestedType(kInputType);
    rowReaderOptions.setEagerFirstStripeLoad(true);
    rowReaderOptions.setStringDecoderZeroCopy(true);
    auto rowReader = reader->createRowReader(rowReaderOptions);
    auto result = BaseVector::create(kInputType, 0, pool());
    std::vector<vector_size_t> rowNumbers(kBatchSize);
    std::iota(rowNumbers.begin(), rowNumbers.end(), 0);

    uint64_t numScanned{0};
    while (true) {
      const auto batchRows = rowReader->next(kBatchSize, result);
      if (batchRows == 0) {
        break;
      }
      numScanned += batchRows;
      auto* lazy = result->asUnchecked<RowVector>()
                       ->childAt(0)
                       ->asUnchecked<LazyVector>();
      lazy->load(RowSet(rowNumbers.data(), result->size()), &hook);
    }
    VELOX_CHECK_EQ(numScanned, numRows);

    uint64_t sum{0};
    for (const auto& state : states) {
      sum += state.sum;
    }
    uint64_t expectedSum{0};
    for (vector_size_t row = 0; row < numRows; ++row) {
      expectedSum += (static_cast<int64_t>(row) * 9'973) % 1'000;
    }
    VELOX_CHECK_EQ(sum, expectedSum);
    return sum;
  }

 private:
  std::map<std::pair<EncodingType, vector_size_t>, std::string> files_;
};

AggregationPushdownBenchmark& benchmarkState() {
  static AggregationPushdownBenchmark benchmark;
  return benchmark;
}

void runBenchmark(
    unsigned iterations,
    EncodingType encodingType,
    Grouping grouping,
    vector_size_t numRows) {
  folly::BenchmarkSuspender suspender;
  for (unsigned iteration = 0; iteration < iterations; ++iteration) {
    suspender.dismiss();
    const auto sum = benchmarkState().run(encodingType, grouping, numRows);
    folly::doNotOptimizeAway(sum);
    suspender.rehire();
  }
}

#define REGISTER_BENCHMARK(encodingName, encodingType, name, mode, rows) \
  BENCHMARK_NAMED_PARAM(                                                 \
      runBenchmark, encodingName##_##name, encodingType, mode, rows)

// 1M rows
REGISTER_BENCHMARK(
    FBW_1M,
    EncodingType::FixedBitWidth,
    GlobalSum,
    Grouping::kGlobal,
    1'000'000);
REGISTER_BENCHMARK(
    BBP_1M,
    EncodingType::BlockBitPacking,
    GlobalSum,
    Grouping::kGlobal,
    1'000'000);
REGISTER_BENCHMARK(
    SIS_1M,
    EncodingType::SubIntSplit,
    GlobalSum,
    Grouping::kGlobal,
    1'000'000);
BENCHMARK_DRAW_LINE();

// 100K rows
REGISTER_BENCHMARK(
    FBW_100K,
    EncodingType::FixedBitWidth,
    GlobalSum,
    Grouping::kGlobal,
    100'000);
REGISTER_BENCHMARK(
    BBP_100K,
    EncodingType::BlockBitPacking,
    GlobalSum,
    Grouping::kGlobal,
    100'000);
REGISTER_BENCHMARK(
    SIS_100K,
    EncodingType::SubIntSplit,
    GlobalSum,
    Grouping::kGlobal,
    100'000);

#undef REGISTER_BENCHMARK

} // namespace
} // namespace facebook::nimble

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::initializeMemoryManager(
      facebook::velox::memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}
