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

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
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
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

constexpr vector_size_t kNumRows{1'000'000};
constexpr int64_t kCardinality{1'000};
const auto kInputType = ROW({"filter_value", "row_id"}, BIGINT());
const auto kFilterOnlyOutputType = ROW("row_id", BIGINT());
const auto kFilterAndProjectOutputType =
    ROW({"filter_value", "row_id"}, BIGINT());

enum class NullPattern : uint8_t {
  kNone,
  kEvery17,
  kLeadingHalf,
  kNumPatterns,
};

bool isNullAt(NullPattern nullPattern, vector_size_t row) {
  switch (nullPattern) {
    case NullPattern::kNone:
      return false;
    case NullPattern::kEvery17:
      return row % 17 == 0;
    case NullPattern::kLeadingHalf:
      return row < kNumRows / 2;
    case NullPattern::kNumPatterns:
      VELOX_UNREACHABLE();
  }
}

int64_t valueAt(vector_size_t row) {
  return (static_cast<int64_t>(row) * 9'973) % kCardinality;
}

struct Readers {
  std::unique_ptr<dwio::common::Reader> reader;
  std::unique_ptr<dwio::common::RowReader> rowReader;
  VectorPtr result;
};

class IntegerDictionaryFusionBenchmark : public velox::test::VectorTestBase {
 public:
  IntegerDictionaryFusionBenchmark() {
    registerSelectiveNimbleReaderFactory();
  }

  ~IntegerDictionaryFusionBenchmark() {
    unregisterSelectiveNimbleReaderFactory();
  }

  Readers createReaders(
      uint32_t selectivity,
      bool applyFilter,
      bool projectFilterColumn,
      bool dictionaryAwareReads,
      EncodingType indicesEncoding,
      NullPattern nullPattern,
      bool nullAllowed) {
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*kInputType);
    const auto numPassingValues = kCardinality * selectivity / 100;
    const auto upperBound = numPassingValues == 0 ? -1 : numPassingValues - 1;
    auto* filterSpec = scanSpec->childByName("filter_value");
    if (applyFilter) {
      filterSpec->setFilter(
          std::make_unique<common::BigintRange>(
              numPassingValues == 0 ? -1 : 0, upperBound, nullAllowed));
    }
    filterSpec->setProjectOut(projectFilterColumn);
    filterSpec->setChannel(
        projectFilterColumn ? 0 : common::ScanSpec::kNoChannel);
    scanSpec->childByName("row_id")->setChannel(projectFilterColumn ? 1 : 0);

    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions readerOptions(pool());
    readerOptions.setScanSpec(scanSpec);

    Readers readers;
    readers.reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(
            std::make_shared<InMemoryReadFile>(
                file(indicesEncoding, nullPattern)),
            *pool()),
        readerOptions);
    dwio::common::RowReaderOptions rowReaderOptions;
    rowReaderOptions.setScanSpec(scanSpec);
    rowReaderOptions.setRequestedType(kInputType);
    rowReaderOptions.setEagerFirstStripeLoad(true);
    rowReaderOptions.setStringDecoderZeroCopy(true);
    rowReaderOptions.setNimbleDictionaryAwareReads(dictionaryAwareReads);
    readers.rowReader = readers.reader->createRowReader(rowReaderOptions);
    readers.result = BaseVector::create(
        projectFilterColumn ? kFilterAndProjectOutputType
                            : kFilterOnlyOutputType,
        0,
        pool());
    return readers;
  }

  uint64_t readAll(
      dwio::common::RowReader& rowReader,
      VectorPtr& result,
      vector_size_t batchSize,
      bool materializeOutput) {
    uint64_t numOutputRows{0};
    uint64_t numScannedRows{0};
    while (true) {
      const auto numScanned = rowReader.next(batchSize, result);
      if (numScanned == 0) {
        break;
      }
      numScannedRows += numScanned;
      numOutputRows += result->size();
      if (materializeOutput) {
        for (const auto& child : result->asUnchecked<RowVector>()->children()) {
          folly::doNotOptimizeAway(child->loadedVector());
        }
      }
    }
    VELOX_CHECK_EQ(numScannedRows, kNumRows);
    return numOutputRows;
  }

 private:
  const std::string& file(
      EncodingType indicesEncoding,
      NullPattern nullPattern) {
    VELOX_CHECK(
        indicesEncoding == EncodingType::Trivial ||
            indicesEncoding == EncodingType::FixedBitWidth,
        "Unsupported dictionary indices encoding: {}",
        static_cast<int32_t>(indicesEncoding));
    auto& files = indicesEncoding == EncodingType::Trivial
        ? trivialIndicesFiles_
        : fixedBitWidthIndicesFiles_;
    auto& file = files[static_cast<size_t>(nullPattern)];
    if (!file.empty()) {
      return file;
    }

    auto input = makeRowVector(
        {"filter_value", "row_id"},
        {makeFlatVector<int64_t>(
             kNumRows,
             valueAt,
             [nullPattern](auto row) { return isNullAt(nullPattern, row); }),
         makeFlatVector<int64_t>(kNumRows, folly::identity)});
    EncodingLayout indicesLayout{
        indicesEncoding, {}, CompressionType::Uncompressed};
    EncodingLayout dictionaryLayout{
        EncodingType::Dictionary,
        {},
        CompressionType::Uncompressed,
        {std::nullopt, std::move(indicesLayout)}};
    WriterOptions writerOptions;
    writerOptions.encodingLayoutTree.emplace(
        Kind::Row,
        std::unordered_map<
            EncodingLayoutTree::StreamIdentifier,
            EncodingLayout>{},
        "",
        std::vector<EncodingLayoutTree>{
            EncodingLayoutTree{
                Kind::Scalar,
                {{0, std::move(dictionaryLayout)}},
                "filter_value"},
            EncodingLayoutTree{Kind::Scalar, {}, "row_id"},
        });
    file = test::createNimbleFile(*pool()->parent(), input, writerOptions);
    return file;
  }

  std::array<std::string, static_cast<size_t>(NullPattern::kNumPatterns)>
      trivialIndicesFiles_;
  std::array<std::string, static_cast<size_t>(NullPattern::kNumPatterns)>
      fixedBitWidthIndicesFiles_;
};

IntegerDictionaryFusionBenchmark& benchmarkState() {
  static IntegerDictionaryFusionBenchmark benchmark;
  return benchmark;
}

void runBenchmark(
    unsigned iterations,
    vector_size_t batchSize,
    uint32_t selectivity,
    bool applyFilter,
    bool projectFilterColumn,
    bool dictionaryAwareReads,
    EncodingType indicesEncoding,
    NullPattern nullPattern,
    bool nullAllowed) {
  folly::BenchmarkSuspender suspender;
  const auto numPassingValues = kCardinality * selectivity / 100;
  uint64_t expectedRows{0};
  for (vector_size_t row = 0; row < kNumRows; ++row) {
    const auto isNull = isNullAt(nullPattern, row);
    if (!applyFilter ||
        (isNull ? nullAllowed : valueAt(row) < numPassingValues)) {
      ++expectedRows;
    }
  }
  for (unsigned iteration = 0; iteration < iterations; ++iteration) {
    auto readers = benchmarkState().createReaders(
        selectivity,
        applyFilter,
        projectFilterColumn,
        dictionaryAwareReads,
        indicesEncoding,
        nullPattern,
        nullAllowed);
    suspender.dismiss();
    // Leave filter-only output lazy so timing measures filter evaluation and
    // row selection. Materialize all projected children to include their
    // decoding cost.
    const auto numOutputRows = benchmarkState().readAll(
        *readers.rowReader, readers.result, batchSize, projectFilterColumn);
    folly::doNotOptimizeAway(numOutputRows);
    suspender.rehire();
    VELOX_CHECK_EQ(numOutputRows, expectedRows);
  }
}

void baseline(
    unsigned iterations,
    vector_size_t batchSize,
    uint32_t selectivity,
    bool projectFilterColumn,
    EncodingType indicesEncoding) {
  runBenchmark(
      iterations,
      batchSize,
      selectivity,
      /*applyFilter=*/true,
      projectFilterColumn,
      /*dictionaryAwareReads=*/false,
      indicesEncoding,
      NullPattern::kNone,
      /*nullAllowed=*/false);
}

void dictionaryAware(
    unsigned iterations,
    vector_size_t batchSize,
    uint32_t selectivity,
    bool projectFilterColumn,
    EncodingType indicesEncoding) {
  runBenchmark(
      iterations,
      batchSize,
      selectivity,
      /*applyFilter=*/true,
      projectFilterColumn,
      /*dictionaryAwareReads=*/true,
      indicesEncoding,
      NullPattern::kNone,
      /*nullAllowed=*/false);
}

void projectionBaseline(
    unsigned iterations,
    vector_size_t batchSize,
    EncodingType indicesEncoding) {
  runBenchmark(
      iterations,
      batchSize,
      /*selectivity=*/100,
      /*applyFilter=*/false,
      /*projectFilterColumn=*/true,
      /*dictionaryAwareReads=*/false,
      indicesEncoding,
      NullPattern::kNone,
      /*nullAllowed=*/false);
}

void projectionDictionaryAware(
    unsigned iterations,
    vector_size_t batchSize,
    EncodingType indicesEncoding) {
  runBenchmark(
      iterations,
      batchSize,
      /*selectivity=*/100,
      /*applyFilter=*/false,
      /*projectFilterColumn=*/true,
      /*dictionaryAwareReads=*/true,
      indicesEncoding,
      NullPattern::kNone,
      /*nullAllowed=*/false);
}

void nullableProjectionBaseline(
    unsigned iterations,
    vector_size_t batchSize,
    NullPattern nullPattern) {
  runBenchmark(
      iterations,
      batchSize,
      /*selectivity=*/100,
      /*applyFilter=*/false,
      /*projectFilterColumn=*/true,
      /*dictionaryAwareReads=*/false,
      EncodingType::FixedBitWidth,
      nullPattern,
      /*nullAllowed=*/false);
}

void nullableProjectionDictionaryAware(
    unsigned iterations,
    vector_size_t batchSize,
    NullPattern nullPattern) {
  runBenchmark(
      iterations,
      batchSize,
      /*selectivity=*/100,
      /*applyFilter=*/false,
      /*projectFilterColumn=*/true,
      /*dictionaryAwareReads=*/true,
      EncodingType::FixedBitWidth,
      nullPattern,
      /*nullAllowed=*/false);
}

void nullableFilterOnlyBaseline(
    unsigned iterations,
    vector_size_t batchSize,
    NullPattern nullPattern) {
  runBenchmark(
      iterations,
      batchSize,
      /*selectivity=*/50,
      /*applyFilter=*/true,
      /*projectFilterColumn=*/false,
      /*dictionaryAwareReads=*/false,
      EncodingType::FixedBitWidth,
      nullPattern,
      /*nullAllowed=*/true);
}

void nullableFilterOnlyDictionaryAware(
    unsigned iterations,
    vector_size_t batchSize,
    NullPattern nullPattern) {
  runBenchmark(
      iterations,
      batchSize,
      /*selectivity=*/50,
      /*applyFilter=*/true,
      /*projectFilterColumn=*/false,
      /*dictionaryAwareReads=*/true,
      EncodingType::FixedBitWidth,
      nullPattern,
      /*nullAllowed=*/true);
}

#define REGISTER_FILTER_BENCHMARK(                                     \
    encodingName, encodingType, name, batchSize, selectivity, project) \
  BENCHMARK_NAMED_PARAM(                                               \
      baseline,                                                        \
      encodingName##_##name##_B##batchSize##_S##selectivity,           \
      batchSize,                                                       \
      selectivity,                                                     \
      project,                                                         \
      encodingType);                                                   \
  BENCHMARK_RELATIVE_NAMED_PARAM(                                      \
      dictionaryAware,                                                 \
      encodingName##_##name##_B##batchSize##_S##selectivity,           \
      batchSize,                                                       \
      selectivity,                                                     \
      project,                                                         \
      encodingType)

#define REGISTER_PROJECTION_BENCHMARK(encodingName, encodingType, batchSize) \
  BENCHMARK_NAMED_PARAM(                                                     \
      projectionBaseline,                                                    \
      encodingName##_ProjectionOnly_B##batchSize,                            \
      batchSize,                                                             \
      encodingType);                                                         \
  BENCHMARK_RELATIVE_NAMED_PARAM(                                            \
      projectionDictionaryAware,                                             \
      encodingName##_ProjectionOnly_B##batchSize,                            \
      batchSize,                                                             \
      encodingType)

#define REGISTER_ENCODING_BENCHMARKS(encodingName, encodingType)      \
  REGISTER_PROJECTION_BENCHMARK(encodingName, encodingType, 1024);    \
  REGISTER_PROJECTION_BENCHMARK(encodingName, encodingType, 4096);    \
  REGISTER_PROJECTION_BENCHMARK(encodingName, encodingType, 10000);   \
  BENCHMARK_DRAW_LINE();                                              \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 1024, 0, false);        \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 1024, 1, false);        \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 1024, 10, false);       \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 1024, 50, false);       \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 1024, 90, false);       \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 1024, 100, false);      \
  BENCHMARK_DRAW_LINE();                                              \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 4096, 0, false);        \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 4096, 1, false);        \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 4096, 10, false);       \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 4096, 50, false);       \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 4096, 90, false);       \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 4096, 100, false);      \
  BENCHMARK_DRAW_LINE();                                              \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 10000, 0, false);       \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 10000, 1, false);       \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 10000, 10, false);      \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 10000, 50, false);      \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 10000, 90, false);      \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterOnly, 10000, 100, false);     \
  BENCHMARK_DRAW_LINE();                                              \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 1024, 0, true);   \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 1024, 1, true);   \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 1024, 10, true);  \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 1024, 50, true);  \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 1024, 90, true);  \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 1024, 100, true); \
  BENCHMARK_DRAW_LINE();                                              \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 4096, 0, true);   \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 4096, 1, true);   \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 4096, 10, true);  \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 4096, 50, true);  \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 4096, 90, true);  \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 4096, 100, true); \
  BENCHMARK_DRAW_LINE();                                              \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 10000, 0, true);  \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 10000, 1, true);  \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 10000, 10, true); \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 10000, 50, true); \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 10000, 90, true); \
  REGISTER_FILTER_BENCHMARK(                                          \
      encodingName, encodingType, FilterAndProject, 10000, 100, true)

REGISTER_ENCODING_BENCHMARKS(Trivial, EncodingType::Trivial);

BENCHMARK_DRAW_LINE();

REGISTER_ENCODING_BENCHMARKS(FixedBitWidth, EncodingType::FixedBitWidth);

BENCHMARK_DRAW_LINE();

#define REGISTER_NULLABLE_BENCHMARKS(name, nullPattern)                        \
  BENCHMARK_NAMED_PARAM(                                                       \
      nullableProjectionBaseline, name##_Projection_B4096, 4096, nullPattern); \
  BENCHMARK_RELATIVE_NAMED_PARAM(                                              \
      nullableProjectionDictionaryAware,                                       \
      name##_Projection_B4096,                                                 \
      4096,                                                                    \
      nullPattern);                                                            \
  BENCHMARK_NAMED_PARAM(                                                       \
      nullableFilterOnlyBaseline, name##_FilterOnly_B4096, 4096, nullPattern); \
  BENCHMARK_RELATIVE_NAMED_PARAM(                                              \
      nullableFilterOnlyDictionaryAware,                                       \
      name##_FilterOnly_B4096,                                                 \
      4096,                                                                    \
      nullPattern)

REGISTER_NULLABLE_BENCHMARKS(NullableEvery17, NullPattern::kEvery17);
BENCHMARK_DRAW_LINE();
REGISTER_NULLABLE_BENCHMARKS(NullableLeadingHalf, NullPattern::kLeadingHalf);

#undef REGISTER_ENCODING_BENCHMARKS
#undef REGISTER_PROJECTION_BENCHMARK
#undef REGISTER_FILTER_BENCHMARK
#undef REGISTER_NULLABLE_BENCHMARKS

} // namespace
} // namespace facebook::nimble

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::initializeMemoryManager(
      facebook::velox::memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}
