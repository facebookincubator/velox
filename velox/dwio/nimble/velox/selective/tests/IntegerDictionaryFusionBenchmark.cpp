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
#include <type_traits>
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
const auto kFilterOnlyOutputType = ROW("row_id", BIGINT());

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

template <typename T>
T dictionaryValue(int64_t index) {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(index) / 4 - 125.5;
  } else {
    return index;
  }
}

template <typename T>
std::unique_ptr<common::Filter> rangeFilter(
    int64_t numPassingValues,
    bool nullAllowed) {
  const auto lower = dictionaryValue<T>(numPassingValues == 0 ? -1 : 0);
  const auto upper = dictionaryValue<T>(numPassingValues - 1);
  if constexpr (std::is_floating_point_v<T>) {
    return std::make_unique<common::FloatingPointRange<T>>(
        lower, false, false, upper, false, false, nullAllowed);
  } else {
    return std::make_unique<common::BigintRange>(lower, upper, nullAllowed);
  }
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
      bool nullAllowed,
      TypeKind valueType) {
    const auto inputType = ROW(
        {"filter_value", "row_id"}, {createScalarType(valueType), BIGINT()});
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*inputType);
    const auto numPassingValues = kCardinality * selectivity / 100;
    auto* filterSpec = scanSpec->childByName("filter_value");
    if (applyFilter) {
      filterSpec->setFilter(
          valueType == TypeKind::REAL
              ? rangeFilter<float>(numPassingValues, nullAllowed)
              : valueType == TypeKind::DOUBLE
              ? rangeFilter<double>(numPassingValues, nullAllowed)
              : rangeFilter<int64_t>(numPassingValues, nullAllowed));
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
                file(valueType, indicesEncoding, nullPattern)),
            *pool()),
        readerOptions);
    dwio::common::RowReaderOptions rowReaderOptions;
    rowReaderOptions.setScanSpec(scanSpec);
    rowReaderOptions.setRequestedType(inputType);
    rowReaderOptions.setEagerFirstStripeLoad(true);
    rowReaderOptions.setStringDecoderZeroCopy(true);
    rowReaderOptions.setNimbleDictionaryAwareReads(dictionaryAwareReads);
    readers.rowReader = readers.reader->createRowReader(rowReaderOptions);
    readers.result = BaseVector::create(
        projectFilterColumn ? inputType : kFilterOnlyOutputType, 0, pool());
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
      TypeKind valueType,
      EncodingType indicesEncoding,
      NullPattern nullPattern) {
    switch (valueType) {
      case TypeKind::REAL:
        return file<float>(indicesEncoding, nullPattern);
      case TypeKind::DOUBLE:
        return file<double>(indicesEncoding, nullPattern);
      case TypeKind::BIGINT:
        return file<int64_t>(indicesEncoding, nullPattern);
      default:
        VELOX_UNREACHABLE();
    }
  }

  template <typename T>
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
    auto& file =
        files[CppToType<T>::typeKind][static_cast<size_t>(nullPattern)];
    if (!file.empty()) {
      return file;
    }

    auto input = makeRowVector(
        {"filter_value", "row_id"},
        {makeFlatVector<T>(
             kNumRows,
             [](auto row) { return dictionaryValue<T>(valueAt(row)); },
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

  using Files = std::unordered_map<
      TypeKind,
      std::array<std::string, static_cast<size_t>(NullPattern::kNumPatterns)>>;
  Files trivialIndicesFiles_;
  Files fixedBitWidthIndicesFiles_;
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
    bool nullAllowed,
    TypeKind valueType) {
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
        nullAllowed,
        valueType);
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
    EncodingType indicesEncoding,
    TypeKind valueType = TypeKind::BIGINT) {
  runBenchmark(
      iterations,
      batchSize,
      selectivity,
      /*applyFilter=*/true,
      projectFilterColumn,
      /*dictionaryAwareReads=*/false,
      indicesEncoding,
      NullPattern::kNone,
      /*nullAllowed=*/false,
      valueType);
}

void dictionaryAware(
    unsigned iterations,
    vector_size_t batchSize,
    uint32_t selectivity,
    bool projectFilterColumn,
    EncodingType indicesEncoding,
    TypeKind valueType = TypeKind::BIGINT) {
  runBenchmark(
      iterations,
      batchSize,
      selectivity,
      /*applyFilter=*/true,
      projectFilterColumn,
      /*dictionaryAwareReads=*/true,
      indicesEncoding,
      NullPattern::kNone,
      /*nullAllowed=*/false,
      valueType);
}

void projectionBaseline(
    unsigned iterations,
    vector_size_t batchSize,
    EncodingType indicesEncoding,
    TypeKind valueType = TypeKind::BIGINT) {
  runBenchmark(
      iterations,
      batchSize,
      /*selectivity=*/100,
      /*applyFilter=*/false,
      /*projectFilterColumn=*/true,
      /*dictionaryAwareReads=*/false,
      indicesEncoding,
      NullPattern::kNone,
      /*nullAllowed=*/false,
      valueType);
}

void projectionDictionaryAware(
    unsigned iterations,
    vector_size_t batchSize,
    EncodingType indicesEncoding,
    TypeKind valueType = TypeKind::BIGINT) {
  runBenchmark(
      iterations,
      batchSize,
      /*selectivity=*/100,
      /*applyFilter=*/false,
      /*projectFilterColumn=*/true,
      /*dictionaryAwareReads=*/true,
      indicesEncoding,
      NullPattern::kNone,
      /*nullAllowed=*/false,
      valueType);
}

void nullableProjectionBaseline(
    unsigned iterations,
    vector_size_t batchSize,
    NullPattern nullPattern,
    TypeKind valueType = TypeKind::BIGINT) {
  runBenchmark(
      iterations,
      batchSize,
      /*selectivity=*/100,
      /*applyFilter=*/false,
      /*projectFilterColumn=*/true,
      /*dictionaryAwareReads=*/false,
      EncodingType::FixedBitWidth,
      nullPattern,
      /*nullAllowed=*/false,
      valueType);
}

void nullableProjectionDictionaryAware(
    unsigned iterations,
    vector_size_t batchSize,
    NullPattern nullPattern,
    TypeKind valueType = TypeKind::BIGINT) {
  runBenchmark(
      iterations,
      batchSize,
      /*selectivity=*/100,
      /*applyFilter=*/false,
      /*projectFilterColumn=*/true,
      /*dictionaryAwareReads=*/true,
      EncodingType::FixedBitWidth,
      nullPattern,
      /*nullAllowed=*/false,
      valueType);
}

void nullableFilterOnlyBaseline(
    unsigned iterations,
    vector_size_t batchSize,
    NullPattern nullPattern,
    TypeKind valueType = TypeKind::BIGINT) {
  runBenchmark(
      iterations,
      batchSize,
      /*selectivity=*/50,
      /*applyFilter=*/true,
      /*projectFilterColumn=*/false,
      /*dictionaryAwareReads=*/false,
      EncodingType::FixedBitWidth,
      nullPattern,
      /*nullAllowed=*/true,
      valueType);
}

void nullableFilterOnlyDictionaryAware(
    unsigned iterations,
    vector_size_t batchSize,
    NullPattern nullPattern,
    TypeKind valueType = TypeKind::BIGINT) {
  runBenchmark(
      iterations,
      batchSize,
      /*selectivity=*/50,
      /*applyFilter=*/true,
      /*projectFilterColumn=*/false,
      /*dictionaryAwareReads=*/true,
      EncodingType::FixedBitWidth,
      nullPattern,
      /*nullAllowed=*/true,
      valueType);
}

#define REGISTER_FILTER_BENCHMARK(                           \
    encodingName,                                            \
    encodingType,                                            \
    valueType,                                               \
    name,                                                    \
    batchSize,                                               \
    selectivity,                                             \
    project)                                                 \
  BENCHMARK_NAMED_PARAM(                                     \
      baseline,                                              \
      encodingName##_##name##_B##batchSize##_S##selectivity, \
      batchSize,                                             \
      selectivity,                                           \
      project,                                               \
      encodingType,                                          \
      valueType);                                            \
  BENCHMARK_RELATIVE_NAMED_PARAM(                            \
      dictionaryAware,                                       \
      encodingName##_##name##_B##batchSize##_S##selectivity, \
      batchSize,                                             \
      selectivity,                                           \
      project,                                               \
      encodingType,                                          \
      valueType)

#define REGISTER_PROJECTION_BENCHMARK(                \
    encodingName, encodingType, valueType, batchSize) \
  BENCHMARK_NAMED_PARAM(                              \
      projectionBaseline,                             \
      encodingName##_ProjectionOnly_B##batchSize,     \
      batchSize,                                      \
      encodingType,                                   \
      valueType);                                     \
  BENCHMARK_RELATIVE_NAMED_PARAM(                     \
      projectionDictionaryAware,                      \
      encodingName##_ProjectionOnly_B##batchSize,     \
      batchSize,                                      \
      encodingType,                                   \
      valueType)

#define REGISTER_ENCODING_BENCHMARKS(encodingName, encodingType, valueType)    \
  REGISTER_PROJECTION_BENCHMARK(encodingName, encodingType, valueType, 1024);  \
  REGISTER_PROJECTION_BENCHMARK(encodingName, encodingType, valueType, 4096);  \
  REGISTER_PROJECTION_BENCHMARK(encodingName, encodingType, valueType, 10000); \
  BENCHMARK_DRAW_LINE();                                                       \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 1024, 0, false);      \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 1024, 1, false);      \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 1024, 10, false);     \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 1024, 50, false);     \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 1024, 90, false);     \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 1024, 100, false);    \
  BENCHMARK_DRAW_LINE();                                                       \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 4096, 0, false);      \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 4096, 1, false);      \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 4096, 10, false);     \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 4096, 50, false);     \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 4096, 90, false);     \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 4096, 100, false);    \
  BENCHMARK_DRAW_LINE();                                                       \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 10000, 0, false);     \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 10000, 1, false);     \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 10000, 10, false);    \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 10000, 50, false);    \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 10000, 90, false);    \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterOnly, 10000, 100, false);   \
  BENCHMARK_DRAW_LINE();                                                       \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterAndProject, 1024, 0, true); \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterAndProject, 1024, 1, true); \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      1024,                                                                    \
      10,                                                                      \
      true);                                                                   \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      1024,                                                                    \
      50,                                                                      \
      true);                                                                   \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      1024,                                                                    \
      90,                                                                      \
      true);                                                                   \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      1024,                                                                    \
      100,                                                                     \
      true);                                                                   \
  BENCHMARK_DRAW_LINE();                                                       \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterAndProject, 4096, 0, true); \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName, encodingType, valueType, FilterAndProject, 4096, 1, true); \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      4096,                                                                    \
      10,                                                                      \
      true);                                                                   \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      4096,                                                                    \
      50,                                                                      \
      true);                                                                   \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      4096,                                                                    \
      90,                                                                      \
      true);                                                                   \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      4096,                                                                    \
      100,                                                                     \
      true);                                                                   \
  BENCHMARK_DRAW_LINE();                                                       \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      10000,                                                                   \
      0,                                                                       \
      true);                                                                   \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      10000,                                                                   \
      1,                                                                       \
      true);                                                                   \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      10000,                                                                   \
      10,                                                                      \
      true);                                                                   \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      10000,                                                                   \
      50,                                                                      \
      true);                                                                   \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      10000,                                                                   \
      90,                                                                      \
      true);                                                                   \
  REGISTER_FILTER_BENCHMARK(                                                   \
      encodingName,                                                            \
      encodingType,                                                            \
      valueType,                                                               \
      FilterAndProject,                                                        \
      10000,                                                                   \
      100,                                                                     \
      true)

REGISTER_ENCODING_BENCHMARKS(Trivial, EncodingType::Trivial, TypeKind::BIGINT);

BENCHMARK_DRAW_LINE();

REGISTER_ENCODING_BENCHMARKS(
    FixedBitWidth,
    EncodingType::FixedBitWidth,
    TypeKind::BIGINT);
BENCHMARK_DRAW_LINE();
REGISTER_ENCODING_BENCHMARKS(
    Float_Trivial,
    EncodingType::Trivial,
    TypeKind::REAL);
BENCHMARK_DRAW_LINE();
REGISTER_ENCODING_BENCHMARKS(
    Float_FixedBitWidth,
    EncodingType::FixedBitWidth,
    TypeKind::REAL);
BENCHMARK_DRAW_LINE();
REGISTER_ENCODING_BENCHMARKS(
    Double_Trivial,
    EncodingType::Trivial,
    TypeKind::DOUBLE);
BENCHMARK_DRAW_LINE();
REGISTER_ENCODING_BENCHMARKS(
    Double_FixedBitWidth,
    EncodingType::FixedBitWidth,
    TypeKind::DOUBLE);

BENCHMARK_DRAW_LINE();

#define REGISTER_NULLABLE_BENCHMARKS(name, nullPattern, valueType) \
  BENCHMARK_NAMED_PARAM(                                           \
      nullableProjectionBaseline,                                  \
      name##_Projection_B4096,                                     \
      4096,                                                        \
      nullPattern,                                                 \
      valueType);                                                  \
  BENCHMARK_RELATIVE_NAMED_PARAM(                                  \
      nullableProjectionDictionaryAware,                           \
      name##_Projection_B4096,                                     \
      4096,                                                        \
      nullPattern,                                                 \
      valueType);                                                  \
  BENCHMARK_NAMED_PARAM(                                           \
      nullableFilterOnlyBaseline,                                  \
      name##_FilterOnly_B4096,                                     \
      4096,                                                        \
      nullPattern,                                                 \
      valueType);                                                  \
  BENCHMARK_RELATIVE_NAMED_PARAM(                                  \
      nullableFilterOnlyDictionaryAware,                           \
      name##_FilterOnly_B4096,                                     \
      4096,                                                        \
      nullPattern,                                                 \
      valueType)

REGISTER_NULLABLE_BENCHMARKS(
    NullableEvery17,
    NullPattern::kEvery17,
    TypeKind::BIGINT);
BENCHMARK_DRAW_LINE();
REGISTER_NULLABLE_BENCHMARKS(
    NullableLeadingHalf,
    NullPattern::kLeadingHalf,
    TypeKind::BIGINT);
BENCHMARK_DRAW_LINE();
REGISTER_NULLABLE_BENCHMARKS(
    Float_NullableEvery17,
    NullPattern::kEvery17,
    TypeKind::REAL);
BENCHMARK_DRAW_LINE();
REGISTER_NULLABLE_BENCHMARKS(
    Float_NullableLeadingHalf,
    NullPattern::kLeadingHalf,
    TypeKind::REAL);
BENCHMARK_DRAW_LINE();
REGISTER_NULLABLE_BENCHMARKS(
    Double_NullableEvery17,
    NullPattern::kEvery17,
    TypeKind::DOUBLE);
BENCHMARK_DRAW_LINE();
REGISTER_NULLABLE_BENCHMARKS(
    Double_NullableLeadingHalf,
    NullPattern::kLeadingHalf,
    TypeKind::DOUBLE);

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
