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

/// Compares dictionary-aware reads with the existing dictionary read path on
/// dictionary-encoded Nimble columns. Each case runs twice: "regular" with
/// dictionaryAwareReads off and "fused" with it on.
///
/// Data: 1,000,000 rows with 1,000 distinct values, in BIGINT, INTEGER,
/// SMALLINT, REAL and DOUBLE columns, with Trivial or FixedBitWidth indices.
/// Cases cover projection only, filter only, and filter plus projection, plus
/// nullable layouts with filters that accept nulls. Reads use batches of 4,096
/// rows, and filters pass 50% of the non-null values.
///
/// Run:
///   buck2 run @//mode/opt \
///     fbcode//velox/dwio/nimble/velox/selective/tests:integer_dictionary_fusion_benchmark

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
  kEvery2,
  kEvery17,
  kNumPatterns,
};

bool isNullAt(NullPattern nullPattern, vector_size_t row) {
  switch (nullPattern) {
    case NullPattern::kNone:
      return false;
    case NullPattern::kEvery2:
      return row % 2 == 0;
    case NullPattern::kEvery17:
      return row % 17 == 0;
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

// One benchmark case. Each case runs twice, with dictionary-aware reads off
// and on.
struct BenchmarkConfig {
  TypeKind valueType{TypeKind::BIGINT};
  EncodingType indicesEncoding{EncodingType::FixedBitWidth};
  NullPattern nullPattern{NullPattern::kNone};
  // Percentage of dictionary values that pass the filter on 'filter_value'.
  // Projection-only cases leave it unset and read without a filter.
  std::optional<uint32_t> selectivity;
  // Whether 'filter_value' is projected. Filter-only cases set it to false.
  bool projectFilterColumn{true};
  // Whether the filter accepts nulls.
  bool nullAllowed{false};
  vector_size_t batchSize{4'096};
};

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
      const BenchmarkConfig& config,
      bool dictionaryAwareReads) {
    const auto inputType =
        ROW({"filter_value", "row_id"},
            {createScalarType(config.valueType), BIGINT()});
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*inputType);
    auto* filterSpec = scanSpec->childByName("filter_value");
    if (config.selectivity.has_value()) {
      const auto numPassingValues =
          kCardinality * config.selectivity.value() / 100;
      filterSpec->setFilter(
          config.valueType == TypeKind::REAL
              ? rangeFilter<float>(numPassingValues, config.nullAllowed)
              : config.valueType == TypeKind::DOUBLE
              ? rangeFilter<double>(numPassingValues, config.nullAllowed)
              : rangeFilter<int64_t>(numPassingValues, config.nullAllowed));
    }
    filterSpec->setProjectOut(config.projectFilterColumn);
    filterSpec->setChannel(
        config.projectFilterColumn ? 0 : common::ScanSpec::kNoChannel);
    scanSpec->childByName("row_id")->setChannel(
        config.projectFilterColumn ? 1 : 0);

    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions readerOptions(pool());
    readerOptions.setScanSpec(scanSpec);

    Readers readers;
    readers.reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(
            std::make_shared<InMemoryReadFile>(file(
                config.valueType, config.indicesEncoding, config.nullPattern)),
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
        config.projectFilterColumn ? inputType : kFilterOnlyOutputType,
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
      case TypeKind::INTEGER:
        return file<int32_t>(indicesEncoding, nullPattern);
      case TypeKind::SMALLINT:
        return file<int16_t>(indicesEncoding, nullPattern);
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
    const BenchmarkConfig& config,
    bool dictionaryAwareReads) {
  folly::BenchmarkSuspender suspender;
  const auto numPassingValues =
      kCardinality * config.selectivity.value_or(100) / 100;
  uint64_t expectedRows{0};
  for (vector_size_t row = 0; row < kNumRows; ++row) {
    const auto isNull = isNullAt(config.nullPattern, row);
    if (!config.selectivity.has_value() ||
        (isNull ? config.nullAllowed : valueAt(row) < numPassingValues)) {
      ++expectedRows;
    }
  }
  for (unsigned iteration = 0; iteration < iterations; ++iteration) {
    auto readers = benchmarkState().createReaders(config, dictionaryAwareReads);
    suspender.dismiss();
    // Leave filter-only output lazy so timing measures filter evaluation and
    // row selection. Materialize all projected children to include their
    // decoding cost.
    const auto numOutputRows = benchmarkState().readAll(
        *readers.rowReader,
        readers.result,
        config.batchSize,
        config.projectFilterColumn);
    folly::doNotOptimizeAway(numOutputRows);
    suspender.rehire();
    VELOX_CHECK_EQ(numOutputRows, expectedRows);
  }
}

void regular(unsigned iterations, const BenchmarkConfig& config) {
  runBenchmark(iterations, config, /*dictionaryAwareReads=*/false);
}

void fused(unsigned iterations, const BenchmarkConfig& config) {
  runBenchmark(iterations, config, /*dictionaryAwareReads=*/true);
}

// Registers a regular and a fused run of one case. The variadic arguments are
// designated initializers for BenchmarkConfig.
#define REGISTER_BENCHMARK(name, ...)                                 \
  BENCHMARK_NAMED_PARAM(regular, name, BenchmarkConfig{__VA_ARGS__}); \
  BENCHMARK_RELATIVE_NAMED_PARAM(fused, name, BenchmarkConfig{__VA_ARGS__})

// Registers every case for one value type.
#define REGISTER_TYPE_BENCHMARKS(prefix, type)                              \
  REGISTER_BENCHMARK(                                                       \
      prefix##_Trivial_Projection,                                          \
      .valueType = type,                                                    \
      .indicesEncoding = EncodingType::Trivial);                            \
  REGISTER_BENCHMARK(prefix##_FixedBitWidth_Projection, .valueType = type); \
  REGISTER_BENCHMARK(                                                       \
      prefix##_Trivial_FilterOnly,                                          \
      .valueType = type,                                                    \
      .indicesEncoding = EncodingType::Trivial,                             \
      .selectivity = 50,                                                    \
      .projectFilterColumn = false);                                        \
  REGISTER_BENCHMARK(                                                       \
      prefix##_FixedBitWidth_FilterOnly,                                    \
      .valueType = type,                                                    \
      .selectivity = 50,                                                    \
      .projectFilterColumn = false);                                        \
  REGISTER_BENCHMARK(                                                       \
      prefix##_Trivial_FilterAndProject,                                    \
      .valueType = type,                                                    \
      .indicesEncoding = EncodingType::Trivial,                             \
      .selectivity = 50);                                                   \
  REGISTER_BENCHMARK(                                                       \
      prefix##_FixedBitWidth_FilterAndProject,                              \
      .valueType = type,                                                    \
      .selectivity = 50);                                                   \
  REGISTER_BENCHMARK(                                                       \
      prefix##_NullEvery17_FilterOnly,                                      \
      .valueType = type,                                                    \
      .nullPattern = NullPattern::kEvery17,                                 \
      .selectivity = 50,                                                    \
      .projectFilterColumn = false,                                         \
      .nullAllowed = true);                                                 \
  REGISTER_BENCHMARK(                                                       \
      prefix##_NullEvery17_FilterAndProject,                                \
      .valueType = type,                                                    \
      .nullPattern = NullPattern::kEvery17,                                 \
      .selectivity = 50,                                                    \
      .nullAllowed = true);                                                 \
  REGISTER_BENCHMARK(                                                       \
      prefix##_NullEvery2_FilterOnly,                                       \
      .valueType = type,                                                    \
      .nullPattern = NullPattern::kEvery2,                                  \
      .selectivity = 50,                                                    \
      .projectFilterColumn = false,                                         \
      .nullAllowed = true);                                                 \
  REGISTER_BENCHMARK(                                                       \
      prefix##_NullEvery2_FilterAndProject,                                 \
      .valueType = type,                                                    \
      .nullPattern = NullPattern::kEvery2,                                  \
      .selectivity = 50,                                                    \
      .nullAllowed = true);                                                 \
  BENCHMARK_DRAW_LINE()

REGISTER_TYPE_BENCHMARKS(Bigint, TypeKind::BIGINT);
REGISTER_TYPE_BENCHMARKS(Integer, TypeKind::INTEGER);
REGISTER_TYPE_BENCHMARKS(Smallint, TypeKind::SMALLINT);
REGISTER_TYPE_BENCHMARKS(Real, TypeKind::REAL);
REGISTER_TYPE_BENCHMARKS(Double, TypeKind::DOUBLE);

#undef REGISTER_TYPE_BENCHMARKS
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
