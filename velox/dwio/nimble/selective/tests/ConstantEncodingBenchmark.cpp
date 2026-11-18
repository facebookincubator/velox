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

/// Compares reads of Constant-encoded Nimble columns with the scan fast path
/// off and on. Each case runs twice: "perRow" with nimbleScanFastPath off and
/// "bulk" with it on.
///
/// Data: 1,000,000 rows. Column 'c' holds one value in every row and is written
/// with Constant encoding, in BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT,
/// REAL, DOUBLE and VARCHAR. The VARCHAR value is longer than the inline limit
/// of StringView. Column 'other' holds values in [0, 99] with Trivial encoding.
/// Cases cover projecting 'c' with and without nulls, a filter on 'c' that
/// passes or rejects every row, a filter-only read of 'c', and projecting 'c'
/// behind a filter on 'other' that passes 50% of the rows. Reads use batches of
/// 1,024 rows.
///
/// Run:
///   buck2 run @//mode/opt \
///     fbcode//velox/dwio/nimble/common/selective/tests:constant_encoding_benchmark

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <folly/Benchmark.h>
#include <folly/container/F14Map.h>
#include <folly/init/Init.h>

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

constexpr vector_size_t kNumRows{1'000'000};
constexpr vector_size_t kBatchSize{1'024};
constexpr std::string_view kConstantString{
    "constant-string-longer-than-inline"};
constexpr std::string_view kOtherString{"another-string-longer-than-inline"};

// How a case filters 'c': not at all, with a filter its value passes, or with
// one its value fails.
enum class ConstantFilter : uint8_t {
  kNone,
  kPass,
  kReject,
};

bool isNullAt(vector_size_t row) {
  return row % 17 == 0;
}

int64_t otherValueAt(vector_size_t row) {
  return (static_cast<int64_t>(row) * 9'973) % 100;
}

template <typename T>
T constantValue() {
  if constexpr (std::is_same_v<T, bool>) {
    return true;
  } else if constexpr (std::is_same_v<T, StringView>) {
    return StringView(kConstantString);
  } else {
    return 7;
  }
}

template <typename T>
std::unique_ptr<common::Filter> makeConstantFilter(ConstantFilter filter) {
  const bool pass = filter == ConstantFilter::kPass;
  if constexpr (std::is_same_v<T, bool>) {
    return std::make_unique<common::BoolValue>(pass, false);
  } else if constexpr (std::is_same_v<T, StringView>) {
    return std::make_unique<common::BytesValues>(
        std::vector<std::string>{
            std::string(pass ? kConstantString : kOtherString)},
        false);
  } else if constexpr (std::is_floating_point_v<T>) {
    return std::make_unique<common::FloatingPointRange<T>>(
        pass ? 7 : 8, false, false, 100, false, false, false);
  } else {
    return std::make_unique<common::BigintRange>(pass ? 7 : 8, 100, false);
  }
}

// One benchmark case. Each case runs twice, with the scan fast path off and
// on.
struct BenchmarkConfig {
  TypeKind valueType{TypeKind::BIGINT};
  ConstantFilter filter{ConstantFilter::kNone};
  // Whether 'c' is projected. Filter-only cases project 'other' instead and
  // leave it lazy.
  bool projectConstantColumn{true};
  // Whether every 17th row of 'c' is null.
  bool nullable{false};
  // Percentage of rows that pass a filter on 'other'. Cases that leave it
  // unset do not filter 'other'.
  std::optional<int64_t> otherSelectivity;
};

uint64_t expectedNumOutputRows(const BenchmarkConfig& config) {
  uint64_t numRows{0};
  for (vector_size_t row = 0; row < kNumRows; ++row) {
    const bool passesOther = !config.otherSelectivity.has_value() ||
        otherValueAt(row) < config.otherSelectivity.value();
    const bool passesConstant = config.filter == ConstantFilter::kNone ||
        (config.filter == ConstantFilter::kPass &&
         !(config.nullable && isNullAt(row)));
    if (passesOther && passesConstant) {
      ++numRows;
    }
  }
  return numRows;
}

struct Readers {
  std::unique_ptr<dwio::common::Reader> reader;
  std::unique_ptr<dwio::common::RowReader> rowReader;
  VectorPtr result;
};

class ConstantEncodingBenchmark : public velox::test::VectorTestBase {
 public:
  ConstantEncodingBenchmark() {
    registerSelectiveNimbleReaderFactory();
  }

  ~ConstantEncodingBenchmark() {
    unregisterSelectiveNimbleReaderFactory();
  }

  template <typename T>
  Readers createReaders(const BenchmarkConfig& config, bool scanFastPath) {
    const auto constantType = CppToType<T>::create();
    // The reader decodes every column in the scan spec, so 'other' is added
    // only for the cases that filter or project it.
    const bool readOther =
        config.otherSelectivity.has_value() || !config.projectConstantColumn;
    const auto inputType = readOther
        ? ROW({{"c", constantType}, {"other", BIGINT()}})
        : ROW("c", constantType);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*inputType);
    auto* constantSpec = scanSpec->childByName("c");
    if (config.filter != ConstantFilter::kNone) {
      constantSpec->setFilter(makeConstantFilter<T>(config.filter));
    }
    constantSpec->setProjectOut(config.projectConstantColumn);
    constantSpec->setChannel(
        config.projectConstantColumn ? 0 : common::ScanSpec::kNoChannel);
    if (readOther) {
      auto* otherSpec = scanSpec->childByName("other");
      if (config.otherSelectivity.has_value()) {
        otherSpec->setFilter(
            std::make_unique<common::BigintRange>(
                0, config.otherSelectivity.value() - 1, false));
      }
      otherSpec->setProjectOut(!config.projectConstantColumn);
      otherSpec->setChannel(
          config.projectConstantColumn ? common::ScanSpec::kNoChannel : 0);
    }

    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions readerOptions(pool());
    readerOptions.setScanSpec(scanSpec);

    Readers readers;
    readers.reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(
            std::make_shared<InMemoryReadFile>(file<T>(config.nullable)),
            *pool()),
        readerOptions);
    dwio::common::RowReaderOptions rowReaderOptions;
    rowReaderOptions.setScanSpec(scanSpec);
    rowReaderOptions.setRequestedType(inputType);
    rowReaderOptions.setEagerFirstStripeLoad(true);
    // Constant chunks take the bulk path only through the non-legacy encodings.
    rowReaderOptions.setNimbleScanFastPath(scanFastPath);
    readers.rowReader = readers.reader->createRowReader(rowReaderOptions);
    readers.result = BaseVector::create(
        config.projectConstantColumn ? ROW("c", constantType)
                                     : ROW("other", BIGINT()),
        0,
        pool());
    return readers;
  }

  uint64_t readAll(
      dwio::common::RowReader& rowReader,
      VectorPtr& result,
      bool materializeOutput) {
    uint64_t numOutputRows{0};
    uint64_t numScannedRows{0};
    while (true) {
      const auto numScanned = rowReader.next(kBatchSize, result);
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
  template <typename T>
  const std::string& file(bool nullable) {
    auto& file = files_[CppToType<T>::typeKind][nullable ? 1 : 0];
    if (!file.empty()) {
      return file;
    }

    auto input = makeRowVector(
        {"c", "other"},
        {makeFlatVector<T>(
             kNumRows,
             [](auto /*row*/) { return constantValue<T>(); },
             nullable ? std::function<bool(vector_size_t)>(isNullAt) : nullptr),
         makeFlatVector<int64_t>(kNumRows, otherValueAt)});
    constexpr auto kScalarStream =
        EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream;
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
                {{kScalarStream,
                  EncodingLayout{
                      EncodingType::Constant,
                      {},
                      CompressionType::Uncompressed}}},
                "c"},
            EncodingLayoutTree{
                Kind::Scalar,
                {{kScalarStream,
                  EncodingLayout{
                      EncodingType::Trivial,
                      {},
                      CompressionType::Uncompressed}}},
                "other"},
        });
    file = test::createNimbleFile(*pool()->parent(), input, writerOptions);
    return file;
  }

  // Holds the file for each value type, without and with nulls in 'c'.
  folly::F14NodeMap<TypeKind, std::array<std::string, 2>> files_;
};

ConstantEncodingBenchmark& benchmarkState() {
  static ConstantEncodingBenchmark benchmark;
  return benchmark;
}

template <typename T>
void runBenchmark(
    unsigned iterations,
    const BenchmarkConfig& config,
    bool scanFastPath) {
  folly::BenchmarkSuspender suspender;
  const auto expectedRows = expectedNumOutputRows(config);
  for (unsigned iteration = 0; iteration < iterations; ++iteration) {
    auto readers = benchmarkState().createReaders<T>(config, scanFastPath);
    suspender.dismiss();
    // Leave filter-only output lazy so timing measures filter evaluation and
    // row selection. Materialize a projected 'c' to include its decoding cost.
    const auto numOutputRows = benchmarkState().readAll(
        *readers.rowReader, readers.result, config.projectConstantColumn);
    folly::doNotOptimizeAway(numOutputRows);
    suspender.rehire();
    VELOX_CHECK_EQ(numOutputRows, expectedRows);
  }
}

void runBenchmark(
    unsigned iterations,
    const BenchmarkConfig& config,
    bool scanFastPath) {
  const auto valueType = config.valueType;
  if (valueType == TypeKind::BOOLEAN) {
    runBenchmark<bool>(iterations, config, scanFastPath);
  } else if (valueType == TypeKind::TINYINT) {
    runBenchmark<int8_t>(iterations, config, scanFastPath);
  } else if (valueType == TypeKind::SMALLINT) {
    runBenchmark<int16_t>(iterations, config, scanFastPath);
  } else if (valueType == TypeKind::INTEGER) {
    runBenchmark<int32_t>(iterations, config, scanFastPath);
  } else if (valueType == TypeKind::BIGINT) {
    runBenchmark<int64_t>(iterations, config, scanFastPath);
  } else if (valueType == TypeKind::REAL) {
    runBenchmark<float>(iterations, config, scanFastPath);
  } else if (valueType == TypeKind::DOUBLE) {
    runBenchmark<double>(iterations, config, scanFastPath);
  } else if (valueType == TypeKind::VARCHAR) {
    runBenchmark<StringView>(iterations, config, scanFastPath);
  } else {
    VELOX_UNREACHABLE("Unsupported type: {}", valueType);
  }
}

void perRow(unsigned iterations, const BenchmarkConfig& config) {
  runBenchmark(iterations, config, /*scanFastPath=*/false);
}

void bulk(unsigned iterations, const BenchmarkConfig& config) {
  runBenchmark(iterations, config, /*scanFastPath=*/true);
}

// Registers a per-row and a bulk run of one case. The variadic arguments are
// designated initializers for BenchmarkConfig.
#define REGISTER_BENCHMARK(name, ...)                                \
  BENCHMARK_NAMED_PARAM(perRow, name, BenchmarkConfig{__VA_ARGS__}); \
  BENCHMARK_RELATIVE_NAMED_PARAM(bulk, name, BenchmarkConfig{__VA_ARGS__})

// Registers every case for one value type.
#define REGISTER_TYPE_BENCHMARKS(prefix, type)                           \
  REGISTER_BENCHMARK(prefix##_Projection, .valueType = type);            \
  REGISTER_BENCHMARK(                                                    \
      prefix##_NullableProjection, .valueType = type, .nullable = true); \
  REGISTER_BENCHMARK(                                                    \
      prefix##_FilterPass,                                               \
      .valueType = type,                                                 \
      .filter = ConstantFilter::kPass);                                  \
  REGISTER_BENCHMARK(                                                    \
      prefix##_FilterReject,                                             \
      .valueType = type,                                                 \
      .filter = ConstantFilter::kReject);                                \
  REGISTER_BENCHMARK(                                                    \
      prefix##_FilterOnly,                                               \
      .valueType = type,                                                 \
      .filter = ConstantFilter::kPass,                                   \
      .projectConstantColumn = false);                                   \
  REGISTER_BENCHMARK(                                                    \
      prefix##_Sparse50, .valueType = type, .otherSelectivity = 50);     \
  BENCHMARK_DRAW_LINE()

REGISTER_TYPE_BENCHMARKS(Boolean, TypeKind::BOOLEAN);
REGISTER_TYPE_BENCHMARKS(Tinyint, TypeKind::TINYINT);
REGISTER_TYPE_BENCHMARKS(Smallint, TypeKind::SMALLINT);
REGISTER_TYPE_BENCHMARKS(Integer, TypeKind::INTEGER);
REGISTER_TYPE_BENCHMARKS(Bigint, TypeKind::BIGINT);
REGISTER_TYPE_BENCHMARKS(Real, TypeKind::REAL);
REGISTER_TYPE_BENCHMARKS(Double, TypeKind::DOUBLE);
REGISTER_TYPE_BENCHMARKS(Varchar, TypeKind::VARCHAR);

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
