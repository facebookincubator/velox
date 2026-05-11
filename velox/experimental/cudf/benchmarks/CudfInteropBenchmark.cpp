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

#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/memory/Memory.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

#include <cudf/utilities/default_stream.hpp>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

namespace facebook::velox::cudf_velox {
namespace {

class CudfInteropBenchmark {
 public:
  CudfInteropBenchmark() {
    pool_ = memory::memoryManager()->addLeafPool();
  }

  std::unique_ptr<cudf::table> veloxToCudf(const RowVectorPtr& data) {
    // Ensure the vector is flat before exporting.
    VectorPtr flatData = data;
    BaseVector::flattenVector(flatData);
    auto cudfTable = with_arrow::toCudfTable(
        std::static_pointer_cast<RowVector>(flatData), pool_.get(), stream, mr);
    stream.synchronize();
    VELOX_CHECK_NOT_NULL(cudfTable);
    VELOX_CHECK_EQ(cudfTable->num_rows(), flatData->size());
    return cudfTable;
  }

  void cudfToVelox(
      std::unique_ptr<cudf::table>& cudfTable,
      const RowTypePtr& rowType) {
    auto veloxData = with_arrow::toVeloxColumn(
        cudfTable->view(), pool_.get(), rowType, stream, mr);
    stream.synchronize();
    VELOX_CHECK_NOT_NULL(veloxData);
    VELOX_CHECK_EQ(veloxData->size(), cudfTable->num_rows());
  }

  std::unique_ptr<cudf::table> veloxToCudfLarge(
      const RowTypePtr& rowType,
      size_t numRows) {
    folly::BenchmarkSuspender suspender;
    auto data = makeLargeData(rowType, numRows);
    // Ensure the vector is flat before exporting.
    VectorPtr flatData = data;
    BaseVector::flattenVector(flatData);
    suspender.dismiss();

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto cudfTable = with_arrow::toCudfTable(
        std::static_pointer_cast<RowVector>(flatData), pool_.get(), stream, mr);
    stream.synchronize();
    VELOX_CHECK_NOT_NULL(cudfTable);
    VELOX_CHECK_EQ(cudfTable->num_rows(), flatData->size());
    return cudfTable;
  }

  void cudfToVeloxLarge(
      std::unique_ptr<cudf::table>& cudfTable,
      const RowTypePtr& rowType) {
    auto veloxData = with_arrow::toVeloxColumn(
        cudfTable->view(), pool_.get(), rowType, stream, mr);
    stream.synchronize();
    VELOX_CHECK_NOT_NULL(veloxData);
    VELOX_CHECK_EQ(veloxData->size(), cudfTable->num_rows());
  }

  RowVectorPtr makeLargeData(const RowTypePtr& rowType, size_t maxRows) {
    VectorFuzzer::Options options;
    options.vectorSize = 1'000;
    options.nullRatio = 0.25;
    options.allowDictionaryVector = false;

    const auto seed = 1; // For reproducibility.
    VectorFuzzer fuzzer(options, pool_.get(), seed);

    // Create initial batch
    auto result = fuzzer.fuzzInputRow(rowType);

    // Use exponential growth: append result to itself to double the size
    // Check if doubling would exceed maxRows before appending
    size_t currentSize = result->size();
    while (currentSize * 2 < maxRows) {
      // Append result to itself for exponential growth
      result->append(result.get());
      currentSize = result->size();
    }

    return result;
  }

  RowVectorPtr makeData(const RowTypePtr& rowType, double nullRatio = 0.25) {
    folly::BenchmarkSuspender suspender;
    VectorFuzzer::Options options;
    options.vectorSize = 1'000;
    options.nullRatio = nullRatio;
    options.allowDictionaryVector = false;

    const auto seed = 1; // For reproducibility.
    VectorFuzzer fuzzer(options, pool_.get(), seed);
    auto vector = fuzzer.fuzzInputRow(rowType);
    suspender.dismiss();
    return vector;
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  rmm::cuda_stream_view stream = cudf::get_default_stream();
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
};

size_t iters = 2000;

#define INTEROP_BENCHMARKS(name, rowType)         \
  BENCHMARK(velox_to_cudf_##name) {               \
    CudfInteropBenchmark benchmark;               \
                                                  \
    folly::BenchmarkSuspender suspender;          \
    auto data = benchmark.makeData(rowType);      \
    suspender.dismiss();                          \
                                                  \
    for (auto i = 0; i < iters; ++i) {            \
      benchmark.veloxToCudf(data);                \
    }                                             \
  }                                               \
                                                  \
  BENCHMARK(cudf_to_velox_##name) {               \
    CudfInteropBenchmark benchmark;               \
                                                  \
    folly::BenchmarkSuspender suspender;          \
    auto data = benchmark.makeData(rowType);      \
    auto cudfTable = benchmark.veloxToCudf(data); \
    suspender.dismiss();                          \
                                                  \
    for (auto i = 0; i < iters; ++i) {            \
      benchmark.cudfToVelox(cudfTable, rowType);  \
    }                                             \
  }                                               \
                                                  \
  BENCHMARK(velox_to_cudf_non_null_##name) {      \
    CudfInteropBenchmark benchmark;               \
                                                  \
    folly::BenchmarkSuspender suspender;          \
    auto data = benchmark.makeData(rowType, 0);   \
    suspender.dismiss();                          \
                                                  \
    for (auto i = 0; i < iters; ++i) {            \
      benchmark.veloxToCudf(data);                \
    }                                             \
  }                                               \
                                                  \
  BENCHMARK(cudf_to_velox_non_null_##name) {      \
    CudfInteropBenchmark benchmark;               \
                                                  \
    folly::BenchmarkSuspender suspender;          \
    auto data = benchmark.makeData(rowType, 0);   \
    auto cudfTable = benchmark.veloxToCudf(data); \
    suspender.dismiss();                          \
                                                  \
    for (auto i = 0; i < iters; ++i) {            \
      benchmark.cudfToVelox(cudfTable, rowType);  \
    }                                             \
  }

// Fixed-width type benchmarks with non-null variants (1000 rows)
INTEROP_BENCHMARKS(
    fixedWidth5,
    ROW({BIGINT(), DOUBLE(), BOOLEAN(), TINYINT(), REAL()}));

INTEROP_BENCHMARKS(
    fixedWidth10,
    ROW({
        BIGINT(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
        DOUBLE(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
    }));

INTEROP_BENCHMARKS(
    fixedWidth20,
    ROW({
        BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(),
        BIGINT(), BIGINT(), BIGINT(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(),
        DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), BIGINT(),
    }));

// Decimal benchmarks
INTEROP_BENCHMARKS(decimals, ROW({BIGINT(), DECIMAL(12, 2), DECIMAL(38, 18)}));

// String benchmarks with non-null variants (1000 rows)
INTEROP_BENCHMARKS(strings1, ROW({BIGINT(), VARCHAR()}));

INTEROP_BENCHMARKS(
    strings5,
    ROW({
        BIGINT(),
        VARCHAR(),
        VARCHAR(),
        VARCHAR(),
        VARCHAR(),
        VARCHAR(),
    }));

// Array benchmarks with non-null variants (1000 rows)
INTEROP_BENCHMARKS(arrays, ROW({BIGINT(), ARRAY(BIGINT())}));

INTEROP_BENCHMARKS(nestedArrays, ROW({BIGINT(), ARRAY(ARRAY(BIGINT()))}));

// Struct benchmarks with non-null variants (1000 rows)
INTEROP_BENCHMARKS(
    structs,
    ROW({BIGINT(), ROW({BIGINT(), DOUBLE(), BOOLEAN(), TINYINT(), REAL()})}));

// Mixed type benchmarks with non-null variants (1000 rows)
INTEROP_BENCHMARKS(
    mixed,
    ROW({
        BIGINT(),
        VARCHAR(),
        DOUBLE(),
        ARRAY(BIGINT()),
        ROW({BIGINT(), VARCHAR()}),
    }));

// Date and timestamp benchmarks with non-null variants (1000 rows)
INTEROP_BENCHMARKS(dates, ROW({BIGINT(), DATE()}));

INTEROP_BENCHMARKS(timestamps, ROW({BIGINT(), TIMESTAMP()}));

INTEROP_BENCHMARKS(
    dateTimeMixed,
    ROW({BIGINT(), DATE(), TIMESTAMP(), VARCHAR()}));

// Large vector benchmarks with different row counts
#define LARGE_INTEROP_BENCHMARKS(name, rowType, numRows)   \
  BENCHMARK(velox_to_cudf_large_##name##_##numRows) {      \
    CudfInteropBenchmark benchmark;                        \
                                                           \
    folly::BenchmarkSuspender suspender;                   \
    auto data = benchmark.makeLargeData(rowType, numRows); \
    suspender.dismiss();                                   \
                                                           \
    for (auto i = 0; i < iters; ++i) {                     \
      benchmark.veloxToCudf(data);                         \
    }                                                      \
  }                                                        \
                                                           \
  BENCHMARK(cudf_to_velox_large_##name##_##numRows) {      \
    CudfInteropBenchmark benchmark;                        \
                                                           \
    folly::BenchmarkSuspender suspender;                   \
    auto data = benchmark.makeLargeData(rowType, numRows); \
    auto cudfTable = benchmark.veloxToCudf(data);          \
    suspender.dismiss();                                   \
                                                           \
    for (auto i = 0; i < iters; ++i) {                     \
      benchmark.cudfToVeloxLarge(cudfTable, rowType);      \
    }                                                      \
  }

// Test with 10,000 rows
LARGE_INTEROP_BENCHMARKS(
    fixedWidth10,
    ROW({
        BIGINT(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
        DOUBLE(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
    }),
    10000);

LARGE_INTEROP_BENCHMARKS(
    strings5,
    ROW({
        BIGINT(),
        VARCHAR(),
        VARCHAR(),
        VARCHAR(),
        VARCHAR(),
        VARCHAR(),
    }),
    10000);

LARGE_INTEROP_BENCHMARKS(
    mixed,
    ROW({
        BIGINT(),
        VARCHAR(),
        DOUBLE(),
        ARRAY(BIGINT()),
        ROW({BIGINT(), VARCHAR()}),
    }),
    10000);

LARGE_INTEROP_BENCHMARKS(timestamps, ROW({BIGINT(), TIMESTAMP()}), 10000);

LARGE_INTEROP_BENCHMARKS(booleans, ROW({BIGINT(), BOOLEAN()}), 10000);

LARGE_INTEROP_BENCHMARKS(arrays, ROW({ARRAY(BIGINT())}), 10000);

// Test with 40,000 rows
LARGE_INTEROP_BENCHMARKS(
    fixedWidth10,
    ROW({
        BIGINT(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
        DOUBLE(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
    }),
    40000);

LARGE_INTEROP_BENCHMARKS(
    strings5,
    ROW({
        BIGINT(),
        VARCHAR(),
        VARCHAR(),
        VARCHAR(),
        VARCHAR(),
        VARCHAR(),
    }),
    40000);

LARGE_INTEROP_BENCHMARKS(
    mixed,
    ROW({
        BIGINT(),
        VARCHAR(),
        DOUBLE(),
        ARRAY(BIGINT()),
        ROW({BIGINT(), VARCHAR()}),
    }),
    40000);

// Test with 80,000 rows
LARGE_INTEROP_BENCHMARKS(
    fixedWidth10,
    ROW({
        BIGINT(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
        DOUBLE(),
        BIGINT(),
        BIGINT(),
        BIGINT(),
    }),
    80000);

LARGE_INTEROP_BENCHMARKS(
    strings5,
    ROW({
        BIGINT(),
        VARCHAR(),
        VARCHAR(),
        VARCHAR(),
        VARCHAR(),
        VARCHAR(),
    }),
    80000);

LARGE_INTEROP_BENCHMARKS(
    mixed,
    ROW({
        BIGINT(),
        VARCHAR(),
        DOUBLE(),
        ARRAY(BIGINT()),
        ROW({BIGINT(), VARCHAR()}),
    }),
    80000);

} // namespace
} // namespace facebook::velox::cudf_velox

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}

// Made with Bob
