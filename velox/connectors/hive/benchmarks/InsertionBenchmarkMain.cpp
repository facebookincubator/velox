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
#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include "velox/connectors/hive/benchmarks/Benchmark.h"
#include "velox/connectors/hive/benchmarks/InsertionBenchmark.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"

DEFINE_int32(vector_size, 100'000, "Number of rows to benchmark");
DEFINE_bool(with_counter, false, "Run with customize counters");

using namespace facebook::velox;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::connector::hive::insert::test;

const int32_t kRows = FLAGS_vector_size;

#define HIVE_BENCHMARK_SIMPLE(_type_, _name_, _rows_) \
  BENCHMARK(run_Hive_##_name_##_##_rows_) {           \
    runHive(1, _type_, _rows_);                       \
  }

#define ICEBERG_BENCHMARK_SIMPLE(_type_, _name_, _rows_)                     \
  BENCHMARK_RELATIVE(run_Iceberg_##_name_##_##_rows_) {                      \
    runIceberg(                                                              \
        1, _type_, iceberg::TransformType::kIdentity, std::nullopt, _rows_); \
  }

#define ICEBERG_BENCHMARK_BUCKET(_type_, _name_, _rows_)       \
  BENCHMARK_RELATIVE(run_Iceberg_Bucket_##_name_##_##_rows_) { \
    runIceberg(                                                \
        1,                                                     \
        _type_,                                                \
        iceberg::TransformType::kBucket,                       \
        std::optional<int32_t>{128},                           \
        _rows_);                                               \
  }

#define ICEBERG_BENCHMARK_TRUNCATE(_type_, _name_, _rows_)    \
  BENCHMARK_RELATIVE(run_Iceberg_Trunc_##_name_##_##_rows_) { \
    runIceberg(                                               \
        1,                                                    \
        _type_,                                               \
        iceberg::TransformType::kTruncate,                    \
        std::optional<int32_t>{100},                          \
        _rows_);                                              \
  }

#define COMPARISON_BENCHMARK(_type_, _name_, _rows_)                 \
  BENCHMARK_COUNTERS(run_Comparison_##_name_##_##_rows_, counters) { \
    runComparison(1, _type_, _rows_, counters);                      \
  }

BENCHMARK_DRAW_LINE();
HIVE_BENCHMARK_SIMPLE(BOOLEAN(), Boolean, kRows);
ICEBERG_BENCHMARK_SIMPLE(BOOLEAN(), Boolean, kRows);
BENCHMARK_DRAW_LINE();

HIVE_BENCHMARK_SIMPLE(SMALLINT(), SmallInt, kRows);
ICEBERG_BENCHMARK_SIMPLE(SMALLINT(), SmallInt, kRows);
BENCHMARK_DRAW_LINE();

HIVE_BENCHMARK_SIMPLE(INTEGER(), Int, kRows);
ICEBERG_BENCHMARK_SIMPLE(INTEGER(), Int, kRows);
ICEBERG_BENCHMARK_BUCKET(INTEGER(), Int, kRows);
BENCHMARK_DRAW_LINE();

HIVE_BENCHMARK_SIMPLE(BIGINT(), BigInt, kRows);
ICEBERG_BENCHMARK_SIMPLE(BIGINT(), BigInt, kRows);
ICEBERG_BENCHMARK_BUCKET(BIGINT(), BigInt, kRows);
ICEBERG_BENCHMARK_TRUNCATE(BIGINT(), BigInt, kRows);
BENCHMARK_DRAW_LINE();

HIVE_BENCHMARK_SIMPLE(DECIMAL(10, 2), DecimalShort, kRows);
ICEBERG_BENCHMARK_SIMPLE(DECIMAL(10, 2), DecimalShort, kRows);
ICEBERG_BENCHMARK_BUCKET(DECIMAL(10, 2), DecimalShort, kRows);
ICEBERG_BENCHMARK_BUCKET(DECIMAL(38, 4), DecimalLong, kRows);
ICEBERG_BENCHMARK_TRUNCATE(DECIMAL(10, 2), DecimalLong, kRows);
BENCHMARK_DRAW_LINE();

HIVE_BENCHMARK_SIMPLE(DATE(), Date, kRows);
ICEBERG_BENCHMARK_SIMPLE(DATE(), Date, kRows);
ICEBERG_BENCHMARK_BUCKET(DATE(), Date, kRows);
BENCHMARK_DRAW_LINE();

HIVE_BENCHMARK_SIMPLE(VARCHAR(), Varchar, kRows);
ICEBERG_BENCHMARK_SIMPLE(VARCHAR(), Varchar, kRows);
ICEBERG_BENCHMARK_BUCKET(VARCHAR(), Varchar, kRows);
ICEBERG_BENCHMARK_TRUNCATE(VARCHAR(), Varchar, kRows);
BENCHMARK_DRAW_LINE();

HIVE_BENCHMARK_SIMPLE(VARBINARY(), Varbinary, kRows);
ICEBERG_BENCHMARK_SIMPLE(VARBINARY(), Varbinary, kRows);
ICEBERG_BENCHMARK_BUCKET(VARBINARY(), Varbinary, kRows);
ICEBERG_BENCHMARK_TRUNCATE(VARBINARY(), Varbinary, kRows);
BENCHMARK_DRAW_LINE();

// Comparison benchmarks: print detailed counters and compare Hive vs Iceberg
// COMPARISON_BENCHMARK(INTEGER(), Int, kRows);
// COMPARISON_BENCHMARK(BIGINT(), BigInt, kRows);
// COMPARISON_BENCHMARK(DECIMAL(10, 2), DecimalShort, kRows);
// COMPARISON_BENCHMARK(DATE(), Date, kRows);
// COMPARISON_BENCHMARK(VARCHAR(), Varchar, kRows);
// COMPARISON_BENCHMARK(VARBINARY(), Varbinary, kRows);
BENCHMARK_DRAW_LINE();
int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_with_counter)
    gflags::SetCommandLineOption("benchmark_filter", "^run_Comparison_.*");
  else
    gflags::SetCommandLineOption("benchmark_filter", "^run_(Hive|Iceberg)_.*");

  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}
