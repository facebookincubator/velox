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
#include "velox/connectors/hive/iceberg/tests/IcebergInsertBenchmark.h"

using namespace facebook::velox;
using namespace facebook::velox::iceberg::insert::test;

#define IDENTITY_BENCHMARKS(_type_, _name_, _rows_)                \
  BENCHMARK_COUNTERS(run_Identity_##_name_##_##_rows_, counters) { \
    run(1,                                                         \
        _type_,                                                    \
        connector::hive::iceberg::TransformType::kIdentity,        \
        std::nullopt,                                              \
        _rows_,                                                    \
        counters);                                                 \
  }

#define YEAR_BENCHMARKS(_type_, _name_, _rows_)                \
  BENCHMARK_COUNTERS(run_Year_##_name_##_##_rows_, counters) { \
    run(1,                                                     \
        _type_,                                                \
        connector::hive::iceberg::TransformType::kYear,        \
        std::nullopt,                                          \
        _rows_,                                                \
        counters);                                             \
  }

#define MONTH_BENCHMARKS(_type_, _name_, _rows_)                \
  BENCHMARK_COUNTERS(run_Month_##_name_##_##_rows_, counters) { \
    run(1,                                                      \
        _type_,                                                 \
        connector::hive::iceberg::TransformType::kMonth,        \
        std::nullopt,                                           \
        _rows_,                                                 \
        counters);                                              \
  }

#define DAY_BENCHMARKS(_type_, _name_, _rows_)                \
  BENCHMARK_COUNTERS(run_Day_##_name_##_##_rows_, counters) { \
    run(1,                                                    \
        _type_,                                               \
        connector::hive::iceberg::TransformType::kDay,        \
        std::nullopt,                                         \
        _rows_,                                               \
        counters);                                            \
  }

#define HOUR_BENCHMARKS(_type_, _name_, _rows_)                \
  BENCHMARK_COUNTERS(run_Hour_##_name_##_##_rows_, counters) { \
    run(1,                                                     \
        _type_,                                                \
        connector::hive::iceberg::TransformType::kHour,        \
        std::nullopt,                                          \
        _rows_,                                                \
        counters);                                             \
  }

#define BUCKET_BENCHMARKS(_type_, _name_, _buckets_, _rows_)                  \
  BENCHMARK_COUNTERS(run_Bucket##_buckets_##_##_name_##_##_rows_, counters) { \
    run(1,                                                                    \
        _type_,                                                               \
        connector::hive::iceberg::TransformType::kBucket,                     \
        _buckets_,                                                            \
        _rows_,                                                               \
        counters);                                                            \
  }

#define TRUNCATE_BENCHMARKS(_type_, _name_, _width_, _rows_)                  \
  BENCHMARK_COUNTERS(run_Truncate##_width_##_##_name_##_##_rows_, counters) { \
    run(1,                                                                    \
        _type_,                                                               \
        connector::hive::iceberg::TransformType::kTruncate,                   \
        _width_,                                                              \
        _rows_,                                                               \
        counters);                                                            \
  }

constexpr int32_t kRows = 100'000;

IDENTITY_BENCHMARKS(BOOLEAN(), Boolean, kRows);
IDENTITY_BENCHMARKS(TINYINT(), Tinyint, kRows);
IDENTITY_BENCHMARKS(SMALLINT(), Smallint, kRows);
IDENTITY_BENCHMARKS(INTEGER(), Int, kRows);
IDENTITY_BENCHMARKS(BIGINT(), Bigint, kRows);
IDENTITY_BENCHMARKS(DECIMAL(18, 5), DecimalShort, kRows);
IDENTITY_BENCHMARKS(VARCHAR(), Varchar, kRows);
IDENTITY_BENCHMARKS(VARBINARY(), Varbinary, kRows);
IDENTITY_BENCHMARKS(DATE(), Date, kRows);
IDENTITY_BENCHMARKS(TIMESTAMP(), Timestamp, kRows);

YEAR_BENCHMARKS(DATE(), Date, kRows);
YEAR_BENCHMARKS(TIMESTAMP(), Timestamp, kRows);

MONTH_BENCHMARKS(DATE(), Date, kRows);
MONTH_BENCHMARKS(TIMESTAMP(), Timestamp, kRows);

DAY_BENCHMARKS(DATE(), Date, kRows);
DAY_BENCHMARKS(TIMESTAMP(), Timestamp, kRows);

HOUR_BENCHMARKS(TIMESTAMP(), Timestamp, kRows);

BUCKET_BENCHMARKS(INTEGER(), Int, 10, kRows);
BUCKET_BENCHMARKS(INTEGER(), Int, 50, kRows);
BUCKET_BENCHMARKS(BIGINT(), Bigint, 10, kRows);
BUCKET_BENCHMARKS(BIGINT(), Bigint, 50, kRows);
BUCKET_BENCHMARKS(DECIMAL(10, 2), DecimalShort, 10, kRows);
BUCKET_BENCHMARKS(DECIMAL(38, 4), DecimalLong, 10, kRows);
BUCKET_BENCHMARKS(DATE(), Date, 11, kRows);
BUCKET_BENCHMARKS(TIMESTAMP(), Timestamp, 13, kRows);
BUCKET_BENCHMARKS(VARCHAR(), Varchar, 10, kRows);
BUCKET_BENCHMARKS(VARBINARY(), Varbinary, 10, kRows);

TRUNCATE_BENCHMARKS(INTEGER(), Int, 100, kRows);
TRUNCATE_BENCHMARKS(INTEGER(), Int, 1000, kRows);
TRUNCATE_BENCHMARKS(BIGINT(), Bigint, 1000, kRows);
TRUNCATE_BENCHMARKS(BIGINT(), Bigint, 10000, kRows);
TRUNCATE_BENCHMARKS(DECIMAL(10, 2), DecimalShort, 100, kRows);
TRUNCATE_BENCHMARKS(VARCHAR(), Varchar, 16, kRows);
TRUNCATE_BENCHMARKS(VARCHAR(), Varchar, 32, kRows);
TRUNCATE_BENCHMARKS(VARBINARY(), Varbinary, 16, kRows);

BENCHMARK_DRAW_LINE();

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}
