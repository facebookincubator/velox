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
#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::test {
namespace {

void BM_HivePartFunc(
    size_t,
    size_t vectorSize,
    size_t bucketCount,
    const TypePtr& type,
    size_t stringLength = 0) {
  functions::test::FunctionBenchmarkBase benchmarkBase;
  folly::BenchmarkSuspender suspender;

  VectorFuzzer::Options opts;
  opts.vectorSize = vectorSize;
  opts.stringLength = stringLength;
  VectorFuzzer fuzzer(opts, benchmarkBase.pool());
  auto vector = fuzzer.fuzzRow(ROW({type}));

  std::vector<int> bucketToPartition(bucketCount);
  std::vector<ChannelIndex> keyChannels;
  std::vector<uint32_t> partitions(vectorSize);

  std::iota(bucketToPartition.begin(), bucketToPartition.end(), 0);
  keyChannels.emplace_back(0);
  connector::hive::HivePartitionFunction partitionFunction(
      bucketCount, bucketToPartition, keyChannels);

  suspender.dismiss();

  partitionFunction.partition(
      *reinterpret_cast<RowVector*>(vector.get()), partitions);
}

} // namespace

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    BOOLEAN_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    BOOLEAN());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    TINYINT_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    TINYINT());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    SMALLINT_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    SMALLINT());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    INTEGER_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    INTEGER());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    BIGINT_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    BIGINT());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    REAL_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    REAL());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    DOUBLE_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    DOUBLE());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    TIMESTAMP_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    TIMESTAMP());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    DATE_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    DATE());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    VARCHAR_10M_ROWS_1K_BUCKETS_20,
    10'000'000,
    1'000,
    VARCHAR(),
    20);

BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_HivePartFunc,
    VARCHAR_10M_ROWS_1K_BUCKETS_40,
    10'000'000,
    1'000,
    VARCHAR(),
    40);

} // namespace facebook::velox::test

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
