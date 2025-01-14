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

#include "velox/exec/benchmarks/OrderByBenchmarkUtil.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

DEFINE_double(data_null_ratio, 0.7, "Data null ratio");

using namespace facebook::velox;

namespace facebook::velox::exec {
namespace {

using OrderByBenchmarkFunction = std::function<void(
    const std::string& benchmarkName,
    vector_size_t numRows,
    const RowTypePtr& rowType,
    int iterations,
    int numKeys)>;

void addBenchmark(
    const std::string& prefix,
    const std::string& keyName,
    const std::vector<vector_size_t>& batchSizes,
    const std::vector<RowTypePtr>& rowTypes,
    const std::vector<int>& numKeys,
    int32_t iterations,
    const OrderByBenchmarkFunction& benchmark) {
  for (auto batchSize : batchSizes) {
    for (auto i = 0; i < rowTypes.size(); ++i) {
      const auto name = fmt::format(
          "{}_{}_{}_{}k", prefix, numKeys[i], keyName, batchSize / 1000.0);
      benchmark(name, batchSize, rowTypes[i], iterations, numKeys[i]);
    }
  }
}

std::vector<RowTypePtr> smallintRowTypes(bool noPayload) {
  if (noPayload) {
    return {
        ROW({SMALLINT()}),
        ROW({SMALLINT(), SMALLINT()}),
        ROW({SMALLINT(), SMALLINT(), SMALLINT()}),
        ROW({SMALLINT(), SMALLINT(), SMALLINT(), SMALLINT()}),
    };
  }
  return {
      ROW({SMALLINT(), VARCHAR(), VARCHAR()}),
      ROW({SMALLINT(), SMALLINT(), VARCHAR(), VARCHAR()}),
      ROW({SMALLINT(), SMALLINT(), SMALLINT(), VARCHAR(), VARCHAR()}),
      ROW(
          {SMALLINT(),
           SMALLINT(),
           SMALLINT(),
           SMALLINT(),
           VARCHAR(),
           VARCHAR()}),
  };
}

std::vector<RowTypePtr> bigintRowTypes(bool noPayload) {
  if (noPayload) {
    return {
        test::VectorMaker::rowType({BIGINT()}),
        test::VectorMaker::rowType({BIGINT(), BIGINT()}),
        test::VectorMaker::rowType({BIGINT(), BIGINT(), BIGINT()}),
        test::VectorMaker::rowType({BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
    };
  }
  return {
      test::VectorMaker::rowType({BIGINT(), VARCHAR(), VARCHAR()}),
      test::VectorMaker::rowType({BIGINT(), BIGINT(), VARCHAR(), VARCHAR()}),
      test::VectorMaker::rowType(
          {BIGINT(), BIGINT(), BIGINT(), VARCHAR(), VARCHAR()}),
      test::VectorMaker::rowType(
          {BIGINT(), BIGINT(), BIGINT(), BIGINT(), VARCHAR(), VARCHAR()}),
  };
}

std::vector<RowTypePtr> largeVarcharRowTypes() {
  return {
      test::VectorMaker::rowType({VARCHAR()}),
      test::VectorMaker::rowType({VARCHAR(), VARCHAR()}),
      test::VectorMaker::rowType({VARCHAR(), VARCHAR(), VARCHAR()}),
      test::VectorMaker::rowType({VARCHAR(), VARCHAR(), VARCHAR(), VARCHAR()}),
  };
}

void bigint(
    bool noPayload,
    int numIterations,
    const std::vector<vector_size_t>& batchSizes,
    const OrderByBenchmarkFunction& benchmark) {
  const std::vector<RowTypePtr> rowTypes = bigintRowTypes(noPayload);
  const std::vector<int> numKeys = {1, 2, 3, 4};
  addBenchmark(
      noPayload ? "no-payload" : "payload",
      "bigint",
      batchSizes,
      rowTypes,
      numKeys,
      numIterations,
      benchmark);
}

void smallBigint(const OrderByBenchmarkFunction& benchmark) {
  // For small dateset, iterations need to be large enough to ensure that the
  // benchmark runs for enough time.
  const auto iterations = 100'000;
  const std::vector<vector_size_t> batchSizes = {10, 50, 100, 500};
  bigint(true, iterations, batchSizes, benchmark);
}

void smallBigintWithPayload(const OrderByBenchmarkFunction& benchmark) {
  const auto iterations = 100'000;
  const std::vector<vector_size_t> batchSizes = {10, 50, 100, 500};
  bigint(false, iterations, batchSizes, benchmark);
}

void largeBigint(const OrderByBenchmarkFunction& benchmark) {
  const auto iterations = 10;
  const std::vector<vector_size_t> batchSizes = {
      1'000, 10'000, 100'000, 1'000'000};
  bigint(true, iterations, batchSizes, benchmark);
}

void largeBigintWithPayloads(const OrderByBenchmarkFunction& benchmark) {
  const auto iterations = 10;
  const std::vector<vector_size_t> batchSizes = {
      1'000, 10'000, 100'000, 1'000'000};
  bigint(false, iterations, batchSizes, benchmark);
}

void largeVarchar(const OrderByBenchmarkFunction& benchmark) {
  const auto iterations = 10;
  const std::vector<vector_size_t> batchSizes = {
      1'000, 10'000, 100'000, 1'000'000};
  std::vector<RowTypePtr> rowTypes = largeVarcharRowTypes();
  std::vector<int> numKeys = {1, 2, 3, 4};
  addBenchmark(
      "no-payloads",
      "varchar",
      batchSizes,
      rowTypes,
      numKeys,
      iterations,
      benchmark);
}

void smallint(
    bool noPayload,
    int numIterations,
    const std::vector<vector_size_t>& batchSizes,
    const OrderByBenchmarkFunction& benchmark) {
  const std::vector<RowTypePtr> rowTypes = smallintRowTypes(noPayload);
  const std::vector<int> numKeys = {1, 2, 3, 4};
  addBenchmark(
      noPayload ? "no-payload" : "payload",
      "smallint",
      batchSizes,
      rowTypes,
      numKeys,
      numIterations,
      benchmark);
}

void smallSmallint(const OrderByBenchmarkFunction& benchmark) {
  // For small dateset, iterations need to be large enough to ensure that the
  // benchmark runs for enough time.
  const auto iterations = 100'000;
  const std::vector<vector_size_t> batchSizes = {10, 50, 100, 500};
  smallint(true, iterations, batchSizes, benchmark);
}

void smallSmallintWithPayload(const OrderByBenchmarkFunction& benchmark) {
  const auto iterations = 100'000;
  const std::vector<vector_size_t> batchSizes = {10, 50, 100, 500};
  smallint(false, iterations, batchSizes, benchmark);
}

void largeSmallint(const OrderByBenchmarkFunction& benchmark) {
  const auto iterations = 10;
  const std::vector<vector_size_t> batchSizes = {
      1'000, 10'000, 100'000, 1'000'000};
  smallint(true, iterations, batchSizes, benchmark);
}

void largeSmallintWithPayloads(const OrderByBenchmarkFunction& benchmark) {
  const auto iterations = 10;
  const std::vector<vector_size_t> batchSizes = {
      1'000, 10'000, 100'000, 1'000'000};
  smallint(false, iterations, batchSizes, benchmark);
}
} // namespace

RowVectorPtr OrderByBenchmarkUtil::fuzzRows(
    const RowTypePtr& rowType,
    vector_size_t numRows,
    int numKeys,
    memory::MemoryPool* pool) {
  VectorFuzzer fuzzer({.vectorSize = static_cast<size_t>(numRows)}, pool);
  VectorFuzzer fuzzerWithNulls(
      {.vectorSize = static_cast<size_t>(numRows),
       .nullRatio = FLAGS_data_null_ratio},
      pool);
  std::vector<VectorPtr> children;

  // Fuzz keys: for front keys (column 0 to numKeys -2) use high
  // nullRatio to enforce all columns to be compared.
  for (auto i = 0; i < numKeys - 1; ++i) {
    children.push_back(fuzzerWithNulls.fuzz(rowType->childAt(i)));
  }
  children.push_back(fuzzer.fuzz(rowType->childAt(numKeys - 1)));

  // Fuzz payload.
  for (auto i = numKeys; i < rowType->size(); ++i) {
    children.push_back(fuzzer.fuzz(rowType->childAt(i)));
  }
  return std::make_shared<RowVector>(
      pool, rowType, nullptr, numRows, std::move(children));
}

void OrderByBenchmarkUtil::addBenchmarks(
    const OrderByBenchmarkFunction& benchmark) {
  smallBigint(benchmark);
  largeBigint(benchmark);
  largeBigintWithPayloads(benchmark);
  smallBigintWithPayload(benchmark);
  largeVarchar(benchmark);
  smallSmallint(benchmark);
  largeSmallint(benchmark);
  smallSmallintWithPayload(benchmark);
  largeSmallintWithPayloads(benchmark);
}
} // namespace facebook::velox::exec
