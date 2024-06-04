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

#include "glog/logging.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

class TestCase {
 public:
  TestCase(
      memory::MemoryPool* pool,
      const std::string& testName,
      size_t numRows,
      int numVectors,
      const RowTypePtr& rowType,
      int numKeys)
      : testName_(testName) {
    std::vector<RowVectorPtr> vectors;
    for (auto i = 0; i < numVectors; ++i) {
      vectors.emplace_back(fuzzRows(rowType, numRows, numKeys, pool));
    }
    plan_ = makeOrderByPlan(vectors, makeOrderByKeys(numKeys));
  };

  const std::string& testName() const {
    return testName_;
  }
  core::PlanNodePtr plan_;

 private:
  RowVectorPtr fuzzRows(
      const RowTypePtr& rowType,
      size_t numRows,
      int numKeys,
      memory::MemoryPool* pool) {
    VectorFuzzer fuzzer({.vectorSize = numRows}, pool);
    VectorFuzzer fuzzerWithNulls(
        {.vectorSize = numRows, .nullRatio = 0.7}, pool);
    std::vector<VectorPtr> children;

    // Fuzz keys: for front keys (column 0 to numKeys -2) use high
    // nullRatio to enforce all columns to be compared.
    {
      for (auto i = 0; i < numKeys - 1; ++i) {
        children.push_back(fuzzerWithNulls.fuzz(rowType->childAt(i)));
      }
      children.push_back(fuzzer.fuzz(rowType->childAt(numKeys - 1)));
    }
    // Fuzz payload
    {
      for (auto i = numKeys; i < rowType->size(); ++i) {
        children.push_back(fuzzer.fuzz(rowType->childAt(i)));
      }
    }
    return std::make_shared<RowVector>(
        pool, rowType, nullptr, numRows, std::move(children));
  }

  core::PlanNodePtr makeOrderByPlan(
      const std::vector<RowVectorPtr>& vectors,
      const std::vector<std::string>& keys) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId orderNodeId;
    return test::PlanBuilder(planNodeIdGenerator)
        .values(vectors)
        .orderBy(keys, false)
        .capturePlanNodeId(orderNodeId)
        .planNode();
  }

  std::vector<std::string> makeOrderByKeys(int numKeys) {
    std::vector<std::string> keys;
    keys.reserve(numKeys);
    for (auto i = 0; i < numKeys; i++) {
      keys.emplace_back(fmt::format("c{} ASC NULLS LAST", i));
    }
    return keys;
  }

  const std::string testName_;
};

class OrderByBenchmark {
 public:
  void addBenchmark(
      const std::string& testName,
      size_t numRows,
      const RowTypePtr& rowType,
      int numVectors,
      int numKeys) {
    auto testCase = std::make_unique<TestCase>(
        pool_.get(), testName, numRows, numVectors, rowType, numKeys);
    {
      folly::addBenchmark(
          __FILE__,
          "OrderBy_" + testCase->testName(),
          [plan = &testCase->plan_, this]() {
            run(*plan);
            return 1;
          });
    }
    testCases_.push_back(std::move(testCase));
  }

  void benchmark(
      const std::string& prefix,
      const std::string& keyName,
      const std::vector<vector_size_t>& batchSizes,
      const std::vector<RowTypePtr>& rowTypes,
      const std::vector<int>& numKeys,
      int32_t numVectors) {
    for (auto batchSize : batchSizes) {
      for (auto i = 0; i < rowTypes.size(); ++i) {
        const auto name = fmt::format(
            "{}_{}_{}_{}k", prefix, numKeys[i], keyName, batchSize / 1000.0);
        addBenchmark(name, batchSize, rowTypes[i], numVectors, numKeys[i]);
      }
    }
  }

  std::vector<RowTypePtr> bigintRowTypes(bool noPayload) {
    if (noPayload) {
      return {
          rowWithName({BIGINT()}),
          rowWithName({BIGINT(), BIGINT()}),
          rowWithName({BIGINT(), BIGINT(), BIGINT()}),
          rowWithName({BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
      };
    } else {
      return {
          rowWithName({BIGINT(), VARCHAR(), VARCHAR()}),
          rowWithName({BIGINT(), BIGINT(), VARCHAR(), VARCHAR()}),
          rowWithName({BIGINT(), BIGINT(), BIGINT(), VARCHAR(), VARCHAR()}),
          rowWithName(
              {BIGINT(), BIGINT(), BIGINT(), BIGINT(), VARCHAR(), VARCHAR()}),
      };
    }
  }

  void bigint(
      bool noPayload,
      int numVectors,
      const std::vector<vector_size_t>& batchSizes) {
    std::vector<RowTypePtr> rowTypes = bigintRowTypes(noPayload);
    std::vector<int> numKeys = {1, 2, 3, 4};
    benchmark(
        noPayload ? "no-payload" : "payload",
        "bigint",
        batchSizes,
        rowTypes,
        numKeys,
        numVectors);
  }

  void smallBigint() {
    const std::vector<vector_size_t> batchSizes = {10, 50, 100, 500};
    bigint(true, 200, batchSizes);
  }

  void smallBigintWithPayload() {
    const std::vector<vector_size_t> batchSizes = {10, 50, 100, 500};
    bigint(false, 200, batchSizes);
  }

  void largeBigint() {
    const std::vector<vector_size_t> batchSizes = {
        1'000, 10'000, 100'000, 1'000'000};
    bigint(true, 10, batchSizes);
  }

  void largeBigintWithPayloads() {
    const std::vector<vector_size_t> batchSizes = {
        1'000, 10'000, 100'000, 1'000'000};
    bigint(false, 10, batchSizes);
  }

  void largeVarchar() {
    const std::vector<vector_size_t> batchSizes = {
        1'000, 10'000, 100'000, 1'000'000};
    std::vector<RowTypePtr> rowTypes = {
        rowWithName({VARCHAR()}),
        rowWithName({VARCHAR(), VARCHAR()}),
        rowWithName({VARCHAR(), VARCHAR(), VARCHAR()}),
        rowWithName({VARCHAR(), VARCHAR(), VARCHAR(), VARCHAR()}),
    };
    std::vector<int> numKeys = {1, 2, 3, 4};
    benchmark(
        "no-payloads", "varchar", batchSizes, rowTypes, numKeys, 10);
  }

 private:
  int64_t run(core::PlanNodePtr plan) {
    auto start = getCurrentTimeMicro();
    auto result = exec::test::AssertQueryBuilder(plan).copyResults(pool_.get());
    auto elapsedMicros = getCurrentTimeMicro() - start;
    return elapsedMicros;
  }

  std::shared_ptr<const RowType> rowWithName(std::vector<TypePtr>&& types) {
    std::vector<std::string> names;
    for (auto i = 0; i < types.size(); ++i) {
      names.emplace_back(fmt::format("c{}", i));
    }
    return ROW(std::move(names), std::move(types));
  }

  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{rootPool_->addLeafChild("leaf")};
  std::vector<std::unique_ptr<TestCase>> testCases_;
};
} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  memory::MemoryManager::initialize({});
  OrderByBenchmark bm;

  bm.smallBigint();
  bm.largeBigint();
  bm.largeBigintWithPayloads();
  bm.smallBigintWithPayload();
  bm.largeVarchar();
  folly::runBenchmarks();

  return 0;
}
