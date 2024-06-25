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

#include "OrderByBenchmarkUtil.h"
#include "glog/logging.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

struct TestCase {
  explicit TestCase(
      const std::string& testName,
      size_t numRows,
      int numVectors,
      const RowTypePtr& rowType,
      int numKeys)
      : testName{testName},
        numRows{numRows},
        numVectors{numVectors},
        rowType{rowType},
        numKeys{numKeys} {}

  std::string testName;
  size_t numRows;
  int numVectors;
  RowTypePtr rowType;
  int numKeys;
};

class OrderByBenchmark {
 public:
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
    bigint(true, 1, batchSizes);
  }

  void largeBigintWithPayloads() {
    const std::vector<vector_size_t> batchSizes = {
        1'000, 10'000, 100'000, 1'000'000};
    bigint(false, 1, batchSizes);
  }

  void largeVarchar() {
    const std::vector<vector_size_t> batchSizes = {
        1'000, 10'000, 100'000, 1'000'000};
    std::vector<RowTypePtr> rowTypes =
        OrderByBenchmarkUtil::largeVarcharRowTypes();
    std::vector<int> numKeys = {1, 2, 3, 4};
    benchmark("no-payloads", "varchar", batchSizes, rowTypes, numKeys, 1);
  }

 private:
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

  void addBenchmark(
      const std::string& testName,
      size_t numRows,
      const RowTypePtr& rowType,
      int numVectors,
      int numKeys) {
    auto testCase = std::make_unique<TestCase>(
        testName, numRows, numVectors, rowType, numKeys);
    {
      folly::addBenchmark(
          __FILE__,
          "OrderBy_" + testCase->testName,
          [test = testCase.get(), this]() {
            auto plan = makeOrderByPlan(*test);
            run(plan);
            return 1;
          });
    }
    testCases_.push_back(std::move(testCase));
  }

  void bigint(
      bool noPayload,
      int numVectors,
      const std::vector<vector_size_t>& batchSizes) {
    std::vector<RowTypePtr> rowTypes =
        OrderByBenchmarkUtil::bigintRowTypes(noPayload);
    std::vector<int> numKeys = {1, 2, 3, 4};
    benchmark(
        noPayload ? "no-payload" : "payload",
        "bigint",
        batchSizes,
        rowTypes,
        numKeys,
        numVectors);
  }

  core::PlanNodePtr makeOrderByPlan(TestCase& test) {
    folly::BenchmarkSuspender suspender;
    std::vector<RowVectorPtr> vectors;
    for (auto i = 0; i < test.numVectors; ++i) {
      vectors.emplace_back(OrderByBenchmarkUtil::fuzzRows(
          test.rowType, test.numRows, test.numKeys, pool_.get()));
    }
    return makeOrderByPlan(vectors, makeOrderByKeys(test.numKeys));
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

  int64_t run(core::PlanNodePtr plan) {
    auto start = getCurrentTimeMicro();
    auto result = exec::test::AssertQueryBuilder(plan).copyResults(pool_.get());
    auto elapsedMicros = getCurrentTimeMicro() - start;
    return elapsedMicros;
  }

  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{
      rootPool_->addLeafChild("OrderByBenchmark")};
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
