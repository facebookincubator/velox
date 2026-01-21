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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/Window.h"
#include "velox/functions/lib/window/tests/WindowTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"
#include "velox/functions/sparksql/window/WindowFunctionsRegistration.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::window::test {

namespace {

static const std::vector<std::string> kSparkWindowFunctions = {
    std::string("nth_value(c0, 1)"),
    std::string("nth_value(c0, c3)"),
    std::string("row_number()"),
    std::string("rank()"),
    std::string("dense_rank()"),
    std::string("ntile(c3)"),
    std::string("ntile(1)"),
    std::string("ntile(7)"),
    std::string("ntile(10)"),
    std::string("ntile(16)")};

struct SparkWindowTestParam {
  const std::string function;
  const std::string overClause;
};

class SparkWindowTestBase : public WindowTestBase {
 protected:
  explicit SparkWindowTestBase(const SparkWindowTestParam& testParam)
      : function_(testParam.function), overClause_(testParam.overClause) {}

  void testWindowFunction(const std::vector<RowVectorPtr>& vectors) {
    WindowTestBase::testWindowFunction(vectors, function_, {overClause_});
  }

  void SetUp() override {
    WindowTestBase::SetUp();
    WindowTestBase::options_.parseIntegerAsBigint = false;
    velox::functions::window::sparksql::registerWindowFunctions("");
  }

  const std::string function_;
  const std::string overClause_;
};

std::vector<SparkWindowTestParam> getSparkWindowTestParams() {
  std::vector<SparkWindowTestParam> params;

  for (auto function : kSparkWindowFunctions) {
    for (auto overClause : kOverClauses) {
      params.push_back({function, overClause});
    }
  }

  return params;
}

class SparkWindowTest
    : public SparkWindowTestBase,
      public testing::WithParamInterface<SparkWindowTestParam> {
 public:
  SparkWindowTest() : SparkWindowTestBase(GetParam()) {}
};

// Tests all functions with a dataset with uniform distribution of partitions.
TEST_P(SparkWindowTest, basic) {
  testWindowFunction({makeSimpleVector(40)});
}

// Tests all functions with a dataset with all rows in a single partition,
// but in 2 input vectors.
TEST_P(SparkWindowTest, singlePartition) {
  testWindowFunction(
      {makeSinglePartitionVector(40), makeSinglePartitionVector(50)});
}

// Tests all functions with a dataset in which all partitions have a single row.
TEST_P(SparkWindowTest, singleRowPartitions) {
  testWindowFunction({makeSingleRowPartitionsVector(40)});
}

// Tests all functions with a dataset with randomly generated data.
TEST_P(SparkWindowTest, randomInput) {
  testWindowFunction({makeRandomInputVector(30)});
}

// Run above tests for all combinations of rank function and over clauses.
VELOX_INSTANTIATE_TEST_SUITE_P(
    SparkWindowTestInstantiation,
    SparkWindowTest,
    testing::ValuesIn(getSparkWindowTestParams()));

class SparkAggregateWindowTest : public WindowTestBase {
 public:
  void SetUp() override {
    WindowTestBase::SetUp();
    WindowTestBase::options_.parseIntegerAsBigint = false;
    velox::functions::aggregate::sparksql::registerAggregateFunctions("");
  }
};

DEBUG_ONLY_TEST_F(SparkAggregateWindowTest, destroyPreviousAccumulator) {
  const auto size = 100;
  auto input = makeRowVector(
      {"d", "p0", "s"},
      {
          // Payload Data.
          makeFlatVector<std::string>(size, [](auto row){ return std::string(1024, 'a'); }),
          // Partition key.
          makeFlatVector<int64_t>(size, [](auto row) { return row % 11; }),
          // Sorting key.
          makeFlatVector<int32_t>(size, [](auto row) { return row; }),
      });

  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values(split(input, 10))
                  .window({"last(d) over (partition by p0 order by s)"})
                  .planNode();

  const HashStringAllocator* stringAllocator = nullptr;
  uint64_t usedBytes = 0;
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Window::callApplyForPartitionRows",
      std::function<void(exec::Window*)>([&](exec::Window* windowOp) {
        if (stringAllocator == nullptr) {
          stringAllocator = windowOp->testingGetHashStringAllocator();
          // Record how many bytes have been used.
          usedBytes = stringAllocator->currentBytes();
        } else {
          // Because we will destroy previous created accumulator and every string in input
          // is of the same length, so here we check if the `usedBytes` is not changed.
          ASSERT_EQ(usedBytes, stringAllocator->currentBytes());
        }
      }));

  auto task =
      AssertQueryBuilder(plan, duckDbQueryRunner_)
          .config(core::QueryConfig::kPreferredOutputBatchBytes, "1024")
          .serialExecution(true)
          .assertResults(
              "SELECT *, last(d) over (partition by p0 order by s) "
              "FROM tmp ");
}

} // namespace
} // namespace facebook::velox::window::test
