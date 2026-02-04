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

class SparkAggregateWindowLimitMemoryTest
    : public exec::test::OperatorTestBase {
 public:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    OperatorTestBase::setupMemory(
        256 << 20, // allocatorCapacity
        256 << 20, // arbitratorCapacity
        0, // arbitratorReservedCapacity
        0, // memoryPoolInitCapacity
        0, // memoryPoolReservedCapacity
        0, // memoryPoolMinReclaimBytes
        0 // memoryPoolAbortCapacityLimit
    );
  }

  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    velox::functions::aggregate::sparksql::registerAggregateFunctions("");
  }
};

// This test limits the memory capacity to check that string buffers are cleared
// in time during the aggregation window function processing.
// Without timely clearing the string buffers, the memory usage would exceed
// the limit and cause failure.
// The capacity is calculated as:
//   1. size of input data: 1 rows * 32KB (the length of the string) = 32KB
//   2. size of RowContainer: ((1,024 rows * 32KB (data) + 1,024 rows * 32KB
//      (Accumulators))) * 3 = 192MB
//      (Accumulators won't be destroyed now)
//   3. size of results: 10 rows * 32KB * 2 (column 'd' and the result column) =
//      640KB
//   4. other overheads
//    Total: ~ 192MB
// If we don't clear the string buffers in time, the size of the string buffers
// would be at least 1,024 rows * 32KB * 3 = 96MB. So without the fix, we need
// capacity to be set to more than 192MB + 96MB = 288MB to pass the test.
// So we set the capacity to 256MB here.
TEST_F(SparkAggregateWindowLimitMemoryTest, clearStringBuffersInTime) {
  constexpr vector_size_t size = 1'024 * 3;
  constexpr vector_size_t resultSize = 10;
  // For this test, it is important to create a single-row partition.
  // When we process a new partition, we will call 'fillArgVectors' in
  // 'AggregateWindow'. Single-row partition can let us call 'fillArgVectors' as
  // many as possible. If we don't call 'prepareForReuse' in 'fillArgVectors',
  // the string buffers will accumulate and cause high memory usage.
  auto data = makeRowVector(
      {"d", "p"},
      {// Payload Data.
       makeConstant(std::string(32 * 1024, 'a'), size),
       // Partition key.
       makeFlatVector<int16_t>(size, [](auto row) { return row; })});

  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .streamingWindow({"first(d) over (partition by p)"})
                  // We use LIMIT clause here to limit the result size to limit
                  // the memory consumed by results.
                  // Besides, we set offset to 'size - resultSize' to make the
                  // Window operator process all window partitions. In this way,
                  // if we don't clear the string buffers during the window
                  // function processing, the string buffers would accumulate to
                  // cause high memory usage.
                  // TODO: remove the use of LIMIT clause
                  // after fixing
                  // https://github.com/facebookincubator/velox/issues/16210
                  .limit(size - resultSize, resultSize, true)
                  .planNode();

  const auto realResultSize =
      AssertQueryBuilder(plan)
          .serialExecution(true)
          // Because we set offset to 'size - resultSize' in the LIMIT clause,
          // the Window operator will produce intermediate results that will be
          // discarded by LIMIT clause.
          // Set preferred output batch rows to 10 to limit the memory
          // consumed by these intermediate results.
          // By limiting the memory consumed by input and intermediate results,
          // we can know that the reason why we can pass this test is that we
          // clear the string buffers in time during the window function
          // processing.
          .config(core::QueryConfig::kPreferredOutputBatchRows, resultSize)
          .countResults();
  ASSERT_EQ(realResultSize, resultSize);
}

} // namespace
} // namespace facebook::velox::window::test
