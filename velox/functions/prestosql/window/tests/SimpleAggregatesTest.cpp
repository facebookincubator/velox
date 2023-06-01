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
#include "velox/functions/prestosql/window/tests/WindowTestBase.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::window::test {

namespace {

static const std::vector<std::string> kAggregateFunctions = {
    std::string("sum(c2)"),
    std::string("min(c2)"),
    std::string("max(c2)"),
    std::string("count(c2)"),
    std::string("avg(c2)"),
    std::string("sum(1)")};

// This AggregateWindowTestBase class is used to instantiate parameterized
// aggregate window function tests. The parameters are (function, over clause).
// The window function is tested for the over clause and all combinations of
// frame clauses. Doing so helps to construct input vectors and DuckDB table
// only once for the (function, over clause) combination over all frame clauses.
struct AggregateWindowTestParam {
  const std::string function;
  const std::string overClause;
};

class AggregateWindowTestBase : public WindowTestBase {
 protected:
  explicit AggregateWindowTestBase(const AggregateWindowTestParam& testParam)
      : function_(testParam.function), overClause_(testParam.overClause) {}

  void testWindowFunction(const std::vector<RowVectorPtr>& vectors) {
    WindowTestBase::testWindowFunction(
        vectors, function_, {overClause_}, kFrameClauses);
  }

  const std::string function_;
  const std::string overClause_;
};

std::vector<AggregateWindowTestParam> getAggregateTestParams() {
  std::vector<AggregateWindowTestParam> params;
  for (auto function : kAggregateFunctions) {
    for (auto overClause : kOverClauses) {
      params.push_back({function, overClause});
    }
  }
  return params;
}

class SimpleAggregatesTest
    : public AggregateWindowTestBase,
      public testing::WithParamInterface<AggregateWindowTestParam> {
 public:
  SimpleAggregatesTest() : AggregateWindowTestBase(GetParam()) {}
};

// Tests function with a dataset with uniform partitions.
TEST_P(SimpleAggregatesTest, basic) {
  testWindowFunction({makeSimpleVector(10)});
}

// Tests function with a dataset with a single partition containing all the
// rows.
TEST_P(SimpleAggregatesTest, singlePartition) {
  testWindowFunction({makeSinglePartitionVector(100)});
}

// Tests function with a dataset with a single partition but 2 input row
// vectors.
TEST_P(SimpleAggregatesTest, multiInput) {
  testWindowFunction(
      {makeSinglePartitionVector(250), makeSinglePartitionVector(50)});
}

// Tests function with a dataset where all partitions have a single row.
TEST_P(SimpleAggregatesTest, singleRowPartitions) {
  testWindowFunction({makeSingleRowPartitionsVector(50)});
}

// Tests function with a randomly generated input dataset.
TEST_P(SimpleAggregatesTest, randomInput) {
  testWindowFunction({makeRandomInputVector(50)});
}

// Tests function with a randomly generated input dataset.
TEST_P(SimpleAggregatesTest, rangeFrames) {
  testKRangeFrames(function_);
}

// Instantiate all the above tests for each combination of aggregate function
// and over clause.
VELOX_INSTANTIATE_TEST_SUITE_P(
    AggregatesTestInstantiation,
    SimpleAggregatesTest,
    testing::ValuesIn(getAggregateTestParams()));

class StringAggregatesTest : public WindowTestBase {};

// Test for an aggregate function with strings that needs out of line storage.
TEST_F(StringAggregatesTest, nonFixedWidthAggregate) {
  auto size = 10;
  auto input = {makeRowVector({
      makeRandomInputVector(BIGINT(), size, 0.2),
      makeRandomInputVector(SMALLINT(), size, 0.2),
      makeRandomInputVector(VARCHAR(), size, 0.3),
      makeRandomInputVector(VARCHAR(), size, 0.3),
  })};

  testWindowFunction(input, "min(c2)", kOverClauses);
  testWindowFunction(input, "max(c2)", kOverClauses);
}

class KPrecedingFollowingTest : public WindowTestBase {
 public:
  const std::vector<std::string> kRangeFrames = {
      "range between unbounded preceding and 1 following",
      "range between unbounded preceding and 2 following",
      "range between unbounded preceding and 3 following",
      "range between 1 preceding and unbounded following",
      "range between 2 preceding and unbounded following",
      "range between 3 preceding and unbounded following",
      "range between 1 preceding and 3 following",
      "range between 3 preceding and 1 following",
      "range between 2 preceding and 2 following"};
};

TEST_F(KPrecedingFollowingTest, rangeFrames1) {
  auto vectors = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2147483650, 3, 2, 2147483650}),
      makeFlatVector<std::string>({"1", "1", "1", "2", "1", "2"}),
  });

  const std::string overClause = "partition by c1 order by c0";
  const std::vector<std::string> kRangeFrames1 = {
      "range between current row and 2147483648 following",
  };
  testWindowFunction({vectors}, "count(c0)", {overClause}, kRangeFrames1);

  const std::vector<std::string> kRangeFrames2 = {
      "range between 2147483648 preceding and current row",
  };
  testWindowFunction({vectors}, "count(c0)", {overClause}, kRangeFrames2);
}

TEST_F(KPrecedingFollowingTest, rangeFrames2) {
  const std::vector<RowVectorPtr> vectors = {
      makeRowVector(
          {makeFlatVector<int64_t>({5, 6, 8, 9, 10, 2, 8, 9, 3}),
           makeFlatVector<std::string>(
               {"1", "1", "1", "1", "1", "2", "2", "2", "2"})}),
      // Has repeated sort key.
      makeRowVector(
          {makeFlatVector<int64_t>({5, 5, 3, 2, 8}),
           makeFlatVector<std::string>({"1", "1", "1", "2", "1"})}),
      makeRowVector(
          {makeFlatVector<int64_t>({5, 5, 4, 6, 3, 2, 8, 9, 9}),
           makeFlatVector<std::string>(
               {"1", "1", "2", "2", "1", "2", "1", "1", "2"})}),
      makeRowVector(
          {makeFlatVector<int64_t>({5, 5, 4, 6, 3, 2}),
           makeFlatVector<std::string>({"1", "2", "2", "2", "1", "2"})}),
      // Uses int32 type for sort column.
      makeRowVector(
          {makeFlatVector<int32_t>({5, 5, 4, 6, 3, 2}),
           makeFlatVector<std::string>({"1", "2", "2", "2", "1", "2"})}),
  };
  const std::string overClause = "partition by c1 order by c0";
  for (int i = 0; i < vectors.size(); i++) {
    testWindowFunction({vectors[i]}, "avg(c0)", {overClause}, kRangeFrames);
    testWindowFunction({vectors[i]}, "sum(c0)", {overClause}, kRangeFrames);
    testWindowFunction({vectors[i]}, "count(c0)", {overClause}, kRangeFrames);
  }
}

TEST_F(KPrecedingFollowingTest, rangeFrames3) {
  const std::vector<RowVectorPtr> vectors = {
      // Uses date type for sort column.
      makeRowVector(
          {makeFlatVector<Date>(
               {Date(6), Date(1), Date(5), Date(0), Date(7), Date(1)}),
           makeFlatVector<std::string>({"1", "2", "2", "2", "1", "2"})}),
      makeRowVector(
          {makeFlatVector<Date>(
               {Date(5), Date(5), Date(4), Date(6), Date(3), Date(2)}),
           makeFlatVector<std::string>({"1", "2", "2", "2", "1", "2"})}),
  };
  const std::string overClause = "partition by c1 order by c0";
  for (int i = 0; i < vectors.size(); i++) {
    testWindowFunction({vectors[i]}, "count(c0)", {overClause}, kRangeFrames);
  }
}

TEST_F(KPrecedingFollowingTest, rowsFrames) {
  auto vectors = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2147483650, 3, 2, 2147483650}),
      makeFlatVector<std::string>({"1", "1", "1", "2", "1", "2"}),
  });
  const std::string overClause = "partition by c1 order by c0";
  const std::vector<std::string> kRangeFrames = {
      "rows between current row and 2147483647 following",
  };
  testWindowFunction({vectors}, "count(c0)", {overClause}, kRangeFrames);
}

}; // namespace
}; // namespace facebook::velox::window::test
