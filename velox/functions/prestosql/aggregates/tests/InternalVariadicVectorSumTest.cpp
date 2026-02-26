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
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/prestosql/aggregates/InternalVariadicVectorSumAggregate.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::test {

namespace {

/// Tests for the internal variadic vector sum aggregate function:
/// $internal$variadic_vector_sum(T, T, ..., T) -> array(T)
/// This is an internal function used by optimizer transformations.
class InternalVariadicVectorSumTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    prestosql::registerInternalAggregateFunctions("");
  }
};

TEST_F(InternalVariadicVectorSumTest, globalTwoColumns) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, 10, 9}),
      makeNullableFlatVector<int64_t>({2, 5, std::nullopt}),
  });

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({{20, 7}}),
  });

  testAggregations(
      {data}, {}, {"\"$internal$variadic_vector_sum\"(c0, c1)"}, {expected});
}

TEST_F(InternalVariadicVectorSumTest, globalThreeColumns) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, 10, 9}),
      makeNullableFlatVector<int64_t>({2, 5, std::nullopt}),
      makeNullableFlatVector<int64_t>({3, 4, 5}),
  });

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({{20, 7, 12}}),
  });

  testAggregations(
      {data},
      {},
      {"\"$internal$variadic_vector_sum\"(c0, c1, c2)"},
      {expected});
}

TEST_F(InternalVariadicVectorSumTest, globalWithNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 9}),
      makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}),
      makeNullableFlatVector<int64_t>({3, 4, std::nullopt}),
  });

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({{10, 0, 7}}),
  });

  testAggregations(
      {data},
      {},
      {"\"$internal$variadic_vector_sum\"(c0, c1, c2)"},
      {expected});
}

TEST_F(InternalVariadicVectorSumTest, globalWithZeros) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({0, 0, 5}),
      makeNullableFlatVector<int64_t>({0, 3, 0}),
  });

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({{5, 3}}),
  });

  testAggregations(
      {data}, {}, {"\"$internal$variadic_vector_sum\"(c0, c1)"}, {expected});
}

TEST_F(InternalVariadicVectorSumTest, groupBy) {
  auto batch1 = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2}),
      makeNullableFlatVector<int64_t>({1, 10, 100}),
      makeNullableFlatVector<int64_t>({2, 5, 200}),
  });

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeArrayVector<int64_t>({{11, 7}, {100, 200}}),
  });

  testAggregations(
      {batch1},
      {"c0"},
      {"\"$internal$variadic_vector_sum\"(c1, c2)"},
      {expected});
}

TEST_F(InternalVariadicVectorSumTest, singleColumn) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, 10, 9}),
  });

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({{20}}),
  });

  testAggregations(
      {data}, {}, {"\"$internal$variadic_vector_sum\"(c0)"}, {expected});
}

TEST_F(InternalVariadicVectorSumTest, doubleType) {
  auto data = makeRowVector({
      makeNullableFlatVector<double>({1.5, 10.5}),
      makeNullableFlatVector<double>({2.5, 5.5}),
  });

  auto expected = makeRowVector({
      makeArrayVector<double>({{12.0, 8.0}}),
  });

  testAggregations(
      {data}, {}, {"\"$internal$variadic_vector_sum\"(c0, c1)"}, {expected});
}

TEST_F(InternalVariadicVectorSumTest, manyColumns) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1}),
      makeFlatVector<int64_t>({2, 2}),
      makeFlatVector<int64_t>({3, 3}),
      makeFlatVector<int64_t>({4, 4}),
      makeFlatVector<int64_t>({5, 5}),
      makeFlatVector<int64_t>({6, 6}),
      makeFlatVector<int64_t>({7, 7}),
      makeFlatVector<int64_t>({8, 8}),
      makeFlatVector<int64_t>({9, 9}),
      makeFlatVector<int64_t>({10, 10}),
  });

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({{2, 4, 6, 8, 10, 12, 14, 16, 18, 20}}),
  });

  testAggregations(
      {data},
      {},
      {"\"$internal$variadic_vector_sum\"(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)"},
      {expected});
}

TEST_F(InternalVariadicVectorSumTest, tinyintOverflow) {
  // c0 = {100, 30} -> sum = 130 -> overflow (exceeds 127)
  // c1 = {50, 50} -> sum = 100 -> no overflow
  auto data = makeRowVector({
      makeNullableFlatVector<int8_t>({100, 30}),
      makeNullableFlatVector<int8_t>({50, 50}),
  });

  auto plan =
      PlanBuilder()
          .values({data})
          .singleAggregation({}, {"\"$internal$variadic_vector_sum\"(c0, c1)"})
          .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()), "Value 130 exceeds 127");

  // Test negative overflow.
  auto negData = makeRowVector({
      makeNullableFlatVector<int8_t>({-100, -30}),
      makeNullableFlatVector<int8_t>({-50, -50}),
  });

  auto negPlan =
      PlanBuilder()
          .values({negData})
          .singleAggregation({}, {"\"$internal$variadic_vector_sum\"(c0, c1)"})
          .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(negPlan).copyResults(pool()),
      "Value -130 is less than -128");
}

TEST_F(InternalVariadicVectorSumTest, smallintOverflow) {
  const int16_t largeValue = std::numeric_limits<int16_t>::max() - 20;
  const int16_t smallValue = std::numeric_limits<int16_t>::min() + 20;

  // c0 = {largeValue, 30} -> sum overflows (exceeds 32767)
  // c1 = {50, 50} -> sum = 100 -> no overflow
  auto data = makeRowVector({
      makeNullableFlatVector<int16_t>({largeValue, 30}),
      makeNullableFlatVector<int16_t>({50, 50}),
  });

  auto plan =
      PlanBuilder()
          .values({data})
          .singleAggregation({}, {"\"$internal$variadic_vector_sum\"(c0, c1)"})
          .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value 32777 exceeds 32767");

  // Test negative overflow.
  auto negData = makeRowVector({
      makeNullableFlatVector<int16_t>({smallValue, -30}),
      makeNullableFlatVector<int16_t>({-50, -50}),
  });

  auto negPlan =
      PlanBuilder()
          .values({negData})
          .singleAggregation({}, {"\"$internal$variadic_vector_sum\"(c0, c1)"})
          .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(negPlan).copyResults(pool()),
      "Value -32778 is less than -32768");
}

TEST_F(InternalVariadicVectorSumTest, integerOverflow) {
  const int32_t largeValue = std::numeric_limits<int32_t>::max() - 20;
  const int32_t smallValue = std::numeric_limits<int32_t>::min() + 20;

  // c0 = {largeValue, 30} -> sum overflows (exceeds 2147483647)
  // c1 = {50, 50} -> sum = 100 -> no overflow
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>({largeValue, 30}),
      makeNullableFlatVector<int32_t>({50, 50}),
  });

  auto plan =
      PlanBuilder()
          .values({data})
          .singleAggregation({}, {"\"$internal$variadic_vector_sum\"(c0, c1)"})
          .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value 2147483657 exceeds 2147483647");

  // Test negative overflow.
  auto negData = makeRowVector({
      makeNullableFlatVector<int32_t>({smallValue, -30}),
      makeNullableFlatVector<int32_t>({-50, -50}),
  });

  auto negPlan =
      PlanBuilder()
          .values({negData})
          .singleAggregation({}, {"\"$internal$variadic_vector_sum\"(c0, c1)"})
          .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(negPlan).copyResults(pool()),
      "Value -2147483658 is less than -2147483648");
}

TEST_F(InternalVariadicVectorSumTest, bigintOverflow) {
  const int64_t largeValue = std::numeric_limits<int64_t>::max() - 10;

  // c0 = {largeValue, 20} -> sum = largeValue + 20 = max + 10 -> overflow
  // c1 = {1, 1} -> sum = 2 -> no overflow
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({largeValue, 20}),
      makeNullableFlatVector<int64_t>({1, 1}),
  });

  auto plan =
      PlanBuilder()
          .values({data})
          .singleAggregation({}, {"\"$internal$variadic_vector_sum\"(c0, c1)"})
          .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "exceeds 9223372036854775807");
}

TEST_F(InternalVariadicVectorSumTest, realType) {
  auto data = makeRowVector({
      makeNullableFlatVector<float>({1.5F, 10.5F}),
      makeNullableFlatVector<float>({2.5F, 5.5F}),
  });

  auto expected = makeRowVector({
      makeArrayVector<float>({{12.0F, 8.0F}}),
  });

  testAggregations(
      {data}, {}, {"\"$internal$variadic_vector_sum\"(c0, c1)"}, {expected});
}

TEST_F(InternalVariadicVectorSumTest, floatNan) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

  auto data = makeRowVector({
      makeNullableFlatVector<float>({10.0F, kNan, 30.0F}),
      makeNullableFlatVector<float>({20.0F, 30.0F, kInf}),
  });

  auto expected = makeRowVector({
      makeArrayVector<float>({{kNan, kInf}}),
  });

  testAggregations(
      {data}, {}, {"\"$internal$variadic_vector_sum\"(c0, c1)"}, {expected});
}

TEST_F(InternalVariadicVectorSumTest, doubleNan) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

  auto data = makeRowVector({
      makeNullableFlatVector<double>({10.0, kNan, 30.0}),
      makeNullableFlatVector<double>({20.0, 30.0, kInf}),
  });

  auto expected = makeRowVector({
      makeArrayVector<double>({{kNan, kInf}}),
  });

  testAggregations(
      {data}, {}, {"\"$internal$variadic_vector_sum\"(c0, c1)"}, {expected});
}

/// SQL equivalence tests

TEST_F(InternalVariadicVectorSumTest, mismatchedIntermediateArraySizes) {
  // Test that mismatched intermediate array sizes throw an error.
  // This can occur when combining partial aggregation results from different
  // workers where the intermediate arrays have different lengths.

  // Register with companion functions to enable the _merge function.
  prestosql::registerInternalVariadicVectorSumAggregate(
      {"test_variadic_vector_sum"}, true, true);

  // Create intermediate results with different array sizes.
  // First batch: arrays of size 2
  auto batch1 = makeRowVector({
      makeArrayVector<int64_t>({{10, 20}}),
  });

  // Second batch: arrays of size 3 (mismatched!)
  auto batch2 = makeRowVector({
      makeArrayVector<int64_t>({{1, 2, 3}}),
  });

  // Combining these intermediate results should fail because array sizes
  // don't match.
  auto plan = PlanBuilder()
                  .values({batch1, batch2})
                  .singleAggregation({}, {"test_variadic_vector_sum_merge(c0)"})
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "All arrays must have the same length");
}

/// SQL equivalence tests verify that $internal$variadic_vector_sum(c0, c1, ...)
/// produces the same result as array[sum(coalesce(c0,0)), sum(coalesce(c1,0)),
/// ...] This demonstrates the function is equivalent to element-wise column
/// sums.
class InternalVariadicVectorSumSqlEquivalenceTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    prestosql::registerInternalAggregateFunctions("");
  }
};

TEST_F(InternalVariadicVectorSumSqlEquivalenceTest, equivalentToColumnSums) {
  // $internal$variadic_vector_sum(c0, c1, c2) should equal
  // array[sum(c0), sum(c1), sum(c2)]
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 10, 100}),
      makeFlatVector<int64_t>({2, 20, 200}),
      makeFlatVector<int64_t>({3, 30, 300}),
  });

  // Using the internal variadic function
  auto variadicResult =
      AssertQueryBuilder(
          PlanBuilder()
              .values({data})
              .singleAggregation(
                  {}, {"\"$internal$variadic_vector_sum\"(c0, c1, c2)"})
              .planNode())
          .copyResults(pool());

  // Using individual sum() calls and constructing an array
  // sum(c0) = 111, sum(c1) = 222, sum(c2) = 333
  auto expected = makeRowVector({
      makeArrayVector<int64_t>({{111, 222, 333}}),
  });

  assertEqualResults({expected}, {variadicResult});
}

TEST_F(
    InternalVariadicVectorSumSqlEquivalenceTest,
    withNullsEquivalentToCoalesce) {
  // With nulls, $internal$variadic_vector_sum treats them as 0
  // This is equivalent to sum(coalesce(col, 0))
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 100}),
      makeNullableFlatVector<int64_t>({std::nullopt, 20, std::nullopt}),
  });

  // Using the internal variadic function
  auto variadicResult =
      AssertQueryBuilder(
          PlanBuilder()
              .values({data})
              .singleAggregation(
                  {}, {"\"$internal$variadic_vector_sum\"(c0, c1)"})
              .planNode())
          .copyResults(pool());

  // sum(coalesce(c0, 0)) = 1 + 0 + 100 = 101
  // sum(coalesce(c1, 0)) = 0 + 20 + 0 = 20
  auto expected = makeRowVector({
      makeArrayVector<int64_t>({{101, 20}}),
  });

  assertEqualResults({expected}, {variadicResult});
}

TEST_F(InternalVariadicVectorSumSqlEquivalenceTest, groupByEquivalence) {
  // For group by, each group's column sums should match
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2, 2}),
      makeFlatVector<int64_t>({10, 20, 100, 200}),
      makeFlatVector<int64_t>({1, 2, 10, 20}),
  });

  auto variadicResult =
      AssertQueryBuilder(
          PlanBuilder()
              .values({data})
              .singleAggregation(
                  {"c0"}, {"\"$internal$variadic_vector_sum\"(c1, c2)"})
              .planNode())
          .copyResults(pool());

  // Group 1: sum(c1) = 10+20 = 30, sum(c2) = 1+2 = 3 -> [30, 3]
  // Group 2: sum(c1) = 100+200 = 300, sum(c2) = 10+20 = 30 -> [300, 30]
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeArrayVector<int64_t>({{30, 3}, {300, 30}}),
  });

  assertEqualResults({expected}, {variadicResult});
}

TEST_F(InternalVariadicVectorSumSqlEquivalenceTest, floatingPointEquivalence) {
  // Verify floating point sums match
  auto data = makeRowVector({
      makeFlatVector<double>({0.1, 0.2, 0.3}),
      makeFlatVector<double>({1.0, 2.0, 3.0}),
  });

  auto variadicResult =
      AssertQueryBuilder(
          PlanBuilder()
              .values({data})
              .singleAggregation(
                  {}, {"\"$internal$variadic_vector_sum\"(c0, c1)"})
              .planNode())
          .copyResults(pool());

  // sum(c0) = 0.1 + 0.2 + 0.3 = 0.6
  // sum(c1) = 1.0 + 2.0 + 3.0 = 6.0
  auto expected = makeRowVector({
      makeArrayVector<double>({{0.6, 6.0}}),
  });

  assertEqualResults({expected}, {variadicResult});
}

TEST_F(InternalVariadicVectorSumSqlEquivalenceTest, singleColumnEquivalence) {
  // Single column case: $internal$variadic_vector_sum(c0) = array[sum(c0)]
  auto data = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40}),
  });

  auto variadicResult =
      AssertQueryBuilder(
          PlanBuilder()
              .values({data})
              .singleAggregation({}, {"\"$internal$variadic_vector_sum\"(c0)"})
              .planNode())
          .copyResults(pool());

  // sum(c0) = 10 + 20 + 30 + 40 = 100
  auto expected = makeRowVector({
      makeArrayVector<int64_t>({{100}}),
  });

  assertEqualResults({expected}, {variadicResult});
}

} // namespace

} // namespace facebook::velox::aggregate::test
