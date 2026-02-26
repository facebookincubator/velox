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

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::test {

namespace {

class VectorSumTest : public AggregationTestBase {};

TEST_F(VectorSumTest, global) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{1, 2, 3, 0}},
      {{10, 5, 4, 1}},
      {{9, std::nullopt, 5, 4}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {20, 7, 12, 5},
      }),
  });

  testAggregations({data}, {}, {"vector_sum(c0)"}, {expected});
}

TEST_F(VectorSumTest, globalWithNulls) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{1, std::nullopt, 3}},
      {{10, 5, std::nullopt}},
      {{std::nullopt, std::nullopt, 5}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {11, 5, 8},
      }),
  });

  testAggregations({data}, {}, {"vector_sum(c0)"}, {expected});
}

TEST_F(VectorSumTest, differentLengthsThrowsError) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{1, 2}},
      {{10, 5, 4, 1}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"vector_sum(c0)"})
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "All arrays must have the same length. Expected 2, but got 4.");
}

TEST_F(VectorSumTest, emptyArrays) {
  auto allEmptyArrays = makeRowVector({
      makeArrayVector<int64_t>({
          {},
          {},
          {},
      }),
  });

  auto expectedEmpty = makeRowVector({
      makeArrayVector<int64_t>({
          {},
      }),
  });

  testAggregations({allEmptyArrays}, {}, {"vector_sum(c0)"}, {expectedEmpty});
}

TEST_F(VectorSumTest, nullArrays) {
  auto allNullArrays = makeRowVector(
      {BaseVector::createNullConstant(ARRAY(BIGINT()), 3, pool())});

  auto expectedNull = makeRowVector(
      {BaseVector::createNullConstant(ARRAY(BIGINT()), 1, pool())});

  testAggregations({allNullArrays}, {}, {"vector_sum(c0)"}, {expectedNull});
}

TEST_F(VectorSumTest, tinyintOverflow) {
  const std::vector<std::vector<std::optional<int8_t>>> inputData = {
      {{10, 20}},
      {{100, 30}},
      {{30, 50}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"vector_sum(c0)"})
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()), "Value 140 exceeds 127");

  const std::vector<std::vector<std::optional<int8_t>>> negInputData = {
      {{-10, -20}},
      {{-100, -30}},
      {{-30, -50}},
  };
  data = makeRowVector({makeNullableArrayVector(negInputData)});

  plan = PlanBuilder()
             .values({data})
             .singleAggregation({}, {"vector_sum(c0)"})
             .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value -140 is less than -128");
}

TEST_F(VectorSumTest, smallintOverflow) {
  const int16_t largeValue = std::numeric_limits<int16_t>::max() - 20;
  const int16_t smallValue = std::numeric_limits<int16_t>::min() + 20;

  const std::vector<std::vector<std::optional<int16_t>>> inputData = {
      {{10, 20}},
      {{largeValue, 30}},
      {{30, 50}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"vector_sum(c0)"})
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value 32787 exceeds 32767");

  const std::vector<std::vector<std::optional<int16_t>>> negInputData = {
      {{-10, -20}},
      {{smallValue, -30}},
      {{-30, -50}},
  };
  data = makeRowVector({makeNullableArrayVector(negInputData)});

  plan = PlanBuilder()
             .values({data})
             .singleAggregation({}, {"vector_sum(c0)"})
             .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value -32788 is less than -32768");
}

TEST_F(VectorSumTest, integerOverflow) {
  const int32_t largeValue = std::numeric_limits<int32_t>::max() - 20;
  const int32_t smallValue = std::numeric_limits<int32_t>::min() + 20;

  const std::vector<std::vector<std::optional<int32_t>>> inputData = {
      {{10, 20}},
      {{largeValue, 30}},
      {{30, 50}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"vector_sum(c0)"})
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value 2147483667 exceeds 2147483647");

  const std::vector<std::vector<std::optional<int32_t>>> negInputData = {
      {{-10, -20}},
      {{smallValue, -30}},
      {{-30, -50}},
  };
  data = makeRowVector({makeNullableArrayVector(negInputData)});

  plan = PlanBuilder()
             .values({data})
             .singleAggregation({}, {"vector_sum(c0)"})
             .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value -2147483668 is less than -2147483648");
}

TEST_F(VectorSumTest, bigintOverflow) {
  const int64_t largeValue = std::numeric_limits<int64_t>::max() - 20;
  const int64_t smallValue = std::numeric_limits<int64_t>::min() + 20;

  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{10, 20}},
      {{largeValue, 30}},
      {{30, 50}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"vector_sum(c0)"})
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value 9223372036854775827 exceeds 9223372036854775807");

  const std::vector<std::vector<std::optional<int64_t>>> negInputData = {
      {{-10, -20}},
      {{smallValue, -30}},
      {{-30, -50}},
  };
  data = makeRowVector({makeNullableArrayVector(negInputData)});

  plan = PlanBuilder()
             .values({data})
             .singleAggregation({}, {"vector_sum(c0)"})
             .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value -9223372036854775828 is less than -9223372036854775808");
}

TEST_F(VectorSumTest, floatNan) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

  const std::vector<std::vector<std::optional<float>>> inputData = {
      {{10.0F, 20.0F}},
      {{kNan, 30.0F}},
      {{30.0F, kInf}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<float>({
          {kNan, kInf},
      }),
  });

  testAggregations({data}, {}, {"vector_sum(c0)"}, {expected});
}

TEST_F(VectorSumTest, doubleNan) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

  const std::vector<std::vector<std::optional<double>>> inputData = {
      {{10.0, 20.0}},
      {{kNan, 30.0}},
      {{30.0, kInf}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<double>({
          {kNan, kInf},
      }),
  });

  testAggregations({data}, {}, {"vector_sum(c0)"}, {expected});
}

TEST_F(VectorSumTest, groupBy) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData1 = {
      {{1, 2, 3}},
  };
  const std::vector<std::vector<std::optional<int64_t>>> inputData2 = {
      {{10, 5, 4}},
  };
  const std::vector<std::vector<std::optional<int64_t>>> inputData3 = {
      {{1, 2, 7}},
  };
  const std::vector<std::vector<std::optional<int64_t>>> inputData4 = {
      {{9, 0, 5}},
  };

  auto batch1 = makeRowVector({
      makeFlatVector<int64_t>({1}),
      makeNullableArrayVector(inputData1),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int64_t>({1}),
      makeNullableArrayVector(inputData2),
  });
  auto batch3 = makeRowVector({
      makeFlatVector<int64_t>({2}),
      makeNullableArrayVector(inputData3),
  });
  auto batch4 = makeRowVector({
      makeFlatVector<int64_t>({1}),
      makeNullableArrayVector(inputData4),
  });

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeArrayVector<int64_t>({
          {20, 7, 12},
          {1, 2, 7},
      }),
  });

  testAggregations(
      {batch1, batch2, batch3, batch4}, {"c0"}, {"vector_sum(c1)"}, {expected});
}

TEST_F(VectorSumTest, zerosSkipped) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{0, 0, 0, 0}},
      {{0, 0, 0, 0}},
      {{0, 0, 0, 0}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {0, 0, 0, 0},
      }),
  });

  testAggregations({data}, {}, {"vector_sum(c0)"}, {expected});
}

TEST_F(VectorSumTest, mixedZerosNullsAndValues) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{0, std::nullopt, 5, 0}},
      {{std::nullopt, 0, 0, 3}},
      {{7, 0, std::nullopt, 0}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {7, 0, 5, 3},
      }),
  });

  testAggregations({data}, {}, {"vector_sum(c0)"}, {expected});
}

TEST_F(VectorSumTest, allZerosAndNulls) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{0, std::nullopt}},
      {{std::nullopt, 0}},
      {{0, 0}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {0, 0},
      }),
  });

  testAggregations({data}, {}, {"vector_sum(c0)"}, {expected});
}

TEST_F(VectorSumTest, zerosWithFloats) {
  const std::vector<std::vector<std::optional<float>>> inputData = {
      {{0.0F, 1.5F, 0.0F}},
      {{3.5F, 0.0F, 0.0F}},
      {{0.0F, 0.0F, 2.5F}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<float>({
          {3.5F, 1.5F, 2.5F},
      }),
  });

  testAggregations({data}, {}, {"vector_sum(c0)"}, {expected});
}

TEST_F(VectorSumTest, zerosWithDoubles) {
  const std::vector<std::vector<std::optional<double>>> inputData = {
      {{0.0, 1.5, 0.0}},
      {{3.5, 0.0, 0.0}},
      {{0.0, 0.0, 2.5}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<double>({
          {3.5, 1.5, 2.5},
      }),
  });

  testAggregations({data}, {}, {"vector_sum(c0)"}, {expected});
}

TEST_F(VectorSumTest, singleElement) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{100}},
      {{200}},
      {{300}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {600},
      }),
  });

  testAggregations({data}, {}, {"vector_sum(c0)"}, {expected});
}

TEST_F(VectorSumTest, allNullElements) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{std::nullopt, std::nullopt}},
      {{std::nullopt, std::nullopt}},
      {{std::nullopt, std::nullopt}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {0, 0},
      }),
  });

  testAggregations({data}, {}, {"vector_sum(c0)"}, {expected});
}

TEST_F(VectorSumTest, realType) {
  const std::vector<std::vector<std::optional<float>>> inputData = {
      {{1.5F, 2.5F, 0.0F}},
      {{10.5F, 5.5F, 4.0F}},
      {{9.0F, std::nullopt, 5.5F}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<float>({
          {21.0F, 8.0F, 9.5F},
      }),
  });

  testAggregations({data}, {}, {"vector_sum(c0)"}, {expected});
}

TEST_F(VectorSumTest, doubleType) {
  const std::vector<std::vector<std::optional<double>>> inputData = {
      {{1.5, 2.5, 0.0}},
      {{10.5, 5.5, 4.0}},
      {{9.0, std::nullopt, 5.5}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<double>({
          {21.0, 8.0, 9.5},
      }),
  });

  testAggregations({data}, {}, {"vector_sum(c0)"}, {expected});
}

TEST_F(VectorSumTest, multipleBatches) {
  auto batch1 = makeRowVector({
      makeArrayVector<int64_t>({{1, 2, 3}, {4, 5, 6}}),
  });
  auto batch2 = makeRowVector({
      makeArrayVector<int64_t>({{10, 20, 30}, {40, 50, 60}}),
  });

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({{55, 77, 99}}),
  });

  testAggregations({batch1, batch2}, {}, {"vector_sum(c0)"}, {expected});
}

/// SQL equivalence tests verify vector_sum results by comparing against
/// element-wise sums computed using individual sum() aggregations.
/// This verifies that vector_sum(array[a,b,c]) equals array[sum(a), sum(b),
/// sum(c)] when arrays are of equal length.
class VectorSumSqlEquivalenceTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
  }
};

TEST_F(VectorSumSqlEquivalenceTest, equivalentToElementWiseSum) {
  // Create data where we can verify vector_sum against element-wise sums.
  // If we have rows with arrays [a0, a1], [b0, b1], [c0, c1]
  // then vector_sum should equal [a0+b0+c0, a1+b1+c1]
  // which is [sum of first elements, sum of second elements]
  auto data = makeRowVector({
      makeArrayVector<int64_t>({{1, 2}, {10, 20}, {100, 200}}),
  });

  // vector_sum result
  auto vectorSumResult = AssertQueryBuilder(
                             PlanBuilder()
                                 .values({data})
                                 .singleAggregation({}, {"vector_sum(c0)"})
                                 .planNode())
                             .copyResults(pool());

  // Expected: [1+10+100, 2+20+200] = [111, 222]
  auto expected = makeRowVector({
      makeArrayVector<int64_t>({{111, 222}}),
  });

  assertEqualResults({expected}, {vectorSumResult});
}

TEST_F(VectorSumSqlEquivalenceTest, groupByEquivalence) {
  // For group by, vector_sum should accumulate arrays within each group.
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2, 2}),
      makeArrayVector<int64_t>({{1, 2}, {10, 20}, {100, 200}, {1000, 2000}}),
  });

  auto vectorSumResult = AssertQueryBuilder(
                             PlanBuilder()
                                 .values({data})
                                 .singleAggregation({"c0"}, {"vector_sum(c1)"})
                                 .planNode())
                             .copyResults(pool());

  // Group 1: [1,2] + [10,20] = [11, 22]
  // Group 2: [100,200] + [1000,2000] = [1100, 2200]
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeArrayVector<int64_t>({{11, 22}, {1100, 2200}}),
  });

  assertEqualResults({expected}, {vectorSumResult});
}

TEST_F(VectorSumSqlEquivalenceTest, floatingPointPrecision) {
  // Verify floating point accumulation works correctly
  auto data = makeRowVector({
      makeArrayVector<double>({{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}),
  });

  auto vectorSumResult = AssertQueryBuilder(
                             PlanBuilder()
                                 .values({data})
                                 .singleAggregation({}, {"vector_sum(c0)"})
                                 .planNode())
                             .copyResults(pool());

  // Expected: [0.1+0.3+0.5, 0.2+0.4+0.6] = [0.9, 1.2]
  auto expected = makeRowVector({
      makeArrayVector<double>({{0.9, 1.2}}),
  });

  assertEqualResults({expected}, {vectorSumResult});
}

TEST_F(VectorSumSqlEquivalenceTest, nullElementsAreZero) {
  // Verify that null elements in arrays are treated as 0
  // This is equivalent to: coalesce(element, 0)
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{1, std::nullopt, 3}},
      {{10, 20, std::nullopt}},
      {{std::nullopt, 30, 40}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto vectorSumResult = AssertQueryBuilder(
                             PlanBuilder()
                                 .values({data})
                                 .singleAggregation({}, {"vector_sum(c0)"})
                                 .planNode())
                             .copyResults(pool());

  // Expected: [1+10+0, 0+20+30, 3+0+40] = [11, 50, 43]
  auto expected = makeRowVector({
      makeArrayVector<int64_t>({{11, 50, 43}}),
  });

  assertEqualResults({expected}, {vectorSumResult});
}

TEST_F(VectorSumSqlEquivalenceTest, differentLengthArraysThrowsError) {
  // Arrays of different lengths now throw an error
  auto data = makeRowVector({
      makeArrayVector<int64_t>({{1, 2, 3, 4}, {10, 20}, {100}}),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"vector_sum(c0)"})
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "All arrays must have the same length. Expected 4, but got 2.");
}

} // namespace

} // namespace facebook::velox::aggregate::test
