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
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"

using facebook::velox::exec::test::PlanBuilder;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::functions::aggregate::sparksql::test {
namespace {
class DecimalSumAggregateTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerAggregateFunctions("spark_");
    allowInputShuffle();
  }

 protected:
  // check global partial agg overflow, and final agg output null
  void decimalGlobalSumOverflow(
      const std::vector<std::optional<int128_t>>& input,
      const std::vector<std::optional<int128_t>>& output) {
    const TypePtr type = DECIMAL(38, 0);
    auto in = makeRowVector({makeNullableFlatVector<int128_t>({input}, type)});
    auto expected =
        makeRowVector({makeNullableFlatVector<int128_t>({output}, type)});
    PlanBuilder builder(pool());
    builder.values({in});
    builder.partialAggregation({}, {"spark_sum(c0)"}).finalAggregation();
    AssertQueryBuilder queryBuilder(
        builder.planNode(), this->duckDbQueryRunner_);
    queryBuilder.assertResults({expected});
  }

  // check group by partial agg overflow, and final agg output null
  void decimalGroupBySumOverflow(
      const std::vector<std::optional<int128_t>>& input) {
    const TypePtr type = DECIMAL(38, 0);
    auto in = makeRowVector(
        {makeFlatVector<int32_t>(20, [](auto row) { return row % 10; }),
         makeNullableFlatVector<int128_t>(input, type)});
    auto expected = makeRowVector(
        {makeFlatVector<int32_t>(10, [](auto row) { return row; }),
         makeNullableFlatVector<int128_t>(
             std::vector<std::optional<int128_t>>(10, std::nullopt), type)});
    PlanBuilder builder(pool());
    builder.values({in});
    builder.partialAggregation({"c0"}, {"spark_sum(c1)"}).finalAggregation();
    AssertQueryBuilder queryBuilder(
        builder.planNode(), this->duckDbQueryRunner_);
    queryBuilder.assertResults({expected});
  }

  template <typename TIn, typename TOut>
  void decimalSumAllNulls(
      const std::vector<std::optional<TIn>>& input,
      const TypePtr& inputType,
      const std::vector<std::optional<TOut>>& output,
      const TypePtr& outputType) {
    std::vector<RowVectorPtr> vectors;
    FlatVectorPtr<TIn> inputDecimalVector;
    if constexpr (std::is_same_v<int64_t, TIn>) {
      inputDecimalVector = makeNullableFlatVector<int64_t>(input, inputType);
    } else {
      inputDecimalVector = makeNullableFlatVector<int128_t>(input, inputType);
    }
    for (int i = 0; i < 5; ++i) {
      vectors.emplace_back(makeRowVector(
          {makeFlatVector<int32_t>(20, [](auto row) { return row % 4; }),
           inputDecimalVector}));
    }

    FlatVectorPtr<TOut> outputDecimalVector;
    if constexpr (std::is_same_v<int64_t, TOut>) {
      outputDecimalVector = makeNullableFlatVector<int64_t>(output, outputType);
    } else {
      outputDecimalVector =
          makeNullableFlatVector<int128_t>(output, outputType);
    }
    auto expected = makeRowVector(
        {makeFlatVector<int32_t>(std::vector<int32_t>{0, 1, 2, 3}),
         outputDecimalVector});
    PlanBuilder builder(pool());
    builder.values({vectors});
    builder.singleAggregation({"c0"}, {"spark_sum(c1)"});
    AssertQueryBuilder queryBuilder(
        builder.planNode(), this->duckDbQueryRunner_);
    queryBuilder.assertResults({expected});
  }
};

/*
TEST_F(DecimalSumAggregateTest, sumDecimal) {
  std::vector<std::optional<int64_t>> shortDecimalRawVector;
  std::vector<std::optional<int128_t>> longDecimalRawVector;
  for (int i = 0; i < 1000; ++i) {
    shortDecimalRawVector.emplace_back(i * 1000);
    longDecimalRawVector.emplace_back(HugeInt::build(i * 10, i * 100));
  }
  shortDecimalRawVector.emplace_back(std::nullopt);
  longDecimalRawVector.emplace_back(std::nullopt);
  auto input = makeRowVector(
      {makeNullableFlatVector<int64_t>(shortDecimalRawVector, DECIMAL(10, 1)),
       makeNullableFlatVector<int128_t>(longDecimalRawVector, DECIMAL(23, 4))});
  createDuckDbTable({input});
  testAggregations(
      {input},
      {},
      {"spark_sum(c0)", "spark_sum(c1)"},
      "SELECT sum(c0), sum(c1) FROM tmp");

  // Short decimal sum aggregation with multiple groups.
  auto inputShortDecimalRows = {
      makeRowVector(
          {makeNullableFlatVector<int32_t>({1, 1}),
           makeFlatVector<int64_t>(
               std::vector<int64_t>{37220, 53450}, DECIMAL(5, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({2, 2}),
           makeFlatVector<int64_t>(
               std::vector<int64_t>{10410, 9250}, DECIMAL(5, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({3, 3}),
           makeFlatVector<int64_t>(
               std::vector<int64_t>{-12783, 0}, DECIMAL(5, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({1, 2}),
           makeFlatVector<int64_t>(
               std::vector<int64_t>{23178, 41093}, DECIMAL(5, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({2, 3}),
           makeFlatVector<int64_t>(
               std::vector<int64_t>{-10023, 5290}, DECIMAL(5, 2))}),
  };

  auto expectedShortDecimalResult = {
      makeRowVector(
          {makeNullableFlatVector<int32_t>({1}),
           makeFlatVector<int64_t>(
               std::vector<int64_t>{113848}, DECIMAL(15, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({2}),
           makeFlatVector<int64_t>(
               std::vector<int64_t>{50730}, DECIMAL(15, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({3}),
           makeFlatVector<int64_t>(
               std::vector<int64_t>{-7493}, DECIMAL(15, 2))})};

  testAggregations(
      inputShortDecimalRows,
      {"c0"},
      {"spark_sum(c1)"},
      expectedShortDecimalResult);

  // Long decimal sum aggregation with multiple groups.
  auto inputLongDecimalRows = {
      makeRowVector(
          {makeNullableFlatVector<int32_t>({1, 1}),
           makeFlatVector<int128_t>(
               {HugeInt::build(13, 113848), HugeInt::build(12, 53450)},
               DECIMAL(20, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({2, 2}),
           makeFlatVector<int128_t>(
               {HugeInt::build(21, 10410), HugeInt::build(17, 9250)},
               DECIMAL(20, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({3, 3}),
           makeFlatVector<int128_t>(
               {HugeInt::build(25, 12783), HugeInt::build(19, 0)},
               DECIMAL(20, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({1, 2}),
           makeFlatVector<int128_t>(
               {HugeInt::build(31, 23178), HugeInt::build(82, 41093)},
               DECIMAL(20, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({2, 3}),
           makeFlatVector<int128_t>(
               {HugeInt::build(25, 10023), HugeInt::build(43, 5290)},
               DECIMAL(20, 2))}),
  };

  auto expectedLongDecimalResult = {
      makeRowVector(
          {makeNullableFlatVector<int32_t>({1}),
           makeFlatVector<int128_t>(
               std::vector<int128_t>{HugeInt::build(56, 190476)},
               DECIMAL(38, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({2}),
           makeFlatVector<int128_t>(
               std::vector<int128_t>{HugeInt::build(145, 70776)},
               DECIMAL(38, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({3}),
           makeFlatVector<int128_t>(
               std::vector<int128_t>{HugeInt::build(87, 18073)},
               DECIMAL(38, 2))})};

  testAggregations(
      inputLongDecimalRows,
      {"c0"},
      {"spark_sum(c1)"},
      expectedLongDecimalResult);
}
*/

TEST_F(DecimalSumAggregateTest, globalSumDecimalOverflow) {
  // Test Positive Overflow.
  std::vector<std::optional<int128_t>> longDecimalInput;
  std::vector<std::optional<int128_t>> longDecimalOutput;
  // Create input with 2 kLongDecimalMax.
  longDecimalInput.emplace_back(DecimalUtil::kLongDecimalMax);
  longDecimalInput.emplace_back(DecimalUtil::kLongDecimalMax);
  // The sum must overflow, and will return null
  decimalGlobalSumOverflow(longDecimalInput, {std::nullopt});

  // Now add kLongDecimalMin.
  // The sum now must not overflow.
  longDecimalInput.emplace_back(DecimalUtil::kLongDecimalMin);
  longDecimalOutput.emplace_back(DecimalUtil::kLongDecimalMax);
  decimalGlobalSumOverflow(longDecimalInput, longDecimalOutput);

  // Test Negative Overflow.
  longDecimalInput.clear();
  longDecimalOutput.clear();

  // Create input with 2 kLongDecimalMin.
  longDecimalInput.emplace_back(DecimalUtil::kLongDecimalMin);
  longDecimalInput.emplace_back(DecimalUtil::kLongDecimalMin);

  // The sum must overflow, and will return null
  decimalGlobalSumOverflow(longDecimalInput, {std::nullopt});

  // Now add kLongDecimalMax.
  // The sum now must not overflow.
  longDecimalInput.emplace_back(DecimalUtil::kLongDecimalMax);
  longDecimalOutput.emplace_back(DecimalUtil::kLongDecimalMin);
  decimalGlobalSumOverflow(longDecimalInput, longDecimalOutput);

  // Check value in range.
  longDecimalInput.clear();
  longDecimalInput.emplace_back(DecimalUtil::kLongDecimalMax);
  longDecimalInput.emplace_back(1);
  decimalGlobalSumOverflow(longDecimalInput, {std::nullopt});

  longDecimalInput.clear();
  longDecimalInput.emplace_back(DecimalUtil::kLongDecimalMin);
  longDecimalInput.emplace_back(-1);
  decimalGlobalSumOverflow(longDecimalInput, {std::nullopt});
}

TEST_F(DecimalSumAggregateTest, groupBySumDecimalOverflow) {
  // Test Positive Overflow.
  decimalGroupBySumOverflow(
      std::vector<std::optional<int128_t>>(20, DecimalUtil::kLongDecimalMax));

  // Test Negative Overflow.
  decimalGroupBySumOverflow(
      std::vector<std::optional<int128_t>>(20, DecimalUtil::kLongDecimalMin));

  // Check value in range.
  auto decimalVector =
      std::vector<std::optional<int128_t>>(10, DecimalUtil::kLongDecimalMax);
  auto oneValueVector = std::vector<std::optional<int128_t>>(10, 1);
  decimalVector.insert(
      decimalVector.end(), oneValueVector.begin(), oneValueVector.end());
  decimalGroupBySumOverflow(decimalVector);

  decimalVector =
      std::vector<std::optional<int128_t>>(10, DecimalUtil::kLongDecimalMin);
  oneValueVector = std::vector<std::optional<int128_t>>(10, -1);
  decimalVector.insert(
      decimalVector.end(), oneValueVector.begin(), oneValueVector.end());
  decimalGroupBySumOverflow(decimalVector);
}

/// Test if all values in some groups are null,
/// the final sum of this group should be null.
TEST_F(DecimalSumAggregateTest, someGroupsAllnullValues) {
  std::vector<std::optional<int64_t>> shortDecimalNulls(20);
  std::vector<std::optional<int128_t>> longDecimalNulls(20);
  for (int i = 0; i < 20; i++) {
    if (i % 4 == 1 || i % 4 == 3) {
      // not all groups are null
      shortDecimalNulls[i] = 1;
      longDecimalNulls[i] = 1;
    }
  }

  // Test short decimal inputs and the output sum is short decimal.
  decimalSumAllNulls<int64_t, int64_t>(
      shortDecimalNulls,
      DECIMAL(7, 2),
      std::vector<std::optional<int64_t>>{std::nullopt, 25, std::nullopt, 25},
      DECIMAL(17, 2));

  // Test short decimal inputs and the output sum is long decimal.
  decimalSumAllNulls<int64_t, int128_t>(
      shortDecimalNulls,
      DECIMAL(17, 2),
      std::vector<std::optional<int128_t>>{std::nullopt, 25, std::nullopt, 25},
      DECIMAL(27, 2));

  // Test long decimal inputs and the output sum is long decimal.
  decimalSumAllNulls<int128_t, int128_t>(
      longDecimalNulls,
      DECIMAL(25, 2),
      std::vector<std::optional<int128_t>>{std::nullopt, 25, std::nullopt, 25},
      DECIMAL(35, 2));
}
} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
