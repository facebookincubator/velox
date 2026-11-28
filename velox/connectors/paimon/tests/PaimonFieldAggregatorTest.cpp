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
#include "velox/connectors/paimon/PaimonFieldAggregator.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::connector::paimon {
namespace {

class PaimonFieldAggregatorTest : public testing::Test,
                                  public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  // One call on an aggregator, with the input row it reads.
  struct Step {
    enum class Kind { kAdd, kRetract, kSet };

    Kind kind;
    vector_size_t row;
  };

  static Step add(vector_size_t row) {
    return {Step::Kind::kAdd, row};
  }

  static Step retract(vector_size_t row) {
    return {Step::Kind::kRetract, row};
  }

  static Step set(vector_size_t row) {
    return {Step::Kind::kSet, row};
  }

  // Applies 'steps' to the rows of 'input' after a reset, and returns the
  // accumulator as a vector of one row.
  VectorPtr aggregate(
      const std::string& function,
      const VectorPtr& input,
      const std::vector<Step>& steps,
      bool ignoreRetract = false) {
    const auto aggregator = PaimonFieldAggregator::create(
        function, "c", input->type(), ignoreRetract);
    aggregator->reset();
    for (const auto& step : steps) {
      switch (step.kind) {
        case Step::Kind::kAdd:
          aggregator->add(*input, step.row);
          break;
        case Step::Kind::kRetract:
          aggregator->retract(*input, step.row);
          break;
        case Step::Kind::kSet:
          aggregator->set(*input, step.row);
          break;
      }
    }
    auto result = BaseVector::create(input->type(), 1, pool());
    aggregator->write(0, *result);
    return result;
  }

  // Checks that applying 'steps' to the rows of 'input' with 'function'
  // returns 'expected'.
  template <typename T>
  void assertAggregate(
      const std::string& function,
      const VectorPtr& input,
      const std::vector<Step>& steps,
      std::optional<T> expected) {
    test::assertEqualVectors(
        makeNullableFlatVector<T>({expected}, input->type()),
        aggregate(function, input, steps));
  }
};

// Functions that pick an input value, from Paimon Java's FieldPrimaryKeyAgg,
// FieldLastValueAgg, FieldLastNonNullValueAgg, FieldFirstValueAgg,
// FieldFirstNonNullValueAgg, FieldMinAgg and FieldMaxAgg.
TEST_F(PaimonFieldAggregatorTest, pickFunctions) {
  const auto input =
      makeNullableFlatVector<int64_t>({10, std::nullopt, 30, 20});
  struct TestParam {
    std::string testName;
    std::string function;
    std::vector<Step> steps;
    std::optional<int64_t> expected;

    std::string debugString() const {
      return fmt::format("{}, function {}", testName, function);
    }
  };
  const std::vector<TestParam> testSettings{
      {"takes the last value", "primary-key", {add(0), add(2), add(1)}, {}},
      {"retract takes the value", "primary-key", {add(0), retract(2)}, 30},
      {"takes the last value", "last_value", {add(0), add(1)}, {}},
      {"retract clears", "last_value", {add(0), add(2), retract(3)}, {}},
      {"skips nulls", "last_non_null_value", {add(0), add(1)}, 10},
      {"retracting a null keeps the value",
       "last_non_null_value",
       {add(0), retract(1)},
       10},
      {"retracting a value clears",
       "last_non_null_value",
       {add(0), retract(2)},
       {}},
      {"takes the first value", "first_value", {add(1), add(0)}, {}},
      {"takes the first non-null value",
       "first_non_null_value",
       {add(1), add(0), add(2)},
       10},
      {"legacy name", "first_not_null_value", {add(1), add(2), add(0)}, 30},
      {"takes the smallest value", "min", {add(2), add(1), add(0), add(3)}, 10},
      {"takes the largest value", "max", {add(1), add(0), add(2), add(3)}, 30},
      // A DELETE row replaces the accumulator but not the state of the
      // function.
      {"set keeps the state", "first_value", {add(0), set(2), add(3)}, 30},
      {"set before any value",
       "first_non_null_value",
       {set(2), add(0), add(3)},
       10},
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    assertAggregate<int64_t>(
        testData.function, input, testData.steps, testData.expected);
  }
}

// Paimon Java's FieldSumAgg wraps integers around on overflow, as Java
// arithmetic does, and negates the first retracted value.
TEST_F(PaimonFieldAggregatorTest, sumIntegers) {
  {
    const auto input = makeNullableFlatVector<int64_t>({1, 2, std::nullopt, 5});
    assertAggregate<int64_t>("sum", input, {add(0), add(1), add(2)}, 3);
    assertAggregate<int64_t>("sum", input, {add(0), add(1), retract(1)}, 1);
    assertAggregate<int64_t>("sum", input, {retract(3)}, -5);
    assertAggregate<int64_t>("sum", input, {add(0), retract(2)}, 1);
    assertAggregate<int64_t>("sum", input, {retract(2)}, std::nullopt);
    assertAggregate<int64_t>("sum", input, {add(2)}, std::nullopt);
  }
  {
    const auto input = makeFlatVector<int64_t>(
        {std::numeric_limits<int64_t>::max(),
         1,
         std::numeric_limits<int64_t>::min()});
    assertAggregate<int64_t>(
        "sum", input, {add(0), add(1)}, std::numeric_limits<int64_t>::min());
    assertAggregate<int64_t>(
        "sum", input, {retract(2)}, std::numeric_limits<int64_t>::min());
    assertAggregate<int64_t>(
        "sum",
        input,
        {add(2), retract(1)},
        std::numeric_limits<int64_t>::max());
  }
  {
    const auto input =
        makeFlatVector<int32_t>({std::numeric_limits<int32_t>::max(), 1});
    assertAggregate<int32_t>(
        "sum", input, {add(0), add(1)}, std::numeric_limits<int32_t>::min());
  }
  {
    const auto input = makeFlatVector<int16_t>({32'767, 1});
    assertAggregate<int16_t>("sum", input, {add(0), add(1)}, -32'768);
  }
  {
    const auto input = makeFlatVector<int8_t>({127, 1, -128});
    assertAggregate<int8_t>("sum", input, {add(0), add(1)}, -128);
    assertAggregate<int8_t>("sum", input, {retract(2)}, -128);
  }
  {
    const auto input = makeFlatVector<double>({1.5, 2.25});
    assertAggregate<double>("sum", input, {add(0), add(1)}, 3.75);
    assertAggregate<double>("sum", input, {retract(0)}, -1.5);
  }
  {
    const auto input = makeFlatVector<float>({1.5, 2.25});
    assertAggregate<float>("sum", input, {add(0), retract(1)}, -0.75);
  }
}

// Paimon Java sums decimals of precision up to 18 as longs and checks only for
// long overflow, which returns null. Wider decimals return null when the sum
// exceeds the precision.
TEST_F(PaimonFieldAggregatorTest, sumDecimals) {
  {
    const auto input = makeFlatVector<int64_t>({99'999, 1}, DECIMAL(5, 2));
    assertAggregate<int64_t>("sum", input, {add(0), add(1)}, 100'000);
    assertAggregate<int64_t>("sum", input, {retract(0)}, -99'999);
  }
  {
    constexpr int64_t kLarge{900'000'000'000'000'000};
    const auto input = makeFlatVector<int64_t>({kLarge, -1}, DECIMAL(18, 0));
    std::vector<Step> steps(10, add(0));
    assertAggregate<int64_t>("sum", input, steps, 10 * kLarge);
    // The 11th value overflows a long, and the next value starts over.
    steps.push_back(add(0));
    assertAggregate<int64_t>("sum", input, steps, std::nullopt);
    steps.push_back(add(1));
    assertAggregate<int64_t>("sum", input, steps, -1);
  }
  {
    const int128_t largest = DecimalUtil::kPowersOfTen[20] - 1;
    const auto input =
        makeFlatVector<int128_t>({largest, 1, -largest}, DECIMAL(20, 0));
    assertAggregate<int128_t>("sum", input, {add(0), add(1)}, std::nullopt);
    assertAggregate<int128_t>("sum", input, {add(2), retract(1)}, std::nullopt);
    assertAggregate<int128_t>("sum", input, {add(0), retract(1)}, largest - 1);
    assertAggregate<int128_t>("sum", input, {retract(2)}, largest);
  }
}

// Paimon Java's FieldProductAgg multiplies and retracts by dividing, with Java
// arithmetic for integers. Decimals round half up to the scale, and return
// null when the result exceeds the precision.
TEST_F(PaimonFieldAggregatorTest, product) {
  {
    const auto input = makeNullableFlatVector<int64_t>({3, 4, std::nullopt, 0});
    assertAggregate<int64_t>("product", input, {add(0), add(1)}, 12);
    assertAggregate<int64_t>("product", input, {add(0), add(1), retract(1)}, 3);
    assertAggregate<int64_t>("product", input, {add(0), add(2)}, 3);
    assertAggregate<int64_t>("product", input, {add(0), retract(2)}, 3);
    // Retracting from a null product keeps it null.
    assertAggregate<int64_t>("product", input, {retract(0)}, std::nullopt);
    VELOX_ASSERT_USER_THROW(
        aggregate("product", input, {add(0), retract(3)}),
        "Paimon aggregate function product cannot retract a zero: c");
  }
  {
    const auto input = makeFlatVector<int32_t>(
        {65'536, std::numeric_limits<int32_t>::min(), -1});
    assertAggregate<int32_t>("product", input, {add(0), add(0)}, 0);
    assertAggregate<int32_t>(
        "product",
        input,
        {add(1), retract(2)},
        std::numeric_limits<int32_t>::min());
  }
  {
    const auto input = makeFlatVector<int8_t>({16, -128, -1});
    assertAggregate<int8_t>("product", input, {add(0), add(0)}, 0);
    assertAggregate<int8_t>("product", input, {add(1), retract(2)}, -128);
  }
  {
    const auto input = makeFlatVector<double>({1.5, 0.0});
    assertAggregate<double>("product", input, {add(0), add(0)}, 2.25);
    assertAggregate<double>(
        "product",
        input,
        {add(0), retract(1)},
        std::numeric_limits<double>::infinity());
  }
  {
    // 1.50 * 2.25 = 3.375 rounds to 3.38, 0.01 / 0.08 = 0.125 to 0.13.
    const auto input =
        makeFlatVector<int64_t>({150, 225, 1, 8, 300, 0}, DECIMAL(10, 2));
    assertAggregate<int64_t>("product", input, {add(0), add(1)}, 338);
    assertAggregate<int64_t>("product", input, {add(2), retract(3)}, 13);
    assertAggregate<int64_t>("product", input, {add(4), retract(0)}, 200);
    VELOX_ASSERT_USER_THROW(
        aggregate("product", input, {add(2), retract(4)}),
        "Paimon aggregate function product cannot retract a decimal whose "
        "quotient does not terminate: c");
    VELOX_ASSERT_USER_THROW(
        aggregate("product", input, {add(2), retract(5)}),
        "Paimon aggregate function product cannot retract a zero: c");
  }
  {
    const auto input = makeFlatVector<int64_t>({99, 9}, DECIMAL(3, 0));
    assertAggregate<int64_t>("product", input, {add(1), add(0)}, 891);
    assertAggregate<int64_t>("product", input, {add(0), add(0)}, std::nullopt);
  }
  {
    const int128_t large = DecimalUtil::kPowersOfTen[30];
    const auto input =
        makeFlatVector<int128_t>({large, large, 25}, DECIMAL(38, 10));
    assertAggregate<int128_t>("product", input, {add(0), add(1)}, std::nullopt);
    // 10^20 * 0.0000000025 = 250'000'000'000.
    assertAggregate<int128_t>(
        "product", input, {add(0), add(2)}, DecimalUtil::kPowersOfTen[20] * 25);
  }
}

TEST_F(PaimonFieldAggregatorTest, booleans) {
  const auto input =
      makeNullableFlatVector<bool>({true, std::nullopt, false, true});
  assertAggregate<bool>("bool_and", input, {add(0), add(1), add(2)}, false);
  assertAggregate<bool>("bool_and", input, {add(1), add(0), add(3)}, true);
  assertAggregate<bool>("bool_or", input, {add(2), add(1), add(0)}, true);
  assertAggregate<bool>("bool_or", input, {add(1), add(2)}, false);
  assertAggregate<bool>("bool_or", input, {add(1)}, std::nullopt);
}

// Paimon Java compares values as Java's compare methods do: -0.0 is less than
// 0.0, NaN is larger than any other value, and strings compare by their bytes.
TEST_F(PaimonFieldAggregatorTest, minMaxOrder) {
  constexpr auto kNaN = std::numeric_limits<double>::quiet_NaN();
  constexpr auto kInfinity = std::numeric_limits<double>::infinity();
  const auto doubles = makeFlatVector<double>({0.0, -0.0, kNaN, kInfinity});
  {
    const auto result = aggregate("min", doubles, {add(0), add(1)});
    EXPECT_TRUE(std::signbit(result->as<SimpleVector<double>>()->valueAt(0)));
  }
  {
    const auto result = aggregate("max", doubles, {add(1), add(0)});
    EXPECT_FALSE(std::signbit(result->as<SimpleVector<double>>()->valueAt(0)));
  }
  {
    const auto result = aggregate("max", doubles, {add(3), add(2)});
    EXPECT_TRUE(std::isnan(result->as<SimpleVector<double>>()->valueAt(0)));
  }
  assertAggregate<double>("min", doubles, {add(2), add(3)}, kInfinity);

  const auto strings = makeFlatVector<std::string>({"b", "\xc3\xa9", "a", ""});
  assertAggregate<StringView>(
      "min", strings, {add(0), add(1), add(2)}, StringView("a"));
  assertAggregate<StringView>(
      "max", strings, {add(0), add(1), add(2)}, StringView("\xc3\xa9"));
  assertAggregate<StringView>("min", strings, {add(0), add(3)}, StringView(""));

  const auto decimals = makeFlatVector<int128_t>(
      {-DecimalUtil::kPowersOfTen[30], 5}, DECIMAL(38, 2));
  assertAggregate<int128_t>(
      "min", decimals, {add(1), add(0)}, -DecimalUtil::kPowersOfTen[30]);

  const auto timestamps =
      makeFlatVector<Timestamp>({Timestamp(10, 5), Timestamp(10, 7)});
  assertAggregate<Timestamp>(
      "max", timestamps, {add(0), add(1)}, Timestamp(10, 7));
}

// 'fields.<column>.ignore-retract' makes a function keep its accumulator on a
// DELETE or UPDATE_BEFORE row, as Paimon Java's FieldIgnoreRetractAgg does.
TEST_F(PaimonFieldAggregatorTest, ignoreRetract) {
  const auto input = makeFlatVector<int64_t>({1, 5});
  test::assertEqualVectors(
      makeFlatVector<int64_t>({1}),
      aggregate("sum", input, {add(0), retract(1)}, /*ignoreRetract=*/true));
  test::assertEqualVectors(
      makeFlatVector<int64_t>({1}),
      aggregate(
          "first_value", input, {add(0), retract(1)}, /*ignoreRetract=*/true));
  for (const auto* function :
       {"first_value", "first_non_null_value", "min", "max"}) {
    SCOPED_TRACE(function);
    VELOX_ASSERT_USER_THROW(
        aggregate(function, input, {add(0), retract(1)}),
        fmt::format(
            "Paimon aggregate function {} cannot retract a row unless "
            "'fields.c.ignore-retract' is true",
            function));
  }
  const auto booleans = makeFlatVector<bool>({true, false});
  for (const auto* function : {"bool_and", "bool_or"}) {
    SCOPED_TRACE(function);
    VELOX_ASSERT_USER_THROW(
        aggregate(function, booleans, {retract(1)}),
        fmt::format(
            "Paimon aggregate function {} cannot retract a row unless "
            "'fields.c.ignore-retract' is true",
            function));
  }
}

TEST_F(PaimonFieldAggregatorTest, unsupported) {
  VELOX_ASSERT_USER_THROW(
      PaimonFieldAggregator::create("Sum", "c", BIGINT(), false),
      "Unknown Paimon aggregate function: Sum");
  VELOX_ASSERT_THROW(
      PaimonFieldAggregator::create("collect", "c", ARRAY(BIGINT()), false),
      "Paimon merge-on-read does not support the aggregate function: collect");
  struct TestParam {
    std::string function;
    TypePtr type;

    std::string debugString() const {
      return fmt::format("function {}, type {}", function, type->toString());
    }
  };
  const std::vector<TestParam> testSettings{
      {"sum", VARCHAR()},
      {"sum", DATE()},
      {"sum", BOOLEAN()},
      {"product", TIMESTAMP()},
      {"bool_and", BIGINT()},
      {"bool_or", VARCHAR()},
      {"min", BOOLEAN()},
      {"max", ARRAY(BIGINT())},
      {"min", ROW("a", BIGINT())},
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    VELOX_ASSERT_USER_THROW(
        PaimonFieldAggregator::create(
            testData.function, "c", testData.type, false),
        fmt::format(
            "Paimon aggregate function {} does not support the type of "
            "column c: {}",
            testData.function,
            testData.type->toString()));
  }
}

} // namespace
} // namespace facebook::velox::connector::paimon
