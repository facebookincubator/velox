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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

#include <velox/vector/SimpleVector.h>

namespace facebook::velox::functions::sparksql::test {
namespace {
static constexpr double kInf = std::numeric_limits<double>::infinity();
static constexpr float kInfF = std::numeric_limits<float>::infinity();
static constexpr auto kNaN = std::numeric_limits<double>::quiet_NaN();
class CompareTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  std::optional<bool> equaltonullsafe(std::optional<T> a, std::optional<T> b) {
    return evaluateOnce<bool>("equalnullsafe(c0, c1)", a, b);
  }

  template <typename T>
  std::optional<bool> equalto(std::optional<T> a, std::optional<T> b) {
    return evaluateOnce<bool>("equalto(c0, c1)", a, b);
  }

  template <typename T>
  std::optional<bool> lessthan(std::optional<T> a, std::optional<T> b) {
    return evaluateOnce<bool>("lessthan(c0, c1)", a, b);
  }

  template <typename T>
  std::optional<bool> lessthanorequal(std::optional<T> a, std::optional<T> b) {
    return evaluateOnce<bool>("lessthanorequal(c0, c1)", a, b);
  }

  template <typename T>
  std::optional<bool> greaterthan(std::optional<T> a, std::optional<T> b) {
    return evaluateOnce<bool>("greaterthan(c0, c1)", a, b);
  }

  template <typename T>
  std::optional<bool> greaterthanorequal(
      std::optional<T> a,
      std::optional<T> b) {
    return evaluateOnce<bool>("greaterthanorequal(c0, c1)", a, b);
  }
};

TEST_F(CompareTest, equaltonullsafe) {
  EXPECT_EQ(equaltonullsafe<int64_t>(1, 1), true);
  EXPECT_EQ(equaltonullsafe<int32_t>(1, 2), false);
  EXPECT_EQ(equaltonullsafe<float>(std::nullopt, std::nullopt), true);
  EXPECT_EQ(equaltonullsafe<std::string>(std::nullopt, "abcs"), false);
  EXPECT_EQ(equaltonullsafe<std::string>(std::nullopt, std::nullopt), true);
  EXPECT_EQ(equaltonullsafe<double>(1, std::nullopt), false);
  EXPECT_EQ(equaltonullsafe<double>(kNaN, std::nullopt), false);
  EXPECT_EQ(equaltonullsafe<double>(kNaN, 1), false);
  EXPECT_EQ(equaltonullsafe<double>(kNaN, kNaN), true);
}

TEST_F(CompareTest, equalto) {
  EXPECT_EQ(equalto<int64_t>(1, 1), true);
  EXPECT_EQ(equalto<int32_t>(1, 2), false);
  EXPECT_EQ(equalto<float>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(equalto<std::string>(std::nullopt, "abcs"), std::nullopt);
  EXPECT_EQ(equalto<std::string>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(equalto<double>(1, std::nullopt), std::nullopt);
  EXPECT_EQ(equalto<double>(kNaN, std::nullopt), std::nullopt);
  EXPECT_EQ(equalto<double>(kNaN, 1), false);
  EXPECT_EQ(equalto<double>(0, kNaN), false);
  EXPECT_EQ(equalto<double>(kNaN, kNaN), true);
  EXPECT_EQ(equalto<double>(kInf, kInf), true);
  EXPECT_EQ(equalto<float>(kInfF, kInfF), true);
  EXPECT_EQ(equalto<double>(kInf, 2.0), false);
  EXPECT_EQ(equalto<double>(-kInf, 2.0), false);
  EXPECT_EQ(equalto<float>(kInfF, 1.0), false);
  EXPECT_EQ(equalto<float>(-kInfF, 1.0), false);
  EXPECT_EQ(equalto<float>(kInfF, -kInfF), false);
  EXPECT_EQ(equalto<double>(kInf, kNaN), false);
}

TEST_F(CompareTest, testdecimal) {
  auto runAndCompare = [&](const std::string& exprStr,
                           std::vector<VectorPtr>& input,
                           VectorPtr expectedResult) {
    auto actual = evaluate<SimpleVector<bool>>(exprStr, makeRowVector(input));
    facebook::velox::test::assertEqualVectors(actual, expectedResult);
  };
  std::vector<VectorPtr> inputs = {
      makeNullableShortDecimalFlatVector(
          {1, std::nullopt, 3, -2, std::nullopt, 4}, DECIMAL(10, 5)),
      makeNullableShortDecimalFlatVector(
          {0, 2, 3, -3, std::nullopt, 5}, DECIMAL(10, 5))};
  auto expected = makeNullableFlatVector<bool>(
      {true, std::nullopt, false, true, std::nullopt, false});
  runAndCompare(fmt::format("{}(c0, c1)", "greaterthan"), inputs, expected);
  std::vector<VectorPtr> longDecimalsInputs = {
      makeNullableLongDecimalFlatVector(
          {DecimalUtil::kLongDecimalMax,
           std::nullopt,
           3,
           DecimalUtil::kLongDecimalMin + 1,
           std::nullopt,
           4},
          DECIMAL(38, 5)),
      makeNullableLongDecimalFlatVector(
          {DecimalUtil::kLongDecimalMax - 1,
           2,
           3,
           DecimalUtil::kLongDecimalMin,
           std::nullopt,
           5},
          DECIMAL(38, 5))};
  auto expectedGteLte = makeNullableFlatVector<bool>(
      {true, std::nullopt, true, true, std::nullopt, false});
  runAndCompare(
      fmt::format("{}(c1, c0)", "lessthanorequal"),
      longDecimalsInputs,
      expectedGteLte);

  // Test with different data types.
  std::vector<VectorPtr> invalidInputs = {
      makeNullableShortDecimalFlatVector({1}, DECIMAL(10, 5)),
      makeNullableShortDecimalFlatVector({1}, DECIMAL(10, 4))};
  auto invalidResult = makeNullableFlatVector<bool>({true});
  VELOX_ASSERT_THROW(
      runAndCompare(
          fmt::format("{}(c1, c0)", "equalto"), invalidInputs, invalidResult),
      "Scalar function signature is not supported: "
      "equalto(DECIMAL(10,4), DECIMAL(10,5))");
}

TEST_F(CompareTest, testdictionary) {
  // Identity mapping, however this will result in non-simd path.
  auto makeDictionary = [&](const VectorPtr& base) {
    auto indices = makeIndices(base->size(), [](auto row) { return row; });
    return wrapInDictionary(indices, base->size(), base);
  };

  auto makeConstantDic = [&](const VectorPtr& base) {
    return BaseVector::wrapInConstant(base->size(), 0, base);
  };
  // lhs: 0, null, 2, null, 4

  auto lhs = makeFlatVector<int16_t>(
      5, [](auto row) { return row; }, nullEvery(2, 1));
  auto rhs = makeFlatVector<int16_t>({1, 0, 3, 0, 5});
  auto lhsVector = makeDictionary(lhs);
  auto rhsVector = makeDictionary(rhs);

  auto rowVector = makeRowVector({lhsVector, rhsVector});
  auto result = evaluate<SimpleVector<bool>>(
      fmt::format("{}(c0, c1)", "greaterthan"), rowVector);
  // result : false, null, false, null, false
  facebook::velox::test::assertEqualVectors(
      result,
      makeFlatVector<bool>(
          5, [](auto row) { return false; }, nullEvery(2, 1)));
  auto constVector = makeConstant(100, 5);
  auto testConstVector =
      makeRowVector({makeDictionary(lhs), makeConstantDic(constVector)});
  // lhs: 0, null, 2, null, 4
  // rhs: const 100
  // lessthanorequal result : true, null, true, null, true
  auto constResult = evaluate<SimpleVector<bool>>(
      fmt::format("{}(c0, c1)", "lessthanorequal"), testConstVector);
  facebook::velox::test::assertEqualVectors(
      constResult,
      makeFlatVector<bool>(
          5, [](auto row) { return true; }, nullEvery(2, 1)));
  // lhs: const 100
  // rhs: 0, null, 2, null, 4
  // greaterthanorequal result : true, null, true, null, true
  auto testConstVector1 =
      makeRowVector({makeConstantDic(constVector), makeDictionary(lhs)});
  auto constResult1 = evaluate<SimpleVector<bool>>(
      fmt::format("{}(c0, c1)", "greaterthanorequal"), testConstVector1);
  facebook::velox::test::assertEqualVectors(
      constResult1,
      makeFlatVector<bool>(
          5, [](auto row) { return true; }, nullEvery(2, 1)));
}

TEST_F(CompareTest, testflat) {
  auto vector0 = makeFlatVector<int32_t>({0, 1, 2, 3});
  auto vector1 = makeFlatVector<int32_t>(
      4, [](auto row) { return row + 1; }, nullEvery(2));

  auto expectedResult = makeFlatVector<bool>(
      4, [](auto row) { return true; }, nullEvery(2));
  auto actualResult = evaluate<SimpleVector<bool>>(
      "lessthan(c0, c1)", makeRowVector({vector0, vector1}));
  facebook::velox::test::assertEqualVectors(expectedResult, actualResult);
}

TEST_F(CompareTest, lessthan) {
  EXPECT_EQ(lessthan<int64_t>(1, 1), false);
  EXPECT_EQ(lessthan<int32_t>(1, 2), true);
  EXPECT_EQ(lessthan<float>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(lessthan<std::string>(std::nullopt, "abcs"), std::nullopt);
  EXPECT_EQ(lessthan<std::string>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(lessthan<double>(1, std::nullopt), std::nullopt);
  EXPECT_EQ(lessthan<double>(kNaN, std::nullopt), std::nullopt);
  EXPECT_EQ(lessthan<double>(kNaN, 1), false);
  EXPECT_EQ(lessthan<double>(0, kNaN), true);
  EXPECT_EQ(lessthan<double>(kNaN, kNaN), false);
  EXPECT_EQ(lessthan<double>(kInf, kInf), false);
  EXPECT_EQ(lessthan<float>(kInfF, kInfF), false);
  EXPECT_EQ(lessthan<double>(kInf, 2.0), false);
  EXPECT_EQ(lessthan<double>(-kInf, 2.0), true);
  EXPECT_EQ(lessthan<float>(kInfF, 1.0), false);
  EXPECT_EQ(lessthan<float>(-kInfF, 1.0), true);
  EXPECT_EQ(lessthan<float>(kInfF, -kInfF), false);
  EXPECT_EQ(lessthan<double>(kInf, kNaN), true);
}

TEST_F(CompareTest, lessthanorequal) {
  EXPECT_EQ(lessthanorequal<int64_t>(1, 1), true);
  EXPECT_EQ(lessthanorequal<int32_t>(1, 2), true);
  EXPECT_EQ(lessthanorequal<float>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(lessthanorequal<std::string>(std::nullopt, "abcs"), std::nullopt);
  EXPECT_EQ(
      lessthanorequal<std::string>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(lessthanorequal<double>(1, std::nullopt), std::nullopt);
  EXPECT_EQ(lessthanorequal<double>(kNaN, std::nullopt), std::nullopt);
  EXPECT_EQ(lessthanorequal<double>(kNaN, 1), false);
  EXPECT_EQ(lessthanorequal<double>(0, kNaN), true);
  EXPECT_EQ(lessthanorequal<double>(kNaN, kNaN), true);
  EXPECT_EQ(lessthanorequal<double>(kInf, kInf), true);
  EXPECT_EQ(lessthanorequal<float>(kInfF, kInfF), true);
  EXPECT_EQ(lessthanorequal<double>(kInf, 2.0), false);
  EXPECT_EQ(lessthanorequal<double>(-kInf, 2.0), true);
  EXPECT_EQ(lessthanorequal<float>(kInfF, 1.0), false);
  EXPECT_EQ(lessthanorequal<float>(-kInfF, 1.0), true);
  EXPECT_EQ(lessthanorequal<float>(kInfF, -kInfF), false);
  EXPECT_EQ(lessthanorequal<double>(kInf, kNaN), true);
}

TEST_F(CompareTest, greaterthan) {
  EXPECT_EQ(greaterthan<int64_t>(1, 1), false);
  EXPECT_EQ(greaterthan<int32_t>(1, 2), false);
  EXPECT_EQ(greaterthan<float>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(greaterthan<std::string>(std::nullopt, "abcs"), std::nullopt);
  EXPECT_EQ(greaterthan<std::string>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(greaterthan<double>(1, std::nullopt), std::nullopt);
  EXPECT_EQ(greaterthan<double>(kNaN, std::nullopt), std::nullopt);
  EXPECT_EQ(greaterthan<double>(kNaN, 1), true);
  EXPECT_EQ(greaterthan<double>(0, kNaN), false);
  EXPECT_EQ(greaterthan<double>(kNaN, kNaN), false);
  EXPECT_EQ(greaterthan<double>(kInf, kInf), false);
  EXPECT_EQ(greaterthan<float>(kInfF, kInfF), false);
  EXPECT_EQ(greaterthan<double>(kInf, 2.0), true);
  EXPECT_EQ(greaterthan<double>(-kInf, 2.0), false);
  EXPECT_EQ(greaterthan<float>(kInfF, 1.0), true);
  EXPECT_EQ(greaterthan<float>(-kInfF, 1.0), false);
  EXPECT_EQ(greaterthan<float>(kInfF, -kInfF), true);
  EXPECT_EQ(greaterthan<float>(kInf, kNaN), false);
}

TEST_F(CompareTest, greaterthanorequal) {
  EXPECT_EQ(greaterthanorequal<int64_t>(1, 1), true);
  EXPECT_EQ(greaterthanorequal<int32_t>(1, 2), false);
  EXPECT_EQ(
      greaterthanorequal<float>(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(
      greaterthanorequal<std::string>(std::nullopt, "abcs"), std::nullopt);
  EXPECT_EQ(
      greaterthanorequal<std::string>(std::nullopt, std::nullopt),
      std::nullopt);
  EXPECT_EQ(greaterthanorequal<double>(1, std::nullopt), std::nullopt);
  EXPECT_EQ(greaterthanorequal<double>(kNaN, std::nullopt), std::nullopt);
  EXPECT_EQ(greaterthanorequal<double>(kNaN, 1), true);
  EXPECT_EQ(greaterthanorequal<double>(0, kNaN), false);
  EXPECT_EQ(greaterthanorequal<double>(kNaN, kNaN), true);
  EXPECT_EQ(greaterthanorequal<double>(kInf, kInf), true);
  EXPECT_EQ(greaterthanorequal<float>(kInfF, kInfF), true);
  EXPECT_EQ(greaterthanorequal<double>(kInf, 2.0), true);
  EXPECT_EQ(greaterthanorequal<double>(-kInf, 2.0), false);
  EXPECT_EQ(greaterthanorequal<float>(kInfF, 1.0), true);
  EXPECT_EQ(greaterthanorequal<float>(-kInfF, 1.0), false);
  EXPECT_EQ(greaterthanorequal<float>(kInfF, -kInfF), true);
  EXPECT_EQ(greaterthanorequal<float>(kInf, kNaN), false);
}
} // namespace
}; // namespace facebook::velox::functions::sparksql::test
