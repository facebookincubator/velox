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

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class RoundTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  std::optional<T> round(
      const std::optional<T>& value,
      const std::optional<int32_t>& scale) {
    return evaluateOnce<T>("round(c0, c1)", value, scale);
  }

  template <typename T>
  std::optional<T> round(const std::optional<T>& value) {
    return evaluateOnce<T>("round(c0)", value);
  }

  void testRoundEncodings(
      const VectorPtr& input,
      int32_t scale,
      const VectorPtr& expected) {
    auto scales = makeFlatVector<int32_t>(
        input->size(), [scale](vector_size_t) { return scale; });
    auto data = makeRowVector({input, scales});
    for (const auto& expression :
         {std::string{"round(c0, c1)"},
          fmt::format("round(c0, cast({} as integer))", scale)}) {
      testEncodings(
          makeTypedExpr(expression, data->rowType()),
          data->children(),
          expected);
    }
  }
};

TEST_F(RoundTest, finiteInputsDoNotProduceNonFiniteResults) {
  EXPECT_EQ(round<double>(-3.5, 308), -3.5);

  const auto doubleMinimum = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(round<double>(doubleMinimum, 309), 0.0);
  EXPECT_EQ(
      round<double>(doubleMinimum, std::numeric_limits<int32_t>::max()),
      doubleMinimum);
  EXPECT_EQ(round<double>(1.0, std::numeric_limits<int32_t>::min()), 0.0);

  const auto floatMinimum = std::numeric_limits<float>::denorm_min();
  EXPECT_EQ(round<float>(floatMinimum, 309), floatMinimum);
  EXPECT_EQ(round<float>(-3.5F, 308), -3.5F);

  const auto floatMaximum = std::numeric_limits<float>::max();
  EXPECT_EQ(round<float>(floatMaximum, -35), floatMaximum);
  EXPECT_EQ(round<float>(-floatMaximum, -35), -floatMaximum);
}

TEST_F(RoundTest, overflowingScalesPreserveOrdinaryValues) {
  const auto doubles = makeNullableFlatVector<double>(
      {1.5,
       -1.5,
       3.5,
       -3.5,
       std::numeric_limits<double>::max(),
       -std::numeric_limits<double>::max(),
       1e-291,
       std::nextafter(1e-291, 0.0),
       std::nullopt});
  const auto floats = makeNullableFlatVector<float>(
      {1.5F,
       -1.5F,
       std::numeric_limits<float>::max(),
       std::numeric_limits<float>::min(),
       std::numeric_limits<float>::denorm_min(),
       -std::numeric_limits<float>::denorm_min(),
       std::nullopt});
  for (const auto scale :
       {309, 310, 323, 324, 340, 341, std::numeric_limits<int32_t>::max()}) {
    SCOPED_TRACE(scale);
    testRoundEncodings(doubles, scale, doubles);
    testRoundEncodings(floats, scale, floats);
  }
}

TEST_F(RoundTest, overflowingScaleRoundsSubnormalValues) {
  // Match Spark's decimal HALF_UP results, not an overflowing binary round
  // trip.
  const auto minimum = std::numeric_limits<double>::denorm_min();
  testRoundEncodings(
      makeNullableFlatVector<double>(
          {4e-310,
           5e-310,
           6e-310,
           1.5e-309,
           -4e-310,
           -5e-310,
           -6e-310,
           -1.5e-309,
           std::nextafter(5e-310, 0.0),
           std::nextafter(5e-310, 1.0),
           minimum,
           -minimum,
           std::numeric_limits<double>::min(),
           std::nullopt}),
      309,
      makeNullableFlatVector<double>(
          {0.0,
           1e-309,
           1e-309,
           2e-309,
           0.0,
           -1e-309,
           -1e-309,
           -2e-309,
           0.0,
           1e-309,
           0.0,
           0.0,
           2.2e-308,
           std::nullopt}));
}

TEST_F(RoundTest, overflowingScaleBoundaries) {
  const auto minimum = std::numeric_limits<double>::denorm_min();
  testRoundEncodings(
      makeNullableFlatVector<double>(
          {minimum, -minimum, 5e-323, -5e-323, 6e-323, -6e-323}),
      322,
      makeNullableFlatVector<double>({0.0, 0.0, 0.0, 0.0, 1e-322, -1e-322}));
  testRoundEncodings(
      makeNullableFlatVector<double>(
          {minimum, -minimum, 1e-323, -1e-323, 1.5e-323, -1.5e-323}),
      323,
      makeNullableFlatVector<double>(
          {0.0, 0.0, 1e-323, -1e-323, 2e-323, -2e-323}));
  for (const auto scale :
       {324, 325, 340, 341, std::numeric_limits<int32_t>::max()}) {
    const auto inputs = makeNullableFlatVector<double>(
        {minimum, -minimum, std::numeric_limits<double>::min(), std::nullopt});
    testRoundEncodings(inputs, scale, inputs);
  }

  // Preserve the shortest decimal at an asymmetric power-of-two boundary.
  EXPECT_EQ(round<double>(0x1p-1017, 321), 7.12023634722305e-307);
  EXPECT_EQ(round<double>(0x1p-1017, 322), 0x1p-1017);
}

TEST_F(RoundTest, ordinaryValues) {
  EXPECT_EQ(round<double>(2.5), 3.0);
  EXPECT_EQ(round<double>(-2.5), -3.0);
  EXPECT_EQ(round<float>(1.5F), 2.0F);
  EXPECT_EQ(round<double>(1.234, 2), 1.23);
  EXPECT_EQ(round<double>(125.0, -1), 130.0);
  EXPECT_EQ(round<double>(17'592'186'044'416.25, 1), 17'592'186'044'416.3);
}

TEST_F(RoundTest, canonicalZero) {
  const auto unaryResult = round<double>(-0.0);
  ASSERT_TRUE(unaryResult.has_value());
  EXPECT_FALSE(std::signbit(unaryResult.value()));

  for (const auto scale : {-40, -1, 0, 1, 308, 309}) {
    const auto result = round<double>(-0.0, scale);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 0.0);
    EXPECT_FALSE(std::signbit(result.value()));
  }
  for (const auto input :
       {-0.4, -4e-310, -std::numeric_limits<double>::denorm_min()}) {
    const auto scale = input == -0.4 ? 0 : 309;
    const auto result = round<double>(input, scale);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 0.0);
    EXPECT_FALSE(std::signbit(result.value()));
  }
}

TEST_F(RoundTest, nonFiniteInputs) {
  const auto infinity = std::numeric_limits<double>::infinity();
  for (const auto scale : {2, 309, std::numeric_limits<int32_t>::max()}) {
    EXPECT_EQ(round<double>(infinity, scale), infinity);
    EXPECT_EQ(round<double>(-infinity, scale), -infinity);
    EXPECT_TRUE(
        std::isnan(
            round<double>(std::numeric_limits<double>::quiet_NaN(), scale)
                .value()));
  }
  EXPECT_EQ(round<double>(1.5, std::nullopt), std::nullopt);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
