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
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <fmt/format.h>
#include <gmock/gmock.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

using velox::test::assertEqualVectors;

struct RoundTestParam {
  bool ansiEnabled;
  bool halfEven;
};

class RoundTest : public SparkFunctionBaseTest,
                  public testing::WithParamInterface<RoundTestParam> {
 protected:
  void SetUp() override {
    SparkFunctionBaseTest::SetUp();
    setAnsi(GetParam().ansiEnabled);
  }

  void setAnsi(bool enabled) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled),
          enabled ? "true" : "false"}});
  }

  std::string name() const {
    return GetParam().halfEven ? "bround" : "round";
  }

  std::string expression(std::string_view arguments) const {
    return fmt::format("{}({})", name(), arguments);
  }

  template <typename T>
  std::optional<T> roundValue(
      std::optional<T> value,
      std::optional<int32_t> scale) {
    return evaluateOnce<T>(
        expression(
            fmt::format(
                "c0, cast({} as integer)",
                scale ? std::to_string(*scale) : "null")),
        value);
  }

  // Group by scale so both functions exercise vectors with constant scales.
  template <typename T>
  void checkRows(const std::vector<std::tuple<T, int32_t, T, T>>& rows) {
    if (!GetParam().halfEven) {
      assertEqualVectors(
          makeFlatVector<T, 2>(rows),
          evaluate(
              expression("c0, c1"),
              makeRowVector(
                  {makeFlatVector<T, 0>(rows),
                   makeFlatVector<int32_t, 1>(rows)})));
    }
    std::map<int32_t, std::vector<std::tuple<T, T>>> groups;
    for (const auto& [value, scale, halfUp, halfEven] : rows) {
      groups[scale].emplace_back(
          value, GetParam().halfEven ? halfEven : halfUp);
    }
    for (const auto& [scale, values] : groups) {
      SCOPED_TRACE(fmt::format("{} at scale {}", name(), scale));
      assertEqualVectors(
          makeFlatVector<T, 1>(values),
          evaluate(
              expression(fmt::format("c0, cast({} as integer)", scale)),
              makeRowVector({makeFlatVector<T, 0>(values)})));
    }
  }

  template <typename T>
  void checkCommon() {
    checkRows<T>({
        {25, -1, 30, 20},
        {35, -1, 40, 40},
        {-25, -1, -30, -20},
        {-35, -1, -40, -40},
        {24, -1, 20, 20},
        {26, -1, 30, 30},
        {-24, -1, -20, -20},
        {-26, -1, -30, -30},
        {0, -1, 0, 0},
        {25, -2, 0, 0},
        {-25, 1, -25, -25},
        {25, -100, 0, 0},
    });

    auto values = makeNullableFlatVector<T>(
        {std::numeric_limits<T>::min(),
         std::numeric_limits<T>::max(),
         -25,
         0,
         25,
         std::nullopt});
    auto input = makeRowVector({values});
    for (const auto* arguments :
         {"c0", "c0, cast(0 as integer)", "c0, cast(2147483647 as integer)"}) {
      assertEqualVectors(values, evaluate(expression(arguments), input));
    }
    assertEqualVectors(
        makeNullableFlatVector<T>(
            {std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt}),
        evaluate(expression("c0, cast(null as integer)"), input));
    if (!GetParam().halfEven) {
      assertEqualVectors(
          makeNullableFlatVector<T>({30, 0, std::nullopt, std::nullopt, 25}),
          evaluate(
              expression("c0, c1"),
              makeRowVector(
                  {makeNullableFlatVector<T>({25, 25, 25, std::nullopt, 25}),
                   makeNullableFlatVector<int32_t>(
                       {-1, -2, std::nullopt, -1, 0})})));
    }
  }

  template <typename T>
  void checkMinimumScale() {
    auto values = makeNullableFlatVector<T>(
        {std::numeric_limits<T>::min(),
         std::numeric_limits<T>::max(),
         -25,
         0,
         25,
         std::nullopt});
    const auto expected =
        makeNullableFlatVector<T>({0, 0, 0, 0, 0, std::nullopt});
    assertEqualVectors(
        expected,
        evaluate(
            expression("c0, cast(-2147483648 as integer)"),
            makeRowVector({values})));
    if (!GetParam().halfEven) {
      assertEqualVectors(
          expected,
          evaluate(
              expression("c0, c1"),
              makeRowVector({values, makeFlatVector<int32_t>(6, [](auto) {
                               return std::numeric_limits<int32_t>::min();
                             })})));
    }
  }

  template <typename T>
  void checkOverflow(T value, int32_t scale, T wrapped) {
    if (GetParam().ansiEnabled) {
      VELOX_ASSERT_USER_THROW(
          roundValue<T>(value, scale),
          fmt::format(
              "Arithmetic overflow: {}({}, {})",
              name(),
              static_cast<int64_t>(value),
              scale));
    } else {
      EXPECT_EQ(roundValue<T>(value, scale), wrapped);
    }
  }

  template <typename T>
  void checkIntegralEncodings() {
    const auto values = makeNullableFlatVector<T>(
        {14,  15,  16,  24,  25,  26,  34,  35,  36, -14,
         -15, -16, -24, -25, -26, -34, -35, -36, 0,  std::nullopt});
    const T evenTie = GetParam().halfEven ? 20 : 30;
    const auto expected = makeNullableFlatVector<T>(
        {10,  20,  20,  20,  evenTie, 30,          30,
         40,  40,  -10, -20, -20,     -20,         static_cast<T>(-evenTie),
         -30, -30, -40, -40, 0,       std::nullopt});
    testEncodings(
        makeTypedExpr(
            expression("c0, cast(-1 as integer)"),
            makeRowVector({values})->rowType()),
        {values},
        expected);
  }

  template <typename T>
  void checkExtremeNegativeScales() {
    constexpr int32_t firstZeroScale = -(std::numeric_limits<T>::digits10 + 2);
    for (const int32_t scale :
         {firstZeroScale, std::numeric_limits<int32_t>::min()}) {
      for (const T value :
           {T{1},
            T{-1},
            T{0},
            std::numeric_limits<T>::max(),
            std::numeric_limits<T>::min()}) {
        SCOPED_TRACE(fmt::format("value {}, scale {}", value, scale));
        EXPECT_EQ(roundValue<T>(value, scale), T{0});
      }
    }
    EXPECT_EQ(
        (evaluateOnce<T, T>(
            fmt::format(
                "try({})",
                expression(
                    fmt::format("c0, cast({} as integer)", firstZeroScale))),
            T{1})),
        T{0});
  }
};

TEST_P(RoundTest, integralTiesAndIdentity) {
  checkCommon<int8_t>();
  checkCommon<int16_t>();
  checkCommon<int32_t>();
  checkCommon<int64_t>();
}

TEST_P(RoundTest, minimumScale) {
  checkMinimumScale<int8_t>();
  checkMinimumScale<int16_t>();
  checkMinimumScale<int32_t>();
  checkMinimumScale<int64_t>();
  checkExtremeNegativeScales<int8_t>();
  checkExtremeNegativeScales<int16_t>();
  checkExtremeNegativeScales<int32_t>();
  checkExtremeNegativeScales<int64_t>();
}

TEST_P(RoundTest, integralOverflow) {
  checkOverflow<int8_t>(127, -1, -126);
  checkOverflow<int8_t>(-128, -1, 126);
  checkOverflow<int16_t>(32'767, -1, -32'766);
  checkOverflow<int16_t>(-32'768, -1, 32'766);
  checkOverflow<int32_t>(2'147'483'647, -1, -2'147'483'646);
  checkOverflow<int32_t>(-2'147'483'647 - 1, -1, 2'147'483'646);
  checkOverflow<int64_t>(
      9'223'372'036'854'775'807LL, -1, -9'223'372'036'854'775'806LL);
  checkOverflow<int64_t>(
      -9'223'372'036'854'775'807LL - 1, -1, 9'223'372'036'854'775'806LL);
  for (const int64_t value :
       {9'223'372'036'854'775'807LL, 5'000'000'000'000'000'001LL}) {
    checkOverflow<int64_t>(value, -19, -8'446'744'073'709'551'616LL);
    checkOverflow<int64_t>(-value, -19, 8'446'744'073'709'551'616LL);
  }
  checkOverflow<int64_t>(
      -9'223'372'036'854'775'807LL - 1, -19, 8'446'744'073'709'551'616LL);
  for (const int64_t value :
       {5'000'000'000'000'000'000LL, -5'000'000'000'000'000'000LL}) {
    if (GetParam().halfEven) {
      EXPECT_EQ(roundValue<int64_t>(value, -19), 0);
    } else {
      checkOverflow<int64_t>(
          value,
          -19,
          value > 0 ? -8'446'744'073'709'551'616LL
                    : 8'446'744'073'709'551'616LL);
    }
  }
}

TEST_P(RoundTest, coarseScales) {
  checkRows<int8_t>({
      {127, -2, 100, 100},
      {-128, -2, -100, -100},
      {127, -3, 0, 0},
      {-128, -3, 0, 0},
  });
  checkRows<int16_t>({
      {32'767, -4, 30'000, 30'000},
      {-32'768, -4, -30'000, -30'000},
      {32'767, -5, 0, 0},
      {-32'768, -5, 0, 0},
  });
  checkRows<int32_t>({
      {2'147'483'647, -9, 2'000'000'000, 2'000'000'000},
      {-2'147'483'647 - 1, -9, -2'000'000'000, -2'000'000'000},
      {2'147'483'647, -10, 0, 0},
      {-2'147'483'647 - 1, -10, 0, 0},
  });
  checkRows<int64_t>({
      {9'223'372'036'854'775'807LL,
       -18,
       9'000'000'000'000'000'000LL,
       9'000'000'000'000'000'000LL},
      {-9'223'372'036'854'775'807LL - 1,
       -18,
       -9'000'000'000'000'000'000LL,
       -9'000'000'000'000'000'000LL},
      {4'999'999'999'999'999'999LL, -19, 0, 0},
      {-4'999'999'999'999'999'999LL, -19, 0, 0},
      {9'223'372'036'854'775'807LL, -20, 0, 0},
      {-9'223'372'036'854'775'807LL - 1, -20, 0, 0},
      {9'007'199'254'740'995LL,
       -1,
       9'007'199'254'741'000LL,
       9'007'199'254'741'000LL},
      {-9'007'199'254'740'995LL,
       -1,
       -9'007'199'254'741'000LL,
       -9'007'199'254'741'000LL},
  });
}

TEST_P(RoundTest, integralEncodings) {
  checkIntegralEncodings<int8_t>();
  checkIntegralEncodings<int16_t>();
  checkIntegralEncodings<int32_t>();
  checkIntegralEncodings<int64_t>();
}

TEST_P(RoundTest, tryAndPartialSelection) {
  auto values =
      makeNullableFlatVector<int8_t>({25, 127, -35, -128, std::nullopt, 45});
  const int8_t roundedEvenTie = GetParam().halfEven ? 20 : 30;
  const int8_t roundedHigherTie = GetParam().halfEven ? 40 : 50;
  auto expected = GetParam().ansiEnabled
      ? makeNullableFlatVector<int8_t>(
            {roundedEvenTie,
             std::nullopt,
             -40,
             std::nullopt,
             std::nullopt,
             roundedHigherTie})
      : makeNullableFlatVector<int8_t>(
            {roundedEvenTie, -126, -40, 126, std::nullopt, roundedHigherTie});
  auto input = makeRowVector({values});
  assertEqualVectors(
      expected,
      evaluate(
          fmt::format("try({})", expression("c0, cast(-1 as integer)")),
          input));

  SelectivityVector selected(input->size(), false);
  selected.setValid(0, true);
  selected.setValid(2, true);
  selected.setValid(5, true);
  selected.updateBounds();
  assertEqualVectors(
      expected,
      evaluate(expression("c0, cast(-1 as integer)"), input, selected),
      selected);

  assertEqualVectors(
      GetParam().ansiEnabled ? makeNullableFlatVector<int8_t>(
                                   {std::nullopt, std::nullopt, std::nullopt})
                             : makeFlatVector<int8_t>({-126, -126, -126}),
      evaluate(
          fmt::format("try({})", expression("c0, c1")),
          makeRowVector(
              {makeConstant<int8_t>(127, 3), makeConstant<int32_t>(-1, 3)})));
}

TEST_P(RoundTest, scaleColumnContract) {
  const auto signatures = exec::simpleFunctions().getFunctionSignatures(name());
  size_t numBinarySignatures{0};
  for (const auto* signature : signatures) {
    if (signature->argumentTypes().size() == 2) {
      EXPECT_THAT(
          signature->constantArguments(),
          testing::ElementsAre(false, GetParam().halfEven));
      ++numBinarySignatures;
    }
  }
  EXPECT_EQ(numBinarySignatures, 6);

  auto input = makeRowVector(
      {makeFlatVector<double>({2.5, 3.5}), makeFlatVector<int32_t>({0, 1})});
  if (GetParam().halfEven) {
    VELOX_ASSERT_USER_THROW(
        evaluate(expression("c0, c1"), input),
        "The second argument of bround must be a constant expression.");
    VELOX_ASSERT_USER_THROW(
        evaluate(
            expression("c0, c1"),
            makeRowVector(
                {makeFlatVector<int64_t>({25, 35}),
                 makeFlatVector<int32_t>({-1, 0})})),
        "The second argument of bround must be a constant expression.");
  } else {
    assertEqualVectors(
        makeFlatVector<double>({3.0, 3.5}),
        evaluate(expression("c0, c1"), input));
  }
  assertEqualVectors(
      makeFlatVector<double>(
          GetParam().halfEven ? std::vector<double>{2.0, 4.0}
                              : std::vector<double>{3.0, 4.0}),
      evaluate(expression("c0, cast(subtract(1, 1) as integer)"), input));
}

TEST_P(RoundTest, scaleColumnEncodings) {
  auto values =
      makeNullableFlatVector<int8_t>({25, 127, 35, -128, -25, std::nullopt});
  auto scales = makeFlatVector<int32_t>({-1, -1, -2, -1, 0, -1});
  auto input = makeRowVector({values, scales});
  if (GetParam().halfEven) {
    VELOX_ASSERT_USER_THROW(
        evaluate(expression("c0, c1"), input),
        "The second argument of bround must be a constant expression.");
    return;
  }
  auto expected = GetParam().ansiEnabled
      ? makeNullableFlatVector<int8_t>(
            {30, std::nullopt, 0, std::nullopt, -25, std::nullopt})
      : makeNullableFlatVector<int8_t>({30, -126, 0, 126, -25, std::nullopt});
  assertEqualVectors(expected, evaluate("try(round(c0, c1))", input));

  auto indices = makeIndices({2, 1, 4, 3, 0, 5, 1});
  assertEqualVectors(
      GetParam().ansiEnabled ? makeNullableFlatVector<int8_t>(
                                   {0,
                                    std::nullopt,
                                    -25,
                                    std::nullopt,
                                    30,
                                    std::nullopt,
                                    std::nullopt})
                             : makeNullableFlatVector<int8_t>(
                                   {0, -126, -25, 126, 30, std::nullopt, -126}),
      evaluate(
          "try(round(c0, c1))",
          makeRowVector(
              {wrapInDictionary(indices, 7, values),
               wrapInDictionary(indices, 7, scales)})));
  assertEqualVectors(
      expected,
      evaluate(
          "try(round(c0, c1))",
          makeRowVector(
              {wrapInLazyDictionary(values), wrapInLazyDictionary(scales)})));

  SelectivityVector selected(6, false);
  selected.setValid(0, true);
  selected.setValid(2, true);
  selected.setValid(4, true);
  selected.updateBounds();
  assertEqualVectors(
      expected, evaluate("round(c0, c1)", input, selected), selected);
}

TEST_P(RoundTest, ansiCapturedAtInitialization) {
  auto input = makeRowVector(
      {makeFlatVector<int8_t>({127, 25, -35, -128}),
       makeFlatVector<int32_t>({-1, -1, -1, -1})});
  const auto source = GetParam().halfEven
      ? "try(bround(c0, cast(-1 as integer)))"
      : "try(round(c0, c1))";
  auto compiled = compileExpression(source, input->rowType());
  const int8_t roundedTie = GetParam().halfEven ? 20 : 30;
  auto expected = GetParam().ansiEnabled
      ? makeNullableFlatVector<int8_t>(
            {std::nullopt, roundedTie, -40, std::nullopt})
      : makeNullableFlatVector<int8_t>({-126, roundedTie, -40, 126});
  auto updatedExpected = GetParam().ansiEnabled
      ? makeNullableFlatVector<int8_t>({-126, roundedTie, -40, 126})
      : makeNullableFlatVector<int8_t>(
            {std::nullopt, roundedTie, -40, std::nullopt});
  assertEqualVectors(expected, evaluate(*compiled, input));
  setAnsi(!GetParam().ansiEnabled);
  assertEqualVectors(expected, evaluate(*compiled, input));
  assertEqualVectors(updatedExpected, evaluate(source, input));
}

TEST_P(RoundTest, floatingPointTies) {
  checkRows<double>({
      {0.5, 0, 1.0, 0.0},
      {1.5, 0, 2.0, 2.0},
      {2.5, 0, 3.0, 2.0},
      {3.5, 0, 4.0, 4.0},
      {-2.5, 0, -3.0, -2.0},
      {-3.5, 0, -4.0, -4.0},
      {1.25, 1, 1.3, 1.2},
      {1.75, 1, 1.8, 1.8},
      {1.245, 2, 1.25, 1.25},
      // Native binary rounding differs from decimal-string rounding.
      {0.575, 2, 0.57, 0.57},
      {-0.575, 2, -0.57, -0.57},
      {150.0, -2, 200.0, 200.0},
      {250.0, -2, 300.0, 200.0},
      {350.0, -2, 400.0, 400.0},
  });
  checkRows<float>({
      {1.25f, 1, 1.3f, 1.2f},
      {1.75f, 1, 1.8f, 1.8f},
      {15.0f, -1, 20.0f, 20.0f},
      {25.0f, -1, 30.0f, 20.0f},
  });
}

TEST_P(RoundTest, floatingPointNeighbors) {
  checkRows<double>({
      {std::nextafter(0.5, 0.0), 0, 0.0, 0.0},
      {std::nextafter(0.5, 1.0), 0, 1.0, 1.0},
      {std::nextafter(-0.5, -1.0), 0, -1.0, -1.0},
      {std::nextafter(-0.5, 0.0), 0, 0.0, 0.0},
  });
}

TEST_P(RoundTest, floatingPointEncodings) {
  const auto check = [this]<typename T>() {
    {
      const auto values = makeNullableFlatVector<T>(
          {T{1.25}, T{1.75}, T{-1.25}, T{-1.75}, std::nullopt});
      const T evenTie =
          GetParam().halfEven ? static_cast<T>(1.2) : static_cast<T>(1.3);
      testEncodings(
          makeTypedExpr(
              expression("c0, cast(1 as integer)"),
              makeRowVector({values})->rowType()),
          {values},
          makeNullableFlatVector<T>(
              {evenTie,
               static_cast<T>(1.8),
               -evenTie,
               static_cast<T>(-1.8),
               std::nullopt}));
    }
    {
      const auto values = makeNullableFlatVector<T>(
          {144, 145, 146, 154, 155, 156, -145, -155, std::nullopt});
      const T evenTie = GetParam().halfEven ? 140 : 150;
      testEncodings(
          makeTypedExpr(
              expression("c0, cast(-1 as integer)"),
              makeRowVector({values})->rowType()),
          {values},
          makeNullableFlatVector<T>(
              {140,
               evenTie,
               150,
               150,
               160,
               160,
               -evenTie,
               -160,
               std::nullopt}));
    }
  };
  check.template operator()<float>();
  check.template operator()<double>();
}

TEST_P(RoundTest, floatingPointLargePositiveScales) {
  auto values = makeNullableFlatVector<double>(
      {9'007'199'254'740'991.0,
       -9'007'199'254'740'991.0,
       4'503'599'627'370'495.5,
       -4'503'599'627'370'495.5,
       std::nullopt});
  for (const int32_t scale : {1, 2, 15}) {
    testEncodings(
        makeTypedExpr(
            expression(fmt::format("c0, cast({} as integer)", scale)),
            makeRowVector({values})->rowType()),
        {values},
        values);
  }

  auto ties = makeNullableFlatVector<double>(
      {17'592'186'044'415.75,
       17'592'186'044'416.0,
       17'592'186'044'416.25,
       17'592'186'044'416.75,
       -17'592'186'044'415.75,
       -17'592'186'044'416.0,
       -17'592'186'044'416.25,
       -17'592'186'044'416.75,
       std::nullopt});
  const double evenTie =
      GetParam().halfEven ? 17'592'186'044'416.2 : 17'592'186'044'416.3;
  testEncodings(
      makeTypedExpr(
          expression("c0, cast(1 as integer)"),
          makeRowVector({ties})->rowType()),
      {ties},
      makeNullableFlatVector<double>(
          {17'592'186'044'415.8,
           17'592'186'044'416.0,
           evenTie,
           17'592'186'044'416.8,
           -17'592'186'044'415.8,
           -17'592'186'044'416.0,
           -evenTie,
           -17'592'186'044'416.8,
           std::nullopt}));
}

TEST_P(RoundTest, floatingPointExtremePositiveScales) {
  EXPECT_EQ(
      roundValue<double>(std::numeric_limits<double>::max(), 1),
      std::numeric_limits<double>::max());
  for (const auto value : {2.5, -2.5}) {
    EXPECT_EQ(roundValue<double>(value, 308), value);
    EXPECT_EQ(roundValue<float>(static_cast<float>(value), 308), value);
  }

  const auto minimum = std::numeric_limits<double>::denorm_min();
  for (const auto& [value, scale] : std::vector<std::tuple<double, int32_t>>{
           {minimum, 309},
           {minimum, 323},
           {minimum, 324},
           {3 * minimum, 323},
           {-3 * minimum, 323},
           {4.7e-36, 309},
           {6e-310, 309},
           {-6e-310, 309}}) {
    SCOPED_TRACE(fmt::format("value {}, scale {}", value, scale));
    if (GetParam().halfEven) {
      EXPECT_EQ(roundValue<double>(value, scale), value);
    } else {
      EXPECT_TRUE(std::isnan(roundValue<double>(value, scale).value()));
    }
  }
  for (const auto value :
       {std::numeric_limits<float>::denorm_min(),
        -std::numeric_limits<float>::denorm_min()}) {
    if (GetParam().halfEven) {
      EXPECT_EQ(roundValue<float>(value, 309), value);
    } else {
      EXPECT_TRUE(std::isnan(roundValue<float>(value, 309).value()));
    }
  }
}

TEST_P(RoundTest, specialValuesAndScales) {
  EXPECT_TRUE(
      std::isnan(
          roundValue<double>(std::numeric_limits<double>::quiet_NaN(), 2)
              .value()));
  EXPECT_EQ(
      roundValue<double>(std::numeric_limits<double>::infinity(), 2),
      std::numeric_limits<double>::infinity());
  EXPECT_EQ(
      roundValue<double>(-std::numeric_limits<double>::infinity(), -2),
      -std::numeric_limits<double>::infinity());
  EXPECT_EQ(
      roundValue<double>(
          std::numeric_limits<double>::max(),
          std::numeric_limits<int32_t>::max()),
      std::numeric_limits<double>::max());
  const auto coarsest = roundValue<double>(
                            std::numeric_limits<double>::max(),
                            std::numeric_limits<int32_t>::min())
                            .value();
  if (GetParam().halfEven) {
    EXPECT_EQ(coarsest, 0.0);
  } else {
    EXPECT_TRUE(std::isnan(coarsest));
  }
  EXPECT_EQ(
      roundValue<double>(std::numeric_limits<double>::max(), -308),
      std::numeric_limits<double>::infinity());
  EXPECT_EQ(
      roundValue<double>(-std::numeric_limits<double>::max(), -308),
      -std::numeric_limits<double>::infinity());
}

TEST_P(RoundTest, zeroSigns) {
  const auto check = [this](auto value, int32_t scale) {
    const auto result = roundValue<decltype(value)>(value, scale).value();
    if (!GetParam().halfEven && scale == std::numeric_limits<int32_t>::min()) {
      EXPECT_TRUE(std::isnan(result));
    } else {
      EXPECT_EQ(result, 0);
      EXPECT_EQ(std::signbit(result), !GetParam().halfEven);
    }
  };
  for (const int32_t scale :
       {std::numeric_limits<int32_t>::min(),
        -309,
        -10,
        0,
        10,
        309,
        std::numeric_limits<int32_t>::max()}) {
    SCOPED_TRACE(scale);
    check(-0.0, scale);
    check(-0.0f, scale);
  }
  for (const auto value : {-0.5, -0.1}) {
    if (GetParam().halfEven || value != -0.5) {
      check(value, 0);
    } else {
      EXPECT_EQ(roundValue<double>(value, 0), -1.0);
    }
  }
}

TEST_P(RoundTest, unaryAndNulls) {
  const double roundedTie = GetParam().halfEven ? 2.0 : 3.0;
  EXPECT_EQ((evaluateOnce<double, double>(expression("c0"), 2.5)), roundedTie);
  EXPECT_EQ((evaluateOnce<double, double>(expression("c0"), 3.5)), 4.0);
  EXPECT_EQ((evaluateOnce<int64_t, int64_t>(expression("c0"), 42)), 42);
  EXPECT_EQ(roundValue<double>(std::nullopt, 0), std::nullopt);
  EXPECT_EQ(roundValue<double>(2.5, std::nullopt), std::nullopt);
  EXPECT_EQ(roundValue<int64_t>(std::nullopt, -1), std::nullopt);
  EXPECT_EQ(roundValue<int64_t>(42, std::nullopt), std::nullopt);
}

INSTANTIATE_TEST_SUITE_P(
    RoundingModes,
    RoundTest,
    testing::Values(
        RoundTestParam{false, false},
        RoundTestParam{true, false},
        RoundTestParam{false, true},
        RoundTestParam{true, true}),
    [](const testing::TestParamInfo<RoundTestParam>& info) {
      return fmt::format(
          "{}Ansi{}",
          info.param.halfEven ? "HalfEven" : "HalfUp",
          info.param.ansiEnabled ? "On" : "Off");
    });

} // namespace
} // namespace facebook::velox::functions::sparksql::test
