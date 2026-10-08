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

namespace facebook::velox::functions::sparksql::test {
namespace {

using velox::test::assertEqualVectors;

class IntegralRoundTest : public SparkFunctionBaseTest,
                          public testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    SparkFunctionBaseTest::SetUp();
    setAnsi(GetParam());
  }

  void setAnsi(bool enabled) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled),
          enabled ? "true" : "false"}});
  }

  template <typename T>
  void checkRows(const std::vector<std::tuple<T, int32_t, T>>& rows) {
    assertEqualVectors(
        makeFlatVector<T, 2>(rows),
        evaluate(
            "round(c0, c1)",
            makeRowVector(
                {makeFlatVector<T, 0>(rows),
                 makeFlatVector<int32_t, 1>(rows)})));
  }

  template <typename T>
  void checkCommon() {
    checkRows<T>({
        {25, -1, 30},
        {35, -1, 40},
        {-25, -1, -30},
        {-35, -1, -40},
        {24, -1, 20},
        {26, -1, 30},
        {-24, -1, -20},
        {-26, -1, -30},
        {0, -1, 0},
        {25, -2, 0},
        {-25, 1, -25},
        {25, -100, 0},
    });

    auto values = makeNullableFlatVector<T>(
        {std::numeric_limits<T>::min(),
         std::numeric_limits<T>::max(),
         -25,
         0,
         25,
         std::nullopt});
    auto input = makeRowVector({values});
    for (const auto* expression :
         {"round(c0)",
          "round(c0, cast(0 as integer))",
          "round(c0, cast(2147483647 as integer))"}) {
      assertEqualVectors(values, evaluate(expression, input));
    }
    assertEqualVectors(
        makeNullableFlatVector<T>({0, 0, 0, 0, 0, std::nullopt}),
        evaluate("round(c0, cast(-2147483648 as integer))", input));

    assertEqualVectors(
        makeNullableFlatVector<T>({30, 0, std::nullopt, std::nullopt, 25}),
        evaluate(
            "round(c0, c1)",
            makeRowVector(
                {makeNullableFlatVector<T>({25, 25, 25, std::nullopt, 25}),
                 makeNullableFlatVector<int32_t>(
                     {-1, -2, std::nullopt, -1, 0})})));
    assertEqualVectors(
        makeNullableFlatVector<T>(
            {std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt}),
        evaluate("round(c0, cast(null as integer))", input));
  }

  template <typename T>
  void checkOverflow(T value, int32_t scale, T wrapped) {
    auto input = makeRowVector(
        {makeFlatVector<T>({value}), makeFlatVector<int32_t>({scale})});
    if (GetParam()) {
      VELOX_ASSERT_USER_THROW(
          evaluate("round(c0, c1)", input), "Arithmetic overflow");
    } else {
      assertEqualVectors(
          makeFlatVector<T>({wrapped}), evaluate("round(c0, c1)", input));
    }
  }
};

TEST_P(IntegralRoundTest, halfUpAndIdentity) {
  checkCommon<int8_t>();
  checkCommon<int16_t>();
  checkCommon<int32_t>();
  checkCommon<int64_t>();
}

TEST_P(IntegralRoundTest, narrowOverflow) {
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
}

TEST_P(IntegralRoundTest, coarseScales) {
  checkRows<int8_t>({
      {127, -2, 100},
      {-128, -2, -100},
      {127, -3, 0},
      {-128, -3, 0},
  });
  checkRows<int16_t>({
      {32'767, -4, 30'000},
      {-32'768, -4, -30'000},
      {32'767, -5, 0},
      {-32'768, -5, 0},
  });
  checkRows<int32_t>({
      {2'147'483'647, -9, 2'000'000'000},
      {-2'147'483'647 - 1, -9, -2'000'000'000},
      {2'147'483'647, -10, 0},
      {-2'147'483'647 - 1, -10, 0},
  });
  checkRows<int64_t>({
      {9'223'372'036'854'775'807LL, -18, 9'000'000'000'000'000'000LL},
      {-9'223'372'036'854'775'807LL - 1, -18, -9'000'000'000'000'000'000LL},
      {4'999'999'999'999'999'999LL, -19, 0},
      {-4'999'999'999'999'999'999LL, -19, 0},
      {9'223'372'036'854'775'807LL, -20, 0},
      {-9'223'372'036'854'775'807LL - 1, -20, 0},
      {9'007'199'254'740'995LL, -1, 9'007'199'254'741'000LL},
      {-9'007'199'254'740'995LL, -1, -9'007'199'254'741'000LL},
  });
  checkOverflow<int64_t>(
      9'223'372'036'854'775'807LL, -19, -8'446'744'073'709'551'616LL);
  checkOverflow<int64_t>(
      -9'223'372'036'854'775'807LL - 1, -19, 8'446'744'073'709'551'616LL);
  checkOverflow<int64_t>(
      5'000'000'000'000'000'000LL, -19, -8'446'744'073'709'551'616LL);
  checkOverflow<int64_t>(
      -5'000'000'000'000'000'000LL, -19, 8'446'744'073'709'551'616LL);
}

TEST_P(IntegralRoundTest, tryAndEncodings) {
  auto values =
      makeNullableFlatVector<int8_t>({25, 127, 35, -128, -25, std::nullopt});
  auto scales = makeFlatVector<int32_t>({-1, -1, -2, -1, 0, -1});
  auto expected = GetParam()
      ? makeNullableFlatVector<int8_t>(
            {30, std::nullopt, 0, std::nullopt, -25, std::nullopt})
      : makeNullableFlatVector<int8_t>({30, -126, 0, 126, -25, std::nullopt});
  auto input = makeRowVector({values, scales});
  assertEqualVectors(expected, evaluate("try(round(c0, c1))", input));

  auto indices = makeIndices({2, 1, 4, 3, 0, 5, 1});
  assertEqualVectors(
      GetParam() ? makeNullableFlatVector<int8_t>(
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

  assertEqualVectors(
      GetParam() ? makeNullableFlatVector<int8_t>(
                       {std::nullopt, std::nullopt, std::nullopt})
                 : makeFlatVector<int8_t>({-126, -126, -126}),
      evaluate(
          "try(round(c0, c1))",
          makeRowVector(
              {makeConstant<int8_t>(127, 3), makeConstant<int32_t>(-1, 3)})));
}

TEST_P(IntegralRoundTest, ansiCapturedAtInitialization) {
  auto input = makeRowVector(
      {makeFlatVector<int8_t>({127, 25}), makeFlatVector<int32_t>({-1, -1})});
  auto expression =
      compileExpression("try(round(c0, c1))", asRowType(input->type()));
  auto expected = GetParam()
      ? makeNullableFlatVector<int8_t>({std::nullopt, 30})
      : makeNullableFlatVector<int8_t>({-126, 30});
  assertEqualVectors(expected, evaluate(*expression, input));
  setAnsi(!GetParam());
  assertEqualVectors(expected, evaluate(*expression, input));
  assertEqualVectors(
      GetParam() ? makeNullableFlatVector<int8_t>({-126, 30})
                 : makeNullableFlatVector<int8_t>({std::nullopt, 30}),
      evaluate("try(round(c0, c1))", input));
}

INSTANTIATE_TEST_SUITE_P(
    SessionModes,
    IntegralRoundTest,
    testing::Values(false, true));

} // namespace
} // namespace facebook::velox::functions::sparksql::test
