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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/expression/FunctionCallToSpecialForm.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

std::vector<int64_t> evaluatedInputs;

template <typename T>
struct CountedIdentityFunction {
  static constexpr bool is_deterministic = false;

  void call(int64_t& result, int64_t input) {
    evaluatedInputs.push_back(input);
    result = input;
  }
};

class PmodSpecialFormTest : public SparkFunctionBaseTest {
 protected:
  static std::string expression(bool ansi, bool early = false) {
    return fmt::format("pmod_with_mode(c0, c1, {}, {})", ansi, early);
  }

  template <typename T>
  void checkValues(
      const std::string& expression,
      const RowVectorPtr& input,
      const std::vector<std::optional<T>>& expected) {
    auto result = evaluate<SimpleVector<T>>(expression, input);
    ASSERT_EQ(*result->type(), *CppToType<T>::create());
    ASSERT_EQ(result->size(), expected.size());
    for (auto row = 0; row < expected.size(); ++row) {
      SCOPED_TRACE(row);
      ASSERT_EQ(result->isNullAt(row), !expected[row].has_value());
      if (expected[row]) {
        const auto actual = result->valueAt(row);
        if constexpr (std::is_floating_point_v<T>) {
          if (std::isnan(*expected[row])) {
            EXPECT_TRUE(std::isnan(actual));
            continue;
          }
          if (*expected[row] == 0) {
            EXPECT_EQ(std::signbit(actual), std::signbit(*expected[row]));
          }
        }
        EXPECT_EQ(actual, *expected[row]);
      }
    }
  }

  template <typename T>
  void testEncodings() {
    auto dividend = makeNullableFlatVector<T>({-5, -5, 1, std::nullopt, 5});
    auto divisor = makeNullableFlatVector<T>({3, 0, std::nullopt, 0, -3});
    auto indices = makeIndices({4, 3, 2, 1, 0});
    auto dictionary = makeRowVector(
        {wrapInDictionary(indices, 5, dividend),
         wrapInDictionary(indices, 5, divisor)});
    for (const bool ansi : {false, true}) {
      checkValues<T>(
          "try(" + expression(ansi) + ")",
          dictionary,
          {2, std::nullopt, std::nullopt, std::nullopt, 1});
      checkValues<T>(
          expression(ansi),
          makeRowVector({makeConstant<T>(-5, 5), makeConstant<T>(3, 5)}),
          {1, 1, 1, 1, 1});
    }
    auto lazyDivisor = std::make_shared<LazyVector>(
        pool(),
        CppToType<T>::create(),
        5,
        std::make_unique<facebook::velox::test::SimpleVectorLoader>(
            [divisor](RowSet) { return divisor; }));
    checkValues<T>(
        "try(" + expression(true) + ")",
        makeRowVector({dividend, lazyDivisor}),
        {1, std::nullopt, std::nullopt, std::nullopt, 2});

    SelectivityVector selected(5, false);
    selected.setValid(0, true);
    selected.setValid(4, true);
    selected.updateBounds();
    auto partial = evaluate<SimpleVector<T>>(
        expression(true), makeRowVector({dividend, divisor}), selected);
    EXPECT_EQ(partial->valueAt(0), 1);
    EXPECT_EQ(partial->valueAt(4), 2);
  }

  template <typename T>
  void testIntegral() {
    constexpr auto kMin = std::numeric_limits<T>::min();
    constexpr auto kMax = std::numeric_limits<T>::max();
    const std::vector<std::optional<T>> dividend = {
        -1, -1, kMin, kMin, kMin, kMax, 5, -5, 5, -5, 0, std::nullopt, 7};
    const std::vector<std::optional<T>> divisor = {
        kMin, -kMax, -1, kMin, kMax, kMin, 3, 3, -3, -3, -3, 0, std::nullopt};
    const std::vector<std::optional<T>> expected = {
        sizeof(T) < 4 ? T(-1) : kMax,
        -1,
        0,
        0,
        T(kMax - 1),
        kMax,
        2,
        1,
        2,
        -2,
        0,
        std::nullopt,
        std::nullopt};
    auto input = makeRowVector(
        {makeNullableFlatVector<T>(dividend),
         makeNullableFlatVector<T>(divisor)});
    for (const bool ansi : {false, true}) {
      checkValues<T>(expression(ansi), input, expected);
    }
    checkValues<T>("pmod(c0, c1)", input, expected);

    auto zeros = makeRowVector(
        {makeNullableFlatVector<T>({kMin, std::nullopt, 1, 5}),
         makeFlatVector<T>({0, 0, 0, 3})});
    checkValues<T>(
        expression(false),
        zeros,
        {std::nullopt, std::nullopt, std::nullopt, 2});
    checkValues<T>(
        "try(" + expression(true) + ")",
        zeros,
        {std::nullopt, std::nullopt, std::nullopt, 2});
    VELOX_ASSERT_USER_THROW(
        evaluate(expression(true), zeros), "Division by zero");
    testEncodings<T>();
  }

  template <typename T>
  void testFloating() {
    constexpr auto kInf = std::numeric_limits<T>::infinity();
    constexpr auto kNaN = std::numeric_limits<T>::quiet_NaN();
    constexpr auto kMax = std::numeric_limits<T>::max();
    constexpr auto kTiny = std::numeric_limits<T>::denorm_min();
    // The correction must round before computing the second remainder.
    constexpr auto kSmall = std::numeric_limits<T>::epsilon() / 4;
    const std::vector<std::optional<T>> dividend = {
        -T(0),  T(0),  -6,        6,    -1,    1,       kInf,
        -kInf,  kNaN,  1,         -1,   T(0),  -T(0),   kTiny,
        -kTiny, -kMax, -kMax / 2, kMax, -kMax, -kSmall, std::nullopt};
    const std::vector<std::optional<T>> divisor = {
        3,    -3,   3, -3, kInf,  -kInf, 1,  1,  1, kNaN, -kInf,
        kInf, kInf, 1, 1,  -kMax, -kMax, -1, -1, 1, 0};
    const std::vector<std::optional<T>> expected = {
        -T(0), T(0),  -T(0), T(0), kNaN,  1,     kNaN,
        kNaN,  kNaN,  kNaN,  kNaN, T(0),  -T(0), kTiny,
        T(0),  -T(0), kNaN,  T(0), -T(0), T(0),  std::nullopt};
    auto input = makeRowVector(
        {makeNullableFlatVector<T>(dividend),
         makeNullableFlatVector<T>(divisor)});
    for (const bool ansi : {false, true}) {
      checkValues<T>(expression(ansi), input, expected);
    }
    checkValues<T>("pmod(c0, c1)", input, expected);

    auto zeros = makeRowVector(
        {makeNullableFlatVector<T>({1, kNaN, kInf, std::nullopt, 5}),
         makeFlatVector<T>({T(0), -T(0), T(0), -T(0), 3})});
    checkValues<T>(
        expression(false),
        zeros,
        {std::nullopt, std::nullopt, std::nullopt, std::nullopt, 2});
    checkValues<T>(
        "try(" + expression(true) + ")",
        zeros,
        {std::nullopt, std::nullopt, std::nullopt, std::nullopt, 2});
    VELOX_ASSERT_USER_THROW(
        evaluate(expression(true), zeros), "Division by zero");
    testEncodings<T>();
  }
};

TEST_F(PmodSpecialFormTest, tinyint) {
  testIntegral<int8_t>();
}

TEST_F(PmodSpecialFormTest, smallint) {
  testIntegral<int16_t>();
}

TEST_F(PmodSpecialFormTest, integer) {
  testIntegral<int32_t>();
}

TEST_F(PmodSpecialFormTest, bigint) {
  testIntegral<int64_t>();
}

TEST_F(PmodSpecialFormTest, real) {
  testFloating<float>();
}

TEST_F(PmodSpecialFormTest, double) {
  testFloating<double>();
}

TEST_F(PmodSpecialFormTest, constantAndDictionary) {
  auto values = makeFlatVector<int64_t>({-5, 5, INT64_MIN, -1});
  auto indices = makeIndices({3, 0, 1, 2, 3, 0});
  auto input = makeRowVector(
      {wrapInDictionary(indices, 6, values), makeConstant<int64_t>(3, 6)});
  checkValues<int64_t>(expression(false), input, {2, 1, 2, 1, 2, 1});
  checkValues<int64_t>(expression(true), input, {2, 1, 2, 1, 2, 1});
  checkValues<int64_t>(
      "pmod_with_mode(cast(-1 as bigint), cast(3 as bigint), true, false)",
      input,
      {2, 2, 2, 2, 2, 2});
}

TEST_F(PmodSpecialFormTest, selectionAndReusedResult) {
  auto input = makeRowVector(
      {makeFlatVector<int64_t>({1, -5, 1, 5}),
       makeFlatVector<int64_t>({0, 3, 0, 3})});
  SelectivityVector rows(4, false);
  rows.setValid(1, true);
  rows.setValid(3, true);
  rows.updateBounds();
  VectorPtr result = makeFlatVector<int64_t>({91, 92, 93, 94});
  evaluate<FlatVector<int64_t>>(expression(true), input, rows, result);
  facebook::velox::test::assertEqualVectors(
      makeFlatVector<int64_t>({91, 1, 93, 2}), result);
}

TEST_F(PmodSpecialFormTest, lazyLeftShortCircuit) {
  std::vector<vector_size_t> loaded;
  auto left = std::make_shared<LazyVector>(
      pool(),
      BIGINT(),
      5,
      std::make_unique<facebook::velox::test::SimpleVectorLoader>(
          [&](RowSet rows) -> VectorPtr {
            loaded.assign(rows.begin(), rows.end());
            EXPECT_EQ(loaded, (std::vector<vector_size_t>{2, 4}));
            return makeFlatVector<int64_t>({-5, -5, -5, -5, -5});
          }));
  auto input = makeRowVector(
      {left, makeNullableFlatVector<int64_t>({0, std::nullopt, 3, 0, 3})});
  checkValues<int64_t>(
      expression(false),
      input,
      {std::nullopt, std::nullopt, 1, std::nullopt, 1});
  EXPECT_EQ(loaded, (std::vector<vector_size_t>{2, 4}));
}

TEST_F(PmodSpecialFormTest, sharedLazyField) {
  const std::vector<int64_t> values = {-5, 5, -7, 4};
  auto lazy = makeLazyFlatVector<int64_t>(
      4, [&](auto row) { return values[row]; }, [](auto) { return false; }, 4);
  auto input =
      makeRowVector({lazy, makeFlatVector<bool>({true, false, true, false})});
  checkValues<int64_t>(
      "pmod_with_mode(c0, if(c1, c0, cast(3 as bigint)), false, false)",
      input,
      {0, 2, 0, 1});
}

TEST_F(PmodSpecialFormTest, legacyZeroWithEarlyCheckSkipsThrowingLeft) {
  auto input = makeRowVector({makeFlatVector<std::string>({"left error"})});
  for (const auto& type : std::vector<TypePtr>{
           TINYINT(), SMALLINT(), INTEGER(), BIGINT(), REAL(), DOUBLE()}) {
    SCOPED_TRACE(type->toString());
    const auto left =
        fmt::format("cast(raise_error(c0) as {})", type->toString());
    VELOX_ASSERT_USER_THROW(evaluate(left, input), "left error");

    auto result = evaluate(
        fmt::format(
            "pmod_with_mode({}, cast(0 as {}), false, true)",
            left,
            type->toString()),
        input);
    ASSERT_EQ(*result->type(), *type);
    ASSERT_EQ(result->size(), 1);
    EXPECT_TRUE(result->isNullAt(0));
  }
}

TEST_F(PmodSpecialFormTest, nullAndErrorPrecedence) {
  auto input = makeRowVector(
      {makeFlatVector<int64_t>({0}),
       makeFlatVector<std::string>({"left error"})});
  const std::string left = "cast(raise_error(c1) as bigint)";
  const std::string right = "cast(raise_error('right error') as bigint)";
  checkValues<int64_t>(
      "pmod_with_mode(" + left + ", c0, false, false)", input, {std::nullopt});
  checkValues<int64_t>(
      "pmod_with_mode(" + left + ", cast(null as bigint), true, false)",
      input,
      {std::nullopt});
  checkValues<int64_t>(
      "pmod_with_mode(cast(null as bigint), c0, true, false)",
      input,
      {std::nullopt});
  VELOX_ASSERT_USER_THROW(
      evaluate("pmod_with_mode(" + left + ", c0, true, false)", input),
      "left error");
  VELOX_ASSERT_USER_THROW(
      evaluate("pmod_with_mode(" + left + ", c0, true, true)", input),
      "Division by zero");
  VELOX_ASSERT_USER_THROW(
      evaluate(
          "pmod_with_mode(" + left + ", " + right + ", true, false)", input),
      "right error");
  VELOX_ASSERT_USER_THROW(
      evaluate(
          "pmod_with_mode(cast(null as bigint), " + right + ", false, false)",
          input),
      "right error");
  checkValues<int64_t>(
      "if(equalto(c0, 0), cast(7 as bigint), pmod_with_mode(" + left +
          ", c0, true, false))",
      input,
      {7});
}

TEST_F(PmodSpecialFormTest, evaluatesArgumentsOnce) {
  registerFunction<CountedIdentityFunction, int64_t, int64_t>(
      {"pmod_test_counted_identity"});
  evaluatedInputs.clear();
  auto input = makeRowVector(
      {makeFlatVector<int64_t>({-5, -7, -8}),
       makeFlatVector<int64_t>({3, 0, 3})});
  checkValues<int64_t>(
      "pmod_with_mode(pmod_test_counted_identity(c0), "
      "pmod_test_counted_identity(c1), false, false)",
      input,
      {1, std::nullopt, 1});
  EXPECT_EQ(evaluatedInputs, (std::vector<int64_t>{3, 0, 3, -5, -8}));
}

TEST_F(PmodSpecialFormTest, tryMixedArgumentErrors) {
  auto input = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5})});
  checkValues<int64_t>(
      "try(pmod_with_mode("
      "if(equalto(c0, 1), cast(raise_error('left') as bigint), "
      "if(equalto(c0, 2), cast(null as bigint), cast(-5 as bigint))),"
      "if(equalto(c0, 3), cast(raise_error('right') as bigint), "
      "if(equalto(c0, 4), cast(null as bigint), if(equalto(c0, 0), cast(0 as bigint), "
      "cast(3 as bigint)))), true, false))",
      input,
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       1});
}

TEST_F(PmodSpecialFormTest, capturedModeSurvivesConfigChanges) {
  auto input = makeRowVector(
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({0})});
  for (const bool initialAnsi : {false, true}) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled),
          initialAnsi ? "true" : "false"}});
    auto legacy =
        compileExpression(expression(false), asRowType(input->type()));
    auto ansi = compileExpression(expression(true), asRowType(input->type()));
    auto ordinary = compileExpression("pmod(c0, c1)", asRowType(input->type()));
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled),
          initialAnsi ? "false" : "true"}});
    EXPECT_TRUE(evaluate(*legacy, input)->isNullAt(0));
    VELOX_ASSERT_USER_THROW(evaluate(*ansi, input), "Division by zero");
    if (initialAnsi) {
      VELOX_ASSERT_USER_THROW(evaluate(*ordinary, input), "Division by zero");
    } else {
      EXPECT_TRUE(evaluate(*ordinary, input)->isNullAt(0));
    }
  }
}

TEST_F(PmodSpecialFormTest, registrationAndInvalidArguments) {
  ASSERT_TRUE(exec::isFunctionCallToSpecialFormRegistered("pmod_with_mode"));
  for (const auto& type : std::vector<TypePtr>{
           TINYINT(), SMALLINT(), INTEGER(), BIGINT(), REAL(), DOUBLE()}) {
    EXPECT_EQ(
        *exec::resolveTypeForSpecialForm(
            "pmod_with_mode", {type, type, BOOLEAN(), BOOLEAN()}),
        *type);
  }
  for (const auto& type :
       std::vector<TypePtr>{BOOLEAN(), VARCHAR(), DATE(), DECIMAL(10, 2)}) {
    VELOX_ASSERT_USER_THROW(
        exec::resolveTypeForSpecialForm(
            "pmod_with_mode", {type, type, BOOLEAN(), BOOLEAN()}),
        "primitive numeric");
  }
  auto input = makeRowVector(
      {makeFlatVector<int64_t>({1}), makeFlatVector<bool>({true})});
  VELOX_ASSERT_USER_THROW(
      evaluate("pmod_with_mode(c0, c0, c1, false)", input), "constant");
  VELOX_ASSERT_USER_THROW(
      evaluate("pmod_with_mode(c0, c0, true, c1)", input), "constant");
  VELOX_ASSERT_USER_THROW(
      evaluate("pmod_with_mode(c0, c0, cast(null as boolean), false)", input),
      "non-null");
  VELOX_ASSERT_USER_THROW(
      evaluate("pmod_with_mode(c0, c0, true, cast(null as boolean))", input),
      "non-null");
  VELOX_ASSERT_USER_THROW(
      evaluate("pmod_with_mode(c0, c0, true)", input), "4 arguments");
  VELOX_ASSERT_USER_THROW(
      evaluate("pmod_with_mode(c0, cast(c0 as double), true, false)", input),
      "same type");
}

TEST_F(PmodSpecialFormTest, rejectsNonBooleanConstantOptions) {
  auto input = makeRowVector({makeFlatVector<int64_t>({1})});
  for (const auto& option : {"cast(1 as integer)", "'true'"}) {
    SCOPED_TRACE(option);
    VELOX_ASSERT_USER_THROW(
        evaluate(
            fmt::format("pmod_with_mode(c0, c0, {}, false)", option), input),
        "pmod_with_mode options must be booleans");
    VELOX_ASSERT_USER_THROW(
        evaluate(
            fmt::format("pmod_with_mode(c0, c0, true, {})", option), input),
        "pmod_with_mode options must be booleans");
  }
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
