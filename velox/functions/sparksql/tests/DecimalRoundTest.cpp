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

#include "velox/functions/sparksql/specialforms/DecimalRound.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/Expressions.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/type/HugeInt.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class DecimalRoundTest : public SparkFunctionBaseTest {
 protected:
  core::CallTypedExprPtr createDecimalRounding(
      const TypePtr& inputType,
      const TypePtr& resultType,
      const std::optional<int32_t>& scaleOpt,
      bool castScale,
      const char* functionName) {
    std::vector<core::TypedExprPtr> inputs = {
        std::make_shared<core::FieldAccessTypedExpr>(inputType, "c0")};
    int32_t scale = 0;
    if (scaleOpt.has_value()) {
      scale = scaleOpt.value();
      if (castScale) {
        // It is a common case in Spark for the second argument to be cast from
        // bigint to integer.
        inputs.emplace_back(
            std::make_shared<core::CastTypedExpr>(
                INTEGER(),
                std::make_shared<core::ConstantTypedExpr>(
                    BIGINT(), variant((int64_t)scale)),
                true /*nullOnFailure*/));
      } else {
        inputs.emplace_back(
            std::make_shared<core::ConstantTypedExpr>(
                INTEGER(), variant(scale)));
      }
    }

    return std::make_shared<const core::CallTypedExpr>(
        resultType, std::move(inputs), functionName);
  }

  core::CallTypedExprPtr createDecimalRound(
      const TypePtr& inputType,
      const TypePtr& resultType,
      const std::optional<int32_t>& scaleOpt,
      bool castScale) {
    return createDecimalRounding(
        inputType, resultType, scaleOpt, castScale, kRoundDecimal);
  }

  void testDecimalRound(
      const VectorPtr& input,
      const std::optional<int32_t>& scaleOpt,
      const VectorPtr& expected) {
    for (auto castScale : {true, false}) {
      auto expr = createDecimalRound(
          input->type(), expected->type(), scaleOpt, castScale);
      testEncodings(expr, {input}, expected);
    }
  }

  void testDecimalRoundingWithNullScale(
      const VectorPtr& input,
      const VectorPtr& expected,
      const char* functionName) {
    auto expr = std::make_shared<const core::CallTypedExpr>(
        expected->type(),
        std::vector<core::TypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(input->type(), "c0"),
            std::make_shared<core::ConstantTypedExpr>(
                INTEGER(), variant::null(TypeKind::INTEGER))},
        functionName);
    testEncodings(expr, {input}, expected);
  }
};

TEST_F(DecimalRoundTest, round) {
  // Round to 'scale'.
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      3,
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(4, 3)));

  // Round to 'scale - 1'.
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      2,
      makeFlatVector<int64_t>({12, 55, -100, 0}, DECIMAL(3, 2)));

  // Round to 0 decimal scale.
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      0,
      makeFlatVector<int64_t>({0, 1, -1, 0}, DECIMAL(1, 0)));
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      std::nullopt,
      makeFlatVector<int64_t>({0, 1, -1, 0}, DECIMAL(1, 0)));
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 2)),
      0,
      makeFlatVector<int64_t>({1, 6, -10, 0}, DECIMAL(2, 0)));
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 2)),
      std::nullopt,
      makeFlatVector<int64_t>({1, 6, -10, 0}, DECIMAL(2, 0)));

  // Round to negative decimal scale.
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      -1,
      makeFlatVector<int64_t>({0, 0, 0, 0}, DECIMAL(2, 0)));
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1)),
      -1,
      makeFlatVector<int64_t>({10, 60, -100, 0}, DECIMAL(3, 0)));
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1)),
      -3,
      makeFlatVector<int64_t>({0, 0, 0, 0}, DECIMAL(4, 0)));

  // Round long decimals to short decimals.
  testDecimalRound(
      makeFlatVector<int128_t>(
          {1234567890123456789, 5000000000000000000, -999999999999999999, 0},
          DECIMAL(19, 19)),
      14,
      makeNullableFlatVector<int64_t>(
          {12345678901235, 50000000000000, -10'000'000'000'000, 0},
          DECIMAL(15, 14)));
  testDecimalRound(
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(19, 5)),
      -9,
      makeFlatVector<int64_t>(
          {12346000000000, 55556000000000, -10000000000000, 0},
          DECIMAL(15, 0)));

  // Round long decimals to long decimals.
  testDecimalRound(
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(19, 5)),
      14,
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(20, 5)));
  testDecimalRound(
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(32, 5)),
      -9,
      makeFlatVector<int128_t>(
          {12346000000000, 55556000000000, -10000000000000, 0},
          DECIMAL(28, 0)));

  // Result precision is 38.
  testDecimalRound(
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(32, 0)),
      -38,
      makeFlatVector<int128_t>({0, 0, 0, 0}, DECIMAL(38, 0)));

  // Round to supported scales exceeding the max precision of long decimal.
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1)),
      400,
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(4, 1)));
  testDecimalRound(
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1)),
      -400,
      makeFlatVector<int128_t>({0, 0, 0, 0}, DECIMAL(38, 0)));
}

TEST_F(DecimalRoundTest, sharedExtremeScaleAndErrors) {
  const auto type = DECIMAL(38, 0);
  const auto maximum = HugeInt::parse("99999999999999999999999999999999999999");
  const auto input = makeNullableFlatVector<int128_t>(
      {maximum, -maximum, 0, std::nullopt}, type);
  struct TestCase {
    const char* name;
    int32_t scale;
    bool overflows;
    std::vector<std::optional<int128_t>> expected;
  };
  const std::vector<TestCase> cases{
      {kRoundDecimal, -38, true, {std::nullopt, std::nullopt, 0, std::nullopt}},
      {kRoundDecimal, -39, false, {0, 0, 0, std::nullopt}},
      {kRoundDecimal, -400, false, {0, 0, 0, std::nullopt}},
      {kCeilDecimal, -38, true, {std::nullopt, 0, 0, std::nullopt}},
      {kCeilDecimal, -39, true, {std::nullopt, 0, 0, std::nullopt}},
      {kCeilDecimal, -400, true, {std::nullopt, 0, 0, std::nullopt}},
      {kFloorDecimal, -38, true, {0, std::nullopt, 0, std::nullopt}},
      {kFloorDecimal, -39, true, {0, std::nullopt, 0, std::nullopt}},
      {kFloorDecimal, -400, true, {0, std::nullopt, 0, std::nullopt}}};
  for (const auto& test : cases) {
    SCOPED_TRACE(fmt::format("{} at scale {}", test.name, test.scale));
    const auto call = createDecimalRounding(
        type, DECIMAL(38, 0), test.scale, false, test.name);
    const auto tryCall = std::make_shared<core::CallTypedExpr>(
        DECIMAL(38, 0), std::vector<core::TypedExprPtr>{call}, "try");
    testEncodings(
        tryCall,
        {input},
        makeNullableFlatVector<int128_t>(test.expected, DECIMAL(38, 0)));
    if (test.overflows) {
      VELOX_ASSERT_USER_THROW(
          evaluate(call, makeRowVector({input})),
          fmt::format("Decimal overflow in {}.", test.name));
    }
  }
}

TEST_F(DecimalRoundTest, sharedFlatNoNullsFastPath) {
  struct TestCase {
    TypePtr inputType;
    TypePtr resultType;
    int64_t input;
    std::vector<std::pair<int64_t, int64_t>> expected;
  };
  const std::vector<TestCase> cases{
      {DECIMAL(3, 1), DECIMAL(3, 0), 25, {{3, -3}, {3, -2}, {2, -3}}},
      {DECIMAL(18, 0), DECIMAL(19, 0), 25, {{25, -25}, {25, -25}, {25, -25}}},
      {DECIMAL(19, 2), DECIMAL(18, 0), 250, {{3, -3}, {3, -2}, {2, -3}}},
      {DECIMAL(19, 1), DECIMAL(19, 0), 25, {{3, -3}, {3, -2}, {2, -3}}}};
  const std::vector<const char*> names{
      kRoundDecimal, kCeilDecimal, kFloorDecimal};
  const auto decimalVector = [&](const TypePtr& type,
                                 int64_t positive,
                                 int64_t negative) -> VectorPtr {
    if (type->isShortDecimal()) {
      return makeFlatVector<int64_t>({positive, negative, 0}, type);
    }
    return makeFlatVector<int128_t>({positive, negative, 0}, type);
  };
  for (const auto& test : cases) {
    for (size_t i = 0; i < names.size(); ++i) {
      SCOPED_TRACE(
          fmt::format(
              "{}: {} -> {}",
              names[i],
              test.inputType->toString(),
              test.resultType->toString()));
      const auto call = createDecimalRounding(
          test.inputType, test.resultType, 0, false, names[i]);
      exec::ExprSet expressions({call}, &execCtx_);
      ASSERT_TRUE(expressions.exprs()[0]->supportsFlatNoNullsFastPath());
      const auto input = decimalVector(test.inputType, test.input, -test.input);
      const auto expected = decimalVector(
          test.resultType, test.expected[i].first, test.expected[i].second);
      for (bool constant : {false, true}) {
        const auto row = makeRowVector(
            {constant ? BaseVector::wrapInConstant(3, 0, input) : input});
        exec::EvalCtx context(&execCtx_, &expressions, row.get());
        ASSERT_TRUE(context.inputFlatNoNulls());
        ASSERT_TRUE(context.throwOnError());
        ASSERT_TRUE(queryCtx_->queryConfig().exprEvalFlatNoNulls());
        std::vector<VectorPtr> results(1);
        expressions.eval(SelectivityVector(3), context, results);
        ASSERT_TRUE(
            results[0]->isFlatEncoding() || results[0]->isConstantEncoding());
        facebook::velox::test::assertEqualVectors(
            constant ? BaseVector::wrapInConstant(3, 0, expected) : expected,
            results[0]);
      }
    }
  }
}

TEST_F(DecimalRoundTest, documentedDirectionalResultTypes) {
  const auto input = makeFlatVector<int64_t>({990, -990, 0}, DECIMAL(3, 1));
  for (const char* name : {kCeilDecimal, kFloorDecimal}) {
    SCOPED_TRACE(name);
    const auto call =
        createDecimalRounding(input->type(), DECIMAL(3, 0), -1, false, name);
    testEncodings(
        call,
        {input},
        makeFlatVector<int64_t>(
            name == kCeilDecimal ? std::vector<int64_t>{100, -90, 0}
                                 : std::vector<int64_t>{90, -100, 0},
            DECIMAL(3, 0)));
  }
}

TEST_F(DecimalRoundTest, sharedNullScale) {
  const auto shortInput = makeNullableFlatVector<int64_t>(
      {25, -25, 0, std::nullopt}, DECIMAL(3, 1));
  const auto longInput = makeNullableFlatVector<int128_t>(
      {25, -25, 0, std::nullopt}, DECIMAL(19, 2));
  for (const char* name : {kRoundDecimal, kCeilDecimal, kFloorDecimal}) {
    SCOPED_TRACE(name);
    testDecimalRoundingWithNullScale(
        shortInput,
        BaseVector::createNullConstant(DECIMAL(3, 0), 4, pool()),
        name);
    testDecimalRoundingWithNullScale(
        longInput,
        BaseVector::createNullConstant(DECIMAL(18, 0), 4, pool()),
        name);
  }
}

TEST_F(DecimalRoundTest, directionalOverflowIsUserError) {
  const auto type = DECIMAL(38, 0);
  for (const char* name : {kCeilDecimal, kFloorDecimal}) {
    const auto value = name == kCeilDecimal ? DecimalUtil::kLongDecimalMax
                                            : DecimalUtil::kLongDecimalMin;
    const auto input = makeRowVector({makeFlatVector<int128_t>({value}, type)});
    VELOX_ASSERT_USER_THROW(
        evaluate(
            createDecimalRounding(type, DECIMAL(38, 0), -1, false, name),
            input),
        fmt::format("Decimal overflow in {}.", name));
  }
}

TEST_F(DecimalRoundTest, roundCoarseScaleDoesNotClampDivisor) {
  const int128_t sixTenths = 6 * DecimalUtil::kPowersOfTen[37];
  testDecimalRound(
      makeFlatVector<int128_t>({sixTenths, -sixTenths, 0}, DECIMAL(38, 38)),
      -1,
      makeFlatVector<int64_t>({0, 0, 0}, DECIMAL(2, 0)));
  testDecimalRound(
      makeFlatVector<int128_t>({sixTenths, -sixTenths, 0}, DECIMAL(38, 1)),
      -38,
      makeFlatVector<int128_t>({0, 0, 0}, DECIMAL(38, 0)));
}

TEST_F(DecimalRoundTest, roundNegativeScaleOverflow) {
  const auto maximum = DecimalUtil::kPowersOfTen[38] - 1;
  const auto input =
      makeRowVector({makeFlatVector<int128_t>({maximum}, DECIMAL(38, 0))});
  const auto call =
      createDecimalRound(DECIMAL(38, 0), DECIMAL(38, 0), -1, false);
  VELOX_ASSERT_USER_THROW(
      evaluate(call, input), "Decimal overflow in decimal_round.");
}

TEST_F(DecimalRoundTest, roundMixedErrorsAndCoarseBoundary) {
  const auto type = DECIMAL(38, 0);
  const auto half = 5 * DecimalUtil::kPowersOfTen[37];
  const auto input = makeNullableFlatVector<int128_t>(
      {half - 1, half, half + 1, -half + 1, -half, -half - 1, std::nullopt},
      type);
  const auto call = createDecimalRound(type, DECIMAL(38, 0), -38, false);
  const auto tryCall = std::make_shared<core::CallTypedExpr>(
      type, std::vector<core::TypedExprPtr>{call}, "try");
  const auto expected = makeNullableFlatVector<int128_t>(
      {0,
       std::nullopt,
       std::nullopt,
       0,
       std::nullopt,
       std::nullopt,
       std::nullopt},
      type);
  testEncodings(tryCall, {input}, expected);
  facebook::velox::test::assertEqualVectors(
      expected,
      evaluate(tryCall, makeRowVector({wrapInLazyDictionary(input)})));
  SelectivityVector selected(input->size(), false);
  selected.setValid(0, true);
  selected.setValid(3, true);
  selected.updateBounds();
  const auto result =
      evaluate<SimpleVector<int128_t>>(call, makeRowVector({input}), selected);
  EXPECT_EQ(result->valueAt(0), 0);
  EXPECT_EQ(result->valueAt(3), 0);
  testDecimalRound(
      input,
      -39,
      makeNullableFlatVector<int128_t>({0, 0, 0, 0, 0, 0, std::nullopt}, type));
}

TEST_F(DecimalRoundTest, precision18To19Boundary) {
  const auto input = makeNullableFlatVector<int64_t>(
      {999'999'999'999'999'999LL,
       -999'999'999'999'999'999LL,
       15,
       -15,
       std::nullopt},
      DECIMAL(18, 0));
  const auto resultType = DECIMAL(19, 0);
  const std::vector<
      std::pair<const char*, std::vector<std::optional<int128_t>>>>
      cases{
          {kRoundDecimal,
           {1'000'000'000'000'000'000LL,
            -1'000'000'000'000'000'000LL,
            20,
            -20,
            std::nullopt}},
          {kCeilDecimal,
           {1'000'000'000'000'000'000LL,
            -999'999'999'999'999'990LL,
            20,
            -10,
            std::nullopt}},
          {kFloorDecimal,
           {999'999'999'999'999'990LL,
            -1'000'000'000'000'000'000LL,
            10,
            -20,
            std::nullopt}}};
  for (const auto& [name, values] : cases) {
    testEncodings(
        createDecimalRounding(input->type(), resultType, -1, false, name),
        {input},
        makeNullableFlatVector<int128_t>(values, resultType));
  }
}

TEST_F(DecimalRoundTest, longDecimalCarry) {
  const auto maximum19 = HugeInt::parse("9999999999999999999");
  const auto carry20 = HugeInt::parse("10000000000000000000");
  testDecimalRound(
      makeNullableFlatVector<int128_t>(
          {maximum19, -maximum19, std::nullopt}, DECIMAL(19, 0)),
      -1,
      makeNullableFlatVector<int128_t>(
          {carry20, -carry20, std::nullopt}, DECIMAL(20, 0)));

  const auto maximum38 =
      HugeInt::parse("99999999999999999999999999999999999999");
  const auto carry38 = HugeInt::parse("10000000000000000000000000000000000000");
  testDecimalRound(
      makeNullableFlatVector<int128_t>(
          {maximum38, -maximum38, std::nullopt}, DECIMAL(38, 1)),
      0,
      makeNullableFlatVector<int128_t>(
          {carry38, -carry38, std::nullopt}, DECIMAL(38, 0)));
}

TEST_F(DecimalRoundTest, sharedSelectionErrorsAndResultReuse) {
  const auto type = DECIMAL(38, 0);
  const auto maximum = HugeInt::parse("99999999999999999999999999999999999999");
  const auto towardMaximum =
      HugeInt::parse("99999999999999999999999999999999999990");
  std::vector<std::optional<int128_t>> inputValues;
  for (int i = 0; i < 33; ++i) {
    inputValues.insert(
        inputValues.end(), {maximum, -maximum, 0, std::nullopt, 15, -15});
  }
  const auto input = makeNullableFlatVector<int128_t>(inputValues, type);
  for (bool ansi : {false, true}) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled),
          ansi ? "true" : "false"}});
    const std::vector<
        std::pair<const char*, std::vector<std::optional<int128_t>>>>
        cases{
            {kRoundDecimal,
             {std::nullopt, std::nullopt, 0, std::nullopt, 20, -20}},
            {kCeilDecimal,
             {std::nullopt, -towardMaximum, 0, std::nullopt, 20, -10}},
            {kFloorDecimal,
             {towardMaximum, std::nullopt, 0, std::nullopt, 10, -20}}};
    for (const auto& [name, expectedBlock] : cases) {
      const auto call =
          createDecimalRounding(type, DECIMAL(38, 0), -1, false, name);
      const auto tryCall = std::make_shared<core::CallTypedExpr>(
          type, std::vector<core::TypedExprPtr>{call}, "try");
      std::vector<std::optional<int128_t>> expectedValues;
      for (int i = 0; i < 33; ++i) {
        expectedValues.insert(
            expectedValues.end(), expectedBlock.begin(), expectedBlock.end());
      }
      const auto expected =
          makeNullableFlatVector<int128_t>(expectedValues, type);
      testEncodings(tryCall, {input}, expected);
      facebook::velox::test::assertEqualVectors(
          expected,
          evaluate(tryCall, makeRowVector({wrapInLazyDictionary(input)})));
      const auto selectedCondition = makeFlatVector<bool>(
          input->size(), [](auto row) { return row % 6 >= 4; });
      const auto row = makeRowVector({input, selectedCondition});
      const auto conditional = std::make_shared<core::CallTypedExpr>(
          type,
          std::vector<core::TypedExprPtr>{
              std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "c1"),
              call,
              std::make_shared<core::FieldAccessTypedExpr>(type, "c0")},
          "if");
      auto conditionalExpected = inputValues;
      for (int i = 0; i < input->size(); ++i) {
        if (i % 6 >= 4) {
          conditionalExpected[i] = expectedValues[i];
        }
      }
      facebook::velox::test::assertEqualVectors(
          makeNullableFlatVector<int128_t>(conditionalExpected, type),
          evaluate(conditional, row));

      SelectivityVector selected(input->size(), false);
      for (auto index : {4, 5, 64, 65, 130, 131, 196, 197}) {
        selected.setValid(index, true);
      }
      selected.updateBounds();
      exec::ExprSet expressions({call}, &execCtx_);
      exec::EvalCtx context(&execCtx_, &expressions, row.get());
      std::vector<VectorPtr> results{makeFlatVector<int128_t>(
          input->size(), [](auto) { return -777; }, nullptr, type)};
      expressions.eval(selected, context, results);
      const auto* output = results[0]->as<SimpleVector<int128_t>>();
      for (int i = 0; i < input->size(); ++i) {
        EXPECT_FALSE(output->isNullAt(i));
        EXPECT_EQ(
            output->valueAt(i),
            selected.isValid(i) ? *expectedValues[i] : -777);
      }
      exec::ExprSet tryExpressions({tryCall}, &execCtx_);
      exec::EvalCtx tryContext(&execCtx_, &tryExpressions, row.get());
      tryExpressions.eval(
          SelectivityVector(input->size()), tryContext, results);
      facebook::velox::test::assertEqualVectors(expected, results[0]);
      const auto zeros = makeFlatVector<int128_t>(
          input->size(), [](auto) { return 0; }, nullptr, type);
      const auto zeroRow = makeRowVector({zeros});
      exec::EvalCtx reuseContext(&execCtx_, &tryExpressions, zeroRow.get());
      tryExpressions.eval(
          SelectivityVector(input->size()), reuseContext, results);
      facebook::velox::test::assertEqualVectors(zeros, results[0]);
    }
  }
}

TEST_F(DecimalRoundTest, unaryRoundMetadataValidation) {
  const auto type = DECIMAL(3, 1);
  const auto input = makeRowVector({makeFlatVector<int64_t>({25, 35}, type)});
  const auto field = std::make_shared<core::FieldAccessTypedExpr>(type, "c0");
  const auto call = [&](TypePtr resultType,
                        std::vector<core::TypedExprPtr> args) {
    return std::make_shared<core::CallTypedExpr>(
        resultType, std::move(args), kRoundDecimal);
  };
  for (const auto& output : {DECIMAL(1, 0), DECIMAL(3, 1), TypePtr(BIGINT())}) {
    VELOX_ASSERT_USER_THROW(
        evaluate(call(output, {field}), input),
        fmt::format(
            "Invalid result type for decimal_round: expected DECIMAL(3, 0), got {}.",
            output->toString()));
  }
}

TEST_F(DecimalRoundTest, sharedArgumentAndResultValidation) {
  const auto type = DECIMAL(19, 2);
  const auto row = makeRowVector(
      {makeFlatVector<int128_t>({25, -25, 0}, type),
       makeFlatVector<int32_t>({0, 1, 2})});
  const auto field = std::make_shared<core::FieldAccessTypedExpr>(type, "c0");
  const auto scale =
      std::make_shared<core::ConstantTypedExpr>(INTEGER(), variant(int32_t{0}));
  const auto nullScale = std::make_shared<core::ConstantTypedExpr>(
      INTEGER(), variant::null(TypeKind::INTEGER));
  for (const char* name : {kRoundDecimal, kCeilDecimal, kFloorDecimal}) {
    SCOPED_TRACE(name);
    const auto call = [&](TypePtr result,
                          std::vector<core::TypedExprPtr> args) {
      return std::make_shared<core::CallTypedExpr>(
          result, std::move(args), name);
    };
    for (const auto& output :
         {TypePtr(BIGINT()), DECIMAL(19, 0), DECIMAL(18, 1), DECIMAL(17, 0)}) {
      VELOX_ASSERT_USER_THROW(
          evaluate(call(output, {field, scale}), row),
          fmt::format(
              "Invalid result type for {}: expected DECIMAL(18, 0), got {}.",
              name,
              output->toString()));
      VELOX_ASSERT_USER_THROW(
          evaluate(call(output, {field, nullScale}), row),
          fmt::format(
              "Invalid result type for {}: expected DECIMAL(18, 0), got {}.",
              name,
              output->toString()));
    }
    const auto argumentError = fmt::format(
        "{} expects {} arguments.",
        name,
        name == kRoundDecimal ? "one or two" : "two");
    VELOX_ASSERT_USER_THROW(
        evaluate(call(DECIMAL(18, 0), {}), row), argumentError);
    VELOX_ASSERT_USER_THROW(
        evaluate(call(DECIMAL(18, 0), {field, scale, scale}), row),
        argumentError);
    VELOX_ASSERT_USER_THROW(
        evaluate(
            call(
                DECIMAL(18, 0),
                {field,
                 std::make_shared<core::FieldAccessTypedExpr>(
                     INTEGER(), "c1")}),
            row),
        fmt::format(
            "The second argument of {} must be a constant expression.", name));
    for (const auto& invalidScale :
         {std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), variant(true)),
          std::make_shared<core::ConstantTypedExpr>(
              BIGINT(), variant(int64_t{0}))}) {
      VELOX_ASSERT_USER_THROW(
          evaluate(call(DECIMAL(18, 0), {field, invalidScale}), row),
          fmt::format("The second argument of {} must be INTEGER.", name));
    }
    VELOX_ASSERT_USER_THROW(
        evaluate(
            call(
                DECIMAL(18, 0),
                {std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c1"),
                 scale}),
            row),
        fmt::format("The first argument of {} must be decimal.", name));
    VELOX_ASSERT_USER_THROW(
        evaluate(fmt::format("{}(c0, cast(0 as integer))", name), row),
        fmt::format("{} requires an explicitly resolved result type.", name));
    if (name == kCeilDecimal || name == kFloorDecimal) {
      VELOX_ASSERT_USER_THROW(
          evaluate(call(DECIMAL(18, 0), {field}), row), argumentError);
    }
  }
}

TEST_F(DecimalRoundTest, sharedUnsupportedScale) {
  const std::vector<VectorPtr> inputs{
      makeNullableFlatVector<int64_t>({0, std::nullopt}, DECIMAL(3, 1)),
      makeNullableFlatVector<int128_t>({0, std::nullopt}, DECIMAL(19, 1))};
  for (bool ansi : {false, true}) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled),
          ansi ? "true" : "false"}});
    for (const auto& input : inputs) {
      const auto row = makeRowVector({input});
      for (const char* name : {kRoundDecimal, kCeilDecimal, kFloorDecimal}) {
        for (int32_t scale :
             {-401,
              401,
              std::numeric_limits<int32_t>::min(),
              std::numeric_limits<int32_t>::max()}) {
          SCOPED_TRACE(
              fmt::format("{} at scale {}, ANSI {}", name, scale, ansi));
          const auto resultType = scale < 0     ? DECIMAL(38, 0)
              : input->type()->isShortDecimal() ? DECIMAL(4, 1)
                                                : DECIMAL(20, 1);
          for (bool castScale : {false, true}) {
            const auto call = createDecimalRounding(
                input->type(), resultType, scale, castScale, name);
            VELOX_ASSERT_USER_THROW(
                evaluate(call, row), "Scale must be in [-400, 400]");
            const auto tryCall = std::make_shared<core::CallTypedExpr>(
                resultType, std::vector<core::TypedExprPtr>{call}, "try");
            VELOX_ASSERT_USER_THROW(
                evaluate(tryCall, row), "Scale must be in [-400, 400]");
          }
        }
      }
    }
  }
}

TEST_F(DecimalRoundTest, sharedSupportedScaleBoundaries) {
  const std::vector<std::pair<VectorPtr, VectorPtr>> inputs{
      {makeNullableFlatVector<int64_t>(
           {25, -25, 0, std::nullopt}, DECIMAL(3, 1)),
       makeNullableFlatVector<int64_t>(
           {25, -25, 0, std::nullopt}, DECIMAL(4, 1))},
      {makeNullableFlatVector<int128_t>(
           {25, -25, 0, std::nullopt}, DECIMAL(19, 1)),
       makeNullableFlatVector<int128_t>(
           {25, -25, 0, std::nullopt}, DECIMAL(20, 1))}};
  const std::vector<
      std::pair<const char*, std::vector<std::optional<int128_t>>>>
      lowerBoundResults{
          {kRoundDecimal, {0, 0, 0, std::nullopt}},
          {kCeilDecimal, {std::nullopt, 0, 0, std::nullopt}},
          {kFloorDecimal, {0, std::nullopt, 0, std::nullopt}}};
  for (const auto& [input, unchanged] : inputs) {
    for (const auto& [name, lowerBoundResult] : lowerBoundResults) {
      SCOPED_TRACE(name);
      for (bool castScale : {false, true}) {
        const auto upper = createDecimalRounding(
            input->type(), unchanged->type(), 400, castScale, name);
        testEncodings(upper, {input}, unchanged);
        const auto lower = createDecimalRounding(
            input->type(), DECIMAL(38, 0), -400, castScale, name);
        const auto tryLower = std::make_shared<core::CallTypedExpr>(
            DECIMAL(38, 0), std::vector<core::TypedExprPtr>{lower}, "try");
        testEncodings(
            tryLower,
            {input},
            makeNullableFlatVector<int128_t>(lowerBoundResult, DECIMAL(38, 0)));
      }
    }
  }
}

TEST_F(DecimalRoundTest, sharedNullScaleSkipsChild) {
  const auto type = DECIMAL(38, 0);
  const auto maximum = DecimalUtil::kPowersOfTen[38] - 1;
  const auto input =
      makeNullableFlatVector<int128_t>({maximum, -maximum, std::nullopt}, type);
  const auto child = createDecimalRound(type, DECIMAL(38, 0), -1, false);
  for (const char* name : {kRoundDecimal, kCeilDecimal, kFloorDecimal}) {
    const auto call = std::make_shared<core::CallTypedExpr>(
        type,
        std::vector<core::TypedExprPtr>{
            child,
            std::make_shared<core::ConstantTypedExpr>(
                INTEGER(), variant::null(TypeKind::INTEGER))},
        name);
    testEncodings(
        call,
        {input},
        BaseVector::createNullConstant(type, input->size(), pool()));
  }
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
