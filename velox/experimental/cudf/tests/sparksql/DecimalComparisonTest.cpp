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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::cudf_velox {
namespace {

class DecimalComparisonTest : public CudfFunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::sparksql::registerFunctions();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
    registerSparkFunctions("");
  }

  static void TearDownTestCase() {
    unregisterFunctions();
    unregisterCudf();
  }

  VectorPtr makeDecimalVector(
      const TypePtr& type,
      const std::vector<std::optional<int128_t>>& values) {
    if (type->isLongDecimal()) {
      return makeNullableFlatVector<int128_t>(values, type);
    }
    std::vector<std::optional<int64_t>> shortValues;
    for (const auto& value : values) {
      shortValues.emplace_back(
          value.has_value() ? std::make_optional(static_cast<int64_t>(*value))
                            : std::nullopt);
    }
    return makeNullableFlatVector<int64_t>(shortValues, type);
  }

  const std::vector<std::string> comparisons_{
      "decimal_equalto",
      "decimal_notequalto",
      "decimal_lessthan",
      "decimal_lessthanorequal",
      "decimal_greaterthan",
      "decimal_greaterthanorequal"};
};

TEST_F(DecimalComparisonTest, columns) {
  const std::vector<std::pair<TypePtr, TypePtr>> types{
      {DECIMAL(18, 0), DECIMAL(18, 0)},
      {DECIMAL(38, 0), DECIMAL(38, 0)},
      {DECIMAL(12, 2), DECIMAL(14, 4)},
      {DECIMAL(18, 2), DECIMAL(38, 4)},
      {DECIMAL(28, 3), DECIMAL(30, 5)},
      {DECIMAL(38, 20), DECIMAL(38, 20)},
      {DECIMAL(10, 0), DECIMAL(10, 8)},
      {DECIMAL(38, 38), DECIMAL(38, 38)}};
  for (const auto& [leftType, rightType] : types) {
    SCOPED_TRACE(leftType->toString() + ", " + rightType->toString());
    const auto leftMax =
        DecimalUtil::kPowersOfTen[getDecimalPrecisionScale(*leftType).first] -
        1;
    const auto rightMax =
        DecimalUtil::kPowersOfTen[getDecimalPrecisionScale(*rightType).first] -
        1;
    auto input = makeRowVector({
        makeDecimalVector(
            leftType,
            {0,
             1,
             -1,
             999,
             -999,
             leftMax,
             -leftMax,
             std::nullopt,
             0,
             std::nullopt}),
        makeDecimalVector(
            rightType,
            {0,
             -1,
             1,
             999,
             -999,
             rightMax,
             -rightMax,
             0,
             std::nullopt,
             std::nullopt}),
    });
    for (const auto& comparison : comparisons_) {
      for (const auto& arguments : {"c0, c1", "c1, c0"}) {
        auto expression = fmt::format("{}({})", comparison, arguments);
        SCOPED_TRACE(expression);
        auto typed = makeTypedExpr(expression, input->rowType());
        ASSERT_TRUE(FunctionExpression::canEvaluate(typed));
        ASSERT_FALSE(ASTExpression::canEvaluate(typed));
        assertExpressionMatchesCpu(expression, input, input->rowType());
      }
    }
  }
}

TEST_F(DecimalComparisonTest, constants) {
  const std::vector<std::pair<TypePtr, TypePtr>> types{
      {DECIMAL(18, 2), DECIMAL(18, 2)},
      {DECIMAL(38, 20), DECIMAL(38, 20)},
      {DECIMAL(12, 2), DECIMAL(14, 4)},
      {DECIMAL(18, 2), DECIMAL(38, 4)},
      {DECIMAL(28, 3), DECIMAL(30, 5)}};
  for (const auto& [columnType, literalType] : types) {
    SCOPED_TRACE(columnType->toString() + ", " + literalType->toString());
    const auto scale = getDecimalPrecisionScale(*columnType).second;
    const auto unit = DecimalUtil::kPowersOfTen[scale];
    auto input = makeRowVector({makeDecimalVector(
        columnType, {-unit, -unit / 2, 0, unit / 2, unit, std::nullopt})});
    for (const auto& comparison : comparisons_) {
      for (const auto& constant : {"'0.90'", "'-0.90'", "null"}) {
        auto literal =
            fmt::format("cast({} as {})", constant, literalType->toString());
        for (const auto& arguments :
             {fmt::format("c0, {}", literal), fmt::format("{}, c0", literal)}) {
          auto expression = fmt::format("{}({})", comparison, arguments);
          SCOPED_TRACE(expression);
          assertExpressionMatchesCpu(expression, input, input->rowType());
        }
      }
    }
  }
}

TEST_F(DecimalComparisonTest, hashJoinResidual) {
  auto probe = makeRowVector(
      {"probe_key", "numerator"},
      {makeFlatVector<int64_t>({1, 1, 1, 1, 2, 3}),
       makeNullableFlatVector<int64_t>(
           {800, 900, 1'000, std::nullopt, 100, 400}, DECIMAL(17, 2))});
  auto build = makeRowVector(
      {"build_key", "denominator"},
      {makeFlatVector<int64_t>({1, 1, 2}),
       makeFlatVector<int64_t>({1'000, 2'000, 200}, DECIMAL(17, 2))});
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  // A decimal predicate spanning both join sides must use the function
  // evaluator after joining because cuDF AST does not support decimals.
  auto plan = exec::test::PlanBuilder(planNodeIdGenerator)
                  .values({probe})
                  .hashJoin(
                      {"probe_key"},
                      {"build_key"},
                      exec::test::PlanBuilder(planNodeIdGenerator)
                          .values({build})
                          .planNode(),
                      "decimal_lessthan(divide(numerator, denominator), "
                      "cast('0.9' as decimal(37, 20)))",
                      {"probe_key", "numerator"},
                      core::JoinType::kInner)
                  .planNode();
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 1, 1, 2}),
      makeFlatVector<int64_t>({800, 800, 900, 1'000, 100}, DECIMAL(17, 2)),
  });
  exec::test::AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(DecimalComparisonTest, selection) {
  // Aligning scales must not overflow the common decimal storage width.
  const std::vector<std::pair<TypePtr, TypePtr>> unsupportedTypes{
      {DECIMAL(18, 0), DECIMAL(18, 18)},
      {DECIMAL(38, 0), DECIMAL(38, 38)},
      {DECIMAL(38, 0), DECIMAL(18, 2)},
      {BIGINT(), BIGINT()},
      {DOUBLE(), DOUBLE()},
      {DECIMAL(18, 2), BIGINT()}};
  for (const auto& [leftType, rightType] : unsupportedTypes) {
    SCOPED_TRACE(leftType->toString() + ", " + rightType->toString());
    for (const auto& comparison : comparisons_) {
      SCOPED_TRACE(comparison);
      auto expression = std::make_shared<core::CallTypedExpr>(
          BOOLEAN(),
          std::vector<core::TypedExprPtr>{
              std::make_shared<core::FieldAccessTypedExpr>(leftType, "left"),
              std::make_shared<core::FieldAccessTypedExpr>(rightType, "right")},
          comparison);
      ASSERT_FALSE(FunctionExpression::canEvaluate(expression));
      ASSERT_EQ(
          createCudfFunction(comparison, expression, pool_.get()), nullptr);
    }
  }

  registerBuiltinFunctions("spark_");
  auto decimal =
      std::make_shared<core::FieldAccessTypedExpr>(DECIMAL(18, 2), "value");
  auto literal = std::make_shared<core::ConstantTypedExpr>(
      makeFlatVector<int64_t>({90}, DECIMAL(18, 2)));
  for (const auto& comparison : comparisons_) {
    SCOPED_TRACE(comparison);
    auto prefixed = std::make_shared<core::CallTypedExpr>(
        BOOLEAN(),
        std::vector<core::TypedExprPtr>{decimal, decimal},
        "spark_" + comparison);
    ASSERT_TRUE(FunctionExpression::canEvaluate(prefixed));

    for (const auto& resultType :
         std::vector<TypePtr>{BIGINT(), DECIMAL(18, 2)}) {
      auto invalidResult = std::make_shared<core::CallTypedExpr>(
          resultType,
          std::vector<core::TypedExprPtr>{decimal, decimal},
          comparison);
      ASSERT_FALSE(FunctionExpression::canEvaluate(invalidResult));
    }

    for (const auto& arguments : std::vector<std::vector<core::TypedExprPtr>>{
             {decimal}, {decimal, decimal, decimal}, {literal, literal}}) {
      auto unsupported = std::make_shared<core::CallTypedExpr>(
          BOOLEAN(), arguments, comparison);
      ASSERT_FALSE(FunctionExpression::canEvaluate(unsupported));
      ASSERT_EQ(
          createCudfFunction(comparison, unsupported, pool_.get()), nullptr);
    }
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox
