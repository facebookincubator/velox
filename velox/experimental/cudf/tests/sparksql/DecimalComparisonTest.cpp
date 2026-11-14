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

#include <folly/ScopeGuard.h>

#include <array>

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

  void assertGpuComparison(
      const std::string& expression,
      const RowVectorPtr& input,
      const VectorPtr& expected) {
    const auto typed = makeTypedExpr(expression, input->rowType());
    ASSERT_TRUE(FunctionExpression::canEvaluate(typed));
    ASSERT_FALSE(ASTExpression::canEvaluate(typed));
    facebook::velox::test::assertEqualVectors(expected, evaluate(typed, input));
  }

  struct CpuComparisonCase {
    std::string name;
    std::vector<bool> shortResults;
    std::array<bool, 2> alternatingResults;
    bool longResult;
  };

  // Expected values from functions/sparksql/tests/DecimalCompareTest.cpp.
  const std::vector<CpuComparisonCase> cpuComparisons_{
      {"decimal_equalto", {true, false, false, false}, {true, false}, false},
      {"decimal_notequalto", {false, true, true, true}, {false, true}, true},
      {"decimal_lessthan", {false, false, false, true}, {false, false}, true},
      {"decimal_lessthanorequal",
       {true, false, false, true},
       {true, false},
       true},
      {"decimal_greaterthan", {false, true, true, false}, {false, true}, false},
      {"decimal_greaterthanorequal",
       {true, true, true, false},
       {true, true},
       false}};

  const std::vector<std::string> comparisons_{
      "decimal_equalto",
      "decimal_notequalto",
      "decimal_lessthan",
      "decimal_lessthanorequal",
      "decimal_greaterthan",
      "decimal_greaterthanorequal"};
};

TEST_F(DecimalComparisonTest, cpuInputEncodings) {
  const auto left =
      makeFlatVector<int64_t>({1000, 2000, 3000, 400}, DECIMAL(6, 2));
  const auto right =
      makeFlatVector<int64_t>({100, 120, 130, 350}, DECIMAL(5, 1));
  const std::vector<std::pair<std::string, RowVectorPtr>> inputs{
      {"constant",
       makeRowVector({left, makeConstant<int64_t>(100, 4, DECIMAL(5, 1))})},
      {"flat", makeRowVector({left, right})},
      {"dictionary",
       makeRowVector({
           wrapInDictionary(makeIndices({0, 1, 2, 3}), left),
           wrapInDictionary(makeIndices({0, 1, 2, 3}), right),
       })}};
  const auto nullable = makeRowVector({
      left,
      makeNullableFlatVector<int64_t>(
          {100, std::nullopt, 130, 350}, DECIMAL(5, 1)),
  });

  for (const auto& comparison : cpuComparisons_) {
    SCOPED_TRACE(comparison.name);
    const auto expression = fmt::format("{}(c0, c1)", comparison.name);
    const auto expected = makeFlatVector<bool>(comparison.shortResults);
    for (const auto& [encoding, input] : inputs) {
      SCOPED_TRACE(encoding);
      assertGpuComparison(expression, input, expected);
    }
    assertGpuComparison(
        expression,
        nullable,
        makeNullableFlatVector<bool>({
            comparison.shortResults[0],
            std::nullopt,
            comparison.shortResults[2],
            comparison.shortResults[3],
        }));
  }
}

TEST_F(DecimalComparisonTest, cpuBatchInputs) {
  for (const vector_size_t size : {70, 140}) {
    SCOPED_TRACE(size);
    VectorPtr left;
    if (size == 70) {
      left = makeFlatVector<int64_t>(
          size,
          [](auto row) { return row % 2 == 0 ? 1000 : 3000; },
          nullptr,
          DECIMAL(6, 2));
    } else {
      left = wrapInDictionary(
          makeIndices(size, [](auto row) { return row % 2; }),
          makeFlatVector<int64_t>({1000, 3000}, DECIMAL(6, 2)));
    }
    const auto input = makeRowVector({
        left,
        makeFlatVector<int64_t>(
            size,
            [](auto row) { return row % 2 == 0 ? 100 : 130; },
            nullptr,
            DECIMAL(5, 1)),
    });
    for (const auto& comparison : cpuComparisons_) {
      SCOPED_TRACE(comparison.name);
      assertGpuComparison(
          fmt::format("{}(c0, c1)", comparison.name),
          input,
          makeFlatVector<bool>(size, [&](auto row) {
            return comparison.alternatingResults[row % 2];
          }));
    }
  }
}

TEST_F(DecimalComparisonTest, cpuSelectedRows) {
  // The CPU tests exclude the first 13 of 130 rows using SelectivityVector.
  // The GPU fixture evaluates whole batches, so select the same rows in a
  // query and retain row IDs to check each selected row's result.
  const auto input = makeRowVector({
      makeFlatVector<int64_t>(
          130,
          [](auto row) { return row % 2 == 0 ? 1000 : 3000; },
          nullptr,
          DECIMAL(6, 2)),
      makeFlatVector<int64_t>(
          130,
          [](auto row) { return row % 2 == 0 ? 100 : 130; },
          nullptr,
          DECIMAL(5, 1)),
      makeFlatIdentityVector<int64_t>(130),
  });
  for (const auto& comparison : cpuComparisons_) {
    SCOPED_TRACE(comparison.name);
    const auto plan = exec::test::PlanBuilder()
                          .values({input})
                          .filter("greaterthanorequal(c2, 13)")
                          .project({
                              "c2",
                              fmt::format("{}(c0, c1)", comparison.name),
                          })
                          .planNode();
    const auto expected = makeRowVector({
        makeFlatVector<int64_t>(117, [](auto row) { return row + 13; }),
        makeFlatVector<bool>(
            117,
            [&](auto row) {
              return comparison.alternatingResults[(row + 13) % 2];
            }),
    });
    exec::test::AssertQueryBuilder(plan).assertResults(expected);
  }
}

TEST_F(DecimalComparisonTest, cpuWideScaleFallback) {
  // These CPU cases need 39 digits when DECIMAL(6, 1) is aligned to scale 34.
  // Preserve the CPU types and expected results while checking GPU rejection.
  const auto left = makeFlatVector<int128_t>(
      {10'000'000'000'000'000,
       12'000'000'000'000'000,
       13'000'000'000'000'000,
       35'000'000'000'000'000},
      DECIMAL(38, 34));
  const auto right =
      makeFlatVector<int64_t>({100000, 120000, 130000, 350000}, DECIMAL(6, 1));
  const std::vector<std::pair<std::string, RowVectorPtr>> inputs{
      {"constant",
       makeRowVector({left, makeConstant<int64_t>(100000, 4, DECIMAL(6, 1))})},
      {"flat", makeRowVector({left, right})},
      {"dictionary",
       makeRowVector({
           wrapInDictionary(makeIndices({0, 1, 2, 3}), left),
           wrapInDictionary(makeIndices({0, 1, 2, 3}), right),
       })},
      {"nullable",
       makeRowVector({
           left,
           makeNullableFlatVector<int64_t>(
               {100000, std::nullopt, 130000, 350000}, DECIMAL(6, 1)),
       })}};

  auto& config = CudfConfig::getInstance();
  const auto previousFallback = config.allowCpuFallback;
  unregisterCudf();
  config.allowCpuFallback = true;
  registerCudf();
  SCOPE_EXIT {
    unregisterCudf();
    config.allowCpuFallback = previousFallback;
    registerCudf();
  };

  for (const auto& comparison : cpuComparisons_) {
    SCOPED_TRACE(comparison.name);
    const auto expression = fmt::format("{}(c0, c1)", comparison.name);
    for (const auto& [encoding, input] : inputs) {
      SCOPED_TRACE(encoding);
      const auto typed = makeTypedExpr(expression, input->rowType());
      ASSERT_FALSE(FunctionExpression::canEvaluate(typed));
      ASSERT_FALSE(ASTExpression::canEvaluate(typed));
      ASSERT_FALSE(canExprRunOnGpu(typed, execCtx_.queryCtx(), pool_.get()));
      ASSERT_EQ(
          createCudfFunction(comparison.name, typed, pool_.get()), nullptr);
      const auto plan = exec::test::PlanBuilder()
                            .values({input})
                            .project({expression})
                            .planNode();
      const auto expected = makeRowVector({makeFlatVector<bool>(
          4,
          [&](auto /*row*/) { return comparison.longResult; },
          [&](auto row) { return encoding == "nullable" && row == 1; })});
      exec::test::AssertQueryBuilder(plan).assertResults(expected);
    }
  }
}

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
        columnType,
        {-unit,
         -9 * unit / 10,
         -unit / 2,
         0,
         unit / 2,
         9 * unit / 10,
         unit,
         std::nullopt})});
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
