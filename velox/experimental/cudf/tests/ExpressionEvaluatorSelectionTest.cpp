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
#include "velox/experimental/cudf/expression/CudfExpressionCompiler.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/JitExpression.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryCtx.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/type/Type.h"

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::test_utils;

namespace {

static CudfExprCtx makeExprCtx(
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  return CudfExprCtx{queryCtx, pool};
}

class CudfExpressionSelectionTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    facebook::velox::functions::sparksql::registerFunctions();
    facebook::velox::functions::prestosql::registerAllScalarFunctions();
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool("", false);
    queryCtx_ = core::QueryCtx::create();
    execCtx_ = std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get());
    cudf_velox::registerCudf();
    cudf_velox::registerSparkFunctions("");
    rowType_ = ROW({
        {"a", BIGINT()},
        {"b", BIGINT()},
        {"c", INTEGER()},
        {"name", VARCHAR()},
        {"date", TIMESTAMP()},
        {"c", INTEGER()},
    });

    parse::registerTypeResolver();
  }

  void TearDown() override {
    cudf_velox::unregisterFunctions();
    cudf_velox::unregisterCudf();
    execCtx_.reset();
    queryCtx_.reset();
    pool_.reset();
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<core::ExecCtx> execCtx_;
  RowTypePtr rowType_;
};

TEST_F(CudfExpressionSelectionTest, astRoot) {
  auto prevAst = CudfConfig::getInstance().astExpressionEnabled;
  auto prevJit = CudfConfig::getInstance().jitExpressionEnabled;
  CudfConfig::getInstance().astExpressionEnabled = true;
  CudfConfig::getInstance().jitExpressionEnabled = true;
  auto expr =
      optimizeTypedExpr("a + c", rowType_, queryCtx_.get(), execCtx_.get());
    auto cudfExpr =
      createCudfExpression(expr, rowType_, makeExprCtx(queryCtx_.get(), pool_.get()));
  auto* ast = dynamic_cast<ASTExpression*>(cudfExpr.get());
  auto* jit = dynamic_cast<JitExpression*>(cudfExpr.get());
  ASSERT_TRUE(ast != nullptr || jit != nullptr);
  CudfConfig::getInstance().astExpressionEnabled = prevAst;
  CudfConfig::getInstance().jitExpressionEnabled = prevJit;
}

TEST_F(CudfExpressionSelectionTest, functionRoot) {
  auto expr = optimizeTypedExpr(
      "lower(name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/false));
    auto cudfExpr =
      createCudfExpression(expr, rowType_, makeExprCtx(queryCtx_.get(), pool_.get()));
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
}

TEST_F(CudfExpressionSelectionTest, astTopLevelWithFunctionPrecompute) {
  auto prevAst = CudfConfig::getInstance().astExpressionEnabled;
  auto prevJit = CudfConfig::getInstance().jitExpressionEnabled;
  CudfConfig::getInstance().astExpressionEnabled = true;
  CudfConfig::getInstance().jitExpressionEnabled = true;
  auto expr = optimizeTypedExpr(
      "(year(date) > 2020) AND (length(name) < 10)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/false));
  auto cudfExpr =
      createCudfExpression(expr, rowType_, makeExprCtx(queryCtx_.get(), pool_.get()));
  auto* ast = dynamic_cast<ASTExpression*>(cudfExpr.get());
  auto* jit = dynamic_cast<JitExpression*>(cudfExpr.get());
  ASSERT_TRUE(ast != nullptr || jit != nullptr);
  CudfConfig::getInstance().astExpressionEnabled = prevAst;
  CudfConfig::getInstance().jitExpressionEnabled = prevJit;
}

TEST_F(CudfExpressionSelectionTest, functionTopLevelWithNestedFunction) {
  auto expr = optimizeTypedExpr(
      "lower(substr(name, 1, 5))",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/false));
  auto cudfExpr =
      createCudfExpression(expr, rowType_, makeExprCtx(queryCtx_.get(), pool_.get()));

  // Top level should be Function
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
}

// Disabled because this test segfaults in CI while building the typed
// not use cudf code.
TEST_F(CudfExpressionSelectionTest, DISABLED_functionTopLevelWithNestedAst) {
  auto expr = optimizeTypedExpr(
      "hash_with_seed(42, add(a, b))",
      rowType_,
      queryCtx_.get(),
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  auto cudfExpr =
      createCudfExpression(expr, rowType_, makeExprCtx(queryCtx_.get(), pool_.get()));
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
}

// Disabled because this test segfaults in CI while building the typed
// not use cudf code.
TEST_F(
    CudfExpressionSelectionTest,
    DISABLED_signatureEnforcesConstantArgsSplit) {
  // OK: delimiter and limit are constants
  auto ok = optimizeTypedExpr(
      "split(name, ',', 3)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  ASSERT_TRUE(canBeEvaluatedByCudf(ok, /*deep=*/true));

  // Bad: delimiter is not a constant
  auto bad = optimizeTypedExpr(
      "split(name, name, 3)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  ASSERT_FALSE(canBeEvaluatedByCudf(bad, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, signatureAllowsColumnPatternLike) {
  // OK: pattern is a constant
  auto ok = optimizeTypedExpr(
      "like(name, '%abc%')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok, /*deep=*/true));

  // OK: pattern can also come from a column.
  auto okColumn = optimizeTypedExpr(
      "like(name, name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okColumn, /*deep=*/true));

  // OK: constant input still works when pattern comes from a column.
  auto okConstantInput = optimizeTypedExpr(
      "like('abc', name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okConstantInput, /*deep=*/true));

  // OK: constant null input should also remain on the cuDF path.
  auto okNullInput = optimizeTypedExpr(
      "like(cast(null as varchar), name)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okNullInput, /*deep=*/true));

  // OK: escape can be a constant too.
  auto okWithEscape = optimizeTypedExpr(
      "like(name, '%#_%', '#')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okWithEscape, /*deep=*/true));

  // OK: pattern column + constant escape is supported.
  auto okColumnWithEscape = optimizeTypedExpr(
      "like(name, name, '#')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okColumnWithEscape, /*deep=*/true));

  // OK: constant input + pattern column + constant escape is supported.
  auto okConstantInputWithEscape = optimizeTypedExpr(
      "like('a_c', name, '#')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okConstantInputWithEscape, /*deep=*/true));

  // OK: constant null input + pattern column + constant escape is supported.
  auto okNullInputWithEscape = optimizeTypedExpr(
      "like(cast(null as varchar), name, '#')",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okNullInputWithEscape, /*deep=*/true));

  // OK: null constants should remain on the cuDF path.
  auto okNullPattern = optimizeTypedExpr(
      "like(name, cast(null as varchar))",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okNullPattern, /*deep=*/true));

  auto okNullEscape = optimizeTypedExpr(
      "like(name, '%#_%', cast(null as varchar))",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okNullEscape, /*deep=*/true));

  // Bad: escape is not a constant.
  auto badEscape = optimizeTypedExpr(
      "like(name, '%#_%', name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_FALSE(canBeEvaluatedByCudf(badEscape, /*deep=*/true));

  // Bad: escape column is still unsupported when pattern comes from a column.
  auto badColumnEscape = optimizeTypedExpr(
      "like(name, name, name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_FALSE(canBeEvaluatedByCudf(badColumnEscape, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, signatureAllowsColumnArgsStartswith) {
  // OK: pattern is a constant
  auto ok = optimizeTypedExpr("startswith(name, 'ab')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok, /*deep=*/true));

  // OK: null pattern is still a constant and should remain on the cuDF path.
  auto okNull = optimizeTypedExpr(
      "startswith(name, cast(null as varchar))", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okNull, /*deep=*/true));

  // OK: pattern can also come from a column.
  auto okColumn =
      optimizeTypedExpr("startswith(name, name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okColumn, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, signatureAllowsColumnArgsContains) {
  // OK: pattern is a constant
  auto ok = optimizeTypedExpr("contains(name, 'ab')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok, /*deep=*/true));

  // OK: the input can also be a constant.
  auto okConstantInput =
      optimizeTypedExpr("contains('ab', name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okConstantInput, /*deep=*/true));

  // OK: null pattern is still a constant and should remain on the cuDF path.
  auto okNull = optimizeTypedExpr(
      "contains(name, cast(null as varchar))", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okNull, /*deep=*/true));

  // OK: pattern can also come from a column.
  auto okColumn =
      optimizeTypedExpr("contains(name, name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okColumn, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, signatureAllowsColumnArgsEndswith) {
  // OK: pattern is a constant
  auto ok = optimizeTypedExpr("endswith(name, 'ab')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok, /*deep=*/true));

  // OK: the input can also be a constant.
  auto okConstantInput =
      optimizeTypedExpr("endswith('ab', name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okConstantInput, /*deep=*/true));

  // OK: null pattern is still a constant and should remain on the cuDF path.
  auto okNull = optimizeTypedExpr(
      "endswith(name, cast(null as varchar))", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okNull, /*deep=*/true));

  // OK: pattern can also come from a column.
  auto okColumn =
      optimizeTypedExpr("endswith(name, name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okColumn, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, signatureArityAndConstantsSubstr) {
  // OK: 2-arg substr with constant start
  auto ok2 = optimizeTypedExpr(
      "substr(name, 1)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok2, /*deep=*/true));

  // OK: 3-arg substr with constant start and length
  auto ok3 = optimizeTypedExpr(
      "substr(name, 1, 5)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok3, /*deep=*/true));

  // Bad: start must be constant
  auto badConst = optimizeTypedExpr(
      "substr(name, a)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_FALSE(canBeEvaluatedByCudf(badConst, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, signatureCastsInDivide) {
  // OK: numeric args are castable to double
  auto ok = optimizeTypedExpr(
      "divide(a, b)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, signatureVarargsHashWithSeed) {
  facebook::velox::functions::sparksql::registerFunctions();

  // TODO: Assert TRUE after https://github.com/rapidsai/cudf/issues/21720.
  // Multi-column hash_with_seed cannot be evaluated by cudf because cudf's
  // murmurhash3_x86_32 combines columns via hash_combine(h(col0, seed),
  // h(col1, seed)), while Spark hashes iteratively: h(col1, h(col0, seed)).
  // The cudf API only accepts a scalar seed, so per-row seeding is not
  // possible without a custom CUDA kernel.
  auto multiCol = optimizeTypedExpr(
      "hash_with_seed(42, a, b)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  ASSERT_FALSE(canBeEvaluatedByCudf(multiCol, /*deep=*/true));

  // Single-column hash_with_seed is supported (no column combining needed).
  auto singleCol = optimizeTypedExpr(
      "hash_with_seed(42, a)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  ASSERT_TRUE(canBeEvaluatedByCudf(singleCol, /*deep=*/true));

  // Bad: first arg must be constant seed
  try {
    auto bad = optimizeTypedExpr(
        "hash_with_seed(c, b)",
        rowType_,
        queryCtx_.get(),
        execCtx_.get(),
        {.parseIntegerAsBigint = false, .functionPrefix = ""});
    // If compilation succeeds, the compiled check must fail.
    ASSERT_FALSE(canBeEvaluatedByCudf(bad, /*deep=*/true));
  } catch (const VeloxUserError&) {
    // Treat compile-time validation failure as unsupported.
    SUCCEED();
  }
}

TEST_F(CudfExpressionSelectionTest, signatureTypeVariableCoalesce) {
  // OK: same type BIGINT
  auto ok1 = optimizeTypedExpr(
      "coalesce(a, b)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok1, /*deep=*/true));

  // OK: VARCHAR with literal
  auto ok2 = optimizeTypedExpr(
      "coalesce(name, 'x')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok2, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, signatureTypeVariableSwitchIf) {
  // OK: boolean + same type BIGINT
  auto ok1 = optimizeTypedExpr(
      "if(true, a, b)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(ok1, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, DISABLED_castAndTryCast) {
  // TODO (dm): This is required for passing of castAndTryCast test but breaks
  // others. This is because ASTExpr agrees to support bad casts. remove after
  // ASTExpr checks cast types
  // CudfConfig::getInstance().astExpressionEnabled = false;

  // OK: cast bigint -> double (supported by cuDF)
  auto okCast = optimizeTypedExpr(
      "cast(a AS double)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okCast, /*deep=*/true));

  // OK: try_cast bigint -> double (supported by cuDF)
  auto okTryCast = optimizeTypedExpr(
      "try_cast(a AS double)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canBeEvaluatedByCudf(okTryCast, /*deep=*/true));

  // BAD: cast boolean -> date (expected unsupported by cuDF)
  auto badCast = optimizeTypedExpr(
      "cast(length(name) < 10 AS date)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_FALSE(canBeEvaluatedByCudf(badCast, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, constantFoldingStringAllocatesOnCompile) {
  auto optimized = optimizeTypedExpr(
      "lower('ABCDEF')", rowType_, queryCtx_.get(), execCtx_.get());

  ASSERT_TRUE(optimized->isConstantKind());
  auto* constant = optimized->asUnchecked<core::ConstantTypedExpr>();
  auto value = constant->toConstantVector(execCtx_->pool());
  ASSERT_EQ(value->toString(0), "abcdef");
  if (constant->hasValueVector()) {
    ASSERT_EQ(constant->valueVector()->pool(), execCtx_->pool());
  }
}

// ---------------------------------------------------------------------------
// CudfExpressionCompiler tests — verify stateful compilation and
// expression optimization.
// ---------------------------------------------------------------------------

TEST_F(CudfExpressionSelectionTest, compilerPureAstNoBoundaries) {
  // A simple arithmetic expression handled entirely by AST should compile
  // successfully.
  CudfExpressionCompiler compiler(
      rowType_, makeExprCtx(queryCtx_.get(), pool_.get()));
  auto expr = parseAndInferTypedExpr("a + b", rowType_, execCtx_.get());
  auto result = compiler.compile(expr);
  ASSERT_NE(result, nullptr);
}

TEST_F(CudfExpressionSelectionTest, compilerFunctionBoundaryInAst) {
  // An expression like "a + b > cardinality(names)" where the top-level
  // comparison is AST but cardinality is only supported as a CudfFunction.
  // The compiler should handle mixed evaluators transparently.
  auto arrayType = ROW({
      {"a", BIGINT()},
      {"b", BIGINT()},
      {"names", ARRAY(VARCHAR())},
  });

  CudfExpressionCompiler compiler(
      arrayType, makeExprCtx(queryCtx_.get(), pool_.get()));
  auto expr = parseAndInferTypedExpr(
      "a + b > cardinality(names)", arrayType, execCtx_.get());
  auto result = compiler.compile(expr);
  ASSERT_NE(result, nullptr);
}

TEST_F(CudfExpressionSelectionTest, compilerMultipleCompileCalls) {
  // Multiple compile() calls on the same compiler should each produce a
  // valid expression.
  auto arrayType = ROW({
      {"a", BIGINT()},
      {"b", BIGINT()},
      {"names", ARRAY(VARCHAR())},
  });

  CudfExpressionCompiler compiler(
      arrayType, makeExprCtx(queryCtx_.get(), pool_.get()));

  // First expression with mixed evaluators.
  auto expr1 = parseAndInferTypedExpr(
      "a > cardinality(names)", arrayType, execCtx_.get());
  auto result1 = compiler.compile(expr1);
  ASSERT_NE(result1, nullptr);

  // Second expression — pure AST.
  auto expr2 = parseAndInferTypedExpr("a + b", arrayType, execCtx_.get());
  auto result2 = compiler.compile(expr2);
  ASSERT_NE(result2, nullptr);

  // Third expression with mixed evaluators again.
  auto expr3 = parseAndInferTypedExpr(
      "b > cardinality(names)", arrayType, execCtx_.get());
  auto result3 = compiler.compile(expr3);
  ASSERT_NE(result3, nullptr);
}

TEST_F(CudfExpressionSelectionTest, compilerOptimizesConstantExpr) {
  // Verify that the compiler performs constant folding.
  // "a + (1 + 2)" should optimize to "a + 3".
  CudfExpressionCompiler compiler(
      rowType_, makeExprCtx(queryCtx_.get(), pool_.get()));
  auto expr = parseAndInferTypedExpr("a + (1 + 2)", rowType_, execCtx_.get());
  auto result = compiler.compile(expr);
  ASSERT_NE(result, nullptr);

  // After optimization, the optimizedExpr should differ from input
  // (constant folding turns (1+2) into 3).
  const auto& optimized = compiler.optimizedExpr();
  ASSERT_NE(optimized, nullptr);
  // The optimized tree should have a constant child for the folded value.
  // It should be "a + 3" which has one FieldAccess child and one Constant.
  bool hasConstant = false;
  for (const auto& child : optimized->inputs()) {
    if (child->isConstantKind()) {
      hasConstant = true;
    }
  }
  EXPECT_TRUE(hasConstant)
      << "Constant folding should produce a constant child in 'a + (1+2)'";
}

TEST_F(CudfExpressionSelectionTest, compilerSimpleExpressionCompiles) {
  CudfExpressionCompiler compiler(
      rowType_, makeExprCtx(queryCtx_.get(), pool_.get()));
  auto expr = parseAndInferTypedExpr("a + b", rowType_, execCtx_.get());
  auto result = compiler.compile(expr);
  ASSERT_NE(result, nullptr);
}

} // namespace
