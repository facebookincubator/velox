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
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/JitExpression.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"

#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/type/Type.h"

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <string>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::test_utils;

namespace {

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
    cudf_velox::registerPrestoFunctions("");
    cudf_velox::registerSparkFunctions("");
    rowType_ = ROW({
        {"a", BIGINT()},
        {"b", BIGINT()},
        {"c", INTEGER()},
        {"name", VARCHAR()},
        {"date", DATE()},
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
  SCOPE_EXIT {
    CudfConfig::getInstance().astExpressionEnabled = prevAst;
    CudfConfig::getInstance().jitExpressionEnabled = prevJit;
  };
  CudfConfig::getInstance().astExpressionEnabled = true;
  CudfConfig::getInstance().jitExpressionEnabled = true;
  auto expr =
      optimizeTypedExpr("a + c", rowType_, queryCtx_.get(), execCtx_.get());
  auto cudfExpr = createCudfExpression(expr, rowType_, pool_.get());
  auto* ast = dynamic_cast<ASTExpression*>(cudfExpr.get());
  auto* jit = dynamic_cast<JitExpression*>(cudfExpr.get());
  ASSERT_TRUE(ast != nullptr || jit != nullptr);
}

TEST_F(CudfExpressionSelectionTest, functionRoot) {
  auto expr = optimizeTypedExpr(
      "lower(name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  auto cudfExpr = createCudfExpression(expr, rowType_, pool_.get());
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
}

TEST_F(CudfExpressionSelectionTest, astTopLevelWithFunctionPrecompute) {
  auto prevAst = CudfConfig::getInstance().astExpressionEnabled;
  auto prevJit = CudfConfig::getInstance().jitExpressionEnabled;
  SCOPE_EXIT {
    CudfConfig::getInstance().astExpressionEnabled = prevAst;
    CudfConfig::getInstance().jitExpressionEnabled = prevJit;
  };
  CudfConfig::getInstance().astExpressionEnabled = true;
  CudfConfig::getInstance().jitExpressionEnabled = true;
  auto expr = optimizeTypedExpr(
      "(year(date) > 2020) AND (length(name) < 10)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  auto cudfExpr = createCudfExpression(expr, rowType_, pool_.get());
  auto* ast = dynamic_cast<ASTExpression*>(cudfExpr.get());
  auto* jit = dynamic_cast<JitExpression*>(cudfExpr.get());
  ASSERT_TRUE(ast != nullptr || jit != nullptr);
}

TEST_F(CudfExpressionSelectionTest, functionTopLevelWithNestedFunction) {
  auto expr = optimizeTypedExpr(
      "lower(substr(name, 1, 5))", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  auto cudfExpr = createCudfExpression(expr, rowType_, pool_.get());

  // Top level should be Function
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
}

TEST_F(
    CudfExpressionSelectionTest,
    signatureAllowsRowConstructorAndDereference) {
  auto row =
      parseAndInferTypedExpr("row_constructor(a, b)", rowType_, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(row, queryCtx_.get(), pool_.get()));

  auto firstField = parseAndInferTypedExpr(
      "row_constructor(a, b).c1", rowType_, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(firstField, queryCtx_.get(), pool_.get()));

  auto secondField = parseAndInferTypedExpr(
      "row_constructor(a, 1).c2", rowType_, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(secondField, queryCtx_.get(), pool_.get()));

  auto nullLiteralField = parseAndInferTypedExpr(
      "row_constructor(a, cast(null as bigint)).c2", rowType_, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(nullLiteralField, queryCtx_.get(), pool_.get()));

  auto leadingNullField = parseAndInferTypedExpr(
      "row_constructor(cast(null as bigint), b).c1", rowType_, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(leadingNullField, queryCtx_.get(), pool_.get()));

  auto nestedField = parseAndInferTypedExpr(
      "row_constructor(row_constructor(a, cast(null as bigint)), b).c1.c2",
      rowType_,
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(nestedField, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, nestedRowDereferenceUsesFunctionEvaluator) {
  auto prevAst = CudfConfig::getInstance().astExpressionEnabled;
  auto prevJit = CudfConfig::getInstance().jitExpressionEnabled;
  SCOPE_EXIT {
    CudfConfig::getInstance().astExpressionEnabled = prevAst;
    CudfConfig::getInstance().jitExpressionEnabled = prevJit;
  };
  CudfConfig::getInstance().astExpressionEnabled = true;
  CudfConfig::getInstance().jitExpressionEnabled = true;

  auto expr = parseAndInferTypedExpr(
      "row_constructor(row_constructor(a, b), cast(null as bigint)).c1.c1",
      rowType_,
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));

  auto cudfExpr = createCudfExpression(expr, rowType_, pool_.get());
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
}

TEST_F(
    CudfExpressionSelectionTest,
    signatureAllowsRowConstructorDereferenceByIndex) {
  auto unnamedRowType = ROW({{"", BIGINT()}, {"", BIGINT()}});
  core::TypedExprPtr expr = std::make_shared<core::DereferenceTypedExpr>(
      BIGINT(),
      std::make_shared<core::CallTypedExpr>(
          unnamedRowType,
          std::vector<core::TypedExprPtr>{
              std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "a"),
              std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "b"),
          },
          "row_constructor"),
      1);

  ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  ASSERT_NE(createCudfExpression(expr, rowType_, pool_.get()), nullptr);
}

TEST_F(
    CudfExpressionSelectionTest,
    signatureAllowsRowConstructorDereferenceByName) {
  auto namedRowType = ROW({{"left", BIGINT()}, {"right", BIGINT()}});
  core::TypedExprPtr expr = std::make_shared<core::FieldAccessTypedExpr>(
      BIGINT(),
      std::make_shared<core::CallTypedExpr>(
          namedRowType,
          std::vector<core::TypedExprPtr>{
              std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "a"),
              std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "b"),
          },
          "row_constructor"),
      "right");

  ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  ASSERT_NE(createCudfExpression(expr, rowType_, pool_.get()), nullptr);
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
  auto cudfExpr = createCudfExpression(expr, rowType_, pool_.get());
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
  ASSERT_TRUE(canExprRunOnGpu(ok, queryCtx_.get(), pool_.get()));

  // Bad: delimiter is not a constant
  auto bad = optimizeTypedExpr(
      "split(name, name, 3)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  ASSERT_FALSE(canExprRunOnGpu(bad, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, signatureAllowsColumnPatternLike) {
  // OK: pattern is a constant
  auto ok = optimizeTypedExpr(
      "like(name, '%abc%')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok, queryCtx_.get(), pool_.get()));

  // OK: pattern can also come from a column.
  auto okColumn = optimizeTypedExpr(
      "like(name, name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okColumn, queryCtx_.get(), pool_.get()));

  // OK: constant input still works when pattern comes from a column.
  auto okConstantInput = optimizeTypedExpr(
      "like('abc', name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okConstantInput, queryCtx_.get(), pool_.get()));

  // OK: constant null input should also remain on the cuDF path.
  auto okNullInput = optimizeTypedExpr(
      "like(cast(null as varchar), name)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okNullInput, queryCtx_.get(), pool_.get()));

  // OK: escape can be a constant too.
  auto okWithEscape = optimizeTypedExpr(
      "like(name, '%#_%', '#')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okWithEscape, queryCtx_.get(), pool_.get()));

  // OK: pattern column + constant escape is supported.
  auto okColumnWithEscape = optimizeTypedExpr(
      "like(name, name, '#')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(
      canExprRunOnGpu(okColumnWithEscape, queryCtx_.get(), pool_.get()));

  // OK: constant input + pattern column + constant escape is supported.
  auto okConstantInputWithEscape = optimizeTypedExpr(
      "like('a_c', name, '#')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(
      canExprRunOnGpu(okConstantInputWithEscape, queryCtx_.get(), pool_.get()));

  // OK: constant null input + pattern column + constant escape is supported.
  auto okNullInputWithEscape = optimizeTypedExpr(
      "like(cast(null as varchar), name, '#')",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(
      canExprRunOnGpu(okNullInputWithEscape, queryCtx_.get(), pool_.get()));

  // OK: null constants should remain on the cuDF path.
  auto okNullPattern = optimizeTypedExpr(
      "like(name, cast(null as varchar))",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okNullPattern, queryCtx_.get(), pool_.get()));

  auto okNullEscape = optimizeTypedExpr(
      "like(name, '%#_%', cast(null as varchar))",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okNullEscape, queryCtx_.get(), pool_.get()));

  // Bad: escape is not a constant.
  auto badEscape = optimizeTypedExpr(
      "like(name, '%#_%', name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_FALSE(canExprRunOnGpu(badEscape, queryCtx_.get(), pool_.get()));

  // Bad: escape column is still unsupported when pattern comes from a column.
  auto badColumnEscape = optimizeTypedExpr(
      "like(name, name, name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_FALSE(canExprRunOnGpu(badColumnEscape, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, signatureAllowsColumnArgsStartswith) {
  // OK: pattern is a constant
  auto ok = optimizeTypedExpr(
      "startswith(name, 'ab')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok, queryCtx_.get(), pool_.get()));

  // OK: null pattern is still a constant and should remain on the cuDF path.
  auto okNull = optimizeTypedExpr(
      "startswith(name, cast(null as varchar))",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okNull, queryCtx_.get(), pool_.get()));

  // OK: pattern can also come from a column.
  auto okColumn = optimizeTypedExpr(
      "startswith(name, name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okColumn, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, signatureAllowsColumnArgsContains) {
  // OK: pattern is a constant
  auto ok = optimizeTypedExpr(
      "contains(name, 'ab')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok, queryCtx_.get(), pool_.get()));

  // OK: the input can also be a constant.
  auto okConstantInput = optimizeTypedExpr(
      "contains('ab', name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okConstantInput, queryCtx_.get(), pool_.get()));

  // OK: null pattern is still a constant and should remain on the cuDF path.
  auto okNull = optimizeTypedExpr(
      "contains(name, cast(null as varchar))",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okNull, queryCtx_.get(), pool_.get()));

  // OK: pattern can also come from a column.
  auto okColumn = optimizeTypedExpr(
      "contains(name, name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okColumn, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, signatureAllowsColumnArgsEndswith) {
  // OK: pattern is a constant
  auto ok = optimizeTypedExpr(
      "endswith(name, 'ab')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok, queryCtx_.get(), pool_.get()));

  // OK: the input can also be a constant.
  auto okConstantInput = optimizeTypedExpr(
      "endswith('ab', name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okConstantInput, queryCtx_.get(), pool_.get()));

  // OK: null pattern is still a constant and should remain on the cuDF path.
  auto okNull = optimizeTypedExpr(
      "endswith(name, cast(null as varchar))",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okNull, queryCtx_.get(), pool_.get()));

  // OK: pattern can also come from a column.
  auto okColumn = optimizeTypedExpr(
      "endswith(name, name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okColumn, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, signatureArityAndConstantsSubstr) {
  // The default parser keeps integer literals as BIGINT, which exercises the
  // existing Presto-compatible `substr` candidate. Spark-specific coverage is
  // below, using INTEGER literals or INTEGER columns.

  // OK: 2-arg substr with constant start
  auto ok2 = optimizeTypedExpr(
      "substr(name, 1)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok2, queryCtx_.get(), pool_.get()));

  // OK: 3-arg substr with constant start and length
  auto ok3 = optimizeTypedExpr(
      "substr(name, 1, 5)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok3, queryCtx_.get(), pool_.get()));

  // OK: Spark substring registers integer positions and lengths.
  parse::ParseOptions sparkLiteralOptions;
  sparkLiteralOptions.parseIntegerAsBigint = false;
  auto okSparkLiteralArgs = optimizeTypedExpr(
      "substring(name, 1, 5)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get(),
      sparkLiteralOptions);
  ASSERT_TRUE(
      canExprRunOnGpu(okSparkLiteralArgs, queryCtx_.get(), pool_.get()));

  // OK: Spark substring supports integer start and length columns. This also
  // verifies that the cuDF `substr` function name routes to Spark semantics
  // when Spark functions are registered.
  auto okStartColumn = optimizeTypedExpr(
      "substr(name, c)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okStartColumn, queryCtx_.get(), pool_.get()));

  auto okStartAndLengthColumns = optimizeTypedExpr(
      "substring(name, c, c)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(
      canExprRunOnGpu(okStartAndLengthColumns, queryCtx_.get(), pool_.get()));

  // Bad: Spark substr accepts integer positions, not bigint positions.
  auto badBigintStart = optimizeTypedExpr(
      "substr(name, a)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_FALSE(canExprRunOnGpu(badBigintStart, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, signatureArrayAccess) {
  auto arrayRowType = ROW({
      {"arr", ARRAY(INTEGER())},
      {"idx_bigint", BIGINT()},
      {"idx_integer", INTEGER()},
  });

  for (const auto& functionName : {"element_at", "subscript", "get"}) {
    SCOPED_TRACE(functionName);

    auto bigintExpr = parseAndInferTypedExpr(
        std::string(functionName) + "(arr, idx_bigint)",
        arrayRowType,
        execCtx_.get());
    ASSERT_TRUE(canExprRunOnGpu(bigintExpr, queryCtx_.get(), pool_.get()));

    auto integerExpr = parseAndInferTypedExpr(
        std::string(functionName) + "(arr, idx_integer)",
        arrayRowType,
        execCtx_.get());
    ASSERT_TRUE(canExprRunOnGpu(integerExpr, queryCtx_.get(), pool_.get()));
  }
}

TEST_F(CudfExpressionSelectionTest, signatureSparkGetSmallIntegralIndices) {
  auto arrayRowType = ROW({
      {"arr", ARRAY(INTEGER())},
      {"idx_tinyint", TINYINT()},
      {"idx_smallint", SMALLINT()},
  });

  auto tinyintExpr = parseAndInferTypedExpr(
      "get(arr, idx_tinyint)", arrayRowType, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(tinyintExpr, queryCtx_.get(), pool_.get()));

  auto smallintExpr = parseAndInferTypedExpr(
      "get(arr, idx_smallint)", arrayRowType, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(smallintExpr, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, signatureCastsInDivide) {
  // OK: numeric args are castable to double
  auto ok = optimizeTypedExpr(
      "divide(a, b)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok, queryCtx_.get(), pool_.get()));
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
  ASSERT_FALSE(canExprRunOnGpu(multiCol, queryCtx_.get(), pool_.get()));

  // Single-column hash_with_seed is supported (no column combining needed).
  auto singleCol = optimizeTypedExpr(
      "hash_with_seed(42, a)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  ASSERT_TRUE(canExprRunOnGpu(singleCol, queryCtx_.get(), pool_.get()));

  // Bad: first arg must be constant seed
  try {
    auto bad = optimizeTypedExpr(
        "hash_with_seed(c, b)",
        rowType_,
        queryCtx_.get(),
        execCtx_.get(),
        {.parseIntegerAsBigint = false, .functionPrefix = ""});
    // If compilation succeeds, the compiled check must fail.
    ASSERT_FALSE(canExprRunOnGpu(bad, queryCtx_.get(), pool_.get()));
  } catch (const VeloxUserError&) {
    // Treat compile-time validation failure as unsupported.
    SUCCEED();
  }
}

TEST_F(CudfExpressionSelectionTest, signatureTypeVariableCoalesce) {
  // OK: same type BIGINT
  auto ok1 = optimizeTypedExpr(
      "coalesce(a, b)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok1, queryCtx_.get(), pool_.get()));

  // OK: VARCHAR with literal
  auto ok2 = optimizeTypedExpr(
      "coalesce(name, 'x')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok2, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, signatureTypeVariableSwitchIf) {
  // OK: boolean + same type BIGINT
  auto ok1 = optimizeTypedExpr(
      "if(a > 0, a, b)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok1, queryCtx_.get(), pool_.get()));

  // Constant conditions are folded before cuDF expression selection.
  auto folded = optimizeTypedExpr(
      "if(true, a, b)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(folded, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, DISABLED_castAndTryCast) {
  // TODO (dm): This is required for passing of castAndTryCast test but breaks
  // others. This is because ASTExpr agrees to support bad casts. remove after
  // ASTExpr checks cast types
  // CudfConfig::getInstance().astExpressionEnabled = false;

  // OK: cast bigint -> double (supported by cuDF)
  auto okCast = optimizeTypedExpr(
      "cast(a AS double)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okCast, queryCtx_.get(), pool_.get()));

  // OK: try_cast bigint -> double (supported by cuDF)
  auto okTryCast = optimizeTypedExpr(
      "try_cast(a AS double)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okTryCast, queryCtx_.get(), pool_.get()));

  // BAD: cast boolean -> date (expected unsupported by cuDF)
  auto badCast = optimizeTypedExpr(
      "cast(length(name) < 10 AS date)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_FALSE(canExprRunOnGpu(badCast, queryCtx_.get(), pool_.get()));
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
// createCudfExpression tests — verify the pure-function compilation API
// and expression optimization.
// ---------------------------------------------------------------------------

TEST_F(CudfExpressionSelectionTest, compilerPureAstNoBoundaries) {
  // A simple arithmetic expression handled entirely by AST should compile
  // successfully.
  auto expr = parseAndInferTypedExpr("a + b", rowType_, execCtx_.get());
  auto result = createCudfExpression(expr, rowType_, pool_.get());
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

  auto expr = parseAndInferTypedExpr(
      "a + b > cardinality(names)", arrayType, execCtx_.get());
  auto result = createCudfExpression(expr, arrayType, pool_.get());
  ASSERT_NE(result, nullptr);
}

TEST_F(CudfExpressionSelectionTest, compilerOptimizesConstantExpr) {
  // expression::optimize folds constant subtrees; "a + (1 + 2)" optimizes to
  // "a + 3".
  auto expr = parseAndInferTypedExpr("a + (1 + 2)", rowType_, execCtx_.get());

  const auto optimized =
      expression::optimize(expr, queryCtx_.get(), pool_.get());
  ASSERT_NE(optimized, nullptr);

  auto result = createCudfExpression(optimized, rowType_, pool_.get());
  ASSERT_NE(result, nullptr);

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
  auto expr = parseAndInferTypedExpr("a + b", rowType_, execCtx_.get());
  auto result = createCudfExpression(expr, rowType_, pool_.get());
  ASSERT_NE(result, nullptr);
}

} // namespace
