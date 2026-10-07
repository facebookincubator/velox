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
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
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

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::test_utils;

namespace {

// Selection coverage for the Spark functions cuDF supports. The Presto and
// type-system cases live in tests/ExpressionEvaluatorSelectionTest.cpp, which
// builds without Spark; this target is only added when
// VELOX_ENABLE_SPARK_FUNCTIONS is on.
class CudfSparkExpressionSelectionTest : public ::testing::Test {
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
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
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

// Disabled because this test segfaults in CI while building the typed
// not use cudf code.
TEST_F(
    CudfSparkExpressionSelectionTest,
    DISABLED_functionTopLevelWithNestedAst) {
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

TEST_F(CudfSparkExpressionSelectionTest, signatureAllowsColumnArgsStartswith) {
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

TEST_F(CudfSparkExpressionSelectionTest, signatureAllowsColumnArgsContains) {
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

TEST_F(CudfSparkExpressionSelectionTest, signatureAllowsColumnArgsEndswith) {
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

TEST_F(CudfSparkExpressionSelectionTest, signatureArityAndConstantsSubstring) {
  // Spark substring registers integer positions and lengths, where the
  // Presto-compatible substr covered in the common test takes bigint.
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
}

TEST_F(CudfSparkExpressionSelectionTest, signatureArrayAccessGet) {
  auto arrayRowType = ROW({
      {"arr", ARRAY(INTEGER())},
      {"idx_bigint", BIGINT()},
      {"idx_integer", INTEGER()},
  });

  auto bigintExpr = parseAndInferTypedExpr(
      "get(arr, idx_bigint)", arrayRowType, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(bigintExpr, queryCtx_.get(), pool_.get()));

  auto integerExpr = parseAndInferTypedExpr(
      "get(arr, idx_integer)", arrayRowType, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(integerExpr, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfSparkExpressionSelectionTest, signatureGetSmallIntegralIndices) {
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

TEST_F(CudfSparkExpressionSelectionTest, signatureVarargsHashWithSeed) {
  // canExprRunOnGpu reads this setting directly; no driver re-registration is
  // needed.
  CudfConfig::getInstance().allowCpuFallback = true;
  SCOPE_EXIT {
    CudfConfig::getInstance().allowCpuFallback = false;
  };

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

} // namespace
