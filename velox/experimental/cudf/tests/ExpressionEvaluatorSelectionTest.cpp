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

#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
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

class CudfExpressionSelectionTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
    queryCtx_ = core::QueryCtx::create();
    execCtx_ = std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get());
    facebook::velox::functions::prestosql::registerAllScalarFunctions();
    cudf_velox::registerCudf();
    rowType_ = ROW({
        {"a", BIGINT()},
        {"b", BIGINT()},
        {"name", VARCHAR()},
        {"date", TIMESTAMP()},
    });

    parse::registerTypeResolver();
  }

  void TearDown() override {
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
  auto expr = compileExecExpr("a + b", rowType_, execCtx_.get());
  auto cudfExpr = createCudfExpression(expr, rowType_);
  auto* ast = dynamic_cast<ASTExpression*>(cudfExpr.get());
  ASSERT_NE(ast, nullptr);
  ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, functionRoot) {
  auto expr = compileExecExpr("lower(name)", rowType_, execCtx_.get());
  auto cudfExpr = createCudfExpression(expr, rowType_);
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
  ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, astTopLevelWithFunctionPrecompute) {
  // AST handles AND and comparisons; functions (year/length) are precomputed.
  auto expr = compileExecExpr(
      "(year(date) > 2020) AND (length(name) < 10)", rowType_, execCtx_.get());
  auto cudfExpr = createCudfExpression(expr, rowType_);
  auto* ast = dynamic_cast<ASTExpression*>(cudfExpr.get());
  ASSERT_NE(ast, nullptr);

  // Shallow: AST root supported
  ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/false));
  // Deep: children (functions) supported via nested evaluators
  ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, functionTopLevelWithNestedFunction) {
  auto expr =
      compileExecExpr("lower(substr(name, 1, 5))", rowType_, execCtx_.get());
  auto cudfExpr = createCudfExpression(expr, rowType_);

  // Top level should be Function
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
  ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/true));
}

TEST_F(CudfExpressionSelectionTest, functionTopLevelWithNestedAst) {
  facebook::velox::functions::sparksql::registerFunctions();

  auto expr = compileExecExpr(
      "hash_with_seed(42, add(a, b))",
      rowType_,
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  auto cudfExpr = createCudfExpression(expr, rowType_);
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
  ASSERT_TRUE(canBeEvaluatedByCudf(expr, /*deep=*/true));
}

} // namespace
