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

#include "gtest/gtest.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/ExprCompiler.h"
#include "velox/expression/FieldReference.h"
#include "velox/expression/RegisterSpecialForm.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using functions::test::FunctionBaseTest;

class SpecialFormExprOptimizationTest : public FunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    exec::registerFunctionCallToSpecialForms();
    functions::prestosql::registerAllScalarFunctions();
    memory::MemoryManager::testingSetInstance({});
  }

  const core::QueryConfig config_{{}};
};

TEST_F(SpecialFormExprOptimizationTest, andConjunct) {
  // true AND false -> false
  auto compiledExpr =
      compileExpression("(1 = 1) AND (2 != 2)", ROW({}))->exprs().front();
  ASSERT_TRUE(compiledExpr->isConstant());
  ASSERT_EQ(compiledExpr->type(), BOOLEAN());
  assertEqualVectors(
      vectorMaker_.constantVector<bool>({false}),
      compiledExpr->as<ConstantExpr>()->value());

  // true AND NULL -> NULL
  compiledExpr =
      compileExpression("(2 != 3) AND cast(NULL as BOOLEAN)", ROW({}))
          ->exprs()
          .front();
  ASSERT_TRUE(compiledExpr->isConstant());
  ASSERT_EQ(compiledExpr->type(), BOOLEAN());
  ASSERT_TRUE(compiledExpr->as<ConstantExpr>()->value()->isNullAt(0));

  // true AND true -> true
  compiledExpr =
      compileExpression("(2 != 3) AND (4 + 5 = 9)", ROW({}))->exprs().front();
  ASSERT_TRUE(compiledExpr->isConstant());
  ASSERT_EQ(compiledExpr->type(), BOOLEAN());
  assertEqualVectors(
      vectorMaker_.constantVector<bool>({true}),
      compiledExpr->as<ConstantExpr>()->value());

  // true AND true AND a -> a
  auto compiledExprs =
      compileExpression("(2 != 3) AND (1 = 1) AND a", ROW({"a"}, {BOOLEAN()}))
          ->exprs();
  ASSERT_EQ(compiledExprs.size(), 1);
  compiledExpr = compiledExprs.front();
  ASSERT_EQ(compiledExpr->name(), "a");
  ASSERT_EQ(compiledExpr->type(), BOOLEAN());
}

TEST_F(SpecialFormExprOptimizationTest, orConjunct) {
  // false OR true -> true
  auto compiledExpr =
      compileExpression("(1 != 1) OR (2 = 2)", ROW({}))->exprs().front();
  ASSERT_TRUE(compiledExpr->isConstant());
  ASSERT_TRUE(compiledExpr->type()->isBoolean());
  assertEqualVectors(
      vectorMaker_.constantVector<bool>({true}),
      compiledExpr->as<ConstantExpr>()->value());

  // false OR NULL -> NULL
  compiledExpr = compileExpression("(2 = 3) OR cast(NULL as BOOLEAN)", ROW({}))
                     ->exprs()
                     .front();
  ASSERT_TRUE(compiledExpr->isConstant());
  ASSERT_TRUE(compiledExpr->type()->isBoolean());
  ASSERT_TRUE(compiledExpr->as<ConstantExpr>()->value()->isNullAt(0));

  // false OR false -> false
  compiledExpr =
      compileExpression("(2 = 3) OR (4 + 4 = 9)", ROW({}))->exprs().front();
  ASSERT_TRUE(compiledExpr->isConstant());
  ASSERT_TRUE(compiledExpr->type()->isBoolean());
  assertEqualVectors(
      vectorMaker_.constantVector<bool>({false}),
      compiledExpr->as<ConstantExpr>()->value());

  // false OR false OR a -> a
  auto compiledExprs =
      compileExpression("(2 = 3) OR (1 != 1) OR a", ROW({"a"}, {BOOLEAN()}))
          ->exprs();
  ASSERT_EQ(compiledExprs.size(), 1);
  compiledExpr = compiledExprs.front();
  ASSERT_TRUE(compiledExpr->type()->isBoolean());
  ASSERT_EQ(compiledExpr->name(), "a");
}

TEST_F(SpecialFormExprOptimizationTest, coalesce) {
  // coalesce((NULL AND true), NULL) -> NULL
  auto compiledExpr =
      compileExpression(
          "coalesce((NULL AND true), cast(NULL as BOOLEAN))", ROW({}))
          ->exprs()
          .front();
  ASSERT_TRUE(compiledExpr->isConstant());
  ASSERT_TRUE(compiledExpr->type()->isUnKnown());
  ASSERT_TRUE(compiledExpr->as<ConstantExpr>()->value()->isNullAt(0));

  // coalesce((NULL OR false), NULL, true) -> true
  compiledExpr =
      compileExpression(
          "coalesce((NULL OR false), cast(NULL as BOOLEAN), true)", ROW({}))
          ->exprs()
          .front();
  ASSERT_TRUE(compiledExpr->isConstant());
  ASSERT_TRUE(compiledExpr->type()->isBoolean());
  assertEqualVectors(
      vectorMaker_.constantVector<bool>({true}),
      compiledExpr->as<ConstantExpr>()->value());

  // coalesce((NULL OR false), (NULL AND true), a) -> a
  compiledExpr = compileExpression(
                     "coalesce((NULL OR false), (NULL AND true), a)",
                     ROW({"a"}, {BOOLEAN()}))
                     ->exprs()
                     .front();
  ASSERT_EQ(compiledExpr->inputs().size(), 1);
  auto input = compiledExpr->inputs().front();
  ASSERT_TRUE(input->type()->isBoolean());
  ASSERT_EQ(input->name(), "a");
}
