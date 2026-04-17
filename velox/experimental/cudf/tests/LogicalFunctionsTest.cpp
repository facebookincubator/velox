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

#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace {

// Exercises the `not` / `is_null` / `isnotnull` CudfFunction classes
// introduced by this PR. The AST evaluator natively supports all three
// operators, and its default priority (100) beats the Function evaluator
// (50), so under normal configuration these CudfFunction classes are dormant
// for primitive types. This fixture re-registers the AST evaluator at
// priority 0, which drops AST to a last-resort fallback and forces
// expressions that FunctionExpression can handle through the new CudfFunction
// classes. Tests use primitive types only so they don't depend on the
// timestamp or decimal PRs.
class CudfLogicalFunctionsTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();

    // Overwrite the AST registration with priority 0 so any expression that
    // FunctionExpression can also evaluate routes through the Function path.
    cudf_velox::registerCudfExpressionEvaluator(
        cudf_velox::kAstEvaluatorName,
        /*priority=*/0,
        [](std::shared_ptr<velox::exec::Expr> expr) {
          return cudf_velox::ASTExpression::canEvaluate(expr);
        },
        [](std::shared_ptr<velox::exec::Expr> expr, const RowTypePtr& row) {
          return std::make_shared<cudf_velox::ASTExpression>(
              std::move(expr), row);
        },
        /*overwrite=*/true);
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  void runProject(
      const std::vector<RowVectorPtr>& input,
      const std::string& projection,
      const std::string& sql) {
    createDuckDbTable(input);
    auto plan =
        PlanBuilder().values(input).project({projection}).planNode();
    assertQuery(plan, sql);
  }
};

// NotFunction: negation of a boolean column — the base column-only path.
TEST_F(CudfLogicalFunctionsTest, notColumn) {
  auto data = makeRowVector(
      {"a"}, {makeFlatVector<bool>({true, false, true, false})});
  runProject({data}, "NOT a AS r", "SELECT NOT a AS r FROM tmp");
}

// NotFunction: negation of a comparison — `NotFunction` wraps the comparison
// result, which itself comes from a nested FunctionExpression path.
TEST_F(CudfLogicalFunctionsTest, notComparison) {
  auto data = makeRowVector(
      {"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  runProject(
      {data}, "NOT (c0 = 3) AS r", "SELECT NOT (c0 = 3) AS r FROM tmp");
}

// NotFunction: null row — cudf::unary_operation with NOT should propagate
// null through untouched.
TEST_F(CudfLogicalFunctionsTest, notWithNullRows) {
  auto data = makeRowVector(
      {"a"},
      {makeNullableFlatVector<bool>(
          {true, false, std::nullopt, true, std::nullopt})});
  runProject({data}, "NOT a AS r", "SELECT NOT a AS r FROM tmp");
}

// IsNullFunction: nullable INTEGER column — mix of nulls and non-nulls.
TEST_F(CudfLogicalFunctionsTest, isNullInteger) {
  auto data = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<int32_t>(
          {1, std::nullopt, 3, std::nullopt, 5})});
  runProject({data}, "c0 IS NULL AS r", "SELECT c0 IS NULL AS r FROM tmp");
}

// IsNullFunction: nullable VARCHAR column — exercises string-typed input.
TEST_F(CudfLogicalFunctionsTest, isNullVarchar) {
  auto data = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<std::string>(
          {"x", std::nullopt, "y", std::nullopt, ""})});
  runProject({data}, "c0 IS NULL AS r", "SELECT c0 IS NULL AS r FROM tmp");
}

// IsNullFunction: column with no nulls — result is all false.
TEST_F(CudfLogicalFunctionsTest, isNullNoNulls) {
  auto data = makeRowVector(
      {"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  runProject({data}, "c0 IS NULL AS r", "SELECT c0 IS NULL AS r FROM tmp");
}

// IsNotNullFunction: nullable INTEGER column — inverse of IS NULL.
TEST_F(CudfLogicalFunctionsTest, isNotNullInteger) {
  auto data = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<int32_t>(
          {1, std::nullopt, 3, std::nullopt, 5})});
  runProject(
      {data}, "c0 IS NOT NULL AS r", "SELECT c0 IS NOT NULL AS r FROM tmp");
}

// IsNotNullFunction: nullable VARCHAR column.
TEST_F(CudfLogicalFunctionsTest, isNotNullVarchar) {
  auto data = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<std::string>(
          {"x", std::nullopt, "y", std::nullopt, ""})});
  runProject(
      {data}, "c0 IS NOT NULL AS r", "SELECT c0 IS NOT NULL AS r FROM tmp");
}

// Composition: NOT wrapping IS NULL — exercises NotFunction operating on the
// column produced by IsNullFunction.
TEST_F(CudfLogicalFunctionsTest, notOfIsNull) {
  auto data = makeRowVector(
      {"c0"},
      {makeNullableFlatVector<int32_t>(
          {1, std::nullopt, 3, std::nullopt, 5})});
  runProject(
      {data},
      "NOT (c0 IS NULL) AS r",
      "SELECT NOT (c0 IS NULL) AS r FROM tmp");
}

} // namespace
