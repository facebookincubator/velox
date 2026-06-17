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

#pragma once

#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"

#include "velox/expression/ExprOptimizer.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

#include <string>

namespace facebook::velox::cudf_velox {

class CudfFunctionBaseTest : public velox::functions::test::FunctionBaseTest {
 protected:
  void assertExpressionMatchesCpu(
      const std::string& expr,
      const RowVectorPtr& input,
      const RowTypePtr& rowType) {
    auto exprSet = compileExpression(expr, rowType);
    auto expected =
        functions::test::FunctionBaseTest::evaluate(*exprSet, input);
    // Build the typed expression directly from the SQL and the declared row
    // type rather than reparsing exprSet's SQL, which loses information for
    // computed-ROW dereferences and unnamed ROW fields.
    auto actual = evaluate(makeTypedExpr(expr, rowType), input);
    facebook::velox::test::assertEqualVectors(expected, actual);
  }

  VectorPtr evaluate(
      exec::ExprSet& exprSet,
      const RowVectorPtr& input,
      const std::optional<SelectivityVector>& rows = std::nullopt) override {
    VELOX_CHECK(!rows.has_value());
    // exec::Expr::toSql() cannot faithfully round-trip computed-ROW
    // dereferences or unnamed ROW fields, so optimize and evaluate the typed
    // expression directly instead.
    return evaluate(
        test_utils::optimizeTypedExpr(
            exprSet.expr(0)->toSql(),
            input->rowType(),
            execCtx_.queryCtx(),
            &execCtx_),
        input);
  }

  // Optimizes and evaluates a typed expression on the GPU, returning the result
  // column. Use this overload for expressions (e.g. dereferences of
  // row_constructor) that cannot survive an exec::Expr SQL round-trip.
  VectorPtr evaluate(
      const core::TypedExprPtr& expr,
      const RowVectorPtr& input) {
    auto stream = cudfGlobalStreamPool().get_stream();
    auto mr = get_output_mr();
    auto cudfTable = velox::cudf_velox::with_arrow::toCudfTable(
        input, pool_.get(), stream, mr);
    auto optimized =
        expression::optimize(expr, execCtx_.queryCtx(), execCtx_.pool());
    auto filterEvaluator =
        createCudfExpression(optimized, input->rowType(), pool_.get());
    auto ownedColumns = cudfTable->release();
    std::vector<cudf::column_view> inputViews;
    inputViews.reserve(ownedColumns.size());
    for (auto& col : ownedColumns) {
      inputViews.push_back(col->view());
    }
    auto filterColumn = filterEvaluator->eval(inputViews, stream, mr);
    auto filterColumnView = asView(filterColumn);
    cudf::table_view resultTable({filterColumnView});
    // Preserve logical Velox output types, e.g. VARBINARY, when converting
    // the cuDF result back through Arrow.
    auto outputType = ROW({"c0"}, {optimized->type()});
    auto result = velox::cudf_velox::with_arrow::toVeloxColumn(
        resultTable,
        pool_.get(),
        outputType,
        "",
        stream,
        cudf::get_current_device_resource_ref());
    result->setType(outputType);
    return result->childAt(0);
  }
};

} // namespace facebook::velox::cudf_velox
