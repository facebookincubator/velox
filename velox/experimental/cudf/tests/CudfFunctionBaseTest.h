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

#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox::cudf_velox {

class CudfFunctionBaseTest : public velox::functions::test::FunctionBaseTest {
 protected:
  VectorPtr evaluate(
      exec::ExprSet& exprSet,
      const RowVectorPtr& input,
      const std::optional<SelectivityVector>& rows = std::nullopt) override {
    exec::EvalCtx context(&execCtx_, &exprSet, input.get());

    VELOX_CHECK(!rows.has_value());
    auto stream = cudfGlobalStreamPool().get_stream();
    auto mr = get_output_mr();
    auto cudfTable = velox::cudf_velox::with_arrow::toCudfTable(
      input, pool_.get(), stream, mr);
    auto typedExpr = test_utils::optimizeTypedExpr(
      exprSet.expr(0)->toSql(), input->rowType(), execCtx_.queryCtx(), &execCtx_);
    auto filterEvaluator = createCudfExpression(
        typedExpr,
        input->rowType(),
        CudfExprCtx{execCtx_.queryCtx(), pool_.get()});
    auto ownedColumns = cudfTable->release();
    std::vector<cudf::column_view> inputViews;
    inputViews.reserve(ownedColumns.size());
    for (auto& col : ownedColumns) {
      inputViews.push_back(col->view());
    }
    auto filterColumn = filterEvaluator->eval(inputViews, stream, mr);
    auto filterColumnView = asView(filterColumn);
    std::vector<cudf::column_view> resultViews;
    resultViews.reserve(1);
    resultViews.emplace_back(filterColumnView);
    cudf::table_view resultTable(resultViews);
    auto result = velox::cudf_velox::with_arrow::toVeloxColumn(
      resultTable, pool_.get(), "", stream, mr);
    return result->childAt(0);
  }
};

} // namespace facebook::velox::cudf_velox
