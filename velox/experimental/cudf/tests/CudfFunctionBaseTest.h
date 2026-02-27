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
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

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
    auto stream = cudf::get_default_stream();
    auto cudfTable =
        velox::cudf_velox::with_arrow::toCudfTable(input, pool_.get(), stream);
    auto filterEvaluator =
        createCudfExpression({exprSet.exprs()[0]}, input->rowType());
    auto ownedColumns = cudfTable->release();
    std::vector<cudf::column_view> inputViews;
    inputViews.reserve(ownedColumns.size());
    for (auto& col : ownedColumns) {
      inputViews.push_back(col->view());
    }
    auto filterColumn = filterEvaluator->eval(
        inputViews, stream, cudf::get_current_device_resource_ref());
    auto filterColumnView = asView(filterColumn);
    cudf::table_view resultTable({filterColumnView});
    auto result = velox::cudf_velox::with_arrow::toVeloxColumn(
        resultTable, pool_.get(), "", stream);
    return result->childAt(0);
  }
};

} // namespace facebook::velox::cudf_velox
