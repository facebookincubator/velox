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
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/expression/JitExpression.h"

namespace facebook::velox::cudf_velox {

JitExpression::JitExpression(
    std::shared_ptr<velox::exec::Expr> expr,
    const RowTypePtr& inputRowSchema)
    : expr_{expr, inputRowSchema} {}

void JitExpression::close() {
  expr_.close();
}

ColumnOrView JitExpression::eval(
    std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool finalize) {
  auto precomputedColumns = precomputeSubexpressions(
      inputTableColumns,
      expr_.precomputeInstructions_,
      expr_.scalars_,
      expr_.inputRowSchema_,
      stream);

  // Make table_view from input columns and precomputed columns
  std::vector<cudf::column_view> allColumnViews;
  allColumnViews.reserve(inputTableColumns.size() + precomputedColumns.size());

  for (const auto& col : inputTableColumns) {
    allColumnViews.push_back(col->view());
  }

  for (auto& precomputedCol : precomputedColumns) {
    allColumnViews.push_back(asView(precomputedCol));
  }

  cudf::table_view astInputTableView(allColumnViews);

  auto result = [&]() -> ColumnOrView {
    if (auto colRefPtr = dynamic_cast<cudf::ast::column_reference const*>(
            &expr_.cudfTree_.back())) {
      auto columnIndex = colRefPtr->get_column_index();
      if (columnIndex < inputTableColumns.size()) {
        return inputTableColumns[columnIndex]->view();
      } else {
        // Referencing a precomputed column return as it is (view or owned)
        return std::move(
            precomputedColumns[columnIndex - inputTableColumns.size()]);
      }
    } else {
      return cudf::compute_column_jit(
          astInputTableView, expr_.cudfTree_.back(), stream, mr);
    }
  }();
  if (finalize) {
    const auto requestedType =
        cudf::data_type(cudf_velox::veloxToCudfTypeId(expr_.expr_->type()));
    auto resultView = asView(result);
    if (resultView.type() != requestedType) {
      result = cudf::cast(resultView, requestedType, stream, mr);
    }
  }
  return result;
}

bool JitExpression::canEvaluate(std::shared_ptr<velox::exec::Expr> expr) {
  return ASTExpression::canEvaluate(expr);
}

void registerJitEvaluator(int priority) {
  registerCudfExpressionEvaluator(
      kJitEvaluatorName,
      priority,
      [](std::shared_ptr<velox::exec::Expr> expr) {
        return JitExpression::canEvaluate(expr);
      },
      [](std::shared_ptr<velox::exec::Expr> expr, const RowTypePtr& row) {
        return std::make_shared<JitExpression>(std::move(expr), row);
      },
      /*overwrite=*/false);
}

} // namespace facebook::velox::cudf_velox
