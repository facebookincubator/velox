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
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/expression/Expr.h"

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>

#include <unordered_map>

namespace facebook::velox::cudf_velox {

namespace {

void debug_print_tree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    int indent = 0) {
  std::cout << std::string(indent, ' ') << expr->name() << std::endl;
  for (auto& input : expr->inputs()) {
    debug_print_tree(input, indent + 2);
  }
}
} // namespace

CudfFilterProject::CudfFilterProject(
    int32_t operatorId,
    velox::exec::DriverCtx* driverCtx,
    const velox::exec::FilterProject::Export& info,
    std::vector<velox::exec::IdentityProjection> identityProjections,
    const std::shared_ptr<const core::FilterNode>& filter,
    const std::shared_ptr<const core::ProjectNode>& project)
    : Operator(
          driverCtx,
          project ? project->outputType() : filter->outputType(),
          operatorId,
          project ? project->id() : filter->id(),
          "CudfFilterProject"),
      NvtxHelper(nvtx3::rgb{220, 20, 60}, operatorId), // Crimson
      hasFilter_(info.hasFilter),
      project_(project),
      filter_(filter) {
  resultProjections_ = *(info.resultProjections);
  identityProjections_ = std::move(identityProjections);
  const auto inputType = project_ ? project_->sources()[0]->outputType()
                                  : filter_->sources()[0]->outputType();

  // convert to AST
  if (cudfDebugEnabled()) {
    int i = 0;
    for (auto expr : info.exprs->exprs()) {
      std::cout << "expr[" << i++ << "] " << expr->toString() << std::endl;
      debug_print_tree(expr);
    }
  }
  std::vector<std::shared_ptr<velox::exec::Expr>> projectExprs;
  if (hasFilter_) {
    // First expr is Filter, rest are Project
    filterEvaluator_ = ExpressionEvaluator({info.exprs->exprs()[0]}, inputType);
    projectExprs = {info.exprs->exprs().begin() + 1, info.exprs->exprs().end()};
  }
  projectEvaluator_ = ExpressionEvaluator(
      hasFilter_ ? projectExprs : info.exprs->exprs(), inputType);
}

void CudfFilterProject::addInput(RowVectorPtr input) {
  input_ = std::move(input);
}

RowVectorPtr CudfFilterProject::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (allInputProcessed()) {
    return nullptr;
  }
  if (input_->size() == 0) {
    input_.reset();
    return nullptr;
  }

  auto cudf_input = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudf_input);
  auto stream = cudf_input->stream();
  auto input_table_columns = cudf_input->release()->release();

  if (hasFilter_) {
    filter(input_table_columns, stream);
  }
  auto output_columns = project(input_table_columns, stream);

  auto output_table = std::make_unique<cudf::table>(std::move(output_columns));
  stream.synchronize();
  auto const num_columns = output_table->num_columns();
  auto const size = output_table->num_rows();
  if (cudfDebugEnabled()) {
    std::cout << "cudfProject Output: " << size << " rows, " << num_columns
              << " columns " << std::endl;
  }

  auto cudf_output = std::make_shared<CudfVector>(
      input_->pool(), outputType_, size, std::move(output_table), stream);
  input_.reset();
  if (num_columns == 0 or size == 0) {
    return nullptr;
  }
  return cudf_output;
}

void CudfFilterProject::filter(
    std::vector<std::unique_ptr<cudf::column>>& input_table_columns,
    rmm::cuda_stream_view stream) {
  // Evaluate the Filter
  auto filter_columns = filterEvaluator_.compute(
      input_table_columns, stream, cudf::get_current_device_resource_ref());
  auto filter_column = filter_columns[0]->view();
  // is all true in filter_column
  auto is_all_true = cudf::reduce(
      filter_column,
      *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type(cudf::type_id::BOOL8),
      stream,
      cudf::get_current_device_resource_ref());
  using ScalarType = cudf::scalar_type_t<bool>;
  auto result = static_cast<ScalarType*>(is_all_true.get());
  // If filter is not all true, apply the filter
  if (!(result->is_valid() && result->value())) {
    // Apply the Filter
    auto filter_table =
        std::make_unique<cudf::table>(std::move(input_table_columns));
    auto filtered_table =
        cudf::apply_boolean_mask(*filter_table, filter_column, stream);
    input_table_columns = filtered_table->release();
  }
}

std::vector<std::unique_ptr<cudf::column>> CudfFilterProject::project(
    std::vector<std::unique_ptr<cudf::column>>& input_table_columns,
    rmm::cuda_stream_view stream) {
  auto columns = projectEvaluator_.compute(
      input_table_columns, stream, cudf::get_current_device_resource_ref());

  // Rearrange columns to match outputType_
  std::vector<std::unique_ptr<cudf::column>> output_columns(
      outputType_->size());
  // computed resultProjections
  for (int i = 0; i < resultProjections_.size(); i++) {
    VELOX_CHECK_NOT_NULL(columns[i]);
    output_columns[resultProjections_[i].outputChannel] = std::move(columns[i]);
  }

  // Count occurrences of each inputChannel, and move columns if they occur only
  // once
  std::unordered_map<column_index_t, int> inputChannelCount;
  for (const auto& identity : identityProjections_) {
    inputChannelCount[identity.inputChannel]++;
  }

  // identityProjections (input to output copy)
  for (auto const& identity : identityProjections_) {
    VELOX_CHECK_NOT_NULL(input_table_columns[identity.inputChannel]);
    if (inputChannelCount[identity.inputChannel] == 1) {
      // Move the column if it occurs only once
      output_columns[identity.outputChannel] =
          std::move(input_table_columns[identity.inputChannel]);
    } else {
      // Otherwise, copy the column and decrement the count
      output_columns[identity.outputChannel] = std::make_unique<cudf::column>(
          *input_table_columns[identity.inputChannel],
          stream,
          cudf::get_current_device_resource_ref());
    }
    VELOX_CHECK_GT(inputChannelCount[identity.inputChannel], 0);
    inputChannelCount[identity.inputChannel]--;
  }

  return output_columns;
}

bool CudfFilterProject::allInputProcessed() {
  return !input_;
}

bool CudfFilterProject::isFinished() {
  return noMoreInput_ && allInputProcessed();
}

} // namespace facebook::velox::cudf_velox
