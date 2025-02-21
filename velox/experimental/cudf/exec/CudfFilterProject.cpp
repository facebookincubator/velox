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
#include "velox/experimental/cudf/exec/ExpressionEvaluator.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FieldReference.h"
#include "velox/type/Type.h"
#include "velox/vector/ConstantVector.h"

#include <cudf/datetime.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include <sstream>

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
      hasFilter_(filter != nullptr),
      project_(project),
      filter_(filter) {
  // If Filter is present, ctor fails.
  VELOX_CHECK(!hasFilter_, "Filter not supported yet");
  resultProjections_ = *(info.resultProjections);
  identityProjections_ = std::move(identityProjections);
  const auto& inputType = project_->sources()[0]->outputType();

  // convert to AST
  if (cudfDebugEnabled()) {
    int i = 0;
    for (auto expr : info.exprs->exprs()) {
      std::cout << "expr[" << i++ << "] " << expr->toString() << std::endl;
      debug_print_tree(expr);
    }
  }
  for (auto expr : info.exprs->exprs()) {
    cudf::ast::tree tree;
    create_ast_tree(expr, tree, scalars_, inputType, precompute_instructions_);
    // If tree has only field reference, then it is a custom op or column
    // reference. so we need to move it to identityProjections_
    projectAst_.emplace_back(std::move(tree));
  }
}

void CudfFilterProject::addInput(RowVectorPtr input) {
  input_ = std::move(input);
}

RowVectorPtr CudfFilterProject::getOutput() {
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

  // Usage of the function
  addPrecomputedColumns(
      input_table_columns, precompute_instructions_, scalars_, stream);

  auto input_table =
      std::make_unique<cudf::table>(std::move(input_table_columns));
  auto cudf_table_view = input_table->view();
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (auto& tree : projectAst_) {
    if (auto col_ref_ptr =
            dynamic_cast<cudf::ast::column_reference const*>(&tree.back())) {
      auto col = std::make_unique<cudf::column>(
          cudf_table_view.column(col_ref_ptr->get_column_index()),
          stream,
          cudf::get_current_device_resource_ref());
      columns.emplace_back(std::move(col));
    } else {
      auto col = cudf::compute_column(
          cudf_table_view,
          tree.back(),
          stream,
          cudf::get_current_device_resource_ref());
      columns.emplace_back(std::move(col));
    }
  }

  // Rearrange columns to match outputType_
  std::vector<std::unique_ptr<cudf::column>> output_columns(
      outputType_->size());
  // computed resultProjections
  for (int i = 0; i < resultProjections_.size(); i++) {
    output_columns[resultProjections_[i].outputChannel] = std::move(columns[i]);
  }
  // identityProjections (input to output copy)
  for (auto const& identity : identityProjections_) {
    output_columns[identity.outputChannel] = std::make_unique<cudf::column>(
        cudf_table_view.column(identity.inputChannel),
        stream,
        cudf::get_current_device_resource_ref());
  }

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

bool CudfFilterProject::allInputProcessed() {
  return !input_;
}

bool CudfFilterProject::isFinished() {
  return noMoreInput_ && allInputProcessed();
}

} // namespace facebook::velox::cudf_velox
