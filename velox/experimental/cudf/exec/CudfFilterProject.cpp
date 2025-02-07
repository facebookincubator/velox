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
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/type/Type.h"
#include "velox/vector/ConstantVector.h"

#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

namespace facebook::velox::cudf_velox {

template <TypeKind kind>
cudf::ast::literal make_scalar_and_literal(
    VectorPtr vector,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars) {
  using T = typename KindToFlatVector<kind>::WrapperType;
  if constexpr (cudf::is_fixed_width<T>()) {
    VELOX_CHECK(vector->isConstantEncoding());
    auto constVector = vector->as<ConstantVector<T>>();
    T value = constVector->valueAt(0);
    // store scalar and use its reference in the literal
    scalars.emplace_back(std::make_unique<cudf::numeric_scalar<T>>(value));
    return cudf::ast::literal{
        *static_cast<cudf::numeric_scalar<T>*>(scalars.back().get())};
  } else if (kind == TypeKind::VARCHAR) {
    VELOX_CHECK(vector->isConstantEncoding());
    auto constVector = vector->as<ConstantVector<StringView>>();
    auto value = constVector->valueAt(0);
    scalars.emplace_back(std::make_unique<cudf::string_scalar>(value));
    return cudf::ast::literal{
        *static_cast<cudf::string_scalar*>(scalars.back().get())};
  } else {
    // TODO for non-numeric types too.
    VELOX_CHECK(false, "Not implemented");
  }
}

cudf::ast::literal createLiteral(
    VectorPtr vector,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars) {
  const auto kind = vector->typeKind();
  return VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      make_scalar_and_literal, kind, std::move(vector), scalars);
}

// Create tree from Expr
cudf::ast::expression const& create_ast_tree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    tree& t,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema) {
  using op = cudf::ast::ast_operator;
  using operation = cudf::ast::operation;
  auto& name = expr->name();
  std::cout << "name: " << name << std::endl;
  if (name == "literal") {
    velox::exec::ConstantExpr* c =
        dynamic_cast<velox::exec::ConstantExpr*>(expr.get());
    VELOX_CHECK_NOT_NULL(c, "literal expression should be ConstantExpr");
    auto value = c->value();
    // convert to cudf scalar
    auto lit = createLiteral(value, scalars);
    return t.push(std::move(lit));
  } else if (name == "and") {
    auto len = expr->inputs().size();
    VELOX_CHECK_EQ(len, 2);
    auto const& op1 =
        create_ast_tree(expr->inputs()[0], t, scalars, inputRowSchema);
    auto const& op2 =
        create_ast_tree(expr->inputs()[1], t, scalars, inputRowSchema);
    return t.push(operation{op::NULL_LOGICAL_AND, op1, op2});
  } else if (name == "or") {
    auto len = expr->inputs().size();
    VELOX_CHECK_EQ(len, 2);
    auto const& op1 =
        create_ast_tree(expr->inputs()[0], t, scalars, inputRowSchema);
    auto const& op2 =
        create_ast_tree(expr->inputs()[1], t, scalars, inputRowSchema);
    return t.push(operation{op::NULL_LOGICAL_OR, op1, op2});
  } else if (name == "eq") {
    auto len = expr->inputs().size();
    VELOX_CHECK_EQ(len, 2);
    auto const& op1 =
        create_ast_tree(expr->inputs()[0], t, scalars, inputRowSchema);
    auto const& op2 =
        create_ast_tree(expr->inputs()[1], t, scalars, inputRowSchema);
    return t.push(operation{op::EQUAL, op1, op2});
  } else if (name == "neq") {
    auto len = expr->inputs().size();
    VELOX_CHECK_EQ(len, 2);
    auto const& op1 =
        create_ast_tree(expr->inputs()[0], t, scalars, inputRowSchema);
    auto const& op2 =
        create_ast_tree(expr->inputs()[1], t, scalars, inputRowSchema);
    return t.push(operation{op::NOT_EQUAL, op1, op2});
  } else if (name == "plus") {
    auto len = expr->inputs().size();
    VELOX_CHECK_EQ(len, 2);
    auto const& op1 =
        create_ast_tree(expr->inputs()[0], t, scalars, inputRowSchema);
    auto const& op2 =
        create_ast_tree(expr->inputs()[1], t, scalars, inputRowSchema);
    return t.push(operation{op::ADD, op1, op2});
  } else if (name == "minus") {
    auto len = expr->inputs().size();
    VELOX_CHECK_EQ(len, 2);
    auto const& op1 =
        create_ast_tree(expr->inputs()[0], t, scalars, inputRowSchema);
    auto const& op2 =
        create_ast_tree(expr->inputs()[1], t, scalars, inputRowSchema);
    return t.push(operation{op::SUB, op1, op2});
  } else if (name == "multiply") {
    auto len = expr->inputs().size();
    VELOX_CHECK_EQ(len, 2);
    auto const& op1 =
        create_ast_tree(expr->inputs()[0], t, scalars, inputRowSchema);
    auto const& op2 =
        create_ast_tree(expr->inputs()[1], t, scalars, inputRowSchema);
    return t.push(operation{op::MUL, op1, op2});
  } else if (name == "divide") {
    auto len = expr->inputs().size();
    VELOX_CHECK_EQ(len, 2);
    auto const& op1 =
        create_ast_tree(expr->inputs()[0], t, scalars, inputRowSchema);
    auto const& op2 =
        create_ast_tree(expr->inputs()[1], t, scalars, inputRowSchema);
    return t.push(operation{op::DIV, op1, op2});
  } else if (name == "cast") {
    auto len = expr->inputs().size();
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 =
        create_ast_tree(expr->inputs()[0], t, scalars, inputRowSchema);
    if (expr->type()->kind() == TypeKind::INTEGER) {
      // No int32 cast in cudf ast
      return t.push(operation{op::CAST_TO_INT64, op1});
    } else if (expr->type()->kind() == TypeKind::BIGINT) {
      return t.push(operation{op::CAST_TO_INT64, op1});
    } else if (expr->type()->kind() == TypeKind::DOUBLE) {
      return t.push(operation{op::CAST_TO_FLOAT64, op1});
    } else {
      VELOX_CHECK(false, "Unsupported type for cast operation");
    }
  } else {
    // Field? (not all are fields. Need better way to confirm Field)
    auto column_index = inputRowSchema->getChildIdx(name);
    // std::cout << "Column index: " << column_index << std::endl;
    return t.push(cudf::ast::column_reference(column_index));
  }
}

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
  for (auto expr : info.exprs->exprs()) {
    tree t;
    create_ast_tree(expr, t, scalars_, inputType);
    projectAst_.emplace_back(std::move(t));
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
    return nullptr;
  }
  auto cudf_input = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudf_input);
  auto input_table = cudf_input->release();
  auto cudf_table_view = input_table->view();

  std::vector<std::unique_ptr<cudf::column>> columns;
  for (auto& tree : projectAst_) {
    auto col = cudf::compute_column(
        cudf_table_view,
        tree.back(),
        cudf::get_default_stream(),
        cudf::get_current_device_resource_ref());
    columns.emplace_back(std::move(col));
  }
  std::vector<std::unique_ptr<cudf::column>> output_columns(
      outputType_->size());
  // computed resultProjections
  for (int i = 0; i < resultProjections_.size(); i++) {
    output_columns[resultProjections_[i].outputChannel] = std::move(columns[i]);
  }
  // identityProjections (input to output copy)
  for (auto& identity : identityProjections_) {
    output_columns[identity.outputChannel] = std::make_unique<cudf::column>(
        cudf_table_view.column(identity.inputChannel));
  }

  auto output_table = std::make_unique<cudf::table>(std::move(output_columns));
  auto const size = output_table->num_rows();
  if (cudfDebugEnabled()) {
    std::cout << "cudfProject Output: " << size << " rows " << std::endl;
    std::cout << "cudfProject Output: " << output_table->num_columns()
              << " columns " << std::endl;
  }
  input_.reset();
  if (output_table->num_columns() == 0 or size == 0) {
    return nullptr;
  }
  return std::make_shared<CudfVector>(
      pool(), outputType_, size, std::move(output_table));
}

bool CudfFilterProject::allInputProcessed() {
  if (!input_) {
    return true;
  }
  return false;
}

bool CudfFilterProject::isFinished() {
  return noMoreInput_ && allInputProcessed();
}

void CudfFilterProject::initialize() {
  Operator::initialize();
  // all of the initialization is done in ctor
}

} // namespace facebook::velox::cudf_velox
