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
#include "velox/expression/FieldReference.h"
#include "velox/type/Type.h"
#include "velox/vector/ConstantVector.h"

#include <cudf/datetime.hpp>
#include <cudf/strings/attributes.hpp>
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

using op = cudf::ast::ast_operator;
const std::map<std::string, op> binary_ops = {
    {"plus", op::ADD},
    {"minus", op::SUB},
    {"multiply", op::MUL},
    {"divide", op::DIV},
    {"eq", op::EQUAL},
    {"neq", op::NOT_EQUAL},
    {"and", op::NULL_LOGICAL_AND},
    {"or", op::NULL_LOGICAL_OR}};

void debug_print_tree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    int indent = 0) {
  std::cout << std::string(indent, ' ') << expr->name() << std::endl;
  for (auto& input : expr->inputs()) {
    debug_print_tree(input, indent + 2);
  }
}

// Create tree from Expr
// and collect precompute instructions for non-ast operations
cudf::ast::expression const& create_ast_tree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    std::vector<std::tuple<int, std::string, int>>& precompute_instructions) {
  using op = cudf::ast::ast_operator;
  using operation = cudf::ast::operation;
  auto& name = expr->name();

  if (name == "literal") {
    velox::exec::ConstantExpr* c =
        dynamic_cast<velox::exec::ConstantExpr*>(expr.get());
    VELOX_CHECK_NOT_NULL(c, "literal expression should be ConstantExpr");
    auto value = c->value();
    // convert to cudf scalar
    return tree.push(createLiteral(value, scalars));
  } else if (binary_ops.find(name) != binary_ops.end()) {
    auto len = expr->inputs().size();
    VELOX_CHECK_EQ(len, 2);
    auto const& op1 = create_ast_tree(
        expr->inputs()[0],
        tree,
        scalars,
        inputRowSchema,
        precompute_instructions);
    auto const& op2 = create_ast_tree(
        expr->inputs()[1],
        tree,
        scalars,
        inputRowSchema,
        precompute_instructions);
    return tree.push(operation{binary_ops.at(name), op1, op2});
  } else if (name == "cast") {
    auto len = expr->inputs().size();
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 = create_ast_tree(
        expr->inputs()[0],
        tree,
        scalars,
        inputRowSchema,
        precompute_instructions);
    if (expr->type()->kind() == TypeKind::INTEGER) {
      // No int32 cast in cudf ast
      return tree.push(operation{op::CAST_TO_INT64, op1});
    } else if (expr->type()->kind() == TypeKind::BIGINT) {
      return tree.push(operation{op::CAST_TO_INT64, op1});
    } else if (expr->type()->kind() == TypeKind::DOUBLE) {
      return tree.push(operation{op::CAST_TO_FLOAT64, op1});
    } else {
      VELOX_CHECK(false, "Unsupported type for cast operation");
    }
  } else if (name == "switch") {
    auto len = expr->inputs().size();
    VELOX_CHECK_EQ(len, 3);
    // check if input[1], input[2] are literals 1 and 0.
    // then simplify as typecast bool to int
    velox::exec::ConstantExpr* c1 =
        dynamic_cast<velox::exec::ConstantExpr*>(expr->inputs()[1].get());
    velox::exec::ConstantExpr* c2 =
        dynamic_cast<velox::exec::ConstantExpr*>(expr->inputs()[2].get());
    if (c1 and c2 and c1->toString() == "1:BIGINT" and
        c2->toString() == "0:BIGINT") {
      auto const& op1 = create_ast_tree(
          expr->inputs()[0],
          tree,
          scalars,
          inputRowSchema,
          precompute_instructions);
      return tree.push(operation{op::CAST_TO_INT64, op1});
    } else {
      std::cerr << "switch subexpr: " << expr->toString() << std::endl;
      VELOX_CHECK(false, "Unsupported switch complex operation");
    }
  } else if (name == "year") {
    // ensure expr->inputs()[0] is a field
    auto fieldExpr = std::dynamic_pointer_cast<velox::exec::FieldReference>(
        expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");
    auto dependent_column_index =
        inputRowSchema->getChildIdx(fieldExpr->name());
    auto new_column_index =
        inputRowSchema->size() + precompute_instructions.size();
    // add this index and precompute instruciton to a datastructure
    precompute_instructions.emplace_back(
        dependent_column_index, "year", new_column_index);
    // This custom op should be added to input columns.
    // cast to big int
    auto const& col_ref =
        tree.push(cudf::ast::column_reference(new_column_index));
    return tree.push(operation{op::CAST_TO_INT64, col_ref});
  } else {
    auto fieldExpr =
        std::dynamic_pointer_cast<velox::exec::FieldReference>(expr);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field, name: " + name);
    // Field? (not all are fields. Need better way to confirm Field)
    auto column_index = inputRowSchema->getChildIdx(name);
    return tree.push(cudf::ast::column_reference(column_index));
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
  auto input_table_columns = cudf_input->release()->release();
  // add ast unsupported precomputed columns to input_table
  // Works only directly on column in input table, not intermediate columns
  for (auto& instruction : precompute_instructions_) {
    auto [dependent_column_index, ins_name, new_column_index] = instruction;
    if (ins_name == "year") {
      // TODO use extract_datetime_component after cudf 24.12 update
      auto new_column = cudf::datetime::extract_year(
          input_table_columns[dependent_column_index]->view(),
          cudf::get_default_stream(),
          cudf::get_current_device_resource_ref());
      input_table_columns.emplace_back(std::move(new_column));
    } else if (ins_name == "length") {
      auto new_column = cudf::strings::count_characters(
          input_table_columns[dependent_column_index]->view(),
          // cudf::get_default_stream(), // TODO add this after stream API
          // update
          cudf::get_current_device_resource_ref());
      input_table_columns.emplace_back(std::move(new_column));
    } else {
      VELOX_CHECK(false, "Unsupported precompute operation " + ins_name);
    }
  }

  auto input_table =
      std::make_unique<cudf::table>(std::move(input_table_columns));
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
  return !input_;
}

bool CudfFilterProject::isFinished() {
  return noMoreInput_ && allInputProcessed();
}

} // namespace facebook::velox::cudf_velox
