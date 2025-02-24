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
#include "velox/experimental/cudf/exec/ExpressionEvaluator.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FieldReference.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ConstantVector.h"
#include "velox/vector/VectorTypeUtils.h"

#include <cudf/datetime.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include <sstream>

namespace facebook::velox::cudf_velox {
namespace {
template <TypeKind kind>
cudf::ast::literal make_scalar_and_literal(
    VectorPtr vector,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars) {
  using T = typename facebook::velox::KindToFlatVector<kind>::WrapperType;
  if constexpr (cudf::is_fixed_width<T>()) {
    VELOX_CHECK(vector->isConstantEncoding());
    auto constVector = vector->as<facebook::velox::ConstantVector<T>>();
    T value = constVector->valueAt(0);
    // store scalar and use its reference in the literal
    scalars.emplace_back(std::make_unique<cudf::numeric_scalar<T>>(value));
    return cudf::ast::literal{
        *static_cast<cudf::numeric_scalar<T>*>(scalars.back().get())};
  } else if (kind == TypeKind::VARCHAR) {
    VELOX_CHECK(vector->isConstantEncoding());
    auto constVector =
        vector->as<facebook::velox::ConstantVector<StringView>>();
    auto value = constVector->valueAt(0);
    std::string_view stringValue = static_cast<std::string_view>(value);
    scalars.emplace_back(std::make_unique<cudf::string_scalar>(stringValue));
    return cudf::ast::literal{
        *static_cast<cudf::string_scalar*>(scalars.back().get())};
  } else {
    // TODO for non-numeric types too.
    VELOX_FAIL("Not implemented");
  }
}

cudf::ast::literal createLiteral(
    VectorPtr vector,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars) {
  const auto kind = vector->typeKind();
  return VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      make_scalar_and_literal, kind, std::move(vector), scalars);
}
} // namespace

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
    VELOX_CHECK_EQ(expr->inputs().size(), 1);
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
    VELOX_CHECK_EQ(expr->inputs().size(), 1);
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
      VELOX_FAIL("Unsupported type for cast operation");
    }
  } else if (name == "switch") {
    VELOX_CHECK_EQ(expr->inputs().size(), 3);
    // check if input[1], input[2] are literals 1 and 0.
    // then simplify as typecast bool to int
    velox::exec::ConstantExpr* c1 =
        dynamic_cast<velox::exec::ConstantExpr*>(expr->inputs()[1].get());
    velox::exec::ConstantExpr* c2 =
        dynamic_cast<velox::exec::ConstantExpr*>(expr->inputs()[2].get());
    if (c1 and c1->toString() == "1:BIGINT" and c2 and
        c2->toString() == "0:BIGINT") {
      auto const& op1 = create_ast_tree(
          expr->inputs()[0],
          tree,
          scalars,
          inputRowSchema,
          precompute_instructions);
      return tree.push(operation{op::CAST_TO_INT64, op1});
    } else if (c2 and c2->toString() == "0:DOUBLE") {
      auto const& op1 = create_ast_tree(
          expr->inputs()[0],
          tree,
          scalars,
          inputRowSchema,
          precompute_instructions);
      auto const& op1d = tree.push(operation{op::CAST_TO_FLOAT64, op1});
      auto const& op2 = create_ast_tree(
          expr->inputs()[1],
          tree,
          scalars,
          inputRowSchema,
          precompute_instructions);
      return tree.push(operation{op::MUL, op1d, op2});
    } else {
      std::cerr << "switch subexpr: " << expr->toString() << std::endl;
      VELOX_FAIL("Unsupported switch complex operation");
    }
  } else if (name == "year") {
    VELOX_CHECK_EQ(expr->inputs().size(), 1);
    // ensure expr->inputs()[0] is a field
    auto fieldExpr = std::dynamic_pointer_cast<velox::exec::FieldReference>(
        expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");
    auto dependent_column_index =
        inputRowSchema->getChildIdx(fieldExpr->name());
    auto new_column_index =
        inputRowSchema->size() + precompute_instructions.size();
    // add this index and precompute instruction to a data structure
    precompute_instructions.emplace_back(
        dependent_column_index, "year", new_column_index);
    // This custom op should be added to input columns.
    // cast to big int
    auto const& col_ref =
        tree.push(cudf::ast::column_reference(new_column_index));
    return tree.push(operation{op::CAST_TO_INT64, col_ref});
  } else if (name == "length") {
    VELOX_CHECK_EQ(expr->inputs().size(), 1);
    // ensure expr->inputs()[0] is a field
    auto fieldExpr = std::dynamic_pointer_cast<velox::exec::FieldReference>(
        expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");
    auto dependent_column_index =
        inputRowSchema->getChildIdx(fieldExpr->name());
    auto new_column_index =
        inputRowSchema->size() + precompute_instructions.size();
    // add this index and precompute instruction to a data structure
    precompute_instructions.emplace_back(
        dependent_column_index, "length", new_column_index);
    // This custom op should be added to input columns.
    auto const& col_ref =
        tree.push(cudf::ast::column_reference(new_column_index));
    return tree.push(operation{op::CAST_TO_INT64, col_ref});
  } else if (name == "substr") {
    // add precompute instruction, special handling col_ref during ast
    // evaluation
    VELOX_CHECK_EQ(expr->inputs().size(), 3);
    auto fieldExpr = std::dynamic_pointer_cast<velox::exec::FieldReference>(
        expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");
    auto dependent_column_index =
        inputRowSchema->getChildIdx(fieldExpr->name());
    auto new_column_index =
        inputRowSchema->size() + precompute_instructions.size();
    // add this index and precompute instruction to a data structure
    velox::exec::ConstantExpr* c1 =
        dynamic_cast<velox::exec::ConstantExpr*>(expr->inputs()[1].get());
    velox::exec::ConstantExpr* c2 =
        dynamic_cast<velox::exec::ConstantExpr*>(expr->inputs()[2].get());
    std::string substr_expr =
        "substr " + c1->value()->toString(0) + " " + c2->value()->toString(0);
    precompute_instructions.emplace_back(
        dependent_column_index, substr_expr, new_column_index);
    // This custom op should be added to input columns.
    return tree.push(cudf::ast::column_reference(new_column_index));
  } else if (name == "like") {
    VELOX_CHECK_EQ(expr->inputs().size(), 2);
    auto fieldExpr = std::dynamic_pointer_cast<velox::exec::FieldReference>(
        expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");
    auto dependent_column_index =
        inputRowSchema->getChildIdx(fieldExpr->name());
    auto new_column_index =
        inputRowSchema->size() + precompute_instructions.size();
    auto literalExpr =
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(literalExpr, "Expression is not a literal");
    createLiteral(literalExpr->value(), scalars);
    std::string like_expr = "like " + std::to_string(scalars.size() - 1);
    std::cout << "like_expr: " << like_expr << std::endl;
    precompute_instructions.emplace_back(
        dependent_column_index, like_expr, new_column_index);
    return tree.push(cudf::ast::column_reference(new_column_index));
  } else if (
      auto fieldExpr =
          std::dynamic_pointer_cast<velox::exec::FieldReference>(expr)) {
    auto column_index = inputRowSchema->getChildIdx(name);
    VELOX_CHECK(column_index != -1, "Field not found, " + name);
    return tree.push(cudf::ast::column_reference(column_index));
  } else {
    VELOX_FAIL("Unsupported expression: " + name);
  }
}

void addPrecomputedColumns(
    std::vector<std::unique_ptr<cudf::column>>& input_table_columns,
    const std::vector<std::tuple<int, std::string, int>>&
        precompute_instructions,
    const std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    rmm::cuda_stream_view stream) {
  for (const auto& instruction : precompute_instructions) {
    auto [dependent_column_index, ins_name, new_column_index] = instruction;
    if (ins_name == "year") {
      auto new_column = cudf::datetime::extract_datetime_component(
          input_table_columns[dependent_column_index]->view(),
          cudf::datetime::datetime_component::YEAR,
          stream,
          cudf::get_current_device_resource_ref());
      input_table_columns.emplace_back(std::move(new_column));
    } else if (ins_name == "length") {
      auto new_column = cudf::strings::count_characters(
          input_table_columns[dependent_column_index]->view(),
          stream,
          cudf::get_current_device_resource_ref());
      input_table_columns.emplace_back(std::move(new_column));
    } else if (ins_name.rfind("substr", 0) == 0) {
      std::istringstream iss(ins_name.substr(6));
      int begin_value, end_value;
      iss >> begin_value >> end_value;
      auto begin_scalar = cudf::numeric_scalar<cudf::size_type>(
          begin_value, true, stream, cudf::get_current_device_resource_ref());
      auto end_scalar = cudf::numeric_scalar<cudf::size_type>(
          end_value, true, stream, cudf::get_current_device_resource_ref());
      auto step_scalar = cudf::numeric_scalar<cudf::size_type>(
          1, true, stream, cudf::get_current_device_resource_ref());
      auto new_column = cudf::strings::slice_strings(
          input_table_columns[dependent_column_index]->view(),
          begin_scalar,
          end_scalar,
          step_scalar,
          stream,
          cudf::get_current_device_resource_ref());
      input_table_columns.emplace_back(std::move(new_column));
    } else if (ins_name.rfind("like", 0) == 0) {
      auto scalar_index = std::stoi(ins_name.substr(4));
      auto new_column = cudf::strings::like(
          input_table_columns[dependent_column_index]->view(),
          *static_cast<cudf::string_scalar*>(scalars[scalar_index].get()),
          cudf::string_scalar(
              "", true, stream, cudf::get_current_device_resource_ref()),
          stream,
          cudf::get_current_device_resource_ref());
      input_table_columns.emplace_back(std::move(new_column));
    } else {
      VELOX_FAIL("Unsupported precompute operation " + ins_name);
    }
  }
}

} // namespace facebook::velox::cudf_velox
