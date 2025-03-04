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
    const VectorPtr& vector,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars) {
  using T = typename facebook::velox::KindToFlatVector<kind>::WrapperType;
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto& type = vector->type();
  auto constVector = vector->as<facebook::velox::ConstantVector<T>>();
  T value = constVector->valueAt(0);
  if constexpr (cudf::is_fixed_width<T>()) {
    VELOX_CHECK(vector->isConstantEncoding());
    // check if decimal (unsupported by ast), if interval, if date
    if (type->isShortDecimal()) {
      VELOX_FAIL("Short decimal not supported");
      /* TODO: enable after rewriting using binary ops
      using CudfDecimalType = cudf::numeric::decimal64;
      using cudfScalarType = cudf::fixed_point_scalar<CudfDecimalType>;
      auto scalar = std::make_unique<cudfScalarType>(value,
                    type->scale(),
                     true,
                     stream,
                     mr);
      scalars.emplace_back(std::move(scalar));
      return cudf::ast::literal{
          *static_cast<cudfScalarType*>(scalars.back().get())};
      */
    } else if (type->isLongDecimal()) {
      VELOX_FAIL("Long decimal not supported");
      /* TODO: enable after rewriting using binary ops
      using CudfDecimalType = cudf::numeric::decimal128;
      using cudfScalarType = cudf::fixed_point_scalar<CudfDecimalType>;
      auto scalar = std::make_unique<cudfScalarType>(value,
                    type->scale(),
                     true,
                     stream,
                     mr);
      scalars.emplace_back(std::move(scalar));
      return cudf::ast::literal{
          *static_cast<cudfScalarType*>(scalars.back().get())};
      */
    } else if (type->isIntervalYearMonth()) {
      // no support for interval year month in cudf
      VELOX_FAIL("Interval year month not supported");
    } else if (type->isIntervalDayTime()) {
      using CudfDurationType = cudf::duration_ms;
      if constexpr (std::is_same_v<T, CudfDurationType::rep>) {
        using cudfScalarType = cudf::duration_scalar<CudfDurationType>;
        auto scalar = std::make_unique<cudfScalarType>(value, true, stream, mr);
        scalars.emplace_back(std::move(scalar));
        return cudf::ast::literal{
            *static_cast<cudfScalarType*>(scalars.back().get())};
      }
    } else if (type->isDate()) {
      using CudfDateType = cudf::timestamp_D;
      if constexpr (std::is_same_v<T, CudfDateType::rep>) {
        using cudfScalarType = cudf::timestamp_scalar<CudfDateType>;
        auto scalar = std::make_unique<cudfScalarType>(value, true, stream, mr);
        scalars.emplace_back(std::move(scalar));
        return cudf::ast::literal{
            *static_cast<cudfScalarType*>(scalars.back().get())};
      }
    } else {
      // store scalar and use its reference in the literal
      using cudfScalarType = cudf::numeric_scalar<T>;
      scalars.emplace_back(
          std::make_unique<cudfScalarType>(value, true, stream, mr));
      return cudf::ast::literal{
          *static_cast<cudfScalarType*>(scalars.back().get())};
    }
    VELOX_FAIL("Unsupported base type for literal");
  } else if (kind == TypeKind::VARCHAR) {
    VELOX_CHECK(vector->isConstantEncoding());
    auto constVector =
        vector->as<facebook::velox::ConstantVector<StringView>>();
    auto value = constVector->valueAt(0);
    std::string_view stringValue = static_cast<std::string_view>(value);
    scalars.emplace_back(
        std::make_unique<cudf::string_scalar>(stringValue, true, stream, mr));
    return cudf::ast::literal{
        *static_cast<cudf::string_scalar*>(scalars.back().get())};
  } else {
    // TODO for non-numeric types too.
    VELOX_NYI("Non-numeric types not yet implemented");
  }
}

cudf::ast::literal createLiteral(
    const VectorPtr& vector,
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
    {"lt", op::LESS},
    {"gt", op::GREATER},
    {"lte", op::LESS_EQUAL},
    {"gte", op::GREATER_EQUAL},
    {"and", op::NULL_LOGICAL_AND},
    {"or", op::NULL_LOGICAL_OR}};

const std::map<std::string, op> unary_ops = {{"not", op::NOT}};

const std::unordered_set<std::string> supported_ops = {
    "literal",
    "between",
    "cast",
    "switch",
    "year",
    "length",
    "substr",
    "like"};

struct SingleTableAstContext {
  // All members are references
  cudf::ast::tree& tree;
  std::vector<std::unique_ptr<cudf::scalar>>& scalars;
  const RowTypePtr& inputRowSchema;
  std::vector<PrecomputeInstruction>& precompute_instructions;
  cudf::ast::expression const& push_expr_to_tree(
      const std::shared_ptr<velox::exec::Expr>& expr);
  static bool can_be_evaluated(const std::shared_ptr<velox::exec::Expr>& expr);
};

struct TwoTableAstContext {
  // All members are references
  cudf::ast::tree& tree;
  std::vector<std::unique_ptr<cudf::scalar>>& scalars;
  const RowTypePtr& leftRowSchema;
  const RowTypePtr& rightRowSchema;
  std::vector<PrecomputeInstruction>& precompute_instructions;
  cudf::ast::expression const& push_expr_to_tree(
      const std::shared_ptr<velox::exec::Expr>& expr);
  static bool can_be_evaluated(const std::shared_ptr<velox::exec::Expr>& expr);
};

// Create tree from Expr
// and collect precompute instructions for non-ast operations
cudf::ast::expression const& create_ast_tree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    std::vector<PrecomputeInstruction>& precompute_instructions) {
  SingleTableAstContext context{
      tree, scalars, inputRowSchema, precompute_instructions};
  return context.push_expr_to_tree(expr);
}

cudf::ast::expression const& create_ast_tree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& leftRowSchema,
    const RowTypePtr& rightRowSchema,
    std::vector<PrecomputeInstruction>& precompute_instructions) {
  TwoTableAstContext context{
      tree, scalars, leftRowSchema, rightRowSchema, precompute_instructions};
  return context.push_expr_to_tree(expr);
}

bool SingleTableAstContext::can_be_evaluated(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  const auto& name = expr->name();
  if (supported_ops.count(name) || binary_ops.count(name) ||
      unary_ops.count(name)) {
    return std::all_of(
        expr->inputs().begin(), expr->inputs().end(), can_be_evaluated);
  }
  return std::dynamic_pointer_cast<velox::exec::FieldReference>(expr) !=
      nullptr;
}

cudf::ast::expression const& SingleTableAstContext::push_expr_to_tree(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  using op = cudf::ast::ast_operator;
  using operation = cudf::ast::operation;
  using velox::exec::ConstantExpr;
  using velox::exec::FieldReference;

  auto& name = expr->name();
  auto len = expr->inputs().size();

  if (name == "literal") {
    auto c = dynamic_cast<ConstantExpr*>(expr.get());
    VELOX_CHECK_NOT_NULL(c, "literal expression should be ConstantExpr");
    auto value = c->value();
    // convert to cudf scalar
    return tree.push(createLiteral(value, scalars));
  } else if (binary_ops.find(name) != binary_ops.end()) {
    VELOX_CHECK_EQ(len, 2);
    auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
    auto const& op2 = push_expr_to_tree(expr->inputs()[1]);
    return tree.push(operation{binary_ops.at(name), op1, op2});
  } else if (unary_ops.find(name) != unary_ops.end()) {
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
    return tree.push(operation{unary_ops.at(name), op1});
  } else if (name == "between") {
    VELOX_CHECK_EQ(len, 3);
    auto const& value = push_expr_to_tree(expr->inputs()[0]);
    auto const& lower = push_expr_to_tree(expr->inputs()[1]);
    auto const& upper = push_expr_to_tree(expr->inputs()[2]);
    // construct between(op2, op3) using >= and <=
    auto const& ge_lower =
        tree.push(operation{op::GREATER_EQUAL, value, lower});
    auto const& le_upper = tree.push(operation{op::LESS_EQUAL, value, upper});
    return tree.push(operation{op::NULL_LOGICAL_AND, ge_lower, le_upper});
  } else if (name == "cast") {
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
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
    VELOX_CHECK_EQ(len, 3);
    // check if input[1], input[2] are literals 1 and 0.
    // then simplify as typecast bool to int
    auto c1 = dynamic_cast<ConstantExpr*>(expr->inputs()[1].get());
    auto c2 = dynamic_cast<ConstantExpr*>(expr->inputs()[2].get());
    if (c1 and c1->toString() == "1:BIGINT" and c2 and
        c2->toString() == "0:BIGINT") {
      auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
      return tree.push(operation{op::CAST_TO_INT64, op1});
    } else if (c2 and c2->toString() == "0:DOUBLE") {
      auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
      auto const& op1d = tree.push(operation{op::CAST_TO_FLOAT64, op1});
      auto const& op2 = push_expr_to_tree(expr->inputs()[1]);
      return tree.push(operation{op::MUL, op1d, op2});
    } else {
      VELOX_NYI("Unsupported switch complex operation " + expr->toString());
    }
  } else if (name == "year") {
    VELOX_CHECK_EQ(len, 1);
    // ensure expr->inputs()[0] is a field
    auto fieldExpr =
        std::dynamic_pointer_cast<FieldReference>(expr->inputs()[0]);
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
    VELOX_CHECK_EQ(len, 1);
    // ensure expr->inputs()[0] is a field
    auto fieldExpr =
        std::dynamic_pointer_cast<FieldReference>(expr->inputs()[0]);
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
    VELOX_CHECK_EQ(len, 3);
    auto fieldExpr =
        std::dynamic_pointer_cast<FieldReference>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");
    auto dependent_column_index =
        inputRowSchema->getChildIdx(fieldExpr->name());
    auto new_column_index =
        inputRowSchema->size() + precompute_instructions.size();
    // add this index and precompute instruction to a data structure
    auto c1 = dynamic_cast<ConstantExpr*>(expr->inputs()[1].get());
    auto c2 = dynamic_cast<ConstantExpr*>(expr->inputs()[2].get());
    std::string substr_expr =
        "substr " + c1->value()->toString(0) + " " + c2->value()->toString(0);
    precompute_instructions.emplace_back(
        dependent_column_index, substr_expr, new_column_index);
    // This custom op should be added to input columns.
    return tree.push(cudf::ast::column_reference(new_column_index));
  } else if (name == "like") {
    VELOX_CHECK_EQ(len, 2);
    auto fieldExpr =
        std::dynamic_pointer_cast<FieldReference>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");
    auto dependent_column_index =
        inputRowSchema->getChildIdx(fieldExpr->name());
    auto new_column_index =
        inputRowSchema->size() + precompute_instructions.size();
    auto literalExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(literalExpr, "Expression is not a literal");
    createLiteral(literalExpr->value(), scalars);
    std::string like_expr = "like " + std::to_string(scalars.size() - 1);
    precompute_instructions.emplace_back(
        dependent_column_index, like_expr, new_column_index);
    return tree.push(cudf::ast::column_reference(new_column_index));
  } else if (auto fieldExpr = std::dynamic_pointer_cast<FieldReference>(expr)) {
    auto column_index = inputRowSchema->getChildIdx(name);
    VELOX_CHECK(column_index != -1, "Field not found, " + name);
    return tree.push(cudf::ast::column_reference(column_index));
  } else {
    std::cerr << "Unsupported expression: " << expr->toString() << std::endl;
    VELOX_FAIL("Unsupported expression: " + name);
  }
}

// TODO (dm): This is a copy of the single table case. refactor.
cudf::ast::expression const& TwoTableAstContext::push_expr_to_tree(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  using op = cudf::ast::ast_operator;
  using operation = cudf::ast::operation;
  using velox::exec::ConstantExpr;
  using velox::exec::FieldReference;

  auto& name = expr->name();
  auto len = expr->inputs().size();

  if (name == "literal") {
    auto c = dynamic_cast<ConstantExpr*>(expr.get());
    VELOX_CHECK_NOT_NULL(c, "literal expression should be ConstantExpr");
    auto value = c->value();
    // convert to cudf scalar
    return tree.push(createLiteral(value, scalars));
  } else if (binary_ops.find(name) != binary_ops.end()) {
    VELOX_CHECK_EQ(len, 2);
    auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
    auto const& op2 = push_expr_to_tree(expr->inputs()[1]);
    return tree.push(operation{binary_ops.at(name), op1, op2});
  } else if (unary_ops.find(name) != unary_ops.end()) {
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
    return tree.push(operation{unary_ops.at(name), op1});
  } else if (name == "between") {
    VELOX_CHECK_EQ(len, 3);
    auto const& value = push_expr_to_tree(expr->inputs()[0]);
    auto const& lower = push_expr_to_tree(expr->inputs()[1]);
    auto const& upper = push_expr_to_tree(expr->inputs()[2]);
    // construct between(op2, op3) using >= and <=
    auto const& ge_lower =
        tree.push(operation{op::GREATER_EQUAL, value, lower});
    auto const& le_upper = tree.push(operation{op::LESS_EQUAL, value, upper});
    return tree.push(operation{op::NULL_LOGICAL_AND, ge_lower, le_upper});
  } else if (name == "cast") {
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
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
    VELOX_CHECK_EQ(len, 3);
    // check if input[1], input[2] are literals 1 and 0.
    // then simplify as typecast bool to int
    auto c1 = dynamic_cast<ConstantExpr*>(expr->inputs()[1].get());
    auto c2 = dynamic_cast<ConstantExpr*>(expr->inputs()[2].get());
    if (c1 and c1->toString() == "1:BIGINT" and c2 and
        c2->toString() == "0:BIGINT") {
      auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
      return tree.push(operation{op::CAST_TO_INT64, op1});
    } else if (c2 and c2->toString() == "0:DOUBLE") {
      auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
      auto const& op1d = tree.push(operation{op::CAST_TO_FLOAT64, op1});
      auto const& op2 = push_expr_to_tree(expr->inputs()[1]);
      return tree.push(operation{op::MUL, op1d, op2});
    } else {
      VELOX_NYI("Unsupported switch complex operation " + expr->toString());
    }
  } else if (name == "year") {
    VELOX_NYI("Precomputed not supported in two table case yet");
    VELOX_CHECK_EQ(len, 1);
    // ensure expr->inputs()[0] is a field
    auto fieldExpr =
        std::dynamic_pointer_cast<FieldReference>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");
    auto dependent_column_index = leftRowSchema->getChildIdx(fieldExpr->name());
    auto new_column_index =
        leftRowSchema->size() + precompute_instructions.size();
    // add this index and precompute instruction to a data structure
    precompute_instructions.emplace_back(
        dependent_column_index, "year", new_column_index);
    // This custom op should be added to input columns.
    // cast to big int
    auto const& col_ref =
        tree.push(cudf::ast::column_reference(new_column_index));
    return tree.push(operation{op::CAST_TO_INT64, col_ref});
  } else if (name == "length") {
    VELOX_NYI("Precomputed not supported in two table case yet");
    VELOX_CHECK_EQ(len, 1);
    // ensure expr->inputs()[0] is a field
    auto fieldExpr =
        std::dynamic_pointer_cast<FieldReference>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");
    auto dependent_column_index = leftRowSchema->getChildIdx(fieldExpr->name());
    auto new_column_index =
        leftRowSchema->size() + precompute_instructions.size();
    // add this index and precompute instruction to a data structure
    precompute_instructions.emplace_back(
        dependent_column_index, "length", new_column_index);
    // This custom op should be added to input columns.
    auto const& col_ref =
        tree.push(cudf::ast::column_reference(new_column_index));
    return tree.push(operation{op::CAST_TO_INT64, col_ref});
  } else if (name == "substr") {
    VELOX_NYI("Precomputed not supported in two table case yet");
    // add precompute instruction, special handling col_ref during ast
    // evaluation
    VELOX_CHECK_EQ(len, 3);
    auto fieldExpr =
        std::dynamic_pointer_cast<FieldReference>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");
    auto dependent_column_index = leftRowSchema->getChildIdx(fieldExpr->name());
    auto new_column_index =
        leftRowSchema->size() + precompute_instructions.size();
    // add this index and precompute instruction to a data structure
    auto c1 = dynamic_cast<ConstantExpr*>(expr->inputs()[1].get());
    auto c2 = dynamic_cast<ConstantExpr*>(expr->inputs()[2].get());
    std::string substr_expr =
        "substr " + c1->value()->toString(0) + " " + c2->value()->toString(0);
    precompute_instructions.emplace_back(
        dependent_column_index, substr_expr, new_column_index);
    // This custom op should be added to input columns.
    return tree.push(cudf::ast::column_reference(new_column_index));
  } else if (name == "like") {
    VELOX_NYI("Precomputed not supported in two table case yet");
    VELOX_CHECK_EQ(len, 2);
    auto fieldExpr =
        std::dynamic_pointer_cast<FieldReference>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");
    auto dependent_column_index = leftRowSchema->getChildIdx(fieldExpr->name());
    auto new_column_index =
        leftRowSchema->size() + precompute_instructions.size();
    auto literalExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(literalExpr, "Expression is not a literal");
    createLiteral(literalExpr->value(), scalars);
    std::string like_expr = "like " + std::to_string(scalars.size() - 1);
    precompute_instructions.emplace_back(
        dependent_column_index, like_expr, new_column_index);
    return tree.push(cudf::ast::column_reference(new_column_index));
  } else if (auto fieldExpr = std::dynamic_pointer_cast<FieldReference>(expr)) {
    // figure out which table the field belongs to
    if (leftRowSchema->containsChild(name)) {
      auto column_index = leftRowSchema->getChildIdx(name);
      return tree.push(cudf::ast::column_reference(
          column_index, cudf::ast::table_reference::LEFT));
    } else if (rightRowSchema->containsChild(name)) {
      auto column_index = rightRowSchema->getChildIdx(name);
      return tree.push(cudf::ast::column_reference(
          column_index, cudf::ast::table_reference::RIGHT));
    } else {
      VELOX_FAIL("Field not found, " + name);
    }
  } else {
    std::cerr << "Unsupported expression: " << expr->toString() << std::endl;
    VELOX_FAIL("Unsupported expression: " + name);
  }
}

void addPrecomputedColumns(
    std::vector<std::unique_ptr<cudf::column>>& input_table_columns,
    const std::vector<PrecomputeInstruction>& precompute_instructions,
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
      int begin_value, length;
      iss >> begin_value >> length;
      auto begin_scalar = cudf::numeric_scalar<cudf::size_type>(
          begin_value - 1,
          true,
          stream,
          cudf::get_current_device_resource_ref());
      auto end_scalar = cudf::numeric_scalar<cudf::size_type>(
          begin_value - 1 + length,
          true,
          stream,
          cudf::get_current_device_resource_ref());
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

ExpressionEvaluator::ExpressionEvaluator(
    const std::vector<std::shared_ptr<velox::exec::Expr>>& exprs,
    const RowTypePtr& inputRowSchema) {
  for (const auto& expr : exprs) {
    cudf::ast::tree tree;
    create_ast_tree(
        expr, tree, scalars_, inputRowSchema, precompute_instructions_);
    projectAst_.emplace_back(std::move(tree));
  }
}

void ExpressionEvaluator::close() {
  projectAst_.clear();
  scalars_.clear();
  precompute_instructions_.clear();
}

std::vector<std::unique_ptr<cudf::column>> ExpressionEvaluator::compute(
    std::vector<std::unique_ptr<cudf::column>>& input_table_columns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  addPrecomputedColumns(
      input_table_columns, precompute_instructions_, scalars_, stream);
  auto ast_input_table =
      std::make_unique<cudf::table>(std::move(input_table_columns));
  auto ast_input_table_view = ast_input_table->view();
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (auto& tree : projectAst_) {
    if (auto col_ref_ptr =
            dynamic_cast<cudf::ast::column_reference const*>(&tree.back())) {
      auto col = std::make_unique<cudf::column>(
          ast_input_table_view.column(col_ref_ptr->get_column_index()),
          stream,
          mr);
      columns.emplace_back(std::move(col));
    } else {
      auto col =
          cudf::compute_column(ast_input_table_view, tree.back(), stream, mr);
      columns.emplace_back(std::move(col));
    }
  }
  input_table_columns = ast_input_table->release();
  return columns;
}

bool ExpressionEvaluator::can_be_evaluated(
    const std::vector<std::shared_ptr<velox::exec::Expr>>& exprs) {
  return std::all_of(
      exprs.begin(), exprs.end(), SingleTableAstContext::can_be_evaluated);
}
} // namespace facebook::velox::cudf_velox
