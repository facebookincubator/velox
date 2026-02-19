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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
// TODO(kn): in another PR
// #include "velox/experimental/cudf/CudfNoDefaults.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FieldReference.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/ConstantVector.h"

#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

namespace facebook::velox::cudf_velox {
namespace {

cudf::ast::literal createLiteral(
    const VectorPtr& vector,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    size_t atIndex = 0) {
  const auto kind = vector->typeKind();
  const auto& type = vector->type();
  variant value =
      VELOX_DYNAMIC_TYPE_DISPATCH(getVariant, kind, vector, atIndex);
  return VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      makeScalarAndLiteral, kind, type, value, scalars);
}

// Helper function to extract literals from array elements based on type
void extractArrayLiterals(
    const ArrayVector* arrayVector,
    std::vector<cudf::ast::literal>& literals,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    vector_size_t offset,
    vector_size_t size) {
  auto elements = arrayVector->elements();

  for (auto i = offset; i < offset + size; ++i) {
    if (elements->isNullAt(i)) {
      // Skip null values for IN expressions
      continue;
    } else {
      literals.emplace_back(createLiteral(elements, scalars, i));
    }
  }
}

// Function to create literals from an array vector
std::vector<cudf::ast::literal> createLiteralsFromArray(
    const VectorPtr& vector,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars) {
  std::vector<cudf::ast::literal> literals;

  // Check if it's a constant vector containing an array
  if (vector->isConstantEncoding()) {
    auto constantVector = vector->asUnchecked<ConstantVector<ComplexType>>();
    if (constantVector->isNullAt(0)) {
      // Return empty vector for null array
      return literals;
    }

    auto valueVector = constantVector->valueVector();
    if (valueVector->encoding() == VectorEncoding::Simple::ARRAY) {
      auto arrayVector = valueVector->as<ArrayVector>();
      auto index = constantVector->index();
      auto size = arrayVector->sizeAt(index);
      if (size == 0) {
        // Return empty vector for empty array
        return literals;
      }

      auto offset = arrayVector->offsetAt(index);
      auto elements = arrayVector->elements();

      // Handle different element types
      if (elements->isScalar()) {
        literals.reserve(size);
        extractArrayLiterals(arrayVector, literals, scalars, offset, size);
      } else if (elements->typeKind() == TypeKind::ARRAY) {
        // Nested arrays not supported in IN expressions
        VELOX_FAIL("Nested arrays not supported in IN expressions");
      } else {
        VELOX_FAIL(
            "Unsupported element type in array: {}",
            elements->type()->toString());
      }
    } else {
      VELOX_FAIL("Expected ARRAY encoding");
    }
  } else {
    VELOX_FAIL("Expected constant vector for IN list");
  }

  return literals;
}

std::string stripPrefix(const std::string& input, const std::string& prefix) {
  if (input.size() >= prefix.size() &&
      input.compare(0, prefix.size(), prefix) == 0) {
    return input.substr(prefix.size());
  }
  return input;
}

using Op = cudf::ast::ast_operator;
const std::unordered_map<std::string, Op> prestoBinaryOps = {
    {"plus", Op::ADD},
    {"minus", Op::SUB},
    {"multiply", Op::MUL},
    {"divide", Op::DIV},
    {"eq", Op::EQUAL},
    {"neq", Op::NOT_EQUAL},
    {"lt", Op::LESS},
    {"gt", Op::GREATER},
    {"lte", Op::LESS_EQUAL},
    {"gte", Op::GREATER_EQUAL},
    {"and", Op::NULL_LOGICAL_AND},
    {"or", Op::NULL_LOGICAL_OR},
    {"mod", Op::MOD},
};

const std::unordered_map<std::string, Op> sparkBinaryOps = {
    {"add", Op::ADD},
    {"subtract", Op::SUB},
    {"multiply", Op::MUL},
    {"divide", Op::DIV},
    {"equalto", Op::EQUAL},
    {"lessthan", Op::LESS},
    {"greaterthan", Op::GREATER},
    {"lessthanorequal", Op::LESS_EQUAL},
    {"greaterthanorequal", Op::GREATER_EQUAL},
    {"and", Op::NULL_LOGICAL_AND},
    {"or", Op::NULL_LOGICAL_OR},
    {"mod", Op::MOD},
};

const std::unordered_map<std::string, Op> binaryOps = [] {
  std::unordered_map<std::string, Op> merged(
      sparkBinaryOps.begin(), sparkBinaryOps.end());
  merged.insert(prestoBinaryOps.begin(), prestoBinaryOps.end());
  return merged;
}();

const std::map<std::string, Op> unaryOps = {
    {"not", Op::NOT},
    {"is_null", Op::IS_NULL}};

namespace detail {

// return the AST operator for the given expression name, if any
std::optional<Op> opFromFunctionName(const std::string& funcName) {
  if (binaryOps.find(funcName) != binaryOps.end()) {
    return binaryOps.at(funcName);
  } else if (unaryOps.find(funcName) != unaryOps.end()) {
    return unaryOps.at(funcName);
  }
  return std::nullopt;
}

bool isOpAndInputsSupported(
    const cudf::ast::ast_operator op,
    const std::vector<cudf::data_type>& inputCudfDataTypes) {
  // check arity
  const auto arity = cudf::ast::detail::ast_operator_arity(op);
  if (arity != static_cast<int>(inputCudfDataTypes.size())) {
    LOG(WARNING) << "Arity mismatch for operator: " << static_cast<int>(op)
                 << " with input types size " << inputCudfDataTypes.size();
    return false;
  }
  // check for a cuDF implementation of this op with these inputs
  try {
    // this will throw if no matching implementation is found
    const auto returnCudfType =
        cudf::ast::detail::ast_operator_return_type(op, inputCudfDataTypes);
    // check it's a sensible type
    return returnCudfType.id() != cudf::type_id::EMPTY;
  } catch (...) {
    // no matching cuDF implementation
  }
  return false;
}

// not special form, name = function, so unsupported for astpure
// "in", "between", "isnotnull" are not special form, but supported for astpure
// enum class SpecialFormKind : int32_t {
//   kFieldAccess = 0, supported if not nested column / function
//   kConstant = 1, "literal" for fixed_width and VARCHAR / function
//   kCast = 2, "cast" or "try_cast" to int32, int64, double only / function
//   kCoalesce = 3, unsupported/function
//   kSwitch = 4, unsupported
//   kLambda = 5, unsupported
//   kTry = 6, unsupported
//   kAnd = 7, "and" or "or" with multiple inputs
//   kOr = 8,
//   kCustom = 999,
// };
// check if the expression (name + input types) is supported in AST
bool isAstExprSupported(const std::shared_ptr<velox::exec::Expr>& expr) {
  using velox::exec::FieldReference;
  using Op = cudf::ast::ast_operator;

  const auto name =
      stripPrefix(expr->name(), CudfConfig::getInstance().functionNamePrefix);
  const auto len = expr->inputs().size();

  // Literals and field references are always supported
  auto isSupportedLiteral = [&](const TypePtr& type) {
    try {
      auto cudfType = cudf::data_type(veloxToCudfTypeId(type));
      return cudf::is_fixed_width(cudfType) ||
          cudfType.id() == cudf::type_id::STRING;
    } catch (...) {
      LOG(WARNING) << "Unsupported type for literal: " << type->toString();
      return false;
    }
  };
  if (name == "literal") {
    auto type = expr->type();
    return isSupportedLiteral(type);
  }
  if (auto fieldExpr = std::dynamic_pointer_cast<FieldReference>(expr)) {
    const auto fieldName =
        fieldExpr->inputs().empty() ? name : fieldExpr->inputs()[0]->name();
    if (fieldExpr->field() == fieldName) {
      return true;
    }
    LOG(WARNING) << "Field " << name << "not found, in expression "
                 << expr->toString();
    return false;
  }

  // Convert input types to CUDF types once
  std::vector<cudf::data_type> inputCudfDataTypes;
  inputCudfDataTypes.reserve(len);
  for (const auto& input : expr->inputs()) {
    try {
      inputCudfDataTypes.push_back(
          cudf::data_type(veloxToCudfTypeId(input->type())));
    } catch (...) {
      return false;
    }
  }

  // Binary operations
  if (binaryOps.find(name) != binaryOps.end()) {
    // AND/OR can handle multiple inputs by chaining
    if ((name == "and" || name == "or") && len > 2) {
      for (size_t i = 1; i < len; i++) {
        if (!isOpAndInputsSupported(
                binaryOps.at(name),
                {inputCudfDataTypes[0], inputCudfDataTypes[i]})) {
          return false;
        }
      }
      return true;
    }
    return len == 2 &&
        isOpAndInputsSupported(binaryOps.at(name), inputCudfDataTypes);
  }

  // Unary operations (includes both unaryOps and "isnotnull")
  if (unaryOps.find(name) != unaryOps.end()) {
    return isOpAndInputsSupported(unaryOps.at(name), inputCudfDataTypes);
  }
  if (name == "isnotnull" && len == 1) {
    return isOpAndInputsSupported(Op::IS_NULL, inputCudfDataTypes);
  }

  // Between: value >= lower AND value <= upper
  if (name == "between" && len == 3) {
    return isOpAndInputsSupported(
               Op::GREATER_EQUAL,
               {inputCudfDataTypes[0], inputCudfDataTypes[1]}) &&
        isOpAndInputsSupported(
               Op::LESS_EQUAL, {inputCudfDataTypes[0], inputCudfDataTypes[2]});
  }

  // In: chain of EQUAL operations
  if (name == "in") {
    return len == 2 && isSupportedLiteral(expr->inputs()[0]->type()) &&
        isOpAndInputsSupported(
               Op::EQUAL, {inputCudfDataTypes[0], inputCudfDataTypes[0]});
  }

  // Cast operations: only INTEGER, BIGINT, DOUBLE supported in pure AST
  if ((name == "cast" || name == "try_cast") && len == 1) {
    const auto outputKind = expr->type()->kind();
    if (outputKind == TypeKind::INTEGER || outputKind == TypeKind::BIGINT) {
      return isOpAndInputsSupported(Op::CAST_TO_INT64, inputCudfDataTypes);
    }
    if (outputKind == TypeKind::DOUBLE) {
      return isOpAndInputsSupported(Op::CAST_TO_FLOAT64, inputCudfDataTypes);
    }
    return false;
  }

  LOG(WARNING) << "Unsupported expression by AST: " << expr->toString();
  return false;
}

} // namespace detail

struct AstContext {
  cudf::ast::tree& tree;
  std::vector<std::unique_ptr<cudf::scalar>>& scalars;
  const std::vector<RowTypePtr> inputRowSchema;
  const std::vector<std::reference_wrapper<std::vector<PrecomputeInstruction>>>
      precomputeInstructions;
  const std::shared_ptr<velox::exec::Expr>
      rootExpr; // Track the root expression
  bool allowPureAstOnly;

  cudf::ast::expression const& pushExprToTree(
      const std::shared_ptr<velox::exec::Expr>& expr);
  cudf::ast::expression const& addPrecomputeInstructionOnSide(
      size_t sideIdx,
      size_t columnIndex,
      std::string const& instruction,
      std::string const& fieldName,
      const std::shared_ptr<CudfExpression>& node = nullptr);
  cudf::ast::expression const& addPrecomputeInstruction(
      std::string const& name,
      std::string const& instruction,
      std::string const& fieldName = {},
      const std::shared_ptr<CudfExpression>& node = nullptr);
  cudf::ast::expression const& multipleInputsToPairWise(
      const std::shared_ptr<velox::exec::Expr>& expr);
  static bool canBeEvaluated(const std::shared_ptr<velox::exec::Expr>& expr);
  // Determines which side (0=left, 1=right) an expression references by
  // examining its field references. Returns -1 if no fields found.
  int findExpressionSide(const std::shared_ptr<velox::exec::Expr>& expr) const;
};

// get nested column indices
std::vector<int> getNestedColumnIndices(
    const TypePtr& rowType,
    const std::string& fieldName) {
  std::vector<int> indices;
  auto rowTypePtr = asRowType(rowType);
  if (rowTypePtr->containsChild(fieldName)) {
    auto columnIndex = rowTypePtr->getChildIdx(fieldName);
    indices.push_back(columnIndex);
  }
  return indices;
}

cudf::ast::expression const& AstContext::addPrecomputeInstructionOnSide(
    size_t sideIdx,
    size_t columnIndex,
    std::string const& instruction,
    std::string const& fieldName,
    const std::shared_ptr<CudfExpression>& node) {
  auto newColumnIndex = inputRowSchema[sideIdx].get()->size() +
      precomputeInstructions[sideIdx].get().size();
  if (fieldName.empty()) {
    // This custom op should be added to input columns.
    precomputeInstructions[sideIdx].get().emplace_back(
        columnIndex, instruction, newColumnIndex, node);
  } else {
    auto nestedIndices = getNestedColumnIndices(
        inputRowSchema[sideIdx].get()->childAt(columnIndex), fieldName);
    precomputeInstructions[sideIdx].get().emplace_back(
        columnIndex, instruction, newColumnIndex, nestedIndices, node);
  }
  auto side = static_cast<cudf::ast::table_reference>(sideIdx);
  return tree.push(cudf::ast::column_reference(newColumnIndex, side));
}

cudf::ast::expression const& AstContext::addPrecomputeInstruction(
    std::string const& name,
    std::string const& instruction,
    std::string const& fieldName,
    const std::shared_ptr<CudfExpression>& node) {
  for (size_t sideIdx = 0; sideIdx < inputRowSchema.size(); ++sideIdx) {
    if (inputRowSchema[sideIdx].get()->containsChild(name)) {
      auto columnIndex = inputRowSchema[sideIdx].get()->getChildIdx(name);
      return addPrecomputeInstructionOnSide(
          sideIdx, columnIndex, instruction, fieldName, node);
    }
  }
  VELOX_FAIL("Field not found, " + name);
}

/// Handles logical AND/OR expressions with multiple inputs by converting them
/// into a chain of binary operations. For example, "a AND b AND c" becomes
/// "(a AND b) AND c".
///
/// @param expr The expression containing multiple inputs for AND/OR operation
/// @return A reference to the resulting AST expression
cudf::ast::expression const& AstContext::multipleInputsToPairWise(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  using Operation = cudf::ast::operation;

  const auto name =
      stripPrefix(expr->name(), CudfConfig::getInstance().functionNamePrefix);
  auto len = expr->inputs().size();
  // Create a simple chain of operations
  auto result = &pushExprToTree(expr->inputs()[0]);

  // Chain the rest of the inputs sequentially
  for (size_t i = 1; i < len; i++) {
    auto const& nextInput = pushExprToTree(expr->inputs()[i]);
    result = &tree.push(Operation{binaryOps.at(name), *result, nextInput});
  }
  return *result;
}

/// Pushes an expression into the AST tree and returns a reference to the
/// resulting expression.
///
/// @param expr The expression to push into the AST tree
/// @return A reference to the resulting AST expression
cudf::ast::expression const& AstContext::pushExprToTree(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  using Op = cudf::ast::ast_operator;
  using Operation = cudf::ast::operation;
  using velox::exec::ConstantExpr;
  using velox::exec::FieldReference;

  const auto name =
      stripPrefix(expr->name(), CudfConfig::getInstance().functionNamePrefix);
  auto len = expr->inputs().size();
  auto& type = expr->type();

  if (name == "literal") {
    auto c = dynamic_cast<ConstantExpr*>(expr.get());
    VELOX_CHECK_NOT_NULL(c, "literal expression should be ConstantExpr");
    auto value = c->value();
    VELOX_CHECK(value->isConstantEncoding());

    // Special case: VARCHAR literals cannot be handled by cudf::compute_column
    // as the final output due to variable-width output limitation.
    // However, if this is part of a larger expression tree (e.g., string
    // comparison), then cudf can handle it fine since the final output won't be
    // VARCHAR. We only need special handling when this literal will be the
    // final output.
    if (expr->type()->kind() == TypeKind::VARCHAR && expr == rootExpr) {
      // convert to cudf scalar and store it
      createLiteral(value, scalars);
      // The scalar index is scalars.size() - 1 since we just added it
      std::string fillExpr = "fill " + std::to_string(scalars.size() - 1);
      // For literals, we use the first column just to get the size, but create
      // a new column The new column will be appended after the original input
      // columns
      return addPrecomputeInstruction(inputRowSchema[0]->nameOf(0), fillExpr);
    }

    return tree.push(createLiteral(value, scalars));
  } else if (binaryOps.find(name) != binaryOps.end()) {
    if (name == "and" or name == "or") {
      return multipleInputsToPairWise(expr);
    }
    VELOX_CHECK_EQ(len, 2);
    auto const& op1 = pushExprToTree(expr->inputs()[0]);
    auto const& op2 = pushExprToTree(expr->inputs()[1]);
    return tree.push(Operation{binaryOps.at(name), op1, op2});
  } else if (unaryOps.find(name) != unaryOps.end()) {
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 = pushExprToTree(expr->inputs()[0]);
    return tree.push(Operation{unaryOps.at(name), op1});
  } else if (name == "isnotnull") {
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 = pushExprToTree(expr->inputs()[0]);
    auto const& nullOp = tree.push(Operation{Op::IS_NULL, op1});
    return tree.push(Operation{Op::NOT, nullOp});
  } else if (name == "between") {
    VELOX_CHECK_EQ(len, 3);
    auto const& value = pushExprToTree(expr->inputs()[0]);
    auto const& lower = pushExprToTree(expr->inputs()[1]);
    auto const& upper = pushExprToTree(expr->inputs()[2]);
    // construct between(op2, op3) using >= and <=
    auto const& geLower = tree.push(Operation{Op::GREATER_EQUAL, value, lower});
    auto const& leUpper = tree.push(Operation{Op::LESS_EQUAL, value, upper});
    return tree.push(Operation{Op::NULL_LOGICAL_AND, geLower, leUpper});
  } else if (name == "in") {
    // number of inputs is variable. >=2
    VELOX_CHECK_EQ(len, 2);
    // actually len is 2, second input is ARRAY
    auto const& op1 = pushExprToTree(expr->inputs()[0]);
    auto c = dynamic_cast<ConstantExpr*>(expr->inputs()[1].get());
    VELOX_CHECK_NOT_NULL(c, "literal expression should be ConstantExpr");
    auto value = c->value();
    VELOX_CHECK_NOT_NULL(value, "ConstantExpr value is null");

    // Use the new createLiteralsFromArray function to get literals
    auto literals = createLiteralsFromArray(value, scalars);

    // Create equality expressions for each literal and OR them together
    std::vector<const cudf::ast::expression*> exprVec;
    for (auto& literal : literals) {
      auto const& opi = tree.push(std::move(literal));
      auto const& logicalNode = tree.push(Operation{Op::EQUAL, op1, opi});
      exprVec.push_back(&logicalNode);
    }

    // Handle empty IN list case
    if (exprVec.empty()) {
      // FAIL
      VELOX_FAIL("Empty IN list");
      // Return FALSE for empty IN list
      // auto falseValue = std::make_shared<ConstantVector<bool>>(
      //     value->pool(), 1, false, TypeKind::BOOLEAN, false);
      // return tree.push(createLiteral(falseValue, scalars));
    }

    // OR all logical nodes
    auto* result = exprVec[0];
    for (size_t i = 1; i < exprVec.size(); i++) {
      auto const& treeNode =
          tree.push(Operation{Op::NULL_LOGICAL_OR, *result, *exprVec[i]});
      result = &treeNode;
    }
    return *result;
  } else if (name == "cast" || name == "try_cast") {
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 = pushExprToTree(expr->inputs()[0]);
    if (expr->type()->kind() == TypeKind::INTEGER) {
      // No int32 cast in cudf ast
      return tree.push(Operation{Op::CAST_TO_INT64, op1});
    } else if (expr->type()->kind() == TypeKind::BIGINT) {
      return tree.push(Operation{Op::CAST_TO_INT64, op1});
    } else if (expr->type()->kind() == TypeKind::DOUBLE) {
      return tree.push(Operation{Op::CAST_TO_FLOAT64, op1});
    } else {
      VELOX_FAIL("Unsupported type for cast operation");
    }
  } else if (auto fieldExpr = std::dynamic_pointer_cast<FieldReference>(expr)) {
    // Refer to the appropriate side
    const auto fieldName =
        fieldExpr->inputs().empty() ? name : fieldExpr->inputs()[0]->name();
    for (size_t sideIdx = 0; sideIdx < inputRowSchema.size(); ++sideIdx) {
      auto& schema = inputRowSchema[sideIdx];
      if (schema.get()->containsChild(fieldName)) {
        auto columnIndex = schema.get()->getChildIdx(fieldName);
        // This column may be complex data type like ROW, we need to get the
        // name from row. Push fieldName.name to the tree.
        auto side = static_cast<cudf::ast::table_reference>(sideIdx);
        if (fieldExpr->field() == fieldName) {
          return tree.push(cudf::ast::column_reference(columnIndex, side));
        } else if (!allowPureAstOnly) {
          return addPrecomputeInstruction(
              fieldName, "nested_column", fieldExpr->field());
        } else {
          VELOX_FAIL("Unsupported type for nested column operation");
        }
      }
    }
    VELOX_FAIL("Field not found, " + name);
  } else if (!allowPureAstOnly && canBeEvaluatedByCudf(expr, /*deep=*/false)) {
    // Shallow check: only verify this operation is supported
    // Children will be recursively handled by createCudfExpression
    // Determine which side this expression references
    int sideIdx = findExpressionSide(expr);
    if (sideIdx < 0) {
      sideIdx = 0; // Default to left side if no fields found
    }
    auto node =
        createCudfExpression(expr, inputRowSchema[sideIdx], kAstEvaluatorName);
    return addPrecomputeInstructionOnSide(sideIdx, 0, name, "", node);
  } else {
    VELOX_FAIL("Unsupported expression: " + name);
  }
}

int AstContext::findExpressionSide(
    const std::shared_ptr<velox::exec::Expr>& expr) const {
  for (const auto* field : expr->distinctFields()) {
    for (size_t sideIdx = 0; sideIdx < inputRowSchema.size(); ++sideIdx) {
      if (inputRowSchema[sideIdx].get()->containsChild(field->field())) {
        return static_cast<int>(sideIdx);
      }
    }
  }
  return -1;
}

std::vector<ColumnOrView> precomputeSubexpressions(
    const std::vector<cudf::column_view>& inputColumnViews,
    const std::vector<PrecomputeInstruction>& precomputeInstructions,
    const std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    rmm::cuda_stream_view stream) {
  std::vector<ColumnOrView> precomputedColumns;
  precomputedColumns.reserve(precomputeInstructions.size());

  for (const auto& instruction : precomputeInstructions) {
    auto
        [dependent_column_index,
         ins_name,
         new_column_index,
         nested_dependent_column_indices,
         cudf_expression] = instruction;

    // If a compiled cudf node is available, evaluate it directly.
    if (cudf_expression) {
      auto result = cudf_expression->eval(
          inputColumnViews,
          stream,
          cudf_velox::get_output_mr(),
          /*finalize=*/true);
      precomputedColumns.push_back(std::move(result));
      continue;
    }
    if (ins_name.rfind("fill", 0) == 0) {
      auto scalarIndex =
          std::stoi(ins_name.substr(5)); // "fill " is 5 characters
      auto newColumn = cudf::make_column_from_scalar(
          *static_cast<cudf::string_scalar*>(scalars[scalarIndex].get()),
          inputColumnViews[dependent_column_index].size(),
          stream,
          cudf_velox::get_output_mr());
      precomputedColumns.push_back(std::move(newColumn));
    } else if (ins_name == "nested_column") {
      // Nested column already exists in input. Don't materialize.
      auto view = inputColumnViews[dependent_column_index].child(
          nested_dependent_column_indices[0]);
      precomputedColumns.push_back(view);
    } else {
      VELOX_FAIL("Unsupported precompute operation " + ins_name);
    }
  }

  return precomputedColumns;
}

} // namespace
} // namespace facebook::velox::cudf_velox
