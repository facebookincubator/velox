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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/AstUtils.h"

#include "velox/core/Expressions.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FieldReference.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/ConstantVector.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

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
    {"or", Op::NULL_LOGICAL_OR}};

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

const std::map<std::string, Op> unaryOps = {{"not", Op::NOT}};

const std::unordered_set<std::string> astSupportedOps =
    {"literal", "between", "in", "cast", "switch", "if"};

namespace detail {

// Check if this specific operation is supported by AST (shallow check only)
bool isAstSupported(const std::string& exprName) {
  const auto name =
      stripPrefix(exprName, CudfConfig::getInstance().functionNamePrefix);

  return astSupportedOps.count(name) || binaryOps.count(name) ||
      unaryOps.count(name);
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
    if (len > 2 and (name == "and" or name == "or")) {
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
      // TODO (dm): Similar to else block in switch/if, remove this once
      // ASTEvaluator provides type based canEvaluate
      auto node =
          createCudfExpression(expr, inputRowSchema[0], kAstEvaluatorName);
      return addPrecomputeInstructionOnSide(0, 0, name, "", node);
    }
  } else if (name == "switch" || name == "if") {
    VELOX_CHECK_EQ(len, 3);
    // check if input[1], input[2] are literals 1 and 0.
    // then simplify as typecast bool to int
    auto c1 = dynamic_cast<ConstantExpr*>(expr->inputs()[1].get());
    auto c2 = dynamic_cast<ConstantExpr*>(expr->inputs()[2].get());
    if ((c1 and c1->toString() == "1:BIGINT" and c2 and
         c2->toString() == "0:BIGINT") ||
        (c1 and c1->toString() == "1:INTEGER" and c2 and
         c2->toString() == "0:INTEGER")) {
      auto const& op1 = pushExprToTree(expr->inputs()[0]);
      return tree.push(Operation{Op::CAST_TO_INT64, op1});
    } else if (c2 and c2->toString() == "0:DOUBLE") {
      auto const& op1 = pushExprToTree(expr->inputs()[0]);
      auto const& op1d = tree.push(Operation{Op::CAST_TO_FLOAT64, op1});
      auto const& op2 = pushExprToTree(expr->inputs()[1]);
      return tree.push(Operation{Op::MUL, op1d, op2});
    } else if (
        c1 and c1->toString() == "1:INTEGER" and c2 and
        c2->toString() == "0:INTEGER") {
      return pushExprToTree(expr->inputs()[0]);
    } else {
      // TODO (dm): This can be better handled by checking which function
      // signatures are supported before dispatching. e.g. in this case, it
      // would be better if ast never agreed to evaluate a top level switch on
      // unsupported types
      auto node =
          createCudfExpression(expr, inputRowSchema[0], kAstEvaluatorName);
      return addPrecomputeInstructionOnSide(0, 0, name, "", node);
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
        } else {
          return addPrecomputeInstruction(
              fieldName, "nested_column", fieldExpr->field());
        }
      }
    }
    VELOX_FAIL("Field not found, " + name);
  } else if (canBeEvaluatedByCudf(expr, /*deep=*/false)) {
    // Shallow check: only verify this operation is supported
    // Children will be recursively handled by createCudfExpression
    auto node =
        createCudfExpression(expr, inputRowSchema[0], kAstEvaluatorName);
    return addPrecomputeInstructionOnSide(0, 0, name, "", node);
  } else {
    VELOX_FAIL("Unsupported expression: " + name);
  }
}

std::vector<ColumnOrView> precomputeSubexpressions(
    std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
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
          inputTableColumns,
          stream,
          cudf::get_current_device_resource_ref(),
          /*finalize=*/true);
      precomputedColumns.push_back(std::move(result));
      continue;
    }
    if (ins_name.rfind("fill", 0) == 0) {
      auto scalarIndex =
          std::stoi(ins_name.substr(5)); // "fill " is 5 characters
      auto newColumn = cudf::make_column_from_scalar(
          *static_cast<cudf::string_scalar*>(scalars[scalarIndex].get()),
          inputTableColumns[dependent_column_index]->size(),
          stream,
          cudf::get_current_device_resource_ref());
      precomputedColumns.push_back(std::move(newColumn));
    } else if (ins_name == "nested_column") {
      // Nested column already exists in input. Don't materialize.
      auto view = inputTableColumns[dependent_column_index]->view().child(
          nested_dependent_column_indices[0]);
      precomputedColumns.push_back(view);
    } else {
      VELOX_FAIL("Unsupported precompute operation " + ins_name);
    }
  }

  return precomputedColumns;
}

} // namespace

// Create tree from Expr
// and collect precompute instructions for non-ast operations
cudf::ast::expression const& createAstTree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    std::vector<PrecomputeInstruction>& precomputeInstructions) {
  AstContext context{
      tree, scalars, {inputRowSchema}, {precomputeInstructions}, expr};
  return context.pushExprToTree(expr);
}

cudf::ast::expression const& createAstTree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& leftRowSchema,
    const RowTypePtr& rightRowSchema,
    std::vector<PrecomputeInstruction>& leftPrecomputeInstructions,
    std::vector<PrecomputeInstruction>& rightPrecomputeInstructions) {
  AstContext context{
      tree,
      scalars,
      {leftRowSchema, rightRowSchema},
      {leftPrecomputeInstructions, rightPrecomputeInstructions},
      expr};
  return context.pushExprToTree(expr);
}

ASTExpression::ASTExpression(
    std::shared_ptr<velox::exec::Expr> expr,
    const RowTypePtr& inputRowSchema)
    : inputRowSchema_(inputRowSchema) {
  createAstTree(
      expr, cudfTree_, scalars_, inputRowSchema, precomputeInstructions_);
}

void ASTExpression::close() {
  cudfTree_ = {};
  scalars_.clear();
  precomputeInstructions_.clear();
}

ColumnOrView ASTExpression::eval(
    std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool finalize) {
  auto precomputedColumns = precomputeSubexpressions(
      inputTableColumns,
      precomputeInstructions_,
      scalars_,
      inputRowSchema_,
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

  if (auto colRefPtr =
          dynamic_cast<cudf::ast::column_reference const*>(&cudfTree_.back())) {
    auto columnIndex = colRefPtr->get_column_index();
    if (columnIndex < inputTableColumns.size()) {
      return inputTableColumns[columnIndex]->view();
    } else {
      // Referencing a precomputed column return as it is (view or owned)
      return std::move(
          precomputedColumns[columnIndex - inputTableColumns.size()]);
    }
  } else {
    return cudf::compute_column(
        astInputTableView, cudfTree_.back(), stream, mr);
  }
}

bool ASTExpression::canEvaluate(std::shared_ptr<velox::exec::Expr> expr) {
  return detail::isAstSupported(expr->name()) ||
      std::dynamic_pointer_cast<velox::exec::FieldReference>(expr) != nullptr;
}

bool ASTExpression::canEvaluate(const core::TypedExprPtr& expr) {
  using core::ExprKind;
  switch (expr->kind()) {
    case ExprKind::kFieldAccess:
    case ExprKind::kDereference:
    case ExprKind::kConstant:
    case ExprKind::kInput:
      return true;
    case ExprKind::kCall: {
      const auto* call =
          expr->asUnchecked<facebook::velox::core::CallTypedExpr>();
      return detail::isAstSupported(call->name());
    }
    case core::ExprKind::kCast: {
      const auto* cast = expr->asUnchecked<core::CastTypedExpr>();
      if (cast->isTryCast()) {
        return false;
      }
      return true;
    }

    default:
      return false;
  }
}

void registerAstEvaluator(int priority) {
  registerCudfExpressionEvaluator(
      kAstEvaluatorName,
      priority,
      [](const core::TypedExprPtr& typed) {
        return ASTExpression::canEvaluate(typed);
      },
      [](std::shared_ptr<velox::exec::Expr> expr) {
        return ASTExpression::canEvaluate(expr);
      },
      [](std::shared_ptr<velox::exec::Expr> expr, const RowTypePtr& row) {
        return std::make_shared<ASTExpression>(std::move(expr), row);
      },
      /*overwrite=*/false);
}

} // namespace facebook::velox::cudf_velox
