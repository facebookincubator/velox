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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergExpressionTransformers.hpp"

#include "velox/common/base/Exceptions.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

bool isLogicalAnd(cudf::ast::ast_operator op) {
  return op == cudf::ast::ast_operator::LOGICAL_AND or
      op == cudf::ast::ast_operator::NULL_LOGICAL_AND;
}

bool isLogicalOr(cudf::ast::ast_operator op) {
  return op == cudf::ast::ast_operator::LOGICAL_OR or
      op == cudf::ast::ast_operator::NULL_LOGICAL_OR;
}

} // namespace

CudfIcebergExpressionTransformer::CudfIcebergExpressionTransformer(
    cudf::ast::expression const& expression,
    std::vector<cudf::size_type> injectedColumnIndices)
    : injectedColumnIndices_(std::move(injectedColumnIndices)) {
  std::sort(injectedColumnIndices_.begin(), injectedColumnIndices_.end());
  injectedColumnIndices_.erase(
      std::unique(injectedColumnIndices_.begin(), injectedColumnIndices_.end()),
      injectedColumnIndices_.end());
  result_ = transform(expression);
}

CudfIcebergExpressionTransformer::Result
CudfIcebergExpressionTransformer::transform(
    cudf::ast::expression const& expression) {
  expression.accept(*this);
  return result_;
}

std::reference_wrapper<cudf::ast::expression const>
CudfIcebergExpressionTransformer::visit(cudf::ast::literal const& expression) {
  result_ = {&expression, false};
  return expression;
}

std::reference_wrapper<cudf::ast::expression const>
CudfIcebergExpressionTransformer::visit(
    cudf::ast::column_reference const& expression) {
  const auto columnIndex = expression.get_column_index();
  const auto iter = std::lower_bound(
      injectedColumnIndices_.begin(),
      injectedColumnIndices_.end(),
      columnIndex);
  if (iter != injectedColumnIndices_.end() and *iter == columnIndex) {
    referencesInjectedColumn_ = true;
    changed_ = true;
    result_ = {nullptr, true};
    return expression;
  }

  const auto numPrecedingInjectedColumns = static_cast<cudf::size_type>(
      std::distance(injectedColumnIndices_.begin(), iter));
  const auto rebasedIndex = columnIndex - numPrecedingInjectedColumns;
  if (rebasedIndex == columnIndex) {
    result_ = {&expression, false};
    return expression;
  }

  changed_ = true;
  const auto& transformed = transformedTree_.push(
      cudf::ast::column_reference{
          static_cast<cudf::size_type>(rebasedIndex),
          expression.get_table_source()});
  result_ = {&transformed, false};
  return transformed;
}

std::reference_wrapper<cudf::ast::expression const>
CudfIcebergExpressionTransformer::visit(
    cudf::ast::operation const& expression) {
  const auto& operands = expression.get_operands();
  VELOX_CHECK(
      operands.size() == 1 or operands.size() == 2,
      "Expected a unary or binary cuDF AST operation");

  std::vector<Result> transformedOperands;
  transformedOperands.reserve(operands.size());
  for (const auto& operand : operands) {
    transformedOperands.push_back(transform(operand.get()));
  }

  const auto op = expression.get_operator();
  const auto wasRelaxed = std::any_of(
      transformedOperands.begin(),
      transformedOperands.end(),
      [](const auto& operand) { return operand.wasRelaxed; });

  if (isLogicalAnd(op)) {
    const auto* lhs = transformedOperands[0].expression;
    const auto* rhs = transformedOperands[1].expression;
    if (lhs == nullptr or rhs == nullptr) {
      result_ = {lhs == nullptr ? rhs : lhs, wasRelaxed};
      return result_.expression == nullptr ? expression : *result_.expression;
    }
  } else if (isLogicalOr(op)) {
    if (transformedOperands[0].expression == nullptr or
        transformedOperands[1].expression == nullptr) {
      result_ = {nullptr, true};
      return expression;
    }
  } else if (wasRelaxed) {
    result_ = {nullptr, true};
    return expression;
  }

  const auto& transformed = operands.size() == 1
      ? transformedTree_.push(
            cudf::ast::operation{op, *transformedOperands[0].expression})
      : transformedTree_.push(
            cudf::ast::operation{
                op,
                *transformedOperands[0].expression,
                *transformedOperands[1].expression});
  result_ = {&transformed, wasRelaxed};
  return transformed;
}

std::reference_wrapper<cudf::ast::expression const>
CudfIcebergExpressionTransformer::visit(
    cudf::ast::column_name_reference const& expression) {
  VELOX_FAIL("Iceberg subfield filter must use column index references");
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
