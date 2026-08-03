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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergFilterTransform.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/ast/detail/expression_transformer.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

// Checks if the operator is a logical AND.
bool isLogicalAnd(cudf::ast::ast_operator op) {
  return op == cudf::ast::ast_operator::LOGICAL_AND or
      op == cudf::ast::ast_operator::NULL_LOGICAL_AND;
}

// Checks if the operator is a logical OR.
bool isLogicalOr(cudf::ast::ast_operator op) {
  return op == cudf::ast::ast_operator::LOGICAL_OR or
      op == cudf::ast::ast_operator::NULL_LOGICAL_OR;
}

// Drops the predicates that reference injected columns and rebases the
// remaining column indices past them.
class InjectedColumnFilterTransformer
    : private cudf::ast::detail::expression_transformer {
 public:
  InjectedColumnFilterTransformer(
      const cudf::ast::expression& filter,
      std::span<const cudf::size_type> sortedInjectedColumnIndices)
      : injectedColumnIndices_(sortedInjectedColumnIndices) {
    filter.accept(*this);
  }

  // Returns the transformed filter, handing over the nodes created while
  // transforming.
  TransformedFilter transformedFilter() && {
    return TransformedFilter{
        std::move(nodes_), current_.expr, referencesInjectedColumn_};
  }

 private:
  // Struct to hold the result of a subexpression transformation.
  struct Transformed {
    // Transformed expr, nullptr when it evaluates to always true.
    const cudf::ast::expression* expr;
    // Whether the transformed expr was relaxed, meaning it accepts rows the
    // input filter rejects.
    bool wasRelaxed;
  };

  std::reference_wrapper<const cudf::ast::expression> visit(
      const cudf::ast::literal& expr) override {
    current_ = {&expr, false};
    return expr;
  }

  std::reference_wrapper<const cudf::ast::expression> visit(
      const cudf::ast::column_reference& expr) override {
    const auto columnIndex = expr.get_column_index();
    const auto iter = std::lower_bound(
        injectedColumnIndices_.begin(),
        injectedColumnIndices_.end(),
        columnIndex);
    if (iter != injectedColumnIndices_.end() and *iter == columnIndex) {
      referencesInjectedColumn_ = true;
      current_ = {nullptr, true};
      return expr;
    }

    const auto numPrecedingInjectedColumns = static_cast<cudf::size_type>(
        std::distance(injectedColumnIndices_.begin(), iter));
    if (numPrecedingInjectedColumns == 0) {
      current_ = {&expr, false};
      return expr;
    }

    const auto& rebased = nodes_.push(
        cudf::ast::column_reference{
            columnIndex - numPrecedingInjectedColumns,
            expr.get_table_source()});
    current_ = {&rebased, false};
    return rebased;
  }

  std::reference_wrapper<const cudf::ast::expression> visit(
      const cudf::ast::operation& expr) override {
    const auto& operands = expr.get_operands();
    VELOX_CHECK(
        operands.size() == 1 or operands.size() == 2,
        "Expected a unary or binary cuDF AST operation. Operands: {}",
        operands.size());

    std::vector<Transformed> transformedOperands;
    transformedOperands.reserve(operands.size());
    for (const auto& operand : operands) {
      operand.get().accept(*this);
      transformedOperands.push_back(current_);
    }

    const auto op = expr.get_operator();
    const auto wasRelaxed = std::any_of(
        transformedOperands.begin(),
        transformedOperands.end(),
        [](const auto& operand) { return operand.wasRelaxed; });

    if (isLogicalAnd(op)) {
      VELOX_CHECK_EQ(
          operands.size(), 2, "Expected a binary cuDF AST logical AND");
      // A dropped conjunct is always true, so the conjunction reduces to the
      // other operand, and to always true when both are dropped.
      const auto* lhs = transformedOperands[0].expr;
      const auto* rhs = transformedOperands[1].expr;
      if (lhs == nullptr or rhs == nullptr) {
        current_ = {lhs == nullptr ? rhs : lhs, wasRelaxed};
        return current_.expr == nullptr ? expr : *current_.expr;
      }
    } else if (isLogicalOr(op)) {
      VELOX_CHECK_EQ(
          operands.size(), 2, "Expected a binary cuDF AST logical OR");
      // A dropped disjunct is always true, and so is the disjunction.
      if (transformedOperands[0].expr == nullptr or
          transformedOperands[1].expr == nullptr) {
        current_ = {nullptr, true};
        return expr;
      }
    } else if (wasRelaxed) {
      // Any other operator, `NOT` in particular, can turn a relaxed operand
      // into a stricter predicate, so drop the whole subexpression.
      current_ = {nullptr, true};
      return expr;
    }

    const auto& transformed = operands.size() == 1
        ? nodes_.push(cudf::ast::operation{op, *transformedOperands[0].expr})
        : nodes_.push(
              cudf::ast::operation{
                  op,
                  *transformedOperands[0].expr,
                  *transformedOperands[1].expr});
    current_ = {&transformed, wasRelaxed};
    return transformed;
  }

  std::reference_wrapper<const cudf::ast::expression> visit(
      const cudf::ast::column_name_reference& /*expr*/) override {
    VELOX_FAIL("Iceberg subfield filter must use column index references");
  }

  // Owns the expression nodes created while transforming.
  cudf::ast::tree nodes_;
  const std::span<const cudf::size_type> injectedColumnIndices_;
  bool referencesInjectedColumn_{false};

  // Result of the last visited subexpression. visit() returns a reference,
  // which cannot express a dropped subexpression, so its return value is unused
  // and results are carried here instead.
  Transformed current_{nullptr, false};
};

} // namespace

TransformedFilter transformFilterForInjectedColumns(
    const cudf::ast::expression& filter,
    std::span<const cudf::size_type> sortedInjectedColumnIndices) {
  // Nothing to drop or rebase, so return the input filter as is.
  if (sortedInjectedColumnIndices.empty()) {
    return TransformedFilter{
        cudf::ast::tree{}, &filter, /*referencesInjectedColumn=*/false};
  }

  // Ensure the injected column indices are ascending and unique.
  VELOX_CHECK(
      std::adjacent_find(
          sortedInjectedColumnIndices.begin(),
          sortedInjectedColumnIndices.end(),
          std::greater_equal<cudf::size_type>{}) ==
          sortedInjectedColumnIndices.end(),
      "Injected column indices must be ascending and unique");

  return InjectedColumnFilterTransformer(filter, sortedInjectedColumnIndices)
      .transformedFilter();
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
