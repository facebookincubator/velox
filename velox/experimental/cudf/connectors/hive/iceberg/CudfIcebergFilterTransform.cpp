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
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

// Folds the predicates over injected columns into their known value, rebases
// the remaining column indices past the injected columns, and collects
// whatever the rebased filter no longer enforces.
class InjectedColumnFilterTransformer
    : private cudf::ast::detail::expression_transformer {
 public:
  InjectedColumnFilterTransformer(
      const cudf::ast::expression& filter,
      std::span<const cudf::size_type> sortedInjectedColumnIndices,
      std::span<const ConstantFilterFold> injectedColumnFolds)
      : injectedColumnIndices_(sortedInjectedColumnIndices),
        injectedColumnFolds_(injectedColumnFolds) {
    filter.accept(*this);
    fold(current_);
  }

  // Returns the transformed filter, handing over the nodes created while
  // transforming.
  TransformedFilter transformedFilter() && {
    return TransformedFilter{
        .nodes = std::move(nodes_),
        .pushedExpr = current_.pushed,
        .deferredExpr = current_.deferred,
        .skipSplit = current_.constant == false,
        .requiresSplitSpecificDecimalTypes =
            current_.requiresSplitSpecificDecimalTypes};
  }

 private:
  // Result of subexpression transformation. Once transformed, it is in exactly
  // one of four states:
  //
  //   constant set          the subexpression is known true or false
  //   pushed, no deferred   the pushed filter enforces it exactly
  //   pushed and deferred   the pushed filter enforces part of it
  //   deferred only         nothing of it can be pushed
  struct Transformed {
    // The subexpression this result was derived from.
    const cudf::ast::expression* original{nullptr};

    // Result over the columns the parquet reader projects. Null when nothing
    // of the subexpression can be pushed, which relaxes the pushed filter.
    const cudf::ast::expression* pushed{nullptr};

    // What `pushed` does not enforce, in the input filter's index space. Null
    // when `pushed` is exact.
    const cudf::ast::expression* deferred{nullptr};

    // Value of the subexpression if it folded to a constant.
    std::optional<bool> constant{std::nullopt};

    // The single injected column the subexpression references, while its fold
    // is still to be applied. See `fold()`.
    std::optional<size_t> pendingColumn{std::nullopt};

    // Whether the subexpression is a literal. Literals are neutral when
    // deciding whether a subtree references a single injected column only.
    bool isLiteral{false};

    // Whether `pushed` retains a decimal literal whose storage width must
    // match the current split.
    bool requiresSplitSpecificDecimalTypes{false};

    // Whether the pushed expression is false in rows where the subexpression
    // is NULL. Dropping a disjunct folded to false loses that distinction.
    bool valueInexact{false};
  };

  // Applies the fold of the injected column a subexpression references.
  //
  // Deliberately not applied as soon as an injected column reference is
  // visited: a fold is the value of the whole predicate the column appears in,
  // for example of `"a" <= part AND part <= "z"` rather than of either
  // comparison on its own. It is therefore applied at the highest node that
  // references that column and nothing else.
  void fold(Transformed& transformed) {
    if (not transformed.pendingColumn.has_value()) {
      return;
    }
    const auto columnFold = injectedColumnFolds_[*transformed.pendingColumn];
    transformed.pendingColumn.reset();
    switch (columnFold) {
      case ConstantFilterFold::kAlwaysTrue:
        transformed.constant = true;
        return;
      case ConstantFilterFold::kAlwaysFalse:
        transformed.constant = false;
        return;
      case ConstantFilterFold::kUnknown:
        // The columns it references are not read from the data file, so it can
        // only be applied to the assembled table.
        transformed.deferred = transformed.original;
        return;
    }
  }

  // Returns the injected column the operands reference, when they reference
  // that one and no other.
  static std::optional<size_t> pendingColumnOf(
      std::span<const Transformed> operands) {
    std::optional<size_t> pendingColumn;
    for (const auto& operand : operands) {
      if (operand.isLiteral) {
        continue;
      }
      if (not operand.pendingColumn.has_value() or
          (pendingColumn.has_value() and
           *pendingColumn != *operand.pendingColumn)) {
        return std::nullopt;
      }
      pendingColumn = operand.pendingColumn;
    }
    return pendingColumn;
  }

  static Transformed constantResult(
      const cudf::ast::expression& expr,
      bool value) {
    return Transformed{.original = &expr, .constant = value};
  }

  // Result for a subexpression that cannot be pushed at all and must be
  // deferred to post-read (assembled) table.
  static Transformed deferredResult(const cudf::ast::expression& expr) {
    return Transformed{.original = &expr, .deferred = &expr};
  }

  std::reference_wrapper<const cudf::ast::expression> visit(
      const cudf::ast::literal& expr) override {
    current_ = Transformed{
        .original = &expr,
        .pushed = &expr,
        .isLiteral = true,
        .requiresSplitSpecificDecimalTypes =
            cudf::is_fixed_point(expr.get_data_type())};
    return expr;
  }

  std::reference_wrapper<const cudf::ast::expression> visit(
      const cudf::ast::column_reference& expr) override {
    const auto columnIndex = expr.get_column_index();
    const auto iter = std::lower_bound(
        injectedColumnIndices_.begin(),
        injectedColumnIndices_.end(),
        columnIndex);
    const auto numPrecedingInjectedColumns = static_cast<size_t>(
        std::distance(injectedColumnIndices_.begin(), iter));

    if (iter != injectedColumnIndices_.end() and *iter == columnIndex) {
      current_ = Transformed{
          .original = &expr, .pendingColumn = numPrecedingInjectedColumns};
      return expr;
    }

    const auto& rebased = numPrecedingInjectedColumns == 0
        ? expr
        : nodes_.push(
              cudf::ast::column_reference{
                  columnIndex -
                      static_cast<cudf::size_type>(numPrecedingInjectedColumns),
                  expr.get_table_source()});
    current_ = Transformed{.original = &expr, .pushed = &rebased};
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

    // Keep climbing while the subtree stays within a single injected column,
    // so that its fold is applied to the whole predicate at once.
    if (const auto pendingColumn = pendingColumnOf(transformedOperands)) {
      current_ = Transformed{.original = &expr, .pendingColumn = pendingColumn};
      return expr;
    }

    for (auto& operand : transformedOperands) {
      fold(operand);
    }

    // Folding uses Kleene operators emitted by `createAstFromSubfieldFilter`.
    // Under the null-propagating ones `X OR TRUE` is NULL rather than TRUE
    // wherever X is NULL, so dropping a filter folded to TRUE would keep the
    // rows they reject.
    const auto op = expr.get_operator();
    VELOX_CHECK(
        op != cudf::ast::ast_operator::LOGICAL_AND and
            op != cudf::ast::ast_operator::LOGICAL_OR,
        "Iceberg subfield filter must use the null-aware logical operators");

    if (op == cudf::ast::ast_operator::NULL_LOGICAL_AND) {
      VELOX_CHECK_EQ(
          operands.size(), 2, "Expected a binary cuDF AST logical AND");
      current_ = transformLogicalAnd(expr, op, transformedOperands);
    } else if (op == cudf::ast::ast_operator::NULL_LOGICAL_OR) {
      VELOX_CHECK_EQ(
          operands.size(), 2, "Expected a binary cuDF AST logical OR");
      current_ = transformLogicalOr(expr, op, transformedOperands);
    } else {
      current_ = transformOperation(expr, op, transformedOperands);
    }
    return current_.pushed == nullptr ? expr : *current_.pushed;
  }

  // Folds the operands with a known value out of a logical operation.
  // 'shortCircuitValue' is the value that decides it: false short-circuits an
  // AND, true short-circuits an OR.
  //
  // Returns that value if an operand has it, its negation if every operand
  // dropped out, or the lone survivor. Returns nothing when two survive, which
  // 'retained' then holds.
  static std::optional<Transformed> applyConstants(
      const cudf::ast::expression& expr,
      bool shortCircuitValue,
      std::span<const Transformed> operands,
      std::vector<const Transformed*>& retained) {
    for (const auto& operand : operands) {
      if (operand.constant == shortCircuitValue) {
        return constantResult(expr, shortCircuitValue);
      }
      if (not operand.constant.has_value()) {
        retained.push_back(&operand);
      }
    }
    if (retained.empty()) {
      return constantResult(expr, not shortCircuitValue);
    }
    if (retained.size() == 1) {
      auto result = *retained.front();
      result.original = &expr;
      result.valueInexact = result.valueInexact or shortCircuitValue;
      return result;
    }
    return std::nullopt;
  }

  // Transforms a logical AND. A `false` operand rejects the whole split and a
  // `true` drops out, and the remaining ones can be pushed and deferred
  // independently of each other.
  Transformed transformLogicalAnd(
      const cudf::ast::expression& expr,
      cudf::ast::ast_operator op,
      std::span<const Transformed> operands) {
    std::vector<const Transformed*> retained;
    if (auto decided = applyConstants(expr, false, operands, retained)) {
      return *decided;
    }

    const auto& lhs = *retained[0];
    const auto& rhs = *retained[1];
    return Transformed{
        .original = &expr,
        .pushed = combine(op, lhs.pushed, rhs.pushed),
        .deferred = combine(op, lhs.deferred, rhs.deferred),
        .requiresSplitSpecificDecimalTypes =
            lhs.requiresSplitSpecificDecimalTypes or
            rhs.requiresSplitSpecificDecimalTypes,
        .valueInexact = lhs.valueInexact or rhs.valueInexact};
  }

  // Transforms a logical OR. A `true` operand accepts the whole split and a
  // `false` drops out, but an operand that is only partly pushed relaxes the OR
  // as a whole, which only re-applying all of it can recover.
  Transformed transformLogicalOr(
      const cudf::ast::expression& expr,
      cudf::ast::ast_operator op,
      std::span<const Transformed> operands) {
    std::vector<const Transformed*> retained;
    if (auto decided = applyConstants(expr, true, operands, retained)) {
      return *decided;
    }

    const auto& lhs = *retained[0];
    const auto& rhs = *retained[1];
    if (lhs.pushed == nullptr or rhs.pushed == nullptr) {
      return deferredResult(expr);
    }
    const auto isExact = lhs.deferred == nullptr and rhs.deferred == nullptr;
    return Transformed{
        .original = &expr,
        .pushed = combine(op, lhs.pushed, rhs.pushed),
        .deferred = isExact ? nullptr : &expr,
        .requiresSplitSpecificDecimalTypes =
            lhs.requiresSplitSpecificDecimalTypes or
            rhs.requiresSplitSpecificDecimalTypes,
        .valueInexact = lhs.valueInexact or rhs.valueInexact};
  }

  // Transforms any other operation, `NOT` and the comparisons in particular.
  // None of them tolerate a relaxed or `false` operand where the original is
  // NULL. Defer the whole operation unless every operand was pushed exactly to
  // avoid false negatives.
  Transformed transformOperation(
      const cudf::ast::expression& expr,
      cudf::ast::ast_operator op,
      std::span<const Transformed> operands) {
    for (const auto& operand : operands) {
      if (operand.pushed == nullptr or operand.deferred != nullptr or
          operand.valueInexact) {
        return deferredResult(expr);
      }
    }

    const auto& transformed = operands.size() == 1
        ? nodes_.push(cudf::ast::operation{op, *operands[0].pushed})
        : nodes_.push(
              cudf::ast::operation{
                  op, *operands[0].pushed, *operands[1].pushed});
    return Transformed{
        .original = &expr,
        .pushed = &transformed,
        .requiresSplitSpecificDecimalTypes = std::any_of(
            operands.begin(), operands.end(), [](const auto& operand) {
              return operand.requiresSplitSpecificDecimalTypes;
            })};
  }

  // Combines two operands of a logical operation, either of which may be
  // absent.
  const cudf::ast::expression* combine(
      cudf::ast::ast_operator op,
      const cudf::ast::expression* lhs,
      const cudf::ast::expression* rhs) {
    if (lhs == nullptr or rhs == nullptr) {
      return lhs == nullptr ? rhs : lhs;
    }
    return &nodes_.push(cudf::ast::operation{op, *lhs, *rhs});
  }

  std::reference_wrapper<const cudf::ast::expression> visit(
      const cudf::ast::column_name_reference& /*expr*/) override {
    VELOX_FAIL("Iceberg subfield filter must use column index references");
  }

  // Owns the expression nodes created while transforming.
  cudf::ast::tree nodes_;
  const std::span<const cudf::size_type> injectedColumnIndices_;
  const std::span<const ConstantFilterFold> injectedColumnFolds_;

  // Result of the last visited subexpression. visit() returns a reference,
  // which cannot express a folded or dropped subexpression, so its return
  // value is unused and results are carried here instead.
  Transformed current_;
};

} // namespace

TransformedFilter transformFilterForInjectedColumns(
    const cudf::ast::expression& filter,
    std::span<const cudf::size_type> sortedInjectedColumnIndices,
    std::span<const ConstantFilterFold> injectedColumnFolds) {
  // Ensure the injected column indices are non-empty, ascending, and unique.
  VELOX_CHECK(
      not sortedInjectedColumnIndices.empty() and
          std::adjacent_find(
              sortedInjectedColumnIndices.begin(),
              sortedInjectedColumnIndices.end(),
              std::greater_equal<cudf::size_type>{}) ==
              sortedInjectedColumnIndices.end(),
      "Injected column indices to cuDF filter transformer must be non-empty, ascending, and unique");
  VELOX_CHECK_EQ(
      injectedColumnFolds.size(),
      sortedInjectedColumnIndices.size(),
      "cuDF filter transformer needs one fold per injected column");

  return InjectedColumnFilterTransformer(
             filter, sortedInjectedColumnIndices, injectedColumnFolds)
      .transformedFilter();
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
