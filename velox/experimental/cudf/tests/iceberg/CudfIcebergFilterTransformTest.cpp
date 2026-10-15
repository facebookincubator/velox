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
#include "velox/common/base/tests/GTestUtils.h"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <span>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {
namespace {

using testing::ElementsAre;

constexpr auto kFoldTrue = ConstantFilterFold::kAlwaysTrue;
constexpr auto kFoldFalse = ConstantFilterFold::kAlwaysFalse;
constexpr auto kFoldUnknown = ConstantFilterFold::kUnknown;

// Returns the column indices referenced by the operands of 'operation', which
// are all expected to be column references.
std::vector<cudf::size_type> operandColumnIndices(
    const cudf::ast::operation& operation) {
  std::vector<cudf::size_type> indices;
  for (const auto& operand : operation.get_operands()) {
    const auto* column =
        dynamic_cast<const cudf::ast::column_reference*>(&operand.get());
    EXPECT_NE(column, nullptr);
    if (column != nullptr) {
      indices.push_back(column->get_column_index());
    }
  }
  return indices;
}

// Owns a two-column logical expression used across fold truth-table tests.
struct LogicalExpression {
  LogicalExpression(
      cudf::ast::ast_operator logicalOperator,
      cudf::size_type physicalColumnIndex)
      : injected(tree.push(cudf::ast::column_reference{0})),
        physical(tree.push(cudf::ast::column_reference{physicalColumnIndex})),
        expression(tree.push(
            cudf::ast::operation{logicalOperator, injected, physical})) {}

  TransformedFilter transform(ConstantFilterFold fold) const {
    return transformFilterForInjectedColumns(
        expression, std::array<cudf::size_type, 1>{0}, std::array{fold});
  }

  cudf::ast::tree tree;
  const cudf::ast::expression& injected;
  const cudf::ast::expression& physical;
  const cudf::ast::expression& expression;
};

// Owns a negated two-column logical expression used across fold tests.
struct NegatedLogicalExpression {
  explicit NegatedLogicalExpression(cudf::ast::ast_operator logicalOperator)
      : injected(tree.push(cudf::ast::column_reference{0})),
        physical(tree.push(cudf::ast::column_reference{1})),
        logical(tree.push(
            cudf::ast::operation{logicalOperator, injected, physical})),
        expression(tree.push(
            cudf::ast::operation{cudf::ast::ast_operator::NOT, logical})) {}

  TransformedFilter transform(ConstantFilterFold fold) const {
    return transformFilterForInjectedColumns(
        expression, std::array<cudf::size_type, 1>{0}, std::array{fold});
  }

  cudf::ast::tree tree;
  const cudf::ast::expression& injected;
  const cudf::ast::expression& physical;
  const cudf::ast::expression& logical;
  const cudf::ast::expression& expression;
};

} // namespace

TEST(CudfIcebergFilterTransformTest, unknownConjunctIsDeferred) {
  LogicalExpression expression{
      cudf::ast::ast_operator::NULL_LOGICAL_AND, /*physicalColumnIndex=*/2};
  const auto result = expression.transform(kFoldUnknown);

  EXPECT_FALSE(result.skipSplit);
  const auto* transformed =
      dynamic_cast<const cudf::ast::column_reference*>(result.pushedExpr);
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_column_index(), 1);
  // Only the undecided conjunct is left for the assembled table.
  EXPECT_EQ(result.deferredExpr, &expression.injected);
}

TEST(CudfIcebergFilterTransformTest, trueConjunctDropsOut) {
  LogicalExpression expression{
      cudf::ast::ast_operator::NULL_LOGICAL_AND, /*physicalColumnIndex=*/2};
  const auto result = expression.transform(kFoldTrue);

  EXPECT_FALSE(result.skipSplit);
  const auto* transformed =
      dynamic_cast<const cudf::ast::column_reference*>(result.pushedExpr);
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_column_index(), 1);
  // The pushed filter is exact, so no pass over the assembled table is needed.
  EXPECT_EQ(result.deferredExpr, nullptr);
}

TEST(CudfIcebergFilterTransformTest, falseConjunctRejectsSplit) {
  LogicalExpression expression{
      cudf::ast::ast_operator::NULL_LOGICAL_AND, /*physicalColumnIndex=*/2};
  const auto result = expression.transform(kFoldFalse);

  EXPECT_TRUE(result.skipSplit);
  EXPECT_EQ(result.pushedExpr, nullptr);
  EXPECT_EQ(result.deferredExpr, nullptr);
}

TEST(CudfIcebergFilterTransformTest, unknownDisjunctIsDeferred) {
  LogicalExpression expression{
      cudf::ast::ast_operator::NULL_LOGICAL_OR, /*physicalColumnIndex=*/1};
  const auto result = expression.transform(kFoldUnknown);

  EXPECT_EQ(result.pushedExpr, nullptr);
  // An undecided disjunct relaxes the whole disjunction.
  EXPECT_EQ(result.deferredExpr, &expression.expression);
}

TEST(CudfIcebergFilterTransformTest, falseDisjunctDropsOut) {
  LogicalExpression expression{
      cudf::ast::ast_operator::NULL_LOGICAL_OR, /*physicalColumnIndex=*/1};
  const auto result = expression.transform(kFoldFalse);

  const auto* transformed =
      dynamic_cast<const cudf::ast::column_reference*>(result.pushedExpr);
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_column_index(), 0);
  EXPECT_EQ(result.deferredExpr, nullptr);
}

TEST(CudfIcebergFilterTransformTest, trueDisjunctAcceptsEveryRow) {
  LogicalExpression expression{
      cudf::ast::ast_operator::NULL_LOGICAL_OR, /*physicalColumnIndex=*/1};
  const auto result = expression.transform(kFoldTrue);

  EXPECT_FALSE(result.skipSplit);
  EXPECT_EQ(result.pushedExpr, nullptr);
  EXPECT_EQ(result.deferredExpr, nullptr);
}

TEST(CudfIcebergFilterTransformTest, negatedConjunctionIsPushedOnceFolded) {
  NegatedLogicalExpression expression{
      cudf::ast::ast_operator::NULL_LOGICAL_AND};
  const auto result = expression.transform(kFoldTrue);

  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(result.pushedExpr);
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_operator(), cudf::ast::ast_operator::NOT);
  EXPECT_THAT(operandColumnIndices(*transformed), ElementsAre(0));
  EXPECT_EQ(result.deferredExpr, nullptr);
}

// A negation over a `kAlwaysFalse` fold must be deferred, not inverted.
TEST(CudfIcebergFilterTransformTest, negatedRejectingConjunctionIsDeferred) {
  NegatedLogicalExpression expression{
      cudf::ast::ast_operator::NULL_LOGICAL_AND};
  const auto result = expression.transform(kFoldFalse);

  EXPECT_FALSE(result.skipSplit);
  EXPECT_EQ(result.pushedExpr, nullptr);
  EXPECT_EQ(result.deferredExpr, &expression.expression);
}

// A fold is false for a NULL value too, which the predicate itself evaluates
// to NULL, so a negation of the disjunction it was dropped from is deferred.
TEST(CudfIcebergFilterTransformTest, negatedRejectingDisjunctionIsDeferred) {
  NegatedLogicalExpression expression{cudf::ast::ast_operator::NULL_LOGICAL_OR};
  const auto result = expression.transform(kFoldFalse);

  EXPECT_FALSE(result.skipSplit);
  EXPECT_EQ(result.pushedExpr, nullptr);
  EXPECT_EQ(result.deferredExpr, &expression.expression);
}

TEST(CudfIcebergFilterTransformTest, rejectingDisjunctIsStillDropped) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  const auto& firstPhysical = tree.push(cudf::ast::column_reference{2});
  const auto& disjunction = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_OR, injected, physical});
  // The inexactness survives a logical operator, which cannot tell false from
  // NULL either.
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND,
          disjunction,
          firstPhysical});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0}, std::array{kFoldFalse});

  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(result.pushedExpr);
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(
      transformed->get_operator(), cudf::ast::ast_operator::NULL_LOGICAL_AND);
  EXPECT_THAT(operandColumnIndices(*transformed), ElementsAre(0, 1));
  EXPECT_EQ(result.deferredExpr, nullptr);
}

TEST(CudfIcebergFilterTransformTest, negatedUnknownConjunctionIsDeferred) {
  NegatedLogicalExpression expression{
      cudf::ast::ast_operator::NULL_LOGICAL_AND};
  const auto result = expression.transform(kFoldUnknown);

  EXPECT_EQ(result.pushedExpr, nullptr);
  EXPECT_EQ(result.deferredExpr, &expression.expression);
}

// A fold is the value of the whole predicate on the column, so it is applied
// once, at the highest expression referencing that column and nothing else.
// Applying it to each reference separately would let the negation invert it.
TEST(CudfIcebergFilterTransformTest, negatedRangeOnInjectedColumnFoldsOnce) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  cudf::numeric_scalar<int32_t> lowValue{1};
  cudf::numeric_scalar<int32_t> highValue{9};
  const auto& low = tree.push(cudf::ast::literal{lowValue});
  const auto& high = tree.push(cudf::ast::literal{highValue});
  const auto& lowerBound = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::GREATER_EQUAL, injected, low});
  const auto& upperBound = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::LESS_EQUAL, injected, high});
  const auto& range = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND, lowerBound, upperBound});
  const auto& expression =
      tree.push(cudf::ast::operation{cudf::ast::ast_operator::NOT, range});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0}, std::array{kFoldFalse});

  EXPECT_TRUE(result.skipSplit);
  EXPECT_EQ(result.pushedExpr, nullptr);
  EXPECT_EQ(result.deferredExpr, nullptr);
}

TEST(CudfIcebergFilterTransformTest, comparisonOnInjectedColumnFolds) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  cudf::numeric_scalar<int32_t> literalValue{5};
  const auto& literal = tree.push(cudf::ast::literal{literalValue});
  const auto& comparison = tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::EQUAL, injected, literal});
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND, comparison, physical});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0}, std::array{kFoldTrue});

  const auto* transformed =
      dynamic_cast<const cudf::ast::column_reference*>(result.pushedExpr);
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_column_index(), 0);
  EXPECT_EQ(result.deferredExpr, nullptr);
}

TEST(CudfIcebergFilterTransformTest, negatedInjectedOnlyPredicateFolds) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  cudf::numeric_scalar<int32_t> literalValue{5};
  const auto& literal = tree.push(cudf::ast::literal{literalValue});
  const auto& comparison = tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::EQUAL, injected, literal});
  const auto& negation =
      tree.push(cudf::ast::operation{cudf::ast::ast_operator::NOT, comparison});
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND, negation, physical});

  // The fold is the value of the negation, so the split is rejected rather
  // than the negation being inverted a second time.
  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0}, std::array{kFoldFalse});

  EXPECT_TRUE(result.skipSplit);
}

TEST(CudfIcebergFilterTransformTest, foldedConjunctInsideOrIsPushed) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& firstPhysical = tree.push(cudf::ast::column_reference{1});
  const auto& secondPhysical = tree.push(cudf::ast::column_reference{2});
  const auto& conjunction = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND, injected, firstPhysical});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_OR,
          conjunction,
          secondPhysical});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0}, std::array{kFoldTrue});

  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(result.pushedExpr);
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(
      transformed->get_operator(), cudf::ast::ast_operator::NULL_LOGICAL_OR);
  EXPECT_THAT(operandColumnIndices(*transformed), ElementsAre(0, 1));
  EXPECT_EQ(result.deferredExpr, nullptr);
}

TEST(CudfIcebergFilterTransformTest, unknownConjunctInsideOrDefersTheOr) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& firstPhysical = tree.push(cudf::ast::column_reference{1});
  const auto& secondPhysical = tree.push(cudf::ast::column_reference{2});
  const auto& conjunction = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND, injected, firstPhysical});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_OR,
          conjunction,
          secondPhysical});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0}, std::array{kFoldUnknown});

  // The relaxed conjunction accepts rows the disjunction rejects, which only
  // re-applying the whole disjunction can recover.
  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(result.pushedExpr);
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(
      transformed->get_operator(), cudf::ast::ast_operator::NULL_LOGICAL_OR);
  EXPECT_THAT(operandColumnIndices(*transformed), ElementsAre(0, 1));
  EXPECT_EQ(result.deferredExpr, &expression);
}

TEST(CudfIcebergFilterTransformTest, foldsEachInjectedColumnSeparately) {
  cudf::ast::tree tree;
  const auto& firstInjected = tree.push(cudf::ast::column_reference{2});
  const auto& secondInjected = tree.push(cudf::ast::column_reference{0});
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  const auto& conjunction = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND,
          firstInjected,
          secondInjected});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND, conjunction, physical});

  const auto result = transformFilterForInjectedColumns(
      expression,
      std::array<cudf::size_type, 2>{0, 2},
      std::array{kFoldTrue, kFoldUnknown});

  const auto* transformed =
      dynamic_cast<const cudf::ast::column_reference*>(result.pushedExpr);
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_column_index(), 0);
  // The column that folded true drops out, the undecided one is deferred.
  EXPECT_EQ(result.deferredExpr, &firstInjected);
}

TEST(CudfIcebergFilterTransformTest, nestedLogicalAndRetainsOr) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& firstPhysical = tree.push(cudf::ast::column_reference{1});
  const auto& secondPhysical = tree.push(cudf::ast::column_reference{2});
  const auto& disjunction = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_OR,
          firstPhysical,
          secondPhysical});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND, injected, disjunction});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0}, std::array{kFoldUnknown});

  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(result.pushedExpr);
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(
      transformed->get_operator(), cudf::ast::ast_operator::NULL_LOGICAL_OR);
  EXPECT_THAT(operandColumnIndices(*transformed), ElementsAre(0, 1));
  EXPECT_EQ(result.deferredExpr, &injected);
}

TEST(CudfIcebergFilterTransformTest, rebasesMultiplePhysicalColumns) {
  cudf::ast::tree tree;
  const auto& firstPhysical = tree.push(cudf::ast::column_reference{1});
  const auto& secondPhysical = tree.push(cudf::ast::column_reference{3});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND,
          firstPhysical,
          secondPhysical});

  const auto result = transformFilterForInjectedColumns(
      expression,
      std::array<cudf::size_type, 2>{0, 2},
      std::array{kFoldUnknown, kFoldUnknown});

  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(result.pushedExpr);
  ASSERT_NE(transformed, nullptr);
  EXPECT_THAT(operandColumnIndices(*transformed), ElementsAre(0, 1));
  EXPECT_EQ(result.deferredExpr, nullptr);
}

TEST(CudfIcebergFilterTransformTest, negatedPhysicalExpressionIsPushed) {
  cudf::ast::tree tree;
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  cudf::numeric_scalar<int32_t> literalValue{5};
  const auto& literal = tree.push(cudf::ast::literal{literalValue});
  const auto& comparison = tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::EQUAL, physical, literal});
  const auto& expression =
      tree.push(cudf::ast::operation{cudf::ast::ast_operator::NOT, comparison});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0}, std::array{kFoldUnknown});

  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(result.pushedExpr);
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_operator(), cudf::ast::ast_operator::NOT);
  const auto* transformedComparison = dynamic_cast<const cudf::ast::operation*>(
      &transformed->get_operands()[0].get());
  ASSERT_NE(transformedComparison, nullptr);
  EXPECT_EQ(
      transformedComparison->get_operator(), cudf::ast::ast_operator::EQUAL);
  const auto* column = dynamic_cast<const cudf::ast::column_reference*>(
      &transformedComparison->get_operands()[0].get());
  ASSERT_NE(column, nullptr);
  EXPECT_EQ(column->get_column_index(), 0);
  EXPECT_EQ(result.deferredExpr, nullptr);
}

TEST(
    CudfIcebergFilterTransformTest,
    retainedDecimalRequiresSplitSpecificTypes) {
  cudf::ast::tree tree;
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  cudf::fixed_point_scalar<numeric::decimal64> literalValue{
      500, numeric::scale_type{-2}};
  const auto& literal = tree.push(cudf::ast::literal{literalValue});
  const auto& expression = tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::EQUAL, physical, literal});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0}, std::array{kFoldUnknown});

  EXPECT_TRUE(result.requiresSplitSpecificDecimalTypes);
}

TEST(
    CudfIcebergFilterTransformTest,
    foldedDecimalDoesNotRequireSplitSpecificTypes) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  cudf::fixed_point_scalar<numeric::decimal64> decimalValue{
      500, numeric::scale_type{-2}};
  const auto& decimalLiteral = tree.push(cudf::ast::literal{decimalValue});
  const auto& decimalComparison = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::EQUAL, injected, decimalLiteral});

  const auto& physical = tree.push(cudf::ast::column_reference{1});
  cudf::numeric_scalar<int32_t> integerValue{5};
  const auto& integerLiteral = tree.push(cudf::ast::literal{integerValue});
  const auto& integerComparison = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::EQUAL, physical, integerLiteral});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND,
          decimalComparison,
          integerComparison});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0}, std::array{kFoldTrue});

  EXPECT_FALSE(result.requiresSplitSpecificDecimalTypes);
  EXPECT_EQ(result.deferredExpr, nullptr);
}

TEST(
    CudfIcebergFilterTransformTest,
    nonLogicalExpressionReferencingInjectedColumnIsNotPushed) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  const auto& expression = tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::ADD, injected, physical});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0}, std::array{kFoldTrue});

  EXPECT_EQ(result.pushedExpr, nullptr);
  EXPECT_EQ(result.deferredExpr, &expression);
}

TEST(CudfIcebergFilterTransformTest, isNullOnInjectedColumnFolds) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{1});
  const auto& expression = tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::IS_NULL, injected});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{1}, std::array{kFoldTrue});

  EXPECT_FALSE(result.skipSplit);
  EXPECT_EQ(result.pushedExpr, nullptr);
  EXPECT_EQ(result.deferredExpr, nullptr);
}

TEST(CudfIcebergFilterTransformTest, rebasePhysicalColumn) {
  cudf::ast::tree tree;
  const auto& expression = tree.push(cudf::ast::column_reference{2});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0}, std::array{kFoldUnknown});

  const auto* transformed =
      dynamic_cast<const cudf::ast::column_reference*>(result.pushedExpr);
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_column_index(), 1);
}

TEST(CudfIcebergFilterTransformTest, trailingInjectedColumnDoesNotRebase) {
  cudf::ast::tree tree;
  const auto& expression = tree.push(cudf::ast::column_reference{1});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{2}, std::array{kFoldUnknown});

  // Nothing changed, so the input expression is pushed as is.
  EXPECT_EQ(result.pushedExpr, &expression);
}

TEST(CudfIcebergFilterTransformTest, nullPropagatingOperatorFails) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::LOGICAL_OR, injected, physical});

  // Folding this to TRUE would keep the rows a null-propagating OR rejects.
  VELOX_ASSERT_THROW(
      (transformFilterForInjectedColumns(
          expression,
          std::array<cudf::size_type, 1>{0},
          std::array{kFoldTrue})),
      "Iceberg subfield filter must use the null-aware logical operators");
}

TEST(CudfIcebergFilterTransformTest, emptyInjectedColumnIndicesFail) {
  cudf::ast::tree tree;
  const auto& expression = tree.push(cudf::ast::column_reference{2});

  VELOX_ASSERT_THROW(
      (transformFilterForInjectedColumns(
          expression,
          std::span<const cudf::size_type>{},
          std::span<const ConstantFilterFold>{})),
      "Injected column indices to cuDF filter transformer must be non-empty, ascending, and unique");
}

TEST(CudfIcebergFilterTransformTest, unsortedInjectedColumnIndicesFail) {
  cudf::ast::tree tree;
  const auto& expression = tree.push(cudf::ast::column_reference{3});

  VELOX_ASSERT_THROW(
      (transformFilterForInjectedColumns(
          expression,
          std::array<cudf::size_type, 2>{2, 0},
          std::array{kFoldUnknown, kFoldUnknown})),
      "Injected column indices to cuDF filter transformer must be non-empty, ascending, and unique");
}

TEST(CudfIcebergFilterTransformTest, missingFoldsFail) {
  cudf::ast::tree tree;
  const auto& expression = tree.push(cudf::ast::column_reference{3});

  VELOX_ASSERT_THROW(
      (transformFilterForInjectedColumns(
          expression,
          std::array<cudf::size_type, 2>{0, 2},
          std::array{kFoldUnknown})),
      "cuDF filter transformer needs one fold per injected column");
}

TEST(CudfIcebergFilterTransformTest, columnNameReferenceFails) {
  cudf::ast::tree tree;
  const auto& expression = tree.push(cudf::ast::column_name_reference{"c0"});

  VELOX_ASSERT_THROW(
      (transformFilterForInjectedColumns(
          expression,
          std::array<cudf::size_type, 1>{0},
          std::array{kFoldUnknown})),
      "Iceberg subfield filter must use column index references");
}
} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
