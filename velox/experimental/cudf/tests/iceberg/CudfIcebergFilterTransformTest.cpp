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

TEST(CudfIcebergFilterTransformTest, logicalAnd) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& physical = tree.push(cudf::ast::column_reference{2});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND, injected, physical});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0});

  EXPECT_TRUE(result.referencesInjectedColumn());
  const auto* transformed =
      dynamic_cast<const cudf::ast::column_reference*>(result.expr());
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_column_index(), 1);
}

TEST(CudfIcebergFilterTransformTest, logicalOr) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_OR, injected, physical});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0});

  EXPECT_TRUE(result.referencesInjectedColumn());
  EXPECT_EQ(result.expr(), nullptr);
}

TEST(CudfIcebergFilterTransformTest, negatedDroppedExpression) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  const auto& conjunction = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND, injected, physical});
  const auto& expression = tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::NOT, conjunction});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0});

  EXPECT_TRUE(result.referencesInjectedColumn());
  EXPECT_EQ(result.expr(), nullptr);
}

TEST(CudfIcebergFilterTransformTest, nestedLogicalAndRetainsOr) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& firstPhysical = tree.push(cudf::ast::column_reference{1});
  const auto& secondPhysical = tree.push(cudf::ast::column_reference{2});
  const auto& disjunction = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::LOGICAL_OR, firstPhysical, secondPhysical});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::LOGICAL_AND, injected, disjunction});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0});

  EXPECT_TRUE(result.referencesInjectedColumn());
  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(result.expr());
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_operator(), cudf::ast::ast_operator::LOGICAL_OR);
  EXPECT_THAT(operandColumnIndices(*transformed), ElementsAre(0, 1));
}

TEST(CudfIcebergFilterTransformTest, droppedAndInsideOrIsPushed) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& firstPhysical = tree.push(cudf::ast::column_reference{1});
  const auto& secondPhysical = tree.push(cudf::ast::column_reference{2});
  const auto& conjunction = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::LOGICAL_AND, injected, firstPhysical});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::LOGICAL_OR, conjunction, secondPhysical});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0});

  EXPECT_TRUE(result.referencesInjectedColumn());
  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(result.expr());
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_operator(), cudf::ast::ast_operator::LOGICAL_OR);
  EXPECT_THAT(operandColumnIndices(*transformed), ElementsAre(0, 1));
}

TEST(CudfIcebergFilterTransformTest, rebasesMultiplePhysicalColumns) {
  cudf::ast::tree tree;
  const auto& firstPhysical = tree.push(cudf::ast::column_reference{1});
  const auto& secondPhysical = tree.push(cudf::ast::column_reference{3});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::LOGICAL_AND, firstPhysical, secondPhysical});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 2>{0, 2});

  EXPECT_FALSE(result.referencesInjectedColumn());
  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(result.expr());
  ASSERT_NE(transformed, nullptr);
  EXPECT_THAT(operandColumnIndices(*transformed), ElementsAre(0, 1));
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
      expression, std::array<cudf::size_type, 1>{0});

  EXPECT_FALSE(result.referencesInjectedColumn());
  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(result.expr());
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
      expression, std::array<cudf::size_type, 1>{0});

  EXPECT_FALSE(result.referencesInjectedColumn());
  EXPECT_TRUE(result.requiresSplitSpecificDecimalTypes());
}

TEST(
    CudfIcebergFilterTransformTest,
    droppedDecimalDoesNotRequireSplitSpecificTypes) {
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
          cudf::ast::ast_operator::LOGICAL_AND,
          decimalComparison,
          integerComparison});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0});

  EXPECT_TRUE(result.referencesInjectedColumn());
  EXPECT_FALSE(result.requiresSplitSpecificDecimalTypes());
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
      expression, std::array<cudf::size_type, 1>{0});

  EXPECT_TRUE(result.referencesInjectedColumn());
  EXPECT_EQ(result.expr(), nullptr);
}

TEST(CudfIcebergFilterTransformTest, isNullOnInjectedColumnIsNotPushed) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{1});
  const auto& expression = tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::IS_NULL, injected});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{1});

  EXPECT_TRUE(result.referencesInjectedColumn());
  EXPECT_EQ(result.expr(), nullptr);
}

TEST(CudfIcebergFilterTransformTest, rebasePhysicalColumn) {
  cudf::ast::tree tree;
  const auto& expression = tree.push(cudf::ast::column_reference{2});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{0});

  EXPECT_FALSE(result.referencesInjectedColumn());
  const auto* transformed =
      dynamic_cast<const cudf::ast::column_reference*>(result.expr());
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_column_index(), 1);
}

TEST(CudfIcebergFilterTransformTest, trailingInjectedColumnDoesNotRebase) {
  cudf::ast::tree tree;
  const auto& expression = tree.push(cudf::ast::column_reference{1});

  const auto result = transformFilterForInjectedColumns(
      expression, std::array<cudf::size_type, 1>{2});

  EXPECT_FALSE(result.referencesInjectedColumn());
  // Nothing changed, so the input expression is pushed as is.
  EXPECT_EQ(result.expr(), &expression);
}

TEST(CudfIcebergFilterTransformTest, noInjectedColumnsPushesInputFilter) {
  cudf::ast::tree tree;
  const auto& expression = tree.push(cudf::ast::column_reference{2});

  const auto result = transformFilterForInjectedColumns(
      expression, std::span<const cudf::size_type>{});

  EXPECT_FALSE(result.referencesInjectedColumn());
  EXPECT_EQ(result.expr(), &expression);
}

TEST(CudfIcebergFilterTransformTest, unsortedInjectedColumnIndicesFail) {
  cudf::ast::tree tree;
  const auto& expression = tree.push(cudf::ast::column_reference{3});

  VELOX_ASSERT_THROW(
      (transformFilterForInjectedColumns(
          expression, std::array<cudf::size_type, 2>{2, 0})),
      "Injected column indices must be ascending and unique");
}

TEST(CudfIcebergFilterTransformTest, columnNameReferenceFails) {
  cudf::ast::tree tree;
  const auto& expression = tree.push(cudf::ast::column_name_reference{"c0"});

  VELOX_ASSERT_THROW(
      (transformFilterForInjectedColumns(
          expression, std::array<cudf::size_type, 1>{0})),
      "Iceberg subfield filter must use column index references");
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
