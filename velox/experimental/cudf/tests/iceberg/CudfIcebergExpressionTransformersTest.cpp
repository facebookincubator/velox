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

#include <cudf/scalar/scalar.hpp>

#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {
namespace {

TEST(CudfIcebergExpressionTransformersTest, logicalAnd) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& physical = tree.push(cudf::ast::column_reference{2});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND, injected, physical});

  CudfIcebergExpressionTransformer transformer(expression, {0});

  EXPECT_TRUE(transformer.referencesInjectedColumn());
  const auto* transformed = dynamic_cast<const cudf::ast::column_reference*>(
      transformer.expression());
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_column_index(), 1);
}

TEST(CudfIcebergExpressionTransformersTest, logicalOr) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_OR, injected, physical});

  CudfIcebergExpressionTransformer transformer(expression, {0});

  EXPECT_TRUE(transformer.referencesInjectedColumn());
  EXPECT_EQ(transformer.expression(), nullptr);
}

TEST(CudfIcebergExpressionTransformersTest, negatedRelaxedExpression) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  const auto& conjunction = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::NULL_LOGICAL_AND, injected, physical});
  const auto& expression = tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::NOT, conjunction});

  CudfIcebergExpressionTransformer transformer(expression, {0});

  EXPECT_TRUE(transformer.referencesInjectedColumn());
  EXPECT_EQ(transformer.expression(), nullptr);
}

TEST(CudfIcebergExpressionTransformersTest, nestedLogicalAndRetainsOr) {
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

  CudfIcebergExpressionTransformer transformer(expression, {0});

  EXPECT_TRUE(transformer.referencesInjectedColumn());
  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(transformer.expression());
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_operator(), cudf::ast::ast_operator::LOGICAL_OR);
  const auto& operands = transformed->get_operands();
  const auto* firstColumn =
      dynamic_cast<const cudf::ast::column_reference*>(&operands[0].get());
  const auto* secondColumn =
      dynamic_cast<const cudf::ast::column_reference*>(&operands[1].get());
  ASSERT_NE(firstColumn, nullptr);
  ASSERT_NE(secondColumn, nullptr);
  EXPECT_EQ(firstColumn->get_column_index(), 0);
  EXPECT_EQ(secondColumn->get_column_index(), 1);
}

TEST(CudfIcebergExpressionTransformersTest, relaxedAndInsideOrIsPushed) {
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

  CudfIcebergExpressionTransformer transformer(expression, {0});

  EXPECT_TRUE(transformer.referencesInjectedColumn());
  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(transformer.expression());
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_operator(), cudf::ast::ast_operator::LOGICAL_OR);
  const auto& operands = transformed->get_operands();
  const auto* firstColumn =
      dynamic_cast<const cudf::ast::column_reference*>(&operands[0].get());
  const auto* secondColumn =
      dynamic_cast<const cudf::ast::column_reference*>(&operands[1].get());
  ASSERT_NE(firstColumn, nullptr);
  ASSERT_NE(secondColumn, nullptr);
  EXPECT_EQ(firstColumn->get_column_index(), 0);
  EXPECT_EQ(secondColumn->get_column_index(), 1);
}

TEST(CudfIcebergExpressionTransformersTest, rebasesMultiplePhysicalColumns) {
  cudf::ast::tree tree;
  const auto& firstPhysical = tree.push(cudf::ast::column_reference{1});
  const auto& secondPhysical = tree.push(cudf::ast::column_reference{3});
  const auto& expression = tree.push(
      cudf::ast::operation{
          cudf::ast::ast_operator::LOGICAL_AND, firstPhysical, secondPhysical});

  CudfIcebergExpressionTransformer transformer(expression, {0, 2});

  EXPECT_FALSE(transformer.referencesInjectedColumn());
  EXPECT_TRUE(transformer.changed());
  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(transformer.expression());
  ASSERT_NE(transformed, nullptr);
  const auto& operands = transformed->get_operands();
  const auto* firstColumn =
      dynamic_cast<const cudf::ast::column_reference*>(&operands[0].get());
  const auto* secondColumn =
      dynamic_cast<const cudf::ast::column_reference*>(&operands[1].get());
  ASSERT_NE(firstColumn, nullptr);
  ASSERT_NE(secondColumn, nullptr);
  EXPECT_EQ(firstColumn->get_column_index(), 0);
  EXPECT_EQ(secondColumn->get_column_index(), 1);
}

TEST(CudfIcebergExpressionTransformersTest, negatedPhysicalExpressionIsPushed) {
  cudf::ast::tree tree;
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  cudf::numeric_scalar<int32_t> literalValue{5};
  const auto& literal = tree.push(cudf::ast::literal{literalValue});
  const auto& comparison = tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::EQUAL, physical, literal});
  const auto& expression =
      tree.push(cudf::ast::operation{cudf::ast::ast_operator::NOT, comparison});

  CudfIcebergExpressionTransformer transformer(expression, {0});

  EXPECT_FALSE(transformer.referencesInjectedColumn());
  const auto* transformed =
      dynamic_cast<const cudf::ast::operation*>(transformer.expression());
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
    CudfIcebergExpressionTransformersTest,
    nonLogicalExpressionReferencingInjectedColumnIsNotPushed) {
  cudf::ast::tree tree;
  const auto& injected = tree.push(cudf::ast::column_reference{0});
  const auto& physical = tree.push(cudf::ast::column_reference{1});
  const auto& expression = tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::ADD, injected, physical});

  CudfIcebergExpressionTransformer transformer(expression, {0});

  EXPECT_TRUE(transformer.referencesInjectedColumn());
  EXPECT_EQ(transformer.expression(), nullptr);
}

TEST(CudfIcebergExpressionTransformersTest, rebasePhysicalColumn) {
  cudf::ast::tree tree;
  const auto& expression = tree.push(cudf::ast::column_reference{2});

  CudfIcebergExpressionTransformer transformer(expression, {0});

  EXPECT_FALSE(transformer.referencesInjectedColumn());
  EXPECT_TRUE(transformer.changed());
  const auto* transformed = dynamic_cast<const cudf::ast::column_reference*>(
      transformer.expression());
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->get_column_index(), 1);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
