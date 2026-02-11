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

#include <gtest/gtest.h>
#include <optional>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {

class NamedStructTest : public SparkFunctionBaseTest {
 protected:
  void testNamedStruct(
      const std::string& expression,
      const VectorPtr& expected) {
    auto result = evaluateExpression(expression, makeRowVector(ROW({}), 1));
    assertEqualVectors(expected, result);
  }

  void testNamedStructWithInput(
      const std::string& expression,
      const RowVectorPtr& input,
      const VectorPtr& expected) {
    auto result = evaluateExpression(expression, input);
    assertEqualVectors(expected, result);
  }
};

// Basic tests with 2-6 fields
TEST_F(NamedStructTest, basic2Fields) {
  auto expected = makeRowVector(
      {"id", "name"},
      {makeFlatVector<int32_t>({123}), makeFlatVector<std::string>({"John"})});

  testNamedStruct("named_struct('id', 123, 'name', 'John')", expected);
}

TEST_F(NamedStructTest, basic4Fields) {
  auto expected = makeRowVector(
      {"id", "name", "age", "active"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<std::string>({"Alice"}),
       makeFlatVector<int32_t>({30}),
       makeFlatVector<bool>({true})});

  testNamedStruct(
      "named_struct('id', 1, 'name', 'Alice', 'age', 30, 'active', true)",
      expected);
}

TEST_F(NamedStructTest, basic6Fields) {
  auto expected = makeRowVector(
      {"a", "b", "c", "d", "e", "f"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int32_t>({2}),
       makeFlatVector<int32_t>({3}),
       makeFlatVector<int32_t>({4}),
       makeFlatVector<int32_t>({5}),
       makeFlatVector<int32_t>({6})});

  testNamedStruct(
      "named_struct('a', 1, 'b', 2, 'c', 3, 'd', 4, 'e', 5, 'f', 6)", expected);
}

TEST_F(NamedStructTest, allPrimitiveTypes) {
  auto expected = makeRowVector(
      {"int_field", "double_field", "bool_field", "string_field"},
      {makeFlatVector<int32_t>({42}),
       makeFlatVector<double>({3.14}),
       makeFlatVector<bool>({false}),
       makeFlatVector<std::string>({"hello"})});

  testNamedStruct(
      "named_struct('int_field', 42, 'double_field', 3.14, 'bool_field', false, 'string_field', 'hello')",
      expected);
}

TEST_F(NamedStructTest, nullValues) {
  auto expected = makeRowVector(
      {"id", "name"},
      {makeNullableFlatVector<int32_t>({std::nullopt}),
       makeNullableFlatVector<std::string>({std::nullopt})});

  testNamedStruct("named_struct('id', CAST(NULL AS INTEGER), 'name', CAST(NULL AS VARCHAR))", expected);
}

TEST_F(NamedStructTest, emptyFieldName) {
  // Spark allows empty field names
  auto expected = makeRowVector(
      {"", "name"},
      {makeFlatVector<int32_t>({1}), makeFlatVector<std::string>({"Alice"})});

  testNamedStruct("named_struct('', 1, 'name', 'Alice')", expected);
}

TEST_F(NamedStructTest, singleField) {
  auto expected = makeRowVector(
      {"x"}, {makeFlatVector<double>({1.5})});

  testNamedStruct("named_struct('x', 1.5)", expected);
}

TEST_F(NamedStructTest, manyFields) {
  std::vector<std::string> fieldNames;
  std::vector<VectorPtr> fieldVectors;

  // Create 10 fields
  for (int i = 0; i < 10; i++) {
    fieldNames.push_back(fmt::format("field_{}", i));
    fieldVectors.push_back(makeFlatVector<int32_t>({i * 10}));
  }

  auto expected = makeRowVector(fieldNames, fieldVectors);

  std::string expr = "named_struct(";
  for (int i = 0; i < 10; i++) {
    if (i > 0)
      expr += ", ";
    expr += fmt::format("'field_{}', {}", i, i * 10);
  }
  expr += ")";

  testNamedStruct(expr, expected);
}

// Complex type tests
TEST_F(NamedStructTest, nestedStruct) {
  auto innerStruct = makeRowVector(
      {"x", "y"}, {makeFlatVector<int32_t>({1}), makeFlatVector<int32_t>({2})});

  auto expected = makeRowVector(
      {"id", "point"},
      {makeFlatVector<int32_t>({100}), innerStruct});

  testNamedStruct(
      "named_struct('id', 100, 'point', named_struct('x', 1, 'y', 2))",
      expected);
}

TEST_F(NamedStructTest, arrayField) {
  auto expected = makeRowVector(
      {"id", "values"},
      {makeFlatVector<int32_t>({1}),
       makeArrayVector<int32_t>({{1, 2, 3, 4, 5}})});

  testNamedStruct("named_struct('id', 1, 'values', array(1, 2, 3, 4, 5))", expected);
}

TEST_F(NamedStructTest, mapField) {
  auto expected = makeRowVector(
      {"id", "attributes"},
      {makeFlatVector<int32_t>({1}),
       makeMapVector<std::string, int32_t>({{{"a", 1}, {"b", 2}}})});

  testNamedStruct("named_struct('id', 1, 'attributes', map(array('a', 'b'), array(1, 2)))", expected);
}

TEST_F(NamedStructTest, mixedComplexTypes) {
  auto innerStruct = makeRowVector(
      {"nested_id"}, {makeFlatVector<int32_t>({99})});

  auto expected = makeRowVector(
      {"id", "tags", "metadata", "nested"},
      {makeFlatVector<int32_t>({1}),
       makeArrayVector<std::string>({{"tag1", "tag2"}}),
       makeMapVector<std::string, int32_t>({{{"key", 100}}}),
       innerStruct});

  testNamedStruct(
      "named_struct('id', 1, 'tags', array('tag1', 'tag2'), 'metadata', map(array('key'), array(100)), 'nested', named_struct('nested_id', 99))",
      expected);
}

// Error tests
TEST_F(NamedStructTest, oddArguments) {
  VELOX_ASSERT_THROW(
      evaluateExpression("named_struct('id', 1, 'name')", makeRowVector(ROW({}), 1)),
      "named_struct requires an even number of arguments");
}

TEST_F(NamedStructTest, zeroArguments) {
  VELOX_ASSERT_THROW(
      evaluateExpression("named_struct()", makeRowVector(ROW({}), 1)),
      "named_struct requires at least 2 arguments");
}

TEST_F(NamedStructTest, oneArgument) {
  VELOX_ASSERT_THROW(
      evaluateExpression("named_struct('id')", makeRowVector(ROW({}), 1)),
      "named_struct requires at least 2 arguments");
}

TEST_F(NamedStructTest, nullFieldName) {
  VELOX_ASSERT_THROW(
      evaluateExpression(
          "named_struct(CAST(NULL AS VARCHAR), 1)",
          makeRowVector(ROW({}), 1)),
      "Field name at position 0 cannot be null");
}

TEST_F(NamedStructTest, nonStringFieldName) {
  VELOX_ASSERT_THROW(
      evaluateExpression("named_struct(123, 'value')", makeRowVector(ROW({}), 1)),
      "Field name at position 0 must be VARCHAR");
}

// Test with column references
TEST_F(NamedStructTest, withColumnReferences) {
  auto input = makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<std::string>({"Alice", "Bob", "Charlie"}),
       makeFlatVector<int32_t>({30, 25, 35})},
      {"id", "name", "age"});

  auto expected = makeRowVector(
      {"user_id", "user_name", "user_age"},
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<std::string>({"Alice", "Bob", "Charlie"}),
       makeFlatVector<int32_t>({30, 25, 35})});

  testNamedStructWithInput(
      "named_struct('user_id', c0, 'user_name', c1, 'user_age', c2)",
      input,
      expected);
}

TEST_F(NamedStructTest, duplicateFieldNames) {
  // Spark allows duplicate field names (creates ambiguity but doesn't error)
  auto expected = makeRowVector(
      {"id", "id"},
      {makeFlatVector<int32_t>({1}), makeFlatVector<int32_t>({2})});

  testNamedStruct("named_struct('id', 1, 'id', 2)", expected);
}

TEST_F(NamedStructTest, allNullValues) {
  auto expected = makeRowVector(
      {"a", "b", "c"},
      {makeNullableFlatVector<int32_t>({std::nullopt}),
       makeNullableFlatVector<std::string>({std::nullopt}),
       makeNullableFlatVector<double>({std::nullopt})});

  testNamedStruct(
      "named_struct('a', CAST(NULL AS INTEGER), 'b', CAST(NULL AS VARCHAR), 'c', CAST(NULL AS DOUBLE))",
      expected);
}

TEST_F(NamedStructTest, emptyStringValue) {
  auto expected = makeRowVector(
      {"id", "name"},
      {makeFlatVector<int32_t>({1}), makeFlatVector<std::string>({""})});

  testNamedStruct("named_struct('id', 1, 'name', '')", expected);
}

TEST_F(NamedStructTest, unicodeFieldNames) {
  auto expected = makeRowVector(
      {"名前", "年齢", "emoji_😀"},
      {makeFlatVector<std::string>({"太郎"}),
       makeFlatVector<int32_t>({25}),
       makeFlatVector<std::string>({"🎉"})});

  testNamedStruct("named_struct('名前', '太郎', '年齢', 25, 'emoji_😀', '🎉')", expected);
}

TEST_F(NamedStructTest, numericTypes) {
  auto expected = makeRowVector(
      {"tinyint_val", "smallint_val", "int_val", "bigint_val", "float_val", "double_val"},
      {makeFlatVector<int8_t>({1}),
       makeFlatVector<int16_t>({2}),
       makeFlatVector<int32_t>({3}),
       makeFlatVector<int64_t>({4}),
       makeFlatVector<float>({5.5}),
       makeFlatVector<double>({6.6})});

  testNamedStruct(
      "named_struct('tinyint_val', CAST(1 AS TINYINT), 'smallint_val', CAST(2 AS SMALLINT), 'int_val', 3, 'bigint_val', CAST(4 AS BIGINT), 'float_val', CAST(5.5 AS REAL), 'double_val', 6.6)",
      expected);
}

TEST_F(NamedStructTest, withNullsInColumns) {
  auto input = makeRowVector(
      {makeNullableFlatVector<int32_t>({1, std::nullopt, 3}),
       makeNullableFlatVector<std::string>({"Alice", "Bob", std::nullopt})},
      {"id", "name"});

  auto expected = makeRowVector(
      {"user_id", "user_name"},
      {makeNullableFlatVector<int32_t>({1, std::nullopt, 3}),
       makeNullableFlatVector<std::string>({"Alice", "Bob", std::nullopt})});

  testNamedStructWithInput(
      "named_struct('user_id', c0, 'user_name', c1)", input, expected);
}

TEST_F(NamedStructTest, complexExpressionsAsValues) {
  auto input = makeRowVector(
      {makeFlatVector<int32_t>({10, 20, 30}),
       makeFlatVector<int32_t>({5, 10, 15})},
      {"a", "b"});

  auto expected = makeRowVector(
      {"sum_field", "product_field"},
      {makeFlatVector<int32_t>({15, 30, 45}),
       makeFlatVector<int32_t>({50, 200, 450})});

  testNamedStructWithInput(
      "named_struct('sum_field', c0 + c1, 'product_field', c0 * c1)",
      input,
      expected);
}

} // namespace facebook::velox::functions::sparksql::test
