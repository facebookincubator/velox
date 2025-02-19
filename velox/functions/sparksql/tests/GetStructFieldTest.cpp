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

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/vector/tests/TestingDictionaryFunction.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class GetStructFieldTest : public SparkFunctionBaseTest {
 protected:
  void testGetStructField(
      const VectorPtr& input,
      int ordinal,
      const VectorPtr& expected) {
    std::vector<core::TypedExprPtr> inputs = {
        std::make_shared<const core::FieldAccessTypedExpr>(input->type(), "c0"),
        std::make_shared<core::ConstantTypedExpr>(INTEGER(), variant(ordinal))};
    auto resultType = expected->type();
    auto expr = std::make_shared<const core::CallTypedExpr>(
        resultType, std::move(inputs), "get_struct_field");
    auto result = evaluate(expr, makeRowVector({input}));
    ::facebook::velox::test::assertEqualVectors(expected, result);
  }
};

TEST_F(GetStructFieldTest, simpleType) {
  auto col0 = makeFlatVector<int32_t>({1, 2});
  auto col1 = makeFlatVector<std::string>({"hello", "world"});
  auto col2 = makeNullableFlatVector<int32_t>({std::nullopt, 12});
  auto data = makeRowVector({col0, col1, col2});

  // Get int field.
  testGetStructField(data, 0, col0);

  // Get string field.
  testGetStructField(data, 1, col1);

  // Get int field with null.
  testGetStructField(data, 2, col2);
}

TEST_F(GetStructFieldTest, complexType) {
  auto col0 = makeArrayVector<int32_t>({{1, 2}, {3, 4}});
  auto col1 = makeMapVector<std::string, int32_t>(
      {{{"a", 0}, {"b", 1}}, {{"c", 3}, {"d", 4}}});
  auto col2 = makeRowVector(
      {makeArrayVector<int32_t>(
           {{100, 101, 102}, {200, 201, 202}, {300, 301, 302}}),
       makeFlatVector<std::string>({"a", "b", "c"})});
  auto data = makeRowVector({col0, col1, col2});

  // Get array field.
  testGetStructField(data, 0, col0);

  // Get map field.
  testGetStructField(data, 1, col1);

  // Get row field.
  testGetStructField(data, 2, col2);
}

TEST_F(GetStructFieldTest, dictionaryEncodedElementsInFlat) {
  // Verify that get_struct_field properly handles an input where the top level
  // vector has a dictionary layer wrapped.
  exec::registerVectorFunction(
      "add_dict",
      TestingDictionaryFunction::signatures(),
      std::make_unique<TestingDictionaryFunction>(1));

  auto col0 = makeNullableFlatVector<int32_t>({1, std::nullopt, 2, 3});
  auto col1 = makeNullableFlatVector<std::string>(
      {"hello", "world", std::nullopt, "hi"});
  auto col2 = makeNullableArrayVector<int32_t>(
      {{1, 2, std::nullopt},
       {3, std::nullopt, 4},
       {5, std::nullopt, std::nullopt},
       {6, 7, 8}});
  auto data = makeRowVector({col0, col1, col2});
  auto wrappedData = evaluate("add_dict(c0)", makeRowVector({data}));

  auto expectedCol0 =
      makeNullableFlatVector<int32_t>({3, 2, std::nullopt, std::nullopt});
  testGetStructField(wrappedData, 0, expectedCol0);

  auto expectedCol1 = makeNullableFlatVector<std::string>(
      {"hi", std::nullopt, "world", std::nullopt});
  testGetStructField(wrappedData, 1, expectedCol1);

  auto expectedCol2 = makeNullableArrayVector<int32_t>({
      {{6, 7, 8}},
      {{5, std::nullopt, std::nullopt}},
      {{3, std::nullopt, 4}},
      std::nullopt,
  });
  testGetStructField(wrappedData, 2, expectedCol2);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
