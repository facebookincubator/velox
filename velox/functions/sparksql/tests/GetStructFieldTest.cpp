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

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
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
    auto batchSize = expected->size();
    auto ordinalVector = makeConstant<int32_t>(ordinal, batchSize);
    std::vector<core::TypedExprPtr> inputs = {
        std::make_shared<const core::FieldAccessTypedExpr>(input->type(), "c0"),
        std::make_shared<const core::FieldAccessTypedExpr>(INTEGER(), "c1")};
    auto resultType = expected->type();
    auto expr = std::make_shared<const core::CallTypedExpr>(
        resultType, std::move(inputs), "get_struct_field");
    auto result =
        evaluate(expr, makeRowVector({"c0", "c1"}, {input, ordinalVector}));
    ::facebook::velox::test::assertEqualVectors(expected, result);
  }
};

TEST_F(GetStructFieldTest, simpleType) {
  auto col0 = makeFlatVector<int32_t>({1, 2});
  auto col1 = makeFlatVector<std::string>({"hello", "world"});
  auto col2 = makeNullableFlatVector<int32_t>({std::nullopt, 12});
  auto data = makeRowVector({col0, col1, col2});

  // Get int field
  testGetStructField(data, 0, col0);

  // Get string field
  testGetStructField(data, 1, col1);

  // Get int field with null
  testGetStructField(data, 2, col2);
}

TEST_F(GetStructFieldTest, complexType) {
  auto col0 = makeArrayVector<int32_t>({{1, 2}, {3, 4}});
  auto col1 = makeMapVector<std::string, int32_t>(
      {{{"a", 0}, {"b", 1}}, {{"c", 3}, {"d", 4}}});
  auto col2 = makeRowVector(
      {makeArrayVector<int32_t>({{100, 101}, {200, 202}, {300, 303}}),
       makeFlatVector<std::string>({"a", "b"})});
  auto data = makeRowVector({col0, col1, col2});

  // Get array field
  testGetStructField(data, 0, col0);

  // Get map field
  testGetStructField(data, 1, col1);

  // Get row field
  testGetStructField(data, 2, col2);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
