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
#include "velox/core/Expressions.h"
#include "velox/functions/sparksql/specialforms/GetStructField.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class GetStructFieldTest : public SparkFunctionBaseTest {
 protected:
  core::TypedExprPtr makeGetStructFieldExpr(
      const TypePtr& inputType,
      const TypePtr& resultType,
      const std::string& inputName,
      int32_t ordinal) {
    return std::make_shared<const core::CallTypedExpr>(
        resultType,
        GetStructFieldCallToSpecialForm::kGetStructField,
        std::make_shared<const core::FieldAccessTypedExpr>(
            inputType, inputName),
        std::make_shared<core::ConstantTypedExpr>(INTEGER(), variant(ordinal)));
  }

  void testGetStructField(
      const VectorPtr& input,
      int ordinal,
      const VectorPtr& expected) {
    auto expr =
        makeGetStructFieldExpr(input->type(), expected->type(), "c0", ordinal);

    // Input is flat.
    auto result = evaluate(expr, makeRowVector({input}));
    assertEqualVectors(expected, result);

    // Input is dictionary or constant encoding.
    testEncodings(expr, {input}, expected);
  }
};

TEST_F(GetStructFieldTest, simpleType) {
  auto colInt = makeFlatVector<int32_t>({1, 2, 3, 4});
  auto colString = makeNullableFlatVector<std::string>(
      {"hello", "world", std::nullopt, "hi"});
  auto colIntWithNull =
      makeNullableFlatVector<int32_t>({11, std::nullopt, 13, 14});
  auto data = makeRowVector({colInt, colString, colIntWithNull});

  // Get int field.
  testGetStructField(data, 0, colInt);

  // Get string field.
  testGetStructField(data, 1, colString);

  // Get int field with null.
  testGetStructField(data, 2, colIntWithNull);
}

TEST_F(GetStructFieldTest, complexType) {
  auto colArray = makeNullableArrayVector<int32_t>(
      {{1, 2, std::nullopt},
       {3, std::nullopt, 4},
       {5, std::nullopt, std::nullopt},
       {6, 7, 8}});
  auto colMap = makeMapVector<std::string, int32_t>(
      {{{"a", 0}, {"b", 1}},
       {{"c", 3}, {"d", 4}},
       {{"e", 5}, {"f", 6}},
       {{"g", 7}, {"h", 8}}});
  auto colRow = makeRowVector(
      {makeNullableArrayVector<int32_t>(
           {{100, 101, std::nullopt},
            {std::nullopt},
            {200, std::nullopt, 202},
            {300, 301, 302}}),
       makeNullableFlatVector<std::string>({"a", "b", std::nullopt, "c"})});
  auto data = makeRowVector({colArray, colMap, colRow});

  // Get array field.
  testGetStructField(data, 0, colArray);

  // Get map field.
  testGetStructField(data, 1, colMap);

  // Get row field.
  testGetStructField(data, 2, colRow);
}

TEST_F(GetStructFieldTest, invalidOrdinal) {
  auto colInt = makeFlatVector<int32_t>({1, 2, 3, 4});
  auto colString = makeNullableFlatVector<std::string>(
      {"hello", "world", std::nullopt, "hi"});
  auto colIntWithNull =
      makeNullableFlatVector<int32_t>({11, std::nullopt, 13, 14});
  auto data = makeRowVector({colInt, colString, colIntWithNull});

  // Get int field.
  VELOX_ASSERT_RUNTIME_THROW(
      testGetStructField(data, -1, colInt),
      "Invalid ordinal. Should be greater than or equal to 0.");

  // Get string field.
  VELOX_ASSERT_THROW(
      testGetStructField(data, 4, colString),
      fmt::format(
          "(4 vs. 3) Invalid ordinal. Should be smaller than the children size of input row vector."));
}

TEST_F(GetStructFieldTest, switchWithGetStructFieldInBranches) {
  auto data = makeRowVector({
      makeFlatVector<bool>({true, false, true, false}),
      makeRowVector({makeFlatVector<int32_t>({1, 2, 3, 4})}),
      makeRowVector({makeFlatVector<int32_t>({10, 20, 30, 40})}),
  });

  auto switchExpr = std::make_shared<const core::CallTypedExpr>(
      INTEGER(),
      "switch",
      std::make_shared<const core::FieldAccessTypedExpr>(BOOLEAN(), "c0"),
      makeGetStructFieldExpr(data->childAt(1)->type(), INTEGER(), "c1", 0),
      makeGetStructFieldExpr(data->childAt(2)->type(), INTEGER(), "c2", 0));
  auto result = evaluate(switchExpr, data);
  assertEqualVectors(makeFlatVector<int32_t>({1, 20, 3, 40}), result);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
