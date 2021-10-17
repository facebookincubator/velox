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

#include <optional>
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::functions::test;

namespace {

// Class to test the array_has_dupes operator.
class ArrayHasDupesTest : public FunctionBaseTest {
 protected:
  // Evaluate an expression.
  void testExpr(
      const VectorPtr& expected,
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    auto result =
        evaluate<SimpleVector<bool>>(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }

  // Execute test for integer types.
  template <typename T>
  void testInt() {
    auto array = makeNullableArrayVector<T>({
        {},
        {0},
        {1},
        {std::numeric_limits<T>::min()},
        {std::numeric_limits<T>::max()},
        {std::nullopt},
        {-1},
        {1, 2, 3},
        {1, 2, 1},
        {1, 1, 1},
        {-1, -2, -3},
        {-1, -2, -1},
        {-1, -1, -1},
        {std::nullopt, std::nullopt, std::nullopt},
        {1, 2, -2, 1},
        {1, 1, -2, -2, -2, 4, 8},
        {3, 8, std::nullopt},
        {1, 2, 3, std::nullopt, 4, 1, 2, std::nullopt},
    });

    auto expected = makeNullableFlatVector<bool>({
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        false,
        true,
        true,
        true,
        true,
        true,
        false,
        true,
    });

    testExpr(expected, "array_has_dupes(C0)", {array});
  }
};

} // namespace

// Test integer arrays.
TEST_F(ArrayHasDupesTest, integerArrays) {
  testInt<int8_t>();
  testInt<int16_t>();
  testInt<int32_t>();
  testInt<int64_t>();
}

// Test inline (short) strings.
TEST_F(ArrayHasDupesTest, inlineStringArrays) {
  using S = StringView;

  auto array = makeNullableArrayVector<StringView>({
      {},
      {S("")},
      {S(" ")},
      {S("a")},
      {std::nullopt},
      {S("a"), S("b")},
      {S("a"), S("A")},
      {S("a"), S("a")},
      {std::nullopt, std::nullopt},
      {S("a"), std::nullopt, S("b")},
      {S("a"), S("b"), S("a"), S("a")},
      {std::nullopt, S("b"), std::nullopt},
      {S("abc")},
  });

  auto expected = makeNullableFlatVector<bool>({
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      true,
      true,
      false,
      true,
      true,
      false,
  });

  testExpr(expected, "array_has_dupes(C0)", {array});
}

// Test non-inline (> 12 character length) strings.
TEST_F(ArrayHasDupesTest, stringArrays) {
  using S = StringView;

  auto array = makeNullableArrayVector<StringView>({
      {S("red shiny car ahead"), S("blue clear sky above")},
      {std::nullopt,
       S("blue clear sky above"),
       S("yellow rose flowers"),
       S("blue clear sky above"),
       S("orange beautiful sunset")},
      {std::nullopt, std::nullopt},
      {},
      {S("red shiny car ahead"),
       S("purple is an elegant color"),
       S("green plants make us happy")},
  });

  auto expected = makeNullableFlatVector<bool>({
      false,
      true,
      true,
      false,
      false,
  });

  testExpr(expected, "array_has_dupes(C0)", {array});
}

// Test for invalid signature and types.
TEST_F(ArrayHasDupesTest, invalidTypes) {
  auto array = makeNullableArrayVector<int32_t>({{1, 1}});
  auto expected = makeNullableFlatVector<bool>({true});

  EXPECT_THROW(
      testExpr(expected, "array_has_dupes(1)", {array}), std::invalid_argument);
  EXPECT_THROW(
      testExpr(expected, "array_has_dupes(C0, CO)", {array, array}),
      std::invalid_argument);
  EXPECT_THROW(
      testExpr(expected, "array_has_dupes(ARRAY[1], 1)", {array}),
      std::invalid_argument);
  EXPECT_THROW(
      testExpr(expected, "array_has_dupes(ARRAY[ARRAY[1]])", {array}),
      facebook::velox::VeloxUserError);
  EXPECT_THROW(
      testExpr(expected, "array_has_dupes()", {array}), std::invalid_argument);

  EXPECT_NO_THROW(testExpr(expected, "array_has_dupes(C0)", {array}));
}

TEST_F(ArrayHasDupesTest, invalidBooleanElementType) {
  auto array = makeNullableArrayVector<bool>({{true, true}});
  auto expected = makeNullableFlatVector<bool>({true});

  EXPECT_THROW(
      testExpr(expected, "array_has_dupes(C0)", {array}),
      facebook::velox::VeloxUserError);
}

TEST_F(ArrayHasDupesTest, invalidFloatElementType) {
  auto array = makeNullableArrayVector<float>({{1.1, 1.1}});
  auto expected = makeNullableFlatVector<bool>({true});

  EXPECT_THROW(
      testExpr(expected, "array_has_dupes(C0)", {array}),
      facebook::velox::VeloxUserError);
}

TEST_F(ArrayHasDupesTest, invalidDoubleElementType) {
  auto array = makeNullableArrayVector<double>({{1.1, 1.1}});
  auto expected = makeNullableFlatVector<bool>({true});

  EXPECT_THROW(
      testExpr(expected, "array_has_dupes(C0)", {array}),
      facebook::velox::VeloxUserError);
}
