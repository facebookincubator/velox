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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class ArrayInsertTest : public SparkFunctionBaseTest {
 protected:
  void SetUp() override {
    SparkFunctionBaseTest::SetUp();
    options_.parseIntegerAsBigint = false;
  }

  void testExpression(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    const auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }

  void testExpression(
      const std::string& expression,
      const RowVectorPtr& input,
      const VectorPtr& expected) {
    const auto result = evaluate(expression, input);
    assertEqualVectors(expected, result);
  }

  void evaluateExpression(
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    evaluate(expression, makeRowVector(input));
  }

  template <typename T>
  void testInt() {
    const auto input = makeRowVector({
        makeArrayVectorFromJson<T>({"[1]", "[2, 2]"}),
        makeNullableFlatVector<T>({0, std::nullopt}),
    });

    auto expected = makeArrayVectorFromJson<T>({"[0, 1]", "[null, 2, 2]"});
    testExpression("array_insert(c0, 1, c1, false)", input, expected);

    expected = makeArrayVectorFromJson<T>({"[0, 1]", "[2, null, 2]"});
    testExpression("array_insert(c0, -2, c1, false)", input, expected);
  }

  template <typename T>
  void testFloat() {
    const auto input = makeRowVector({
        makeNullableArrayVector<T>(
            {{1.0001, -2.0},
             {2.0001, std::numeric_limits<T>::infinity()},
             {std::numeric_limits<T>::quiet_NaN(), 9.0009},
             {3.0001}}),
        makeNullableFlatVector<T>(
            {0,
             std::nullopt,
             std::numeric_limits<T>::infinity(),
             std::numeric_limits<T>::quiet_NaN()}),
    });

    auto expected = makeNullableArrayVector<T>(
        {{0, 1.0001, -2.0},
         {std::nullopt, 2.0001, std::numeric_limits<T>::infinity()},
         {std::numeric_limits<T>::infinity(),
          std::numeric_limits<T>::quiet_NaN(),
          9.0009},
         {std::numeric_limits<T>::quiet_NaN(), 3.0001}});
    testExpression("array_insert(c0, 1, c1, false)", input, expected);

    expected = makeNullableArrayVector<T>(
        {{1.0001, 0, -2.0},
         {2.0001, std::nullopt, std::numeric_limits<T>::infinity()},
         {std::numeric_limits<T>::quiet_NaN(),
          std::numeric_limits<T>::infinity(),
          9.0009},
         {std::numeric_limits<T>::quiet_NaN(), 3.0001}});
    testExpression("array_insert(c0, -2, c1, false)", input, expected);
  }
};

TEST_F(ArrayInsertTest, intArrays) {
  testInt<int8_t>();
  testInt<int16_t>();
  testInt<int32_t>();
  testInt<int64_t>();
}

TEST_F(ArrayInsertTest, floatArrays) {
  testInt<float_t>();
  testInt<double_t>();
}

TEST_F(ArrayInsertTest, boolArrays) {
  const auto arrays = makeArrayVectorFromJson<bool>(
      {"[true]", "[true, false]", "[true, false, true]"});

  auto expected = makeArrayVectorFromJson<bool>(
      {"[true, true]", "[true, true, false]", "[true, true, false, true]"});
  testExpression("array_insert(c0, 1, true, false)", {arrays}, expected);

  expected = makeArrayVectorFromJson<bool>(
      {"[null, true]", "[null, true, false]", "[null, true, false, true]"});
  testExpression(
      "array_insert(c0, 1, null::BOOLEAN, false)", {arrays}, expected);

  expected = expected = makeArrayVectorFromJson<bool>(
      {"[true, true]", "[true, true, false]", "[true, false, true, true]"});
  testExpression("array_insert(c0, -2, true, false)", {arrays}, expected);
}

TEST_F(ArrayInsertTest, strArrays) {
  using S = StringView;

  auto arrays = makeNullableArrayVector<StringView>({
      {S("a"), std::nullopt, S("b")},
      {S("a"), S("b"), S("a"), S("a")},
      {std::nullopt, S("b"), std::nullopt},
      {S("purple is an elegant color")},
  });

  auto expected = makeNullableArrayVector<StringView>({
      {S("X"), S("a"), std::nullopt, S("b")},
      {S("X"), S("a"), S("b"), S("a"), S("a")},
      {S("X"), std::nullopt, S("b"), std::nullopt},
      {S("X"), S("purple is an elegant color")},
  });
  testExpression("array_insert(c0, 1, 'X', false)", {arrays}, expected);

  expected = makeNullableArrayVector<StringView>({
      {std::nullopt, S("a"), std::nullopt, S("b")},
      {std::nullopt, S("a"), S("b"), S("a"), S("a")},
      {std::nullopt, std::nullopt, S("b"), std::nullopt},
      {std::nullopt, S("purple is an elegant color")},
  });
  testExpression(
      "array_insert(c0, 1, null::STRING, false)", {arrays}, expected);

  expected = makeNullableArrayVector<StringView>({
      {S("a"), std::nullopt, S("X"), S("b")},
      {S("a"), S("b"), S("a"), S("X"), S("a")},
      {std::nullopt, S("b"), S("X"), std::nullopt},
      {S("X"), S("purple is an elegant color")},
  });
  testExpression("array_insert(c0, -2, 'X', false)", {arrays}, expected);
}

TEST_F(ArrayInsertTest, varbinary) {
  auto arrays = makeNullableArrayVector<StringView>(
      {{"a"_sv, "b"_sv, "c"_sv}}, ARRAY(VARBINARY()));

  auto expected = makeNullableArrayVector<StringView>(
      {{"X"_sv, "a"_sv, "b"_sv, "c"_sv}}, ARRAY(VARBINARY()));
  testExpression(
      "array_insert(c0, 1, 'X'::VARBINARY, false)", {arrays}, expected);

  expected = makeNullableArrayVector<StringView>(
      {{std::nullopt, "a"_sv, "b"_sv, "c"_sv}}, ARRAY(VARBINARY()));
  testExpression(
      "array_insert(c0, 1, null::VARBINARY, false)", {arrays}, expected);

  expected = makeNullableArrayVector<StringView>(
      {{"a"_sv, "b"_sv, "X"_sv, "c"_sv}}, ARRAY(VARBINARY()));
  testExpression(
      "array_insert(c0, -2, 'X'::VARBINARY, false)", {arrays}, expected);
}

TEST_F(ArrayInsertTest, nullSrcArrays) {
  const auto arrays = makeArrayVectorFromJson<int64_t>({"null"});
  const auto expected = makeArrayVectorFromJson<int64_t>({"null"});
  testExpression("array_insert(c0, 1, 1, false)", {arrays}, expected);
}

TEST_F(ArrayInsertTest, nullPosition) {
  const auto arrays = makeArrayVectorFromJson<int64_t>({"[1, 1]"});
  const auto expected = makeArrayVectorFromJson<int64_t>({"null"});
  testExpression(
      "array_insert(c0, null::INTEGER, 1, false)", {arrays}, expected);
}

TEST_F(ArrayInsertTest, posGTArraySize) {
  const auto arrays = makeArrayVectorFromJson<int64_t>({"[1]", "[2, 2]"});

  auto expected =
      makeArrayVectorFromJson<int64_t>({"[1, null, 0]", "[2, 2, 0]"});
  testExpression("array_insert(c0, 3, 0, false)", {arrays}, expected);

  expected =
      makeArrayVectorFromJson<int64_t>({"[1, null, null]", "[2, 2, null]"});
  testExpression(
      "array_insert(c0, 3, null::INTEGER, false)", {arrays}, expected);

  // Insert into position INT_MAX.
  EXPECT_THROW(
      evaluateExpression("array_insert(c0, 2147483647, 0, false)", {arrays}),
      VeloxUserError);
  // Insert into position INT_MIN.
  EXPECT_THROW(
      evaluateExpression(
          "array_insert(c0, cast(-2147483648 as integer), 0, false)", {arrays}),
      VeloxUserError);
}

TEST_F(ArrayInsertTest, negativePos) {
  const auto arrays =
      makeArrayVectorFromJson<int64_t>({"[1]", "[2, 2]", "[3, 3, 3]"});

  auto expected = makeArrayVectorFromJson<int64_t>(
      {"[0, null, 1]", "[0, 2, 2]", "[3, 0, 3, 3]"});
  testExpression("array_insert(c0, -3, 0, false)", {arrays}, expected);

  expected =
      makeArrayVectorFromJson<int64_t>({"[1, 0]", "[2, 2, 0]", "[3, 3, 3, 0]"});
  testExpression("array_insert(c0, -1, 0, false)", {arrays}, expected);

  expected = makeArrayVectorFromJson<int64_t>(
      {"[null, null, 1]", "[null, 2, 2]", "[3, null, 3, 3]"});
  testExpression(
      "array_insert(c0, -3, null::INTEGER, false)", {arrays}, expected);
}

TEST_F(ArrayInsertTest, negativePosLegacy) {
  const auto arrays =
      makeArrayVectorFromJson<int64_t>({"[1]", "[2, 2]", "[3, 3, 3]"});

  auto expected = makeArrayVectorFromJson<int64_t>(
      {"[0, null, null, 1]", "[0, null, 2, 2]", "[0, 3, 3, 3]"});
  testExpression("array_insert(c0, -3, 0, true)", {arrays}, expected);

  expected =
      makeArrayVectorFromJson<int64_t>({"[0, 1]", "[2, 0, 2]", "[3, 3, 0, 3]"});
  testExpression("array_insert(c0, -1, 0, true)", {arrays}, expected);

  expected = makeArrayVectorFromJson<int64_t>(
      {"[null, null, null, 1]", "[null, null, 2, 2]", "[null, 3, 3, 3]"});
  testExpression(
      "array_insert(c0, -3, null::INTEGER, true)", {arrays}, expected);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
