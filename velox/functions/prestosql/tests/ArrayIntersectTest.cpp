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
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/vector/tests/TestingDictionaryArrayElementsFunction.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace {

class ArrayIntersectTest : public FunctionBaseTest {
 protected:
  void testExpr(
      const VectorPtr& expected,
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    auto result = evaluate<ArrayVector>(expression, makeRowVector(input));
    assertEqualVectors(expected, result);

    // Also test using dictionary encodings.
    if (input.size() == 2) {
      // Wrap first column in a dictionary: repeat each row twice. Wrap second
      // column in the same dictionary, then flatten to prevent peeling of
      // encodings. Wrap the expected result in the same dictionary.
      // The expression evaluation on both dictionary inputs should result in
      // the dictionary of the expected result vector.
      auto newSize = input[0]->size() * 2;
      auto indices = makeIndices(newSize, [](auto row) { return row / 2; });
      auto firstDict = wrapInDictionary(indices, newSize, input[0]);
      auto secondFlat = flatten(wrapInDictionary(indices, newSize, input[1]));

      auto dictResult = evaluate<ArrayVector>(
          expression, makeRowVector({firstDict, secondFlat}));
      auto dictExpected = wrapInDictionary(indices, newSize, expected);
      assertEqualVectors(dictExpected, dictResult);
    }
  }

  template <typename T>
  void testInt() {
    auto array1 = makeNullableArrayVector<T>({
        {{1, -2, 3, std::nullopt, 4, 5, 6, std::nullopt}},
        {{1, 2, -2, 1}},
        {{3, 8, std::nullopt}},
        std::nullopt,
        {{1, 1, -2, -2, -2, 4, 8}},
    });
    auto array2 = makeNullableArrayVector<T>({
        {1, -2, 4},
        {1, -2, 4},
        {1, -2, 4},
        {1, 2},
        {1, -2, 4},
    });
    auto expected = makeNullableArrayVector<T>({
        {{1, -2, 4}},
        {{1, -2}},
        {{}},
        std::nullopt,
        {{1, -2, 4}},
    });
    testExpr(expected, "array_intersect(C0, C1)", {array1, array2});
    testExpr(expected, "array_intersect(C1, C0)", {array1, array2});

    // Change C1.
    array2 = makeNullableArrayVector<T>({
        {10, -24, 43},
        {std::nullopt, -2, 2},
        {std::nullopt, std::nullopt, std::nullopt},
        {0, 0, 0},
        {8, 1, 8, 1},
    });
    expected = makeNullableArrayVector<T>({
        {{}},
        {{2, -2}},
        {std::vector<std::optional<T>>{std::nullopt}},
        std::nullopt,
        {{1, 8}},
    });
    testExpr(expected, "array_intersect(C0, C1)", {array1, array2});
  }

  template <typename T>
  void testFloatingPoint() {
    auto array1 = makeNullableArrayVector<T>({
        {1.0001, -2.0, 3.03, std::nullopt, 4.00004},
        {std::numeric_limits<T>::min(), 2.02, -2.001, 1},
        {std::numeric_limits<T>::max(), 8.0001, std::nullopt},
        {9.0009,
         std::numeric_limits<T>::infinity(),
         std::numeric_limits<T>::max()},
        {std::numeric_limits<T>::quiet_NaN(), 9.0009},
    });
    auto array2 = makeNullableArrayVector<T>({
        {1.0, -2.0, 4.0},
        {std::numeric_limits<T>::min(), 2.0199, -2.001, 1.000001},
        {1.0001, -2.02, std::numeric_limits<T>::max(), 8.00099},
        {9.0009, std::numeric_limits<T>::infinity()},
        {9.0009, std::numeric_limits<T>::quiet_NaN()},
    });
    auto expected = makeNullableArrayVector<T>({
        {-2.0},
        {std::numeric_limits<T>::min(), -2.001},
        {std::numeric_limits<T>::max()},
        {9.0009, std::numeric_limits<T>::infinity()},
        {9.0009},
    });

    testExpr(expected, "array_intersect(C0, C1)", {array1, array2});
    testExpr(expected, "array_intersect(C1, C0)", {array1, array2});
  }
};

} // namespace

TEST_F(ArrayIntersectTest, intArrays) {
  testInt<int8_t>();
  testInt<int16_t>();
  testInt<int32_t>();
  testInt<int64_t>();
}

TEST_F(ArrayIntersectTest, floatArrays) {
  testFloatingPoint<float>();
  testFloatingPoint<double>();
}

TEST_F(ArrayIntersectTest, boolArrays) {
  auto array1 = makeNullableArrayVector<bool>(
      {{true, false},
       {true, true},
       {false, false},
       {},
       {true, false, true, std::nullopt},
       {std::nullopt, true, false, true},
       {false, true, false},
       {true, false, true}});

  auto array2 = makeNullableArrayVector<bool>(
      {{true},
       {true, true},
       {false, false},
       {},
       {true, std::nullopt},
       {std::nullopt, false},
       {false, true, false},
       {true, false, true}});

  auto expected = makeNullableArrayVector<bool>(
      {{true},
       {true},
       {false},
       {},
       {true, std::nullopt},
       {std::nullopt, false},
       {false, true},
       {true, false}});

  testExpr(expected, "array_intersect(C0, C1)", {array1, array2});
  testExpr(expected, "array_intersect(C1, C0)", {array1, array2});
}

// Test inline strings.
TEST_F(ArrayIntersectTest, strArrays) {
  using S = StringView;

  auto array1 = makeNullableArrayVector<StringView>({
      {S("a"), std::nullopt, S("b")},
      {S("a"), S("b"), S("a"), S("a")},
      {std::nullopt, S("b"), std::nullopt},
      {S("abc")},
  });
  auto array2 = makeNullableArrayVector<StringView>({
      {S("a")},
      {S("a"), S("b"), S("b")},
      {std::nullopt, std::nullopt, std::nullopt},
      {S("abc"), S("a"), S("b")},
  });
  auto expected = makeNullableArrayVector<StringView>({
      {S("a")},
      {S("a"), S("b")},
      {std::nullopt},
      {S("abc")},
  });
  testExpr(expected, "array_intersect(C0, C1)", {array1, array2});
  testExpr(expected, "array_intersect(C1, C0)", {array1, array2});
}

// Test non-inline (> 12 length) strings.
TEST_F(ArrayIntersectTest, longStrArrays) {
  using S = StringView;

  auto array1 = makeNullableArrayVector<StringView>({
      {S("red shiny car ahead"), S("blue clear sky above")},
      {std::nullopt,
       S("blue clear sky above"),
       S("yellow rose flowers"),
       S("orange beautiful sunset")},
      {},
      {S("red shiny car ahead"),
       S("purple is an elegant color"),
       S("green plants make us happy")},
  });
  auto array2 = makeNullableArrayVector<StringView>({
      {S("red shiny car ahead")},
      {std::nullopt},
      {},
      {S("red shiny car ahead"), S("green plants make us happy")},
  });
  auto expected = makeNullableArrayVector<StringView>({
      {S("red shiny car ahead")},
      {std::nullopt},
      {},
      {S("red shiny car ahead"), S("green plants make us happy")},
  });
  testExpr(expected, "array_intersect(C0, C1)", {array1, array2});
  testExpr(expected, "array_intersect(C1, C0)", {array1, array2});
}

TEST_F(ArrayIntersectTest, varbinary) {
  auto left = makeNullableArrayVector<StringView>(
      {{"a"_sv, "b"_sv, "c"_sv}}, ARRAY(VARBINARY()));
  auto right = makeNullableArrayVector<StringView>(
      {{"b"_sv, "d"_sv}}, ARRAY(VARBINARY()));

  testExpr(left, "array_intersect(c0, c1)", {left, left});
  testExpr(right, "array_intersect(c0, c1)", {right, right});

  auto expected =
      makeNullableArrayVector<StringView>({{"b"_sv}}, ARRAY(VARBINARY()));
  testExpr(expected, "array_intersect(c0, c1)", {left, right});
  testExpr(expected, "array_intersect(c0, c1)", {right, left});
}

// When one of the arrays is constant.
TEST_F(ArrayIntersectTest, constant) {
  auto array1 = makeNullableArrayVector<int32_t>({
      {1, -2, 3, std::nullopt, 4, 5, 6, std::nullopt},
      {1, 2, -2, 1},
      {3, 8, std::nullopt},
      {1, 1, -2, -2, -2, 4, 8},
  });
  auto expected = makeNullableArrayVector<int32_t>({
      {1, -2, 4},
      {1, -2},
      {},
      {1, -2, 4},
  });
  testExpr(expected, "array_intersect(C0, ARRAY[1,4,-2])", {array1});
  testExpr(expected, "array_intersect(ARRAY[1,-2,4], C0)", {array1});
  testExpr(
      expected, "array_intersect(ARRAY[1,1,-2,1,-2,4,1,4,4], C0)", {array1});

  // Array containing NULLs.
  expected = makeNullableArrayVector<int32_t>({
      {1, std::nullopt, 4},
      {1},
      {std::nullopt},
      {1, 4},
  });
  testExpr(expected, "array_intersect(C0, ARRAY[1,NULL,4])", {array1});
  testExpr(expected, "array_intersect(ARRAY[1,NULL,4], C0)", {array1});
}

// Check that results are deterministic regardless of the encoding; literals
// (constant exprs) and regular columns should always return the same results.
TEST_F(ArrayIntersectTest, deterministic) {
  auto c0 = makeNullableArrayVector<int32_t>({
      {1, -2, 3, std::nullopt, 4, 5, 6, std::nullopt},
      {4, -2, 6, std::nullopt, 1},
  });
  auto c1 = makeNullableArrayVector<int32_t>({
      {1, 4, -2},
      {1, 4, -2},
  });

  // C0 then C1.
  auto expectedC0C1 = makeNullableArrayVector<int32_t>({
      {1, -2, 4},
      {4, -2, 1},
  });
  testExpr(expectedC0C1, "array_intersect(C0, ARRAY[1,4,-2])", {c0});
  testExpr(expectedC0C1, "array_intersect(C0, C1)", {c0, c1});

  // C1 then C0.
  auto expectedC1C0 = makeNullableArrayVector<int32_t>({
      {1, 4, -2},
      {1, 4, -2},
  });
  testExpr(expectedC1C0, "array_intersect(C1, C0)", {c0, c1});
  testExpr(expectedC1C0, "array_intersect(ARRAY[1,4,-2], C0)", {c0});
}

TEST_F(ArrayIntersectTest, dictionaryEncodedElementsInConstant) {
  exec::registerVectorFunction(
      "testing_dictionary_array_elements",
      test::TestingDictionaryArrayElementsFunction::signatures(),
      std::make_unique<test::TestingDictionaryArrayElementsFunction>());

  auto array = makeArrayVector<int32_t>({{1, 3}, {2, 5}, {0, 6}});
  auto expected = makeArrayVector<int32_t>({{1, 3}, {2}, {}});
  testExpr(
      expected,
      "array_intersect(c0, testing_dictionary_array_elements(ARRAY [2, 2, 3, 1, 2, 2]))",
      {array});
}
