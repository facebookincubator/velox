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
#include "velox/functions/sparksql/tests/CanonicalFloatingPoint.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/type/tests/utils/CustomTypesForTesting.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class ArrayUnionTest : public SparkFunctionBaseTest {
 protected:
  static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

  void testExpression(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
    expectCanonicalFloatingPoint(*result);
  }

  // Infinity and NaN, including the signaling NaN.
  template <typename T>
  void floatingPointExtremeValues() {
    static const T kQuietNaN = std::numeric_limits<T>::quiet_NaN();
    static const T kSignalingNaN = std::numeric_limits<T>::signaling_NaN();
    static const T kInfinity = std::numeric_limits<T>::infinity();
    const auto array1 = makeArrayVector<T>(
        {{1.1, 2.2, 3.3, 4.4},
         {3.3, 4.4},
         {3.3, 4.4, kQuietNaN},
         {3.3, 4.4, kQuietNaN},
         {3.3, 4.4, kQuietNaN},
         {3.3, 4.4, kQuietNaN, kInfinity}});
    const auto array2 = makeArrayVector<T>(
        {{3.3, 4.4},
         {3.3, 5.5},
         {5.5},
         {3.3, kQuietNaN},
         {5.5, kSignalingNaN},
         {5.5, kInfinity}});
    testExpression(
        "array_union(c0, c1)",
        {array1, array2},
        makeArrayVector<T>({
            {1.1, 2.2, 3.3, 4.4},
            {3.3, 4.4, 5.5},
            {3.3, 4.4, kQuietNaN, 5.5},
            {3.3, 4.4, kQuietNaN},
            {3.3, 4.4, kQuietNaN, 5.5},
            {3.3, 4.4, kQuietNaN, kInfinity, 5.5},
        }));
  }
};

TEST_F(ArrayUnionTest, intArray) {
  const auto array1 = makeArrayVector<int64_t>(
      {{1, 2, 3, 4}, {3, 4, 5}, {7, 8, 9}, {10, 20, 30}});
  const auto array2 =
      makeArrayVector<int64_t>({{2, 4, 5}, {3, 4, 5}, {}, {40, 50}});

  testExpression(
      "array_union(c0, c1)",
      {array1, array2},
      makeArrayVector<int64_t>({
          {1, 2, 3, 4, 5},
          {3, 4, 5},
          {7, 8, 9},
          {10, 20, 30, 40, 50},
      }));
  testExpression(
      "array_union(c0, c1)",
      {array2, array1},
      makeArrayVector<int64_t>({
          {2, 4, 5, 1, 3},
          {3, 4, 5},
          {7, 8, 9},
          {40, 50, 10, 20, 30},
      }));

  // Both arrays are empty.
  testExpression(
      "array_union(c0, c1)",
      {makeArrayVector<int64_t>({{}}), makeArrayVector<int64_t>({{}})},
      makeArrayVector<int64_t>({{}}));
}

TEST_F(ArrayUnionTest, stringArray) {
  testExpression(
      "array_union(c0, c1)",
      {makeArrayVector<StringView>({{"foo", "bar"}, {"foo", "baz"}}),
       makeArrayVector<StringView>({{"foo", "bar"}, {"bar", "baz"}})},
      makeArrayVector<StringView>({
          {"foo", "bar"},
          {"foo", "baz", "bar"},
      }));

  // Strings that are not inlined.
  testExpression(
      "array_union(c0, c1)",
      {makeArrayVector<StringView>(
           {{"foo", "a long string that is not inlined"}}),
       makeArrayVector<StringView>(
           {{"a long string that is not inlined", "baz"}})},
      makeArrayVector<StringView>({
          {"foo", "a long string that is not inlined", "baz"},
      }));
}

TEST_F(ArrayUnionTest, nulls) {
  const auto array1 = makeNullableArrayVector<int64_t>({
      {{1, std::nullopt, 3, 4}},
      {7, 8, 9},
      {{10, std::nullopt, std::nullopt}},
  });
  const auto array2 = makeNullableArrayVector<int64_t>({
      {{std::nullopt, std::nullopt, 3, 5}},
      std::nullopt,
      {{1, 10}},
  });

  testExpression(
      "array_union(c0, c1)",
      {array1, array2},
      makeNullableArrayVector<int64_t>({
          {{1, std::nullopt, 3, 4, 5}},
          std::nullopt,
          {{10, std::nullopt, 1}},
      }));
  testExpression(
      "array_union(c0, c1)",
      {array2, array1},
      makeNullableArrayVector<int64_t>({
          {{std::nullopt, 3, 5, 1, 4}},
          std::nullopt,
          {{1, 10, std::nullopt}},
      }));
}

TEST_F(ArrayUnionTest, encodings) {
  // Dictionary-encoded inputs that reference the same array more than once.
  auto base = makeArrayVector<int64_t>({{1, 2}, {2, 3}});
  auto left = wrapInDictionary(makeIndices({0, 1, 0}), base);
  auto right = wrapInDictionary(makeIndices({1, 1, 0}), base);
  testExpression(
      "array_union(c0, c1)",
      {left, right},
      makeArrayVector<int64_t>({{1, 2, 3}, {2, 3}, {1, 2}}));

  // Constant input.
  testExpression(
      "array_union(c0, c1)",
      {BaseVector::wrapInConstant(3, 1, base), left},
      makeArrayVector<int64_t>({{2, 3, 1}, {2, 3}, {2, 3, 1}}));
}

TEST_F(ArrayUnionTest, unknownType) {
  // [null], [null, null]
  auto array = makeArrayVector(
      {0, 1}, BaseVector::createNullConstant(UNKNOWN(), 3, pool()));
  testExpression(
      "array_union(c0, c1)",
      {array, array},
      makeArrayVector(
          {0, 1}, BaseVector::createNullConstant(UNKNOWN(), 2, pool())));
}

TEST_F(ArrayUnionTest, complexTypes) {
  auto baseVector = makeArrayVector<int64_t>(
      {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}});
  // [[1, 1], [2, 2]], [[3, 3], [4, 4]], [[5, 5], [6, 6]]
  auto arrayOfArrays1 = makeArrayVector({0, 2, 4}, baseVector);
  // [[1, 1], [2, 2], [3, 3]], [[4, 4]], [[5, 5], [6, 6]]
  auto arrayOfArrays2 = makeArrayVector({0, 3, 4}, baseVector);

  testExpression(
      "array_union(c0, c1)",
      {arrayOfArrays1, arrayOfArrays2},
      makeArrayVector(
          {0, 3, 5},
          makeArrayVector<int64_t>(
              {{1, 1}, {2, 2}, {3, 3}, {3, 3}, {4, 4}, {5, 5}, {6, 6}})));
}

TEST_F(ArrayUnionTest, floatingPoint) {
  // -0.0 and 0.0 are equal, and so are all NaNs. The result has 0.0 and the
  // canonical NaN.
  testExpression(
      "array_union(c0, c1)",
      {makeArrayVector<double>(
           {{-0.0, 1.0}, {kNonCanonicalNaN}, {1.5, kNaN}, {-0.0}}),
       makeArrayVector<double>({{0.0, kNaN}, {-0.0}, {kNaN, 2.5}, {}})},
      makeArrayVector<double>(
          {{0.0, 1.0, kNaN}, {kNaN, 0.0}, {1.5, kNaN, 2.5}, {0.0}}));

  testExpression(
      "array_union(c0, c1)",
      {makeArrayVector<float>({{-0.0f, std::nanf("")}}),
       makeArrayVector<float>({{0.0f}})},
      makeArrayVector<float>({{0.0f, std::nanf("")}}));
}

TEST_F(ArrayUnionTest, floatingPointExtremeValues) {
  floatingPointExtremeValues<float>();
  floatingPointExtremeValues<double>();
}

TEST_F(ArrayUnionTest, customComparison) {
  // BIGINT_TYPE_WITH_CUSTOM_COMPARISON compares only the bottom 8 bits, so 1,
  // 257 and 513 are equal.
  const auto type = BIGINT_TYPE_WITH_CUSTOM_COMPARISON();
  testExpression(
      "array_union(c0, c1)",
      {makeArrayVector({0}, makeFlatVector<int64_t>({1, 257, 3}, type)),
       makeArrayVector({0}, makeFlatVector<int64_t>({513, 2}, type))},
      makeArrayVector({0}, makeFlatVector<int64_t>({1, 3, 2}, type)));
}

TEST_F(ArrayUnionTest, nestedFloatingPoint) {
  // [[-0.0]], [[1.0, -0.0]]
  auto left =
      makeArrayVector({0, 1}, makeArrayVector<double>({{-0.0}, {1.0, -0.0}}));
  // [[0.0]], [[2.0]]
  auto right = makeArrayVector({0, 1}, makeArrayVector<double>({{0.0}, {2.0}}));
  testExpression(
      "array_union(c0, c1)",
      {left, right},
      makeArrayVector(
          {0, 1}, makeArrayVector<double>({{0.0}, {1.0, 0.0}, {2.0}})));

  // [{-0.0, 1}], [{0.0, 1}, {-0.0, 2}]
  auto structs = makeRowVector({
      makeFlatVector<double>({-0.0, 0.0, -0.0}),
      makeFlatVector<int32_t>({1, 1, 2}),
  });
  testExpression(
      "array_union(c0, c1)",
      {makeArrayVector({0}, structs->slice(0, 1)),
       makeArrayVector({0}, structs->slice(1, 2))},
      makeArrayVector(
          {0},
          makeRowVector({
              makeFlatVector<double>({0.0, 0.0}),
              makeFlatVector<int32_t>({1, 2}),
          })));
}

TEST_F(ArrayUnionTest, nestedNull) {
  const auto array1 = makeNestedArrayVectorFromJson<int32_t>({
      "[[1], [null]]",
  });
  const auto array2 = makeNestedArrayVectorFromJson<int32_t>({
      "[[2], [null]]",
  });

  auto expected = makeNestedArrayVectorFromJson<int32_t>({
      "[[1], [null], [2]]",
  });

  testExpression("array_union(c0, c1)", {array1, array2}, expected);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
