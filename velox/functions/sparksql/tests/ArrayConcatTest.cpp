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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class ArrayConcatTest : public SparkFunctionBaseTest {
 protected:
  void testExpression(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }
};

// Concatenate two integer arrays.
TEST_F(ArrayConcatTest, intArray) {
  const auto intArray = makeArrayVector<int64_t>(
      {{1, 2, 3, 4}, {3, 4, 5}, {7, 8, 9}, {10, 20, 30}});
  const auto emptyArray = makeArrayVector<int64_t>({{9, 8}, {}, {777}, {}});

  VectorPtr expected = makeArrayVector<int64_t>({
      {1, 2, 3, 4, 9, 8},
      {3, 4, 5},
      {7, 8, 9, 777},
      {10, 20, 30},
  });
  testExpression("concat(c0, c1)", {intArray, emptyArray}, expected);

  expected = makeArrayVector<int64_t>({
      {9, 8, 1, 2, 3, 4},
      {3, 4, 5},
      {777, 7, 8, 9},
      {10, 20, 30},
  });
  testExpression("concat(c0, c1)", {emptyArray, intArray}, expected);
}

// Concatenate two integer arrays with null.
TEST_F(ArrayConcatTest, nullArray) {
  const auto intArray =
      makeArrayVector<int64_t>({{1, 2, 3, 4}, {7, 8, 9}, {10, 20, 30}});
  const auto nullArray = makeNullableArrayVector<int64_t>({
      {{std::nullopt, std::nullopt}},
      std::nullopt,
      {{1}},
  });

  VectorPtr expected = makeNullableArrayVector<int64_t>({
      {{1, 2, 3, 4, std::nullopt, std::nullopt}},
      std::nullopt,
      {{10, 20, 30, 1}},
  });
  testExpression("concat(c0, c1)", {intArray, nullArray}, expected);

  expected = makeNullableArrayVector<int64_t>({
      {{std::nullopt, std::nullopt, 1, 2, 3, 4}},
      std::nullopt,
      {{1, 10, 20, 30}},
  });
  testExpression("concat(c0, c1)", {nullArray, intArray}, expected);
}

TEST_F(ArrayConcatTest, arity) {
  const auto array1 =
      makeArrayVector<int64_t>({{1, 2, 3, 4}, {3, 4, 5}, {7, 8, 9}});
  const auto array2 = makeArrayVector<int64_t>({{9, 8}, {}, {777}});
  const auto array3 = makeArrayVector<int64_t>({{123, 42}, {55}, {10, 20, 30}});
  const auto array4 = makeNullableArrayVector<int64_t>({
      {{std::nullopt, std::nullopt}},
      std::nullopt,
      {{std::nullopt, std::nullopt, std::nullopt}},
  });

  testExpression("concat(c0)", {array1}, array1);

  VectorPtr expected = makeArrayVector<int64_t>(
      {{1, 2, 3, 4, 9, 8, 123, 42}, {3, 4, 5, 55}, {7, 8, 9, 777, 10, 20, 30}});
  testExpression("concat(c0, c1, c2)", {array1, array2, array3}, expected);

  expected = makeArrayVector<int64_t>(
      {{123, 42, 9, 8, 1, 2, 3, 4, 9, 8},
       {55, 3, 4, 5},
       {10, 20, 30, 777, 7, 8, 9, 777}});
  testExpression(
      "concat(c0, c1, c2, c3)", {array3, array2, array1, array2}, expected);

  expected = makeNullableArrayVector<int64_t>(
      {{{1, 2, 3, 4, std::nullopt, std::nullopt, 123, 42, 9, 8}},
       std::nullopt,
       {{7, 8, 9, std::nullopt, std::nullopt, std::nullopt, 10, 20, 30, 777}}});
  testExpression(
      "concat(c0, c1, c2, c3)", {array1, array4, array3, array2}, expected);
}

// Concatenate complex types.
TEST_F(ArrayConcatTest, complexTypes) {
  auto baseVector = makeArrayVector<int64_t>(
      {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}});

  // Create arrays of array vector using above base vector.
  // [[1, 1], [2, 2]]
  // [[3, 3], [4, 4]]
  // [[5, 5], [6, 6]]
  auto arrayOfArrays1 = makeArrayVector({0, 2, 4}, baseVector);
  // [[1, 1], [2, 2], [3, 3]]
  // [[4, 4]]
  // [[5, 5], [6, 6]]
  auto arrayOfArrays2 = makeArrayVector({0, 3, 4}, baseVector);

  // [[1, 1], [2, 2], [1, 1], [2, 2], [3, 3]]
  // [[3, 3], [4, 4], [4, 4]]
  // [[5, 5], [6, 6], [5, 5], [6, 6]]
  auto expected = makeArrayVector(
      {0, 5, 8},
      makeArrayVector<int64_t>(
          {{1, 1},
           {2, 2},
           {1, 1},
           {2, 2},
           {3, 3},
           {3, 3},
           {4, 4},
           {4, 4},
           {5, 5},
           {6, 6},
           {5, 5},
           {6, 6}}));

  testExpression("concat(c0, c1)", {arrayOfArrays1, arrayOfArrays2}, expected);
}

TEST_F(ArrayConcatTest, unknown) {
  const auto array1 = makeArrayVectorFromJson<UnknownValue>({"[null, null]"});
  const auto array2 = makeArrayVectorFromJson<UnknownValue>({"[null]"});

  const auto expected =
      makeArrayVectorFromJson<UnknownValue>({"[null, null, null]"});
  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test concatenating arrays containing only null elements
TEST_F(ArrayConcatTest, allNullElements) {
  auto array1 = makeNullableArrayVector<int64_t>({
      {{std::nullopt, std::nullopt}},
      {{std::nullopt}},
  });
  auto array2 = makeNullableArrayVector<int64_t>({
      {{std::nullopt}},
      {{std::nullopt, std::nullopt, std::nullopt}},
  });

  VectorPtr expected = makeNullableArrayVector<int64_t>({
      {{std::nullopt, std::nullopt, std::nullopt}},
      {{std::nullopt, std::nullopt, std::nullopt, std::nullopt}},
  });

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test concatenating only empty arrays
TEST_F(ArrayConcatTest, allEmptyArrays) {
  auto array1 = makeArrayVector<int64_t>({{}, {}});
  auto array2 = makeArrayVector<int64_t>({{}, {}});
  auto array3 = makeArrayVector<int64_t>({{}, {}});

  VectorPtr expected = makeArrayVector<int64_t>({{}, {}});

  testExpression("concat(c0, c1, c2)", {array1, array2, array3}, expected);
}

// Test float arrays with nulls
TEST_F(ArrayConcatTest, floatArraysWithNulls) {
  auto array1 = makeNullableArrayVector<float>({
      {{1.5f, std::nullopt, 2.5f}},
      {{std::nullopt}},
  });
  auto array2 = makeNullableArrayVector<float>({
      {{std::nullopt, 3.5f}},
      {{4.5f}},
  });

  VectorPtr expected = makeNullableArrayVector<float>({
      {{1.5f, std::nullopt, 2.5f, std::nullopt, 3.5f}},
      {{std::nullopt, 4.5f}},
  });

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test string/VARCHAR arrays with nulls
TEST_F(ArrayConcatTest, stringArraysWithNulls) {
  auto array1 = makeNullableArrayVector<StringView>({
      {{"a"_sv, std::nullopt, "b"_sv}},
      {{std::nullopt, "c"_sv}},
  });
  auto array2 = makeNullableArrayVector<StringView>({
      {{std::nullopt, "d"_sv}},
      {{"e"_sv}},
  });

  VectorPtr expected = makeNullableArrayVector<StringView>({
      {{"a"_sv, std::nullopt, "b"_sv, std::nullopt, "d"_sv}},
      {{std::nullopt, "c"_sv, "e"_sv}},
  });

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test variadic with many arrays (10 arrays)
TEST_F(ArrayConcatTest, manyArrays) {
  std::vector<VectorPtr> inputs;
  for (int i = 0; i < 10; i++) {
    inputs.push_back(makeArrayVector<int64_t>({{i}}));
  }

  VectorPtr expected =
      makeArrayVector<int64_t>({{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}});

  testExpression(
      "concat(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)", inputs, expected);
}

// Test size limit enforcement
TEST_F(ArrayConcatTest, sizeLimit) {
  // Create two large arrays that would exceed the limit when concatenated
  constexpr int32_t kMaxElements = INT32_MAX - 15;
  constexpr int32_t kHalfSize = kMaxElements / 2 + 1;

  std::vector<int64_t> largeArray1(kHalfSize, 1);
  std::vector<int64_t> largeArray2(kHalfSize, 2);

  auto array1 = makeArrayVector<int64_t>({largeArray1});
  auto array2 = makeArrayVector<int64_t>({largeArray2});

  VELOX_ASSERT_THROW(
      evaluate("concat(c0, c1)", makeRowVector({array1, array2})),
      "Unsuccessful try to concat arrays with");
}

// Test concatenation with boolean arrays
TEST_F(ArrayConcatTest, booleanArrays) {
  auto array1 = makeArrayVector<bool>({{true, false, true}});
  auto array2 = makeArrayVector<bool>({{false, true}});

  VectorPtr expected =
      makeArrayVector<bool>({{true, false, true, false, true}});

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test with timestamp arrays
TEST_F(ArrayConcatTest, timestampArrays) {
  auto array1 = makeArrayVector<Timestamp>({
      {Timestamp(1000, 0), Timestamp(2000, 0)},
  });
  auto array2 = makeArrayVector<Timestamp>({
      {Timestamp(3000, 0)},
  });

  VectorPtr expected = makeArrayVector<Timestamp>({
      {Timestamp(1000, 0), Timestamp(2000, 0), Timestamp(3000, 0)},
  });

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
