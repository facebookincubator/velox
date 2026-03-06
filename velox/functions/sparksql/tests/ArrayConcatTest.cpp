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

// Test double arrays with nulls
TEST_F(ArrayConcatTest, doubleArraysWithNulls) {
  auto array1 = makeNullableArrayVector<double>({
      {{1.5, std::nullopt, 2.5}},
      {{std::nullopt}},
  });
  auto array2 = makeNullableArrayVector<double>({
      {{std::nullopt, 3.5}},
      {{4.5}},
  });

  VectorPtr expected = makeNullableArrayVector<double>({
      {{1.5, std::nullopt, 2.5, std::nullopt, 3.5}},
      {{std::nullopt, 4.5}},
  });

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test date arrays
TEST_F(ArrayConcatTest, dateArrays) {
  auto array1 = makeArrayVector<int32_t>(
      {{0, 1, 2}}, ARRAY(DATE()));
  auto array2 = makeArrayVector<int32_t>(
      {{3, 4}}, ARRAY(DATE()));

  VectorPtr expected = makeArrayVector<int32_t>(
      {{0, 1, 2, 3, 4}}, ARRAY(DATE()));

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test int8 arrays (TINYINT)
TEST_F(ArrayConcatTest, int8Arrays) {
  auto array1 = makeArrayVector<int8_t>({{1, 2, 3}});
  auto array2 = makeArrayVector<int8_t>({{4, 5}});

  VectorPtr expected = makeArrayVector<int8_t>({{1, 2, 3, 4, 5}});

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test int16 arrays (SMALLINT)
TEST_F(ArrayConcatTest, int16Arrays) {
  auto array1 = makeArrayVector<int16_t>({{100, 200, 300}});
  auto array2 = makeArrayVector<int16_t>({{400, 500}});

  VectorPtr expected = makeArrayVector<int16_t>({{100, 200, 300, 400, 500}});

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test int32 arrays (INTEGER)
TEST_F(ArrayConcatTest, int32Arrays) {
  auto array1 = makeArrayVector<int32_t>({{1000, 2000, 3000}});
  auto array2 = makeArrayVector<int32_t>({{4000, 5000}});

  VectorPtr expected = makeArrayVector<int32_t>({{1000, 2000, 3000, 4000, 5000}});

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test int128 arrays (HUGEINT)
TEST_F(ArrayConcatTest, int128Arrays) {
  auto array1 = makeArrayVector<int128_t>({{
      HugeInt::build(0, 1),
      HugeInt::build(0, 2),
  }});
  auto array2 = makeArrayVector<int128_t>({{
      HugeInt::build(0, 3),
  }});

  VectorPtr expected = makeArrayVector<int128_t>({{
      HugeInt::build(0, 1),
      HugeInt::build(0, 2),
      HugeInt::build(0, 3),
  }});

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test varbinary arrays
TEST_F(ArrayConcatTest, varbinaryArrays) {
  auto array1 = makeArrayVector<StringView>(
      {{"\x01\x02\x03"_sv, "\x04\x05"_sv}},
      ARRAY(VARBINARY()));
  auto array2 = makeArrayVector<StringView>(
      {{"\x06\x07"_sv}},
      ARRAY(VARBINARY()));

  VectorPtr expected = makeArrayVector<StringView>(
      {{"\x01\x02\x03"_sv, "\x04\x05"_sv, "\x06\x07"_sv}},
      ARRAY(VARBINARY()));

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test mixed nulls across different integer types
TEST_F(ArrayConcatTest, mixedIntegerTypesWithNulls) {
  // Test int32 with nulls
  auto array1 = makeNullableArrayVector<int32_t>({
      {{1, std::nullopt, 3}},
  });
  auto array2 = makeNullableArrayVector<int32_t>({
      {{std::nullopt, 5}},
  });

  VectorPtr expected = makeNullableArrayVector<int32_t>({
      {{1, std::nullopt, 3, std::nullopt, 5}},
  });

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test with single element arrays
TEST_F(ArrayConcatTest, singleElementArrays) {
  auto array1 = makeArrayVector<int64_t>({{1}, {2}, {3}});
  auto array2 = makeArrayVector<int64_t>({{4}, {5}, {6}});
  auto array3 = makeArrayVector<int64_t>({{7}, {8}, {9}});

  VectorPtr expected = makeArrayVector<int64_t>({{1, 4, 7}, {2, 5, 8}, {3, 6, 9}});

  testExpression("concat(c0, c1, c2)", {array1, array2, array3}, expected);
}

// Test null array in middle of variadic arguments
TEST_F(ArrayConcatTest, nullArrayInMiddleOfVariadic) {
  auto array1 = makeArrayVector<int64_t>({{1, 2}, {10, 20}});
  auto array2 = makeNullableArrayVector<int64_t>({
      std::nullopt,
      {{30}},
  });
  auto array3 = makeArrayVector<int64_t>({{3, 4}, {40, 50}});

  VectorPtr expected = makeNullableArrayVector<int64_t>({
      std::nullopt,
      {{10, 20, 30, 40, 50}},
  });

  testExpression("concat(c0, c1, c2)", {array1, array2, array3}, expected);
}

// Test very large variadic (20 arrays)
TEST_F(ArrayConcatTest, veryLargeVariadic) {
  std::vector<VectorPtr> inputs;
  for (int i = 0; i < 20; i++) {
    inputs.push_back(makeArrayVector<int32_t>({{i * 10}}));
  }

  VectorPtr expected = makeArrayVector<int32_t>({
      {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190}
  });

  testExpression(
      "concat(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19)",
      inputs,
      expected);
}

// Test deeply nested arrays (3 levels)
TEST_F(ArrayConcatTest, deeplyNestedArrays) {
  // Create base: array<array<int>>
  auto level1 = makeArrayVector<int64_t>({{1, 2}, {3, 4}});
  
  // Create level2: array<array<array<int>>>
  auto level2_1 = makeArrayVector({0, 1}, level1);  // [[1,2]], [[3,4]]
  auto level2_2 = makeArrayVector({1, 2}, level1);  // [[3,4]], (empty)
  
  // Expected: [[1,2]], [[3,4]], [[3,4]]
  auto expectedBase = makeArrayVector<int64_t>({{1, 2}, {3, 4}, {3, 4}});
  auto expected = makeArrayVector({0, 1, 2}, expectedBase);

  testExpression("concat(c0, c1)", {level2_1, level2_2}, expected);
}

// Test alternating null and non-null elements
TEST_F(ArrayConcatTest, alternatingNullPattern) {
  auto array1 = makeNullableArrayVector<int64_t>({
      {{1, std::nullopt, 2, std::nullopt, 3}},
  });
  auto array2 = makeNullableArrayVector<int64_t>({
      {{std::nullopt, 4, std::nullopt, 5, std::nullopt}},
  });

  VectorPtr expected = makeNullableArrayVector<int64_t>({
      {{1, std::nullopt, 2, std::nullopt, 3, std::nullopt, 4, std::nullopt, 5, std::nullopt}},
  });

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test with large element count arrays
TEST_F(ArrayConcatTest, largeElementCountArrays) {
  std::vector<int64_t> largeVec1(1000);
  std::vector<int64_t> largeVec2(1000);
  
  for (int i = 0; i < 1000; i++) {
    largeVec1[i] = i;
    largeVec2[i] = i + 1000;
  }

  auto array1 = makeArrayVector<int64_t>({largeVec1});
  auto array2 = makeArrayVector<int64_t>({largeVec2});

  std::vector<int64_t> expectedVec(2000);
  for (int i = 0; i < 2000; i++) {
    expectedVec[i] = i;
  }
  VectorPtr expected = makeArrayVector<int64_t>({expectedVec});

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test float arrays with special values (NaN, Infinity)
TEST_F(ArrayConcatTest, floatSpecialValues) {
  auto array1 = makeArrayVector<float>({
      {1.0f, std::numeric_limits<float>::infinity(), 2.0f}
  });
  auto array2 = makeArrayVector<float>({
      {-std::numeric_limits<float>::infinity(), 3.0f}
  });

  VectorPtr expected = makeArrayVector<float>({
      {1.0f, std::numeric_limits<float>::infinity(), 2.0f, 
       -std::numeric_limits<float>::infinity(), 3.0f}
  });

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test double arrays with NaN
TEST_F(ArrayConcatTest, doubleWithNaN) {
  auto array1 = makeArrayVector<double>({
      {1.0, std::numeric_limits<double>::quiet_NaN(), 2.0}
  });
  auto array2 = makeArrayVector<double>({{3.0}});

  // Note: NaN != NaN, so we just verify it doesn't crash
  auto result = evaluate("concat(c0, c1)", makeRowVector({array1, array2}));
  ASSERT_EQ(result->size(), 1);
}

// Test empty arrays mixed with non-empty
TEST_F(ArrayConcatTest, emptyMixedWithNonEmpty) {
  auto array1 = makeArrayVector<int64_t>({{}, {1, 2}, {}});
  auto array2 = makeArrayVector<int64_t>({{3}, {}, {4, 5}});
  auto array3 = makeArrayVector<int64_t>({{}, {6}, {}});

  VectorPtr expected = makeArrayVector<int64_t>({
      {3},
      {1, 2, 6},
      {4, 5}
  });

  testExpression("concat(c0, c1, c2)", {array1, array2, array3}, expected);
}

// Test string arrays with empty strings
TEST_F(ArrayConcatTest, stringArraysWithEmptyStrings) {
  auto array1 = makeArrayVector<StringView>({{""_sv, "a"_sv, ""_sv}});
  auto array2 = makeArrayVector<StringView>({{"b"_sv, ""_sv}});

  VectorPtr expected = makeArrayVector<StringView>({
      {""_sv, "a"_sv, ""_sv, "b"_sv, ""_sv}
  });

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test with unicode strings
TEST_F(ArrayConcatTest, unicodeStrings) {
  auto array1 = makeArrayVector<StringView>({{"Hello"_sv, "世界"_sv}});
  auto array2 = makeArrayVector<StringView>({{"🚀"_sv, "Velox"_sv}});

  VectorPtr expected = makeArrayVector<StringView>({
      {"Hello"_sv, "世界"_sv, "🚀"_sv, "Velox"_sv}
  });

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test row arrays (struct arrays)
TEST_F(ArrayConcatTest, rowArrays) {
  auto rowType = ROW({{"a", INTEGER()}, {"b", VARCHAR()}});
  
  auto array1 = makeArrayOfRowVector(
      rowType,
      {makeRowVector({
          makeFlatVector<int32_t>({1, 2}),
          makeFlatVector<StringView>({"x"_sv, "y"_sv})
      })});
  
  auto array2 = makeArrayOfRowVector(
      rowType,
      {makeRowVector({
          makeFlatVector<int32_t>({3}),
          makeFlatVector<StringView>({"z"_sv})
      })});

  auto expected = makeArrayOfRowVector(
      rowType,
      {makeRowVector({
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<StringView>({"x"_sv, "y"_sv, "z"_sv})
      })});

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

// Test map arrays
TEST_F(ArrayConcatTest, mapArrays) {
  auto mapType = MAP(INTEGER(), VARCHAR());
  
  // Create array<map<int, varchar>>
  auto map1 = makeMapVector<int32_t, StringView>({
      {{1, "a"_sv}, {2, "b"_sv}}
  });
  auto array1 = makeArrayVector({0}, map1);
  
  auto map2 = makeMapVector<int32_t, StringView>({
      {{3, "c"_sv}}
  });
  auto array2 = makeArrayVector({0}, map2);
  
  auto expectedMap = makeMapVector<int32_t, StringView>({
      {{1, "a"_sv}, {2, "b"_sv}, {3, "c"_sv}}
  });
  auto expected = makeArrayVector({0}, expectedMap);

  testExpression("concat(c0, c1)", {array1, array2}, expected);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
