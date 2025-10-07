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

#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class ArraySubsetTest : public test::FunctionBaseTest {
 protected:
  void testArraySubset(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }
};

TEST_F(ArraySubsetTest, integerArray) {
  auto inputArray = makeArrayVector<int32_t>({
      {1, 2, 3, 4, 5, 6},
      {10, 20, 30, 40},
      {},
      {100, 200},
  });

  auto indices = makeArrayVector<int32_t>({
      {1, 3, 5},
      {2, 4, 6},
      {1, 2},
      {1, 2, 3},
  });

  auto expected = makeArrayVector<int32_t>({
      {1, 3, 5},
      {20, 40},
      {},
      {100, 200},
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, constantIndices) {
  auto inputArray = makeArrayVector<int32_t>({
      {1, 2, 3, 4, 5},
      {10, 20, 30},
      {100, 200, 300, 400, 500},
  });

  auto expected = makeArrayVector<int32_t>({
      {1, 3, 5},
      {10, 30},
      {100, 300, 500},
  });

  auto result = evaluate(
      "array_subset(c0, array_constructor(cast(1 as integer), cast(3 as integer), cast(5 as integer)))",
      makeRowVector({inputArray}));
  assertEqualVectors(expected, result);
}

TEST_F(ArraySubsetTest, outOfBoundsIndices) {
  auto inputArray = makeArrayVector<int32_t>({
      {1, 2, 3},
      {10, 20},
  });

  auto indices = makeArrayVector<int32_t>({
      {1, 5, 2}, // Index 5 is out of bounds
      {1, 3, 4}, // Indices 3 and 4 are out of bounds
  });

  auto expected = makeArrayVector<int32_t>({
      {1, 2}, // Only valid indices 1 and 2
      {10}, // Only valid index 1
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, negativeAndZeroIndices) {
  auto inputArray = makeArrayVector<int32_t>({
      {1, 2, 3, 4, 5},
      {10, 20, 30},
  });

  auto indices = makeArrayVector<int32_t>({
      {-1, 0, 1, 3}, // Negative and zero indices should be ignored
      {0, 2, -5}, // Only index 2 is valid
  });

  auto expected = makeArrayVector<int32_t>({
      {1, 3}, // Only indices 1 and 3 are valid
      {20}, // Only index 2 is valid
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, duplicateIndices) {
  auto inputArray = makeArrayVector<int32_t>({
      {1, 2, 3, 4, 5},
      {10, 20, 30},
  });

  auto indices = makeArrayVector<int32_t>({
      {1, 1, 2, 2, 3}, // Duplicate indices
      {2, 2, 1, 1}, // Duplicate indices
  });

  auto expected = makeArrayVector<int32_t>({
      {1, 2, 3}, // Duplicates removed, sorted order
      {10, 20}, // Duplicates removed, sorted order (1-based: 1->10, 2->20)
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, emptyIndices) {
  auto inputArray = makeArrayVector<int32_t>({
      {1, 2, 3, 4, 5},
      {10, 20, 30},
  });

  auto indices = makeArrayVector<int32_t>({
      {},
      {},
  });

  auto expected = makeArrayVector<int32_t>({
      {},
      {},
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, emptyIndicesWithEmptyArray) {
  // Test edge case: empty array with empty indices
  auto inputArray = makeArrayVector<int32_t>({
      {},
  });

  auto indices = makeArrayVector<int32_t>({
      {},
  });

  auto expected = makeArrayVector<int32_t>({
      {},
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, emptyArrayWithValidIndices) {
  // Test edge case: empty array with valid indices (should return empty)
  auto inputArray = makeArrayVector<int32_t>({
      {},
  });

  auto indices = makeArrayVector<int32_t>({
      {1, 2, 3},
  });

  auto expected = makeArrayVector<int32_t>({
      {},
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, emptyArray) {
  auto inputArray = makeArrayVector<int32_t>({
      {},
      {},
  });

  auto indices = makeArrayVector<int32_t>({
      {1, 2, 3},
      {1},
  });

  auto expected = makeArrayVector<int32_t>({
      {},
      {},
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, nullElementsInArray) {
  auto inputArray = makeNullableArrayVector<int32_t>({
      {1, std::nullopt, 3, 4, std::nullopt},
      {std::nullopt, 20, std::nullopt, 40},
  });

  auto indices = makeArrayVector<int32_t>({
      {1, 2, 3, 5}, // Includes indices pointing to null elements
      {1, 2, 3, 4}, // Includes indices pointing to null elements
  });

  // Null elements should be included in output
  auto expected = makeNullableArrayVector<int32_t>({
      {1, std::nullopt, 3, std::nullopt}, // Elements at indices 1, 2, 3, 5
      {std::nullopt, 20, std::nullopt, 40}, // Elements at indices 1, 2, 3, 4
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, nullElementsInIndices) {
  auto inputArray = makeArrayVector<int32_t>({
      {1, 2, 3, 4, 5},
      {10, 20, 30},
  });

  auto indices = makeNullableArrayVector<int32_t>({
      {1, std::nullopt, 3, std::nullopt, 5}, // Null indices should be ignored
      {std::nullopt, 2, std::nullopt}, // Null indices should be ignored
  });

  auto expected = makeArrayVector<int32_t>({
      {1, 3, 5}, // Only valid non-null indices
      {20}, // Only valid non-null index
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, stringArray) {
  auto inputArray = makeArrayVector<StringView>({
      {"apple", "banana", "cherry", "date"},
      {"hello", "world"},
      {"a", "b", "c", "d", "e"},
  });

  auto indices = makeArrayVector<int32_t>({
      {2, 4, 1},
      {1, 2, 3},
      {1, 3, 5},
  });

  auto expected = makeArrayVector<StringView>({
      {"apple", "banana", "date"}, // Sorted by index: 1, 2, 4
      {"hello", "world"}, // Indices 1, 2 (3 is out of bounds)
      {"a", "c", "e"}, // Indices 1, 3, 5
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, floatArray) {
  auto inputArray = makeArrayVector<float>({
      {1.1f, 2.2f, 3.3f, 4.4f},
      {10.5f, 20.5f, 30.5f},
  });

  auto indices = makeArrayVector<int32_t>({
      {1, 3, 4},
      {2, 1, 3},
  });

  auto expected = makeArrayVector<float>({
      {1.1f, 3.3f, 4.4f},
      {10.5f, 20.5f, 30.5f},
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, booleanArray) {
  auto inputArray = makeArrayVector<bool>({
      {true, false, true, false, true},
      {false, true, false},
  });

  auto indices = makeArrayVector<int32_t>({
      {1, 3, 5},
      {2, 1, 3},
  });

  auto expected = makeArrayVector<bool>({
      {true, true, true},
      {false, true, false},
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, largeIndices) {
  auto inputArray = makeArrayVector<int32_t>({
      {1, 2, 3},
  });

  auto indices = makeArrayVector<int32_t>({
      {1000, 2000, 1, 3000}, // Very large indices mixed with valid ones
  });

  auto expected = makeArrayVector<int32_t>({
      {1}, // Only index 1 is valid
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, singleElementArray) {
  auto inputArray = makeArrayVector<int32_t>({
      {42},
      {100},
  });

  auto indices = makeArrayVector<int32_t>({
      {1, 2, 3}, // Only index 1 is valid
      {1}, // Valid index
  });

  auto expected = makeArrayVector<int32_t>({
      {42},
      {100},
  });

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, complexArray) {
  // Test with array of arrays
  auto innerArrays = makeArrayVector<int32_t>({
      {1, 2},
      {3, 4, 5},
      {6},
      {7, 8, 9, 10},
  });

  auto inputArray = makeArrayVector({0, 2, 4}, innerArrays);

  auto indices = makeArrayVector<int32_t>({
      {1, 2},
      {2, 1},
  });

  // Row 0: array_subset([{1,2}, {3,4,5}], [1,2]) = [{1,2}, {3,4,5}]
  // Row 1: array_subset([{6}, {7,8,9,10}], [2,1]) = [{6}, {7,8,9,10}] (sorted
  // by index)
  auto expectedInnerArrays = makeArrayVector<int32_t>({
      {1, 2},
      {3, 4, 5},
      {6},
      {7, 8, 9, 10},
  });

  auto expected = makeArrayVector({0, 2, 4}, expectedInnerArrays);

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, nullAtFirstIndex) {
  // Test the specific case: array_subset(array[null,1], array[1]) = array[null]
  std::vector<std::vector<std::optional<int32_t>>> inputData = {
      {std::nullopt, 1}};
  auto inputArray = makeNullableArrayVector(inputData);

  auto indices = makeArrayVector<int32_t>({{1}});

  std::vector<std::vector<std::optional<int32_t>>> expectedData = {
      {std::nullopt}};
  auto expected = makeNullableArrayVector(expectedData);

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, nullArguments) {
  // Test when either argument is null, result should be null
  auto inputArray = makeArrayVector<int32_t>({{1, 2, 3}});
  auto indices = makeArrayVector<int32_t>({{1, 2}});

  // Create null array and null indices with proper types
  auto nullArray = BaseVector::createNullConstant(ARRAY(INTEGER()), 1, pool());
  auto nullIndices =
      BaseVector::createNullConstant(ARRAY(INTEGER()), 1, pool());

  // Test null input array
  auto result1 =
      evaluate("array_subset(c0, c1)", makeRowVector({nullArray, indices}));
  auto expected1 = BaseVector::createNullConstant(ARRAY(INTEGER()), 1, pool());
  assertEqualVectors(expected1, result1);

  // Test null indices array
  auto result2 = evaluate(
      "array_subset(c0, c1)", makeRowVector({inputArray, nullIndices}));
  auto expected2 = BaseVector::createNullConstant(ARRAY(INTEGER()), 1, pool());
  assertEqualVectors(expected2, result2);

  // Test both null
  auto result3 =
      evaluate("array_subset(c0, c1)", makeRowVector({nullArray, nullIndices}));
  auto expected3 = BaseVector::createNullConstant(ARRAY(INTEGER()), 1, pool());
  assertEqualVectors(expected3, result3);
}

TEST_F(ArraySubsetTest, duplicateIndicesWithNullAndOutOfBounds) {
  // Test the specific case: array_subset(array[null,1], array[1,1,3,1]) =
  // array[null]
  std::vector<std::vector<std::optional<int32_t>>> inputData = {
      {std::nullopt, 1}};
  auto inputArray = makeNullableArrayVector(inputData);

  auto indices = makeArrayVector<int32_t>({{1, 1, 3, 1}});

  std::vector<std::vector<std::optional<int32_t>>> expectedData = {
      {std::nullopt}};
  auto expected = makeNullableArrayVector(expectedData);

  testArraySubset("array_subset(c0, c1)", {inputArray, indices}, expected);
}

TEST_F(ArraySubsetTest, basicTest) {
  auto inputArray = makeArrayVector<int32_t>({{1, 2, 3, 4, 5}});
  auto indices = makeArrayVector<int32_t>({{1, 3, 5}});
  auto expected = makeArrayVector<int32_t>({{1, 3, 5}});

  auto result =
      evaluate("array_subset(c0, c1)", makeRowVector({inputArray, indices}));
  assertEqualVectors(expected, result);
}

TEST_F(ArraySubsetTest, nullTest) {
  // Test array_subset(array[null,1], array[1]) = array[null]
  std::vector<std::vector<std::optional<int32_t>>> inputData = {
      {std::nullopt, 1}};
  auto inputArray = makeNullableArrayVector(inputData);
  auto indices = makeArrayVector<int32_t>({{1}});
  std::vector<std::vector<std::optional<int32_t>>> expectedData = {
      {std::nullopt}};
  auto expected = makeNullableArrayVector(expectedData);

  auto result =
      evaluate("array_subset(c0, c1)", makeRowVector({inputArray, indices}));
  assertEqualVectors(expected, result);
}

} // namespace
} // namespace facebook::velox::functions
