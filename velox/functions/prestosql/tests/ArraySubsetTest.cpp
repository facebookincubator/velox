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
#include "velox/vector/fuzzer/VectorFuzzer.h"

#include <random>

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

// Custom fuzzer tests to compare array_subset with equivalent expression
// using existing UDFs. The equivalent expression is:
// transform(
//   array_sort(array_distinct(filter(indices, i -> i IS NOT NULL AND i > 0 AND
//   i <= cardinality(arr)))), i -> element_at(arr, i)
// )
//
// Note: We already have explicit tests for NULL handling in ArraySubsetTest.
// These fuzzer tests focus on comparing behavior with non-null inputs.
class ArraySubsetFuzzerTest : public test::FunctionBaseTest {
 protected:
  // Helper to flatten vectors for consistent comparison across encodings
  template <typename T>
  static VectorPtr flatten(const std::shared_ptr<T>& vector) {
    VectorPtr result = vector;
    BaseVector::flattenVector(result);
    return result;
  }

  // The expression below is equivalent to array_subset(c0, c1) using existing
  // UDFs.
  static constexpr const char* kEquivalentExpression =
      "transform("
      "  array_sort(array_distinct("
      "    filter(c1, i -> i IS NOT NULL AND i > 0 AND i <= cardinality(c0))"
      "  )),"
      "  i -> element_at(c0, i)"
      ")";

  // Get a SelectivityVector that excludes rows where either input is null.
  // This is because array_subset returns NULL when either input is NULL,
  // but the equivalent expression may return empty array.
  static SelectivityVector getNonNullRows(const RowVectorPtr& data) {
    auto inputArray = data->childAt(0);
    auto indices = data->childAt(1);
    SelectivityVector nonNullRows(data->size());

    for (vector_size_t i = 0; i < data->size(); ++i) {
      if (inputArray->isNullAt(i) || indices->isNullAt(i)) {
        nonNullRows.setValid(i, false);
      }
    }
    nonNullRows.updateBounds();
    return nonNullRows;
  }

  void testEquivalence(const RowVectorPtr& data) {
    auto result = evaluate("array_subset(c0, c1)", data);
    auto expected = evaluate(kEquivalentExpression, data);

    // Get rows where neither input is null
    auto nonNullRows = getNonNullRows(data);

    // Compare only non-null rows (null propagation is tested separately)
    nonNullRows.applyToSelected([&](vector_size_t row) {
      ASSERT_TRUE(expected->equalValueAt(result.get(), row, row))
          << "Mismatch at row " << row << ": expected "
          << expected->toString(row) << ", got " << result->toString(row);
    });
  }

  // Generate indices array with values in valid range [1, maxIndex].
  // This ensures we actually test the subsetting logic, not just filtering.
  ArrayVectorPtr makeValidIndicesArray(
      vector_size_t numRows,
      vector_size_t indicesPerRow,
      int32_t maxIndex,
      double nullRatio = 0.0) {
    std::vector<std::vector<std::optional<int32_t>>> indicesData(numRows);
    std::mt19937 rng(42); // Fixed seed for reproducibility

    for (vector_size_t row = 0; row < numRows; ++row) {
      indicesData[row].reserve(indicesPerRow);
      for (vector_size_t i = 0; i < indicesPerRow; ++i) {
        if (nullRatio > 0.0 &&
            (rng() % 100) < static_cast<int>(nullRatio * 100)) {
          indicesData[row].push_back(std::nullopt);
        } else {
          // Generate index in range [1, maxIndex] (1-based)
          int32_t index = (rng() % maxIndex) + 1;
          indicesData[row].push_back(index);
        }
      }
    }

    return makeNullableArrayVector(indicesData);
  }

  // Generate indices with mix of valid, invalid (out of bounds, zero,
  // negative).
  ArrayVectorPtr makeMixedIndicesArray(
      vector_size_t numRows,
      vector_size_t indicesPerRow,
      int32_t maxValidIndex,
      double nullRatio = 0.0) {
    std::vector<std::vector<std::optional<int32_t>>> indicesData(numRows);
    std::mt19937 rng(42);

    for (vector_size_t row = 0; row < numRows; ++row) {
      indicesData[row].reserve(indicesPerRow);
      for (vector_size_t i = 0; i < indicesPerRow; ++i) {
        if (nullRatio > 0.0 &&
            (rng() % 100) < static_cast<int>(nullRatio * 100)) {
          indicesData[row].push_back(std::nullopt);
        } else {
          int choice = rng() % 4;
          if (choice == 0) {
            // Valid index [1, maxValidIndex]
            indicesData[row].push_back((rng() % maxValidIndex) + 1);
          } else if (choice == 1) {
            // Out of bounds (too large)
            indicesData[row].push_back(maxValidIndex + (rng() % 100) + 1);
          } else if (choice == 2) {
            // Zero
            indicesData[row].push_back(0);
          } else {
            // Negative
            indicesData[row].push_back(-static_cast<int32_t>(rng() % 100) - 1);
          }
        }
      }
    }

    return makeNullableArrayVector(indicesData);
  }
};

// Fuzz test with flat vectors, no nulls, fixed-size arrays
TEST_F(ArraySubsetFuzzerTest, fuzzFlatNoNulls) {
  VectorFuzzer::Options options;
  options.vectorSize = 100;
  options.nullRatio = 0.0;
  options.containerHasNulls = false;
  options.containerLength = 10;
  options.containerVariableLength = false;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 10; ++i) {
    // Generate random integer arrays
    auto inputArray = fuzzer.fuzz(ARRAY(INTEGER()));

    // Generate indices in valid range [1, containerLength] to actually test
    // the subset extraction logic, not just the index filtering
    auto indices = makeValidIndicesArray(
        options.vectorSize, 5, options.containerLength, 0.0);

    auto data = makeRowVector({inputArray, indices});
    testEquivalence(data);
  }
}

// Fuzz test with nulls in arrays and indices
TEST_F(ArraySubsetFuzzerTest, fuzzWithNulls) {
  VectorFuzzer::Options options;
  options.vectorSize = 100;
  options.nullRatio = 0.1;
  options.containerHasNulls = true;
  options.containerLength = 10;
  options.containerVariableLength = true;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 10; ++i) {
    auto inputArray = fuzzer.fuzz(ARRAY(INTEGER()));
    // Use mixed indices (valid, invalid, nulls) to test various edge cases
    auto indices = makeMixedIndicesArray(
        options.vectorSize, 8, options.containerLength, options.nullRatio);

    auto data = makeRowVector({inputArray, indices});
    testEquivalence(data);
  }
}

// Fuzz test with dictionary-encoded vectors
TEST_F(ArraySubsetFuzzerTest, fuzzDictionaryEncoded) {
  VectorFuzzer::Options options;
  options.vectorSize = 100;
  options.nullRatio = 0.1;
  options.containerHasNulls = true;
  options.containerLength = 10;
  options.containerVariableLength = true;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 10; ++i) {
    // Generate base vectors
    auto baseInputArray = fuzzer.fuzz(ARRAY(INTEGER()));
    auto baseIndices = makeMixedIndicesArray(
        options.vectorSize, 8, options.containerLength, options.nullRatio);

    // Wrap in dictionary encoding
    auto inputArray = fuzzer.fuzzDictionary(baseInputArray, options.vectorSize);
    auto indices = fuzzer.fuzzDictionary(baseIndices, options.vectorSize);

    auto data = makeRowVector({inputArray, indices});

    // Flatten for comparison
    auto flatData = makeRowVector({flatten(inputArray), flatten(indices)});

    // Verify results match between dictionary and flat encodings
    auto result = evaluate("array_subset(c0, c1)", data);
    auto expectedResult = evaluate("array_subset(c0, c1)", flatData);
    assertEqualVectors(expectedResult, result);

    // Also verify against the equivalent expression
    testEquivalence(flatData);
  }
}

// Fuzz test with variable-length arrays and high null ratio
TEST_F(ArraySubsetFuzzerTest, fuzzVariableLengthWithHighNullRatio) {
  VectorFuzzer::Options options;
  options.vectorSize = 100;
  options.nullRatio = 0.3;
  options.containerHasNulls = true;
  options.containerLength = 20;
  options.containerVariableLength = true;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 10; ++i) {
    auto inputArray = fuzzer.fuzz(ARRAY(INTEGER()));
    // Use high null ratio in indices to match the test's intent
    auto indices = makeMixedIndicesArray(
        options.vectorSize, 10, options.containerLength, options.nullRatio);

    auto data = makeRowVector({inputArray, indices});
    testEquivalence(data);
  }
}

// Fuzz test with string arrays
TEST_F(ArraySubsetFuzzerTest, fuzzStringArrays) {
  VectorFuzzer::Options options;
  options.vectorSize = 100;
  options.nullRatio = 0.1;
  options.containerHasNulls = true;
  options.containerLength = 10;
  options.containerVariableLength = true;
  options.stringLength = 20;
  options.stringVariableLength = true;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 10; ++i) {
    auto inputArray = fuzzer.fuzz(ARRAY(VARCHAR()));
    auto indices = makeMixedIndicesArray(
        options.vectorSize, 8, options.containerLength, options.nullRatio);

    auto data = makeRowVector({inputArray, indices});
    testEquivalence(data);
  }
}

// Fuzz test with nested arrays (array of arrays)
TEST_F(ArraySubsetFuzzerTest, fuzzNestedArrays) {
  VectorFuzzer::Options options;
  options.vectorSize = 50;
  options.nullRatio = 0.1;
  options.containerHasNulls = true;
  options.containerLength = 5;
  options.containerVariableLength = true;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 10; ++i) {
    auto inputArray = fuzzer.fuzz(ARRAY(ARRAY(INTEGER())));
    auto indices = makeMixedIndicesArray(
        options.vectorSize, 5, options.containerLength, options.nullRatio);

    auto data = makeRowVector({inputArray, indices});
    testEquivalence(data);
  }
}

// Fuzz test with empty arrays
TEST_F(ArraySubsetFuzzerTest, fuzzWithEmptyArrays) {
  VectorFuzzer::Options options;
  options.vectorSize = 100;
  options.nullRatio = 0.1;
  options.containerHasNulls = true;
  // Use very small container length to increase likelihood of empty arrays
  options.containerLength = 2;
  options.containerVariableLength = true;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 10; ++i) {
    auto inputArray = fuzzer.fuzz(ARRAY(INTEGER()));
    auto indices = makeMixedIndicesArray(
        options.vectorSize, 3, options.containerLength, options.nullRatio);

    auto data = makeRowVector({inputArray, indices});
    testEquivalence(data);
  }
}

// Fuzz test with various numeric types
TEST_F(ArraySubsetFuzzerTest, fuzzVariousNumericTypes) {
  VectorFuzzer::Options options;
  options.vectorSize = 100;
  options.nullRatio = 0.1;
  options.containerHasNulls = true;
  options.containerLength = 10;
  options.containerVariableLength = true;

  VectorFuzzer fuzzer(options, pool());

  // Test with BIGINT
  for (auto i = 0; i < 5; ++i) {
    auto inputArray = fuzzer.fuzz(ARRAY(BIGINT()));
    auto indices = makeMixedIndicesArray(
        options.vectorSize, 8, options.containerLength, options.nullRatio);
    auto data = makeRowVector({inputArray, indices});
    testEquivalence(data);
  }

  // Test with DOUBLE
  for (auto i = 0; i < 5; ++i) {
    auto inputArray = fuzzer.fuzz(ARRAY(DOUBLE()));
    auto indices = makeMixedIndicesArray(
        options.vectorSize, 8, options.containerLength, options.nullRatio);
    auto data = makeRowVector({inputArray, indices});
    testEquivalence(data);
  }

  // Test with BOOLEAN
  for (auto i = 0; i < 5; ++i) {
    auto inputArray = fuzzer.fuzz(ARRAY(BOOLEAN()));
    auto indices = makeMixedIndicesArray(
        options.vectorSize, 8, options.containerLength, options.nullRatio);
    auto data = makeRowVector({inputArray, indices});
    testEquivalence(data);
  }
}

// Fuzz test with constant vectors
TEST_F(ArraySubsetFuzzerTest, fuzzConstantVectors) {
  VectorFuzzer::Options options;
  options.vectorSize = 100;
  options.nullRatio = 0.0;
  options.containerHasNulls = false;
  options.containerLength = 5;
  options.containerVariableLength = false;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 10; ++i) {
    // Create a constant input array
    auto inputArray = fuzzer.fuzzConstant(ARRAY(INTEGER()), options.vectorSize);

    // Create varying indices with mix of valid and invalid values
    auto indices =
        makeMixedIndicesArray(options.vectorSize, 5, options.containerLength);

    auto data = makeRowVector({inputArray, indices});
    testEquivalence(data);
  }
}

// Stress test with large vectors
TEST_F(ArraySubsetFuzzerTest, fuzzLargeVectors) {
  VectorFuzzer::Options options;
  options.vectorSize = 1000;
  options.nullRatio = 0.1;
  options.containerHasNulls = true;
  options.containerLength = 50;
  options.containerVariableLength = true;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 5; ++i) {
    auto inputArray = fuzzer.fuzz(ARRAY(INTEGER()));
    auto indices = makeMixedIndicesArray(
        options.vectorSize, 20, options.containerLength, options.nullRatio);

    auto data = makeRowVector({inputArray, indices});
    testEquivalence(data);
  }
}

} // namespace
} // namespace facebook::velox::functions
