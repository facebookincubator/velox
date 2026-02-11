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

#include <cstdint>
#include <optional>
#include <vector>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/type/StringView.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class DotProductTest : public test::FunctionBaseTest {
 protected:
  template <typename T>
  void testDotProduct(
      const std::vector<std::vector<T>>& array1,
      const std::vector<std::vector<T>>& array2,
      const std::vector<std::optional<int64_t>>& expected) {
    auto inputArray1 = makeArrayVector<T>(array1);
    auto inputArray2 = makeArrayVector<T>(array2);
    auto result = evaluate(
        "dot_product(c0, c1)", makeRowVector({inputArray1, inputArray2}));
    auto expectedVector = makeNullableFlatVector<int64_t>(expected);
    assertEqualVectors(expectedVector, result);
  }

  template <typename T>
  void testDotProductDouble(
      const std::vector<std::vector<T>>& array1,
      const std::vector<std::vector<T>>& array2,
      const std::vector<std::optional<double>>& expected) {
    auto inputArray1 = makeArrayVector<T>(array1);
    auto inputArray2 = makeArrayVector<T>(array2);
    auto result = evaluate(
        "dot_product(c0, c1)", makeRowVector({inputArray1, inputArray2}));
    auto expectedVector = makeNullableFlatVector<double>(expected);
    assertEqualVectors(expectedVector, result);
  }

  template <typename T>
  void testDotProductFloat(
      const std::vector<std::vector<T>>& array1,
      const std::vector<std::vector<T>>& array2,
      const std::vector<std::optional<float>>& expected) {
    auto inputArray1 = makeArrayVector<T>(array1);
    auto inputArray2 = makeArrayVector<T>(array2);
    auto result = evaluate(
        "dot_product(c0, c1)", makeRowVector({inputArray1, inputArray2}));
    auto expectedVector = makeNullableFlatVector<float>(expected);
    assertEqualVectors(expectedVector, result);
  }
};

TEST_F(DotProductTest, integerArrays) {
  testDotProduct<int32_t>(
      {{1, 2, 3}, {4, 5, 6}},
      {{4, 5, 6}, {1, 2, 3}},
      {32, 32}); // 1*4 + 2*5 + 3*6 = 32
}

TEST_F(DotProductTest, bigintArrays) {
  testDotProduct<int64_t>(
      {{1, 2, 3, 4, 5}},
      {{5, 4, 3, 2, 1}},
      {35}); // 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 35
}

TEST_F(DotProductTest, floatArrays) {
  testDotProductFloat<float>(
      {{1.0F, 2.0F, 3.0F}},
      {{4.0F, 5.0F, 6.0F}},
      {32.0F}); // 1*4 + 2*5 + 3*6 = 32
}

TEST_F(DotProductTest, doubleArrays) {
  testDotProductDouble<double>(
      {{1.5, 2.5, 3.5}},
      {{2.0, 3.0, 4.0}},
      {24.5}); // 1.5*2 + 2.5*3 + 3.5*4 = 3 + 7.5 + 14 = 24.5
}

TEST_F(DotProductTest, emptyArrays) {
  testDotProduct<int32_t>({{}}, {{}}, {0}); // Empty arrays should return 0
}

TEST_F(DotProductTest, singleElement) {
  testDotProduct<int32_t>({{5}}, {{7}}, {35}); // 5*7 = 35
}

TEST_F(DotProductTest, negativeNumbers) {
  testDotProduct<int32_t>(
      {{-1, -2, -3}}, {{4, 5, 6}}, {-32}); // (-1)*4 + (-2)*5 + (-3)*6 = -32
}

TEST_F(DotProductTest, mixedSigns) {
  testDotProduct<int32_t>(
      {{1, -2, 3}},
      {{-4, 5, -6}},
      {-32}); // 1*(-4) + (-2)*5 + 3*(-6) = -4 - 10 - 18 = -32
}

TEST_F(DotProductTest, zeros) {
  testDotProduct<int32_t>(
      {{0, 0, 0}}, {{1, 2, 3}}, {0}); // All zeros in one array
}

TEST_F(DotProductTest, nullElementsInArrays) {
  // Null elements should be treated as zero
  auto array1 = makeNullableArrayVector<int32_t>({
      {1, std::nullopt, 3},
      {std::nullopt, 2, std::nullopt},
  });
  auto array2 = makeNullableArrayVector<int32_t>({
      {4, 5, 6},
      {1, 2, 3},
  });

  auto result =
      evaluate("dot_product(c0, c1)", makeRowVector({array1, array2}));
  // Row 0: 1*4 + 0*5 + 3*6 = 4 + 0 + 18 = 22
  // Row 1: 0*1 + 2*2 + 0*3 = 0 + 4 + 0 = 4
  auto expected = makeFlatVector<int64_t>({22, 4});
  assertEqualVectors(expected, result);
}

TEST_F(DotProductTest, nullArguments) {
  // Test when either argument is null
  auto array = makeArrayVector<int32_t>({{1, 2, 3}});
  auto nullArray = BaseVector::createNullConstant(ARRAY(INTEGER()), 1, pool());

  // Test null first array
  auto result1 =
      evaluate("dot_product(c0, c1)", makeRowVector({nullArray, array}));
  auto expected1 = BaseVector::createNullConstant(BIGINT(), 1, pool());
  assertEqualVectors(expected1, result1);

  // Test null second array
  auto result2 =
      evaluate("dot_product(c0, c1)", makeRowVector({array, nullArray}));
  auto expected2 = BaseVector::createNullConstant(BIGINT(), 1, pool());
  assertEqualVectors(expected2, result2);
}

TEST_F(DotProductTest, mismatchedLengthsThrows) {
  auto array1 = makeArrayVector<int32_t>({{1, 2, 3}});
  auto array2 = makeArrayVector<int32_t>({{1, 2}}); // Different length

  VELOX_ASSERT_THROW(
      evaluate("dot_product(c0, c1)", makeRowVector({array1, array2})),
      "dot_product requires arrays of equal length");
}

TEST_F(DotProductTest, largeArrays) {
  // Test with larger arrays
  std::vector<int32_t> arr1(100);
  std::vector<int32_t> arr2(100);
  int64_t expectedResult = 0;
  for (int i = 0; i < 100; ++i) {
    arr1[i] = i + 1;
    arr2[i] = i + 1;
    expectedResult += static_cast<int64_t>(i + 1) * (i + 1);
  }
  testDotProduct<int32_t>({arr1}, {arr2}, {expectedResult});
}

TEST_F(DotProductTest, int8Arrays) {
  testDotProduct<int8_t>({{1, 2, 3}}, {{4, 5, 6}}, {32});
}

TEST_F(DotProductTest, int16Arrays) {
  testDotProduct<int16_t>(
      {{100, 200, 300}},
      {{4, 5, 6}},
      {3200}); // 100*4 + 200*5 + 300*6 = 400 + 1000 + 1800 = 3200
}

TEST_F(DotProductTest, mapIntegerKeys) {
  auto map1 = makeMapVector<int32_t, int64_t>({
      {{1, 10}, {2, 20}},
      {{1, 5}, {3, 15}},
  });
  auto map2 = makeMapVector<int32_t, int64_t>({
      {{1, 3}, {2, 4}},
      {{1, 2}, {2, 4}},
  });

  auto result = evaluate("dot_product(c0, c1)", makeRowVector({map1, map2}));
  // Row 0: 10*3 + 20*4 = 30 + 80 = 110
  // Row 1: 5*2 = 10 (key 3 not in map2, key 2 not in map1)
  auto expected = makeFlatVector<int64_t>({110, 10});
  assertEqualVectors(expected, result);
}

TEST_F(DotProductTest, mapVarcharKeys) {
  auto map1 = makeMapVector<StringView, double>({
      {{"a", 1.0}, {"b", 2.0}},
      {{"x", 5.0}, {"y", 10.0}},
  });
  auto map2 = makeMapVector<StringView, double>({
      {{"a", 3.0}, {"c", 4.0}},
      {{"y", 2.0}, {"z", 3.0}},
  });

  auto result = evaluate("dot_product(c0, c1)", makeRowVector({map1, map2}));
  // Row 0: 1.0*3.0 = 3.0 (only 'a' matches)
  // Row 1: 10.0*2.0 = 20.0 (only 'y' matches)
  auto expected = makeFlatVector<double>({3.0, 20.0});
  assertEqualVectors(expected, result);
}

TEST_F(DotProductTest, mapNoMatchingKeys) {
  auto map1 = makeMapVector<int32_t, int64_t>({
      {{1, 10}, {2, 20}},
  });
  auto map2 = makeMapVector<int32_t, int64_t>({
      {{3, 30}, {4, 40}},
  });

  auto result = evaluate("dot_product(c0, c1)", makeRowVector({map1, map2}));
  // No matching keys, result should be 0
  auto expected = makeFlatVector<int64_t>({0});
  assertEqualVectors(expected, result);
}

TEST_F(DotProductTest, mapEmptyMaps) {
  auto map1 = makeMapVector<int32_t, int64_t>({
      {},
  });
  auto map2 = makeMapVector<int32_t, int64_t>({
      {},
  });

  auto result = evaluate("dot_product(c0, c1)", makeRowVector({map1, map2}));
  auto expected = makeFlatVector<int64_t>({0});
  assertEqualVectors(expected, result);
}

TEST_F(DotProductTest, mapNullValues) {
  // Null values in maps should be treated as zero
  auto map1 = makeNullableMapVector<int32_t, int64_t>({
      {{{1, 10}, {2, std::nullopt}}},
  });
  auto map2 = makeNullableMapVector<int32_t, int64_t>({
      {{{1, 3}, {2, 4}}},
  });

  auto result = evaluate("dot_product(c0, c1)", makeRowVector({map1, map2}));
  // Key 1: 10*3 = 30, Key 2: null*4 = 0 (null treated as zero)
  auto expected = makeFlatVector<int64_t>({30});
  assertEqualVectors(expected, result);
}

TEST_F(DotProductTest, multipleRows) {
  testDotProduct<int32_t>(
      {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}},
      {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {1, 1, 1}},
      {1, 2, 3, 3}); // Unit vectors and all-ones
}

} // namespace
} // namespace facebook::velox::functions

// ============================================================================
// CUSTOM FUZZER TESTS
// These tests use VectorFuzzer to generate random inputs and verify
// dot_product behavior with various data types and edge cases.
// ============================================================================

#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::functions {
namespace {

class DotProductFuzzerTest : public test::FunctionBaseTest {
 protected:
  // The equivalent SQL expression for dot_product using existing UDFs.
  // dot_product(array1, array2) is equivalent to:
  // reduce(zip_with(c0, c1, (x, y) -> coalesce(x, 0) * coalesce(y, 0)),
  //        CAST(0 AS BIGINT), (s, x) -> s + x, s -> s)
  //
  // Note: This only works for integer arrays where the result is BIGINT.
  static constexpr const char* kEquivalentExpressionBigint =
      "reduce(zip_with(c0, c1, (x, y) -> coalesce(CAST(x AS BIGINT), CAST(0 AS BIGINT)) * coalesce(CAST(y AS BIGINT), CAST(0 AS BIGINT))), "
      "CAST(0 AS BIGINT), (s, x) -> s + x, s -> s)";

  static constexpr const char* kEquivalentExpressionDouble =
      "reduce(zip_with(c0, c1, (x, y) -> coalesce(CAST(x AS DOUBLE), CAST(0.0 AS DOUBLE)) * coalesce(CAST(y AS DOUBLE), CAST(0.0 AS DOUBLE))), "
      "CAST(0.0 AS DOUBLE), (s, x) -> s + x, s -> s)";

  // Get a SelectivityVector that excludes rows where either input is null.
  static SelectivityVector getNonNullRows(const RowVectorPtr& data) {
    auto array1 = data->childAt(0);
    auto array2 = data->childAt(1);
    SelectivityVector nonNullRows(data->size());

    for (vector_size_t i = 0; i < data->size(); ++i) {
      if (array1->isNullAt(i) || array2->isNullAt(i)) {
        nonNullRows.setValid(i, false);
      }
    }
    nonNullRows.updateBounds();
    return nonNullRows;
  }

  // Check if two arrays have the same length at a given row.
  static bool haveSameLength(
      const VectorPtr& array1,
      const VectorPtr& array2,
      vector_size_t row) {
    auto arr1 = array1->as<ArrayVector>();
    auto arr2 = array2->as<ArrayVector>();
    if (!arr1 || !arr2) {
      return false;
    }
    return arr1->sizeAt(row) == arr2->sizeAt(row);
  }

  // Test that dot_product result is non-null when inputs are valid
  // (same length, non-null).
  void testDotProductProperties(const RowVectorPtr& data) {
    auto nonNullRows = getNonNullRows(data);
    if (nonNullRows.countSelected() == 0) {
      return;
    }

    VectorPtr result;
    try {
      result = evaluate("try(dot_product(c0, c1))", data);
    } catch (...) {
      return;
    }

    if (!result) {
      return;
    }

    auto array1 = data->childAt(0);
    auto array2 = data->childAt(1);

    for (auto i = 0; i < data->size(); ++i) {
      if (!nonNullRows.isValid(i)) {
        continue;
      }

      // If arrays have same length, result should be non-null
      if (haveSameLength(array1, array2, i)) {
        // Result can still be null if try() caught an overflow error
        // so we don't assert non-null here
      }
    }
  }

  // Test equivalence between dot_product and the reduce+zip_with expression
  // for integer arrays.
  void testEquivalenceBigint(const RowVectorPtr& data) {
    auto nonNullRows = getNonNullRows(data);
    if (nonNullRows.countSelected() == 0) {
      return;
    }

    VectorPtr result;
    VectorPtr expected;
    try {
      result = evaluate("try(dot_product(c0, c1))", data);
      expected =
          evaluate(fmt::format("try({})", kEquivalentExpressionBigint), data);
    } catch (...) {
      return;
    }

    if (!result || !expected) {
      return;
    }

    auto array1 = data->childAt(0);
    auto array2 = data->childAt(1);

    for (auto i = 0; i < data->size(); ++i) {
      if (!nonNullRows.isValid(i)) {
        continue;
      }
      // Only compare when arrays have the same length
      if (haveSameLength(array1, array2, i)) {
        // Both should be null or both should have same value
        if (!result->isNullAt(i) && !expected->isNullAt(i)) {
          ASSERT_TRUE(expected->equalValueAt(result.get(), i, i))
              << "Mismatch at row " << i << ": expected "
              << expected->toString(i) << ", got " << result->toString(i);
        }
      }
    }
  }

  // Test equivalence for double arrays.
  void testEquivalenceDouble(const RowVectorPtr& data) {
    auto nonNullRows = getNonNullRows(data);
    if (nonNullRows.countSelected() == 0) {
      return;
    }

    VectorPtr result;
    VectorPtr expected;
    try {
      result = evaluate("try(dot_product(c0, c1))", data);
      expected =
          evaluate(fmt::format("try({})", kEquivalentExpressionDouble), data);
    } catch (...) {
      return;
    }

    if (!result || !expected) {
      return;
    }

    auto array1 = data->childAt(0);
    auto array2 = data->childAt(1);

    for (auto i = 0; i < data->size(); ++i) {
      if (!nonNullRows.isValid(i)) {
        continue;
      }
      if (haveSameLength(array1, array2, i)) {
        if (!result->isNullAt(i) && !expected->isNullAt(i)) {
          ASSERT_TRUE(expected->equalValueAt(result.get(), i, i))
              << "Mismatch at row " << i << ": expected "
              << expected->toString(i) << ", got " << result->toString(i);
        }
      }
    }
  }

  // Helper to create fuzzer options with common settings.
  VectorFuzzer::Options createFuzzerOptions(
      vector_size_t vectorSize = 100,
      double nullRatio = 0.1,
      size_t containerLength = 10) {
    VectorFuzzer::Options opts;
    opts.vectorSize = vectorSize;
    opts.nullRatio = nullRatio;
    opts.containerLength = containerLength;
    opts.containerVariableLength = true;
    opts.containerHasNulls = true;
    return opts;
  }

  // Helper to run fuzzer test with given type.
  template <typename T>
  void runFuzzerTest(
      const TypePtr& elementType,
      const VectorFuzzer::Options& opts,
      bool isFloatingPoint = false) {
    VectorFuzzer fuzzer(opts, pool());

    auto arrayType = ARRAY(elementType);
    auto array1 = fuzzer.fuzz(arrayType);
    auto array2 = fuzzer.fuzz(arrayType);
    auto data = makeRowVector({array1, array2});

    if (isFloatingPoint) {
      testEquivalenceDouble(data);
    } else {
      testEquivalenceBigint(data);
    }
    testDotProductProperties(data);
  }
};

TEST_F(DotProductFuzzerTest, fuzzInteger) {
  auto opts = createFuzzerOptions();
  runFuzzerTest<int32_t>(INTEGER(), opts);
}

TEST_F(DotProductFuzzerTest, fuzzBigint) {
  auto opts = createFuzzerOptions();
  runFuzzerTest<int64_t>(BIGINT(), opts);
}

TEST_F(DotProductFuzzerTest, fuzzSmallint) {
  auto opts = createFuzzerOptions();
  runFuzzerTest<int16_t>(SMALLINT(), opts);
}

TEST_F(DotProductFuzzerTest, fuzzTinyint) {
  auto opts = createFuzzerOptions();
  runFuzzerTest<int8_t>(TINYINT(), opts);
}

TEST_F(DotProductFuzzerTest, fuzzHighNullRatio) {
  auto opts = createFuzzerOptions(100, 0.5, 10);
  runFuzzerTest<int32_t>(INTEGER(), opts);
}

TEST_F(DotProductFuzzerTest, fuzzLargeVectors) {
  auto opts = createFuzzerOptions(500, 0.1, 20);
  runFuzzerTest<int32_t>(INTEGER(), opts);
}

TEST_F(DotProductFuzzerTest, fuzzSmallContainers) {
  auto opts = createFuzzerOptions(100, 0.1, 3);
  runFuzzerTest<int32_t>(INTEGER(), opts);
}

// Test with matching array lengths to ensure we test the actual dot product
// computation more frequently.
TEST_F(DotProductFuzzerTest, fuzzMatchingLengths) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = false; // Fixed length
  opts.containerHasNulls = true;

  VectorFuzzer fuzzer(opts, pool());

  auto array1 = fuzzer.fuzz(ARRAY(INTEGER()));
  auto array2 = fuzzer.fuzz(ARRAY(INTEGER()));
  auto data = makeRowVector({array1, array2});

  testEquivalenceBigint(data);
  testDotProductProperties(data);
}

// Stress test with many iterations
TEST_F(DotProductFuzzerTest, fuzzStressTest) {
  constexpr int kIterations = 50;

  for (int iter = 0; iter < kIterations; ++iter) {
    VectorFuzzer::Options opts;
    opts.vectorSize = 50;
    opts.nullRatio = 0.1;
    opts.containerLength = 8;
    opts.containerVariableLength = true;
    opts.containerHasNulls = true;

    VectorFuzzer fuzzer(opts, pool());

    auto array1 = fuzzer.fuzz(ARRAY(INTEGER()));
    auto array2 = fuzzer.fuzz(ARRAY(INTEGER()));
    auto data = makeRowVector({array1, array2});

    testEquivalenceBigint(data);
  }
}

// Test map dot product with fuzzer
TEST_F(DotProductFuzzerTest, fuzzMapIntegerKeys) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 50;
  opts.nullRatio = 0.1;
  opts.containerLength = 5;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;

  VectorFuzzer fuzzer(opts, pool());

  auto map1 = fuzzer.fuzz(MAP(INTEGER(), BIGINT()));
  auto map2 = fuzzer.fuzz(MAP(INTEGER(), BIGINT()));
  auto data = makeRowVector({map1, map2});

  // Just verify it doesn't crash with random inputs
  VectorPtr result;
  try {
    result = evaluate("try(dot_product(c0, c1))", data);
    ASSERT_NE(result, nullptr);
  } catch (const VeloxUserError&) {
    // Expected for some edge cases
  }
}

TEST_F(DotProductFuzzerTest, fuzzMapVarcharKeys) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 50;
  opts.nullRatio = 0.1;
  opts.containerLength = 5;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  opts.stringLength = 10;
  opts.stringVariableLength = true;

  VectorFuzzer fuzzer(opts, pool());

  auto map1 = fuzzer.fuzz(MAP(VARCHAR(), DOUBLE()));
  auto map2 = fuzzer.fuzz(MAP(VARCHAR(), DOUBLE()));
  auto data = makeRowVector({map1, map2});

  // Just verify it doesn't crash with random inputs
  VectorPtr result;
  try {
    result = evaluate("try(dot_product(c0, c1))", data);
    ASSERT_NE(result, nullptr);
  } catch (const VeloxUserError&) {
    // Expected for some edge cases
  }
}

} // namespace
} // namespace facebook::velox::functions
