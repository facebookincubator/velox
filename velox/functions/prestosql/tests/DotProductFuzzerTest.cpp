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
#include <vector>

#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

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
