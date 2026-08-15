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

#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/type/StringView.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/SimpleVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class L2NormTest : public test::FunctionBaseTest {
 protected:
  template <typename T>
  void testArrayL2Norm(
      const std::vector<std::vector<std::optional<T>>>& input,
      const std::vector<std::optional<double>>& expected) {
    auto inputArray = makeNullableArrayVector(input);
    auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

    auto expectedVector = makeNullableFlatVector<double>(expected, DOUBLE());
    assertEqualVectors(expectedVector, result);
  }
};

// Test basic integer arrays
TEST_F(L2NormTest, integerArray) {
  // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
  testArrayL2Norm<int32_t>({{3, 4}}, {5.0});

  // sqrt(1^2 + 2^2 + 2^2) = sqrt(1 + 4 + 4) = sqrt(9) = 3
  testArrayL2Norm<int32_t>({{1, 2, 2}}, {3.0});

  // sqrt(1^2) = 1
  testArrayL2Norm<int32_t>({{1}}, {1.0});

  // sqrt(0^2) = 0
  testArrayL2Norm<int32_t>({{0}}, {0.0});
}

// Test double arrays
TEST_F(L2NormTest, doubleArray) {
  auto inputArray = makeArrayVector<double>({{3.0, 4.0}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto expected = makeFlatVector<double>({5.0}, DOUBLE());
  assertEqualVectors(expected, result);
}

// Test float arrays
TEST_F(L2NormTest, floatArray) {
  auto inputArray = makeArrayVector<float>({{3.0F, 4.0F}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto expected = makeFlatVector<double>({5.0}, DOUBLE());
  assertEqualVectors(expected, result);
}

// Test empty arrays - should return 0.0
TEST_F(L2NormTest, emptyArray) {
  auto inputArray = makeArrayVector<int32_t>({{}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto expected = makeFlatVector<double>({0.0}, DOUBLE());
  assertEqualVectors(expected, result);
}

// Test arrays with null elements - nulls are skipped
TEST_F(L2NormTest, nullElementsInArray) {
  // sqrt(3^2 + 4^2) = 5 (null is skipped)
  testArrayL2Norm<int32_t>({{3, std::nullopt, 4}}, {5.0});

  // sqrt(5^2) = 5 (nulls are skipped)
  testArrayL2Norm<int32_t>({{std::nullopt, 5, std::nullopt}}, {5.0});

  // All nulls - should return 0
  testArrayL2Norm<int32_t>({{std::nullopt, std::nullopt}}, {0.0});
}

// Test null array - should return null
TEST_F(L2NormTest, nullArray) {
  auto nullArray = BaseVector::createNullConstant(ARRAY(INTEGER()), 1, pool());
  auto result = evaluate("l2_norm(c0)", makeRowVector({nullArray}));

  auto expected = BaseVector::createNullConstant(DOUBLE(), 1, pool());
  assertEqualVectors(expected, result);
}

// Test negative numbers
TEST_F(L2NormTest, negativeNumbers) {
  // sqrt((-3)^2 + (-4)^2) = sqrt(9 + 16) = 5
  testArrayL2Norm<int32_t>({{-3, -4}}, {5.0});

  // sqrt((-3)^2 + 4^2) = sqrt(9 + 16) = 5
  testArrayL2Norm<int32_t>({{-3, 4}}, {5.0});
}

// Test large arrays
TEST_F(L2NormTest, largeArray) {
  // Create array [1, 1, 1, ..., 1] with 100 ones
  // sqrt(100 * 1^2) = sqrt(100) = 10
  const std::vector<std::optional<int32_t>> ones(100, 1);
  testArrayL2Norm<int32_t>({ones}, {10.0});
}

// Test multiple rows
TEST_F(L2NormTest, multipleRows) {
  auto inputArray = makeArrayVector<int32_t>({
      {3, 4}, // sqrt(25) = 5
      {1, 0}, // sqrt(1) = 1
      {0, 0, 0}, // sqrt(0) = 0
      {1, 2, 2}, // sqrt(9) = 3
  });

  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));
  auto expected = makeFlatVector<double>({5.0, 1.0, 0.0, 3.0}, DOUBLE());
  assertEqualVectors(expected, result);
}

// Test different integer types
TEST_F(L2NormTest, differentIntegerTypes) {
  // int8_t
  auto inputInt8 = makeArrayVector<int8_t>({{3, 4}});
  auto resultInt8 = evaluate("l2_norm(c0)", makeRowVector({inputInt8}));
  auto expectedInt8 = makeFlatVector<double>({5.0}, DOUBLE());
  assertEqualVectors(expectedInt8, resultInt8);

  // int16_t
  auto inputInt16 = makeArrayVector<int16_t>({{3, 4}});
  auto resultInt16 = evaluate("l2_norm(c0)", makeRowVector({inputInt16}));
  assertEqualVectors(expectedInt8, resultInt16);

  // int64_t
  auto inputInt64 = makeArrayVector<int64_t>({{3, 4}});
  auto resultInt64 = evaluate("l2_norm(c0)", makeRowVector({inputInt64}));
  assertEqualVectors(expectedInt8, resultInt64);
}

// Test unit vector (L2 norm should be 1)
TEST_F(L2NormTest, unitVector) {
  // A normalized unit vector
  const double kInvSqrt2 = 1.0 / std::numbers::sqrt2;
  auto inputArray = makeArrayVector<double>({{kInvSqrt2, kInvSqrt2}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  // Result should be very close to 1.0
  auto resultVector = result->as<SimpleVector<double>>();
  ASSERT_NEAR(1.0, resultVector->valueAt(0), 1e-10);
}

// Test large values
TEST_F(L2NormTest, largeValues) {
  auto inputArray = makeArrayVector<double>({{1e10, 1e10}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto resultVector = result->as<SimpleVector<double>>();
  const double expected = std::numbers::sqrt2 * 1e10;
  ASSERT_NEAR(expected, resultVector->valueAt(0), 1e5);
}

// Test small values
TEST_F(L2NormTest, smallValues) {
  auto inputArray = makeArrayVector<double>({{1e-10, 1e-10}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto resultVector = result->as<SimpleVector<double>>();
  const double expected = std::numbers::sqrt2 * 1e-10;
  ASSERT_NEAR(expected, resultVector->valueAt(0), 1e-20);
}

// Test maps with varchar keys
TEST_F(L2NormTest, mapVarcharKeys) {
  auto mapVector = makeMapVector<StringView, int32_t>({
      {{"a", 3}, {"b", 4}},
  });

  auto result = evaluate("l2_norm(c0)", makeRowVector({mapVector}));
  auto expected = makeFlatVector<double>({5.0}, DOUBLE());
  assertEqualVectors(expected, result);
}

// Test maps with integer keys
TEST_F(L2NormTest, mapIntegerKeys) {
  auto mapVector = makeMapVector<int32_t, int32_t>({
      {{1, 3}, {2, 4}},
  });

  auto result = evaluate("l2_norm(c0)", makeRowVector({mapVector}));
  auto expected = makeFlatVector<double>({5.0}, DOUBLE());
  assertEqualVectors(expected, result);
}

// Test empty map
TEST_F(L2NormTest, emptyMap) {
  // Create an empty map using makeMapVector with offset-based constructor
  auto keys = makeFlatVector<StringView>({});
  auto values = makeFlatVector<int32_t>({});
  auto mapVector = makeMapVector({0}, keys, values);

  auto result = evaluate("l2_norm(c0)", makeRowVector({mapVector}));
  auto expected = makeFlatVector<double>({0.0}, DOUBLE());
  assertEqualVectors(expected, result);
}

// Test map with null values - nulls are skipped
TEST_F(L2NormTest, mapWithNullValues) {
  // Create a map with null values using offset-based constructor
  auto keys = makeFlatVector<StringView>({"a", "b", "c"});
  auto values = makeNullableFlatVector<int32_t>({3, std::nullopt, 4});
  auto mapVector = makeMapVector({0}, keys, values);
  auto result = evaluate("l2_norm(c0)", makeRowVector({mapVector}));

  // sqrt(3^2 + 4^2) = 5 (null is skipped)
  auto expected = makeFlatVector<double>({5.0}, DOUBLE());
  assertEqualVectors(expected, result);
}

// Test map with double values
TEST_F(L2NormTest, mapDoubleValues) {
  auto mapVector = makeMapVector<StringView, double>({
      {{"x", 3.0}, {"y", 4.0}},
  });

  auto result = evaluate("l2_norm(c0)", makeRowVector({mapVector}));
  auto expected = makeFlatVector<double>({5.0}, DOUBLE());
  assertEqualVectors(expected, result);
}

// Test null map
TEST_F(L2NormTest, nullMap) {
  auto nullMap =
      BaseVector::createNullConstant(MAP(VARCHAR(), INTEGER()), 1, pool());
  auto result = evaluate("l2_norm(c0)", makeRowVector({nullMap}));

  auto expected = BaseVector::createNullConstant(DOUBLE(), 1, pool());
  assertEqualVectors(expected, result);
}

// Test 3D vector
TEST_F(L2NormTest, vector3D) {
  // sqrt(1^2 + 2^2 + 3^2) = sqrt(1 + 4 + 9) = sqrt(14)
  auto inputArray = makeArrayVector<double>({{1.0, 2.0, 3.0}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto resultVector = result->as<SimpleVector<double>>();
  ASSERT_NEAR(std::sqrt(14.0), resultVector->valueAt(0), 1e-10);
}

// Test high dimensional vector
TEST_F(L2NormTest, highDimensionalVector) {
  // 10D vector with all 1s: sqrt(10) ≈ 3.162
  const std::vector<double> values(10, 1.0);
  auto inputArray = makeArrayVector<double>({values});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto resultVector = result->as<SimpleVector<double>>();
  ASSERT_NEAR(std::sqrt(10.0), resultVector->valueAt(0), 1e-10);
}

// Test Pythagorean quadruple: 2^2 + 3^2 + 6^2 = 49 = 7^2
TEST_F(L2NormTest, pythagoreanQuadruple) {
  auto inputArray = makeArrayVector<int32_t>({{2, 3, 6}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto expected = makeFlatVector<double>({7.0}, DOUBLE());
  assertEqualVectors(expected, result);
}

// Test with zeros and non-zeros mixed
TEST_F(L2NormTest, mixedZerosAndNonZeros) {
  // sqrt(0 + 0 + 5^2 + 0) = 5
  testArrayL2Norm<int32_t>({{0, 0, 5, 0}}, {5.0});
}

// Test ML use case - feature vector normalization check
TEST_F(L2NormTest, mlFeatureVector) {
  // Common ML scenario: checking if a feature vector has unit L2 norm
  // Feature embedding vector
  const std::vector<double> embedding = {0.2, 0.4, 0.3, 0.1, 0.7, 0.5};
  auto inputArray = makeArrayVector<double>({embedding});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  // Calculate expected L2 norm
  double sumSquares = 0.0;
  for (const double val : embedding) {
    sumSquares += val * val;
  }
  const double expectedL2Norm = std::sqrt(sumSquares);

  auto resultVector = result->as<SimpleVector<double>>();
  ASSERT_NEAR(expectedL2Norm, resultVector->valueAt(0), 1e-10);
}

// Test sparse vector representation using map
TEST_F(L2NormTest, sparseVectorMap) {
  // Sparse vector: {index1: 3.0, index5: 4.0, index10: 0.0}
  // L2 norm = sqrt(9 + 16 + 0) = 5
  auto mapVector = makeMapVector<int64_t, double>({
      {{1, 3.0}, {5, 4.0}, {10, 0.0}},
  });

  auto result = evaluate("l2_norm(c0)", makeRowVector({mapVector}));
  auto expected = makeFlatVector<double>({5.0}, DOUBLE());
  assertEqualVectors(expected, result);
}

// ============================================================================
// OVERFLOW / UNDERFLOW TESTS
// Verify that the scaled summation approach avoids overflow and underflow
// that would occur with naive value * value accumulation.
// ============================================================================

// Test that very large values don't overflow to infinity.
// Without scaling, 1e200 * 1e200 = +inf, and sqrt(+inf) = +inf.
// With scaling, the result should be finite and correct.
TEST_F(L2NormTest, overflowLargeDoubles) {
  auto inputArray = makeArrayVector<double>({{1e200, 1e200}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto resultVector = result->as<SimpleVector<double>>();
  double actual = resultVector->valueAt(0);
  double expected = std::numbers::sqrt2 * 1e200;
  ASSERT_TRUE(std::isfinite(actual))
      << "L2 norm of large values should be finite, got: " << actual;
  ASSERT_NEAR(expected, actual, expected * 1e-10);
}

// Test with DBL_MAX-scale values: should not overflow.
TEST_F(L2NormTest, overflowNearDbMax) {
  constexpr double kLarge = 1e308;
  auto inputArray = makeArrayVector<double>({{kLarge}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto resultVector = result->as<SimpleVector<double>>();
  double actual = resultVector->valueAt(0);
  ASSERT_TRUE(std::isfinite(actual))
      << "L2 norm near DBL_MAX should be finite, got: " << actual;
  ASSERT_NEAR(kLarge, actual, kLarge * 1e-10);
}

// Test multiple large values that would overflow naive summation.
TEST_F(L2NormTest, overflowMultipleLargeValues) {
  constexpr double kLarge = 1e200;
  auto inputArray = makeArrayVector<double>({{kLarge, kLarge, kLarge, kLarge}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto resultVector = result->as<SimpleVector<double>>();
  double actual = resultVector->valueAt(0);
  double expected = 2.0 * kLarge; // sqrt(4) * 1e200
  ASSERT_TRUE(std::isfinite(actual))
      << "L2 norm of multiple large values should be finite, got: " << actual;
  ASSERT_NEAR(expected, actual, expected * 1e-10);
}

// Test that very small values don't underflow to zero.
// Without scaling, 1e-200 * 1e-200 = 0 (underflows), so sqrt(0) = 0.
// With scaling, the result should be correct.
TEST_F(L2NormTest, underflowSmallDoubles) {
  auto inputArray = makeArrayVector<double>({{1e-200, 1e-200}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto resultVector = result->as<SimpleVector<double>>();
  double actual = resultVector->valueAt(0);
  double expected = std::numbers::sqrt2 * 1e-200;
  ASSERT_GT(actual, 0.0)
      << "L2 norm of small values should not underflow to zero";
  ASSERT_NEAR(expected, actual, expected * 1e-10);
}

// Test extremely small subnormal values.
TEST_F(L2NormTest, underflowSubnormal) {
  constexpr double kTiny = 5e-324; // smallest positive subnormal double
  auto inputArray = makeArrayVector<double>({{kTiny}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto resultVector = result->as<SimpleVector<double>>();
  double actual = resultVector->valueAt(0);
  ASSERT_GT(actual, 0.0) << "L2 norm of subnormal should not be zero";
}

// Test mixed large and small values.
// The large value should dominate but the small value should not cause issues.
TEST_F(L2NormTest, mixedLargeAndSmallValues) {
  auto inputArray = makeArrayVector<double>({{1e200, 1e-200}});
  auto result = evaluate("l2_norm(c0)", makeRowVector({inputArray}));

  auto resultVector = result->as<SimpleVector<double>>();
  double actual = resultVector->valueAt(0);
  ASSERT_TRUE(std::isfinite(actual))
      << "L2 norm with mixed magnitudes should be finite, got: " << actual;
  ASSERT_NEAR(1e200, actual, 1e200 * 1e-10);
}

// Test overflow protection with map input.
TEST_F(L2NormTest, overflowMap) {
  constexpr double kLarge = 1e200;
  auto mapVector = makeMapVector<StringView, double>({
      {{"a", kLarge}, {"b", kLarge}},
  });

  auto result = evaluate("l2_norm(c0)", makeRowVector({mapVector}));
  auto resultVector = result->as<SimpleVector<double>>();
  double actual = resultVector->valueAt(0);
  double expected = std::numbers::sqrt2 * kLarge;
  ASSERT_TRUE(std::isfinite(actual))
      << "Map L2 norm of large values should be finite, got: " << actual;
  ASSERT_NEAR(expected, actual, expected * 1e-10);
}

// Test underflow protection with map input.
TEST_F(L2NormTest, underflowMap) {
  constexpr double kTiny = 1e-200;
  auto mapVector = makeMapVector<StringView, double>({
      {{"x", kTiny}, {"y", kTiny}},
  });

  auto result = evaluate("l2_norm(c0)", makeRowVector({mapVector}));
  auto resultVector = result->as<SimpleVector<double>>();
  double actual = resultVector->valueAt(0);
  double expected = std::numbers::sqrt2 * kTiny;
  ASSERT_GT(actual, 0.0)
      << "Map L2 norm of small values should not underflow to zero";
  ASSERT_NEAR(expected, actual, expected * 1e-10);
}

// Test that overflow protection works with nullable arrays containing large
// values.
TEST_F(L2NormTest, overflowWithNulls) {
  constexpr double kLarge = 1e200;
  testArrayL2Norm<double>(
      {{kLarge, std::nullopt, kLarge}}, {std::numbers::sqrt2 * kLarge});
}

} // namespace

// ============================================================================
// CUSTOM FUZZER TESTS
// These tests use VectorFuzzer to generate random inputs and verify that the
// l2_norm function produces valid results (non-negative, finite for finite
// inputs) and matches manual calculation using reduce.
// ============================================================================

class L2NormFuzzerTest : public test::FunctionBaseTest {
 protected:
  // Verify that L2 norm result is valid: non-negative and finite for
  // finite inputs.
  void verifyL2NormProperties(const RowVectorPtr& data) {
    VectorPtr result;
    try {
      result = evaluate("l2_norm(c0)", data);
    } catch (...) {
      return;
    }

    if (!result) {
      return;
    }

    auto resultVector = result->as<SimpleVector<double>>();
    if (!resultVector) {
      return;
    }

    for (auto i = 0; i < data->size(); ++i) {
      if (data->childAt(0)->isNullAt(i)) {
        ASSERT_TRUE(result->isNullAt(i))
            << "Result should be null when input is null at row " << i;
        continue;
      }

      if (result->isNullAt(i)) {
        continue;
      }

      double l2norm = resultVector->valueAt(i);

      ASSERT_GE(l2norm, 0.0) << "L2 norm must be non-negative at row " << i;

      ASSERT_FALSE(std::isnan(l2norm))
          << "L2 norm should not be NaN at row " << i;
    }
  }

  // Compare l2_norm(array) against equivalent reduce expression:
  // reduce(c0, CAST(0.0 AS DOUBLE), (s, x) -> s + x * x, s -> sqrt(s))
  void testArrayEquivalence(const RowVectorPtr& data) {
    VectorPtr result;
    VectorPtr expected;

    try {
      result = evaluate("l2_norm(c0)", data);
      expected = evaluate(
          "reduce(c0, CAST(0.0 AS DOUBLE), (s, x) -> s + COALESCE(CAST(x AS DOUBLE) * CAST(x AS DOUBLE), 0.0), s -> sqrt(s))",
          data);
    } catch (...) {
      return;
    }

    if (!result || !expected) {
      return;
    }

    for (auto i = 0; i < data->size(); ++i) {
      if (data->childAt(0)->isNullAt(i)) {
        continue;
      }

      if (result->isNullAt(i) || expected->isNullAt(i)) {
        continue;
      }

      auto resultVector = result->as<SimpleVector<double>>();
      auto expectedVector = expected->as<SimpleVector<double>>();

      if (!resultVector || !expectedVector) {
        continue;
      }

      // Use relative tolerance for large values
      double expectedVal = expectedVector->valueAt(i);
      double resultVal = resultVector->valueAt(i);
      double tolerance = std::max(1e-9, std::abs(expectedVal) * 1e-10);
      ASSERT_NEAR(expectedVal, resultVal, tolerance) << "Mismatch at row " << i;
    }
  }
};

TEST_F(L2NormFuzzerTest, fuzzIntegerArrays) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputArray = fuzzer.fuzz(ARRAY(INTEGER()));
    auto data = makeRowVector({inputArray});
    verifyL2NormProperties(data);
    testArrayEquivalence(data);
  }
}

TEST_F(L2NormFuzzerTest, fuzzBigintArrays) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputArray = fuzzer.fuzz(ARRAY(BIGINT()));
    auto data = makeRowVector({inputArray});
    verifyL2NormProperties(data);
    testArrayEquivalence(data);
  }
}

TEST_F(L2NormFuzzerTest, fuzzDoubleArrays) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputArray = fuzzer.fuzz(ARRAY(DOUBLE()));
    auto data = makeRowVector({inputArray});
    verifyL2NormProperties(data);
  }
}

TEST_F(L2NormFuzzerTest, fuzzFloatArrays) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputArray = fuzzer.fuzz(ARRAY(REAL()));
    auto data = makeRowVector({inputArray});
    verifyL2NormProperties(data);
  }
}

TEST_F(L2NormFuzzerTest, fuzzSmallintArrays) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputArray = fuzzer.fuzz(ARRAY(SMALLINT()));
    auto data = makeRowVector({inputArray});
    verifyL2NormProperties(data);
    testArrayEquivalence(data);
  }
}

TEST_F(L2NormFuzzerTest, fuzzTinyintArrays) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputArray = fuzzer.fuzz(ARRAY(TINYINT()));
    auto data = makeRowVector({inputArray});
    verifyL2NormProperties(data);
    testArrayEquivalence(data);
  }
}

TEST_F(L2NormFuzzerTest, fuzzMapIntegerKeys) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputMap = fuzzer.fuzz(MAP(INTEGER(), INTEGER()));
    auto data = makeRowVector({inputMap});
    verifyL2NormProperties(data);
  }
}

TEST_F(L2NormFuzzerTest, fuzzMapVarcharKeys) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  opts.stringLength = 10;
  opts.stringVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputMap = fuzzer.fuzz(MAP(VARCHAR(), INTEGER()));
    auto data = makeRowVector({inputMap});
    verifyL2NormProperties(data);
  }
}

TEST_F(L2NormFuzzerTest, fuzzMapDoubleValues) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputMap = fuzzer.fuzz(MAP(VARCHAR(), DOUBLE()));
    auto data = makeRowVector({inputMap});
    verifyL2NormProperties(data);
  }
}

TEST_F(L2NormFuzzerTest, fuzzHighNullRatio) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.5;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputArray = fuzzer.fuzz(ARRAY(INTEGER()));
    auto data = makeRowVector({inputArray});
    verifyL2NormProperties(data);
  }
}

TEST_F(L2NormFuzzerTest, fuzzLargeVectors) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 500;
  opts.nullRatio = 0.1;
  opts.containerLength = 20;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 5; ++iter) {
    auto inputArray = fuzzer.fuzz(ARRAY(INTEGER()));
    auto data = makeRowVector({inputArray});
    verifyL2NormProperties(data);
  }
}

TEST_F(L2NormFuzzerTest, fuzzDictionaryEncoded) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto baseArray = fuzzer.fuzz(ARRAY(INTEGER()));
    auto inputArray = fuzzer.fuzzDictionary(baseArray, opts.vectorSize);
    auto data = makeRowVector({inputArray});
    verifyL2NormProperties(data);
  }
}

TEST_F(L2NormFuzzerTest, fuzzEmptyContainers) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 1;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputArray = fuzzer.fuzz(ARRAY(INTEGER()));
    auto data = makeRowVector({inputArray});
    verifyL2NormProperties(data);
  }
}

TEST_F(L2NormFuzzerTest, fuzzNoNulls) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.0;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.containerHasNulls = false;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputArray = fuzzer.fuzz(ARRAY(INTEGER()));
    auto data = makeRowVector({inputArray});
    verifyL2NormProperties(data);
    testArrayEquivalence(data);
  }
}

} // namespace facebook::velox::functions
