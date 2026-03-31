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
#include <limits>
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
  using expected_t = std::conditional_t<std::is_integral_v<T>, int64_t, double>;

  template <typename T>
  void testDotProduct(
      const std::vector<std::vector<T>>& array1,
      const std::vector<std::vector<T>>& array2,
      const std::vector<std::optional<expected_t<T>>>& expected) {
    auto inputArray1 = makeArrayVector<T>(array1);
    auto inputArray2 = makeArrayVector<T>(array2);
    auto result = evaluate(
        "dot_product(c0, c1)", makeRowVector({inputArray1, inputArray2}));
    auto expectedVector = makeNullableFlatVector<expected_t<T>>(expected);
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

TEST_F(DotProductTest, doubleArrays) {
  testDotProduct<double>(
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

TEST_F(DotProductTest, integerOverflow) {
  constexpr auto kMax = std::numeric_limits<int64_t>::max();
  auto array1 = makeArrayVector<int64_t>({{kMax, 1}});
  auto array2 = makeArrayVector<int64_t>({{2, 1}});

  VELOX_ASSERT_THROW(
      evaluate("dot_product(c0, c1)", makeRowVector({array1, array2})), "");
}

TEST_F(DotProductTest, integerOverflowOnSum) {
  constexpr auto kMax = std::numeric_limits<int64_t>::max();
  auto array1 = makeArrayVector<int64_t>({{kMax, 1}});
  auto array2 = makeArrayVector<int64_t>({{1, 1}});

  VELOX_ASSERT_THROW(
      evaluate("dot_product(c0, c1)", makeRowVector({array1, array2})), "");
}

TEST_F(DotProductTest, int32Overflow) {
  constexpr auto kMax = std::numeric_limits<int32_t>::max();
  // int32 max * int32 max fits in int64, so this should succeed.
  testDotProduct<int32_t>(
      {{kMax}}, {{kMax}}, {static_cast<int64_t>(kMax) * kMax});
}

TEST_F(DotProductTest, doubleNaN) {
  constexpr auto kNaN = std::numeric_limits<double>::quiet_NaN();

  auto array1 = makeArrayVector<double>({{kNaN, 1.0}});
  auto array2 = makeArrayVector<double>({{1.0, 2.0}});

  auto result =
      evaluate("dot_product(c0, c1)", makeRowVector({array1, array2}));
  auto flatResult = result->as<SimpleVector<double>>();
  ASSERT_TRUE(std::isnan(flatResult->valueAt(0)));
}

TEST_F(DotProductTest, doubleNaNBothArrays) {
  constexpr auto kNaN = std::numeric_limits<double>::quiet_NaN();

  auto array1 = makeArrayVector<double>({{kNaN}});
  auto array2 = makeArrayVector<double>({{kNaN}});

  auto result =
      evaluate("dot_product(c0, c1)", makeRowVector({array1, array2}));
  auto flatResult = result->as<SimpleVector<double>>();
  ASSERT_TRUE(std::isnan(flatResult->valueAt(0)));
}

TEST_F(DotProductTest, doubleInfTimesZero) {
  constexpr auto kInf = std::numeric_limits<double>::infinity();

  auto array1 = makeArrayVector<double>({{kInf}});
  auto array2 = makeArrayVector<double>({{0.0}});

  // inf * 0 = NaN per IEEE 754.
  auto result =
      evaluate("dot_product(c0, c1)", makeRowVector({array1, array2}));
  auto flatResult = result->as<SimpleVector<double>>();
  ASSERT_TRUE(std::isnan(flatResult->valueAt(0)));
}

TEST_F(DotProductTest, doubleNegativeInfTimesNegativeZero) {
  constexpr auto kNegInf = -std::numeric_limits<double>::infinity();

  auto array1 = makeArrayVector<double>({{kNegInf}});
  auto array2 = makeArrayVector<double>({{-0.0}});

  // -inf * -0 = NaN per IEEE 754.
  auto result =
      evaluate("dot_product(c0, c1)", makeRowVector({array1, array2}));
  auto flatResult = result->as<SimpleVector<double>>();
  ASSERT_TRUE(std::isnan(flatResult->valueAt(0)));
}

TEST_F(DotProductTest, doubleInfTimesInf) {
  constexpr auto kInf = std::numeric_limits<double>::infinity();

  // inf * inf = inf.
  testDotProduct<double>({{kInf}}, {{kInf}}, {kInf});

  // inf * -inf = -inf.
  testDotProduct<double>({{kInf}}, {{-kInf}}, {-kInf});
}

TEST_F(DotProductTest, doubleOverflow) {
  constexpr auto kMax = std::numeric_limits<double>::max();
  constexpr auto kInf = std::numeric_limits<double>::infinity();
  // max + max = inf (overflow to infinity for doubles).
  testDotProduct<double>({{kMax, kMax}}, {{1.0, 1.0}}, {kInf});
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
