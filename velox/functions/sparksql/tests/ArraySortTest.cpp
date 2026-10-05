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
#include <optional>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/sparksql/tests/ArraySortTestData.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

using namespace facebook::velox::test;

using facebook::velox::functions::test::FunctionBaseTest;

class ArraySortTest : public SparkFunctionBaseTest {
 protected:
  ArraySortTest() {
    options_.parseIntegerAsBigint = false;
  }

  void testArraySort(const VectorPtr& input, const VectorPtr& expected) {
    auto result = evaluate("array_sort(c0)", makeRowVector({input}));
    assertEqualVectors(expected, result);
  }

  void testArraySort(
      const std::string& lamdaExpr,
      bool asc,
      const VectorPtr& input,
      const VectorPtr& expected) {
    std::string name = asc ? "array_sort" : "array_sort_desc";
    auto result = evaluate(
        fmt::format("{}(c0, {})", name, lamdaExpr), makeRowVector({input}));
    assertEqualVectors(expected, result);

    SelectivityVector firstRow(1);
    result = evaluate(
        fmt::format("{}(c0, {})", name, lamdaExpr),
        makeRowVector({input}),
        firstRow);
    assertEqualVectors(expected->slice(0, 1), result);
  }

  template <typename T>
  void testInt() {
    auto input = makeNullableArrayVector(intInput<T>());
    auto expected = makeNullableArrayVector(intAscNullLargest<T>());
    testArraySort(input, expected);
  }

  template <typename T>
  void testFloatingPoint() {
    auto input = makeNullableArrayVector(floatingPointInput<T>());
    auto expected = makeNullableArrayVector(floatingPointAscNullLargest<T>());
    testArraySort(input, expected);
  }
};

TEST_F(ArraySortTest, int8) {
  testInt<int8_t>();
}

TEST_F(ArraySortTest, int16) {
  testInt<int16_t>();
}

TEST_F(ArraySortTest, int32) {
  testInt<int32_t>();
}

TEST_F(ArraySortTest, int64) {
  testInt<int64_t>();
}

TEST_F(ArraySortTest, float) {
  testFloatingPoint<float>();
}

TEST_F(ArraySortTest, double) {
  testFloatingPoint<double>();
}

TEST_F(ArraySortTest, string) {
  auto input = makeNullableArrayVector(stringInput());
  auto expected = makeNullableArrayVector(stringAscNullLargest());
  testArraySort(input, expected);
}

TEST_F(ArraySortTest, binary) {
  auto input = makeNullableArrayVector<std::string>(
      {{std::string("\xff", 1),
        std::string("\x00", 1),
        std::string("\x80", 1),
        std::nullopt}},
      ARRAY(VARBINARY()));
  auto expected = makeNullableArrayVector<std::string>(
      {{std::string("\x00", 1),
        std::string("\x80", 1),
        std::string("\xff", 1),
        std::nullopt}},
      ARRAY(VARBINARY()));
  testArraySort(input, expected);
}

TEST_F(ArraySortTest, unknown) {
  auto input = makeNullableArrayVector<UnknownValue>({
      {std::nullopt, std::nullopt},
      {std::nullopt, std::nullopt, std::nullopt},
  });
  testArraySort(input, input);
}

TEST_F(ArraySortTest, timestamp) {
  auto input = makeNullableArrayVector(timestampInput());
  auto expected = makeNullableArrayVector(timestampAscNullLargest());
  testArraySort(input, expected);
}

TEST_F(ArraySortTest, timestampUtc) {
  auto input = makeNullableArrayVector<Timestamp>(
      {{Timestamp(20, 0), Timestamp(-10, 0), Timestamp(0, 0), std::nullopt}},
      ARRAY(TIMESTAMP_UTC()));
  auto expected = makeNullableArrayVector<Timestamp>(
      {{Timestamp(-10, 0), Timestamp(0, 0), Timestamp(20, 0), std::nullopt}},
      ARRAY(TIMESTAMP_UTC()));
  testArraySort(input, expected);
}

TEST_F(ArraySortTest, date) {
  auto input = makeNullableArrayVector(dateInput(), ARRAY(DATE()));
  auto expected = makeNullableArrayVector(dateAscNullLargest(), ARRAY(DATE()));
  testArraySort(input, expected);
}

TEST_F(ArraySortTest, yearMonthInterval) {
  auto input = makeNullableArrayVector<int32_t>(
      {{24, -6, 18, std::nullopt}}, ARRAY(INTERVAL_YEAR_MONTH()));
  auto expected = makeNullableArrayVector<int32_t>(
      {{-6, 18, 24, std::nullopt}}, ARRAY(INTERVAL_YEAR_MONTH()));
  testArraySort(input, expected);
}

TEST_F(ArraySortTest, bool) {
  auto input = makeNullableArrayVector(boolInput());
  auto expected = makeNullableArrayVector(boolAscNullLargest());
  testArraySort(input, expected);
}

TEST_F(ArraySortTest, array) {
  auto input = makeNullableNestedArrayVector(arrayInput());
  auto expected =
      makeNullableNestedArrayVector(arrayAscTopNullLargestNestedNullSmallest());
  testArraySort(input, expected);
}

// Map is not orderable, so sorting is not supported.
TEST_F(ArraySortTest, failOnMapTypeSort) {
  auto input = makeArrayOfMapVector(mapInput());
  const std::string kErrorMessage =
      "Scalar function signature is not supported";

  VELOX_ASSERT_THROW(
      evaluate("array_sort(c0)", makeRowVector({input})), kErrorMessage);
}

TEST_F(ArraySortTest, row) {
  auto rowType = ROW({INTEGER(), VARCHAR()});
  auto input = makeArrayOfRowVector(rowType, rowInput());
  auto expected =
      makeArrayOfRowVector(rowType, rowAscTopNullLargestNestedNullSmallest());
  testArraySort(input, expected);
}

TEST_F(ArraySortTest, constant) {
  vector_size_t size = 1'000;
  auto data =
      makeArrayVector<int64_t>({{1, 2, 3, 0}, {4, 5, 4, 5}, {6, 6, 6, 6}});

  auto evaluateConstant = [&](vector_size_t row, const VectorPtr& vector) {
    return evaluate(
        "array_sort(c0)",
        makeRowVector({BaseVector::wrapInConstant(size, row, vector)}));
  };

  auto result = evaluateConstant(0, data);
  auto expected = makeConstantArray<int64_t>(size, {0, 1, 2, 3});
  assertEqualVectors(expected, result);

  result = evaluateConstant(1, data);
  expected = makeConstantArray<int64_t>(size, {4, 4, 5, 5});
  assertEqualVectors(expected, result);

  result = evaluateConstant(2, data);
  expected = makeConstantArray<int64_t>(size, {6, 6, 6, 6});
  assertEqualVectors(expected, result);
}

TEST_F(ArraySortTest, lambda) {
  auto data = makeNullableArrayVector<std::string>({
      {"abc123", "abc", std::nullopt, "abcd"},
      {std::nullopt, "x", "xyz123", "xyz"},
  });

  auto sortedAsc = makeNullableArrayVector<std::string>({
      {"abc", "abcd", "abc123", std::nullopt},
      {"x", "xyz", "xyz123", std::nullopt},
  });

  auto sortedDesc = makeNullableArrayVector<std::string>({
      {"abc123", "abcd", "abc", std::nullopt},
      {"xyz123", "xyz", "x", std::nullopt},
  });

  // Different ways to sort by length ascending.
  testArraySort("x -> length(x)", true, data, sortedAsc);
  testArraySort("x -> length(x) * -1", false, data, sortedAsc);
  testArraySort(
      "(x, y) -> if(lessthan(length(x), length(y)), -1, if(greaterthan(length(x), length(y)), 1, 0))",
      true,
      data,
      sortedAsc);
  testArraySort(
      "(x, y) -> if(lessthan(length(x), length(y)), -1, if(equalto(length(x), length(y)), 0, 1))",
      true,
      data,
      sortedAsc);
  testArraySort(
      "(x, y) -> if(lessthan(length(x), length(y)), -10, if(greaterthan(length(x), length(y)), 10, 0))",
      true,
      data,
      sortedAsc);
  testArraySort(
      "(x, y) -> if(lessthan(length(x), length(y)), "
      "-2147483647 - 1, "
      "if(greaterthan(length(x), length(y)), 2147483647, 0))",
      true,
      data,
      sortedAsc);

  // Different ways to sort by length descending.
  testArraySort("x -> length(x)", false, data, sortedDesc);
  testArraySort("x -> length(x) * -1", true, data, sortedDesc);
  testArraySort(
      "(x, y) -> if(lessthan(length(x), length(y)), 1, if(greaterthan(length(x), length(y)), -1, 0))",
      true,
      data,
      sortedDesc);
  testArraySort(
      "(x, y) -> if(lessthan(length(x), length(y)), 1, if(equalto(length(x), length(y)), 0, -1))",
      true,
      data,
      sortedDesc);

  testArraySort(
      "(x, y) -> if(lessthan(length(x), length(y)), 10, if(greaterthan(length(x), length(y)), -10, 0))",
      true,
      data,
      sortedDesc);

  auto tiedData =
      makeNullableArrayVector<std::string>({{"bb", "aa", "c", "dd"}});
  testArraySort(
      "(x, y) -> if(lessthan(length(x), length(y)), -10, if(greaterthan(length(x), length(y)), 10, 0))",
      true,
      tiedData,
      makeNullableArrayVector<std::string>({{"c", "bb", "aa", "dd"}}));
  testArraySort(
      "(x, y) -> if(lessthan(length(x), length(y)), 10, if(greaterthan(length(x), length(y)), -10, 0))",
      true,
      tiedData,
      makeNullableArrayVector<std::string>({{"bb", "aa", "dd", "c"}}));

  auto capturedData = makeRowVector({
      makeNullableArrayVector<int32_t>({{3, 1, 2}, {6, 4, 5}}),
      makeFlatVector<int32_t>({10, -10}),
  });
  auto capturedResult = evaluate(
      "array_sort(c0, (x, y) -> "
      "if(lessthan(greatest(x, c1), greatest(y, c1)), -1, "
      "if(greaterthan(greatest(x, c1), greatest(y, c1)), 1, 0)))",
      capturedData);
  assertEqualVectors(
      makeNullableArrayVector<int32_t>({{3, 1, 2}, {4, 5, 6}}), capturedResult);

  auto nestedNullData =
      makeNullableArrayVector<int32_t>({{1, std::nullopt, 2}});
  testArraySort(
      "(x, y) -> if(lessthan(array(x), array(y)), -1, "
      "if(greaterthan(array(x), array(y)), 1, 0))",
      true,
      nestedNullData,
      makeNullableArrayVector<int32_t>({{std::nullopt, 1, 2}}));
  testArraySort(
      "(x, y) -> if(lessthan(array(x), array(y)), 1, "
      "if(greaterthan(array(x), array(y)), -1, 0))",
      true,
      nestedNullData,
      makeNullableArrayVector<int32_t>({{2, 1, std::nullopt}}));
}

TEST_F(ArraySortTest, identityComparatorPreservesSignedZeroOrder) {
  auto input = makeArrayVector<double>({{-0.0, 0.0, 0.0, -0.0}, {0.0, -0.0}});
  auto result = evaluate(
      "array_sort(c0, (x, y) -> "
      "if(lessthan(x, y), -10, if(greaterthan(x, y), 10, 0)))",
      makeRowVector({input}));

  auto* arrays = result->as<ArrayVector>();
  auto* elements = arrays->elements()->as<SimpleVector<double>>();
  ASSERT_NE(elements, nullptr);

  std::vector<bool> expectedSigns{true, false, false, true, false, true};
  for (vector_size_t index = 0; index < expectedSigns.size(); ++index) {
    EXPECT_EQ(std::signbit(elements->valueAt(index)), expectedSigns[index]);
  }
}

TEST_F(ArraySortTest, comparatorEncodings) {
  auto input =
      makeNullableArrayVector<int32_t>({{3, 1, 2}, {1, std::nullopt, 2}});
  const std::string comparator =
      "(x, y) -> if(lessthan(array(x), array(y)), -10, "
      "if(greaterthan(array(x), array(y)), 10, 0))";

  auto dictionaryInput =
      BaseVector::wrapInDictionary(makeIndices({1, 0, 1}), input);
  auto result = evaluate(
      fmt::format("array_sort(c0, {})", comparator),
      makeRowVector({dictionaryInput}));
  assertEqualVectors(
      makeNullableArrayVector<int32_t>(
          {{std::nullopt, 1, 2}, {1, 2, 3}, {std::nullopt, 1, 2}}),
      result);

  auto constantInput = BaseVector::wrapInConstant(3, 0, input);
  result = evaluate(
      fmt::format("array_sort(c0, {})", comparator),
      makeRowVector({constantInput}));
  assertEqualVectors(
      makeNullableArrayVector<int32_t>({{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}),
      result);
}

TEST_F(ArraySortTest, unsupportedLambda) {
  auto data = makeRowVector({
      makeNullableArrayVector(intInput<int32_t>()),
  });

  VELOX_ASSERT_THROW(
      evaluate("array_sort(c0, (a, b) -> 0)", data),
      "array_sort with comparator lambda that cannot be rewritten into a transform is not supported");

  VELOX_ASSERT_THROW(
      evaluate(
          "array_sort(c0, (a, b) -> if(lessthan(a, b), -10, if(greaterthan(a, b), 10, 5)))",
          data),
      "array_sort with comparator lambda that cannot be rewritten into a transform is not supported");

  VELOX_ASSERT_THROW(
      evaluate(
          "array_sort(c0, (a, b) -> if(lessthan(a, b), -10, if(greaterthan(a, b), -20, 0)))",
          data),
      "array_sort with comparator lambda that cannot be rewritten into a transform is not supported");

  VELOX_ASSERT_THROW(
      evaluate(
          "array_sort(c0, (a, b) -> if(equalto(a, b), 5, if(lessthan(a, b), -10, 37)))",
          data),
      "array_sort with comparator lambda that cannot be rewritten into a transform is not supported");

  VELOX_ASSERT_THROW(
      evaluate(
          "array_sort(c0, (a, b) -> if(lessthan(a, b), -10, if(equalto(a, b), 5, 37)))",
          data),
      "array_sort with comparator lambda that cannot be rewritten into a transform is not supported");

  VELOX_ASSERT_THROW(
      evaluate(
          "array_sort(c0, (a, b) -> if(lessthan(a, a), -10, if(greaterthan(a, a), 37, 0)))",
          data),
      "array_sort with comparator lambda that cannot be rewritten into a transform is not supported");

  auto dataWithCapture = makeRowVector(
      {data->childAt(0),
       makeFlatVector<int32_t>(data->size(), [](auto) { return 5; })});
  VELOX_ASSERT_THROW(
      evaluate(
          "array_sort(c0, (a, b) -> if(lessthan(a, c1), -10, if(greaterthan(a, c1), 37, 0)))",
          dataWithCapture),
      "array_sort with comparator lambda that cannot be rewritten into a transform is not supported");
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
