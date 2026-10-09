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
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/ArrayConstructor.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/sparksql/tests/ArraySortTestData.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

using namespace facebook::velox::test;

using facebook::velox::functions::test::FunctionBaseTest;

template <typename T>
struct AlwaysThrowArraySortFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(int64_t& /*result*/, int64_t /*value*/) {
    VELOX_USER_FAIL("array_sort comparator transform was evaluated");
  }
};

class ArraySortTest : public SparkFunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    SparkFunctionBaseTest::SetUpTestCase();
    facebook::velox::functions::registerArrayConstructor("array_constructor");
  }

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
    assertEqualVectors(expected, result, firstRow);
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

  template <typename T>
  void testNaturalSortPreservesSignedZeroOrder() {
    auto input = makeNullableArrayVector<T>({
        {-0.0, std::nullopt, 0.0, std::nullopt, -0.0},
        {0.0, -0.0},
    });
    auto result = evaluate("array_sort(c0)", makeRowVector({input}));

    auto* arrays = result->template as<ArrayVector>();
    DecodedVector decodedElements(*arrays->elements());
    const std::vector<std::optional<bool>> expectedSigns{
        true,
        false,
        true,
        std::nullopt,
        std::nullopt,
        false,
        true,
    };
    for (vector_size_t index = 0; index < expectedSigns.size(); ++index) {
      if (expectedSigns[index].has_value()) {
        ASSERT_FALSE(decodedElements.isNullAt(index));
        EXPECT_EQ(
            std::signbit(decodedElements.valueAt<T>(index)),
            expectedSigns[index].value());
      } else {
        EXPECT_TRUE(decodedElements.isNullAt(index));
      }
    }
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

TEST_F(ArraySortTest, naturalSortPreservesSignedZeroOrder) {
  testNaturalSortPreservesSignedZeroOrder<float>();
  testNaturalSortPreservesSignedZeroOrder<double>();
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
      {"abc123", "abc", "abcd"},
      {"x", "xyz123", "xyz"},
  });

  auto sortedAsc = makeNullableArrayVector<std::string>({
      {"abc", "abcd", "abc123"},
      {"x", "xyz", "xyz123"},
  });

  auto sortedDesc = makeNullableArrayVector<std::string>({
      {"abc123", "abcd", "abc"},
      {"xyz123", "xyz", "x"},
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
      "(x, y) -> if(equalto(length(x), length(y)), 0, "
      "if(lessthan(length(x), length(y)), -10, 37))",
      true,
      data,
      sortedAsc);
  testArraySort(
      "(x, y) -> if(lessthan(length(x), length(y)), -10, "
      "if(equalto(length(y), length(x)), 0, 37))",
      true,
      data,
      sortedAsc);
  testArraySort(
      "(x, y) -> if(lessthan(length(x), length(y)), "
      "subtract(-2147483647, 1), "
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
  testArraySort(
      "x -> ARRAY[length(x)]",
      false,
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
      "x -> ARRAY[x]",
      true,
      nestedNullData,
      makeNullableArrayVector<int32_t>({{std::nullopt, 1, 2}}));
  testArraySort(
      "x -> ARRAY[x]",
      false,
      nestedNullData,
      makeNullableArrayVector<int32_t>({{2, 1, std::nullopt}}));

  registerFunction<AlwaysThrowArraySortFunction, int64_t, int64_t>(
      {"always_throw_array_sort"});
  auto trivialArrays = makeArrayVector<int64_t>({{}, {0}});
  testArraySort(
      "(x, y) -> "
      "if(lessthan(always_throw_array_sort(x), always_throw_array_sort(y)), "
      "-1, "
      "if(greaterthan(always_throw_array_sort(x), "
      "always_throw_array_sort(y)), 1, 0))",
      true,
      trivialArrays,
      trivialArrays);
}

TEST_F(ArraySortTest, comparatorRejectsNullSortKeys) {
  auto input = makeNullableArrayVector<std::string>(
      {{"abcd123", "abcd", std::nullopt, "abc"}});

  VELOX_ASSERT_THROW(
      evaluate(
          "array_sort(c0, (x, y) -> "
          "if(lessthan(length(x), length(y)), -1, "
          "if(greaterthan(length(x), length(y)), 1, 0)))",
          makeRowVector({input})),
      "array_sort comparator does not support NULL sort keys");

  auto identityInput = makeNullableArrayVector<int32_t>({{2, std::nullopt, 1}});
  VELOX_ASSERT_THROW(
      evaluate(
          "array_sort(c0, (x, y) -> "
          "if(lessthan(x, y), -1, if(greaterthan(x, y), 1, 0)))",
          makeRowVector({identityInput})),
      "array_sort comparator does not support NULL sort keys");
}

TEST_F(ArraySortTest, comparatorNullSortKeysLastWhenRejectionDisabled) {
  queryCtx_->testingOverrideConfigUnsafe({
      {SparkQueryConfig::qualify(
           SparkQueryConfig::kArraySortRejectNullComparatorKeys),
       "false"},
  });

  auto input = makeNullableArrayVector<std::string>(
      {{"abcd123", "abcd", std::nullopt, "abc"}});
  auto result = evaluate(
      "array_sort(c0, (x, y) -> "
      "if(lessthan(length(x), length(y)), -1, "
      "if(greaterthan(length(x), length(y)), 1, 0)))",
      makeRowVector({input}));
  assertEqualVectors(
      makeNullableArrayVector<std::string>(
          {{"abc", "abcd", "abcd123", std::nullopt}}),
      result);

  auto identityInput = makeNullableArrayVector<int32_t>({{2, std::nullopt, 1}});
  result = evaluate(
      "array_sort(c0, (x, y) -> "
      "if(lessthan(x, y), 1, if(greaterthan(x, y), -1, 0)))",
      makeRowVector({identityInput}));
  assertEqualVectors(
      makeNullableArrayVector<int32_t>({{2, 1, std::nullopt}}), result);
}

TEST_F(ArraySortTest, comparatorIgnoresNullArrayWithStaleSize) {
  // Row 1 is null, but its offset and size still cover a null element.
  auto elements = makeNullableFlatVector<int32_t>({3, 1, 2, std::nullopt, 4});
  auto offsets = allocateOffsets(2, pool());
  auto sizes = allocateSizes(2, pool());
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  auto* rawSizes = sizes->asMutable<vector_size_t>();
  rawOffsets[0] = 0;
  rawSizes[0] = 3;
  rawOffsets[1] = 3;
  rawSizes[1] = 2;
  auto nulls = allocateNulls(2, pool());
  bits::setNull(nulls->asMutable<uint64_t>(), 1);
  auto input = std::make_shared<ArrayVector>(
      pool(), ARRAY(INTEGER()), nulls, 2, offsets, sizes, elements);

  using NullableIntArray = std::optional<std::vector<std::optional<int32_t>>>;
  auto expected = makeNullableArrayVector<int32_t>(
      std::vector<NullableIntArray>{{{1, 2, 3}}, std::nullopt});
  for (const auto& comparator :
       {"(x, y) -> if(lessthan(x, y), -1, if(greaterthan(x, y), 1, 0))",
        "(x, y) -> if(lessthan(x + 1, y + 1), -1, "
        "if(greaterthan(x + 1, y + 1), 1, 0))"}) {
    SCOPED_TRACE(comparator);
    auto result = evaluate(
        fmt::format("array_sort(c0, {})", comparator), makeRowVector({input}));
    assertEqualVectors(expected, result);
  }
}

TEST_F(ArraySortTest, comparatorSparseSelection) {
  auto input = makeArrayVector<int32_t>({
      {9},
      {3, 1, 2},
      {8},
      {6, 4, 5},
      {7},
  });
  auto expected = makeArrayVector<int32_t>({
      {9},
      {1, 2, 3},
      {8},
      {4, 5, 6},
      {7},
  });

  SelectivityVector rows(input->size(), false);
  rows.setValid(1, true);
  rows.setValid(3, true);
  rows.setValid(4, true);
  rows.updateBounds();

  auto result = evaluate(
      "array_sort(c0, (x, y) -> "
      "if(lessthan(x, y), -1, if(greaterthan(x, y), 1, 0)))",
      makeRowVector({input}),
      rows);
  for (const auto row : {1, 3, 4}) {
    assertEqualVectors(expected->slice(row, 1), result->slice(row, 1));
  }
}

TEST_F(ArraySortTest, comparatorNullKeyErrorIsRowIsolated) {
  auto input = makeNullableArrayVector<std::string>({
      {"bbb", "a", "cc"},
      {"bbb", std::nullopt, "a"},
      {"dddd", "bb", "c"},
  });
  using NullableStringArray =
      std::optional<std::vector<std::optional<std::string>>>;
  auto expected =
      makeNullableArrayVector<std::string>(std::vector<NullableStringArray>{
          {{"a", "cc", "bbb"}},
          std::nullopt,
          {{"c", "bb", "dddd"}},
      });

  auto result = evaluate(
      "try(array_sort(c0, (x, y) -> "
      "if(lessthan(length(x), length(y)), -1, "
      "if(greaterthan(length(x), length(y)), 1, 0))))",
      makeRowVector({input}));
  assertEqualVectors(expected, result);
}

TEST_F(ArraySortTest, comparatorRepeatedDictionaryRowsWithCaptures) {
  auto arrays = makeNullableArrayVector<int32_t>({{3, 1, 2}, {6, 4, 5}});
  auto dictionaryArrays = wrapInDictionary(makeIndices({0, 0, 0}), arrays);
  auto input = makeRowVector({
      dictionaryArrays,
      makeFlatVector<int32_t>({10, 0, 2}),
  });

  auto result = evaluate(
      "array_sort(c0, (x, y) -> "
      "if(lessthan(greatest(x, c1), greatest(y, c1)), -1, "
      "if(greaterthan(greatest(x, c1), greatest(y, c1)), 1, 0)))",
      input);
  assertEqualVectors(
      makeNullableArrayVector<int32_t>({{3, 1, 2}, {1, 2, 3}, {1, 2, 3}}),
      result);
}

TEST_F(ArraySortTest, comparatorLazyTopLevelArray) {
  auto input = makeArrayVector<int32_t>({{3, 1, 2}, {6, 4, 5}});
  auto lazyInput = wrapInLazyDictionary(input);

  auto result = evaluate(
      "array_sort(c0, (x, y) -> "
      "if(lessthan(x, y), -1, if(greaterthan(x, y), 1, 0)))",
      makeRowVector({lazyInput}));
  assertEqualVectors(makeArrayVector<int32_t>({{1, 2, 3}, {4, 5, 6}}), result);
}

TEST_F(ArraySortTest, identityComparatorPreservesSignedZeroOrder) {
  const std::string comparator =
      "(x, y) -> "
      "if(lessthan(x, y), -10, if(greaterthan(x, y), 10, 0))";

  {
    SCOPED_TRACE("scalar");
    auto input = makeArrayVector<double>({{-0.0, 0.0, 0.0, -0.0}, {0.0, -0.0}});
    auto result = evaluate(
        fmt::format("array_sort(c0, {})", comparator), makeRowVector({input}));

    auto* arrays = result->as<ArrayVector>();
    auto* elements = arrays->elements()->as<SimpleVector<double>>();
    ASSERT_NE(elements, nullptr);

    std::vector<bool> expectedSigns{true, false, false, true, false, true};
    for (vector_size_t index = 0; index < expectedSigns.size(); ++index) {
      EXPECT_EQ(std::signbit(elements->valueAt(index)), expectedSigns[index]);
    }
  }

  {
    SCOPED_TRACE("real");
    auto input =
        makeArrayVector<float>({{-0.0f, 0.0f, 0.0f, -0.0f}, {0.0f, -0.0f}});
    auto result = evaluate(
        fmt::format("array_sort(c0, {})", comparator), makeRowVector({input}));

    auto* arrays = result->as<ArrayVector>();
    auto* elements = arrays->elements()->as<SimpleVector<float>>();
    ASSERT_NE(elements, nullptr);

    std::vector<bool> expectedSigns{true, false, false, true, false, true};
    for (vector_size_t index = 0; index < expectedSigns.size(); ++index) {
      EXPECT_EQ(std::signbit(elements->valueAt(index)), expectedSigns[index]);
    }
  }

  {
    SCOPED_TRACE("array");
    const std::string arrayComparator =
        "(x, y) -> "
        "if(lessthan(element_at(x, 1), element_at(y, 1)), -10, "
        "if(greaterthan(element_at(x, 1), element_at(y, 1)), 10, 0))";
    using InnerArray = std::vector<std::optional<double>>;
    using OuterArray = std::vector<std::optional<InnerArray>>;
    auto input = makeNullableNestedArrayVector<double>({OuterArray{
        InnerArray{-0.0},
        InnerArray{0.0},
        InnerArray{0.0},
        InnerArray{-0.0},
    }});
    auto result = evaluate(
        fmt::format("array_sort(c0, {})", arrayComparator),
        makeRowVector({input}));

    auto* arrays = result->as<ArrayVector>();
    DecodedVector decodedElements(*arrays->elements());
    auto* innerArrays = decodedElements.base()->as<ArrayVector>();
    auto* values = innerArrays->elements()->as<SimpleVector<double>>();
    ASSERT_NE(values, nullptr);

    std::vector<bool> expectedSigns{true, false, false, true};
    for (vector_size_t index = 0; index < expectedSigns.size(); ++index) {
      const auto innerArrayIndex = decodedElements.indices()[index];
      EXPECT_EQ(
          std::signbit(values->valueAt(innerArrays->offsetAt(innerArrayIndex))),
          expectedSigns[index]);
    }
  }

  {
    SCOPED_TRACE("row");
    const std::string rowComparator =
        "(x, y) -> "
        "if(lessthan(x.value, y.value), -10, "
        "if(greaterthan(x.value, y.value), 10, 0))";
    auto rowType = ROW({"key", "value"}, {INTEGER(), DOUBLE()});
    auto input = makeArrayOfRowVector(
        rowType,
        {{
            variant::row({1, -0.0}),
            variant::row({1, 0.0}),
            variant::row({1, 0.0}),
            variant::row({1, -0.0}),
        }});
    auto result = evaluate(
        fmt::format("array_sort(c0, {})", rowComparator),
        makeRowVector({input}));

    auto* arrays = result->as<ArrayVector>();
    DecodedVector decodedElements(*arrays->elements());
    auto* rows = decodedElements.base()->as<RowVector>();
    auto* values = rows->childAt(1)->as<SimpleVector<double>>();
    ASSERT_NE(values, nullptr);

    std::vector<bool> expectedSigns{true, false, false, true};
    for (vector_size_t index = 0; index < expectedSigns.size(); ++index) {
      EXPECT_EQ(
          std::signbit(values->valueAt(decodedElements.indices()[index])),
          expectedSigns[index]);
    }
  }
}

TEST_F(ArraySortTest, comparatorEncodings) {
  auto input =
      makeNullableArrayVector<int32_t>({{3, 1, 2}, {1, std::nullopt, 2}});
  const std::string comparator =
      "(x, y) -> if(lessthan(coalesce(x, -1000), coalesce(y, -1000)), -10, "
      "if(greaterthan(coalesce(x, -1000), coalesce(y, -1000)), 10, 0))";

  auto dictionaryInput = wrapInDictionary(makeIndices({1, 0, 1}), input);
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

// Comparator rewrites order elements by the sort key, so the elements do not
// need to be orderable.
TEST_F(ArraySortTest, comparatorNonOrderableElements) {
  using Map = std::vector<std::pair<int32_t, std::optional<int32_t>>>;
  using MapArrays = std::vector<std::vector<Map>>;
  auto input = makeArrayOfMapVector<int32_t, int32_t>(MapArrays{
      {Map{{1, 10}, {2, 20}}, Map{{3, 30}}, Map{}, Map{{5, 50}}},
  });

  auto result = evaluate(
      "array_sort(c0, (x, y) -> "
      "if(lessthan(size(x, false), size(y, false)), -1, "
      "if(greaterthan(size(x, false), size(y, false)), 1, 0)))",
      makeRowVector({input}));
  assertEqualVectors(
      makeArrayOfMapVector<int32_t, int32_t>(MapArrays{
          {Map{}, Map{{3, 30}}, Map{{5, 50}}, Map{{1, 10}, {2, 20}}},
      }),
      result);

  result = evaluate(
      "array_sort(c0, (x, y) -> "
      "if(lessthan(size(x, false), size(y, false)), 1, "
      "if(greaterthan(size(x, false), size(y, false)), -1, 0)))",
      makeRowVector({input}));
  assertEqualVectors(
      makeArrayOfMapVector<int32_t, int32_t>(MapArrays{
          {Map{{1, 10}, {2, 20}}, Map{{3, 30}}, Map{{5, 50}}, Map{}},
      }),
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

  VELOX_ASSERT_THROW(
      evaluate(
          "array_sort(c0, (a, b) -> "
          "if(lessthan(add(a, cast(rand() as integer)), "
          "add(b, cast(rand() as integer))), -1, "
          "if(greaterthan(add(a, cast(rand() as integer)), "
          "add(b, cast(rand() as integer))), 1, 0)))",
          data),
      "array_sort with comparator lambda that cannot be rewritten into a transform is not supported");

  VELOX_ASSERT_THROW(
      evaluate(
          "array_sort(c0, (a, b) -> "
          "if(lessthan(a, b), -1, "
          "if(greaterthan(a, b), 1, cast(null as integer))))",
          data),
      "array_sort with comparator lambda that cannot be rewritten into a transform is not supported");
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
