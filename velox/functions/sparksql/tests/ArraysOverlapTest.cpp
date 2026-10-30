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
#include <cstring>
#include <limits>
#include <optional>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/Expressions.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

template <typename T>
T nanWithPayload() {
  if constexpr (std::is_same_v<T, double>) {
    const uint64_t bits = 0x7ff8000000000001ULL;
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  } else {
    const uint32_t bits = 0x7fc00001U;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
}

class ArraysOverlapTest : public SparkFunctionBaseTest {
 protected:
  // Checks arrays_overlap on two input columns, in both argument orders.
  void testOverlap(
      const VectorPtr& left,
      const VectorPtr& right,
      const VectorPtr& expected) {
    auto input = makeRowVector({left, right});
    assertEqualVectors(expected, evaluate("arrays_overlap(c0, c1)", input));
    assertEqualVectors(expected, evaluate("arrays_overlap(c1, c0)", input));
  }

  // Checks arrays_overlap with 'column' as an input column and 'constant', a
  // single-row array vector, as a constant argument on either side. This takes
  // the path that builds the constant's set once, in initialize().
  void testOverlapWithConstant(
      const VectorPtr& column,
      const VectorPtr& constant,
      const VectorPtr& expected) {
    auto input = makeRowVector({column});
    auto field =
        std::make_shared<core::FieldAccessTypedExpr>(column->type(), "c0");
    auto constantExpr = std::make_shared<core::ConstantTypedExpr>(
        BaseVector::wrapInConstant(1, 0, constant));
    for (bool constantOnLeft : {false, true}) {
      SCOPED_TRACE(fmt::format("constantOnLeft: {}", constantOnLeft));
      std::vector<core::TypedExprPtr> args = constantOnLeft
          ? std::vector<core::TypedExprPtr>{constantExpr, field}
          : std::vector<core::TypedExprPtr>{field, constantExpr};
      auto call = std::make_shared<core::CallTypedExpr>(
          BOOLEAN(), std::move(args), "arrays_overlap");
      assertEqualVectors(expected, evaluate(call, input));
    }
  }

  template <typename T>
  void testIntegral() {
    auto left = makeNullableArrayVector<T>({
        {1, 2, 3},
        {1, 2, 3},
        {1, 2, std::nullopt},
        {1, 2, std::nullopt},
        {},
        {std::nullopt},
        {std::nullopt},
        {1, std::nullopt},
        {1, 1, 2},
        {std::nullopt, 2},
        {2},
        {},
    });
    auto right = makeNullableArrayVector<T>({
        {3, 4, 5},
        {4, 5, 6},
        {4, 5, 6},
        {2, 4, 5},
        {1, 2},
        {std::nullopt},
        {},
        {3},
        {2, 2},
        {2, 3},
        {std::nullopt, 2, 5},
        {},
    });
    // A null before the matching element does not stop the search: the
    // elements of the bigger array are looked up in order, so {2} vs
    // {null, 2, 5} meets the null first.
    auto expected = makeNullableFlatVector<bool>({
        true,
        false,
        std::nullopt,
        true,
        false,
        std::nullopt,
        false,
        std::nullopt,
        true,
        true,
        true,
        false,
    });
    testOverlap(left, right, expected);
  }

  template <typename T>
  void testFloatingPoint() {
    constexpr T kNaN = std::numeric_limits<T>::quiet_NaN();
    constexpr T kSignalingNaN = std::numeric_limits<T>::signaling_NaN();
    constexpr T kInf = std::numeric_limits<T>::infinity();
    const T kNaNWithPayload = nanWithPayload<T>();

    auto left = makeNullableArrayVector<T>({
        {-0.0},
        {-0.0},
        {0.0},
        {kNaN},
        {kSignalingNaN},
        {kNaNWithPayload},
        {kInf},
        {-kInf},
        {1.5, -0.0},
        {-0.0, std::nullopt},
    });
    auto right = makeNullableArrayVector<T>({
        {0.0},
        {-0.0},
        {0.0},
        {kNaN},
        {kNaN},
        {kNaN},
        {kInf},
        {kInf},
        {0.0, 1.5},
        {0.0},
    });
    // Spark compares top-level floating-point elements by their bits, with all
    // NaNs equal: -0.0 and 0.0 are different values.
    auto expected = makeNullableFlatVector<bool>({
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        false,
        true,
        std::nullopt,
    });
    testOverlap(left, right, expected);
  }
};

// Spark 3.4 results for these cases, from running the same inputs:
// overlap -> true, no overlap -> false, no overlap with a null -> null, an
// empty array -> false even if the other array has a null.
TEST_F(ArraysOverlapTest, integralTypes) {
  testIntegral<int8_t>();
  testIntegral<int16_t>();
  testIntegral<int32_t>();
  testIntegral<int64_t>();
}

TEST_F(ArraysOverlapTest, nullArray) {
  auto left = makeNullableArrayVector<int32_t>(
      std::vector<std::optional<std::vector<std::optional<int32_t>>>>{
          std::nullopt,
          std::vector<std::optional<int32_t>>{1, 2},
          std::nullopt});
  auto right = makeNullableArrayVector<int32_t>(
      std::vector<std::optional<std::vector<std::optional<int32_t>>>>{
          std::vector<std::optional<int32_t>>{1, 2},
          std::vector<std::optional<int32_t>>{3, 4},
          std::vector<std::optional<int32_t>>{}});
  auto expected =
      makeNullableFlatVector<bool>({std::nullopt, false, std::nullopt});
  testOverlap(left, right, expected);
}

TEST_F(ArraysOverlapTest, boolean) {
  auto left = makeNullableArrayVector<bool>({
      {true},
      {true, false},
      {true, std::nullopt},
  });
  auto right = makeArrayVector<bool>({
      {false},
      {false},
      {false},
  });
  auto expected = makeNullableFlatVector<bool>({false, true, std::nullopt});
  testOverlap(left, right, expected);
}

TEST_F(ArraysOverlapTest, floatingPoint) {
  testFloatingPoint<double>();
  testFloatingPoint<float>();
}

TEST_F(ArraysOverlapTest, strings) {
  const std::string kLong1 = "a string longer than twelve bytes";
  const std::string kLong2 = "another string longer than twelve bytes";
  auto left = makeNullableArrayVector<StringView>({
      {"sua"_sv, "foo"_sv},
      {"bar"_sv, "baz"_sv},
      {"x"_sv, std::nullopt},
      {StringView(kLong1), "y"_sv},
      {StringView(kLong1)},
      {""_sv},
  });
  auto right = makeNullableArrayVector<StringView>({
      {"sua"_sv, "cmc"_sv, "ea"_sv},
      {"sua"_sv, "cmc"_sv, "ea"_sv},
      {"sua"_sv, "cmc"_sv, "ea"_sv},
      {StringView(kLong2), StringView(kLong1)},
      {StringView(kLong2)},
      {""_sv},
  });
  auto expected = makeNullableFlatVector<bool>(
      {true, false, std::nullopt, true, false, true});
  testOverlap(left, right, expected);
}

TEST_F(ArraysOverlapTest, varbinary) {
  auto left = makeNullableArrayVector<StringView>(
      {{"\x01"_sv, "\x02"_sv}, {"\x03"_sv}, {std::nullopt}},
      ARRAY(VARBINARY()));
  auto right = makeNullableArrayVector<StringView>(
      {{"\x02"_sv}, {"\x04"_sv}, {"\x05"_sv}}, ARRAY(VARBINARY()));
  auto expected = makeNullableFlatVector<bool>({true, false, std::nullopt});
  testOverlap(left, right, expected);
}

TEST_F(ArraysOverlapTest, decimal) {
  auto shortLeft = makeNullableArrayVector<int64_t>(
      {{12345, 100}, {12345}, {std::nullopt}}, ARRAY(DECIMAL(10, 2)));
  auto shortRight =
      makeArrayVector<int64_t>({{100}, {54321}, {54321}}, DECIMAL(10, 2));
  testOverlap(
      shortLeft,
      shortRight,
      makeNullableFlatVector<bool>({true, false, std::nullopt}));

  const int128_t kBig = HugeInt::build(1, 0);
  auto longLeft = makeNullableArrayVector<int128_t>(
      {{kBig, 7}, {kBig}, {std::nullopt}}, ARRAY(DECIMAL(30, 5)));
  auto longRight =
      makeArrayVector<int128_t>({{7}, {kBig + 1}, {kBig}}, DECIMAL(30, 5));
  testOverlap(
      longLeft,
      longRight,
      makeNullableFlatVector<bool>({true, false, std::nullopt}));
}

TEST_F(ArraysOverlapTest, dateAndTimestamp) {
  auto dateLeft = makeNullableArrayVector<int32_t>(
      {{18000, 18001}, {18000}, {std::nullopt}}, ARRAY(DATE()));
  auto dateRight =
      makeArrayVector<int32_t>({{18001}, {18002}, {18002}}, DATE());
  testOverlap(
      dateLeft,
      dateRight,
      makeNullableFlatVector<bool>({true, false, std::nullopt}));

  auto timestampLeft = makeNullableArrayVector<Timestamp>({
      {Timestamp(1, 0), Timestamp(2, 500)},
      {Timestamp(2, 500)},
      {std::nullopt},
  });
  auto timestampRight = makeNullableArrayVector<Timestamp>({
      {Timestamp(2, 500)},
      {Timestamp(2, 501)},
      {Timestamp(3, 0)},
  });
  testOverlap(
      timestampLeft,
      timestampRight,
      makeNullableFlatVector<bool>({true, false, std::nullopt}));
}

TEST_F(ArraysOverlapTest, intervals) {
  auto dayTimeLeft = makeNullableArrayVector<int64_t>(
      {{1000, 2000}, {1000}, {std::nullopt}}, ARRAY(INTERVAL_DAY_TIME()));
  auto dayTimeRight =
      makeArrayVector<int64_t>({{2000}, {3000}, {3000}}, INTERVAL_DAY_TIME());
  testOverlap(
      dayTimeLeft,
      dayTimeRight,
      makeNullableFlatVector<bool>({true, false, std::nullopt}));

  auto yearMonthLeft = makeNullableArrayVector<int32_t>(
      {{12, 24}, {12}, {std::nullopt}}, ARRAY(INTERVAL_YEAR_MONTH()));
  auto yearMonthRight =
      makeArrayVector<int32_t>({{24}, {36}, {36}}, INTERVAL_YEAR_MONTH());
  testOverlap(
      yearMonthLeft,
      yearMonthRight,
      makeNullableFlatVector<bool>({true, false, std::nullopt}));
}

TEST_F(ArraysOverlapTest, constantArgument) {
  const std::string kLong = "a constant string longer than twelve bytes";
  auto column = makeNullableArrayVector<StringView>({
      {"sua"_sv, "foo"_sv},
      {"bar"_sv},
      {"bar"_sv, std::nullopt},
      {},
      {StringView(kLong)},
  });

  // The dominant production pattern: a column against a constant array.
  testOverlapWithConstant(
      column,
      makeArrayVector<StringView>({{"sua"_sv, "cmc"_sv, "ea"_sv}}),
      makeNullableFlatVector<bool>({true, false, std::nullopt, false, false}));

  // A long string in the constant is copied in initialize(), so it is still
  // found after the constant vector's own view is gone.
  testOverlapWithConstant(
      column,
      makeArrayVector<StringView>({{StringView(kLong), "zzz"_sv}}),
      makeNullableFlatVector<bool>({false, false, std::nullopt, false, true}));

  // A constant with a null and no match gives null, except against an empty
  // array.
  testOverlapWithConstant(
      column,
      makeNullableArrayVector<StringView>({{"cmc"_sv, std::nullopt}}),
      makeNullableFlatVector<bool>(
          {std::nullopt, std::nullopt, std::nullopt, false, std::nullopt}));

  // An empty constant never overlaps, even with a null in the other array.
  testOverlapWithConstant(
      column,
      makeArrayVector<StringView>({{}}),
      makeFlatVector<bool>({false, false, false, false, false}));

  // A null in either array before the matching element still gives true.
  testOverlapWithConstant(
      makeNullableArrayVector<StringView>({
          {std::nullopt, "sua"_sv},
          {"sua"_sv},
          {"x"_sv},
      }),
      makeNullableArrayVector<StringView>({{std::nullopt, "sua"_sv}}),
      makeNullableFlatVector<bool>({true, true, std::nullopt}));
}

TEST_F(ArraysOverlapTest, constantFloatingPoint) {
  auto column = makeArrayVector<double>({
      {0.0},
      {-0.0},
      {nanWithPayload<double>()},
      {1.0},
  });
  testOverlapWithConstant(
      column,
      makeArrayVector<double>(
          {{-0.0, std::numeric_limits<double>::quiet_NaN()}}),
      makeFlatVector<bool>({false, true, true, false}));
}

TEST_F(ArraysOverlapTest, bothArgumentsConstant) {
  auto input = makeRowVector({makeFlatVector<int32_t>({1, 2})});
  assertEqualVectors(
      makeConstant(true, 2),
      evaluate("arrays_overlap(ARRAY[1, 2], ARRAY[2, 3])", input));
  assertEqualVectors(
      makeConstant(false, 2),
      evaluate("arrays_overlap(ARRAY[1, 2], ARRAY[3, 4])", input));
}

// The constant set built in initialize() is reused for every batch an
// expression evaluates.
TEST_F(ArraysOverlapTest, constantAcrossBatches) {
  const std::string kLong = "a constant string longer than twelve bytes";
  auto field =
      std::make_shared<core::FieldAccessTypedExpr>(ARRAY(VARCHAR()), "c0");
  auto constantExpr =
      std::make_shared<core::ConstantTypedExpr>(BaseVector::wrapInConstant(
          1, 0, makeArrayVector<StringView>({{StringView(kLong), "ea"_sv}})));
  exec::ExprSet exprSet(
      {std::make_shared<core::CallTypedExpr>(
          BOOLEAN(),
          std::vector<core::TypedExprPtr>{field, constantExpr},
          "arrays_overlap")},
      &execCtx_);

  auto firstBatch = makeRowVector({makeArrayVector<StringView>({
      {"ea"_sv},
      {"x"_sv},
  })});
  assertEqualVectors(
      makeFlatVector<bool>({true, false}), evaluate(exprSet, firstBatch));

  auto secondBatch = makeRowVector({makeArrayVector<StringView>({
      {StringView(kLong)},
      {"y"_sv, "ea"_sv},
      {"z"_sv},
  })});
  assertEqualVectors(
      makeFlatVector<bool>({true, true, false}),
      evaluate(exprSet, secondBatch));
}

TEST_F(ArraysOverlapTest, largeArrays) {
  std::vector<int64_t> big(1'000);
  for (int64_t i = 0; i < big.size(); ++i) {
    big[i] = i;
  }
  auto left = makeArrayVector<int64_t>({big, big});
  auto right = makeArrayVector<int64_t>({{-1, 999}, {-1, 1'000}});
  testOverlap(left, right, makeFlatVector<bool>({true, false}));
}

TEST_F(ArraysOverlapTest, dictionaryEncodedInput) {
  auto left = makeNullableArrayVector<int32_t>({
      {1, 2},
      {3, std::nullopt},
      {5},
  });
  auto right = makeArrayVector<int32_t>({
      {5},
      {4},
      {2},
  });
  // Reverses the rows of 'left': {5}, {3, null}, {1, 2}.
  auto dictionaryLeft = wrapInDictionary(makeIndicesInReverse(3), left);
  testOverlap(
      dictionaryLeft,
      right,
      makeNullableFlatVector<bool>({true, std::nullopt, true}));
}

// Structs and arrays are compared like Spark's ordering: -0.0 equals 0.0, NaNs
// are equal and nulls inside an element are equal.
TEST_F(ArraysOverlapTest, arrayOfRows) {
  const auto rowType = ROW({{"a", DOUBLE()}, {"b", VARCHAR()}});
  auto left = makeArrayOfRowVector(
      rowType,
      {
          {Variant::row({-0.0, "x"})},
          {Variant::row({std::numeric_limits<double>::quiet_NaN(), "x"})},
          {Variant::row({1.0, Variant::null(TypeKind::VARCHAR)})},
          {Variant::row({1.0, "x"})},
          {Variant::row({1.0, "x"}), Variant(TypeKind::ROW)},
          {},
          {Variant(TypeKind::ROW), Variant::row({1.0, "x"})},
      });
  auto right = makeArrayOfRowVector(
      rowType,
      {
          {Variant::row({0.0, "x"})},
          {Variant::row({nanWithPayload<double>(), "x"})},
          {Variant::row({1.0, Variant::null(TypeKind::VARCHAR)})},
          {Variant::row({1.0, "y"})},
          {Variant::row({2.0, "x"})},
          {Variant(TypeKind::ROW)},
          {Variant::row({1.0, "x"})},
      });
  // The last row has a null element before the matching one.
  auto expected = makeNullableFlatVector<bool>(
      {true, true, true, false, std::nullopt, false, true});
  testOverlap(left, right, expected);
}

TEST_F(ArraysOverlapTest, arrayOfArrays) {
  auto left = makeArrayVector(
      {0, 1, 2},
      makeNullableArrayVector<double>({
          {-0.0},
          {std::numeric_limits<double>::quiet_NaN()},
          {1.0, std::nullopt},
      }));
  auto right = makeArrayVector(
      {0, 1, 2},
      makeNullableArrayVector<double>({
          {0.0},
          {nanWithPayload<double>()},
          {1.0, std::nullopt},
      }));
  testOverlap(left, right, makeFlatVector<bool>({true, true, true}));
}

// Arrays whose element type is unknown hold only nulls, as Spark's
// array<void>.
TEST_F(ArraysOverlapTest, unknownElementType) {
  auto left = makeArrayVector(
      {0, 1, 1, 2}, BaseVector::createNullConstant(UNKNOWN(), 2, pool()));
  auto right = makeArrayVector(
      {0, 1, 2, 3}, BaseVector::createNullConstant(UNKNOWN(), 3, pool()));
  // Rows: [null] vs [null], [] vs [null], [null] vs [null], [] vs []. The last
  // is Spark's arrays_overlap(array(), array()).
  testOverlap(
      left,
      right,
      makeNullableFlatVector<bool>({std::nullopt, false, std::nullopt, false}));
}

// Spark rejects element types that are not orderable, such as maps, before
// execution. No signature matches them here either.
TEST_F(ArraysOverlapTest, mapElementsNotSupported) {
  std::vector<
      std::vector<std::vector<std::pair<StringView, std::optional<int64_t>>>>>
      data{{{{"a"_sv, 1}}}};
  auto arrayOfMap = makeArrayOfMapVector<StringView, int64_t>(data);
  VELOX_ASSERT_THROW(
      evaluate(
          "arrays_overlap(c0, c1)", makeRowVector({arrayOfMap, arrayOfMap})),
      "Scalar function signature is not supported: arrays_overlap");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
