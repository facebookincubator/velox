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
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class MapSubsetKeyInRangeTest : public test::FunctionBaseTest {};

TEST_F(MapSubsetKeyInRangeTest, basicBigintKey) {
  auto inputMap = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:40, 5:50}",
      "{7:70, 10:100, 14:140, 20:200}",
      "{}",
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{2:20, 3:30, 4:40, 5:50}",
      "{7:70, 10:100, 14:140}",
      "{}",
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, cast(2 as bigint), cast(14 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, boundaryInclusivity) {
  auto inputMap = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:40, 5:50}",
  });

  // Bounds are inclusive: keys 1 and 5 should be included.
  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:40, 5:50}",
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, cast(1 as bigint), cast(5 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, allKeysInRange) {
  auto inputMap = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30}",
      "{5:50}",
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30}",
      "{5:50}",
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, cast(0 as bigint), cast(100 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, allKeysOutOfRange) {
  auto inputMap = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30}",
      "{100:1000, 200:2000}",
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{}",
      "{}",
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, cast(50 as bigint), cast(60 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, lowGreaterThanHigh) {
  auto inputMap = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:40, 5:50}",
      "{7:70, 14:140}",
  });

  // When low > high, the result is always an empty map.
  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{}",
      "{}",
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, cast(14 as bigint), cast(7 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, emptyMap) {
  auto inputMap =
      makeMapVectorFromJson<int64_t, int32_t>({"{}", "{}", "{1:10}"});

  auto expected =
      makeMapVectorFromJson<int64_t, int32_t>({"{}", "{}", "{1:10}"});

  auto result = evaluate(
      "map_subset_key_in_range(c0, cast(0 as bigint), cast(100 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, nullMap) {
  auto inputMap = makeNullableMapVector<int64_t, int32_t>({
      std::nullopt,
      {{{1, 10}, {2, 20}, {3, 30}}},
      std::nullopt,
  });

  auto expected = makeNullableMapVector<int64_t, int32_t>({
      std::nullopt,
      {{{2, 20}}},
      std::nullopt,
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, cast(2 as bigint), cast(2 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

// Per Velox default null behavior (void call, not callNullable), if low_key
// is null the result is null regardless of input map contents.
TEST_F(MapSubsetKeyInRangeTest, nullLowKey) {
  auto inputMap = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30}",
      "{4:40, 5:50}",
      "{}",
  });

  auto lowKeys = makeNullableFlatVector<int64_t>({
      std::nullopt,
      2,
      std::nullopt,
  });
  auto highKeys = makeFlatVector<int64_t>({10, 4, 100});

  // Row 0: low is null -> null. Row 1: normal -> {4:40}. Row 2: low is
  // null -> null (even though input map is empty).
  auto expected = makeNullableMapVector<int64_t, int32_t>({
      std::nullopt,
      {{{4, 40}}},
      std::nullopt,
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, c1, c2)",
      makeRowVector({inputMap, lowKeys, highKeys}));
  assertEqualVectors(expected, result);
}

// Per Velox default null behavior, if high_key is null the result is null.
TEST_F(MapSubsetKeyInRangeTest, nullHighKey) {
  auto inputMap = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30}",
      "{4:40, 5:50}",
  });

  auto lowKeys = makeFlatVector<int64_t>({1, 4});
  auto highKeys = makeNullableFlatVector<int64_t>({std::nullopt, 5});

  auto expected = makeNullableMapVector<int64_t, int32_t>({
      std::nullopt,
      {{{4, 40}, {5, 50}}},
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, c1, c2)",
      makeRowVector({inputMap, lowKeys, highKeys}));
  assertEqualVectors(expected, result);
}

// Both bounds null -> null. Also covers the varchar specialization to make
// sure default null handling applies there too.
TEST_F(MapSubsetKeyInRangeTest, nullBothBoundsVarchar) {
  auto inputMap = makeMapVectorFromJson<std::string, int32_t>({
      R"({"a":1, "b":2, "c":3})",
      R"({"x":10, "y":20})",
  });

  auto lowKeys = makeNullableFlatVector<std::string>({std::nullopt, "x"});
  auto highKeys = makeNullableFlatVector<std::string>({std::nullopt, "y"});

  auto expected = makeNullableMapVector<std::string, int32_t>({
      std::nullopt,
      {{{"x", 10}, {"y", 20}}},
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, c1, c2)",
      makeRowVector({inputMap, lowKeys, highKeys}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, nullValuesPreserved) {
  auto inputMap = makeNullableMapVector<int64_t, int32_t>({
      {{{1, 10}, {2, std::nullopt}, {3, 30}, {4, std::nullopt}, {5, 50}}},
  });

  auto expected = makeNullableMapVector<int64_t, int32_t>({
      {{{2, std::nullopt}, {3, 30}, {4, std::nullopt}}},
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, cast(2 as bigint), cast(4 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, varcharKey) {
  auto inputMap = makeMapVectorFromJson<std::string, int32_t>({
      R"({"apple":1, "banana":2, "cherry":3, "date":4, "eggplant":5})",
      R"({"x":10, "y":20, "z":30})",
  });

  auto expected = makeMapVectorFromJson<std::string, int32_t>({
      R"({"banana":2, "cherry":3, "date":4})",
      R"({})",
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, 'banana', 'date')",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, varcharKeyBoundaryInclusive) {
  auto inputMap = makeMapVectorFromJson<std::string, int32_t>({
      R"({"apple":1, "banana":2, "cherry":3})",
  });

  auto expected = makeMapVectorFromJson<std::string, int32_t>({
      R"({"apple":1, "banana":2, "cherry":3})",
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, 'apple', 'cherry')",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, intKeyTypes) {
  // Verify int32 (integer) keys work via primitive specialization.
  auto inputMap = makeMapVectorFromJson<int32_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:40}",
  });

  auto expected = makeMapVectorFromJson<int32_t, int32_t>({
      "{2:20, 3:30}",
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, cast(2 as integer), cast(3 as integer))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, doubleKey) {
  auto inputMap = makeMapVectorFromJson<double, int32_t>({
      "{1.5:1, 2.5:2, 3.5:3, 4.5:4}",
  });

  auto expected = makeMapVectorFromJson<double, int32_t>({
      "{2.5:2, 3.5:3}",
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, 2.0, 4.0)", makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, sameLowAndHigh) {
  auto inputMap = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30}",
      "{5:50}",
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{2:20}",
      "{}",
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, cast(2 as bigint), cast(2 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, complexValues) {
  // Map<bigint, array<int>>: verify generic value type is preserved.
  auto inputMap = makeMapVector(
      {0, 3},
      makeFlatVector<int64_t>({1, 2, 3, 10, 20}),
      makeArrayVectorFromJson<int32_t>({
          "[1, 2]",
          "[3, 4]",
          "[5, 6]",
          "[100]",
          "[200]",
      }));

  auto expected = makeMapVector(
      {0, 2},
      makeFlatVector<int64_t>({2, 3, 10}),
      makeArrayVectorFromJson<int32_t>({
          "[3, 4]",
          "[5, 6]",
          "[100]",
      }));

  auto result = evaluate(
      "map_subset_key_in_range(c0, cast(2 as bigint), cast(10 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

// Exercises MapSubsetKeyInRangeGenericFunction (the Orderable<T1> fallback)
// using an array key. Each input row carries its own pair of array bounds.
TEST_F(MapSubsetKeyInRangeTest, arrayKeyGeneric) {
  auto inputMap = makeMapVector(
      {0, 2, 4},
      makeArrayVectorFromJson<int32_t>({
          "[1, 2, 3]",
          "[4, 5]",
          "[1, 2]",
          "[7, 8]",
          "[3, 4]",
          "[6, 7]",
      }),
      makeFlatVector<int32_t>({10, 20, 30, 40, 50, 60}));

  auto lowKeys = makeArrayVectorFromJson<int32_t>({
      "[1, 2, 3]",
      "[1, 2]",
      "[3, 4]",
  });
  auto highKeys = makeArrayVectorFromJson<int32_t>({
      "[4, 5]",
      "[7, 8]",
      "[3, 4]",
  });

  // Lexicographic order of arrays:
  //   row 0: [1,2,3] <= [1,2,3] and [4,5] <= [4,5] -> both kept
  //   row 1: [1,2] kept; [7,8] kept (= high)
  //   row 2: [3,4] kept (= low = high), [6,7] excluded (> high)
  auto expected = makeMapVector(
      {0, 2, 4},
      makeArrayVectorFromJson<int32_t>({
          "[1, 2, 3]",
          "[4, 5]",
          "[1, 2]",
          "[7, 8]",
          "[3, 4]",
      }),
      makeFlatVector<int32_t>({10, 20, 30, 40, 50}));

  auto result = evaluate(
      "map_subset_key_in_range(c0, c1, c2)",
      makeRowVector({inputMap, lowKeys, highKeys}));
  assertEqualVectors(expected, result);
}

// Exercises the generic path with low > high — should yield an empty map.
TEST_F(MapSubsetKeyInRangeTest, arrayKeyGenericLowGreaterThanHigh) {
  auto inputMap = makeMapVector(
      {0},
      makeArrayVectorFromJson<int32_t>({
          "[1, 2]",
          "[3, 4]",
          "[5, 6]",
      }),
      makeFlatVector<int32_t>({10, 20, 30}));

  auto lowKeys = makeArrayVectorFromJson<int32_t>({"[5, 6]"});
  auto highKeys = makeArrayVectorFromJson<int32_t>({"[1, 2]"});

  auto expected = makeMapVector(
      {0}, makeArrayVectorFromJson<int32_t>({}), makeFlatVector<int32_t>({}));

  auto result = evaluate(
      "map_subset_key_in_range(c0, c1, c2)",
      makeRowVector({inputMap, lowKeys, highKeys}));
  assertEqualVectors(expected, result);
}

// Exercises the generic path with row keys (an Orderable type).
TEST_F(MapSubsetKeyInRangeTest, rowKeyGeneric) {
  auto keyRows = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4}),
      makeFlatVector<int32_t>({10, 20, 30, 40}),
  });
  auto inputMap = makeMapVector(
      {0}, keyRows, makeFlatVector<int32_t>({100, 200, 300, 400}));

  auto lowKeyRows = makeRowVector({
      makeFlatVector<int32_t>({2}),
      makeFlatVector<int32_t>({20}),
  });
  auto highKeyRows = makeRowVector({
      makeFlatVector<int32_t>({3}),
      makeFlatVector<int32_t>({30}),
  });

  // Rows compare lexicographically: only (2,20) and (3,30) are in [low, high].
  auto expectedKeyRows = makeRowVector({
      makeFlatVector<int32_t>({2, 3}),
      makeFlatVector<int32_t>({20, 30}),
  });
  auto expected =
      makeMapVector({0}, expectedKeyRows, makeFlatVector<int32_t>({200, 300}));

  auto result = evaluate(
      "map_subset_key_in_range(c0, c1, c2)",
      makeRowVector({inputMap, lowKeyRows, highKeyRows}));
  assertEqualVectors(expected, result);
}

// The generic path must reject keys containing null elements because
// comparison on indeterminate values is not defined.
TEST_F(MapSubsetKeyInRangeTest, genericKeyWithNullElementsThrows) {
  auto inputMap = makeMapVector(
      {0},
      makeArrayVectorFromJson<int32_t>({
          "[1, null]",
          "[2, 3]",
      }),
      makeFlatVector<int32_t>({10, 20}));

  auto lowKeys = makeArrayVectorFromJson<int32_t>({"[1, 1]"});
  auto highKeys = makeArrayVectorFromJson<int32_t>({"[3, 3]"});

  VELOX_ASSERT_THROW(
      evaluate(
          "map_subset_key_in_range(c0, c1, c2)",
          makeRowVector({inputMap, lowKeys, highKeys})),
      "Ordering nulls is not supported");
}

// Floating-point keys must use NaN-aware ordering: NaN > all non-NaN values
// and NaN == NaN.
TEST_F(MapSubsetKeyInRangeTest, doubleKeyNaN) {
  static const auto kNaN = std::numeric_limits<double>::quiet_NaN();
  static const auto kSNaN = std::numeric_limits<double>::signaling_NaN();

  // A NaN key is included only when the upper bound is NaN. Bounds [1, 5]
  // exclude NaN; bounds [1, NaN] include it; bounds [NaN, NaN] keep only NaN.
  auto inputMap = makeMapVectorFromJson<double, int32_t>({
      "{1.0:10, 2.0:20, NaN:99, 5.0:50}",
      "{1.0:10, 2.0:20, NaN:99, 5.0:50}",
      "{1.0:10, 2.0:20, NaN:99, 5.0:50}",
  });

  auto lowKeys = makeFlatVector<double>({1.0, 1.0, kNaN});
  auto highKeys = makeFlatVector<double>({5.0, kSNaN, kNaN});

  auto expected = makeMapVectorFromJson<double, int32_t>({
      "{1.0:10, 2.0:20, 5.0:50}",
      "{1.0:10, 2.0:20, NaN:99, 5.0:50}",
      "{NaN:99}",
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, c1, c2)",
      makeRowVector({inputMap, lowKeys, highKeys}));
  assertEqualVectors(expected, result);
}

TEST_F(MapSubsetKeyInRangeTest, floatKeyNaN) {
  static const auto kNaN = std::numeric_limits<float>::quiet_NaN();

  auto inputMap = makeMapVectorFromJson<float, int32_t>({
      "{1.0:10, 2.0:20, NaN:99}",
  });

  auto lowKeys = makeFlatVector<float>({2.0f});
  auto highKeys = makeFlatVector<float>({kNaN});

  // NaN is treated as the maximum, so [2.0, NaN] keeps 2.0 and NaN.
  auto expected = makeMapVectorFromJson<float, int32_t>({
      "{2.0:20, NaN:99}",
  });

  auto result = evaluate(
      "map_subset_key_in_range(c0, c1, c2)",
      makeRowVector({inputMap, lowKeys, highKeys}));
  assertEqualVectors(expected, result);
}

} // namespace
} // namespace facebook::velox::functions
