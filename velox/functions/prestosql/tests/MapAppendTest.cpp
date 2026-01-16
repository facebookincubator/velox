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

class MapAppendTest : public test::FunctionBaseTest {};

TEST_F(MapAppendTest, basicIntegerMap) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30}",
          "{10:100, 20:200}",
          "{}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[4, 5]",
          "[30, 40]",
          "[1, 2]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[40, 50]",
          "[300, 400]",
          "[10, 20]",
      }),
  });

  auto result = evaluate("map_append(c0, c1, c2)", data);

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:40, 5:50}",
      "{10:100, 20:200, 30:300, 40:400}",
      "{1:10, 2:20}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapAppendTest, varcharKeyMap) {
  auto data = makeRowVector({
      makeMapVectorFromJson<std::string, int32_t>({
          R"({"a": 1, "b": 2})",
          R"({"x": 10})",
      }),
      makeArrayVectorFromJson<std::string>({
          R"(["c", "d"])",
          R"(["y", "z"])",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[3, 4]",
          "[20, 30]",
      }),
  });

  auto result = evaluate("map_append(c0, c1, c2)", data);

  auto expected = makeMapVectorFromJson<std::string, int32_t>({
      R"({"a": 1, "b": 2, "c": 3, "d": 4})",
      R"({"x": 10, "y": 20, "z": 30})",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapAppendTest, emptyArrays) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20}",
          "{}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[]",
          "[]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[]",
          "[]",
      }),
  });

  auto result = evaluate("map_append(c0, c1, c2)", data);

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20}",
      "{}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapAppendTest, nullValues) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[3, 4]",
      }),
      makeNullableArrayVector<int32_t>({
          {std::nullopt, 40},
      }),
  });

  auto result = evaluate("map_append(c0, c1, c2)", data);

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:null, 4:40}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapAppendTest, duplicateKeyInNewKeys) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[3, 3]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[30, 40]",
      }),
  });

  VELOX_ASSERT_THROW(
      evaluate("map_append(c0, c1, c2)", data), "Duplicate key in keys array");
}

TEST_F(MapAppendTest, keyAlreadyExistsInMap) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[2, 3]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[30, 40]",
      }),
  });

  VELOX_ASSERT_THROW(
      evaluate("map_append(c0, c1, c2)", data),
      "Key already exists in the map");
}

TEST_F(MapAppendTest, mismatchedArrayLengths) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[3, 4]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[30]",
      }),
  });

  VELOX_ASSERT_THROW(
      evaluate("map_append(c0, c1, c2)", data),
      "Keys and values arrays must have the same length");
}

TEST_F(MapAppendTest, floatKeys) {
  auto data = makeRowVector({
      makeMapVectorFromJson<float, int32_t>({
          "{1.5:10, 2.5:20}",
      }),
      makeArrayVectorFromJson<float>({
          "[3.5, 4.5]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[30, 40]",
      }),
  });

  auto result = evaluate("map_append(c0, c1, c2)", data);

  auto expected = makeMapVectorFromJson<float, int32_t>({
      "{1.5:10, 2.5:20, 3.5:30, 4.5:40}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapAppendTest, booleanKeys) {
  auto data = makeRowVector({
      makeMapVector<bool, int32_t>({{{true, 10}}}),
      makeArrayVector<bool>({{false}}),
      makeArrayVector<int32_t>({{20}}),
  });

  auto result = evaluate("map_append(c0, c1, c2)", data);

  auto expected = makeMapVector<bool, int32_t>({{{true, 10}, {false, 20}}});

  assertEqualVectors(expected, result);
}

TEST_F(MapAppendTest, nullKeysIgnored) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20}",
      }),
      makeNullableArrayVector<int64_t>({
          {3, std::nullopt, 4},
      }),
      makeArrayVectorFromJson<int32_t>({
          "[30, 40, 50]",
      }),
  });

  auto result = evaluate("map_append(c0, c1, c2)", data);

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:50}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapAppendTest, emptyMapWithNewEntries) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[1, 2, 3]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[10, 20, 30]",
      }),
  });

  auto result = evaluate("map_append(c0, c1, c2)", data);

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapAppendTest, complexValueTypes) {
  // Create value arrays for the map
  auto inputMapValues = makeArrayVectorFromJson<int32_t>({
      "[1, 2]",
      "[3, 4]",
  });

  // Create a map with keys [1, 2] and values [[1,2], [3,4]]
  auto inputMap =
      makeMapVector({0}, makeFlatVector<int64_t>({1, 2}), inputMapValues);

  // Keys to append: [3, 4] (single row with 2 keys)
  auto newKeys = makeArrayVectorFromJson<int64_t>({
      "[3, 4]",
  });

  // Values to append: [[5,6], [7,8]] (single row with 2 array values)
  auto newValues = makeNestedArrayVectorFromJson<int32_t>({
      "[[5, 6], [7, 8]]",
  });

  auto data = makeRowVector({inputMap, newKeys, newValues});

  auto result = evaluate("map_append(c0, c1, c2)", data);

  auto expectedValues = makeArrayVectorFromJson<int32_t>({
      "[1, 2]",
      "[3, 4]",
      "[5, 6]",
      "[7, 8]",
  });

  auto expected =
      makeMapVector({0}, makeFlatVector<int64_t>({1, 2, 3, 4}), expectedValues);

  assertEqualVectors(expected, result);
}

TEST_F(MapAppendTest, largeMap) {
  // Create a map with 10 entries and append 3 more
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30, 4:40, 5:50, 6:60, 7:70, 8:80, 9:90, 10:100}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[11, 12, 13]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[110, 120, 130]",
      }),
  });

  auto result = evaluate("map_append(c0, c1, c2)", data);

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:40, 5:50, 6:60, 7:70, 8:80, 9:90, 10:100, 11:110, 12:120, 13:130}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapAppendTest, timestampKeys) {
  auto data = makeRowVector({
      makeMapVector<Timestamp, int32_t>(
          {{{Timestamp(1, 0), 10}, {Timestamp(2, 0), 20}}}),
      makeArrayVector<Timestamp>({{Timestamp(3, 0), Timestamp(4, 0)}}),
      makeArrayVector<int32_t>({{30, 40}}),
  });

  auto result = evaluate("map_append(c0, c1, c2)", data);

  auto expected = makeMapVector<Timestamp, int32_t>(
      {{{Timestamp(1, 0), 10},
        {Timestamp(2, 0), 20},
        {Timestamp(3, 0), 30},
        {Timestamp(4, 0), 40}}});

  assertEqualVectors(expected, result);
}

TEST_F(MapAppendTest, preserveMapOrder) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{5:50, 3:30, 1:10}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[2, 4]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[20, 40]",
      }),
  });

  auto result = evaluate("map_append(c0, c1, c2)", data);

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{5:50, 3:30, 1:10, 2:20, 4:40}",
  });

  assertEqualVectors(expected, result);
}

} // namespace
} // namespace facebook::velox::functions
