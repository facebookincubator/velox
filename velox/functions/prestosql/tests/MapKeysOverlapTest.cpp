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

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class MapKeysOverlapTest : public test::FunctionBaseTest {
 protected:
  void testMapKeysOverlap(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }
};

TEST_F(MapKeysOverlapTest, basicIntegerKeys) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
      {{4, 40}, {5, 50}},
      {{6, 60}, {7, 70}},
  });

  auto keys = makeArrayVector<int32_t>({
      {1, 5},
      {4},
      {8, 9},
  });

  auto expected = makeFlatVector<bool>({true, true, false});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, stringKeys) {
  auto inputMap = makeMapVector<StringView, int32_t>({
      {{"a", 1}, {"b", 2}, {"c", 3}},
      {{"x", 10}, {"y", 20}},
      {{"p", 100}, {"q", 200}},
  });

  auto keys = makeArrayVector<StringView>({
      {"a", "d"},
      {"z"},
      {"p"},
  });

  auto expected = makeFlatVector<bool>({true, false, true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, emptyMap) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {},
      {{1, 10}},
  });

  auto keys = makeArrayVector<int32_t>({
      {1, 2},
      {1},
  });

  auto expected = makeFlatVector<bool>({false, true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, emptyKeys) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}},
      {{3, 30}},
  });

  auto keys = makeArrayVector<int32_t>({
      {},
      {},
  });

  auto expected = makeFlatVector<bool>({false, false});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, noMatchingKeys) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
      {{4, 40}},
  });

  auto keys = makeArrayVector<int32_t>({
      {4, 5, 6},
      {1, 2, 3},
  });

  auto expected = makeFlatVector<bool>({false, false});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, allKeysMatch) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });

  auto keys = makeArrayVector<int32_t>({
      {1, 2, 3},
  });

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, partialMatch) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}, {4, 40}},
  });

  auto keys = makeArrayVector<int32_t>({
      {3, 5, 6, 7},
  });

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, nullValuesInMap) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, std::nullopt}, {3, 30}},
  });

  auto keys = makeArrayVector<int32_t>({
      {2},
  });

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, nullKeysInArray) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });

  std::vector<std::vector<std::optional<int32_t>>> keyData = {
      {std::nullopt, 4, 5},
  };
  auto keys = makeNullableArrayVector<int32_t>(keyData);

  auto expected = makeFlatVector<bool>({false});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, nullKeysMatchInArray) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });

  std::vector<std::vector<std::optional<int32_t>>> keyData = {
      {1, std::nullopt, 4},
  };
  auto keys = makeNullableArrayVector<int32_t>(keyData);

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, floatKeys) {
  auto inputMap = makeMapVector<float, int32_t>({
      {{1.5f, 10}, {2.5f, 20}, {3.5f, 30}},
  });

  auto keys = makeArrayVector<float>({
      {1.5f, 10.5f},
  });

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, doubleKeys) {
  auto inputMap = makeMapVector<double, int32_t>({
      {{1.5, 10}, {2.5, 20}},
  });

  auto keys = makeArrayVector<double>({
      {3.5, 4.5},
  });

  auto expected = makeFlatVector<bool>({false});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, booleanKeys) {
  auto inputMap = makeMapVector<bool, int32_t>({
      {{true, 10}, {false, 20}},
      {{true, 30}},
  });

  auto keys = makeArrayVector<bool>({
      {true},
      {false},
  });

  auto expected = makeFlatVector<bool>({true, false});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, timestampKeys) {
  auto inputMap = makeMapVector<Timestamp, int32_t>({
      {{Timestamp(1, 0), 10}, {Timestamp(2, 0), 20}},
  });

  auto keys = makeArrayVector<Timestamp>({
      {Timestamp(1, 0), Timestamp(100, 0)},
  });

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, complexValues) {
  auto mapKeys = makeFlatVector<int32_t>({0, 1, 2});
  auto innerArrays = makeArrayVector<int32_t>({
      {1, 2},
      {3, 4, 5},
      {6},
  });

  auto inputMap = makeMapVector({0}, mapKeys, innerArrays);

  auto keys = makeArrayVector<int32_t>({
      {0, 5},
  });

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, largeMap) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{0, 0},
       {1, 10},
       {2, 20},
       {3, 30},
       {4, 40},
       {5, 50},
       {6, 60},
       {7, 70},
       {8, 80},
       {9, 90}},
  });

  auto keys = makeArrayVector<int32_t>({
      {5, 15, 25},
  });

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, largeMapNoMatch) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{0, 0},
       {1, 10},
       {2, 20},
       {3, 30},
       {4, 40},
       {5, 50},
       {6, 60},
       {7, 70},
       {8, 80},
       {9, 90}},
  });

  auto keys = makeArrayVector<int32_t>({
      {15, 25, 35},
  });

  auto expected = makeFlatVector<bool>({false});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, multipleRows) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}},
      {{3, 30}, {4, 40}},
      {{5, 50}, {6, 60}},
      {{7, 70}},
  });

  auto keys = makeArrayVector<int32_t>({
      {2, 3},
      {1, 2},
      {5},
      {8, 9},
  });

  auto expected = makeFlatVector<bool>({true, false, true, false});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, longStringKeys) {
  auto inputMap = makeMapVector<StringView, int32_t>({
      {{"very_long_key_name_1", 1},
       {"very_long_key_name_2", 2},
       {"very_long_key_name_3", 3}},
  });

  auto keys = makeArrayVector<StringView>({
      {"very_long_key_name_2", "very_long_key_name_4"},
  });

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, singleKeyMatch) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}},
  });

  auto keys = makeArrayVector<int32_t>({
      {3},
  });

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, duplicateKeysInArray) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });

  auto keys = makeArrayVector<int32_t>({
      {1, 1, 1, 4, 5},
  });

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, varcharEmptyStrings) {
  auto inputMap = makeMapVector<StringView, int32_t>({
      {{"", 1}, {"a", 2}, {"b", 3}},
  });

  auto keys = makeArrayVector<StringView>({
      {"", "c"},
  });

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, allNullKeysInArray) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}},
  });

  std::vector<std::vector<std::optional<int32_t>>> keyData = {
      {std::nullopt, std::nullopt},
  };
  auto keys = makeNullableArrayVector<int32_t>(keyData);

  auto expected = makeFlatVector<bool>({false});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, int8Keys) {
  auto inputMap = makeMapVector<int8_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });

  auto keys = makeArrayVector<int8_t>({
      {2, 5},
  });

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, int16Keys) {
  auto inputMap = makeMapVector<int16_t, int32_t>({
      {{100, 10}, {200, 20}},
  });

  auto keys = makeArrayVector<int16_t>({
      {300, 400},
  });

  auto expected = makeFlatVector<bool>({false});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapKeysOverlapTest, int64Keys) {
  auto inputMap = makeMapVector<int64_t, int32_t>({
      {{1000000000, 10}, {2000000000, 20}},
  });

  auto keys = makeArrayVector<int64_t>({
      {1000000000},
  });

  auto expected = makeFlatVector<bool>({true});

  testMapKeysOverlap("map_keys_overlap(c0, c1)", {inputMap, keys}, expected);
}

} // namespace
} // namespace facebook::velox::functions
