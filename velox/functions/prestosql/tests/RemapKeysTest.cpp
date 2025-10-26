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

class RemapKeysTest : public test::FunctionBaseTest {
 protected:
  void testRemapKeys(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }
};

TEST_F(RemapKeysTest, basicIntegerKeys) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
      {{4, 40}, {5, 50}},
  });

  auto oldKeys = makeArrayVector<int32_t>({
      {1, 2},
      {4},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {100, 200},
      {400},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{100, 10}, {200, 20}, {3, 30}},
      {{400, 40}, {5, 50}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, stringKeys) {
  auto inputMap = makeMapVector<StringView, int32_t>({
      {{"a", 1}, {"b", 2}, {"c", 3}},
      {{"x", 10}, {"y", 20}},
  });

  auto oldKeys = makeArrayVector<StringView>({
      {"a", "c"},
      {"x"},
  });

  auto newKeys = makeArrayVector<StringView>({
      {"alpha", "charlie"},
      {"xray"},
  });

  auto expected = makeMapVector<StringView, int32_t>({
      {{"alpha", 1}, {"b", 2}, {"charlie", 3}},
      {{"xray", 10}, {"y", 20}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, emptyMap) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {},
  });

  auto oldKeys = makeArrayVector<int32_t>({
      {1, 2},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {10, 20},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, emptyOldKeys) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}},
  });

  auto oldKeys = makeArrayVector<int32_t>({
      {},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, noMatchingKeys) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });

  auto oldKeys = makeArrayVector<int32_t>({
      {4, 5, 6},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {40, 50, 60},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, mismatchedArrayLengths) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
      {{4, 40}, {5, 50}},
  });

  auto oldKeys = makeArrayVector<int32_t>({
      {1, 2, 3},
      {4, 5},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {100, 200},
      {400, 500, 600},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{100, 10}, {200, 20}, {3, 30}},
      {{400, 40}, {500, 50}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, nullValuesInMap) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, std::nullopt}, {3, 30}},
  });

  auto oldKeys = makeArrayVector<int32_t>({
      {1, 2},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {100, 200},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{100, 10}, {200, std::nullopt}, {3, 30}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, nullKeysInArrays) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });

  auto oldKeys = makeNullableArrayVector<int32_t>({
      {1, std::nullopt, 3},
  });

  auto newKeys = makeNullableArrayVector<int32_t>({
      {100, std::nullopt, 300},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{100, 10}, {2, 20}, {300, 30}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, duplicateOldKeys) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });

  auto oldKeys = makeArrayVector<int32_t>({
      {1, 1, 2},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {100, 101, 200},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{101, 10}, {200, 20}, {3, 30}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, floatKeys) {
  auto inputMap = makeMapVector<float, int32_t>({
      {{1.5f, 10}, {2.5f, 20}, {3.5f, 30}},
  });

  auto oldKeys = makeArrayVector<float>({
      {1.5f, 3.5f},
  });

  auto newKeys = makeArrayVector<float>({
      {10.5f, 30.5f},
  });

  auto expected = makeMapVector<float, int32_t>({
      {{10.5f, 10}, {2.5f, 20}, {30.5f, 30}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, booleanKeys) {
  auto inputMap = makeMapVector<bool, int32_t>({
      {{true, 10}, {false, 20}},
  });

  auto oldKeys = makeArrayVector<bool>({
      {true},
  });

  auto newKeys = makeArrayVector<bool>({
      {false},
  });

  auto expected = makeMapVector<bool, int32_t>({
      {{false, 10}, {false, 20}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, complexValues) {
  // Test with map of integer keys to array values
  auto keys = makeFlatVector<int32_t>({0, 1, 2});
  auto innerArrays = makeArrayVector<int32_t>({
      {1, 2},
      {3, 4, 5},
      {6},
  });

  auto inputMap = makeMapVector({0}, keys, innerArrays);

  auto oldKeys = makeArrayVector<int32_t>({
      {0, 1},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {10, 11},
  });

  auto expectedKeys = makeFlatVector<int32_t>({10, 11, 2});
  auto expectedInnerArrays = makeArrayVector<int32_t>({
      {1, 2},
      {3, 4, 5},
      {6},
  });

  auto expected = makeMapVector({0}, expectedKeys, expectedInnerArrays);

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, allKeysRemapped) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });

  auto oldKeys = makeArrayVector<int32_t>({
      {1, 2, 3},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {100, 200, 300},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{100, 10}, {200, 20}, {300, 30}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, partialRemapping) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}, {4, 40}},
  });

  auto oldKeys = makeArrayVector<int32_t>({
      {2, 4},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {200, 400},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {200, 20}, {3, 30}, {400, 40}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, largeMap) {
  // Create a map with 10 entries for testing
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

  auto oldKeys = makeArrayVector<int32_t>({
      {0, 5, 9},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {1000, 5000, 9000},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{1000, 0},
       {1, 10},
       {2, 20},
       {3, 30},
       {4, 40},
       {5000, 50},
       {6, 60},
       {7, 70},
       {8, 80},
       {9000, 90}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, timestampKeys) {
  auto inputMap = makeMapVector<Timestamp, int32_t>({
      {{Timestamp(1, 0), 10}, {Timestamp(2, 0), 20}},
  });

  auto oldKeys = makeArrayVector<Timestamp>({
      {Timestamp(1, 0)},
  });

  auto newKeys = makeArrayVector<Timestamp>({
      {Timestamp(100, 0)},
  });

  auto expected = makeMapVector<Timestamp, int32_t>({
      {{Timestamp(100, 0), 10}, {Timestamp(2, 0), 20}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, oldKeysLongerThanNewKeys) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}},
  });

  auto oldKeys = makeArrayVector<int32_t>({
      {1, 2, 3, 4, 5},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {100, 200},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{100, 10}, {200, 20}, {3, 30}, {4, 40}, {5, 50}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, newKeysLongerThanOldKeys) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });

  auto oldKeys = makeArrayVector<int32_t>({
      {1, 2},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {100, 200, 300, 400, 500},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{100, 10}, {200, 20}, {3, 30}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, significantSizeDifference) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}, {6, 60}, {7, 70}},
  });

  auto oldKeys = makeArrayVector<int32_t>({
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
  });

  auto newKeys = makeArrayVector<int32_t>({
      {100},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{100, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}, {6, 60}, {7, 70}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

TEST_F(RemapKeysTest, stringKeysWithDifferentSizes) {
  auto inputMap = makeMapVector<StringView, int32_t>({
      {{"apple", 1}, {"banana", 2}, {"cherry", 3}, {"date", 4}},
  });

  auto oldKeys = makeArrayVector<StringView>({
      {"apple", "banana", "cherry", "date", "elderberry", "fig"},
  });

  auto newKeys = makeArrayVector<StringView>({
      {"APPLE", "BANANA"},
  });

  auto expected = makeMapVector<StringView, int32_t>({
      {{"APPLE", 1}, {"BANANA", 2}, {"cherry", 3}, {"date", 4}},
  });

  testRemapKeys(
      "remap_keys(c0, c1, c2)", {inputMap, oldKeys, newKeys}, expected);
}

} // namespace
} // namespace facebook::velox::functions
