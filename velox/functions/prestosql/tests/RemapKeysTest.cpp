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

#include <random>
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

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

  // Helper to build transform_keys CASE expression equivalent to remap_keys
  // for integer keys. Returns a transform_keys expression that maps each
  // oldKey to its corresponding newKey.
  // Note: Integer literals are cast to INTEGER to match the key type (int32_t).
  template <typename T>
  std::string buildTransformKeysCaseExpression(
      const std::vector<T>& oldKeys,
      const std::vector<T>& newKeys) {
    if (oldKeys.empty() || newKeys.empty()) {
      return "transform_keys(c0, (k, v) -> k)";
    }

    size_t mappingSize = std::min(oldKeys.size(), newKeys.size());
    std::stringstream ss;
    ss << "transform_keys(c0, (k, v) -> CASE ";
    for (size_t i = 0; i < mappingSize; ++i) {
      ss << "WHEN k = CAST(" << oldKeys[i] << " AS INTEGER) THEN CAST("
         << newKeys[i] << " AS INTEGER) ";
    }
    ss << "ELSE k END)";
    return ss.str();
  }

  // Specialization for string keys
  std::string buildTransformKeysCaseExpressionString(
      const std::vector<std::string>& oldKeys,
      const std::vector<std::string>& newKeys) {
    if (oldKeys.empty() || newKeys.empty()) {
      return "transform_keys(c0, (k, v) -> k)";
    }

    size_t mappingSize = std::min(oldKeys.size(), newKeys.size());
    std::stringstream ss;
    ss << "transform_keys(c0, (k, v) -> CASE ";
    for (size_t i = 0; i < mappingSize; ++i) {
      ss << "WHEN k = '" << oldKeys[i] << "' THEN '" << newKeys[i] << "' ";
    }
    ss << "ELSE k END)";
    return ss.str();
  }

  // Generates random integer keys within a specified range
  template <typename T>
  std::vector<T> generateRandomIntegerKeys(
      std::mt19937& rng,
      size_t count,
      T minVal,
      T maxVal) {
    std::uniform_int_distribution<T> dist(minVal, maxVal);
    std::vector<T> keys;
    keys.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      keys.push_back(dist(rng));
    }
    return keys;
  }

  // Generates random string keys
  std::vector<std::string> generateRandomStringKeys(
      std::mt19937& rng,
      size_t count) {
    static const std::string chars =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::uniform_int_distribution<size_t> lenDist(1, 10);
    std::uniform_int_distribution<size_t> charDist(0, chars.size() - 1);

    std::vector<std::string> keys;
    keys.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      size_t len = lenDist(rng);
      std::string key;
      key.reserve(len);
      for (size_t j = 0; j < len; ++j) {
        key += chars[charDist(rng)];
      }
      keys.push_back(key);
    }
    return keys;
  }

  // Helper to check if applying the key mapping would create duplicate keys
  template <typename K>
  bool wouldCreateDuplicateKeys(
      const std::vector<K>& mapKeys,
      const std::vector<K>& oldKeys,
      const std::vector<K>& newKeys) {
    size_t mappingSize = std::min(oldKeys.size(), newKeys.size());
    std::unordered_set<K> resultKeys;

    for (const auto& key : mapKeys) {
      K resultKey = key;
      for (size_t i = 0; i < mappingSize; ++i) {
        if (key == oldKeys[i]) {
          resultKey = newKeys[i];
          break;
        }
      }
      if (resultKeys.count(resultKey) > 0) {
        return true;
      }
      resultKeys.insert(resultKey);
    }
    return false;
  }

  // Helper to wrap map data for makeMapVector (expects vector<vector<pair>>)
  template <typename K, typename V>
  std::vector<std::vector<std::pair<K, std::optional<V>>>> wrapMapData(
      const std::vector<std::pair<K, std::optional<V>>>& data) {
    return std::vector<std::vector<std::pair<K, std::optional<V>>>>{data};
  }

  // Helper to wrap array data for makeArrayVector (expects vector<vector<T>>)
  template <typename T>
  std::vector<std::vector<T>> wrapArrayData(const std::vector<T>& data) {
    return std::vector<std::vector<T>>{data};
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

// ============================================================================
// CUSTOM FUZZER TESTS
// These tests compare remap_keys() against the equivalent transform_keys()
// with CASE expression to ensure behavioral consistency.
// ============================================================================

// Simplified fuzzer test that uses the lambda-based makeMapVector API
TEST_F(RemapKeysTest, fuzzerIntegerKeysSimplified) {
  constexpr int kIterations = 1000;

  std::mt19937 rng(42);
  std::uniform_int_distribution<int32_t> keyDist(-100, 100);
  std::uniform_int_distribution<int32_t> valueDist(-1000, 1000);
  std::uniform_int_distribution<int> sizeDist(1, 10);

  for (int iter = 0; iter < kIterations; ++iter) {
    int mapSize = sizeDist(rng);

    // Generate unique keys for the map
    std::vector<int32_t> keys;
    std::unordered_set<int32_t> usedKeys;
    while (static_cast<int>(keys.size()) < mapSize) {
      int32_t key = keyDist(rng);
      if (usedKeys.find(key) == usedKeys.end()) {
        usedKeys.insert(key);
        keys.push_back(key);
      }
    }

    std::vector<int32_t> values;
    for (int i = 0; i < mapSize; ++i) {
      values.push_back(valueDist(rng));
    }

    // Generate oldKeys/newKeys for mapping
    int mappingSize = std::min(3, mapSize);
    std::vector<int32_t> oldKeysVec, newKeysVec;
    std::unordered_set<int32_t> mappedTo;

    for (int i = 0; i < mappingSize; ++i) {
      oldKeysVec.push_back(keys[i]);
      int32_t newKey = keyDist(rng);
      // Ensure new key doesn't create duplicates
      while (usedKeys.count(newKey) > 0 && newKey != keys[i]) {
        newKey = keyDist(rng);
      }
      newKeysVec.push_back(newKey);
      if (newKey != keys[i]) {
        usedKeys.insert(newKey);
      }
    }

    // Create vectors using lambda-based API
    auto inputMap = makeMapVector<int32_t, int32_t>(
        1, // single row
        [&](vector_size_t) { return static_cast<vector_size_t>(mapSize); },
        [&](vector_size_t idx) { return keys[idx]; },
        [&](vector_size_t idx) { return values[idx]; });

    auto oldKeys = makeArrayVector<int32_t>(
        1,
        [&](vector_size_t) { return static_cast<vector_size_t>(mappingSize); },
        [&](vector_size_t idx) { return oldKeysVec[idx]; });

    auto newKeys = makeArrayVector<int32_t>(
        1,
        [&](vector_size_t) { return static_cast<vector_size_t>(mappingSize); },
        [&](vector_size_t idx) { return newKeysVec[idx]; });

    // Build equivalent transform_keys expression
    std::string transformExpr =
        buildTransformKeysCaseExpression(oldKeysVec, newKeysVec);

    // Evaluate both expressions
    auto remapResult = evaluate(
        "remap_keys(c0, c1, c2)", makeRowVector({inputMap, oldKeys, newKeys}));
    auto transformResult = evaluate(transformExpr, makeRowVector({inputMap}));

    // Compare results
    assertEqualVectors(transformResult, remapResult);
  }
}

// Fuzzer test for string keys using lambda API
TEST_F(RemapKeysTest, fuzzerStringKeysSimplified) {
  constexpr int kIterations = 500;

  std::mt19937 rng(123);
  std::uniform_int_distribution<int32_t> valueDist(-1000, 1000);
  std::uniform_int_distribution<int> sizeDist(1, 8);

  for (int iter = 0; iter < kIterations; ++iter) {
    int mapSize = sizeDist(rng);

    // Generate unique string keys
    std::vector<std::string> keys = generateRandomStringKeys(rng, mapSize * 2);
    keys.resize(mapSize);
    std::vector<int32_t> values;
    for (int i = 0; i < mapSize; ++i) {
      values.push_back(valueDist(rng));
    }

    // Generate mapping
    int mappingSize = std::min(2, mapSize);
    std::vector<std::string> oldKeysVec, newKeysVec;

    for (int i = 0; i < mappingSize; ++i) {
      oldKeysVec.push_back(keys[i]);
      auto newKeyVec = generateRandomStringKeys(rng, 1);
      newKeysVec.push_back(newKeyVec[0]);
    }

    // Create vectors using lambda-based API
    auto inputMap = makeMapVector<StringView, int32_t>(
        1,
        [&](vector_size_t) { return static_cast<vector_size_t>(mapSize); },
        [&](vector_size_t idx) { return StringView(keys[idx]); },
        [&](vector_size_t idx) { return values[idx]; });

    auto oldKeys = makeArrayVector<StringView>(
        1,
        [&](vector_size_t) { return static_cast<vector_size_t>(mappingSize); },
        [&](vector_size_t idx) { return StringView(oldKeysVec[idx]); });

    auto newKeys = makeArrayVector<StringView>(
        1,
        [&](vector_size_t) { return static_cast<vector_size_t>(mappingSize); },
        [&](vector_size_t idx) { return StringView(newKeysVec[idx]); });

    // Build equivalent transform_keys expression
    std::string transformExpr =
        buildTransformKeysCaseExpressionString(oldKeysVec, newKeysVec);

    // Evaluate both expressions
    try {
      auto remapResult = evaluate(
          "remap_keys(c0, c1, c2)",
          makeRowVector({inputMap, oldKeys, newKeys}));
      auto transformResult = evaluate(transformExpr, makeRowVector({inputMap}));
      assertEqualVectors(transformResult, remapResult);
    } catch (const VeloxUserError&) {
      // Duplicate key errors are expected when mapping creates duplicates
    }
  }
}

// Edge case fuzzer test using lambda-based API
TEST_F(RemapKeysTest, fuzzerEdgeCasesSimplified) {
  // Test 1: Empty map with non-empty mapping
  {
    // Empty map
    auto inputMap = makeMapVector<int32_t, int32_t>(
        1,
        [](vector_size_t) { return 0; }, // empty map
        [](vector_size_t) { return 0; },
        [](vector_size_t) { return 0; });

    // oldKeys = [1, 2], newKeys = [10, 20]
    std::vector<int32_t> oldKeysData = {1, 2};
    std::vector<int32_t> newKeysData = {10, 20};
    auto oldKeys = makeArrayVector<int32_t>(
        1,
        [](vector_size_t) { return 2; },
        [&](vector_size_t idx) { return oldKeysData[idx]; });
    auto newKeys = makeArrayVector<int32_t>(
        1,
        [](vector_size_t) { return 2; },
        [&](vector_size_t idx) { return newKeysData[idx]; });

    auto remapResult = evaluate(
        "remap_keys(c0, c1, c2)", makeRowVector({inputMap, oldKeys, newKeys}));
    auto transformResult =
        evaluate("transform_keys(c0, (k, v) -> k)", makeRowVector({inputMap}));
    assertEqualVectors(transformResult, remapResult);
  }

  // Test 2: Non-empty map with empty mapping
  {
    // inputMap = {1: 10, 2: 20}
    std::vector<int32_t> mapKeys = {1, 2};
    std::vector<int32_t> mapVals = {10, 20};
    auto inputMap = makeMapVector<int32_t, int32_t>(
        1,
        [](vector_size_t) { return 2; },
        [&](vector_size_t idx) { return mapKeys[idx]; },
        [&](vector_size_t idx) { return mapVals[idx]; });

    // Empty arrays
    auto oldKeys = makeArrayVector<int32_t>(
        1, [](vector_size_t) { return 0; }, [](vector_size_t) { return 0; });
    auto newKeys = makeArrayVector<int32_t>(
        1, [](vector_size_t) { return 0; }, [](vector_size_t) { return 0; });

    auto remapResult = evaluate(
        "remap_keys(c0, c1, c2)", makeRowVector({inputMap, oldKeys, newKeys}));
    auto transformResult =
        evaluate("transform_keys(c0, (k, v) -> k)", makeRowVector({inputMap}));
    assertEqualVectors(transformResult, remapResult);
  }

  // Test 3: All keys remapped
  {
    // inputMap = {1: 10, 2: 20}
    std::vector<int32_t> mapKeys = {1, 2};
    std::vector<int32_t> mapVals = {10, 20};
    auto inputMap = makeMapVector<int32_t, int32_t>(
        1,
        [](vector_size_t) { return 2; },
        [&](vector_size_t idx) { return mapKeys[idx]; },
        [&](vector_size_t idx) { return mapVals[idx]; });

    // oldKeys = [1, 2], newKeys = [100, 200]
    std::vector<int32_t> oldKeysData = {1, 2};
    std::vector<int32_t> newKeysData = {100, 200};
    auto oldKeys = makeArrayVector<int32_t>(
        1,
        [](vector_size_t) { return 2; },
        [&](vector_size_t idx) { return oldKeysData[idx]; });
    auto newKeys = makeArrayVector<int32_t>(
        1,
        [](vector_size_t) { return 2; },
        [&](vector_size_t idx) { return newKeysData[idx]; });

    auto remapResult = evaluate(
        "remap_keys(c0, c1, c2)", makeRowVector({inputMap, oldKeys, newKeys}));
    auto transformResult = evaluate(
        "transform_keys(c0, (k, v) -> CASE WHEN k = CAST(1 AS INTEGER) THEN CAST(100 AS INTEGER) WHEN k = CAST(2 AS INTEGER) THEN CAST(200 AS INTEGER) ELSE k END)",
        makeRowVector({inputMap}));
    assertEqualVectors(transformResult, remapResult);
  }

  // Test 4: Partial remapping
  {
    // inputMap = {1: 10, 2: 20, 3: 30}
    std::vector<int32_t> mapKeys = {1, 2, 3};
    std::vector<int32_t> mapVals = {10, 20, 30};
    auto inputMap = makeMapVector<int32_t, int32_t>(
        1,
        [](vector_size_t) { return 3; },
        [&](vector_size_t idx) { return mapKeys[idx]; },
        [&](vector_size_t idx) { return mapVals[idx]; });

    // oldKeys = [1], newKeys = [100]
    std::vector<int32_t> oldKeysData = {1};
    std::vector<int32_t> newKeysData = {100};
    auto oldKeys = makeArrayVector<int32_t>(
        1,
        [](vector_size_t) { return 1; },
        [&](vector_size_t idx) { return oldKeysData[idx]; });
    auto newKeys = makeArrayVector<int32_t>(
        1,
        [](vector_size_t) { return 1; },
        [&](vector_size_t idx) { return newKeysData[idx]; });

    auto remapResult = evaluate(
        "remap_keys(c0, c1, c2)", makeRowVector({inputMap, oldKeys, newKeys}));
    auto transformResult = evaluate(
        "transform_keys(c0, (k, v) -> CASE WHEN k = CAST(1 AS INTEGER) THEN CAST(100 AS INTEGER) ELSE k END)",
        makeRowVector({inputMap}));
    assertEqualVectors(transformResult, remapResult);
  }

  // Test 5: Mismatched array lengths (uses min)
  {
    // inputMap = {1: 10, 2: 20}
    std::vector<int32_t> mapKeys = {1, 2};
    std::vector<int32_t> mapVals = {10, 20};
    auto inputMap = makeMapVector<int32_t, int32_t>(
        1,
        [](vector_size_t) { return 2; },
        [&](vector_size_t idx) { return mapKeys[idx]; },
        [&](vector_size_t idx) { return mapVals[idx]; });

    // oldKeys = [1, 2, 3, 4, 5], newKeys = [100, 200]
    std::vector<int32_t> oldKeysData = {1, 2, 3, 4, 5};
    std::vector<int32_t> newKeysData = {100, 200};
    auto oldKeys = makeArrayVector<int32_t>(
        1,
        [](vector_size_t) { return 5; },
        [&](vector_size_t idx) { return oldKeysData[idx]; });
    auto newKeys = makeArrayVector<int32_t>(
        1,
        [](vector_size_t) { return 2; },
        [&](vector_size_t idx) { return newKeysData[idx]; });

    auto remapResult = evaluate(
        "remap_keys(c0, c1, c2)", makeRowVector({inputMap, oldKeys, newKeys}));
    auto transformResult = evaluate(
        "transform_keys(c0, (k, v) -> CASE WHEN k = CAST(1 AS INTEGER) THEN CAST(100 AS INTEGER) WHEN k = CAST(2 AS INTEGER) THEN CAST(200 AS INTEGER) ELSE k END)",
        makeRowVector({inputMap}));
    assertEqualVectors(transformResult, remapResult);
  }
}

// Fuzzer test using VectorFuzzer for stress testing
TEST_F(RemapKeysTest, fuzzerWithVectorFuzzerSimplified) {
  constexpr int kIterations = 500;
  constexpr vector_size_t kBatchSize = 10;

  VectorFuzzer::Options options;
  options.vectorSize = kBatchSize;
  options.nullRatio = 0.0;
  options.containerLength = 5;
  options.containerVariableLength = true;

  VectorFuzzer fuzzer(options, pool());

  for (int iter = 0; iter < kIterations; ++iter) {
    auto mapType = MAP(INTEGER(), INTEGER());
    auto inputMap = fuzzer.fuzzFlat(mapType);

    // Use fixed mapping that won't cause duplicates
    auto oldKeys = makeArrayVector<int32_t>(
        kBatchSize,
        [](vector_size_t) { return 2; },
        [](vector_size_t row, vector_size_t idx) {
          return static_cast<int32_t>(row * 1000 + idx);
        });

    auto newKeys = makeArrayVector<int32_t>(
        kBatchSize,
        [](vector_size_t) { return 2; },
        [](vector_size_t row, vector_size_t idx) {
          return static_cast<int32_t>(row * 1000 + idx + 10000);
        });

    // Verify remap_keys doesn't crash
    try {
      auto result = evaluate(
          "remap_keys(c0, c1, c2)",
          makeRowVector({inputMap, oldKeys, newKeys}));
      ASSERT_NE(result, nullptr);
    } catch (const VeloxUserError&) {
      // Expected: duplicate keys, null keys
    }
  }
}

} // namespace
} // namespace facebook::velox::functions
