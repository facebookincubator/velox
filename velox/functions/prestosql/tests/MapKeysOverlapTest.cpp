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
#include "velox/vector/fuzzer/VectorFuzzer.h"

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

  // Helper to compare map_keys_overlap against equivalent expression.
  // The equivalent expression is:
  // cardinality(array_intersect(map_keys(c0), filter(c1, x -> x IS NOT NULL)))
  // > 0
  void testMapKeysOverlapAgainstEquivalent(const RowVectorPtr& data) {
    auto udfResult = evaluate("map_keys_overlap(c0, c1)", data);

    // Equivalent expression using existing UDFs:
    // 1. Extract map keys: map_keys(c0)
    // 2. Filter out nulls from the array: filter(c1, x -> x IS NOT NULL)
    // 3. Find intersection: array_intersect(...)
    // 4. Check if non-empty: cardinality(...) > 0
    auto equivalentResult = evaluate(
        "cardinality(array_intersect(map_keys(c0), filter(c1, x -> x IS NOT NULL))) > 0",
        data);

    assertEqualVectors(equivalentResult, udfResult);
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

// ============================================================================
// Custom Fuzzer Tests
// These tests compare map_keys_overlap against an equivalent expression:
// cardinality(array_intersect(map_keys(c0), filter(c1, x -> x IS NOT NULL))) >
// 0
// ============================================================================

TEST_F(MapKeysOverlapTest, fuzzIntegerKeys) {
  // Create base data with representative edge cases
  auto baseMap = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}, {3, 30}},
      {{4, 40}, {5, 50}},
      {{6, 60}, {7, 70}, {8, 80}},
      {}, // empty map
      {{100, 1000}},
  });

  auto baseKeys = makeArrayVector<int64_t>({
      {1, 5, 10},
      {4, 6},
      {9, 10, 11},
      {1, 2, 3},
      {100},
  });

  VectorFuzzer::Options options;
  options.vectorSize = 1024;
  options.nullRatio = 0.0; // Start without nulls

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 10; ++i) {
    auto data = makeRowVector({
        fuzzer.fuzzDictionary(baseMap, options.vectorSize),
        fuzzer.fuzzDictionary(baseKeys, options.vectorSize),
    });

    testMapKeysOverlapAgainstEquivalent(data);

    // Also test with flattened data
    auto flatData = flatten<RowVector>(data);
    testMapKeysOverlapAgainstEquivalent(flatData);
  }
}

TEST_F(MapKeysOverlapTest, fuzzIntegerKeysWithNulls) {
  // Create base data including null elements in the array
  auto baseMap = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}, {3, 30}},
      {{4, 40}, {5, 50}},
      {}, // empty map
      {{100, 1000}, {200, 2000}},
  });

  std::vector<std::vector<std::optional<int64_t>>> keyData = {
      {1, std::nullopt, 5},
      {std::nullopt, 4},
      {std::nullopt, std::nullopt},
      {50, 100, std::nullopt},
  };
  auto baseKeys = makeNullableArrayVector<int64_t>(keyData);

  VectorFuzzer::Options options;
  options.vectorSize = 1024;
  options.nullRatio = 0.1;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 10; ++i) {
    auto data = makeRowVector({
        fuzzer.fuzzDictionary(baseMap, options.vectorSize),
        fuzzer.fuzzDictionary(baseKeys, options.vectorSize),
    });

    testMapKeysOverlapAgainstEquivalent(data);
  }
}

TEST_F(MapKeysOverlapTest, fuzzVarcharKeys) {
  // Create base data with string keys
  auto baseMap = makeMapVector<StringView, int64_t>({
      {{"apple", 1}, {"banana", 2}, {"cherry", 3}},
      {{"dog", 10}, {"elephant", 20}},
      {}, // empty map
      {{"xyz", 100}},
      {{"a", 1}, {"b", 2}, {"c", 3}, {"d", 4}, {"e", 5}},
  });

  auto baseKeys = makeArrayVector<StringView>({
      {"apple", "date"},
      {"cat", "dog"},
      {"foo", "bar"},
      {"xyz"},
      {"c", "f", "g"},
  });

  VectorFuzzer::Options options;
  options.vectorSize = 1024;
  options.nullRatio = 0.0;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 10; ++i) {
    auto data = makeRowVector({
        fuzzer.fuzzDictionary(baseMap, options.vectorSize),
        fuzzer.fuzzDictionary(baseKeys, options.vectorSize),
    });

    testMapKeysOverlapAgainstEquivalent(data);

    // Also test with flattened data
    auto flatData = flatten<RowVector>(data);
    testMapKeysOverlapAgainstEquivalent(flatData);
  }
}

TEST_F(MapKeysOverlapTest, fuzzVarcharKeysWithNulls) {
  // Create base data with string keys and nullable arrays
  auto baseMap = makeMapVector<StringView, int64_t>({
      {{"apple", 1}, {"banana", 2}},
      {{"cat", 10}},
      {},
      {{"xyz", 100}, {"abc", 200}},
  });

  std::vector<std::vector<std::optional<StringView>>> keyData = {
      {StringView("apple"), std::nullopt, StringView("date")},
      {std::nullopt, StringView("cat")},
      {std::nullopt},
      {StringView("abc"), std::nullopt},
  };
  auto baseKeys = makeNullableArrayVector<StringView>(keyData);

  VectorFuzzer::Options options;
  options.vectorSize = 1024;
  options.nullRatio = 0.1;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 10; ++i) {
    auto data = makeRowVector({
        fuzzer.fuzzDictionary(baseMap, options.vectorSize),
        fuzzer.fuzzDictionary(baseKeys, options.vectorSize),
    });

    testMapKeysOverlapAgainstEquivalent(data);
  }
}

TEST_F(MapKeysOverlapTest, fuzzRandomGeneratedData) {
  // Use VectorFuzzer to generate completely random data
  VectorFuzzer::Options options;
  options.vectorSize = 100;
  options.containerLength = 5;
  options.containerVariableLength = true;
  options.nullRatio = 0.1;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 20; ++i) {
    // Generate random map and array with int64 keys.
    // Map keys must be non-null when normalizeMapKeys is true (default).
    auto randomMap = fuzzer.fuzzMap(
        fuzzer.fuzzFlatNotNull(BIGINT()),
        fuzzer.fuzzFlat(BIGINT()),
        options.vectorSize);
    auto randomArray =
        fuzzer.fuzzArray(fuzzer.fuzzFlat(BIGINT()), options.vectorSize);

    auto data = makeRowVector({randomMap, randomArray});

    testMapKeysOverlapAgainstEquivalent(data);
  }
}

TEST_F(MapKeysOverlapTest, fuzzMixedScenarios) {
  // Test with various combinations of empty/non-empty maps and arrays
  auto baseMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}},
      {{10, 100}, {20, 200}},
      {}, // empty
      {{1, 1}},
      {{100, 1000}, {200, 2000}, {300, 3000}},
  });

  auto baseKeys = makeArrayVector<int32_t>({
      {1, 2, 3, 100}, // partial match
      {30, 40, 50}, // no match
      {1, 2, 3}, // search in empty map
      {1}, // exact match
      {}, // empty search array
  });

  VectorFuzzer::Options options;
  options.vectorSize = 500;
  options.nullRatio = 0.0;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 10; ++i) {
    auto data = makeRowVector({
        fuzzer.fuzzDictionary(baseMap, options.vectorSize),
        fuzzer.fuzzDictionary(baseKeys, options.vectorSize),
    });

    testMapKeysOverlapAgainstEquivalent(data);

    // Test with lazy vector generation
    data = fuzzer.fuzzRowChildrenToLazy(data);
    testMapKeysOverlapAgainstEquivalent(data);
  }
}

TEST_F(MapKeysOverlapTest, fuzzLargeMapAndArray) {
  // Create larger maps and arrays to stress test the implementation
  // Use explicit initialization compatible with makeMapVector
  auto baseMap = makeMapVector<int64_t, int64_t>({
      {{0, 0},    {1, 10},   {2, 20},   {3, 30},   {4, 40},   {5, 50},
       {6, 60},   {7, 70},   {8, 80},   {9, 90},   {10, 100}, {11, 110},
       {12, 120}, {13, 130}, {14, 140}, {15, 150}, {16, 160}, {17, 170},
       {18, 180}, {19, 190}, {20, 200}},
      {}, // empty
      {{1000, 10000}},
      {{100, 1000}, {200, 2000}, {300, 3000}},
  });

  auto baseKeys = makeArrayVector<int64_t>({
      {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20}, // match some keys
      {100, 101, 102, 103, 104}, // no match in first map
      {1000, 2000}, // match in third map
      {100, 200, 400}, // partial match in fourth map
  });

  VectorFuzzer::Options options;
  options.vectorSize = 500;
  options.nullRatio = 0.0;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 5; ++i) {
    auto data = makeRowVector({
        fuzzer.fuzzDictionary(baseMap, options.vectorSize),
        fuzzer.fuzzDictionary(baseKeys, options.vectorSize),
    });

    testMapKeysOverlapAgainstEquivalent(data);
  }
}

TEST_F(MapKeysOverlapTest, fuzzBooleanKeys) {
  // Test with boolean keys
  auto baseMap = makeMapVector<bool, int32_t>({
      {{true, 10}, {false, 20}},
      {{true, 30}},
      {{false, 40}},
      {},
  });

  auto baseKeys = makeArrayVector<bool>({
      {true, false},
      {false},
      {true},
      {true, false},
  });

  VectorFuzzer::Options options;
  options.vectorSize = 256;
  options.nullRatio = 0.0;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 5; ++i) {
    auto data = makeRowVector({
        fuzzer.fuzzDictionary(baseMap, options.vectorSize),
        fuzzer.fuzzDictionary(baseKeys, options.vectorSize),
    });

    testMapKeysOverlapAgainstEquivalent(data);
  }
}

TEST_F(MapKeysOverlapTest, fuzzAllNullsInArray) {
  // Specific test case where all elements in array are null
  // map_keys_overlap should return false in this case
  auto baseMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
      {{4, 40}},
      {},
  });

  std::vector<std::vector<std::optional<int32_t>>> keyData = {
      {std::nullopt, std::nullopt, std::nullopt},
      {std::nullopt},
      {std::nullopt, std::nullopt},
  };
  auto baseKeys = makeNullableArrayVector<int32_t>(keyData);

  VectorFuzzer::Options options;
  options.vectorSize = 256;
  options.nullRatio = 0.0;

  VectorFuzzer fuzzer(options, pool());

  for (auto i = 0; i < 5; ++i) {
    auto data = makeRowVector({
        fuzzer.fuzzDictionary(baseMap, options.vectorSize),
        fuzzer.fuzzDictionary(baseKeys, options.vectorSize),
    });

    testMapKeysOverlapAgainstEquivalent(data);
  }
}

} // namespace
} // namespace facebook::velox::functions
