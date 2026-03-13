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

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {

class MapZipWithTest : public SparkFunctionBaseTest {
 protected:
  void testMapZipWith(
      const std::string& expression,
      const RowVectorPtr& input,
      const VectorPtr& expected) {
    auto result = evaluateExpression(expression, input);
    assertEqualVectors(expected, result);
  }
};

// Type Coverage Tests

TEST_F(MapZipWithTest, stringKeys) {
  auto data = makeRowVector({
      makeMapVector<std::string, int64_t>({
          {{"a", 1}, {"b", 2}, {"c", 3}},
          {{"x", 10}, {"y", 20}},
          {{"foo", 100}},
      }),
      makeMapVector<std::string, int64_t>({
          {{"a", 10}, {"b", 20}, {"d", 40}},
          {{"y", 200}, {"z", 300}},
          {{"bar", 200}, {"foo", 150}},
      }),
  });

  auto expected = makeMapVector<std::string, int64_t>({
      {{"a", 11}, {"b", 22}, {"c", std::nullopt}, {"d", std::nullopt}},
      {{"x", std::nullopt}, {"y", 220}, {"z", std::nullopt}},
      {{"bar", std::nullopt}, {"foo", 250}},
  });

  testMapZipWith("map_zip_with(c0, c1, (k, v1, v2) -> v1 + v2)", data, expected);
}

TEST_F(MapZipWithTest, mixedNumericTypes) {
  auto data = makeRowVector({
      makeMapVector<int32_t, double>({
          {{1, 1.5}, {2, 2.5}, {3, 3.5}},
          {{10, 10.1}, {20, 20.2}},
      }),
      makeMapVector<int32_t, double>({
          {{1, 0.5}, {3, 1.5}, {4, 2.5}},
          {{10, 0.9}, {30, 30.3}},
      }),
  });

  auto expected = makeMapVector<int32_t, double>({
      {{1, 2.0}, {2, std::nullopt}, {3, 5.0}, {4, std::nullopt}},
      {{10, 11.0}, {20, std::nullopt}, {30, std::nullopt}},
  });

  testMapZipWith("map_zip_with(c0, c1, (k, v1, v2) -> v1 + v2)", data, expected);
}

TEST_F(MapZipWithTest, booleanValues) {
  auto data = makeRowVector({
      makeMapVector<int64_t, bool>({
          {{1, true}, {2, false}, {3, true}},
          {{1, false}, {2, false}},
      }),
      makeMapVector<int64_t, bool>({
          {{1, false}, {2, true}, {4, true}},
          {{1, true}, {3, true}},
      }),
  });

  // Use AND logic in lambda
  auto expected = makeMapVector<int64_t, bool>({
      {{1, false}, {2, false}, {3, std::nullopt}, {4, std::nullopt}},
      {{1, false}, {2, std::nullopt}, {3, std::nullopt}},
  });

  testMapZipWith("map_zip_with(c0, c1, (k, v1, v2) -> v1 AND v2)", data, expected);
}

TEST_F(MapZipWithTest, doubleTypes) {
  auto data = makeRowVector({
      makeMapVector<int64_t, double>({
          {{1, 1.1}, {2, 2.2}, {3, 3.3}},
          {{1, 10.5}, {2, 20.5}},
      }),
      makeMapVector<int64_t, double>({
          {{1, 0.9}, {2, 1.8}, {4, 4.4}},
          {{2, 5.5}, {3, 15.5}},
      }),
  });

  auto expected = makeMapVector<int64_t, double>({
      {{1, 2.0}, {2, 4.0}, {3, std::nullopt}, {4, std::nullopt}},
      {{1, std::nullopt}, {2, 26.0}, {3, std::nullopt}},
  });

  testMapZipWith("map_zip_with(c0, c1, (k, v1, v2) -> v1 + v2)", data, expected);
}

// Key Overlap Scenarios

TEST_F(MapZipWithTest, noCommonKeys) {
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}},
          {{100, 1000}},
      }),
      makeMapVector<int64_t, int64_t>({
          {{3, 30}, {4, 40}},
          {{200, 2000}, {300, 3000}},
      }),
  });

  // All results should be NULL since no keys match
  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, std::nullopt}, {2, std::nullopt}, {3, std::nullopt}, {4, std::nullopt}},
      {{100, std::nullopt}, {200, std::nullopt}, {300, std::nullopt}},
  });

  testMapZipWith("map_zip_with(c0, c1, (k, v1, v2) -> v1 + v2)", data, expected);
}

TEST_F(MapZipWithTest, allCommonKeys) {
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}, {3, 30}},
          {{5, 50}, {6, 60}},
      }),
      makeMapVector<int64_t, int64_t>({
          {{1, 100}, {2, 200}, {3, 300}},
          {{5, 500}, {6, 600}},
      }),
  });

  // All keys match, no NULLs expected
  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 110}, {2, 220}, {3, 330}},
      {{5, 550}, {6, 660}},
  });

  testMapZipWith("map_zip_with(c0, c1, (k, v1, v2) -> v1 + v2)", data, expected);
}

TEST_F(MapZipWithTest, partialOverlap) {
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}, {3, 30}, {4, 40}},
          {{10, 100}, {20, 200}, {30, 300}},
      }),
      makeMapVector<int64_t, int64_t>({
          {{2, 200}, {3, 300}, {5, 500}, {6, 600}},
          {{20, 2000}, {40, 4000}, {50, 5000}},
      }),
  });

  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, std::nullopt}, {2, 220}, {3, 330}, {4, std::nullopt}, {5, std::nullopt}, {6, std::nullopt}},
      {{10, std::nullopt}, {20, 2200}, {30, std::nullopt}, {40, std::nullopt}, {50, std::nullopt}},
  });

  testMapZipWith("map_zip_with(c0, c1, (k, v1, v2) -> v1 + v2)", data, expected);
}

// NULL Handling Tests

TEST_F(MapZipWithTest, allNullValues) {
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>(
          {{{1, std::nullopt}, {2, std::nullopt}}},
          makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt})),
      makeMapVector<int64_t, int64_t>(
          {{{1, std::nullopt}, {2, std::nullopt}}},
          makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt})),
  });

  auto expected = makeMapVector<int64_t, int64_t>(
      {{{1, std::nullopt}, {2, std::nullopt}}},
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt}));

  testMapZipWith("map_zip_with(c0, c1, (k, v1, v2) -> v1 + v2)", data, expected);
}

TEST_F(MapZipWithTest, coalescePattern) {
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}, {3, 30}},
      }),
      makeMapVector<int64_t, int64_t>({
          {{1, 100}, {4, 400}, {5, 500}},
      }),
  });

  // Use coalesce to handle NULLs - treat missing as 0
  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 110}, {2, 20}, {3, 30}, {4, 400}, {5, 500}},
  });

  testMapZipWith(
      "map_zip_with(c0, c1, (k, v1, v2) -> coalesce(v1, 0) + coalesce(v2, 0))",
      data,
      expected);
}

TEST_F(MapZipWithTest, lambdaReturnsNull) {
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}},
      }),
      makeMapVector<int64_t, int64_t>({
          {{1, 100}, {2, 200}},
      }),
  });

  // Lambda explicitly returns NULL
  auto expected = makeMapVector<int64_t, int64_t>(
      {{{1, std::nullopt}, {2, std::nullopt}}},
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt}));

  testMapZipWith(
      "map_zip_with(c0, c1, (k, v1, v2) -> CAST(NULL AS BIGINT))",
      data,
      expected);
}

// Lambda Pattern Tests

TEST_F(MapZipWithTest, lambdaUsesKey) {
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}, {3, 30}},
      }),
      makeMapVector<int64_t, int64_t>({
          {{1, 100}, {2, 200}, {3, 300}},
      }),
  });

  // Lambda uses key: (v1 + v2) * key
  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 110}, {2, 440}, {3, 990}},
  });

  testMapZipWith(
      "map_zip_with(c0, c1, (k, v1, v2) -> (v1 + v2) * k)",
      data,
      expected);
}

TEST_F(MapZipWithTest, lambdaIgnoresValues) {
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}},
      }),
      makeMapVector<int64_t, int64_t>({
          {{1, 100}, {2, 200}},
      }),
  });

  // Lambda returns constant
  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 42}, {2, 42}},
  });

  testMapZipWith("map_zip_with(c0, c1, (k, v1, v2) -> 42)", data, expected);
}

TEST_F(MapZipWithTest, lambdaComplexExpression) {
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}, {3, 30}},
      }),
      makeMapVector<int64_t, int64_t>({
          {{1, 5}, {2, 10}, {3, 15}},
      }),
  });

  // Nested function calls: abs(v1 - v2) * 2
  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });

  testMapZipWith(
      "map_zip_with(c0, c1, (k, v1, v2) -> abs(v1 - v2) * 2)",
      data,
      expected);
}

TEST_F(MapZipWithTest, lambdaConditional) {
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}, {3, 30}},
      }),
      makeMapVector<int64_t, int64_t>({
          {{1, 15}, {2, 5}, {3, 50}},
      }),
  });

  // IF condition: return larger value
  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 15}, {2, 20}, {3, 50}},
  });

  testMapZipWith(
      "map_zip_with(c0, c1, (k, v1, v2) -> IF(v1 > v2, v1, v2))",
      data,
      expected);
}

// Edge Case Tests

TEST_F(MapZipWithTest, singleEntryMaps) {
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}},
          {{2, 20}},
      }),
      makeMapVector<int64_t, int64_t>({
          {{1, 100}},
          {{3, 300}},
      }),
  });

  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 110}},
      {{2, std::nullopt}, {3, std::nullopt}},
  });

  testMapZipWith("map_zip_with(c0, c1, (k, v1, v2) -> v1 + v2)", data, expected);
}

TEST_F(MapZipWithTest, manyEntries) {
  // Create maps with 50 entries each
  std::map<int64_t, int64_t> map1, map2;
  for (int64_t i = 1; i <= 50; i++) {
    map1[i] = i * 10;
    map2[i] = i * 100;
  }

  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({map1}),
      makeMapVector<int64_t, int64_t>({map2}),
  });

  // Expected: all keys present, values summed
  std::map<int64_t, int64_t> expectedMap;
  for (int64_t i = 1; i <= 50; i++) {
    expectedMap[i] = i * 110;
  }

  auto expected = makeMapVector<int64_t, int64_t>({expectedMap});

  testMapZipWith("map_zip_with(c0, c1, (k, v1, v2) -> v1 + v2)", data, expected);
}

TEST_F(MapZipWithTest, emptyMaps) {
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {},
          {},
      }),
      makeMapVector<int64_t, int64_t>({
          {},
          {{1, 100}},
      }),
  });

  auto expected = makeMapVector<int64_t, int64_t>({
      {},
      {{1, std::nullopt}},
  });

  testMapZipWith("map_zip_with(c0, c1, (k, v1, v2) -> v1 + v2)", data, expected);
}

TEST_F(MapZipWithTest, arrayValues) {
  auto data = makeRowVector({
      makeMapVector(
          {0, 2},
          makeNullableFlatVector<int64_t>({1, 2}),
          makeArrayVector<int64_t>({{10, 11}, {20, 21}})),
      makeMapVector(
          {0, 2},
          makeNullableFlatVector<int64_t>({1, 2}),
          makeArrayVector<int64_t>({{100, 101}, {200, 201}})),
  });

  // concat_array: concatenate two arrays
  auto expected = makeMapVector(
      {0, 2},
      makeNullableFlatVector<int64_t>({1, 2}),
      makeArrayVector<int64_t>({{10, 11, 100, 101}, {20, 21, 200, 201}}));

  testMapZipWith(
      "map_zip_with(c0, c1, (k, v1, v2) -> concat(v1, v2))",
      data,
      expected);
}

TEST_F(MapZipWithTest, structValues) {
  auto data = makeRowVector({
      makeMapVector(
          {0, 2},
          makeNullableFlatVector<int64_t>({1, 2}),
          makeRowVector({
              makeNullableFlatVector<int64_t>({10, 20}),
              makeNullableFlatVector<std::string>({"a", "b"}),
          })),
      makeMapVector(
          {0, 2},
          makeNullableFlatVector<int64_t>({1, 2}),
          makeRowVector({
              makeNullableFlatVector<int64_t>({100, 200}),
              makeNullableFlatVector<std::string>({"x", "y"}),
          })),
  });

  // Access struct fields and concatenate strings
  auto expected = makeMapVector(
      {0, 2},
      makeNullableFlatVector<int64_t>({1, 2}),
      makeNullableFlatVector<std::string>({"ax", "by"}));

  testMapZipWith(
      "map_zip_with(c0, c1, (k, v1, v2) -> concat(v1.c1, v2.c1))",
      data,
      expected);
}

TEST_F(MapZipWithTest, captureMultipleColumns) {
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}},
      }),
      makeMapVector<int64_t, int64_t>({
          {{1, 100}, {2, 200}},
      }),
      makeFlatVector<int64_t>({5}),
      makeFlatVector<int64_t>({3}),
  });

  // Lambda captures two columns: (v1 + v2) * c2 + c3
  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 553}, {2, 1103}},
  });

  testMapZipWith(
      "map_zip_with(c0, c1, (k, v1, v2) -> (v1 + v2) * c2 + c3)",
      data,
      expected);
}

TEST_F(MapZipWithTest, unicodeKeys) {
  auto data = makeRowVector({
      makeMapVector<std::string, int64_t>({
          {{"日本", 1}, {"中国", 2}, {"한국", 3}},
      }),
      makeMapVector<std::string, int64_t>({
          {{"日本", 10}, {"中国", 20}, {"🌍", 30}},
      }),
  });

  auto expected = makeMapVector<std::string, int64_t>({
      {{"日本", 11}, {"中国", 22}, {"한국", std::nullopt}, {"🌍", std::nullopt}},
  });

  testMapZipWith("map_zip_with(c0, c1, (k, v1, v2) -> v1 + v2)", data, expected);
}

TEST_F(MapZipWithTest, specialFloatValues) {
  // Test NaN, Infinity
  auto data = makeRowVector({
      makeMapVector<int64_t, double>({
          {{1, std::numeric_limits<double>::quiet_NaN()},
           {2, std::numeric_limits<double>::infinity()},
           {3, 3.14}},
      }),
      makeMapVector<int64_t, double>({
          {{1, 1.0}, {2, 2.0}, {3, std::numeric_limits<double>::infinity()}},
      }),
  });

  auto expected = makeMapVector<int64_t, double>({
      {{1, std::numeric_limits<double>::quiet_NaN()},
       {2, std::numeric_limits<double>::infinity()},
       {3, std::numeric_limits<double>::infinity()}},
  });

  auto result = evaluateExpression(
      "map_zip_with(c0, c1, (k, v1, v2) -> v1 + v2)", data);
  
  // Special handling for NaN comparison
  auto resultMap = result->as<MapVector>();
  auto expectedMap = expected->as<MapVector>();
  
  ASSERT_EQ(resultMap->size(), expectedMap->size());
  
  // Verify key 1 has NaN (NaN != NaN, so check explicitly)
  auto resultValues = resultMap->mapValues()->as<FlatVector<double>>();
  ASSERT_TRUE(std::isnan(resultValues->valueAt(0)));
}

} // namespace facebook::velox::functions::sparksql::test
