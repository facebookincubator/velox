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
#include "velox/functions/prestosql/tests/utils/FuzzerTestUtils.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class MapIntersectTest : public test::FunctionBaseTest {
 protected:
  void testMapIntersect(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }
};

TEST_F(MapIntersectTest, integerMap) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}, {4, 40}},
      {{5, 50}, {6, 60}, {7, 70}},
      {},
      {{8, 80}, {9, 90}},
  });

  auto keys = makeArrayVector<int32_t>({
      {1, 3},
      {5, 7, 8},
      {1, 2},
      {8, 9, 10},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {3, 30}},
      {{5, 50}, {7, 70}},
      {},
      {{8, 80}, {9, 90}},
  });

  testMapIntersect("map_intersect(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapIntersectTest, constantKeys) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}},
      {{1, 100}, {2, 200}, {3, 300}},
      {{1, 1000}, {2, 2000}, {3, 3000}, {4, 4000}, {5, 5000}},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {3, 30}, {5, 50}},
      {{1, 100}, {3, 300}},
      {{1, 1000}, {3, 3000}, {5, 5000}},
  });

  auto result = evaluate(
      "map_intersect(c0, array_constructor(cast(1 as integer), cast(3 as integer), cast(5 as integer)))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapIntersectTest, nonExistentKeys) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
      {{4, 40}, {5, 50}},
  });

  auto keys = makeArrayVector<int32_t>({
      {4, 5, 6},
      {1, 2, 3},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {},
      {},
  });

  testMapIntersect("map_intersect(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapIntersectTest, duplicateKeys) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}},
      {{6, 60}, {7, 70}, {8, 80}},
  });

  auto keys = makeArrayVector<int32_t>({
      {1, 1, 2, 2, 3},
      {6, 6, 7, 7},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
      {{6, 60}, {7, 70}},
  });

  testMapIntersect("map_intersect(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapIntersectTest, emptyKeys) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}},
      {{4, 40}, {5, 50}},
  });

  auto keys = makeArrayVector<int32_t>({
      {},
      {},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {},
      {},
  });

  testMapIntersect("map_intersect(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapIntersectTest, emptyMap) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {},
      {},
  });

  auto keys = makeArrayVector<int32_t>({
      {1, 2, 3},
      {4, 5},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {},
      {},
  });

  testMapIntersect("map_intersect(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapIntersectTest, nullKeysInArray) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}},
      {{6, 60}, {7, 70}, {8, 80}},
  });

  auto keys = makeNullableArrayVector<int32_t>({
      {1, std::nullopt, 3, std::nullopt, 5},
      {std::nullopt, 7, std::nullopt},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {3, 30}, {5, 50}},
      {{7, 70}},
  });

  testMapIntersect("map_intersect(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapIntersectTest, stringMap) {
  auto inputMap = makeMapVector<StringView, int32_t>({
      {{"apple", 1}, {"banana", 2}, {"cherry", 3}, {"date", 4}},
      {{"hello", 10}, {"world", 20}},
      {{"a", 100}, {"b", 200}, {"c", 300}, {"d", 400}, {"e", 500}},
  });

  auto keys = makeArrayVector<StringView>({
      {"banana", "date", "apple"},
      {"hello", "world", "foo"},
      {"a", "c", "e"},
  });

  auto expected = makeMapVector<StringView, int32_t>({
      {{"apple", 1}, {"banana", 2}, {"date", 4}},
      {{"hello", 10}, {"world", 20}},
      {{"a", 100}, {"c", 300}, {"e", 500}},
  });

  testMapIntersect("map_intersect(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapIntersectTest, floatMap) {
  auto inputMap = makeMapVector<float, float>({
      {{1.1f, 10.1f}, {2.2f, 20.2f}, {3.3f, 30.3f}, {4.4f, 40.4f}},
      {{5.5f, 50.5f}, {6.6f, 60.6f}, {7.7f, 70.7f}},
  });

  auto keys = makeArrayVector<float>({
      {1.1f, 3.3f, 4.4f},
      {5.5f, 6.6f, 8.8f},
  });

  auto expected = makeMapVector<float, float>({
      {{1.1f, 10.1f}, {3.3f, 30.3f}, {4.4f, 40.4f}},
      {{5.5f, 50.5f}, {6.6f, 60.6f}},
  });

  testMapIntersect("map_intersect(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapIntersectTest, booleanMap) {
  auto inputMap = makeMapVector<bool, int32_t>({
      {{true, 1}, {false, 0}},
      {{true, 10}},
      {{false, 100}},
  });

  auto keys = makeArrayVector<bool>({
      {true},
      {true, false},
      {true, false},
  });

  auto expected = makeMapVector<bool, int32_t>({
      {{true, 1}},
      {{true, 10}},
      {{false, 100}},
  });

  testMapIntersect("map_intersect(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapIntersectTest, nullArguments) {
  auto inputMap =
      makeMapVector<int32_t, int32_t>({{{1, 10}, {2, 20}, {3, 30}}});
  auto keys = makeArrayVector<int32_t>({{1, 2}});

  auto nullMap =
      BaseVector::createNullConstant(MAP(INTEGER(), INTEGER()), 1, pool());
  auto nullKeys = BaseVector::createNullConstant(ARRAY(INTEGER()), 1, pool());

  auto result1 =
      evaluate("map_intersect(c0, c1)", makeRowVector({nullMap, keys}));
  auto expected1 =
      BaseVector::createNullConstant(MAP(INTEGER(), INTEGER()), 1, pool());
  assertEqualVectors(expected1, result1);

  auto result2 =
      evaluate("map_intersect(c0, c1)", makeRowVector({inputMap, nullKeys}));
  auto expected2 =
      BaseVector::createNullConstant(MAP(INTEGER(), INTEGER()), 1, pool());
  assertEqualVectors(expected2, result2);

  auto result3 =
      evaluate("map_intersect(c0, c1)", makeRowVector({nullMap, nullKeys}));
  auto expected3 =
      BaseVector::createNullConstant(MAP(INTEGER(), INTEGER()), 1, pool());
  assertEqualVectors(expected3, result3);
}

TEST_F(MapIntersectTest, partialMatch) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}},
      {{6, 60}, {7, 70}, {8, 80}, {9, 90}},
  });

  auto keys = makeArrayVector<int32_t>({
      {1, 3, 5, 7, 9},
      {6, 8, 10, 12},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{1, 10}, {3, 30}, {5, 50}},
      {{6, 60}, {8, 80}},
  });

  testMapIntersect("map_intersect(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapIntersectTest, singleElementMap) {
  auto inputMap = makeMapVector<int32_t, int32_t>({
      {{1, 10}},
      {{2, 20}},
  });

  auto keys = makeArrayVector<int32_t>({
      {1, 2, 3},
      {2},
  });

  auto expected = makeMapVector<int32_t, int32_t>({
      {{1, 10}},
      {{2, 20}},
  });

  testMapIntersect("map_intersect(c0, c1)", {inputMap, keys}, expected);
}

TEST_F(MapIntersectTest, basicTest) {
  auto inputMap = makeMapVector<int32_t, int32_t>(
      {{{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}}});
  auto keys = makeArrayVector<int32_t>({{1, 3, 5}});
  auto expected =
      makeMapVector<int32_t, int32_t>({{{1, 10}, {3, 30}, {5, 50}}});

  auto result =
      evaluate("map_intersect(c0, c1)", makeRowVector({inputMap, keys}));
  assertEqualVectors(expected, result);
}

} // namespace

class MapIntersectFuzzerTest : public test::FunctionBaseTest {
 protected:
  static constexpr const char* kEquivalentExpression =
      "map_filter(c0, (k, v) -> coalesce(contains(c1, k), false))";

  static SelectivityVector getNonNullRows(const RowVectorPtr& data) {
    auto inputMap = data->childAt(0);
    auto keys = data->childAt(1);
    SelectivityVector nonNullRows(data->size());
    for (vector_size_t i = 0; i < data->size(); ++i) {
      if (inputMap->isNullAt(i) || keys->isNullAt(i)) {
        nonNullRows.setValid(i, false);
      }
    }
    nonNullRows.updateBounds();
    return nonNullRows;
  }

  void testEquivalence(const RowVectorPtr& data) {
    auto result = evaluate("map_intersect(c0, c1)", data);
    auto expected = evaluate(kEquivalentExpression, data);
    auto nonNullRows = getNonNullRows(data);
    for (auto i = 0; i < data->size(); ++i) {
      if (nonNullRows.isValid(i)) {
        ASSERT_TRUE(expected->equalValueAt(result.get(), i, i))
            << "Mismatch at row " << i << ": expected " << expected->toString(i)
            << ", got " << result->toString(i);
      }
    }
  }

  void runFuzzTest(const TypePtr& type, const test::FuzzerTestOptions& opts) {
    test::FuzzerTestHelper helper(pool());
    helper.runMapArrayTest(
        type,
        type,
        [this](const VectorPtr& inputMap, const VectorPtr& keys) {
          auto data = makeRowVector({inputMap, keys});
          testEquivalence(data);
        },
        opts);
  }
};

TEST_F(MapIntersectFuzzerTest, fuzzInteger) {
  runFuzzTest(INTEGER(), {.vectorSize = 100});
}

TEST_F(MapIntersectFuzzerTest, fuzzBigint) {
  runFuzzTest(BIGINT(), {.vectorSize = 100});
}

TEST_F(MapIntersectFuzzerTest, fuzzVarchar) {
  runFuzzTest(VARCHAR(), {.vectorSize = 100});
}

TEST_F(MapIntersectFuzzerTest, fuzzDouble) {
  runFuzzTest(DOUBLE(), {.vectorSize = 100});
}

TEST_F(MapIntersectFuzzerTest, fuzzHighNullRatio) {
  runFuzzTest(INTEGER(), {.vectorSize = 100, .nullRatio = 0.5});
}

TEST_F(MapIntersectFuzzerTest, fuzzSmallint) {
  runFuzzTest(SMALLINT(), {.vectorSize = 100});
}

TEST_F(MapIntersectFuzzerTest, fuzzLargeVectors) {
  runFuzzTest(INTEGER(), {.vectorSize = 500});
}

} // namespace facebook::velox::functions
