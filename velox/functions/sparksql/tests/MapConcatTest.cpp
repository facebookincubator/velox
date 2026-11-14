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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class MapConcatTest : public SparkFunctionBaseTest {
 protected:
  template <typename TKey, typename TValue>
  static std::vector<TKey> mapKeys(const std::map<TKey, TValue>& m) {
    std::vector<TKey> keys;
    keys.reserve(m.size());
    for (const auto& [key, value] : m) {
      keys.push_back(key);
    }
    return keys;
  }

  template <typename TKey, typename TValue>
  static std::vector<TValue> mapValues(const std::map<TKey, TValue>& m) {
    std::vector<TValue> values;
    values.reserve(m.size());
    for (const auto& [key, value] : m) {
      values.push_back(value);
    }
    return values;
  }

  MapVectorPtr makeMapVector(
      vector_size_t size,
      const std::map<std::string, int32_t>& m,
      std::function<bool(vector_size_t /*row*/)> isNullAt = nullptr) {
    std::vector<std::string> keys = mapKeys(m);
    std::vector<int32_t> values = mapValues(m);
    return vectorMaker_.mapVector<StringView, int32_t>(
        size,
        [&](vector_size_t /*row*/) { return keys.size(); },
        [&](vector_size_t /*mapRow*/, vector_size_t row) {
          return StringView(keys[row]);
        },
        [&](vector_size_t mapRow, vector_size_t row) {
          return mapRow % 11 + values[row];
        },
        isNullAt);
  }

  template <typename TKey, typename TValue>
  std::map<TKey, TValue> concat(
      const std::map<TKey, TValue>& a,
      const std::map<TKey, TValue>& b) {
    std::map<TKey, TValue> result;
    result.insert(b.begin(), b.end());
    result.insert(a.begin(), a.end());
    return result;
  }

  // Checks that the expression gives a flat map equal to 'expected' over the
  // flat map inputs, and the same maps over the same inputs as MapVectors.
  void testFlatMap(
      const std::string& expression,
      const std::vector<VectorPtr>& flatMapInputs,
      const std::vector<VectorPtr>& mapInputs,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(flatMapInputs));
    ASSERT_EQ(result->encoding(), VectorEncoding::Simple::FLAT_MAP);
    result->validate();
    velox::test::assertEqualVectors(expected, result);
    velox::test::assertEqualVectors(
        evaluate(expression, makeRowVector(mapInputs)), result);
  }
};

TEST_F(MapConcatTest, duplicateKeys) {
  vector_size_t size = 1'000;

  std::map<std::string, int32_t> a = {
      {"a1", 1}, {"a2", 2}, {"a3", 3}, {"a4", 4}};
  std::map<std::string, int32_t> b = {
      {"b1", 1}, {"b2", 2}, {"b3", 3}, {"b4", 4}, {"a2", -1}};
  auto aMap = makeMapVector(size, a);
  auto bMap = makeMapVector(size, b);

  // By default, if a key is found in multiple given maps, that key's value in
  // the resulting map comes from the last one of those maps.
  std::map<std::string, int32_t> ab = concat(a, b);
  auto expectedMap = makeMapVector(size, ab);
  auto result =
      evaluate<MapVector>("map_concat(c0, c1)", makeRowVector({aMap, bMap}));
  velox::test::assertEqualVectors(expectedMap, result);

  std::map<std::string, int32_t> ba = concat(b, a);
  expectedMap = makeMapVector(size, ba);
  result =
      evaluate<MapVector>("map_concat(c1, c0)", makeRowVector({aMap, bMap}));
  velox::test::assertEqualVectors(expectedMap, result);

  result =
      evaluate<MapVector>("map_concat(c0, c1)", makeRowVector({aMap, aMap}));
  velox::test::assertEqualVectors(aMap, result);

  // Throws exception when duplicate keys are found.
  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kThrowExceptionOnDuplicateMapKeys, "true"},
  });
  VELOX_ASSERT_USER_THROW(
      evaluate<MapVector>("map_concat(c0, c1)", makeRowVector({aMap, bMap})),
      "Duplicate map key a2 was found");
}

TEST_F(MapConcatTest, flatMapWithMap) {
  auto flatMap = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20}",
      "{3:30}",
  });
  auto map = makeMapVectorFromJson<int64_t, int64_t>({
      "{4:40}",
      "{3:31}",
  });

  VELOX_ASSERT_THROW(
      evaluate("map_concat(c0, c1)", makeRowVector({flatMap, map})),
      "Wrong type cast expected");
  VELOX_ASSERT_THROW(
      evaluate("map_concat(c1, c0)", makeRowVector({flatMap, map})),
      "Wrong type cast expected");
  VELOX_ASSERT_THROW(
      evaluate("map_concat(c0, map(7, 70))", makeRowVector({flatMap})),
      "Wrong type cast expected");
}

TEST_F(MapConcatTest, flatMapWithMapLiteral) {
  auto flatMap = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20}",
      "null",
      "{3:30}",
  });

  setGenerateConstantMapsAsFlatMaps(false);
  auto result =
      evaluate("map_concat(c0, map(2, 99, 7, 70))", makeRowVector({flatMap}));
  ASSERT_EQ(result->encoding(), VectorEncoding::Simple::FLAT_MAP);
  result->validate();
  velox::test::assertEqualVectors(
      makeMapVectorFromJson<int64_t, int64_t>({
          "{1:10, 2:99, 7:70}",
          "null",
          "{2:99, 3:30, 7:70}",
      }),
      result);
  const auto& flatMapResult = *result->asUnchecked<FlatMapVector>();
  EXPECT_EQ(
      flatMapResult.mapValuesAt(
          flatMapResult.getKeyChannel(int64_t{1}).value()),
      flatMap->mapValuesAt(flatMap->getKeyChannel(int64_t{1}).value()));
}

TEST_F(MapConcatTest, flatMapDuplicateKeys) {
  auto flatMap = makeFlatMapVectorFromJson<int32_t, int64_t>({
      "{1:10, 2:20}",
      "null",
      "{3:30}",
  });
  auto other = makeFlatMapVectorFromJson<int32_t, int64_t>({
      "{2:21, 4:40}",
      "{5:50}",
      "{6:60}",
  });

  testFlatMap(
      "map_concat(c0, c1)",
      {flatMap, other},
      {flatMap->toMapVector(), other->toMapVector()},
      makeMapVectorFromJson<int32_t, int64_t>({
          "{1:10, 2:21, 4:40}",
          "null",
          "{3:30, 6:60}",
      }));
  testFlatMap(
      "map_concat(c1, c0)",
      {flatMap, other},
      {flatMap->toMapVector(), other->toMapVector()},
      makeMapVectorFromJson<int32_t, int64_t>({
          "{1:10, 2:20, 4:40}",
          "null",
          "{3:30, 6:60}",
      }));

  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kThrowExceptionOnDuplicateMapKeys, "true"},
  });
  VELOX_ASSERT_USER_THROW(
      evaluate("map_concat(c0, c1)", makeRowVector({flatMap, other})),
      "Duplicate map key 2 was found.");
}

TEST_F(MapConcatTest, singleArg) {
  vector_size_t size = 1'000;

  std::map<std::string, int32_t> a = {{"a1", 1}, {"a2", 2}, {"a3", 3}};
  auto aMap = makeMapVector(size, a, nullEvery(5));

  auto expectedMap =
      makeMapVector(size, a, [](vector_size_t row) { return row % 5 == 0; });

  auto result = evaluate<MapVector>("map_concat(c0)", makeRowVector({aMap}));
  velox::test::assertEqualVectors(expectedMap, result);
}

TEST_F(MapConcatTest, flatMapSingleArg) {
  auto flatMap = makeFlatMapVectorFromJson<int32_t, int64_t>({
      "{1:10, 2:20}",
      "null",
      "{}",
  });

  testFlatMap("map_concat(c0)", {flatMap}, {flatMap->toMapVector()}, flatMap);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
