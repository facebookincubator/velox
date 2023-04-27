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
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace {

class MapTest : public FunctionBaseTest {};

TEST_F(MapTest, noNulls) {
  auto size = 1'000;

  auto sizeAt = [](vector_size_t row) { return row % 5; };
  auto keyAt = [](vector_size_t row) { return row % 11; };
  auto valueAt = [](vector_size_t row) { return row % 13; };
  auto keys = makeArrayVector<int64_t>(size, sizeAt, keyAt);
  auto values = makeArrayVector<int32_t>(size, sizeAt, valueAt);

  auto expectedMap =
      makeMapVector<int64_t, int32_t>(size, sizeAt, keyAt, valueAt);

  auto result =
      evaluate<MapVector>("map(c0, c1)", makeRowVector({keys, values}));
  assertEqualVectors(expectedMap, result);
}

TEST_F(MapTest, someNulls) {
  auto size = 1'000;

  auto sizeAt = [](vector_size_t row) { return row % 5; };
  auto keyAt = [](vector_size_t row) { return row % 11; };
  auto valueAt = [](vector_size_t row) { return row % 13; };
  auto keys = makeArrayVector<int64_t>(size, sizeAt, keyAt, nullEvery(7));
  auto values = makeArrayVector<int32_t>(size, sizeAt, valueAt, nullEvery(7));

  auto expectedMap = makeMapVector<int64_t, int32_t>(
      size, sizeAt, keyAt, valueAt, nullEvery(7));

  auto result =
      evaluate<MapVector>("map(c0, c1)", makeRowVector({keys, values}));
  assertEqualVectors(expectedMap, result);
}

TEST_F(MapTest, partiallyPopulated) {
  auto size = 1'000;

  auto sizeAt = [](vector_size_t row) { return row % 5; };
  auto keyAt = [](vector_size_t row) { return row % 11; };
  auto valueAt = [](vector_size_t row) { return row % 13; };
  auto keys = makeArrayVector<int64_t>(size, sizeAt, keyAt);
  auto values = makeArrayVector<int64_t>(size, sizeAt, valueAt);
  auto condition =
      makeFlatVector<int16_t>(size, [](vector_size_t row) { return row % 2; });

  auto expectedEvenMap =
      makeMapVector<int64_t, int64_t>(size, sizeAt, keyAt, valueAt);

  auto expectedOddMap =
      makeMapVector<int64_t, int64_t>(size, sizeAt, valueAt, keyAt);

  auto result = evaluate<MapVector>(
      "if(c2 = 0, map(c0, c1), map(c1, c0))",
      makeRowVector({keys, values, condition}));
  ASSERT_EQ(result->size(), size);
  for (vector_size_t i = 0; i < size; i += 2) {
    ASSERT_TRUE(expectedEvenMap->equalValueAt(result.get(), i, i))
        << "at " << i << ": expected " << expectedEvenMap->toString(i)
        << ", got " << result->toString(i);
  }

  for (vector_size_t i = 1; i < size; i += 2) {
    ASSERT_TRUE(expectedOddMap->equalValueAt(result.get(), i, i))
        << "at " << i << ": expected " << expectedOddMap->toString(i)
        << ", got " << result->toString(i);
  }
}

TEST_F(MapTest, nullKeys) {
  auto keys = makeNullableArrayVector<int64_t>({
      {1, 2, 3, std::nullopt},
      {1, 2},
      {std::nullopt},
  });

  auto values = makeNullableArrayVector<int64_t>({
      {10, 20, 30, 40},
      {10, 20},
      {10},
  });

  VELOX_ASSERT_THROW(
      evaluate<MapVector>("map(c0, c1)", makeRowVector({keys, values})),
      "map key cannot be null");

  auto result =
      evaluate<MapVector>("try(map(c0, c1))", makeRowVector({keys, values}));
  assertEqualVectors(
      makeNullableMapVector<int64_t, int64_t>({
          std::nullopt,
          {{{1, 10}, {2, 20}}},
          std::nullopt,
      }),
      result);
}

TEST_F(MapTest, duplicateKeys) {
  auto size = 1'000;

  auto sizeAt = [](vector_size_t row) { return row % 7; };
  auto keys = makeArrayVector<int64_t>(
      size, sizeAt, [](vector_size_t row) { return 10 + row % 3; });
  auto values = makeArrayVector<int32_t>(
      size, sizeAt, [](vector_size_t row) { return row % 5; });

  VELOX_ASSERT_THROW(
      evaluate<MapVector>("map(c0, c1)", makeRowVector({keys, values})),
      "Duplicate map keys (10) are not allowed");

  ASSERT_NO_THROW(
      evaluate<MapVector>("try(map(c0, c1))", makeRowVector({keys, values})));

  // Trying the map version with allowing duplicates.
  functions::prestosql::registerMapAllowingDuplicates("map2");
  ASSERT_NO_THROW(
      evaluate<MapVector>("map2(c0, c1)", makeRowVector({keys, values})));
}

TEST_F(MapTest, fewerValuesThanKeys) {
  auto size = 1'000;

  // Make sure that some rows have fewer 'values' than 'keys'.
  auto keys = makeArrayVector<int64_t>(
      size,
      [](vector_size_t row) { return row % 7; },
      [](vector_size_t row) { return row % 11; });
  auto values = makeArrayVector<int32_t>(
      size,
      [](vector_size_t row) { return row % 5; },
      [](vector_size_t row) { return row % 13; });

  VELOX_ASSERT_THROW(
      evaluate<MapVector>("map(c0, c1)", makeRowVector({keys, values})),
      "(5 vs. 0) Key and value arrays must be the same length");

  ASSERT_NO_THROW(
      evaluate<MapVector>("try(map(c0, c1))", makeRowVector({keys, values})));
}

TEST_F(MapTest, fewerValuesThanKeysInLast) {
  // Element 0 of the map vector is valid, element 1 is missing values
  // and should come out empty when not throwing errors. The starts of
  // the keys and values are aligned but the lengths are not.
  auto size = 2;

  auto keys = makeArrayVector<int64_t>(
      size,
      [](vector_size_t row) { return 10; },
      [](vector_size_t row) { return row % 11; });
  auto values = makeArrayVector<int32_t>(
      size,
      [](vector_size_t row) { return row == 0 ? 10 : 1; },
      [](vector_size_t row) { return row % 13; });

  VELOX_ASSERT_THROW(
      evaluate<MapVector>("map(c0, c1)", makeRowVector({keys, values})),
      "(10 vs. 1) Key and value arrays must be the same length");

  auto map =
      evaluate<MapVector>("try(map(c0, c1))", makeRowVector({keys, values}));
  EXPECT_EQ(10, map->sizeAt(0));
  EXPECT_EQ(0, map->sizeAt(1));

  auto condition = makeFlatVector<bool>({true, false, true, false, true});

  // Makes a vector of keys and values where items 0, 2 and 4 are
  // aligned. The keys vector as a whole is still longer than the
  // values vector.
  auto keys2 = evaluate(
      "if(c0, array[3, 2, 1], array[6, 5, 4])", makeRowVector({condition}));
  auto values2 = evaluate(
      "if(c0, array[30, 20, 10], array[40])", makeRowVector({condition}));

  auto result = evaluate<MapVector>(
      "try("
      "   if(c0, "
      "       map(c1, c2), "
      "       cast(null as map(integer, integer))))",
      makeRowVector({condition, keys2, values2}));
  EXPECT_EQ(0, result->offsetAt(0));
  EXPECT_EQ(3, result->offsetAt(2));
  EXPECT_EQ(6, result->offsetAt(4));
  EXPECT_EQ(9, result->mapKeys()->size());
  EXPECT_EQ(9, result->mapValues()->size());
}

TEST_F(MapTest, fewerKeysThanValues) {
  auto size = 1'000;

  // Make sure that some rows have fewer 'keys' than 'values'.
  auto keys = makeArrayVector<int64_t>(
      size,
      [](vector_size_t row) { return row % 5; },
      [](vector_size_t row) { return row % 11; });
  auto values = makeArrayVector<int32_t>(
      size,
      [](vector_size_t row) { return row % 7; },
      [](vector_size_t row) { return row % 13; });

  VELOX_ASSERT_THROW(
      evaluate<MapVector>("map(c0, c1)", makeRowVector({keys, values})),
      "(0 vs. 5) Key and value arrays must be the same length");

  ASSERT_NO_THROW(
      evaluate<MapVector>("try(map(c0, c1))", makeRowVector({keys, values})));
}

TEST_F(MapTest, encodings) {
  auto size = 1'000;

  auto sizeAt = [](vector_size_t row) { return (row / 2) % 5; };
  auto keyAt = [](vector_size_t row) { return row % 11; };
  auto valueAt = [](vector_size_t row) { return row % 13; };

  // Use different dictionary encodings for keys and values
  auto keys = wrapInDictionary(
      makeOddIndices(size),
      size,
      makeArrayVector<int64_t>(size * 2, sizeAt, keyAt));
  auto values = wrapInDictionary(
      makeEvenIndices(size),
      size,
      makeArrayVector<int32_t>(size * 2, sizeAt, valueAt));

  auto flatKeys = std::dynamic_pointer_cast<ArrayVector>(flatten(keys));
  auto flatValues = std::dynamic_pointer_cast<ArrayVector>(flatten(values));

  auto expectedMap = std::make_shared<MapVector>(
      execCtx_.pool(),
      MAP(BIGINT(), INTEGER()),
      BufferPtr(nullptr),
      size,
      flatKeys->offsets(),
      flatKeys->sizes(),
      flatKeys->elements(),
      flatValues->elements());

  auto result =
      evaluate<MapVector>("map(c0, c1)", makeRowVector({keys, values}));
  assertEqualVectors(expectedMap, result);
}

// Test map function applied to a constant array of keys and flat array of
// values.
TEST_F(MapTest, constantKeys) {
  auto size = 1'000;

  auto sizeAt = [](vector_size_t /*row*/) { return 1; };
  auto keyAt = [](vector_size_t /*row*/) { return "key"_sv; };
  auto valueAt = [](vector_size_t row) { return row; };

  auto expectedMap =
      makeMapVector<StringView, int32_t>(size, sizeAt, keyAt, valueAt);

  auto result = evaluate<MapVector>(
      "map(array['key'], array_constructor(c0))",
      makeRowVector({
          makeFlatVector<int32_t>(size, valueAt),
      }));
  assertEqualVectors(expectedMap, result);
}

// Test map function applied to a flat array of keys and constant array of
// values.
TEST_F(MapTest, constantValues) {
  auto size = 1'000;

  auto sizeAt = [](vector_size_t /*row*/) { return 1; };
  auto keyAt = [](vector_size_t row) { return row; };
  auto valueAt = [](vector_size_t /*row*/) { return "value"_sv; };

  auto expectedMap =
      makeMapVector<int32_t, StringView>(size, sizeAt, keyAt, valueAt);

  auto result = evaluate<MapVector>(
      "map(array_constructor(c0), array['value'])",
      makeRowVector({
          makeFlatVector<int32_t>(size, keyAt),
      }));
  assertEqualVectors(expectedMap, result);
}

TEST_F(MapTest, outOfOrder) {
  auto size = 1'000;

  auto sizeAt = [](vector_size_t row) { return row % 5; };
  auto keyAt = [](vector_size_t row) { return row % 11; };
  auto valueAt = [](vector_size_t row) { return row % 13; };

  auto keys1 = makeArrayVector<int64_t>(size, sizeAt, keyAt);
  auto keys2 = makeArrayVector<int64_t>(size, sizeAt, keyAt);

  auto values1 = makeArrayVector<int32_t>(size, sizeAt, valueAt);
  auto values2 = makeArrayVector<int32_t>(size, sizeAt, valueAt);

  auto intVector =
      makeFlatVector<int32_t>(size, [](vector_size_t row) { return row; });

  auto expectedMap =
      makeMapVector<int64_t, int32_t>(size, sizeAt, keyAt, valueAt);

  auto result = evaluate<MapVector>(
      "map(if(c0 \% 2 = 1, c1, c2), if(c0 \% 3 = 0, c3, c4))",
      makeRowVector({intVector, keys1, keys2, values1, values2}));
  assertEqualVectors(expectedMap, result);
}

} // namespace
