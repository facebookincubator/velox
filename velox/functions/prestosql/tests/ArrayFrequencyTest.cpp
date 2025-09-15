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

using facebook::velox::test::assertEqualVectors;

namespace facebook::velox::functions::test {

namespace {

class ArrayFrequencyTest : public functions::test::FunctionBaseTest {
 protected:
  void testArrayFrequency(const VectorPtr& expected, const VectorPtr& input) {
    auto result =
        evaluate<BaseVector>("array_frequency(C0)", makeRowVector({input}));
    assertEqualVectors(expected, result);
  }
};

} // namespace

TEST_F(ArrayFrequencyTest, integerArray) {
  auto array = makeNullableArrayVector<int64_t>(
      {{2, 1, 1, -2},
       {},
       {1, 2, 1, 1, 1, 1},
       {-1, std::nullopt, -1, -1},
       {std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::max(),
        1,
        std::nullopt,
        0,
        1,
        std::nullopt,
        0}});

  auto expected = makeMapVector<int64_t, int>(
      {{{1, 2}, {2, 1}, {-2, 1}},
       {},
       {{1, 5}, {2, 1}},
       {{-1, 3}},
       {{std::numeric_limits<int64_t>::max(), 2}, {1, 2}, {0, 2}}});

  testArrayFrequency(expected, array);
}

TEST_F(ArrayFrequencyTest, integerArrayWithoutNull) {
  auto array =
      makeArrayVector<int64_t>({{2, 1, 1, -2}, {}, {1, 2, 1, 1, 1, 1}});

  auto expected = makeMapVector<int64_t, int>(
      {{{1, 2}, {2, 1}, {-2, 1}}, {}, {{1, 5}, {2, 1}}});

  testArrayFrequency(expected, array);
}

TEST_F(ArrayFrequencyTest, varcharArray) {
  auto array = makeNullableArrayVector<StringView>({
      {"hello"_sv, "world"_sv, "!"_sv, "!"_sv, "!"_sv},
      {},
      {"hello"_sv, "world"_sv, std::nullopt, "!"_sv, "!"_sv},
      {"helloworldhelloworld"_sv,
       "helloworldhelloworld"_sv,
       std::nullopt,
       "!"_sv,
       "!"_sv},
  });

  auto expected = makeMapVector<StringView, int>(
      {{{"hello"_sv, 1}, {"world"_sv, 1}, {"!"_sv, 3}},
       {},
       {{"hello"_sv, 1}, {"world"_sv, 1}, {"!"_sv, 2}},
       {{"helloworldhelloworld"_sv, 2}, {"!"_sv, 2}}});

  testArrayFrequency(expected, array);
}

TEST_F(ArrayFrequencyTest, varcharArrayWithoutNull) {
  auto array = makeNullableArrayVector<StringView>({
      {"hello"_sv, "world"_sv, "!"_sv, "!"_sv, "!"_sv},
      {},
      {"helloworldhelloworld"_sv, "helloworldhelloworld"_sv, "!"_sv, "!"_sv},
  });
  auto expected = makeMapVector<StringView, int>(
      {{{"hello"_sv, 1}, {"world"_sv, 1}, {"!"_sv, 3}},
       {},
       {{"helloworldhelloworld"_sv, 2}, {"!"_sv, 2}}});

  testArrayFrequency(expected, array);
}

TEST_F(ArrayFrequencyTest, nestedArrayOfVarchar) {
  // Create nested array: [["a","b"], ["a","b"], ["c"]], [], [["x"], ["x"],
  // ["y","z"]]

  // inner arrays
  auto innerArrays = makeArrayVector<StringView>({
      {"a"_sv, "b"_sv},
      {"a"_sv, "b"_sv},
      {"c"_sv},
      {"x"_sv},
      {"x"_sv},
      {"y"_sv, "z"_sv},
  });

  // outer array using offsets
  auto nestedArray = makeArrayVector({0, 3, 3, 6}, innerArrays);

  auto result =
      evaluate<BaseVector>("array_frequency(C0)", makeRowVector({nestedArray}));

  // verify it doesn't crash and returns the expected type
  EXPECT_EQ(result->type()->toString(), "MAP<ARRAY<VARCHAR>,INTEGER>");
  EXPECT_EQ(result->size(), nestedArray->size());

  // verify that it's not null
  EXPECT_FALSE(result->isNullAt(0));

  // empty array, array_frequency returns an empty map, not null
  auto mapResult = result->as<MapVector>();
  EXPECT_EQ(mapResult->sizeAt(1), 0); // empty array should result in empty map
  EXPECT_FALSE(result->isNullAt(2));
  EXPECT_FALSE(result->isNullAt(3));
}

TEST_F(ArrayFrequencyTest, nestedArrayOfVarcharWithNulls) {
  // Test case matching Presto's actual behavior: arrays with null elements
  auto innerArrays = makeNullableArrayVector<StringView>({
      {"a"_sv, "b"_sv},
      {std::nullopt, "b"_sv}, // Contains null - should be included
      {"a"_sv, "b"_sv}, // Duplicate of first
      {}, // Empty array
      {"x"_sv},
      {"x"_sv}, // Duplicate
      {std::nullopt}, // Contains only null - should be included
      {"y"_sv, "z"_sv},
  });

  // [["a", "b"], [null, "b"], ["a", "b"]] -> ["a", "b"] appears twice, [null,
  // "b"] once
  // [] -> empty outer array
  // [[], ["x"], ["x"], [null]] -> [] once, ["x"] twice, [null] once
  // [["y", "z"]] -> ["y", "z"] once
  auto nestedArray = makeArrayVector({0, 3, 3, 7, 8}, innerArrays);

  auto result =
      evaluate<BaseVector>("array_frequency(C0)", makeRowVector({nestedArray}));

  EXPECT_EQ(result->type()->toString(), "MAP<ARRAY<VARCHAR>,INTEGER>");
  EXPECT_EQ(result->size(), nestedArray->size());
  EXPECT_FALSE(result->isNullAt(0));
  EXPECT_FALSE(result->isNullAt(1));
  EXPECT_FALSE(result->isNullAt(2));
  EXPECT_FALSE(result->isNullAt(3));

  auto mapResult = result->as<MapVector>();

  // Row 0: Should have 2 entries: ["a", "b"] -> 2, [null, "b"] -> 1
  EXPECT_EQ(mapResult->sizeAt(0), 2);

  // Row 1: Empty outer array should result in empty map
  EXPECT_EQ(mapResult->sizeAt(1), 0);

  // Row 2: Should have 3 entries: [] -> 1, ["x"] -> 2, [null] -> 1
  EXPECT_EQ(mapResult->sizeAt(2), 3);

  // Row 3: Should have 1 entry: ["y", "z"] -> 1
  EXPECT_EQ(mapResult->sizeAt(3), 1);
}

TEST_F(ArrayFrequencyTest, nestedArrayOfIntegers) {
  // Test generic implementation with nested arrays of integers
  auto innerArrays = makeArrayVector<int64_t>({
      {1, 2},
      {1, 2}, // Duplicate
      {3},
      {}, // Empty array
      {10, 20, 30},
      {10, 20, 30}, // Duplicate
  });

  // [[1, 2], [1, 2], [3]] -> [1, 2] appears twice, [3] once
  // [] -> empty outer array
  // [[], [10, 20, 30], [10, 20, 30]] -> [] once, [10, 20, 30] twice
  auto nestedArray = makeArrayVector({0, 3, 3, 6}, innerArrays);

  auto result =
      evaluate<BaseVector>("array_frequency(C0)", makeRowVector({nestedArray}));

  EXPECT_EQ(result->type()->toString(), "MAP<ARRAY<BIGINT>,INTEGER>");
  EXPECT_EQ(result->size(), nestedArray->size());
  EXPECT_FALSE(result->isNullAt(0));
  EXPECT_FALSE(result->isNullAt(1));
  EXPECT_FALSE(result->isNullAt(2));

  auto mapResult = result->as<MapVector>();

  // Row 0: Should have 2 entries: [1, 2] -> 2, [3] -> 1
  EXPECT_EQ(mapResult->sizeAt(0), 2);

  // Row 1: Empty outer array should result in empty map
  EXPECT_EQ(mapResult->sizeAt(1), 0);

  // Row 2: Should have 2 entries: [] -> 1, [10, 20, 30] -> 2
  EXPECT_EQ(mapResult->sizeAt(2), 2);
}
} // namespace facebook::velox::functions::test
