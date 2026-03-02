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
#include <optional>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {
class ArraySplitIntoChunksTest : public test::FunctionBaseTest {
 protected:
  template <typename T>
  void testSplitIntoChunks(
      const std::vector<std::optional<T>>& inputArray,
      int32_t sz,
      const std::vector<std::optional<std::vector<std::optional<T>>>>&
          expectedOutput) {
    std::vector<std::optional<std::vector<std::optional<T>>>> inputVec(
        {inputArray});
    auto input = makeNullableArrayVector<T>(inputVec);
    auto result = evaluate(
        fmt::format("array_split_into_chunks(c0, {}::INTEGER)", sz),
        makeRowVector({input}));

    auto expected = makeNullableNestedArrayVector<T>({expectedOutput});
    assertEqualVectors(expected, result);
  }
};

TEST_F(ArraySplitIntoChunksTest, integers) {
  testSplitIntoChunks<int64_t>({1, 2, 3}, 2, {{{1, 2}}, {{3}}});
  testSplitIntoChunks<int64_t>({1, 2, 3, 4, 5}, 2, {{{1, 2}}, {{3, 4}}, {{5}}});
  testSplitIntoChunks<int64_t>({2, 3, 4, 5}, 4, {{{2, 3, 4, 5}}});
  testSplitIntoChunks<int64_t>(
      {-66, 3, -66, 5}, 1, {{{-66}}, {{3}}, {{-66}}, {{5}}});
}

TEST_F(ArraySplitIntoChunksTest, evenSplit) {
  testSplitIntoChunks<int64_t>(
      {1, 2, 3, 4, 5, 6}, 2, {{{1, 2}}, {{3, 4}}, {{5, 6}}});
}

TEST_F(ArraySplitIntoChunksTest, chunkLargerThanArray) {
  testSplitIntoChunks<int64_t>({1, 2, 3}, 5, {{{1, 2, 3}}});
}

TEST_F(ArraySplitIntoChunksTest, chunkEqualsArrayLength) {
  testSplitIntoChunks<int64_t>({1, 2, 3}, 3, {{{1, 2, 3}}});
}

TEST_F(ArraySplitIntoChunksTest, chunkOfOne) {
  testSplitIntoChunks<int64_t>({1, 2, 3, 4}, 1, {{{1}}, {{2}}, {{3}}, {{4}}});
}

TEST_F(ArraySplitIntoChunksTest, emptyArray) {
  testSplitIntoChunks<int64_t>({}, 2, {});
}

TEST_F(ArraySplitIntoChunksTest, strings) {
  testSplitIntoChunks<std::string>({"a", "b", "c"}, 2, {{{"a", "b"}}, {{"c"}}});
  testSplitIntoChunks<std::string>(
      {"a", "b", "c", "d", "e"}, 2, {{{"a", "b"}}, {{"c", "d"}}, {{"e"}}});
  testSplitIntoChunks<std::string>(
      {"a", "b", "c", "d"}, 4, {{{"a", "b", "c", "d"}}});
  testSplitIntoChunks<std::string>(
      {"a", "b", "c", "d"}, 1, {{{"a"}}, {{"b"}}, {{"c"}}, {{"d"}}});
}

TEST_F(ArraySplitIntoChunksTest, doubles) {
  testSplitIntoChunks<double>({1.0, 2.0, 3.0}, 2, {{{1.0, 2.0}}, {{3.0}}});
  testSplitIntoChunks<double>(
      {1.0, 2.0, 3.0, 4.0, 5.0}, 2, {{{1.0, 2.0}}, {{3.0, 4.0}}, {{5.0}}});
  testSplitIntoChunks<double>(
      {2.0, 3.0, 4.0, 5.0}, 4, {{{2.0, 3.0, 4.0, 5.0}}});
  testSplitIntoChunks<double>(
      {1.0, 2.0, 3.0, 4.0}, 1, {{{1.0}}, {{2.0}}, {{3.0}}, {{4.0}}});
}

TEST_F(ArraySplitIntoChunksTest, nullElements) {
  testSplitIntoChunks<int64_t>(
      {std::nullopt, 1, std::nullopt, 2},
      2,
      {{{std::nullopt, 1}}, {{std::nullopt, 2}}});
}

TEST_F(ArraySplitIntoChunksTest, nullInputs) {
  auto input = makeNullableArrayVector<int64_t>(
      {std::optional<std::vector<std::optional<int64_t>>>(std::nullopt)});
  auto result = evaluate(
      "array_split_into_chunks(c0, 2::INTEGER)", makeRowVector({input}));
  ASSERT_TRUE(result->isNullAt(0));
}

TEST_F(ArraySplitIntoChunksTest, invalidChunkSize) {
  auto input = makeArrayVector<int64_t>({{1, 2, 3}});
  VELOX_ASSERT_THROW(
      evaluate(
          "array_split_into_chunks(c0, 0::INTEGER)", makeRowVector({input})),
      "Invalid slice size: 0. Size must be greater than zero.");
  VELOX_ASSERT_THROW(
      evaluate(
          "array_split_into_chunks(c0, -1::INTEGER)", makeRowVector({input})),
      "Scalar function signature is not supported");
}

TEST_F(ArraySplitIntoChunksTest, tooManyChunks) {
  // Create an array with 12001 elements.
  std::vector<std::optional<int64_t>> largeArray;
  largeArray.reserve(12'001);
  for (int i = 0; i < 12'001; ++i) {
    largeArray.push_back(i);
  }
  std::vector<std::optional<std::vector<std::optional<int64_t>>>> inputVec(
      {largeArray});
  auto input = makeNullableArrayVector<int64_t>(inputVec);
  VELOX_ASSERT_THROW(
      evaluate(
          "array_split_into_chunks(c0, 1::INTEGER)", makeRowVector({input})),
      "Cannot split array of size: 12001 into more than 10000 parts.");
}
} // namespace

} // namespace facebook::velox::functions
