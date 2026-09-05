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

#include <limits>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using facebook::velox::test::assertEqualVectors;

namespace facebook::velox::functions::test {

namespace {

class ArrayLeastFrequentTest : public functions::test::FunctionBaseTest {
 protected:
  template <typename T>
  void testArrayLeastFrequent(
      const std::vector<std::vector<std::optional<T>>>& inputArrays,
      const std::vector<std::optional<std::vector<T>>>& expectedArrays) {
    auto input = makeNullableArrayVector<T>(inputArrays);
    auto result = evaluate<BaseVector>(
        "array_least_frequent(C0)", makeRowVector({input}));

    auto expected =
        makeNullableArrayVector<T>(toOptionalVectors(expectedArrays));
    assertEqualVectors(expected, result);
  }

  template <typename T>
  void testArrayLeastFrequentN(
      const std::vector<std::vector<std::optional<T>>>& inputArrays,
      int64_t n,
      const std::vector<std::optional<std::vector<T>>>& expectedArrays) {
    auto input = makeNullableArrayVector<T>(inputArrays);
    auto result = evaluate<BaseVector>(
        fmt::format("array_least_frequent(C0, {})", n), makeRowVector({input}));

    auto expected =
        makeNullableArrayVector<T>(toOptionalVectors(expectedArrays));
    assertEqualVectors(expected, result);
  }

 private:
  template <typename T>
  std::vector<std::vector<std::optional<T>>> toOptionalVectors(
      const std::vector<std::optional<std::vector<T>>>& input) {
    std::vector<std::vector<std::optional<T>>> result;
    for (const auto& vec : input) {
      if (vec.has_value()) {
        std::vector<std::optional<T>> row;
        for (const auto& v : vec.value()) {
          row.push_back(v);
        }
        result.push_back(row);
      } else {
        result.push_back({});
      }
    }
    return result;
  }
};

} // namespace

TEST_F(ArrayLeastFrequentTest, integerSingleArg) {
  // Basic: each element appears once, return smallest value.
  auto input =
      makeNullableArrayVector<int64_t>({{1, 0, 5}, {1, 1, 2, 2, 3}, {5}});
  auto expected = makeArrayVector<int64_t>({{0}, {3}, {5}});
  auto result =
      evaluate<BaseVector>("array_least_frequent(C0)", makeRowVector({input}));
  assertEqualVectors(expected, result);
}

TEST_F(ArrayLeastFrequentTest, integerWithN) {
  // [3, 2, 2, 6, 6, 1, 1] with n=3:
  // freq: 3->1, 2->2, 6->2, 1->2
  // sorted by (freq, value): (1,3), (2,1), (2,2), (2,6)
  // take 3: [3, 1, 2]
  auto input = makeNullableArrayVector<int64_t>({{3, 2, 2, 6, 6, 1, 1}});
  auto expected = makeArrayVector<int64_t>({{3, 1, 2}});
  auto result = evaluate<BaseVector>(
      "array_least_frequent(C0, 3)", makeRowVector({input}));
  assertEqualVectors(expected, result);
}

TEST_F(ArrayLeastFrequentTest, nGreaterThanDistinct) {
  // n=10 but only 3 distinct elements -> returns all 3 sorted.
  auto input = makeNullableArrayVector<int64_t>({{3, 2, 1}});
  auto expected = makeArrayVector<int64_t>({{1, 2, 3}});
  auto result = evaluate<BaseVector>(
      "array_least_frequent(C0, 10)", makeRowVector({input}));
  assertEqualVectors(expected, result);
}

TEST_F(ArrayLeastFrequentTest, nullHandling) {
  // Null elements are stripped.
  // [1, null, 1] -> only non-null: [1,1] -> freq: 1->2 -> result: [1]
  auto input = makeNullableArrayVector<int64_t>({{1, std::nullopt, 1}});
  auto expected = makeArrayVector<int64_t>({{1}});
  auto result =
      evaluate<BaseVector>("array_least_frequent(C0)", makeRowVector({input}));
  assertEqualVectors(expected, result);

  // [1, null, 1] with n=2 -> only 1 distinct non-null element -> [1]
  auto input2 = makeNullableArrayVector<int64_t>({{1, std::nullopt, 1}});
  auto expected2 = makeArrayVector<int64_t>({{1}});
  auto result2 = evaluate<BaseVector>(
      "array_least_frequent(C0, 2)", makeRowVector({input2}));
  assertEqualVectors(expected2, result2);
}

TEST_F(ArrayLeastFrequentTest, allNulls) {
  // All-null elements -> NULL result.
  std::vector<std::vector<std::optional<int64_t>>> data = {
      {std::nullopt, std::nullopt}};
  auto input = makeNullableArrayVector<int64_t>(data);
  auto result =
      evaluate<BaseVector>("array_least_frequent(C0)", makeRowVector({input}));
  ASSERT_TRUE(result->isNullAt(0));
}

TEST_F(ArrayLeastFrequentTest, nullArray) {
  // Null array input -> NULL result.
  using O = std::optional<int64_t>;
  std::vector<std::optional<std::vector<O>>> data = {
      {{O{1}, O{2}}}, std::nullopt};
  auto input = makeNullableArrayVector<int64_t>(data);
  auto result =
      evaluate<BaseVector>("array_least_frequent(C0)", makeRowVector({input}));
  ASSERT_FALSE(result->isNullAt(0));
  ASSERT_TRUE(result->isNullAt(1));
}

TEST_F(ArrayLeastFrequentTest, emptyArray) {
  // Empty array -> NULL (no non-null elements).
  auto input = makeArrayVector<int64_t>({{}});
  auto result =
      evaluate<BaseVector>("array_least_frequent(C0)", makeRowVector({input}));
  ASSERT_TRUE(result->isNullAt(0));
}

TEST_F(ArrayLeastFrequentTest, nZero) {
  // n=0 -> empty array, not null.
  auto input = makeNullableArrayVector<int64_t>({{1, 2, 3}});
  auto expected = makeArrayVector<int64_t>({{}});
  auto result = evaluate<BaseVector>(
      "array_least_frequent(C0, 0)", makeRowVector({input}));
  assertEqualVectors(expected, result);
  ASSERT_FALSE(result->isNullAt(0));
}

TEST_F(ArrayLeastFrequentTest, nNegative) {
  auto input = makeNullableArrayVector<int64_t>({{1, 2, 3}});
  VELOX_ASSERT_THROW(
      evaluate<BaseVector>(
          "array_least_frequent(C0, -1)", makeRowVector({input})),
      "n must be greater than or equal to 0");
}

TEST_F(ArrayLeastFrequentTest, tieBreaking) {
  // All elements have same frequency -> sort by value ascending.
  // [3, 2, 1] -> freq: all 1 -> sorted: [1, 2, 3]
  auto input = makeNullableArrayVector<int64_t>({{3, 2, 1}});

  // Single arg: returns smallest value.
  auto expected1 = makeArrayVector<int64_t>({{1}});
  auto result1 =
      evaluate<BaseVector>("array_least_frequent(C0)", makeRowVector({input}));
  assertEqualVectors(expected1, result1);

  // n=3: returns all sorted by value.
  auto expected3 = makeArrayVector<int64_t>({{1, 2, 3}});
  auto result3 = evaluate<BaseVector>(
      "array_least_frequent(C0, 3)", makeRowVector({input}));
  assertEqualVectors(expected3, result3);
}

TEST_F(ArrayLeastFrequentTest, floatWithNaN) {
  // NaN participates as a valid element.
  auto nan = std::numeric_limits<double>::quiet_NaN();
  auto input = makeNullableArrayVector<double>({{nan, nan, 1.0, 2.0, 2.0}});
  // freq: nan->2, 1.0->1, 2.0->2
  // sorted by (freq, value): (1, 1.0), (2, 2.0), (2, nan) [NaN > all]
  auto expected = makeArrayVector<double>({{1.0}});
  auto result =
      evaluate<BaseVector>("array_least_frequent(C0)", makeRowVector({input}));
  assertEqualVectors(expected, result);

  // With n=3: [1.0, 2.0, nan]
  auto expected3 = makeArrayVector<double>({{1.0, 2.0, nan}});
  auto result3 = evaluate<BaseVector>(
      "array_least_frequent(C0, 3)", makeRowVector({input}));
  assertEqualVectors(expected3, result3);
}

TEST_F(ArrayLeastFrequentTest, varcharArray) {
  auto input = makeNullableArrayVector<StringView>(
      {{"a"_sv, "b"_sv, "b"_sv, "c"_sv, "c"_sv, "c"_sv}});
  // freq: a->1, b->2, c->3 -> least frequent: [a]
  auto expected = makeArrayVector<StringView>({{"a"_sv}});
  auto result =
      evaluate<BaseVector>("array_least_frequent(C0)", makeRowVector({input}));
  assertEqualVectors(expected, result);
}

TEST_F(ArrayLeastFrequentTest, varcharArrayWithN) {
  auto input = makeNullableArrayVector<StringView>(
      {{"a"_sv, "b"_sv, "b"_sv, "c"_sv, "c"_sv, "c"_sv}});
  // freq: a->1, b->2, c->3 -> n=2: [a, b]
  auto expected = makeArrayVector<StringView>({{"a"_sv, "b"_sv}});
  auto result = evaluate<BaseVector>(
      "array_least_frequent(C0, 2)", makeRowVector({input}));
  assertEqualVectors(expected, result);
}

TEST_F(ArrayLeastFrequentTest, singleElement) {
  std::vector<std::vector<std::optional<int64_t>>> data = {{5}};
  auto input = makeNullableArrayVector<int64_t>(data);
  auto expected = makeArrayVector<int64_t>({{5}});
  auto result =
      evaluate<BaseVector>("array_least_frequent(C0)", makeRowVector({input}));
  assertEqualVectors(expected, result);
}

} // namespace facebook::velox::functions::test
