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
namespace {

class ArraysOverlapTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  void testOverlap(
      const std::vector<std::optional<std::vector<std::optional<T>>>>& array1,
      const std::vector<std::optional<std::vector<std::optional<T>>>>& array2,
      const std::vector<std::optional<bool>>& expected) {
    auto input1 = makeNullableArrayVector(array1);
    auto input2 = makeNullableArrayVector(array2);
    auto result =
        evaluate("arrays_overlap(c0, c1)", makeRowVector({input1, input2}));
    auto expectedVector = makeNullableFlatVector(expected);
    assertEqualVectors(expectedVector, result);
  }
};

TEST_F(ArraysOverlapTest, intArrays) {
  // Basic overlap.
  testOverlap<int32_t>({{{{1, 2, 3}}}}, {{{{3, 4, 5}}}}, {{true}});

  // No overlap.
  testOverlap<int32_t>({{{{1, 2, 3}}}}, {{{{4, 5, 6}}}}, {{false}});
}

TEST_F(ArraysOverlapTest, nullHandling) {
  // No overlap but both have NULLs -> NULL.
  testOverlap<int32_t>(
      {{{{1, 2, std::nullopt}}}}, {{{{4, 5, std::nullopt}}}}, {{std::nullopt}});

  // Overlap found despite NULLs -> true.
  testOverlap<int32_t>(
      {{{{1, std::nullopt, 3}}}}, {{{{3, std::nullopt, 5}}}}, {{true}});

  // NULL array -> NULL.
  testOverlap<int32_t>({{std::nullopt}}, {{{{1, 2, 3}}}}, {{std::nullopt}});
}

TEST_F(ArraysOverlapTest, emptyArrays) {
  // Empty arrays -> false.
  auto emptyArray = makeArrayVector<int32_t>({{}, {}});
  auto result = evaluate(
      "arrays_overlap(c0, c1)",
      makeRowVector({emptyArray, makeArrayVector<int32_t>({{}, {}})}));
  assertEqualVectors(makeNullableFlatVector<bool>({false, false}), result);

  // Empty vs non-empty -> false.
  auto nonEmpty = makeArrayVector<int32_t>({{1, 2, 3}, {4, 5}});
  result =
      evaluate("arrays_overlap(c0, c1)", makeRowVector({emptyArray, nonEmpty}));
  assertEqualVectors(makeNullableFlatVector<bool>({false, false}), result);
}

TEST_F(ArraysOverlapTest, stringArrays) {
  testOverlap<StringView>(
      {{{{StringView("a"), StringView("b")}}}},
      {{{{StringView("b"), StringView("c")}}}},
      {{true}});

  testOverlap<StringView>(
      {{{{StringView("a")}}}},
      {{{{StringView("c"), StringView("d")}}}},
      {{false}});
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
