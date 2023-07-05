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
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions::test;

namespace {

class ArrayRemoveTest : public FunctionBaseTest {
 protected:
  void testExpression(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }
};

/// Remove all elements that equal element from array
TEST_F(ArrayRemoveTest, removeArray) {
  const auto arrayVector = makeArrayVector<std::string>(
      {{"1", "2", "3", "4"},
       {"3", "4", "5"},
       {"7", "8", "9"},
       {"10", "20", "30"}});
  const auto elementVector = makeFlatVector<std::string>({"2", "3", "9", "20"});
  VectorPtr expected;

  expected = makeArrayVector<std::string>({
      {"1", "3", "4"},
      {"4", "5"},
      {"7", "8"},
      {"10", "30"},
  });
  testExpression(
      "array_remove(c0, c1)", {arrayVector, elementVector}, expected);
}

/// Remove all elements that equal element from array with null
TEST_F(ArrayRemoveTest, removeArrayWithNull) {
  const auto arrayVector = makeNullableArrayVector<int64_t>(
      {{1, 2, std::nullopt, 4},
       {3, 4, 5},
       {7, 8, 9},
       {10, 20, 30, std::nullopt}});
  const auto elementVector = makeFlatVector<int64_t>({2, 3, 9, 20});
  VectorPtr expected;

  expected = makeNullableArrayVector<int64_t>({
      {1, std::nullopt, 4},
      {4, 5},
      {7, 8},
      {10, 30, std::nullopt},
  });
  testExpression(
      "array_remove(c0, c1)", {arrayVector, elementVector}, expected);
}

/// Remove all elements that equal element from char array with/without NULL
TEST_F(ArrayRemoveTest, removeCharArray) {
  const auto arrayVector = makeNullableArrayVector<StringView>(
      {{"aa"_sv, "bb"_sv, std::nullopt, "cc"_sv},
       {"aa"_sv, "cc"_sv, "dd"_sv},
       {"aa"_sv, "ee"_sv, "ff"_sv},
       {"aa"_sv, "dd"_sv, "ff"_sv, std::nullopt}});
  const auto elementVector =
      makeFlatVector<StringView>({"aa"_sv, "cc"_sv, "ff"_sv, "ff"_sv});
  VectorPtr expected;

  expected = makeNullableArrayVector<StringView>({
      {"bb"_sv, std::nullopt, "cc"_sv},
      {"aa"_sv, "dd"_sv},
      {"aa"_sv, "ee"_sv},
      {"aa"_sv, "dd"_sv, std::nullopt},
  });
  testExpression(
      "array_remove(c0, c1)", {arrayVector, elementVector}, expected);
}

/// Remove complex type.
TEST_F(ArrayRemoveTest, complexTypes) {
  // [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
  auto baseVector = makeArrayVector<int64_t>(
      {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}});

  // [[1, 1], [2, 2], [3, 3], [4, 4]]
  // [[5, 5], [6, 6]]
  auto arrayVector = makeArrayVector({0, 4}, baseVector);

  // [[2, 2]]
  // [[5, 5]]
  auto elementVector = makeArrayVector<int64_t>({{2, 2}, {5, 5}});

  // [[1, 1], [3, 3], [4, 4]]
  // [[6, 6]]
  auto expected = makeArrayVector(
      {0, 3}, makeArrayVector<int64_t>({{1, 1}, {3, 3}, {4, 4}, {6, 6}}));
  testExpression(
      "array_remove(c0, c1)", {arrayVector, elementVector}, expected);
}

/// Remove complex type with NULL.
TEST_F(ArrayRemoveTest, complexTypesWithNull) {
  // Make a vector with a NULL.
  // [[1, 2], null, [3, 2], [2, 2, 3], [2, 1, 5]]
  auto baseVectorWithNull = makeNullableArrayVector<int64_t>(
      {{1, 2}, {std::nullopt}, {3, 2}, {2, 2, 3}, {2, 1, 5}});

  // [[1, 2], null, [3, 2]]
  // [[2, 2, 3], [2, 1, 5]]
  auto arrayVector = makeArrayVector({0, 3}, baseVectorWithNull);

  // [[1, 9]]
  // [[2, 1, 5]]
  auto elementVector = makeNullableArrayVector<int64_t>({{1, 9}, {2, 1, 5}});

  // [[1, 2], null, [3, 2]]
  // [[2, 2, 3]]
  auto expected = makeArrayVector(
      {0, 3},
      makeNullableArrayVector<int64_t>(
          {{1, 2}, {std::nullopt}, {3, 2}, {2, 2, 3}}));

  testExpression(
      "array_remove(c0, c1)", {arrayVector, elementVector}, expected);
}

//  Remove array when element is null
TEST_F(ArrayRemoveTest, removeArrayNullElement) {
  auto input = makeNullableArrayVector<std::int64_t>(
      {{1, 2, 3}, {1, std::nullopt, std::nullopt, 4}});
  auto elementVector =
      makeNullableFlatVector<std::int64_t>({std::nullopt, std::nullopt});
  auto expected =
      makeNullableArrayVector<std::int64_t>({std::nullopt, std::nullopt});
  testExpression("array_remove(c0, c1)", {input, elementVector}, expected);
}
} // namespace
