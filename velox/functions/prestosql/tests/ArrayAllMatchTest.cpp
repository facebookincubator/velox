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

class ArrayAllMatchTest : public functions::test::FunctionBaseTest {};

TEST_F(ArrayAllMatchTest, basic) {
  auto input = makeNullableArrayVector<int64_t>({{std::nullopt, 2, 3}});
  auto result = evaluate<SimpleVector<bool>>(
      "all_match(c0, x -> (x > 1))", makeRowVector({input}));
  auto expectedResult = makeNullableFlatVector<bool>({std::nullopt});
  assertEqualVectors(expectedResult, result);

  result = evaluate<SimpleVector<bool>>(
      "all_match(c0, x -> (x is null))", makeRowVector({input}));
  expectedResult = makeNullableFlatVector<bool>({false});
  assertEqualVectors(expectedResult, result);

  input = makeNullableArrayVector<bool>({{true, false}});
  result = evaluate<SimpleVector<bool>>(
      "all_match(c0, x -> x)", makeRowVector({input}));
  expectedResult = makeNullableFlatVector<bool>({false});
  assertEqualVectors(expectedResult, result);

  auto emptyInput = makeArrayVector<int32_t>({{}});
  result = evaluate<SimpleVector<bool>>(
      "all_match(c0, x -> (x > 1))", makeRowVector({emptyInput}));
  expectedResult = makeNullableFlatVector<bool>({true});
  assertEqualVectors(expectedResult, result);

  result = evaluate<SimpleVector<bool>>(
      "all_match(c0, x -> (x < 1))", makeRowVector({emptyInput}));
  expectedResult = makeNullableFlatVector<bool>({true});
  assertEqualVectors(expectedResult, result);
}

TEST_F(ArrayAllMatchTest, complexTypes) {
  auto baseVector =
      makeArrayVector<int64_t>({{1, 2, 3}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {}});
  // Create an array of array vector using above base vector using offsets.
  // [
  //  [[1, 2, 3]],
  //  [[2, 2], [3, 3], [4, 4], [5, 5]],
  //  [[]]
  // ]
  auto arrayOfArrays = makeArrayVector({0, 1, 5}, baseVector);
  auto result = evaluate<SimpleVector<bool>>(
      "all_match(c0, x -> (cardinality(x) > 0))",
      makeRowVector({arrayOfArrays}));
  auto expectedResult = makeNullableFlatVector<bool>({true, true, false});
  assertEqualVectors(expectedResult, result);

  // Create an array of array vector using above base vector using offsets.
  // [
  //  [[1, 2, 3]],  cardinalities is 3
  //  [[2, 2], [3, 3], [4, 4], [5, 5]], all cardinalities is 2
  //  [[]],
  //  null
  // ]
  arrayOfArrays = makeArrayVector({0, 1, 5, 6}, baseVector, {3});
  result = evaluate<SimpleVector<bool>>(
      "all_match(c0, x -> (cardinality(x) > 2))",
      makeRowVector({arrayOfArrays}));
  expectedResult =
      makeNullableFlatVector<bool>({true, false, false, std::nullopt});
  assertEqualVectors(expectedResult, result);
}

TEST_F(ArrayAllMatchTest, bigints) {
  auto input = makeNullableArrayVector<int64_t>(
      {{},
       {2},
       {std::numeric_limits<int64_t>::max()},
       {std::numeric_limits<int64_t>::min()},
       {std::nullopt, std::nullopt}, // return null if all is null
       {2,
        std::nullopt}, // return null if one or more is null and others matched
       {1, std::nullopt, 2}}); // return false if one is not matched
  auto result = evaluate<SimpleVector<bool>>(
      "all_match(c0, x -> (x % 2 = 0))", makeRowVector({input}));

  auto expectedResult = makeNullableFlatVector<bool>(
      {true, true, false, true, std::nullopt, std::nullopt, false});
  assertEqualVectors(expectedResult, result);
}

TEST_F(ArrayAllMatchTest, strings) {
  auto input = makeNullableArrayVector<StringView>(
      {{}, {"abc"}, {"ab", "abc"}, {std::nullopt}});
  auto result = evaluate<SimpleVector<bool>>(
      "all_match(c0, x -> (x = 'abc'))", makeRowVector({input}));

  auto expectedResult =
      makeNullableFlatVector<bool>({true, true, false, std::nullopt});
  assertEqualVectors(expectedResult, result);
}

TEST_F(ArrayAllMatchTest, doubles) {
  auto input =
      makeNullableArrayVector<double>({{}, {1.2}, {3.0, 0}, {std::nullopt}});
  auto result = evaluate<SimpleVector<bool>>(
      "all_match(c0, x -> (x > 1.1))", makeRowVector({input}));

  auto expectedResult =
      makeNullableFlatVector<bool>({true, true, false, std::nullopt});
  assertEqualVectors(expectedResult, result);
}

TEST_F(ArrayAllMatchTest, errorSuppress) {
  auto input =
      makeNullableArrayVector<int8_t>({{2, 5, 0}, {5, std::nullopt, 0}});
  auto result = evaluate<SimpleVector<bool>>(
      "all_match(c0, x -> ((10 / x) > 2))", makeRowVector({input}));

  auto expectedResult = makeFlatVector<bool>({false, false});
  assertEqualVectors(expectedResult, result);
}

TEST_F(ArrayAllMatchTest, errorReThrow) {
  static constexpr std::string_view kErrorMessage{"division by zero"};

  VELOX_ASSERT_THROW(
      evaluate<SimpleVector<bool>>(
          "all_match(c0, x -> ((10 / x) > 2))",
          makeRowVector({makeArrayVector<int8_t>({{1, 0}})})),
      kErrorMessage);
  VELOX_ASSERT_THROW(
      evaluate<SimpleVector<bool>>(
          "all_match(c0, x -> ((10 / x) > 2))",
          makeRowVector(
              {makeNullableArrayVector<int8_t>({{1, 0, std::nullopt}})})),
      kErrorMessage);
}

TEST_F(ArrayAllMatchTest, withTrys) {
  auto result = evaluate<SimpleVector<bool>>(
      "TRY(all_match(c0, x -> ((10 / x) > 2)))",
      makeRowVector({makeNullableArrayVector<int8_t>({{1, 0}, {2}, {6}})}));
  auto expectedResult =
      makeNullableFlatVector<bool>({std::nullopt, true, false});
  assertEqualVectors(expectedResult, result);

  result = evaluate<SimpleVector<bool>>(
      "all_match(c0, x -> (TRY((10 / x) > 2)))",
      makeRowVector({makeArrayVector<int8_t>({{1, 0}, {1}, {6}})}));
  expectedResult = makeNullableFlatVector<bool>({std::nullopt, true, false});
  assertEqualVectors(expectedResult, result);
}
