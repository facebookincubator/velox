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

using namespace facebook::velox;
using namespace facebook::velox::test;

class ArrayAllMatchTest : public functions::test::FunctionBaseTest {};

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
      "all_match(c0, x -> (x == 'abc'))", makeRowVector({input}));

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
