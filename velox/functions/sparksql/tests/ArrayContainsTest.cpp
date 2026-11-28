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

class ArrayContainsTest : public SparkFunctionBaseTest {};

TEST_F(ArrayContainsTest, nestedNulls) {
  // Nested nulls are compared as values, i.e. null equals null.
  {
    auto arrays = makeNestedArrayVectorFromJson<int32_t>({
        "[[1, null]]",
        "[[1, null]]",
        "[[1, null], null]",
        "[[null]]",
    });
    auto elements = makeArrayVectorFromJson<int32_t>({
        "[1, null]",
        "[2, null]",
        "[1, 5]",
        "[1]",
    });
    auto expected =
        makeNullableFlatVector<bool>({true, false, std::nullopt, false});
    auto result =
        evaluate("array_contains(c0, c1)", makeRowVector({arrays, elements}));
    assertEqualVectors(expected, result);
  }
  {
    auto arrays = makeArrayVector(
        {0, 1},
        makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt, 1}),
             makeFlatVector<int64_t>({0, 1})}));
    auto elements = makeRowVector(
        {makeNullableFlatVector<int32_t>({std::nullopt, std::nullopt}),
         makeFlatVector<int64_t>({0, 1})});
    auto expected = makeFlatVector<bool>({true, false});
    auto result =
        evaluate("array_contains(c0, c1)", makeRowVector({arrays, elements}));
    assertEqualVectors(expected, result);
  }
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
