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

class ArrayPrependTest : public SparkFunctionBaseTest {
 protected:
  void testExpression(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }
};

TEST_F(ArrayPrependTest, intArrays) {
  const auto arrayVector = makeArrayVector<int64_t>(
      {{1, 2, 3, 4}, {3, 4, 5}, {7, 8, 9}, {10, 20, 30}});
  const auto elementVector = makeFlatVector<int64_t>({11, 22, 33, 44});

  VectorPtr expected = makeArrayVector<int64_t>({
      {11, 1, 2, 3, 4},
      {22, 3, 4, 5},
      {33, 7, 8, 9},
      {44, 10, 20, 30},
  });
  testExpression(
      "array_prepend(c0, c1)", {arrayVector, elementVector}, expected);
}

TEST_F(ArrayPrependTest, nullArrays) {
  const auto arrayVector = makeArrayVectorFromJson<int64_t>(
      {"[1, 2, 3, null]", "[3, 4, 5]", "null", "[10, 20, null]"});

  const auto elementVector = makeNullableFlatVector<int64_t>(
      {11, std::nullopt, std::nullopt, std::nullopt});

  VectorPtr expected = makeArrayVectorFromJson<int64_t>(
      {"[11, 1, 2, 3, null]",
       "[null, 3, 4, 5]",
       "null",
       "[null, 10, 20, null]"});
  testExpression(
      "array_prepend(c0, c1)", {arrayVector, elementVector}, expected);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
