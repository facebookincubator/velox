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

class ArrayCompactTest : public SparkFunctionBaseTest {
 protected:
  void testExpression(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }
};

TEST_F(ArrayCompactTest, basicArray) {
  const auto array = makeArrayVectorFromJson<int64_t>(
      {"[1, 2, 3, 4, null]",
       "[3, 4, 5, 6]",
       "[1, null, 2, null]",
       "[10, null, 20, 30]",
       "[null]",
       "[]",
       "null"});

  VectorPtr expected = makeArrayVectorFromJson<int64_t>(
      {"[1, 2, 3, 4]",
       "[3, 4, 5, 6]",
       "[1, 2]",
       "[10, 20, 30]",
       "[]",
       "[]",
       "null"});

  testExpression("array_compact(c0)", {array}, expected);
}

TEST_F(ArrayCompactTest, complexArray) {
  const auto array = makeNestedArrayVectorFromJson<int32_t>(
      {"[[1, 2, null], [null], [3, null, 4, null]]",
       "[[1, null, 2], null, [3, null, 4, null]]",
       "[[1, 2], [], [null, 3, 4]]",
       "[null, [1,2]]",
       "[[], [1, 2]]",
       "[[1, 2], null]",
       "[[1, 2], []]"});

  auto expected = makeNestedArrayVectorFromJson<int32_t>(
      {"[[1, 2, null], [null], [3, null, 4, null]]",
       "[[1, null, 2], [3, null, 4, null]]",
       "[[1, 2], [], [null, 3, 4]]",
       "[[1, 2]]",
       "[[], [1, 2]]",
       "[[1, 2]]",
       "[[1, 2], []]"});

  testExpression("array_compact(c0)", {array}, expected);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
