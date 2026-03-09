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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

using namespace facebook::velox::test;

class FilterTest : public SparkFunctionBaseTest {
 protected:
  void testFilter(
      const std::string& expression,
      const VectorPtr& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector({input}));
    assertEqualVectors(expected, result);
  }
};

// Test filter with element-only lambda: filter(array, x -> greaterthan(x, 0)).
TEST_F(FilterTest, elementOnly) {
  auto input = makeArrayVectorFromJson<int64_t>({
      "[1, -2, 3, -4, 5]",
      "[10, 20]",
      "[-1, -2]",
      "[]",
      "null",
  });

  auto expected = makeArrayVectorFromJson<int64_t>({
      "[1, 3, 5]",
      "[10, 20]",
      "[]",
      "[]",
      "null",
  });

  testFilter("filter(c0, x -> greaterthan(x, 0))", input, expected);
}

// Test filter with element-only lambda containing null elements.
TEST_F(FilterTest, elementOnlyWithNulls) {
  auto input = makeArrayVectorFromJson<int64_t>({
      "[1, null, 3]",
      "[null, null]",
      "[4, 5, null]",
  });

  // Null elements: predicate returns null, treated as false (filtered out).
  auto expected = makeArrayVectorFromJson<int64_t>({
      "[1, 3]",
      "[]",
      "[4, 5]",
  });

  testFilter("filter(c0, x -> greaterthan(x, 0))", input, expected);
}

// Test filter with element + index lambda:
// filter(array, (x, i) -> greaterthan(x, i)).
TEST_F(FilterTest, elementWithIndex) {
  auto input = makeArrayVectorFromJson<int64_t>({
      "[0, 2, 3]",
      "[10, 0, 30]",
      "[0, 1, 2]",
      "[]",
      "null",
  });

  // Expected: keep elements where x > i (0-based index)
  // [0>0=F, 2>1=T, 3>2=T] -> [2, 3]
  // [10>0=T, 0>1=F, 30>2=T] -> [10, 30]
  // [0>0=F, 1>1=F, 2>2=F] -> []
  // [] -> []
  // null -> null
  auto expected = makeArrayVectorFromJson<int64_t>({
      "[2, 3]",
      "[10, 30]",
      "[]",
      "[]",
      "null",
  });

  testFilter("filter(c0, (x, i) -> greaterthan(x, i))", input, expected);
}

// Test filter with index-only usage: filter(array, (x, i) -> lessthan(i, 2)).
TEST_F(FilterTest, indexOnly) {
  auto input = makeArrayVectorFromJson<int64_t>({
      "[100, 200, 300, 400]",
      "[1, 2]",
      "[42]",
      "[]",
  });

  // Expected: keep first 2 elements (index < 2)
  auto expected = makeArrayVectorFromJson<int64_t>({
      "[100, 200]",
      "[1, 2]",
      "[42]",
      "[]",
  });

  testFilter("filter(c0, (x, i) -> lessthan(i, 2))", input, expected);
}

// Test filter with element + index and null elements.
TEST_F(FilterTest, elementWithIndexAndNulls) {
  auto input = makeArrayVectorFromJson<int64_t>({
      "[1, null, 3]",
      "[null, 5]",
  });

  // For (x, i) -> greaterthan(x, i):
  // [1>0=T, null>1=null(F), 3>2=T] -> [1, 3]
  // [null>0=null(F), 5>1=T] -> [5]
  auto expected = makeArrayVectorFromJson<int64_t>({
      "[1, 3]",
      "[5]",
  });

  testFilter("filter(c0, (x, i) -> greaterthan(x, i))", input, expected);
}

// Test filter with isnotnull predicate.
TEST_F(FilterTest, isNotNull) {
  auto input = makeArrayVectorFromJson<int64_t>({
      "[0, null, 2, 3, null]",
      "[null, null]",
      "[1, 2, 3]",
  });

  auto expected = makeArrayVectorFromJson<int64_t>({
      "[0, 2, 3]",
      "[]",
      "[1, 2, 3]",
  });

  testFilter("filter(c0, x -> isnotnull(x))", input, expected);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
