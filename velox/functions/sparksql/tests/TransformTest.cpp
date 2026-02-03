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

class TransformTest : public SparkFunctionBaseTest {
 protected:
  void testTransform(
      const std::string& expression,
      const VectorPtr& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector({input}));
    assertEqualVectors(expected, result);
  }
};

// Test transform with element-only lambda: transform(array, x -> x * 2)
TEST_F(TransformTest, elementOnly) {
  auto input = makeArrayVectorFromJson<int64_t>({
      "[1, 2, 3]",
      "[4, 5]",
      "[6]",
      "[]",
      "null",
  });

  auto expected = makeArrayVectorFromJson<int64_t>({
      "[2, 4, 6]",
      "[8, 10]",
      "[12]",
      "[]",
      "null",
  });

  testTransform("transform(c0, x -> x * 2)", input, expected);
}

// Test transform with element-only lambda containing null elements.
TEST_F(TransformTest, elementOnlyWithNulls) {
  auto input = makeArrayVectorFromJson<int64_t>({
      "[1, null, 3]",
      "[null, null]",
      "[4, 5, null]",
  });

  auto expected = makeArrayVectorFromJson<int64_t>({
      "[2, null, 6]",
      "[null, null]",
      "[8, 10, null]",
  });

  testTransform("transform(c0, x -> x * 2)", input, expected);
}

// Test transform with element + index lambda: transform(array, (x, i) -> x + i)
TEST_F(TransformTest, elementWithIndex) {
  auto input = makeArrayVectorFromJson<int64_t>({
      "[10, 20, 30]",
      "[100, 200]",
      "[1000]",
      "[]",
      "null",
  });

  // Expected: element + index
  // [10+0, 20+1, 30+2] = [10, 21, 32]
  // [100+0, 200+1] = [100, 201]
  // [1000+0] = [1000]
  // [] = []
  // null = null
  auto expected = makeArrayVectorFromJson<int64_t>({
      "[10, 21, 32]",
      "[100, 201]",
      "[1000]",
      "[]",
      "null",
  });

  testTransform("transform(c0, (x, i) -> add(x, i))", input, expected);
}

// Test transform with index-only usage: transform(array, (x, i) -> i)
TEST_F(TransformTest, indexOnly) {
  auto input = makeArrayVectorFromJson<int64_t>({
      "[100, 200, 300]",
      "[1, 2]",
      "[42]",
      "[]",
  });

  // Expected: just the indices (as INTEGER/int32, matching Spark's IntegerType)
  auto expected = makeArrayVectorFromJson<int32_t>({
      "[0, 1, 2]",
      "[0, 1]",
      "[0]",
      "[]",
  });

  testTransform("transform(c0, (x, i) -> i)", input, expected);
}

// Test transform with index and null elements.
TEST_F(TransformTest, elementWithIndexAndNulls) {
  auto input = makeArrayVectorFromJson<int64_t>({
      "[1, null, 3]",
      "[null, 5]",
  });

  // Expected: element + index, null elements stay null
  // [1+0, null, 3+2] = [1, null, 5]
  // [null, 5+1] = [null, 6]
  auto expected = makeArrayVectorFromJson<int64_t>({
      "[1, null, 5]",
      "[null, 6]",
  });

  testTransform("transform(c0, (x, i) -> add(x, i))", input, expected);
}

// Test transform with index multiplication.
TEST_F(TransformTest, indexMultiplication) {
  auto input = makeArrayVectorFromJson<int64_t>({
      "[1, 2, 3, 4]",
      "[10, 20]",
  });

  // Expected: element * (index + 1)
  // [1*1, 2*2, 3*3, 4*4] = [1, 4, 9, 16]
  // [10*1, 20*2] = [10, 40]
  auto expected = makeArrayVectorFromJson<int64_t>({
      "[1, 4, 9, 16]",
      "[10, 40]",
  });

  testTransform(
      "transform(c0, (x, i) -> multiply(x, add(i, 1)))", input, expected);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
