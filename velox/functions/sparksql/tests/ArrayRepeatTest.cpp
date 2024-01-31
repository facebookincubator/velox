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

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::sparksql::test;

namespace {

class ArrayRepeatTest : public SparkFunctionBaseTest {
 protected:
  void testExpression(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }

  void testExpressionWithError(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const std::string& expectedError) {
    VELOX_ASSERT_THROW(
        evaluate(expression, makeRowVector(input)), expectedError);
  }
};
} // namespace

TEST_F(ArrayRepeatTest, arrayRepeat) {
  const auto elementVector = makeNullableFlatVector<float>(
      {0.0, -2.0, 3.333333, 4.0004, std::nullopt, 5.12345});
  auto countVector =
      makeNullableFlatVector<int32_t>({1, 2, 3, 0, 4, std::nullopt});
  VectorPtr expected;

  expected = makeNullableArrayVector<float>({
      {{0.0}},
      {{-2.0, -2.0}},
      {{3.333333, 3.333333, 3.333333}},
      {{}},
      {{std::nullopt, std::nullopt, std::nullopt, std::nullopt}},
      std::nullopt,
  });
  testExpression(
      "array_repeat(C0, C1)", {elementVector, countVector}, expected);

  expected = makeArrayVector<float>({{}, {}, {}, {}, {}, {}});
  // Test all zero count.
  countVector = makeNullableFlatVector<int32_t>({0, 0, 0, 0, 0, 0});
  testExpression(
      "array_repeat(C0, C1)", {elementVector, countVector}, expected);

  // Test negative count.
  countVector = makeNullableFlatVector<int32_t>({-1, -2, -3, -5, 0, -100});
  testExpression(
      "array_repeat(C0, C1)", {elementVector, countVector}, expected);

  // Test using a constant as the count argument.
  testExpression("array_repeat(C0, '-5'::INTEGER)", {elementVector}, expected);

  // Test using a null constant as the count argument.
  expected = BaseVector::createNullConstant(ARRAY(REAL()), 6, pool());
  testExpression("array_repeat(C0, null::INTEGER)", {elementVector}, expected);

  // Test using a non-null constant as the count argument.
  expected = makeNullableArrayVector<float>({
      {0.0, 0.0, 0.0},
      {-2.0, -2.0, -2.0},
      {3.333333, 3.333333, 3.333333},
      {4.0004, 4.0004, 4.0004},
      {std::nullopt, std::nullopt, std::nullopt},
      {5.12345, 5.12345, 5.12345},
  });
  testExpression("array_repeat(C0, '3'::INTEGER)", {elementVector}, expected);

  expected = makeArrayVector<float>({{}, {}, {}, {}, {}, {}});
  testExpression("array_repeat(C0, '0'::INTEGER)", {elementVector}, expected);
}

TEST_F(ArrayRepeatTest, arrayRepeatWithInvalidCount) {
  const auto elementVector =
      makeNullableFlatVector<float>({0.0, 2.0, 3.333333});

  VectorPtr countVector;

  countVector = makeNullableFlatVector<int32_t>({2147483647, 3});
  testExpressionWithError(
      "array_repeat(C0, C1)",
      {elementVector, countVector},
      "(2147483647 vs. 2147483632) Count argument of array_repeat function must be less than or equal to (MAX_INT32 - 15)");

  // Test using an invalid constant as the count argument.
  testExpressionWithError(
      "array_repeat(C0, '2147483647'::INTEGER)",
      {elementVector},
      "(2147483647 vs. 2147483632) Count argument of array_repeat function must be less than or equal to (MAX_INT32 - 15)");
}
