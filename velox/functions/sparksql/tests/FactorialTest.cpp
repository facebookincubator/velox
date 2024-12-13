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

#include "velox/functions/sparksql/Factorial.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class FactorialTest : public SparkFunctionBaseTest {
 protected:
  void testFactorial(
      const VectorPtr& input,
      const VectorPtr& expected) {
    auto result = evaluate<SimpleVector<int64_t>>(
        "factorial(c0)", makeRowVector({input}));
    velox::test::assertEqualVectors(expected, result);
  }
};

TEST_F(FactorialTest, basic) {
  auto input = makeFlatVector<int32_t>({0, 1, 2, 5, 10, 15, 20});
  auto expected = makeFlatVector<int64_t>(
      {1, 1, 2, 120, 3628800, 1307674368000L, 2432902008176640000L});
  testFactorial(input, expected);
}

TEST_F(FactorialTest, nullInput) {
  auto input = makeNullableFlatVector<int32_t>(
      {0, std::nullopt, 5, 20, std::nullopt});
  auto expected = makeNullableFlatVector<int64_t>(
      {1, std::nullopt, 120, 2432902008176640000L, std::nullopt});
  testFactorial(input, expected);
}

TEST_F(FactorialTest, outOfRangeInput) {
  auto input = makeFlatVector<int32_t>({-1, 21, -5, 25});
  auto expected = makeNullConstant(TypeKind::BIGINT, input->size());
  testFactorial(input, expected);
}

TEST_F(FactorialTest, mixedInputs) {
  auto input = makeNullableFlatVector<int32_t>(
      {3, 5, std::nullopt, 25, -3, 10, 15});
  auto expected = makeNullableFlatVector<int64_t>(
      {6, 120, std::nullopt, std::nullopt, std::nullopt, 3628800, 1307674368000L});
  testFactorial(input, expected);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
