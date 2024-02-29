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

namespace facebook::velox::functions::sparksql::test {
namespace {

class ArrayUnionTest : public SparkFunctionBaseTest {
 protected:
  void testExpression(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }

  template <typename T>
  void testFloatArray() {
    const auto array1 = makeArrayVector<T>(
        {{1.99, 2.78, 3.98, 4.01},
         {3.89, 4.99, 5.13},
         {7.13, 8.91, std::numeric_limits<T>::quiet_NaN()},
         {10.02, 20.01, std::numeric_limits<T>::quiet_NaN()}});
    const auto array2 = makeArrayVector<T>(
        {{2.78, 4.01, 5.99},
         {3.89, 4.99, 5.13},
         {7.13, 8.91, std::numeric_limits<T>::quiet_NaN()},
         {40.99, 50.12}});

    VectorPtr expected;
    expected = makeArrayVector<T>({
        {1.99, 2.78, 3.98, 4.01, 5.99},
        {3.89, 4.99, 5.13},
        {7.13, 8.91, std::numeric_limits<T>::quiet_NaN()},
        {10.02, 20.01, std::numeric_limits<T>::quiet_NaN(), 40.99, 50.12},
    });
    testExpression("array_union(c0, c1)", {array1, array2}, expected);

    expected = makeArrayVector<T>({
        {2.78, 4.01, 5.99, 1.99, 3.98},
        {3.89, 4.99, 5.13},
        {7.13, 8.91, std::numeric_limits<T>::quiet_NaN()},
        {40.99, 50.12, 10.02, 20.01, std::numeric_limits<T>::quiet_NaN()},
    });
    testExpression("array_union(c0, c1)", {array2, array1}, expected);
  }
};

// Union two float or double arrays.
TEST_F(ArrayUnionTest, floatArray) {
  testFloatArray<float>();
  testFloatArray<double>();
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
