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

#include <optional>
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/vector/tests/TestingDictionaryArrayElementsFunction.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class ArrayIntersectExceptTest : public SparkFunctionBaseTest {
 protected:
  void testExpr(
      const VectorPtr& expected,
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    auto result = evaluate<ArrayVector>(expression, makeRowVector(input));
    assertEqualVectors(expected, result);

    // Also test using dictionary encodings.
    if (input.size() == 2) {
      // Wrap first column in a dictionary: repeat each row twice. Wrap second
      // column in the same dictionary, then flatten to prevent peeling of
      // encodings. Wrap the expected result in the same dictionary.
      // The expression evaluation on both dictionary inputs should result in
      // the dictionary of the expected result vector.
      auto newSize = input[0]->size() * 2;
      auto indices = makeIndices(newSize, [](auto row) { return row / 2; });
      auto firstDict = wrapInDictionary(indices, newSize, input[0]);
      auto secondFlat = flatten(wrapInDictionary(indices, newSize, input[1]));

      auto dictResult = evaluate<ArrayVector>(
          expression, makeRowVector({firstDict, secondFlat}));
      auto dictExpected = wrapInDictionary(indices, newSize, expected);
      assertEqualVectors(dictExpected, dictResult);
    }
  }

  template <typename T>
  void testArrayExceptFloatingPoint() {
    auto expected = makeNullableArrayVector<T>({
        {1.0001, 3.03, std::nullopt, 4.00004},
        {2.02, 1},
        {8.0001, std::nullopt},
        {std::numeric_limits<T>::max()},
        {},
    });
    testExpr(expected, "array_except(C0, C1)", getInputVectors<T>());

    expected = makeNullableArrayVector<T>({
        {1.0, 4.0},
        {2.0199, 1.000001},
        {1.0001, -2.02, 8.00099},
        {},
        {},
    });
    testExpr(expected, "array_except(C1, C0)", getInputVectors<T>());
  }

  template <typename T>
  void testArrayIntersectFloatingPoint() {
    auto expected = makeNullableArrayVector<T>({
        {-2.0},
        {std::numeric_limits<T>::min(), -2.001},
        {std::numeric_limits<T>::max()},
        {9.0009, std::numeric_limits<T>::infinity()},
        {std::numeric_limits<T>::quiet_NaN(), 9.0009},
    });
    testExpr(expected, "array_intersect(C0, C1)", getInputVectors<T>());

    expected = makeNullableArrayVector<T>({
        {-2.0},
        {std::numeric_limits<T>::min(), -2.001},
        {std::numeric_limits<T>::max()},
        {9.0009, std::numeric_limits<T>::infinity()},
        {9.0009, std::numeric_limits<T>::quiet_NaN()},
    });
    testExpr(expected, "array_intersect(C1, C0)", getInputVectors<T>());
  }

 private:
  template <typename T>
  std::vector<VectorPtr> getInputVectors() {
    const auto array1 = makeNullableArrayVector<T>({
        {1.0001, -2.0, 3.03, std::nullopt, 4.00004},
        {std::numeric_limits<T>::min(), 2.02, -2.001, 1},
        {std::numeric_limits<T>::max(), 8.0001, std::nullopt},
        {9.0009,
         std::numeric_limits<T>::infinity(),
         std::numeric_limits<T>::max()},
        {std::numeric_limits<T>::quiet_NaN(), 9.0009},
    });
    const auto array2 = makeNullableArrayVector<T>({
        {1.0, -2.0, 4.0},
        {std::numeric_limits<T>::min(), 2.0199, -2.001, 1.000001},
        {1.0001, -2.02, std::numeric_limits<T>::max(), 8.00099},
        {9.0009, std::numeric_limits<T>::infinity()},
        {9.0009, std::numeric_limits<T>::quiet_NaN()},
    });
    return {array1, array2};
  }
};

TEST_F(ArrayIntersectExceptTest, except) {
  testArrayExceptFloatingPoint<float>();
  testArrayExceptFloatingPoint<double>();
}

TEST_F(ArrayIntersectExceptTest, intersect) {
  testArrayIntersectFloatingPoint<float>();
  testArrayIntersectFloatingPoint<double>();
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
