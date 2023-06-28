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
using namespace facebook::velox::functions::test;

namespace facebook::velox::functions::sparksql::test {
namespace {
class DecimalVectorFunctionTest : public SparkFunctionBaseTest {
 protected:
  template <TypeKind KIND>
  void testDecimalExpr(
      const VectorPtr& expected,
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    using EvalType = typename velox::TypeTraits<KIND>::NativeType;
    auto result =
        evaluate<SimpleVector<EvalType>>(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
    testOpDictVectors<EvalType>(expression, expected, input);
  }

  template <typename T>
  void testOpDictVectors(
      const std::string& operation,
      const VectorPtr& expected,
      const std::vector<VectorPtr>& flatVector) {
    // Dictionary vectors as arguments.
    auto newSize = flatVector[0]->size() * 2;
    std::vector<VectorPtr> dictVectors;
    for (auto i = 0; i < flatVector.size(); ++i) {
      auto indices = makeIndices(newSize, [&](int row) { return row / 2; });
      dictVectors.push_back(
          VectorTestBase::wrapInDictionary(indices, newSize, flatVector[i]));
    }
    auto resultIndices = makeIndices(newSize, [&](int row) { return row / 2; });
    auto expectedResultDictionary =
        VectorTestBase::wrapInDictionary(resultIndices, newSize, expected);
    auto actual =
        evaluate<SimpleVector<T>>(operation, makeRowVector(dictVectors));
    assertEqualVectors(expectedResultDictionary, actual);
  }
};

TEST_F(DecimalVectorFunctionTest, makeDecimal) {
  testDecimalExpr<TypeKind::BIGINT>(
      {makeFlatVector<int64_t>({1111, -1112, 9999, 0}, DECIMAL(5, 1))},
      "make_decimal_by_unscaled_value(c0, c1, true)",
      {makeFlatVector<int64_t>({1111, -1112, 9999, 0}),
       makeConstant<int64_t>(0, 4, DECIMAL(5, 1))});
  testDecimalExpr<TypeKind::HUGEINT>(
      {makeFlatVector<int128_t>(
          {11111111, -11112112, 99999999, DecimalUtil::kShortDecimalMax + 1},
          DECIMAL(38, 19))},
      "make_decimal_by_unscaled_value(c0, c1, true)",
      {makeFlatVector<int64_t>(
           {11111111, -11112112, 99999999, DecimalUtil::kShortDecimalMax + 1}),
       makeConstant<int128_t>(0, 4, DECIMAL(38, 19))});

  testDecimalExpr<TypeKind::BIGINT>(
      {makeNullableFlatVector<int64_t>(
          {101, std::nullopt, std::nullopt}, DECIMAL(3, 1))},
      "make_decimal_by_unscaled_value(c0, c1, true)",
      {makeNullableFlatVector<int64_t>({101, std::nullopt, 1000}),
       makeConstant<int64_t>(0, 3, DECIMAL(3, 1))});
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
