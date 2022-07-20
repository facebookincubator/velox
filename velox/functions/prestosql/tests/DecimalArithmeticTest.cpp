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
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"
#include "velox/vector/tests/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace facebook::velox {

class DecimalArithmeticTest : public FunctionBaseTest {
 protected:
  template <typename T>
  void testDecimalExpr(
      const VectorPtr& expected,
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    auto result = evaluate<SimpleVector<T>>(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
    ASSERT_EQ(result->type()->toString(), expected->type()->toString());
  }

  template <typename T>
  void testOpFlatConstant(
      const std::string& operation,
      const std::string& constant,
      const VectorPtr& flatVector,
      const VectorPtr& result,
      bool isLeftConstant) {
    auto shortConstant = BaseVector::wrapInConstant(
        1, 0, makeDecimalFlatVector<ShortDecimal>({ShortDecimal(1000)}, 10, 3));
    std::vector<VectorPtr> input({flatVector});
    RowVectorPtr rowVector = makeRowVector(input);
    auto rowType = std::dynamic_pointer_cast<const RowType>(rowVector->type());
    core::TypedExprPtr typedExpr;
    std::vector<core::TypedExprPtr> params;
    auto fieldTypedExpr =
        std::make_shared<core::FieldAccessTypedExpr>(flatVector->type(), "c0");
    auto constantTypedExpr =
        std::make_shared<core::ConstantTypedExpr>(shortConstant);
    if (isLeftConstant) {
      typedExpr =
          makeTypedExpr(fmt::format("{}({},c0)", operation, constant), rowType);
      params.push_back(constantTypedExpr);
      params.push_back(fieldTypedExpr);
    } else {
      typedExpr =
          makeTypedExpr(fmt::format("{}(c0,{})", operation, constant), rowType);
      params.push_back(fieldTypedExpr);
      params.push_back(constantTypedExpr);
    }
    auto newCallTypedExpr = std::make_shared<core::CallTypedExpr>(
        typedExpr->type(), params, operation);
    auto actual = evaluate(newCallTypedExpr, rowVector);
    VectorPtr actualSimple = std::dynamic_pointer_cast<SimpleVector<T>>(actual);
    assertEqualVectors(actualSimple, result);
    ASSERT_EQ(result->type()->toString(), actual->type()->toString());
  }

  template <typename T>
  void testOpDictVectors(
      const std::string& operation,
      const VectorPtr& result,
      const std::vector<VectorPtr>& flatVector) {
    // Dictionary vectors as arguments.
    vector_size_t newSize = flatVector[0]->size() * 2;
    auto indices1 = makeIndices(newSize, [&](int row) { return row / 2; });
    auto indices2 = makeIndices(newSize, [&](int row) { return row / 2; });
    auto shortDictA =
        VectorTestBase::wrapInDictionary(indices1, newSize, flatVector[0]);
    auto shortDictB =
        VectorTestBase::wrapInDictionary(indices2, newSize, flatVector[1]);
    auto resultDict =
        VectorTestBase::wrapInDictionary(indices1, newSize, result);

    testDecimalExpr<T>(
        resultDict,
        fmt::format("{}(c0,c1)", operation),
        {shortDictA, shortDictB});
  }
};
} // namespace facebook::velox

TEST_F(DecimalArithmeticTest, decimalAddTest) {
  auto resultLongFlat = makeDecimalFlatVector<LongDecimal>(
      {LongDecimal(2000), LongDecimal(4000)}, 19, 3);
  auto shortFlat = makeDecimalFlatVector<ShortDecimal>(
      {ShortDecimal(1000), ShortDecimal(2000)}, 18, 3);
  // Add short and short, returning long.
  testDecimalExpr<LongDecimal>(
      resultLongFlat, "plus(c0, c1)", {shortFlat, shortFlat});
  // Add short and long, returning long.
  auto longFlat = makeDecimalFlatVector<LongDecimal>(
      {LongDecimal(1000), LongDecimal(2000)}, 19, 3);
  resultLongFlat = makeDecimalFlatVector<LongDecimal>(
      {LongDecimal(2000), LongDecimal(4000)}, 20, 3);
  testDecimalExpr<LongDecimal>(
      resultLongFlat, "plus(c0, c1)", {shortFlat, longFlat});
  // Add short and long, returning long.
  testDecimalExpr<LongDecimal>(
      resultLongFlat, "plus(c0, c1)", {longFlat, shortFlat});

  // Add long and long, returning long.
  testDecimalExpr<LongDecimal>(resultLongFlat, "c0 + c1", {longFlat, longFlat});
  // Add short and short, returning short.
  shortFlat = makeDecimalFlatVector<ShortDecimal>(
      {ShortDecimal(1000), ShortDecimal(2000)}, 10, 3);
  auto resultShortFlat = makeDecimalFlatVector<ShortDecimal>(
      {ShortDecimal(2000), ShortDecimal(4000)}, 11, 3);
  testDecimalExpr<ShortDecimal>(
      resultShortFlat, "c0 + c1", {shortFlat, shortFlat});

  auto resultConstantFlat = makeDecimalFlatVector<ShortDecimal>(
      {ShortDecimal(2000), ShortDecimal(3000)}, 11, 3);
  // Constant and Flat arguments.
  testOpFlatConstant<ShortDecimal>(
      "plus", "'1.000'::decimal(10,3)", shortFlat, resultConstantFlat, true);
  // Flat and Constant arguments.
  testOpFlatConstant<ShortDecimal>(
      "plus", "'1.000'::decimal(10,3)", shortFlat, resultConstantFlat, false);
  testOpDictVectors<ShortDecimal>(
      "plus", resultShortFlat, {shortFlat, shortFlat});
}
