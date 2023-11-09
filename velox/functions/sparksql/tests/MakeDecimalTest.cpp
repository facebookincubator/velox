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

namespace facebook::velox::functions::sparksql::test {
namespace {
class MakeDecimalTest : public SparkFunctionBaseTest {
 protected:
  void testDictionary(
      const core::CallTypedExprPtr& expr,
      const VectorPtr& input,
      const VectorPtr& expected) {
    // Dictionary vectors as arguments.
    auto newSize = input->size() * 2;
    std::vector<VectorPtr> vectors;
    vectors.reserve(1);
    auto indices = makeIndices(newSize, [&](int row) { return row / 2; });
    vectors.emplace_back(
        VectorTestBase::wrapInDictionary(indices, newSize, input));
    auto resultIndices = makeIndices(newSize, [&](int row) { return row / 2; });
    auto expectedDic =
        VectorTestBase::wrapInDictionary(resultIndices, newSize, expected);
    auto actual = evaluate(expr, makeRowVector(vectors));
    velox::test::assertEqualVectors(expectedDic, actual);
  }

  void testMakeDecimal(
      bool nullOnOverflow,
      const VectorPtr& input,
      const VectorPtr& expected) {
    std::vector<core::TypedExprPtr> inputs = {
        std::make_shared<core::FieldAccessTypedExpr>(input->type(), "c0"),
        std::make_shared<core::ConstantTypedExpr>(
            BOOLEAN(), variant(nullOnOverflow))};
    auto makeDecimal = std::make_shared<const core::CallTypedExpr>(
        expected->type(), std::move(inputs), "make_decimal");
    auto result = evaluate(makeDecimal, makeRowVector({input}));
    velox::test::assertEqualVectors(expected, result);
    testDictionary(makeDecimal, input, expected);
  }
};

TEST_F(MakeDecimalTest, makeDecimal) {
  testMakeDecimal(
      true,
      makeFlatVector<int64_t>({1111, -1112, 9999, 0}),
      makeFlatVector<int64_t>({1111, -1112, 9999, 0}, DECIMAL(5, 1)));
  testMakeDecimal(
      true,
      makeFlatVector<int64_t>(
          {11111111, -11112112, 99999999, DecimalUtil::kShortDecimalMax + 1}),
      makeFlatVector<int128_t>(
          {11111111, -11112112, 99999999, DecimalUtil::kShortDecimalMax + 1},
          DECIMAL(38, 19)));
  EXPECT_ANY_THROW(testMakeDecimal(
      false,
      makeFlatVector<int64_t>(
          {11111111, -11112112, 99999999, DecimalUtil::kShortDecimalMax + 1}),
      makeFlatVector<int128_t>(
          {11111111, -11112112, 99999999, DecimalUtil::kShortDecimalMax + 1},
          DECIMAL(18, 0))));
  testMakeDecimal(
      true,
      makeNullableFlatVector<int64_t>({101, std::nullopt, 1000}),
      makeNullableFlatVector<int64_t>(
          {101, std::nullopt, std::nullopt}, DECIMAL(3, 1)));
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
