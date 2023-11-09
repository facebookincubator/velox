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

#include "velox/functions/sparksql/specialforms/DecimalRound.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {
class DecimalRoundTest : public SparkFunctionBaseTest {
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

  void testDecimalRound(
      std::optional<int32_t> scaleOpt,
      const VectorPtr& input,
      const VectorPtr& expected) {
    auto inputType = input->type();
    std::vector<core::TypedExprPtr> inputs = {
        std::make_shared<core::FieldAccessTypedExpr>(inputType, "c0")};
    int32_t scale = 0;
    if (scaleOpt.has_value()) {
      scale = scaleOpt.value();
      inputs.emplace_back(
          std::make_shared<core::ConstantTypedExpr>(INTEGER(), variant(scale)));
    }
    auto [inputPrecision, inputScale] = getDecimalPrecisionScale(*inputType);
    auto [resultPrecision, resultScale] =
        getResultPrecisionScale(inputPrecision, inputScale, scale);
    auto resultType = DECIMAL(resultPrecision, resultScale);
    auto round = std::make_shared<const core::CallTypedExpr>(
        resultType, std::move(inputs), "decimal_round");
    auto result = evaluate(round, makeRowVector({input}));
    velox::test::assertEqualVectors(expected, result);
    testDictionary(round, input, expected);

    // It is a common case in Spark for the second argument to be cast.
    inputs = {
        std::make_shared<core::FieldAccessTypedExpr>(inputType, "c0"),
        std::make_shared<core::CastTypedExpr>(
            INTEGER(),
            std::make_shared<core::ConstantTypedExpr>(
                BIGINT(), variant((int64_t)scale)),
            true /*nullOnFailure*/)};
    round = std::make_shared<const core::CallTypedExpr>(
        resultType, std::move(inputs), "decimal_round");
    result = evaluate(round, makeRowVector({input}));
    velox::test::assertEqualVectors(expected, result);
    testDictionary(round, input, expected);
  }
};

TEST_F(DecimalRoundTest, round) {
  // Round up to 'scale' decimal places.
  testDecimalRound(
      3,
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(4, 3)));
  // Round up to 'scale - 1' decimal places.
  testDecimalRound(
      2,
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      makeFlatVector<int64_t>({12, 55, -100, 0}, DECIMAL(3, 2)));
  // Round up to 0 decimal places.
  testDecimalRound(
      0,
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      makeFlatVector<int64_t>({0, 1, -1, 0}, DECIMAL(1, 0)));
  testDecimalRound(
      std::nullopt,
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      makeFlatVector<int64_t>({0, 1, -1, 0}, DECIMAL(1, 0)));
  testDecimalRound(
      0,
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 2)),
      makeFlatVector<int64_t>({1, 6, -10, 0}, DECIMAL(2, 0)));
  testDecimalRound(
      std::nullopt,
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 2)),
      makeFlatVector<int64_t>({1, 6, -10, 0}, DECIMAL(2, 0)));
  // Round up to negative decimal places.
  testDecimalRound(
      -1,
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3)),
      makeFlatVector<int64_t>({0, 0, 0, 0}, DECIMAL(2, 0)));
  testDecimalRound(
      -1,
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1)),
      makeFlatVector<int64_t>({10, 60, -100, 0}, DECIMAL(3, 0)));
  testDecimalRound(
      -3,
      makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1)),
      makeFlatVector<int64_t>({0, 0, 0, 0}, DECIMAL(4, 0)));
  // Round up long decimals to short decimals.
  testDecimalRound(
      14,
      makeFlatVector<int128_t>(
          {1234567890123456789, 5000000000000000000, -999999999999999999, 0},
          DECIMAL(19, 19)),
      makeNullableFlatVector<int64_t>(
          {12345678901235, 50000000000000, -10'000'000'000'000, 0},
          DECIMAL(15, 14)));
  testDecimalRound(
      -9,
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(19, 5)),
      makeFlatVector<int64_t>(
          {12346000000000, 55556000000000, -10000000000000, 0},
          DECIMAL(15, 0)));
  // Round up long decimals to long decimals.
  testDecimalRound(
      14,
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(19, 5)),
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(20, 5)));
  testDecimalRound(
      -9,
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(32, 5)),
      makeFlatVector<int128_t>(
          {12346000000000, 55556000000000, -10000000000000, 0},
          DECIMAL(28, 0)));
  // Result precision is 38.
  testDecimalRound(
      -38,
      makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(32, 0)),
      makeFlatVector<int128_t>({0, 0, 0, 0}, DECIMAL(38, 0)));
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
