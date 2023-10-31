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

class DecimalVectorFunctionsTest : public SparkFunctionBaseTest {
 protected:
  template <TypeKind KIND>
  void testDecimalExpr(
      const VectorPtr& expected,
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    using EvalType = typename velox::TypeTraits<KIND>::NativeType;
    auto result =
        evaluate<SimpleVector<EvalType>>(expression, makeRowVector(input));
    velox::test::assertEqualVectors(expected, result);
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
    velox::test::assertEqualVectors(expectedResultDictionary, actual);
  }
};

// The result can be obtained by Spark spark-shell CLI.
// scala> val df = spark.sql("select round(cast(0.123 as decimal(3,3)), 30);")
// df: org.apache.spark.sql.DataFrame = [round(CAST(0.123 AS DECIMAL(3,3)), 30):
// decimal(4,3)]
TEST_F(DecimalVectorFunctionsTest, round) {
  //   Round up to 'scale' decimal places.
  testDecimalExpr<TypeKind::BIGINT>(
      {makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(4, 3))},
      "decimal_round(c0, CAST(30 as integer))",
      {makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3))});

  //   Round up to scale-1 decimal places.
  testDecimalExpr<TypeKind::BIGINT>(
      {makeFlatVector<int64_t>({12, 55, -100, 0}, DECIMAL(3, 2))},
      "decimal_round(c0, CAST(2 as integer))",
      {makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3))});
  // Round up to 0 decimal places.
  testDecimalExpr<TypeKind::BIGINT>(
      {makeFlatVector<int64_t>({0, 1, -1, 0}, DECIMAL(1, 0))},
      "decimal_round(c0, CAST(0 as integer))",
      {makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 3))});
  testDecimalExpr<TypeKind::BIGINT>(
      {makeFlatVector<int64_t>({1, 6, -10, 0}, DECIMAL(2, 0))},
      "decimal_round(c0, CAST(0 as integer))",
      {makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 2))});
  testDecimalExpr<TypeKind::BIGINT>(
      {makeFlatVector<int64_t>({1, 6, -10, 0}, DECIMAL(2, 0))},
      "decimal_round(c0)",
      {makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 2))});
  //   Round up to -1 decimal places.
  testDecimalExpr<TypeKind::BIGINT>(
      {makeFlatVector<int64_t>({10, 60, -100, 0}, DECIMAL(3, 0))},
      "decimal_round(c0, CAST(-1 as integer))",
      {makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1))});
  // Round up to -2 decimal places. Here precision == -scale + 1.
  testDecimalExpr<TypeKind::BIGINT>(
      {makeFlatVector<int64_t>({0, 0, 0, 0}, DECIMAL(4, 0))},
      "decimal_round(c0, CAST(-3 as integer))",
      {makeFlatVector<int64_t>({123, 552, -999, 0}, DECIMAL(3, 1))});

  // Round up long decimals to short decimals.
  testDecimalExpr<TypeKind::BIGINT>(
      {makeNullableFlatVector<int64_t>(
          {12345678901235, 50000000000000, -10'000'000'000'000, 0},
          DECIMAL(15, 14))},
      "decimal_round(c0, CAST(14 as integer))",
      {makeFlatVector<int128_t>(
          {1234567890123456789, 5000000000000000000, -999999999999999999, 0},
          DECIMAL(19, 19))});
  testDecimalExpr<TypeKind::BIGINT>(
      {makeFlatVector<int64_t>(
          {12346000000000, 55556000000000, -10000000000000, 0},
          DECIMAL(15, 0))},
      "decimal_round(c0, CAST(-9 as integer))",
      {makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(19, 5))});
  // Round up long decimals to long decimals.
  testDecimalExpr<TypeKind::HUGEINT>(
      {makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(20, 5))},
      "decimal_round(c0, CAST(14 as integer))",
      {makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(19, 5))});
  testDecimalExpr<TypeKind::HUGEINT>(
      {makeFlatVector<int128_t>(
          {12346000000000, 55556000000000, -10000000000000, 0},
          DECIMAL(28, 0))},
      "decimal_round(c0, CAST(-9 as integer))",
      {makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(32, 5))});
  // Result precision is 38
  testDecimalExpr<TypeKind::HUGEINT>(
      {makeFlatVector<int128_t>({0, 0, 0, 0}, DECIMAL(38, 0))},
      "decimal_round(c0, CAST(-38 as integer))",
      {makeFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, -999999999999999999, 0},
          DECIMAL(32, 0))});
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
