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
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class DecimalArithmeticTest : public SparkFunctionBaseTest {
 public:
  DecimalArithmeticTest() {
    options_.parseDecimalAsDouble = false;
  }

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
    // testOpDictVectors<EvalType>(expression, expected, input);
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

  VectorPtr makeLongDecimalVector(
      const std::vector<std::string>& value,
      int8_t precision,
      int8_t scale) {
    std::vector<int128_t> int128s;
    for (auto& v : value) {
      bool nullOutput;
      int128s.emplace_back(convertStringToInt128(std::move(v), nullOutput));
      VELOX_CHECK(!nullOutput);
    }
    return makeLongDecimalFlatVector(int128s, DECIMAL(precision, scale));
  }

  int128_t convertStringToInt128(const std::string& value, bool& nullOutput) {
    // Handling integer target cases
    const char* v = value.c_str();
    nullOutput = true;
    bool negative = false;
    int128_t result = 0;
    int index = 0;
    int len = value.size();
    if (len == 0) {
      return -1;
    }
    // Setting negative flag
    if (v[0] == '-') {
      if (len == 1) {
        return -1;
      }
      negative = true;
      index = 1;
    }
    if (negative) {
      for (; index < len; index++) {
        if (!std::isdigit(v[index])) {
          return -1;
        }
        result = result * 10 - (v[index] - '0');
        // Overflow check
        if (result > 0) {
          return -1;
        }
      }
    } else {
      for (; index < len; index++) {
        if (!std::isdigit(v[index])) {
          return -1;
        }
        result = result * 10 + (v[index] - '0');
        // Overflow check
        if (result < 0) {
          return -1;
        }
      }
    }
    // Final result
    nullOutput = false;
    return result;
  }
}; // namespace

TEST_F(DecimalArithmeticTest, tmp) {
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalFlatVector({2123210}, DECIMAL(38, 6)),
      "decimal_add(c0, c1)",
      {makeLongDecimalFlatVector({11232100}, DECIMAL(38, 7)),
       makeShortDecimalFlatVector({1}, DECIMAL(10, 0))});
}

TEST_F(DecimalArithmeticTest, add) {
  // The result can be obtained by Spark unit test
  //       test("add") {
  //     val l1 = Literal.create(
  //       Decimal(BigDecimal(1), 17, 3),
  //       DecimalType(17, 3))
  //     val l2 = Literal.create(
  //       Decimal(BigDecimal(1), 17, 3),
  //       DecimalType(17, 3))
  //     checkEvaluation(Add(l1, l2), null)
  //   }

  // Precision < 38
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalFlatVector({502}, DECIMAL(31, 3)),
      "decimal_add(c0, c1)",
      {makeLongDecimalFlatVector({201}, DECIMAL(30, 3)),
       makeLongDecimalFlatVector({301}, DECIMAL(30, 3))});

  // Min leading zero >= 3
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalFlatVector({2123210}, DECIMAL(38, 6)),
      "decimal_add(c0, c1)",
      {makeLongDecimalFlatVector({11232100}, DECIMAL(38, 7)),
       makeShortDecimalFlatVector({1}, DECIMAL(10, 0))});

  // Carry to left 0.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalVector({"99999999999999999999999999999990000010"}, 38, 6),
      "decimal_add(c0, c1)",
      {makeLongDecimalVector({"9999999999999999999999999999999000000"}, 38, 5),
       makeLongDecimalFlatVector({100}, DECIMAL(38, 7))});

  // Carry to left 1
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalVector({"99999999999999999999999999999991500000"}, 38, 6),
      "decimal_add(c0, c1)",
      {makeLongDecimalVector({"9999999999999999999999999999999070000"}, 38, 5),
       makeLongDecimalFlatVector({8000000}, DECIMAL(38, 7))});

  //   Both -ve
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalFlatVector({-3211}, DECIMAL(32, 3)),
      "decimal_add(c0, c1)",
      {makeLongDecimalFlatVector({-201}, DECIMAL(30, 3)),
       makeLongDecimalFlatVector({-301}, DECIMAL(30, 2))});

  // -ve and max precision
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalVector({"-99999999999999999999999999999990000010"}, 38, 6),
      "decimal_add(c0, c1)",
      {makeLongDecimalVector(
           {"-09999999999999999999999999999999000000"}, 38, 5),
       makeLongDecimalFlatVector({-100}, DECIMAL(38, 7))});
  // ve and -ve
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalVector({"99999999999999999999999999999989999990"}, 38, 6),
      "decimal_add(c0, c1)",
      {makeLongDecimalVector({"9999999999999999999999999999999000000"}, 38, 5),
       makeLongDecimalFlatVector({-100}, DECIMAL(38, 7))});
  // -ve and ve
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalVector({"99999999999999999999999999999989999990"}, 38, 6),
      "decimal_add(c0, c1)",
      {makeLongDecimalFlatVector({-100}, DECIMAL(38, 7)),
       makeLongDecimalVector(
           {"9999999999999999999999999999999000000"}, 38, 5)});
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
