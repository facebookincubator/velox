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
 protected:
  void testArithmeticFunction(
      const std::string& functionName,
      const std::vector<VectorPtr>& inputs,
      const VectorPtr& expected) {
    VELOX_USER_CHECK_EQ(
        inputs.size(),
        2,
        "Two input vectors are needed for arithmetic function test.");
    std::vector<core::TypedExprPtr> inputExprs = {
        std::make_shared<core::FieldAccessTypedExpr>(inputs[0]->type(), "c0"),
        std::make_shared<core::FieldAccessTypedExpr>(inputs[1]->type(), "c1")};
    auto expr = std::make_shared<const core::CallTypedExpr>(
        expected->type(), std::move(inputExprs), functionName);
    testEncodings(expr, inputs, expected);
  }

  VectorPtr makeNullableLongDecimalVector(
      const std::vector<std::string>& values,
      const TypePtr& type) {
    VELOX_USER_CHECK(
        type->isDecimal(),
        "Decimal type is needed to create long decimal vector.");
    std::vector<std::optional<int128_t>> numbers;
    numbers.reserve(values.size());
    for (const auto& value : values) {
      if (value == "null") {
        numbers.emplace_back(std::nullopt);
      } else {
        numbers.emplace_back(HugeInt::parse(value));
      }
    }
    return makeNullableFlatVector<int128_t>(numbers, type);
  }
}; // namespace

TEST_F(DecimalArithmeticTest, add) {
  // Precision < 38.
  testArithmeticFunction(
      "add",
      {makeNullableLongDecimalVector(
           {"201", "601", "1366", "999999999999999999999999999999"},
           DECIMAL(30, 3)),
       makeNullableLongDecimalVector(
           {"301", "901", "9866", "999999999999999999999999999999"},
           DECIMAL(30, 3))},
      makeNullableLongDecimalVector(
          {"502", "1502", "11232", "1999999999999999999999999999998"},
          DECIMAL(31, 3)));

  // Min leading zero >= 3.
  testArithmeticFunction(
      "add",
      {makeFlatVector(
           std::vector<int128_t>{11232100, 9998888, 12345678, 2135632},
           DECIMAL(38, 7)),
       makeFlatVector(std::vector<int64_t>{1, 2, 3, 4}, DECIMAL(10, 0))},
      makeFlatVector(
          std::vector<int128_t>{2123210, 2999889, 4234568, 4213563},
          DECIMAL(38, 6)));

  // No carry to left.
  testArithmeticFunction(
      "add",
      {makeNullableLongDecimalVector(
           {"9999999999999999999999999999999000000",
            "9999999999999999999999999999999900000",
            "9999999999999999999999999999999990000",
            "9999999999999999999999999999999999000"},
           DECIMAL(38, 5)),
       makeFlatVector(
           std::vector<int128_t>{100, 99999, 1234, 999}, DECIMAL(38, 7))},
      makeNullableLongDecimalVector(
          {"99999999999999999999999999999990000010",
           "99999999999999999999999999999999010000",
           "99999999999999999999999999999999900123",
           "99999999999999999999999999999999990100"},
          DECIMAL(38, 6)));

  // Carry to left.
  testArithmeticFunction(
      "add",
      {makeNullableLongDecimalVector(
           {"9999999999999999999999999999999070000",
            "9999999999999999999999999999999050000",
            "9999999999999999999999999999999870000",
            "9999999999999999999999999999999890000"},
           DECIMAL(38, 5)),
       makeFlatVector(
           std::vector<int128_t>{8000000, 5000000, 8000000, 1999999},
           DECIMAL(38, 7))},
      makeNullableLongDecimalVector(
          {"99999999999999999999999999999991500000",
           "99999999999999999999999999999991000000",
           "99999999999999999999999999999999500000",
           "99999999999999999999999999999999100000"},
          DECIMAL(38, 6)));

  // Both -ve.
  testArithmeticFunction(
      "add",
      {makeNullableLongDecimalVector(
           {"-201", "-601", "-1366", "-999999999999999999999999999999"},
           DECIMAL(30, 3)),
       makeNullableLongDecimalVector(
           {"-301", "-901", "-9866", "-999999999999999999999999999999"},
           DECIMAL(30, 3))},
      makeNullableLongDecimalVector(
          {"-502", "-1502", "-11232", "-1999999999999999999999999999998"},
          DECIMAL(31, 3)));

  // Overflow when scaling up the whole part.
  testArithmeticFunction(
      "add",
      {makeNullableLongDecimalVector(
           {"-99999999999999999999999999999999990000",
            "99999999999999999999999999999999999000",
            "-99999999999999999999999999999999999900",
            "99999999999999999999999999999999999990"},
           DECIMAL(38, 3)),
       makeFlatVector(
           std::vector<int128_t>{-100, 9999999, -999900, 99999},
           DECIMAL(38, 7))},
      makeNullableLongDecimalVector(
          {"null", "null", "null", "null"}, DECIMAL(38, 6)));

  // Ve and -ve.
  testArithmeticFunction(
      "add",
      {makeNullableLongDecimalVector(
           {"99999999999999999999999999999989999990",
            "-99999999999999999999999999999989999990",
            "99999999999999999999999999999999999980",
            "-99999999999999999999999999999999999980"},
           DECIMAL(38, 6)),
       makeNullableLongDecimalVector(
           {"-9999999999999999999999999999998900000",
            "9999999999999999999999999999998900000",
            "-9999999999999999999999999999999999999",
            "9999999999999999999999999999999999999"},
           DECIMAL(38, 5))},
      makeNullableLongDecimalVector(
          {"999990", "-999990", "-10", "10"}, DECIMAL(38, 6)));
}

TEST_F(DecimalArithmeticTest, subtract) {
  testArithmeticFunction(
      "subtract",
      {makeNullableLongDecimalVector(
           {"201", "601", "1366", "999999999999999999999999999999"},
           DECIMAL(30, 3)),
       makeNullableLongDecimalVector(
           {"301", "901", "9866", "-999999999999999999999999999999"},
           DECIMAL(30, 3))},
      makeNullableLongDecimalVector(
          {"-100", "-300", "-8500", "1999999999999999999999999999998"},
          DECIMAL(31, 3)));

  // Min leading zero >= 3.
  testArithmeticFunction(
      "subtract",
      {makeFlatVector(
           std::vector<int128_t>{11232100, 9998888, 12345678, 2135632},
           DECIMAL(38, 7)),
       makeFlatVector(std::vector<int64_t>{1, 2, 3, 4}, DECIMAL(10, 0))},
      makeFlatVector(
          std::vector<int128_t>{123210, -1000111, -1765432, -3786437},
          DECIMAL(38, 6)));

  // No carry to left.
  testArithmeticFunction(
      "subtract",
      {makeNullableLongDecimalVector(
           {"9999999999999999999999999999999000000",
            "9999999999999999999999999999999900000",
            "9999999999999999999999999999999990000",
            "9999999999999999999999999999999999000"},
           DECIMAL(38, 5)),
       makeFlatVector(
           std::vector<int128_t>{-100, -99999, -1234, -999}, DECIMAL(38, 7))},
      makeNullableLongDecimalVector(
          {"99999999999999999999999999999990000010",
           "99999999999999999999999999999999010000",
           "99999999999999999999999999999999900123",
           "99999999999999999999999999999999990100"},
          DECIMAL(38, 6)));

  // Carry to left.
  testArithmeticFunction(
      "subtract",
      {makeNullableLongDecimalVector(
           {"9999999999999999999999999999999070000",
            "9999999999999999999999999999999050000",
            "9999999999999999999999999999999870000",
            "9999999999999999999999999999999890000"},
           DECIMAL(38, 5)),
       makeFlatVector(
           std::vector<int128_t>{-8000000, -5000000, -8000000, -1999999},
           DECIMAL(38, 7))},
      makeNullableLongDecimalVector(
          {"99999999999999999999999999999991500000",
           "99999999999999999999999999999991000000",
           "99999999999999999999999999999999500000",
           "99999999999999999999999999999999100000"},
          DECIMAL(38, 6)));

  // Both -ve.
  testArithmeticFunction(
      "subtract",
      {makeNullableLongDecimalVector(
           {"-201", "-601", "-1366", "-999999999999999999999999999999"},
           DECIMAL(30, 3)),
       makeNullableLongDecimalVector(
           {"-301", "-901", "-9866", "-999999999999999999999999999999"},
           DECIMAL(30, 3))},
      makeNullableLongDecimalVector(
          {"100", "300", "8500", "0"}, DECIMAL(31, 3)));

  // Overflow when scaling up the whole part.
  testArithmeticFunction(
      "subtract",
      {makeNullableLongDecimalVector(
           {"-99999999999999999999999999999999990000",
            "99999999999999999999999999999999999000",
            "-99999999999999999999999999999999999900",
            "99999999999999999999999999999999999990"},
           DECIMAL(38, 3)),
       makeFlatVector(
           std::vector<int128_t>{100, -9999999, 999900, -99999},
           DECIMAL(38, 7))},
      makeNullableLongDecimalVector(
          {"null", "null", "null", "null"}, DECIMAL(38, 6)));

  // Ve and -ve.
  testArithmeticFunction(
      "subtract",
      {makeNullableLongDecimalVector(
           {"99999999999999999999999999999989999990",
            "-99999999999999999999999999999989999990",
            "99999999999999999999999999999999999980",
            "-99999999999999999999999999999999999980"},
           DECIMAL(38, 6)),
       makeFlatVector(
           std::vector<int128_t>{-1000000, 1000000, -1, 1}, DECIMAL(38, 5))},
      makeNullableLongDecimalVector(
          {"99999999999999999999999999999999999990",
           "-99999999999999999999999999999999999990",
           "99999999999999999999999999999999999990",
           "-99999999999999999999999999999999999990"},
          DECIMAL(38, 6)));
}

TEST_F(DecimalArithmeticTest, multiply) {
  // Multiply short and short decimals, and return long decimals.
  testArithmeticFunction(
      "multiply",
      {makeFlatVector<int64_t>({1000, 2000}, DECIMAL(17, 3)),
       makeFlatVector<int64_t>({1000, 2000}, DECIMAL(17, 3))},
      makeFlatVector<int128_t>({1000000, 4000000}, DECIMAL(35, 6)));

  // Multiply short and long decimals, and return long decimals.
  testArithmeticFunction(
      "multiply",
      {makeFlatVector<int64_t>({1000, 2000}, DECIMAL(17, 3)),
       makeFlatVector<int128_t>({1000, 2000}, DECIMAL(20, 3))},
      makeFlatVector<int128_t>({1000000, 4000000}, DECIMAL(38, 6)));

  // Multiply long and short decimals, and return long decimals.
  testArithmeticFunction(
      "multiply",
      {makeFlatVector<int128_t>({1000, 2000}, DECIMAL(20, 3)),
       makeFlatVector<int64_t>({1000, 2000}, DECIMAL(17, 3))},
      makeFlatVector<int128_t>({1000000, 4000000}, DECIMAL(38, 6)));

  // Multiply long and long decimals, and return long decimals.
  testArithmeticFunction(
      "multiply",
      {makeFlatVector<int128_t>({1000, 2000}, DECIMAL(20, 3)),
       makeFlatVector<int128_t>({1000, 2000}, DECIMAL(20, 3))},
      makeFlatVector<int128_t>({1000000, 4000000}, DECIMAL(38, 6)));

  testArithmeticFunction(
      "multiply",
      {makeFlatVector<int128_t>({0, 1, 0}, DECIMAL(20, 3)),
       makeFlatVector<int128_t>({1, 0, 0}, DECIMAL(20, 2))},
      makeFlatVector<int128_t>({0, 0, 0}, DECIMAL(38, 5)));

  // Multiply short and short decimals, and return short decimals.
  testArithmeticFunction(
      "multiply",
      {makeFlatVector<int64_t>({1000, 2000}, DECIMAL(6, 3)),
       makeFlatVector<int64_t>({1000, 2000}, DECIMAL(6, 3))},
      makeFlatVector<int64_t>({1000000, 4000000}, DECIMAL(13, 6)));

  // Multiply constant and flat arguments.
  testArithmeticFunction(
      "multiply",
      {makeConstant<int64_t>(100, 2, DECIMAL(3, 2)),
       makeFlatVector<int64_t>({1000, 2000}, DECIMAL(6, 3))},
      makeFlatVector<int64_t>({100000, 200000}, DECIMAL(10, 5)));

  // Multiply flat and constant arguments.
  testArithmeticFunction(
      "multiply",
      {makeFlatVector<int64_t>({1000, 2000}, DECIMAL(6, 3)),
       makeConstant<int64_t>(100, 2, DECIMAL(3, 2))},
      makeFlatVector<int64_t>({100000, 200000}, DECIMAL(10, 5)));

  // Output precision is 38 with small input values and trimmed scale.
  testArithmeticFunction(
      "multiply",
      {makeConstant<int128_t>(201, 1, DECIMAL(20, 5)),
       makeConstant<int128_t>(301, 1, DECIMAL(20, 5))},
      makeConstant<int128_t>(61, 1, DECIMAL(38, 7)));

  // Output precision is 38 with large input values and trimmed scale.
  testArithmeticFunction(
      "multiply",
      {makeConstant<int128_t>(201, 1, DECIMAL(20, 5)),
       makeConstant<int128_t>(
           HugeInt::parse(std::string(35, '9')), 1, DECIMAL(35, 5))},
      makeConstant<int128_t>(
          HugeInt::parse("201" + std::string(31, '0')), 1, DECIMAL(38, 6)));

  // Output precision is 38 with very large input values and trimmed scale.
  // Requires for int256.
  testArithmeticFunction(
      "multiply",
      {makeConstant<int128_t>(
           HugeInt::parse(std::string(35, '9')), 1, DECIMAL(38, 20)),
       makeConstant<int128_t>(
           HugeInt::parse(std::string(36, '9')), 1, DECIMAL(38, 20))},
      makeConstant<int128_t>(
          HugeInt::parse("9999999999999999999999999999999999890"),
          1,
          DECIMAL(38, 6)));

  // Output precision is 38 with very large input values and trimmed scale.
  // Requires for int256 and overflow occurs.
  testArithmeticFunction(
      "multiply",
      {makeConstant<int128_t>(
           HugeInt::parse(std::string(35, '9')), 1, DECIMAL(38, 4)),
       makeConstant<int128_t>(
           HugeInt::parse(std::string(36, '9')), 1, DECIMAL(38, 4))},
      makeConstant<int128_t>(std::nullopt, 1, DECIMAL(38, 6)));

  // Multiply two inputs of big scale.
  testArithmeticFunction(
      "multiply",
      {makeConstant<int128_t>(201, 1, DECIMAL(38, 38)),
       makeConstant<int128_t>(301, 1, DECIMAL(38, 38))},
      makeConstant<int128_t>(0, 1, DECIMAL(38, 37)));

  // Exceeds long decimal limit.
  testArithmeticFunction(
      "multiply",
      {makeConstant<int128_t>(
           HugeInt::build(0x08FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
           1,
           DECIMAL(38, 0)),
       makeConstant<int64_t>(10, 1, DECIMAL(2, 0))},
      makeConstant<int128_t>(std::nullopt, 1, DECIMAL(38, 0)));

  // Rescaling the final result overflows.
  testArithmeticFunction(
      "multiply",
      {makeConstant<int128_t>(
           HugeInt::build(0x08FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
           1,
           DECIMAL(38, 0)),
       makeConstant<int64_t>(10, 1, DECIMAL(2, 1))},
      makeConstant<int128_t>(std::nullopt, 1, DECIMAL(38, 1)));
}

TEST_F(DecimalArithmeticTest, divide) {
  // Divide short and short decimals, and return long decimals.
  testArithmeticFunction(
      "divide",
      {makeFlatVector<int64_t>({500, 4000}, DECIMAL(17, 3)),
       makeFlatVector<int64_t>({1000, 2000}, DECIMAL(17, 3))},
      makeNullableLongDecimalVector(
          {"500000000000000000000", "2000000000000000000000"},
          DECIMAL(38, 21)));

  // Divide long and short decimals, and return long decimals.
  testArithmeticFunction(
      "divide",
      {makeFlatVector<int128_t>({500, 4000}, DECIMAL(20, 2)),
       makeFlatVector<int64_t>({1000, 2000}, DECIMAL(17, 3))},
      makeFlatVector<int128_t>(
          {500000000000000000, 2000000000000000000}, DECIMAL(38, 17)));

  // Divide short and long decimals, and return long decimals.
  testArithmeticFunction(
      "divide",
      {makeFlatVector<int64_t>({1000, 2000}, DECIMAL(17, 3)),
       makeFlatVector<int128_t>({500, 4000}, DECIMAL(20, 2))},
      makeNullableLongDecimalVector(
          {"20" + std::string(20, '0'), "5" + std::string(20, '0')},
          DECIMAL(38, 22)));

  // Divide long and long decimals, and return long decimals.
  testArithmeticFunction(
      "divide",
      {makeFlatVector<int128_t>({2500, 12000}, DECIMAL(20, 2)),
       makeFlatVector<int128_t>({500, 4000}, DECIMAL(20, 2))},
      makeNullableLongDecimalVector(
          {"5" + std::string(18, '0'), "3" + std::string(18, '0')},
          DECIMAL(38, 18)));

  // Divide short and short decimals, and return short decimals.
  testArithmeticFunction(
      "divide",
      {makeFlatVector<int64_t>({2500, 12000}, DECIMAL(5, 5)),
       makeFlatVector<int64_t>({500, 4000}, DECIMAL(5, 2))},
      makeFlatVector<int64_t>({500000000, 300000000}, DECIMAL(13, 11)));

  testArithmeticFunction(
      "divide",
      {makeNullableLongDecimalVector({std::string(35, '9')}, DECIMAL(35, 6)),
       makeConstant<int128_t>(201, 1, DECIMAL(20, 3))},
      makeNullableLongDecimalVector(
          {"497512437810945273631840796019900493"}, DECIMAL(38, 6)));

  testArithmeticFunction(
      "divide",
      {makeConstant<int64_t>(100, 2, DECIMAL(3, 2)),
       makeFlatVector<int64_t>({1000, 2000}, DECIMAL(17, 3))},
      makeNullableLongDecimalVector(
          {"1000" + std::string(17, '0'), "500" + std::string(17, '0')},
          DECIMAL(24, 20)));

  // Divide flat and constant arguments.
  testArithmeticFunction(
      "divide",
      {makeFlatVector<int64_t>({1000, 2000}, DECIMAL(17, 3)),
       makeConstant<int64_t>(200, 2, DECIMAL(3, 2))},
      makeNullableLongDecimalVector(
          {"500" + std::string(4, '0'), "1000" + std::string(4, '0')},
          DECIMAL(23, 7)));

  // Divide and round-up.
  testArithmeticFunction(
      "divide",
      {makeFlatVector<int64_t>({-34, 5, 65, 90, 2, -49}, DECIMAL(2, 1)),
       makeConstant<int64_t>(-60, 6, DECIMAL(2, 1))},
      makeFlatVector<int64_t>(
          {566667, -83333, -1083333, -1500000, -33333, 816667}, DECIMAL(8, 6)));

  // Divide by zero.
  testArithmeticFunction(
      "divide",
      {makeFlatVector<int64_t>({1000, 2000}, DECIMAL(17, 3)),
       makeConstant<int64_t>(0, 2, DECIMAL(2, 1))},
      makeConstant<int128_t>(std::nullopt, 2, DECIMAL(21, 6)));

  // Exceeds long decimal limit.
  testArithmeticFunction(
      "divide",
      {makeConstant<int128_t>(DecimalUtil::kLongDecimalMax, 1, DECIMAL(38, 0)),
       makeConstant<int64_t>(1, 1, DECIMAL(3, 2))},
      makeConstant<int128_t>(std::nullopt, 1, DECIMAL(38, 6)));
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
