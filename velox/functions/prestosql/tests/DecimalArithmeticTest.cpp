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

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace facebook::velox {

class DecimalArithmeticTest : public FunctionBaseTest {
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
    testOpDictVectors<EvalType>(expression, expected, input);
  }

  // Returns a tester lambda that takes an InputType and an EvalType and which
  // can be called to test the expression with specific input and expected
  // output values. See Issue #16464 for details.
  template <typename InputType, TypeKind K>
  std::function<void(InputType, typename TypeTraits<K>::NativeType)>
  makeUnaryDecimalExprTester(
      const TypePtr& inType,
      const TypePtr& outType,
      const std::string& exprStr) {
    using EvalType = typename TypeTraits<K>::NativeType;
    return [this, inType, outType, exprStr](InputType input, EvalType out) {
      auto v = makeFlatVector<InputType>({input}, inType);
      testDecimalExpr<K>(
          makeFlatVector<EvalType>({out}, outType), exprStr, {v});
    };
  }

  // Returns a lambda (InputType) that runs the expression with one value and
  // asserts the given exception message.
  template <typename InputType, TypeKind K = TypeKind::BIGINT>
  std::function<void(InputType)> makeUnaryDecimalExprExceptionTester(
      const TypePtr& inType,
      const std::string& exprStr,
      const std::string& exceptionMessage) {
    return [this, inType, exprStr, exceptionMessage](InputType input) {
      VELOX_ASSERT_USER_THROW(
          testDecimalExpr<K>(
              {}, exprStr, {makeFlatVector<InputType>({input}, inType)}),
          exceptionMessage);
    };
  }

  // Same for binary expressions: returns (in1, in2) -> assert expression
  // throws.
  template <
      typename InputType1,
      typename InputType2,
      TypeKind K = TypeKind::HUGEINT>
  std::function<void(InputType1, InputType2)>
  makeBinaryDecimalExprExceptionTester(
      const TypePtr& inType1,
      const TypePtr& inType2,
      const std::string& exprStr,
      const std::string& exceptionMessage) {
    return [this, inType1, inType2, exprStr, exceptionMessage](
               InputType1 in1, InputType2 in2) {
      VELOX_ASSERT_USER_THROW(
          testDecimalExpr<K>(
              {},
              exprStr,
              {makeFlatVector<InputType1>({in1}, inType1),
               makeFlatVector<InputType2>({in2}, inType2)}),
          exceptionMessage);
    };
  }

  // Same idiom for binary decimal expressions (e.g. "plus(c0, c1)"): one test
  // call per (in1, in2, expected) value.
  template <typename InputType1, typename InputType2, TypeKind K>
  std::function<
      void(InputType1, InputType2, typename TypeTraits<K>::NativeType)>
  makeBinaryDecimalExprTester(
      const TypePtr& inType1,
      const TypePtr& inType2,
      const TypePtr& outType,
      const std::string& exprStr) {
    using EvalType = typename TypeTraits<K>::NativeType;
    return [this, inType1, inType2, outType, exprStr](
               InputType1 in1, InputType2 in2, EvalType out) {
      auto v1 = makeFlatVector<InputType1>({in1}, inType1);
      auto v2 = makeFlatVector<InputType2>({in2}, inType2);
      testDecimalExpr<K>(
          makeFlatVector<EvalType>({out}, outType), exprStr, {v1, v2});
    };
  }

  // Same idiom for binary expressions with nullable inputs/output: one test
  // call per (in1, in2, expected) where any may be std::nullopt.
  template <typename InputType1, typename InputType2, TypeKind K>
  std::function<void(
      std::optional<InputType1>,
      std::optional<InputType2>,
      std::optional<typename TypeTraits<K>::NativeType>)>
  makeBinaryDecimalExprTesterNullable(
      const TypePtr& inType1,
      const TypePtr& inType2,
      const TypePtr& outType,
      const std::string& exprStr) {
    using EvalType = typename TypeTraits<K>::NativeType;
    return [this, inType1, inType2, outType, exprStr](
               std::optional<InputType1> in1,
               std::optional<InputType2> in2,
               std::optional<EvalType> expected) {
      auto v1 = makeNullableFlatVector<InputType1>({in1}, inType1);
      auto v2 = makeNullableFlatVector<InputType2>({in2}, inType2);
      auto expectedVec = makeNullableFlatVector<EvalType>({expected}, outType);
      testDecimalExpr<K>(expectedVec, exprStr, {v1, v2});
    };
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
} // namespace facebook::velox

TEST_F(DecimalArithmeticTest, add) {
  // Add short and short, returning long.
  {
    const auto inType = DECIMAL(18, 3);
    const auto outType = DECIMAL(19, 3);
    auto test =
        makeBinaryDecimalExprTester<int64_t, int64_t, TypeKind::HUGEINT>(
            inType, inType, outType, "plus(c0, c1)");
    test(1000, 1000, 2000);
    test(2000, 2000, 4000);
  }
  // Add short and long, returning long.
  {
    const auto shortType = DECIMAL(18, 3);
    const auto longType = DECIMAL(19, 3);
    const auto outType = DECIMAL(20, 3);
    auto test =
        makeBinaryDecimalExprTester<int64_t, int128_t, TypeKind::HUGEINT>(
            shortType, longType, outType, "plus(c0, c1)");
    test(1000, int128_t{1000}, 2000);
    test(2000, int128_t{2000}, 4000);
  }
  // Add long and short, returning long.
  {
    const auto longType = DECIMAL(19, 3);
    const auto shortType = DECIMAL(18, 3);
    const auto outType = DECIMAL(20, 3);
    auto test =
        makeBinaryDecimalExprTester<int128_t, int64_t, TypeKind::HUGEINT>(
            longType, shortType, outType, "plus(c0, c1)");
    test(int128_t{1000}, 1000, 2000);
    test(int128_t{2000}, 2000, 4000);
  }
  // Add long and long, returning long.
  {
    const auto inType = DECIMAL(19, 3);
    const auto outType = DECIMAL(20, 3);
    auto test =
        makeBinaryDecimalExprTester<int128_t, int128_t, TypeKind::HUGEINT>(
            inType, inType, outType, "c0 + c1");
    test(int128_t{1000}, int128_t{1000}, 2000);
    test(int128_t{2000}, int128_t{2000}, 4000);
  }
  // Add short and short, returning short.
  {
    const auto inType = DECIMAL(10, 3);
    const auto outType = DECIMAL(11, 3);
    auto test = makeBinaryDecimalExprTester<int64_t, int64_t, TypeKind::BIGINT>(
        inType, inType, outType, "c0 + c1");
    test(1000, 1000, 2000);
    test(2000, 2000, 4000);
  }
  // Constant and Flat arguments.
  {
    const auto inType = DECIMAL(10, 3);
    const auto outType = DECIMAL(11, 3);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "plus(1.00, c0)");
    test(1000, 2000);
    test(2000, 3000);
  }
  // Flat and Constant arguments.
  {
    const auto inType = DECIMAL(10, 3);
    const auto outType = DECIMAL(11, 3);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "plus(c0,1.00)");
    test(1000, 2000);
    test(2000, 3000);
  }
  // Nullable inputs (one call per row).
  {
    const auto inType = DECIMAL(10, 3);
    const auto outType = DECIMAL(11, 3);
    auto test =
        makeBinaryDecimalExprTesterNullable<int64_t, int64_t, TypeKind::BIGINT>(
            inType, inType, outType, "plus(c0, c1)");
    test(1000, 1000, 2000);
    test(2000, 2000, 4000);
    test(std::nullopt, 5000, std::nullopt);
    test(6000, std::nullopt, std::nullopt);
    test(std::nullopt, std::nullopt, std::nullopt);
  }

  // Addition overflow.
  {
    const auto inType = DECIMAL(38, 0);
    auto test = makeUnaryDecimalExprExceptionTester<
        int128_t,
        TypeKind::HUGEINT>(
        inType,
        "c0 + cast(1.00 as decimal(2,0))",
        "Decimal overflow. Value '100000000000000000000000000000000000000' is not in the range of Decimal Type");
    test(DecimalUtil::kLongDecimalMax);
  }
  // Rescaling LHS overflows.
  {
    const auto inType = DECIMAL(38, 0);
    auto test =
        makeUnaryDecimalExprExceptionTester<int128_t, TypeKind::HUGEINT>(
            inType,
            "c0 + 0.01",
            "Decimal overflow: 99999999999999999999999999999999999999 + 1");
    test(DecimalUtil::kLongDecimalMax);
  }
  // Rescaling RHS overflows.
  {
    const auto inType = DECIMAL(38, 0);
    auto test =
        makeUnaryDecimalExprExceptionTester<int128_t, TypeKind::HUGEINT>(
            inType,
            "0.01 + c0",
            "Decimal overflow: 1 + 99999999999999999999999999999999999999");
    test(DecimalUtil::kLongDecimalMax);
  }
}

TEST_F(DecimalArithmeticTest, subtract) {
  // Subtract short and short, returning long.
  {
    const auto inType = DECIMAL(18, 3);
    const auto outType = DECIMAL(19, 3);
    auto test =
        makeBinaryDecimalExprTester<int64_t, int64_t, TypeKind::HUGEINT>(
            inType, inType, outType, "minus(c0, c1)");
    test(1000, 500, 500);
    test(2000, 1000, 1000);
  }
  // Subtract short and long, returning long.
  {
    const auto shortType = DECIMAL(18, 3);
    const auto longType = DECIMAL(19, 3);
    const auto outType = DECIMAL(20, 3);
    auto test =
        makeBinaryDecimalExprTester<int64_t, int128_t, TypeKind::HUGEINT>(
            shortType, longType, outType, "minus(c0, c1)");
    test(1000, 100, 900);
    test(2000, 200, 1800);
  }
  // Subtract long and short, returning long.
  {
    const auto longType = DECIMAL(19, 3);
    const auto shortType = DECIMAL(18, 3);
    const auto outType = DECIMAL(20, 3);
    auto test =
        makeBinaryDecimalExprTester<int128_t, int64_t, TypeKind::HUGEINT>(
            longType, shortType, outType, "minus(c0, c1)");
    test(100, 1000, -900);
    test(200, 2000, -1800);
  }
  // Subtract long and long, returning long.
  {
    const auto inType1 = DECIMAL(19, 3);
    const auto inType2 = DECIMAL(19, 2);
    const auto outType = DECIMAL(21, 3);
    auto test =
        makeBinaryDecimalExprTester<int128_t, int128_t, TypeKind::HUGEINT>(
            inType1, inType2, outType, "c0 - c1");
    test(100, 100, -900);
    test(200, 200, -1800);
  }
  // Subtract short and short, returning short.
  {
    const auto inType = DECIMAL(10, 3);
    const auto outType = DECIMAL(11, 3);
    auto test = makeBinaryDecimalExprTester<int64_t, int64_t, TypeKind::BIGINT>(
        inType, inType, outType, "minus(c0, c1)");
    test(1000, 500, 500);
    test(2000, 1000, 1000);
  }
  // Constant and Flat arguments.
  {
    const auto inType = DECIMAL(10, 3);
    const auto outType = DECIMAL(11, 3);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "minus(1.00, c0)");
    test(1000, 0);
    test(2000, -1000);
  }
  // Flat and Constant arguments.
  {
    const auto inType = DECIMAL(10, 3);
    const auto outType = DECIMAL(11, 3);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "minus(c0, 1.00)");
    test(1000, 0);
    test(2000, 1000);
  }
  // Input with NULLs (one call per row).
  {
    const auto inType = DECIMAL(10, 3);
    const auto outType = DECIMAL(11, 3);
    auto test =
        makeBinaryDecimalExprTesterNullable<int64_t, int64_t, TypeKind::BIGINT>(
            inType, inType, outType, "minus(c0, c1)");
    test(3000, 1000, 2000);
    test(6000, 2000, 4000);
    test(std::nullopt, 5000, std::nullopt);
    test(6000, std::nullopt, std::nullopt);
    test(std::nullopt, std::nullopt, std::nullopt);
  }

  // Subtraction overflow.
  {
    const auto inType = DECIMAL(38, 0);
    auto test = makeUnaryDecimalExprExceptionTester<
        int128_t,
        TypeKind::HUGEINT>(
        inType,
        "c0 - cast(1.00 as decimal(2,0))",
        "Decimal overflow. Value '-100000000000000000000000000000000000000' is not in the range of Decimal Type");
    test(DecimalUtil::kLongDecimalMin);
  }
  // Rescaling LHS overflows.
  {
    const auto inType = DECIMAL(38, 0);
    auto test =
        makeUnaryDecimalExprExceptionTester<int128_t, TypeKind::HUGEINT>(
            inType,
            "c0 - 0.01",
            "Decimal overflow: -99999999999999999999999999999999999999 - 1");
    test(DecimalUtil::kLongDecimalMin);
  }
  // Rescaling RHS overflows.
  {
    const auto inType = DECIMAL(38, 0);
    auto test =
        makeUnaryDecimalExprExceptionTester<int128_t, TypeKind::HUGEINT>(
            inType,
            "0.01 - c0",
            "Decimal overflow: 1 - -99999999999999999999999999999999999999");
    test(DecimalUtil::kLongDecimalMin);
  }
}

TEST_F(DecimalArithmeticTest, multiply) {
  // Multiply short and short, returning long.
  {
    const auto inType = DECIMAL(17, 3);
    const auto outType = DECIMAL(34, 6);
    auto test =
        makeBinaryDecimalExprTester<int64_t, int64_t, TypeKind::HUGEINT>(
            inType, inType, outType, "multiply(c0, c1)");
    test(1000, 1000, 1000000);
    test(2000, 2000, 4000000);
  }
  // Multiply short and long, returning long.
  {
    const auto shortType = DECIMAL(17, 3);
    const auto longType = DECIMAL(20, 3);
    const auto outType = DECIMAL(37, 6);
    auto test =
        makeBinaryDecimalExprTester<int64_t, int128_t, TypeKind::HUGEINT>(
            shortType, longType, outType, "multiply(c0, c1)");
    test(1000, int128_t{1000}, 1000000);
    test(2000, int128_t{2000}, 4000000);
  }
  // Multiply long and short, returning long.
  {
    const auto longType = DECIMAL(20, 3);
    const auto shortType = DECIMAL(17, 3);
    const auto outType = DECIMAL(37, 6);
    auto test =
        makeBinaryDecimalExprTester<int128_t, int64_t, TypeKind::HUGEINT>(
            longType, shortType, outType, "multiply(c0, c1)");
    test(int128_t{1000}, 1000, 1000000);
    test(int128_t{2000}, 2000, 4000000);
  }
  // Multiply long and long, returning long.
  {
    const auto inType = DECIMAL(20, 3);
    const auto outType = DECIMAL(38, 6);
    auto test =
        makeBinaryDecimalExprTester<int128_t, int128_t, TypeKind::HUGEINT>(
            inType, inType, outType, "multiply(c0, c1)");
    test(int128_t{1000}, int128_t{1000}, 1000000);
    test(int128_t{2000}, int128_t{2000}, 4000000);
  }
  // Multiply short and short, returning short.
  {
    const auto inType = DECIMAL(6, 3);
    const auto outType = DECIMAL(12, 6);
    auto test = makeBinaryDecimalExprTester<int64_t, int64_t, TypeKind::BIGINT>(
        inType, inType, outType, "c0 * c1");
    test(1000, 1000, 1000000);
    test(2000, 2000, 4000000);
  }
  // Constant and Flat arguments.
  {
    const auto inType = DECIMAL(6, 3);
    const auto outType = DECIMAL(9, 5);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "1.00 * c0");
    test(1000, 100000);
    test(2000, 200000);
  }
  // Flat and Constant arguments.
  {
    const auto inType = DECIMAL(6, 3);
    const auto outType = DECIMAL(9, 5);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "c0 * 1.00");
    test(1000, 100000);
    test(2000, 200000);
  }

  // Long decimal limits
  {
    const auto inType = DECIMAL(38, 0);
    const auto overflowValue =
        HugeInt::build(0x08FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    auto test = makeUnaryDecimalExprExceptionTester<
        int128_t,
        TypeKind::HUGEINT>(
        inType,
        "c0 * cast(10.00 as decimal(2,0))",
        "Decimal overflow. Value '119630519620642428561342635425231011830' is not in the range of Decimal Type");
    test(overflowValue);
  }

  // Rescaling the final result overflows.
  {
    const auto inType = DECIMAL(38, 0);
    const auto overflowValue =
        HugeInt::build(0x08FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    auto test = makeUnaryDecimalExprExceptionTester<
        int128_t,
        TypeKind::HUGEINT>(
        inType,
        "c0 * cast(1.00 as decimal(2,1))",
        "Decimal overflow. Value '119630519620642428561342635425231011830' is not in the range of Decimal Type");
    test(overflowValue);
  }

  // The sum of input scales exceeds 38.
  VELOX_ASSERT_THROW(
      evaluate(
          "c0 * c0",
          makeRowVector(
              {makeFlatVector<int128_t>({1000, 2000}, DECIMAL(38, 30))})),
      "");
}

TEST_F(DecimalArithmeticTest, decimalDivTest) {
  // Divide short and short, returning long.
  {
    const auto inType = DECIMAL(17, 3);
    const auto outType = DECIMAL(20, 3);
    auto test =
        makeBinaryDecimalExprTester<int64_t, int64_t, TypeKind::HUGEINT>(
            inType, inType, outType, "divide(c0, c1)");
    test(500, 1000, 500);
    test(4000, 2000, 2000);
  }
  // Divide short and long, returning long.
  {
    const auto longType = DECIMAL(20, 2);
    const auto shortType = DECIMAL(17, 3);
    const auto outType = DECIMAL(24, 3);
    auto test =
        makeBinaryDecimalExprTester<int128_t, int64_t, TypeKind::HUGEINT>(
            longType, shortType, outType, "divide(c0, c1)");
    test(500, 1000, 5000);
    test(4000, 2000, 20000);
  }
  // Divide long and short, returning long.
  {
    const auto shortType = DECIMAL(17, 3);
    const auto longType = DECIMAL(20, 2);
    const auto outType = DECIMAL(19, 3);
    auto test =
        makeBinaryDecimalExprTester<int64_t, int128_t, TypeKind::HUGEINT>(
            shortType, longType, outType, "divide(c0, c1)");
    test(1000, 500, 200);
    test(2000, 4000, 50);
  }
  // Divide with large values.
  {
    const auto shortType = DECIMAL(17, 4);
    const auto longType = DECIMAL(21, 19);
    const auto outType = DECIMAL(38, 19);
    auto test =
        makeBinaryDecimalExprTester<int64_t, int128_t, TypeKind::HUGEINT>(
            shortType, longType, outType, "divide(c0, c1)");
    test(
        100,
        HugeInt::parse("50000000000000000000"),
        HugeInt::parse("20000000000000000"));
    test(
        200,
        HugeInt::parse("40000000000000000000"),
        HugeInt::parse("50000000000000000"));
  }
  // Divide long and long, returning long.
  {
    const auto inType1 = DECIMAL(20, 2);
    const auto inType2 = DECIMAL(20, 2);
    const auto outType = DECIMAL(22, 2);
    auto test =
        makeBinaryDecimalExprTester<int128_t, int128_t, TypeKind::HUGEINT>(
            inType1, inType2, outType, "divide(c0, c1)");
    test(2500, 500, 500);
    test(12000, 4000, 300);
  }
  // Divide short and short, returning short.
  {
    const auto inType1 = DECIMAL(5, 5);
    const auto inType2 = DECIMAL(5, 2);
    const auto outType = DECIMAL(7, 5);
    auto test = makeBinaryDecimalExprTester<int64_t, int64_t, TypeKind::BIGINT>(
        inType1, inType2, outType, "divide(c0, c1)");
    test(2500, 500, 500);
    test(12000, 4000, 300);
  }
  // Constant and Flat: 1.00 / c0.
  {
    const auto inType = DECIMAL(17, 3);
    const auto outType = DECIMAL(7, 3);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "1.00 / c0");
    test(1000, 1000);
    test(2000, 500);
  }
  // Flat and Constant: c0 / 2.00.
  {
    const auto inType = DECIMAL(17, 3);
    const auto outType = DECIMAL(19, 3);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::HUGEINT>(
        inType, outType, "c0 / 2.00");
    test(1000, 500);
    test(2000, 1000);
  }
  // Divide and round-up.
  {
    const auto inType = DECIMAL(2, 1);
    const auto outType = DECIMAL(3, 1);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "c0 / -6.0");
    test(-34, 6);
    test(5, -1);
    test(65, -11);
    test(90, -15);
    test(2, 0);
    test(-49, 8);
  }

  // Divide by zero.
  {
    const auto inType = DECIMAL(17, 3);
    auto test = makeUnaryDecimalExprExceptionTester<int64_t>(
        inType, "c0 / 0.0", "Division by zero");
    test(1000);
    test(2000);
  }

  // Long decimal limits.
  {
    const auto inType = DECIMAL(38, 0);
    auto test =
        makeUnaryDecimalExprExceptionTester<int128_t, TypeKind::HUGEINT>(
            inType,
            "c0 / 0.01",
            "Decimal overflow: 99999999999999999999999999999999999999 * 10000");
    test(DecimalUtil::kLongDecimalMax);
  }

  // Rescale factor > max precision (38).
  {
    const auto inType1 = DECIMAL(20, 1);
    const auto inType2 = DECIMAL(33, 32);
    auto test = makeBinaryDecimalExprExceptionTester<int128_t, int128_t>(
        inType1, inType2, "divide(c0, c1)", "Decimal overflow");
    test(5000, 5000);
    test(20000, 20000);
  }
}

TEST_F(DecimalArithmeticTest, decimalDivDifferentTypes) {
  {
    const auto inType1 = DECIMAL(12, 2);
    const auto inType2 = DECIMAL(19, 0);
    const auto outType = DECIMAL(12, 2);
    auto test =
        makeBinaryDecimalExprTester<int64_t, int128_t, TypeKind::BIGINT>(
            inType1, inType2, outType, "cast(c0 as decimal(12,2)) / c1");
    test(100, 100, 1);
    test(200, 200, 1);
    test(-300, 300, -1);
    test(400, 400, 1);
  }
  {
    const auto inType1 = DECIMAL(19, 0);
    const auto inType2 = DECIMAL(12, 2);
    const auto outType = DECIMAL(14, 2);
    auto test =
        makeBinaryDecimalExprTester<int128_t, int64_t, TypeKind::BIGINT>(
            inType1, inType2, outType, "cast(c0 as decimal(12,2)) / c1");
    test(1, 100, 100);
    test(2, 200, 100);
    test(3, -300, -100);
    test(4, 400, 100);
  }
}

TEST_F(DecimalArithmeticTest, decimalMod) {
  // short % short -> short.
  {
    const auto inType = DECIMAL(2, 1);
    const auto outType = DECIMAL(2, 1);
    auto test = makeBinaryDecimalExprTester<int64_t, int64_t, TypeKind::BIGINT>(
        inType, inType, outType, "mod(c0, c1)");
    test(0, 20, 0);
    test(50, 25, 0);
  }
  {
    const auto inType1 = DECIMAL(3, 1);
    const auto inType2 = DECIMAL(2, 1);
    const auto outType = DECIMAL(2, 1);
    auto test = makeBinaryDecimalExprTester<int64_t, int64_t, TypeKind::BIGINT>(
        inType1, inType2, outType, "mod(c0, c1)");
    test(13, 5, 3);
    test(-13, 5, -3);
    test(13, -5, 3);
    test(-13, -5, -3);
  }
  {
    const auto inType1 = DECIMAL(2, 1);
    const auto inType2 = DECIMAL(3, 2);
    const auto outType = DECIMAL(3, 2);
    auto test = makeBinaryDecimalExprTester<int64_t, int64_t, TypeKind::BIGINT>(
        inType1, inType2, outType, "mod(c0, c1)");
    test(50, 205, 90);
    test(-50, 255, -245);
    test(50, -255, 245);
    test(-50, -205, -90);
  }
  {
    const auto inType1 = DECIMAL(5, 3);
    const auto inType2 = DECIMAL(5, 2);
    const auto outType = DECIMAL(5, 3);
    auto test = makeBinaryDecimalExprTester<int64_t, int64_t, TypeKind::BIGINT>(
        inType1, inType2, outType, "mod(c0, c1)");
    test(2500, 600, 2500);
    test(-12000, 5000, -12000);
  }
  // short % long -> short.
  {
    const auto shortType = DECIMAL(17, 15);
    const auto longType = DECIMAL(20, 10);
    const auto outType = DECIMAL(17, 15);
    auto test =
        makeBinaryDecimalExprTester<int64_t, int128_t, TypeKind::BIGINT>(
            shortType, longType, outType, "mod(c0, c1)");
    test(1000, 13, 1000);
    test(-600, 17, -600);
    test(1000, -13, 1000);
    test(-600, -17, -600);
  }
  // long % short -> short.
  {
    const auto longType = DECIMAL(20, 10);
    const auto shortType = DECIMAL(17, 15);
    const auto outType = DECIMAL(17, 15);
    auto test =
        makeBinaryDecimalExprTester<int128_t, int64_t, TypeKind::BIGINT>(
            longType, shortType, outType, "mod(c0, c1)");
    test(500, 17, 8);
    test(-4000, 19, -11);
    test(500, -17, 8);
    test(-4000, -19, -11);
  }
  // short % long -> long.
  {
    const auto shortType = DECIMAL(17, 2);
    const auto longType = DECIMAL(30, 10);
    const auto outType = DECIMAL(25, 10);
    auto test =
        makeBinaryDecimalExprTester<int64_t, int128_t, TypeKind::HUGEINT>(
            shortType, longType, outType, "mod(c0, c1)");
    test(1000, 400, 0);
    test(-600, 38, -16);
    test(1000, -400, 0);
    test(-600, -38, -16);
  }
  // long % short -> long.
  {
    const auto longType = DECIMAL(30, 10);
    const auto shortType = DECIMAL(17, 2);
    const auto outType = DECIMAL(25, 10);
    auto test =
        makeBinaryDecimalExprTester<int128_t, int64_t, TypeKind::HUGEINT>(
            longType, shortType, outType, "mod(c0, c1)");
    test(500, 1000, 500);
    test(-4000, 2000, -4000);
    test(500, -1000, 500);
    test(-4000, -2000, -4000);
  }
  // long % long -> long.
  {
    const auto inType1 = DECIMAL(25, 5);
    const auto inType2 = DECIMAL(20, 2);
    const auto outType = DECIMAL(23, 5);
    auto test =
        makeBinaryDecimalExprTester<int128_t, int128_t, TypeKind::HUGEINT>(
            inType1, inType2, outType, "mod(c0, c1)");
    test(2500, 500, 2500);
    test(-12000, 4000, -12000);
    test(2500, -500, 2500);
    test(-12000, -4000, -12000);
  }

  // Modulus by zero: one test per value using unary exception tester.
  {
    const auto inType = DECIMAL(17, 3);
    auto test = makeUnaryDecimalExprExceptionTester<int64_t>(
        inType, "c0 % 0.0", "Modulus by zero");
    test(1000);
    test(2000);
  }
}

TEST_F(DecimalArithmeticTest, round) {
  // Round short decimals.
  {
    const auto inType = DECIMAL(3, 3);
    const auto outType = DECIMAL(1, 0);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "round(c0)");
    test(123, 0);
    test(542, 1);
    test(-999, -1);
    test(0, 0);
  }
  {
    const auto inType = DECIMAL(5, 1);
    const auto outType = DECIMAL(5, 0);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "round(c0)");
    test(11112, 1111);
    test(11115, 1112);
    test(-99989, -9999);
    test(99999, 10000);
  }
  // Round long decimals.
  {
    const auto inType = DECIMAL(19, 19);
    const auto outType = DECIMAL(1, 0);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::BIGINT>(
        inType, outType, "round(c0)");
    test(int128_t{1234567890123456789}, 0);
    test(int128_t{5000000000000000000}, 1);
    test(int128_t{-9000000000000000000}, -1);
    test(int128_t{0}, 0);
  }
  {
    const auto inType = DECIMAL(38, 1);
    const auto outType = DECIMAL(38, 0);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "round(c0)");
    test(DecimalUtil::kLongDecimalMax, DecimalUtil::kPowersOfTen[37]);
    test(DecimalUtil::kLongDecimalMin, -DecimalUtil::kPowersOfTen[37]);
  }
  // Min and max short decimals.
  {
    const auto inType = DECIMAL(15, 0);
    const auto outType = DECIMAL(15, 0);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "round(c0)");
    test(DecimalUtil::kShortDecimalMax, DecimalUtil::kShortDecimalMax);
    test(DecimalUtil::kShortDecimalMin, DecimalUtil::kShortDecimalMin);
  }
  // Min and max long decimals.
  {
    const auto inType = DECIMAL(38, 0);
    const auto outType = DECIMAL(38, 0);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "round(c0)");
    test(DecimalUtil::kLongDecimalMax, DecimalUtil::kLongDecimalMax);
    test(DecimalUtil::kLongDecimalMin, DecimalUtil::kLongDecimalMin);
  }
}

TEST_F(DecimalArithmeticTest, roundN) {
  // Round up to 'scale' decimal places.
  {
    const auto inType = DECIMAL(3, 3);
    const auto outType = DECIMAL(4, 3);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "round(c0, CAST(3 as integer))");
    test(123, 123);
    test(552, 552);
    test(-999, -999);
    test(0, 0);
  }
  // Round up to scale-1 decimal places.
  {
    const auto inType = DECIMAL(3, 3);
    const auto outType = DECIMAL(4, 3);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "round(c0, CAST(2 as integer))");
    test(123, 120);
    test(552, 550);
    test(-999, -1000);
    test(0, 0);
  }
  // Round up to 0 decimal places.
  {
    const auto inType = DECIMAL(3, 2);
    const auto outType = DECIMAL(4, 2);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "round(c0, CAST(0 as integer))");
    test(123, 100);
    test(552, 600);
    test(-999, -1000);
    test(0, 0);
  }
  // Round up to -1 decimal places.
  {
    const auto inType = DECIMAL(3, 1);
    const auto outType = DECIMAL(4, 1);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "round(c0, CAST(-1 as integer))");
    test(123, 100);
    test(552, 600);
    test(-999, -1000);
    test(0, 0);
  }
  // Round up to -2 decimal places.
  {
    const auto inType = DECIMAL(3, 1);
    const auto outType = DECIMAL(4, 1);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "round(c0, CAST(-2 as integer))");
    test(123, 0);
    test(552, 0);
    test(-999, 0);
    test(0, 0);
  }
  // Round long decimals to 14 decimal places.
  {
    const auto inType = DECIMAL(19, 19);
    const auto outType = DECIMAL(20, 19);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "round(c0, CAST(14 as integer))");
    test(int128_t{1234567890123456789}, int128_t{1234567890123500000});
    test(int128_t{5000000000000000000}, int128_t{5000000000000000000});
    test(int128_t{-999999999999999999}, int128_t{-10'000'000'000'000'000'00});
    test(int128_t{0}, int128_t{0});
  }
  // Round long decimals to -9 decimal places.
  {
    const auto inType = DECIMAL(19, 5);
    const auto outType = DECIMAL(20, 5);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "round(c0, CAST(-9 as integer))");
    test(int128_t{1234567890123456789}, int128_t{1234600000000000000});
    test(int128_t{5555555555555555555}, int128_t{5555600000000000000});
    test(int128_t{-999999999999999999}, int128_t{-1000000000000000000});
    test(int128_t{0}, int128_t{0});
  }
}

// Proposed new style for all these tests (and perhaps the SparkSQL equivalents)
// tests to make them more readable and maintainable. A helper function was
// added to the test class which returns a lambda which is then called to test
// the expression with specific input and expected output values.
// @TODO: Refactor other tests to use this style for clarity. See Issue #16464.

TEST_F(DecimalArithmeticTest, floorAndCeil) {
  // Short DECIMAL(3,2) -> Short DECIMAL(2.0).
  // e.g. floor(0.49) = 0, floor(-0.49) = -1, ceil(0.49) = 1, ceil(-0.49) = 0
  {
    const auto inType = DECIMAL(3, 2);
    const auto outType = DECIMAL(2, 0);
    auto testFloor = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "floor(c0)");
    auto testCeil = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "ceil(c0)");
    testFloor(0, 0);
    testFloor(1, 0);
    testFloor(-1, -1);
    testFloor(49, 0);
    testFloor(-49, -1);
    testFloor(50, 0);
    testFloor(-50, -1);
    testFloor(99, 0);
    testFloor(-99, -1);
    testCeil(0, 0);
    testCeil(1, 1);
    testCeil(-1, 0);
    testCeil(49, 1);
    testCeil(-49, 0);
    testCeil(50, 1);
    testCeil(-50, 0);
    testCeil(99, 1);
    testCeil(-99, 0);
  }

  // Short DECIMAL(5,2) -> Short DECIMAL(4,0).
  // e.g. floor(123.45) = 123, floor(-123.45) = -124, ceil(123.45) = 124,
  // ceil(-123.45) = -123
  {
    const auto inType = DECIMAL(5, 2);
    const auto outType = DECIMAL(4, 0);
    auto testFloor = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "floor(c0)");
    auto testCeil = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "ceil(c0)");
    testFloor(12300, 123);
    testFloor(-12300, -123);
    testFloor(12301, 123);
    testFloor(-12301, -124);
    testFloor(12345, 123);
    testFloor(-12345, -124);
    testFloor(12349, 123);
    testFloor(-12349, -124);
    testFloor(12350, 123);
    testFloor(-12350, -124);
    testFloor(12399, 123);
    testFloor(-12399, -124);
    testCeil(12300, 123);
    testCeil(-12300, -123);
    testCeil(12301, 124);
    testCeil(-12301, -123);
    testCeil(12345, 124);
    testCeil(-12345, -123);
    testCeil(12349, 124);
    testCeil(-12349, -123);
    testCeil(12350, 124);
    testCeil(-12350, -123);
    testCeil(12399, 124);
    testCeil(-12399, -123);
  }

  // Short DECIMAL(18,0) -> Short DECIMAL(18,0).
  // Min and max are unchanged if there are no fractional digits.
  {
    const auto inType = DECIMAL(18, 0);
    const auto outType = DECIMAL(18, 0);
    auto testFloor = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "floor(c0)");
    auto testCeil = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "ceil(c0)");
    testFloor(DecimalUtil::kShortDecimalMax, DecimalUtil::kShortDecimalMax);
    testFloor(DecimalUtil::kShortDecimalMin, DecimalUtil::kShortDecimalMin);
    testCeil(DecimalUtil::kShortDecimalMax, DecimalUtil::kShortDecimalMax);
    testCeil(DecimalUtil::kShortDecimalMin, DecimalUtil::kShortDecimalMin);
  }

  // Long DECIMAL(20,2) -> Long DECIMAL(19,0).
  // e.g. floor(0.49) = 0, floor(-0.49) = -1, ceil(0.49) = 1, ceil(-0.49) = 0
  {
    const auto inType = DECIMAL(20, 2);
    const auto outType = DECIMAL(19, 0);
    auto testFloor = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "floor(c0)");
    auto testCeil = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "ceil(c0)");
    testFloor(0, 0);
    testFloor(1, 0);
    testFloor(-1, -1);
    testFloor(49, 0);
    testFloor(-49, -1);
    testFloor(50, 0);
    testFloor(-50, -1);
    testFloor(99, 0);
    testFloor(-99, -1);
    testCeil(0, 0);
    testCeil(1, 1);
    testCeil(-1, 0);
    testCeil(49, 1);
    testCeil(-49, 0);
    testCeil(50, 1);
    testCeil(-50, 0);
    testCeil(99, 1);
    testCeil(-99, 0);
  }

  // Long DECIMAL(38,5) -> Long DECIMAL(34,0).
  // Min and max are rounded to the nearest whole number within the range of the
  // output type.
  {
    const auto inType = DECIMAL(38, 5);
    const auto outType = DECIMAL(34, 0);
    auto testFloor = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "floor(c0)");
    auto testCeil = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "ceil(c0)");
    testFloor(DecimalUtil::kLongDecimalMax, DecimalUtil::kPowersOfTen[33] - 1);
    testFloor(DecimalUtil::kLongDecimalMin, -DecimalUtil::kPowersOfTen[33]);
    testCeil(DecimalUtil::kLongDecimalMax, DecimalUtil::kPowersOfTen[33]);
    testCeil(DecimalUtil::kLongDecimalMin, -DecimalUtil::kPowersOfTen[33] + 1);
  }

  // Long DECIMAL(38,0) -> Long DECIMAL(38,0).
  // Min and max are unchanged if there are no fractional digits.
  {
    const auto inType = DECIMAL(38, 0);
    const auto outType = DECIMAL(38, 0);
    auto testFloor = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "floor(c0)");
    auto testCeil = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "ceil(c0)");
    testFloor(DecimalUtil::kLongDecimalMax, DecimalUtil::kLongDecimalMax);
    testFloor(DecimalUtil::kLongDecimalMin, DecimalUtil::kLongDecimalMin);
    testCeil(DecimalUtil::kLongDecimalMax, DecimalUtil::kLongDecimalMax);
    testCeil(DecimalUtil::kLongDecimalMin, DecimalUtil::kLongDecimalMin);
  }

  // Long DECIMAL(19,19) -> Short DECIMAL(1,0).
  // e.g. floor(0.1234567890123456789) = 0, ceil(0.1234567890123456789) = 1
  {
    const auto inType = DECIMAL(19, 19);
    const auto outType = DECIMAL(1, 0);
    auto testFloor = makeUnaryDecimalExprTester<int128_t, TypeKind::BIGINT>(
        inType, outType, "floor(c0)");
    auto testCeil = makeUnaryDecimalExprTester<int128_t, TypeKind::BIGINT>(
        inType, outType, "ceil(c0)");
    testFloor(int128_t{1234567890123456789}, 0);
    testFloor(int128_t{5000000000000000000}, 0);
    testFloor(int128_t{-9000000000000000000}, -1);
    testFloor(int128_t{-1000000000000000000}, -1);
    testFloor(int128_t{0}, 0);
    testCeil(int128_t{1234567890123456789}, 1);
    testCeil(int128_t{5000000000000000000}, 1);
    testCeil(int128_t{-9000000000000000000}, 0);
    testCeil(int128_t{-1000000000000000000}, 0);
    testCeil(int128_t{0}, 0);
  }
}

TEST_F(DecimalArithmeticTest, truncate) {
  // Truncate short decimals.
  {
    const auto inType = DECIMAL(3, 3);
    const auto outType = DECIMAL(1, 0);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "truncate(c0)");
    test(123, 0);
    test(542, 0);
    test(-999, 0);
    test(0, 0);
  }
  {
    const auto inType = DECIMAL(5, 1);
    const auto outType = DECIMAL(4, 0);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "truncate(c0)");
    test(11112, 1111);
    test(11115, 1111);
    test(-99989, -9998);
    test(99999, 9999);
  }
  // Truncate long decimals.
  {
    const auto inType = DECIMAL(19, 19);
    const auto outType = DECIMAL(1, 0);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::BIGINT>(
        inType, outType, "truncate(c0)");
    test(int128_t{1234567890123456789}, 0);
    test(int128_t{5000000000000000000}, 0);
    test(int128_t{-9000000000000000000}, 0);
    test(int128_t{0}, 0);
  }
  {
    const auto inType = DECIMAL(38, 1);
    const auto outType = DECIMAL(37, 0);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "truncate(c0)");
    test(DecimalUtil::kLongDecimalMax, DecimalUtil::kPowersOfTen[37] - 1);
    test(DecimalUtil::kLongDecimalMin, -DecimalUtil::kPowersOfTen[37] + 1);
  }
  // Min and max short decimals.
  {
    const auto inType = DECIMAL(15, 0);
    const auto outType = DECIMAL(15, 0);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "truncate(c0)");
    test(DecimalUtil::kShortDecimalMax, DecimalUtil::kShortDecimalMax);
    test(DecimalUtil::kShortDecimalMin, DecimalUtil::kShortDecimalMin);
  }
  // Min and max long decimals.
  {
    const auto inType = DECIMAL(38, 0);
    const auto outType = DECIMAL(38, 0);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "truncate(c0)");
    test(DecimalUtil::kLongDecimalMax, DecimalUtil::kLongDecimalMax);
    test(DecimalUtil::kLongDecimalMin, DecimalUtil::kLongDecimalMin);
  }
}

TEST_F(DecimalArithmeticTest, truncateN) {
  // Truncate to 3 decimal places.
  {
    const auto inType = DECIMAL(3, 3);
    const auto outType = DECIMAL(3, 3);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "truncate(c0, 3::integer)");
    test(123, 123);
    test(552, 552);
    test(-999, -999);
    test(0, 0);
  }
  // Truncate to 2 decimal places.
  {
    const auto inType = DECIMAL(3, 3);
    const auto outType = DECIMAL(3, 3);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "truncate(c0, 2::integer)");
    test(123, 120);
    test(552, 550);
    test(-999, -990);
    test(0, 0);
  }
  // Truncate to 0 decimal places.
  {
    const auto inType = DECIMAL(3, 2);
    const auto outType = DECIMAL(3, 2);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "truncate(c0, 0::integer)");
    test(123, 100);
    test(552, 500);
    test(-999, -900);
    test(0, 0);
  }
  // Truncate to -1 decimal places.
  {
    const auto inType = DECIMAL(3, 1);
    const auto outType = DECIMAL(3, 1);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "truncate(c0, '-1'::integer)");
    test(123, 100);
    test(552, 500);
    test(-999, -900);
    test(0, 0);
  }
  // Truncate to -2 decimal places.
  {
    const auto inType = DECIMAL(3, 1);
    const auto outType = DECIMAL(3, 1);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "truncate(c0, '-2'::integer)");
    test(123, 0);
    test(552, 0);
    test(-999, 0);
    test(0, 0);
  }
  // Truncate long decimals to 14 decimal places.
  {
    const auto inType = DECIMAL(19, 19);
    const auto outType = DECIMAL(19, 19);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "truncate(c0, 14::integer)");
    test(int128_t{1234567890123456789}, int128_t{1234567890123400000});
    test(int128_t{5000000000000000000}, int128_t{5000000000000000000});
    test(int128_t{-999999999999999999}, int128_t{-999999999999900000});
    test(int128_t{0}, int128_t{0});
  }
  // Truncate long decimals to -9 decimal places.
  {
    const auto inType = DECIMAL(19, 5);
    const auto outType = DECIMAL(19, 5);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "truncate(c0, '-9'::integer)");
    test(int128_t{1234567890123456789}, int128_t{1234500000000000000});
    test(int128_t{5555555555555555555}, int128_t{5555500000000000000});
    test(int128_t{-999999999999999999}, int128_t{-999900000000000000});
    test(int128_t{0}, int128_t{0});
  }
}

TEST_F(DecimalArithmeticTest, abs) {
  {
    const auto inType = DECIMAL(5, 1);
    const auto outType = DECIMAL(5, 1);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "abs(c0)");
    test(-1111, 1111);
    test(1112, 1112);
    test(-9999, 9999);
    test(0, 0);
  }
  {
    const auto inType = DECIMAL(19, 19);
    const auto outType = DECIMAL(19, 19);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "abs(c0)");
    test(int128_t{-11111111}, int128_t{11111111});
    test(int128_t{11112112}, int128_t{11112112});
    test(int128_t{-99999999}, int128_t{99999999});
    test(int128_t{0}, int128_t{0});
  }
  // Min and max short decimals.
  {
    const auto inType = DECIMAL(15, 0);
    const auto outType = DECIMAL(15, 0);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "abs(c0)");
    test(DecimalUtil::kShortDecimalMin, DecimalUtil::kShortDecimalMax);
  }
  {
    const auto inType = DECIMAL(15, 15);
    const auto outType = DECIMAL(15, 15);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "abs(c0)");
    test(DecimalUtil::kShortDecimalMin, DecimalUtil::kShortDecimalMax);
  }
  // Min and max long decimals.
  {
    const auto inType = DECIMAL(38, 0);
    const auto outType = DECIMAL(38, 0);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "abs(c0)");
    test(DecimalUtil::kLongDecimalMin, DecimalUtil::kLongDecimalMax);
  }
  {
    const auto inType = DECIMAL(38, 38);
    const auto outType = DECIMAL(38, 38);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "abs(c0)");
    test(DecimalUtil::kLongDecimalMin, DecimalUtil::kLongDecimalMax);
  }
}

TEST_F(DecimalArithmeticTest, negate) {
  {
    const auto inType = DECIMAL(5, 1);
    const auto outType = DECIMAL(5, 1);
    auto test = makeUnaryDecimalExprTester<int64_t, TypeKind::BIGINT>(
        inType, outType, "negate(c0)");
    test(-1111, 1111);
    test(1112, -1112);
    test(-9999, 9999);
    test(0, 0);
  }
  {
    const auto inType = DECIMAL(38, 19);
    const auto outType = DECIMAL(38, 19);
    auto test = makeUnaryDecimalExprTester<int128_t, TypeKind::HUGEINT>(
        inType, outType, "negate(c0)");
    test(int128_t{-11111111}, int128_t{11111111});
    test(int128_t{11112112}, int128_t{-11112112});
    test(int128_t{-99999999}, int128_t{99999999});
    test(-DecimalUtil::kLongDecimalMax, DecimalUtil::kLongDecimalMax);
    test(-DecimalUtil::kLongDecimalMin, DecimalUtil::kLongDecimalMin);
  }
}
