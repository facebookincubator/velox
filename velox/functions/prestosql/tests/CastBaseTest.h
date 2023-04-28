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
#pragma once

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/Expressions.h"
#include "velox/core/ITypedExpr.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/vector/tests/TestingDictionaryFunction.h"

namespace facebook::velox::functions::test {

using namespace facebook::velox::test;

class CastBaseTest : public FunctionBaseTest {
 protected:
  CastBaseTest() {
    exec::registerVectorFunction(
        "testing_dictionary",
        test::TestingDictionaryFunction::signatures(),
        std::make_unique<test::TestingDictionaryFunction>());
  }

  // Build an ITypedExpr for cast(fromType as toType).
  core::TypedExprPtr buildCastExpr(
      const TypePtr& fromType,
      const TypePtr& toType,
      bool isTryCast) {
    core::TypedExprPtr inputField =
        std::make_shared<const core::FieldAccessTypedExpr>(fromType, "c0");
    return std::make_shared<const core::CastTypedExpr>(
        toType, std::vector<core::TypedExprPtr>{inputField}, isTryCast);
  }

  // Evaluate cast(fromType as toType) and return the result vector.
  template <typename TTo>
  VectorPtr evaluateCast(
      const TypePtr& fromType,
      const TypePtr& toType,
      const RowVectorPtr& input,
      bool isTryCast = false) {
    auto castExpr = buildCastExpr(fromType, toType, isTryCast);

    if constexpr (std::is_same_v<TTo, ComplexType>) {
      return evaluate(castExpr, input);
    } else {
      return evaluate<SimpleVector<EvalType<TTo>>>(castExpr, input);
    }
  }

  // Evaluate cast(fromType as toType) and verify the result matches the
  // expected one.
  template <typename TTo>
  void evaluateAndVerify(
      const TypePtr& fromType,
      const TypePtr& toType,
      const RowVectorPtr& input,
      const VectorPtr& expected,
      bool isTryCast = false) {
    auto result = evaluateCast<TTo>(fromType, toType, input, isTryCast);
    assertEqualVectors(expected, result);
  }

  // Build an ITypedExpr for cast(testing_dictionary(fromType) as toType).
  core::TypedExprPtr buildCastExprWithDictionaryInput(
      const TypePtr& fromType,
      const TypePtr& toType,
      bool isTryCast) {
    core::TypedExprPtr inputField =
        std::make_shared<const core::FieldAccessTypedExpr>(fromType, "c0");
    core::TypedExprPtr callExpr = std::make_shared<const core::CallTypedExpr>(
        fromType,
        std::vector<core::TypedExprPtr>{inputField},
        "testing_dictionary");
    return std::make_shared<const core::CastTypedExpr>(
        toType, std::vector<core::TypedExprPtr>{callExpr}, isTryCast);
  }

  // Evaluate cast(testing_dictionary(fromType) as toType) and verify the result
  // matches the expected one. Values in expected should correspond to values in
  // input at the same rows.
  template <typename TTo>
  void evaluateAndVerifyDictEncoding(
      const TypePtr& fromType,
      const TypePtr& toType,
      const RowVectorPtr& input,
      const VectorPtr& expected,
      bool isTryCast = false) {
    auto castExpr =
        buildCastExprWithDictionaryInput(fromType, toType, isTryCast);

    VectorPtr result;
    if constexpr (std::is_same_v<TTo, ComplexType>) {
      result = evaluate(castExpr, input);
    } else {
      result = evaluate<SimpleVector<EvalType<TTo>>>(castExpr, input);
    }

    auto indices = test::makeIndicesInReverse(expected->size(), pool());
    assertEqualVectors(wrapInDictionary(indices, expected), result);
  }

  // Evaluate try(cast(testing_dictionary(fromType) as toType)) and verify the
  // result matches the expected one. Values in expected should correspond to
  // values in input at the same rows.
  void evaluateAndVerifyCastInTryDictEncoding(
      const TypePtr& fromType,
      const TypePtr& toType,
      const RowVectorPtr& input,
      const VectorPtr& expected) {
    auto castExpr = buildCastExprWithDictionaryInput(fromType, toType, false);
    core::TypedExprPtr tryExpr = std::make_shared<const core::CallTypedExpr>(
        toType, std::vector<core::TypedExprPtr>{castExpr}, "try");

    auto result = evaluate(tryExpr, input);
    auto indices = test::makeIndicesInReverse(expected->size(), pool());
    assertEqualVectors(wrapInDictionary(indices, expected), result);
  }

  template <typename TTo>
  void testCast(
      const TypePtr& fromType,
      const TypePtr& toType,
      const VectorPtr& input,
      const VectorPtr& expected) {
    SCOPED_TRACE(fmt::format(
        "Cast from {} to {}", fromType->toString(), toType->toString()));
    // Test with flat encoding.
    {
      SCOPED_TRACE("Flat encoding");
      evaluateAndVerify<TTo>(
          fromType, toType, makeRowVector({input}), expected);
      evaluateAndVerify<TTo>(
          fromType, toType, makeRowVector({input}), expected, true);
    }

    // Test with constant encoding that repeats the first element five times.
    {
      SCOPED_TRACE("Constant encoding");
      auto constInput = BaseVector::wrapInConstant(5, 0, input);
      auto constExpected = BaseVector::wrapInConstant(5, 0, expected);

      evaluateAndVerify<TTo>(
          fromType, toType, makeRowVector({constInput}), constExpected);
      evaluateAndVerify<TTo>(
          fromType, toType, makeRowVector({constInput}), constExpected, true);
    }

    // Test with dictionary encoding that reverses the indices.
    {
      SCOPED_TRACE("Dictionary encoding");
      evaluateAndVerifyDictEncoding<TTo>(
          fromType, toType, makeRowVector({input}), expected);
      evaluateAndVerifyDictEncoding<TTo>(
          fromType, toType, makeRowVector({input}), expected, true);
    }
  }

  template <typename TFrom, typename TTo>
  void testCast(
      const TypePtr& fromType,
      const TypePtr& toType,
      std::vector<std::optional<TFrom>> input,
      std::vector<std::optional<TTo>> expected) {
    auto inputVector = makeNullableFlatVector<TFrom>(input, fromType);
    auto expectedVector = makeNullableFlatVector<TTo>(expected, toType);

    testCast<TTo>(fromType, toType, inputVector, expectedVector);
  }

  template <typename TFrom, typename TTo>
  void testThrow(
      const TypePtr& fromType,
      const TypePtr& toType,
      const std::vector<std::optional<TFrom>>& input,
      const std::string& expectedErrorMessage) {
    VELOX_ASSERT_THROW(
        evaluateCast<TTo>(
            fromType,
            toType,
            makeRowVector({makeNullableFlatVector<TFrom>(input, fromType)})),
        expectedErrorMessage);
  }
};

} // namespace facebook::velox::functions::test
