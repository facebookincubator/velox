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

#include "velox/functions/sparksql/specialforms/SparkCastKernel.h"

namespace facebook::velox::functions::sparksql {
template <typename FromNativeType, TypeKind ToKind>
VectorPtr SparkCastKernel::applyDecimalToIntegralCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& fromType,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  using To = typename TypeTraits<ToKind>::NativeType;

  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);
  auto resultBuffer = result->asUnchecked<FlatVector<To>>()->mutableRawValues();
  const auto precisionScale = getDecimalPrecisionScale(*fromType);
  const auto simpleInput = input.as<SimpleVector<FromNativeType>>();
  const auto scaleFactor = DecimalUtil::kPowersOfTen[precisionScale.second];
  if (allowOverflow_) {
    applyToSelectedNoThrowLocal(
        rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
          resultBuffer[row] =
              static_cast<To>(simpleInput->valueAt(row) / scaleFactor);
        });
  } else {
    applyToSelectedNoThrowLocal(
        rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
          auto value = simpleInput->valueAt(row);
          auto integralPart = value / scaleFactor;

          if (integralPart > std::numeric_limits<To>::max() ||
              integralPart < std::numeric_limits<To>::min()) {
            setError(
                input,
                context,
                *result,
                row,
                "Out of bounds.",
                setNullInResultAtError);
            return;
          }

          resultBuffer[row] = static_cast<To>(integralPart);
        });
  }
  return result;
}

template <typename ToNativeType>
void SparkCastKernel::applyVarcharToDecimalCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError,
    VectorPtr& result) const {
  auto sourceVector = input.as<SimpleVector<StringView>>();
  auto rawBuffer =
      result->asUnchecked<FlatVector<ToNativeType>>()->mutableRawValues();
  const auto toPrecisionScale = getDecimalPrecisionScale(*toType);

  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
        ToNativeType decimalValue;
        const auto status = DecimalUtil::castFromString<ToNativeType>(
            removeWhiteSpaces(sourceVector->valueAt(row)),
            toPrecisionScale.first,
            toPrecisionScale.second,
            decimalValue);
        if (status.ok()) {
          rawBuffer[row] = decimalValue;
        } else {
          setError(
              input,
              context,
              *result,
              row,
              status.message(),
              setNullInResultAtError);
        }
      });
}
} // namespace facebook::velox::functions::sparksql
