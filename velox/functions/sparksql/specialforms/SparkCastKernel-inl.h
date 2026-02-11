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

template <TypeKind FromTypeKind>
VectorPtr SparkCastKernel::castToBooleanImpl(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  if constexpr (
      FromTypeKind == TypeKind::TINYINT || FromTypeKind == TypeKind::SMALLINT ||
      FromTypeKind == TypeKind::INTEGER || FromTypeKind == TypeKind::BIGINT) {
    if (!allowOverflow_) {
      return exec::PrestoCastKernel::castToBoolean(
          rows, input, context, toType, setNullInResultAtError);
    }

    VectorPtr result;
    initializeResultVector(rows, toType, context, result);

    auto sourceVector =
        input.as<SimpleVector<typename TypeTraits<FromTypeKind>::NativeType>>();
    auto* resultFlatVector = result->as<FlatVector<bool>>();

    applyToSelectedNoThrowLocal(
        rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
          resultFlatVector->set(row, sourceVector->valueAt(row));
        });

    return result;
  } else if constexpr (
      FromTypeKind == TypeKind::REAL || FromTypeKind == TypeKind::DOUBLE) {
    if (!allowOverflow_) {
      return exec::PrestoCastKernel::castToBoolean(
          rows, input, context, toType, setNullInResultAtError);
    }

    VectorPtr result;
    initializeResultVector(rows, toType, context, result);

    auto sourceVector =
        input.as<SimpleVector<typename TypeTraits<FromTypeKind>::NativeType>>();
    auto* resultFlatVector = result->as<FlatVector<bool>>();

    applyToSelectedNoThrowLocal(
        rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
          auto value = sourceVector->valueAt(row);
          resultFlatVector->set(row, !std::isnan(value) && value != 0);
        });

    return result;
  } else if constexpr (FromTypeKind == TypeKind::HUGEINT) {
    if (!allowOverflow_) {
      return exec::PrestoCastKernel::castToBoolean(
          rows, input, context, toType, setNullInResultAtError);
    }

    VectorPtr result =
        BaseVector::createNullConstant(toType, rows.end(), context.pool());

    if (!setNullInResultAtError) {
      context.setStatuses(
          rows, Status::UserError("Conversion to BOOLEAN is not supported"));
    }

    return result;
  } else if constexpr (
      FromTypeKind == TypeKind::VARCHAR ||
      FromTypeKind == TypeKind::VARBINARY) {
    VectorPtr result;
    initializeResultVector(rows, toType, context, result);
    auto* resultFlatVector = result->as<FlatVector<bool>>();

    const auto simpleInput = input.as<SimpleVector<StringView>>();

    applyToSelectedNoThrowLocal(
        rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
          const auto& TU = static_cast<int (*)(int)>(std::toupper);

          StringView inputStr = simpleInput->valueAt(row);
          inputStr = removeWhiteSpaces(inputStr);
          const auto len = inputStr.size();
          const auto data = inputStr.data();

          if (len == 1) {
            auto character = TU(data[0]);
            if (character == 'T' || character == '1' || character == 'Y') {
              resultFlatVector->set(row, true);
              return;
            }
            if (character == 'F' || character == '0' || character == 'N') {
              resultFlatVector->set(row, false);
              return;
            }
          }

          // Case-insensitive 'true'.
          if ((len == 4) && (TU(data[0]) == 'T') && (TU(data[1]) == 'R') &&
              (TU(data[2]) == 'U') && (TU(data[3]) == 'E')) {
            resultFlatVector->set(row, true);
            return;
          }

          // Case-insensitive 'false'.
          if ((len == 5) && (TU(data[0]) == 'F') && (TU(data[1]) == 'A') &&
              (TU(data[2]) == 'L') && (TU(data[3]) == 'S') &&
              (TU(data[4]) == 'E')) {
            resultFlatVector->set(row, false);
            return;
          }

          // Case-insensitive 'yes'.
          if ((len == 3) && (TU(data[0]) == 'Y') && (TU(data[1]) == 'E') &&
              (TU(data[2]) == 'S')) {
            resultFlatVector->set(row, true);
            return;
          }

          // Case-insensitive 'no'.
          if ((len == 2) && (TU(data[0]) == 'N') && (TU(data[1]) == 'O')) {
            resultFlatVector->set(row, false);
            return;
          }

          if (setNullInResultAtError) {
            resultFlatVector->setNull(row, true);
          } else if (context.captureErrorDetails()) {
            context.setStatus(
                row,
                Status::UserError(
                    "{} Cannot cast {} to BOOLEAN",
                    makeErrorMessage(input, row, BOOLEAN()),
                    std::string_view(data, len)));
          } else {
            context.setStatus(row, Status::UserError());
          }
        });

    return result;
  } else {
    return exec::PrestoCastKernel::castToBoolean(
        rows, input, context, toType, setNullInResultAtError);
  }
}
} // namespace facebook::velox::functions::sparksql
