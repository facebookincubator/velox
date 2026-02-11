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

#include "velox/expression/PrestoCastKernel.h"

namespace facebook::velox::exec {
template <typename FromNativeType>
VectorPtr PrestoCastKernel::applyDecimalToVarcharCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& fromType,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);
  const auto simpleInput = input.as<SimpleVector<FromNativeType>>();
  int precision = getDecimalPrecisionScale(*fromType).first;
  int scale = getDecimalPrecisionScale(*fromType).second;
  auto rowSize = DecimalUtil::maxStringViewSize(precision, scale);
  auto flatResult = result->asFlatVector<StringView>();
  if (StringView::isInline(rowSize)) {
    char inlined[StringView::kInlineSize];
    applyToSelectedNoThrowLocal(
        rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
          auto actualSize = DecimalUtil::castToString<FromNativeType>(
              simpleInput->valueAt(row), scale, rowSize, inlined);
          flatResult->setNoCopy(row, StringView(inlined, actualSize));
        });
    return result;
  }

  Buffer* buffer =
      flatResult->getBufferWithSpace(rows.countSelected() * rowSize);
  char* rawBuffer = buffer->asMutable<char>() + buffer->size();

  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
        auto actualSize = DecimalUtil::castToString<FromNativeType>(
            simpleInput->valueAt(row), scale, rowSize, rawBuffer);
        flatResult->setNoCopy(row, StringView(rawBuffer, actualSize));
        if (!StringView::isInline(actualSize)) {
          // If string view is inline, corresponding bytes on the raw string
          // buffer are not needed.
          rawBuffer += actualSize;
        }
      });
  // Update the exact buffer size.
  buffer->setSize(rawBuffer - buffer->asMutable<char>());
  return result;
}

template <typename FromNativeType, TypeKind ToKind>
VectorPtr PrestoCastKernel::applyDecimalToFloatCast(
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
  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](int row) {
        const auto output =
            util::Converter<ToKind>::tryCast(simpleInput->valueAt(row))
                .thenOrThrow(folly::identity, [&](const Status& status) {
                  VELOX_USER_FAIL("{}", status.message());
                });
        resultBuffer[row] = output / scaleFactor;
      });
  return result;
}

template <typename FromNativeType, TypeKind ToKind>
VectorPtr PrestoCastKernel::applyDecimalToIntegralCast(
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
  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
        auto value = simpleInput->valueAt(row);
        auto integralPart = value / scaleFactor;
        auto fractionPart = value % scaleFactor;
        auto sign = value >= 0 ? 1 : -1;
        bool needsRoundUp =
            (scaleFactor != 1) && (sign * fractionPart >= (scaleFactor >> 1));
        integralPart += needsRoundUp ? sign : 0;

        if (integralPart > std::numeric_limits<To>::max() ||
            integralPart < std::numeric_limits<To>::min()) {
          if (setNullInResultAtError) {
            result->setNull(row, true);
          } else if (context.captureErrorDetails()) {
            context.setStatus(
                row,
                Status::UserError(
                    "{}",
                    makeErrorMessage(input, row, toType, "Out of bounds.")));
          } else {
            context.setStatus(row, Status::UserError());
          }
          return;
        }

        resultBuffer[row] = static_cast<To>(integralPart);
      });

  return result;
}

template <typename FromNativeType>
VectorPtr PrestoCastKernel::applyDecimalToBooleanCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);
  auto resultBuffer =
      result->asUnchecked<FlatVector<bool>>()->mutableRawValues<uint64_t>();
  const auto simpleInput = input.as<SimpleVector<FromNativeType>>();
  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](int row) {
        auto value = simpleInput->valueAt(row);
        bits::setBit(resultBuffer, row, value != 0);
      });
  return result;
}

template <typename ToNativeType, typename FromNativeType>
VectorPtr PrestoCastKernel::applyIntToDecimalCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);

  auto sourceVector = input.as<SimpleVector<FromNativeType>>();
  auto resultRawBuffer =
      result->asUnchecked<FlatVector<ToNativeType>>()->mutableRawValues();
  const auto& toPrecisionScale = getDecimalPrecisionScale(*toType);
  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
        auto rescaledValue =
            DecimalUtil::rescaleInt<FromNativeType, ToNativeType>(
                sourceVector->valueAt(row),
                toPrecisionScale.first,
                toPrecisionScale.second);
        if (rescaledValue.has_value()) {
          resultRawBuffer[row] = rescaledValue.value();
        } else {
          result->setNull(row, true);
        }
      });

  return result;
}

template <typename FromNativeType, typename ToNativeType>
VectorPtr PrestoCastKernel::applyDecimalToDecimalCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);

  auto sourceVector = input.as<SimpleVector<FromNativeType>>();
  auto resultRawBuffer =
      result->asUnchecked<FlatVector<ToNativeType>>()->mutableRawValues();
  const auto& fromPrecisionScale = getDecimalPrecisionScale(*input.type());
  const auto& toPrecisionScale = getDecimalPrecisionScale(*toType);

  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
        ToNativeType rescaledValue;
        const auto status =
            DecimalUtil::rescaleWithRoundUp<FromNativeType, ToNativeType>(
                sourceVector->valueAt(row),
                fromPrecisionScale.first,
                fromPrecisionScale.second,
                toPrecisionScale.first,
                toPrecisionScale.second,
                rescaledValue);
        if (status.ok()) {
          resultRawBuffer[row] = rescaledValue;
        } else {
          if (setNullInResultAtError) {
            result->setNull(row, true);
          } else {
            context.setStatus(row, status);
          }
        }
      });

  return result;
}

template <typename ToNativeType, typename FromNativeType>
VectorPtr PrestoCastKernel::applyFloatingPointToDecimalCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);

  const auto floatingInput = input.as<SimpleVector<FromNativeType>>();
  auto rawResults =
      result->asUnchecked<FlatVector<ToNativeType>>()->mutableRawValues();
  const auto toPrecisionScale = getDecimalPrecisionScale(*toType);

  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
        ToNativeType output;
        const auto status =
            DecimalUtil::rescaleFloatingPoint<FromNativeType, ToNativeType>(
                floatingInput->valueAt(row),
                toPrecisionScale.first,
                toPrecisionScale.second,
                output);
        if (status.ok()) {
          rawResults[row] = output;
        } else {
          if (setNullInResultAtError) {
            result->setNull(row, true);
          } else if (context.captureErrorDetails()) {
            context.setStatus(
                row,
                Status::UserError(
                    "{}",
                    makeErrorMessage(input, row, toType, status.message())));
          } else {
            context.setStatus(row, Status::UserError());
          }
        }
      });

  return result;
}

template <typename ToNativeType>
VectorPtr PrestoCastKernel::applyVarcharToDecimalCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);

  auto sourceVector = input.as<SimpleVector<StringView>>();
  auto rawBuffer =
      result->asUnchecked<FlatVector<ToNativeType>>()->mutableRawValues();
  const auto toPrecisionScale = getDecimalPrecisionScale(*toType);

  rows.applyToSelected([&](auto row) {
    ToNativeType decimalValue;
    const auto status = DecimalUtil::castFromString<ToNativeType>(
        sourceVector->valueAt(row),
        toPrecisionScale.first,
        toPrecisionScale.second,
        decimalValue);
    if (status.ok()) {
      rawBuffer[row] = decimalValue;
    } else {
      if (setNullInResultAtError) {
        result->setNull(row, true);
      } else if (context.captureErrorDetails()) {
        context.setStatus(
            row,
            Status::UserError(
                "{}", makeErrorMessage(input, row, toType, status.message())));
      } else {
        context.setStatus(row, Status::UserError());
      }
    }
  });

  return result;
}
} // namespace facebook::velox::exec
