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
#include "velox/functions/lib/string/StringImpl.h"

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

template <typename T>
FOLLY_ALWAYS_INLINE Expected<T> PrestoCastKernel::doCastToFloatingPoint(
    const StringView& data) {
  static const T kNan = std::numeric_limits<T>::quiet_NaN();
  static const double_conversion::StringToDoubleConverter
      stringToDoubleConverter{
          double_conversion::StringToDoubleConverter::ALLOW_TRAILING_SPACES,
          /*empty_string_value*/ kNan,
          /*junk_string_value*/ kNan,
          "Infinity",
          "NaN"};
  int processedCharactersCount;
  T result;
  auto* begin = std::find_if_not(data.begin(), data.end(), [](char c) {
    return functions::stringImpl::isAsciiWhiteSpace(c);
  });
  auto length = data.end() - begin;
  if (length == 0) {
    // 'data' only contains white spaces.
    return folly::makeUnexpected(Status::UserError());
  }
  if constexpr (std::is_same_v<T, float>) {
    result = stringToDoubleConverter.StringToFloat(
        begin, length, &processedCharactersCount);
  } else if constexpr (std::is_same_v<T, double>) {
    result = stringToDoubleConverter.StringToDouble(
        begin, length, &processedCharactersCount);
  }
  // Since we already removed leading space, if processedCharactersCount == 0,
  // it means the remaining string is either empty or a junk string. So return a
  // user error in this case.
  if UNLIKELY (processedCharactersCount == 0) {
    return folly::makeUnexpected(Status::UserError());
  }
  return result;
}

/// The per-row level Kernel
/// @tparam ToKind The cast target type
/// @tparam FromKind The expression type
/// @tparam TPolicy The policy used by the cast
/// @param row The index of the current row
/// @param input The input vector (of type FromKind)
/// @param result The output vector (of type ToKind)
template <TypeKind ToKind, TypeKind FromKind, typename TPolicy>
FOLLY_ALWAYS_INLINE void PrestoCastKernel::applyCastPrimitives(
    vector_size_t row,
    EvalCtx& context,
    const SimpleVector<typename TypeTraits<FromKind>::NativeType>* input,
    bool setNullInResultAtError,
    FlatVector<typename TypeTraits<ToKind>::NativeType>* result) const {
  auto inputRowValue = input->valueAt(row);

  // Optimize empty input strings casting by avoiding throwing exceptions.
  if constexpr (is_string_kind(FromKind)) {
    if constexpr (
        TypeTraits<ToKind>::isPrimitiveType &&
        TypeTraits<ToKind>::isFixedWidth) {
      if (inputRowValue.size() == 0) {
        setError(
            *input,
            context,
            *result,
            row,
            "Empty string",
            setNullInResultAtError);
        return;
      }
    }

    if constexpr (ToKind == TypeKind::REAL || ToKind == TypeKind::DOUBLE) {
      const auto castResult =
          doCastToFloatingPoint<typename TypeTraits<ToKind>::NativeType>(
              inputRowValue);
      if (castResult.hasError()) {
        setError(
            *input,
            context,
            *result,
            row,
            castResult.error().message(),
            setNullInResultAtError);
      } else {
        result->set(row, castResult.value());
      }
      return;
    }

    if constexpr (
        ToKind == TypeKind::TINYINT || ToKind == TypeKind::SMALLINT ||
        ToKind == TypeKind::INTEGER || ToKind == TypeKind::BIGINT ||
        ToKind == TypeKind::HUGEINT) {
      if constexpr (TPolicy::throwOnUnicode) {
        if (!functions::stringCore::isAscii(
                inputRowValue.data(), inputRowValue.size())) {
          setError(
              *input,
              context,
              *result,
              row,
              "Unicode characters are not supported for conversion to integer types",
              setNullInResultAtError);
          return;
        }
      }
    }
  }

  const auto castResult =
      util::Converter<ToKind, void, TPolicy>::tryCast(inputRowValue);
  if (castResult.hasError()) {
    setError(
        *input,
        context,
        *result,
        row,
        castResult.error().message(),
        setNullInResultAtError);
    return;
  }

  const auto& output = castResult.value();

  result->set(row, output);
}

template <TypeKind ToKind, TypeKind FromKind>
FOLLY_ALWAYS_INLINE VectorPtr
PrestoCastKernel::applyCastPrimitivesPolicyDispatch(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  using To = typename TypeTraits<ToKind>::NativeType;
  using From = typename TypeTraits<FromKind>::NativeType;

  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  auto* resultFlatVector = result->as<FlatVector<To>>();
  auto* inputSimpleVector = input.as<SimpleVector<From>>();

  if (legacyCast_) {
    applyToSelectedNoThrowLocal(
        rows, context, result, setNullInResultAtError, [&](int row) {
          applyCastPrimitives<ToKind, FromKind, util::LegacyCastPolicy>(
              row,
              context,
              inputSimpleVector,
              setNullInResultAtError,
              resultFlatVector);
        });
  } else {
    applyToSelectedNoThrowLocal(
        rows, context, result, setNullInResultAtError, [&](int row) {
          applyCastPrimitives<ToKind, FromKind, util::PrestoCastPolicy>(
              row,
              context,
              inputSimpleVector,
              setNullInResultAtError,
              resultFlatVector);
        });
  }

  return result;
}

template <TypeKind ToKind>
VectorPtr PrestoCastKernel::applyCastPrimitivesDispatch(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& fromType,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  // This already excludes complex types, hugeint and unknown from type kinds.
  return VELOX_DYNAMIC_SCALAR_TEMPLATE_TYPE_DISPATCH(
      applyCastPrimitivesPolicyDispatch,
      ToKind,
      fromType->kind() /*dispatched*/,
      rows,
      input,
      context,
      toType,
      setNullInResultAtError);
}
} // namespace facebook::velox::exec
