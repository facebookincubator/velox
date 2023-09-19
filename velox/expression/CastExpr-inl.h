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

#include <charconv>

#include "velox/common/base/Exceptions.h"
#include "velox/core/CoreTypeSystem.h"
#include "velox/expression/StringWriter.h"
#include "velox/external/date/tz.h"
#include "velox/type/Type.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox::exec {
namespace {

inline std::string makeErrorMessage(
    const BaseVector& input,
    vector_size_t row,
    const TypePtr& toType,
    const std::string& details = "") {
  return fmt::format(
      "Cannot cast {} '{}' to {}. {}",
      input.type()->toString(),
      input.toString(row),
      toType->toString(),
      details);
}

inline std::exception_ptr makeBadCastException(
    const TypePtr& resultType,
    const BaseVector& input,
    vector_size_t row,
    const std::string& errorDetails) {
  return std::make_exception_ptr(VeloxUserError(
      std::current_exception(),
      makeErrorMessage(input, row, resultType, errorDetails),
      false));
}

/// Represent the varchar fragment.
///
/// For example:
/// | value | wholeDigits | fractionalDigits | exponent | sign
/// | 9999999999.99 | 9999999999 | 99 | nullopt | 1
/// | 15 | 15 |  | nullopt | 1
/// | 1.5 | 1 | 5 | nullopt | 1
/// | -1.5 | 1 | 5 | nullopt | -1
/// | 31.523e-2 | 31 | 523 | -2 | 1
struct DecimalComponents {
  std::string_view wholeDigits;
  std::string_view fractionalDigits;
  std::optional<int32_t> exponent = std::nullopt;
  int8_t sign = 1;
};

// Copied from format.h of fmt.
inline int countDigits(uint128_t n) {
  int count = 1;
  for (;;) {
    if (n < 10) {
      return count;
    }
    if (n < 100) {
      return count + 1;
    }
    if (n < 1000) {
      return count + 2;
    }
    if (n < 10000) {
      return count + 3;
    }
    n /= 10000u;
    count += 4;
  }
}

/// @brief Convert the unscaled value of a decimal to varchar and write to raw
/// string buffer from start position.
/// @tparam T The type of input value.
/// @param unscaledValue The input unscaled value.
/// @param scale The scale of decimal.
/// @param maxVarcharSize The estimated max size of a varchar.
/// @param startPosition The start position to write from.
/// @return A string view.
template <typename T>
StringView convertToStringView(
    T unscaledValue,
    int32_t scale,
    int32_t maxVarcharSize,
    char* const startPosition) {
  char* writePosition = startPosition;
  if (unscaledValue == 0) {
    *writePosition++ = '0';
  } else {
    if (unscaledValue < 0) {
      *writePosition++ = '-';
      unscaledValue = -unscaledValue;
    }
    auto [position, errorCode] = std::to_chars(
        writePosition,
        writePosition + maxVarcharSize,
        unscaledValue / DecimalUtil::kPowersOfTen[scale]);
    VELOX_DCHECK_EQ(
        errorCode,
        std::errc(),
        "Failed to cast decimal to varchar: {}",
        std::make_error_code(errorCode).message());
    writePosition = position;

    if (scale > 0) {
      *writePosition++ = '.';
      uint128_t fraction = unscaledValue % DecimalUtil::kPowersOfTen[scale];
      // Append leading zeros.
      int numLeadingZeros = std::max(scale - countDigits(fraction), 0);
      std::memset(writePosition, '0', numLeadingZeros);
      writePosition += numLeadingZeros;
      // Append remaining fraction digits.
      auto [position, errorCode] = std::to_chars(
          writePosition, writePosition + maxVarcharSize, fraction);
      VELOX_DCHECK_EQ(
          errorCode,
          std::errc(),
          "Failed to cast decimal to varchar: {}",
          std::make_error_code(errorCode).message());
      writePosition = position;
    }
  }
  return StringView(startPosition, writePosition - startPosition);
}

size_t parseDigitsRun(
    const char* s,
    size_t start,
    size_t size,
    std::string_view& out) {
  size_t pos = start;
  for (; pos < size; ++pos) {
    if (!std::isdigit(s[pos])) {
      break;
    }
  }
  out = std::string_view(s + start, pos - start);
  return pos;
}

std::optional<DecimalComponents> parseDecimalComponents(
    const char* s,
    size_t size) {
  if (size == 0) {
    return std::nullopt;
  }
  DecimalComponents out;
  size_t pos = 0;
  // Sign of the number.
  if (s[pos] == '-') {
    out.sign = -1;
    ++pos;
  } else if (s[pos] == '+') {
    out.sign = 1;
    ++pos;
  }
  // First run of digits.
  pos = parseDigitsRun(s, pos, size, out.wholeDigits);
  if (pos == size) {
    return out.wholeDigits.empty() ? std::nullopt
                                   : std::optional<DecimalComponents>(out);
  }
  // Optional dot (if given in fractional form).
  if (s[pos] == '.') {
    // Second run of digits.
    ++pos;
    pos = parseDigitsRun(s, pos, size, out.fractionalDigits);
  }
  if (out.wholeDigits.empty() && out.fractionalDigits.empty()) {
    // Need at least some digits (whole or fractional).
    return std::nullopt;
  }
  if (pos == size) {
    return out;
  }
  // Optional exponent.
  if (s[pos] == 'e' || s[pos] == 'E') {
    ++pos;
    if (pos != size && s[pos] == '+') {
      ++pos;
    }
    folly::StringPiece p = {s + pos, size - pos};
    auto tryExp =
        folly::tryTo<int32_t>(folly::StringPiece(s + pos, size - pos));
    if (tryExp.hasError()) {
      return std::nullopt;
    }
    out.exponent = tryExp.value();
    return out;
  }
  return pos == size ? std::optional<DecimalComponents>(out) : std::nullopt;
}

/// Multiple out by the appropriate power of 10 necessary to add source parsed
/// as int128_t and then adds the parsed value of source.
bool shiftAndAdd(std::string_view input, int128_t& out) {
  auto length = input.size();
  if (length == 0) {
    return true;
  }

  bool overflow =
      __builtin_mul_overflow(out, DecimalUtil::kPowersOfTen[length], &out);
  if (overflow) {
    return false;
  }
  auto tryValue =
      folly::tryTo<int128_t>(folly::StringPiece(input.data(), length));
  if (tryValue.hasError()) {
    return false;
  }

  overflow = __builtin_add_overflow(out, tryValue.value(), &out);
  VELOX_DCHECK(!overflow)
  return true;
}

/// Derives from Arrow function DecimalFromString.
/// Arrow implementation:
/// https://github.com/apache/arrow/blob/main/cpp/src/arrow/util/decimal.cc#L637
///
/// Firstly, it will parse the varchar to DecimalComponents which contains the
/// message that can represent a value. Secondly, process the exponent to get
/// the value parsedScale. Thirdly, compute the rescaled value.
/// The caller should test if `error` is empty
template <typename T>
std::optional<T> rescaleVarchar(
    const StringView s,
    int toPrecision,
    int toScale,
    std::string& error) {
  auto decimalComponentsOpt = parseDecimalComponents(s.data(), s.size());
  if (!decimalComponentsOpt.has_value()) {
    error = "Value is not a number.";
    return std::nullopt;
  }
  auto decimalComponents = decimalComponentsOpt.value();

  // Count number of significant digits (without leading zeros).
  size_t firstNonZero = decimalComponents.wholeDigits.find_first_not_of('0');
  size_t significantDigits = decimalComponents.fractionalDigits.size();
  if (firstNonZero != std::string::npos) {
    significantDigits += decimalComponents.wholeDigits.size() - firstNonZero;
  }
  int32_t parsedPrecision = static_cast<int32_t>(significantDigits);

  int32_t parsedScale = 0;
  bool addOne = false;
  int32_t fractionalDigitsSize = decimalComponents.fractionalDigits.size();
  if (decimalComponents.exponent.has_value()) {
    auto adjustedExponent = decimalComponents.exponent.value();
    parsedScale = -adjustedExponent + fractionalDigitsSize;
    // Truncate the fractionalDigits.
    if (parsedScale > toScale) {
      // adjustedExponent is negative, fractionalDigits only consider the last
      // digit to round up.
      if (-adjustedExponent >= toScale) {
        if (fractionalDigitsSize > 0 &&
            decimalComponents.fractionalDigits[0] >= '5') {
          addOne = true;
        }
        decimalComponents.fractionalDigits = "";
        parsedScale -= fractionalDigitsSize;
      } else {
        auto reduceDigits = adjustedExponent + toScale;
        if (fractionalDigitsSize > reduceDigits &&
            decimalComponents.fractionalDigits[reduceDigits] >= '5') {
          addOne = true;
        }
        decimalComponents.fractionalDigits = std::string_view(
            decimalComponents.fractionalDigits.data(),
            std::min(reduceDigits, fractionalDigitsSize));
        parsedScale -=
            fractionalDigitsSize - decimalComponents.fractionalDigits.size();
      }
    }
  } else {
    if (fractionalDigitsSize > toScale) {
      if (decimalComponents.fractionalDigits[toScale] >= '5') {
        addOne = true;
      }
      parsedScale = toScale;
      decimalComponents.fractionalDigits =
          std::string_view(decimalComponents.fractionalDigits.data(), toScale);
    } else {
      parsedScale = fractionalDigitsSize;
    }
  }

  int128_t out = 0;
  if (!shiftAndAdd(decimalComponents.wholeDigits, out)) {
    error = "Value too large.";
    return std::nullopt;
  }

  if (!shiftAndAdd(decimalComponents.fractionalDigits, out)) {
    error = "Value too large.";
    return std::nullopt;
  }
  if (addOne) {
    bool overflow = __builtin_add_overflow(out, 1, &out);
    if (UNLIKELY(overflow)) {
      error = "Value too large.";
      return std::nullopt;
    }
  }
  out = out * decimalComponents.sign;

  if (parsedScale < 0) {
    /// Force the scale to zero, to avoid negative scales (due to
    /// compatibility issues with external systems such as databases).
    if (-parsedScale + toScale > LongDecimalType::kMaxScale) {
      error = "Value too large.";
      return std::nullopt;
    }

    bool overflow = __builtin_mul_overflow(
        out, DecimalUtil::kPowersOfTen[-parsedScale + toScale], &out);
    if (UNLIKELY(overflow)) {
      error = "Value too large.";
      return std::nullopt;
    }
    parsedPrecision -= parsedScale;
    parsedScale = toScale;
  }
  bool overflow = false;
  auto rescaledValue = DecimalUtil::rescaleWithRoundUp<int128_t, T>(
      out, parsedPrecision, parsedScale, toPrecision, toScale, overflow, false);
  if (overflow) {
    error = "Value too large.";
    return std::nullopt;
  }
  return rescaledValue;
}
} // namespace

template <bool adjustForTimeZone>
void CastExpr::castTimestampToDate(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    VectorPtr& result,
    const date::time_zone* timeZone) {
  auto* resultFlatVector = result->as<FlatVector<int32_t>>();
  static const int32_t kSecsPerDay{86'400};
  auto inputVector = input.as<SimpleVector<Timestamp>>();
  applyToSelectedNoThrowLocal(context, rows, result, [&](int row) {
    auto input = inputVector->valueAt(row);
    if constexpr (adjustForTimeZone) {
      input.toTimezone(*timeZone);
    }
    auto seconds = input.getSeconds();
    if (seconds >= 0 || seconds % kSecsPerDay == 0) {
      resultFlatVector->set(row, seconds / kSecsPerDay);
    } else {
      // For division with negatives, minus 1 to compensate the discarded
      // fractional part. e.g. -1/86'400 yields 0, yet it should be
      // considered as -1 day.
      resultFlatVector->set(row, seconds / kSecsPerDay - 1);
    }
  });
}

template <typename Func>
void CastExpr::applyToSelectedNoThrowLocal(
    EvalCtx& context,
    const SelectivityVector& rows,
    VectorPtr& result,
    Func&& func) {
  if (setNullInResultAtError()) {
    rows.template applyToSelected([&](auto row) INLINE_LAMBDA {
      try {
        func(row);
      } catch (...) {
        result->setNull(row, true);
      }
    });
  } else {
    rows.template applyToSelected([&](auto row) INLINE_LAMBDA {
      try {
        func(row);
      } catch (const VeloxException& e) {
        if (!e.isUserError()) {
          throw;
        }
        // Avoid double throwing.
        context.setVeloxExceptionError(row, std::current_exception());
      } catch (const std::exception& e) {
        context.setError(row, std::current_exception());
      }
    });
  }
}

/// The per-row level Kernel
/// @tparam ToKind The cast target type
/// @tparam FromKind The expression type
/// @param row The index of the current row
/// @param input The input vector (of type FromKind)
/// @param result The output vector (of type ToKind)
template <TypeKind ToKind, TypeKind FromKind, bool Truncate, bool LegacyCast>
void CastExpr::applyCastKernel(
    vector_size_t row,
    EvalCtx& context,
    const SimpleVector<typename TypeTraits<FromKind>::NativeType>* input,
    FlatVector<typename TypeTraits<ToKind>::NativeType>* result) {
  auto setError = [&](const std::string& details) {
    if (setNullInResultAtError()) {
      result->setNull(row, true);
    } else {
      context.setVeloxExceptionError(
          row, makeBadCastException(result->type(), *input, row, details));
    }
  };

  try {
    auto inputRowValue = input->valueAt(row);

    // Optimize empty input strings casting by avoiding throwing exceptions.
    if constexpr (
        FromKind == TypeKind::VARCHAR || FromKind == TypeKind::VARBINARY) {
      if constexpr (
          TypeTraits<ToKind>::isPrimitiveType &&
          TypeTraits<ToKind>::isFixedWidth) {
        if (inputRowValue.size() == 0) {
          setError("Empty string");
          return;
        }
      }
    }

    auto output = util::Converter<ToKind, void, Truncate, LegacyCast>::cast(
        inputRowValue);

    if constexpr (
        ToKind == TypeKind::VARCHAR || ToKind == TypeKind::VARBINARY) {
      // Write the result output to the output vector
      auto writer = exec::StringWriter<>(result, row);
      writer.copy_from(output);
      writer.finalize();
    } else {
      result->set(row, output);
    }

  } catch (const VeloxException& ue) {
    if (!ue.isUserError()) {
      throw;
    }
    setError(ue.message());
  } catch (const std::exception& e) {
    setError(e.what());
  }
}

template <typename TInput, typename TOutput>
void CastExpr::applyDecimalCastKernel(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& fromType,
    const TypePtr& toType,
    VectorPtr& castResult) {
  auto sourceVector = input.as<SimpleVector<TInput>>();
  auto castResultRawBuffer =
      castResult->asUnchecked<FlatVector<TOutput>>()->mutableRawValues();
  const auto& fromPrecisionScale = getDecimalPrecisionScale(*fromType);
  const auto& toPrecisionScale = getDecimalPrecisionScale(*toType);

  applyToSelectedNoThrowLocal(
      context, rows, castResult, [&](vector_size_t row) {
        bool overflow = false;
        auto rescaledValue = DecimalUtil::rescaleWithRoundUp<TInput, TOutput>(
            sourceVector->valueAt(row),
            fromPrecisionScale.first,
            fromPrecisionScale.second,
            toPrecisionScale.first,
            toPrecisionScale.second,
            overflow);
        if (rescaledValue.has_value()) {
          castResultRawBuffer[row] = rescaledValue.value();
        } else {
          castResult->setNull(row, true);
        }
      });
}

template <typename TInput, typename TOutput>
void CastExpr::applyIntToDecimalCastKernel(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    VectorPtr& castResult) {
  auto sourceVector = input.as<SimpleVector<TInput>>();
  auto castResultRawBuffer =
      castResult->asUnchecked<FlatVector<TOutput>>()->mutableRawValues();
  const auto& toPrecisionScale = getDecimalPrecisionScale(*toType);
  applyToSelectedNoThrowLocal(
      context, rows, castResult, [&](vector_size_t row) {
        auto rescaledValue = DecimalUtil::rescaleInt<TInput, TOutput>(
            sourceVector->valueAt(row),
            toPrecisionScale.first,
            toPrecisionScale.second);
        if (rescaledValue.has_value()) {
          castResultRawBuffer[row] = rescaledValue.value();
        } else {
          castResult->setNull(row, true);
        }
      });
}

template <typename T>
void CastExpr::applyVarcharToDecimalCastKernel(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    VectorPtr& result) {
  auto sourceVector = input.as<SimpleVector<StringView>>();
  auto rawBuffer = result->asUnchecked<FlatVector<T>>()->mutableRawValues();
  const auto toPrecisionScale = getDecimalPrecisionScale(*toType);
  auto setError = [&](vector_size_t row, const std::string& details) {
    if (setNullInResultAtError()) {
      result->setNull(row, true);
    } else {
      context.setVeloxExceptionError(
          row, makeBadCastException(toType, input, row, details));
    }
  };

  rows.applyToSelected([&](auto row) {
    std::string error;
    auto rescaledValue = rescaleVarchar<T>(
        sourceVector->valueAt(row),
        toPrecisionScale.first,
        toPrecisionScale.second,
        error);
    if (!error.empty()) {
      setError(row, error);
    } else if (rescaledValue.has_value()) {
      rawBuffer[row] = rescaledValue.value();
    } else {
      result->setNull(row, true);
    }
  });
}

template <typename FromNativeType, TypeKind ToKind>
VectorPtr CastExpr::applyDecimalToFloatCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& fromType,
    const TypePtr& toType) {
  using To = typename TypeTraits<ToKind>::NativeType;

  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);
  auto resultBuffer = result->asUnchecked<FlatVector<To>>()->mutableRawValues();
  const auto precisionScale = getDecimalPrecisionScale(*fromType);
  const auto simpleInput = input.as<SimpleVector<FromNativeType>>();
  const auto scaleFactor = DecimalUtil::kPowersOfTen[precisionScale.second];
  applyToSelectedNoThrowLocal(context, rows, result, [&](int row) {
    auto output = util::Converter<ToKind, void, false, false>::cast(
        simpleInput->valueAt(row));
    resultBuffer[row] = output / scaleFactor;
  });
  return result;
}

template <typename FromNativeType, TypeKind ToKind>
VectorPtr CastExpr::applyDecimalToIntegralCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& fromType,
    const TypePtr& toType) {
  using To = typename TypeTraits<ToKind>::NativeType;

  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);
  auto resultBuffer = result->asUnchecked<FlatVector<To>>()->mutableRawValues();
  const auto precisionScale = getDecimalPrecisionScale(*fromType);
  const auto simpleInput = input.as<SimpleVector<FromNativeType>>();
  const auto scaleFactor = DecimalUtil::kPowersOfTen[precisionScale.second];
  const auto castToIntByTruncate =
      context.execCtx()->queryCtx()->queryConfig().isCastToIntByTruncate();
  applyToSelectedNoThrowLocal(context, rows, result, [&](int row) {
    auto value = simpleInput->valueAt(row);
    auto integralPart = value / scaleFactor;
    if (!castToIntByTruncate) {
      auto fractionPart = value % scaleFactor;
      auto sign = value >= 0 ? 1 : -1;
      bool needsRoundUp =
          (scaleFactor != 1) && (sign * fractionPart >= (scaleFactor >> 1));
      integralPart += needsRoundUp ? sign : 0;
    }

    if (integralPart > std::numeric_limits<To>::max() ||
        integralPart < std::numeric_limits<To>::min()) {
      if (setNullInResultAtError()) {
        result->setNull(row, true);
      } else {
        context.setVeloxExceptionError(
            row,
            makeBadCastException(
                result->type(),
                input,
                row,
                makeErrorMessage(input, row, toType) + "Out of bounds."));
      }
      return;
    }

    resultBuffer[row] = static_cast<To>(integralPart);
  });
  return result;
}

template <typename FromNativeType>
VectorPtr CastExpr::applyDecimalToBooleanCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context) {
  VectorPtr result;
  context.ensureWritable(rows, BOOLEAN(), result);
  (*result).clearNulls(rows);
  auto resultBuffer =
      result->asUnchecked<FlatVector<bool>>()->mutableRawValues<uint64_t>();
  const auto simpleInput = input.as<SimpleVector<FromNativeType>>();
  applyToSelectedNoThrowLocal(context, rows, result, [&](int row) {
    auto value = simpleInput->valueAt(row);
    bits::setBit(resultBuffer, row, value != 0);
  });
  return result;
}

template <typename FromNativeType>
VectorPtr CastExpr::applyDecimalToVarcharCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& fromType) {
  VectorPtr result;
  context.ensureWritable(rows, VARCHAR(), result);
  (*result).clearNulls(rows);
  const auto simpleInput = input.as<SimpleVector<FromNativeType>>();
  int precision = getDecimalPrecisionScale(*fromType).first;
  int scale = getDecimalPrecisionScale(*fromType).second;
  // A varchar's size is estimated with unscaled value digits, dot, leading
  // zero, and possible minus sign.
  int32_t rowSize = precision + 1;
  if (scale > 0) {
    ++rowSize; // A dot.
  }
  if (precision == scale) {
    ++rowSize; // Leading zero.
  }

  auto flatResult = result->asFlatVector<StringView>();
  if (StringView::isInline(rowSize)) {
    char inlined[StringView::kInlineSize];
    applyToSelectedNoThrowLocal(context, rows, result, [&](vector_size_t row) {
      if (simpleInput->isNullAt(row)) {
        result->setNull(row, true);
      } else {
        flatResult->setNoCopy(
            row,
            convertToStringView<FromNativeType>(
                simpleInput->valueAt(row), scale, rowSize, inlined));
      }
    });
    return result;
  }

  Buffer* buffer =
      flatResult->getBufferWithSpace(rows.countSelected() * rowSize);
  char* rawBuffer = buffer->asMutable<char>() + buffer->size();

  applyToSelectedNoThrowLocal(context, rows, result, [&](vector_size_t row) {
    if (simpleInput->isNullAt(row)) {
      result->setNull(row, true);
    } else {
      auto stringView = convertToStringView<FromNativeType>(
          simpleInput->valueAt(row), scale, rowSize, rawBuffer);
      flatResult->setNoCopy(row, stringView);
      if (!stringView.isInline()) {
        // If string view is inline, correponding bytes on the raw string buffer
        // are not needed.
        rawBuffer += stringView.size();
      }
    }
  });
  // Update the exact buffer size.
  buffer->setSize(rawBuffer - buffer->asMutable<char>());
  return result;
}

template <typename FromNativeType>
VectorPtr CastExpr::applyDecimalToPrimitiveCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& fromType,
    const TypePtr& toType) {
  switch (toType->kind()) {
    case TypeKind::BOOLEAN:
      return applyDecimalToBooleanCast<FromNativeType>(rows, input, context);
    case TypeKind::TINYINT:
      return applyDecimalToIntegralCast<FromNativeType, TypeKind::TINYINT>(
          rows, input, context, fromType, toType);
    case TypeKind::SMALLINT:
      return applyDecimalToIntegralCast<FromNativeType, TypeKind::SMALLINT>(
          rows, input, context, fromType, toType);
    case TypeKind::INTEGER:
      return applyDecimalToIntegralCast<FromNativeType, TypeKind::INTEGER>(
          rows, input, context, fromType, toType);
    case TypeKind::BIGINT:
      return applyDecimalToIntegralCast<FromNativeType, TypeKind::BIGINT>(
          rows, input, context, fromType, toType);
    case TypeKind::REAL:
      return applyDecimalToFloatCast<FromNativeType, TypeKind::REAL>(
          rows, input, context, fromType, toType);
    case TypeKind::DOUBLE:
      return applyDecimalToFloatCast<FromNativeType, TypeKind::DOUBLE>(
          rows, input, context, fromType, toType);
    default:
      VELOX_UNSUPPORTED(
          "Cast from {} to {} is not supported",
          fromType->toString(),
          toType->toString());
  }
}

template <TypeKind ToKind, TypeKind FromKind>
void CastExpr::applyCastPrimitives(
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const BaseVector& input,
    VectorPtr& result) {
  using To = typename TypeTraits<ToKind>::NativeType;
  using From = typename TypeTraits<FromKind>::NativeType;
  auto* resultFlatVector = result->as<FlatVector<To>>();
  auto* inputSimpleVector = input.as<SimpleVector<From>>();

  const auto& queryConfig = context.execCtx()->queryCtx()->queryConfig();
  auto& resultType = resultFlatVector->type();

  if (!queryConfig.isCastToIntByTruncate()) {
    if (!queryConfig.isLegacyCast()) {
      applyToSelectedNoThrowLocal(context, rows, result, [&](int row) {
        applyCastKernel<ToKind, FromKind, false /*truncate*/, false /*legacy*/>(
            row, context, inputSimpleVector, resultFlatVector);
      });
    } else {
      applyToSelectedNoThrowLocal(context, rows, result, [&](int row) {
        applyCastKernel<ToKind, FromKind, false /*truncate*/, true /*legacy*/>(
            row, context, inputSimpleVector, resultFlatVector);
      });
    }
  } else {
    if (!queryConfig.isLegacyCast()) {
      applyToSelectedNoThrowLocal(context, rows, result, [&](int row) {
        applyCastKernel<ToKind, FromKind, true /*truncate*/, false /*legacy*/>(
            row, context, inputSimpleVector, resultFlatVector);
      });
    } else {
      applyToSelectedNoThrowLocal(context, rows, result, [&](int row) {
        applyCastKernel<ToKind, FromKind, true /*truncate*/, true /*legacy*/>(
            row, context, inputSimpleVector, resultFlatVector);
      });
    }
  }

  // If we're converting to a TIMESTAMP, check if we need to adjust the
  // current GMT timezone to the user provided session timezone.
  if constexpr (ToKind == TypeKind::TIMESTAMP) {
    // If user explicitly asked us to adjust the timezone.
    if (queryConfig.adjustTimestampToTimezone()) {
      auto sessionTzName = queryConfig.sessionTimezone();
      if (!sessionTzName.empty()) {
        // When context.throwOnError is false, some rows will be marked as
        // 'failed'. These rows should not be processed further. 'remainingRows'
        // will contain a subset of 'rows' that have passed all the checks (e.g.
        // keys are not nulls and number of keys and values is the same).
        exec::LocalSelectivityVector remainingRows(context, rows);
        context.deselectErrors(*remainingRows);

        // locate_zone throws runtime_error if the timezone couldn't be found
        // (so we're safe to dereference the pointer).
        auto* timeZone = date::locate_zone(sessionTzName);
        auto rawTimestamps = resultFlatVector->mutableRawValues();

        applyToSelectedNoThrowLocal(
            context, *remainingRows, result, [&](int row) {
              rawTimestamps[row].toGMT(*timeZone);
            });
      }
    }
  }
}

template <TypeKind ToKind>
void CastExpr::applyCastPrimitivesDispatch(
    const TypePtr& fromType,
    const TypePtr& toType,
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const BaseVector& input,
    VectorPtr& result) {
  context.ensureWritable(rows, toType, result);

  // This already excludes complex types, hugeint and unknown from type kinds.
  VELOX_DYNAMIC_SCALAR_TEMPLATE_TYPE_DISPATCH(
      applyCastPrimitives,
      ToKind,
      fromType->kind() /*dispatched*/,
      rows,
      context,
      input,
      result);
}

} // namespace facebook::velox::exec
