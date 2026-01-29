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

#include <folly/Likely.h>

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
            if (setNullInResultAtError) {
              result->setNull(row, true);
            } else if (context.captureErrorDetails()) {
              context.setStatus(
                  row,
                  Status::UserError(
                      makeErrorMessage(input, row, toType, "Out of bounds.")));
            } else {
              context.setStatus(row, Status::UserError());
            }
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
          if (setNullInResultAtError) {
            result->setNull(row, true);
          } else {
            context.setStatus(
                row,
                Status::UserError(
                    makeErrorMessage(input, row, toType, status.message())));
          }
        }
      });
}

template <TypeKind FromTypeKind>
void SparkCastKernel::applyIntegerToBooleanCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError,
    VectorPtr& result) const {
  auto sourceVector =
      input.as<SimpleVector<typename TypeTraits<FromTypeKind>::NativeType>>();
  auto* resultFlatVector = result->as<FlatVector<bool>>();

  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
        resultFlatVector->set(row, sourceVector->valueAt(row));
      });
}

template <TypeKind FromTypeKind>
void SparkCastKernel::applyFloatingPointToBooleanCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError,
    VectorPtr& result) const {
  auto sourceVector =
      input.as<SimpleVector<typename TypeTraits<FromTypeKind>::NativeType>>();
  auto* resultFlatVector = result->as<FlatVector<bool>>();

  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
        auto value = sourceVector->valueAt(row);
        resultFlatVector->set(row, !std::isnan(value) && value != 0);
      });
}

template <TypeKind ToTypeKind>
void SparkCastKernel::applyTimestampToIntegerCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError,
    VectorPtr& result) const {
  auto sourceVector = input.as<SimpleVector<Timestamp>>();
  auto* resultFlatVector =
      result->as<FlatVector<typename TypeTraits<ToTypeKind>::NativeType>>();

  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
        auto micros = sourceVector->valueAt(row).toMicros();
        if (micros < 0) {
          resultFlatVector->set(
              row,
              static_cast<int64_t>(std::floor(
                  static_cast<double>(micros) /
                  Timestamp::kMicrosecondsInSecond)));
        } else {
          resultFlatVector->set(row, micros / Timestamp::kMicrosecondsInSecond);
        }
      });
}

template <TypeKind ToTypeKind>
void SparkCastKernel::applyStringToIntegerCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError,
    VectorPtr& result) const {
  using ToNativeType = typename TypeTraits<ToTypeKind>::NativeType;

  auto sourceVector = input.as<SimpleVector<StringView>>();
  auto* resultFlatVector = result->as<FlatVector<ToNativeType>>();

  if (allowOverflow_) {
    applyToSelectedNoThrowLocal(
        rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
          auto inputStr = sourceVector->valueAt(row);
          inputStr = removeWhiteSpaces(inputStr);
          const auto len = inputStr.size();
          const auto data = inputStr.data();

          ToNativeType result = 0;
          int index = 0;
          if (len == 0) {
            if (setNullInResultAtError) {
              resultFlatVector->setNull(row, true);
            } else if (context.captureErrorDetails()) {
              context.setStatus(
                  row,
                  Status::UserError(
                      "{} Cannot cast an empty string to an integral value.",
                      makeErrorMessage(input, row, resultFlatVector->type())));
            } else {
              context.setStatus(row, Status::UserError());
            }

            return;
          }

          // Setting negative flag
          bool negative = false;
          // Setting decimalPoint flag
          bool decimalPoint = false;
          if (data[0] == '-' || data[0] == '+') {
            if (len == 1) {
              if (setNullInResultAtError) {
                resultFlatVector->setNull(row, true);
              } else if (context.captureErrorDetails()) {
                context.setStatus(
                    row,
                    Status::UserError(
                        "{} Cannot cast an '{}' string to an integral value.",
                        makeErrorMessage(input, row, resultFlatVector->type()),
                        data[0]));
              } else {
                context.setStatus(row, Status::UserError());
              }

              return;
            }
            negative = data[0] == '-';
            index = 1;
          }
          if (negative) {
            for (; index < len; index++) {
              // Truncate the decimal
              if (!decimalPoint && data[index] == '.') {
                decimalPoint = true;
                if (++index == len) {
                  break;
                }
              }
              if (!std::isdigit(data[index])) {
                if (setNullInResultAtError) {
                  resultFlatVector->setNull(row, true);
                } else if (context.captureErrorDetails()) {
                  context.setStatus(
                      row,
                      Status::UserError(
                          "{} Encountered a non-digit character",
                          makeErrorMessage(
                              input, row, resultFlatVector->type())));
                } else {
                  context.setStatus(row, Status::UserError());
                }

                return;
              }
              if (!decimalPoint) {
                result = checkedMultiply<ToNativeType>(
                    result, 10, CppToType<ToNativeType>::name);
                result = checkedMinus<ToNativeType>(
                    result, data[index] - '0', CppToType<ToNativeType>::name);
              }
            }
          } else {
            for (; index < len; index++) {
              // Truncate the decimal
              if (!decimalPoint && data[index] == '.') {
                decimalPoint = true;
                if (++index == len) {
                  break;
                }
              }
              if (!std::isdigit(data[index])) {
                if (setNullInResultAtError) {
                  resultFlatVector->setNull(row, true);
                } else if (context.captureErrorDetails()) {
                  context.setStatus(
                      row,
                      Status::UserError(
                          "{} Encountered a non-digit character",
                          makeErrorMessage(
                              input, row, resultFlatVector->type())));
                } else {
                  context.setStatus(row, Status::UserError());
                }

                return;
              }
              if (!decimalPoint) {
                result = checkedMultiply<ToNativeType>(
                    result, 10, CppToType<ToNativeType>::name);
                result = checkedPlus<ToNativeType>(
                    result, data[index] - '0', CppToType<ToNativeType>::name);
              }
            }
          }

          resultFlatVector->set(row, result);
        });
  } else {
    applyToSelectedNoThrowLocal(
        rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
          auto inputStr = sourceVector->valueAt(row);
          inputStr = removeWhiteSpaces(inputStr);
          auto trimmed = util::trimWhiteSpace(inputStr.data(), inputStr.size());
          const auto result = folly::tryTo<ToNativeType>(trimmed);
          if (result.hasError()) {
            if (setNullInResultAtError) {
              resultFlatVector->setNull(row, true);
            } else if (context.captureErrorDetails()) {
              context.setStatus(
                  row,
                  Status::UserError(
                      "{} {}",
                      makeErrorMessage(input, row, resultFlatVector->type()),
                      folly::makeConversionError(result.error(), "").what()));
            } else {
              context.setStatus(row, Status::UserError());
            }
          } else {
            resultFlatVector->set(row, result.value());
          }
        });
  }
}

template <TypeKind FromTypeKind, TypeKind ToTypeKind>
void SparkCastKernel::applyIntegerToIntegerCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError,
    VectorPtr& result) const {
  using FromNativeType = typename TypeTraits<FromTypeKind>::NativeType;
  using ToNativeType = typename TypeTraits<ToTypeKind>::NativeType;

  auto sourceVector = input.as<SimpleVector<FromNativeType>>();
  auto* resultFlatVector = result->as<FlatVector<ToNativeType>>();

  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
        resultFlatVector->set(row, sourceVector->valueAt(row));
      });
}

template <typename T>
struct LimitType {
  static constexpr bool kByteOrSmallInt =
      std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>;

  static int64_t minLimit() {
    if (kByteOrSmallInt) {
      return std::numeric_limits<int32_t>::min();
    }
    return std::numeric_limits<T>::min();
  }

  static int64_t maxLimit() {
    if (kByteOrSmallInt) {
      return std::numeric_limits<int32_t>::max();
    }
    return std::numeric_limits<T>::max();
  }

  static T min() {
    if (kByteOrSmallInt) {
      return 0;
    }
    return std::numeric_limits<T>::min();
  }

  static T max() {
    if (kByteOrSmallInt) {
      return -1;
    }
    return std::numeric_limits<T>::max();
  }

  template <typename FP>
  static T tryCast(const FP& v) {
    if (kByteOrSmallInt) {
      return T(int32_t(v));
    }
    return T(v);
  }
};

template <TypeKind FromTypeKind, TypeKind ToTypeKind>
void SparkCastKernel::applyFloatingPointToIntegerCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError,
    VectorPtr& result) const {
  using FromNativeType = typename TypeTraits<FromTypeKind>::NativeType;
  using ToNativeType = typename TypeTraits<ToTypeKind>::NativeType;

  auto sourceVector = input.as<SimpleVector<FromNativeType>>();
  auto* resultFlatVector = result->as<FlatVector<ToNativeType>>();

  if (allowOverflow_) {
    applyToSelectedNoThrowLocal(
        rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
          auto value = sourceVector->valueAt(row);

          if (std::isnan(value)) {
            resultFlatVector->set(row, 0);
            return;
          }

          if constexpr (std::is_same_v<ToNativeType, int128_t>) {
            resultFlatVector->set(row, std::numeric_limits<int128_t>::max());
            return;
          } else if (value > LimitType<ToNativeType>::maxLimit()) {
            resultFlatVector->set(row, LimitType<ToNativeType>::max());
            return;
          } else if (value < LimitType<ToNativeType>::minLimit()) {
            resultFlatVector->set(row, LimitType<ToNativeType>::min());
            return;
          }

          resultFlatVector->set(row, LimitType<ToNativeType>::tryCast(value));
        });
  } else {
    applyToSelectedNoThrowLocal(
        rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
          auto value = sourceVector->valueAt(row);

          if (std::isnan(value)) {
            if (setNullInResultAtError) {
              resultFlatVector->setNull(row, true);
            } else if (context.captureErrorDetails()) {
              context.setStatus(
                  row,
                  Status::UserError(
                      "{} Cannot cast a NaN to an integral value.",
                      makeErrorMessage(input, row, resultFlatVector->type())));
            } else {
              context.setStatus(row, Status::UserError());
            }

            return;
          }

          const auto result = folly::tryTo<ToNativeType>(std::trunc(value));
          if (result.hasError()) {
            if (setNullInResultAtError) {
              resultFlatVector->setNull(row, true);
            } else if (context.captureErrorDetails()) {
              context.setStatus(
                  row,
                  Status::UserError(
                      "{} {}",
                      makeErrorMessage(input, row, resultFlatVector->type()),
                      folly::makeConversionError(result.error(), "").what()));
            } else {
              context.setStatus(row, Status::UserError());
            }
          } else {
            resultFlatVector->set(row, result.value());
          }
        });
  }
}

template <TypeKind ToTypeKind>
void SparkCastKernel::applyStringToFloatingPointCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError,
    VectorPtr& result) const {
  using ToNativeType = typename TypeTraits<ToTypeKind>::NativeType;

  auto sourceVector = input.as<SimpleVector<StringView>>();
  auto* resultFlatVector = result->as<FlatVector<ToNativeType>>();

  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
        auto inputStr = sourceVector->valueAt(row);
        inputStr = removeWhiteSpaces(inputStr);

        auto value = util::Converter<ToTypeKind>::tryCast(inputStr);
        if (value.hasError()) {
          if (setNullInResultAtError) {
            resultFlatVector->setNull(row, true);
          } else if (context.captureErrorDetails()) {
            context.setStatus(
                row,
                Status::UserError(
                    "{}",
                    makeErrorMessage(
                        input,
                        row,
                        resultFlatVector->type(),
                        value.error().message())));
          } else {
            context.setStatus(row, Status::UserError());
          }
        } else {
          resultFlatVector->set(row, value.value());
        }
      });
}

template <TypeKind FromTypeKind>
void SparkCastKernel::applyNumberToTimestampCast(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError,
    VectorPtr& result) const {
  using FromNativeType = typename TypeTraits<FromTypeKind>::NativeType;

  auto sourceVector = input.as<SimpleVector<FromNativeType>>();
  auto* resultFlatVector = result->as<FlatVector<Timestamp>>();

  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
        const auto value = sourceVector->valueAt(row);

        if constexpr (
            FromTypeKind == TypeKind::REAL ||
            FromTypeKind == TypeKind::DOUBLE) {
          if (FOLLY_UNLIKELY(std::isnan(value) || std::isinf(value))) {
            resultFlatVector->setNull(row, true);

            return;
          }

          resultFlatVector->set(
              row, castNumberToTimestamp(static_cast<double>(value)));
          return;
        } else {
          resultFlatVector->set(row, castNumberToTimestamp(value));
        }
      });
}

template <typename T>
Timestamp SparkCastKernel::castNumberToTimestamp(T seconds) const {
  // Spark internally use microsecond precision for timestamp.
  // To avoid overflow, we need to check the range of seconds.
  static constexpr int64_t maxSeconds =
      std::numeric_limits<int64_t>::max() / Timestamp::kMicrosecondsInSecond;
  if (seconds > maxSeconds) {
    return Timestamp::fromMicrosNoError(std::numeric_limits<int64_t>::max());
  }
  if (seconds < -maxSeconds) {
    return Timestamp::fromMicrosNoError(std::numeric_limits<int64_t>::min());
  }

  if constexpr (std::is_floating_point_v<T>) {
    return Timestamp::fromMicrosNoError(
        static_cast<int64_t>(seconds * Timestamp::kMicrosecondsInSecond));
  }

  return Timestamp(seconds, 0);
}
} // namespace facebook::velox::functions::sparksql
