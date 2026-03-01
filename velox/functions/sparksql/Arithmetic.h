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

#include <bitset>
#include <cmath>
#include <limits>
#include <system_error>
#include <type_traits>

#include "velox/common/base/Doubles.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Status.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/ToHex.h"
#include "velox/functions/sparksql/DecimalUtil.h"

namespace facebook::velox::functions::sparksql {

// The abs implementation is used for primitive types except for decimal type.
template <typename TExec>
struct AbsFunction {
  template <typename T>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const T* /*a*/) {
    ansiEnabled_ = config.sparkAnsiEnabled();
  }

  template <typename T>
  FOLLY_ALWAYS_INLINE Status call(T& result, const T& a) {
    if constexpr (std::is_integral_v<T>) {
      if (FOLLY_UNLIKELY(a == std::numeric_limits<T>::min())) {
        if (ansiEnabled_) {
          // In ANSI mode, returns an overflow error.
          if (threadSkipErrorDetails()) {
            return Status::UserError();
          }
          return Status::UserError("Arithmetic overflow: abs({})", a);
        }
        // In ANSI off mode, returns the same negative minimum value.
        result = a;
        return Status::OK();
      }
    }
    result = std::abs(a);
    return Status::OK();
  }

 private:
  bool ansiEnabled_ = false;
};

template <typename T>
struct RemainderFunction {
  template <
      typename TInput,
      typename std::enable_if_t<!std::is_floating_point_v<TInput>, int> = 0>
  FOLLY_ALWAYS_INLINE bool
  call(TInput& result, const TInput a, const TInput n) {
    if (UNLIKELY(n == 0)) {
      return false;
    }
    // std::numeric_limits<int64_t>::min() % -1 could crash the program since
    // abs(std::numeric_limits<int64_t>::min()) can not be represented in
    // int64_t.
    if (UNLIKELY(n == 1 || n == -1)) {
      result = 0;
    } else {
      result = a % n;
    }
    return true;
  }

  // Specialization for floating point types.
  template <
      typename TInput,
      typename std::enable_if_t<std::is_floating_point_v<TInput>, int> = 0>
  FOLLY_ALWAYS_INLINE bool
  call(TInput& result, const TInput a, const TInput n) {
    if (UNLIKELY(n == 0)) {
      return false;
    }
    // If either the dividend or the divisor is NaN, or if the dividend is
    // infinity, the result is set to NaN.
    if (UNLIKELY(std::isnan(a) || std::isnan(n) || std::isinf(a))) {
      result = std::numeric_limits<TInput>::quiet_NaN();
    }
    // If the divisor is infinity, the result is equal to the dividend.
    else if (UNLIKELY(std::isinf(n))) {
      result = a;
    } else {
      result = std::fmod(a, n);
    }
    return true;
  }
};

template <typename T>
struct PModIntFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool call(TInput& result, const TInput a, const TInput n)
#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
      __attribute__((__no_sanitize__("signed-integer-overflow")))
#endif
#endif
  {
    TInput r;
    bool notNull = RemainderFunction<T>().call(r, a, n);
    if (!notNull) {
      return false;
    }

    result = (r > 0) ? r : (r + n) % n;
    return true;
  }
};

template <typename T>
struct PModFloatFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool
  call(TInput& result, const TInput a, const TInput n) {
    if (UNLIKELY(n == (TInput)0)) {
      return false;
    }
    TInput r = fmod(a, n);
    result = (r > 0) ? r : fmod(r + n, n);
    return true;
  }
};

template <typename T>
struct UnaryMinusFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool call(TInput& result, const TInput a) {
    if constexpr (std::is_integral_v<TInput>) {
      // Avoid undefined integer overflow.
      result = a == std::numeric_limits<TInput>::min() ? a : -a;
    } else {
      result = -a;
    }
    return true;
  }
};

template <typename T>
struct DivideFunction {
  FOLLY_ALWAYS_INLINE bool
  call(double& result, const double num, const double denom) {
    if (UNLIKELY(denom == 0)) {
      return false;
    }
    result = num / denom;
    return true;
  }
};

/*
  In Spark both ceil and floor must return Long type
  sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/mathExpressions.scala
*/
template <typename T>
int64_t safeDoubleToInt64(const T& /*value*/) {
  throw std::runtime_error("Invalid input for floor/ceil");
}

template <>
inline int64_t safeDoubleToInt64(const double& arg) {
  if (std::isnan(arg)) {
    return 0;
  }
  static const int64_t kMax = std::numeric_limits<int64_t>::max();
  static const int64_t kMin = std::numeric_limits<int64_t>::min();

  if (arg >= kMinDoubleAboveInt64Max) {
    return kMax;
  }
  if (arg < kMin) {
    return kMin;
  }
  return arg;
}

template <>
inline int64_t safeDoubleToInt64(const int64_t& arg) {
  return arg;
}

template <typename T>
struct CeilFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const TInput value) {
    if constexpr (std::is_integral_v<TInput>) {
      result = value;
    } else {
      result = safeDoubleToInt64(std::ceil(value));
    }
    return true;
  }
};

template <typename T>
struct FloorFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const TInput value) {
    if constexpr (std::is_integral_v<TInput>) {
      result = value;
    } else {
      result = safeDoubleToInt64(std::floor(value));
    }
    return true;
  }
};

template <typename T>
struct AcoshFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, TInput a) {
    result = std::acosh(a);
  }
};

template <typename T>
struct AsinhFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, TInput a) {
    result = std::asinh(a);
  }
};

template <typename T>
struct AtanhFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, TInput a) {
    result = std::atanh(a);
  }
};

template <typename T>
struct SecFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, TInput a) {
    result = 1 / std::cos(a);
  }
};

template <typename T>
struct CscFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, TInput a) {
    result = 1 / std::sin(a);
  }
};

template <typename T>
struct CoshFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, TInput a) {
    result = std::cosh(a);
  }
};

template <typename T>
struct ToBinaryStringFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE
  void call(out_type<Varchar>& result, const arg_type<int64_t>& input) {
    auto str = std::bitset<64>(input).to_string();
    str.erase(0, std::min(str.find_first_not_of('0'), str.size() - 1));
    result = str;
  }
};

template <typename T>
struct SinhFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, TInput a) {
    result = std::sinh(a);
  }
};

template <typename T>
struct HypotFunction {
  FOLLY_ALWAYS_INLINE void call(double& result, double a, double b) {
    result = std::hypot(a, b);
  }
};

template <typename T>
struct Log2Function {
  FOLLY_ALWAYS_INLINE bool call(double& result, double a) {
    if (a <= 0.0) {
      return false;
    }
    result = std::log2(a);
    return true;
  }
};

template <typename T>
struct Log1pFunction {
  FOLLY_ALWAYS_INLINE bool call(double& result, double a) {
    if (a <= -1) {
      return false;
    }
    result = std::log1p(a);
    return true;
  }
};

template <typename T>
struct LogarithmFunction {
  FOLLY_ALWAYS_INLINE bool call(double& result, double a, double b) {
    if (a <= 0 || b <= 0) {
      return false;
    }
    result = std::log(b) / std::log(a);
    return true;
  }
};

template <typename T>
struct Expm1Function {
  FOLLY_ALWAYS_INLINE void call(double& result, double a) {
    // The std::expm1 is more accurate than the expression std::exp(num) - 1.0
    // if num is close to zero. This matches Spark's implementation that uses
    // java.lang.StrictMath.expm1 as below. Ref:
    // https://docs.oracle.com/javase/8/docs/api/java/lang/StrictMath.html#expm1-double-.
    result = std::expm1(a);
  }
};

template <typename T>
struct CotFunction {
  FOLLY_ALWAYS_INLINE void call(double& result, double a) {
    result = 1 / std::tan(a);
  }
};

template <typename T>
struct Atan2Function {
  FOLLY_ALWAYS_INLINE void call(double& result, double y, double x) {
    // Spark (as of Spark 3.5)'s atan2 SQL function is internally calculated by
    // Math.atan2(y + 0.0, x + 0.0). We do the same here for compatibility.
    //
    // The sign (+/-) for 0.0 matters because it could make atan2 output
    // different results. For example:

    // * std::atan2(0.0, 0.0) = 0
    // * std::atan2(0.0, -0.0) = 3.1415926535897931
    // * std::atan2(-0.0, -0.0) = -3.1415926535897931
    // * std::atan2(-0.0, 0.0) = 0

    // By doing x + 0.0 or y + 0.0, we make sure all the -0s have been
    // replaced by 0s before sending to atan2 function. So the function
    // will always return atan2(0.0, 0.0) = 0 for atan2(+0.0/-0.0, +0.0/-0.0).
    result = std::atan2(y + 0.0, x + 0.0);
  }
};

template <typename T>
struct Log10Function {
  FOLLY_ALWAYS_INLINE bool call(double& result, double a) {
    if (a <= 0.0) {
      return false;
    }
    result = std::log10(a);
    return true;
  }
};

template <typename T>
struct IsNanFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(bool& result, TInput a) {
    result = std::isnan(a);
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void callNullable(bool& result, const TInput* a) {
    if (a) {
      call(result, *a);
    } else {
      result = false;
    }
  }
};

template <typename T>
struct ToHexVarbinaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varbinary>& input) {
    ToHexUtil::toHex(input, result);
  }
};

template <typename T>
struct ToHexVarcharFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input) {
    ToHexUtil::toHex(input, result);
  }
};

template <typename T>
struct ToHexBigintFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<int64_t>& input) {
    ToHexUtil::toHex(input, result);
  }
};

namespace detail {
FOLLY_ALWAYS_INLINE static int8_t fromHex(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }

  if (c >= 'A' && c <= 'F') {
    return 10 + c - 'A';
  }

  if (c >= 'a' && c <= 'f') {
    return 10 + c - 'a';
  }
  return -1;
}
} // namespace detail

template <typename T>
struct WidthBucketFunction {
  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      double value,
      double bound1,
      double bound2,
      int64_t numBuckets) {
    // NULL would be returned if the input arguments don't follow conditions
    // list belows:
    // - `numBuckets` must be greater than zero and be less than Long.MaxValue.
    // - `value`, `bound1`, and `bound2` cannot be NaN.
    // - `bound1` bound cannot equal `bound2`.
    // - `bound1` and `bound2` must be finite.
    if (shouldReturnNull(value, bound1, bound2, numBuckets)) {
      return false;
    }

    result = computeBucketNumber(value, bound1, bound2, numBuckets);
    return true;
  }

 private:
  static FOLLY_ALWAYS_INLINE bool shouldReturnNull(
      double value,
      double bound1,
      double bound2,
      int64_t numBuckets) {
    return numBuckets <= 0 ||
        numBuckets == std::numeric_limits<int64_t>::max() ||
        std::isnan(value) || bound1 == bound2 || !std::isfinite(bound1) ||
        !std::isfinite(bound2);
  }

  static FOLLY_ALWAYS_INLINE int64_t computeBucketNumber(
      double value,
      double bound1,
      double bound2,
      int64_t numBuckets) {
    if (bound1 < bound2) {
      if (value < bound1) {
        return 0;
      }

      if (value >= bound2) {
        return numBuckets + 1;
      }
    } else { // bound1 > bound2 case
      if (value > bound1) {
        return 0;
      }
      if (value <= bound2) {
        return numBuckets + 1;
      }
    }
    return static_cast<int64_t>(
               (numBuckets * (value - bound1) / (bound2 - bound1))) +
        1;
  }
};

template <typename T>
struct UnHexFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varbinary>& result,
      const arg_type<Varchar>& input) {
    const auto resultSize = (input.size() + 1) >> 1;
    result.resize(resultSize);
    const char* inputBuffer = input.data();
    char* resultBuffer = result.data();

    int32_t i = 0;
    if ((input.size() & 0x01) != 0) {
      const auto v = detail::fromHex(inputBuffer[0]);
      if (v == -1) {
        return false;
      }
      resultBuffer[0] = v;
      i += 1;
    }

    while (i < input.size()) {
      const auto first = detail::fromHex(inputBuffer[i]);
      const auto second = detail::fromHex(inputBuffer[i + 1]);
      if (first == -1 || second == -1) {
        return false;
      }
      resultBuffer[(i + 1) / 2] = (first << 4) | second;
      i += 2;
    }
    return true;
  }
};

template <typename T>
struct RIntFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(double& result, double input) {
    result = std::rint(input);
  }
};

template <typename TExec>
struct CheckedAddFunction {
  template <typename T>
  FOLLY_ALWAYS_INLINE Status call(T& result, const T& a, const T& b) {
    if constexpr (std::is_integral_v<T>) {
      T res;
      VELOX_USER_RETURN(
          __builtin_add_overflow(a, b, &res),
          "Arithmetic overflow: {} + {}",
          a,
          b);
      result = res;
    } else {
      result = a + b;
    }
    return Status::OK();
  }
};

template <typename TExec>
struct CheckedSubtractFunction {
  template <typename T>
  FOLLY_ALWAYS_INLINE Status call(T& result, const T& a, const T& b) {
    if constexpr (std::is_integral_v<T>) {
      VELOX_USER_RETURN(
          __builtin_sub_overflow(a, b, &result),
          "Arithmetic overflow: {} - {}",
          a,
          b);
    } else {
      result = a - b;
    }
    return Status::OK();
  }
};

template <typename TExec>
struct CheckedMultiplyFunction {
  template <typename T>
  FOLLY_ALWAYS_INLINE Status call(T& result, const T& a, const T& b) {
    if constexpr (std::is_integral_v<T>) {
      VELOX_USER_RETURN(
          __builtin_mul_overflow(a, b, &result),
          "Arithmetic overflow: {} * {}",
          a,
          b);
    } else {
      result = a * b;
    }
    return Status::OK();
  }
};

template <typename TExec>
struct CheckedDivideFunction {
  template <typename T>
  FOLLY_ALWAYS_INLINE Status call(T& result, const T& a, const T& b) {
    VELOX_USER_RETURN_EQ(b, 0, "division by zero");
    if constexpr (std::is_integral_v<T>) {
      VELOX_USER_RETURN(
          a == std::numeric_limits<T>::min() && b == -1,
          "Arithmetic overflow: {} / {}",
          a,
          b);
    }
    result = a / b;
    return Status::OK();
  }
};

/// Implements integral division with truncation towards zero.
/// Returns Null if divisor is 0.
template <typename TExec>
struct IntegralDivideFunction {
  template <typename T>
  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const T& a, const T& b) {
    if (b == 0) {
      return false;
    }
    // In Java, Long.MIN_VALUE is -2^63 and Long.MAX_VALUE is 2^63 - 1.
    // Dividing Long.MIN_VALUE by -1 overflows because the positive
    // result (+2^63) cannot be represented in a signed 64-bit integer.
    // Java integer arithmetic wraps around on overflow (two's complement),
    // so Long.MIN_VALUE / -1 evaluates to Long.MIN_VALUE itself instead
    // of throwing an exception.
    if (a == std::numeric_limits<int64_t>::min() && b == -1) {
      result = a;
      return true;
    }

    result = a / b;
    return true;
  }
};

/// Implements integral division with truncation towards zero.
/// Returns Error if divisor is 0 or overflow.
template <typename TExec>
struct CheckedIntegralDivideFunction {
  template <typename T>
  FOLLY_ALWAYS_INLINE Status call(int64_t& result, const T& a, const T& b) {
    VELOX_USER_RETURN_EQ(b, 0, "Division by zero");
    VELOX_USER_RETURN(
        a == std::numeric_limits<int64_t>::min() && b == -1,
        "Overflow in integral divide");
    result = a / b;
    return Status::OK();
  }
};
/// Checked interval arithmetic functions for ANSI mode support

namespace detail {

template <typename TResult>
FOLLY_ALWAYS_INLINE Status setIntervalResult(
    TResult& result,
    const int256_t& value,
    const char* overflowMessage) {
  const int256_t min = std::numeric_limits<TResult>::min();
  const int256_t max = std::numeric_limits<TResult>::max();
  VELOX_USER_RETURN(value < min || value > max, overflowMessage);
  result = static_cast<TResult>(value);
  return Status::OK();
}

template <typename TResult>
FOLLY_ALWAYS_INLINE Status
roundHalfUpDouble(TResult& result, double value, const char* overflowMessage) {
  if (!std::isfinite(value)) {
    VELOX_USER_RETURN(true, overflowMessage);
  }
  const double rounded = std::round(value);
  const double min = static_cast<double>(std::numeric_limits<TResult>::min());
  const double max = static_cast<double>(std::numeric_limits<TResult>::max());
  VELOX_USER_RETURN(rounded < min || rounded > max, overflowMessage);
  result = static_cast<TResult>(rounded);
  return Status::OK();
}

template <typename TResult, typename TNum>
FOLLY_ALWAYS_INLINE Status multiplyIntegralInterval(
    TResult& result,
    TResult interval,
    TNum num,
    const char* overflowMessage) {
  const int128_t product =
      static_cast<int128_t>(interval) * static_cast<int128_t>(num);
  return setIntervalResult<TResult>(result, int256_t(product), overflowMessage);
}

template <typename TResult>
FOLLY_ALWAYS_INLINE Status multiplyDecimalInterval(
    TResult& result,
    TResult interval,
    int128_t decimalValue,
    uint8_t scale,
    const char* overflowMessage) {
  const int256_t product = int256_t(interval) * int256_t(decimalValue);
  const int256_t scaleFactor =
      int256_t(velox::DecimalUtil::kPowersOfTen[scale]);

  int256_t quotient = product / scaleFactor;
  const int256_t remainder = product % scaleFactor;
  if (remainder != 0) {
    const int256_t absRemainder = remainder < 0 ? -remainder : remainder;
    if (absRemainder * 2 >= scaleFactor) {
      quotient += (product < 0) ? -1 : 1;
    }
  }

  return setIntervalResult<TResult>(result, quotient, overflowMessage);
}

template <typename TResult, typename TNum>
FOLLY_ALWAYS_INLINE Status divideIntegralInterval(
    TResult& result,
    TResult interval,
    TNum num,
    const char* overflowMessage) {
  VELOX_USER_RETURN_EQ(num, 0, "Division by zero");

  const int128_t numerator = static_cast<int128_t>(interval);
  const int128_t denominator = static_cast<int128_t>(num);
  int128_t quotient = numerator / denominator;
  const int128_t remainder = numerator % denominator;
  if (remainder != 0) {
    const int128_t absRemainder = remainder < 0 ? -remainder : remainder;
    const int128_t absDenominator =
        denominator < 0 ? -denominator : denominator;
    if (absRemainder * 2 >= absDenominator) {
      quotient += ((numerator < 0) ^ (denominator < 0)) ? -1 : 1;
    }
  }

  return setIntervalResult<TResult>(
      result, int256_t(quotient), overflowMessage);
}

template <typename TResult>
FOLLY_ALWAYS_INLINE Status divideDecimalInterval(
    TResult& result,
    TResult interval,
    int128_t decimalValue,
    uint8_t scale,
    const char* overflowMessage) {
  VELOX_USER_RETURN_EQ(decimalValue, 0, "Division by zero");

  const int256_t scaleFactor =
      int256_t(velox::DecimalUtil::kPowersOfTen[scale]);
  const int256_t numerator = int256_t(interval) * scaleFactor;
  const int256_t denominator = int256_t(decimalValue);
  int256_t quotient = numerator / denominator;
  const int256_t remainder = numerator % denominator;
  if (remainder != 0) {
    const int256_t absRemainder = remainder < 0 ? -remainder : remainder;
    const int256_t absDenominator =
        denominator < 0 ? -denominator : denominator;
    if (absRemainder * 2 >= absDenominator) {
      quotient += ((numerator < 0) ^ (denominator < 0)) ? -1 : 1;
    }
  }

  return setIntervalResult<TResult>(result, quotient, overflowMessage);
}

template <typename T>
FOLLY_ALWAYS_INLINE int128_t toInt128(const T& value) {
  if constexpr (std::is_same_v<T, int128_t>) {
    return value;
  }
  return static_cast<int128_t>(value);
}

} // namespace detail

/// Multiply interval by double with overflow checking.
/// Throws error on overflow instead of clamping to min/max.
template <typename TExec>
struct CheckedIntervalMultiplyFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename T1, typename T2>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      const T1* /*a*/,
      const T2* /*b*/) {
    VELOX_CHECK_EQ(inputTypes.size(), 2);
    if (inputTypes[0]->isIntervalDayTime() ||
        inputTypes[0]->isIntervalYearMonth()) {
      intervalIndex_ = 0;
      numericIndex_ = 1;
    } else {
      intervalIndex_ = 1;
      numericIndex_ = 0;
    }
    numericIsDecimal_ = inputTypes[numericIndex_]->isDecimal();
    if (numericIsDecimal_) {
      numericScale_ =
          getDecimalPrecisionScale(*inputTypes[numericIndex_]).second;
    }
  }

  template <typename TResult, typename T1, typename T2>
  FOLLY_ALWAYS_INLINE Status call(TResult& result, const T1& a, const T2& b) {
    if (intervalIndex_ == 0) {
      return multiplyImpl<TResult>(result, a, b);
    }
    return multiplyImpl<TResult>(result, b, a);
  }

 private:
  template <typename TResult, typename TInterval, typename TNum>
  FOLLY_ALWAYS_INLINE Status
  multiplyImpl(TResult& result, const TInterval& interval, const TNum& num) {
    static constexpr const char* kOverflow = "Interval overflow in multiply";
    if (numericIsDecimal_) {
      return detail::multiplyDecimalInterval<TResult>(
          result,
          static_cast<TResult>(interval),
          detail::toInt128(num),
          numericScale_,
          kOverflow);
    }
    if constexpr (std::is_floating_point_v<TNum>) {
      const double value =
          static_cast<double>(interval) * static_cast<double>(num);
      return detail::roundHalfUpDouble<TResult>(result, value, kOverflow);
    }
    if constexpr (std::is_integral_v<TNum>) {
      return detail::multiplyIntegralInterval<TResult>(
          result, static_cast<TResult>(interval), num, kOverflow);
    }
    VELOX_USER_RETURN(true, kOverflow);
  }

  int8_t intervalIndex_{0};
  int8_t numericIndex_{1};
  bool numericIsDecimal_{false};
  uint8_t numericScale_{0};
};

/// Divide interval by double with overflow and division-by-zero checking.
template <typename TExec>
struct CheckedIntervalDivideFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename T1, typename T2>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      const T1* /*a*/,
      const T2* /*b*/) {
    VELOX_CHECK_EQ(inputTypes.size(), 2);
    numericIsDecimal_ = inputTypes[1]->isDecimal();
    if (numericIsDecimal_) {
      numericScale_ = getDecimalPrecisionScale(*inputTypes[1]).second;
    }
  }

  template <typename TResult, typename TNum>
  FOLLY_ALWAYS_INLINE Status call(TResult& result, TResult interval, TNum num) {
    static constexpr const char* kOverflow = "Interval overflow in divide";
    if (numericIsDecimal_) {
      return detail::divideDecimalInterval<TResult>(
          result, interval, detail::toInt128(num), numericScale_, kOverflow);
    }
    if constexpr (std::is_floating_point_v<TNum>) {
      VELOX_USER_RETURN_EQ(num, 0, "Division by zero");
      const double value =
          static_cast<double>(interval) / static_cast<double>(num);
      return detail::roundHalfUpDouble<TResult>(result, value, kOverflow);
    }
    if constexpr (std::is_integral_v<TNum>) {
      return detail::divideIntegralInterval<TResult>(
          result, interval, num, kOverflow);
    }
    VELOX_USER_RETURN(true, kOverflow);
  }

 private:
  bool numericIsDecimal_{false};
  uint8_t numericScale_{0};
};

/// Add two intervals with overflow checking.
template <typename TExec>
struct CheckedIntervalAddFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename T>
  FOLLY_ALWAYS_INLINE Status call(T& result, const T& a, const T& b) {
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, int32_t>) {
      VELOX_USER_RETURN(
          __builtin_add_overflow(a, b, &result), "Interval overflow in add");
    } else {
      result = a + b;
    }
    return Status::OK();
  }
};

/// Subtract two intervals with overflow checking.
template <typename TExec>
struct CheckedIntervalSubtractFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename T>
  FOLLY_ALWAYS_INLINE Status call(T& result, const T& a, const T& b) {
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, int32_t>) {
      VELOX_USER_RETURN(
          __builtin_sub_overflow(a, b, &result),
          "Interval overflow in subtract");
    } else {
      result = a - b;
    }
    return Status::OK();
  }
};

} // namespace facebook::velox::functions::sparksql
