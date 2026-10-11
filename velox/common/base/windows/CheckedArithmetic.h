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

#if !defined(_MSC_VER) || defined(__SIZEOF_INT128__)
#error "Include the common CheckedArithmetic header instead."
#endif

#include <folly/Conv.h>
#include <cctype>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include "velox/common/base/Int128.h"

namespace facebook::velox::detail {

template <typename A, typename B, typename R>
using OverflowUnsigned = std::conditional_t<
    (sizeof(A) > sizeof(uint64_t) || sizeof(B) > sizeof(uint64_t) ||
     sizeof(R) > sizeof(uint64_t)),
    uint128_t,
    uint64_t>;

template <typename U, typename T>
U unsignedMagnitude(T value) {
  static_assert(std::numeric_limits<T>::is_integer && sizeof(T) <= sizeof(U));
  if constexpr (std::is_same_v<T, bool>) {
    return U{value ? 1 : 0};
  } else {
    const auto bits = static_cast<U>(value);
    return value < 0 ? U{0} - bits : bits;
  }
}

template <typename U, typename R>
U magnitudeLimit(bool negative) {
  static_assert(
      std::numeric_limits<R>::is_integer && !std::is_same_v<R, bool> &&
      sizeof(R) <= sizeof(U));
  // An unsigned destination cannot represent a negative nonzero value.
  return negative ? unsignedMagnitude<U>(std::numeric_limits<R>::min())
                  : static_cast<U>(std::numeric_limits<R>::max());
}

template <typename A, typename B, typename R>
bool addOverflow(A a, B b, R* result, bool subtract) {
  using U = OverflowUnsigned<A, B, R>;
  U left = unsignedMagnitude<U>(a);
  U right = unsignedMagnitude<U>(b);
  bool negative = a < 0;
  const bool rightNegative = (b < 0) != subtract;
  U magnitude;
  bool overflow;
  if (negative == rightNegative) {
    const U limit = magnitudeLimit<U, R>(negative);
    overflow = left > limit || right > limit - left;
    magnitude = left + right;
  } else {
    if (left < right) {
      std::swap(left, right);
      negative = rightNegative;
    }
    magnitude = left - right;
    overflow = magnitude > magnitudeLimit<U, R>(negative);
  }
  *result = static_cast<R>(negative ? U{0} - magnitude : magnitude);
  return overflow;
}

template <typename A, typename B, typename R>
bool multiplyOverflow(A a, B b, R* result) {
  using U = OverflowUnsigned<A, B, R>;
  const U left = unsignedMagnitude<U>(a);
  const U right = unsignedMagnitude<U>(b);
  const bool negative = (a < 0) != (b < 0);
  const U limit = magnitudeLimit<U, R>(negative);
  const bool overflow = left != 0 && right > limit / left;
  const U product = left * right;
  *result = static_cast<R>(negative ? U{0} - product : product);
  return overflow;
}

template <typename T>
folly::Expected<T, folly::ConversionCode> parseInt128(
    folly::StringPiece input) {
  using Code = folly::ConversionCode;
  while (!input.empty() &&
         ((input.front() >= '\t' && input.front() <= '\r') ||
          input.front() == ' ')) {
    input.advance(1);
  }
  if (input.empty()) {
    return folly::makeUnexpected(Code::EMPTY_INPUT_STRING);
  }
  bool negative = false;
  if constexpr (std::numeric_limits<T>::is_signed) {
    if (input.front() == '-' || input.front() == '+') {
      negative = input.front() == '-';
      input.advance(1);
    } else if (input.front() < '0' || input.front() > '9') {
      return folly::makeUnexpected(Code::INVALID_LEADING_CHAR);
    }
    if (input.empty()) {
      return folly::makeUnexpected(Code::NO_DIGITS);
    }
  }
  if (input.front() < '0' || input.front() > '9') {
    return folly::makeUnexpected(Code::NON_DIGIT_CHAR);
  }

  const uint128_t limit = magnitudeLimit<uint128_t, T>(negative);
  const uint128_t quotient = limit / 10;
  const auto remainder = static_cast<unsigned>(limit % 10);
  uint128_t magnitude = 0;
  while (!input.empty() && input.front() >= '0' && input.front() <= '9') {
    const auto digit = static_cast<unsigned>(input.front() - '0');
    if (magnitude > quotient || (magnitude == quotient && digit > remainder)) {
      return folly::makeUnexpected(
          negative ? Code::NEGATIVE_OVERFLOW : Code::POSITIVE_OVERFLOW);
    }
    magnitude = magnitude * 10 + digit;
    input.advance(1);
  }
  for (const auto character : input) {
    if (!std::isspace(static_cast<unsigned char>(character))) {
      return folly::makeUnexpected(Code::NON_WHITESPACE_AFTER_END);
    }
  }
  return static_cast<T>(negative ? uint128_t{0} - magnitude : magnitude);
}

template <typename T, typename F>
folly::Expected<T, folly::ConversionCode> tryToInt128(const F& value) {
  using Code = folly::ConversionCode;
  if constexpr (std::is_convertible_v<F, folly::StringPiece>) {
    return parseInt128<T>(folly::StringPiece(value));
  } else if constexpr (std::is_same_v<T, bool>) {
    return value != 0;
  } else if constexpr (std::is_same_v<F, bool>) {
    return T(value ? 1 : 0);
  } else if constexpr (std::is_floating_point_v<F>) {
    // Powers of two are exact. In particular, (float)max rounds up to the
    // exclusive upper bound, so comparing against that rounded max is unsafe.
    const F upper = std::ldexp(F{1}, std::numeric_limits<T>::digits);
    const F lower = std::numeric_limits<T>::is_signed ? -upper : F{0};
    if (!std::isfinite(value) || value < lower || value >= upper ||
        std::trunc(value) != value) {
      return folly::makeUnexpected(Code::ARITH_LOSS_OF_PRECISION);
    }
    return static_cast<T>(value);
  } else if constexpr (std::is_floating_point_v<T>) {
    const T result = static_cast<T>(value);
    const auto witness = tryToInt128<F>(result);
    if (!witness.hasValue() || witness.value() != value) {
      return folly::makeUnexpected(Code::ARITH_LOSS_OF_PRECISION);
    }
    return result;
  } else {
    const bool negative = value < 0;
    const auto magnitude = unsignedMagnitude<uint128_t>(value);
    if (magnitude > magnitudeLimit<uint128_t, T>(negative)) {
      return folly::makeUnexpected(
          negative ? Code::ARITH_NEGATIVE_OVERFLOW
                   : Code::ARITH_POSITIVE_OVERFLOW);
    }
    return static_cast<T>(value);
  }
}

} // namespace facebook::velox::detail
