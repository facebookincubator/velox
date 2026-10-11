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

#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include "folly/Conv.h"
#include "folly/Likely.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Int128.h"

#if defined(_MSC_VER) && !defined(__SIZEOF_INT128__)
#include "velox/common/base/windows/CheckedArithmetic.h"
#endif

namespace facebook::velox {

/// Compute in mathematical integers, reporting whether the result fits R.
/// Always write the low sizeof(R) bytes, including on overflow. Operands are
/// copied before writing, so result may alias either operand.
/// Operands and result must be integers of at most 128 bits; R cannot be bool.
template <typename A, typename B, typename R>
bool addOverflow(A a, B b, R* result) noexcept {
#if defined(_MSC_VER) && !defined(__SIZEOF_INT128__)
  return detail::addOverflow(a, b, result, false);
#else
  return __builtin_add_overflow(a, b, result);
#endif
}

template <typename A, typename B, typename R>
bool subtractOverflow(A a, B b, R* result) noexcept {
#if defined(_MSC_VER) && !defined(__SIZEOF_INT128__)
  return detail::addOverflow(a, b, result, true);
#else
  return __builtin_sub_overflow(a, b, result);
#endif
}

template <typename A, typename B, typename R>
bool multiplyOverflow(A a, B b, R* result) noexcept {
#if defined(_MSC_VER) && !defined(__SIZEOF_INT128__)
  return detail::multiplyOverflow(a, b, result);
#else
  return __builtin_mul_overflow(a, b, result);
#endif
}

/// Folly's checked conversions, extended to the MSVC 128-bit integer classes.
/// Strings must contain a complete number (surrounding whitespace is allowed).
/// This is not a parseTo customization: Folly does not validate the unconsumed
/// suffix for non-arithmetic custom types. No StringPiece* prefix parsing is
/// provided for the MSVC classes.
template <typename T, typename F>
auto tryToWithInt128(const F& value) {
#if defined(_MSC_VER) && !defined(__SIZEOF_INT128__)
  if constexpr (
      detail::Int128Type<T> ||
      (detail::Int128Type<F> && std::is_arithmetic_v<T>)) {
    return detail::tryToInt128<T>(value);
  } else
#endif
  {
    return folly::tryTo<T>(value);
  }
}

template <typename T>
T checkedPlus(T a, T b, const char* typeName = "integer") {
  T result;
  bool overflow = addOverflow(a, b, &result);
  if (UNLIKELY(overflow)) {
    VELOX_ARITHMETIC_ERROR("{} overflow: {} + {}", typeName, a, b);
  }
  return result;
}

template <typename T>
T checkedMinus(T a, T b, const char* typeName = "integer") {
  T result;
  bool overflow = subtractOverflow(a, b, &result);
  if (UNLIKELY(overflow)) {
    VELOX_ARITHMETIC_ERROR("{} overflow: {} - {}", typeName, a, b);
  }
  return result;
}

template <typename T>
T checkedMultiply(T a, T b, const char* typeName = "integer") {
  T result;
  bool overflow = multiplyOverflow(a, b, &result);
  if (UNLIKELY(overflow)) {
    VELOX_ARITHMETIC_ERROR("{} overflow: {} * {}", typeName, a, b);
  }
  return result;
}

template <typename T>
T checkedDivide(T a, T b) {
  if (b == 0) {
    VELOX_ARITHMETIC_ERROR("division by zero");
  }

  // Signed integers cannot represent abs(std::numeric_limits<T>::min()).
  if constexpr (
      std::numeric_limits<T>::is_integer && std::numeric_limits<T>::is_signed) {
    if (UNLIKELY(a == std::numeric_limits<T>::min() && b == -1)) {
      VELOX_ARITHMETIC_ERROR("integer overflow: {} / {}", a, b);
    }
  }
  return a / b;
}

template <typename T>
T checkedModulus(T a, T b) {
  if (UNLIKELY(b == 0)) {
    VELOX_ARITHMETIC_ERROR("Cannot divide by 0");
  }
  // std::numeric_limits<int64_t>::min() % -1 could crash the program since
  // abs(std::numeric_limits<int64_t>::min()) can not be represented in
  // int64_t.
  if (b == -1) {
    return 0;
  }
  return (a % b);
}

template <typename T>
T checkedNegate(T a) {
  if (UNLIKELY(a == std::numeric_limits<T>::min())) {
    VELOX_ARITHMETIC_ERROR("Cannot negate minimum value");
  }
  return std::negate<std::remove_cv_t<T>>()(a);
}

} // namespace facebook::velox
