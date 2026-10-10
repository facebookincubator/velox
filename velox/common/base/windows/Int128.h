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
#error "Use the common Int128 selector; this implementation requires MSVC."
#endif

/// Implement MSVC 128-bit integers using two little-endian 64-bit limbs,
/// wrapping arithmetic, saturating conversions, and decimal formatting.

#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <fmt/format.h>
#include <folly/Conv.h>
#include <folly/Hash.h>
#include <intrin.h>

namespace facebook::velox {

namespace detail {

std::to_chars_result uint128ToChars(
    char* first,
    char* last,
    uint64_t high,
    uint64_t low,
    bool negative);

// Cold, out-of-line throw so the division hot paths stay small enough for
// MSVC to inline (an inline `throw` pulls in exception-object construction).
[[noreturn]] __declspec(noinline) inline void throwDivideByZero(
    const char* message) {
  throw std::runtime_error(message);
}

template <typename T>
inline constexpr bool kIsIntegralOperand =
    std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>;

// Arithmetic also accepts bool and unscoped enums, with native integral
// promotion. Keep construction and shift constraints separate (notably bool).
template <typename T, bool = std::is_enum_v<T>>
struct IsArithmeticOperand : std::is_integral<T> {};

template <typename T>
struct IsArithmeticOperand<T, true>
    : std::is_convertible<T, std::underlying_type_t<T>> {};

template <typename T>
inline constexpr bool kIsArithmeticOperand = IsArithmeticOperand<T>::value;

template <typename T>
constexpr auto normalizeIntegralOperand(T value) {
  return +value;
}

template <typename T>
inline constexpr bool kIsFloatingOperand = std::is_floating_point_v<T>;

// 64-bit carry/borrow/multiply/divide primitives. x64 maps each to its MSVC
// intrinsic (`adc`, `sbb`, `mul`, `div`); other targets (ARM64) use portable
// forms that MSVC lowers to `adds/adcs`, `subs/sbcs`, `mul/umulh` and two
// hardware `udiv`s respectively.
inline unsigned char
addCarry64(unsigned char carryIn, uint64_t a, uint64_t b, uint64_t* out) {
#if defined(_M_X64)
  return _addcarry_u64(carryIn, a, b, out);
#else
  const uint64_t partial = a + b;
  const uint64_t sum = partial + carryIn;
  *out = sum;
  return static_cast<unsigned char>((partial < a) | (sum < partial));
#endif
}

inline unsigned char
subBorrow64(unsigned char borrowIn, uint64_t a, uint64_t b, uint64_t* out) {
#if defined(_M_X64)
  return _subborrow_u64(borrowIn, a, b, out);
#else
  const uint64_t partial = a - b;
  const uint64_t difference = partial - borrowIn;
  *out = difference;
  return static_cast<unsigned char>((a < b) | (partial < borrowIn));
#endif
}

inline uint64_t umul128(uint64_t a, uint64_t b, uint64_t* high) {
#if defined(_M_X64)
  return _umul128(a, b, high);
#else
  *high = __umulh(a, b);
  return a * b;
#endif
}

// Quotient and remainder of a 64/64 divide. Forming both in one helper lets
// MSVC x64 emit a single `div` for the pair (as separate expressions at the
// call sites it issued two), and keeps the magic-multiply lowering for a
// constant `d`, which `_udiv128(0, n, d, ...)` would replace with a `div`.
inline uint64_t udivrem64(uint64_t n, uint64_t d, uint64_t* remainder) {
  *remainder = n % d;
  return n / d;
}

// Two's-complement negation of (hi:lo) when `mask` is all-ones; identity when
// it is zero. Spelled (x + mask) ^ mask rather than the equivalent
// (x ^ mask) - mask so that the xors come after the carry chain: in the
// xor-first form MSVC x64 schedules the high-limb `xor` between `sub` and
// `sbb`, clobbering CF, and spills it via `setb; add al, -1`.
inline void negateIfMask(
    uint64_t mask,
    uint64_t lo,
    uint64_t hi,
    uint64_t* outLo,
    uint64_t* outHi) {
  uint64_t sumLo;
  uint64_t sumHi;
  addCarry64(addCarry64(0, lo, mask, &sumLo), hi, mask, &sumHi);
  *outLo = sumLo ^ mask;
  *outHi = sumHi ^ mask;
}

// Divides the 128-bit value (high:low) by `divisor`, returning the 64-bit
// quotient and storing the remainder. Requires `high < divisor`, so the
// quotient fits in 64 bits (the `_udiv128` / x64 `div` precondition).
//
// Non-x64 targets have no 128/64 divide instruction. They use Hacker's
// Delight `divlu` (Warren, 2nd ed., fig. 9-3), i.e. Knuth Algorithm D at base
// 2^32: normalize the divisor, then produce the two 32-bit quotient digits
// with one hardware 64/64 `udiv` each plus at most two corrections per digit.
inline uint64_t udiv128By64(
    uint64_t high,
    uint64_t low,
    uint64_t divisor,
    uint64_t* remainder) {
#if defined(_M_X64)
  return _udiv128(high, low, divisor, remainder);
#else
  constexpr uint64_t kBase = uint64_t{1} << 32;
  constexpr uint64_t kDigitMask = kBase - 1;
  const int shift = std::countl_zero(divisor);
  const uint64_t normalizedDivisor = divisor << shift;
  // `low >> (64 - shift)` is undefined for shift == 0; `(low >> 1) >> 63` is
  // the same value for shift in [1, 63] and 0 for shift == 0.
  const uint64_t numeratorHigh = (high << shift) | ((low >> 1) >> (63 - shift));
  const uint64_t numeratorLow = low << shift;
  const uint64_t divisorHigh = normalizedDivisor >> 32;
  const uint64_t divisorLow = normalizedDivisor & kDigitMask;
  const uint64_t numeratorDigit1 = numeratorLow >> 32;
  const uint64_t numeratorDigit0 = numeratorLow & kDigitMask;

  uint64_t quotientDigit1 = numeratorHigh / divisorHigh;
  uint64_t partialRemainder = numeratorHigh - quotientDigit1 * divisorHigh;
  while (quotientDigit1 >= kBase ||
         quotientDigit1 * divisorLow >
             ((partialRemainder << 32) | numeratorDigit1)) {
    --quotientDigit1;
    partialRemainder += divisorHigh;
    if (partialRemainder >= kBase) {
      break;
    }
  }

  const uint64_t middle = ((numeratorHigh << 32) | numeratorDigit1) -
      quotientDigit1 * normalizedDivisor;
  uint64_t quotientDigit0 = middle / divisorHigh;
  partialRemainder = middle - quotientDigit0 * divisorHigh;
  while (quotientDigit0 >= kBase ||
         quotientDigit0 * divisorLow >
             ((partialRemainder << 32) | numeratorDigit0)) {
    --quotientDigit0;
    partialRemainder += divisorHigh;
    if (partialRemainder >= kBase) {
      break;
    }
  }

  *remainder = (((middle << 32) | numeratorDigit0) -
                quotientDigit0 * normalizedDivisor) >>
      shift;
  return (quotientDigit1 << 32) | quotientDigit0;
#endif
}

// Truncating floating-point -> 128-bit conversion returning the raw
// two's-complement limbs {high, low}. Out-of-range values saturate like
// libgcc/compiler-rt `__fixdfti` / `__fixunsdfti`: +/-inf and finite values
// beyond the range clamp to the nearest bound, NaN clamps toward its sign bit,
// and any negative input yields 0 for the unsigned form. Decodes the IEEE bit
// pattern directly instead of calling frexp/ldexp.
template <bool isSigned, typename T>
std::pair<uint64_t, uint64_t> floatingPointTo128(T value) {
  static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));
  using Bits =
      std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint64_t>;
  constexpr int32_t kFractionBits = std::numeric_limits<T>::digits - 1;
  constexpr int32_t kExponentBias = std::numeric_limits<T>::max_exponent - 1;
  constexpr Bits kExponentMask = 2 * std::numeric_limits<T>::max_exponent - 1;
  constexpr Bits kFractionMask = (Bits{1} << kFractionBits) - 1;
  constexpr int32_t kBits = sizeof(T) * 8;

  const Bits bits = std::bit_cast<Bits>(value);
  const bool negative = (bits >> (kBits - 1)) != 0;
  // Inf/NaN (all-ones exponent) decode to an exponent above every limit below.
  const int32_t exponent =
      static_cast<int32_t>((bits >> kFractionBits) & kExponentMask) -
      kExponentBias;
  if (exponent < 0) {
    return {0, 0};
  }
  if constexpr (isSigned) {
    // |value| >= 2^127. -2^127 itself is INT128_MIN, so a single bound check
    // on the magnitude covers both directions.
    if (exponent >= 127) {
      return negative
          ? std::pair<uint64_t, uint64_t>{uint64_t{1} << 63, 0}
          : std::pair<uint64_t, uint64_t>{~uint64_t{0} >> 1, ~uint64_t{0}};
    }
  } else {
    if (negative) {
      return {0, 0};
    }
    if (exponent >= 128) {
      return {~uint64_t{0}, ~uint64_t{0}};
    }
  }

  // |value| = significand * 2^shift with a 24/53-bit significand and
  // exponent <= 127, so every shift below is in [0, 63].
  const uint64_t significand = static_cast<uint64_t>(
      (bits & kFractionMask) | (Bits{1} << kFractionBits));
  const int32_t shift = exponent - kFractionBits;
  uint64_t high;
  uint64_t low;
  if (shift < 0) {
    high = 0;
    low = significand >> -shift;
  } else if (shift < 64) {
    high = (significand >> 1) >> (63 - shift);
    low = significand << shift;
  } else {
    high = significand << (shift - 64);
    low = 0;
  }
  if (negative) {
    low = 0 - low;
    high = ~high + (low == 0 ? 1 : 0);
  }
  return {high, low};
}

template <typename T>
T int128ToFloatingPoint(
    uint64_t magnitudeHigh,
    uint64_t magnitudeLow,
    bool negative) {
  using Bits = std::conditional_t<std::is_same_v<T, float>, uint32_t, uint64_t>;
  constexpr int32_t kPrecision = std::numeric_limits<T>::digits;
  constexpr int32_t kFractionBits = kPrecision - 1;
  constexpr int32_t kExponentBias = std::numeric_limits<T>::max_exponent - 1;
  constexpr int32_t kBits = sizeof(T) * 8;

  if (magnitudeHigh == 0 && magnitudeLow == 0) {
    return T{0};
  }

  if (magnitudeHigh == 0) {
    const T result = static_cast<T>(magnitudeLow);
    return negative ? -result : result;
  }

  const int32_t leadingZeros = std::countl_zero(magnitudeHigh);
  int32_t exponent = 127 - leadingZeros;
  const uint64_t normalizedHigh = (magnitudeHigh << leadingZeros) |
      (leadingZeros == 0 ? 0 : magnitudeLow >> (64 - leadingZeros));
  const uint64_t normalizedLow = magnitudeLow << leadingZeros;
  uint64_t significand = normalizedHigh >> (64 - kPrecision);
  const uint64_t discarded =
      normalizedHigh & ((uint64_t{1} << (64 - kPrecision)) - 1);
  const uint64_t halfway = uint64_t{1} << (63 - kPrecision);
  if (discarded > halfway ||
      (discarded == halfway &&
       (normalizedLow != 0 || (significand & 1) != 0))) {
    ++significand;
    if (significand == (uint64_t{1} << kPrecision)) {
      significand >>= 1;
      ++exponent;
    }
  }

  const Bits sign = negative ? Bits{1} << (kBits - 1) : 0;
  const Bits biasedExponent = static_cast<Bits>(exponent + kExponentBias)
      << kFractionBits;
  const Bits fraction =
      static_cast<Bits>(significand & ((uint64_t{1} << kFractionBits) - 1));
  return std::bit_cast<T>(sign | biasedExponent | fraction);
}

} // namespace detail

// Forward declarations
class UInt128;

class Int128 {
 public:
  constexpr Int128() : low_(0), high_(0) {}

  constexpr Int128(int64_t high, uint64_t low) : low_(low), high_(high) {}

  explicit constexpr Int128(bool value) : low_(value), high_(0) {}

  // Truncates toward zero; out-of-range values, +/-inf and NaN saturate (see
  // detail::floatingPointTo128).
  template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
  explicit Int128(T value) {
    const auto [high, low] = detail::floatingPointTo128<true>(value);
    low_ = low;
    high_ = static_cast<int64_t>(high);
  }

  template <
      typename T,
      std::enable_if_t<detail::kIsIntegralOperand<T>, int> = 0>
  constexpr Int128(T value)
      : low_(static_cast<uint64_t>(value)),
        high_(std::is_signed_v<T> && value < 0 ? -1 : 0) {}

  constexpr Int128(const Int128& other) = default;
  constexpr Int128& operator=(const Int128& other) = default;

  // Forward declaration - defined after UInt128 class
  constexpr Int128(const UInt128& value);

  // Getters
  constexpr int64_t high() const {
    return high_;
  }
  constexpr uint64_t low() const {
    return low_;
  }

  // Sign test via the high limb only. Equivalent to `*this < 0` but lowers to a
  // single sign-bit extract on MSVC, avoiding the branchy generic operator<
  // (which compares high_ then low_). Hot for decimal aggregation; see
  // DecimalUtil::addWithOverflow.
  constexpr bool isNegative() const {
    return high_ < 0;
  }

  // Conversion operators
  // `(high_ | low_)` is non-zero iff the 128-bit value is non-zero. The bit-or
  // is branch-free (`or; setne`) on MSVC.
  explicit constexpr operator bool() const {
    return (static_cast<uint64_t>(high_) | low_) != 0;
  }
  template <
      typename T,
      std::enable_if_t<detail::kIsIntegralOperand<T>, int> = 0>
  constexpr operator T() const {
    return static_cast<T>(low_);
  }
  explicit operator float() const {
    return toFloatingPoint<float>();
  }
  explicit operator double() const {
    return toFloatingPoint<double>();
  }
  explicit operator long double() const {
    // MSVC represents long double in the same IEEE format as double.
    return static_cast<double>(*this);
  }

  // Comparison with primitive integer types. `<` / `>=` use
  // lessThanIntegralOperand so `x < 0` / `x >= 0` fold to a sign-bit test.
  constexpr bool operator<(int64_t other) const {
    return lessThanIntegralOperand(*this, Int128(other));
  }

  constexpr bool operator>(int64_t other) const {
    return *this > Int128(other);
  }

  constexpr bool operator==(int64_t other) const {
    return *this == Int128(other);
  }

  constexpr bool operator!=(int64_t other) const {
    return *this != Int128(other);
  }

  constexpr bool operator<=(int64_t other) const {
    return *this <= Int128(other);
  }

  constexpr bool operator>=(int64_t other) const {
    return !lessThanIntegralOperand(*this, Int128(other));
  }

  // Arithmetic operators. At runtime use `_addcarry_u64`/`_subborrow_u64` so
  // MSVC lowers the 128-bit add/sub to branch-free `add; adc` / `sub; sbb`.
  // The intrinsics are not constexpr; the `is_constant_evaluated` branch
  // keeps the operator usable in constant expressions.
  //
  // Single-return shape: every runtime/constexpr split in this header computes
  // the limbs into locals and constructs the result in one `return`.
  // This avoids stack spills and store-forwarding stalls on MSVC.
  //
  // The high half is computed in uint64_t (defined two's-complement wrap)
  // and reinterpreted as int64_t at the end. Doing the add as signed int64_t
  // would be UB on overflow -- reachable e.g. via unary negation of
  // INT128_MIN, which lowers to `0 - INT128_MIN` and would do `0 - INT64_MIN`
  // on the high half.
  constexpr Int128 operator+(const Int128& other) const {
    uint64_t newLow;
    uint64_t newHigh;
#if defined(_M_X64)
    if (!std::is_constant_evaluated()) {
      const unsigned char carry = _addcarry_u64(0, low_, other.low_, &newLow);
      _addcarry_u64(
          carry,
          static_cast<uint64_t>(high_),
          static_cast<uint64_t>(other.high_),
          &newHigh);
    } else
#endif
    {
      // Portable / constant-evaluated path (also the ARM64 runtime path,
      // which MSVC lowers to `adds; adcs`).
      newLow = low_ + other.low_;
      newHigh = static_cast<uint64_t>(high_) +
          static_cast<uint64_t>(other.high_) + (newLow < low_ ? 1 : 0);
    }
    return Int128(static_cast<int64_t>(newHigh), newLow);
  }

  constexpr Int128 operator-(const Int128& other) const {
    // High half in uint64_t to avoid signed-overflow UB; see operator+.
    uint64_t newLow;
    uint64_t newHigh;
#if defined(_M_X64)
    if (!std::is_constant_evaluated()) {
      const unsigned char borrow = _subborrow_u64(0, low_, other.low_, &newLow);
      _subborrow_u64(
          borrow,
          static_cast<uint64_t>(high_),
          static_cast<uint64_t>(other.high_),
          &newHigh);
    } else
#endif
    {
      // Portable / constant-evaluated path (also the ARM64 runtime path).
      newLow = low_ - other.low_;
      newHigh = static_cast<uint64_t>(high_) -
          static_cast<uint64_t>(other.high_) - (newLow > low_ ? 1 : 0);
    }
    return Int128(static_cast<int64_t>(newHigh), newLow);
  }

#if defined(_M_X64)
  // x64 overflow-checked addition and subtraction: `add; adc; seto` /
  // `sub; sbb; seto`. Writing the result limbs in place avoids
  // partial-register dependencies. *result may alias either operand.
  static bool addOverflowX64(const Int128& a, const Int128& b, Int128* result) {
    const int64_t aHigh = a.high_;
    const int64_t bHigh = b.high_;
    const unsigned char carry = _addcarry_u64(0, a.low_, b.low_, &result->low_);
    return _add_overflow_i64(carry, aHigh, bHigh, &result->high_);
  }

  static bool subOverflowX64(const Int128& a, const Int128& b, Int128* result) {
    const int64_t aHigh = a.high_;
    const int64_t bHigh = b.high_;
    const unsigned char borrow =
        _subborrow_u64(0, a.low_, b.low_, &result->low_);
    return _sub_overflow_i64(borrow, aHigh, bHigh, &result->high_);
  }
#endif

  constexpr Int128 operator+() const {
    return *this;
  }

  // Branch-free negation with defined wrapping for INT128_MIN.
  constexpr Int128 operator-() const {
    return Int128(0, 0) - *this;
  }

  Int128& operator+=(const Int128& other) {
    *this = *this + other;
    return *this;
  }

  Int128& operator-=(const Int128& other) {
    *this = *this - other;
    return *this;
  }

  // Division/modulo by zero throw std::runtime_error from operator/ and
  // operator%, for the compound forms too.
  Int128& operator/=(const Int128& other) {
    *this = *this / other;
    return *this;
  }

  Int128& operator%=(const Int128& other) {
    *this = *this % other;
    return *this;
  }

  constexpr Int128& operator*=(const Int128& other) {
    *this = *this * other;
    return *this;
  }

  // Increment/decrement operators
  Int128& operator++() {
    *this = *this + Int128(1);
    return *this;
  }

  Int128 operator++(int) {
    Int128 temp = *this;
    ++(*this);
    return temp;
  }

  Int128& operator--() {
    *this = *this - Int128(1);
    return *this;
  }

  Int128 operator--(int) {
    Int128 temp = *this;
    --(*this);
    return temp;
  }

  // Additional assignment operators for built-in types
  constexpr Int128& operator*=(int64_t value) {
    return *this *= Int128(value);
  }

  constexpr Int128& operator*=(int32_t value) {
    return *this *= Int128(value);
  }

  // Match unsigned operands without narrowing to a signed scalar overload.
  template <
      typename T,
      std::enable_if_t<detail::kIsIntegralOperand<T>, int> = 0>
  constexpr Int128& operator*=(T value) {
    return *this *= Int128(value);
  }

  // Specialized signed 128/64 division. Constant divisors allow MSVC to fold
  // the sign handling and use reciprocal multiplication where possible.
  //
  // INT128_MIN / -1 wraps to INT128_MIN. Native signed integer division does
  // not define this case.
  FOLLY_ALWAYS_INLINE Int128 divideByInt64(int64_t other) const {
    // Each sign is a zero/all-ones mask: |x| = (x + s) ^ s across both limbs.
    // Re-sign the quotient the same way to keep both limbs in registers.
    //
    // Force-inlined: out of line, MSVC x64 returns the 16-byte result through
    // a hidden pointer and constant divisors (`x / 10`) lose their
    // magic-multiply lowering.
    const uint64_t dividendSign = static_cast<uint64_t>(high_ >> 63);
    const uint64_t divisorSign = static_cast<uint64_t>(other >> 63);
    uint64_t numeratorLow;
    uint64_t numeratorHigh;
    detail::negateIfMask(
        dividendSign,
        low_,
        static_cast<uint64_t>(high_),
        &numeratorLow,
        &numeratorHigh);
    // |other| as uint64 (well-defined on INT64_MIN: yields 2^63).
    const uint64_t denominator =
        (static_cast<uint64_t>(other) ^ divisorSign) - divisorSign;
    if (denominator == 0) {
      detail::throwDivideByZero("Division by zero");
    }
    uint64_t quotientLow;
    uint64_t quotientHigh;
    uint64_t remainder;
    if (numeratorHigh == 0) {
      // Single 64/64 divide.
      quotientHigh = 0;
      quotientLow = numeratorLow / denominator;
    } else if (numeratorHigh < denominator) {
      // Quotient fits in 64 bits.
      quotientHigh = 0;
      quotientLow = detail::udiv128By64(
          numeratorHigh, numeratorLow, denominator, &remainder);
    } else {
      // Two-step: high quotient, then carry the remainder into the low divide.
      uint64_t highRemainder;
      quotientHigh =
          detail::udivrem64(numeratorHigh, denominator, &highRemainder);
      quotientLow = detail::udiv128By64(
          highRemainder, numeratorLow, denominator, &remainder);
    }
    uint64_t resultLow;
    uint64_t resultHigh;
    detail::negateIfMask(
        dividendSign ^ divisorSign,
        quotientLow,
        quotientHigh,
        &resultLow,
        &resultHigh);
    return Int128(static_cast<int64_t>(resultHigh), resultLow);
  }

  FOLLY_ALWAYS_INLINE Int128 modByInt64(int64_t other) const {
    // Result sign of `%` follows the dividend. Branch-free sign handling; see
    // divideByInt64.
    const uint64_t dividendSign = static_cast<uint64_t>(high_ >> 63);
    const uint64_t divisorSign = static_cast<uint64_t>(other >> 63);
    uint64_t numeratorLow;
    uint64_t numeratorHigh;
    detail::negateIfMask(
        dividendSign,
        low_,
        static_cast<uint64_t>(high_),
        &numeratorLow,
        &numeratorHigh);
    const uint64_t denominator =
        (static_cast<uint64_t>(other) ^ divisorSign) - divisorSign;
    if (denominator == 0) {
      detail::throwDivideByZero("Modulo by zero");
    }
    uint64_t remainder;
    if (numeratorHigh == 0) {
      remainder = numeratorLow % denominator;
    } else {
      const uint64_t highRemainder = numeratorHigh < denominator
          ? numeratorHigh
          : (numeratorHigh % denominator);
      (void)detail::udiv128By64(
          highRemainder, numeratorLow, denominator, &remainder);
    }
    // Remainder fits in uint64 (remainder < denominator <= 2^63).
    uint64_t resultLow;
    uint64_t resultHigh;
    detail::negateIfMask(dividendSign, remainder, 0, &resultLow, &resultHigh);
    return Int128(static_cast<int64_t>(resultHigh), resultLow);
  }

  // Compile-time-constant divisor. When the divisor is a known non-zero
  // constant, the compiler folds the sign work, picks the right branch of
  // `divideByInt64`, and (for divisors < 2^31) can sometimes lower to
  // multiplicative-inverse code. Use as `x.divideByConstant<1000000>()`.
  template <int64_t divisor>
  FOLLY_ALWAYS_INLINE Int128 divideByConstant() const {
    static_assert(divisor != 0, "Divisor must be non-zero");
    return divideByInt64(divisor);
  }

  template <int64_t divisor>
  FOLLY_ALWAYS_INLINE Int128 modByConstant() const {
    static_assert(divisor != 0, "Divisor must be non-zero");
    return modByInt64(divisor);
  }

  // Built-in-type division overloads. Route through the specialized 128/64
  // path instead of constructing a full Int128 divisor and going through the
  // general 128/128 algorithm. Force-inlined like divideByInt64 so a literal
  // divisor reaches it as a constant.
  FOLLY_ALWAYS_INLINE Int128 operator/(int64_t other) const {
    return divideByInt64(other);
  }
  FOLLY_ALWAYS_INLINE Int128 operator/(int32_t other) const {
    return divideByInt64(other);
  }
  FOLLY_ALWAYS_INLINE Int128 operator/(long other) const {
    return divideByInt64(static_cast<int64_t>(other));
  }

  // Additional multiplication operators for compatibility with int64_t
  constexpr Int128 operator*(int64_t other) const {
    return *this * Int128(other);
  }

  friend constexpr Int128 operator*(int64_t left, const Int128& right) {
    return Int128(left) * right;
  }

  // Multiplication: full 128x128 -> low-128 product.
  //   (a_hi:a_lo) * (b_hi:b_lo) mod 2^128
  //     = a_lo*b_lo + ((a_lo*b_hi + a_hi*b_lo) << 64)
  // Use `_umul128` (x64) / `__umulh` (ARM64) for the low*low full 128-bit
  // product, then add the two cross terms (low 64 bits only). 3 muls + 2 adds
  // total. The `is_constant_evaluated` branch keeps a constexpr-callable
  // fallback since the intrinsics are not constexpr. Single-return shape; see
  // operator+.
  constexpr Int128 operator*(const Int128& other) const {
    uint64_t resultLow;
    uint64_t productHigh;
#if defined(_M_X64) || defined(_M_ARM64)
    if (!std::is_constant_evaluated()) {
      resultLow = detail::umul128(low_, other.low_, &productHigh);
    } else
#endif
    {
      // Portable / constant-evaluated path: 32-bit-halves construction of the
      // high 64 bits of low_ * other.low_.
      const uint64_t lowLow = low_ & 0xFFFFFFFF;
      const uint64_t lowHigh = low_ >> 32;
      const uint64_t otherLowLow = other.low_ & 0xFFFFFFFF;
      const uint64_t otherLowHigh = other.low_ >> 32;
      const uint64_t lowProduct = lowLow * otherLowLow;
      const uint64_t leftCrossProduct = lowLow * otherLowHigh;
      const uint64_t rightCrossProduct = lowHigh * otherLowLow;
      const uint64_t highProduct = lowHigh * otherLowHigh;
      const uint64_t carry =
          ((lowProduct >> 32) + (leftCrossProduct & 0xFFFFFFFF) +
           (rightCrossProduct & 0xFFFFFFFF)) >>
          32;
      resultLow = low_ * other.low_;
      productHigh = highProduct + (leftCrossProduct >> 32) +
          (rightCrossProduct >> 32) + carry;
    }
    const uint64_t resultHigh = productHigh +
        low_ * static_cast<uint64_t>(other.high_) +
        static_cast<uint64_t>(high_) * other.low_;
    return Int128(static_cast<int64_t>(resultHigh), resultLow);
  }

  // Helper: unsigned 128-bit division returning quotient and remainder.
  // Operates on absolute (unsigned) values represented as (high, low).
  static void udivmod128(
      uint64_t dividendHigh,
      uint64_t dividendLow,
      uint64_t divisorHigh,
      uint64_t divisorLow,
      uint64_t& quotientHigh,
      uint64_t& quotientLow,
      uint64_t& remainderHigh,
      uint64_t& remainderLow) {
    quotientHigh = quotientLow = remainderHigh = remainderLow = 0;
    if ((divisorHigh | divisorLow) == 0) {
      detail::throwDivideByZero("Division by zero");
    }
    // If dividend < divisor, quotient is 0, remainder is dividend.
    if (dividendHigh < divisorHigh ||
        (dividendHigh == divisorHigh && dividendLow < divisorLow)) {
      remainderHigh = dividendHigh;
      remainderLow = dividendLow;
      return;
    }
    // If divisor fits in 64 bits, use optimized path.
    if (divisorHigh == 0) {
      if (dividendHigh == 0) {
        // Both fit in 64 bits.
        quotientLow = detail::udivrem64(dividendLow, divisorLow, &remainderLow);
        return;
      }
      // 128-bit by 64-bit division. udiv128By64 requires high < divisor so
      // the quotient fits in 64 bits: divide the high limb first and carry its
      // remainder into the low divide.
      uint64_t highRemainder;
      quotientHigh =
          detail::udivrem64(dividendHigh, divisorLow, &highRemainder);
      quotientLow = detail::udiv128By64(
          highRemainder, dividendLow, divisorLow, &remainderLow);
      return;
    }
    // General case: 128/128 unsigned divide with divisorHigh != 0, so the
    // quotient fits in 64 bits.
    //
    // Knuth Algorithm D (TAOCP vol 2, 4.3.1) at base 2^64: a normalized
    // 2-digit-by-2-digit divide with at most 2 qhat corrections, built on the
    // detail:: 128/64 divide, 64x64 multiply and carry/borrow primitives.
    //
    // Normalize: shift both operands left until divisor's high bit is set.
    // This bounds the qhat estimate to be off by at most 2 (Knuth thm T).
    const int d = std::countl_zero(divisorHigh);
    uint64_t denHi = divisorHigh;
    uint64_t denLo = divisorLow;
    uint64_t numHi = dividendHigh;
    uint64_t numLo = dividendLow;
    uint64_t numTop = 0; // 3rd "digit" produced by shifting num left by d
    if (d != 0) {
      denHi = (denHi << d) | (denLo >> (64 - d));
      denLo <<= d;
      numTop = numHi >> (64 - d);
      numHi = (numHi << d) | (numLo >> (64 - d));
      numLo <<= d;
    }
    // Estimate qhat = (numTop:numHi) / denHi. After normalization
    // numTop < 2^d <= 2^63 <= denHi, so the udiv128By64 precondition holds
    // and qhat fits in 64 bits.
    uint64_t rhat;
    uint64_t qhat = detail::udiv128By64(numTop, numHi, denHi, &rhat);
    // Correction loop: decrement qhat while qhat * denLo > (rhat:numLo).
    // Bounded at <= 2 iterations.
    for (;;) {
      uint64_t prodHi;
      const uint64_t prodLo = detail::umul128(qhat, denLo, &prodHi);
      // Algorithm D step D3: continue correcting while
      //   qhat * denLo > b * rhat + u[j+n-2]
      // For our 2-by-2 divide (n=2, j=0) the digit u[j+n-2] is u[0] = numLo
      // (the LOW digit of the normalized numerator), not numHi. Using numHi
      // here could leave qhat off by more than the post-multiply
      // borrow-correction can recover from for narrow adversarial inputs
      // (prodHi == rhat && numLo < prodLo <= numHi).
      if (prodHi < rhat || (prodHi == rhat && prodLo <= numLo)) {
        break;
      }
      --qhat;
      const uint64_t sum = rhat + denHi;
      if (rhat > sum) {
        // rhat += denHi overflowed -> any further qhat*denLo can never
        // exceed (rhat:numLo); stop correcting.
        break;
      }
      rhat = sum;
    }
    // Subtract qhat * den from (numTop:numHi:numLo). This is a 3-digit
    // minus 1-digit-times-2-digit subtract; never underflows due to the
    // correction loop above (modulo the one borrow we may need to undo).
    // Both multiplies and the product's carry are formed before the borrow
    // chain: `mul` clobbers the flags, so interleaving them forces MSVC to
    // spill the borrow between steps.
    uint64_t prod0Hi;
    const uint64_t prod0Lo = detail::umul128(qhat, denLo, &prod0Hi);
    uint64_t prod1Hi;
    uint64_t prod1Lo = detail::umul128(qhat, denHi, &prod1Hi);
    prod1Hi += detail::addCarry64(0, prod1Lo, prod0Hi, &prod1Lo);
    unsigned char borrow = detail::subBorrow64(0, numLo, prod0Lo, &numLo);
    borrow = detail::subBorrow64(borrow, numHi, prod1Lo, &numHi);
    borrow = detail::subBorrow64(borrow, numTop, prod1Hi, &numTop);
    // If borrow remains, qhat was 1 too large; correct by adding den back.
    if (borrow) {
      --qhat;
      const unsigned char carry = detail::addCarry64(0, numLo, denLo, &numLo);
      detail::addCarry64(carry, numHi, denHi, &numHi);
    }
    // Un-normalize the remainder (right-shift by d bits).
    if (d != 0) {
      numLo = (numLo >> d) | (numHi << (64 - d));
      numHi >>= d;
    }
    quotientHigh = 0;
    quotientLow = qhat;
    remainderHigh = numHi;
    remainderLow = numLo;
  }

  // Division
  Int128 operator/(const Int128& other) const {
    // Fast path: when the divisor fits in a signed 64-bit integer -- true for
    // every power of ten up to 10^18, i.e. the common decimal-rescale factor --
    // route to the single-call 128/64 divide instead of the full 128/128 Knuth
    // algorithm.
    // A zero divisor takes this path too, and divideByInt64 throws.
    if (other.high_ == (static_cast<int64_t>(other.low_) >> 63)) {
      return divideByInt64(static_cast<int64_t>(other.low_));
    }
    // Branch-free sign handling; see divideByInt64.
    const uint64_t dividendSign = static_cast<uint64_t>(high_ >> 63);
    const uint64_t divisorSign = static_cast<uint64_t>(other.high_ >> 63);
    uint64_t dividendLow;
    uint64_t dividendHigh;
    uint64_t divisorLow;
    uint64_t divisorHigh;
    detail::negateIfMask(
        dividendSign,
        low_,
        static_cast<uint64_t>(high_),
        &dividendLow,
        &dividendHigh);
    detail::negateIfMask(
        divisorSign,
        other.low_,
        static_cast<uint64_t>(other.high_),
        &divisorLow,
        &divisorHigh);
    uint64_t quotientHigh;
    uint64_t quotientLow;
    uint64_t remainderHigh;
    uint64_t remainderLow;
    udivmod128(
        dividendHigh,
        dividendLow,
        divisorHigh,
        divisorLow,
        quotientHigh,
        quotientLow,
        remainderHigh,
        remainderLow);
    uint64_t resultLow;
    uint64_t resultHigh;
    detail::negateIfMask(
        dividendSign ^ divisorSign,
        quotientLow,
        quotientHigh,
        &resultLow,
        &resultHigh);
    return Int128(static_cast<int64_t>(resultHigh), resultLow);
  }

  // Modulo operator
  Int128 operator%(const Int128& other) const {
    // Fast path: divisor fits in a signed 64-bit integer (see operator/). A
    // zero divisor takes this path too, and modByInt64 throws.
    if (other.high_ == (static_cast<int64_t>(other.low_) >> 63)) {
      return modByInt64(static_cast<int64_t>(other.low_));
    }
    // Branch-free sign handling; the remainder takes the dividend's sign.
    const uint64_t dividendSign = static_cast<uint64_t>(high_ >> 63);
    const uint64_t divisorSign = static_cast<uint64_t>(other.high_ >> 63);
    uint64_t dividendLow;
    uint64_t dividendHigh;
    uint64_t divisorLow;
    uint64_t divisorHigh;
    detail::negateIfMask(
        dividendSign,
        low_,
        static_cast<uint64_t>(high_),
        &dividendLow,
        &dividendHigh);
    detail::negateIfMask(
        divisorSign,
        other.low_,
        static_cast<uint64_t>(other.high_),
        &divisorLow,
        &divisorHigh);
    uint64_t quotientHigh;
    uint64_t quotientLow;
    uint64_t remainderHigh;
    uint64_t remainderLow;
    udivmod128(
        dividendHigh,
        dividendLow,
        divisorHigh,
        divisorLow,
        quotientHigh,
        quotientLow,
        remainderHigh,
        remainderLow);
    uint64_t resultLow;
    uint64_t resultHigh;
    detail::negateIfMask(
        dividendSign, remainderLow, remainderHigh, &resultLow, &resultHigh);
    return Int128(static_cast<int64_t>(resultHigh), resultLow);
  }

  // Comparison operators. Bitwise `|` of the limb differences: one
  // `xor; xor; or; sete` with no branch. `a && b` compiles to two
  // data-dependent branches on MSVC.
  constexpr bool operator==(const Int128& other) const {
    return ((static_cast<uint64_t>(high_) ^
             static_cast<uint64_t>(other.high_)) |
            (low_ ^ other.low_)) == 0;
  }

  constexpr bool operator!=(const Int128& other) const {
    return !(*this == other);
  }

  constexpr bool operator<(const Int128& other) const {
#if defined(_M_X64)
    if (!std::is_constant_evaluated()) {
      // Flipping the sign bit of both high limbs maps signed order onto
      // unsigned order, so `a < b` is the borrow out of the 128-bit unsigned
      // subtraction: `sub; sbb; setb` (or `jb` / `cmovb` when the result
      // feeds a branch or select) -- clang's native __int128 lowering. The
      // biased limbs are formed before the borrow chain so MSVC keeps CF
      // live from `sub` to `sbb`; biasing in between makes it spill CF to a
      // register and rebuild it (`setb; add r8b,-1`).
      constexpr uint64_t kSignBit = 0x8000000000000000ULL;
      const uint64_t biasedHigh = static_cast<uint64_t>(high_) ^ kSignBit;
      const uint64_t otherBiasedHigh =
          static_cast<uint64_t>(other.high_) ^ kSignBit;
      unsigned long long ignored;
      return _subborrow_u64(
          _subborrow_u64(0, low_, other.low_, &ignored),
          biasedHigh,
          otherBiasedHigh,
          &ignored);
    }
#endif
    // Branchless select between the unsigned low-limb compare (high limbs
    // equal) and the signed high-limb compare: MSVC ARM64 lowers it to
    // `cmp; cset; cmp; cset; csel`, one instruction shorter than the
    // `(hiLt) | (hiEq & loLt)` form in streaming, select and sort contexts.
    // The branchy form (`if (high_ != other.high_) ...`) mispredicts when
    // high-limb equality is data dependent (mixed-sign small decimals,
    // clustered sort keys).
    const bool highLess = high_ < other.high_;
    const bool lowLess = low_ < other.low_;
    return high_ == other.high_ ? lowLess : highLess;
  }

  // `a < b` in the `(hiLt) | (hiEq & loLt)` form. Used when b is a
  // primitive integer: for a literal 0 MSVC folds it to a sign-bit test
  // (`shr 63` / `lsr #63`), which neither form in operator< folds to. The
  // `x > v` / `x <= v` directions stay on operator<, which is cheaper for
  // those even when v is 0.
  static constexpr bool lessThanIntegralOperand(
      const Int128& a,
      const Int128& b) {
    return (a.high_ < b.high_) | ((a.high_ == b.high_) & (a.low_ < b.low_));
  }

  constexpr bool operator<=(const Int128& other) const {
    return !(other < *this);
  }

  constexpr bool operator>(const Int128& other) const {
    return other < *this;
  }

  constexpr bool operator>=(const Int128& other) const {
    return !(*this < other);
  }

  // Bitwise operators
  constexpr Int128 operator&(const Int128& other) const {
    return Int128(high_ & other.high_, low_ & other.low_);
  }

  constexpr Int128 operator|(const Int128& other) const {
    return Int128(high_ | other.high_, low_ | other.low_);
  }

  constexpr Int128 operator^(const Int128& other) const {
    return Int128(high_ ^ other.high_, low_ ^ other.low_);
  }

  constexpr Int128 operator~() const {
    return Int128(~high_, ~low_);
  }

  // Shift operators
  //
  // Branchless on x64 and ARM64: `shld`/`shl` (x64) or `lsl`/`lsr` (ARM64)
  // plus one `cmov`/`csel` per limb keyed on `shift >= 64`, then the
  // `shift>=128 -> 0` guard (native __int128 leaves it undefined). Below 128
  // `shift >= 64` is bit 6 of the count, so both limb selects reuse a single
  // compare (a separate `shift & 64` test costs MSVC `shr; test` on top). Each
  // select tests exactly one condition. MSVC 14.38 turns `zero || big` and
  // nested `a ? b : (c ? d : e)` selects into data-dependent branches that
  // mispredict on variable shift counts; flat single-condition selects
  // if-convert on both targets. For a compile-time-constant shift the whole
  // thing folds to the same one or two instructions dedicated branches would
  // emit (`x << 3` -> `shld;lea`). Single-return shape; see operator+.
  //
  // The portable (constexpr / non-x64) funnel shift is written exactly once;
  // the constant-evaluated path and the ARM64 runtime path share it.
  // `low_ >> (64 - s)` is written `(low_ >> 1) >> (63 - s)` (here
  // `(~s) & 63`) to avoid the undefined 64-bit shift when s == 0. low_ is
  // shifted as unsigned (an int64 cast would sign-corrupt the high half for
  // any value with low_ bit63 set).
  template <
      typename T,
      std::enable_if_t<detail::kIsIntegralOperand<T>, int> = 0>
  constexpr Int128 operator<<(T shift) const {
    const auto unsignedShift = static_cast<std::make_unsigned_t<T>>(shift);
    const unsigned shiftValue = static_cast<unsigned>(unsignedShift);
    const unsigned s = shiftValue & 63u;
    const uint64_t shiftedLow = low_ << s;
    uint64_t shiftedHigh;
#if defined(_M_X64)
    if (!std::is_constant_evaluated()) {
      shiftedHigh = __shiftleft128(
          low_, static_cast<uint64_t>(high_), static_cast<unsigned char>(s));
    } else
#endif
    {
      shiftedHigh =
          (static_cast<uint64_t>(high_) << s) | ((low_ >> 1) >> ((~s) & 63u));
    }
    const bool atLeast64 = unsignedShift >= 64;
    const bool zero = unsignedShift >= 128;
    uint64_t newHigh = atLeast64 ? shiftedLow : shiftedHigh;
    newHigh = zero ? 0ull : newHigh;
    const uint64_t newLow = atLeast64 ? 0ull : shiftedLow;
    return Int128(static_cast<int64_t>(newHigh), newLow);
  }

  // Arithmetic (sign-propagating) right shift. Branchless on x64 and ARM64:
  // `shrd` + `sar` (x64) / `lsr` + `asr` (ARM64) plus single-condition
  // `cmov`/`csel`s keyed on `shift >= 64` and the `shift>=128 -> sign
  // fill` guard (native __int128 leaves it UB), mirroring clang. The portable
  // (ARM64 / constexpr) funnel shift writes `high_ << (64 - s)` as
  // `(high_ << 1) << (63 - s)` to avoid the undefined 64-bit shift when
  // s == 0. Results for >=64 fill the high limb with the sign.
  template <
      typename T,
      std::enable_if_t<detail::kIsIntegralOperand<T>, int> = 0>
  constexpr Int128 operator>>(T shift) const {
    const uint64_t sign = static_cast<uint64_t>(high_ >> 63); // 0 or ~0
    const auto unsignedShift = static_cast<std::make_unsigned_t<T>>(shift);
    const unsigned shiftValue = static_cast<unsigned>(unsignedShift);
    const unsigned s = shiftValue & 63u;
    uint64_t shiftedLow;
#if defined(_M_X64)
    if (!std::is_constant_evaluated()) {
      shiftedLow = __shiftright128(
          low_, static_cast<uint64_t>(high_), static_cast<unsigned char>(s));
    } else
#endif
    {
      shiftedLow =
          (low_ >> s) | ((static_cast<uint64_t>(high_) << 1) << ((~s) & 63u));
    }
    const uint64_t arithmeticHigh = static_cast<uint64_t>(high_ >> s);
    const bool atLeast64 = unsignedShift >= 64;
    const bool all = unsignedShift >= 128;
    uint64_t newLow = atLeast64 ? arithmeticHigh : shiftedLow;
    newLow = all ? sign : newLow;
    const uint64_t newHigh = atLeast64 ? sign : arithmeticHigh;
    return Int128(static_cast<int64_t>(newHigh), newLow);
  }

  // Compound assignment shift operators
  template <
      typename T,
      std::enable_if_t<detail::kIsIntegralOperand<T>, int> = 0>
  Int128& operator<<=(T shift) {
    *this = *this << shift;
    return *this;
  }

  template <
      typename T,
      std::enable_if_t<detail::kIsIntegralOperand<T>, int> = 0>
  Int128& operator>>=(T shift) {
    *this = *this >> shift;
    return *this;
  }

  // Compound assignment bitwise operators
  Int128& operator&=(const Int128& other) {
    high_ &= other.high_;
    low_ &= other.low_;
    return *this;
  }

  Int128& operator|=(const Int128& other) {
    high_ |= other.high_;
    low_ |= other.low_;
    return *this;
  }

  Int128& operator^=(const Int128& other) {
    high_ ^= other.high_;
    low_ ^= other.low_;
    return *this;
  }

  std::to_chars_result toChars(char* first, char* last) const {
    const bool negative = high_ < 0;

    // Compute the magnitude in unsigned limbs, including 2^127 for INT128_MIN.
    // Like operator-(), this uses defined two's-complement arithmetic.
    uint64_t hi;
    uint64_t lo;
    if (!negative) {
      hi = static_cast<uint64_t>(high_);
      lo = low_;
    } else {
      lo = ~low_ + 1;
      hi = ~static_cast<uint64_t>(high_) + (lo == 0 ? 1 : 0);
    }

    return detail::uint128ToChars(first, last, hi, lo, negative);
  }

  // String conversion
  std::string toString() const {
    char buffer[40];
    const auto result = toChars(buffer, buffer + sizeof(buffer));
    return std::string(buffer, result.ptr);
  }

 private:
  template <typename T>
  T toFloatingPoint() const {
    const bool negative = high_ < 0;
    const uint64_t magnitudeLow = negative ? uint64_t{0} - low_ : low_;
    const uint64_t magnitudeHigh = negative
        ? ~static_cast<uint64_t>(high_) + (magnitudeLow == 0 ? 1 : 0)
        : static_cast<uint64_t>(high_);
    return detail::int128ToFloatingPoint<T>(
        magnitudeHigh, magnitudeLow, negative);
  }

  uint64_t low_; // Least significant 64 bits (offset 0, matches __int128_t)
  int64_t high_; // Most significant 64 bits (offset 8, matches __int128_t)
};

namespace detail {

inline std::to_chars_result uint128ToChars(
    char* first,
    char* last,
    uint64_t high,
    uint64_t low,
    bool negative) {
  // Extract 19 decimal digits per 128/64 division. A 128-bit value needs at
  // most three chunks, avoiding one full 128-bit division per digit.
  constexpr uint64_t kPow19 = 10'000'000'000'000'000'000ULL;
  char buffer[40];
  int position = static_cast<int>(sizeof(buffer));
  if (high == 0 && low == 0) {
    buffer[--position] = '0';
  }
  while (high != 0 || low != 0) {
    uint64_t quotientHigh;
    uint64_t quotientLow;
    uint64_t remainderHigh;
    uint64_t remainderLow;
    Int128::udivmod128(
        high,
        low,
        0,
        kPow19,
        quotientHigh,
        quotientLow,
        remainderHigh,
        remainderLow);
    uint64_t chunk = remainderLow;
    high = quotientHigh;
    low = quotientLow;
    if (high != 0 || low != 0) {
      for (int i = 0; i < 19; ++i) {
        buffer[--position] = static_cast<char>('0' + (chunk % 10));
        chunk /= 10;
      }
    } else {
      while (chunk != 0) {
        buffer[--position] = static_cast<char>('0' + (chunk % 10));
        chunk /= 10;
      }
    }
  }
  if (negative) {
    buffer[--position] = '-';
  }

  const size_t length = sizeof(buffer) - position;
  if (static_cast<size_t>(last - first) < length) {
    return {last, std::errc::value_too_large};
  }
  for (size_t i = 0; i < length; ++i) {
    first[i] = buffer[position + i];
  }
  return {first + length, std::errc{}};
}

} // namespace detail

class UInt128 {
 public:
  constexpr UInt128() : low_(0), high_(0) {}

  constexpr UInt128(uint64_t high, uint64_t low) : low_(low), high_(high) {}

  explicit constexpr UInt128(bool value) : low_(value), high_(0) {}

  // Truncates toward zero; negative inputs yield 0 and values >= 2^128, +inf
  // and NaN saturate (see detail::floatingPointTo128).
  template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
  explicit UInt128(T value) {
    const auto [high, low] = detail::floatingPointTo128<false>(value);
    low_ = low;
    high_ = high;
  }

  template <
      typename T,
      std::enable_if_t<detail::kIsIntegralOperand<T>, int> = 0>
  constexpr UInt128(T value)
      : low_(static_cast<uint64_t>(value)),
        high_(std::is_signed_v<T> && value < 0 ? UINT64_MAX : 0) {}

  constexpr UInt128(const UInt128& other) = default;
  constexpr UInt128& operator=(const UInt128& other) = default;

  // Conversion constructor from Int128
  constexpr UInt128(const Int128& value)
      : low_(value.low()), high_(static_cast<uint64_t>(value.high())) {}

  // Getters
  constexpr uint64_t high() const {
    return high_;
  }
  constexpr uint64_t low() const {
    return low_;
  }
  constexpr uint64_t hi() const {
    return high_;
  }
  constexpr uint64_t lo() const {
    return low_;
  }

  // Conversion operators
  // `(high_ | low_)` is non-zero iff the 128-bit value is non-zero. The bit-or
  // is branch-free (`or; setne`) on MSVC.
  explicit constexpr operator bool() const {
    return (high_ | low_) != 0;
  }
  template <
      typename T,
      std::enable_if_t<detail::kIsIntegralOperand<T>, int> = 0>
  constexpr operator T() const {
    return static_cast<T>(low_);
  }
  explicit operator float() const {
    return detail::int128ToFloatingPoint<float>(high_, low_, false);
  }
  explicit operator double() const {
    return detail::int128ToFloatingPoint<double>(high_, low_, false);
  }
  explicit operator long double() const {
    return static_cast<double>(*this);
  }

  constexpr UInt128 operator+() const {
    return *this;
  }

  constexpr UInt128 operator-() const {
    return UInt128(0, 0) - *this;
  }

  // Arithmetic operators. See Int128 for rationale, including the
  // single-return shape.
  constexpr UInt128 operator+(const UInt128& other) const {
    uint64_t newLow;
    uint64_t newHigh;
#if defined(_M_X64)
    if (!std::is_constant_evaluated()) {
      const unsigned char carry = _addcarry_u64(0, low_, other.low_, &newLow);
      _addcarry_u64(carry, high_, other.high_, &newHigh);
    } else
#endif
    {
      // Portable / constant-evaluated path (also the ARM64 runtime path).
      newLow = low_ + other.low_;
      newHigh = high_ + other.high_ + (newLow < low_ ? 1 : 0);
    }
    return UInt128(newHigh, newLow);
  }

  constexpr UInt128 operator-(const UInt128& other) const {
    uint64_t newLow;
    uint64_t newHigh;
#if defined(_M_X64)
    if (!std::is_constant_evaluated()) {
      const unsigned char borrow = _subborrow_u64(0, low_, other.low_, &newLow);
      _subborrow_u64(borrow, high_, other.high_, &newHigh);
    } else
#endif
    {
      // Portable / constant-evaluated path (also the ARM64 runtime path).
      newLow = low_ - other.low_;
      newHigh = high_ - other.high_ - (newLow > low_ ? 1 : 0);
    }
    return UInt128(newHigh, newLow);
  }

#if defined(_M_X64)
  // See Int128::addOverflowX64: `add; adc; setb` / `sub; sbb; setb`.
  static bool
  addOverflowX64(const UInt128& a, const UInt128& b, UInt128* result) {
    const uint64_t aHigh = a.high_;
    const uint64_t bHigh = b.high_;
    const unsigned char carry = _addcarry_u64(0, a.low_, b.low_, &result->low_);
    return _addcarry_u64(carry, aHigh, bHigh, &result->high_);
  }

  static bool
  subOverflowX64(const UInt128& a, const UInt128& b, UInt128* result) {
    const uint64_t aHigh = a.high_;
    const uint64_t bHigh = b.high_;
    const unsigned char borrow =
        _subborrow_u64(0, a.low_, b.low_, &result->low_);
    return _subborrow_u64(borrow, aHigh, bHigh, &result->high_);
  }
#endif

  UInt128& operator+=(const UInt128& other) {
    *this = *this + other;
    return *this;
  }

  UInt128& operator-=(const UInt128& other) {
    *this = *this - other;
    return *this;
  }

  // Multiplication: see Int128::operator* for rationale.
  constexpr UInt128 operator*(const UInt128& other) const {
    uint64_t resultLow;
    uint64_t productHigh;
#if defined(_M_X64) || defined(_M_ARM64)
    if (!std::is_constant_evaluated()) {
      resultLow = detail::umul128(low_, other.low_, &productHigh);
    } else
#endif
    {
      // Portable / constant-evaluated path: 32-bit halves. See
      // Int128::operator*.
      const uint64_t lowLow = low_ & 0xFFFFFFFF;
      const uint64_t lowHigh = low_ >> 32;
      const uint64_t otherLowLow = other.low_ & 0xFFFFFFFF;
      const uint64_t otherLowHigh = other.low_ >> 32;
      const uint64_t lowProduct = lowLow * otherLowLow;
      const uint64_t leftCrossProduct = lowLow * otherLowHigh;
      const uint64_t rightCrossProduct = lowHigh * otherLowLow;
      const uint64_t highProduct = lowHigh * otherLowHigh;
      const uint64_t carry =
          ((lowProduct >> 32) + (leftCrossProduct & 0xFFFFFFFF) +
           (rightCrossProduct & 0xFFFFFFFF)) >>
          32;
      resultLow = low_ * other.low_;
      productHigh = highProduct + (leftCrossProduct >> 32) +
          (rightCrossProduct >> 32) + carry;
    }
    const uint64_t resultHigh =
        productHigh + low_ * other.high_ + high_ * other.low_;
    return UInt128(resultHigh, resultLow);
  }

  constexpr UInt128 operator*(int64_t other) const {
    return *this * UInt128(other);
  }

  friend constexpr UInt128 operator*(int64_t left, const UInt128& right) {
    return UInt128(left) * right;
  }

  constexpr UInt128& operator*=(const UInt128& other) {
    *this = *this * other;
    return *this;
  }

  // Division/modulo by zero throw std::runtime_error (from udivmod128), like
  // Int128, for the compound forms too.
  UInt128 operator/(const UInt128& other) const {
    uint64_t quotientHigh;
    uint64_t quotientLow;
    uint64_t remainderHigh;
    uint64_t remainderLow;
    Int128::udivmod128(
        high_,
        low_,
        other.high_,
        other.low_,
        quotientHigh,
        quotientLow,
        remainderHigh,
        remainderLow);
    return UInt128(quotientHigh, quotientLow);
  }

  UInt128& operator/=(const UInt128& other) {
    *this = *this / other;
    return *this;
  }

  // Modulo operator
  UInt128 operator%(const UInt128& other) const {
    uint64_t quotientHigh;
    uint64_t quotientLow;
    uint64_t remainderHigh;
    uint64_t remainderLow;
    Int128::udivmod128(
        high_,
        low_,
        other.high_,
        other.low_,
        quotientHigh,
        quotientLow,
        remainderHigh,
        remainderLow);
    return UInt128(remainderHigh, remainderLow);
  }

  UInt128& operator%=(const UInt128& other) {
    *this = *this % other;
    return *this;
  }

  // Increment/decrement operators
  UInt128& operator++() {
    *this = *this + UInt128(0, 1);
    return *this;
  }

  UInt128 operator++(int) {
    UInt128 temp = *this;
    ++(*this);
    return temp;
  }

  UInt128& operator--() {
    *this = *this - UInt128(0, 1);
    return *this;
  }

  UInt128 operator--(int) {
    UInt128 temp = *this;
    --(*this);
    return temp;
  }

  // Comparison operators. Branch-free; see Int128::operator==.
  constexpr bool operator==(const UInt128& other) const {
    return ((high_ ^ other.high_) | (low_ ^ other.low_)) == 0;
  }

  constexpr bool operator!=(const UInt128& other) const {
    return !(*this == other);
  }

  constexpr bool operator<(const UInt128& other) const {
    // Branchless; see Int128::operator< for rationale. On x64 the borrow out
    // of the full 128-bit subtraction is exactly `a < b`: `sub; sbb; setb`,
    // identical to clang's native unsigned __int128 lowering. Elsewhere (and
    // in constant evaluation) select between the limb compares, which MSVC
    // ARM64 lowers to `cmp; cset; cmp; cset; csel`.
#if defined(_M_X64)
    if (!std::is_constant_evaluated()) {
      unsigned long long ignored;
      return _subborrow_u64(
                 _subborrow_u64(0, low_, other.low_, &ignored),
                 high_,
                 other.high_,
                 &ignored) != 0;
    }
#endif
    const bool highLess = high_ < other.high_;
    const bool lowLess = low_ < other.low_;
    return high_ == other.high_ ? lowLess : highLess;
  }

  constexpr bool operator<=(const UInt128& other) const {
    return !(other < *this);
  }

  constexpr bool operator>(const UInt128& other) const {
    return other < *this;
  }

  constexpr bool operator>=(const UInt128& other) const {
    return !(*this < other);
  }

  // Bitwise operators
  constexpr UInt128 operator&(const UInt128& other) const {
    return UInt128(high_ & other.high_, low_ & other.low_);
  }

  constexpr UInt128 operator|(const UInt128& other) const {
    return UInt128(high_ | other.high_, low_ | other.low_);
  }

  constexpr UInt128 operator^(const UInt128& other) const {
    return UInt128(high_ ^ other.high_, low_ ^ other.low_);
  }

  constexpr UInt128 operator~() const {
    return UInt128(~high_, ~low_);
  }

  // Shift operators. Branchless single-return shape; see Int128::operator<<.
  // The portable fallback is written once and shared by the constexpr and
  // ARM64 paths.
  template <
      typename T,
      std::enable_if_t<detail::kIsIntegralOperand<T>, int> = 0>
  constexpr UInt128 operator<<(T shift) const {
    const auto unsignedShift = static_cast<std::make_unsigned_t<T>>(shift);
    const unsigned shiftValue = static_cast<unsigned>(unsignedShift);
    const unsigned s = shiftValue & 63u;
    const uint64_t shiftedLow = low_ << s;
    uint64_t shiftedHigh;
#if defined(_M_X64)
    if (!std::is_constant_evaluated()) {
      shiftedHigh = __shiftleft128(low_, high_, static_cast<unsigned char>(s));
    } else
#endif
    {
      shiftedHigh = (high_ << s) | ((low_ >> 1) >> ((~s) & 63u));
    }
    const bool atLeast64 = unsignedShift >= 64;
    const bool zero = unsignedShift >= 128;
    uint64_t newHigh = atLeast64 ? shiftedLow : shiftedHigh;
    newHigh = zero ? 0ull : newHigh;
    const uint64_t newLow = atLeast64 ? 0ull : shiftedLow;
    return UInt128(newHigh, newLow);
  }

  // Logical right shift. Branchless single-return shape; see
  // Int128::operator>>. Portable fallback (ARM64) written once.
  template <
      typename T,
      std::enable_if_t<detail::kIsIntegralOperand<T>, int> = 0>
  constexpr UInt128 operator>>(T shift) const {
    const auto unsignedShift = static_cast<std::make_unsigned_t<T>>(shift);
    const unsigned shiftValue = static_cast<unsigned>(unsignedShift);
    const unsigned s = shiftValue & 63u;
    uint64_t shiftedLow;
#if defined(_M_X64)
    if (!std::is_constant_evaluated()) {
      shiftedLow = __shiftright128(low_, high_, static_cast<unsigned char>(s));
    } else
#endif
    {
      shiftedLow = (low_ >> s) | ((high_ << 1) << ((~s) & 63u));
    }
    const uint64_t logicalHigh = high_ >> s;
    const bool atLeast64 = unsignedShift >= 64;
    const bool zero = unsignedShift >= 128;
    uint64_t newLow = atLeast64 ? logicalHigh : shiftedLow;
    newLow = zero ? 0ull : newLow;
    const uint64_t newHigh = atLeast64 ? 0ull : logicalHigh;
    return UInt128(newHigh, newLow);
  }

  // Compound assignment shift operators
  template <
      typename T,
      std::enable_if_t<detail::kIsIntegralOperand<T>, int> = 0>
  UInt128& operator<<=(T shift) {
    *this = *this << shift;
    return *this;
  }

  template <
      typename T,
      std::enable_if_t<detail::kIsIntegralOperand<T>, int> = 0>
  UInt128& operator>>=(T shift) {
    *this = *this >> shift;
    return *this;
  }

  // Compound assignment bitwise operators
  UInt128& operator&=(const UInt128& other) {
    *this = *this & other;
    return *this;
  }

  UInt128& operator|=(const UInt128& other) {
    *this = *this | other;
    return *this;
  }

  UInt128& operator^=(const UInt128& other) {
    *this = *this ^ other;
    return *this;
  }

  std::to_chars_result toChars(char* first, char* last) const {
    return detail::uint128ToChars(first, last, high_, low_, false);
  }

 private:
  uint64_t low_; // Least significant 64 bits (offset 0, matches __uint128_t)
  uint64_t high_; // Most significant 64 bits (offset 8, matches __uint128_t)
};

// Define UInt128 conversion constructor to Int128 after UInt128 is defined
inline constexpr Int128::Int128(const UInt128& value)
    : low_(value.low()), high_(static_cast<int64_t>(value.high())) {}

namespace detail {

template <typename T>
concept Int128Type = std::is_same_v<T, Int128> || std::is_same_v<T, UInt128>;

template <typename T>
concept Int128Operand =
    Int128Type<T> || kIsArithmeticOperand<T> || kIsFloatingOperand<T>;

// Deduce both operands: native expressions and user-defined conversions must
// not be captured. Same-type custom operands use their existing member ops.
template <typename L, typename R>
concept MixedInt128Operands = Int128Operand<L> && Int128Operand<R> &&
    (Int128Type<L> || Int128Type<R>) && !std::is_same_v<L, R>;

template <typename L, typename R>
concept MixedIntegralInt128Operands = MixedInt128Operands<L, R> &&
    !kIsFloatingOperand<L> && !kIsFloatingOperand<R>;

// Floating point wins first, then UInt128, then Int128. A native unsigned
// integer still fits in Int128, unlike UInt128.
template <typename L, typename R>
using Int128Result = std::conditional_t<
    kIsFloatingOperand<L>,
    std::remove_cv_t<L>,
    std::conditional_t<
        kIsFloatingOperand<R>,
        std::remove_cv_t<R>,
        std::conditional_t<
            std::is_same_v<L, UInt128> || std::is_same_v<R, UInt128>,
            UInt128,
            Int128>>>;

template <typename Result, typename T>
constexpr decltype(auto) promoteInt128Operand(const T& value) {
  if constexpr (std::is_same_v<Result, T>) {
    return (value);
  } else if constexpr (kIsArithmeticOperand<T>) {
    return Result(normalizeIntegralOperand(value));
  } else {
    // Explicit floating conversions retain the full 128-bit value. On MSVC,
    // long double has the same format as double (see the conversion members).
    return static_cast<Result>(value);
  }
}

template <typename T>
constexpr bool fitsInt64(T value) {
  if constexpr (std::is_signed_v<T> || sizeof(T) < sizeof(uint64_t)) {
    return true;
  } else {
    return value <= static_cast<T>(std::numeric_limits<int64_t>::max());
  }
}

template <typename L, typename R>
  requires MixedInt128Operands<L, R>
constexpr bool mixedInt128Less(const L& left, const R& right) {
  using Result = Int128Result<L, R>;
  if constexpr (std::is_same_v<L, Int128> && kIsArithmeticOperand<R>) {
    // Preserve the sign-bit test for x < 0 (and the reversed/negated forms).
    return Int128::lessThanIntegralOperand(
        left, promoteInt128Operand<Int128>(right));
  } else {
    return promoteInt128Operand<Result>(left) <
        promoteInt128Operand<Result>(right);
  }
}

// Native destinations cannot be enums or const. Custom destinations already
// have member compounds for ordinary integers; keep those (including *=).
template <typename L, typename R>
concept Int128CompoundOperands =
    MixedInt128Operands<L, R> && !std::is_const_v<L> && !std::is_enum_v<L> &&
    (Int128Type<R> || !kIsIntegralOperand<R>);

} // namespace detail

template <typename L, typename R>
  requires detail::MixedInt128Operands<L, R>
constexpr auto operator+(const L& left, const R& right) {
  using Result = detail::Int128Result<L, R>;
  return detail::promoteInt128Operand<Result>(left) +
      detail::promoteInt128Operand<Result>(right);
}

template <typename L, typename R>
  requires detail::MixedInt128Operands<L, R>
constexpr auto operator-(const L& left, const R& right) {
  using Result = detail::Int128Result<L, R>;
  return detail::promoteInt128Operand<Result>(left) -
      detail::promoteInt128Operand<Result>(right);
}

template <typename L, typename R>
  requires detail::MixedInt128Operands<L, R>
constexpr auto operator*(const L& left, const R& right) {
  using Result = detail::Int128Result<L, R>;
  return detail::promoteInt128Operand<Result>(left) *
      detail::promoteInt128Operand<Result>(right);
}

// Force inlining so a signed scalar divisor reaches divideByInt64/modByInt64
// as a constant. Full-width unsigned scalars still use the general division.
template <typename L, typename R>
  requires detail::MixedInt128Operands<L, R>
FOLLY_ALWAYS_INLINE auto operator/(const L& left, const R& right) {
  using Result = detail::Int128Result<L, R>;
  if constexpr (std::is_same_v<L, Int128> && detail::kIsArithmeticOperand<R>) {
    const auto operand = detail::normalizeIntegralOperand(right);
    if (detail::fitsInt64(operand)) {
      return left.divideByInt64(static_cast<int64_t>(operand));
    }
    return left.operator/(Int128(operand));
  } else {
    return detail::promoteInt128Operand<Result>(left) /
        detail::promoteInt128Operand<Result>(right);
  }
}

template <typename L, typename R>
  requires detail::MixedIntegralInt128Operands<L, R>
FOLLY_ALWAYS_INLINE auto operator%(const L& left, const R& right) {
  using Result = detail::Int128Result<L, R>;
  if constexpr (std::is_same_v<L, Int128> && detail::kIsArithmeticOperand<R>) {
    const auto operand = detail::normalizeIntegralOperand(right);
    if (detail::fitsInt64(operand)) {
      return left.modByInt64(static_cast<int64_t>(operand));
    }
    return left.operator%(Int128(operand));
  } else {
    return detail::promoteInt128Operand<Result>(left) %
        detail::promoteInt128Operand<Result>(right);
  }
}

template <typename L, typename R>
  requires detail::MixedIntegralInt128Operands<L, R>
constexpr auto operator&(const L& left, const R& right) {
  using Result = detail::Int128Result<L, R>;
  return detail::promoteInt128Operand<Result>(left) &
      detail::promoteInt128Operand<Result>(right);
}

template <typename L, typename R>
  requires detail::MixedIntegralInt128Operands<L, R>
constexpr auto operator|(const L& left, const R& right) {
  using Result = detail::Int128Result<L, R>;
  return detail::promoteInt128Operand<Result>(left) |
      detail::promoteInt128Operand<Result>(right);
}

template <typename L, typename R>
  requires detail::MixedIntegralInt128Operands<L, R>
constexpr auto operator^(const L& left, const R& right) {
  using Result = detail::Int128Result<L, R>;
  return detail::promoteInt128Operand<Result>(left) ^
      detail::promoteInt128Operand<Result>(right);
}

template <typename L, typename R>
  requires detail::MixedInt128Operands<L, R>
constexpr bool operator==(const L& left, const R& right) {
  using Result = detail::Int128Result<L, R>;
  return detail::promoteInt128Operand<Result>(left) ==
      detail::promoteInt128Operand<Result>(right);
}

template <typename L, typename R>
  requires detail::MixedInt128Operands<L, R>
constexpr bool operator!=(const L& left, const R& right) {
  return !(left == right);
}

template <typename L, typename R>
  requires detail::MixedInt128Operands<L, R>
constexpr bool operator<(const L& left, const R& right) {
  return detail::mixedInt128Less(left, right);
}

template <typename L, typename R>
  requires detail::MixedInt128Operands<L, R>
constexpr bool operator>(const L& left, const R& right) {
  return detail::mixedInt128Less(right, left);
}

template <typename L, typename R>
  requires detail::MixedInt128Operands<L, R>
constexpr bool operator<=(const L& left, const R& right) {
  using Result = detail::Int128Result<L, R>;
  if constexpr (detail::kIsFloatingOperand<Result>) {
    // Negating < would turn unordered (NaN) comparisons into true.
    return detail::promoteInt128Operand<Result>(left) <=
        detail::promoteInt128Operand<Result>(right);
  } else {
    return !detail::mixedInt128Less(right, left);
  }
}

template <typename L, typename R>
  requires detail::MixedInt128Operands<L, R>
constexpr bool operator>=(const L& left, const R& right) {
  using Result = detail::Int128Result<L, R>;
  if constexpr (detail::kIsFloatingOperand<Result>) {
    return detail::promoteInt128Operand<Result>(left) >=
        detail::promoteInt128Operand<Result>(right);
  } else {
    return !detail::mixedInt128Less(left, right);
  }
}

// Convert back only after the promoted operation succeeds: in particular,
// never narrow a full-width divisor or modify the destination on a throw.
template <typename L, typename R>
  requires detail::Int128CompoundOperands<L, R>
L& operator+=(L& left, const R& right) {
  left = static_cast<L>(left + right);
  return left;
}

template <typename L, typename R>
  requires detail::Int128CompoundOperands<L, R>
L& operator-=(L& left, const R& right) {
  left = static_cast<L>(left - right);
  return left;
}

template <typename L, typename R>
  requires detail::Int128CompoundOperands<L, R>
L& operator*=(L& left, const R& right) {
  left = static_cast<L>(left * right);
  return left;
}

template <typename L, typename R>
  requires detail::Int128CompoundOperands<L, R>
L& operator/=(L& left, const R& right) {
  left = static_cast<L>(left / right);
  return left;
}

template <typename L, typename R>
  requires detail::Int128CompoundOperands<L, R> &&
    detail::MixedIntegralInt128Operands<L, R>
L& operator%=(L& left, const R& right) {
  left = static_cast<L>(left % right);
  return left;
}

template <typename L, typename R>
  requires detail::Int128CompoundOperands<L, R> &&
    detail::MixedIntegralInt128Operands<L, R>
L& operator&=(L& left, const R& right) {
  left = static_cast<L>(left & right);
  return left;
}

template <typename L, typename R>
  requires detail::Int128CompoundOperands<L, R> &&
    detail::MixedIntegralInt128Operands<L, R>
L& operator|=(L& left, const R& right) {
  left = static_cast<L>(left | right);
  return left;
}

template <typename L, typename R>
  requires detail::Int128CompoundOperands<L, R> &&
    detail::MixedIntegralInt128Operands<L, R>
L& operator^=(L& left, const R& right) {
  left = static_cast<L>(left ^ right);
  return left;
}

// Type aliases for compatibility
using int128_t = Int128;
using uint128_t = UInt128;

} // namespace facebook::velox

// Supply the compiler type spellings used by existing headers on MSVC only.
using __int128_t = facebook::velox::Int128;
using __uint128_t = facebook::velox::UInt128;

// std::numeric_limits specialization
namespace std {

template <>
struct make_unsigned<facebook::velox::Int128> {
  using type = facebook::velox::UInt128;
};

template <>
struct make_unsigned<facebook::velox::UInt128> {
  using type = facebook::velox::UInt128;
};

template <>
struct make_signed<facebook::velox::Int128> {
  using type = facebook::velox::Int128;
};

template <>
struct make_signed<facebook::velox::UInt128> {
  using type = facebook::velox::Int128;
};

// common_type is a permitted customization point for program-defined types.
// Restrict these to decayed operands; the primary template handles cv/ref.
template <typename T>
  requires(is_arithmetic_v<T> && is_same_v<T, decay_t<T>>)
struct common_type<facebook::velox::Int128, T> {
  using type =
      conditional_t<is_floating_point_v<T>, T, facebook::velox::Int128>;
};

template <typename T>
  requires(is_arithmetic_v<T> && is_same_v<T, decay_t<T>>)
struct common_type<T, facebook::velox::Int128>
    : common_type<facebook::velox::Int128, T> {};

template <typename T>
  requires(is_arithmetic_v<T> && is_same_v<T, decay_t<T>>)
struct common_type<facebook::velox::UInt128, T> {
  using type =
      conditional_t<is_floating_point_v<T>, T, facebook::velox::UInt128>;
};

template <typename T>
  requires(is_arithmetic_v<T> && is_same_v<T, decay_t<T>>)
struct common_type<T, facebook::velox::UInt128>
    : common_type<facebook::velox::UInt128, T> {};

template <>
struct common_type<facebook::velox::Int128, facebook::velox::UInt128> {
  using type = facebook::velox::UInt128;
};

template <>
struct common_type<facebook::velox::UInt128, facebook::velox::Int128> {
  using type = facebook::velox::UInt128;
};

template <>
class numeric_limits<facebook::velox::Int128> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr int digits = 127;
  static constexpr int digits10 = 38;
  static constexpr int max_digits10 = 0;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = true;
  static constexpr bool is_exact = true;
  static constexpr int radix = 2;
  static constexpr int min_exponent = 0;
  static constexpr int min_exponent10 = 0;
  static constexpr int max_exponent = 0;
  static constexpr int max_exponent10 = 0;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = false;
  static constexpr bool has_signaling_NaN = false;
  static constexpr float_denorm_style has_denorm = denorm_absent;
  static constexpr bool has_denorm_loss = false;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr bool traps = numeric_limits<int64_t>::traps;
  static constexpr bool tinyness_before = false;
  static constexpr float_round_style round_style = round_toward_zero;

  static constexpr facebook::velox::Int128 min() noexcept {
    return facebook::velox::Int128(std::numeric_limits<int64_t>::min(), 0);
  }

  static constexpr facebook::velox::Int128 max() noexcept {
    return facebook::velox::Int128(
        std::numeric_limits<int64_t>::max(),
        std::numeric_limits<uint64_t>::max());
  }

  static constexpr facebook::velox::Int128 lowest() noexcept {
    return min();
  }

  static constexpr facebook::velox::Int128 epsilon() noexcept {
    return facebook::velox::Int128(0);
  }

  static constexpr facebook::velox::Int128 round_error() noexcept {
    return facebook::velox::Int128(0);
  }

  static constexpr facebook::velox::Int128 infinity() noexcept {
    return facebook::velox::Int128(0);
  }

  static constexpr facebook::velox::Int128 quiet_NaN() noexcept {
    return facebook::velox::Int128(0);
  }

  static constexpr facebook::velox::Int128 signaling_NaN() noexcept {
    return facebook::velox::Int128(0);
  }

  static constexpr facebook::velox::Int128 denorm_min() noexcept {
    return facebook::velox::Int128(0);
  }
};

template <>
class numeric_limits<facebook::velox::UInt128> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr int digits = 128;
  static constexpr int digits10 = 38;
  static constexpr int max_digits10 = 0;
  static constexpr bool is_signed = false;
  static constexpr bool is_integer = true;
  static constexpr bool is_exact = true;
  static constexpr int radix = 2;
  static constexpr int min_exponent = 0;
  static constexpr int min_exponent10 = 0;
  static constexpr int max_exponent = 0;
  static constexpr int max_exponent10 = 0;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = false;
  static constexpr bool has_signaling_NaN = false;
  static constexpr float_denorm_style has_denorm = denorm_absent;
  static constexpr bool has_denorm_loss = false;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = true;
  static constexpr bool traps = numeric_limits<uint64_t>::traps;
  static constexpr bool tinyness_before = false;
  static constexpr float_round_style round_style = round_toward_zero;

  static constexpr facebook::velox::UInt128 min() noexcept {
    return facebook::velox::UInt128(0, 0);
  }

  static constexpr facebook::velox::UInt128 max() noexcept {
    return facebook::velox::UInt128(
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max());
  }

  static constexpr facebook::velox::UInt128 lowest() noexcept {
    return min();
  }

  static constexpr facebook::velox::UInt128 epsilon() noexcept {
    return facebook::velox::UInt128(0);
  }

  static constexpr facebook::velox::UInt128 round_error() noexcept {
    return facebook::velox::UInt128(0);
  }

  static constexpr facebook::velox::UInt128 infinity() noexcept {
    return facebook::velox::UInt128(0);
  }

  static constexpr facebook::velox::UInt128 quiet_NaN() noexcept {
    return facebook::velox::UInt128(0);
  }

  static constexpr facebook::velox::UInt128 signaling_NaN() noexcept {
    return facebook::velox::UInt128(0);
  }

  static constexpr facebook::velox::UInt128 denorm_min() noexcept {
    return facebook::velox::UInt128(0);
  }
};

} // namespace std

// std::hash specialization for Int128
namespace std {
template <>
struct hash<facebook::velox::Int128> {
  size_t operator()(const facebook::velox::Int128& value) const noexcept {
    // Combine high and low parts
    size_t h1 = hash<int64_t>{}(value.high());
    size_t h2 = hash<uint64_t>{}(value.low());
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

template <>
struct hash<facebook::velox::UInt128> {
  size_t operator()(const facebook::velox::UInt128& value) const noexcept {
    size_t h1 = hash<uint64_t>{}(value.high());
    size_t h2 = hash<uint64_t>{}(value.low());
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};
} // namespace std

// Folly hasher specialization for Int128. This must stay a hash_combine of the
// two limbs: SimpleVector::hashValueAt and ConstantTypedExpr::hash use this
// hasher, and their values must equal the hash_combine of the limbs that
// VectorHasher, RowContainer and ContainerRowSerde compute for HUGEINT.
namespace folly {
template <>
struct hasher<facebook::velox::Int128> {
  size_t operator()(const facebook::velox::Int128& value) const noexcept {
    return folly::hash::hash_combine(
        folly::hasher<int64_t>{}(value.high()),
        folly::hasher<uint64_t>{}(value.low()));
  }
};

template <>
struct hasher<facebook::velox::UInt128> {
  size_t operator()(const facebook::velox::UInt128& value) const noexcept {
    return folly::hash::hash_combine(
        folly::hasher<uint64_t>{}(value.high()),
        folly::hasher<uint64_t>{}(value.low()));
  }
};

// toAppend specializations for folly::to<std::string>() support. Decimal, the
// same text native __int128 produces on Linux.
// These need to be templates to match folly's ADL expectations
template <class Tgt>
typename std::enable_if<folly::IsSomeString<Tgt>::value, void>::type toAppend(
    const facebook::velox::Int128& value,
    Tgt* result) {
  char buffer[40];
  const auto converted = value.toChars(buffer, buffer + sizeof(buffer));
  result->append(buffer, converted.ptr - buffer);
}

template <class Tgt>
typename std::enable_if<folly::IsSomeString<Tgt>::value, void>::type toAppend(
    const facebook::velox::UInt128& value,
    Tgt* result) {
  char buffer[40];
  const auto converted = value.toChars(buffer, buffer + sizeof(buffer));
  result->append(buffer, converted.ptr - buffer);
}

} // namespace folly

// Free-standing reverse comparison operators (int64_t op Int128)
namespace facebook::velox {

inline constexpr bool operator<(int64_t lhs, const Int128& rhs) {
  return Int128(lhs) < rhs;
}
inline constexpr bool operator>(int64_t lhs, const Int128& rhs) {
  return Int128::lessThanIntegralOperand(rhs, Int128(lhs));
}
inline constexpr bool operator<=(int64_t lhs, const Int128& rhs) {
  return !Int128::lessThanIntegralOperand(rhs, Int128(lhs));
}
inline constexpr bool operator>=(int64_t lhs, const Int128& rhs) {
  return Int128(lhs) >= rhs;
}
inline constexpr bool operator==(int64_t lhs, const Int128& rhs) {
  return Int128(lhs) == rhs;
}
inline constexpr bool operator!=(int64_t lhs, const Int128& rhs) {
  return Int128(lhs) != rhs;
}

} // namespace facebook::velox

// Forwarding toAppend functions in facebook::velox namespace for ADL
namespace facebook::velox {

template <class Tgt>
typename std::enable_if<folly::IsSomeString<Tgt>::value, void>::type toAppend(
    const int128_t& value,
    Tgt* result) {
  folly::toAppend(value, result);
}

template <class Tgt>
typename std::enable_if<folly::IsSomeString<Tgt>::value, void>::type toAppend(
    const uint128_t& value,
    Tgt* result) {
  folly::toAppend(value, result);
}

} // namespace facebook::velox

// fmt formatter specialization for Int128
template <>
struct fmt::formatter<facebook::velox::Int128> : fmt::formatter<std::string> {
  auto format(const facebook::velox::Int128& value, format_context& ctx) const {
    return fmt::formatter<std::string>::format(value.toString(), ctx);
  }
};

template <>
struct fmt::formatter<facebook::velox::UInt128> : fmt::formatter<std::string> {
  auto format(const facebook::velox::UInt128& value, format_context& ctx)
      const {
    char buffer[40];
    const auto converted = value.toChars(buffer, buffer + sizeof(buffer));
    return fmt::formatter<std::string>::format(
        std::string(buffer, converted.ptr), ctx);
  }
};
