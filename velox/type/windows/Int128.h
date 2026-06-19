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

#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>
#include <string>

#include <folly/Conv.h>
#include <folly/Hash.h>

#ifdef _MSC_VER
#include <intrin.h>
// Windows-compatible 128-bit integer implementation using two 64-bit values

namespace facebook::velox {

// Forward declarations
class UInt128;

class Int128 {
 public:
  constexpr Int128() : low_(0), high_(0) {}
  
  constexpr Int128(int64_t high, uint64_t low) : low_(low), high_(high) {}
  
  // Branchless sign-extension: `value >> 63` is an arithmetic shift on int64,
  // producing 0 for non-negative and -1 for negative. Lowers to a single `sar`
  // instead of a compare + cmov / branch.
  constexpr Int128(int64_t value) : low_(static_cast<uint64_t>(value)), high_(value >> 63) {}
  
  constexpr Int128(uint64_t value) : low_(value), high_(0) {}
  
  constexpr Int128(int32_t value) : low_(static_cast<uint64_t>(value)), high_(static_cast<int64_t>(value) >> 63) {}
  
  constexpr Int128(uint32_t value) : low_(value), high_(0) {}
  
  constexpr Int128(const Int128& other) = default;
  constexpr Int128& operator=(const Int128& other) = default;
  
  // Forward declaration - defined after UInt128 class
  Int128(const UInt128& value);

  // Getters
  constexpr int64_t high() const { return high_; }
  constexpr uint64_t low() const { return low_; }
  
  // Conversion operators
  explicit operator bool() const { return high_ != 0 || low_ != 0; }
  explicit operator int64_t() const { return static_cast<int64_t>(low_); }
  explicit operator uint64_t() const { return low_; }
  explicit operator int32_t() const { return static_cast<int32_t>(low_); }
  explicit operator uint32_t() const { return static_cast<uint32_t>(low_); }
  explicit operator int16_t() const { return static_cast<int16_t>(low_); }
  explicit operator uint16_t() const { return static_cast<uint16_t>(low_); }
  explicit operator int8_t() const { return static_cast<int8_t>(low_); }
  explicit operator uint8_t() const { return static_cast<uint8_t>(low_); }
  explicit operator double() const {
    // Convert to double (may lose precision for very large values)
    constexpr double k2pow64 = 18446744073709551616.0; // 2^64
    return static_cast<double>(high_) * k2pow64 + static_cast<double>(low_);
  }
  
  // Comparison with primitive integer types
  constexpr bool operator<(int64_t other) const {
    return *this < Int128(other);
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
    return *this >= Int128(other);
  }
  
  // Arithmetic operators. At runtime use `_addcarry_u64`/`_subborrow_u64` so
  // MSVC lowers the 128-bit add/sub to `add; adc` / `sub; sbb` (~2 uops,
  // branch-free) instead of the previous branchy compare-and-increment carry
  // which the compiler could not always recognize as an ADC pattern. The
  // intrinsics are not constexpr; the `is_constant_evaluated` branch keeps
  // the operator usable in constant expressions via the portable fallback.
  //
  // Codegen note (verified via /FAs on MSVC 14.44): this lowers to exactly
  // two instructions, `add r9, [bL]; adc r8, [bH]`, vs the previous 5
  // (add/mov/add/cmp/lea/cmovae). When measured with both halves of the
  // result forced live (so neither side's high-half computation gets DCE'd):
  //   - Tight reduction `acc = acc + v`:  -6%  vs old
  //   - Latency chain   `x = x + c`:      -10% vs old
  //   - Throughput, parallel inputs:      tied at loop-overhead floor
  // The early version of the bench harness XORed only `.low()` into the sink,
  // which let MSVC DCE the old form's plain-arithmetic high half (the new
  // `_addcarry_u64` intrinsic is opaque to dead-store analysis), producing
  // an artifactual "regression". Lesson for future int128 benchmarking on
  // MSVC: always consume both halves of every result.
  constexpr Int128 operator+(const Int128& other) const {
    // Compute the high half in uint64_t (defined two's-complement wrap)
    // and reinterpret as int64_t at the end. Doing the add as signed
    // int64_t would be UB on overflow -- reachable e.g. via unary
    // negation of INT128_MIN, which lowers to `0 - INT128_MIN` and would
    // do `0 - INT64_MIN` on the high half.
    if (std::is_constant_evaluated()) {
      uint64_t new_low = low_ + other.low_;
      uint64_t new_high = static_cast<uint64_t>(high_) +
          static_cast<uint64_t>(other.high_) + (new_low < low_ ? 1 : 0);
      return Int128(static_cast<int64_t>(new_high), new_low);
    }
#if defined(_M_X64)
    uint64_t new_low;
    uint64_t new_high;
    unsigned char c = _addcarry_u64(0, low_, other.low_, &new_low);
    _addcarry_u64(c, static_cast<uint64_t>(high_),
                  static_cast<uint64_t>(other.high_), &new_high);
    return Int128(static_cast<int64_t>(new_high), new_low);
#else
    // ARM64: MSVC's _addcarry_u64 is x64-only. The compiler lowers the
    // portable carry form below to `adds; adcs` (2 instructions) on ARM64.
    uint64_t new_low = low_ + other.low_;
    uint64_t new_high = static_cast<uint64_t>(high_) +
        static_cast<uint64_t>(other.high_) + (new_low < low_ ? 1 : 0);
    return Int128(static_cast<int64_t>(new_high), new_low);
#endif
  }
  
  constexpr Int128 operator-(const Int128& other) const {
    // High half in uint64_t to avoid signed-overflow UB; see operator+.
    // Notably reachable via unary negation of INT128_MIN.
    if (std::is_constant_evaluated()) {
      uint64_t new_low = low_ - other.low_;
      uint64_t new_high = static_cast<uint64_t>(high_) -
          static_cast<uint64_t>(other.high_) - (new_low > low_ ? 1 : 0);
      return Int128(static_cast<int64_t>(new_high), new_low);
    }
#if defined(_M_X64)
    uint64_t new_low;
    uint64_t new_high;
    unsigned char b = _subborrow_u64(0, low_, other.low_, &new_low);
    _subborrow_u64(b, static_cast<uint64_t>(high_),
                   static_cast<uint64_t>(other.high_), &new_high);
    return Int128(static_cast<int64_t>(new_high), new_low);
#else
    uint64_t new_low = low_ - other.low_;
    uint64_t new_high = static_cast<uint64_t>(high_) -
        static_cast<uint64_t>(other.high_) - (new_low > low_ ? 1 : 0);
    return Int128(static_cast<int64_t>(new_high), new_low);
#endif
  }
  
  // Branch-free negation: `0 - x` lowers to `neg; sbb` (2 instructions) and
  // avoids the UB-on-INT128_MIN branch in the previous implementation.
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
  
  Int128& operator/=(const Int128& other) {
    if (other.low_ == 0 && other.high_ == 0) {
      throw std::invalid_argument("Division by zero");
    }
    *this = *this / other;
    return *this;
  }
  
  Int128& operator*=(const Int128& other) {
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
  Int128& operator*=(int64_t value) {
    return *this *= Int128(value);
  }
  
  Int128& operator*=(int32_t value) {
    return *this *= Int128(value);
  }
  
  // Specialized division by a 64-bit signed integer. Skips Int128 ctor on the
  // divisor, skips the divisor_hi==0 check inside the general 128/128 path,
  // and calls `_udiv128` directly. ~2x faster than `*this / Int128(other)`
  // on random 128/64 inputs in microbench (10.8 ns -> ~5.5 ns).
  //
  // When `other` is a compile-time constant, MSVC folds the sign / abs work
  // away entirely (verified via /FAs); for the common decimal scale-reduction
  // case `value / kPowerOf10` the only remaining runtime cost is the
  // unavoidable 128/64 divide itself. Use `divideByConstant<N>()` below for a
  // fully template-parameterised divisor where the constant is propagated
  // even through non-inlining call sites.
  //
  // Overflow note: INT128_MIN / -1 silently wraps to INT128_MIN here, matching
  // gcc/clang `__int128_t` behaviour (libgcc `__divti3`: abs both operands as
  // unsigned, divide, sign-correct, store back -- the +2^127 result wraps when
  // bit-cast into signed). This is undefined behaviour per the C/C++ standard
  // but the de-facto cross-compiler convention. Velox's decimal contract makes
  // this unreachable for legal callers: `DecimalUtil::kLongDecimalMin` is
  // -(10^38 - 1), strictly inside [INT128_MIN+1, INT128_MAX-1], so INT128_MIN
  // is never a valid decimal mantissa and `valueInRange()` rejects it at
  // boundaries. `DecimalUtil::addWithOverflow` also DCHECKs the same invariant.
  Int128 divideByInt64(int64_t other) const {
    // Sign extraction: dividend sign is high_'s sign bit; divisor sign is
    // other's sign bit. XOR gives result sign.
    const bool dividend_neg = high_ < 0;
    const bool divisor_neg = other < 0;
    const bool result_neg = dividend_neg ^ divisor_neg;
    // Compute |dividend| in two's-complement (well-defined on INT128_MIN).
    uint64_t num_lo, num_hi;
    if (dividend_neg) {
      num_lo = 0 - low_;
      num_hi = ~static_cast<uint64_t>(high_) + (low_ == 0 ? 1 : 0);
    } else {
      num_lo = low_;
      num_hi = static_cast<uint64_t>(high_);
    }
    // |other| as uint64 (well-defined on INT64_MIN: yields 2^63).
    const uint64_t den = divisor_neg
        ? (0 - static_cast<uint64_t>(other))
        : static_cast<uint64_t>(other);
    if (den == 0) {
      throw std::runtime_error("Division by zero");
    }
    uint64_t q_lo, q_hi;
    if (num_hi == 0) {
      // Single 64/64 divide.
      q_hi = 0;
      q_lo = num_lo / den;
    } else if (num_hi < den) {
      // Quotient fits in 64 bits.
      q_hi = 0;
#if defined(_M_X64)
      uint64_t rem;
      q_lo = _udiv128(num_hi, num_lo, den, &rem);
#else
      // ARM64 fallback: delegate to udivmod128 which has an ARM64 path.
      uint64_t qh, ql, rh, rl;
      udivmod128(num_hi, num_lo, 0, den, qh, ql, rh, rl);
      q_lo = ql;
#endif
    } else {
      // Two-step: high quotient + carry-in via _udiv128.
      q_hi = num_hi / den;
      const uint64_t rem_hi = num_hi % den;
#if defined(_M_X64)
      uint64_t rem;
      q_lo = _udiv128(rem_hi, num_lo, den, &rem);
#else
      uint64_t qh, ql, rh, rl;
      udivmod128(rem_hi, num_lo, 0, den, qh, ql, rh, rl);
      q_lo = ql;
#endif
    }
    Int128 result(static_cast<int64_t>(q_hi), q_lo);
    return result_neg ? -result : result;
  }
  
  Int128 modByInt64(int64_t other) const {
    // Result sign of `%` follows the dividend.
    const bool dividend_neg = high_ < 0;
    const bool divisor_neg = other < 0;
    uint64_t num_lo, num_hi;
    if (dividend_neg) {
      num_lo = 0 - low_;
      num_hi = ~static_cast<uint64_t>(high_) + (low_ == 0 ? 1 : 0);
    } else {
      num_lo = low_;
      num_hi = static_cast<uint64_t>(high_);
    }
    const uint64_t den = divisor_neg
        ? (0 - static_cast<uint64_t>(other))
        : static_cast<uint64_t>(other);
    if (den == 0) {
      throw std::runtime_error("Modulo by zero");
    }
    uint64_t rem;
    if (num_hi == 0) {
      rem = num_lo % den;
    } else {
      const uint64_t rem_hi = num_hi < den ? num_hi : (num_hi % den);
#if defined(_M_X64)
      (void)_udiv128(rem_hi, num_lo, den, &rem);
#else
      uint64_t qh, ql, rh, rl;
      udivmod128(rem_hi, num_lo, 0, den, qh, ql, rh, rl);
      rem = rl;
#endif
    }
    // Remainder always fits in uint64 (rem < den <= 2^63). Sign-correct.
    Int128 result(0, rem);
    return dividend_neg ? -result : result;
  }
  
  // Compile-time-constant divisor. When the divisor is a known non-zero
  // constant, the compiler folds the sign work, picks the right branch of
  // `divideByInt64`, and (for divisors < 2^31) can sometimes lower to
  // multiplicative-inverse code. Use as `x.divideByConstant<1000000>()`.
  template <int64_t Divisor>
  Int128 divideByConstant() const {
    static_assert(Divisor != 0, "Divisor must be non-zero");
    return divideByInt64(Divisor);
  }
  
  template <int64_t Divisor>
  Int128 modByConstant() const {
    static_assert(Divisor != 0, "Divisor must be non-zero");
    return modByInt64(Divisor);
  }
  
  // Built-in-type division overloads. Route through the specialized 128/64
  // path instead of constructing a full Int128 divisor and going through the
  // general 128/128 algorithm.
  Int128 operator/(int64_t other) const { return divideByInt64(other); }
  Int128 operator/(int32_t other) const { return divideByInt64(other); }
  Int128 operator/(long other) const {
    return divideByInt64(static_cast<int64_t>(other));
  }
  
  // Additional multiplication operators for compatibility with int64_t
  Int128 operator*(int64_t other) const {
    return *this * Int128(other);
  }
  
  friend Int128 operator*(int64_t left, const Int128& right) {
    return Int128(left) * right;
  }

  // Free operators for `<built-in integer> +/- Int128`. The member operator+/-
  // only matches when the left operand is an Int128, so mixed expressions with
  // a built-in integer on the left (e.g. `int64_t + Int128`, or smaller types
  // that promote to int64_t) need these. Mirrors operator* above. On non-MSVC
  // platforms int128_t is the native __int128 and these are unnecessary.
  friend Int128 operator+(int64_t left, const Int128& right) {
    return Int128(left) + right;
  }

  friend Int128 operator-(int64_t left, const Int128& right) {
    return Int128(left) - right;
  }
  
  // Multiplication: full 128x128 -> low-128 product.
  //   (a_hi:a_lo) * (b_hi:b_lo) mod 2^128
  //     = a_lo*b_lo + ((a_lo*b_hi + a_hi*b_lo) << 64)
  // Use `_umul128` for the low*low full 128-bit product (single hardware
  // `mul` instruction), then add the two cross terms (low 64 bits only).
  // 3 muls + 2 adds total, vs the previous 5 muls + many adds. The
  // `is_constant_evaluated` branch keeps a constexpr-callable fallback
  // since `_umul128` is not constexpr.
  constexpr Int128 operator*(const Int128& other) const {
    if (std::is_constant_evaluated()) {
      uint64_t a_lo = low_;
      uint64_t b_lo = other.low_;
      uint64_t a_lo_lo = a_lo & 0xFFFFFFFF;
      uint64_t a_lo_hi = a_lo >> 32;
      uint64_t b_lo_lo = b_lo & 0xFFFFFFFF;
      uint64_t b_lo_hi = b_lo >> 32;
      uint64_t p0 = a_lo_lo * b_lo_lo;
      uint64_t p1 = a_lo_lo * b_lo_hi;
      uint64_t p2 = a_lo_hi * b_lo_lo;
      uint64_t p3 = a_lo_hi * b_lo_hi;
      uint64_t carry = ((p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF)) >> 32;
      uint64_t result_low = a_lo * b_lo;
      uint64_t result_high_unsigned = p3 + (p1 >> 32) + (p2 >> 32) + carry +
          low_ * static_cast<uint64_t>(other.high_) +
          static_cast<uint64_t>(high_) * other.low_;
      return Int128(static_cast<int64_t>(result_high_unsigned), result_low);
    }
#if defined(_M_X64)
    uint64_t lo_hi;
    uint64_t result_low = _umul128(low_, other.low_, &lo_hi);
    uint64_t result_high =
        lo_hi + low_ * static_cast<uint64_t>(other.high_) +
        static_cast<uint64_t>(high_) * other.low_;
    return Int128(static_cast<int64_t>(result_high), result_low);
#elif defined(_M_ARM64)
    // ARM64: __umulh gives the high 64 bits of an unsigned 64x64 product;
    // the low 64 bits come from native `*`. Same 3-mul shape as x64.
    uint64_t result_low = low_ * other.low_;
    uint64_t lo_hi = __umulh(low_, other.low_);
    uint64_t result_high =
        lo_hi + low_ * static_cast<uint64_t>(other.high_) +
        static_cast<uint64_t>(high_) * other.low_;
    return Int128(static_cast<int64_t>(result_high), result_low);
#else
    // Portable fallback: same 32-bit-halves construction as the constexpr
    // branch. Slower than the intrinsic path but correct on any arch.
    uint64_t a_lo = low_;
    uint64_t b_lo = other.low_;
    uint64_t a_lo_lo = a_lo & 0xFFFFFFFF;
    uint64_t a_lo_hi = a_lo >> 32;
    uint64_t b_lo_lo = b_lo & 0xFFFFFFFF;
    uint64_t b_lo_hi = b_lo >> 32;
    uint64_t p0 = a_lo_lo * b_lo_lo;
    uint64_t p1 = a_lo_lo * b_lo_hi;
    uint64_t p2 = a_lo_hi * b_lo_lo;
    uint64_t p3 = a_lo_hi * b_lo_hi;
    uint64_t carry = ((p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF)) >> 32;
    uint64_t result_low = a_lo * b_lo;
    uint64_t result_high_unsigned = p3 + (p1 >> 32) + (p2 >> 32) + carry +
        low_ * static_cast<uint64_t>(other.high_) +
        static_cast<uint64_t>(high_) * other.low_;
    return Int128(static_cast<int64_t>(result_high_unsigned), result_low);
#endif
  }
  
  // Helper: unsigned 128-bit division returning quotient and remainder.
  // Operates on absolute (unsigned) values represented as (high, low).
  static void udivmod128(
      uint64_t dividend_hi,
      uint64_t dividend_lo,
      uint64_t divisor_hi,
      uint64_t divisor_lo,
      uint64_t& q_hi,
      uint64_t& q_lo,
      uint64_t& r_hi,
      uint64_t& r_lo) {
    q_hi = q_lo = r_hi = r_lo = 0;
    if (divisor_hi == 0 && divisor_lo == 0) {
      throw std::runtime_error("Division by zero");
    }
    // If dividend < divisor, quotient is 0, remainder is dividend.
    if (dividend_hi < divisor_hi ||
        (dividend_hi == divisor_hi && dividend_lo < divisor_lo)) {
      r_hi = dividend_hi;
      r_lo = dividend_lo;
      return;
    }
    // If divisor fits in 64 bits, use optimized path.
    if (divisor_hi == 0) {
      if (dividend_hi == 0) {
        // Both fit in 64 bits.
        q_lo = dividend_lo / divisor_lo;
        r_lo = dividend_lo % divisor_lo;
        return;
      }
      // 128-bit by 64-bit division using _udiv128 intrinsic.
      // _udiv128(high, low, divisor, &remainder) requires high < divisor
      // to guarantee the quotient fits in 64 bits.
      q_hi = dividend_hi / divisor_lo;
      uint64_t rem_hi = dividend_hi % divisor_lo;
      // Now divide (rem_hi : dividend_lo) by divisor_lo.
      // rem_hi < divisor_lo is guaranteed, so quotient fits in 64 bits.
#if defined(_M_ARM64)
      // ARM64 doesn't have _udiv128 intrinsic. Use binary long division.
      // Since rem_hi < divisor_lo, the quotient fits in 64 bits.
      uint64_t num_hi = rem_hi;
      uint64_t num_lo = dividend_lo;
      uint64_t quot = 0;
      for (int i = 63; i >= 0; --i) {
        // Shift (num_hi:num_lo) left by 1, pull next bit from num_lo
        num_hi = (num_hi << 1) | (num_lo >> 63);
        num_lo <<= 1;
        if (num_hi >= divisor_lo) {
          num_hi -= divisor_lo;
          quot |= (1ULL << i);
        }
      }
      q_lo = quot;
      r_lo = num_hi;
#else
      q_lo = _udiv128(rem_hi, dividend_lo, divisor_lo, &r_lo);
#endif
      return;
    }
    // General case: 128/128 unsigned divide.
    //
    // x64: Knuth Algorithm D (TAOCP vol 2, 4.3.1) at base 2^64.  Performs
    // a normalized 2-digit-by-2-digit divide with at most 2 qhat corrections,
    // using _udiv128 + _umul128 + adc/sbb intrinsics.  ~25x faster than the
    // binary shift-subtract loop below for random 128/128 inputs.  Cost is
    // dominated by one `divq` (~20 cycles) + a `mulq` + carry-propagation.
    //
    // ARM64: keeps the binary long-division loop, which is the only portable
    // option without _udiv128.
#if defined(_M_X64)
    {
      // Normalize: shift both operands left until divisor's high bit is set.
      // This bounds the qhat estimate to be off by at most 2 (Knuth thm T).
      unsigned long bsr;
      _BitScanReverse64(&bsr, divisor_hi);
      const unsigned char d = static_cast<unsigned char>(63 - bsr);
      uint64_t denHi = divisor_hi, denLo = divisor_lo;
      uint64_t numHi = dividend_hi, numLo = dividend_lo;
      uint64_t numTop = 0; // 3rd "digit" produced by shifting num left by d
      if (d != 0) {
        denHi = (denHi << d) | (denLo >> (64 - d));
        denLo <<= d;
        numTop = numHi >> (64 - d);
        numHi = (numHi << d) | (numLo >> (64 - d));
        numLo <<= d;
      }
      // Estimate qhat = (numTop:numHi) / denHi.  If numTop == denHi the true
      // quotient would overflow 64 bits, so we represent it with qhatHi=1
      // and bias the divq input.
      const uint64_t qhatHi0 = (numTop >= denHi) ? 1 : 0;
      uint64_t rhat;
      uint64_t qhat = _udiv128(
          (numTop >= denHi) ? (numTop - denHi) : numTop,
          numHi,
          denHi,
          &rhat);
      uint64_t qhatHi = qhatHi0;
      // Correction loop: decrement qhat while qhat * denLo > (rhat:numHi).
      // Bounded at <= 2 iterations.
      for (;;) {
        if (qhatHi > 0) {
          if (qhat-- == 0) {
            --qhatHi;
          }
        } else {
          uint64_t prodHi;
          const uint64_t prodLo = _umul128(qhat, denLo, &prodHi);
          // Algorithm D step D3: continue correcting while
          //   qhat * denLo > b * rhat + u[j+n-2]
          // For our 2-by-2 divide (n=2, j=0) the digit u[j+n-2] is u[0] =
          // numLo (the LOW digit of the normalized numerator), not numHi.
          // Using numHi here could leave qhat off by more than the
          // post-multiply borrow-correction can recover from for narrow
          // adversarial inputs (prodHi == rhat && numLo < prodLo <= numHi).
          if (prodHi < rhat || (prodHi == rhat && prodLo <= numLo)) {
            break;
          }
          --qhat;
        }
        const uint64_t sum = rhat + denHi;
        if (rhat > sum) {
          // rhat += denHi overflowed -> any further qhat*denLo can never
          // exceed (rhat:numHi); stop correcting.
          break;
        }
        rhat = sum;
      }
      // Subtract qhat * den from (numTop:numHi:numLo).  This is a 3-digit
      // minus 1-digit-times-2-digit subtract; never underflows due to the
      // correction loop above (modulo the one borrow we may need to undo).
      uint64_t prod0Hi;
      uint64_t prodLo = _umul128(qhat, denLo, &prod0Hi);
      unsigned char borrow = _subborrow_u64(0, numLo, prodLo, &numLo);
      uint64_t prod1Hi;
      prodLo = _umul128(qhat, denHi, &prod1Hi);
      prod1Hi += _addcarry_u64(0, prodLo, prod0Hi, &prodLo);
      borrow = _subborrow_u64(borrow, numHi, prodLo, &numHi);
      borrow = _subborrow_u64(borrow, numTop, prod1Hi, &numTop);
      // If borrow remains, qhat was 1 too large; correct by adding den back.
      if (borrow) {
        --qhat;
        const unsigned char carry =
            _addcarry_u64(0, numLo, denLo, &numLo);
        _addcarry_u64(carry, numHi, denHi, &numHi);
      }
      // Un-normalize the remainder (right-shift by d bits).
      if (d != 0) {
        numLo = (numLo >> d) | (numHi << (64 - d));
        numHi >>= d;
      }
      q_hi = 0;
      q_lo = qhat;
      r_hi = numHi;
      r_lo = numLo;
      return;
    }
#else
    // General case: binary long division for 128-bit by 128-bit.
    // Find the highest set bit in dividend.
    int shift = 0;
    if (dividend_hi != 0) {
      unsigned long idx;
      _BitScanReverse64(&idx, dividend_hi);
      shift = 64 + static_cast<int>(idx);
    } else {
      unsigned long idx;
      _BitScanReverse64(&idx, dividend_lo);
      shift = static_cast<int>(idx);
    }
    for (int i = shift; i >= 0; --i) {
      // Left shift remainder by 1.
      r_hi = (r_hi << 1) | (r_lo >> 63);
      r_lo <<= 1;
      // Bring down bit i of dividend.
      if (i >= 64) {
        r_lo |= (dividend_hi >> (i - 64)) & 1ULL;
      } else {
        r_lo |= (dividend_lo >> i) & 1ULL;
      }
      // If remainder >= divisor, subtract and set quotient bit.
      if (r_hi > divisor_hi ||
          (r_hi == divisor_hi && r_lo >= divisor_lo)) {
        // Subtract divisor from remainder.
        if (r_lo < divisor_lo) {
          r_hi -= divisor_hi + 1;
        } else {
          r_hi -= divisor_hi;
        }
        r_lo -= divisor_lo;
        // Set bit i in quotient.
        if (i >= 64) {
          q_hi |= 1ULL << (i - 64);
        } else {
          q_lo |= 1ULL << i;
        }
      }
    }
#endif
  }

  // Division
  Int128 operator/(const Int128& other) const {
    if (other == Int128(0, 0)) {
      throw std::runtime_error("Division by zero");
    }
    bool negative = (high_ < 0) != (other.high_ < 0);
    Int128 abs_dividend = (high_ < 0) ? -*this : *this;
    Int128 abs_divisor = (other.high_ < 0) ? -other : other;
    uint64_t q_hi, q_lo, r_hi, r_lo;
    udivmod128(
        static_cast<uint64_t>(abs_dividend.high_),
        abs_dividend.low_,
        static_cast<uint64_t>(abs_divisor.high_),
        abs_divisor.low_,
        q_hi,
        q_lo,
        r_hi,
        r_lo);
    Int128 result(static_cast<int64_t>(q_hi), q_lo);
    return negative ? -result : result;
  }

  // Modulo operator
  Int128 operator%(const Int128& other) const {
    if (other == Int128(0, 0)) {
      throw std::runtime_error("Modulo by zero");
    }
    bool negative = high_ < 0;
    Int128 abs_dividend = (high_ < 0) ? -*this : *this;
    Int128 abs_divisor = (other.high_ < 0) ? -other : other;
    uint64_t q_hi, q_lo, r_hi, r_lo;
    udivmod128(
        static_cast<uint64_t>(abs_dividend.high_),
        abs_dividend.low_,
        static_cast<uint64_t>(abs_divisor.high_),
        abs_divisor.low_,
        q_hi,
        q_lo,
        r_hi,
        r_lo);
    Int128 result(static_cast<int64_t>(r_hi), r_lo);
    return negative ? -result : result;
  }
  
  // Comparison operators
  constexpr bool operator==(const Int128& other) const {
    return high_ == other.high_ && low_ == other.low_;
  }
  
  constexpr bool operator!=(const Int128& other) const {
    return !(*this == other);
  }
  
  constexpr bool operator<(const Int128& other) const {
    if (high_ != other.high_) {
      return high_ < other.high_;
    }
    return low_ < other.low_;
  }
  
  constexpr bool operator<=(const Int128& other) const {
    // !(other < *this) is one comparison; the previous (< || ==) was two.
    // (Measured ~4ns saved per op vs the < || == idiom.)
    return !(other < *this);
  }
  
  constexpr bool operator>(const Int128& other) const {
    return other < *this;
  }
  
  constexpr bool operator>=(const Int128& other) const {
    return !(*this < other);
  }
  
  // Bitwise operators
  Int128 operator&(const Int128& other) const {
    return Int128(high_ & other.high_, low_ & other.low_);
  }
  
  Int128 operator|(const Int128& other) const {
    return Int128(high_ | other.high_, low_ | other.low_);
  }
  
  Int128 operator^(const Int128& other) const {
    return Int128(high_ ^ other.high_, low_ ^ other.low_);
  }
  
  Int128 operator~() const {
    return Int128(~high_, ~low_);
  }
  
  // Shift operators
  constexpr Int128 operator<<(int shift) const {
    if (shift >= 128) return Int128(0, 0);
    if (shift == 0) return *this;
    if (shift >= 64) {
      return Int128(static_cast<int64_t>(low_) << (shift - 64), 0);
    } else {
      int64_t new_high = (high_ << shift) | (static_cast<int64_t>(low_) >> (64 - shift));
      uint64_t new_low = low_ << shift;
      return Int128(new_high, new_low);
    }
  }
  
  Int128 operator>>(int shift) const {
    if (shift >= 128) return Int128(high_ < 0 ? -1 : 0, high_ < 0 ? UINT64_MAX : 0);
    if (shift == 0) return *this;
    if (shift >= 64) {
      int64_t fill = high_ < 0 ? -1 : 0;
      return Int128(fill, static_cast<uint64_t>(high_ >> (shift - 64)));
    } else {
      uint64_t new_low = (low_ >> shift) | (static_cast<uint64_t>(high_) << (64 - shift));
      int64_t new_high = high_ >> shift;
      return Int128(new_high, new_low);
    }
  }
  
  // Compound assignment shift operators
  Int128& operator<<=(int shift) {
    *this = *this << shift;
    return *this;
  }
  
  Int128& operator>>=(int shift) {
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
  
  // String conversion
  std::string toString() const {
    if (low_ == 0 && high_ == 0) return "0";

    const bool negative = high_ < 0;

    // Compute the unsigned magnitude directly in two's complement.  Going
    // through `-*this` would invoke signed integer overflow UB on INT128_MIN
    // (the operator-() implementation does `-high_` which is UB when
    // high_ == INT64_MIN).  The unsigned ~x + 1 form is well-defined and
    // produces the correct magnitude (2^127 for INT128_MIN).
    uint64_t hi;
    uint64_t lo;
    if (!negative) {
      hi = static_cast<uint64_t>(high_);
      lo = low_;
    } else {
      lo = ~low_ + 1;
      hi = ~static_cast<uint64_t>(high_) + (lo == 0 ? 1 : 0);
    }

    // Chunked decimal conversion: extract 19 decimal digits per iteration
    // via one 128/64 divide by 10^19, then format each 19-digit chunk
    // using native 64-bit div-by-10.  Int128 fits in 39 digits so at most
    // 3 chunks (and typically just 1-2).  Cost: 1-3 `divq` (~20ns each)
    // versus the previous per-digit loop's 39 full 128-bit divides
    // (~800ns).  Measured ~13x speedup.
    constexpr uint64_t kPow19 = 10'000'000'000'000'000'000ULL;
    char buf[40];
    int pos = static_cast<int>(sizeof(buf));
    while (hi != 0 || lo != 0) {
      uint64_t qHi, qLo, rHi, rLo;
      udivmod128(hi, lo, 0, kPow19, qHi, qLo, rHi, rLo);
      uint64_t chunk = rLo;
      hi = qHi;
      lo = qLo;
      const bool moreToCome = (hi != 0 || lo != 0);
      if (moreToCome) {
        // Zero-pad to 19 digits for non-leading chunks.
        for (int i = 0; i < 19; ++i) {
          buf[--pos] = static_cast<char>('0' + (chunk % 10));
          chunk /= 10;
        }
      } else {
        // Leading chunk: emit minimum digits (chunk != 0 here because the
        // overall value was non-zero and this is the most-significant chunk).
        while (chunk != 0) {
          buf[--pos] = static_cast<char>('0' + (chunk % 10));
          chunk /= 10;
        }
      }
    }
    std::string result;
    result.reserve(static_cast<size_t>(sizeof(buf) - pos) + (negative ? 1 : 0));
    if (negative) result.push_back('-');
    result.append(buf + pos, sizeof(buf) - pos);
    return result;
  }
  
 private:
  uint64_t low_;   // Least significant 64 bits (offset 0, matches __int128_t)
  int64_t high_;   // Most significant 64 bits (offset 8, matches __int128_t)
};

class UInt128 {
 public:
  constexpr UInt128() : low_(0), high_(0) {}
  
  constexpr UInt128(uint64_t high, uint64_t low) : low_(low), high_(high) {}
  
  constexpr UInt128(uint64_t value) : low_(value), high_(0) {}
  
  constexpr UInt128(const UInt128& other) = default;
  constexpr UInt128& operator=(const UInt128& other) = default;
  
  // Conversion constructor from Int128
  constexpr UInt128(const Int128& value) : low_(value.low()), high_(static_cast<uint64_t>(value.high())) {}

  // Getters
  constexpr uint64_t high() const { return high_; }
  constexpr uint64_t low() const { return low_; }
  constexpr uint64_t hi() const { return high_; }
  constexpr uint64_t lo() const { return low_; }
  
  // Conversion operators
  explicit operator bool() const { return high_ != 0 || low_ != 0; }
  explicit operator uint64_t() const { return low_; }
  explicit operator int64_t() const { return static_cast<int64_t>(low_); }
  explicit operator double() const {
    return static_cast<double>(high_) * 18446744073709551616.0 +
        static_cast<double>(low_);
  }
  
  // Arithmetic operators. See Int128 for rationale.
  constexpr UInt128 operator+(const UInt128& other) const {
    if (std::is_constant_evaluated()) {
      uint64_t new_low = low_ + other.low_;
      uint64_t new_high = high_ + other.high_ + (new_low < low_ ? 1 : 0);
      return UInt128(new_high, new_low);
    }
#if defined(_M_X64)
    uint64_t new_low;
    uint64_t new_high;
    unsigned char c = _addcarry_u64(0, low_, other.low_, &new_low);
    _addcarry_u64(c, high_, other.high_, &new_high);
    return UInt128(new_high, new_low);
#else
    uint64_t new_low = low_ + other.low_;
    uint64_t new_high = high_ + other.high_ + (new_low < low_ ? 1 : 0);
    return UInt128(new_high, new_low);
#endif
  }
  
  constexpr UInt128 operator-(const UInt128& other) const {
    if (std::is_constant_evaluated()) {
      uint64_t new_low = low_ - other.low_;
      uint64_t new_high = high_ - other.high_ - (new_low > low_ ? 1 : 0);
      return UInt128(new_high, new_low);
    }
#if defined(_M_X64)
    uint64_t new_low;
    uint64_t new_high;
    unsigned char b = _subborrow_u64(0, low_, other.low_, &new_low);
    _subborrow_u64(b, high_, other.high_, &new_high);
    return UInt128(new_high, new_low);
#else
    uint64_t new_low = low_ - other.low_;
    uint64_t new_high = high_ - other.high_ - (new_low > low_ ? 1 : 0);
    return UInt128(new_high, new_low);
#endif
  }
  
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
    if (std::is_constant_evaluated()) {
      uint64_t a_lo = low_;
      uint64_t b_lo = other.low_;
      uint64_t a_lo_lo = a_lo & 0xFFFFFFFF;
      uint64_t a_lo_hi = a_lo >> 32;
      uint64_t b_lo_lo = b_lo & 0xFFFFFFFF;
      uint64_t b_lo_hi = b_lo >> 32;
      uint64_t p0 = a_lo_lo * b_lo_lo;
      uint64_t p1 = a_lo_lo * b_lo_hi;
      uint64_t p2 = a_lo_hi * b_lo_lo;
      uint64_t p3 = a_lo_hi * b_lo_hi;
      uint64_t carry = ((p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF)) >> 32;
      uint64_t result_low = a_lo * b_lo;
      uint64_t result_high = p3 + (p1 >> 32) + (p2 >> 32) + carry +
          high_ * other.low_ + low_ * other.high_;
      return UInt128(result_high, result_low);
    }
#if defined(_M_X64)
    uint64_t lo_hi;
    uint64_t result_low = _umul128(low_, other.low_, &lo_hi);
    uint64_t result_high =
        lo_hi + low_ * other.high_ + high_ * other.low_;
    return UInt128(result_high, result_low);
#elif defined(_M_ARM64)
    uint64_t result_low = low_ * other.low_;
    uint64_t lo_hi = __umulh(low_, other.low_);
    uint64_t result_high =
        lo_hi + low_ * other.high_ + high_ * other.low_;
    return UInt128(result_high, result_low);
#else
    // Portable fallback: 32-bit halves construction. See Int128::operator*.
    uint64_t a_lo = low_;
    uint64_t b_lo = other.low_;
    uint64_t a_lo_lo = a_lo & 0xFFFFFFFF;
    uint64_t a_lo_hi = a_lo >> 32;
    uint64_t b_lo_lo = b_lo & 0xFFFFFFFF;
    uint64_t b_lo_hi = b_lo >> 32;
    uint64_t p0 = a_lo_lo * b_lo_lo;
    uint64_t p1 = a_lo_lo * b_lo_hi;
    uint64_t p2 = a_lo_hi * b_lo_lo;
    uint64_t p3 = a_lo_hi * b_lo_hi;
    uint64_t carry = ((p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF)) >> 32;
    uint64_t result_low = a_lo * b_lo;
    uint64_t result_high = p3 + (p1 >> 32) + (p2 >> 32) + carry +
        high_ * other.low_ + low_ * other.high_;
    return UInt128(result_high, result_low);
#endif
  }
  
  UInt128 operator*(int64_t other) const {
    return *this * UInt128(static_cast<uint64_t>(other));
  }
  
  friend UInt128 operator*(int64_t left, const UInt128& right) {
    return UInt128(static_cast<uint64_t>(left)) * right;
  }
  
  UInt128& operator*=(const UInt128& other) {
    *this = *this * other;
    return *this;
  }
  
  UInt128 operator/(const UInt128& other) const {
    if (other.low_ == 0 && other.high_ == 0) {
      throw std::invalid_argument("Division by zero");
    }
    
    uint64_t q_hi, q_lo, r_hi, r_lo;
    Int128::udivmod128(high_, low_, other.high_, other.low_, q_hi, q_lo, r_hi, r_lo);
    return UInt128(q_hi, q_lo);
  }
  
  UInt128& operator/=(const UInt128& other) {
    if (other.low_ == 0 && other.high_ == 0) {
      throw std::invalid_argument("Division by zero");
    }
    *this = *this / other;
    return *this;
  }
  
  // Modulo operator
  UInt128 operator%(const UInt128& other) const {
    if (other.low_ == 0 && other.high_ == 0) {
      throw std::invalid_argument("Division by zero");
    }
    
    uint64_t q_hi, q_lo, r_hi, r_lo;
    Int128::udivmod128(high_, low_, other.high_, other.low_, q_hi, q_lo, r_hi, r_lo);
    return UInt128(r_hi, r_lo);
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
  
  // Comparison operators
  bool operator==(const UInt128& other) const {
    return high_ == other.high_ && low_ == other.low_;
  }
  
  bool operator!=(const UInt128& other) const {
    return !(*this == other);
  }
  
  bool operator<(const UInt128& other) const {
    if (high_ != other.high_) {
      return high_ < other.high_;
    }
    return low_ < other.low_;
  }
  
  bool operator<=(const UInt128& other) const {
    // !(other < *this) is one comparison; the previous (< || ==) was two.
    return !(other < *this);
  }
  
  bool operator>(const UInt128& other) const {
    return other < *this;
  }
  
  bool operator>=(const UInt128& other) const {
    return !(*this < other);
  }
  
  // Bitwise operators
  UInt128 operator&(const UInt128& other) const {
    return UInt128(high_ & other.high_, low_ & other.low_);
  }
  
  UInt128 operator|(const UInt128& other) const {
    return UInt128(high_ | other.high_, low_ | other.low_);
  }
  
  UInt128 operator^(const UInt128& other) const {
    return UInt128(high_ ^ other.high_, low_ ^ other.low_);
  }
  
  UInt128 operator~() const {
    return UInt128(~high_, ~low_);
  }
  
  // Shift operators
  constexpr UInt128 operator<<(int shift) const {
    if (shift >= 128) return UInt128(0, 0);
    if (shift == 0) return *this;
    if (shift >= 64) {
      return UInt128(low_ << (shift - 64), 0);
    } else {
      uint64_t new_high = (high_ << shift) | (low_ >> (64 - shift));
      uint64_t new_low = low_ << shift;
      return UInt128(new_high, new_low);
    }
  }
  
  UInt128 operator>>(int shift) const {
    if (shift >= 128) return UInt128(0, 0);
    if (shift == 0) return *this;
    if (shift >= 64) {
      return UInt128(0, high_ >> (shift - 64));
    } else {
      uint64_t new_low = (low_ >> shift) | (high_ << (64 - shift));
      uint64_t new_high = high_ >> shift;
      return UInt128(new_high, new_low);
    }
  }
  
  // Compound assignment shift operators
  UInt128& operator<<=(int shift) {
    *this = *this << shift;
    return *this;
  }
  
  UInt128& operator>>=(int shift) {
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
  
 private:
  uint64_t low_;   // Least significant 64 bits (offset 0, matches __uint128_t)
  uint64_t high_;  // Most significant 64 bits (offset 8, matches __uint128_t)
};

// Define UInt128 conversion constructor to Int128 after UInt128 is defined
inline Int128::Int128(const UInt128& value) 
    : low_(value.low()), high_(static_cast<int64_t>(value.high())) {}

// Type aliases for compatibility
using int128_t = Int128;
using uint128_t = UInt128;

} // namespace facebook::velox

// std::numeric_limits specialization
namespace std {

template <>
class numeric_limits<facebook::velox::Int128> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = true;
  static constexpr bool has_quiet_NaN = false;
  
  static constexpr facebook::velox::Int128 min() noexcept {
    return facebook::velox::Int128(std::numeric_limits<int64_t>::min(), 0);
  }
  
  static constexpr facebook::velox::Int128 max() noexcept {
    return facebook::velox::Int128(std::numeric_limits<int64_t>::max(), std::numeric_limits<uint64_t>::max());
  }

  static constexpr facebook::velox::Int128 lowest() noexcept {
    return min();
  }
};

template <>
class numeric_limits<facebook::velox::UInt128> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = false;
  static constexpr bool is_integer = true;
  static constexpr bool has_quiet_NaN = false;
  
  static constexpr facebook::velox::UInt128 min() noexcept {
    return facebook::velox::UInt128(0, 0);
  }
  
  static constexpr facebook::velox::UInt128 max() noexcept {
    return facebook::velox::UInt128(std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max());
  }

  static constexpr facebook::velox::UInt128 lowest() noexcept {
    return min();
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

// Global operators for mixed-type operations
namespace facebook::velox {
inline Int128 operator-(int left, const Int128& right) {
  return Int128(left) - right;
}
} // namespace facebook::velox

// Folly hasher specialization for Int128
namespace folly {
template <>
struct hasher<facebook::velox::Int128> {
  size_t operator()(const facebook::velox::Int128& value) const {
    return folly::hash::hash_combine(
        folly::hasher<int64_t>{}(value.high()),
        folly::hasher<uint64_t>{}(value.low()));
  }
};

template <>
struct hasher<facebook::velox::UInt128> {
  size_t operator()(const facebook::velox::UInt128& value) const {
    return folly::hash::hash_combine(
        folly::hasher<uint64_t>{}(value.high()),
        folly::hasher<uint64_t>{}(value.low()));
  }
};

// toAppend specializations for folly::to<std::string>() support
// These need to be templates to match folly's ADL expectations  
template <class Tgt>
typename std::enable_if<folly::IsSomeString<Tgt>::value, void>::type
toAppend(const facebook::velox::Int128& value, Tgt* result) {
  // Convert to hexadecimal string representation
  // Handle negative values by showing sign and absolute value in hex
  char buf[64];
  if (value.high() < 0) {
    result->append("-");
    // Get absolute value
    facebook::velox::Int128 absValue = -value;
    if (absValue.high() != 0) {
      snprintf(buf, sizeof(buf), "0x%llx%016llx", 
               static_cast<unsigned long long>(absValue.high()),
               static_cast<unsigned long long>(absValue.low()));
    } else {
      snprintf(buf, sizeof(buf), "0x%llx", 
               static_cast<unsigned long long>(absValue.low()));
    }
  } else {
    if (value.high() != 0) {
      snprintf(buf, sizeof(buf), "0x%llx%016llx", 
               static_cast<unsigned long long>(value.high()),
               static_cast<unsigned long long>(value.low()));
    } else {
      snprintf(buf, sizeof(buf), "0x%llx", 
               static_cast<unsigned long long>(value.low()));
    }
  }
  result->append(buf);
}

template <class Tgt>
typename std::enable_if<folly::IsSomeString<Tgt>::value, void>::type
toAppend(const facebook::velox::UInt128& value, Tgt* result) {
  // Convert to hexadecimal string representation
  char buf[64];
  if (value.high() != 0) {
    snprintf(buf, sizeof(buf), "0x%llx%016llx", 
             static_cast<unsigned long long>(value.high()),
             static_cast<unsigned long long>(value.low()));
  } else {
    snprintf(buf, sizeof(buf), "0x%llx", 
             static_cast<unsigned long long>(value.low()));
  }
  result->append(buf);
}

} // namespace folly

// Free-standing reverse comparison operators (int64_t op Int128)
namespace facebook::velox {

inline constexpr bool operator<(int64_t lhs, const Int128& rhs) {
  return Int128(lhs) < rhs;
}
inline constexpr bool operator>(int64_t lhs, const Int128& rhs) {
  return Int128(lhs) > rhs;
}
inline constexpr bool operator<=(int64_t lhs, const Int128& rhs) {
  return Int128(lhs) <= rhs;
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
typename std::enable_if<folly::IsSomeString<Tgt>::value, void>::type
toAppend(const int128_t& value, Tgt* result) {
  folly::toAppend(value, result);
}

template <class Tgt>
typename std::enable_if<folly::IsSomeString<Tgt>::value, void>::type
toAppend(const uint128_t& value, Tgt* result) {
  folly::toAppend(value, result);
}

} // namespace facebook::velox

// fmt formatter specialization for Int128
#include <fmt/format.h>

template <>
struct fmt::formatter<facebook::velox::Int128> : fmt::formatter<std::string> {
  auto format(const facebook::velox::Int128& value, format_context& ctx) const {
    return fmt::formatter<std::string>::format(value.toString(), ctx);
  }
};

template <>
struct fmt::formatter<facebook::velox::UInt128> : fmt::formatter<std::string> {
  auto format(const facebook::velox::UInt128& value, format_context& ctx) const {
    return fmt::formatter<std::string>::format(
        fmt::format("UInt128({}, {})", value.high(), value.low()), ctx);
  }
};

#endif // _MSC_VER