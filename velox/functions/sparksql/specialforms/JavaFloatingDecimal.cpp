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

#include "velox/functions/sparksql/specialforms/JavaFloatingDecimal.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <fmt/format.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Names and structure below follow jdk.internal.math.FloatingDecimal so the
// two can be diffed; see appendJavaDouble for what the port guarantees.

constexpr int kExponentShift{52};
constexpr int kExponentBias{1'023};
constexpr uint64_t kSignificandMask{(uint64_t{1} << kExponentShift) - 1};
constexpr int kMaxSmallBinaryExponent{62};
constexpr int kMinSmallBinaryExponent{-21};

// ceil(log2(5^i)).
constexpr std::array<int, 27> kFiveBits{0,  3,  5,  7,  10, 12, 14, 17, 19,
                                        21, 24, 26, 28, 31, 33, 35, 38, 40,
                                        42, 45, 47, 49, 52, 54, 56, 59, 61};

constexpr std::array<uint64_t, 27> kFivePow{
    1ULL,
    5ULL,
    25ULL,
    125ULL,
    625ULL,
    3'125ULL,
    15'625ULL,
    78'125ULL,
    390'625ULL,
    1'953'125ULL,
    9'765'625ULL,
    48'828'125ULL,
    244'140'625ULL,
    1'220'703'125ULL,
    6'103'515'625ULL,
    30'517'578'125ULL,
    152'587'890'625ULL,
    762'939'453'125ULL,
    3'814'697'265'625ULL,
    19'073'486'328'125ULL,
    95'367'431'640'625ULL,
    476'837'158'203'125ULL,
    2'384'185'791'015'625ULL,
    11'920'928'955'078'125ULL,
    59'604'644'775'390'625ULL,
    298'023'223'876'953'125ULL,
    1'490'116'119'384'765'625ULL};

// Number of digits of 1L << index that carry no information, i.e.
// insignificantDigits(1L << index) in FloatingDecimal.
constexpr std::array<int, 64> kInsignificantDigitsForPow2{
    0,  0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,  3,  4,  4,
    4,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  8,  8,  8,  9,  9,
    9,  9,  10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14,
    14, 14, 15, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19};

int insignificantDigitsForPow2(int power) {
  if (power > 1 &&
      power < static_cast<int>(kInsignificantDigitsForPow2.size())) {
    return kInsignificantDigitsForPow2[power];
  }
  return 0;
}

// The JDK computes with Java's wrapping signed arithmetic, and the wrap is
// observable: for values in [2^84, 2^85) the sum 'b + m' overflows, the
// stopping test reads false, and the digits printed differ from the ones the
// unwrapped comparison would produce.
template <typename T>
T wrappingAdd(T lhs, T rhs) {
  using Unsigned = std::make_unsigned_t<T>;
  return static_cast<T>(
      static_cast<Unsigned>(lhs) + static_cast<Unsigned>(rhs));
}

template <typename T>
T wrappingSubtract(T lhs, T rhs) {
  using Unsigned = std::make_unsigned_t<T>;
  return static_cast<T>(
      static_cast<Unsigned>(lhs) - static_cast<Unsigned>(rhs));
}

template <typename T>
T wrappingMultiply(T lhs, T rhs) {
  using Unsigned = std::make_unsigned_t<T>;
  return static_cast<T>(
      static_cast<Unsigned>(lhs) * static_cast<Unsigned>(rhs));
}

template <typename T>
T wrappingShiftLeft(T value, int bits) {
  using Unsigned = std::make_unsigned_t<T>;
  VELOX_DCHECK_LT(bits, static_cast<int>(sizeof(T) * 8));
  return static_cast<T>(static_cast<Unsigned>(value) << bits);
}

// Decimal form of a double: value == 0.<digits> * 10^decimalExponent.
struct DecimalDigits {
  // FloatingDecimal sizes its own buffer at 20 for doubles; the longest
  // sequence this port produces is 19, from the integer path.
  std::array<char, 24> digits{};
  int size{0};
  int decimalExponent{0};

  void push(int digit) {
    VELOX_DCHECK_LT(size, static_cast<int>(digits.size()));
    digits[size++] = static_cast<char>('0' + digit);
  }
};

// Adds one to the last digit, carrying into the exponent when every digit is
// a nine.
void roundUp(DecimalDigits& result) {
  int index = result.size - 1;
  if (result.digits[index] == '9') {
    while (result.digits[index] == '9' && index > 0) {
      result.digits[index] = '0';
      --index;
    }
    if (result.digits[index] == '9') {
      result.decimalExponent += 1;
      result.digits[0] = '1';
      return;
    }
  }
  ++result.digits[index];
}

// A first guess at floor(log10(value)) + 1, derived from the binary exponent.
// The guess is never too low, and when it is one too high the first quotient
// digit comes out zero and the caller corrects it.
int estimateDecimalExponent(uint64_t fractBits, int binaryExponent) {
  const double mantissa = std::bit_cast<double>(
      (static_cast<uint64_t>(kExponentBias) << kExponentShift) |
      (fractBits & kSignificandMask));
  const double estimate = (mantissa - 1.5) * 0.289529654 + 0.176091259 +
      static_cast<double>(binaryExponent) * 0.301029995663981;
  const auto estimateBits = std::bit_cast<uint64_t>(estimate);
  const int exponent =
      static_cast<int>((estimateBits >> kExponentShift) & 0x7FF) -
      kExponentBias;
  const bool isNegative = (estimateBits >> 63) != 0;
  if (exponent >= 0 && exponent < 52) {
    const uint64_t mask = kSignificandMask >> exponent;
    const int truncated = static_cast<int>(
        ((estimateBits & kSignificandMask) | (uint64_t{1} << kExponentShift)) >>
        (kExponentShift - exponent));
    if (isNegative) {
      return (mask & estimateBits) == 0 ? -truncated : -truncated - 1;
    }
    return truncated;
  }
  if (exponent < 0) {
    if ((estimateBits & ~(uint64_t{1} << 63)) == 0) {
      return 0;
    }
    return isNegative ? -1 : 0;
  }
  return static_cast<int>(estimate);
}

// Renders an integral value whose low 'insignificantDigits' digits carry no
// information, rounding them away first.
DecimalDigits developLongDigits(uint64_t value, int insignificantDigits) {
  int decimalExponent = 0;
  if (insignificantDigits != 0) {
    const uint64_t powerOfTen = kFivePow[insignificantDigits]
        << insignificantDigits;
    const uint64_t residue = value % powerOfTen;
    value /= powerOfTen;
    decimalExponent += insignificantDigits;
    if (residue >= (powerOfTen >> 1)) {
      ++value;
    }
  }

  std::array<char, 24> buffer{};
  const auto written =
      fmt::format_to_n(buffer.data(), buffer.size(), "{}", value);
  int length = static_cast<int>(written.out - buffer.data());
  while (length > 1 && buffer[length - 1] == '0') {
    --length;
    ++decimalExponent;
  }

  DecimalDigits result;
  std::memcpy(result.digits.data(), buffer.data(), length);
  result.size = length;
  result.decimalExponent = decimalExponent + length;
  return result;
}

// The quotient-digit loop over native integers, used when B, M and 10*S all
// fit in 'Integer'. 'Integer' is int32_t or int64_t, matching the two native
// branches of FloatingDecimal, and every operation wraps the way Java's does.
template <typename Integer>
DecimalDigits generateDigitsNative(
    uint64_t fractBits,
    int decimalExponent,
    int b5,
    int b2,
    int s5,
    int s2,
    int m5,
    int m2) {
  const auto pow5 = [](int power) {
    return static_cast<Integer>(kFivePow[power]);
  };
  Integer b = wrappingShiftLeft(
      wrappingMultiply(static_cast<Integer>(fractBits), pow5(b5)), b2);
  const Integer s = wrappingShiftLeft(pow5(s5), s2);
  Integer m = wrappingShiftLeft(pow5(m5), m2);
  const Integer tens = wrappingMultiply(s, static_cast<Integer>(10));

  DecimalDigits result;
  int digit = static_cast<int>(b / s);
  b = wrappingMultiply(static_cast<Integer>(10), b % s);
  m = wrappingMultiply(m, static_cast<Integer>(10));
  bool low = b < m;
  bool high = wrappingAdd(b, m) > tens;
  if (digit == 0 && !high) {
    --decimalExponent;
  } else {
    result.push(digit);
  }

  // Scientific notation always prints a digit after the point, so the loop
  // keeps going for at least two digits.
  if (decimalExponent < -3 || decimalExponent >= 8) {
    high = false;
    low = false;
  }
  while (!low && !high) {
    digit = static_cast<int>(b / s);
    b = wrappingMultiply(static_cast<Integer>(10), b % s);
    m = wrappingMultiply(m, static_cast<Integer>(10));
    if (m > 0) {
      low = b < m;
      high = wrappingAdd(b, m) > tens;
    } else {
      // m overflowed, so it exceeds both b and tens - b.
      low = true;
      high = true;
    }
    result.push(digit);
  }

  result.decimalExponent = decimalExponent + 1;
  if (high) {
    if (!low) {
      roundUp(result);
    } else {
      const Integer lowDigitDifference =
          wrappingSubtract(wrappingShiftLeft(b, 1), tens);
      if (lowDigitDifference > 0) {
        roundUp(result);
      } else if (
          lowDigitDifference == 0 &&
          ((result.digits[result.size - 1] - '0') & 1) != 0) {
        roundUp(result);
      }
    }
  }
  return result;
}

// Fixed-capacity unsigned integer, little-endian in 64-bit limbs, holding the
// B, S, M and 10*S of the general case. Those reach 816 bits for the widest
// double, so the capacity below leaves room for several more limbs.
class WideInteger {
 public:
  void setToPow5(int power5) {
    limbs_[0] = 1;
    size_ = 1;
    for (; power5 >= kPow5StepSize; power5 -= kPow5StepSize) {
      multiplyBy(kPow5PerStep);
    }
    if (power5 > 0) {
      multiplyBy(kFivePow[power5]);
    }
  }

  void multiplyBy(uint64_t multiplier) {
    uint64_t carry = 0;
    for (int i = 0; i < size_; ++i) {
      const unsigned __int128 product =
          static_cast<unsigned __int128>(limbs_[i]) * multiplier + carry;
      limbs_[i] = static_cast<uint64_t>(product);
      carry = static_cast<uint64_t>(product >> 64);
    }
    if (carry != 0) {
      VELOX_DCHECK_LT(size_, kMaxLimbs);
      limbs_[size_++] = carry;
    }
  }

  void shiftLeftBy(int bits) {
    VELOX_DCHECK_GE(bits, 0);
    if (size_ == 0 || bits == 0) {
      return;
    }
    const int limbShift = bits / 64;
    const int bitShift = bits % 64;
    const int newSize = size_ + limbShift +
        (bitShift != 0 && (limbs_[size_ - 1] >> (64 - bitShift)) != 0 ? 1 : 0);
    VELOX_DCHECK_LE(newSize, kMaxLimbs);
    for (int i = newSize - 1; i >= 0; --i) {
      const int source = i - limbShift;
      uint64_t limb = 0;
      if (source >= 0 && source < size_) {
        limb = bitShift == 0 ? limbs_[source] : (limbs_[source] << bitShift);
      }
      if (bitShift != 0 && source >= 1 && source - 1 < size_) {
        limb |= limbs_[source - 1] >> (64 - bitShift);
      }
      limbs_[i] = limb;
    }
    size_ = newSize;
  }

  int compare(const WideInteger& other) const {
    if (size_ != other.size_) {
      return size_ < other.size_ ? -1 : 1;
    }
    for (int i = size_ - 1; i >= 0; --i) {
      if (limbs_[i] != other.limbs_[i]) {
        return limbs_[i] < other.limbs_[i] ? -1 : 1;
      }
    }
    return 0;
  }

  // Returns the sign of 'this' - ('lhs' + 'rhs'), adding the two operands
  // limb by limb rather than materializing the sum. Walking upwards and
  // keeping the last disagreement leaves the most significant one.
  int compareToSum(const WideInteger& lhs, const WideInteger& rhs) const {
    const int limit = std::max({size_, lhs.size_, rhs.size_}) + 1;
    VELOX_DCHECK_LE(limit, kMaxLimbs);
    int result = 0;
    uint64_t carry = 0;
    for (int i = 0; i < limit; ++i) {
      const unsigned __int128 sum =
          static_cast<unsigned __int128>(lhs.limbAt(i)) + rhs.limbAt(i) + carry;
      const auto sumLimb = static_cast<uint64_t>(sum);
      carry = static_cast<uint64_t>(sum >> 64);
      const uint64_t limb = limbAt(i);
      if (limb != sumLimb) {
        result = limb < sumLimb ? -1 : 1;
      }
    }
    return result;
  }

  void subtract(const WideInteger& other) {
    VELOX_DCHECK_GE(compare(other), 0);
    uint64_t borrow = 0;
    for (int i = 0; i < size_; ++i) {
      const uint64_t right = other.limbAt(i);
      const uint64_t left = limbs_[i];
      const uint64_t difference = left - right - borrow;
      borrow = (left < right || (left == right && borrow != 0)) ? 1 : 0;
      limbs_[i] = difference;
    }
    trim();
  }

  // Takes the next quotient digit of 'this' divided by 'divisor', leaving
  // 'this' holding ten times the remainder, like FDBigInteger's
  // quoRemIteration. The caller's invariants keep the digit below ten.
  int takeQuotientDigit(const WideInteger& divisor) {
    int quotient = 0;
    while (compare(divisor) >= 0) {
      subtract(divisor);
      ++quotient;
    }
    VELOX_DCHECK_LT(quotient, 10);
    multiplyBy(10);
    return quotient;
  }

 private:
  // 5^27 is the largest power of five that fits in a limb.
  static constexpr uint64_t kPow5PerStep{7'450'580'596'923'828'125ULL};
  static constexpr int kPow5StepSize{27};
  // B, S, M and 10*S reach 816 bits for the widest double, and compareToSum
  // walks one limb past the longest operand.
  static constexpr int kMaxLimbs{18};

  uint64_t limbAt(int index) const {
    return index < size_ ? limbs_[index] : 0;
  }

  void trim() {
    while (size_ > 0 && limbs_[size_ - 1] == 0) {
      --size_;
    }
  }

  std::array<uint64_t, kMaxLimbs> limbs_{};
  int size_{0};
};

// The quotient-digit loop for values whose B, M and 10*S exceed 64 bits.
DecimalDigits generateDigitsWide(
    uint64_t fractBits,
    int decimalExponent,
    int b5,
    int b2,
    int s5,
    int s2,
    int m5,
    int m2) {
  // B and M share the same power of five, and 10*S is S scaled by ten, so two
  // powers cover all four operands.
  VELOX_DCHECK_EQ(b5, m5);
  WideInteger powerOfFive;
  powerOfFive.setToPow5(b5);
  WideInteger b = powerOfFive;
  b.multiplyBy(fractBits);
  b.shiftLeftBy(b2);
  // M and 10*S arrive pre-multiplied by ten, matching FloatingDecimal.
  WideInteger m = powerOfFive;
  m.multiplyBy(5);
  m.shiftLeftBy(m2 + 1);
  WideInteger s;
  s.setToPow5(s5);
  s.shiftLeftBy(s2);
  WideInteger tenS = s;
  tenS.multiplyBy(10);

  DecimalDigits result;
  int digit = b.takeQuotientDigit(s);
  bool low = b.compare(m) < 0;
  bool high = tenS.compareToSum(b, m) <= 0;
  if (digit == 0 && !high) {
    --decimalExponent;
  } else {
    result.push(digit);
  }

  if (decimalExponent < -3 || decimalExponent >= 8) {
    high = false;
    low = false;
  }
  while (!low && !high) {
    digit = b.takeQuotientDigit(s);
    m.multiplyBy(10);
    low = b.compare(m) < 0;
    high = tenS.compareToSum(b, m) <= 0;
    result.push(digit);
  }

  result.decimalExponent = decimalExponent + 1;
  if (high) {
    if (!low) {
      roundUp(result);
    } else {
      b.shiftLeftBy(1);
      const int lowDigitDifference = b.compare(tenS);
      if (lowDigitDifference > 0) {
        roundUp(result);
      } else if (
          lowDigitDifference == 0 &&
          ((result.digits[result.size - 1] - '0') & 1) != 0) {
        roundUp(result);
      }
    }
  }
  return result;
}

DecimalDigits toDecimalDigits(double value) {
  const auto bits = std::bit_cast<uint64_t>(value);
  const auto exponentBits = static_cast<int>((bits >> kExponentShift) & 0x7FF);
  uint64_t fractBits = bits & kSignificandMask;
  int binaryExponent;
  int significantBits;
  if (exponentBits == 0) {
    const int leadingZeros = std::countl_zero(fractBits);
    const int shift = leadingZeros - (63 - kExponentShift);
    fractBits <<= shift;
    binaryExponent = 1 - shift;
    significantBits = 64 - leadingZeros;
  } else {
    fractBits |= uint64_t{1} << kExponentShift;
    binaryExponent = exponentBits;
    significantBits = kExponentShift + 1;
  }
  binaryExponent -= kExponentBias;

  const int tailZeros = std::countr_zero(fractBits);
  const int fractionBits = kExponentShift + 1 - tailZeros;
  const int tinyBits = std::max(0, fractionBits - binaryExponent - 1);

  // Integers in this range convert through plain 64-bit arithmetic, and the
  // digits beyond the precision of the double are rounded away rather than
  // dropped, which is what makes 2^53 and up print more digits than the
  // shortest round-trip form would.
  if (binaryExponent <= kMaxSmallBinaryExponent &&
      binaryExponent >= kMinSmallBinaryExponent && tinyBits == 0) {
    const int insignificant = binaryExponent > significantBits
        ? insignificantDigitsForPow2(binaryExponent - significantBits - 1)
        : 0;
    const uint64_t integer = binaryExponent >= kExponentShift
        ? fractBits << (binaryExponent - kExponentShift)
        : fractBits >> (kExponentShift - binaryExponent);
    return developLongDigits(integer, insignificant);
  }

  int decimalExponent = estimateDecimalExponent(fractBits, binaryExponent);
  int b5 = std::max(0, -decimalExponent);
  int b2 = b5 + tinyBits + binaryExponent;
  int s5 = std::max(0, decimalExponent);
  int s2 = s5 + tinyBits;
  const int m5 = b5;
  int m2 = b2 - significantBits;

  fractBits >>= tailZeros;
  b2 -= fractionBits - 1;
  const int commonFactor = std::min(b2, s2);
  b2 -= commonFactor;
  s2 -= commonFactor;
  m2 -= commonFactor;

  // Below an exact power of two the neighbouring double is only half as far
  // away, so the acceptance interval shrinks on that side.
  if (fractionBits == 1) {
    m2 -= 1;
  }
  if (m2 < 0) {
    b2 -= m2;
    s2 -= m2;
    m2 = 0;
  }

  const int bBits = fractionBits + b2 +
      (b5 < static_cast<int>(kFiveBits.size()) ? kFiveBits[b5] : b5 * 3);
  const int tenSBits = s2 + 1 +
      (s5 + 1 < static_cast<int>(kFiveBits.size()) ? kFiveBits[s5 + 1]
                                                   : (s5 + 1) * 3);
  if (bBits < 32 && tenSBits < 32) {
    return generateDigitsNative<int32_t>(
        fractBits, decimalExponent, b5, b2, s5, s2, m5, m2);
  }
  if (bBits < 64 && tenSBits < 64) {
    return generateDigitsNative<int64_t>(
        fractBits, decimalExponent, b5, b2, s5, s2, m5, m2);
  }
  return generateDigitsWide(fractBits, decimalExponent, b5, b2, s5, s2, m5, m2);
}

// FloatingDecimal.getChars: plain notation for exponents in (-3, 8], and
// scientific notation otherwise, always with at least one digit after the
// decimal point.
void appendDecimalDigits(
    const DecimalDigits& value,
    bool isNegative,
    std::string& output) {
  if (isNegative) {
    output.push_back('-');
  }
  const std::string_view digits{
      value.digits.data(), static_cast<size_t>(value.size)};
  const int decimalExponent = value.decimalExponent;
  if (decimalExponent > 0 && decimalExponent < 8) {
    const int whole = std::min(value.size, decimalExponent);
    output.append(digits.substr(0, whole));
    if (whole < decimalExponent) {
      output.append(decimalExponent - whole, '0');
      output.append(".0");
    } else {
      output.push_back('.');
      if (whole < value.size) {
        output.append(digits.substr(whole));
      } else {
        output.push_back('0');
      }
    }
  } else if (decimalExponent <= 0 && decimalExponent > -3) {
    output.append("0.");
    output.append(-decimalExponent, '0');
    output.append(digits);
  } else {
    output.push_back(digits.front());
    output.push_back('.');
    if (value.size > 1) {
      output.append(digits.substr(1));
    } else {
      output.push_back('0');
    }
    output.push_back('E');
    fmt::format_to(std::back_inserter(output), "{}", decimalExponent - 1);
  }
}

} // namespace

void appendJavaDouble(double value, std::string& output) {
  VELOX_DCHECK(std::isfinite(value));
  const bool isNegative = std::signbit(value);
  if (value == 0) {
    output.append(isNegative ? "-0.0" : "0.0");
    return;
  }
  appendDecimalDigits(toDecimalDigits(std::abs(value)), isNegative, output);
}

} // namespace facebook::velox::functions::sparksql
