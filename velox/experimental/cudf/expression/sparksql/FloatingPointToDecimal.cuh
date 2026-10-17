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
#include <cstring>

#ifdef __CUDACC__
#define VELOX_CUDF_CAST_HOST_DEVICE __host__ __device__
#else
#define VELOX_CUDF_CAST_HOST_DEVICE
#endif

namespace facebook::velox::cudf_velox::sparksql::detail {

// Decimal casts only need the finite interval [1e-39, 1e38). Smaller values
// round to zero at every supported scale; larger values overflow precision 38.
// Exact digit generation in this interval needs fewer than 256 bits, including
// the numerator, denominator, rounding margins, and at most 18 output digits.
// This deliberately small integer type implements only those operations.
struct CastUInt256 {
  uint64_t words[4]{};

  VELOX_CUDF_CAST_HOST_DEVICE explicit CastUInt256(uint64_t value = 0) {
    words[0] = value;
  }

  VELOX_CUDF_CAST_HOST_DEVICE int32_t compare(const CastUInt256& other) const {
    for (int32_t i = 3; i >= 0; --i) {
      if (words[i] != other.words[i]) {
        return words[i] < other.words[i] ? -1 : 1;
      }
    }
    return 0;
  }

  VELOX_CUDF_CAST_HOST_DEVICE void multiply(uint32_t factor) {
    unsigned __int128 carry = 0;
    for (auto& word : words) {
      const unsigned __int128 product =
          static_cast<unsigned __int128>(word) * factor + carry;
      word = static_cast<uint64_t>(product);
      carry = product >> 64;
    }
  }

  VELOX_CUDF_CAST_HOST_DEVICE void add(const CastUInt256& other) {
    unsigned __int128 carry = 0;
    for (int32_t i = 0; i < 4; ++i) {
      const unsigned __int128 sum =
          static_cast<unsigned __int128>(words[i]) + other.words[i] + carry;
      words[i] = static_cast<uint64_t>(sum);
      carry = sum >> 64;
    }
  }

  VELOX_CUDF_CAST_HOST_DEVICE void subtract(const CastUInt256& other) {
    uint64_t borrow = 0;
    for (int32_t i = 0; i < 4; ++i) {
      const uint64_t subtrahend = other.words[i] + borrow;
      const bool nextBorrow =
          subtrahend < other.words[i] || words[i] < subtrahend;
      words[i] -= subtrahend;
      borrow = nextBorrow;
    }
  }

  VELOX_CUDF_CAST_HOST_DEVICE void shiftLeft(int32_t bits) {
    CastUInt256 shifted;
    const int32_t whole = bits / 64;
    const int32_t partial = bits % 64;
    for (int32_t i = whole; i < 4; ++i) {
      shifted.words[i] = words[i - whole] << partial;
      if (partial != 0 && i > whole) {
        shifted.words[i] |= words[i - whole - 1] >> (64 - partial);
      }
    }
    *this = shifted;
  }

  VELOX_CUDF_CAST_HOST_DEVICE void shiftRight(int32_t bits) {
    CastUInt256 shifted;
    const int32_t whole = bits / 64;
    const int32_t partial = bits % 64;
    for (int32_t i = 0; i + whole < 4; ++i) {
      shifted.words[i] = words[i + whole] >> partial;
      if (partial != 0 && i + whole + 1 < 4) {
        shifted.words[i] |= words[i + whole + 1] << (64 - partial);
      }
    }
    *this = shifted;
  }

  VELOX_CUDF_CAST_HOST_DEVICE int32_t bitWidth() const {
    for (int32_t i = 3; i >= 0; --i) {
      if (words[i] != 0) {
#ifdef __CUDA_ARCH__
        return i * 64 + 64 - __clzll(words[i]);
#else
        return i * 64 + 64 - __builtin_clzll(words[i]);
#endif
      }
    }
    return 0;
  }

  VELOX_CUDF_CAST_HOST_DEVICE int32_t trailingZeros() const {
    for (int32_t i = 0; i < 4; ++i) {
      if (words[i] != 0) {
#ifdef __CUDA_ARCH__
        return i * 64 + __ffsll(static_cast<long long>(words[i])) - 1;
#else
        return i * 64 + __builtin_ctzll(words[i]);
#endif
      }
    }
    return 256;
  }
};

struct CastFloatingDecimal {
  uint64_t significand;
  int32_t exponent;
};

VELOX_CUDF_CAST_HOST_DEVICE inline void scaledFraction(
    uint64_t significand,
    int32_t exponent,
    int32_t decimalExponent,
    CastUInt256& numerator,
    CastUInt256& denominator,
    CastUInt256& margin) {
  numerator = CastUInt256(significand);
  numerator.shiftLeft(2);
  denominator = CastUInt256(1);
  // Java 8/11/17 uses the narrower spacing at powers of two on both sides of
  // the conversion interval. Other normal doubles have equal adjacent spacing.
  margin = CastUInt256(significand == (uint64_t{1} << 52) ? 1 : 2);
  if (exponent >= 0) {
    numerator.shiftLeft(exponent);
    margin.shiftLeft(exponent);
  } else {
    denominator.shiftLeft(-exponent);
  }
  if (decimalExponent >= 0) {
    for (int32_t i = 0; i < decimalExponent; ++i) {
      denominator.multiply(10);
    }
  } else {
    for (int32_t i = 0; i < -decimalExponent; ++i) {
      numerator.multiply(10);
      margin.multiply(10);
    }
  }
}

// Generate the decimal significand used by Java 8/11/17 Double.toString, which
// Spark feeds to BigDecimal before applying the requested decimal scale. This
// is integer interval digit generation, not binary rounding at the final scale.
// In particular, 1.2344499999999998 must not first become 1.23445, and 1e23's
// binary double is represented as 9.999999999999999e22 on these Java versions.
VELOX_CUDF_CAST_HOST_DEVICE inline CastFloatingDecimal floatingDecimal(
    uint64_t absoluteBits) {
  const auto binaryExponent = static_cast<int32_t>(absoluteBits >> 52) - 1023;
  const uint64_t significand =
      (absoluteBits & 0xfffffffffffffULL) | (uint64_t{1} << 52);
  const int32_t valueExponent = binaryExponent - 52;

  // Java's integer conversion path preserves more digits than a shortest
  // representation. Above 2^58 and 2^61 it rounds off one and two decimal
  // places, respectively. The exact integer and its rounding fit uint64.
  if (binaryExponent >= 0 && binaryExponent <= 62 &&
      (valueExponent >= 0 ||
       (significand & ((uint64_t{1} << -valueExponent) - 1)) == 0)) {
    const uint64_t integer = valueExponent >= 0 ? significand << valueExponent
                                                : significand >> -valueExponent;
    const int32_t dropped =
        binaryExponent >= 61 ? 2 : (binaryExponent >= 58 ? 1 : 0);
    const uint64_t divisor = dropped == 2 ? 100 : (dropped == 1 ? 10 : 1);
    return {(integer + divisor / 2) / divisor, dropped};
  }

  // Integer approximation of floor(log10(2^binaryExponent)); normalization
  // below makes the decimal exponent exact without floating-point log/pow.
  const int32_t scaledExponent = binaryExponent * 78913;
  int32_t decimalExponent = scaledExponent / 262144;
  if (scaledExponent < 0 && scaledExponent % 262144 != 0) {
    --decimalExponent;
  }
  const int32_t exponent = valueExponent - 2;
  CastUInt256 numerator;
  CastUInt256 denominator;
  CastUInt256 margin;
  scaledFraction(
      significand, exponent, decimalExponent, numerator, denominator, margin);
  auto tenDenominator = denominator;
  tenDenominator.multiply(10);
  if (numerator.compare(tenDenominator) >= 0) {
    ++decimalExponent;
    scaledFraction(
        significand, exponent, decimalExponent, numerator, denominator, margin);
  } else if (numerator.compare(denominator) < 0) {
    --decimalExponent;
    scaledFraction(
        significand, exponent, decimalExponent, numerator, denominator, margin);
  }

  const int32_t denominatorZeros = denominator.trailingZeros();
  const int32_t marginZeros = margin.trailingZeros();
  const int32_t cancelled =
      denominatorZeros < marginZeros ? denominatorZeros : marginZeros;
  numerator.shiftRight(cancelled);
  denominator.shiftRight(cancelled);
  margin.shiftRight(cancelled);
  tenDenominator = denominator;
  tenDenominator.multiply(10);

  // Preserve the legacy Java small-integer interval decisions. Its signed
  // 32/64-bit addition can wrap when testing the upper rounding margin. These
  // decisions affect observable decimal digits (e.g. around 2.4e25), so doing
  // every comparison in unlimited precision would not reproduce Spark's input
  // to BigDecimal. Unsigned arithmetic below makes wraparound well-defined.
  int32_t numeratorBits = 55 + (exponent > 0 ? exponent : 0) - cancelled;
  if (decimalExponent < 0) {
    CastUInt256 powerOfFive(1);
    for (int32_t i = 0; i < -decimalExponent; ++i) {
      powerOfFive.multiply(5);
    }
    numeratorBits += -decimalExponent + powerOfFive.bitWidth();
  }
  const int32_t tenDenominatorBits = tenDenominator.bitWidth();
  const int32_t comparisonBits = numeratorBits < 32 && tenDenominatorBits < 32
      ? 32
      : (numeratorBits < 64 && tenDenominatorBits < 64 ? 64 : 0);

  uint64_t digits = 0;
  int32_t count = 0;
  for (;;) {
    uint32_t digit = 0;
    while (numerator.compare(denominator) >= 0) {
      numerator.subtract(denominator);
      ++digit;
    }
    digits = digits * 10 + digit;
    ++count;

    bool low = numerator.compare(margin) < 0;
    auto upper = numerator;
    upper.add(margin);
    bool high = upper.compare(denominator) > 0;
    if (comparisonBits != 0) {
      const uint64_t mask = comparisonBits == 32 ? 0xffffffffULL : ~uint64_t{0};
      const uint64_t sign = uint64_t{1} << (comparisonBits - 1);
      const uint64_t scaledMargin = (margin.words[0] * 10) & mask;
      const uint64_t remainder = (numerator.words[0] * 10) & mask;
      if (scaledMargin != 0 && (scaledMargin & sign) == 0) {
        low = remainder < scaledMargin;
        const uint64_t sum = (remainder + scaledMargin) & mask;
        high = (sum & sign) == 0 && sum > tenDenominator.words[0];
      } else {
        low = true;
        high = true;
      }
    }
    if (low || high) {
      auto twiceRemainder = numerator;
      twiceRemainder.multiply(2);
      const int32_t distance = twiceRemainder.compare(denominator);
      if (high && (!low || distance > 0 || (distance == 0 && (digits & 1)))) {
        ++digits;
      }
      return {digits, decimalExponent - count + 1};
    }
    numerator.multiply(10);
    margin.multiply(10);
  }
}

VELOX_CUDF_CAST_HOST_DEVICE inline bool floatingPointToDecimal(
    double input,
    int32_t precision,
    int32_t scale,
    __int128& output) {
  uint64_t bits;
#ifdef __CUDA_ARCH__
  bits = static_cast<uint64_t>(__double_as_longlong(input));
#else
  std::memcpy(&bits, &input, sizeof(bits));
#endif
  const uint64_t absoluteBits = bits & 0x7fffffffffffffffULL;
  if ((absoluteBits >> 52) == 0x7ff) {
    return false;
  }
  const double absolute = input < 0 ? -input : input;
  if (absolute >= 1e38) {
    return false;
  }
  if (absolute < 1e-39) {
    output = 0;
    return true;
  }
  const auto decimal = floatingDecimal(absoluteBits);
  unsigned __int128 value = decimal.significand;
  unsigned __int128 limit = 1;
  for (int32_t i = 0; i < precision; ++i) {
    limit *= 10;
  }
  --limit;
  const int32_t delta = scale + decimal.exponent;
  if (delta >= 0) {
    if (delta > 38) {
      return false;
    }
    unsigned __int128 factor = 1;
    for (int32_t i = 0; i < delta; ++i) {
      factor *= 10;
    }
    if (value > limit / factor) {
      return false;
    }
    value *= factor;
  } else if (delta < -19) {
    value = 0;
  } else {
    unsigned __int128 factor = 1;
    for (int32_t i = 0; i < -delta; ++i) {
      factor *= 10;
    }
    value = value / factor + (value % factor >= factor / 2);
  }
  if (value > limit) {
    return false;
  }
  output = static_cast<__int128>(value);
  if (bits >> 63) {
    output = -output;
  }
  return true;
}

} // namespace facebook::velox::cudf_velox::sparksql::detail

#undef VELOX_CUDF_CAST_HOST_DEVICE
