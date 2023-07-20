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

#include "velox/type/DecimalUtil.h"
#include "velox/type/HugeInt.h"

namespace facebook::velox {
namespace {
std::string formatDecimal(uint8_t scale, int128_t unscaledValue) {
  VELOX_DCHECK_GE(scale, 0);
  VELOX_DCHECK_LT(
      static_cast<size_t>(scale), sizeof(DecimalUtil::kPowersOfTen));
  if (unscaledValue == 0) {
    return "0";
  }

  bool isNegative = (unscaledValue < 0);
  if (isNegative) {
    unscaledValue = ~unscaledValue + 1;
  }
  int128_t integralPart = unscaledValue / DecimalUtil::kPowersOfTen[scale];

  bool isFraction = (scale > 0);
  std::string fractionString;
  if (isFraction) {
    auto fraction =
        std::to_string(unscaledValue % DecimalUtil::kPowersOfTen[scale]);
    fractionString += ".";
    // Append leading zeros.
    fractionString += std::string(
        std::max(scale - static_cast<int>(fraction.size()), 0), '0');
    // Append the fraction part.
    fractionString += fraction;
  }

  return fmt::format(
      "{}{}{}", isNegative ? "-" : "", integralPart, fractionString);
}

/**
 * Origins from BigInteger#firstNonzeroIntNum.
 *
 * Returns the index of the int that contains the first nonzero int in the
 * little-endian binary representation of the magnitude (int 0 is the
 * least significant). If the magnitude is zero, return value is undefined.
 *
 * <p>Note: never used for a BigInteger with a magnitude of zero.
 * @see #getInt
 */
int32_t lastNonzeroIntIndex(const std::vector<int32_t>& mag) {
  int32_t i;
  for (i = mag.size() - 1; i >= 0 && mag[i] == 0; --i) {
    ;
  }
  return mag.size() - i - 1;
}

/**
 * Origins from BigInteger#getInt.
 *
 * Returns the specified int of the little-endian two's complement
 * representation (int 0 is the least significant).  The int number can
 * be arbitrarily high (values are logically preceded by infinitely many
 * sign ints).
 */
int32_t getInt(
    int32_t n,
    int8_t sig,
    const std::vector<int32_t>& mag,
    int32_t lastNonzeroIntIndex) {
  if (n < 0) {
    return 0;
  }
  if (n >= mag.size()) {
    return sig < 0 ? -1 : 0;
  }

  int32_t magInt = mag[mag.size() - n - 1];
  return (sig >= 0 ? magInt : (n <= lastNonzeroIntIndex ? -magInt : ~magInt));
}

int32_t getBitCount(uint32_t i) {
  static constexpr int kMaxBits = std::numeric_limits<uint64_t>::digits;
  uint64_t num = static_cast<uint32_t>(i);
  return bits::countBits(reinterpret_cast<uint64_t*>(&num), 0, kMaxBits);
}

/**
 * Origins from java side BigInteger#bitLength.
 *
 * Returns the number of bits in the minimal two's-complement
 * representation of this BigInteger, <em>excluding</em> a sign bit.
 * For positive BigIntegers, this is equivalent to the number of bits in
 * the ordinary binary representation.  For zero this method returns
 * {@code 0}.  (Computes {@code (ceil(log2(this < 0 ? -this : this+1)))}.)
 *
 * @return number of bits in the minimal two's-complement
 *         representation of this BigInteger, <em>excluding</em> a sign bit.
 */
int32_t getBitLength(int8_t sig, const std::vector<int32_t>& mag) {
  int32_t len = mag.size();
  int32_t n = -1;
  if (len == 0) {
    n = 0;
  } else {
    // Calculate the bit length of the magnitude.
    int32_t magBitLength = ((len - 1) << 5) + 64 -
        bits::countLeadingZeros(static_cast<uint64_t>(mag[0]));
    if (sig < 0) {
      // Check if magnitude is a power of two.
      bool pow2 = (getBitCount((uint32_t)mag[0]) == 1);
      for (int i = 1; i < len && pow2; ++i) {
        pow2 = (mag[i] == 0);
      }
      n = (pow2 ? magBitLength - 1 : magBitLength);
    } else {
      n = magBitLength;
    }
  }
  return n;
}

/**
 * Refer to BigInteger(byte[] val) to get the mag definition
 *
 * The magnitude of this BigInteger, in <i>big-endian</i> order: the
 * zeroth element of this array is the most-significant int of the
 * magnitude.  The magnitude must be "minimal" in that the most-significant
 * int ({@code mag[0]}) must be non-zero.  This is necessary to
 * ensure that there is exactly one representation for each BigInteger
 * value.  Note that this implies that the BigInteger zero has a
 * zero-length mag array.
 */
std::vector<int32_t> convertToIntMags(uint128_t value) {
  std::vector<int32_t> mag;
  int32_t v1 = value >> 96;
  int32_t v2 = value >> 64;
  int32_t v3 = value >> 32;
  int32_t v4 = value;
  if (v1 != 0) {
    mag.emplace_back(v1);
    mag.emplace_back(v2);
    mag.emplace_back(v3);
    mag.emplace_back(v4);
  } else if (v2 != 0) {
    mag.emplace_back(v2);
    mag.emplace_back(v3);
    mag.emplace_back(v4);
  } else if (v3 != 0) {
    mag.emplace_back(v3);
    mag.emplace_back(v4);
  } else if (v4 != 0) {
    mag.emplace_back(v4);
  }
  return mag;
}
} // namespace

std::string DecimalUtil::toString(const int128_t value, const TypePtr& type) {
  auto [precision, scale] = getDecimalPrecisionScale(*type);
  return formatDecimal(scale, value);
}

int32_t DecimalUtil::getByteArrayLength(int128_t value) {
  uint128_t absValue;
  int32_t sig;
  if (value > 0) {
    absValue = value;
    sig = 1;
  } else if (value < 0) {
    absValue = -value;
    sig = -1;
  } else {
    absValue = value;
    sig = 0;
  }

  std::vector<int32_t> mag = convertToIntMags(absValue);
  return getBitLength(sig, mag) / 8 + 1;
}

void DecimalUtil::toByteArray(int128_t value, char* out, int32_t* length) {
  uint128_t absValue;
  int8_t sig;
  if (value > 0) {
    absValue = value;
    sig = 1;
  } else if (value < 0) {
    absValue = -value;
    sig = -1;
  } else {
    absValue = value;
    sig = 0;
  }

  std::vector<int32_t> mag = convertToIntMags(absValue);
  int32_t byteLength = getBitLength(sig, mag) / 8 + 1;

  uint32_t nextInt;
  int32_t lastNonzeroIntIndexValue = lastNonzeroIntIndex(mag);
  for (int32_t i = byteLength - 1, bytesCopied = 4, intIndex = 0; i >= 0; --i) {
    if (bytesCopied == 4) {
      nextInt = getInt(intIndex++, sig, mag, lastNonzeroIntIndexValue);
      bytesCopied = 1;
    } else {
      nextInt >>= 8;
      bytesCopied++;
    }

    out[i] = (uint8_t)nextInt;
  }
  *length = byteLength;
}

} // namespace facebook::velox
