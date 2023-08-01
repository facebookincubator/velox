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

// Returns first mag and magLength.
// The magnitude of this BigInteger, in <i>big-endian</i> order: the
// zeroth element of this array is the most-significant int of the
// magnitude.  The magnitude must be "minimal" in that the most-significant
// int ({@code mag[0]}) must be non-zero.  This is necessary to
// ensure that there is exactly one representation for each BigInteger
// value.  Note that this implies that the BigInteger zero has a
// zero-length mag array.
std::pair<int32_t, int32_t> getFirstMag(uint128_t value) {
  int32_t v1 = value >> 96;
  if (v1 != 0) {
    return {v1, 4};
  }
  int32_t v2 = value >> 64;
  if (v2 != 0) {
    return {v2, 3};
  }
  int32_t v3 = value >> 32;
  if (v3 != 0) {
    return {v3, 2};
  }
  int32_t v4 = value;
  if (v4 != 0) {
    return {v4, 1};
  }
  return {-1, 0};
}

// Origins from java side BigInteger#bitLength.
//
// Returns the number of bits in the minimal two's-complement
// representation of this BigInteger, <em>excluding</em> a sign bit.
// For positive BigIntegers, this is equivalent to the number of bits in
// the ordinary binary representation.  For zero this method returns
// {@code 0}.  (Computes {@code (ceil(log2(this < 0 ? -this : this+1)))}.)
//
// @return number of bits in the minimal two's-complement
//         representation of this BigInteger, <em>excluding</em> a sign bit.
int32_t getBitLength(int8_t sig, uint128_t value) {
  int32_t n = -1;
  if (value == 0) {
    n = 0;
  } else {
    auto [firstMag, len] = getFirstMag(value);
    int32_t magBitLength = ((len - 1) << 5) + 64 -
        bits::countLeadingZeros(static_cast<uint64_t>(firstMag));
    if (sig < 0) {
      // Check if value is a power of two.
      bool pow2 =
          bits::countBits(
              reinterpret_cast<uint64_t*>(&value), 0, sizeof(uint128_t) * 8) <=
          1;
      n = (pow2 ? magBitLength - 1 : magBitLength);
    } else {
      n = magBitLength;
    }
  }
  return n;
}

} // namespace

std::string DecimalUtil::toString(const int128_t value, const TypePtr& type) {
  auto [precision, scale] = getDecimalPrecisionScale(*type);
  return formatDecimal(scale, value);
}

int32_t DecimalUtil::getByteArrayLength(int128_t value) {
  uint128_t absValue;
  int8_t sig;
  if (value >= 0) {
    absValue = value;
    sig = 1;
  } else {
    absValue = -value;
    sig = -1;
  }

  return getBitLength(sig, absValue) / 8 + 1;
}

void DecimalUtil::toByteArray(int128_t value, char* out, int32_t& length) {
  length = getByteArrayLength(value);
  auto lowBig = folly::Endian::big<int64_t>(value);
  uint8_t* lowAddr = reinterpret_cast<uint8_t*>(&lowBig);
  if (length <= sizeof(int64_t)) {
    memcpy(out, lowAddr + sizeof(int64_t) - length, length);
  } else {
    auto highBig = folly::Endian::big<int64_t>(value >> 64);
    uint8_t* highAddr = reinterpret_cast<uint8_t*>(&highBig);
    memcpy(out, highAddr + sizeof(int128_t) - length, length - sizeof(int64_t));
    memcpy(out + length - sizeof(int64_t), lowAddr, sizeof(int64_t));
  }
}

} // namespace facebook::velox
