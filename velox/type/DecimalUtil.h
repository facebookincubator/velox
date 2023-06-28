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

#include <string>
#include "velox/common/base/CheckedArithmetic.h"
#include "velox/common/base/Exceptions.h"
#include "velox/type/Type.h"

namespace facebook::velox {

/// A static class that holds helper functions for DECIMAL type.
class DecimalUtil {
 public:
  static constexpr int128_t kPowersOfTen[LongDecimalType::kMaxPrecision + 1] = {
      1,
      10,
      100,
      1'000,
      10'000,
      100'000,
      1'000'000,
      10'000'000,
      100'000'000,
      1'000'000'000,
      10'000'000'000,
      100'000'000'000,
      1'000'000'000'000,
      10'000'000'000'000,
      100'000'000'000'000,
      1'000'000'000'000'000,
      10'000'000'000'000'000,
      100'000'000'000'000'000,
      1'000'000'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)10,
      1'000'000'000'000'000'000 * (int128_t)100,
      1'000'000'000'000'000'000 * (int128_t)1'000,
      1'000'000'000'000'000'000 * (int128_t)10'000,
      1'000'000'000'000'000'000 * (int128_t)100'000,
      1'000'000'000'000'000'000 * (int128_t)1'000'000,
      1'000'000'000'000'000'000 * (int128_t)10'000'000,
      1'000'000'000'000'000'000 * (int128_t)100'000'000,
      1'000'000'000'000'000'000 * (int128_t)1'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)10'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)100'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)10'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)100'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)10'000'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)100'000'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000'000'000,
      1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000'000'000 *
          (int128_t)10,
      1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000'000'000 *
          (int128_t)100};

  static constexpr int128_t kLongDecimalMin =
      -kPowersOfTen[LongDecimalType::kMaxPrecision] + 1;
  static constexpr int128_t kLongDecimalMax =
      kPowersOfTen[LongDecimalType::kMaxPrecision] - 1;
  static constexpr int128_t kShortDecimalMin =
      -kPowersOfTen[ShortDecimalType::kMaxPrecision] + 1;
  static constexpr int128_t kShortDecimalMax =
      kPowersOfTen[ShortDecimalType::kMaxPrecision] - 1;

  FOLLY_ALWAYS_INLINE static void valueInRange(int128_t value) {
    VELOX_CHECK(
        (value >= kLongDecimalMin && value <= kLongDecimalMax),
        "Value '{}' is not in the range of Decimal Type",
        value);
  }

  /// Helper function to convert a decimal value to string.
  static std::string toString(const int128_t value, const TypePtr& type);

  template <typename TInput, typename TOutput>
  inline static std::optional<TOutput> rescaleWithRoundUp(
      const TInput inputValue,
      const int fromPrecision,
      const int fromScale,
      const int toPrecision,
      const int toScale) {
    int128_t rescaledValue = inputValue;
    auto scaleDifference = toScale - fromScale;
    bool isOverflow = false;
    if (scaleDifference >= 0) {
      isOverflow = __builtin_mul_overflow(
          rescaledValue,
          DecimalUtil::kPowersOfTen[scaleDifference],
          &rescaledValue);
    } else {
      scaleDifference = -scaleDifference;
      const auto scalingFactor = DecimalUtil::kPowersOfTen[scaleDifference];
      rescaledValue /= scalingFactor;
      int128_t remainder = inputValue % scalingFactor;
      if (inputValue >= 0 && remainder >= scalingFactor / 2) {
        ++rescaledValue;
      } else if (remainder <= -scalingFactor / 2) {
        --rescaledValue;
      }
    }
    // Check overflow.
    if (rescaledValue < -DecimalUtil::kPowersOfTen[toPrecision] ||
        rescaledValue > DecimalUtil::kPowersOfTen[toPrecision] || isOverflow) {
      VELOX_USER_FAIL(
          "Cannot cast DECIMAL '{}' to DECIMAL({},{})",
          DecimalUtil::toString(inputValue, DECIMAL(fromPrecision, fromScale)),
          toPrecision,
          toScale);
    }
    return static_cast<TOutput>(rescaledValue);
  }

  template <typename TInput, typename TOutput>
  inline static std::optional<TOutput> rescaleInt(
      const TInput inputValue,
      const int toPrecision,
      const int toScale) {
    int128_t rescaledValue = static_cast<int128_t>(inputValue);
    bool isOverflow = __builtin_mul_overflow(
        rescaledValue, DecimalUtil::kPowersOfTen[toScale], &rescaledValue);
    // Check overflow.
    if (rescaledValue < -DecimalUtil::kPowersOfTen[toPrecision] ||
        rescaledValue > DecimalUtil::kPowersOfTen[toPrecision] || isOverflow) {
      VELOX_USER_FAIL(
          "Cannot cast {} '{}' to DECIMAL({},{})",
          SimpleTypeTrait<TInput>::name,
          inputValue,
          toPrecision,
          toScale);
    }
    return static_cast<TOutput>(rescaledValue);
  }

  template <typename R, typename A, typename B>
  inline static R divideWithRoundUp(
      R& r,
      const A& a,
      const B& b,
      bool noRoundUp,
      uint8_t aRescale,
      uint8_t /*bRescale*/) {
    VELOX_CHECK_NE(b, 0, "Division by zero");
    int resultSign = 1;
    A unsignedDividendRescaled(a);
    if (a < 0) {
      resultSign = -1;
      unsignedDividendRescaled *= -1;
    }
    B unsignedDivisor(b);
    if (b < 0) {
      resultSign *= -1;
      unsignedDivisor *= -1;
    }
    unsignedDividendRescaled = checkedMultiply<R>(
        unsignedDividendRescaled,
        R(DecimalUtil::kPowersOfTen[aRescale]),
        "Decimal");
    R quotient = unsignedDividendRescaled / unsignedDivisor;
    R remainder = unsignedDividendRescaled % unsignedDivisor;
    if (!noRoundUp && static_cast<const B>(remainder) * 2 >= unsignedDivisor) {
      ++quotient;
    }
    r = quotient * resultSign;
    return remainder;
  }

  /*
   * sum up and return overflow/underflow.
   */
  inline static int64_t addUnsignedValues(
      int128_t& sum,
      const int128_t& lhs,
      const int128_t& rhs,
      bool isResultNegative) {
    __uint128_t unsignedSum = (__uint128_t)lhs + (__uint128_t)rhs;
    // Ignore overflow value.
    sum = (int128_t)unsignedSum & ~kOverflowMultiplier;
    sum = isResultNegative ? -sum : sum;
    return (unsignedSum >> 127);
  }

  inline static int64_t
  addWithOverflow(int128_t& result, const int128_t& lhs, const int128_t& rhs) {
    bool isLhsNegative = lhs < 0;
    bool isRhsNegative = rhs < 0;
    int64_t overflow = 0;
    if (isLhsNegative == isRhsNegative) {
      // Both inputs of same time.
      if (isLhsNegative) {
        // Both negative, ignore signs and add.
        overflow = addUnsignedValues(result, -lhs, -rhs, true);
        overflow = -overflow;
      } else {
        overflow = addUnsignedValues(result, lhs, rhs, false);
      }
    } else {
      // If one of them is negative, use addition.
      result = lhs + rhs;
    }
    return overflow;
  }

  /*
   * Computes average. If there is an overflow value uses the following
   * expression to compute the average.
   *                       ---                                         ---
   *                      |    overflow_multiplier          sum          |
   * average = overflow * |     -----------------  +  ---------------    |
   *                      |         count              count * overflow  |
   *                       ---                                         ---
   */
  inline static void computeAverage(
      int128_t& avg,
      const int128_t& sum,
      const int64_t count,
      const int64_t overflow) {
    if (overflow == 0) {
      divideWithRoundUp<int128_t, int128_t, int64_t>(
          avg, sum, count, false, 0, 0);
    } else {
      __uint128_t sumA{0};
      auto remainderA =
          DecimalUtil::divideWithRoundUp<__uint128_t, __uint128_t, int64_t>(
              sumA, kOverflowMultiplier, count, true, 0, 0);
      double totalRemainder = (double)remainderA / count;
      __uint128_t sumB{0};
      auto remainderB =
          DecimalUtil::divideWithRoundUp<__uint128_t, __int128_t, int64_t>(
              sumB, sum, count * overflow, true, 0, 0);
      totalRemainder += (double)remainderB / (count * overflow);
      DecimalUtil::addWithOverflow(avg, sumA, sumB);
      avg = avg * overflow + (int)(totalRemainder * overflow);
    }
  }

  // This method refer to the BigInterger#toByteArray() method in Java side.
  // Returns a byte array containing the two's-complement representation of this
  // BigInteger. The byte array will be in big-endian byte-order: the most
  // significant byte is in the zeroth element. The array will contain the
  // minimum number of bytes required to represent this BigInteger, including at
  // least one sign bit, which is (ceil((this.bitLength() + 1)/8)).
  inline static void toByteArray(int128_t value, char* out, int32_t* length) {
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
    for (int32_t i = byteLength - 1, bytesCopied = 4, intIndex = 0; i >= 0;
         --i) {
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

  static constexpr __uint128_t kOverflowMultiplier = ((__uint128_t)1 << 127);

 private:
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
  inline static int32_t lastNonzeroIntIndex(const std::vector<int32_t>& mag) {
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
  inline static int32_t getInt(
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

  inline static int32_t getBitCount(uint32_t i) {
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
  inline static int32_t getBitLength(
      int8_t sig,
      const std::vector<int32_t>& mag) {
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
  inline static std::vector<int32_t> convertToIntMags(uint128_t value) {
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

  inline static int32_t getByteArrayLength(int128_t value) {
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

}; // DecimalUtil
} // namespace facebook::velox
