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
      const int toScale,
      bool nullOnOverflow = false,
      bool roundUp = true) {
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
      if (roundUp && inputValue >= 0 && remainder >= scalingFactor / 2) {
        ++rescaledValue;
      } else if (roundUp && remainder <= -scalingFactor / 2) {
        --rescaledValue;
      }
    }
    // Check overflow.
    if (rescaledValue < -DecimalUtil::kPowersOfTen[toPrecision] ||
        rescaledValue > DecimalUtil::kPowersOfTen[toPrecision] || isOverflow) {
      if (nullOnOverflow) {
        return std::nullopt;
      } else {
        VELOX_USER_FAIL(
            "Cannot cast DECIMAL '{}' to DECIMAL({},{})",
            DecimalUtil::toString(
                inputValue, DECIMAL(fromPrecision, fromScale)),
            toPrecision,
            toScale);
      }
    }
    return static_cast<TOutput>(rescaledValue);
  }

  template <typename TInput, typename TOutput>
  inline static std::optional<TOutput> rescaleDouble(
      const TInput inputValue,
      const int toPrecision,
      const int toScale) {
    static_assert(
        std::is_same_v<TOutput, int64_t> || std::is_same_v<TOutput, int128_t>);

    // Multiply decimal with the scale
    auto unscaled = inputValue * DecimalUtil::kPowersOfTen[toScale];

    bool isOverflow = std::isnan(unscaled);

    unscaled = std::round(unscaled);

    // convert scaled double to int128
    int32_t sign = unscaled < 0 ? -1 : 1;
    auto unscaled_abs = std::abs(unscaled);

    uint64_t high_bits = static_cast<uint64_t>(std::ldexp(unscaled_abs, -64));
    uint64_t low_bits = static_cast<uint64_t>(
        unscaled_abs - std::ldexp(static_cast<double>(high_bits), 64));

    auto rescaledValue = HugeInt::build(high_bits, low_bits);

    if (rescaledValue < -DecimalUtil::kPowersOfTen[toPrecision] ||
        rescaledValue > DecimalUtil::kPowersOfTen[toPrecision] || isOverflow) {
      VELOX_USER_FAIL(
          "Cannot cast DECIMAL '{}' to DECIMAL({},{})",
          inputValue,
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
          CppToType<TInput>::name,
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
    R unsignedDividendRescaled(a);
    if (a < 0) {
      resultSign = -1;
      unsignedDividendRescaled *= -1;
    }
    R unsignedDivisor(b);
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
    if (!noRoundUp && remainder * 2 >= unsignedDivisor) {
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

  inline static int32_t FirstNonzeroLongNum(
      const std::vector<int32_t>& mag,
      int32_t length) {
    int32_t fn = 0;
    int32_t i;
    for (i = length - 1; i >= 0 && mag[i] == 0; i--)
      ;
    fn = length - i - 1;
    return fn;
  }

  inline static int32_t GetInt(
      int32_t n,
      int32_t sig,
      const std::vector<int32_t>& mag,
      int32_t length) {
    if (n < 0)
      return 0;
    if (n >= length)
      return sig < 0 ? -1 : 0;

    int32_t magInt = mag[length - n - 1];
    return (
        sig >= 0 ? magInt
                 : (n <= FirstNonzeroLongNum(mag, length) ? -magInt : ~magInt));
  }

  inline static int32_t GetNumberOfLeadingZeros(uint32_t i) {
    // TODO: we can get faster implementation by gcc build-in function
    // HD, Figure 5-6
    if (i == 0)
      return 32;
    int32_t n = 1;
    if (i >> 16 == 0) {
      n += 16;
      i <<= 16;
    }
    if (i >> 24 == 0) {
      n += 8;
      i <<= 8;
    }
    if (i >> 28 == 0) {
      n += 4;
      i <<= 4;
    }
    if (i >> 30 == 0) {
      n += 2;
      i <<= 2;
    }
    n -= i >> 31;
    return n;
  }

  inline static int32_t GetBitLengthForInt(uint32_t n) {
    return 32 - GetNumberOfLeadingZeros(n);
  }

  inline static int32_t GetBitCount(uint32_t i) {
    // HD, Figure 5-2
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    i = (i + (i >> 4)) & 0x0f0f0f0f;
    i = i + (i >> 8);
    i = i + (i >> 16);
    return i & 0x3f;
  }

  inline static int32_t
  GetBitLength(int32_t sig, const std::vector<int32_t>& mag, int32_t len) {
    int32_t n = -1;
    if (len == 0) {
      n = 0;
    } else {
      // Calculate the bit length of the magnitude
      int32_t mag_bit_length =
          ((len - 1) << 5) + GetBitLengthForInt((uint32_t)mag[0]);
      if (sig < 0) {
        // Check if magnitude is a power of two
        bool pow2 = (GetBitCount((uint32_t)mag[0]) == 1);
        for (int i = 1; i < len && pow2; i++)
          pow2 = (mag[i] == 0);

        n = (pow2 ? mag_bit_length - 1 : mag_bit_length);
      } else {
        n = mag_bit_length;
      }
    }
    return n;
  }

  static std::vector<uint32_t>
  ConvertMagArray(int64_t new_high, uint64_t new_low, int32_t* size) {
    std::vector<uint32_t> mag;
    int64_t orignal_low = new_low;
    int64_t orignal_high = new_high;
    mag.push_back(new_high >>= 32);
    mag.push_back((uint32_t)orignal_high);
    mag.push_back(new_low >>= 32);
    mag.push_back((uint32_t)orignal_low);

    int32_t start = 0;
    // remove the front 0
    for (int32_t i = 0; i < 4; i++) {
      if (mag[i] == 0)
        start++;
      if (mag[i] != 0)
        break;
    }

    int32_t length = 4 - start;
    std::vector<uint32_t> new_mag;
    // get the mag after remove the high 0
    for (int32_t i = start; i < 4; i++) {
      new_mag.push_back(mag[i]);
    }

    *size = length;
    return new_mag;
  }

  /*
   *  This method refer to the BigInterger#toByteArray() method in Java side.
   */
  inline static char* ToByteArray(int128_t value, int32_t* length = nullptr) {
    int128_t new_value;
    int32_t sig;
    if (value > 0) {
      new_value = value;
      sig = 1;
    } else if (value < 0) {
      new_value = std::abs(value);
      sig = -1;
    } else {
      new_value = value;
      sig = 0;
    }

    int64_t new_high;
    uint64_t new_low;

    int128_t orignal_value = new_value;
    new_high = new_value >> 64;
    new_low = (uint64_t)orignal_value;

    std::vector<uint32_t> mag;
    int32_t size;
    mag = ConvertMagArray(new_high, new_low, &size);

    std::vector<int32_t> final_mag;
    for (auto i = 0; i < size; i++) {
      final_mag.push_back(mag[i]);
    }

    int32_t byte_length = GetBitLength(sig, final_mag, size) / 8 + 1;
    if (length) {
      *length = byte_length;
    }
    char* out = new char[16];
    uint32_t next_int = 0;
    for (int32_t i = byte_length - 1, bytes_copied = 4, int_index = 0; i >= 0;
         i--) {
      if (bytes_copied == 4) {
        next_int = GetInt(int_index++, sig, final_mag, size);
        bytes_copied = 1;
      } else {
        next_int >>= 8;
        bytes_copied++;
      }

      out[i] = (uint8_t)next_int;
    }
    return out;
  }

  inline static double toDoubleValue(int128_t value, uint8_t scale) {
    int128_t new_value;
    int32_t sig;
    if (value > 0) {
      new_value = value;
      sig = 1;
    } else if (value < 0) {
      new_value = std::abs(value);
      sig = -1;
    } else {
      new_value = value;
      sig = 0;
    }

    int64_t new_high;
    uint64_t new_low;

    int128_t orignal_value = new_value;
    new_high = new_value >> 64;
    new_low = (uint64_t)orignal_value;

    double unscaled = static_cast<double>(new_low) +
        std::ldexp(static_cast<double>(new_high), 64);

    // scale double.
    return (unscaled * sig) / DecimalUtil::kPowersOfTen[scale];
  }

  template <class T>
  inline static int numDigits(T number) {
    int digits = 0;
    if (number < 0)
      digits = 1; // remove this line if '-' counts as a digit
    while (number) {
      number /= 10;
      digits++;
    }
    return digits;
  }

  static constexpr double double10pow[] = {
      1.0e0,  1.0e1,  1.0e2,  1.0e3,  1.0e4,  1.0e5,  1.0e6,  1.0e7,
      1.0e8,  1.0e9,  1.0e10, 1.0e11, 1.0e12, 1.0e13, 1.0e14, 1.0e15,
      1.0e16, 1.0e17, 1.0e18, 1.0e19, 1.0e20, 1.0e21, 1.0e22};

  static constexpr __uint128_t kOverflowMultiplier = ((__uint128_t)1 << 127);
  static constexpr long kLongMinValue = 0x8000000000000000L;
  static constexpr long kLONG_MASK = 0xffffffffL;

}; // DecimalUtil
} // namespace facebook::velox
