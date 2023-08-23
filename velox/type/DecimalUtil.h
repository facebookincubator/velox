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
#include "velox/common/base/Nulls.h"
#include "velox/common/base/RawVector.h"
#include "velox/common/process/ProcessBase.h"
#include "velox/type/Filter.h"
#include "velox/type/Type.h"

namespace facebook::velox {

/// A static class that holds helper functions for DECIMAL type.
class DecimalUtil {
 public:
  static constexpr int64_t
      kPowersOfTenShort[ShortDecimalType::kMaxPrecision + 1] = {
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
          1'000'000'000'000'000'000};

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
        "Decimal overflow. Value '{}' is not in the range of Decimal Type",
        value);
  }

  /// Helper function to convert a decimal value to string.
  static std::string toString(const int128_t value, const TypePtr& type);

  template <typename T>
  inline static void fillDecimals(
      const uint64_t* nullsPtr,
      T* decimals,
      const T* values,
      const int64_t* scales,
      int32_t targetScale,
      raw_vector<int32_t>& rows,
      const velox::common::Filter& filter) {
    int32_t numRows = rows.size();
    bool isDense = (rows.back() - rows[0] == numRows - 1);
    int32_t rowsCnt = 0;
    if constexpr (std::is_same_v<T, int64_t>) { // Short Decimal
      int startIndex = 0;
      // short decimal can go fast path
      if (!nullsPtr && process::hasAvx2()) {
        constexpr int kBatchSize = xsimd::batch<int32_t>::size;
        int halfBatch = kBatchSize / 2;
        int32_t vecSize = numRows - numRows % kBatchSize;
        auto targetScales = xsimd::batch<int64_t>::broadcast(targetScale);
        for (int32_t i = 0; i < vecSize; i += kBatchSize) {
          // deal with first half
          auto fstValues = xsimd::load_aligned(values + i);
          auto scalesVec = xsimd::load_aligned(scales + i);
          int32_t matchMask = simd::toBitMask(scalesVec == targetScales);
          if (matchMask != simd::allSetBitMask<int64_t>()) {
            auto powersVec = simd::gather<int64_t, int64_t>(
                kPowersOfTenShort, xsimd::abs(scalesVec - targetScales));
            fstValues = xsimd::select(
                scalesVec < targetScales,
                fstValues * powersVec,
                fstValues / powersVec);
          }
          // deal with second half
          int32_t startRow = i + halfBatch;
          auto secValues = xsimd::load_aligned(values + startRow);
          scalesVec = xsimd::load_aligned(scales + startRow);
          matchMask = simd::toBitMask(scalesVec == targetScales);
          if (matchMask != simd::allSetBitMask<int64_t>()) {
            auto powersVec = simd::gather<int64_t, int64_t>(
                kPowersOfTenShort, xsimd::abs(scalesVec - targetScales));
            secValues = xsimd::select(
                scalesVec < targetScales,
                secValues * powersVec,
                secValues / powersVec);
          }
          int32_t fstMask = simd::toBitMask(filter.testValues(fstValues));
          int32_t secMask = simd::toBitMask(filter.testValues(secValues));
          int32_t mask = fstMask | secMask << halfBatch;

          // save values and outputRows
          if (mask) {
            if (mask == simd::allSetBitMask<int32_t>()) {
              xsimd::load_aligned(&rows[i]).store_aligned(&rows[rowsCnt]);
              fstValues.store_aligned(decimals + rowsCnt);
              rowsCnt += halfBatch;
              secValues.store_aligned(decimals + rowsCnt);
              rowsCnt += halfBatch;
            } else {
              if (isDense) {
                auto fstRow = rows[rowsCnt];
                (xsimd::load_unaligned(simd::byteSetBits(mask)) + fstRow)
                    .store_aligned(&rows[rowsCnt]);
              } else {
                simd::filter(xsimd::load_aligned(&rows[i]), mask)
                    .store_aligned(&rows[rowsCnt]);
              }
              if (fstMask) {
                if (fstMask == simd::allSetBitMask<int64_t>()) {
                  fstValues.store_aligned(decimals + rowsCnt);
                  rowsCnt += halfBatch;
                } else {
                  simd::filter(fstValues, fstMask)
                      .store_aligned(decimals + rowsCnt);
                  rowsCnt += __builtin_popcount(fstMask);
                }
              }
              if (secMask) {
                if (secMask == simd::allSetBitMask<int64_t>()) {
                  secValues.store_aligned(decimals + rowsCnt);
                  rowsCnt += halfBatch;
                } else {
                  simd::filter(secValues, secMask)
                      .store_aligned(decimals + rowsCnt);
                  rowsCnt += __builtin_popcount(secMask);
                }
              }
            }
          }
        }
        startIndex = vecSize;
      }
      // remaining part can't be vectorize
      for (int32_t rowIndex = startIndex; rowIndex < numRows; rowIndex++) {
        if (!nullsPtr || !bits::isBitNull(nullsPtr, rowIndex)) {
          int64_t currentScale = scales[rowIndex];
          T value = values[rowIndex];
          if (targetScale > currentScale &&
              targetScale - currentScale <= ShortDecimalType::kMaxPrecision) {
            value *= kPowersOfTenShort[targetScale - currentScale];
          } else if (
              targetScale < currentScale &&
              currentScale - targetScale <= ShortDecimalType::kMaxPrecision) {
            value /= kPowersOfTenShort[currentScale - targetScale];
          } else if (targetScale != currentScale) {
            VELOX_FAIL("Decimal scale out of range");
          }
          if (filter.testInt64(value)) {
            if (nullsPtr) { // NULL value is needed in this case
              decimals[rowIndex] = value;
            } else { // we can skip NULL values
              decimals[rowsCnt] = value;
              rows[rowsCnt++] = rows[rowIndex];
            }
          }
        }
      }
    } else { // Long Decimal
      for (int32_t rowIndex = 0; rowIndex < numRows; rowIndex++) {
        if (!nullsPtr || !bits::isBitNull(nullsPtr, rowIndex)) {
          int32_t currentScale = scales[rowIndex];
          T value = values[rowIndex];
          if (targetScale > currentScale) {
            while (targetScale > currentScale) {
              int32_t scaleAdjust = std::min<int32_t>(
                  ShortDecimalType::kMaxPrecision, targetScale - currentScale);
              value *= kPowersOfTen[scaleAdjust];
              currentScale += scaleAdjust;
            }
          } else if (targetScale < currentScale) {
            while (currentScale > targetScale) {
              int32_t scaleAdjust = std::min<int32_t>(
                  ShortDecimalType::kMaxPrecision, currentScale - targetScale);
              value /= kPowersOfTen[scaleAdjust];
              currentScale -= scaleAdjust;
            }
          }
          if (filter.testInt128(value)) {
            if (nullsPtr) { // NULL value is needed in this case
              decimals[rowIndex] = value;
            } else { // we can skip NULL values
              decimals[rowsCnt] = value;
              rows[rowsCnt++] = rows[rowIndex];
            }
          }
        }
      }
    }
    if (!nullsPtr) { // we can skip NULL values
      rows.resize(rowsCnt);
    }
  }

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
          "Cannot cast DECIMAL '{}' to DECIMAL({}, {})",
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
          "Cannot cast {} '{}' to DECIMAL({}, {})",
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

  /// Origins from java side BigInteger#bitLength.
  ///
  /// Returns the number of bits in the minimal two's-complement
  /// representation of this BigInteger, <em>excluding</em> a sign bit.
  /// For positive BigIntegers, this is equivalent to the number of bits in
  /// the ordinary binary representation.  For zero this method returns
  /// {@code 0}.  (Computes {@code (ceil(log2(this < 0 ? -this : this+1)))}.)
  ///
  /// @return number of bits in the minimal two's-complement
  ///         representation of this BigInteger, <em>excluding</em> a sign bit.
  static int32_t getByteArrayLength(int128_t value);

  /// This method return the same result with the BigInterger#toByteArray()
  /// method in Java side.
  ///
  /// Returns a byte array containing the two's-complement representation of
  /// this BigInteger. The byte array will be in big-endian byte-order: the most
  /// significant byte is in the zeroth element. The array will contain the
  /// minimum number of bytes required to represent this BigInteger, including
  /// at least one sign bit, which is (ceil((this.bitLength() + 1)/8)).
  ///
  /// @return The length of out.
  static int32_t toByteArray(int128_t value, char* out);

  static constexpr __uint128_t kOverflowMultiplier = ((__uint128_t)1 << 127);

}; // DecimalUtil
} // namespace facebook::velox
