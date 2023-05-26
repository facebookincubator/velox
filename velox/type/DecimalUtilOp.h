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
#include "velox/type/DecimalUtil.h"
#include "velox/type/Type.h"
#include "velox/type/UnscaledLongDecimal.h"
#include "velox/type/UnscaledShortDecimal.h"

namespace facebook::velox {

class DecimalUtilOp {
 public:
  inline static int32_t maxBitsRequiredIncreaseAfterScaling(int32_t scale_by) {
    // We rely on the following formula:
    // bits_required(x * 10^y) <= bits_required(x) + floor(log2(10^y)) + 1
    // We precompute floor(log2(10^x)) + 1 for x = 0, 1, 2...75, 76

    static const int32_t floor_log2_plus_one[] = {
        0,   4,   7,   10,  14,  17,  20,  24,  27,  30,  34,  37,  40,
        44,  47,  50,  54,  57,  60,  64,  67,  70,  74,  77,  80,  84,
        87,  90,  94,  97,  100, 103, 107, 110, 113, 117, 120, 123, 127,
        130, 133, 137, 140, 143, 147, 150, 153, 157, 160, 163, 167, 170,
        173, 177, 180, 183, 187, 190, 193, 196, 200, 203, 206, 210, 213,
        216, 220, 223, 226, 230, 233, 236, 240, 243, 246, 250, 253};
    return floor_log2_plus_one[scale_by];
  }

  template <typename A>
  inline static int32_t maxBitsRequiredAfterScaling(
      const A& num,
      uint8_t aRescale) {
    auto value = num.unscaledValue();
    auto valueAbs = std::abs(value);
    int32_t num_occupied = 0;
    if constexpr (std::is_same_v<A, UnscaledShortDecimal>) {
      num_occupied = 64 - bits::countLeadingZeros(valueAbs);
    } else {
      num_occupied = 128 - num.countLeadingZeros();
    }

    return num_occupied + maxBitsRequiredIncreaseAfterScaling(aRescale);
  }

  // If we have a number with 'numLz' leading zeros, and we scale it up by
  // 10^scale_by,
  // this function returns the minimum number of leading zeros the result can
  // have.
  inline static int32_t minLeadingZerosAfterScaling(
      int32_t numLz,
      int32_t scaleBy) {
    int32_t result = numLz - maxBitsRequiredIncreaseAfterScaling(scaleBy);
    return result;
  }

  template <typename A, typename B>
  inline static int32_t
  minLeadingZeros(const A& a, const B& b, uint8_t aScale, uint8_t bScale) {
    auto x_value_abs = std::abs(a.unscaledValue());

    auto y_value_abs = std::abs(b.unscaledValue());
    ;

    int32_t x_lz = a.countLeadingZeros();
    int32_t y_lz = b.countLeadingZeros();
    if (aScale < bScale) {
      x_lz = minLeadingZerosAfterScaling(x_lz, bScale - aScale);
    } else if (aScale > bScale) {
      y_lz = minLeadingZerosAfterScaling(y_lz, aScale - bScale);
    }
    return std::min(x_lz, y_lz);
  }

  template <typename R, typename A, typename B>
  inline static R divideWithRoundUp(
      R& r,
      const A& a,
      const B& b,
      bool noRoundUp,
      uint8_t aRescale,
      uint8_t /*bRescale*/,
      bool* overflow) {
    if (b.unscaledValue() == 0) {
      *overflow = true;
      return R(-1);
    }
    int resultSign = 1;
    R unsignedDividendRescaled(a);
    int aSign = 1;
    int bSign = 1;
    if (a < 0) {
      resultSign = -1;
      unsignedDividendRescaled *= -1;
      aSign = -1;
    }
    R unsignedDivisor(b);
    if (b < 0) {
      resultSign *= -1;
      unsignedDivisor *= -1;
      bSign = -1;
    }
    auto bitsRequiredAfterScaling = maxBitsRequiredAfterScaling<A>(a, aRescale);
    if (bitsRequiredAfterScaling <= 127) {
      unsignedDividendRescaled = unsignedDividendRescaled.multiply(
          R(DecimalUtil::kPowersOfTen[aRescale]), overflow);
      if (*overflow) {
        return R(-1);
      }
      R quotient = unsignedDividendRescaled / unsignedDivisor;
      R remainder = unsignedDividendRescaled % unsignedDivisor;
      if (!noRoundUp && remainder * 2 >= unsignedDivisor) {
        ++quotient;
      }
      r = quotient * resultSign;
      return remainder;
    } else if constexpr (
        std::is_same_v<R, UnscaledShortDecimal> ||
        std::is_same_v<R, UnscaledLongDecimal>) {
      // Derives from Arrow BasicDecimal128 Divide
      if (aRescale > 38 && bitsRequiredAfterScaling > 255) {
        *overflow = true;
        return R(-1);
      }
      int256_t aLarge = a.unscaledValue();
      int256_t x_large_scaled_up = aLarge * DecimalUtil::kPowersOfTen[aRescale];
      int256_t y_large = b.unscaledValue();
      int256_t result_large = x_large_scaled_up / y_large;
      int256_t remainder_large = x_large_scaled_up % y_large;
      // Since we are scaling up and then, scaling down, round-up the result (+1
      // for +ve, -1 for -ve), if the remainder is >= 2 * divisor.
      if (abs(2 * remainder_large) >= abs(y_large)) {
        // x +ve and y +ve, result is +ve =>   (1 ^ 1)  + 1 =  0 + 1 = +1
        // x +ve and y -ve, result is -ve =>  (-1 ^ 1)  + 1 = -2 + 1 = -1
        // x +ve and y -ve, result is -ve =>   (1 ^ -1) + 1 = -2 + 1 = -1
        // x -ve and y -ve, result is +ve =>  (-1 ^ -1) + 1 =  0 + 1 = +1
        result_large += (aSign ^ bSign) + 1;
      }

      auto result = R::convert(result_large, overflow);
      auto remainder = R::convert(remainder_large, overflow);
      if (!R::valueInRange(result.unscaledValue())) {
        *overflow = true;
      } else {
        r = result;
      }
      return remainder;
    } else {
      VELOX_FAIL("Should not reach here in DecimalUtilOp.h");
    }
  }

  // Convert a number of scientific notation to normal.
  inline static std::string getNormalNumber(const std::string& value) {
    size_t dotPos = value.find('.');
    size_t expPos = value.find('E');
    if (expPos == std::string::npos) {
      return value;
    }

    std::string ints;
    std::string digits;
    // Get the integers and digits from the base number.
    if (dotPos == std::string::npos) {
      ints = value.substr(0, expPos);
      digits = "";
    } else {
      ints = value.substr(0, dotPos);
      digits = value.substr(dotPos + 1, expPos - dotPos - 1);
    }

    size_t pos = value.find("E+");
    // Handle number with positive exponent.
    if (pos != std::string::npos) {
      int exponent = std::stoi(value.substr(pos + 2, value.length()));
      std::string number = ints;
      if (exponent >= digits.length()) {
        // Dot is not needed.
        number = ints + digits;
        for (int i = 0; i < exponent - digits.length(); i++) {
          number += '0';
        }
      } else {
        number += digits.substr(0, exponent) + '.' +
            digits.substr(exponent + 1, digits.length());
      }
      return number;
    }
    pos = value.find("E-");
    if (pos != std::string::npos) {
      int exponent = std::stoi(value.substr(pos + 2, value.length()));
      std::string number;
      if (exponent < ints.length()) {
        number = ints.substr(0, ints.length() - exponent) + '.' +
            ints.substr(ints.length() - exponent + 1, ints.length());
      } else {
        number = "0.";
        for (int i = 0; i < exponent - ints.length(); i++) {
          number += '0';
        }
        number += ints;
        number += digits;
      }
      return number;
    }
    return value;
  }

  // Round double to certain precision with half up.
  inline static double roundTo(double value, int precision) {
    int charsNeeded = 1 + snprintf(NULL, 0, "%.*f", (int)precision, value);
    char* buffer = reinterpret_cast<char*>(malloc(charsNeeded));
    double nextValue;
    if (value < 0) {
      nextValue = nextafter(value, value - 0.1);
    } else {
      nextValue = nextafter(value, value + 0.1);
    }
    snprintf(buffer, charsNeeded, "%.*f", (int)precision, nextValue);
    return atof(buffer);
  }

  // return unscaled value and scale
  inline static std::pair<std::string, uint8_t> splitVarChar(
      const StringView& value,
      int toScale) {
    std::string s = getNormalNumber(value.str());
    size_t pos = s.find('.');
    if (pos == std::string::npos) {
      return {s.substr(0, pos), 0};
    } else if (toScale < s.length() - pos - 1) {
      // If toScale is less than scales.length(), the string scales will be cut
      // and rounded.
      std::string roundedValue = std::to_string(roundTo(std::stod(s), toScale));
      pos = roundedValue.find('.');
      std::string scales = roundedValue.substr(pos + 1, toScale);
      return {roundedValue.substr(0, pos) + scales, scales.length()};
    } else {
      std::string scales = s.substr(pos + 1, s.length());
      return {s.substr(0, pos) + scales, scales.length()};
    }
  }

  static int128_t convertStringToInt128(
      const std::string& value,
      bool& nullOutput) {
    // Handling integer target cases
    const char* v = value.c_str();
    nullOutput = true;
    bool negative = false;
    int128_t result = 0;
    int index = 0;
    int len = value.size();
    if (len == 0) {
      return -1;
    }
    // Setting negative flag
    if (v[0] == '-') {
      if (len == 1) {
        return -1;
      }
      negative = true;
      index = 1;
    }
    if (negative) {
      for (; index < len; index++) {
        if (!std::isdigit(v[index])) {
          return -1;
        }
        result = result * 10 - (v[index] - '0');
        // Overflow check
        if (result > 0) {
          return -1;
        }
      }
    } else {
      for (; index < len; index++) {
        if (!std::isdigit(v[index])) {
          return -1;
        }
        result = result * 10 + (v[index] - '0');
        // Overflow check
        if (result < 0) {
          return -1;
        }
      }
    }
    // Final result
    nullOutput = false;
    return result;
  }

  template <typename TOutput>
  inline static std::optional<TOutput> rescaleVarchar(
      const StringView& inputValue,
      const int toPrecision,
      const int toScale) {
    static_assert(
        std::is_same_v<TOutput, UnscaledShortDecimal> ||
        std::is_same_v<TOutput, UnscaledLongDecimal>);
    auto [unscaledStr, fromScale] = splitVarChar(inputValue, toScale);
    uint8_t fromPrecision = unscaledStr.size();
    VELOX_CHECK_LE(
        fromPrecision, DecimalType<TypeKind::LONG_DECIMAL>::kMaxPrecision);
    if (fromPrecision <= 18) {
      int64_t fromUnscaledValue = folly::to<int64_t>(unscaledStr);
      return DecimalUtil::rescaleWithRoundUp<UnscaledShortDecimal, TOutput>(
          UnscaledShortDecimal(fromUnscaledValue),
          fromPrecision,
          fromScale,
          toPrecision,
          toScale,
          false,
          true);
    } else {
      bool nullOutput = true;
      int128_t decimalValue = convertStringToInt128(unscaledStr, nullOutput);
      if (nullOutput) {
        VELOX_USER_FAIL(
            "Cannot cast StringView '{}' to DECIMAL({},{})",
            inputValue,
            toPrecision,
            toScale);
      }
      return DecimalUtil::rescaleWithRoundUp<UnscaledLongDecimal, TOutput>(
          UnscaledLongDecimal(decimalValue),
          fromPrecision,
          fromScale,
          toPrecision,
          toScale,
          false,
          true);
    }
  }

  template <typename TInput, typename TOutput>
  inline static std::optional<TOutput> rescaleDouble(
      const TInput inputValue,
      const int toPrecision,
      const int toScale) {
    static_assert(
        std::is_same_v<TOutput, UnscaledShortDecimal> ||
        std::is_same_v<TOutput, UnscaledLongDecimal>);
    auto str = velox::to<std::string>(inputValue);
    auto stringView = StringView(str.c_str(), str.size());
    return rescaleVarchar<TOutput>(stringView, toPrecision, toScale);
  }
};
} // namespace facebook::velox
