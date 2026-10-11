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
  VELOX_DCHECK_LT(scale, std::size(DecimalUtil::kPowersOfTen));
  const bool isFraction = (scale > 0);
  if (unscaledValue == 0) {
    if (isFraction) {
      return fmt::format("{:.{}f}", 0.0, scale);
    }
    return "0";
  }

  bool isNegative = (unscaledValue < 0);
  if (isNegative) {
    unscaledValue = ~unscaledValue + 1;
  }
  int128_t integralPart = unscaledValue / DecimalUtil::kPowersOfTen[scale];

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
} // namespace

std::string DecimalUtil::toString(int128_t value, const Type& type) {
  auto [precision, scale] = getDecimalPrecisionScale(type);
  return formatDecimal(scale, value);
}

int32_t DecimalUtil::getByteArrayLength(int128_t value) {
  if (value < 0) {
    value = ~value;
  }
  int nbits;
  if (auto hi = HugeInt::upper(value)) {
    nbits = 128 - bits::countLeadingZeros(hi);
  } else if (auto lo = HugeInt::lower(value)) {
    nbits = 64 - bits::countLeadingZeros(lo);
  } else {
    nbits = 0;
  }
  return 1 + nbits / 8;
}

int32_t DecimalUtil::toByteArray(int128_t value, char* out) {
  int32_t length = getByteArrayLength(value);
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
  return length;
}

void DecimalUtil::computeAverage(
    int128_t& avg,
    int128_t sum,
    int64_t count,
    int64_t overflow) {
  if (overflow == 0) {
    divideWithRoundUp<int128_t, int128_t, int64_t>(
        avg, sum, count, false, 0, 0);
  } else {
    // The average is computed as `(sum + overflow * kOverflowMultiplier) /
    // count`:
    //  1. sum / count → quotient quotSum, remainder remSum.
    //  2. (overflow * kOverflowMultiplier) / count → quotient quotMul,
    //  remainder remMul.
    //  3. (remSum + remMul) / count -> remTotal
    //  4. final result = quotSum + quotMul + remTotal
    // Since we only use quotients (no rounding needed, set "noRoundUp = true")
    // and combine remainders to compute the final average, rounding is applied
    // based on the combined remainder values (set "noRoundUp = false"), which
    // ensures accuracy.
    VELOX_DCHECK_LE(overflow, count);
    __uint128_t quotMul, remMul;
    __int128_t quotSum, remSum;
    __int128_t remTotal;
    remMul = DecimalUtil::divideWithRoundUp<__uint128_t, __uint128_t, int64_t>(
        quotMul, kOverflowMultiplier, count, true, 0, 0);
    remMul *= overflow;
    quotMul *= overflow;
    remSum = DecimalUtil::divideWithRoundUp<__int128_t, __int128_t, int64_t>(
        quotSum, sum, count, true, 0, 0);
    DecimalUtil::divideWithRoundUp<__int128_t, __int128_t, int64_t>(
        remTotal, remMul + remSum, count, false, 0, 0);
    avg = quotMul + quotSum + remTotal;
  }
}

int32_t DecimalUtil::maxStringViewSize(int precision, int scale) {
  int32_t rowSize = precision + 1; // Number and symbol.
  if (scale > 0) {
    ++rowSize; // A dot.
  }
  if (precision == scale) {
    ++rowSize; // Leading zero.
  }
  return rowSize;
}

namespace {

// Represent the varchar fragment.
//
// For example:
// | value | wholeDigits | fractionalDigits | exponent | sign |
// | 9999999999.99 | 9999999999 | 99 | nullopt | 1 |
// | 15 | 15 |  | nullopt | 1 |
// | 1.5 | 1 | 5 | nullopt | 1 |
// | -1.5 | 1 | 5 | nullopt | -1 |
// | 31.523e-2 | 31 | 523 | -2 | 1 |
struct DecimalComponents {
  std::string_view wholeDigits;
  std::string_view fractionalDigits;
  std::optional<int32_t> exponent = std::nullopt;
  int8_t sign = 1;
};

// Extract a string view of continuous digits.
std::string_view extractDigits(const char* s, size_t start, size_t size) {
  size_t pos = start;
  for (; pos < size; ++pos) {
    if (!std::isdigit(s[pos])) {
      break;
    }
  }
  return std::string_view(s + start, pos - start);
}

// Parse decimal components, including whole digits, fractional digits,
// exponent and sign, from input chars. Returns error status if input chars
// do not represent a valid value.
Status
parseDecimalComponents(const char* s, size_t size, DecimalComponents& out) {
  if (size == 0) {
    return Status::UserError("Input is empty.");
  }

  size_t pos = 0;

  // Sign of the number.
  if (s[pos] == '-') {
    out.sign = -1;
    ++pos;
  } else if (s[pos] == '+') {
    out.sign = 1;
    ++pos;
  }

  // Extract the whole digits.
  out.wholeDigits = extractDigits(s, pos, size);
  pos += out.wholeDigits.size();
  if (pos == size) {
    return out.wholeDigits.empty()
        ? Status::UserError("Extracted digits are empty.")
        : Status::OK();
  }

  // Optional dot (if given in fractional form).
  if (s[pos] == '.') {
    // Extract the fractional digits.
    ++pos;
    out.fractionalDigits = extractDigits(s, pos, size);
    pos += out.fractionalDigits.size();
  }

  if (out.wholeDigits.empty() && out.fractionalDigits.empty()) {
    return Status::UserError("Extracted digits are empty.");
  }
  if (pos == size) {
    return Status::OK();
  }
  // Optional exponent.
  if (s[pos] == 'e' || s[pos] == 'E') {
    ++pos;
    if (pos == size) {
      return Status::UserError("The exponent part is empty.");
    }
    bool withSign = s[pos] == '+' || s[pos] == '-';
    if (withSign && pos == size - 1) {
      return Status::UserError("The exponent part only contains sign.");
    }
    // Make sure all chars after sign are digits, as folly::tryTo allows
    // leading and trailing whitespaces.
    for (auto i = static_cast<size_t>(withSign); i < size - pos; ++i) {
      if (!std::isdigit(s[pos + i])) {
        return Status::UserError(
            "Non-digit character is not allowed in the exponent part.");
      }
    }
    out.exponent = folly::to<int32_t>(std::string_view(s + pos, size - pos));
    return Status::OK();
  }
  return pos == size ? Status::OK() : Status::UserError("Chars are invalid.");
}

// Parse huge int from decimal components. The fractional part is scaled up by
// required power of 10, and added with the whole part. Returns error status
// if overflows.
Status parseHugeInt(const DecimalComponents& decimalComponents, int128_t& out) {
  // Parse the whole digits.
  if (decimalComponents.wholeDigits.size() > 0) {
    const auto tryValue = folly::tryTo<int128_t>(std::string_view(
        decimalComponents.wholeDigits.data(),
        decimalComponents.wholeDigits.size()));
    if (tryValue.hasError()) {
      return Status::UserError("Value too large.");
    }
    out = tryValue.value();
  }

  // Parse the fractional digits.
  if (decimalComponents.fractionalDigits.size() > 0) {
    const auto length = decimalComponents.fractionalDigits.size();
    bool overflow =
        __builtin_mul_overflow(out, DecimalUtil::kPowersOfTen[length], &out);
    if (overflow) {
      return Status::UserError("Value too large.");
    }
    const auto tryValue = folly::tryTo<int128_t>(
        std::string_view(decimalComponents.fractionalDigits.data(), length));
    if (tryValue.hasError()) {
      return Status::UserError("Value too large.");
    }
    overflow = __builtin_add_overflow(out, tryValue.value(), &out);
    VELOX_DCHECK(!overflow);
  }
  return Status::OK();
}
} // namespace

Status DecimalUtil::normalizeScientificNotation(
    const StringView& input,
    int32_t precision,
    int32_t scale,
    std::string& normalized,
    bool& isZero) {
  normalized.clear();
  isZero = false;

  std::string_view text(input.data(), input.size());
  const auto exponentPosition = text.find_first_of("eE");
  if (exponentPosition == std::string_view::npos) {
    return Status::OK();
  }

  auto exponentText = text.substr(exponentPosition + 1);
  bool negativeExponent{false};
  if (!exponentText.empty() &&
      (exponentText.front() == '+' || exponentText.front() == '-')) {
    negativeExponent = exponentText.front() == '-';
    exponentText.remove_prefix(1);
  }
  if (exponentText.empty()) {
    return Status::OK();
  }

  uint64_t exponentMagnitude{0};
  const auto [exponentEnd, exponentError] = std::from_chars(
      exponentText.data(),
      exponentText.data() + exponentText.size(),
      exponentMagnitude);
  if (exponentError == std::errc::invalid_argument ||
      exponentEnd != exponentText.data() + exponentText.size()) {
    return Status::OK();
  }
  const uint64_t maxExponentMagnitude = negativeExponent
      ? static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1
      : static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  if (exponentError == std::errc::result_out_of_range ||
      exponentMagnitude > maxExponentMagnitude) {
    return Status::UserError(
        "Value is not a number. Exponent is out of range.");
  }

  auto mantissa = text.substr(0, exponentPosition);
  bool negative{false};
  if (!mantissa.empty() &&
      (mantissa.front() == '+' || mantissa.front() == '-')) {
    negative = mantissa.front() == '-';
    mantissa.remove_prefix(1);
  }

  std::string digits;
  digits.reserve(mantissa.size());
  bool sawDecimalPoint{false};
  size_t numFractionDigits{0};
  for (const char character : mantissa) {
    if (character == '.') {
      if (sawDecimalPoint) {
        return Status::OK();
      }
      sawDecimalPoint = true;
    } else if (character >= '0' && character <= '9') {
      digits.push_back(character);
      if (sawDecimalPoint) {
        ++numFractionDigits;
      }
    } else {
      return Status::OK();
    }
  }
  if (digits.empty()) {
    return Status::OK();
  }

  const int64_t exponent = negativeExponent
      ? -static_cast<int64_t>(exponentMagnitude)
      : static_cast<int64_t>(exponentMagnitude);
  const int64_t decimalScale =
      static_cast<int64_t>(numFractionDigits) - exponent;
  if (decimalScale < std::numeric_limits<int32_t>::min() ||
      decimalScale > std::numeric_limits<int32_t>::max()) {
    return Status::UserError("Value too large.");
  }

  const auto firstNonZero = digits.find_first_not_of('0');
  if (firstNonZero == std::string::npos) {
    isZero = true;
    return Status::OK();
  }
  digits.erase(0, firstNonZero);

  if (negative) {
    normalized.push_back('-');
  }
  if (decimalScale <= 0) {
    const auto numTrailingZeros = static_cast<uint64_t>(-decimalScale);
    if (numTrailingZeros > static_cast<uint64_t>(precision) ||
        digits.size() > static_cast<uint64_t>(precision) - numTrailingZeros) {
      return Status::UserError("Value too large.");
    }
    normalized.append(digits);
    normalized.append(numTrailingZeros, '0');
  } else if (static_cast<uint64_t>(decimalScale) >= digits.size()) {
    const auto numLeadingZeros =
        static_cast<uint64_t>(decimalScale) - digits.size();
    if (numLeadingZeros > static_cast<uint64_t>(scale)) {
      isZero = true;
      normalized.clear();
      return Status::OK();
    }
    normalized.append("0.");
    normalized.append(numLeadingZeros, '0');
    normalized.append(digits);
  } else {
    const auto numIntegerDigits = digits.size() - decimalScale;
    if (numIntegerDigits > static_cast<uint64_t>(precision)) {
      return Status::UserError("Value too large.");
    }
    normalized.append(digits.data(), numIntegerDigits);
    normalized.push_back('.');
    normalized.append(digits.data() + numIntegerDigits, decimalScale);
  }
  return Status::OK();
}

Status DecimalUtil::parseStringToDecimalComponents(
    const StringView& s,
    int32_t toScale,
    int32_t& parsedPrecision,
    int32_t& parsedScale,
    int128_t& out) {
  DecimalComponents decimalComponents;
  if (auto status =
          parseDecimalComponents(s.data(), s.size(), decimalComponents);
      !status.ok()) {
    return Status::UserError("Value is not a number. " + status.message());
  }

  // Count number of significant digits (without leading zeros).
  const size_t firstNonZero =
      decimalComponents.wholeDigits.find_first_not_of('0');
  size_t significantDigits = decimalComponents.fractionalDigits.size();
  if (firstNonZero != std::string::npos) {
    significantDigits += decimalComponents.wholeDigits.size() - firstNonZero;
  }
  parsedPrecision = static_cast<int32_t>(significantDigits);

  parsedScale = 0;
  bool roundUp = false;
  const int32_t fractionSize = decimalComponents.fractionalDigits.size();
  if (!decimalComponents.exponent.has_value()) {
    if (fractionSize > toScale) {
      if (decimalComponents.fractionalDigits[toScale] >= '5') {
        roundUp = true;
      }
      parsedScale = toScale;
      decimalComponents.fractionalDigits =
          std::string_view(decimalComponents.fractionalDigits.data(), toScale);
    } else {
      parsedScale = fractionSize;
    }
  } else {
    const auto exponent = decimalComponents.exponent.value();
    parsedScale = -exponent + fractionSize;
    // Truncate the fractionalDigits.
    if (parsedScale > toScale) {
      if (-exponent >= toScale) {
        // The fractional digits could be dropped.
        if (fractionSize > 0 && decimalComponents.fractionalDigits[0] >= '5') {
          roundUp = true;
        }
        decimalComponents.fractionalDigits = "";
        parsedScale -= fractionSize;
      } else {
        const auto reduceDigits = exponent + toScale;
        if (fractionSize > reduceDigits &&
            decimalComponents.fractionalDigits[reduceDigits] >= '5') {
          roundUp = true;
        }
        decimalComponents.fractionalDigits = std::string_view(
            decimalComponents.fractionalDigits.data(),
            std::min(reduceDigits, fractionSize));
        parsedScale -= fractionSize - decimalComponents.fractionalDigits.size();
      }
    }
  }

  VELOX_RETURN_NOT_OK(parseHugeInt(decimalComponents, out));

  if (roundUp) {
    bool overflow = __builtin_add_overflow(out, 1, &out);
    if (UNLIKELY(overflow)) {
      return Status::UserError("Value too large.");
    }
  }
  out *= decimalComponents.sign;

  if (parsedScale < 0) {
    // Force the scale to be zero, to avoid negative scales (due to
    // compatibility issues with external systems such as databases).
    if (-parsedScale + toScale > LongDecimalType::kMaxPrecision) {
      return Status::UserError("Value too large.");
    }

    bool overflow =
        __builtin_mul_overflow(out, kPowersOfTen[-parsedScale + toScale], &out);
    if (UNLIKELY(overflow)) {
      return Status::UserError("Value too large.");
    }
    parsedPrecision -= parsedScale;
    parsedScale = toScale;
  }
  return Status::OK();
}

} // namespace facebook::velox
