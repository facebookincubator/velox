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

std::string_view
detail::extractDigits(const char* s, size_t start, size_t size) {
  size_t pos = start;
  for (; pos < size; ++pos) {
    if (!std::isdigit(s[pos])) {
      break;
    }
  }
  return std::string_view(s + start, pos - start);
}

Status detail::parseDecimalComponents(
    const char* s,
    size_t size,
    detail::DecimalComponents& out) {
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
  out.wholeDigits = detail::extractDigits(s, pos, size);
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
    out.fractionalDigits = detail::extractDigits(s, pos, size);
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
    bool withSign = pos < size && (s[pos] == '+' || s[pos] == '-');
    if (withSign && pos == size - 1) {
      return Status::UserError("The exponent part only contains sign.");
    }
    // Make sure all chars after sign are digits, as as folly::tryTo allows
    // leading and trailing whitespaces.
    for (auto i = static_cast<size_t>(withSign); i < size - pos; ++i) {
      if (!std::isdigit(s[pos + i])) {
        return Status::UserError(
            "Non-digit character '{}' is not allowed in the exponent part.",
            s[pos + i]);
      }
    }
    out.exponent = folly::to<int32_t>(folly::StringPiece(s + pos, size - pos));
    return Status::OK();
  }
  return pos == size
      ? Status::OK()
      : Status::UserError(
            "Chars '{}' are invalid.", std::string(s + pos, size - pos));
}

Status detail::parseHugeInt(
    const DecimalComponents& decimalComponents,
    int128_t& out) {
  // Parse the whole digits.
  if (decimalComponents.wholeDigits.size() > 0) {
    const auto tryValue = folly::tryTo<int128_t>(folly::StringPiece(
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
        folly::StringPiece(decimalComponents.fractionalDigits.data(), length));
    if (tryValue.hasError()) {
      return Status::UserError("Value too large.");
    }
    overflow = __builtin_add_overflow(out, tryValue.value(), &out);
    VELOX_DCHECK(!overflow);
  }
  return Status::OK();
}

std::string DecimalUtil::toString(int128_t value, const TypePtr& type) {
  auto [precision, scale] = getDecimalPrecisionScale(*type);
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
        remTotal, remMul + remSum, count, true, 0, 0);
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

} // namespace facebook::velox
