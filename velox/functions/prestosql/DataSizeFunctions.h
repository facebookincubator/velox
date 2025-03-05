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

#include <boost/lexical_cast.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>
#include <map>
#include <string>
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/ArithmeticImpl.h"
#include "velox/type/Conversions.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions {

enum class Unit : int128_t {
  BYTE = 1,
  KILOBYTE = int128_t(1) << 10,
  MEGABYTE = int128_t(1) << 20,
  GIGABYTE = int128_t(1) << 30,
  TERABYTE = int128_t(1) << 40,
  PETABYTE = int128_t(1) << 50,
  EXABYTE = int128_t(1) << 60,
  ZETTABYTE = int128_t(1) << 70,
  YOTTABYTE = int128_t(1) << 80
};

inline Unit parseUnit(const std::string& dataSize, const size_t& valueLength) {
  static const std::map<std::string, Unit> unitMap = {
      {"B", Unit::BYTE},
      {"kB", Unit::KILOBYTE},
      {"MB", Unit::MEGABYTE},
      {"GB", Unit::GIGABYTE},
      {"TB", Unit::TERABYTE},
      {"PB", Unit::PETABYTE},
      {"EB", Unit::EXABYTE},
      {"ZB", Unit::ZETTABYTE},
      {"YB", Unit::YOTTABYTE}};
  try {
    std::string unitString = dataSize.substr(valueLength);
    auto it = unitMap.find(unitString);
    VELOX_USER_CHECK(
        it != unitMap.end(),
        "Invalid data size: '{} NOT FOUND {}'",
        dataSize,
        unitString);
    return it->second;
  } catch (const std::exception&) {
    VELOX_USER_FAIL("Invalid data size: '{}'", dataSize);
  }
}

/// Represent the varchar fragment.
///
/// For example:
/// | value | wholeDigits | fractionalDigits | exponent | sign |
/// | 9999999999.99 | 9999999999 | 99 | nullopt | 1 |
/// | 15 | 15 |  | nullopt | 1 |
/// | 1.5 | 1 | 5 | nullopt | 1 |
/// | -1.5 | 1 | 5 | nullopt | -1 |
/// | 31.523e-2 | 31 | 523 | -2 | 1 |
struct DecimalComponents {
  std::string_view wholeDigits;
  std::string_view fractionalDigits;
  std::optional<int32_t> exponent = std::nullopt;
  int8_t sign = 1;
};

std::string_view extractDigits(const char* s, size_t start, size_t size) {
  size_t pos = start;
  for (; pos < size; ++pos) {
    if (!std::isdigit(s[pos])) {
      break;
    }
  }
  return std::string_view(s + start, pos - start);
}

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

Status parseHugeInt(const DecimalComponents& decimalComponents, int128_t& out) {
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

template <typename T>
Status toDecimalValue(
    const StringView s,
    int toPrecision,
    int toScale,
    T& decimalValue) {
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
  int32_t parsedPrecision = static_cast<int32_t>(significantDigits);

  int32_t parsedScale = 0;
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

  int128_t out = 0;
  if (auto status = parseHugeInt(decimalComponents, out); !status.ok()) {
    return status;
  }

  if (roundUp) {
    bool overflow = __builtin_add_overflow(out, 1, &out);
    if (UNLIKELY(overflow)) {
      return Status::UserError("Value too large.");
    }
  }
  out *= decimalComponents.sign;

  if (parsedScale < 0) {
    /// Force the scale to be zero, to avoid negative scales (due to
    /// compatibility issues with external systems such as databases).
    if (-parsedScale + toScale > LongDecimalType::kMaxPrecision) {
      return Status::UserError("Value too large.");
    }

    bool overflow = __builtin_mul_overflow(
        out, DecimalUtil::kPowersOfTen[-parsedScale + toScale], &out);
    if (UNLIKELY(overflow)) {
      return Status::UserError("Value too large.");
    }
    parsedPrecision -= parsedScale;
    parsedScale = toScale;
  }
  const auto status = DecimalUtil::rescaleWithRoundUp<int128_t, T>(
      out,
      std::min((uint8_t)parsedPrecision, LongDecimalType::kMaxPrecision),
      parsedScale,
      toPrecision,
      toScale,
      decimalValue);
  if (!status.ok()) {
    return Status::UserError("Value too large.");
  }
  return status;
}

void getPrecisionAndScale(
    const std::string& decimalString,
    size_t& precision,
    size_t& scale) {
  size_t valueLength = 0;
  size_t decimalIndex = std::string::npos;

  // Find the length of the numeric part and the index of the decimal point.
  while (valueLength < decimalString.length() &&
         (isdigit(decimalString[valueLength]) ||
          decimalString[valueLength] == '.')) {
    if (decimalString[valueLength] == '.') {
      decimalIndex = valueLength;
    }
    valueLength++;
  }
  // Calculate precision and scale.
  precision = valueLength - (decimalIndex != std::string::npos ? 1 : 0);
  scale = (decimalIndex != std::string::npos) ? (valueLength - decimalIndex - 1)
                                              : 0;
}

// TODO - how to create Long Decimal to store toDecimalValue result, this is a
// private constructor
template <typename P1, typename S1>
inline LongDecimal<P1, S1> parseDecimal(
    const std::string& dataSize,
    const std::string& str,
    const size_t& precision,
    const size_t& scale) {
  try {
    LongDecimal<P1, S1> result;
    // todo handle status error
    Status status = toDecimalValue(StringView(str), precision, scale, result);
    if (status.ok()) {
      return result;
    }
    VELOX_USER_FAIL("Invalid data size: '{}'", dataSize);
  } catch (const std::exception&) {
    VELOX_USER_FAIL("Invalid data size: '{}'", dataSize);
  }
}

template <typename P1, typename S1>
inline LongDecimal<P1, S1> parseStringToDecimal(
    const std::string& dataSize,
    const std::string& str,
    int precision,
    int scale) {
  try {
    std::string value = dataSize.substr(0, precision + 1);
    LongDecimal<P1, S1> result;
    // todo handle status error
    Status status = toDecimalValue(StringView(value), precision, scale, result);
    if (status.ok()) {
      return result;
    }
    VELOX_USER_FAIL("Invalid data size: '{}'", dataSize);
  } catch (const std::exception&) {
    VELOX_USER_FAIL("Invalid data size: '{}'", dataSize);
  }
}

template <typename R, typename A, typename B>
void decimalMultiplyFunction(R& out, const A& a, const B& b) {
  out = checkedMultiply<R>(checkedMultiply<R>(R(a), R(b)), R(1));
  DecimalUtil::valueInRange(out);
}

inline int128_t getDecimal(const std::string& decimalString) {
  size_t precision = 0;
  size_t scale = 0;
  getPrecisionAndScale(decimalString, precision, scale);
  VELOX_USER_CHECK_GT(precision, 0, "Invalid data size: '{}'", decimalString);
  // TODO - Change parameters use calculated precision, scale. Using output
  // parameters (38,0) here truncates information after decimal point.
  std::string valueStr = decimalString.substr(0, precision + 1);
  LongDecimal<P1, S1> value =
      parseDecimal<P1, S1>(decimalString, valueStr, precision, scale);
  // TODO - any better way to convert int128_t to LongDecimal<P1, S1> ?
  Unit unit = parseUnit(decimalString, precision + 1);
  int128_t unitValue = static_cast<int128_t>(unit);
  std::string unitStr = boost::lexical_cast<std::string>(unitValue);
  LongDecimal<P1, S1> factor =
      parseDecimal<P1, S1>(decimalString, unitStr, precision, scale);
  int128_t scaledValue;
  // LongDecimal<P1, S1> scaledValue;
  try {
    // TODO - once LongDecimal<P1, S1> is working above, use correct
    // multiplication code.
    decimalMultiplyFunction(scaledValue, value, factor);
    // TODO - LongDecimal<P1, S1> scaledValue input doesn't work with decimal
    // utils below, check after fixing other parts of code.
    std::vector<char> encodedValue(
        DecimalUtil::getByteArrayLength(scaledValue));
    DecimalUtil::toByteArray(scaledValue, encodedValue.data());
  } catch (const std::exception&) {
    VELOX_USER_FAIL(
        "Value out of range: '{}' ('{}B')", decimalString, scaledValue);
  }
  return scaledValue;
}

template <typename TExec>
struct ParsePrestoDataSizeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);
  FOLLY_ALWAYS_INLINE void call(
      out_type<LongDecimal<P1, S1>>& result,
      const arg_type<Varchar>& input) {
    std::string inputUtf8 = std::string(input.data(), input.size());
    result = getDecimal(inputUtf8);
  }
};

} // namespace facebook::velox::functions

// template <typename P1, typename S1>
// inline int128_t getDecimal(const std::string& dataSize) {
//   size_t valueLength = 0;
//   size_t decimalIndex = std::string::npos;

//   // Find the length of the numeric part and the index of the decimal point.
//   while (valueLength < dataSize.length() &&
//          (isdigit(dataSize[valueLength]) || dataSize[valueLength] == '.')) {
//     if (dataSize[valueLength] == '.') {
//       decimalIndex = valueLength;
//     }
//     valueLength++;
//   }
//   VELOX_USER_CHECK_GT(valueLength, 0, "Invalid data size: '{}'", dataSize);
//   std::string valueStr = dataSize.substr(0, valueLength);

//   // Q - how to use different template
//   LongDecimal<P1, S1> value =
//       parseStringToDecimal<P1, S1>(dataSize, valueStr, 38, 0);

//   Unit unit = parseUnit(dataSize, valueLength);
//   std::string unitStr =
//       DecimalUtil::toString(static_cast<int128_t>(unit), DECIMAL(38, 0));
//   LongDecimal<P1, S1> factor =
//       parseStringToDecimal<P1, S1>(dataSize, unitStr, 38, 0);

//   // TODO: convert the input to a decimal and use decimal arithmetic when
//   // scaling it
//   //  up to the specified Unit similar to what presto java does.
//   // The decimalarithmetic multiplier uses int128_t ..
//   auto scaledValue = value * factor;

// VELOX_USER_FAIL("Value {}", value);
// uint8_t precision1, const uint8_t scale2;
//   DecimalType<TypeKind::HUGEINT> kDecimalType(precision1, scale2);
//   DecimalType<TypeKind::HUGEINT> kDecimalType(38, 0);
//   LongDecimalType kDecimalType(38, 0);
// int128_t result;
// auto status = toDecimalValue(StringView(value), precision, scale, result);
