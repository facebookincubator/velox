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

#include "velox/common/text/TextFieldParser.h"

#include <boost/algorithm/string/predicate.hpp>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

#include "velox/common/base/VeloxException.h"
#include "velox/common/encode/Base64.h"
#include "velox/type/Conversions.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/TimestampConversion.h"

namespace facebook::velox::text {

namespace {
std::optional<int64_t> parseInt64(std::string_view field) {
  if (field.empty()) {
    return std::nullopt;
  }

  const char first = field.front();
  if (first != '-' && (first < '0' || first > '9')) {
    return std::nullopt;
  }

  int64_t value{0};
  const char* const begin = field.data();
  const char* const end = begin + field.size();
  const auto [parseEnd, error] = std::from_chars(begin, end, value);
  if (error != std::errc{}) {
    return std::nullopt;
  }

  for (const char* current = parseEnd; current != end; ++current) {
    if (current == parseEnd && *current == '.') {
      continue;
    }
    if (*current >= '0' && *current <= '9') {
      continue;
    }
    return std::nullopt;
  }
  return value;
}

} // namespace

template <typename T>
std::optional<T> TextFieldParser::parseInteger(std::string_view field) {
  static_assert(
      std::is_signed_v<T> && std::is_integral_v<T>,
      "parseInteger requires a signed integral type.");
  const auto value = parseInt64(field);
  if (!value.has_value() || *value < std::numeric_limits<T>::min() ||
      *value > std::numeric_limits<T>::max()) {
    return std::nullopt;
  }
  return static_cast<T>(*value);
}

std::optional<bool> TextFieldParser::parseBoolean(std::string_view field) {
  switch (field.size()) {
    case 4:
      if (boost::algorithm::iequals(field, std::string_view{"TRUE"})) {
        return true;
      }
      break;
    case 5:
      if (boost::algorithm::iequals(field, std::string_view{"FALSE"})) {
        return false;
      }
      break;
    default:
      break;
  }
  return std::nullopt;
}

namespace {
bool unacceptableFloatingPoint(std::string_view field) {
  for (const char character : field) {
    if (!(std::isalpha(static_cast<unsigned char>(character)) ||
          character == '-')) {
      return false;
    }
  }

  return !boost::algorithm::iequals(field, std::string_view{"NaN"}) &&
      !boost::algorithm::iequals(field, std::string_view{"Infinity"}) &&
      !boost::algorithm::iequals(field, std::string_view{"Inf"}) &&
      !boost::algorithm::iequals(field, std::string_view{"-Infinity"}) &&
      !boost::algorithm::iequals(field, std::string_view{"-Inf"});
}

std::string_view trimJavaWhitespace(std::string_view field) {
  while (!field.empty() && static_cast<unsigned char>(field.front()) <= 0x20) {
    field.remove_prefix(1);
  }

  while (!field.empty() && static_cast<unsigned char>(field.back()) <= 0x20) {
    field.remove_suffix(1);
  }
  return field;
}

// Copies 'field' into a null-terminated buffer, optionally prefixed with '0'.
const char* nullTerminateFloatingPoint(
    std::string_view field,
    bool prependZero,
    std::array<char, 128>& stackBuffer,
    std::string& heapBuffer) {
  const size_t normalizedSize = field.size() + prependZero;
  if (normalizedSize < stackBuffer.size()) {
    size_t offset{0};
    if (prependZero) {
      stackBuffer[offset++] = '0';
    }
    std::memcpy(stackBuffer.data() + offset, field.data(), field.size());
    stackBuffer[normalizedSize] = '\0';
    return stackBuffer.data();
  }

  heapBuffer.reserve(normalizedSize);
  if (prependZero) {
    heapBuffer.push_back('0');
  }
  heapBuffer.append(field);
  return heapBuffer.c_str();
}
} // namespace

template <typename T>
std::optional<T> TextFieldParser::parseFloatingPoint(std::string_view field) {
  static_assert(
      std::is_floating_point_v<T>,
      "parseFloatingPoint requires a floating-point type.");
  const auto trimmed = trimJavaWhitespace(field);
  if (trimmed.empty()) {
    return std::nullopt;
  }
  if (unacceptableFloatingPoint(trimmed)) {
    return std::nullopt;
  }

  const bool prependZero = trimmed.front() == '.';
  const size_t normalizedSize = trimmed.size() + prependZero;
  std::array<char, 128> stackBuffer;
  std::string heapBuffer;
  const char* const normalized =
      nullTerminateFloatingPoint(trimmed, prependZero, stackBuffer, heapBuffer);

  T value{0};
  long long scanPosition{0};
  int scanCount{0};
  if constexpr (std::is_same_v<T, float>) {
    scanCount = std::sscanf(normalized, "%f%lln", &value, &scanPosition);
  } else {
    scanCount = std::sscanf(normalized, "%lf%lln", &value, &scanPosition);
  }
  if (scanCount != 1 || static_cast<size_t>(scanPosition) != normalizedSize) {
    return std::nullopt;
  }
  return value;
}

template <typename T>
Expected<T> TextFieldParser::parseDecimal(
    std::string_view field,
    uint8_t precision,
    uint8_t scale) {
  T value{0};
  const auto status = DecimalUtil::castFromString(
      StringView(field.data(), static_cast<int32_t>(field.size())),
      precision,
      scale,
      value);
  if (!status.ok()) {
    return folly::makeUnexpected(status);
  }
  return value;
}

Expected<int32_t> TextFieldParser::parseDate(std::string_view field) {
  try {
    return util::fromDateString(
        field.data(), field.size(), util::ParseMode::kPrestoCast);
  } catch (const VeloxUserError& error) {
    return folly::makeUnexpected(Status::UserError("{}", error.message()));
  }
}

Expected<Timestamp> TextFieldParser::parseTimestamp(
    std::string_view field,
    const tz::TimeZone& timeZone) {
  try {
    auto result = util::Converter<TypeKind::TIMESTAMP>::tryCast(field);
    if (result.hasError()) {
      return folly::makeUnexpected(result.error());
    }
    auto timestamp = result.value();
    timestamp.toGMT(timeZone);
    return timestamp;
  } catch (const VeloxUserError& error) {
    return folly::makeUnexpected(Status::UserError("{}", error.message()));
  }
}

std::string_view TextFieldParser::parseVarbinary(
    std::string_view field,
    std::span<char> output) {
  // calculateDecodedSize strips padding from 'unpaddedSize'; decode() expects
  // the original size.
  auto unpaddedSize = field.size();
  const auto decodedSize =
      encoding::Base64::calculateDecodedSize(field.data(), unpaddedSize);
  if (!decodedSize.hasValue()) {
    return field;
  }
  VELOX_CHECK_GE(output.size(), decodedSize.value());
  if (decodedSize.value() == 0) {
    return {};
  }

  const auto status = encoding::Base64::decode(
      field.data(), field.size(), output.data(), decodedSize.value());
  if (!status.ok()) {
    return field;
  }
  return {output.data(), decodedSize.value()};
}

template std::optional<int8_t> TextFieldParser::parseInteger<int8_t>(
    std::string_view);
template std::optional<int16_t> TextFieldParser::parseInteger<int16_t>(
    std::string_view);
template std::optional<int32_t> TextFieldParser::parseInteger<int32_t>(
    std::string_view);
template std::optional<int64_t> TextFieldParser::parseInteger<int64_t>(
    std::string_view);
template std::optional<float> TextFieldParser::parseFloatingPoint<float>(
    std::string_view);
template std::optional<double> TextFieldParser::parseFloatingPoint<double>(
    std::string_view);
template Expected<int64_t>
TextFieldParser::parseDecimal<int64_t>(std::string_view, uint8_t, uint8_t);
template Expected<int128_t>
TextFieldParser::parseDecimal<int128_t>(std::string_view, uint8_t, uint8_t);

} // namespace facebook::velox::text
