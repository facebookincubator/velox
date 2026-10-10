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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "velox/common/base/Status.h"
#include "velox/type/HugeInt.h"
#include "velox/type/Timestamp.h"

namespace facebook::velox::tz {
class TimeZone;
} // namespace facebook::velox::tz

namespace facebook::velox::text {

/// Converts delimited-text field values to Velox types using the semantics of
/// the Hive text format reader (velox/dwio/text). Each method parses a single,
/// already-split and unescaped field.
///
/// Integer, boolean and floating-point parsers return std::nullopt on failure:
/// malformed values are common in text data and map to null, so these hot paths
/// avoid building an error. Decimal, date and timestamp parsers delegate to
/// Velox cast conversions and return their error so callers can either report
/// it or map it to null.
class TextFieldParser {
 public:
  /// Parses a signed integer (int8, int16, int32, int64) with overflow
  /// checking against `T`'s range. A leading '+' is rejected.
  /// Trailing characters are accepted only when they form a decimal
  /// continuation, e.g. "123.45" is parsed as 123.
  template <typename T>
  static std::optional<T> parseInteger(std::string_view field);

  /// Parses a boolean from `field`. Accepts case-insensitive "TRUE"/"FALSE".
  /// Returns std::nullopt for any other input, including "0", "1", empty
  /// input, or values with surrounding whitespace.
  static std::optional<bool> parseBoolean(std::string_view field);

  /// Parses a floating-point value. Bytes 0x00 through 0x20 are trimmed,
  /// special values (NaN, Infinity, Inf, -Infinity, -Inf) are matched
  /// case-insensitively, and other syntax accepted by C sscanf is allowed,
  /// including a leading '+' and hexadecimal. Out-of-range values are not
  /// rejected, since denormals and infinities are acceptable. sscanf honors the
  /// process locale, so a locale whose decimal point is not '.' changes the
  /// accepted syntax.
  template <typename T>
  static std::optional<T> parseFloatingPoint(std::string_view field);

  /// Parses a decimal value using Velox's decimal cast conversion.
  template <typename T>
  static Expected<T>
  parseDecimal(std::string_view field, uint8_t precision, uint8_t scale);

  /// Parses a date using Velox's Presto-cast date semantics and returns the
  /// number of days since 1970-01-01.
  static Expected<int32_t> parseDate(std::string_view field);

  /// Parses a timestamp using Velox's Presto-cast timestamp semantics,
  /// interpreting the value as local time in `timeZone` and converting it to
  /// GMT. Ambiguous local times resolve to the earlier instant. Returns an
  /// error if `field` cannot be parsed or the local time does not exist in
  /// `timeZone`, e.g. during a daylight saving time gap.
  static Expected<Timestamp> parseTimestamp(
      std::string_view field,
      const tz::TimeZone& timeZone);

  /// Decodes a base64-encoded VARBINARY field into caller-owned storage.
  /// `output` must hold at least the decoded size; `field.size()` bytes is
  /// always enough. Invalid base64 is returned as-is rather than rejected, for
  /// compatibility with other readers of files that store raw binary in
  /// VARBINARY columns. The returned view aliases `output` for valid base64 or
  /// `field` otherwise, so both must outlive it.
  static std::string_view parseVarbinary(
      std::string_view field,
      std::span<char> output);
};

extern template std::optional<int8_t> TextFieldParser::parseInteger<int8_t>(
    std::string_view);
extern template std::optional<int16_t> TextFieldParser::parseInteger<int16_t>(
    std::string_view);
extern template std::optional<int32_t> TextFieldParser::parseInteger<int32_t>(
    std::string_view);
extern template std::optional<int64_t> TextFieldParser::parseInteger<int64_t>(
    std::string_view);
extern template std::optional<float> TextFieldParser::parseFloatingPoint<float>(
    std::string_view);
extern template std::optional<double>
    TextFieldParser::parseFloatingPoint<double>(std::string_view);
extern template Expected<int64_t>
TextFieldParser::parseDecimal<int64_t>(std::string_view, uint8_t, uint8_t);
extern template Expected<int128_t>
TextFieldParser::parseDecimal<int128_t>(std::string_view, uint8_t, uint8_t);

} // namespace facebook::velox::text
