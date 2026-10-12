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
#include <string>
#include <string_view>

#include "velox/common/base/Status.h"
#include "velox/type/HugeInt.h"
#include "velox/type/Timestamp.h"

namespace facebook::velox::text {

/// Groups the canonical field-level parsers shared by the Hive text-file
/// reader (TextReader) and the Spark `from_csv` special form (FromCsv).
class TextFieldParser {
 public:
  /// Parses a narrow signed integer (int8, int16, int32, int64) with
  /// overflow checking against `T`'s range. A leading '+' is rejected.
  /// Trailing characters are accepted only when they form a decimal
  /// continuation, e.g. "123.45" is parsed as 123.
  template <typename T>
  static std::optional<T> parseNarrowInteger(std::string_view field);

  /// Parses a boolean from `field`. Accepts case-insensitive "TRUE"/"FALSE".
  /// Returns std::nullopt for any other input, including "0", "1", empty
  /// input, or values with surrounding whitespace.
  static std::optional<bool> parseBoolean(std::string_view field);

  /// Parses floating-point values using the canonical text-reader semantics.
  /// Bytes 0x00 through 0x20 are trimmed, special values are matched
  /// case-insensitively, and hexadecimal syntax is accepted. Conversion uses
  /// the C numeric locale regardless of the calling thread's locale.
  template <typename T>
  static std::optional<T> parseFloatingPoint(std::string_view field);

  /// Parses a decimal value using Velox's canonical decimal conversion.
  template <typename T>
  static Expected<T>
  parseDecimal(std::string_view field, uint8_t precision, uint8_t scale);

  /// Parses a date using Velox's Presto-cast date semantics.
  static Expected<int32_t> parseDate(std::string_view field);

  /// Parses a timestamp using Velox's Presto-cast timestamp semantics.
  /// Interprets naive values in America/Los_Angeles and converts them to GMT.
  static Expected<Timestamp> parseTimestamp(std::string_view field);

  /// Decodes a base64-encoded VARBINARY field. Invalid base64 is copied
  /// unchanged for compatibility with the text reader.
  static void parseVarbinary(std::string_view field, std::string& output);
};

extern template std::optional<int8_t>
    TextFieldParser::parseNarrowInteger<int8_t>(std::string_view);
extern template std::optional<int16_t>
    TextFieldParser::parseNarrowInteger<int16_t>(std::string_view);
extern template std::optional<int32_t>
    TextFieldParser::parseNarrowInteger<int32_t>(std::string_view);
extern template std::optional<int64_t>
    TextFieldParser::parseNarrowInteger<int64_t>(std::string_view);
extern template std::optional<float> TextFieldParser::parseFloatingPoint<float>(
    std::string_view);
extern template std::optional<double>
    TextFieldParser::parseFloatingPoint<double>(std::string_view);
extern template Expected<int64_t>
TextFieldParser::parseDecimal<int64_t>(std::string_view, uint8_t, uint8_t);
extern template Expected<int128_t>
TextFieldParser::parseDecimal<int128_t>(std::string_view, uint8_t, uint8_t);

} // namespace facebook::velox::text
