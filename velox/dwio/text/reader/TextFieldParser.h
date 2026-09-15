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
#include <string_view>
#include <type_traits>

namespace facebook::velox::text {

/// Groups the field-level parsers shared by the Hive text-file reader
/// (TextReader) and the Spark `from_csv` special form (FromCsv). Opt-in flags
/// preserve each caller's semantics:
///   - allowTrailingDecimal: accept "123.45" as 123 (Hive) vs reject (Spark).
///   - allowOneZero: accept "0"/"1" as booleans (Hive) vs reject (Spark).
class TextFieldParser {
 public:
  /// Stack buffer size used for null-terminating short numeric fields before
  /// passing to sscanf, avoiding heap allocation in the hot path.
  static constexpr size_t kStackBufSize = 64;

  /// Parses a narrow signed integer (int8, int16, int32, int64) with
  /// overflow checking against `T`'s range. See parseInt64 (private) for
  /// details on `allowTrailingDecimal`.
  template <typename T>
  static std::optional<T> parseNarrowInteger(
      std::string_view field,
      bool allowTrailingDecimal) {
    static_assert(
        std::is_signed_v<T> && std::is_integral_v<T>,
        "parseNarrowInteger requires a signed integral type.");
    auto wide = parseInt64(field, allowTrailingDecimal);
    if (!wide.has_value()) {
      return std::nullopt;
    }
    if (static_cast<int64_t>(static_cast<T>(*wide)) != *wide) {
      return std::nullopt;
    }
    return static_cast<T>(*wide);
  }

  /// Parses a boolean from `field`. Accepts case-insensitive "TRUE"/"FALSE".
  /// When `allowOneZero` is true (Hive), also accepts the single characters
  /// '1' and '0'. Returns std::nullopt for any other input, including empty
  /// input or values with surrounding whitespace.
  static std::optional<bool> parseBoolean(
      std::string_view field,
      bool allowOneZero);

 private:
  // Parses a signed 64-bit integer from `field`. Internal helper for
  // parseNarrowInteger; kept private because it has no direct external
  // callers.
  //
  // Rejects empty input, leading whitespace, and any non-digit/non-sign
  // first character. When `allowTrailingDecimal` is true (Hive default),
  // trailing characters are accepted iff they form a valid decimal
  // continuation (one optional '.' followed by digits, e.g. "123.45" → 123).
  // When false (Spark `from_csv` default), any trailing character causes
  // rejection.
  static std::optional<int64_t> parseInt64(
      std::string_view field,
      bool allowTrailingDecimal);
};

} // namespace facebook::velox::text
