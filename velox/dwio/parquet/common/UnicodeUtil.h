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

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

namespace facebook::velox::parquet {

class UnicodeUtil {
 public:
  /// Returns the byte length of the minimum UTF-8 value containing at most the
  /// specified number of code points.
  ///
  /// This function computes the byte length of the input prefix containing at
  /// most numCodePoints Unicode code points. The caller can reuse the original
  /// input pointer together with the returned length to reference the minimum
  /// prefix value.
  ///
  /// @param input Pointer to the UTF-8 encoded input string
  /// @param inputLength Length of the input string in bytes
  /// @param numCodePoints Maximum number of Unicode code points to include
  /// @return Byte length of the truncated prefix. If the input has fewer than
  ///         numCodePoints code points, returns inputLength.
  static int32_t truncateStringMinLength(
      const char* input,
      int32_t inputLength,
      int32_t numCodePoints);

  /// Truncates a UTF-8 string to at most the specified number of code points
  /// and attempts to increment the last code point to produce an upper bound
  /// for range queries.
  ///
  /// This function truncates the input string to contain at most numCodePoints
  /// Unicode code points, then attempts to increment the last code point.
  /// It tries incrementing from the last code point backwards until a valid
  /// increment is found.
  ///
  /// The increment logic properly handles:
  /// - Surrogate range: U+D7FF increments to U+E000 (skipping U+D800..U+DFFF)
  /// - Maximum code point: U+10FFFF cannot be incremented
  /// - Invalid UTF-8: Treated as single-byte sequences
  ///
  /// @param input Pointer to the UTF-8 encoded input string
  /// @param inputLength Length of the input string in bytes
  /// @param numCodePoints Maximum number of Unicode code points to include
  /// @return A string_view into the original input when the entire input fits
  ///         within numCodePoints, otherwise a new string containing:
  ///         - A string with the last incrementable code point incremented
  ///         - The truncated string as fallback if no code point can be
  ///           incremented (e.g., all retained code points are U+10FFFF)
  static std::variant<std::string_view, std::string> truncateStringUpper(
      const char* input,
      int32_t inputLength,
      int32_t numCodePoints);

 private:
  UnicodeUtil() = delete;
};

} // namespace facebook::velox::parquet
