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

#include "velox/common/base/Exceptions.h"
#include "velox/expression/VectorFunction.h"
#include "velox/external/utf8proc/utf8proc.h"

namespace facebook::velox::parquet {

class UnicodeUtil {
 private:
  UnicodeUtil() = delete;

 public:
  static bool isCharHighSurrogate(char16_t ch) {
    return (ch & 0xFC00) == 0xD800;
  }

  // Truncates a string to the specified number of Unicode code points.
  static std::string truncateString(
      const std::string& input,
      const int32_t length) {
    VELOX_CHECK_GT(length, 0, "Truncate length should be positive");

    // Count the number of code points in the input string.
    const char* data = input.data();
    const char* end = data + input.size();
    int32_t codePointCount = 0;
    const char* current = data;

    // First pass: count code points.
    while (current < end) {
      int32_t charLength;
      utf8proc_codepoint(current, end, charLength);
      current += charLength;
      codePointCount++;
    }

    if (length >= codePointCount) {
      return input;
    }

    // Second pass: find the byte offset for the truncation point
    current = data;
    int32_t currentCodePoint = 0;
    while (current < end && currentCodePoint < length) {
      int charLength;
      utf8proc_codepoint(current, end, charLength);
      current += charLength;
      currentCodePoint++;
    }
    return input.substr(0, current - data);
  }

  static std::optional<std::string> truncateStringMin(
      const std::optional<std::string>& input,
      const int32_t length) {
    if (!input.has_value()) {
      return std::nullopt;
    }
    return truncateString(input.value(), length);
  }

  static std::optional<std::string> truncateStringMax(
      const std::optional<std::string>& input,
      const int32_t length) {
    if (!input.has_value()) {
      return std::nullopt;
    }

    const std::string& inputStr = input.value();
    const std::string truncated = truncateString(inputStr, length);
    if (truncated.length() == inputStr.length()) {
      return inputStr;
    }

    // Try to increment the last code point.
    for (int32_t i = length - 1; i >= 0; --i) {
      // Find the byte offset for the i-th code point.
      const char* data = truncated.data();
      const char* end = data + truncated.size();
      const char* current = data;
      int32_t currentCodePoint = 0;

      while (current < end && currentCodePoint < i) {
        int32_t charLength;
        utf8proc_codepoint(current, end, charLength);
        current += charLength;
        currentCodePoint++;
      }

      // Get the code point at this position.
      int charLength;
      int32_t codePoint = utf8proc_codepoint(current, end, charLength);
      int32_t nextCodePoint = codePoint + 1;

      // Check if the incremented code point is valid.
      if (nextCodePoint != 0 && utf8proc_codepoint_valid(nextCodePoint)) {
        std::string result = truncated.substr(0, current - data);
        // Append the incremented code point.
        char buffer[4]; // UTF-8 uses up to 4 bytes per code point.
        int bytesWritten = utf8proc_encode_char(
            nextCodePoint, reinterpret_cast<utf8proc_uint8_t*>(buffer));
        result.append(buffer, bytesWritten);
        return result;
      }
    }
    return std::nullopt;
  }
};

} // namespace facebook::velox::parquet
