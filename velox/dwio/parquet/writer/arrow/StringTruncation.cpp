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

#include "velox/dwio/parquet/writer/arrow/StringTruncation.h"

#include <folly/CPortability.h>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "velox/functions/lib/string/StringCore.h"
#include "velox/functions/lib/string/StringImpl.h"

namespace facebook::velox::parquet::arrow {

// Import necessary functions from stringImpl namespace
using facebook::velox::functions::stringCore::isAscii;
using facebook::velox::functions::stringImpl::cappedByteLength;

namespace {

// Increments a Unicode code point to the next valid Unicode scalar value.
// Returns 0 if overflow (input is max code point).
FOLLY_ALWAYS_INLINE int32_t incrementCodePoint(int32_t codePoint) {
  static constexpr int32_t kMaxCodePoint = 0x10FFFF;
  static constexpr int32_t kMinSurrogate = 0xD800;
  static constexpr int32_t kMaxSurrogate = 0xDFFF;
  if (codePoint == (kMinSurrogate - 1)) {
    // Skip the surrogate range.
    return kMaxSurrogate + 1;
  } else if (codePoint == kMaxCodePoint) {
    return 0;
  }
  return codePoint + 1;
}

// ASCII fast-path for roundUp.
FOLLY_ALWAYS_INLINE std::optional<std::string> roundUpAscii(
    std::string_view input,
    int32_t numCodePoints) {
  const size_t truncatedLength =
      std::min(input.size(), static_cast<size_t>(numCodePoints));

  if (truncatedLength == input.size()) {
    return std::string(input);
  }

  if (truncatedLength == 0) {
    return std::nullopt;
  }

  for (int32_t i = truncatedLength - 1; i >= 0; --i) {
    const auto byte = static_cast<unsigned char>(input[i]);
    if (byte < 0x7F) {
      std::string result(input.data(), i);
      result.push_back(static_cast<char>(byte + 1));
      return result;
    }
  }

  // All bytes are 0x7F (DEL character), no valid upper bound.
  return std::nullopt;
}

// Unicode path for roundUp.
FOLLY_ALWAYS_INLINE std::optional<std::string> roundUpUnicode(
    std::string_view input,
    int32_t numCodePoints) {
  const auto truncatedLength = cappedByteLength<false>(input, numCodePoints);

  if (truncatedLength == input.size()) {
    return std::string(input);
  }

  if (truncatedLength == 0) {
    return std::nullopt;
  }

  const char* data = input.data();
  const char* truncatedEnd = data + truncatedLength;

  // Collect the byte offset of each code point.
  std::vector<size_t> codePointOffsets;
  codePointOffsets.reserve(numCodePoints);
  const char* current = data;
  while (current < truncatedEnd) {
    codePointOffsets.push_back(current - data);
    int32_t charLength;
    utf8proc_codepoint(current, truncatedEnd, charLength);
    current += charLength;
  }

  // Try incrementing from the last code point backwards.
  for (int32_t i = codePointOffsets.size() - 1; i >= 0; --i) {
    const char* pos = data + codePointOffsets[i];
    int32_t charLength;
    const auto codePoint = utf8proc_codepoint(pos, truncatedEnd, charLength);
    const auto nextCodePoint = incrementCodePoint(codePoint);
    if (nextCodePoint != 0) {
      std::string result(data, codePointOffsets[i]);
      char buffer[4];
      const auto bytesWritten = utf8proc_encode_char(
          nextCodePoint, reinterpret_cast<utf8proc_uint8_t*>(buffer));
      result.append(buffer, bytesWritten);
      return result;
    }
  }

  // No valid upper bound can be found.
  return std::nullopt;
}

} // namespace

std::string_view truncateUtf8(std::string_view input, int32_t numCodePoints) {
  if (isAscii(input.data(), input.size())) {
    return std::string_view(
        input.data(), std::min(input.size(), (size_t)numCodePoints));
  }
  const auto truncatedLength = cappedByteLength<false>(input, numCodePoints);
  return std::string_view(input.data(), truncatedLength);
}

std::optional<std::string> roundUpUtf8(
    std::string_view input,
    int32_t numCodePoints) {
  if (isAscii(input.data(), input.size())) {
    return roundUpAscii(input, numCodePoints);
  }
  return roundUpUnicode(input, numCodePoints);
}

std::optional<std::string> roundUpBinary(
    std::string_view input,
    int32_t truncateLength) {
  if (truncateLength <= 0) {
    return std::nullopt;
  }

  const size_t length = static_cast<size_t>(truncateLength);
  if (input.size() <= length) {
    return std::string(input);
  }

  // Create a mutable copy of the truncated input.
  std::string result(input.data(), length);

  // Try incrementing bytes from the end.
  for (size_t i = length; i-- > 0;) {
    unsigned char byte = static_cast<unsigned char>(result[i]);

    if (byte != 0xFF) { // Can increment without overflow.
      result[i] = static_cast<char>(byte + 1);
      // Truncate to i + 1 bytes (remove trailing bytes after increment point).
      result.resize(i + 1);
      return result;
    }
    // If byte == 0xFF, it will overflow, continue to previous byte.
  }

  // All bytes were 0xFF and overflowed - no valid upper bound.
  return std::nullopt;
}

} // namespace facebook::velox::parquet::arrow
