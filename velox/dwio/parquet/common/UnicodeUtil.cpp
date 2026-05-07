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

#include "velox/dwio/parquet/common/UnicodeUtil.h"

#include <vector>
#include "velox/external/utf8proc/utf8proc.h"
#include "velox/functions/lib/string/StringImpl.h"

namespace facebook::velox::parquet {

int32_t UnicodeUtil::truncateStringMinLength(
    const char* input,
    int32_t inputLength,
    int32_t numCodePoints) {
  return functions::stringImpl::cappedByteLength<false>(
      StringView(input, inputLength), numCodePoints);
}

std::variant<std::string_view, std::string> UnicodeUtil::truncateStringUpper(
    const char* input,
    int32_t inputLength,
    int32_t numCodePoints) {
  // Maximum number of bytes a UTF-8 encoded code point can occupy.
  constexpr int32_t kMaxUtf8BytesPerCodePoint = 4;

  auto truncatedLength = functions::stringImpl::cappedByteLength<false>(
      StringView(input, inputLength), numCodePoints);

  if (truncatedLength == inputLength) {
    return std::string_view{input, static_cast<size_t>(inputLength)};
  }

  // Collect the byte offset of each code point to avoid O(n^2) rescanning.
  std::vector<size_t> codePointOffsets;
  codePointOffsets.reserve(static_cast<size_t>(numCodePoints));
  const char* current = input;
  const char* truncatedEnd = input + truncatedLength;

  while (current < truncatedEnd) {
    codePointOffsets.push_back(static_cast<size_t>(current - input));
    int32_t codePointSize = 1;
    auto cp = utf8proc_codepoint(current, truncatedEnd, codePointSize);
    // If invalid UTF-8, advance by 1 byte.
    current += (cp == -1) ? 1 : codePointSize;
  }

  // Try incrementing from the last code point backwards.
  for (auto i = static_cast<int32_t>(codePointOffsets.size()) - 1; i >= 0;
       --i) {
    const char* pos = input + codePointOffsets[static_cast<size_t>(i)];
    int32_t codePointSize = 1;
    auto codePoint = utf8proc_codepoint(pos, truncatedEnd, codePointSize);

    // Skip invalid UTF-8 sequences.
    if (codePoint == -1) {
      continue;
    }

    auto nextCodePoint =
        functions::stringImpl::detail::incrementCodePoint(codePoint);

    if (nextCodePoint != 0) {
      std::string result;
      result.reserve(truncatedLength + kMaxUtf8BytesPerCodePoint);
      result.assign(input, codePointOffsets[static_cast<size_t>(i)]);
      char buffer[kMaxUtf8BytesPerCodePoint];
      auto bytesWritten = utf8proc_encode_char(
          nextCodePoint, reinterpret_cast<utf8proc_uint8_t*>(buffer));
      result.append(buffer, bytesWritten);
      return result;
    }
  }

  // No valid upper bound can be computed (e.g. all code points are U+10FFFF).
  // Return the truncated string as a fallback.
  return std::string_view{input, static_cast<size_t>(truncatedLength)};
}

} // namespace facebook::velox::parquet
