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

#include "velox/external/utf8proc/utf8proc.h"
#include "velox/functions/lib/string/StringImpl.h"

namespace facebook::velox::parquet {

std::string UnicodeUtil::truncateStringMin(
    const char* input,
    int32_t inputLength,
    int32_t numCodePoints) {
  auto length = functions::stringImpl::cappedByteLength<false>(
      StringView(input, inputLength), numCodePoints);
  return std::string(input, length);
}

std::string UnicodeUtil::truncateStringMax(
    const char* input,
    int32_t inputLength,
    int32_t numCodePoints) {
  auto truncatedLength = functions::stringImpl::cappedByteLength<false>(
      StringView(input, inputLength), numCodePoints);

  if (truncatedLength == inputLength) {
    return std::string(input, inputLength);
  }

  // Try to increment the last code point.
  for (auto i = numCodePoints - 1; i >= 0; --i) {
    const char* current = input;
    int32_t currentCodePoint = 0;

    // Find the i-th code point position.
    while (current < input + truncatedLength && currentCodePoint < i) {
      int32_t charLength;
      utf8proc_codepoint(current, input + truncatedLength, charLength);
      current += charLength;
      currentCodePoint++;
    }

    if (current >= input + truncatedLength)
      continue;

    int32_t charLength;
    auto codePoint =
        utf8proc_codepoint(current, input + truncatedLength, charLength);
    auto nextCodePoint = codePoint + 1;

    // Check if the incremented code point is valid.
    if (nextCodePoint != 0 && utf8proc_codepoint_valid(nextCodePoint)) {
      std::string result;
      result.reserve(truncatedLength + 4);
      result.assign(input, current - input);
      char buffer[4];
      auto bytesWritten = utf8proc_encode_char(
          nextCodePoint, reinterpret_cast<utf8proc_uint8_t*>(buffer));
      result.append(buffer, bytesWritten);
      return result;
    }
  }

  return std::string(input, truncatedLength);
}

} // namespace facebook::velox::parquet
