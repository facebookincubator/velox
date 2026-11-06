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

#include "velox/common/encode/Base64.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/Utf8Utils.h"

namespace facebook::velox::functions::sparksql {

/// UnBase64Function decodes a base64-encoded input string into its original
/// binary form. It uses the Base64 MIME decoding functions provided by
/// velox::encoding. Returns a Status indicating success or error during
/// decoding.
template <typename T>
struct UnBase64Function {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Decodes the base64-encoded input and stores the result in 'result'.
  // Returns a Status object indicating success or failure.
  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varbinary>& result, const arg_type<Varchar>& input) {
    return decodeMime(input, result);
  }

 protected:
  template <typename TResult>
  FOLLY_ALWAYS_INLINE static Status decodeMime(
      const StringView& input,
      TResult& result) {
    auto decodedSize =
        encoding::Base64::calculateMimeDecodedSize(input.data(), input.size());
    if (decodedSize.hasError()) {
      return decodedSize.error();
    }
    result.resize(decodedSize.value());
    return encoding::Base64::decodeMime(
        input.data(), input.size(), result.data());
  }
};

/// to_binary_base64(x): Spark's to_binary(x, 'base64'). Like unbase64, but
/// first checks that 'x' is a well-formed base64 string under Spark's rule
/// (RFC 4648): groups of four symbols from A-Z, a-z, 0-9, '+' and '/', a last
/// group of two to four symbols, optional '=' padding that completes the last
/// group to four, nothing after the padding, and whitespace anywhere before
/// it. The MIME decoder used by unbase64 silently skips other characters;
/// to_binary_base64 raises an error for them instead.
template <typename T>
struct ToBinaryBase64Function : public UnBase64Function<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varbinary>& result, const arg_type<Varchar>& input) {
    VELOX_USER_RETURN(
        !isValidBase64(input),
        "Malformed BASE64 input cannot be converted to BINARY: {}",
        std::string(input.data(), input.size()));
    return this->decodeMime(input, result);
  }

 private:
  // Spark's UnBase64.isValidBase64.
  static bool isValidBase64(const StringView& input) {
    const char* data = input.data();
    const size_t size{input.size()};
    size_t position{0};
    size_t padSize{0};
    size_t i{0};
    while (i < size) {
      const char c = data[i];
      if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
          (c >= 'a' && c <= 'z') || c == '/' || c == '+') {
        // Padding must conclude the string.
        if (padSize != 0) {
          return false;
        }
        ++position;
        ++i;
      } else if (c == '=') {
        ++padSize;
        // The group before the padding needs at least two symbols, and there
        // are at most two padding characters.
        if (padSize > 2 || position % 4 < 2) {
          return false;
        }
        ++i;
      } else {
        const int32_t length = whitespaceLength(data + i, size - i);
        if (length == 0 || padSize != 0) {
          return false;
        }
        i += length;
      }
    }
    if (padSize > 0) {
      // With padding, the last group has exactly four symbols.
      return (position + padSize) % 4 == 0;
    }
    // Without padding, the last group has two or more symbols.
    return position % 4 != 1;
  }

  // Returns the length in bytes of the character at 'input' if it is
  // whitespace under Java's Character.isWhitespace, which Spark uses, or 0
  // otherwise.
  static int32_t whitespaceLength(const char* input, int64_t size) {
    const auto c = static_cast<unsigned char>(input[0]);
    if (c < 0x80) {
      // Tab, line feed, vertical tab, form feed, carriage return, the file,
      // group, record and unit separators, and space.
      return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) ? 1 : 0;
    }
    int32_t codePoint;
    const int32_t length = tryGetUtf8CharLength(input, size, codePoint);
    if (length <= 0) {
      return 0;
    }
    // Unicode space, line and paragraph separators, except the no-break
    // spaces U+00A0, U+2007 and U+202F.
    switch (codePoint) {
      case 0x1680:
      case 0x2000:
      case 0x2001:
      case 0x2002:
      case 0x2003:
      case 0x2004:
      case 0x2005:
      case 0x2006:
      case 0x2008:
      case 0x2009:
      case 0x200A:
      case 0x2028:
      case 0x2029:
      case 0x205F:
      case 0x3000:
        return length;
      default:
        return 0;
    }
  }
};

} // namespace facebook::velox::functions::sparksql
