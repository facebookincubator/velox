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

#include <cctype>

#include "velox/common/encode/Base64.h"
#include "velox/functions/Macros.h"

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

/// checked_unbase64(x): Spark's to_binary(x, 'base64'). Like unbase64, but
/// first checks that 'x' is a well-formed base64 string under Spark's rule
/// (RFC 4648): groups of four symbols from A-Z, a-z, 0-9, '+' and '/', a last
/// group of two to four symbols, optional '=' padding that completes the last
/// group to four, nothing after the padding, and whitespace anywhere before
/// it. The MIME decoder used by unbase64 silently skips other characters;
/// checked_unbase64 raises an error for them instead.
template <typename T>
struct CheckedUnBase64Function : public UnBase64Function<T> {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varbinary>& result, const arg_type<Varchar>& input) {
    VELOX_USER_RETURN(
        !isValidBase64(input),
        "The value '{}' of the type STRING is invalid because it violates the constraint 'BASE64'",
        std::string(input.data(), input.size()));
    return this->decodeMime(input, result);
  }

 private:
  static bool isValidBase64(const StringView& input) {
    size_t position = 0;
    size_t padSize = 0;
    for (const char c : input) {
      if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
          (c >= 'a' && c <= 'z') || c == '/' || c == '+') {
        // Padding must conclude the string.
        if (padSize != 0) {
          return false;
        }
        ++position;
      } else if (c == '=') {
        ++padSize;
        // The group before the padding needs at least two symbols, and there
        // are at most two padding characters.
        if (padSize > 2 || position % 4 < 2) {
          return false;
        }
      } else if (std::isspace(static_cast<unsigned char>(c))) {
        if (padSize != 0) {
          return false;
        }
      } else {
        return false;
      }
    }
    if (padSize > 0) {
      // With padding, the last group has exactly four symbols.
      return (position + padSize) % 4 == 0;
    }
    // Without padding, the last group has two or more symbols.
    return position % 4 != 1;
  }
};

} // namespace facebook::velox::functions::sparksql
