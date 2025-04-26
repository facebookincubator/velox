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

#include <string>
#include "../Macros.h"
#include "../lib/string/StringImpl.h"

namespace facebook::velox::functions::sparksql {

/// Trims trailing ASCII space characters (0x20) from ``inputStr``
/// to ensure its length does not exceed the specified Unicode string length
/// ``limit`` in characters. Throws an exception if the string still exceeds
/// `limit` after trimming.
FOLLY_ALWAYS_INLINE void trimTrailingSpaces(
    exec::StringWriter& output,
    StringView inputStr,
    int32_t numChars,
    int32_t limit) {
  const auto numTailSpacesToTrim = numChars - limit;
  VELOX_USER_CHECK_GT(numTailSpacesToTrim, 0);

  auto curPos = inputStr.end() - 1;
  const auto trimTo = inputStr.end() - numTailSpacesToTrim;

  while (curPos >= trimTo && stringImpl::isAsciiSpace(*curPos)) {
    curPos--;
  }
  // Get the length of the trimmed string in characters.
  const auto trimmedSize = numChars - std::distance(curPos + 1, inputStr.end());

  VELOX_USER_CHECK_LE(
      trimmedSize,
      limit,
      "Exceeds char/varchar type length limitation: {}",
      limit);
  output.setNoCopy(
      StringView(inputStr.data(), std::distance(inputStr.begin(), curPos + 1)));
}

/// Ensures that ``input`` fit within the specified length ``limit`` in
/// characters. If the length of ``input`` exceeds the limit, trailing spaces
/// are trimmed to fit within the limit. If the length of ``input`` is less than
/// or equal to
/// ``limit``, it is returned as-is. Throws exception if the trimmed string
/// still exceeds ``limit`` or if ``limit`` is negative. This function will trim
/// at most (length of ``input`` - ``limit``) space characters (ASCII 32) from
/// the end of ``input``.
template <typename T>
struct VarcharTypeWriteSideCheckFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      int32_t limit) {
    doCall<false>(result, input, limit);
  }

  FOLLY_ALWAYS_INLINE void callAscii(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      int32_t limit) {
    doCall<true>(result, input, limit);
  }

 private:
  template <bool isAscii>
  FOLLY_ALWAYS_INLINE void doCall(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      int32_t limit) {
    VELOX_USER_CHECK_GE(limit, 0);

    auto numCharacters = stringImpl::length<isAscii>(input);
    if (numCharacters <= limit) {
      result.setNoCopy(input);
    } else {
      trimTrailingSpaces(result, input, numCharacters, limit);
    }
  }
};

} // namespace facebook::velox::functions::sparksql
