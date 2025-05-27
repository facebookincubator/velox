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
#include "utf8proc/utf8proc.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions::sparksql {
namespace {
inline bool isAlphaUtf8(utf8proc_int32_t cp) {
  auto category = utf8proc_category(cp);
  return (category >= UTF8PROC_CATEGORY_LU && category <= UTF8PROC_CATEGORY_LO);
}

inline bool isSpaceUtf8(utf8proc_int32_t cp) {
  auto category = utf8proc_category(cp);
  return (category >= UTF8PROC_CATEGORY_ZS && category <= UTF8PROC_CATEGORY_ZP);
}

template <bool onlySpaceDelimiter>
inline bool isDelimiterAscii(unsigned char ch) {
  if constexpr (onlySpaceDelimiter) {
    return ch == ' ';
  } else {
    return std::isspace(ch);
  }
}

template <bool onlySpaceDelimiter>
inline bool isDelimiterUtf8(utf8proc_int32_t cp) {
  if constexpr (onlySpaceDelimiter) {
    return cp == 0x20;
  } else {
    return isSpaceUtf8(cp);
  }
}

// ASCII InitCap Implementation
template <bool onlySpaceDelimiter, typename TOutString, typename TInString>
FOLLY_ALWAYS_INLINE bool initcapAsciiImpl(
    TOutString& output,
    const TInString& input) {
  output.resize(input.size());
  const char* inputChars = input.data();
  char* outputChars = output.data();

  bool isStartOfWord = true;
  for (size_t i = 0; i < input.size(); ++i) {
    unsigned char currentChar = static_cast<unsigned char>(inputChars[i]);

    if (isDelimiterAscii<onlySpaceDelimiter>(currentChar)) {
      isStartOfWord = true;
      outputChars[i] = currentChar;
    } else if (isStartOfWord && std::isalpha(currentChar)) {
      outputChars[i] = std::toupper(currentChar);
      isStartOfWord = false;
    } else {
      outputChars[i] = std::tolower(currentChar);
    }
  }
  return true;
}

// UTF-8 InitCap Implementation
template <bool onlySpaceDelimiter, typename TOutString, typename TInString>
FOLLY_ALWAYS_INLINE bool initcapUtf8Impl(
    TOutString& output,
    const TInString& input) {
  const uint8_t* inputBytes = reinterpret_cast<const uint8_t*>(input.data());
  const uint8_t* inputEnd = inputBytes + input.size();
  output.resize(input.size() * 4); // Max size for UTF-8 characters
  uint8_t* outputBytes = reinterpret_cast<uint8_t*>(output.data());

  bool isStartOfWord = true;

  while (inputBytes < inputEnd) {
    utf8proc_int32_t originalCodepoint;
    auto numBytesRead =
        utf8proc_iterate(inputBytes, inputEnd - inputBytes, &originalCodepoint);
    if (numBytesRead < 0) {
      return false;
    }

    utf8proc_int32_t capitalizedCodepoint;
    if (isDelimiterUtf8<onlySpaceDelimiter>(originalCodepoint)) {
      capitalizedCodepoint = originalCodepoint;
      isStartOfWord = true;
    } else if (isStartOfWord && isAlphaUtf8(originalCodepoint)) {
      capitalizedCodepoint = utf8proc_toupper(originalCodepoint);
      isStartOfWord = false;
    } else {
      capitalizedCodepoint = utf8proc_tolower(originalCodepoint);
    }

    auto numBytesWritten =
        utf8proc_encode_char(capitalizedCodepoint, outputBytes);
    outputBytes += numBytesWritten;
    inputBytes += numBytesRead;
  }

  output.resize(reinterpret_cast<char*>(outputBytes) - output.data());
  return true;
}

template <
    bool onlySpaceDelimiter,
    bool isAscii,
    typename TOutString,
    typename TInString>
FOLLY_ALWAYS_INLINE bool initcap(TOutString& output, const TInString& input) {
  if constexpr (isAscii) {
    return initcapAsciiImpl<onlySpaceDelimiter>(output, input);
  } else {
    return initcapUtf8Impl<onlySpaceDelimiter>(output, input);
  }
}
} // namespace

template <typename T>
struct InitCapFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input) {
    initcap<false /* all whitespace */, false /* UTF-8 */>(result, input);
  }

  // Presto style: only ASCII-space → UTF-8 path
  FOLLY_ALWAYS_INLINE void callSpace(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input) {
    initcap<true /* only space */, false /* UTF-8 */>(result, input);
  }

  // ASCII fast-path, all whitespace
  FOLLY_ALWAYS_INLINE void callAscii(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input) {
    initcap<false /* all whitespace */, true /* ASCII */>(result, input);
  }

  // ASCII fast-path, only ASCII-space
  FOLLY_ALWAYS_INLINE void callAsciiSpace(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input) {
    initcap<true /* only space */, true /* ASCII */>(result, input);
  }
};

} // namespace facebook::velox::functions::sparksql
