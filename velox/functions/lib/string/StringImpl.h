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

// The cpp library for xxhash requires one of the few macros to be set in
// order to get the library to even work (there's no default mode set).
// This macro forces the hash function to be inlined and is not set by default.
// We do not want to change the external library to set this default behavior.
#define XXH_INLINE_ALL

#include <assert.h>
#include <fmt/format.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include "folly/CPortability.h"
#include "velox/common/base/Exceptions.h"
#include "velox/external/md5/md5.h"
#include "velox/external/xxhash.h"
#include "velox/functions/lib/string/StringCore.h"

namespace facebook {
namespace velox {
namespace functions {
namespace stringImpl {
using namespace stringCore;

/// Perfrom upper for a UTF8 string
template <
    StringEncodingMode stringEncoding,
    typename TOutString,
    typename TInString>
FOLLY_ALWAYS_INLINE bool upper(TOutString& output, const TInString& input) {
  bool isAsciiInput = isAscii<stringEncoding>(input.data(), input.size());
  if (isAsciiInput) {
    output.resize(input.size());
    upperAscii(output.data(), input.data(), input.size());
  } else {
    output.resize(input.size() * 4);
    auto size =
        upperUnicode(output.data(), output.size(), input.data(), input.size());
    output.resize(size);
  }
  return true;
}

/// Perfrom lower for a UTF8 string
template <
    StringEncodingMode stringEncoding,
    typename TOutString,
    typename TInString>
FOLLY_ALWAYS_INLINE bool lower(TOutString& output, const TInString& input) {
  bool isAsciiInput = isAscii<stringEncoding>(input.data(), input.size());
  if (isAsciiInput) {
    output.resize(input.size());
    lowerAscii(output.data(), input.data(), input.size());
  } else {
    output.resize(input.size() * 4);
    auto size =
        lowerUnicode(output.data(), output.size(), input.data(), input.size());
    output.resize(size);
  }
  return true;
}

/// Inplace ascii lower
template <typename T>
FOLLY_ALWAYS_INLINE bool lowerAsciiInPlace(T& str) {
  lowerAscii(str.data(), str.data(), str.size());
  return true;
}

/// Inplace ascii upper
template <typename T>
FOLLY_ALWAYS_INLINE bool upperAsciiInPlace(T& str) {
  upperAscii(str.data(), str.data(), str.size());
  return true;
}

/// Apply a set of appenders on an output string, an appender is a lambda
/// that takes an output string and append a string to it. This can be used by
/// code-gen to reduce copying in concat by evaluating nested expressions
/// in place ex concat(lower(..), upper(..))
template <typename TOutString, typename... Funcs>
void concatLazy(TOutString& output, Funcs... funcs) {
  applyAppendersRecursive(output, funcs...);
}

/// Concat function that operates on a variadic number of string arguments, the
/// number is known at compile time
template <typename TOutString, typename... Args>
void concatStatic(TOutString& output, const Args&... inputs) {
  concatLazy(output, [&](TOutString& out) {
    if (inputs.size() != 0) {
      auto writeOffset = out.size();
      out.resize(out.size() + inputs.size());
      std::memcpy(out.data() + writeOffset, inputs.data(), inputs.size());
    }
  }...);
}

/// Concat function that operates on a dynamic number of inputs packed in vector
template <typename TOutString, typename TInString>
void concatDynamic(TOutString& output, const std::vector<TInString>& inputs) {
  for (const auto& curInput : inputs) {
    if (curInput.size() == 0) {
      continue;
    }
    applyAppendersRecursive(output, [&](TOutString& out) {
      auto writeOffset = out.size();
      out.resize(out.size() + curInput.size());
      std::memcpy(out.data() + writeOffset, curInput.data(), curInput.size());
    });
  }
}

/// Return length of the input string in chars
template <StringEncodingMode stringEncoding, typename T>
FOLLY_ALWAYS_INLINE int64_t length(const T& input) {
  if constexpr (stringEncoding == StringEncodingMode::ASCII) {
    return input.size();
  } else {
    return lengthUnicode(input.data(), input.size());
  }
}

/// Write the Unicode codePoint as string to the output string. The function
/// behavior is undefined when code point it invalid. Implements the logic of
/// presto chr function.
template <typename TOutString>
FOLLY_ALWAYS_INLINE void codePointToString(
    TOutString& output,
    const int64_t codePoint) {
  auto validCodePoint =
      codePoint <= INT32_MAX && utf8proc_codepoint_valid(codePoint);
  VELOX_USER_CHECK(
      validCodePoint, "Not a valid Unicode code point: {}", codePoint);

  output.reserve(4);
  auto size = utf8proc_encode_char(
      codePoint, reinterpret_cast<unsigned char*>(output.data()));

  output.resize(size);
}

/// Returns the Unicode code point of the first char in a single char input
/// string. Implements the logic of presto codepoint function.
template <typename T>
FOLLY_ALWAYS_INLINE int32_t charToCodePoint(const T& inputString) {
  auto length = stringImpl::length<StringEncodingMode::UTF8>(inputString);
  VELOX_USER_CHECK_EQ(
      length,
      1,
      "Unexpected parameters (varchar({})) for function codepoint. Expected: codepoint(varchar(1))",
      length);

  int size;
  auto codePoint = utf8proc_codepoint(inputString.data(), size);
  return codePoint;
}

/// Returns the starting position in characters of the Nth instance of the
/// substring in string. Positions start with 1. If not found, 0 is returned. If
/// subString is empty result is 1.
template <StringEncodingMode stringEncoding, typename T>
FOLLY_ALWAYS_INLINE int64_t
stringPosition(const T& string, const T& subString, int64_t instance = 0) {
  if (subString.size() == 0) {
    return 1;
  }

  VELOX_USER_CHECK_GT(instance, 0, "'instance' must be a positive number");

  auto byteIndex = findNthInstanceByteIndex(
      std::string_view(string.data(), string.size()),
      std::string_view(subString.data(), subString.size()),
      instance);

  if (byteIndex == -1) {
    return 0;
  }

  // Return the number of characters from the beginning of the string to
  // byteIndex.
  return length<stringEncoding>(std::string_view(string.data(), byteIndex)) + 1;
}

/// Replace replaced with replacement in inputString and write results to
/// outputString.
template <typename TOutString, typename TInString>
FOLLY_ALWAYS_INLINE void replace(
    TOutString& outputString,
    const TInString& inputString,
    const TInString& replaced,
    const TInString& replacement) {
  if (replaced.size() == 0) {
    // Add replacement before and after each character.
    outputString.reserve(
        inputString.size() + replacement.size() +
        inputString.size() * replacement.size());
  } else {
    outputString.reserve(
        inputString.size() + replacement.size() +
        (inputString.size() / replaced.size()) * replacement.size());
  }

  auto outputSize = stringCore::replace(
      outputString.data(),
      std::string_view(inputString.data(), inputString.size()),
      std::string_view(replaced.data(), replaced.size()),
      std::string_view(replacement.data(), replacement.size()),
      false);

  outputString.resize(outputSize);
}

/// Replace replaced with replacement in place in string.
template <typename TInOutString, typename TInString>
FOLLY_ALWAYS_INLINE void replaceInPlace(
    TInOutString& string,
    const TInString& replaced,
    const TInString& replacement) {
  assert(replacement.size() <= replaced.size() && "invlaid inplace replace");

  auto outputSize = stringCore::replace(
      string.data(),
      std::string_view(string.data(), string.size()),
      std::string_view(replaced.data(), replaced.size()),
      std::string_view(replacement.data(), replacement.size()),
      true);

  string.resize(outputSize);
}

/// Extract the hash for a given string
/// Following the implementation in HIVE UDF
/// fbcode/fbjava/hive-udfs/core-udfs/src/main/java/com/facebook/hive/udf/UDFXxhash64.java
template <typename TInString>
FOLLY_ALWAYS_INLINE bool
xxhash64int(int64_t& result, const TInString& input, const int64_t seed = 0) {
  // Following the implementation in Hive
  // They use utf8Slice constructor which is not necessary for correctness
  result = XXH64(input.data(), input.size(), seed);
  return true;
}

/// Extract the hash for a given string as string
/// Following the implementation in Presto
/// presto/presto-main/src/main/java/com/facebook/presto/operator/scalar/VarbinaryFunctions.java
template <typename TOutString, typename TInString>
FOLLY_ALWAYS_INLINE bool xxhash64(TOutString& output, const TInString& input) {
  // Following the implementation in Presto (seed is set to 0)
  int64_t hash;
  xxhash64int(hash, input, 0);
  static const auto kLen = sizeof(int64_t);

  // Resizing output and copy
  output.resize(kLen);
  std::memcpy(output.data(), &hash, kLen);
  return true;
}

/// Compute the MD5 Hash.
template <typename TOutString, typename TInString>
FOLLY_ALWAYS_INLINE bool md5(TOutString& output, const TInString& input) {
  static const auto kByteLength = 16;
  output.resize(kByteLength);
  crypto::MD5Context md5Context;
  md5Context.Add((const uint8_t*)input.data(), input.size());
  md5Context.Finish((uint8_t*)output.data());
  return true;
}

/// Compute the MD5 Hash.
template <typename TOutString, typename TInString>
FOLLY_ALWAYS_INLINE bool md5_radix(
    TOutString& output,
    const TInString& input,
    const int32_t radix = 16) {
  static const auto kMaxTextLength = 64;

  crypto::MD5Context md5Context;
  md5Context.Add((const uint8_t*)input.data(), input.size());
  output.reserve(kMaxTextLength);
  int size = 0;
  switch (radix) {
    case 16:
      size = md5Context.FinishHex((char*)output.data());
      break;
    case 10:
      size = md5Context.FinishDec((char*)output.data());
      break;
    default:
      VELOX_USER_FAIL(
          "Not a valid radix for md5: {}. Supported values are 10 or 16",
          radix);
  }

  output.resize(size);
  return true;
}
} // namespace stringImpl
} // namespace functions
} // namespace velox
} // namespace facebook
