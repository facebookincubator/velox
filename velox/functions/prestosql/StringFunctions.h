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

#include "velox/functions/Udf.h"
#include "velox/functions/lib/string/StringCore.h"
#include "velox/functions/lib/string/StringImpl.h"

namespace facebook::velox::functions {

/// chr(n) → varchar
/// Returns the Unicode code point n as a single character string.
template <typename T>
struct ChrFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const int64_t& codePoint) {
    stringImpl::codePointToString(result, codePoint);
  }
};

/// codepoint(string) → integer
/// Returns the Unicode code point of the only character of string.
template <typename T>
struct CodePointFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int32_t& result,
      const arg_type<Varchar>& inputChar) {
    result = stringImpl::charToCodePoint(inputChar);
  }
};

/// substr(string, start) -> varchar
///
///     Returns the rest of string from the starting position start.
///     Positions start with 1. A negative starting position is interpreted as
///     being relative to the end of the string.
///
/// substr(string, start, length) -> varchar
///
///     Returns a substring from string of length length from the
///     starting position start. Positions start with 1. A negative starting
///     position is interpreted as being relative to the end of the string.
template <typename T>
struct SubstrFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  template <typename I>
  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      I start,
      I length = std::numeric_limits<I>::max()) {
    doCall<false>(result, input, start, length);
  }

  template <typename I>
  FOLLY_ALWAYS_INLINE void callAscii(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      I start,
      I length = std::numeric_limits<I>::max()) {
    doCall<true>(result, input, start, length);
  }

  template <bool isAscii, typename I>
  FOLLY_ALWAYS_INLINE void doCall(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      I start,
      I length = std::numeric_limits<I>::max()) {
    // Following Presto semantics
    if (start == 0) {
      result.setEmpty();
      return;
    }

    I numCharacters = stringImpl::length<isAscii>(input);

    // Adjusting start
    if (start < 0) {
      start = numCharacters + start + 1;
    }

    // Following Presto semantics
    if (start <= 0 || start > numCharacters || length <= 0) {
      result.setEmpty();
      return;
    }

    // Adjusting length
    if (length == std::numeric_limits<I>::max() ||
        length + start - 1 > numCharacters) {
      // set length to the max valid length
      length = numCharacters - start + 1;
    }

    auto byteRange =
        stringCore::getByteRange<isAscii>(input.data(), start, length);

    // Generating output string
    result.setNoCopy(StringView(
        input.data() + byteRange.first, byteRange.second - byteRange.first));
  }
};

/// Trim functions
/// ltrim(string) -> varchar
///    Removes leading whitespaces from the string.
/// rtrim(string) -> varchar
///    Removes trailing whitespaces from the string.
/// trim(string) -> varchar
///    Removes leading and trailing whitespaces from the string.
template <typename T, bool leftTrim, bool rightTrim>
struct TrimFunctionBase {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input) {
    stringImpl::trimUnicodeWhiteSpace<leftTrim, rightTrim>(result, input);
  }

  FOLLY_ALWAYS_INLINE void callAscii(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input) {
    stringImpl::trimAsciiWhiteSpace<leftTrim, rightTrim>(result, input);
  }
};

template <typename T>
struct TrimFunction : public TrimFunctionBase<T, true, true> {};

template <typename T>
struct LTrimFunction : public TrimFunctionBase<T, true, false> {};

template <typename T>
struct RTrimFunction : public TrimFunctionBase<T, false, true> {};

/// Returns number of characters in the specified string.
template <typename T>
struct LengthFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(int64_t& result, const StringView& input) {
    result = stringImpl::length<false>(input);
  }

  FOLLY_ALWAYS_INLINE void callAscii(int64_t& result, const StringView& input) {
    result = stringImpl::length<true>(input);
  }
};

/// Pad functions
/// lpad(string, size, padString) → varchar
///     Left pads string to size characters with padString.  If size is
///     less than the length of string, the result is truncated to size
///     characters.  size must not be negative and padString must be non-empty.
/// rpad(string, size, padString) → varchar
///     Right pads string to size characters with padString.  If size is
///     less than the length of string, the result is truncated to size
///     characters.  size must not be negative and padString must be non-empty.
template <typename T, bool lpad>
struct PadFunctionBase {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& string,
      const arg_type<int64_t>& size,
      const arg_type<Varchar>& padString) {
    stringImpl::pad<lpad, false /*isAscii*/>(result, string, size, padString);
  }

  FOLLY_ALWAYS_INLINE void callAscii(
      out_type<Varchar>& result,
      const arg_type<Varchar>& string,
      const arg_type<int64_t>& size,
      const arg_type<Varchar>& padString) {
    stringImpl::pad<lpad, true /*isAscii*/>(result, string, size, padString);
  }
};

template <typename T>
struct LPadFunction : public PadFunctionBase<T, true> {};

template <typename T>
struct RPadFunction : public PadFunctionBase<T, false> {};

/// strpos and strrpos functions
/// strpos(string, substring) → bigint
///     Returns the starting position of the first instance of substring in
///     string. Positions start with 1. If not found, 0 is returned.
/// strpos(string, substring, instance) → bigint
///     Returns the position of the N-th instance of substring in string.
///     instance must be a positive number. Positions start with 1. If not
///     found, 0 is returned.
/// strrpos(string, substring) → bigint
///     Returns the starting position of the first instance of substring in
///     string counting from the end. Positions start with 1. If not found, 0 is
///     returned.
/// strrpos(string, substring, instance) → bigint
///     Returns the position of the N-th instance of substring in string
///     counting from the end. Instance must be a positive number. Positions
///     start with 1. If not found, 0 is returned.
template <typename T, bool lpos>
struct StrPosFunctionBase {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<int64_t>& result,
      const arg_type<Varchar>& string,
      const arg_type<Varchar>& subString,
      const arg_type<int64_t>& instance = 1) {
    result = stringImpl::stringPosition<false /*isAscii*/, lpos>(
        string, subString, instance);
  }

  FOLLY_ALWAYS_INLINE void callAscii(
      out_type<int64_t>& result,
      const arg_type<Varchar>& string,
      const arg_type<Varchar>& subString,
      const arg_type<int64_t>& instance = 1) {
    result = stringImpl::stringPosition<true /*isAscii*/, lpos>(
        string, subString, instance);
  }
};

template <typename T>
struct StrLPosFunction : public StrPosFunctionBase<T, true> {};

template <typename T>
struct StrRPosFunction : public StrPosFunctionBase<T, false> {};

} // namespace facebook::velox::functions
