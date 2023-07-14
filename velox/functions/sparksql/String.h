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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "folly/ssl/OpenSSLHash.h"
#pragma GCC diagnostic pop

#include "velox/expression/VectorFunction.h"
#include "velox/functions/Macros.h"
#include "velox/functions/UDFOutputString.h"
#include "velox/functions/lib/string/StringImpl.h"

namespace facebook::velox::functions::sparksql {

template <typename T>
struct AsciiFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(int32_t& result, const arg_type<Varchar>& s) {
    result = s.empty() ? 0 : s.data()[0];
    return true;
  }
};

template <typename T>
struct ChrFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(out_type<Varchar>& result, int64_t ord) {
    if (ord < 0) {
      result.resize(0);
    } else {
      result.resize(1);
      *result.data() = ord;
    }
    return true;
  }
};

template <typename T>
struct Md5Function {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TTo, typename TFrom>
  FOLLY_ALWAYS_INLINE bool call(TTo& result, const TFrom& input) {
    stringImpl::md5_radix(result, input, 16);
    return true;
  }
};

std::vector<std::shared_ptr<exec::FunctionSignature>> instrSignatures();

std::shared_ptr<exec::VectorFunction> makeInstr(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

std::vector<std::shared_ptr<exec::FunctionSignature>> lengthSignatures();

std::shared_ptr<exec::VectorFunction> makeLength(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

/// Expands each char of the digest data to two chars,
/// representing the hex value of each digest char, in order.
/// Note: digestSize must be one-half of outputSize.
void encodeDigestToBase16(uint8_t* output, int digestSize);

/// sha1 function
/// sha1(varbinary) -> string
/// Calculate SHA-1 digest and convert the result to a hex string.
/// Returns SHA-1 digest as a 40-character hex string.
template <typename T>
struct Sha1HexStringFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE
  void call(out_type<Varchar>& result, const arg_type<Varbinary>& input) {
    static const int kSha1Length = 20;
    result.resize(kSha1Length * 2);
    folly::ssl::OpenSSLHash::sha1(
        folly::MutableByteRange((uint8_t*)result.data(), kSha1Length),
        folly::ByteRange((const uint8_t*)input.data(), input.size()));
    encodeDigestToBase16((uint8_t*)result.data(), kSha1Length);
  }
};

/// sha2 function
/// sha2(varbinary, bitLength) -> string
/// Calculate SHA-2 family of functions (SHA-224, SHA-256,
/// SHA-384, and SHA-512) and convert the result to a hex string.
/// The second argument indicates the desired bit length of the result, which
/// must have a value of 224, 256, 384, 512, or 0 (which is equivalent to 256).
/// If asking for an unsupported bitLength, the return value is NULL.
/// Returns SHA-2 digest as hex string.
template <typename T>
struct Sha2HexStringFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE
  bool call(
      out_type<Varchar>& result,
      const arg_type<Varbinary>& input,
      const int32_t& bitLength) {
    const int32_t nonzeroBitLength = (bitLength == 0) ? 256 : bitLength;
    const EVP_MD* hashAlgorithm;
    switch (nonzeroBitLength) {
      case 224:
        hashAlgorithm = EVP_sha224();
        break;
      case 256:
        hashAlgorithm = EVP_sha256();
        break;
      case 384:
        hashAlgorithm = EVP_sha384();
        break;
      case 512:
        hashAlgorithm = EVP_sha512();
        break;
      default:
        // For an unsupported bitLength, the return value is NULL.
        return false;
    }
    const int32_t digestLength = nonzeroBitLength >> 3;
    result.resize(digestLength * 2);
    auto resultBuffer =
        folly::MutableByteRange((uint8_t*)result.data(), digestLength);
    auto inputBuffer =
        folly::ByteRange((const uint8_t*)input.data(), input.size());
    folly::ssl::OpenSSLHash::hash(resultBuffer, hashAlgorithm, inputBuffer);
    encodeDigestToBase16((uint8_t*)result.data(), digestLength);
    return true;
  }
};

/// contains function
/// contains(string, string) -> bool
/// Searches the second argument in the first one.
/// Returns true if it is found
template <typename T>
struct ContainsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<bool>& result,
      const arg_type<Varchar>& str,
      const arg_type<Varchar>& pattern) {
    result = std::string_view(str).find(std::string_view(pattern)) !=
        std::string_view::npos;
    return true;
  }
};

/// startsWith function
/// startsWith(string, string) -> bool
/// Returns true if the first string starts with the second string
template <typename T>
struct StartsWithFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<bool>& result,
      const arg_type<Varchar>& str,
      const arg_type<Varchar>& pattern) {
    auto str1 = std::string_view(str);
    auto str2 = std::string_view(pattern);
    // TODO: Once C++20 supported we may want to replace this with
    // string_view::starts_with

    if (str2.length() > str1.length()) {
      result = false;
    } else {
      result = str1.substr(0, str2.length()) == str2;
      ;
    }
    return true;
  }
};

/// endsWith function
/// endsWith(string, string) -> bool
/// Returns true if the first string ends with the second string
template <typename T>
struct EndsWithFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<bool>& result,
      const arg_type<Varchar>& str,
      const arg_type<Varchar>& pattern) {
    auto str1 = std::string_view(str);
    auto str2 = std::string_view(pattern);
    // TODO Once C++20 supported we may want to replace this with
    // string_view::ends_with
    if (str2.length() > str1.length()) {
      result = false;
    } else {
      result =
          str1.substr(str1.length() - str2.length(), str2.length()) == str2;
    }
    return true;
  }
};

/// ltrim(trimStr, srcStr) -> varchar
///     Remove leading specified characters from srcStr. The specified character
///     is any character contained in trimStr.
/// rtrim(trimStr, srcStr) -> varchar
///     Remove trailing specified characters from srcStr. The specified
///     character is any character contained in trimStr.
/// trim(trimStr, srcStr) -> varchar
///     Remove leading and trailing specified characters from srcStr. The
///     specified character is any character contained in trimStr.
template <typename T, bool leftTrim, bool rightTrim>
struct TrimFunctionBase {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the second argument.
  static constexpr int32_t reuse_strings_from_arg = 1;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void callAscii(
      out_type<Varchar>& result,
      const arg_type<Varchar>& trimStr,
      const arg_type<Varchar>& srcStr) {
    if (srcStr.empty()) {
      result.setEmpty();
      return;
    }
    if (trimStr.empty()) {
      result.setNoCopy(srcStr);
      return;
    }

    auto trimStrView = std::string_view(trimStr);
    size_t resultStartIndex = 0;
    if constexpr (leftTrim) {
      resultStartIndex =
          std::string_view(srcStr).find_first_not_of(trimStrView);
      if (resultStartIndex == std::string_view::npos) {
        result.setEmpty();
        return;
      }
    }

    size_t resultSize = srcStr.size() - resultStartIndex;
    if constexpr (rightTrim) {
      size_t lastIndex =
          std::string_view(srcStr.data() + resultStartIndex, resultSize)
              .find_last_not_of(trimStrView);
      if (lastIndex == std::string_view::npos) {
        result.setEmpty();
        return;
      }
      resultSize = lastIndex + 1;
    }

    result.setNoCopy(StringView(srcStr.data() + resultStartIndex, resultSize));
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& trimStr,
      const arg_type<Varchar>& srcStr) {
    if (srcStr.empty()) {
      result.setEmpty();
      return;
    }
    if (trimStr.empty()) {
      result.setNoCopy(srcStr);
      return;
    }

    auto trimStrView = std::string_view(trimStr);
    auto resultBegin = srcStr.begin();
    if constexpr (leftTrim) {
      while (resultBegin < srcStr.end()) {
        int charLen = utf8proc_char_length(resultBegin);
        auto c = std::string_view(resultBegin, charLen);
        if (trimStrView.find(c) == std::string_view::npos) {
          break;
        }
        resultBegin += charLen;
      }
    }

    auto resultEnd = srcStr.end();
    if constexpr (rightTrim) {
      auto curPos = resultEnd - 1;
      while (curPos >= resultBegin) {
        if (utf8proc_char_first_byte(curPos)) {
          auto c = std::string_view(curPos, resultEnd - curPos);
          if (trimStrView.find(c) == std::string_view::npos) {
            break;
          }
          resultEnd = curPos;
        }
        --curPos;
      }
    }

    result.setNoCopy(StringView(resultBegin, resultEnd - resultBegin));
  }
};

/// ltrim(srcStr) -> varchar
///     Removes leading 0x20(space) characters from srcStr.
/// rtrim(srcStr) -> varchar
///     Removes trailing 0x20(space) characters from srcStr.
/// trim(srcStr) -> varchar
///     Remove leading and trailing 0x20(space) characters from srcStr.
template <typename T, bool leftTrim, bool rightTrim>
struct TrimSpaceFunctionBase {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& srcStr) {
    // Because utf-8 and Ascii have the same space character code, both are
    // char=32. So trimAsciiSpace can be reused here.
    stringImpl::
        trimAsciiWhiteSpace<leftTrim, rightTrim, stringImpl::isAsciiSpace>(
            result, srcStr);
  }
};

template <typename T>
struct TrimFunction : public TrimFunctionBase<T, true, true> {};

template <typename T>
struct LTrimFunction : public TrimFunctionBase<T, true, false> {};

template <typename T>
struct RTrimFunction : public TrimFunctionBase<T, false, true> {};

template <typename T>
struct TrimSpaceFunction : public TrimSpaceFunctionBase<T, true, true> {};

template <typename T>
struct LTrimSpaceFunction : public TrimSpaceFunctionBase<T, true, false> {};

template <typename T>
struct RTrimSpaceFunction : public TrimSpaceFunctionBase<T, false, true> {};

/// substr(string, start) -> varchar
///
///     Returns the rest of string from the starting position start.
///     Positions start with 1. A negative starting position is interpreted as
///     being relative to the end of the string. When the starting position is
///     0, the meaning is to refer to the first character.

///
/// substr(string, start, length) -> varchar
///
///     Returns a substring from string of length length from the
///     starting position start. Positions start with 1. A negative starting
///     position is interpreted as being relative to the end of the string.
///     When the starting position is 0, the meaning is to refer to the
///     first character.
template <typename T>
struct SubstrFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      int32_t start,
      int32_t length = std::numeric_limits<int32_t>::max()) {
    doCall<false>(result, input, start, length);
  }

  FOLLY_ALWAYS_INLINE void callAscii(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      int32_t start,
      int32_t length = std::numeric_limits<int32_t>::max()) {
    doCall<true>(result, input, start, length);
  }

  template <bool isAscii>
  FOLLY_ALWAYS_INLINE void doCall(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      int32_t start,
      int32_t length = std::numeric_limits<int32_t>::max()) {
    if (length <= 0) {
      result.setEmpty();
      return;
    }
    // Following Spark semantics
    if (start == 0) {
      start = 1;
    }

    int32_t numCharacters = stringImpl::length<isAscii>(input);

    // negative starting position
    if (start < 0) {
      start = numCharacters + start + 1;
    }

    // Adjusting last
    int32_t last;
    bool lastOverflow = __builtin_add_overflow(start, length - 1, &last);
    if (lastOverflow || last > numCharacters) {
      last = numCharacters;
    }

    // Following Spark semantics
    if (start <= 0) {
      start = 1;
    }

    // Adjusting length
    length = last - start + 1;
    if (length <= 0) {
      result.setEmpty();
      return;
    }

    auto byteRange =
        stringCore::getByteRange<isAscii>(input.data(), start, length);

    // Generating output string
    result.setNoCopy(StringView(
        input.data() + byteRange.first, byteRange.second - byteRange.first));
  }
};

struct OverlayFunctionBase {
  template <bool isAscii, bool isVarchar>
  FOLLY_ALWAYS_INLINE void doCall(
      exec::StringWriter<false>& result,
      StringView input,
      StringView replace,
      int32_t pos,
      int32_t len) {
    // Calculate and append first part.
    auto startAndLength = substring<isAscii, isVarchar>(input, 1, pos - 1);
    append<isAscii, isVarchar>(result, input, startAndLength);

    // Append second part.
    result.append(replace);

    // Calculate and append last part.
    int32_t length = 0;
    if (len >= 0) {
      length = len;
    } else {
      if constexpr (isVarchar && !isAscii) {
        length = stringImpl::lengthUnicode(replace.data(), replace.size());
      } else {
        length = replace.size();
      }
    }
    int64_t start = (int64_t)pos + (int64_t)length;
    startAndLength = substring<isAscii, isVarchar>(input, start, INT32_MAX);
    append<isAscii, isVarchar>(result, input, startAndLength);
  }

  template <bool isAscii, bool isVarchar>
  FOLLY_ALWAYS_INLINE void append(
      exec::StringWriter<false>& result,
      StringView input,
      std::pair<int32_t, int32_t> pair) {
    if constexpr (isVarchar && !isAscii) {
      auto byteRange = stringCore::getByteRange<false>(
          input.data(), pair.first + 1, pair.second);
      result.append(StringView(
          input.data() + byteRange.first, byteRange.second - byteRange.first));
    } else {
      result.append(StringView(input.data() + pair.first, pair.second));
    }
  }

  // Information regarding the pos calculation:
  // Hive and SQL use one-based indexing for SUBSTR arguments but also accept
  // zero and negative indices for start positions. If a start index i is
  // greater than 0, it refers to element i-1 in the sequence. If a start index
  // i is less than 0, it refers to the -ith element before the end of the
  // sequence. If a start index i is 0, it refers to the first element. Return
  // pair of first indices and length.
  template <bool isAscii, bool isVarchar>
  FOLLY_ALWAYS_INLINE std::pair<int32_t, int32_t>
  substring(const StringView& input, const int64_t pos, const int64_t length) {
    int64_t len = 0;
    if constexpr (isVarchar && !isAscii) {
      len = stringImpl::lengthUnicode(input.data(), input.size());
    } else {
      len = input.size();
    }
    int64_t start = (pos > 0) ? pos - 1 : ((pos < 0) ? len + pos : 0);
    int64_t end = 0;
    if (start + length > INT32_MAX) {
      end = INT32_MAX;
    } else if (start + length < INT32_MIN) {
      end = INT32_MIN;
    } else {
      end = start + length;
    }

    if (end <= start || start >= len) {
      return std::make_pair(0, 0);
    }

    int64_t zero = 0;
    int32_t i = std::min(len, std::max(zero, start));
    int32_t j = std::min(len, std::max(zero, end));

    if (j > i) {
      return std::make_pair(i, j - i);
    } else {
      return std::make_pair(0, 0);
    }
  }
};

template <typename T>
struct OverlayVarcharFunction : public OverlayFunctionBase {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& replace,
      const int32_t pos,
      const int32_t len) {
    OverlayFunctionBase::doCall<false, true>(result, input, replace, pos, len);
  }

  FOLLY_ALWAYS_INLINE void callAscii(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& replace,
      const int32_t pos,
      const int32_t len) {
    OverlayFunctionBase::doCall<true, true>(result, input, replace, pos, len);
  }
};

template <typename T>
struct OverlayVarbinaryFunction : public OverlayFunctionBase {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varbinary>& result,
      const arg_type<Varbinary>& input,
      const arg_type<Varbinary>& replace,
      const int32_t pos,
      const int32_t len) {
    OverlayFunctionBase::doCall<false, false>(result, input, replace, pos, len);
  }
};

/// left function
/// left(string, length) -> string
/// Returns the leftmost length characters from the string
/// Return an empty string if length is less or equal than 0
template <typename T>
struct LeftFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      int32_t length) {
    doCall<false>(result, input, length);
  }

  FOLLY_ALWAYS_INLINE void callAscii(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      int32_t length) {
    doCall<true>(result, input, length);
  }

  template <bool isAscii>
  FOLLY_ALWAYS_INLINE void doCall(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      int32_t length) {
    if (length <= 0) {
      result.setEmpty();
      return;
    }

    int32_t numCharacters = stringImpl::length<isAscii>(input);

    if (length > numCharacters) {
      length = numCharacters;
    }

    int32_t start = 1;

    auto byteRange =
        stringCore::getByteRange<isAscii>(input.data(), start, length);

    // Generating output string
    result.setNoCopy(StringView(
        input.data() + byteRange.first, byteRange.second - byteRange.first));
  }
};

/// translate(string, match, replace) -> varchar
///
///   Returns a new translated string. It translates the character in ``string``
///   by a character in ``replace``. The character in ``replace`` is
///   corresponding to the character in ``match``. The translation will
///   happen when any character in ``string`` matching with a character in
///   ``match``.
template <typename T>
struct TranslateFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  std::optional<folly::F14FastMap<std::string, std::string>> unicodeDictionary_;
  std::optional<folly::F14FastMap<char, char>> asciiDictionary_;

  bool isConstantDictionary_ = false;

  folly::F14FastMap<std::string, std::string> buildUnicodeDictionary(
      const arg_type<Varchar>& match,
      const arg_type<Varchar>& replace) {
    folly::F14FastMap<std::string, std::string> dictionary;
    int i = 0;
    int j = 0;
    while (i < match.size()) {
      std::string replaceChar;
      // If match's character size is larger than replace's, the extra
      // characters in match will be removed from input string.
      if (j < replace.size()) {
        int replaceCharLength = utf8proc_char_length(replace.data() + j);
        replaceChar = std::string(replace.data() + j, replaceCharLength);
        j += replaceCharLength;
      }
      int matchCharLength = utf8proc_char_length(match.data() + i);
      std::string matchChar = std::string(match.data() + i, matchCharLength);
      // Only considers the first occurrence of a character in match.
      dictionary.emplace(matchChar, replaceChar);
      i += matchCharLength;
    }
    return dictionary;
  }

  folly::F14FastMap<char, char> buildAsciiDictionary(
      const arg_type<Varchar>& match,
      const arg_type<Varchar>& replace) {
    folly::F14FastMap<char, char> dictionary;
    int i = 0;
    for (; i < std::min(match.size(), replace.size()); i++) {
      char matchChar = *(match.data() + i);
      char replaceChar = *(replace.data() + i);
      // Only consider the first occurrence of a character in match.
      dictionary.emplace(matchChar, replaceChar);
    }
    for (; i < match.size(); i++) {
      char matchChar = *(match.data() + i);
      dictionary.emplace(matchChar, '\0');
    }
    return dictionary;
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* /*string*/,
      const arg_type<Varchar>* match,
      const arg_type<Varchar>* replace) {
    if (match != nullptr && replace != nullptr) {
      isConstantDictionary_ = true;
    }
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& match,
      const arg_type<Varchar>& replace) {
    if (!isConstantDictionary_ || !unicodeDictionary_.has_value()) {
      unicodeDictionary_ = buildUnicodeDictionary(match, replace);
    }
    // No need to do the translation.
    if (unicodeDictionary_->empty()) {
      result.append(input);
      return;
    }
    // Initial capacity is input size. Larger capacity can be reserved below.
    result.reserve(input.size());
    int i = 0;
    int k = 0;
    while (k < input.size()) {
      int inputCharLength = utf8proc_char_length(input.data() + k);
      auto inputChar = std::string(input.data() + k, inputCharLength);
      auto it = unicodeDictionary_->find(inputChar);
      if (it == unicodeDictionary_->end()) {
        // Final result size can be larger than the initial size (input size),
        // e.g., replace a ascii character with a longer utf8 character.
        result.reserve(i + inputCharLength);
        std::memcpy(result.data() + i, inputChar.data(), inputCharLength);
        i += inputCharLength;
      } else {
        result.reserve(i + it->second.size());
        std::memcpy(result.data() + i, it->second.data(), it->second.size());
        i += it->second.size();
      }
      k += inputCharLength;
    }
    result.resize(i);
  }

  FOLLY_ALWAYS_INLINE void callAscii(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& match,
      const arg_type<Varchar>& replace) {
    if (!isConstantDictionary_ || !asciiDictionary_.has_value()) {
      asciiDictionary_ = buildAsciiDictionary(match, replace);
    }
    // No need to do the translation.
    if (asciiDictionary_->empty()) {
      result.append(input);
      return;
    }
    // Result size cannot be larger than input size for all ascii input.
    result.reserve(input.size());
    int i = 0;
    for (int k = 0; k < input.size(); k++) {
      auto inputChar = *(input.data() + k);
      auto it = asciiDictionary_->find(inputChar);
      if (it == asciiDictionary_->end()) {
        std::memcpy(result.data() + i, &inputChar, 1);
        ++i;
      } else {
        if (it->second != '\0') {
          std::memcpy(result.data() + i, &(it->second), 1);
          ++i;
        }
      }
    }
    result.resize(i);
  }
};

} // namespace facebook::velox::functions::sparksql
