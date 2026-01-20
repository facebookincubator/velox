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

#include <cstring>
#include <string>
#include <string_view>
#include "folly/CPortability.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/external/utf8proc/utf8procImpl.h"

#if (ENABLE_VECTORIZATION > 0) && !defined(_DEBUG) && !defined(DEBUG)
#if defined(__clang__) && (__clang_major__ > 7)
#define IS_SANITIZER                          \
  ((__has_feature(address_sanitizer) == 1) || \
   (__has_feature(memory_sanitizer) == 1) ||  \
   (__has_feature(thread_sanitizer) == 1) ||  \
   (__has_feature(undefined_sanitizer) == 1))

#if IS_SANITIZER == 0
#define VECTORIZE_LOOP_IF_POSSIBLE _Pragma("clang loop vectorize(enable)")
#endif
#endif
#endif

#ifndef VECTORIZE_LOOP_IF_POSSIBLE
// Not supported
#define VECTORIZE_LOOP_IF_POSSIBLE
#endif

namespace facebook::velox::functions {
namespace detail {

// Helper function to check if a character is cased. Compatible with the
// 'isCased' implementation in 'ConditionalSpecialCasting.java' of JDK, which is
// used by 'toLowerCase' function in Spark SQL.
FOLLY_ALWAYS_INLINE bool isCased(utf8proc_int32_t ch) {
  auto type = utf8proc_category(ch);
  // Lowercase letter, uppercase letter or titlecase letter.
  if (type == UTF8PROC_CATEGORY_LL || type == UTF8PROC_CATEGORY_LU ||
      type == UTF8PROC_CATEGORY_LT) {
    return true;
  }
  // Modifier letters and special cases.
  if ((ch >= 0x02B0 && ch <= 0x02B8) || (ch >= 0x02C0 && ch <= 0x02C1) ||
      (ch >= 0x02E0 && ch <= 0x02E4) || ch == 0x0345 || ch == 0x037A ||
      (ch >= 0x1D2C && ch <= 0x1D61) || (ch >= 0x2160 && ch <= 0x217F) ||
      (ch >= 0x24B6 && ch <= 0x24E9)) {
    return true;
  }
  return false;
}

// Helper function to check if a character is case-ignorable according to
// Unicode specification. Case-ignorable characters can be skipped when
// determining the context for final sigma conversion.
// Reference: https://www.unicode.org/Public/UCD/latest/ucd/DerivedCoreProperties.txt
FOLLY_ALWAYS_INLINE bool isCaseIgnorable(utf8proc_int32_t ch) {
  auto cat = utf8proc_category(ch);

  // General_Category: Mn (Mark, Nonspacing), Me (Mark, Enclosing),
  // Cf (Format), Lm (Letter, Modifier), Sk (Symbol, Modifier)
  if (cat == UTF8PROC_CATEGORY_MN || cat == UTF8PROC_CATEGORY_ME ||
      cat == UTF8PROC_CATEGORY_CF || cat == UTF8PROC_CATEGORY_LM ||
      cat == UTF8PROC_CATEGORY_SK) {
    return true;
  }

  // Word_Break property: MidLetter, MidNumLet, Single_Quote
  // Reference: https://www.unicode.org/Public/UCD/latest/ucd/auxiliary/WordBreakProperty.txt
  switch (ch) {
    // MidLetter
    case 0x00B7: // · MIDDLE DOT
    case 0x0387: // · GREEK ANO TELEIA
    case 0x05F4: // ״ HEBREW PUNCTUATION GERSHAYIM
    case 0x2027: // ‧ HYPHENATION POINT
    case 0xFE13: // ︓ PRESENTATION FORM FOR VERTICAL COLON
    case 0xFE55: // ﹕ SMALL COLON
    case 0xFF1A: // ： FULLWIDTH COLON
    // MidNumLet
    case 0x002E: // . FULL STOP
    case 0x2018: // ' LEFT SINGLE QUOTATION MARK
    case 0x2019: // ' RIGHT SINGLE QUOTATION MARK
    case 0x2024: // ․ ONE DOT LEADER
    case 0xFE52: // ﹒ SMALL FULL STOP
    case 0xFF07: // ＇ FULLWIDTH APOSTROPHE
    case 0xFF0E: // ． FULLWIDTH FULL STOP
    // Single_Quote
    case 0x0027: // ' APOSTROPHE
      return true;
    default:
      return false;
  }
}

// Scan backward from the given position to check if there is a cased character.
// Skips case-ignorable characters during the scan.
// Returns true if a cased character is found before encountering a non-cased,
// non-case-ignorable character.
FOLLY_ALWAYS_INLINE bool hasCasedBefore(
    const char* input,
    size_t sigmaStartPos) {
  if (sigmaStartPos == 0) {
    return false;
  }

  size_t pos = sigmaStartPos;
  while (pos > 0) {
    // Move backward to find the start of the previous UTF-8 character.
    // UTF-8 continuation bytes have the form 10xxxxxx (0x80-0xBF).
    size_t prevPos = pos - 1;
    while (prevPos > 0 &&
           (static_cast<unsigned char>(input[prevPos]) & 0xC0) == 0x80) {
      --prevPos;
    }

    int size;
    utf8proc_int32_t cp =
        utf8proc_codepoint(&input[prevPos], &input[pos], size);

    if (cp == -1) {
      // Invalid UTF-8, treat as non-cased, non-case-ignorable.
      return false;
    }

    if (isCased(cp)) {
      return true; // Found a cased character.
    }
    if (!isCaseIgnorable(cp)) {
      return false; // Non-case-ignorable stops the search.
    }

    pos = prevPos; // Continue scanning backward.
  }

  return false;
}

// Scan forward from the given position to check if there is a cased character.
// Skips case-ignorable characters during the scan.
// Returns true if a cased character is found before encountering a non-cased,
// non-case-ignorable character.
FOLLY_ALWAYS_INLINE bool hasCasedAfter(
    const char* input,
    size_t inputLength,
    size_t afterSigmaPos) {
  size_t pos = afterSigmaPos;

  while (pos < inputLength) {
    int size;
    utf8proc_int32_t cp =
        utf8proc_codepoint(&input[pos], input + inputLength, size);

    if (cp == -1) {
      // Invalid UTF-8, treat as non-cased, non-case-ignorable.
      return false;
    }

    if (isCased(cp)) {
      return true; // Found a cased character.
    }
    if (!isCaseIgnorable(cp)) {
      return false; // Non-case-ignorable stops the search.
    }

    pos += size; // Continue scanning forward.
  }

  return false;
}

} // namespace detail

namespace stringCore {

/// Check if a given string is ascii
static bool isAscii(const char* str, size_t length);

FOLLY_ALWAYS_INLINE bool isAscii(const char* str, size_t length) {
  const auto mask = xsimd::broadcast<uint8_t>(0x80);
  size_t i = 0;
  for (; i + mask.size <= length; i += mask.size) {
    auto batch =
        xsimd::load_unaligned(reinterpret_cast<const uint8_t*>(str) + i);
#if XSIMD_WITH_AVX
    // 1 instruction instead of 2 on AVX.
    if (!_mm256_testz_si256(batch, mask)) {
#else
    if (xsimd::any(batch >= mask)) {
#endif
      return false;
    }
  }
  for (; i < length; ++i) {
    if (str[i] & 0x80) {
      return false;
    }
  }
  return true;
}

/// Perform reverse for ascii string input
FOLLY_ALWAYS_INLINE static void
reverseAscii(char* output, const char* input, size_t length) {
  auto j = length - 1;
  VECTORIZE_LOOP_IF_POSSIBLE for (size_t i = 0; i < length; ++i, --j) {
    output[i] = input[j];
  }
}

/// Perform reverse for utf8 string input
FOLLY_ALWAYS_INLINE static void
reverseUnicode(char* output, const char* input, int64_t length) {
  int64_t inputIdx = 0;
  int64_t outputIdx = static_cast<int64_t>(length);
  while (inputIdx < length) {
    int size = 1;
    const auto valid =
        utf8proc_codepoint(&input[inputIdx], input + length, size);

    // if invalid utf8 gets byte sequence with nextCodePoint==-1 and size==1,
    // continue reverse invalid sequence byte by byte.
    if (valid == -1) {
      size = 1;
    }

    VELOX_USER_CHECK_GE(outputIdx, size, "access out of bound");
    outputIdx -= size;

    VELOX_USER_CHECK_LT(outputIdx, length, "access out of bound");
    std::memcpy(&output[outputIdx], &input[inputIdx], size);
    inputIdx += size;
  }
}

/// Perform upper for ascii string input
FOLLY_ALWAYS_INLINE static void
upperAscii(char* output, const char* input, size_t length) {
  VECTORIZE_LOOP_IF_POSSIBLE for (size_t i = 0; i < length; i++) {
    if (input[i] >= 'a' && input[i] <= 'z') {
      output[i] = input[i] - 32;
    } else {
      output[i] = input[i];
    }
  }
}

/// Perform lower for ascii string input
FOLLY_ALWAYS_INLINE static void
lowerAscii(char* output, const char* input, size_t length) {
  VECTORIZE_LOOP_IF_POSSIBLE for (size_t i = 0; i < length; i++) {
    if (input[i] >= 'A' && input[i] <= 'Z') {
      output[i] = input[i] + 32;
    } else {
      output[i] = input[i];
    }
  }
}

/// Perform upper for utf8 string input, output should be pre-allocated and
/// large enough for the results. outputLength refers to the number of bytes
/// available in the output buffer, and inputLength is the number of bytes in
/// the input string
FOLLY_ALWAYS_INLINE size_t upperUnicode(
    char* output,
    size_t outputLength,
    const char* input,
    size_t inputLength) {
  size_t inputIdx = 0;
  size_t outputIdx = 0;

  while (inputIdx < inputLength) {
    utf8proc_int32_t nextCodePoint;
    int size;
    nextCodePoint =
        utf8proc_codepoint(&input[inputIdx], input + inputLength, size);
    if (UNLIKELY(nextCodePoint == -1)) {
      // invalid input string, copy the remaining of the input string as is to
      // the output.
      std::memcpy(&output[outputIdx], &input[inputIdx], inputLength - inputIdx);
      outputIdx += inputLength - inputIdx;
      return outputIdx;
    }
    inputIdx += size;

    auto upperCodePoint = utf8proc_toupper(nextCodePoint);

    assert(
        (outputIdx + utf8proc_codepoint_length(upperCodePoint)) <
            outputLength &&
        "access out of bound");

    auto newSize = utf8proc_encode_char(
        upperCodePoint, reinterpret_cast<unsigned char*>(&output[outputIdx]));
    outputIdx += newSize;
  }
  return outputIdx;
}

/// Perform lower for utf8 string input, output should be pre-allocated and
/// large enough for the results outputLength refers to the number of bytes
/// available in the output buffer, and inputLength is the number of bytes in
/// the input string.
/// @tparam turkishCasing If true, Spark's specific behavior on Turkish casing
/// is considered. For 'İ' Spark's lower case is 'i̇' and Presto's is 'i'.
/// @tparam greekFinalSigma If true, Greek final sigma rule is applied. For the
/// uppercase letter Σ, if it appears at the end of a word, it becomes ς. In all
/// other positions, it becomes σ. If false, it is always converted to σ.
template <bool turkishCasing, bool greekFinalSigma>
FOLLY_ALWAYS_INLINE size_t lowerUnicode(
    char* output,
    size_t outputLength,
    const char* input,
    size_t inputLength) {
  size_t inputIdx = 0;
  size_t outputIdx = 0;

  while (inputIdx < inputLength) {
    utf8proc_int32_t nextCodePoint;
    int size;
    nextCodePoint =
        utf8proc_codepoint(&input[inputIdx], input + inputLength, size);
    if (UNLIKELY(nextCodePoint == -1)) {
      // invalid input string, copy the remaining of the input string as is to
      // the output.
      std::memcpy(&output[outputIdx], &input[inputIdx], inputLength - inputIdx);
      outputIdx += inputLength - inputIdx;
      return outputIdx;
    }

    inputIdx += size;

    if constexpr (turkishCasing) {
      // Handle Turkish-specific case for İ (U+0130).
      if (UNLIKELY(nextCodePoint == 0x0130)) {
        // Map to i̇ (U+0069 U+0307).
        output[outputIdx++] = 0x69;
        output[outputIdx++] = 0xCC;
        output[outputIdx++] = 0x87;
        continue;
      }
    }

    if constexpr (greekFinalSigma) {
      // Handle Greek final sigma for Σ (U+03A3).
      // According to Unicode specification, Σ should be converted to ς (final
      // sigma) only when:
      // 1. There is a cased character before Σ (possibly separated by
      //    case-ignorable characters)
      // 2. There is no cased character after Σ (possibly separated by
      //    case-ignorable characters)
      if (nextCodePoint == 0x03A3) {
        // inputIdx now points to the byte after Σ.
        // We need to check:
        // - hasCasedBefore: scan backward from the start of Σ
        // - hasCasedAfter: scan forward from after Σ
        size_t sigmaStartPos = inputIdx - size;
        bool isFinal = detail::hasCasedBefore(input, sigmaStartPos) &&
            !detail::hasCasedAfter(input, inputLength, inputIdx);

        // Convert to ς (U+03C2) if final, otherwise σ (U+03C3).
        utf8proc_int32_t lowerSigma = isFinal ? 0x03C2 : 0x03C3;
        auto newSize = utf8proc_encode_char(
            lowerSigma, reinterpret_cast<unsigned char*>(&output[outputIdx]));
        outputIdx += newSize;
        continue;
      }
    }

    auto lowerCodePoint = utf8proc_tolower(nextCodePoint);

    assert(
        (outputIdx + utf8proc_codepoint_length(lowerCodePoint)) <
            outputLength &&
        "access out of bound");

    auto newSize = utf8proc_encode_char(
        lowerCodePoint, reinterpret_cast<unsigned char*>(&output[outputIdx]));
    outputIdx += newSize;
  }
  return outputIdx;
}

/// Apply a sequence of appenders to the output string sequentially.
/// @param output the output string that appenders are applied to
/// @param appenderFunc a function that appends some string to an input string
/// of type TOutStr
template <typename TOutStr, typename Func>
static void applyAppendersRecursive(TOutStr& output, Func appenderFunc) {
  appenderFunc(output);
}

template <typename TOutStr, typename Func, typename... Funcs>
static void
applyAppendersRecursive(TOutStr& output, Func appenderFunc, Funcs... funcs) {
  appenderFunc(output);
  applyAppendersRecursive(output, funcs...);
}

/**
 * Return the length in chars of a utf8 string stored in the input buffer
 * @param inputBuffer input buffer that hold the string
 * @param bufferLength size of input buffer
 * @return the number of characters represented by the input utf8 string
 */
FOLLY_ALWAYS_INLINE int64_t
lengthUnicode(const char* inputBuffer, size_t bufferLength) {
  // First address after the last byte in the buffer
  auto buffEndAddress = inputBuffer + bufferLength;
  auto currentChar = inputBuffer;
  int64_t size = 0;
  while (currentChar < buffEndAddress) {
    // This function detects bytes that come after the first byte in a
    // multi-byte UTF-8 character (provided that the string is valid UTF-8). We
    // increment size only for the first byte so that we treat all bytes as part
    // of a single character.
    if (!utf_cont(*currentChar)) {
      size++;
    }

    currentChar++;
  }
  return size;
}

/**
 * Return an capped length(controlled by maxChars) of a unicode string. The
 * returned length is not greater than maxChars.
 *
 * This method is used to tell whether a string is longer or the same length of
 * another string, in these scenarios we don't need accurate length, by
 * providing maxChars we can get better performance by avoid calculating whole
 * length of a string which might be very long.
 *
 * @param input input buffer that hold the string
 * @param size size of input buffer
 * @param maxChars stop counting characters if the string is longer
 * than this value
 * @return the number of characters represented by the input utf8 string
 */
FOLLY_ALWAYS_INLINE int64_t
cappedLengthUnicode(const char* input, size_t size, int64_t maxChars) {
  // First address after the last byte in the input
  auto end = input + size;
  auto currentChar = input;
  int64_t numChars = 0;

  // Use maxChars to early stop to avoid calculating the whole
  // length of long string.
  while (currentChar < end && numChars < maxChars) {
    auto charSize = utf8proc_char_length(currentChar);
    // Skip bad byte if we get utf length < 0.
    currentChar += UNLIKELY(charSize < 0) ? 1 : charSize;
    numChars++;
  }

  return numChars;
}

///
/// Return an capped length in bytes(controlled by maxChars) of a unicode
/// string. The returned length may be greater than maxCharacters if there are
/// multi-byte characters present in the input string.
///
/// This method is used to help with indexing unicode strings by byte position.
/// It is used to find the byte position of the Nth character in a string.
///
/// @param input input buffer that hold the string
/// @param size size of input buffer
/// @param maxChars stop counting characters if the string is longer
/// than this value
/// @return the number of bytes represented by the input utf8 string up to
/// maxChars
///
FOLLY_ALWAYS_INLINE int64_t
cappedByteLengthUnicode(const char* input, int64_t size, int64_t maxChars) {
  int64_t utf8Position = 0;
  int64_t numCharacters = 0;
  while (utf8Position < size && numCharacters < maxChars) {
    auto charSize = utf8proc_char_length(input + utf8Position);
    utf8Position += UNLIKELY(charSize < 0) ? 1 : charSize;
    numCharacters++;
  }
  return utf8Position;
}

/// Returns the start byte index of the Nth instance of subString in
/// string. Search starts from startPosition. Positions start with 0. If not
/// found, -1 is returned. To facilitate finding overlapping strings, the
/// nextStartPosition is incremented by 1
static inline int64_t findNthInstanceByteIndexFromStart(
    const std::string_view& string,
    const std::string_view subString,
    const size_t instance = 1,
    const size_t startPosition = 0) {
  assert(instance > 0);

  if (startPosition >= string.size()) {
    return -1;
  }

  auto byteIndex = string.find(subString, startPosition);
  // Not found
  if (byteIndex == std::string_view::npos) {
    return -1;
  }

  // Search done
  if (instance == 1) {
    return byteIndex;
  }

  // Find next occurrence
  return findNthInstanceByteIndexFromStart(
      string, subString, instance - 1, byteIndex + 1);
}

/// Returns the start byte index of the Nth instance of subString in
/// string from the end. Search starts from endPosition. Positions start with 0.
/// If not found, -1 is returned. To facilitate finding overlapping strings, the
/// nextStartPosition is incremented by 1
inline int64_t findNthInstanceByteIndexFromEnd(
    const std::string_view string,
    const std::string_view subString,
    const size_t instance = 1) {
  assert(instance > 0);

  if (subString.empty()) {
    return 0;
  }

  size_t foundCnt = 0;
  size_t index = string.size();
  do {
    if (index == 0) {
      return -1;
    }

    index = string.rfind(subString, index - 1);
    if (index == std::string_view::npos) {
      return -1;
    }
    ++foundCnt;
  } while (foundCnt < instance);
  return index;
}

/// Replace replaced with replacement in inputString and write results in
/// outputString. If inPlace=true inputString and outputString are assumed to
/// be the same. When replaced is empty and ignoreEmptyReplaced is false,
/// replacement is added before and after each charecter. When replaced is
/// empty and ignoreEmptyReplaced is true, the result is the inputString value.
/// When inputString and replaced strings are empty, result is the
/// replacement string if ignoreEmptyReplaced is false, otherwise the result is
/// empty.
///
/// Note: if replaceFirst=true, the only the first found occurence of replaced
/// is replaced. If replaced is empty, then replacement is added before the
/// inputString.
///
/// replace("", "", "x") = "" -- when ignoreEmptyReplaced is true
/// replace("", "", "x") = "x" -- when ignoreEmptyReplaced is false
/// replace("aa", "", "x") = "xaxax" -- when ignoreEmptyReplaced is false
/// replace("aa", "", "x") = "aa" -- when ignoreEmptyReplaced is true
template <bool ignoreEmptyReplaced = false>
inline static size_t replace(
    char* outputString,
    const std::string_view& inputString,
    const std::string_view& replaced,
    const std::string_view& replacement,
    bool inPlace = false,
    bool replaceFirst = false) {
  if (inputString.empty()) {
    if (!ignoreEmptyReplaced && replaced.empty() && !replacement.empty()) {
      std::memcpy(outputString, replacement.data(), replacement.size());
      return replacement.size();
    }
    return 0;
  }

  if constexpr (ignoreEmptyReplaced) {
    if (replaced.empty()) {
      if (!inPlace) {
        std::memcpy(outputString, inputString.data(), inputString.size());
      }
      return inputString.size();
    }
  }

  size_t readPosition = 0;
  size_t writePosition = 0;
  // Copy needed in out of place replace, and when replaced and replacement are
  // of different sizes.
  bool doCopyUnreplaced = !inPlace || (replaced.size() != replacement.size());

  auto findNextReplaced = [&]() {
    return findNthInstanceByteIndexFromStart(
        inputString, replaced, 1, readPosition);
  };

  auto writeUnchanged = [&](ssize_t size) {
    assert(size >= 0 && "Probable math error?");
    if (size <= 0) {
      return;
    }

    if (inPlace) {
      if (doCopyUnreplaced) {
        // memcpy does not allow overllapping
        std::memmove(
            &outputString[writePosition],
            &inputString.data()[readPosition],
            size);
      }
    } else {
      std::memcpy(
          &outputString[writePosition],
          &inputString.data()[readPosition],
          size);
    }
    writePosition += size;
    readPosition += size;
  };

  auto writeReplacement = [&]() {
    if (replacement.size() > 0) {
      std::memcpy(
          &outputString[writePosition], replacement.data(), replacement.size());
      writePosition += replacement.size();
    }
    readPosition += replaced.size();
  };

  // Special case when size of replaced is 0
  if (replaced.empty()) {
    if (replacement.empty()) {
      if (!inPlace) {
        std::memcpy(outputString, inputString.data(), inputString.size());
      }
      return inputString.size();
    }

    // Can never be in place since replacement.size()>replaced.size()
    assert(!inPlace && "wrong inplace replace usage");

    if (replaceFirst) {
      // writes replacement to the beginning of outputString
      std::memcpy(&outputString[0], replacement.data(), replacement.size());
      // writes the original string
      std::memcpy(
          &outputString[replacement.size()],
          inputString.data(),
          inputString.size());
      return replacement.size() + inputString.size();
    }

    // add replacement before and after each char in inputString
    for (size_t i = 0; i < inputString.size(); i++) {
      writeReplacement();

      outputString[writePosition] = inputString[i];
      writePosition++;
    }

    writeReplacement();
    return writePosition;
  }

  while (readPosition < inputString.size()) {
    // Find next token to replace
    auto position = findNextReplaced();

    if (position == -1) {
      break;
    }
    assert(position >= 0 && "invalid position found");
    auto unchangedSize = position - readPosition;
    writeUnchanged(unchangedSize);
    writeReplacement();
    // If replaceFirst is true, we only replace the first occurence
    // of the found replaced
    if (replaceFirst) {
      break;
    }
  }

  auto unchangedSize = inputString.size() - readPosition;
  writeUnchanged(unchangedSize);

  return writePosition;
}

/// Given a utf8 string, a starting position and length returns the
/// corresponding underlying byte range [startByteIndex, endByteIndex).
/// Byte indicies starts from 0, UTF8 character positions starts from 1.
/// If a bad unicode byte is encountered, then we skip that bad byte and
/// count that as one codepoint.
template <bool isAscii>
static inline std::pair<size_t, size_t> getByteRange(
    const char* str,
    size_t strLength,
    size_t startCharPosition,
    size_t length) {
  // If the length is 0, then we return an empty range directly.
  if (length == 0) {
    return std::make_pair(0, 0);
  }

  if (startCharPosition < 1) {
    throw std::invalid_argument("start position must be >= 1");
  }

  VELOX_CHECK_GE(
      strLength,
      length,
      "The length of the string must be at least as large as the length of the substring requested.");

  VELOX_CHECK_GE(
      strLength,
      startCharPosition,
      "The offset of the substring requested must be within the string.");

  if constexpr (isAscii) {
    return std::make_pair(
        startCharPosition - 1, startCharPosition + length - 1);
  } else {
    size_t startByteIndex = 0;
    size_t nextCharOffset = 0;

    // Skips any Unicode continuation bytes. These are bytes that appear after
    // the first byte in a multi-byte Unicode character.  They do not count
    // towards the position in or length of a string.
    auto skipContBytes = [&]() {
      while (nextCharOffset < strLength && utf_cont(str[nextCharOffset])) {
        nextCharOffset++;
      }
    };

    // Skip any invalid continuation bytes at the beginning of the string.
    skipContBytes();

    // Find startByteIndex
    for (size_t i = 0; nextCharOffset < strLength && i < startCharPosition - 1;
         i++) {
      nextCharOffset++;

      skipContBytes();
    }

    startByteIndex = nextCharOffset;
    size_t charCountInRange = 0;

    // Find endByteIndex
    for (size_t i = 0; nextCharOffset < strLength && i < length; i++) {
      nextCharOffset++;
      charCountInRange++;

      skipContBytes();
    }

    VELOX_CHECK_EQ(
        charCountInRange,
        length,
        "The substring requested at {} of length {} exceeds the bounds of the string.",
        startCharPosition,
        length);

    return std::make_pair(startByteIndex, nextCharOffset);
  }
}
} // namespace stringCore
} // namespace facebook::velox::functions
