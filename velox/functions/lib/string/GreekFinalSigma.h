/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "folly/CPortability.h"
#include "velox/external/utf8proc/utf8procImpl.h"

namespace facebook::velox::functions {

/// Helper function to check if a character is cased. Compatible with the
/// 'isCased' implementation in 'ConditionalSpecialCasting.java' of JDK, which
/// is used by 'toLowerCase' function in Spark SQL.
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

/// Helper function to check if a character is case-ignorable.
/// Case-ignorable characters can be skipped when checking for cased characters
/// before or after Σ.
FOLLY_ALWAYS_INLINE bool isCaseIgnorable(utf8proc_int32_t ch) {
  auto cat = utf8proc_category(ch);
  // General_Category: Mn (Nonspacing Mark), Me (Enclosing Mark),
  // Cf (Format), Lm (Modifier Letter), Sk (Modifier Symbol).
  if (cat == UTF8PROC_CATEGORY_MN || cat == UTF8PROC_CATEGORY_ME ||
      cat == UTF8PROC_CATEGORY_CF || cat == UTF8PROC_CATEGORY_LM ||
      cat == UTF8PROC_CATEGORY_SK) {
    return true;
  }
  // Word_Break property: MidLetter, MidNumLet, Single_Quote.
  switch (ch) {
    case 0x00B7: // · MIDDLE DOT
    case 0x0387: // · GREEK ANO TELEIA
    case 0x05F4: // ״ HEBREW PUNCTUATION GERSHAYIM
    case 0x2027: // ‧ HYPHENATION POINT
    case 0xFE13: // ︓ PRESENTATION FORM FOR VERTICAL COLON
    case 0xFE55: // ﹕ SMALL COLON
    case 0xFF1A: // ： FULLWIDTH COLON
    case 0x002E: // . FULL STOP
    case 0x2018: // ' LEFT SINGLE QUOTATION MARK
    case 0x2019: // ' RIGHT SINGLE QUOTATION MARK
    case 0x2024: // ․ ONE DOT LEADER
    case 0xFE52: // ﹒ SMALL FULL STOP
    case 0xFF07: // ＇ FULLWIDTH APOSTROPHE
    case 0xFF0E: // ． FULLWIDTH FULL STOP
    case 0x0027: // ' APOSTROPHE
      return true;
    default:
      return false;
  }
}

/// Scan backward to check if there is a cased character before the position.
/// Skips case-ignorable characters during the scan.
FOLLY_ALWAYS_INLINE bool hasCasedBefore(
    const char* input,
    size_t sigmaStartPos) {
  if (sigmaStartPos == 0) {
    return false;
  }

  size_t pos = sigmaStartPos;
  while (pos > 0) {
    // Move backward to the start of the previous UTF-8 character.
    size_t prevPos = pos - 1;
    while (prevPos > 0 &&
           (static_cast<unsigned char>(input[prevPos]) & 0xC0) == 0x80) {
      --prevPos;
    }

    int size;
    utf8proc_int32_t cp =
        utf8proc_codepoint(&input[prevPos], &input[pos], size);

    if (cp == -1) {
      return false;
    }
    if (isCased(cp)) {
      return true;
    }
    if (!isCaseIgnorable(cp)) {
      return false;
    }
    pos = prevPos;
  }
  return false;
}

/// Scan forward to check if there is a cased character after the position.
/// Skips case-ignorable characters during the scan.
FOLLY_ALWAYS_INLINE bool
hasCasedAfter(const char* input, size_t inputLength, size_t afterSigmaPos) {
  size_t pos = afterSigmaPos;
  while (pos < inputLength) {
    int size;
    utf8proc_int32_t cp =
        utf8proc_codepoint(&input[pos], input + inputLength, size);

    if (cp == -1) {
      return false;
    }
    if (isCased(cp)) {
      return true;
    }
    if (!isCaseIgnorable(cp)) {
      return false;
    }
    pos += size;
  }
  return false;
}

/// Check if the Greek capital letter Σ (U+03A3) at the given position should
/// be converted to final sigma (ς) or regular sigma (σ).
///
/// According to the Final_Sigma rule in Unicode Standard Section 3.13:
/// https://www.unicode.org/versions/Unicode15.0.0/ch03.pdf
///
/// A capital sigma maps to final sigma (ς) if and only if:
/// 1. There is a cased character before Σ (possibly separated by
///    case-ignorable characters)
/// 2. There is no cased character after Σ (possibly separated by
///    case-ignorable characters)
///
/// @param input The input string.
/// @param inputLength The length of the input string in bytes.
/// @param sigmaStartPos The byte position where Σ starts.
/// @param afterSigmaPos The byte position immediately after Σ.
/// @return true if Σ should be converted to final sigma (ς), false for regular
/// sigma (σ).
FOLLY_ALWAYS_INLINE bool isFinalSigma(
    const char* input,
    size_t inputLength,
    size_t sigmaStartPos,
    size_t afterSigmaPos) {
  return hasCasedBefore(input, sigmaStartPos) &&
      !hasCasedAfter(input, inputLength, afterSigmaPos);
}

/// Write the lowercase form of Greek capital letter Σ (U+03A3) to output.
/// Writes ς (U+03C2) if isFinal is true, otherwise writes σ (U+03C3).
///
/// @param output The output buffer to write to.
/// @param isFinal If true, write final sigma (ς); otherwise write regular
/// sigma (σ).
/// @return The number of bytes written (always 2 for these characters).
FOLLY_ALWAYS_INLINE int writeLowerSigma(char* output, bool isFinal) {
  // ς (U+03C2) = 0xCF 0x82, σ (U+03C3) = 0xCF 0x83
  constexpr utf8proc_int32_t kFinalSigma = 0x03C2;
  constexpr utf8proc_int32_t kRegularSigma = 0x03C3;
  utf8proc_int32_t lowerSigma = isFinal ? kFinalSigma : kRegularSigma;
  return utf8proc_encode_char(
      lowerSigma, reinterpret_cast<unsigned char*>(output));
}

} // namespace facebook::velox::functions
