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

#include "velox/functions/sparksql/GetJsonObject.h"

#include <cstddef>
#include <string>
#include <string_view>

#include "velox/functions/lib/Utf8Utils.h"

namespace facebook::velox::functions::sparksql::detail {

namespace {

// Appends `value` as a 4-hex-digit '\uXXXX' escape (uppercase) to `out`.
void appendUtf16Hex(char16_t value, std::string& out) {
  static constexpr char kHexDigits[] = "0123456789ABCDEF";
  out.push_back('\\');
  out.push_back('u');
  out.push_back(kHexDigits[(value >> 12) & 0x0F]);
  out.push_back(kHexDigits[(value >> 8) & 0x0F]);
  out.push_back(kHexDigits[(value >> 4) & 0x0F]);
  out.push_back(kHexDigits[value & 0x0F]);
}

// Returns the byte length of the UTF-8 character at [p, end) and, when it is a
// supplementary-plane character (code point >= U+10000, i.e. a 4-byte
// sequence), sets `isSupplementary` and fills `codePoint`. For BMP characters
// (<= U+FFFF, which Spark leaves literal) and for invalid sequences, advances
// by the reported length without setting `isSupplementary`; the caller copies
// those bytes verbatim.
size_t nextCharLength(
    const unsigned char* p,
    const unsigned char* end,
    bool& isSupplementary,
    char32_t& codePoint) {
  isSupplementary = false;
  int32_t decoded;
  const int32_t length = tryGetUtf8CharLength(
      reinterpret_cast<const char*>(p), static_cast<int64_t>(end - p), decoded);
  if (length < 0) {
    // Invalid sequence; copy its bytes verbatim.
    return static_cast<size_t>(-length);
  }
  if (length == 4) {
    isSupplementary = true;
    codePoint = static_cast<char32_t>(decoded);
  }
  return static_cast<size_t>(length);
}

// Counts the supplementary-plane characters (4-byte UTF-8 sequences) in `raw`,
// to pick the fast path (none) and to size the escape buffer exactly.
size_t countSupplementaryCharacters(std::string_view raw) {
  size_t count = 0;
  const auto* p = reinterpret_cast<const unsigned char*>(raw.data());
  const auto* end = p + raw.size();
  while (p < end) {
    bool isSupplementary;
    char32_t codePoint;
    p += nextCharLength(p, end, isSupplementary, codePoint);
    if (isSupplementary) {
      ++count;
    }
  }
  return count;
}

// Copies `raw` into `out`, rewriting each supplementary-plane character into a
// '\uXXXX\uXXXX' surrogate-pair escape and copying every other byte verbatim in
// bulk runs.
void escapeSupplementaryCharacters(std::string_view raw, std::string& out) {
  const auto* const base = reinterpret_cast<const unsigned char*>(raw.data());
  const auto* const end = base + raw.size();
  const auto* p = base;
  const auto* runStart = base;
  while (p < end) {
    bool isSupplementary;
    char32_t codePoint;
    const size_t length = nextCharLength(p, end, isSupplementary, codePoint);
    if (isSupplementary) {
      // Flush the verbatim run accumulated before this sequence in one copy.
      if (p > runStart) {
        out.append(
            reinterpret_cast<const char*>(runStart),
            static_cast<size_t>(p - runStart));
      }
      const char32_t v = codePoint - 0x10000u;
      appendUtf16Hex(
          static_cast<char16_t>(0xD800u + ((v >> 10) & 0x3FFu)), out);
      appendUtf16Hex(static_cast<char16_t>(0xDC00u + (v & 0x3FFu)), out);
      runStart = p + length;
    }
    p += length;
  }
  // Flush the trailing verbatim run.
  if (p > runStart) {
    out.append(
        reinterpret_cast<const char*>(runStart),
        static_cast<size_t>(p - runStart));
  }
}

} // namespace

void appendWithSupplementaryEscapes(
    std::string_view raw,
    exec::StringWriter& out) {
  const size_t supplementaryCount = countSupplementaryCharacters(raw);
  if (supplementaryCount == 0) {
    // Fast path: no supplementary-plane characters, so the raw slice already
    // matches Spark. Append it directly, no allocation.
    out.append(raw);
    return;
  }
  // Each 4-byte sequence (1 char) becomes a 12-byte surrogate-pair escape
  // ('\uXXXX\uXXXX'), i.e. 8 bytes larger than the raw 4.
  std::string escaped;
  escaped.reserve(raw.size() + 8 * supplementaryCount);
  escapeSupplementaryCharacters(raw, escaped);
  out.append(escaped);
}

} // namespace facebook::velox::functions::sparksql::detail
