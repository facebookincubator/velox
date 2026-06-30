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

// Returns whether `p` begins a valid 4-byte UTF-8 sequence (a supplementary
// plane code point, >= U+10000) within [p, end). 1-/2-/3-byte sequences encode
// BMP code points (<= U+FFFF), which Spark leaves literal.
bool isSupplementaryLead(const unsigned char* p, const unsigned char* end) {
  // Lead byte 11110xxx, followed by three 10xxxxxx continuation bytes.
  return (*p & 0xF8) == 0xF0 && end - p >= 4 && (p[1] & 0xC0) == 0x80 &&
      (p[2] & 0xC0) == 0x80 && (p[3] & 0xC0) == 0x80;
}

// Counts the supplementary-plane characters (4-byte UTF-8 sequences) in `raw`,
// to pick the fast path (none) and to size the escape buffer exactly.
size_t countSupplementaryCharacters(std::string_view raw) {
  size_t count = 0;
  const auto* p = reinterpret_cast<const unsigned char*>(raw.data());
  const auto* end = p + raw.size();
  while (p < end) {
    if (isSupplementaryLead(p, end)) {
      ++count;
      p += 4;
    } else {
      ++p;
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
    if (isSupplementaryLead(p, end)) {
      // Flush the verbatim run accumulated before this sequence in one copy.
      if (p > runStart) {
        out.append(
            reinterpret_cast<const char*>(runStart),
            static_cast<size_t>(p - runStart));
      }
      const char32_t codePoint = (*p & 0x07) << 18 | (p[1] & 0x3F) << 12 |
          (p[2] & 0x3F) << 6 | (p[3] & 0x3F);
      const char32_t v = codePoint - 0x10000u;
      appendUtf16Hex(
          static_cast<char16_t>(0xD800u + ((v >> 10) & 0x3FFu)), out);
      appendUtf16Hex(static_cast<char16_t>(0xDC00u + (v & 0x3FFu)), out);
      p += 4;
      runStart = p;
    } else {
      ++p;
    }
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
