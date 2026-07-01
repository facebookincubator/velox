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
#include <cstring>
#include <string>
#include <string_view>

#include "folly/Unicode.h"

#include "velox/functions/lib/Utf8Utils.h"

namespace facebook::velox::functions::sparksql::detail {

namespace {

// Counts the supplementary-plane characters (4-byte UTF-8 sequences) in `raw`,
// to pick the fast path (none) and to size the escape buffer exactly.
size_t countSupplementaryCharacters(std::string_view raw) {
  size_t count = 0;
  const auto* p = reinterpret_cast<const unsigned char*>(raw.data());
  const auto* const end = p + raw.size();
  while (p < end) {
    const int length = validateAndGetNextUtf8Length(p, end);
    // A 4-byte sequence is a supplementary-plane code point; shorter sequences
    // are BMP (left literal) and a negative length marks invalid bytes.
    if (length == 4) {
      ++count;
    }
    p += length > 0 ? length : 1;
  }
  return count;
}

// Copies `raw` into `out`, rewriting each supplementary-plane character into a
// '\uXXXX\uXXXX' surrogate-pair escape and copying every other byte verbatim in
// bulk runs. `out` must have room for the escaped result (see
// countSupplementaryCharacters for sizing); returns the past-the-end write
// position.
char* escapeSupplementaryCharacters(std::string_view raw, char* out) {
  const auto* const base = reinterpret_cast<const unsigned char*>(raw.data());
  const auto* const end = base + raw.size();
  const auto* p = base;
  const auto* runStart = base;
  while (p < end) {
    const int length = validateAndGetNextUtf8Length(p, end);
    if (length == 4) {
      // Flush the verbatim run accumulated before this sequence in one copy.
      if (p > runStart) {
        const auto runSize = static_cast<size_t>(p - runStart);
        std::memcpy(out, runStart, runSize);
        out += runSize;
      }
      // folly::utf8ToCodePoint advances `p` past the 4-byte sequence.
      encodeUtf16Hex(folly::utf8ToCodePoint(p, end, true), out);
      runStart = p;
    } else {
      p += length > 0 ? length : 1;
    }
  }
  // Flush the trailing verbatim run.
  if (p > runStart) {
    const auto runSize = static_cast<size_t>(p - runStart);
    std::memcpy(out, runStart, runSize);
    out += runSize;
  }
  return out;
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
  escaped.resize(raw.size() + 8 * supplementaryCount);
  char* const begin = escaped.data();
  const char* const written = escapeSupplementaryCharacters(raw, begin);
  escaped.resize(static_cast<size_t>(written - begin));
  out.append(escaped);
}

} // namespace facebook::velox::functions::sparksql::detail
