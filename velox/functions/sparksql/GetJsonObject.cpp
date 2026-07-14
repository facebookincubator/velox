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
#include <string_view>

#include "folly/Unicode.h"

#include "velox/functions/lib/Utf8Utils.h"

namespace facebook::velox::functions::sparksql::detail {

namespace {

// Counts the supplementary-plane characters (4-byte UTF-8 sequences) in `raw`.
size_t countSupplementaryCharacters(std::string_view raw) {
  size_t count = 0;
  const auto* p = reinterpret_cast<const unsigned char*>(raw.data());
  const auto* const end = p + raw.size();
  while (p < end) {
    const int length = validateAndGetNextUtf8Length(p, end);
    if (length == 4) {
      ++count;
    }
    p += length > 0 ? length : 1;
  }
  return count;
}

// Writes `raw` to `out`, applying the escaping described on
// appendWithSupplementaryEscapes. `out` must have room for the result (see
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
    // Fast path: nothing to escape, the raw slice already matches Spark.
    out.append(raw);
    return;
  }
  // Each 4-byte sequence (1 char) becomes a 12-byte surrogate-pair escape
  // ('\uXXXX\uXXXX'), i.e. 8 bytes larger than the raw 4. Escape directly into
  // `out`'s buffer -- reserve the exact worst case, write in place, then resize
  // down to what was actually written, avoiding an intermediate string + copy.
  const size_t oldSize = out.size();
  const size_t maxSize = oldSize + raw.size() + 8 * supplementaryCount;
  out.reserve(maxSize);
  char* const begin = out.data() + oldSize;
  const char* const written = escapeSupplementaryCharacters(raw, begin);
  out.resize(oldSize + static_cast<size_t>(written - begin));
}

} // namespace facebook::velox::functions::sparksql::detail
