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
#include "velox/functions/sparksql/Decode.h"

#include <algorithm>
#include <array>
#include <optional>

#include <folly/Range.h>

#include "velox/external/utf8proc/utf8procImpl.h"
#include "velox/functions/lib/Utf8Utils.h"

namespace facebook::velox::functions::sparksql::detail {
namespace {

enum class Charset {
  kAscii,
  kLatin1,
  kUtf8,
  kUtf16Be,
  kUtf16Le,
  kUtf16,
  kUtf32
};

std::optional<Charset> findCharset(const StringView& name) {
  static constexpr std::array<std::pair<std::string_view, Charset>, 7>
      kCharsets{{
          {"US-ASCII", Charset::kAscii},
          {"ISO-8859-1", Charset::kLatin1},
          {"UTF-8", Charset::kUtf8},
          {"UTF-16BE", Charset::kUtf16Be},
          {"UTF-16LE", Charset::kUtf16Le},
          {"UTF-16", Charset::kUtf16},
          {"UTF-32", Charset::kUtf32},
      }};
  for (const auto& [canonical, charset] : kCharsets) {
    if (name.size() == canonical.size() &&
        std::equal(
            name.data(),
            name.data() + name.size(),
            canonical.data(),
            folly::AsciiCaseInsensitive{})) {
      return charset;
    }
  }
  return std::nullopt;
}

// Only Unicode scalar values reach this helper, including the replacement
// character. Use explicit lengths so U+0000 remains part of the output.
void appendCodePoint(uint32_t codePoint, exec::StringWriter& out) {
  utf8proc_uint8_t bytes[4];
  const auto size =
      utf8proc_encode_char(static_cast<utf8proc_int32_t>(codePoint), bytes);
  out.append(std::string_view(reinterpret_cast<const char*>(bytes), size));
}

uint32_t read16(const uint8_t* bytes, bool littleEndian) {
  return littleEndian ? (uint32_t{bytes[1]} << 8) | bytes[0]
                      : (uint32_t{bytes[0]} << 8) | bytes[1];
}

void decodeUtf16(
    const StringView& input,
    Charset mode,
    exec::StringWriter& out) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(input.data());
  const size_t size = input.size();
  size_t offset = 0;
  bool littleEndian = mode == Charset::kUtf16Le;
  if (mode == Charset::kUtf16 && size >= 2) {
    const auto first = read16(bytes, false);
    if (first == 0xfeff || first == 0xfffe) {
      littleEndian = first == 0xfffe;
      offset = 2;
    }
  }
  while (size - offset >= 2) {
    auto codePoint = read16(bytes + offset, littleEndian);
    offset += 2;
    if (codePoint >= 0xd800 && codePoint <= 0xdbff) {
      if (size - offset >= 2) {
        const auto low = read16(bytes + offset, littleEndian);
        offset += 2;
        // JDK replacement consumes both units even when the second is not
        // a low surrogate. It must not be decoded again as ordinary text.
        codePoint = low >= 0xdc00 && low <= 0xdfff
            ? 0x10000 + ((codePoint - 0xd800) << 10) + (low - 0xdc00)
            : 0xfffd;
      } else {
        // A high surrogate and one final byte form a single malformed unit.
        offset = size;
        codePoint = 0xfffd;
      }
    } else if (codePoint >= 0xdc00 && codePoint <= 0xdfff) {
      codePoint = 0xfffd;
    }
    appendCodePoint(codePoint, out);
  }
  if (offset < size) {
    appendCodePoint(0xfffd, out);
  }
}

uint32_t read32(const uint8_t* bytes, bool littleEndian) {
  return littleEndian ? (uint32_t{bytes[3]} << 24) |
          (uint32_t{bytes[2]} << 16) | (uint32_t{bytes[1]} << 8) | bytes[0]
                      : (uint32_t{bytes[0]} << 24) |
          (uint32_t{bytes[1]} << 16) | (uint32_t{bytes[2]} << 8) | bytes[3];
}

void decodeUtf32(const StringView& input, exec::StringWriter& out) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(input.data());
  const size_t size = input.size();
  size_t offset = 0;
  bool littleEndian = false;
  if (size >= 4) {
    const auto first = read32(bytes, false);
    if (first == 0xfeff || first == 0xfffe0000) {
      littleEndian = first == 0xfffe0000;
      offset = 4;
    }
  }
  // JDK's UTF-32 decoder can emit surrogate-valued UTF-16 units. Model the
  // subsequent Java String -> UTF-8 conversion: adjacent high/low units pair,
  // while unpaired units become '?' (not U+FFFD or encoded surrogates).
  uint32_t high = 0;
  while (size - offset >= 4) {
    auto codePoint = read32(bytes + offset, littleEndian);
    offset += 4;
    if (codePoint > 0x10ffff) {
      codePoint = 0xfffd;
    }
    if (high != 0) {
      if (codePoint >= 0xdc00 && codePoint <= 0xdfff) {
        appendCodePoint(
            0x10000 + ((high - 0xd800) << 10) + (codePoint - 0xdc00), out);
        high = 0;
        continue;
      }
      appendCodePoint('?', out);
      high = 0;
    }
    if (codePoint >= 0xd800 && codePoint <= 0xdbff) {
      high = codePoint;
    } else {
      appendCodePoint(
          codePoint >= 0xdc00 && codePoint <= 0xdfff ? '?' : codePoint, out);
    }
  }
  if (high != 0) {
    appendCodePoint('?', out);
  }
  if (offset < size) {
    appendCodePoint(0xfffd, out);
  }
}

} // namespace

Status decode(
    const StringView& input,
    const StringView& charset,
    exec::StringWriter& out) {
  const auto mode = findCharset(charset);
  if (!mode.has_value()) {
    return Status::UserError("Unsupported charset: {}", charset);
  }
  if (input.size() == 0) {
    return Status::OK();
  }
  if (*mode == Charset::kUtf8) {
    functions::replaceInvalidUTF8Characters(
        out, input.data(), static_cast<int32_t>(input.size()));
    return Status::OK();
  }
  // StringWriter does not grow geometrically. Reserve an upper bound once
  // to avoid repeated allocation/copying during scalar appends. Keep sizing
  // in size_t; the writer enforces the actual result size when finalized.
  out.reserve(input.size() * 3);
  if (*mode == Charset::kAscii || *mode == Charset::kLatin1) {
    for (size_t i = 0; i < input.size(); ++i) {
      const auto byte = static_cast<uint8_t>(input.data()[i]);
      appendCodePoint(
          *mode == Charset::kAscii && byte > 0x7f ? 0xfffd : byte, out);
    }
    return Status::OK();
  }
  if (*mode == Charset::kUtf32) {
    decodeUtf32(input, out);
  } else {
    decodeUtf16(input, *mode, out);
  }
  return Status::OK();
}

} // namespace facebook::velox::functions::sparksql::detail
