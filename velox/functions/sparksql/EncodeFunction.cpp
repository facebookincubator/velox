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

#include "velox/functions/sparksql/EncodeFunction.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>

#include <folly/Range.h>

#include "velox/common/base/Exceptions.h"
#include "velox/functions/lib/Utf8Utils.h"

namespace facebook::velox::functions::sparksql::detail {
namespace {

constexpr char32_t kReplacementCodePoint = 0xFFFD;

bool equalsIgnoreAsciiCase(std::string_view value, std::string_view expected) {
  return value.size() == expected.size() &&
      std::equal(
             value.begin(),
             value.end(),
             expected.begin(),
             folly::AsciiCaseInsensitive{});
}

size_t decodeCodePoint(
    const char* input,
    size_t inputSize,
    size_t position,
    char32_t& codePoint) {
  int32_t decodedCodePoint;
  const auto bytesConsumed = decodeUtf8CodePointOrReplacement(
      input + position, inputSize - position, decodedCodePoint);
  codePoint = static_cast<char32_t>(decodedCodePoint);
  return bytesConsumed;
}

size_t utf8Length(char32_t codePoint) {
  if (codePoint <= 0x7F) {
    return 1;
  }
  if (codePoint <= 0x7FF) {
    return 2;
  }
  if (codePoint <= 0xFFFF) {
    return 3;
  }
  return 4;
}

size_t writeUtf8(char32_t codePoint, char* output) {
  if (codePoint <= 0x7F) {
    output[0] = static_cast<char>(codePoint);
    return 1;
  }
  if (codePoint <= 0x7FF) {
    output[0] = static_cast<char>(0xC0 | (codePoint >> 6));
    output[1] = static_cast<char>(0x80 | (codePoint & 0x3F));
    return 2;
  }
  if (codePoint <= 0xFFFF) {
    output[0] = static_cast<char>(0xE0 | (codePoint >> 12));
    output[1] = static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F));
    output[2] = static_cast<char>(0x80 | (codePoint & 0x3F));
    return 3;
  }
  output[0] = static_cast<char>(0xF0 | (codePoint >> 18));
  output[1] = static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F));
  output[2] = static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F));
  output[3] = static_cast<char>(0x80 | (codePoint & 0x3F));
  return 4;
}

size_t writeUtf16(char32_t codePoint, char* output, bool bigEndian) {
  auto writeUnit = [&](char16_t unit, char* destination) {
    if (bigEndian) {
      destination[0] = static_cast<char>((unit >> 8) & 0xFF);
      destination[1] = static_cast<char>(unit & 0xFF);
    } else {
      destination[0] = static_cast<char>(unit & 0xFF);
      destination[1] = static_cast<char>((unit >> 8) & 0xFF);
    }
  };

  if (codePoint <= 0xFFFF) {
    writeUnit(static_cast<char16_t>(codePoint), output);
    return 2;
  }
  const auto adjusted = codePoint - 0x10000;
  writeUnit(static_cast<char16_t>(0xD800 + (adjusted >> 10)), output);
  writeUnit(static_cast<char16_t>(0xDC00 + (adjusted & 0x3FF)), output + 2);
  return 4;
}

void writeUtf32(char32_t codePoint, char* output, bool bigEndian) {
  for (size_t i = 0; i < 4; ++i) {
    const auto shift = bigEndian ? 24 - 8 * i : 8 * i;
    output[i] = static_cast<char>((codePoint >> shift) & 0xFF);
  }
}

Status unmappableCharacter(const StringView& charset) {
  return Status::UserError(
      "encode: input contains a character that cannot be encoded using '{}'",
      std::string(charset.data(), charset.size()));
}

Status encodeUtf8(exec::StringWriter& result, const StringView& input) {
  size_t outputSize{0};
  size_t position{0};
  bool isValid{true};
  while (position < input.size()) {
    char32_t codePoint;
    const auto bytesConsumed =
        decodeCodePoint(input.data(), input.size(), position, codePoint);
    if (codePoint == kReplacementCodePoint &&
        !(bytesConsumed == 3 &&
          static_cast<unsigned char>(input.data()[position]) == 0xEF &&
          static_cast<unsigned char>(input.data()[position + 1]) == 0xBF &&
          static_cast<unsigned char>(input.data()[position + 2]) == 0xBD)) {
      isValid = false;
    }
    outputSize += utf8Length(codePoint);
    position += bytesConsumed;
  }

  result.resize(outputSize);
  if (isValid) {
    if (!input.empty()) {
      std::memcpy(result.data(), input.data(), input.size());
    }
    return Status::OK();
  }

  position = 0;
  size_t outputPosition{0};
  while (position < input.size()) {
    char32_t codePoint;
    const auto bytesConsumed =
        decodeCodePoint(input.data(), input.size(), position, codePoint);
    outputPosition += writeUtf8(codePoint, result.data() + outputPosition);
    position += bytesConsumed;
  }
  return Status::OK();
}

Status encodeSingleByte(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& charset,
    char32_t limit,
    bool replaceUnmappable) {
  result.resize(input.size());
  size_t inputPosition{0};
  size_t outputPosition{0};
  while (inputPosition < input.size()) {
    char32_t codePoint;
    inputPosition +=
        decodeCodePoint(input.data(), input.size(), inputPosition, codePoint);
    if (codePoint >= limit) {
      if (!replaceUnmappable) {
        return unmappableCharacter(charset);
      }
      codePoint = '?';
    }
    result.data()[outputPosition++] = static_cast<char>(codePoint);
  }
  result.resize(outputPosition);
  return Status::OK();
}

Status encodeUtf16(
    exec::StringWriter& result,
    const StringView& input,
    bool bigEndian,
    bool includeBom) {
  if (input.empty()) {
    result.resize(0);
    return Status::OK();
  }
  size_t outputSize{includeBom ? 2u : 0u};
  size_t position{0};
  while (position < input.size()) {
    char32_t codePoint;
    position +=
        decodeCodePoint(input.data(), input.size(), position, codePoint);
    outputSize += codePoint <= 0xFFFF ? 2 : 4;
  }
  result.resize(outputSize);
  size_t outputPosition{0};
  if (includeBom) {
    result.data()[outputPosition++] =
        static_cast<char>(bigEndian ? 0xFE : 0xFF);
    result.data()[outputPosition++] =
        static_cast<char>(bigEndian ? 0xFF : 0xFE);
  }
  size_t inputPosition{0};
  while (inputPosition < input.size()) {
    char32_t codePoint;
    inputPosition +=
        decodeCodePoint(input.data(), input.size(), inputPosition, codePoint);
    outputPosition +=
        writeUtf16(codePoint, result.data() + outputPosition, bigEndian);
  }
  return Status::OK();
}

Status encodeUtf32(
    exec::StringWriter& result,
    const StringView& input,
    bool bigEndian) {
  size_t numCodePoints{0};
  size_t inputPosition{0};
  while (inputPosition < input.size()) {
    char32_t codePoint;
    inputPosition +=
        decodeCodePoint(input.data(), input.size(), inputPosition, codePoint);
    ++numCodePoints;
  }
  result.resize(numCodePoints * 4);
  inputPosition = 0;
  size_t outputPosition{0};
  while (inputPosition < input.size()) {
    char32_t codePoint;
    inputPosition +=
        decodeCodePoint(input.data(), input.size(), inputPosition, codePoint);
    writeUtf32(codePoint, result.data() + outputPosition, bigEndian);
    outputPosition += 4;
  }
  return Status::OK();
}

} // namespace

CharsetType resolveCharset(const StringView& charset) {
  if (equalsIgnoreAsciiCase(charset, "UTF-8")) {
    return CharsetType::kUtf8;
  }
  if (equalsIgnoreAsciiCase(charset, "US-ASCII")) {
    return CharsetType::kUsAscii;
  }
  if (equalsIgnoreAsciiCase(charset, "ISO-8859-1")) {
    return CharsetType::kIso8859_1;
  }
  if (equalsIgnoreAsciiCase(charset, "UTF-16")) {
    return CharsetType::kUtf16;
  }
  if (equalsIgnoreAsciiCase(charset, "UTF-16BE")) {
    return CharsetType::kUtf16BE;
  }
  if (equalsIgnoreAsciiCase(charset, "UTF-16LE")) {
    return CharsetType::kUtf16LE;
  }
  if (equalsIgnoreAsciiCase(charset, "UTF-32")) {
    return CharsetType::kUtf32;
  }
  return CharsetType::kUnsupported;
}

Status encode(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& charset,
    CharsetType type,
    bool legacyCodingErrorAction) {
  switch (type) {
    case CharsetType::kUtf8:
      return encodeUtf8(result, input);
    case CharsetType::kUsAscii:
      return encodeSingleByte(
          result, input, charset, 0x80, legacyCodingErrorAction);
    case CharsetType::kIso8859_1:
      return encodeSingleByte(
          result, input, charset, 0x100, legacyCodingErrorAction);
    case CharsetType::kUtf16:
      return encodeUtf16(result, input, true, true);
    case CharsetType::kUtf16BE:
      return encodeUtf16(result, input, true, false);
    case CharsetType::kUtf16LE:
      return encodeUtf16(result, input, false, false);
    case CharsetType::kUtf32:
      return encodeUtf32(result, input, true);
    case CharsetType::kUnsupported:
      VELOX_UNREACHABLE();
  }
  VELOX_UNREACHABLE();
}

} // namespace facebook::velox::functions::sparksql::detail
