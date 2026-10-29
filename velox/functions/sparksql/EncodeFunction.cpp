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
#include <array>
#include <compare>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include <simdjson.h>

#include "velox/common/base/Exceptions.h"
#include "velox/functions/lib/Utf8Utils.h"
#include "velox/functions/lib/string/StringCore.h"
#include "velox/functions/sparksql/EncodeCharsets.h"

namespace facebook::velox::functions::sparksql::detail {
namespace {

constexpr std::array<std::pair<std::string_view, CharsetType>, 7>
    kStandardCharsets{{
        {"UTF-8", CharsetType::kUtf8},
        {"US-ASCII", CharsetType::kUsAscii},
        {"ISO-8859-1", CharsetType::kIso8859_1},
        {"UTF-16", CharsetType::kUtf16},
        {"UTF-16BE", CharsetType::kUtf16BE},
        {"UTF-16LE", CharsetType::kUtf16LE},
        {"UTF-32", CharsetType::kUtf32},
    }};

static_assert(
    [] {
      if (kStandardCharsets.size() != single_byte::kStandardNames.size()) {
        return false;
      }
      for (size_t i = 0; i < kStandardCharsets.size(); ++i) {
        if (kStandardCharsets[i].first != single_byte::kStandardNames[i]) {
          return false;
        }
      }
      for (const auto& alias : single_byte::kStandardAliases) {
        if (alias.charsetIndex >= kStandardCharsets.size()) {
          return false;
        }
      }
      return true;
    }(),
    "Standard charset types must match the generated alias order");

constexpr auto kStandardTypes = [] {
  std::array<CharsetType, kStandardCharsets.size()> types{};
  for (size_t i = 0; i < types.size(); ++i) {
    types[i] = kStandardCharsets[i].second;
  }
  return types;
}();

template <bool foldKelvinSign>
std::strong_ordering compareIgnoreCase(
    const StringView& value,
    std::string_view expected) {
  const auto compareCharacter = [](char left, char right) {
    const auto toUpper = [](char character) {
      return character >= 'a' && character <= 'z'
          ? static_cast<char>(character - ('a' - 'A'))
          : character;
    };
    return toUpper(left) <=> toUpper(right);
  };
  size_t position{0};
  size_t expectedPosition{0};
  while (position < value.size() && expectedPosition < expected.size()) {
    auto character = value.data()[position++];
    if constexpr (foldKelvinSign) {
      // Spark lowercases names; only U+212A folds from non-ASCII to ASCII.
      if (character == '\xE2' && value.size() - position >= 2 &&
          value.data()[position] == '\x84' &&
          value.data()[position + 1] == '\xAA') {
        character = 'k';
        position += 2;
      }
    }
    if (const auto comparison =
            compareCharacter(character, expected[expectedPosition++]);
        comparison != 0) {
      return comparison;
    }
  }
  return (value.size() - position) <=> (expected.size() - expectedPosition);
}

bool equalsIgnoreCase(const StringView& value, std::string_view expected) {
  return compareIgnoreCase<false>(value, expected) == 0;
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

size_t decodeCodePointWithAsciiFastPath(
    const char* input,
    size_t inputSize,
    size_t position,
    char32_t& codePoint) {
  const auto firstByte = static_cast<uint8_t>(input[position]);
  if (firstByte < 0x80) {
    codePoint = firstByte;
    return 1;
  }
  return decodeCodePoint(input, inputSize, position, codePoint);
}

// Keep SIMD stack alignment out of the scalar conversion paths.
FOLLY_NOINLINE bool isAsciiRange(const char* input, size_t size) {
  return stringCore::isAscii(input, size);
}

bool isAscii(const StringView& input) {
  if (input.empty()) {
    return true;
  }
  if (static_cast<uint8_t>(input.data()[0]) >= 0x80) {
    return false;
  }
  if (input.isInline()) {
    for (size_t i = 1; i < input.size(); ++i) {
      if (static_cast<uint8_t>(input.data()[i]) >= 0x80) {
        return false;
      }
    }
    return true;
  }
  return isAsciiRange(input.data(), input.size());
}

bool isValidUtf8(const StringView& input) {
  if (!input.isInline()) {
    return simdjson::validate_utf8(input.data(), input.size());
  }
  size_t position{0};
  while (position < input.size()) {
    if (static_cast<uint8_t>(input.data()[position]) < 0x80) {
      ++position;
      continue;
    }
    int32_t codePoint;
    const auto size = tryGetUtf8CharLength(
        input.data() + position, input.size() - position, codePoint);
    if (size < 0) {
      return false;
    }
    position += size;
  }
  return true;
}

template <bool asciiFastPath>
size_t countCodePoints(const StringView& input) {
  size_t count{0};
  size_t position{0};
  while (position < input.size()) {
    char32_t codePoint;
    if constexpr (asciiFastPath) {
      position += decodeCodePointWithAsciiFastPath(
          input.data(), input.size(), position, codePoint);
    } else {
      position +=
          decodeCodePoint(input.data(), input.size(), position, codePoint);
    }
    ++count;
  }
  return count;
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
  if (isValidUtf8(input)) {
    result.setNoCopy(input);
    return Status::OK();
  }

  size_t outputSize{0};
  size_t position{0};
  while (position < input.size()) {
    char32_t codePoint;
    const auto bytesConsumed = decodeCodePointWithAsciiFastPath(
        input.data(), input.size(), position, codePoint);
    outputSize += utf8Length(codePoint);
    position += bytesConsumed;
  }

  result.resize(outputSize);
  position = 0;
  size_t outputPosition{0};
  while (position < input.size()) {
    char32_t codePoint;
    const auto bytesConsumed = decodeCodePointWithAsciiFastPath(
        input.data(), input.size(), position, codePoint);
    outputPosition += writeUtf8(codePoint, result.data() + outputPosition);
    position += bytesConsumed;
  }
  return Status::OK();
}

template <bool copyAsciiRuns>
FOLLY_ALWAYS_INLINE Status encodeSingleByteNonAscii(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& charset,
    char32_t limit,
    bool replaceUnmappable) {
  result.resize(input.size());
  size_t inputPosition{0};
  size_t outputPosition{0};
  while (inputPosition < input.size()) {
    if constexpr (copyAsciiRuns) {
      if (static_cast<uint8_t>(input.data()[inputPosition]) < 0x80) {
        const auto begin = inputPosition++;
        while (inputPosition < input.size() &&
               static_cast<uint8_t>(input.data()[inputPosition]) < 0x80) {
          ++inputPosition;
        }
        const auto size = inputPosition - begin;
        std::memcpy(result.data() + outputPosition, input.data() + begin, size);
        outputPosition += size;
        continue;
      }
    }
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

Status encodeSingleByte(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& charset,
    char32_t limit,
    bool replaceUnmappable) {
  if (isAscii(input)) {
    result.setNoCopy(input);
    return Status::OK();
  }
  if (static_cast<uint8_t>(input.data()[0]) >= 0x80) {
    return encodeSingleByteNonAscii<false>(
        result, input, charset, limit, replaceUnmappable);
  }
  return encodeSingleByteNonAscii<true>(
      result, input, charset, limit, replaceUnmappable);
}

Status encodeLegacySingleByte(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& charset,
    uint8_t charsetIndex,
    bool replaceUnmappable) {
  const auto& encoding = single_byte::kCharsets[charsetIndex];
  if (encoding.asciiCompatible && isAscii(input)) {
    result.setNoCopy(input);
    return Status::OK();
  }
  result.resize(input.size());
  size_t inputPosition{0};
  size_t outputPosition{0};
  while (inputPosition < input.size()) {
    char32_t codePoint;
    inputPosition += decodeCodePointWithAsciiFastPath(
        input.data(), input.size(), inputPosition, codePoint);
    uint16_t encodedByte{single_byte::kUnmappable};
    if (codePoint < 256) {
      encodedByte = single_byte::kLowMappings[charsetIndex][codePoint];
    } else {
      const auto begin =
          single_byte::kExtraMappings.begin() + encoding.extraOffset;
      const auto end = begin + encoding.extraSize;
      const auto mapping = std::lower_bound(
          begin, end, codePoint, [](const auto& entry, char32_t value) {
            return entry.codePoint < value;
          });
      if (mapping != end && mapping->codePoint == codePoint) {
        encodedByte = mapping->encodedByte;
      }
    }
    if (encodedByte == single_byte::kUnmappable) {
      if (!replaceUnmappable) {
        return unmappableCharacter(charset);
      }
      encodedByte = encoding.replacement;
    }
    result.data()[outputPosition++] = static_cast<char>(encodedByte);
  }
  result.resize(outputPosition);
  return Status::OK();
}

template <bool bigEndian, bool includeBom>
Status encodeUtf16(exec::StringWriter& result, const StringView& input) {
  if (input.empty()) {
    result.resize(0);
    return Status::OK();
  }
  const bool ascii = isAscii(input);
  size_t outputSize{includeBom ? 2u : 0u};
  if (ascii) {
    outputSize += input.size() * 2;
  } else {
    size_t position{0};
    while (position < input.size()) {
      char32_t codePoint;
      position += decodeCodePointWithAsciiFastPath(
          input.data(), input.size(), position, codePoint);
      outputSize += codePoint <= 0xFFFF ? 2 : 4;
    }
  }
  result.resize(outputSize);
  size_t outputPosition{0};
  if (includeBom) {
    result.data()[outputPosition++] =
        static_cast<char>(bigEndian ? 0xFE : 0xFF);
    result.data()[outputPosition++] =
        static_cast<char>(bigEndian ? 0xFF : 0xFE);
  }
  if (ascii) {
    for (size_t i = 0; i < input.size(); ++i) {
      writeUtf16(
          static_cast<uint8_t>(input.data()[i]),
          result.data() + outputPosition,
          bigEndian);
      outputPosition += 2;
    }
    return Status::OK();
  }
  size_t inputPosition{0};
  while (inputPosition < input.size()) {
    char32_t codePoint;
    inputPosition += decodeCodePointWithAsciiFastPath(
        input.data(), input.size(), inputPosition, codePoint);
    outputPosition +=
        writeUtf16(codePoint, result.data() + outputPosition, bigEndian);
  }
  return Status::OK();
}

Status encodeUtf32(
    exec::StringWriter& result,
    const StringView& input,
    bool bigEndian) {
  if (isAscii(input)) {
    result.resize(input.size() * 4);
    for (size_t i = 0; i < input.size(); ++i) {
      writeUtf32(
          static_cast<uint8_t>(input.data()[i]),
          result.data() + i * 4,
          bigEndian);
    }
    return Status::OK();
  }
  size_t numCodePoints{0};
  // Amortize validation before using the vectorizable UTF-8 byte counter.
  constexpr size_t kMinByteCountSize = 64;
  if (input.size() >= kMinByteCountSize &&
      simdjson::validate_utf8(input.data(), input.size())) {
    numCodePoints = stringCore::lengthUnicode(input.data(), input.size());
  } else if (static_cast<uint8_t>(input.data()[0]) < 0x80) {
    numCodePoints = countCodePoints<true>(input);
  } else {
    numCodePoints = countCodePoints<false>(input);
  }
  result.resize(numCodePoints * 4);
  size_t inputPosition{0};
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

Charset resolveCharset(const StringView& charset, bool legacyJavaCharsets) {
  if (equalsIgnoreCase(charset, "UTF-8")) {
    return {CharsetType::kUtf8};
  }
  if (equalsIgnoreCase(charset, "US-ASCII")) {
    return {CharsetType::kUsAscii};
  }
  if (equalsIgnoreCase(charset, "ISO-8859-1")) {
    return {CharsetType::kIso8859_1};
  }
  if (equalsIgnoreCase(charset, "UTF-16")) {
    return {CharsetType::kUtf16};
  }
  if (equalsIgnoreCase(charset, "UTF-16BE")) {
    return {CharsetType::kUtf16BE};
  }
  if (equalsIgnoreCase(charset, "UTF-16LE")) {
    return {CharsetType::kUtf16LE};
  }
  if (equalsIgnoreCase(charset, "UTF-32")) {
    return {CharsetType::kUtf32};
  }
  if (legacyJavaCharsets) {
    const auto findAlias = [&](const auto& aliases) {
      return std::lower_bound(
          aliases.begin(),
          aliases.end(),
          charset,
          [](const auto& alias, const StringView& name) {
            return compareIgnoreCase<true>(name, alias.name) > 0;
          });
    };
    const auto standard = findAlias(single_byte::kStandardAliases);
    if (standard != single_byte::kStandardAliases.end() &&
        compareIgnoreCase<true>(charset, standard->name) == 0) {
      return {kStandardTypes[standard->charsetIndex]};
    }
    const auto legacy = findAlias(single_byte::kLegacyAliases);
    if (legacy != single_byte::kLegacyAliases.end() &&
        compareIgnoreCase<true>(charset, legacy->name) == 0) {
      return {CharsetType::kLegacySingleByte, legacy->charsetIndex};
    }
  }
  return {};
}

Status encode(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& charset,
    Charset resolved,
    bool legacyCodingErrorAction) {
  switch (resolved.type) {
    case CharsetType::kUtf8:
      return encodeUtf8(result, input);
    case CharsetType::kUsAscii:
      return encodeSingleByte(
          result, input, charset, 0x80, legacyCodingErrorAction);
    case CharsetType::kIso8859_1:
      return encodeSingleByte(
          result, input, charset, 0x100, legacyCodingErrorAction);
    case CharsetType::kUtf16:
      return encodeUtf16<true, true>(result, input);
    case CharsetType::kUtf16BE:
      return encodeUtf16<true, false>(result, input);
    case CharsetType::kUtf16LE:
      return encodeUtf16<false, false>(result, input);
    case CharsetType::kUtf32:
      return encodeUtf32(result, input, true);
    case CharsetType::kLegacySingleByte:
      return encodeLegacySingleByte(
          result,
          input,
          charset,
          resolved.legacyIndex,
          legacyCodingErrorAction);
    case CharsetType::kUnsupported:
      VELOX_UNREACHABLE();
  }
  VELOX_UNREACHABLE();
}

} // namespace facebook::velox::functions::sparksql::detail
