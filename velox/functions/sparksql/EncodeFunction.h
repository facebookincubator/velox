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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include <folly/Likely.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Status.h"
#include "velox/functions/Macros.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions::sparksql {

namespace detail {

enum class CharsetType : uint8_t {
  kUtf8,
  kUsAscii,
  kIso8859_1,
  kUtf16, // BOM (0xFE 0xFF) + big-endian data (Java "UTF-16" / "UnicodeBig")
  kUtf16BE, // Big-endian without BOM (Java "UTF-16BE" / "UnicodeBigUnmarked")
  kUtf16LE, // Little-endian without BOM (Java "UTF-16LE" /
            // "UnicodeLittleUnmarked")
  kUtf16LEWithBom, // BOM (0xFF 0xFE) + little-endian data (Java
                   // "UnicodeLittle")
  kUnsupported,
};

/// Resolves a charset name to a CharsetType enum. Supports canonical IANA names
/// and common Java aliases (case-insensitive) to match Spark/Java's
/// Charset.forName() behavior.
/// Uses a stack buffer to avoid heap allocations on hot paths.
inline CharsetType resolveCharset(const char* data, size_t length) {
  // The longest name that can match is "UNICODELITTLEUNMARKED" (21 chars);
  // kMaxCharsetLen (24) leaves a little headroom and bounds the stack buffer.
  // Names longer than kMaxCharsetLen cannot match any candidate, so reject
  // early.
  static constexpr size_t kMaxCharsetLen = 24;
  if (length == 0 || length > kMaxCharsetLen) {
    return CharsetType::kUnsupported;
  }

  char upperName[kMaxCharsetLen];
  for (size_t i = 0; i < length; ++i) {
    // ASCII-only uppercasing: charset names are ASCII, and a locale-sensitive
    // std::toupper could mis-map bytes (e.g. under a Turkish locale) and break
    // matching.
    const char c = data[i];
    upperName[i] =
        (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
  }

  // Use a StringView-like comparison without heap allocation.
  auto eq = [&](const char* candidate, size_t candidateLen) {
    return length == candidateLen &&
        std::memcmp(upperName, candidate, candidateLen) == 0;
  };

  // Length-based dispatch to reduce comparisons on the hot path.
  switch (length) {
    case 4:
      if (eq("UTF8", 4))
        return CharsetType::kUtf8;
      break;
    case 5:
      if (eq("UTF-8", 5))
        return CharsetType::kUtf8;
      if (eq("ASCII", 5))
        return CharsetType::kUsAscii;
      if (eq("UTF16", 5))
        return CharsetType::kUtf16;
      break;
    case 6:
      if (eq("UTF-16", 6))
        return CharsetType::kUtf16;
      if (eq("LATIN1", 6))
        return CharsetType::kIso8859_1;
      break;
    case 7:
      if (eq("UTF16BE", 7))
        return CharsetType::kUtf16BE;
      if (eq("UTF16LE", 7))
        return CharsetType::kUtf16LE;
      break;
    case 8:
      if (eq("US-ASCII", 8))
        return CharsetType::kUsAscii;
      if (eq("US_ASCII", 8))
        return CharsetType::kUsAscii;
      if (eq("UTF-16BE", 8))
        return CharsetType::kUtf16BE;
      if (eq("UTF-16LE", 8))
        return CharsetType::kUtf16LE;
      break;
    case 9:
      if (eq("ISO8859_1", 9))
        return CharsetType::kIso8859_1;
      if (eq("ISO8859-1", 9))
        return CharsetType::kIso8859_1;
      break;
    case 10:
      if (eq("ISO-8859-1", 10))
        return CharsetType::kIso8859_1;
      if (eq("ISO_8859_1", 10))
        return CharsetType::kIso8859_1;
      if (eq("UNICODEBIG", 10))
        return CharsetType::kUtf16; // BOM + big-endian (matches Java
                                    // UnicodeBig)
      break;
    case 13:
      if (eq("UNICODELITTLE", 13))
        return CharsetType::kUtf16LEWithBom; // BOM + little-endian (matches
                                             // Java UnicodeLittle)
      break;
    case 18:
      if (eq("UNICODEBIGUNMARKED", 18))
        return CharsetType::kUtf16BE;
      break;
    case 21:
      if (eq("UNICODELITTLEUNMARKED", 21))
        return CharsetType::kUtf16LE;
      break;
    default:
      break;
  }
  return CharsetType::kUnsupported;
}

/// Decodes the next UTF-8 codepoint from 'data' at position 'pos'.
/// Advances 'pos' past the consumed bytes. Returns U+FFFD on invalid
/// sequences. Note: Velox VARCHAR may contain malformed UTF-8; invalid
/// sequences are replaced with U+FFFD, which then flows into the charset
/// encoder (a Velox-specific behavior since Java Strings are always valid).
inline char32_t
decodeUtf8Codepoint(const char* data, size_t length, size_t& pos) {
  auto byte = static_cast<unsigned char>(data[pos]);
  char32_t codePoint;
  size_t extraBytes;

  if (FOLLY_LIKELY(byte < 0x80)) {
    ++pos;
    return static_cast<char32_t>(byte);
  } else if ((byte & 0xE0) == 0xC0) {
    codePoint = byte & 0x1F;
    extraBytes = 1;
  } else if ((byte & 0xF0) == 0xE0) {
    codePoint = byte & 0x0F;
    extraBytes = 2;
  } else if ((byte & 0xF8) == 0xF0) {
    codePoint = byte & 0x07;
    extraBytes = 3;
  } else {
    ++pos;
    return 0xFFFD;
  }

  if (pos + 1 + extraBytes > length) {
    pos = length;
    return 0xFFFD;
  }

  for (size_t i = 0; i < extraBytes; ++i) {
    auto continuation = static_cast<unsigned char>(data[pos + 1 + i]);
    if ((continuation & 0xC0) != 0x80) {
      pos += 1 + i;
      return 0xFFFD;
    }
    codePoint = (codePoint << 6) | (continuation & 0x3F);
  }
  pos += 1 + extraBytes;

  // Reject overlong encodings, surrogate code points, and values > U+10FFFF.
  if ((extraBytes == 1 && codePoint < 0x80) ||
      (extraBytes == 2 && codePoint < 0x800) ||
      (extraBytes == 3 && codePoint < 0x10000) ||
      (codePoint >= 0xD800 && codePoint <= 0xDFFF) || codePoint > 0x10FFFF) {
    return 0xFFFD;
  }
  return codePoint;
}

/// Writes a UTF-16BE encoding of 'codePoint' to 'output'. codePoint must be a
/// valid Unicode scalar value (<= U+10FFFF, not a surrogate). Returns the
/// number of bytes written (2 for BMP, 4 for supplementary).
inline size_t writeUtf16BE(char32_t codePoint, char* output) {
  if (codePoint <= 0xFFFF) {
    output[0] = static_cast<char>((codePoint >> 8) & 0xFF);
    output[1] = static_cast<char>(codePoint & 0xFF);
    return 2;
  }
  char32_t adjusted = codePoint - 0x10000;
  char32_t highSurrogate = 0xD800 + (adjusted >> 10);
  char32_t lowSurrogate = 0xDC00 + (adjusted & 0x3FF);
  output[0] = static_cast<char>((highSurrogate >> 8) & 0xFF);
  output[1] = static_cast<char>(highSurrogate & 0xFF);
  output[2] = static_cast<char>((lowSurrogate >> 8) & 0xFF);
  output[3] = static_cast<char>(lowSurrogate & 0xFF);
  return 4;
}

/// Writes a UTF-16LE encoding of 'codePoint' to 'output'. codePoint must be a
/// valid Unicode scalar value (<= U+10FFFF, not a surrogate). Returns the
/// number of bytes written (2 for BMP, 4 for supplementary).
inline size_t writeUtf16LE(char32_t codePoint, char* output) {
  if (codePoint <= 0xFFFF) {
    output[0] = static_cast<char>(codePoint & 0xFF);
    output[1] = static_cast<char>((codePoint >> 8) & 0xFF);
    return 2;
  }
  char32_t adjusted = codePoint - 0x10000;
  char32_t highSurrogate = 0xD800 + (adjusted >> 10);
  char32_t lowSurrogate = 0xDC00 + (adjusted & 0x3FF);
  output[0] = static_cast<char>(highSurrogate & 0xFF);
  output[1] = static_cast<char>((highSurrogate >> 8) & 0xFF);
  output[2] = static_cast<char>(lowSurrogate & 0xFF);
  output[3] = static_cast<char>((lowSurrogate >> 8) & 0xFF);
  return 4;
}

} // namespace detail

/// Spark-compatible encode function. Converts a VARCHAR (UTF-8) to VARBINARY
/// using the specified charset.
/// Supported charsets (case-insensitive): US-ASCII, ISO-8859-1, UTF-8,
/// UTF-16BE, UTF-16LE, UTF-16. Common Java aliases (e.g. UTF8, ASCII, LATIN1)
/// are also accepted.
template <typename T>
struct EncodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* charset) {
    if (charset != nullptr) {
      // Resolve the constant charset once. An unsupported charset is not
      // rejected here: the error is deferred to call() so that a constant and
      // a per-row charset behave identically and both are catchable by TRY.
      charsetType_ = detail::resolveCharset(charset->data(), charset->size());
      isConstantCharset_ = true;
    }
  }

  Status call(
      out_type<Varbinary>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& charset) {
    auto type = charsetType_;
    if (!isConstantCharset_) {
      // Non-constant charset path: resolve per row.
      type = detail::resolveCharset(charset.data(), charset.size());
    }
    if (type == detail::CharsetType::kUnsupported) {
      return Status::UserError(
          "encode: unsupported charset '{}'",
          std::string(charset.data(), charset.size()));
    }

    const char* inputData = input.data();
    const size_t inputSize = input.size();

    // UTF-8: normalize the input by decoding and re-encoding. This replaces
    // any malformed UTF-8 sequences with U+FFFD (3 bytes each), matching
    // Spark/Java behavior where the input String is always valid.
    if (type == detail::CharsetType::kUtf8) {
      return encodeUtf8(result, inputData, inputSize);
    }

    // All charsets return empty bytes for empty input, matching Spark's
    // Encode.encode() which returns input.getBytes (empty) before the encoder
    // runs.
    if (inputSize == 0) {
      result.resize(0);
      return Status::OK();
    }

    // UTF-16 with BOM: emit the 2-byte BOM (0xFE 0xFF) followed by
    // big-endian data, matching Java's String.getBytes("UTF-16").
    if (type == detail::CharsetType::kUtf16) {
      return encodeUtf16WithBom(result, inputData, inputSize);
    }

    // UTF-16LE with BOM: emit the 2-byte BOM (0xFF 0xFE) followed by
    // little-endian data, matching Java's UnicodeLittle charset.
    if (type == detail::CharsetType::kUtf16LEWithBom) {
      return encodeUtf16LEWithBom(result, inputData, inputSize);
    }

    switch (type) {
      case detail::CharsetType::kUtf16BE:
        return encodeUtf16Impl(
            result, inputData, inputSize, /*bigEndian=*/true);
      case detail::CharsetType::kUtf16LE:
        return encodeUtf16Impl(
            result, inputData, inputSize, /*bigEndian=*/false);
      case detail::CharsetType::kUsAscii:
        return encodeSingleByte(result, inputData, inputSize, 0x80);
      case detail::CharsetType::kIso8859_1:
        return encodeSingleByte(result, inputData, inputSize, 0x100);
      default:
        // kUtf8, kUtf16, kUtf16LEWithBom, and kUnsupported are all handled
        // before this switch, so no other charset type can reach here.
        VELOX_UNREACHABLE(
            "encode: charset type {} should have been handled earlier",
            static_cast<int>(type));
    }
  }

 private:
  detail::CharsetType charsetType_ = detail::CharsetType::kUnsupported;
  bool isConstantCharset_ = false;

  /// Encodes to UTF-8 by decoding and re-encoding. Replaces malformed UTF-8
  /// sequences with U+FFFD (0xEF 0xBF 0xBD). For valid UTF-8 input (the common
  /// case), the output is identical to the input.
  static Status encodeUtf8(
      out_type<Varbinary>& result,
      const char* inputData,
      size_t inputSize) {
    if (inputSize == 0) {
      result.resize(0);
      return Status::OK();
    }
    // Fast path: if the entire input is ASCII, it's guaranteed valid UTF-8
    // and the output is identical to the input. Avoids per-character decode.
    bool allAscii = true;
    for (size_t i = 0; i < inputSize; ++i) {
      if (FOLLY_UNLIKELY(static_cast<unsigned char>(inputData[i]) >= 0x80)) {
        allAscii = false;
        break;
      }
    }
    if (allAscii) {
      result.resize(inputSize);
      std::memcpy(result.data(), inputData, inputSize);
      return Status::OK();
    }
    // Worst case: every byte is invalid → each becomes 3-byte U+FFFD.
    result.resize(inputSize * 3);
    char* dest = result.data();
    size_t writePos = 0;
    size_t readPos = 0;
    while (readPos < inputSize) {
      size_t startPos = readPos;
      char32_t codePoint =
          detail::decodeUtf8Codepoint(inputData, inputSize, readPos);
      if (codePoint == 0xFFFD &&
          (readPos - startPos != 3 ||
           static_cast<unsigned char>(inputData[startPos]) != 0xEF)) {
        // Invalid sequence was replaced — emit U+FFFD in UTF-8.
        dest[writePos++] = static_cast<char>(0xEF);
        dest[writePos++] = static_cast<char>(0xBF);
        dest[writePos++] = static_cast<char>(0xBD);
      } else {
        // Valid sequence — copy original bytes.
        std::memcpy(dest + writePos, inputData + startPos, readPos - startPos);
        writePos += readPos - startPos;
      }
    }
    result.resize(writePos);
    return Status::OK();
  }

  /// Encodes to a single-byte charset. Codepoints >= limit are replaced with
  /// '?' (0x3F), matching Java's CharsetEncoder REPLACE behavior.
  static Status encodeSingleByte(
      out_type<Varbinary>& result,
      const char* inputData,
      size_t inputSize,
      char32_t limit) {
    // Number of codepoints <= inputSize bytes. Over-allocate, then shrink.
    result.resize(inputSize);
    char* dest = result.data();
    size_t writePos = 0;
    size_t readPos = 0;
    while (readPos < inputSize) {
      char32_t codePoint =
          detail::decodeUtf8Codepoint(inputData, inputSize, readPos);
      dest[writePos++] = codePoint < limit ? static_cast<char>(codePoint) : '?';
    }
    result.resize(writePos);
    return Status::OK();
  }

  /// Encodes to UTF-16 big-endian with a BOM prefix (0xFE 0xFF).
  /// Caller guarantees inputSize > 0 (empty input is handled before this).
  static Status encodeUtf16WithBom(
      out_type<Varbinary>& result,
      const char* inputData,
      size_t inputSize) {
    // Worst case: 2 (BOM) + 2 bytes per input byte (all ASCII → 2 UTF-16
    // bytes each). Supplementary codepoints use 4 UTF-8 bytes → 4 UTF-16
    // bytes, so the ratio never exceeds 2x.
    const size_t maxOutput = 2 + inputSize * 2;
    result.resize(maxOutput);
    char* dest = result.data();
    dest[0] = static_cast<char>(0xFE);
    dest[1] = static_cast<char>(0xFF);
    size_t writePos = 2;
    size_t readPos = 0;
    while (readPos < inputSize) {
      char32_t codePoint =
          detail::decodeUtf8Codepoint(inputData, inputSize, readPos);
      writePos += detail::writeUtf16BE(codePoint, dest + writePos);
    }
    result.resize(writePos);
    return Status::OK();
  }

  /// Encodes to UTF-16 little-endian with a BOM prefix (0xFF 0xFE).
  /// Matches Java's UnicodeLittle charset behavior.
  /// Caller guarantees inputSize > 0 (empty input is handled before this).
  static Status encodeUtf16LEWithBom(
      out_type<Varbinary>& result,
      const char* inputData,
      size_t inputSize) {
    const size_t maxOutput = 2 + inputSize * 2;
    result.resize(maxOutput);
    char* dest = result.data();
    dest[0] = static_cast<char>(0xFF);
    dest[1] = static_cast<char>(0xFE);
    size_t writePos = 2;
    size_t readPos = 0;
    while (readPos < inputSize) {
      char32_t codePoint =
          detail::decodeUtf8Codepoint(inputData, inputSize, readPos);
      writePos += detail::writeUtf16LE(codePoint, dest + writePos);
    }
    result.resize(writePos);
    return Status::OK();
  }

  /// Encodes to UTF-16 (BE or LE) without BOM.
  static Status encodeUtf16Impl(
      out_type<Varbinary>& result,
      const char* inputData,
      size_t inputSize,
      bool bigEndian) {
    // Worst case: 2 bytes per input byte (see encodeUtf16WithBom comment).
    const size_t maxOutput = inputSize * 2;
    result.resize(maxOutput);
    char* dest = result.data();
    size_t writePos = 0;
    size_t readPos = 0;
    while (readPos < inputSize) {
      char32_t codePoint =
          detail::decodeUtf8Codepoint(inputData, inputSize, readPos);
      if (bigEndian) {
        writePos += detail::writeUtf16BE(codePoint, dest + writePos);
      } else {
        writePos += detail::writeUtf16LE(codePoint, dest + writePos);
      }
    }
    result.resize(writePos);
    return Status::OK();
  }
};

} // namespace facebook::velox::functions::sparksql
