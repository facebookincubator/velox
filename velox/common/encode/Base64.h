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

#include <folly/io/IOBuf.h>

#include <array>
#include <string>

#include "velox/common/base/GTestMacros.h"
#include "velox/common/base/Status.h"

namespace facebook::velox::encoding {

class Base64 {
 public:
  static const size_t kCharsetSize = 64;
  static const size_t kReverseIndexSize = 256;

  /// Character set used for Base64 encoding.
  /// Contains specific characters that form the encoding scheme.
  using Charset = std::array<char, kCharsetSize>;

  /// Reverse lookup table for decoding.
  /// Maps each possible encoded character to its corresponding numeric value
  /// within the encoding base.
  using ReverseIndex = std::array<uint8_t, kReverseIndexSize>;

  /// Encodes the specified number of characters from the 'input'.
  static std::string encode(
      std::string_view input,
      bool includePadding = false);

  /// Encodes the specified IOBuf data.
  static std::string encode(const folly::IOBuf* inputBuffer);

  /// Encodes the specified number of characters from the 'input' and writes the
  /// result to the 'output'.
  static void encode(std::string_view input, std::string& output);

  /// Encodes the specified text using URL encoding.
  static std::string encodeUrl(
      std::string_view input,
      bool includePadding = false);

  /// Encodes the specified IOBuf data using URL encoding.
  static std::string encodeUrl(const folly::IOBuf* inputBuffer);

  /// Encodes the specified number of characters from the 'input' and writes the
  /// result to the 'output' using URL encoding.
  static void encodeUrl(std::string_view input, std::string& output);

  /// Decodes the input Base64 encoded string.
  static std::string decode(std::string_view input);

  /// Decodes the specified number of characters from the 'input' and writes the
  /// result to the 'output'.
  static Status decode(std::string_view input, std::string& output);

  /// Decodes the input Base64 URL encoded string.
  static std::string decodeUrl(std::string_view input);

  /// Decodes the specified number of characters from the 'input' using URL
  /// encoding and writes the result to the 'output'
  static Status decodeUrl(std::string_view input, std::string& output);

  /// Decodes a Base64 MIME‐mode buffer back to binary.
  /// Skips any non-Base64 chars (e.g. CR/LF).
  static Status decodeMime(std::string_view input, std::string& output);

  /// Encodes the input buffer into Base64 MIME format.
  /// Inserts a CRLF every kMaxLineLength output characters.
  static void encodeMime(std::string_view input, std::string& output);

  /// Calculates the decoded binary size of a MIME‐mode Base64 input,
  /// accounting for padding and ignoring whitespace.
  static Expected<size_t> calculateMimeDecodedSize(std::string_view input);

  /// Computes the exact output length for MIME‐mode Base64 encoding,
  /// including required CRLF line breaks.
  static size_t calculateMimeEncodedSize(size_t inputSize);

 private:
  // Padding character used in encoding.
  static const char kPadding = '=';

  // Soft Line breaks used in mime encoding as defined in RFC 2045, section 6.8:
  // https://www.rfc-editor.org/rfc/rfc2045#section-6.8
  inline static const std::string kNewline{"\r\n"};
  static const size_t kMaxLineLength = 76;

  // Checks if the input Base64 string is padded.
  static inline bool isPadded(std::string_view input) {
    return (!input.empty() && input.back() == kPadding);
  }

  // Counts the number of padding characters in encoded input.
  static inline size_t numPadding(std::string_view input) {
    size_t numPadding{0};
    while (!input.empty() && input.back() == kPadding) {
      numPadding++;
      input.remove_suffix(1);
    }
    return numPadding;
  }

  // Reverse lookup helper function to get the original index of a Base64
  // character.
  static Expected<uint8_t> base64ReverseLookup(
      char encodedChar,
      const ReverseIndex& reverseIndex);

  // Encodes the specified data using the provided charset.
  template <class T>
  static std::string
  encodeImpl(const T& input, const Charset& charset, bool includePadding);

  // Encodes the specified data using the provided charset.
  template <class T>
  static void encodeImpl(
      const T& input,
      const Charset& charset,
      bool includePadding,
      std::string& output);

  // Decodes the specified data using the provided reverse lookup table.
  static Expected<size_t> decodeImpl(
      std::string_view input,
      std::string& output,
      const ReverseIndex& reverseIndex);

  // Calculates the encoded size based on input 'inputSize'.
  static size_t calculateEncodedSize(size_t inputSize, bool withPadding = true);

  // Calculates the decoded size based on encoded input and adjusts the input
  // size for padding.
  static Expected<size_t> calculateDecodedSize(std::string_view input);

  VELOX_FRIEND_TEST(Base64Test, checksPadding);
  VELOX_FRIEND_TEST(Base64Test, countsPaddingCorrectly);
  VELOX_FRIEND_TEST(Base64Test, calculateDecodedSizeProperSize);
};

} // namespace facebook::velox::encoding
