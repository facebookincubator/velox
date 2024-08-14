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

#include <array>
#include <string>

#include <folly/Range.h>
#include <folly/io/IOBuf.h>

#include "velox/common/base/GTestMacros.h"
#include "velox/common/base/Status.h"

namespace facebook::velox::encoding {

class Base64 {
 public:
  static const size_t kCharsetSize = 64;
  static const size_t kReverseIndexSize = 256;

  /// Character set used for encoding purposes.
  /// Contains specific characters that form the encoding scheme.
  using Charset = std::array<char, kCharsetSize>;

  /// Reverse lookup table for decoding purposes.
  /// Maps each possible encoded character to its corresponding numeric value
  /// within the encoding base.
  using ReverseIndex = std::array<uint8_t, kReverseIndexSize>;

  /// Encodes the specified number of characters from the 'input'.
  static std::string encode(std::string_view input, size_t len);

  /// Encodes the specified text.
  static std::string encode(std::string_view text);

  /// Encodes the specified IOBuf input.
  static std::string encode(const folly::IOBuf* input);

  /// Returns encoded size for the input of the specified size.
  static size_t calculateEncodedSize(
      size_t inputSize,
      bool includePadding = true);

  /// Encodes the specified number of characters from the 'input' and writes the
  /// result to the 'output'. The output must have enough space, e.g., as
  /// returned by calculateEncodedSize().
  static Status encode(std::string_view input, char* output);

  /// Decodes the specified encoded text.
  static std::string decode(std::string_view encoded);

  /// Returns the actual size of the decoded data. Will also remove the padding
  /// length from the 'inputSize'.
  static Status calculateDecodedSize(
      std::string_view input,
      size_t& inputSize,
      size_t& decodedSize);

  /// Decodes the specified number of characters from the 'input' and writes the
  /// result to the 'output'. The output must have enough space, e.g., as
  /// returned by calculateDecodedSize().
  static void decode(std::string_view input, size_t inputSize, char* output);

  static void decode(std::string_view input, std::string& output);

  /// Encodes the specified number of characters from the 'input' and writes the
  /// result to the 'output' using URL encoding. The output must have enough
  /// space as returned by calculateEncodedSize().
  static Status encodeUrl(std::string_view input, char* output);

  /// Encodes the specified IOBuf input using URL encoding.
  static std::string encodeUrl(const folly::IOBuf* input);

  /// Encodes the specified text using URL encoding.
  static std::string encodeUrl(std::string_view input);

  /// Decodes the specified URL encoded input and writes the result to the
  /// 'output'.
  static void decodeUrl(std::string_view input, std::string& output);

  /// Decodes the specified URL encoded text.
  static std::string decodeUrl(std::string_view input);

  /// Decodes the specified number of characters from the 'input' and writes the
  /// result to the 'output'.
  static Status decode(
      std::string_view input,
      size_t inputSize,
      char* output,
      size_t outputSize);

  /// Decodes the specified number of characters from the 'input' using URL
  /// encoding and writes the result to the 'output'.
  static Status decodeUrl(
      std::string_view input,
      size_t inputSize,
      char* output,
      size_t outputSize);

 private:
  // Padding character used in encoding.
  static const char kPadding = '=';

  // Checks if there is padding in encoded input.
  static inline bool isPadded(std::string_view input, size_t inputSize) {
    return (inputSize > 0 && input[inputSize - 1] == kPadding);
  }

  // Counts the number of padding characters in encoded input.
  static inline size_t numPadding(std::string_view input, size_t inputSize) {
    size_t padding = 0;
    while (inputSize > 0 && input[inputSize - 1] == kPadding) {
      padding++;
      inputSize--;
    }
    return padding;
  }

  // Performs a reverse lookup in the reverse index to retrieve the original
  // index of a character in the base.
  static uint8_t base64ReverseLookup(
      char character,
      const ReverseIndex& reverseIndex);

  // Encodes the specified input using the provided charset.
  template <class T>
  static std::string
  encodeImpl(const T& input, const Charset& charset, bool includePadding);

  // Encodes the specified input using the provided charset.
  template <class T>
  static Status encodeImpl(
      const T& input,
      const Charset& charset,
      bool includePadding,
      char* output);

  // Decodes the specified input using the provided reverse lookup table.
  static Status decodeImpl(
      std::string_view input,
      size_t inputSize,
      char* output,
      size_t outputSize,
      const ReverseIndex& reverseIndex);

  VELOX_FRIEND_TEST(Base64Test, isPadded);
  VELOX_FRIEND_TEST(Base64Test, numPadding);
};

} // namespace facebook::velox::encoding
