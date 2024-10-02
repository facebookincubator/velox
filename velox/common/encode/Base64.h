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

namespace facebook::velox::encoding {

class Base64 {
 public:
  static const size_t kCharsetSize = 64;
  static const size_t kReverseIndexSize = 256;

  /// Character set used for Base64 encoding.
  using Charset = std::array<char, kCharsetSize>;

  /// Reverse lookup table for decoding.
  using ReverseIndex = std::array<uint8_t, kReverseIndexSize>;

  /// Padding character used in encoding.
  static const char kPadding = '=';

  // Encoding Functions
  /// Encodes the input data using Base64 encoding.
  static std::string encode(const char* input, size_t inputSize);
  static std::string encode(folly::StringPiece text);
  static std::string encode(const folly::IOBuf* inputBuffer);
  static void encode(const char* input, size_t inputSize, char* outputBuffer);

  /// Encodes the input data using Base64 URL encoding.
  static std::string encodeUrl(const char* input, size_t inputSize);
  static std::string encodeUrl(folly::StringPiece text);
  static std::string encodeUrl(const folly::IOBuf* inputBuffer);
  static void
  encodeUrl(const char* input, size_t inputSize, char* outputBuffer);

  // Decoding Functions
  /// Decodes the input Base64 encoded string.
  static std::string decode(folly::StringPiece encodedText);
  static void decode(
      const std::pair<const char*, int32_t>& payload,
      std::string& output);
  static void decode(const char* input, size_t inputSize, char* outputBuffer);
  static size_t decode(
      const char* input,
      size_t inputSize,
      char* outputBuffer,
      size_t outputSize);

  /// Decodes the input Base64 URL encoded string.
  static std::string decodeUrl(folly::StringPiece encodedText);
  static void decodeUrl(
      const std::pair<const char*, int32_t>& payload,
      std::string& output);
  static void decodeUrl(
      const char* input,
      size_t inputSize,
      char* outputBuffer,
      size_t outputSize);

  // Helper Functions
  /// Calculates the encoded size based on input size.
  static size_t calculateEncodedSize(size_t inputSize, bool withPadding = true);

  /// Calculates the decoded size based on encoded input and adjusts the input
  /// size for padding.
  static size_t calculateDecodedSize(const char* input, size_t& inputSize);

 private:
  // Checks if the input Base64 string is padded.
  static inline bool isPadded(const char* input, size_t inputSize) {
    return (inputSize > 0 && input[inputSize - 1] == kPadding);
  }

  // Counts the number of padding characters in encoded input.
  static inline size_t numPadding(const char* input, size_t inputSize) {
    size_t numPadding{0};
    while (inputSize > 0 && input[inputSize - 1] == kPadding) {
      numPadding++;
      inputSize--;
    }
    return numPadding;
  }

  // Reverse lookup helper function to get the original index of a Base64
  // character.
  static uint8_t base64ReverseLookup(
      char encodedChar,
      const ReverseIndex& reverseIndex);

  template <class T>
  static std::string
  encodeImpl(const T& input, const Charset& charset, bool includePadding);

  template <class T>
  static void encodeImpl(
      const T& input,
      const Charset& charset,
      bool includePadding,
      char* outputBuffer);

  static size_t decodeImpl(
      const char* input,
      size_t inputSize,
      char* outputBuffer,
      size_t outputSize,
      const ReverseIndex& reverseIndex);

  VELOX_FRIEND_TEST(Base64Test, checksPadding);
  VELOX_FRIEND_TEST(Base64Test, countsPaddingCorrectly);
};

} // namespace facebook::velox::encoding
