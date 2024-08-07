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

#include <folly/Range.h>
#include <folly/io/IOBuf.h>
#include <array>
#include <string>
#include "velox/common/base/GTestMacros.h"
#include "velox/common/base/Status.h"
#include "velox/common/encode/EncoderUtils.h"

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
  static std::string encode(folly::StringPiece input);
  static std::string encode(const folly::IOBuf* inputBuffer);
  static Status encode(std::string_view input, std::string& outputBuffer);

  /// Encodes the input data using Base64 URL encoding.
  static std::string encodeUrl(const char* input, size_t inputSize);
  static std::string encodeUrl(folly::StringPiece text);
  static std::string encodeUrl(const folly::IOBuf* inputBuffer);
  static Status encodeUrl(std::string_view input, std::string& output);

  // Decoding Functions
  /// Decodes the input Base64 encoded string.
  static std::string decode(folly::StringPiece encodedText);
  static void decode(
      const std::pair<const char*, int32_t>& payload,
      std::string& output);
  static void decode(const char* input, size_t inputSize, char* outputBuffer);
  static Status decode(std::string_view input, std::string& output);

  /// Decodes the input Base64 URL encoded string.
  static std::string decodeUrl(folly::StringPiece encodedText);
  static void decodeUrl(
      const std::pair<const char*, int32_t>& payload,
      std::string& output);
  static Status decodeUrl(std::string_view input, std::string& output);

 private:
  // Checks if the input Base64 string is padded.
  static inline bool isPadded(std::string_view input) {
    size_t inputSize{input.size()};
    return (inputSize > 0 && input[inputSize - 1] == kPadding);
  }

  // Counts the number of padding characters in encoded input.
  static inline size_t numPadding(std::string_view input) {
    size_t numPadding{0};
    size_t inputSize{input.size()};
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
      const ReverseIndex& reverseIndex,
      Status& status);

  template <class T>
  static std::string
  encodeImpl(const T& input, const Charset& charset, bool includePadding);

  template <class T>
  static Status encodeImpl(
      const T& input,
      const Charset& charset,
      bool includePadding,
      std::string& output);

  static Status decodeImpl(
      std::string_view input,
      std::string& output,
      const ReverseIndex& reverseIndex);

  VELOX_FRIEND_TEST(Base64Test, isPadded);
  VELOX_FRIEND_TEST(Base64Test, numPadding);
  VELOX_FRIEND_TEST(Base64Test, calculateDecodedSize);
};

} // namespace facebook::velox::encoding
