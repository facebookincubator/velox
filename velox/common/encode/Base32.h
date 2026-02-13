/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <array>
#include <string>

#include "velox/common/base/Status.h"

namespace facebook::velox::encoding {

class Base32 {
 public:
  static const size_t kCharsetSize = 32;
  static const size_t kReverseIndexSize = 256;

  /// Character set used for Base32 encoding.
  /// Contains specific characters that form the encoding scheme.
  using Charset = std::array<char, kCharsetSize>;

  /// Reverse lookup table for decoding.
  /// Maps each possible encoded character to its corresponding numeric value
  /// within the encoding base.
  using ReverseIndex = std::array<uint8_t, kReverseIndexSize>;

  /// Decodes the specified number of characters from the 'input' and writes the
  /// result to the 'outputBuffer'.
  static Status decode(
      const char* input,
      size_t inputSize,
      char* outputBuffer,
      size_t outputSize);

  /// Calculates the decoded size based on encoded input.
  static folly::Expected<size_t, Status> calculateDecodedSize(
      const char* input,
      const size_t inputSize);

  // Padding character used in encoding.
  static const char kPadding = '=';

  // Constants defining the size in bytes of binary and encoded blocks for
  // Base32 encoding. Size of a binary block in bytes (5 bytes = 40 bits)
  static const int kBinaryBlockByteSize = 5;
  // Size of an encoded block in bytes (8 bytes = 40 bits)
  static const int kEncodedBlockByteSize = 8;

 private:
  // Reverse lookup helper function to get the original index of a Base32
  // character.
  static folly::Expected<uint8_t, Status> base32ReverseLookup(
      char encodedChar,
      const ReverseIndex& reverseIndex);

  // Decodes the specified data using the provided reverse lookup table.
  static folly::Expected<size_t, Status> decodeImpl(
      const char* input,
      size_t inputSize,
      char* outputBuffer,
      size_t outputSize,
      const ReverseIndex& reverseIndex);
};

} // namespace facebook::velox::encoding
