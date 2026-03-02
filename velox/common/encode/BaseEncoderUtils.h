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

#include <cstdint>
#include <string_view>

#include "velox/common/base/Status.h"

namespace facebook::velox::encoding {

/// Describes the binary and encoded block sizes for a base encoding codec.
struct CodecBlockSizes {
  int binaryBlockByteSize;
  int encodedBlockByteSize;
};

static constexpr CodecBlockSizes kBase64BlockSizes{3, 4};
static constexpr CodecBlockSizes kBase32BlockSizes{5, 8};

/// Shared utility functions for base encoding schemes (e.g. Base64, Base32).
/// Provides common operations for padding detection, charset validation,
/// reverse lookup, and encoded size calculation.
class BaseEncoderUtils {
 public:
  /// Padding character used in base encoding schemes.
  static constexpr char kPadding = '=';

  /// Sentinel value in reverse index tables indicating an invalid character.
  static constexpr uint8_t kInvalidBase = 255;

  /// Checks if the encoded input ends with padding character(s).
  static bool isPadded(std::string_view input) {
    return !input.empty() && input.back() == kPadding;
  }

  /// Counts the number of trailing padding characters in encoded input.
  static size_t numPadding(std::string_view input) {
    size_t count{0};
    for (auto it = input.rbegin(); it != input.rend() && *it == kPadding;
         ++it) {
      ++count;
    }
    return count;
  }

  /// Validates that each character in the charset maps correctly through the
  /// reverse index table back to its original index.
  /// @param index The current index to validate (iterates down to 0).
  /// @param charset The character set used for encoding.
  /// @param reverseIndex The reverse lookup table mapping characters to
  /// indices.
  template <typename Charset, typename ReverseIndex>
  static constexpr bool checkForwardIndex(
      uint8_t index,
      const Charset& charset,
      const ReverseIndex& reverseIndex) {
    return (reverseIndex[static_cast<uint8_t>(charset[index])] == index) &&
        (index > 0 ? checkForwardIndex(index - 1, charset, reverseIndex)
                   : true);
  }

  /// Searches for a target character within a charset starting from the given
  /// index.
  /// @param charset The character set to search.
  /// @param index The starting index for the search.
  /// @param targetChar The character to find.
  template <typename Charset>
  static constexpr bool findCharacterInCharset(
      const Charset& charset,
      uint8_t index,
      const char targetChar) {
    return index < charset.size() &&
        ((charset[index] == targetChar) ||
         findCharacterInCharset(charset, index + 1, targetChar));
  }

  /// Checks the consistency of a reverse index mapping for a given character
  /// set. For each byte value, verifies that either:
  /// - The reverse index is kInvalidBase and the character is not in the
  ///   charset, or
  /// - The charset at the reverse index position equals the byte value.
  /// @param index The current byte value to validate (iterates down to 0).
  /// @param charset The character set used for encoding.
  /// @param reverseIndex The reverse lookup table to validate.
  template <typename Charset, typename ReverseIndex>
  static constexpr bool checkReverseIndex(
      uint8_t index,
      const Charset& charset,
      const ReverseIndex& reverseIndex) {
    return (reverseIndex[index] == kInvalidBase
                ? !findCharacterInCharset(charset, 0, static_cast<char>(index))
                : (charset[reverseIndex[index]] == index)) &&
        (index > 0 ? checkReverseIndex(index - 1, charset, reverseIndex)
                   : true);
  }

  /// Performs reverse lookup of an encoded character to its numeric value.
  /// @param encodedChar The encoded character to look up.
  /// @param reverseIndex The reverse index table for the encoding scheme.
  /// @param base The base of the encoding (e.g. 64 for Base64, 32 for
  ///   Base32). Characters mapping to values >= base are considered invalid.
  template <typename ReverseIndex>
  static Expected<uint8_t> reverseLookup(
      char encodedChar,
      const ReverseIndex& reverseIndex,
      uint8_t base) {
    auto value = reverseIndex[static_cast<uint8_t>(encodedChar)];
    if (value >= base) {
      return folly::makeUnexpected(
          Status::UserError(
              "decode() - invalid input string: invalid character '{}'",
              encodedChar));
    }
    return value;
  }

  /// Calculates the decoded output size from encoded input, accounting for
  /// padding. Also strips padding from 'inputSize' so that the caller can
  /// iterate over only the non-padding characters.
  /// @param input The encoded input string (used to detect trailing padding).
  /// @param inputSize The length of the encoded input. Modified on return to
  ///   exclude padding characters.
  /// @param blockSizes The binary and encoded block sizes for the codec.
  static Expected<size_t> calculateDecodedSize(
      std::string_view input,
      size_t& inputSize,
      const CodecBlockSizes& blockSizes);

  /// Calculates the encoded output size based on input size and block sizes.
  /// @param inputSize Size of the input data in bytes.
  /// @param includePadding Whether to include padding in the output.
  /// @param blockSizes The binary and encoded block sizes for the codec.
  static size_t calculateEncodedSize(
      size_t inputSize,
      bool includePadding,
      const CodecBlockSizes& blockSizes);
};

} // namespace facebook::velox::encoding
