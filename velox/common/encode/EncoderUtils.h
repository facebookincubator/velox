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

#include <string>
#include "velox/common/base/Status.h"

namespace facebook::velox::encoding {

// Padding character used in encoding.
const static char kPadding = '=';

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

// Validate the character in charset with ReverseIndex table
template <typename Charset, typename ReverseIndex>
constexpr bool checkForwardIndex(
    uint8_t index,
    const Charset& charset,
    const ReverseIndex& reverseIndex) {
  return (reverseIndex[static_cast<uint8_t>(charset[index])] == index) &&
      (index > 0 ? checkForwardIndex(index - 1, charset, reverseIndex) : true);
}

// Searches for a character within a charset up to a certain index.
template <typename Charset>
constexpr bool findCharacterInCharset(
    const Charset& charset,
    uint8_t index,
    const char targetChar) {
  return index < charset.size() &&
      ((charset[index] == targetChar) ||
       findCharacterInCharset(charset, index + 1, targetChar));
}

// Checks the consistency of a reverse index mapping for a given character set.
template <typename Charset, typename ReverseIndex>
constexpr bool checkReverseIndex(
    uint8_t index,
    const Charset& charset,
    const ReverseIndex& reverseIndex) {
  return (reverseIndex[index] == 255
              ? !findCharacterInCharset(charset, 0, static_cast<char>(index))
              : (charset[reverseIndex[index]] == index)) &&
      (index > 0 ? checkReverseIndex(index - 1, charset, reverseIndex) : true);
}

template <typename ReverseIndexType>
uint8_t reverseLookup(
    char encodedChar,
    const ReverseIndexType& reverseIndex,
    Status& status,
    uint8_t kBase) {
  auto curr = reverseIndex[static_cast<uint8_t>(encodedChar)];
  if (curr >= kBase) {
    status =
        Status::UserError("invalid input string: contains invalid characters.");
    return 0; // Return 0 or any other error code indicating failure
  }
  return curr;
}

// Returns the actual size of the decoded data. Will also remove the padding
// length from the 'inputSize'.
static Status calculateDecodedSize(
    std::string_view input,
    size_t& inputSize,
    size_t& decodedSize,
    const int binaryBlockByteSize,
    const int encodedBlockByteSize) {
  if (inputSize == 0) {
    decodedSize = 0;
    return Status::OK();
  }

  // Check if the input string is padded
  if (isPadded(input)) {
    // If padded, ensure that the string length is a multiple of the encoded
    // block size
    if (inputSize % encodedBlockByteSize != 0) {
    }

    decodedSize = (inputSize * binaryBlockByteSize) / encodedBlockByteSize;
    auto paddingCount = numPadding(input);
    inputSize -= paddingCount;

    // Adjust the needed size by deducting the bytes corresponding to the
    // padding from the calculated size.
    decodedSize -=
        ((paddingCount * binaryBlockByteSize) + (encodedBlockByteSize - 1)) /
        encodedBlockByteSize;
  } else {
    // If not padded, calculate extra bytes, if any
    auto extraBytes = inputSize % encodedBlockByteSize;
    decodedSize = (inputSize / encodedBlockByteSize) * binaryBlockByteSize;
    // Adjust the needed size for extra bytes, if present
    if (extraBytes) {
      if (extraBytes == 1) {
      }
      decodedSize += (extraBytes * binaryBlockByteSize) / encodedBlockByteSize;
    }
  }
  return Status::OK();
}

// Calculates the encoded size based on input size.
static size_t calculateEncodedSize(
    size_t inputSize,
    bool includePadding,
    const int binaryBlockByteSize,
    const int encodedBlockByteSize) {
  if (inputSize == 0) {
    return 0;
  }

  // Calculate the output size assuming that we are including padding.
  size_t encodedSize =
      ((inputSize + binaryBlockByteSize - 1) / binaryBlockByteSize) *
      encodedBlockByteSize;

  if (!includePadding) {
    // If the padding was not requested, subtract the padding bytes.
    size_t remainder = inputSize % binaryBlockByteSize;
    if (remainder != 0) {
      encodedSize -= (binaryBlockByteSize - remainder);
    }
  }
  return encodedSize;
}

} // namespace facebook::velox::encoding
