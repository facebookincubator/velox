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
#include "velox/common/encode/Base32.h"

#include <glog/logging.h>

namespace facebook::velox::encoding {

// Encoding base to be used.
constexpr static int kBase = 32;

// Constants defining the size in bytes of binary and encoded blocks for Base32
// encoding.
// Size of a binary block in bytes (5 bytes = 40 bits)
constexpr static int kBinaryBlockByteSize = 5;
// Size of an encoded block in bytes (8 bytes = 40 bits)
constexpr static int kEncodedBlockByteSize = 8;

constexpr Base32::Charset kBase32Charset = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '2', '3', '4', '5', '6', '7'};

constexpr Base32::ReverseIndex kBase32ReverseIndexTable = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 26,  27,  28,  29,  30,  31,  255, 255, 255, 255,
    255, 255, 255, 255, 255, 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
    10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
    25,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255};

// Verify that for every entry in kBase32Charset, the corresponding entry
// in kBase32everseIndexTable is correct.
static_assert(
    checkForwardIndex(
        sizeof(kBase32Charset) - 1,
        kBase32Charset,
        kBase32ReverseIndexTable),
    "kBase32Charset has incorrect entries");

// Verify that for every entry in kBase32ReverseIndexTable, the corresponding
// entry in kBase32Charset is correct.
static_assert(
    checkReverseIndex(
        sizeof(kBase32ReverseIndexTable) - 1,
        kBase32Charset,
        kBase32ReverseIndexTable),
    "kBase32ReverseIndexTable has incorrect entries.");

// static
size_t Base32::calculateEncodedSize(size_t size, bool withPadding) {
  if (size == 0) {
    return 0;
  }

  // Calculate the output size assuming that we are including padding.
  size_t encodedSize = ((size + 4) / 5) * 8;
  if (!withPadding) {
    // If the padding was not requested, subtract the padding bytes.
    encodedSize -= (5 - (size % 5)) % 5;
  }
  return encodedSize;
}

// static
Status Base32::encode(std::string_view input, char* output) {
  return encodeImpl(input, kBase32Charset, true, output);
}

// static
template <class T>
Status Base32::encodeImpl(
    const T& input,
    const Base32::Charset& charset,
    bool includePadding,
    char* output) {
  auto inputSize = input.size();
  if (inputSize == 0) {
    return Status::OK();
  }

  auto outputPtr = output;
  auto dataIterator = input.begin();

  auto appendPadding = [includePadding](char* str, int numPadding) -> char* {
    if (includePadding) {
      for (int i = 0; i < numPadding; ++i) {
        *str++ = kPadding;
      }
    }
    return str;
  };

  // Process 5-byte (40-bit) blocks, split into 8 groups of 5 bits
  for (; inputSize > 4; inputSize -= 5) {
    uint64_t currentBlock = uint64_t(*dataIterator++) << 32;
    currentBlock |= uint8_t(*dataIterator++) << 24;
    currentBlock |= uint8_t(*dataIterator++) << 16;
    currentBlock |= uint8_t(*dataIterator++) << 8;
    currentBlock |= uint8_t(*dataIterator++);

    *outputPtr++ = charset[(currentBlock >> 35) & 0x1f];
    *outputPtr++ = charset[(currentBlock >> 30) & 0x1f];
    *outputPtr++ = charset[(currentBlock >> 25) & 0x1f];
    *outputPtr++ = charset[(currentBlock >> 20) & 0x1f];
    *outputPtr++ = charset[(currentBlock >> 15) & 0x1f];
    *outputPtr++ = charset[(currentBlock >> 10) & 0x1f];
    *outputPtr++ = charset[(currentBlock >> 5) & 0x1f];
    *outputPtr++ = charset[currentBlock & 0x1f];
  }

  // Handle remaining bytes (1 to 4 bytes)
  if (inputSize > 0) {
    uint64_t currentBlock = uint64_t(*dataIterator++) << 32;
    *outputPtr++ = charset[(currentBlock >> 35) & 0x1f];

    if (inputSize > 3) {
      currentBlock |= uint8_t(*dataIterator++) << 24;
      currentBlock |= uint8_t(*dataIterator++) << 16;
      currentBlock |= uint8_t(*dataIterator++) << 8;

      *outputPtr++ = charset[(currentBlock >> 30) & 0x1f];
      *outputPtr++ = charset[(currentBlock >> 25) & 0x1f];
      *outputPtr++ = charset[(currentBlock >> 20) & 0x1f];
      *outputPtr++ = charset[(currentBlock >> 15) & 0x1f];
      *outputPtr++ = charset[(currentBlock >> 10) & 0x1f];
      *outputPtr++ = charset[(currentBlock >> 5) & 0x1f];
      appendPadding(outputPtr, 1);

    } else if (inputSize > 2) {
      currentBlock |= uint8_t(*dataIterator++) << 24;
      currentBlock |= uint8_t(*dataIterator++) << 16;

      *outputPtr++ = charset[(currentBlock >> 30) & 0x1f];
      *outputPtr++ = charset[(currentBlock >> 25) & 0x1f];
      *outputPtr++ = charset[(currentBlock >> 20) & 0x1f];
      *outputPtr++ = charset[(currentBlock >> 15) & 0x1f];
      appendPadding(outputPtr, 3);

    } else if (inputSize > 1) {
      currentBlock |= uint8_t(*dataIterator++) << 24;

      *outputPtr++ = charset[(currentBlock >> 30) & 0x1f];
      *outputPtr++ = charset[(currentBlock >> 25) & 0x1f];
      *outputPtr++ = charset[(currentBlock >> 20) & 0x1f];
      appendPadding(outputPtr, 4);

    } else {
      *outputPtr++ = charset[(currentBlock >> 30) & 0x1f];
      appendPadding(outputPtr, 6);
    }
  }

  return Status::OK();
}

} // namespace facebook::velox::encoding
