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

// Verify that for each 32 entries in kBase32Charset, the corresponding entry
// in kBase32ReverseIndexTable is correct.
static_assert(
    checkForwardIndex(
        sizeof(kBase32Charset) / 2 - 1,
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
Status Base32::encode(std::string_view input, std::string& output) {
  return encodeImpl(input, true, output);
}

// static
template <class T>
Status
Base32::encodeImpl(const T& input, bool includePadding, std::string& output) {
  auto inputSize = input.size();
  if (inputSize == 0) {
    output.clear();
    return Status::OK();
  }

  // Calculate the output size and resize the string beforehand
  size_t outputSize = calculateEncodedSize(
      inputSize, includePadding, kBinaryBlockByteSize, kEncodedBlockByteSize);
  output.resize(outputSize);

  // Use a pointer to write into the pre-allocated buffer
  auto outputPointer = output.data();
  auto inputIterator = input.begin();

  // Process 5-byte (40-bit) blocks, split into 8 groups of 5 bits
  for (; inputSize > 4; inputSize -= 5) {
    uint64_t currentBlock = static_cast<uint64_t>(*inputIterator++) << 32;
    currentBlock |= static_cast<uint64_t>(*inputIterator++) << 24;
    currentBlock |= static_cast<uint64_t>(*inputIterator++) << 16;
    currentBlock |= static_cast<uint64_t>(*inputIterator++) << 8;
    currentBlock |= static_cast<uint64_t>(*inputIterator++);

    *outputPointer++ = kBase32Charset[(currentBlock >> 35) & 0x1f];
    *outputPointer++ = kBase32Charset[(currentBlock >> 30) & 0x1f];
    *outputPointer++ = kBase32Charset[(currentBlock >> 25) & 0x1f];
    *outputPointer++ = kBase32Charset[(currentBlock >> 20) & 0x1f];
    *outputPointer++ = kBase32Charset[(currentBlock >> 15) & 0x1f];
    *outputPointer++ = kBase32Charset[(currentBlock >> 10) & 0x1f];
    *outputPointer++ = kBase32Charset[(currentBlock >> 5) & 0x1f];
    *outputPointer++ = kBase32Charset[currentBlock & 0x1f];
  }

  // Handle remaining bytes (1 to 4 bytes)
  if (inputSize > 0) {
    uint64_t currentBlock = static_cast<uint64_t>(*inputIterator++) << 32;
    *outputPointer++ = kBase32Charset[(currentBlock >> 35) & 0x1f];

    if (inputSize > 3) {
      currentBlock |= static_cast<uint64_t>(*inputIterator++) << 24;
      currentBlock |= static_cast<uint64_t>(*inputIterator++) << 16;
      currentBlock |= static_cast<uint64_t>(*inputIterator++) << 8;

      *outputPointer++ = kBase32Charset[(currentBlock >> 30) & 0x1f];
      *outputPointer++ = kBase32Charset[(currentBlock >> 25) & 0x1f];
      *outputPointer++ = kBase32Charset[(currentBlock >> 20) & 0x1f];
      *outputPointer++ = kBase32Charset[(currentBlock >> 15) & 0x1f];
      *outputPointer++ = kBase32Charset[(currentBlock >> 10) & 0x1f];
      *outputPointer++ = kBase32Charset[(currentBlock >> 5) & 0x1f];
      if (includePadding) {
        *outputPointer++ = kPadding;
      }
    } else if (inputSize > 2) {
      currentBlock |= static_cast<uint64_t>(*inputIterator++) << 24;
      currentBlock |= static_cast<uint64_t>(*inputIterator++) << 16;

      *outputPointer++ = kBase32Charset[(currentBlock >> 30) & 0x1f];
      *outputPointer++ = kBase32Charset[(currentBlock >> 25) & 0x1f];
      *outputPointer++ = kBase32Charset[(currentBlock >> 20) & 0x1f];
      *outputPointer++ = kBase32Charset[(currentBlock >> 15) & 0x1f];
      if (includePadding) {
        *outputPointer++ = kPadding;
        *outputPointer++ = kPadding;
        *outputPointer++ = kPadding;
      }
    } else if (inputSize > 1) {
      currentBlock |= static_cast<uint64_t>(*inputIterator++) << 24;

      *outputPointer++ = kBase32Charset[(currentBlock >> 30) & 0x1f];
      *outputPointer++ = kBase32Charset[(currentBlock >> 25) & 0x1f];
      *outputPointer++ = kBase32Charset[(currentBlock >> 20) & 0x1f];
      if (includePadding) {
        *outputPointer++ = kPadding;
        *outputPointer++ = kPadding;
        *outputPointer++ = kPadding;
        *outputPointer++ = kPadding;
      }
    } else {
      *outputPointer++ = kBase32Charset[(currentBlock >> 30) & 0x1f];
      if (includePadding) {
        *outputPointer++ = kPadding;
        *outputPointer++ = kPadding;
        *outputPointer++ = kPadding;
        *outputPointer++ = kPadding;
        *outputPointer++ = kPadding;
        *outputPointer++ = kPadding;
      }
    }
  }

  return Status::OK();
}

// static
uint8_t Base32::base32ReverseLookup(char encodedChar, Status& status) {
  return reverseLookup(
      encodedChar, kBase32ReverseIndexTable, status, kCharsetSize);
}

// static
Status Base32::decode(std::string_view input, std::string& output) {
  return decodeImpl(input, output);
}

// static
Status Base32::decodeImpl(std::string_view input, std::string& output) {
  size_t inputSize = input.size();

  // If input is empty, clear output and return OK status.
  if (inputSize == 0) {
    output.clear();
    return Status::OK();
  }

  // Calculate the decoded size based on the input size.
  size_t decodedSize;
  auto status = calculateDecodedSize(
      input,
      inputSize,
      decodedSize,
      kBinaryBlockByteSize,
      kEncodedBlockByteSize);
  if (!status.ok()) {
    return status;
  }

  // Resize the output to accommodate the decoded data.
  output.resize(decodedSize);

  const char* inputPtr = input.data();
  char* outputPtr = output.data();
  Status lookupStatus;

  // Process full blocks of 8 characters
  size_t fullBlockCount = inputSize / 8;
  for (size_t i = 0; i < fullBlockCount; ++i) {
    uint64_t inputBlock = 0;

    // Decode 8 characters into a 40-bit block
    for (int shift = 35, j = 0; j < 8; ++j, shift -= 5) {
      uint64_t value = base32ReverseLookup(inputPtr[j], lookupStatus);
      if (!lookupStatus.ok()) {
        return lookupStatus;
      }
      inputBlock |= (value << shift);
    }

    // Write the decoded block to the output
    outputPtr[0] = static_cast<char>((inputBlock >> 32) & 0xFF);
    outputPtr[1] = static_cast<char>((inputBlock >> 24) & 0xFF);
    outputPtr[2] = static_cast<char>((inputBlock >> 16) & 0xFF);
    outputPtr[3] = static_cast<char>((inputBlock >> 8) & 0xFF);
    outputPtr[4] = static_cast<char>(inputBlock & 0xFF);

    inputPtr += 8;
    outputPtr += 5;
  }

  // Handle remaining characters (2, 4, 5, 7)
  size_t remaining = inputSize % 8;
  if (remaining >= 2) {
    uint64_t inputBlock = 0;

    // Decode the first two characters
    inputBlock |=
        (static_cast<uint64_t>(base32ReverseLookup(inputPtr[0], lookupStatus))
         << 35);
    inputBlock |=
        (static_cast<uint64_t>(base32ReverseLookup(inputPtr[1], lookupStatus))
         << 30);

    if (!lookupStatus.ok()) {
      return lookupStatus;
    }
    outputPtr[0] = static_cast<char>((inputBlock >> 32) & 0xFF);

    if (remaining > 2) {
      // Decode the next two characters
      inputBlock |= (base32ReverseLookup(inputPtr[2], lookupStatus) << 25);
      inputBlock |= (base32ReverseLookup(inputPtr[3], lookupStatus) << 20);

      if (!lookupStatus.ok()) {
        return lookupStatus;
      }
      outputPtr[1] = static_cast<char>((inputBlock >> 24) & 0xFF);

      if (remaining > 4) {
        // Decode the next character
        inputBlock |= (base32ReverseLookup(inputPtr[4], lookupStatus) << 15);

        if (!lookupStatus.ok()) {
          return lookupStatus;
        }
        outputPtr[2] = static_cast<char>((inputBlock >> 16) & 0xFF);

        if (remaining > 5) {
          // Decode the next two characters
          inputBlock |= (base32ReverseLookup(inputPtr[5], lookupStatus) << 10);
          inputBlock |= (base32ReverseLookup(inputPtr[6], lookupStatus) << 5);

          if (!lookupStatus.ok()) {
            return lookupStatus;
          }
          outputPtr[3] = static_cast<char>((inputBlock >> 8) & 0xFF);

          if (remaining > 7) {
            // Decode the last character
            inputBlock |= base32ReverseLookup(inputPtr[7], lookupStatus);

            if (!lookupStatus.ok()) {
              return lookupStatus;
            }
            outputPtr[4] = static_cast<char>(inputBlock & 0xFF);
          }
        }
      }
    }
  }

  // Return status
  return Status::OK();
}

} // namespace facebook::velox::encoding
