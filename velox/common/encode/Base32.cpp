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

#include <folly/Expected.h>
#include <cctype>
#include <cstdint>
#include <cstring>

#include "velox/common/base/CheckedArithmetic.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::encoding {

// Reverse lookup table for decoding. 255 means invalid character.
// Only uppercase letters (A-Z) and digits 2-7 are valid per RFC 4648.
// Lowercase letters are NOT supported (matching Google Guava's
// BaseEncoding.base32()).
constexpr const Base32::ReverseIndex kBase32ReverseIndexTable = {
    255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, // 0-15
    255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, // 16-31
    255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, // 32-47
    255, 255, 26,  27,  28,  29,  30,  31,  255,
    255, 255, 255, 255, 255, 255, 255, // 48-63 ('2'-'7')
    255, 0,   1,   2,   3,   4,   5,   6,   7,
    8,   9,   10,  11,  12,  13,  14, // 64-79 ('A'-'O')
    15,  16,  17,  18,  19,  20,  21,  22,  23,
    24,  25,  255, 255, 255, 255, 255, // 80-95 ('P'-'Z')
    255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, // 96-111 (lowercase 'a'-'o' - INVALID)
    255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, // 112-127 (lowercase 'p'-'z' - INVALID)
    255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, // 128-143
    255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, // 144-159
    255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, // 160-175
    255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, // 176-191
    255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, // 192-207
    255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, // 208-223
    255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, // 224-239
    255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255 // 240-255
};

// Character set for Base32 encoding (RFC 4648).
// Uses uppercase letters A-Z (values 0-25) and digits 2-7 (values 26-31).
constexpr const Base32::Charset kBase32Charset = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '2', '3', '4', '5', '6', '7'};

// Verify that for every entry in kBase32Charset, the corresponding entry
// in kBase32ReverseIndexTable is correct.
static_assert(
    BaseEncoderUtils::checkForwardIndex(
        sizeof(kBase32Charset) - 1,
        kBase32Charset,
        kBase32ReverseIndexTable),
    "kBase32Charset has incorrect entries");

// Verify that for every entry in kBase32ReverseIndexTable, the corresponding
// entry in kBase32Charset is correct.
static_assert(
    BaseEncoderUtils::checkReverseIndex(
        sizeof(kBase32ReverseIndexTable) - 1,
        kBase32Charset,
        kBase32ReverseIndexTable),
    "kBase32ReverseIndexTable has incorrect entries.");

// static
size_t Base32::calculateEncodedSize(size_t inputSize, bool withPadding) {
  return BaseEncoderUtils::calculateEncodedSize(
      inputSize, withPadding, kBase32BlockSizes);
}

// static
std::string Base32::encode(std::string_view text) {
  return encode(text.data(), text.size());
}

// static
std::string Base32::encode(const char* input, size_t inputSize) {
  size_t encodedSize = calculateEncodedSize(inputSize);
  std::string encodedResult;
  encodedResult.resize(encodedSize);
  encode(input, inputSize, encodedResult.data());
  return encodedResult;
}

// static
void Base32::encode(const char* input, size_t inputSize, char* outputBuffer) {
  encodeImpl(input, inputSize, kBase32Charset, outputBuffer);
}

// static
void Base32::encodeImpl(
    const char* input,
    size_t inputSize,
    const Charset& charset,
    char* outputBuffer) {
  if (inputSize == 0) {
    return;
  }

  auto outputPointer = outputBuffer;

  // Process full blocks of 5 bytes (40 bits) -> 8 Base32 characters.
  for (; inputSize >= 5; inputSize -= 5, input += 5) {
    // Pack 5 bytes into a uint64_t for bit extraction.
    uint64_t block = static_cast<uint64_t>(static_cast<uint8_t>(input[0]))
            << 32 |
        static_cast<uint64_t>(static_cast<uint8_t>(input[1])) << 24 |
        static_cast<uint64_t>(static_cast<uint8_t>(input[2])) << 16 |
        static_cast<uint64_t>(static_cast<uint8_t>(input[3])) << 8 |
        static_cast<uint64_t>(static_cast<uint8_t>(input[4]));

    // Extract 8 groups of 5 bits from the 40-bit block.
    *outputPointer++ = charset[(block >> 35) & 0x1f];
    *outputPointer++ = charset[(block >> 30) & 0x1f];
    *outputPointer++ = charset[(block >> 25) & 0x1f];
    *outputPointer++ = charset[(block >> 20) & 0x1f];
    *outputPointer++ = charset[(block >> 15) & 0x1f];
    *outputPointer++ = charset[(block >> 10) & 0x1f];
    *outputPointer++ = charset[(block >> 5) & 0x1f];
    *outputPointer++ = charset[block & 0x1f];
  }

  // Handle trailing 1-4 bytes with padding (RFC 4648).
  if (inputSize > 0) {
    // Pack remaining bytes into the top of a uint64_t.
    uint64_t block = 0;
    for (size_t i = 0; i < inputSize; ++i) {
      block |= static_cast<uint64_t>(static_cast<uint8_t>(input[i]))
          << (32 - 8 * i);
    }

    // Number of encoded characters: ceil(inputBits / 5).
    // 1 byte (8 bits)  -> 2 chars + 6 padding
    // 2 bytes (16 bits) -> 4 chars + 4 padding
    // 3 bytes (24 bits) -> 5 chars + 3 padding
    // 4 bytes (32 bits) -> 7 chars + 1 padding
    size_t encodedChars = (inputSize * 8 + 4) / 5;

    for (size_t i = 0; i < encodedChars; ++i) {
      *outputPointer++ = charset[(block >> (35 - 5 * i)) & 0x1f];
    }

    // Pad to the next multiple of 8.
    size_t paddingCount = Base32::kEncodedBlockByteSize - encodedChars;
    memset(outputPointer, BaseEncoderUtils::kPadding, paddingCount);
    outputPointer += paddingCount;
  }
}

// static
folly::Expected<uint8_t, Status> Base32::base32ReverseLookup(
    char encodedChar,
    const ReverseIndex& reverseIndex) {
  auto index = reverseIndex[static_cast<uint8_t>(encodedChar)];
  if (index >= kCharsetSize) {
    return folly::makeUnexpected(
        Status::UserError("Unrecognized character: {}", encodedChar));
  }
  return index;
}

// static
folly::Expected<size_t, Status> Base32::calculateDecodedSize(
    const char* input,
    const size_t inputSize) {
  if (inputSize == 0) {
    return 0;
  }

  // Count valid (non-padding, non-whitespace) characters and validate them
  size_t validCharCount = 0;
  for (size_t i = 0; i < inputSize; ++i) {
    char c = input[i];
    if (c == BaseEncoderUtils::kPadding ||
        std::isspace(static_cast<unsigned char>(c))) {
      continue;
    }

    // Validate character first
    auto index = kBase32ReverseIndexTable[static_cast<uint8_t>(c)];
    if (index >= kCharsetSize) {
      return folly::makeUnexpected(
          Status::UserError("Unrecognized character: {}", c));
    }
    validCharCount++;
  }

  // Validate input length matches Google Guava's Base32 behavior.
  // Base32 encoding groups characters into quantums of 8 characters (40 bits).
  // Valid character counts (mod 8) are: 0, 2, 4, 5, 7
  // Invalid character counts (mod 8) are: 1, 3, 6
  // These invalid counts leave too many incomplete bits that cannot form
  // complete bytes.
  if (validCharCount > 0) {
    size_t remainder = validCharCount % 8;
    if (remainder == 1 || remainder == 3 || remainder == 6) {
      return folly::makeUnexpected(
          Status::UserError("Invalid input length {}", validCharCount));
    }
  }

  // Calculate decoded size
  // Each base32 character represents 5 bits
  // We need to convert to bytes (8 bits each)
  size_t totalBits = checkedMultiply(validCharCount, size_t(5));
  size_t decodedSize = totalBits / 8;

  return decodedSize;
}

// static
Status Base32::decode(
    const char* input,
    size_t inputSize,
    char* outputBuffer,
    size_t outputSize) {
  auto decodedSize = decodeImpl(
      input, inputSize, outputBuffer, outputSize, kBase32ReverseIndexTable);
  if (decodedSize.hasError()) {
    return decodedSize.error();
  }
  return Status::OK();
}

// Decodes Base32 input using the provided reverse lookup table.
// This is the core decoding implementation that accumulates 5-bit values
// from Base32 characters and outputs 8-bit bytes.
// static
folly::Expected<size_t, Status> Base32::decodeImpl(
    const char* input,
    size_t inputSize,
    char* outputBuffer,
    size_t outputSize,
    const ReverseIndex& reverseIndex) {
  if (inputSize == 0) {
    return 0;
  }

  size_t outputPos = 0;
  uint64_t accumulator = 0;
  size_t bitsAccumulated = 0;

  for (size_t i = 0; i < inputSize; ++i) {
    char c = input[i];

    // Skip padding and whitespace (RFC 4648 allows whitespace in encoded data)
    if (c == BaseEncoderUtils::kPadding ||
        std::isspace(static_cast<unsigned char>(c))) {
      continue;
    }

    // Validate and convert character to 5-bit value
    auto value = base32ReverseLookup(c, reverseIndex);
    if (value.hasError()) {
      return folly::makeUnexpected(value.error());
    }

    // Accumulate 5 bits from each Base32 character
    // Each character contributes 5 bits to the bit accumulator
    accumulator = (accumulator << 5) | value.value();
    bitsAccumulated += 5;

    // Extract full bytes (8 bits) when we have accumulated enough bits
    if (bitsAccumulated >= 8) {
      if (outputPos >= outputSize) {
        return folly::makeUnexpected(
            Status::UserError("Output buffer too small"));
      }
      outputBuffer[outputPos++] =
          static_cast<char>((accumulator >> (bitsAccumulated - 8)) & 0xFF);
      bitsAccumulated -= 8;
    }
  }

  return outputPos;
}

} // namespace facebook::velox::encoding
