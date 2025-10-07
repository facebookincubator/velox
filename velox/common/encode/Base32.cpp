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
#include <folly/container/Foreach.h>
#include <folly/io/Cursor.h>
#include <cctype>
#include <cstdint>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::encoding {

// Character set for Base32 encoding (RFC 4648)
constexpr const Base32::Charset kBase32Charset = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '2', '3', '4', '5', '6', '7'};

// Reverse lookup table for decoding. -1 means invalid character.
constexpr const Base32::ReverseIndex kBase32ReverseIndexTable = {
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, // 0-15
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, // 16-31
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, // 32-47
    -1, -1, 26, 27, 28, 29, 30, 31,
    -1, -1, -1, -1, -1, -1, -1, -1, // 48-63 ('2'-'7')
    -1, 0,  1,  2,  3,  4,  5,  6,
    7,  8,  9,  10, 11, 12, 13, 14, // 64-79 ('A'-'O')
    15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, -1, -1, -1, -1, -1, // 80-95 ('P'-'Z')
    -1, 0,  1,  2,  3,  4,  5,  6,
    7,  8,  9,  10, 11, 12, 13, 14, // 96-111 ('a'-'o')
    15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, -1, -1, -1, -1, -1, // 112-127 ('p'-'z')
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, // 128-143
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, // 144-159
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, // 160-175
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, // 176-191
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, // 192-207
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, // 208-223
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, // 224-239
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1 // 240-255
};

// static
folly::Expected<uint8_t, Status> Base32::base32ReverseLookup(
    char encodedChar,
    const ReverseIndex& reverseIndex) {
  int8_t index = reverseIndex[static_cast<uint8_t>(encodedChar)];
  if (index == -1) {
    return folly::makeUnexpected(
        Status::UserError("Unrecognized character: {}", encodedChar));
  }
  return static_cast<uint8_t>(index);
}

// static
size_t Base32::calculateEncodedSize(size_t inputSize, bool withPadding) {
  if (inputSize == 0) {
    return 0;
  }

  // Base32 encodes 5 bytes into 8 characters
  size_t encodedSize = ((inputSize + Base32::kBinaryBlockByteSize - 1) /
                        Base32::kBinaryBlockByteSize) *
      Base32::kEncodedBlockByteSize;

  if (!withPadding) {
    // Calculate the exact number of characters needed without padding
    size_t bitsNeeded = inputSize * 8; // total bits in input
    encodedSize = (bitsNeeded + 4) / 5; // bits per base32 character is 5
  }

  return encodedSize;
}

// static
folly::Expected<size_t, Status> Base32::calculateDecodedSize(
    const char* input,
    size_t& inputSize) {
  if (inputSize == 0) {
    return 0;
  }

  // Count valid (non-padding, non-whitespace) characters and validate them
  size_t validCharCount = 0;
  for (size_t i = 0; i < inputSize; ++i) {
    char c = input[i];
    if (c == Base32::kPadding || std::isspace(static_cast<unsigned char>(c))) {
      continue;
    }

    // Validate character first
    int8_t index = kBase32ReverseIndexTable[static_cast<uint8_t>(c)];
    if (index == -1) {
      return folly::makeUnexpected(Status::UserError(
          "decode() - invalid input string: invalid character '{}'", c));
    }
    validCharCount++;
  }

  // Check for invalid input length - Base32 cannot have exactly 1 valid
  // character
  if (validCharCount == 1) {
    return folly::makeUnexpected(
        Status::UserError("Invalid input length {}", validCharCount));
  }

  // Calculate decoded size
  // Each base32 character represents 5 bits
  // We need to convert to bytes (8 bits each)
  size_t totalBits = validCharCount * 5;
  size_t decodedSize = totalBits / 8;

  return decodedSize;
}

// Forward declare template specializations
template <>
void Base32::encodeImpl<folly::StringPiece>(
    const folly::StringPiece& input,
    const Charset& charset,
    bool includePadding,
    char* outputBuffer);

template <>
void Base32::encodeImpl<folly::IOBuf>(
    const folly::IOBuf& input,
    const Charset& charset,
    bool includePadding,
    char* outputBuffer);

// static
std::string Base32::encode(const char* input, size_t inputSize) {
  return encodeImpl(folly::StringPiece(input, inputSize), kBase32Charset, true);
}

// static
std::string Base32::encode(folly::StringPiece text) {
  return encodeImpl(text, kBase32Charset, true);
}

// static
std::string Base32::encode(const folly::IOBuf* inputBuffer) {
  size_t encodedSize =
      calculateEncodedSize(inputBuffer->computeChainDataLength(), true);
  std::string output;
  output.resize(encodedSize);
  encodeImpl(*inputBuffer, kBase32Charset, true, output.data());
  return output;
}

// static
void Base32::encode(const char* input, size_t inputSize, char* outputBuffer) {
  encodeImpl(
      folly::StringPiece(input, inputSize), kBase32Charset, true, outputBuffer);
}

// static
std::string Base32::decode(folly::StringPiece encodedText) {
  auto inputSize = encodedText.size();
  auto decodedSize = calculateDecodedSize(encodedText.data(), inputSize);
  if (decodedSize.hasError()) {
    VELOX_USER_FAIL("{}", decodedSize.error().message());
  }
  std::string output;
  output.resize(decodedSize.value());
  decode(encodedText.data(), encodedText.size(), output.data());
  return output;
}

// static
void Base32::decode(
    const std::pair<const char*, int32_t>& payload,
    std::string& output) {
  auto inputSize = static_cast<size_t>(payload.second);
  auto decodedSize = calculateDecodedSize(payload.first, inputSize);
  if (decodedSize.hasError()) {
    VELOX_USER_FAIL("{}", decodedSize.error().message());
  }
  output.resize(decodedSize.value());
  decode(payload.first, payload.second, output.data());
}

// static
void Base32::decode(const char* input, size_t inputSize, char* outputBuffer) {
  auto decodedSize = calculateDecodedSize(input, inputSize);
  if (decodedSize.hasError()) {
    VELOX_USER_FAIL("{}", decodedSize.error().message());
  }
  auto status = decode(input, inputSize, outputBuffer, decodedSize.value());
  if (!status.ok()) {
    VELOX_USER_FAIL("{}", status.message());
  }
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

// Template implementations for encoding
template <class T>
std::string Base32::encodeImpl(
    const T& input,
    const Charset& charset,
    bool includePadding) {
  size_t encodedSize = calculateEncodedSize(input.size(), includePadding);
  std::string output;
  output.resize(encodedSize);
  encodeImpl(input, charset, includePadding, output.data());
  return output;
}

// Specialization for folly::StringPiece
template <>
void Base32::encodeImpl<folly::StringPiece>(
    const folly::StringPiece& input,
    const Charset& charset,
    bool includePadding,
    char* outputBuffer) {
  auto inputSize = input.size();
  if (inputSize == 0) {
    return;
  }

  auto outputPointer = outputBuffer;
  auto inputData = input.data();
  size_t inputPos = 0;

  // For each group of 5 bytes (40 bits) in the input, split that into
  // 8 groups of 5 bits and encode that using the supplied charset lookup
  for (; inputSize > 4; inputSize -= 5, inputPos += 5) {
    uint64_t inputBlock =
        static_cast<uint64_t>(static_cast<uint8_t>(inputData[inputPos])) << 32;
    inputBlock |=
        static_cast<uint64_t>(static_cast<uint8_t>(inputData[inputPos + 1]))
        << 24;
    inputBlock |=
        static_cast<uint64_t>(static_cast<uint8_t>(inputData[inputPos + 2]))
        << 16;
    inputBlock |=
        static_cast<uint64_t>(static_cast<uint8_t>(inputData[inputPos + 3]))
        << 8;
    inputBlock |=
        static_cast<uint64_t>(static_cast<uint8_t>(inputData[inputPos + 4]));

    *outputPointer++ = charset[(inputBlock >> 35) & 0x1f];
    *outputPointer++ = charset[(inputBlock >> 30) & 0x1f];
    *outputPointer++ = charset[(inputBlock >> 25) & 0x1f];
    *outputPointer++ = charset[(inputBlock >> 20) & 0x1f];
    *outputPointer++ = charset[(inputBlock >> 15) & 0x1f];
    *outputPointer++ = charset[(inputBlock >> 10) & 0x1f];
    *outputPointer++ = charset[(inputBlock >> 5) & 0x1f];
    *outputPointer++ = charset[inputBlock & 0x1f];
  }

  if (inputSize > 0) {
    // We have 1, 2, 3, or 4 input bytes left. Encode this similar to the
    // above (assuming 0 for all other bytes). Optionally append the '='
    // character if it is requested.
    uint64_t inputBlock =
        static_cast<uint64_t>(static_cast<uint8_t>(inputData[inputPos])) << 32;
    *outputPointer++ = charset[(inputBlock >> 35) & 0x1f];
    if (inputSize > 1) {
      inputBlock |=
          static_cast<uint64_t>(static_cast<uint8_t>(inputData[inputPos + 1]))
          << 24;
      *outputPointer++ = charset[(inputBlock >> 30) & 0x1f];
      *outputPointer++ = charset[(inputBlock >> 25) & 0x1f];
      if (inputSize > 2) {
        inputBlock |=
            static_cast<uint64_t>(static_cast<uint8_t>(inputData[inputPos + 2]))
            << 16;
        *outputPointer++ = charset[(inputBlock >> 20) & 0x1f];
        if (inputSize > 3) {
          inputBlock |= static_cast<uint64_t>(
                            static_cast<uint8_t>(inputData[inputPos + 3]))
              << 8;
          *outputPointer++ = charset[(inputBlock >> 15) & 0x1f];
          *outputPointer++ = charset[(inputBlock >> 10) & 0x1f];
          *outputPointer++ = charset[(inputBlock >> 5) & 0x1f];
          if (includePadding) {
            *outputPointer++ = Base32::kPadding;
          }
        } else {
          *outputPointer++ = charset[(inputBlock >> 15) & 0x1f];
          if (includePadding) {
            *outputPointer++ = Base32::kPadding;
            *outputPointer++ = Base32::kPadding;
            *outputPointer++ = Base32::kPadding;
          }
        }
      } else {
        *outputPointer++ = charset[(inputBlock >> 25) & 0x1f];
        if (includePadding) {
          *outputPointer++ = Base32::kPadding;
          *outputPointer++ = Base32::kPadding;
          *outputPointer++ = Base32::kPadding;
          *outputPointer++ = Base32::kPadding;
          *outputPointer++ = Base32::kPadding;
        }
      }
    } else {
      *outputPointer++ = charset[(inputBlock >> 30) & 0x1f];
      if (includePadding) {
        *outputPointer++ = Base32::kPadding;
        *outputPointer++ = Base32::kPadding;
        *outputPointer++ = Base32::kPadding;
        *outputPointer++ = Base32::kPadding;
        *outputPointer++ = Base32::kPadding;
        *outputPointer++ = Base32::kPadding;
      }
    }
  }
}

// Specialization for folly::IOBuf
template <>
void Base32::encodeImpl<folly::IOBuf>(
    const folly::IOBuf& input,
    const Charset& charset,
    bool includePadding,
    char* outputBuffer) {
  // For IOBuf, iterate through all the buffers in the chain
  folly::io::Cursor cursor(&input);
  auto outputPointer = outputBuffer;

  // Process full 5-byte blocks
  while (cursor.length() >= 5) {
    uint64_t inputBlock = static_cast<uint64_t>(cursor.read<uint8_t>()) << 32;
    inputBlock |= static_cast<uint64_t>(cursor.read<uint8_t>()) << 24;
    inputBlock |= static_cast<uint64_t>(cursor.read<uint8_t>()) << 16;
    inputBlock |= static_cast<uint64_t>(cursor.read<uint8_t>()) << 8;
    inputBlock |= static_cast<uint64_t>(cursor.read<uint8_t>());

    *outputPointer++ = charset[(inputBlock >> 35) & 0x1f];
    *outputPointer++ = charset[(inputBlock >> 30) & 0x1f];
    *outputPointer++ = charset[(inputBlock >> 25) & 0x1f];
    *outputPointer++ = charset[(inputBlock >> 20) & 0x1f];
    *outputPointer++ = charset[(inputBlock >> 15) & 0x1f];
    *outputPointer++ = charset[(inputBlock >> 10) & 0x1f];
    *outputPointer++ = charset[(inputBlock >> 5) & 0x1f];
    *outputPointer++ = charset[inputBlock & 0x1f];
  }

  // Handle remaining bytes
  size_t remaining = cursor.length();
  if (remaining > 0) {
    uint64_t inputBlock = static_cast<uint64_t>(cursor.read<uint8_t>()) << 32;
    *outputPointer++ = charset[(inputBlock >> 35) & 0x1f];
    if (remaining > 1) {
      inputBlock |= static_cast<uint64_t>(cursor.read<uint8_t>()) << 24;
      *outputPointer++ = charset[(inputBlock >> 30) & 0x1f];
      *outputPointer++ = charset[(inputBlock >> 25) & 0x1f];
      if (remaining > 2) {
        inputBlock |= static_cast<uint64_t>(cursor.read<uint8_t>()) << 16;
        *outputPointer++ = charset[(inputBlock >> 20) & 0x1f];
        if (remaining > 3) {
          inputBlock |= static_cast<uint64_t>(cursor.read<uint8_t>()) << 8;
          *outputPointer++ = charset[(inputBlock >> 15) & 0x1f];
          *outputPointer++ = charset[(inputBlock >> 10) & 0x1f];
          *outputPointer++ = charset[(inputBlock >> 5) & 0x1f];
          if (includePadding) {
            *outputPointer++ = Base32::kPadding;
          }
        } else {
          *outputPointer++ = charset[(inputBlock >> 15) & 0x1f];
          if (includePadding) {
            *outputPointer++ = Base32::kPadding;
            *outputPointer++ = Base32::kPadding;
            *outputPointer++ = Base32::kPadding;
          }
        }
      } else {
        *outputPointer++ = charset[(inputBlock >> 25) & 0x1f];
        if (includePadding) {
          *outputPointer++ = Base32::kPadding;
          *outputPointer++ = Base32::kPadding;
          *outputPointer++ = Base32::kPadding;
          *outputPointer++ = Base32::kPadding;
          *outputPointer++ = Base32::kPadding;
        }
      }
    } else {
      *outputPointer++ = charset[(inputBlock >> 30) & 0x1f];
      if (includePadding) {
        *outputPointer++ = Base32::kPadding;
        *outputPointer++ = Base32::kPadding;
        *outputPointer++ = Base32::kPadding;
        *outputPointer++ = Base32::kPadding;
        *outputPointer++ = Base32::kPadding;
        *outputPointer++ = Base32::kPadding;
      }
    }
  }
}

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
  int bitsAccumulated = 0;

  for (size_t i = 0; i < inputSize; ++i) {
    char c = input[i];

    // Skip padding and whitespace
    if (c == Base32::kPadding || std::isspace(static_cast<unsigned char>(c))) {
      continue;
    }

    // Validate and convert character
    auto value = base32ReverseLookup(c, reverseIndex);
    if (value.hasError()) {
      return folly::makeUnexpected(value.error());
    }

    // Accumulate bits
    accumulator = (accumulator << 5) | value.value();
    bitsAccumulated += 5;

    // Extract bytes when we have enough bits
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
