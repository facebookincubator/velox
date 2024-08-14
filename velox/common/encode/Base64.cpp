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
#include "velox/common/encode/Base64.h"

#include <folly/Portability.h>
#include <folly/container/Foreach.h>
#include <folly/io/Cursor.h>
#include <stdint.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::encoding {

// Constants defining the size in bytes of binary and encoded blocks for Base64
// encoding.
// Size of a binary block in bytes (3 bytes = 24 bits)
constexpr static int kBinaryBlockByteSize = 3;
// Size of an encoded block in bytes (4 bytes = 24 bits)
constexpr static int kEncodedBlockByteSize = 4;

// Character sets for Base64 and Base64 URL encoding
constexpr const Base64::Charset kBase64Charset = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};

constexpr const Base64::Charset kBase64UrlCharset = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '_'};

// Reverse lookup tables for decoding
constexpr const Base64::ReverseIndex kBase64ReverseIndexTable = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 62,  255,
    255, 255, 63,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  255, 255,
    255, 255, 255, 255, 255, 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
    10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
    25,  255, 255, 255, 255, 255, 255, 26,  27,  28,  29,  30,  31,  32,  33,
    34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
    49,  50,  51,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255};

constexpr const Base64::ReverseIndex kBase64UrlReverseIndexTable = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 62,  255,
    62,  255, 63,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  255, 255,
    255, 255, 255, 255, 255, 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
    10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
    25,  255, 255, 255, 255, 63,  255, 26,  27,  28,  29,  30,  31,  32,  33,
    34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
    49,  50,  51,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255};

// Validate the character in charset with ReverseIndex table
constexpr bool checkForwardIndex(
    uint8_t idx,
    const Base64::Charset& charset,
    const Base64::ReverseIndex& reverseIndex) {
  return (reverseIndex[static_cast<uint8_t>(charset[idx])] == idx) &&
      (idx > 0 ? checkForwardIndex(idx - 1, charset, reverseIndex) : true);
}

// Verify that for every entry in kBase64Charset, the corresponding entry
// in kBase64ReverseIndexTable is correct.
static_assert(
    checkForwardIndex(
        sizeof(kBase64Charset) - 1,
        kBase64Charset,
        kBase64ReverseIndexTable),
    "kBase64Charset has incorrect entries");

// Verify that for every entry in kBase64UrlCharset, the corresponding entry
// in kBase64UrlReverseIndexTable is correct.
static_assert(
    checkForwardIndex(
        sizeof(kBase64UrlCharset) - 1,
        kBase64UrlCharset,
        kBase64UrlReverseIndexTable),
    "kBase64UrlCharset has incorrect entries");

// Searches for a character within a charset up to a certain index.
constexpr bool findCharacterInCharset(
    const Base64::Charset& charset,
    uint8_t index,
    const char character) {
  return index < charset.size() &&
      ((charset[index] == character) ||
       findCharacterInCharset(charset, index + 1, character));
}

// Checks the consistency of a reverse index mapping for a given character set.
constexpr bool checkReverseIndex(
    uint8_t index,
    const Base64::Charset& charset,
    const Base64::ReverseIndex& reverseIndex) {
  return (reverseIndex[index] == 255
              ? !findCharacterInCharset(charset, 0, static_cast<char>(index))
              : (charset[reverseIndex[index]] == index)) &&
      (index > 0 ? checkReverseIndex(index - 1, charset, reverseIndex) : true);
}

// Verify that for every entry in kBase64ReverseIndexTable, the corresponding
// entry in kBase64Charset is correct.
static_assert(
    checkReverseIndex(
        sizeof(kBase64ReverseIndexTable) - 1,
        kBase64Charset,
        kBase64ReverseIndexTable),
    "kBase64ReverseIndexTable has incorrect entries.");

// Verify that for every entry in kBase64ReverseIndexTable, the corresponding
// entry in kBase64Charset is correct.
// We can't run this check as the URL version has two duplicate entries so
// that the url decoder can handle url encodings and default encodings
// static_assert(
//     checkReverseIndex(
//         sizeof(kBase64UrlReverseIndexTable) - 1,
//         kBase64UrlCharset,
//         kBase64UrlReverseIndexTable),
//     "kBase64UrlReverseIndexTable has incorrect entries.");

// Implementation of Base64 encoding and decoding functions.
// static
template <class T>
std::string Base64::encodeImpl(
    const T& input,
    const Base64::Charset& charset,
    bool includePadding) {
  size_t outputLength = calculateEncodedSize(input.size(), includePadding);
  std::string output;
  output.resize(outputLength);
  encodeImpl(input, charset, includePadding, output.data());
  return output;
}

// static
size_t Base64::calculateEncodedSize(size_t inputSize, bool includePadding) {
  if (inputSize == 0) {
    return 0;
  }

  // Calculate the output size assuming that we are including padding.
  size_t encodedSize = ((inputSize + 2) / 3) * 4;
  if (!includePadding) {
    encodedSize -= (3 - (inputSize % 3)) % 3;
  }
  return encodedSize;
}

// static
Status Base64::encode(std::string_view input, char* output) {
  return encodeImpl(input, kBase64Charset, true, output);
}

// static
Status Base64::encodeUrl(std::string_view input, char* output) {
  return encodeImpl(input, kBase64UrlCharset, true, output);
}

// static
template <class T>
Status Base64::encodeImpl(
    const T& input,
    const Base64::Charset& charset,
    bool includePadding,
    char* output) {
  auto inputSize = input.size();
  if (inputSize == 0) {
    return Status::OK();
  }

  auto outputPtr = output;
  auto dataIterator = input.begin();

  for (; inputSize > 2; inputSize -= 3) {
    uint32_t currentBlock = uint8_t(*dataIterator++) << 16;
    currentBlock |= uint8_t(*dataIterator++) << 8;
    currentBlock |= uint8_t(*dataIterator++);

    *outputPtr++ = charset[(currentBlock >> 18) & 0x3f];
    *outputPtr++ = charset[(currentBlock >> 12) & 0x3f];
    *outputPtr++ = charset[(currentBlock >> 6) & 0x3f];
    *outputPtr++ = charset[currentBlock & 0x3f];
  }

  if (inputSize > 0) {
    uint32_t currentBlock = uint8_t(*dataIterator++) << 16;
    *outputPtr++ = charset[(currentBlock >> 18) & 0x3f];
    if (inputSize > 1) {
      currentBlock |= uint8_t(*dataIterator) << 8;
      *outputPtr++ = charset[(currentBlock >> 12) & 0x3f];
      *outputPtr++ = charset[(currentBlock >> 6) & 0x3f];
      if (includePadding) {
        *outputPtr = kPadding;
      }
    } else {
      *outputPtr++ = charset[(currentBlock >> 12) & 0x3f];
      if (includePadding) {
        *outputPtr++ = kPadding;
        *outputPtr = kPadding;
      }
    }
  }
  return Status::OK();
}

// static
std::string Base64::encode(std::string_view text) {
  return encodeImpl(text, kBase64Charset, true);
}

// static
std::string Base64::encode(std::string_view input, size_t /*len*/) {
  return encodeImpl(input, kBase64Charset, true);
}

namespace {

/**
 * this is a quick and dirty iterator implementation for an IOBuf so that the
 * template that uses iterators can work on IOBuf chains.  It only implements
 * postfix increment because that is all the algorithm needs, and it is a noop
 * since the read<>() function already incremented the cursor.
 */
class IOBufWrapper {
 private:
  class Iterator {
   public:
    explicit Iterator(const folly::IOBuf* data) : cursor_(data) {}

    Iterator& operator++(int32_t) {
      // This is a noop since reading from the Cursor has already moved the
      // position
      return *this;
    }

    uint8_t operator*() {
      // This will read _and_ increment
      return cursor_.read<uint8_t>();
    }

   private:
    folly::io::Cursor cursor_;
  };

 public:
  explicit IOBufWrapper(const folly::IOBuf* data) : data_(data) {}

  size_t size() const {
    return data_->computeChainDataLength();
  }

  Iterator begin() const {
    return Iterator(data_);
  }

 private:
  const folly::IOBuf* data_;
};

} // namespace

// static
std::string Base64::encode(const folly::IOBuf* input) {
  return encodeImpl(IOBufWrapper(input), kBase64Charset, true);
}

// static
std::string Base64::decode(std::string_view encoded) {
  std::string output;
  Base64::decode(encoded, output);
  return output;
}

// static
void Base64::decode(std::string_view input, std::string& output) {
  size_t inputSize{input.size()};
  size_t decodedSize;

  calculateDecodedSize(input, inputSize, decodedSize);
  output.resize(decodedSize);
  decode(input.data(), inputSize, output.data(), output.size());
}

// static
void Base64::decode(std::string_view input, size_t size, char* output) {
  size_t outputLength = size / 4 * 3;
  Base64::decode(input, size, output, outputLength);
}

// static
uint8_t Base64::base64ReverseLookup(
    char character,
    const Base64::ReverseIndex& reverseIndex) {
  auto lookupValue = reverseIndex[(uint8_t)character];
  if (lookupValue >= 0x40) {
    VELOX_USER_FAIL("decode() - invalid input string: invalid characters");
  }
  return lookupValue;
}

// static
Status Base64::decode(
    std::string_view input,
    size_t inputSize,
    char* output,
    size_t outputSize) {
  return decodeImpl(
      input, inputSize, output, outputSize, kBase64ReverseIndexTable);
}

// static
Status Base64::calculateDecodedSize(
    std::string_view input,
    size_t& inputSize,
    size_t& decodedSize) {
  if (inputSize == 0) {
    decodedSize = 0;
    return Status::OK();
  }

  // Check if the input data is padded
  if (isPadded(input, inputSize)) {
    // If padded, ensure that the string length is a multiple of the encoded
    // block size
    if (inputSize % kEncodedBlockByteSize != 0) {
      return Status::UserError(
          "Base64::decode() - invalid input string: string length is not a multiple of 4.");
    }

    decodedSize = (inputSize * kBinaryBlockByteSize) / kEncodedBlockByteSize;
    auto padding = numPadding(input, inputSize);
    inputSize -= padding;

    // Adjust the needed size by deducting the bytes corresponding to the
    // padding from the calculated size.
    decodedSize -=
        ((padding * kBinaryBlockByteSize) + (kEncodedBlockByteSize - 1)) /
        kEncodedBlockByteSize;
    return Status::OK();
  }

  // If not padded, calculate extra bytes, if any
  auto extraBytes = inputSize % kEncodedBlockByteSize;
  decodedSize = (inputSize / kEncodedBlockByteSize) * kBinaryBlockByteSize;

  // Adjust the needed size for extra bytes, if present
  if (extraBytes) {
    if (extraBytes == 1) {
      return Status::UserError(
          "Base64::decode() - invalid input string: string length cannot be 1 more than a multiple of 4.");
    }
    decodedSize += (extraBytes * kBinaryBlockByteSize) / kEncodedBlockByteSize;
  }

  return Status::OK();
}

// static
Status Base64::decodeImpl(
    std::string_view input,
    size_t inputSize,
    char* output,
    size_t outputSize,
    const Base64::ReverseIndex& reverseIndex) {
  if (!inputSize) {
    return Status::OK();
  }

  size_t decodedSize;
  // Calculate decoded size and check for status
  auto status = calculateDecodedSize(input, inputSize, decodedSize);
  if (!status.ok()) {
    return status;
  }

  if (outputSize < decodedSize) {
    return Status::UserError(
        "Base64::decode() - invalid output string: output string is too small.");
  }

  // Handle full groups of 4 characters
  for (; inputSize > 4; inputSize -= 4, input.remove_prefix(4), output += 3) {
    // Each character of the 4 encodes 6 bits of the original, grab each with
    // the appropriate shifts to rebuild the original and then split that back
    // into the original 8-bit bytes.
    uint32_t currentBlock =
        (base64ReverseLookup(input[0], reverseIndex) << 18) |
        (base64ReverseLookup(input[1], reverseIndex) << 12) |
        (base64ReverseLookup(input[2], reverseIndex) << 6) |
        base64ReverseLookup(input[3], reverseIndex);
    output[0] = (currentBlock >> 16) & 0xff;
    output[1] = (currentBlock >> 8) & 0xff;
    output[2] = currentBlock & 0xff;
  }

  // Handle the last 2-4 characters. This is similar to the above, but the
  // last 2 characters may or may not exist.
  DCHECK(inputSize >= 2);
  uint32_t currentBlock = (base64ReverseLookup(input[0], reverseIndex) << 18) |
      (base64ReverseLookup(input[1], reverseIndex) << 12);
  output[0] = (currentBlock >> 16) & 0xff;
  if (inputSize > 2) {
    currentBlock |= base64ReverseLookup(input[2], reverseIndex) << 6;
    output[1] = (currentBlock >> 8) & 0xff;
    if (inputSize > 3) {
      currentBlock |= base64ReverseLookup(input[3], reverseIndex);
      output[2] = currentBlock & 0xff;
    }
  }

  return Status::OK();
}

// static
std::string Base64::encodeUrl(std::string_view input) {
  return encodeImpl(input, kBase64UrlCharset, false);
}

// static
std::string Base64::encodeUrl(const folly::IOBuf* input) {
  return encodeImpl(IOBufWrapper(input), kBase64UrlCharset, false);
}

// static
Status Base64::decodeUrl(
    std::string_view input,
    size_t inputSize,
    char* output,
    size_t outputSize) {
  return decodeImpl(
      input, inputSize, output, outputSize, kBase64UrlReverseIndexTable);
}

// static
std::string Base64::decodeUrl(std::string_view input) {
  std::string output;
  Base64::decodeUrl(input, output);
  return output;
}

// static
void Base64::decodeUrl(std::string_view input, std::string& output) {
  size_t out_len = (input.size() + 3) / 4 * 3;
  output.resize(out_len, '\0');
  Base64::decodeImpl(
      input.data(),
      input.size(),
      &output[0],
      out_len,
      kBase64UrlReverseIndexTable);
  output.resize(out_len);
}

} // namespace facebook::velox::encoding
