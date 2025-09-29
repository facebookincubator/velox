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
#include <cstdint>
#include <cstring>

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
    uint8_t index,
    const Base64::Charset& charset,
    const Base64::ReverseIndex& reverseIndex) {
  return (reverseIndex[static_cast<uint8_t>(charset[index])] == index) &&
      (index > 0 ? checkForwardIndex(index - 1, charset, reverseIndex) : true);
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
    const char targetChar) {
  return index < charset.size() &&
      ((charset[index] == targetChar) ||
       findCharacterInCharset(charset, index + 1, targetChar));
}

// Checks the consistency of a reverse index mapping for a given character
// set.
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
    const Charset& charset,
    bool includePadding) {
  std::string encodedResult;
  encodeImpl(input, charset, includePadding, encodedResult);
  return encodedResult;
}

// static
size_t Base64::calculateEncodedSize(size_t inputSize, bool withPadding) {
  if (inputSize == 0) {
    return 0;
  }

  // Calculate the output size assuming that we are including padding.
  size_t encodedSize = ((inputSize + 2) / 3) * 4;
  if (!withPadding) {
    // If the padding was not requested, subtract the padding bytes.
    encodedSize -= (3 - (inputSize % 3)) % 3;
  }
  return encodedSize;
}

// static
void Base64::encode(std::string_view input, std::string& output) {
  encodeImpl(input, kBase64Charset, false, output);
}

// static
void Base64::encodeUrl(std::string_view input, std::string& output) {
  encodeImpl(input, kBase64UrlCharset, true, output);
}

// static
template <class T>
void Base64::encodeImpl(
    const T& input,
    const Charset& charset,
    bool includePadding,
    std::string& output) {
  auto inputSize = input.size();
  output.clear();
  if (inputSize == 0) {
    return;
  }

  size_t encodedSize = calculateEncodedSize(inputSize, includePadding);
  output.reserve(encodedSize);

  auto inputIterator = input.begin();

  // For each group of 3 bytes (24 bits) in the input, split that into
  // 4 groups of 6 bits and encode that using the supplied charset lookup
  for (; inputSize > 2; inputSize -= 3) {
    uint32_t inputBlock = static_cast<uint8_t>(*inputIterator++) << 16;
    inputBlock |= static_cast<uint8_t>(*inputIterator++) << 8;
    inputBlock |= static_cast<uint8_t>(*inputIterator++);

    output.push_back(charset[(inputBlock >> 18) & 0x3F]);
    output.push_back(charset[(inputBlock >> 12) & 0x3F]);
    output.push_back(charset[(inputBlock >> 6) & 0x3F]);
    output.push_back(charset[inputBlock & 0x3F]);
  }

  if (inputSize > 0) {
    // We have either 1 or 2 input bytes left.  Encode this similar to the
    // above (assuming 0 for all other bytes).  Optionally append the '='
    // character if it is requested.
    uint32_t inputBlock = static_cast<uint8_t>(*inputIterator++) << 16;
    output.push_back(charset[(inputBlock >> 18) & 0x3F]);

    if (inputSize > 1) {
      inputBlock |= static_cast<uint8_t>(*inputIterator) << 8;
      output.push_back(charset[(inputBlock >> 12) & 0x3F]);
      output.push_back(charset[(inputBlock >> 6) & 0x3F]);
      if (includePadding) {
        output.push_back(kPadding);
      }
    } else {
      output.push_back(charset[(inputBlock >> 12) & 0x3F]);
      if (includePadding) {
        output.push_back(kPadding);
        output.push_back(kPadding);
      }
    }
  }
}

// static
std::string Base64::encode(std::string_view input, bool includePadding) {
  return encodeImpl(input, kBase64Charset, includePadding);
}

namespace {

// This is a quick and simple iterator implementation for an IOBuf so that the
// template that uses iterators can work on IOBuf chains. It only implements
// postfix increment because that is all the algorithm needs, and it is a no-op
// since the read<>() function already increments the cursor.
class IOBufWrapper {
 private:
  class Iterator {
   public:
    explicit Iterator(const folly::IOBuf* inputBuffer) : cursor_(inputBuffer) {}

    Iterator& operator++(int32_t) {
      // This is a no-op since reading from the Cursor has already moved the
      // position.
      return *this;
    }

    uint8_t operator*() {
      // This will read _and_ increment the cursor.
      return cursor_.read<uint8_t>();
    }

   private:
    folly::io::Cursor cursor_;
  };

 public:
  explicit IOBufWrapper(const folly::IOBuf* inputBuffer)
      : input_(inputBuffer) {}
  size_t size() const {
    return input_->computeChainDataLength();
  }

  Iterator begin() const {
    return Iterator(input_);
  }

 private:
  const folly::IOBuf* input_;
};

} // namespace

// static
std::string Base64::encode(const folly::IOBuf* inputBuffer) {
  return encodeImpl(IOBufWrapper(inputBuffer), kBase64Charset, true);
}

// static
std::string Base64::decode(std::string_view input) {
  std::string output;
  auto decodedSize = decodeImpl(input, output, kBase64ReverseIndexTable);
  if (decodedSize.hasError()) {
    VELOX_USER_FAIL(decodedSize.error().message());
  }
  return output;
}

// static
Expected<uint8_t> Base64::base64ReverseLookup(
    char encodedChar,
    const ReverseIndex& reverseIndex) {
  auto reverseLookupValue = reverseIndex[static_cast<uint8_t>(encodedChar)];
  if (reverseLookupValue >= 0x40) {
    return folly::makeUnexpected(
        Status::UserError(
            "decode() - invalid input string: invalid character '{}'",
            encodedChar));
  }
  return reverseLookupValue;
}

// static
Status Base64::decode(std::string_view input, std::string& output) {
  auto decodedSize = decodeImpl(input, output, kBase64ReverseIndexTable);
  if (decodedSize.hasError()) {
    return decodedSize.error();
  }
  return Status::OK();
}

// static
Expected<size_t> Base64::calculateDecodedSize(std::string_view input) {
  size_t inputSize = input.size();
  if (inputSize == 0) {
    return 0;
  }

  // Check if the input string is padded
  if (isPadded(input)) {
    // If padded, ensure that the string length is a multiple of the encoded
    // block size
    if (inputSize % kEncodedBlockByteSize != 0) {
      return folly::makeUnexpected(
          Status::UserError(
              "Base64::decode() - invalid input string: "
              "string length is not a multiple of 4."));
    }

    auto decodedSize =
        (inputSize * kBinaryBlockByteSize) / kEncodedBlockByteSize;
    auto paddingCount = numPadding(input);

    // Adjust the needed size by deducting the bytes corresponding to the
    // padding from the calculated size.
    return decodedSize -
        ((paddingCount * kBinaryBlockByteSize) + (kEncodedBlockByteSize - 1)) /
        kEncodedBlockByteSize;
  }
  // If not padded, Calculate extra bytes, if any
  auto extraBytes = inputSize % kEncodedBlockByteSize;
  auto decodedSize = (inputSize / kEncodedBlockByteSize) * kBinaryBlockByteSize;

  // Adjust the needed size for extra bytes, if present
  if (extraBytes) {
    if (extraBytes == 1) {
      return folly::makeUnexpected(
          Status::UserError(
              "Base64::decode() - invalid input string: "
              "string length cannot be 1 more than a multiple of 4."));
    }
    decodedSize += (extraBytes * kBinaryBlockByteSize) / kEncodedBlockByteSize;
  }

  return decodedSize;
}

// static
Expected<size_t> Base64::decodeImpl(
    std::string_view input,
    std::string& output,
    const ReverseIndex& reverseIndex) {
  if (input.size() == 0) {
    output.clear();
    return 0;
  }

  const char* inputData = input.data();
  size_t inputSize = input.size();
  auto decodedSize = calculateDecodedSize(input);
  if (decodedSize.hasError()) {
    return folly::makeUnexpected(decodedSize.error());
  }

  output.clear();
  output.reserve(decodedSize.value());
  inputSize -= numPadding(input);

  // Handle full groups of 4 characters
  for (; inputSize > 4; inputSize -= 4, inputData += 4) {
    // Each character of the 4 encodes 6 bits of the original, grab each with
    // the appropriate shifts to rebuild the original and then split that back
    // into the original 8-bit bytes.
    uint32_t decodedBlock = 0;
    for (int i = 0; i < 4; ++i) {
      auto reverseLookupValue = base64ReverseLookup(inputData[i], reverseIndex);
      if (reverseLookupValue.hasError()) {
        return folly::makeUnexpected(reverseLookupValue.error());
      }
      decodedBlock |= reverseLookupValue.value() << (18 - 6 * i);
    }
    output.push_back(static_cast<char>((decodedBlock >> 16) & 0xff));
    output.push_back(static_cast<char>((decodedBlock >> 8) & 0xff));
    output.push_back(static_cast<char>(decodedBlock & 0xff));
  }

  // Handle the last 2-4 characters. This is similar to the above, but the
  // last 2 characters may or may not exist.
  if (inputSize >= 2) {
    uint32_t decodedBlock = 0;

    // Process the first two characters
    for (int i = 0; i < 2; ++i) {
      auto reverseLookupValue = base64ReverseLookup(inputData[i], reverseIndex);
      if (reverseLookupValue.hasError()) {
        return folly::makeUnexpected(reverseLookupValue.error());
      }
      decodedBlock |= reverseLookupValue.value() << (18 - 6 * i);
    }
    output.push_back(static_cast<char>((decodedBlock >> 16) & 0xff));

    if (inputSize > 2) {
      auto reverseLookupValue = base64ReverseLookup(inputData[2], reverseIndex);
      if (reverseLookupValue.hasError()) {
        return folly::makeUnexpected(reverseLookupValue.error());
      }
      decodedBlock |= reverseLookupValue.value() << 6;
      output.push_back(static_cast<char>((decodedBlock >> 8) & 0xff));

      if (inputSize > 3) {
        auto reverseLookupValue =
            base64ReverseLookup(inputData[3], reverseIndex);
        if (reverseLookupValue.hasError()) {
          return folly::makeUnexpected(reverseLookupValue.error());
        }
        decodedBlock |= reverseLookupValue.value();
        output.push_back(static_cast<char>(decodedBlock & 0xff));
      }
    }
  }

  return decodedSize.value();
}

// static
std::string Base64::encodeUrl(std::string_view input, bool includePadding) {
  return encodeImpl(input, kBase64UrlCharset, includePadding);
}

// static
std::string Base64::encodeUrl(const folly::IOBuf* inputBuffer) {
  return encodeImpl(IOBufWrapper(inputBuffer), kBase64UrlCharset, false);
}

// static
Status Base64::decodeUrl(std::string_view input, std::string& output) {
  auto decodedSize = decodeImpl(input, output, kBase64UrlReverseIndexTable);
  if (decodedSize.hasError()) {
    return decodedSize.error();
  }
  return Status::OK();
}

// static
std::string Base64::decodeUrl(std::string_view input) {
  std::string output;
  auto decodedSize = decodeImpl(input, output, kBase64UrlReverseIndexTable);
  if (decodedSize.hasError()) {
    VELOX_USER_FAIL(decodedSize.error().message());
  }
  return output;
}

// static
Status Base64::decodeMime(std::string_view input, std::string& output) {
  size_t inputSize = input.size();
  if (inputSize == 0) {
    return Status::OK();
  }

  // 24-bit buffer.
  uint32_t accumulator = 0;
  // Next shift amount.
  int bitsNeeded = 18;
  size_t idx = 0;

  while (idx < inputSize) {
    unsigned char c = static_cast<unsigned char>(input[idx++]);
    int val = kBase64ReverseIndexTable[c];

    // Padding character.
    if (c == kPadding) {
      // If we see '=' too early or only one '=' when two are needed → error.
      if (bitsNeeded == 18 ||
          (bitsNeeded == 6 && (idx == inputSize || input[idx++] != kPadding))) {
        return Status::UserError(
            "Input byte array has wrong 4-byte ending unit.");
      }
      break;
    }

    // Skip whitespace or other non-Base64 chars.
    if (val < 0 || val >= 0x40) {
      continue;
    }

    // Accumulate 6 bits
    accumulator |= (static_cast<uint32_t>(val) << bitsNeeded);
    bitsNeeded -= 6;

    // If we've collected 24 bits, write out 3 bytes.
    if (bitsNeeded < 0) {
      output.push_back(static_cast<char>((accumulator >> 16) & 0xFF));
      output.push_back(static_cast<char>((accumulator >> 8) & 0xFF));
      output.push_back(static_cast<char>(accumulator & 0xFF));
      accumulator = 0;
      bitsNeeded = 18;
    }
  }

  // Handle any remaining bits (1 or 2 bytes).
  if (bitsNeeded == 0) {
    output.push_back(static_cast<char>((accumulator >> 16) & 0xFF));
    output.push_back(static_cast<char>((accumulator >> 8) & 0xFF));
  } else if (bitsNeeded == 6) {
    output.push_back(static_cast<char>((accumulator >> 16) & 0xFF));
  } else if (bitsNeeded == 12) {
    return Status::UserError("Last unit does not have enough valid bits.");
  }

  // Verify no illegal trailing Base64 data.
  while (idx < inputSize) {
    unsigned char c = static_cast<unsigned char>(input[idx++]);
    int val = kBase64ReverseIndexTable[c];
    // Valid data after completion is an error.
    if (val >= 0 && val < 0x40) {
      return Status::UserError("Input byte array has incorrect ending.");
    }
    // '=' padding beyond handled ones is OK; other negatives are skips.
  }

  return Status::OK();
}

// static
Expected<size_t> Base64::calculateMimeDecodedSize(std::string_view input) {
  size_t inputSize = input.size();
  if (inputSize == 0) {
    return 0;
  }
  if (inputSize < 2) {
    if (kBase64ReverseIndexTable[static_cast<uint8_t>(input[0])] >= 0x40) {
      return 0;
    }
    return folly::makeUnexpected(
        Status::UserError(
            "Input should at least have 2 bytes for base64 bytes."));
  }
  auto decodedSize = inputSize;
  // Compute how many true Base64 chars.
  for (size_t i = 0; i < inputSize; ++i) {
    auto c = input[i];
    if (c == kPadding) {
      decodedSize -= inputSize - i;
      break;
    }
    if (kBase64ReverseIndexTable[static_cast<uint8_t>(c)] >= 0x40) {
      decodedSize--;
    }
  }
  // If no explicit padding but validChars ≢ 0 mod 4, infer missing '='.
  size_t paddings = 0;
  if ((decodedSize & 0x3) != 0) {
    paddings = 4 - (decodedSize & 0x3);
  }
  // Each 4-char block yields 3 bytes; subtract padding.
  decodedSize = 3 * ((decodedSize + 3) / 4) - paddings;
  return decodedSize;
}

// static
void Base64::encodeMime(std::string_view input, std::string& output) {
  // If there's nothing to encode, do nothing.
  size_t inputSize = input.size();
  if (inputSize == 0) {
    return;
  }

  const char* readPtr = input.data();
  // Bytes per 76-char line.
  const size_t bytesPerLine = (kMaxLineLength / 4) * 3;
  size_t remaining = inputSize;

  // Process full lines of up to 'bytesPerLine' bytes.
  while (remaining >= 3) {
    // Round down to a multiple of 3, but not more than one line.
    size_t chunk = std::min(bytesPerLine, (remaining / 3) * 3);
    const char* chunkEnd = readPtr + chunk;

    // Encode each group of 3 bytes into 4 Base64 characters.
    while (readPtr + 2 < chunkEnd) {
      // Read three bytes separately to avoid undefined behavior.
      uint8_t b0 = static_cast<uint8_t>(*readPtr++);
      uint8_t b1 = static_cast<uint8_t>(*readPtr++);
      uint8_t b2 = static_cast<uint8_t>(*readPtr++);
      // Pack into a 24-bit value.
      uint32_t trio = (static_cast<uint32_t>(b0) << 16) |
          (static_cast<uint32_t>(b1) << 8) | static_cast<uint32_t>(b2);
      // Emit four Base64 characters.
      output.push_back(kBase64Charset[(trio >> 18) & 0x3F]);
      output.push_back(kBase64Charset[(trio >> 12) & 0x3F]);
      output.push_back(kBase64Charset[(trio >> 6) & 0x3F]);
      output.push_back(kBase64Charset[trio & 0x3F]);
    }

    remaining -= chunk;

    // Insert CRLF if we filled exactly one line and still have more data.
    if (chunk == bytesPerLine && remaining > 0) {
      output.push_back(kNewline[0]);
      output.push_back(kNewline[1]);
    }
  }

  // Handle the final 1 or 2 leftover bytes with padding.
  if (remaining > 0) {
    uint8_t b0 = static_cast<uint8_t>(*readPtr++);
    // First Base64 character from the high 6 bits.
    output.push_back(kBase64Charset[b0 >> 2]);

    if (remaining == 1) {
      // Only one byte remains: produce two chars + two '=' paddings.
      output.push_back(kBase64Charset[(b0 & 0x03) << 4]);
      output.push_back(kPadding);
      output.push_back(kPadding);
    } else {
      // Two bytes remain: produce three chars + one '=' padding.
      uint8_t b1 = static_cast<uint8_t>(*readPtr);
      output.push_back(kBase64Charset[((b0 & 0x03) << 4) | (b1 >> 4)]);
      output.push_back(kBase64Charset[(b1 & 0x0F) << 2]);
      output.push_back(kPadding);
    }
  }
}

// static
size_t Base64::calculateMimeEncodedSize(size_t inputSize) {
  if (inputSize == 0) {
    return 0;
  }

  size_t encodedSize = calculateEncodedSize(inputSize, true);
  // Add CRLFs: one per full kMaxLineLength block.
  encodedSize += (encodedSize - 1) / kMaxLineLength * kNewline.size();
  return encodedSize;
}

} // namespace facebook::velox::encoding
