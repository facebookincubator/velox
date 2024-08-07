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

#include <cstdint>

#include <folly/Portability.h>
#include <folly/container/Foreach.h>
#include <folly/io/Cursor.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/encode/EncoderUtils.h"

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

// static
template <class T>
std::string Base64::encodeImpl(
    const T& input,
    const Charset& charset,
    bool includePadding) {
  std::string output;
  encodeImpl(input, charset, includePadding, output);
  return output;
}

// static
void Base64::encode(std::string_view input, std::string& output) {
  encodeImpl(input, kBase64Charset, true, output);
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

  size_t encodedSize = calculateEncodedSize(
      inputSize, includePadding, kBinaryBlockByteSize, kEncodedBlockByteSize);
  output.reserve(encodedSize);

  auto inputIterator = input.begin();

  // For each group of 3 bytes (24 bits) in the input, split that into
  // 4 groups of 6 bits and encode that using the supplied charset lookup
  for (; inputSize > 2; inputSize -= 3) {
    uint32_t inputBlock = (static_cast<uint8_t>(*inputIterator++) << 16);
    inputBlock |= (static_cast<uint8_t>(*inputIterator++) << 8);
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
    uint32_t inputBlock = (static_cast<uint8_t>(*inputIterator++) << 16);
    output.push_back(charset[(inputBlock >> 18) & 0x3F]);

    if (inputSize > 1) {
      inputBlock |= (static_cast<uint8_t>(*inputIterator) << 8);
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
std::string Base64::encode(std::string_view input) {
  return encodeImpl(input, kBase64Charset, true);
}

// static
std::string Base64::encode(const char* input, size_t inputSize) {
  return encodeImpl(std::string_view(input, inputSize), kBase64Charset, true);
}

// static
std::string Base64::decode(std::string_view input) {
  std::string output;
  auto status = decodeImpl(input, output, kBase64ReverseIndexTable);
  if (!status.ok()) {
    VELOX_USER_FAIL(status.message());
  }
  return output;
}

// static
Status Base64::decode(std::string_view input, std::string& output) {
  return decodeImpl(input, output, kBase64ReverseIndexTable);
}

// static
Expected<uint8_t> Base64::base64ReverseLookup(
    char encodedChar,
    const ReverseIndex& reverseIndex) {
  return reverseLookup(encodedChar, reverseIndex, 64);
}

// static
Status Base64::decodeImpl(
    std::string_view input,
    std::string& output,
    const ReverseIndex& reverseIndex) {
  output.clear();
  if (input.empty()) {
    return Status::OK();
  }

  size_t inputSize = input.size();
  auto decodedSize = calculateDecodedSize(
      input, inputSize, kBinaryBlockByteSize, kEncodedBlockByteSize);
  if (decodedSize.hasError()) {
    return decodedSize.error();
  }
  output.reserve(decodedSize.value());
  const char* inputPointer = input.data();
  // Handle full groups of 4 characters
  for (; inputSize > 4; inputSize -= 4, inputPointer += 4) {
    // Each character of the 4 encodes 6 bits of the original, grab each with
    // the appropriate shifts to rebuild the original and then split that back
    // into the original 8-bit bytes.
    uint32_t decodedBlock = 0;
    for (int i = 0; i < 4; ++i) {
      auto reverseLookupValue =
          base64ReverseLookup(inputPointer[i], reverseIndex);
      if (reverseLookupValue.hasError()) {
        return reverseLookupValue.error();
      }
      decodedBlock |= reverseLookupValue.value() << (18 - 6 * i);
    }
    output.push_back(static_cast<char>((decodedBlock >> 16) & 0xFF));
    output.push_back(static_cast<char>((decodedBlock >> 8) & 0xFF));
    output.push_back(static_cast<char>(decodedBlock & 0xFF));
  }

  // Handle the last 2-4 characters. This is similar to the above, but the
  // last 2 characters may or may not exist.
  if (inputSize >= 2) {
    uint32_t decodedBlock = 0;

    // Process the first two characters
    for (int i = 0; i < 2; ++i) {
      auto reverseLookupValue =
          base64ReverseLookup(inputPointer[i], reverseIndex);
      if (reverseLookupValue.hasError()) {
        return reverseLookupValue.error();
      }
      decodedBlock |= reverseLookupValue.value() << (18 - 6 * i);
    }
    output.push_back(static_cast<char>((decodedBlock >> 16) & 0xFF));

    if (inputSize > 2) {
      auto reverseLookupValue =
          base64ReverseLookup(inputPointer[2], reverseIndex);
      if (reverseLookupValue.hasError()) {
        return reverseLookupValue.error();
      }
      decodedBlock |= reverseLookupValue.value() << 6;
      output.push_back(static_cast<char>((decodedBlock >> 8) & 0xFF));

      if (inputSize > 3) {
        reverseLookupValue = base64ReverseLookup(inputPointer[3], reverseIndex);
        if (reverseLookupValue.hasError()) {
          return reverseLookupValue.error();
        }
        decodedBlock |= reverseLookupValue.value();
        output.push_back(static_cast<char>(decodedBlock & 0xFF));
      }
    }
  }

  return Status::OK();
}

// static
std::string Base64::encodeUrl(std::string_view input) {
  return encodeImpl(input, kBase64UrlCharset, false);
}

// static
Status Base64::decodeUrl(std::string_view input, std::string& output) {
  return decodeImpl(input, output, kBase64UrlReverseIndexTable);
}

// static
std::string Base64::decodeUrl(std::string_view input) {
  std::string output;
  auto status = decodeImpl(input, output, kBase64UrlReverseIndexTable);
  if (!status.ok()) {
    VELOX_USER_FAIL(status.message());
  }
  return output;
}

} // namespace facebook::velox::encoding
