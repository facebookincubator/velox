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

// Constants defining the size of binary and encoded blocks for Base32 encoding.
constexpr static int kBinaryBlockSize = 5; // 5 bytes of binary = 40 bits
constexpr static int kEncodedBlockSize = 8; // 8 bytes of encoded = 40 bits

constexpr Charset kBase32Charset = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                                    'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                                    'Y', 'Z', '2', '3', '4', '5', '6', '7'};

constexpr ReverseIndex kBase32ReverseIndexTable = {
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

/// Verify that for each 32 entries in kBase32Charset, the corresponding entry
/// in kBase32ReverseIndexTable is correct.
static_assert(
    checkForwardIndex(
        sizeof(kBase32Charset) / 2 - 1,
        kBase32Charset,
        kBase32ReverseIndexTable),
    "kBase32Charset has incorrect entries");

/// Verify that for every entry in kBase32ReverseIndexTable, the corresponding
/// entry in kBase32Charset is correct.
static_assert(
    checkReverseIndex(
        sizeof(kBase32ReverseIndexTable) - 1,
        kBase32Charset,
        kBase,
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
void Base32::encode(const char* data, size_t len, char* output) {
  encodeImpl(folly::StringPiece(data, len), kBase32Charset, true, output);
}

template <class T>
/* static */ void Base32::encodeImpl(
    const T& data,
    const Charset& charset,
    bool include_pad,
    char* out) {
  auto len = data.size();
  if (len == 0) {
    return;
  }

  auto wp = out;
  auto it = data.begin();

  auto append_padding = [include_pad](char* str, int n) -> char* {
    if (include_pad) {
      for (int i = 0; i < n; ++i) {
        *str++ = kPadding;
      }
    }
    return str;
  };

  /// For each group of 5 bytes (40 bits) in the input, split that into
  /// 8 groups of 5 bits and encode that using the supplied charset lookup.
  for (; len > 4; len -= 5) {
    uint64_t curr = uint64_t(*it++) << 32;
    curr |= uint8_t(*it++) << 24;
    curr |= uint8_t(*it++) << 16;
    curr |= uint8_t(*it++) << 8;
    curr |= uint8_t(*it++);

    *wp++ = charset[(curr >> 35) & 0x1f];
    *wp++ = charset[(curr >> 30) & 0x1f];
    *wp++ = charset[(curr >> 25) & 0x1f];
    *wp++ = charset[(curr >> 20) & 0x1f];
    *wp++ = charset[(curr >> 15) & 0x1f];
    *wp++ = charset[(curr >> 10) & 0x1f];
    *wp++ = charset[(curr >> 5) & 0x1f];
    *wp++ = charset[curr & 0x1f];
  }

  if (len > 0) {
    /// We have either 1 to 4 input bytes left.  Encode this similar to the
    /// above (assuming 0 for all other bytes).  Optionally append the '='
    /// character if it is requested.
    uint64_t curr = uint64_t(*it++) << 32;
    *wp++ = charset[(curr >> 35) & 0x1f];

    if (len > 3) {
      curr |= uint8_t(*it++) << 24;
      curr |= uint8_t(*it++) << 16;
      curr |= uint8_t(*it++) << 8;

      *wp++ = charset[(curr >> 30) & 0x1f];
      *wp++ = charset[(curr >> 25) & 0x1f];
      *wp++ = charset[(curr >> 20) & 0x1f];
      *wp++ = charset[(curr >> 15) & 0x1f];
      *wp++ = charset[(curr >> 10) & 0x1f];
      *wp++ = charset[(curr >> 5) & 0x1f];

      append_padding(wp, 1);
    } else if (len > 2) {
      curr |= uint8_t(*it++) << 24;
      curr |= uint8_t(*it++) << 16;

      *wp++ = charset[(curr >> 30) & 0x1f];
      *wp++ = charset[(curr >> 25) & 0x1f];
      *wp++ = charset[(curr >> 20) & 0x1f];
      *wp++ = charset[(curr >> 15) & 0x1f];

      append_padding(wp, 3);

    } else if (len > 1) {
      curr |= uint8_t(*it) << 24;

      *wp++ = charset[(curr >> 30) & 0x1f];
      *wp++ = charset[(curr >> 25) & 0x1f];
      *wp++ = charset[(curr >> 20) & 0x1f];

      append_padding(wp, 4);
    } else {
      *wp++ = charset[(curr >> 30) & 0x1f];

      append_padding(wp, 6);
    }
  }
}

} // namespace facebook::velox::encoding
