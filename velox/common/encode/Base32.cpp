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

size_t Base32::calculateDecodedSize(const char* data, size_t& size) {
  if (size == 0) {
    return 0;
  }

  // Check if the input data is padded
  if (isPadded(data, size)) {
    /// If padded, ensure that the string length is a multiple of the encoded
    /// block size.
    if (size % kEncodedBlockByteSize != 0) {
      VELOX_USER_FAIL(
          "Base32::decode() - invalid input string: "
          "string length is not a multiple of 8.");
    }

    auto needed = (size * kBinaryBlockByteSize) / kEncodedBlockByteSize;
    auto padding = numPadding(data, size);
    size -= padding;

    // Adjust the needed size by deducting the bytes corresponding to the
    // padding from the calculated size.
    return needed -
        ((padding * kBinaryBlockByteSize) + (kEncodedBlockByteSize - 1)) /
        kEncodedBlockByteSize;
  } else {
    // If not padded, calculate extra bytes, if any.
    auto extra = size % kEncodedBlockByteSize;
    auto needed = (size / kEncodedBlockByteSize) * kBinaryBlockByteSize;

    // Adjust the needed size for extra bytes, if present.
    if (extra) {
      if ((extra == 6) || (extra == 3) || (extra == 1)) {
        VELOX_USER_FAIL(
            "Base32::decode() - invalid input string: "
            "string length cannot be 6, 3 or 1 more than a multiple of 8.");
      }
      needed += (extra * kBinaryBlockByteSize) / kEncodedBlockByteSize;
    }

    return needed;
  }
}

size_t
Base32::decode(const char* src, size_t src_len, char* dst, size_t dst_len) {
  return decodeImpl(src, src_len, dst, dst_len, kBase32ReverseIndexTable);
}

size_t Base32::decodeImpl(
    const char* src,
    size_t src_len,
    char* dst,
    size_t dst_len,
    const ReverseIndex& reverse_lookup) {
  if (!src_len) {
    return 0;
  }

  auto needed = calculateDecodedSize(src, src_len);
  if (dst_len < needed) {
    VELOX_USER_FAIL(
        "Base32::decode() - invalid output string: "
        "output string is too small.");
  }

  // Handle full groups of 8 characters.
  for (; src_len > 8; src_len -= 8, src += 8, dst += 5) {
    /// Each character of the 8 bytes encode 5 bits of the original, grab each
    /// with the appropriate shifts to rebuild the original and then split that
    /// back into the original 8 bit bytes.
    uint64_t last =
        (uint64_t(baseReverseLookup(kBase, src[0], reverse_lookup)) << 35) |
        (uint64_t(baseReverseLookup(kBase, src[1], reverse_lookup)) << 30) |
        (baseReverseLookup(kBase, src[2], reverse_lookup) << 25) |
        (baseReverseLookup(kBase, src[3], reverse_lookup) << 20) |
        (baseReverseLookup(kBase, src[4], reverse_lookup) << 15) |
        (baseReverseLookup(kBase, src[5], reverse_lookup) << 10) |
        (baseReverseLookup(kBase, src[6], reverse_lookup) << 5) |
        baseReverseLookup(kBase, src[7], reverse_lookup);
    dst[0] = (last >> 32) & 0xff;
    dst[1] = (last >> 24) & 0xff;
    dst[2] = (last >> 16) & 0xff;
    dst[3] = (last >> 8) & 0xff;
    dst[4] = last & 0xff;
  }

  /// Handle the last 2, 4, 5, 7 or 8 characters.  This is similar to the above,
  /// but the last characters may or may not exist.
  DCHECK(src_len >= 2);
  uint64_t last =
      (uint64_t(baseReverseLookup(kBase, src[0], reverse_lookup)) << 35) |
      (uint64_t(baseReverseLookup(kBase, src[1], reverse_lookup)) << 30);
  dst[0] = (last >> 32) & 0xff;
  if (src_len > 2) {
    last |= baseReverseLookup(kBase, src[2], reverse_lookup) << 25;
    last |= baseReverseLookup(kBase, src[3], reverse_lookup) << 20;
    dst[1] = (last >> 24) & 0xff;
    if (src_len > 4) {
      last |= baseReverseLookup(kBase, src[4], reverse_lookup) << 15;
      dst[2] = (last >> 16) & 0xff;
      if (src_len > 5) {
        last |= baseReverseLookup(kBase, src[5], reverse_lookup) << 10;
        last |= baseReverseLookup(kBase, src[6], reverse_lookup) << 5;
        dst[3] = (last >> 8) & 0xff;
        if (src_len > 7) {
          last |= baseReverseLookup(kBase, src[7], reverse_lookup);
          dst[4] = last & 0xff;
        }
      }
    }
  }

  return needed;
}

} // namespace facebook::velox::encoding
