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

// Adapted from Apache DataSketches

#pragma once

#include "velox/common/base/Exceptions.h"

#include <memory>
#include <string>

namespace facebook::velox::common::theta {

static inline uint8_t
packBits(uint64_t value, uint8_t bits, uint8_t*& ptr, uint8_t offset) {
  if (offset > 0) {
    const uint8_t chunkBits = 8 - offset;
    const uint8_t mask = (1 << chunkBits) - 1;
    if (bits < chunkBits) {
      *ptr |= (value << (chunkBits - bits)) & mask;
      return offset + bits;
    }
    *ptr++ |= (value >> (bits - chunkBits)) & mask;
    bits -= chunkBits;
  }
  while (bits >= 8) {
    *ptr++ = static_cast<uint8_t>(value >> (bits - 8));
    bits -= 8;
  }
  if (bits > 0) {
    *ptr = static_cast<uint8_t>(value << (8 - bits));
    return bits;
  }
  return 0;
}

static inline uint8_t
unpackBits(uint64_t& value, uint8_t bits, const uint8_t*& ptr, uint8_t offset) {
  const uint8_t availBits = 8 - offset;
  const uint8_t chunkBits = std::min(availBits, bits);
  const uint8_t mask = (1 << chunkBits) - 1;
  value = (*ptr >> (availBits - chunkBits)) & mask;
  ptr += availBits == chunkBits;
  offset = (offset + chunkBits) & 7;
  bits -= chunkBits;
  while (bits >= 8) {
    value <<= 8;
    value |= *ptr++;
    bits -= 8;
  }
  if (bits > 0) {
    value <<= bits;
    value |= *ptr >> (8 - bits);
    return bits;
  }
  return offset;
}

// pack given number of bits from a block of 8 64-bit values into bytes
// we don't need 0 and 64 bits
// we assume that higher bits (which we are not packing) are zeros
// this assumption allows to avoid masking operations

static inline void packBits1(const uint64_t* values, uint8_t* ptr) {
  *ptr = static_cast<uint8_t>(values[0] << 7);
  *ptr |= static_cast<uint8_t>(values[1] << 6);
  *ptr |= static_cast<uint8_t>(values[2] << 5);
  *ptr |= static_cast<uint8_t>(values[3] << 4);
  *ptr |= static_cast<uint8_t>(values[4] << 3);
  *ptr |= static_cast<uint8_t>(values[5] << 2);
  *ptr |= static_cast<uint8_t>(values[6] << 1);
  *ptr |= static_cast<uint8_t>(values[7]);
}

static inline void packBits2(const uint64_t* values, uint8_t* ptr) {
  *ptr = static_cast<uint8_t>(values[0] << 6);
  *ptr |= static_cast<uint8_t>(values[1] << 4);
  *ptr |= static_cast<uint8_t>(values[2] << 2);
  *ptr++ |= static_cast<uint8_t>(values[3]);

  *ptr = static_cast<uint8_t>(values[4] << 6);
  *ptr |= static_cast<uint8_t>(values[5] << 4);
  *ptr |= static_cast<uint8_t>(values[6] << 2);
  *ptr |= static_cast<uint8_t>(values[7]);
}

static inline void packBits3(const uint64_t* values, uint8_t* ptr) {
  *ptr = static_cast<uint8_t>(values[0] << 5);
  *ptr |= static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 1);

  *ptr = static_cast<uint8_t>(values[2] << 7);
  *ptr |= static_cast<uint8_t>(values[3] << 4);
  *ptr |= static_cast<uint8_t>(values[4] << 1);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr |= static_cast<uint8_t>(values[6] << 3);
  *ptr |= static_cast<uint8_t>(values[7]);
}

static inline void packBits4(const uint64_t* values, uint8_t* ptr) {
  *ptr = static_cast<uint8_t>(values[0] << 4);
  *ptr++ |= static_cast<uint8_t>(values[1]);

  *ptr = static_cast<uint8_t>(values[2] << 4);
  *ptr++ |= static_cast<uint8_t>(values[3]);

  *ptr = static_cast<uint8_t>(values[4] << 4);
  *ptr++ |= static_cast<uint8_t>(values[5]);

  *ptr = static_cast<uint8_t>(values[6] << 4);
  *ptr |= static_cast<uint8_t>(values[7]);
}

static inline void packBits5(const uint64_t* values, uint8_t* ptr) {
  *ptr = static_cast<uint8_t>(values[0] << 3);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr |= static_cast<uint8_t>(values[2] << 1);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 1);

  *ptr = static_cast<uint8_t>(values[4] << 7);
  *ptr |= static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 3);

  *ptr = static_cast<uint8_t>(values[6] << 5);
  *ptr |= static_cast<uint8_t>(values[7]);
}

static inline void packBits6(const uint64_t* values, uint8_t* ptr) {
  *ptr = static_cast<uint8_t>(values[0] << 2);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 2);

  *ptr = static_cast<uint8_t>(values[2] << 6);
  *ptr++ |= static_cast<uint8_t>(values[3]);

  *ptr = static_cast<uint8_t>(values[4] << 2);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 2);

  *ptr = static_cast<uint8_t>(values[6] << 6);
  *ptr |= static_cast<uint8_t>(values[7]);
}

static inline void packBits7(const uint64_t* values, uint8_t* ptr) {
  *ptr = static_cast<uint8_t>(values[0] << 1);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 5);

  *ptr = static_cast<uint8_t>(values[2] << 3);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 3);

  *ptr = static_cast<uint8_t>(values[4] << 5);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 1);

  *ptr = static_cast<uint8_t>(values[6] << 7);
  *ptr |= static_cast<uint8_t>(values[7]);
}

static inline void packBits8(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0]);
  *ptr++ = static_cast<uint8_t>(values[1]);
  *ptr++ = static_cast<uint8_t>(values[2]);
  *ptr++ = static_cast<uint8_t>(values[3]);
  *ptr++ = static_cast<uint8_t>(values[4]);
  *ptr++ = static_cast<uint8_t>(values[5]);
  *ptr++ = static_cast<uint8_t>(values[6]);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits9(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 1);

  *ptr = static_cast<uint8_t>(values[0] << 7);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 3);

  *ptr = static_cast<uint8_t>(values[2] << 5);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 5);

  *ptr = static_cast<uint8_t>(values[4] << 3);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 7);

  *ptr = static_cast<uint8_t>(values[6] << 1);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits10(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 2);

  *ptr = static_cast<uint8_t>(values[0] << 6);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 6);

  *ptr = static_cast<uint8_t>(values[2] << 2);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 8);

  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 2);

  *ptr = static_cast<uint8_t>(values[4] << 6);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 6);

  *ptr = static_cast<uint8_t>(values[6] << 2);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits11(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 3);

  *ptr = static_cast<uint8_t>(values[0] << 5);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 9);

  *ptr++ = static_cast<uint8_t>(values[2] >> 1);

  *ptr = static_cast<uint8_t>(values[2] << 7);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 7);

  *ptr = static_cast<uint8_t>(values[4] << 1);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 10);

  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 5);

  *ptr = static_cast<uint8_t>(values[6] << 3);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits12(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 4);

  *ptr = static_cast<uint8_t>(values[0] << 4);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 8);

  *ptr++ = static_cast<uint8_t>(values[1]);

  *ptr++ = static_cast<uint8_t>(values[2] >> 4);

  *ptr = static_cast<uint8_t>(values[2] << 4);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 8);

  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 4);

  *ptr = static_cast<uint8_t>(values[4] << 4);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 8);

  *ptr++ = static_cast<uint8_t>(values[5]);

  *ptr++ = static_cast<uint8_t>(values[6] >> 4);

  *ptr = static_cast<uint8_t>(values[6] << 4);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits13(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 5);

  *ptr = static_cast<uint8_t>(values[0] << 3);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 10);

  *ptr++ = static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 7);

  *ptr = static_cast<uint8_t>(values[2] << 1);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 12);

  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 9);

  *ptr++ = static_cast<uint8_t>(values[4] >> 1);

  *ptr = static_cast<uint8_t>(values[4] << 7);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 11);

  *ptr++ = static_cast<uint8_t>(values[6] >> 3);

  *ptr = static_cast<uint8_t>(values[6] << 5);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits14(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 6);

  *ptr = static_cast<uint8_t>(values[0] << 2);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 12);

  *ptr++ = static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 10);

  *ptr++ = static_cast<uint8_t>(values[2] >> 2);

  *ptr = static_cast<uint8_t>(values[2] << 6);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 8);

  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 6);

  *ptr = static_cast<uint8_t>(values[4] << 2);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 12);

  *ptr++ = static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 10);

  *ptr++ = static_cast<uint8_t>(values[6] >> 2);

  *ptr = static_cast<uint8_t>(values[6] << 6);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits15(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 7);

  *ptr = static_cast<uint8_t>(values[0] << 1);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 14);

  *ptr++ = static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 13);

  *ptr++ = static_cast<uint8_t>(values[2] >> 5);

  *ptr = static_cast<uint8_t>(values[2] << 3);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 12);

  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 11);

  *ptr++ = static_cast<uint8_t>(values[4] >> 3);

  *ptr = static_cast<uint8_t>(values[4] << 5);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 10);

  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 9);

  *ptr++ = static_cast<uint8_t>(values[6] >> 1);

  *ptr = static_cast<uint8_t>(values[6] << 7);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits16(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 8);
  *ptr++ = static_cast<uint8_t>(values[0]);

  *ptr++ = static_cast<uint8_t>(values[1] >> 8);
  *ptr++ = static_cast<uint8_t>(values[1]);

  *ptr++ = static_cast<uint8_t>(values[2] >> 8);
  *ptr++ = static_cast<uint8_t>(values[2]);

  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 8);
  *ptr++ = static_cast<uint8_t>(values[4]);

  *ptr++ = static_cast<uint8_t>(values[5] >> 8);
  *ptr++ = static_cast<uint8_t>(values[5]);

  *ptr++ = static_cast<uint8_t>(values[6] >> 8);
  *ptr++ = static_cast<uint8_t>(values[6]);

  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits17(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 9);

  *ptr++ = static_cast<uint8_t>(values[0] >> 1);

  *ptr = static_cast<uint8_t>(values[0] << 7);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 10);

  *ptr++ = static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 11);

  *ptr++ = static_cast<uint8_t>(values[2] >> 3);

  *ptr = static_cast<uint8_t>(values[2] << 5);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 12);

  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 13);

  *ptr++ = static_cast<uint8_t>(values[4] >> 5);

  *ptr = static_cast<uint8_t>(values[4] << 3);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 14);

  *ptr++ = static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 15);

  *ptr++ = static_cast<uint8_t>(values[6] >> 7);

  *ptr = static_cast<uint8_t>(values[6] << 1);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 16);

  *ptr++ = static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits18(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 10);

  *ptr++ = static_cast<uint8_t>(values[0] >> 2);

  *ptr = static_cast<uint8_t>(values[0] << 6);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 12);

  *ptr++ = static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 14);

  *ptr++ = static_cast<uint8_t>(values[2] >> 6);

  *ptr = static_cast<uint8_t>(values[2] << 2);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 16);

  *ptr++ = static_cast<uint8_t>(values[3] >> 8);

  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 10);

  *ptr++ = static_cast<uint8_t>(values[4] >> 2);

  *ptr = static_cast<uint8_t>(values[4] << 6);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 12);

  *ptr++ = static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 14);

  *ptr++ = static_cast<uint8_t>(values[6] >> 6);

  *ptr = static_cast<uint8_t>(values[6] << 2);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 16);

  *ptr++ = static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits19(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 11);

  *ptr++ = static_cast<uint8_t>(values[0] >> 3);

  *ptr = static_cast<uint8_t>(values[0] << 5);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 14);

  *ptr++ = static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 17);

  *ptr++ = static_cast<uint8_t>(values[2] >> 9);

  *ptr++ = static_cast<uint8_t>(values[2] >> 1);

  *ptr = static_cast<uint8_t>(values[2] << 7);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 12);

  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 15);

  *ptr++ |= static_cast<uint8_t>(values[4] >> 7);

  *ptr = static_cast<uint8_t>(values[4] << 1);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 18);

  *ptr++ = static_cast<uint8_t>(values[5] >> 10);

  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 13);

  *ptr++ = static_cast<uint8_t>(values[6] >> 5);

  *ptr = static_cast<uint8_t>(values[6] << 3);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 16);

  *ptr++ = static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits20(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 12);

  *ptr++ = static_cast<uint8_t>(values[0] >> 4);

  *ptr = static_cast<uint8_t>(values[0] << 4);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 16);

  *ptr++ = static_cast<uint8_t>(values[1] >> 8);

  *ptr++ = static_cast<uint8_t>(values[1]);

  *ptr++ = static_cast<uint8_t>(values[2] >> 12);

  *ptr++ = static_cast<uint8_t>(values[2] >> 4);

  *ptr = static_cast<uint8_t>(values[2] << 4);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 16);

  *ptr++ = static_cast<uint8_t>(values[3] >> 8);

  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 12);

  *ptr++ = static_cast<uint8_t>(values[4] >> 4);

  *ptr = static_cast<uint8_t>(values[4] << 4);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 16);

  *ptr++ = static_cast<uint8_t>(values[5] >> 8);

  *ptr++ = static_cast<uint8_t>(values[5]);

  *ptr++ = static_cast<uint8_t>(values[6] >> 12);

  *ptr++ = static_cast<uint8_t>(values[6] >> 4);

  *ptr = static_cast<uint8_t>(values[6] << 4);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 16);

  *ptr++ = static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits21(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 13);

  *ptr++ = static_cast<uint8_t>(values[0] >> 5);

  *ptr = static_cast<uint8_t>(values[0] << 3);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 18);

  *ptr++ = static_cast<uint8_t>(values[1] >> 10);

  *ptr++ = static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 15);

  *ptr++ = static_cast<uint8_t>(values[2] >> 7);

  *ptr = static_cast<uint8_t>(values[2] << 1);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 20);

  *ptr++ = static_cast<uint8_t>(values[3] >> 12);

  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 17);

  *ptr++ = static_cast<uint8_t>(values[4] >> 9);

  *ptr++ = static_cast<uint8_t>(values[4] >> 1);

  *ptr = static_cast<uint8_t>(values[4] << 7);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 14);

  *ptr++ = static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 19);

  *ptr++ = static_cast<uint8_t>(values[6] >> 11);

  *ptr++ = static_cast<uint8_t>(values[6] >> 3);

  *ptr = static_cast<uint8_t>(values[6] << 5);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 16);

  *ptr++ = static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits22(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 14);

  *ptr++ = static_cast<uint8_t>(values[0] >> 6);

  *ptr = static_cast<uint8_t>(values[0] << 2);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 20);

  *ptr++ = static_cast<uint8_t>(values[1] >> 12);

  *ptr++ = static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 18);

  *ptr++ = static_cast<uint8_t>(values[2] >> 10);

  *ptr++ = static_cast<uint8_t>(values[2] >> 2);

  *ptr = static_cast<uint8_t>(values[2] << 6);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 16);

  *ptr++ = static_cast<uint8_t>(values[3] >> 8);

  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 14);

  *ptr++ = static_cast<uint8_t>(values[4] >> 6);

  *ptr = static_cast<uint8_t>(values[4] << 2);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 20);

  *ptr++ = static_cast<uint8_t>(values[5] >> 12);

  *ptr++ = static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 18);

  *ptr++ = static_cast<uint8_t>(values[6] >> 10);

  *ptr++ = static_cast<uint8_t>(values[6] >> 2);

  *ptr = static_cast<uint8_t>(values[6] << 6);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 16);

  *ptr++ = static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits23(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 15);

  *ptr++ = static_cast<uint8_t>(values[0] >> 7);

  *ptr = static_cast<uint8_t>(values[0] << 1);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 22);

  *ptr++ = static_cast<uint8_t>(values[1] >> 14);

  *ptr++ = static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 21);

  *ptr++ = static_cast<uint8_t>(values[2] >> 13);

  *ptr++ = static_cast<uint8_t>(values[2] >> 5);

  *ptr = static_cast<uint8_t>(values[2] << 3);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 20);

  *ptr++ = static_cast<uint8_t>(values[3] >> 12);

  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 19);

  *ptr++ = static_cast<uint8_t>(values[4] >> 11);

  *ptr++ = static_cast<uint8_t>(values[4] >> 3);

  *ptr = static_cast<uint8_t>(values[4] << 5);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 18);

  *ptr++ = static_cast<uint8_t>(values[5] >> 10);

  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 17);

  *ptr++ = static_cast<uint8_t>(values[6] >> 9);

  *ptr++ = static_cast<uint8_t>(values[6] >> 1);

  *ptr = static_cast<uint8_t>(values[6] << 7);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 16);

  *ptr++ = static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits24(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 16);
  *ptr++ = static_cast<uint8_t>(values[0] >> 8);
  *ptr++ = static_cast<uint8_t>(values[0]);

  *ptr++ = static_cast<uint8_t>(values[1] >> 16);
  *ptr++ = static_cast<uint8_t>(values[1] >> 8);
  *ptr++ = static_cast<uint8_t>(values[1]);

  *ptr++ = static_cast<uint8_t>(values[2] >> 16);
  *ptr++ = static_cast<uint8_t>(values[2] >> 8);
  *ptr++ = static_cast<uint8_t>(values[2]);

  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 16);
  *ptr++ = static_cast<uint8_t>(values[4] >> 8);
  *ptr++ = static_cast<uint8_t>(values[4]);

  *ptr++ = static_cast<uint8_t>(values[5] >> 16);
  *ptr++ = static_cast<uint8_t>(values[5] >> 8);
  *ptr++ = static_cast<uint8_t>(values[5]);

  *ptr++ = static_cast<uint8_t>(values[6] >> 16);
  *ptr++ = static_cast<uint8_t>(values[6] >> 8);
  *ptr++ = static_cast<uint8_t>(values[6]);

  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits25(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 17);

  *ptr++ = static_cast<uint8_t>(values[0] >> 9);

  *ptr++ = static_cast<uint8_t>(values[0] >> 1);

  *ptr = static_cast<uint8_t>(values[0] << 7);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 18);

  *ptr++ = static_cast<uint8_t>(values[1] >> 10);

  *ptr++ = static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 19);

  *ptr++ = static_cast<uint8_t>(values[2] >> 11);

  *ptr++ = static_cast<uint8_t>(values[2] >> 3);

  *ptr = static_cast<uint8_t>(values[2] << 5);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 20);

  *ptr++ = static_cast<uint8_t>(values[3] >> 12);

  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 21);

  *ptr++ = static_cast<uint8_t>(values[4] >> 13);

  *ptr++ = static_cast<uint8_t>(values[4] >> 5);

  *ptr = static_cast<uint8_t>(values[4] << 3);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 22);

  *ptr++ = static_cast<uint8_t>(values[5] >> 14);

  *ptr++ = static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 23);

  *ptr++ = static_cast<uint8_t>(values[6] >> 15);

  *ptr++ = static_cast<uint8_t>(values[6] >> 7);

  *ptr = static_cast<uint8_t>(values[6] << 1);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 24);

  *ptr++ = static_cast<uint8_t>(values[7] >> 16);

  *ptr++ = static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits26(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 18);

  *ptr++ = static_cast<uint8_t>(values[0] >> 10);

  *ptr++ = static_cast<uint8_t>(values[0] >> 2);

  *ptr = static_cast<uint8_t>(values[0] << 6);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 20);

  *ptr++ = static_cast<uint8_t>(values[1] >> 12);

  *ptr++ = static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 22);

  *ptr++ = static_cast<uint8_t>(values[2] >> 14);

  *ptr++ = static_cast<uint8_t>(values[2] >> 6);

  *ptr = static_cast<uint8_t>(values[2] << 2);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 24);

  *ptr++ = static_cast<uint8_t>(values[3] >> 16);

  *ptr++ = static_cast<uint8_t>(values[3] >> 8);

  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 18);

  *ptr++ = static_cast<uint8_t>(values[4] >> 10);

  *ptr++ = static_cast<uint8_t>(values[4] >> 2);

  *ptr = static_cast<uint8_t>(values[4] << 6);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 20);

  *ptr++ = static_cast<uint8_t>(values[5] >> 12);

  *ptr++ = static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 22);

  *ptr++ = static_cast<uint8_t>(values[6] >> 14);

  *ptr++ = static_cast<uint8_t>(values[6] >> 6);

  *ptr = static_cast<uint8_t>(values[6] << 2);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 24);

  *ptr++ = static_cast<uint8_t>(values[7] >> 16);

  *ptr++ = static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits27(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 19);

  *ptr++ = static_cast<uint8_t>(values[0] >> 11);

  *ptr++ = static_cast<uint8_t>(values[0] >> 3);

  *ptr = static_cast<uint8_t>(values[0] << 5);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 22);

  *ptr++ = static_cast<uint8_t>(values[1] >> 14);

  *ptr++ = static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 25);

  *ptr++ = static_cast<uint8_t>(values[2] >> 17);

  *ptr++ = static_cast<uint8_t>(values[2] >> 9);

  *ptr++ = static_cast<uint8_t>(values[2] >> 1);

  *ptr = static_cast<uint8_t>(values[2] << 7);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 20);

  *ptr++ = static_cast<uint8_t>(values[3] >> 12);

  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 23);

  *ptr++ = static_cast<uint8_t>(values[4] >> 15);

  *ptr++ = static_cast<uint8_t>(values[4] >> 7);

  *ptr = static_cast<uint8_t>(values[4] << 1);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 26);

  *ptr++ = static_cast<uint8_t>(values[5] >> 18);

  *ptr++ = static_cast<uint8_t>(values[5] >> 10);

  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 21);

  *ptr++ = static_cast<uint8_t>(values[6] >> 13);

  *ptr++ = static_cast<uint8_t>(values[6] >> 5);

  *ptr = static_cast<uint8_t>(values[6] << 3);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 24);

  *ptr++ = static_cast<uint8_t>(values[7] >> 16);

  *ptr++ = static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits28(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 20);
  *ptr++ = static_cast<uint8_t>(values[0] >> 12);
  *ptr++ = static_cast<uint8_t>(values[0] >> 4);
  *ptr = static_cast<uint8_t>(values[0] << 4);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 24);
  *ptr++ = static_cast<uint8_t>(values[1] >> 16);
  *ptr++ = static_cast<uint8_t>(values[1] >> 8);
  *ptr++ = static_cast<uint8_t>(values[1]);
  *ptr++ = static_cast<uint8_t>(values[2] >> 20);
  *ptr++ = static_cast<uint8_t>(values[2] >> 12);
  *ptr++ = static_cast<uint8_t>(values[2] >> 4);
  *ptr = static_cast<uint8_t>(values[2] << 4);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);
  *ptr++ = static_cast<uint8_t>(values[4] >> 20);
  *ptr++ = static_cast<uint8_t>(values[4] >> 12);
  *ptr++ = static_cast<uint8_t>(values[4] >> 4);
  *ptr = static_cast<uint8_t>(values[4] << 4);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 24);
  *ptr++ = static_cast<uint8_t>(values[5] >> 16);
  *ptr++ = static_cast<uint8_t>(values[5] >> 8);
  *ptr++ = static_cast<uint8_t>(values[5]);
  *ptr++ = static_cast<uint8_t>(values[6] >> 20);
  *ptr++ = static_cast<uint8_t>(values[6] >> 12);
  *ptr++ = static_cast<uint8_t>(values[6] >> 4);
  *ptr = static_cast<uint8_t>(values[6] << 4);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits29(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 21);

  *ptr++ = static_cast<uint8_t>(values[0] >> 13);

  *ptr++ = static_cast<uint8_t>(values[0] >> 5);

  *ptr = static_cast<uint8_t>(values[0] << 3);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 26);

  *ptr++ = static_cast<uint8_t>(values[1] >> 18);

  *ptr++ = static_cast<uint8_t>(values[1] >> 10);

  *ptr++ = static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 23);

  *ptr++ = static_cast<uint8_t>(values[2] >> 15);

  *ptr++ = static_cast<uint8_t>(values[2] >> 7);

  *ptr = static_cast<uint8_t>(values[2] << 1);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 28);

  *ptr++ = static_cast<uint8_t>(values[3] >> 20);

  *ptr++ = static_cast<uint8_t>(values[3] >> 12);

  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 25);

  *ptr++ = static_cast<uint8_t>(values[4] >> 17);

  *ptr++ = static_cast<uint8_t>(values[4] >> 9);

  *ptr++ = static_cast<uint8_t>(values[4] >> 1);

  *ptr = static_cast<uint8_t>(values[4] << 7);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 22);

  *ptr++ = static_cast<uint8_t>(values[5] >> 14);

  *ptr++ = static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 27);

  *ptr++ = static_cast<uint8_t>(values[6] >> 19);

  *ptr++ = static_cast<uint8_t>(values[6] >> 11);

  *ptr++ = static_cast<uint8_t>(values[6] >> 3);

  *ptr = static_cast<uint8_t>(values[6] << 5);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 24);

  *ptr++ = static_cast<uint8_t>(values[7] >> 16);

  *ptr++ = static_cast<uint8_t>(values[7] >> 8);

  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits30(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 22);
  *ptr++ = static_cast<uint8_t>(values[0] >> 14);
  *ptr++ = static_cast<uint8_t>(values[0] >> 6);

  *ptr = static_cast<uint8_t>(values[0] << 2);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 28);
  *ptr++ = static_cast<uint8_t>(values[1] >> 20);
  *ptr++ = static_cast<uint8_t>(values[1] >> 12);
  *ptr++ = static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 26);
  *ptr++ = static_cast<uint8_t>(values[2] >> 18);
  *ptr++ = static_cast<uint8_t>(values[2] >> 10);
  *ptr++ = static_cast<uint8_t>(values[2] >> 2);

  *ptr = static_cast<uint8_t>(values[2] << 6);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 22);
  *ptr++ = static_cast<uint8_t>(values[4] >> 14);
  *ptr++ = static_cast<uint8_t>(values[4] >> 6);

  *ptr = static_cast<uint8_t>(values[4] << 2);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 28);
  *ptr++ = static_cast<uint8_t>(values[5] >> 20);
  *ptr++ = static_cast<uint8_t>(values[5] >> 12);
  *ptr++ = static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 26);
  *ptr++ = static_cast<uint8_t>(values[6] >> 18);
  *ptr++ = static_cast<uint8_t>(values[6] >> 10);
  *ptr++ = static_cast<uint8_t>(values[6] >> 2);

  *ptr = static_cast<uint8_t>(values[6] << 6);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits31(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 23);
  *ptr++ = static_cast<uint8_t>(values[0] >> 15);
  *ptr++ = static_cast<uint8_t>(values[0] >> 7);

  *ptr = static_cast<uint8_t>(values[0] << 1);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 30);
  *ptr++ = static_cast<uint8_t>(values[1] >> 22);
  *ptr++ = static_cast<uint8_t>(values[1] >> 14);
  *ptr++ = static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 29);
  *ptr++ = static_cast<uint8_t>(values[2] >> 21);
  *ptr++ = static_cast<uint8_t>(values[2] >> 13);
  *ptr++ = static_cast<uint8_t>(values[2] >> 5);

  *ptr = static_cast<uint8_t>(values[2] << 3);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 27);
  *ptr++ = static_cast<uint8_t>(values[4] >> 19);
  *ptr++ = static_cast<uint8_t>(values[4] >> 11);
  *ptr++ = static_cast<uint8_t>(values[4] >> 3);

  *ptr = static_cast<uint8_t>(values[4] << 5);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 26);
  *ptr++ = static_cast<uint8_t>(values[5] >> 18);
  *ptr++ = static_cast<uint8_t>(values[5] >> 10);
  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 25);
  *ptr++ = static_cast<uint8_t>(values[6] >> 17);
  *ptr++ = static_cast<uint8_t>(values[6] >> 9);
  *ptr++ = static_cast<uint8_t>(values[6] >> 1);

  *ptr = static_cast<uint8_t>(values[6] << 7);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits32(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 24);
  *ptr++ = static_cast<uint8_t>(values[0] >> 16);
  *ptr++ = static_cast<uint8_t>(values[0] >> 8);
  *ptr++ = static_cast<uint8_t>(values[0]);

  *ptr++ = static_cast<uint8_t>(values[1] >> 24);
  *ptr++ = static_cast<uint8_t>(values[1] >> 16);
  *ptr++ = static_cast<uint8_t>(values[1] >> 8);
  *ptr++ = static_cast<uint8_t>(values[1]);

  *ptr++ = static_cast<uint8_t>(values[2] >> 24);
  *ptr++ = static_cast<uint8_t>(values[2] >> 16);
  *ptr++ = static_cast<uint8_t>(values[2] >> 8);
  *ptr++ = static_cast<uint8_t>(values[2]);

  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 24);
  *ptr++ = static_cast<uint8_t>(values[4] >> 16);
  *ptr++ = static_cast<uint8_t>(values[4] >> 8);
  *ptr++ = static_cast<uint8_t>(values[4]);

  *ptr++ = static_cast<uint8_t>(values[5] >> 24);
  *ptr++ = static_cast<uint8_t>(values[5] >> 16);
  *ptr++ = static_cast<uint8_t>(values[5] >> 8);
  *ptr++ = static_cast<uint8_t>(values[5]);

  *ptr++ = static_cast<uint8_t>(values[6] >> 24);
  *ptr++ = static_cast<uint8_t>(values[6] >> 16);
  *ptr++ = static_cast<uint8_t>(values[6] >> 8);
  *ptr++ = static_cast<uint8_t>(values[6]);

  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits33(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 25);
  *ptr++ = static_cast<uint8_t>(values[0] >> 17);
  *ptr++ = static_cast<uint8_t>(values[0] >> 9);
  *ptr++ = static_cast<uint8_t>(values[0] >> 1);

  *ptr = static_cast<uint8_t>(values[0] << 7);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 26);
  *ptr++ = static_cast<uint8_t>(values[1] >> 18);
  *ptr++ = static_cast<uint8_t>(values[1] >> 10);
  *ptr++ = static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 27);
  *ptr++ = static_cast<uint8_t>(values[2] >> 19);
  *ptr++ = static_cast<uint8_t>(values[2] >> 11);
  *ptr++ = static_cast<uint8_t>(values[2] >> 3);

  *ptr = static_cast<uint8_t>(values[2] << 5);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 29);
  *ptr++ = static_cast<uint8_t>(values[4] >> 21);
  *ptr++ = static_cast<uint8_t>(values[4] >> 13);
  *ptr++ = static_cast<uint8_t>(values[4] >> 5);

  *ptr = static_cast<uint8_t>(values[4] << 3);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 30);
  *ptr++ = static_cast<uint8_t>(values[5] >> 22);
  *ptr++ = static_cast<uint8_t>(values[5] >> 14);
  *ptr++ = static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 31);
  *ptr++ = static_cast<uint8_t>(values[6] >> 23);
  *ptr++ = static_cast<uint8_t>(values[6] >> 15);
  *ptr++ = static_cast<uint8_t>(values[6] >> 7);

  *ptr = static_cast<uint8_t>(values[6] << 1);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits34(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 26);
  *ptr++ = static_cast<uint8_t>(values[0] >> 18);
  *ptr++ = static_cast<uint8_t>(values[0] >> 10);
  *ptr++ = static_cast<uint8_t>(values[0] >> 2);

  *ptr = static_cast<uint8_t>(values[0] << 6);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 28);
  *ptr++ = static_cast<uint8_t>(values[1] >> 20);
  *ptr++ = static_cast<uint8_t>(values[1] >> 12);
  *ptr++ = static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 30);
  *ptr++ = static_cast<uint8_t>(values[2] >> 22);
  *ptr++ = static_cast<uint8_t>(values[2] >> 14);
  *ptr++ = static_cast<uint8_t>(values[2] >> 6);

  *ptr = static_cast<uint8_t>(values[2] << 2);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 26);
  *ptr++ = static_cast<uint8_t>(values[4] >> 18);
  *ptr++ = static_cast<uint8_t>(values[4] >> 10);
  *ptr++ = static_cast<uint8_t>(values[4] >> 2);

  *ptr = static_cast<uint8_t>(values[4] << 6);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 28);
  *ptr++ = static_cast<uint8_t>(values[5] >> 20);
  *ptr++ = static_cast<uint8_t>(values[5] >> 12);
  *ptr++ = static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 30);
  *ptr++ = static_cast<uint8_t>(values[6] >> 22);
  *ptr++ = static_cast<uint8_t>(values[6] >> 14);
  *ptr++ = static_cast<uint8_t>(values[6] >> 6);

  *ptr = static_cast<uint8_t>(values[6] << 2);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits35(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 27);
  *ptr++ = static_cast<uint8_t>(values[0] >> 19);
  *ptr++ = static_cast<uint8_t>(values[0] >> 11);
  *ptr++ = static_cast<uint8_t>(values[0] >> 3);

  *ptr = static_cast<uint8_t>(values[0] << 5);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 30);
  *ptr++ = static_cast<uint8_t>(values[1] >> 22);
  *ptr++ = static_cast<uint8_t>(values[1] >> 14);
  *ptr++ = static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 33);
  *ptr++ = static_cast<uint8_t>(values[2] >> 25);
  *ptr++ = static_cast<uint8_t>(values[2] >> 17);
  *ptr++ = static_cast<uint8_t>(values[2] >> 9);
  *ptr++ = static_cast<uint8_t>(values[2] >> 1);

  *ptr = static_cast<uint8_t>(values[2] << 7);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 31);
  *ptr++ = static_cast<uint8_t>(values[4] >> 23);
  *ptr++ = static_cast<uint8_t>(values[4] >> 15);
  *ptr++ = static_cast<uint8_t>(values[4] >> 7);

  *ptr = static_cast<uint8_t>(values[4] << 1);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 34);
  *ptr++ = static_cast<uint8_t>(values[5] >> 26);
  *ptr++ = static_cast<uint8_t>(values[5] >> 18);
  *ptr++ = static_cast<uint8_t>(values[5] >> 10);
  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 29);
  *ptr++ = static_cast<uint8_t>(values[6] >> 21);
  *ptr++ = static_cast<uint8_t>(values[6] >> 13);
  *ptr++ = static_cast<uint8_t>(values[6] >> 5);

  *ptr = static_cast<uint8_t>(values[6] << 3);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits36(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 28);
  *ptr++ = static_cast<uint8_t>(values[0] >> 20);
  *ptr++ = static_cast<uint8_t>(values[0] >> 12);
  *ptr++ = static_cast<uint8_t>(values[0] >> 4);

  *ptr = static_cast<uint8_t>(values[0] << 4);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 32);
  *ptr++ = static_cast<uint8_t>(values[1] >> 24);
  *ptr++ = static_cast<uint8_t>(values[1] >> 16);
  *ptr++ = static_cast<uint8_t>(values[1] >> 8);
  *ptr++ = static_cast<uint8_t>(values[1]);

  *ptr++ = static_cast<uint8_t>(values[2] >> 28);
  *ptr++ = static_cast<uint8_t>(values[2] >> 20);
  *ptr++ = static_cast<uint8_t>(values[2] >> 12);
  *ptr++ = static_cast<uint8_t>(values[2] >> 4);

  *ptr = static_cast<uint8_t>(values[2] << 4);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 28);
  *ptr++ = static_cast<uint8_t>(values[4] >> 20);
  *ptr++ = static_cast<uint8_t>(values[4] >> 12);
  *ptr++ = static_cast<uint8_t>(values[4] >> 4);

  *ptr = static_cast<uint8_t>(values[4] << 4);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 32);
  *ptr++ = static_cast<uint8_t>(values[5] >> 24);
  *ptr++ = static_cast<uint8_t>(values[5] >> 16);
  *ptr++ = static_cast<uint8_t>(values[5] >> 8);
  *ptr++ = static_cast<uint8_t>(values[5]);

  *ptr++ = static_cast<uint8_t>(values[6] >> 28);
  *ptr++ = static_cast<uint8_t>(values[6] >> 20);
  *ptr++ = static_cast<uint8_t>(values[6] >> 12);
  *ptr++ = static_cast<uint8_t>(values[6] >> 4);

  *ptr = static_cast<uint8_t>(values[6] << 4);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits37(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 29);
  *ptr++ = static_cast<uint8_t>(values[0] >> 21);
  *ptr++ = static_cast<uint8_t>(values[0] >> 13);
  *ptr++ = static_cast<uint8_t>(values[0] >> 5);

  *ptr = static_cast<uint8_t>(values[0] << 3);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 34);
  *ptr++ = static_cast<uint8_t>(values[1] >> 26);
  *ptr++ = static_cast<uint8_t>(values[1] >> 18);
  *ptr++ = static_cast<uint8_t>(values[1] >> 10);
  *ptr++ = static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 31);
  *ptr++ = static_cast<uint8_t>(values[2] >> 23);
  *ptr++ = static_cast<uint8_t>(values[2] >> 15);
  *ptr++ = static_cast<uint8_t>(values[2] >> 7);

  *ptr = static_cast<uint8_t>(values[2] << 1);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 33);
  *ptr++ = static_cast<uint8_t>(values[4] >> 25);
  *ptr++ = static_cast<uint8_t>(values[4] >> 17);
  *ptr++ = static_cast<uint8_t>(values[4] >> 9);
  *ptr++ = static_cast<uint8_t>(values[4] >> 1);

  *ptr = static_cast<uint8_t>(values[4] << 7);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 30);
  *ptr++ = static_cast<uint8_t>(values[5] >> 22);
  *ptr++ = static_cast<uint8_t>(values[5] >> 14);
  *ptr++ = static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 35);
  *ptr++ = static_cast<uint8_t>(values[6] >> 27);
  *ptr++ = static_cast<uint8_t>(values[6] >> 19);
  *ptr++ = static_cast<uint8_t>(values[6] >> 11);
  *ptr++ = static_cast<uint8_t>(values[6] >> 3);

  *ptr = static_cast<uint8_t>(values[6] << 5);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits38(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 30);
  *ptr++ = static_cast<uint8_t>(values[0] >> 22);
  *ptr++ = static_cast<uint8_t>(values[0] >> 14);
  *ptr++ = static_cast<uint8_t>(values[0] >> 6);

  *ptr = static_cast<uint8_t>(values[0] << 2);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 36);
  *ptr++ = static_cast<uint8_t>(values[1] >> 28);
  *ptr++ = static_cast<uint8_t>(values[1] >> 20);
  *ptr++ = static_cast<uint8_t>(values[1] >> 12);
  *ptr++ = static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 34);
  *ptr++ = static_cast<uint8_t>(values[2] >> 26);
  *ptr++ = static_cast<uint8_t>(values[2] >> 18);
  *ptr++ = static_cast<uint8_t>(values[2] >> 10);
  *ptr++ = static_cast<uint8_t>(values[2] >> 2);

  *ptr = static_cast<uint8_t>(values[2] << 6);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 30);
  *ptr++ = static_cast<uint8_t>(values[4] >> 22);
  *ptr++ = static_cast<uint8_t>(values[4] >> 14);
  *ptr++ = static_cast<uint8_t>(values[4] >> 6);

  *ptr = static_cast<uint8_t>(values[4] << 2);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 36);
  *ptr++ = static_cast<uint8_t>(values[5] >> 28);
  *ptr++ = static_cast<uint8_t>(values[5] >> 20);
  *ptr++ = static_cast<uint8_t>(values[5] >> 12);
  *ptr++ = static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 34);
  *ptr++ = static_cast<uint8_t>(values[6] >> 26);
  *ptr++ = static_cast<uint8_t>(values[6] >> 18);
  *ptr++ = static_cast<uint8_t>(values[6] >> 10);
  *ptr++ = static_cast<uint8_t>(values[6] >> 2);

  *ptr = static_cast<uint8_t>(values[6] << 6);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits39(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 31);
  *ptr++ = static_cast<uint8_t>(values[0] >> 23);
  *ptr++ = static_cast<uint8_t>(values[0] >> 15);
  *ptr++ = static_cast<uint8_t>(values[0] >> 7);

  *ptr = static_cast<uint8_t>(values[0] << 1);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 38);
  *ptr++ = static_cast<uint8_t>(values[1] >> 30);
  *ptr++ = static_cast<uint8_t>(values[1] >> 22);
  *ptr++ = static_cast<uint8_t>(values[1] >> 14);
  *ptr++ = static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 37);
  *ptr++ = static_cast<uint8_t>(values[2] >> 29);
  *ptr++ = static_cast<uint8_t>(values[2] >> 21);
  *ptr++ = static_cast<uint8_t>(values[2] >> 13);
  *ptr++ = static_cast<uint8_t>(values[2] >> 5);

  *ptr = static_cast<uint8_t>(values[2] << 3);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 35);
  *ptr++ = static_cast<uint8_t>(values[4] >> 27);
  *ptr++ = static_cast<uint8_t>(values[4] >> 19);
  *ptr++ = static_cast<uint8_t>(values[4] >> 11);
  *ptr++ = static_cast<uint8_t>(values[4] >> 3);

  *ptr = static_cast<uint8_t>(values[4] << 5);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 34);
  *ptr++ = static_cast<uint8_t>(values[5] >> 26);
  *ptr++ = static_cast<uint8_t>(values[5] >> 18);
  *ptr++ = static_cast<uint8_t>(values[5] >> 10);
  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 33);
  *ptr++ = static_cast<uint8_t>(values[6] >> 25);
  *ptr++ = static_cast<uint8_t>(values[6] >> 17);
  *ptr++ = static_cast<uint8_t>(values[6] >> 9);
  *ptr++ = static_cast<uint8_t>(values[6] >> 1);

  *ptr = static_cast<uint8_t>(values[6] << 7);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits40(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 32);
  *ptr++ = static_cast<uint8_t>(values[0] >> 24);
  *ptr++ = static_cast<uint8_t>(values[0] >> 16);
  *ptr++ = static_cast<uint8_t>(values[0] >> 8);
  *ptr++ = static_cast<uint8_t>(values[0]);

  *ptr++ = static_cast<uint8_t>(values[1] >> 32);
  *ptr++ = static_cast<uint8_t>(values[1] >> 24);
  *ptr++ = static_cast<uint8_t>(values[1] >> 16);
  *ptr++ = static_cast<uint8_t>(values[1] >> 8);
  *ptr++ = static_cast<uint8_t>(values[1]);

  *ptr++ = static_cast<uint8_t>(values[2] >> 32);
  *ptr++ = static_cast<uint8_t>(values[2] >> 24);
  *ptr++ = static_cast<uint8_t>(values[2] >> 16);
  *ptr++ = static_cast<uint8_t>(values[2] >> 8);
  *ptr++ = static_cast<uint8_t>(values[2]);

  *ptr++ = static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 32);
  *ptr++ = static_cast<uint8_t>(values[4] >> 24);
  *ptr++ = static_cast<uint8_t>(values[4] >> 16);
  *ptr++ = static_cast<uint8_t>(values[4] >> 8);
  *ptr++ = static_cast<uint8_t>(values[4]);

  *ptr++ = static_cast<uint8_t>(values[5] >> 32);
  *ptr++ = static_cast<uint8_t>(values[5] >> 24);
  *ptr++ = static_cast<uint8_t>(values[5] >> 16);
  *ptr++ = static_cast<uint8_t>(values[5] >> 8);
  *ptr++ = static_cast<uint8_t>(values[5]);

  *ptr++ = static_cast<uint8_t>(values[6] >> 32);
  *ptr++ = static_cast<uint8_t>(values[6] >> 24);
  *ptr++ = static_cast<uint8_t>(values[6] >> 16);
  *ptr++ = static_cast<uint8_t>(values[6] >> 8);
  *ptr++ = static_cast<uint8_t>(values[6]);

  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits41(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 33);
  *ptr++ = static_cast<uint8_t>(values[0] >> 25);
  *ptr++ = static_cast<uint8_t>(values[0] >> 17);
  *ptr++ = static_cast<uint8_t>(values[0] >> 9);
  *ptr++ = static_cast<uint8_t>(values[0] >> 1);

  *ptr = static_cast<uint8_t>(values[0] << 7);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 34);
  *ptr++ = static_cast<uint8_t>(values[1] >> 26);
  *ptr++ = static_cast<uint8_t>(values[1] >> 18);
  *ptr++ = static_cast<uint8_t>(values[1] >> 10);
  *ptr++ = static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 35);
  *ptr++ = static_cast<uint8_t>(values[2] >> 27);
  *ptr++ = static_cast<uint8_t>(values[2] >> 19);
  *ptr++ = static_cast<uint8_t>(values[2] >> 11);
  *ptr++ = static_cast<uint8_t>(values[2] >> 3);

  *ptr = static_cast<uint8_t>(values[2] << 5);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 37);
  *ptr++ = static_cast<uint8_t>(values[4] >> 29);
  *ptr++ = static_cast<uint8_t>(values[4] >> 21);
  *ptr++ = static_cast<uint8_t>(values[4] >> 13);
  *ptr++ = static_cast<uint8_t>(values[4] >> 5);

  *ptr = static_cast<uint8_t>(values[4] << 3);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 38);
  *ptr++ = static_cast<uint8_t>(values[5] >> 30);
  *ptr++ = static_cast<uint8_t>(values[5] >> 22);
  *ptr++ = static_cast<uint8_t>(values[5] >> 14);
  *ptr++ = static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 39);
  *ptr++ = static_cast<uint8_t>(values[6] >> 31);
  *ptr++ = static_cast<uint8_t>(values[6] >> 23);
  *ptr++ = static_cast<uint8_t>(values[6] >> 15);
  *ptr++ = static_cast<uint8_t>(values[6] >> 7);

  *ptr = static_cast<uint8_t>(values[6] << 1);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits42(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 34);
  *ptr++ = static_cast<uint8_t>(values[0] >> 26);
  *ptr++ = static_cast<uint8_t>(values[0] >> 18);
  *ptr++ = static_cast<uint8_t>(values[0] >> 10);
  *ptr++ = static_cast<uint8_t>(values[0] >> 2);

  *ptr = static_cast<uint8_t>(values[0] << 6);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 36);
  *ptr++ = static_cast<uint8_t>(values[1] >> 28);
  *ptr++ = static_cast<uint8_t>(values[1] >> 20);
  *ptr++ = static_cast<uint8_t>(values[1] >> 12);
  *ptr++ = static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 38);
  *ptr++ = static_cast<uint8_t>(values[2] >> 30);
  *ptr++ = static_cast<uint8_t>(values[2] >> 22);
  *ptr++ = static_cast<uint8_t>(values[2] >> 14);
  *ptr++ = static_cast<uint8_t>(values[2] >> 6);

  *ptr = static_cast<uint8_t>(values[2] << 2);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 40);
  *ptr++ = static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 34);
  *ptr++ = static_cast<uint8_t>(values[4] >> 26);
  *ptr++ = static_cast<uint8_t>(values[4] >> 18);
  *ptr++ = static_cast<uint8_t>(values[4] >> 10);
  *ptr++ = static_cast<uint8_t>(values[4] >> 2);

  *ptr = static_cast<uint8_t>(values[4] << 6);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 36);
  *ptr++ = static_cast<uint8_t>(values[5] >> 28);
  *ptr++ = static_cast<uint8_t>(values[5] >> 20);
  *ptr++ = static_cast<uint8_t>(values[5] >> 12);
  *ptr++ = static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 38);
  *ptr++ = static_cast<uint8_t>(values[6] >> 30);
  *ptr++ = static_cast<uint8_t>(values[6] >> 22);
  *ptr++ = static_cast<uint8_t>(values[6] >> 14);
  *ptr++ = static_cast<uint8_t>(values[6] >> 6);

  *ptr = static_cast<uint8_t>(values[6] << 2);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits43(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 35);
  *ptr++ = static_cast<uint8_t>(values[0] >> 27);
  *ptr++ = static_cast<uint8_t>(values[0] >> 19);
  *ptr++ = static_cast<uint8_t>(values[0] >> 11);
  *ptr++ = static_cast<uint8_t>(values[0] >> 3);

  *ptr = static_cast<uint8_t>(values[0] << 5);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 38);
  *ptr++ = static_cast<uint8_t>(values[1] >> 30);
  *ptr++ = static_cast<uint8_t>(values[1] >> 22);
  *ptr++ = static_cast<uint8_t>(values[1] >> 14);
  *ptr++ = static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 41);
  *ptr++ = static_cast<uint8_t>(values[2] >> 33);
  *ptr++ = static_cast<uint8_t>(values[2] >> 25);
  *ptr++ = static_cast<uint8_t>(values[2] >> 17);
  *ptr++ = static_cast<uint8_t>(values[2] >> 9);
  *ptr++ = static_cast<uint8_t>(values[2] >> 1);

  *ptr = static_cast<uint8_t>(values[2] << 7);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 39);
  *ptr++ = static_cast<uint8_t>(values[4] >> 31);
  *ptr++ = static_cast<uint8_t>(values[4] >> 23);
  *ptr++ = static_cast<uint8_t>(values[4] >> 15);
  *ptr++ = static_cast<uint8_t>(values[4] >> 7);

  *ptr = static_cast<uint8_t>(values[4] << 1);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 42);
  *ptr++ = static_cast<uint8_t>(values[5] >> 34);
  *ptr++ = static_cast<uint8_t>(values[5] >> 26);
  *ptr++ = static_cast<uint8_t>(values[5] >> 18);
  *ptr++ = static_cast<uint8_t>(values[5] >> 10);
  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 37);
  *ptr++ = static_cast<uint8_t>(values[6] >> 29);
  *ptr++ = static_cast<uint8_t>(values[6] >> 21);
  *ptr++ = static_cast<uint8_t>(values[6] >> 13);
  *ptr++ = static_cast<uint8_t>(values[6] >> 5);

  *ptr = static_cast<uint8_t>(values[6] << 3);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits44(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 36);
  *ptr++ = static_cast<uint8_t>(values[0] >> 28);
  *ptr++ = static_cast<uint8_t>(values[0] >> 20);
  *ptr++ = static_cast<uint8_t>(values[0] >> 12);
  *ptr++ = static_cast<uint8_t>(values[0] >> 4);

  *ptr = static_cast<uint8_t>(values[0] << 4);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 40);
  *ptr++ = static_cast<uint8_t>(values[1] >> 32);
  *ptr++ = static_cast<uint8_t>(values[1] >> 24);
  *ptr++ = static_cast<uint8_t>(values[1] >> 16);
  *ptr++ = static_cast<uint8_t>(values[1] >> 8);
  *ptr++ = static_cast<uint8_t>(values[1]);

  *ptr++ = static_cast<uint8_t>(values[2] >> 36);
  *ptr++ = static_cast<uint8_t>(values[2] >> 28);
  *ptr++ = static_cast<uint8_t>(values[2] >> 20);
  *ptr++ = static_cast<uint8_t>(values[2] >> 12);
  *ptr++ = static_cast<uint8_t>(values[2] >> 4);

  *ptr = static_cast<uint8_t>(values[2] << 4);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 40);
  *ptr++ = static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 36);
  *ptr++ = static_cast<uint8_t>(values[4] >> 28);
  *ptr++ = static_cast<uint8_t>(values[4] >> 20);
  *ptr++ = static_cast<uint8_t>(values[4] >> 12);
  *ptr++ = static_cast<uint8_t>(values[4] >> 4);

  *ptr = static_cast<uint8_t>(values[4] << 4);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 40);
  *ptr++ = static_cast<uint8_t>(values[5] >> 32);
  *ptr++ = static_cast<uint8_t>(values[5] >> 24);
  *ptr++ = static_cast<uint8_t>(values[5] >> 16);
  *ptr++ = static_cast<uint8_t>(values[5] >> 8);
  *ptr++ = static_cast<uint8_t>(values[5]);

  *ptr++ = static_cast<uint8_t>(values[6] >> 36);
  *ptr++ = static_cast<uint8_t>(values[6] >> 28);
  *ptr++ = static_cast<uint8_t>(values[6] >> 20);
  *ptr++ = static_cast<uint8_t>(values[6] >> 12);
  *ptr++ = static_cast<uint8_t>(values[6] >> 4);

  *ptr = static_cast<uint8_t>(values[6] << 4);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits45(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 37);
  *ptr++ = static_cast<uint8_t>(values[0] >> 29);
  *ptr++ = static_cast<uint8_t>(values[0] >> 21);
  *ptr++ = static_cast<uint8_t>(values[0] >> 13);
  *ptr++ = static_cast<uint8_t>(values[0] >> 5);

  *ptr = static_cast<uint8_t>(values[0] << 3);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 42);
  *ptr++ = static_cast<uint8_t>(values[1] >> 34);
  *ptr++ = static_cast<uint8_t>(values[1] >> 26);
  *ptr++ = static_cast<uint8_t>(values[1] >> 18);
  *ptr++ = static_cast<uint8_t>(values[1] >> 10);
  *ptr++ = static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 39);
  *ptr++ = static_cast<uint8_t>(values[2] >> 31);
  *ptr++ = static_cast<uint8_t>(values[2] >> 23);
  *ptr++ = static_cast<uint8_t>(values[2] >> 15);
  *ptr++ = static_cast<uint8_t>(values[2] >> 7);

  *ptr = static_cast<uint8_t>(values[2] << 1);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 44);
  *ptr++ = static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 41);
  *ptr++ = static_cast<uint8_t>(values[4] >> 33);
  *ptr++ = static_cast<uint8_t>(values[4] >> 25);
  *ptr++ = static_cast<uint8_t>(values[4] >> 17);
  *ptr++ = static_cast<uint8_t>(values[4] >> 9);
  *ptr++ = static_cast<uint8_t>(values[4] >> 1);

  *ptr = static_cast<uint8_t>(values[4] << 7);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 38);
  *ptr++ = static_cast<uint8_t>(values[5] >> 30);
  *ptr++ = static_cast<uint8_t>(values[5] >> 22);
  *ptr++ = static_cast<uint8_t>(values[5] >> 14);
  *ptr++ = static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 43);
  *ptr++ = static_cast<uint8_t>(values[6] >> 35);
  *ptr++ = static_cast<uint8_t>(values[6] >> 27);
  *ptr++ = static_cast<uint8_t>(values[6] >> 19);
  *ptr++ = static_cast<uint8_t>(values[6] >> 11);
  *ptr++ = static_cast<uint8_t>(values[6] >> 3);

  *ptr = static_cast<uint8_t>(values[6] << 5);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits46(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 38);
  *ptr++ = static_cast<uint8_t>(values[0] >> 30);
  *ptr++ = static_cast<uint8_t>(values[0] >> 22);
  *ptr++ = static_cast<uint8_t>(values[0] >> 14);
  *ptr++ = static_cast<uint8_t>(values[0] >> 6);

  *ptr = static_cast<uint8_t>(values[0] << 2);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 44);
  *ptr++ = static_cast<uint8_t>(values[1] >> 36);
  *ptr++ = static_cast<uint8_t>(values[1] >> 28);
  *ptr++ = static_cast<uint8_t>(values[1] >> 20);
  *ptr++ = static_cast<uint8_t>(values[1] >> 12);
  *ptr++ = static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 42);
  *ptr++ = static_cast<uint8_t>(values[2] >> 34);
  *ptr++ = static_cast<uint8_t>(values[2] >> 26);
  *ptr++ = static_cast<uint8_t>(values[2] >> 18);
  *ptr++ = static_cast<uint8_t>(values[2] >> 10);
  *ptr++ = static_cast<uint8_t>(values[2] >> 2);

  *ptr = static_cast<uint8_t>(values[2] << 6);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 40);
  *ptr++ = static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 38);
  *ptr++ = static_cast<uint8_t>(values[4] >> 30);
  *ptr++ = static_cast<uint8_t>(values[4] >> 22);
  *ptr++ = static_cast<uint8_t>(values[4] >> 14);
  *ptr++ = static_cast<uint8_t>(values[4] >> 6);

  *ptr = static_cast<uint8_t>(values[4] << 2);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 44);
  *ptr++ = static_cast<uint8_t>(values[5] >> 36);
  *ptr++ = static_cast<uint8_t>(values[5] >> 28);
  *ptr++ = static_cast<uint8_t>(values[5] >> 20);
  *ptr++ = static_cast<uint8_t>(values[5] >> 12);
  *ptr++ = static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 42);
  *ptr++ = static_cast<uint8_t>(values[6] >> 34);
  *ptr++ = static_cast<uint8_t>(values[6] >> 26);
  *ptr++ = static_cast<uint8_t>(values[6] >> 18);
  *ptr++ = static_cast<uint8_t>(values[6] >> 10);
  *ptr++ = static_cast<uint8_t>(values[6] >> 2);

  *ptr = static_cast<uint8_t>(values[6] << 6);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits47(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 39);
  *ptr++ = static_cast<uint8_t>(values[0] >> 31);
  *ptr++ = static_cast<uint8_t>(values[0] >> 23);
  *ptr++ = static_cast<uint8_t>(values[0] >> 15);
  *ptr++ = static_cast<uint8_t>(values[0] >> 7);

  *ptr = static_cast<uint8_t>(values[0] << 1);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 46);
  *ptr++ = static_cast<uint8_t>(values[1] >> 38);
  *ptr++ = static_cast<uint8_t>(values[1] >> 30);
  *ptr++ = static_cast<uint8_t>(values[1] >> 22);
  *ptr++ = static_cast<uint8_t>(values[1] >> 14);
  *ptr++ = static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 45);
  *ptr++ = static_cast<uint8_t>(values[2] >> 37);
  *ptr++ = static_cast<uint8_t>(values[2] >> 29);
  *ptr++ = static_cast<uint8_t>(values[2] >> 21);
  *ptr++ = static_cast<uint8_t>(values[2] >> 13);
  *ptr++ = static_cast<uint8_t>(values[2] >> 5);

  *ptr = static_cast<uint8_t>(values[2] << 3);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 44);
  *ptr++ = static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 43);
  *ptr++ = static_cast<uint8_t>(values[4] >> 35);
  *ptr++ = static_cast<uint8_t>(values[4] >> 27);
  *ptr++ = static_cast<uint8_t>(values[4] >> 19);
  *ptr++ = static_cast<uint8_t>(values[4] >> 11);
  *ptr++ = static_cast<uint8_t>(values[4] >> 3);

  *ptr = static_cast<uint8_t>(values[4] << 5);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 42);
  *ptr++ = static_cast<uint8_t>(values[5] >> 34);
  *ptr++ = static_cast<uint8_t>(values[5] >> 26);
  *ptr++ = static_cast<uint8_t>(values[5] >> 18);
  *ptr++ = static_cast<uint8_t>(values[5] >> 10);
  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 41);
  *ptr++ = static_cast<uint8_t>(values[6] >> 33);
  *ptr++ = static_cast<uint8_t>(values[6] >> 25);
  *ptr++ = static_cast<uint8_t>(values[6] >> 17);
  *ptr++ = static_cast<uint8_t>(values[6] >> 9);
  *ptr++ = static_cast<uint8_t>(values[6] >> 1);

  *ptr = static_cast<uint8_t>(values[6] << 7);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits48(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 40);
  *ptr++ = static_cast<uint8_t>(values[0] >> 32);
  *ptr++ = static_cast<uint8_t>(values[0] >> 24);
  *ptr++ = static_cast<uint8_t>(values[0] >> 16);
  *ptr++ = static_cast<uint8_t>(values[0] >> 8);
  *ptr++ = static_cast<uint8_t>(values[0]);

  *ptr++ = static_cast<uint8_t>(values[1] >> 40);
  *ptr++ = static_cast<uint8_t>(values[1] >> 32);
  *ptr++ = static_cast<uint8_t>(values[1] >> 24);
  *ptr++ = static_cast<uint8_t>(values[1] >> 16);
  *ptr++ = static_cast<uint8_t>(values[1] >> 8);
  *ptr++ = static_cast<uint8_t>(values[1]);

  *ptr++ = static_cast<uint8_t>(values[2] >> 40);
  *ptr++ = static_cast<uint8_t>(values[2] >> 32);
  *ptr++ = static_cast<uint8_t>(values[2] >> 24);
  *ptr++ = static_cast<uint8_t>(values[2] >> 16);
  *ptr++ = static_cast<uint8_t>(values[2] >> 8);
  *ptr++ = static_cast<uint8_t>(values[2]);

  *ptr++ = static_cast<uint8_t>(values[3] >> 40);
  *ptr++ = static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 40);
  *ptr++ = static_cast<uint8_t>(values[4] >> 32);
  *ptr++ = static_cast<uint8_t>(values[4] >> 24);
  *ptr++ = static_cast<uint8_t>(values[4] >> 16);
  *ptr++ = static_cast<uint8_t>(values[4] >> 8);
  *ptr++ = static_cast<uint8_t>(values[4]);

  *ptr++ = static_cast<uint8_t>(values[5] >> 40);
  *ptr++ = static_cast<uint8_t>(values[5] >> 32);
  *ptr++ = static_cast<uint8_t>(values[5] >> 24);
  *ptr++ = static_cast<uint8_t>(values[5] >> 16);
  *ptr++ = static_cast<uint8_t>(values[5] >> 8);
  *ptr++ = static_cast<uint8_t>(values[5]);

  *ptr++ = static_cast<uint8_t>(values[6] >> 40);
  *ptr++ = static_cast<uint8_t>(values[6] >> 32);
  *ptr++ = static_cast<uint8_t>(values[6] >> 24);
  *ptr++ = static_cast<uint8_t>(values[6] >> 16);
  *ptr++ = static_cast<uint8_t>(values[6] >> 8);
  *ptr++ = static_cast<uint8_t>(values[6]);

  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits49(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 41);
  *ptr++ = static_cast<uint8_t>(values[0] >> 33);
  *ptr++ = static_cast<uint8_t>(values[0] >> 25);
  *ptr++ = static_cast<uint8_t>(values[0] >> 17);
  *ptr++ = static_cast<uint8_t>(values[0] >> 9);
  *ptr++ = static_cast<uint8_t>(values[0] >> 1);

  *ptr = static_cast<uint8_t>(values[0] << 7);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 42);
  *ptr++ = static_cast<uint8_t>(values[1] >> 34);
  *ptr++ = static_cast<uint8_t>(values[1] >> 26);
  *ptr++ = static_cast<uint8_t>(values[1] >> 18);
  *ptr++ = static_cast<uint8_t>(values[1] >> 10);
  *ptr++ = static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 43);
  *ptr++ = static_cast<uint8_t>(values[2] >> 35);
  *ptr++ = static_cast<uint8_t>(values[2] >> 27);
  *ptr++ = static_cast<uint8_t>(values[2] >> 19);
  *ptr++ = static_cast<uint8_t>(values[2] >> 11);
  *ptr++ = static_cast<uint8_t>(values[2] >> 3);

  *ptr = static_cast<uint8_t>(values[2] << 5);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 44);
  *ptr++ = static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 45);
  *ptr++ = static_cast<uint8_t>(values[4] >> 37);
  *ptr++ = static_cast<uint8_t>(values[4] >> 29);
  *ptr++ = static_cast<uint8_t>(values[4] >> 21);
  *ptr++ = static_cast<uint8_t>(values[4] >> 13);
  *ptr++ = static_cast<uint8_t>(values[4] >> 5);

  *ptr = static_cast<uint8_t>(values[4] << 3);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 46);
  *ptr++ = static_cast<uint8_t>(values[5] >> 38);
  *ptr++ = static_cast<uint8_t>(values[5] >> 30);
  *ptr++ = static_cast<uint8_t>(values[5] >> 22);
  *ptr++ = static_cast<uint8_t>(values[5] >> 14);
  *ptr++ = static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 47);
  *ptr++ = static_cast<uint8_t>(values[6] >> 39);
  *ptr++ = static_cast<uint8_t>(values[6] >> 31);
  *ptr++ = static_cast<uint8_t>(values[6] >> 23);
  *ptr++ = static_cast<uint8_t>(values[6] >> 15);
  *ptr++ = static_cast<uint8_t>(values[6] >> 7);

  *ptr = static_cast<uint8_t>(values[6] << 1);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits50(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 42);
  *ptr++ = static_cast<uint8_t>(values[0] >> 34);
  *ptr++ = static_cast<uint8_t>(values[0] >> 26);
  *ptr++ = static_cast<uint8_t>(values[0] >> 18);
  *ptr++ = static_cast<uint8_t>(values[0] >> 10);
  *ptr++ = static_cast<uint8_t>(values[0] >> 2);

  *ptr = static_cast<uint8_t>(values[0] << 6);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 44);
  *ptr++ = static_cast<uint8_t>(values[1] >> 36);
  *ptr++ = static_cast<uint8_t>(values[1] >> 28);
  *ptr++ = static_cast<uint8_t>(values[1] >> 20);
  *ptr++ = static_cast<uint8_t>(values[1] >> 12);
  *ptr++ = static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 46);
  *ptr++ = static_cast<uint8_t>(values[2] >> 38);
  *ptr++ = static_cast<uint8_t>(values[2] >> 30);
  *ptr++ = static_cast<uint8_t>(values[2] >> 22);
  *ptr++ = static_cast<uint8_t>(values[2] >> 14);
  *ptr++ = static_cast<uint8_t>(values[2] >> 6);

  *ptr = static_cast<uint8_t>(values[2] << 2);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 48);
  *ptr++ = static_cast<uint8_t>(values[3] >> 40);
  *ptr++ = static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 42);
  *ptr++ = static_cast<uint8_t>(values[4] >> 34);
  *ptr++ = static_cast<uint8_t>(values[4] >> 26);
  *ptr++ = static_cast<uint8_t>(values[4] >> 18);
  *ptr++ = static_cast<uint8_t>(values[4] >> 10);
  *ptr++ = static_cast<uint8_t>(values[4] >> 2);

  *ptr = static_cast<uint8_t>(values[4] << 6);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 44);
  *ptr++ = static_cast<uint8_t>(values[5] >> 36);
  *ptr++ = static_cast<uint8_t>(values[5] >> 28);
  *ptr++ = static_cast<uint8_t>(values[5] >> 20);
  *ptr++ = static_cast<uint8_t>(values[5] >> 12);
  *ptr++ = static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 46);
  *ptr++ = static_cast<uint8_t>(values[6] >> 38);
  *ptr++ = static_cast<uint8_t>(values[6] >> 30);
  *ptr++ = static_cast<uint8_t>(values[6] >> 22);
  *ptr++ = static_cast<uint8_t>(values[6] >> 14);
  *ptr++ = static_cast<uint8_t>(values[6] >> 6);

  *ptr = static_cast<uint8_t>(values[6] << 2);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits51(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 43);
  *ptr++ = static_cast<uint8_t>(values[0] >> 35);
  *ptr++ = static_cast<uint8_t>(values[0] >> 27);
  *ptr++ = static_cast<uint8_t>(values[0] >> 19);
  *ptr++ = static_cast<uint8_t>(values[0] >> 11);
  *ptr++ = static_cast<uint8_t>(values[0] >> 3);

  *ptr = static_cast<uint8_t>(values[0] << 5);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 46);
  *ptr++ = static_cast<uint8_t>(values[1] >> 38);
  *ptr++ = static_cast<uint8_t>(values[1] >> 30);
  *ptr++ = static_cast<uint8_t>(values[1] >> 22);
  *ptr++ = static_cast<uint8_t>(values[1] >> 14);
  *ptr++ = static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 49);
  *ptr++ = static_cast<uint8_t>(values[2] >> 41);
  *ptr++ = static_cast<uint8_t>(values[2] >> 33);
  *ptr++ = static_cast<uint8_t>(values[2] >> 25);
  *ptr++ = static_cast<uint8_t>(values[2] >> 17);
  *ptr++ = static_cast<uint8_t>(values[2] >> 9);
  *ptr++ = static_cast<uint8_t>(values[2] >> 1);

  *ptr = static_cast<uint8_t>(values[2] << 7);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 44);
  *ptr++ = static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 47);
  *ptr++ = static_cast<uint8_t>(values[4] >> 39);
  *ptr++ = static_cast<uint8_t>(values[4] >> 31);
  *ptr++ = static_cast<uint8_t>(values[4] >> 23);
  *ptr++ = static_cast<uint8_t>(values[4] >> 15);
  *ptr++ = static_cast<uint8_t>(values[4] >> 7);

  *ptr = static_cast<uint8_t>(values[4] << 1);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 50);
  *ptr++ = static_cast<uint8_t>(values[5] >> 42);
  *ptr++ = static_cast<uint8_t>(values[5] >> 34);
  *ptr++ = static_cast<uint8_t>(values[5] >> 26);
  *ptr++ = static_cast<uint8_t>(values[5] >> 18);
  *ptr++ = static_cast<uint8_t>(values[5] >> 10);
  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 45);
  *ptr++ = static_cast<uint8_t>(values[6] >> 37);
  *ptr++ = static_cast<uint8_t>(values[6] >> 29);
  *ptr++ = static_cast<uint8_t>(values[6] >> 21);
  *ptr++ = static_cast<uint8_t>(values[6] >> 13);
  *ptr++ = static_cast<uint8_t>(values[6] >> 5);

  *ptr = static_cast<uint8_t>(values[6] << 3);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits52(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 44);
  *ptr++ = static_cast<uint8_t>(values[0] >> 36);
  *ptr++ = static_cast<uint8_t>(values[0] >> 28);
  *ptr++ = static_cast<uint8_t>(values[0] >> 20);
  *ptr++ = static_cast<uint8_t>(values[0] >> 12);
  *ptr++ = static_cast<uint8_t>(values[0] >> 4);

  *ptr = static_cast<uint8_t>(values[0] << 4);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 48);
  *ptr++ = static_cast<uint8_t>(values[1] >> 40);
  *ptr++ = static_cast<uint8_t>(values[1] >> 32);
  *ptr++ = static_cast<uint8_t>(values[1] >> 24);
  *ptr++ = static_cast<uint8_t>(values[1] >> 16);
  *ptr++ = static_cast<uint8_t>(values[1] >> 8);
  *ptr++ = static_cast<uint8_t>(values[1]);

  *ptr++ = static_cast<uint8_t>(values[2] >> 44);
  *ptr++ = static_cast<uint8_t>(values[2] >> 36);
  *ptr++ = static_cast<uint8_t>(values[2] >> 28);
  *ptr++ = static_cast<uint8_t>(values[2] >> 20);
  *ptr++ = static_cast<uint8_t>(values[2] >> 12);
  *ptr++ = static_cast<uint8_t>(values[2] >> 4);

  *ptr = static_cast<uint8_t>(values[2] << 4);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 48);
  *ptr++ = static_cast<uint8_t>(values[3] >> 40);
  *ptr++ = static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 44);
  *ptr++ = static_cast<uint8_t>(values[4] >> 36);
  *ptr++ = static_cast<uint8_t>(values[4] >> 28);
  *ptr++ = static_cast<uint8_t>(values[4] >> 20);
  *ptr++ = static_cast<uint8_t>(values[4] >> 12);
  *ptr++ = static_cast<uint8_t>(values[4] >> 4);

  *ptr = static_cast<uint8_t>(values[4] << 4);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 48);
  *ptr++ = static_cast<uint8_t>(values[5] >> 40);
  *ptr++ = static_cast<uint8_t>(values[5] >> 32);
  *ptr++ = static_cast<uint8_t>(values[5] >> 24);
  *ptr++ = static_cast<uint8_t>(values[5] >> 16);
  *ptr++ = static_cast<uint8_t>(values[5] >> 8);
  *ptr++ = static_cast<uint8_t>(values[5]);

  *ptr++ = static_cast<uint8_t>(values[6] >> 44);
  *ptr++ = static_cast<uint8_t>(values[6] >> 36);
  *ptr++ = static_cast<uint8_t>(values[6] >> 28);
  *ptr++ = static_cast<uint8_t>(values[6] >> 20);
  *ptr++ = static_cast<uint8_t>(values[6] >> 12);
  *ptr++ = static_cast<uint8_t>(values[6] >> 4);

  *ptr = static_cast<uint8_t>(values[6] << 4);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits53(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 45);
  *ptr++ = static_cast<uint8_t>(values[0] >> 37);
  *ptr++ = static_cast<uint8_t>(values[0] >> 29);
  *ptr++ = static_cast<uint8_t>(values[0] >> 21);
  *ptr++ = static_cast<uint8_t>(values[0] >> 13);
  *ptr++ = static_cast<uint8_t>(values[0] >> 5);

  *ptr = static_cast<uint8_t>(values[0] << 3);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 50);
  *ptr++ = static_cast<uint8_t>(values[1] >> 42);
  *ptr++ = static_cast<uint8_t>(values[1] >> 34);
  *ptr++ = static_cast<uint8_t>(values[1] >> 26);
  *ptr++ = static_cast<uint8_t>(values[1] >> 18);
  *ptr++ = static_cast<uint8_t>(values[1] >> 10);
  *ptr++ = static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 47);
  *ptr++ = static_cast<uint8_t>(values[2] >> 39);
  *ptr++ = static_cast<uint8_t>(values[2] >> 31);
  *ptr++ = static_cast<uint8_t>(values[2] >> 23);
  *ptr++ = static_cast<uint8_t>(values[2] >> 15);
  *ptr++ = static_cast<uint8_t>(values[2] >> 7);

  *ptr = static_cast<uint8_t>(values[2] << 1);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 52);
  *ptr++ = static_cast<uint8_t>(values[3] >> 44);
  *ptr++ = static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 49);
  *ptr++ = static_cast<uint8_t>(values[4] >> 41);
  *ptr++ = static_cast<uint8_t>(values[4] >> 33);
  *ptr++ = static_cast<uint8_t>(values[4] >> 25);
  *ptr++ = static_cast<uint8_t>(values[4] >> 17);
  *ptr++ = static_cast<uint8_t>(values[4] >> 9);
  *ptr++ = static_cast<uint8_t>(values[4] >> 1);

  *ptr = static_cast<uint8_t>(values[4] << 7);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 46);
  *ptr++ = static_cast<uint8_t>(values[5] >> 38);
  *ptr++ = static_cast<uint8_t>(values[5] >> 30);
  *ptr++ = static_cast<uint8_t>(values[5] >> 22);
  *ptr++ = static_cast<uint8_t>(values[5] >> 14);
  *ptr++ = static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 51);
  *ptr++ = static_cast<uint8_t>(values[6] >> 43);
  *ptr++ = static_cast<uint8_t>(values[6] >> 35);
  *ptr++ = static_cast<uint8_t>(values[6] >> 27);
  *ptr++ = static_cast<uint8_t>(values[6] >> 19);
  *ptr++ = static_cast<uint8_t>(values[6] >> 11);
  *ptr++ = static_cast<uint8_t>(values[6] >> 3);

  *ptr = static_cast<uint8_t>(values[6] << 5);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits54(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 46);
  *ptr++ = static_cast<uint8_t>(values[0] >> 38);
  *ptr++ = static_cast<uint8_t>(values[0] >> 30);
  *ptr++ = static_cast<uint8_t>(values[0] >> 22);
  *ptr++ = static_cast<uint8_t>(values[0] >> 14);
  *ptr++ = static_cast<uint8_t>(values[0] >> 6);

  *ptr = static_cast<uint8_t>(values[0] << 2);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 52);
  *ptr++ = static_cast<uint8_t>(values[1] >> 44);
  *ptr++ = static_cast<uint8_t>(values[1] >> 36);
  *ptr++ = static_cast<uint8_t>(values[1] >> 28);
  *ptr++ = static_cast<uint8_t>(values[1] >> 20);
  *ptr++ = static_cast<uint8_t>(values[1] >> 12);
  *ptr++ = static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 50);
  *ptr++ = static_cast<uint8_t>(values[2] >> 42);
  *ptr++ = static_cast<uint8_t>(values[2] >> 34);
  *ptr++ = static_cast<uint8_t>(values[2] >> 26);
  *ptr++ = static_cast<uint8_t>(values[2] >> 18);
  *ptr++ = static_cast<uint8_t>(values[2] >> 10);
  *ptr++ = static_cast<uint8_t>(values[2] >> 2);

  *ptr = static_cast<uint8_t>(values[2] << 6);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 48);
  *ptr++ = static_cast<uint8_t>(values[3] >> 40);
  *ptr++ = static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 46);
  *ptr++ = static_cast<uint8_t>(values[4] >> 38);
  *ptr++ = static_cast<uint8_t>(values[4] >> 30);
  *ptr++ = static_cast<uint8_t>(values[4] >> 22);
  *ptr++ = static_cast<uint8_t>(values[4] >> 14);
  *ptr++ = static_cast<uint8_t>(values[4] >> 6);

  *ptr = static_cast<uint8_t>(values[4] << 2);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 52);
  *ptr++ = static_cast<uint8_t>(values[5] >> 44);
  *ptr++ = static_cast<uint8_t>(values[5] >> 36);
  *ptr++ = static_cast<uint8_t>(values[5] >> 28);
  *ptr++ = static_cast<uint8_t>(values[5] >> 20);
  *ptr++ = static_cast<uint8_t>(values[5] >> 12);
  *ptr++ = static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 50);
  *ptr++ = static_cast<uint8_t>(values[6] >> 42);
  *ptr++ = static_cast<uint8_t>(values[6] >> 34);
  *ptr++ = static_cast<uint8_t>(values[6] >> 26);
  *ptr++ = static_cast<uint8_t>(values[6] >> 18);
  *ptr++ = static_cast<uint8_t>(values[6] >> 10);
  *ptr++ = static_cast<uint8_t>(values[6] >> 2);

  *ptr = static_cast<uint8_t>(values[6] << 6);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits55(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 47);
  *ptr++ = static_cast<uint8_t>(values[0] >> 39);
  *ptr++ = static_cast<uint8_t>(values[0] >> 31);
  *ptr++ = static_cast<uint8_t>(values[0] >> 23);
  *ptr++ = static_cast<uint8_t>(values[0] >> 15);
  *ptr++ = static_cast<uint8_t>(values[0] >> 7);

  *ptr = static_cast<uint8_t>(values[0] << 1);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 54);
  *ptr++ = static_cast<uint8_t>(values[1] >> 46);
  *ptr++ = static_cast<uint8_t>(values[1] >> 38);
  *ptr++ = static_cast<uint8_t>(values[1] >> 30);
  *ptr++ = static_cast<uint8_t>(values[1] >> 22);
  *ptr++ = static_cast<uint8_t>(values[1] >> 14);
  *ptr++ = static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 53);
  *ptr++ = static_cast<uint8_t>(values[2] >> 45);
  *ptr++ = static_cast<uint8_t>(values[2] >> 37);
  *ptr++ = static_cast<uint8_t>(values[2] >> 29);
  *ptr++ = static_cast<uint8_t>(values[2] >> 21);
  *ptr++ = static_cast<uint8_t>(values[2] >> 13);
  *ptr++ = static_cast<uint8_t>(values[2] >> 5);

  *ptr = static_cast<uint8_t>(values[2] << 3);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 52);
  *ptr++ = static_cast<uint8_t>(values[3] >> 44);
  *ptr++ = static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 51);
  *ptr++ = static_cast<uint8_t>(values[4] >> 43);
  *ptr++ = static_cast<uint8_t>(values[4] >> 35);
  *ptr++ = static_cast<uint8_t>(values[4] >> 27);
  *ptr++ = static_cast<uint8_t>(values[4] >> 19);
  *ptr++ = static_cast<uint8_t>(values[4] >> 11);
  *ptr++ = static_cast<uint8_t>(values[4] >> 3);

  *ptr = static_cast<uint8_t>(values[4] << 5);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 50);
  *ptr++ = static_cast<uint8_t>(values[5] >> 42);
  *ptr++ = static_cast<uint8_t>(values[5] >> 34);
  *ptr++ = static_cast<uint8_t>(values[5] >> 26);
  *ptr++ = static_cast<uint8_t>(values[5] >> 18);
  *ptr++ = static_cast<uint8_t>(values[5] >> 10);
  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 49);
  *ptr++ = static_cast<uint8_t>(values[6] >> 41);
  *ptr++ = static_cast<uint8_t>(values[6] >> 33);
  *ptr++ = static_cast<uint8_t>(values[6] >> 25);
  *ptr++ = static_cast<uint8_t>(values[6] >> 17);
  *ptr++ = static_cast<uint8_t>(values[6] >> 9);
  *ptr++ = static_cast<uint8_t>(values[6] >> 1);

  *ptr = static_cast<uint8_t>(values[6] << 7);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits56(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 48);
  *ptr++ = static_cast<uint8_t>(values[0] >> 40);
  *ptr++ = static_cast<uint8_t>(values[0] >> 32);
  *ptr++ = static_cast<uint8_t>(values[0] >> 24);
  *ptr++ = static_cast<uint8_t>(values[0] >> 16);
  *ptr++ = static_cast<uint8_t>(values[0] >> 8);
  *ptr++ = static_cast<uint8_t>(values[0]);

  *ptr++ = static_cast<uint8_t>(values[1] >> 48);
  *ptr++ = static_cast<uint8_t>(values[1] >> 40);
  *ptr++ = static_cast<uint8_t>(values[1] >> 32);
  *ptr++ = static_cast<uint8_t>(values[1] >> 24);
  *ptr++ = static_cast<uint8_t>(values[1] >> 16);
  *ptr++ = static_cast<uint8_t>(values[1] >> 8);
  *ptr++ = static_cast<uint8_t>(values[1]);

  *ptr++ = static_cast<uint8_t>(values[2] >> 48);
  *ptr++ = static_cast<uint8_t>(values[2] >> 40);
  *ptr++ = static_cast<uint8_t>(values[2] >> 32);
  *ptr++ = static_cast<uint8_t>(values[2] >> 24);
  *ptr++ = static_cast<uint8_t>(values[2] >> 16);
  *ptr++ = static_cast<uint8_t>(values[2] >> 8);
  *ptr++ = static_cast<uint8_t>(values[2]);

  *ptr++ = static_cast<uint8_t>(values[3] >> 48);
  *ptr++ = static_cast<uint8_t>(values[3] >> 40);
  *ptr++ = static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 48);
  *ptr++ = static_cast<uint8_t>(values[4] >> 40);
  *ptr++ = static_cast<uint8_t>(values[4] >> 32);
  *ptr++ = static_cast<uint8_t>(values[4] >> 24);
  *ptr++ = static_cast<uint8_t>(values[4] >> 16);
  *ptr++ = static_cast<uint8_t>(values[4] >> 8);
  *ptr++ = static_cast<uint8_t>(values[4]);

  *ptr++ = static_cast<uint8_t>(values[5] >> 48);
  *ptr++ = static_cast<uint8_t>(values[5] >> 40);
  *ptr++ = static_cast<uint8_t>(values[5] >> 32);
  *ptr++ = static_cast<uint8_t>(values[5] >> 24);
  *ptr++ = static_cast<uint8_t>(values[5] >> 16);
  *ptr++ = static_cast<uint8_t>(values[5] >> 8);
  *ptr++ = static_cast<uint8_t>(values[5]);

  *ptr++ = static_cast<uint8_t>(values[6] >> 48);
  *ptr++ = static_cast<uint8_t>(values[6] >> 40);
  *ptr++ = static_cast<uint8_t>(values[6] >> 32);
  *ptr++ = static_cast<uint8_t>(values[6] >> 24);
  *ptr++ = static_cast<uint8_t>(values[6] >> 16);
  *ptr++ = static_cast<uint8_t>(values[6] >> 8);
  *ptr++ = static_cast<uint8_t>(values[6]);

  *ptr++ = static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits57(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 49);
  *ptr++ = static_cast<uint8_t>(values[0] >> 41);
  *ptr++ = static_cast<uint8_t>(values[0] >> 33);
  *ptr++ = static_cast<uint8_t>(values[0] >> 25);
  *ptr++ = static_cast<uint8_t>(values[0] >> 17);
  *ptr++ = static_cast<uint8_t>(values[0] >> 9);
  *ptr++ = static_cast<uint8_t>(values[0] >> 1);

  *ptr = static_cast<uint8_t>(values[0] << 7);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 50);
  *ptr++ = static_cast<uint8_t>(values[1] >> 42);
  *ptr++ = static_cast<uint8_t>(values[1] >> 34);
  *ptr++ = static_cast<uint8_t>(values[1] >> 26);
  *ptr++ = static_cast<uint8_t>(values[1] >> 18);
  *ptr++ = static_cast<uint8_t>(values[1] >> 10);
  *ptr++ = static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 51);
  *ptr++ = static_cast<uint8_t>(values[2] >> 43);
  *ptr++ = static_cast<uint8_t>(values[2] >> 35);
  *ptr++ = static_cast<uint8_t>(values[2] >> 27);
  *ptr++ = static_cast<uint8_t>(values[2] >> 19);
  *ptr++ = static_cast<uint8_t>(values[2] >> 11);
  *ptr++ = static_cast<uint8_t>(values[2] >> 3);

  *ptr = static_cast<uint8_t>(values[2] << 5);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 52);
  *ptr++ = static_cast<uint8_t>(values[3] >> 44);
  *ptr++ = static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 53);
  *ptr++ = static_cast<uint8_t>(values[4] >> 45);
  *ptr++ = static_cast<uint8_t>(values[4] >> 37);
  *ptr++ = static_cast<uint8_t>(values[4] >> 29);
  *ptr++ = static_cast<uint8_t>(values[4] >> 21);
  *ptr++ = static_cast<uint8_t>(values[4] >> 13);
  *ptr++ = static_cast<uint8_t>(values[4] >> 5);

  *ptr = static_cast<uint8_t>(values[4] << 3);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 54);
  *ptr++ = static_cast<uint8_t>(values[5] >> 46);
  *ptr++ = static_cast<uint8_t>(values[5] >> 38);
  *ptr++ = static_cast<uint8_t>(values[5] >> 30);
  *ptr++ = static_cast<uint8_t>(values[5] >> 22);
  *ptr++ = static_cast<uint8_t>(values[5] >> 14);
  *ptr++ = static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 55);
  *ptr++ = static_cast<uint8_t>(values[6] >> 47);
  *ptr++ = static_cast<uint8_t>(values[6] >> 39);
  *ptr++ = static_cast<uint8_t>(values[6] >> 31);
  *ptr++ = static_cast<uint8_t>(values[6] >> 23);
  *ptr++ = static_cast<uint8_t>(values[6] >> 15);
  *ptr++ = static_cast<uint8_t>(values[6] >> 7);

  *ptr = static_cast<uint8_t>(values[6] << 1);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 56);
  *ptr++ = static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits58(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 50);
  *ptr++ = static_cast<uint8_t>(values[0] >> 42);
  *ptr++ = static_cast<uint8_t>(values[0] >> 34);
  *ptr++ = static_cast<uint8_t>(values[0] >> 26);
  *ptr++ = static_cast<uint8_t>(values[0] >> 18);
  *ptr++ = static_cast<uint8_t>(values[0] >> 10);
  *ptr++ = static_cast<uint8_t>(values[0] >> 2);

  *ptr = static_cast<uint8_t>(values[0] << 6);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 52);
  *ptr++ = static_cast<uint8_t>(values[1] >> 44);
  *ptr++ = static_cast<uint8_t>(values[1] >> 36);
  *ptr++ = static_cast<uint8_t>(values[1] >> 28);
  *ptr++ = static_cast<uint8_t>(values[1] >> 20);
  *ptr++ = static_cast<uint8_t>(values[1] >> 12);
  *ptr++ = static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 54);
  *ptr++ = static_cast<uint8_t>(values[2] >> 46);
  *ptr++ = static_cast<uint8_t>(values[2] >> 38);
  *ptr++ = static_cast<uint8_t>(values[2] >> 30);
  *ptr++ = static_cast<uint8_t>(values[2] >> 22);
  *ptr++ = static_cast<uint8_t>(values[2] >> 14);
  *ptr++ = static_cast<uint8_t>(values[2] >> 6);

  *ptr = static_cast<uint8_t>(values[2] << 2);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 56);
  *ptr++ = static_cast<uint8_t>(values[3] >> 48);
  *ptr++ = static_cast<uint8_t>(values[3] >> 40);
  *ptr++ = static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 50);
  *ptr++ = static_cast<uint8_t>(values[4] >> 42);
  *ptr++ = static_cast<uint8_t>(values[4] >> 34);
  *ptr++ = static_cast<uint8_t>(values[4] >> 26);
  *ptr++ = static_cast<uint8_t>(values[4] >> 18);
  *ptr++ = static_cast<uint8_t>(values[4] >> 10);
  *ptr++ = static_cast<uint8_t>(values[4] >> 2);

  *ptr = static_cast<uint8_t>(values[4] << 6);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 52);
  *ptr++ = static_cast<uint8_t>(values[5] >> 44);
  *ptr++ = static_cast<uint8_t>(values[5] >> 36);
  *ptr++ = static_cast<uint8_t>(values[5] >> 28);
  *ptr++ = static_cast<uint8_t>(values[5] >> 20);
  *ptr++ = static_cast<uint8_t>(values[5] >> 12);
  *ptr++ = static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 54);
  *ptr++ = static_cast<uint8_t>(values[6] >> 46);
  *ptr++ = static_cast<uint8_t>(values[6] >> 38);
  *ptr++ = static_cast<uint8_t>(values[6] >> 30);
  *ptr++ = static_cast<uint8_t>(values[6] >> 22);
  *ptr++ = static_cast<uint8_t>(values[6] >> 14);
  *ptr++ = static_cast<uint8_t>(values[6] >> 6);

  *ptr = static_cast<uint8_t>(values[6] << 2);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 56);
  *ptr++ = static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits59(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 51);
  *ptr++ = static_cast<uint8_t>(values[0] >> 43);
  *ptr++ = static_cast<uint8_t>(values[0] >> 35);
  *ptr++ = static_cast<uint8_t>(values[0] >> 27);
  *ptr++ = static_cast<uint8_t>(values[0] >> 19);
  *ptr++ = static_cast<uint8_t>(values[0] >> 11);
  *ptr++ = static_cast<uint8_t>(values[0] >> 3);

  *ptr = static_cast<uint8_t>(values[0] << 5);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 54);
  *ptr++ = static_cast<uint8_t>(values[1] >> 46);
  *ptr++ = static_cast<uint8_t>(values[1] >> 38);
  *ptr++ = static_cast<uint8_t>(values[1] >> 30);
  *ptr++ = static_cast<uint8_t>(values[1] >> 22);
  *ptr++ = static_cast<uint8_t>(values[1] >> 14);
  *ptr++ = static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 57);
  *ptr++ = static_cast<uint8_t>(values[2] >> 49);
  *ptr++ = static_cast<uint8_t>(values[2] >> 41);
  *ptr++ = static_cast<uint8_t>(values[2] >> 33);
  *ptr++ = static_cast<uint8_t>(values[2] >> 25);
  *ptr++ = static_cast<uint8_t>(values[2] >> 17);
  *ptr++ = static_cast<uint8_t>(values[2] >> 9);
  *ptr++ = static_cast<uint8_t>(values[2] >> 1);

  *ptr = static_cast<uint8_t>(values[2] << 7);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 52);
  *ptr++ = static_cast<uint8_t>(values[3] >> 44);
  *ptr++ = static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 55);
  *ptr++ = static_cast<uint8_t>(values[4] >> 47);
  *ptr++ = static_cast<uint8_t>(values[4] >> 39);
  *ptr++ = static_cast<uint8_t>(values[4] >> 31);
  *ptr++ = static_cast<uint8_t>(values[4] >> 23);
  *ptr++ = static_cast<uint8_t>(values[4] >> 15);
  *ptr++ = static_cast<uint8_t>(values[4] >> 7);

  *ptr = static_cast<uint8_t>(values[4] << 1);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 58);
  *ptr++ = static_cast<uint8_t>(values[5] >> 50);
  *ptr++ = static_cast<uint8_t>(values[5] >> 42);
  *ptr++ = static_cast<uint8_t>(values[5] >> 34);
  *ptr++ = static_cast<uint8_t>(values[5] >> 26);
  *ptr++ = static_cast<uint8_t>(values[5] >> 18);
  *ptr++ = static_cast<uint8_t>(values[5] >> 10);
  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 53);
  *ptr++ = static_cast<uint8_t>(values[6] >> 45);
  *ptr++ = static_cast<uint8_t>(values[6] >> 37);
  *ptr++ = static_cast<uint8_t>(values[6] >> 29);
  *ptr++ = static_cast<uint8_t>(values[6] >> 21);
  *ptr++ = static_cast<uint8_t>(values[6] >> 13);
  *ptr++ = static_cast<uint8_t>(values[6] >> 5);

  *ptr = static_cast<uint8_t>(values[6] << 3);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 56);
  *ptr++ = static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits60(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 52);
  *ptr++ = static_cast<uint8_t>(values[0] >> 44);
  *ptr++ = static_cast<uint8_t>(values[0] >> 36);
  *ptr++ = static_cast<uint8_t>(values[0] >> 28);
  *ptr++ = static_cast<uint8_t>(values[0] >> 20);
  *ptr++ = static_cast<uint8_t>(values[0] >> 12);
  *ptr++ = static_cast<uint8_t>(values[0] >> 4);

  *ptr = static_cast<uint8_t>(values[0] << 4);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 56);
  *ptr++ = static_cast<uint8_t>(values[1] >> 48);
  *ptr++ = static_cast<uint8_t>(values[1] >> 40);
  *ptr++ = static_cast<uint8_t>(values[1] >> 32);
  *ptr++ = static_cast<uint8_t>(values[1] >> 24);
  *ptr++ = static_cast<uint8_t>(values[1] >> 16);
  *ptr++ = static_cast<uint8_t>(values[1] >> 8);
  *ptr++ = static_cast<uint8_t>(values[1]);

  *ptr++ = static_cast<uint8_t>(values[2] >> 52);
  *ptr++ = static_cast<uint8_t>(values[2] >> 44);
  *ptr++ = static_cast<uint8_t>(values[2] >> 36);
  *ptr++ = static_cast<uint8_t>(values[2] >> 28);
  *ptr++ = static_cast<uint8_t>(values[2] >> 20);
  *ptr++ = static_cast<uint8_t>(values[2] >> 12);
  *ptr++ = static_cast<uint8_t>(values[2] >> 4);

  *ptr = static_cast<uint8_t>(values[2] << 4);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 56);
  *ptr++ = static_cast<uint8_t>(values[3] >> 48);
  *ptr++ = static_cast<uint8_t>(values[3] >> 40);
  *ptr++ = static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 52);
  *ptr++ = static_cast<uint8_t>(values[4] >> 44);
  *ptr++ = static_cast<uint8_t>(values[4] >> 36);
  *ptr++ = static_cast<uint8_t>(values[4] >> 28);
  *ptr++ = static_cast<uint8_t>(values[4] >> 20);
  *ptr++ = static_cast<uint8_t>(values[4] >> 12);
  *ptr++ = static_cast<uint8_t>(values[4] >> 4);

  *ptr = static_cast<uint8_t>(values[4] << 4);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 56);
  *ptr++ = static_cast<uint8_t>(values[5] >> 48);
  *ptr++ = static_cast<uint8_t>(values[5] >> 40);
  *ptr++ = static_cast<uint8_t>(values[5] >> 32);
  *ptr++ = static_cast<uint8_t>(values[5] >> 24);
  *ptr++ = static_cast<uint8_t>(values[5] >> 16);
  *ptr++ = static_cast<uint8_t>(values[5] >> 8);
  *ptr++ = static_cast<uint8_t>(values[5]);

  *ptr++ = static_cast<uint8_t>(values[6] >> 52);
  *ptr++ = static_cast<uint8_t>(values[6] >> 44);
  *ptr++ = static_cast<uint8_t>(values[6] >> 36);
  *ptr++ = static_cast<uint8_t>(values[6] >> 28);
  *ptr++ = static_cast<uint8_t>(values[6] >> 20);
  *ptr++ = static_cast<uint8_t>(values[6] >> 12);
  *ptr++ = static_cast<uint8_t>(values[6] >> 4);

  *ptr = static_cast<uint8_t>(values[6] << 4);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 56);
  *ptr++ = static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits61(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 53);
  *ptr++ = static_cast<uint8_t>(values[0] >> 45);
  *ptr++ = static_cast<uint8_t>(values[0] >> 37);
  *ptr++ = static_cast<uint8_t>(values[0] >> 29);
  *ptr++ = static_cast<uint8_t>(values[0] >> 21);
  *ptr++ = static_cast<uint8_t>(values[0] >> 13);
  *ptr++ = static_cast<uint8_t>(values[0] >> 5);

  *ptr = static_cast<uint8_t>(values[0] << 3);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 58);
  *ptr++ = static_cast<uint8_t>(values[1] >> 50);
  *ptr++ = static_cast<uint8_t>(values[1] >> 42);
  *ptr++ = static_cast<uint8_t>(values[1] >> 34);
  *ptr++ = static_cast<uint8_t>(values[1] >> 26);
  *ptr++ = static_cast<uint8_t>(values[1] >> 18);
  *ptr++ = static_cast<uint8_t>(values[1] >> 10);
  *ptr++ = static_cast<uint8_t>(values[1] >> 2);

  *ptr = static_cast<uint8_t>(values[1] << 6);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 55);
  *ptr++ = static_cast<uint8_t>(values[2] >> 47);
  *ptr++ = static_cast<uint8_t>(values[2] >> 39);
  *ptr++ = static_cast<uint8_t>(values[2] >> 31);
  *ptr++ = static_cast<uint8_t>(values[2] >> 23);
  *ptr++ = static_cast<uint8_t>(values[2] >> 15);
  *ptr++ = static_cast<uint8_t>(values[2] >> 7);

  *ptr = static_cast<uint8_t>(values[2] << 1);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 60);
  *ptr++ = static_cast<uint8_t>(values[3] >> 52);
  *ptr++ = static_cast<uint8_t>(values[3] >> 44);
  *ptr++ = static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 57);
  *ptr++ = static_cast<uint8_t>(values[4] >> 49);
  *ptr++ = static_cast<uint8_t>(values[4] >> 41);
  *ptr++ = static_cast<uint8_t>(values[4] >> 33);
  *ptr++ = static_cast<uint8_t>(values[4] >> 25);
  *ptr++ = static_cast<uint8_t>(values[4] >> 17);
  *ptr++ = static_cast<uint8_t>(values[4] >> 9);
  *ptr++ = static_cast<uint8_t>(values[4] >> 1);

  *ptr = static_cast<uint8_t>(values[4] << 7);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 54);
  *ptr++ = static_cast<uint8_t>(values[5] >> 46);
  *ptr++ = static_cast<uint8_t>(values[5] >> 38);
  *ptr++ = static_cast<uint8_t>(values[5] >> 30);
  *ptr++ = static_cast<uint8_t>(values[5] >> 22);
  *ptr++ = static_cast<uint8_t>(values[5] >> 14);
  *ptr++ = static_cast<uint8_t>(values[5] >> 6);

  *ptr = static_cast<uint8_t>(values[5] << 2);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 59);
  *ptr++ = static_cast<uint8_t>(values[6] >> 51);
  *ptr++ = static_cast<uint8_t>(values[6] >> 43);
  *ptr++ = static_cast<uint8_t>(values[6] >> 35);
  *ptr++ = static_cast<uint8_t>(values[6] >> 27);
  *ptr++ = static_cast<uint8_t>(values[6] >> 19);
  *ptr++ = static_cast<uint8_t>(values[6] >> 11);
  *ptr++ = static_cast<uint8_t>(values[6] >> 3);

  *ptr = static_cast<uint8_t>(values[6] << 5);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 56);
  *ptr++ = static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits62(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 54);
  *ptr++ = static_cast<uint8_t>(values[0] >> 46);
  *ptr++ = static_cast<uint8_t>(values[0] >> 38);
  *ptr++ = static_cast<uint8_t>(values[0] >> 30);
  *ptr++ = static_cast<uint8_t>(values[0] >> 22);
  *ptr++ = static_cast<uint8_t>(values[0] >> 14);
  *ptr++ = static_cast<uint8_t>(values[0] >> 6);

  *ptr = static_cast<uint8_t>(values[0] << 2);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 60);
  *ptr++ = static_cast<uint8_t>(values[1] >> 52);
  *ptr++ = static_cast<uint8_t>(values[1] >> 44);
  *ptr++ = static_cast<uint8_t>(values[1] >> 36);
  *ptr++ = static_cast<uint8_t>(values[1] >> 28);
  *ptr++ = static_cast<uint8_t>(values[1] >> 20);
  *ptr++ = static_cast<uint8_t>(values[1] >> 12);
  *ptr++ = static_cast<uint8_t>(values[1] >> 4);

  *ptr = static_cast<uint8_t>(values[1] << 4);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 58);
  *ptr++ = static_cast<uint8_t>(values[2] >> 50);
  *ptr++ = static_cast<uint8_t>(values[2] >> 42);
  *ptr++ = static_cast<uint8_t>(values[2] >> 34);
  *ptr++ = static_cast<uint8_t>(values[2] >> 26);
  *ptr++ = static_cast<uint8_t>(values[2] >> 18);
  *ptr++ = static_cast<uint8_t>(values[2] >> 10);
  *ptr++ = static_cast<uint8_t>(values[2] >> 2);

  *ptr = static_cast<uint8_t>(values[2] << 6);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 56);
  *ptr++ = static_cast<uint8_t>(values[3] >> 48);
  *ptr++ = static_cast<uint8_t>(values[3] >> 40);
  *ptr++ = static_cast<uint8_t>(values[3] >> 32);
  *ptr++ = static_cast<uint8_t>(values[3] >> 24);
  *ptr++ = static_cast<uint8_t>(values[3] >> 16);
  *ptr++ = static_cast<uint8_t>(values[3] >> 8);
  *ptr++ = static_cast<uint8_t>(values[3]);

  *ptr++ = static_cast<uint8_t>(values[4] >> 54);
  *ptr++ = static_cast<uint8_t>(values[4] >> 46);
  *ptr++ = static_cast<uint8_t>(values[4] >> 38);
  *ptr++ = static_cast<uint8_t>(values[4] >> 30);
  *ptr++ = static_cast<uint8_t>(values[4] >> 22);
  *ptr++ = static_cast<uint8_t>(values[4] >> 14);
  *ptr++ = static_cast<uint8_t>(values[4] >> 6);

  *ptr = static_cast<uint8_t>(values[4] << 2);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 60);
  *ptr++ = static_cast<uint8_t>(values[5] >> 52);
  *ptr++ = static_cast<uint8_t>(values[5] >> 44);
  *ptr++ = static_cast<uint8_t>(values[5] >> 36);
  *ptr++ = static_cast<uint8_t>(values[5] >> 28);
  *ptr++ = static_cast<uint8_t>(values[5] >> 20);
  *ptr++ = static_cast<uint8_t>(values[5] >> 12);
  *ptr++ = static_cast<uint8_t>(values[5] >> 4);

  *ptr = static_cast<uint8_t>(values[5] << 4);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 58);
  *ptr++ = static_cast<uint8_t>(values[6] >> 50);
  *ptr++ = static_cast<uint8_t>(values[6] >> 42);
  *ptr++ = static_cast<uint8_t>(values[6] >> 34);
  *ptr++ = static_cast<uint8_t>(values[6] >> 26);
  *ptr++ = static_cast<uint8_t>(values[6] >> 18);
  *ptr++ = static_cast<uint8_t>(values[6] >> 10);
  *ptr++ = static_cast<uint8_t>(values[6] >> 2);

  *ptr = static_cast<uint8_t>(values[6] << 6);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 56);
  *ptr++ = static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void packBits63(const uint64_t* values, uint8_t* ptr) {
  *ptr++ = static_cast<uint8_t>(values[0] >> 55);
  *ptr++ = static_cast<uint8_t>(values[0] >> 47);
  *ptr++ = static_cast<uint8_t>(values[0] >> 39);
  *ptr++ = static_cast<uint8_t>(values[0] >> 31);
  *ptr++ = static_cast<uint8_t>(values[0] >> 23);
  *ptr++ = static_cast<uint8_t>(values[0] >> 15);
  *ptr++ = static_cast<uint8_t>(values[0] >> 7);

  *ptr = static_cast<uint8_t>(values[0] << 1);
  *ptr++ |= static_cast<uint8_t>(values[1] >> 62);
  *ptr++ = static_cast<uint8_t>(values[1] >> 54);
  *ptr++ = static_cast<uint8_t>(values[1] >> 46);
  *ptr++ = static_cast<uint8_t>(values[1] >> 38);
  *ptr++ = static_cast<uint8_t>(values[1] >> 30);
  *ptr++ = static_cast<uint8_t>(values[1] >> 22);
  *ptr++ = static_cast<uint8_t>(values[1] >> 14);
  *ptr++ = static_cast<uint8_t>(values[1] >> 6);

  *ptr = static_cast<uint8_t>(values[1] << 2);
  *ptr++ |= static_cast<uint8_t>(values[2] >> 61);
  *ptr++ = static_cast<uint8_t>(values[2] >> 53);
  *ptr++ = static_cast<uint8_t>(values[2] >> 45);
  *ptr++ = static_cast<uint8_t>(values[2] >> 37);
  *ptr++ = static_cast<uint8_t>(values[2] >> 29);
  *ptr++ = static_cast<uint8_t>(values[2] >> 21);
  *ptr++ = static_cast<uint8_t>(values[2] >> 13);
  *ptr++ = static_cast<uint8_t>(values[2] >> 5);

  *ptr = static_cast<uint8_t>(values[2] << 3);
  *ptr++ |= static_cast<uint8_t>(values[3] >> 60);
  *ptr++ = static_cast<uint8_t>(values[3] >> 52);
  *ptr++ = static_cast<uint8_t>(values[3] >> 44);
  *ptr++ = static_cast<uint8_t>(values[3] >> 36);
  *ptr++ = static_cast<uint8_t>(values[3] >> 28);
  *ptr++ = static_cast<uint8_t>(values[3] >> 20);
  *ptr++ = static_cast<uint8_t>(values[3] >> 12);
  *ptr++ = static_cast<uint8_t>(values[3] >> 4);

  *ptr = static_cast<uint8_t>(values[3] << 4);
  *ptr++ |= static_cast<uint8_t>(values[4] >> 59);
  *ptr++ = static_cast<uint8_t>(values[4] >> 51);
  *ptr++ = static_cast<uint8_t>(values[4] >> 43);
  *ptr++ = static_cast<uint8_t>(values[4] >> 35);
  *ptr++ = static_cast<uint8_t>(values[4] >> 27);
  *ptr++ = static_cast<uint8_t>(values[4] >> 19);
  *ptr++ = static_cast<uint8_t>(values[4] >> 11);
  *ptr++ = static_cast<uint8_t>(values[4] >> 3);

  *ptr = static_cast<uint8_t>(values[4] << 5);
  *ptr++ |= static_cast<uint8_t>(values[5] >> 58);
  *ptr++ = static_cast<uint8_t>(values[5] >> 50);
  *ptr++ = static_cast<uint8_t>(values[5] >> 42);
  *ptr++ = static_cast<uint8_t>(values[5] >> 34);
  *ptr++ = static_cast<uint8_t>(values[5] >> 26);
  *ptr++ = static_cast<uint8_t>(values[5] >> 18);
  *ptr++ = static_cast<uint8_t>(values[5] >> 10);
  *ptr++ = static_cast<uint8_t>(values[5] >> 2);

  *ptr = static_cast<uint8_t>(values[5] << 6);
  *ptr++ |= static_cast<uint8_t>(values[6] >> 57);
  *ptr++ = static_cast<uint8_t>(values[6] >> 49);
  *ptr++ = static_cast<uint8_t>(values[6] >> 41);
  *ptr++ = static_cast<uint8_t>(values[6] >> 33);
  *ptr++ = static_cast<uint8_t>(values[6] >> 25);
  *ptr++ = static_cast<uint8_t>(values[6] >> 17);
  *ptr++ = static_cast<uint8_t>(values[6] >> 9);
  *ptr++ = static_cast<uint8_t>(values[6] >> 1);

  *ptr = static_cast<uint8_t>(values[6] << 7);
  *ptr++ |= static_cast<uint8_t>(values[7] >> 56);
  *ptr++ = static_cast<uint8_t>(values[7] >> 48);
  *ptr++ = static_cast<uint8_t>(values[7] >> 40);
  *ptr++ = static_cast<uint8_t>(values[7] >> 32);
  *ptr++ = static_cast<uint8_t>(values[7] >> 24);
  *ptr++ = static_cast<uint8_t>(values[7] >> 16);
  *ptr++ = static_cast<uint8_t>(values[7] >> 8);
  *ptr = static_cast<uint8_t>(values[7]);
}

static inline void unpackBits1(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr >> 7;
  values[1] = (*ptr >> 6) & 1;
  values[2] = (*ptr >> 5) & 1;
  values[3] = (*ptr >> 4) & 1;
  values[4] = (*ptr >> 3) & 1;
  values[5] = (*ptr >> 2) & 1;
  values[6] = (*ptr >> 1) & 1;
  values[7] = *ptr & 1;
}

static inline void unpackBits2(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr >> 6;
  values[1] = (*ptr >> 4) & 3;
  values[2] = (*ptr >> 2) & 3;
  values[3] = *ptr++ & 3;
  values[4] = *ptr >> 6;
  values[5] = (*ptr >> 4) & 3;
  values[6] = (*ptr >> 2) & 3;
  values[7] = *ptr & 3;
}

static inline void unpackBits3(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr >> 5;
  values[1] = (*ptr >> 2) & 7;
  values[2] = (*ptr++ & 3) << 1;
  values[2] |= *ptr >> 7;
  values[3] = (*ptr >> 4) & 7;
  values[4] = (*ptr >> 1) & 7;
  values[5] = (*ptr++ & 1) << 2;
  values[5] |= *ptr >> 6;
  values[6] = (*ptr >> 3) & 7;
  values[7] = *ptr & 7;
}

static inline void unpackBits4(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr >> 4;
  values[1] = *ptr++ & 0xf;
  values[2] = *ptr >> 4;
  values[3] = *ptr++ & 0xf;
  values[4] = *ptr >> 4;
  values[5] = *ptr++ & 0xf;
  values[6] = *ptr >> 4;
  values[7] = *ptr & 0xf;
}

static inline void unpackBits5(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr >> 3;

  values[1] = (*ptr++ & 7) << 2;
  values[1] |= *ptr >> 6;

  values[2] = (*ptr >> 1) & 0x1f;

  values[3] = (*ptr++ & 1) << 4;
  values[3] |= *ptr >> 4;

  values[4] = (*ptr++ & 0xf) << 1;
  values[4] |= *ptr >> 7;

  values[5] = (*ptr >> 2) & 0x1f;

  values[6] = (*ptr++ & 3) << 3;
  values[6] |= *ptr >> 5;

  values[7] = *ptr & 0x1f;
}

static inline void unpackBits6(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr >> 2;

  values[1] = (*ptr++ & 3) << 4;
  values[1] |= *ptr >> 4;

  values[2] = (*ptr++ & 0xf) << 2;
  values[2] |= *ptr >> 6;

  values[3] = *ptr++ & 0x3f;

  values[4] = *ptr >> 2;

  values[5] = (*ptr++ & 3) << 4;
  values[5] |= *ptr >> 4;

  values[6] = (*ptr++ & 0xf) << 2;
  values[6] |= *ptr >> 6;

  values[7] = *ptr & 0x3f;
}

static inline void unpackBits7(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr >> 1;

  values[1] = (*ptr++ & 1) << 6;
  values[1] |= *ptr >> 2;

  values[2] = (*ptr++ & 3) << 5;
  values[2] |= *ptr >> 3;

  values[3] = (*ptr++ & 7) << 4;
  values[3] |= *ptr >> 4;

  values[4] = (*ptr++ & 0xf) << 3;
  values[4] |= *ptr >> 5;

  values[5] = (*ptr++ & 0x1f) << 2;
  values[5] |= *ptr >> 6;

  values[6] = (*ptr++ & 0x3f) << 1;
  values[6] |= *ptr >> 7;

  values[7] = *ptr & 0x7f;
}

static inline void unpackBits8(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++;
  values[1] = *ptr++;
  values[2] = *ptr++;
  values[3] = *ptr++;
  values[4] = *ptr++;
  values[5] = *ptr++;
  values[6] = *ptr++;
  values[7] = *ptr;
}

static inline void unpackBits9(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 1;
  values[0] |= *ptr >> 7;

  values[1] = (*ptr++ & 0x7f) << 2;
  values[1] |= *ptr >> 6;

  values[2] = (*ptr++ & 0x3f) << 3;
  values[2] |= *ptr >> 5;

  values[3] = (*ptr++ & 0x1f) << 4;
  values[3] |= *ptr >> 4;

  values[4] = (*ptr++ & 0xf) << 5;
  values[4] |= *ptr >> 3;

  values[5] = (*ptr++ & 7) << 6;
  values[5] |= *ptr >> 2;

  values[6] = (*ptr++ & 3) << 7;
  values[6] |= *ptr >> 1;

  values[7] = (*ptr++ & 1) << 8;
  values[7] |= *ptr;
}

static inline void unpackBits10(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 2;
  values[0] |= *ptr >> 6;

  values[1] = (*ptr++ & 0x3f) << 4;
  values[1] |= *ptr >> 4;

  values[2] = (*ptr++ & 0xf) << 6;
  values[2] |= *ptr >> 2;

  values[3] = (*ptr++ & 3) << 8;
  values[3] |= *ptr++;

  values[4] = *ptr++ << 2;
  values[4] |= *ptr >> 6;

  values[5] = (*ptr++ & 0x3f) << 4;
  values[5] |= *ptr >> 4;

  values[6] = (*ptr++ & 0xf) << 6;
  values[6] |= *ptr >> 2;

  values[7] = (*ptr++ & 3) << 8;
  values[7] |= *ptr;
}

static inline void unpackBits11(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 3;
  values[0] |= *ptr >> 5;

  values[1] = (*ptr++ & 0x1f) << 6;
  values[1] |= *ptr >> 2;

  values[2] = (*ptr++ & 3) << 9;
  values[2] |= *ptr++ << 1;
  values[2] |= *ptr >> 7;

  values[3] = (*ptr++ & 0x7f) << 4;
  values[3] |= *ptr >> 4;

  values[4] = (*ptr++ & 0xf) << 7;
  values[4] |= *ptr >> 1;

  values[5] = (*ptr++ & 1) << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = (*ptr++ & 0x3f) << 5;
  values[6] |= *ptr >> 3;

  values[7] = (*ptr++ & 7) << 8;
  values[7] |= *ptr;
}

static inline void unpackBits12(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 4;
  values[0] |= *ptr >> 4;

  values[1] = (*ptr++ & 0xf) << 8;
  values[1] |= *ptr++;

  values[2] = *ptr++ << 4;
  values[2] |= *ptr >> 4;

  values[3] = (*ptr++ & 0xf) << 8;
  values[3] |= *ptr++;

  values[4] = *ptr++ << 4;
  values[4] |= *ptr >> 4;

  values[5] = (*ptr++ & 0xf) << 8;
  values[5] |= *ptr++;

  values[6] = *ptr++ << 4;
  values[6] |= *ptr >> 4;

  values[7] = (*ptr++ & 0xf) << 8;
  values[7] |= *ptr;
}

static inline void unpackBits13(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 5;
  values[0] |= *ptr >> 3;

  values[1] = (*ptr++ & 7) << 10;
  values[1] |= *ptr++ << 2;
  values[1] |= *ptr >> 6;

  values[2] = (*ptr++ & 0x3f) << 7;
  values[2] |= *ptr >> 1;

  values[3] = (*ptr++ & 1) << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = (*ptr++ & 0xf) << 9;
  values[4] |= *ptr++ << 1;
  values[4] |= *ptr >> 7;

  values[5] = (*ptr++ & 0x7f) << 6;
  values[5] |= *ptr >> 2;

  values[6] = (*ptr++ & 3) << 11;
  values[6] |= *ptr++ << 3;
  values[6] |= *ptr >> 5;

  values[7] = (*ptr++ & 0x1f) << 8;
  values[7] |= *ptr;
}

static inline void unpackBits14(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 6;
  values[0] |= *ptr >> 2;

  values[1] = (*ptr++ & 3) << 12;
  values[1] |= *ptr++ << 4;
  values[1] |= *ptr >> 4;

  values[2] = (*ptr++ & 0xf) << 10;
  values[2] |= *ptr++ << 2;
  values[2] |= *ptr >> 6;

  values[3] = (*ptr++ & 0x3f) << 8;
  values[3] |= *ptr++;

  values[4] = *ptr++ << 6;
  values[4] |= *ptr >> 2;

  values[5] = (*ptr++ & 3) << 12;
  values[5] |= *ptr++ << 4;
  values[5] |= *ptr >> 4;

  values[6] = (*ptr++ & 0xf) << 10;
  values[6] |= *ptr++ << 2;
  values[6] |= *ptr >> 6;

  values[7] = (*ptr++ & 0x3f) << 8;
  values[7] |= *ptr;
}

static inline void unpackBits15(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 7;
  values[0] |= *ptr >> 1;

  values[1] = (*ptr++ & 1) << 14;
  values[1] |= *ptr++ << 6;
  values[1] |= *ptr >> 2;

  values[2] = (*ptr++ & 3) << 13;
  values[2] |= *ptr++ << 5;
  values[2] |= *ptr >> 3;

  values[3] = (*ptr++ & 7) << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = (*ptr++ & 0xf) << 11;
  values[4] |= *ptr++ << 3;
  values[4] |= *ptr >> 5;

  values[5] = (*ptr++ & 0x1f) << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = (*ptr++ & 0x3f) << 9;
  values[6] |= *ptr++ << 1;
  values[6] |= *ptr >> 7;

  values[7] = (*ptr++ & 0x7f) << 8;
  values[7] |= *ptr;
}

static inline void unpackBits16(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 8;
  values[0] |= *ptr++;
  values[1] = *ptr++ << 8;
  values[1] |= *ptr++;
  values[2] = *ptr++ << 8;
  values[2] |= *ptr++;
  values[3] = *ptr++ << 8;
  values[3] |= *ptr++;
  values[4] = *ptr++ << 8;
  values[4] |= *ptr++;
  values[5] = *ptr++ << 8;
  values[5] |= *ptr++;
  values[6] = *ptr++ << 8;
  values[6] |= *ptr++;
  values[7] = *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits17(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 9;
  values[0] |= *ptr++ << 1;
  values[0] |= *ptr >> 7;

  values[1] = (*ptr++ & 0x7f) << 10;
  values[1] |= *ptr++ << 2;
  values[1] |= *ptr >> 6;

  values[2] = (*ptr++ & 0x3f) << 11;
  values[2] |= *ptr++ << 3;
  values[2] |= *ptr >> 5;

  values[3] = (*ptr++ & 0x1f) << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = (*ptr++ & 0xf) << 13;
  values[4] |= *ptr++ << 5;
  values[4] |= *ptr >> 3;

  values[5] = (*ptr++ & 7) << 14;
  values[5] |= *ptr++ << 6;
  values[5] |= *ptr >> 2;

  values[6] = (*ptr++ & 3) << 15;
  values[6] |= *ptr++ << 7;
  values[6] |= *ptr >> 1;

  values[7] = (*ptr++ & 1) << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits18(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 10;
  values[0] |= *ptr++ << 2;
  values[0] |= *ptr >> 6;

  values[1] = (*ptr++ & 0x3f) << 12;
  values[1] |= *ptr++ << 4;
  values[1] |= *ptr >> 4;

  values[2] = (*ptr++ & 0xf) << 14;
  values[2] |= *ptr++ << 6;
  values[2] |= *ptr >> 2;

  values[3] = (*ptr++ & 3) << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = *ptr++ << 10;
  values[4] |= *ptr++ << 2;
  values[4] |= *ptr >> 6;

  values[5] = (*ptr++ & 0x3f) << 12;
  values[5] |= *ptr++ << 4;
  values[5] |= *ptr >> 4;

  values[6] = (*ptr++ & 0xf) << 14;
  values[6] |= *ptr++ << 6;
  values[6] |= *ptr >> 2;

  values[7] = (*ptr++ & 3) << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits19(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 11;
  values[0] |= *ptr++ << 3;
  values[0] |= *ptr >> 5;

  values[1] = (*ptr++ & 0x1f) << 14;
  values[1] |= *ptr++ << 6;
  values[1] |= *ptr >> 2;

  values[2] = (*ptr++ & 3) << 17;
  values[2] |= *ptr++ << 9;
  values[2] |= *ptr++ << 1;
  values[2] |= *ptr >> 7;

  values[3] = (*ptr++ & 0x7f) << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = (*ptr++ & 0xf) << 15;
  values[4] |= *ptr++ << 7;
  values[4] |= *ptr >> 1;

  values[5] = (*ptr++ & 1) << 18;
  values[5] |= *ptr++ << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = (*ptr++ & 0x3f) << 13;
  values[6] |= *ptr++ << 5;
  values[6] |= *ptr >> 3;

  values[7] = (*ptr++ & 7) << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits20(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 12;
  values[0] |= *ptr++ << 4;
  values[0] |= *ptr >> 4;

  values[1] = (*ptr++ & 0xf) << 16;
  values[1] |= *ptr++ << 8;
  values[1] |= *ptr++;

  values[2] = *ptr++ << 12;
  values[2] |= *ptr++ << 4;
  values[2] |= *ptr >> 4;

  values[3] = (*ptr++ & 0xf) << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = *ptr++ << 12;
  values[4] |= *ptr++ << 4;
  values[4] |= *ptr >> 4;

  values[5] = (*ptr++ & 0xf) << 16;
  values[5] |= *ptr++ << 8;
  values[5] |= *ptr++;

  values[6] = *ptr++ << 12;
  values[6] |= *ptr++ << 4;
  values[6] |= *ptr >> 4;

  values[7] = (*ptr++ & 0xf) << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits21(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 13;
  values[0] |= *ptr++ << 5;
  values[0] |= *ptr >> 3;

  values[1] = (*ptr++ & 7) << 18;
  values[1] |= *ptr++ << 10;
  values[1] |= *ptr++ << 2;
  values[1] |= *ptr >> 6;

  values[2] = (*ptr++ & 0x3f) << 15;
  values[2] |= *ptr++ << 7;
  values[2] |= *ptr >> 1;

  values[3] = (*ptr++ & 1) << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = (*ptr++ & 0xf) << 17;
  values[4] |= *ptr++ << 9;
  values[4] |= *ptr++ << 1;
  values[4] |= *ptr >> 7;

  values[5] = (*ptr++ & 0x7f) << 14;
  values[5] |= *ptr++ << 6;
  values[5] |= *ptr >> 2;

  values[6] = (*ptr++ & 3) << 19;
  values[6] |= *ptr++ << 11;
  values[6] |= *ptr++ << 3;
  values[6] |= *ptr >> 5;

  values[7] = (*ptr++ & 0x1f) << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits22(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 14;
  values[0] |= *ptr++ << 6;
  values[0] |= *ptr >> 2;

  values[1] = (*ptr++ & 3) << 20;
  values[1] |= *ptr++ << 12;
  values[1] |= *ptr++ << 4;
  values[1] |= *ptr >> 4;

  values[2] = (*ptr++ & 0xf) << 18;
  values[2] |= *ptr++ << 10;
  values[2] |= *ptr++ << 2;
  values[2] |= *ptr >> 6;

  values[3] = (*ptr++ & 0x3f) << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = *ptr++ << 14;
  values[4] |= *ptr++ << 6;
  values[4] |= *ptr >> 2;

  values[5] = (*ptr++ & 3) << 20;
  values[5] |= *ptr++ << 12;
  values[5] |= *ptr++ << 4;
  values[5] |= *ptr >> 4;

  values[6] = (*ptr++ & 0xf) << 18;
  values[6] |= *ptr++ << 10;
  values[6] |= *ptr++ << 2;
  values[6] |= *ptr >> 6;

  values[7] = (*ptr++ & 0x3f) << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits23(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 15;
  values[0] |= *ptr++ << 7;
  values[0] |= *ptr >> 1;

  values[1] = (*ptr++ & 1) << 22;
  values[1] |= *ptr++ << 14;
  values[1] |= *ptr++ << 6;
  values[1] |= *ptr >> 2;

  values[2] = (*ptr++ & 3) << 21;
  values[2] |= *ptr++ << 13;
  values[2] |= *ptr++ << 5;
  values[2] |= *ptr >> 3;

  values[3] = (*ptr++ & 7) << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = (*ptr++ & 0xf) << 19;
  values[4] |= *ptr++ << 11;
  values[4] |= *ptr++ << 3;
  values[4] |= *ptr >> 5;

  values[5] = (*ptr++ & 0x1f) << 18;
  values[5] |= *ptr++ << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = (*ptr++ & 0x3f) << 17;
  values[6] |= *ptr++ << 9;
  values[6] |= *ptr++ << 1;
  values[6] |= *ptr >> 7;

  values[7] = (*ptr++ & 0x7f) << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits24(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 16;
  values[0] |= *ptr++ << 8;
  values[0] |= *ptr++;
  values[1] = *ptr++ << 16;
  values[1] |= *ptr++ << 8;
  values[1] |= *ptr++;
  values[2] = *ptr++ << 16;
  values[2] |= *ptr++ << 8;
  values[2] |= *ptr++;
  values[3] = *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;
  values[4] = *ptr++ << 16;
  values[4] |= *ptr++ << 8;
  values[4] |= *ptr++;
  values[5] = *ptr++ << 16;
  values[5] |= *ptr++ << 8;
  values[5] |= *ptr++;
  values[6] = *ptr++ << 16;
  values[6] |= *ptr++ << 8;
  values[6] |= *ptr++;
  values[7] = *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits25(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 17;
  values[0] |= *ptr++ << 9;
  values[0] |= *ptr++ << 1;
  values[0] |= *ptr >> 7;

  values[1] = (*ptr++ & 0x7f) << 18;
  values[1] |= *ptr++ << 10;
  values[1] |= *ptr++ << 2;
  values[1] |= *ptr >> 6;

  values[2] = (*ptr++ & 0x3f) << 19;
  values[2] |= *ptr++ << 11;
  values[2] |= *ptr++ << 3;
  values[2] |= *ptr >> 5;

  values[3] = (*ptr++ & 0x1f) << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = (*ptr++ & 0xf) << 21;
  values[4] |= *ptr++ << 13;
  values[4] |= *ptr++ << 5;
  values[4] |= *ptr >> 3;

  values[5] = (*ptr++ & 7) << 22;
  values[5] |= *ptr++ << 14;
  values[5] |= *ptr++ << 6;
  values[5] |= *ptr >> 2;

  values[6] = (*ptr++ & 3) << 23;
  values[6] |= *ptr++ << 15;
  values[6] |= *ptr++ << 7;
  values[6] |= *ptr >> 1;

  values[7] = static_cast<uint64_t>(*ptr++ & 1) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits26(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 18;
  values[0] |= *ptr++ << 10;
  values[0] |= *ptr++ << 2;
  values[0] |= *ptr >> 6;

  values[1] = (*ptr++ & 0x3f) << 20;
  values[1] |= *ptr++ << 12;
  values[1] |= *ptr++ << 4;
  values[1] |= *ptr >> 4;

  values[2] = (*ptr++ & 0xf) << 22;
  values[2] |= *ptr++ << 14;
  values[2] |= *ptr++ << 6;
  values[2] |= *ptr >> 2;

  values[3] = static_cast<uint64_t>(*ptr++ & 3) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = *ptr++ << 18;
  values[4] |= *ptr++ << 10;
  values[4] |= *ptr++ << 2;
  values[4] |= *ptr >> 6;

  values[5] = (*ptr++ & 0x3f) << 20;
  values[5] |= *ptr++ << 12;
  values[5] |= *ptr++ << 4;
  values[5] |= *ptr >> 4;

  values[6] = (*ptr++ & 0xf) << 22;
  values[6] |= *ptr++ << 14;
  values[6] |= *ptr++ << 6;
  values[6] |= *ptr >> 2;

  values[7] = static_cast<uint64_t>(*ptr++ & 3) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits27(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 19;
  values[0] |= *ptr++ << 11;
  values[0] |= *ptr++ << 3;
  values[0] |= *ptr >> 5;

  values[1] = (*ptr++ & 0x1f) << 22;
  values[1] |= *ptr++ << 14;
  values[1] |= *ptr++ << 6;
  values[1] |= *ptr >> 2;

  values[2] = static_cast<uint64_t>(*ptr++ & 3) << 25;
  values[2] |= *ptr++ << 17;
  values[2] |= *ptr++ << 9;
  values[2] |= *ptr++ << 1;
  values[2] |= *ptr >> 7;

  values[3] = (*ptr++ & 0x7f) << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = (*ptr++ & 0xf) << 23;
  values[4] |= *ptr++ << 15;
  values[4] |= *ptr++ << 7;
  values[4] |= *ptr >> 1;

  values[5] = static_cast<uint64_t>(*ptr++ & 1) << 26;
  values[5] |= *ptr++ << 18;
  values[5] |= *ptr++ << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = (*ptr++ & 0x3f) << 21;
  values[6] |= *ptr++ << 13;
  values[6] |= *ptr++ << 5;
  values[6] |= *ptr >> 3;

  values[7] = static_cast<uint64_t>(*ptr++ & 7) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits28(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 20;
  values[0] |= *ptr++ << 12;
  values[0] |= *ptr++ << 4;
  values[0] |= *ptr >> 4;

  values[1] = static_cast<uint64_t>(*ptr++ & 0xf) << 24;
  values[1] |= *ptr++ << 16;
  values[1] |= *ptr++ << 8;
  values[1] |= *ptr++;

  values[2] = *ptr++ << 20;
  values[2] |= *ptr++ << 12;
  values[2] |= *ptr++ << 4;
  values[2] |= *ptr >> 4;

  values[3] = static_cast<uint64_t>(*ptr++ & 0xf) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = *ptr++ << 20;
  values[4] |= *ptr++ << 12;
  values[4] |= *ptr++ << 4;
  values[4] |= *ptr >> 4;

  values[5] = static_cast<uint64_t>(*ptr++ & 0xf) << 24;
  values[5] |= *ptr++ << 16;
  values[5] |= *ptr++ << 8;
  values[5] |= *ptr++;

  values[6] = *ptr++ << 20;
  values[6] |= *ptr++ << 12;
  values[6] |= *ptr++ << 4;
  values[6] |= *ptr >> 4;

  values[7] = static_cast<uint64_t>(*ptr++ & 0xf) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits29(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 21;
  values[0] |= *ptr++ << 13;
  values[0] |= *ptr++ << 5;
  values[0] |= *ptr >> 3;

  values[1] = static_cast<uint64_t>(*ptr++ & 7) << 26;
  values[1] |= *ptr++ << 18;
  values[1] |= *ptr++ << 10;
  values[1] |= *ptr++ << 2;
  values[1] |= *ptr >> 6;

  values[2] = (*ptr++ & 0x3f) << 23;
  values[2] |= *ptr++ << 15;
  values[2] |= *ptr++ << 7;
  values[2] |= *ptr >> 1;

  values[3] = static_cast<uint64_t>(*ptr++ & 1) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 25;
  values[4] |= *ptr++ << 17;
  values[4] |= *ptr++ << 9;
  values[4] |= *ptr++ << 1;
  values[4] |= *ptr >> 7;

  values[5] = (*ptr++ & 0x7f) << 22;
  values[5] |= *ptr++ << 14;
  values[5] |= *ptr++ << 6;
  values[5] |= *ptr >> 2;

  values[6] = static_cast<uint64_t>(*ptr++ & 3) << 27;
  values[6] |= *ptr++ << 19;
  values[6] |= *ptr++ << 11;
  values[6] |= *ptr++ << 3;
  values[6] |= *ptr >> 5;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x1f) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits30(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 22;
  values[0] |= *ptr++ << 14;
  values[0] |= *ptr++ << 6;
  values[0] |= *ptr >> 2;

  values[1] = static_cast<uint64_t>(*ptr++ & 3) << 28;
  values[1] |= *ptr++ << 20;
  values[1] |= *ptr++ << 12;
  values[1] |= *ptr++ << 4;
  values[1] |= *ptr >> 4;

  values[2] = static_cast<uint64_t>(*ptr++ & 0xf) << 26;
  values[2] |= *ptr++ << 18;
  values[2] |= *ptr++ << 10;
  values[2] |= *ptr++ << 2;
  values[2] |= *ptr >> 6;

  values[3] = static_cast<uint64_t>(*ptr++ & 0x3f) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = *ptr++ << 22;
  values[4] |= *ptr++ << 14;
  values[4] |= *ptr++ << 6;
  values[4] |= *ptr >> 2;

  values[5] = static_cast<uint64_t>(*ptr++ & 3) << 28;
  values[5] |= *ptr++ << 20;
  values[5] |= *ptr++ << 12;
  values[5] |= *ptr++ << 4;
  values[5] |= *ptr >> 4;

  values[6] = static_cast<uint64_t>(*ptr++ & 0xf) << 26;
  values[6] |= *ptr++ << 18;
  values[6] |= *ptr++ << 10;
  values[6] |= *ptr++ << 2;
  values[6] |= *ptr >> 6;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x3f) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits31(uint64_t* values, const uint8_t* ptr) {
  values[0] = *ptr++ << 23;
  values[0] |= *ptr++ << 15;
  values[0] |= *ptr++ << 7;
  values[0] |= *ptr >> 1;

  values[1] = static_cast<uint64_t>(*ptr++ & 1) << 30;
  values[1] |= *ptr++ << 22;
  values[1] |= *ptr++ << 14;
  values[1] |= *ptr++ << 6;
  values[1] |= *ptr >> 2;

  values[2] = static_cast<uint64_t>(*ptr++ & 3) << 29;
  values[2] |= *ptr++ << 21;
  values[2] |= *ptr++ << 13;
  values[2] |= *ptr++ << 5;
  values[2] |= *ptr >> 3;

  values[3] = static_cast<uint64_t>(*ptr++ & 7) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 27;
  values[4] |= *ptr++ << 19;
  values[4] |= *ptr++ << 11;
  values[4] |= *ptr++ << 3;
  values[4] |= *ptr >> 5;

  values[5] = static_cast<uint64_t>(*ptr++ & 0x1f) << 26;
  values[5] |= *ptr++ << 18;
  values[5] |= *ptr++ << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = static_cast<uint64_t>(*ptr++ & 0x3f) << 25;
  values[6] |= *ptr++ << 17;
  values[6] |= *ptr++ << 9;
  values[6] |= *ptr++ << 1;
  values[6] |= *ptr >> 7;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x7f) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits32(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 24;
  values[0] |= *ptr++ << 16;
  values[0] |= *ptr++ << 8;
  values[0] |= *ptr++;
  values[1] = static_cast<uint64_t>(*ptr++) << 24;
  values[1] |= *ptr++ << 16;
  values[1] |= *ptr++ << 8;
  values[1] |= *ptr++;
  values[2] = static_cast<uint64_t>(*ptr++) << 24;
  values[2] |= *ptr++ << 16;
  values[2] |= *ptr++ << 8;
  values[2] |= *ptr++;
  values[3] = static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;
  values[4] = static_cast<uint64_t>(*ptr++) << 24;
  values[4] |= *ptr++ << 16;
  values[4] |= *ptr++ << 8;
  values[4] |= *ptr++;
  values[5] = static_cast<uint64_t>(*ptr++) << 24;
  values[5] |= *ptr++ << 16;
  values[5] |= *ptr++ << 8;
  values[5] |= *ptr++;
  values[6] = static_cast<uint64_t>(*ptr++) << 24;
  values[6] |= *ptr++ << 16;
  values[6] |= *ptr++ << 8;
  values[6] |= *ptr++;
  values[7] = static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits33(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 25;
  values[0] |= *ptr++ << 17;
  values[0] |= *ptr++ << 9;
  values[0] |= *ptr++ << 1;
  values[0] |= *ptr >> 7;

  values[1] = static_cast<uint64_t>(*ptr++ & 0x7f) << 26;
  values[1] |= *ptr++ << 18;
  values[1] |= *ptr++ << 10;
  values[1] |= *ptr++ << 2;
  values[1] |= *ptr >> 6;

  values[2] = static_cast<uint64_t>(*ptr++ & 0x3f) << 27;
  values[2] |= *ptr++ << 19;
  values[2] |= *ptr++ << 11;
  values[2] |= *ptr++ << 3;
  values[2] |= *ptr >> 5;

  values[3] = static_cast<uint64_t>(*ptr++ & 0x1f) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 29;
  values[4] |= *ptr++ << 21;
  values[4] |= *ptr++ << 13;
  values[4] |= *ptr++ << 5;
  values[4] |= *ptr >> 3;

  values[5] = static_cast<uint64_t>(*ptr++ & 7) << 30;
  values[5] |= *ptr++ << 22;
  values[5] |= *ptr++ << 14;
  values[5] |= *ptr++ << 6;
  values[5] |= *ptr >> 2;

  values[6] = static_cast<uint64_t>(*ptr++ & 3) << 31;
  values[6] |= *ptr++ << 23;
  values[6] |= *ptr++ << 15;
  values[6] |= *ptr++ << 7;
  values[6] |= *ptr >> 1;

  values[7] = static_cast<uint64_t>(*ptr++ & 1) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits34(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 26;
  values[0] |= *ptr++ << 18;
  values[0] |= *ptr++ << 10;
  values[0] |= *ptr++ << 2;
  values[0] |= *ptr >> 6;

  values[1] = static_cast<uint64_t>(*ptr++ & 0x3f) << 28;
  values[1] |= *ptr++ << 20;
  values[1] |= *ptr++ << 12;
  values[1] |= *ptr++ << 4;
  values[1] |= *ptr >> 4;

  values[2] = static_cast<uint64_t>(*ptr++ & 0xf) << 30;
  values[2] |= *ptr++ << 22;
  values[2] |= *ptr++ << 14;
  values[2] |= *ptr++ << 6;
  values[2] |= *ptr >> 2;

  values[3] = static_cast<uint64_t>(*ptr++ & 3) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = static_cast<uint64_t>(*ptr++) << 26;
  values[4] |= *ptr++ << 18;
  values[4] |= *ptr++ << 10;
  values[4] |= *ptr++ << 2;
  values[4] |= *ptr >> 6;

  values[5] = static_cast<uint64_t>(*ptr++ & 0x3f) << 28;
  values[5] |= *ptr++ << 20;
  values[5] |= *ptr++ << 12;
  values[5] |= *ptr++ << 4;
  values[5] |= *ptr >> 4;

  values[6] = static_cast<uint64_t>(*ptr++ & 0xf) << 30;
  values[6] |= *ptr++ << 22;
  values[6] |= *ptr++ << 14;
  values[6] |= *ptr++ << 6;
  values[6] |= *ptr >> 2;

  values[7] = static_cast<uint64_t>(*ptr++ & 3) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr++;
}

static inline void unpackBits35(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 27;
  values[0] |= *ptr++ << 19;
  values[0] |= *ptr++ << 11;
  values[0] |= *ptr++ << 3;
  values[0] |= *ptr >> 5;

  values[1] = static_cast<uint64_t>(*ptr++ & 0x1f) << 30;
  values[1] |= *ptr++ << 22;
  values[1] |= *ptr++ << 14;
  values[1] |= *ptr++ << 6;
  values[1] |= *ptr >> 2;

  values[2] = static_cast<uint64_t>(*ptr++ & 3) << 33;
  values[2] |= static_cast<uint64_t>(*ptr++) << 25;
  values[2] |= *ptr++ << 17;
  values[2] |= *ptr++ << 9;
  values[2] |= *ptr++ << 1;
  values[2] |= *ptr >> 7;

  values[3] = static_cast<uint64_t>(*ptr++ & 0x7f) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 31;
  values[4] |= *ptr++ << 23;
  values[4] |= *ptr++ << 15;
  values[4] |= *ptr++ << 7;
  values[4] |= *ptr >> 1;

  values[5] = static_cast<uint64_t>(*ptr++ & 1) << 34;
  values[5] |= static_cast<uint64_t>(*ptr++) << 26;
  values[5] |= *ptr++ << 18;
  values[5] |= *ptr++ << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = static_cast<uint64_t>(*ptr++ & 0x3f) << 29;
  values[6] |= *ptr++ << 21;
  values[6] |= *ptr++ << 13;
  values[6] |= *ptr++ << 5;
  values[6] |= *ptr >> 3;

  values[7] = static_cast<uint64_t>(*ptr++ & 7) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits36(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 28;
  values[0] |= *ptr++ << 20;
  values[0] |= *ptr++ << 12;
  values[0] |= *ptr++ << 4;
  values[0] |= *ptr >> 4;

  values[1] = static_cast<uint64_t>(*ptr++ & 0xf) << 32;
  values[1] |= static_cast<uint64_t>(*ptr++) << 24;
  values[1] |= *ptr++ << 16;
  values[1] |= *ptr++ << 8;
  values[1] |= *ptr++;

  values[2] = static_cast<uint64_t>(*ptr++) << 28;
  values[2] |= *ptr++ << 20;
  values[2] |= *ptr++ << 12;
  values[2] |= *ptr++ << 4;
  values[2] |= *ptr >> 4;

  values[3] = static_cast<uint64_t>(*ptr++ & 0xf) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = static_cast<uint64_t>(*ptr++) << 28;
  values[4] |= *ptr++ << 20;
  values[4] |= *ptr++ << 12;
  values[4] |= *ptr++ << 4;
  values[4] |= *ptr >> 4;

  values[5] = static_cast<uint64_t>(*ptr++ & 0xf) << 32;
  values[5] |= static_cast<uint64_t>(*ptr++) << 24;
  values[5] |= *ptr++ << 16;
  values[5] |= *ptr++ << 8;
  values[5] |= *ptr++;

  values[6] = static_cast<uint64_t>(*ptr++) << 28;
  values[6] |= *ptr++ << 20;
  values[6] |= *ptr++ << 12;
  values[6] |= *ptr++ << 4;
  values[6] |= *ptr >> 4;

  values[7] = static_cast<uint64_t>(*ptr++ & 0xf) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits37(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 29;
  values[0] |= *ptr++ << 21;
  values[0] |= *ptr++ << 13;
  values[0] |= *ptr++ << 5;
  values[0] |= *ptr >> 3;

  values[1] = static_cast<uint64_t>(*ptr++ & 7) << 34;
  values[1] |= static_cast<uint64_t>(*ptr++) << 26;
  values[1] |= *ptr++ << 18;
  values[1] |= *ptr++ << 10;
  values[1] |= *ptr++ << 2;
  values[1] |= *ptr >> 6;

  values[2] = static_cast<uint64_t>(*ptr++ & 0x3f) << 31;
  values[2] |= static_cast<uint64_t>(*ptr++) << 23;
  values[2] |= *ptr++ << 15;
  values[2] |= *ptr++ << 7;
  values[2] |= *ptr >> 1;

  values[3] = static_cast<uint64_t>(*ptr++ & 1) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 33;
  values[4] |= static_cast<uint64_t>(*ptr++) << 25;
  values[4] |= *ptr++ << 17;
  values[4] |= *ptr++ << 9;
  values[4] |= *ptr++ << 1;
  values[4] |= *ptr >> 7;

  values[5] = static_cast<uint64_t>(*ptr++ & 0x7f) << 30;
  values[5] |= static_cast<uint64_t>(*ptr++) << 22;
  values[5] |= *ptr++ << 14;
  values[5] |= *ptr++ << 6;
  values[5] |= *ptr >> 2;

  values[6] = static_cast<uint64_t>(*ptr++ & 3) << 35;
  values[6] |= static_cast<uint64_t>(*ptr++) << 27;
  values[6] |= *ptr++ << 19;
  values[6] |= *ptr++ << 11;
  values[6] |= *ptr++ << 3;
  values[6] |= *ptr >> 5;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x1f) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits38(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 30;
  values[0] |= *ptr++ << 22;
  values[0] |= *ptr++ << 14;
  values[0] |= *ptr++ << 6;
  values[0] |= *ptr >> 2;

  values[1] = static_cast<uint64_t>(*ptr++ & 3) << 36;
  values[1] |= static_cast<uint64_t>(*ptr++) << 28;
  values[1] |= *ptr++ << 20;
  values[1] |= *ptr++ << 12;
  values[1] |= *ptr++ << 4;
  values[1] |= *ptr >> 4;

  values[2] = static_cast<uint64_t>(*ptr++ & 0xf) << 34;
  values[2] |= static_cast<uint64_t>(*ptr++) << 26;
  values[2] |= *ptr++ << 18;
  values[2] |= *ptr++ << 10;
  values[2] |= *ptr++ << 2;
  values[2] |= *ptr >> 6;

  values[3] = static_cast<uint64_t>(*ptr++ & 0x3f) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = static_cast<uint64_t>(*ptr++) << 30;
  values[4] |= *ptr++ << 22;
  values[4] |= *ptr++ << 14;
  values[4] |= *ptr++ << 6;
  values[4] |= *ptr >> 2;

  values[5] = static_cast<uint64_t>(*ptr++ & 3) << 36;
  values[5] |= static_cast<uint64_t>(*ptr++) << 28;
  values[5] |= *ptr++ << 20;
  values[5] |= *ptr++ << 12;
  values[5] |= *ptr++ << 4;
  values[5] |= *ptr >> 4;

  values[6] = static_cast<uint64_t>(*ptr++ & 0xf) << 34;
  values[6] |= static_cast<uint64_t>(*ptr++) << 26;
  values[6] |= *ptr++ << 18;
  values[6] |= *ptr++ << 10;
  values[6] |= *ptr++ << 2;
  values[6] |= *ptr >> 6;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x3f) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits39(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 31;
  values[0] |= *ptr++ << 23;
  values[0] |= *ptr++ << 15;
  values[0] |= *ptr++ << 7;
  values[0] |= *ptr >> 1;

  values[1] = static_cast<uint64_t>(*ptr++ & 1) << 38;
  values[1] |= static_cast<uint64_t>(*ptr++) << 30;
  values[1] |= *ptr++ << 22;
  values[1] |= *ptr++ << 14;
  values[1] |= *ptr++ << 6;
  values[1] |= *ptr >> 2;

  values[2] = static_cast<uint64_t>(*ptr++ & 3) << 37;
  values[2] |= static_cast<uint64_t>(*ptr++) << 29;
  values[2] |= *ptr++ << 21;
  values[2] |= *ptr++ << 13;
  values[2] |= *ptr++ << 5;
  values[2] |= *ptr >> 3;

  values[3] = static_cast<uint64_t>(*ptr++ & 7) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 35;
  values[4] |= static_cast<uint64_t>(*ptr++) << 27;
  values[4] |= *ptr++ << 19;
  values[4] |= *ptr++ << 11;
  values[4] |= *ptr++ << 3;
  values[4] |= *ptr >> 5;

  values[5] = static_cast<uint64_t>(*ptr++ & 0x1f) << 34;
  values[5] |= static_cast<uint64_t>(*ptr++) << 26;
  values[5] |= *ptr++ << 18;
  values[5] |= *ptr++ << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = static_cast<uint64_t>(*ptr++ & 0x3f) << 33;
  values[6] |= static_cast<uint64_t>(*ptr++) << 25;
  values[6] |= *ptr++ << 17;
  values[6] |= *ptr++ << 9;
  values[6] |= *ptr++ << 1;
  values[6] |= *ptr >> 7;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x7f) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits40(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 32;
  values[0] |= static_cast<uint64_t>(*ptr++) << 24;
  values[0] |= *ptr++ << 16;
  values[0] |= *ptr++ << 8;
  values[0] |= *ptr++;
  values[1] = static_cast<uint64_t>(*ptr++) << 32;
  values[1] |= static_cast<uint64_t>(*ptr++) << 24;
  values[1] |= *ptr++ << 16;
  values[1] |= *ptr++ << 8;
  values[1] |= *ptr++;
  values[2] = static_cast<uint64_t>(*ptr++) << 32;
  values[2] |= static_cast<uint64_t>(*ptr++) << 24;
  values[2] |= *ptr++ << 16;
  values[2] |= *ptr++ << 8;
  values[2] |= *ptr++;
  values[3] = static_cast<uint64_t>(*ptr++) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;
  values[4] = static_cast<uint64_t>(*ptr++) << 32;
  values[4] |= static_cast<uint64_t>(*ptr++) << 24;
  values[4] |= *ptr++ << 16;
  values[4] |= *ptr++ << 8;
  values[4] |= *ptr++;
  values[5] = static_cast<uint64_t>(*ptr++) << 32;
  values[5] |= static_cast<uint64_t>(*ptr++) << 24;
  values[5] |= *ptr++ << 16;
  values[5] |= *ptr++ << 8;
  values[5] |= *ptr++;
  values[6] = static_cast<uint64_t>(*ptr++) << 32;
  values[6] |= static_cast<uint64_t>(*ptr++) << 24;
  values[6] |= *ptr++ << 16;
  values[6] |= *ptr++ << 8;
  values[6] |= *ptr++;
  values[7] = static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits41(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 33;
  values[0] |= static_cast<uint64_t>(*ptr++) << 25;
  values[0] |= *ptr++ << 17;
  values[0] |= *ptr++ << 9;
  values[0] |= *ptr++ << 1;
  values[0] |= *ptr >> 7;

  values[1] = static_cast<uint64_t>(*ptr++ & 0x7f) << 34;
  values[1] |= static_cast<uint64_t>(*ptr++) << 26;
  values[1] |= *ptr++ << 18;
  values[1] |= *ptr++ << 10;
  values[1] |= *ptr++ << 2;
  values[1] |= *ptr >> 6;

  values[2] = static_cast<uint64_t>(*ptr++ & 0x3f) << 35;
  values[2] |= static_cast<uint64_t>(*ptr++) << 27;
  values[2] |= *ptr++ << 19;
  values[2] |= *ptr++ << 11;
  values[2] |= *ptr++ << 3;
  values[2] |= *ptr >> 5;

  values[3] = static_cast<uint64_t>(*ptr++ & 0x1f) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 37;
  values[4] |= static_cast<uint64_t>(*ptr++) << 29;
  values[4] |= *ptr++ << 21;
  values[4] |= *ptr++ << 13;
  values[4] |= *ptr++ << 5;
  values[4] |= *ptr >> 3;

  values[5] = static_cast<uint64_t>(*ptr++ & 7) << 38;
  values[5] |= static_cast<uint64_t>(*ptr++) << 30;
  values[5] |= *ptr++ << 22;
  values[5] |= *ptr++ << 14;
  values[5] |= *ptr++ << 6;
  values[5] |= *ptr >> 2;

  values[6] = static_cast<uint64_t>(*ptr++ & 3) << 39;
  values[6] |= static_cast<uint64_t>(*ptr++) << 31;
  values[6] |= *ptr++ << 23;
  values[6] |= *ptr++ << 15;
  values[6] |= *ptr++ << 7;
  values[6] |= *ptr >> 1;

  values[7] = static_cast<uint64_t>(*ptr++ & 1) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits42(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 34;
  values[0] |= static_cast<uint64_t>(*ptr++) << 26;
  values[0] |= *ptr++ << 18;
  values[0] |= *ptr++ << 10;
  values[0] |= *ptr++ << 2;
  values[0] |= *ptr >> 6;

  values[1] = static_cast<uint64_t>(*ptr++ & 0x3f) << 36;
  values[1] |= static_cast<uint64_t>(*ptr++) << 28;
  values[1] |= *ptr++ << 20;
  values[1] |= *ptr++ << 12;
  values[1] |= *ptr++ << 4;
  values[1] |= *ptr >> 4;

  values[2] = static_cast<uint64_t>(*ptr++ & 0xf) << 38;
  values[2] |= static_cast<uint64_t>(*ptr++) << 30;
  values[2] |= *ptr++ << 22;
  values[2] |= *ptr++ << 14;
  values[2] |= *ptr++ << 6;
  values[2] |= *ptr >> 2;

  values[3] = static_cast<uint64_t>(*ptr++ & 3) << 40;
  values[3] |= static_cast<uint64_t>(*ptr++) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = static_cast<uint64_t>(*ptr++) << 34;
  values[4] |= static_cast<uint64_t>(*ptr++) << 26;
  values[4] |= *ptr++ << 18;
  values[4] |= *ptr++ << 10;
  values[4] |= *ptr++ << 2;
  values[4] |= *ptr >> 6;

  values[5] = static_cast<uint64_t>(*ptr++ & 0x3f) << 36;
  values[5] |= static_cast<uint64_t>(*ptr++) << 28;
  values[5] |= *ptr++ << 20;
  values[5] |= *ptr++ << 12;
  values[5] |= *ptr++ << 4;
  values[5] |= *ptr >> 4;

  values[6] = static_cast<uint64_t>(*ptr++ & 0xf) << 38;
  values[6] |= static_cast<uint64_t>(*ptr++) << 30;
  values[6] |= *ptr++ << 22;
  values[6] |= *ptr++ << 14;
  values[6] |= *ptr++ << 6;
  values[6] |= *ptr >> 2;

  values[7] = static_cast<uint64_t>(*ptr++ & 3) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits43(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 35;
  values[0] |= static_cast<uint64_t>(*ptr++) << 27;
  values[0] |= *ptr++ << 19;
  values[0] |= *ptr++ << 11;
  values[0] |= *ptr++ << 3;
  values[0] |= *ptr >> 5;

  values[1] = static_cast<uint64_t>(*ptr++ & 0x1f) << 38;
  values[1] |= static_cast<uint64_t>(*ptr++) << 30;
  values[1] |= *ptr++ << 22;
  values[1] |= *ptr++ << 14;
  values[1] |= *ptr++ << 6;
  values[1] |= *ptr >> 2;

  values[2] = static_cast<uint64_t>(*ptr++ & 3) << 41;
  values[2] |= static_cast<uint64_t>(*ptr++) << 33;
  values[2] |= static_cast<uint64_t>(*ptr++) << 25;
  values[2] |= *ptr++ << 17;
  values[2] |= *ptr++ << 9;
  values[2] |= *ptr++ << 1;
  values[2] |= *ptr >> 7;

  values[3] = static_cast<uint64_t>(*ptr++ & 0x7f) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 39;
  values[4] |= static_cast<uint64_t>(*ptr++) << 31;
  values[4] |= *ptr++ << 23;
  values[4] |= *ptr++ << 15;
  values[4] |= *ptr++ << 7;
  values[4] |= *ptr >> 1;

  values[5] = static_cast<uint64_t>(*ptr++ & 1) << 42;
  values[5] |= static_cast<uint64_t>(*ptr++) << 34;
  values[5] |= static_cast<uint64_t>(*ptr++) << 26;
  values[5] |= *ptr++ << 18;
  values[5] |= *ptr++ << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = static_cast<uint64_t>(*ptr++ & 0x3f) << 37;
  values[6] |= static_cast<uint64_t>(*ptr++) << 29;
  values[6] |= *ptr++ << 21;
  values[6] |= *ptr++ << 13;
  values[6] |= *ptr++ << 5;
  values[6] |= *ptr >> 3;

  values[7] = static_cast<uint64_t>(*ptr++ & 7) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits44(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 36;
  values[0] |= static_cast<uint64_t>(*ptr++) << 28;
  values[0] |= *ptr++ << 20;
  values[0] |= *ptr++ << 12;
  values[0] |= *ptr++ << 4;
  values[0] |= *ptr >> 4;

  values[1] = static_cast<uint64_t>(*ptr++ & 0xf) << 40;
  values[1] |= static_cast<uint64_t>(*ptr++) << 32;
  values[1] |= static_cast<uint64_t>(*ptr++) << 24;
  values[1] |= *ptr++ << 16;
  values[1] |= *ptr++ << 8;
  values[1] |= *ptr++;

  values[2] = static_cast<uint64_t>(*ptr++) << 36;
  values[2] |= static_cast<uint64_t>(*ptr++) << 28;
  values[2] |= *ptr++ << 20;
  values[2] |= *ptr++ << 12;
  values[2] |= *ptr++ << 4;
  values[2] |= *ptr >> 4;

  values[3] = static_cast<uint64_t>(*ptr++ & 0xf) << 40;
  values[3] |= static_cast<uint64_t>(*ptr++) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = static_cast<uint64_t>(*ptr++) << 36;
  values[4] |= static_cast<uint64_t>(*ptr++) << 28;
  values[4] |= *ptr++ << 20;
  values[4] |= *ptr++ << 12;
  values[4] |= *ptr++ << 4;
  values[4] |= *ptr >> 4;

  values[5] = static_cast<uint64_t>(*ptr++ & 0xf) << 40;
  values[5] |= static_cast<uint64_t>(*ptr++) << 32;
  values[5] |= static_cast<uint64_t>(*ptr++) << 24;
  values[5] |= *ptr++ << 16;
  values[5] |= *ptr++ << 8;
  values[5] |= *ptr++;

  values[6] = static_cast<uint64_t>(*ptr++) << 36;
  values[6] |= static_cast<uint64_t>(*ptr++) << 28;
  values[6] |= *ptr++ << 20;
  values[6] |= *ptr++ << 12;
  values[6] |= *ptr++ << 4;
  values[6] |= *ptr >> 4;

  values[7] = static_cast<uint64_t>(*ptr++ & 0xf) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits45(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 37;
  values[0] |= static_cast<uint64_t>(*ptr++) << 29;
  values[0] |= *ptr++ << 21;
  values[0] |= *ptr++ << 13;
  values[0] |= *ptr++ << 5;
  values[0] |= *ptr >> 3;

  values[1] = static_cast<uint64_t>(*ptr++ & 7) << 42;
  values[1] |= static_cast<uint64_t>(*ptr++) << 34;
  values[1] |= static_cast<uint64_t>(*ptr++) << 26;
  values[1] |= *ptr++ << 18;
  values[1] |= *ptr++ << 10;
  values[1] |= *ptr++ << 2;
  values[1] |= *ptr >> 6;

  values[2] = static_cast<uint64_t>(*ptr++ & 0x3f) << 39;
  values[2] |= static_cast<uint64_t>(*ptr++) << 31;
  values[2] |= static_cast<uint64_t>(*ptr++) << 23;
  values[2] |= *ptr++ << 15;
  values[2] |= *ptr++ << 7;
  values[2] |= *ptr >> 1;

  values[3] = static_cast<uint64_t>(*ptr++ & 1) << 44;
  values[3] |= static_cast<uint64_t>(*ptr++) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 41;
  values[4] |= static_cast<uint64_t>(*ptr++) << 33;
  values[4] |= static_cast<uint64_t>(*ptr++) << 25;
  values[4] |= *ptr++ << 17;
  values[4] |= *ptr++ << 9;
  values[4] |= *ptr++ << 1;
  values[4] |= *ptr >> 7;

  values[5] = static_cast<uint64_t>(*ptr++ & 0x7f) << 38;
  values[5] |= static_cast<uint64_t>(*ptr++) << 30;
  values[5] |= static_cast<uint64_t>(*ptr++) << 22;
  values[5] |= *ptr++ << 14;
  values[5] |= *ptr++ << 6;
  values[5] |= *ptr >> 2;

  values[6] = static_cast<uint64_t>(*ptr++ & 3) << 43;
  values[6] |= static_cast<uint64_t>(*ptr++) << 35;
  values[6] |= static_cast<uint64_t>(*ptr++) << 27;
  values[6] |= *ptr++ << 19;
  values[6] |= *ptr++ << 11;
  values[6] |= *ptr++ << 3;
  values[6] |= *ptr >> 5;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x1f) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits46(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 38;
  values[0] |= static_cast<uint64_t>(*ptr++) << 30;
  values[0] |= *ptr++ << 22;
  values[0] |= *ptr++ << 14;
  values[0] |= *ptr++ << 6;
  values[0] |= *ptr >> 2;

  values[1] = static_cast<uint64_t>(*ptr++ & 3) << 44;
  values[1] |= static_cast<uint64_t>(*ptr++) << 36;
  values[1] |= static_cast<uint64_t>(*ptr++) << 28;
  values[1] |= *ptr++ << 20;
  values[1] |= *ptr++ << 12;
  values[1] |= *ptr++ << 4;
  values[1] |= *ptr >> 4;

  values[2] = static_cast<uint64_t>(*ptr++ & 0xf) << 42;
  values[2] |= static_cast<uint64_t>(*ptr++) << 34;
  values[2] |= static_cast<uint64_t>(*ptr++) << 26;
  values[2] |= *ptr++ << 18;
  values[2] |= *ptr++ << 10;
  values[2] |= *ptr++ << 2;
  values[2] |= *ptr >> 6;

  values[3] = static_cast<uint64_t>(*ptr++ & 0x3f) << 40;
  values[3] |= static_cast<uint64_t>(*ptr++) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = static_cast<uint64_t>(*ptr++) << 38;
  values[4] |= static_cast<uint64_t>(*ptr++) << 30;
  values[4] |= *ptr++ << 22;
  values[4] |= *ptr++ << 14;
  values[4] |= *ptr++ << 6;
  values[4] |= *ptr >> 2;

  values[5] = static_cast<uint64_t>(*ptr++ & 3) << 44;
  values[5] |= static_cast<uint64_t>(*ptr++) << 36;
  values[5] |= static_cast<uint64_t>(*ptr++) << 28;
  values[5] |= *ptr++ << 20;
  values[5] |= *ptr++ << 12;
  values[5] |= *ptr++ << 4;
  values[5] |= *ptr >> 4;

  values[6] = static_cast<uint64_t>(*ptr++ & 0xf) << 42;
  values[6] |= static_cast<uint64_t>(*ptr++) << 34;
  values[6] |= static_cast<uint64_t>(*ptr++) << 26;
  values[6] |= *ptr++ << 18;
  values[6] |= *ptr++ << 10;
  values[6] |= *ptr++ << 2;
  values[6] |= *ptr >> 6;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x3f) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits47(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 39;
  values[0] |= static_cast<uint64_t>(*ptr++) << 31;
  values[0] |= *ptr++ << 23;
  values[0] |= *ptr++ << 15;
  values[0] |= *ptr++ << 7;
  values[0] |= *ptr >> 1;

  values[1] = static_cast<uint64_t>(*ptr++ & 1) << 46;
  values[1] |= static_cast<uint64_t>(*ptr++) << 38;
  values[1] |= static_cast<uint64_t>(*ptr++) << 30;
  values[1] |= *ptr++ << 22;
  values[1] |= *ptr++ << 14;
  values[1] |= *ptr++ << 6;
  values[1] |= *ptr >> 2;

  values[2] = static_cast<uint64_t>(*ptr++ & 3) << 45;
  values[2] |= static_cast<uint64_t>(*ptr++) << 37;
  values[2] |= static_cast<uint64_t>(*ptr++) << 29;
  values[2] |= *ptr++ << 21;
  values[2] |= *ptr++ << 13;
  values[2] |= *ptr++ << 5;
  values[2] |= *ptr >> 3;

  values[3] = static_cast<uint64_t>(*ptr++ & 7) << 44;
  values[3] |= static_cast<uint64_t>(*ptr++) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 43;
  values[4] |= static_cast<uint64_t>(*ptr++) << 35;
  values[4] |= static_cast<uint64_t>(*ptr++) << 27;
  values[4] |= *ptr++ << 19;
  values[4] |= *ptr++ << 11;
  values[4] |= *ptr++ << 3;
  values[4] |= *ptr >> 5;

  values[5] = static_cast<uint64_t>(*ptr++ & 0x1f) << 42;
  values[5] |= static_cast<uint64_t>(*ptr++) << 34;
  values[5] |= static_cast<uint64_t>(*ptr++) << 26;
  values[5] |= *ptr++ << 18;
  values[5] |= *ptr++ << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = static_cast<uint64_t>(*ptr++ & 0x3f) << 41;
  values[6] |= static_cast<uint64_t>(*ptr++) << 33;
  values[6] |= static_cast<uint64_t>(*ptr++) << 25;
  values[6] |= *ptr++ << 17;
  values[6] |= *ptr++ << 9;
  values[6] |= *ptr++ << 1;
  values[6] |= *ptr >> 7;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x7f) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits48(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 40;
  values[0] |= static_cast<uint64_t>(*ptr++) << 32;
  values[0] |= static_cast<uint64_t>(*ptr++) << 24;
  values[0] |= *ptr++ << 16;
  values[0] |= *ptr++ << 8;
  values[0] |= *ptr++;
  values[1] = static_cast<uint64_t>(*ptr++) << 40;
  values[1] |= static_cast<uint64_t>(*ptr++) << 32;
  values[1] |= static_cast<uint64_t>(*ptr++) << 24;
  values[1] |= *ptr++ << 16;
  values[1] |= *ptr++ << 8;
  values[1] |= *ptr++;
  values[2] = static_cast<uint64_t>(*ptr++) << 40;
  values[2] |= static_cast<uint64_t>(*ptr++) << 32;
  values[2] |= static_cast<uint64_t>(*ptr++) << 24;
  values[2] |= *ptr++ << 16;
  values[2] |= *ptr++ << 8;
  values[2] |= *ptr++;
  values[3] = static_cast<uint64_t>(*ptr++) << 40;
  values[3] |= static_cast<uint64_t>(*ptr++) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;
  values[4] = static_cast<uint64_t>(*ptr++) << 40;
  values[4] |= static_cast<uint64_t>(*ptr++) << 32;
  values[4] |= static_cast<uint64_t>(*ptr++) << 24;
  values[4] |= *ptr++ << 16;
  values[4] |= *ptr++ << 8;
  values[4] |= *ptr++;
  values[5] = static_cast<uint64_t>(*ptr++) << 40;
  values[5] |= static_cast<uint64_t>(*ptr++) << 32;
  values[5] |= static_cast<uint64_t>(*ptr++) << 24;
  values[5] |= *ptr++ << 16;
  values[5] |= *ptr++ << 8;
  values[5] |= *ptr++;
  values[6] = static_cast<uint64_t>(*ptr++) << 40;
  values[6] |= static_cast<uint64_t>(*ptr++) << 32;
  values[6] |= static_cast<uint64_t>(*ptr++) << 24;
  values[6] |= *ptr++ << 16;
  values[6] |= *ptr++ << 8;
  values[6] |= *ptr++;
  values[7] = static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits49(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 41;
  values[0] |= static_cast<uint64_t>(*ptr++) << 33;
  values[0] |= static_cast<uint64_t>(*ptr++) << 25;
  values[0] |= *ptr++ << 17;
  values[0] |= *ptr++ << 9;
  values[0] |= *ptr++ << 1;
  values[0] |= *ptr >> 7;

  values[1] = static_cast<uint64_t>(*ptr++ & 0x7f) << 42;
  values[1] |= static_cast<uint64_t>(*ptr++) << 34;
  values[1] |= static_cast<uint64_t>(*ptr++) << 26;
  values[1] |= *ptr++ << 18;
  values[1] |= *ptr++ << 10;
  values[1] |= *ptr++ << 2;
  values[1] |= *ptr >> 6;

  values[2] = static_cast<uint64_t>(*ptr++ & 0x3f) << 43;
  values[2] |= static_cast<uint64_t>(*ptr++) << 35;
  values[2] |= static_cast<uint64_t>(*ptr++) << 27;
  values[2] |= *ptr++ << 19;
  values[2] |= *ptr++ << 11;
  values[2] |= *ptr++ << 3;
  values[2] |= *ptr >> 5;

  values[3] = static_cast<uint64_t>(*ptr++ & 0x1f) << 44;
  values[3] |= static_cast<uint64_t>(*ptr++) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 45;
  values[4] |= static_cast<uint64_t>(*ptr++) << 37;
  values[4] |= static_cast<uint64_t>(*ptr++) << 29;
  values[4] |= *ptr++ << 21;
  values[4] |= *ptr++ << 13;
  values[4] |= *ptr++ << 5;
  values[4] |= *ptr >> 3;

  values[5] = static_cast<uint64_t>(*ptr++ & 7) << 46;
  values[5] |= static_cast<uint64_t>(*ptr++) << 38;
  values[5] |= static_cast<uint64_t>(*ptr++) << 30;
  values[5] |= *ptr++ << 22;
  values[5] |= *ptr++ << 14;
  values[5] |= *ptr++ << 6;
  values[5] |= *ptr >> 2;

  values[6] = static_cast<uint64_t>(*ptr++ & 3) << 47;
  values[6] |= static_cast<uint64_t>(*ptr++) << 39;
  values[6] |= static_cast<uint64_t>(*ptr++) << 31;
  values[6] |= *ptr++ << 23;
  values[6] |= *ptr++ << 15;
  values[6] |= *ptr++ << 7;
  values[6] |= *ptr >> 1;

  values[7] = static_cast<uint64_t>(*ptr++ & 1) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits50(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 42;
  values[0] |= static_cast<uint64_t>(*ptr++) << 34;
  values[0] |= static_cast<uint64_t>(*ptr++) << 26;
  values[0] |= *ptr++ << 18;
  values[0] |= *ptr++ << 10;
  values[0] |= *ptr++ << 2;
  values[0] |= *ptr >> 6;

  values[1] = static_cast<uint64_t>(*ptr++ & 0x3f) << 44;
  values[1] |= static_cast<uint64_t>(*ptr++) << 36;
  values[1] |= static_cast<uint64_t>(*ptr++) << 28;
  values[1] |= *ptr++ << 20;
  values[1] |= *ptr++ << 12;
  values[1] |= *ptr++ << 4;
  values[1] |= *ptr >> 4;

  values[2] = static_cast<uint64_t>(*ptr++ & 0xf) << 46;
  values[2] |= static_cast<uint64_t>(*ptr++) << 38;
  values[2] |= static_cast<uint64_t>(*ptr++) << 30;
  values[2] |= *ptr++ << 22;
  values[2] |= *ptr++ << 14;
  values[2] |= *ptr++ << 6;
  values[2] |= *ptr >> 2;

  values[3] = static_cast<uint64_t>(*ptr++ & 3) << 48;
  values[3] |= static_cast<uint64_t>(*ptr++) << 40;
  values[3] |= static_cast<uint64_t>(*ptr++) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = static_cast<uint64_t>(*ptr++) << 42;
  values[4] |= static_cast<uint64_t>(*ptr++) << 34;
  values[4] |= static_cast<uint64_t>(*ptr++) << 26;
  values[4] |= *ptr++ << 18;
  values[4] |= *ptr++ << 10;
  values[4] |= *ptr++ << 2;
  values[4] |= *ptr >> 6;

  values[5] = static_cast<uint64_t>(*ptr++ & 0x3f) << 44;
  values[5] |= static_cast<uint64_t>(*ptr++) << 36;
  values[5] |= static_cast<uint64_t>(*ptr++) << 28;
  values[5] |= *ptr++ << 20;
  values[5] |= *ptr++ << 12;
  values[5] |= *ptr++ << 4;
  values[5] |= *ptr >> 4;

  values[6] = static_cast<uint64_t>(*ptr++ & 0xf) << 46;
  values[6] |= static_cast<uint64_t>(*ptr++) << 38;
  values[6] |= static_cast<uint64_t>(*ptr++) << 30;
  values[6] |= *ptr++ << 22;
  values[6] |= *ptr++ << 14;
  values[6] |= *ptr++ << 6;
  values[6] |= *ptr >> 2;

  values[7] = static_cast<uint64_t>(*ptr++ & 3) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits51(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 43;
  values[0] |= static_cast<uint64_t>(*ptr++) << 35;
  values[0] |= static_cast<uint64_t>(*ptr++) << 27;
  values[0] |= *ptr++ << 19;
  values[0] |= *ptr++ << 11;
  values[0] |= *ptr++ << 3;
  values[0] |= *ptr >> 5;

  values[1] = static_cast<uint64_t>(*ptr++ & 0x1f) << 46;
  values[1] |= static_cast<uint64_t>(*ptr++) << 38;
  values[1] |= static_cast<uint64_t>(*ptr++) << 30;
  values[1] |= *ptr++ << 22;
  values[1] |= *ptr++ << 14;
  values[1] |= *ptr++ << 6;
  values[1] |= *ptr >> 2;

  values[2] = static_cast<uint64_t>(*ptr++ & 3) << 49;
  values[2] |= static_cast<uint64_t>(*ptr++) << 41;
  values[2] |= static_cast<uint64_t>(*ptr++) << 33;
  values[2] |= static_cast<uint64_t>(*ptr++) << 25;
  values[2] |= *ptr++ << 17;
  values[2] |= *ptr++ << 9;
  values[2] |= *ptr++ << 1;
  values[2] |= *ptr >> 7;

  values[3] = static_cast<uint64_t>(*ptr++ & 0x7f) << 44;
  values[3] |= static_cast<uint64_t>(*ptr++) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 47;
  values[4] |= static_cast<uint64_t>(*ptr++) << 39;
  values[4] |= static_cast<uint64_t>(*ptr++) << 31;
  values[4] |= *ptr++ << 23;
  values[4] |= *ptr++ << 15;
  values[4] |= *ptr++ << 7;
  values[4] |= *ptr >> 1;

  values[5] = static_cast<uint64_t>(*ptr++ & 1) << 50;
  values[5] |= static_cast<uint64_t>(*ptr++) << 42;
  values[5] |= static_cast<uint64_t>(*ptr++) << 34;
  values[5] |= static_cast<uint64_t>(*ptr++) << 26;
  values[5] |= *ptr++ << 18;
  values[5] |= *ptr++ << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = static_cast<uint64_t>(*ptr++ & 0x3f) << 45;
  values[6] |= static_cast<uint64_t>(*ptr++) << 37;
  values[6] |= static_cast<uint64_t>(*ptr++) << 29;
  values[6] |= *ptr++ << 21;
  values[6] |= *ptr++ << 13;
  values[6] |= *ptr++ << 5;
  values[6] |= *ptr >> 3;

  values[7] = static_cast<uint64_t>(*ptr++ & 7) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits52(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 44;
  values[0] |= static_cast<uint64_t>(*ptr++) << 36;
  values[0] |= static_cast<uint64_t>(*ptr++) << 28;
  values[0] |= *ptr++ << 20;
  values[0] |= *ptr++ << 12;
  values[0] |= *ptr++ << 4;
  values[0] |= *ptr >> 4;

  values[1] = static_cast<uint64_t>(*ptr++ & 0xf) << 48;
  values[1] |= static_cast<uint64_t>(*ptr++) << 40;
  values[1] |= static_cast<uint64_t>(*ptr++) << 32;
  values[1] |= static_cast<uint64_t>(*ptr++) << 24;
  values[1] |= *ptr++ << 16;
  values[1] |= *ptr++ << 8;
  values[1] |= *ptr++;

  values[2] = static_cast<uint64_t>(*ptr++) << 44;
  values[2] |= static_cast<uint64_t>(*ptr++) << 36;
  values[2] |= static_cast<uint64_t>(*ptr++) << 28;
  values[2] |= *ptr++ << 20;
  values[2] |= *ptr++ << 12;
  values[2] |= *ptr++ << 4;
  values[2] |= *ptr >> 4;

  values[3] = static_cast<uint64_t>(*ptr++ & 0xf) << 48;
  values[3] |= static_cast<uint64_t>(*ptr++) << 40;
  values[3] |= static_cast<uint64_t>(*ptr++) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = static_cast<uint64_t>(*ptr++) << 44;
  values[4] |= static_cast<uint64_t>(*ptr++) << 36;
  values[4] |= static_cast<uint64_t>(*ptr++) << 28;
  values[4] |= *ptr++ << 20;
  values[4] |= *ptr++ << 12;
  values[4] |= *ptr++ << 4;
  values[4] |= *ptr >> 4;

  values[5] = static_cast<uint64_t>(*ptr++ & 0xf) << 48;
  values[5] |= static_cast<uint64_t>(*ptr++) << 40;
  values[5] |= static_cast<uint64_t>(*ptr++) << 32;
  values[5] |= static_cast<uint64_t>(*ptr++) << 24;
  values[5] |= *ptr++ << 16;
  values[5] |= *ptr++ << 8;
  values[5] |= *ptr++;

  values[6] = static_cast<uint64_t>(*ptr++) << 44;
  values[6] |= static_cast<uint64_t>(*ptr++) << 36;
  values[6] |= static_cast<uint64_t>(*ptr++) << 28;
  values[6] |= *ptr++ << 20;
  values[6] |= *ptr++ << 12;
  values[6] |= *ptr++ << 4;
  values[6] |= *ptr >> 4;

  values[7] = static_cast<uint64_t>(*ptr++ & 0xf) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits53(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 45;
  values[0] |= static_cast<uint64_t>(*ptr++) << 37;
  values[0] |= static_cast<uint64_t>(*ptr++) << 29;
  values[0] |= *ptr++ << 21;
  values[0] |= *ptr++ << 13;
  values[0] |= *ptr++ << 5;
  values[0] |= *ptr >> 3;

  values[1] = static_cast<uint64_t>(*ptr++ & 7) << 50;
  values[1] |= static_cast<uint64_t>(*ptr++) << 42;
  values[1] |= static_cast<uint64_t>(*ptr++) << 34;
  values[1] |= static_cast<uint64_t>(*ptr++) << 26;
  values[1] |= *ptr++ << 18;
  values[1] |= *ptr++ << 10;
  values[1] |= *ptr++ << 2;
  values[1] |= *ptr >> 6;

  values[2] = static_cast<uint64_t>(*ptr++ & 0x3f) << 47;
  values[2] |= static_cast<uint64_t>(*ptr++) << 39;
  values[2] |= static_cast<uint64_t>(*ptr++) << 31;
  values[2] |= static_cast<uint64_t>(*ptr++) << 23;
  values[2] |= *ptr++ << 15;
  values[2] |= *ptr++ << 7;
  values[2] |= *ptr >> 1;

  values[3] = static_cast<uint64_t>(*ptr++ & 1) << 52;
  values[3] |= static_cast<uint64_t>(*ptr++) << 44;
  values[3] |= static_cast<uint64_t>(*ptr++) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 49;
  values[4] |= static_cast<uint64_t>(*ptr++) << 41;
  values[4] |= static_cast<uint64_t>(*ptr++) << 33;
  values[4] |= static_cast<uint64_t>(*ptr++) << 25;
  values[4] |= *ptr++ << 17;
  values[4] |= *ptr++ << 9;
  values[4] |= *ptr++ << 1;
  values[4] |= *ptr >> 7;

  values[5] = static_cast<uint64_t>(*ptr++ & 0x7f) << 46;
  values[5] |= static_cast<uint64_t>(*ptr++) << 38;
  values[5] |= static_cast<uint64_t>(*ptr++) << 30;
  values[5] |= *ptr++ << 22;
  values[5] |= *ptr++ << 14;
  values[5] |= *ptr++ << 6;
  values[5] |= *ptr >> 2;

  values[6] = static_cast<uint64_t>(*ptr++ & 3) << 51;
  values[6] |= static_cast<uint64_t>(*ptr++) << 43;
  values[6] |= static_cast<uint64_t>(*ptr++) << 35;
  values[6] |= static_cast<uint64_t>(*ptr++) << 27;
  values[6] |= *ptr++ << 19;
  values[6] |= *ptr++ << 11;
  values[6] |= *ptr++ << 3;
  values[6] |= *ptr >> 5;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x1f) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits54(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 46;
  values[0] |= static_cast<uint64_t>(*ptr++) << 38;
  values[0] |= static_cast<uint64_t>(*ptr++) << 30;
  values[0] |= *ptr++ << 22;
  values[0] |= *ptr++ << 14;
  values[0] |= *ptr++ << 6;
  values[0] |= *ptr >> 2;

  values[1] = static_cast<uint64_t>(*ptr++ & 3) << 52;
  values[1] |= static_cast<uint64_t>(*ptr++) << 44;
  values[1] |= static_cast<uint64_t>(*ptr++) << 36;
  values[1] |= static_cast<uint64_t>(*ptr++) << 28;
  values[1] |= *ptr++ << 20;
  values[1] |= *ptr++ << 12;
  values[1] |= *ptr++ << 4;
  values[1] |= *ptr >> 4;

  values[2] = static_cast<uint64_t>(*ptr++ & 0xf) << 50;
  values[2] |= static_cast<uint64_t>(*ptr++) << 42;
  values[2] |= static_cast<uint64_t>(*ptr++) << 34;
  values[2] |= static_cast<uint64_t>(*ptr++) << 26;
  values[2] |= *ptr++ << 18;
  values[2] |= *ptr++ << 10;
  values[2] |= *ptr++ << 2;
  values[2] |= *ptr >> 6;

  values[3] = static_cast<uint64_t>(*ptr++ & 0x3f) << 48;
  values[3] |= static_cast<uint64_t>(*ptr++) << 40;
  values[3] |= static_cast<uint64_t>(*ptr++) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = static_cast<uint64_t>(*ptr++) << 46;
  values[4] |= static_cast<uint64_t>(*ptr++) << 38;
  values[4] |= static_cast<uint64_t>(*ptr++) << 30;
  values[4] |= *ptr++ << 22;
  values[4] |= *ptr++ << 14;
  values[4] |= *ptr++ << 6;
  values[4] |= *ptr >> 2;

  values[5] = static_cast<uint64_t>(*ptr++ & 3) << 52;
  values[5] |= static_cast<uint64_t>(*ptr++) << 44;
  values[5] |= static_cast<uint64_t>(*ptr++) << 36;
  values[5] |= static_cast<uint64_t>(*ptr++) << 28;
  values[5] |= *ptr++ << 20;
  values[5] |= *ptr++ << 12;
  values[5] |= *ptr++ << 4;
  values[5] |= *ptr >> 4;

  values[6] = static_cast<uint64_t>(*ptr++ & 0xf) << 50;
  values[6] |= static_cast<uint64_t>(*ptr++) << 42;
  values[6] |= static_cast<uint64_t>(*ptr++) << 34;
  values[6] |= static_cast<uint64_t>(*ptr++) << 26;
  values[6] |= *ptr++ << 18;
  values[6] |= *ptr++ << 10;
  values[6] |= *ptr++ << 2;
  values[6] |= *ptr >> 6;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x3f) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr++;
}

static inline void unpackBits55(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 47;
  values[0] |= static_cast<uint64_t>(*ptr++) << 39;
  values[0] |= static_cast<uint64_t>(*ptr++) << 31;
  values[0] |= *ptr++ << 23;
  values[0] |= *ptr++ << 15;
  values[0] |= *ptr++ << 7;
  values[0] |= *ptr >> 1;

  values[1] = static_cast<uint64_t>(*ptr++ & 1) << 54;
  values[1] |= static_cast<uint64_t>(*ptr++) << 46;
  values[1] |= static_cast<uint64_t>(*ptr++) << 38;
  values[1] |= static_cast<uint64_t>(*ptr++) << 30;
  values[1] |= *ptr++ << 22;
  values[1] |= *ptr++ << 14;
  values[1] |= *ptr++ << 6;
  values[1] |= *ptr >> 2;

  values[2] = static_cast<uint64_t>(*ptr++ & 3) << 53;
  values[2] |= static_cast<uint64_t>(*ptr++) << 45;
  values[2] |= static_cast<uint64_t>(*ptr++) << 37;
  values[2] |= static_cast<uint64_t>(*ptr++) << 29;
  values[2] |= *ptr++ << 21;
  values[2] |= *ptr++ << 13;
  values[2] |= *ptr++ << 5;
  values[2] |= *ptr >> 3;

  values[3] = static_cast<uint64_t>(*ptr++ & 7) << 52;
  values[3] |= static_cast<uint64_t>(*ptr++) << 44;
  values[3] |= static_cast<uint64_t>(*ptr++) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 51;
  values[4] |= static_cast<uint64_t>(*ptr++) << 43;
  values[4] |= static_cast<uint64_t>(*ptr++) << 35;
  values[4] |= static_cast<uint64_t>(*ptr++) << 27;
  values[4] |= *ptr++ << 19;
  values[4] |= *ptr++ << 11;
  values[4] |= *ptr++ << 3;
  values[4] |= *ptr >> 5;

  values[5] = static_cast<uint64_t>(*ptr++ & 0x1f) << 50;
  values[5] |= static_cast<uint64_t>(*ptr++) << 42;
  values[5] |= static_cast<uint64_t>(*ptr++) << 34;
  values[5] |= static_cast<uint64_t>(*ptr++) << 26;
  values[5] |= *ptr++ << 18;
  values[5] |= *ptr++ << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = static_cast<uint64_t>(*ptr++ & 0x3f) << 49;
  values[6] |= static_cast<uint64_t>(*ptr++) << 41;
  values[6] |= static_cast<uint64_t>(*ptr++) << 33;
  values[6] |= static_cast<uint64_t>(*ptr++) << 25;
  values[6] |= *ptr++ << 17;
  values[6] |= *ptr++ << 9;
  values[6] |= *ptr++ << 1;
  values[6] |= *ptr >> 7;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x7f) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits56(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 48;
  values[0] |= static_cast<uint64_t>(*ptr++) << 40;
  values[0] |= static_cast<uint64_t>(*ptr++) << 32;
  values[0] |= static_cast<uint64_t>(*ptr++) << 24;
  values[0] |= *ptr++ << 16;
  values[0] |= *ptr++ << 8;
  values[0] |= *ptr++;
  values[1] = static_cast<uint64_t>(*ptr++) << 48;
  values[1] |= static_cast<uint64_t>(*ptr++) << 40;
  values[1] |= static_cast<uint64_t>(*ptr++) << 32;
  values[1] |= static_cast<uint64_t>(*ptr++) << 24;
  values[1] |= *ptr++ << 16;
  values[1] |= *ptr++ << 8;
  values[1] |= *ptr++;
  values[2] = static_cast<uint64_t>(*ptr++) << 48;
  values[2] |= static_cast<uint64_t>(*ptr++) << 40;
  values[2] |= static_cast<uint64_t>(*ptr++) << 32;
  values[2] |= static_cast<uint64_t>(*ptr++) << 24;
  values[2] |= *ptr++ << 16;
  values[2] |= *ptr++ << 8;
  values[2] |= *ptr++;
  values[3] = static_cast<uint64_t>(*ptr++) << 48;
  values[3] |= static_cast<uint64_t>(*ptr++) << 40;
  values[3] |= static_cast<uint64_t>(*ptr++) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;
  values[4] = static_cast<uint64_t>(*ptr++) << 48;
  values[4] |= static_cast<uint64_t>(*ptr++) << 40;
  values[4] |= static_cast<uint64_t>(*ptr++) << 32;
  values[4] |= static_cast<uint64_t>(*ptr++) << 24;
  values[4] |= *ptr++ << 16;
  values[4] |= *ptr++ << 8;
  values[4] |= *ptr++;
  values[5] = static_cast<uint64_t>(*ptr++) << 48;
  values[5] |= static_cast<uint64_t>(*ptr++) << 40;
  values[5] |= static_cast<uint64_t>(*ptr++) << 32;
  values[5] |= static_cast<uint64_t>(*ptr++) << 24;
  values[5] |= *ptr++ << 16;
  values[5] |= *ptr++ << 8;
  values[5] |= *ptr++;
  values[6] = static_cast<uint64_t>(*ptr++) << 48;
  values[6] |= static_cast<uint64_t>(*ptr++) << 40;
  values[6] |= static_cast<uint64_t>(*ptr++) << 32;
  values[6] |= static_cast<uint64_t>(*ptr++) << 24;
  values[6] |= *ptr++ << 16;
  values[6] |= *ptr++ << 8;
  values[6] |= *ptr++;
  values[7] = static_cast<uint64_t>(*ptr++) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits57(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 49;
  values[0] |= static_cast<uint64_t>(*ptr++) << 41;
  values[0] |= static_cast<uint64_t>(*ptr++) << 33;
  values[0] |= static_cast<uint64_t>(*ptr++) << 25;
  values[0] |= *ptr++ << 17;
  values[0] |= *ptr++ << 9;
  values[0] |= *ptr++ << 1;
  values[0] |= *ptr >> 7;

  values[1] = static_cast<uint64_t>(*ptr++ & 0x7f) << 50;
  values[1] |= static_cast<uint64_t>(*ptr++) << 42;
  values[1] |= static_cast<uint64_t>(*ptr++) << 34;
  values[1] |= static_cast<uint64_t>(*ptr++) << 26;
  values[1] |= *ptr++ << 18;
  values[1] |= *ptr++ << 10;
  values[1] |= *ptr++ << 2;
  values[1] |= *ptr >> 6;

  values[2] = static_cast<uint64_t>(*ptr++ & 0x3f) << 51;
  values[2] |= static_cast<uint64_t>(*ptr++) << 43;
  values[2] |= static_cast<uint64_t>(*ptr++) << 35;
  values[2] |= static_cast<uint64_t>(*ptr++) << 27;
  values[2] |= *ptr++ << 19;
  values[2] |= *ptr++ << 11;
  values[2] |= *ptr++ << 3;
  values[2] |= *ptr >> 5;

  values[3] = static_cast<uint64_t>(*ptr++ & 0x1f) << 52;
  values[3] |= static_cast<uint64_t>(*ptr++) << 44;
  values[3] |= static_cast<uint64_t>(*ptr++) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 53;
  values[4] |= static_cast<uint64_t>(*ptr++) << 45;
  values[4] |= static_cast<uint64_t>(*ptr++) << 37;
  values[4] |= static_cast<uint64_t>(*ptr++) << 29;
  values[4] |= *ptr++ << 21;
  values[4] |= *ptr++ << 13;
  values[4] |= *ptr++ << 5;
  values[4] |= *ptr >> 3;

  values[5] = static_cast<uint64_t>(*ptr++ & 7) << 54;
  values[5] |= static_cast<uint64_t>(*ptr++) << 46;
  values[5] |= static_cast<uint64_t>(*ptr++) << 38;
  values[5] |= static_cast<uint64_t>(*ptr++) << 30;
  values[5] |= *ptr++ << 22;
  values[5] |= *ptr++ << 14;
  values[5] |= *ptr++ << 6;
  values[5] |= *ptr >> 2;

  values[6] = static_cast<uint64_t>(*ptr++ & 3) << 55;
  values[6] |= static_cast<uint64_t>(*ptr++) << 47;
  values[6] |= static_cast<uint64_t>(*ptr++) << 39;
  values[6] |= static_cast<uint64_t>(*ptr++) << 31;
  values[6] |= *ptr++ << 23;
  values[6] |= *ptr++ << 15;
  values[6] |= *ptr++ << 7;
  values[6] |= *ptr >> 1;

  values[7] = static_cast<uint64_t>(*ptr++ & 1) << 56;
  values[7] |= static_cast<uint64_t>(*ptr++) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits58(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 50;
  values[0] |= static_cast<uint64_t>(*ptr++) << 42;
  values[0] |= static_cast<uint64_t>(*ptr++) << 34;
  values[0] |= static_cast<uint64_t>(*ptr++) << 26;
  values[0] |= *ptr++ << 18;
  values[0] |= *ptr++ << 10;
  values[0] |= *ptr++ << 2;
  values[0] |= *ptr >> 6;

  values[1] = static_cast<uint64_t>(*ptr++ & 0x3f) << 52;
  values[1] |= static_cast<uint64_t>(*ptr++) << 44;
  values[1] |= static_cast<uint64_t>(*ptr++) << 36;
  values[1] |= static_cast<uint64_t>(*ptr++) << 28;
  values[1] |= *ptr++ << 20;
  values[1] |= *ptr++ << 12;
  values[1] |= *ptr++ << 4;
  values[1] |= *ptr >> 4;

  values[2] = static_cast<uint64_t>(*ptr++ & 0xf) << 54;
  values[2] |= static_cast<uint64_t>(*ptr++) << 46;
  values[2] |= static_cast<uint64_t>(*ptr++) << 38;
  values[2] |= static_cast<uint64_t>(*ptr++) << 30;
  values[2] |= *ptr++ << 22;
  values[2] |= *ptr++ << 14;
  values[2] |= *ptr++ << 6;
  values[2] |= *ptr >> 2;

  values[3] = static_cast<uint64_t>(*ptr++ & 3) << 56;
  values[3] |= static_cast<uint64_t>(*ptr++) << 48;
  values[3] |= static_cast<uint64_t>(*ptr++) << 40;
  values[3] |= static_cast<uint64_t>(*ptr++) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = static_cast<uint64_t>(*ptr++) << 50;
  values[4] |= static_cast<uint64_t>(*ptr++) << 42;
  values[4] |= static_cast<uint64_t>(*ptr++) << 34;
  values[4] |= static_cast<uint64_t>(*ptr++) << 26;
  values[4] |= *ptr++ << 18;
  values[4] |= *ptr++ << 10;
  values[4] |= *ptr++ << 2;
  values[4] |= *ptr >> 6;

  values[5] = static_cast<uint64_t>(*ptr++ & 0x3f) << 52;
  values[5] |= static_cast<uint64_t>(*ptr++) << 44;
  values[5] |= static_cast<uint64_t>(*ptr++) << 36;
  values[5] |= static_cast<uint64_t>(*ptr++) << 28;
  values[5] |= *ptr++ << 20;
  values[5] |= *ptr++ << 12;
  values[5] |= *ptr++ << 4;
  values[5] |= *ptr >> 4;

  values[6] = static_cast<uint64_t>(*ptr++ & 0xf) << 54;
  values[6] |= static_cast<uint64_t>(*ptr++) << 46;
  values[6] |= static_cast<uint64_t>(*ptr++) << 38;
  values[6] |= static_cast<uint64_t>(*ptr++) << 30;
  values[6] |= *ptr++ << 22;
  values[6] |= *ptr++ << 14;
  values[6] |= *ptr++ << 6;
  values[6] |= *ptr >> 2;

  values[7] = static_cast<uint64_t>(*ptr++ & 3) << 56;
  values[7] |= static_cast<uint64_t>(*ptr++) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr++;
}

static inline void unpackBits59(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 51;
  values[0] |= static_cast<uint64_t>(*ptr++) << 43;
  values[0] |= static_cast<uint64_t>(*ptr++) << 35;
  values[0] |= static_cast<uint64_t>(*ptr++) << 27;
  values[0] |= *ptr++ << 19;
  values[0] |= *ptr++ << 11;
  values[0] |= *ptr++ << 3;
  values[0] |= *ptr >> 5;

  values[1] = static_cast<uint64_t>(*ptr++ & 0x1f) << 54;
  values[1] |= static_cast<uint64_t>(*ptr++) << 46;
  values[1] |= static_cast<uint64_t>(*ptr++) << 38;
  values[1] |= static_cast<uint64_t>(*ptr++) << 30;
  values[1] |= *ptr++ << 22;
  values[1] |= *ptr++ << 14;
  values[1] |= *ptr++ << 6;
  values[1] |= *ptr >> 2;

  values[2] = static_cast<uint64_t>(*ptr++ & 3) << 57;
  values[2] |= static_cast<uint64_t>(*ptr++) << 49;
  values[2] |= static_cast<uint64_t>(*ptr++) << 41;
  values[2] |= static_cast<uint64_t>(*ptr++) << 33;
  values[2] |= static_cast<uint64_t>(*ptr++) << 25;
  values[2] |= *ptr++ << 17;
  values[2] |= *ptr++ << 9;
  values[2] |= *ptr++ << 1;
  values[2] |= *ptr >> 7;

  values[3] = static_cast<uint64_t>(*ptr++ & 0x7f) << 52;
  values[3] |= static_cast<uint64_t>(*ptr++) << 44;
  values[3] |= static_cast<uint64_t>(*ptr++) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 55;
  values[4] |= static_cast<uint64_t>(*ptr++) << 47;
  values[4] |= static_cast<uint64_t>(*ptr++) << 39;
  values[4] |= static_cast<uint64_t>(*ptr++) << 31;
  values[4] |= *ptr++ << 23;
  values[4] |= *ptr++ << 15;
  values[4] |= *ptr++ << 7;
  values[4] |= *ptr >> 1;

  values[5] = static_cast<uint64_t>(*ptr++ & 1) << 58;
  values[5] |= static_cast<uint64_t>(*ptr++) << 50;
  values[5] |= static_cast<uint64_t>(*ptr++) << 42;
  values[5] |= static_cast<uint64_t>(*ptr++) << 34;
  values[5] |= static_cast<uint64_t>(*ptr++) << 26;
  values[5] |= *ptr++ << 18;
  values[5] |= *ptr++ << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = static_cast<uint64_t>(*ptr++ & 0x3f) << 53;
  values[6] |= static_cast<uint64_t>(*ptr++) << 45;
  values[6] |= static_cast<uint64_t>(*ptr++) << 37;
  values[6] |= static_cast<uint64_t>(*ptr++) << 29;
  values[6] |= *ptr++ << 21;
  values[6] |= *ptr++ << 13;
  values[6] |= *ptr++ << 5;
  values[6] |= *ptr >> 3;

  values[7] = static_cast<uint64_t>(*ptr++ & 7) << 56;
  values[7] |= static_cast<uint64_t>(*ptr++) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits60(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 52;
  values[0] |= static_cast<uint64_t>(*ptr++) << 44;
  values[0] |= static_cast<uint64_t>(*ptr++) << 36;
  values[0] |= static_cast<uint64_t>(*ptr++) << 28;
  values[0] |= *ptr++ << 20;
  values[0] |= *ptr++ << 12;
  values[0] |= *ptr++ << 4;
  values[0] |= *ptr >> 4;

  values[1] = static_cast<uint64_t>(*ptr++ & 0xf) << 56;
  values[1] |= static_cast<uint64_t>(*ptr++) << 48;
  values[1] |= static_cast<uint64_t>(*ptr++) << 40;
  values[1] |= static_cast<uint64_t>(*ptr++) << 32;
  values[1] |= static_cast<uint64_t>(*ptr++) << 24;
  values[1] |= *ptr++ << 16;
  values[1] |= *ptr++ << 8;
  values[1] |= *ptr++;

  values[2] = static_cast<uint64_t>(*ptr++) << 52;
  values[2] |= static_cast<uint64_t>(*ptr++) << 44;
  values[2] |= static_cast<uint64_t>(*ptr++) << 36;
  values[2] |= static_cast<uint64_t>(*ptr++) << 28;
  values[2] |= *ptr++ << 20;
  values[2] |= *ptr++ << 12;
  values[2] |= *ptr++ << 4;
  values[2] |= *ptr >> 4;

  values[3] = static_cast<uint64_t>(*ptr++ & 0xf) << 56;
  values[3] |= static_cast<uint64_t>(*ptr++) << 48;
  values[3] |= static_cast<uint64_t>(*ptr++) << 40;
  values[3] |= static_cast<uint64_t>(*ptr++) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = static_cast<uint64_t>(*ptr++) << 52;
  values[4] |= static_cast<uint64_t>(*ptr++) << 44;
  values[4] |= static_cast<uint64_t>(*ptr++) << 36;
  values[4] |= static_cast<uint64_t>(*ptr++) << 28;
  values[4] |= *ptr++ << 20;
  values[4] |= *ptr++ << 12;
  values[4] |= *ptr++ << 4;
  values[4] |= *ptr >> 4;

  values[5] = static_cast<uint64_t>(*ptr++ & 0xf) << 56;
  values[5] |= static_cast<uint64_t>(*ptr++) << 48;
  values[5] |= static_cast<uint64_t>(*ptr++) << 40;
  values[5] |= static_cast<uint64_t>(*ptr++) << 32;
  values[5] |= static_cast<uint64_t>(*ptr++) << 24;
  values[5] |= *ptr++ << 16;
  values[5] |= *ptr++ << 8;
  values[5] |= *ptr++;

  values[6] = static_cast<uint64_t>(*ptr++) << 52;
  values[6] |= static_cast<uint64_t>(*ptr++) << 44;
  values[6] |= static_cast<uint64_t>(*ptr++) << 36;
  values[6] |= static_cast<uint64_t>(*ptr++) << 28;
  values[6] |= *ptr++ << 20;
  values[6] |= *ptr++ << 12;
  values[6] |= *ptr++ << 4;
  values[6] |= *ptr >> 4;

  values[7] = static_cast<uint64_t>(*ptr++ & 0xf) << 56;
  values[7] |= static_cast<uint64_t>(*ptr++) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits61(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 53;
  values[0] |= static_cast<uint64_t>(*ptr++) << 45;
  values[0] |= static_cast<uint64_t>(*ptr++) << 37;
  values[0] |= static_cast<uint64_t>(*ptr++) << 29;
  values[0] |= *ptr++ << 21;
  values[0] |= *ptr++ << 13;
  values[0] |= *ptr++ << 5;
  values[0] |= *ptr >> 3;

  values[1] = static_cast<uint64_t>(*ptr++ & 7) << 58;
  values[1] |= static_cast<uint64_t>(*ptr++) << 50;
  values[1] |= static_cast<uint64_t>(*ptr++) << 42;
  values[1] |= static_cast<uint64_t>(*ptr++) << 34;
  values[1] |= static_cast<uint64_t>(*ptr++) << 26;
  values[1] |= *ptr++ << 18;
  values[1] |= *ptr++ << 10;
  values[1] |= *ptr++ << 2;
  values[1] |= *ptr >> 6;

  values[2] = static_cast<uint64_t>(*ptr++ & 0x3f) << 55;
  values[2] |= static_cast<uint64_t>(*ptr++) << 47;
  values[2] |= static_cast<uint64_t>(*ptr++) << 39;
  values[2] |= static_cast<uint64_t>(*ptr++) << 31;
  values[2] |= *ptr++ << 23;
  values[2] |= *ptr++ << 15;
  values[2] |= *ptr++ << 7;
  values[2] |= *ptr >> 1;

  values[3] = static_cast<uint64_t>(*ptr++ & 1) << 60;
  values[3] |= static_cast<uint64_t>(*ptr++) << 52;
  values[3] |= static_cast<uint64_t>(*ptr++) << 44;
  values[3] |= static_cast<uint64_t>(*ptr++) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 57;
  values[4] |= static_cast<uint64_t>(*ptr++) << 49;
  values[4] |= static_cast<uint64_t>(*ptr++) << 41;
  values[4] |= static_cast<uint64_t>(*ptr++) << 33;
  values[4] |= static_cast<uint64_t>(*ptr++) << 25;
  values[4] |= *ptr++ << 17;
  values[4] |= *ptr++ << 9;
  values[4] |= *ptr++ << 1;
  values[4] |= *ptr >> 7;

  values[5] = static_cast<uint64_t>(*ptr++ & 0x7f) << 54;
  values[5] |= static_cast<uint64_t>(*ptr++) << 46;
  values[5] |= static_cast<uint64_t>(*ptr++) << 38;
  values[5] |= static_cast<uint64_t>(*ptr++) << 30;
  values[5] |= *ptr++ << 22;
  values[5] |= *ptr++ << 14;
  values[5] |= *ptr++ << 6;
  values[5] |= *ptr >> 2;

  values[6] = static_cast<uint64_t>(*ptr++ & 3) << 59;
  values[6] |= static_cast<uint64_t>(*ptr++) << 51;
  values[6] |= static_cast<uint64_t>(*ptr++) << 43;
  values[6] |= static_cast<uint64_t>(*ptr++) << 35;
  values[6] |= static_cast<uint64_t>(*ptr++) << 27;
  values[6] |= *ptr++ << 19;
  values[6] |= *ptr++ << 11;
  values[6] |= *ptr++ << 3;
  values[6] |= *ptr >> 5;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x1f) << 56;
  values[7] |= static_cast<uint64_t>(*ptr++) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits62(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 54;
  values[0] |= static_cast<uint64_t>(*ptr++) << 46;
  values[0] |= static_cast<uint64_t>(*ptr++) << 38;
  values[0] |= static_cast<uint64_t>(*ptr++) << 30;
  values[0] |= *ptr++ << 22;
  values[0] |= *ptr++ << 14;
  values[0] |= *ptr++ << 6;
  values[0] |= *ptr >> 2;

  values[1] = static_cast<uint64_t>(*ptr++ & 3) << 60;
  values[1] |= static_cast<uint64_t>(*ptr++) << 52;
  values[1] |= static_cast<uint64_t>(*ptr++) << 44;
  values[1] |= static_cast<uint64_t>(*ptr++) << 36;
  values[1] |= static_cast<uint64_t>(*ptr++) << 28;
  values[1] |= *ptr++ << 20;
  values[1] |= *ptr++ << 12;
  values[1] |= *ptr++ << 4;
  values[1] |= *ptr >> 4;

  values[2] = static_cast<uint64_t>(*ptr++ & 0xf) << 58;
  values[2] |= static_cast<uint64_t>(*ptr++) << 50;
  values[2] |= static_cast<uint64_t>(*ptr++) << 42;
  values[2] |= static_cast<uint64_t>(*ptr++) << 34;
  values[2] |= static_cast<uint64_t>(*ptr++) << 26;
  values[2] |= *ptr++ << 18;
  values[2] |= *ptr++ << 10;
  values[2] |= *ptr++ << 2;
  values[2] |= *ptr >> 6;

  values[3] = static_cast<uint64_t>(*ptr++ & 0x3f) << 56;
  values[3] |= static_cast<uint64_t>(*ptr++) << 48;
  values[3] |= static_cast<uint64_t>(*ptr++) << 40;
  values[3] |= static_cast<uint64_t>(*ptr++) << 32;
  values[3] |= static_cast<uint64_t>(*ptr++) << 24;
  values[3] |= *ptr++ << 16;
  values[3] |= *ptr++ << 8;
  values[3] |= *ptr++;

  values[4] = static_cast<uint64_t>(*ptr++) << 54;
  values[4] |= static_cast<uint64_t>(*ptr++) << 46;
  values[4] |= static_cast<uint64_t>(*ptr++) << 38;
  values[4] |= static_cast<uint64_t>(*ptr++) << 30;
  values[4] |= *ptr++ << 22;
  values[4] |= *ptr++ << 14;
  values[4] |= *ptr++ << 6;
  values[4] |= *ptr >> 2;

  values[5] = static_cast<uint64_t>(*ptr++ & 3) << 60;
  values[5] |= static_cast<uint64_t>(*ptr++) << 52;
  values[5] |= static_cast<uint64_t>(*ptr++) << 44;
  values[5] |= static_cast<uint64_t>(*ptr++) << 36;
  values[5] |= static_cast<uint64_t>(*ptr++) << 28;
  values[5] |= *ptr++ << 20;
  values[5] |= *ptr++ << 12;
  values[5] |= *ptr++ << 4;
  values[5] |= *ptr >> 4;

  values[6] = static_cast<uint64_t>(*ptr++ & 0xf) << 58;
  values[6] |= static_cast<uint64_t>(*ptr++) << 50;
  values[6] |= static_cast<uint64_t>(*ptr++) << 42;
  values[6] |= static_cast<uint64_t>(*ptr++) << 34;
  values[6] |= static_cast<uint64_t>(*ptr++) << 26;
  values[6] |= *ptr++ << 18;
  values[6] |= *ptr++ << 10;
  values[6] |= *ptr++ << 2;
  values[6] |= *ptr >> 6;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x3f) << 56;
  values[7] |= static_cast<uint64_t>(*ptr++) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void unpackBits63(uint64_t* values, const uint8_t* ptr) {
  values[0] = static_cast<uint64_t>(*ptr++) << 55;
  values[0] |= static_cast<uint64_t>(*ptr++) << 47;
  values[0] |= static_cast<uint64_t>(*ptr++) << 39;
  values[0] |= static_cast<uint64_t>(*ptr++) << 31;
  values[0] |= *ptr++ << 23;
  values[0] |= *ptr++ << 15;
  values[0] |= *ptr++ << 7;
  values[0] |= *ptr >> 1;

  values[1] = static_cast<uint64_t>(*ptr++ & 1) << 62;
  values[1] |= static_cast<uint64_t>(*ptr++) << 54;
  values[1] |= static_cast<uint64_t>(*ptr++) << 46;
  values[1] |= static_cast<uint64_t>(*ptr++) << 38;
  values[1] |= static_cast<uint64_t>(*ptr++) << 30;
  values[1] |= *ptr++ << 22;
  values[1] |= *ptr++ << 14;
  values[1] |= *ptr++ << 6;
  values[1] |= *ptr >> 2;

  values[2] = static_cast<uint64_t>(*ptr++ & 3) << 61;
  values[2] |= static_cast<uint64_t>(*ptr++) << 53;
  values[2] |= static_cast<uint64_t>(*ptr++) << 45;
  values[2] |= static_cast<uint64_t>(*ptr++) << 37;
  values[2] |= static_cast<uint64_t>(*ptr++) << 29;
  values[2] |= *ptr++ << 21;
  values[2] |= *ptr++ << 13;
  values[2] |= *ptr++ << 5;
  values[2] |= *ptr >> 3;

  values[3] = static_cast<uint64_t>(*ptr++ & 7) << 60;
  values[3] |= static_cast<uint64_t>(*ptr++) << 52;
  values[3] |= static_cast<uint64_t>(*ptr++) << 44;
  values[3] |= static_cast<uint64_t>(*ptr++) << 36;
  values[3] |= static_cast<uint64_t>(*ptr++) << 28;
  values[3] |= *ptr++ << 20;
  values[3] |= *ptr++ << 12;
  values[3] |= *ptr++ << 4;
  values[3] |= *ptr >> 4;

  values[4] = static_cast<uint64_t>(*ptr++ & 0xf) << 59;
  values[4] |= static_cast<uint64_t>(*ptr++) << 51;
  values[4] |= static_cast<uint64_t>(*ptr++) << 43;
  values[4] |= static_cast<uint64_t>(*ptr++) << 35;
  values[4] |= static_cast<uint64_t>(*ptr++) << 27;
  values[4] |= *ptr++ << 19;
  values[4] |= *ptr++ << 11;
  values[4] |= *ptr++ << 3;
  values[4] |= *ptr >> 5;

  values[5] = static_cast<uint64_t>(*ptr++ & 0x1f) << 58;
  values[5] |= static_cast<uint64_t>(*ptr++) << 50;
  values[5] |= static_cast<uint64_t>(*ptr++) << 42;
  values[5] |= static_cast<uint64_t>(*ptr++) << 34;
  values[5] |= static_cast<uint64_t>(*ptr++) << 26;
  values[5] |= *ptr++ << 18;
  values[5] |= *ptr++ << 10;
  values[5] |= *ptr++ << 2;
  values[5] |= *ptr >> 6;

  values[6] = static_cast<uint64_t>(*ptr++ & 0x3f) << 57;
  values[6] |= static_cast<uint64_t>(*ptr++) << 49;
  values[6] |= static_cast<uint64_t>(*ptr++) << 41;
  values[6] |= static_cast<uint64_t>(*ptr++) << 33;
  values[6] |= static_cast<uint64_t>(*ptr++) << 25;
  values[6] |= *ptr++ << 17;
  values[6] |= *ptr++ << 9;
  values[6] |= *ptr++ << 1;
  values[6] |= *ptr >> 7;

  values[7] = static_cast<uint64_t>(*ptr++ & 0x7f) << 56;
  values[7] |= static_cast<uint64_t>(*ptr++) << 48;
  values[7] |= static_cast<uint64_t>(*ptr++) << 40;
  values[7] |= static_cast<uint64_t>(*ptr++) << 32;
  values[7] |= static_cast<uint64_t>(*ptr++) << 24;
  values[7] |= *ptr++ << 16;
  values[7] |= *ptr++ << 8;
  values[7] |= *ptr;
}

static inline void
packBitsBlock8(const uint64_t* values, uint8_t* ptr, uint8_t bits) {
  switch (bits) {
    case 1:
      packBits1(values, ptr);
      break;
    case 2:
      packBits2(values, ptr);
      break;
    case 3:
      packBits3(values, ptr);
      break;
    case 4:
      packBits4(values, ptr);
      break;
    case 5:
      packBits5(values, ptr);
      break;
    case 6:
      packBits6(values, ptr);
      break;
    case 7:
      packBits7(values, ptr);
      break;
    case 8:
      packBits8(values, ptr);
      break;
    case 9:
      packBits9(values, ptr);
      break;
    case 10:
      packBits10(values, ptr);
      break;
    case 11:
      packBits11(values, ptr);
      break;
    case 12:
      packBits12(values, ptr);
      break;
    case 13:
      packBits13(values, ptr);
      break;
    case 14:
      packBits14(values, ptr);
      break;
    case 15:
      packBits15(values, ptr);
      break;
    case 16:
      packBits16(values, ptr);
      break;
    case 17:
      packBits17(values, ptr);
      break;
    case 18:
      packBits18(values, ptr);
      break;
    case 19:
      packBits19(values, ptr);
      break;
    case 20:
      packBits20(values, ptr);
      break;
    case 21:
      packBits21(values, ptr);
      break;
    case 22:
      packBits22(values, ptr);
      break;
    case 23:
      packBits23(values, ptr);
      break;
    case 24:
      packBits24(values, ptr);
      break;
    case 25:
      packBits25(values, ptr);
      break;
    case 26:
      packBits26(values, ptr);
      break;
    case 27:
      packBits27(values, ptr);
      break;
    case 28:
      packBits28(values, ptr);
      break;
    case 29:
      packBits29(values, ptr);
      break;
    case 30:
      packBits30(values, ptr);
      break;
    case 31:
      packBits31(values, ptr);
      break;
    case 32:
      packBits32(values, ptr);
      break;
    case 33:
      packBits33(values, ptr);
      break;
    case 34:
      packBits34(values, ptr);
      break;
    case 35:
      packBits35(values, ptr);
      break;
    case 36:
      packBits36(values, ptr);
      break;
    case 37:
      packBits37(values, ptr);
      break;
    case 38:
      packBits38(values, ptr);
      break;
    case 39:
      packBits39(values, ptr);
      break;
    case 40:
      packBits40(values, ptr);
      break;
    case 41:
      packBits41(values, ptr);
      break;
    case 42:
      packBits42(values, ptr);
      break;
    case 43:
      packBits43(values, ptr);
      break;
    case 44:
      packBits44(values, ptr);
      break;
    case 45:
      packBits45(values, ptr);
      break;
    case 46:
      packBits46(values, ptr);
      break;
    case 47:
      packBits47(values, ptr);
      break;
    case 48:
      packBits48(values, ptr);
      break;
    case 49:
      packBits49(values, ptr);
      break;
    case 50:
      packBits50(values, ptr);
      break;
    case 51:
      packBits51(values, ptr);
      break;
    case 52:
      packBits52(values, ptr);
      break;
    case 53:
      packBits53(values, ptr);
      break;
    case 54:
      packBits54(values, ptr);
      break;
    case 55:
      packBits55(values, ptr);
      break;
    case 56:
      packBits56(values, ptr);
      break;
    case 57:
      packBits57(values, ptr);
      break;
    case 58:
      packBits58(values, ptr);
      break;
    case 59:
      packBits59(values, ptr);
      break;
    case 60:
      packBits60(values, ptr);
      break;
    case 61:
      packBits61(values, ptr);
      break;
    case 62:
      packBits62(values, ptr);
      break;
    case 63:
      packBits63(values, ptr);
      break;
    default:
      throw VeloxRuntimeError(
          __FILE__,
          __LINE__,
          __FUNCTION__,
          "",
          "wrong number of bits in packBitsBlock8: " + std::to_string(bits),
          error_source::kErrorSourceUser,
          error_code::kInvalidArgument,
          false /*retriable*/);
  }
}

static inline void
unpackBitsBlock8(uint64_t* values, const uint8_t* ptr, uint8_t bits) {
  switch (bits) {
    case 1:
      unpackBits1(values, ptr);
      break;
    case 2:
      unpackBits2(values, ptr);
      break;
    case 3:
      unpackBits3(values, ptr);
      break;
    case 4:
      unpackBits4(values, ptr);
      break;
    case 5:
      unpackBits5(values, ptr);
      break;
    case 6:
      unpackBits6(values, ptr);
      break;
    case 7:
      unpackBits7(values, ptr);
      break;
    case 8:
      unpackBits8(values, ptr);
      break;
    case 9:
      unpackBits9(values, ptr);
      break;
    case 10:
      unpackBits10(values, ptr);
      break;
    case 11:
      unpackBits11(values, ptr);
      break;
    case 12:
      unpackBits12(values, ptr);
      break;
    case 13:
      unpackBits13(values, ptr);
      break;
    case 14:
      unpackBits14(values, ptr);
      break;
    case 15:
      unpackBits15(values, ptr);
      break;
    case 16:
      unpackBits16(values, ptr);
      break;
    case 17:
      unpackBits17(values, ptr);
      break;
    case 18:
      unpackBits18(values, ptr);
      break;
    case 19:
      unpackBits19(values, ptr);
      break;
    case 20:
      unpackBits20(values, ptr);
      break;
    case 21:
      unpackBits21(values, ptr);
      break;
    case 22:
      unpackBits22(values, ptr);
      break;
    case 23:
      unpackBits23(values, ptr);
      break;
    case 24:
      unpackBits24(values, ptr);
      break;
    case 25:
      unpackBits25(values, ptr);
      break;
    case 26:
      unpackBits26(values, ptr);
      break;
    case 27:
      unpackBits27(values, ptr);
      break;
    case 28:
      unpackBits28(values, ptr);
      break;
    case 29:
      unpackBits29(values, ptr);
      break;
    case 30:
      unpackBits30(values, ptr);
      break;
    case 31:
      unpackBits31(values, ptr);
      break;
    case 32:
      unpackBits32(values, ptr);
      break;
    case 33:
      unpackBits33(values, ptr);
      break;
    case 34:
      unpackBits34(values, ptr);
      break;
    case 35:
      unpackBits35(values, ptr);
      break;
    case 36:
      unpackBits36(values, ptr);
      break;
    case 37:
      unpackBits37(values, ptr);
      break;
    case 38:
      unpackBits38(values, ptr);
      break;
    case 39:
      unpackBits39(values, ptr);
      break;
    case 40:
      unpackBits40(values, ptr);
      break;
    case 41:
      unpackBits41(values, ptr);
      break;
    case 42:
      unpackBits42(values, ptr);
      break;
    case 43:
      unpackBits43(values, ptr);
      break;
    case 44:
      unpackBits44(values, ptr);
      break;
    case 45:
      unpackBits45(values, ptr);
      break;
    case 46:
      unpackBits46(values, ptr);
      break;
    case 47:
      unpackBits47(values, ptr);
      break;
    case 48:
      unpackBits48(values, ptr);
      break;
    case 49:
      unpackBits49(values, ptr);
      break;
    case 50:
      unpackBits50(values, ptr);
      break;
    case 51:
      unpackBits51(values, ptr);
      break;
    case 52:
      unpackBits52(values, ptr);
      break;
    case 53:
      unpackBits53(values, ptr);
      break;
    case 54:
      unpackBits54(values, ptr);
      break;
    case 55:
      unpackBits55(values, ptr);
      break;
    case 56:
      unpackBits56(values, ptr);
      break;
    case 57:
      unpackBits57(values, ptr);
      break;
    case 58:
      unpackBits58(values, ptr);
      break;
    case 59:
      unpackBits59(values, ptr);
      break;
    case 60:
      unpackBits60(values, ptr);
      break;
    case 61:
      unpackBits61(values, ptr);
      break;
    case 62:
      unpackBits62(values, ptr);
      break;
    case 63:
      unpackBits63(values, ptr);
      break;
    default:
      throw VeloxRuntimeError(
          __FILE__,
          __LINE__,
          __FUNCTION__,
          "",
          "wrong number of bits in unpackBitsBlock8: " + std::to_string(bits),
          error_source::kErrorSourceUser,
          error_code::kInvalidArgument,
          false /*retriable*/);
  }
}

} // namespace facebook::velox::common::theta
