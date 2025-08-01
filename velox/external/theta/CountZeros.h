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

#include <cstdint>

namespace facebook::velox::common::theta {

static const uint8_t byteLeadingZerosTable[256] = {
    8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

static const uint8_t byteTrailingZerosTable[256] = {
    8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
    3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};

static const uint64_t FCLZ_MASK_56 = 0x00ffffffffffffff;
static const uint64_t FCLZ_MASK_48 = 0x0000ffffffffffff;
static const uint64_t FCLZ_MASK_40 = 0x000000ffffffffff;
static const uint64_t FCLZ_MASK_32 = 0x00000000ffffffff;
static const uint64_t FCLZ_MASK_24 = 0x0000000000ffffff;
static const uint64_t FCLZ_MASK_16 = 0x000000000000ffff;
static const uint64_t FCLZ_MASK_08 = 0x00000000000000ff;

static inline uint8_t countLeadingZerosInU64(uint64_t input) {
  if (input > FCLZ_MASK_56)
    return byteLeadingZerosTable[(input >> 56) & FCLZ_MASK_08];
  if (input > FCLZ_MASK_48)
    return 8 + byteLeadingZerosTable[(input >> 48) & FCLZ_MASK_08];
  if (input > FCLZ_MASK_40)
    return 16 + byteLeadingZerosTable[(input >> 40) & FCLZ_MASK_08];
  if (input > FCLZ_MASK_32)
    return 24 + byteLeadingZerosTable[(input >> 32) & FCLZ_MASK_08];
  if (input > FCLZ_MASK_24)
    return 32 + byteLeadingZerosTable[(input >> 24) & FCLZ_MASK_08];
  if (input > FCLZ_MASK_16)
    return 40 + byteLeadingZerosTable[(input >> 16) & FCLZ_MASK_08];
  if (input > FCLZ_MASK_08)
    return 48 + byteLeadingZerosTable[(input >> 8) & FCLZ_MASK_08];
  if (true)
    return 56 + byteLeadingZerosTable[(input)&FCLZ_MASK_08];
}

static inline uint8_t countLeadingZerosInU32(uint32_t input) {
  if (input > FCLZ_MASK_24)
    return byteLeadingZerosTable[(input >> 24) & FCLZ_MASK_08];
  if (input > FCLZ_MASK_16)
    return 8 + byteLeadingZerosTable[(input >> 16) & FCLZ_MASK_08];
  if (input > FCLZ_MASK_08)
    return 16 + byteLeadingZerosTable[(input >> 8) & FCLZ_MASK_08];
  if (true)
    return 24 + byteLeadingZerosTable[(input)&FCLZ_MASK_08];
}

static inline uint8_t countTrailingZerosInU32(uint32_t input) {
  for (int i = 0; i < 4; i++) {
    const int byte = input & 0xff;
    if (byte != 0)
      return static_cast<uint8_t>((i << 3) + byteLeadingZerosTable[byte]);
    input >>= 8;
  }
  return 32;
}

static inline uint8_t countTrailingZerosInU64(uint64_t input) {
  for (int i = 0; i < 8; i++) {
    const int byte = input & 0xff;
    if (byte != 0)
      return static_cast<uint8_t>((i << 3) + byteLeadingZerosTable[byte]);
    input >>= 8;
  }
  return 64;
}

} // namespace facebook::velox::common::theta
