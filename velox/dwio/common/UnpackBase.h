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

#pragma once

#include <folly/Range.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::dwio::common {

#define OWN_BIT_MASK(x) \
  (((1ULL) << (x)) - 1u) /**< Bit mask below bit position */
#define OWN_BYTE_WIDTH 8u /**< Byte width in bits */
#define OWN_WORD_WIDTH 16u /**< Word width in bits */
#define OWN_DWORD_WIDTH 32u /**< Dword width in bits */
#define OWN_BITS_2_WORD(x) \
  (((x) + 15u) >> 4u) /**< Convert a number of bits to a number of words */
#define OWN_BITS_2_DWORD(x) \
  (((x) + 31u) >>           \
   5u) /**< Convert a number of bits to a number of double words */

#if defined(_MSC_VER)
#define OWN_ALIGNED_ARRAY(array_declaration, alignment) \
  __declspec(align(alignment)) array_declaration
#elif defined(__GNUC__)
#define OWN_ALIGNED_ARRAY(array_declaration, alignment) \
  array_declaration __attribute__((aligned(alignment)))
#endif
#define OWN_ALIGNED_64_ARRAY(array_declaration) \
  OWN_ALIGNED_ARRAY(array_declaration, 64u)

static constexpr const uint32_t kBitMask8[] = {
    0x00000000,
    0x01010101,
    0x03030303,
    0x07070707,
    0x0f0f0f0f,
    0x1f1f1f1f,
    0x3f3f3f3f,
    0x7f7f7f7f,
    0xffffffff};

static constexpr const uint32_t kBitMask16[] = {
    0x00000000,
    0x00010001,
    0x00030003,
    0x00070007,
    0x000f000f,
    0x001f001f,
    0x003f003f,
    0x007f007f,
    0x00ff00ff,
    0x01ff01ff,
    0x03ff03ff,
    0x07ff07ff,
    0x0fff0fff,
    0x1fff1fff,
    0x3fff3fff,
    0x7fff7fff,
    0xffffffff};

static constexpr const uint32_t kBitMask32[] = {
    0x00000000, 0x00000001, 0x00000003, 0x00000007, 0x0000000f, 0x0000001f,
    0x0000003f, 0x0000007f, 0x000000ff, 0x000001ff, 0x000003ff, 0x000007ff,
    0x00000fff, 0x00001fff, 0x00003fff, 0x00007fff, 0x0000ffff, 0x0001ffff,
    0x0003ffff, 0x0007ffff, 0x000fffff, 0x001fffff, 0x003fffff, 0x007fffff,
    0x00ffffff, 0x01ffffff, 0x03ffffff, 0x07ffffff, 0x0fffffff, 0x1fffffff,
    0x3fffffff, 0x7fffffff, 0xffffffff};

// ------------------------------------ 3u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint8_t shift_idx_table_3u[32]) = {
    0u, 3u, 6u, 9u, 12u, 15u, 18u, 21u, 0u, 3u, 6u, 9u, 12u, 15u, 18u, 21u,
    0u, 3u, 6u, 9u, 12u, 15u, 18u, 21u, 0u, 3u, 6u, 9u, 12u, 15u, 18u, 21u};

// ------------------------------------ 4u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint8_t shift_idx_table_4u[32]) = {
    0u, 4u, 8u, 12u, 16u, 20u, 24u, 28u, 0u, 4u, 8u, 12u, 16u, 20u, 24u, 28u,
    0u, 4u, 8u, 12u, 16u, 20u, 24u, 28u, 0u, 4u, 8u, 12u, 16u, 20u, 24u, 28u};

// ------------------------------------ 5u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint8_t shift_idx_table_5u[32]) = {
    0u, 5u, 10u, 15u, 20u, 25u, 30u, 35u, 0u, 5u, 10u, 15u, 20u, 25u, 30u, 35u,
    0u, 5u, 10u, 15u, 20u, 25u, 30u, 35u, 0u, 5u, 10u, 15u, 20u, 25u, 30u, 35u};

// ------------------------------------ 6u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint8_t shift_idx_table_6u[32]) = {
    0u, 6u, 12u, 18u, 24u, 30u, 36u, 42u, 0u, 6u, 12u, 18u, 24u, 30u, 36u, 42u,
    0u, 6u, 12u, 18u, 24u, 30u, 36u, 42u, 0u, 6u, 12u, 18u, 24u, 30u, 36u, 42u};

// ------------------------------------ 7u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint8_t shift_idx_table_7u[32]) = {
    0u, 7u, 14u, 21u, 28u, 35u, 42u, 49u, 0u, 7u, 14u, 21u, 28u, 35u, 42u, 49u,
    0u, 7u, 14u, 21u, 28u, 35u, 42u, 49u, 0u, 7u, 14u, 21u, 28u, 35u, 42u, 49u};

// ------------------------------------ 9u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_9u_0[32]) = {
    0u,  1u,  1u,  2u,  2u,  3u,  3u,  4u,  4u,  5u,  5u,
    6u,  6u,  7u,  7u,  8u,  9u,  10u, 10u, 11u, 11u, 12u,
    12u, 13u, 13u, 14u, 14u, 15u, 15u, 16u, 16u, 17u};
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_9u_1[32]) = {
    0u,  1u,  1u,  2u,  2u,  3u,  3u,  4u,  5u,  6u,  6u,
    7u,  7u,  8u,  8u,  9u,  9u,  10u, 10u, 11u, 11u, 12u,
    12u, 13u, 14u, 15u, 15u, 16u, 16u, 17u, 17u, 18u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_9u_0[16]) =
    {0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u, 0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_9u_1[16]) =
    {7u, 5u, 3u, 1u, 15u, 13u, 11u, 9u, 7u, 5u, 3u, 1u, 15u, 13u, 11u, 9u};
OWN_ALIGNED_64_ARRAY(static uint8_t shuffle_idx_table_9u[32]) = {
    0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 4u, 5u, 5u, 6u, 6u, 7u, 7u, 8u,
    0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 4u, 5u, 5u, 6u, 6u, 7u, 7u, 8u};
OWN_ALIGNED_64_ARRAY(static uint8_t shift_idx_table_9u[32]) = {
    0u,  8u,  17u, 25u, 34u, 42u, 51u, 59u, 4u,  12u, 21u,
    29u, 38u, 46u, 55u, 63u, 0u,  8u,  17u, 25u, 34u, 42u,
    51u, 59u, 4u,  12u, 21u, 29u, 38u, 46u, 55u, 63u};

// ------------------------------------ 10u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_10u_0[32]) = {
    0u,  1u,  1u,  2u,  2u,  3u,  3u,  4u,  5u,  6u,  6u,
    7u,  7u,  8u,  8u,  9u,  10u, 11u, 11u, 12u, 12u, 13u,
    13u, 14u, 15u, 16u, 16u, 17u, 17u, 18u, 18u, 19u};
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_10u_1[32]) = {
    0u,  1u,  1u,  2u,  3u,  4u,  4u,  5u,  5u,  6u,  6u,
    7u,  8u,  9u,  9u,  10u, 10u, 11u, 11u, 12u, 13u, 14u,
    14u, 15u, 15u, 16u, 16u, 17u, 18u, 19u, 19u, 20u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_10u_0[16]) =
    {0u, 4u, 8u, 12u, 0u, 4u, 8u, 12u, 0u, 4u, 8u, 12u, 0u, 4u, 8u, 12u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_10u_1[16]) =
    {6u, 2u, 14u, 10u, 6u, 2u, 14u, 10u, 6u, 2u, 14u, 10u, 6u, 2u, 14u, 10u};
OWN_ALIGNED_64_ARRAY(static uint8_t shuffle_idx_table_10u[32]) = {
    0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 5u, 6u, 6u, 7u, 7u, 8u, 8u, 9u,
    0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 5u, 6u, 6u, 7u, 7u, 8u, 8u, 9u,
};
OWN_ALIGNED_64_ARRAY(static uint8_t shift_idx_table_10u[32]) = {
    0u, 8u, 18u, 26u, 36u, 44u, 54u, 62u, 0u, 8u, 18u, 26u, 36u, 44u, 54u, 62u,
    0u, 8u, 18u, 26u, 36u, 44u, 54u, 62u, 0u, 8u, 18u, 26u, 36u, 44u, 54u, 62u};

// ------------------------------------ 11u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_11u_0[32]) = {
    0u,  1u,  1u,  2u,  2u,  3u,  4u,  5u,  5u,  6u,  6u,
    7u,  8u,  9u,  9u,  10u, 11u, 12u, 12u, 13u, 13u, 14u,
    15u, 16u, 16u, 17u, 17u, 18u, 19u, 20u, 20u, 21u};
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_11u_1[32]) = {
    0u,  1u,  2u,  3u,  3u,  4u,  4u,  5u,  6u,  7u,  7u,
    8u,  8u,  9u,  10u, 11u, 11u, 12u, 13u, 14u, 14u, 15u,
    15u, 16u, 17u, 18u, 18u, 19u, 19u, 20u, 21u, 22u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_11u_0[16]) =
    {0u, 6u, 12u, 2u, 8u, 14u, 4u, 10u, 0u, 6u, 12u, 2u, 8u, 14u, 4u, 10u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_11u_1[16]) =
    {5u, 15u, 9u, 3u, 13u, 7u, 1u, 11u, 5u, 15u, 9u, 3u, 13u, 7u, 1u, 11u};
OWN_ALIGNED_64_ARRAY(static uint8_t shuffle_idx_table_11u[32]) = {
    0u, 1u, 1u, 2u, 2u, 3u, 4u, 5u, 5u, 6u, 6u, 7u, 8u, 9u, 9u, 10u,
    0u, 1u, 1u, 2u, 2u, 3u, 4u, 5u, 5u, 6u, 6u, 7u, 8u, 9u, 9u, 10u};
OWN_ALIGNED_64_ARRAY(static uint8_t shift_idx_table_11u[32]) = {
    0u,  8u,  19u, 27u, 38u, 46u, 49u, 57u, 4u,  12u, 23u,
    31u, 34u, 42u, 53u, 61u, 0u,  8u,  19u, 27u, 38u, 46u,
    49u, 57u, 4u,  12u, 23u, 31u, 34u, 42u, 53u, 61u};

// ------------------------------------ 12u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_12u_0[32]) = {
    0u,  1u,  1u,  2u,  3u,  4u,  4u,  5u,  6u,  7u,  7u,
    8u,  9u,  10u, 10u, 11u, 12u, 13u, 13u, 14u, 15u, 16u,
    16u, 17u, 18u, 19u, 19u, 20u, 21u, 22u, 22u, 23u};
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_12u_1[32]) = {
    0u,  1u,  2u,  3u,  3u,  4u,  5u,  6u,  6u,  7u,  8u,
    9u,  9u,  10u, 11u, 12u, 12u, 13u, 14u, 15u, 15u, 16u,
    17u, 18u, 18u, 19u, 20u, 21u, 21u, 22u, 23u, 24u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_12u_0[16]) =
    {0u, 8u, 0u, 8u, 0u, 8u, 0u, 8u, 0u, 8u, 0u, 8u, 0u, 8u, 0u, 8u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_12u_1[16]) =
    {4u, 12u, 4u, 12u, 4u, 12u, 4u, 12u, 4u, 12u, 4u, 12u, 4u, 12u, 4u, 12u};
OWN_ALIGNED_64_ARRAY(static uint8_t shuffle_idx_table_12u[32]) = {
    0u, 1u, 1u, 2u, 3u, 4u, 4u, 5u, 6u, 7u, 7u, 8u, 9u, 10u, 10u, 11u,
    0u, 1u, 1u, 2u, 3u, 4u, 4u, 5u, 6u, 7u, 7u, 8u, 9u, 10u, 10u, 11u};
OWN_ALIGNED_64_ARRAY(static uint8_t shift_idx_table_12u[32]) = {
    0u, 8u, 20u, 28u, 32u, 40u, 52u, 60u, 0u, 8u, 20u, 28u, 32u, 40u, 52u, 60u,
    0u, 8u, 20u, 28u, 32u, 40u, 52u, 60u, 0u, 8u, 20u, 28u, 32u, 40u, 52u, 60u};

// ------------------------------------ 13u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_13u_0[32]) = {
    0u,  1u,  1u,  2u,  3u,  4u,  4u,  5u,  6u,  7u,  8u,
    9u,  9u,  10u, 11u, 12u, 13u, 14u, 14u, 15u, 16u, 17u,
    17u, 18u, 19u, 20u, 21u, 22u, 22u, 23u, 24u, 25u};
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_13u_1[32]) = {
    0u,  1u,  2u,  3u,  4u,  5u,  5u,  6u,  7u,  8u,  8u,
    9u,  10u, 11u, 12u, 13u, 13u, 14u, 15u, 16u, 17u, 18u,
    18u, 19u, 20u, 21u, 21u, 22u, 23u, 24u, 25u, 26u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_13u_0[16]) =
    {0u, 10u, 4u, 14u, 8u, 2u, 12u, 6u, 0u, 10u, 4u, 14u, 8u, 2u, 12u, 6u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_13u_1[16]) =
    {3u, 9u, 15u, 5u, 11u, 1u, 7u, 13u, 3u, 9u, 15u, 5u, 11u, 1u, 7u, 13u};

// ------------------------------------ 14u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_14u_0[32]) = {
    0u,  1u,  1u,  2u,  3u,  4u,  5u,  6u,  7u,  8u,  8u,
    9u,  10u, 11u, 12u, 13u, 14u, 15u, 15u, 16u, 17u, 18u,
    19u, 20u, 21u, 22u, 22u, 23u, 24u, 25u, 26u, 27u};
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_14u_1[32]) = {
    0u,  1u,  2u,  3u,  4u,  5u,  6u,  7u,  7u,  8u,  9u,
    10u, 11u, 12u, 13u, 14u, 14u, 15u, 16u, 17u, 18u, 19u,
    20u, 21u, 21u, 22u, 23u, 24u, 25u, 26u, 27u, 28u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_14u_0[16]) =
    {0u, 12u, 8u, 4u, 0u, 12u, 8u, 4u, 0u, 12u, 8u, 4u, 0u, 12u, 8u, 4u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_14u_1[16]) =
    {2u, 6u, 10u, 14u, 2u, 6u, 10u, 14u, 2u, 6u, 10u, 14u, 2u, 6u, 10u, 14u};
OWN_ALIGNED_64_ARRAY(static uint8_t shuffle_idx_table_14u[32]) = {
    0u, 1u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 8u, 9u, 10u, 11u, 12u, 13u,
    0u, 1u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 8u, 9u, 10u, 11u, 12u, 13u};
OWN_ALIGNED_64_ARRAY(static uint8_t shift_idx_table_14u[32]) = {
    0u, 8u, 22u, 30u, 36u, 44u, 50u, 58u, 0u, 8u, 22u, 30u, 36u, 44u, 50u, 58u,
    0u, 8u, 22u, 30u, 36u, 44u, 50u, 58u, 0u, 8u, 22u, 30u, 36u, 44u, 50u, 58u};

// ------------------------------------ 15u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_15u_0[32]) = {
    0u,  1u,  1u,  2u,  3u,  4u,  5u,  6u,  7u,  8u,  9u,
    10u, 11u, 12u, 13u, 14u, 15u, 16u, 16u, 17u, 18u, 19u,
    20u, 21u, 22u, 23u, 24u, 25u, 26u, 27u, 28u, 29u};
OWN_ALIGNED_64_ARRAY(static uint16_t permutex_idx_table_15u_1[32]) = {
    0u,  1u,  2u,  3u,  4u,  5u,  6u,  7u,  8u,  9u,  10u,
    11u, 12u, 13u, 14u, 15u, 15u, 16u, 17u, 18u, 19u, 20u,
    21u, 22u, 23u, 24u, 25u, 26u, 27u, 28u, 29u, 30u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_15u_0[16]) =
    {0u, 14u, 12u, 10u, 8u, 6u, 4u, 2u, 0u, 14u, 12u, 10u, 8u, 6u, 4u, 2u};
OWN_ALIGNED_64_ARRAY(static uint32_t shift_table_15u_1[16]) =
    {1u, 3u, 5u, 7u, 9u, 11u, 13u, 15u, 1u, 3u, 5u, 7u, 9u, 11u, 13u, 15u};

// ------------------------------------ 17u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_17u_0[16]) =
    {0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 4u, 5u, 5u, 6u, 6u, 7u, 7u, 8u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_17u_1[16]) =
    {0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 4u, 5u, 5u, 6u, 6u, 7u, 7u, 8u};
OWN_ALIGNED_64_ARRAY(
    static uint64_t shift_table_17u_0[8]) = {0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u};
OWN_ALIGNED_64_ARRAY(
    static uint64_t shift_table_17u_1[8]) = {15u, 13u, 11u, 9u, 7u, 5u, 3u, 1u};

// ------------------------------------ 18u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_18u_0[16]) =
    {0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 4u, 5u, 5u, 6u, 6u, 7u, 7u, 8u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_18u_1[16]) =
    {0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 5u, 6u, 6u, 7u, 7u, 8u, 8u, 9u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_18u_0[8]) =
    {0u, 4u, 8u, 12u, 16u, 20u, 24u, 28u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_18u_1[8]) =
    {14u, 10u, 6u, 2u, 30u, 26u, 22u, 18u};

// ------------------------------------ 19u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_19u_0[16]) =
    {0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 4u, 5u, 5u, 6u, 7u, 8u, 8u, 9u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_19u_1[16]) =
    {0u, 1u, 1u, 2u, 2u, 3u, 4u, 5u, 5u, 6u, 6u, 7u, 7u, 8u, 8u, 9u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_19u_0[8]) =
    {0u, 6u, 12u, 18u, 24u, 30u, 4u, 10u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_19u_1[8]) =
    {13u, 7u, 1u, 27u, 21u, 15u, 9u, 3u};

// ------------------------------------ 20u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_20u_0[16]) =
    {0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 5u, 6u, 6u, 7u, 7u, 8u, 8u, 9u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_20u_1[16]) =
    {0u, 1u, 1u, 2u, 3u, 4u, 4u, 5u, 5u, 6u, 6u, 7u, 8u, 9u, 9u, 10u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_20u_0[8]) =
    {0u, 8u, 16u, 24u, 0u, 8u, 16u, 24u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_20u_1[8]) =
    {12u, 4u, 28u, 20u, 12u, 4u, 28u, 20u};

// ------------------------------------ 21u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_21u_0[16]) =
    {0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 5u, 6u, 6u, 7u, 7u, 8u, 9u, 10u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_21u_1[16]) =
    {0u, 1u, 1u, 2u, 3u, 4u, 4u, 5u, 5u, 6u, 7u, 8u, 8u, 9u, 9u, 10u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_21u_0[8]) =
    {0u, 10u, 20u, 30u, 8u, 18u, 28u, 6u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_21u_1[8]) =
    {11u, 1u, 23u, 13u, 3u, 25u, 15u, 5u};

// ------------------------------------ 22u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_22u_0[16]) =
    {0u, 1u, 1u, 2u, 2u, 3u, 4u, 5u, 5u, 6u, 6u, 7u, 8u, 9u, 9u, 10u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_22u_1[16]) =
    {0u, 1u, 2u, 3u, 3u, 4u, 4u, 5u, 6u, 7u, 7u, 8u, 8u, 9u, 10u, 11u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_22u_0[8]) =
    {0u, 12u, 24u, 4u, 16u, 28u, 8u, 20u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_22u_1[8]) =
    {10u, 30u, 18u, 6u, 26u, 14u, 2u, 22u};

// ------------------------------------ 23u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_23u_0[16]) =
    {0u, 1u, 1u, 2u, 2u, 3u, 4u, 5u, 5u, 6u, 7u, 8u, 8u, 9u, 10u, 11u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_23u_1[16]) =
    {0u, 1u, 2u, 3u, 3u, 4u, 5u, 6u, 6u, 7u, 7u, 8u, 9u, 10u, 10u, 11u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_23u_0[8]) =
    {0u, 14u, 28u, 10u, 24u, 6u, 20u, 2u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_23u_1[8]) =
    {9u, 27u, 13u, 31u, 17u, 3u, 21u, 7u};

// ------------------------------------ 24u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_24u_0[16]) =
    {0u, 1u, 1u, 2u, 3u, 4u, 4u, 5u, 6u, 7u, 7u, 8u, 9u, 10u, 10u, 11u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_24u_1[16]) =
    {0u, 1u, 2u, 3u, 3u, 4u, 5u, 6u, 6u, 7u, 8u, 9u, 9u, 10u, 11u, 12u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_24u_0[8]) =
    {0u, 16u, 0u, 16u, 0u, 16u, 0u, 16u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_24u_1[8]) =
    {8u, 24u, 8u, 24u, 8u, 24u, 8u, 24u};

// ------------------------------------ 25u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_25u_0[16]) =
    {0u, 1u, 1u, 2u, 3u, 4u, 4u, 5u, 6u, 7u, 7u, 8u, 9u, 10u, 10u, 11u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_25u_1[16]) =
    {0u, 1u, 2u, 3u, 3u, 4u, 5u, 6u, 7u, 8u, 8u, 9u, 10u, 11u, 11u, 12u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_25u_0[8]) =
    {0u, 18u, 4u, 22u, 8u, 26u, 12u, 30u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_25u_1[8]) =
    {7u, 21u, 3u, 17u, 31u, 13u, 27u, 9u};

// ------------------------------------ 26u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_26u_0[16]) =
    {0u, 1u, 1u, 2u, 3u, 4u, 4u, 5u, 6u, 7u, 8u, 9u, 9u, 10u, 11u, 12u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_26u_1[16]) =
    {0u, 1u, 2u, 3u, 4u, 5u, 5u, 6u, 7u, 8u, 8u, 9u, 10u, 11u, 12u, 13u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_26u_0[8]) =
    {0u, 20u, 8u, 28u, 16u, 4u, 24u, 12u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_26u_1[8]) =
    {6u, 18u, 30u, 10u, 22u, 2u, 14u, 26u};

// ------------------------------------ 27u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_27u_0[16]) =
    {0u, 1u, 1u, 2u, 3u, 4u, 5u, 6u, 6u, 7u, 8u, 9u, 10u, 11u, 11u, 12u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_27u_1[16]) =
    {0u, 1u, 2u, 3u, 4u, 5u, 5u, 6u, 7u, 8u, 9u, 10u, 10u, 11u, 12u, 13u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_27u_0[8]) =
    {0u, 22u, 12u, 2u, 24u, 14u, 4u, 26u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_27u_1[8]) =
    {5u, 15u, 25u, 3u, 13u, 23u, 1u, 11u};

// ------------------------------------ 28u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_28u_0[16]) =
    {0u, 1u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 8u, 9u, 10u, 11u, 12u, 13u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_28u_1[16]) =
    {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_28u_0[8]) =
    {0u, 24u, 16u, 8u, 0u, 24u, 16u, 8u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_28u_1[8]) =
    {4u, 12u, 20u, 28u, 4u, 12u, 20u, 28u};

// ------------------------------------ 29u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_29u_0[16]) =
    {0u, 1u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 10u, 11u, 12u, 13u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_29u_1[16]) =
    {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 9u, 10u, 11u, 12u, 13u, 14u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_29u_0[8]) =
    {0u, 26u, 20u, 14u, 8u, 2u, 28u, 22u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_29u_1[8]) =
    {3u, 9u, 15u, 21u, 27u, 1u, 7u, 13u};

// ------------------------------------ 30u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_30u_0[16]) =
    {0u, 1u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_30u_1[16]) =
    {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_30u_0[8]) =
    {0u, 28u, 24u, 20u, 16u, 12u, 8u, 4u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_30u_1[8]) =
    {2u, 6u, 10u, 14u, 18u, 22u, 26u, 30u};

// ------------------------------------ 31u
// -----------------------------------------
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_31u_0[16]) =
    {0u, 1u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u};
OWN_ALIGNED_64_ARRAY(static uint32_t permutex_idx_table_31u_1[16]) =
    {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u};
OWN_ALIGNED_64_ARRAY(static uint64_t shift_table_31u_0[8]) =
    {0u, 30u, 28u, 26u, 24u, 22u, 20u, 18u};
OWN_ALIGNED_64_ARRAY(
    static uint64_t shift_table_31u_1[8]) = {1u, 3u, 5u, 7u, 9u, 11u, 13u, 15u};

OWN_ALIGNED_64_ARRAY(static uint8_t* shuffle_idx_table[6]) = {
    shuffle_idx_table_9u,
    shuffle_idx_table_10u,
    shuffle_idx_table_11u,
    shuffle_idx_table_12u,
    {},
    shuffle_idx_table_14u};

OWN_ALIGNED_64_ARRAY(static uint8_t* shift_idx_table[12]) = {
    shift_idx_table_3u,
    shift_idx_table_4u,
    shift_idx_table_5u,
    shift_idx_table_6u,
    shift_idx_table_7u,
    {},
    shift_idx_table_9u,
    shift_idx_table_10u,
    shift_idx_table_11u,
    shift_idx_table_12u,
    {},
    shift_idx_table_14u};

OWN_ALIGNED_64_ARRAY(static uint16_t* permutex_idx_table_16_0[7]) = {
    permutex_idx_table_9u_0,
    permutex_idx_table_10u_0,
    permutex_idx_table_11u_0,
    permutex_idx_table_12u_0,
    permutex_idx_table_13u_0,
    permutex_idx_table_14u_0,
    permutex_idx_table_15u_0};

OWN_ALIGNED_64_ARRAY(static uint32_t* permutex_idx_table_32_0[15]) = {
    permutex_idx_table_17u_0,
    permutex_idx_table_18u_0,
    permutex_idx_table_19u_0,
    permutex_idx_table_20u_0,
    permutex_idx_table_21u_0,
    permutex_idx_table_22u_0,
    permutex_idx_table_23u_0,
    permutex_idx_table_24u_0,
    permutex_idx_table_25u_0,
    permutex_idx_table_26u_0,
    permutex_idx_table_27u_0,
    permutex_idx_table_28u_0,
    permutex_idx_table_29u_0,
    permutex_idx_table_30u_0,
    permutex_idx_table_31u_0};

OWN_ALIGNED_64_ARRAY(static uint16_t* permutex_idx_table_16_1[7]) = {
    permutex_idx_table_9u_1,
    permutex_idx_table_10u_1,
    permutex_idx_table_11u_1,
    permutex_idx_table_12u_1,
    permutex_idx_table_13u_1,
    permutex_idx_table_14u_1,
    permutex_idx_table_15u_1};

OWN_ALIGNED_64_ARRAY(static uint32_t* permutex_idx_table_32_1[15]) = {
    permutex_idx_table_17u_1,
    permutex_idx_table_18u_1,
    permutex_idx_table_19u_1,
    permutex_idx_table_20u_1,
    permutex_idx_table_21u_1,
    permutex_idx_table_22u_1,
    permutex_idx_table_23u_1,
    permutex_idx_table_24u_1,
    permutex_idx_table_25u_1,
    permutex_idx_table_26u_1,
    permutex_idx_table_27u_1,
    permutex_idx_table_28u_1,
    permutex_idx_table_29u_1,
    permutex_idx_table_30u_1,
    permutex_idx_table_31u_1};

OWN_ALIGNED_64_ARRAY(static uint32_t* shift_table_32_0[7]) = {
    shift_table_9u_0,
    shift_table_10u_0,
    shift_table_11u_0,
    shift_table_12u_0,
    shift_table_13u_0,
    shift_table_14u_0,
    shift_table_15u_0};

OWN_ALIGNED_64_ARRAY(static uint64_t* shift_table_64_0[15]) = {
    shift_table_17u_0,
    shift_table_18u_0,
    shift_table_19u_0,
    shift_table_20u_0,
    shift_table_21u_0,
    shift_table_22u_0,
    shift_table_23u_0,
    shift_table_24u_0,
    shift_table_25u_0,
    shift_table_26u_0,
    shift_table_27u_0,
    shift_table_28u_0,
    shift_table_29u_0,
    shift_table_30u_0,
    shift_table_31u_0};

OWN_ALIGNED_64_ARRAY(static uint32_t* shift_table_32_1[7]) = {
    shift_table_9u_1,
    shift_table_10u_1,
    shift_table_11u_1,
    shift_table_12u_1,
    shift_table_13u_1,
    shift_table_14u_1,
    shift_table_15u_1};

OWN_ALIGNED_64_ARRAY(static uint64_t* shift_table_64_1[15]) = {
    shift_table_17u_1,
    shift_table_18u_1,
    shift_table_19u_1,
    shift_table_20u_1,
    shift_table_21u_1,
    shift_table_22u_1,
    shift_table_23u_1,
    shift_table_24u_1,
    shift_table_25u_1,
    shift_table_26u_1,
    shift_table_27u_1,
    shift_table_28u_1,
    shift_table_29u_1,
    shift_table_30u_1,
    shift_table_31u_1};
} // namespace facebook::velox::dwio::common