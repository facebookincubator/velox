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
#ifdef VELOX_ENABLE_AVX512
#pragma once

#include "velox/dwio/common/BitPackDecoderUtil.h"
#include "velox/dwio/common/UnpackBase.h"

namespace facebook::velox::dwio::common {

// Unpack numValues number of uint8_t values with bitWidth [1 - 8]
static inline void unpack1(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint8_t* FOLLY_NONNULL& result) {
  while (numValues >= 64) {
    uint64_t src_64 = *reinterpret_cast<const uint64_t*>(inputBits);
    // convert mask to 512-bit register. 0 --> 0x00, 1 --> 0xFF
    __m512i unpacked_src = _mm512_movm_epi8(src_64);
    // make 0x00 --> 0x00, 0xFF --> 0x01
    unpacked_src = _mm512_abs_epi8(unpacked_src);
    __builtin_prefetch(result + 64);
    _mm512_storeu_si512(result, unpacked_src);

    inputBits += 8;
    result += 64;
    numValues -= 64;
  }
  if (numValues > 0) {
    // reuse avx2 impl to unpack remaining values
    uint64_t numBytes = (numValues * 1 + 7) / 8;
    unpack1to8(inputBits, numBytes, numValues, 1, result);
  }
}

static inline void unpack2(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint8_t* FOLLY_NONNULL& result) {
  __m256i masks = _mm256_set1_epi32(0x03030303);
  __m256i shiftMask = _mm256_set_epi8(
      62,
      60,
      58,
      56,
      54,
      52,
      50,
      48,
      46,
      44,
      42,
      40,
      38,
      36,
      34,
      32,
      30,
      28,
      26,
      24,
      22,
      20,
      18,
      16,
      14,
      12,
      10,
      8,
      6,
      4,
      2,
      0);

  while (numValues >= 32) {
    const uint64_t* in64_pos = reinterpret_cast<const uint64_t*>(inputBits);
    __m256i data = _mm256_maskz_set1_epi64(0xFF, in64_pos[0]);
    __m256i out = _mm256_multishift_epi64_epi8(shiftMask, data);
    out = _mm256_and_si256(out, masks);
    __builtin_prefetch(result + 64);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(result), out);

    inputBits += 8;
    result += 32;
    numValues -= 32;
  }

  if (numValues > 0) {
    uint64_t numBytes = (numValues * 2 + 7) / 8;
    unpack1to8(inputBits, numBytes, numValues, 2, result);
  }
}

static inline void unpack3to7u(
    uint8_t bitWidth,
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint8_t* FOLLY_NONNULL& result) {
  uint32_t mask = kBitMask8[bitWidth];
  __m256i masks = _mm256_set1_epi32(mask);
  __m256i shiftMask = _mm256_loadu_epi32(shift_idx_table[bitWidth - 3]);
  while (numValues >= 32) {
    __m256i val = _mm256_maskz_expandloadu_epi8(mask, inputBits);
    __m256i out = _mm256_multishift_epi64_epi8(shiftMask, val);
    out = _mm256_and_si256(out, masks);
    __builtin_prefetch(result + 64);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(result), out);

    inputBits += 4 * bitWidth;
    result += 32;
    numValues -= 32;
  }

  if (numValues > 0) {
    uint64_t numBytes = (numValues * bitWidth + 7) / 8;
    unpack1to8(inputBits, numBytes, numValues, bitWidth, result);
  }
}

static inline void unpack8(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint8_t* FOLLY_NONNULL& result) {
  memcpy(result, inputBits, numValues);
  inputBits += numValues;
  result += numValues;
}

// Unpack numValues number of uint16_t values with bitWidth [1 - 16]
static inline void unpack1(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint16_t* FOLLY_NONNULL& result) {
  while (numValues >= 32) {
    uint32_t src_32 = *reinterpret_cast<const uint32_t*>(inputBits);
    // convert mask to 256-bit register. 0 --> 0x00, 1 --> 0xFF
    __m256i unpacked_src = _mm256_movm_epi8(src_32);
    // make 0x00 --> 0x00, 0xFF --> 0x01
    unpacked_src = _mm256_abs_epi8(unpacked_src);
    __m512i out = _mm512_cvtepu8_epi16(unpacked_src);
    __builtin_prefetch(result + 64);
    _mm512_storeu_si512(result, out);

    inputBits += 4;
    result += 32;
    numValues -= 32;
  }

  if (numValues > 0) {
    // reuse avx2 impl to unpack remaining values
    unpack1to4(1, inputBits, numValues, result);
  }
}

static inline void unpack2(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint16_t* FOLLY_NONNULL& result) {
  __m256i masks = _mm256_set1_epi32(0x03030303);
  __m256i shiftMask = _mm256_set_epi8(
      62,
      60,
      58,
      56,
      54,
      52,
      50,
      48,
      46,
      44,
      42,
      40,
      38,
      36,
      34,
      32,
      30,
      28,
      26,
      24,
      22,
      20,
      18,
      16,
      14,
      12,
      10,
      8,
      6,
      4,
      2,
      0);

  while (numValues >= 32) {
    const uint64_t* in64_pos = reinterpret_cast<const uint64_t*>(inputBits);
    __m256i data = _mm256_maskz_set1_epi64(0xFF, in64_pos[0]);
    __m256i tmp = _mm256_multishift_epi64_epi8(shiftMask, data);
    tmp = _mm256_and_si256(tmp, masks);
    __m512i out = _mm512_cvtepu8_epi16(tmp);
    __builtin_prefetch(result + 64);
    _mm512_storeu_si512(result, out);

    inputBits += 8;
    result += 32;
    numValues -= 32;
  }

  if (numValues > 0) {
    unpack1to4(2, inputBits, numValues, result);
  }
}

static inline void unpack3(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint16_t* FOLLY_NONNULL& result) {
  __m256i masks = _mm256_set1_epi32(0x07070707);
  __m256i shiftMask = _mm256_set_epi8(
      21,
      18,
      15,
      12,
      9,
      6,
      3,
      0,
      21,
      18,
      15,
      12,
      9,
      6,
      3,
      0,
      21,
      18,
      15,
      12,
      9,
      6,
      3,
      0,
      21,
      18,
      15,
      12,
      9,
      6,
      3,
      0);

  while (numValues >= 32) {
    __m256i val = _mm256_maskz_expandloadu_epi8(0x07070707, inputBits);
    __m256i tmp = _mm256_multishift_epi64_epi8(shiftMask, val);
    tmp = _mm256_and_si256(tmp, masks);
    __m512i out = _mm512_cvtepu8_epi16(tmp);
    __builtin_prefetch(result + 64);
    _mm512_storeu_si512(result, out);

    inputBits += 12;
    result += 32;
    numValues -= 32;
  }

  if (numValues > 0) {
    unpack1to4(3, inputBits, numValues, result);
  }
}

static inline void unpack4(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint16_t* FOLLY_NONNULL& result) {
  __m256i masks = _mm256_set1_epi32(0x0f0f0f0f);
  __m256i shiftMask = _mm256_set_epi8(
      28,
      24,
      20,
      16,
      12,
      8,
      4,
      0,
      28,
      24,
      20,
      16,
      12,
      8,
      4,
      0,
      28,
      24,
      20,
      16,
      12,
      8,
      4,
      0,
      28,
      24,
      20,
      16,
      12,
      8,
      4,
      0);

  while (numValues >= 32) {
    __m256i val = _mm256_maskz_expandloadu_epi8(0x0f0f0f0f, inputBits);
    __m256i tmp = _mm256_multishift_epi64_epi8(shiftMask, val);
    tmp = _mm256_and_si256(tmp, masks);
    __m512i out = _mm512_cvtepu8_epi16(tmp);
    __builtin_prefetch(result + 64);
    _mm512_storeu_si512(result, out);

    inputBits += 16;
    result += 32;
    numValues -= 32;
  }

  if (numValues > 0) {
    unpack1to4(4, inputBits, numValues, result);
  }
}

static inline void unpack5to7u(
    uint8_t bitWidth,
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint16_t* FOLLY_NONNULL& result) {
  uint32_t mask = kBitMask8[bitWidth];
  __m256i masks = _mm256_set1_epi32(mask);
  __m256i shiftMask = _mm256_loadu_epi32(shift_idx_table[bitWidth - 3]);

  while (numValues >= 32) {
    __m256i data = _mm256_maskz_expandloadu_epi8(mask, inputBits);
    __m256i tmp = _mm256_multishift_epi64_epi8(shiftMask, data);
    tmp = _mm256_and_si256(tmp, masks);
    __m512i out = _mm512_cvtepu8_epi16(tmp);
    __builtin_prefetch(result + 64);
    _mm512_storeu_si512(result, out);

    inputBits += 4 * bitWidth;
    result += 32;
    numValues -= 32;
  }

  if (numValues > 0) {
    unpack5to8(bitWidth, inputBits, numValues, result);
  }
}

static inline void unpack8(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint16_t* FOLLY_NONNULL& result) {
  uint64_t mask = kPdepMask8[8];
  auto writeEndOffset = result + numValues;

  // Process bitWidth bytes (8 values) a time. Note that for bitWidth 8, the
  // performance of direct memcpy is about the same as this solution.
  while (result + 8 <= writeEndOffset) {
    // Using memcpy() here may result in non-optimized loops by clong.
    uint64_t val = *reinterpret_cast<const uint64_t*>(inputBits);
    uint64_t intermediateValue = _pdep_u64(val, mask);
    __m128i out = _mm_cvtepu8_epi16(_mm_loadl_epi64(
        (reinterpret_cast<const __m128i*>(&intermediateValue))));
    __builtin_prefetch(result + 64);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(result), out);
    inputBits += 8;
    result += 8;
  }

  numValues = writeEndOffset - result;
  unpackNaive(inputBits, (8 * numValues + 7) / 8, numValues, 8, result);
}

// Unpack numValues number of uint16_t values with bitWidth [9 - 15]
static inline void unpack9to15u(
    uint8_t bitWidth,
    const uint8_t* inputBits,
    uint64_t numValues,
    uint16_t* result) {
  if (numValues >= 32) {
    __mmask32 read_mask =
        OWN_BIT_MASK(OWN_BITS_2_WORD(bitWidth * OWN_DWORD_WIDTH));
    __m512i parse_mask0 = _mm512_set1_epi16(OWN_BIT_MASK(bitWidth));

    __m512i permutex_idx_ptr[2];
    permutex_idx_ptr[0] =
        _mm512_load_si512(permutex_idx_table_16_0[bitWidth - 9]);
    permutex_idx_ptr[1] =
        _mm512_load_si512(permutex_idx_table_16_1[bitWidth - 9]);

    __m512i shift_mask_ptr[2];
    shift_mask_ptr[0] = _mm512_load_si512(shift_table_32_0[bitWidth - 9]);
    shift_mask_ptr[1] = _mm512_load_si512(shift_table_32_1[bitWidth - 9]);

    while (numValues >= 32) {
      __m512i srcmm, zmm[2];

      srcmm = _mm512_maskz_loadu_epi16(read_mask, inputBits);

      // permuting so in zmm[0] will be elements with even indexes and in zmm[1]
      // - with odd ones
      zmm[0] = _mm512_permutexvar_epi16(permutex_idx_ptr[0], srcmm);
      zmm[1] = _mm512_permutexvar_epi16(permutex_idx_ptr[1], srcmm);

      // shifting elements so they start from the start of the word
      zmm[0] = _mm512_srlv_epi32(zmm[0], shift_mask_ptr[0]);
      zmm[1] = _mm512_sllv_epi32(zmm[1], shift_mask_ptr[1]);

      // gathering even and odd elements together
      zmm[0] = _mm512_mask_mov_epi16(zmm[0], 0xAAAAAAAA, zmm[1]);
      zmm[0] = _mm512_and_si512(zmm[0], parse_mask0);

      __builtin_prefetch(result + 64);
      _mm512_storeu_si512(result, zmm[0]);

      inputBits += 4 * bitWidth;
      result += 32;
      numValues -= 32;
    }
  }

  if (numValues > 0) {
    unpackNaive(
        inputBits, (bitWidth * numValues + 7) / 8, numValues, bitWidth, result);
  }
}

static inline void unpack16u(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint16_t* FOLLY_NONNULL& result) {
  memcpy(result, inputBits, numValues * 2);
  inputBits += numValues * 2;
  result += numValues;
}

// Unpack numValues number of uint32_t values with bitWidth [1 - 32]
static inline void unpack1(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint32_t* FOLLY_NONNULL& result) {
  while (numValues >= 16) {
    uint16_t src_16 = *reinterpret_cast<const uint16_t*>(inputBits);
    // convert mask to 128-bit register. 0 --> 0x00, 1 --> 0xFF
    __m128i unpacked_src = _mm_movm_epi8(src_16);
    // make 0x00 --> 0x00, 0xFF --> 0x01
    unpacked_src = _mm_abs_epi8(unpacked_src);

    __m512i out = _mm512_cvtepu8_epi32(unpacked_src);
    __builtin_prefetch(result + 64);
    _mm512_storeu_si512(result, out);

    result += 16;
    inputBits += 2;
    numValues -= 16;
  }

  if (numValues > 0) {
    unpack1to7(1, inputBits, numValues, result);
  }
}

static inline void unpack2(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint32_t* FOLLY_NONNULL& result) {
  __m512i shiftMask = _mm512_set_epi32(
      30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
  __m512i masks = _mm512_set1_epi32(0x00000003);

  const uint32_t* in32_pos;
  while (numValues >= 16) {
    in32_pos = reinterpret_cast<const uint32_t*>(inputBits);
    __m512i data = _mm512_set1_epi32(in32_pos[0]);
    __m512i cm = _mm512_multishift_epi64_epi8(shiftMask, data);
    cm = _mm512_and_epi32(cm, masks);
    __builtin_prefetch(result + 64);
    _mm512_storeu_epi8(result, cm);

    result += 16;
    inputBits += 4;
    numValues -= 16;
  }

  if (numValues > 0) {
    unpack1to7(2, inputBits, numValues, result);
  }
}

static inline void unpack3(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint32_t* FOLLY_NONNULL& result) {
  __m512i masks = _mm512_set1_epi32(0x00000007);
  __m512i shiftMask = _mm512_set_epi32(
      45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0);

  const uint64_t* in64_pos;
  while (numValues >= 16) {
    in64_pos = reinterpret_cast<const uint64_t*>(inputBits);
    __m512i data = _mm512_set1_epi64(in64_pos[0]);
    __m512i cm = _mm512_multishift_epi64_epi8(shiftMask, data);
    cm = _mm512_and_epi32(cm, masks);
    __builtin_prefetch(result + 64);
    _mm512_storeu_epi8(result, cm);
    result += 16;
    inputBits += 6;
    numValues -= 16;
  }

  if (numValues > 0) {
    unpack1to7(3, inputBits, numValues, result);
  }
}

static inline void unpack4(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint32_t* FOLLY_NONNULL& result) {
  __m512i shiftMask = _mm512_set_epi32(
      60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0);
  __m512i masks = _mm512_set1_epi32(0x0000000f);

  while (numValues >= 16) {
    const uint64_t* in64_pos = reinterpret_cast<const uint64_t*>(inputBits);
    __m512i data = _mm512_set1_epi64(in64_pos[0]);
    __m512i cm = _mm512_multishift_epi64_epi8(shiftMask, data);
    cm = _mm512_and_epi32(cm, masks);
    __builtin_prefetch(result + 64);
    _mm512_storeu_epi8(result, cm);

    inputBits += 8;
    result += 16;
    numValues -= 16;
  }

  if (numValues > 0) {
    unpack1to7(4, inputBits, numValues, result);
  }
}

static inline void unpack5to7u(
    uint8_t bitWidth,
    const uint8_t* FOLLY_NONNULL& inputBuffer,
    uint64_t numValues,
    uint32_t* FOLLY_NONNULL& outputBuffer) {
  uint64_t pdepMask = kPdepMask8[bitWidth];

  auto writeEndOffset = outputBuffer + numValues;
  alignas(16) uint64_t intermediateValues[2];

  // Process 2 * bitWidth bytes (16 values) a time.
  while (outputBuffer + 16 <= writeEndOffset) {
    uint64_t value1 = 0;
    std::memcpy(&value1, inputBuffer, bitWidth);
    intermediateValues[0] = _pdep_u64(value1, pdepMask);

    uint64_t value2 = 0;
    std::memcpy(&value2, inputBuffer + bitWidth, bitWidth);
    intermediateValues[1] = _pdep_u64(value2, pdepMask);

    __m512i result = _mm512_cvtepu8_epi32(
        _mm_load_si128(reinterpret_cast<const __m128i*>(intermediateValues)));
    __builtin_prefetch(outputBuffer + 64);
    _mm512_storeu_si512(outputBuffer, result);

    inputBuffer += bitWidth * 2;
    outputBuffer += 16;
  }

  // Finish the last batch which has < 16 bytes. Now Process bitWidth
  // bytes (8 values) a time.
  while (outputBuffer + 8 <= writeEndOffset) {
    uint64_t value = 0;
    std::memcpy(&value, inputBuffer, bitWidth);
    uint64_t intermediateValue = _pdep_u64(value, pdepMask);

    __m256i result = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
        (reinterpret_cast<const __m128i*>(&intermediateValue))));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(outputBuffer), result);

    inputBuffer += bitWidth;
    outputBuffer += 8;
  }

  numValues = writeEndOffset - outputBuffer;
  unpackNaive(
      inputBuffer,
      (bitWidth * numValues + 7) / 8,
      numValues,
      bitWidth,
      outputBuffer);
}

static inline void unpack8u(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint32_t* FOLLY_NONNULL& result) {
  uint64_t numBytes = (numValues * 8 + 7) / 8;

  auto writeEndOffset = result + numValues;
  alignas(16) uint64_t vals[2];

  // Process bitWidth bytes (16 values) a time.
  while (result + 16 <= writeEndOffset) {
    vals[0] = *reinterpret_cast<const uint64_t*>(inputBits);
    vals[1] = *reinterpret_cast<const uint64_t*>(inputBits + 8);

    __m512i out = _mm512_cvtepu8_epi32(*((const __m128i*)vals));
    __builtin_prefetch(result + 64);
    _mm512_storeu_si512(result, out);

    inputBits += 16;
    result += 16;
  }

  // Finish the last batch which has < 16 bytes. Now process 8
  // bytes (8 values) a time.
  uint64_t val = 0;
  while (result + 8 <= writeEndOffset) {
    std::memcpy(&val, inputBits, 8);

    __m256i out = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)&val));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(result), out);

    inputBits += 8;
    result += 8;
  }

  numValues = writeEndOffset - result;
  unpackNaive(inputBits, numBytes, numValues, 8, result);
}

// Unpack numValues number of uint32_t values with bitWidth [9 - 15]
static inline void unpack9to15u(
    uint8_t bitWidth,
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint32_t* FOLLY_NONNULL& result) {
  uint32_t mask16 = kBitMask16[bitWidth];
  uint32_t mask32 = kBitMask32[bitWidth];
  __m512i masks = _mm512_set1_epi32(mask32);
  __m256i shuffleMask = _mm256_loadu_epi32(shuffle_idx_table[bitWidth - 9]);
  __m256i shiftMask = _mm256_loadu_epi32(shift_idx_table[bitWidth - 3]);
  while (numValues >= 16) {
    __m256i val = _mm256_maskz_expandloadu_epi8(mask16, inputBits);
    __m256i shuffle = _mm256_shuffle_epi8(val, shuffleMask);
    __m256i tmp = _mm256_multishift_epi64_epi8(shiftMask, shuffle);
    __m512i out = _mm512_cvtepu16_epi32(tmp);
    __builtin_prefetch(result + 128);
    _mm512_storeu_si512(result, _mm512_and_epi32(out, masks));

    inputBits += 2 * bitWidth;
    result += 16;
    numValues -= 16;
  }

  // Process remaining values w/ naive impl.
  if (numValues > 0) {
    unpackNaive(
        inputBits, (bitWidth * numValues + 7) / 8, numValues, bitWidth, result);
  }
}

static inline void unpack13(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint32_t* FOLLY_NONNULL& result) {
  __m256i am1 = _mm256_set_epi8(
      12,
      11,
      10,
      9,
      9,
      8,
      7,
      6,
      5,
      4,
      4,
      3,
      2,
      1,
      1,
      0,
      12,
      11,
      10,
      9,
      9,
      8,
      7,
      6,
      5,
      4,
      4,
      3,
      2,
      1,
      1,
      0);
  __m256i am2 = _mm256_set_epi8(
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      6,
      5,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      6,
      5,
      0,
      0,
      0,
      0,
      0,
      0);
  __m256i cm1 = _mm256_set_epi8(
      59,
      51,
      46,
      38,
      25,
      17,
      12,
      4,
      63,
      55,
      42,
      34,
      29,
      21,
      8,
      0,
      59,
      51,
      46,
      38,
      25,
      17,
      12,
      4,
      63,
      55,
      42,
      34,
      29,
      21,
      8,
      0);
  __m256i cm2 = _mm256_set_epi8(
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      55,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      55,
      0,
      0,
      0,
      0,
      0,
      0,
      0);
  __m512i masks = _mm512_set1_epi32(0x00001fff);
  __m256i mask1 = _mm256_set_epi64x(
      0xffffffffffffffff,
      0x00ffffffffffffff,
      0xffffffffffffffff,
      0x00ffffffffffffff);
  __m256i mask2 = _mm256_set_epi64x(
      0x0000000000000000,
      0x1f00000000000000,
      0x0000000000000000,
      0x1f00000000000000);
  // First unpack as many full batches as possible.
  while (numValues >= 16) {
    __m256i val = _mm256_maskz_expandloadu_epi8(0x1fff1fff, inputBits);
    __m256i bm1 = _mm256_shuffle_epi8(val, am1);
    __m256i dm1 = _mm256_multishift_epi64_epi8(cm1, bm1);
    dm1 = _mm256_and_si256(dm1, mask1);

    __m256i bm2 = _mm256_shuffle_epi8(val, am2);
    __m256i dm2 = _mm256_multishift_epi64_epi8(cm2, bm2);
    dm2 = _mm256_and_si256(dm2, mask2);

    __m256i em = _mm256_or_epi32(dm1, dm2);
    __m512i out = _mm512_cvtepu16_epi32(em);
    __builtin_prefetch(result + 64);
    _mm512_storeu_si512(result, _mm512_and_epi32(out, masks));

    result += 16;
    inputBits += 26;
    numValues -= 16;
  }

  if (numValues > 0) {
    unpackNaive(inputBits, (13 * numValues + 7) / 8, numValues, 13, result);
  }
}

static inline void unpack15(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint32_t* FOLLY_NONNULL& result) {
  __m256i am1 = _mm256_set_epi8(
      14,
      13,
      12,
      11,
      10,
      9,
      8,
      7,
      6,
      5,
      4,
      3,
      2,
      1,
      1,
      0,
      14,
      13,
      12,
      11,
      10,
      9,
      8,
      7,
      6,
      5,
      4,
      3,
      2,
      1,
      1,
      0);
  __m256i am2 = _mm256_set_epi8(
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      7,
      6,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      7,
      6,
      0,
      0,
      0,
      0,
      0,
      0);
  __m256i cm1 = _mm256_set_epi8(
      57,
      49,
      42,
      34,
      27,
      19,
      12,
      4,
      61,
      53,
      46,
      38,
      31,
      23,
      8,
      0,
      57,
      49,
      42,
      34,
      27,
      19,
      12,
      4,
      61,
      53,
      46,
      38,
      31,
      23,
      8,
      0);
  __m256i cm2 = _mm256_set_epi8(
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      53,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      53,
      0,
      0,
      0,
      0,
      0,
      0,
      0);
  __m512i masks = _mm512_set1_epi32(0x00007fff);
  __m256i mask1 = _mm256_set_epi64x(
      0xffffffffffffffff,
      0x00ffffffffffffff,
      0xffffffffffffffff,
      0x00ffffffffffffff);
  __m256i mask2 = _mm256_set_epi64x(
      0x0000000000000000,
      0x7f00000000000000,
      0x0000000000000000,
      0x7f00000000000000);
  // First unpack as many full batches as possible.
  while (numValues >= 16) {
    __m256i val = _mm256_maskz_expandloadu_epi8(0x7fff7fff, inputBits);
    __m256i bm1 = _mm256_shuffle_epi8(val, am1);
    __m256i dm1 = _mm256_multishift_epi64_epi8(cm1, bm1);
    dm1 = _mm256_and_si256(dm1, mask1);

    __m256i bm2 = _mm256_shuffle_epi8(val, am2);
    __m256i dm2 = _mm256_multishift_epi64_epi8(cm2, bm2);
    dm2 = _mm256_and_si256(dm2, mask2);

    __m256i em = _mm256_or_epi32(dm1, dm2);
    __m512i out = _mm512_cvtepu16_epi32(em);
    __builtin_prefetch(result + 64);
    _mm512_storeu_si512(result, _mm512_and_epi32(out, masks));

    result += 16;
    inputBits += 30;
    numValues -= 16;
  }

  if (numValues > 0) {
    unpackNaive(inputBits, (15 * numValues + 7) / 8, numValues, 15, result);
  }
}

static inline void unpack16u(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint32_t* FOLLY_NONNULL& result) {
  uint16_t* src16u_ptr = (uint16_t*)inputBits;

  if (numValues >= 16) {
    while (numValues >= 16) {
      __m256i src = _mm256_loadu_epi16(src16u_ptr);
      __m512i out = _mm512_cvtepu16_epi32(src);
      __builtin_prefetch(result + 64);
      _mm512_storeu_si512(result, out);

      result += 16;
      numValues -= 16;
      src16u_ptr += 16;
    }
  }
  while (numValues > 0) {
    *result = (uint32_t)(*src16u_ptr);
    result++;
    src16u_ptr++;
    numValues--;
  }
}

// Unpack numValues number of uint32_t values with bitWidth [17 - 35]
static inline void unpack17to31u(
    uint8_t bitWidth,
    const uint8_t* inputBits,
    uint64_t numValues,
    uint32_t* FOLLY_NONNULL& result) {
  if (numValues >= 16) {
    __mmask32 read_mask = OWN_BIT_MASK(bitWidth);
    __m512i parse_mask0 = _mm512_set1_epi32(OWN_BIT_MASK(bitWidth));

    __m512i permutex_idx_ptr[2];
    permutex_idx_ptr[0] =
        _mm512_load_si512(permutex_idx_table_32_0[bitWidth - 17]);
    permutex_idx_ptr[1] =
        _mm512_load_si512(permutex_idx_table_32_1[bitWidth - 17]);

    __m512i shift_mask_ptr[2];
    shift_mask_ptr[0] = _mm512_load_si512(shift_table_64_0[bitWidth - 17]);
    shift_mask_ptr[1] = _mm512_load_si512(shift_table_64_1[bitWidth - 17]);

    while (numValues >= 16) {
      __m512i srcmm, zmm[2];

      srcmm = _mm512_maskz_loadu_epi16(read_mask, inputBits);

      // permuting so in zmm[0] will be elements with even indexes and in zmm[1]
      // - with odd ones
      zmm[0] = _mm512_permutexvar_epi32(permutex_idx_ptr[0], srcmm);
      zmm[1] = _mm512_permutexvar_epi32(permutex_idx_ptr[1], srcmm);

      // shifting elements so they start from the start of the word
      zmm[0] = _mm512_srlv_epi64(zmm[0], shift_mask_ptr[0]);
      zmm[1] = _mm512_sllv_epi64(zmm[1], shift_mask_ptr[1]);

      // gathering even and odd elements together
      zmm[0] = _mm512_mask_mov_epi32(zmm[0], 0xAAAA, zmm[1]);
      zmm[0] = _mm512_and_si512(zmm[0], parse_mask0);

      __builtin_prefetch(result + 128);
      _mm512_storeu_si512(result, zmm[0]);

      inputBits += 2 * bitWidth;
      result += 16;
      numValues -= 16;
    }
  }

  if (numValues > 0) {
    unpackNaive(
        inputBits, (bitWidth * numValues + 7) / 8, numValues, bitWidth, result);
  }
}

static inline void unpack32u(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t numValues,
    uint32_t* FOLLY_NONNULL& result) {
  memcpy(result, inputBits, numValues * 4);
  inputBits += numValues * 4;
  result += numValues;
}
} // namespace facebook::velox::dwio::common
#endif