/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

// Adapted from Apache Arrow.

#pragma once

#include "arrow/util/endian.h"
#include "arrow/util/simd.h"
#include "arrow/util/ubsan.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#ifdef ARROW_HAVE_SSE4_2
// Enable the SIMD for ByteStreamSplit Encoder/Decoder.
#define ARROW_HAVE_SIMD_SPLIT
#endif // ARROW_HAVE_SSE4_2

namespace facebook::velox::parquet::arrow {

//
// SIMD implementations.
//

#if defined(ARROW_HAVE_SSE4_2)
template <typename T>
void byteStreamSplitDecodeSse2(
    const uint8_t* data,
    int64_t numValues,
    int64_t stride,
    T* out) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(
      kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  constexpr size_t kNumStreamsLog2 = (kNumStreams == 8U ? 3U : 2U);
  constexpr int64_t kBlockSize = sizeof(__m128i) * kNumStreams;

  const int64_t size = numValues * sizeof(T);
  const int64_t numBlocks = size / kBlockSize;
  uint8_t* outputData = reinterpret_cast<uint8_t*>(out);

  // First handle suffix.
  // This helps catch if the simd-based processing overflows into the suffix.
  // Since almost surely a test would fail.
  const int64_t numProcessedElements = (numBlocks * kBlockSize) / kNumStreams;
  for (int64_t i = numProcessedElements; i < numValues; ++i) {
    uint8_t gatheredByteData[kNumStreams];
    for (size_t b = 0; b < kNumStreams; ++b) {
      const size_t byteIndex = b * stride + i;
      gatheredByteData[b] = data[byteIndex];
    }
    out[i] = arrow::util::SafeLoadAs<T>(&gatheredByteData[0]);
  }

  // The blocks get processed hierarchically using the unpack intrinsics.
  // Example with four streams:
  // Stage 1: AAAA BBBB CCCC DDDD.
  // Stage 2: ACAC ACAC BDBD BDBD.
  // Stage 3: ABCD ABCD ABCD ABCD.
  __m128i stage[kNumStreamsLog2 + 1U][kNumStreams];
  constexpr size_t kNumStreamsHalf = kNumStreams / 2U;

  for (int64_t i = 0; i < numBlocks; ++i) {
    for (size_t j = 0; j < kNumStreams; ++j) {
      stage[0][j] = mmLoaduSi128(
          reinterpret_cast<const __m128i*>(
              &data[i * sizeof(__m128i) + j * stride]));
    }
    for (size_t step = 0; step < kNumStreamsLog2; ++step) {
      for (size_t j = 0; j < kNumStreamsHalf; ++j) {
        stage[step + 1U][j * 2] =
            mmUnpackloEpi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
        stage[step + 1U][j * 2 + 1U] =
            mmUnpackhiEpi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
      }
    }
    for (size_t j = 0; j < kNumStreams; ++j) {
      mmStoreuSi128(
          reinterpret_cast<__m128i*>(
              &outputData[(i * kNumStreams + j) * sizeof(__m128i)]),
          stage[kNumStreamsLog2][j]);
    }
  }
}

template <typename T>
void byteStreamSplitEncodeSse2(
    const uint8_t* rawValues,
    const size_t numValues,
    uint8_t* outputBufferRaw) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(
      kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  constexpr size_t kBlockSize = sizeof(__m128i) * kNumStreams;

  __m128i stage[3][kNumStreams];
  __m128i finalResult[kNumStreams];

  const size_t size = numValues * sizeof(T);
  const size_t numBlocks = size / kBlockSize;
  const __m128i* rawValuesSse = reinterpret_cast<const __m128i*>(rawValues);
  __m128i* outputBufferStreams[kNumStreams];
  for (size_t i = 0; i < kNumStreams; ++i) {
    outputBufferStreams[i] =
        reinterpret_cast<__m128i*>(&outputBufferRaw[numValues * i]);
  }

  // First handle suffix.
  const size_t numProcessedElements = (numBlocks * kBlockSize) / sizeof(T);
  for (size_t i = numProcessedElements; i < numValues; ++i) {
    for (size_t j = 0U; j < kNumStreams; ++j) {
      const uint8_t byteInValue = rawValues[i * kNumStreams + j];
      outputBufferRaw[j * numValues + i] = byteInValue;
    }
  }
  // The current shuffling algorithm diverges for float and double types but
  // the. Compiler should be able to remove the branch since only one path is
  // taken. For each template instantiation. Example run for floats: Step 0,
  // copy:
  //   0: ABCD ABCD ABCD ABCD 1: ABCD ABCD ABCD ABCD ...
  // Step 1: _mm_unpacklo_epi8 and mm_unpackhi_epi8:
  //   0: AABB CCDD AABB CCDD 1: AABB CCDD AABB CCDD ...
  //   0: AAAA BBBB CCCC DDDD 1: AAAA BBBB CCCC DDDD ...
  // Step 3: __mm_unpacklo_epi8 and _mm_unpackhi_epi8:
  //   0: AAAA AAAA BBBB BBBB 1: CCCC CCCC DDDD DDDD ...
  // Step 4: __mm_unpacklo_epi64 and _mm_unpackhi_epi64:
  //   0: AAAA AAAA AAAA AAAA 1: BBBB BBBB BBBB BBBB ...
  for (size_t blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
    // First copy the data to stage 0.
    for (size_t i = 0; i < kNumStreams; ++i) {
      stage[0][i] = mmLoaduSi128(&rawValuesSse[blockIndex * kNumStreams + i]);
    }

    // The shuffling of bytes is performed through the unpack intrinsics.
    // In my measurements this gives better performance then an implementation
    // which uses the shuffle intrinsics.
    for (size_t stageLvl = 0; stageLvl < 2U; ++stageLvl) {
      for (size_t i = 0; i < kNumStreams / 2U; ++i) {
        stage[stageLvl + 1][i * 2] =
            mmUnpackloEpi8(stage[stageLvl][i * 2], stage[stageLvl][i * 2 + 1]);
        stage[stageLvl + 1][i * 2 + 1] =
            mmUnpackhiEpi8(stage[stageLvl][i * 2], stage[stageLvl][i * 2 + 1]);
      }
    }
    if constexpr (kNumStreams == 8U) {
      // This is the path for double.
      __m128i tmp[8];
      for (size_t i = 0; i < 4; ++i) {
        tmp[i * 2] = mmUnpackloEpi32(stage[2][i], stage[2][i + 4]);
        tmp[i * 2 + 1] = mmUnpackhiEpi32(stage[2][i], stage[2][i + 4]);
      }

      for (size_t i = 0; i < 4; ++i) {
        finalResult[i * 2] = mmUnpackloEpi32(tmp[i], tmp[i + 4]);
        finalResult[i * 2 + 1] = mmUnpackhiEpi32(tmp[i], tmp[i + 4]);
      }
    } else {
      // This is the path for float.
      __m128i tmp[4];
      for (size_t i = 0; i < 2; ++i) {
        tmp[i * 2] = mmUnpackloEpi8(stage[2][i * 2], stage[2][i * 2 + 1]);
        tmp[i * 2 + 1] = mmUnpackhiEpi8(stage[2][i * 2], stage[2][i * 2 + 1]);
      }
      for (size_t i = 0; i < 2; ++i) {
        finalResult[i * 2] = mmUnpackloEpi64(tmp[i], tmp[i + 2]);
        finalResult[i * 2 + 1] = mmUnpackhiEpi64(tmp[i], tmp[i + 2]);
      }
    }
    for (size_t i = 0; i < kNumStreams; ++i) {
      mmStoreuSi128(&outputBufferStreams[i][blockIndex], finalResult[i]);
    }
  }
}
#endif // ARROW_HAVE_SSE4_2

#if defined(ARROW_HAVE_AVX2)
template <typename T>
void byteStreamSplitDecodeAvx2(
    const uint8_t* data,
    int64_t numValues,
    int64_t stride,
    T* out) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(
      kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  constexpr size_t kNumStreamsLog2 = (kNumStreams == 8U ? 3U : 2U);
  constexpr int64_t kBlockSize = sizeof(__m256i) * kNumStreams;

  const int64_t size = numValues * sizeof(T);
  if (size < kBlockSize) // Back to SSE for small size
    return byteStreamSplitDecodeSse2(data, numValues, stride, out);
  const int64_t numBlocks = size / kBlockSize;
  uint8_t* outputData = reinterpret_cast<uint8_t*>(out);

  // First handle suffix.
  const int64_t numProcessedElements = (numBlocks * kBlockSize) / kNumStreams;
  for (int64_t i = numProcessedElements; i < numValues; ++i) {
    uint8_t gatheredByteData[kNumStreams];
    for (size_t b = 0; b < kNumStreams; ++b) {
      const size_t byteIndex = b * stride + i;
      gatheredByteData[b] = data[byteIndex];
    }
    out[i] = arrow::util::SafeLoadAs<T>(&gatheredByteData[0]);
  }

  // Processed hierarchically using unpack intrinsics, then permute intrinsics.
  __m256i stage[kNumStreamsLog2 + 1U][kNumStreams];
  __m256i finalResult[kNumStreams];
  constexpr size_t kNumStreamsHalf = kNumStreams / 2U;

  for (int64_t i = 0; i < numBlocks; ++i) {
    for (size_t j = 0; j < kNumStreams; ++j) {
      stage[0][j] = mm256LoaduSi256(
          reinterpret_cast<const __m256i*>(
              &data[i * sizeof(__m256i) + j * stride]));
    }

    for (size_t step = 0; step < kNumStreamsLog2; ++step) {
      for (size_t j = 0; j < kNumStreamsHalf; ++j) {
        stage[step + 1U][j * 2] =
            mm256UnpackloEpi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
        stage[step + 1U][j * 2 + 1U] =
            mm256UnpackhiEpi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
      }
    }

    if constexpr (kNumStreams == 8U) {
      // Path for double, 128i index:
      //   {0X00, 0x08}, {0x01, 0x09}, {0x02, 0x0A}, {0x03, 0x0B},
      //   {0X04, 0x0C}, {0x05, 0x0D}, {0x06, 0x0E}, {0x07, 0x0F},
      finalResult[0] = mm256Permute2x128Si256(
          stage[kNumStreamsLog2][0], stage[kNumStreamsLog2][1], 0b00100000);
      finalResult[1] = mm256Permute2x128Si256(
          stage[kNumStreamsLog2][2], stage[kNumStreamsLog2][3], 0b00100000);
      finalResult[2] = mm256Permute2x128Si256(
          stage[kNumStreamsLog2][4], stage[kNumStreamsLog2][5], 0b00100000);
      finalResult[3] = mm256Permute2x128Si256(
          stage[kNumStreamsLog2][6], stage[kNumStreamsLog2][7], 0b00100000);
      finalResult[4] = mm256Permute2x128Si256(
          stage[kNumStreamsLog2][0], stage[kNumStreamsLog2][1], 0b00110001);
      finalResult[5] = mm256Permute2x128Si256(
          stage[kNumStreamsLog2][2], stage[kNumStreamsLog2][3], 0b00110001);
      finalResult[6] = mm256Permute2x128Si256(
          stage[kNumStreamsLog2][4], stage[kNumStreamsLog2][5], 0b00110001);
      finalResult[7] = mm256Permute2x128Si256(
          stage[kNumStreamsLog2][6], stage[kNumStreamsLog2][7], 0b00110001);
    } else {
      // Path for float, 128i index:
      //   {0x00, 0x04}, {0x01, 0x05}, {0x02, 0x06}, {0x03, 0x07}
      finalResult[0] = mm256Permute2x128Si256(
          stage[kNumStreamsLog2][0], stage[kNumStreamsLog2][1], 0b00100000);
      finalResult[1] = mm256Permute2x128Si256(
          stage[kNumStreamsLog2][2], stage[kNumStreamsLog2][3], 0b00100000);
      finalResult[2] = mm256Permute2x128Si256(
          stage[kNumStreamsLog2][0], stage[kNumStreamsLog2][1], 0b00110001);
      finalResult[3] = mm256Permute2x128Si256(
          stage[kNumStreamsLog2][2], stage[kNumStreamsLog2][3], 0b00110001);
    }

    for (size_t j = 0; j < kNumStreams; ++j) {
      mm256StoreuSi256(
          reinterpret_cast<__m256i*>(
              &outputData[(i * kNumStreams + j) * sizeof(__m256i)]),
          finalResult[j]);
    }
  }
}

template <typename T>
void byteStreamSplitEncodeAvx2(
    const uint8_t* rawValues,
    const size_t numValues,
    uint8_t* outputBufferRaw) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(
      kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  constexpr size_t kBlockSize = sizeof(__m256i) * kNumStreams;

  if constexpr (kNumStreams == 8U) // Back to SSE, currently no path for double.
    return byteStreamSplitEncodeSse2<T>(rawValues, numValues, outputBufferRaw);

  const size_t size = numValues * sizeof(T);
  if (size < kBlockSize) // Back to SSE for small size
    return byteStreamSplitEncodeSse2<T>(rawValues, numValues, outputBufferRaw);
  const size_t numBlocks = size / kBlockSize;
  const __m256i* rawValuesSimd = reinterpret_cast<const __m256i*>(rawValues);
  __m256i* outputBufferStreams[kNumStreams];

  for (size_t i = 0; i < kNumStreams; ++i) {
    outputBufferStreams[i] =
        reinterpret_cast<__m256i*>(&outputBufferRaw[numValues * i]);
  }

  // First handle suffix.
  const size_t numProcessedElements = (numBlocks * kBlockSize) / sizeof(T);
  for (size_t i = numProcessedElements; i < numValues; ++i) {
    for (size_t j = 0U; j < kNumStreams; ++j) {
      const uint8_t byteInValue = rawValues[i * kNumStreams + j];
      outputBufferRaw[j * numValues + i] = byteInValue;
    }
  }

  // Path for float.
  // 1. Processed hierarchically to 32i block using the unpack intrinsics.
  // 2. Pack 128i block using _mm256_permutevar8x32_epi32.
  // 3. Pack final 256i block with _mm256_permute2x128_si256.
  constexpr size_t kNumUnpack = 3U;
  __m256i stage[kNumUnpack + 1][kNumStreams];
  static const __m256i kPermuteMask =
      mm256SetEpi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
  __m256i permute[kNumStreams];
  __m256i finalResult[kNumStreams];

  for (size_t blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
    for (size_t i = 0; i < kNumStreams; ++i) {
      stage[0][i] =
          mm256LoaduSi256(&rawValuesSimd[blockIndex * kNumStreams + i]);
    }

    for (size_t stageLvl = 0; stageLvl < kNumUnpack; ++stageLvl) {
      for (size_t i = 0; i < kNumStreams / 2U; ++i) {
        stage[stageLvl + 1][i * 2] = mm256UnpackloEpi8(
            stage[stageLvl][i * 2], stage[stageLvl][i * 2 + 1]);
        stage[stageLvl + 1][i * 2 + 1] = mm256UnpackhiEpi8(
            stage[stageLvl][i * 2], stage[stageLvl][i * 2 + 1]);
      }
    }

    for (size_t i = 0; i < kNumStreams; ++i) {
      permute[i] = mm256Permutevar8x32Epi32(stage[kNumUnpack][i], kPermuteMask);
    }

    finalResult[0] = mm256Permute2x128Si256(permute[0], permute[2], 0b00100000);
    finalResult[1] = mm256Permute2x128Si256(permute[0], permute[2], 0b00110001);
    finalResult[2] = mm256Permute2x128Si256(permute[1], permute[3], 0b00100000);
    finalResult[3] = mm256Permute2x128Si256(permute[1], permute[3], 0b00110001);

    for (size_t i = 0; i < kNumStreams; ++i) {
      mm256StoreuSi256(&outputBufferStreams[i][blockIndex], finalResult[i]);
    }
  }
}
#endif // ARROW_HAVE_AVX2

#if defined(ARROW_HAVE_AVX512)
template <typename T>
void byteStreamSplitDecodeAvx512(
    const uint8_t* data,
    int64_t numValues,
    int64_t stride,
    T* out) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(
      kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  constexpr size_t kNumStreamsLog2 = (kNumStreams == 8U ? 3U : 2U);
  constexpr int64_t kBlockSize = sizeof(__m512i) * kNumStreams;

  const int64_t size = numValues * sizeof(T);
  if (size < kBlockSize) // Back to AVX2 for small size
    return byteStreamSplitDecodeAvx2(data, numValues, stride, out);
  const int64_t numBlocks = size / kBlockSize;
  uint8_t* outputData = reinterpret_cast<uint8_t*>(out);

  // First handle suffix.
  const int64_t numProcessedElements = (numBlocks * kBlockSize) / kNumStreams;
  for (int64_t i = numProcessedElements; i < numValues; ++i) {
    uint8_t gatheredByteData[kNumStreams];
    for (size_t b = 0; b < kNumStreams; ++b) {
      const size_t byteIndex = b * stride + i;
      gatheredByteData[b] = data[byteIndex];
    }
    out[i] = arrow::util::SafeLoadAs<T>(&gatheredByteData[0]);
  }

  // Processed hierarchically using the unpack, then two shuffles.
  __m512i stage[kNumStreamsLog2 + 1U][kNumStreams];
  __m512i shuffle[kNumStreams];
  __m512i finalResult[kNumStreams];
  constexpr size_t kNumStreamsHalf = kNumStreams / 2U;

  for (int64_t i = 0; i < numBlocks; ++i) {
    for (size_t j = 0; j < kNumStreams; ++j) {
      stage[0][j] = mm512LoaduSi512(
          reinterpret_cast<const __m512i*>(
              &data[i * sizeof(__m512i) + j * stride]));
    }

    for (size_t step = 0; step < kNumStreamsLog2; ++step) {
      for (size_t j = 0; j < kNumStreamsHalf; ++j) {
        stage[step + 1U][j * 2] =
            mm512UnpackloEpi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
        stage[step + 1U][j * 2 + 1U] =
            mm512UnpackhiEpi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
      }
    }

    if constexpr (kNumStreams == 8U) {
      // Path for double, 128i index:
      // {0X00, 0x04, 0x08, 0x0C}, {0x10, 0x14, 0x18, 0x1C},
      // {0X01, 0x05, 0x09, 0x0D}, {0x11, 0x15, 0x19, 0x1D},
      // {0X02, 0x06, 0x0A, 0x0E}, {0x12, 0x16, 0x1A, 0x1E},
      // {0X03, 0x07, 0x0B, 0x0F}, {0x13, 0x17, 0x1B, 0x1F},
      shuffle[0] = mm512ShuffleI32x4(
          stage[kNumStreamsLog2][0], stage[kNumStreamsLog2][1], 0b01000100);
      shuffle[1] = mm512ShuffleI32x4(
          stage[kNumStreamsLog2][2], stage[kNumStreamsLog2][3], 0b01000100);
      shuffle[2] = mm512ShuffleI32x4(
          stage[kNumStreamsLog2][4], stage[kNumStreamsLog2][5], 0b01000100);
      shuffle[3] = mm512ShuffleI32x4(
          stage[kNumStreamsLog2][6], stage[kNumStreamsLog2][7], 0b01000100);
      shuffle[4] = mm512ShuffleI32x4(
          stage[kNumStreamsLog2][0], stage[kNumStreamsLog2][1], 0b11101110);
      shuffle[5] = mm512ShuffleI32x4(
          stage[kNumStreamsLog2][2], stage[kNumStreamsLog2][3], 0b11101110);
      shuffle[6] = mm512ShuffleI32x4(
          stage[kNumStreamsLog2][4], stage[kNumStreamsLog2][5], 0b11101110);
      shuffle[7] = mm512ShuffleI32x4(
          stage[kNumStreamsLog2][6], stage[kNumStreamsLog2][7], 0b11101110);

      finalResult[0] = mm512ShuffleI32x4(shuffle[0], shuffle[1], 0b10001000);
      finalResult[1] = mm512ShuffleI32x4(shuffle[2], shuffle[3], 0b10001000);
      finalResult[2] = mm512ShuffleI32x4(shuffle[0], shuffle[1], 0b11011101);
      finalResult[3] = mm512ShuffleI32x4(shuffle[2], shuffle[3], 0b11011101);
      finalResult[4] = mm512ShuffleI32x4(shuffle[4], shuffle[5], 0b10001000);
      finalResult[5] = mm512ShuffleI32x4(shuffle[6], shuffle[7], 0b10001000);
      finalResult[6] = mm512ShuffleI32x4(shuffle[4], shuffle[5], 0b11011101);
      finalResult[7] = mm512ShuffleI32x4(shuffle[6], shuffle[7], 0b11011101);
    } else {
      // Path for float, 128i index:
      // {0x00, 0x04, 0x08, 0x0C}, {0x01, 0x05, 0x09, 0x0D}
      // {0X02, 0x06, 0x0A, 0x0E}, {0x03, 0x07, 0x0B, 0x0F},
      shuffle[0] = mm512ShuffleI32x4(
          stage[kNumStreamsLog2][0], stage[kNumStreamsLog2][1], 0b01000100);
      shuffle[1] = mm512ShuffleI32x4(
          stage[kNumStreamsLog2][2], stage[kNumStreamsLog2][3], 0b01000100);
      shuffle[2] = mm512ShuffleI32x4(
          stage[kNumStreamsLog2][0], stage[kNumStreamsLog2][1], 0b11101110);
      shuffle[3] = mm512ShuffleI32x4(
          stage[kNumStreamsLog2][2], stage[kNumStreamsLog2][3], 0b11101110);

      finalResult[0] = mm512ShuffleI32x4(shuffle[0], shuffle[1], 0b10001000);
      finalResult[1] = mm512ShuffleI32x4(shuffle[0], shuffle[1], 0b11011101);
      finalResult[2] = mm512ShuffleI32x4(shuffle[2], shuffle[3], 0b10001000);
      finalResult[3] = mm512ShuffleI32x4(shuffle[2], shuffle[3], 0b11011101);
    }

    for (size_t j = 0; j < kNumStreams; ++j) {
      mm512StoreuSi512(
          reinterpret_cast<__m512i*>(
              &outputData[(i * kNumStreams + j) * sizeof(__m512i)]),
          finalResult[j]);
    }
  }
}

template <typename T>
void byteStreamSplitEncodeAvx512(
    const uint8_t* rawValues,
    const size_t numValues,
    uint8_t* outputBufferRaw) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(
      kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  constexpr size_t kBlockSize = sizeof(__m512i) * kNumStreams;

  const size_t size = numValues * sizeof(T);

  if (size < kBlockSize) // Back to AVX2 for small size
    return byteStreamSplitEncodeAvx2<T>(rawValues, numValues, outputBufferRaw);

  const size_t numBlocks = size / kBlockSize;
  const __m512i* rawValuesSimd = reinterpret_cast<const __m512i*>(rawValues);
  __m512i* outputBufferStreams[kNumStreams];
  for (size_t i = 0; i < kNumStreams; ++i) {
    outputBufferStreams[i] =
        reinterpret_cast<__m512i*>(&outputBufferRaw[numValues * i]);
  }

  // First handle suffix.
  const size_t numProcessedElements = (numBlocks * kBlockSize) / sizeof(T);
  for (size_t i = numProcessedElements; i < numValues; ++i) {
    for (size_t j = 0U; j < kNumStreams; ++j) {
      const uint8_t byteInValue = rawValues[i * kNumStreams + j];
      outputBufferRaw[j * numValues + i] = byteInValue;
    }
  }

  constexpr size_t KNumUnpack = (kNumStreams == 8U) ? 2U : 3U;
  __m512i finalResult[kNumStreams];
  __m512i unpack[KNumUnpack + 1][kNumStreams];
  __m512i permutex[kNumStreams];
  __m512i permutexMask;
  if constexpr (kNumStreams == 8U) {
    // Use _mm512_set_epi32, no _mm512_set_epi16 for some old gcc version.
    permutexMask = mm512SetEpi32(
        0x001F0017,
        0x000F0007,
        0x001E0016,
        0x000E0006,
        0x001D0015,
        0x000D0005,
        0x001C0014,
        0x000C0004,
        0x001B0013,
        0x000B0003,
        0x001A0012,
        0x000A0002,
        0x00190011,
        0x00090001,
        0x00180010,
        0x00080000);
  } else {
    permutexMask = mm512SetEpi32(
        0x0F,
        0x0B,
        0x07,
        0x03,
        0x0E,
        0x0A,
        0x06,
        0x02,
        0x0D,
        0x09,
        0x05,
        0x01,
        0x0C,
        0x08,
        0x04,
        0x00);
  }

  for (size_t blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
    for (size_t i = 0; i < kNumStreams; ++i) {
      unpack[0][i] =
          mm512LoaduSi512(&rawValuesSimd[blockIndex * kNumStreams + i]);
    }

    for (size_t unpackLvl = 0; unpackLvl < KNumUnpack; ++unpackLvl) {
      for (size_t i = 0; i < kNumStreams / 2U; ++i) {
        unpack[unpackLvl + 1][i * 2] = mm512UnpackloEpi8(
            unpack[unpackLvl][i * 2], unpack[unpackLvl][i * 2 + 1]);
        unpack[unpackLvl + 1][i * 2 + 1] = mm512UnpackhiEpi8(
            unpack[unpackLvl][i * 2], unpack[unpackLvl][i * 2 + 1]);
      }
    }

    if constexpr (kNumStreams == 8U) {
      // Path for double.
      // 1. Unpack to epi16 block.
      // 2. Permutexvar_epi16 to 128i block.
      // 3. Shuffle 128i to final 512i target, index:
      //   {0X00, 0x04, 0x08, 0x0C}, {0x10, 0x14, 0x18, 0x1C},
      //   {0X01, 0x05, 0x09, 0x0D}, {0x11, 0x15, 0x19, 0x1D},
      //   {0X02, 0x06, 0x0A, 0x0E}, {0x12, 0x16, 0x1A, 0x1E},
      //   {0X03, 0x07, 0x0B, 0x0F}, {0x13, 0x17, 0x1B, 0x1F},
      for (size_t i = 0; i < kNumStreams; ++i)
        permutex[i] =
            mm512PermutexvarEpi16(permutexMask, unpack[KNumUnpack][i]);

      __m512i shuffle[kNumStreams];
      shuffle[0] = mm512ShuffleI32x4(permutex[0], permutex[2], 0b01000100);
      shuffle[1] = mm512ShuffleI32x4(permutex[4], permutex[6], 0b01000100);
      shuffle[2] = mm512ShuffleI32x4(permutex[0], permutex[2], 0b11101110);
      shuffle[3] = mm512ShuffleI32x4(permutex[4], permutex[6], 0b11101110);
      shuffle[4] = mm512ShuffleI32x4(permutex[1], permutex[3], 0b01000100);
      shuffle[5] = mm512ShuffleI32x4(permutex[5], permutex[7], 0b01000100);
      shuffle[6] = mm512ShuffleI32x4(permutex[1], permutex[3], 0b11101110);
      shuffle[7] = mm512ShuffleI32x4(permutex[5], permutex[7], 0b11101110);

      finalResult[0] = mm512ShuffleI32x4(shuffle[0], shuffle[1], 0b10001000);
      finalResult[1] = mm512ShuffleI32x4(shuffle[0], shuffle[1], 0b11011101);
      finalResult[2] = mm512ShuffleI32x4(shuffle[2], shuffle[3], 0b10001000);
      finalResult[3] = mm512ShuffleI32x4(shuffle[2], shuffle[3], 0b11011101);
      finalResult[4] = mm512ShuffleI32x4(shuffle[4], shuffle[5], 0b10001000);
      finalResult[5] = mm512ShuffleI32x4(shuffle[4], shuffle[5], 0b11011101);
      finalResult[6] = mm512ShuffleI32x4(shuffle[6], shuffle[7], 0b10001000);
      finalResult[7] = mm512ShuffleI32x4(shuffle[6], shuffle[7], 0b11011101);
    } else {
      // Path for float.
      // 1. Processed hierarchically to 32i block using the unpack intrinsics.
      // 2. Pack 128i block using _mm256_permutevar8x32_epi32.
      // 3. Pack final 256i block with _mm256_permute2x128_si256.
      for (size_t i = 0; i < kNumStreams; ++i)
        permutex[i] =
            mm512PermutexvarEpi32(permutexMask, unpack[KNumUnpack][i]);

      finalResult[0] = mm512ShuffleI32x4(permutex[0], permutex[2], 0b01000100);
      finalResult[1] = mm512ShuffleI32x4(permutex[0], permutex[2], 0b11101110);
      finalResult[2] = mm512ShuffleI32x4(permutex[1], permutex[3], 0b01000100);
      finalResult[3] = mm512ShuffleI32x4(permutex[1], permutex[3], 0b11101110);
    }

    for (size_t i = 0; i < kNumStreams; ++i) {
      mm512StoreuSi512(&outputBufferStreams[i][blockIndex], finalResult[i]);
    }
  }
}
#endif // ARROW_HAVE_AVX512

#if defined(ARROW_HAVE_SIMD_SPLIT)
template <typename T>
void inline byteStreamSplitDecodeSimd(
    const uint8_t* data,
    int64_t numValues,
    int64_t stride,
    T* out) {
#if defined(ARROW_HAVE_AVX512)
  return byteStreamSplitDecodeAvx512(data, numValues, stride, out);
#elif defined(ARROW_HAVE_AVX2)
  return byteStreamSplitDecodeAvx2(data, numValues, stride, out);
#elif defined(ARROW_HAVE_SSE4_2)
  return byteStreamSplitDecodeSse2(data, numValues, stride, out);
#else
#error "ByteStreamSplitDecodeSimd not implemented"
#endif
}

template <typename T>
void inline byteStreamSplitEncodeSimd(
    const uint8_t* rawValues,
    const int64_t numValues,
    uint8_t* outputBufferRaw) {
#if defined(ARROW_HAVE_AVX512)
  return byteStreamSplitEncodeAvx512<T>(
      rawValues, static_cast<size_t>(numValues), outputBufferRaw);
#elif defined(ARROW_HAVE_AVX2)
  return byteStreamSplitEncodeAvx2<T>(
      rawValues, static_cast<size_t>(numValues), outputBufferRaw);
#elif defined(ARROW_HAVE_SSE4_2)
  return byteStreamSplitEncodeSse2<T>(
      rawValues, static_cast<size_t>(numValues), outputBufferRaw);
#else
#error "ByteStreamSplitEncodeSimd not implemented"
#endif
}
#endif

//
// Scalar implementations.
//

inline void doSplitStreams(
    const uint8_t* src,
    int width,
    int64_t nvalues,
    uint8_t** destStreams) {
  // Value empirically chosen to provide the best performance on the author's
  // machine.
  constexpr int kBlockSize = 32;

  while (nvalues >= kBlockSize) {
    for (int stream = 0; stream < width; ++stream) {
      uint8_t* dest = destStreams[stream];
      for (int i = 0; i < kBlockSize; i += 8) {
        uint64_t a = src[stream + i * width];
        uint64_t b = src[stream + (i + 1) * width];
        uint64_t c = src[stream + (i + 2) * width];
        uint64_t d = src[stream + (i + 3) * width];
        uint64_t e = src[stream + (i + 4) * width];
        uint64_t f = src[stream + (i + 5) * width];
        uint64_t g = src[stream + (i + 6) * width];
        uint64_t h = src[stream + (i + 7) * width];
#if ARROW_LITTLE_ENDIAN
        uint64_t r = a | (b << 8) | (c << 16) | (d << 24) | (e << 32) |
            (f << 40) | (g << 48) | (h << 56);
#else
        uint64_t r = (a << 56) | (b << 48) | (c << 40) | (d << 32) | (e << 24) |
            (f << 16) | (g << 8) | h;
#endif
        ::arrow::util::SafeStore(&dest[i], r);
      }
      destStreams[stream] += kBlockSize;
    }
    src += width * kBlockSize;
    nvalues -= kBlockSize;
  }

  // Epilog.
  for (int stream = 0; stream < width; ++stream) {
    uint8_t* dest = destStreams[stream];
    for (int64_t i = 0; i < nvalues; ++i) {
      dest[i] = src[stream + i * width];
    }
  }
}

inline void doMergeStreams(
    const uint8_t** srcStreams,
    int width,
    int64_t nvalues,
    uint8_t* dest) {
  // Value empirically chosen to provide the best performance on the author's
  // machine.
  constexpr int kBlockSize = 128;

  while (nvalues >= kBlockSize) {
    for (int stream = 0; stream < width; ++stream) {
      // Take kBlockSize bytes from the given stream and spread them.
      // To their logical places in destination.
      const uint8_t* src = srcStreams[stream];
      for (int i = 0; i < kBlockSize; i += 8) {
        uint64_t v = ::arrow::util::SafeLoadAs<uint64_t>(&src[i]);
#if ARROW_LITTLE_ENDIAN
        dest[stream + i * width] = static_cast<uint8_t>(v);
        dest[stream + (i + 1) * width] = static_cast<uint8_t>(v >> 8);
        dest[stream + (i + 2) * width] = static_cast<uint8_t>(v >> 16);
        dest[stream + (i + 3) * width] = static_cast<uint8_t>(v >> 24);
        dest[stream + (i + 4) * width] = static_cast<uint8_t>(v >> 32);
        dest[stream + (i + 5) * width] = static_cast<uint8_t>(v >> 40);
        dest[stream + (i + 6) * width] = static_cast<uint8_t>(v >> 48);
        dest[stream + (i + 7) * width] = static_cast<uint8_t>(v >> 56);
#else
        dest[stream + i * width] = static_cast<uint8_t>(v >> 56);
        dest[stream + (i + 1) * width] = static_cast<uint8_t>(v >> 48);
        dest[stream + (i + 2) * width] = static_cast<uint8_t>(v >> 40);
        dest[stream + (i + 3) * width] = static_cast<uint8_t>(v >> 32);
        dest[stream + (i + 4) * width] = static_cast<uint8_t>(v >> 24);
        dest[stream + (i + 5) * width] = static_cast<uint8_t>(v >> 16);
        dest[stream + (i + 6) * width] = static_cast<uint8_t>(v >> 8);
        dest[stream + (i + 7) * width] = static_cast<uint8_t>(v);
#endif
      }
      srcStreams[stream] += kBlockSize;
    }
    dest += width * kBlockSize;
    nvalues -= kBlockSize;
  }

  // Epilog.
  for (int stream = 0; stream < width; ++stream) {
    const uint8_t* src = srcStreams[stream];
    for (int64_t i = 0; i < nvalues; ++i) {
      dest[stream + i * width] = src[i];
    }
  }
}

template <typename T>
void byteStreamSplitEncodeScalar(
    const uint8_t* rawValues,
    const int64_t numValues,
    uint8_t* outputBufferRaw) {
  constexpr int kNumStreams = static_cast<int>(sizeof(T));
  std::array<uint8_t*, kNumStreams> destStreams;
  for (int stream = 0; stream < kNumStreams; ++stream) {
    destStreams[stream] = &outputBufferRaw[stream * numValues];
  }
  doSplitStreams(rawValues, kNumStreams, numValues, destStreams.data());
}

template <typename T>
void byteStreamSplitDecodeScalar(
    const uint8_t* data,
    int64_t numValues,
    int64_t stride,
    T* out) {
  constexpr int kNumStreams = static_cast<int>(sizeof(T));
  std::array<const uint8_t*, kNumStreams> srcStreams;
  for (int stream = 0; stream < kNumStreams; ++stream) {
    srcStreams[stream] = &data[stream * stride];
  }
  doMergeStreams(
      srcStreams.data(),
      kNumStreams,
      numValues,
      reinterpret_cast<uint8_t*>(out));
}

template <typename T>
void inline byteStreamSplitEncode(
    const uint8_t* rawValues,
    const int64_t numValues,
    uint8_t* outputBufferRaw) {
#if defined(ARROW_HAVE_SIMD_SPLIT)
  return byteStreamSplitEncodeSimd<T>(rawValues, numValues, outputBufferRaw);
#else
  return byteStreamSplitEncodeScalar<T>(rawValues, numValues, outputBufferRaw);
#endif
}

template <typename T>
void inline byteStreamSplitDecode(
    const uint8_t* data,
    int64_t numValues,
    int64_t stride,
    T* out) {
#if defined(ARROW_HAVE_SIMD_SPLIT)
  return byteStreamSplitDecodeSimd(data, numValues, stride, out);
#else
  return byteStreamSplitDecodeScalar(data, numValues, stride, out);
#endif
}

} // namespace facebook::velox::parquet::arrow
