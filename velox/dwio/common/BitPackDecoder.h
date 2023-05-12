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

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/dwio/common/BitPackDecoderUtil.h"
#ifdef VELOX_ENABLE_AVX512
#include "velox/dwio/common/BitPackDecoderUtilAVX512.h"
#endif

namespace facebook::velox::dwio::common {

using RowSet = folly::Range<const facebook::velox::vector_size_t*>;

/// Copies bit fields starting at 'bitOffset'th bit of 'bits' into
/// 'result'.  The indices of the fields are in 'rows' and their
/// bit-width is 'bitWidth'.  'rowBias' is subtracted from each
/// index in 'rows' before calculating the bit field's position. The
/// bit fields are considered little endian. 'bufferEnd' is the address of the
/// first undefined byte after the buffer containing the bits. If non-null,
/// extra-wide memory accesses will not be used at thee end of the range to
/// stay under 'bufferEnd'.
template <typename T>
void unpack(
    const uint64_t* FOLLY_NULLABLE bits,
    int32_t bitOffset,
    RowSet rows,
    int32_t rowBias,
    uint8_t bitWidth,
    const char* FOLLY_NULLABLE bufferEnd,
    T* FOLLY_NONNULL result);

#ifdef VELOX_ENABLE_AVX512
/// Unpack numValues number of input values from inputBuffer. The results
/// will be written to result. The
/// caller needs to make sure the inputBufferLen contains at least numValues
/// number of packed values. The inputBits and result pointers will be updated
/// to the next to read position after this call.
template <typename T>
inline void unpackAVX512(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    T* FOLLY_NONNULL& result);

template <>
inline void unpackAVX512<uint8_t>(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    uint8_t* FOLLY_NONNULL& result) {
  switch (bitWidth) {
    case 1:
      unpack1(inputBits, numValues, result);
      break;
    case 2:
      unpack2(inputBits, numValues, result);
      break;
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
      unpack3to7u(bitWidth, inputBits, numValues, result);
      break;
    case 8:
      unpack8(inputBits, numValues, result);
      break;
    default:
      VELOX_UNREACHABLE("invalid bitWidth");
  }
}

template <>
inline void unpackAVX512<uint16_t>(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    uint16_t* FOLLY_NONNULL& result) {
  switch (bitWidth) {
    case 1:
      unpack1(inputBits, numValues, result);
      break;
    case 2:
      unpack2(inputBits, numValues, result);
      break;
    case 3:
      unpack3(inputBits, numValues, result);
      break;
    case 4:
      unpack4(inputBits, numValues, result);
      break;
    case 5:
    case 6:
    case 7:
      unpack5to7u(bitWidth, inputBits, numValues, result);
      break;
    case 8:
      unpack8(inputBits, numValues, result);
      break;
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
      unpack9to15u(bitWidth, inputBits, numValues, result);
      break;
    case 16:
      unpack16u(inputBits, numValues, result);
      break;
    default:
      VELOX_UNREACHABLE("invalid bitWidth");
  }
}

template <>
inline void unpackAVX512<uint32_t>(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    uint32_t* FOLLY_NONNULL& result) {
  switch (bitWidth) {
    case 1:
      unpack1(inputBits, numValues, result);
      break;
    case 2:
      unpack2(inputBits, numValues, result);
      break;
    case 3:
      unpack3(inputBits, numValues, result);
      break;
    case 4:
      unpack4(inputBits, numValues, result);
      break;
    case 5:
    case 6:
    case 7:
      unpack5to7u(bitWidth, inputBits, numValues, result);
      break;
    case 8:
      unpack8u(inputBits, numValues, result);
      break;
    case 9:
    case 10:
    case 11:
    case 12:
    case 14:
      unpack9to15u(bitWidth, inputBits, numValues, result);
      break;
    case 13:
      unpack13(inputBits, numValues, result);
      break;
    case 15:
      unpack15(inputBits, numValues, result);
      break;
    case 16:
      unpack16u(inputBits, numValues, result);
      break;
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
    case 24:
    case 25:
    case 26:
    case 27:
    case 28:
    case 29:
    case 30:
    case 31:
      unpack17to31u(bitWidth, inputBits, numValues, result);
      break;
    case 32:
      unpack32u(inputBits, numValues, result);
      break;
    default:
      VELOX_UNREACHABLE("invalid bitWidth");
  }
}
#endif

/// Unpack numValues number of input values from inputBuffer. The results
/// will be written to result. The
/// caller needs to make sure the inputBufferLen contains at least numValues
/// number of packed values. The inputBits and result pointers will be updated
/// to the next to read position after this call.
template <typename T>
inline void unpackAVX2(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    T* FOLLY_NONNULL& result);

template <>
inline void unpackAVX2<uint8_t>(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    uint8_t* FOLLY_NONNULL& result) {
  uint64_t mask = kPdepMask8[bitWidth];
  auto writeEndOffset = result + numValues;

  // Process bitWidth bytes (8 values) a time. Note that for bitWidth 8, the
  // performance of direct memcpy is about the same as this solution.
  while (result + 8 <= writeEndOffset) {
    // Using memcpy() here may result in non-optimized loops by clong.
    uint64_t val = *reinterpret_cast<const uint64_t*>(inputBits);
    *(reinterpret_cast<uint64_t*>(result)) = _pdep_u64(val, mask);
    inputBits += bitWidth;
    result += 8;
  }

  numValues = writeEndOffset - result;
  unpackNaive(
      inputBits, (bitWidth * numValues + 7) / 8, numValues, bitWidth, result);
}

template <>
inline void unpackAVX2<uint16_t>(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    uint16_t* FOLLY_NONNULL& result) {
  switch (bitWidth) {
    case 1:
    case 2:
    case 3:
    case 4:
      unpack1to4(bitWidth, inputBits, numValues, result);
      break;
    case 5:
    case 6:
    case 7:
      unpack5to8(bitWidth, inputBits, numValues, result);
      break;
    case 8:
      unpack8_cast(inputBits, numValues, result);
      break;
    case 9:
    case 11:
    case 13:
    case 15:
    case 10:
    case 12:
    case 14:
      unpack9to15(bitWidth, inputBits, numValues, result);
      break;
    case 16:
      unpack16(inputBits, numValues, result);
      break;
    default:
      VELOX_UNREACHABLE("invalid bitWidth");
  }
}

template <>
inline void unpackAVX2<uint32_t>(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    uint32_t* FOLLY_NONNULL& result) {
  switch (bitWidth) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
      unpack1to7(bitWidth, inputBits, numValues, result);
      break;
    case 8:
      unpack8(inputBits, numValues, result);
      break;
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
      unpack9to15(bitWidth, inputBits, numValues, result);
      break;
    case 16:
      unpack16(inputBits, numValues, result);
      break;
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
      unpack17to21(bitWidth, inputBits, numValues, result);
      break;
    case 22:
    case 23:
    case 24:
    case 25:
    case 26:
    case 27:
    case 28:
    case 29:
    case 30:
    case 31:
      unpack22to31(bitWidth, inputBits, numValues, result);
      break;
    case 32:
      unpack32(inputBits, numValues, result);
      break;
    default:
      VELOX_UNREACHABLE("invalid bitWidth");
  }
}

/// Unpack numValues number of input values from inputBuffer. The results
/// will be written to result. numValues must be a multiple of 8. The
/// caller needs to make sure the inputBufferLen contains at least numValues
/// number of packed values. The inputBits and result pointers will be updated
/// to the next to read position after this call.
template <typename T>
inline void unpack(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    T* FOLLY_NONNULL& result) {
  unpackNaive<T>(inputBits, inputBufferLen, numValues, bitWidth, result);
}

template <>
inline void unpack<uint8_t>(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    uint8_t* FOLLY_NONNULL& result);

template <>
inline void unpack<uint16_t>(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    uint16_t* FOLLY_NONNULL& result);

template <>
inline void unpack<uint32_t>(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    uint32_t* FOLLY_NONNULL& result);

template <>
inline void unpack<uint8_t>(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    uint8_t* FOLLY_NONNULL& result) {
  VELOX_CHECK(bitWidth >= 1 && bitWidth <= 8);
  VELOX_CHECK(inputBufferLen * 8 >= bitWidth * numValues);

#ifdef VELOX_ENABLE_AVX512

  unpackAVX512<uint8_t>(inputBits, inputBufferLen, numValues, bitWidth, result);

#elif XSIMD_WITH_AVX2

  unpackAVX2<uint8_t>(inputBits, inputBufferLen, numValues, bitWidth, result);

#else

  unpackNaive<uint8_t>(inputBits, inputBufferLen, numValues, bitWidth, result);

#endif
}

template <>
inline void unpack<uint16_t>(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    uint16_t* FOLLY_NONNULL& result) {
  VELOX_CHECK(bitWidth >= 1 && bitWidth <= 16);
  VELOX_CHECK(inputBufferLen * 8 >= bitWidth * numValues);

#ifdef VELOX_ENABLE_AVX512

  unpackAVX512<uint16_t>(
      inputBits, inputBufferLen, numValues, bitWidth, result);

#elif XSIMD_WITH_AVX2

  unpackAVX2<uint16_t>(inputBits, inputBufferLen, numValues, bitWidth, result);

#else

  unpackNaive<uint16_t>(inputBits, inputBufferLen, numValues, bitWidth, result);

#endif
}

template <>
inline void unpack<uint32_t>(
    const uint8_t* FOLLY_NONNULL& inputBits,
    uint64_t inputBufferLen,
    uint64_t numValues,
    uint8_t bitWidth,
    uint32_t* FOLLY_NONNULL& result) {
  VELOX_CHECK(bitWidth >= 1 && bitWidth <= 32);
  VELOX_CHECK(inputBufferLen * 8 >= bitWidth * numValues);

#ifdef VELOX_ENABLE_AVX512

  unpackAVX512<uint32_t>(
      inputBits, inputBufferLen, numValues, bitWidth, result);

#elif XSIMD_WITH_AVX2

  unpackAVX2<uint32_t>(inputBits, inputBufferLen, numValues, bitWidth, result);

#else

  unpackNaive<uint32_t>(inputBits, inputBufferLen, numValues, bitWidth, result);

#endif
}

// Loads a bit field from 'ptr' + bitOffset for up to 'bitWidth' bits. makes
// sure not to access bytes past lastSafeWord + 7. The definition is put here
// because it's inlined.
inline uint64_t safeLoadBits(
    const char* FOLLY_NONNULL ptr,
    int32_t bitOffset,
    uint8_t bitWidth,
    const char* FOLLY_NONNULL lastSafeWord) {
  VELOX_DCHECK_GE(7, bitOffset);
  VELOX_DCHECK_GE(56, bitWidth);
  if (ptr < lastSafeWord) {
    return *reinterpret_cast<const uint64_t*>(ptr) >> bitOffset;
  }
  int32_t byteWidth =
      facebook::velox::bits::roundUp(bitOffset + bitWidth, 8) / 8;
  return facebook::velox::bits::loadPartialWord(
             reinterpret_cast<const uint8_t*>(ptr), byteWidth) >>
      bitOffset;
}

} // namespace facebook::velox::dwio::common
