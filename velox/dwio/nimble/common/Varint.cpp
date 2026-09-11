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
#ifdef __x86_64__
#include <immintrin.h>
#define VARINT_SVE2 0
#elif defined(__aarch64__)
#include <arm_neon.h>
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE2_BITPERM) && \
    __has_include(<arm_neon_sve_bridge.h>)
#include <arm_neon_sve_bridge.h>
#include <arm_sve.h>
#define VARINT_SVE2 1
#else
#define VARINT_SVE2 0
#endif
#endif //__x86_64__

#include <array>
#include <cstring>

#include <xsimd/xsimd.hpp>

#include "folly/Likely.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Varint.h"

namespace facebook::nimble::varint {

const char* bulkVarintSkip(uint64_t n, const char* pos) {
  const uint64_t* word = reinterpret_cast<const uint64_t*>(pos);
  while (n >= 8) {
    // Zeros in the 8 * ith bits indicate termination of a varint.
    n -= __builtin_popcountll(~(*word++) & 0x8080808080808080ULL);
  }
  pos = reinterpret_cast<const char*>(word);
  while (n--) {
    skipVarint(&pos);
  }
  return pos;
}

uint64_t bulkVarintSize32(std::span<const uint32_t> values) {
  constexpr uint8_t kLookupSizeTable32[32] = {5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4,
                                              3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
                                              2, 2, 2, 1, 1, 1, 1, 1, 1, 1};
  uint64_t size = 0;
  for (uint32_t value : values) {
    size += kLookupSizeTable32[__builtin_clz(value | 1U)];
  }
  return size;
}

uint64_t bulkVarintSize64(std::span<const uint64_t> values) {
  constexpr uint8_t kLookupSizeTable64[64] = {
      10, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7,
      6,  6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 3,
      3,  3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1};
  uint64_t size = 0;
  for (uint64_t value : values) {
    size += kLookupSizeTable64[__builtin_clzll(value | 1ULL)];
  }
  return size;
}

#if !VARINT_SVE2

// Declaration of the function we build via generated code below.
template <typename T>
#ifdef __x86_64__
__attribute__((__target__("bmi2")))
#endif
// __attribute__ ((optimize("Os")))
const char* bulkVarintDecodeBmi2(uint64_t n, const char* pos, T* output);

#endif // !VARINT_SVE2

// Zero-extend 8 consecutive bytes into T-sized output elements using xsimd
// batch construction and store.
template <typename T>
inline void expandByteWord(const uint8_t* bytes, T* output) {
  using batch_type = xsimd::batch<T>;
  constexpr auto kBatchSize = batch_type::size;

  if constexpr (kBatchSize >= 8) {
    batch_type(
        static_cast<T>(bytes[0]),
        static_cast<T>(bytes[1]),
        static_cast<T>(bytes[2]),
        static_cast<T>(bytes[3]),
        static_cast<T>(bytes[4]),
        static_cast<T>(bytes[5]),
        static_cast<T>(bytes[6]),
        static_cast<T>(bytes[7]))
        .store_unaligned(output);
  } else if constexpr (kBatchSize == 4) {
    batch_type(
        static_cast<T>(bytes[0]),
        static_cast<T>(bytes[1]),
        static_cast<T>(bytes[2]),
        static_cast<T>(bytes[3]))
        .store_unaligned(output);
    batch_type(
        static_cast<T>(bytes[4]),
        static_cast<T>(bytes[5]),
        static_cast<T>(bytes[6]),
        static_cast<T>(bytes[7]))
        .store_unaligned(output + 4);
  } else if constexpr (kBatchSize == 2) {
    batch_type(static_cast<T>(bytes[0]), static_cast<T>(bytes[1]))
        .store_unaligned(output);
    batch_type(static_cast<T>(bytes[2]), static_cast<T>(bytes[3]))
        .store_unaligned(output + 2);
    batch_type(static_cast<T>(bytes[4]), static_cast<T>(bytes[5]))
        .store_unaligned(output + 4);
    batch_type(static_cast<T>(bytes[6]), static_cast<T>(bytes[7]))
        .store_unaligned(output + 6);
  }
}

// Process runs of single-byte varints using xsimd for both the high-bit
// check and byte-to-element widening. Works with uint8_t* throughout,
// avoiding reinterpret_cast to uint64_t* (alignment/strict-aliasing issues).
// Returns the number of elements remaining after processing.
template <typename T>
inline uint64_t
bulkDecodeSingleByteRun(uint64_t n, const char*& pos, T*& output) {
  using u8_batch = xsimd::batch<uint8_t>;
  constexpr auto kU8Size = u8_batch::size;
  constexpr uint64_t wordSize = 8;
  constexpr uint64_t kHighBits = 0x8080808080808080ULL;

  const auto* src = reinterpret_cast<const uint8_t*>(pos);
  uint64_t remaining = n;

  // Process kU8BatchSize bytes at a time.
  // Single wide load + vptest
  while (remaining >= kU8Size) {
    auto bytes = u8_batch::load_unaligned(src);
    if (xsimd::any((bytes & u8_batch(0x80)) != u8_batch(0))) {
      break;
    }
    for (size_t i = 0; i < kU8Size; i += wordSize) {
      expandByteWord(src + i, output + i);
    }
    src += kU8Size;
    output += kU8Size;
    remaining -= kU8Size;
  }

  // Process 8 bytes at a time. Use memcpy for the high-bit check to avoid
  // reinterpret_cast<const uint64_t*> strict-aliasing/alignment issues.
  while (remaining >= wordSize) {
    uint64_t word;
    std::memcpy(&word, src, sizeof(word));

    if (word & kHighBits) {
      break;
    }
    expandByteWord(src, output);
    src += wordSize;
    output += wordSize;
    remaining -= wordSize;
  }

  // Handle trailing single-byte varints one at a time.
  while (remaining > 0 && !(src[0] & 0x80)) {
    *output++ = static_cast<T>(src[0]);
    ++src;
    --remaining;
  }

  pos = reinterpret_cast<const char*>(src);
  return remaining;
}

constexpr std::size_t kCacheLineBytes = 64;
constexpr std::size_t kMaxControlBitsValue = 64;
constexpr std::size_t kMaskLength = 6;

// Process runs of two-byte varints. Each 2-byte varint has continuation bit
// set on byte 0 and clear on byte 1. We detect this pattern in 8-byte words
// and decode 4 varints at a time using simple scalar ops.
// Returns the number of elements remaining after processing.
template <typename T>
inline uint64_t bulkDecodeTwoByteRun(uint64_t n, const char*& pos, T*& output) {
  auto remaining = n;
  const auto* src = reinterpret_cast<const uint8_t*>(pos);

  // In a run of 2-byte varints, each 8-byte word has alternating high bits:
  // bytes 0,2,4,6 have 0x80 set (continuation), bytes 1,3,5,7 have 0x80
  // clear (terminator). This gives the pattern 0x0080008000800080.
  constexpr uint64_t kHighBits = 0x8080808080808080ULL;
  constexpr uint64_t kTwoBytePattern = 0x0080008000800080ULL;
  constexpr uint64_t kWordSize = 8;

  // Process 8 bytes at a time (4 two-byte varints).
  while (remaining >= 4) {
    uint64_t word;
    std::memcpy(&word, src, sizeof(word));

    if ((word & kHighBits) != kTwoBytePattern) {
      break;
    }

    output[0] = static_cast<T>((src[0] & 0x7f) | (uint32_t(src[1]) << 7));
    output[1] = static_cast<T>((src[2] & 0x7f) | (uint32_t(src[3]) << 7));
    output[2] = static_cast<T>((src[4] & 0x7f) | (uint32_t(src[5]) << 7));
    output[3] = static_cast<T>((src[6] & 0x7f) | (uint32_t(src[7]) << 7));

    src += kWordSize;
    output += 4;
    remaining -= 4;
  }

  // Handle trailing 2-byte varints one at a time. After bulkDecodeSingleByteRun
  // we know the first byte has high bit set (otherwise it would have been
  // consumed as a 1-byte varint), so we just check that the second byte is a
  // terminator.
  while (remaining > 0 && (src[0] & 0x80) && !(src[1] & 0x80)) {
    *output++ = static_cast<T>((src[0] & 0x7f) | (uint32_t(src[1]) << 7));
    src += 2;
    --remaining;
  }

  pos = reinterpret_cast<const char*>(src);
  return remaining;
}

// Lookup table entry for table-driven BMI2 varint decode.
struct alignas(kCacheLineBytes) VarintLookupEntry {
  // Extraction masks for up to 6 completed varints. Unused slots are 0
  uint64_t valueMasks[kMaskLength];
  // Extraction mask for carryover bytes (partial varint at end of chunk), and
  // zero when the chunk ends on a clean varint boundary.
  uint64_t carryOverMask;
  uint8_t numCompleted;
  uint8_t carryOverBits;
  uint8_t padding[6];
};

static_assert(
    sizeof(VarintLookupEntry) == kCacheLineBytes,
    "Must fit one cache line");

// We build the full 64-entry/kMaxControlBitsValue for control bit lookup table
// at compile time.
static constexpr auto kDecodeTable = [] {
  std::array<VarintLookupEntry, kMaxControlBitsValue> table{};
  for (int i = 0; i < kMaxControlBitsValue; ++i) {
    VarintLookupEntry entry{};
    uint64_t currentMask = 0;

    int lastZero = -1, numCompleted = 0;
    const uint8_t controlBits = static_cast<uint8_t>(i);
    for (int j = 0; j < kMaskLength; ++j) {
      currentMask |= uint64_t(0x7f) << (j * 8);
      if (!((controlBits >> j) & 1)) {
        entry.valueMasks[numCompleted] = currentMask;
        ++numCompleted;

        lastZero = j;
        currentMask = 0;
      }
    }

    entry.numCompleted = static_cast<uint8_t>(numCompleted);

    entry.carryOverMask = 0;
    entry.carryOverBits = 0;

    if (lastZero < 5) {
      // Partial varint at end of chunk.
      entry.carryOverMask = currentMask;
      entry.carryOverBits = static_cast<uint8_t>(7 * (5 - lastZero));
    } else if (lastZero == -1) {
      // All 6 bytes are continuation bytes (case 63). Accumulate carryover.
      // This is a rare case
      entry.carryOverMask = currentMask;
      entry.carryOverBits = 42;
    }
    table[i] = entry;
  }
  return table;
}();

unsigned long long pext_u64(unsigned long long __X, unsigned long long __M) {
#ifdef __x86_64__
  return _pext_u64(__X, __M);
#else
  unsigned long p = 0x4040404040404040UL; // initial bit permute control
  const unsigned long mask = 0x8000000000000000UL;
  unsigned long m = __M;
  unsigned long c;
  unsigned long result;

  p = 64 - __builtin_popcountl(__M);
  result = 0;
  /* We could a use a for loop here, but that combined with
   -funroll-loops can expand to a lot of code.  The while
   loop avoids unrolling and the compiler commons the xor
   from clearing the mask bit with the (m != 0) test.  The
   result is a more compact loop setup and body.  */
  while (m != 0) {
    unsigned long t;
    c = __builtin_clzl(m);
    t = (__X & (mask >> c)) >> (p - c);
    m ^= (mask >> c);
    result |= (t);
    p++;
  }
  return (result);
#endif
}

#if !VARINT_SVE2

// Table-driven BMI2 varint decode. Reads extraction masks from a lookup table.
// Takes n and output by reference so the caller can re-dispatch to fast paths
// after this function yields on a single-byte or two-byte run boundary.
template <typename T>
#ifdef __x86_64__
__attribute__((__target__("bmi2")))
#endif
const char*
bulkVarintDecodeBmi2Table(uint64_t& n, const char* pos, T*& output) {
  constexpr uint64_t kControlMask = 0x0000808080808080ULL;
  constexpr int kChunkLen = 6;
  // Control bit pattern for uniform single-byte chunks.
  // cb=0: all 6 bytes are terminators (6 single-byte varints).
  constexpr uint64_t kAllSingleByteCb = 0;

  uint64_t carryover = 0;
  int carryoverBits = 0;
  pos -= kChunkLen;

  while (n >= 8) {
    pos += kChunkLen;

    uint64_t word;
    std::memcpy(&word, pos, sizeof(word));
    const uint64_t cb = pext_u64(word, kControlMask);

    // If there is no carryover from a previous chunk and we see a run of
    // single-byte varints, break out so the caller can use the dedicated
    // fast path.
    if (carryoverBits == 0 && cb == kAllSingleByteCb) {
      return pos;
    }

    // Case 63 (all continuation bytes) requires accumulating carryover
    // rather than replacing it. This case is extremely rare
    if (FOLLY_UNLIKELY(cb == 63)) {
      carryover |= pext_u64(word, 0x00007f7f7f7f7f7fULL) << carryoverBits;
      carryoverBits += 42;
      continue;
    }

    const auto& info = kDecodeTable[cb];

    // Extract and store up to 6 values. Unused mask slots are 0, producing
    // harmless zero writes that will be overwritten by subsequent iterations
    output[0] = static_cast<T>(
        (pext_u64(word, info.valueMasks[0]) << carryoverBits) | carryover);
    output[1] = static_cast<T>(pext_u64(word, info.valueMasks[1]));
    output[2] = static_cast<T>(pext_u64(word, info.valueMasks[2]));
    output[3] = static_cast<T>(pext_u64(word, info.valueMasks[3]));
    output[4] = static_cast<T>(pext_u64(word, info.valueMasks[4]));
    output[5] = static_cast<T>(pext_u64(word, info.valueMasks[5]));

    output += info.numCompleted;
    n -= info.numCompleted;

    // Update carryover. When carryoverMask is 0, _pext returns 0 and
    // carryoverBits is 0, effectively clearing the carryover state.
    carryover = pext_u64(word, info.carryOverMask);
    carryoverBits = info.carryOverBits;
  }

  pos += kChunkLen;
  if (n > 0) {
    if constexpr (std::is_same_v<T, uint32_t>) {
      *output++ = readVarint32(&pos) << carryoverBits | carryover;
      for (uint64_t i = 1; i < n; ++i) {
        *output++ = readVarint32(&pos);
      }
    } else {
      *output++ = readVarint64(&pos) << carryoverBits | carryover;
      for (uint64_t i = 1; i < n; ++i) {
        *output++ = readVarint64(&pos);
      }
    }
    n = 0;
  }
  return pos;
}

#else

// Table-driven BEXT varint decode. Reads extraction masks from a lookup table.
// Takes n and output by reference so the caller can re-dispatch to fast paths
// after this function yields on a single-byte or two-byte run boundary.
template <typename T>
const char*
bulkVarintDecodeBmi2Table(uint64_t& n, const char* pos, T*& output) {
  svuint64_t kControlMask = svdup_n_u64(0x0000808080808080ULL);
  svuint64_t kCarryoverMask = svdup_n_u64(0x00007f7f7f7f7f7fULL);
  constexpr int kChunkLen = 6;
  // Control bit pattern for uniform single-byte chunks.
  // cb=0: all 6 bytes are terminators (6 single-byte varints).
  constexpr double kAllSingleByteCb = 0.0;

  svuint64_t carryover = svdup_n_u64(0);
  svuint64_t carryoverBits = svdup_n_u64(0);
  pos -= kChunkLen;

  while (n >= 8) {
    pos += kChunkLen;

    svuint64_t svWord = svdup_n_u64(*reinterpret_cast<const uint64_t*>(pos));
    uint64x2_t cb = svget_neonq(svbext_u64(svWord, kControlMask));
    float64x2_t fb = vreinterpretq_f64_u64(cb);
    float64x2_t fob = vreinterpretq_f64_u64(svget_neonq(carryoverBits));

    // If there is no carryover from a previous chunk and we see a run of
    // single-byte varints, break out so the caller can use the dedicated
    // fast path.
    if ((fob[0] == 0.0) && (fb[0] == kAllSingleByteCb)) {
      return pos;
    }

    uint64_t gb = cb[0];

    // Case 63 (all continuation bytes) requires accumulating carryover
    // rather than replacing it. This case is extremely rare
    if (FOLLY_UNLIKELY(gb == 63)) {
      svuint64_t cbits = svbext_u64(svWord, kCarryoverMask);
      carryover |= cbits << carryoverBits;
      carryoverBits += 42;
      continue;
    }

    const auto& info = kDecodeTable[gb];

    // Extract and store up to 6 values. Unused mask slots are 0, producing
    // harmless zero writes that will be overwritten by subsequent iterations

    svuint64_t valueMasks0 =
        svset_neonq(svundef_u64(), vld1q_u64(&(info.valueMasks[0])));
    svuint64_t valueMasks1 =
        svset_neonq(svundef_u64(), vld1q_u64(&(info.valueMasks[2])));
    svuint64_t valueMasks2 =
        svset_neonq(svundef_u64(), vld1q_u64(&(info.valueMasks[4])));

    svuint64_t svo0 = svbext_u64(svWord, valueMasks0);
    svuint64_t svo1 = svbext_u64(svWord, valueMasks1);
    svuint64_t svo2 = svbext_u64(svWord, valueMasks2);

    uint64x2_t vo0 = svget_neonq(svo0);
    uint64x2_t vo1 = svget_neonq(svo1);
    uint64x2_t vo2 = svget_neonq(svo2);

    auto outputPtr = output;

    uint64x2_t cvo0 = svget_neonq((svo0 << carryoverBits) | carryover);

    if constexpr (sizeof(T) == 8) {
      vst1q_lane_u64(outputPtr, cvo0, 0);
      vst1q_lane_u64(outputPtr + 1, vo0, 1);
      vst1q_u64(outputPtr + 2, vo1);
      vst1q_u64(outputPtr + 4, vo2);
    } else if constexpr (sizeof(T) == 4) {
      vst1q_lane_u32(outputPtr, vreinterpretq_u32_u64(cvo0), 0);
      vst1q_lane_u32(outputPtr + 1, vreinterpretq_u32_u64(vo0), 2);
      svst1w_u64(svwhilelt_b32_u64(0, 4), outputPtr + 2, svo1);
      svst1w_u64(svwhilelt_b32_u64(0, 4), outputPtr + 4, svo2);
    } else if constexpr (sizeof(T) == 2) {
      vst1q_lane_u16(outputPtr, vreinterpretq_u16_u64(cvo0), 0);
      vst1q_lane_u16(outputPtr + 1, vreinterpretq_u16_u64(vo0), 4);
      svst1h_u64(svwhilelt_b16_u64(0, 8), outputPtr + 2, svo1);
      svst1h_u64(svwhilelt_b16_u64(0, 8), outputPtr + 4, svo2);
    } else if constexpr (sizeof(T) == 1) {
      vst1q_lane_u8(outputPtr, vreinterpretq_u8_u64(cvo0), 0);
      vst1q_lane_u8(outputPtr + 1, vreinterpretq_u8_u64(vo0), 8);
      svst1b_u64(svwhilelt_b8_u64(0, 16), outputPtr + 2, svo1);
      svst1b_u64(svwhilelt_b8_u64(0, 16), outputPtr + 4, svo2);
    }

    output += info.numCompleted;
    n -= info.numCompleted;

    // Update carryover. When carryoverMask is 0, _pext returns 0 and
    // carryoverBits is 0, effectively clearing the carryover state.
    carryover = svbext_u64(svWord, svdup_n_u64(info.carryOverMask));
    carryoverBits = svdup_n_u64(info.carryOverBits);
  }

  uint64_t carryoverVar = carryover[0];
  uint64_t carryoverBitsVar = carryoverBits[0];

  pos += kChunkLen;
  if (n > 0) {
    if constexpr (std::is_same_v<T, uint32_t>) {
      *output++ = readVarint32(&pos) << carryoverBitsVar | carryoverVar;
      for (uint64_t i = 1; i < n; ++i) {
        *output++ = readVarint32(&pos);
      }
    } else {
      *output++ = readVarint64(&pos) << carryoverBitsVar | carryoverVar;
      for (uint64_t i = 1; i < n; ++i) {
        *output++ = readVarint64(&pos);
      }
    }
    n = 0;
  }
  return pos;
}

#endif

// Dispatch loop: cycles between fast paths and the general BMI2 decoder.
// When the BMI2 decoder detects a uniform single-byte or two-byte chunk
// boundary (with no carryover), it yields back here so the dedicated fast
// paths can handle the run efficiently.
template <typename T>
inline const char*
bulkVarintDecodeDispatch(uint64_t n, const char* pos, T* output) {
  auto remaining = n;
  while (remaining > 0) {
    remaining = bulkDecodeSingleByteRun(remaining, pos, output);
    if (remaining == 0) {
      break;
    }
    remaining = bulkDecodeTwoByteRun(remaining, pos, output);
    if (remaining == 0) {
      break;
    }
    pos = bulkVarintDecodeBmi2Table(remaining, pos, output);
  }
  return pos;
}

const char* bulkVarintDecode32(uint64_t n, const char* pos, uint32_t* output) {
  return bulkVarintDecodeDispatch(n, pos, output);
}

const char* bulkVarintDecode64(uint64_t n, const char* pos, uint64_t* output) {
  return bulkVarintDecodeDispatch(n, pos, output);
}

} // namespace facebook::nimble::varint
