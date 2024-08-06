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

#include "velox/common/base/SimdUtil.h"
#include <folly/Preprocessor.h>

namespace facebook::velox::simd {

void gatherBits(
    const uint64_t* bits,
    folly::Range<const int32_t*> indexRange,
    uint64_t* result) {
  constexpr int32_t kStep = xsimd::batch<int32_t>::size;
  const auto size = indexRange.size();
  auto indices = indexRange.data();
  uint8_t* resultPtr = reinterpret_cast<uint8_t*>(result);
  if (FOLLY_LIKELY(size < 5)) {
    uint8_t smallResult = 0;
    for (auto i = 0; i < size; ++i) {
      smallResult |= static_cast<uint8_t>(bits::isBitSet(bits, indices[i]))
          << i;
    }
    *resultPtr = smallResult;
    return;
  }

  int32_t i = 0;
  for (; i + kStep < size; i += kStep) {
    uint16_t flags =
        simd::gather8Bits(bits, xsimd::load_unaligned(indices + i), kStep);
    bits::storeBitsToByte<kStep>(flags, resultPtr, i);
  }
  const auto bitsLeft = size - i;
  if (bitsLeft > 0) {
    uint16_t flags =
        simd::gather8Bits(bits, xsimd::load_unaligned(indices + i), bitsLeft);
    bits::storeBitsToByte<kStep>(flags, resultPtr, i);
  }
}

namespace detail {

alignas(kPadding) int32_t byteSetBits[256][8];
alignas(kPadding) int32_t permute4x64Indices[16][8];

const LeadingMask<int32_t, xsimd::default_arch> leadingMask32;
const LeadingMask<int64_t, xsimd::default_arch> leadingMask64;

const FromBitMask<int32_t, xsimd::default_arch> fromBitMask32;
const FromBitMask<int64_t, xsimd::default_arch> fromBitMask64;

} // namespace detail

namespace {

void initByteSetBits() {
  for (int32_t i = 0; i < 256; ++i) {
    int32_t* entry = detail::byteSetBits[i];
    int32_t fill = 0;
    for (auto b = 0; b < 8; ++b) {
      if (i & (1 << b)) {
        entry[fill++] = b;
      }
    }
    for (; fill < 8; ++fill) {
      entry[fill] = fill;
    }
  }
}

void initPermute4x64Indices() {
  for (int i = 0; i < 16; ++i) {
    int32_t* result = detail::permute4x64Indices[i];
    int32_t fill = 0;
    for (int bit = 0; bit < 4; ++bit) {
      if (i & (1 << bit)) {
        result[fill++] = bit * 2;
        result[fill++] = bit * 2 + 1;
      }
    }
    for (; fill < 8; ++fill) {
      result[fill] = fill;
    }
  }
}

} // namespace

bool initializeSimdUtil() {
  static bool inited = false;
  if (inited) {
    return true;
  }
  initByteSetBits();
  initPermute4x64Indices();
  inited = true;
  return true;
}

static bool FB_ANONYMOUS_VARIABLE(g_simdConstants) = initializeSimdUtil();

namespace detail {

#if XSIMD_WITH_SSE4_2
using CharVector = xsimd::batch<uint8_t, xsimd::sse4_2>;
#elif XSIMD_WITH_NEON
using CharVector = xsimd::batch<uint8_t, xsimd::neon>;
#endif

const int kPageSize = sysconf(_SC_PAGESIZE);
FOLLY_ALWAYS_INLINE bool pageSafe(const void* const ptr) {
  return ((kPageSize - 1) & reinterpret_cast<std::uintptr_t>(ptr)) <=
      kPageSize - CharVector::size;
}

template <bool compiled, size_t compiledNeedleSize>
size_t FOLLY_ALWAYS_INLINE smidStrstrMemcmp(
    const char* s,
    size_t n,
    const char* needle,
    size_t needleSize) {
  static_assert(compiledNeedleSize >= 2);
  VELOX_CHECK_GT(needleSize, 1);
  VELOX_CHECK_GT(n, 0);
  auto first = CharVector::broadcast(needle[0]);
  auto last = CharVector::broadcast(needle[needleSize - 1]);
  size_t i = 0;
  // Fast path for page-safe data.
  // It`s safe to over-read CharVector if all-data are in same page.
  // see: https://mudongliang.github.io/x86/html/file_module_x86_id_208.html
  // While executing in 16-bit addressing mode, a linear address for a 128-bit
  // data access that overlaps the end of a 16-bit segment is not allowed and is
  // defined as reserved behavior. A specific processor implementation may or
  // may not generate a general-protection exception (#GP) in this situation,
  // and the address that spans the end of the segment may or may not wrap
  // around to the beginning of the segment.
  for (; i <= n - needleSize && pageSafe(s + i + needleSize - 1) &&
       pageSafe(s + i);
       i += CharVector::size) {
    auto blockFirst = CharVector::load_unaligned(s + i);
    auto blockLast = CharVector::load_unaligned(s + i + needleSize - 1);

    const auto eqFirst = (first == blockFirst);
    const auto eqLast = (last == blockLast);

    auto mask = toBitMask(eqFirst && eqLast);

    while (mask != 0) {
      const auto bitpos = __builtin_ctz(mask);
      if constexpr (compiled) {
        if constexpr (compiledNeedleSize == 2) {
          return i + bitpos;
        }
        if (memcmp(s + i + bitpos + 1, needle + 1, compiledNeedleSize - 2) ==
            0) {
          return i + bitpos;
        }
      } else {
        if (memcmp(s + i + bitpos + 1, needle + 1, needleSize - 2) == 0) {
          return i + bitpos;
        }
      }
      mask = mask & (mask - 1);
    }
  }
  // Fallback path for generic path.
  for (; i <= n - needleSize; ++i) {
    if constexpr (compiled) {
      if (memcmp(s + i, needle, compiledNeedleSize) == 0) {
        return i;
      }
    } else {
      if (memcmp(s + i, needle, needleSize) == 0) {
        return i;
      }
    }
  }

  return std::string::npos;
};

} // namespace detail

/// A faster implementation for c_strstr(), about 2x faster than string_view`s
/// find(), proved by TpchLikeBenchmark. Use xsmid-batch to compare first&&last
/// char first, use fixed-memcmp to compare left chars. Inline in header file
/// will be a little faster.
size_t simdStrstr(const char* s, size_t n, const char* needle, size_t k) {
  size_t result = std::string::npos;

  if (n < k) {
    return result;
  }

  switch (k) {
    case 0:
      return 0;

    case 1: {
      const char* res = strchr(s, needle[0]);

      return (res != nullptr) ? res - s : std::string::npos;
    }
#define FIXED_MEM_STRSTR(size)                                         \
  case size:                                                           \
    result = detail::smidStrstrMemcmp<true, size>(s, n, needle, size); \
    break;
      FIXED_MEM_STRSTR(2)
      FIXED_MEM_STRSTR(3)
      FIXED_MEM_STRSTR(4)
      FIXED_MEM_STRSTR(5)
      FIXED_MEM_STRSTR(6)
      FIXED_MEM_STRSTR(7)
      FIXED_MEM_STRSTR(8)
      FIXED_MEM_STRSTR(9)
      FIXED_MEM_STRSTR(10)
      FIXED_MEM_STRSTR(11)
      FIXED_MEM_STRSTR(12)
      FIXED_MEM_STRSTR(13)
      FIXED_MEM_STRSTR(14)
      FIXED_MEM_STRSTR(15)
      FIXED_MEM_STRSTR(16)
      FIXED_MEM_STRSTR(17)
      FIXED_MEM_STRSTR(18)
    default:
      result = detail::smidStrstrMemcmp<false, 2>(s, n, needle, k);
      break;
  }
#undef FIXED_MEM_STRSTR
  // load_unaligned is used for better performance, so result maybe bigger than
  // n-k.
  if (result <= n - k) {
    return result;
  } else {
    return std::string::npos;
  }
}

} // namespace facebook::velox::simd
