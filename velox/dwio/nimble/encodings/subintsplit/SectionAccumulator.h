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
#pragma once

#include <cstdint>
#include <cstring>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// Combines one decoded section's values back into the output stream: mask off
// the section's bits, shift them to their place in the value, and either write
// or OR them into the output.

namespace facebook::nimble::subintsplit {
namespace detail {

#ifdef __AVX2__

// 256-bit lane operations for a given output element width.
template <int kOutputBytes>
struct LaneOps;

template <>
struct LaneOps<8> {
  static constexpr int kLanes = 4;

  static __m256i broadcast(uint64_t value) noexcept {
    return _mm256_set1_epi64x(static_cast<int64_t>(value));
  }

  static __m128i shiftCount(int shift) noexcept {
    return _mm_cvtsi64_si128(static_cast<int64_t>(shift));
  }

  static __m256i shiftLeft(__m256i values, __m128i count) noexcept {
    return _mm256_sll_epi64(values, count);
  }
};

template <>
struct LaneOps<4> {
  static constexpr int kLanes = 8;

  static __m256i broadcast(uint64_t value) noexcept {
    return _mm256_set1_epi32(static_cast<int32_t>(value));
  }

  static __m128i shiftCount(int shift) noexcept {
    return _mm_cvtsi32_si128(static_cast<int32_t>(shift));
  }

  static __m256i shiftLeft(__m256i values, __m128i count) noexcept {
    return _mm256_sll_epi32(values, count);
  }
};

// Zero-extending load of one vector's worth of narrow section values.
//
// kSrcPrefetchElements is tuned per conversion and deliberately not uniform:
// the conversions consume their source at different byte rates.
template <int kSectionBytes, int kOutputBytes>
struct SectionWidening {
  static constexpr bool kSupported = false;
};

template <>
struct SectionWidening<1, 8> {
  static constexpr bool kSupported = true;
  static constexpr int kSrcPrefetchElements = 32;

  static __m256i load(const uint8_t* src) noexcept {
    int32_t packed;
    std::memcpy(&packed, src, sizeof(packed));
    return _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(packed));
  }
};

template <>
struct SectionWidening<2, 8> {
  static constexpr bool kSupported = true;
  static constexpr int kSrcPrefetchElements = 32;

  static __m256i load(const uint16_t* src) noexcept {
    return _mm256_cvtepu16_epi64(
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src)));
  }
};

template <>
struct SectionWidening<4, 8> {
  static constexpr bool kSupported = true;
  static constexpr int kSrcPrefetchElements = 16;

  static __m256i load(const uint32_t* src) noexcept {
    return _mm256_cvtepu32_epi64(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
  }
};

template <>
struct SectionWidening<1, 4> {
  static constexpr bool kSupported = true;
  static constexpr int kSrcPrefetchElements = 64;

  static __m256i load(const uint8_t* src) noexcept {
    int64_t packed;
    std::memcpy(&packed, src, sizeof(packed));
    return _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(packed));
  }
};

template <>
struct SectionWidening<2, 4> {
  static constexpr bool kSupported = true;
  static constexpr int kSrcPrefetchElements = 32;

  static __m256i load(const uint16_t* src) noexcept {
    return _mm256_cvtepu16_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
  }
};

// Widens, masks, shifts and accumulates whole vectors. Returns the number of
// elements consumed, leaving the tail to the scalar loop.
template <bool kIsFirst, typename SectionT, typename OutputT>
uint32_t accumulateVectors(
    const SectionT* __restrict__ src,
    OutputT* __restrict__ dst,
    uint32_t count,
    uint64_t mask,
    int shift,
    OutputT orConstant) noexcept {
  using Widening = SectionWidening<sizeof(SectionT), sizeof(OutputT)>;
  using Ops = LaneOps<sizeof(OutputT)>;

  const __m128i vectorShift = Ops::shiftCount(shift);
  const __m256i vectorMask = Ops::broadcast(mask);
  const __m256i vectorConstant =
      Ops::broadcast(static_cast<uint64_t>(orConstant));

  uint32_t i = 0;
  for (; i + Ops::kLanes <= count; i += Ops::kLanes) {
    _mm_prefetch(
        reinterpret_cast<const char*>(src + i + Widening::kSrcPrefetchElements),
        _MM_HINT_T1);
    _mm_prefetch(reinterpret_cast<const char*>(dst + i + 32), _MM_HINT_T1);

    __m256i values = _mm256_and_si256(Widening::load(src + i), vectorMask);
    if (shift != 0) {
      values = Ops::shiftLeft(values, vectorShift);
    }

    auto* out = reinterpret_cast<__m256i*>(dst + i);
    if constexpr (kIsFirst) {
      _mm256_storeu_si256(out, _mm256_or_si256(values, vectorConstant));
    } else {
      const __m256i existing =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
      _mm256_storeu_si256(out, _mm256_or_si256(existing, values));
    }
  }
  return i;
}

#endif // __AVX2__

} // namespace detail

/// Accumulates one section's decoded values into the output buffer.
///
/// `kIsFirst` selects a pure write that initialises the output element and ORs
/// in `orConstant` (which carries the bits of every constant section); the
/// remaining sections OR their bits into what is already there. This is what
/// lets the chunk loop skip a separate fill pass.
///
/// `__restrict__` tells the compiler src and dst do not alias, which is what
/// enables auto-vectorisation of the same-width loop below and gives the AVX2
/// path correct alias semantics.
template <bool kIsFirst, typename SectionT, typename OutputT>
void accumulateSection(
    const SectionT* __restrict__ src,
    OutputT* __restrict__ dst,
    uint32_t count,
    uint64_t mask,
    int shift,
    OutputT orConstant) noexcept {
  const SectionT narrowMask = static_cast<SectionT>(mask);
  uint32_t i = 0;

#ifdef __AVX2__
  if constexpr (detail::SectionWidening<sizeof(SectionT), sizeof(OutputT)>::
                    kSupported) {
    i = detail::accumulateVectors<kIsFirst>(
        src, dst, count, mask, shift, orConstant);
  }
#endif

  // Same-width sections, builds without AVX2, and the vector tail. The
  // __restrict__ qualifiers let the compiler vectorise this on its own for the
  // same-width cases.
  if constexpr (kIsFirst) {
    for (; i < count; ++i) {
      dst[i] =
          (static_cast<OutputT>(src[i] & narrowMask) << shift) | orConstant;
    }
  } else {
    for (; i < count; ++i) {
      dst[i] |= static_cast<OutputT>(src[i] & narrowMask) << shift;
    }
  }
}

} // namespace facebook::nimble::subintsplit
