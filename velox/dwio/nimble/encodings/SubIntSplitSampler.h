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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Stream sampler for SubIntSplitEncoding's DP planner.
//
// Writes into a caller-supplied std::vector<uint64_t> so the allocation can
// be reused across multiple encode() calls or DP iterations. Values are
// stored as uint64_t (physical bit pattern, zero-extended for <64-bit types).
//
// Config is passed explicitly; callers obtain a default via
// defaultSamplerConfig() and may override fields before calling.

namespace facebook::nimble::detail::subintsplit {

struct SamplerConfig {
  // Maximum number of samples to draw. The split selector runs an
  // O(kBits^2 * sampleSize) DP over this sample (larger
  // samples sharpen the cost estimates only marginally while making encode
  // dramatically slower (especially for 64-bit types)).
  size_t maxSamples{2'048};
  // Contiguous block size for block-stratified sampling.
  // 0 → use uniform stride sampling instead.
  size_t blockSize{128};
};

inline SamplerConfig defaultSamplerConfig() noexcept {
  return SamplerConfig{.maxSamples = 2'048, .blockSize = 128};
}

// Fill `out` with up to cfg.maxSamples uint64_t values drawn from `values`.
// Block-stratified sampling (when cfg.blockSize > 0) preserves local temporal
// structure, which is important for accurate run-length and frame-residual
// metrics. Stride sampling (blockSize == 0) spreads samples uniformly.
//
// Writes directly into `out` (which is resized); no heap allocation occurs if
// `out` already has sufficient capacity.
template <typename physicalType>
void sampleIntoU64(
    std::span<const physicalType> values,
    std::vector<uint64_t>& out,
    const SamplerConfig& cfg = defaultSamplerConfig()) {
  static_assert(sizeof(physicalType) <= 8);

  const size_t n = values.size();
  out.clear();
  if (n == 0) {
    return;
  }

  const size_t target = std::min(cfg.maxSamples > 0 ? cfg.maxSamples : n, n);
  out.reserve(target);

  if (cfg.blockSize > 0) {
    // Block-stratified: evenly-spaced contiguous windows.
    const size_t numBlocks = std::max<size_t>(1, target / cfg.blockSize);
    const size_t blockStride = std::max<size_t>(1, n / numBlocks);
    for (size_t b = 0; b < numBlocks && out.size() < target; ++b) {
      const size_t start = b * blockStride;
      const size_t end = std::min(start + cfg.blockSize, n);
      for (size_t i = start; i < end && out.size() < target; ++i) {
        uint64_t bits = 0;
        __builtin_memcpy(&bits, &values[i], sizeof(physicalType));
        out.push_back(bits);
      }
    }
  } else {
    // Uniform stride sampling.
    const size_t stride = std::max<size_t>(1, n / target);
    for (size_t i = 0; i < n; i += stride) {
      uint64_t bits = 0;
      __builtin_memcpy(&bits, &values[i], sizeof(physicalType));
      out.push_back(bits);
    }
  }
}

} // namespace facebook::nimble::detail::subintsplit
