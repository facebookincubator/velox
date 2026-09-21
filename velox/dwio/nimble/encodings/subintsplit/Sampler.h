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

// Stream sampler for the SubIntSplit DP planner.
//
// Writes into a caller-supplied std::vector<uint64_t> so the allocation can be
// reused across multiple encode() calls or DP iterations. Values are stored as
// uint64_t (physical bit pattern, zero-extended for <64-bit types).

namespace facebook::nimble::subintsplit {

struct SamplerConfig {
  /// Maximum number of samples to draw. The split selector runs an
  /// O(kBits^2 * numSamples) DP over this sample: larger samples sharpen the
  /// cost estimates only marginally while making encode dramatically slower
  /// (especially for 64-bit types).
  size_t maxSamples{2'048};

  /// Contiguous block size for block-stratified sampling. 0 selects uniform
  /// stride sampling instead.
  size_t blockSize{128};
};

inline SamplerConfig defaultSamplerConfig() noexcept {
  return SamplerConfig{};
}

namespace detail {

template <typename PhysicalType>
inline uint64_t toBits(const PhysicalType& value) noexcept {
  uint64_t bits = 0;
  __builtin_memcpy(&bits, &value, sizeof(PhysicalType));
  return bits;
}

// Evenly-spaced contiguous windows. Preserves local temporal structure, which
// is what makes the run-length and frame-residual metrics meaningful.
template <typename PhysicalType>
inline void sampleBlocks(
    std::span<const PhysicalType> values,
    size_t target,
    size_t blockSize,
    std::vector<uint64_t>& out) {
  const size_t numBlocks = std::max<size_t>(1, target / blockSize);
  const size_t blockStride = std::max<size_t>(1, values.size() / numBlocks);
  for (size_t block = 0; block < numBlocks && out.size() < target; ++block) {
    const size_t start = block * blockStride;
    const size_t end = std::min(start + blockSize, values.size());
    for (size_t i = start; i < end && out.size() < target; ++i) {
      out.push_back(toBits(values[i]));
    }
  }
}

template <typename PhysicalType>
inline void sampleStride(
    std::span<const PhysicalType> values,
    size_t target,
    std::vector<uint64_t>& out) {
  const size_t stride = std::max<size_t>(1, values.size() / target);
  for (size_t i = 0; i < values.size(); i += stride) {
    out.push_back(toBits(values[i]));
  }
}

} // namespace detail

/// Fills `out` with up to `config.maxSamples` bit patterns drawn from `values`.
/// `out` is resized in place, so no heap allocation occurs when it already has
/// sufficient capacity.
template <typename PhysicalType>
void sampleIntoU64(
    std::span<const PhysicalType> values,
    std::vector<uint64_t>& out,
    const SamplerConfig& config = defaultSamplerConfig()) {
  static_assert(sizeof(PhysicalType) <= 8);

  out.clear();
  if (values.empty()) {
    return;
  }

  const size_t target = std::min(
      config.maxSamples > 0 ? config.maxSamples : values.size(), values.size());
  out.reserve(target);

  if (config.blockSize > 0) {
    detail::sampleBlocks(values, target, config.blockSize, out);
  } else {
    detail::sampleStride(values, target, out);
  }
}

} // namespace facebook::nimble::subintsplit
