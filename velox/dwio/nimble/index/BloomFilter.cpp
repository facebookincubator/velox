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
#include "velox/dwio/nimble/index/BloomFilter.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "velox/dwio/nimble/common/Exceptions.h"

#define XXH_INLINE_ALL
#include <xxhash.h>

#include "velox/common/base/BitUtil.h"

namespace facebook::nimble::index {

namespace {

// Computes the number of 256-bit blocks needed for the bloom filter based on
// the expected number of entries and the desired bits per key.
uint32_t computeNumBlocks(uint64_t numEntries, float bitsPerKey) {
  NIMBLE_USER_CHECK(
      std::isfinite(bitsPerKey) && bitsPerKey > 0,
      "Bloom filter bits per key must be finite and positive, but got: {}",
      bitsPerKey);
  // Total bits needed, rounded up to block boundaries.
  const uint64_t totalBits = std::max(
      static_cast<uint64_t>(numEntries * bitsPerKey),
      uint64_t{BloomFilter::kBlockSizeBits});
  const uint32_t numBlocks = static_cast<uint32_t>(
      velox::bits::divRoundUp(totalBits, BloomFilter::kBlockSizeBits));
  // Always allocate at least one block to avoid degenerate cases.
  return std::max(numBlocks, uint32_t{1});
}

} // namespace

BloomFilter::BloomFilter(
    uint64_t numEntries,
    float bitsPerKey,
    velox::memory::MemoryPool* pool)
    : numBlocks_{computeNumBlocks(numEntries, bitsPerKey)},
      bitsPerKey_{bitsPerKey},
      data_{velox::AlignedBuffer::allocate<uint8_t>(
          numBlocks_ * kBlockSizeBytes,
          pool,
          0)} {}

BloomFilter::BloomFilter(
    uint32_t numBlocks,
    const uint8_t* data,
    size_t dataSize,
    velox::memory::MemoryPool* pool)
    : numBlocks_{numBlocks},
      bitsPerKey_{0.0f},
      data_{velox::AlignedBuffer::allocate<uint8_t>(dataSize, pool)} {
  NIMBLE_CHECK_GT(numBlocks, 0u, "numBlocks must be positive");
  NIMBLE_CHECK_EQ(
      dataSize,
      static_cast<size_t>(numBlocks_) * kBlockSizeBytes,
      "data size mismatch");
  std::memcpy(data_->asMutable<uint8_t>(), data, dataSize);
}

uint64_t BloomFilter::hashKey(std::string_view key) {
  return XXH64(key.data(), key.size(), /*seed=*/0);
}

void BloomFilter::insert(std::string_view key) {
  insertHash(hashKey(key));
}

bool BloomFilter::testKey(std::string_view key) const {
  return testHash(hashKey(key));
}

void BloomFilter::insertHash(uint64_t hash) {
  auto* words = mutableBlock(blockIndex(hash));
  const auto key32 = static_cast<uint32_t>(hash);
  for (uint32_t i = 0; i < kNumProbesPerBlock; ++i) {
    const uint32_t bitIndex = (key32 * kSalts[i]) >> kBitShift;
    words[i] |= (uint32_t{1} << bitIndex);
  }
}

bool BloomFilter::testHash(uint64_t hash) const {
  const auto* words = block(blockIndex(hash));
  const auto key32 = static_cast<uint32_t>(hash);
  for (uint32_t i = 0; i < kNumProbesPerBlock; ++i) {
    const uint32_t bitIndex = (key32 * kSalts[i]) >> kBitShift;
    if ((words[i] & (uint32_t{1} << bitIndex)) == 0) {
      return false;
    }
  }
  return true;
}

} // namespace facebook::nimble::index
