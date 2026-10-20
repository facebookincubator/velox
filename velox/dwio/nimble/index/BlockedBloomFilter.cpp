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
#include "velox/dwio/nimble/index/BlockedBloomFilter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#include "velox/dwio/nimble/common/Exceptions.h"

#define XXH_INLINE_ALL
#include <xxhash.h>

namespace facebook::nimble::index {

namespace {

// Eight 32-bit words form the 256-bit block that a single key touches.
constexpr uint32_t kNumProbesPerBlock{8};
constexpr uint32_t kBlockSizeBytes{kNumProbesPerBlock * sizeof(uint32_t)};
constexpr uint32_t kBlockSizeBits{kBlockSizeBytes * 8};

// Keys hashed and prefetched before any of them is probed. Large enough to
// keep several loads in flight, small enough that the hashes stay in
// registers and L1.
constexpr size_t kPrefetchRunLength{16};

// Odd multipliers from the Parquet split-block filter. Multiplying the hash by
// each one and keeping the top bits sends the eight probes to independent bit
// positions within their own words.
constexpr uint32_t kSalts[kNumProbesPerBlock] = {
    0x47b6137bU,
    0x44974d91U,
    0x8824ad5bU,
    0xa2b7289dU,
    0x705495c7U,
    0x2df1424bU,
    0x9efc4947U,
    0x5c6bfb31U,
};

// Discards all but the top five bits of a salted product, which selects one of
// the 32 bits in a word.
constexpr uint32_t kBitShift{32 - 5};

uint64_t hashKey(std::string_view key) {
  return XXH64(key.data(), key.size(), /*seed=*/0);
}

// Selects the block from the high half of the hash and leaves the low half for
// the probes inside it, so block choice and bit choice stay independent.
// Returns the offset of the block's first word. Widening before scaling
// matters: in 32-bit arithmetic the product wraps once a filter passes 2^29
// blocks, which would send reads and writes outside the buffer.
size_t blockWordOffset(uint64_t hash, uint32_t numBlocks) {
  const auto blockIndex = static_cast<uint32_t>((hash >> 32) % numBlocks);
  return static_cast<size_t>(blockIndex) * kNumProbesPerBlock;
}

uint32_t probeMask(uint64_t hash, uint32_t probe) {
  const auto probeSeed = static_cast<uint32_t>(hash);
  return uint32_t{1} << ((probeSeed * kSalts[probe]) >> kBitShift);
}

void insertHash(uint32_t* words, uint32_t numBlocks, uint64_t hash) {
  auto* block = words + blockWordOffset(hash, numBlocks);
  for (uint32_t probe = 0; probe < kNumProbesPerBlock; ++probe) {
    block[probe] |= probeMask(hash, probe);
  }
}

bool testHash(const uint32_t* words, uint32_t numBlocks, uint64_t hash) {
  const auto* block = words + blockWordOffset(hash, numBlocks);
  for (uint32_t probe = 0; probe < kNumProbesPerBlock; ++probe) {
    if ((block[probe] & probeMask(hash, probe)) == 0) {
      return false;
    }
  }
  return true;
}

uint32_t computeNumBlocks(uint64_t numKeys, float bitsPerKey) {
  NIMBLE_USER_CHECK(
      std::isfinite(bitsPerKey) && bitsPerKey > 0,
      "Bloom filter bits per key must be finite and positive, but got: {}",
      bitsPerKey);
  // Size in double. The same product in float reaches infinity for a
  // bitsPerKey that still passes the check above, and converting infinity back
  // to an integer is undefined. Truncating the bit count before rounding up to
  // whole blocks is what the integer arithmetic this replaced did, and keeping
  // it means an existing config still produces a filter of the same size.
  const double totalBits = std::max(
      std::floor(static_cast<double>(numKeys) * bitsPerKey),
      static_cast<double>(kBlockSizeBits));
  const double numBlocks = std::ceil(totalBits / kBlockSizeBits);
  NIMBLE_USER_CHECK(
      numBlocks <= static_cast<double>(std::numeric_limits<uint32_t>::max()),
      "Bloom filter is too large to address: {} keys at {} bits per key",
      numKeys,
      bitsPerKey);
  return static_cast<uint32_t>(numBlocks);
}

uint32_t blockCountOf(std::string_view payload) {
  NIMBLE_CHECK_GT(payload.size(), 0u, "Blocked bloom filter payload is empty");
  NIMBLE_CHECK_EQ(
      payload.size() % kBlockSizeBytes,
      0u,
      "Blocked bloom filter payload is not a whole number of blocks");
  return static_cast<uint32_t>(payload.size() / kBlockSizeBytes);
}

} // namespace

BlockedBloomFilterBuilder::BlockedBloomFilterBuilder(
    uint64_t numKeys,
    float bitsPerKey,
    velox::memory::MemoryPool* pool)
    : numBlocks_{computeNumBlocks(numKeys, bitsPerKey)},
      data_{velox::AlignedBuffer::allocate<char>(
          static_cast<size_t>(numBlocks_) * kBlockSizeBytes +
              BloomFilterTrailer::kSize,
          pool,
          0)} {}

void BlockedBloomFilterBuilder::insert(std::string_view key) {
  NIMBLE_CHECK(!finished_, "Cannot insert into a finished bloom filter");
  insertHash(
      reinterpret_cast<uint32_t*>(data_->asMutable<char>()),
      numBlocks_,
      hashKey(key));
}

std::string_view BlockedBloomFilterBuilder::finish() {
  NIMBLE_CHECK(!finished_, "Bloom filter is already finished");
  finished_ = true;
  const auto blocksSize = static_cast<size_t>(numBlocks_) * kBlockSizeBytes;
  auto* raw = data_->asMutable<char>();
  BloomFilterTrailer::write(raw + blocksSize, BloomFilterType::kBlocked);
  return {raw, blocksSize + BloomFilterTrailer::kSize};
}

BlockedBloomFilterReader::BlockedBloomFilterReader(
    std::string_view payload,
    velox::memory::MemoryPool* pool)
    : numBlocks_{blockCountOf(payload)},
      data_{velox::AlignedBuffer::allocate<char>(payload.size(), pool)} {
  std::memcpy(data_->asMutable<char>(), payload.data(), payload.size());
}

bool BlockedBloomFilterReader::maybeContains(std::string_view key) const {
  const auto* words = reinterpret_cast<const uint32_t*>(data_->as<char>());
  return testHash(words, numBlocks_, hashKey(key));
}

void BlockedBloomFilterReader::maybeContains(
    std::span<const std::string_view> keys,
    std::span<bool> out) const {
  NIMBLE_CHECK_GE(out.size(), keys.size());
  const auto* words = reinterpret_cast<const uint32_t*>(data_->as<char>());
  std::array<uint64_t, kPrefetchRunLength> hashes{};
  for (size_t begin = 0; begin < keys.size(); begin += kPrefetchRunLength) {
    const auto runLength = std::min(kPrefetchRunLength, keys.size() - begin);
    // Hash a run and start its loads before probing any of it. Probes are
    // random accesses into a filter that is usually larger than the last-level
    // cache, so overlapping the misses is what makes the batch form faster
    // than the same keys tested one at a time.
    for (size_t i = 0; i < runLength; ++i) {
      hashes[i] = hashKey(keys[begin + i]);
      __builtin_prefetch(words + blockWordOffset(hashes[i], numBlocks_));
    }
    for (size_t i = 0; i < runLength; ++i) {
      out[begin + i] = testHash(words, numBlocks_, hashes[i]);
    }
  }
}

std::unique_ptr<BloomFilterBuilder> BlockedBloomFilterFactory::createBuilder(
    const BloomFilterConfig& config,
    uint64_t numKeys,
    velox::memory::MemoryPool* pool) const {
  const auto& blockedConfig =
      checkedBloomFilterConfig<BlockedBloomFilterConfig>(config);
  return std::make_unique<BlockedBloomFilterBuilder>(
      numKeys, blockedConfig.bitsPerKey, pool);
}

std::unique_ptr<BloomFilterReader> BlockedBloomFilterFactory::createReader(
    std::string_view payload,
    velox::memory::MemoryPool* pool) const {
  return std::make_unique<BlockedBloomFilterReader>(payload, pool);
}

} // namespace facebook::nimble::index
