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

#include "velox/common/base/SplitBlockBloomFilter.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <sstream>

namespace facebook::velox {

namespace {

constexpr uint32_t kBitIndexShift{32 - 5};

// Keep this order aligned with the legacy Nimble hash index layout. It is
// intentionally different from SplitBlockBloomFilter's SIMD salts.
constexpr uint32_t kSalts[ModuloSplitBlockBloomFilter::kNumWords] = {
    0x47b6137bU,
    0x44974d91U,
    0x8824ad5bU,
    0xa2b7289dU,
    0x705495c7U,
    0x2df1424bU,
    0x9efc4947U,
    0x5c6bfb31U,
};

uint32_t wordMask(uint32_t hash, uint32_t wordIndex) {
  const uint32_t bitIndex = (hash * kSalts[wordIndex]) >> kBitIndexShift;
  return uint32_t{1} << bitIndex;
}

} // namespace

int64_t SplitBlockBloomFilter::numBlocks(
    int64_t numElements,
    double falsePositive) {
  constexpr int K = xsimd::batch<uint32_t>::size;
  int64_t numBits = static_cast<int64_t>(std::ceil(
      -static_cast<double>(K) * static_cast<double>(numElements) /
      std::log(1 - std::pow(falsePositive, 1.0 / K))));
  return bits::divRoundUp(numBits, 8 * sizeof(Block));
}

SplitBlockBloomFilter::SplitBlockBloomFilter(const std::span<Block>& blocks)
    : blocks_(blocks) {
  VELOX_CHECK_EQ(reinterpret_cast<uintptr_t>(blocks.data()) % sizeof(Block), 0);
}

std::string SplitBlockBloomFilter::debugString() const {
  std::ostringstream out;
  out << "numBlocks=" << blocks_.size() << '\n';
  int64_t byBlockSetBits[1 + 8 * sizeof(Block)]{};
  int64_t byBitPosition[8 * sizeof(Block)]{};
  for (auto& block : blocks_) {
    int numSetBits = 0;
    for (int i = 0; i < xsimd::batch<uint32_t>::size; ++i) {
      auto n = std::popcount(block.data[i]);
      numSetBits += n;
      for (int j = 0; j < 32; ++j) {
        byBitPosition[j + i * 32] += 1 & (block.data[i] >> j);
      }
    }
    ++byBlockSetBits[numSetBits];
  }
  for (int i = 0; i <= 8 * sizeof(Block); ++i) {
    out << "Block set bits " << i << ": " << byBlockSetBits[i] << '\n';
  }
  for (int i = 0; i < 8 * sizeof(Block); ++i) {
    out << "Bit " << i << ": " << byBitPosition[i] << '\n';
  }
  return out.str();
}

ModuloSplitBlockBloomFilter::ModuloSplitBlockBloomFilter(
    const std::span<Block>& blocks)
    : blocks_(blocks) {
  VELOX_CHECK_GT(blocks_.size(), 0);
}

uint32_t ModuloSplitBlockBloomFilter::numBlocks(
    uint64_t numEntries,
    float bitsPerKey) {
  const uint64_t totalBits = std::max(
      static_cast<uint64_t>(numEntries * bitsPerKey), uint64_t{kBlockSizeBits});
  return std::max(
      static_cast<uint32_t>(bits::divRoundUp(totalBits, kBlockSizeBits)),
      uint32_t{1});
}

void ModuloSplitBlockBloomFilter::insert(uint64_t hash) {
  auto& block = blocks_[blockIndex(hash)];
  const auto key = static_cast<uint32_t>(hash);
  for (uint32_t wordIndex = 0; wordIndex < kNumWords; ++wordIndex) {
    block.data[wordIndex] |= wordMask(key, wordIndex);
  }
}

bool ModuloSplitBlockBloomFilter::mayContain(uint64_t hash) const {
  const auto& block = blocks_[blockIndex(hash)];
  const auto key = static_cast<uint32_t>(hash);
  for (uint32_t wordIndex = 0; wordIndex < kNumWords; ++wordIndex) {
    if ((block.data[wordIndex] & wordMask(key, wordIndex)) == 0) {
      return false;
    }
  }
  return true;
}

} // namespace facebook::velox
