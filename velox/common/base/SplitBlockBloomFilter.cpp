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

#include "velox/common/base/SplitBlockBloomFilter.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"

#include <bit>
#include <sstream>

namespace facebook::velox {

int64_t SplitBlockBloomFilter::numBlocks(
    int64_t numElements,
    double falsePositive) {
  constexpr int K = xsimd::batch<uint32_t>::size;
  int64_t numBits = std::ceil(
      -K * numElements / std::log(1 - std::pow(falsePositive, 1.0 / K)));
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

} // namespace facebook::velox
