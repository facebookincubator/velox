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

#include "velox/common/base/SimdUtil.h"

#include <cstdint>
#include <span>

namespace facebook::velox {

/// SIMDized bloom filter implementation.  We take 8 or 4 (depending on the SIMD
/// register size) bits from the inserted value, and split them into a block.  A
/// block is the same size of a SIMD register.
///
/// This data structure is not responsible for memory management or hashing.
///
/// A detailed explanation about how the data structure works can be found here:
/// https://parquet.apache.org/docs/file-format/bloomfilter/
class SplitBlockBloomFilter {
 public:
  /// A block is basically a SIMD register.  Made public so user can calculate
  /// the size needed for memory allocation; otherwise it's implementation
  /// detail.
  struct alignas(sizeof(xsimd::batch<uint32_t>)) Block {
    uint32_t data[xsimd::batch<uint32_t>::size];
  };

  /// Calculate the number of blocks needed to satisfy certain number of inserts
  /// and false positive rate.  A rough estimation between the memory usage and
  /// false positive rate can be found below:
  ///
  /// | Bits of space per insert | False positive probability |
  /// |--------------------------|----------------------------|
  /// |                      6.0 |                        10% |
  /// |                     10.5 |                         1% |
  /// |                     16.9 |                       0.1% |
  /// |                     26.4 |                      0.01% |
  /// |                       41 |                     0.001% |
  static int64_t numBlocks(int64_t numElements, double falsePositive);

  /// Construct the bloom filter using the blocks memory passed as parameter.
  /// The block memory address must be aligned with block size (i.e. SIMD
  /// register size).  It is recommended to use numBlocks() to calculate the
  /// number of block for allocation.
  explicit SplitBlockBloomFilter(const std::span<Block>& blocks);

  /// Delete copy constructor and assignment to avoid accidentally mutating the
  /// same underlying block data from multiple places.
  SplitBlockBloomFilter(const SplitBlockBloomFilter&) = delete;
  SplitBlockBloomFilter& operator=(const SplitBlockBloomFilter&) = delete;

  SplitBlockBloomFilter(SplitBlockBloomFilter&&) = default;

  /// Insert a hash into the bloom filter.  The function used to generate this
  /// hash should be avalanching.
  void insert(uint64_t hash) {
    auto mask = makeMask(hash);
    auto* block = blocks_[blockIndex(hash)].data;
    (xsimd::load_aligned(block) | mask).store_aligned(block);
  }

  /// Check whether a hash has been inserted before.  Could return true when it
  /// has not been inserted.  Never return false when the hash has been
  /// inserted.
  bool mayContain(uint64_t hash) const {
    auto mask = makeMask(hash);
    auto block = xsimd::load_aligned(blocks_[blockIndex(hash)].data);
#if XSIMD_WITH_AVX
    return _mm256_testc_si256(block, mask);
#else
    return xsimd::all(xsimd::bitwise_andnot(mask, block) == 0);
#endif
  }

  /// Return the block index of the given hash.
  uint64_t blockIndex(uint64_t hash) const {
    return ((hash >> 32) * blocks_.size()) >> 32;
  }

  std::string debugString() const;

 private:
  static_assert(64 % sizeof(Block) == 0);

  template <typename A = xsimd::default_arch>
  static xsimd::batch<uint32_t, A> makeSaltsVec() {
    constexpr uint32_t kSalts[] = {
        0x2df1424bU,
        0x44974d91U,
        0x47b6137bU,
        0x5c6bfb31U,
        0x705495c7U,
        0x8824ad5bU,
        0x9efc4947U,
        0xa2b7289dU,
    };
    if constexpr (xsimd::batch<uint32_t, A>::size == 8) {
      return xsimd::batch<uint32_t, A>(
          kSalts[0],
          kSalts[1],
          kSalts[2],
          kSalts[3],
          kSalts[4],
          kSalts[5],
          kSalts[6],
          kSalts[7]);
    } else {
      static_assert(xsimd::batch<uint32_t, A>::size == 4);
      return xsimd::batch<uint32_t, A>(
          kSalts[0], kSalts[2], kSalts[4], kSalts[6]);
    }
  }

  static xsimd::batch<uint32_t> makeMask(uint32_t hash) {
    auto shifts = (makeSaltsVec() * xsimd::broadcast(hash)) >> 27;
    return xsimd::broadcast<uint32_t>(1) << shifts;
  }

  std::span<Block> const blocks_;
};

} // namespace facebook::velox
