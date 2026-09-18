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
#include <string_view>

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"

namespace facebook::nimble::index {

/// Split block bloom filter following the Parquet bloom filter design.
///
/// Uses 256-bit (32-byte) blocks with multiple hash probes per block.
/// Each lookup touches exactly one block, making it cache-friendly.
/// The filter uses xxHash64 to generate hash values.
///
/// Design reference: "Cache-, Hash- and Space-Efficient Bloom Filters"
/// (Putze, Sanders, Singler, 2007) and Apache Parquet format spec.
class BloomFilter {
 public:
  /// Number of hash probes per block. Each probe sets/tests one bit in a
  /// separate uint32_t word, so the block has exactly this many words.
  static constexpr uint32_t kNumProbesPerBlock{8};
  /// Bytes per word used in each block.
  static constexpr uint32_t kBytesPerWord{sizeof(uint32_t)};
  /// Bits per word.
  static constexpr uint32_t kBitsPerWord{kBytesPerWord * 8};
  /// Block size in bytes (one word per probe).
  static constexpr uint32_t kBlockSizeBytes{kNumProbesPerBlock * kBytesPerWord};
  /// Block size in bits.
  static constexpr uint32_t kBlockSizeBits{kNumProbesPerBlock * kBitsPerWord};

  /// Constructs a bloom filter sized for the expected number of entries.
  ///
  /// @param numEntries Expected number of distinct entries.
  /// @param bitsPerKey Target bits per key (controls false positive rate).
  /// @param pool Memory pool for filter data allocation.
  BloomFilter(
      uint64_t numEntries,
      float bitsPerKey,
      velox::memory::MemoryPool* pool);

  /// Constructs a bloom filter from pre-existing data (for reading).
  ///
  /// @param numBlocks Number of 256-bit blocks.
  /// @param data Raw filter data. Must be numBlocks * kBlockSizeBytes bytes.
  /// @param pool Memory pool for filter data allocation.
  BloomFilter(
      uint32_t numBlocks,
      const uint8_t* data,
      size_t dataSize,
      velox::memory::MemoryPool* pool);

  /// Inserts a key into the filter.
  void insert(std::string_view key);

  /// Tests whether a key might be in the filter.
  /// Returns false if the key is definitely not present (no false negatives).
  /// Returns true if the key might be present (possible false positives).
  bool testKey(std::string_view key) const;

  /// Returns the number of 256-bit blocks.
  uint32_t numBlocks() const {
    return numBlocks_;
  }

  /// Returns the raw filter data.
  const uint8_t* data() const {
    return data_->as<uint8_t>();
  }

  /// Returns the size of the raw filter data in bytes.
  size_t dataSize() const {
    return data_->size();
  }

  /// Returns the bits per key used during construction.
  float bitsPerKey() const {
    return bitsPerKey_;
  }

 private:
  // Right-shift to select a bit position within a word (top 5 bits of the
  // salted 32-bit hash select one of 32 bit positions).
  static constexpr uint32_t kBitShift{32 - 5};

  // Salts for generating multiple hash probes from a single hash value.
  // These are odd constants that provide good bit mixing when multiplied
  // with the hash value.
  static constexpr uint32_t kSalts[kNumProbesPerBlock] = {
      0x47b6137bU,
      0x44974d91U,
      0x8824ad5bU,
      0xa2b7289dU,
      0x705495c7U,
      0x2df1424bU,
      0x9efc4947U,
      0x5c6bfb31U,
  };

  // Maps hash to block index using the upper 32 bits.
  uint32_t blockIndex(uint64_t hash) const {
    return static_cast<uint32_t>((hash >> 32) % numBlocks_);
  }

  // Accesses the block words at the given block index.
  uint32_t* mutableBlock(uint32_t index) {
    return reinterpret_cast<uint32_t*>(
        data_->asMutable<uint8_t>() +
        static_cast<size_t>(index) * kBlockSizeBytes);
  }

  const uint32_t* block(uint32_t index) const {
    return reinterpret_cast<const uint32_t*>(
        data_->as<uint8_t>() + static_cast<size_t>(index) * kBlockSizeBytes);
  }

  void insertHash(uint64_t hash);
  bool testHash(uint64_t hash) const;

  static uint64_t hashKey(std::string_view key);

  const uint32_t numBlocks_;
  const float bitsPerKey_;
  velox::BufferPtr data_;
};

} // namespace facebook::nimble::index
