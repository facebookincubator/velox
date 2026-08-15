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

// Adapted from Apache Arrow.

#pragma once

#include <cmath>
#include <cstdint>
#include <memory>

#include "arrow/util/bit_util.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"
#include "velox/dwio/parquet/writer/arrow/tests/Hasher.h"

namespace facebook::velox::parquet::arrow {

// A Bloom filter is a compact structure to indicate whether an item is not in
// a. Set or probably in a set. The Bloom filter usually consists of a bit set
// that. Represents a set of elements, a hash strategy and a Bloom filter
// algorithm.
class PARQUET_EXPORT BloomFilter {
 public:
  // Maximum Bloom filter size, it sets to HDFS default block size 128MB.
  // This value will be reconsidered when implementing Bloom filter producer.
  static constexpr uint32_t kMaximumBloomFilterBytes = 128 * 1024 * 1024;

  /// Determine whether an element exist in set or not.
  ///
  /// @param hash the element to contain.
  /// @return false if value is definitely not in set, and true means PROBABLY.
  /// In set.
  virtual bool findHash(uint64_t hash) const = 0;

  /// Insert element to set represented by Bloom filter bitset.
  /// @param hash the hash of value to insert into Bloom filter.
  virtual void insertHash(uint64_t hash) = 0;

  /// Insert elements to set represented by Bloom filter bitset.
  /// @param hashes the hash values to insert into Bloom filter.
  /// @param num_values the number of hash values to insert.
  virtual void insertHashes(const uint64_t* hashes, int numValues) = 0;

  /// Write this Bloom filter to an output stream. A Bloom filter structure.
  /// Should include bitset length, hash strategy, algorithm, and bitset.
  ///
  /// @param sink the output stream to write.
  virtual void writeTo(ArrowOutputStream* sink) const = 0;

  /// Get the number of bytes of bitset.
  virtual uint32_t getBitsetSize() const = 0;

  /// Compute hash for 32 bits value by using its plain encoding result.
  ///
  /// @param value the value to hash.
  /// @return hash result.
  virtual uint64_t hash(int32_t value) const = 0;

  /// Compute hash for 64 bits value by using its plain encoding result.
  ///
  /// @param value the value to hash.
  /// @return hash result.
  virtual uint64_t hash(int64_t value) const = 0;

  /// Compute hash for float value by using its plain encoding result.
  ///
  /// @param value the value to hash.
  /// @return hash result.
  virtual uint64_t hash(float value) const = 0;

  /// Compute hash for double value by using its plain encoding result.
  ///
  /// @param value the value to hash.
  /// @return hash result.
  virtual uint64_t hash(double value) const = 0;

  /// Compute hash for Int96 value by using its plain encoding result.
  ///
  /// @param value the value to hash.
  /// @return hash result.
  virtual uint64_t hash(const Int96* value) const = 0;

  /// Compute hash for ByteArray value by using its plain encoding result.
  ///
  /// @param value the value to hash.
  /// @return hash result.
  virtual uint64_t hash(const ByteArray* value) const = 0;

  /// Compute hash for fixed byte array value by using its plain encoding.
  /// Result.
  ///
  /// @param value the value address.
  /// @param len the value length.
  /// @return hash result.
  virtual uint64_t hash(const FLBA* value, uint32_t len) const = 0;

  /// Batch compute hashes for 32 bits values by using its plain encoding.
  /// Result.
  ///
  /// @param values values a pointer to the values to hash.
  /// @param num_values the number of values to hash.
  /// @param hashes a pointer to the output hash values, its length should be.
  /// Equal to num_values.
  virtual void hashes(const int32_t* values, int numValues, uint64_t* hashes)
      const = 0;

  /// Batch compute hashes for 64 bits values by using its plain encoding.
  /// Result.
  ///
  /// @param values values a pointer to the values to hash.
  /// @param num_values the number of values to hash.
  /// @param hashes a pointer to the output hash values, its length should be.
  /// Equal to num_values.
  virtual void hashes(const int64_t* values, int numValues, uint64_t* hashes)
      const = 0;

  /// Batch compute hashes for float values by using its plain encoding result.
  ///
  /// @param values values a pointer to the values to hash.
  /// @param num_values the number of values to hash.
  /// @param hashes a pointer to the output hash values, its length should be.
  /// Equal to num_values.
  virtual void hashes(const float* values, int numValues, uint64_t* hashes)
      const = 0;

  /// Batch compute hashes for double values by using its plain encoding result.
  ///
  /// @param values values a pointer to the values to hash.
  /// @param num_values the number of values to hash.
  /// @param hashes a pointer to the output hash values, its length should be.
  /// Equal to num_values.
  virtual void hashes(const double* values, int numValues, uint64_t* hashes)
      const = 0;

  /// Batch compute hashes for Int96 values by using its plain encoding result.
  ///
  /// @param values values a pointer to the values to hash.
  /// @param num_values the number of values to hash.
  /// @param hashes a pointer to the output hash values, its length should be.
  /// Equal to num_values.
  virtual void hashes(const Int96* values, int numValues, uint64_t* hashes)
      const = 0;

  /// Batch compute hashes for ByteArray values by using its plain encoding.
  /// Result.
  ///
  /// @param values values a pointer to the values to hash.
  /// @param num_values the number of values to hash.
  /// @param hashes a pointer to the output hash values, its length should be.
  /// Equal to num_values.
  virtual void hashes(const ByteArray* values, int numValues, uint64_t* hashes)
      const = 0;

  /// Batch compute hashes for fixed byte array values by using its plain.
  /// Encoding result.
  ///
  /// @param values values a pointer to the values to hash.
  /// @param type_len the value length.
  /// @param num_values the number of values to hash.
  /// @param hashes a pointer to the output hash values, its length should be.
  /// Equal to num_values.
  virtual void hashes(
      const FLBA* values,
      uint32_t typeLen,
      int numValues,
      uint64_t* hashes) const = 0;

  virtual ~BloomFilter() = default;

 protected:
  // Hash strategy available for Bloom filter.
  enum class HashStrategy : uint32_t { XXHASH = 0 };

  // Bloom filter algorithm.
  enum class Algorithm : uint32_t { BLOCK = 0 };

  enum class CompressionStrategy : uint32_t { UNCOMPRESSED = 0 };
};

/// The BlockSplitBloomFilter is implemented using block-based Bloom filters.
/// From Putze et al.'s "Cache-,Hash- and Space-Efficient Bloom filters". The.
/// Basic idea is to hash the item to a tiny Bloom filter which size fit a.
/// Single cache line or smaller.
///
/// This implementation sets 8 bits in each tiny Bloom filter. Each tiny Bloom.
/// Filter is 32 bytes to take advantage of 32-byte SIMD instructions.
class PARQUET_EXPORT BlockSplitBloomFilter : public BloomFilter {
 public:
  /// The constructor of BlockSplitBloomFilter. It uses XXH64 as hash function.
  ///
  /// \param pool memory pool to use.
  explicit BlockSplitBloomFilter(
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  /// Initialize the BlockSplitBloomFilter. The range of num_bytes should be.
  /// Within [kMinimumBloomFilterBytes, kMaximumBloomFilterBytes], it will be.
  /// Rounded up/down to lower/upper bound if num_bytes is out of range and
  /// also. Will be rounded up to a power of 2.
  ///
  /// @param num_bytes The number of bytes to store Bloom filter bitset.
  void init(uint32_t numBytes);

  /// Initialize the BlockSplitBloomFilter. It copies the bitset as underlying.
  /// Bitset because the given bitset may not satisfy the 32-byte alignment.
  /// Requirement which may lead to segfault when performing SIMD instructions.
  /// It is the caller's responsibility to free the bitset passed in. This is.
  /// Used when reconstructing a Bloom filter from a parquet file.
  ///
  /// @param bitset The given bitset to initialize the Bloom filter.
  /// @param num_bytes  The number of bytes of given bitset.
  void init(const uint8_t* bitset, uint32_t numBytes);

  /// Minimum Bloom filter size, it sets to 32 bytes to fit a tiny Bloom filter.
  static constexpr uint32_t kMinimumBloomFilterBytes = 32;

  /// Calculate optimal size according to the number of distinct values and.
  /// False positive probability.
  ///
  /// @param ndv The number of distinct values.
  /// @param fpp The false positive probability.
  /// @return it always return a value between kMinimumBloomFilterBytes and.
  /// KMaximumBloomFilterBytes, and the return value is always a power of 2.
  static uint32_t optimalNumOfBytes(uint32_t ndv, double fpp) {
    uint32_t optimalNumBits = optimalNumOfBits(ndv, fpp);
    VELOX_DCHECK(::arrow::bit_util::IsMultipleOf8(optimalNumBits));
    return optimalNumBits >> 3;
  }

  /// Calculate optimal size according to the number of distinct values and.
  /// False positive probability.
  ///
  /// @param ndv The number of distinct values.
  /// @param fpp The false positive probability.
  /// @return it always return a value between kMinimumBloomFilterBytes * 8 and.
  /// KMaximumBloomFilterBytes * 8, and the return value is always a power
  /// of 16.
  static uint32_t optimalNumOfBits(uint32_t ndv, double fpp) {
    VELOX_DCHECK(fpp > 0.0 && fpp < 1.0);
    const double m = -8.0 * ndv / log(1 - pow(fpp, 1.0 / 8));
    uint32_t numBits;

    // Handle overflow.
    if (m < 0 || m > kMaximumBloomFilterBytes << 3) {
      numBits = static_cast<uint32_t>(kMaximumBloomFilterBytes << 3);
    } else {
      numBits = static_cast<uint32_t>(m);
    }

    // Round up to lower bound.
    if (numBits < kMinimumBloomFilterBytes << 3) {
      numBits = kMinimumBloomFilterBytes << 3;
    }

    // Get next power of 2 if bits is not power of 2.
    if ((numBits & (numBits - 1)) != 0) {
      numBits = static_cast<uint32_t>(::arrow::bit_util::NextPower2(numBits));
    }

    // Round down to upper bound.
    if (numBits > kMaximumBloomFilterBytes << 3) {
      numBits = kMaximumBloomFilterBytes << 3;
    }

    return numBits;
  }

  bool findHash(uint64_t hash) const override;
  void insertHash(uint64_t hash) override;
  void insertHashes(const uint64_t* hashes, int numValues) override;
  void writeTo(ArrowOutputStream* sink) const override;
  uint32_t getBitsetSize() const override {
    return numBytes_;
  }

  uint64_t hash(int32_t value) const override {
    return hasher_->hash(value);
  }
  uint64_t hash(int64_t value) const override {
    return hasher_->hash(value);
  }
  uint64_t hash(float value) const override {
    return hasher_->hash(value);
  }
  uint64_t hash(double value) const override {
    return hasher_->hash(value);
  }
  uint64_t hash(const Int96* value) const override {
    return hasher_->hash(value);
  }
  uint64_t hash(const ByteArray* value) const override {
    return hasher_->hash(value);
  }
  uint64_t hash(const FLBA* value, uint32_t len) const override {
    return hasher_->hash(value, len);
  }

  void hashes(const int32_t* values, int numValues, uint64_t* hashes)
      const override {
    hasher_->hashes(values, numValues, hashes);
  }
  void hashes(const int64_t* values, int numValues, uint64_t* hashes)
      const override {
    hasher_->hashes(values, numValues, hashes);
  }
  void hashes(const float* values, int numValues, uint64_t* hashes)
      const override {
    hasher_->hashes(values, numValues, hashes);
  }
  void hashes(const double* values, int numValues, uint64_t* hashes)
      const override {
    hasher_->hashes(values, numValues, hashes);
  }
  void hashes(const Int96* values, int numValues, uint64_t* hashes)
      const override {
    hasher_->hashes(values, numValues, hashes);
  }
  void hashes(const ByteArray* values, int numValues, uint64_t* hashes)
      const override {
    hasher_->hashes(values, numValues, hashes);
  }
  void hashes(
      const FLBA* values,
      uint32_t typeLen,
      int numValues,
      uint64_t* hashes) const override {
    hasher_->hashes(values, typeLen, numValues, hashes);
  }

  uint64_t hash(const int32_t* value) const {
    return hasher_->hash(*value);
  }
  uint64_t hash(const int64_t* value) const {
    return hasher_->hash(*value);
  }
  uint64_t hash(const float* value) const {
    return hasher_->hash(*value);
  }
  uint64_t hash(const double* value) const {
    return hasher_->hash(*value);
  }

  /// Deserialize the Bloom filter from an input stream. It is used when.
  /// Reconstructing a Bloom filter from a parquet filter.
  ///
  /// @param properties The parquet reader properties.
  /// @param input_stream The input stream from which to construct the Bloom.
  /// Filter.
  /// @return The BlockSplitBloomFilter.
  static BlockSplitBloomFilter deserialize(
      const ReaderProperties& properties,
      ArrowInputStream* inputStream);

 private:
  inline void insertHashImpl(uint64_t hash);

  // Bytes in a tiny Bloom filter block.
  static constexpr int kBytesPerFilterBlock = 32;

  // The number of bits to be set in each tiny Bloom filter.
  static constexpr int kBitsSetPerBlock = 8;

  // A mask structure used to set bits in each tiny Bloom filter.
  struct BlockMask {
    uint32_t item[kBitsSetPerBlock];
  };

  // The block-based algorithm needs eight odd SALT values to calculate eight.
  // Indexes of bit to set, one bit in each 32-bit word.
  static constexpr uint32_t SALT[kBitsSetPerBlock] = {
      0x47b6137bU,
      0x44974d91U,
      0x8824ad5bU,
      0xa2b7289dU,
      0x705495c7U,
      0x2df1424bU,
      0x9efc4947U,
      0x5c6bfb31U};

  // Memory pool to allocate aligned buffer for bitset.
  ::arrow::MemoryPool* pool_;

  // The underlying buffer of bitset.
  std::shared_ptr<Buffer> data_;

  // The number of bytes of Bloom filter bitset.
  uint32_t numBytes_;

  // Hash strategy used in this Bloom filter.
  HashStrategy hashStrategy_;

  // Algorithm used in this Bloom filter.
  Algorithm algorithm_;

  // Compression used in this Bloom filter.
  CompressionStrategy compressionStrategy_;

  // The hash pointer points to actual hash class used.
  std::unique_ptr<Hasher> hasher_;
};

} // namespace facebook::velox::parquet::arrow
