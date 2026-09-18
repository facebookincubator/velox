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

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "flatbuffers/flatbuffers.h"
#include "velox/dwio/nimble/index/BloomFilter.h"
#include "velox/dwio/nimble/index/IndexLookup.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/dwio/nimble/tablet/MetadataCache.h"

namespace facebook::nimble::serialization {
struct HashIndex;
} // namespace facebook::nimble::serialization

namespace facebook::nimble::index {

/// A hash index for point lookups on unsorted data.
///
/// Each HashIndex covers one set of composite key columns and is an
/// independent object (analogous to ClusterIndex). Implements the IndexLookup
/// interface for unified index dispatch.
///
/// Lookup flow:
///   1. Optional bloom filter check for fast negative result
///   2. Hash(key) mod numBuckets → bucket
///   3. Load the partition containing the bucket (on-demand, cached)
///   4. Scan entries in bucket by exact encoded key match
///   5. Return file-level row locations
///
/// Example usage:
///   auto* hashIndex = registry->findIndex({"col_a", "col_b"});
///   auto results = hashIndex->lookup(key);
///   for (const auto& loc : results) {
///     // loc.range is a file-level [startRow, endRow) range
///   }
class HashIndex : public IndexLookup {
 public:
  // Total number of individual key lookups performed.
  static constexpr std::string_view kNumLookups{"hashIndex.numLookups"};
  // Number of key lookups skipped by the bloom filter (fast negatives).
  static constexpr std::string_view kNumBloomFilterSkips{
      "hashIndex.numBloomFilterSkips"};
  // Number of row ranges returned across all lookups.
  static constexpr std::string_view kNumMatchedRows{"hashIndex.numMatchedRows"};
  // Number of key lookups skipped by the min/max key check.
  static constexpr std::string_view kNumMinMaxSkips{"hashIndex.numMinMaxSkips"};

  const std::vector<std::string>& indexColumns() const override {
    return columns_;
  }

  LookupResult lookup(const LookupRequest& request) const override;

  std::string_view minKey() const override;

  std::string_view maxKey() const override;

  folly::F14FastMap<std::string, velox::RuntimeMetric> stats() const override;

  static std::unique_ptr<HashIndex> create(
      std::vector<std::string> columns,
      std::unique_ptr<MetadataBuffer> indexMetadata,
      std::shared_ptr<MetadataInput> metadataInput,
      velox::memory::MemoryPool* pool);

 private:
  HashIndex(
      std::vector<std::string> columns,
      std::unique_ptr<MetadataBuffer> indexMetadata,
      std::shared_ptr<MetadataInput> metadataInput,
      velox::memory::MemoryPool* pool);

  // Loaded partition data. Raw pointers reference the underlying FlatBuffer
  // data owned by the MetadataBuffer.
  class Partition {
   public:
    static std::shared_ptr<Partition> create(
        std::unique_ptr<MetadataBuffer> metadata);

    // Returns file row numbers matching the encoded key in the given local
    // bucket.
    std::vector<uint32_t> findRows(uint32_t bucketIndex, std::string_view key)
        const;

   private:
    Partition(
        std::unique_ptr<MetadataBuffer> metadata,
        const uint32_t* bucketOffsets,
        const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>*
            keys,
        const uint32_t* rows);

    const std::unique_ptr<MetadataBuffer> metadata_;
    // Size = bucketCount + 1. Bucket i spans [offsets[i], offsets[i+1]).
    const uint32_t* const bucketOffsets_;
    const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>* const
        keys_;
    const uint32_t* const rows_;
  };

  // Partition section info parsed from the parallel partition arrays.
  struct PartitionDescriptor {
    uint32_t startBucket{};
    uint32_t bucketCount{};
    MetadataSection section;

    std::string toString() const {
      return fmt::format(
          "startBucket: {}, bucketCount: {}", startBucket, bucketCount);
    }
  };

  // Parses partition descriptors from the serialized index metadata.
  static std::vector<PartitionDescriptor> buildPartitionDescriptors(
      const MetadataBuffer& metadata,
      uint32_t numBuckets);

  // Finds the partition index for a given global bucket number.
  uint32_t findPartitionIndex(uint32_t bucket) const;

  const std::vector<std::string> columns_;
  // Owns the loaded FlatBuffer data for this index.
  const std::unique_ptr<MetadataBuffer> indexMetadata_;
  velox::memory::MemoryPool* const pool_;

  const uint32_t numBuckets_;
  const uint32_t bucketMask_;
  const std::string_view minKey_;
  const std::string_view maxKey_;

  // Constructed from FlatBuffer data during initialization.
  const std::unique_ptr<BloomFilter> bloomFilter_;

  // Partition descriptors and lazily loaded partition data.
  // Always has at least one partition.
  const std::vector<PartitionDescriptor> partitions_;
  const std::shared_ptr<MetadataInput> metadataInput_;
  mutable MetadataCache<uint32_t, Partition> partitionCache_;

  mutable std::atomic_uint64_t numLookups_{0};
  mutable std::atomic_uint64_t numMinMaxSkips_{0};
  mutable std::atomic_uint64_t numBloomFilterSkips_{0};
  mutable std::atomic_uint64_t numMatchedRows_{0};
};

} // namespace facebook::nimble::index
