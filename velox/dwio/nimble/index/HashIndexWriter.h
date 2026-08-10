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

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/index/IndexKeyEncoder.h"
#include "velox/dwio/nimble/index/IndexWriter.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::nimble::serialization {
struct BloomFilter;
} // namespace facebook::nimble::serialization

namespace facebook::nimble::index {

class IndexConfig;

namespace test {
class HashIndexWriterTestHelper;
} // namespace test

/// Builds one or more hash indices during file write.
///
/// The HashIndexWriter accumulates encoded keys and their file row numbers
/// during writing, then builds hash tables and optional bloom filters at
/// finalization (file close) time. The result is serialized as a FlatBuffer
/// and stored as an optional section in the Nimble file.
///
/// Lifecycle:
///   1. Create with configs and input schema
///   2. For each batch: write(batch) to accumulate keys and row mappings
///   3. At file close: finalize() to build and serialize the index
///
/// Serialization layout:
///
///   HashIndexDirectory (optional section, one per file)
///   +-- HashIndexSection[0] --> HashIndex (separate section)
///   |     +-- num_buckets, load_factor, bloom_filter
///   |     +-- stripe_row_counts[]
///   |     +-- partitions[]
///   |           +-- Partition 0 (separate section)
///   |           |     bucket_offsets:  [0, 2, 2, 3]
///   |           |     bucket_counts:   [2, 0, 1, 1]
///   |           |     encoded_keys:    [k0, k1, k2, k3]
///   |           |     row_numbers:     [10, 42, 7, 99]
///   |           +-- Partition 1 (loaded on-demand)
///   |                 ...
///   +-- HashIndexSection[1] --> HashIndex (second index)
///         ...
///
///   Lookup: hash(key) & (num_buckets-1) --> bucket --> partition
///     --> scan encoded_keys[offset..offset+count] for exact match
///     --> return row_numbers[matched_offset]
///
/// Memory: Keys are accumulated in memory during writing. For files with
/// billions of rows, consider the memory impact (~40-60 bytes per row per
/// index for encoded keys + metadata).
class HashIndexWriter : public IndexWriter {
 public:
  /// Factory method. Returns nullptr if configs is empty.
  ///
  /// @param configs Hash index configurations. One index per config.
  /// @param inputType Schema of the input data.
  /// @param pool Memory pool for allocations.
  static std::unique_ptr<HashIndexWriter> create(
      std::span<const IndexConfig*> configs,
      const velox::TypePtr& inputType,
      velox::memory::MemoryPool* pool);

  ~HashIndexWriter() override = default;

  HashIndexWriter(const HashIndexWriter&) = delete;
  HashIndexWriter& operator=(const HashIndexWriter&) = delete;
  HashIndexWriter(HashIndexWriter&&) = delete;
  HashIndexWriter& operator=(HashIndexWriter&&) = delete;

  /// Encodes keys from the input batch and records row mappings.
  void write(const velox::VectorPtr& input) override;

  std::optional<IndexDescriptor> close(
      const WriteDataFn& writeDataFn,
      const CreateMetadataSectionFn& createMetadataFn) override;

 private:
  struct Options {
    std::vector<std::string> columns;
    float loadFactor;
    std::optional<float> bloomFilterBitsPerKey;
    uint64_t maxPartitionSizeBytes;
  };

  static Options makeOptions(const IndexConfig& config);
  static std::vector<std::vector<std::string>> extractColumnSets(
      const std::vector<Options>& options);

  HashIndexWriter(
      std::string indexName,
      const velox::RowTypePtr& inputType,
      std::vector<Options> options,
      velox::memory::MemoryPool* pool);

  // An indexed key and its file row.
  // The key is a string_view into the encodingBuffer_ arena which
  // guarantees pointer stability for its lifetime.
  struct IndexEntry {
    std::string_view key;
    uint32_t row;
  };

  // Per-index accumulator that collects keys and builds the hash table.
  struct IndexAccumulator {
    Options options;
    std::unique_ptr<IndexKeyEncoder> encoder;
    // Accumulated index entries with encoded keys pointing into
    // encodingBuffer_.
    std::vector<IndexEntry> entries;

    void clear() {
      entries.clear();
      entries.shrink_to_fit();
      encoder.reset();
    }
  };

  // Estimates the serialized size of one bucket's keys.
  // Accounts for encoded key strings + row numbers + bucket directory overhead.
  static size_t estimateBucketSize(
      const std::vector<uint32_t>& keyIndices,
      const IndexAccumulator& accumulator);

  // Builds a HashIndexPartition FlatBuffer for a range of buckets.
  static void buildIndexPartitionFlatBuffer(
      flatbuffers::FlatBufferBuilder& builder,
      const std::vector<std::vector<uint32_t>>& keyIndices,
      const IndexAccumulator& accumulator,
      uint32_t startBucket,
      uint32_t bucketCount);

  // Builds a bloom filter FlatBuffer offset, or 0 if bloom filter is disabled.
  flatbuffers::Offset<serialization::BloomFilter> buildBloomFilter(
      flatbuffers::FlatBufferBuilder& builder,
      const IndexAccumulator& accumulator) const;

  // Builds a single HashIndex FlatBuffer for one accumulator.
  // When partitioning is enabled, partition data sections are written via
  // createMetadataFn and the entry contains partition descriptors.
  void buildIndexFlatBuffer(
      flatbuffers::FlatBufferBuilder& builder,
      const IndexAccumulator& accumulator,
      const CreateMetadataSectionFn& createMetadataFn) const;

  // Builds the HashIndexDirectory FlatBuffer with HashIndexSection entries.
  void buildIndexDirectoryFlatBuffer(
      flatbuffers::FlatBufferBuilder& builder,
      const std::vector<MetadataSection>& indexSections) const;

  // Returns a string_view of the FlatBufferBuilder's output buffer.
  static std::string_view asStringView(
      const flatbuffers::FlatBufferBuilder& builder) {
    return {
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize()};
  }

  inline void ensureEncodingBuffer() {
    if (encodingBuffer_ == nullptr) {
      encodingBuffer_ = std::make_unique<Buffer>(*pool_);
    }
  }

  const std::string indexName_;
  velox::memory::MemoryPool* const pool_;
  // Deduplicated key column indices across all hash indices for null
  // validation.
  const std::vector<velox::column_index_t> keyColumnIndices_;
  // Arena buffer for encoded keys. Lazily created on first write().
  // Encoded keys (string_views) in IndexEntry point into this buffer.
  std::unique_ptr<Buffer> encodingBuffer_;
  std::vector<IndexAccumulator> accumulators_;
  uint32_t numRows_{0};

  friend class test::HashIndexWriterTestHelper;
};

} // namespace facebook::nimble::index
