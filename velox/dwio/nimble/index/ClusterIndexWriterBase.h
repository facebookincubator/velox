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
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/index/IndexTypes.h"
#include "velox/dwio/nimble/index/IndexWriter.h"
#include "velox/dwio/nimble/index/SortOrder.h"
#include "velox/dwio/nimble/tablet/Chunk.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::nimble::index {

class IndexConfig;
class KeyChunkBuilder;

/// Shared partitioning and serialization for cluster index writers.
///
/// It transforms specified index columns into byte-comparable keys that enable
/// efficient range-based filtering during reads. Key construction, chunking,
/// and partition flushing are all handled internally.
///
/// Lifecycle:
///   1. Create with config and input schema
///   2. For each batch: write(batch) to encode keys (auto-encodes chunks
///      when row count threshold is reached)
///   3. At stripe group boundaries: finishStripeGroup() writes key stream data
///      and partition metadata
///   4. At file close: close() to finalize, serialize, and release resources
///
/// Created with the Stripe KeyStreamLayout, it stores the keys in stripes
/// instead: finishStripe() returns the keys of each stripe as one key chunk
/// for the key stream of that stripe, and finishStripeGroup() writes the last
/// key of each stripe of the stripe group in a ClusterIndexGroup.
class ClusterIndexWriterBase : public IndexWriter {
 public:
  ~ClusterIndexWriterBase() override;

  ClusterIndexWriterBase(const ClusterIndexWriterBase&) = delete;
  ClusterIndexWriterBase& operator=(const ClusterIndexWriterBase&) = delete;
  ClusterIndexWriterBase(ClusterIndexWriterBase&&) = delete;
  ClusterIndexWriterBase& operator=(ClusterIndexWriterBase&&) = delete;

  /// Encodes index keys from the input vector. Internally encodes a chunk
  /// when accumulated rows reach the row count threshold.
  ///
  /// @param input Input vector containing rows to encode.
  void write(const velox::VectorPtr& input) override;

  /// In the Stripe layout, encodes the keys written since the last call into
  /// 'buffer' as the key chunk of the stripe being written, and returns it as
  /// the content of the stripe's key stream. Returns no chunks in the
  /// Partition layout.
  std::vector<Chunk> finishStripe(Buffer& buffer) override;

  /// Writes the key stream data and the partition metadata of the stripe group
  /// in the Partition layout, and the ClusterIndexGroup of the stripe group in
  /// the Stripe layout.
  void finishStripeGroup(
      const WriteDataFn& writeDataFn,
      const CreateMetadataSectionFn& createMetadataFn) override;

  std::optional<IndexDescriptor> close(
      const WriteDataFn& writeDataFn,
      const CreateMetadataSectionFn& createMetadataFn) override;

 protected:
  struct Options {
    std::string indexName;
    std::vector<std::string> columns;
    std::vector<SortOrder> sortOrders;
    uint64_t maxRowsPerKeyChunk;
    CompressionType keyChunkCompressionType;
    /// Where the index stores its key chunks. The Stripe layout stores the keys
    /// of each stripe as one key chunk in stream 'keyStreamId' of that stripe,
    /// so maxRowsPerKeyChunk does not apply, and the index root records
    /// 'keyStreamId' for readers to find the stream in each stripe.
    KeyStreamLayout keyStreamLayout{KeyStreamLayout::kPartition};
    /// Stripe layout only.
    std::optional<uint32_t> keyStreamId{};
  };

  ClusterIndexWriterBase(
      const velox::RowTypePtr& inputType,
      Options options,
      std::unique_ptr<KeyChunkBuilder> keyChunkBuilder,
      velox::memory::MemoryPool* pool);

 private:
  // Chunk in a key stream with key boundaries.
  struct KeyChunk : public Chunk {
    // Last key in the chunk of keys.
    std::string key;
  };

  // Encodes accumulated keys into chunks. When force is false, encoding is
  // skipped if accumulated rows are below maxRowsPerKeyChunk_. When force is
  // true (e.g., at partition flush), all remaining keys are encoded.
  void encodeKeyChunks(bool force = false);

  // Encodes all buffered keys into 'buffer' as chunks of up to
  // 'maxRowsPerChunk' rows, or as one chunk when it is 0, records each chunk
  // in the partition, and returns the chunks. Clears the buffered keys.
  std::vector<Chunk> encodeBufferedKeys(
      uint64_t maxRowsPerChunk,
      Buffer& buffer);

  // Encodes buffered rows [offset, offset + count) into 'buffer' as a
  // KeyChunk.
  void encodeKeyChunk(
      size_t offset,
      uint32_t count,
      Buffer& buffer,
      KeyChunk& chunk);

  // Writes the key stream data and the ClusterIndexPartition of the finished
  // stripe group in the Partition layout.
  void writeIndexPartition(
      const WriteDataFn& writeDataFn,
      const CreateMetadataSectionFn& createMetadataFn);

  // Writes the ClusterIndexGroup of the finished stripe group: the last key of
  // each of its stripes, whose key chunks the stripes already hold.
  void writeIndexGroup(const CreateMetadataSectionFn& createMetadataFn);

  // Records the flushed partition, whose metadata is at 'partitionSection', in
  // the root index, and resets the partition state.
  void finishPartition(MetadataSection partitionSection);

  // Lazily initializes rootIndex_ on first use.
  void ensureIndexRoot();

  // Lazily initializes indexPartition_ on first use.
  void ensureIndexPartition();

  // Holds all partition-scoped state (reset per partition flush).
  struct IndexPartition {
    // Accumulated row counts per chunk (prefix sum).
    std::vector<uint32_t> chunkRows;
    // Byte offset of each chunk within the key stream blob. Not written in
    // the Stripe layout, where each chunk is a whole stripe key stream.
    std::vector<uint32_t> chunkOffsets;
    // Last key of each chunk for binary search (owned copies): the last key of
    // each stripe in the Stripe layout.
    std::vector<std::string> chunkKeys;
    // Encoded key stream segments from all chunks (flattened). Each chunk may
    // produce multiple segments via ChunkedStreamWriter. Concatenated and
    // written to file during finishStripeGroup().
    // Views into encodingBuffer.
    std::vector<std::string_view> encodedChunks;

    // Buffer for encoded chunk stream data within this partition.
    // Backs encodedChunks. Reset after finishStripeGroup() writes the blob
    // to file. Null in the Stripe layout, which encodes into the buffer that
    // finishStripe() is given.
    std::unique_ptr<Buffer> encodingBuffer;
    // Current write offset in the partition's key stream blob. Equals the
    // total encoded bytes so far. Used as the offset for the next chunk. In
    // the Stripe layout, restarts at 0 with each stripe's key stream.
    uint32_t keyStreamOffset{0};

    // File-level offset and size after finishStripeGroup() writes the key
    // data blob.
    uint64_t keyStreamFileOffset{0};
    uint32_t keyStreamFileSize{0};

    bool empty() const {
      return chunkKeys.empty();
    }
  };

  // Holds the root index data for the entire file.
  struct IndexRoot {
    // Metadata sections of the partitions, or of the ClusterIndexGroup of each
    // stripe group in the Stripe layout.
    std::vector<MetadataSection> indexPartitions;
    // Keys for root-level binary search across all partitions.
    // Layout: [firstKey, lastKey0, lastKey1, ..., lastKeyN]
    // firstKey is pushed during the first encodeKeyChunks() call.
    // Each finishStripeGroup() appends a lastKey.
    std::vector<std::string> partitionKeys;
    // Row count per partition, derived from chunkRows.back().
    std::vector<uint32_t> partitionRowCounts;
  };

  const std::string indexName_;
  velox::memory::MemoryPool* const pool_;
  const std::vector<std::string> columns_;
  const std::vector<SortOrder> sortOrders_;
  const std::vector<velox::column_index_t> keyColumnIndices_;
  const uint64_t maxRowsPerKeyChunk_;
  // Chunk-level compression applied to each encoded key chunk (Prefix encoding
  // never compresses its own output).
  const CompressionParams keyCompressionParams_;
  const KeyStreamLayout keyStreamLayout_;
  // Id of the stream of each stripe that holds its key chunk in the Stripe
  // layout. Unset in the Partition layout.
  const std::optional<uint32_t> keyStreamId_;

  // Buffers keys between chunk encodes. Released on close().
  std::unique_ptr<KeyChunkBuilder> keyChunkBuilder_;

  // Partition-level state accumulator.
  std::unique_ptr<IndexPartition> indexPartition_;
  // Root-level index accumulator.
  std::unique_ptr<IndexRoot> indexRoot_;
};

} // namespace facebook::nimble::index
