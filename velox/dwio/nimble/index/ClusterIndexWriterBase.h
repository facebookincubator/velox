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
///   3. At stripe group boundaries: flush() writes key stream data
///      and partition metadata
///   4. At file close: close() to finalize, serialize, and release resources
///
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

  /// Flushes index data at stripe group boundary.
  void flush(
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

  // Encodes buffered rows [offset, offset + count) into a KeyChunk.
  void encodeKeyChunk(size_t offset, uint32_t count, KeyChunk& chunk);

  // Lazily initializes rootIndex_ on first use.
  void ensureIndexRoot();

  // Lazily initializes indexPartition_ on first use.
  void ensureIndexPartition();

  // Holds all partition-scoped state (reset per partition flush).
  struct IndexPartition {
    // Accumulated row counts per chunk (prefix sum).
    std::vector<uint32_t> chunkRows;
    // Byte offset of each chunk within the key stream blob.
    std::vector<uint32_t> chunkOffsets;
    // Last key of each chunk for binary search (owned copies).
    std::vector<std::string> chunkKeys;
    // Encoded key stream segments from all chunks (flattened). Each chunk may
    // produce multiple segments via ChunkedStreamWriter. Concatenated and
    // written to file during flush().
    // Views into encodingBuffer.
    std::vector<std::string_view> encodedChunks;

    // Buffer for encoded chunk stream data within this partition.
    // Backs encodedChunks. Reset after flush() writes the blob
    // to file.
    std::unique_ptr<Buffer> encodingBuffer;
    // Current write offset in the partition's key stream blob. Equals the
    // total encoded bytes so far. Used as the offset for the next chunk.
    uint32_t keyStreamOffset{0};

    // File-level offset and size after flush() writes the key
    // data blob.
    uint64_t keyStreamFileOffset{0};
    uint32_t keyStreamFileSize{0};

    bool empty() const {
      return chunkKeys.empty();
    }
  };

  // Holds the root index data for the entire file.
  struct IndexRoot {
    std::vector<MetadataSection> indexPartitions;
    // Keys for root-level binary search across all partitions.
    // Layout: [firstKey, lastKey0, lastKey1, ..., lastKeyN]
    // firstKey is pushed during the first encodeKeyChunks() call.
    // Each flush() appends a lastKey.
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

  // Buffers keys between chunk encodes. Released on close().
  std::unique_ptr<KeyChunkBuilder> keyChunkBuilder_;

  // Partition-level state accumulator.
  std::unique_ptr<IndexPartition> indexPartition_;
  // Root-level index accumulator.
  std::unique_ptr<IndexRoot> indexRoot_;
};

} // namespace facebook::nimble::index
