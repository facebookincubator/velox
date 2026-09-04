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

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <folly/SharedMutex.h>

#include "velox/buffer/Buffer.h"
#include "velox/common/file/Region.h"
#include "velox/dwio/nimble/index/IndexLookup.h"
#include "velox/dwio/nimble/index/IndexTypes.h"
#include "velox/dwio/nimble/index/KeyChunkDecoder.h"
#include "velox/dwio/nimble/index/SortOrder.h"
#include "velox/dwio/nimble/tablet/ClusterIndexGenerated.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/dwio/nimble/tablet/MetadataCache.h"

namespace facebook::nimble {
class Encoding;
class MetadataInput;
} // namespace facebook::nimble

namespace facebook::velox::dwio::common {
class BufferedInput;
class SeekableInputStream;
} // namespace facebook::velox::dwio::common

namespace facebook::nimble::index {

namespace test {
class ClusterIndexTestHelper;
} // namespace test

/// ClusterIndex is a value index that maps encoded keys to file-level row
/// locations. It provides key-to-row-ID mapping with chunk-level granularity
/// using only FlatBuffer metadata (no I/O for the common path).
///
/// The index uses partition keys at the root level for binary search across
/// all partitions, then chunk keys within each partition for finer-grained
/// lookup. When a key stream loader is provided, lookup returns exact row
/// positions by decoding key stream data within chunks.
///
/// Thread-safety: `lookup()` and `keyAtRow()` are safe for concurrent use
/// from multiple threads. Partition metadata and key-stream chunks are
/// lazily loaded under internal locks (SharedMutex for partitions, per-
/// partition mutex for chunks).
///
/// Example usage:
///   auto index = ClusterIndex::create(std::move(rootSection), ...);
///   auto locations = index->lookup(encodedKey);
///   // locations[0].range is a file-level [startRow, endRow) range
///
///   // With row range constraint (e.g., within a stripe):
///   auto locations = index->lookup(encodedKey,
///       {.rowRange = RowRange(stripeStart, stripeEnd)});
///
class ClusterIndex : public IndexLookup {
 public:
  ~ClusterIndex() override;

  /// @param rootSection Root index section from the file footer.
  /// Creates with Options — index creates its own MetadataInput and data
  /// input internally with index-specific IO stats.
  static std::unique_ptr<ClusterIndex> create(
      Section rootSection,
      velox::memory::MemoryPool* pool,
      const Options& options);

  const std::vector<std::string>& indexColumns() const override {
    return indexColumns_;
  }

  LookupResult lookup(const LookupRequest& request) const override;

  // -- ClusterIndex-specific methods --

  /// Returns the minimum key across all partitions.
  std::string_view minKey() const override {
    return firstKey_;
  }

  /// Returns the maximum key across all partitions.
  std::string_view maxKey() const override {
    return lastKey_;
  }

  /// Returns the encoded key at the given file-level row position.
  /// Requires loadData to have been provided at construction.
  /// Used to compute resume keys when lookup results are truncated.
  std::string keyAtRow(uint32_t row) const override;

  /// Returns the number of partitions in the index.
  uint32_t numPartitions() const {
    return numPartitions_;
  }

  /// Returns the sort orders for each index column.
  const std::vector<SortOrder>& sortOrders() const {
    return sortOrders_;
  }

  /// Returns a cursor over the encoded keys of the rows in 'rows'.
  ///
  /// Scanning advances a chunk cursor instead of binary searching per row,
  /// takes the partition lock once per chunk rather than once per row, and
  /// returns views instead of owned strings. Chunks come from the same
  /// decoded-chunk cache that lookups use, so short cursors walking
  /// consecutive row ranges of one chunk decode it only once between them.
  ///
  /// A cursor holds no locks between calls and is not thread-safe; use one
  /// per thread. Cursors share no state with one another, so concurrent
  /// scans need no coordination beyond that. The ClusterIndex must outlive
  /// them.
  std::unique_ptr<IndexKeyCursor> keyCursor(RowRange rows) const override;

  /// File-level index layout for diagnostic output (nimble_dump, FileLayout).
  struct Layout {
    /// Per-partition detail. Only populated when layout(detail=true).
    struct Partition {
      /// Location and compression of partition metadata in the file.
      MetadataSection metadataSection;
      /// File region containing the encoded key data blob.
      /// Only populated when detail=true.
      velox::common::Region keyStreamRegion;
      /// Number of key chunks in this partition.
      uint32_t numChunks{0};
      /// Total rows in this partition.
      uint32_t numRows{0};
      /// Size of the partition metadata FlatBuffer in bytes.
      uint32_t metadataSizeBytes{0};
    };

    std::vector<std::string> indexColumns;
    std::vector<SortOrder> sortOrders;
    uint32_t numPartitions{0};

    /// Per-partition details. Empty when detail=false.
    std::vector<Partition> partitions;
  };

  /// Returns the metadata section for a specific partition (from root
  /// FlatBuffer, no I/O).
  MetadataSection partitionSection(uint32_t partitionId) const;

  /// Returns the index layout for diagnostic consumers.
  /// @param detail When true, loads per-partition metadata (requires I/O).
  ///        When false, returns only root-level metadata (no I/O).
  Layout layout(bool detail = true) const;

 private:
  // Walks the index's keys a chunk at a time. Reached only through
  // keyCursor(), so callers depend on IndexKeyCursor rather than on this
  // type.
  class KeyIterator final : public IndexKeyCursor {
   public:
    KeyIterator(const ClusterIndex& index, RowRange rows);

    bool hasNext() const override {
      return nextRow_ < endRow_;
    }

    std::string_view next() override;

   private:
    // Loads the chunk holding 'row' and positions the key cursor at it. Runs
    // once per chunk, so the binary searches it performs are amortized over
    // the chunk's rows.
    void openChunkAt(uint32_t row);

    // Index being scanned.
    const ClusterIndex& index_;

    // File-level row the iteration stops at, exclusive.
    const uint32_t endRow_;

    // File-level row that the next next() call reads.
    uint32_t nextRow_;

    // Chunk the key cursor reads. Held so that keys returned from a
    // pre-materialized encoding stay valid while the chunk is current.
    std::shared_ptr<DecodedKeyChunk> chunk_;

    // Reads keys from chunk_. Exhausted exactly at the chunk boundary, which
    // is what drives the transition to the next chunk.
    std::unique_ptr<KeyEncoding::Cursor> cursor_;
  };

  ClusterIndex(
      Section rootSection,
      std::shared_ptr<MetadataInput> metadataInput,
      std::shared_ptr<velox::dwio::common::BufferedInput> dataInput,
      bool pinIndex,
      bool preloadIndex,
      velox::memory::MemoryPool* pool);

  // Eagerly loads all partition metadata via a single coalesced metadata IO,
  // then decodes each partition's key-stream chunks via one coalesced data
  // IO per partition. After this returns, subsequent lookups against any
  // partition/chunk hit pure-memory state (assuming pinIndex_=true, which is
  // enforced).
  void preloadIndex();

  // Cached decoded chunk. The DecodedKeyChunk owns its own backing buffer
  // (decodeKeyChunk appends it to DecodedKeyChunk::stringBuffers when no
  // external reuse buffer is provided), so each cached entry is fully
  // self-contained.
  struct DecodedChunk {
    uint32_t chunkOffset{0};
    std::shared_ptr<DecodedKeyChunk> data;
  };

  // Parsed partition metadata, cached on demand.
  struct IndexPartition {
    const uint32_t id;
    const std::unique_ptr<MetadataBuffer> metadata;
    const serialization::ClusterIndexPartition* const index;

    // Protects decodedChunks against concurrent lazy decoding. Shared lock
    // for cache hits, exclusive lock for cache misses (lazy decode).
    mutable folly::SharedMutex mutex;

    // Lazily-populated decoded chunks. Sized to numChunks() when
    // pinIndex_=true (each slot indexed by chunkIndex, retained across
    // lookups). Sized to 1 when pinIndex_=false (a single shared scratch
    // slot, evicted on every different-chunk lookup). Each slot starts
    // empty (data == nullptr) and is filled on first access.
    mutable std::vector<DecodedChunk> decodedChunks;

    // Constructs the partition and allocates the decoded chunks vector.
    // When pinIndex=true, allocates one slot per chunk for full caching;
    // when false, allocates a single scratch slot that retains at most one
    // decoded chunk.
    IndexPartition(
        uint32_t id,
        std::unique_ptr<MetadataBuffer> metadata,
        bool pinIndex);

    uint32_t numChunks() const {
      return index->chunk_keys()->size();
    }

    // Returns the chunk index for the given encoded key via binary search.
    // The key must be within the partition's chunk range.
    uint32_t chunkIndex(std::string_view encodedKey) const;

    uint32_t chunkOffset(uint32_t chunkIdx) const {
      return index->chunk_offsets()->Get(chunkIdx);
    }

    uint32_t chunkSize(uint32_t chunkIdx) const {
      const uint32_t offset = chunkOffset(chunkIdx);
      const uint32_t nextOffset = chunkIdx + 1 < numChunks()
          ? chunkOffset(chunkIdx + 1)
          : index->key_stream_size();
      return nextOffset - offset;
    }

    uint32_t rowOffset(uint32_t chunkIdx) const {
      return chunkIdx == 0 ? 0 : index->chunk_rows()->Get(chunkIdx - 1);
    }

    velox::common::Region chunkStreamRegion(
        const ChunkLocation& chunkLocation) const;
  };

  // Resolves an encoded key to a file-level row number within a partition.
  // The partition's last key must be >= encodedKey (guaranteed by the caller's
  // binary search on partition keys).
  // @param inclusive true for first row >= key, false for first row > key.
  uint32_t resolvePartitionRow(
      const IndexPartition* partition,
      std::string_view encodedKey,
      bool inclusive) const;

  // Finds the partition index for a key via binary search on partition keys.
  // Returns nullopt if the key is beyond all partitions.
  std::optional<uint32_t> lookupPartition(std::string_view key) const;

  // Finds the partition containing the given file-level row.
  const IndexPartition* lookupPartition(uint32_t row) const;

  // Converts a file-level row to a partition-relative row number.
  uint32_t partitionRow(uint32_t partitionId, uint32_t row) const;

  // A bound to resolve within a partition.
  // Stores a pointer to the target row field (startRow or endRow) in the
  // caller's RowRange. resolvePartitionBounds writes the resolved file-level
  // row directly through this pointer.
  struct PartitionLookup {
    uint32_t partitionId;
    std::string_view encodedKey;
    // true for lower bound (first row >= key), false for upper bound
    // (first row > key).
    bool inclusive;
    uint32_t* targetRow;

    PartitionLookup(
        uint32_t _partitionId,
        std::string_view _encodedKey,
        bool _inclusive,
        uint32_t* _targetRow)
        : partitionId{_partitionId},
          encodedKey{_encodedKey},
          inclusive{_inclusive},
          targetRow{_targetRow} {
      NIMBLE_CHECK_NOT_NULL(targetRow);
    }
  };

  // Builds partition lookups from request into partitionLookups
  // and initializes partitionRowRanges.
  void lookupPartitions(
      const LookupRequest& request,
      std::vector<PartitionLookup>& partitionLookups,
      std::vector<RowRange>& partitionRowRanges) const;
  void partitionPointLookups(
      const LookupRequest& request,
      std::vector<PartitionLookup>& partitionLookups,
      std::vector<RowRange>& partitionRowRanges) const;
  void partitionRangeLookups(
      const LookupRequest& request,
      std::vector<PartitionLookup>& partitionLookups,
      std::vector<RowRange>& partitionRowRanges) const;

  // Resolves all partition lookups grouped by partition.
  void resolvePartitionBounds(
      std::vector<PartitionLookup>& partitionLookups) const;

  // Builds the flat LookupResult from partitionRowRanges.
  LookupResult buildLookupResult(
      const LookupOptions& options,
      const std::vector<RowRange>& partitionRowRanges) const;

  // Binary searches chunk keys in the partition. The returned ChunkLocation
  // carries the chunkIndex for indexing into per-chunk arrays.
  ChunkLocation lookupChunk(
      const IndexPartition* partition,
      std::string_view encodedKey) const;

  // Finds the chunk containing the given partition-relative row via binary
  // search on chunk_rows.
  ChunkLocation lookupChunk(
      const IndexPartition* partition,
      uint32_t partitionRow) const;

  // Returns the decoded chunk for the given chunk location. When
  // pinIndex_=true, lazily fills the per-chunk slot in
  // IndexPartition::decodedChunks (indexed by chunkIndex) and returns a copy.
  // When pinIndex_=false, uses a single shared scratch slot — a hit only
  // when the requested chunk matches the cached chunkOffset, otherwise the
  // slot is evicted and refilled. The returned DecodedChunk owns its
  // underlying buffer via shared_ptr / BufferPtr so it is safe to use past
  // the call.
  DecodedChunk getDecodedChunk(
      const IndexPartition* partition,
      const ChunkLocation& chunkLocation) const;

  // Seeks within a chunk's decoded encoding.
  // @param inclusive true: first key >= encodedKey. false: first key >
  //        encodedKey.
  // Returns the partition-level row offset.
  uint32_t seekInChunk(
      const IndexPartition* partition,
      const ChunkLocation& chunkLocation,
      std::string_view encodedKey,
      bool inclusive) const;

  // Returns the partition, loading on demand if needed. Never evicted.
  const IndexPartition* loadPartition(uint32_t partitionId) const;

  // Batch-loads the metadata for the given partitionIds via a single
  // coalesced IO. Caller guarantees each id is < numPartitions_ and not
  // already loaded. Used by preloadIndex().
  void loadPartitions(std::span<const uint32_t> partitionIds) const;

  // Loads and decodes all key-stream chunks for a single partition via one
  // coalesced data IO. Populates partition->decodedChunks. Called per
  // partition from preloadIndex().
  void loadPartitionKeyStream(IndexPartition* partition) const;

  // Constructs IndexPartition for partitionId from the given metadata buffer
  // and (when pinIndex_=true) pre-sizes its decodedChunks vector. Shared by
  // loadPartition() (single IO) and loadPartitions() (batched IO).
  void initializePartition(uint32_t partitionId, MetadataBuffer&& buffer) const;

  const Section rootSection_;
  const serialization::ClusterIndex* const indexRoot_;
  const uint32_t numPartitions_;
  const std::vector<std::string> indexColumns_;
  const std::vector<SortOrder> sortOrders_;

  // File-level key range for early pruning.
  const std::string_view firstKey_;
  const std::string_view lastKey_;

  // Last key of each partition for binary search.
  // partitionKeys_[i] is the last key of partition i.
  const std::vector<std::string_view> partitionKeys_;

  // Cumulative row offsets per partition (prefix sum).
  // partitionRows_[i] is the start row of partition i.
  // partitionRows_[numPartitions_] == numRows_.
  // Size = numPartitions_ + 1.
  const std::vector<uint32_t> partitionRows_;

  // Total number of rows in the file.
  const uint32_t numRows_;

  const std::shared_ptr<MetadataInput> metadataInput_;
  const std::shared_ptr<velox::dwio::common::BufferedInput> dataInput_;
  velox::memory::MemoryPool* const pool_;

  // When true, every decoded chunk stays pinned in IndexPartition::
  // decodedChunks. When false, decodedChunks holds at most one entry.
  const bool pinIndex_;

  // --- Mutable state ---

  // Protects partitions_ against concurrent lazy loading.
  mutable folly::SharedMutex mutex_;

  // Partitions loaded on demand and never evicted. Accessed by raw pointer.
  mutable std::vector<std::unique_ptr<IndexPartition>> partitions_;

  friend class test::ClusterIndexTestHelper;
};

} // namespace facebook::nimble::index
