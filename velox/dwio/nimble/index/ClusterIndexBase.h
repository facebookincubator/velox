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
#include "velox/dwio/nimble/index/KeyComparator.h"
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

/// Shared partition routing, metadata loading, and chunk caching for cluster
/// index implementations.
///
/// The index uses partition keys at the root level for binary search across
/// all partitions, then chunk keys within each partition for finer-grained
/// lookup. When a key stream loader is provided, lookup returns exact row
/// positions by decoding key stream data within chunks. Key chunks live in one
/// blob per partition, or each stripe's key stream is its key chunk, as
/// keyStreamLayout() tells.
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
class ClusterIndexBase : public IndexLookup {
 public:
  ~ClusterIndexBase() override;

  const std::vector<std::string>& indexColumns() const override {
    return indexColumns_;
  }

  LookupResult lookup(const LookupRequest& request) const override;

  /// The rows of one key range and a cursor over their keys.
  struct SeekResult {
    /// The rows whose keys are in the range. Empty on a miss.
    RowRange rows;
    /// Reads the keys of 'rows' in row order, or is null on a miss. Starts
    /// with the decoded key chunk that the lookup found the first row in, and
    /// loads each later chunk when it reaches it. Holds only the chunk that it
    /// reads. The index must outlive it.
    std::unique_ptr<KeyCursor> keyCursor;
  };

  /// Looks up the rows of 'bounds' as lookup() does, and returns them with a
  /// cursor over their keys. Without pinIndex, the index keeps one decoded
  /// chunk per partition for all of its readers, and a reader of another chunk
  /// of the partition can replace it between the lookup and the read of the
  /// keys. The cursor keeps the chunk of the first row, so it does not decode
  /// that chunk again.
  SeekResult seek(const velox::serializer::EncodedKeyBounds& bounds) const;

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

  /// Returns where the index stores its key chunks.
  KeyStreamLayout keyStreamLayout() const {
    return keyStreamLayout_;
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
  /// scans need no coordination beyond that. The index must outlive
  /// them.
  std::unique_ptr<KeyCursor> keyCursor(RowRange rows) const override;

  /// File-level index layout for diagnostic output (nimble_dump, FileLayout).
  struct Layout {
    /// Per-partition detail. Only populated when layout(detail=true).
    struct Partition {
      /// Location and compression of partition metadata in the file.
      MetadataSection metadataSection;
      /// File region of each key stream of the partition: its one key blob in
      /// the Partition layout, and each stripe's key stream, in stripe order,
      /// in the Stripe layout. Only populated when detail=true.
      std::vector<velox::common::Region> keyStreamRegions;
      /// Number of key chunks in this partition: one per stripe in the Stripe
      /// layout.
      uint32_t numChunks{0};
      /// Total rows in this partition.
      uint32_t numRows{0};
      /// Size of the partition metadata FlatBuffer in bytes.
      uint32_t metadataSizeBytes{0};
    };

    std::vector<std::string> indexColumns;
    std::vector<SortOrder> sortOrders;
    /// Where the index stores its key chunks.
    KeyStreamLayout keyStreamLayout{KeyStreamLayout::kPartition};
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

 protected:
  // Walks the index's keys a chunk at a time. Reached only through
  // keyCursor() and seek().
  class KeyIterator;

  // 'keyComparator' is the order of the encoded keys, or null for bytewise
  // order; 'keyReaderFactory' must create readers that seek in the same order.
  // 'keyStreamLocator' locates the key streams of the stripes, and may be
  // empty unless the index stores its key chunks in stripes.
  ClusterIndexBase(
      Section rootSection,
      std::shared_ptr<MetadataInput> metadataInput,
      std::shared_ptr<velox::dwio::common::BufferedInput> dataInput,
      KeyStreamLocator keyStreamLocator,
      KeyReaderFactory keyReaderFactory,
      const KeyComparator* keyComparator,
      bool pinIndex,
      bool preloadIndex,
      velox::memory::MemoryPool* pool);

  // Eagerly loads all partition metadata via a single coalesced metadata IO,
  // then decodes each partition's key-stream chunks via one coalesced data
  // IO per partition. After this returns, subsequent lookups against any
  // partition/chunk hit pure-memory state (assuming pinIndex_=true, which is
  // enforced).
  void preloadIndex();

  // Builds the flat LookupResult from partitionRowRanges. Subclasses override
  // this to apply their own narrowing. The base implementation rejects any
  // lookup extension it is handed, because ignoring one would silently return
  // a wider range than the caller asked for.
  virtual LookupResult buildLookupResult(
      const LookupOptions& options,
      const std::vector<RowRange>& partitionRowRanges) const;

  // Accumulated row counts of a partition's key chunks (prefix sum), so that
  // a subclass can align per-chunk structures of its own to the key chunks.
  // Loads the partition metadata on first access; the returned range stays
  // valid for the lifetime of this index because partitions are never
  // evicted.
  folly::Range<const uint32_t*> partitionChunkRows(uint32_t partitionId) const;

  // Start row of each partition. Size is numPartitions() + 1; the last entry
  // is the file's total row count.
  const std::vector<uint32_t>& partitionRows() const {
    return partitionRows_;
  }

  uint32_t numRows() const {
    return numRows_;
  }

 private:
  // The decoded key chunk that the lower bound of a seek resolved in, which
  // holds the first row of its range. The upper bound of a key is usually in
  // the same chunk, so the lookup decodes it once, and the key cursor of seek()
  // starts with it.
  struct LowerBoundChunk {
    uint32_t partitionId{0};
    uint32_t chunkIndex{0};
    // Null until the lower bound resolves.
    std::shared_ptr<DecodedKeyChunk> data;
  };

  // Cached decoded chunk. The DecodedKeyChunk owns its own backing buffer
  // (decodeKeyChunk appends it to DecodedKeyChunk::stringBuffers when no
  // external reuse buffer is provided), so each cached entry is fully
  // self-contained.
  struct DecodedChunk {
    // Position of the chunk in its partition. Tells which chunk the single
    // scratch slot of an unpinned index holds. Chunk offsets cannot tell it:
    // in the Stripe layout every chunk starts its stripe's key stream.
    uint32_t chunkIndex{0};
    std::shared_ptr<DecodedKeyChunk> data;
  };

 public:
  /// Parsed metadata of a partition, the index data of one stripe group,
  /// cached on demand. ClusterIndexBase.cpp derives a class that reads a
  /// Partition layout partition, whose key chunks are in one blob after its
  /// stripes, and one that reads a Stripe layout partition, whose stripes each
  /// hold one key chunk in their key stream.
  class IndexPartitionBase {
   public:
    virtual ~IndexPartitionBase() = default;

    /// Position of the partition in the index, which is also the index of its
    /// stripe group.
    uint32_t id() const {
      return id_;
    }

    /// The partition metadata.
    const MetadataBuffer& metadata() const {
      return *metadata_;
    }

    /// Where the partition keeps its key chunks.
    virtual KeyStreamLayout layout() const = 0;

    uint32_t numChunks() const {
      return chunkKeys().size();
    }

    /// Returns the last key of each chunk: the stripe keys in the Stripe
    /// layout.
    virtual const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>&
    chunkKeys() const = 0;

    /// Returns the accumulated row counts per chunk (prefix sum,
    /// partition-wide).
    virtual folly::Range<const uint32_t*> chunkRows() const = 0;

    /// Returns the byte offset of a chunk within the key stream that holds it.
    virtual uint32_t chunkOffset(uint32_t chunkIdx) const = 0;

    /// Returns the encoded size of a chunk in bytes.
    virtual uint32_t chunkSize(uint32_t chunkIdx) const = 0;

    /// Returns the index of the key stream that holds a chunk: 0 in the
    /// Partition layout, whose chunks are all in its one key stream, and the
    /// chunk's stripe in the Stripe layout.
    virtual uint32_t keyStreamIndex(uint32_t chunkIdx) const = 0;

    /// Returns the file region of key stream 'keyStreamIdx', without its
    /// stream trailer: the partition's one blob, or the key stream of stripe
    /// 'keyStreamIdx', which is the whole chunk of the stripe.
    virtual velox::common::Region keyStreamRegion(
        uint32_t keyStreamIdx) const = 0;

    /// Returns the chunk index for the given encoded key via binary search
    /// under 'keyComparator', or bytewise when it is null. The key must be
    /// within the partition's chunk range.
    uint32_t chunkIndex(
        std::string_view encodedKey,
        const KeyComparator* keyComparator) const;

    uint32_t rowOffset(uint32_t chunkIdx) const {
      return chunkIdx == 0 ? 0 : chunkRows()[chunkIdx - 1];
    }

    /// Returns the file region of a chunk in either layout: in the Stripe
    /// layout, the key stream of its stripe.
    velox::common::Region chunkStreamRegion(
        const ChunkLocation& chunkLocation) const;

    /// Protects decodedChunks() against concurrent lazy decoding. Shared lock
    /// for cache hits, exclusive lock for cache misses (lazy decode).
    folly::SharedMutex& mutex() const {
      return mutex_;
    }

    /// Lazily-populated decoded chunks. Sized to numChunks() when the index
    /// pins its chunks (each slot indexed by chunkIndex, retained across
    /// lookups), and to 1 otherwise (a single shared scratch slot, evicted on
    /// every different-chunk lookup). Each slot starts empty (data ==
    /// nullptr) and is filled on first access.
    std::vector<DecodedChunk>& decodedChunks() const {
      return decodedChunks_;
    }

   protected:
    IndexPartitionBase(uint32_t id, std::unique_ptr<MetadataBuffer> metadata);

    // Allocates the decoded chunks once the derived class has validated its
    // chunks: one slot per chunk when 'pinIndex', so every chunk stays
    // cached, otherwise a single scratch slot that retains at most one
    // decoded chunk.
    void allocateDecodedChunks(bool pinIndex);

    const uint32_t id_;
    // The partition metadata, which the derived class reads its chunks from.
    const std::unique_ptr<MetadataBuffer> metadata_;

    mutable folly::SharedMutex mutex_;

    mutable std::vector<DecodedChunk> decodedChunks_;
  };

 private:
  // Looks up 'request' as lookup(request) does, and sets 'lowerBoundChunk' to
  // the decoded key chunk of the lower bound, which the upper bound reads when
  // it is in the same chunk. 'lowerBoundChunk' can be null.
  LookupResult lookup(
      const LookupRequest& request,
      LowerBoundChunk* lowerBoundChunk) const;

  // Finds the partition index for a key via binary search on partition keys.
  // Returns nullopt if the key is beyond all partitions.
  std::optional<uint32_t> lookupPartition(std::string_view key) const;

  // Finds the partition containing the given file-level row.
  const IndexPartitionBase* lookupPartition(uint32_t row) const;

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
    // Whether the key is the lower bound of its range, whose row is the first
    // row of the range.
    bool lowerBound;
    uint32_t* targetRow;

    PartitionLookup(
        uint32_t _partitionId,
        std::string_view _encodedKey,
        bool _inclusive,
        bool _lowerBound,
        uint32_t* _targetRow)
        : partitionId{_partitionId},
          encodedKey{_encodedKey},
          inclusive{_inclusive},
          lowerBound{_lowerBound},
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

  // Resolves 'bound' to a file-level row number within 'partition'. The
  // partition's last key must be >= the key of the bound (guaranteed by the
  // caller's binary search on partition keys).
  // @param lowerBoundChunk Is set for a lower bound, and read otherwise, as
  //        seekInChunk() does. Can be null.
  uint32_t resolvePartitionRow(
      const IndexPartitionBase* partition,
      const PartitionLookup& bound,
      LowerBoundChunk* lowerBoundChunk) const;

  // Resolves all partition lookups grouped by partition. Sets
  // 'lowerBoundChunk' when not null, as seekInChunk() does.
  void resolvePartitionBounds(
      std::vector<PartitionLookup>& partitionLookups,
      LowerBoundChunk* lowerBoundChunk) const;

  // Binary searches chunk keys in the partition. The returned ChunkLocation
  // carries the chunkIndex for indexing into per-chunk arrays.
  ChunkLocation lookupChunk(
      const IndexPartitionBase* partition,
      std::string_view encodedKey) const;

  // Finds the chunk containing the given partition-relative row via binary
  // search on chunk_rows.
  ChunkLocation lookupChunk(
      const IndexPartitionBase* partition,
      uint32_t partitionRow) const;

  // Returns the decoded chunk for the given chunk location. When
  // pinIndex_=true, lazily fills the per-chunk slot in
  // IndexPartitionBase::decodedChunks_ (indexed by chunkIndex) and returns a
  // copy.
  // When pinIndex_=false, uses a single shared scratch slot — a hit only
  // when the requested chunk matches the cached chunkIndex, otherwise the
  // slot is evicted and refilled. The returned DecodedChunk owns its
  // underlying buffer via shared_ptr / BufferPtr so it is safe to use past
  // the call.
  DecodedChunk getDecodedChunk(
      const IndexPartitionBase* partition,
      const ChunkLocation& chunkLocation) const;

  // Seeks the key of 'bound' within a chunk's decoded encoding.
  // @param lowerBoundChunk Supplies the decoded chunk when it is that chunk,
  //        and is set to the chunk of a lower bound. Can be null.
  // Returns the partition-level row offset.
  uint32_t seekInChunk(
      const IndexPartitionBase* partition,
      const ChunkLocation& chunkLocation,
      const PartitionLookup& bound,
      LowerBoundChunk* lowerBoundChunk) const;

  // Returns the partition, loading on demand if needed. Never evicted.
  const IndexPartitionBase* loadPartition(uint32_t partitionId) const;

  // Batch-loads the metadata for the given partitionIds via a single
  // coalesced IO. Caller guarantees each id is < numPartitions_ and not
  // already loaded. Used by preloadIndex().
  void loadPartitions(std::span<const uint32_t> partitionIds) const;

  // Loads and decodes all key-stream chunks for a single partition via one
  // coalesced data IO. Populates partition->decodedChunks_. Called per
  // partition from preloadIndex().
  void loadPartitionKeyStream(IndexPartitionBase* partition) const;

  // Constructs the IndexPartition or, in the Stripe layout, the IndexGroup for
  // partitionId from the given metadata buffer and (when pinIndex_=true)
  // pre-sizes its decodedChunks_ vector. Shared by loadPartition() (single IO)
  // and loadPartitions() (batched IO).
  void initializePartition(uint32_t partitionId, MetadataBuffer&& buffer) const;

  const Section rootSection_;
  const serialization::ClusterIndex* const indexRoot_;
  const uint32_t numPartitions_;
  const std::vector<std::string> indexColumns_;
  const std::vector<SortOrder> sortOrders_;
  // Where the index stores its key chunks, from the index root. The same for
  // every partition.
  const KeyStreamLayout keyStreamLayout_;

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
  // Stripe layout only. Locates the key streams of the stripes.
  const KeyStreamLocator keyStreamLocator_;
  // Stripe layout only. Id of the key stream of every stripe.
  const uint32_t keyStreamId_;
  const KeyReaderFactory keyReaderFactory_;

  // Order of the encoded keys, or null for bytewise order.
  const KeyComparator* const keyComparator_;
  velox::memory::MemoryPool* const pool_;

  // When true, every decoded chunk stays pinned in IndexPartitionBase::
  // decodedChunks_. When false, decodedChunks_ holds at most one entry.
  const bool pinIndex_;

  // --- Mutable state ---

  // Protects partitions_ against concurrent lazy loading.
  mutable folly::SharedMutex mutex_;

  // Partitions loaded on demand and never evicted. Accessed by raw pointer.
  mutable std::vector<std::unique_ptr<IndexPartitionBase>> partitions_;

  friend class test::ClusterIndexTestHelper;
};

} // namespace facebook::nimble::index
