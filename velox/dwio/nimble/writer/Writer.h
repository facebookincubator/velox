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

#include <string_view>

#include "velox/buffer/BufferPool.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/file/File.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/common/Writer.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/index/IndexWriter.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"
#include "velox/dwio/nimble/velox/FieldWriter.h"
#include "velox/dwio/nimble/velox/SharedDictionaryWriter.h"
#include "velox/dwio/nimble/writer/BufferPolicy.h"
#include "velox/dwio/nimble/writer/NimbleFileMetadata.h"
#include "velox/dwio/nimble/writer/WriterOptions.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/DecodedVector.h"

/// The Writer takes a velox VectorPtr and writes it to an Nimble file
/// format.

namespace facebook::nimble {

namespace detail {

class WriterContext;

} // namespace detail

// Drives the writer's reclaim path during memory arbitration. Defined once per
// build flavour, since the internal one derives from
// velox::exec::MemoryReclaimer and velox/exec is not part of the OSS build.
class WriterMemoryReclaimer;

/// Writer that takes velox vector as input and produces nimble file.
class Writer : public velox::dwio::common::Writer {
 public:
  Writer(
      const velox::TypePtr& type,
      std::unique_ptr<velox::WriteFile> file,
      velox::memory::MemoryPool& pool,
      WriterOptions options);

  ~Writer() override;

  void write(const velox::VectorPtr& input) override;

  void flush() override;

  /// Nimble writes each stripe eagerly, so there is no deferred work to yield
  /// on; the finish always completes in one call.
  bool finish() override;

  /// Flushes the trailing stripe, writes the footer and closes the file.
  /// Returns a NimbleFileMetadata carrying an owned snapshot of the per-column
  /// statistics, or nullptr when statistics collection is disabled.
  std::unique_ptr<velox::dwio::common::FileMetadata> close() override;

  /// Abandons the file without writing a footer, leaving the partially written
  /// bytes for the caller (or the sink) to discard.
  void abort() override;

  /// Names the writer publishes its counters under. Consumers name a key
  /// instead of a struct field, so adding a counter no longer changes this
  /// class's API.
  struct RuntimeStats {
    /// Total bytes written to the file.
    static constexpr std::string_view kWrittenBytes = "nimble.writtenBytes";
    /// Uncompressed size of data written to the file.
    static constexpr std::string_view kInputBytes = "nimble.inputBytes";
    /// CPU time spent in tabletWriter write.
    static constexpr std::string_view kWriteCpuNanos = "nimble.writeCpuNanos";
    /// Wall clock time spent in tabletWriter write.
    static constexpr std::string_view kWriteWallNanos = "nimble.writeWallNanos";
    /// CPU time spent ingesting vectors into field writer buffers.
    static constexpr std::string_view kIngestionCpuNanos =
        "nimble.ingestionCpuNanos";
    /// Wall clock time spent ingesting vectors into field writer buffers.
    static constexpr std::string_view kIngestionWallNanos =
        "nimble.ingestionWallNanos";
    /// CPU time spent on encoding and compression.
    // TODO: Separate encoding and compression costs.
    static constexpr std::string_view kEncodingCpuNanos =
        "nimble.encodingCpuNanos";
    /// Wall clock time spent on encoding and compression. Encoding is
    /// parallelized via encodingExecutor, so wall < CPU.
    static constexpr std::string_view kEncodingWallNanos =
        "nimble.encodingWallNanos";
    /// CPU time spent on encoding selection. Subset of the encoding timing.
    /// Sequential — no wall time needed.
    static constexpr std::string_view kEncodingSelectionCpuNanos =
        "nimble.encodingSelectionCpuNanos";
    /// Rows per stripe distribution. One value is recorded per stripe, so its
    /// `count` is the number of stripes written and no separate stripe counter
    /// is published.
    static constexpr std::string_view kRowsPerStripe = "nimble.rowsPerStripe";
    /// Encoded chunk size distribution in bytes.
    static constexpr std::string_view kChunkSizeBytes = "nimble.chunkSizeBytes";
    /// Number of streams deduplicated by the tablet writer.
    static constexpr std::string_view kDuplicateStreamCount =
        "nimble.duplicateStreamCount";
    /// Encoded bytes deduplicated by the tablet writer.
    static constexpr std::string_view kDuplicateStreamBytes =
        "nimble.duplicateStreamBytes";
  };

  /// Returns the writer's counters keyed by the names in RuntimeStats. This is
  /// the only form the writer publishes them in. Remains readable after
  /// close(), which is when the connector collects them.
  folly::F14FastMap<std::string, velox::RuntimeMetric> runtimeStats()
      const override;

  /// Returns the per-column statistics, populated at file close and empty when
  /// statistics collection is disabled. Indexed by pre-order schema node id,
  /// with a null entry for a node that collected none, so a caller may treat
  /// the index as the node id. Kept out of runtimeStats() because column
  /// statistics have no RuntimeMetric representation.
  ///
  /// Builds the view on each call; hold the result rather than re-reading it.
  std::vector<ColumnStatistics*> columnStats() const;

 private:
  // Reaches reclaimableBytes()/reclaimBytes() below, so that memory
  // arbitration does not require widening the writer's public API.
  friend class WriterMemoryReclaimer;

  // Reports the bytes the writer could release by flushing its buffered
  // stripe, or false when a flush would free too little to be worth the
  // encode cost.
  bool reclaimableBytes(
      const velox::memory::MemoryPool& pool,
      uint64_t& reclaimableBytes) const;

  // Releases memory by flushing the buffered stripe. Returns the bytes freed,
  // or 0 when the writer is no longer running or holds too little to flush.
  uint64_t reclaimBytes(
      velox::memory::MemoryPool* pool,
      velox::memory::MemoryReclaimer::Stats& stats);

  struct DenseIndexWriter {
    std::string name;
    std::unique_ptr<index::IndexWriter> writer;
  };

  // True when memory arbitration can reclaim from this writer, which requires
  // the caller to have supplied a spill config.
  bool canReclaim() const;

  // Builds the reclaimer installed on the writer's aggregate pool. Defined
  // once per build flavour: the internal build returns one derived from
  // velox::exec::MemoryReclaimer so arbitration suspends the Velox Driver,
  // while the OSS build returns nullptr because velox/exec is not part of
  // VELOX_BUILD_MINIMAL_WITH_DWIO.
  std::unique_ptr<velox::memory::MemoryReclaimer> makeMemoryReclaimer();

  // Snapshots the per-column statistics produced at file close into an owned
  // FileMetadata. Returns nullptr when statistics collection is disabled, in
  // which case downstream stats aggregation yields an empty result.
  std::unique_ptr<NimbleFileMetadata> buildFileMetadata() const;

  // Publishes the writer's timing breakdown to the Velox driver thread-local
  // runtime stats at file close.
  void reportRuntimeStats() const;

  static std::unique_ptr<index::IndexWriter> createClusterIndexWriter(
      const WriterOptions& options,
      const velox::TypePtr& type,
      velox::memory::MemoryPool* pool);

  static std::vector<DenseIndexWriter> createDenseIndexWriters(
      const WriterOptions& options,
      const velox::TypePtr& type,
      velox::memory::MemoryPool* pool);

  // Adds index keys to all configured index writers.
  void addIndexKey(const velox::VectorPtr& input);

  void writeProperties(const WriteOptionalSectionFn& writeMetadataFn);

  // Returns the vector written to data streams. When cluster index key column
  // storage is omitted, this is a top-level row projection excluding key
  // columns; index writers still consume the original input. In that mode,
  // write input must load to a top-level RowVector.
  velox::VectorPtr storedDataInput(const velox::VectorPtr& input) const;

  bool shouldFlush(FlushPolicy* policy) const;

  bool shouldChunk(FlushPolicy* policy) const;

  bool flushChunks(
      const std::vector<uint32_t>& indices,
      bool ensureFullChunks,
      FlushPolicy* policy);

  bool encodeStreamChunk(
      StreamData& streamData,
      uint64_t minChunkSize,
      uint64_t maxChunkSize,
      bool ensureFullChunks,
      Stream& stream,
      velox::BufferPool* encodingScratchBufferPool,
      EncodingBufferPool* encodingBufferPool,
      uint64_t& streamBytes,
      std::atomic_uint64_t& chunkBytes,
      std::atomic_uint64_t& logicalBytes);

  // Encodes a single chunk view and writes it to the encoded chunk.
  // Returns the number of bytes written to the encoded chunk.
  uint32_t encodeChunk(
      const StreamData& chunkView,
      Chunk& chunk,
      velox::BufferPool* encodingScratchBufferPool,
      EncodingBufferPool* encodingBufferPool);

  void encodeStream(
      StreamData& streamData,
      velox::BufferPool* encodingScratchBufferPool,
      EncodingBufferPool* encodingBufferPool,
      uint64_t& streamSize,
      std::atomic_uint64_t& chunkSize);

  // Drops all-true flat map in-map streams whose key is still provable from a
  // value stream. Runs once the stripe is fully encoded, where every stream's
  // fate is known -- the concurrent encode cannot decide this, since a stream
  // may not inspect its peers there.
  void suppressRedundantInMapStreams();

  void processStream(
      StreamData& streamData,
      velox::BufferPool* encodingScratchBufferPool,
      EncodingBufferPool* encodingBufferPool,
      uint64_t& streamSize,
      std::atomic_uint64_t& chunkSize);

  // Returning 'true' if stripe was flushed.
  bool evaluateFlushPolicy();

  // Drains buffered inputs from the installed BufferPolicy by repeatedly
  // calling writeBuffer(), ingesting the emitted (input, slice) pairs and
  // flushing one stripe per emitted BufferRange until the policy returns an
  // empty range. When 'finalize' is true, first calls BufferPolicy::finalize()
  // so any remaining rows the policy was still buffering get emitted; used at
  // close() to make sure no rows are left behind. Only invoked when a
  // BufferPolicy is installed. Returns true if any stripe was flushed.
  bool flushInputBuffers(bool finalize);

  // Ingests a (possibly sliced) input into the per-column stream buffers,
  // adds index keys, and refreshes per-stripe accounting. Callers pre-slice
  // via input->slice(start, length) when they need to write only a subset of
  // rows (as the BufferPolicy path does when emitting sub-batch ranges).
  // Does NOT consult the flush policy — the caller owns that.
  void writeBatch(const velox::VectorPtr& input);

  // Returning 'true' if stripe was written.
  bool writeStripe();

  // Encodes and writes all streams to the tablet writer. This method iterates
  // through all field writers to encode their stream data and append them to
  // the tablet. Note: This method does not perform chunking.
  void writeStreams();

  // Writes stream chunks for the specified stream indices. This method performs
  // chunking of encoded stream data and writes them to the tablet writer.
  // Returns 'true' if chunks were written.
  // Parameters:
  //   streamIndices: Indices of streams to write chunks for
  //   ensureFullChunks: If true, ensures chunks meet minimum size requirements
  //   lastChunk: If true, indicates this is the final chunk for the streams
  bool writeChunks(
      std::span<const uint32_t> streamIndices,
      bool ensureFullChunks = false,
      bool lastChunk = false);

  // Forwards the written-bytes and write-wall-time accrued since the previous
  // call to `options.ioStatistics`. No-op when the caller supplied no counters,
  // but the high-water marks are tracked either way so a caller that only reads
  // them at close still sees consistent deltas.
  void updateIoStatistics();

  // Writes caller-supplied key/value metadata into the optional metadata
  // section.
  void writeMetadata();
  // Writes the column statistics section, using the vectorized representation
  // when enabled and the legacy raw-size section otherwise.
  void writeColumnStats();
  // Writes the serialized Nimble schema section built from the writer context.
  void writeSchema();
  // Writes the dictionary catalog and any file-scope alphabet payloads.
  // File-scope alphabet bytes go through writeDataFn so the catalog can point
  // at their final file offsets.
  void writeDictionarySection(
      const WriteDataFn& writeDataFn,
      const WriteOptionalSectionFn& writeMetadataFn);
  // Encodes stripe-scope alphabet chunks into their dedicated dictionary
  // streams after value streams have chosen shared dictionary encoding.
  void writeStripeDictionaryStreams();
  // Finalizes and writes all indexes. Called via TabletWriter close callback.
  void writeIndexes(
      const WriteDataFn& writeDataFn,
      const CreateMetadataSectionFn& createMetadataFn,
      const WriteOptionalSectionFn& writeMetadataFn);

  // Top-level stream encoding buffer reused after encoded bytes are copied out.
  void ensureEncodingBuffer();
  void clearEncodingBuffer();

  // Scratch vector buffer pools. Pool 0 is used by sequential writes; parallel
  // writes use one pool per concurrent encode task because velox::BufferPool is
  // not thread-safe.
  std::unique_ptr<velox::BufferPool> makeEncodingScratchBufferPool() const;

  // Nested encoding buffer pools used by ScopedEncodingBuffer. Pool 0 is used
  // by sequential writes; parallel writes use one pool per concurrent encode
  // task because EncodingBufferPool is not thread-safe.
  std::unique_ptr<EncodingBufferPool> makeEncodingBufferPool() const;
  uint32_t encodingConcurrency(uint32_t streamCount) const;
  void ensureEncodingScratchBufferPools(uint32_t poolCount);
  void ensureEncodingBufferPools(uint32_t poolCount);
  velox::BufferPool* encodingScratchBufferPool(uint32_t index = 0);
  EncodingBufferPool* encodingBufferPool(uint32_t index = 0);

  void ensureWriteStreams();
  void resetFieldWriter();

  // Schema used to build normal data stream writers. Usually the input schema;
  // when cluster index key column storage is omitted, excludes the key columns
  // so no normal data streams are created for them.
  const velox::TypePtr storedDataType_;
  // Input column indices retained in storedDataType_. Empty unless key column
  // storage is omitted.
  const std::vector<velox::column_index_t> storedInputColumnIndices_;
  const std::shared_ptr<const velox::dwio::common::TypeWithId> schema_;
  // Flat row type of the file, kept for the closed-file metadata: the consumer
  // rebuilds a TypeWithId tree from it to line statistics up with pre-order
  // node ids.
  const velox::RowTypePtr rowType_;
  // Read by the memory reclaimer, which goes live on `pool_` below before the
  // rest of the writer is built and stays reachable until `pool_` is
  // destroyed. Declared ahead of `pool_` so it brackets that whole window;
  // `context_` does not, and reaching the spill config through it is what let
  // arbitration fault during construction and teardown.
  const velox::common::SpillConfig* const spillConfig_;
  MemoryPoolHolder pool_;
  MemoryPoolHolder encodingMemoryPool_;
  const std::unique_ptr<detail::WriterContext> context_;
  std::unique_ptr<velox::WriteFile> file_;
  const std::unique_ptr<index::IndexWriter> clusterIndexWriter_;
  const std::vector<DenseIndexWriter> denseIndexWriters_;
  const std::unique_ptr<TabletWriter> tabletWriter_;
  // Built once at construction from `options.bufferPolicyFactory`; null if
  // the caller didn't set the factory (legacy FlushPolicy path).
  const std::unique_ptr<BufferPolicy> bufferPolicy_;

  std::unique_ptr<FieldWriter> rootWriter_;
  std::unique_ptr<Buffer> encodingBuffer_;

  // Per-encode-task scratch vector buffer caches. They are retained across
  // flushes for reuse, cleared under memory arbitration, and released when the
  // writer is destroyed.
  std::vector<std::unique_ptr<velox::BufferPool>> encodingScratchBufferPools_;

  // Per-encode-task temporary encoded-buffer caches used by
  // ScopedEncodingBuffer. They follow the same lifetime as the scratch vector
  // buffer caches above.
  std::vector<std::unique_ptr<EncodingBufferPool>> encodingBufferPools_;
  std::vector<Stream> encodedStreams_;
  std::exception_ptr lastException_;

  // Totals already reported to `options.ioStatistics`; the next update forwards
  // only what accrued beyond these.
  uint64_t reportedBytesWritten_{0};
  uint64_t reportedWriteWallTimeNs_{0};
};

/// Reads one counter out of a runtime-stat map, or a zeroed metric when the key
/// is absent, which is the case for a map merged from writers that did not
/// publish it. Callers name a Writer::RuntimeStats constant, so a mistyped
/// key is a compile error rather than a silently zero counter.
velox::RuntimeMetric runtimeStat(
    const folly::F14FastMap<std::string, velox::RuntimeMetric>& stats,
    std::string_view key);

} // namespace facebook::nimble
