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
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <folly/Executor.h>
#include <folly/coro/Task.h>

#include "velox/common/time/CpuWallTimer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/velox/RowRange.h"
#include "velox/serializers/KeyEncoder.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox {
class BufferPool;
namespace memory {
class MemoryPool;
}
} // namespace facebook::velox

namespace facebook::nimble {

class NimbleIndexProjector;

/// Iterates index-selected Nimble rows as serialized rows or Velox batches.
/// Each iterator exclusively owns its non-thread-safe projector and must be
/// used by one thread at a time.
class NimbleIterator {
 public:
  /// Names a serialized row format accepted by the row-iterator factory.
  using OutputRowFormat = std::string;

  /// Holds serialized rows back to back. Row i of the range occupies the bytes
  /// [offsets[i], offsets[i + 1]) of 'buffer'.
  struct SerializedRows {
    velox::BufferPtr buffer;
    std::vector<std::size_t> offsets;

    /// Returns the bytes of row 'index' of the range, which stay valid while
    /// this object holds the buffer. The buffer is null when every row of the
    /// range is empty, and each row is then an empty view.
    std::string_view row(std::size_t index) const;
  };

  /// Serializes the non-null rows [rowOffset, rowOffset + numRows) of a vector
  /// into 'output'. The iterator passes the output of its previous call, so a
  /// serializer can reuse the buffer. The serializer sets numRows + 1 offsets
  /// and fills the buffer with the bytes they cover. It may leave the buffer
  /// null when every row is empty. 'bufferPool' is optional and can be null. A
  /// serializer that takes buffers from it lets the iterator recycle them
  /// across stripes and iterators. Use this callback to provide formats beyond
  /// the built-in row formats. The row iterator calls it once for all the rows
  /// of a decoded stripe.
  using OutputRowSerializer = std::function<void(
      const velox::RowVectorPtr& rows,
      velox::vector_size_t rowOffset,
      velox::vector_size_t numRows,
      velox::memory::MemoryPool* pool,
      velox::BufferPool* bufferPool,
      SerializedRows& output)>;

  /// Selects the CompactRow serializer.
  static constexpr std::string_view kCompactRowFormat{"CompactRow"};

  /// Selects the UnsafeRow serializer.
  static constexpr std::string_view kUnsafeRowFormat{"UnsafeRow"};

  /// Controls output batching and stripe prefetch for an index iterator.
  struct Options {
    /// Maximum rows returned in one batch. A batch never crosses a stripe
    /// boundary. Row iterators ignore this limit. They decode and serialize all
    /// remaining selected rows in a projected stripe when the first row of the
    /// stripe is read.
    uint32_t outputBatchRows{1'024};

    /// Executor used to project later stripes ahead of the consumer.
    /// Must accept task submissions while its keep-alive token is held.
    /// Prefetch is disabled when the executor is empty or the stripe limit is
    /// zero.
    folly::Executor::KeepAlive<> prefetchExecutor{};

    /// Maximum stripe slots reserved by queued or loading prefetch batches.
    /// Batches target half this limit, rounded down, with a minimum of one
    /// stripe. Reservations are held until the batch's last stripe is consumed.
    /// Zero disables prefetch.
    uint32_t maxPrefetchStripes{1};

    /// Selects the serialized row format. Batch iterators ignore this option.
    OutputRowFormat rowFormat{std::string{kCompactRowFormat}};

    /// Optional pool that recycles the buffer of serialized rows. A row
    /// iterator takes the buffer from the pool and returns it when it leaves a
    /// stripe or is destroyed, so iterators that share a pool reuse the buffer.
    /// The pool is not thread-safe, so iterators used concurrently need their
    /// own pools. A buffer keeps the memory pool of the iterator that allocated
    /// it, so share a pool only among iterators with the same memory pool.
    /// The pool must outlive the iterator. Batch iterators ignore this option.
    velox::BufferPool* bufferPool{nullptr};

    /// Reads the CPU and wall clocks around index lookups, plan building,
    /// stripe loads and the decoding of stripes to fill the timings of Stats.
    /// Reading the thread CPU clock is a system call, so a point lookup spends
    /// about 3 us on it. The timings stay at zero when this is false.
    bool collectTimings{true};

    /// Builds the lookup structures of an encoding view when it is created, so
    /// that later reads of the view are faster. A scan reads many rows of each
    /// stripe and benefits. A point lookup reads one row and does better
    /// without them. See Encoding::Options::cacheEncodingView.
    bool cacheEncodingView{true};
  };

  /// Creates an iterator that serializes one selected row at a time.
  static std::unique_ptr<NimbleIterator> createRowIterator(
      std::unique_ptr<NimbleIndexProjector> projector,
      velox::memory::MemoryPool* pool,
      const Options& options);

  /// Creates an iterator that materializes selected rows into Velox batches.
  static std::unique_ptr<NimbleIterator> createBatchIterator(
      std::unique_ptr<NimbleIndexProjector> projector,
      velox::memory::MemoryPool* pool,
      Options options);

  /// Statistics accumulated while the iterator projects rows.
  struct Stats {
    /// Number of stripes read from the tablet.
    uint32_t numReadStripes{0};
    /// Number of prefetched stripes discarded after row navigation skips them.
    uint32_t numSkippedStripes{0};
    /// Number of read stripes serialized through stream slicing.
    uint32_t numSlicedStripes{0};
    /// Total rows read from storage, including rows outside requested ranges.
    uint64_t numReadRows{0};
    /// Total rows selected for the iterator's output.
    uint64_t numOutputRows{0};
    /// Total rows decoded into vectors. A row iterator decodes and serializes
    /// the selected rows of a stripe when it reads the first of them, so a
    /// range that excludes rows keeps this count below the rows of the stripe.
    uint64_t numMaterializedRows{0};
    /// Total bytes retained by projected stripe streams.
    uint64_t numOutputBytes{0};
    /// Total projected stream bytes planned by NimbleIndexProjector.
    uint64_t numPlannedBytes{0};

    /// Time spent looking up stripes and row ranges via the tablet index.
    velox::CpuWallTiming lookupTiming{};
    /// Time spent building per-stripe stream and request-range plans.
    velox::CpuWallTiming prepareTiming{};
    /// Time spent loading stripe stream data from tablet.
    velox::CpuWallTiming scanTiming{};
    /// Time spent assembling projected stripe streams.
    velox::CpuWallTiming projectionTiming{};
    /// Time spent decoding projected streams into vectors or serialized rows.
    velox::CpuWallTiming materializationTiming{};
  };

  virtual ~NimbleIterator() = default;

  NimbleIterator(const NimbleIterator&) = delete;
  NimbleIterator& operator=(const NimbleIterator&) = delete;
  NimbleIterator(NimbleIterator&&) = delete;
  NimbleIterator& operator=(NimbleIterator&&) = delete;

  /// Positions the iterator at the first row matching the encoded bounds.
  /// Blocks until the prefetch of the previous scan stops. The methods that
  /// block must not run on a thread of `Options::prefetchExecutor`, which the
  /// prefetch needs to stop.
  virtual void seek(const velox::serializer::EncodedKeyBounds& bounds) = 0;

  /// Positions the iterator at the rows whose key equals the encoded key, and
  /// leaves it invalid when there are none. The rows of a key can span chunks
  /// and stripes, so this selects the range from the key to its successor, the
  /// key followed by a zero byte. Blocks until the prefetch of the previous
  /// scan stops.
  virtual void seek(std::string encodedKey) = 0;

  /// Positions the iterator at the first indexed row. Blocks until the
  /// prefetch of the previous scan stops.
  virtual void seekToFirst() = 0;

  /// Cancels pending prefetch and invalidates the current scan. Blocks until
  /// the prefetch stops.
  virtual void reset() = 0;

  /// Returns whether the iterator is positioned on a selected row.
  bool valid() const {
    return hasCurrentRow();
  }

  /// Advances a row iterator past its current row. Invalidates the key and
  /// value views of the current row. Fails if the iterator is not positioned on
  /// a row.
  virtual void nextRow() = 0;

  /// Materializes the next batch and advances past it. Returns false after all
  /// matching rows have been emitted.
  virtual folly::coro::Task<bool> co_nextBatch(velox::RowVectorPtr& output) = 0;

  /// Returns the encoded cluster-index key for the current row. Requires the
  /// iterator to be positioned on a row.
  virtual std::string_view key() const = 0;

  /// Loads and decodes the stripe of the current row when the previous rows did
  /// not, which blocks the calling thread until the stripe is available. Lets a
  /// caller learn about a failure to load the value before it reads the value.
  /// Requires the iterator to be positioned on a row.
  virtual void prepareValue() = 0;

  /// Returns the serialized current row, preparing the value first if the
  /// caller did not. The view stays valid until the next call to nextRow(),
  /// seek(), seekToFirst() or reset(), because the buffer of a stripe returns
  /// to `Options::bufferPool` once the iterator leaves the stripe. Requires the
  /// iterator to be positioned on a row.
  virtual std::string_view value() = 0;

  /// Returns the current file-level row number. Requires the iterator to be
  /// positioned on a row.
  virtual uint32_t rowNumber() const = 0;

  /// Returns cumulative statistics for this iterator.
  /// Call only between iterator operations.
  virtual Stats stats() const = 0;

 protected:
  NimbleIterator() = default;

  // Returns whether the iterator still has a selected row to emit.
  bool hasCurrentRow() const {
    return rowRange_.has_value() && row_ < rowRange_->endRow;
  }

  // Returns the current file-level row, checking that it is selected. Defined
  // here because every row access calls it.
  uint32_t currentRow() const {
    NIMBLE_CHECK(hasCurrentRow(), "Iterator is not positioned on a row");
    return row_;
  }

  // Initializes row navigation from the cursor's resolved file-row bounds.
  void setRowRange(std::optional<RowRange> rowRange);

  // Advances row navigation without changing stripe-loading progress.
  void advanceRows(uint32_t numRows);

 private:
  // File-level bounds selected for this iterator's current scan.
  std::optional<RowRange> rowRange_;
  // Current file-level row, advanced by the row or batch iterator.
  uint32_t row_{0};
};

/// Registers CompactRow and UnsafeRow serializers process-wide. Repeated
/// calls are safe. The row-iterator factory also ensures they are registered.
void registerDefaultOutputRowSerializers();

/// Registers a custom row serializer process-wide. Fails if the format is
/// empty, built-in, already registered, or the serializer is empty.
void registerOutputRowSerializer(
    NimbleIterator::OutputRowFormat rowFormat,
    const NimbleIterator::OutputRowSerializer& serializer);

} // namespace facebook::nimble
