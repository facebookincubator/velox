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

#include <folly/Executor.h>
#include <folly/coro/Task.h>

#include "velox/common/time/CpuWallTimer.h"
#include "velox/dwio/nimble/velox/RowRange.h"
#include "velox/serializers/KeyEncoder.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox {
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

  /// Serializes a non-null row into a buffer and returns its serialized size.
  /// The buffer must contain at least the returned number of bytes. Use this
  /// callback to provide formats beyond the built-in row formats. A serializer
  /// may return zero without allocating a buffer.
  using OutputRowSerializer = std::function<std::size_t(
      const velox::RowVectorPtr&,
      velox::vector_size_t,
      velox::memory::MemoryPool*,
      velox::BufferPtr&)>;

  /// Selects the CompactRow serializer.
  static constexpr std::string_view kCompactRowFormat{"CompactRow"};

  /// Selects the UnsafeRow serializer.
  static constexpr std::string_view kUnsafeRowFormat{"UnsafeRow"};

  /// Controls output batching and stripe prefetch for an index iterator.
  struct Options {
    /// Maximum rows returned in one batch. A batch never crosses a stripe
    /// boundary. Row iterators decode all remaining selected rows in a
    /// projected stripe when the row value is requested.
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
  virtual folly::coro::Task<void> co_seek(
      velox::serializer::EncodedKeyBounds bounds) = 0;

  /// Positions the iterator at the first indexed row.
  virtual folly::coro::Task<void> co_seekToFirst() = 0;

  /// Cancels pending prefetch and invalidates the current scan.
  virtual folly::coro::Task<void> co_reset() = 0;

  /// Returns whether the iterator is positioned on a selected row.
  bool valid() const {
    return hasCurrentRow();
  }

  /// Advances a row iterator past its current row.
  /// Fails if the iterator is not positioned on a row.
  virtual folly::coro::Task<void> co_nextRow() = 0;

  /// Materializes the next batch and advances past it. Returns false after all
  /// matching rows have been emitted.
  virtual folly::coro::Task<bool> co_nextBatch(velox::RowVectorPtr& output) = 0;

  /// Returns the encoded cluster-index key for the current row. Requires the
  /// iterator to be positioned on a row.
  virtual std::string_view key() const = 0;

  /// Returns the serialized current row, materializing it on demand. Requires
  /// the iterator to be positioned on a row.
  virtual folly::coro::Task<std::string_view> co_value() = 0;

  /// Returns the current file-level row number. Requires the iterator to be
  /// positioned on a row.
  virtual uint32_t rowNumber() const = 0;

  /// Returns cumulative statistics for this iterator.
  /// Call only between iterator operations.
  virtual Stats stats() const = 0;

 protected:
  NimbleIterator() = default;

  // Returns whether the iterator still has a selected row to emit.
  bool hasCurrentRow() const;

  // Returns the current file-level row, checking that it is selected.
  uint32_t currentRow() const;

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
