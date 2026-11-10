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
#include <folly/io/IOBufQueue.h>

#include "velox/common/time/CpuWallTimer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/RowRange.h"
#include "velox/serializers/KeyEncoder.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::memory {
class MemoryPool;
} // namespace facebook::velox::memory

namespace facebook::nimble {

class NimbleIndexProjector;

namespace index {
class KeyCursor;
} // namespace index

/// Iterates index-selected Nimble rows as output entries. Each iterator
/// exclusively owns its non-thread-safe projector and must be used by one
/// thread at a time.
class NimbleIterator {
 public:
  /// Decodes the rows of a stripe into output entries, a key and a value each,
  /// in key order, however many entries a row has. 'keyCursor' supplies one
  /// encoded cluster-index key per row of the stripe. The decoder writes the
  /// bytes of the entries into 'keyStorage' and 'valueStorage', and sets
  /// 'keys' and 'values', which view those storages, to the entries. The
  /// iterator releases 'rows' after the call, so a decoder whose entries refer
  /// to 'rows' ties them to a storage, for example with an IOBuf whose free
  /// function holds them. A non-empty input must produce at least one entry.
  /// The iterator calls the decoder on the thread consuming the iterator.
  /// I/O prefetch does not call the decoder.
  using EntryDecoder = std::function<void(
      const velox::RowVectorPtr& rows,
      index::KeyCursor& keyCursor,
      folly::IOBufQueue& keyStorage,
      folly::IOBufQueue& valueStorage,
      std::vector<std::string_view>& keys,
      std::vector<std::string_view>& values)>;

  /// Controls stripe prefetch for an index iterator.
  struct Options {
    /// Executor used to load projected encoded streams ahead of the consumer.
    /// Decoding and entry conversion run on the consumer thread.
    /// Must accept task submissions while its keep-alive token is held.
    /// Prefetch is disabled when the executor is empty or the stripe limit is
    /// zero.
    folly::Executor::KeepAlive<> prefetchExecutor{};

    /// Maximum stripe slots reserved by queued or running I/O prefetch batches.
    /// Batches target half this limit, rounded down, with a minimum of one
    /// stripe. Reservations are held until the batch's last stripe is dequeued.
    /// Zero disables prefetch.
    uint32_t maxPrefetchStripes{1};

    /// Reads the CPU and wall clocks around index lookups, plan building,
    /// stripe loads and the decoding and conversion of stripes to fill the
    /// timings of Stats.
    /// Reading the thread CPU clock is a system call, so a point lookup spends
    /// about 3 us on it. The timings stay at zero when this is false.
    bool collectTimings{true};

    /// Builds the lookup structures of an encoding view when it is created, so
    /// that later reads of the view are faster. A scan reads many rows of each
    /// stripe and benefits. A point lookup reads one row and does better
    /// without them. See Encoding::Options::cacheEncodingView.
    bool cacheEncodingView{true};
  };

  /// Creates an iterator over the output entries that 'entryDecoder' decodes
  /// from the selected rows. The iterator moves over entries: valid(),
  /// nextRow(), key() and value() refer to the current entry, whose key is the
  /// key that the decoder returned and not the cluster-index key. A row of the
  /// file can have any number of entries.
  /// After a prefetched decode failure, reset or seek before reading entries
  /// again. valid() only reports whether the iterator is positioned on an
  /// entry.
  static std::unique_ptr<NimbleIterator> createRowIterator(
      std::unique_ptr<NimbleIndexProjector> projector,
      velox::memory::MemoryPool* pool,
      EntryDecoder entryDecoder,
      const Options& options);

  /// Statistics accumulated while the iterator projects rows.
  struct Stats {
    /// Number of stripes read from the tablet.
    uint32_t numReadStripes{0};
    /// Number of read stripes served through stream slicing.
    uint32_t numSlicedStripes{0};
    /// Total rows read from storage, including rows outside requested ranges.
    uint64_t numReadRows{0};
    /// Total rows selected for the iterator's output.
    uint64_t numOutputRows{0};
    /// Total selected rows decoded into vectors on the consumer thread.
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
    /// Time spent decoding projected streams into vectors on the consumer.
    velox::CpuWallTiming decodingTiming{};
    /// Time spent converting decoded rows into entries on the consumer.
    velox::CpuWallTiming conversionTiming{};
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

  /// Turns the verification of stream checksums on or off for the next seeks,
  /// as ReaderOptions::verifyStreamChecksums() of the projector does at
  /// creation, so that a caller that keeps an iterator between lookups reads
  /// each lookup with its own setting. Must be called after reset().
  virtual void setVerifyStreamChecksums(bool verifyStreamChecksums) = 0;

  /// Returns whether the iterator is positioned on a selected row.
  bool valid() const {
    return hasCurrentRow();
  }

  /// Advances the iterator past its current entry. Invalidates the key and
  /// value views of the current entry. Fails if the iterator is not positioned
  /// on an entry.
  virtual void nextRow() = 0;

  /// Returns whether an entry follows the current one, which is whether
  /// nextRow() leaves the iterator on an entry. Loads the stripe of the current
  /// entry when the previous entries did not, and never the stripe after it.
  /// Fails if the iterator is not positioned on an entry.
  virtual bool hasNext() const = 0;

  /// Returns the key of the current entry, and loads its stripe when the
  /// previous entries did not. The view stays valid until the iterator reads an
  /// entry of another stripe, seeks or resets. Requires the iterator to be
  /// positioned on an entry.
  virtual std::string_view key() const = 0;

  /// Returns the value of the current entry, and loads its stripe when the
  /// previous entries did not. The view stays valid until the iterator reads an
  /// entry of another stripe, seeks or resets. Requires the iterator to be
  /// positioned on an entry.
  virtual std::string_view value() = 0;

  /// Returns cumulative statistics for this iterator.
  /// Call only between iterator operations.
  virtual Stats stats() const = 0;

 protected:
  NimbleIterator() = default;

  // Returns whether the file-level 'row' is before the end of the selected
  // rows.
  bool isBeforeEndRow(uint32_t row) const {
    return rowRange_.has_value() && row < rowRange_->endRow;
  }

  // Returns whether the iterator still has a selected row to emit.
  bool hasCurrentRow() const {
    return isBeforeEndRow(row_);
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
  // Current file-level row, advanced by the iterator.
  uint32_t row_{0};
};

} // namespace facebook::nimble
