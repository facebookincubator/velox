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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <folly/Executor.h>
#include <folly/coro/Task.h>
#include <folly/io/IOBufQueue.h>

#include "velox/buffer/BufferPool.h"
#include "velox/dwio/nimble/common/RowRange.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/index/KeyCursor.h"
#include "velox/dwio/nimble/index/NimbleIterator.h"
#include "velox/dwio/nimble/index/StripeStreams.h"
#include "velox/serializers/KeyEncoder.h"

namespace facebook::nimble {

class Type;

namespace detail {

class ScanPrefetcher;

/// Owns an index scan and its bounded stripe-stream prefetch for one iterator.
/// Is not thread-safe. The owning iterator must serialize cursor calls.
/// Resets and joins prefetch before using mutable projector APIs on the caller.
class IndexScanCursor final {
 public:
  using ProjectorPtr = std::shared_ptr<NimbleIndexProjector>;

  /// Turns the streams of a loaded stripe into the entries that the iterator
  /// serves. The iterator that serves entries implements it.
  class StripeMaterializer {
   public:
    /// The entries of a materialized stripe: the key and the value of each
    /// entry, in key order, and the storage that they view.
    struct Entries {
      /// Allocates the storage of the entries with 'iobufFactory', which must
      /// outlive the entries.
      explicit Entries(folly::IOBufFactory& iobufFactory);

      /// Hold the bytes of the keys and of the values that the entries view.
      folly::IOBufQueue keyStorage{folly::IOBufQueue::cacheChainLength()};
      folly::IOBufQueue valueStorage{folly::IOBufQueue::cacheChainLength()};
      /// The key and the value of each entry.
      std::vector<std::string_view> keys;
      std::vector<std::string_view> values;
      /// The rows of the file that the entries came from.
      velox::vector_size_t numRows{0};
    };

    virtual ~StripeMaterializer() = default;

    /// Decodes the selected rows of a stripe and turns them into the entries
    /// that the iterator serves. The selection of the stripe holds the
    /// selected rows. The stripe holds the streams that the cursor loaded, or
    /// no streams when the materializer keeps the stripe from an earlier
    /// call, as loadedStripe() reports. 'keyCursor' reads the keys of the
    /// selected rows from the cluster index of the file, in row order, and
    /// the entries can take their keys from it. It reads on from the key
    /// cursor of the scan, which decodes the keys of the stripe from its key
    /// chunk when the stripe came with one, and then skips the keys of the
    /// stripe that the materializer left unread. Runs on the caller of
    /// co_nextStripe(), including when prefetch is enabled.
    virtual folly::coro::Task<std::unique_ptr<Entries>> materialize(
        ProjectedStripe stripe,
        index::KeyCursor& keyCursor) = 0;

    /// Returns the stripe 'stripeIndex' when the materializer keeps it from
    /// an earlier call, with its streams and its key chunk, so that the
    /// cursor does not load the stripe again. Returns null otherwise.
    virtual const ProjectedStripe* loadedStripe(uint32_t stripeIndex) const = 0;
  };

  /// Creates a cursor that owns `projector` and materializes loaded stripes
  /// with `materializer`. Neither may be null. When configured, prefetch loads
  /// encoded streams ahead; materialization stays on the caller of
  /// co_nextStripe().
  static std::shared_ptr<IndexScanCursor> create(
      std::unique_ptr<NimbleIndexProjector> projector,
      std::shared_ptr<StripeMaterializer> materializer,
      const NimbleIterator::Options& options);

  /// Returns the decoding options of the files of 'projector', with the
  /// caller's scratch-buffer pool. See NimbleIterator::Options for
  /// 'cacheEncodingView'.
  static Encoding::Options encodingOptions(
      const NimbleIndexProjector& projector,
      bool cacheEncodingView,
      velox::BufferPool* bufferPool);

  /// Requests cancellation and defers producer cleanup until its tasks stop.
  /// Joins synchronously if asynchronous cleanup cannot start.
  ~IndexScanCursor();

  IndexScanCursor(const IndexScanCursor&) = delete;
  IndexScanCursor& operator=(const IndexScanCursor&) = delete;
  IndexScanCursor(IndexScanCursor&&) = delete;
  IndexScanCursor& operator=(IndexScanCursor&&) = delete;

  /// Resolves the encoded key bounds and starts stripe loading. Returns the
  /// selected file-row bounds for the iterator to navigate, or nullopt on a
  /// miss. Blocks until the prefetch of the previous scan stops.
  std::optional<RowRange> seek(
      const velox::serializer::EncodedKeyBounds& bounds);

  /// Starts a scan at the first file row. Blocks until the prefetch of the
  /// previous scan stops.
  std::optional<RowRange> seekToFirst();

  /// Clears the active scan and its outstanding prefetch, and releases its key
  /// cursor. Blocks until the prefetch stops.
  void reset();

  /// Turns the verification of stream checksums on or off for the next scans.
  /// Must be called without an active scan, such as after reset().
  void setVerifyStreamChecksums(bool verifyStreamChecksums);

  /// Loads the next selected stripe, or waits for its prefetched streams, then
  /// materializes it on the caller with a key cursor over its selected rows.
  /// Its selected rows start at the iterator's file-level 'row', and the call
  /// returns its entries. The iterator reads the stripes in order: 'row' is the
  /// first row of the scan, and then the end of the rows of the previous
  /// stripe. The cursor does not load a stripe whose streams the materializer
  /// keeps. The first stripe of a seek reads its keys from the key cursor of
  /// the seek, which starts with the key chunk that the seek found its first
  /// row in. A later stripe that came with its key chunk reads its keys from a
  /// new key cursor over its selected rows, which decodes them from that
  /// chunk, and a stripe without one reads on from the key cursor of the
  /// stripe before it. The cursor
  /// releases the key cursor after the last stripe. With prefetch
  /// enabled, later calls rethrow a load failure until the next seek. A stripe
  /// that fails after its streams load stays the next stripe, so without
  /// prefetch a call with the same 'row' retries it. A prefetched stripe has
  /// left the prefetch queue, so reset or seek before reading again after it
  /// fails. Calls fail until then.
  folly::coro::Task<std::unique_ptr<StripeMaterializer::Entries>> co_nextStripe(
      uint32_t row);

  /// Returns approximate cumulative projector and cursor statistics.
  /// Counters can reflect different points of an active prefetch load.
  NimbleIterator::Stats stats() const;

 private:
  // Initializes projector access and the configured scan limits.
  IndexScanCursor(
      std::unique_ptr<NimbleIndexProjector> projector,
      std::shared_ptr<StripeMaterializer> materializer,
      const NimbleIterator::Options& options);

  // Resolves 'bounds' for seek() when the projector co-loads key streams. The
  // stripe that the first selected row can lie in comes with its key stream:
  // from the materializer when it holds the stripe, and otherwise from a load.
  // The lookup resolves the lower bound in that key chunk instead of reading
  // the index. The scan reads a loaded stripe first, and a seek that selects
  // no row counts it as skipped.
  std::vector<StripeSelection> coLoadLookup(
      const velox::serializer::EncodedKeyBounds& bounds);

  // Positions the cursor at the start of the resolved stripe selections.
  void setStripeSelections(std::vector<StripeSelection> selections);

  // Returns the selection of the next stripe, and checks that its selected rows
  // start at 'row'.
  const StripeSelection& selectStripe(uint32_t row) const;

  // Returns the key cursor of the active scan, at the first selected row of
  // 'selection'. A stripe that came with 'keyChunk' gets a key cursor over its
  // selected rows, which decodes their keys from the chunk, except for the
  // first stripe of a seek, whose key cursor starts with the key chunk that
  // the seek found its first row in. A stripe without a key chunk reads on
  // from the key cursor before it, or gets one over the rows from there to the
  // end of the scan when the scan has none: a scan from the first row has no
  // key cursor of a seek, and a stripe that failed dropped it.
  index::KeyCursor* ensureKeyCursor(
      const StripeSelection& selection,
      const std::optional<index::ClusterIndexBase::KeyChunk>& keyChunk);

  // Starts bounded stripe prefetch for the active scan, when configured.
  void startPrefetch();

  // Returns whether the configured limits allow a non-empty scan to prefetch.
  bool canPrefetch() const;

  // Creates owned cancellation work before clearing the producer.
  folly::coro::Task<void> co_resetPrefetch();

  // Shares the projector with its background prefetch task.
  const ProjectorPtr projector_;
  // Executor used to load encoded stripe streams ahead of the consumer.
  const folly::Executor::KeepAlive<> prefetchExecutor_;
  // Maximum number of prefetched stripes queued ahead of the consumer.
  const uint32_t maxPrefetchStripes_;
  // Number of batches that 'maxPrefetchStripes_' is split into, each loaded in
  // one read.
  const uint32_t numPrefetchBatches_;
  // Materializes stripes on the iterator caller.
  const std::shared_ptr<StripeMaterializer> stripeMaterializer_;
  // Contiguous file-row range selected by the active key bounds.
  std::optional<RowRange> range_;
  // Selects contiguous stripes and file rows for the active scan. Empty means
  // the scan has no selected rows and cannot prefetch.
  std::vector<StripeSelection> stripeSelections_;
  // Reads the keys of the active scan from the first row of the next stripe,
  // holding only the key chunks that it reads. A seek starts it with the chunk
  // of its first row, which other readers of the index can otherwise evict
  // from the index's cache before the scan reads its keys. A later stripe that
  // came with its key chunk replaces it with a key cursor over the stripe's
  // selected rows that decodes their keys from the chunk, and a stripe without
  // one reads its keys on from it, through the index. Null after the last
  // stripe and after a stripe
  // that failed, and for a scan from the first row, so that the next stripe
  // creates one.
  std::unique_ptr<index::KeyCursor> keyCursor_;
  // The first selected stripe, which the seek loaded with the key stream that
  // resolved it, until the iterator reads it.
  std::optional<ProjectedStripe> loadedStripe_;
  // First selected stripe not yet returned to the iterator.
  size_t stripeOffset_{0};
  // Rows made available to the iterator from loaded stripe selections.
  uint64_t numOutputRows_{0};
  // Stripes that seeks loaded with their key streams and then skipped, because
  // no selected row came from them, as when a key is missing.
  uint32_t numSkippedStripes_{0};
  // Owns the bounded queue and task that load later stripe streams.
  std::shared_ptr<ScanPrefetcher> scanPrefetcher_;
};

} // namespace detail
} // namespace facebook::nimble
