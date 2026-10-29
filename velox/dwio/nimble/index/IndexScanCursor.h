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

namespace index {
class ClusterIndexBase;
} // namespace index

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

    /// Decodes the selected rows of a loaded stripe and turns them into the
    /// entries that the iterator serves. The selection of the stripe holds the
    /// selected rows. 'clusterIndex' belongs to the file, and the entries can
    /// take the keys of the rows from it. A cursor that prefetches calls it on
    /// threads of the prefetch executor, for several stripes at a time, so it
    /// must be thread-safe. One that does not calls it on the caller of
    /// co_nextStripe().
    virtual folly::coro::Task<std::unique_ptr<Entries>> materialize(
        ProjectedStripe stripe,
        const index::ClusterIndexBase& clusterIndex) = 0;
  };

  /// Creates a cursor that owns `projector` and materializes the stripes that
  /// it loads with `materializer`, ahead of the iterator when it prefetches.
  /// Neither may be null. The iterator reads the stripes with co_nextStripe().
  /// The prefetch shares the materializer, and keeps it until the prefetch
  /// stops, which can be after the cursor goes away.
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

  /// Resolves the rows whose key equals the encoded key and starts stripe
  /// loading. Returns the selected file-row bounds, or nullopt on a miss.
  /// Blocks until the prefetch of the previous scan stops.
  std::optional<RowRange> seek(std::string encodedKey);

  /// Starts a scan at the first file row. Blocks until the prefetch of the
  /// previous scan stops.
  std::optional<RowRange> seekToFirst();

  /// Clears the active scan and its outstanding prefetch. Blocks until the
  /// prefetch stops.
  void reset();

  /// Loads and materializes the next selected stripe, whose selected rows start
  /// at the iterator's file-level 'row', and returns its entries. The iterator
  /// reads the stripes in order: 'row' is the first row of the scan, and then
  /// the end of the rows of the previous stripe. A stripe that fails to load or
  /// to materialize ends the scan: this call throws the failure when the
  /// iterator reaches the stripe, and later calls throw it again until the next
  /// seek.
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

  // Positions the cursor at the start of the resolved stripe selections.
  void setStripeSelections(std::vector<StripeSelection> selections);

  // Returns the selection of the next stripe, and checks that its selected rows
  // start at 'row'.
  const StripeSelection& selectStripe(uint32_t row) const;

  // Starts bounded stripe prefetch for the active scan, when configured.
  void startPrefetch();

  // Returns whether the configured limits allow a non-empty scan to prefetch.
  bool canPrefetch() const;

  // Creates owned cancellation work before clearing the producer.
  folly::coro::Task<void> co_resetPrefetch();

  // Shares the projector with its background prefetch task.
  const ProjectorPtr projector_;
  // Executor used to project later stripes.
  const folly::Executor::KeepAlive<> prefetchExecutor_;
  // Maximum number of stripe streams queued ahead of the consumer.
  const uint32_t maxPrefetchStripes_;
  // Materializes the stripes that the cursor loads, and is shared with the
  // prefetch.
  const std::shared_ptr<StripeMaterializer> stripeMaterializer_;
  // Contiguous file-row range selected by the active key bounds.
  std::optional<RowRange> range_;
  // Selects contiguous stripes and file rows for the active scan. Empty means
  // the scan has no selected rows and cannot prefetch.
  std::vector<StripeSelection> stripeSelections_;
  // First selected stripe not yet returned to the iterator.
  size_t stripeOffset_{0};
  // Rows made available to the iterator from loaded stripe selections.
  uint64_t numOutputRows_{0};
  // Owns the bounded queue and task that project later stripe streams.
  std::shared_ptr<ScanPrefetcher> scanPrefetcher_;
};

} // namespace detail
} // namespace facebook::nimble
