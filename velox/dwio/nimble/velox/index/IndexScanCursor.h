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
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <folly/Executor.h>
#include <folly/coro/Task.h>

#include "velox/buffer/BufferPool.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/index/KeyCursor.h"
#include "velox/dwio/nimble/velox/RowRange.h"
#include "velox/dwio/nimble/velox/index/NimbleIterator.h"
#include "velox/dwio/nimble/velox/index/StripeStreams.h"
#include "velox/serializers/KeyEncoder.h"

namespace facebook::nimble {

class Type;

namespace detail {

/// Owns an index scan and its bounded stripe-stream prefetch for one iterator.
/// Is not thread-safe. The owning iterator must serialize cursor calls.
/// Resets and joins prefetch before using mutable projector APIs on the caller.
class IndexScanCursor final
    : public std::enable_shared_from_this<IndexScanCursor> {
 public:
  using ProjectorPtr = std::shared_ptr<NimbleIndexProjector>;

  /// Creates a cursor that owns `projector` and uses the configured scan
  /// options. `projector` must be non-null.
  static std::shared_ptr<IndexScanCursor> create(
      std::unique_ptr<NimbleIndexProjector> projector,
      const NimbleIterator::Options& options);

  /// Requests cancellation and defers producer cleanup until its tasks stop.
  /// Logs failures when detached cleanup cannot be started or completed.
  ~IndexScanCursor();

  IndexScanCursor(const IndexScanCursor&) = delete;
  IndexScanCursor& operator=(const IndexScanCursor&) = delete;
  IndexScanCursor(IndexScanCursor&&) = delete;
  IndexScanCursor& operator=(IndexScanCursor&&) = delete;

  /// Resolves the encoded key bounds and starts stripe loading. Returns the
  /// selected file-row bounds for the iterator to navigate, or nullopt on a
  /// miss.
  folly::coro::Task<std::optional<RowRange>> co_seek(
      velox::serializer::EncodedKeyBounds bounds);

  /// Starts a scan at the first file row.
  folly::coro::Task<std::optional<RowRange>> co_seekToFirst();

  /// Clears the active scan and its outstanding prefetch.
  folly::coro::Task<void> co_reset();

  /// Returns the projected schema used to decode stripe streams.
  std::shared_ptr<const Type> projectedNimbleType() const;

  /// Returns file decoding options with the caller's scratch-buffer pool.
  /// Keeps consumer decoding scratch separate from the prefetch producer.
  Encoding::Options encodingOptions(velox::BufferPool* bufferPool) const;

  /// Creates a forward key cursor from `row` to the end of the active scan.
  /// The owning iterator retains it alongside its projected data streams.
  std::unique_ptr<index::KeyCursor> keyCursor(uint32_t row) const;

  /// Loads the selected stripe containing the iterator's file-level row.
  /// Skips earlier stripes and clips the selection to this row. The iterator
  /// retains the result while navigating rows within the stripe.
  /// Successive calls must advance to a later stripe.
  folly::coro::Task<ProjectedStripeStreams> co_nextStripe(uint32_t row);

  /// Returns cumulative projector and cursor statistics.
  /// Waits for an active prefetch load before reading projector counters.
  NimbleIterator::Stats stats() const;

 private:
  class ScanPrefetcher;

  // Initializes projector access and the configured scan limits.
  IndexScanCursor(
      std::unique_ptr<NimbleIndexProjector> projector,
      const NimbleIterator::Options& options);

  // Positions the cursor at the start of the resolved stripe selections.
  void setStripeSelections(std::vector<StripeSelection> selections);

  // Starts bounded stripe prefetch for the active scan, when configured.
  void startPrefetch();

  // Returns whether the configured limits allow a non-empty scan to prefetch.
  bool canPrefetch() const;

  // Creates owned cancellation work before clearing the producer.
  folly::coro::Task<void> co_resetPrefetch();

  // Shares the projector with its background prefetch task.
  const ProjectorPtr projector_;
  // Serializes background loads with reads of the projector's plain counters.
  mutable std::mutex lock_;
  // Executor used to project later stripes.
  const folly::Executor::KeepAlive<> prefetchExecutor_;
  // Maximum number of stripe streams queued ahead of the consumer.
  const uint32_t maxPrefetchStripes_;
  // Contiguous file-row range selected by the active key bounds.
  std::optional<RowRange> range_;
  // Selects contiguous stripes and file rows for the active scan. Empty means
  // the scan has no selected rows and cannot prefetch.
  std::vector<StripeSelection> stripeSelections_;
  // First selected stripe not yet returned to the iterator.
  size_t stripeOffset_{0};
  // Rows made available to the iterator from loaded stripe selections.
  uint64_t numOutputRows_{0};
  // Counts prefetched stripe streams discarded after row navigation skips them.
  uint32_t numSkippedStripes_{0};
  // Owns the bounded queue and task that project later stripe streams.
  std::shared_ptr<ScanPrefetcher> scanPrefetcher_;
};

} // namespace detail
} // namespace facebook::nimble
