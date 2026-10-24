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

// Implements indexed positioning and bounded stripe-stream prefetch.

#include "velox/dwio/nimble/index/IndexScanCursor.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include <folly/ExceptionWrapper.h>
#include <folly/Try.h>
#include <folly/coro/AsyncScope.h>
#include <folly/coro/Baton.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Invoke.h>
#include <folly/coro/Task.h>
#include <folly/coro/UnboundedQueue.h>

#include "velox/common/Casts.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/NimbleIndexProjector.h"

namespace facebook::nimble::detail {

// Prefetches contiguous stripe batches within the configured stripe limit.
class ScanPrefetcher {
 public:
  using ProjectorPtr = IndexScanCursor::ProjectorPtr;

  // Starts prefetching the stripes selected for the active scan.
  ScanPrefetcher(
      ProjectorPtr projector,
      uint32_t stripeOffset,
      uint32_t numStripes,
      folly::Executor::KeepAlive<> prefetchExecutor,
      uint32_t maxPrefetchStripes)
      : projector_{std::move(projector)},
        stripeOffset_{stripeOffset},
        numStripes_{numStripes},
        prefetchExecutor_{std::move(prefetchExecutor)},
        maxPrefetchStripes_{maxPrefetchStripes},
        maxBatchStripes_{std::max<uint32_t>(1, maxPrefetchStripes / 2)} {
    NIMBLE_CHECK_GT(maxPrefetchStripes_, 0);
  }

  ScanPrefetcher(const ScanPrefetcher&) = delete;
  ScanPrefetcher& operator=(const ScanPrefetcher&) = delete;
  ScanPrefetcher(ScanPrefetcher&&) = delete;
  ScanPrefetcher& operator=(ScanPrefetcher&&) = delete;

  ~ScanPrefetcher() = default;

  // Stops queue admission and wakes a producer waiting for capacity.
  void requestStop() noexcept {
    stopped_.store(true, std::memory_order_release);
    prefetchBaton_.post();
  }

  // Stops queue admission and waits for the producer to finish.
  folly::coro::Task<void> cancelAndJoin() noexcept {
    requestStop();
    co_await prefetchScope_.cancelAndJoinAsync();
  }

  // Schedules the producer once on the configured executor.
  void start() {
    if (prefetchStarted_.exchange(true)) {
      return;
    }
    prefetchScope_.add(folly::coro::co_withExecutor(prefetchExecutor_, run()));
  }

  // Waits for and returns the next queued stripe's streams.
  folly::coro::Task<ProjectedStripe> co_nextStripe() {
    checkLoadError();
    if (currentBatchStripe_ == currentBatch_.size()) {
      folly::Try<std::vector<ProjectedStripe>> stripesOr;
      co_await loadedStripes_.dequeue(stripesOr);
      if (stripesOr.hasException()) {
        loadError_ = stripesOr.exception();
        checkLoadError();
      }
      currentBatch_ = std::move(stripesOr).value();
      NIMBLE_CHECK(!currentBatch_.empty(), "Loaded prefetch batch is empty");
      currentBatchStripe_ = 0;
    }
    auto stripe = std::move(currentBatch_.at(currentBatchStripe_++));
    if (currentBatchStripe_ == currentBatch_.size()) {
      releaseReservation(static_cast<uint32_t>(currentBatch_.size()));
    }
    co_return std::move(stripe);
  }

 private:
  // Makes consumer retries fail promptly after a terminal load failure.
  void checkLoadError() const {
    if (loadError_) {
      loadError_.throw_exception();
    }
  }

  // Reserves a batch before issuing a coalesced load.
  folly::coro::Task<uint32_t> acquireReservation(uint32_t maxStripes) {
    NIMBLE_CHECK_GT(maxStripes, 0);
    while (true) {
      prefetchBaton_.reset();
      if (stopped_.load(std::memory_order_acquire)) {
        co_return 0;
      }
      const auto queuedStripes =
          consumedStripes_.load(std::memory_order_acquire);
      NIMBLE_CHECK_LE(queuedStripes, maxPrefetchStripes_);
      if (queuedStripes == maxPrefetchStripes_) {
        co_await prefetchBaton_;
        continue;
      }
      const auto numStripes =
          std::min(maxStripes, maxPrefetchStripes_ - queuedStripes);
      consumedStripes_.fetch_add(numStripes, std::memory_order_acq_rel);
      co_return numStripes;
    }
  }

  // Releases the batch after its last stripe is dequeued or its load fails.
  void releaseReservation(uint32_t numStripes) {
    if (numStripes == 0) {
      return;
    }
    NIMBLE_CHECK_GE(
        consumedStripes_.load(std::memory_order_acquire), numStripes);
    consumedStripes_.fetch_sub(numStripes, std::memory_order_acq_rel);
    prefetchBaton_.post();
  }

  // Loads consecutive full stripes and enqueues them for the consumer.
  folly::coro::Task<void> run() {
    while (nextStripeOffset_ < numStripes_ &&
           !stopped_.load(std::memory_order_acquire)) {
      uint32_t reservedStripes{0};
      std::vector<ProjectedStripe> stripeStreams;
      try {
        const auto numStripes =
            std::min(maxBatchStripes_, numStripes_ - nextStripeOffset_);
        NIMBLE_CHECK_GT(numStripes, 0);
        reservedStripes = co_await acquireReservation(numStripes);
        velox::common::testutil::TestValue::adjust(
            "facebook::nimble::ScanPrefetcher::run::afterReservation",
            &stopped_);
        if (reservedStripes == 0 || stopped_.load(std::memory_order_acquire)) {
          releaseReservation(reservedStripes);
          co_return;
        }
        velox::common::testutil::TestValue::adjust(
            "facebook::nimble::ScanPrefetcher::run::beforeLoad", &stopped_);
        if (stopped_.load(std::memory_order_acquire)) {
          // The consumer can be destroyed before queued cancellation starts.
          releaseReservation(reservedStripes);
          co_return;
        }
        stripeStreams = projector_->loadStripeStreams(
            stripeOffset_ + nextStripeOffset_, reservedStripes);
        NIMBLE_CHECK_EQ(stripeStreams.size(), reservedStripes);
      } catch (...) {
        releaseReservation(reservedStripes);
        loadedStripes_.enqueue(
            folly::Try<std::vector<ProjectedStripe>>{
                folly::exception_wrapper{std::current_exception()}});
        co_return;
      }
      loadedStripes_.enqueue(
          folly::Try<std::vector<ProjectedStripe>>{std::move(stripeStreams)});
      nextStripeOffset_ += reservedStripes;
    }
  }

  // Keeps projector resources alive through detached producer cleanup.
  const ProjectorPtr projector_;
  // First source stripe selected by the consumer's active scan.
  const uint32_t stripeOffset_;
  // Total source stripes selected for the active scan.
  const uint32_t numStripes_;
  // Keeps the background executor alive while this producer is active.
  const folly::Executor::KeepAlive<> prefetchExecutor_;
  // Maximum stripe streams queued ahead of the consumer.
  // TODO: Add a limit on projected stripe bytes.
  const uint32_t maxPrefetchStripes_;
  // Targets two batches so loading can overlap consumption.
  const uint32_t maxBatchStripes_;
  // First selected stripe that the producer has not loaded yet.
  uint32_t nextStripeOffset_{0};
  // One prefetch coroutine produces batches; serialized iterator calls consume
  // them.
  folly::coro::UnboundedQueue<
      folly::Try<std::vector<ProjectedStripe>>,
      /*SingleProducer=*/true,
      /*SingleConsumer=*/true>
      loadedStripes_;
  // Holds the current batch until the consumer dequeues its final stripe.
  std::vector<ProjectedStripe> currentBatch_;
  // Next stripe to return from the consumer's current batch.
  size_t currentBatchStripe_{0};
  // Retains the terminal producer failure for serialized consumer calls.
  folly::exception_wrapper loadError_;
  // Ensures the producer is scheduled only once.
  std::atomic_bool prefetchStarted_{false};
  // Stripes reserved by batches that have not been completely dequeued.
  std::atomic_uint32_t consumedStripes_{0};
  // Stops the producer while it waits for a queue limit.
  std::atomic_bool stopped_{false};
  // Wakes the producer when the consumer releases a batch reservation.
  folly::coro::Baton prefetchBaton_;
  // Owns the producer until cancellation and join complete.
  folly::coro::CancellableAsyncScope prefetchScope_;
};

std::shared_ptr<IndexScanCursor> IndexScanCursor::create(
    std::unique_ptr<NimbleIndexProjector> projector,
    const NimbleIterator::Options& options) {
  return std::shared_ptr<IndexScanCursor>{
      new IndexScanCursor{std::move(projector), options}};
}

IndexScanCursor::IndexScanCursor(
    std::unique_ptr<NimbleIndexProjector> projector,
    const NimbleIterator::Options& options)
    : projector_{velox::checkedNotNull(std::move(projector))},
      prefetchExecutor_{options.prefetchExecutor},
      maxPrefetchStripes_{options.maxPrefetchStripes},
      cacheEncodingView_{options.cacheEncodingView} {
  NIMBLE_CHECK_LE(projector_->numRows(), std::numeric_limits<uint32_t>::max());
  projector_->setTimingEnabled(options.collectTimings);
}

IndexScanCursor::~IndexScanCursor() {
  // A cursor with prefetch disabled can have no executor for detached cleanup.
  if (scanPrefetcher_ == nullptr) {
    return;
  }
  const auto scanPrefetcher = scanPrefetcher_;
  scanPrefetcher->requestStop();
  try {
    // Schedules with add() because TaskWithExecutor::start() calls add() from a
    // noexcept await_suspend, which terminates on a scheduling failure instead
    // of reaching the handler below.
    prefetchExecutor_->add(
        [scanPrefetcher,
         cleanup = folly::coro::co_withExecutor(
             prefetchExecutor_, co_resetPrefetch())]() mutable {
          std::move(cleanup).startInlineUnsafe(
              [scanPrefetcher](folly::Try<void>&& result) noexcept {
                if (result.hasException()) {
                  LOG(ERROR)
                      << "Failed to complete asynchronous prefetch cleanup: "
                      << result.exception().what();
                  folly::coro::blockingWait(scanPrefetcher->cancelAndJoin());
                }
              });
        });
  } catch (...) {
    LOG(ERROR) << "Failed to start asynchronous prefetch cleanup: "
               << folly::exception_wrapper{std::current_exception()}.what();
    folly::coro::blockingWait(scanPrefetcher->cancelAndJoin());
  }
}

std::optional<RowRange> IndexScanCursor::seek(
    const velox::serializer::EncodedKeyBounds& bounds) {
  reset();
  setStripeSelections(projector_->lookup(bounds));
  startPrefetch();
  return range_;
}

std::optional<RowRange> IndexScanCursor::seek(std::string encodedKey) {
  reset();
  setStripeSelections(projector_->lookup(std::move(encodedKey)));
  startPrefetch();
  return range_;
}

std::optional<RowRange> IndexScanCursor::seekToFirst() {
  reset();
  setStripeSelections(projector_->lookup(
      RowRange{0, static_cast<uint32_t>(projector_->numRows())}));
  startPrefetch();
  return range_;
}

void IndexScanCursor::reset() {
  // Skips the wait for cursors without a running prefetch, such as the ones
  // that serve point lookups.
  if (scanPrefetcher_ != nullptr) {
    folly::coro::blockingWait(co_resetPrefetch());
  }
  range_.reset();
  stripeSelections_.clear();
  stripeOffset_ = 0;
}

void IndexScanCursor::setStripeSelections(
    std::vector<StripeSelection> selections) {
  NIMBLE_CHECK(
      stripeSelections_.empty(),
      "Stripe selections must be cleared before selecting stripes");
  if (selections.empty()) {
    return;
  }
  NIMBLE_CHECK_LE(selections.size(), std::numeric_limits<uint32_t>::max());
  for (size_t stripeOffset = 1; stripeOffset < selections.size();
       ++stripeOffset) {
    const auto& previous = selections[stripeOffset - 1];
    const auto& selection = selections[stripeOffset];
    NIMBLE_CHECK_EQ(
        static_cast<uint64_t>(previous.stripeIndex) + 1,
        selection.stripeIndex,
        "Selected stripe indices must be consecutive");
    NIMBLE_CHECK_EQ(
        previous.selectedRowRange().endRow,
        selection.selectedRowRange().startRow,
        "Selected row ranges must be contiguous");
  }
  range_ = RowRange{
      selections.front().selectedRowRange().startRow,
      selections.back().selectedRowRange().endRow};
  stripeSelections_ = std::move(selections);
}

std::shared_ptr<const Type> IndexScanCursor::projectedNimbleType() const {
  return projector_->projectedNimbleType();
}

Encoding::Options IndexScanCursor::encodingOptions(
    velox::BufferPool* bufferPool) const {
  return Encoding::Options{
      .useVarintRowCount = projector_->properties().compactRowCountEncoding(),
      .bufferPool = bufferPool,
      .cacheEncodingView = cacheEncodingView_,
  };
}

std::unique_ptr<index::KeyCursor> IndexScanCursor::keyCursor(
    uint32_t row) const {
  NIMBLE_CHECK(range_.has_value(), "No index scan is active");
  NIMBLE_CHECK(
      range_->contains(row),
      "Row is outside the active scan: row {}, range {}",
      row,
      range_->toString());
  return projector_->clusterIndex().keyCursor(RowRange{row, range_->endRow});
}

folly::coro::Task<ProjectedStripe> IndexScanCursor::co_nextStripe(
    uint32_t row) {
  NIMBLE_CHECK(range_.has_value(), "No index scan is active");
  NIMBLE_CHECK(
      range_->contains(row),
      "Row is outside the active scan: row {}, range {}",
      row,
      range_->toString());
  while (stripeOffset_ < stripeSelections_.size()) {
    const auto& selection = stripeSelections_[stripeOffset_];
    if (row < selection.selectedRowRange().endRow) {
      break;
    }
    ++stripeOffset_;
  }
  NIMBLE_CHECK_LT(
      stripeOffset_, stripeSelections_.size(), "No selected stripes remain");
  const auto& selection = stripeSelections_[stripeOffset_];
  NIMBLE_CHECK(
      selection.contains(row),
      "Row is outside the selected stripe: row {}, selection {}",
      row,
      selection);
  ProjectedStripe stripeStreams;
  if (scanPrefetcher_ == nullptr) {
    auto stripes =
        projector_->loadStripeStreams(selection.stripeIndex, /*numStripes=*/1);
    NIMBLE_CHECK_EQ(stripes.size(), 1);
    stripeStreams = std::move(stripes.at(0));
  } else {
    // Discard loaded stripes skipped by row iteration.
    while (true) {
      stripeStreams = co_await scanPrefetcher_->co_nextStripe();
      if (stripeStreams.selection.stripeIndex >= selection.stripeIndex) {
        break;
      }
      ++numSkippedStripes_;
    }
  }

  auto& loadedSelection = stripeStreams.selection;
  NIMBLE_CHECK_EQ(
      loadedSelection.stripeIndex,
      selection.stripeIndex,
      "Loaded stripe index does not match the selected stripe");
  loadedSelection.rowRange =
      RowRange{row - selection.startRow, selection.rowRange.endRow};
  numOutputRows_ += loadedSelection.rowRange.numRows();
  ++stripeOffset_;
  co_return stripeStreams;
}

void IndexScanCursor::startPrefetch() {
  NIMBLE_CHECK_NULL(scanPrefetcher_);
  if (!canPrefetch()) {
    return;
  }
  scanPrefetcher_ = std::make_shared<ScanPrefetcher>(
      projector_,
      stripeSelections_.front().stripeIndex,
      static_cast<uint32_t>(stripeSelections_.size()),
      prefetchExecutor_,
      maxPrefetchStripes_);
  scanPrefetcher_->start();
}

bool IndexScanCursor::canPrefetch() const {
  return !stripeSelections_.empty() && prefetchExecutor_ &&
      maxPrefetchStripes_ > 0;
}

folly::coro::Task<void> IndexScanCursor::co_resetPrefetch() {
  if (scanPrefetcher_ == nullptr) {
    return folly::coro::makeTask();
  }
  // Keep the producer alive through suspension and nested task allocation.
  auto cleanup =
      folly::coro::co_invoke(&ScanPrefetcher::cancelAndJoin, scanPrefetcher_);
  scanPrefetcher_.reset();
  return cleanup;
}

NimbleIterator::Stats IndexScanCursor::stats() const {
  const auto projectorStats = projector_->stats();
  return {
      .numReadStripes = projectorStats.numReadStripes,
      .numSkippedStripes = numSkippedStripes_,
      .numSlicedStripes = projectorStats.numSlicedStripes,
      .numReadRows = projectorStats.numReadRows,
      .numOutputRows = numOutputRows_,
      .numOutputBytes = projectorStats.numOutputBytes,
      .numPlannedBytes = projectorStats.numPlannedBytes,
      .lookupTiming = projectorStats.lookupTiming,
      .prepareTiming = projectorStats.prepareTiming,
      .scanTiming = projectorStats.scanTiming,
      .projectionTiming = projectorStats.projectionTiming,
  };
}

} // namespace facebook::nimble::detail
