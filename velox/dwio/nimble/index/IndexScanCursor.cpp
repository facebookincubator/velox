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

// Loads encoded streams in contiguous stripe batches within the configured
// stripe limit. The consumer materializes them.
class ScanPrefetcher {
 public:
  using ProjectorPtr = IndexScanCursor::ProjectorPtr;

  // Starts prefetching the stripes selected for the active scan.
  ScanPrefetcher(
      ProjectorPtr projector,
      std::vector<StripeSelection> selections,
      folly::Executor::KeepAlive<> prefetchExecutor,
      uint32_t maxPrefetchStripes)
      : projector_{std::move(projector)},
        selections_{std::move(selections)},
        prefetchExecutor_{std::move(prefetchExecutor)},
        maxPrefetchStripes_{maxPrefetchStripes},
        maxBatchStripes_{std::max<uint32_t>(1, maxPrefetchStripes / 2)} {
    NIMBLE_CHECK_GT(maxPrefetchStripes_, 0);
    NIMBLE_CHECK(!selections_.empty(), "No stripes are selected to prefetch");
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

  // Waits for the next queued stripe, which is 'stripeIndex', and returns its
  // encoded streams for the consumer to materialize.
  folly::coro::Task<ProjectedStripe> co_nextStripe(uint32_t stripeIndex) {
    auto stripe = co_await co_dequeueStripe();
    NIMBLE_CHECK_EQ(
        stripe.selection.stripeIndex,
        stripeIndex,
        "Loaded stripe index does not match the selected stripe");
    co_return std::move(stripe);
  }

 private:
  // Waits for and returns the next queued stripe.
  folly::coro::Task<ProjectedStripe> co_dequeueStripe() {
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

  // Releases a batch's slots after its last stripe is dequeued or loading
  // stops.
  void releaseReservation(uint32_t numStripes) {
    if (numStripes == 0) {
      return;
    }
    NIMBLE_CHECK_GE(
        consumedStripes_.load(std::memory_order_acquire), numStripes);
    consumedStripes_.fetch_sub(numStripes, std::memory_order_acq_rel);
    prefetchBaton_.post();
  }

  // Loads encoded stripe batches for the consumer to materialize.
  folly::coro::Task<void> run() {
    // Stripes reserved for a batch that the consumer has not received, which
    // the producer releases when it stops.
    uint32_t reservedStripes{0};
    while (nextStripeOffset_ < selections_.size() &&
           !stopped_.load(std::memory_order_acquire)) {
      std::vector<ProjectedStripe> stripeStreams;
      try {
        const auto numStripes = std::min<uint32_t>(
            maxBatchStripes_,
            static_cast<uint32_t>(selections_.size() - nextStripeOffset_));
        NIMBLE_CHECK_GT(numStripes, 0);
        reservedStripes = co_await acquireReservation(numStripes);
        velox::common::testutil::TestValue::adjust(
            "facebook::nimble::ScanPrefetcher::run::afterReservation",
            &stopped_);
        if (reservedStripes == 0 || stopped_.load(std::memory_order_acquire)) {
          break;
        }
        velox::common::testutil::TestValue::adjust(
            "facebook::nimble::ScanPrefetcher::run::beforeLoad", &stopped_);
        if (stopped_.load(std::memory_order_acquire)) {
          // The consumer can be destroyed before queued cancellation starts.
          break;
        }
        const auto firstStripe =
            selections_.front().stripeIndex + nextStripeOffset_;
        stripeStreams =
            projector_->loadStripeStreams(firstStripe, reservedStripes);
        NIMBLE_CHECK_EQ(stripeStreams.size(), reservedStripes);
      } catch (...) {
        loadedStripes_.enqueue(
            folly::Try<std::vector<ProjectedStripe>>{
                folly::exception_wrapper{std::current_exception()}});
        break;
      }
      nextStripeOffset_ += reservedStripes;
      reservedStripes = 0;
      loadedStripes_.enqueue(
          folly::Try<std::vector<ProjectedStripe>>{std::move(stripeStreams)});
    }
    releaseReservation(reservedStripes);
  }

  // Keeps projector resources alive through detached producer cleanup.
  const ProjectorPtr projector_;
  // The stripes selected by the consumer's active scan, which are consecutive.
  const std::vector<StripeSelection> selections_;
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
  // Retains producer failures, including cancellation, across consumer retries.
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

IndexScanCursor::StripeMaterializer::Entries::Entries(
    folly::IOBufFactory& iobufFactory) {
  keyStorage.setIOBufFactory(&iobufFactory);
  valueStorage.setIOBufFactory(&iobufFactory);
}

std::shared_ptr<IndexScanCursor> IndexScanCursor::create(
    std::unique_ptr<NimbleIndexProjector> projector,
    std::shared_ptr<StripeMaterializer> materializer,
    const NimbleIterator::Options& options) {
  return std::shared_ptr<IndexScanCursor>{new IndexScanCursor{
      std::move(projector), std::move(materializer), options}};
}

Encoding::Options IndexScanCursor::encodingOptions(
    const NimbleIndexProjector& projector,
    bool cacheEncodingView,
    velox::BufferPool* bufferPool) {
  return Encoding::Options{
      .useVarintRowCount = projector.properties().compactRowCountEncoding(),
      .bufferPool = bufferPool,
      .cacheEncodingView = cacheEncodingView,
  };
}

IndexScanCursor::IndexScanCursor(
    std::unique_ptr<NimbleIndexProjector> projector,
    std::shared_ptr<StripeMaterializer> materializer,
    const NimbleIterator::Options& options)
    : projector_{velox::checkedNotNull(std::move(projector))},
      prefetchExecutor_{options.prefetchExecutor},
      maxPrefetchStripes_{options.maxPrefetchStripes},
      stripeMaterializer_{velox::checkedNotNull(std::move(materializer))} {
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

void IndexScanCursor::setVerifyStreamChecksums(bool verifyStreamChecksums) {
  NIMBLE_CHECK(
      stripeSelections_.empty(),
      "Stream checksum verification must not change during a scan");
  projector_->setVerifyStreamChecksums(verifyStreamChecksums);
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

const StripeSelection& IndexScanCursor::selectStripe(uint32_t row) const {
  NIMBLE_CHECK(range_.has_value(), "No index scan is active");
  NIMBLE_CHECK(
      range_->contains(row),
      "Row is outside the active scan: row {}, range {}",
      row,
      range_->toString());
  NIMBLE_CHECK_LT(
      stripeOffset_, stripeSelections_.size(), "No selected stripes remain");
  const auto& selection = stripeSelections_[stripeOffset_];
  NIMBLE_CHECK_EQ(
      row,
      selection.selectedRowRange().startRow,
      "Row is not the first selected row of the next stripe");
  return selection;
}

folly::coro::Task<std::unique_ptr<IndexScanCursor::StripeMaterializer::Entries>>
IndexScanCursor::co_nextStripe(uint32_t row) {
  const auto& selection = selectStripe(row);
  ProjectedStripe stripe;
  if (scanPrefetcher_ == nullptr) {
    auto stripes =
        projector_->loadStripeStreams(selection.stripeIndex, /*numStripes=*/1);
    NIMBLE_CHECK_EQ(stripes.size(), 1);
    stripe = std::move(stripes.at(0));
  } else {
    stripe = co_await scanPrefetcher_->co_nextStripe(selection.stripeIndex);
  }
  stripe.selection.rowRange = selection.rowRange;
  auto entries = co_await stripeMaterializer_->materialize(
      std::move(stripe), projector_->clusterIndex());
  numOutputRows_ += selection.selectedRowRange().numRows();
  ++stripeOffset_;
  co_return entries;
}

void IndexScanCursor::startPrefetch() {
  NIMBLE_CHECK_NULL(scanPrefetcher_);
  if (!canPrefetch()) {
    return;
  }
  scanPrefetcher_ = std::make_shared<ScanPrefetcher>(
      projector_, stripeSelections_, prefetchExecutor_, maxPrefetchStripes_);
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
