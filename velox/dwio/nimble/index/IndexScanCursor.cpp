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
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include <folly/ExceptionWrapper.h>
#include <folly/ScopeGuard.h>
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

namespace {

// Reads the keys of the selected rows of one stripe from the key cursor of the
// scan, which then reads on from the first row of the next stripe.
class StripeKeyCursor final : public index::KeyCursor {
 public:
  StripeKeyCursor(index::KeyCursor* scanKeyCursor, uint32_t numRows)
      : scanKeyCursor_{velox::checkedNotNull(scanKeyCursor)},
        numRemaining_{numRows} {}

  bool hasNext() const override {
    return numRemaining_ > 0 && scanKeyCursor_->hasNext();
  }

  std::string_view next() override {
    NIMBLE_CHECK_GT(numRemaining_, 0, "The stripe has no more keys");
    const auto key = scanKeyCursor_->next();
    --numRemaining_;
    return key;
  }

  void next(
      size_t maxKeys,
      std::vector<std::string_view>& keys,
      const BufferAllocator& bufferAllocator) override {
    const auto numKeys = keys.size();
    scanKeyCursor_->next(
        std::min<size_t>(maxKeys, numRemaining_), keys, bufferAllocator);
    numRemaining_ -= static_cast<uint32_t>(keys.size() - numKeys);
  }

  void skip(size_t numKeys) override {
    const auto numSkipped =
        static_cast<uint32_t>(std::min<size_t>(numKeys, numRemaining_));
    scanKeyCursor_->skip(numSkipped);
    numRemaining_ -= numSkipped;
  }

  // Returns the number of keys of the stripe that were not read.
  uint32_t numRemaining() const {
    return numRemaining_;
  }

 private:
  index::KeyCursor* const scanKeyCursor_;
  uint32_t numRemaining_;
};

} // namespace

// Loads and materializes selected stripes in one background task.
class ScanPrefetcher {
 public:
  using ProjectorPtr = IndexScanCursor::ProjectorPtr;
  using MaterializerPtr = std::shared_ptr<IndexScanCursor::StripeMaterializer>;
  using Entries = IndexScanCursor::StripeMaterializer::Entries;
  using EntriesResult = folly::Try<std::unique_ptr<Entries>>;

  ScanPrefetcher(
      ProjectorPtr projector,
      MaterializerPtr materializer,
      std::vector<StripeSelection> selections,
      std::unique_ptr<index::KeyCursor> keyCursor,
      folly::Executor::KeepAlive<> prefetchExecutor,
      uint32_t maxPrefetchStripes)
      : projector_{std::move(projector)},
        materializer_{std::move(materializer)},
        selections_{std::move(selections)},
        keyCursor_{std::move(keyCursor)},
        prefetchExecutor_{std::move(prefetchExecutor)},
        maxPrefetchStripes_{maxPrefetchStripes} {
    NIMBLE_CHECK_GT(maxPrefetchStripes_, 0);
    NIMBLE_CHECK(!selections_.empty(), "No stripes are selected to prefetch");
  }

  ScanPrefetcher(const ScanPrefetcher&) = delete;
  ScanPrefetcher& operator=(const ScanPrefetcher&) = delete;
  ScanPrefetcher(ScanPrefetcher&&) = delete;
  ScanPrefetcher& operator=(ScanPrefetcher&&) = delete;

  ~ScanPrefetcher() = default;

  // Stops admission and wakes the producer waiting for capacity.
  void requestStop() noexcept {
    stopped_.store(true, std::memory_order_release);
    prefetchBaton_.post();
  }

  // Stops loading before joining the task and its decoding children.
  folly::coro::Task<void> cancelAndJoin() noexcept {
    requestStop();
    co_await prefetchScope_.cancelAndJoinAsync();
  }

  void start() {
    prefetchScope_.add(
        folly::coro::co_withExecutor(
            prefetchExecutor_, run(std::move(keyCursor_))));
  }

  // Canceled waits leave entries available for retry.
  folly::coro::Task<std::unique_ptr<Entries>> co_nextStripe(
      uint32_t stripeIndex) {
    checkLoadError();
    velox::common::testutil::TestValue::adjust(
        "facebook::nimble::ScanPrefetcher::co_nextStripe::beforeReady",
        &stripeIndex);
    auto entries = co_await loadedStripes_.dequeue();
    if (entries.hasException()) {
      loadError_ = entries.exception();
      checkLoadError();
    }
    releaseReservation();
    co_return std::move(entries).value();
  }

 private:
  // Makes retries fail promptly after a terminal load or materialization error.
  void checkLoadError() const {
    if (loadError_) {
      loadError_.throw_exception();
    }
  }

  // Reserves a slot before reading or materializing a stripe.
  folly::coro::Task<bool> acquireReservation() {
    while (true) {
      prefetchBaton_.reset();
      if (stopped_.load(std::memory_order_acquire)) {
        co_return false;
      }
      const auto outstanding =
          outstandingStripes_.load(std::memory_order_acquire);
      NIMBLE_CHECK_LE(outstanding, maxPrefetchStripes_);
      if (outstanding == maxPrefetchStripes_) {
        co_await prefetchBaton_;
        continue;
      }
      outstandingStripes_.fetch_add(1, std::memory_order_acq_rel);
      co_return true;
    }
  }

  void releaseReservation() {
    NIMBLE_CHECK_GT(outstandingStripes_.load(std::memory_order_acquire), 0);
    outstandingStripes_.fetch_sub(1, std::memory_order_acq_rel);
    prefetchBaton_.post();
  }

  // Materializes encoded batches serially under the decoded-stripe limit.
  folly::coro::Task<void> run(std::unique_ptr<index::KeyCursor> keyCursor) {
    std::deque<ProjectedStripe> stripeBatch;
    bool reserved{false};
    while (nextStripeOffset_ < selections_.size() &&
           !stopped_.load(std::memory_order_acquire)) {
      const auto& selection = selections_[nextStripeOffset_];
      try {
        reserved = co_await acquireReservation();
        velox::common::testutil::TestValue::adjust(
            "facebook::nimble::ScanPrefetcher::run::afterReservation",
            &stopped_);
        if (!reserved || stopped_.load(std::memory_order_acquire)) {
          break;
        }
        velox::common::testutil::TestValue::adjust(
            "facebook::nimble::ScanPrefetcher::run::beforeLoad", &stopped_);
        if (stopped_.load(std::memory_order_acquire)) {
          break;
        }
        if (stripeBatch.empty()) {
          const auto numStripes = std::min(
              projector_->numStripesToRead(selection.stripeIndex),
              static_cast<uint32_t>(selections_.size() - nextStripeOffset_));
          auto stripes =
              projector_->loadStripeStreams(selection.stripeIndex, numStripes);
          for (auto& stripe : stripes) {
            stripeBatch.push_back(std::move(stripe));
          }
        }
        if (stopped_.load(std::memory_order_acquire)) {
          break;
        }
        auto stripe = std::move(stripeBatch.front());
        stripeBatch.pop_front();
        stripe.selection = selection;
        if (keyCursor == nullptr) {
          keyCursor = projector_->clusterIndex().keyCursor(
              RowRange{
                  selection.selectedRowRange().startRow,
                  selections_.back().selectedRowRange().endRow});
        }
        StripeKeyCursor stripeKeyCursor{
            keyCursor.get(), selection.selectedRowRange().numRows()};
        auto entries = co_await materializer_->materialize(
            std::move(stripe), stripeKeyCursor);
        if (nextStripeOffset_ + 1 == selections_.size()) {
          keyCursor.reset();
        } else {
          stripeKeyCursor.skip(stripeKeyCursor.numRemaining());
        }
        loadedStripes_.enqueue(EntriesResult{std::move(entries)});
      } catch (...) {
        loadedStripes_.enqueue(
            EntriesResult{folly::exception_wrapper{std::current_exception()}});
        break;
      }
      reserved = false;
      ++nextStripeOffset_;
    }
    keyCursor.reset();
    if (reserved) {
      releaseReservation();
    }
  }

  const ProjectorPtr projector_;
  const MaterializerPtr materializer_;
  const std::vector<StripeSelection> selections_;
  // Moves the seek cursor into the producer frame when it starts.
  std::unique_ptr<index::KeyCursor> keyCursor_;
  const folly::Executor::KeepAlive<> prefetchExecutor_;
  const uint32_t maxPrefetchStripes_;
  uint32_t nextStripeOffset_{0};
  // Publishes completed stripe entries in file order.
  folly::coro::UnboundedQueue<
      EntriesResult,
      /*SingleProducer=*/true,
      /*SingleConsumer=*/true>
      loadedStripes_;
  folly::exception_wrapper loadError_;
  // Includes the running stripe and completed entries until consumer handoff.
  std::atomic_uint32_t outstandingStripes_{0};
  std::atomic_bool stopped_{false};
  folly::coro::Baton prefetchBaton_;
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
      // A get reads the rows of one key, which a prefetch would not speed up,
      // and reads the stripe that the materializer holds without loading it.
      maxPrefetchStripes_{options.forGet ? 0 : options.maxPrefetchStripes},
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
  setStripeSelections(projector_->lookup(bounds, keyCursor_));
  startPrefetch();
  return range_;
}

std::optional<RowRange> IndexScanCursor::seek(std::string encodedKey) {
  reset();
  setStripeSelections(projector_->lookup(std::move(encodedKey), keyCursor_));
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
  projector_->resetStreamScan();
  stripeBatch_.clear();
  range_.reset();
  stripeSelections_.clear();
  keyCursor_.reset();
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
  std::unique_ptr<StripeMaterializer::Entries> entries;
  if (scanPrefetcher_ != nullptr) {
    entries = co_await scanPrefetcher_->co_nextStripe(selection.stripeIndex);
  } else {
    try {
      ProjectedStripe stripe;
      // The materializer decodes the selected rows from the streams that it
      // holds. Only a get iterator holds a stripe, and it does not prefetch,
      // which would load the stripe again.
      if (!stripeMaterializer_->hasLoadedStripe(selection.stripeIndex)) {
        if (stripeBatch_.empty()) {
          const auto numStripes = std::min(
              projector_->numStripesToRead(selection.stripeIndex),
              static_cast<uint32_t>(stripeSelections_.size() - stripeOffset_));
          auto stripes =
              projector_->loadStripeStreams(selection.stripeIndex, numStripes);
          for (auto& projectedStripe : stripes) {
            stripeBatch_.push_back(std::move(projectedStripe));
          }
        }
        stripe = std::move(stripeBatch_.front());
        stripeBatch_.pop_front();
      }
      stripe.selection = selection;
      StripeKeyCursor stripeKeyCursor{
          ensureKeyCursor(selection), selection.selectedRowRange().numRows()};
      // A failed stripe can stop the key cursor at any row, so a retry of the
      // stripe starts a new one. Dismissed on success.
      auto resetKeyCursor = folly::makeGuard([this] { keyCursor_.reset(); });
      entries = co_await stripeMaterializer_->materialize(
          std::move(stripe), stripeKeyCursor);
      if (stripeOffset_ + 1 == stripeSelections_.size()) {
        // The last stripe releases the key chunk that the key cursor holds.
        keyCursor_.reset();
      } else {
        // Moves the key cursor of the scan past the keys that the materializer
        // left unread, to the first row of the next stripe.
        stripeKeyCursor.skip(stripeKeyCursor.numRemaining());
      }
      resetKeyCursor.dismiss();
    } catch (...) {
      // Materialization can fail after popping the stripe. Reload it on retry.
      stripeBatch_.clear();
      throw;
    }
  }
  numOutputRows_ += selection.selectedRowRange().numRows();
  ++stripeOffset_;
  co_return entries;
}

index::KeyCursor* IndexScanCursor::ensureKeyCursor(
    const StripeSelection& selection) {
  if (keyCursor_ == nullptr) {
    keyCursor_ = projector_->clusterIndex().keyCursor(
        RowRange{selection.selectedRowRange().startRow, range_->endRow});
  }
  return keyCursor_.get();
}

void IndexScanCursor::startPrefetch() {
  NIMBLE_CHECK_NULL(scanPrefetcher_);
  if (!canPrefetch()) {
    return;
  }
  scanPrefetcher_ = std::make_shared<ScanPrefetcher>(
      projector_,
      stripeMaterializer_,
      stripeSelections_,
      std::move(keyCursor_),
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
  // Keep the prefetcher alive through suspension and nested task allocation.
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
