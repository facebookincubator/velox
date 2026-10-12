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

// Implements batch materialization for cluster-index scans.

#include "velox/dwio/nimble/velox/index/NimbleIterator.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>

#include <folly/ScopeGuard.h>

#include "velox/buffer/Buffer.h"
#include "velox/buffer/BufferPool.h"
#include "velox/common/base/Exceptions.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/serializer/StreamReader.h"
#include "velox/dwio/nimble/velox/index/IndexScanCursor.h"
#include "velox/dwio/nimble/velox/index/NimbleIndexProjector.h"
#include "velox/dwio/nimble/velox/index/StripeStreams.h"

namespace facebook::nimble {
using detail::IndexScanCursor;

namespace {

// Materializes consecutive selected rows into bounded Velox batches.
// Maintains no key cursor and provides no access to the current row key.
class NimbleBatchIterator final : public NimbleIterator {
 public:
  NimbleBatchIterator(
      std::unique_ptr<NimbleIndexProjector> projector,
      velox::memory::MemoryPool* pool,
      NimbleIterator::Options options)
      : pool_{pool},
        options_{std::move(options)},
        cursor_{IndexScanCursor::create(std::move(projector), options_)},
        streamReader_{std::make_unique<StreamReader>(
            cursor_->projectedNimbleType(),
            velox::checkedNotNull(pool_),
            cursor_->encodingOptions(&encodingBufferPool_))} {
    NIMBLE_CHECK_GT(options_.outputBatchRows, 0);
  }

  ~NimbleBatchIterator() override = default;

  folly::coro::Task<void> co_seek(
      velox::serializer::EncodedKeyBounds bounds) override {
    resetState();
    setRowRange(co_await cursor_->co_seek(std::move(bounds)));
  }

  folly::coro::Task<void> co_seekToFirst() override {
    resetState();
    setRowRange(co_await cursor_->co_seekToFirst());
  }

  folly::coro::Task<void> co_reset() override {
    resetState();
    co_await cursor_->co_reset();
  }

  folly::coro::Task<void> co_nextRow() override {
    NIMBLE_UNSUPPORTED("Batch iterator cannot produce serialized row output");
    co_return;
  }

  folly::coro::Task<bool> co_nextBatch(velox::RowVectorPtr& output) override {
    if (!hasCurrentRow()) {
      output.reset();
      co_return false;
    }

    const auto row = currentRow();
    co_await ensureStripe(row);
    const auto& selection = stripeStreams_->selection;
    const auto numRows = std::min(
        options_.outputBatchRows,
        selection.startRow + selection.rowRange.endRow - row);
    // TODO: Benchmark Deserializer batch decoding against StreamReader's
    // encoding views.
    {
      velox::CpuWallTimer timer{materializationTiming_};
      velox::VectorPtr decodedRows{std::move(output)};
      SCOPE_EXIT {
        output =
            std::static_pointer_cast<velox::RowVector>(std::move(decodedRows));
      };
      co_await streamReader_->co_next(numRows, decodedRows);
    }
    advanceRows(numRows);
    co_return true;
  }

  folly::coro::Task<std::string_view> co_value() override {
    NIMBLE_UNSUPPORTED("Batch iterator does not expose serialized row values");
    co_return std::string_view{};
  }

  std::string_view key() const override {
    NIMBLE_UNSUPPORTED("Batch iterator does not expose the current row key");
  }

  uint32_t rowNumber() const override {
    return currentRow();
  }

  NimbleIterator::Stats stats() const override {
    auto result = cursor_->stats();
    result.materializationTiming = materializationTiming_;
    return result;
  }

 private:
  // Invalidates row navigation before asynchronous seek or reset can fail.
  void resetState() {
    clearStripe();
    setRowRange(std::nullopt);
  }

  // Releases decoded views before their stripe storage.
  void clearStripe() {
    if (streamReader_ != nullptr) {
      streamReader_->reset();
    }
    stripeStreams_.reset();
  }

  // Reuses the current stripe or binds the stripe containing the selected row.
  folly::coro::Task<void> ensureStripe(uint32_t row) {
    NIMBLE_DCHECK_EQ(row, currentRow());
    if (stripeStreams_.has_value()) {
      const auto& selection = stripeStreams_->selection;
      const auto selectedStartRow =
          selection.startRow + selection.rowRange.startRow;
      const auto selectedEndRow =
          selection.startRow + selection.rowRange.endRow;
      NIMBLE_CHECK_GE(row, selectedStartRow);
      NIMBLE_CHECK_LE(row, selectedEndRow);
      if (row < selectedEndRow) {
        co_return;
      }
    }
    clearStripe();
    auto stripeStreams = co_await cursor_->co_nextStripe(row);
    const auto& selection = stripeStreams.selection;
    NIMBLE_CHECK_EQ(
        selection.startRow + selection.rowRange.startRow,
        row,
        "Prefetched stripe streams do not match the iterator position");
    stripeStreams_ = std::move(stripeStreams);
    streamReader_->setStreams(stripeStreams_->streams);
    if (stripeStreams_->selection.rowRange.startRow > 0) {
      co_await streamReader_->co_skip(
          stripeStreams_->selection.rowRange.startRow);
    }
    co_return;
  }

  // Owns allocations made while decoding output batches.
  velox::memory::MemoryPool* const pool_;
  // Controls output batch size and stripe-stream prefetch.
  const NimbleIterator::Options options_;
  // Shares stripe-loading state with active prefetch operations.
  const std::shared_ptr<IndexScanCursor> cursor_;
  // Reuses scratch buffers created by encoding views.
  velox::BufferPool encodingBufferPool_{velox::BufferPool::kDefaultCapacity};
  // Decodes the currently bound stripe into output vectors.
  const std::unique_ptr<StreamReader> streamReader_;
  // Keeps the current stripe streams alive while they are decoded.
  std::optional<ProjectedStripeStreams> stripeStreams_;
  // Tracks vector decoding time separately from stripe projection time.
  velox::CpuWallTiming materializationTiming_{};
};

} // namespace

std::unique_ptr<NimbleIterator> NimbleIterator::createBatchIterator(
    std::unique_ptr<NimbleIndexProjector> projector,
    velox::memory::MemoryPool* pool,
    Options options) {
  return std::make_unique<NimbleBatchIterator>(
      std::move(projector), pool, std::move(options));
}

} // namespace facebook::nimble
