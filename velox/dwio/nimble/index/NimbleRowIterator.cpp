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

// Implements the row iterator, which decodes the rows of index scans into
// output entries.

#include "velox/dwio/nimble/index/NimbleIterator.h"

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Task.h>
#include <folly/io/IOBuf.h>
#include <folly/io/IOBufQueue.h>

#include "velox/buffer/BufferPool.h"
#include "velox/common/Casts.h"
#include "velox/common/base/Exceptions.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/SchemaUtils.h"
#include "velox/dwio/nimble/index/ClusterIndexBase.h"
#include "velox/dwio/nimble/index/IndexScanCursor.h"
#include "velox/dwio/nimble/index/KeyCursor.h"
#include "velox/dwio/nimble/index/NimbleIndexProjector.h"
#include "velox/dwio/nimble/index/StreamReaderCache.h"
#include "velox/dwio/nimble/serializer/StreamReader.h"

namespace facebook::nimble {
using detail::IndexScanCursor;

namespace {

// Allocates the chunks of an IOBufQueue from a memory pool. The pool must
// outlive the factory and every IOBuf that it allocated.
folly::IOBufFactory createIOBufFactory(velox::memory::MemoryPool& pool) {
  NIMBLE_CHECK(pool.isLeaf(), "Memory pool must be a leaf");
  return [pool = &pool](size_t capacity) {
    if (capacity == 0) {
      return folly::IOBuf::create(0);
    }

    auto freeBuffer = [pool, capacity](uint8_t* buffer) {
      pool->free(buffer, static_cast<int64_t>(capacity));
    };
    auto data = std::unique_ptr<uint8_t, decltype(freeBuffer)>{
        static_cast<uint8_t*>(pool->allocate(static_cast<int64_t>(capacity))),
        std::move(freeBuffer)};
    auto buffer = folly::IOBuf::takeOwnership(std::move(data), capacity);
    if (buffer == nullptr) {
      NIMBLE_FAIL("IOBuf allocation returned null");
    }
    buffer->trimEnd(capacity);
    return buffer;
  };
}

// Decodes a stripe's selected rows into a vector and converts them to entries
// on the consumer thread. The materializer keeps one stream reader for its
// lifetime and returns it to the stream reader cache of the options when it is
// destroyed, so that the iterators that share the cache, such as those of a
// file, reuse the readers. Without a cache, it builds its reader on the pool of
// the iterator and destroys it with itself.
class StripeMaterializer final : public IndexScanCursor::StripeMaterializer {
 public:
  // Decodes the stripes with a stream reader of 'schema' from
  // 'streamReaderCache', or with one that it builds when the cache is null or
  // has no idle reader.
  StripeMaterializer(
      std::shared_ptr<const Type> schema,
      velox::memory::MemoryPool* pool,
      Encoding::Options encodingOptions,
      NimbleIterator::EntryDecoder entryDecoder,
      std::shared_ptr<StreamReaderCache> streamReaderCache,
      bool collectTimings)
      : schema_{std::move(schema)},
        pool_{velox::checkedNotNull(pool)},
        encodingOptions_{std::move(encodingOptions)},
        entryDecoder_{std::move(entryDecoder)},
        streamReaderCache_{std::move(streamReaderCache)},
        collectTimings_{collectTimings},
        iobufFactory_{createIOBufFactory(*pool_)} {}

  StripeMaterializer(const StripeMaterializer&) = delete;
  StripeMaterializer& operator=(const StripeMaterializer&) = delete;
  StripeMaterializer(StripeMaterializer&&) = delete;
  StripeMaterializer& operator=(StripeMaterializer&&) = delete;

  // Returns the reader to the cache, for later iterators.
  ~StripeMaterializer() override {
    if (streamReaderCache_ != nullptr && cachedStreamReader_ != nullptr) {
      streamReaderCache_->release(std::move(cachedStreamReader_));
    }
  }

  folly::coro::Task<std::unique_ptr<Entries>> materialize(
      ProjectedStripe stripe,
      const index::ClusterIndexBase& clusterIndex) override {
    const auto startRow = stripe.selection.selectedRowRange().startRow;
    auto rows = co_await co_decode(std::move(stripe));
    co_return convert(std::move(rows), startRow, clusterIndex);
  }

  uint64_t numMaterializedRows() const {
    return numMaterializedRows_.load(std::memory_order_relaxed);
  }

  velox::CpuWallTiming decodingTiming() const {
    return *decodingTiming_.rlock();
  }

  velox::CpuWallTiming conversionTiming() const {
    return *conversionTiming_.rlock();
  }

 private:
  // Binds the streams of 'stripe' to the stream reader of the materializer,
  // with the decoding options of the file and the scratch-buffer pool of the
  // reader, so that the reader recycles its scratch buffers in that pool. The
  // first stripe gets the reader, and the later stripes reuse it. Sets
  // streamReader_ to the reader.
  void ensureStripeReader(const ProjectedStripe& stripe) {
    if (cachedStreamReader_ == nullptr) {
      cachedStreamReader_ = getStreamReader();
      streamReader_ = cachedStreamReader_->streamReader.get();
    }
    auto options = encodingOptions_;
    options.bufferPool = &cachedStreamReader_->bufferPool;
    streamReader_->setStreams(stripe.streams, options);
  }

  // Returns an idle stream reader of the cache, or a new one when the cache is
  // null or has none. A new reader for the cache allocates from a leaf pool of
  // its own, so that it does not wait on the readers of other iterators.
  // Without a cache, the reader allocates from the pool of the iterator.
  std::unique_ptr<StreamReaderCache::Entry> getStreamReader() const {
    if (streamReaderCache_ == nullptr) {
      return makeStreamReader(pool_->shared_from_this());
    }
    if (auto entry = streamReaderCache_->get()) {
      return entry;
    }
    return makeStreamReader(streamReaderCache_->addReaderPool());
  }

  // Returns a new stream reader of the schema that allocates from 'pool'.
  std::unique_ptr<StreamReaderCache::Entry> makeStreamReader(
      std::shared_ptr<velox::memory::MemoryPool> pool) const {
    auto streamReader = std::make_unique<StreamReader>(schema_, pool.get());
    return std::make_unique<StreamReaderCache::Entry>(
        std::move(pool), std::move(streamReader));
  }

  // Decodes the rows that the selection of 'stripe' holds, and drops the stripe
  // when it returns.
  folly::coro::Task<velox::RowVectorPtr> co_decode(ProjectedStripe stripe) {
    const auto& selection = stripe.selection;
    const auto numRows = selection.rowRange.numRows();
    NIMBLE_CHECK_GT(numRows, 0);
    ensureStripeReader(stripe);

    // Unbinds the reader from the streams of the stripe, which the stripe
    // frees, and keeps it for the next stripe.
    SCOPE_EXIT {
      streamReader_->reset();
    };
    if (selection.rowRange.startRow > 0) {
      co_await streamReader_->co_skip(selection.rowRange.startRow);
    }

    // Times the decoding of the stripe.
    velox::OptionalCpuWallTimer timer{
        collectTimings_, [this](const velox::CpuWallTiming& timing) {
          decodingTiming_.wlock()->add(timing);
        }};
    velox::VectorPtr rows;
    co_await streamReader_->co_next(numRows, rows);
    NIMBLE_CHECK_NOT_NULL(rows);
    NIMBLE_CHECK_EQ(rows->size(), numRows);
    numMaterializedRows_.fetch_add(numRows, std::memory_order_relaxed);
    co_return std::static_pointer_cast<velox::RowVector>(std::move(rows));
  }

  // Decodes the entries of decoded rows that start at the file row 'startRow'.
  std::unique_ptr<Entries> convert(
      velox::RowVectorPtr rows,
      uint32_t startRow,
      const index::ClusterIndexBase& clusterIndex) {
    // Times the conversion of the stripe.
    velox::OptionalCpuWallTimer timer{
        collectTimings_, [this](const velox::CpuWallTiming& timing) {
          conversionTiming_.wlock()->add(timing);
        }};
    auto entries = std::make_unique<Entries>(iobufFactory_);
    entries->numRows = rows->size();
    const auto keyCursor = clusterIndex.keyCursor(
        RowRange{startRow, startRow + static_cast<uint32_t>(rows->size())});
    entryDecoder_(
        rows,
        *keyCursor,
        entries->keyStorage,
        entries->valueStorage,
        entries->keys,
        entries->values);
    NIMBLE_CHECK_EQ(
        entries->values.size(),
        entries->keys.size(),
        "Entry decoder returned different key and value counts");
    NIMBLE_CHECK(
        !entries->keys.empty(),
        "Entry decoder returned no entries for a stripe");
    return entries;
  }

  // The projected schema of the stripes.
  const std::shared_ptr<const Type> schema_;
  // Owns the storage of the entries, and the stream reader when the
  // materializer has no cache.
  velox::memory::MemoryPool* const pool_;
  // Decoding options of the file, which each stripe binds the stream reader
  // with, together with the scratch-buffer pool of the entry of the reader.
  const Encoding::Options encodingOptions_;
  // Decodes the rows of a decoded stripe into entries.
  const NimbleIterator::EntryDecoder entryDecoder_;
  // Supplies the stream reader and takes it back when the materializer is
  // destroyed, so that the iterators that share the cache reuse the readers.
  // Null when the options have none.
  const std::shared_ptr<StreamReaderCache> streamReaderCache_;
  // Whether decoding a stripe reads the CPU and wall clocks.
  const bool collectTimings_;
  // Allocates the storage of the entries from the pool.
  folly::IOBufFactory iobufFactory_;
  // Counts the rows decoded from the selected rows of stripes.
  std::atomic_uint64_t numMaterializedRows_{0};
  // The stream reader that decodes the stripes, with the pool that it
  // allocates from and the pool of its scratch buffers. The materializer gets
  // it for the first stripe and keeps it until it is destroyed.
  std::unique_ptr<StreamReaderCache::Entry> cachedStreamReader_;
  // The reader of 'cachedStreamReader_', not null after the first
  // ensureStripeReader().
  StreamReader* streamReader_{nullptr};
  // Track the time spent decoding stripes, and converting their rows into
  // entries.
  folly::Synchronized<velox::CpuWallTiming> decodingTiming_;
  folly::Synchronized<velox::CpuWallTiming> conversionTiming_;
};

// Serves the entries of the stripes that its cursor materializes on demand.
// The base class navigates the file rows and stays at the first selected row
// of a stripe while this class serves its entries.
class NimbleRowIterator final : public NimbleIterator {
 public:
  NimbleRowIterator(
      std::unique_ptr<NimbleIndexProjector> projector,
      velox::memory::MemoryPool* pool,
      NimbleIterator::EntryDecoder entryDecoder,
      const NimbleIterator::Options& options)
      : materializer_{std::make_shared<StripeMaterializer>(
            projector->projectedNimbleType(),
            pool,

            IndexScanCursor::encodingOptions(
                *projector,
                options.cacheEncodingView,
                /*bufferPool=*/nullptr),
            std::move(entryDecoder),
            options.streamReaderCache,
            options.collectTimings)},
        cursor_{IndexScanCursor::create(
            std::move(projector),
            materializer_,
            options)} {}

  NimbleRowIterator(const NimbleRowIterator&) = delete;
  NimbleRowIterator& operator=(const NimbleRowIterator&) = delete;
  NimbleRowIterator(NimbleRowIterator&&) = delete;
  NimbleRowIterator& operator=(NimbleRowIterator&&) = delete;

  ~NimbleRowIterator() override {
    resetState();
  }

  void seek(const velox::serializer::EncodedKeyBounds& bounds) override {
    resetState();
    setRowRange(cursor_->seek(bounds));
  }

  void seek(std::string encodedKey) override {
    resetState();
    setRowRange(cursor_->seek(std::move(encodedKey)));
  }

  void seekToFirst() override {
    resetState();
    setRowRange(cursor_->seekToFirst());
  }

  void reset() override {
    resetState();
    cursor_->reset();
  }

  void nextRow() override {
    prepareStripe();
    if (++entryIndex_ < entries_->keys.size()) {
      return;
    }
    // Moves to the first row after the stripe. The stripe after it loads when
    // it is read.
    entryIndex_ = 0;
    advanceRows(stripeEndRow_ - currentRow());
  }

  bool hasNext() const override {
    prepareStripe();
    return entryIndex_ + 1 < entries_->keys.size() ||
        isBeforeEndRow(stripeEndRow_);
  }

  std::string_view value() override {
    prepareStripe();
    return entries_->values.at(entryIndex_);
  }

  std::string_view key() const override {
    prepareStripe();
    return entries_->keys.at(entryIndex_);
  }

  NimbleIterator::Stats stats() const override {
    auto result = cursor_->stats();
    result.numMaterializedRows = materializer_->numMaterializedRows();
    result.decodingTiming = materializer_->decodingTiming();
    result.conversionTiming = materializer_->conversionTiming();
    return result;
  }

 private:
  // Drops the current stripe, preserving row navigation.
  void resetState() {
    resetValues();
    setRowRange(std::nullopt);
  }

  // Loads the stripe of the current row, when the previous entries did not, and
  // checks that the iterator is on an entry. The stripe loads from const
  // methods that read an entry.
  void prepareStripe() const {
    const auto row = currentRow();
    if (row < stripeStartRow_ || row >= stripeEndRow_) {
      loadStripe(row);
    }
  }

  // Waits for the stripe of the row, which serves all of its entries. Stays out
  // of line to keep the blocking wait out of the per-entry path.
  [[gnu::noinline]] void loadStripe(uint32_t row) const {
    folly::coro::blockingWait(ensureStripe(row));
  }

  // Drops the current stripe, preserving row navigation.
  void resetValues() const {
    entries_.reset();
    entryIndex_ = 0;
    stripeStartRow_ = 0;
    stripeEndRow_ = 0;
  }

  // Waits for the materialized stripe of the row.
  folly::coro::Task<void> ensureStripe(uint32_t row) const {
    if (row >= stripeStartRow_ && row < stripeEndRow_) {
      co_return;
    }

    resetValues();
    entries_ = co_await cursor_->co_nextStripe(row);
    NIMBLE_CHECK_NOT_NULL(entries_);
    stripeStartRow_ = row;
    stripeEndRow_ = row + static_cast<uint32_t>(entries_->numRows);
  }

  // Materializes the stripes loaded by the cursor.
  // Declared before the cursor, which takes it at construction.
  const std::shared_ptr<StripeMaterializer> materializer_;
  // Shares stripe-loading state with active prefetch operations.
  const std::shared_ptr<IndexScanCursor> cursor_;
  // The stripe loads when a read needs it, which reads do from const methods.
  //
  // The entries of the current stripe, or null.
  mutable std::unique_ptr<IndexScanCursor::StripeMaterializer::Entries>
      entries_;
  // The position in the entries of the current stripe.
  mutable size_t entryIndex_{0};
  // File-level row represented by the first row of the current stripe.
  mutable uint32_t stripeStartRow_{0};
  // Exclusive file-level end of the rows of the current stripe.
  mutable uint32_t stripeEndRow_{0};
};

} // namespace

std::unique_ptr<NimbleIterator> NimbleIterator::createRowIterator(
    std::unique_ptr<NimbleIndexProjector> projector,
    velox::memory::MemoryPool* pool,
    EntryDecoder entryDecoder,
    const Options& options) {
  NIMBLE_CHECK_NOT_NULL(pool);
  NIMBLE_CHECK_NOT_NULL(entryDecoder, "Entry decoder is not configured");
  return std::make_unique<NimbleRowIterator>(
      velox::checkedNotNull(std::move(projector)),
      pool,
      std::move(entryDecoder),
      options);
}

} // namespace facebook::nimble
