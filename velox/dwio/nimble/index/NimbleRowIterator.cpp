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
#include <span>
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
#include "velox/dwio/nimble/index/IndexScanCursor.h"
#include "velox/dwio/nimble/index/KeyCursor.h"
#include "velox/dwio/nimble/index/NimbleIndexProjector.h"
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
// on the consumer thread. The materializer builds one stream reader for the
// first stripe and keeps it for its lifetime, so that an iterator that a caller
// keeps between lookups reuses the reader. A scan decodes each stripe once and
// releases it. A point lookup keeps its stripe bound to the reader until
// another stripe replaces it or the iterator resets, so that the next lookups
// of a batch, such as a MultiGet, read their rows of the stripe without loading
// and decompressing it again. The reads of a point lookup leave the reader
// where it is, so later lookups read any rows of the stripe in any order.
class StripeMaterializer final : public IndexScanCursor::StripeMaterializer {
 public:
  StripeMaterializer(
      std::shared_ptr<const Type> schema,
      velox::memory::MemoryPool* pool,
      Encoding::Options encodingOptions,
      NimbleIterator::EntryDecoder entryDecoder,
      bool forGet,
      bool collectTimings)
      : schema_{std::move(schema)},
        rowType_{convertToVeloxType(*schema_)},
        pool_{velox::checkedNotNull(pool)},
        encodingOptions_{std::move(encodingOptions)},
        entryDecoder_{std::move(entryDecoder)},
        forGet_{forGet},
        collectTimings_{collectTimings},
        iobufFactory_{createIOBufFactory(*pool_)} {}

  StripeMaterializer(const StripeMaterializer&) = delete;
  StripeMaterializer& operator=(const StripeMaterializer&) = delete;
  StripeMaterializer(StripeMaterializer&&) = delete;
  StripeMaterializer& operator=(StripeMaterializer&&) = delete;

  ~StripeMaterializer() override = default;

  folly::coro::Task<std::unique_ptr<Entries>> materialize(
      ProjectedStripe stripe,
      index::KeyCursor& keyCursor) override {
    auto rows = co_await co_decode(std::move(stripe));
    co_return convert(std::move(rows), keyCursor);
  }

  bool hasLoadedStripe(uint32_t stripeIndex) const override {
    return stripeDecoder_ != nullptr && stripeDecoder_->hasStripe(stripeIndex);
  }

  // Releases the stripe that the materializer holds and frees the output of
  // the last decode, so that an idle iterator holds neither.
  void reset() {
    releaseStripe();
    rows_.reset();
  }

  // Unbinds the stream reader from the stripe that the materializer holds, and
  // frees the stripe.
  void releaseStripe() {
    if (stripeDecoder_ != nullptr) {
      stripeDecoder_->reset();
    }
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
  // Decodes one stripe at a time: a stream reader bound to the stripe, with the
  // pool of the scratch buffers of its encoding views. The reader keeps its
  // field readers from stripe to stripe and rebinds its stream decoders to the
  // chunks of the next stripe.
  struct StripeDecoder {
    StripeDecoder(
        std::shared_ptr<const Type> schema,
        velox::memory::MemoryPool* pool)
        : streamReader{std::move(schema), pool} {}

    // Returns whether the reader is bound to the stripe 'stripeIndex'.
    bool hasStripe(uint32_t stripeIndex) const {
      return stripe.has_value() && stripe->selection.stripeIndex == stripeIndex;
    }

    // Unbinds the reader from the stripe and frees the stripe. The reader
    // stays for the next stripe.
    void reset() {
      if (!stripe.has_value()) {
        return;
      }
      streamReader.reset();
      stripe.reset();
    }

    // Declared before the reader, whose views return their buffers to it.
    velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};
    // The stripe that the reader is bound to. Declared before the reader,
    // whose views point into its streams, so that the reader is destroyed
    // first.
    std::optional<ProjectedStripe> stripe;
    StreamReader streamReader;
  };

  // Returns the decoder, which the first stripe builds.
  StripeDecoder& ensureStripeDecoder() {
    if (stripeDecoder_ == nullptr) {
      stripeDecoder_ = std::make_unique<StripeDecoder>(schema_, pool_);
    }
    return *stripeDecoder_;
  }

  // Binds the stream reader to 'stripe' in place of the stripe that it holds,
  // with the decoding options of the file and the scratch-buffer pool of the
  // decoder. A stripe that fails to bind is not held. Returns the reader.
  StreamReader& setStripe(ProjectedStripe stripe) {
    releaseStripe();
    auto& decoder = ensureStripeDecoder();
    auto options = encodingOptions_;
    options.bufferPool = &decoder.bufferPool;
    decoder.streamReader.setStreams(stripe.streams, options);
    // The data handle owns the bytes that the reader views, so they stay put.
    decoder.stripe = std::move(stripe);
    return decoder.streamReader;
  }

  // Decodes the rows that the selection of 'stripe' holds.
  folly::coro::Task<velox::RowVectorPtr> co_decode(ProjectedStripe stripe) {
    NIMBLE_CHECK_GT(stripe.selection.rowRange.numRows(), 0);
    if (forGet_) {
      co_return co_await co_decodeGet(std::move(stripe));
    }
    co_return co_await co_decodeScan(std::move(stripe));
  }

  // Decodes the selected rows of 'stripe' for a get, which holds the stripe for
  // the next gets. A stripe without streams selects rows of the held stripe.
  folly::coro::Task<velox::RowVectorPtr> co_decodeGet(ProjectedStripe stripe) {
    const auto selection = stripe.selection;
    if (!stripe.streams.empty()) {
      setStripe(std::move(stripe));
    }
    NIMBLE_CHECK(
        hasLoadedStripe(selection.stripeIndex),
        "A get must read a stripe that the materializer holds: {}",
        selection);

    // Times the decoding of the stripe.
    velox::OptionalCpuWallTimer timer{
        collectTimings_, [this](const velox::CpuWallTiming& timing) {
          decodingTiming_.wlock()->add(timing);
        }};
    const auto numRows = selection.rowRange.numRows();
    prepareRows(static_cast<velox::vector_size_t>(numRows));
    co_await stripeDecoder_->streamReader.co_read(
        std::span<const RowRange>{&selection.rowRange, 1},
        /*outputOffset=*/0,
        rows_);
    co_return finishDecode(rows_, numRows);
  }

  // Makes 'rows_' an output of 'numRows' rows for a get, which reads into
  // an output of its size: the output of the last decode when nothing else
  // holds it or any of its children and buffers, as after the entry decoder
  // released it, or a new vector.
  void prepareRows(velox::vector_size_t numRows) {
    if (rows_ != nullptr && velox::BaseVector::recursivelyReusable(rows_)) {
      velox::BaseVector::prepareForReuse(rows_, numRows);
      return;
    }
    rows_ = velox::BaseVector::create(rowType_, numRows, pool_);
  }

  // Decodes the selected rows of 'stripe' for a scan, which reads each stripe
  // once, and releases the stripe.
  folly::coro::Task<velox::RowVectorPtr> co_decodeScan(ProjectedStripe stripe) {
    const auto selection = stripe.selection;
    auto& streamReader = setStripe(std::move(stripe));
    SCOPE_EXIT {
      releaseStripe();
    };
    NIMBLE_CHECK(
        hasLoadedStripe(selection.stripeIndex),
        "A scan must read the stripe that the cursor loaded: {}",
        selection);
    const auto& rowRange = selection.rowRange;
    if (rowRange.startRow > 0) {
      co_await streamReader.co_skip(rowRange.startRow);
    }

    // Times the decoding of the stripe.
    velox::OptionalCpuWallTimer timer{
        collectTimings_, [this](const velox::CpuWallTiming& timing) {
          decodingTiming_.wlock()->add(timing);
        }};
    // co_next() decodes into the output of the last decode when nothing else
    // holds it, and into a new vector otherwise.
    co_await streamReader.co_next(rowRange.numRows(), rows_);
    co_return finishDecode(rows_, rowRange.numRows());
  }

  // Checks the decoded 'rows' and counts them.
  velox::RowVectorPtr finishDecode(velox::VectorPtr rows, uint32_t numRows) {
    NIMBLE_CHECK_NOT_NULL(rows);
    NIMBLE_CHECK_EQ(rows->size(), numRows);
    numMaterializedRows_.fetch_add(numRows, std::memory_order_relaxed);
    return std::static_pointer_cast<velox::RowVector>(std::move(rows));
  }

  // Decodes the entries of decoded rows, whose keys 'keyCursor' reads.
  std::unique_ptr<Entries> convert(
      velox::RowVectorPtr rows,
      index::KeyCursor& keyCursor) {
    // Times the conversion of the stripe.
    velox::OptionalCpuWallTimer timer{
        collectTimings_, [this](const velox::CpuWallTiming& timing) {
          conversionTiming_.wlock()->add(timing);
        }};
    auto entries = std::make_unique<Entries>(iobufFactory_);
    entries->numRows = rows->size();
    entryDecoder_(
        rows,
        keyCursor,
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
  // The Velox type of the rows that a point lookup reads.
  const velox::TypePtr rowType_;
  // Owns the storage of the entries and the allocations of the stream reader.
  velox::memory::MemoryPool* const pool_;
  // Decoding options of the file, which each stripe binds the stream reader
  // with, together with the scratch-buffer pool of the decoder.
  const Encoding::Options encodingOptions_;
  // Decodes the rows of a decoded stripe into entries.
  const NimbleIterator::EntryDecoder entryDecoder_;
  // Whether the iterator serves point lookups, which keep their stripe.
  const bool forGet_;
  // Whether decoding a stripe reads the CPU and wall clocks.
  const bool collectTimings_;
  // Allocates the storage of the entries from the pool.
  folly::IOBufFactory iobufFactory_;
  // Counts the rows decoded from the selected rows of stripes.
  std::atomic_uint64_t numMaterializedRows_{0};
  // Decodes the stripes, built for the first stripe and kept until the
  // materializer is destroyed.
  std::unique_ptr<StripeDecoder> stripeDecoder_;
  // The output of the last decode, which the next decode reuses unless the
  // entry decoder still holds it. Freed by reset().
  velox::VectorPtr rows_;
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
            options.forGet,
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
    materializer_->reset();
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

  void setVerifyStreamChecksums(bool verifyStreamChecksums) override {
    cursor_->setVerifyStreamChecksums(verifyStreamChecksums);
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
