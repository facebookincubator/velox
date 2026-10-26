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

// Implements lazy row materialization and output serialization for index scans.

#include "velox/dwio/nimble/index/NimbleIterator.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>
#include <folly/coro/BlockingWait.h>

#include "velox/buffer/Buffer.h"
#include "velox/buffer/BufferPool.h"
#include "velox/common/Casts.h"
#include "velox/common/base/Exceptions.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/SchemaUtils.h"
#include "velox/dwio/nimble/index/IndexScanCursor.h"
#include "velox/dwio/nimble/index/KeyCursor.h"
#include "velox/dwio/nimble/index/NimbleIndexProjector.h"
#include "velox/dwio/nimble/serializer/StreamReader.h"
#include "velox/row/CompactRow.h"
#include "velox/row/UnsafeRowFast.h"

namespace facebook::nimble {
using detail::IndexScanCursor;

namespace {

using OutputRowSerializerMap = folly::F14FastMap<
    NimbleIterator::OutputRowFormat,
    NimbleIterator::OutputRowSerializer>;

folly::Synchronized<OutputRowSerializerMap>& outputRowSerializerRegistry() {
  static folly::Synchronized<OutputRowSerializerMap> registry;
  return registry;
}

// Returns the serializer registered for the selected row format.
NimbleIterator::OutputRowSerializer getOutputRowSerializer(
    std::string_view rowFormat) {
  registerDefaultOutputRowSerializers();
  const auto serializers = outputRowSerializerRegistry().rlock();
  const auto it = serializers->find(std::string{rowFormat});
  NIMBLE_CHECK(
      it != serializers->end(), "Unsupported output row format: {}", rowFormat);
  return it->second;
}

// Preserves the largest serialized row allocation to avoid repeated growth. A
// buffer that is too small goes back to the optional buffer pool, and the
// pool, when set, supplies the replacement.
char* ensureBuffer(
    velox::BufferPtr& buffer,
    size_t size,
    velox::memory::MemoryPool* pool,
    velox::BufferPool* bufferPool) {
  const auto allocationSize = std::max<size_t>(size, 1);
  if (buffer != nullptr && buffer->capacity() >= allocationSize) {
    buffer->setSize(allocationSize);
    return buffer->asMutable<char>();
  }
  if (bufferPool == nullptr) {
    if (buffer == nullptr) {
      buffer = velox::AlignedBuffer::allocate<char>(allocationSize, pool);
    } else {
      velox::AlignedBuffer::reallocate<char>(&buffer, allocationSize);
    }
    return buffer->asMutable<char>();
  }
  bufferPool->release(std::move(buffer));
  buffer = bufferPool->get(allocationSize);
  if (buffer == nullptr) {
    buffer = velox::AlignedBuffer::allocate<char>(allocationSize, pool);
  } else {
    buffer->setSize(allocationSize);
  }
  return buffer->asMutable<char>();
}

// Sets the offsets of 'numRows' rows from their sizes and prepares a zeroed
// buffer for all of them. Returns the start of the buffer.
template <typename RowSize>
char* prepareSerializedRows(
    velox::vector_size_t numRows,
    const RowSize& rowSize,
    velox::memory::MemoryPool* pool,
    velox::BufferPool* bufferPool,
    NimbleIterator::SerializedRows& output) {
  auto& offsets = output.offsets;
  offsets.resize(static_cast<size_t>(numRows) + 1);
  offsets[0] = 0;
  for (velox::vector_size_t row{0}; row < numRows; ++row) {
    offsets[row + 1] = offsets[row] + rowSize(row);
  }
  const auto totalBytes = offsets[numRows];
  auto* data = ensureBuffer(output.buffer, totalBytes, pool, bufferPool);
  std::memset(data, 0, totalBytes);
  return data;
}

// Writes rows as CompactRows using reusable pool-backed output storage.
void serializeCompactRows(
    const velox::RowVectorPtr& rows,
    velox::vector_size_t rowOffset,
    velox::vector_size_t numRows,
    velox::memory::MemoryPool* pool,
    velox::BufferPool* bufferPool,
    NimbleIterator::SerializedRows& output) {
  velox::row::CompactRow compactRow{rows};
  const auto fixedRowSize =
      velox::row::CompactRow::fixedRowSize(rows->rowType());
  auto* data = prepareSerializedRows(
      numRows,
      [&](velox::vector_size_t row) -> size_t {
        return fixedRowSize.has_value() ? *fixedRowSize
                                        : compactRow.rowSize(rowOffset + row);
      },
      pool,
      bufferPool,
      output);
  compactRow.serialize(rowOffset, numRows, output.offsets.data(), data);
}

// Writes rows as UnsafeRows using reusable pool-backed output storage.
void serializeUnsafeRows(
    const velox::RowVectorPtr& rows,
    velox::vector_size_t rowOffset,
    velox::vector_size_t numRows,
    velox::memory::MemoryPool* pool,
    velox::BufferPool* bufferPool,
    NimbleIterator::SerializedRows& output) {
  velox::row::UnsafeRowFast unsafeRow{rows};
  auto* data = prepareSerializedRows(
      numRows,
      [&](velox::vector_size_t row) -> size_t {
        return unsafeRow.rowSize(rowOffset + row);
      },
      pool,
      bufferPool,
      output);
  const auto& offsets = output.offsets;
  for (velox::vector_size_t row{0}; row < numRows; ++row) {
    const auto serializedSize =
        unsafeRow.serialize(rowOffset + row, data + offsets[row]);
    NIMBLE_CHECK_EQ(serializedSize, offsets[row + 1] - offsets[row]);
  }
}

// Serializes selected rows on demand, reusing decoded projected-stripe rows.
class NimbleRowIterator final : public NimbleIterator {
 public:
  NimbleRowIterator(
      std::unique_ptr<NimbleIndexProjector> projector,
      velox::memory::MemoryPool* pool,
      NimbleIterator::OutputRowSerializer outputSerializer,
      const NimbleIterator::Options& options)
      : outputSerializer_{std::move(outputSerializer)},
        pool_{pool},
        bufferPool_{options.bufferPool},
        collectTimings_{options.collectTimings},
        cursor_{IndexScanCursor::create(std::move(projector), options)},
        schema_{cursor_->projectedNimbleType()},
        streamReader_{std::make_unique<StreamReader>(
            schema_,
            velox::checkedNotNull(pool_),
            cursor_->encodingOptions(&encodingBufferPool_))} {
    NIMBLE_CHECK_NOT_NULL(
        outputSerializer_, "Output row serializer is not configured");
  }

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
    advanceRows(1);
    if (!key_.has_value()) {
      // Keep key reads lazy by reopening at the new row after an unread key.
      keyCursor_.reset();
    }
    key_.reset();
  }

  folly::coro::Task<bool> co_nextBatch(
      velox::RowVectorPtr& /*output*/) override {
    NIMBLE_UNSUPPORTED("Row iterator cannot produce batch output");
    co_return false;
  }

  void prepareValue() override {
    const auto row = currentRow();
    if (row < stripeStartRow_ || row >= stripeEndRow_) {
      loadStripe(row);
    }
  }

  std::string_view value() override {
    const auto row = currentRow();
    if (row < stripeStartRow_ || row >= stripeEndRow_) {
      loadStripe(row);
    }
    return serializedRows_.row(row - stripeStartRow_);
  }

  std::string_view key() const override {
    const auto row = currentRow();
    if (!key_.has_value()) {
      if (keyCursor_ == nullptr) {
        keyCursor_ = cursor_->keyCursor(row);
      }
      key_ = keyCursor_->next();
    }
    return *key_;
  }

  uint32_t rowNumber() const override {
    return currentRow();
  }

  NimbleIterator::Stats stats() const override {
    auto result = cursor_->stats();
    result.numMaterializedRows = numMaterializedRows_;
    result.materializationTiming = materializationTiming_;
    return result;
  }

 private:
  // Resets navigation and key/value state while preserving reusable buffers.
  void resetState() {
    resetValues();
    setRowRange(std::nullopt);
    resetKeys();
  }

  // Releases key chunks when navigation starts a different scan.
  void resetKeys() {
    key_.reset();
    keyCursor_.reset();
  }

  // Waits for the stripe of the row, which serves all of its rows. Stays out of
  // line to keep the blocking wait out of the per-row path of value().
  [[gnu::noinline]] void loadStripe(uint32_t row) {
    folly::coro::blockingWait(ensureStripe(row));
  }

  // Serializes all rows of the decoded stripe in one call.
  void serializeRows() {
    const auto numRows = rowVector_->size();
    outputSerializer_(
        velox::checkedPointerCast<velox::RowVector>(rowVector_),
        /*rowOffset=*/0,
        numRows,
        pool_,
        bufferPool_,
        serializedRows_);
    NIMBLE_CHECK_EQ(
        serializedRows_.offsets.size(),
        static_cast<size_t>(numRows) + 1,
        "Output row serializer must set an offset per row and the end of the last row");
    const auto serializedBytes =
        serializedRows_.offsets.at(static_cast<size_t>(numRows));
    NIMBLE_CHECK(
        serializedBytes == 0 || serializedRows_.buffer != nullptr,
        "Output row serializer returned bytes without an output buffer");
    NIMBLE_CHECK(
        serializedRows_.buffer == nullptr ||
            serializedBytes <= serializedRows_.buffer->size(),
        "Output row serializer returned more bytes than the buffer contains");
  }

  // Clears decoder views before releasing stripe storage, preserving row
  // navigation, key state, and reusable output buffers. Returns the buffer of
  // serialized rows to the buffer pool when there is one.
  void resetValues() {
    streamReader_->reset();
    stripeStreams_.reset();
    stripeStartRow_ = 0;
    stripeEndRow_ = 0;
    if (bufferPool_ != nullptr) {
      bufferPool_->release(std::move(serializedRows_.buffer));
    }
  }

  // Loads the projected stripe and decodes its remaining rows into one vector.
  folly::coro::Task<void> ensureStripe(uint32_t row) {
    if (row >= stripeStartRow_ && row < stripeEndRow_) {
      NIMBLE_CHECK_NOT_NULL(rowVector_);
      co_return;
    }

    resetValues();

    stripeStreams_ = co_await cursor_->co_nextStripe(row);
    const auto& streams = *stripeStreams_;
    const auto& selection = streams.selection;
    const auto stripeStartRow =
        selection.startRow + selection.rowRange.startRow;
    NIMBLE_CHECK_EQ(
        stripeStartRow,
        row,
        "Projected stripe streams do not match the iterator position");
    const auto numRows = selection.rowRange.numRows();
    NIMBLE_CHECK_GT(numRows, 0);
    streamReader_->setStreams(streams.streams);
    if (selection.rowRange.startRow > 0) {
      co_await streamReader_->co_skip(selection.rowRange.startRow);
    }

    // Times the decoding and the serialization of the stripe.
    velox::OptionalCpuWallTimer timer{
        collectTimings_, [this](const velox::CpuWallTiming& timing) {
          materializationTiming_.add(timing);
        }};
    {
      velox::VectorPtr decodedRows{std::move(rowVector_)};
      SCOPE_EXIT {
        rowVector_ =
            std::static_pointer_cast<velox::RowVector>(std::move(decodedRows));
      };
      co_await streamReader_->co_next(numRows, decodedRows);
    }
    NIMBLE_CHECK_NOT_NULL(rowVector_);
    NIMBLE_CHECK_EQ(rowVector_->size(), numRows);
    numMaterializedRows_ += numRows;
    serializeRows();
    // Publish the materialized interval only after decoding and serializing
    // succeed.
    stripeStartRow_ = stripeStartRow;
    stripeEndRow_ = stripeStartRow + numRows;
    co_return;
  }

  // Serializes the rows of a decoded stripe in the format selected by the
  // factory.
  const NimbleIterator::OutputRowSerializer outputSerializer_;
  // Owns allocations made while decoding rows and serializing output.
  velox::memory::MemoryPool* const pool_;
  // Recycles the buffer of serialized rows when it is set.
  velox::BufferPool* const bufferPool_;
  // Whether decoding a stripe reads the CPU and wall clocks.
  const bool collectTimings_;
  // Shares stripe-loading state with active prefetch operations.
  const std::shared_ptr<IndexScanCursor> cursor_;
  // Projected Nimble schema for the stream reader.
  const std::shared_ptr<const Type> schema_;
  // Reuses scratch buffers created by encoding views.
  velox::BufferPool encodingBufferPool_{velox::BufferPool::kDefaultCapacity};
  const std::unique_ptr<StreamReader> streamReader_;
  // Reuses decoded vectors across serialized-row requests.
  velox::VectorPtr rowVector_;
  // Keeps the current stripe streams alive while they are decoded.
  std::optional<ProjectedStripe> stripeStreams_;
  // Holds all rows of the decoded stripe serialized back to back. Row i of the
  // stripe occupies the bytes [offsets[i], offsets[i + 1]) of the buffer, and
  // the buffer is reused across stripes.
  NimbleIterator::SerializedRows serializedRows_;
  // Counts the rows decoded and serialized from the selected rows of stripes.
  uint64_t numMaterializedRows_{0};
  // Tracks the time spent decoding and serializing stripes.
  velox::CpuWallTiming materializationTiming_{};
  // File-level row represented by the first materialized row in rowVector_.
  uint32_t stripeStartRow_{0};
  // Exclusive file-level end of the rows materialized in rowVector_.
  uint32_t stripeEndRow_{0};
  // Retains decoded key chunks independently of lazy projected data streams.
  // When present, points to the current row if key_ is unset, or the next row
  // if key_ is cached.
  mutable std::unique_ptr<index::KeyCursor> keyCursor_;
  // Caches a view owned by the key cursor for the iterator's current row.
  mutable std::optional<std::string_view> key_;
};

} // namespace

void registerDefaultOutputRowSerializers() {
  static std::once_flag registered;
  std::call_once(registered, []() {
    auto serializers = outputRowSerializerRegistry().wlock();
    serializers->emplace(
        NimbleIterator::OutputRowFormat{NimbleIterator::kCompactRowFormat},
        serializeCompactRows);
    serializers->emplace(
        NimbleIterator::OutputRowFormat{NimbleIterator::kUnsafeRowFormat},
        serializeUnsafeRows);
  });
}

void registerOutputRowSerializer(
    NimbleIterator::OutputRowFormat rowFormat,
    const NimbleIterator::OutputRowSerializer& serializer) {
  NIMBLE_CHECK(!rowFormat.empty(), "Output row format cannot be empty");
  const std::string_view formatView{rowFormat};
  NIMBLE_CHECK(
      formatView != NimbleIterator::kCompactRowFormat &&
          formatView != NimbleIterator::kUnsafeRowFormat,
      "Cannot register a built-in output row format: {}",
      rowFormat);
  NIMBLE_CHECK(
      static_cast<bool>(serializer),
      "Output row serializer cannot be empty for format: {}",
      rowFormat);

  auto serializers = outputRowSerializerRegistry().wlock();
  const auto inserted = serializers->emplace(rowFormat, serializer).second;
  NIMBLE_CHECK(
      inserted,
      "Output row serializer is already registered for format: {}",
      rowFormat);
}

std::unique_ptr<NimbleIterator> NimbleIterator::createRowIterator(
    std::unique_ptr<NimbleIndexProjector> projector,
    velox::memory::MemoryPool* pool,
    const Options& options) {
  NIMBLE_CHECK_NOT_NULL(pool);
  auto outputSerializer = getOutputRowSerializer(options.rowFormat);
  return std::make_unique<NimbleRowIterator>(
      std::move(projector), pool, std::move(outputSerializer), options);
}

} // namespace facebook::nimble
