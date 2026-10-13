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

#include "velox/dwio/nimble/velox/index/NimbleIterator.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>

#include "velox/buffer/Buffer.h"
#include "velox/buffer/BufferPool.h"
#include "velox/common/Casts.h"
#include "velox/common/base/Exceptions.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/KeyCursor.h"
#include "velox/dwio/nimble/serializer/StreamReader.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"
#include "velox/dwio/nimble/velox/index/IndexScanCursor.h"
#include "velox/dwio/nimble/velox/index/NimbleIndexProjector.h"
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

// Preserve the largest serialized row allocation to avoid repeated growth.
char* ensureBuffer(
    velox::BufferPtr& buffer,
    size_t size,
    velox::memory::MemoryPool* pool) {
  const auto allocationSize = std::max<size_t>(size, 1);
  if (buffer == nullptr) {
    buffer = velox::AlignedBuffer::allocate<char>(allocationSize, pool);
  } else if (buffer->capacity() < allocationSize) {
    velox::AlignedBuffer::reallocate<char>(&buffer, allocationSize);
  } else {
    buffer->setSize(allocationSize);
  }
  return buffer->asMutable<char>();
}

// Writes a CompactRow using reusable pool-backed output storage.
size_t serializeCompactRow(
    const velox::RowVectorPtr& rows,
    velox::vector_size_t rowOffset,
    velox::memory::MemoryPool* pool,
    velox::BufferPtr& buffer) {
  velox::row::CompactRow compactRow{rows};
  const auto size = compactRow.rowSize(rowOffset);
  auto* data = ensureBuffer(buffer, size, pool);
  std::memset(data, 0, size);
  const auto serializedSize = compactRow.serialize(rowOffset, data);
  NIMBLE_CHECK_EQ(serializedSize, size);
  return size;
}

// Writes an UnsafeRow using reusable pool-backed output storage.
size_t serializeUnsafeRow(
    const velox::RowVectorPtr& rows,
    velox::vector_size_t rowOffset,
    velox::memory::MemoryPool* pool,
    velox::BufferPtr& buffer) {
  velox::row::UnsafeRowFast unsafeRow{rows};
  const auto size = unsafeRow.rowSize(rowOffset);
  auto* data = ensureBuffer(buffer, size, pool);
  std::memset(data, 0, size);
  const auto serializedSize = unsafeRow.serialize(rowOffset, data);
  NIMBLE_CHECK_EQ(serializedSize, size);
  return size;
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
    advanceRows(1);
    if (!key_.has_value()) {
      // Keep key reads lazy by reopening at the new row after an unread key.
      keyCursor_.reset();
    }
    key_.reset();
    value_.reset();
    co_return;
  }

  folly::coro::Task<bool> co_nextBatch(
      velox::RowVectorPtr& /*output*/) override {
    NIMBLE_UNSUPPORTED("Row iterator cannot produce batch output");
    co_return false;
  }

  folly::coro::Task<std::string_view> co_value() override {
    const auto row = currentRow();
    if (value_.has_value()) {
      co_return *value_;
    }
    co_await materializeValue(row);
    co_return *value_;
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

  // Materializes the current row from its projected stripe streams.
  folly::coro::Task<void> materializeValue(uint32_t row) {
    co_await ensureStripe(row);
    NIMBLE_CHECK_GE(row, stripeStartRow_);
    NIMBLE_CHECK_LT(row, stripeEndRow_);
    const auto rowOffset = row - stripeStartRow_;
    NIMBLE_CHECK_NOT_NULL(rowVector_);
    {
      velox::CpuWallTimer timer{materializationTiming_};
      materializeRow(
          velox::checkedPointerCast<velox::RowVector>(rowVector_), rowOffset);
    }
    co_return;
  }

  // Clears decoder views before releasing stripe storage, preserving row
  // navigation, key state, and reusable output buffers.
  void resetValues() {
    streamReader_->reset();
    stripeStreams_.reset();
    stripeStartRow_ = 0;
    stripeEndRow_ = 0;
    value_.reset();
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

    {
      velox::CpuWallTimer timer{materializationTiming_};
      velox::VectorPtr decodedRows{std::move(rowVector_)};
      SCOPE_EXIT {
        rowVector_ =
            std::static_pointer_cast<velox::RowVector>(std::move(decodedRows));
      };
      co_await streamReader_->co_next(numRows, decodedRows);
    }
    NIMBLE_CHECK_NOT_NULL(rowVector_);
    NIMBLE_CHECK_EQ(rowVector_->size(), numRows);
    // Publish the materialized interval only after decoding succeeds.
    stripeStartRow_ = stripeStartRow;
    stripeEndRow_ = stripeStartRow + numRows;
    co_return;
  }

  void materializeRow(const velox::RowVectorPtr& rows, uint32_t rowOffset) {
    const auto size = outputSerializer_(rows, rowOffset, pool_, valueBuffer_);
    NIMBLE_CHECK(
        size == 0 || valueBuffer_ != nullptr,
        "Output row serializer returned bytes without an output buffer");
    if (valueBuffer_ == nullptr) {
      value_ = std::string_view{};
      return;
    }
    NIMBLE_CHECK_LE(
        size,
        valueBuffer_->size(),
        "Output row serializer returned more bytes than the buffer contains");
    value_ = std::string_view{valueBuffer_->as<char>(), size};
  }

  // Serializes the current row in the format selected by the factory.
  const NimbleIterator::OutputRowSerializer outputSerializer_;
  // Owns allocations made while decoding rows and serializing output.
  velox::memory::MemoryPool* const pool_;
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
  // Tracks row serialization time.
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
  // Engaged after serializing the current row, including an empty value.
  std::optional<std::string_view> value_;
  // Reuses serialized output storage across materialized rows.
  velox::BufferPtr valueBuffer_;
};

} // namespace

void registerDefaultOutputRowSerializers() {
  static std::once_flag registered;
  std::call_once(registered, []() {
    auto serializers = outputRowSerializerRegistry().wlock();
    serializers->emplace(
        NimbleIterator::OutputRowFormat{NimbleIterator::kCompactRowFormat},
        serializeCompactRow);
    serializers->emplace(
        NimbleIterator::OutputRowFormat{NimbleIterator::kUnsafeRowFormat},
        serializeUnsafeRow);
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
