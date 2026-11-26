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

#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "velox/buffer/BufferPool.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/reader/Decoder.h"

namespace facebook::nimble {

namespace detail {
class StringRetainer;

// Holds the view of one chunk bound to an EncodingViewDecoder and where the
// chunk ends in the stream.
struct BoundChunk {
  // Holds the bytes of the chunk, or null when no buffer does. The view's
  // Encoding::Options::inputStringBuffer points to it, so it outlives the
  // view, which is destroyed first.
  velox::BufferPtr bufferHolder;

  // Random-access view over the chunk's encoded values.
  std::shared_ptr<const EncodingView> view;

  // First stream row after this chunk. The chunk starts at the previous
  // chunk's endRow, or at row zero.
  uint32_t endRow;
};
} // namespace detail

/// Adapts the chunks of one stream to the FieldReader decoder API. Callers
/// pass the encoding of each chunk: its bytes after the chunk header,
/// decompressed. Reads and the sequential cursor use stream rows, numbered
/// across the chunks in stream order. Each chunk starts where the previous one
/// ends, so a read can span chunks. A decoder serves one read at a time.
///
/// A read of strings that covers at least a quarter of the rows of the chunks
/// it spans adds to `stringBuffers` the buffers that the views pass to the
/// addStringBuffer of their reads, which hold every value that the reads
/// return, instead of copying the values. A view passes the buffer that holds
/// its chunk when its values point into the chunk, the buffers it decoded or
/// decompressed the values into, or one buffer that it copies the values into
/// when no buffer holds them. A sparser read copies every value, so that its
/// output does not keep whole chunks alive.
class EncodingViewDecoder final : public Decoder {
 public:
  /// Creates an EncodingView over the encoding of one chunk.
  /// `inputStringBuffer` holds the bytes of the encoding, or is null when no
  /// buffer does. The factory points the view's
  /// Encoding::Options::inputStringBuffer to it, and the decoder keeps it for
  /// as long as the view is bound.
  using EncodingViewFactory = std::function<std::shared_ptr<const EncodingView>(
      std::string_view encoding,
      const velox::BufferPtr& inputStringBuffer)>;

  /// Creates a decoder over the chunks of one stream, in stream order.
  /// `chunks` must be non-empty, and every chunk must store the same data type.
  /// `chunkBufferHolders` is empty, or holds for each chunk the buffer that
  /// holds its bytes, such as the stream it was split from or its
  /// decompressed bytes, or null for a chunk that no buffer holds. The decoder
  /// passes each to `encodingViewFactory` with its chunk. `pool` and the views
  /// returned by `encodingViewFactory` must be non-null. The decoder keeps the
  /// factory to create the views of the chunks bound later with rebind().
  /// `bufferPool` is as in rebind().
  EncodingViewDecoder(
      std::span<const std::string_view> chunks,
      std::span<const velox::BufferPtr> chunkBufferHolders,
      velox::memory::MemoryPool* pool,
      velox::BufferPool* bufferPool,
      const EncodingViewFactory& encodingViewFactory);

  ~EncodingViewDecoder() override;

  /// Replaces the bound chunks with `chunks` and rewinds the sequential
  /// cursor. `chunks` and `chunkBufferHolders` have the same requirements as
  /// in the constructor. The reads of `chunks` take scratch buffers from
  /// `bufferPool` when it is not null, until the next rebind() or release().
  /// A failed call leaves the decoder released.
  void rebind(
      std::span<const std::string_view> chunks,
      std::span<const velox::BufferPtr> chunkBufferHolders,
      velox::BufferPool* bufferPool);

  /// Drops the views of the current chunks so their bytes can be freed. The
  /// decoder must be rebound before it is used again.
  void release();

  uint32_t next(
      uint32_t count,
      void* output,
      std::function<void*()> getOutputNulls,
      std::vector<velox::BufferPtr>& stringBuffers,
      const velox::bits::Bitmap* scatterOutputBitmap = nullptr) override;

  uint32_t remainingRows() override;

  /// Decodes the selected stream rows densely into `output`.
  uint32_t read(
      std::span<const uint32_t> rows,
      DataType dataType,
      void* output,
      std::function<void*()> getOutputNulls,
      std::vector<velox::BufferPtr>& stringBuffers) override;

  /// Decodes ordered, disjoint stream row ranges densely into `output`.
  uint32_t read(
      std::span<const RowRange> ranges,
      DataType dataType,
      void* output,
      std::function<void*()> getOutputNulls,
      std::vector<velox::BufferPtr>& stringBuffers) override;

  void skip(uint32_t count) override;

  void reset() override;

  const Encoding* encoding() const override;

 private:
  // Decodes the next 'count' rows into staging storage and copies them to the
  // positions set in 'scatterBits', leaving the rest of the first
  // 'scatterSize' output positions null. Returns the number of non-null
  // decoded values.
  uint32_t scatterNext(
      uint32_t count,
      void* output,
      void* outputNulls,
      std::vector<velox::BufferPtr>& stringBuffers,
      const uint64_t* scatterBits,
      int32_t scatterSize);

  // Reads stream rows 'rows' densely into 'output' through the chunk views.
  // Reports each null output position through 'setNull'. Passes
  // 'addStringBuffer' to the read of each view. Returns the number of non-null
  // rows.
  uint32_t readRows(
      std::span<const uint32_t> rows,
      const std::function<void(uint32_t)>& setNull,
      const std::function<void(velox::BufferPtr)>& addStringBuffer,
      void* output);

  // Same as readRows() for non-empty, ordered and disjoint stream row ranges.
  uint32_t readRanges(
      std::span<const RowRange> ranges,
      const std::function<void(uint32_t)>& setNull,
      const std::function<void(velox::BufferPtr)>& addStringBuffer,
      void* output);

  // Returns the index in chunks_ of the chunk holding stream row 'row'. Empty
  // chunks hold no row, so the result is never an empty chunk.
  size_t findChunkIndex(uint32_t row) const;

  // Returns the stream row where chunks_[chunkIndex] starts.
  uint32_t chunkStartRow(size_t chunkIndex) const {
    return chunkIndex == 0 ? 0 : chunks_[chunkIndex - 1].endRow;
  }

  // Returns the number of rows in the stream.
  uint32_t numRows() const {
    return chunks_.back().endRow;
  }

  // Returns whether a read of strings of 'numSelectedRows' rows from stream
  // rows 'firstRow' to 'lastRow', both included, holds the buffers of its
  // values instead of copying them: whether it selects at least a quarter of
  // the rows of the chunks that the rows span.
  bool holdsStringBuffers(
      uint32_t numSelectedRows,
      uint32_t firstRow,
      uint32_t lastRow) const;

  // Returns the retainer of a read of strings of 'numSelectedRows' rows from
  // stream rows 'firstRow' to 'lastRow', both included, that holds the buffers
  // of its values (holdsStringBuffers()), or nullptr for a read that copies
  // them. The read resets the retainer when it ends.
  detail::StringRetainer*
  stringRetainer(uint32_t numSelectedRows, uint32_t firstRow, uint32_t lastRow);

  // Returns the retainer above for a read of 'rows' of 'dataType' values, or
  // nullptr when the read selects no rows or reads no strings.
  detail::StringRetainer* stringRetainer(
      DataType dataType,
      std::span<const uint32_t> rows);

  // Does the same as the method above for a read of the 'numRows' rows of
  // 'ranges'.
  detail::StringRetainer* stringRetainer(
      DataType dataType,
      std::span<const RowRange> ranges,
      uint32_t numRows);

  // Returns the data type that every bound chunk stores.
  DataType dataType() const {
    return chunks_.front().view->dataType();
  }

  // Memory pool used to allocate retained string buffers.
  velox::memory::MemoryPool* const pool_;

  // Creates the view of each chunk bound to this decoder.
  const EncodingViewFactory encodingViewFactory_;

  // Returns temporary decode storage to the caller-owned cache of the bound
  // chunks, which each rebind() sets.
  velox::BufferPool* bufferPool_{nullptr};

  // Bound chunks in stream order.
  std::vector<detail::BoundChunk> chunks_;

  // Collects the buffers that hold the strings of the current read, when the
  // read holds them instead of copying. The first such read creates it, and
  // each read resets it when it ends.
  std::unique_ptr<detail::StringRetainer> stringRetainer_;

  // Next stream row consumed by the sequential interface.
  uint32_t nextRow_{0};
};

} // namespace facebook::nimble
