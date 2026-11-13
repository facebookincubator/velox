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

/// Adapts the chunks of one stream to the FieldReader decoder API. Callers
/// pass the encoding of each chunk: its bytes after the chunk header,
/// decompressed. Reads and the sequential cursor use stream rows, numbered
/// across the chunks in stream order. Each chunk starts where the previous one
/// ends, so a read can span chunks.
class EncodingViewDecoder final : public Decoder {
 public:
  /// Creates an EncodingView over the encoding of one chunk.
  using EncodingViewFactory =
      std::function<std::shared_ptr<const EncodingView>(std::string_view)>;

  /// Creates a decoder over the chunks of one stream, in stream order.
  /// `chunks` must be non-empty, and every chunk must store the same data type.
  /// `pool` and the views returned by `encodingViewFactory` must be non-null.
  /// The decoder keeps the factory to create the views of the chunks bound
  /// later with rebind(). `bufferPool` is as in rebind().
  EncodingViewDecoder(
      std::span<const std::string_view> chunks,
      velox::memory::MemoryPool* pool,
      velox::BufferPool* bufferPool,
      const EncodingViewFactory& encodingViewFactory);

  /// Replaces the bound chunks with `chunks` and rewinds the sequential
  /// cursor. `chunks` has the same requirements as in the constructor. The
  /// reads of `chunks` take scratch buffers from `bufferPool` when it is not
  /// null, until the next rebind() or release(). A failed call leaves the
  /// decoder released.
  void rebind(
      std::span<const std::string_view> chunks,
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
  // Holds the view of one bound chunk and where the chunk ends in the stream.
  struct Chunk {
    // Random-access view over the chunk's encoded values.
    std::shared_ptr<const EncodingView> view;

    // First stream row after this chunk. The chunk starts at the previous
    // chunk's endRow, or at row zero.
    uint32_t endRow;
  };

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
  // Reports each null output position through 'setNull'. Returns the number of
  // non-null rows.
  uint32_t readRows(
      std::span<const uint32_t> rows,
      const std::function<void(uint32_t)>& setNull,
      void* output);

  // Same as readRows() for non-empty, ordered and disjoint stream row ranges.
  uint32_t readRanges(
      std::span<const RowRange> ranges,
      const std::function<void(uint32_t)>& setNull,
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
  std::vector<Chunk> chunks_;

  // Next stream row consumed by the sequential interface.
  uint32_t nextRow_{0};
};

} // namespace facebook::nimble
