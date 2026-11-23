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

#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "velox/buffer/Buffer.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

/// A random-access FSST view that decodes strings lazily in cached chunks.
///
/// FSST's native decoder already unrolls four codes and uses wide symbol
/// stores. Chunking amortizes view bookkeeping without forcing callers to
/// materialize the whole stream, and cached chunks keep returned string_views
/// stable.
///
/// A cached view decodes a chunk only once reads have asked for enough of its
/// rows. Until then it decodes just the rows that reads ask for, so a sparse
/// read, such as a take of a few rows of each chunk, costs its own rows and
/// not whole chunks.
///
/// A read holds a lock of the view while it decodes rows alone, so concurrent
/// reads of one view decode such rows one read at a time.
class FsstEncodingView final : public TypedEncodingView<std::string_view> {
 public:
  using TypedEncodingView<std::string_view>::read;

  FsstEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options);

  /// Reads ordered, disjoint row ranges. A cached view first counts the rows
  /// that the ranges ask for in each chunk, so that it decides once for all of
  /// them whether to decode the chunk.
  uint32_t read(
      std::span<const RowRange> ranges,
      const std::function<void(uint32_t)>& setNull,
      void* output) const final;

 private:
  static constexpr uint32_t kRowsPerChunk = 1'024;

  // A cached view decodes a chunk once reads have asked for this many of its
  // rows, or proportionally fewer of a last chunk with fewer rows. Until then,
  // it decodes each row that a read asks for alone. A row decoded alone costs
  // a little less than its share of a chunk decode, so decoding the chunk does
  // not pay off in the read that triggers it. It pays off in later reads of
  // the chunk's rows, which read them without the exclusive lock and the map
  // lookup that rows decoded alone take on every read.
  //
  // TODO: Retune with a benchmark that reads the same rows more than once.
  static constexpr uint32_t kMinRowsToDecodeChunk = kRowsPerChunk / 4;
  static constexpr size_t kMaxSymbolLength = 8;

  struct DecodedChunk;
  class DecodedStrings;

  std::string_view readTypedAt(uint32_t index) const final;

  // Reads a contiguous row range from the decoded chunk cache, or row by row
  // from a chunk that chunkForRead() leaves undecoded.
  void readPhysical(uint32_t offset, uint32_t length, std::string_view* output)
      const final;

  // Reads arbitrary row indices while reusing the most recently accessed chunk,
  // or row by row from a chunk that chunkForRead() leaves undecoded.
  void readPhysicalAt(
      std::span<const uint32_t> indices,
      std::string_view* output) const final;

  // Reads the length of every row once and keeps the compressed offset of every
  // row, so that reads decode chunks of rows. Checks every length against the
  // blob.
  void initializeCached(std::string_view encodedLengths);

  // Keeps the lengths stream as a view and decodes only the strings that reads
  // ask for, which skips the validation of every length that a cached view
  // performs.
  void initializeUncached(std::string_view encodedLengths);

  // Returns whether the view holds the compressed offset of every row and
  // decodes chunks of rows.
  bool cachesDecodedChunks() const {
    return this->options_.cacheEncodingView;
  }

  static std::string_view valueAt(
      const DecodedChunk& chunk,
      uint32_t rowInChunk);

  // Decodes the string of 'row' alone from its compressed bytes, which start
  // at 'compressedBegin', for a read without cacheEncodingView or a read of a
  // cached view that leaves the row's chunk undecoded. Keeps the string in
  // 'strings' for the lifetime of the view, and returns it again when a row is
  // read again. A view without cacheEncodingView first checks the compressed
  // length against the blob, which a cached view checks for every row when it
  // is created.
  std::string_view decodeValue(
      uint32_t row,
      uint64_t compressedBegin,
      uint32_t compressedSize,
      DecodedStrings& strings) const;

  // Decodes the strings of 'rows' of a cached view alone with decodeValue(),
  // from the compressed offsets of the rows, under one lock of
  // decodedStrings_. 'rows' must not be empty.
  void decodeRows(std::span<const uint32_t> rows, std::string_view* output)
      const;

  // Returns the index of the chunk that holds 'row'.
  static uint32_t rowChunk(uint32_t row) {
    return row / kRowsPerChunk;
  }

  // Returns the number of rows in a chunk: kRowsPerChunk, or fewer for the
  // last chunk.
  uint32_t chunkRows(uint32_t chunkIndex) const;

  // Returns the decoded chunk for a read of 'numRows' of its rows. Decodes the
  // chunk and caches it for later reads once reads, this one included, have
  // asked for kMinRowsToDecodeChunk of its rows. Returns nullptr while they
  // have asked for fewer, and the read then decodes its rows with decodeRows().
  std::shared_ptr<const DecodedChunk> chunkForRead(
      uint32_t chunkIndex,
      uint32_t numRows) const;

  // Decodes one chunk and builds offsets into its stable string storage.
  std::shared_ptr<const DecodedChunk> decodeChunk(uint32_t chunkIndex) const;

  // Imported FSST symbol table used to decode every chunk in this view.
  fsst_decoder_t decoder_{};

  // Compressed string payload, either borrowed from input or pool-owned.
  std::string_view blob_;

  // Prefix sums mapping row indices to byte offsets in blob_.
  std::vector<size_t> compressedOffsets_;

  // TODO: Limit the number of chunks cached by each FSST view.
  // Lazily decoded chunks indexed by chunk number and shared across readers.
  mutable folly::Synchronized<std::vector<std::shared_ptr<const DecodedChunk>>>
      decodedChunks_;

  // Counts, for each chunk, the rows that reads asked for while the chunk was
  // not decoded, which they decoded alone. Set only for cached views.
  std::unique_ptr<std::atomic_uint32_t[]> rowsReadAlone_;

  // Reads the compressed length of one row at a time. Set only for single
  // access, which keeps no offsets and no chunks.
  std::unique_ptr<TypedEncodingView<uint32_t>> lengths_;

  // TODO: Free the strings of a chunk's rows read alone once the chunk is
  // decoded, at a point where callers hold no views of them. Callers such as
  // MainlyConstant hold them across reads, so the view cannot tell when on its
  // own. Until then, a cached view keeps fewer than kMinRowsToDecodeChunk of
  // them for each chunk besides the decoded chunk.
  // The strings that decodeValue() decoded, one for each row read alone. Keeps
  // them until the view is destroyed, so that returned views stay valid, and a
  // row read again decodes nothing.
  //
  // Not thread-safe. The view keeps it in folly::Synchronized. A read locks it
  // once for each batch of rows that it decodes alone: the rows of each chunk
  // for a cached view, and the rows of each call for a view without caching.
  class DecodedStrings {
   public:
    DecodedStrings() = default;

    ~DecodedStrings() = default;

    // Returned strings point into 'firstBlock_', so it must not move.
    DecodedStrings(const DecodedStrings&) = delete;
    DecodedStrings& operator=(const DecodedStrings&) = delete;
    DecodedStrings(DecodedStrings&&) = delete;
    DecodedStrings& operator=(DecodedStrings&&) = delete;

    // Returns the string of 'row', or nothing if no read has decoded it.
    std::optional<std::string_view> find(uint32_t row) const;

    // Returns where the next string can take up to 'maxSize' bytes. Allocates
    // a block from 'pool' when the last block has less room. Takes the pool
    // for each call so that the view's folly::Synchronized default-constructs
    // this object, which must not move.
    char* reserve(size_t maxSize, velox::memory::MemoryPool* pool);

    // Keeps the first 'size' bytes of the space that the last reserve()
    // returned as the string of 'row', and returns the string.
    std::string_view add(uint32_t row, size_t size);

   private:
    // Bytes of 'firstBlock_': the strings of a read of a few rows of the view,
    // such as a take of a few rows of each stripe.
    static constexpr size_t kFirstBlockSize{1 << 10};

    // Bytes of each block from the pool, unless one string needs more: about
    // what a cached view decodes alone of a chunk before it decodes the chunk.
    static constexpr size_t kBlockSize{16 << 10};

    // Maps a row to its string in 'firstBlock_' or 'blocks_'.
    folly::F14FastMap<uint32_t, std::string_view> strings_;
    // Holds the first strings, so that a view that decodes a few rows alone
    // allocates nothing from the pool, which costs more than their decoding.
    std::array<char, kFirstBlockSize> firstBlock_{};
    // Hold the strings that come after 'firstBlock_' is full, one after
    // another, and never move them.
    std::vector<velox::BufferPtr> blocks_;
    // The first byte of the last block that no string holds.
    char* free_{firstBlock_.data()};
    // The bytes from 'free_' to the end of the last block.
    size_t freeBytes_{kFirstBlockSize};
  };
  mutable folly::Synchronized<DecodedStrings> decodedStrings_;
};

} // namespace facebook::nimble
