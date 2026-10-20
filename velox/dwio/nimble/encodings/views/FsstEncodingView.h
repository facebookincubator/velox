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
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

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
class FsstEncodingView final : public TypedEncodingView<std::string_view> {
 public:
  FsstEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options);

 private:
  static constexpr uint32_t kRowsPerChunk = 1'024;
  static constexpr size_t kMaxSymbolLength = 8;

  struct DecodedChunk;

  std::string_view readTypedAt(uint32_t index) const final;

  // Reads a contiguous row range from the decoded chunk cache.
  void readPhysical(uint32_t offset, uint32_t length, std::string_view* output)
      const final;

  // Reads arbitrary row indices while reusing the most recently accessed chunk.
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

  // Decodes the string of 'row' for a read without cacheEncodingView from its
  // compressed bytes, which start at 'compressedBegin'. Keeps the string alive
  // for the lifetime of the view, and returns it again when a row is read
  // again.
  std::string_view decodeValue(
      uint32_t row,
      uint64_t compressedBegin,
      uint32_t compressedSize) const;

  // Returns a cached chunk, decoding and publishing it on first access.
  std::shared_ptr<const DecodedChunk> getChunk(uint32_t chunkIndex) const;

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

  // Reads the compressed length of one row at a time. Set only for single
  // access, which keeps no offsets and no chunks.
  std::unique_ptr<TypedEncodingView<uint32_t>> lengths_;

  // The strings that reads without cacheEncodingView decoded, one for each row
  // read. Keeps them until the view is destroyed, so that returned views stay
  // valid, and a row read again decodes nothing.
  class DecodedStrings {
   public:
    // Returns the string of 'row', or nothing if no read has decoded it. Takes
    // the shared lock.
    std::optional<std::string_view> find(uint32_t row) const;

    // Keeps 'decoded' as the string of 'row', unless a concurrent read kept one
    // first, and returns the string that the row has. Takes the exclusive lock.
    std::string_view add(uint32_t row, std::string decoded);

   private:
    mutable std::shared_mutex lock_;
    // Maps a row to the position of its string in 'values_'.
    folly::F14FastMap<uint32_t, size_t> positions_;
    // Never moves its strings, so views of them stay valid.
    std::deque<std::string> values_;
  };
  mutable DecodedStrings decodedStrings_;
};

} // namespace facebook::nimble
