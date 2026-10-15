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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

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

  static std::string_view valueAt(
      const DecodedChunk& chunk,
      uint32_t rowInChunk);

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
};

} // namespace facebook::nimble
