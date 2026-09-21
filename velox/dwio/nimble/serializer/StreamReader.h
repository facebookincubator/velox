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

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <folly/container/F14Map.h>
#include <folly/coro/Task.h>
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/velox/Decoder.h"
#include "velox/dwio/nimble/velox/RowRange.h"
#include "velox/vector/BaseVector.h"

namespace facebook::nimble {

class FieldReader;
class FieldReaderFactory;
class InMemoryChunkedStream;
class Type;

/// Decodes selected rows from the streams of one projected Nimble type through
/// FieldReader. Scalar leaves use Decoder's random-access EncodingView path.
class StreamReader {
 public:
  /// Binds the projected schema and initializes its reusable reader metadata.
  StreamReader(
      std::shared_ptr<const Type> type,
      velox::memory::MemoryPool* pool,
      Encoding::Options options);

  ~StreamReader();

  /// Decodes source ranges contiguously starting at `outputOffset`.
  /// Source ranges must be non-empty, ordered by `startRow`, and disjoint.
  /// `streams` must follow the projected schema's stream traversal order and
  /// contain exactly one encoded chunk per present stream.
  /// String outputs retain backing buffers across calls so values written by
  /// earlier calls remain valid until the assembled output is released.
  void read(
      std::span<const std::string_view> streams,
      std::span<const RowRange> ranges,
      velox::vector_size_t outputOffset,
      velox::VectorPtr& output);

  /// Coroutine form of `read` with the same input and output contract. Inputs
  /// and output must remain valid until the returned task completes.
  folly::coro::Task<void> co_read(
      std::span<const std::string_view> streams,
      std::span<const RowRange> ranges,
      velox::vector_size_t outputOffset,
      velox::VectorPtr& output);

 private:
  // Initializes the schema-derived reader factory and stream offsets.
  void init();

  // Builds the FieldReader tree for one stripe's streams.
  void prepareRead(std::span<const std::string_view> streams);

  // Decodes selected rows through the configured reader tree asynchronously.
  folly::coro::Task<void> co_decodeRows(
      std::span<const RowRange> sourceRanges,
      std::span<const velox::BaseVector::CopyRange> outputRanges,
      velox::VectorPtr& output);

  // Projected schema shared by every stream set decoded by this instance.
  const std::shared_ptr<const Type> type_;

  // Velox counterpart of type_, used when allocating each result.
  const velox::TypePtr outputType_;

  // Owns all buffers allocated while decoding.
  velox::memory::MemoryPool* const pool_;

  // Configures encoding creation and temporary buffer reuse.
  const Encoding::Options options_;

  // Stream offsets in projected-schema traversal order.
  std::vector<uint32_t> readerOffsets_;

  // Builds a FieldReader tree for each stream binding.
  std::unique_ptr<FieldReaderFactory> readerFactory_;

  // Decoders for present streams, keyed by projected stream offset.
  folly::F14FastMap<uint32_t, std::unique_ptr<Decoder>> streamDecoders_;

  // Owns outer-chunk decompression buffers referenced by decoders.
  std::vector<std::unique_ptr<InMemoryChunkedStream>> chunkedStreams_;

  // Root reader for the current stream binding.
  std::unique_ptr<FieldReader> reader_;
};

} // namespace facebook::nimble
