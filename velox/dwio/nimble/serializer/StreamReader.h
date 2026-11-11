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
#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/RowRange.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/reader/Decoder.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::nimble {

class EncodingView;
class FieldReader;
class FieldReaderFactory;
class Type;

/// Decodes selected rows from the streams of one projected Nimble type through
/// FieldReader. Scalar leaves use Decoder's random-access EncodingView path.
/// Requires an EncodingView for the encoding of every chunk.
class StreamReader {
 public:
  /// Binds the projected schema and initializes its reusable reader metadata.
  StreamReader(
      std::shared_ptr<const Type> type,
      velox::memory::MemoryPool* pool);

  ~StreamReader();

  /// Decodes source ranges contiguously starting at `outputOffset`.
  /// Source ranges must be non-empty, ordered by `startRow`, and disjoint.
  /// `streams` must follow the projected schema's serialization traversal
  /// order. A present stream can hold several chunks. `options` configure the
  /// encodings of the chunks as in setStreams().
  /// String outputs retain backing buffers across calls so values written by
  /// earlier calls remain valid until the assembled output is released.
  /// Inputs and output must remain valid until the returned task completes.
  folly::coro::Task<void> co_read(
      std::span<const std::string_view> streams,
      std::span<const RowRange> ranges,
      velox::vector_size_t outputOffset,
      const Encoding::Options& options,
      velox::VectorPtr& output);

  /// Decodes source ranges of the stripe bound by setStreams() contiguously
  /// starting at `outputOffset`, without rebinding it. Ranges, output and
  /// lifetime requirements match the overload above.
  folly::coro::Task<void> co_read(
      std::span<const RowRange> ranges,
      velox::vector_size_t outputOffset,
      velox::VectorPtr& output);

  /// Binds one stripe's already-loaded streams for decoding. `streams` must
  /// follow the same layout as in co_read(). The stream storage must outlive
  /// this binding. `options` configure the encodings of the chunks of this
  /// binding only, so each stripe can decode with options of its own. The
  /// decoders and the views of their encodings take scratch buffers from
  /// `options.bufferPool` when it is not null. The pool must outlive this
  /// binding, which ends with reset(), the next binding or the destruction of
  /// the reader, and must not serve another thread until then.
  void setStreams(
      std::span<const std::string_view> streams,
      const Encoding::Options& options);

  /// Decodes the next rows from the currently bound stripe asynchronously.
  /// Reuses uniquely owned vectors. Supplied output must match the projected
  /// schema; reset incompatible output before calling.
  /// `output` must remain valid until the returned task completes.
  folly::coro::Task<void> co_next(uint32_t numRows, velox::VectorPtr& output);

  /// Advances within the currently bound stripe asynchronously without
  /// materializing rows.
  folly::coro::Task<void> co_skip(uint32_t numRows);

  /// Releases the currently bound stripe. The cached reader stays for the next
  /// stripe.
  void reset();

 private:
  // Keeps the decoders of the present streams together with the FieldReader
  // tree built over them for one stream-presence pattern. Binding another
  // stripe with the same pattern replaces the decoders' views and keeps the
  // tree. Another pattern keeps the decoders of the streams that stay present
  // and rebuilds the tree.
  struct CachedReader;

  // Initializes the schema-derived reader factory and stream offsets.
  void init();

  // Binds one stripe's streams with 'options', reusing the cached reader for
  // their stream-presence pattern when there is one. Requires one stream per
  // projected stream.
  void prepareRead(
      std::span<const std::string_view> streams,
      const Encoding::Options& options);

  // Splits one stream into the encodings of its chunks, in stream order, and
  // stores them in loadedChunks_. Decompresses the encodings of compressed
  // chunks into buffers that stay alive until the next reset().
  void loadChunks(std::string_view stream);

  // Rebuilds the FieldReader tree over one stripe's streams in the cached
  // reader. Rebinds the decoders of the streams that stay present, creates the
  // decoders of new streams and drops those of absent streams.
  void rebuildReader(std::span<const std::string_view> streams);

  // Creates the view of the encoding of one chunk for the decoders.
  std::shared_ptr<const EncodingView> makeEncodingView(
      std::string_view encoding) const;

  // Returns true when a stripe is bound to the cached reader.
  bool hasBoundStreams() const {
    return streamsBound_;
  }

  // Returns the cached reader of the bound stripe. Fails when no stripe is
  // bound.
  CachedReader& reader() const;

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

  // Configures the encodings of the bound stripe. Each binding sets it, and the
  // decoders read it when they create the views of the chunks of the binding.
  Encoding::Options options_;

  // Descriptor offsets corresponding positionally to input projection streams.
  std::vector<uint32_t> inputOffsets_;

  // Builds a FieldReader tree for each stream binding.
  std::unique_ptr<FieldReaderFactory> readerFactory_;

  // Holds the reader of the most recently bound stream-presence pattern. A
  // stripe with the same pattern reuses it, and a stripe with any other pattern
  // rebuilds its tree in place.
  std::unique_ptr<CachedReader> cachedReader_;

  // Holds the presence key of cachedReader_. Bit i of a key is set when stream
  // i is present.
  Vector<uint8_t> cachedKey_;

  // Keeps the decompressed bytes of compressed chunks alive until reset(),
  // because the decoders' views point into them. Buffers from every stream of
  // the bound stripe accumulate in load order. Nothing looks them up, so the
  // order has no meaning.
  std::vector<velox::BufferPtr> decompressedChunks_;

  // Holds the encodings of the stream that loadChunks() split last, one per
  // chunk in stream order, for the decoder that binds them next. Reused across
  // calls to avoid an allocation per stream. Owns no bytes: each view points
  // into the stream or into a buffer in decompressedChunks_.
  std::vector<std::string_view> loadedChunks_;

  // Set while a stripe is bound to cachedReader_.
  bool streamsBound_{false};
};

} // namespace facebook::nimble
