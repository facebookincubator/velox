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
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <folly/Executor.h>
#include <folly/container/F14Map.h>
#include <folly/coro/Task.h>
#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/RowRange.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/views/DecodedStringBuffers.h"
#include "velox/dwio/nimble/reader/Decoder.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::nimble {

class EncodingView;
class FieldReader;
class FieldReaderFactory;
class SharedDictionaryAlphabet;
class Type;

/// Decodes selected rows from the streams of one projected Nimble type through
/// FieldReader. Scalar leaves use Decoder's random-access EncodingView path.
/// Requires an EncodingView for the encoding of every chunk.
class StreamReader {
 public:
  /// Maps the index of a stream in the layout of co_read() and setStreams() to
  /// the alphabet of its shared dictionary. The index, not a descriptor
  /// offset, names the stream, because a projected type numbers its streams
  /// apart from the file it was projected from.
  using SharedDictionaryAlphabets = folly::
      F14FastMap<uint32_t, std::shared_ptr<const SharedDictionaryAlphabet>>;

  /// Configures how the reader decodes the fields of a stripe.
  struct Options {
    /// Runs the tasks that decode child fields in parallel. Null decodes
    /// serially. Must outlive the reader.
    folly::Executor* decodeExecutor{nullptr};

    /// Maximum parallel decode tasks across the field reader tree. Zero or one
    /// decodes serially.
    uint32_t maxDecodeParallelism{0};

    /// Minimum nested scalar streams per planned decode task, so that each
    /// task has enough work to amortize scheduling it.
    uint32_t minStreamsPerDecodeTask{1};

    /// Returns true when the reader can decode child fields in parallel.
    bool parallelDecodeEnabled() const {
      return decodeExecutor != nullptr && maxDecodeParallelism > 1;
    }
  };

  /// Binds the projected schema and initializes its reusable reader metadata.
  /// Decodes child fields as 'options' configure.
  StreamReader(
      std::shared_ptr<const Type> type,
      velox::memory::MemoryPool* pool,
      Options options);

  ~StreamReader();

  /// Decodes source ranges contiguously starting at `outputOffset`.
  /// Source ranges must be non-empty, ordered by `startRow`, and disjoint.
  /// `streams` must follow the projected schema's serialization traversal
  /// order. A present stream can hold several chunks. `options` configure the
  /// encodings of the chunks as in setStreams(). Binds no shared dictionary
  /// alphabets: read streams encoded with them through setStreams().
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
  /// this binding.
  ///
  /// `streamBufferHolders` holds the bytes of the streams: it is empty, or has
  /// one buffer per stream, aligned with `streams`, null for a stream without
  /// one. Streams can share a buffer, which must hold all of each stream. A
  /// read of strings that covers at least a quarter of the rows of the chunks
  /// it spans keeps the values that lie in its stream's buffer, or in the
  /// buffers that the encodings' views decompress or decode them into, by
  /// holding the buffers in the output instead of copying the values (see
  /// EncodingViewDecoder). Without a buffer, the values that lie in the
  /// stream are copied.
  ///
  /// `alphabets` maps the index in `streams` of each stream encoded with a
  /// shared dictionary to its alphabet. A file or external dictionary is the
  /// same in every stripe of a file, while a stripe dictionary differs from
  /// stripe to stripe, so each binding passes its own.
  /// `options.sharedDictionaryAlphabet` must be null when `alphabets` is not
  /// empty. The views of the chunks keep the alphabets they use, so the map
  /// need not outlive this call.
  ///
  /// `options` configure the encodings of the chunks of this binding only, so
  /// each stripe can decode with options of its own. The decoders and the
  /// views of their encodings take scratch buffers from `options.bufferPool`
  /// when it is not null. The pool must outlive this binding, which ends with
  /// reset(), the next binding or the destruction of the reader, and must not
  /// serve another thread until then. With `options.cacheEncodingView`, the
  /// views of the chunks of each stream decode strings into buffers that they
  /// share, which the reader creates for each stream of each binding, so that
  /// a read of a stream holds a few buffers instead of at least one for each
  /// chunk. A dense read of FSST strings, such as a scan, decodes its rows
  /// straight into them, and holds just the buffers of its rows.
  /// `options.decodedStringBuffers` must be null.
  void setStreams(
      std::span<const std::string_view> streams,
      std::span<const velox::BufferPtr> streamBufferHolders,
      const SharedDictionaryAlphabets& alphabets,
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

  // Binds one stripe's streams with 'options' and 'alphabets',
  // reusing the cached reader for their stream-presence pattern when there is
  // one. Requires one stream per projected stream. 'streamBufferHolders' is
  // empty, or has one buffer or null per stream, as in setStreams().
  void prepareRead(
      std::span<const std::string_view> streams,
      std::span<const velox::BufferPtr> streamBufferHolders,
      const SharedDictionaryAlphabets& alphabets,
      const Encoding::Options& options);

  // Splits one stream into the encodings of its chunks, in stream order, and
  // stores them in loadedChunks_, and the buffer that holds each in
  // loadedChunkBufferHolders_: 'streamBufferHolder', which can be null, for
  // an uncompressed chunk, and its decompressed bytes for a compressed chunk.
  void loadChunks(
      std::string_view stream,
      const velox::BufferPtr& streamBufferHolder);

  // Rebuilds the FieldReader tree over one stripe's streams in the cached
  // reader. Rebinds the decoders of the streams that stay present, creates the
  // decoders of new streams and drops those of absent streams.
  void rebuildReader(
      std::span<const std::string_view> streams,
      std::span<const velox::BufferPtr> streamBufferHolders,
      const SharedDictionaryAlphabets& alphabets);

  // Loads the chunks of the stream at 'streamIndex' in 'streams', laid out as
  // in co_read(), with its buffer in 'streamBufferHolders', and runs
  // 'bindDecoder' to bind a decoder to them. Meanwhile options_ carries the
  // stream's alphabet in 'alphabets' if it has one, so the views that the
  // decoder creates read SharedDictionary chunks with it. The views keep the
  // alphabet, and options_ drops it on return, so nothing refers to the map
  // after this call. Meanwhile options_ also points to the stream's buffers
  // from ensureDecodedStringBuffers(), if any, which the views of the
  // stream's chunks share.
  void bindStream(
      std::span<const std::string_view> streams,
      std::span<const velox::BufferPtr> streamBufferHolders,
      size_t streamIndex,
      const SharedDictionaryAlphabets& alphabets,
      const std::function<void()>& bindDecoder);

  // Creates the view of the encoding of one chunk for the decoders, with
  // options_ and 'inputStringBuffer', the buffer that holds the chunk or null,
  // as what its Encoding::Options::inputStringBuffer points to. The decoder
  // keeps the buffer while the view is bound.
  std::shared_ptr<const EncodingView> makeEncodingView(
      std::string_view encoding,
      const velox::BufferPtr& inputStringBuffer);

  // Returns the buffers that the cached views of the chunks in loadedChunks_,
  // those of the stream at 'streamIndex', share to decode strings into, which
  // it creates at the first binding of the stream and reuses after. Returns
  // null without cacheEncodingView, and for a stream of values other than
  // strings, whose views do not use them.
  DecodedStringBuffers* ensureDecodedStringBuffers(size_t streamIndex);

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

  // Configures the parallel decoding of child fields.
  const Options readerOptions_;

  // Configures the encodings of the bound stripe. Each binding sets it, and the
  // decoders read it when they create the views of the chunks of the binding.
  Encoding::Options options_;

  // Descriptor offsets corresponding positionally to input projection streams.
  std::vector<uint32_t> inputOffsets_;

  // Builds a FieldReader tree for each stream binding.
  std::unique_ptr<FieldReaderFactory> readerFactory_;

  // Holds the buffers that the views of the chunks of each stream of strings
  // decode strings into, in the layout of co_read(), or null for a stream
  // that has bound no strings. ensureDecodedStringBuffers() creates them, and
  // reset() clears them for the next stripe. Declared before cachedReader_, so
  // that they outlive the views that point to them.
  std::vector<std::unique_ptr<DecodedStringBuffers>> decodedStringBuffers_;

  // Holds the reader of the most recently bound stream-presence pattern. A
  // stripe with the same pattern reuses it, and a stripe with any other pattern
  // rebuilds its tree in place.
  std::unique_ptr<CachedReader> cachedReader_;

  // Holds the presence key of cachedReader_. Bit i of a key is set when stream
  // i is present.
  Vector<uint8_t> cachedKey_;

  // Holds the encodings of the stream that loadChunks() split last, one per
  // chunk in stream order, for the decoder that binds them next. Reused across
  // calls to avoid an allocation per stream. Owns no bytes: each view points
  // into the stream or into a buffer in loadedChunkBufferHolders_.
  std::vector<std::string_view> loadedChunks_;

  // Holds the buffer that holds each of loadedChunks_, or null, for the
  // decoder that binds them next. The views of the chunks keep their buffers,
  // which outlive this list, which reset() empties.
  std::vector<velox::BufferPtr> loadedChunkBufferHolders_;

  // Set while a stripe is bound to cachedReader_.
  bool streamsBound_{false};
};

} // namespace facebook::nimble
