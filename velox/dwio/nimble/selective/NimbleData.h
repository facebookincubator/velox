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

#include <folly/container/F14Set.h>

#include "velox/dwio/common/FormatData.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/selective/ReaderBase.h"
#include "velox/dwio/nimble/selective/RowSizeTracker.h"

namespace facebook::nimble {

class ChunkedDecoder;
class FlatMapKeyNodePool;

class NimbleData : public velox::dwio::common::FormatData {
 public:
  NimbleData(
      const std::shared_ptr<const Type>& nimbleType,
      StripeStreams& streams,
      velox::memory::MemoryPool& memoryPool,
      ChunkedDecoder* inMapDecoder,
      const EncodingFactory& encodingFactory,
      bool preserveDictionaryEncoding = false,
      bool lazyColumnIo = false,
      velox::dwio::common::DecodingStats* decodingStats = nullptr,
      bool dictionaryAwareReads = false,
      bool scanFastPath = false);

  /// Read internal node nulls. For leaf nodes, we only copy `incomingNulls' if
  /// it exists. A nulls-only read of a scalar column reads its nulls from the
  /// value stream instead.
  void readNulls(
      velox::vector_size_t numValues,
      const uint64_t* incomingNulls,
      velox::BufferPtr& nulls,
      bool nullsOnly) override;

  /// With nulls-only reads enabled, a nulls-only skip skips the value stream,
  /// which holds the nulls, and returns 0.
  uint64_t skipNulls(uint64_t numValues, bool nullsOnly) override;

  uint64_t skip(uint64_t numValues) final {
    return skipNulls(numValues, false);
  }

  bool hasNulls() const final {
    return nullsDecoder_ != nullptr;
  }

  velox::dwio::common::PositionProvider seekToRowGroup(
      int64_t /*index*/) final {
    NIMBLE_UNSUPPORTED();
  }

  void filterRowGroups(
      const velox::common::ScanSpec& /*scanSpec*/,
      uint64_t /*rowsPerRowGroup*/,
      const velox::dwio::common::StatsContext& /*writerContext*/,
      FilterRowGroupsResult& /*result*/) final {
    NIMBLE_UNSUPPORTED();
  }

  const Type& nimbleType() const {
    return *nimbleType_;
  }

  ChunkedDecoder makeScalarDecoder();

  /// Rebinds the decoder that this column owns for its nulls, lengths or
  /// offsets stream to the stripe that its StripeStreams now reads. Makes the
  /// decoder if only the new stripe has the stream, and drops it if only the
  /// earlier stripe had it, as a column made for the new stripe would. Scalar
  /// columns decode their nulls with their values and own no such decoder. The
  /// columns of a reader tree share one StripeStreams. The caller moves it to
  /// the new stripe once for all of them with StripeStreams::setStripe()
  /// before this call, and loads the streams enqueued with
  /// StripeStreams::load() after it (see rebindColumnReader()).
  void rebindNullsDecoder();

  /// Drops the stream of the decoder that rebindNullsDecoder() rebinds, and
  /// keeps the decoder and its buffers for a later stripe (see
  /// SelectiveColumnReader::unbind()).
  void unbindNullsDecoder();

  /// Points 'decoder', which makeScalarDecoder() made for an earlier stripe, at
  /// this column's scalar stream in the current stripe. Call it between
  /// StripeStreams::setStripe() and StripeStreams::load(), like
  /// rebindNullsDecoder(). Returns false, having enqueued no stream, if the
  /// current stripe has no such stream.
  bool rebindScalarDecoder(ChunkedDecoder& decoder);

  /// Returns the StripeStreams that this column reads its streams from, which
  /// the columns of a reader tree share.
  const StripeStreams& streams() const {
    return *streams_;
  }

  /// Enables nulls-only reads through 'decoder', which decodes this scalar
  /// column's value stream. Takes effect only when:
  ///
  /// - scanFastPath is on.
  /// - The encodings are non-legacy.
  /// - The column has no in-map stream, which also holds nulls.
  void setNullsOnlyDecoder(ChunkedDecoder* decoder);

  /// Returns true if nulls-only reads are enabled and 'scanSpec' needs only the
  /// column's nulls:
  ///
  /// - An IS NULL filter.
  /// - An IS NOT NULL filter on a column whose values are not kept.
  bool readsNullsOnly(const velox::common::ScanSpec& scanSpec) const;

  // Decoders for Timestamps
  ChunkedDecoder makeMicrosDecoder();
  ChunkedDecoder makeNanosDecoder();

  const velox::BufferPtr& getPreloadedValues();

  const uint64_t* inMap() const {
    return inMapDecoder_ ? inMap_->as<uint64_t>() : nullptr;
  }

  const velox::BufferPtr& inMapBuffer() {
    return inMap_;
  }

  const ChunkedDecoder* nullsDecoder() const {
    return nullsDecoder_.get();
  }

  const ChunkedDecoder* inMapDecoder() const {
    return inMapDecoder_;
  }

  std::unique_ptr<ChunkedDecoder> makeLengthDecoder();

  bool nimblePreserveDictionaryEncoding() const {
    return preserveDictionaryEncoding_;
  }

  /// Loads the deferred lazy clone of this column's streams. Called from the
  /// shared column-loader read path, so it fires for every loader type and the
  /// streams are always resident before decode. No-op unless this column was
  /// given lazy I/O.
  void loadLazyInputStreams() override;

 private:
  std::unique_ptr<ChunkedDecoder> makeDecoder(
      const StreamDescriptor& descriptor,
      bool decodeValuesWithNulls);

  // Makes the decoder of the stream of 'nullsDescriptor_' in the current
  // stripe. Returns nullptr if this column has no such stream, or the current
  // stripe lacks it.
  std::unique_ptr<ChunkedDecoder> makeNullsDecoder();

  // Points 'decoder' at stream 'streamId' of the current stripe. Returns false
  // if the current stripe has no such stream.
  bool rebindDecoder(ChunkedDecoder& decoder, uint32_t streamId);

  const std::shared_ptr<const Type> nimbleType_;

  // Descriptor of the stream that 'nullsDecoder_' decodes: the nulls of a row
  // or flat map, or the lengths or offsets, with their nulls, of the other
  // kinds. Null for the kinds that decode their nulls with their values.
  const StreamDescriptor* const nullsDescriptor_;

  StripeStreams* const streams_;
  velox::memory::MemoryPool* const pool_;
  ChunkedDecoder* const inMapDecoder_;
  const EncodingFactory* const encodingFactory_;
  const bool lazyColumnIo_{false};
  const bool preserveDictionaryEncoding_{false};
  // Per-column decoding statistics. May be nullptr if column stats collection
  // is disabled.
  velox::dwio::common::DecodingStats* const decodingStats_{nullptr};
  const bool dictionaryAwareReads_{false};
  const bool scanFastPath_{false};

  // Decoder of a scalar column's value stream (which holds its nulls), owned by
  // the column reader. Null when nulls-only reads are disabled.
  ChunkedDecoder* nullsOnlyDecoder_{nullptr};
  std::unique_ptr<ChunkedDecoder> nullsDecoder_;
  velox::BufferPtr inMap_;
};

class NimbleParams : public velox::dwio::common::FormatParams {
 public:
  NimbleParams(
      velox::memory::MemoryPool& pool,
      velox::dwio::common::SplitStats& stats,
      const std::shared_ptr<const Type>& nimbleType,
      StripeStreams& streams,
      RowSizeTracker* rowSizeTracker,
      const EncodingFactory& encodingFactory,
      bool preserveFlatMapsInMemory = false,
      bool nimblePreserveDictionaryEncoding = false,
      const folly::F14FastSet<std::string>* lazyIoColumns = nullptr,
      bool lazyColumnIo = false,
      bool dictionaryAwareReads = false,
      bool scanFastPath = false,
      FlatMapKeyNodePool* flatMapKeyNodePool = nullptr,
      bool generateLazyChildren = true)
      : FormatParams(pool, stats),
        nimbleType_(nimbleType),
        streams_(&streams),
        rowSizeTracker_(rowSizeTracker),
        preserveFlatMapsInMemory_(preserveFlatMapsInMemory),
        lazyIoColumns_(lazyIoColumns),
        encodingFactory_(&encodingFactory),
        nimblePreserveDictionaryEncoding_{nimblePreserveDictionaryEncoding},
        dictionaryAwareReads_{dictionaryAwareReads},
        scanFastPath_{scanFastPath},
        flatMapKeyNodePool_{flatMapKeyNodePool},
        generateLazyChildren_{generateLazyChildren},
        lazyColumnIo_(lazyColumnIo) {}

  std::unique_ptr<velox::dwio::common::FormatData> toFormatData(
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& /*type*/,
      const velox::common::ScanSpec& /*scanSpec*/) override;

  NimbleParams makeChildParams(const std::shared_ptr<const Type>& type) {
    NimbleParams childParams(
        pool(),
        splitStats(),
        type,
        *streams_,
        rowSizeTracker_,
        *encodingFactory_,
        preserveFlatMapsInMemory_,
        nimblePreserveDictionaryEncoding_,
        /*lazyIoColumns=*/nullptr,
        lazyColumnIo_,
        dictionaryAwareReads_,
        scanFastPath_,
        flatMapKeyNodePool_,
        generateLazyChildren_);
    return childParams;
  }

  const std::shared_ptr<const Type>& nimbleType() const {
    return nimbleType_;
  }

  StripeStreams& streams() {
    return *streams_;
  }

  /// Marks this column for lazy I/O routing. Called by StructColumnReader
  /// for top-level columns eligible for lazy I/O.
  void setLazyColumnIo(bool lazyColumnIo) {
    lazyColumnIo_ = lazyColumnIo;
  }

  /// Returns true if this column's streams should be routed to the lazy
  /// input clone instead of the main (eager) input.
  bool lazyColumnIo() const {
    return lazyColumnIo_;
  }

  void setInMapDecoder(ChunkedDecoder* decoder) {
    inMapDecoder_ = decoder;
  }

  FlatMapKeyNodePool* flatMapKeyNodePool() const {
    return flatMapKeyNodePool_;
  }

  bool preserveFlatMapsInMemory() const {
    return preserveFlatMapsInMemory_;
  }

  /// Returns true if any top-level columns are marked for lazy I/O.
  bool hasLazyIoColumns() const {
    return lazyIoColumns_ != nullptr;
  }

  /// Returns true if the top-level column 'name' should use lazy I/O.
  bool lazyIoColumn(const std::string& name) const {
    return lazyIoColumns_ != nullptr && lazyIoColumns_->count(name) > 0;
  }

  RowSizeTracker* rowSizeTracker() const {
    return rowSizeTracker_;
  }

  const EncodingFactory& encodingFactory() const {
    return *encodingFactory_;
  }

  bool dictionaryAwareReads() const {
    return dictionaryAwareReads_;
  }

  bool scanFastPath() const {
    return scanFastPath_;
  }

  /// Returns true if the struct readers built with these params return the
  /// fields that they read as top level as lazy vectors, as DWRF's struct
  /// reader does unless it is built without lazy children.
  bool generateLazyChildren() const {
    return generateLazyChildren_;
  }

 private:
  const std::shared_ptr<const Type> nimbleType_;
  StripeStreams* const streams_{nullptr};
  RowSizeTracker* const rowSizeTracker_{nullptr};
  const bool preserveFlatMapsInMemory_{false};
  const folly::F14FastSet<std::string>* const lazyIoColumns_{nullptr};
  const EncodingFactory* const encodingFactory_;
  const bool nimblePreserveDictionaryEncoding_{false};
  const bool dictionaryAwareReads_{false};
  const bool scanFastPath_{false};

  // Lets the FlatMap readers built with these params, or with params derived
  // from them, rebind the key readers kept in it, and keeps theirs when they
  // are destroyed. Null builds every key reader anew.
  FlatMapKeyNodePool* const flatMapKeyNodePool_{nullptr};

  // When false, the struct readers built with these params, or with params
  // derived from them, read their top-level fields eagerly, which suits a
  // reader that copies every row that it reads right away.
  const bool generateLazyChildren_{true};

  // When true, enqueue() routes to the lazy input clone instead of the
  // main eager input. Set for lazy columns; propagated to children via
  // makeChildParams() so nested streams inherit the parent's routing.
  bool lazyColumnIo_{false};
  ChunkedDecoder* inMapDecoder_{nullptr};
};

} // namespace facebook::nimble
