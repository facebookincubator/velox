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
#include "velox/dwio/nimble/velox/selective/ReaderBase.h"
#include "velox/dwio/nimble/velox/selective/RowSizeTracker.h"

namespace facebook::nimble {

class ChunkedDecoder;

class NimbleData : public velox::dwio::common::FormatData {
 public:
  NimbleData(
      const std::shared_ptr<const Type>& nimbleType,
      StripeStreams& streams,
      velox::memory::MemoryPool& memoryPool,
      ChunkedDecoder* inMapDecoder,
      const EncodingFactory& encodingFactory,
      bool stringDecoderZeroCopy,
      bool nimblePreserveDictionaryEncoding = false,
      bool lazyColumnIo = false,
      velox::dwio::common::DecodingStats* decodingStats = nullptr);

  /// Read internal node nulls. For leaf nodes, we only copy `incomingNulls' if
  /// it exists.
  void readNulls(
      velox::vector_size_t numValues,
      const uint64_t* incomingNulls,
      velox::BufferPtr& nulls,
      bool nullsOnly) override;

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
    return nimblePreserveDictionaryEncoding_;
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

  const std::shared_ptr<const Type> nimbleType_;
  StripeStreams* const streams_;
  velox::memory::MemoryPool* const pool_;
  ChunkedDecoder* const inMapDecoder_;
  std::unique_ptr<ChunkedDecoder> nullsDecoder_;
  velox::BufferPtr inMap_;
  const EncodingFactory* const encodingFactory_;
  bool nimblePreserveDictionaryEncoding_{false};
  bool lazyColumnIo_{false};
  // Per-column decoding statistics. May be nullptr if column stats collection
  // is disabled.
  velox::dwio::common::DecodingStats* const decodingStats_{nullptr};
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
      bool stringDecoderZeroCopy = false,
      bool preserveFlatMapsInMemory = false,
      bool nimblePreserveDictionaryEncoding = false,
      const folly::F14FastSet<std::string>* lazyIoColumns = nullptr,
      bool lazyColumnIo = false)
      : FormatParams(pool, stats),
        nimbleType_(nimbleType),
        streams_(&streams),
        rowSizeTracker_(rowSizeTracker),
        preserveFlatMapsInMemory_(preserveFlatMapsInMemory),
        lazyIoColumns_(lazyIoColumns),
        encodingFactory_(&encodingFactory),
        lazyColumnIo_(lazyColumnIo),
        stringDecoderZeroCopy_{stringDecoderZeroCopy},
        nimblePreserveDictionaryEncoding_{nimblePreserveDictionaryEncoding} {}

  std::unique_ptr<velox::dwio::common::FormatData> toFormatData(
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& /*type*/,
      const velox::common::ScanSpec& /*scanSpec*/) override;

  NimbleParams makeChildParams(const std::shared_ptr<const Type>& type) {
    return NimbleParams(
        pool(),
        splitStats(),
        type,
        *streams_,
        rowSizeTracker_,
        *encodingFactory_,
        stringDecoderZeroCopy_,
        preserveFlatMapsInMemory_,
        nimblePreserveDictionaryEncoding_,
        /*lazyIoColumns=*/nullptr,
        lazyColumnIo_);
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

 private:
  const std::shared_ptr<const Type> nimbleType_;
  StripeStreams* const streams_{nullptr};
  RowSizeTracker* const rowSizeTracker_{nullptr};
  const bool preserveFlatMapsInMemory_{false};
  const folly::F14FastSet<std::string>* const lazyIoColumns_{nullptr};
  const EncodingFactory* const encodingFactory_;
  // When true, enqueue() routes to the lazy input clone instead of the
  // main eager input. Set for lazy columns; propagated to children via
  // makeChildParams() so nested streams inherit the parent's routing.
  bool lazyColumnIo_{false};
  ChunkedDecoder* inMapDecoder_{nullptr};
  bool stringDecoderZeroCopy_{false};
  bool nimblePreserveDictionaryEncoding_{false};
};

} // namespace facebook::nimble
