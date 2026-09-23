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

#include "velox/dwio/nimble/serializer/StreamReader.h"

#include <limits>
#include <utility>

#include <folly/coro/BlockingWait.h>
#include "velox/common/Casts.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/nimble/common/NimbleException.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/serializer/EncodingViewDecoder.h"
#include "velox/dwio/nimble/velox/ChunkedStream.h"
#include "velox/dwio/nimble/velox/FieldReader.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"

namespace facebook::nimble {
namespace {

// Exposes already-loaded bytes through the ChunkedStream interface.
class StringViewStreamLoader final : public StreamLoader {
 public:
  explicit StringViewStreamLoader(std::string_view stream) : stream_{stream} {}

  const std::string_view getStream() const override {
    return stream_;
  }

 private:
  const std::string_view stream_;
};

// Validates one request and builds its dense source-to-output mapping.
std::vector<velox::BaseVector::CopyRange> makeOutputRanges(
    size_t numStreams,
    size_t expectedNumStreams,
    const velox::TypePtr& outputType,
    std::span<const RowRange> ranges,
    velox::vector_size_t outputOffset,
    velox::VectorPtr& output) {
  NIMBLE_CHECK_NOT_NULL(output);
  NIMBLE_CHECK(
      output->type()->equivalent(*outputType),
      "Output type does not match the projected schema: {} vs. {}",
      output->type()->toString(),
      outputType->toString());
  NIMBLE_CHECK_EQ(
      numStreams,
      expectedNumStreams,
      "Stream count must match the projected schema");
  NIMBLE_CHECK(!ranges.empty(), "Read ranges must not be empty");
  NIMBLE_CHECK_GE(outputOffset, 0, "Output offset must be non-negative");

  uint64_t numReadRows{0};
  for (size_t i{0}; i < ranges.size(); ++i) {
    const auto& range = ranges[i];
    NIMBLE_CHECK(!range.empty(), "Read range must not be empty");
    if (i > 0) {
      NIMBLE_CHECK_LE(
          ranges[i - 1].endRow,
          range.startRow,
          "Read ranges must be ordered and disjoint");
    }

    const auto numRows = range.numRows();
    NIMBLE_CHECK_LE(
        numRows,
        std::numeric_limits<velox::vector_size_t>::max() - numReadRows,
        "Read rows exceed the Velox vector row limit");
    numReadRows += numRows;
  }

  const auto outputEnd = static_cast<uint64_t>(outputOffset) + numReadRows;
  NIMBLE_CHECK_LE(
      outputEnd,
      static_cast<uint64_t>(output->size()),
      "Read rows exceed the output vector");

  std::vector<velox::BaseVector::CopyRange> outputRanges;
  outputRanges.reserve(ranges.size());
  auto nextOutputOffset = outputOffset;
  velox::vector_size_t nextSourceOffset{0};
  for (const auto& range : ranges) {
    outputRanges.push_back({
        .sourceIndex = nextSourceOffset,
        .targetIndex = nextOutputOffset,
        .count = static_cast<velox::vector_size_t>(range.numRows()),
    });
    nextSourceOffset += static_cast<velox::vector_size_t>(range.numRows());
    nextOutputOffset += static_cast<velox::vector_size_t>(range.numRows());
  }
  return outputRanges;
}

} // namespace

StreamReader::StreamReader(
    std::shared_ptr<const Type> type,
    velox::memory::MemoryPool* pool,
    Encoding::Options options)
    : type_{std::move(type)},
      outputType_{convertToVeloxType(*velox::checkedNotNull(type_.get()))},
      pool_{velox::checkedNotNull(pool)},
      options_{std::move(options)} {
  NIMBLE_CHECK_NOT_NULL(outputType_);

  init();
}

void StreamReader::init() {
  const auto typeWithId =
      std::shared_ptr<const velox::dwio::common::TypeWithId>{
          velox::dwio::common::TypeWithId::create(outputType_)};
  auto readerParams = FieldReaderParams{};
  readerParams.optimizeStringBufferHandling = true;
  readerFactory_ = FieldReaderFactory::create(
      readerParams,
      type_,
      typeWithId,
      readerOffsets_,
      [](uint32_t /*nodeId*/) { return true; },
      pool_);
}

StreamReader::~StreamReader() = default;

void StreamReader::read(
    std::span<const std::string_view> streams,
    std::span<const RowRange> ranges,
    velox::vector_size_t outputOffset,
    velox::VectorPtr& output) {
  folly::coro::blockingWait(co_read(streams, ranges, outputOffset, output));
}

folly::coro::Task<void> StreamReader::co_read(
    std::span<const std::string_view> streams,
    std::span<const RowRange> ranges,
    velox::vector_size_t outputOffset,
    velox::VectorPtr& output) {
  const auto outputRanges = makeOutputRanges(
      streams.size(),
      readerOffsets_.size(),
      outputType_,
      ranges,
      outputOffset,
      output);
  prepareRead(streams);
  co_await co_decodeRows(ranges, outputRanges, output);
}

void StreamReader::prepareRead(std::span<const std::string_view> streams) {
  const auto makeDecoder = [this](std::string_view stream) {
    auto chunkedStream = std::make_unique<InMemoryChunkedStream>(
        *pool_, std::make_unique<StringViewStreamLoader>(stream));
    NIMBLE_CHECK(
        chunkedStream->hasNext(), "StreamReader requires a non-empty stream");
    const auto encoded = chunkedStream->nextChunk();
    NIMBLE_CHECK(
        !chunkedStream->hasNext(),
        "StreamReader requires exactly one encoded chunk per stream");
    auto decoder = std::make_unique<EncodingViewDecoder>(
        encoded, pool_, options_.bufferPool, [this](std::string_view encoding) {
          return createEncodingView(encoding, pool_, options_);
        });
    chunkedStreams_.push_back(std::move(chunkedStream));
    return decoder;
  };

  // TODO: Cache reader trees by stream-presence pattern if rebuild cost
  // becomes material.
  reader_.reset();
  streamDecoders_.clear();
  chunkedStreams_.clear();
  streamDecoders_.reserve(streams.size());
  chunkedStreams_.reserve(streams.size());
  for (size_t i{0}; i < streams.size(); ++i) {
    if (!streams[i].empty()) {
      streamDecoders_.emplace(readerOffsets_[i], makeDecoder(streams[i]));
    }
  }
  reader_ = readerFactory_->createReader(streamDecoders_);
}

folly::coro::Task<void> StreamReader::co_decodeRows(
    std::span<const RowRange> sourceRanges,
    std::span<const velox::BaseVector::CopyRange> outputRanges,
    velox::VectorPtr& output) {
  co_await reader_->co_read(sourceRanges, outputRanges, output);
}

} // namespace facebook::nimble
