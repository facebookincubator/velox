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

#include <algorithm>
#include <atomic>
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

void validateFieldReaderOffsets(
    std::span<const uint32_t> projectionOffsets,
    std::vector<uint32_t> readerOffsets) {
  NIMBLE_CHECK_EQ(
      projectionOffsets.size(),
      readerOffsets.size(),
      "Projection and FieldReader stream counts must match");
  auto sortedProjectionOffsets =
      std::vector<uint32_t>{projectionOffsets.begin(), projectionOffsets.end()};
  std::sort(sortedProjectionOffsets.begin(), sortedProjectionOffsets.end());
  std::sort(readerOffsets.begin(), readerOffsets.end());
  NIMBLE_CHECK(
      sortedProjectionOffsets == readerOffsets,
      "Projection and FieldReader stream offsets must match");
}

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

class CachedStripeStreams::State {
 public:
  using ViewMap =
      folly::F14FastMap<size_t, std::shared_ptr<const EncodingView>>;

  explicit State(size_t numStreams) {
    views.reserve(numStreams);
    for (size_t i{0}; i < numStreams; ++i) {
      views.push_back(std::make_unique<folly::Synchronized<ViewMap>>());
    }
  }

  std::vector<std::unique_ptr<folly::Synchronized<ViewMap>>> views;
  std::atomic<uint64_t> numCreated{0};
  std::atomic<uint64_t> numHits{0};
  std::atomic<uint64_t> numStreamChunksParsed{0};
};

CachedStripeStreams::CachedStripeStreams(
    std::shared_ptr<void> bytesOwner,
    std::vector<std::string_view> streams,
    velox::memory::MemoryPool* viewPool,
    Encoding::Options encodingOptions,
    Options options)
    : bytesOwner_{std::move(bytesOwner)},
      streams_{std::move(streams)},
      viewPool_{viewPool},
      encodingOptions_{std::move(encodingOptions)},
      options_{options},
      state_{std::make_unique<State>(streams_.size())} {
  NIMBLE_CHECK_NOT_NULL(viewPool_);
  if (!options_.enableEncodingViewCache) {
    return;
  }

  preparsedStreamChunks_.resize(streams_.size());
  for (size_t i{0}; i < streams_.size(); ++i) {
    const auto stream = streams_[i];
    if (stream.empty()) {
      continue;
    }
    InMemoryChunkedStream chunkedStream{
        *viewPool_, std::make_unique<StringViewStreamLoader>(stream)};
    NIMBLE_CHECK(
        chunkedStream.hasNext(), "StreamReader requires a non-empty stream");
    const auto encoded = chunkedStream.nextChunk();
    NIMBLE_CHECK(
        !chunkedStream.hasNext(),
        "StreamReader requires exactly one encoded chunk per stream");
    NIMBLE_CHECK_GE(
        encoded.data(),
        stream.data(),
        "Caching an outer-compressed stream chunk is not supported");
    NIMBLE_CHECK_LE(
        encoded.end(),
        stream.end(),
        "Caching an outer-compressed stream chunk is not supported");
    preparsedStreamChunks_[i] = encoded;
    ++state_->numStreamChunksParsed;
  }
}

CachedStripeStreams::~CachedStripeStreams() = default;

std::shared_ptr<const EncodingView> CachedStripeStreams::encodingView(
    size_t streamIndex,
    std::string_view encodedChunk) const {
  NIMBLE_CHECK_LT(streamIndex, streams_.size());
  const auto createView = [&]() -> std::shared_ptr<const EncodingView> {
    ++state_->numCreated;
    return createEncodingView(encodedChunk, viewPool_, encodingOptions_);
  };
  if (!options_.enableEncodingViewCache) {
    return createView();
  }

  const auto stream = streams_[streamIndex];
  NIMBLE_CHECK_GE(
      encodedChunk.data(),
      stream.data(),
      "Caching an outer-compressed stream chunk is not supported");
  NIMBLE_CHECK_LE(
      encodedChunk.end(),
      stream.end(),
      "Caching an outer-compressed stream chunk is not supported");
  const auto chunkOffset =
      static_cast<size_t>(encodedChunk.data() - stream.data());
  auto views = state_->views[streamIndex]->wlock();
  if (const auto it = views->find(chunkOffset); it != views->end()) {
    ++state_->numHits;
    return it->second;
  }
  auto view = createView();
  views->emplace(chunkOffset, view);
  return view;
}

uint64_t CachedStripeStreams::numEncodingViewsCreated() const {
  return state_->numCreated.load(std::memory_order_relaxed);
}

uint64_t CachedStripeStreams::numEncodingViewCacheHits() const {
  return state_->numHits.load(std::memory_order_relaxed);
}

std::string_view CachedStripeStreams::preparsedStreamChunk(
    size_t streamIndex) const {
  NIMBLE_CHECK(options_.enableEncodingViewCache);
  NIMBLE_CHECK_LT(streamIndex, preparsedStreamChunks_.size());
  return preparsedStreamChunks_[streamIndex];
}

uint64_t CachedStripeStreams::numStreamChunksParsed() const {
  return state_->numStreamChunksParsed.load(std::memory_order_relaxed);
}

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
  std::vector<uint32_t> readerOffsets;
  readerFactory_ = FieldReaderFactory::create(
      readerParams,
      type_,
      typeWithId,
      readerOffsets,
      [](uint32_t /*nodeId*/) { return true; },
      pool_);
  inputOffsets_ = projectionStreamOffsets(*type_);
  validateFieldReaderOffsets(inputOffsets_, std::move(readerOffsets));
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
      inputOffsets_.size(),
      outputType_,
      ranges,
      outputOffset,
      output);
  prepareRead(
      std::make_shared<CachedStripeStreams>(
          /*bytesOwner=*/nullptr,
          std::vector<std::string_view>{streams.begin(), streams.end()},
          pool_,
          options_,
          CachedStripeStreams::Options{.enableEncodingViewCache = false}));
  co_await co_decodeRows(ranges, outputRanges, output);
}

folly::coro::Task<void> StreamReader::co_read(
    std::shared_ptr<CachedStripeStreams> stripe,
    std::span<const RowRange> ranges,
    velox::vector_size_t outputOffset,
    velox::VectorPtr& output) {
  NIMBLE_CHECK_NOT_NULL(stripe);
  const auto outputRanges = makeOutputRanges(
      stripe->streams().size(),
      inputOffsets_.size(),
      outputType_,
      ranges,
      outputOffset,
      output);
  prepareRead(std::move(stripe));
  co_await co_decodeRows(ranges, outputRanges, output);
}

void StreamReader::prepareRead(std::shared_ptr<CachedStripeStreams> stripe) {
  NIMBLE_CHECK_NOT_NULL(stripe);
  const auto& streams = stripe->streams();
  const auto makeDecoder = [this, stripe](
                               std::string_view stream, size_t streamIndex) {
    std::string_view encoded;
    if (stripe->hasPreparsedStreamChunks()) {
      encoded = stripe->preparsedStreamChunk(streamIndex);
    } else {
      ++numStreamChunksParsed_;
      auto chunkedStream = std::make_unique<InMemoryChunkedStream>(
          *pool_, std::make_unique<StringViewStreamLoader>(stream));
      NIMBLE_CHECK(
          chunkedStream->hasNext(), "StreamReader requires a non-empty stream");
      encoded = chunkedStream->nextChunk();
      NIMBLE_CHECK(
          !chunkedStream->hasNext(),
          "StreamReader requires exactly one encoded chunk per stream");
      chunkedStreams_.push_back(std::move(chunkedStream));
    }
    auto decoder = std::make_unique<EncodingViewDecoder>(
        encoded,
        pool_,
        options_.bufferPool,
        [stripe, streamIndex](std::string_view encoding) {
          return stripe->encodingView(streamIndex, encoding);
        });
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
      streamDecoders_.emplace(inputOffsets_[i], makeDecoder(streams[i], i));
    }
  }
  reader_ = readerFactory_->createReader(streamDecoders_);
  ++numReaderTreesCreated_;
}

folly::coro::Task<void> StreamReader::co_decodeRows(
    std::span<const RowRange> sourceRanges,
    std::span<const velox::BaseVector::CopyRange> outputRanges,
    velox::VectorPtr& output) {
  co_await reader_->co_read(sourceRanges, outputRanges, output);
}

} // namespace facebook::nimble
