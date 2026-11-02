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
#include <limits>
#include <utility>

#include <folly/ScopeGuard.h>

#include "velox/common/Casts.h"
#include "velox/common/base/BitUtil.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/nimble/common/ChunkedStream.h"
#include "velox/dwio/nimble/common/SchemaUtils.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/reader/FieldReader.h"
#include "velox/dwio/nimble/serializer/EncodingViewDecoder.h"

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

// Checks that a stripe supplies one stream per projected stream.
void checkStreamCount(size_t numStreams, size_t numProjectedStreams) {
  NIMBLE_CHECK_EQ(
      numStreams,
      numProjectedStreams,
      "Stream count must match the projected schema");
}

// Packs one bit per stream into 'key', set when the stream is present. Reuses
// the capacity of 'key'.
void makePresenceKey(
    std::span<const std::string_view> streams,
    Vector<uint8_t>& key) {
  key.resize(velox::bits::nbytes(streams.size()));
  for (size_t i{0}; i < streams.size(); ++i) {
    velox::bits::setBit(key.data(), i, !streams[i].empty());
  }
}

// Returns true when 'key' is the presence key of 'streams'. Compares the
// streams with 'key' directly, so no key is built for the streams.
bool matchesPresenceKey(
    std::span<const std::string_view> streams,
    const Vector<uint8_t>& key) {
  if (key.size() != velox::bits::nbytes(streams.size())) {
    return false;
  }
  for (size_t i{0}; i < streams.size(); ++i) {
    if (velox::bits::isBitSet(key.data(), i) == streams[i].empty()) {
      return false;
    }
  }
  return true;
}

} // namespace

struct StreamReader::CachedReader {
  // Decoders for present streams, keyed by stream id.
  folly::F14FastMap<uint32_t, std::unique_ptr<Decoder>> decodersByStreamId;

  // Points to the same decoders in stream order, with null for absent streams.
  std::vector<EncodingViewDecoder*> streamDecoders;

  // Root of the FieldReader tree built over decodersByStreamId.
  std::unique_ptr<FieldReader> root;

  // Releases the views of every decoder. The decoders and the tree stay, since
  // only the views reference a stripe's bytes, so that the next stripe can
  // rebind the decoders and, with the same presence pattern, keep the tree.
  void reset() {
    for (auto* decoder : streamDecoders) {
      if (decoder != nullptr) {
        decoder->release();
      }
    }
  }
};

StreamReader::StreamReader(
    std::shared_ptr<const Type> type,
    velox::memory::MemoryPool* pool,
    Encoding::Options options)
    : type_{std::move(type)},
      outputType_{convertToVeloxType(*velox::checkedNotNull(type_.get()))},
      pool_{velox::checkedNotNull(pool)},
      options_{std::move(options)},
      cachedKey_{pool_} {
  NIMBLE_CHECK_NOT_NULL(outputType_);

  init();
}

void StreamReader::init() {
  const auto typeWithId =
      std::shared_ptr<const velox::dwio::common::TypeWithId>{
          velox::dwio::common::TypeWithId::create(outputType_)};
  auto readerParams = FieldReaderParams{};
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

folly::coro::Task<void> StreamReader::co_read(
    std::span<const std::string_view> streams,
    std::span<const RowRange> ranges,
    velox::vector_size_t outputOffset,
    velox::VectorPtr& output) {
  checkStreamCount(streams.size(), inputOffsets_.size());
  const auto outputRanges =
      makeOutputRanges(outputType_, ranges, outputOffset, output);
  prepareRead(streams);

  // The streams are only valid during this call.
  SCOPE_EXIT {
    reset();
  };
  co_await co_decodeRows(ranges, outputRanges, output);
}

folly::coro::Task<void> StreamReader::co_read(
    std::span<const RowRange> ranges,
    velox::vector_size_t outputOffset,
    velox::VectorPtr& output) {
  const auto outputRanges =
      makeOutputRanges(outputType_, ranges, outputOffset, output);
  co_await co_decodeRows(ranges, outputRanges, output);
}

void StreamReader::setStreams(std::span<const std::string_view> streams) {
  checkStreamCount(streams.size(), inputOffsets_.size());
  prepareRead(streams);
}

StreamReader::CachedReader& StreamReader::reader() const {
  NIMBLE_CHECK(hasBoundStreams(), "No streams are bound");
  return *cachedReader_;
}

std::shared_ptr<const EncodingView> StreamReader::makeEncodingView(
    std::string_view encoding) const {
  return createEncodingView(encoding, pool_, options_);
}

folly::coro::Task<void> StreamReader::co_next(
    uint32_t numRows,
    velox::VectorPtr& output) {
  co_await reader().root->co_next(numRows, output);
}

folly::coro::Task<void> StreamReader::co_skip(uint32_t numRows) {
  co_await reader().root->co_skip(numRows);
}

void StreamReader::reset() {
  // Release the views before the chunk buffers they reference. This covers a
  // reader that a failed bind left partly rebound.
  if (cachedReader_ != nullptr) {
    cachedReader_->reset();
  }
  streamsBound_ = false;
  chunkedStreams_.clear();
}

std::string_view StreamReader::loadChunk(std::string_view stream) {
  auto chunkedStream = std::make_unique<InMemoryChunkedStream>(
      *pool_, std::make_unique<StringViewStreamLoader>(stream));
  NIMBLE_CHECK(
      chunkedStream->hasNext(), "StreamReader requires a non-empty stream");
  const auto encoded = chunkedStream->nextChunk();

  // TODO: Support multiple encoding chunks per stream.
  NIMBLE_CHECK(
      !chunkedStream->hasNext(),
      "StreamReader requires a single encoding chunk per stream");
  chunkedStreams_.push_back(std::move(chunkedStream));
  return encoded;
}

void StreamReader::rebuildReader(std::span<const std::string_view> streams) {
  const EncodingViewDecoder::EncodingViewFactory encodingViewFactory =
      [this](std::string_view encoding) { return makeEncodingView(encoding); };
  auto& cachedReader = *cachedReader_;
  cachedReader.streamDecoders.assign(streams.size(), nullptr);
  cachedReader.decodersByStreamId.reserve(streams.size());
  for (size_t i{0}; i < streams.size(); ++i) {
    const auto streamId = inputOffsets_[i];
    if (streams[i].empty()) {
      cachedReader.decodersByStreamId.erase(streamId);
      continue;
    }

    const auto it = cachedReader.decodersByStreamId.find(streamId);
    if (it != cachedReader.decodersByStreamId.end()) {
      auto* decoder = static_cast<EncodingViewDecoder*>(it->second.get());
      decoder->rebind(loadChunk(streams[i]));
      cachedReader.streamDecoders[i] = decoder;
      continue;
    }

    // TODO: Support encodings without an EncodingView.
    auto decoder = std::make_unique<EncodingViewDecoder>(
        loadChunk(streams[i]), pool_, options_.bufferPool, encodingViewFactory);
    cachedReader.streamDecoders[i] = decoder.get();
    cachedReader.decodersByStreamId.emplace(streamId, std::move(decoder));
  }
  cachedReader.root =
      readerFactory_->createReader(cachedReader.decodersByStreamId);
}

void StreamReader::prepareRead(std::span<const std::string_view> streams) {
  reset();
  chunkedStreams_.reserve(streams.size());
  if (cachedReader_ != nullptr && cachedReader_->root != nullptr &&
      matchesPresenceKey(streams, cachedKey_)) {
    // The reader points to its decoders, so rebinding replaces their views and
    // keeps the reader.
    for (size_t i{0}; i < streams.size(); ++i) {
      if (!streams[i].empty()) {
        cachedReader_->streamDecoders[i]->rebind(loadChunk(streams[i]));
      }
    }
    cachedReader_->root->reset();
  } else {
    if (cachedReader_ == nullptr) {
      cachedReader_ = std::make_unique<CachedReader>();
    }
    // Drop the tree first, so that a failure below leaves no usable reader
    // paired with another pattern's key.
    cachedReader_->root.reset();
    makePresenceKey(streams, cachedKey_);
    rebuildReader(streams);
  }
  streamsBound_ = true;
}

folly::coro::Task<void> StreamReader::co_decodeRows(
    std::span<const RowRange> sourceRanges,
    std::span<const velox::BaseVector::CopyRange> outputRanges,
    velox::VectorPtr& output) {
  co_await reader().root->co_read(sourceRanges, outputRanges, output);
}

} // namespace facebook::nimble
