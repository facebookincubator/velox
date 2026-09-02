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

#include "velox/dwio/nimble/serializer/StreamSlicer.h"

#include <algorithm>
#include <cstdint>
#include <memory>

#include "folly/ScopeGuard.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/serializer/ChunkedStreamPayload.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"
#include "velox/dwio/nimble/serializer/StreamDataParser.h"
#include "velox/dwio/nimble/serializer/StreamDataWriter.h"

namespace facebook::nimble::serde {
namespace {

SerializationVersion getInputVersion(std::string_view input) {
  NIMBLE_CHECK(!input.empty(), "Input cannot be empty");
  const auto version =
      static_cast<SerializationVersion>(static_cast<uint8_t>(input.front()));
  // This overload labels its output kProjection, which always carries varint
  // stream row counts, while tablet bodies use fixed u32. kTablet does not
  // describe the output either, because slicing strips the per-stream chunk
  // headers that kTablet mandates. Neither version states what a tablet-sourced
  // slice actually is, so reject the input rather than mislabel it.
  // TODO: Accept kTablet again and emit kTablet with the chunk-header flag
  // cleared once that flag exists.
  NIMBLE_CHECK(
      version == SerializationVersion::kSerialization ||
          version == SerializationVersion::kProjection,
      "Unsupported StreamSlicer input version: {}",
      version);
  return version;
}

std::unique_ptr<Encoding> createEncoding(
    std::string_view encoded,
    velox::memory::MemoryPool* pool,
    const Encoding::Options& options) {
  return EncodingFactory{options}.create(
      *pool,
      encoded,
      // These helpers only compute metadata-derived counts/ranges. Reaching
      // the string allocation callback means a string value stream was routed
      // through a path that should not materialize values.
      [](uint32_t /* size */) -> void* {
        NIMBLE_FAIL(
            "Unexpected string buffer allocation while slicing streams");
      },
      options);
}

Encoding::Options streamEncodingOptions(
    bool useVarintRowCount,
    velox::BufferPool* bufferPool,
    EncodingBufferPool* encodingBufferPool) {
  Encoding::Options options;
  options.useVarintRowCount = useVarintRowCount;
  options.bufferPool = bufferPool;
  options.encodingBufferPool = encodingBufferPool;
  return options;
}

Encoding::Options streamEncodingOptions(
    SerializationVersion version,
    bool streamsUseVarintRowCount,
    velox::BufferPool* bufferPool,
    EncodingBufferPool* encodingBufferPool) {
  NIMBLE_CHECK(
      version == SerializationVersion::kSerialization ||
          version == SerializationVersion::kProjection ||
          version == SerializationVersion::kTablet,
      "StreamSlicer raw streams must be kSerialization, kProjection, or "
      "kTablet encoded. Got: {}",
      version);
  NIMBLE_CHECK(
      isTabletVersion(version) || streamsUseVarintRowCount,
      "Non-tablet streams must use varint row counts");
  return streamEncodingOptions(
      streamsUseVarintRowCount, bufferPool, encodingBufferPool);
}

std::string_view nullableNullsStream(
    std::string_view encoded,
    const Encoding::Options& encodingOptions) {
  const char* pos = encoded.data() +
      EncodingPrefix::prefixSize(encoded, encodingOptions.useVarintRowCount);
  const auto valuesSize = encoding::readUint32(pos);
  NIMBLE_CHECK_LE(
      valuesSize,
      static_cast<size_t>(encoded.data() + encoded.size() - pos),
      "Nullable values child exceeds encoding size");
  pos += valuesSize;
  return {pos, encoded.data() + encoded.size()};
}

uint32_t maxStreamOffset(const StreamDescriptor& descriptor) {
  return descriptor.offset();
}

uint32_t maxStreamOffset(const Type& type) {
  switch (type.kind()) {
    case Kind::Scalar:
      return maxStreamOffset(type.asScalar().scalarDescriptor());
    case Kind::TimestampMicroNano:
      return std::max(
          maxStreamOffset(type.asTimestampMicroNano().microsDescriptor()),
          maxStreamOffset(type.asTimestampMicroNano().nanosDescriptor()));
    case Kind::Array:
      return std::max(
          maxStreamOffset(type.asArray().lengthsDescriptor()),
          maxStreamOffset(*type.asArray().elements()));
    case Kind::Map:
      return std::max(
          {maxStreamOffset(type.asMap().lengthsDescriptor()),
           maxStreamOffset(*type.asMap().keys()),
           maxStreamOffset(*type.asMap().values())});
    case Kind::Row: {
      const auto& row = type.asRow();
      uint32_t offset = maxStreamOffset(row.nullsDescriptor());
      for (size_t i = 0; i < row.childrenCount(); ++i) {
        offset = std::max(offset, maxStreamOffset(*row.childAt(i)));
      }
      return offset;
    }
    case Kind::FlatMap: {
      const auto& flatMap = type.asFlatMap();
      uint32_t offset = maxStreamOffset(flatMap.nullsDescriptor());
      for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
        offset = std::max(
            {offset,
             maxStreamOffset(flatMap.inMapDescriptorAt(i)),
             maxStreamOffset(*flatMap.childAt(i))});
      }
      return offset;
    }
    case Kind::ArrayWithOffsets:
      return std::max(
          {maxStreamOffset(type.asArrayWithOffsets().offsetsDescriptor()),
           maxStreamOffset(type.asArrayWithOffsets().lengthsDescriptor()),
           maxStreamOffset(*type.asArrayWithOffsets().elements())});
    case Kind::SlidingWindowMap:
      return std::max(
          {maxStreamOffset(type.asSlidingWindowMap().offsetsDescriptor()),
           maxStreamOffset(type.asSlidingWindowMap().lengthsDescriptor()),
           maxStreamOffset(*type.asSlidingWindowMap().keys()),
           maxStreamOffset(*type.asSlidingWindowMap().values())});
  }
  NIMBLE_UNREACHABLE("Unsupported type kind: {}.", toString(type.kind()));
}

uint32_t streamCount(const std::shared_ptr<const Type>& schema) {
  NIMBLE_CHECK_NOT_NULL(schema, "Schema cannot be null");
  return maxStreamOffset(*schema) + 1;
}

using BufferChunks = std::vector<velox::BufferPtr>;

void freeIOBufOwner(void* /* buf */, void* userData) {
  delete static_cast<std::shared_ptr<const void>*>(userData);
}

// transferBuffers() transfers allocation ownership only, so capacity is the
// valid ownership range for stream views.
bool isBackedByChunks(std::string_view stream, const BufferChunks& chunks) {
  const auto streamBegin = reinterpret_cast<uintptr_t>(stream.data());
  const auto streamEnd = streamBegin + stream.size();
  for (const auto& chunk : chunks) {
    const auto chunkBegin = reinterpret_cast<uintptr_t>(chunk->as<char>());
    const auto chunkEnd = chunkBegin + chunk->capacity();
    if (chunkBegin <= streamBegin && streamEnd <= chunkEnd) {
      return true;
    }
  }
  return false;
}

void checkStreamsBackedByChunks(
    const std::vector<std::string_view>& streams,
    const BufferChunks& chunks) {
  for (const auto stream : streams) {
    if (stream.empty()) {
      continue;
    }
    NIMBLE_CHECK(
        isBackedByChunks(stream, chunks),
        "Sliced stream must be backed by transferred buffer chunks");
  }
}

folly::IOBuf takeOwnershipAsIOBuf(
    const std::vector<std::string_view>& streams,
    std::shared_ptr<const void> owner) {
  NIMBLE_CHECK_NOT_NULL(owner, "Stream owner must be initialized");

  std::unique_ptr<folly::IOBuf> chain;
  const char* runData{nullptr};
  size_t runLength{0};
  const auto flushRun = [&]() {
    if (runData == nullptr) {
      return;
    }
    auto node = folly::IOBuf::takeOwnership(
        const_cast<char*>(runData),
        runLength,
        runLength,
        freeIOBufOwner,
        new std::shared_ptr<const void>(owner));
    if (chain == nullptr) {
      chain = std::move(node);
    } else {
      chain->appendToChain(std::move(node));
    }
    runData = nullptr;
    runLength = 0;
  };

  for (const auto stream : streams) {
    if (stream.empty()) {
      continue;
    }
    if (runData != nullptr && runData + runLength == stream.data()) {
      runLength += stream.size();
      continue;
    }
    flushRun();
    runData = stream.data();
    runLength = stream.size();
  }
  flushRun();

  return chain == nullptr ? folly::IOBuf{} : std::move(*chain);
}

size_t totalBytes(const std::vector<std::string_view>& streams) {
  size_t bytes{0};
  for (const auto stream : streams) {
    bytes += stream.size();
  }
  return bytes;
}

} // namespace

StreamSlicer::StreamSlicer(
    std::shared_ptr<const Type> schema,
    velox::memory::MemoryPool* pool,
    Options options)
    : schema_{std::move(schema)},
      pool_{pool},
      options_{std::move(options)},
      streamCount_{streamCount(schema_)},
      bufferPool_{velox::BufferPool::kDefaultCapacity},
      encodingBufferPool_{pool_},
      strippedStreamBufferPool_{pool_, /*maxCachedBuffers=*/1},
      headerBuffer_{pool_},
      trailerBuffer_{pool_} {
  NIMBLE_CHECK_NOT_NULL(pool_, "Memory pool cannot be null");
}

folly::IOBuf StreamSlicer::takeOwnershipAsIOBuf(
    const std::vector<std::string_view>& streams,
    Buffer& buffer) {
  auto chunks = std::make_shared<BufferChunks>(buffer.transferBuffers());
  checkStreamsBackedByChunks(streams, *chunks);
  return takeOwnershipAsIOBuf(
      streams, std::shared_ptr<const void>{std::move(chunks)});
}

folly::IOBuf StreamSlicer::takeOwnershipAsIOBuf(
    const std::vector<std::string_view>& streams,
    std::shared_ptr<const void> owner) {
  return ::facebook::nimble::serde::takeOwnershipAsIOBuf(
      streams, std::move(owner));
}

folly::IOBuf StreamSlicer::slice(
    std::string_view input,
    uint32_t offset,
    uint32_t length) const {
  const auto inputVersion = getInputVersion(input);
  DeserializerOptions parserOptions{.hasHeader = true};
  auto parser = StreamDataParser{pool_, parserOptions};
  const auto rowCount = parser.initialize(input);
  NIMBLE_CHECK_EQ(parser.version(), inputVersion, "Unexpected input version");
  NIMBLE_CHECK_LE(offset, rowCount, "Slice offset exceeds row count");
  NIMBLE_CHECK_LE(length, rowCount - offset, "Slice length exceeds row count");
  NIMBLE_CHECK_GT(length, 0, "Slice length must be positive");
  if (offset == 0 && length == rowCount) {
    return *folly::IOBuf::copyBuffer(input.data(), input.size());
  }

  inputStreams_.clear();
  inputStreams_.reserve(streamCount_);
  parser.iterateStreams([&](uint32_t streamId, std::string_view streamData) {
    if (streamId >= inputStreams_.size()) {
      inputStreams_.resize(streamId + 1);
    }
    inputStreams_[streamId] = streamData;
  });
  auto slicedStreams = sliceStreams(
      inputStreams_,
      {.offset = offset, .length = length},
      /*outputBuffer=*/nullptr,
      streamEncodingOptions(
          parser.streamEncodingUsesVarintRowCount(),
          &bufferPool_,
          &encodingBufferPool_));

  streamSizes_.assign(slicedStreams.streams.size(), 0);
  for (uint32_t i = 0; i < slicedStreams.streams.size(); ++i) {
    streamSizes_[i] = static_cast<uint32_t>(slicedStreams.streams[i].size());
  }

  headerBuffer_.resize(0);
  const auto flagsOffset = writeSerializationHeader(
      headerBuffer_, SerializationVersion::kProjection, length);
  headerBuffer_[flagsOffset] = static_cast<char>(detail::makeFlagsByte(
      slicedStreams.requiresNullBarrier,
      parser.streamEncodingUsesVarintRowCount(),
      /*streamHasChunkHeader=*/false));

  trailerBuffer_.resize(0);
  detail::writeTrailer(
      streamSizes_,
      options_.streamIndicesEncodingType,
      options_.streamSizesEncodingType,
      trailerBuffer_);

  auto output =
      folly::IOBuf::copyBuffer(headerBuffer_.data(), headerBuffer_.size());
  auto body = std::make_unique<folly::IOBuf>(std::move(slicedStreams.data));
  output->appendToChain(std::move(body));
  output->appendToChain(
      folly::IOBuf::copyBuffer(trailerBuffer_.data(), trailerBuffer_.size()));
  return std::move(*output);
}

StreamSlicer::SlicedStreams StreamSlicer::slice(
    const std::vector<std::string_view>& inputStreams,
    uint32_t offset,
    uint32_t length,
    Buffer* outputBuffer) const {
  NIMBLE_CHECK_GT(length, 0, "Slice length must be positive");
  NIMBLE_CHECK(
      !options_.streamHasChunkHeader || isTabletVersion(options_.streamVersion),
      "Chunk headers are only supported for kTablet streams");
  const auto encodingOptions = streamEncodingOptions(
      options_.streamVersion,
      options_.streamsUseVarintRowCount,
      &bufferPool_,
      &encodingBufferPool_);
  const Range range{.offset = offset, .length = length};

  if (!options_.streamHasChunkHeader) {
    return sliceStreams(inputStreams, range, outputBuffer, encodingOptions);
  }

  ScopedEncodingBuffer strippedStreamBuffer{pool_, &strippedStreamBufferPool_};
  SCOPE_EXIT {
    strippedStreamCache_.clear();
  };
  stripChunkHeaders(
      inputStreams,
      strippedStreamBuffer.get(),
      inputStreams_,
      strippedStreamCache_);
  return sliceStreams(inputStreams_, range, outputBuffer, encodingOptions);
}

StreamSlicer::SlicedStreams StreamSlicer::sliceStreams(
    const std::vector<std::string_view>& inputStreams,
    Range range,
    Buffer* outputBuffer,
    const Encoding::Options& encodingOptions) const {
  SlicedStreams result;
  if (outputBuffer != nullptr) {
    sliceType(
        *schema_, range, inputStreams, result, *outputBuffer, encodingOptions);
    return result;
  }

  Buffer ownedOutputBuffer{*pool_, totalBytes(inputStreams)};
  sliceType(
      *schema_,
      range,
      inputStreams,
      result,
      ownedOutputBuffer,
      encodingOptions);
  result.data = takeOwnershipAsIOBuf(result.streams, ownedOutputBuffer);
  return result;
}

void StreamSlicer::sliceType(
    const Type& type,
    Range range,
    const std::vector<std::string_view>& inputStreams,
    SlicedStreams& outputStreams,
    Buffer& outputBuffer,
    const Encoding::Options& encodingOptions) const {
  if (range.length == 0) {
    return;
  }

  switch (type.kind()) {
    case Kind::Scalar:
      sliceDescriptor(
          type.asScalar().scalarDescriptor(),
          range,
          inputStreams,
          /*isRowOrFlatMapNullStream=*/false,
          outputStreams,
          outputBuffer,
          encodingOptions);
      return;
    case Kind::TimestampMicroNano: {
      const auto& timestamp = type.asTimestampMicroNano();
      sliceDescriptor(
          timestamp.microsDescriptor(),
          range,
          inputStreams,
          /*isRowOrFlatMapNullStream=*/false,
          outputStreams,
          outputBuffer,
          encodingOptions);
      auto nanosRange = range;
      if (hasStream(inputStreams, timestamp.microsDescriptor())) {
        nanosRange = nonNullRange(
            inputStreams[timestamp.microsDescriptor().offset()],
            range,
            outputBuffer,
            encodingOptions);
      }
      sliceDescriptor(
          timestamp.nanosDescriptor(),
          nanosRange,
          inputStreams,
          /*isRowOrFlatMapNullStream=*/false,
          outputStreams,
          outputBuffer,
          encodingOptions);
      return;
    }
    case Kind::Row: {
      const auto& row = type.asRow();
      auto childRange = range;
      if (hasStream(inputStreams, row.nullsDescriptor())) {
        sliceDescriptor(
            row.nullsDescriptor(),
            range,
            inputStreams,
            /*isRowOrFlatMapNullStream=*/true,
            outputStreams,
            outputBuffer,
            encodingOptions);
        childRange = trueRange(
            inputStreams[row.nullsDescriptor().offset()],
            range,
            outputBuffer,
            encodingOptions);
      }
      for (size_t i = 0; i < row.childrenCount(); ++i) {
        sliceType(
            *row.childAt(i),
            childRange,
            inputStreams,
            outputStreams,
            outputBuffer,
            encodingOptions);
      }
      return;
    }
    case Kind::Array: {
      const auto& array = type.asArray();
      // Missing lengths means the projected stream set does not contain this
      // container, so there is no child range to derive.
      if (!hasStream(inputStreams, array.lengthsDescriptor())) {
        return;
      }
      const auto childRange = offsetsRange(
          inputStreams[array.lengthsDescriptor().offset()],
          range,
          encodingOptions);
      sliceDescriptor(
          array.lengthsDescriptor(),
          range,
          inputStreams,
          /*isRowOrFlatMapNullStream=*/false,
          outputStreams,
          outputBuffer,
          encodingOptions);
      sliceType(
          *array.elements(),
          childRange,
          inputStreams,
          outputStreams,
          outputBuffer,
          encodingOptions);
      return;
    }
    case Kind::Map: {
      const auto& map = type.asMap();
      // Missing lengths means the projected stream set does not contain this
      // container, so there is no child range to derive.
      if (!hasStream(inputStreams, map.lengthsDescriptor())) {
        return;
      }
      const auto childRange = offsetsRange(
          inputStreams[map.lengthsDescriptor().offset()],
          range,
          encodingOptions);
      sliceDescriptor(
          map.lengthsDescriptor(),
          range,
          inputStreams,
          /*isRowOrFlatMapNullStream=*/false,
          outputStreams,
          outputBuffer,
          encodingOptions);
      sliceType(
          *map.keys(),
          childRange,
          inputStreams,
          outputStreams,
          outputBuffer,
          encodingOptions);
      sliceType(
          *map.values(),
          childRange,
          inputStreams,
          outputStreams,
          outputBuffer,
          encodingOptions);
      return;
    }
    case Kind::FlatMap: {
      const auto& flatMap = type.asFlatMap();
      auto mapRange = range;
      if (hasStream(inputStreams, flatMap.nullsDescriptor())) {
        sliceDescriptor(
            flatMap.nullsDescriptor(),
            range,
            inputStreams,
            /*isRowOrFlatMapNullStream=*/true,
            outputStreams,
            outputBuffer,
            encodingOptions);
        mapRange = trueRange(
            inputStreams[flatMap.nullsDescriptor().offset()],
            range,
            outputBuffer,
            encodingOptions);
      }
      for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
        const auto& inMapDescriptor = flatMap.inMapDescriptorAt(i);
        const auto childHasInMap = hasStream(inputStreams, inMapDescriptor);
        const auto childHasValues =
            hasFlatMapValues(*flatMap.childAt(i), inputStreams);
        if (!childHasInMap && !childHasValues) {
          continue;
        }
        auto valueRange = mapRange;
        if (childHasInMap) {
          sliceDescriptor(
              inMapDescriptor,
              mapRange,
              inputStreams,
              /*isRowOrFlatMapNullStream=*/false,
              outputStreams,
              outputBuffer,
              encodingOptions);
          valueRange = trueRange(
              inputStreams[inMapDescriptor.offset()],
              mapRange,
              outputBuffer,
              encodingOptions);
        }
        sliceType(
            *flatMap.childAt(i),
            valueRange,
            inputStreams,
            outputStreams,
            outputBuffer,
            encodingOptions);
      }
      return;
    }
    default:
      NIMBLE_UNSUPPORTED(
          "StreamSlicer does not support slicing {} yet", type.kind());
  }
}

void StreamSlicer::sliceDescriptor(
    const StreamDescriptor& descriptor,
    Range range,
    const std::vector<std::string_view>& inputStreams,
    bool isRowOrFlatMapNullStream,
    SlicedStreams& outputStreams,
    Buffer& outputBuffer,
    const Encoding::Options& encodingOptions) const {
  NIMBLE_CHECK_GT(range.length, 0, "Stream slice length must be positive");
  if (!hasStream(inputStreams, descriptor)) {
    return;
  }
  if (descriptor.offset() >= outputStreams.streams.size()) {
    outputStreams.streams.resize(descriptor.offset() + 1);
  }
  auto sliced = EncodingFactory::slice(
      inputStreams[descriptor.offset()],
      range.offset,
      range.length,
      outputBuffer,
      encodingOptions);
  outputStreams.streams[descriptor.offset()] = sliced;
  if (isRowOrFlatMapNullStream) {
    NIMBLE_CHECK(!sliced.empty(), "Sliced null stream must not be empty");
    outputStreams.requiresNullBarrier |=
        countTrue(
            sliced,
            {.offset = 0, .length = range.length},
            outputBuffer,
            encodingOptions) < range.length;
  }
}

bool StreamSlicer::hasStream(
    const std::vector<std::string_view>& inputStreams,
    const StreamDescriptor& descriptor) const {
  return descriptor.offset() < inputStreams.size() &&
      !inputStreams[descriptor.offset()].empty();
}

bool StreamSlicer::hasFlatMapValues(
    const Type& type,
    const std::vector<std::string_view>& inputStreams) const {
  return visitPresenceStreamOffsets(type, [&](offset_size offset) {
    return offset < inputStreams.size() && !inputStreams[offset].empty();
  });
}

void StreamSlicer::stripChunkHeaders(
    const std::vector<std::string_view>& inputStreams,
    Buffer& strippedStreamBuffer,
    std::vector<std::string_view>& strippedStreams,
    StrippedStreamCache& strippedStreamCache) {
  strippedStreams.resize(inputStreams.size());
  NIMBLE_CHECK(
      strippedStreamCache.empty(),
      "Stripped stream cache must be empty before slicing.");
  strippedStreamCache.reserve(inputStreams.size());
  for (size_t i = 0; i < inputStreams.size(); ++i) {
    const auto input = inputStreams[i];
    if (input.empty()) {
      strippedStreams[i] = {};
      continue;
    }
    // Multiple projected stream slots can alias the same tablet stream bytes.
    // Strip once so duplicate slots share the same decoded view.
    auto cached = strippedStreamCache.find(input);
    if (cached != strippedStreamCache.end()) {
      strippedStreams[i] = cached->second;
      continue;
    }
    auto stripped =
        facebook::nimble::serde::stripChunkHeaders(input, strippedStreamBuffer);
    strippedStreamCache.emplace(input, stripped);
    strippedStreams[i] = stripped;
  }
}

StreamSlicer::Range StreamSlicer::nonNullRange(
    std::string_view encoded,
    Range range,
    Buffer& outputBuffer,
    const Encoding::Options& encodingOptions) const {
  return {
      .offset = countNonNull(
          encoded,
          {.offset = 0, .length = range.offset},
          outputBuffer,
          encodingOptions),
      .length = countNonNull(encoded, range, outputBuffer, encodingOptions),
  };
}

StreamSlicer::Range StreamSlicer::trueRange(
    std::string_view encoded,
    Range range,
    Buffer& outputBuffer,
    const Encoding::Options& encodingOptions) const {
  if (range.length == 0) {
    return {};
  }
  NIMBLE_CHECK_EQ(
      EncodingPrefix::dataType(encoded),
      DataType::Bool,
      "Expected a bool stream");
  const auto rowCount =
      EncodingPrefix::readRowCount(encoded, encodingOptions.useVarintRowCount);
  NIMBLE_CHECK_LE(range.offset, rowCount);
  NIMBLE_CHECK_LE(range.length, rowCount - range.offset);
  const auto encodingType = EncodingPrefix::encodingType(encoded);
  switch (encodingType) {
    case EncodingType::Constant: {
      const char* pos = encoded.data() +
          EncodingPrefix::prefixSize(
                            encoded, encodingOptions.useVarintRowCount);
      const bool value = encoding::read<bool>(pos);
      return {
          .offset = value ? range.offset : 0,
          .length = value ? range.length : 0,
      };
    }
    case EncodingType::RLE: {
      RLEEncoding<bool>::RangeCounts counts;
      RLEEncoding<bool>::countTrue(
          encoded,
          range.offset,
          range.length,
          outputBuffer,
          counts,
          encodingOptions);
      return {
          .offset = counts.numTrueBeforeRange,
          .length = counts.numTrueInRange,
      };
    }
    case EncodingType::SparseBool: {
      SparseBoolEncoding::RangeCounts counts;
      SparseBoolEncoding::countTrue(
          encoded, range.offset, range.length, pool_, counts, encodingOptions);
      return {
          .offset = counts.numTrueBeforeRange,
          .length = counts.numTrueInRange,
      };
    }
    default:
      break;
  }
  return {
      .offset = countTrue(
          encoded,
          {.offset = 0, .length = range.offset},
          outputBuffer,
          encodingOptions),
      .length = countTrue(encoded, range, outputBuffer, encodingOptions),
  };
}

StreamSlicer::Range StreamSlicer::offsetsRange(
    std::string_view encoded,
    Range range,
    const Encoding::Options& encodingOptions) const {
  NIMBLE_CHECK_GT(range.length, 0, "Offsets range length must be positive");
  auto encoding = createEncoding(encoded, pool_, encodingOptions);
  NIMBLE_CHECK_EQ(
      encoding->dataType(), DataType::Uint32, "Expected a uint32 stream");
  const auto count = range.offset + range.length;
  NIMBLE_CHECK_LE(count, encoding->rowCount());
  ScopedVector<uint32_t> values{count, pool_, encodingOptions.bufferPool};
  encoding->materialize(count, values.data());

  uint32_t childOffset = 0;
  for (uint32_t i = 0; i < range.offset; ++i) {
    childOffset += values[i];
  }
  uint32_t childLength = 0;
  for (uint32_t i = range.offset; i < count; ++i) {
    childLength += values[i];
  }
  return {.offset = childOffset, .length = childLength};
}

uint32_t StreamSlicer::countNonNull(
    std::string_view encoded,
    Range range,
    Buffer& outputBuffer,
    const Encoding::Options& encodingOptions) const {
  if (range.length == 0) {
    return 0;
  }
  const auto rowCount =
      EncodingPrefix::readRowCount(encoded, encodingOptions.useVarintRowCount);
  NIMBLE_CHECK_LE(range.offset, rowCount);
  NIMBLE_CHECK_LE(range.length, rowCount - range.offset);
  if (EncodingPrefix::encodingType(encoded) != EncodingType::Nullable) {
    return range.length;
  }
  return countTrue(
      nullableNullsStream(encoded, encodingOptions),
      range,
      outputBuffer,
      encodingOptions);
}

uint32_t StreamSlicer::countTrue(
    std::string_view encoded,
    Range range,
    Buffer& outputBuffer,
    const Encoding::Options& encodingOptions) const {
  if (range.length == 0) {
    return 0;
  }
  NIMBLE_CHECK_EQ(
      EncodingPrefix::dataType(encoded),
      DataType::Bool,
      "Expected a bool stream");
  const auto rowCount =
      EncodingPrefix::readRowCount(encoded, encodingOptions.useVarintRowCount);
  NIMBLE_CHECK_LE(range.offset, rowCount);
  NIMBLE_CHECK_LE(range.length, rowCount - range.offset);
  const auto encodingType = EncodingPrefix::encodingType(encoded);
  switch (encodingType) {
    case EncodingType::Constant: {
      const char* pos = encoded.data() +
          EncodingPrefix::prefixSize(
                            encoded, encodingOptions.useVarintRowCount);
      return encoding::read<bool>(pos) ? range.length : 0;
    }
    case EncodingType::RLE:
      return RLEEncoding<bool>::countTrue(
          encoded, range.offset, range.length, outputBuffer, encodingOptions);
    case EncodingType::SparseBool:
      return SparseBoolEncoding::countTrue(
          encoded, range.offset, range.length, pool_, encodingOptions);
    default:
      break;
  }

  auto encoding = createEncoding(encoded, pool_, encodingOptions);
  encoding->skip(range.offset);
  ScopedVector<uint64_t> bits{
      velox::bits::nwords(range.length), pool_, encodingOptions.bufferPool};
  encoding->materializeBoolsAsBits(range.length, bits.data(), /*begin=*/0);
  return velox::bits::countBits(bits.data(), 0, range.length);
}

} // namespace facebook::nimble::serde
