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

#include "velox/dwio/nimble/velox/selective/NimbleData.h"

#include "velox/dwio/common/BufferUtil.h"
#include "velox/dwio/nimble/velox/selective/ChunkedDecoder.h"

namespace facebook::nimble {

using namespace facebook::velox;

void NimbleData::loadLazyInputStreams() {
  if (lazyColumnIo_) {
    streams_->loadLazyInput();
  }
}

NimbleData::NimbleData(
    const std::shared_ptr<const Type>& nimbleType,
    StripeStreams& streams,
    memory::MemoryPool& memoryPool,
    ChunkedDecoder* inMapDecoder,
    const EncodingFactory& encodingFactory,
    bool stringDecoderZeroCopy,
    bool nimblePreserveDictionaryEncoding,
    bool lazyColumnIo,
    velox::dwio::common::DecodingStats* decodingStats)
    : nimbleType_(nimbleType),
      streams_(&streams),
      pool_(&memoryPool),
      inMapDecoder_(inMapDecoder),
      encodingFactory_(&encodingFactory),
      lazyColumnIo_(lazyColumnIo),
      decodingStats_(decodingStats) {
  switch (nimbleType->kind()) {
    case Kind::Scalar:
      // Nulls in scalar types will be decoded along with values.
      break;
    case Kind::TimestampMicroNano:
      // Nulls are decoded along with micros values (nullable stream).
      break;
    case Kind::Row: {
      auto& rowType = nimbleType->asRow();
      nullsDecoder_ = makeDecoder(
          rowType.nullsDescriptor(), /*decodeValuesWithNulls=*/false);
      break;
    }
    case Kind::Array: {
      auto& arrayType = nimbleType->asArray();
      nullsDecoder_ = makeDecoder(
          arrayType.lengthsDescriptor(), /*decodeValuesWithNulls=*/true);
      break;
    }
    case Kind::Map: {
      auto& mapType = nimbleType->asMap();
      nullsDecoder_ = makeDecoder(
          mapType.lengthsDescriptor(), /*decodeValuesWithNulls=*/true);
      break;
    }
    case Kind::ArrayWithOffsets: {
      auto& arrayWithOffsetsType = nimbleType->asArrayWithOffsets();
      nullsDecoder_ = makeDecoder(
          arrayWithOffsetsType.offsetsDescriptor(),
          /*decodeValuesWithNulls=*/true);
      break;
    }
    case Kind::SlidingWindowMap: {
      auto& slidingWindowMapType = nimbleType->asSlidingWindowMap();
      nullsDecoder_ = makeDecoder(
          slidingWindowMapType.offsetsDescriptor(),
          /*decodeValuesWithNulls=*/true);
      break;
    }
    case Kind::FlatMap: {
      auto& flatMapType = nimbleType->asFlatMap();
      nullsDecoder_ = makeDecoder(
          flatMapType.nullsDescriptor(), /*decodeValuesWithNulls=*/false);
      break;
    }
    default:
      NIMBLE_UNSUPPORTED("{}", toString(nimbleType->kind()));
  }
  stringDecoderZeroCopy_ = stringDecoderZeroCopy;
  nimblePreserveDictionaryEncoding_ = nimblePreserveDictionaryEncoding;
}

void NimbleData::readNulls(
    vector_size_t numValues,
    const uint64_t* incomingNulls,
    BufferPtr& nulls,
    bool /*nullsOnly*/) {
  if (!nullsDecoder_ && !inMapDecoder_ && !incomingNulls) {
    nulls.reset();
    return;
  }
  auto numBytes = velox::bits::nbytes(numValues);
  if (!nulls || nulls->capacity() < numBytes) {
    nulls = AlignedBuffer::allocate<char>(numBytes, pool_);
  }
  nulls->setSize(numBytes);
  auto* nullsPtr = nulls->asMutable<uint64_t>();
  if (inMapDecoder_) {
    if (nullsDecoder_) {
      dwio::common::ensureCapacity<char>(inMap_, numBytes, pool_);
      inMapDecoder_->nextBools(
          inMap_->asMutable<uint64_t>(), numValues, incomingNulls);
      nullsDecoder_->nextBools(nullsPtr, numValues, inMap_->as<uint64_t>());
    } else {
      inMapDecoder_->nextBools(nullsPtr, numValues, incomingNulls);
      if (inMap_.get() != nulls.get()) {
        inMap_ = nulls;
      }
    }
  } else if (nullsDecoder_) {
    nullsDecoder_->nextBools(nullsPtr, numValues, incomingNulls);
  } else {
    memcpy(nullsPtr, incomingNulls, numBytes);
  }
}

const velox::BufferPtr& NimbleData::getPreloadedValues() {
  NIMBLE_CHECK_NOT_NULL(nullsDecoder_, "A valid nullable decoder is required");
  return nullsDecoder_->getPreloadedValues();
}

uint64_t NimbleData::skipNulls(uint64_t numValues, bool /*nullsOnly*/) {
  if (!nullsDecoder_ && !inMapDecoder_) {
    return numValues;
  }

  // Materialization path.
  constexpr uint64_t kBufferWords = 256;
  constexpr auto kBitCount = 64 * kBufferWords;
  const auto countNulls = [](ChunkedDecoder& decoder, size_t size) {
    uint64_t buffer[kBufferWords];
    decoder.nextBools(buffer, size, nullptr);
    return velox::bits::countNulls(buffer, 0, size);
  };
  auto remaining = numValues;
  while (remaining > 0) {
    const auto chunkSize = std::min(remaining, kBitCount);
    uint64_t nullCount{0};
    if (inMapDecoder_) {
      nullCount = countNulls(*inMapDecoder_, chunkSize);
      if (nullsDecoder_) {
        if (auto inMapSize = chunkSize - nullCount; inMapSize > 0) {
          nullCount += countNulls(*nullsDecoder_, inMapSize);
        }
      }
    } else {
      nullCount = countNulls(*nullsDecoder_, chunkSize);
    }
    remaining -= chunkSize;
    numValues -= nullCount;
  }
  return numValues;
}

ChunkedDecoder NimbleData::makeScalarDecoder() {
  const auto streamId = nimbleType_->asScalar().scalarDescriptor().offset();
  auto input = streams_->enqueue(static_cast<int>(streamId), lazyColumnIo_);
  return ChunkedDecoder(
      std::move(input),
      streams_->streamIndex(static_cast<int>(streamId)),
      /*decodeValuesWithNulls=*/false,
      encodingFactory_,
      pool_,
      stringDecoderZeroCopy_,
      decodingStats_,
      streams_->dictionaryAlphabetLoader(streamId));
}

ChunkedDecoder NimbleData::makeMicrosDecoder() {
  VELOX_CHECK(nimbleType_->isTimestampMicroNano());
  const auto streamId =
      nimbleType_->asTimestampMicroNano().microsDescriptor().offset();
  auto input = streams_->enqueue(static_cast<int>(streamId), lazyColumnIo_);
  return ChunkedDecoder(
      std::move(input),
      streams_->streamIndex(static_cast<int>(streamId)),
      /*decodeValuesWithNulls=*/false,
      encodingFactory_,
      pool_,
      stringDecoderZeroCopy_,
      decodingStats_,
      streams_->dictionaryAlphabetLoader(streamId));
}

ChunkedDecoder NimbleData::makeNanosDecoder() {
  VELOX_CHECK(nimbleType_->isTimestampMicroNano());
  const auto streamId =
      nimbleType_->asTimestampMicroNano().nanosDescriptor().offset();
  auto input = streams_->enqueue(static_cast<int>(streamId), lazyColumnIo_);
  return ChunkedDecoder(
      std::move(input),
      streams_->streamIndex(static_cast<int>(streamId)),
      /*decodeValuesWithNulls=*/false,
      encodingFactory_,
      pool_,
      stringDecoderZeroCopy_,
      decodingStats_,
      streams_->dictionaryAlphabetLoader(streamId));
}

std::unique_ptr<ChunkedDecoder> NimbleData::makeLengthDecoder() {
  VELOX_CHECK(
      nimbleType().isArrayWithOffsets() || nimbleType().isSlidingWindowMap());
  if (nimbleType().isArrayWithOffsets()) {
    return makeDecoder(
        nimbleType().asArrayWithOffsets().lengthsDescriptor(),
        false /* decodeValuesWithNulls */);
  } else {
    return makeDecoder(
        nimbleType().asSlidingWindowMap().lengthsDescriptor(),
        false /* decodeValuesWithNulls */);
  }
}

std::unique_ptr<ChunkedDecoder> NimbleData::makeDecoder(
    const StreamDescriptor& descriptor,
    bool decodeValuesWithNulls) {
  auto input = streams_->enqueue(descriptor.offset(), lazyColumnIo_);
  if (!input) {
    return nullptr;
  }
  return std::make_unique<ChunkedDecoder>(
      std::move(input),
      streams_->streamIndex(descriptor.offset()),
      decodeValuesWithNulls,
      encodingFactory_,
      pool_,
      stringDecoderZeroCopy_,
      decodingStats_,
      streams_->dictionaryAlphabetLoader(descriptor.offset()));
}

std::unique_ptr<velox::dwio::common::FormatData> NimbleParams::toFormatData(
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& type,
    const velox::common::ScanSpec& /*scanSpec*/) {
  // Per column counters now live on the column's ColumnRuntimeStats and are
  // only populated when decoding stats collection is enabled.
  auto& columnStatistics = columnStats(type->id(), type->type()->kind());
  velox::dwio::common::DecodingStats* decodingStats =
      columnStatistics.decodingStats ? &*columnStatistics.decodingStats
                                     : nullptr;
  return std::make_unique<NimbleData>(
      nimbleType_,
      *streams_,
      pool(),
      inMapDecoder_,
      *encodingFactory_,
      stringDecoderZeroCopy_,
      nimblePreserveDictionaryEncoding_,
      lazyColumnIo_,
      decodingStats);
}

} // namespace facebook::nimble
