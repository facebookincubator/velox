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

#include "velox/dwio/nimble/selective/NimbleData.h"

#include "velox/dwio/common/BufferUtil.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/nimble/selective/ChunkedDecoder.h"
#include "velox/type/Filter.h"

namespace facebook::nimble {

using namespace facebook::velox;

void NimbleData::loadLazyInputStreams() {
  if (lazyColumnIo_) {
    streams_->loadLazyInput();
  }
}

namespace {

// Returns the descriptor of the stream that holds the nulls of a column of
// 'type': the nulls stream of a row or flat map, or the lengths or offsets
// stream of the other kinds. Returns nullptr for the kinds that decode their
// nulls with their values.
const StreamDescriptor* nullsDescriptor(const Type& type) {
  switch (type.kind()) {
    case Kind::Scalar:
      // Nulls in scalar types will be decoded along with values.
      return nullptr;
    case Kind::TimestampMicroNano:
      // Nulls are decoded along with micros values (nullable stream).
      return nullptr;
    case Kind::Row:
      return &type.asRow().nullsDescriptor();
    case Kind::Array:
      return &type.asArray().lengthsDescriptor();
    case Kind::Map:
      return &type.asMap().lengthsDescriptor();
    case Kind::ArrayWithOffsets:
      return &type.asArrayWithOffsets().offsetsDescriptor();
    case Kind::SlidingWindowMap:
      return &type.asSlidingWindowMap().offsetsDescriptor();
    case Kind::FlatMap:
      return &type.asFlatMap().nullsDescriptor();
    default:
      NIMBLE_UNSUPPORTED("{}", toString(type.kind()));
  }
}

} // namespace

NimbleData::NimbleData(
    const std::shared_ptr<const Type>& nimbleType,
    StripeStreams& streams,
    memory::MemoryPool& memoryPool,
    ChunkedDecoder* inMapDecoder,
    const EncodingFactory& encodingFactory,
    bool preserveDictionaryEncoding,
    bool lazyColumnIo,
    velox::dwio::common::DecodingStats* decodingStats,
    bool dictionaryAwareReads,
    bool scanFastPath)
    : nimbleType_(nimbleType),
      nullsDescriptor_(nullsDescriptor(*nimbleType)),
      streams_(&streams),
      pool_(&memoryPool),
      inMapDecoder_(inMapDecoder),
      encodingFactory_(&encodingFactory),
      lazyColumnIo_(lazyColumnIo),
      preserveDictionaryEncoding_(preserveDictionaryEncoding),
      decodingStats_(decodingStats),
      dictionaryAwareReads_(dictionaryAwareReads),
      scanFastPath_(scanFastPath) {
  nullsDecoder_ = makeNullsDecoder();

  // Nimble encodings hand out StringViews into decoder-owned buffers, which the
  // reader keeps alive via setStringBuffers, so values are never copied.
  stringDecoderZeroCopy_ = true;
}

void NimbleData::setNullsOnlyDecoder(ChunkedDecoder* decoder) {
  if (scanFastPath_ && inMapDecoder_ == nullptr) {
    nullsOnlyDecoder_ = decoder;
  }
}

bool NimbleData::readsNullsOnly(const velox::common::ScanSpec& scanSpec) const {
  if (nullsOnlyDecoder_ == nullptr || scanSpec.valueHook() != nullptr ||
      !scanSpec.readsNullsOnly()) {
    return false;
  }
  return scanSpec.filter()->kind() == velox::common::FilterKind::kIsNull ||
      !scanSpec.keepValues();
}

void NimbleData::readNulls(
    vector_size_t numValues,
    const uint64_t* incomingNulls,
    BufferPtr& nulls,
    bool nullsOnly) {
  const bool readScalarNulls = nullsOnly && nullsOnlyDecoder_ != nullptr;
  if (!readScalarNulls && !nullsDecoder_ && !inMapDecoder_ && !incomingNulls) {
    nulls.reset();
    return;
  }
  auto numBytes = velox::bits::nbytes(numValues);
  if (!nulls || nulls->capacity() < numBytes) {
    nulls = AlignedBuffer::allocate<char>(numBytes, pool_);
  }
  nulls->setSize(numBytes);
  auto* nullsPtr = nulls->asMutable<uint64_t>();
  if (readScalarNulls) {
    nullsOnlyDecoder_->nextNulls(nullsPtr, numValues, incomingNulls);
  } else if (inMapDecoder_) {
    if (nullsDecoder_) {
      dwio::common::ensureCapacity<char>(inMap_, numBytes, pool_);
      inMapDecoder_->nextBools(
          inMap_->asMutable<uint64_t>(), numValues, incomingNulls);
      nullsDecoder_->nextBools(nullsPtr, numValues, inMap_->as<uint64_t>());
    } else {
      dwio::common::ensureCapacity<char>(inMap_, numBytes, pool_);
      inMapDecoder_->nextBools(
          inMap_->asMutable<uint64_t>(), numValues, incomingNulls);
      std::memcpy(nullsPtr, inMap_->as<uint64_t>(), numBytes);
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

uint64_t NimbleData::skipNulls(uint64_t numValues, bool nullsOnly) {
  if (nullsOnly && nullsOnlyDecoder_ != nullptr) {
    nullsOnlyDecoder_->skip(static_cast<int64_t>(numValues));
    return 0;
  }
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
      decodingStats_,
      streams_->dictionaryAlphabetLoader(streamId),
      dictionaryAwareReads_,
      scanFastPath_);
}

void NimbleData::rebindNullsDecoder() {
  if (nullsDecoder_ == nullptr) {
    // This column has no such stream, or the earlier stripe lacked it.
    nullsDecoder_ = makeNullsDecoder();
    return;
  }
  if (!rebindDecoder(*nullsDecoder_, nullsDescriptor_->offset())) {
    // The current stripe lacks the stream.
    nullsDecoder_.reset();
  }
}

void NimbleData::unbindNullsDecoder() {
  if (nullsDecoder_ != nullptr) {
    nullsDecoder_->unbindStream();
  }
}

bool NimbleData::rebindScalarDecoder(ChunkedDecoder& decoder) {
  return rebindDecoder(
      decoder, nimbleType_->asScalar().scalarDescriptor().offset());
}

std::unique_ptr<ChunkedDecoder> NimbleData::makeNullsDecoder() {
  if (nullsDescriptor_ == nullptr) {
    return nullptr;
  }

  // Rows and flat maps decode only their nulls. The other kinds decode their
  // lengths or offsets with their nulls.
  const auto kind = nimbleType_->kind();
  return makeDecoder(
      *nullsDescriptor_,
      /*decodeValuesWithNulls=*/kind != Kind::Row && kind != Kind::FlatMap);
}

bool NimbleData::rebindDecoder(ChunkedDecoder& decoder, uint32_t streamId) {
  auto input = streams_->enqueue(static_cast<int>(streamId), lazyColumnIo_);

  // The current stripe has no data in the stream, for example the elements
  // stream of an array column whose arrays in that stripe are all empty.
  if (input == nullptr) {
    return false;
  }
  decoder.rebindStream(
      std::move(input),
      streams_->streamIndex(static_cast<int>(streamId)),
      streams_->dictionaryAlphabetLoader(streamId));
  return true;
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
      decodingStats_,
      streams_->dictionaryAlphabetLoader(streamId),
      dictionaryAwareReads_,
      scanFastPath_);
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
      decodingStats_,
      streams_->dictionaryAlphabetLoader(streamId),
      dictionaryAwareReads_,
      scanFastPath_);
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
      decodingStats_,
      streams_->dictionaryAlphabetLoader(descriptor.offset()),
      dictionaryAwareReads_,
      scanFastPath_);
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
      nimblePreserveDictionaryEncoding_,
      lazyColumnIo_,
      decodingStats,
      dictionaryAwareReads_,
      scanFastPath_);
}

} // namespace facebook::nimble
