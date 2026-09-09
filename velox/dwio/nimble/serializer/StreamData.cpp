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

#include "velox/dwio/nimble/serializer/StreamData.h"

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

namespace facebook::nimble::serde {

StreamData::StreamData(
    std::vector<velox::BufferPtr>& stringBuffers,
    velox::memory::MemoryPool* pool)
    : pool_{pool}, stringBuffers_{&stringBuffers} {
  NIMBLE_CHECK_NOT_NULL(pool_, "Memory pool required for encoding");
}

StreamData::StreamData(
    std::string_view data,
    std::vector<velox::BufferPtr>& stringBuffers,
    velox::memory::MemoryPool* pool,
    const Options& options)
    : pool_{pool},
      useVarintRowCount_{options.streamEncodingUsesVarintRowCount},
      bufferPool_{options.bufferPool},
      stringBuffers_{&stringBuffers} {
  NIMBLE_CHECK_NOT_NULL(pool_, "Memory pool required for encoding");
  init(data);
}

void StreamData::reset(
    std::string_view data,
    bool streamEncodingUsesVarintRowCount) {
  readRows_ = 0;
  encoding_.reset();
  useVarintRowCount_ = streamEncodingUsesVarintRowCount;
  init(data);
}

void StreamData::skip(uint32_t count) {
  if (count == 0) {
    return;
  }
  NIMBLE_CHECK_NOT_NULL(
      encoding_, "StreamData::skip requires an encoded stream");
  NIMBLE_CHECK_LE(
      count, remainingRows(), "StreamData::skip past end of segment");
  encoding_->skip(count);
  readRows_ += count;
}

StreamData::DecodeResult StreamData::decodeStrings(
    uint32_t count,
    std::string_view* output) {
  return decode(output, /*offset=*/0, count, /*width=*/0);
}

void StreamData::init(std::string_view data) {
  if (data.empty()) {
    return;
  }
  prepareForDecoding(data);
}

void StreamData::prepareForDecoding(std::string_view data) {
  NIMBLE_CHECK_NULL(encoding_, "Encoding already set");
  NIMBLE_CHECK_NOT_NULL(
      stringBuffers_, "String buffer storage required for encoded stream data");

  // Use nimble EncodingFactory to decode the data.
  // The encoded data is self-describing with type information.
  // For string types, provide a stringBufferFactory that allocates separate
  // buffers using velox::AlignedBuffer for memory tracking.
  Encoding::Options options{
      .useVarintRowCount = useVarintRowCount_, .bufferPool = bufferPool_};
  encoding_ = EncodingFactory().create(
      *pool_,
      data,
      [this](uint32_t size) {
        auto& buffer = stringBuffers_->emplace_back(
            velox::AlignedBuffer::allocate<char>(size, pool_));
        return buffer->asMutable<void>();
      },
      options);
}

StreamData::DecodeResult StreamData::decode(
    void* output,
    uint32_t offset,
    uint32_t count,
    uint32_t width,
    const std::function<void*()>& getOutputNulls,
    const velox::bits::Bitmap* scatterOutputBitmap) {
  // Decoding zero rows is a no-op. Empty streams are omitted from the
  // serialized payload (see StreamDataWriter::writeData) and so are never
  // reset() with an encoding; a zero-count decode of such a stream must not
  // require one. Mirrors the count == 0 guard in materialize().
  if (count == 0) {
    return {};
  }

  NIMBLE_CHECK_NOT_NULL(encoding_, "StreamData has no encoding to decode");
  // Only decode as many values as remain in the encoding.
  const uint32_t remainingRows = this->remainingRows();
  const uint32_t readCount = std::min(count, remainingRows);
  const bool segmentExhausted = readCount == remainingRows;
  // materializeNullable() is also the scatter-output API. Use it when the
  // encoding carries nulls, or when a parent in-map stream selected sparse
  // output rows that must be filled with decoded values and absent-row nulls.
  if (encoding_->isNullable() || scatterOutputBitmap != nullptr) {
    const auto nonNulls = decodeNullable(
        output, offset, readCount, width, getOutputNulls, scatterOutputBitmap);
    return {
        .numOutputRows = readCount,
        .nonNullOutputRows = nonNulls,
        .segmentExhausted = segmentExhausted};
  }
  decodeNonNull(output, offset, readCount, width);
  return {
      .numOutputRows = readCount,
      .nonNullOutputRows = readCount,
      .segmentExhausted = segmentExhausted};
}

uint32_t StreamData::decodeNullable(
    void* output,
    uint32_t offset,
    uint32_t readCount,
    uint32_t width,
    const std::function<void*()>& getOutputNulls,
    const velox::bits::Bitmap* scatterOutputBitmap) {
  NIMBLE_CHECK_NOT_NULL(getOutputNulls, "Null buffer callback required");
  switch (width) {
    case 0: {
      const auto nonNulls = encoding_->materializeNullable(
          readCount, output, getOutputNulls, scatterOutputBitmap, offset);
      readRows_ += readCount;
      return nonNulls;
    }
    case 1: {
      const auto nonNulls = encoding_->materializeNullable(
          readCount,
          reinterpret_cast<int8_t*>(output),
          getOutputNulls,
          scatterOutputBitmap,
          offset);
      readRows_ += readCount;
      return nonNulls;
    }
    case 2: {
      const auto nonNulls = encoding_->materializeNullable(
          readCount,
          reinterpret_cast<int16_t*>(output),
          getOutputNulls,
          scatterOutputBitmap,
          offset);
      readRows_ += readCount;
      return nonNulls;
    }
    case 4: {
      const auto nonNulls = encoding_->materializeNullable(
          readCount,
          reinterpret_cast<int32_t*>(output),
          getOutputNulls,
          scatterOutputBitmap,
          offset);
      readRows_ += readCount;
      return nonNulls;
    }
    case 8: {
      const auto nonNulls = encoding_->materializeNullable(
          readCount,
          reinterpret_cast<int64_t*>(output),
          getOutputNulls,
          scatterOutputBitmap,
          offset);
      readRows_ += readCount;
      return nonNulls;
    }
    default:
      NIMBLE_FAIL("Unexpected width {} for nimble decoding", width);
  }
}

void StreamData::decodeNonNull(
    void* output,
    uint32_t offset,
    uint32_t readCount,
    uint32_t width) {
  switch (width) {
    case 0: {
      // String type: output is std::string_view*
      auto* dest = static_cast<std::string_view*>(output) + offset;
      materialize(readCount, dest);
      return;
    }
    case 1: {
      auto* dest = static_cast<char*>(output) + offset * width;
      materialize(readCount, reinterpret_cast<int8_t*>(dest));
      return;
    }
    case 2: {
      auto* dest = static_cast<char*>(output) + offset * width;
      materialize(readCount, reinterpret_cast<int16_t*>(dest));
      return;
    }
    case 4: {
      auto* dest = static_cast<char*>(output) + offset * width;
      materialize(readCount, reinterpret_cast<int32_t*>(dest));
      return;
    }
    case 8: {
      auto* dest = static_cast<char*>(output) + offset * width;
      materialize(readCount, reinterpret_cast<int64_t*>(dest));
      return;
    }
    default:
      NIMBLE_FAIL("Unexpected width {} for nimble decoding", width);
  }
}

} // namespace facebook::nimble::serde
