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
#include "velox/dwio/nimble/velox/ChunkedStreamDecoder.h"

#include "velox/buffer/Buffer.h"

namespace facebook::nimble {

namespace {
void bufferStringContent(
    Vector<char>& buffer,
    void* values,
    void* FOLLY_NULLABLE notNulls,
    uint32_t offset,
    uint32_t count) {
  auto* source = reinterpret_cast<std::string_view*>(values) + offset;
  uint64_t size = 0;
  if (notNulls) {
    for (auto i = 0; i < count; ++i) {
      if (velox::bits::isBitSet(
              reinterpret_cast<const uint8_t*>(notNulls), i + offset)) {
        size += (source + i)->size();
      }
    }
  } else {
    for (auto i = 0; i < count; ++i) {
      size += (source + i)->size();
    }
  }

  uint64_t targetOffset = 0;
  buffer.resize(size);

  if (notNulls) {
    for (auto i = 0; i < count; ++i) {
      if (velox::bits::isBitSet(
              reinterpret_cast<const uint8_t*>(notNulls), i + offset)) {
        auto* value = source + i;
        auto* target = buffer.data() + targetOffset;
        if (!value->empty()) {
          std::copy(value->cbegin(), value->cend(), target);
          targetOffset += value->size();
        }
        *value = std::string_view{target, value->size()};
      }
    }
  } else {
    for (auto i = 0; i < count; ++i) {
      auto* value = source + i;
      auto* target = buffer.data() + targetOffset;
      if (!value->empty()) {
        std::copy(value->cbegin(), value->cend(), target);
        targetOffset += value->size();
      }
      *value = std::string_view{target, value->size()};
    }
  }
}

} // namespace

uint32_t ChunkedStreamDecoder::next(
    uint32_t count,
    void* output,
    std::vector<velox::BufferPtr>& stringBuffers,
    std::function<void*()> getOutputNulls,
    const velox::bits::Bitmap* scatterOutputBitmap) {
  NIMBLE_DCHECK(stringBuffers.empty());

  if (count == 0) {
    if (getOutputNulls && scatterOutputBitmap) {
      auto nullsPtr = getOutputNulls();
      // @lint-ignore CLANGTIDY facebook-hte-BadMemset
      memset(nullsPtr, 0, velox::bits::nbytes(scatterOutputBitmap->size()));
    }
    return 0;
  }

  LoggingScope scope{logger_};

  uint32_t nonNullCount = 0;
  bool hasNulls = false;
  uint32_t offset = 0;
  void* nullsPtr = nullptr;
  std::function<void*()> initNulls = [&]() {
    if (!nullsPtr) {
      nullsPtr = getOutputNulls();
    }
    return nullsPtr;
  };

  while (count > 0) {
    ensureLoaded();

    auto rowsToRead = std::min(count, remaining_);
    uint32_t chunkNonNullCount = 0;
    uint32_t endOffset = 0;

    if (!getOutputNulls || !scatterOutputBitmap) {
      NIMBLE_CHECK_NULL(
          scatterOutputBitmap, "Unexpected scatter output bitmap");
      chunkNonNullCount = encoding_->materializeNullable(
          rowsToRead, output, initNulls, nullptr, offset);
      endOffset = offset + rowsToRead;
    } else {
      endOffset = velox::bits::findSetBit(
          static_cast<const char*>(scatterOutputBitmap->bits()),
          offset,
          scatterOutputBitmap->size(),
          rowsToRead + 1);
      velox::bits::Bitmap localBitmap{scatterOutputBitmap->bits(), endOffset};
      chunkNonNullCount = encoding_->materializeNullable(
          rowsToRead, output, initNulls, &localBitmap, offset);
    }

    auto chunkHasNulls = chunkNonNullCount != (endOffset - offset);
    if (chunkHasNulls && !hasNulls) {
      // back fill the nulls bitmap to all non-nulls
      velox::bits::BitmapBuilder builder{nullsPtr, offset};
      builder.set(0, offset);
    }
    hasNulls = hasNulls || chunkHasNulls;
    if (hasNulls && !chunkHasNulls) {
      // fill nulls bitmap to reflect that all values are non-null
      velox::bits::BitmapBuilder builder{nullsPtr, endOffset};
      builder.set(offset, endOffset);
    }

    if (encoding_->dataType() == DataType::String &&
        !optimizeStringBufferHandling_ && rowsToRead != count) {
      // We are going to load a new chunk.
      // For string values, this means that the memory pointed by the
      // string_views is going to be freed. Before we do so, we copy all
      // strings to a temporary buffer and fix the string_views to point to the
      // new location.
      // NOTE1: Instead of copying the data, we can just hold on to
      // the previous chunk(s) for a while. However, this means holding on to
      // more memory, which is undesirable.
      // NOTE2: We perform an additional copy of the strings later on, into the
      // string buffers of the Velox Vector. Later diff will change this logic
      // to directly copy the strings into the Velox string buffers directly.
      auto& buffer = stringBuffers_.emplace_back(&pool_);
      bufferStringContent(
          buffer, output, nullsPtr, offset, (endOffset - offset));
    }

    offset = endOffset;
    remaining_ -= rowsToRead;
    count -= rowsToRead;
    nonNullCount += chunkNonNullCount;

    // Note: we can over queue the current string buffers by exactly
    // once, but that doesn't change the life cycle of the buffers for now.
    // Keeping this pattern for simplicity.
    // For non-string types, currentStringBuffers_ will be empty
    stringBuffers.reserve(currentStringBuffers_.size());
    for (const auto& buf : currentStringBuffers_) {
      stringBuffers.push_back(buf);
    }
  }

  return nonNullCount;
}

void ChunkedStreamDecoder::skip(uint32_t count) {
  while (count > 0) {
    ensureLoaded();
    auto toSkip = std::min(count, remaining_);
    encoding_->skip(toSkip);
    count -= toSkip;
    remaining_ -= toSkip;
  }
}

void ChunkedStreamDecoder::reset() {
  stream_->reset();
  remaining_ = 0;
}

void ChunkedStreamDecoder::ensureLoaded() {
  if (UNLIKELY(remaining_ == 0)) {
    currentStringBuffers_.clear();
    encoding_ = encodingFactory_(
        pool_, stream_->nextChunk(), [&](uint32_t totalLength) {
          auto& buffer = currentStringBuffers_.emplace_back(
              velox::AlignedBuffer::allocate<char>(totalLength, &pool_));
          return buffer->asMutable<void>();
        });
    remaining_ = encoding_->rowCount();
    NIMBLE_CHECK_GT(remaining_, 0, "Empty chunk");
  }
}

} // namespace facebook::nimble
