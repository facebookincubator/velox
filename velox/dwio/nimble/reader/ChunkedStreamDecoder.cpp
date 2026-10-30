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
#include "velox/dwio/nimble/reader/ChunkedStreamDecoder.h"

#include "velox/buffer/Buffer.h"

namespace facebook::nimble {

uint32_t ChunkedStreamDecoder::next(
    uint32_t count,
    void* output,
    std::function<void*()> getOutputNulls,
    std::vector<velox::BufferPtr>& stringBuffers,
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
  NIMBLE_CHECK_NOT_NULL(output);

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

uint32_t ChunkedStreamDecoder::remainingRows() {
  if (remaining_ == 0 && !stream_->hasNext()) {
    return 0;
  }
  ensureLoaded();
  return remaining_;
}

uint32_t ChunkedStreamDecoder::read(
    std::span<const uint32_t> /*rows*/,
    DataType /*dataType*/,
    void* /*output*/,
    std::function<void*()> /*getOutputNulls*/,
    std::vector<velox::BufferPtr>& /*stringBuffers*/) {
  NIMBLE_UNSUPPORTED(
      "ChunkedStreamDecoder does not support selective row decoding");
}

uint32_t ChunkedStreamDecoder::read(
    std::span<const RowRange> /*ranges*/,
    DataType /*dataType*/,
    void* /*output*/,
    std::function<void*()> /*getOutputNulls*/,
    std::vector<velox::BufferPtr>& /*stringBuffers*/) {
  NIMBLE_UNSUPPORTED(
      "ChunkedStreamDecoder does not support range row decoding");
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
        *pool_, stream_->nextChunk(), [&](uint32_t totalLength) {
          auto& buffer = currentStringBuffers_.emplace_back(
              velox::AlignedBuffer::allocate<char>(totalLength, pool_));
          return buffer->asMutable<void>();
        });
    NIMBLE_CHECK_NOT_NULL(encoding_);
    auto* const encoding = encoding_.get();
    remaining_ = encoding->rowCount();
    NIMBLE_CHECK_GT(remaining_, 0, "Empty chunk");
  }
}

} // namespace facebook::nimble
