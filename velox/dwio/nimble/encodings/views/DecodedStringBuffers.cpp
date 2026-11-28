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

#include "velox/dwio/nimble/encodings/views/DecodedStringBuffers.h"

#include <algorithm>
#include <utility>

#include "velox/common/Casts.h"
#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {

DecodedStringBuffers::DecodedStringBuffers(velox::memory::MemoryPool* pool)
    : pool_{velox::checkedNotNull(pool)} {}

void DecodedStringBuffers::clear() {
  buffer_ = nullptr;
  usedBytes_ = 0;
  nextBufferBytes_ = kMinBufferBytes;
}

std::span<char> DecodedStringBuffers::reserve(size_t numReservedBytes) {
  NIMBLE_CHECK(
      !decoding_,
      "A decode must not start before the decode in progress finishes.");
  if (numReservedBytes > kMaxBufferBytes) {
    oversizeBuffer_ =
        velox::AlignedBuffer::allocate<char>(numReservedBytes, pool_);
    decoding_ = true;
    return {oversizeBuffer_->asMutable<char>(), oversizeBuffer_->size()};
  }
  if (buffer_ == nullptr || numReservedBytes > buffer_->size() - usedBytes_) {
    // Takes the header out of the bytes of a buffer of the usual size, so
    // that it takes a power of two from the pool.
    buffer_ = velox::AlignedBuffer::allocate<char>(
        std::max(
            numReservedBytes,
            nextBufferBytes_ - velox::AlignedBuffer::kPaddedSize),
        pool_);
    nextBufferBytes_ = std::min(2 * nextBufferBytes_, kMaxBufferBytes);
    usedBytes_ = 0;
  }
  decoding_ = true;
  return {
      buffer_->asMutable<char>() + usedBytes_, buffer_->size() - usedBytes_};
}

velox::BufferPtr DecodedStringBuffers::finish(size_t numUsedBytes) {
  NIMBLE_CHECK(decoding_, "No decode is in progress to finish.");
  if (oversizeBuffer_ != nullptr) {
    NIMBLE_CHECK_LE(numUsedBytes, oversizeBuffer_->size());
    decoding_ = false;
    return std::move(oversizeBuffer_);
  }
  NIMBLE_CHECK_LE(numUsedBytes, buffer_->size() - usedBytes_);
  decoding_ = false;
  usedBytes_ += numUsedBytes;
  return buffer_;
}

} // namespace facebook::nimble
