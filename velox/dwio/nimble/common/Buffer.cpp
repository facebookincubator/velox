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

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {

char* Buffer::reserve(uint64_t bytes) {
  std::scoped_lock<std::mutex> l(mutex_);
  if (reserveEnd_ + bytes <= chunkEnd_) {
    pos_ = reserveEnd_;
    reserveEnd_ += bytes;
  } else if (!tryAdvanceToNextChunk(bytes)) {
    addChunk(bytes);
  }
  return pos_;
}

std::vector<velox::BufferPtr> Buffer::transferBuffers() {
  std::scoped_lock<std::mutex> l(mutex_);
  auto transferred = std::move(chunks_);
  chunkIndex_ = 0;
  chunkEnd_ = nullptr;
  pos_ = nullptr;
  reserveEnd_ = nullptr;
  return transferred;
}

void Buffer::reset() {
  std::scoped_lock<std::mutex> l(mutex_);
  chunkIndex_ = 0;
  pos_ = chunks_.front()->asMutable<char>();
  chunkEnd_ = pos_ + chunks_.front()->capacity();
  reserveEnd_ = pos_;
}

bool Buffer::tryAdvanceToNextChunk(uint64_t bytes) {
  while (chunkIndex_ + 1 < chunks_.size()) {
    ++chunkIndex_;
    auto& chunk = chunks_[chunkIndex_];
    if (chunk->capacity() >= bytes) {
      pos_ = chunk->asMutable<char>();
      chunkEnd_ = pos_ + chunk->capacity();
      reserveEnd_ = pos_ + bytes;
      return true;
    }
  }
  return false;
}

void Buffer::addChunk(uint64_t bytes) {
  const uint64_t chunkSize = std::max(bytes, kMinChunkSize);
  auto bufferPtr = velox::AlignedBuffer::allocateExact<char>(chunkSize, pool_);
  pos_ = bufferPtr->asMutable<char>();
  chunkEnd_ = pos_ + chunkSize;
  reserveEnd_ = pos_ + bytes;
  chunks_.emplace_back(std::move(bufferPtr));
  chunkIndex_ = chunks_.size() - 1;
}

EncodingBufferPool::EncodingBufferPool(
    MemoryPool* pool,
    uint32_t maxCachedBuffers)
    : pool_{pool}, maxCachedBuffers_{maxCachedBuffers} {
  NIMBLE_CHECK_NOT_NULL(pool_, "Memory pool cannot be null");
}

std::unique_ptr<Buffer> EncodingBufferPool::acquire() {
  if (!buffers_.empty()) {
    auto buffer = std::move(buffers_.back());
    buffers_.pop_back();
    // Keep acquire() as the handoff point that guarantees a clean scratch
    // buffer, even though release() also resets before caching.
    buffer->reset();
    return buffer;
  }

  return std::make_unique<Buffer>(*pool_);
}

void EncodingBufferPool::release(std::unique_ptr<Buffer> buffer) {
  NIMBLE_CHECK_NOT_NULL(buffer, "Buffer cannot be null");

  buffer->reset();
  if (buffers_.size() < maxCachedBuffers_) {
    buffers_.emplace_back(std::move(buffer));
  }
}

ScopedEncodingBuffer::ScopedEncodingBuffer(
    MemoryPool* memoryPool,
    EncodingBufferPool* bufferPool)
    : bufferPool_{bufferPool} {
  NIMBLE_CHECK_NOT_NULL(memoryPool, "Memory pool cannot be null");
  buffer_ = bufferPool_ != nullptr ? bufferPool_->acquire()
                                   : std::make_unique<Buffer>(*memoryPool);
}

ScopedEncodingBuffer::~ScopedEncodingBuffer() {
  if (bufferPool_ != nullptr) {
    bufferPool_->release(std::move(buffer_));
  }
}

} // namespace facebook::nimble
