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
#pragma once

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

// Basic memory buffer interface, aka arena.
//
// Standard usage:
//   char* pos = buffer->reserve(100);
//   write data to [pos, pos + 100)
//   pos = buffer->Reserve(27);
//   write data to [pos, pos + 27)
//   and so on

namespace facebook::nimble {

/// Internally manages memory in chunks. Releases memory only upon destruction.
///
/// NOTE: This class is not thread-safe. External locking is required.
class Buffer {
  using MemoryPool = facebook::velox::memory::MemoryPool;

 public:
  explicit Buffer(MemoryPool& pool, uint64_t initialChunkSize = kMinChunkSize)
      : pool_{&pool} {
    addChunk(initialChunkSize);
    reserveEnd_ = pos_;
  }

  /// Returns a pointer to a block of memory of size bytes that can be written
  /// to, and guarantees for the lifetime of *this that that region will remain
  /// valid. Does NOT guarantee that the region is initially 0'd.
  char* reserve(uint64_t bytes);

  /// Copies |data| into the chunk, returning a view to the copied data.
  std::string_view writeString(std::string_view data) {
    char* pos = reserve(data.size());
    // @lint-ignore CLANGSECURITY facebook-security-vulnerable-memcpy
    std::memcpy(pos, data.data(), data.size());
    return {pos, data.size()};
  }

  MemoryPool& getMemoryPool() {
    return *pool_;
  }

  std::string_view takeOwnership(velox::BufferPtr&& bufferPtr) {
    std::string_view chunk{bufferPtr->as<char>(), bufferPtr->size()};
    chunks_.push_back(std::move(bufferPtr));
    return chunk;
  }

  /// Transfers ownership of allocated chunks.
  std::vector<velox::BufferPtr> transferBuffers();

  /// Resets write pointers to the beginning of the first chunk.
  /// Keeps all allocated chunks around for reuse. Previously returned
  /// pointers/string_views are invalidated.
  void reset();

  /// Returns the number of allocated chunks. Only for testing.
  uint32_t testingChunkCount() const {
    return chunks_.size();
  }

  /// Returns the current chunk index. Only for testing.
  uint32_t testingCurrentChunkIndex() const {
    return chunkIndex_;
  }

 private:
  static constexpr uint64_t kMinChunkSize = 1LL << 20;

  // Tries to advance to the next existing chunk that can fit 'bytes'.
  // Returns true if a suitable chunk was found, false if a new allocation
  // is needed. Must be called under mutex_.
  bool tryAdvanceToNextChunk(uint64_t bytes);

  void addChunk(uint64_t bytes);

  // --- Const members ---
  MemoryPool* const pool_;

  // --- Mutable members (protected by mutex_) ---

  // NOTE: this is temporary fix, to quickly enable parallel access to the
  // buffer class. In the near future, we are going to templatize this class to
  // produce a concurrent and a non-concurrent variants, and change the call
  // sites to use each variant when needed.
  std::mutex mutex_;
  char* chunkEnd_;
  char* pos_;
  char* reserveEnd_;
  uint32_t chunkIndex_{0};
  std::vector<velox::BufferPtr> chunks_;
};

/// Reuses Nimble encoding scratch buffers across nested encode calls.
/// Not thread-safe: each writer thread or encode task should use its own pool.
class EncodingBufferPool {
 private:
  using MemoryPool = facebook::velox::memory::MemoryPool;

 public:
  static constexpr uint32_t kDefaultMaxCachedBuffers = 8;

  explicit EncodingBufferPool(
      MemoryPool* pool,
      uint32_t maxCachedBuffers = kDefaultMaxCachedBuffers);

  std::unique_ptr<Buffer> acquire();

  void release(std::unique_ptr<Buffer> buffer);

 private:
  MemoryPool* const pool_;
  const uint32_t maxCachedBuffers_;

  std::vector<std::unique_ptr<Buffer>> buffers_;
};

class ScopedEncodingBuffer {
  using MemoryPool = facebook::velox::memory::MemoryPool;

 public:
  ScopedEncodingBuffer(MemoryPool* memoryPool, EncodingBufferPool* bufferPool);

  ~ScopedEncodingBuffer();

  ScopedEncodingBuffer(const ScopedEncodingBuffer&) = delete;
  ScopedEncodingBuffer& operator=(const ScopedEncodingBuffer&) = delete;

  Buffer& get() {
    return *buffer_;
  }

 private:
  EncodingBufferPool* const bufferPool_;
  std::unique_ptr<Buffer> buffer_;
};

} // namespace facebook::nimble
