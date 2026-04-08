/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <vector>

#include "velox/buffer/Buffer.h"

namespace facebook::velox {

/// Caches BufferPtr objects for reuse across encoding lifetimes.
/// When encoding objects are destroyed, their scratch buffers can be returned
/// to the pool instead of being freed. New encodings can then grab
/// pre-allocated buffers from the pool, avoiding MemoryPool allocation
/// overhead (including mutex contention).
///
/// Thread safety: NOT thread-safe. Intended for use within a single
/// deserialization stream.
class BufferPool {
 public:
  /// Returns a cached buffer with capacity >= minBytes, or nullptr if none
  /// available.
  BufferPtr get(uint64_t minBytes);

  /// Returns a cached buffer (any size), or nullptr if none available.
  BufferPtr get();

  /// Returns a buffer to the pool for future reuse. Drops the buffer if the
  /// pool is full or if 'buffer' is null.
  void release(BufferPtr buffer);

  /// Returns the number of cached buffers currently in the pool.
  size_t size() const;

 private:
  // Maximum number of BufferPtr objects retained in the pool.
  static constexpr size_t kMaxCached = 32;
  // Cached buffers available for reuse.
  std::vector<BufferPtr> buffers_;
};

} // namespace facebook::velox
