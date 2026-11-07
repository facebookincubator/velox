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

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <folly/Synchronized.h>

#include "velox/buffer/BufferPool.h"
#include "velox/dwio/nimble/serializer/StreamReader.h"

namespace facebook::velox::memory {
class MemoryPool;
} // namespace facebook::velox::memory

namespace facebook::nimble {

/// Keeps the idle stream readers of one projected schema, which the row
/// iterators of a file share, so that a new iterator reuses the readers of
/// earlier iterators instead of building its own. An entry holds a stream
/// reader of the schema, with its field readers and stream decoders, the leaf
/// pool that the reader allocates from, and the pool of its scratch buffers. A
/// reader takes the encoding options of a stripe when it binds the stripe, so
/// iterators with other encoding options share entries. The cache does not know
/// the schema of its readers, so every iterator that shares a cache must
/// project the same schema.
/// Thread-safe: the iterators get and release entries on any thread.
class StreamReaderCache {
 public:
  /// A stream reader of the schema, which one iterator at a time uses. The
  /// reader keeps its field readers from stripe to stripe and rebinds its
  /// stream decoders to the chunks of the next stripe.
  struct Entry {
    /// Takes 'reader', which allocates from 'pool'. When get() returns none,
    /// the caller builds the reader on a pool from addReaderPool().
    Entry(
        std::shared_ptr<velox::memory::MemoryPool> pool,
        std::unique_ptr<StreamReader> reader);

    /// The leaf pool that the reader allocates from. Only the iterator that
    /// holds the entry allocates from it, so iterators on several threads do
    /// not wait on the mutex of one shared pool. Declared first, so that it
    /// outlives the memory of the reader.
    const std::shared_ptr<velox::memory::MemoryPool> pool;

    /// Recycles the scratch buffers of the reader, which binds every stripe
    /// with this pool. One iterator at a time uses the entry, so the pool
    /// serves one thread at a time. Declared before the reader, whose views
    /// return their buffers to it when they are destroyed.
    velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};

    /// Decodes the stripes.
    const std::unique_ptr<StreamReader> streamReader;
  };

  /// Keeps at most 'maxReaders' idle readers. The reader of each entry
  /// allocates from a leaf pool of its own under the aggregate pool 'pool',
  /// which must outlive the cache. The cache numbers the names of its pools
  /// from 0, so two caches must not share 'pool'.
  StreamReaderCache(
      const std::shared_ptr<velox::memory::MemoryPool>& pool,
      uint32_t maxReaders);

  /// Returns a new leaf pool under the pool of the cache, for the reader of a
  /// new entry. The pool is thread-safe, since the iterator that holds the
  /// entry can move between threads.
  std::shared_ptr<velox::memory::MemoryPool> addReaderPool();

  /// Returns the idle entry that was released last, or nullptr when there is
  /// none.
  std::unique_ptr<Entry> get();

  /// Unbinds the reader of 'entry' from the streams of its last stripe, which
  /// the stripe frees, and keeps the entry for a later iterator. Destroys the
  /// entry instead when the cache already keeps its maximum of idle entries.
  void release(std::unique_ptr<Entry> entry);

 private:
  // The aggregate pool that holds the leaf pools of the readers.
  velox::memory::MemoryPool* const pool_;
  // The idle readers that the cache keeps at most.
  const uint32_t maxReaders_;
  // The number in the name of the next pool of a reader, since the names of
  // the children of 'pool_' must be unique.
  std::atomic_uint64_t nextReaderPool_{0};
  // The idle entries, with the entry released last at the back. Has room for
  // maxReaders_ entries from the start, so a release never grows it. Every
  // get() and release() changes the entries, so a plain mutex guards them
  // rather than a shared one.
  folly::Synchronized<std::vector<std::unique_ptr<Entry>>, std::mutex>
      cachedReaders_;
};

} // namespace facebook::nimble
