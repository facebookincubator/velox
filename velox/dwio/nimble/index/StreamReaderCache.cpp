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

#include "velox/dwio/nimble/index/StreamReaderCache.h"

#include <string>
#include <utility>

#include "velox/common/Casts.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {

StreamReaderCache::Entry::Entry(
    std::shared_ptr<velox::memory::MemoryPool> pool,
    std::unique_ptr<StreamReader> reader)
    : pool{velox::checkedNotNull(std::move(pool))},
      streamReader{velox::checkedNotNull(std::move(reader))} {}

StreamReaderCache::StreamReaderCache(
    const std::shared_ptr<velox::memory::MemoryPool>& pool,
    uint32_t maxReaders)
    : pool_{velox::checkedNotNull(pool.get())}, maxReaders_{maxReaders} {
  NIMBLE_CHECK(
      !pool_->isLeaf(),
      "The pool of a stream reader cache must be an aggregate pool");
  cachedReaders_.lock()->reserve(maxReaders_);
}

std::shared_ptr<velox::memory::MemoryPool> StreamReaderCache::addReaderPool() {
  return pool_->addLeafChild(
      "nimble_stream_reader_" +
          std::to_string(
              nextReaderPool_.fetch_add(1, std::memory_order_relaxed)),
      /*threadSafe=*/true);
}

std::unique_ptr<StreamReaderCache::Entry> StreamReaderCache::get() {
  auto cachedReaders = cachedReaders_.lock();
  if (cachedReaders->empty()) {
    return nullptr;
  }
  auto entry = std::move(cachedReaders->back());
  cachedReaders->pop_back();
  return entry;
}

void StreamReaderCache::release(std::unique_ptr<Entry> entry) {
  entry->streamReader->reset();
  auto cachedReaders = cachedReaders_.lock();
  if (cachedReaders->size() >= maxReaders_) {
    // Destroys 'entry' after the lock is released, since the parameter
    // outlives the locals.
    return;
  }
  cachedReaders->push_back(std::move(entry));
}

} // namespace facebook::nimble
