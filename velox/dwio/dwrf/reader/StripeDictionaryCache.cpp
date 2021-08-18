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

#include "velox/dwio/dwrf/reader/StripeDictionaryCache.h"

namespace facebook::velox::dwrf {
StripeDictionaryCache::DictionaryEntry::DictionaryEntry(
    folly::Function<BufferPtr(velox::memory::MemoryPool*)>&& dictGen)
    : dictGen_{std::move(dictGen)} {}

BufferPtr StripeDictionaryCache::DictionaryEntry::getDictionaryBuffer(
    velox::memory::MemoryPool* pool) {
  if (!dictionaryBuffer_) {
    dictionaryBuffer_ = dictGen_(pool);
    dictGen_ = nullptr;
  }
  return dictionaryBuffer_;
}

StripeDictionaryCache::StripeDictionaryCache(velox::memory::MemoryPool* pool)
    : pool_{pool} {}

// It might be more elegant to pass in a StripeStream here instead.
void StripeDictionaryCache::registerIntDictionary(
    const EncodingKey& ek,
    folly::Function<BufferPtr(velox::memory::MemoryPool*)>&& dictGen) {
  intDictionaryFactories_.emplace(
      ek, std::make_unique<DictionaryEntry>(std::move(dictGen)));
}

BufferPtr StripeDictionaryCache::getIntDictionary(const EncodingKey& ek) {
  return intDictionaryFactories_.at(ek)->getDictionaryBuffer(pool_);
}

} // namespace facebook::velox::dwrf
