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

#include <functional>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {

/// Thread-safe cache that stores entries as either strong or weak references.
///
/// When pinEntries is true, all entries are stored as strong references in
/// pinnedEntries and never expire.
///
/// When pinEntries is false (default), on-demand entries use weak references
/// that expire when no external holder remains. Entries added via pin() always
/// hold strong references regardless of the pinEntries setting.
///
/// This class is thread-safe. All state is protected by a single shared_mutex.
/// Read-lock is used for cache lookups (common path), write-lock only for
/// inserts (rare path).
template <typename Key, typename Value>
class MetadataCache {
 public:
  using ValueBuilder = std::function<std::shared_ptr<Value>(Key)>;

  explicit MetadataCache(ValueBuilder builder, bool pinEntries = false)
      : valueBuilder_{std::move(builder)}, pinEntries_{pinEntries} {}

  /// Returns the cached entry for the given key, or creates one using the
  /// default builder if not present.
  std::shared_ptr<Value> getOrCreate(Key key) {
    return getOrCreateCacheEntry(key, valueBuilder_);
  }

  /// Returns the cached entry for the given key, or creates one using the
  /// provided builder if not present.
  std::shared_ptr<Value> getOrCreate(Key key, const ValueBuilder& builder) {
    return getOrCreateCacheEntry(key, builder);
  }

  /// Pins an entry with a strong reference that never expires, regardless
  /// of the pinEntries setting.
  void pin(Key key, std::shared_ptr<Value> value) {
    std::unique_lock lock(mutex_);
    pinnedEntries_.emplace(key, std::move(value));
  }

  /// Returns the number of non-expired cached entries for testing purposes.
  size_t testingCacheCount() const {
    std::shared_lock lock(mutex_);
    size_t count = pinnedEntries_.size();
    if (pinEntries_) {
      NIMBLE_CHECK(cacheEntries_.empty());
      return count;
    }
    for (const auto& [_, entry] : cacheEntries_) {
      if (!entry.expired()) {
        ++count;
      }
    }
    return count;
  }

  /// Returns all valid (non-expired) cache entries across all internal maps
  /// (pinned and weak cache) for testing purposes.
  std::unordered_map<Key, std::shared_ptr<Value>> testingGetAllEntries() const {
    std::shared_lock lock(mutex_);
    std::unordered_map<Key, std::shared_ptr<Value>> result;
    for (const auto& [key, value] : pinnedEntries_) {
      result.emplace(key, value);
    }
    for (const auto& [key, weakValue] : cacheEntries_) {
      auto sharedPtr = weakValue.lock();
      if (sharedPtr != nullptr) {
        result.emplace(key, std::move(sharedPtr));
      }
    }
    return result;
  }

  /// Returns whether the given key has a non-expired cached entry.
  bool hasCacheEntry(Key key) const {
    std::shared_lock lock(mutex_);
    if (pinnedEntries_.contains(key)) {
      return true;
    }
    if (pinEntries_) {
      return false;
    }
    auto it = cacheEntries_.find(key);
    if (it == cacheEntries_.end()) {
      return false;
    }
    return !it->second.expired();
  }

 private:
  // Uses read-lock → release → build → write-lock → check-again-and-insert.
  // Most accesses hit the cache on the first read lock, so the write lock and
  // build are rarely needed. Duplicate builds may occur under contention but
  // are harmless since builds are idempotent and infrequent (once per key).
  std::shared_ptr<Value> getOrCreateCacheEntry(
      Key key,
      const ValueBuilder& builder) {
    {
      std::shared_lock lock(mutex_);
      // Check pinned entries first — these never expire.
      auto pinnedIt = pinnedEntries_.find(key);
      if (pinnedIt != pinnedEntries_.end()) {
        return pinnedIt->second;
      }
      if (!pinEntries_) {
        auto it = cacheEntries_.find(key);
        if (it != cacheEntries_.end()) {
          auto element = it->second.lock();
          if (element != nullptr) {
            return element;
          }
        }
      } else {
        NIMBLE_CHECK(cacheEntries_.empty());
      }
    }

    // Build outside lock if not found in cache.
    auto element = builder(key);

    std::unique_lock lock(mutex_);
    if (pinEntries_) {
      auto [it, _] = pinnedEntries_.emplace(key, std::move(element));
      return it->second;
    }

    // Insert or update weak cache entry.
    auto [it, inserted] = cacheEntries_.emplace(key, element);
    if (!inserted) {
      // Entry exists — return it if alive, otherwise update with our element.
      auto existing = it->second.lock();
      if (existing != nullptr) {
        return existing;
      }
      it->second = element;
    }
    return element;
  }

  const ValueBuilder valueBuilder_;
  const bool pinEntries_;

  mutable std::shared_mutex mutex_;

  // Strong references that never expire. Used for all entries when
  // pinEntries_ is true, and for explicitly pinned entries otherwise.
  std::unordered_map<Key, std::shared_ptr<Value>> pinnedEntries_;

  // Weak references when pinEntries_ is false. Expire when no external
  // holder remains.
  std::unordered_map<Key, std::weak_ptr<Value>> cacheEntries_;
};

} // namespace facebook::nimble
