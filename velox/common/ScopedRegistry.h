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

#include "folly/Synchronized.h"
#include "folly/container/F14Map.h"
#include "folly/container/F14Set.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox {

/// Layered key-value registry. Lookups check the local scope first, then fall
/// back to the parent chain. Supports arbitrary nesting (e.g., global ->
/// session -> query). All operations are synchronized.
///
/// Two usage modes:
///
/// - Override mode (parent != nullptr): The registry inherits all entries from
///   the parent and selectively overrides specific entries. Lookups that miss
///   locally fall through to the parent.
///
/// - Isolation mode (parent == nullptr): The registry is self-contained. Only
///   explicitly registered entries are visible.
///
/// Thread safety: Each scope has its own lock. Lookups and snapshots acquire
/// locks one scope at a time while walking the parent chain. This is safe and
/// consistent as long as parent registries are not mutated while child scopes
/// exist. Mutations should be limited to leaf scopes (e.g., per-query
/// overrides). Parent scopes may be freely mutated before children are created
/// and after all children are destroyed.
///
/// @tparam K Key type. Must be copyable, hashable, and equality-comparable.
/// @tparam V Value type. Stored as shared_ptr<V>.
template <typename K, typename V>
class ScopedRegistry {
 public:
  using ValuePtr = std::shared_ptr<V>;

  /// Create a root registry (no parent).
  ScopedRegistry() : parent_{nullptr} {}

  /// Create a derived scope that falls back to 'parent' when a key is not
  /// found locally. The parent must outlive this registry.
  explicit ScopedRegistry(const ScopedRegistry* parent) : parent_{parent} {}

  /// Insert an entry in the local scope. Returns true if the key was newly
  /// inserted. Throws if the key already exists unless 'overwrite' is true,
  /// in which case the existing entry is replaced and false is returned.
  bool insert(K key, ValuePtr value, bool overwrite = false) {
    return local_.withWLock([&](auto& map) {
      auto [it, inserted] = map.emplace(std::move(key), value);
      if (!inserted) {
        VELOX_CHECK(overwrite, "Key already registered: {}", it->first);
        it->second = std::move(value);
      }
      return inserted;
    });
  }

  /// Look up a key. Checks local scope first, then walks the parent chain.
  /// Returns nullptr if not found.
  ValuePtr find(const K& key) const {
    auto result = local_.withRLock([&](const auto& map) -> ValuePtr {
      auto it = map.find(key);
      return it != map.end() ? it->second : nullptr;
    });
    if (result) {
      return result;
    }
    return parent_ ? parent_->find(key) : nullptr;
  }

  /// Remove an entry from the local scope. Returns true if the entry was
  /// removed, false if the key was not found locally. Does not affect parent
  /// scopes.
  bool erase(const K& key) {
    return local_.withWLock([&](auto& map) { return map.erase(key) > 0; });
  }

  /// Remove all entries from the local scope. Does not affect parent scopes.
  /// Entries are moved out under the lock and destroyed outside to avoid
  /// holding the lock during potentially slow destructors.
  void clear() {
    folly::F14FastMap<K, ValuePtr> entries;
    local_.withWLock([&](auto& map) { entries.swap(map); });
  }

  /// Return a snapshot of all visible entries. Local entries take precedence
  /// over parent entries with the same key. The snapshot is a copy — no locks
  /// are held after this returns.
  std::vector<std::pair<K, ValuePtr>> snapshot() const {
    // Collect local entries.
    std::vector<std::pair<K, ValuePtr>> result;
    local_.withRLock([&](const auto& map) {
      result.reserve(map.size());
      for (const auto& [key, value] : map) {
        result.emplace_back(key, value);
      }
    });

    if (!parent_) {
      return result;
    }

    // Merge parent entries, skipping keys already present locally.
    auto parentEntries = parent_->snapshot();
    folly::F14FastSet<K> localKeys;
    localKeys.reserve(result.size());
    for (const auto& [key, _] : result) {
      localKeys.insert(key);
    }
    for (auto& [key, value] : parentEntries) {
      if (!localKeys.contains(key)) {
        result.emplace_back(std::move(key), std::move(value));
      }
    }
    return result;
  }

 private:
  folly::Synchronized<folly::F14FastMap<K, ValuePtr>> local_;
  const ScopedRegistry* parent_;
};

} // namespace facebook::velox
