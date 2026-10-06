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

#include <cstdint>
#include <deque>
#include <memory>

#include "velox/common/EnumDeclare.h"

namespace facebook::velox::cache {

class AsyncDataCacheEntry;

/// Enumerates the concrete eviction policies known to the cache.
enum class EvictionPolicyKind : uint8_t {
  kApproxLrfu,
};
VELOX_DECLARE_ENUM_NAME(EvictionPolicyKind);

/// Per-shard mutable state opaque to CacheShard. Each EvictionPolicy defines
/// its own subclass.
class EvictionPolicyShardState {
 public:
  virtual ~EvictionPolicyShardState() = default;
};

/// Iterates eviction candidates for a single evict() call. Constructed by the
/// policy under the shard mutex and walked once.
class EvictionCandidateCursor {
 public:
  virtual ~EvictionCandidateCursor() = default;

  /// Represents a single candidate returned by next(). entry == nullptr
  /// signals the cursor is exhausted. entryIndex is the position of entry in
  /// the shard's entries deque, used by the shard to reclaim the slot after
  /// eviction. score is meaningful for score-based policies (LRFU) and zero
  /// for others.
  struct Candidate {
    AsyncDataCacheEntry* entry{nullptr};
    int32_t entryIndex{-1};
    int32_t score{0};
  };

  /// Yields the next candidate the policy considers evictable, in
  /// policy-defined priority order.
  virtual Candidate next() = 0;
};

/// Provides a pluggable RAM cache eviction policy. Instances are shared
/// across shards; per-shard mutable state lives in EvictionPolicyShardState.
/// All methods are called under the owning shard's mutex.
class EvictionPolicy {
 public:
  virtual ~EvictionPolicy() = default;

  /// Constructs an instance for the given kind, with policy-specific
  /// defaults.
  static std::unique_ptr<EvictionPolicy> create(EvictionPolicyKind kind);

  virtual std::unique_ptr<EvictionPolicyShardState> makeShardState() const = 0;

  /// Called on cache hit after AccessStats::touch(). Override to update
  /// per-entry state that isn't captured by AccessStats.
  virtual void onEntryAccessLocked(
      AsyncDataCacheEntry& entry,
      EvictionPolicyShardState& shardState) const = 0;

  /// Called after a new entry becomes visible in the shard. Override to
  /// register the entry in policy-owned data structures (e.g. 2Q queue).
  virtual void onEntryInsertedLocked(
      AsyncDataCacheEntry& entry,
      EvictionPolicyShardState& shardState) const = 0;

  /// Called before an entry is unlinked from the shard (eviction, aging, or
  /// file removal). Override to unregister the entry from policy-owned data
  /// structures.
  virtual void onEntryRemovedLocked(
      AsyncDataCacheEntry& entry,
      EvictionPolicyShardState& shardState) const = 0;

  /// Builds a cursor for a single evict() call. Cursor lifetime is scoped to
  /// that call, all under the shard mutex. eventCounter is the shard's
  /// hit-count-based recalibration trigger; the cursor may read and reset it.
  /// evictAllUnpinned is a panic mode: cursor should yield every entry
  /// regardless of the algorithm's normal ordering.
  virtual std::unique_ptr<EvictionCandidateCursor> createEvictionCursorLocked(
      EvictionPolicyShardState& shardState,
      const std::deque<std::unique_ptr<AsyncDataCacheEntry>>& entries,
      uint32_t& clockHand,
      uint32_t& eventCounter,
      bool evictAllUnpinned) const = 0;
};

} // namespace facebook::velox::cache
