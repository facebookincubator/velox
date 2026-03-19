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

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "velox/common/future/VeloxPromise.h"

namespace facebook::velox::exec::rpc {

/// Per-tier (per-process) rate limiter for RPC dispatch.
///
/// Each backend service tier (e.g., "service.backend.prod") gets its own
/// independent concurrency limit and waiter queue. This allows different
/// backends to have different concurrency budgets.
///
/// The rate limiter is per-process. Each Presto worker is a separate process
/// with its own ServiceRouter connections. Cross-worker coordination is
/// handled by the backend's own admission control.
///
/// The tier key comes from IRPCClient::tierKey(). Empty string means
/// "no tier configured" and falls back to the global default limit.
class RPCRateLimiter {
 public:
  /// RAII token representing one in-flight request slot.
  /// Decrements the pending count on destruction, guaranteeing cleanup
  /// even if the RPC future is abandoned (e.g., query cancellation).
  class Token {
   public:
    Token() = default;

    Token(Token&& other) noexcept
        : tierKey_(std::move(other.tierKey_)), valid_(other.valid_) {
      other.valid_ = false;
    }

    Token& operator=(Token&& other) noexcept;

    ~Token();

    // Non-copyable.
    Token(const Token&) = delete;
    Token& operator=(const Token&) = delete;

   private:
    friend class RPCRateLimiter;
    explicit Token(const std::string& tierKey);

    std::string tierKey_;
    bool valid_{false};
  };

  /// Acquire a slot for the given tier. Increments pending count and
  /// returns a Token that will decrement on destruction.
  static Token acquire(const std::string& tierKey);

  /// Check backpressure for a specific tier.
  /// Returns a future to wait on if that tier's limit is reached.
  /// @return Future to wait on if blocked, nullopt if can proceed.
  static std::optional<ContinueFuture> checkBackpressure(
      const std::string& tierKey);

  /// Get current pending count for a specific tier.
  static int64_t pendingCount(const std::string& tierKey);

  /// Configure the max pending limit for a specific tier.
  /// If not configured, falls back to the global default.
  static void setMaxPending(const std::string& tierKey, int64_t limit);

  /// Set the global default max pending (used when no per-tier config).
  static void setDefaultMaxPending(int64_t limit);

  /// Get the global default max pending.
  static int64_t defaultMaxPending();

  /// Reset all state. Intended ONLY for unit tests
  /// to avoid test contamination across test cases.
  /// WARNING: Do NOT call this in production code.
  static void testingResetAllState();

 private:
  /// Per-tier state. Allocated via unique_ptr for pointer stability across
  /// map inserts (std::unordered_map does not invalidate existing entries).
  struct TierState {
    std::mutex mutex;
    std::atomic<int64_t> pendingCount{0};
    int64_t maxPending{0}; // 0 = use global default
    std::deque<ContinuePromise> waiters;
  };

  static void incrementPending(const std::string& tierKey);
  static void decrementPending(const std::string& tierKey);

  static TierState& getOrCreateTierState(const std::string& tierKey);

  // Global mutex protects only the tier map for inserts/lookups.
  static std::mutex& mapMutex();
  static std::atomic<int64_t>& defaultMaxPendingRef();
  static std::unordered_map<std::string, std::unique_ptr<TierState>>& tiers();
};

} // namespace facebook::velox::exec::rpc
