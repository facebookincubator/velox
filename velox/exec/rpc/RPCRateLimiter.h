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
#include <shared_mutex>
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

  /// Available dispatch headroom for a tier: max(0, effectiveCap - pending).
  /// Admission-controlled dispatch uses this to size each drip chunk so the
  /// per-tier cap actually bounds offered concurrency (rather than being
  /// overrun by a whole-vector blast).
  static int64_t availableHeadroom(const std::string& tierKey);

  /// Configure the max pending limit for a specific tier.
  /// If not configured, falls back to the global default.
  static void setMaxPending(const std::string& tierKey, int64_t limit);

  /// Set the global default max pending (used when no per-tier config).
  static void setDefaultMaxPending(int64_t limit);

  /// Get the global default max pending.
  static int64_t defaultMaxPending();

  /// Configure the process-global adaptive AIMD limiter. When enabled, each
  /// tier's effective cap adapts: multiplicative-decrease (by decreaseFactor,
  /// floored at minLimit) on onRateLimited(), additive-increase back toward the
  /// static per-tier cap on onSuccess(). When disabled (default), the static
  /// per-tier cap is used and behavior is unchanged. Idempotent; safe to call
  /// per operator setup with the cluster-default config.
  static void
  setAdaptiveConfig(bool enabled, int64_t minLimit, double decreaseFactor);

  /// Whether the adaptive limiter is currently enabled.
  static bool adaptiveEnabled();

  /// Overload signal for a tier: multiplicative-decrease its adaptive cap.
  /// No-op when the adaptive limiter is disabled. Call once per
  /// overload-classified drain (not per errored row).
  static void onRateLimited(const std::string& tierKey);

  /// Recovery signal for a tier: additive-increase its adaptive cap back toward
  /// the static ceiling, waking a waiter if headroom opened. No-op when the
  /// adaptive limiter is disabled or the tier is already at its ceiling.
  ///
  /// @param successes Number of successful units this signal represents (e.g.
  /// the row count of a drained PER_ROW batch). Recovery is AIMD-linear — one
  /// step of additive-increase per cap-worth of successes — so it tracks the
  /// multiplicative-decrease's aggressiveness instead of crawling +1 per
  /// arbitrarily-sized drain (which never recovered within a query).
  static void onSuccess(const std::string& tierKey, int64_t successes);

  /// Observability: the tier's current effective cap (adaptive value if shrunk,
  /// else the static ceiling). Snapshotted into per-query runtime stats.
  static int64_t currentLimit(const std::string& tierKey);

  /// Observability: high-water in-flight (pending) count for the tier.
  static int64_t peakPending(const std::string& tierKey);

  /// Observability: lowest adaptive cap the tier ever reached (0 = never shrank
  /// / adaptive disabled).
  static int64_t minLimitReached(const std::string& tierKey);

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
    // Adaptive AIMD cap. 0 = unshrunk (fall back to maxPending/default as the
    // effective cap). Only consulted when the adaptive limiter is enabled.
    int64_t adaptiveLimit{0};
    std::deque<ContinuePromise> waiters;
    // Observability: high-water pending count and the lowest adaptive cap ever
    // reached for this tier (0 = never shrank). Read at operator close() so the
    // adaptive cap's trajectory is visible in per-query runtime stats.
    std::atomic<int64_t> peakPending{0};
    int64_t minAdaptiveLimit{0};
  };

  static void incrementPending(const std::string& tierKey);
  static void decrementPending(const std::string& tierKey);

  static TierState& getOrCreateTierState(const std::string& tierKey);

  // Read-write mutex over the tier map. Lookups (the hot path: every dispatch
  // and completion resolves its TierState here) take a shared lock and so do
  // not serialize against each other; only first-time tier creation takes the
  // exclusive lock. The map is read-mostly (tiers are created once, then only
  // looked up), so this removes a worker-wide serialization point.
  static std::shared_mutex& mapMutex();
  static std::atomic<int64_t>& defaultMaxPendingRef();
  static std::unordered_map<std::string, std::unique_ptr<TierState>>& tiers();

  // Adaptive limiter config (process-global).
  static std::atomic<bool>& adaptiveEnabledRef();
  static std::atomic<int64_t>& adaptiveMinRef();
  static std::atomic<double>& adaptiveFactorRef();

  // Effective per-tier cap under the current adaptive state. Caller must hold
  // state.mutex.
  static int64_t effectiveLimitLocked(const TierState& state);
};

} // namespace facebook::velox::exec::rpc
