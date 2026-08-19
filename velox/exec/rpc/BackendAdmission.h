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
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "velox/common/future/VeloxPromise.h"

namespace facebook::velox::exec::rpc {

/// Admission control for one backend service.
///
/// A backend (e.g. "translation-service") is shared by every driver in the
/// process that dispatches to it, so admission is a process-scoped concern:
/// each Presto worker holds its own ServiceRouter connections, and
/// cross-worker coordination is the backend's own job. Instances are obtained
/// from BackendRegistry rather than constructed directly, except in tests.
///
/// Capacity is a semaphore whose ceiling is configured and whose current value
/// adapts (AIMD) to overload reported by the caller. Occupancy is held as a
/// pending count plus a FIFO queue of parked drivers.
class BackendAdmission {
 public:
  /// Tuning for one backend. One field per user-facing session property, kept
  /// deliberately literal so the mapping is checkable by eye.
  struct Config {
    /// Whether capacity adapts to overload. When false, capacity is pinned at
    /// the ceiling. Maps to rpc_rate_limiter_adaptive_enabled.
    bool adaptive{false};

    /// Unshrunk capacity. Zero defers to defaultCapacity(), resolved on read
    /// so a later change to the process default still takes effect. Maps to
    /// rpc_rate_limiter_max_limit.
    int64_t ceiling{0};

    /// Lower bound on adapted capacity. Maps to rpc_rate_limiter_min_limit.
    int64_t floor{1};

    /// Multiplicative-decrease factor applied on overload, in (0, 1). Maps to
    /// rpc_rate_limiter_decrease_factor.
    double decreaseFactor{0.5};
  };

  /// What one drained batch of work told us about the backend. The admission
  /// layer's own vocabulary: callers translate their function-level congestion
  /// signal into it, which keeps this target free of any dependency on the
  /// expression layer.
  enum class Outcome {
    /// The backend served the work. Drives additive increase.
    kSuccess,
    /// The backend shed load. Drives multiplicative decrease.
    kOverload,
    /// Nothing to learn from; capacity is left alone.
    kNone,
  };

  /// A snapshot for per-query runtime stats, read once at operator close().
  struct Stats {
    /// Current capacity, after any adaptation.
    int64_t capacity{0};

    /// In-flight requests right now.
    int64_t pending{0};

    /// High-water in-flight count over the backend's lifetime.
    int64_t peakPending{0};

    /// Lowest capacity ever reached, or the ceiling when capacity never
    /// shrank. Reporting the ceiling rather than a sentinel zero means the
    /// value is always meaningful, so readers need no special case.
    int64_t lowWaterCapacity{0};
  };

  /// One in-flight request slot, released on destruction so a slot is returned
  /// even when the RPC future is abandoned (e.g. query cancellation).
  ///
  /// Holds a back-pointer to its issuing BackendAdmission, which is safe only
  /// because tokens are captured into continuations that outlive the operator
  /// and the registry outlives every future. A locally constructed
  /// BackendAdmission must therefore outlive any token taken from it.
  class Token {
   public:
    Token() = default;

    /// Takes ownership of one slot already claimed on 'owner'. Prefer
    /// BackendAdmission::acquire(); this is public only to avoid a friend
    /// declaration.
    explicit Token(BackendAdmission* owner) : owner_{owner} {}

    Token(Token&& other) noexcept : owner_{other.owner_} {
      other.owner_ = nullptr;
    }

    Token& operator=(Token&& other) noexcept;

    ~Token();

    Token(const Token&) = delete;
    Token& operator=(const Token&) = delete;

   private:
    // Null means moved-from or default-constructed: no slot is held.
    BackendAdmission* owner_{nullptr};
  };

  explicit BackendAdmission(std::string backendKey);

  /// Applies tuning. Last writer wins, matching the two independent writers
  /// that configure a backend today: the function's SQL option first, then the
  /// session property when it is set. Never touches work already in flight —
  /// a ceiling that lands below the current pending count stops admission but
  /// cancels nothing and wakes no waiter, because nothing was freed.
  void configure(const Config& config);

  /// Applies 'mutate' to this backend's tuning under the lock, so a
  /// read-modify-write cannot lose a concurrent writer's fields. Prefer this
  /// over config() + configure() whenever only some fields are being changed:
  /// two initializers race on the same backend when the operator amends the
  /// session-property fields while the function sets its SQL-option ceiling.
  void amend(const std::function<void(Config&)>& mutate);

  /// The tuning currently in effect. Lets a later writer amend individual
  /// fields without discarding what an earlier one set — the operator raises
  /// the ceiling from the session property only when it is set, and must
  /// otherwise preserve the ceiling the function configured from its SQL
  /// option.
  Config config() const;

  /// Claims a slot unconditionally and returns the token that releases it.
  /// Does not block or check capacity; callers gate on available() or
  /// waitForCapacity() first.
  Token acquire();

  /// Slots free right now: capacity minus pending, floored at zero.
  /// Admission-controlled dispatch sizes each chunk by this so the cap
  /// actually bounds offered concurrency instead of being overrun by a
  /// whole-vector blast.
  int64_t available() const;

  /// Parks the caller when the backend is at capacity. Returns a future that
  /// resolves once a slot frees, or nullopt to proceed immediately. Enrols a
  /// waiter, so this mutates despite reading like a query.
  std::optional<ContinueFuture> waitForCapacity();

  /// Feeds one drained batch's verdict back into capacity. 'units' is the
  /// number of successful items the batch represents and is read only for
  /// Outcome::kSuccess, where recovery is AIMD-linear in it; the other
  /// outcomes ignore it, so an error path need not count anything.
  void onOutcome(Outcome outcome, int64_t units);

  Stats stats() const;

  /// Fallback ceiling for backends configured with ceiling == 0.
  static void setDefaultCapacity(int64_t capacity);
  static int64_t defaultCapacity();

  /// Restores this backend to its as-constructed state. Intended only for
  /// tests. Resets in place rather than being destroyed and recreated, because
  /// outstanding tokens hold a back-pointer here and release through it.
  void testingReset();

 private:
  // Capacity under the current adaptive state. Caller holds mutex_.
  int64_t capacityLocked() const;

  // Ceiling with Config::ceiling == 0 resolved against the process default.
  // Caller holds mutex_.
  int64_t ceilingLocked() const;

  // Config::floor clamped to at most the ceiling, so a min_limit configured
  // above the ceiling cannot make std::clamp's lo exceed its hi (undefined
  // behavior). Caller holds mutex_.
  int64_t floorLocked(int64_t ceiling) const;

  void onOverload();
  void onSuccess(int64_t units);

  // Called by Token on destruction; releases the slot and hands it to the
  // longest-waiting parked driver, if any.
  void release();

  const std::string backendKey_;

  // Guards config_, capacity_, lowWater_ and waiters_. pending_ and
  // peakPending_ are atomic so the hot increment path stays lock-free, but
  // they are read under the lock wherever a capacity comparison depends on
  // them.
  mutable std::mutex mutex_;

  // Mutable rather than a constructor argument: two writers configure a backend
  // in a fixed order, and the forwarding shim in the reshape needs a seam to
  // write through.
  Config config_;

  // Adapted capacity. Zero means unshrunk, i.e. defer to the ceiling.
  int64_t capacity_{0};

  // Lowest capacity ever reached. Zero means never shrank; resolved to the
  // ceiling when reported through Stats.
  int64_t lowWater_{0};

  std::atomic<int64_t> pending_{0};
  std::atomic<int64_t> peakPending_{0};

  std::deque<ContinuePromise> waiters_;
};

/// Process-scoped owner of one BackendAdmission per backend key.
///
/// A backend is shared across queries by definition, so the registry is a
/// process singleton rather than something threaded through the operator.
class BackendRegistry {
 public:
  static BackendRegistry& global();

  /// Returns the backend's admission control, creating it on first sight. The
  /// reference stays valid for the process lifetime: values are held by
  /// unique_ptr, so later insertions move only the map nodes.
  BackendAdmission& get(const std::string& backendKey);

  /// Drops every backend and restores process-global defaults. Intended only
  /// for tests and benchmarks that drive a real operator, which resolves its
  /// backend through this registry and so needs process state reset between
  /// iterations. Never call this from production code.
  void testingReset();

 private:
  // Read-mostly: backends are created once then only looked up, so lookups take
  // a shared lock and do not serialize against each other. Only first sight
  // of a backend takes the exclusive lock.
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<BackendAdmission>> backends_;
};

} // namespace facebook::velox::exec::rpc
