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

/// Admission control for one unit of provisioned capacity: a backend tier
/// plus the credential used to reach it. Everything sharing that pair --
/// every driver, every query, both streaming modes -- draws on one quota,
/// because that is what the remote service actually provisions.
///
/// A backend (e.g. "translation-service") is shared by every driver in the
/// process that dispatches to it, so admission is a process-scoped concern:
/// each Presto worker holds its own ServiceRouter connections, and
/// cross-worker coordination is the backend's own job. Instances are obtained
/// from RPCRateLimiterRegistry rather than constructed directly, except in
/// tests.
///
/// The name says "rate limiter" but the mechanism bounds concurrency, not a
/// rate: capacity is a semaphore over in-flight units. The name is kept
/// because it is the one the user-facing session properties
/// (rpc.ratelimiter.*) and the runtime stats (rpcRateLimiterCap) already use,
/// and those cannot be renamed without breaking queries and dashboards.
///
/// Capacity is a semaphore whose ceiling is configured and whose current value
/// adapts (AIMD) to overload reported by the caller. Occupancy is held as a
/// pending count plus a FIFO queue of parked drivers.
class RPCRateLimiter {
 public:
  /// Tuning for one backend. One field per user-facing session property, kept
  /// deliberately literal so the mapping is checkable by eye.
  struct Config {
    /// Whether capacity adapts to overload. When false, capacity is pinned at
    /// the ceiling. Maps to rpc.ratelimiter.adaptive_enabled.
    bool adaptive{false};

    /// Unshrunk capacity. Zero defers to defaultCapacity(), resolved on read
    /// so a later change to the process default still takes effect. Maps to
    /// rpc.ratelimiter.max_limit.
    int64_t ceiling{0};

    /// Lower bound on adapted capacity. Maps to rpc.ratelimiter.min_limit.
    int64_t floor{1};

    /// Multiplicative-decrease factor applied on overload, in (0, 1). Maps to
    /// rpc.ratelimiter.decrease_factor.
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

    /// Lowest capacity ever reached. Zero means capacity never shrank.
    int64_t lowWaterCapacity{0};
  };

  /// One in-flight request slot, released on destruction so a slot is returned
  /// even when the RPC future is abandoned (e.g. query cancellation).
  ///
  /// Holds a back-pointer to its issuing RPCRateLimiter, which is safe only
  /// because tokens are captured into continuations that outlive the operator
  /// and the registry outlives every future. A locally constructed
  /// RPCRateLimiter must therefore outlive any token taken from it.
  class Token {
   public:
    Token() = default;

    /// Takes ownership of one slot already claimed on 'owner'. Prefer
    /// RPCRateLimiter::acquire(); this is public only to avoid a friend
    /// declaration.
    explicit Token(RPCRateLimiter* owner) : owner_{owner} {}

    Token(Token&& other) noexcept : owner_{other.owner_} {
      other.owner_ = nullptr;
    }

    Token& operator=(Token&& other) noexcept;

    ~Token();

    Token(const Token&) = delete;
    Token& operator=(const Token&) = delete;

   private:
    // Null means moved-from or default-constructed: no slot is held.
    RPCRateLimiter* owner_{nullptr};
  };

  explicit RPCRateLimiter(std::string tierKey);

  /// Applies tuning. Last writer wins, matching the two independent writers
  /// that configure a backend today: the function's SQL option first, then the
  /// session properties. rpc.ratelimiter.max_limit defaults to 200, so the
  /// session ceiling wins on every query unless it is explicitly set to 0 --
  /// the function's max_concurrent_requests is honoured only in that case.
  /// Never touches work already in flight — a ceiling that lands below the
  /// current pending count stops admission but cancels nothing and wakes no
  /// waiter, because nothing was freed.
  void configure(const Config& config);

  /// Applies 'mutate' to this backend's tuning under the lock, so a
  /// read-modify-write cannot lose a concurrent writer's fields. Prefer this
  /// over config() + configure() whenever only some fields are being changed:
  /// two initializers race on the same backend when the operator amends the
  /// session-property fields while the function sets its SQL-option ceiling.
  void amend(const std::function<void(Config&)>& mutate);

  /// Applies `mutate` on the first call for this backend and seals the
  /// configuration; later calls are refused.
  /// A backend's configuration is shared by every query running against it and
  /// has to hold still while the limiter adapts, so a later query's settings
  /// are ignored rather than allowed to move the target mid-flight. A
  /// divergent request is logged once so the no-op is visible.
  void initializeOnce(const std::function<void(Config&)>& mutate);

  /// The tuning currently in effect. Lets a later writer amend individual
  /// fields without discarding what an earlier one set — the operator writes
  /// the ceiling from rpc.ratelimiter.max_limit whenever it is positive, and
  /// must otherwise preserve the ceiling the function configured from its SQL
  /// option.
  Config config() const;

  /// Claims a slot unconditionally and returns the token that releases it.
  /// Does not block or check capacity; callers gate on available() or
  /// admitOrWait() first.
  Token acquire();

  /// Claims up to 'want' slots in one pass and returns the tokens granted,
  /// which may be fewer than asked for and may be none.
  ///
  /// Takes the backend's lock once for the whole grant rather than once per
  /// slot. A caller reserving a chunk of rows would otherwise acquire an
  /// exclusive, process-scoped lock once per row -- the same lock that carries
  /// available(), admitOrWait() and the adaptation callbacks. The capacity
  /// read is a snapshot, which is safe because capacity moves only on
  /// adaptation, far slower than dispatch; the grant itself is a single
  /// compare-exchange, so concurrent callers cannot both take the last slot.
  std::vector<Token> tryAcquireUpTo(int64_t want);

  /// Slots free right now: capacity minus pending, floored at zero.
  /// Admission-controlled dispatch sizes each chunk by this so the cap
  /// actually bounds offered concurrency instead of being overrun by a
  /// whole-vector blast.
  int64_t available() const;

  /// The answer to "can you take work now, and if not, what do I wait on?" --
  /// the same question a Velox operator asks its consumer.
  struct Admission {
    /// The backend has room; go and reserve.
    bool admitted{false};

    /// Valid exactly when 'admitted' is false: resolves when a slot frees.
    ContinueFuture wait;
  };

  /// Decides and enrols under one lock, so a slot freeing cannot slip between
  /// the two. Asking separately -- check capacity, then enrol -- leaves a
  /// window where the check says "free", nothing is enrolled, and the caller
  /// is holding nothing to wait on; falling through to some other wait is how
  /// a driver ends up parked on a future nothing can fulfil.
  Admission admitOrWait();

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

  // Clamps a Config into its valid ranges: floor at least 1, decreaseFactor
  // within (0, 1). Caller holds mutex_.
  static void clampLocked(Config& config);

  // Applies `mutate` to config_, clamps the result, and logs when adaptation
  // flips on or off. Shared by amend() and initializeOnce() so the clamping
  // and the logging live in one place. Caller holds mutex_.
  void applyMutationLocked(const std::function<void(Config&)>& mutate);

  // Moves out up to the currently available headroom's worth of parked
  // waiters, FIFO. The caller fulfils them after dropping the lock, so a
  // waiter's continuation never runs under mutex_.
  void collectWakeableLocked(std::vector<ContinuePromise>& toNotify);

  // Lock-free high-water update for stats.
  void notePeakPending(int64_t pending);

  // Multiplicative decrease: halves capacity by Config::decreaseFactor, down
  // to the floor. A no-op when adaptation is off, and never raises capacity.
  void onOverload();

  // Additive increase: one step per capacity-worth of successful units,
  // floored at +1 so a steady stream always makes progress. Reaching the
  // ceiling clears the adapted value so the ceiling governs again.
  void onSuccess(int64_t units);

  // Called by Token on destruction; releases the slot and hands it to the
  // longest-waiting parked driver, if any.
  void release();

  // Identifies the backend this limiter admits for. Composed by the transport
  // from whatever distinguishes one deployment from another, so two
  // deployments never share a limiter.
  const std::string tierKey_;

  // Guards config_, capacity_, lowWater_ and waiters_. pending_ and
  // peakPending_ are atomic so the hot increment path stays lock-free, but
  // they are read under the lock wherever a capacity comparison depends on
  // them.
  mutable std::mutex mutex_;

  // Mutable rather than a constructor argument: the function's SQL option and
  // the session properties configure a backend in a fixed order, so both need
  // a seam to write through before initialization is sealed.
  Config config_;

  // Set once the first query to reach this backend has finished configuring
  // it. Guards config_ against later queries; the adapted capacity below keeps
  // moving, because that is learned from every query's outcomes.
  bool initialized_{false};

  // Warn once per backend about settings a later query asked for and did not
  // get, rather than once per process.
  bool loggedDivergentSettings_{false};

  // Adapted capacity. Zero means unshrunk, i.e. defer to the ceiling.
  int64_t capacity_{0};

  // Lowest capacity ever reached. Zero means never shrank, and Stats reports
  // it unchanged.
  int64_t lowWater_{0};

  // Units currently in flight against this backend. Atomic so acquire() and
  // release() stay off the lock; read under mutex_ wherever it is compared
  // against capacity.
  std::atomic<int64_t> pending_{0};

  // High-water pending count over the backend's lifetime, for runtime stats.
  // Updated with a relaxed compare-exchange: best effort, not a
  // synchronization point.
  std::atomic<int64_t> peakPending_{0};

  // Drivers parked waiting for a slot, oldest first. Woken FIFO, one per
  // release, so a burst of returning slots does not stampede.
  std::deque<ContinuePromise> waiters_;
};

/// Process-scoped owner of one RPCRateLimiter per backend key.
///
/// A backend is shared across queries by definition, so the registry is a
/// process singleton rather than something threaded through the operator.
class RPCRateLimiterRegistry {
 public:
  static RPCRateLimiterRegistry& global();

  /// Returns the backend's admission control, creating it on first sight. The
  /// reference stays valid for the process lifetime: values are held by
  /// unique_ptr, so later insertions move only the map nodes.
  RPCRateLimiter& get(const std::string& tierKey);

  /// Resets every backend in place and restores process-global defaults.
  /// Resets rather than drops: a Token releases through a back-pointer to its
  /// issuing limiter, so erasing the map would leave outstanding tokens
  /// pointing at freed memory. Intended only
  /// for tests and benchmarks that drive a real operator, which resolves its
  /// backend through this registry and so needs process state reset between
  /// iterations. Never call this from production code.
  void testingReset();

 private:
  // Read-mostly: backends are created once then only looked up, so lookups take
  // a shared lock and do not serialize against each other. Only first sight
  // of a backend takes the exclusive lock.
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<RPCRateLimiter>> backends_;
};

} // namespace facebook::velox::exec::rpc
