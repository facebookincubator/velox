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
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "velox/common/future/VeloxPromise.h"

namespace folly {
class exception_wrapper;
class Timekeeper;
} // namespace folly

namespace facebook::velox::exec::rpc {

/// Admission control for one unit of provisioned capacity: a backend
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
/// Uses one adaptive congestion window measured in admission units per
/// successful service horizon. Integral windows bound in-flight work;
/// fractional windows pace discrete admissions. A separate ceiling limits
/// concurrency even when the learned horizon is short.
class RPCRateLimiter : public std::enable_shared_from_this<RPCRateLimiter> {
  // Restricts owner-backed token construction without using friendship.
  struct TokenConstructionKey {};

 public:
  /// Resolved tuning for one admission key. The operator combines session
  /// overrides with stable function-provided bounds before initialization.
  struct Config {
    /// Whether capacity adapts to overload. When false, capacity is pinned at
    /// the ceiling. Maps to rpc.ratelimiter.adaptive_enabled.
    bool adaptive{false};

    /// Unshrunk capacity. Zero defers to defaultCapacity(), resolved on read
    /// so a later change to the process default still takes effect. Maps to
    /// rpc.ratelimiter.max_limit.
    int64_t ceiling{0};

    /// Lower bound on the adaptive congestion window, in admission units per
    /// successful service horizon. A positive rpc.ratelimiter.min_limit
    /// overrides the function-provided value.
    double floor{0.01};

    /// Multiplicative-decrease factor applied on overload, in (0, 1). Maps to
    /// rpc.ratelimiter.decrease_factor.
    double decreaseFactor{0.5};
  };

  /// What completed work told us about the backend. The admission layer's own
  /// vocabulary keeps this target free of expression-layer dependencies.
  enum class Outcome {
    /// The backend served the work. Doubles a fractional window until one,
    /// then drives additive increase.
    kSuccess,
    /// The backend shed load. Drives multiplicative decrease.
    kOverload,
    /// Nothing to learn from; capacity is left alone.
    kNone,
  };

  /// A snapshot for per-query runtime stats. Read at operator close() for the
  /// final stats, and on every RPCOperator::stats() call so the admission
  /// capacity trajectory is observable while the query is still running.
  struct Stats {
    /// Integral in-flight capacity. This remains at least one while a
    /// fractional limit is enforced through paced admission.
    int64_t capacity{0};

    /// Hard upper bound on the adaptive congestion window.
    int64_t hardLimit{0};

    /// In-flight requests right now.
    int64_t pending{0};

    /// High-water in-flight count over the backend's lifetime.
    int64_t peakPending{0};

    /// Lowest integral capacity ever reached. In adaptive mode this includes
    /// the conservative initial window; in fixed mode it is the ceiling. Use
    /// lowWaterLimitMilli to observe a sub-unit low-water mark.
    ///
    /// Scoped to the backend's lifetime in this process, not to one query:
    /// the mark is never cleared on recovery, so a query that saw no overload
    /// still reports a dip an earlier query produced. It also includes the
    /// conservative initial window, so it does not by itself prove overload.
    int64_t lowWaterCapacity{0};

    /// Current adaptive congestion window in milli-admission-units.
    int64_t limitMilli{0};

    /// Lowest adaptive congestion window reached, in milli-admission-units.
    int64_t lowWaterLimitMilli{0};

    /// Smoothed successful service horizon used to pace fractional windows.
    int64_t serviceHorizonNanos{0};
  };

  /// One in-flight request slot, released on destruction so a slot is returned
  /// even when the RPC future is abandoned (e.g. query cancellation).
  ///
  /// Holds a back-pointer to its issuing RPCRateLimiter. Production tokens are
  /// grouped in an AdmissionLease, which keeps the limiter alive. A locally
  /// constructed limiter must outlive any bare token taken from it.
  class Token {
   public:
    Token() = default;

    Token(Token&& other) noexcept
        : owner_{other.owner_}, overloadEpoch_{other.overloadEpoch_} {
      other.owner_ = nullptr;
    }

    Token& operator=(Token&& other) noexcept;

    ~Token() noexcept;

    Token(const Token&) = delete;
    Token& operator=(const Token&) = delete;

    /// Takes ownership of one slot already claimed on 'owner'. The passkey
    /// restricts direct construction to RPCRateLimiter.
    Token(TokenConstructionKey&, RPCRateLimiter* owner, uint64_t overloadEpoch)
        : owner_{owner}, overloadEpoch_{overloadEpoch} {}

    /// Admission epoch captured with this token's reservation.
    uint64_t overloadEpoch() const {
      return overloadEpoch_;
    }

    /// True when this token was issued by `owner`.
    bool belongsTo(const RPCRateLimiter* owner) const {
      return owner_ == owner;
    }

   private:
    // Null means moved-from or default-constructed: no slot is held.
    RPCRateLimiter* owner_{nullptr};
    uint64_t overloadEpoch_{0};
  };

  /// Owns one dispatch's admission tokens until the RPC completes or its
  /// query closes. Completion may apply canonical overload feedback before
  /// atomically releasing every token; cancellation releases without
  /// feedback. Both operations are idempotent and thread-safe.
  class AdmissionLease {
   public:
    struct Completion {
      /// True when canonical shared overload was handled or coalesced.
      bool sharedOverloadHandled{false};
      /// Admission epoch shared by every token in this lease.
      uint64_t overloadEpoch{0};
    };

    AdmissionLease(
        std::shared_ptr<RPCRateLimiter> owner,
        std::vector<Token> tokens,
        uint64_t overloadEpoch);
    AdmissionLease(const AdmissionLease&) = delete;
    AdmissionLease& operator=(const AdmissionLease&) = delete;
    AdmissionLease(AdmissionLease&&) = delete;
    AdmissionLease& operator=(AdmissionLease&&) = delete;
    ~AdmissionLease();

    /// Applies canonical typed-overload feedback, when requested, before any
    /// token destruction can wake waiters. Canonical overloads from the same
    /// admission epoch are coalesced into one multiplicative decrease.
    Completion completeAndRelease(bool hardOverload) noexcept;

    /// Releases without feedback, as used when the owning query closes.
    void release() noexcept;

   private:
    std::mutex mutex_;
    std::shared_ptr<RPCRateLimiter> owner_;
    std::vector<Token> tokens_;
    const uint64_t overloadEpoch_;
    bool completed_{false};
  };

  explicit RPCRateLimiter(std::string admissionKey);

  /// Constructs a limiter with an injected clock and timer for deterministic
  /// tests. Store it in a shared_ptr before enabling pacing; timer callbacks
  /// use weak ownership. The injected clock and timer must outlive it.
  RPCRateLimiter(
      std::string admissionKey,
      std::function<std::chrono::steady_clock::time_point()> now,
      folly::Timekeeper* timekeeper);

  /// Applies already-resolved limiter tuning.
  /// Never touches work already in flight — a ceiling that lands below the
  /// current pending count stops admission but cancels nothing and wakes no
  /// waiter, because nothing was freed.
  void configure(const Config& config);

  /// Applies 'mutate' to this backend's tuning under the lock, so a
  /// read-modify-write cannot lose concurrent changes. Prefer this over
  /// config() + configure() whenever only some fields change.
  void amend(const std::function<void(Config&)>& mutate);

  /// Applies `mutate` on the first call for this backend and seals the
  /// configuration; later calls are refused.
  /// A backend's configuration is shared by every query running against it and
  /// has to hold still while the limiter adapts, so a later query's settings
  /// are ignored rather than allowed to move the target mid-flight. A
  /// divergent request is logged once so the no-op is visible.
  void initializeOnce(const std::function<void(Config&)>& mutate);

  /// The resolved tuning currently in effect.
  Config config() const;

  /// Groups pre-reserved tokens into an idempotent completion lease.
  std::shared_ptr<AdmissionLease> makeLease(std::vector<Token> tokens);

  /// Groups one pre-reserved token into an idempotent completion lease.
  std::shared_ptr<AdmissionLease> makeLease(Token token);

  /// Claims a slot unconditionally for deterministic tests. Production
  /// dispatch uses tryAcquireUpTo() so capacity and pacing are enforced.
  Token acquire();

  /// Claims up to 'want' slots in one pass and returns the tokens granted,
  /// which may be fewer than asked for and may be none.
  ///
  /// Takes the backend's lock at most once for the whole grant rather than
  /// once per slot. Integral admission uses an atomic reservation; fractional
  /// admission is serialized with the pacing deadline under the lock.
  std::vector<Token> tryAcquireUpTo(int64_t want);

  /// Slots available now. Integral mode returns capacity minus pending,
  /// floored at zero; fractional mode returns one only when no unit is pending
  /// and the pacing deadline has arrived.
  int64_t available() const;

  /// The answer to "can you take work now, and if not, what do I wait on?" --
  /// the same question a Velox operator asks its consumer.
  struct Admission {
    /// The backend has room; go and reserve.
    bool admitted{false};

    /// Valid exactly when 'admitted' is false: resolves when admission may be
    /// retried.
    ContinueFuture wait;
  };

  /// Decides and enrols under one lock, so a slot freeing cannot slip between
  /// the two. Asking separately -- check capacity, then enrol -- leaves a
  /// window where the check says "free", nothing is enrolled, and the caller
  /// is holding nothing to wait on; falling through to some other wait is how
  /// a driver ends up parked on a future nothing can fulfil.
  Admission admitOrWait();

  /// Feeds completion feedback without a service-horizon sample. Intended for
  /// tests and outcomes that cannot carry one.
  void onOutcome(Outcome outcome, int64_t units);

  /// Feeds completion feedback back into capacity. `units` is the
  /// number of successful items the batch represents and is read only for
  /// Outcome::kSuccess. Recovery doubles a fractional limit until one, then
  /// grows additively. `rttNs` is a successful service-horizon sample; other
  /// outcomes ignore both values.
  void onOutcome(Outcome outcome, int64_t units, int64_t rttNs);

  /// Observes a successful service-horizon sample, but applies window recovery
  /// only if `overloadEpoch` is still current. This prevents late successes
  /// admitted before an overload from immediately undoing its multiplicative
  /// decrease without discarding their valid service-duration evidence.
  void onSuccessForEpoch(uint64_t overloadEpoch, int64_t units, int64_t rttNs);

  Stats stats() const;

  /// Fallback ceiling for backends configured with ceiling == 0.
  static void setDefaultCapacity(int64_t capacity);
  static int64_t defaultCapacity();

  /// Restores this backend to its as-constructed state. Intended only for
  /// tests. Resets in place rather than being destroyed and recreated, because
  /// outstanding tokens hold a back-pointer here and release through it.
  void testingReset();

 private:
  struct PacingTimerRequest {
    std::chrono::microseconds delay;
    uint64_t generation;
    std::weak_ptr<RPCRateLimiter> owner;
  };

  // Integral in-flight capacity under the current adaptive state. Caller holds
  // mutex_.
  int64_t capacityLocked() const;

  // Current adaptive congestion window, with zero state resolved to the
  // configured ceiling. Caller holds mutex_.
  double limitLocked() const;

  // Ceiling with Config::ceiling == 0 resolved against the process default.
  // Caller holds mutex_.
  int64_t ceilingLocked() const;

  // Config::floor clamped to at most the ceiling. Caller holds mutex_.
  double floorLocked(double ceiling) const;

  // True while overload has reduced the congestion window below one unit.
  // Caller holds mutex_.
  bool isPacingLocked() const;

  // Interval between paced admission starts. Caller holds mutex_.
  std::chrono::steady_clock::duration pacingIntervalLocked() const;

  // Changes the adaptive window and reschedules fractional pacing. Zero means
  // the configured ceiling governs. Caller holds mutex_.
  void setLimitLocked(double limit);

  // Incorporates one successful completion into the service-horizon EWMA.
  // Caller holds mutex_.
  bool updateServiceHorizonLocked(int64_t rttNs);

  // Clamps a Config into its valid ranges. Caller holds mutex_.
  static void clampLocked(Config& config);

  // Applies `mutate` to config_, clamps the result, and logs when adaptation
  // flips on or off. Shared by amend() and initializeOnce() so the clamping
  // and the logging live in one place. Caller holds mutex_.
  void applyMutationLocked(const std::function<void(Config&)>& mutate);

  // Moves out waiters that may retry admission. Both modes wake every waiter
  // once admission is possible, because a notification carries no reservation
  // and an abandoned future must not strand the queue. Atomic acquisition
  // selects the winners. The caller fulfils them after dropping the lock.
  std::optional<PacingTimerRequest> collectWakeableLocked(
      std::vector<ContinuePromise>& toNotify);

  // Reserves the single timer for nextPacedAdmission_ without calling the
  // external Timekeeper. Caller holds mutex_.
  std::optional<PacingTimerRequest> preparePacingTimerLocked();

  // Schedules a timer previously reserved under mutex_. Never called while
  // holding mutex_, because an injected Timekeeper may run callbacks inline.
  void schedulePacingTimer(const PacingTimerRequest& request) noexcept;

  // Handles one timer generation and wakes eligible paced waiters.
  void onPacingTimer(uint64_t generation);

  // Fails waiters when the current timer cannot be scheduled or completed.
  void onPacingTimerFailure(
      uint64_t generation,
      const folly::exception_wrapper& error) noexcept;

  // Lock-free high-water update for stats.
  void notePeakPending(int64_t pending);

  // Multiplicative decrease: halves capacity by Config::decreaseFactor, down
  // to the floor. A no-op when adaptation is off, and never raises capacity.
  void onOverload();

  // Handles canonical overload once per admission epoch. Returns true when
  // handled or coalesced, so the driver never repeats shared feedback.
  bool handleOverloadForEpoch(uint64_t overloadEpoch);

  // Applies one decrease with mutex_ held.
  void applyOverloadLocked();

  // Applies success recovery and learns the successful service horizon.
  void onSuccess(int64_t units, int64_t rttNs);

  // Applies success recovery with mutex_ held.
  void applySuccessLocked(int64_t units, int64_t rttNs);

  // Called by Token on destruction; releases the slot and hands it to the
  // longest-waiting parked driver, if any.
  void release() noexcept;

  // Identifies the backend this limiter admits for. Composed by the transport
  // from whatever distinguishes one deployment from another, so two
  // deployments never share a limiter.
  const std::string admissionKey_;

  // Supplies monotonic time. Injected together with timekeeper_ in tests so
  // pacing can advance without wall-clock sleeps.
  const std::function<std::chrono::steady_clock::time_point()> now_;

  // Schedules paced waits. Null selects Folly's process-global timekeeper.
  folly::Timekeeper* const timekeeper_;

  // Guards config_, limit_, lowWaterLimit_ and waiters_. pending_ and
  // peakPending_ are atomic for the integral reservation and stats paths, but
  // are read under the lock wherever pacing depends on them.
  mutable std::mutex mutex_;

  // Mutable so initialization can install one resolved policy atomically and
  // tests can exercise controlled policy changes.
  Config config_;

  // Set once the first query to reach this backend has finished configuring
  // it. Guards config_ against later queries; the adapted capacity below keeps
  // moving, because that is learned from every query's outcomes.
  bool initialized_{false};

  // Warn once per backend about settings a later query asked for and did not
  // get, rather than once per process.
  bool loggedDivergentSettings_{false};

  // Adaptive congestion window in admission units per service horizon. Zero
  // means the hard ceiling governs.
  double limit_{0.0};

  // Lowest initialized adaptive limit. Zero means no adaptive low-water mark
  // has been recorded.
  double lowWaterLimit_{0.0};

  // Successful service-duration EWMA used as the pacing horizon. Starts at a
  // conservative internal bootstrap until the first successful sample.
  int64_t serviceHorizonNanos_{1'000'000'000};
  bool hasServiceHorizonSample_{false};

  // Earliest start time for the next paced admission. Each admission advances
  // this from the current time, so idle time never accumulates a burst.
  std::chrono::steady_clock::time_point nextPacedAdmission_;

  // Most recent admission start, used to preserve start-to-start spacing when
  // feedback changes the pacing rate after an RPC completes.
  std::optional<std::chrono::steady_clock::time_point> lastAdmission_;

  // Invalidates timers scheduled against an earlier rate or deadline.
  uint64_t pacingGeneration_{0};

  // Identifies the current offered-load wave. The first canonical overload
  // from a wave advances this before shrinking, coalescing later responses
  // from the same wave.
  uint64_t overloadEpoch_{0};

  // True while the current generation has one reserved or scheduled timer.
  bool pacingTimerArmed_{false};

  // Units currently in flight against this backend. Integral reservations and
  // release use atomic updates; fractional admission compares it under mutex_.
  std::atomic<int64_t> pending_{0};

  // High-water pending count over the backend's lifetime, for runtime stats.
  // Updated with a relaxed compare-exchange: best effort, not a
  // synchronization point.
  std::atomic<int64_t> peakPending_{0};

  // Drivers parked waiting for a slot. Woken together when admission becomes
  // possible; each retries the atomic reservation.
  std::deque<ContinuePromise> waiters_;

  // Prevents callers from constructing owner-backed tokens directly.
  [[no_unique_address]] TokenConstructionKey tokenConstructionKey_;
};

/// Owns one process-scoped RPCRateLimiter per admission key.
///
/// A backend is shared across queries by definition, so the registry is a
/// process singleton rather than something threaded through the operator.
class RPCRateLimiterRegistry {
 public:
  static RPCRateLimiterRegistry& global();

  /// Returns the backend's admission control, creating it on first sight. The
  /// reference stays valid for the process lifetime: values are shared-owned,
  /// and later insertions do not replace an existing limiter.
  RPCRateLimiter& get(const std::string& admissionKey);

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
  std::unordered_map<std::string, std::shared_ptr<RPCRateLimiter>> backends_;
};

} // namespace facebook::velox::exec::rpc
