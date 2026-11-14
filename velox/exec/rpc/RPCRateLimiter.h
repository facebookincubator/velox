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
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <folly/container/F14Map.h>
#include <folly/container/small_vector.h>

#include "velox/common/future/VeloxPromise.h"
#include "velox/exec/rpc/RPCFlowControlPolicy.h"

namespace folly {
class exception_wrapper;
class Timekeeper;
} // namespace folly

namespace facebook::velox::exec::rpc {

/// Admission control for one unit of provisioned capacity: a backend
/// plus the credential used to reach it. Every driver and query that shares
/// that pair and runs the same policy -- in both streaming modes -- draws on
/// one limiter. Queries that run different policies against the same pair get
/// one limiter per policy, so together they can admit the sum of their
/// ceilings.
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
/// fractional windows pace discrete admissions. While calls keep failing, a
/// fractional window backs off until probes are a maximum interval apart.
/// After the first success it doubles every service horizon, so recovery takes
/// a few horizons however deep the back-off went. A separate ceiling limits
/// concurrency even when the learned horizon is short.
class RPCRateLimiter : public std::enable_shared_from_this<RPCRateLimiter> {
  // Restricts owner-backed token construction without using friendship.
  struct TokenConstructionKey {};
  // Whether a token is plain or tracked by SMART_PROBING state. Fixed at
  // admission, so a later policy change cannot reroute its release.
  enum class AdmissionKind : uint8_t { kPlain, kProbing };
  struct ProbingCredit;
  using ProbingCreditCells =
      folly::small_vector<std::shared_ptr<ProbingCredit>, 1>;

 public:
  /// Resolved tuning a limiter runs: one query's rate-limiter policy. The
  /// operator combines session overrides with stable function-provided bounds
  /// before initialization.
  struct Config {
    /// Whether capacity adapts to overload. When false, capacity is pinned at
    /// the ceiling. Maps to rpc.ratelimiter.adaptive_enabled.
    bool adaptive{false};

    /// Unshrunk capacity. Zero defers to defaultCapacity(), resolved on read
    /// so a later change to the process default still takes effect. Maps to
    /// rpc.ratelimiter.max_limit.
    int64_t ceiling{0};

    /// Lower bound on the adaptive congestion window, in admission units per
    /// successful service horizon. Resolved from the function's floor and
    /// rpc.ratelimiter.min_limit; rpc.ratelimiter.hard_limit selects which
    /// wins.
    double floor{0.01};

    /// Window multiplicative-decrease factor on overload, in (0, 1). Maps to
    /// rpc.ratelimiter.decrease_factor.
    double decreaseFactor{0.5};

    /// Longest gap between paced probes. It bounds both the calls wasted while
    /// a backend rejects everything and how long the limiter takes to notice
    /// that it recovered. Zero leaves the floor alone, so probes can be as far
    /// apart as the floor allows. Maps to
    /// rpc.ratelimiter.max_probe_interval_ms.
    std::chrono::milliseconds maxProbeInterval{60'000};

    /// Effective admission-window policy after the operator resolves the
    /// request against streaming mode, adaptation and retry capability.
    /// Non-adaptive limiters always run WINDOW_AIMD.
    RPCFlowControlPolicy flowControlPolicy{RPCFlowControlPolicy::kWindowAimd};

    /// Field-wise equality. Limiters compare policies after reducing them to
    /// the fields a limiter reads; see initializeOrMatch().
    bool operator==(const Config& other) const = default;
  };

  /// What completed work told us about the backend. The admission layer's own
  /// vocabulary keeps this target free of expression-layer dependencies.
  enum class Outcome {
    /// The backend served the work. Drives additive increase of an integral
    /// window, and starts time-based recovery of a fractional one.
    kSuccess,
    /// The backend shed load. Drives multiplicative decrease.
    kOverload,
    /// The backend failed the work for a reason other than load. Leaves the
    /// window alone but ends time-based recovery of a fractional one, which
    /// resumes only after the next success.
    kFailure,
    /// Nothing to learn from; capacity is left alone.
    kNone,
  };

  /// A snapshot for per-query runtime stats. Read at operator close() for the
  /// final stats, and on every RPCOperator::stats() call so the admission
  /// capacity trajectory is observable while the query is still running.
  /// Scaled nonnegative gauges saturate at INT64_MAX when their units cannot
  /// represent the control value. Integral capacity and hardLimit stay exact.
  struct Stats {
    /// Integral in-flight capacity. This remains at least one while a
    /// fractional limit is enforced through paced admission.
    int64_t capacity{0};

    /// Hard upper bound on the adaptive congestion window.
    int64_t hardLimit{0};

    /// In-flight requests right now.
    int64_t pending{0};

    /// High-water in-flight count over this limiter's lifetime.
    int64_t peakPending{0};

    /// Lowest integral capacity ever reached. In adaptive mode this includes
    /// the conservative initial window; in fixed mode it is the ceiling. Use
    /// lowWaterLimitMilli to observe a sub-unit low-water mark.
    ///
    /// Scoped to this limiter's lifetime in this process, not to one query:
    /// the mark is never cleared on recovery, so a query that saw no overload
    /// still reports a dip an earlier query sharing the limiter produced. It
    /// also includes the conservative initial window, so it does not by itself
    /// prove overload.
    int64_t lowWaterCapacity{0};

    /// Current adaptive congestion window in milli-admission-units.
    int64_t limitMilli{0};

    /// Lowest adaptive congestion window reached, in milli-admission-units.
    int64_t lowWaterLimitMilli{0};

    /// Smoothed successful service horizon used to pace fractional windows.
    int64_t serviceHorizonNanos{0};

    /// True while a paced window grows with time after a success.
    bool recovering{false};

    /// Time the window has spent below one admission unit, where admission is
    /// paced, including the stretch in progress.
    ///
    /// Like lowWaterCapacity, scoped to the backend's lifetime in this
    /// process, not to one query. The difference between two snapshots is the
    /// pacing time between them.
    int64_t pacingNanos{0};

    /// Time integral of the committed adaptive congestion window, the one
    /// limitMilli shows, in admission-unit seconds, scoped to the backend's
    /// lifetime like pacingNanos. Recovery growth counts once the limiter
    /// commits it to the window. The difference between two snapshots,
    /// divided by the time between them, is the time-weighted average window
    /// over that interval.
    double windowUnitSeconds{0.0};

    /// The limiter's clock at this snapshot, for timing the interval between
    /// two snapshots.
    int64_t clockNanos{0};

    /// Current registered admission waits and whether their timer is armed.
    int64_t admissionWaiters{0};
    bool admissionTimerArmed{false};

    /// Remaining time until the next pacing permit, zero when already due.
    int64_t nextPermitDelayNanos{0};

    /// Effective admission-window policy.
    RPCFlowControlPolicy flowControlPolicy{RPCFlowControlPolicy::kWindowAimd};
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
        : owner_{other.owner_},
          overloadEpoch_{other.overloadEpoch_},
          admissionId_{other.admissionId_},
          kind_{other.kind_} {
      other.owner_ = nullptr;
    }

    Token& operator=(Token&& other) noexcept;

    ~Token() noexcept;

    Token(const Token&) = delete;
    Token& operator=(const Token&) = delete;

    /// Takes ownership of one slot already claimed on 'owner'. The passkey
    /// restricts direct construction to RPCRateLimiter.
    Token(
        TokenConstructionKey&,
        RPCRateLimiter* owner,
        uint64_t overloadEpoch,
        uint64_t admissionId,
        AdmissionKind kind)
        : owner_{owner},
          overloadEpoch_{overloadEpoch},
          admissionId_{admissionId},
          kind_{kind} {}

    /// Admission epoch captured with this token's reservation.
    uint64_t overloadEpoch() const {
      return overloadEpoch_;
    }

    /// Correlates lease feedback with the issuing controller's held record.
    uint64_t admissionId() const {
      return admissionId_;
    }

    /// True when this token was issued by `owner`.
    bool belongsTo(const RPCRateLimiter* owner) const {
      return owner_ == owner;
    }

    /// True for a token tracked by SMART_PROBING probing state.
    bool isProbing() const {
      return owner_ && kind_ == AdmissionKind::kProbing;
    }

    /// Detaches ownership only after the limiter settles its physical charge.
    void disarm(TokenConstructionKey&) noexcept {
      owner_ = nullptr;
    }

    /// Matches internal provenance without exposing controller metadata.
    bool belongsTo(const RPCRateLimiter* owner, AdmissionKind kind) const {
      return owner_ == owner && kind_ == kind;
    }

   private:
    // Null means moved-from or default-constructed: no slot is held.
    RPCRateLimiter* owner_{nullptr};
    uint64_t overloadEpoch_{0};
    // Carries only opaque identity; the controller owns full held metadata.
    uint64_t admissionId_{0};
    // Survives moves and policy changes until physical release.
    AdmissionKind kind_{AdmissionKind::kPlain};
  };

  /// How an overload reached the window: kCanonical cut it,
  /// kExclusivelySpeculative was charged to a probe, and kStale came from an
  /// older epoch.
  enum class FeedbackDisposition {
    kCanonical,
    kExclusivelySpeculative,
    kStale,
  };

  /// Combines the dispositions of feedback applied together: kCanonical wins
  /// over kExclusivelySpeculative, which wins over kStale.
  static FeedbackDisposition strongerDisposition(
      FeedbackDisposition lhs,
      FeedbackDisposition rhs);

  /// Carries immutable admission credit to the actual output crossing.
  class OutputCredit {
   public:
    OutputCredit() = default;
    ~OutputCredit() = default;
    OutputCredit(OutputCredit&&) noexcept = default;
    OutputCredit& operator=(OutputCredit&&) noexcept = default;
    OutputCredit(const OutputCredit&) = delete;
    OutputCredit& operator=(const OutputCredit&) = delete;
    OutputCredit(
        TokenConstructionKey&,
        std::shared_ptr<RPCRateLimiter> owner,
        ProbingCreditCells cells);
    /// Consumes each owned unit at most once, including work from older epochs.
    void consume() noexcept;
    explicit operator bool() const {
      return !cells_.empty();
    }
    /// Exposes cells only to the issuing limiter's internal passkey.
    const auto& cells(TokenConstructionKey&) const {
      return cells_;
    }

   private:
    // Keeps the lifecycle that issued these cells alive until output or drop.
    std::shared_ptr<RPCRateLimiter> owner_;
    // Bounds identity bookkeeping by held attempts and buffered output.
    ProbingCreditCells cells_;
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
      /// Prevents exclusively marginal errors from cutting driver
      /// authorization.
      FeedbackDisposition disposition{FeedbackDisposition::kCanonical};
      /// Drops safely when response or ready-queue allocation fails.
      OutputCredit credit{};
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

    /// Records one completion before releasing admission. 'handedOff' marks a
    /// rejection the operator retries, whose duration is not a service sample.
    Completion completeAndRelease(Outcome outcome, bool handedOff) noexcept;

    /// Binds client ownership before attaching an eager raw-future
    /// continuation.
    void bindRawAttempt() noexcept;
    /// Settles actual raw work independently of timeout and operator teardown.
    void settleRawAttempt(Outcome outcome, bool validServiceProof) noexcept;
    /// True when the lease holds SMART_PROBING tokens, which settle when the
    /// raw attempt completes.
    bool isProbing() const {
      return probingLease_;
    }

    /// Marks an overload of this attempt as not named a lack of serving
    /// capacity: a rate quota, an unnamed cause, or no retry hint. Such a
    /// rejection says nothing about the window, so SMART_PROBING answers it
    /// with WINDOW_AIMD's cut and never charges it to a probe. Must precede the
    /// attempt's feedback.
    void markNonCapacityRejection() noexcept;

    /// Releases without feedback, as used when the owning query closes.
    void release() noexcept;

   private:
    std::mutex mutex_;
    std::shared_ptr<RPCRateLimiter> owner_;
    std::vector<Token> tokens_;
    const uint64_t overloadEpoch_;
    bool completed_{false};
    // Set by markNonCapacityRejection().
    bool nonCapacityRejection_{false};
    // Fixes WINDOW ownership independently of later runtime configuration.
    const bool probingLease_;
    // Distinguishes undispatched rollback from unresolved client work.
    bool rawBound_{false};
    // Detached tokens remain unresolved until their verdict is published.
    bool rawSettled_{false};
    // Withdraws all credit and proof promptly on timeout or close.
    bool cancelled_{false};
    // Transfers the raw settlement verdict once to the ready-output path.
    Completion rawCompletion_;
    // Preallocates output ownership before dispatch starts.
    OutputCredit probingCredit_;
    // Retains cancellation cells even after raw settlement detaches output.
    ProbingCreditCells cancellationCells_;
  };

  explicit RPCRateLimiter(std::string admissionKey);

  /// Constructs a limiter with an injected clock, timer and random source for
  /// deterministic tests. 'random' returns a value in [0, 1] and spreads the
  /// back-off probes of limiters that were cut at the same moment. Store the
  /// limiter in a shared_ptr before enabling pacing; timer callbacks use weak
  /// ownership. The injected clock and timer must outlive it.
  RPCRateLimiter(
      std::string admissionKey,
      std::function<std::chrono::steady_clock::time_point()> now,
      folly::Timekeeper* timekeeper,
      std::function<double()> random);

  /// Applies already-resolved limiter tuning.
  /// Never touches work already in flight — a ceiling that lands below the
  /// current pending count stops admission but cancels nothing and wakes no
  /// waiter, because nothing was freed. On a registry limiter, call it only
  /// before initializeOrMatch() seals a policy: the registry finds a limiter
  /// by the policy it sealed.
  void configure(const Config& config);

  /// Applies 'mutate' to this limiter's tuning under the lock, so a
  /// read-modify-write cannot lose concurrent changes. Prefer this over
  /// config() + configure() whenever only some fields change. On a registry
  /// limiter, call it only before initializeOrMatch() seals a policy.
  void amend(const std::function<void(Config&)>& mutate);

  /// Seals `policy` when no policy is sealed yet, and otherwise leaves the
  /// sealed policy as it is. Returns true when this call sealed `policy` or the
  /// sealed policy matches it, and false otherwise.
  ///
  /// Two policies match when they agree after clamping, ignoring what the
  /// limiter never reads: fixed capacity reads neither the floor nor the
  /// decrease factor, and a floor at or above a positive ceiling pins capacity
  /// at the ceiling, as fixed capacity does. Limiters sealed with matching
  /// policies admit alike.
  bool initializeOrMatch(const Config& policy);

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

  /// True when callers must retain eligible demand ownership across a wake
  /// until dispatch replaces it or eligible work ends.
  bool usesAttributedAdmissionControl() const;

  /// Owns producer runnable intent independently of waits and current headroom.
  class RunnableDemand {
   public:
    RunnableDemand(std::weak_ptr<RPCRateLimiter> owner, uint64_t id);
    ~RunnableDemand();
    RunnableDemand(const RunnableDemand&) = delete;
    RunnableDemand& operator=(const RunnableDemand&) = delete;
    RunnableDemand(RunnableDemand&&) = delete;
    RunnableDemand& operator=(RunnableDemand&&) = delete;

   private:
    // Does not keep an abandoned producer's limiter alive.
    std::weak_ptr<RPCRateLimiter> owner_;
    // Identifies only this producer's registration.
    const uint64_t id_;
  };

  /// Registers runnable candidate work; returns null for other policies.
  std::shared_ptr<RunnableDemand> registerRunnableDemand();

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

  /// Applies an overload that no completion attributed to an attempt, such as
  /// a function's aggregate congestion verdict. It names no capacity limit, so
  /// SMART_PROBING answers it with WINDOW_AIMD's cut and learns nothing.
  void onUnattributedOverload();

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
  /// Returns false when `units` would have earned recovery credit and were
  /// discarded because `overloadEpoch` is no longer current, true otherwise.
  bool onSuccessForEpoch(uint64_t overloadEpoch, int64_t units, int64_t rttNs);

  /// Whether the limiter can still back off for an operator retry: adaptive,
  /// and either pacing or above its floor.
  bool canHandleRetryBackoff() const;

  /// Observes SMART_PROBING state without changing it. Intended only for
  /// tests.
  struct TestingProbingSnapshot {
    /// True once a canonical overload ended the WINDOW-identical phase.
    bool cut{};
    /// Integral allowance in force.
    int64_t baseline{};
    /// Baseline successes still required before the next probe.
    int64_t retryDebt{};
    /// Remembered failures at the next probe level.
    int64_t nextLevelFailures{};
    /// Charged admissions awaiting physical settlement.
    size_t held{};
    /// Probes among them: zero or one.
    size_t heldProbes{};
  };
  TestingProbingSnapshot testingProbingSnapshot() const;

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

  // Config::floor, raised so paced probes are at most the maximum probe
  // interval apart (up to a floor of 0.5, which keeps pacing on), and clamped
  // to at most the ceiling. Caller holds mutex_.
  double floorLocked(double ceiling) const;

  // True while overload has reduced the congestion window below one unit.
  // Caller holds mutex_.
  bool isPacingLocked() const;

  // Interval between paced admission starts. While recovering, it is the time
  // for the window, growing as grownLimitLocked() describes, to accumulate one
  // admission. Caller holds mutex_.
  std::chrono::steady_clock::duration pacingIntervalLocked() const;

  // Wait before the next paced admission: pacingIntervalLocked(), shortened by
  // a random fraction of up to kProbeJitter while backing off, so limiters cut
  // at the same moment do not probe in lockstep. Caller holds mutex_.
  std::chrono::steady_clock::duration nextPacedDelayLocked() const;

  // The fractional window at 'now'. While recovering, it doubles once per
  // service horizon since limitChangedAt_, capped at one; otherwise it is
  // limitLocked(). Caller holds mutex_.
  double grownLimitLocked(std::chrono::steady_clock::time_point now) const;

  // Commits recovery growth, so a cut or a schedule change starts from the
  // grown window. Caller holds mutex_.
  void commitGrowthLocked();

  // Adds the time since the last accrual to pacingTime_ and
  // windowUnitSeconds_ at the window that held over it, so it must run before
  // anything that changes limit_, config_ or the service horizon, which sets
  // the probe floor. Recovery growth counts once committed. Caller holds
  // mutex_.
  void accrueLocked();

  // Changes the adaptive window and reschedules fractional pacing. Zero means
  // the configured ceiling governs. Caller holds mutex_.
  void setLimitLocked(double limit);

  // Incorporates one successful completion into the service-horizon EWMA.
  // Caller holds mutex_.
  bool updateServiceHorizonLocked(int64_t rttNs);

  // Applies `mutate` to config_, clamps the result, and logs when adaptation
  // flips on or off. Shared by amend() and initializeOrMatch() so the clamping
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

  // Advances the overload epoch and applies one decrease. 'nonCapacity' as for
  // applyOverloadLocked().
  void applyOverload(bool nonCapacity);

  // Applies a canonical overload once per admission epoch: the first overload
  // from an epoch advances it and decreases the window; later ones coalesce.
  // Returns true, since the limiter handled the overload either way.
  bool handleOverloadForEpoch(uint64_t overloadEpoch, bool nonCapacity);

  // Lowers the limit to 'target', never raising it and never below the floor.
  // Caller holds mutex_.
  void decreaseLimitLocked(double target);

  // Applies one decrease with mutex_ held. 'nonCapacity' marks a rejection not
  // shown to be a lack of capacity, which gets WINDOW's cut, records no level
  // failure, and leaves SMART_PROBING's window state untouched before its
  // first cut.
  void applyOverloadLocked(bool nonCapacity);

  // Ends time-based recovery after a non-overload failure.
  void onFailure();

  // Applies success recovery and learns the successful service horizon.
  void onSuccess(int64_t units, int64_t rttNs);

  // Applies success recovery with mutex_ held.
  // 'attributed' marks a success from SMART_PROBING output credit, which alone
  // grows the window once probing has started.
  void applySuccessLocked(int64_t units, int64_t rttNs, bool attributed);

  // True when a success would grow the window: adaptation is on and the window
  // is below the ceiling. Only the stale-success verdict uses it: a success at
  // the ceiling still runs applySuccessLocked(), which hands the window back to
  // the ceiling without growing it. Caller holds mutex_.
  bool successEarnsCreditLocked() const;

  // Called by Token on destruction; releases the slot and wakes parked drivers
  // that may now be admitted.
  void release(uint64_t admissionId, AdmissionKind kind) noexcept;

  // Bounds startup dispatch granularity until WINDOW learns a service sample.
  // Caller holds mutex_.
  bool usesProbingStartupLocked() const;

  // Records the reservation's captured population before releases can lower it.
  // Caller holds mutex_.
  void noteStartupPeakLocked(int64_t pending);

  // One charged SMART_PROBING admission, from reservation until its output is
  // consumed or dropped.
  struct ProbingCredit {
    // Never reused, so late feedback cannot match a newer admission.
    uint64_t admissionId{};
    // True for the one exploratory unit above the baseline allowance.
    bool probe{false};
    // In-flight units, this one included, when it was admitted. A probe
    // promotes only if this is still the next level.
    int64_t level{};
    // Starts the service sample measured at settlement.
    std::chrono::steady_clock::time_point admittedAt{};
    // Rejects output credit from a previous controller lifecycle.
    uint64_t lifecycle{};
    // Enforces exactly-once output credit consumption.
    std::atomic<bool> consumed{false};
    // Published before the limiter lock is taken, so a settling attempt sees
    // cancellation that raced with it.
    std::atomic<bool> cancelled{false};
    // Records successful physical completion independently of output.
    bool successful{false};
    // Requires authoritative physical release before output credit.
    bool released{false};
  };
  // Failures remembered for one level.
  struct LevelFailures {
    int64_t count{0};
    // Baseline successes at or above the level since its last failure or
    // forgiveness.
    int64_t cleanSuccesses{0};
  };
  // SMART_PROBING state. Before the first canonical overload the policy admits
  // exactly as WINDOW does. Afterwards the baseline allowance only grows by
  // probes: one exploratory unit admitted when the baseline is full.
  struct ProbingState {
    // Charged admissions awaiting physical settlement.
    folly::F14FastMap<uint64_t, std::shared_ptr<ProbingCredit>> receipts;
    // Keeps runnable producer ownership distinct from parked waits.
    std::vector<uint64_t> demands;
    // Allocates producer identities without reusing cancelled registrations.
    uint64_t nextDemandId{0};
    // Allocates admission identities; never reset.
    uint64_t nextAdmissionId{0};
    // Invalidates output credit across controller resets and policy changes.
    uint64_t lifecycle{0};
    // True once a canonical overload ended the WINDOW-identical phase.
    bool cut{false};
    // Baseline successes still required before the next probe.
    int64_t retryDebt{0};
    // The probe in flight, if any.
    std::optional<uint64_t> probe;
    // True once the probe in flight can no longer promote: it was charged with
    // an overload, or a cut or reset happened while it was in flight. A probe
    // holds one slot, so it is charged with at most one overload.
    bool probeFailed{false};
    // Remembered failures by level. Bounded by the levels at or below the
    // ceiling that failed and were not yet forgiven.
    folly::F14FastMap<int64_t, LevelFailures> failures;
  };
  // SMART_PROBING state; null for a limiter that never ran that policy.
  std::unique_ptr<ProbingState> probing_;
  bool usesProbingPolicyLocked() const;
  // True once a canonical overload ended the WINDOW-identical phase. Before
  // that, SMART_PROBING issues WINDOW's own tokens and feedback path. Caller
  // holds mutex_.
  bool isProbingLocked() const;
  // Clears probing state for a reset or a policy change. Caller holds mutex_.
  void resetProbingLocked();
  int64_t probingAdmissionCapacityLocked() const;
  std::vector<Token> tryAcquireProbingLocked(
      std::unique_lock<std::mutex>& lock,
      int64_t want);
  OutputCredit makeProbingCredit(const std::vector<Token>& tokens);
  // Preserves the authoritative verdict if later settlement or wakeup throws.
  void settleProbingAttempt(
      std::vector<Token> tokens,
      OutputCredit credit,
      Outcome outcome,
      bool validServiceProof,
      bool cancelled,
      bool nonCapacity,
      AdmissionLease::Completion& completion);
  void consumeProbingCredit(const std::shared_ptr<ProbingCredit>& cell);
  void releaseProbingLocked(uint64_t admissionId);
  // Baseline successes a probe at 'level' waits after 'failures' failures
  // there: level * 2^(failures - 1), capped at 2 * level^2, i.e. one round
  // doubling per failure up to 2 * level rounds.
  static int64_t probePenalty(int64_t level, int64_t failures);
  // Baseline successes before the next probe: one round at a level that never
  // failed, otherwise its penalty. Caller holds mutex_.
  int64_t nextRetryDebtLocked() const;
  // Remembers a failure at 'level'. Caller holds mutex_.
  void recordLevelFailureLocked(int64_t level);
  // Forgives one failure at each level the baseline has sustained for as many
  // successes as its penalty. Caller holds mutex_.
  void creditSustainedLevelsLocked();
  // Counts successful baseline units toward probe spacing and forgiveness.
  // Caller holds mutex_.
  void creditBaselineSuccessLocked(int64_t units);
  // Raises the baseline to the probed level. Caller holds mutex_.
  void promoteProbeLocked();
  // Charges an overload to the probe instead of the baseline. Caller holds
  // mutex_.
  void failProbeLocked();

  // Peak charged startup population, retained across releases so late feedback
  // cannot mistake a drained window for the rejected allowance.
  int64_t startupPeak_{0};

  // Withdraws one producer without affecting other producers or held tokens.
  void cancelRunnableDemand(uint64_t id) noexcept;

  // Names the backend this limiter admits for in logs: the admission key, with
  // a suffix on a policy-scoped limiter. The registry, not this name, decides
  // which queries share the limiter.
  const std::string admissionKey_;

  // Supplies monotonic time. Injected together with timekeeper_ in tests so
  // pacing can advance without wall-clock sleeps.
  const std::function<std::chrono::steady_clock::time_point()> now_;
  // Returns a value in [0, 1] for probe jitter.
  const std::function<double()> random_;

  // Schedules paced waits. Null selects Folly's process-global timekeeper.
  folly::Timekeeper* const timekeeper_;

  // Guards config_, limit_, lowWaterLimit_ and waiters_. pending_ and
  // peakPending_ are atomic for the integral reservation and stats paths, but
  // are read under the lock wherever pacing depends on them.
  mutable std::mutex mutex_;

  // Mutable so initialization can install one resolved policy atomically and
  // tests can exercise controlled policy changes.
  Config config_;

  // Set once initializeOrMatch() seals a policy. Later calls compare against
  // config_ rather than change it; the adapted capacity below keeps moving,
  // because that is learned from the outcomes of every query sharing it.
  bool initialized_{false};

  // Adaptive congestion window in admission units per service horizon. Zero
  // means the hard ceiling governs.
  double limit_{0.0};

  // When limit_ last changed by a cut, growth or configuration. A recovering
  // fractional window grows from here.
  std::chrono::steady_clock::time_point limitChangedAt_;

  // True once a paced call has succeeded since the last cut: the backend is
  // serving again, so the window grows with time instead of waiting for
  // successes that pacing makes rare.
  bool recovering_{false};

  // Lowest initialized adaptive limit. Zero means no adaptive low-water mark
  // has been recorded.
  double lowWaterLimit_{0.0};

  // Pacing time and window-time integral accrued up to lastAccrual_. stats()
  // adds the interval since then without accruing it.
  std::chrono::steady_clock::duration pacingTime_{0};
  double windowUnitSeconds_{0.0};
  std::chrono::steady_clock::time_point lastAccrual_;

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

  // Units currently in flight through this limiter. Integral reservations and
  // release use atomic updates; fractional admission compares it under mutex_.
  std::atomic<int64_t> pending_{0};

  // High-water pending count over this limiter's lifetime, for runtime stats.
  // Updated with a relaxed compare-exchange: best effort, not a
  // synchronization point.
  std::atomic<int64_t> peakPending_{0};

  // Drivers parked waiting for a slot. Woken together when admission becomes
  // possible; each retries the atomic reservation.
  std::deque<ContinuePromise> waiters_;

  // Prevents callers from constructing owner-backed tokens directly.
  [[no_unique_address]] TokenConstructionKey tokenConstructionKey_;
};

/// Owns the process-scoped RPCRateLimiters: one shared limiter per admission
/// key, plus one for each other policy that queries run against that key.
///
/// A backend is shared across queries by definition, so the registry is a
/// process singleton rather than something threaded through the operator.
class RPCRateLimiterRegistry {
 public:
  static RPCRateLimiterRegistry& global();

  /// Returns the backend's shared limiter, creating it on first sight. The
  /// reference stays valid for the process lifetime: values are shared-owned,
  /// and later insertions do not replace an existing limiter.
  RPCRateLimiter& get(const std::string& admissionKey);

  /// The limiter a query runs under, from getForPolicy().
  struct Lookup {
    /// Valid for the life of the process.
    RPCRateLimiter* limiter{nullptr};

    /// True when the backend's shared limiter runs another policy and
    /// 'limiter' is the one for the query's own policy.
    bool policyScoped{false};
  };

  /// Returns the limiter for a query that runs `policy` against the backend
  /// named by `admissionKey`. The first query to reach the backend seals the
  /// shared limiter, get(admissionKey), with its policy, and every later query
  /// whose policy matches shares it (see RPCRateLimiter::initializeOrMatch()).
  /// A query whose policy does not match gets the limiter for its policy,
  /// created and sealed on first sight and shared with every later query whose
  /// policy matches that one. A new limiter starts from its initial window
  /// rather than from any backoff another limiter for the same backend has
  /// learned. Limiters live as long as the process.
  Lookup getForPolicy(
      std::string_view admissionKey,
      const RPCRateLimiter::Config& policy);

  /// Resets every backend in place and restores process-global defaults. A
  /// policy-scoped limiter is sealed with the policy it ran before the reset.
  /// Resets rather than drops: a Token releases through a back-pointer to its
  /// issuing limiter, so erasing the map would leave outstanding tokens
  /// pointing at freed memory. Intended only
  /// for tests and benchmarks that drive a real operator, which resolves its
  /// backend through this registry and so needs process state reset between
  /// iterations. Never call this from production code.
  void testingReset();

 private:
  // Read-mostly: limiters are created once then only looked up, so lookups take
  // a shared lock and do not serialize against each other. Only first sight
  // of a backend or of a policy takes the exclusive lock.
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<RPCRateLimiter>> backends_;

  // Limiters for policies other than the shared limiter's, by admission key,
  // each paired with the reduced policy initializeOrMatch() compares. A backend
  // sees few distinct policies, so a lookup scans its list.
  std::unordered_map<
      std::string,
      std::vector<
          std::pair<RPCRateLimiter::Config, std::shared_ptr<RPCRateLimiter>>>>
      policyScopedLimiters_;
};

} // namespace facebook::velox::exec::rpc
