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

#include "velox/exec/rpc/RPCRateLimiter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <string_view>
#include <utility>
#include <vector>

#include <folly/Random.h>
#include <folly/ScopeGuard.h>
#include <folly/executors/InlineExecutor.h>
#include <folly/futures/Future.h>

#include "velox/common/base/CheckedArithmetic.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/testutil/TestValue.h"

#define RPC_RATE_LIMITER_LOG(severity) LOG(severity) << "[RPC_RATE_LIMITER] "
#define RPC_RATE_LIMITER_VLOG(level) VLOG(level) << "[RPC_RATE_LIMITER] "

namespace facebook::velox::exec::rpc {

namespace {
// Fallback ceiling for a backend configured with Config::ceiling == 0: 20
// concurrent RPCs per process per backend.
std::atomic<int64_t>& defaultCapacityRef() {
  static std::atomic<int64_t> capacity{20};
  return capacity;
}

constexpr int64_t kBootstrapServiceHorizonNanos{1'000'000'000};

// Most units one SMART_PROBING acquisition takes before the first service
// sample. Nothing is known about the backend yet, so small grabs keep admission
// incremental: the startup peak then tracks what the backend saw, and the first
// cut halves that rather than an unused ceiling. Repeated acquisitions can
// still fill the window.
constexpr int64_t kStartupAcquisitionCap{4};

// Highest floor the probe interval may impose. A floor of one would end pacing,
// letting probes go out as fast as rejections return; a backend whose calls
// outlast the probe interval is probed every two horizons instead.
constexpr double kMaxProbeFloor{0.5};

// Largest fraction by which a back-off probe comes early. Probes are never
// later than the interval, so the maximum probe interval still holds.
constexpr double kProbeJitter{0.2};

constexpr int64_t kServiceHorizonEwmaWeight{8};
constexpr double kInitialAdaptiveLimit{200.0};

template <typename Period>
std::chrono::steady_clock::duration boundedPacingDuration(
    std::chrono::duration<double, Period> value,
    bool roundUp) {
  using Duration = std::chrono::steady_clock::duration;
  VELOX_CHECK(!std::isnan(value.count()), "Invalid RPC pacing duration");
  const auto maximum = std::chrono::duration<double, Period>(Duration::max());
  if (value >= maximum) {
    return Duration::max();
  }
  if (value.count() <= 0) {
    return Duration::zero();
  }
  return roundUp ? std::chrono::ceil<Duration>(value)
                 : std::chrono::duration_cast<Duration>(value);
}

std::chrono::steady_clock::time_point pacingDeadline(
    std::chrono::steady_clock::time_point origin,
    std::chrono::steady_clock::duration delay) {
  const auto maximum = std::chrono::steady_clock::time_point::max();
  return origin > maximum - delay ? maximum : origin + delay;
}

int64_t boundedCapacity(double window, int64_t ceiling) {
  VELOX_CHECK_GE(ceiling, 1);
  VELOX_CHECK(!std::isnan(window), "Invalid RPC capacity");
  if (window >= static_cast<double>(ceiling)) {
    return ceiling;
  }
  if (window <= 1.0) {
    return 1;
  }
  return static_cast<int64_t>(std::floor(window));
}

int64_t scaledMetric(double value) {
  VELOX_CHECK(!std::isnan(value), "Invalid RPC metric");
  if (value <= 0.0) {
    return 0;
  }
  const auto maximum = std::numeric_limits<int64_t>::max();
  if (value >= static_cast<double>(maximum)) {
    return maximum;
  }
  return static_cast<int64_t>(std::llround(value));
}

// Reports a contained teardown failure without allowing diagnostics to escape
// the noexcept token-release path.
void logLimiterFailure(
    std::string_view admissionKey,
    std::string_view stage) noexcept {
  try {
    RPC_RATE_LIMITER_LOG(ERROR)
        << "limiter[" << admissionKey << "] failed during " << stage;
  } catch (...) {
  }
}

void fulfillWaitersNoThrow(
    std::vector<ContinuePromise>& waiters,
    std::string_view admissionKey) noexcept {
  for (auto& waiter : waiters) {
    try {
      waiter.setValue();
    } catch (...) {
      logLimiterFailure(admissionKey, "waiter notification");
    }
  }
}

void failWaitersNoThrow(
    std::vector<ContinuePromise>& waiters,
    const folly::exception_wrapper& error,
    std::string_view admissionKey) noexcept {
  for (auto& waiter : waiters) {
    try {
      waiter.setException(error);
    } catch (...) {
      logLimiterFailure(admissionKey, "timer failure notification");
    }
  }
}
} // namespace

// --- Token ---

RPCRateLimiter::Token& RPCRateLimiter::Token::operator=(
    Token&& other) noexcept {
  if (this != &other) {
    if (owner_ != nullptr) {
      owner_->release(admissionId_, kind_);
    }
    owner_ = other.owner_;
    overloadEpoch_ = other.overloadEpoch_;
    admissionId_ = other.admissionId_;
    kind_ = other.kind_;
    other.owner_ = nullptr;
  }
  return *this;
}

RPCRateLimiter::Token::~Token() noexcept {
  if (owner_ != nullptr) {
    owner_->release(admissionId_, kind_);
  }
}

// --- AdmissionLease ---

RPCRateLimiter::AdmissionLease::AdmissionLease(
    std::shared_ptr<RPCRateLimiter> owner,
    std::vector<Token> tokens,
    uint64_t overloadEpoch)
    : owner_{std::move(owner)},
      tokens_{std::move(tokens)},
      overloadEpoch_{overloadEpoch},
      probingLease_{
          !tokens_.empty() &&
          tokens_.front().belongsTo(owner_.get(), AdmissionKind::kProbing)},
      probingCredit_{
          probingLease_ ? owner_->makeProbingCredit(tokens_) : OutputCredit{}},
      cancellationCells_{probingCredit_.cells(owner_->tokenConstructionKey_)} {}

RPCRateLimiter::AdmissionLease::~AdmissionLease() {
  release();
}

RPCRateLimiter::AdmissionLease::Completion
RPCRateLimiter::AdmissionLease::completeAndRelease(bool hardOverload) noexcept {
  return completeAndRelease(
      hardOverload ? Outcome::kOverload : Outcome::kNone,
      /*handedOff=*/false);
}

RPCRateLimiter::AdmissionLease::Completion
RPCRateLimiter::AdmissionLease::completeAndRelease(
    Outcome outcome,
    bool handedOff) noexcept {
  if (probingLease_) {
    bool unresolved;
    bool rawBound;
    bool nonCapacity;
    {
      std::lock_guard<std::mutex> l(mutex_);
      if (completed_) {
        return {};
      }
      rawBound = rawBound_;
      nonCapacity = nonCapacityRejection_;
      unresolved = rawBound && !rawSettled_;
    }
    if (unresolved) {
      const bool speculative = !nonCapacity &&
          std::all_of(
              cancellationCells_.begin(),
              cancellationCells_.end(),
              [](const auto& cell) { return cell->probe; });
      release();
      bool stale;
      {
        std::lock_guard<std::mutex> l(owner_->mutex_);
        stale = overloadEpoch_ != owner_->overloadEpoch_;
      }
      const bool handled =
          !stale && !speculative && outcome == Outcome::kOverload
          ? owner_->handleOverloadForEpoch(overloadEpoch_, nonCapacity)
          : (speculative || stale) && outcome == Outcome::kOverload;
      return Completion{
          .sharedOverloadHandled = handled,
          .overloadEpoch = overloadEpoch_,
          .disposition = stale ? FeedbackDisposition::kStale
              : speculative    ? FeedbackDisposition::kExclusivelySpeculative
                               : FeedbackDisposition::kCanonical,
      };
    }
    if (!rawBound) {
      settleRawAttempt(outcome, !handedOff);
    }
    std::lock_guard<std::mutex> l(mutex_);
    completed_ = true;
    return std::move(rawCompletion_);
  }
  std::shared_ptr<RPCRateLimiter> owner;
  std::vector<Token> tokens;
  bool nonCapacity{false};
  try {
    std::lock_guard<std::mutex> l(mutex_);
    if (completed_) {
      return {};
    }
    completed_ = true;
    nonCapacity = nonCapacityRejection_;
    owner = std::move(owner_);
    tokens = std::move(tokens_);
  } catch (...) {
    logLimiterFailure("admission lease", "completion");
    return {};
  }

  bool handled = false;
  if (outcome == Outcome::kOverload && !tokens.empty()) {
    try {
      handled = owner->handleOverloadForEpoch(overloadEpoch_, nonCapacity);
    } catch (...) {
      logLimiterFailure("admission lease", "completion feedback");
    }
  }
  tokens.clear();
  return Completion{
      .sharedOverloadHandled = handled, .overloadEpoch = overloadEpoch_};
}

void RPCRateLimiter::AdmissionLease::markNonCapacityRejection() noexcept {
  std::lock_guard<std::mutex> l(mutex_);
  nonCapacityRejection_ = true;
}

void RPCRateLimiter::AdmissionLease::bindRawAttempt() noexcept {
  if (probingLease_) {
    std::lock_guard<std::mutex> l(mutex_);
    rawBound_ = true;
    rawCompletion_.overloadEpoch = overloadEpoch_;
    rawCompletion_.disposition =
        std::all_of(
            cancellationCells_.begin(),
            cancellationCells_.end(),
            [](const auto& cell) { return cell->probe; })
        ? FeedbackDisposition::kExclusivelySpeculative
        : FeedbackDisposition::kCanonical;
  }
}

void RPCRateLimiter::AdmissionLease::settleRawAttempt(
    Outcome outcome,
    bool validServiceProof) noexcept {
  if (!probingLease_) {
    return;
  }
  std::shared_ptr<RPCRateLimiter> owner;
  std::vector<Token> tokens;
  OutputCredit credit;
  Completion completion;
  bool cancelled;
  bool valid;
  bool nonCapacity;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (tokens_.empty()) {
      return;
    }
    owner = owner_;
    tokens = std::move(tokens_);
    credit = std::move(probingCredit_);
    cancelled = cancelled_;
    valid = validServiceProof;
    nonCapacity = nonCapacityRejection_;
    completion.overloadEpoch = overloadEpoch_;
    completion.disposition = rawCompletion_.disposition;
  }
  try {
    owner->settleProbingAttempt(
        std::move(tokens),
        std::move(credit),
        outcome,
        valid,
        cancelled,
        nonCapacity,
        completion);
  } catch (...) {
    // Keep credit already earned before a late notification failure.
    logLimiterFailure("probing lease", "raw settlement");
  }
  std::lock_guard<std::mutex> l(mutex_);
  if (cancelled_) {
    completion.credit = {};
  }
  rawCompletion_ = std::move(completion);
  rawSettled_ = true;
}

void RPCRateLimiter::AdmissionLease::release() noexcept {
  if (probingLease_) {
    std::shared_ptr<RPCRateLimiter> owner;
    std::vector<Token> rollback;
    {
      std::lock_guard<std::mutex> l(mutex_);
      if (completed_) {
        return;
      }
      owner = owner_;
      completed_ = true;
      cancelled_ = true;
      for (const auto& cell : cancellationCells_) {
        cell->cancelled.store(true, std::memory_order_release);
      }
      rawCompletion_.credit = {};
      if (!rawBound_) {
        rollback = std::move(tokens_);
      }
    }
    rollback.clear();
    return;
  }
  std::shared_ptr<RPCRateLimiter> owner;
  std::vector<Token> tokens;
  try {
    std::lock_guard<std::mutex> l(mutex_);
    if (completed_) {
      return;
    }
    completed_ = true;
    owner = std::move(owner_);
    tokens = std::move(tokens_);
  } catch (...) {
    logLimiterFailure("admission lease", "cancellation");
    return;
  }
  tokens.clear();
}

// --- RPCRateLimiter ---

RPCRateLimiter::FeedbackDisposition RPCRateLimiter::strongerDisposition(
    FeedbackDisposition lhs,
    FeedbackDisposition rhs) {
  const auto rank = [](FeedbackDisposition disposition) {
    switch (disposition) {
      case FeedbackDisposition::kStale:
        return 0;
      case FeedbackDisposition::kExclusivelySpeculative:
        return 1;
      case FeedbackDisposition::kCanonical:
        return 2;
    }
    VELOX_UNREACHABLE();
  };
  return rank(rhs) > rank(lhs) ? rhs : lhs;
}

RPCRateLimiter::RPCRateLimiter(std::string admissionKey)
    : RPCRateLimiter(
          std::move(admissionKey),
          [] { return std::chrono::steady_clock::now(); },
          nullptr,
          [] { return folly::Random::randDouble01(); }) {}

RPCRateLimiter::RPCRateLimiter(
    std::string admissionKey,
    std::function<std::chrono::steady_clock::time_point()> now,
    folly::Timekeeper* timekeeper,
    std::function<double()> random)
    : admissionKey_{std::move(admissionKey)},
      now_{std::move(now)},
      random_{std::move(random)},
      timekeeper_{timekeeper},
      lastAccrual_{now_()} {}

void RPCRateLimiter::setDefaultCapacity(int64_t capacity) {
  defaultCapacityRef().store(capacity);
  RPC_RATE_LIMITER_VLOG(1) << "default capacity set to " << capacity;
}

int64_t RPCRateLimiter::defaultCapacity() {
  return defaultCapacityRef().load();
}

int64_t RPCRateLimiter::ceilingLocked() const {
  return config_.ceiling > 0 ? config_.ceiling : defaultCapacityRef().load();
}

double RPCRateLimiter::floorLocked(double ceiling) const {
  const double probeFloor = config_.maxProbeInterval.count() == 0
      ? 0.0
      : std::min(
            kMaxProbeFloor,
            static_cast<double>(serviceHorizonNanos_) /
                std::chrono::duration<double, std::nano>(
                    config_.maxProbeInterval)
                    .count());
  return std::min(std::max({0.001, config_.floor, probeFloor}), ceiling);
}

double RPCRateLimiter::limitLocked() const {
  const double ceiling = static_cast<double>(ceilingLocked());
  if (!config_.adaptive || limit_ <= 0.0) {
    return ceiling;
  }
  return std::clamp(limit_, floorLocked(ceiling), ceiling);
}

bool RPCRateLimiter::isPacingLocked() const {
  return config_.adaptive && limitLocked() < 1.0;
}

bool RPCRateLimiter::handleOverloadForEpoch(
    uint64_t overloadEpoch,
    bool nonCapacity) {
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (overloadEpoch != overloadEpoch_) {
      return true;
    }
    ++overloadEpoch_;
    applyOverloadLocked(nonCapacity);
    timerRequest = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
  return true;
}

std::chrono::steady_clock::duration RPCRateLimiter::pacingIntervalLocked()
    const {
  const double window = limitLocked();
  // A recovering window w doubling every horizon H accumulates admissions at
  // w(t) / H. One admission takes H * log2(1 + ln2 / w) while w stays below
  // one. A window above 1 - ln2 reaches one first, after H * log2(1 / w), and
  // then admits at 1 / H for the remaining credit.
  const double horizons = !recovering_ ? 1.0 / window
      : window <= 1.0 - std::numbers::ln2
      ? std::log2(1.0 + std::numbers::ln2 / window)
      : std::log2(1.0 / window) + 1.0 - (1.0 - window) / std::numbers::ln2;
  return boundedPacingDuration(
      std::chrono::duration<double, std::nano>{
          static_cast<double>(serviceHorizonNanos_) * horizons},
      /*roundUp=*/true);
}

std::chrono::steady_clock::duration RPCRateLimiter::nextPacedDelayLocked()
    const {
  const auto interval = pacingIntervalLocked();
  if (recovering_) {
    return interval;
  }
  const double fraction = 1.0 - kProbeJitter * (1.0 - random_());
  return boundedPacingDuration(interval * fraction, /*roundUp=*/false);
}

double RPCRateLimiter::grownLimitLocked(
    std::chrono::steady_clock::time_point now) const {
  const double window = limitLocked();
  if (!recovering_ || !isPacingLocked() || now <= limitChangedAt_) {
    return window;
  }
  const double horizons =
      std::chrono::duration<double, std::nano>(now - limitChangedAt_).count() /
      static_cast<double>(serviceHorizonNanos_);
  return std::min(1.0, window * std::exp2(horizons));
}

void RPCRateLimiter::commitGrowthLocked() {
  const double grown = grownLimitLocked(now_());
  if (grown > limitLocked()) {
    setLimitLocked(grown);
  }
}

void RPCRateLimiter::accrueLocked() {
  const auto now = now_();
  const auto elapsed = now - lastAccrual_;
  if (isPacingLocked()) {
    pacingTime_ += elapsed;
  }
  windowUnitSeconds_ +=
      limitLocked() * std::chrono::duration<double>(elapsed).count();
  lastAccrual_ = now;
}

void RPCRateLimiter::setLimitLocked(double limit) {
  accrueLocked();
  ++pacingGeneration_;
  pacingTimerArmed_ = false;
  const double ceiling = static_cast<double>(ceilingLocked());
  limit_ = limit > 0.0 && limit < ceiling
      ? std::clamp(limit, floorLocked(ceiling), ceiling)
      : 0.0;
  const auto now = now_();
  limitChangedAt_ = now;
  if (isPacingLocked()) {
    const auto delay = nextPacedDelayLocked();
    nextPacedAdmission_ = lastAdmission_.has_value()
        ? std::max(now, pacingDeadline(lastAdmission_.value(), delay))
        : pacingDeadline(now, delay);
  } else {
    recovering_ = false;
    nextPacedAdmission_ = std::chrono::steady_clock::time_point{};
  }
}

int64_t RPCRateLimiter::capacityLocked() const {
  return boundedCapacity(limitLocked(), ceilingLocked());
}

namespace {
// Clamps 'config' into its valid ranges and replaces a NaN floor or decrease
// factor with its default.
RPCRateLimiter::Config clamped(RPCRateLimiter::Config config) {
  const RPCRateLimiter::Config defaults;
  config.floor =
      std::isnan(config.floor) ? defaults.floor : std::max(0.001, config.floor);
  config.decreaseFactor = std::isnan(config.decreaseFactor)
      ? defaults.decreaseFactor
      : std::clamp(config.decreaseFactor, 0.01, 0.99);
  config.maxProbeInterval =
      std::max(config.maxProbeInterval, std::chrono::milliseconds{0});
  if (!config.adaptive) {
    config.flowControlPolicy = RPCFlowControlPolicy::kWindowAimd;
  }
  return config;
}

// Reduces 'policy' to the fields a limiter reads, so policies that differ only
// in fields it ignores compare equal.
RPCRateLimiter::Config canonicalize(const RPCRateLimiter::Config& policy) {
  auto canonical = clamped(policy);
  if (canonical.flowControlPolicy == RPCFlowControlPolicy::kWindowAimd &&
      canonical.ceiling > 0 &&
      canonical.floor >= static_cast<double>(canonical.ceiling)) {
    canonical.adaptive = false;
  }
  if (!canonical.adaptive) {
    const RPCRateLimiter::Config defaults;
    canonical.floor = defaults.floor;
    canonical.decreaseFactor = defaults.decreaseFactor;
    canonical.maxProbeInterval = defaults.maxProbeInterval;
    canonical.flowControlPolicy = RPCFlowControlPolicy::kWindowAimd;
  }
  return canonical;
}
} // namespace

void RPCRateLimiter::configure(const Config& config) {
  // Replacing the whole config is one kind of amendment, so it shares
  // amend()'s clamping and adaptive-flip logging rather than repeating them.
  amend([&config](Config& target) { target = config; });
}

bool RPCRateLimiter::initializeOrMatch(const Config& policy) {
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  bool matches{true};
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (initialized_) {
      matches = canonicalize(policy) == canonicalize(config_);
    } else {
      applyMutationLocked([&policy](Config& config) { config = policy; });
      timerRequest = collectWakeableLocked(toNotify);
      // Sealed here rather than by a second call: two calls left a window in
      // which another query saw initialized_ still false and overwrote.
      initialized_ = true;
    }
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
  return matches;
}

void RPCRateLimiter::testingReset() {
  std::lock_guard<std::mutex> l(mutex_);
  config_ = Config{};
  initialized_ = false;
  limit_ = 0.0;
  limitChangedAt_ = std::chrono::steady_clock::time_point{};
  recovering_ = false;
  lowWaterLimit_ = 0.0;
  pacingTime_ = std::chrono::steady_clock::duration::zero();
  windowUnitSeconds_ = 0.0;
  lastAccrual_ = now_();
  serviceHorizonNanos_ = kBootstrapServiceHorizonNanos;
  hasServiceHorizonSample_ = false;
  startupPeak_ = 0;
  nextPacedAdmission_ = std::chrono::steady_clock::time_point{};
  lastAdmission_.reset();
  ++pacingGeneration_;
  ++overloadEpoch_;
  pacingTimerArmed_ = false;
  if (probing_) {
    resetProbingLocked();
  }
  if (!probing_ || probing_->receipts.empty()) {
    pending_.store(0);
  }
  peakPending_.store(pending_.load());
  waiters_.clear();
}

std::optional<RPCRateLimiter::PacingTimerRequest>
RPCRateLimiter::collectWakeableLocked(std::vector<ContinuePromise>& toNotify) {
  if (isProbingLocked()) {
    const auto now = now_();
    if (waiters_.empty() ||
        pending_.load() >= probingAdmissionCapacityLocked() ||
        (isPacingLocked() && pending_.load() != 0)) {
      return std::nullopt;
    }
    if (isPacingLocked() && now < nextPacedAdmission_) {
      return preparePacingTimerLocked();
    }
    while (!waiters_.empty()) {
      common::testutil::TestValue::adjust(
          "facebook::velox::exec::rpc::RPCRateLimiter::probingWakePreflight",
          this);
      toNotify.push_back(std::move(waiters_.front()));
      waiters_.pop_front();
    }
    return std::nullopt;
  }
  if (isPacingLocked()) {
    if (pending_.load() != 0 || waiters_.empty()) {
      return std::nullopt;
    }
    if (now_() < nextPacedAdmission_) {
      return preparePacingTimerLocked();
    }
    while (!waiters_.empty()) {
      toNotify.push_back(std::move(waiters_.front()));
      waiters_.pop_front();
    }
    return std::nullopt;
  }
  int64_t headroom = capacityLocked() - pending_.load();
  if (headroom <= 0) {
    return std::nullopt;
  }
  while (!waiters_.empty()) {
    toNotify.push_back(std::move(waiters_.front()));
    waiters_.pop_front();
  }
  return std::nullopt;
}

std::optional<RPCRateLimiter::PacingTimerRequest>
RPCRateLimiter::preparePacingTimerLocked() {
  if (pacingTimerArmed_ || !isPacingLocked() || pending_.load() != 0 ||
      waiters_.empty()) {
    return std::nullopt;
  }
  const auto now = now_();
  // SMART_PROBING also arms the timer when the admission is already due, so a
  // waiter that parked at the deadline still wakes without another event.
  if (now >= nextPacedAdmission_ &&
      config_.flowControlPolicy != RPCFlowControlPolicy::kSmartProbing) {
    return std::nullopt;
  }
  const auto delay = std::max(
      std::chrono::microseconds{1},
      std::chrono::ceil<std::chrono::microseconds>(nextPacedAdmission_ - now));
  const auto generation = pacingGeneration_;
  const auto weakSelf = weak_from_this();
  VELOX_CHECK(
      !weakSelf.expired(), "Paced RPCRateLimiter requires shared ownership");
  pacingTimerArmed_ = true;
  return PacingTimerRequest{delay, generation, weakSelf};
}

void RPCRateLimiter::schedulePacingTimer(
    const PacingTimerRequest& request) noexcept {
  try {
    const auto weakSelf = request.owner;
    folly::futures::detachOn(
        folly::getKeepAliveToken(folly::InlineExecutor::instance()),
        folly::futures::sleep(request.delay, timekeeper_)
            .deferValue(
                [weakSelf, generation = request.generation](folly::Unit) {
                  if (const auto self = weakSelf.lock()) {
                    self->onPacingTimer(generation);
                  }
                })
            .deferError([weakSelf, generation = request.generation](
                            const folly::exception_wrapper& error) {
              if (const auto self = weakSelf.lock()) {
                self->onPacingTimerFailure(generation, error);
              }
              return folly::Unit{};
            }));
  } catch (...) {
    onPacingTimerFailure(
        request.generation, folly::exception_wrapper(std::current_exception()));
  }
}

void RPCRateLimiter::onPacingTimer(uint64_t generation) {
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (generation != pacingGeneration_ || !pacingTimerArmed_) {
      return;
    }
    pacingTimerArmed_ = false;
    timerRequest = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
}

void RPCRateLimiter::onPacingTimerFailure(
    uint64_t generation,
    const folly::exception_wrapper& error) noexcept {
  std::vector<ContinuePromise> toFail;
  try {
    std::lock_guard<std::mutex> l(mutex_);
    if (generation != pacingGeneration_ || !pacingTimerArmed_) {
      return;
    }
    pacingTimerArmed_ = false;
    while (!waiters_.empty()) {
      toFail.push_back(std::move(waiters_.front()));
      waiters_.pop_front();
    }
  } catch (...) {
    logLimiterFailure(admissionKey_, "timer failure handling");
    return;
  }
  failWaitersNoThrow(toFail, error, admissionKey_);
}

void RPCRateLimiter::applyMutationLocked(
    const std::function<void(Config&)>& mutate) {
  const bool wasAdaptive = config_.adaptive;
  const auto previousFlowControlPolicy = config_.flowControlPolicy;
  accrueLocked();
  auto nextConfig = config_;
  mutate(nextConfig);
  config_ = clamped(nextConfig);
  if (config_.flowControlPolicy == RPCFlowControlPolicy::kSmartProbing &&
      !probing_) {
    probing_ = std::make_unique<ProbingState>();
  }
  if (probing_ && previousFlowControlPolicy != config_.flowControlPolicy) {
    resetProbingLocked();
  }
  if (previousFlowControlPolicy != config_.flowControlPolicy &&
      (previousFlowControlPolicy == RPCFlowControlPolicy::kSmartProbing ||
       config_.flowControlPolicy == RPCFlowControlPolicy::kSmartProbing)) {
    ++overloadEpoch_;
  }
  if (wasAdaptive != config_.adaptive ||
      previousFlowControlPolicy != config_.flowControlPolicy) {
    startupPeak_ = 0;
  }
  if (!config_.adaptive) {
    setLimitLocked(0.0);
  } else if (!wasAdaptive) {
    setLimitLocked(
        std::min(kInitialAdaptiveLimit, static_cast<double>(ceilingLocked())));
    if (lowWaterLimit_ == 0.0 || limitLocked() < lowWaterLimit_) {
      lowWaterLimit_ = limitLocked();
    }
  } else if (limit_ > 0.0) {
    commitGrowthLocked();
    setLimitLocked(limit_);
  }
  if (wasAdaptive != config_.adaptive) {
    RPC_RATE_LIMITER_LOG(WARNING)
        << "adaptive capacity " << (config_.adaptive ? "ENABLED" : "DISABLED")
        << " for admission key " << admissionKey_ << " (floor=" << config_.floor
        << ", decrease=" << config_.decreaseFactor
        << ", max probe interval=" << config_.maxProbeInterval.count() << "ms)";
  }
}

void RPCRateLimiter::amend(const std::function<void(Config&)>& mutate) {
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  {
    std::lock_guard<std::mutex> l(mutex_);
    applyMutationLocked(mutate);
    // A mutation that raises the ceiling grows capacity, and no release or
    // adaptation is guaranteed to follow, so drivers parked under the old
    // capacity would stay parked. Wake whatever now fits.
    timerRequest = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
}

RPCRateLimiter::Config RPCRateLimiter::config() const {
  std::lock_guard<std::mutex> l(mutex_);
  return config_;
}

std::shared_ptr<RPCRateLimiter::AdmissionLease> RPCRateLimiter::makeLease(
    std::vector<Token> tokens) {
  if (tokens.empty()) {
    VELOX_FAIL("Admission lease requires at least one token");
  }
  const auto overloadEpoch = tokens.at(0).overloadEpoch();
  for (const auto& token : tokens) {
    VELOX_CHECK(
        token.belongsTo(this),
        "Admission lease cannot combine tokens from different limiters");
    VELOX_CHECK_EQ(
        token.overloadEpoch(),
        overloadEpoch,
        "Admission lease cannot combine tokens from different epochs");
  }
  return std::make_shared<AdmissionLease>(
      shared_from_this(), std::move(tokens), overloadEpoch);
}

std::shared_ptr<RPCRateLimiter::AdmissionLease> RPCRateLimiter::makeLease(
    Token token) {
  std::vector<Token> tokens;
  tokens.push_back(std::move(token));
  return makeLease(std::move(tokens));
}

void RPCRateLimiter::notePeakPending(int64_t pending) {
  // Relaxed ordering: a best-effort max for stats, not a synchronization
  // point.
  int64_t peak = peakPending_.load(std::memory_order_relaxed);
  while (pending > peak &&
         !peakPending_.compare_exchange_weak(
             peak, pending, std::memory_order_relaxed)) {
  }
}

RPCRateLimiter::Token RPCRateLimiter::acquire() {
  int64_t pending;
  uint64_t overloadEpoch;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(
        !usesProbingPolicyLocked(),
        "Unconditional acquisition cannot bypass RPC probing admission");
    VELOX_CHECK(
        !isPacingLocked(),
        "Unconditional acquisition cannot bypass paced admission");
    lastAdmission_ = now_();
    pending = ++pending_;
    overloadEpoch = overloadEpoch_;
    noteStartupPeakLocked(pending);
  }
  notePeakPending(pending);
  RPC_RATE_LIMITER_VLOG(2) << "acquire[" << admissionKey_
                           << "]: pending=" << pending;
  return Token{
      tokenConstructionKey_, this, overloadEpoch, 0, AdmissionKind::kPlain};
}

bool RPCRateLimiter::usesProbingStartupLocked() const {
  return config_.adaptive &&
      config_.flowControlPolicy == RPCFlowControlPolicy::kSmartProbing &&
      !hasServiceHorizonSample_;
}

void RPCRateLimiter::noteStartupPeakLocked(int64_t pending) {
  if (usesProbingStartupLocked()) {
    startupPeak_ = std::max(startupPeak_, pending);
  }
}

RPCRateLimiter::OutputCredit::OutputCredit(
    TokenConstructionKey&,
    std::shared_ptr<RPCRateLimiter> owner,
    ProbingCreditCells cells)
    : owner_{std::move(owner)}, cells_{std::move(cells)} {}

void RPCRateLimiter::OutputCredit::consume() noexcept {
  for (const auto& cell : cells_) {
    try {
      owner_->consumeProbingCredit(cell);
    } catch (...) {
      logLimiterFailure("probing output", "credit consumption");
    }
  }
  cells_.clear();
  owner_.reset();
}

bool RPCRateLimiter::usesProbingPolicyLocked() const {
  return probing_ && config_.adaptive &&
      config_.flowControlPolicy == RPCFlowControlPolicy::kSmartProbing;
}

bool RPCRateLimiter::isProbingLocked() const {
  return usesProbingPolicyLocked() && probing_->cut;
}

void RPCRateLimiter::resetProbingLocked() {
  ++probing_->lifecycle;
  probing_->cut = false;
  probing_->retryDebt = 0;
  // A probe from before the reset still holds its slot until it settles, but
  // can no longer promote.
  probing_->probeFailed = probing_->probe.has_value();
  probing_->failures.clear();
}

int64_t RPCRateLimiter::probePenalty(int64_t level, int64_t failures) {
  VELOX_CHECK_GE(level, 1);
  VELOX_CHECK_GE(failures, 1);
  constexpr auto kMax = std::numeric_limits<int64_t>::max();
  int64_t cap;
  if (mulWithOverflow(level, level, &cap) ||
      mulWithOverflow(cap, int64_t{2}, &cap)) {
    cap = kMax;
  }
  const auto doublings = std::min<int64_t>(failures - 1, 62);
  int64_t penalty;
  if (mulWithOverflow(level, int64_t{1} << doublings, &penalty)) {
    penalty = kMax;
  }
  return std::min(cap, penalty);
}

int64_t RPCRateLimiter::nextRetryDebtLocked() const {
  const auto baseline = capacityLocked();
  const auto position = probing_->failures.find(baseline + 1);
  return position == probing_->failures.end()
      ? baseline
      : probePenalty(baseline + 1, position->second.count);
}

void RPCRateLimiter::recordLevelFailureLocked(int64_t level) {
  auto& failures = probing_->failures[level];
  if (failures.count < std::numeric_limits<int64_t>::max()) {
    ++failures.count;
  }
  failures.cleanSuccesses = 0;
}

void RPCRateLimiter::creditSustainedLevelsLocked() {
  const auto baseline = capacityLocked();
  folly::small_vector<int64_t, 4> forgiven;
  for (auto& [level, failures] : probing_->failures) {
    if (level > baseline ||
        ++failures.cleanSuccesses < probePenalty(level, failures.count)) {
      continue;
    }
    failures.cleanSuccesses = 0;
    if (--failures.count == 0) {
      forgiven.push_back(level);
    }
  }
  for (const auto level : forgiven) {
    probing_->failures.erase(level);
  }
}

void RPCRateLimiter::creditBaselineSuccessLocked(int64_t units) {
  for (int64_t i = 0; i < units; ++i) {
    if (probing_->retryDebt > 0) {
      --probing_->retryDebt;
    }
    creditSustainedLevelsLocked();
  }
}

void RPCRateLimiter::promoteProbeLocked() {
  setLimitLocked(static_cast<double>(capacityLocked() + 1));
  probing_->retryDebt = nextRetryDebtLocked();
}

void RPCRateLimiter::failProbeLocked() {
  // The baseline has not moved since the probe was admitted: promotion
  // settles the probe, and a cut fails it.
  probing_->probeFailed = true;
  recordLevelFailureLocked(capacityLocked() + 1);
  probing_->retryDebt = nextRetryDebtLocked();
}

int64_t RPCRateLimiter::probingAdmissionCapacityLocked() const {
  const auto baseline = capacityLocked();
  const bool probe = probing_->cut && !isPacingLocked() &&
      !probing_->probe.has_value() && probing_->retryDebt == 0 &&
      baseline < ceilingLocked() && !probing_->demands.empty();
  return probe ? baseline + 1 : baseline;
}

std::vector<RPCRateLimiter::Token> RPCRateLimiter::tryAcquireProbingLocked(
    std::unique_lock<std::mutex>& lock,
    int64_t want) {
  std::vector<Token> granted;
  auto unlockOnExit = folly::makeGuard([&] {
    if (lock.owns_lock()) {
      lock.unlock();
    }
  });
  granted.reserve(static_cast<size_t>(std::min(want, ceilingLocked() + 1)));
  while (static_cast<int64_t>(granted.size()) < want) {
    const auto now = now_();
    const auto capacity = probingAdmissionCapacityLocked();
    if (pending_.load() >= capacity ||
        (isPacingLocked() &&
         (pending_.load() != 0 || now < nextPacedAdmission_))) {
      break;
    }
    if (isPacingLocked()) {
      commitGrowthLocked();
    }
    auto cell = std::make_shared<ProbingCredit>();
    cell->admissionId = ++probing_->nextAdmissionId;
    cell->probe = probing_->cut && !isPacingLocked() &&
        pending_.load() >= capacityLocked();
    cell->level = pending_.load() + 1;
    cell->lifecycle = probing_->lifecycle;
    probing_->receipts.emplace(cell->admissionId, cell);
    bool reserved{false};
    auto rollback = folly::makeGuard([&] {
      if (!reserved) {
        probing_->receipts.erase(cell->admissionId);
      }
    });
    common::testutil::TestValue::adjust(
        "facebook::velox::exec::rpc::RPCRateLimiter::probingReservationPreflight",
        this);
    ++pending_;
    reserved = true;
    cell->admittedAt = now_();
    if (cell->probe) {
      probing_->probe = cell->admissionId;
      probing_->probeFailed = false;
    }
    granted.emplace_back(
        tokenConstructionKey_,
        this,
        overloadEpoch_,
        cell->admissionId,
        AdmissionKind::kProbing);
    noteStartupPeakLocked(pending_.load());
    notePeakPending(pending_.load());
    lastAdmission_ = cell->admittedAt;
    if (isPacingLocked()) {
      ++pacingGeneration_;
      pacingTimerArmed_ = false;
      nextPacedAdmission_ =
          pacingDeadline(cell->admittedAt, nextPacedDelayLocked());
      break;
    }
  }
  return granted;
}

RPCRateLimiter::OutputCredit RPCRateLimiter::makeProbingCredit(
    const std::vector<Token>& tokens) {
  ProbingCreditCells cells;
  cells.reserve(tokens.size());
  std::lock_guard<std::mutex> l(mutex_);
  for (const auto& token : tokens) {
    if (token.belongsTo(this, AdmissionKind::kProbing)) {
      const auto position = probing_->receipts.find(token.admissionId());
      VELOX_CHECK(position != probing_->receipts.end());
      cells.push_back(position->second);
    }
  }
  return OutputCredit{
      tokenConstructionKey_, shared_from_this(), std::move(cells)};
}

void RPCRateLimiter::consumeProbingCredit(
    const std::shared_ptr<ProbingCredit>& cell) {
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timer;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (cell->consumed.exchange(true, std::memory_order_acq_rel) ||
        !usesProbingPolicyLocked() || cell->lifecycle != probing_->lifecycle ||
        cell->cancelled.load(std::memory_order_acquire) || !cell->successful ||
        !cell->released) {
      return;
    }
    if (isPacingLocked()) {
      // WINDOW_AIMD's paced recovery while the window is below one unit.
      applySuccessLocked(1, 0, /*attributed=*/true);
    } else if (!cell->probe) {
      creditBaselineSuccessLocked(1);
    }
    timer = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timer) {
    schedulePacingTimer(*timer);
  }
}

void RPCRateLimiter::releaseProbingLocked(uint64_t admissionId) {
  const auto position = probing_->receipts.find(admissionId);
  if (position == probing_->receipts.end()) {
    return;
  }
  position->second->released = true;
  probing_->receipts.erase(position);
  if (probing_->probe == admissionId) {
    probing_->probe.reset();
    probing_->probeFailed = false;
  }
}

void RPCRateLimiter::settleProbingAttempt(
    std::vector<Token> tokens,
    OutputCredit credit,
    Outcome outcome,
    bool validServiceProof,
    bool cancelled,
    bool nonCapacity,
    AdmissionLease::Completion& completion) {
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timer;
  {
    std::lock_guard<std::mutex> l(mutex_);
    completion.overloadEpoch =
        tokens.empty() ? overloadEpoch_ : tokens.front().overloadEpoch();
    const bool fresh = completion.overloadEpoch == overloadEpoch_;
    for (auto& token : tokens) {
      --pending_;
      token.disarm(tokenConstructionKey_);
    }
    // The tokens no longer release their receipts, so feedback that throws
    // must not leave a receipt, or the probe slot, held.
    const auto releaseReceipts = [&] {
      for (const auto& token : tokens) {
        releaseProbingLocked(token.admissionId());
      }
    };
    auto releaseOnThrow = folly::makeGuard(releaseReceipts);
    const auto now = now_();
    const auto& cells = credit.cells(tokenConstructionKey_);
    bool promote{false};
    bool live{false};
    for (const auto& cell : cells) {
      const bool cellCancelled =
          cancelled || cell->cancelled.load(std::memory_order_acquire);
      cell->successful = !cellCancelled && outcome == Outcome::kSuccess;
      if (cellCancelled) {
        continue;
      }
      live = true;
      if (outcome == Outcome::kSuccess && cell->probe) {
        // The probe was admitted while the baseline was full, so its success
        // shows the backend accepted one unit more than the baseline.
        promote = usesProbingPolicyLocked() &&
            probing_->probe == cell->admissionId && !probing_->probeFailed &&
            cell->lifecycle == probing_->lifecycle &&
            cell->level == capacityLocked() + 1;
      }
      // The sample spans the whole single-unit lease, from its charge to this
      // settlement, so it includes time held outside backend service.
      int64_t ticks;
      if (outcome == Outcome::kSuccess && validServiceProof && !cell->probe &&
          tokens.size() == 1 &&
          !subWithOverflow(
              now.time_since_epoch().count(),
              cell->admittedAt.time_since_epoch().count(),
              &ticks) &&
          ticks > 0) {
        updateServiceHorizonLocked(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::duration{ticks})
                .count());
      }
    }
    if (outcome == Outcome::kOverload) {
      completion.sharedOverloadHandled = true;
      // A probe holds one physical slot, so it explains at most one capacity
      // overload while it is in flight: its own or another unit's. Overload
      // from before the last cut is stale, and a rejection not shown to be a
      // lack of capacity, such as a quota, says nothing about the probe. Any
      // other overload cuts as WINDOW does.
      const bool chargeProbe = live && fresh && !nonCapacity && probing_->cut &&
          probing_->probe.has_value() && !probing_->probeFailed;
      if (chargeProbe) {
        failProbeLocked();
        completion.disposition = FeedbackDisposition::kExclusivelySpeculative;
      } else if (live && fresh) {
        ++overloadEpoch_;
        applyOverloadLocked(nonCapacity);
        completion.disposition = FeedbackDisposition::kCanonical;
      } else {
        completion.disposition = fresh ? FeedbackDisposition::kCanonical
                                       : FeedbackDisposition::kStale;
      }
    } else {
      completion.disposition =
          fresh ? FeedbackDisposition::kCanonical : FeedbackDisposition::kStale;
    }
    if (promote) {
      promoteProbeLocked();
    }
    releaseOnThrow.dismiss();
    releaseReceipts();
    if (!cancelled && outcome == Outcome::kSuccess) {
      completion.credit = std::move(credit);
    }
    timer = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timer) {
    schedulePacingTimer(*timer);
  }
}

RPCRateLimiter::TestingProbingSnapshot RPCRateLimiter::testingProbingSnapshot()
    const {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(probing_);
  size_t probes{0};
  for (const auto& [_, cell] : probing_->receipts) {
    probes += cell->probe;
  }
  const auto next = probing_->failures.find(capacityLocked() + 1);
  return {
      .cut = probing_->cut,
      .baseline = capacityLocked(),
      .retryDebt = probing_->retryDebt,
      .nextLevelFailures =
          next == probing_->failures.end() ? 0 : next->second.count,
      .held = probing_->receipts.size(),
      .heldProbes = probes,
  };
}

std::vector<RPCRateLimiter::Token> RPCRateLimiter::tryAcquireUpTo(
    int64_t want) {
  std::vector<Token> granted;
  if (want <= 0) {
    return granted;
  }

  std::unique_lock<std::mutex> l(mutex_);
  const auto now = now_();
  if (usesProbingStartupLocked()) {
    // Interleave ready feedback between chunks without limiting the healthy
    // aggregate window or introducing a startup timer.
    want = std::min(want, kStartupAcquisitionCap);
  }
  if (isProbingLocked()) {
    // One unit per call, so a rejection completed inline by the caller is
    // seen before the next unit is admitted.
    return tryAcquireProbingLocked(l, std::min<int64_t>(want, 1));
  }
  if (isPacingLocked()) {
    if (pending_.load() > 0 || now < nextPacedAdmission_) {
      return granted;
    }
    commitGrowthLocked();
    const int64_t pending = ++pending_;
    noteStartupPeakLocked(pending);
    ++pacingGeneration_;
    pacingTimerArmed_ = false;
    lastAdmission_ = now;
    nextPacedAdmission_ = isPacingLocked()
        ? pacingDeadline(now, nextPacedDelayLocked())
        : std::chrono::steady_clock::time_point{};
    notePeakPending(pending);
    granted.emplace_back(
        tokenConstructionKey_, this, overloadEpoch_, 0, AdmissionKind::kPlain);
    return granted;
  }
  const int64_t capacity = capacityLocked();

  // One compare-exchange for the whole grant rather than a read followed by an
  // increment, so a release racing this locked capacity decision is reflected
  // without allowing two callers to take the last slot.
  int64_t pending = pending_.load();
  int64_t take = 0;
  do {
    take = std::min<int64_t>(want, std::max<int64_t>(0, capacity - pending));
    if (take == 0) {
      return granted;
    }
    // compare_exchange_weak refreshes 'pending' on failure, so a caller that
    // loses the exchange recomputes its grant against the winner's value
    // rather than its own stale read.
  } while (!pending_.compare_exchange_weak(pending, pending + take));
  noteStartupPeakLocked(pending + take);
  lastAdmission_ = now;

  // The post-exchange total, not the caller's pre-read: reporting the stale
  // value would hide exactly the overshoot this loop exists to prevent, and
  // the contention tests assert on this counter.
  notePeakPending(pending + take);
  granted.reserve(static_cast<size_t>(take));
  for (int64_t i = 0; i < take; ++i) {
    // One token per slot: each releases exactly one on destruction, so the
    // bulk grant unwinds row by row as the requests complete.
    granted.emplace_back(
        tokenConstructionKey_, this, overloadEpoch_, 0, AdmissionKind::kPlain);
  }
  return granted;
}

bool RPCRateLimiter::canHandleRetryBackoff() const {
  std::lock_guard<std::mutex> l(mutex_);
  return config_.adaptive &&
      (isPacingLocked() ||
       limitLocked() > floorLocked(static_cast<double>(ceilingLocked())));
}

bool RPCRateLimiter::usesAttributedAdmissionControl() const {
  std::lock_guard<std::mutex> l(mutex_);
  return isProbingLocked();
}

int64_t RPCRateLimiter::available() const {
  std::lock_guard<std::mutex> l(mutex_);
  if (isPacingLocked()) {
    return pending_.load() == 0 && now_() >= nextPacedAdmission_ ? 1 : 0;
  }
  const auto capacity =
      isProbingLocked() ? probingAdmissionCapacityLocked() : capacityLocked();
  return std::max<int64_t>(0, capacity - pending_.load());
}

RPCRateLimiter::RunnableDemand::RunnableDemand(
    std::weak_ptr<RPCRateLimiter> owner,
    uint64_t id)
    : owner_{std::move(owner)}, id_{id} {}

RPCRateLimiter::RunnableDemand::~RunnableDemand() {
  if (const auto owner = owner_.lock()) {
    owner->cancelRunnableDemand(id_);
  }
}

std::shared_ptr<RPCRateLimiter::RunnableDemand>
RPCRateLimiter::registerRunnableDemand() {
  std::shared_ptr<RunnableDemand> result;
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timer;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (!usesProbingPolicyLocked()) {
      return nullptr;
    }
    auto& demands = probing_->demands;
    auto& nextDemandId = probing_->nextDemandId;
    const auto owner = weak_from_this();
    VELOX_CHECK(!owner.expired(), "Runnable demand requires shared ownership");
    VELOX_CHECK_LT(nextDemandId, std::numeric_limits<uint64_t>::max());
    const auto id = ++nextDemandId;
    demands.push_back(id);
    try {
      result = std::make_shared<RunnableDemand>(owner, id);
    } catch (...) {
      std::erase(demands, id);
      throw;
    }
    timer = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timer) {
    schedulePacingTimer(*timer);
  }
  return result;
}

void RPCRateLimiter::cancelRunnableDemand(uint64_t id) noexcept {
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timer;
  try {
    std::lock_guard<std::mutex> l(mutex_);
    if (probing_) {
      std::erase(probing_->demands, id);
    }
    if (usesProbingPolicyLocked()) {
      timer = collectWakeableLocked(toNotify);
    }
  } catch (...) {
    logLimiterFailure(admissionKey_, "runnable demand cancellation");
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timer) {
    schedulePacingTimer(*timer);
  }
}

RPCRateLimiter::Admission RPCRateLimiter::admitOrWait() {
  Admission result;
  std::optional<PacingTimerRequest> timerRequest;
  {
    std::lock_guard<std::mutex> l(mutex_);
    const int64_t pending = pending_.load();
    const int64_t capacity =
        isProbingLocked() ? probingAdmissionCapacityLocked() : capacityLocked();
    if (isPacingLocked()) {
      if (pending == 0 && now_() >= nextPacedAdmission_) {
        result.admitted = true;
        return result;
      }
      waiters_.emplace_back("RPCRateLimiter::admitOrWait");
      result.wait = waiters_.back().getSemiFuture();
      timerRequest = preparePacingTimerLocked();
    } else if (pending < capacity) {
      RPC_RATE_LIMITER_VLOG(2) << "admitOrWait[" << admissionKey_
                               << "]: admitted (pending=" << pending
                               << ", capacity=" << capacity << ")";
      result.admitted = true;
      return result;
    } else {
      RPC_RATE_LIMITER_VLOG(1)
          << "admitOrWait[" << admissionKey_
          << "]: waiting (pending=" << pending << ", capacity=" << capacity
          << "), waiter #" << waiters_.size();
      // Enrolled under the same lock that decided, so any release from here on
      // must see this waiter.
      waiters_.emplace_back("RPCRateLimiter::admitOrWait");
      result.wait = waiters_.back().getSemiFuture();
    }
  }
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
  return result;
}

void RPCRateLimiter::onOutcome(Outcome outcome, int64_t units) {
  onOutcome(outcome, units, 0);
}

void RPCRateLimiter::onOutcome(Outcome outcome, int64_t units, int64_t rttNs) {
  switch (outcome) {
    case Outcome::kOverload:
      onOverload();
      return;
    case Outcome::kSuccess:
      onSuccess(units, rttNs);
      return;
    case Outcome::kFailure:
      onFailure();
      return;
    case Outcome::kNone:
      return;
  }
}

void RPCRateLimiter::applyOverloadLocked(bool nonCapacity) {
  const double rejectedWindow =
      !nonCapacity && usesProbingStartupLocked() && startupPeak_ > 0
      ? std::min(limitLocked(), static_cast<double>(startupPeak_))
      : limitLocked();
  const auto failedAllowance = boundedCapacity(rejectedWindow, ceilingLocked());
  // Cold-start authorization is not exercised capacity. Back off the charged
  // allowance using the same decrease factor, not an unused bootstrap ceiling.
  decreaseLimitLocked(rejectedWindow * config_.decreaseFactor);
  // A rejection not shown to be a lack of capacity says nothing about the
  // window's levels: before the first capacity cut it leaves SMART_PROBING
  // identical to WINDOW_AIMD, and afterwards it records no level failure.
  if (!usesProbingPolicyLocked() || (nonCapacity && !probing_->cut)) {
    return;
  }
  probing_->cut = true;
  if (probing_->probe.has_value()) {
    probing_->probeFailed = true;
  }
  if (!nonCapacity) {
    recordLevelFailureLocked(failedAllowance);
  }
  probing_->retryDebt = nextRetryDebtLocked();
}

void RPCRateLimiter::decreaseLimitLocked(double target) {
  if (!config_.adaptive) {
    return;
  }
  // Cut the window committed when the rejected call was admitted. Growth
  // projected while it was in flight is discarded: a rejection that arrives
  // late, as timeouts and transport-retried rate limits do, must still cut.
  recovering_ = false;
  const double current = limitLocked();
  const double next = std::max(
      floorLocked(static_cast<double>(ceilingLocked())),
      std::min(current, target));
  // Reschedule even when the window is already at its floor, so the next
  // probe waits the back-off interval rather than the recovery one.
  setLimitLocked(next);
  if (next < current) {
    if (lowWaterLimit_ == 0.0 || next < lowWaterLimit_) {
      lowWaterLimit_ = next;
    }
    RPC_RATE_LIMITER_VLOG(1)
        << "RPC congestion: limit[" << admissionKey_ << "] " << current
        << " -> " << next << " (overload)";
  }
}

void RPCRateLimiter::onOverload() {
  applyOverload(/*nonCapacity=*/false);
}

void RPCRateLimiter::onUnattributedOverload() {
  applyOverload(/*nonCapacity=*/true);
}

void RPCRateLimiter::applyOverload(bool nonCapacity) {
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  {
    std::lock_guard<std::mutex> l(mutex_);
    ++overloadEpoch_;
    applyOverloadLocked(nonCapacity);
    timerRequest = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
}

bool RPCRateLimiter::updateServiceHorizonLocked(int64_t rttNs) {
  if (rttNs <= 0) {
    return false;
  }
  accrueLocked();
  const int64_t previous = serviceHorizonNanos_;
  if (!hasServiceHorizonSample_) {
    serviceHorizonNanos_ = rttNs;
    hasServiceHorizonSample_ = true;
    return serviceHorizonNanos_ != previous;
  }
  if (rttNs >= previous) {
    serviceHorizonNanos_ =
        previous + (rttNs - previous) / kServiceHorizonEwmaWeight;
  } else {
    const auto difference = previous - rttNs;
    const auto decrease = difference / kServiceHorizonEwmaWeight +
        (difference % kServiceHorizonEwmaWeight != 0);
    serviceHorizonNanos_ = previous - decrease;
  }
  return serviceHorizonNanos_ != previous;
}

bool RPCRateLimiter::successEarnsCreditLocked() const {
  return config_.adaptive && limit_ > 0.0 &&
      limitLocked() < static_cast<double>(ceilingLocked());
}

void RPCRateLimiter::applySuccessLocked(
    int64_t units,
    int64_t rttNs,
    bool attributed) {
  if (isProbingLocked() && !attributed) {
    return;
  }
  if (isPacingLocked()) {
    // Growth earned so far is measured on the old horizon; commit it before
    // the sample moves the horizon. Rescheduling restarts growth from now.
    // Committed growth may already have ended pacing.
    commitGrowthLocked();
    updateServiceHorizonLocked(rttNs);
    if (isPacingLocked()) {
      recovering_ = true;
      setLimitLocked(limitLocked());
    }
    return;
  }
  updateServiceHorizonLocked(rttNs);
  if (units <= 0 || !config_.adaptive || limit_ <= 0.0) {
    return;
  }
  const double current = limitLocked();
  const double next =
      std::sqrt(current * current + 2.0 * static_cast<double>(units));
  setLimitLocked(next >= static_cast<double>(ceilingLocked()) ? 0.0 : next);
}

void RPCRateLimiter::onFailure() {
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (!config_.adaptive || !recovering_) {
      return;
    }
    // Growth since the failed call's admission is not committed: the backend
    // did not serve it.
    recovering_ = false;
    setLimitLocked(limitLocked());
    timerRequest = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
}

void RPCRateLimiter::onSuccess(int64_t units, int64_t rttNs) {
  if (units <= 0) {
    return;
  }
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  {
    std::lock_guard<std::mutex> l(mutex_);
    applySuccessLocked(units, rttNs, /*attributed=*/false);
    timerRequest = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
}

bool RPCRateLimiter::onSuccessForEpoch(
    uint64_t overloadEpoch,
    int64_t units,
    int64_t rttNs) {
  if (units <= 0) {
    return true;
  }
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  bool discarded{false};
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (isProbingLocked()) {
      // Units admitted before the first overload carry plain tokens; their
      // successes still count as baseline work for probe spacing.
      if (!isPacingLocked()) {
        creditBaselineSuccessLocked(units);
      }
      discarded = overloadEpoch != overloadEpoch_;
    } else if (overloadEpoch != overloadEpoch_) {
      commitGrowthLocked();
      const bool horizonMoved = updateServiceHorizonLocked(rttNs);
      discarded = successEarnsCreditLocked();
      if (horizonMoved && isPacingLocked()) {
        setLimitLocked(limitLocked());
      }
    } else {
      applySuccessLocked(units, rttNs, /*attributed=*/false);
    }
    timerRequest = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
  return !discarded;
}

void RPCRateLimiter::release(
    uint64_t admissionId,
    AdmissionKind kind) noexcept {
  if (kind == AdmissionKind::kProbing) {
    std::vector<ContinuePromise> toNotify;
    std::optional<PacingTimerRequest> timer;
    try {
      std::lock_guard<std::mutex> l(mutex_);
      --pending_;
      if (probing_) {
        releaseProbingLocked(admissionId);
      }
      timer = collectWakeableLocked(toNotify);
    } catch (...) {
      logLimiterFailure(admissionKey_, "probing rollback release");
    }
    fulfillWaitersNoThrow(toNotify, admissionKey_);
    if (timer) {
      schedulePacingTimer(*timer);
    }
    return;
  }
  // Saturate at zero. A token can outlive testingReset(), which zeroes the
  // count; without the floor its release would drive pending_ negative and
  // make available() report more capacity than exists.
  int64_t pending = pending_.load();
  while (pending > 0 && !pending_.compare_exchange_weak(pending, pending - 1)) {
  }
  pending = std::max<int64_t>(0, pending - 1);
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  try {
    RPC_RATE_LIMITER_VLOG(2)
        << "release[" << admissionKey_ << "]: pending=" << pending;
    common::testutil::TestValue::adjust(
        "facebook::velox::exec::rpc::RPCRateLimiter::release", this);

    // Wake every waiter once admission is possible. Reservations remain
    // atomic, so abandoned futures cannot strand available capacity.
    std::lock_guard<std::mutex> l(mutex_);
    timerRequest = collectWakeableLocked(toNotify);
  } catch (...) {
    logLimiterFailure(admissionKey_, "waiter selection");
    return;
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
}

RPCRateLimiter::Stats RPCRateLimiter::stats() const {
  std::lock_guard<std::mutex> l(mutex_);
  const auto now = now_();
  const auto sinceAccrual = now - lastAccrual_;
  const double currentLimit = limitLocked();
  const double lowWaterLimit =
      lowWaterLimit_ > 0.0 ? lowWaterLimit_ : currentLimit;
  const auto pacingTime =
      isPacingLocked() ? pacingTime_ + sinceAccrual : pacingTime_;
  return Stats{
      .capacity = capacityLocked(),
      .hardLimit = ceilingLocked(),
      .pending = pending_.load(),
      .peakPending = peakPending_.load(),
      .lowWaterCapacity = boundedCapacity(lowWaterLimit, ceilingLocked()),
      .limitMilli = scaledMetric(currentLimit * 1'000),
      .lowWaterLimitMilli = scaledMetric(lowWaterLimit * 1'000),
      .serviceHorizonNanos = serviceHorizonNanos_,
      .recovering = recovering_,
      .pacingNanos =
          std::chrono::duration_cast<std::chrono::nanoseconds>(pacingTime)
              .count(),
      .windowUnitSeconds = windowUnitSeconds_ +
          currentLimit * std::chrono::duration<double>(sinceAccrual).count(),
      .clockNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        now.time_since_epoch())
                        .count(),
      .admissionWaiters = static_cast<int64_t>(waiters_.size()),
      .admissionTimerArmed = pacingTimerArmed_,
      .nextPermitDelayNanos =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::max(
                  nextPacedAdmission_ - now,
                  std::chrono::steady_clock::duration::zero()))
              .count(),
      .flowControlPolicy = config_.flowControlPolicy,
  };
}

// --- RPCRateLimiterRegistry ---

RPCRateLimiterRegistry& RPCRateLimiterRegistry::global() {
  // Intentionally leaked so registry-backed limiters remain valid throughout
  // process teardown.
  static auto* registry = new RPCRateLimiterRegistry();
  return *registry;
}

RPCRateLimiter& RPCRateLimiterRegistry::get(const std::string& admissionKey) {
  // Fast path: the backend already exists, so concurrent lookups from every
  // driver share the lock rather than serializing. Shared ownership keeps the
  // reference valid across later insertions and completion leases.
  {
    std::shared_lock<std::shared_mutex> rl(mutex_);
    auto it = backends_.find(admissionKey);
    if (it != backends_.end()) {
      return *it->second;
    }
  }
  // Slow path: first sight of this backend. Re-check under the exclusive lock
  // in case another thread created it between the two locks.
  std::unique_lock<std::shared_mutex> wl(mutex_);
  auto it = backends_.find(admissionKey);
  if (it != backends_.end()) {
    return *it->second;
  }
  auto [inserted, _] = backends_.emplace(
      admissionKey, std::make_shared<RPCRateLimiter>(admissionKey));
  return *inserted->second;
}

RPCRateLimiterRegistry::Lookup RPCRateLimiterRegistry::getForPolicy(
    std::string_view admissionKey,
    const RPCRateLimiter::Config& policy) {
  const std::string key{admissionKey};
  auto& shared = get(key);
  if (shared.initializeOrMatch(policy)) {
    return {.limiter = &shared, .policyScoped = false};
  }

  const auto canonical = canonicalize(policy);
  // Caller holds mutex_, shared or exclusive.
  auto findLocked = [&]() -> RPCRateLimiter* {
    const auto it = policyScopedLimiters_.find(key);
    if (it == policyScopedLimiters_.end()) {
      return nullptr;
    }
    for (const auto& [scopedPolicy, limiter] : it->second) {
      if (scopedPolicy == canonical) {
        return limiter.get();
      }
    }
    return nullptr;
  };
  {
    std::shared_lock<std::shared_mutex> rl(mutex_);
    if (auto* limiter = findLocked()) {
      return {.limiter = limiter, .policyScoped = true};
    }
  }
  common::testutil::TestValue::adjust(
      "facebook::velox::exec::rpc::RPCRateLimiterRegistry::getForPolicy", this);

  // Sealed before it is published, so no other query can reach it unsealed.
  // Discarded unused if another thread publishes one for this policy first.
  auto created = std::make_shared<RPCRateLimiter>(key + " (policy-scoped)");
  const bool sealed = created->initializeOrMatch(policy);
  VELOX_CHECK(sealed, "A new rate limiter refused its first policy: {}", key);

  size_t numPolicies{0};
  {
    std::unique_lock<std::shared_mutex> wl(mutex_);
    if (auto* limiter = findLocked()) {
      return {.limiter = limiter, .policyScoped = true};
    }
    auto& limiters = policyScopedLimiters_[key];
    limiters.emplace_back(canonical, created);
    // The shared limiter runs one more policy.
    numPolicies = limiters.size() + 1;
  }
  const auto sealedPolicy = created->config();
  RPC_RATE_LIMITER_LOG(INFO)
      << "admission key '" << key << "' runs " << numPolicies
      << " rate-limiter policies on this worker; added adaptive="
      << (sealedPolicy.adaptive ? "true" : "false")
      << ", ceiling=" << sealedPolicy.ceiling
      << ", floor=" << sealedPolicy.floor
      << ", decrease=" << sealedPolicy.decreaseFactor;
  return {.limiter = created.get(), .policyScoped = true};
}

void RPCRateLimiterRegistry::testingReset() {
  std::unique_lock<std::shared_mutex> wl(mutex_);
  defaultCapacityRef().store(20);
  // Reset each backend in place rather than dropping it. Tokens outlive the
  // operator that acquired them and release through a back-pointer, so
  // destroying a RPCRateLimiter that still has outstanding tokens would leave
  // them writing through a dangling pointer.
  for (auto& [admissionKey, admission] : backends_) {
    admission->testingReset();
  }
  // A policy-scoped limiter stays listed under its policy, so it is sealed
  // with the policy it ran before the reset.
  for (auto& [admissionKey, limiters] : policyScopedLimiters_) {
    for (auto& entry : limiters) {
      auto& limiter = entry.second;
      const auto sealedPolicy = limiter->config();
      limiter->testingReset();
      limiter->initializeOrMatch(sealedPolicy);
    }
  }
}

} // namespace facebook::velox::exec::rpc
