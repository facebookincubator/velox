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
#include <numbers>
#include <string_view>
#include <utility>
#include <vector>

#include <folly/Random.h>
#include <folly/executors/InlineExecutor.h>
#include <folly/futures/Future.h>

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

// Highest floor the probe interval may impose. A floor of one would end pacing,
// letting probes go out as fast as rejections return; a backend whose calls
// outlast the probe interval is probed every two horizons instead.
constexpr double kMaxProbeFloor{0.5};

// Largest fraction by which a back-off probe comes early. Probes are never
// later than the interval, so the maximum probe interval still holds.
constexpr double kProbeJitter{0.2};

// Actual-admission measurements follow the AWS SDK adaptive-retry window and
// EWMA weight. Recovery is supplied by the selected sending-rate policy.
constexpr double kAdmissionRateWeight{0.8};
constexpr std::chrono::milliseconds kAdmissionRateWindow{500};

constexpr int64_t kServiceHorizonEwmaWeight{8};
constexpr double kInitialAdaptiveLimit{200.0};

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
      owner_->release();
    }
    owner_ = other.owner_;
    overloadEpoch_ = other.overloadEpoch_;
    other.owner_ = nullptr;
  }
  return *this;
}

RPCRateLimiter::Token::~Token() noexcept {
  if (owner_ != nullptr) {
    owner_->release();
  }
}

// --- AdmissionLease ---

RPCRateLimiter::AdmissionLease::AdmissionLease(
    std::shared_ptr<RPCRateLimiter> owner,
    std::vector<Token> tokens,
    uint64_t overloadEpoch)
    : owner_{std::move(owner)},
      tokens_{std::move(tokens)},
      overloadEpoch_{overloadEpoch} {}

RPCRateLimiter::AdmissionLease::~AdmissionLease() {
  release();
}

RPCRateLimiter::AdmissionLease::Completion
RPCRateLimiter::AdmissionLease::completeAndRelease(bool hardOverload) noexcept {
  return completeAndRelease(hardOverload, /*handedOff=*/false);
}

RPCRateLimiter::AdmissionLease::Completion
RPCRateLimiter::AdmissionLease::completeAndRelease(
    bool hardOverload,
    bool handedOff) noexcept {
  return completeAndRelease(
      hardOverload ? Outcome::kOverload : Outcome::kNone,
      /*rttNs=*/0,
      handedOff);
}

RPCRateLimiter::AdmissionLease::Completion
RPCRateLimiter::AdmissionLease::completeAndRelease(
    Outcome outcome,
    int64_t rttNs,
    bool handedOff) noexcept {
  std::shared_ptr<RPCRateLimiter> owner;
  std::vector<Token> tokens;
  try {
    std::lock_guard<std::mutex> l(mutex_);
    if (completed_) {
      return {};
    }
    completed_ = true;
    owner = std::move(owner_);
    tokens = std::move(tokens_);
  } catch (...) {
    logLimiterFailure("admission lease", "completion");
    return {};
  }

  bool handled = false;
  if (!tokens.empty() && outcome != Outcome::kNone) {
    try {
      handled = owner->recordCompletionForEpoch(
          overloadEpoch_,
          static_cast<int64_t>(tokens.size()),
          outcome,
          rttNs,
          handedOff);
    } catch (...) {
      logLimiterFailure("admission lease", "completion feedback");
    }
  }
  tokens.clear();
  return Completion{
      .sharedOverloadHandled = handled, .overloadEpoch = overloadEpoch_};
}

void RPCRateLimiter::AdmissionLease::release() noexcept {
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

bool RPCRateLimiter::isRatePacingLocked() const {
  return config_.adaptive &&
      config_.ratePolicy != RPCFlowControlPolicy::kWindowAimd &&
      sendRatePerSecond_ > 0.0;
}

std::chrono::steady_clock::duration RPCRateLimiter::ratePacingIntervalLocked()
    const {
  return std::chrono::ceil<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(1.0 / sendRatePerSecond_));
}

void RPCRateLimiter::accrueRateDemandLocked(
    std::chrono::steady_clock::time_point now) {
  if (config_.ratePolicy == RPCFlowControlPolicy::kWindowAimd) {
    return;
  }
  if (rateDemandBlocked_ && now > rateDemandUpdatedAt_) {
    rateWaitingTime_ += now - rateDemandUpdatedAt_;
    cubicDemandSeconds_ +=
        std::chrono::duration<double>(now - rateDemandUpdatedAt_).count();
  }
  rateDemandUpdatedAt_ = now;
}

void RPCRateLimiter::setRateDemandBlockedLocked(bool blocked) {
  if (config_.ratePolicy == RPCFlowControlPolicy::kWindowAimd) {
    return;
  }
  accrueRateDemandLocked(now_());
  rateDemandBlocked_ = blocked;
}

void RPCRateLimiter::setSendRateLocked(double requestsPerSecond) {
  accrueLocked();
  const double minRate = config_.maxProbeInterval.count() > 0
      ? 1.0 / std::chrono::duration<double>(config_.maxProbeInterval).count()
      : floorLocked(static_cast<double>(ceilingLocked())) /
          std::chrono::duration<double>(
              std::chrono::nanoseconds(serviceHorizonNanos_))
              .count();
  sendRatePerSecond_ = std::max(minRate, requestsPerSecond);
  ++pacingGeneration_;
  pacingTimerArmed_ = false;
  const auto now = now_();
  nextPacedAdmission_ = lastAdmission_.has_value()
      ? std::max(
            now,
            lastAdmission_.value() +
                (isPacingLocked()
                     ? std::max(
                           ratePacingIntervalLocked(), nextPacedDelayLocked())
                     : ratePacingIntervalLocked()))
      : now + ratePacingIntervalLocked();
}

void RPCRateLimiter::noteAdmissionLocked(
    int64_t units,
    std::chrono::steady_clock::time_point now) {
  if (config_.ratePolicy == RPCFlowControlPolicy::kWindowAimd) {
    return;
  }
  admittedAttempts_ += units;
  if (!admissionWindowStart_.has_value()) {
    admissionWindowStart_ = now;
  }
  admissionWindowUnits_ += units;
  const auto elapsed = now - admissionWindowStart_.value();
  if (elapsed >= kAdmissionRateWindow) {
    const auto rate = static_cast<double>(admissionWindowUnits_) /
        std::chrono::duration<double>(elapsed).count();
    measuredAdmissionRate_ = measuredAdmissionRate_ > 0
        ? kAdmissionRateWeight * rate +
            (1.0 - kAdmissionRateWeight) * measuredAdmissionRate_
        : rate;
    ++admissionRateBucket_;
    admissionWindowStart_ = now;
    admissionWindowUnits_ = 0;
  }
}

void RPCRateLimiter::updateSendRateLocked(
    uint64_t overloadEpoch,
    Outcome outcome,
    std::chrono::steady_clock::time_point now) {
  accrueRateDemandLocked(now);
  if (outcome == Outcome::kOverload) {
    if (isRatePacingLocked() &&
        (overloadEpoch != overloadEpoch_ ||
         (lastRateDecreaseBucket_.has_value() &&
          lastRateDecreaseBucket_.value() == admissionRateBucket_))) {
      return;
    }
    // A short admission burst is measured against a full bucket interval,
    // never a tiny elapsed time. Sparse requests do not lower the exploration
    // rate permitted by the independent concurrency guard.
    const double exploration = static_cast<double>(capacityLocked()) /
        std::chrono::duration<double>(
            std::chrono::nanoseconds(serviceHorizonNanos_))
            .count();
    const auto observation = admissionWindowStart_.has_value()
        ? std::max(
              now - admissionWindowStart_.value(),
              std::chrono::steady_clock::duration(kAdmissionRateWindow))
        : std::chrono::steady_clock::duration(kAdmissionRateWindow);
    const double partialAdmissionRate =
        static_cast<double>(admissionWindowUnits_) /
        std::chrono::duration<double>(observation).count();
    const double priorRate = sendRatePerSecond_ > 0.0
        ? sendRatePerSecond_
        : std::max({exploration, measuredAdmissionRate_, partialAdmissionRate});
    priorMaxSendRate_ = priorRate;
    lastRateDecreaseBucket_ = admissionRateBucket_;
    cubicDemandSeconds_ = 0.0;
    ++overloadEpoch_;
    ++rateDecreases_;
    setSendRateLocked(rpcFlowControlBackoffRate(priorRate));
    return;
  }
  if (outcome != Outcome::kSuccess || !isRatePacingLocked() ||
      overloadEpoch != overloadEpoch_) {
    return;
  }
  const double horizonSeconds =
      std::chrono::duration<double>(
          std::chrono::nanoseconds(serviceHorizonNanos_))
          .count();
  const double target = rpcFlowControlRecoveryRate(
      config_.ratePolicy,
      priorMaxSendRate_,
      cubicDemandSeconds_,
      horizonSeconds);
  if (target > sendRatePerSecond_) {
    setSendRateLocked(target);
  }
}

bool RPCRateLimiter::recordCompletionForEpoch(
    uint64_t overloadEpoch,
    int64_t units,
    Outcome outcome,
    int64_t rttNs,
    bool handedOff) {
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  bool handled{false};
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (config_.ratePolicy == RPCFlowControlPolicy::kWindowAimd) {
      if (outcome != Outcome::kOverload) {
        return false;
      }
      if (overloadEpoch != overloadEpoch_) {
        return true;
      }
      handled = true;
      ++overloadEpoch_;
      applyOverloadLocked();
    } else {
      if (outcome != Outcome::kNone) {
        completionAttempts_ += units;
      }
      if (outcome == Outcome::kSuccess) {
        successfulAttempts_ += units;
        updateServiceHorizonLocked(rttNs);
      } else if (outcome == Outcome::kOverload) {
        overloadAttempts_ += units;
      }
      const auto now = now_();
      if (outcome == Outcome::kSuccess || handedOff) {
        updateSendRateLocked(overloadEpoch, outcome, now);
        handled = outcome == Outcome::kOverload;
      } else if (outcome == Outcome::kOverload) {
        handled = true;
        if (overloadEpoch == overloadEpoch_) {
          ++overloadEpoch_;
          applyOverloadLocked();
        }
      }
    }
    timerRequest = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
  return handled;
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
  return std::chrono::ceil<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::nano>{
          static_cast<double>(serviceHorizonNanos_) * horizons});
}

std::chrono::steady_clock::duration RPCRateLimiter::nextPacedDelayLocked()
    const {
  const auto interval = pacingIntervalLocked();
  if (recovering_) {
    return interval;
  }
  const double fraction = 1.0 - kProbeJitter * (1.0 - random_());
  return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      interval * fraction);
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
  if (isRatePacingLocked()) {
    ratePacingTime_ += elapsed;
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
    const auto delay = isRatePacingLocked()
        ? std::max(nextPacedDelayLocked(), ratePacingIntervalLocked())
        : nextPacedDelayLocked();
    nextPacedAdmission_ = lastAdmission_.has_value()
        ? std::max(now, lastAdmission_.value() + delay)
        : now + delay;
  } else {
    recovering_ = false;
    nextPacedAdmission_ = isRatePacingLocked() && lastAdmission_.has_value()
        ? std::max(now, lastAdmission_.value() + ratePacingIntervalLocked())
        : std::chrono::steady_clock::time_point{};
  }
}

int64_t RPCRateLimiter::capacityLocked() const {
  const int64_t ceiling = ceilingLocked();
  return std::clamp<int64_t>(
      static_cast<int64_t>(std::floor(limitLocked())), 1, ceiling);
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
  if (config.ratePolicy != RPCFlowControlPolicy::kWindowAimd) {
    config.scheduleFixes = true;
  }
  if (!config.adaptive) {
    config.ratePolicy = RPCFlowControlPolicy::kWindowAimd;
  }
  return config;
}

// Reduces 'policy' to the fields a limiter reads, so policies that differ only
// in fields it ignores compare equal.
RPCRateLimiter::Config canonicalize(const RPCRateLimiter::Config& policy) {
  auto canonical = clamped(policy);
  if (canonical.ratePolicy == RPCFlowControlPolicy::kWindowAimd &&
      canonical.ceiling > 0 &&
      canonical.floor >= static_cast<double>(canonical.ceiling)) {
    canonical.adaptive = false;
  }
  if (!canonical.adaptive) {
    const RPCRateLimiter::Config defaults;
    canonical.floor = defaults.floor;
    canonical.decreaseFactor = defaults.decreaseFactor;
    canonical.maxProbeInterval = defaults.maxProbeInterval;
    canonical.ratePolicy = RPCFlowControlPolicy::kWindowAimd;
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
  ratePolicySelected_.store(false, std::memory_order_relaxed);
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
  nextPacedAdmission_ = std::chrono::steady_clock::time_point{};
  lastAdmission_.reset();
  sendRatePerSecond_ = 0.0;
  priorMaxSendRate_ = 0.0;
  lastRateDecreaseBucket_.reset();
  cubicDemandSeconds_ = 0.0;
  rateDemandBlocked_ = false;
  rateDemandUpdatedAt_ = now_();
  admissionWindowUnits_ = 0;
  measuredAdmissionRate_ = 0.0;
  admissionWindowStart_.reset();
  admissionRateBucket_ = 0;
  admittedAttempts_ = 0;
  completionAttempts_ = 0;
  successfulAttempts_ = 0;
  overloadAttempts_ = 0;
  rateDecreases_ = 0;
  ratePacingTime_ = std::chrono::steady_clock::duration::zero();
  rateWaitingTime_ = std::chrono::steady_clock::duration::zero();
  ++pacingGeneration_;
  ++overloadEpoch_;
  pacingTimerArmed_ = false;
  pending_.store(0);
  peakPending_.store(0);
  waiters_.clear();
}

std::optional<RPCRateLimiter::PacingTimerRequest>
RPCRateLimiter::collectWakeableLocked(std::vector<ContinuePromise>& toNotify) {
  if (waiters_.empty()) {
    setRateDemandBlockedLocked(false);
  }
  if (isPacingLocked()) {
    setRateDemandBlockedLocked(false);
    if (pending_.load() != 0 || waiters_.empty()) {
      return std::nullopt;
    }
    if (now_() < nextPacedAdmission_) {
      return preparePacingTimerLocked();
    }
    while (!waiters_.empty()) {
      toNotify.push_back(std::move(waiters_.front().promise));
      waiters_.pop_front();
    }
    return std::nullopt;
  }
  int64_t headroom = capacityLocked() - pending_.load();
  if (headroom <= 0) {
    setRateDemandBlockedLocked(false);
    return std::nullopt;
  }
  if (isRatePacingLocked() && now_() < nextPacedAdmission_) {
    return preparePacingTimerLocked();
  }
  if (isRatePacingLocked()) {
    setRateDemandBlockedLocked(false);
  }
  while (!waiters_.empty()) {
    toNotify.push_back(std::move(waiters_.front().promise));
    waiters_.pop_front();
  }
  return std::nullopt;
}

std::optional<RPCRateLimiter::PacingTimerRequest>
RPCRateLimiter::preparePacingTimerLocked() {
  const bool canAdmitWhenDue = isPacingLocked()
      ? pending_.load() == 0
      : isRatePacingLocked() && pending_.load() < capacityLocked();
  if (pacingTimerArmed_ || !canAdmitWhenDue || waiters_.empty()) {
    return std::nullopt;
  }
  const auto now = now_();
  if (!config_.scheduleFixes && now >= nextPacedAdmission_) {
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
      toFail.push_back(std::move(waiters_.front().promise));
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
  accrueLocked();
  mutate(config_);
  config_ = clamped(config_);
  ratePolicySelected_.store(
      config_.ratePolicy != RPCFlowControlPolicy::kWindowAimd,
      std::memory_order_relaxed);
  if (config_.ratePolicy == RPCFlowControlPolicy::kWindowAimd) {
    rateDemandBlocked_ = false;
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
        !isPacingLocked(),
        "Unconditional acquisition cannot bypass paced admission");
    VELOX_CHECK(
        !isRatePacingLocked(),
        "Unconditional acquisition cannot bypass rate RPC pacing");
    lastAdmission_ = now_();
    noteAdmissionLocked(1, lastAdmission_.value());
    pending = ++pending_;
    overloadEpoch = overloadEpoch_;
  }
  notePeakPending(pending);
  RPC_RATE_LIMITER_VLOG(2) << "acquire[" << admissionKey_
                           << "]: pending=" << pending;
  return Token{tokenConstructionKey_, this, overloadEpoch};
}

std::vector<RPCRateLimiter::Token> RPCRateLimiter::tryAcquireUpTo(
    int64_t want) {
  std::vector<Token> granted;
  if (want <= 0) {
    if (!ratePolicySelected_.load(std::memory_order_relaxed)) {
      return granted;
    }
    std::lock_guard<std::mutex> l(mutex_);
    if (waiters_.empty()) {
      setRateDemandBlockedLocked(false);
    }
    return granted;
  }

  std::unique_lock<std::mutex> l(mutex_);
  const auto now = now_();
  if (isPacingLocked()) {
    if (pending_.load() > 0 || now < nextPacedAdmission_) {
      return granted;
    }
    commitGrowthLocked();
    const int64_t pending = ++pending_;
    ++pacingGeneration_;
    pacingTimerArmed_ = false;
    lastAdmission_ = now;
    noteAdmissionLocked(1, now);
    nextPacedAdmission_ = isPacingLocked()
        ? now +
            (isRatePacingLocked()
                 ? std::max(nextPacedDelayLocked(), ratePacingIntervalLocked())
                 : nextPacedDelayLocked())
        : std::chrono::steady_clock::time_point{};
    notePeakPending(pending);
    granted.emplace_back(tokenConstructionKey_, this, overloadEpoch_);
    return granted;
  }
  const int64_t capacity = capacityLocked();
  if (isRatePacingLocked()) {
    if (pending_.load() >= capacity || now < nextPacedAdmission_) {
      setRateDemandBlockedLocked(
          pending_.load() < capacity && now < nextPacedAdmission_);
      return granted;
    }
    setRateDemandBlockedLocked(!waiters_.empty());
    const int64_t pending = ++pending_;
    ++pacingGeneration_;
    pacingTimerArmed_ = false;
    lastAdmission_ = now;
    noteAdmissionLocked(1, now);
    nextPacedAdmission_ = now + ratePacingIntervalLocked();
    notePeakPending(pending);
    granted.emplace_back(tokenConstructionKey_, this, overloadEpoch_);
    const auto timerRequest = preparePacingTimerLocked();
    l.unlock();
    if (timerRequest.has_value()) {
      schedulePacingTimer(timerRequest.value());
    }
    return granted;
  }

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
  lastAdmission_ = now;
  noteAdmissionLocked(take, now);

  // The post-exchange total, not the caller's pre-read: reporting the stale
  // value would hide exactly the overshoot this loop exists to prevent, and
  // the contention tests assert on this counter.
  notePeakPending(pending + take);
  granted.reserve(static_cast<size_t>(take));
  for (int64_t i = 0; i < take; ++i) {
    // One token per slot: each releases exactly one on destruction, so the
    // bulk grant unwinds row by row as the requests complete.
    granted.emplace_back(tokenConstructionKey_, this, overloadEpoch_);
  }
  return granted;
}

int64_t RPCRateLimiter::available() const {
  std::lock_guard<std::mutex> l(mutex_);
  if (isPacingLocked()) {
    return pending_.load() == 0 && now_() >= nextPacedAdmission_ ? 1 : 0;
  }
  if (isRatePacingLocked()) {
    return pending_.load() < capacityLocked() && now_() >= nextPacedAdmission_
        ? 1
        : 0;
  }
  return std::max<int64_t>(0, capacityLocked() - pending_.load());
}

RPCRateLimiter::AdmissionWait::AdmissionWait(
    std::weak_ptr<RPCRateLimiter> owner,
    uint64_t id)
    : owner_{std::move(owner)}, id_{id} {}

RPCRateLimiter::AdmissionWait::~AdmissionWait() {
  if (const auto owner = owner_.lock()) {
    owner->cancelAdmissionWait(id_);
  }
}

void RPCRateLimiter::cancelAdmissionWait(uint64_t id) noexcept {
  std::vector<ContinuePromise> canceled;
  try {
    std::lock_guard<std::mutex> l(mutex_);
    const auto position = std::find_if(
        waiters_.begin(), waiters_.end(), [id](const auto& waiter) {
          return waiter.id == id;
        });
    if (position == waiters_.end()) {
      return;
    }
    canceled.push_back(std::move(position->promise));
    waiters_.erase(position);
    if (waiters_.empty()) {
      setRateDemandBlockedLocked(false);
      ++pacingGeneration_;
      pacingTimerArmed_ = false;
    }
  } catch (...) {
    logLimiterFailure(admissionKey_, "admission wait cancellation");
    return;
  }
  fulfillWaitersNoThrow(canceled, admissionKey_);
}

RPCRateLimiter::Admission RPCRateLimiter::admitOrWait() {
  Admission result;
  std::optional<PacingTimerRequest> timerRequest;
  {
    std::lock_guard<std::mutex> l(mutex_);
    const int64_t pending = pending_.load();
    const int64_t capacity = capacityLocked();
    if (isPacingLocked()) {
      if (pending == 0 && now_() >= nextPacedAdmission_) {
        result.admitted = true;
        return result;
      }
      const auto id = nextAdmissionWaitId_++;
      waiters_.push_back({id, ContinuePromise{"RPCRateLimiter::admitOrWait"}});
      result.wait = waiters_.back().promise.getSemiFuture();
      if (config_.adaptive &&
          config_.ratePolicy != RPCFlowControlPolicy::kWindowAimd) {
        result.registration =
            std::make_shared<AdmissionWait>(weak_from_this(), id);
      }
      timerRequest = preparePacingTimerLocked();
    } else if (
        isRatePacingLocked() && pending < capacity &&
        now_() < nextPacedAdmission_) {
      setRateDemandBlockedLocked(true);
      const auto id = nextAdmissionWaitId_++;
      waiters_.push_back({id, ContinuePromise{"RPCRateLimiter::admitOrWait"}});
      result.wait = waiters_.back().promise.getSemiFuture();
      if (config_.adaptive &&
          config_.ratePolicy != RPCFlowControlPolicy::kWindowAimd) {
        result.registration =
            std::make_shared<AdmissionWait>(weak_from_this(), id);
      }
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
      const auto id = nextAdmissionWaitId_++;
      waiters_.push_back({id, ContinuePromise{"RPCRateLimiter::admitOrWait"}});
      result.wait = waiters_.back().promise.getSemiFuture();
      if (config_.adaptive &&
          config_.ratePolicy != RPCFlowControlPolicy::kWindowAimd) {
        result.registration =
            std::make_shared<AdmissionWait>(weak_from_this(), id);
      }
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

void RPCRateLimiter::applyOverloadLocked() {
  decreaseLimitLocked(limitLocked() * config_.decreaseFactor);
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
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  {
    std::lock_guard<std::mutex> l(mutex_);
    ++overloadEpoch_;
    applyOverloadLocked();
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
  serviceHorizonNanos_ =
      ((kServiceHorizonEwmaWeight - 1) * serviceHorizonNanos_ + rttNs) /
      kServiceHorizonEwmaWeight;
  return serviceHorizonNanos_ != previous;
}

bool RPCRateLimiter::successEarnsCreditLocked() const {
  return config_.adaptive && limit_ > 0.0 &&
      limitLocked() < static_cast<double>(ceilingLocked());
}

void RPCRateLimiter::applySuccessLocked(int64_t units, int64_t rttNs) {
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
    applySuccessLocked(units, rttNs);
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
    if (overloadEpoch != overloadEpoch_) {
      commitGrowthLocked();
      const bool horizonMoved = updateServiceHorizonLocked(
          config_.adaptive &&
                  config_.ratePolicy != RPCFlowControlPolicy::kWindowAimd
              ? 0
              : rttNs);
      discarded = successEarnsCreditLocked();
      if (horizonMoved && isPacingLocked()) {
        setLimitLocked(limitLocked());
      }
    } else {
      applySuccessLocked(
          units,
          config_.adaptive &&
                  config_.ratePolicy != RPCFlowControlPolicy::kWindowAimd
              ? 0
              : rttNs);
    }
    timerRequest = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
  return !discarded;
}

void RPCRateLimiter::release() noexcept {
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
      .lowWaterCapacity =
          std::max<int64_t>(1, static_cast<int64_t>(std::floor(lowWaterLimit))),
      .limitMilli = static_cast<int64_t>(std::llround(currentLimit * 1'000)),
      .lowWaterLimitMilli =
          static_cast<int64_t>(std::llround(lowWaterLimit * 1'000)),
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
      .sendRateMilli = config_.ratePolicy == RPCFlowControlPolicy::kWindowAimd
          ? 0
          : static_cast<int64_t>(std::llround(sendRatePerSecond_ * 1'000)),
      .ratePacingNanos =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              ratePacingTime_ +
              (isRatePacingLocked()
                   ? sinceAccrual
                   : std::chrono::steady_clock::duration::zero()))
              .count(),
      .rateWaitingNanos =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              rateWaitingTime_ +
              (rateDemandBlocked_
                   ? now - rateDemandUpdatedAt_
                   : std::chrono::steady_clock::duration::zero()))
              .count(),
      .admittedAttempts = admittedAttempts_,
      .completionAttempts = completionAttempts_,
      .successfulAttempts = successfulAttempts_,
      .overloadAttempts = overloadAttempts_,
      .rateDecreases = rateDecreases_,
      .admissionWaiters = static_cast<int64_t>(waiters_.size()),
      .admissionTimerArmed = pacingTimerArmed_,
      .nextPermitDelayNanos =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::max(
                  nextPacedAdmission_ - now,
                  std::chrono::steady_clock::duration::zero()))
              .count(),
      .ratePolicy = config_.ratePolicy,
      .scheduleFixes = config_.scheduleFixes,
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
