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
#include <string_view>
#include <utility>
#include <vector>

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
  if (hardOverload && !tokens.empty()) {
    try {
      handled = owner->handleOverloadForEpoch(overloadEpoch_);
    } catch (...) {
      try {
        RPC_RATE_LIMITER_LOG(ERROR)
            << "failed to apply completion-time overload feedback";
      } catch (...) {
      }
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
          nullptr) {}

RPCRateLimiter::RPCRateLimiter(
    std::string admissionKey,
    std::function<std::chrono::steady_clock::time_point()> now,
    folly::Timekeeper* timekeeper)
    : admissionKey_{std::move(admissionKey)},
      now_{std::move(now)},
      timekeeper_{timekeeper} {}

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
  return std::min(std::max(0.001, config_.floor), ceiling);
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

std::chrono::steady_clock::duration RPCRateLimiter::pacingIntervalLocked()
    const {
  return std::chrono::ceil<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::nano>{
          static_cast<double>(serviceHorizonNanos_) / limitLocked()});
}

void RPCRateLimiter::setLimitLocked(double limit) {
  ++pacingGeneration_;
  pacingTimerArmed_ = false;
  const double ceiling = static_cast<double>(ceilingLocked());
  limit_ = limit > 0.0 && limit < ceiling
      ? std::clamp(limit, floorLocked(ceiling), ceiling)
      : 0.0;
  if (isPacingLocked()) {
    const auto now = now_();
    nextPacedAdmission_ = lastAdmission_.has_value()
        ? std::max(now, lastAdmission_.value() + pacingIntervalLocked())
        : now + pacingIntervalLocked();
  } else {
    nextPacedAdmission_ = std::chrono::steady_clock::time_point{};
  }
}

int64_t RPCRateLimiter::capacityLocked() const {
  const int64_t ceiling = ceilingLocked();
  return std::clamp<int64_t>(
      static_cast<int64_t>(std::floor(limitLocked())), 1, ceiling);
}

void RPCRateLimiter::configure(const Config& config) {
  // Replacing the whole config is one kind of amendment, so it shares
  // amend()'s clamping and adaptive-flip logging rather than repeating them.
  amend([&config](Config& target) { target = config; });
}

void RPCRateLimiter::initializeOnce(
    const std::function<void(Config&)>& mutate) {
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (initialized_) {
      // Compare what the caller would actually get, not what they passed: an
      // out-of-range floor or factor clamps to the same effective value and is
      // not a divergent request.
      Config wanted = config_;
      mutate(wanted);
      clampLocked(wanted);
      if (wanted.adaptive != config_.adaptive ||
          wanted.floor != config_.floor ||
          wanted.decreaseFactor != config_.decreaseFactor ||
          wanted.ceiling != config_.ceiling) {
        // Per backend, not LOG_FIRST_N: that is process-wide, so the first
        // backend to take this path would silence the warning for every other
        // one even though the message names a specific backend.
        if (!loggedDivergentSettings_) {
          loggedDivergentSettings_ = true;
          RPC_RATE_LIMITER_LOG(WARNING)
              << "admission key " << admissionKey_
              << " is already initialized; ignoring different settings from a "
                 "later query. One configuration is shared by every query on a "
                 "backend, and it is fixed by the first of them.";
        }
      }
    } else {
      applyMutationLocked(mutate);
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
}

void RPCRateLimiter::testingReset() {
  std::lock_guard<std::mutex> l(mutex_);
  config_ = Config{};
  initialized_ = false;
  loggedDivergentSettings_ = false;
  limit_ = 0.0;
  lowWaterLimit_ = 0.0;
  serviceHorizonNanos_ = kBootstrapServiceHorizonNanos;
  hasServiceHorizonSample_ = false;
  nextPacedAdmission_ = std::chrono::steady_clock::time_point{};
  lastAdmission_.reset();
  ++pacingGeneration_;
  ++overloadEpoch_;
  pacingTimerArmed_ = false;
  pending_.store(0);
  peakPending_.store(0);
  waiters_.clear();
}

void RPCRateLimiter::clampLocked(Config& config) {
  config.floor = std::max(0.001, config.floor);
  config.decreaseFactor = std::clamp(config.decreaseFactor, 0.01, 0.99);
}

std::optional<RPCRateLimiter::PacingTimerRequest>
RPCRateLimiter::collectWakeableLocked(std::vector<ContinuePromise>& toNotify) {
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
  if (now >= nextPacedAdmission_) {
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
  mutate(config_);
  clampLocked(config_);
  if (!config_.adaptive) {
    setLimitLocked(0.0);
  } else if (!wasAdaptive) {
    setLimitLocked(
        std::min(kInitialAdaptiveLimit, static_cast<double>(ceilingLocked())));
    if (lowWaterLimit_ == 0.0 || limitLocked() < lowWaterLimit_) {
      lowWaterLimit_ = limitLocked();
    }
  } else if (limit_ > 0.0) {
    setLimitLocked(limit_);
  }
  if (wasAdaptive != config_.adaptive) {
    RPC_RATE_LIMITER_LOG(WARNING)
        << "adaptive capacity " << (config_.adaptive ? "ENABLED" : "DISABLED")
        << " for admission key " << admissionKey_ << " (floor=" << config_.floor
        << ", decrease=" << config_.decreaseFactor << ")";
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
    lastAdmission_ = now_();
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
    return granted;
  }

  std::lock_guard<std::mutex> l(mutex_);
  const auto now = now_();
  if (isPacingLocked()) {
    if (pending_.load() > 0 || now < nextPacedAdmission_) {
      return granted;
    }
    const int64_t pending = ++pending_;
    ++pacingGeneration_;
    pacingTimerArmed_ = false;
    lastAdmission_ = now;
    nextPacedAdmission_ = now + pacingIntervalLocked();
    notePeakPending(pending);
    granted.emplace_back(tokenConstructionKey_, this, overloadEpoch_);
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
  lastAdmission_ = now;

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
  return std::max<int64_t>(0, capacityLocked() - pending_.load());
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
    case Outcome::kNone:
      return;
  }
}

void RPCRateLimiter::applyOverloadLocked() {
  if (!config_.adaptive) {
    return;
  }
  const double current = limitLocked();
  const double next = std::max(
      floorLocked(static_cast<double>(ceilingLocked())),
      current * config_.decreaseFactor);
  if (next < current) {
    setLimitLocked(next);
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

bool RPCRateLimiter::handleOverloadForEpoch(uint64_t overloadEpoch) {
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (overloadEpoch != overloadEpoch_) {
      return true;
    }
    ++overloadEpoch_;
    applyOverloadLocked();
    timerRequest = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
  return true;
}

bool RPCRateLimiter::updateServiceHorizonLocked(int64_t rttNs) {
  if (rttNs <= 0) {
    return false;
  }
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

void RPCRateLimiter::applySuccessLocked(int64_t units, int64_t rttNs) {
  updateServiceHorizonLocked(rttNs);
  if (units <= 0 || !config_.adaptive || limit_ <= 0.0) {
    return;
  }
  double next = limitLocked();
  int64_t remainingUnits = units;
  while (remainingUnits > 0 && next < 1.0) {
    next = std::min(1.0, next * 2.0);
    --remainingUnits;
  }
  if (remainingUnits > 0) {
    next = std::sqrt(next * next + 2.0 * static_cast<double>(remainingUnits));
  }
  setLimitLocked(next >= static_cast<double>(ceilingLocked()) ? 0.0 : next);
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

void RPCRateLimiter::onSuccessForEpoch(
    uint64_t overloadEpoch,
    int64_t units,
    int64_t rttNs) {
  if (units <= 0) {
    return;
  }
  std::vector<ContinuePromise> toNotify;
  std::optional<PacingTimerRequest> timerRequest;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (overloadEpoch != overloadEpoch_) {
      if (updateServiceHorizonLocked(rttNs) && isPacingLocked()) {
        setLimitLocked(limitLocked());
      }
    } else {
      applySuccessLocked(units, rttNs);
    }
    timerRequest = collectWakeableLocked(toNotify);
  }
  fulfillWaitersNoThrow(toNotify, admissionKey_);
  if (timerRequest.has_value()) {
    schedulePacingTimer(timerRequest.value());
  }
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
  const double currentLimit = limitLocked();
  const double lowWaterLimit =
      lowWaterLimit_ > 0.0 ? lowWaterLimit_ : currentLimit;
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
}

} // namespace facebook::velox::exec::rpc
