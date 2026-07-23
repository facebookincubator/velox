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
#include <mutex>
#include <shared_mutex>

#define RPC_RATE_LIMITER_LOG(severity) LOG(severity) << "[RPC_RATE_LIMITER] "
#define RPC_RATE_LIMITER_VLOG(level) VLOG(level) << "[RPC_RATE_LIMITER] "

namespace facebook::velox::exec::rpc {

// --- Token implementation ---

RPCRateLimiter::Token::Token(const std::string& tierKey)
    : tierKey_(tierKey), valid_(true) {}

RPCRateLimiter::Token& RPCRateLimiter::Token::operator=(
    Token&& other) noexcept {
  if (this != &other) {
    if (valid_) {
      decrementPending(tierKey_);
    }
    tierKey_ = std::move(other.tierKey_);
    valid_ = other.valid_;
    other.valid_ = false;
  }
  return *this;
}

RPCRateLimiter::Token::~Token() {
  if (valid_) {
    decrementPending(tierKey_);
  }
}

// --- Function-local statics ---

std::shared_mutex& RPCRateLimiter::mapMutex() {
  static std::shared_mutex mutex;
  return mutex;
}

std::atomic<int64_t>& RPCRateLimiter::defaultMaxPendingRef() {
  // Default: 20 concurrent RPCs per process per tier.
  static std::atomic<int64_t> maxPending{20};
  return maxPending;
}

std::unordered_map<std::string, std::unique_ptr<RPCRateLimiter::TierState>>&
RPCRateLimiter::tiers() {
  static std::unordered_map<std::string, std::unique_ptr<TierState>> tierMap;
  return tierMap;
}

std::atomic<bool>& RPCRateLimiter::adaptiveEnabledRef() {
  static std::atomic<bool> enabled{false};
  return enabled;
}

std::atomic<int64_t>& RPCRateLimiter::adaptiveMinRef() {
  static std::atomic<int64_t> minLimit{1};
  return minLimit;
}

std::atomic<double>& RPCRateLimiter::adaptiveFactorRef() {
  static std::atomic<double> factor{0.5};
  return factor;
}

// Caller holds state.mutex.
int64_t RPCRateLimiter::effectiveLimitLocked(const TierState& state) {
  const int64_t ceiling =
      state.maxPending > 0 ? state.maxPending : defaultMaxPendingRef().load();
  if (!adaptiveEnabledRef().load() || state.adaptiveLimit <= 0) {
    return ceiling;
  }
  // Floor is clamped to <= ceiling so a misconfigured min_limit > ceiling can't
  // make std::clamp's lo exceed hi (undefined behavior).
  const int64_t floor =
      std::min<int64_t>(std::max<int64_t>(1, adaptiveMinRef().load()), ceiling);
  return std::clamp<int64_t>(state.adaptiveLimit, floor, ceiling);
}

// --- TierState lookup ---

RPCRateLimiter::TierState& RPCRateLimiter::getOrCreateTierState(
    const std::string& tierKey) {
  auto& tierMap = tiers();
  // Fast path: shared lock for the common case (tier already exists), so
  // concurrent lookups from all drivers do not serialize. TierState is held by
  // unique_ptr, so the returned reference stays valid after we drop the lock
  // even if other threads later insert new tiers (only the map nodes move, not
  // the pointed-to TierState).
  {
    std::shared_lock<std::shared_mutex> rl(mapMutex());
    auto it = tierMap.find(tierKey);
    if (it != tierMap.end()) {
      return *it->second;
    }
  }
  // Slow path: first sighting of this tier — take the exclusive lock to insert,
  // re-checking in case another thread created it between the two locks.
  std::unique_lock<std::shared_mutex> wl(mapMutex());
  auto it = tierMap.find(tierKey);
  if (it != tierMap.end()) {
    return *it->second;
  }
  auto [newIt, _] = tierMap.emplace(tierKey, std::make_unique<TierState>());
  return *newIt->second;
}

// --- Public API ---

RPCRateLimiter::Token RPCRateLimiter::acquire(const std::string& tierKey) {
  incrementPending(tierKey);
  return Token(tierKey);
}

std::optional<ContinueFuture> RPCRateLimiter::checkBackpressure(
    const std::string& tierKey) {
  auto& state = getOrCreateTierState(tierKey);

  std::lock_guard<std::mutex> l(state.mutex);

  int64_t pending = state.pendingCount.load();
  int64_t maxPending = effectiveLimitLocked(state);

  if (pending < maxPending) {
    RPC_RATE_LIMITER_VLOG(2)
        << "checkBackpressure[" << tierKey << "]: OK (pending=" << pending
        << ", max=" << maxPending << ")";
    return std::nullopt;
  }

  RPC_RATE_LIMITER_VLOG(1) << "checkBackpressure[" << tierKey
                           << "]: BLOCKED (pending=" << pending
                           << ", max=" << maxPending
                           << "), creating wait promise #"
                           << state.waiters.size();
  state.waiters.emplace_back("RPCRateLimiter::checkBackpressure");
  return state.waiters.back().getSemiFuture();
}

int64_t RPCRateLimiter::pendingCount(const std::string& tierKey) {
  auto& state = getOrCreateTierState(tierKey);
  return state.pendingCount.load();
}

int64_t RPCRateLimiter::availableHeadroom(const std::string& tierKey) {
  auto& state = getOrCreateTierState(tierKey);
  std::lock_guard<std::mutex> l(state.mutex);
  const int64_t cap = effectiveLimitLocked(state);
  const int64_t pending = state.pendingCount.load();
  return std::max<int64_t>(0, cap - pending);
}

int64_t RPCRateLimiter::currentLimit(const std::string& tierKey) {
  auto& state = getOrCreateTierState(tierKey);
  std::lock_guard<std::mutex> l(state.mutex);
  return effectiveLimitLocked(state);
}

int64_t RPCRateLimiter::peakPending(const std::string& tierKey) {
  return getOrCreateTierState(tierKey).peakPending.load();
}

int64_t RPCRateLimiter::minLimitReached(const std::string& tierKey) {
  auto& state = getOrCreateTierState(tierKey);
  std::lock_guard<std::mutex> l(state.mutex);
  return state.minAdaptiveLimit;
}

void RPCRateLimiter::setMaxPending(const std::string& tierKey, int64_t limit) {
  auto& state = getOrCreateTierState(tierKey);
  std::lock_guard<std::mutex> l(state.mutex);
  state.maxPending = limit;
  RPC_RATE_LIMITER_VLOG(1) << "setMaxPending[" << tierKey << "]: set to "
                           << limit;
}

void RPCRateLimiter::setDefaultMaxPending(int64_t limit) {
  defaultMaxPendingRef().store(limit);
  RPC_RATE_LIMITER_VLOG(1) << "setDefaultMaxPending: set to " << limit;
}

int64_t RPCRateLimiter::defaultMaxPending() {
  return defaultMaxPendingRef().load();
}

void RPCRateLimiter::setAdaptiveConfig(
    bool enabled,
    int64_t minLimit,
    double decreaseFactor) {
  adaptiveMinRef().store(std::max<int64_t>(1, minLimit));
  adaptiveFactorRef().store(std::clamp(decreaseFactor, 0.01, 0.99));
  const bool was = adaptiveEnabledRef().exchange(enabled);
  if (was != enabled) {
    RPC_RATE_LIMITER_LOG(WARNING)
        << "adaptive limiter " << (enabled ? "ENABLED" : "DISABLED")
        << " (min=" << adaptiveMinRef().load()
        << ", decrease=" << adaptiveFactorRef().load() << ")";
  }
}

bool RPCRateLimiter::adaptiveEnabled() {
  return adaptiveEnabledRef().load();
}

void RPCRateLimiter::onRateLimited(const std::string& tierKey) {
  if (!adaptiveEnabledRef().load()) {
    return;
  }
  auto& state = getOrCreateTierState(tierKey);
  std::lock_guard<std::mutex> l(state.mutex);
  const int64_t ceiling =
      state.maxPending > 0 ? state.maxPending : defaultMaxPendingRef().load();
  const int64_t cur = state.adaptiveLimit > 0 ? state.adaptiveLimit : ceiling;
  // Floor clamped to <= ceiling so a misconfigured min_limit can't make
  // std::clamp's lo exceed hi (undefined behavior).
  const int64_t floor =
      std::min<int64_t>(std::max<int64_t>(1, adaptiveMinRef().load()), ceiling);
  int64_t next = static_cast<int64_t>(
      static_cast<double>(cur) * adaptiveFactorRef().load());
  next = std::clamp<int64_t>(next, floor, ceiling);
  if (next < cur) {
    state.adaptiveLimit = next;
    if (state.minAdaptiveLimit == 0 || next < state.minAdaptiveLimit) {
      state.minAdaptiveLimit = next;
    }
    RPC_RATE_LIMITER_VLOG(1)
        << "RPC congestion: adaptive cap[" << tierKey << "] " << cur << " -> "
        << next << " (rate-limit)";
  }
}

void RPCRateLimiter::onSuccess(const std::string& tierKey, int64_t successes) {
  if (!adaptiveEnabledRef().load() || successes <= 0) {
    return;
  }
  auto& state = getOrCreateTierState(tierKey);
  std::vector<ContinuePromise> waitersToNotify;
  {
    std::lock_guard<std::mutex> l(state.mutex);
    // Only recover if we previously shrank (adaptiveLimit > 0).
    if (state.adaptiveLimit <= 0) {
      return;
    }
    const int64_t ceiling =
        state.maxPending > 0 ? state.maxPending : defaultMaxPendingRef().load();
    // AIMD additive-increase, TCP-Reno style: one +1 step per cap-worth of
    // successes, floored at +1 per call so a steady success stream always makes
    // progress. Scaling the step to the drain size makes recovery track the ÷2
    // decrease's aggressiveness, instead of crawling +1 per (up-to-1k-row)
    // drain — which in practice never recovered within a query. Reaching the
    // ceiling clears the adaptive state so the static cap governs again.
    const int64_t step = std::max<int64_t>(
        1, successes / std::max<int64_t>(1, state.adaptiveLimit));
    const int64_t next = state.adaptiveLimit + step;
    state.adaptiveLimit = next >= ceiling ? 0 : next;
    // A large recovery step can reopen several slots at once, so wake up to the
    // recovered headroom's worth of blocked waiters (FIFO) — not just one, or
    // drivers stay parked in checkBackpressure() despite available capacity.
    int64_t headroom = effectiveLimitLocked(state) - state.pendingCount.load();
    while (headroom > 0 && !state.waiters.empty()) {
      waitersToNotify.push_back(std::move(state.waiters.front()));
      state.waiters.pop_front();
      --headroom;
    }
  }
  for (auto& waiter : waitersToNotify) {
    waiter.setValue();
  }
}

void RPCRateLimiter::testingResetAllState() {
  std::unique_lock<std::shared_mutex> l(mapMutex());
  defaultMaxPendingRef().store(20);
  adaptiveEnabledRef().store(false);
  adaptiveMinRef().store(1);
  adaptiveFactorRef().store(0.5);
  tiers().clear();
}

// --- Internal helpers ---

void RPCRateLimiter::incrementPending(const std::string& tierKey) {
  auto& state = getOrCreateTierState(tierKey);
  int64_t newCount = ++state.pendingCount;
  // Lock-free high-water update for observability. Relaxed ordering: this is a
  // best-effort max for stats only, not a synchronization point.
  int64_t prevPeak = state.peakPending.load(std::memory_order_relaxed);
  while (newCount > prevPeak &&
         !state.peakPending.compare_exchange_weak(
             prevPeak, newCount, std::memory_order_relaxed)) {
  }
  RPC_RATE_LIMITER_VLOG(2) << "incrementPending[" << tierKey
                           << "]: pending=" << newCount;
}

void RPCRateLimiter::decrementPending(const std::string& tierKey) {
  auto& state = getOrCreateTierState(tierKey);
  int64_t newCount = --state.pendingCount;
  RPC_RATE_LIMITER_VLOG(2) << "decrementPending[" << tierKey
                           << "]: pending=" << newCount;

  // CRITICAL: Hold the per-tier mutex when checking whether to notify waiters.
  // This prevents a TOCTOU race where:
  // 1. We read newCount < maxPending (should notify)
  // 2. A new waiter is added in checkBackpressure() between read and notify
  //
  // By holding the lock during check-and-notify, any waiter added in
  // checkBackpressure() will either:
  // - Be in state.waiters before we check (and we'll notify it)
  // - See the updated count and not need to wait at all
  //
  // We notify only one waiter per decrement (FIFO) to avoid thundering herd.
  std::optional<ContinuePromise> waiterToNotify;
  {
    std::lock_guard<std::mutex> l(state.mutex);
    int64_t maxPending = effectiveLimitLocked(state);
    if (newCount < maxPending && !state.waiters.empty()) {
      RPC_RATE_LIMITER_VLOG(1)
          << "decrementPending[" << tierKey << "]: notifying 1 of "
          << state.waiters.size() << " waiters";
      waiterToNotify = std::move(state.waiters.front());
      state.waiters.pop_front();
    }
  }
  if (waiterToNotify) {
    waiterToNotify->setValue();
  }
}

} // namespace facebook::velox::exec::rpc
