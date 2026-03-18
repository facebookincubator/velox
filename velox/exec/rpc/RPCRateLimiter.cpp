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

std::mutex& RPCRateLimiter::mapMutex() {
  static std::mutex mutex;
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

// --- TierState lookup ---

RPCRateLimiter::TierState& RPCRateLimiter::getOrCreateTierState(
    const std::string& tierKey) {
  std::lock_guard<std::mutex> l(mapMutex());
  auto& tierMap = tiers();
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
  int64_t maxPending =
      state.maxPending > 0 ? state.maxPending : defaultMaxPendingRef().load();

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

void RPCRateLimiter::testingResetAllState() {
  std::lock_guard<std::mutex> l(mapMutex());
  defaultMaxPendingRef().store(20);
  tiers().clear();
}

// --- Internal helpers ---

void RPCRateLimiter::incrementPending(const std::string& tierKey) {
  auto& state = getOrCreateTierState(tierKey);
  int64_t newCount = ++state.pendingCount;
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
    int64_t maxPending =
        state.maxPending > 0 ? state.maxPending : defaultMaxPendingRef().load();
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
