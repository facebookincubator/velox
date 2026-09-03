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
#include <utility>
#include <vector>

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
} // namespace

// --- Token ---

RPCRateLimiter::Token& RPCRateLimiter::Token::operator=(
    Token&& other) noexcept {
  if (this != &other) {
    if (owner_ != nullptr) {
      owner_->release();
    }
    owner_ = other.owner_;
    other.owner_ = nullptr;
  }
  return *this;
}

RPCRateLimiter::Token::~Token() {
  if (owner_ != nullptr) {
    owner_->release();
  }
}

// --- RPCRateLimiter ---

RPCRateLimiter::RPCRateLimiter(std::string tierKey)
    : tierKey_{std::move(tierKey)} {}

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

int64_t RPCRateLimiter::floorLocked(int64_t ceiling) const {
  return std::min<int64_t>(std::max<int64_t>(1, config_.floor), ceiling);
}

int64_t RPCRateLimiter::capacityLocked() const {
  const int64_t ceiling = ceilingLocked();
  if (!config_.adaptive || capacity_ <= 0) {
    return ceiling;
  }
  return std::clamp<int64_t>(capacity_, floorLocked(ceiling), ceiling);
}

void RPCRateLimiter::configure(const Config& config) {
  // Replacing the whole config is one kind of amendment, so it shares
  // amend()'s clamping and adaptive-flip logging rather than repeating them.
  amend([&config](Config& target) { target = config; });
}

void RPCRateLimiter::initializeOnce(
    const std::function<void(Config&)>& mutate) {
  std::vector<ContinuePromise> toNotify;
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
        // one even though the message names a specific tier.
        if (!loggedDivergentSettings_) {
          loggedDivergentSettings_ = true;
          RPC_RATE_LIMITER_LOG(WARNING)
              << "backend " << tierKey_
              << " is already initialized; ignoring different settings from a "
                 "later query. One configuration is shared by every query on a "
                 "backend, and it is fixed by the first of them.";
        }
      }
    } else {
      applyMutationLocked(mutate);
      collectWakeableLocked(toNotify);
      // Sealed here rather than by a second call: two calls left a window in
      // which another query saw initialized_ still false and overwrote.
      initialized_ = true;
    }
  }
  for (auto& waiter : toNotify) {
    waiter.setValue();
  }
}

void RPCRateLimiter::testingReset() {
  std::lock_guard<std::mutex> l(mutex_);
  config_ = Config{};
  initialized_ = false;
  loggedDivergentSettings_ = false;
  capacity_ = 0;
  lowWater_ = 0;
  pending_.store(0);
  peakPending_.store(0);
  waiters_.clear();
}

void RPCRateLimiter::clampLocked(Config& config) {
  config.floor = std::max<int64_t>(1, config.floor);
  config.decreaseFactor = std::clamp(config.decreaseFactor, 0.01, 0.99);
}

void RPCRateLimiter::collectWakeableLocked(
    std::vector<ContinuePromise>& toNotify) {
  int64_t headroom = capacityLocked() - pending_.load();
  while (headroom > 0 && !waiters_.empty()) {
    toNotify.push_back(std::move(waiters_.front()));
    waiters_.pop_front();
    --headroom;
  }
}

void RPCRateLimiter::applyMutationLocked(
    const std::function<void(Config&)>& mutate) {
  const bool was = config_.adaptive;
  mutate(config_);
  clampLocked(config_);
  if (was != config_.adaptive) {
    RPC_RATE_LIMITER_LOG(WARNING)
        << "adaptive capacity " << (config_.adaptive ? "ENABLED" : "DISABLED")
        << " for backend " << tierKey_ << " (floor=" << config_.floor
        << ", decrease=" << config_.decreaseFactor << ")";
  }
}

void RPCRateLimiter::amend(const std::function<void(Config&)>& mutate) {
  std::vector<ContinuePromise> toNotify;
  {
    std::lock_guard<std::mutex> l(mutex_);
    applyMutationLocked(mutate);
    // A mutation that raises the ceiling grows capacity, and no release or
    // adaptation is guaranteed to follow, so drivers parked under the old
    // capacity would stay parked. Wake whatever now fits.
    collectWakeableLocked(toNotify);
  }
  for (auto& waiter : toNotify) {
    waiter.setValue();
  }
}

RPCRateLimiter::Config RPCRateLimiter::config() const {
  std::lock_guard<std::mutex> l(mutex_);
  return config_;
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
  const int64_t pending = ++pending_;
  notePeakPending(pending);
  RPC_RATE_LIMITER_VLOG(2) << "acquire[" << tierKey_
                           << "]: pending=" << pending;
  return Token{this};
}

std::vector<RPCRateLimiter::Token> RPCRateLimiter::tryAcquireUpTo(
    int64_t want) {
  std::vector<Token> granted;
  if (want <= 0) {
    return granted;
  }

  int64_t capacity;
  {
    std::lock_guard<std::mutex> l(mutex_);
    capacity = capacityLocked();
  }

  // One compare-exchange for the whole grant rather than a read followed by an
  // increment, so concurrent callers cannot both take the last slot however
  // many race here. The capacity snapshot may be stale, but it moves only on
  // adaptation, which is far slower than dispatch.
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

  // The post-exchange total, not the caller's pre-read: reporting the stale
  // value would hide exactly the overshoot this loop exists to prevent, and
  // the contention tests assert on this counter.
  notePeakPending(pending + take);
  granted.reserve(static_cast<size_t>(take));
  for (int64_t i = 0; i < take; ++i) {
    // One token per slot: each releases exactly one on destruction, so the
    // bulk grant unwinds row by row as the requests complete.
    granted.emplace_back(this);
  }
  return granted;
}

int64_t RPCRateLimiter::available() const {
  std::lock_guard<std::mutex> l(mutex_);
  return std::max<int64_t>(0, capacityLocked() - pending_.load());
}

RPCRateLimiter::Admission RPCRateLimiter::admitOrWait() {
  Admission result;
  std::lock_guard<std::mutex> l(mutex_);
  const int64_t pending = pending_.load();
  const int64_t capacity = capacityLocked();
  if (pending < capacity) {
    RPC_RATE_LIMITER_VLOG(2)
        << "admitOrWait[" << tierKey_ << "]: admitted (pending=" << pending
        << ", capacity=" << capacity << ")";
    result.admitted = true;
    return result;
  }
  RPC_RATE_LIMITER_VLOG(1) << "admitOrWait[" << tierKey_
                           << "]: waiting (pending=" << pending
                           << ", capacity=" << capacity << "), waiter #"
                           << waiters_.size();
  // Enrolled under the same lock that decided, so any release from here on
  // must see this waiter.
  waiters_.emplace_back("RPCRateLimiter::admitOrWait");
  result.wait = waiters_.back().getSemiFuture();
  return result;
}

void RPCRateLimiter::onOutcome(Outcome outcome, int64_t units) {
  switch (outcome) {
    case Outcome::kOverload:
      onOverload();
      return;
    case Outcome::kSuccess:
      onSuccess(units);
      return;
    case Outcome::kNone:
      return;
  }
}

void RPCRateLimiter::onOverload() {
  std::lock_guard<std::mutex> l(mutex_);
  if (!config_.adaptive) {
    return;
  }
  const int64_t ceiling = ceilingLocked();
  const int64_t current = capacity_ > 0 ? capacity_ : ceiling;
  const int64_t next = std::clamp<int64_t>(
      static_cast<int64_t>(
          static_cast<double>(current) * config_.decreaseFactor),
      floorLocked(ceiling),
      ceiling);
  if (next >= current) {
    return;
  }
  capacity_ = next;
  if (lowWater_ == 0 || next < lowWater_) {
    lowWater_ = next;
  }
  RPC_RATE_LIMITER_VLOG(1) << "RPC congestion: capacity[" << tierKey_ << "] "
                           << current << " -> " << next << " (overload)";
}

void RPCRateLimiter::onSuccess(int64_t units) {
  if (units <= 0) {
    return;
  }
  std::vector<ContinuePromise> toNotify;
  {
    std::lock_guard<std::mutex> l(mutex_);
    // Only recover from a capacity we previously shrank.
    if (!config_.adaptive || capacity_ <= 0) {
      return;
    }
    const int64_t ceiling = ceilingLocked();
    // AIMD additive-increase, TCP-Reno style: one +1 step per capacity-worth
    // of successes, floored at +1 so a steady success stream always makes
    // progress. Scaling the step to the drain size makes recovery track the
    // decrease's aggressiveness instead of crawling +1 per arbitrarily-sized
    // drain, which in practice never recovered within a query. Reaching the
    // ceiling clears the adapted value so the ceiling governs again.
    const int64_t step =
        std::max<int64_t>(1, units / std::max<int64_t>(1, capacity_));
    const int64_t next = capacity_ + step;
    capacity_ = next >= ceiling ? 0 : next;
    // A large step can reopen several slots at once, so wake up to the
    // recovered headroom's worth of parked drivers. Waking only one would
    // leave drivers parked despite available capacity.
    collectWakeableLocked(toNotify);
  }
  for (auto& waiter : toNotify) {
    waiter.setValue();
  }
}

void RPCRateLimiter::release() {
  // Saturate at zero. A token can outlive testingReset(), which zeroes the
  // count; without the floor its release would drive pending_ negative and
  // make available() report more capacity than exists.
  int64_t pending = pending_.load();
  while (pending > 0 && !pending_.compare_exchange_weak(pending, pending - 1)) {
  }
  pending = std::max<int64_t>(0, pending - 1);
  RPC_RATE_LIMITER_VLOG(2) << "release[" << tierKey_
                           << "]: pending=" << pending;

  // Hold the lock across check-and-dequeue. Otherwise a waiter parked by
  // admitOrWait() between the capacity read and the dequeue would be missed:
  // with the lock held it is either already queued and gets this slot, or it
  // observes the decremented count and never parks.
  //
  // One waiter per release (FIFO), to avoid a thundering herd.
  std::optional<ContinuePromise> toNotify;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (pending < capacityLocked() && !waiters_.empty()) {
      RPC_RATE_LIMITER_VLOG(1) << "release[" << tierKey_ << "]: waking 1 of "
                               << waiters_.size() << " waiters";
      toNotify = std::move(waiters_.front());
      waiters_.pop_front();
    }
  }
  if (toNotify.has_value()) {
    toNotify->setValue();
  }
}

RPCRateLimiter::Stats RPCRateLimiter::stats() const {
  std::lock_guard<std::mutex> l(mutex_);
  return Stats{
      .capacity = capacityLocked(),
      .pending = pending_.load(),
      .peakPending = peakPending_.load(),
      .lowWaterCapacity = lowWater_,
  };
}

// --- RPCRateLimiterRegistry ---

RPCRateLimiterRegistry& RPCRateLimiterRegistry::global() {
  static RPCRateLimiterRegistry registry;
  return registry;
}

RPCRateLimiter& RPCRateLimiterRegistry::get(const std::string& tierKey) {
  // Fast path: the backend already exists, so concurrent lookups from every
  // driver share the lock rather than serializing. Values are held by
  // unique_ptr, so the reference outlives later insertions.
  {
    std::shared_lock<std::shared_mutex> rl(mutex_);
    auto it = backends_.find(tierKey);
    if (it != backends_.end()) {
      return *it->second;
    }
  }
  // Slow path: first sight of this backend. Re-check under the exclusive lock
  // in case another thread created it between the two locks.
  std::unique_lock<std::shared_mutex> wl(mutex_);
  auto it = backends_.find(tierKey);
  if (it != backends_.end()) {
    return *it->second;
  }
  auto [inserted, _] =
      backends_.emplace(tierKey, std::make_unique<RPCRateLimiter>(tierKey));
  return *inserted->second;
}

void RPCRateLimiterRegistry::testingReset() {
  std::unique_lock<std::shared_mutex> wl(mutex_);
  defaultCapacityRef().store(20);
  // Reset each backend in place rather than dropping it. Tokens outlive the
  // operator that acquired them and release through a back-pointer, so
  // destroying a RPCRateLimiter that still has outstanding tokens would leave
  // them writing through a dangling pointer.
  for (auto& [tierKey, admission] : backends_) {
    admission->testingReset();
  }
}

} // namespace facebook::velox::exec::rpc
