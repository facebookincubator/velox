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

#include "velox/exec/rpc/BackendAdmission.h"

#include <algorithm>
#include <utility>
#include <vector>

#define RPC_BACKEND_ADMISSION_LOG(severity) \
  LOG(severity) << "[RPC_BACKEND_ADMISSION] "
#define RPC_BACKEND_ADMISSION_VLOG(level) \
  VLOG(level) << "[RPC_BACKEND_ADMISSION] "

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

BackendAdmission::Token& BackendAdmission::Token::operator=(
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

BackendAdmission::Token::~Token() {
  if (owner_ != nullptr) {
    owner_->release();
  }
}

// --- BackendAdmission ---

BackendAdmission::BackendAdmission(std::string backendKey)
    : backendKey_{std::move(backendKey)} {}

void BackendAdmission::setDefaultCapacity(int64_t capacity) {
  defaultCapacityRef().store(capacity);
  RPC_BACKEND_ADMISSION_VLOG(1) << "default capacity set to " << capacity;
}

int64_t BackendAdmission::defaultCapacity() {
  return defaultCapacityRef().load();
}

int64_t BackendAdmission::ceilingLocked() const {
  return config_.ceiling > 0 ? config_.ceiling : defaultCapacityRef().load();
}

int64_t BackendAdmission::floorLocked(int64_t ceiling) const {
  return std::min<int64_t>(std::max<int64_t>(1, config_.floor), ceiling);
}

int64_t BackendAdmission::capacityLocked() const {
  const int64_t ceiling = ceilingLocked();
  if (!config_.adaptive || capacity_ <= 0) {
    return ceiling;
  }
  return std::clamp<int64_t>(capacity_, floorLocked(ceiling), ceiling);
}

void BackendAdmission::configure(const Config& config) {
  std::lock_guard<std::mutex> l(mutex_);
  const bool was = config_.adaptive;
  config_ = config;
  config_.floor = std::max<int64_t>(1, config.floor);
  config_.decreaseFactor = std::clamp(config.decreaseFactor, 0.01, 0.99);
  if (was != config_.adaptive) {
    RPC_BACKEND_ADMISSION_LOG(WARNING)
        << "adaptive capacity " << (config_.adaptive ? "ENABLED" : "DISABLED")
        << " for backend " << backendKey_ << " (floor=" << config_.floor
        << ", decrease=" << config_.decreaseFactor << ")";
  }
}

void BackendAdmission::testingReset() {
  std::lock_guard<std::mutex> l(mutex_);
  config_ = Config{};
  capacity_ = 0;
  lowWater_ = 0;
  pending_.store(0);
  peakPending_.store(0);
  waiters_.clear();
}

void BackendAdmission::amend(const std::function<void(Config&)>& mutate) {
  std::lock_guard<std::mutex> l(mutex_);
  const bool was = config_.adaptive;
  mutate(config_);
  config_.floor = std::max<int64_t>(1, config_.floor);
  config_.decreaseFactor = std::clamp(config_.decreaseFactor, 0.01, 0.99);
  if (was != config_.adaptive) {
    RPC_BACKEND_ADMISSION_LOG(WARNING)
        << "adaptive capacity " << (config_.adaptive ? "ENABLED" : "DISABLED")
        << " for backend " << backendKey_ << " (floor=" << config_.floor
        << ", decrease=" << config_.decreaseFactor << ")";
  }
}

BackendAdmission::Config BackendAdmission::config() const {
  std::lock_guard<std::mutex> l(mutex_);
  return config_;
}

BackendAdmission::Token BackendAdmission::acquire() {
  const int64_t pending = ++pending_;
  // Lock-free high-water update. Relaxed ordering: a best-effort max for
  // stats, not a synchronization point.
  int64_t peak = peakPending_.load(std::memory_order_relaxed);
  while (pending > peak &&
         !peakPending_.compare_exchange_weak(
             peak, pending, std::memory_order_relaxed)) {
  }
  RPC_BACKEND_ADMISSION_VLOG(2)
      << "acquire[" << backendKey_ << "]: pending=" << pending;
  return Token{this};
}

int64_t BackendAdmission::available() const {
  std::lock_guard<std::mutex> l(mutex_);
  return std::max<int64_t>(0, capacityLocked() - pending_.load());
}

std::optional<ContinueFuture> BackendAdmission::waitForCapacity() {
  std::lock_guard<std::mutex> l(mutex_);
  const int64_t pending = pending_.load();
  const int64_t capacity = capacityLocked();
  if (pending < capacity) {
    RPC_BACKEND_ADMISSION_VLOG(2)
        << "waitForCapacity[" << backendKey_ << "]: OK (pending=" << pending
        << ", capacity=" << capacity << ")";
    return std::nullopt;
  }
  RPC_BACKEND_ADMISSION_VLOG(1)
      << "waitForCapacity[" << backendKey_ << "]: BLOCKED (pending=" << pending
      << ", capacity=" << capacity << "), parking waiter #" << waiters_.size();
  waiters_.emplace_back("BackendAdmission::waitForCapacity");
  return waiters_.back().getSemiFuture();
}

void BackendAdmission::onOutcome(Outcome outcome, int64_t units) {
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

void BackendAdmission::onOverload() {
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
  RPC_BACKEND_ADMISSION_VLOG(1)
      << "RPC congestion: capacity[" << backendKey_ << "] " << current << " -> "
      << next << " (overload)";
}

void BackendAdmission::onSuccess(int64_t units) {
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
    // recovered headroom's worth of parked drivers (FIFO). Waking only one
    // would leave drivers parked despite available capacity.
    int64_t headroom = capacityLocked() - pending_.load();
    while (headroom > 0 && !waiters_.empty()) {
      toNotify.push_back(std::move(waiters_.front()));
      waiters_.pop_front();
      --headroom;
    }
  }
  for (auto& waiter : toNotify) {
    waiter.setValue();
  }
}

void BackendAdmission::release() {
  // Saturate at zero. A token can outlive testingReset(), which zeroes the
  // count; without the floor its release would drive pending_ negative and
  // make available() report more capacity than exists.
  int64_t pending = pending_.load();
  while (pending > 0 && !pending_.compare_exchange_weak(pending, pending - 1)) {
  }
  pending = std::max<int64_t>(0, pending - 1);
  RPC_BACKEND_ADMISSION_VLOG(2)
      << "release[" << backendKey_ << "]: pending=" << pending;

  // Hold the lock across check-and-dequeue. Otherwise a waiter parked by
  // waitForCapacity() between the capacity read and the dequeue would be
  // missed: with the lock held it is either already queued and gets this slot,
  // or it observes the decremented count and never parks.
  //
  // One waiter per release (FIFO), to avoid a thundering herd.
  std::optional<ContinuePromise> toNotify;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (pending < capacityLocked() && !waiters_.empty()) {
      RPC_BACKEND_ADMISSION_VLOG(1)
          << "release[" << backendKey_ << "]: waking 1 of " << waiters_.size()
          << " waiters";
      toNotify = std::move(waiters_.front());
      waiters_.pop_front();
    }
  }
  if (toNotify.has_value()) {
    toNotify->setValue();
  }
}

BackendAdmission::Stats BackendAdmission::stats() const {
  std::lock_guard<std::mutex> l(mutex_);
  return Stats{
      .capacity = capacityLocked(),
      .pending = pending_.load(),
      .peakPending = peakPending_.load(),
      .lowWaterCapacity = lowWater_,
  };
}

// --- BackendRegistry ---

BackendRegistry& BackendRegistry::global() {
  static BackendRegistry registry;
  return registry;
}

BackendAdmission& BackendRegistry::get(const std::string& backendKey) {
  // Fast path: the backend already exists, so concurrent lookups from every
  // driver share the lock rather than serializing. Values are held by
  // unique_ptr, so the reference outlives later insertions.
  {
    std::shared_lock<std::shared_mutex> rl(mutex_);
    auto it = backends_.find(backendKey);
    if (it != backends_.end()) {
      return *it->second;
    }
  }
  // Slow path: first sight of this backend. Re-check under the exclusive lock
  // in case another thread created it between the two locks.
  std::unique_lock<std::shared_mutex> wl(mutex_);
  auto it = backends_.find(backendKey);
  if (it != backends_.end()) {
    return *it->second;
  }
  auto [inserted, _] = backends_.emplace(
      backendKey, std::make_unique<BackendAdmission>(backendKey));
  return *inserted->second;
}

void BackendRegistry::testingReset() {
  std::unique_lock<std::shared_mutex> wl(mutex_);
  defaultCapacityRef().store(20);
  // Reset each backend in place rather than dropping it. Tokens outlive the
  // operator that acquired them and release through a back-pointer, so
  // destroying a BackendAdmission that still has outstanding tokens would leave
  // them writing through a dangling pointer.
  for (auto& [backendKey, admission] : backends_) {
    admission->testingReset();
  }
}

} // namespace facebook::velox::exec::rpc
