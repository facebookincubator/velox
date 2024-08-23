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

#include "velox/common/memory/SharedArbitrator.h"
#include <mutex>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/config/Config.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/Timer.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::memory {

using namespace facebook::velox::memory;

namespace {
#define RETURN_IF_TRUE(func) \
  {                          \
    bool ret = func;         \
    if (ret) {               \
      return ret;            \
    }                        \
  }

#define CHECKED_GROW(pool, growBytes, reservationBytes) \
  try {                                                 \
    checkedGrow(pool, growBytes, reservationBytes);     \
  } catch (const VeloxRuntimeError& e) {                \
    freeCapacity(growBytes);                            \
    throw;                                              \
  }

template <typename T>
T getConfig(
    const std::unordered_map<std::string, std::string>& configs,
    const std::string_view& key,
    const T& defaultValue) {
  if (configs.count(std::string(key)) > 0) {
    try {
      return folly::to<T>(configs.at(std::string(key)));
    } catch (const std::exception& e) {
      VELOX_USER_FAIL(
          "Failed while parsing SharedArbitrator configs: {}", e.what());
    }
  }
  return defaultValue;
}
} // namespace

int64_t SharedArbitrator::ExtraConfig::getReservedCapacity(
    const std::unordered_map<std::string, std::string>& configs) {
  return config::toCapacity(
      getConfig<std::string>(
          configs, kReservedCapacity, std::string(kDefaultReservedCapacity)),
      config::CapacityUnit::BYTE);
}

uint64_t SharedArbitrator::ExtraConfig::getMemoryPoolInitialCapacity(
    const std::unordered_map<std::string, std::string>& configs) {
  return config::toCapacity(
      getConfig<std::string>(
          configs,
          kMemoryPoolInitialCapacity,
          std::string(kDefaultMemoryPoolInitialCapacity)),
      config::CapacityUnit::BYTE);
}

uint64_t SharedArbitrator::ExtraConfig::getMemoryPoolReservedCapacity(
    const std::unordered_map<std::string, std::string>& configs) {
  return config::toCapacity(
      getConfig<std::string>(
          configs,
          kMemoryPoolReservedCapacity,
          std::string(kDefaultMemoryPoolReservedCapacity)),
      config::CapacityUnit::BYTE);
}

uint64_t SharedArbitrator::ExtraConfig::getMemoryPoolTransferCapacity(
    const std::unordered_map<std::string, std::string>& configs) {
  return config::toCapacity(
      getConfig<std::string>(
          configs,
          kMemoryPoolTransferCapacity,
          std::string(kDefaultMemoryPoolTransferCapacity)),
      config::CapacityUnit::BYTE);
}

uint64_t SharedArbitrator::ExtraConfig::getMemoryReclaimMaxWaitTimeMs(
    const std::unordered_map<std::string, std::string>& configs) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             config::toDuration(getConfig<std::string>(
                 configs,
                 kMemoryReclaimMaxWaitTime,
                 std::string(kDefaultMemoryReclaimMaxWaitTime))))
      .count();
}

uint64_t SharedArbitrator::ExtraConfig::getMemoryPoolMinFreeCapacity(
    const std::unordered_map<std::string, std::string>& configs) {
  return config::toCapacity(
      getConfig<std::string>(
          configs,
          kMemoryPoolMinFreeCapacity,
          std::string(kDefaultMemoryPoolMinFreeCapacity)),
      config::CapacityUnit::BYTE);
}

double SharedArbitrator::ExtraConfig::getMemoryPoolMinFreeCapacityPct(
    const std::unordered_map<std::string, std::string>& configs) {
  return getConfig<double>(
      configs,
      kMemoryPoolMinFreeCapacityPct,
      kDefaultMemoryPoolMinFreeCapacityPct);
}

bool SharedArbitrator::ExtraConfig::getGlobalArbitrationEnabled(
    const std::unordered_map<std::string, std::string>& configs) {
  return getConfig<bool>(
      configs, kGlobalArbitrationEnabled, kDefaultGlobalArbitrationEnabled);
}

bool SharedArbitrator::ExtraConfig::getCheckUsageLeak(
    const std::unordered_map<std::string, std::string>& configs) {
  return getConfig<bool>(configs, kCheckUsageLeak, kDefaultCheckUsageLeak);
}

uint64_t
SharedArbitrator::ExtraConfig::getFastExponentialGrowthCapacityLimitBytes(
    const std::unordered_map<std::string, std::string>& configs) {
  return config::toCapacity(
      getConfig<std::string>(
          configs,
          kFastExponentialGrowthCapacityLimit,
          std::string(kDefaultFastExponentialGrowthCapacityLimit)),
      config::CapacityUnit::BYTE);
}

double SharedArbitrator::ExtraConfig::getSlowCapacityGrowPct(
    const std::unordered_map<std::string, std::string>& configs) {
  return getConfig<double>(
      configs, kSlowCapacityGrowPct, kDefaultSlowCapacityGrowPct);
}

SharedArbitrator::SharedArbitrator(const Config& config)
    : MemoryArbitrator(config),
      reservedCapacity_(ExtraConfig::getReservedCapacity(config.extraConfigs)),
      memoryPoolInitialCapacity_(
          ExtraConfig::getMemoryPoolInitialCapacity(config.extraConfigs)),
      memoryPoolTransferCapacity_(
          ExtraConfig::getMemoryPoolTransferCapacity(config.extraConfigs)),
      memoryArbitrationTimeMs_(
          ExtraConfig::getMemoryReclaimMaxWaitTimeMs(config.extraConfigs)),
      poolConfig_(
          ExtraConfig::getMemoryPoolReservedCapacity(config.extraConfigs),
          ExtraConfig::getFastExponentialGrowthCapacityLimitBytes(
              config.extraConfigs),
          ExtraConfig::getSlowCapacityGrowPct(config.extraConfigs),
          ExtraConfig::getMemoryPoolTransferCapacity(config.extraConfigs),
          ExtraConfig::getMemoryPoolMinFreeCapacity(config.extraConfigs),
          ExtraConfig::getMemoryPoolMinFreeCapacityPct(config.extraConfigs)),
      globalArbitrationEnabled_(
          ExtraConfig::getGlobalArbitrationEnabled(config.extraConfigs)),
      checkUsageLeak_(ExtraConfig::getCheckUsageLeak(config.extraConfigs)),

      freeReservedCapacity_(reservedCapacity_),
      freeNonReservedCapacity_(capacity_ - freeReservedCapacity_) {
  VELOX_CHECK_EQ(kind_, config.kind);
  VELOX_CHECK_LE(reservedCapacity_, capacity_);
  startArbitrationThread();
}

void SharedArbitrator::startArbitrationThread() {
  VELOX_CHECK_NULL(arbitrationThread_);
  arbitrationThread_ =
      std::make_unique<std::thread>([&]() { arbitrationThreadRun(); });
}

void SharedArbitrator::stopArbitrationThread() {
  VELOX_CHECK_NOT_NULL(arbitrationThread_);
  {
    std::lock_guard<std::mutex> l(stateLock_);
    VELOX_CHECK(!shutdown_);
    VELOX_CHECK(arbitrationWaiters_.empty());
    shutdown_ = true;
  }
  arbitrationThreadCv_.notify_one();
  arbitrationThread_->join();
  arbitrationThread_.reset();
}

void SharedArbitrator::wakeupArbitrationThread() {
  arbitrationThreadCv_.notify_one();
}

SharedArbitrator::~SharedArbitrator() {
  VELOX_CHECK(arbitrationPools_.empty());
  if (freeNonReservedCapacity_ + freeReservedCapacity_ != capacity_) {
    const std::string errMsg = fmt::format(
        "Unexpected free capacity leak in arbitrator: freeNonReservedCapacity_[{}] + freeReservedCapacity_[{}] != capacity_[{}])\\n{}",
        freeNonReservedCapacity_,
        freeReservedCapacity_,
        capacity_,
        toString());
    if (checkUsageLeak_) {
      VELOX_FAIL(errMsg);
    } else {
      VELOX_MEM_LOG(ERROR) << errMsg;
    }
  }
  stopArbitrationThread();
}

void SharedArbitrator::startArbitration(ArbitrationOperation* op) {
  updateArbitrationRequestStats();
  ++numPending_;
  op->start();
}

void SharedArbitrator::finishArbitration(ArbitrationOperation* op) {
  VELOX_CHECK_GT(numPending_, 0);
  --numPending_;
  op->finish();

  const auto arbitrationTimeUs = getCurrentTimeMicro() - op->startTimeUs();
  if (arbitrationTimeUs != 0) {
    RECORD_HISTOGRAM_METRIC_VALUE(
        kMetricArbitratorArbitrationTimeMs, arbitrationTimeUs / 1'000);
    arbitrationTimeUs_ += arbitrationTimeUs;
    addThreadLocalRuntimeStat(
        kMemoryArbitrationWallNanos,
        RuntimeCounter(
            arbitrationTimeUs * 1'000, RuntimeCounter::Unit::kNanos));
  }

  if (op->localArbitrationWaitTimeUs() != 0) {
    addThreadLocalRuntimeStat(
        kLocalArbitrationWaitWallNanos,
        RuntimeCounter(
            op->localArbitrationWaitTimeUs() * 1'000,
            RuntimeCounter::Unit::kNanos));
  }
  if (op->globalArbitrationWaitTimeUs() != 0) {
    addThreadLocalRuntimeStat(
        kGlobalArbitrationWaitWallNanos,
        RuntimeCounter(
            op->globalArbitrationWaitTimeUs() * 1'000,
            RuntimeCounter::Unit::kNanos));
  }

  const uint64_t waitTimeUs =
      op->localArbitrationWaitTimeUs() + op->globalArbitrationWaitTimeUs();
  if (waitTimeUs != 0) {
    RECORD_HISTOGRAM_METRIC_VALUE(
        kMetricArbitratorWaitTimeMs, waitTimeUs / 1'000);
    waitTimeUs_ += waitTimeUs;
  }
}

void SharedArbitrator::addPool(const std::shared_ptr<MemoryPool>& memPool) {
  VELOX_CHECK_EQ(memPool->capacity(), 0);
  VELOX_CHECK_LE(memPool->maxCapacity(), capacity_);

  auto newPool =
      std::make_shared<ArbitrationPool>(nextPoolId_++, memPool, &poolConfig_);
  {
    std::unique_lock guard{poolLock_};
    VELOX_CHECK_EQ(
        arbitrationPools_.count(memPool->name()),
        0,
        "Memory pool {} already exists",
        memPool->name());
    arbitrationPools_.emplace(memPool->name(), newPool);
  }

  auto scopedPool = newPool->get();
  std::vector<ContinuePromise> arbitrationWaiters;
  {
    std::lock_guard<std::mutex> l(stateLock_);
    const uint64_t maxBytesToReserve =
        std::min(scopedPool->maxCapacity(), memoryPoolInitialCapacity_);
    const uint64_t minBytesToReserve = scopedPool->minCapacity();
    uint64_t allocatedBytes{0};
    const auto ret = allocateCapacityLocked(
        scopedPool->id(),
        minBytesToReserve,
        maxBytesToReserve,
        minBytesToReserve,
        allocatedBytes);
    VELOX_CHECK(ret || allocatedBytes == 0);
    if (allocatedBytes > 0) {
      try {
        checkedGrow(scopedPool, allocatedBytes, 0);
      } catch (const VeloxRuntimeError& e) {
        VELOX_MEM_LOG(ERROR)
            << "Initial capacity allocation failure for memory pool: "
            << scopedPool->name() << "\n"
            << e.what();
        freeCapacityLocked(allocatedBytes, arbitrationWaiters);
      }
    }
  }
  for (auto& waiter : arbitrationWaiters) {
    waiter.setValue();
  }
}

void SharedArbitrator::removePool(MemoryPool* pool) {
  VELOX_CHECK_EQ(pool->reservedBytes(), 0);
  const uint64_t freedBytes = shrinkPool(pool, 0);
  VELOX_CHECK_EQ(pool->capacity(), 0);
  freeCapacity(freedBytes);

  std::unique_lock guard{poolLock_};
  const auto ret = arbitrationPools_.erase(pool->name());
  VELOX_CHECK_EQ(ret, 1);
}

std::vector<ArbitrationCandidate> SharedArbitrator::getCandidates(
    bool freeCapacityOnly) {
  std::vector<ArbitrationCandidate> candidates;
  std::shared_lock guard{poolLock_};
  candidates.reserve(arbitrationPools_.size());
  for (const auto& entry : arbitrationPools_) {
    auto candidate = entry.second->get();
    if (!candidate.valid()) {
      continue;
    }
    candidates.push_back({std::move(candidate), freeCapacityOnly});
  }
  VELOX_CHECK(!candidates.empty());
  return candidates;
}

void SharedArbitrator::sortCandidatesByReclaimableFreeCapacity(
    std::vector<ArbitrationCandidate>& candidates) {
  std::sort(
      candidates.begin(),
      candidates.end(),
      [&](const ArbitrationCandidate& lhs, const ArbitrationCandidate& rhs) {
        return lhs.freeBytes > rhs.freeBytes;
      });
  TestValue::adjust(
      "facebook::velox::memory::SharedArbitrator::sortCandidatesByReclaimableFreeCapacity",
      &candidates);
}

void SharedArbitrator::sortCandidatesByReclaimableUsedCapacity(
    std::vector<ArbitrationCandidate>& candidates) {
  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const ArbitrationCandidate& lhs, const ArbitrationCandidate& rhs) {
        return lhs.reclaimableBytes > rhs.reclaimableBytes;
      });

  TestValue::adjust(
      "facebook::velox::memory::SharedArbitrator::sortCandidatesByReclaimableUsedCapacity",
      &candidates);
}

void SharedArbitrator::sortCandidatesByUsage(
    std::vector<ArbitrationCandidate>& candidates) {
  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const ArbitrationCandidate& lhs, const ArbitrationCandidate& rhs) {
        return lhs.reservedBytes > rhs.reservedBytes;
      });
}

ArbitrationCandidate SharedArbitrator::findCandidateWithLargestCapacity() {
  const auto candidates = getCandidates();
  VELOX_CHECK(!candidates.empty());
  int32_t candidateIdx{-1};
  uint64_t maxCapacity{0};
  for (int32_t i = 0; i < candidates.size(); ++i) {
    const uint64_t capacity = candidates[i].pool->capacity();
    if (i == 0) {
      candidateIdx = 0;
      maxCapacity = capacity;
      continue;
    }
    if (capacity < maxCapacity) {
      continue;
    }
    if (capacity > maxCapacity) {
      candidateIdx = i;
      maxCapacity = capacity;
      continue;
    }
    // With the same amount of capacity, we favor the old query which has a
    // smaller memory pool id.
    if (candidates[candidateIdx].pool->id() < candidates[i].pool->id()) {
      candidateIdx = i;
    }
  }
  VELOX_CHECK_NE(candidateIdx, -1);
  return candidates[candidateIdx];
}

ArbitrationCandidate
SharedArbitrator::findCandidateWithLargestReclaimableBytes() {
  const auto candidates = getCandidates();
  VELOX_CHECK(!candidates.empty());
  int32_t candidateIdx{-1};
  uint64_t maxReclaimableBytes{0};
  for (int32_t i = 0; i < candidates.size(); ++i) {
    const uint64_t reclaimableBytes = candidates[i].reclaimableBytes;
    if (i == 0) {
      candidateIdx = 0;
      continue;
    }
    if (reclaimableBytes < maxReclaimableBytes) {
      continue;
    }
    if (reclaimableBytes > maxReclaimableBytes) {
      candidateIdx = i;
      maxReclaimableBytes = reclaimableBytes;
      continue;
    }
    // With the same amount of capacity, we favor the old query which has a
    // smaller memory pool id.
    if (candidates[candidateIdx].pool->id() < candidates[i].pool->id()) {
      candidateIdx = i;
    }
  }
  VELOX_CHECK_NE(candidateIdx, -1);
  return candidates[candidateIdx];
}

void SharedArbitrator::updateArbitrationRequestStats() {
  RECORD_METRIC_VALUE(kMetricArbitratorRequestsCount);
  ++numRequests_;
}

void SharedArbitrator::updateArbitrationFailureStats() {
  RECORD_METRIC_VALUE(kMetricArbitratorFailuresCount);
  ++numFailures_;
}

bool SharedArbitrator::allocateCapacity(
    uint64_t requestPoolId,
    uint64_t requestBytes,
    uint64_t maxAllocateBytes,
    uint64_t minAllocateBytes,
    uint64_t& allocatedBytes) {
  std::lock_guard<std::mutex> l(stateLock_);
  return allocateCapacityLocked(
      requestPoolId,
      requestBytes,
      maxAllocateBytes,
      minAllocateBytes,
      allocatedBytes);
}

bool SharedArbitrator::allocateCapacityLocked(
    uint64_t requestPoolId,
    uint64_t requestBytes,
    uint64_t maxAllocateBytes,
    uint64_t minAllocateBytes,
    uint64_t& allocatedBytes) {
  VELOX_CHECK_LE(requestBytes, maxAllocateBytes);

  allocatedBytes = 0;
  if (FOLLY_UNLIKELY(!arbitrationWaiters_.empty())) {
    if ((requestPoolId > arbitrationWaiters_.begin()->first) &&
        (requestBytes > minAllocateBytes)) {
      return false;
    }
    maxAllocateBytes = std::max(requestBytes, minAllocateBytes);
  }

  const uint64_t nonReservedBytes =
      std::min<uint64_t>(freeNonReservedCapacity_, maxAllocateBytes);
  if (nonReservedBytes >= maxAllocateBytes) {
    freeNonReservedCapacity_ -= nonReservedBytes;
    allocatedBytes = nonReservedBytes;
    return true;
  }
  uint64_t reservedBytes{0};
  if (nonReservedBytes < minAllocateBytes) {
    reservedBytes =
        std::min(minAllocateBytes - nonReservedBytes, freeReservedCapacity_);
  }
  if (FOLLY_UNLIKELY(nonReservedBytes + reservedBytes < requestBytes)) {
    return false;
  }

  freeNonReservedCapacity_ -= nonReservedBytes;
  freeReservedCapacity_ -= reservedBytes;
  allocatedBytes = nonReservedBytes + reservedBytes;
  return true;
}

uint64_t SharedArbitrator::shrinkCapacity(
    MemoryPool* memPool,
    uint64_t requestBytes) {
  auto pool = getPool(memPool->name());
  VELOX_CHECK(pool.valid());
  const auto freedBytes = pool->shrink(/*reclaimAll=*/true);
  reclaimedFreeBytes_ += freedBytes;
  freeCapacity(freedBytes);
  return freedBytes;
}

uint64_t SharedArbitrator::shrinkCapacity(
    uint64_t requestBytes,
    bool allowSpill,
    bool allowAbort) {
  const uint64_t targetBytes = requestBytes == 0 ? capacity_ : requestBytes;

  RECORD_METRIC_VALUE(kMetricArbitratorSlowGlobalArbitrationCount);

  uint64_t reclaimedBytes{0};
  if (allowSpill) {
    auto candidates = getCandidates();
    sortCandidatesByReclaimableUsedCapacity(candidates);
    for (auto& candidate : candidates) {
      reclaimedBytes += reclaimUsedMemoryBySpill(
          candidate.pool, targetBytes - reclaimedBytes);
      if (reclaimedBytes >= targetBytes) {
        return reclaimedBytes;
      }
    }
  }

  if (allowAbort) {
    auto candidates = getCandidates();
    sortCandidatesByUsage(candidates);
    for (auto& candidate : candidates) {
      reclaimedBytes += reclaimUsedMemoryByAbort(candidate.pool);
      if (reclaimedBytes >= targetBytes) {
        return reclaimedBytes;
      }
    }
  }
  return reclaimedBytes;
}

void SharedArbitrator::testingFreeCapacity(uint64_t capacity) {
  std::lock_guard<std::mutex> l(stateLock_);
  // freeCapacityLocked(capacity);
}

uint64_t SharedArbitrator::testingNumRequests() const {
  return numRequests_;
}

ArbitrationOperation SharedArbitrator::createArbitrationOperation(
    MemoryPool* memPool,
    uint64_t requestBytes) {
  VELOX_CHECK_NOT_NULL(memPool);
  ScopedArbitrationPool pool = getPool(memPool->name());
  VELOX_CHECK(pool.valid(), "Memory pool {} is invalid", memPool->name());
  return ArbitrationOperation(
      std::move(pool), requestBytes, memoryArbitrationTimeMs_);
}

bool SharedArbitrator::growCapacity(
    MemoryPool* memPool,
    uint64_t requestBytes) {
  auto op = createArbitrationOperation(memPool, requestBytes);
  ScopedArbitration scopedArbitration(this, &op);

  TestValue::adjust(
      "facebook::velox::memory::SharedArbitrator::runLocalArbitration", this);
  checkIfAborted(op);

  RETURN_IF_TRUE(maybeGrowFromSelf(op));

  if (!ensureCapacity(op)) {
    updateArbitrationFailureStats();
    VELOX_MEM_LOG(ERROR) << "Can't grow " << op.pool()->name()
                         << " capacity to "
                         << succinctBytes(
                                op.pool()->capacity() + op.requestBytes())
                         << " which exceeds its max capacity "
                         << succinctBytes(op.pool()->maxCapacity())
                         << ", current capacity "
                         << succinctBytes(op.pool()->capacity()) << ", request "
                         << succinctBytes(op.requestBytes());
    return false;
  }
  op.setGrowTargets();

  checkIfAborted(op);

  RETURN_IF_TRUE(maybeGrowFromSelf(op));
  RETURN_IF_TRUE(growWithFreeCapacity(op));

  reclaimUnusedCapacity();

  RETURN_IF_TRUE(growWithFreeCapacity(op));

  if (!globalArbitrationEnabled_) {
    reclaim(op.pool(), op.requestBytes(), op.timeoutMs(), /*selfReclaim=*/true);
    checkIfAborted(op);
    RETURN_IF_TRUE(maybeGrowFromSelf(op));
    RETURN_IF_TRUE(growWithFreeCapacity(op));
    return false;
  }

  return startAndWaitArbitration(op);
}

bool SharedArbitrator::startAndWaitArbitration(ArbitrationOperation& op) {
  checkIfTimeout(op);

  incrementGlobalArbitrationCount();
  ContinueFuture waitPromise{ContinueFuture::makeEmpty()};
  uint64_t allocatedBytes{0};
  {
    std::lock_guard<std::mutex> l(stateLock_);
    if (allocateCapacityLocked(
            op.pool()->id(),
            op.requestBytes(),
            op.maxGrowBytes(),
            op.minGrowBytes(),
            allocatedBytes)) {
      VELOX_CHECK_EQ(allocatedBytes, op.requestBytes());
    } else {
      ArbitrationWait arbitrationWait{
          &op,
          ContinuePromise{fmt::format(
              "Wait for memory arbitration {} with {}",
              op.pool()->name(),
              succinctBytes(op.requestBytes()))}};
      waitPromise = arbitrationWait.resumePromise.getSemiFuture();
      arbitrationWaiters_.emplace(op.id(), std::move(arbitrationWait));
    }
  }

  TestValue::adjust(
      "facebook::velox::memory::SharedArbitrator::startAndWaitArbitration",
      this);

  if (waitPromise.valid()) {
    wakeupArbitrationThread();

    uint64_t waitTimeUs{0};
    {
      MicrosecondTimer timer(&waitTimeUs);
      const bool timeout =
          !std::move(waitPromise)
               .wait(std::chrono::milliseconds(op.timeoutMs()));
      if (timeout) {
        VELOX_MEM_LOG_EVERY_MS(ERROR, 1'000)
            << "Wait for memory arbitration timed out for pool: "
            << op.pool()->name() << " after "
            << succinctMillis(op.executionTimeMs());
        removeArbitrationWaiter(op.id());
      }
    }
    allocatedBytes = op.arbitratedBytes();
    if (allocatedBytes == 0) {
      checkIfAborted(op);
      return false;
    }
  }
  VELOX_CHECK_GE(allocatedBytes, op.requestBytes());
  CHECKED_GROW(op.pool(), allocatedBytes, op.requestBytes());
  return true;
}

void SharedArbitrator::runArbitrationLoop() {
  for (int i = 0; i < 4; ++i) {
    const uint64_t targetBytes = getArbitrationTarget();
    if (targetBytes == 0) {
      return;
    }
    if (i < 3) {
      reclaimUsedMemoryBySpill(targetBytes);
    } else {
      reclaimUsedMemoryByAbort();
    }
    reclaimUnusedCapacity();
  }
}

void SharedArbitrator::arbitrationThreadRun() {
  VELOX_MEM_LOG(INFO) << "Arbitration thread starts to run";

  while (true) {
    uint64_t targetBytes{0};
    {
      std::unique_lock l(stateLock_);
      arbitrationThreadCv_.wait(
          l, [&] { return shutdown_ || !arbitrationWaiters_.empty(); });
      if (shutdown_) {
        VELOX_CHECK(arbitrationWaiters_.empty());
        break;
      }
    }

    runArbitrationLoop();
  }
  VELOX_MEM_LOG(INFO) << "Arbitration thread stopped";
}

uint64_t SharedArbitrator::getArbitrationTarget() {
  std::lock_guard<std::mutex> l(stateLock_);
  return getArbitrationTargetLocked();
}

uint64_t SharedArbitrator::getArbitrationTargetLocked() {
  uint64_t targetBytes{0};
  for (auto& waiter : arbitrationWaiters_) {
    targetBytes += waiter.second.op->requestBytes();
  }
  return targetBytes;
}

void SharedArbitrator::getGrowTargets(
    ArbitrationOperation& op,
    uint64_t& maxGrowTarget,
    uint64_t& minGrowTarget) {
  op.pool()->getGrowTargets(op.requestBytes(), maxGrowTarget, minGrowTarget);
}

void SharedArbitrator::checkIfAborted(ArbitrationOperation& op) {
  op.pool()->checkIfAborted();
}

void SharedArbitrator::checkIfTimeout(ArbitrationOperation& op) {
  VELOX_CHECK(
      !op.hasTimeout(),
      "Memory arbitration operation on pool: {} timed out after {}/{}",
      op.pool()->name(),
      op.executionTimeMs(),
      op.timeoutMs());
}

bool SharedArbitrator::maybeGrowFromSelf(ArbitrationOperation& op) {
  if (op.pool()->grow(0, op.requestBytes())) {
    return true;
  }
  return false;
}

bool SharedArbitrator::growWithFreeCapacity(ArbitrationOperation& op) {
  uint64_t freedBytes{0};
  if (allocateCapacity(
          op.pool()->id(),
          op.requestBytes(),
          op.maxGrowBytes(),
          op.maxGrowBytes(),
          freedBytes)) {
    VELOX_CHECK_GE(freedBytes, op.requestBytes());
    CHECKED_GROW(op.pool(), freedBytes, op.requestBytes());
    return true;
  }
  VELOX_CHECK_EQ(freedBytes, 0);
  return false;
}

ScopedArbitrationPool SharedArbitrator::getPool(const std::string& name) const {
  std::shared_lock guard{poolLock_};
  auto it = arbitrationPools_.find(name);
  VELOX_CHECK(
      it != arbitrationPools_.end(), "Arbitration pool {} not found", name);
  return it->second->get();
}

bool SharedArbitrator::checkCapacityGrowth(ArbitrationOperation& op) const {
  if (!op.pool()->checkCapacityGrowth(op.requestBytes())) {
    return false;
  }
  return (op.pool()->capacity() + op.requestBytes()) <= capacity_;
}

bool SharedArbitrator::ensureCapacity(ArbitrationOperation& op) {
  if ((op.requestBytes() > capacity_) ||
      (op.requestBytes() > op.pool()->maxCapacity())) {
    return false;
  }
  if (checkCapacityGrowth(op)) {
    return true;
  }

  reclaim(op.pool(), op.requestBytes(), op.timeoutMs(), /*selfReclaim=*/true);
  // Checks if the requestor has been aborted in reclaim operation above.
  checkIfAborted(op);
  return checkCapacityGrowth(op);
}

void SharedArbitrator::checkedGrow(
    ScopedArbitrationPool& pool,
    uint64_t growBytes,
    uint64_t reservationBytes) {
  const auto ret = pool->grow(growBytes, reservationBytes);
  VELOX_CHECK(
      ret,
      "Failed to grow pool {} with {} and commit {} used reservation",
      pool->name(),
      succinctBytes(growBytes),
      succinctBytes(reservationBytes));
}

uint64_t SharedArbitrator::reclaimUnusedCapacity() {
  std::vector<ArbitrationCandidate> candidates =
      getCandidates(/*freeCapacityOnly=*/true);
  // Sort candidate memory pools based on their reclaimable free capacity.
  sortCandidatesByReclaimableFreeCapacity(candidates);

  uint64_t reclaimedBytes{0};
  SCOPE_EXIT {
    freeCapacity(reclaimedBytes);
  };
  for (const auto& candidate : candidates) {
    if (candidate.freeBytes == 0) {
      break;
    }
    reclaimedBytes += candidate.pool->shrink(/*reclaimAll=*/false);
  }
  reclaimedFreeBytes_ += reclaimedBytes;
  return reclaimedBytes;
}

uint64_t SharedArbitrator::reclaimUsedMemoryBySpill(uint64_t targetBytes) {
  auto victim = findCandidateWithLargestReclaimableBytes();
  return reclaim(
      victim.pool,
      targetBytes,
      memoryArbitrationTimeMs_,
      /*selfReclaim=*/false);
}

uint64_t SharedArbitrator::reclaimUsedMemoryBySpill(
    ScopedArbitrationPool& pool,
    uint64_t targetBytes) {
  return reclaim(
      pool,
      targetBytes,
      memoryArbitrationTimeMs_,
      /*selfReclaim=*/false);
}

uint64_t SharedArbitrator::reclaimUsedMemoryByAbort() {
  auto victim = findCandidateWithLargestCapacity();
  return reclaimUsedMemoryByAbort(victim.pool);
}

uint64_t SharedArbitrator::reclaimUsedMemoryByAbort(
    ScopedArbitrationPool& pool) {
  try {
    VELOX_MEM_POOL_ABORTED("aborted here");
  } catch (VeloxRuntimeError&) {
    return abort(pool, std::current_exception());
  }
}

uint64_t SharedArbitrator::reclaim(
    ScopedArbitrationPool& pool,
    uint64_t targetBytes,
    uint64_t timeoutMs,
    bool selfReclaim) noexcept {
  if (selfReclaim) {
    incrementLocalArbitrationCount();
  }
  const int64_t bytesToReclaim =
      std::max(targetBytes, memoryPoolTransferCapacity_);
  uint64_t reclaimTimeUs{0};
  uint64_t reclaimedBytes{0};
  {
    MicrosecondTimer reclaimTimer(&reclaimTimeUs);
    reclaimedBytes = pool->reclaim(bytesToReclaim, timeoutMs, selfReclaim);
  }
  reclaimedUsedBytes_ += reclaimedBytes;
  reclaimTimeUs_ += reclaimTimeUs;
  freeCapacity(reclaimedBytes);
  VELOX_MEM_LOG(INFO) << "Reclaimed from memory pool " << pool->name()
                      << " with target of " << succinctBytes(targetBytes)
                      << ", reclaimed " << succinctBytes(reclaimedBytes)
                      << ", spent " << succinctMicros(reclaimTimeUs)
                      << ", self-reclaim: " << selfReclaim;
  return reclaimedBytes;
}

uint64_t SharedArbitrator::abort(
    ScopedArbitrationPool& pool,
    const std::exception_ptr& error) {
  RECORD_METRIC_VALUE(kMetricArbitratorAbortedCount);
  ++numAborted_;
  pool->abort(error);
  // NOTE: no matter memory pool abort throws or not, it should have been marked
  // as aborted to prevent any new memory arbitration triggered from the aborted
  // memory pool.
  VELOX_CHECK(pool->aborted());
  const uint64_t freedBytes = pool->shrink(/*reclaimAll=*/true);
  reclaimedUsedBytes_ += freedBytes;
  removeArbitrationWaiter(pool->id());
  freeCapacity(freedBytes);
  return freedBytes;
}

void SharedArbitrator::freeCapacity(uint64_t bytes) {
  if (FOLLY_UNLIKELY(bytes == 0)) {
    return;
  }
  std::vector<ContinuePromise> resumes;
  {
    std::lock_guard<std::mutex> l(stateLock_);
    freeCapacityLocked(bytes, resumes);
  }
  for (auto& resume : resumes) {
    resume.setValue();
  }
}

void SharedArbitrator::freeCapacityLocked(
    uint64_t bytes,
    std::vector<ContinuePromise>& resumes) {
  freeReservedCapacityLocked(bytes);
  freeNonReservedCapacity_ += bytes;
  if (FOLLY_UNLIKELY(
          freeNonReservedCapacity_ + freeReservedCapacity_ > capacity_)) {
    VELOX_FAIL(
        "The free capacity {}/{} is larger than the max capacity {}, {}",
        succinctBytes(freeNonReservedCapacity_),
        succinctBytes(freeReservedCapacity_),
        succinctBytes(capacity_),
        toStringLocked());
  }
  resumeArbitrationWaitersLocked(resumes);
}

void SharedArbitrator::resumeArbitrationWaiters() {
  std::vector<ContinuePromise> resumes;
  {
    std::lock_guard<std::mutex> l(stateLock_);
    resumeArbitrationWaitersLocked(resumes);
  }
}

void SharedArbitrator::resumeArbitrationWaitersLocked(
    std::vector<ContinuePromise>& resumes) {
  auto it = arbitrationWaiters_.begin();
  while (it != arbitrationWaiters_.end()) {
    uint64_t allocatedBytes{0};
    auto* op = it->second.op;
    if (!allocateCapacityLocked(
            op->pool()->id(),
            op->requestBytes(),
            op->requestBytes(),
            op->minGrowBytes(),
            allocatedBytes)) {
      VELOX_CHECK_EQ(allocatedBytes, 0);
      break;
    }
    VELOX_CHECK_EQ(allocatedBytes, op->requestBytes());
    VELOX_CHECK_EQ(op->arbitratedBytes(), 0);
    op->arbitratedBytes() = allocatedBytes;
    resumes.push_back(std::move(it->second.resumePromise));
    it = arbitrationWaiters_.erase(it);
  }
}

void SharedArbitrator::removeArbitrationWaiter(uint64_t id) {
  ContinuePromise resume = ContinuePromise::makeEmpty();
  {
    std::lock_guard<std::mutex> l(stateLock_);
    auto it = arbitrationWaiters_.find(id);
    if (it != arbitrationWaiters_.end()) {
      resume = std::move(it->second.resumePromise);
      arbitrationWaiters_.erase(it);
    }
  }
  if (resume.valid()) {
    resume.setValue();
  }
}

void SharedArbitrator::freeReservedCapacityLocked(uint64_t& bytes) {
  VELOX_CHECK_LE(freeReservedCapacity_, reservedCapacity_);
  const uint64_t freedBytes =
      std::min(bytes, reservedCapacity_ - freeReservedCapacity_);
  freeReservedCapacity_ += freedBytes;
  bytes -= freedBytes;
}

MemoryArbitrator::Stats SharedArbitrator::stats() const {
  std::lock_guard<std::mutex> l(stateLock_);
  return statsLocked();
}

MemoryArbitrator::Stats SharedArbitrator::statsLocked() const {
  Stats stats;
  stats.numRequests = numRequests_;
  stats.numAborted = numAborted_;
  stats.numFailures = numFailures_;
  stats.queueTimeUs = waitTimeUs_;
  stats.arbitrationTimeUs = arbitrationTimeUs_;
  stats.numShrunkBytes = reclaimedFreeBytes_;
  stats.numReclaimedBytes = reclaimedUsedBytes_;
  stats.maxCapacityBytes = capacity_;
  stats.freeCapacityBytes = freeNonReservedCapacity_ + freeReservedCapacity_;
  stats.freeReservedCapacityBytes = freeReservedCapacity_;
  stats.reclaimTimeUs = reclaimTimeUs_;
  stats.numNonReclaimableAttempts = numNonReclaimableAttempts_;
  stats.numShrinks = numShrinks_;
  return stats;
}

std::string SharedArbitrator::toString() const {
  std::lock_guard<std::mutex> l(stateLock_);
  return toStringLocked();
}

std::string SharedArbitrator::toStringLocked() const {
  return fmt::format(
      "ARBITRATOR[{} CAPACITY[{}] PENDING[{}] {}]",
      kind_,
      succinctBytes(capacity_),
      numPending_,
      statsLocked().toString());
}

SharedArbitrator::ScopedArbitration::ScopedArbitration(
    SharedArbitrator* arbitrator,
    ArbitrationOperation* operation)
    : arbitrator_(arbitrator),
      operation_(operation),
      arbitrationCtx_(operation->pool()->memPool()),
      startTime_(std::chrono::steady_clock::now()) {
  VELOX_CHECK_NOT_NULL(arbitrator_);
  VELOX_CHECK_NOT_NULL(operation_);
  if (arbitrator_->arbitrationStateCheckCb_ != nullptr) {
    arbitrator_->arbitrationStateCheckCb_(*operation_->pool()->memPool());
  }
  arbitrator_->startArbitration(operation_);
}

SharedArbitrator::ScopedArbitration::~ScopedArbitration() {
  arbitrator_->finishArbitration(operation_);
}

std::string SharedArbitrator::kind() const {
  return kind_;
}

void SharedArbitrator::registerFactory() {
  MemoryArbitrator::registerFactory(
      kind_, [](const MemoryArbitrator::Config& config) {
        return std::make_unique<SharedArbitrator>(config);
      });
}

void SharedArbitrator::unregisterFactory() {
  MemoryArbitrator::unregisterFactory(kind_);
}

void SharedArbitrator::incrementGlobalArbitrationCount() {
  RECORD_METRIC_VALUE(kMetricArbitratorGlobalArbitrationCount);
  addThreadLocalRuntimeStat(
      kGlobalArbitrationCount, RuntimeCounter(1, RuntimeCounter::Unit::kNone));
}

void SharedArbitrator::incrementLocalArbitrationCount() {
  RECORD_METRIC_VALUE(kMetricArbitratorLocalArbitrationCount);
  addThreadLocalRuntimeStat(
      kLocalArbitrationCount, RuntimeCounter(1, RuntimeCounter::Unit::kNone));
}
} // namespace facebook::velox::memory
