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

#include "velox/experimental/cudf/exec/GpuMemoryLedger.h"
#include "velox/experimental/cudf/exec/GpuMemoryNvtx.h"

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Task.h"

#include <folly/concurrency/ConcurrentHashMap.h>
#include <folly/hash/Hash.h>

#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

namespace {

/// Separates fields of a composite key, so concatenation cannot collide with a
/// different split of the same characters.
constexpr char kKeySeparator = '\x1f';

/// The owner and process levels, the only two that report all three values.
///
/// Cache-line padded: adjacent records are updated by different threads, and
/// false sharing between them would reintroduce the contention this design
/// exists to remove.
struct alignas(64) AtomicLiveBytes {
  std::atomic<int64_t> current{0};
  std::atomic<int64_t> peak{0};
  std::atomic<int64_t> total{0};
};

/// The query level. Every operator in a query hits this one record, so counters
/// no reader consumes are omitted rather than contended.
struct alignas(64) QueryLiveBytes {
  std::atomic<int64_t> current{0};
  std::atomic<int64_t> peak{0};
};

/// The fragment and plan node levels. No peak: the maximum of a level's own
/// live row already carries it.
struct alignas(64) SharedLiveBytes {
  std::atomic<int64_t> current{0};
};

/// Spins only while memory is growing, so it is uncontended in steady state.
void raisePeak(std::atomic<int64_t>& peak, int64_t updated) {
  auto observed = peak.load(std::memory_order_relaxed);
  while (observed < updated &&
         !peak.compare_exchange_weak(
             observed, updated, std::memory_order_relaxed)) {
  }
}

/// 'current' is acq_rel rather than relaxed so that a reader seeing the new
/// live value also sees the total that was raised before it. That ordering is
/// the only reason the "live never exceeds total" invariant holds mid-flight.
int64_t applyDelta(AtomicLiveBytes& counter, int64_t deltaBytes) {
  if (deltaBytes >= 0) {
    counter.total.fetch_add(deltaBytes, std::memory_order_relaxed);
  }
  const auto updated =
      counter.current.fetch_add(deltaBytes, std::memory_order_acq_rel) +
      deltaBytes;
  if (deltaBytes >= 0) {
    raisePeak(counter.peak, updated);
  }
  return updated;
}

int64_t applyDelta(QueryLiveBytes& counter, int64_t deltaBytes) {
  const auto updated =
      counter.current.fetch_add(deltaBytes, std::memory_order_relaxed) +
      deltaBytes;
  if (deltaBytes >= 0) {
    raisePeak(counter.peak, updated);
  }
  return updated;
}

int64_t applyDelta(SharedLiveBytes& counter, int64_t deltaBytes) {
  return counter.current.fetch_add(deltaBytes, std::memory_order_relaxed) +
      deltaBytes;
}

uint64_t toUnsigned(int64_t value) {
  return value < 0 ? 0 : static_cast<uint64_t>(value);
}

GpuMemoryOwner unattributedOwner() {
  return GpuMemoryOwner{
      .taskUuid = "<unattributed>",
      .taskId = "<unattributed>",
      .queryId = "<unattributed>",
      .planNodeId = "<unattributed>",
      .planNodeType = "<unattributed>",
      .pipelineId = -1,
      .driverId = -1,
      .operatorId = -1,
      .operatorType = "<unattributed>"};
}

bool isEmptyOwner(const GpuMemoryOwner& owner) {
  return owner.taskUuid.empty() && owner.taskId.empty() &&
      owner.queryId.empty() && owner.planNodeId.empty() &&
      owner.pipelineId < 0 && owner.driverId < 0 && owner.operatorId < 0 &&
      owner.operatorType.empty();
}

void hashCombine(std::size_t& seed, std::size_t value) {
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

struct GpuMemoryOwnerHash {
  std::size_t operator()(const GpuMemoryOwner& owner) const {
    std::size_t result = std::hash<std::string>{}(owner.taskUuid);
    hashCombine(result, std::hash<std::string>{}(owner.taskId));
    hashCombine(result, std::hash<std::string>{}(owner.queryId));
    hashCombine(result, std::hash<std::string>{}(owner.planNodeId));
    hashCombine(result, std::hash<int32_t>{}(owner.pipelineId));
    hashCombine(result, std::hash<int32_t>{}(owner.driverId));
    hashCombine(result, std::hash<int32_t>{}(owner.operatorId));
    hashCombine(result, std::hash<std::string>{}(owner.operatorType));
    return result;
  }
};

/// Ancestor counters and NVTX ids are resolved once here, which is what leaves
/// the allocation path with no lookups.
struct OwnerRecord {
  GpuMemoryOwnerHandle handle;
  GpuMemoryOwner owner;
  AtomicLiveBytes bytes;
  QueryLiveBytes* query{nullptr};
  SharedLiveBytes* task{nullptr};
  SharedLiveBytes* planNode{nullptr};
  GpuMemoryNvtxCounters nvtxCounters;
};

struct LiveAllocation {
  uint64_t ownerId{0};
  uint64_t bytes{0};
};

/// Spreads device addresses across the map's segments.
///
/// folly picks a segment from the low bits of the hash, std::hash of a pointer
/// is the identity, and device allocations are at least 256-byte aligned. The
/// default hash therefore lands every allocation in segment 0.
struct AllocationAddressHash {
  std::size_t operator()(const void* address) const noexcept {
    return folly::hash::twang_mix64(reinterpret_cast<std::uintptr_t>(address));
  }
};

} // namespace

class GpuMemoryLedger::Impl {
 public:
  Impl() {
    // Explicit, so an allocation made outside any operator is reported rather
    // than dropped or guessed at.
    auto* record = createOwnerLocked(
        unattributedOwner(), 0, "<unattributed>", GpuMemoryPlanLocation{});
    VELOX_CHECK_EQ(record->handle.ownerId, 0);
  }

  GpuMemoryOwnerHandle registerOwner(
      const GpuMemoryOwner& owner,
      const GpuMemoryPlanLocation& planLocation) {
    if (isEmptyOwner(owner)) {
      return {};
    }

    std::lock_guard<std::mutex> lock(registrationMutex_);
    const auto existing = ownerIds_.find(owner);
    if (existing != ownerIds_.end()) {
      return existing->second->handle;
    }
    // Keyed on the resolved plan node, not the operator's own id: synthetic
    // conversion operators carry a suffixed id that resolves to the same node,
    // and the raw value would split that node's bytes across two aggregates.
    const auto* resolvedNode = planLocation.node();
    const auto& resolvedPlanNodeId =
        resolvedNode == nullptr ? owner.planNodeId : resolvedNode->planNodeId;
    return createOwnerLocked(
               owner,
               nextOwnerId_++,
               planNodeKeyFor(owner, resolvedPlanNodeId),
               planLocation)
        ->handle;
  }

  std::optional<GpuMemoryOwnerHandle> findOwner(
      const GpuMemoryOwner& owner) const {
    if (isEmptyOwner(owner)) {
      return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(registrationMutex_);
    const auto existing = ownerIds_.find(owner);
    if (existing == ownerIds_.end()) {
      return std::nullopt;
    }
    return existing->second->handle;
  }

  std::optional<GpuMemoryUpdate> recordAllocation(
      void* address,
      std::size_t bytes,
      GpuMemoryOwnerHandle handle) noexcept {
    if (address == nullptr) {
      return std::nullopt;
    }

    try {
      auto* record = ownerRecord(handle.ownerId);
      if (record == nullptr) {
        noteDataLoss("allocation for an unregistered owner");
        return std::nullopt;
      }

      const auto inserted = allocations_.insert(
          address, LiveAllocation{record->handle.ownerId, bytes});
      if (!inserted.second) {
        noteDataLoss("duplicate live allocation address");
        return std::nullopt;
      }

      const auto update =
          applyToAllLevels(*record, static_cast<int64_t>(bytes));
      if (record->handle.ownerId == 0) {
        ensureUnattributedCountersRegistered(*record);
      }
      sampleGpuMemoryNvtxCounters(record->nvtxCounters, update);
      return update;
    } catch (...) {
      noteDataLoss("allocation accounting exception");
      return std::nullopt;
    }
  }

  std::optional<GpuMemoryUpdate> recordDeallocation(
      void* address,
      std::size_t reportedBytes) noexcept {
    if (address == nullptr) {
      return std::nullopt;
    }

    try {
      // Safe as a separate find and erase because the wrapper erases before
      // calling upstream, so the address cannot be recycled in between.
      const auto allocation = allocations_.find(address);
      if (allocation == allocations_.cend()) {
        noteDataLoss("unknown deallocation address");
        return std::nullopt;
      }
      const auto live = allocation->second;
      allocations_.erase(address);

      // The allocator contract guarantees these match. A disagreement would
      // skew a counter, so surface it instead.
      if (reportedBytes != live.bytes) {
        noteDataLoss("deallocation size disagrees with recorded size");
      }

      auto* record = ownerRecord(live.ownerId);
      if (record == nullptr) {
        noteDataLoss("deallocation for an unregistered owner");
        return std::nullopt;
      }
      const auto update =
          applyToAllLevels(*record, -static_cast<int64_t>(live.bytes));
      sampleGpuMemoryNvtxCounters(record->nvtxCounters, update);
      return update;
    } catch (...) {
      noteDataLoss("deallocation accounting exception");
      return std::nullopt;
    }
  }

  GpuMemoryUpdate currentState(GpuMemoryOwnerHandle handle) const noexcept {
    try {
      const auto* record = ownerRecord(handle.ownerId);
      if (record == nullptr) {
        record = ownerRecord(0);
      }
      if (record == nullptr) {
        return {};
      }
      return GpuMemoryUpdate{
          record->handle.ownerId,
          record->handle.planNodeKey,
          toUnsigned(global_.current.load(std::memory_order_relaxed)),
          toUnsigned(global_.peak.load(std::memory_order_relaxed)),
          toUnsigned(record->query->current.load(std::memory_order_relaxed)),
          toUnsigned(record->query->peak.load(std::memory_order_relaxed)),
          toUnsigned(record->task->current.load(std::memory_order_relaxed)),
          toUnsigned(record->planNode->current.load(std::memory_order_relaxed)),
          toUnsigned(record->bytes.current.load(std::memory_order_relaxed))};
    } catch (...) {
      return {};
    }
  }

  GpuMemorySnapshot snapshot() const {
    GpuMemorySnapshot result;
    result.currentBytes = toUnsigned(global_.current.load());
    result.peakBytes = toUnsigned(global_.peak.load());
    result.totalBytes = toUnsigned(global_.total.load());
    result.liveAllocations = allocations_.size();
    result.dataLossEvents = dataLossEvents_.load();

    // Only pointers are gathered under the lock. Records never move and are
    // never erased, so copying each owner's identity strings happens afterwards
    // rather than on the mutex that registration also needs.
    std::vector<const OwnerRecord*> records;
    {
      std::lock_guard<std::mutex> lock(registrationMutex_);
      records.reserve(owners_.size());
      for (const auto& record : owners_) {
        records.push_back(record.get());
      }
    }

    result.owners.reserve(records.size());
    for (const auto* record : records) {
      result.owners.push_back(
          GpuMemoryOwnerSnapshot{
              record->handle,
              record->owner,
              toUnsigned(record->bytes.current.load()),
              toUnsigned(record->bytes.peak.load()),
              toUnsigned(record->bytes.total.load())});
    }

    std::sort(
        result.owners.begin(),
        result.owners.end(),
        [](const auto& left, const auto& right) {
          if (left.currentBytes != right.currentBytes) {
            return left.currentBytes > right.currentBytes;
          }
          return left.handle.ownerId < right.handle.ownerId;
        });
    return result;
  }

 private:
  std::string planNodeKeyFor(
      const GpuMemoryOwner& owner,
      const std::string& resolvedPlanNodeId) const {
    return owner.queryId + kKeySeparator + owner.taskUuid + kKeySeparator +
        resolvedPlanNodeId;
  }

  /// Creates the owner record and any missing ancestor records. Callers hold
  /// 'registrationMutex_', except the constructor.
  ///
  /// Takes 'planLocation' rather than letting the caller apply it, because the
  /// record is inserted into the lock-free lookup maps only once complete: it
  /// must never be reachable while a field is still being assigned.
  OwnerRecord* createOwnerLocked(
      const GpuMemoryOwner& owner,
      uint64_t ownerId,
      const std::string& planNodeKey,
      const GpuMemoryPlanLocation& planLocation) {
    auto* query = &queryLevels_[std::string(gpuMemoryQueryKey(owner))];
    auto* task = &sharedLevels_[owner.taskUuid + kKeySeparator + "task"];

    auto planNode = planNodeKeys_.find(planNodeKey);
    if (planNode == planNodeKeys_.end()) {
      planNode = planNodeKeys_.emplace(planNodeKey, nextPlanNodeKey_++).first;
    }
    auto* planNodeBytes = &sharedLevels_[planNodeKey + kKeySeparator + "plan"];

    owners_.push_back(std::make_unique<OwnerRecord>());
    auto* record = owners_.back().get();
    record->handle = GpuMemoryOwnerHandle{ownerId, planNode->second};
    record->owner = owner;
    record->query = query;
    record->task = task;
    record->planNode = planNodeBytes;
    // Owner 0 registers its row on first use instead. See
    // ensureUnattributedCountersRegistered().
    if (ownerId != 0) {
      record->nvtxCounters = registerGpuMemoryNvtxOwner(
          record->handle.ownerId,
          record->handle.planNodeKey,
          owner,
          planLocation);
    }

    // Published last, now that every field is set.
    ownerIds_.emplace(owner, record);
    ownerById_.insert(ownerId, record);
    return record;
  }

  OwnerRecord* ownerRecord(uint64_t ownerId) const {
    const auto found = ownerById_.find(ownerId);
    return found == ownerById_.cend() ? nullptr : found->second;
  }

  /// Registers the unattributed owner's row on first use, so a process that
  /// never allocates outside an operator shows no always-zero row.
  ///
  /// Unlike every other record this one is already reachable, having been built
  /// by the constructor, so the flag orders its single write against later
  /// reads.
  void ensureUnattributedCountersRegistered(OwnerRecord& record) {
    if (unattributedRegistered_.load(std::memory_order_acquire)) {
      return;
    }
    std::lock_guard<std::mutex> lock(registrationMutex_);
    if (unattributedRegistered_.load(std::memory_order_relaxed)) {
      return;
    }
    record.nvtxCounters = registerGpuMemoryNvtxUnattributedOwner();
    unattributedRegistered_.store(true, std::memory_order_release);
  }

  /// Applies one transition to the owner, plan node, task, query and process
  /// counters and returns the resulting values.
  GpuMemoryUpdate applyToAllLevels(OwnerRecord& record, int64_t deltaBytes) {
    const auto ownerCurrent = applyDelta(record.bytes, deltaBytes);
    const auto planNodeCurrent = applyDelta(*record.planNode, deltaBytes);
    const auto taskCurrent = applyDelta(*record.task, deltaBytes);
    const auto queryCurrent = applyDelta(*record.query, deltaBytes);
    const auto globalCurrent = applyDelta(global_, deltaBytes);

    return GpuMemoryUpdate{
        record.handle.ownerId,
        record.handle.planNodeKey,
        toUnsigned(globalCurrent),
        toUnsigned(global_.peak.load(std::memory_order_relaxed)),
        toUnsigned(queryCurrent),
        toUnsigned(record.query->peak.load(std::memory_order_relaxed)),
        toUnsigned(taskCurrent),
        toUnsigned(planNodeCurrent),
        toUnsigned(ownerCurrent)};
  }

  void noteDataLoss(std::string_view reason) noexcept {
    dataLossEvents_.fetch_add(1, std::memory_order_relaxed);
    markGpuMemoryDataLoss(reason);
  }

  AtomicLiveBytes global_;
  std::atomic<uint64_t> dataLossEvents_{0};

  /// Pointer ownership. Lock-free reads, fine-grained locking on writes.
  folly::ConcurrentHashMap<const void*, LiveAllocation, AllocationAddressHash>
      allocations_;
  /// Lock-free owner lookup on the allocation path.
  folly::ConcurrentHashMap<uint64_t, OwnerRecord*> ownerById_;

  /// Registration-only below. Records never move and are never erased, so the
  /// raw pointers published above stay valid for the ledger's lifetime.
  mutable std::mutex registrationMutex_;
  /// Whether the unattributed owner's counter row has been registered.
  std::atomic<bool> unattributedRegistered_{false};
  std::deque<std::unique_ptr<OwnerRecord>> owners_;
  std::unordered_map<GpuMemoryOwner, OwnerRecord*, GpuMemoryOwnerHash>
      ownerIds_;
  /// Byte counters for the query, task and plan node levels, keyed by a
  /// level-qualified name. std::unordered_map never relocates its values.
  /// Node-based, so pointers cached in owner records survive growth.
  std::unordered_map<std::string, QueryLiveBytes> queryLevels_;
  /// Fragment and plan node levels, keyed by identifier and level.
  std::unordered_map<std::string, SharedLiveBytes> sharedLevels_;
  std::unordered_map<std::string, uint64_t> planNodeKeys_;
  uint64_t nextOwnerId_{1};
  /// Starts at zero so the unattributed owner, registered first, keeps the
  /// all-zero handle that callers use to mean "no attribution".
  uint64_t nextPlanNodeKey_{0};
};

GpuMemoryLedger::GpuMemoryLedger() : impl_(std::make_unique<Impl>()) {}

GpuMemoryLedger::~GpuMemoryLedger() = default;

GpuMemoryOwnerHandle GpuMemoryLedger::registerOwner(
    const GpuMemoryOwner& owner,
    const GpuMemoryPlanLocation& planLocation) {
  return impl_->registerOwner(owner, planLocation);
}

GpuMemoryOwnerHandle GpuMemoryLedger::registerOperator(exec::Operator* op) {
  if (op == nullptr) {
    return {};
  }

  const auto* driverCtx = op->operatorCtx()->driverCtx();
  GpuMemoryOwner owner{
      .taskUuid = op->operatorCtx()->task()->uuid(),
      .taskId = op->taskId(),
      .queryId = op->operatorCtx()->task()->queryCtx()->queryId(),
      .planNodeId = op->planNodeId(),
      // Filled in below from the resolved plan node, and display-only, so it is
      // deliberately absent from the identity the lookup above uses.
      .planNodeType = "",
      .pipelineId = driverCtx->pipelineId,
      .driverId = driverCtx->driverId,
      .operatorId = op->operatorId(),
      .operatorType = op->operatorType()};
  if (const auto existing = impl_->findOwner(owner)) {
    return *existing;
  }

  const auto planLocation =
      gpuMemoryPlanLocation(*op->operatorCtx()->task(), op->planNodeId());
  if (const auto* node = planLocation.node()) {
    owner.planNodeType = node->planNodeType;
  }
  return impl_->registerOwner(owner, planLocation);
}

std::optional<GpuMemoryUpdate> GpuMemoryLedger::recordAllocation(
    void* address,
    std::size_t bytes,
    GpuMemoryOwnerHandle handle) noexcept {
  return impl_->recordAllocation(address, bytes, handle);
}

std::optional<GpuMemoryUpdate> GpuMemoryLedger::recordDeallocation(
    void* address,
    std::size_t bytes) noexcept {
  return impl_->recordDeallocation(address, bytes);
}

GpuMemoryUpdate GpuMemoryLedger::currentState(
    GpuMemoryOwnerHandle handle) const noexcept {
  return impl_->currentState(handle);
}

GpuMemorySnapshot GpuMemoryLedger::snapshot() const {
  return impl_->snapshot();
}

} // namespace facebook::velox::cudf_velox
