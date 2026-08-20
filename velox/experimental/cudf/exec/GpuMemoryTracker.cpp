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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuMemoryNvtx.h"
#include "velox/experimental/cudf/exec/GpuMemoryTracker.h"

#include "velox/common/base/RuntimeMetrics.h"
#include "velox/exec/Operator.h"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/failure_callback_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <glog/logging.h>

#include <memory>
#include <string_view>
#include <utility>

namespace facebook::velox::cudf_velox {

namespace {

/// The process-wide ledger. Atomic rather than mutex-guarded because reading it
/// is the whole operation; nothing else needs to be consistent with it.
std::atomic<std::shared_ptr<GpuMemoryLedger>> installedLedger;

/// Advanced once per compiled driver and whenever the ledger is replaced, which
/// is rare relative to allocation. See discardGpuMemoryOwnerCache().
std::atomic<uint64_t> resolutionGeneration{0};

std::shared_ptr<GpuMemoryLedger> currentLedger() {
  return installedLedger.load(std::memory_order_acquire);
}

/// Velox points the thread-local stat writer at the operator it is running,
/// which is how an allocation finds its owner without threading one through
/// every cuDF call.
exec::Operator* runtimeOperator() {
  return dynamic_cast<exec::Operator*>(getThreadLocalRunTimeStatWriter());
}

/// A driver thread runs one operator at a time and allocates repeatedly within
/// each call, so a single entry absorbs almost every lookup. Without it every
/// allocation copies five identity strings and takes the ledger's registration
/// mutex only to rediscover a handle it already issued.
struct ResolvedOwner {
  const GpuMemoryLedger* ledger{nullptr};
  const exec::Operator* op{nullptr};
  uint64_t generation{0};
  GpuMemoryOwnerHandle handle;
};

thread_local ResolvedOwner resolvedOwner;

GpuMemoryOwnerHandle resolveOwner(
    const std::shared_ptr<GpuMemoryLedger>& ledger) noexcept {
  auto* op = runtimeOperator();
  if (op == nullptr) {
    return {};
  }

  const auto generation = resolutionGeneration.load(std::memory_order_acquire);
  if (resolvedOwner.op == op && resolvedOwner.ledger == ledger.get() &&
      resolvedOwner.generation == generation) {
    return resolvedOwner.handle;
  }

  try {
    const auto handle = ledger->registerOperator(op);
    resolvedOwner = ResolvedOwner{ledger.get(), op, generation, handle};
    return handle;
  } catch (...) {
    // Fall through to the explicit unattributed owner.
    return {};
  }
}

class TrackingResourceImpl {
 public:
  TrackingResourceImpl(
      GpuMemoryResource upstream,
      std::shared_ptr<GpuMemoryLedger> ledger)
      : upstream_(std::move(upstream)), ledger_(std::move(ledger)) {}

  void*
  allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
      return nullptr;
    }

    const auto owner = resolveOwner(ledger_);
    auto* address = upstream_.allocate(stream, bytes, alignment);
    ledger_->recordAllocation(address, bytes, owner);
    return address;
  }

  void deallocate(
      cuda::stream_ref stream,
      void* address,
      std::size_t bytes,
      std::size_t alignment) noexcept {
    // Not also on bytes == 0: allocate() returns nullptr for a zero-byte
    // request, so this is the only guard needed, and skipping on size would
    // turn a caller that reports the wrong size into a silent leak.
    if (address == nullptr) {
      return;
    }

    // Before calling upstream, so an immediately reused address cannot be
    // erased by a delayed cross-thread deallocation.
    ledger_->recordDeallocation(address, bytes);
    upstream_.deallocate(stream, address, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment) {
    return allocate(rmm::cuda_stream_default, bytes, alignment);
  }

  void deallocate_sync(
      void* address,
      std::size_t bytes,
      std::size_t alignment) noexcept {
    deallocate(rmm::cuda_stream_default, address, bytes, alignment);
  }

  bool operator==(const TrackingResourceImpl& other) const noexcept {
    return this == &other;
  }

  bool operator!=(const TrackingResourceImpl& other) const noexcept {
    return !(*this == other);
  }

  friend void get_property(
      const TrackingResourceImpl&,
      cuda::mr::device_accessible) noexcept {}

 private:
  GpuMemoryResource upstream_;
  std::shared_ptr<GpuMemoryLedger> ledger_;
};

class TrackingResource final
    : public cuda::mr::shared_resource<TrackingResourceImpl> {
  using SharedBase = cuda::mr::shared_resource<TrackingResourceImpl>;

 public:
  TrackingResource(
      GpuMemoryResource upstream,
      std::shared_ptr<GpuMemoryLedger> ledger)
      : SharedBase(
            cuda::mr::make_shared_resource<TrackingResourceImpl>(
                std::move(upstream),
                std::move(ledger))) {}

  friend void get_property(
      const TrackingResource&,
      cuda::mr::device_accessible) noexcept {}
};

static_assert(
    cuda::mr::resource_with<TrackingResource, cuda::mr::device_accessible>);

void reportAllocationFailure(
    std::size_t bytes,
    const std::shared_ptr<GpuMemoryLedger>& ledger) noexcept {
  // Runs from RMM's failure callback while an out-of-memory condition unwinds.
  // Formatting and logging can throw there, and an exception leaving a noexcept
  // function terminates, so the whole body is guarded.
  try {
    const auto owner = resolveOwner(ledger);
    const auto state = ledger->currentState(owner);

    std::size_t freeBytes{0};
    std::size_t totalBytes{0};
    const auto cudaStatus = cudaMemGetInfo(&freeBytes, &totalBytes);
    const std::string_view cudaStatusName{cudaGetErrorName(cudaStatus)};

    if (state.ownerId == 0) {
      registerGpuMemoryNvtxUnattributedOwner();
    }
    markGpuMemoryAllocationFailure(
        state.ownerId, bytes, state, freeBytes, totalBytes, cudaStatusName);
    LOG(ERROR) << "GPU_MEMORY_OOM requested_bytes=" << bytes
               << " logical_live_bytes=" << state.globalCurrentBytes
               << " owner_id=" << state.ownerId
               << " plan_node_track_id=" << state.planNodeKey
               << " owner_live_bytes=" << state.ownerCurrentBytes
               << " plan_node_live_bytes=" << state.planNodeCurrentBytes
               << " cuda_free_bytes=" << freeBytes
               << " cuda_total_bytes=" << totalBytes
               << " cuda_status=" << cudaStatusName;
  } catch (...) {
    // The allocation failure is reported to the caller by RMM regardless.
  }
}

GpuMemoryResource makeTrackedResource(
    GpuMemoryResource upstream,
    const std::shared_ptr<GpuMemoryLedger>& ledger) {
  TrackingResource trackingResource(std::move(upstream), ledger);
  // rmm::bad_alloc rather than its subclass rmm::out_of_memory: a failure for
  // any other reason equally deserves a marker naming the operator that asked.
  rmm::mr::failure_callback_resource_adaptor<rmm::bad_alloc> failureResource(
      GpuMemoryResource{std::move(trackingResource)},
      [ledger](std::size_t bytes, void*) {
        reportAllocationFailure(bytes, ledger);
        return false;
      },
      nullptr);
  return GpuMemoryResource{std::move(failureResource)};
}

} // namespace

GpuMemoryResourcePair createGpuMemoryTrackingResources(
    GpuMemoryResource mainUpstream,
    GpuMemoryResource outputUpstream) {
  auto ledger = std::make_shared<GpuMemoryLedger>();
  auto main = makeTrackedResource(std::move(mainUpstream), ledger);
  auto output = makeTrackedResource(std::move(outputUpstream), ledger);
  installedLedger.store(std::move(ledger), std::memory_order_release);
  resolutionGeneration.fetch_add(1, std::memory_order_release);
  return {std::move(main), std::move(output)};
}

bool installGpuMemoryTracking(
    GpuMemoryResource& mainMr,
    GpuMemoryResource& outputMr) {
  if (!CudfConfig::getInstance().memoryTrackingEnabled) {
    resetGpuMemoryTracking();
    return false;
  }

  auto tracked =
      createGpuMemoryTrackingResources(std::move(mainMr), std::move(outputMr));
  mainMr = std::move(tracked.main);
  outputMr = std::move(tracked.output);
  return true;
}

void registerGpuMemoryOperator(exec::Operator* op) noexcept {
  const auto ledger = currentLedger();
  if (ledger == nullptr) {
    return;
  }

  try {
    ledger->registerOperator(op);
  } catch (...) {
    // Registration only affects diagnostics, never execution.
  }
}

void discardGpuMemoryOwnerCache() noexcept {
  resolutionGeneration.fetch_add(1, std::memory_order_release);
}

void resetGpuMemoryTracking() {
  installedLedger.store(nullptr, std::memory_order_release);
  resolutionGeneration.fetch_add(1, std::memory_order_release);
  resetGpuMemoryNvtxCounters();
}

GpuMemorySnapshot getGpuMemorySnapshot() {
  const auto ledger = currentLedger();
  return ledger == nullptr ? GpuMemorySnapshot{} : ledger->snapshot();
}

} // namespace facebook::velox::cudf_velox
