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

#include "velox/experimental/cudf/exec/CudfMemoryResource.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/exec/Driver.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/Task.h"

#include <mutex>

namespace facebook::velox::cudf_velox {
namespace {

std::shared_ptr<memory::CustomMemoryResource> cudfMemoryResourceOwner(
    const core::QueryCtx& queryCtx) {
  auto perQuery =
      queryCtx.registry<memory::CustomMemoryResourceRegistry::Registry>(
          memory::kCustomMemoryResourceRegistryKey);
  auto& registry =
      perQuery ? *perQuery : memory::CustomMemoryResourceRegistry::global();
  return registry.find(std::string(kCudfMemoryResourceTag));
}

std::shared_ptr<CudfMemoryResourceRegistry> cudfMemoryResourceRegistry(
    core::QueryCtx& queryCtx) {
  if (auto registry = queryCtx.registry<CudfMemoryResourceRegistry>(
          kCudfMemoryResourceRegistryKey)) {
    return registry;
  }

  // QueryCtx does not currently offer an atomic get-or-create operation for
  // registries. Serialize this cuDF-local lazy initialization and recheck.
  static std::mutex initMutex;
  std::lock_guard<std::mutex> lock(initMutex);
  if (auto registry = queryCtx.registry<CudfMemoryResourceRegistry>(
          kCudfMemoryResourceRegistryKey)) {
    return registry;
  }
  auto registry = std::make_shared<CudfMemoryResourceRegistry>();
  queryCtx.setRegistry(kCudfMemoryResourceRegistryKey, registry);
  return registry;
}

} // namespace

class CudfOperatorBase::GpuMemoryReclaimer final
    : public exec::MemoryReclaimer {
 public:
  static std::unique_ptr<memory::MemoryReclaimer> create(
      exec::DriverCtx* driverCtx,
      CudfOperatorBase* op) {
    VELOX_CHECK_NOT_NULL(driverCtx);
    VELOX_CHECK_NOT_NULL(driverCtx->driver);
    return std::unique_ptr<memory::MemoryReclaimer>(
        new GpuMemoryReclaimer(driverCtx->driver->shared_from_this(), op));
  }

  bool reclaimableBytes(
      const memory::MemoryPool& pool,
      uint64_t& reclaimableBytes) const override {
    reclaimableBytes = 0;
    auto driver = ensureDriver();
    if (FOLLY_UNLIKELY(driver == nullptr)) {
      return false;
    }

    VELOX_CHECK(
        std::addressof(pool) == op_->customPool(kCudfMemoryResourceTag),
        "GPU reclaimer invoked for a pool other than its operator's cuDF pool");
    return op_->gpuReclaimableBytes(reclaimableBytes);
  }

  uint64_t reclaim(
      memory::MemoryPool* pool,
      uint64_t targetBytes,
      uint64_t /*maxWaitMs*/,
      memory::MemoryReclaimer::Stats& stats) override {
    auto driver = ensureDriver();
    if (FOLLY_UNLIKELY(driver == nullptr)) {
      return 0;
    }

    VELOX_CHECK_NOT_NULL(pool);
    VELOX_CHECK(
        pool == op_->customPool(kCudfMemoryResourceTag),
        "GPU reclaimer invoked for a pool other than its operator's cuDF pool");

    uint64_t reclaimableBytes{0};
    if (!op_->gpuReclaimableBytes(reclaimableBytes) || reclaimableBytes == 0) {
      return 0;
    }

    VELOX_CHECK(
        !driver->state().isOnThread() || driver->state().suspended() ||
            driver->state().isTerminated,
        "driverOnThread {}, driverSuspended {} driverTerminated {} {}",
        driver->state().isOnThread(),
        driver->state().suspended(),
        driver->state().isTerminated,
        pool->name());
    VELOX_CHECK(
        driver->task()->pauseRequested(),
        "GPU memory reclaim requires the owning task to be paused");

    if (op_->nonReclaimableSection_ &&
        !op_->canReclaimGpuWhileNonReclaimable()) {
      ++stats.numNonReclaimableAttempts;
      return 0;
    }

    RuntimeStatWriterScopeGuard opStatsGuard(op_);

    return memory::MemoryReclaimer::run(
        [&]() {
          int64_t reclaimedBytes{0};
          {
            memory::ScopedReclaimedBytesRecorder recorder(
                pool, &reclaimedBytes);
            auto memoryResources = op_->scopedMemoryResources();
            op_->reclaimGpu(targetBytes, stats);
            checkCudaErrorInDebug();
          }
          VELOX_CHECK_GE(
              reclaimedBytes,
              0,
              "Unexpected GPU memory growth after reclaim from pool {}",
              pool->name());
          return reclaimedBytes;
        },
        stats);
  }

 private:
  GpuMemoryReclaimer(
      const std::shared_ptr<exec::Driver>& driver,
      CudfOperatorBase* op)
      : exec::MemoryReclaimer(0), driver_(driver), op_(op) {
    VELOX_CHECK_NOT_NULL(op_);
  }

  std::shared_ptr<exec::Driver> ensureDriver() const {
    return driver_.lock();
  }

  const std::weak_ptr<exec::Driver> driver_;
  CudfOperatorBase* const op_;
};

CudfOperatorBase::CudfOperatorBase(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    RowTypePtr outputType,
    const core::PlanNodeId& planNodeId,
    const std::string& operatorName,
    std::optional<nvtx3::color> color,
    NvtxMethodFlag nvtxMethods,
    std::optional<common::SpillConfig> spillConfig,
    std::optional<std::shared_ptr<const core::PlanNode>> /*planNode*/)
    : Operator(
          driverCtx,
          std::move(outputType),
          operatorId,
          planNodeId,
          operatorName,
          std::move(spillConfig)),
      NvtxHelper(color, operatorId, fmt::format("[{}]", planNodeId)),
      className_(operatorName),
      nvtxMethods_(nvtxMethods) {
  auto* gpuPool = customPool(kCudfMemoryResourceTag);
  if (gpuPool == nullptr) {
    return;
  }

  auto queryCtx = driverCtx->task->queryCtx();
  auto resourceOwner = cudfMemoryResourceOwner(*queryCtx);
  VELOX_CHECK_NOT_NULL(
      resourceOwner,
      "No CustomMemoryResource registered for tag: {}",
      kCudfMemoryResourceTag);
  VELOX_CHECK(
      mr_.has_value() && output_mr_.has_value(),
      "cuDF memory resources must be initialized before creating operators");

  auto resources = cudfMemoryResourceRegistry(*queryCtx)->resourcesFor(
      *mr_, *output_mr_, gpuPool->shared_from_this(), std::move(resourceOwner));
  tempMemoryResource_ = resources.temp;
  outputMemoryResource_ = resources.output;
}

void CudfOperatorBase::maybeSetGpuMemoryReclaimer() {
  auto* gpuPool = customPool(kCudfMemoryResourceTag);
  if (gpuPool == nullptr || gpuPool->reclaimer() != nullptr) {
    return;
  }
  auto* parent = gpuPool->parent();
  if (parent == nullptr || parent->reclaimer() == nullptr) {
    return;
  }
  gpuPool->setReclaimer(
      GpuMemoryReclaimer::create(operatorCtx()->driverCtx(), this));
}

} // namespace facebook::velox::cudf_velox
