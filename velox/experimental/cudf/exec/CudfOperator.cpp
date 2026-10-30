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

#include "velox/experimental/cudf/exec/CudfOperator.h"

#include "velox/exec/Driver.h"
#include "velox/exec/Task.h"

namespace facebook::velox::cudf_velox {

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
  VELOX_CHECK(
      mr_.has_value() && output_mr_.has_value(),
      "cuDF memory resources must be initialized before creating operators");

  memoryResourceOwner_ = cudfMemoryResourceRegistry(*queryCtx);
  auto resources = memoryResourceOwner_->resourcesFor(
      *mr_, *output_mr_, gpuPool->shared_from_this());
  tempMemoryResource_ = resources.temp;
  outputMemoryResource_ = resources.output;
}

void CudfOperatorBase::recordGpuMemoryStats() {
  auto* gpuPool = customPool(kCudfMemoryResourceTag);
  if (gpuPool == nullptr) {
    return;
  }
  addRuntimeStat(
      "cudfOperatorGpuPeakBytes",
      RuntimeCounter(
          static_cast<int64_t>(gpuPool->peakBytes()),
          RuntimeCounter::Unit::kBytes));
  auto* queryPool = gpuPool;
  while (queryPool->parent() != nullptr) {
    queryPool = queryPool->parent();
  }
  addRuntimeStat(
      "cudfQueryGpuPeakBytes",
      RuntimeCounter(
          static_cast<int64_t>(queryPool->peakBytes()),
          RuntimeCounter::Unit::kBytes));
  // This is the configured root limit, not its current granted reservation.
  // Use max rather than sum when aggregating copies across operators.
  addRuntimeStat(
      "cudfQueryGpuMaxCapacityBytes",
      RuntimeCounter(queryPool->maxCapacity(), RuntimeCounter::Unit::kBytes));
}

CudfSourceOperatorBase::CudfSourceOperatorBase(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    RowTypePtr outputType,
    const core::PlanNodeId& planNodeId,
    const std::string& operatorName,
    std::optional<nvtx3::color> color,
    NvtxMethodFlag nvtxMethods)
    : SourceOperator(
          driverCtx,
          std::move(outputType),
          operatorId,
          planNodeId,
          operatorName),
      NvtxHelper(color, operatorId, fmt::format("[{}]", planNodeId)),
      className_(operatorName),
      nvtxMethods_(nvtxMethods) {
  auto* gpuPool = customPool(kCudfMemoryResourceTag);
  if (gpuPool == nullptr) {
    return;
  }

  auto queryCtx = driverCtx->task->queryCtx();
  VELOX_CHECK(
      mr_.has_value() && output_mr_.has_value(),
      "cuDF memory resources must be initialized before creating operators");

  memoryResourceOwner_ = cudfMemoryResourceRegistry(*queryCtx);
  auto resources = memoryResourceOwner_->resourcesFor(
      *mr_, *output_mr_, gpuPool->shared_from_this());
  tempMemoryResource_ = resources.temp;
  outputMemoryResource_ = resources.output;
}

void CudfSourceOperatorBase::recordGpuMemoryStats() {
  auto* gpuPool = customPool(kCudfMemoryResourceTag);
  if (gpuPool == nullptr) {
    return;
  }
  addRuntimeStat(
      "cudfOperatorGpuPeakBytes",
      RuntimeCounter(
          static_cast<int64_t>(gpuPool->peakBytes()),
          RuntimeCounter::Unit::kBytes));
  auto* queryPool = gpuPool;
  while (queryPool->parent() != nullptr) {
    queryPool = queryPool->parent();
  }
  addRuntimeStat(
      "cudfQueryGpuPeakBytes",
      RuntimeCounter(
          static_cast<int64_t>(queryPool->peakBytes()),
          RuntimeCounter::Unit::kBytes));
  // This is the configured root limit, not its current granted reservation.
  // Use max rather than sum when aggregating copies across operators.
  addRuntimeStat(
      "cudfQueryGpuMaxCapacityBytes",
      RuntimeCounter(queryPool->maxCapacity(), RuntimeCounter::Unit::kBytes));
}

} // namespace facebook::velox::cudf_velox
