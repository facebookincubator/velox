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

#include "velox/exec/Driver.h"
#include "velox/exec/Task.h"

namespace facebook::velox::cudf_velox {

CudfOperatorMemoryResources::CudfOperatorMemoryResources(
    exec::DriverCtx* driverCtx,
    memory::MemoryPool* gpuPool) {
  if (gpuPool == nullptr) {
    return;
  }

  auto resourceOwner = cudfCustomMemoryResource(*driverCtx->task->queryCtx());
  VELOX_CHECK(
      mr_.has_value() && output_mr_.has_value(),
      "cuDF memory resources are not initialized; call registerCudf() first");
  auto pool = gpuPool->shared_from_this();
  temp_.emplace(*mr_, pool, resourceOwner);
  if (*mr_ != *output_mr_) {
    output_.emplace(*output_mr_, std::move(pool), std::move(resourceOwner));
  }
}

ScopedCudfMemoryResources CudfOperatorMemoryResources::scoped() {
  if (temp_.has_value()) {
    rmm::device_async_resource_ref temp{*temp_};
    return ScopedCudfMemoryResources{
        temp,
        output_.has_value() ? rmm::device_async_resource_ref{*output_} : temp};
  }
  // Deliberately not get_temp_mr(): that resolves to the thread-local
  // temporary dispatcher, which delegates back here, so installing it as the
  // scope's resource would nest the dispatcher inside itself.
  VELOX_CHECK(
      mr_.has_value(),
      "cuDF memory resources are not initialized; call registerCudf() first");
  return ScopedCudfMemoryResources{
      rmm::device_async_resource_ref{*mr_}, get_output_mr()};
}

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
      nvtxMethods_(nvtxMethods),
      memoryResources_(driverCtx, customPool(kCudfMemoryResourceTag)) {}

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
      nvtxMethods_(nvtxMethods),
      memoryResources_(driverCtx, customPool(kCudfMemoryResourceTag)) {}

} // namespace facebook::velox::cudf_velox
