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
#pragma once

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/DebugUtil.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"

#include "velox/common/base/SpillConfig.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"

#include <glog/logging.h>

#include <type_traits>

namespace facebook::velox::cudf_velox {

/// Bitmask controlling which operator methods get NVTX profiling ranges.
/// Operators pass flags for the do* methods they actually override, so
/// no-op methods don't pollute nsys profiles with empty ranges.
enum NvtxMethodFlag : uint8_t {
  kNone = 0,
  kAddInput = 1 << 0,
  kGetOutput = 1 << 1,
  kNoMoreInput = 1 << 2,
  kClose = 1 << 3,
  kAll = kAddInput | kGetOutput | kNoMoreInput | kClose,
};

inline NvtxMethodFlag operator|(NvtxMethodFlag a, NvtxMethodFlag b) {
  using EnumT = std::underlying_type_t<NvtxMethodFlag>;
  return static_cast<NvtxMethodFlag>(
      static_cast<EnumT>(a) | static_cast<EnumT>(b));
}

inline NvtxMethodFlag operator&(NvtxMethodFlag a, NvtxMethodFlag b) {
  using EnumT = std::underlying_type_t<NvtxMethodFlag>;
  return static_cast<NvtxMethodFlag>(
      static_cast<EnumT>(a) & static_cast<EnumT>(b));
}

/// The user defined operator will inherit this operator, the operator accepts
/// CudfOperator and output CudfVector.
class CudfOperator : public NvtxHelper {
 public:
  CudfOperator(
      int32_t operatorId,
      const core::PlanNodeId& nodeId,
      std::optional<nvtx3::color> color = std::nullopt)
      : NvtxHelper(color, operatorId, fmt::format("[{}]", nodeId)) {}
};

/// Base class for all built-in cuDF operators in Velox.
///
/// All cuDF operators MUST extend this class rather than extending
/// exec::Operator and NvtxHelper directly. This class implements the template
/// method pattern:
/// the public operator interface methods (addInput, getOutput, noMoreInput,
/// close) are marked final and must NOT be overridden by derived classes.
/// Instead, derived classes should ONLY override the corresponding protected
/// do* virtual methods:
///   - doInitialize() -- performs subclass initialization
///   - doIsBlocked()  -- reports subclass blocking state
///   - doAddInput()    -- receives input rows; called by addInput()
///   - doGetOutput()   -- produces output rows; called by getOutput()
///   - doNoMoreInput() -- signals end of input; called by noMoreInput()
///                        (defaults to Operator::noMoreInput())
///   - doClose()       -- releases resources; called by close()
///                        (defaults to Operator::close())
///
/// This design scopes cuDF memory resources around every lifecycle call and
/// applies NVTX profiling ranges uniformly. Subclasses override doInitialize()
/// and doIsBlocked() instead of the final public methods. The
/// nvtxMethods bitmask (NvtxMethodFlag) lets operators suppress NVTX ranges
/// for do* methods they do not override, keeping nsys profiles clean.
///
/// Example:
///   class MyCudfOperator : public CudfOperatorBase {
///    public:
///     MyCudfOperator(int32_t operatorId, exec::DriverCtx* ctx,
///                    RowTypePtr outputType, const core::PlanNodeId& nodeId)
///         : CudfOperatorBase(
///               operatorId, ctx, outputType, nodeId, "MyCudfOperator",
///               std::nullopt,
///               NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput) {}
///
///     bool needsInput() const override { return !noMoreInput_; }
///
///    protected:
///     void doAddInput(RowVectorPtr input) override { /* process input */ }
///     RowVectorPtr doGetOutput() override { /* return output or nullptr */ }
///   };
class CudfOperatorBase : public exec::Operator, public NvtxHelper {
 public:
  CudfOperatorBase(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      RowTypePtr outputType,
      const core::PlanNodeId& planNodeId,
      const std::string& operatorName,
      std::optional<nvtx3::color> color = std::nullopt,
      NvtxMethodFlag nvtxMethods = NvtxMethodFlag::kAll,
      std::optional<common::SpillConfig> spillConfig = std::nullopt,
      std::optional<std::shared_ptr<const core::PlanNode>> planNode =
          std::nullopt);

  void initialize() final {
    auto memoryResources = scopedMemoryResources();
    Operator::initialize();
    maybeSetGpuMemoryReclaimer();
    doInitialize();
    checkCudaErrorInDebug();
  }

  exec::BlockingReason isBlocked(ContinueFuture* future) final {
    auto memoryResources = scopedMemoryResources();
    auto reason = doIsBlocked(future);
    checkCudaErrorInDebug();
    return reason;
  }

  void addInput(RowVectorPtr input) final {
    auto memoryResources = scopedMemoryResources();
    VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
        nvtxMethods_ & NvtxMethodFlag::kAddInput, className_);
    doAddInput(std::move(input));
    checkCudaErrorInDebug();
  }

  RowVectorPtr getOutput() final {
    auto memoryResources = scopedMemoryResources();
    VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
        nvtxMethods_ & NvtxMethodFlag::kGetOutput, className_);
    auto result = doGetOutput();
    checkCudaErrorInDebug();
    return result;
  }

  void noMoreInput() final {
    auto memoryResources = scopedMemoryResources();
    VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
        nvtxMethods_ & NvtxMethodFlag::kNoMoreInput, className_);
    doNoMoreInput();
    checkCudaErrorInDebug();
  }

  void close() final {
    auto memoryResources = scopedMemoryResources();
    VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
        nvtxMethods_ & NvtxMethodFlag::kClose, className_);
    doClose();
    checkCudaErrorInDebug();
  }
  void maybeSetGpuMemoryReclaimer();

 protected:
  [[nodiscard]] ScopedCudfMemoryResources scopedMemoryResources() const {
    return ScopedCudfMemoryResources{
        tempMemoryResource(), outputMemoryResource()};
  }

  virtual void doInitialize() {}

  virtual exec::BlockingReason doIsBlocked(ContinueFuture* /*future*/) {
    return exec::BlockingReason::kNotBlocked;
  }

  virtual void doAddInput(RowVectorPtr input) = 0;

  virtual RowVectorPtr doGetOutput() = 0;

  virtual void doNoMoreInput() {
    Operator::noMoreInput();
  }

  virtual void doClose() {
    Operator::close();
  }

  /// Returns whether this operator has device memory that can be reclaimed
  /// from its cuDF custom memory pool and, if so, reports an estimate.
  ///
  /// This is intentionally separate from Operator::reclaimableBytes(). The
  /// latter belongs to the operator's CPU pool, while this callback belongs to
  /// the mirrored GPU leaf pool under the cuDF custom memory hierarchy.
  virtual bool gpuReclaimableBytes(uint64_t& reclaimableBytes) const {
    reclaimableBytes = 0;
    return false;
  }

  /// Reclaims device memory owned by this operator. Implementations are
  /// invoked only after the owning task has been paused. The cuDF memory
  /// resources are scoped around the callback by the GPU pool reclaimer.
  virtual void reclaimGpu(
      uint64_t /*targetBytes*/,
      memory::MemoryReclaimer::Stats& /*stats*/) {}

 private:
  class GpuMemoryReclaimer;

  rmm::device_async_resource_ref tempMemoryResource() const {
    return tempMemoryResource_.has_value() ? *tempMemoryResource_
                                           : get_temp_mr();
  }

  rmm::device_async_resource_ref outputMemoryResource() const {
    return outputMemoryResource_.has_value() ? *outputMemoryResource_
                                             : get_output_mr();
  }

  const std::string className_;
  const NvtxMethodFlag nvtxMethods_;
  std::optional<rmm::device_async_resource_ref> tempMemoryResource_;
  std::optional<rmm::device_async_resource_ref> outputMemoryResource_;
};

} // namespace facebook::velox::cudf_velox
