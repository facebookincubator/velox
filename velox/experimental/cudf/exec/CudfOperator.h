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
///   - doAddInput()    -- receives input rows; called by addInput()
///   - doGetOutput()   -- produces output rows; called by getOutput()
///   - doNoMoreInput() -- signals end of input; called by noMoreInput()
///                        (defaults to Operator::noMoreInput())
///   - doClose()       -- releases resources; called by close()
///                        (defaults to Operator::close())
///
/// This design ensures that NVTX profiling ranges are applied uniformly
/// across all operators without each subclass having to manage them. The
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
          std::nullopt)
      : Operator(
            driverCtx,
            outputType,
            operatorId,
            planNodeId,
            operatorName,
            spillConfig),
        NvtxHelper(color, operatorId, fmt::format("[{}]", planNodeId)),
        className_(operatorName),
        nvtxMethods_(nvtxMethods) {}

  void addInput(RowVectorPtr input) final {
    VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
        nvtxMethods_ & NvtxMethodFlag::kAddInput, className_);
    doAddInput(std::move(input));
  }

  RowVectorPtr getOutput() final {
    VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
        nvtxMethods_ & NvtxMethodFlag::kGetOutput, className_);
    return doGetOutput();
  }

  void noMoreInput() final {
    VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
        nvtxMethods_ & NvtxMethodFlag::kNoMoreInput, className_);
    doNoMoreInput();
  }

  void close() final {
    VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
        nvtxMethods_ & NvtxMethodFlag::kClose, className_);
    doClose();
  }

 protected:
  virtual void doAddInput(RowVectorPtr input) = 0;

  virtual RowVectorPtr doGetOutput() = 0;

  virtual void doNoMoreInput() {
    Operator::noMoreInput();
  }

  virtual void doClose() {
    Operator::close();
  }

 private:
  const std::string className_;
  const NvtxMethodFlag nvtxMethods_;
};

} // namespace facebook::velox::cudf_velox
