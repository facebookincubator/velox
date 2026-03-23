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

#include "velox/common/base/SpillConfig.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include <glog/logging.h>

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
  return static_cast<NvtxMethodFlag>(
      static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

inline NvtxMethodFlag operator&(NvtxMethodFlag a, NvtxMethodFlag b) {
  return static_cast<NvtxMethodFlag>(
      static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

/// The user defined operator will inherit this operator, the operator accepts
/// CudfOperator and output CudfVector.
class CudfOperator : public NvtxHelper {
 public:
  CudfOperator(
      int32_t operatorId,
      const core::PlanNodeId& nodeId,
      std::optional<nvtx3::color> color = std::nullopt)
      : NvtxHelper(
            color.value_or(nvtx3::rgb{160, 82, 45}),
            operatorId,
            fmt::format("[{}]", nodeId)) {}
};

/// Built-in operators extend this class and override do* methods.
class CudfOperatorBase : public exec::Operator, public CudfOperator {
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
      std::optional<std::shared_ptr<const core::PlanNode>> planNode = std::nullopt)
      : Operator(
            driverCtx,
            outputType,
            operatorId,
            planNodeId,
            operatorName,
            spillConfig),
        CudfOperator(operatorId, planNodeId, color),
        className_(operatorName),
        nvtxMethods_(nvtxMethods) {}

  void addInput(RowVectorPtr input) override {
    if (nvtxMethods_ & NvtxMethodFlag::kAddInput) {
      if (CudfConfig::getInstance().debugEnabled) {
        VLOG(2) << "Calling " << className_ << "::addInput";
      }
      VELOX_NVTX_OPERATOR_FUNC_RANGE();
      doAddInput(std::move(input));
    } else {
      doAddInput(std::move(input));
    }
  }

  RowVectorPtr getOutput() override {
    if (nvtxMethods_ & NvtxMethodFlag::kGetOutput) {
      if (CudfConfig::getInstance().debugEnabled) {
        VLOG(2) << "Calling " << className_ << "::getOutput";
      }
      VELOX_NVTX_OPERATOR_FUNC_RANGE();
      return doGetOutput();
    } else {
      return doGetOutput();
    }
  }

  void noMoreInput() override {
    if (nvtxMethods_ & NvtxMethodFlag::kNoMoreInput) {
      if (CudfConfig::getInstance().debugEnabled) {
        VLOG(2) << "Calling " << className_ << "::noMoreInput";
      }
      VELOX_NVTX_OPERATOR_FUNC_RANGE();
      doNoMoreInput();
    } else {
      doNoMoreInput();
    }
  }

  void close() override {
    if (nvtxMethods_ & NvtxMethodFlag::kClose) {
      VELOX_NVTX_OPERATOR_FUNC_RANGE();
      if (CudfConfig::getInstance().debugEnabled) {
        VLOG(2) << "Calling " << className_ << "::close";
      }
      doClose();
    } else {
      doClose();
    }
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
