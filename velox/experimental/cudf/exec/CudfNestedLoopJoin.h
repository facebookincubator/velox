/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use it except in compliance with the License.
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

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/JoinBridge.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace facebook::velox::cudf_velox {

/// Custom join bridge for GPU NestedLoopJoin that transfers cudf::table objects
/// directly between build and probe operators, avoiding intermediate CudfVector
/// wrapping.
class CudfNestedLoopJoinBridge : public exec::JoinBridge {
 public:
  struct BuildData {
    std::vector<std::shared_ptr<cudf::table>> tables;
    rmm::cuda_stream_view stream;
  };

  void setData(BuildData data);

  std::optional<BuildData> dataOrFuture(ContinueFuture* future);

 private:
  std::optional<BuildData> data_;
};

/// Translator that creates CudfNestedLoopJoinBridge instances for
/// NestedLoopJoinNode plan nodes.
class CudfNestedLoopJoinBridgeTranslator
    : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::JoinBridge> toJoinBridge(
      const core::PlanNodePtr& node) override;
};

/**
 * GPU build operator for NestedLoopJoin (cross join).
 * Accumulates build-side CudfVector inputs, concatenates them on GPU, and
 * passes the result to the probe via CudfNestedLoopJoinBridge.
 */
class CudfNestedLoopJoinBuild : public exec::Operator, public NvtxHelper {
 public:
  CudfNestedLoopJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::NestedLoopJoinNode> joinNode);

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override { return nullptr; }

  bool needsInput() const override { return !noMoreInput_; }

  void noMoreInput() override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 private:
  std::shared_ptr<const core::NestedLoopJoinNode> joinNode_;
  std::vector<CudfVectorPtr> inputs_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

/**
 * GPU probe operator for NestedLoopJoin (cross join).
 * Reads build data from CudfNestedLoopJoinBridge as cudf::table objects,
 * then for each probe batch computes the cross product on GPU using
 * gather operations. Each build table batch is processed independently
 * to avoid exceeding cudf::size_type limits.
 */
class CudfNestedLoopJoinProbe : public exec::Operator, public NvtxHelper {
 public:
  CudfNestedLoopJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::NestedLoopJoinNode> joinNode);

  bool needsInput() const override;

  void addInput(RowVectorPtr input) override;

  void noMoreInput() override;

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  /// Inner join (with or without filter condition) is supported on GPU.
  static bool isSupported(const core::NestedLoopJoinNode* node) {
    return node->joinType() == core::JoinType::kInner;
  }

 private:
  bool getBuildData(ContinueFuture* future);

  RowVectorPtr crossJoinWithBuildBatch(
      const cudf::table_view& probeView,
      const cudf::table_view& buildBatchView,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  std::shared_ptr<const core::NestedLoopJoinNode> joinNode_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};

  /// Build tables received from the bridge (shared across probe drivers)
  std::optional<CudfNestedLoopJoinBridge::BuildData> buildData_;

  /// Filter evaluator for join condition (lazy init)
  std::shared_ptr<CudfExpression> filterEvaluator_;
  RowTypePtr filterConcatType_;

  /// Output column order: which columns come from probe vs build (by output
  /// index).
  std::vector<cudf::size_type> leftColumnIndicesToGather_;
  std::vector<cudf::size_type> rightColumnIndicesToGather_;
  std::vector<size_t> leftColumnOutputIndices_;
  std::vector<size_t> rightColumnOutputIndices_;

  /// Index into build tables for the current probe input
  size_t buildBatchIdx_{0};

  bool finished_{false};
};

} // namespace facebook::velox::cudf_velox
