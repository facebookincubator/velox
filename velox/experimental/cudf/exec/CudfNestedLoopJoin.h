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

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/JoinBridge.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace facebook::velox::cudf_velox {

/**
 * GPU build operator for NestedLoopJoin (cross join).
 * Accumulates build-side CudfVector inputs, concatenates them on GPU, and
 * passes the result to the probe via the existing NestedLoopJoinBridge.
 */
class CudfNestedLoopJoinBuild : public exec::Operator, public NvtxHelper {
 public:
  CudfNestedLoopJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::NestedLoopJoinNode> joinNode);

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override {
    return nullptr;
  }

  bool needsInput() const override {
    return !noMoreInput_;
  }

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
 * Reads build data from NestedLoopJoinBridge (CudfVectors from GPU build),
 * then for each probe batch computes the cross product on GPU using
 * repeat(probe, nBuild) and gather(build, indices).
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

  /// Only cross join (no filter) with inner join type is supported on GPU.
  static bool isSupported(const core::NestedLoopJoinNode* node) {
    return node->joinCondition() == nullptr &&
        node->joinType() == core::JoinType::kInner;
  }

 private:
  bool getBuildData(ContinueFuture* future);

  std::shared_ptr<const core::NestedLoopJoinNode> joinNode_;
  std::optional<std::vector<RowVectorPtr>> buildVectors_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};

  /// Cached concatenated build table (computed once in getBuildData)
  std::unique_ptr<cudf::table> concatenatedBuildTable_;
  cudf::table_view buildView_;

  /// Output column order: which columns come from probe vs build (by output
  /// index).
  std::vector<cudf::size_type> leftColumnIndicesToGather_;
  std::vector<cudf::size_type> rightColumnIndicesToGather_;
  std::vector<size_t> leftColumnOutputIndices_;
  std::vector<size_t> rightColumnOutputIndices_;

  bool finished_{false};
};

} // namespace facebook::velox::cudf_velox
