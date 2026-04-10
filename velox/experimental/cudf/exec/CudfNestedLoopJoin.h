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

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace facebook::velox::cudf_velox {

class CudaEvent;
class CudfExpression;

/// Coordinates data transfer from build to probe operators for nested loop
/// join. Build operators accumulate batches, then one operator transfers them
/// to the bridge. Probe operators block until build data is available, then
/// perform a cartesian product (cross join) with optional filtering.
///
/// Thread-safety: All methods use mutex locking for concurrent access.
class CudfNestedLoopJoinBridge : public exec::JoinBridge {
 public:
  // Build data: concatenated build-side tables.
  using build_data_type = std::vector<std::shared_ptr<cudf::table>>;

  void setData(std::optional<build_data_type> data);

  std::optional<build_data_type> dataOrFuture(ContinueFuture* future);

  void setBuildStream(rmm::cuda_stream_view buildStream);

  std::optional<rmm::cuda_stream_view> getBuildStream();

 private:
  std::optional<build_data_type> data_;
  std::optional<rmm::cuda_stream_view> buildStream_;
};

/// Accumulates build-side input for nested loop join.
///
/// Lifecycle:
/// 1. addInput() - Collects all input batches in memory (as CudfVectors)
/// 2. noMoreInput() - Called when upstream finishes
///    - One build operator (chosen via allPeersFinished) transfers all
///      accumulated data to the bridge
///    - Other operators just mark themselves as finished
/// 3. getOutput() - Always returns nullptr (build operators don't produce
/// output)
/// 4. isFinished() - True after noMoreInput() and bridge transfer completes
///
/// Memory: All build-side data is kept in GPU memory until the join completes.
/// For very large build sides, this could be memory-intensive.
class CudfNestedLoopJoinBuild : public exec::Operator, public NvtxHelper {
 public:
  CudfNestedLoopJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::NestedLoopJoinNode> joinNode);

  void addInput(RowVectorPtr input) override;

  bool needsInput() const override;

  RowVectorPtr getOutput() override;

  void noMoreInput() override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  void close() override {
    inputs_.clear();
    Operator::close();
  }

 private:
  std::shared_ptr<const core::NestedLoopJoinNode> joinNode_;
  std::vector<CudfVectorPtr> inputs_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

/// Performs nested loop join using cuDF APIs.
///
/// Algorithm (two-path approach for optimal performance):
///
/// Path 1 - No filter (cross join):
///   Uses cudf::cross_join(probe, build) for full cartesian product.
///
/// Path 2 - With filter (conditional join):
///   Uses cudf::conditional_inner_join(probe, build, ast) to evaluate the
///   filter on GPU, returning only matching row index pairs, then gathers
///   actual data using indices.
///
/// Lifecycle:
/// 1. Constructor - Builds AST filter if join condition exists
/// 2. isBlocked() - Waits for build data from bridge
/// 3. addInput() - Receives probe batches one at a time
/// 4. getOutput() - Performs cross/conditional join, returns results
///
/// Current limitations:
/// - Only inner join supported
/// - No memory batching for very large cross products (>2^31 rows)
///
/// Thread-safety: Each probe operator processes one batch at a time.
/// Multiple probe operators can run concurrently on different batches.
class CudfNestedLoopJoinProbe : public exec::Operator, public NvtxHelper {
 public:
  using build_data_type = CudfNestedLoopJoinBridge::build_data_type;

  CudfNestedLoopJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::NestedLoopJoinNode> joinNode);

  bool needsInput() const override;

  void addInput(RowVectorPtr input) override;

  void noMoreInput() override;

  RowVectorPtr getOutput() override;

  void close() override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  static bool isSupportedJoinType(core::JoinType joinType) {
    return joinType == core::JoinType::kInner;
  }

 private:
  // Performs cross join between a single probe batch and a single build batch.
  // Uses cross_join for unfiltered joins (optimized cartesian product).
  // Uses conditional_inner_join for filtered joins (only materializes
  // matches).
  std::unique_ptr<cudf::table> crossJoinWithBuildBatch(
      cudf::table_view probeTableView,
      cudf::table_view buildView,
      rmm::cuda_stream_view stream);

  std::shared_ptr<const core::NestedLoopJoinNode> joinNode_;
  std::optional<build_data_type> buildData_;

  // Filter condition AST (for conditional_inner_join when filter exists).
  bool hasFilter_{false};
  cudf::ast::tree tree_;
  std::vector<std::unique_ptr<cudf::scalar>> scalars_;

  // Output column mapping resolved by name from the output type.
  // Handles arbitrary column ordering (e.g., {"b0", "p0"}).
  std::vector<cudf::size_type> probeColumnIndicesToGather_;
  std::vector<cudf::size_type> buildColumnIndicesToGather_;
  std::vector<size_t> probeColumnOutputIndices_;
  std::vector<size_t> buildColumnOutputIndices_;

  // Index into build tables for the current probe input
  size_t buildBatchIdx_{0};

  bool finished_{false};

  // CUDA stream synchronization
  std::optional<rmm::cuda_stream_view> buildStream_;
  std::unique_ptr<CudaEvent> cudaEvent_;
};

/// Creates CUDF nested loop join operators and bridges.
class CudfNestedLoopJoinBridgeTranslator
    : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator>
  toOperator(exec::DriverCtx* ctx, int32_t id, const core::PlanNodePtr& node);

  std::unique_ptr<exec::JoinBridge> toJoinBridge(const core::PlanNodePtr& node);

  exec::OperatorSupplier toOperatorSupplier(const core::PlanNodePtr& node);
};

} // namespace facebook::velox::cudf_velox
