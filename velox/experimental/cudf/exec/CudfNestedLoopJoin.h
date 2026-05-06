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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/PrecomputeInstruction.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/JoinBridge.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace facebook::velox::cudf_velox {

class CudaEvent;

/// Coordinates data transfer from build to probe operators for nested loop
/// join. Build operators accumulate batches, then one operator transfers them
/// to the bridge. Probe operators block until build data is available, then
/// perform a cartesian product (cross join) with optional filtering.
///
/// Thread-safety: All methods use mutex locking for concurrent access.
class CudfNestedLoopJoinBridge : public exec::JoinBridge {
 public:
  // Build data: single concatenated build-side table. NLJ requires a single
  // table because the cross-join output is probe_rows × build_rows — batching
  // the build side does not prevent output overflow, so we enforce a single
  // table and fail early if the build side exceeds cudf::size_type limits.
  using build_data_type = std::shared_ptr<cudf::table>;

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
class CudfNestedLoopJoinBuild : public CudfOperatorBase {
 public:
  CudfNestedLoopJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::NestedLoopJoinNode> joinNode);

  bool needsInput() const override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  std::shared_ptr<const core::NestedLoopJoinNode> joinNode_;
  std::vector<CudfVectorPtr> inputs_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

/// Performs nested loop join using cuDF APIs.
///
/// Supports inner, left, right, full outer, and left semi project joins.
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
/// Mismatch handling (left/right joins):
///   We cannot use per-probe left/right join APIs (a probe row unmatched
///   against the build may still match in another probe batch). Instead, we
///   always use conditional_inner_join and track mismatches via GPU-side BOOL8
///   flag columns:
///   - Left join: probeMatchedFlags_ tracks probe mismatches for the current
///     probe batch. After the build is exhausted, unmatched probe rows are
///     emitted with null build columns.
///   - Right join: buildMatchedFlags_ tracks build mismatches across all probe
///     inputs. After all probes finish, the last driver (via allPeersFinished)
///     merges flags from all peers and emits unmatched build rows with null
///     probe columns.
///
/// Lifecycle:
/// 1. Constructor - Builds AST filter if join condition exists
/// 2. isBlocked() - Waits for build data from bridge; for right join, also
///    blocks on peer probes finishing (allPeersFinished barrier)
/// 3. addInput() - Receives probe batches one at a time
/// 4. getOutput() - Performs join, returns results (including mismatch rows
///    for left/right joins)
/// 5. noMoreInput() - For right join, coordinates multi-driver build mismatch
///    flag merging
///
/// Thread-safety: Each probe operator processes one batch at a time.
/// Multiple probe operators can run concurrently on different batches.
class CudfNestedLoopJoinProbe : public CudfOperatorBase {
 public:
  using build_data_type = CudfNestedLoopJoinBridge::build_data_type;

  CudfNestedLoopJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::NestedLoopJoinNode> joinNode);

  void initialize() override;

  bool needsInput() const override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  static bool isSupportedJoinType(core::JoinType joinType) {
    return joinType == core::JoinType::kInner ||
        joinType == core::JoinType::kLeft ||
        joinType == core::JoinType::kRight ||
        joinType == core::JoinType::kFull ||
        joinType == core::JoinType::kLeftSemiProject;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  /// Joins a single probe batch against the build table. Uses cross_join for
  /// unfiltered joins and conditional_inner_join for filtered joins. Updates
  /// probeMatchedFlags_ for left/full joins and buildMatchedFlags_ for
  /// right/full joins.
  std::unique_ptr<cudf::table> joinWithBuildBatch(
      cudf::table_view probeTableView,
      cudf::table_view buildView,
      rmm::cuda_stream_view stream);

  /// Emits probe rows that had no match across all build batches, with null
  /// build columns. Used for left/full joins after all build batches exhausted.
  std::unique_ptr<cudf::table> emitProbeMismatchRows(
      cudf::table_view probeTableView,
      rmm::cuda_stream_view stream);

  /// Emits build rows that had no match across all probe inputs, with null
  /// probe columns. Used for right/full joins after all probes finish. Only
  /// called by the last driver after merging flags from all peers.
  RowVectorPtr emitBuildMismatchRows(rmm::cuda_stream_view stream);

  /// Ensures build-stream data is visible on the given probe stream.
  void syncBuildStream(rmm::cuda_stream_view probeStream);

  bool isLeftOrFullJoin() const {
    return joinType_ == core::JoinType::kLeft ||
        joinType_ == core::JoinType::kFull;
  }

  bool isRightOrFullJoin() const {
    return joinType_ == core::JoinType::kRight ||
        joinType_ == core::JoinType::kFull;
  }

  std::shared_ptr<const core::NestedLoopJoinNode> joinNode_;
  core::JoinType joinType_;
  std::optional<build_data_type> buildData_;

  // Filter condition AST (for conditional_inner_join when filter exists).
  bool hasFilter_{false};
  cudf::ast::tree tree_;
  std::vector<std::unique_ptr<cudf::scalar>> scalars_;
  // Precompute instructions for expressions not directly representable in cuDF
  // AST. These sub-expressions are evaluated into extra columns appended to
  // each table view before it is passed to cuDF join APIs.
  std::vector<PrecomputeInstruction> leftPrecomputeInstructions_;
  std::vector<PrecomputeInstruction> rightPrecomputeInstructions_;

  // Output column mapping resolved by name from the output type.
  // Handles arbitrary column ordering (e.g., {"b0", "p0"}).
  std::vector<cudf::size_type> probeColumnIndicesToGather_;
  std::vector<cudf::size_type> buildColumnIndicesToGather_;
  std::vector<size_t> probeColumnOutputIndices_;
  std::vector<size_t> buildColumnOutputIndices_;

  // Probe and build types (cached for null column creation in left joins).
  RowTypePtr probeType_;
  RowTypePtr buildType_;

  bool finished_{false};

  // Probe mismatch tracking for left/full joins: BOOL8 column, one element
  // per probe row in the current input batch. Populated by
  // joinWithBuildBatch against the concatenated build table (single batch);
  // reset between probe inputs.
  std::unique_ptr<cudf::column> probeMatchedFlags_;

  // True when build side has no rows.
  bool buildEmpty_{false};

  // Cached precomputed columns for the build table (populated once in
  // isBlocked() when rightPrecomputeInstructions_ is non-empty). Kept alive
  // alongside buildExtendedView_ which holds non-owning views into them.
  std::vector<ColumnOrView> buildPrecomputed_;
  // Extended build table view: original build columns + precomputed columns.
  // Valid only when buildPrecomputed_ is non-empty.
  cudf::table_view buildExtendedView_{};

  // Build mismatch tracking for right/full joins: BOOL8 column, one element
  // per build row. Updated across all probe inputs via BITWISE_OR. Merged from
  // peers in noMoreInput() before the last driver emits unmatched build rows.
  std::unique_ptr<cudf::column> buildMatchedFlags_;

  // Multi-driver coordination for right/full join build mismatch emission.
  bool isLastDriver_{false};
  ContinueFuture peerFuture_{ContinueFuture::makeEmpty()};
  bool buildMismatchEmitted_{false};

  // Last CUDA stream used for probing, needed for join_streams in
  // noMoreInput() to ensure GPU-side ordering before flag merge.
  std::optional<rmm::cuda_stream_view> lastProbeStream_;

  // CUDA stream synchronization for build data visibility.
  std::optional<rmm::cuda_stream_view> buildStream_;
  std::unique_ptr<CudaEvent> cudaEvent_;
};

/// Creates CUDF nested loop join operators and bridges.
class CudfNestedLoopJoinBridgeTranslator
    : public exec::Operator::PlanNodeTranslator {
 public:
  /// Creates a CudfNestedLoopJoinProbe operator for the given plan node.
  std::unique_ptr<exec::Operator>
  toOperator(exec::DriverCtx* ctx, int32_t id, const core::PlanNodePtr& node);

  /// Creates a CudfNestedLoopJoinBridge for the given plan node.
  std::unique_ptr<exec::JoinBridge> toJoinBridge(const core::PlanNodePtr& node);

  /// Returns a supplier that creates CudfNestedLoopJoinBuild operators.
  exec::OperatorSupplier toOperatorSupplier(const core::PlanNodePtr& node);
};

} // namespace facebook::velox::cudf_velox
