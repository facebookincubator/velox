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
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/JoinBridge.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <queue>

namespace facebook::velox::cudf_velox {

class CudaEvent;
class CudfExpression;

/**
 * @brief Bridge for transferring build-side hash tables between build and probe
 * operators.
 *
 * This bridge manages the lifecycle of CUDF hash join objects and ensures
 * proper synchronization between build and probe phases. It stores the
 * constructed hash tables and hash join objects created from build-side data,
 * making them available to probe operators across different driver threads.
 *
 * The bridge handles batched hash tables when build data exceeds
 * cudf::size_type limits, and manages CUDA stream coordination between build
 * and probe operations.
 */
class CudfHashJoinBridge : public exec::JoinBridge {
 public:
  // The bridge transfers all build side batches and the hash join objects
  // constructed from them to the probe operator
  /** @brief Hash tables paired with their corresponding join objects for
   * batched processing */
  using hash_type = std::pair<
      std::vector<std::shared_ptr<cudf::table>>,
      std::vector<std::shared_ptr<cudf::hash_join>>>;

  void setHashTable(std::optional<hash_type> hashObject);

  std::optional<hash_type> hashOrFuture(ContinueFuture* future);

  // Store and retrieve the CUDA stream used for building the hash join.
  void setBuildStream(rmm::cuda_stream_view buildStream);

  std::optional<rmm::cuda_stream_view> getBuildStream();

  void setBuildReadyEvent(std::shared_ptr<CudaEvent> buildReadyEvent);

  std::shared_ptr<CudaEvent> getBuildReadyEvent();

 private:
  /** @brief Hash tables and join objects transferred from build to probe
   * operators */
  std::optional<hash_type> hashObject_;
  /** @brief CUDA stream used by build operator for proper synchronization */
  std::optional<rmm::cuda_stream_view> buildStream_;
  /** @brief Event recorded after build-side CUDA work is ready for probes */
  std::shared_ptr<CudaEvent> buildReadyEvent_;
};

/**
 * @brief Build operator that constructs CUDF hash tables from build-side input
 * data.
 *
 * This operator accumulates all build-side input batches and constructs hash
 * tables when all input is received. It handles batching when data exceeds
 * cudf::size_type limits and coordinates with other driver threads to ensure
 * only one driver performs the final hash table construction. The constructed
 * hash tables are transferred to probe operators via CudfHashJoinBridge.
 */
class CudfHashJoinBuild : public CudfOperatorBase {
 public:
  CudfHashJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode);

  bool needsInput() const override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;

 private:
  std::shared_ptr<const core::HashJoinNode> joinNode_;
  std::vector<CudfVectorPtr> inputs_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

/**
 * @brief Probe operator that performs CUDF hash join operations on probe-side
 * input.
 *
 * This operator receives hash tables from CudfHashJoinBuild via the bridge and
 * performs join operations on probe-side input data. It supports all standard
 * join types (inner, left, right, anti, semi) with optional filter conditions.
 * The operator handles stream synchronization between build and probe phases,
 * manages right join state across multiple drivers, and supports batched
 * processing for large datasets.
 */
class CudfHashJoinProbe : public CudfOperatorBase {
 public:
  using hash_type = CudfHashJoinBridge::hash_type;

  CudfHashJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode);

  void initialize() override;

  bool needsInput() const override;

  bool skipProbeOnEmptyBuild() const;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  /// Returns true if the join type is supported by cudf hash join.
  /// Supported types:
  /// - Inner, Left, Right, Full joins
  /// - Left/Right Semi Filter joins
  /// - Left Semi Project join (excluding null-aware join with filter)
  /// - Anti join (non-null-aware, or null-aware without filter)
  /// Note: Right Semi Project, and null-aware left semi-project join with
  /// filter not yet supported.
  static bool isSupportedJoinType(core::JoinType joinType) {
    return joinType == core::JoinType::kInner ||
        joinType == core::JoinType::kLeft ||
        joinType == core::JoinType::kAnti ||
        joinType == core::JoinType::kLeftSemiFilter ||
        joinType == core::JoinType::kLeftSemiProject ||
        joinType == core::JoinType::kRight ||
        joinType == core::JoinType::kRightSemiFilter ||
        joinType == core::JoinType::kFull;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  void waitForBuildReady(rmm::cuda_stream_view stream);

  std::shared_ptr<const core::HashJoinNode> joinNode_;
  /** @brief Hash tables and join objects received from build operator */
  std::optional<hash_type> hashObject_;

  // Filter related members
  /** @brief Whether to use AST-based filtering (false if filter spans both
   * sides or if filter deals with decimal types) */
  bool useAstFilter_{true};
  /** @brief CUDF AST tree for join filter evaluation */
  cudf::ast::tree tree_;
  /** @brief Scalar values used in filter expressions */
  std::vector<std::unique_ptr<cudf::scalar>> scalars_;
  /** @brief Precompute instructions for left (probe) table columns */
  std::vector<PrecomputeInstruction> leftPrecomputeInstructions_;
  /** @brief Precompute instructions for right (build) table columns */
  std::vector<PrecomputeInstruction> rightPrecomputeInstructions_;
  /** @brief Row type for probe table (needed for precomputation) */
  RowTypePtr probeType_;
  /** @brief Row type for build table (needed for precomputation) */
  RowTypePtr buildType_;
  /** @brief Cached evaluator for post-join filter column */
  std::shared_ptr<CudfExpression> filterEvaluator_;

  bool rightPrecomputed_{false};

  // Batched probe inputs needed for right join
  std::vector<CudfVectorPtr> inputs_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};

  /** @brief Column indices for join keys in left (probe) table */
  std::vector<cudf::size_type> leftKeyIndices_;
  /** @brief Column indices for join keys in right (build) table */
  std::vector<cudf::size_type> rightKeyIndices_;
  /** @brief Column indices to gather from left table for output */
  std::vector<cudf::size_type> leftColumnIndicesToGather_;
  /** @brief Column indices to gather from right table for output */
  std::vector<cudf::size_type> rightColumnIndicesToGather_;
  /** @brief Output column positions for left table columns */
  std::vector<size_t> leftColumnOutputIndices_;
  /** @brief Output column positions for right table columns */
  std::vector<size_t> rightColumnOutputIndices_;
  bool finished_{false};

  /// Queue of output batches when a single probe produces results exceeding
  /// cudf::size_type limits. Drained before accepting new input.
  std::queue<CudfVectorPtr> outputQueue_;

  /// True if any build table has NULL values in join key columns.
  /// Used for null-aware LEFT SEMI PROJECT to determine match column
  /// nullability.
  bool buildSideHasNullKeys_{false};

  // Copied from HashProbe.h
  // Indicates whether to skip probe input data processing or not. It only
  // applies for a specific set of join types (see skipProbeOnEmptyBuild()), and
  // the build table is empty and the probe input is read from non-spilled
  // source. This ensures the hash probe operator keeps running until all the
  // probe input from the sources have been processed. It prevents the exchange
  // hanging problem at the producer side caused by the early query finish.
  bool skipInput_{false};

  /** @brief CUDA stream from build operator for synchronization */
  std::optional<rmm::cuda_stream_view> buildStream_;
  /** @brief Event recorded after build-side CUDA work is ready for probes */
  std::shared_ptr<CudaEvent> buildReadyEvent_;
  /** @brief CUDA event for coordinating stream synchronization */
  std::unique_ptr<CudaEvent> cudaEvent_;

  // Streaming right join state
  // Per-build-table flags indicating whether a build row has had at least one
  // left match.
  /** @brief Flags tracking which build rows have been matched (for right joins)
   */
  std::vector<std::unique_ptr<cudf::column>> rightMatchedFlags_;

  /// Cached precomputed columns for right (build) tables
  std::vector<std::vector<ColumnOrView>> cachedRightPrecomputed_;
  /// Cached extended views for right tables (original + precomputed columns)
  std::vector<cudf::table_view> cachedExtendedRightViews_;

  // For Right joins, only one driver collects the unmatched rows mask and
  // emits. This value is set true only for that driver. See noMoreInput
  bool isLastDriver_{false};

  /// CUDA stream used during the last getOutput() probe operation. Set only
  /// for right/full joins, and only for drivers that process at least one probe
  /// batch. Used in noMoreInput() to synchronize GPU streams across drivers
  /// before combining rightMatchedFlags_. Drivers with no probe input are safe
  /// to skip: the driver loop guarantees all addInput batches are consumed by
  /// getOutput() before noMoreInput() fires, and unset flags remain in their
  /// host-synchronized all-false init state with no pending GPU work.
  std::optional<rmm::cuda_stream_view> lastProbeStream_;

  static constexpr auto oobPolicy = cudf::out_of_bounds_policy::NULLIFY;

  struct JoinOutput {
    std::unique_ptr<cudf::table> table;
    vector_size_t numRows;
  };

  /// Deferred join index pair for lazy gather in doGetOutput().
  /// gatherPendingBatch advances offset and pops the entry when fully consumed.
  struct PendingIndices {
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftIndices;
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> rightIndices;
    size_t offset{0};
    size_t buildChunkIndex;
    size_t remaining() const { return leftIndices->size() - offset; }
  };

  /// Queue of deferred index pairs awaiting gather in doGetOutput().
  std::queue<PendingIndices> pendingIndices_;

  /**
   * @brief Performs inner join between probe table and all build tables.
   * Populates pendingIndices_ (lazy gather) or outputQueue_ (non-AST filter).
   */
  void innerJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Performs left join between probe table and all build tables.
   * Populates pendingIndices_ (lazy gather) or outputQueue_ (non-AST filter).
   */
  void leftJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Performs right join between probe table and all build tables.
   * Populates pendingIndices_ for matched rows. Updates rightMatchedFlags_
   * eagerly. Uses outputQueue_ for non-AST filtered right joins.
   */
  void rightJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Performs full outer join between probe table and all build tables.
   * Populates pendingIndices_ for matched rows. Updates rightMatchedFlags_
   * eagerly. Full join requires AST filter support.
   */
  void fullJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Performs left semi filter join. Pushes results to outputQueue_.
   */
  void leftSemiFilterJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Performs left semi project join. Pushes results to outputQueue_.
   */
  void leftSemiProjectJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Performs right semi filter join. Pushes results to outputQueue_.
   */
  void rightSemiFilterJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Performs anti join. Pushes results to outputQueue_.
   */
  void antiJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Constructs join output table without applying filter conditions.
   * @param leftTableView Input probe table view
   * @param leftIndicesCol Column of indices into left table
   * @param rightTableView Input build table view
   * @param rightIndicesCol Column of indices into right table
   * @param stream CUDA stream for operations
   * @return Join result table with its logical row count
   */
  JoinOutput unfilteredOutput(
      cudf::table_view leftTableView,
      cudf::column_view leftIndicesCol,
      cudf::table_view rightTableView,
      cudf::column_view rightIndicesCol,
      rmm::cuda_stream_view stream);
  /**
   * @brief Constructs join output table with filter condition applied.
   * @param leftTableView Input probe table view
   * @param leftIndicesCol Column of indices into left table
   * @param rightTableView Input build table view
   * @param rightIndicesCol Column of indices into right table
   * @param func Filter function to apply to joined data
   * @param stream CUDA stream for operations
   * @return Filtered join result table with its logical row count
   */
  JoinOutput filteredOutput(
      cudf::table_view leftTableView,
      cudf::column_view leftIndicesCol,
      cudf::table_view rightTableView,
      cudf::column_view rightIndicesCol,
      std::function<std::vector<std::unique_ptr<cudf::column>>(
          std::vector<std::unique_ptr<cudf::column>>&&,
          cudf::column_view)> func,
      rmm::cuda_stream_view stream);

  JoinOutput filteredOutputIndices(
      cudf::table_view leftTableView,
      cudf::column_view leftIndicesCol,
      cudf::table_view rightTableView,
      cudf::column_view rightIndicesCol,
      cudf::table_view extendedLeftView,
      cudf::table_view extendedRightView,
      cudf::join_kind joinKind,
      rmm::cuda_stream_view stream);

  /// Enqueues a pair of join index vectors into pendingIndices_ for lazy
  /// gather. Sub-chunking is handled in gatherPendingBatch via offset.
  void enqueuePendingIndices(
      std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftIndices,
      std::unique_ptr<rmm::device_uvector<cudf::size_type>> rightIndices,
      size_t buildChunkIndex);

  /// Gather one batch from pendingIndices_ and return as CudfVector.
  RowVectorPtr gatherPendingBatch(rmm::cuda_stream_view stream);

  /// Updates rightMatchedFlags_[chunkIndex] by OR-ing in which build rows
  /// appear in rightIndicesCol. Synchronizes the stream before move-assigning
  /// the updated flags to avoid a race with cudaFreeAsync.
  void updateRightMatchedFlags(
      size_t chunkIndex,
      cudf::column_view rightIndicesCol,
      cudf::size_type numBuildRows,
      rmm::cuda_stream_view stream);

  /// Applies AST filter_join_indices to join index pairs in sub-maxBatchRows
  /// chunks and enqueues the filtered results into pendingIndices_.
  void filterAndEnqueueAstIndices(
      cudf::table_view leftTableView,
      size_t rightTableIndex,
      rmm::device_uvector<cudf::size_type>& leftJoinIndices,
      rmm::device_uvector<cudf::size_type>& rightJoinIndices,
      cudf::join_kind joinKind,
      rmm::cuda_stream_view stream);

  /// Eagerly gathers, applies non-AST filter, and pushes results to
  /// outputQueue_. Splits oversized index spans into sub-maxBatchRows chunks.
  void gatherFilterAndEnqueue(
      cudf::table_view leftTableView,
      cudf::table_view rightTableView,
      rmm::device_uvector<cudf::size_type>& leftJoinIndices,
      rmm::device_uvector<cudf::size_type>& rightJoinIndices,
      rmm::cuda_stream_view stream);
};

/**
 * @brief Factory for creating CUDF hash join operators and bridges from plan
 * nodes.
 *
 * This translator converts HashJoinNode plan nodes into the appropriate
 * CUDF-specific operators and bridges. It creates CudfHashJoinProbe operators
 * for probe-side processing, CudfHashJoinBuild operators for build-side
 * processing, and CudfHashJoinBridge instances for coordinating between them.
 */
class CudfHashJoinBridgeTranslator : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator>
  toOperator(exec::DriverCtx* ctx, int32_t id, const core::PlanNodePtr& node);

  std::unique_ptr<exec::JoinBridge> toJoinBridge(const core::PlanNodePtr& node);

  exec::OperatorSupplier toOperatorSupplier(const core::PlanNodePtr& node);
};

} // namespace facebook::velox::cudf_velox
