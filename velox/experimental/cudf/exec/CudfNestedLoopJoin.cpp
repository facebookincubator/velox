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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfNestedLoopJoin.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/expression/PrecomputeInstruction.h"

#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"
#include "velox/type/TypeUtil.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

namespace facebook::velox::cudf_velox {

namespace {

// Appends precomputed columns to a table view for filter AST evaluation.
// TODO: Consolidate with the identical helper in CudfHashJoin.cpp.
cudf::table_view createExtendedTableView(
    cudf::table_view originalView,
    std::vector<ColumnOrView>& precomputedColumns) {
  if (precomputedColumns.empty()) {
    return originalView;
  }
  std::vector<cudf::column_view> allViews;
  allViews.reserve(originalView.num_columns() + precomputedColumns.size());
  for (cudf::size_type i = 0; i < originalView.num_columns(); ++i) {
    allViews.push_back(originalView.column(i));
  }
  for (auto& col : precomputedColumns) {
    allViews.push_back(asView(col));
  }
  return cudf::table_view(allViews);
}

} // namespace

void CudfNestedLoopJoinBridge::setData(
    std::optional<CudfNestedLoopJoinBridge::build_data_type> data) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(!data_.has_value(), "Bridge already has data");
    data_ = std::move(data);
    promises = std::move(promises_); // Extract promises to fulfill outside lock
  }
  notify(std::move(promises)); // Wake up all blocked probe operators
}

// Returns build data if available, otherwise returns a future to wait on.
// Called by probe operators in isBlocked().
std::optional<CudfNestedLoopJoinBridge::build_data_type>
CudfNestedLoopJoinBridge::dataOrFuture(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(!cancelled_, "Getting data after the build side is aborted");
  if (data_.has_value()) {
    return data_;
  }
  // Data not ready yet, create a promise that will be fulfilled by setData()
  promises_.emplace_back("CudfNestedLoopJoinBridge::dataOrFuture");
  *future = promises_.back().getSemiFuture();
  return std::nullopt; // Probe will block on the future
}

void CudfNestedLoopJoinBridge::setBuildStream(
    rmm::cuda_stream_view buildStream) {
  std::lock_guard<std::mutex> l(mutex_);
  buildStream_ = buildStream;
}

std::optional<rmm::cuda_stream_view>
CudfNestedLoopJoinBridge::getBuildStream() {
  std::lock_guard<std::mutex> l(mutex_);
  return buildStream_;
}

// ============================================================================
// Build Operator Implementation
// ============================================================================
// Accumulates all build-side input batches in GPU memory and transfers them
// to the bridge when all input is received.

CudfNestedLoopJoinBuild::CudfNestedLoopJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::NestedLoopJoinNode> joinNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          nullptr,
          joinNode->id(),
          "CudfNestedLoopJoinBuild",
          nvtx3::rgb{65, 105, 225}, // Royal Blue
          NvtxMethodFlag::kNoMoreInput,
          std::nullopt,
          joinNode),
      joinNode_(joinNode) {}

// Accumulates input batches in memory.
// All batches are kept as CudfVectors (GPU memory) until join completes.
void CudfNestedLoopJoinBuild::doAddInput(RowVectorPtr input) {
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    inputs_.push_back(std::move(cudfInput)); // Store in GPU memory
  }
}

bool CudfNestedLoopJoinBuild::needsInput() const {
  return !noMoreInput_;
}

RowVectorPtr CudfNestedLoopJoinBuild::doGetOutput() {
  return nullptr;
}

// Called when upstream finishes. Coordinates with peer build operators
// to transfer accumulated data to the bridge.
//
// Multi-driver coordination:
// - Multiple build operators may run in parallel (one per driver)
// - allPeersFinished() chooses ONE operator to collect and transfer data
// - Other operators just return and mark themselves finished
// - The chosen operator collects data from all peers and sets it on the bridge
void CudfNestedLoopJoinBuild::doNoMoreInput() {
  Operator::noMoreInput();

  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;

  // Synchronization point: only the LAST driver to finish will proceed
  // Other drivers return here and will be woken when data transfer completes
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return; // Not the last driver - just wait
  }

  // This driver was chosen to collect data from all peers
  for (auto& peer : peers) {
    auto op = peer->findOperator(planNodeId());
    auto* build = dynamic_cast<CudfNestedLoopJoinBuild*>(op);
    VELOX_CHECK_NOT_NULL(build);
    inputs_.insert(
        inputs_.end(),
        std::make_move_iterator(build->inputs_.begin()),
        std::make_move_iterator(build->inputs_.end()));
  }

  // Wake up peer build operators when we finish transferring data
  SCOPE_EXIT {
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue(); // Unblock other build operators
    }
  };

  // Concatenate all input batches into a single cuDF table.
  // getConcatenatedTable throws if the total row count exceeds cudf::size_type
  // limits (~2.1B rows). We don't use getConcatenatedTableBatched here because
  // batching the build side does not prevent output overflow for NLJ: a cross
  // join output is probe_rows × build_rows regardless of how the build is
  // split.
  auto stream = cudfGlobalStreamPool().get_stream();
  auto table = getConcatenatedTable(
      std::exchange(inputs_, {}),
      joinNode_->sources()[1]->outputType(),
      stream,
      get_output_mr());

  // Transfer build data to bridge - this will unblock probe operators.
  // No stream sync is required: the probe side uses syncBuildStream() via a
  // CUDA event to ensure the build table is ready before reading it.
  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto bridge = std::dynamic_pointer_cast<CudfNestedLoopJoinBridge>(joinBridge);

  bridge->setBuildStream(stream); // Pass stream for CUDA synchronization
  bridge->setData(
      std::make_optional(
          std::shared_ptr<cudf::table>(std::move(table)))); // Wake probes
}

exec::BlockingReason CudfNestedLoopJoinBuild::isBlocked(
    ContinueFuture* future) {
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfNestedLoopJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

void CudfNestedLoopJoinBuild::doClose() {
  inputs_.clear();
  Operator::close();
}

// ============================================================================
// Probe Operator Implementation
// ============================================================================
// Performs the actual nested loop join by combining probe batches with
// build data using cuDF's cross_join or conditional_inner_join APIs.

CudfNestedLoopJoinProbe::CudfNestedLoopJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::NestedLoopJoinNode> joinNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          joinNode->outputType(),
          joinNode->id(),
          "CudfNestedLoopJoinProbe",
          nvtx3::rgb{0, 128, 128}, // Teal
          NvtxMethodFlag::kGetOutput | NvtxMethodFlag::kNoMoreInput,
          std::nullopt,
          joinNode),
      joinNode_(joinNode) {
  joinType_ = joinNode_->joinType();
  probeType_ = joinNode_->sources()[0]->outputType();
  buildType_ = joinNode_->sources()[1]->outputType();

  // For kLeftSemiProject, the last output column is a BOOLEAN match flag
  // that doesn't exist in probe or build types — skip it during resolution.
  auto numColumnsToResolve = outputType_->size();
  if (joinType_ == core::JoinType::kLeftSemiProject) {
    VELOX_CHECK_GE(numColumnsToResolve, 1);
    --numColumnsToResolve;
  }

  for (size_t i = 0; i < numColumnsToResolve; ++i) {
    const auto& name = outputType_->nameOf(i);
    auto probeIdx = probeType_->getChildIdxIfExists(name);
    if (probeIdx.has_value()) {
      probeColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(probeIdx.value()));
      probeColumnOutputIndices_.push_back(i);
      continue;
    }
    auto buildIdx = buildType_->getChildIdxIfExists(name);
    if (buildIdx.has_value()) {
      buildColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(buildIdx.value()));
      buildColumnOutputIndices_.push_back(i);
      continue;
    }
    VELOX_FAIL("Output column not found in probe or build types: {}", name);
  }
}

void CudfNestedLoopJoinProbe::initialize() {
  // Filter construction is deferred from the ctor to avoid memory allocation
  // during driver initialization. Mirrors #17045 for CudfHashJoinProbe.
  Operator::initialize();

  if (!joinNode_->joinCondition()) {
    return;
  }

  exec::ExprSet exprs({joinNode_->joinCondition()}, operatorCtx_->execCtx());
  VELOX_CHECK_EQ(exprs.exprs().size(), 1);

  // If the condition has a sub-expression that isn't natively
  // AST-representable and references columns from both sides (e.g.
  // `probe.col LIKE build.pattern`), it can't be precomputed on either side
  // before the join, so it can't be turned into a single cuDF AST tree at
  // all. Evaluate the whole condition generally instead, against the full
  // probe x build cross product (see crossJoinConditionalIndices), rather
  // than driving cudf::conditional_inner_join /
  // cudf::conditional_left_semi_join with an AST tree.
  if (hasNonAstSubexprSpanningBothSides(
          exprs.exprs()[0], probeType_, buildType_)) {
    useAstFilter_ = false;
    filterEvaluator_ = createCudfExpression(
        exprs.exprs()[0],
        facebook::velox::type::concatRowTypes({probeType_, buildType_}));
    hasFilter_ = true;
    return;
  }

  // Convert Velox expression to cuDF AST expression tree.
  // The AST will be passed to cudf::conditional_inner_join() for GPU
  // evaluation.
  createAstTree(
      exprs.exprs()[0],
      tree_,
      scalars_,
      probeType_,
      buildType_,
      leftPrecomputeInstructions_,
      rightPrecomputeInstructions_);

  // Set hasFilter_ only after the AST has been fully built so that a throw
  // from createAstTree() does not leave the operator marked as having a filter
  // with a partially-initialized tree.
  hasFilter_ = true;
}

void CudfNestedLoopJoinProbe::doClose() {
  Operator::close();
  buildData_.reset();
  probeMatchedFlags_.reset();
  buildMatchedFlags_.reset();
  buildPrecomputed_.clear();
  scalars_.clear();
  tree_ = {};
  filterEvaluator_.reset();
}

bool CudfNestedLoopJoinProbe::needsInput() const {
  return !noMoreInput_ && !finished_ && input_ == nullptr &&
      buildData_.has_value();
}

void CudfNestedLoopJoinProbe::doAddInput(RowVectorPtr input) {
  // Skip input processing when build is empty for join types with no output.
  if (skipInput_) {
    VELOX_CHECK_NULL(input_);
    return;
  }
  VELOX_CHECK_NULL(input_, "Probe input already set");
  input_ = std::move(input);
  probeMatchedFlags_.reset();
}

void CudfNestedLoopJoinProbe::doNoMoreInput() {
  Operator::noMoreInput();

  if (!isRightOrFullJoin()) {
    return;
  }

  // Empty build has no matched flags to merge across peers.
  if (buildEmpty_) {
    return;
  }

  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;

  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(),
          operatorCtx_->driver(),
          &peerFuture_,
          promises,
          peers)) {
    return;
  }

  SCOPE_EXIT {
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  isLastDriver_ = true;

  // Unfiltered cross_join matches every build row on every probe batch, so
  // every driver's buildMatchedFlags_ would be all-true. Skip the stream-join
  // and peer merge when there is no filter.
  if (!buildEmpty_ && hasFilter_) {
    auto stream = cudfGlobalStreamPool().get_stream();

    // GPU stream synchronization: allPeersFinished synchronizes CPU threads
    // but not GPU streams. A peer's CPU thread may have returned from
    // getOutput() while its GPU work (updating buildMatchedFlags_) is still
    // in flight. join_streams establishes GPU-side ordering.
    std::vector<rmm::cuda_stream_view> inputStreams;
    if (lastProbeStream_.has_value()) {
      inputStreams.push_back(lastProbeStream_.value());
    }
    for (auto& peer : peers) {
      if (peer.get() == operatorCtx_->driver()) {
        continue;
      }
      auto op = peer->findOperator(planNodeId());
      auto* probe = dynamic_cast<CudfNestedLoopJoinProbe*>(op);
      if (probe != nullptr && probe->lastProbeStream_.has_value()) {
        inputStreams.push_back(probe->lastProbeStream_.value());
      }
    }
    if (!inputStreams.empty()) {
      cudf::detail::join_streams(inputStreams, stream);
    }

    // Merge buildMatchedFlags_ from all peers via BITWISE_OR.
    for (auto& peer : peers) {
      if (peer.get() == operatorCtx_->driver()) {
        continue;
      }
      auto op = peer->findOperator(planNodeId());
      auto* probe = dynamic_cast<CudfNestedLoopJoinProbe*>(op);
      if (probe == nullptr) {
        continue;
      }
      auto orResult = cudf::binary_operation(
          buildMatchedFlags_->view(),
          probe->buildMatchedFlags_->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          get_temp_mr());
      // binary_operation is async on `stream`; the old column destructs via
      // cudaFreeAsync on its allocation stream (not `stream`), so the free
      // can race the kernel. Drain `stream` before the move-assign.
      stream.synchronize();
      buildMatchedFlags_ = std::move(orResult);
    }
  }
}

bool CudfNestedLoopJoinProbe::isFinished() {
  if (finished_) {
    return true;
  }
  // For right/full join, the last driver must not finish until build mismatch
  // rows have been emitted. Non-last drivers finish normally.
  if (isRightOrFullJoin() && noMoreInput_ && input_ == nullptr) {
    if (!isLastDriver_) {
      return true;
    }
    return buildMismatchEmitted_;
  }
  return false;
}

exec::BlockingReason CudfNestedLoopJoinProbe::isBlocked(
    ContinueFuture* future) {
  // For right/full join: after build data is available, also block on peer
  // probes finishing (allPeersFinished barrier in noMoreInput).
  if (isRightOrFullJoin() && buildData_.has_value()) {
    if (!peerFuture_.valid()) {
      return exec::BlockingReason::kNotBlocked;
    }
    *future = std::move(peerFuture_);
    return exec::BlockingReason::kWaitForJoinProbe;
  }

  if (buildData_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto bridge = std::dynamic_pointer_cast<CudfNestedLoopJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(bridge);
  VELOX_CHECK_NOT_NULL(future);

  buildData_ = bridge->dataOrFuture(future);
  if (!buildData_.has_value()) {
    return exec::BlockingReason::kWaitForJoinBuild;
  }

  buildStream_ = bridge->getBuildStream();

  if (buildData_.value()->num_rows() == 0) {
    buildEmpty_ = true;
    // For inner/right join, set skipInput_ to consume probe batches without
    // processing (prevents upstream exchange hanging). Match CPU NLJ behavior
    // which always consumes input rather than finishing early.
    if (skipProbeOnEmptyBuild()) {
      skipInput_ = true;
    }
  }

  // Initialize build matched flags for filtered right/full join (single BOOL8
  // column with one element per build row, all false). Unfiltered cross_join
  // matches every build row, so flags aren't needed.
  if (isRightOrFullJoin() && hasFilter_ && !buildEmpty_) {
    auto initStream = cudfGlobalStreamPool().get_stream();
    auto numRows = buildData_.value()->num_rows();
    auto falseScalar =
        cudf::numeric_scalar<bool>(false, true, initStream, get_temp_mr());
    buildMatchedFlags_ = cudf::make_column_from_scalar(
        falseScalar, numRows, initStream, get_temp_mr());
    initStream.synchronize();
  }

  // Precompute build-side sub-expressions for filter evaluation (once, here,
  // since the build table is fixed for the lifetime of this probe operator).
  if (hasFilter_ && !rightPrecomputeInstructions_.empty() && !buildEmpty_) {
    auto precomputeStream = cudfGlobalStreamPool().get_stream();
    auto buildColumnViews = tableViewToColumnViews(buildData_.value()->view());
    buildPrecomputed_ = precomputeSubexpressions(
        buildColumnViews,
        rightPrecomputeInstructions_,
        scalars_,
        buildType_,
        precomputeStream);
    buildExtendedView_ =
        createExtendedTableView(buildData_.value()->view(), buildPrecomputed_);
    precomputeStream.synchronize();
  }

  return exec::BlockingReason::kNotBlocked;
}

void CudfNestedLoopJoinProbe::syncBuildStream(
    rmm::cuda_stream_view probeStream) {
  if (buildStream_.has_value()) {
    if (!cudaEvent_) {
      cudaEvent_ = std::make_unique<CudaEvent>();
    }
    cudaEvent_->recordFrom(buildStream_.value()).waitOn(probeStream);
    buildStream_.reset();
  }
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
CudfNestedLoopJoinProbe::crossJoinConditionalIndices(
    cudf::table_view probeTableView,
    cudf::table_view buildView,
    rmm::cuda_stream_view stream) {
  VELOX_NVTX_FUNC_RANGE();
  auto mr = get_temp_mr();

  const auto numProbeRows = probeTableView.num_rows();
  const auto numBuildRows = buildView.num_rows();
  const auto totalRows =
      static_cast<int64_t>(numProbeRows) * static_cast<int64_t>(numBuildRows);
  VELOX_CHECK_LE(
      totalRows,
      std::numeric_limits<cudf::size_type>::max(),
      "Cross product for join condition exceeds cudf::size_type limit: "
      "{} x {} = {} rows",
      numProbeRows,
      numBuildRows,
      totalRows);

  // probeIndices[i] = i / numBuildRows, buildIndices[i] = i % numBuildRows,
  // for i in [0, numProbeRows * numBuildRows). This is the same probe-major
  // row order cudf::cross_join uses, but materialized as explicit index
  // columns (rather than gathered data) so the filtered result below can
  // reuse the same contains()-based matched-flag logic that
  // cudf::conditional_inner_join's index-pair output already uses in the
  // caller.
  const auto totalRowsAsSizeType = static_cast<cudf::size_type>(totalRows);
  auto zero = cudf::numeric_scalar<cudf::size_type>(0, true, stream, mr);
  auto one = cudf::numeric_scalar<cudf::size_type>(1, true, stream, mr);
  auto sequence = cudf::sequence(totalRowsAsSizeType, zero, one, stream, mr);
  auto buildRowsScalar =
      cudf::numeric_scalar<cudf::size_type>(numBuildRows, true, stream, mr);
  auto probeIndices = cudf::binary_operation(
      sequence->view(),
      buildRowsScalar,
      cudf::binary_operator::DIV,
      cudf::data_type{cudf::type_to_id<cudf::size_type>()},
      stream,
      mr);
  auto buildIndices = cudf::binary_operation(
      sequence->view(),
      buildRowsScalar,
      cudf::binary_operator::MOD,
      cudf::data_type{cudf::type_to_id<cudf::size_type>()},
      stream,
      mr);

  auto gatheredProbe = cudf::gather(
      probeTableView,
      probeIndices->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      mr);
  auto gatheredBuild = cudf::gather(
      buildView,
      buildIndices->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      mr);

  std::vector<cudf::column_view> combinedViews;
  auto gatheredProbeView = gatheredProbe->view();
  auto gatheredBuildView = gatheredBuild->view();
  combinedViews.reserve(
      gatheredProbeView.num_columns() + gatheredBuildView.num_columns());
  for (cudf::size_type i = 0; i < gatheredProbeView.num_columns(); ++i) {
    combinedViews.push_back(gatheredProbeView.column(i));
  }
  for (cudf::size_type i = 0; i < gatheredBuildView.num_columns(); ++i) {
    combinedViews.push_back(gatheredBuildView.column(i));
  }

  VELOX_CHECK_NOT_NULL(
      filterEvaluator_,
      "Join filter evaluator must be initialized before "
      "crossJoinConditionalIndices");
  auto filterColumn = filterEvaluator_->eval(combinedViews, stream, mr);
  auto mask = asView(filterColumn);

  auto filteredProbeIndices = cudf::apply_boolean_mask(
      cudf::table_view{{probeIndices->view()}}, mask, stream, mr);
  auto filteredBuildIndices = cudf::apply_boolean_mask(
      cudf::table_view{{buildIndices->view()}}, mask, stream, mr);

  auto probeIndicesCols = filteredProbeIndices->release();
  auto buildIndicesCols = filteredBuildIndices->release();
  return {std::move(probeIndicesCols[0]), std::move(buildIndicesCols[0])};
}

std::unique_ptr<cudf::table> CudfNestedLoopJoinProbe::joinWithBuildBatch(
    cudf::table_view probeTableView,
    cudf::table_view buildView,
    rmm::cuda_stream_view stream) {
  VELOX_NVTX_FUNC_RANGE();

  syncBuildStream(stream);

  auto numOutputColumns = outputType_->size();

  // Extend probe view with precomputed columns for filter AST evaluation.
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedProbeView = probeTableView;
  if (hasFilter_ && !leftPrecomputeInstructions_.empty()) {
    auto probeColumnViews = tableViewToColumnViews(probeTableView);
    leftPrecomputed = precomputeSubexpressions(
        probeColumnViews,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedProbeView =
        createExtendedTableView(probeTableView, leftPrecomputed);
  }
  // Use cached extended build view if build-side precompute was needed.
  const cudf::table_view& extendedBuildView =
      buildPrecomputed_.empty() ? buildView : buildExtendedView_;

  if (hasFilter_) {
    VELOX_CHECK(
        isInitialized(),
        "Filter must be initialized before joinWithBuildBatch");

    // Owning storage for whichever path below produces the index pairs;
    // leftIndicesView/rightIndicesView alias into one of these two.
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftIndicesBuffer;
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> rightIndicesBuffer;
    std::unique_ptr<cudf::column> leftIndicesColumn;
    std::unique_ptr<cudf::column> rightIndicesColumn;
    cudf::column_view leftIndicesView;
    cudf::column_view rightIndicesView;

    if (useAstFilter_) {
      std::tie(leftIndicesBuffer, rightIndicesBuffer) =
          cudf::conditional_inner_join(
              extendedProbeView,
              extendedBuildView,
              tree_.back(),
              std::nullopt,
              stream,
              get_temp_mr());

      VELOX_CHECK_LE(
          static_cast<int64_t>(leftIndicesBuffer->size()),
          std::numeric_limits<cudf::size_type>::max(),
          "Conditional join output exceeds cudf::size_type limit: {} rows",
          leftIndicesBuffer->size());

      leftIndicesView = cudf::column_view(
          cudf::data_type{cudf::type_to_id<cudf::size_type>()},
          leftIndicesBuffer->size(),
          leftIndicesBuffer->data(),
          nullptr,
          0);
      rightIndicesView = cudf::column_view(
          cudf::data_type{cudf::type_to_id<cudf::size_type>()},
          rightIndicesBuffer->size(),
          rightIndicesBuffer->data(),
          nullptr,
          0);
    } else {
      // Condition spans both sides with a non-AST sub-expression; evaluate
      // it generally against the full cross product instead of driving
      // cudf::conditional_inner_join with an AST tree.
      std::tie(leftIndicesColumn, rightIndicesColumn) =
          crossJoinConditionalIndices(probeTableView, buildView, stream);
      leftIndicesView = leftIndicesColumn->view();
      rightIndicesView = rightIndicesColumn->view();
    }

    // Track which probe rows matched for left/full join mismatch handling.
    // Uses cudf::contains to check which probe row indices [0..N) appear
    // in the join result.
    if (isLeftOrFullJoin()) {
      auto numProbeRows = probeTableView.num_rows();
      auto probeRowSequence = cudf::sequence(
          numProbeRows,
          cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
          cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
          stream,
          get_temp_mr());

      // The build side is concatenated into a single table (see
      // CudfNestedLoopJoinBuild::doNoMoreInput), so joinWithBuildBatch runs
      // exactly once per probe input. probeMatchedFlags_ is the result of
      // this single contains() call; no cross-batch BITWISE_OR is needed.
      probeMatchedFlags_ = cudf::contains(
          leftIndicesView, probeRowSequence->view(), stream, get_temp_mr());
    }

    // Track which build rows matched for right/full join mismatch handling.
    if (isRightOrFullJoin()) {
      auto numBuildRows = buildView.num_rows();
      auto buildRowSequence = cudf::sequence(
          numBuildRows,
          cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
          cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
          stream,
          get_temp_mr());

      auto matchedInBatch = cudf::contains(
          rightIndicesView, buildRowSequence->view(), stream, get_temp_mr());

      auto updatedFlags = cudf::binary_operation(
          buildMatchedFlags_->view(),
          matchedInBatch->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          get_temp_mr());
      stream.synchronize();
      buildMatchedFlags_ = std::move(updatedFlags);
    }

    // Gather only the columns needed for output.
    auto probeGatherView = probeTableView.select(probeColumnIndicesToGather_);
    auto buildGatherView = buildView.select(buildColumnIndicesToGather_);

    auto gatheredProbe = cudf::gather(
        probeGatherView,
        leftIndicesView,
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        get_output_mr());

    auto gatheredBuild = cudf::gather(
        buildGatherView,
        rightIndicesView,
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        get_output_mr());

    std::vector<std::unique_ptr<cudf::column>> outCols(numOutputColumns);
    auto probeCols = gatheredProbe->release();
    auto buildCols = gatheredBuild->release();
    for (size_t i = 0; i < probeColumnOutputIndices_.size(); ++i) {
      outCols[probeColumnOutputIndices_[i]] = std::move(probeCols[i]);
    }
    for (size_t i = 0; i < buildColumnOutputIndices_.size(); ++i) {
      outCols[buildColumnOutputIndices_[i]] = std::move(buildCols[i]);
    }

    return std::make_unique<cudf::table>(std::move(outCols));
  }

  // Unfiltered join using cross_join.
  auto outputRows = static_cast<int64_t>(probeTableView.num_rows()) *
      static_cast<int64_t>(buildView.num_rows());
  VELOX_CHECK_LE(
      outputRows,
      std::numeric_limits<cudf::size_type>::max(),
      "Cross join output exceeds cudf::size_type limit: {} x {} = {} rows",
      probeTableView.num_rows(),
      buildView.num_rows(),
      outputRows);

  auto crossResult =
      cudf::cross_join(probeTableView, buildView, stream, get_output_mr());

  // Cross join matches every row, so no per-row matched flags are needed:
  // probeMatchedFlags_ is only consumed via emitProbeMismatchRows, which is
  // unreachable in the unfiltered path (see doGetOutput). buildMatchedFlags_
  // is skipped in isBlocked for !hasFilter_; emitBuildMismatchRows early-
  // returns in that case.

  auto allCols = crossResult->release();
  auto numProbeCols = probeTableView.num_columns();

  std::vector<std::unique_ptr<cudf::column>> outCols(numOutputColumns);
  for (size_t i = 0; i < probeColumnOutputIndices_.size(); ++i) {
    outCols[probeColumnOutputIndices_[i]] =
        std::move(allCols[probeColumnIndicesToGather_[i]]);
  }
  for (size_t i = 0; i < buildColumnOutputIndices_.size(); ++i) {
    outCols[buildColumnOutputIndices_[i]] =
        std::move(allCols[numProbeCols + buildColumnIndicesToGather_[i]]);
  }

  return std::make_unique<cudf::table>(std::move(outCols));
}

std::unique_ptr<cudf::table> CudfNestedLoopJoinProbe::emitProbeMismatchRows(
    cudf::table_view probeTableView,
    rmm::cuda_stream_view stream) {
  auto probeGatherView = probeTableView.select(probeColumnIndicesToGather_);

  std::unique_ptr<cudf::table> unmatchedProbe;
  if (!probeMatchedFlags_) {
    // No flags means all probe rows are unmatched (empty build case).
    unmatchedProbe =
        std::make_unique<cudf::table>(probeGatherView, stream, get_output_mr());
  } else {
    auto unmatchedMask = cudf::unary_operation(
        probeMatchedFlags_->view(),
        cudf::unary_operator::NOT,
        stream,
        get_temp_mr());
    unmatchedProbe = cudf::apply_boolean_mask(
        probeGatherView, unmatchedMask->view(), stream, get_output_mr());
  }

  auto numUnmatched = static_cast<cudf::size_type>(unmatchedProbe->num_rows());
  if (numUnmatched == 0) {
    return nullptr;
  }

  auto numOutputColumns = outputType_->size();
  std::vector<std::unique_ptr<cudf::column>> outCols(numOutputColumns);

  // Place unmatched probe columns at their output positions.
  auto probeCols = unmatchedProbe->release();
  for (size_t i = 0; i < probeColumnOutputIndices_.size(); ++i) {
    outCols[probeColumnOutputIndices_[i]] = std::move(probeCols[i]);
  }

  // Create all-null columns for the build side.
  for (size_t i = 0; i < buildColumnOutputIndices_.size(); ++i) {
    auto outIdx = buildColumnOutputIndices_[i];
    auto buildChannel = buildColumnIndicesToGather_[i];
    auto buildCudfDataType =
        veloxToCudfDataType(buildType_->childAt(buildChannel));
    auto nullScalar = cudf::make_default_constructed_scalar(
        buildCudfDataType, stream, get_temp_mr());
    outCols[outIdx] = cudf::make_column_from_scalar(
        *nullScalar, numUnmatched, stream, get_output_mr());
  }

  return std::make_unique<cudf::table>(std::move(outCols));
}

RowVectorPtr CudfNestedLoopJoinProbe::emitBuildMismatchRows(
    rmm::cuda_stream_view stream) {
  // Unfiltered cross_join already emitted every build row, so no mismatches
  // to emit. buildMatchedFlags_ is not allocated in that case.
  if (!buildMatchedFlags_) {
    finished_ = true;
    return nullptr;
  }
  auto& buildTable = buildData_.value();
  auto numOutputColumns = outputType_->size();

  // Invert flags: unmatched = NOT(matched).
  auto unmatchedMask = cudf::unary_operation(
      buildMatchedFlags_->view(),
      cudf::unary_operator::NOT,
      stream,
      get_temp_mr());

  // Select unmatched build rows.
  auto buildGatherView = buildTable->view().select(buildColumnIndicesToGather_);
  auto unmatchedBuild = cudf::apply_boolean_mask(
      buildGatherView, unmatchedMask->view(), stream, get_output_mr());
  auto numUnmatched = static_cast<cudf::size_type>(unmatchedBuild->num_rows());

  finished_ = true;
  if (numUnmatched == 0) {
    return nullptr;
  }

  std::vector<std::unique_ptr<cudf::column>> outCols(numOutputColumns);

  // Create all-null columns for the probe side.
  for (size_t li = 0; li < probeColumnOutputIndices_.size(); ++li) {
    auto outIdx = probeColumnOutputIndices_[li];
    auto probeChannel = probeColumnIndicesToGather_[li];
    auto probeCudfDataType =
        veloxToCudfDataType(probeType_->childAt(probeChannel));
    auto nullScalar = cudf::make_default_constructed_scalar(
        probeCudfDataType, stream, get_temp_mr());
    outCols[outIdx] = cudf::make_column_from_scalar(
        *nullScalar, numUnmatched, stream, get_output_mr());
  }

  // Place unmatched build columns at their output positions.
  auto buildCols = unmatchedBuild->release();
  for (size_t ri = 0; ri < buildColumnOutputIndices_.size(); ++ri) {
    outCols[buildColumnOutputIndices_[ri]] = std::move(buildCols[ri]);
  }

  auto out = std::make_unique<cudf::table>(std::move(outCols));
  auto size = static_cast<vector_size_t>(out->num_rows());
  return std::make_shared<CudfVector>(
      operatorCtx_->pool(), outputType_, size, std::move(out), stream);
}

RowVectorPtr CudfNestedLoopJoinProbe::doGetOutput() {
  if (!input_) {
    // Right/full join: after all probe inputs, the last driver emits
    // unmatched build rows with null probe columns.
    if (isRightOrFullJoin() && noMoreInput_ && isLastDriver_ &&
        !buildMismatchEmitted_) {
      buildMismatchEmitted_ = true;
      auto stream = cudfGlobalStreamPool().get_stream();
      return emitBuildMismatchRows(stream);
    }
    if (noMoreInput_) {
      finished_ = true;
    }
    return nullptr;
  }

  VELOX_CHECK(buildData_.has_value(), "Build data not available in getOutput");
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput);
  auto stream = cudfInput->stream();
  lastProbeStream_ = stream;

  // LeftSemiProject: emit all probe rows with a boolean match column.
  if (joinType_ == core::JoinType::kLeftSemiProject) {
    auto probeTableView = cudfInput->getTableView();
    auto numProbeRows = static_cast<cudf::size_type>(probeTableView.num_rows());

    std::unique_ptr<cudf::column> matchFlags;
    if (buildEmpty_ || !hasFilter_) {
      // No filter + non-empty build: all probe rows match (true).
      // Empty build: no probe rows match (false).
      auto scalar =
          cudf::numeric_scalar<bool>(!buildEmpty_, true, stream, get_temp_mr());
      matchFlags = cudf::make_column_from_scalar(
          scalar, numProbeRows, stream, get_temp_mr());
    } else {
      // Filtered: compute matched probe indices against the single build table.
      auto falseScalar =
          cudf::numeric_scalar<bool>(false, true, stream, get_temp_mr());
      matchFlags = cudf::make_column_from_scalar(
          falseScalar, numProbeRows, stream, get_temp_mr());

      // Owning storage for whichever path below produces the matched probe
      // indices; matchedIndicesView aliases into one of these two.
      std::unique_ptr<rmm::device_uvector<cudf::size_type>>
          matchedIndicesBuffer;
      std::unique_ptr<cudf::column> matchedIndicesColumn;
      cudf::size_type matchedIndicesSize = 0;
      cudf::column_view matchedIndicesView;

      if (useAstFilter_) {
        // Extend probe view with precomputed columns if needed.
        std::vector<ColumnOrView> leftPrecomputed;
        cudf::table_view extendedProbeView = probeTableView;
        if (!leftPrecomputeInstructions_.empty()) {
          auto probeColumnViews = tableViewToColumnViews(probeTableView);
          leftPrecomputed = precomputeSubexpressions(
              probeColumnViews,
              leftPrecomputeInstructions_,
              scalars_,
              probeType_,
              stream);
          extendedProbeView =
              createExtendedTableView(probeTableView, leftPrecomputed);
        }
        const cudf::table_view& extendedBuildView = buildPrecomputed_.empty()
            ? buildData_.value()->view()
            : buildExtendedView_;

        matchedIndicesBuffer = cudf::conditional_left_semi_join(
            extendedProbeView,
            extendedBuildView,
            tree_.back(),
            {},
            stream,
            get_temp_mr());
        matchedIndicesSize =
            static_cast<cudf::size_type>(matchedIndicesBuffer->size());
        matchedIndicesView = cudf::column_view(
            cudf::data_type{cudf::type_to_id<cudf::size_type>()},
            matchedIndicesBuffer->size(),
            matchedIndicesBuffer->data(),
            nullptr,
            0);
      } else {
        // Condition spans both sides with a non-AST sub-expression; a probe
        // row "matches" (for the semi-join match flag) if it appears at all
        // among the filtered cross-product probe indices.
        auto [probeIndicesForSemiJoin, buildIndicesForSemiJoin] =
            crossJoinConditionalIndices(
                probeTableView, buildData_.value()->view(), stream);
        matchedIndicesColumn = std::move(probeIndicesForSemiJoin);
        matchedIndicesSize = matchedIndicesColumn->size();
        matchedIndicesView = matchedIndicesColumn->view();
      }

      if (matchedIndicesSize > 0) {
        // Build a sequence [0..numProbeRows) and check which indices
        // appear in the semi-join result.
        auto probeRowSequence = cudf::sequence(
            numProbeRows,
            cudf::numeric_scalar<cudf::size_type>(
                0, true, stream, get_temp_mr()),
            cudf::numeric_scalar<cudf::size_type>(
                1, true, stream, get_temp_mr()),
            stream,
            get_temp_mr());

        auto matchedInBatch = cudf::contains(
            matchedIndicesView,
            probeRowSequence->view(),
            stream,
            get_temp_mr());

        matchFlags = cudf::binary_operation(
            matchFlags->view(),
            matchedInBatch->view(),
            cudf::binary_operator::BITWISE_OR,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            get_temp_mr());
      }
    }

    // Copy match flags into output memory resource since they go into the
    // output table passed downstream.
    auto outputMatchFlags = std::make_unique<cudf::column>(
        matchFlags->view(), stream, get_output_mr());

    // Assemble output: probe columns at their mapped positions + match column
    // at the last position.
    auto probeGatherView = probeTableView.select(probeColumnIndicesToGather_);
    auto gatheredProbe =
        std::make_unique<cudf::table>(probeGatherView, stream, get_output_mr());
    auto probeCols = gatheredProbe->release();

    auto numOutputColumns = outputType_->size();
    std::vector<std::unique_ptr<cudf::column>> outCols(numOutputColumns);
    for (size_t i = 0; i < probeColumnOutputIndices_.size(); ++i) {
      outCols[probeColumnOutputIndices_[i]] = std::move(probeCols[i]);
    }
    outCols[numOutputColumns - 1] = std::move(outputMatchFlags);

    auto result = std::make_unique<cudf::table>(std::move(outCols));
    input_.reset();

    if (result->num_rows() == 0) {
      return nullptr;
    }
    auto size = static_cast<vector_size_t>(result->num_rows());
    return std::make_shared<CudfVector>(
        operatorCtx_->pool(), outputType_, size, std::move(result), stream);
  }

  // For left/full join with filter: two-phase per probe input.
  // Phase 1 (probeMatchedFlags_ null): join; joinWithBuildBatch populates
  // probeMatchedFlags_ from the single build table.
  // Phase 2 (probeMatchedFlags_ set): emit unmatched probe rows.
  if (isLeftOrFullJoin() && hasFilter_ && !buildEmpty_) {
    if (probeMatchedFlags_ == nullptr) {
      auto result = joinWithBuildBatch(
          cudfInput->getTableView(), buildData_.value()->view(), stream);
      if (result->num_rows() > 0) {
        auto size = static_cast<vector_size_t>(result->num_rows());
        return std::make_shared<CudfVector>(
            operatorCtx_->pool(), outputType_, size, std::move(result), stream);
      }
      // Join produced no matched rows; fall through to mismatch emission.
    }

    // Emit unmatched probe rows with null build columns.
    auto mismatchResult =
        emitProbeMismatchRows(cudfInput->getTableView(), stream);
    input_.reset();
    probeMatchedFlags_.reset();
    if (mismatchResult && mismatchResult->num_rows() > 0) {
      auto size = static_cast<vector_size_t>(mismatchResult->num_rows());
      return std::make_shared<CudfVector>(
          operatorCtx_->pool(),
          outputType_,
          size,
          std::move(mismatchResult),
          stream);
    }
    return nullptr;
  }

  // Join probe against the single build table.
  if (!buildEmpty_) {
    auto result = joinWithBuildBatch(
        cudfInput->getTableView(), buildData_.value()->view(), stream);
    if (result->num_rows() > 0) {
      input_.reset();
      auto size = static_cast<vector_size_t>(result->num_rows());
      return std::make_shared<CudfVector>(
          operatorCtx_->pool(), outputType_, size, std::move(result), stream);
    }
  }

  // Left/full join with empty build: emit all probe rows as mismatches.
  if (isLeftOrFullJoin() && buildEmpty_) {
    auto mismatchResult =
        emitProbeMismatchRows(cudfInput->getTableView(), stream);
    input_.reset();
    probeMatchedFlags_.reset();
    if (mismatchResult && mismatchResult->num_rows() > 0) {
      auto size = static_cast<vector_size_t>(mismatchResult->num_rows());
      return std::make_shared<CudfVector>(
          operatorCtx_->pool(),
          outputType_,
          size,
          std::move(mismatchResult),
          stream);
    }
    return nullptr;
  }

  input_.reset();
  return nullptr;
}

// BridgeTranslator implementation
std::unique_ptr<exec::Operator> CudfNestedLoopJoinBridgeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(node)) {
    return std::make_unique<CudfNestedLoopJoinProbe>(id, ctx, joinNode);
  }
  return nullptr;
}

std::unique_ptr<exec::JoinBridge>
CudfNestedLoopJoinBridgeTranslator::toJoinBridge(
    const core::PlanNodePtr& /* node */) {
  return std::make_unique<CudfNestedLoopJoinBridge>();
}

exec::OperatorSupplier CudfNestedLoopJoinBridgeTranslator::toOperatorSupplier(
    const core::PlanNodePtr& node) {
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(node)) {
    return [joinNode](int32_t operatorId, exec::DriverCtx* ctx) {
      return std::make_unique<CudfNestedLoopJoinBuild>(
          operatorId, ctx, joinNode);
    };
  }
  return nullptr;
}

} // namespace facebook::velox::cudf_velox
