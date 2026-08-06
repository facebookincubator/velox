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
#include "velox/expression/ExprOptimizer.h"

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

namespace facebook::velox::cudf_velox {

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

void CudfNestedLoopJoinBridge::setBuildReadyEvent(
    std::shared_ptr<CudaEvent> buildReadyEvent) {
  std::lock_guard<std::mutex> l(mutex_);
  buildReadyEvent_ = std::move(buildReadyEvent);
}

std::shared_ptr<CudaEvent> CudfNestedLoopJoinBridge::getBuildReadyEvent() {
  std::lock_guard<std::mutex> l(mutex_);
  return buildReadyEvent_;
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
    : CudfJoinBuild(
          operatorId,
          driverCtx,
          joinNode,
          "CudfNestedLoopJoinBuild",
          NvtxMethodFlag::kNoMoreInput),
      joinNode_(joinNode) {}

void CudfNestedLoopJoinBuild::buildAndPublish(
    std::vector<CudfVectorPtr> inputs) {
  // Concatenate all input batches into a single cuDF table.
  // getConcatenatedTable throws if the total row count exceeds cudf::size_type
  // limits (~2.1B rows). We don't use getConcatenatedTableBatched here because
  // batching the build side does not prevent output overflow for NLJ: a cross
  // join output is probe_rows × build_rows regardless of how the build is
  // split.
  auto stream = cudfGlobalStreamPool().get_stream();
  auto table = getConcatenatedTable(
      std::move(inputs),
      joinNode_->sources()[1]->outputType(),
      stream,
      get_output_mr());

  // Record the build-ready event now, immediately after the build table
  // is materialized on `stream` - not lazily on the probe side - so it
  // captures this exact completion point before `stream` can be recycled
  // by cudfGlobalStreamPool() for unrelated work. Every probe operator
  // instance/batch just waits on this same event before reading buildData_
  // (see CudfNestedLoopJoinProbe::waitForBuildReady()). `stream` is also
  // exposed via setBuildStream() below: buildData_'s eventual free is
  // stream-ordered on `stream` (it was allocated here), so every probe read
  // makes `stream` wait on its own completion first (see
  // CudfNestedLoopJoinProbe::recordReadCompletion()), ensuring that free
  // can't run before all such reads are done.
  auto buildReadyEvent = std::make_shared<CudaEvent>(cudaEventDisableTiming);
  buildReadyEvent->recordFrom(stream);

  // Transfer build data to bridge - this will unblock probe operators.
  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto bridge = std::dynamic_pointer_cast<CudfNestedLoopJoinBridge>(joinBridge);

  bridge->setBuildReadyEvent(std::move(buildReadyEvent));
  bridge->setBuildStream(stream);
  bridge->setData(
      std::make_optional(
          std::shared_ptr<cudf::table>(std::move(table)))); // Wake probes
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

  std::optional<std::size_t> syntheticOutputPosition;
  if (joinType_ == core::JoinType::kLeftSemiProject) {
    VELOX_CHECK_GT(outputType_->size(), 0);
    syntheticOutputPosition = outputType_->size() - 1;
  }
  outputLayout_ = CudfJoinOutputLayout(
      probeType_, buildType_, outputType_, syntheticOutputPosition);
}

void CudfNestedLoopJoinProbe::initialize() {
  // Filter construction is deferred from the ctor to avoid memory allocation
  // during driver initialization. Mirrors #17045 for CudfHashJoinProbe.
  Operator::initialize();

  if (!joinNode_->joinCondition()) {
    return;
  }

  auto* const pool = operatorCtx_->pool();

  // Optimize (rewrites + constant folding) the join condition before building
  // the AST so CudfFunctions never see scalar-only operand sets. This carries
  // over the constant folding the exec::ExprSet used to perform.
  const auto optimizedCondition = expression::optimize(
      joinNode_->joinCondition(), operatorCtx_->execCtx()->queryCtx(), pool);
  VELOX_CHECK_NOT_NULL(optimizedCondition);

  // Convert Velox typed expression to cuDF AST expression tree.
  // The AST will be passed to cudf::conditional_inner_join() for GPU
  // evaluation.
  createAstTree(
      optimizedCondition,
      tree_,
      scalars_,
      probeType_,
      buildType_,
      leftPrecomputeInstructions_,
      rightPrecomputeInstructions_,
      pool);

  // Set hasFilter_ only after the AST has been fully built so that a throw
  // from createAstTree() does not leave the operator marked as having a filter
  // with a partially-initialized tree.
  hasFilter_ = true;
}

void CudfNestedLoopJoinProbe::doClose() {
  Operator::close();
  // No explicit stream sync needed here: every read of buildData_/
  // buildPrecomputed_/scalars_/buildMatchedFlags_ already called
  // recordReadCompletion(), which makes buildStream_ wait on that read's
  // completion. buildData_'s eventual stream-ordered free is enqueued on
  // buildStream_ too (it was allocated there), so CUDA's in-order stream
  // execution guarantees that free can't run before any of those reads -
  // whichever probe instance's reset() below actually triggers it. See
  // recordReadCompletion().
  buildData_.reset();
  probeMatchedFlags_.reset();
  buildMatchedFlags_.reset();
  buildPrecomputed_.clear();
  scalars_.clear();
  tree_ = {};
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

  buildReadyEvent_ = bridge->getBuildReadyEvent();
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
    waitForBuildReady(precomputeStream);
    auto buildColumnViews = tableViewToColumnViews(buildData_.value()->view());
    buildPrecomputed_ = precomputeSubexpressions(
        buildColumnViews,
        rightPrecomputeInstructions_,
        scalars_,
        buildType_,
        precomputeStream);
    buildExtendedView_ =
        makeExtendedTableView(buildData_.value()->view(), buildPrecomputed_);
    precomputeStream.synchronize();
  }

  return exec::BlockingReason::kNotBlocked;
}

void CudfNestedLoopJoinProbe::waitForBuildReady(
    rmm::cuda_stream_view probeStream) {
  if (buildReadyEvent_ != nullptr) {
    // joinWithBuildBatch() is called once per probe input batch, and each
    // call gets a fresh stream from cudfGlobalStreamPool(). The event was
    // already recorded once by the build side (see
    // CudfNestedLoopJoinBuild::buildAndPublish()); every probe stream that
    // reads build-side data just needs to wait on it, not just the first
    // one - otherwise later batches could start reading before the build
    // data is actually visible on their stream. Matches
    // CudfHashJoinProbe::waitForBuildReady().
    buildReadyEvent_->waitOn(probeStream);
  }
}

void CudfNestedLoopJoinProbe::recordReadCompletion(
    rmm::cuda_stream_view probeStream) {
  if (buildStream_.has_value()) {
    // buildData_'s underlying device memory is allocated on buildStream_
    // (see CudfNestedLoopJoinBuild::buildAndPublish()), so its eventual free
    // is stream-ordered there too, regardless of which probe instance's
    // reference-drop actually triggers it. Recording a completion event
    // from probeStream and waiting on it from buildStream_ chains a
    // dependency: buildStream_ cannot proceed past this point (including
    // to the eventual free) until this read has finished. Every probe
    // batch/instance calls this after reading build-side state, so
    // buildStream_ accumulates a wait for every one of them - matches
    // CudfHashJoinProbe's cudaEvent_/buildStream_ pattern.
    if (!cudaEvent_) {
      cudaEvent_ = std::make_unique<CudaEvent>();
    }
    cudaEvent_->recordFrom(probeStream).waitOn(buildStream_.value());
  }
}

std::unique_ptr<cudf::table> CudfNestedLoopJoinProbe::joinWithBuildBatch(
    cudf::table_view probeTableView,
    cudf::table_view buildView,
    rmm::cuda_stream_view stream) {
  VELOX_NVTX_FUNC_RANGE();

  // Both call sites are in doGetOutput(), which already waits for the
  // build-ready event on this same stream before calling in here.

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
    extendedProbeView = makeExtendedTableView(probeTableView, leftPrecomputed);
  }
  // Use cached extended build view if build-side precompute was needed.
  const cudf::table_view& extendedBuildView =
      buildPrecomputed_.empty() ? buildView : buildExtendedView_;

  if (hasFilter_) {
    VELOX_CHECK(
        isInitialized(),
        "Filter must be initialized before joinWithBuildBatch");
    auto [leftIndices, rightIndices] = cudf::conditional_inner_join(
        extendedProbeView,
        extendedBuildView,
        tree_.back(),
        std::nullopt,
        stream,
        get_temp_mr());

    VELOX_CHECK_LE(
        static_cast<int64_t>(leftIndices->size()),
        std::numeric_limits<cudf::size_type>::max(),
        "Conditional join output exceeds cudf::size_type limit: {} rows",
        leftIndices->size());

    auto leftIndicesView = cudf::column_view(
        cudf::data_type{cudf::type_to_id<cudf::size_type>()},
        leftIndices->size(),
        leftIndices->data(),
        nullptr,
        0);

    auto rightIndicesView = cudf::column_view(
        cudf::data_type{cudf::type_to_id<cudf::size_type>()},
        rightIndices->size(),
        rightIndices->data(),
        nullptr,
        0);

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
      // CudfNestedLoopJoinBuild::buildAndPublish), so joinWithBuildBatch runs
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
    auto probeGatherView =
        probeTableView.select(outputLayout_.probeColumnIndices);
    auto buildGatherView = buildView.select(outputLayout_.buildColumnIndices);

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
    for (size_t i = 0; i < outputLayout_.probeColumnOutputPositions.size();
         ++i) {
      outCols[outputLayout_.probeColumnOutputPositions[i]] =
          std::move(probeCols[i]);
    }
    for (size_t i = 0; i < outputLayout_.buildColumnOutputPositions.size();
         ++i) {
      outCols[outputLayout_.buildColumnOutputPositions[i]] =
          std::move(buildCols[i]);
    }

    recordReadCompletion(stream);
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
  for (size_t i = 0; i < outputLayout_.probeColumnOutputPositions.size(); ++i) {
    outCols[outputLayout_.probeColumnOutputPositions[i]] =
        std::move(allCols[outputLayout_.probeColumnIndices[i]]);
  }
  for (size_t i = 0; i < outputLayout_.buildColumnOutputPositions.size(); ++i) {
    outCols[outputLayout_.buildColumnOutputPositions[i]] =
        std::move(allCols[numProbeCols + outputLayout_.buildColumnIndices[i]]);
  }

  recordReadCompletion(stream);
  return std::make_unique<cudf::table>(std::move(outCols));
}

std::unique_ptr<cudf::table> CudfNestedLoopJoinProbe::emitProbeMismatchRows(
    cudf::table_view probeTableView,
    rmm::cuda_stream_view stream) {
  cudf::size_type numUnmatched;
  std::unique_ptr<cudf::table> unmatchedProbe;

  if (!probeMatchedFlags_) {
    // No flags means all probe rows are unmatched (empty build case).
    numUnmatched = static_cast<cudf::size_type>(probeTableView.num_rows());
    if (!outputLayout_.probeColumnIndices.empty()) {
      auto probeGatherView =
          probeTableView.select(outputLayout_.probeColumnIndices);
      unmatchedProbe = std::make_unique<cudf::table>(
          probeGatherView, stream, get_output_mr());
    }
  } else {
    auto matchedMask = probeMatchedFlags_->view();
    if (!outputLayout_.probeColumnIndices.empty()) {
      auto probeGatherView =
          probeTableView.select(outputLayout_.probeColumnIndices);
      unmatchedProbe = cudf::apply_deletion_mask(
          probeGatherView, matchedMask, stream, get_output_mr());
      numUnmatched = static_cast<cudf::size_type>(unmatchedProbe->num_rows());
    } else {
      // No probe columns in output — count unmatched rows from the mask.
      auto countTable = cudf::apply_deletion_mask(
          cudf::table_view{{matchedMask}}, matchedMask, stream, get_temp_mr());
      numUnmatched = static_cast<cudf::size_type>(countTable->num_rows());
    }
  }

  if (numUnmatched == 0) {
    return nullptr;
  }

  auto numOutputColumns = outputType_->size();
  std::vector<std::unique_ptr<cudf::column>> outCols(numOutputColumns);

  // Place unmatched probe columns at their output positions.
  if (unmatchedProbe) {
    auto probeCols = unmatchedProbe->release();
    for (size_t i = 0; i < outputLayout_.probeColumnOutputPositions.size();
         ++i) {
      outCols[outputLayout_.probeColumnOutputPositions[i]] =
          std::move(probeCols[i]);
    }
  }

  // Create all-null columns for the build side.
  for (size_t i = 0; i < outputLayout_.buildColumnOutputPositions.size(); ++i) {
    auto outIdx = outputLayout_.buildColumnOutputPositions[i];
    auto buildChannel = outputLayout_.buildColumnIndices[i];
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
  // The caller in doGetOutput() already waits for the build-ready event on
  // this stream before calling in here.
  auto& buildTable = buildData_.value();
  auto numOutputColumns = outputType_->size();

  // Select unmatched build rows (deletion mask removes matched rows).
  auto matchedMask = buildMatchedFlags_->view();
  cudf::size_type numUnmatched;
  std::unique_ptr<cudf::table> unmatchedBuild;
  if (!outputLayout_.buildColumnIndices.empty()) {
    auto buildGatherView =
        buildTable->view().select(outputLayout_.buildColumnIndices);
    unmatchedBuild = cudf::apply_deletion_mask(
        buildGatherView, matchedMask, stream, get_output_mr());
    numUnmatched = static_cast<cudf::size_type>(unmatchedBuild->num_rows());
  } else {
    // No build columns in output — count unmatched rows from the mask.
    auto countTable = cudf::apply_deletion_mask(
        cudf::table_view{{matchedMask}}, matchedMask, stream, get_temp_mr());
    numUnmatched = static_cast<cudf::size_type>(countTable->num_rows());
  }

  finished_ = true;
  if (numUnmatched == 0) {
    recordReadCompletion(stream);
    return nullptr;
  }

  std::vector<std::unique_ptr<cudf::column>> outCols(numOutputColumns);

  // Create all-null columns for the probe side.
  for (size_t li = 0; li < outputLayout_.probeColumnOutputPositions.size();
       ++li) {
    auto outIdx = outputLayout_.probeColumnOutputPositions[li];
    auto probeChannel = outputLayout_.probeColumnIndices[li];
    auto probeCudfDataType =
        veloxToCudfDataType(probeType_->childAt(probeChannel));
    auto nullScalar = cudf::make_default_constructed_scalar(
        probeCudfDataType, stream, get_temp_mr());
    outCols[outIdx] = cudf::make_column_from_scalar(
        *nullScalar, numUnmatched, stream, get_output_mr());
  }

  // Place unmatched build columns at their output positions.
  if (unmatchedBuild) {
    auto buildCols = unmatchedBuild->release();
    for (size_t ri = 0; ri < outputLayout_.buildColumnOutputPositions.size();
         ++ri) {
      outCols[outputLayout_.buildColumnOutputPositions[ri]] =
          std::move(buildCols[ri]);
    }
  }

  auto out = std::make_unique<cudf::table>(std::move(outCols));
  auto size = static_cast<vector_size_t>(out->num_rows());
  recordReadCompletion(stream);
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
      // Fresh pool stream - must wait for the build-ready event before
      // emitBuildMismatchRows() reads buildData_.
      waitForBuildReady(stream);
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
  // Wait once here for the rest of doGetOutput(): the LeftSemiProject path
  // below and joinWithBuildBatch() further down both read buildData_ on
  // this same stream.
  waitForBuildReady(stream);

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
            makeExtendedTableView(probeTableView, leftPrecomputed);
      }
      const cudf::table_view& extendedBuildView = buildPrecomputed_.empty()
          ? buildData_.value()->view()
          : buildExtendedView_;

      auto matchedIndices = cudf::conditional_left_semi_join(
          extendedProbeView,
          extendedBuildView,
          tree_.back(),
          {},
          stream,
          get_temp_mr());

      if (matchedIndices->size() > 0) {
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

        auto matchedIndicesView = cudf::column_view(
            cudf::data_type{cudf::type_to_id<cudf::size_type>()},
            matchedIndices->size(),
            matchedIndices->data(),
            nullptr,
            0);

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
    auto probeGatherView =
        probeTableView.select(outputLayout_.probeColumnIndices);
    auto gatheredProbe =
        std::make_unique<cudf::table>(probeGatherView, stream, get_output_mr());
    auto probeCols = gatheredProbe->release();

    auto numOutputColumns = outputType_->size();
    std::vector<std::unique_ptr<cudf::column>> outCols(numOutputColumns);
    for (size_t i = 0; i < outputLayout_.probeColumnOutputPositions.size();
         ++i) {
      outCols[outputLayout_.probeColumnOutputPositions[i]] =
          std::move(probeCols[i]);
    }
    outCols[numOutputColumns - 1] = std::move(outputMatchFlags);

    auto result = std::make_unique<cudf::table>(std::move(outCols));
    input_.reset();

    // Unconditional even though buildData_/buildPrecomputed_ are only
    // actually read in the hasFilter_ branch above - a harmless no-op link
    // in the other case, and much simpler than conditionally tracking it.
    recordReadCompletion(stream);
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
