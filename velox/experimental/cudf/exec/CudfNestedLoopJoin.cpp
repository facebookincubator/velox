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
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/PrecomputeInstruction.h"

#include "velox/exec/Task.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/join.hpp>

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
    : exec::Operator(
          driverCtx,
          nullptr,
          operatorId,
          joinNode->id(),
          "CudfNestedLoopJoinBuild"),
      NvtxHelper(
          nvtx3::rgb{65, 105, 225}, // Royal Blue
          operatorId,
          fmt::format("[{}]", joinNode->id())),
      joinNode_(joinNode) {}

// Accumulates input batches in memory.
// All batches are kept as CudfVectors (GPU memory) until join completes.
void CudfNestedLoopJoinBuild::addInput(RowVectorPtr input) {
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    inputs_.push_back(std::move(cudfInput)); // Store in GPU memory
  }
}

bool CudfNestedLoopJoinBuild::needsInput() const {
  return !noMoreInput_;
}

RowVectorPtr CudfNestedLoopJoinBuild::getOutput() {
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
void CudfNestedLoopJoinBuild::noMoreInput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
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

  // Concatenate all input batches into cuDF tables
  auto stream = cudfGlobalStreamPool().get_stream();
  auto tables = getConcatenatedTableBatched(
      std::exchange(inputs_, {}),
      joinNode_->sources()[1]->outputType(),
      stream,
      get_output_mr());

  stream.synchronize(); // Ensure GPU operations complete

  std::vector<std::shared_ptr<cudf::table>> sharedTables;
  sharedTables.reserve(tables.size());
  for (auto& tbl : tables) {
    sharedTables.push_back(std::move(tbl));
  }

  // Transfer build data to bridge - this will unblock probe operators
  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto bridge = std::dynamic_pointer_cast<CudfNestedLoopJoinBridge>(joinBridge);

  bridge->setBuildStream(stream); // Pass stream for CUDA synchronization
  bridge->setData(std::make_optional(std::move(sharedTables))); // Wake probes
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

// ============================================================================
// Probe Operator Implementation
// ============================================================================
// Performs the actual nested loop join by combining probe batches with
// build data using cuDF's cross_join or conditional_inner_join APIs.

CudfNestedLoopJoinProbe::CudfNestedLoopJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::NestedLoopJoinNode> joinNode)
    : exec::Operator(
          driverCtx,
          joinNode->outputType(),
          operatorId,
          joinNode->id(),
          "CudfNestedLoopJoinProbe"),
      NvtxHelper(
          nvtx3::rgb{0, 128, 128}, // Teal
          operatorId,
          fmt::format("[{}]", joinNode->id())),
      joinNode_(joinNode) {
  // Resolve output column order by name. The output type may interleave
  // probe and build columns in any order (e.g., {"b0", "p0", "p1"}).
  auto probeType = joinNode_->sources()[0]->outputType();
  auto buildType = joinNode_->sources()[1]->outputType();

  for (size_t i = 0; i < outputType_->size(); ++i) {
    const auto& name = outputType_->nameOf(i);
    auto probeIdx = probeType->getChildIdxIfExists(name);
    if (probeIdx.has_value()) {
      probeColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(probeIdx.value()));
      probeColumnOutputIndices_.push_back(i);
      continue;
    }
    auto buildIdx = buildType->getChildIdxIfExists(name);
    if (buildIdx.has_value()) {
      buildColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(buildIdx.value()));
      buildColumnOutputIndices_.push_back(i);
      continue;
    }
    VELOX_FAIL("Output column not found in probe or build types: {}", name);
  }

  // Setup AST filter if join condition exists
  if (joinNode_->joinCondition()) {
    hasFilter_ = true;
    exec::ExprSet exprs({joinNode_->joinCondition()}, operatorCtx_->execCtx());
    VELOX_CHECK_EQ(exprs.exprs().size(), 1);

    // Convert Velox expression to cuDF AST expression tree.
    // The AST will be passed to cudf::conditional_inner_join() for GPU
    // evaluation.
    std::vector<PrecomputeInstruction> leftPrecomputeInstructions;
    std::vector<PrecomputeInstruction> rightPrecomputeInstructions;

    createAstTree(
        exprs.exprs()[0],
        tree_,
        scalars_,
        probeType,
        buildType,
        leftPrecomputeInstructions,
        rightPrecomputeInstructions);

    // Precompute instructions handle expressions not supported by cuDF AST
    // (e.g., complex functions). Not implemented yet for nested loop join.
    if (!leftPrecomputeInstructions.empty() ||
        !rightPrecomputeInstructions.empty()) {
      VELOX_NYI(
          "Filters that require precomputation are not yet supported for NestedLoopJoin");
    }
  }
}

void CudfNestedLoopJoinProbe::close() {
  Operator::close();
  buildData_.reset();
  scalars_.clear();
  tree_ = {};
}

bool CudfNestedLoopJoinProbe::needsInput() const {
  return !noMoreInput_ && !finished_ && input_ == nullptr;
}

void CudfNestedLoopJoinProbe::addInput(RowVectorPtr input) {
  VELOX_CHECK_NULL(input_, "Probe input already set");
  input_ = input;
  buildBatchIdx_ = 0;
}

void CudfNestedLoopJoinProbe::noMoreInput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  Operator::noMoreInput();
}

bool CudfNestedLoopJoinProbe::isFinished() {
  return finished_;
}

exec::BlockingReason CudfNestedLoopJoinProbe::isBlocked(
    ContinueFuture* future) {
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

  // If the entire build side is empty, no probe batch can produce output.
  bool hasBuildRows = false;
  for (const auto& table : buildData_.value()) {
    if (table->num_rows() > 0) {
      hasBuildRows = true;
      break;
    }
  }
  if (!hasBuildRows) {
    finished_ = true;
  }

  return exec::BlockingReason::kNotBlocked;
}

// Performs a cross join between one probe batch and one build batch.
// PATH 1 (no filter): cudf::cross_join() for direct cartesian product.
// PATH 2 (with filter): cudf::conditional_inner_join() evaluates the AST on
// GPU and returns only matching row indices, then gathers actual data.
std::unique_ptr<cudf::table> CudfNestedLoopJoinProbe::crossJoinWithBuildBatch(
    cudf::table_view probeTableView,
    cudf::table_view buildView,
    rmm::cuda_stream_view stream) {
  VELOX_NVTX_FUNC_RANGE();

  // Ensure build data is visible on the probe stream before reading it.
  // Record on the build stream (where data was produced), then make the probe
  // stream wait. Only needed once since build data doesn't change afterward.
  if (buildStream_.has_value()) {
    if (!cudaEvent_) {
      cudaEvent_ = std::make_unique<CudaEvent>();
    }
    cudaEvent_->recordFrom(buildStream_.value()).waitOn(stream);
    buildStream_.reset();
  }

  auto numOutputColumns = outputType_->size();

  if (hasFilter_) {
    // PATH 2: Filtered join using conditional_inner_join.
    // The AST references columns by position in the full probe/build views.
    auto [leftIndices, rightIndices] = cudf::conditional_inner_join(
        probeTableView,
        buildView,
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

    // Gather only the columns needed for output
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

    // Place columns at the positions specified by the output type
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

  // PATH 1: Unfiltered join using cross_join.
  // Guard against cudf::size_type (int32) overflow: the cross product produces
  // probe_rows * build_rows output rows.
  auto outputRows = static_cast<int64_t>(probeTableView.num_rows()) *
      static_cast<int64_t>(buildView.num_rows());
  VELOX_CHECK_LE(
      outputRows,
      std::numeric_limits<cudf::size_type>::max(),
      "Cross join output exceeds cudf::size_type limit: {} x {} = {} rows",
      probeTableView.num_rows(),
      buildView.num_rows(),
      outputRows);

  // cross_join returns [probe_cols..., build_cols...] - reorder to match
  // the output type which may have a different column order.
  auto crossResult =
      cudf::cross_join(probeTableView, buildView, stream, get_output_mr());

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

RowVectorPtr CudfNestedLoopJoinProbe::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (!input_) {
    if (noMoreInput_) {
      finished_ = true;
    }
    return nullptr;
  }

  VELOX_CHECK(buildData_.has_value(), "Build data not available in getOutput");
  auto& buildTables = buildData_.value();

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput);
  auto stream = cudfInput->stream();

  // Iterate through build batches until we produce a non-empty result or
  // exhaust all batches. This avoids returning nullptr to the driver loop
  // when there are still build batches to process.
  while (buildBatchIdx_ < buildTables.size()) {
    if (buildTables[buildBatchIdx_]->num_rows() == 0) {
      ++buildBatchIdx_;
      continue;
    }

    auto result = crossJoinWithBuildBatch(
        cudfInput->getTableView(), buildTables[buildBatchIdx_]->view(), stream);
    ++buildBatchIdx_;

    if (result->num_rows() > 0) {
      // Release probe input if all build batches have been processed.
      if (buildBatchIdx_ >= buildTables.size()) {
        input_.reset();
        buildBatchIdx_ = 0;
      }

      auto size = static_cast<vector_size_t>(result->num_rows());
      return std::make_shared<CudfVector>(
          operatorCtx_->pool(), outputType_, size, std::move(result), stream);
    }
  }

  // All build batches exhausted for this probe input.
  input_.reset();
  buildBatchIdx_ = 0;
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
