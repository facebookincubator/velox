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
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Task.h" // NOLINT(misc-unused-headers)
#include "velox/type/TypeUtil.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <nvtx3/nvtx3.hpp>

namespace facebook::velox::cudf_velox {

namespace {

/// Creates extended table view by appending precomputed columns
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

void CudfHashJoinProbe::doClose() {
  Operator::close();
  filterEvaluator_.reset();
  scalars_.clear();
  tree_ = {};
}

void CudfHashJoinBridge::setHashTable(
    std::optional<CudfHashJoinBridge::hash_type> hashObject) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::setHashTable";
  }
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(
        !hashObject_.has_value(),
        "CudfHashJoinBridge already has a hash table");
    hashObject_ = std::move(hashObject);
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

std::optional<CudfHashJoinBridge::hash_type> CudfHashJoinBridge::hashOrFuture(
    ContinueFuture* future) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture";
  }
  std::lock_guard<std::mutex> l(mutex_);
  if (hashObject_.has_value()) {
    return hashObject_;
  }
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture constructing promise";
  }
  promises_.emplace_back("CudfHashJoinBridge::hashOrFuture");
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture getSemiFuture";
  }
  *future = promises_.back().getSemiFuture();
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture returning nullopt";
  }
  return std::nullopt;
}

void CudfHashJoinBridge::setBuildStream(rmm::cuda_stream_view buildStream) {
  std::lock_guard<std::mutex> l(mutex_);
  buildStream_ = buildStream;
}

std::optional<rmm::cuda_stream_view> CudfHashJoinBridge::getBuildStream() {
  std::lock_guard<std::mutex> l(mutex_);
  return buildStream_;
}

CudfHashJoinBuild::CudfHashJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    // TODO check outputType should be set or not?
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          nullptr, // outputType
          joinNode->id(),
          "CudfHashJoinBuild",
          nvtx3::rgb{65, 105, 225}, // Royal Blue
          NvtxMethodFlag::kAll,
          std::nullopt, // spillConfig
          joinNode),
      joinNode_(joinNode) {}

void CudfHashJoinBuild::doAddInput(RowVectorPtr input) {
  // Queue inputs, process all at once.
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    // Count nulls in join key columns
    auto [_, null_count] = cudf::bitmask_and(
        cudfInput->getTableView(), cudfInput->stream(), get_temp_mr());
    {
      // Update statistics for null keys in join operator.
      auto lockedStats = stats_.wlock();
      lockedStats->numNullKeys += null_count;
    }
    inputs_.push_back(std::move(cudfInput));
  }
}

bool CudfHashJoinBuild::needsInput() const {
  return !noMoreInput_;
}

RowVectorPtr CudfHashJoinBuild::doGetOutput() {
  return nullptr;
}

void CudfHashJoinBuild::doNoMoreInput() {
  Operator::noMoreInput();
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  // Only last driver collects all answers
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }
  // Collect results from peers
  for (auto& peer : peers) {
    auto op = peer->findOperator(planNodeId());
    auto* build = dynamic_cast<CudfHashJoinBuild*>(op);
    VELOX_CHECK_NOT_NULL(build);
    inputs_.insert(inputs_.end(), build->inputs_.begin(), build->inputs_.end());
  }

  SCOPE_EXIT {
    // Realize the promises so that the other Drivers (which were not
    // the last to finish) can continue from the barrier and finish.
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "CudfHashJoinBuild: build batches";
    VLOG(1) << "Build batches number of columns: "
            << inputs_[0]->getTableView().num_columns();
    for (auto i = 0; i < inputs_.size(); i++) {
      VLOG(1) << "Build batch " << i
              << ": number of rows: " << inputs_[i]->getTableView().num_rows();
    }
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  // Using output_mr here to allow spilling queued up large tables
  auto tbls = getConcatenatedTableBatched(
      std::exchange(inputs_, {}),
      joinNode_->sources()[1]->outputType(),
      stream,
      get_output_mr());

  for (auto const& tbl : tbls) {
    VELOX_CHECK_NOT_NULL(tbl);
  }
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Build table number of columns: " << tbls[0]->num_columns();
    for (auto i = 0; i < tbls.size(); i++) {
      VLOG(1) << "Build table " << i
              << ": number of rows: " << tbls[i]->num_rows();
    }
  }

  auto rightKeys = joinNode_->rightKeys();

  auto buildKeyIndices = std::vector<cudf::size_type>(rightKeys.size());
  auto buildType = joinNode_->sources()[1]->outputType();
  for (size_t i = 0; i < buildKeyIndices.size(); i++) {
    buildKeyIndices[i] = static_cast<cudf::size_type>(
        buildType->getChildIdx(rightKeys[i]->name()));
  }

  // Construct hash_join object for join types that use hb->inner_join() or
  // hb->left_join(). Semi filter and anti joins use standalone cudf functions
  // (e.g., mixed_left_semi_join, filtered_join) that build hash tables
  // internally, so they don't need this.
  bool buildHashJoin =
      (joinNode_->isInnerJoin() || joinNode_->isLeftJoin() ||
       joinNode_->isRightJoin() || joinNode_->isFullJoin() ||
       joinNode_->isLeftSemiProjectJoin());

  std::vector<std::shared_ptr<cudf::hash_join>> hashObjects;
  for (auto i = 0; i < tbls.size(); i++) {
    hashObjects.push_back(
        (buildHashJoin) ? std::make_shared<cudf::hash_join>(
                              tbls[i]->view().select(buildKeyIndices),
                              cudf::null_equality::UNEQUAL,
                              stream)
                        : nullptr);
    if (buildHashJoin) {
      VELOX_CHECK_NOT_NULL(hashObjects.back());
    }
    if (CudfConfig::getInstance().debugEnabled) {
      if (hashObjects.back() != nullptr) {
        VLOG(2) << "hashObject " << i << " is not nullptr "
                << hashObjects.back().get() << "\n";
      } else {
        VLOG(2) << "hashObject " << i << " is *** nullptr\n";
      }
    }
  }

  std::vector<std::shared_ptr<cudf::table>> shared_tbls;
  for (auto& tbl : tbls) {
    shared_tbls.push_back(std::move(tbl));
  }
  // set hash table to CudfHashJoinBridge
  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudfHashJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);

  cudfHashJoinBridge->setBuildStream(stream);
  cudfHashJoinBridge->setHashTable(
      std::make_optional(
          std::make_pair(std::move(shared_tbls), std::move(hashObjects))));
}

exec::BlockingReason CudfHashJoinBuild::isBlocked(ContinueFuture* future) {
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfHashJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

CudfHashJoinProbe::CudfHashJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          joinNode->outputType(),
          joinNode->id(),
          "CudfHashJoinProbe",
          nvtx3::rgb{0, 128, 128}, // Teal
          NvtxMethodFlag::kAll,
          std::nullopt, // spillConfig
          joinNode),
      joinNode_(joinNode),
      probeType_(joinNode_->sources()[0]->outputType()),
      buildType_(joinNode_->sources()[1]->outputType()),
      cudaEvent_(std::make_unique<CudaEvent>(cudaEventDisableTiming)) {
  auto const& leftKeys = joinNode_->leftKeys(); // probe keys
  auto const& rightKeys = joinNode_->rightKeys(); // build keys

  if (CudfConfig::getInstance().debugEnabled) {
    for (int i = 0; i < probeType_->names().size(); i++) {
      VLOG(1) << "Left column " << i << ": " << probeType_->names()[i];
    }

    for (int i = 0; i < buildType_->names().size(); i++) {
      VLOG(1) << "Right column " << i << ": " << buildType_->names()[i];
    }

    for (int i = 0; i < leftKeys.size(); i++) {
      VLOG(1) << "Left key " << i << ": " << leftKeys[i]->name() << " "
              << leftKeys[i]->type()->kind();
    }

    for (int i = 0; i < rightKeys.size(); i++) {
      VLOG(1) << "Right key " << i << ": " << rightKeys[i]->name() << " "
              << rightKeys[i]->type()->kind();
    }
  }

  auto const probeTableNumColumns = probeType_->size();
  leftKeyIndices_ = std::vector<cudf::size_type>(leftKeys.size());
  for (size_t i = 0; i < leftKeyIndices_.size(); i++) {
    leftKeyIndices_[i] = static_cast<cudf::size_type>(
        probeType_->getChildIdx(leftKeys[i]->name()));
    VELOX_CHECK_LT(leftKeyIndices_[i], probeTableNumColumns);
  }
  auto const buildTableNumColumns = buildType_->size();
  rightKeyIndices_ = std::vector<cudf::size_type>(rightKeys.size());
  for (size_t i = 0; i < rightKeyIndices_.size(); i++) {
    rightKeyIndices_[i] = static_cast<cudf::size_type>(
        buildType_->getChildIdx(rightKeys[i]->name()));
    VELOX_CHECK_LT(rightKeyIndices_[i], buildTableNumColumns);
  }

  auto outputType = joinNode_->outputType();
  leftColumnIndicesToGather_ = std::vector<cudf::size_type>();
  rightColumnIndicesToGather_ = std::vector<cudf::size_type>();
  leftColumnOutputIndices_ = std::vector<size_t>();
  rightColumnOutputIndices_ = std::vector<size_t>();
  for (int i = 0; i < outputType->names().size(); i++) {
    auto const outputName = outputType->names()[i];
    if (CudfConfig::getInstance().debugEnabled) {
      VLOG(1) << "Output column " << i << ": " << outputName;
    }
    auto channel = probeType_->getChildIdxIfExists(outputName);
    if (channel.has_value()) {
      leftColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      leftColumnOutputIndices_.push_back(i);
      continue;
    }
    channel = buildType_->getChildIdxIfExists(outputName);
    if (channel.has_value()) {
      rightColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      rightColumnOutputIndices_.push_back(i);
      continue;
    }
    // For LEFT SEMI PROJECT, the last column is the boolean "match" column
    // which is not in probe or build types - skip it here, handled separately
    if (isLeftSemiProjectJoin(joinNode_->joinType()) &&
        i == outputType->size() - 1 &&
        outputType->childAt(i)->kind() == TypeKind::BOOLEAN) {
      continue;
    }
    VELOX_FAIL(
        "Join field {} not in probe or build input", outputType->children()[i]);
  }

  if (CudfConfig::getInstance().debugEnabled) {
    for (int i = 0; i < leftColumnIndicesToGather_.size(); i++) {
      VLOG(1) << "Left index to gather " << i << ": "
              << leftColumnIndicesToGather_[i];
    }

    for (int i = 0; i < rightColumnIndicesToGather_.size(); i++) {
      VLOG(1) << "Right index to gather " << i << ": "
              << rightColumnIndicesToGather_[i];
    }
  }

  // Setup filter in case it exists
  if (joinNode_->filter()) {
    // simplify expression
    exec::ExprSet exprs({joinNode_->filter()}, operatorCtx_->execCtx());
    VELOX_CHECK_EQ(exprs.exprs().size(), 1);

    // Create a reusable evaluator for the filter column. This is expensive to
    // build, and the expression + input schema are stable for the lifetime of
    // the operator instance.
    std::vector<velox::RowTypePtr> filterRowTypes{probeType_, buildType_};
    filterEvaluator_ = createCudfExpression(
        exprs.exprs()[0],
        facebook::velox::type::concatRowTypes(filterRowTypes));

    // We don't need to get tables that contain conditional comparison columns
    // We'll pass the entire table. The ast will handle finding the required
    // columns. This is required because we build the ast with whole row schema
    // and the column locations in that schema translate to column locations
    // in whole tables

    // create ast tree
    if (joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin()) {
      createAstTree(
          exprs.exprs()[0],
          tree_,
          scalars_,
          buildType_,
          probeType_,
          rightPrecomputeInstructions_,
          leftPrecomputeInstructions_);
    } else {
      createAstTree(
          exprs.exprs()[0],
          tree_,
          scalars_,
          probeType_,
          buildType_,
          leftPrecomputeInstructions_,
          rightPrecomputeInstructions_);
    }
  }
}

bool CudfHashJoinProbe::needsInput() const {
  if (joinNode_->isRightSemiFilterJoin()) {
    return !noMoreInput_;
  }
  return !noMoreInput_ && !finished_ && input_ == nullptr;
}

void CudfHashJoinProbe::doAddInput(RowVectorPtr input) {
  if (skipInput_) {
    VELOX_CHECK_NULL(input_);
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  // Count nulls in join key columns
  auto [_, null_count] = cudf::bitmask_and(
      cudfInput->getTableView(), cudfInput->stream(), get_temp_mr());
  {
    // Update statistics for null keys in join operator.
    auto lockedStats = stats_.wlock();
    lockedStats->numNullKeys += null_count;
  }
  if (joinNode_->isRightSemiFilterJoin()) {
    // Queue inputs and process all at once
    if (input->size() > 0) {
      inputs_.push_back(std::move(cudfInput));
    }
    return;
  }

  if (input->size() > 0) {
    input_ = std::move(input);
  }
}

void CudfHashJoinProbe::doNoMoreInput() {
  Operator::noMoreInput();
  if (!joinNode_->isRightJoin() && !joinNode_->isRightSemiFilterJoin() &&
      !joinNode_->isFullJoin()) {
    return;
  }
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  // Only last driver collects all answers
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }

  SCOPE_EXIT {
    // Realize the promises so that the other Drivers (which were not
    // the last to finish) can continue from the barrier and finish.
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  if (joinNode_->isRightJoin() || joinNode_->isFullJoin()) {
    isLastDriver_ = true;
    if (hashObject_.has_value()) {
      auto stream = cudfGlobalStreamPool().get_stream();

      // The allPeersFinished barrier above synchronizes CPU threads, but not
      // GPU streams. A driver's CPU thread may return from getOutput() while
      // its GPU work (updating rightMatchedFlags_) is still in flight.
      // join_streams establishes GPU-side ordering so that all probe stream
      // operations complete before the BITWISE_OR reads below.
      // Drivers without lastProbeStream_ (no probe batches) are skipped:
      // their flags are all-false from host-synchronized init with no pending
      // GPU work.
      std::vector<rmm::cuda_stream_view> inputStreams;
      if (lastProbeStream_.has_value()) {
        inputStreams.push_back(lastProbeStream_.value());
      }
      for (auto& peer : peers) {
        if (peer.get() == operatorCtx_->driver()) {
          continue;
        }
        auto op = peer->findOperator(planNodeId());
        auto* probe = dynamic_cast<CudfHashJoinProbe*>(op);
        if (probe != nullptr && probe->lastProbeStream_.has_value()) {
          inputStreams.push_back(probe->lastProbeStream_.value());
        }
      }
      if (!inputStreams.empty()) {
        cudf::detail::join_streams(inputStreams, stream);
      }

      for (auto& peer : peers) {
        if (peer.get() == operatorCtx_->driver()) {
          continue;
        }
        auto op = peer->findOperator(planNodeId());
        auto* probe = dynamic_cast<CudfHashJoinProbe*>(op);
        if (probe == nullptr) {
          continue;
        }
        // Combine flags per partition using cuDF bitwise OR
        // DM: This needs a relook. This is for when build side exceeds cudf
        // size_type limits. In case of multiple right side chunks, I'm not sure
        // if partitions to combine are in the same place p
        for (size_t p = 0; p < rightMatchedFlags_.size(); ++p) {
          auto or_result = cudf::binary_operation(
              rightMatchedFlags_[p]->view(),
              probe->rightMatchedFlags_[p]->view(),
              cudf::binary_operator::BITWISE_OR,
              cudf::data_type{cudf::type_id::BOOL8},
              stream,
              get_temp_mr());
          // binary_operation is async on `stream`; the old column destructs via
          // cudaFreeAsync on its allocation stream (not `stream`), so the free
          // can race the kernel. Drain `stream` before the move-assign.
          stream.synchronize();
          rightMatchedFlags_[p] = std::move(or_result);
        }
      }
      stream.synchronize();
    }
    return;
  }

  // Handling RightSemiFilterJoin
  // Collect results from peers
  for (auto& peer : peers) {
    auto op = peer->findOperator(planNodeId());
    auto* probe = dynamic_cast<CudfHashJoinProbe*>(op);
    VELOX_CHECK_NOT_NULL(probe);
    inputs_.insert(inputs_.end(), probe->inputs_.begin(), probe->inputs_.end());
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  // Using output_mr here to allow spilling queued up large tables
  auto tbl = getConcatenatedTable(
      std::exchange(inputs_, {}),
      joinNode_->sources()[1]->outputType(),
      stream,
      get_output_mr());

  VELOX_CHECK_NOT_NULL(tbl);

  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Probe table number of columns: " << tbl->num_columns();
    VLOG(1) << "Probe table number of rows: " << tbl->num_rows();
  }

  // Store the concatenated table in input_
  input_ = std::make_shared<CudfVector>(
      operatorCtx_->pool(),
      joinNode_->outputType(),
      tbl->num_rows(),
      std::move(tbl),
      stream);
}

std::unique_ptr<cudf::table> CudfHashJoinProbe::unfilteredOutput(
    cudf::table_view leftTableView,
    cudf::column_view leftIndicesCol,
    cudf::table_view rightTableView,
    cudf::column_view rightIndicesCol,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> joinedCols;
  auto leftInput = leftTableView.select(leftColumnIndicesToGather_);
  auto rightInput = rightTableView.select(rightColumnIndicesToGather_);
  auto leftResult = cudf::gather(
      leftInput, leftIndicesCol, oobPolicy, stream, get_output_mr());
  auto rightResult = cudf::gather(
      rightInput, rightIndicesCol, oobPolicy, stream, get_output_mr());

  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Left result number of columns: " << leftResult->num_columns();
    VLOG(1) << "Right result number of columns: " << rightResult->num_columns();
  }

  auto leftCols = leftResult->release();
  auto rightCols = rightResult->release();
  joinedCols.resize(outputType_->names().size());
  for (int i = 0; i < leftColumnOutputIndices_.size(); i++) {
    joinedCols[leftColumnOutputIndices_[i]] = std::move(leftCols[i]);
  }
  for (int i = 0; i < rightColumnOutputIndices_.size(); i++) {
    joinedCols[rightColumnOutputIndices_[i]] = std::move(rightCols[i]);
  }
  if (buildStream_.has_value()) {
    // Ensure deallocation of build table happens after probe gathers
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(joinedCols));
}

std::unique_ptr<cudf::table> CudfHashJoinProbe::filteredOutput(
    cudf::table_view leftTableView,
    cudf::column_view leftIndicesCol,
    cudf::table_view rightTableView,
    cudf::column_view rightIndicesCol,
    std::function<std::vector<std::unique_ptr<cudf::column>>(
        std::vector<std::unique_ptr<cudf::column>>&&,
        cudf::column_view)> func,
    rmm::cuda_stream_view stream) {
  auto leftResult = cudf::gather(
      leftTableView, leftIndicesCol, oobPolicy, stream, get_output_mr());
  auto rightResult = cudf::gather(
      rightTableView, rightIndicesCol, oobPolicy, stream, get_output_mr());
  auto leftColsSize = leftResult->num_columns();
  auto rightColsSize = rightResult->num_columns();

  std::vector<std::unique_ptr<cudf::column>> joinedCols = leftResult->release();
  auto rightCols = rightResult->release();
  joinedCols.insert(
      joinedCols.end(),
      std::make_move_iterator(rightCols.begin()),
      std::make_move_iterator(rightCols.end()));

  VELOX_CHECK_NOT_NULL(
      filterEvaluator_,
      "Join filter evaluator must be initialized before filteredOutput()");
  std::vector<cudf::column_view> joinedColViews;
  joinedColViews.reserve(joinedCols.size());
  for (const auto& col : joinedCols) {
    joinedColViews.push_back(col->view());
  }
  auto filterColumns =
      filterEvaluator_->eval(joinedColViews, stream, get_output_mr());
  auto filterColumn = asView(filterColumns);

  joinedCols = func(std::move(joinedCols), filterColumn);

  auto filteredjoinedCols =
      std::vector<std::unique_ptr<cudf::column>>(outputType_->names().size());
  for (int i = 0; i < leftColumnOutputIndices_.size(); i++) {
    filteredjoinedCols[leftColumnOutputIndices_[i]] =
        std::move(joinedCols[leftColumnIndicesToGather_[i]]);
  }
  for (int i = 0; i < rightColumnOutputIndices_.size(); i++) {
    filteredjoinedCols[rightColumnOutputIndices_[i]] =
        std::move(joinedCols[leftColsSize + rightColumnIndicesToGather_[i]]);
  }
  joinedCols = std::move(filteredjoinedCols);
  if (buildStream_.has_value()) {
    // Ensure any deallocation of join indices is ordered wrt probe gathers
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(joinedCols));
}

std::unique_ptr<cudf::table> CudfHashJoinProbe::filteredOutputIndices(
    cudf::table_view leftTableView,
    cudf::column_view leftIndicesCol,
    cudf::table_view rightTableView,
    cudf::column_view rightIndicesCol,
    cudf::table_view extendedLeftView,
    cudf::table_view extendedRightView,
    cudf::join_kind joinKind,
    rmm::cuda_stream_view stream) {
  // Use extended views (with precomputed columns) for filter evaluation
  auto [filteredLeftJoinIndices, filteredRightJoinIndices] =
      cudf::filter_join_indices(
          extendedLeftView,
          extendedRightView,
          leftIndicesCol,
          rightIndicesCol,
          tree_.back(),
          joinKind,
          stream,
          get_temp_mr());

  auto filteredLeftIndicesSpan =
      cudf::device_span<cudf::size_type const>{*filteredLeftJoinIndices};
  auto filteredRightIndicesSpan =
      cudf::device_span<cudf::size_type const>{*filteredRightJoinIndices};
  auto filteredLeftIndicesCol = cudf::column_view{filteredLeftIndicesSpan};
  auto filteredRightIndicesCol = cudf::column_view{filteredRightIndicesSpan};
  // Use original views (without precomputed columns) for gathering output
  return unfilteredOutput(
      leftTableView,
      filteredLeftIndicesCol,
      rightTableView,
      filteredRightIndicesCol,
      stream);
}

std::vector<std::unique_ptr<cudf::table>> CudfHashJoinProbe::innerJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;

  // Precompute left (probe) table columns if needed (once, outside loop)
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && !leftPrecomputeInstructions_.empty()) {
    auto leftColumnViews = tableViewToColumnViews(leftTableView);
    leftPrecomputed = precomputeSubexpressions(
        leftColumnViews,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedLeftView = createExtendedTableView(leftTableView, leftPrecomputed);
  }

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    // Use cached precomputed columns for right (build) table
    cudf::table_view extendedRightView =
        (joinNode_->filter() && !rightPrecomputeInstructions_.empty())
        ? cachedExtendedRightViews_[i]
        : rightTableView;

    // left = probe, right = build
    VELOX_CHECK_NOT_NULL(hb);
    if (buildStream_.has_value()) {
      // Make build stream wait for probe tables to become valid
      cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
    }
    auto [leftJoinIndices, rightJoinIndices] = hb->inner_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        buildStream_.has_value() ? buildStream_.value() : stream,
        get_temp_mr());
    if (buildStream_.has_value()) {
      // Make probe stream wait for join completion before using indices
      cudaEvent_->recordFrom(buildStream_.value()).waitOn(stream);
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
    std::vector<std::unique_ptr<cudf::column>> joinedCols;

    if (joinNode_->filter()) {
      cudfOutputs.push_back(filteredOutputIndices(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          extendedLeftView,
          extendedRightView,
          cudf::join_kind::INNER_JOIN,
          stream));
    } else {
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  }
  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>> CudfHashJoinProbe::leftJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;

  // Precompute left (probe) table columns if needed (once, outside loop)
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && !leftPrecomputeInstructions_.empty()) {
    auto leftColumnViews = tableViewToColumnViews(leftTableView);
    leftPrecomputed = precomputeSubexpressions(
        leftColumnViews,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedLeftView = createExtendedTableView(leftTableView, leftPrecomputed);
  }

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    // Use cached precomputed columns for right (build) table
    cudf::table_view extendedRightView =
        (joinNode_->filter() && !rightPrecomputeInstructions_.empty())
        ? cachedExtendedRightViews_[i]
        : rightTableView;

    VELOX_CHECK_NOT_NULL(hb);
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
    }
    auto [leftJoinIndices, rightJoinIndices] = hb->left_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        buildStream_.has_value() ? buildStream_.value() : stream,
        get_temp_mr());
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(buildStream_.value()).waitOn(stream);
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
    std::vector<std::unique_ptr<cudf::column>> joinedCols;

    if (joinNode_->filter()) {
      cudfOutputs.push_back(filteredOutputIndices(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          extendedLeftView,
          extendedRightView,
          cudf::join_kind::LEFT_JOIN,
          stream));
    } else {
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  }
  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>> CudfHashJoinProbe::rightJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    VELOX_CHECK_NOT_NULL(hb);
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
    }
    auto [leftJoinIndices, rightJoinIndices] = hb->inner_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        buildStream_.has_value() ? buildStream_.value() : stream,
        get_temp_mr());
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(buildStream_.value()).waitOn(stream);
    }
    if (!joinNode_->filter()) {
      // Mark matched build rows by checking which row indices appear in
      // rightJoinIndices. Use contains to avoid scatter with duplicate indices.
      auto rightIdxCol = cudf::column_view{
          cudf::device_span<cudf::size_type const>{*rightJoinIndices}};

      // Create sequence [0, 1, ..., n-1] for build table row indices
      auto n = rightTableView.num_rows();
      auto rowIndices = cudf::sequence(
          n,
          cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
          cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
          stream,
          get_temp_mr());

      // Check which build row indices are present in the join result
      auto matchedInBatch = cudf::contains(
          rightIdxCol, rowIndices->view(), stream, get_temp_mr());

      // OR with existing flags to accumulate matches across batches
      auto updatedFlags = cudf::binary_operation(
          rightMatchedFlags_[i]->view(),
          matchedInBatch->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          get_temp_mr());
      // binary_operation is async on `stream`; the old column destructs via
      // cudaFreeAsync on its allocation stream (not `stream`), so the free
      // can race the kernel. Drain `stream` before the move-assign.
      stream.synchronize();
      rightMatchedFlags_[i] = std::move(updatedFlags);
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
    std::vector<std::unique_ptr<cudf::column>> joinedCols;

    if (joinNode_->filter()) {
      auto& rightMatchedFlags = rightMatchedFlags_[i];
      auto numBuildRows = rightTableView.num_rows();
      auto filterFunc =
          [&rightMatchedFlags, rightIndicesSpan, numBuildRows, stream](
              std::vector<std::unique_ptr<cudf::column>>&& joinedCols,
              cudf::column_view filterColumn) {
            // apply the filter
            auto filterTable =
                std::make_unique<cudf::table>(std::move(joinedCols));
            auto filteredTable = cudf::apply_boolean_mask(
                *filterTable, filterColumn, stream, get_output_mr());
            joinedCols = filteredTable->release();

            // For streaming right join, after applying filter, we record
            // matched right indices filter rightJoinIndices with the same mask
            // to update matched flags
            auto rightIdxCol = cudf::column_view{rightIndicesSpan};
            auto filteredIdxTable = cudf::apply_boolean_mask(
                cudf::table_view{std::vector<cudf::column_view>{rightIdxCol}},
                filterColumn,
                stream,
                get_temp_mr());
            auto filteredCols = filteredIdxTable->release();
            auto filteredRightIdxCol = std::move(filteredCols[0]);

            // Use contains to check which build row indices passed the filter
            auto rowIndices = cudf::sequence(
                numBuildRows,
                cudf::numeric_scalar<cudf::size_type>(
                    0, true, stream, get_temp_mr()),
                cudf::numeric_scalar<cudf::size_type>(
                    1, true, stream, get_temp_mr()),
                stream,
                get_temp_mr());

            auto matchedInBatch = cudf::contains(
                filteredRightIdxCol->view(),
                rowIndices->view(),
                stream,
                get_temp_mr());

            // OR with existing flags to accumulate matches across batches
            auto updatedFlags = cudf::binary_operation(
                rightMatchedFlags->view(),
                matchedInBatch->view(),
                cudf::binary_operator::BITWISE_OR,
                cudf::data_type{cudf::type_id::BOOL8},
                stream,
                get_temp_mr());
            // binary_operation is async on `stream`; the old column destructs
            // via cudaFreeAsync on its allocation stream (not `stream`), so the
            // free can race the kernel. Drain `stream` before the move-assign.
            stream.synchronize();
            rightMatchedFlags = std::move(updatedFlags);
            return std::move(joinedCols);
          };
      cudfOutputs.push_back(filteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          filterFunc,
          stream));
    } else {
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  }
  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>> CudfHashJoinProbe::fullJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    VELOX_CHECK_NOT_NULL(hb);
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
    }
    // Use left_join to get all probe rows (matched + unmatched).
    // Track matched build rows in rightMatchedFlags_ for last driver to emit
    // unmatched build rows at the end.
    auto [leftJoinIndices, rightJoinIndices] = hb->left_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        buildStream_.has_value() ? buildStream_.value() : stream,
        get_temp_mr());
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(buildStream_.value()).waitOn(stream);
    }
    if (!joinNode_->filter()) {
      // Mark matched build rows by checking which row indices appear in
      // rightJoinIndices. Use contains to avoid scatter with duplicate indices.
      auto rightIdxCol = cudf::column_view{
          cudf::device_span<cudf::size_type const>{*rightJoinIndices}};

      // Create sequence [0, 1, ..., n-1] for build table row indices
      auto n = rightTableView.num_rows();
      auto rowIndices = cudf::sequence(
          n,
          cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
          cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
          stream,
          get_temp_mr());

      // Check which build row indices are present in the join result
      auto matchedInBatch = cudf::contains(
          rightIdxCol, rowIndices->view(), stream, get_temp_mr());

      // OR with existing flags to accumulate matches across batches
      auto updatedFlags = cudf::binary_operation(
          rightMatchedFlags_[i]->view(),
          matchedInBatch->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          get_temp_mr());
      // binary_operation is async on `stream`; the old column destructs via
      // cudaFreeAsync on its allocation stream (not `stream`), so the free
      // can race the kernel. Drain `stream` before the move-assign.
      stream.synchronize();
      rightMatchedFlags_[i] = std::move(updatedFlags);
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};

    if (joinNode_->filter()) {
      // Use filter_join_indices with LEFT_JOIN to get proper full join probe
      // semantics: all probe rows are kept, build columns are NULL when filter
      // fails or no match.
      auto [filteredLeftJoinIndices, filteredRightJoinIndices] =
          cudf::filter_join_indices(
              leftTableView,
              rightTableView,
              leftIndicesCol,
              rightIndicesCol,
              tree_.back(),
              cudf::join_kind::LEFT_JOIN,
              stream,
              get_temp_mr());

      // Track matched build rows for unmatched row emission at end.
      // Use contains to check which build row indices passed the filter.
      auto& rightMatchedFlags = rightMatchedFlags_[i];
      auto filteredRightIndicesSpan =
          cudf::device_span<cudf::size_type const>{*filteredRightJoinIndices};
      auto filteredRightIdxCol = cudf::column_view{filteredRightIndicesSpan};

      // Create sequence [0, 1, ..., n-1] for build table row indices
      auto n = rightTableView.num_rows();
      auto rowIndices = cudf::sequence(
          n,
          cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
          cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
          stream,
          get_temp_mr());

      // Check which build row indices are present in the filtered join result
      auto matchedInBatch = cudf::contains(
          filteredRightIdxCol, rowIndices->view(), stream, get_temp_mr());

      // OR with existing flags to accumulate matches across batches
      auto updatedFlags = cudf::binary_operation(
          rightMatchedFlags->view(),
          matchedInBatch->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          get_temp_mr());
      // binary_operation is async on `stream`; the old column destructs via
      // cudaFreeAsync on its allocation stream (not `stream`), so the free
      // can race the kernel. Drain `stream` before the move-assign.
      stream.synchronize();
      rightMatchedFlags = std::move(updatedFlags);

      // Build output using filtered indices
      auto filteredLeftIndicesSpan =
          cudf::device_span<cudf::size_type const>{*filteredLeftJoinIndices};
      auto filteredLeftIndicesCol = cudf::column_view{filteredLeftIndicesSpan};
      auto filteredRightIndicesCol =
          cudf::column_view{filteredRightIndicesSpan};
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          filteredLeftIndicesCol,
          rightTableView,
          filteredRightIndicesCol,
          stream));
    } else {
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  }
  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>> CudfHashJoinProbe::leftSemiFilterJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;

  auto& rightTables = hashObject_.value().first;

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftJoinIndices;

    if (joinNode_->filter()) {
      leftJoinIndices = cudf::mixed_left_semi_join(
          leftTableView.select(leftKeyIndices_),
          rightTableView.select(rightKeyIndices_),
          leftTableView,
          rightTableView,
          tree_.back(),
          cudf::null_equality::UNEQUAL,
          stream,
          get_temp_mr());
    } else {
      cudf::filtered_join filter_join(
          rightTableView.select(rightKeyIndices_),
          cudf::null_equality::UNEQUAL,
          cudf::set_as_build_table::RIGHT,
          stream);
      leftJoinIndices = filter_join.semi_join(
          leftTableView.select(leftKeyIndices_), stream, get_temp_mr());
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::empty_like(leftIndicesCol);

    cudfOutputs.push_back(unfilteredOutput(
        leftTableView,
        leftIndicesCol,
        rightTableView,
        rightIndicesCol->view(),
        stream));
  }
  return cudfOutputs;
}

namespace {
/// Creates a boolean column indicating which rows have NULL in ANY key column.
/// Returns a column where row[i] = true if ANY key column is NULL at row i.
std::unique_ptr<cudf::column> createProbeKeyNullMask(
    cudf::table_view keyView,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto numRows = keyView.num_rows();

  if (keyView.num_columns() == 0 || numRows == 0) {
    auto falseScalar = cudf::numeric_scalar<bool>(false, true, stream, mr);
    return cudf::make_column_from_scalar(falseScalar, numRows, stream, mr);
  }

  // Start with first column's null mask
  auto result = cudf::is_null(keyView.column(0), stream, mr);

  // OR with other columns' null masks
  for (cudf::size_type i = 1; i < keyView.num_columns(); i++) {
    auto colIsNull = cudf::is_null(keyView.column(i), stream, mr);
    result = cudf::binary_operation(
        result->view(),
        colIsNull->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
  }
  return result;
}

/// Applies a null mask to a boolean column.
/// Where nullMask[i] is true, result[i] becomes NULL.
/// Where nullMask[i] is false, result[i] keeps its original value from col.
std::unique_ptr<cudf::column> applyNullMask(
    cudf::column_view col,
    cudf::column_view nullMask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Create a null scalar (valid=false means NULL)
  auto nullScalar = cudf::numeric_scalar<bool>(false, false, stream, mr);

  // copy_if_else: where nullMask is true, use nullScalar (NULL); else use col
  // value
  return cudf::copy_if_else(nullScalar, col, nullMask, stream, mr);
}
} // namespace

// LEFT SEMI PROJECT returns all probe rows with a boolean "match" column
// indicating whether each probe row has at least one matching build row
// (that also passes the filter, if specified). Unlike LEFT SEMI FILTER
// which filters out non-matching rows, this preserves all probe rows.
// Output cardinality always equals probe side cardinality.
//
// Implementation approach:
// 1. Use inner_join to get valid (probe_idx, build_idx) pairs where keys match
// 2. If filter exists, apply filter_join_indices(INNER_JOIN) to keep only
//    pairs where the filter passes
// 3. Use cudf::contains to check which probe row indices appear in the result.
//    This correctly handles duplicate probe indices (when one probe row matches
//    multiple build rows) by returning true if the index appears at least once.
// 4. Accumulate matches across build table batches using BITWISE_OR
// 5. For null-aware mode (without filter): apply null mask based on probe key
//    nullity and build side null keys presence
// 6. Output: all probe columns + match column
std::vector<std::unique_ptr<cudf::table>>
CudfHashJoinProbe::leftSemiProjectJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;
  auto numProbeRows = leftTableView.num_rows();

  const bool isNullAware = joinNode_->isNullAware() && !joinNode_->filter();

  // Create probe row indices sequence: [0, 1, 2, ..., numProbeRows-1]
  // Used with cudf::contains to create the match column
  auto probeRowIndices = cudf::sequence(
      numProbeRows,
      cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
      cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
      stream,
      get_temp_mr());

  // Initialize match column to all false
  auto falseScalar =
      cudf::numeric_scalar<bool>(false, true, stream, get_output_mr());
  auto matchCol = cudf::make_column_from_scalar(
      falseScalar, numProbeRows, stream, get_output_mr());

  // Precompute left (probe) table columns if needed for filter
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && !leftPrecomputeInstructions_.empty()) {
    auto leftColumnViews = tableViewToColumnViews(leftTableView);
    leftPrecomputed = precomputeSubexpressions(
        leftColumnViews,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedLeftView = createExtendedTableView(leftTableView, leftPrecomputed);
  }

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    // Use cached precomputed columns for right (build) table
    cudf::table_view extendedRightView =
        (joinNode_->filter() && !rightPrecomputeInstructions_.empty())
        ? cachedExtendedRightViews_[i]
        : rightTableView;

    // Step 1: Inner join to get (probe_idx, build_idx) pairs where keys match.
    // Unlike left_join, inner_join only returns valid pairs (no JoinNoMatch).
    VELOX_CHECK_NOT_NULL(hb);
    if (buildStream_.has_value()) {
      // Make build stream wait for probe tables to become valid
      cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
    }
    auto [leftJoinIndices, rightJoinIndices] = hb->inner_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        buildStream_.has_value() ? buildStream_.value() : stream,
        get_temp_mr());
    if (buildStream_.has_value()) {
      // Make probe stream wait for join completion before using indices
      cudaEvent_->recordFrom(buildStream_.value()).waitOn(stream);
    }

    if (leftJoinIndices->size() == 0) {
      continue; // No matches from this build table
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};

    cudf::column_view matchedProbeIndices;
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> filteredLeftIndices;

    if (joinNode_->filter()) {
      // Step 2: Apply filter to the join pairs. INNER_JOIN mode keeps only
      // pairs where the predicate evaluates to true.
      auto [filteredLeft, filteredRight] = cudf::filter_join_indices(
          extendedLeftView,
          extendedRightView,
          leftIndicesSpan,
          rightIndicesSpan,
          tree_.back(),
          cudf::join_kind::INNER_JOIN,
          stream,
          get_temp_mr());

      filteredLeftIndices = std::move(filteredLeft);
      if (filteredLeftIndices->size() == 0) {
        continue; // No matches passed filter
      }
      auto filteredLeftSpan =
          cudf::device_span<cudf::size_type const>{*filteredLeftIndices};
      matchedProbeIndices = cudf::column_view{filteredLeftSpan};
    } else {
      // No filter - use inner join results directly
      matchedProbeIndices = leftIndicesCol;
    }

    // Step 3: Create match flags using cudf::contains. For each probe row index
    // in [0, numProbeRows), check if it appears in matchedProbeIndices.
    // This handles duplicates correctly - if a probe row matches multiple build
    // rows, it appears multiple times in matchedProbeIndices, but contains()
    // returns true if it appears at least once.
    auto matchedInBatch = cudf::contains(
        matchedProbeIndices, probeRowIndices->view(), stream, get_temp_mr());

    // Step 4: Accumulate matches across build table batches using OR.
    // A probe row's final match value is true if it matched in ANY batch.
    auto updatedMatch = cudf::binary_operation(
        matchCol->view(),
        matchedInBatch->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        get_output_mr());
    stream.synchronize();
    matchCol = std::move(updatedMatch);
  }

  // Step 5: For null-aware mode (IN semantics), apply null mask to match
  // column. Empty build: always FALSE. Otherwise: NULL probe key or (no match
  // AND build has nulls) yields NULL; matched rows yield TRUE; else FALSE.
  if (isNullAware) {
    bool buildSideEmpty = true;
    for (const auto& rt : rightTables) {
      if (rt->num_rows() > 0) {
        buildSideEmpty = false;
        break;
      }
    }

    // For empty build side, IN returns FALSE (already set in matchCol).
    if (!buildSideEmpty) {
      auto probeKeyView = leftTableView.select(leftKeyIndices_);
      bool probeHasNulls = cudf::has_nulls(probeKeyView);

      if (probeHasNulls || buildSideHasNullKeys_) {
        // Compute null mask: true where result should be NULL
        auto probeKeyNullMask =
            createProbeKeyNullMask(probeKeyView, stream, get_temp_mr());

        std::unique_ptr<cudf::column> nullMask;
        if (buildSideHasNullKeys_) {
          // NULL where: probe key is NULL OR no match
          auto noMatchMask = cudf::unary_operation(
              matchCol->view(),
              cudf::unary_operator::NOT,
              stream,
              get_temp_mr());
          nullMask = cudf::binary_operation(
              probeKeyNullMask->view(),
              noMatchMask->view(),
              cudf::binary_operator::BITWISE_OR,
              cudf::data_type{cudf::type_id::BOOL8},
              stream,
              get_temp_mr());
        } else {
          // NULL only where probe key is NULL
          nullMask = std::move(probeKeyNullMask);
        }

        matchCol = applyNullMask(
            matchCol->view(), nullMask->view(), stream, get_output_mr());
      }
    }
  }

  // Step 6: Build output table with all probe columns + match column
  std::vector<std::unique_ptr<cudf::column>> outputCols;
  outputCols.resize(outputType_->names().size());

  // Copy probe columns
  auto leftInput = leftTableView.select(leftColumnIndicesToGather_);
  for (size_t i = 0; i < leftColumnIndicesToGather_.size(); i++) {
    outputCols[leftColumnOutputIndices_[i]] = std::make_unique<cudf::column>(
        leftInput.column(i), stream, get_output_mr());
  }

  // Add match column as the last column
  outputCols.back() = std::move(matchCol);

  if (buildStream_.has_value()) {
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();

  cudfOutputs.push_back(std::make_unique<cudf::table>(std::move(outputCols)));
  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>>
CudfHashJoinProbe::rightSemiFilterJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto rightTableView = rightTables[0]->view();

  VELOX_CHECK_EQ(
      rightTables.size(),
      1,
      "Multiple right tables not yet supported for rightSemiFilterJoin");

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> rightJoinIndices;
  if (joinNode_->filter()) {
    rightJoinIndices = cudf::mixed_left_semi_join(
        rightTableView.select(rightKeyIndices_),
        leftTableView.select(leftKeyIndices_),
        rightTableView,
        leftTableView,
        tree_.back(),
        cudf::null_equality::UNEQUAL,
        stream,
        get_temp_mr());
  } else {
    cudf::filtered_join filter_join(
        leftTableView.select(leftKeyIndices_),
        cudf::null_equality::UNEQUAL,
        cudf::set_as_build_table::RIGHT,
        stream);
    rightJoinIndices = filter_join.semi_join(
        rightTableView.select(rightKeyIndices_), stream, get_temp_mr());
  }

  auto rightIndicesSpan =
      cudf::device_span<cudf::size_type const>{*rightJoinIndices};
  auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
  auto leftIndicesCol = cudf::empty_like(rightIndicesCol);
  cudfOutputs.push_back(unfilteredOutput(
      leftTableView,
      leftIndicesCol->view(),
      rightTableView,
      rightIndicesCol,
      stream));

  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>> CudfHashJoinProbe::antiJoin(
    cudf::table_view leftTableViewParam,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;
  auto& rightTables = hashObject_.value().first;

  VELOX_CHECK_EQ(
      rightTables.size(),
      1,
      "Multiple right tables not yet supported for antiJoin");

  auto rightTableView = rightTables[0]->view();

  // For the special case where we need to drop nulls, we create a local table.
  // Otherwise, we use the input view directly.
  std::unique_ptr<cudf::table> modifiedLeftTable;
  cudf::table_view leftTableView = leftTableViewParam;

  // Special case for null-aware anti join where
  // build table is not empty, no nulls, and probe table has nulls
  if (joinNode_->isNullAware() and !joinNode_->filter()) {
    auto const leftTableHasNulls =
        cudf::has_nulls(leftTableViewParam.select(leftKeyIndices_));
    auto const rightTableHasNulls =
        cudf::has_nulls(rightTableView.select(rightKeyIndices_));
    if (rightTables[0]->num_rows() > 0 and !rightTableHasNulls and
        leftTableHasNulls) {
      // drop nulls on probe table - creates a new table
      modifiedLeftTable = cudf::drop_nulls(
          leftTableViewParam, leftKeyIndices_, stream, get_temp_mr());
      leftTableView = modifiedLeftTable->view();
    }
  }

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftJoinIndices;
  if (joinNode_->filter()) {
    leftJoinIndices = cudf::mixed_left_anti_join(
        leftTableView.select(leftKeyIndices_),
        rightTableView.select(rightKeyIndices_),
        leftTableView,
        rightTableView,
        tree_.back(),
        cudf::null_equality::UNEQUAL,
        stream,
        get_temp_mr());
  } else {
    auto const rightTableHasNulls =
        cudf::has_nulls(rightTableView.select(rightKeyIndices_));
    if (joinNode_->isNullAware() and rightTableHasNulls) {
      // empty result
      leftJoinIndices = std::make_unique<rmm::device_uvector<cudf::size_type>>(
          0, stream, get_temp_mr());
    } else {
      cudf::filtered_join filter_join(
          rightTableView.select(rightKeyIndices_),
          cudf::null_equality::UNEQUAL,
          cudf::set_as_build_table::RIGHT,
          stream);
      leftJoinIndices = filter_join.anti_join(
          leftTableView.select(leftKeyIndices_), stream, get_temp_mr());
    }
  }

  auto leftIndicesSpan =
      cudf::device_span<cudf::size_type const>{*leftJoinIndices};
  auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
  auto rightIndicesCol = cudf::empty_like(leftIndicesCol);
  cudfOutputs.push_back(unfilteredOutput(
      leftTableView,
      leftIndicesCol,
      rightTableView,
      rightIndicesCol->view(),
      stream));

  return cudfOutputs;
}

RowVectorPtr CudfHashJoinProbe::doGetOutput() {
  if (finished_ or !hashObject_.has_value()) {
    return nullptr;
  }
  if (!input_) {
    // If no more input, emit unmatched-right rows if needed.
    if ((joinNode_->isRightJoin() || joinNode_->isFullJoin()) && noMoreInput_ &&
        !finished_ && isLastDriver_) {
      auto& rightTables = hashObject_.value().first;
      auto stream = cudfGlobalStreamPool().get_stream();
      std::vector<std::unique_ptr<cudf::table>> toConcat;
      for (size_t i = 0; i < rightTables.size(); ++i) {
        auto& rightTable = rightTables[i];
        auto n = rightTable->num_rows();
        if (n == 0) {
          continue;
        }
        auto& flags = rightMatchedFlags_[i];
        // Build a boolean mask: unmatched = NOT(flags)
        auto boolMask = cudf::unary_operation(
            flags->view(), cudf::unary_operator::NOT, stream, get_temp_mr());

        // Count unmatched rows by summing the boolean mask
        auto unmatchedCountScalar = cudf::reduce(
            boolMask->view(),
            *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
            cudf::data_type{cudf::type_id::INT32},
            stream,
            get_temp_mr());
        auto m = static_cast<cudf::numeric_scalar<int32_t>*>(
                     unmatchedCountScalar.get())
                     ->value(stream);
        if (m == 0) {
          continue;
        }

        // Build left null columns
        std::vector<std::unique_ptr<cudf::column>> outCols(outputType_->size());
        // Left side nulls (types derive from probe schema at the matching
        // channel indices)
        for (size_t li = 0; li < leftColumnOutputIndices_.size(); ++li) {
          auto outIdx = leftColumnOutputIndices_[li];
          auto probeChannel = leftColumnIndicesToGather_[li];
          auto leftCudfType =
              veloxToCudfTypeId(probeType_->childAt(probeChannel));
          auto nullScalar = cudf::make_default_constructed_scalar(
              cudf::data_type{leftCudfType}, stream, get_temp_mr());
          outCols[outIdx] = cudf::make_column_from_scalar(
              *nullScalar, m, stream, get_output_mr());
        }
        // Right side - gather unmatched build columns if any
        if (!rightColumnIndicesToGather_.empty()) {
          auto rightInput =
              rightTable->view().select(rightColumnIndicesToGather_);
          auto unmatchedRight = cudf::apply_boolean_mask(
              rightInput, boolMask->view(), stream, get_output_mr());
          auto rightCols = unmatchedRight->release();
          for (size_t ri = 0; ri < rightColumnOutputIndices_.size(); ++ri) {
            auto outIdx = rightColumnOutputIndices_[ri];
            outCols[outIdx] = std::move(rightCols[ri]);
          }
        }
        toConcat.push_back(std::make_unique<cudf::table>(std::move(outCols)));
      }
      // TODO (dm): We build multiple right chunks only when they are too large
      // to fit in cudf::size_type. In case of a right join which doesn't have a
      // lot of matches we'll get outCols of similar size. This concatenation
      // will overflow. Try emitting result of one right chunk at a time.
      if (!toConcat.empty()) {
        auto out =
            concatenateTables(std::move(toConcat), stream, get_output_mr());
        finished_ = true;
        auto size = out->num_rows();
        if (out->num_columns() == 0 || size == 0) {
          return nullptr;
        }
        return std::make_shared<CudfVector>(
            pool(), outputType_, size, std::move(out), stream);
      }
      finished_ = true;
    }
    return nullptr;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput);
  auto stream = cudfInput->stream();
  // Use getTableView() to avoid expensive materialization for packed_table.
  // cudfInput is staying alive until the table view is no longer needed.
  auto leftTableView = cudfInput->getTableView();
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Probe table number of columns: " << leftTableView.num_columns();
    VLOG(1) << "Probe table number of rows: " << leftTableView.num_rows();
  }

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;
  for (auto i = 0; i < rightTables.size(); i++) {
    auto& rightTable = rightTables[i];
    auto& hb = hbs[i];
    VELOX_CHECK_NOT_NULL(rightTable);
    if (CudfConfig::getInstance().debugEnabled) {
      if (rightTable != nullptr)
        VLOG(2) << "right_table is not nullptr " << rightTable.get()
                << " hasValue(" << hashObject_.has_value() << ")\n";
      if (hb != nullptr)
        VLOG(2) << "hb is not nullptr " << hb.get() << " hasValue("
                << hashObject_.has_value() << ")\n";
    }
  }

  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;
  switch (joinNode_->joinType()) {
    case core::JoinType::kInner:
      cudfOutputs = innerJoin(leftTableView, stream);
      break;
    case core::JoinType::kLeft:
      cudfOutputs = leftJoin(leftTableView, stream);
      break;
    case core::JoinType::kRight:
      cudfOutputs = rightJoin(leftTableView, stream);
      break;
    case core::JoinType::kLeftSemiFilter:
      cudfOutputs = leftSemiFilterJoin(leftTableView, stream);
      break;
    case core::JoinType::kLeftSemiProject:
      cudfOutputs = leftSemiProjectJoin(leftTableView, stream);
      break;
    case core::JoinType::kRightSemiFilter:
      cudfOutputs = rightSemiFilterJoin(leftTableView, stream);
      break;
    case core::JoinType::kAnti:
      cudfOutputs = antiJoin(leftTableView, stream);
      break;
    case core::JoinType::kFull:
      cudfOutputs = fullJoin(leftTableView, stream);
      break;
    default:
      VELOX_FAIL("Unsupported join type: ", joinNode_->joinType());
  }

  // Record probe stream for cross-driver synchronization in noMoreInput().
  if (joinNode_->isRightJoin() || joinNode_->isFullJoin()) {
    lastProbeStream_ = stream;
  }

  // Release input CudfVector to free GPU memory before creating output.
  // This reduces peak memory from (input + output) to max(input, output).
  // cudfInput must be released first since input_.reset() only decrements
  // the refcount while cudfInput still holds a reference.
  cudfInput.reset();
  input_.reset();
  finished_ =
      noMoreInput_ && !joinNode_->isRightJoin() && !joinNode_->isFullJoin();

  auto cudfOutput =
      concatenateTables(std::move(cudfOutputs), stream, get_output_mr());
  auto const size = cudfOutput->num_rows();
  if (cudfOutput->num_columns() == 0 or size == 0) {
    return nullptr;
  }
  return std::make_shared<CudfVector>(
      pool(),
      outputType_,
      cudfOutput->num_rows(),
      std::move(cudfOutput),
      stream);
}

bool CudfHashJoinProbe::skipProbeOnEmptyBuild() const {
  auto const joinType = joinNode_->joinType();
  return isInnerJoin(joinType) || isLeftSemiFilterJoin(joinType) ||
      isRightJoin(joinType) || isRightSemiFilterJoin(joinType) ||
      isRightSemiProjectJoin(joinType);
}

exec::BlockingReason CudfHashJoinProbe::isBlocked(ContinueFuture* future) {
  if ((joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin() ||
       joinNode_->isFullJoin()) &&
      hashObject_.has_value()) {
    if (!future_.valid()) {
      return exec::BlockingReason::kNotBlocked;
    }
    *future = std::move(future_);
    return exec::BlockingReason::kWaitForJoinProbe;
  }

  if (hashObject_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudfJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(cudfJoinBridge);
  VELOX_CHECK_NOT_NULL(future);
  auto hashObject = cudfJoinBridge->hashOrFuture(future);

  if (!hashObject.has_value()) {
    if (CudfConfig::getInstance().debugEnabled) {
      VLOG(2) << "CudfHashJoinProbe is blocked, waiting for join build";
    }
    return exec::BlockingReason::kWaitForJoinBuild;
  }
  hashObject_ = std::move(hashObject);
  buildStream_ = cudfJoinBridge->getBuildStream();

  // Lazy initialize matched flags only when build side is done
  if (joinNode_->isRightJoin() || joinNode_->isFullJoin()) {
    auto& rightTablesInit = hashObject_.value().first;
    rightMatchedFlags_.clear();
    rightMatchedFlags_.reserve(rightTablesInit.size());
    auto initStream = cudfGlobalStreamPool().get_stream();
    for (auto& rt : rightTablesInit) {
      auto n = rt->num_rows();
      auto false_scalar =
          cudf::numeric_scalar<bool>(false, true, initStream, get_temp_mr());
      auto flags_col = cudf::make_column_from_scalar(
          false_scalar, n, initStream, get_temp_mr());
      rightMatchedFlags_.push_back(std::move(flags_col));
    }
    initStream.synchronize();
  }

  // Precompute right table columns if filter exists (once when build is done)
  if (joinNode_->filter() && !rightPrecomputeInstructions_.empty()) {
    auto& rightTablesInit = hashObject_.value().first;
    cachedRightPrecomputed_.clear();
    cachedExtendedRightViews_.clear();
    cachedRightPrecomputed_.reserve(rightTablesInit.size());
    cachedExtendedRightViews_.reserve(rightTablesInit.size());

    auto initStream = cudfGlobalStreamPool().get_stream();
    for (auto& rt : rightTablesInit) {
      auto rightTableView = rt->view();
      auto rightColumnViews = tableViewToColumnViews(rightTableView);
      auto rightPrecomputed = precomputeSubexpressions(
          rightColumnViews,
          rightPrecomputeInstructions_,
          scalars_,
          buildType_,
          initStream);
      auto extendedView =
          createExtendedTableView(rightTableView, rightPrecomputed);
      cachedRightPrecomputed_.push_back(std::move(rightPrecomputed));
      cachedExtendedRightViews_.push_back(extendedView);
    }
    initStream.synchronize();
  }

  // Check if build side has any null keys (needed for null-aware left semi
  // project)
  if (joinNode_->isLeftSemiProjectJoin() && joinNode_->isNullAware()) {
    auto& rightTablesInit = hashObject_.value().first;
    buildSideHasNullKeys_ = false;
    for (auto& rt : rightTablesInit) {
      auto keyView = rt->view().select(rightKeyIndices_);
      for (cudf::size_type k = 0; k < keyView.num_columns(); k++) {
        if (keyView.column(k).has_nulls()) {
          buildSideHasNullKeys_ = true;
          break;
        }
      }
      if (buildSideHasNullKeys_) {
        break;
      }
    }
  }

  auto& rightTables = hashObject_.value().first;
  // should be rightTable->numDistinct() but it needs compute,
  // so we use num_rows()
  if (rightTables[0]->num_rows() == 0) {
    if (skipProbeOnEmptyBuild()) {
      if (operatorCtx_->driverCtx()
              ->queryConfig()
              .hashProbeFinishEarlyOnEmptyBuild()) {
        noMoreInput();
      } else {
        skipInput_ = true;
      }
    }
  }
  if ((joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin() ||
       joinNode_->isFullJoin()) &&
      future_.valid()) {
    *future = std::move(future_);
    return exec::BlockingReason::kWaitForJoinProbe;
  }
  return exec::BlockingReason::kNotBlocked;
}

bool CudfHashJoinProbe::isFinished() {
  auto const isFinished = finished_ || (noMoreInput_ && input_ == nullptr);

  // Release hashObject_ if finished
  if (isFinished) {
    hashObject_.reset();
  }
  return isFinished;
}

std::unique_ptr<exec::Operator> CudfHashJoinBridgeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridgeTranslator::toOperator";
  }
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    return std::make_unique<CudfHashJoinProbe>(id, ctx, joinNode);
  }
  return nullptr;
}

std::unique_ptr<exec::JoinBridge> CudfHashJoinBridgeTranslator::toJoinBridge(
    const core::PlanNodePtr& node) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridgeTranslator::toJoinBridge";
  }
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    auto joinBridge = std::make_unique<CudfHashJoinBridge>();
    return joinBridge;
  }
  return nullptr;
}

exec::OperatorSupplier CudfHashJoinBridgeTranslator::toOperatorSupplier(
    const core::PlanNodePtr& node) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridgeTranslator::toOperatorSupplier";
  }
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    return [joinNode](int32_t operatorId, exec::DriverCtx* ctx) {
      return std::make_unique<CudfHashJoinBuild>(operatorId, ctx, joinNode);
    };
  }
  return nullptr;
}

} // namespace facebook::velox::cudf_velox
