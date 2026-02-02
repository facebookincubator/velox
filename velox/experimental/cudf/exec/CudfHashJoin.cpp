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
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
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
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <nvtx3/nvtx3.hpp>

namespace facebook::velox::cudf_velox {

namespace {

/// Creates columns from a table view for precomputation
std::vector<std::unique_ptr<cudf::column>> tableViewToColumns(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(tableView.num_columns());
  for (cudf::size_type i = 0; i < tableView.num_columns(); ++i) {
    columns.push_back(std::make_unique<cudf::column>(
        tableView.column(i), stream, cudf::get_current_device_resource_ref()));
  }
  return columns;
}

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

void CudfHashJoinBridge::setHashTable(
    std::optional<CudfHashJoinBridge::hash_type> hashObject) {
  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "Calling CudfHashJoinBridge::setHashTable" << std::endl;
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
    LOG(INFO) << "Calling CudfHashJoinBridge::hashOrFuture" << std::endl;
  }
  std::lock_guard<std::mutex> l(mutex_);
  if (hashObject_.has_value()) {
    return hashObject_;
  }
  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "Calling CudfHashJoinBridge::hashOrFuture constructing promise"
              << std::endl;
  }
  promises_.emplace_back("CudfHashJoinBridge::hashOrFuture");
  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "Calling CudfHashJoinBridge::hashOrFuture getSemiFuture"
              << std::endl;
  }
  *future = promises_.back().getSemiFuture();
  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "Calling CudfHashJoinBridge::hashOrFuture returning nullopt"
              << std::endl;
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
    : exec::Operator(
          driverCtx,
          nullptr, // joinNode->sources(),
          operatorId,
          joinNode->id(),
          "CudfHashJoinBuild"),
      NvtxHelper(
          nvtx3::rgb{65, 105, 225}, // Royal Blue
          operatorId,
          fmt::format("[{}]", joinNode->id())),
      joinNode_(joinNode) {
  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "CudfHashJoinBuild constructor" << std::endl;
  }
}

void CudfHashJoinBuild::addInput(RowVectorPtr input) {
  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "Calling CudfHashJoinBuild::addInput" << std::endl;
  }
  // Queue inputs, process all at once.
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    // Count nulls in join key columns
    auto [_, null_count] = cudf::bitmask_and(
        cudfInput->getTableView(),
        cudfInput->stream(),
        cudf::get_current_device_resource_ref());
    {
      // Update statistics for null keys in join operator.
      auto lockedStats = stats_.wlock();
      lockedStats->numNullKeys += null_count;
    }
    inputs_.push_back(std::move(cudfInput));
  }
}

bool CudfHashJoinBuild::needsInput() const {
  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "Calling CudfHashJoinBuild::needsInput" << std::endl;
  }
  return !noMoreInput_;
}

RowVectorPtr CudfHashJoinBuild::getOutput() {
  return nullptr;
}

void CudfHashJoinBuild::noMoreInput() {
  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "Calling CudfHashJoinBuild::noMoreInput" << std::endl;
  }
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
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
    LOG(INFO) << "CudfHashJoinBuild: build batches" << std::endl;
    LOG(INFO) << "Build batches number of columns: "
              << inputs_[0]->getTableView().num_columns() << std::endl;
    for (auto i = 0; i < inputs_.size(); i++) {
      LOG(INFO) << "Build batch " << i
                << ": number of rows: " << inputs_[i]->getTableView().num_rows()
                << std::endl;
    }
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto tbls = getConcatenatedTableBatched(
      inputs_, joinNode_->sources()[1]->outputType(), stream);

  // Release input data after synchronizing
  stream.synchronize();
  inputs_.clear();

  for (auto const& tbl : tbls) {
    VELOX_CHECK_NOT_NULL(tbl);
  }
  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "Build table number of columns: " << tbls[0]->num_columns()
              << std::endl;
    for (auto i = 0; i < tbls.size(); i++) {
      LOG(INFO) << "Build table " << i
                << ": number of rows: " << tbls[i]->num_rows() << std::endl;
    }
  }

  auto buildType = joinNode_->sources()[1]->outputType();
  auto rightKeys = joinNode_->rightKeys();

  auto buildKeyIndices = std::vector<cudf::size_type>(rightKeys.size());
  for (size_t i = 0; i < buildKeyIndices.size(); i++) {
    buildKeyIndices[i] = static_cast<cudf::size_type>(
        buildType->getChildIdx(rightKeys[i]->name()));
  }

  // Only need to construct hash_join object if it's an inner join, left join or
  // right join.
  // All other cases use a standalone function in cudf
  bool buildHashJoin =
      (joinNode_->isInnerJoin() || joinNode_->isLeftJoin() ||
       joinNode_->isRightJoin());

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
        LOG(INFO) << "hashObject " << i << " is not nullptr "
                  << hashObjects.back().get() << "\n";
      } else {
        LOG(INFO) << "hashObject " << i << " is *** nullptr\n";
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
    : exec::Operator(
          driverCtx,
          joinNode->outputType(),
          operatorId,
          joinNode->id(),
          "CudfHashJoinProbe"),
      NvtxHelper(
          nvtx3::rgb{0, 128, 128}, // Teal
          operatorId,
          fmt::format("[{}]", joinNode->id())),
      joinNode_(joinNode),
      cudaEvent_(std::make_unique<CudaEvent>(cudaEventDisableTiming)) {
  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "CudfHashJoinProbe constructor" << std::endl;
  }
  auto probeType = joinNode_->sources()[0]->outputType();
  auto buildType = joinNode_->sources()[1]->outputType();
  auto const& leftKeys = joinNode_->leftKeys(); // probe keys
  auto const& rightKeys = joinNode_->rightKeys(); // build keys

  if (CudfConfig::getInstance().debugEnabled) {
    for (int i = 0; i < probeType->names().size(); i++) {
      LOG(INFO) << "Left column " << i << ": " << probeType->names()[i]
                << std::endl;
    }

    for (int i = 0; i < buildType->names().size(); i++) {
      LOG(INFO) << "Right column " << i << ": " << buildType->names()[i]
                << std::endl;
    }

    for (int i = 0; i < leftKeys.size(); i++) {
      LOG(INFO) << "Left key " << i << ": " << leftKeys[i]->name() << " "
                << leftKeys[i]->type()->kind() << std::endl;
    }

    for (int i = 0; i < rightKeys.size(); i++) {
      LOG(INFO) << "Right key " << i << ": " << rightKeys[i]->name() << " "
                << rightKeys[i]->type()->kind() << std::endl;
    }
  }

  auto const probeTableNumColumns = probeType->size();
  leftKeyIndices_ = std::vector<cudf::size_type>(leftKeys.size());
  for (size_t i = 0; i < leftKeyIndices_.size(); i++) {
    leftKeyIndices_[i] = static_cast<cudf::size_type>(
        probeType->getChildIdx(leftKeys[i]->name()));
    VELOX_CHECK_LT(leftKeyIndices_[i], probeTableNumColumns);
  }
  auto const buildTableNumColumns = buildType->size();
  rightKeyIndices_ = std::vector<cudf::size_type>(rightKeys.size());
  for (size_t i = 0; i < rightKeyIndices_.size(); i++) {
    rightKeyIndices_[i] = static_cast<cudf::size_type>(
        buildType->getChildIdx(rightKeys[i]->name()));
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
      LOG(INFO) << "Output column " << i << ": " << outputName << std::endl;
    }
    auto channel = probeType->getChildIdxIfExists(outputName);
    if (channel.has_value()) {
      leftColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      leftColumnOutputIndices_.push_back(i);
      continue;
    }
    channel = buildType->getChildIdxIfExists(outputName);
    if (channel.has_value()) {
      rightColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      rightColumnOutputIndices_.push_back(i);
      continue;
    }
    VELOX_FAIL(
        "Join field {} not in probe or build input", outputType->children()[i]);
  }

  if (CudfConfig::getInstance().debugEnabled) {
    for (int i = 0; i < leftColumnIndicesToGather_.size(); i++) {
      LOG(INFO) << "Left index to gather " << i << ": "
                << leftColumnIndicesToGather_[i] << std::endl;
    }

    for (int i = 0; i < rightColumnIndicesToGather_.size(); i++) {
      LOG(INFO) << "Right index to gather " << i << ": "
                << rightColumnIndicesToGather_[i] << std::endl;
    }
  }

  // Store row types for precomputation
  probeType_ = probeType;
  buildType_ = buildType;

  // Setup filter in case it exists
  if (joinNode_->filter()) {
    // simplify expression
    exec::ExprSet exprs({joinNode_->filter()}, operatorCtx_->execCtx());
    VELOX_CHECK_EQ(exprs.exprs().size(), 1);

    // We don't need to get tables that contain conditional comparison columns
    // We'll pass the entire table. The ast will handle finding the required
    // columns. This is required because we build the ast with whole row schema
    // and the column locations in that schema translate to column locations
    // in whole tables

    // create ast tree
    std::vector<PrecomputeInstruction> rightPrecomputeInstructions;
    std::vector<PrecomputeInstruction> leftPrecomputeInstructions;
    static constexpr bool kAllowPureAstOnly = false;
    if (joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin()) {
      createAstTree(
          exprs.exprs()[0],
          tree_,
          scalars_,
          buildType,
          probeType,
          rightPrecomputeInstructions,
          leftPrecomputeInstructions,
          kAllowPureAstOnly);
    } else {
      createAstTree(
          exprs.exprs()[0],
          tree_,
          scalars_,
          probeType,
          buildType,
          leftPrecomputeInstructions,
          rightPrecomputeInstructions,
          kAllowPureAstOnly);
    }
    // Store precompute instructions for use during join execution
    leftPrecomputeInstructions_ = std::move(leftPrecomputeInstructions);
    rightPrecomputeInstructions_ = std::move(rightPrecomputeInstructions);
  }
}

bool CudfHashJoinProbe::needsInput() const {
  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "Calling CudfHashJoinProbe::needsInput" << std::endl;
  }
  if (joinNode_->isRightSemiFilterJoin()) {
    return !noMoreInput_;
  }
  return !noMoreInput_ && !finished_ && input_ == nullptr;
}

void CudfHashJoinProbe::addInput(RowVectorPtr input) {
  if (skipInput_) {
    VELOX_CHECK_NULL(input_);
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  // Count nulls in join key columns
  auto [_, null_count] = cudf::bitmask_and(
      cudfInput->getTableView(),
      cudfInput->stream(),
      cudf::get_current_device_resource_ref());
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

void CudfHashJoinProbe::noMoreInput() {
  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "Calling CudfHashJoinProbe::noMoreInput" << std::endl;
  }
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  Operator::noMoreInput();
  if (!joinNode_->isRightJoin() && !joinNode_->isRightSemiFilterJoin()) {
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

  if (joinNode_->isRightJoin()) {
    isLastDriver_ = true;
    if (hashObject_.has_value()) {
      auto stream = cudfGlobalStreamPool().get_stream();
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
              cudf::get_current_device_resource_ref());
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
  auto tbl = getConcatenatedTable(
      inputs_, joinNode_->sources()[1]->outputType(), stream);

  // Release input data after synchronizing
  stream.synchronize();

  VELOX_CHECK_NOT_NULL(tbl);

  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "Probe table number of columns: " << tbl->num_columns()
              << std::endl;
    LOG(INFO) << "Probe table number of rows: " << tbl->num_rows() << std::endl;
  }

  // Store the concatenated table in input_
  input_ = std::make_shared<CudfVector>(
      operatorCtx_->pool(),
      joinNode_->outputType(),
      tbl->num_rows(),
      std::move(tbl),
      stream);

  inputs_.clear();
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
  auto leftResult = cudf::gather(leftInput, leftIndicesCol, oobPolicy, stream);
  auto rightResult =
      cudf::gather(rightInput, rightIndicesCol, oobPolicy, stream);

  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "Left result number of columns: " << leftResult->num_columns()
              << std::endl;
    LOG(INFO) << "Right result number of columns: "
              << rightResult->num_columns() << std::endl;
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
  auto leftResult =
      cudf::gather(leftTableView, leftIndicesCol, oobPolicy, stream);
  auto rightResult =
      cudf::gather(rightTableView, rightIndicesCol, oobPolicy, stream);
  auto leftColsSize = leftResult->num_columns();
  auto rightColsSize = rightResult->num_columns();

  std::vector<std::unique_ptr<cudf::column>> joinedCols = leftResult->release();
  auto rightCols = rightResult->release();
  joinedCols.insert(
      joinedCols.end(),
      std::make_move_iterator(rightCols.begin()),
      std::make_move_iterator(rightCols.end()));

  auto probeType = joinNode_->sources()[0]->outputType();
  auto buildType = joinNode_->sources()[1]->outputType();
  std::vector<velox::RowTypePtr> rowTypes{probeType, buildType};
  exec::ExprSet exprs({joinNode_->filter()}, operatorCtx_->execCtx());
  VELOX_CHECK_EQ(exprs.exprs().size(), 1);
  auto filterEvaluator = createCudfExpression(
      exprs.exprs()[0], facebook::velox::type::concatRowTypes(rowTypes));
  auto filterColumns = filterEvaluator->eval(
      joinedCols, stream, cudf::get_current_device_resource_ref());
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
          stream);

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
  std::vector<std::unique_ptr<cudf::column>> leftOwnedColumns;
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && !leftPrecomputeInstructions_.empty()) {
    leftOwnedColumns = tableViewToColumns(leftTableView, stream);
    leftPrecomputed = precomputeSubexpressions(
        leftOwnedColumns,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedLeftView = createExtendedTableView(leftTableView, leftPrecomputed);
  }

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    // Precompute right (build) table columns if needed (per batch)
    std::vector<std::unique_ptr<cudf::column>> rightOwnedColumns;
    std::vector<ColumnOrView> rightPrecomputed;
    cudf::table_view extendedRightView = rightTableView;
    if (joinNode_->filter() && !rightPrecomputeInstructions_.empty()) {
      rightOwnedColumns = tableViewToColumns(rightTableView, stream);
      rightPrecomputed = precomputeSubexpressions(
          rightOwnedColumns,
          rightPrecomputeInstructions_,
          scalars_,
          buildType_,
          stream);
      extendedRightView =
          createExtendedTableView(rightTableView, rightPrecomputed);
    }

    // left = probe, right = build
    VELOX_CHECK_NOT_NULL(hb);
    if (buildStream_.has_value()) {
      // Make build stream wait for probe tables to become valid
      cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
    }
    auto [leftJoinIndices, rightJoinIndices] = hb->inner_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        buildStream_.has_value() ? buildStream_.value() : stream);
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
  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    VELOX_CHECK_NOT_NULL(hb);
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
    }
    auto [leftJoinIndices, rightJoinIndices] = hb->left_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        buildStream_.has_value() ? buildStream_.value() : stream);
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
      // TODO: Add precomputation support for left join
      cudfOutputs.push_back(filteredOutputIndices(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          leftTableView, // Extended view (stub: using original for now)
          rightTableView, // Extended view (stub: using original for now)
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
        buildStream_.has_value() ? buildStream_.value() : stream);
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(buildStream_.value()).waitOn(stream);
    }
    if (!joinNode_->filter()) {
      // Mark matched rights using scatter of true into flags at matching
      // indices
      // TODO (dm): Use a better implementation that doesn't require making a
      // scalar and broadcasting. My cuDF foo is rusty
      auto true_scalar = cudf::numeric_scalar<bool>(true, true, stream);
      auto true_col = cudf::make_column_from_scalar(
          true_scalar,
          rightJoinIndices->size(),
          stream,
          cudf::get_current_device_resource_ref());
      auto flags_table = cudf::table_view({rightMatchedFlags_[i]->view()});
      auto rightIdxCol = cudf::column_view{
          cudf::device_span<cudf::size_type const>{*rightJoinIndices}};
      auto updated_flags_tbl = cudf::scatter(
          cudf::table_view({true_col->view()}),
          rightIdxCol,
          flags_table,
          stream,
          cudf::get_current_device_resource_ref());
      rightMatchedFlags_[i] = std::move(updated_flags_tbl->release()[0]);
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
      auto filterFunc =
          [&rightMatchedFlags, rightIndicesSpan, stream](
              std::vector<std::unique_ptr<cudf::column>>&& joinedCols,
              cudf::column_view filterColumn) {
            // apply the filter
            auto filterTable =
                std::make_unique<cudf::table>(std::move(joinedCols));
            auto filteredTable =
                cudf::apply_boolean_mask(*filterTable, filterColumn, stream);
            joinedCols = filteredTable->release();

            // For streaming right join, after applying filter, we record
            // matched right indices filter rightJoinIndices with the same mask
            // to update matched flags
            auto rightIdxCol = cudf::column_view{rightIndicesSpan};
            auto filteredIdxTable = cudf::apply_boolean_mask(
                cudf::table_view{std::vector<cudf::column_view>{rightIdxCol}},
                filterColumn,
                stream);
            auto filteredCols = filteredIdxTable->release();
            auto filteredRightIdxCol = std::move(filteredCols[0]);

            // TODO (dm): The below code is repeated from non-filter case. Find
            // a way to consolidate in future
            auto true_scalar = cudf::numeric_scalar<bool>(true, true, stream);
            auto true_col = cudf::make_column_from_scalar(
                true_scalar,
                filteredRightIdxCol->size(),
                stream,
                cudf::get_current_device_resource_ref());
            auto flags_table = cudf::table_view({rightMatchedFlags->view()});
            auto updated_flags_tbl = cudf::scatter(
                cudf::table_view({true_col->view()}),
                filteredRightIdxCol->view(),
                flags_table,
                stream,
                cudf::get_current_device_resource_ref());
            rightMatchedFlags = std::move(updated_flags_tbl->release()[0]);
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
          cudf::get_current_device_resource_ref());
    } else {
      cudf::filtered_join filter_join(
          rightTableView.select(rightKeyIndices_),
          cudf::null_equality::UNEQUAL,
          cudf::set_as_build_table::RIGHT,
          stream);
      leftJoinIndices = filter_join.semi_join(
          leftTableView.select(leftKeyIndices_),
          stream,
          cudf::get_current_device_resource_ref());
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
        cudf::get_current_device_resource_ref());
  } else {
    cudf::filtered_join filter_join(
        leftTableView.select(leftKeyIndices_),
        cudf::null_equality::UNEQUAL,
        cudf::set_as_build_table::RIGHT,
        stream);
    rightJoinIndices = filter_join.semi_join(
        rightTableView.select(rightKeyIndices_),
        stream,
        cudf::get_current_device_resource_ref());
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
      modifiedLeftTable =
          cudf::drop_nulls(leftTableViewParam, leftKeyIndices_, stream);
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
        cudf::get_current_device_resource_ref());
  } else {
    auto const rightTableHasNulls =
        cudf::has_nulls(rightTableView.select(rightKeyIndices_));
    if (joinNode_->isNullAware() and rightTableHasNulls) {
      // empty result
      leftJoinIndices = std::make_unique<rmm::device_uvector<cudf::size_type>>(
          0, stream, cudf::get_current_device_resource_ref());
    } else {
      cudf::filtered_join filter_join(
          rightTableView.select(rightKeyIndices_),
          cudf::null_equality::UNEQUAL,
          cudf::set_as_build_table::RIGHT,
          stream);
      leftJoinIndices = filter_join.anti_join(
          leftTableView.select(leftKeyIndices_),
          stream,
          cudf::get_current_device_resource_ref());
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

RowVectorPtr CudfHashJoinProbe::getOutput() {
  if (CudfConfig::getInstance().debugEnabled) {
    LOG(INFO) << "Calling CudfHashJoinProbe::getOutput" << std::endl;
  }
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (finished_ or !hashObject_.has_value()) {
    return nullptr;
  }
  if (!input_) {
    // If no more input, emit unmatched-right rows if needed.
    if (joinNode_->isRightJoin() && noMoreInput_ && !finished_ &&
        isLastDriver_) {
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
            flags->view(), cudf::unary_operator::NOT, stream);

        auto rightInput =
            rightTable->view().select(rightColumnIndicesToGather_);
        auto unmatchedRight =
            cudf::apply_boolean_mask(rightInput, boolMask->view(), stream);
        auto m = unmatchedRight->num_rows();
        if (m == 0) {
          continue;
        }
        // Build left null columns
        std::vector<std::unique_ptr<cudf::column>> outCols(outputType_->size());
        // Left side nulls (types derive from probe schema at the matching
        // channel indices)
        auto probeType = joinNode_->sources()[0]->outputType();
        for (int li = 0; li < leftColumnOutputIndices_.size(); ++li) {
          auto outIdx = leftColumnOutputIndices_[li];
          auto probeChannel = leftColumnIndicesToGather_[li];
          auto leftCudfType =
              veloxToCudfTypeId(probeType->childAt(probeChannel));
          auto nullScalar = cudf::make_default_constructed_scalar(
              cudf::data_type{leftCudfType});
          outCols[outIdx] = cudf::make_column_from_scalar(
              *nullScalar, m, stream, cudf::get_current_device_resource_ref());
        }
        // Right side
        auto rightCols = unmatchedRight->release();
        for (int ri = 0; ri < rightColumnOutputIndices_.size(); ++ri) {
          auto outIdx = rightColumnOutputIndices_[ri];
          outCols[outIdx] = std::move(rightCols[ri]);
        }
        toConcat.push_back(std::make_unique<cudf::table>(std::move(outCols)));
      }
      // TODO (dm): We build multiple right chunks only when they are too large
      // to fit in cudf::size_type. In case of a right join which doesn't have a
      // lot of matches we'll get outCols of similar size. This concatenation
      // will overflow. Try emitting result of one right chunk at a time.
      if (!toConcat.empty()) {
        auto out = concatenateTables(std::move(toConcat), stream);
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
    LOG(INFO) << "Probe table number of columns: "
              << leftTableView.num_columns() << std::endl;
    LOG(INFO) << "Probe table number of rows: " << leftTableView.num_rows()
              << std::endl;
  }

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;
  for (auto i = 0; i < rightTables.size(); i++) {
    auto& rightTable = rightTables[i];
    auto& hb = hbs[i];
    VELOX_CHECK_NOT_NULL(rightTable);
    if (CudfConfig::getInstance().debugEnabled) {
      if (rightTable != nullptr)
        LOG(INFO) << "right_table is not nullptr " << rightTable.get()
                  << " hasValue(" << hashObject_.has_value() << ")\n";
      if (hb != nullptr)
        LOG(INFO) << "hb is not nullptr " << hb.get() << " hasValue("
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
    case core::JoinType::kRightSemiFilter:
      cudfOutputs = rightSemiFilterJoin(leftTableView, stream);
      break;
    case core::JoinType::kAnti:
      cudfOutputs = antiJoin(leftTableView, stream);
      break;
    default:
      VELOX_FAIL("Unsupported join type: ", joinNode_->joinType());
  }

  // Release input CudfVector to free GPU memory before creating output.
  // This reduces peak memory from (input + output) to max(input, output).
  // cudfInput must be released first since input_.reset() only decrements
  // the refcount while cudfInput still holds a reference.
  cudfInput.reset();
  input_.reset();
  finished_ = noMoreInput_ && !joinNode_->isRightJoin();

  auto cudfOutput = concatenateTables(std::move(cudfOutputs), stream);
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
  if ((joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin()) &&
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
      LOG(INFO) << "CudfHashJoinProbe is blocked, waiting for join build"
                << std::endl;
    }
    return exec::BlockingReason::kWaitForJoinBuild;
  }
  hashObject_ = std::move(hashObject);
  buildStream_ = cudfJoinBridge->getBuildStream();

  // Lazy initialize matched flags only when build side is done
  if (joinNode_->isRightJoin()) {
    auto& rightTablesInit = hashObject_.value().first;
    rightMatchedFlags_.clear();
    rightMatchedFlags_.reserve(rightTablesInit.size());
    auto initStream = cudfGlobalStreamPool().get_stream();
    for (auto& rt : rightTablesInit) {
      auto n = rt->num_rows();
      auto false_scalar = cudf::numeric_scalar<bool>(false, true, initStream);
      auto flags_col = cudf::make_column_from_scalar(
          false_scalar, n, initStream, cudf::get_current_device_resource_ref());
      rightMatchedFlags_.push_back(std::move(flags_col));
    }
    initStream.synchronize();
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
  if ((joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin()) &&
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
    LOG(INFO) << "Calling CudfHashJoinBridgeTranslator::toOperator"
              << std::endl;
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
    LOG(INFO) << "Calling CudfHashJoinBridgeTranslator::toJoinBridge"
              << std::endl;
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
    LOG(INFO) << "Calling CudfHashJoinBridgeTranslator::toOperatorSupplier"
              << std::endl;
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
