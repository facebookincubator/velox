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
#include "velox/experimental/cudf/exec/CudfNestedLoopJoin.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"
#include "velox/type/TypeUtil.h"

#include <cudf/stream_compaction.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <nvtx3/nvtx3.hpp>

namespace facebook::velox::cudf_velox {

namespace {

constexpr auto oobPolicy = cudf::out_of_bounds_policy::DONT_CHECK;

} // namespace

// --- CudfNestedLoopJoinBridge ---

void CudfNestedLoopJoinBridge::setData(BuildData data) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(!data_.has_value(), "setData must be called only once");
    data_ = std::move(data);
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

std::optional<CudfNestedLoopJoinBridge::BuildData>
CudfNestedLoopJoinBridge::dataOrFuture(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(!cancelled_, "Getting data after the build side is aborted");
  if (data_.has_value()) {
    return data_;
  }
  promises_.emplace_back("CudfNestedLoopJoinBridge::dataOrFuture");
  *future = promises_.back().getSemiFuture();
  return std::nullopt;
}

std::unique_ptr<exec::JoinBridge>
CudfNestedLoopJoinBridgeTranslator::toJoinBridge(
    const core::PlanNodePtr& node) {
  if (std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(node)) {
    return std::make_unique<CudfNestedLoopJoinBridge>();
  }
  return nullptr;
}

// --- CudfNestedLoopJoinBuild ---

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
          nvtx3::rgb{65, 105, 225},
          operatorId,
          fmt::format("[{}]", joinNode->id())),
      joinNode_(joinNode) {}

void CudfNestedLoopJoinBuild::addInput(RowVectorPtr input) {
  std::lock_guard<std::mutex> lock(cudfGlobalMutex());
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    if (!cudfInput) {
      auto stream = cudfGlobalStreamPool().get_stream();
      auto tbl = with_arrow::toCudfTable(
          input, input->pool(), stream,
          cudf::get_current_device_resource_ref());
      stream.synchronize();
      cudfInput = std::make_shared<CudfVector>(
          pool(), input->type(), input->size(), std::move(tbl), stream);
    }
    inputs_.push_back(std::move(cudfInput));
  }
}

void CudfNestedLoopJoinBuild::noMoreInput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  exec::Operator::noMoreInput();
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }

  for (auto& peer : peers) {
    auto* op = peer->findOperator(planNodeId());
    auto* build = dynamic_cast<CudfNestedLoopJoinBuild*>(op);
    VELOX_CHECK_NOT_NULL(build);
    inputs_.insert(
        inputs_.end(), build->inputs_.begin(), build->inputs_.end());
  }

  SCOPE_EXIT {
    peers.clear();
    for (auto& p : promises) {
      p.setValue();
    }
  };

  auto stream = cudfGlobalStreamPool().get_stream();
  auto buildType = joinNode_->sources()[1]->outputType();
  auto mr = cudf::get_current_device_resource_ref();
  auto tbls = getConcatenatedTableBatched(
      std::exchange(inputs_, {}), buildType, stream, mr);

  std::vector<std::shared_ptr<cudf::table>> buildTables;
  buildTables.reserve(tbls.size());
  for (auto& tbl : tbls) {
    VELOX_CHECK_NOT_NULL(tbl);
    if (tbl->num_rows() > 0) {
      buildTables.push_back(std::move(tbl));
    }
  }

  if (buildTables.empty()) {
    buildTables.push_back(makeEmptyTable(buildType));
  }

  auto bridge = std::dynamic_pointer_cast<CudfNestedLoopJoinBridge>(
      operatorCtx_->task()->getCustomJoinBridge(
          operatorCtx_->driverCtx()->splitGroupId, planNodeId()));
  VELOX_CHECK_NOT_NULL(bridge);
  bridge->setData({std::move(buildTables), stream});
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

// --- CudfNestedLoopJoinProbe ---

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
          nvtx3::rgb{0, 128, 128},
          operatorId,
          fmt::format("[{}]", joinNode->id())),
      joinNode_(joinNode) {
  auto probeType = joinNode_->sources()[0]->outputType();
  auto buildType = joinNode_->sources()[1]->outputType();
  auto outputType = joinNode_->outputType();
  for (size_t i = 0; i < outputType->names().size(); i++) {
    const auto& name = outputType->names()[i];
    auto probeIdx = probeType->getChildIdxIfExists(name);
    if (probeIdx.has_value()) {
      leftColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(probeIdx.value()));
      leftColumnOutputIndices_.push_back(i);
      continue;
    }
    auto buildIdx = buildType->getChildIdxIfExists(name);
    if (buildIdx.has_value()) {
      rightColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(buildIdx.value()));
      rightColumnOutputIndices_.push_back(i);
      continue;
    }
    VELOX_FAIL("Join output column '{}' not in probe or build", name);
  }
}

bool CudfNestedLoopJoinProbe::getBuildData(ContinueFuture* future) {
  if (buildData_.has_value()) {
    return true;
  }
  auto bridge = std::dynamic_pointer_cast<CudfNestedLoopJoinBridge>(
      operatorCtx_->task()->getCustomJoinBridge(
          operatorCtx_->driverCtx()->splitGroupId, planNodeId()));
  VELOX_CHECK_NOT_NULL(bridge);
  auto data = bridge->dataOrFuture(future);
  if (!data.has_value()) {
    return false;
  }
  buildData_ = std::move(data);
  return true;
}

bool CudfNestedLoopJoinProbe::needsInput() const {
  return !finished_ && input_ == nullptr && !noMoreInput_;
}

void CudfNestedLoopJoinProbe::addInput(RowVectorPtr input) {
  std::lock_guard<std::mutex> lock(cudfGlobalMutex());
  VELOX_CHECK_NULL(input_);
  input_ = std::move(input);
  buildBatchIdx_ = 0;
}

void CudfNestedLoopJoinProbe::noMoreInput() {
  exec::Operator::noMoreInput();
}

exec::BlockingReason CudfNestedLoopJoinProbe::isBlocked(
    ContinueFuture* future) {
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

RowVectorPtr CudfNestedLoopJoinProbe::crossJoinWithBuildBatch(
    const cudf::table_view& probeView,
    const cudf::table_view& buildBatchView,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const cudf::size_type nL = probeView.num_rows();
  const cudf::size_type nR = buildBatchView.num_rows();
  const cudf::size_type outRows = nL * nR;

  auto seqCol = cudf::sequence(
      outRows,
      cudf::numeric_scalar<cudf::size_type>(0, true, stream, mr),
      cudf::numeric_scalar<cudf::size_type>(1, true, stream, mr),
      stream,
      mr);

  auto nRScalar = cudf::numeric_scalar<cudf::size_type>(nR, true, stream, mr);

  auto leftIndicesCol = cudf::binary_operation(
      seqCol->view(),
      nRScalar,
      cudf::binary_operator::DIV,
      cudf::data_type{cudf::type_id::INT32},
      stream,
      mr);

  auto rightIndicesCol = cudf::binary_operation(
      seqCol->view(),
      nRScalar,
      cudf::binary_operator::MOD,
      cudf::data_type{cudf::type_id::INT32},
      stream,
      mr);

  auto leftGatherView = probeView.select(leftColumnIndicesToGather_);
  auto rightGatherView = buildBatchView.select(rightColumnIndicesToGather_);

  auto leftGathered = cudf::gather(
      leftGatherView, leftIndicesCol->view(), oobPolicy, stream, mr);
  auto rightGathered = cudf::gather(
      rightGatherView, rightIndicesCol->view(), oobPolicy, stream, mr);

  std::vector<std::unique_ptr<cudf::column>> outCols(
      joinNode_->outputType()->size());
  auto leftCols = leftGathered->release();
  auto rightCols = rightGathered->release();
  for (size_t i = 0; i < leftColumnOutputIndices_.size(); i++) {
    outCols[leftColumnOutputIndices_[i]] = std::move(leftCols[i]);
  }
  for (size_t i = 0; i < rightColumnOutputIndices_.size(); i++) {
    outCols[rightColumnOutputIndices_[i]] = std::move(rightCols[i]);
  }

  auto outTable = std::make_unique<cudf::table>(std::move(outCols));

  if (joinNode_->joinCondition()) {
    if (!filterEvaluator_) {
      auto probeType = asRowType(joinNode_->sources()[0]->outputType());
      auto buildType = asRowType(joinNode_->sources()[1]->outputType());
      filterConcatType_ = facebook::velox::type::concatRowTypes(
          std::vector<velox::RowTypePtr>{probeType, buildType});
      exec::ExprSet exprs(
          {joinNode_->joinCondition()}, operatorCtx_->execCtx());
      VELOX_CHECK_EQ(exprs.exprs().size(), 1);
      filterEvaluator_ =
          createCudfExpression(exprs.exprs()[0], filterConcatType_);
    }

    std::vector<cudf::column_view> allColViews;
    auto leftView = leftGathered->view();
    auto rightView = rightGathered->view();
    for (cudf::size_type i = 0; i < leftView.num_columns(); i++) {
      allColViews.push_back(leftView.column(i));
    }
    for (cudf::size_type i = 0; i < rightView.num_columns(); i++) {
      allColViews.push_back(rightView.column(i));
    }

    auto filterResult = filterEvaluator_->eval(allColViews, stream, mr);
    auto filterView = asView(filterResult);
    auto filtered = cudf::apply_boolean_mask(
        outTable->view(), filterView, stream, mr);
    outTable = std::move(filtered);
  }

  return std::make_shared<CudfVector>(
      pool(),
      outputType_,
      outTable->num_rows(),
      std::move(outTable),
      stream);
}

RowVectorPtr CudfNestedLoopJoinProbe::getOutput() {
  std::lock_guard<std::mutex> lock(cudfGlobalMutex());
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (input_ == nullptr) {
    if (!noMoreInput_) {
      return nullptr;
    }
    finished_ = true;
    return nullptr;
  }

  if (!getBuildData(&future_)) {
    return nullptr;
  }

  auto& buildTables = buildData_->tables;

  // Skip empty build batches
  while (buildBatchIdx_ < buildTables.size() &&
         buildTables[buildBatchIdx_]->num_rows() == 0) {
    ++buildBatchIdx_;
  }

  // All build batches for this probe input have been processed
  if (buildBatchIdx_ >= buildTables.size()) {
    input_.reset();
    buildBatchIdx_ = 0;
    return nullptr;
  }

  auto probeInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(
      probeInput, "CudfNestedLoopJoinProbe expects CudfVector");

  auto probeView = probeInput->getTableView();
  auto buildBatchView = buildTables[buildBatchIdx_]->view();
  auto stream = probeInput->stream();
  auto mr = get_output_mr();

  auto result = crossJoinWithBuildBatch(probeView, buildBatchView, stream, mr);
  ++buildBatchIdx_;

  // If we've processed all build batches, release the probe input
  if (buildBatchIdx_ >= buildTables.size()) {
    input_.reset();
    buildBatchIdx_ = 0;
  }

  return result;
}

bool CudfNestedLoopJoinProbe::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
