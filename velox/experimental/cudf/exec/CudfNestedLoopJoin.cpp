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
#include "velox/exec/NestedLoopJoinBuild.h"
#include "velox/exec/Task.h"

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
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(
        cudfInput, "CudfNestedLoopJoinBuild expects CudfVector");
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
    inputs_.insert(inputs_.end(), build->inputs_.begin(), build->inputs_.end());
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
  auto tbls = getConcatenatedTableBatched(inputs_, buildType, stream, mr);
  stream.synchronize();
  inputs_.clear();

  std::vector<RowVectorPtr> buildVectors;
  buildVectors.reserve(tbls.size());
  for (auto& tbl : tbls) {
    VELOX_CHECK_NOT_NULL(tbl);
    auto numRows = tbl->num_rows();
    if (numRows == 0) {
      continue;
    }
    auto cudfVec = std::make_shared<CudfVector>(
        pool(), buildType, numRows, std::move(tbl), stream);
    buildVectors.push_back(std::move(cudfVec));
  }

  if (buildVectors.empty()) {
    buildVectors.push_back(
        std::make_shared<CudfVector>(
            pool(), buildType, 0, makeEmptyTable(buildType), stream));
  }

  operatorCtx_->task()
      ->getNestedLoopJoinBridge(
          operatorCtx_->driverCtx()->splitGroupId, planNodeId())
      ->setData(std::move(buildVectors));
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
  if (buildVectors_.has_value()) {
    return true;
  }
  auto bridge = operatorCtx_->task()->getNestedLoopJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto data = bridge->dataOrFuture(future);
  if (!data.has_value()) {
    return false;
  }
  buildVectors_ = std::move(data);

  // Concatenate build side once and cache the result
  auto buildType = joinNode_->sources()[1]->outputType();
  std::vector<CudfVectorPtr> buildCudf;
  for (const auto& row : buildVectors_.value()) {
    auto cv = std::dynamic_pointer_cast<CudfVector>(row);
    VELOX_CHECK_NOT_NULL(cv, "Build side must be CudfVector from GPU build");
    buildCudf.push_back(cv);
  }

  if (buildCudf.size() == 1) {
    buildView_ = buildCudf[0]->getTableView();
  } else {
    auto stream = cudfGlobalStreamPool().get_stream();
    auto mr = cudf::get_current_device_resource_ref();
    concatenatedBuildTable_ =
        getConcatenatedTable(buildCudf, buildType, stream, mr);
    buildView_ = concatenatedBuildTable_->view();
    stream.synchronize();
  }
  return true;
}

bool CudfNestedLoopJoinProbe::needsInput() const {
  return !finished_ && input_ == nullptr && !noMoreInput_;
}

void CudfNestedLoopJoinProbe::addInput(RowVectorPtr input) {
  VELOX_CHECK_NULL(input_);
  input_ = std::move(input);
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

RowVectorPtr CudfNestedLoopJoinProbe::getOutput() {
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

  auto probeInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(
      probeInput, "CudfNestedLoopJoinProbe expects CudfVector");

  auto probeView = probeInput->getTableView();
  const cudf::size_type nL = probeView.num_rows();
  const cudf::size_type nR = buildView_.num_rows();

  input_.reset();

  if (nR == 0) {
    return nullptr;
  }

  const cudf::size_type outRows = nL * nR;
  auto stream = probeInput->stream();
  auto mr = get_output_mr();

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
  auto rightGatherView = buildView_.select(rightColumnIndicesToGather_);

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

  return std::make_shared<CudfVector>(
      pool(), outputType_, outTable->num_rows(), std::move(outTable), stream);
}

bool CudfNestedLoopJoinProbe::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
