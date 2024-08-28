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

// For custom hash join operator
#include "velox/core/Expressions.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Driver.h"
#include "velox/exec/JoinBridge.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Task.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/copying.hpp>
#include <cudf/join.hpp>

#include <nvtx3/nvtx3.hpp>

#include "CudfHashJoin.h"
#include "VeloxCudfInterop.h"

namespace facebook::velox::cudf_velox {

void CudfHashJoinBridge::setHashTable(
    std::optional<CudfHashJoinBridge::hash_type> hashObject) {
  std::cout << "Calling CudfHashJoinBridge::setHashTable" << std::endl;
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
  std::cout << "Calling CudfHashJoinBridge::hashOrFuture" << std::endl;
  std::lock_guard<std::mutex> l(mutex_);
  if (hashObject_.has_value()) {
    return std::move(hashObject_);
  }
  std::cout << "Calling CudfHashJoinBridge::hashOrFuture constructing promise"
            << std::endl;
  promises_.emplace_back("CudfHashJoinBridge::hashOrFuture");
  std::cout << "Calling CudfHashJoinBridge::hashOrFuture getSemiFuture"
            << std::endl;
  *future = promises_.back().getSemiFuture();
  std::cout << "Calling CudfHashJoinBridge::hashOrFuture returning nullopt"
            << std::endl;
  return std::nullopt;
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
      joinNode_(joinNode) {
  std::cout << "CudfHashJoinBuild constructor" << std::endl;
}

void CudfHashJoinBuild::addInput(RowVectorPtr input) {
  std::cout << "Calling CudfHashJoinBuild::addInput" << std::endl;
  // Queue inputs, process all at once.
  // TODO distribute work equally.
  auto inputSize = input->size();
  if (inputSize > 0) {
    inputs_.push_back(std::move(input));
  }
}

bool CudfHashJoinBuild::needsInput() const {
  std::cout << "Calling CudfHashJoinBuild::needsInput" << std::endl;
  return !noMoreInput_;
}

RowVectorPtr CudfHashJoinBuild::getOutput() {
  std::cout << "Calling CudfHashJoinBuild::getOutput" << std::endl;
  return nullptr;
}

void CudfHashJoinBuild::noMoreInput() {
  std::cout << "Calling CudfHashJoinBuild::noMoreInput" << std::endl;
  NVTX3_FUNC_RANGE();
  Operator::noMoreInput();
  // TODO
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
    VELOX_CHECK(build);
    inputs_.insert(inputs_.end(), build->inputs_.begin(), build->inputs_.end());
  }
  // TODO build hash table
  auto tbl = to_cudf_table(inputs_[0]);
  std::cout << "Build table number of columns: " << tbl->num_columns()
            << std::endl;
  std::cout << "Build table number of rows: " << tbl->num_rows() << std::endl;

  auto buildType = joinNode_->sources()[1]->outputType();
  auto buildKeys = joinNode_->rightKeys();

  auto build_key_indices = std::vector<cudf::size_type>(buildKeys.size());
  for (size_t i = 0; i < build_key_indices.size(); i++) {
    build_key_indices[i] = static_cast<cudf::size_type>(
        buildType->getChildIdx(buildKeys[i]->name()));
  }

  auto hashObject = std::make_shared<cudf::hash_join>(
      tbl->view().select(build_key_indices), cudf::null_equality::EQUAL);

  // Copied
  peers.clear();
  for (auto& promise : promises) {
    promise.setValue();
  }

  // set hash table to CudfHashJoinBridge
  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudf_HashJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  cudf_HashJoinBridge->setHashTable(std::make_optional(
      std::make_pair(std::move(tbl), std::move(hashObject))));
}

exec::BlockingReason CudfHashJoinBuild::isBlocked(ContinueFuture* future) {
  std::cout << "Calling CudfHashJoinBuild::isBlocked" << std::endl;
  if (!future_.valid()) {
    std::cout << "CudfHashJoinBuild future is not valid" << std::endl;
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfHashJoinBuild::isFinished() {
  std::cout << "Calling CudfHashJoinBuild::isFinished" << std::endl;
  return !future_.valid() && noMoreInput_;
}

CudfHashJoinProbe::CudfHashJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    : exec::Operator(
          driverCtx,
          nullptr, // joinNode->sources(),
          operatorId,
          joinNode->id(),
          "CudfHashJoinProbe"),
      joinNode_(joinNode) {
  std::cout << "CudfHashJoinProbe constructor" << std::endl;
}

bool CudfHashJoinProbe::needsInput() const {
  std::cout << "Calling CudfHashJoinProbe::needsInput" << std::endl;
  return !finished_ && input_ == nullptr;
}
void CudfHashJoinProbe::addInput(RowVectorPtr input) {
  std::cout << "Calling CudfHashJoinProbe::addInput" << std::endl;
  input_ = std::move(input);
}

RowVectorPtr CudfHashJoinProbe::getOutput() {
  std::cout << "Calling CudfHashJoinProbe::getOutput" << std::endl;
  NVTX3_FUNC_RANGE();
  if (!input_) {
    return nullptr;
  }
  const auto inputSize = input_->size();
  if (!hashObject_.has_value()) {
    return nullptr;
  }
  // TODO convert input to cudf table
  auto tbl = to_cudf_table(input_);
  std::cout << "Probe table number of columns: " << tbl->num_columns()
            << std::endl;
  std::cout << "Probe table number of rows: " << tbl->num_rows() << std::endl;

  auto probeType = joinNode_->sources()[0]->outputType();
  auto buildType = joinNode_->sources()[1]->outputType();
  auto probeKeys = joinNode_->leftKeys();
  auto buildKeys = joinNode_->rightKeys();

  for (int i = 0; i < probeType->names().size(); i++) {
    std::cout << "Left column " << i << ": " << probeType->names()[i]
              << std::endl;
  }

  for (int i = 0; i < buildType->names().size(); i++) {
    std::cout << "Right column " << i << ": " << buildType->names()[i]
              << std::endl;
  }

  for (int i = 0; i < probeKeys.size(); i++) {
    std::cout << "Left key " << i << ": " << probeKeys[i]->name() << std::endl;
  }

  for (int i = 0; i < buildKeys.size(); i++) {
    std::cout << "Right key " << i << ": " << buildKeys[i]->name() << std::endl;
  }

  auto probe_key_indices = std::vector<cudf::size_type>(probeKeys.size());
  for (size_t i = 0; i < probe_key_indices.size(); i++) {
    probe_key_indices[i] = static_cast<cudf::size_type>(
        probeType->getChildIdx(probeKeys[i]->name()));
  }

  // TODO pass the input pool !!!
  // TODO: We should probably subset columns before calling to_cudf_table?
  // Maybe that isn't a problem if we fuse operators together.
  auto const [left_join_indices, right_join_indices] =
      hashObject_.value().second->inner_join(
          tbl->view().select(probe_key_indices));
  auto left_indices_span =
      cudf::device_span<cudf::size_type const>{*left_join_indices};
  auto right_indices_span =
      cudf::device_span<cudf::size_type const>{*right_join_indices};

  auto outputType = joinNode_->outputType();
  auto left_column_indices_to_gather = std::vector<cudf::size_type>();
  auto right_column_indices_to_gather = std::vector<cudf::size_type>();
  auto left_column_output_indices = std::vector<size_t>();
  auto right_column_output_indices = std::vector<size_t>();
  for (int i = 0; i < outputType->names().size(); i++) {
    auto const output_name = outputType->names()[i];
    std::cout << "Output column " << i << ": " << output_name << std::endl;
    auto channel = probeType->getChildIdxIfExists(output_name);
    if (channel.has_value()) {
      left_column_indices_to_gather.push_back(
          static_cast<cudf::size_type>(channel.value()));
      left_column_output_indices.push_back(i);
      continue;
    }
    channel = buildType->getChildIdxIfExists(output_name);
    if (channel.has_value()) {
      right_column_indices_to_gather.push_back(
          static_cast<cudf::size_type>(channel.value()));
      right_column_output_indices.push_back(i);
      continue;
    }
    VELOX_FAIL(
        "Join field {} not in probe or build input", outputType->children()[i]);
  }

  for (int i = 0; i < left_column_indices_to_gather.size(); i++) {
    std::cout << "Left index to gather " << i << ": "
              << left_column_indices_to_gather[i] << std::endl;
  }

  for (int i = 0; i < right_column_indices_to_gather.size(); i++) {
    std::cout << "Right index to gather " << i << ": "
              << right_column_indices_to_gather[i] << std::endl;
  }

  auto left_input = tbl->view().select(left_column_indices_to_gather);
  auto right_input =
      hashObject_.value().first->view().select(right_column_indices_to_gather);

  auto left_indices_col = cudf::column_view{left_indices_span};
  auto right_indices_col = cudf::column_view{right_indices_span};
  auto constexpr oob_policy = cudf::out_of_bounds_policy::DONT_CHECK;
  auto left_result = cudf::gather(left_input, left_indices_col, oob_policy);
  auto right_result = cudf::gather(right_input, right_indices_col, oob_policy);

  std::cout << "Left result number of columns: " << left_result->num_columns()
            << std::endl;
  std::cout << "Right result number of columns: " << right_result->num_columns()
            << std::endl;

  auto left_cols = left_result->release();
  auto right_cols = right_result->release();
  auto joined_cols =
      std::vector<std::unique_ptr<cudf::column>>(outputType->names().size());
  for (int i = 0; i < left_column_output_indices.size(); i++) {
    joined_cols[left_column_output_indices[i]] = std::move(left_cols[i]);
  }
  for (int i = 0; i < right_column_output_indices.size(); i++) {
    joined_cols[right_column_output_indices[i]] = std::move(right_cols[i]);
  }
  auto cudf_output = std::make_unique<cudf::table>(std::move(joined_cols));

  RowVectorPtr output;
  if (cudf_output->num_columns() == 0 or cudf_output->num_rows() == 0) {
    output = nullptr;
  } else {
    output = to_velox_column(cudf_output->view(), input_->pool());
  }

  input_.reset();
  finished_ = true;
  // printResults(output, std::cout);
  return output;
}

exec::BlockingReason CudfHashJoinProbe::isBlocked(ContinueFuture* future) {
  std::cout << "Calling CudfHashJoinProbe::isBlocked" << std::endl;
  if (hashObject_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto hashObject = std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge)
                        ->hashOrFuture(future);

  if (!hashObject.has_value()) {
    std::cout << "CudfHashJoinProbe is blocked, waiting for join build"
              << std::endl;
    return exec::BlockingReason::kWaitForJoinBuild;
  }
  hashObject_ = std::move(hashObject);

  return exec::BlockingReason::kNotBlocked;
}

bool CudfHashJoinProbe::isFinished() {
  std::cout << "Calling CudfHashJoinProbe::isFinished" << std::endl;
  return finished_ || (noMoreInput_ && input_ == nullptr);
}

std::unique_ptr<exec::Operator> CudfHashJoinBridgeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  std::cout << "Calling CudfHashJoinBridgeTranslator::toOperator" << std::endl;
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    return std::make_unique<CudfHashJoinProbe>(id, ctx, joinNode);
  }
  return nullptr;
}

std::unique_ptr<exec::JoinBridge> CudfHashJoinBridgeTranslator::toJoinBridge(
    const core::PlanNodePtr& node) {
  std::cout << "Calling CudfHashJoinBridgeTranslator::toJoinBridge"
            << std::endl;
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    auto joinBridge = std::make_unique<CudfHashJoinBridge>();
    return joinBridge;
  }
  return nullptr;
}

exec::OperatorSupplier CudfHashJoinBridgeTranslator::toOperatorSupplier(
    const core::PlanNodePtr& node) {
  std::cout << "Calling CudfHashJoinBridgeTranslator::toOperatorSupplier"
            << std::endl;
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    return [joinNode](int32_t operatorId, exec::DriverCtx* ctx) {
      return std::make_unique<CudfHashJoinBuild>(operatorId, ctx, joinNode);
    };
  }
  return nullptr;
}

} // namespace facebook::velox::cudf_velox
