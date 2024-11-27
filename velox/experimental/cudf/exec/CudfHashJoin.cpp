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

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/join.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtx3/nvtx3.hpp>

#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace facebook::velox::cudf_velox {

void CudfHashJoinBridge::setHashTable(
    std::optional<CudfHashJoinBridge::hash_type> hashObject) {
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinBridge::setHashTable" << std::endl;
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
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinBridge::hashOrFuture" << std::endl;
  }
  std::lock_guard<std::mutex> l(mutex_);
  if (hashObject_.has_value()) {
    return hashObject_;
  }
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinBridge::hashOrFuture constructing promise"
              << std::endl;
  }
  promises_.emplace_back("CudfHashJoinBridge::hashOrFuture");
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinBridge::hashOrFuture getSemiFuture"
              << std::endl;
  }
  *future = promises_.back().getSemiFuture();
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinBridge::hashOrFuture returning nullopt"
              << std::endl;
  }
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
  if (cudfDebugEnabled()) {
    std::cout << "CudfHashJoinBuild constructor" << std::endl;
  }
}

void CudfHashJoinBuild::addInput(RowVectorPtr input) {
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinBuild::addInput" << std::endl;
  }
  // Queue inputs, process all at once.
  if (input->size() > 0) {
    auto cudf_input = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudf_input);
    inputs_.push_back(std::move(cudf_input));
  }
}

bool CudfHashJoinBuild::needsInput() const {
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinBuild::needsInput" << std::endl;
  }
  return !noMoreInput_;
}

RowVectorPtr CudfHashJoinBuild::getOutput() {
  return nullptr;
}

void CudfHashJoinBuild::noMoreInput() {
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinBuild::noMoreInput" << std::endl;
  }
  NVTX3_FUNC_RANGE();
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

  auto cudf_tables = std::vector<std::unique_ptr<cudf::table>>(inputs_.size());
  auto cudf_table_views = std::vector<cudf::table_view>(inputs_.size());
  for (int i = 0; i < inputs_.size(); i++) {
    VELOX_CHECK_NOT_NULL(inputs_[i]);
    cudf_tables[i] = inputs_[i]->release();
    cudf_table_views[i] = cudf_tables[i]->view();
  }
  auto tbl = cudf::concatenate(cudf_table_views);

  // Release input data
  cudf::get_default_stream().synchronize();
  cudf_table_views.clear();
  cudf_tables.clear();
  inputs_.clear();

  VELOX_CHECK_NOT_NULL(tbl);
  if (cudfDebugEnabled()) {
    std::cout << "Build table number of columns: " << tbl->num_columns()
              << std::endl;
    std::cout << "Build table number of rows: " << tbl->num_rows() << std::endl;
  }

  auto buildType = joinNode_->sources()[1]->outputType();
  auto buildKeys = joinNode_->rightKeys();

  auto build_key_indices = std::vector<cudf::size_type>(buildKeys.size());
  for (size_t i = 0; i < build_key_indices.size(); i++) {
    build_key_indices[i] = static_cast<cudf::size_type>(
        buildType->getChildIdx(buildKeys[i]->name()));
  }

  auto hashObject = std::make_shared<cudf::hash_join>(
      tbl->view().select(build_key_indices), cudf::null_equality::EQUAL);
  VELOX_CHECK_NOT_NULL(hashObject);
  if (cudfDebugEnabled()) {
    if (hashObject != nullptr) {
      printf("hashObject is not nullptr %p\n", hashObject.get());
    } else {
      printf("hashObject is *** nullptr\n");
    }
  }

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
      std::make_pair(std::shared_ptr(std::move(tbl)), std::move(hashObject))));
}

exec::BlockingReason CudfHashJoinBuild::isBlocked(ContinueFuture* future) {
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinBuild::isBlocked" << std::endl;
  }
  if (!future_.valid()) {
    if (cudfDebugEnabled()) {
      std::cout << "CudfHashJoinBuild future is not valid" << std::endl;
    }
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfHashJoinBuild::isFinished() {
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinBuild::isFinished" << std::endl;
  }
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
  if (cudfDebugEnabled()) {
    std::cout << "CudfHashJoinProbe constructor" << std::endl;
  }
}

bool CudfHashJoinProbe::needsInput() const {
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinProbe::needsInput" << std::endl;
  }
  return !finished_ && input_ == nullptr;
}

void CudfHashJoinProbe::addInput(RowVectorPtr input) {
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinProbe::addInput" << std::endl;
  }
  input_ = std::move(input);
}

RowVectorPtr CudfHashJoinProbe::getOutput() {
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinProbe::getOutput" << std::endl;
  }
  NVTX3_FUNC_RANGE();
  if (!input_) {
    return nullptr;
  }
  if (!hashObject_.has_value()) {
    return nullptr;
  }
  auto cudf_input = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudf_input);
  auto tbl = cudf_input->release();
  if (cudfDebugEnabled()) {
    std::cout << "Probe table number of columns: " << tbl->num_columns()
              << std::endl;
    std::cout << "Probe table number of rows: " << tbl->num_rows() << std::endl;
  }

  auto probeType = joinNode_->sources()[0]->outputType();
  auto buildType = joinNode_->sources()[1]->outputType();
  auto probeKeys = joinNode_->leftKeys();
  auto buildKeys = joinNode_->rightKeys();

  if (cudfDebugEnabled()) {
    for (int i = 0; i < probeType->names().size(); i++) {
      std::cout << "Left column " << i << ": " << probeType->names()[i]
                << std::endl;
    }

    for (int i = 0; i < buildType->names().size(); i++) {
      std::cout << "Right column " << i << ": " << buildType->names()[i]
                << std::endl;
    }

    for (int i = 0; i < probeKeys.size(); i++) {
      std::cout << "Left key " << i << ": " << probeKeys[i]->name() << " "
                << probeKeys[i]->type()->kind() << std::endl;
    }

    for (int i = 0; i < buildKeys.size(); i++) {
      std::cout << "Right key " << i << ": " << buildKeys[i]->name() << " "
                << buildKeys[i]->type()->kind() << std::endl;
    }
  }

  auto probe_key_indices = std::vector<cudf::size_type>(probeKeys.size());
  for (size_t i = 0; i < probe_key_indices.size(); i++) {
    probe_key_indices[i] = static_cast<cudf::size_type>(
        probeType->getChildIdx(probeKeys[i]->name()));
  }

  // TODO pass the input pool !!!
  // TODO: We should probably subset columns before calling to_cudf_table?
  // Maybe that isn't a problem if we fuse operators together.
  auto& tb = hashObject_.value().first;
  auto& hb = hashObject_.value().second;
  VELOX_CHECK_NOT_NULL(tb);
  VELOX_CHECK_NOT_NULL(hb);
  if (cudfDebugEnabled()) {
    if (tb != nullptr)
      printf(
          "tb is not nullptr %p hasValue(%d)\n",
          tb.get(),
          hashObject_.has_value());
    if (hb != nullptr)
      printf(
          "hb is not nullptr %p hasValue(%d)\n",
          hb.get(),
          hashObject_.has_value());
  }
  auto const [left_join_indices, right_join_indices] =
      hb->inner_join(tbl->view().select(probe_key_indices));
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
    if (cudfDebugEnabled()) {
      std::cout << "Output column " << i << ": " << output_name << std::endl;
    }
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

  if (cudfDebugEnabled()) {
    for (int i = 0; i < left_column_indices_to_gather.size(); i++) {
      std::cout << "Left index to gather " << i << ": "
                << left_column_indices_to_gather[i] << std::endl;
    }

    for (int i = 0; i < right_column_indices_to_gather.size(); i++) {
      std::cout << "Right index to gather " << i << ": "
                << right_column_indices_to_gather[i] << std::endl;
    }
  }

  auto left_input = tbl->view().select(left_column_indices_to_gather);
  auto right_input =
      hashObject_.value().first->view().select(right_column_indices_to_gather);

  auto left_indices_col = cudf::column_view{left_indices_span};
  auto right_indices_col = cudf::column_view{right_indices_span};
  auto constexpr oob_policy = cudf::out_of_bounds_policy::DONT_CHECK;
  auto left_result = cudf::gather(left_input, left_indices_col, oob_policy);
  auto right_result = cudf::gather(right_input, right_indices_col, oob_policy);

  if (cudfDebugEnabled()) {
    std::cout << "Left result number of columns: " << left_result->num_columns()
              << std::endl;
    std::cout << "Right result number of columns: "
              << right_result->num_columns() << std::endl;
  }

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

  input_.reset();
  finished_ = noMoreInput_;

  auto const size = cudf_output->num_rows();
  if (cudf_output->num_columns() == 0 or size == 0) {
    return nullptr;
  }
  return std::make_shared<CudfVector>(
      pool(), outputType, size, std::move(cudf_output));
}

exec::BlockingReason CudfHashJoinProbe::isBlocked(ContinueFuture* future) {
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinProbe::isBlocked" << std::endl;
  }
  if (hashObject_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudf_joinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(cudf_joinBridge);
  VELOX_CHECK_NOT_NULL(future);
  auto hashObject = cudf_joinBridge->hashOrFuture(future);

  if (!hashObject.has_value()) {
    if (cudfDebugEnabled()) {
      std::cout << "CudfHashJoinProbe is blocked, waiting for join build"
                << std::endl;
    }
    return exec::BlockingReason::kWaitForJoinBuild;
  }
  hashObject_ = std::move(hashObject);

  return exec::BlockingReason::kNotBlocked;
}

bool CudfHashJoinProbe::isFinished() {
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinProbe::isFinished" << std::endl;
  }
  auto const is_finished = finished_ || (noMoreInput_ && input_ == nullptr);

  // Release hashObject_ if finished
  if (is_finished) {
    hashObject_.reset();
  }
  return is_finished;
}

std::unique_ptr<exec::Operator> CudfHashJoinBridgeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinBridgeTranslator::toOperator"
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
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinBridgeTranslator::toJoinBridge"
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
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinBridgeTranslator::toOperatorSupplier"
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
