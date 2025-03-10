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
#include "velox/expression/FieldReference.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/join.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtx3/nvtx3.hpp>

#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/ExpressionEvaluator.h"
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
      NvtxHelper(nvtx3::rgb{65, 105, 225}, operatorId), // Royal Blue
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

  auto cudf_tables = std::vector<std::unique_ptr<cudf::table>>(inputs_.size());
  auto input_streams = std::vector<rmm::cuda_stream_view>(inputs_.size());
  for (int i = 0; i < inputs_.size(); i++) {
    VELOX_CHECK_NOT_NULL(inputs_[i]);
    input_streams[i] = inputs_[i]->stream();
    cudf_tables[i] = inputs_[i]->release();
  }
  auto stream = cudfGlobalStreamPool().get_stream();
  cudf::detail::join_streams(input_streams, stream);
  auto tbl = concatenateTables(std::move(cudf_tables), stream);

  // Release input data after synchronizing
  stream.synchronize();
  input_streams.clear();
  cudf_tables.clear();

  // Release input data
  inputs_.clear();

  VELOX_CHECK_NOT_NULL(tbl);
  if (cudfDebugEnabled()) {
    std::cout << "Build table number of columns: " << tbl->num_columns()
              << std::endl;
    std::cout << "Build table number of rows: " << tbl->num_rows() << std::endl;
  }

  auto buildType = joinNode_->sources()[1]->outputType();
  auto rightKeys = joinNode_->rightKeys();

  auto build_key_indices = std::vector<cudf::size_type>(rightKeys.size());
  for (size_t i = 0; i < build_key_indices.size(); i++) {
    build_key_indices[i] = static_cast<cudf::size_type>(
        buildType->getChildIdx(rightKeys[i]->name()));
  }

  // Only need to construct hash_join object if it's an inner join or left join
  // and doesn't have a filter. All other cases use a standalone function in
  // cudf
  bool buildHashJoin = (joinNode_->isInnerJoin() || joinNode_->isLeftJoin()) &&
      !joinNode_->filter();
  auto hashObject = (buildHashJoin) ? std::make_shared<cudf::hash_join>(
                                          tbl->view().select(build_key_indices),
                                          cudf::null_equality::EQUAL,
                                          stream)
                                    : nullptr;
  if (buildHashJoin) {
    VELOX_CHECK_NOT_NULL(hashObject);
  }

  if (cudfDebugEnabled()) {
    if (hashObject != nullptr) {
      printf("hashObject is not nullptr %p\n", hashObject.get());
    } else {
      printf("hashObject is *** nullptr\n");
    }
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
      NvtxHelper(nvtx3::rgb{0, 128, 128}, operatorId), // Teal
      joinNode_(joinNode) {
  if (cudfDebugEnabled()) {
    std::cout << "CudfHashJoinProbe constructor" << std::endl;
  }
  auto probeType = joinNode_->sources()[0]->outputType();
  auto buildType = joinNode_->sources()[1]->outputType();
  auto const& leftKeys = joinNode_->leftKeys(); // probe keys
  auto const& rightKeys = joinNode_->rightKeys(); // build keys

  if (cudfDebugEnabled()) {
    for (int i = 0; i < probeType->names().size(); i++) {
      std::cout << "Left column " << i << ": " << probeType->names()[i]
                << std::endl;
    }

    for (int i = 0; i < buildType->names().size(); i++) {
      std::cout << "Right column " << i << ": " << buildType->names()[i]
                << std::endl;
    }

    for (int i = 0; i < leftKeys.size(); i++) {
      std::cout << "Left key " << i << ": " << leftKeys[i]->name() << " "
                << leftKeys[i]->type()->kind() << std::endl;
    }

    for (int i = 0; i < rightKeys.size(); i++) {
      std::cout << "Right key " << i << ": " << rightKeys[i]->name() << " "
                << rightKeys[i]->type()->kind() << std::endl;
    }
  }

  auto const probe_table_num_columns = probeType->size();
  left_key_indices_ = std::vector<cudf::size_type>(leftKeys.size());
  for (size_t i = 0; i < left_key_indices_.size(); i++) {
    left_key_indices_[i] = static_cast<cudf::size_type>(
        probeType->getChildIdx(leftKeys[i]->name()));
    VELOX_CHECK_LT(left_key_indices_[i], probe_table_num_columns);
  }
  auto const build_table_num_columns = buildType->size();
  right_key_indices_ = std::vector<cudf::size_type>(rightKeys.size());
  for (size_t i = 0; i < right_key_indices_.size(); i++) {
    right_key_indices_[i] = static_cast<cudf::size_type>(
        buildType->getChildIdx(rightKeys[i]->name()));
    VELOX_CHECK_LT(right_key_indices_[i], build_table_num_columns);
  }

  auto outputType = joinNode_->outputType();
  left_column_indices_to_gather_ = std::vector<cudf::size_type>();
  right_column_indices_to_gather_ = std::vector<cudf::size_type>();
  left_column_output_indices_ = std::vector<size_t>();
  right_column_output_indices_ = std::vector<size_t>();
  for (int i = 0; i < outputType->names().size(); i++) {
    auto const output_name = outputType->names()[i];
    if (cudfDebugEnabled()) {
      std::cout << "Output column " << i << ": " << output_name << std::endl;
    }
    auto channel = probeType->getChildIdxIfExists(output_name);
    if (channel.has_value()) {
      left_column_indices_to_gather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      left_column_output_indices_.push_back(i);
      continue;
    }
    channel = buildType->getChildIdxIfExists(output_name);
    if (channel.has_value()) {
      right_column_indices_to_gather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      right_column_output_indices_.push_back(i);
      continue;
    }
    VELOX_FAIL(
        "Join field {} not in probe or build input", outputType->children()[i]);
  }

  if (cudfDebugEnabled()) {
    for (int i = 0; i < left_column_indices_to_gather_.size(); i++) {
      std::cout << "Left index to gather " << i << ": "
                << left_column_indices_to_gather_[i] << std::endl;
    }

    for (int i = 0; i < right_column_indices_to_gather_.size(); i++) {
      std::cout << "Right index to gather " << i << ": "
                << right_column_indices_to_gather_[i] << std::endl;
    }
  }

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
    if (joinNode_->isRightSemiFilterJoin()) {
      create_ast_tree(
          exprs.exprs()[0],
          tree_,
          scalars_,
          buildType,
          probeType,
          right_precompute_instructions_,
          left_precompute_instructions_);
    } else {
      create_ast_tree(
          exprs.exprs()[0],
          tree_,
          scalars_,
          probeType,
          buildType,
          left_precompute_instructions_,
          right_precompute_instructions_);
    }
  }
}

bool CudfHashJoinProbe::needsInput() const {
  return !finished_ && input_ == nullptr;
}

void CudfHashJoinProbe::addInput(RowVectorPtr input) {
  input_ = std::move(input);
}

RowVectorPtr CudfHashJoinProbe::getOutput() {
  if (cudfDebugEnabled()) {
    std::cout << "Calling CudfHashJoinProbe::getOutput" << std::endl;
  }
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (!input_) {
    return nullptr;
  }
  if (!hashObject_.has_value()) {
    return nullptr;
  }
  auto cudf_input = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudf_input);
  auto stream = cudf_input->stream();
  auto left_table = cudf_input->release(); // probe table
  if (cudfDebugEnabled()) {
    std::cout << "Probe table number of columns: " << left_table->num_columns()
              << std::endl;
    std::cout << "Probe table number of rows: " << left_table->num_rows()
              << std::endl;
  }

  // TODO pass the input pool !!!
  // TODO: We should probably subset columns before calling to_cudf_table?
  // Maybe that isn't a problem if we fuse operators together.
  auto& right_table = hashObject_.value().first;
  auto& hb = hashObject_.value().second;
  VELOX_CHECK_NOT_NULL(right_table);
  if (cudfDebugEnabled()) {
    if (right_table != nullptr)
      printf(
          "right_table is not nullptr %p hasValue(%d)\n",
          right_table.get(),
          hashObject_.has_value());
    if (hb != nullptr)
      printf(
          "hb is not nullptr %p hasValue(%d)\n",
          hb.get(),
          hashObject_.has_value());
  }

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_join_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_join_indices;

  auto left_table_view = left_table->view();
  auto right_table_view = right_table->view();

  // // TODO (dm): Check if releasing the tables affects the table views we use
  // // later in the call
  // auto left_input_cols = left_table->release();

  // // TODO (dm): refactor
  // // right table is precomputed only on first call to probe side. Make it so
  // // that right table is precomputed on build side.
  // if (joinNode_->filter()) {
  //   addPrecomputedColumns(
  //       left_input_cols, left_precompute_instructions_, scalars_, stream);
  //   if (!right_precomputed_) {
  //     auto right_input_cols = right_table->release();
  //     addPrecomputedColumns(
  //         right_input_cols, right_precompute_instructions_, scalars_,
  //         stream);
  //     right_table =
  //     std::make_unique<cudf::table>(std::move(right_input_cols));
  //     right_precomputed_ = true;
  //   }
  // }
  // // expression cols need to be reassembled into the table views
  // cudf::table left_table_for_exprs(std::move(left_input_cols));

  if (joinNode_->isInnerJoin()) {
    // left = probe, right = build
    if (joinNode_->filter()) {
      std::tie(left_join_indices, right_join_indices) = mixed_inner_join(
          left_table_view.select(left_key_indices_),
          right_table_view.select(right_key_indices_),
          left_table_view,
          right_table_view,
          tree_.back(),
          cudf::null_equality::EQUAL,
          std::nullopt,
          stream);
    } else {
      VELOX_CHECK_NOT_NULL(hb);
      std::tie(left_join_indices, right_join_indices) = hb->inner_join(
          left_table_view.select(left_key_indices_), std::nullopt, stream);
    }
  } else if (joinNode_->isLeftJoin()) {
    if (joinNode_->filter()) {
      std::tie(left_join_indices, right_join_indices) = cudf::mixed_left_join(
          left_table_view.select(left_key_indices_),
          right_table_view.select(right_key_indices_),
          left_table_view,
          right_table_view,
          tree_.back(),
          cudf::null_equality::EQUAL,
          std::nullopt,
          stream);
    } else {
      VELOX_CHECK_NOT_NULL(hb);
      std::tie(left_join_indices, right_join_indices) = hb->left_join(
          left_table_view.select(left_key_indices_), std::nullopt, stream);
    }
  } else if (joinNode_->isRightJoin()) {
    std::tie(right_join_indices, left_join_indices) = cudf::left_join(
        right_table_view.select(right_key_indices_),
        left_table_view.select(left_key_indices_),
        cudf::null_equality::EQUAL,
        stream,
        cudf::get_current_device_resource_ref());
  } else if (joinNode_->isAntiJoin()) {
    if (joinNode_->filter()) {
      left_join_indices = cudf::mixed_left_anti_join(
          left_table_view.select(left_key_indices_),
          right_table_view.select(right_key_indices_),
          left_table_view,
          right_table_view,
          tree_.back(),
          cudf::null_equality::EQUAL,
          stream,
          cudf::get_current_device_resource_ref());
    } else {
      left_join_indices = cudf::left_anti_join(
          left_table_view.select(left_key_indices_),
          right_table_view.select(right_key_indices_),
          cudf::null_equality::EQUAL,
          stream,
          cudf::get_current_device_resource_ref());
    }
  } else if (joinNode_->isLeftSemiFilterJoin()) {
    if (joinNode_->filter()) {
      left_join_indices = cudf::mixed_left_semi_join(
          left_table_view.select(left_key_indices_),
          right_table_view.select(right_key_indices_),
          left_table_view,
          right_table_view,
          tree_.back(),
          cudf::null_equality::EQUAL,
          stream,
          cudf::get_current_device_resource_ref());
    } else {
      left_join_indices = cudf::left_semi_join(
          left_table_view.select(left_key_indices_),
          right_table_view.select(right_key_indices_),
          cudf::null_equality::EQUAL,
          stream,
          cudf::get_current_device_resource_ref());
    }
  } else if (joinNode_->isRightSemiFilterJoin()) {
    if (joinNode_->filter()) {
      right_join_indices = cudf::mixed_left_semi_join(
          right_table_view.select(right_key_indices_),
          left_table_view.select(left_key_indices_),
          right_table_view,
          left_table_view,
          tree_.back(),
          cudf::null_equality::EQUAL,
          stream,
          cudf::get_current_device_resource_ref());
    } else {
      right_join_indices = cudf::left_semi_join(
          right_table_view.select(right_key_indices_),
          left_table_view.select(left_key_indices_),
          cudf::null_equality::EQUAL,
          stream,
          cudf::get_current_device_resource_ref());
    }
  } else {
    VELOX_FAIL("Unsupported join type: ", joinNode_->joinType());
  }
  auto left_indices_span = left_join_indices
      ? cudf::device_span<cudf::size_type const>{*left_join_indices}
      : cudf::device_span<cudf::size_type const>{};
  auto right_indices_span = right_join_indices
      ? cudf::device_span<cudf::size_type const>{*right_join_indices}
      : cudf::device_span<cudf::size_type const>{};

  auto left_input = left_table_view.select(left_column_indices_to_gather_);
  auto right_input = right_table_view.select(right_column_indices_to_gather_);

  auto left_indices_col = cudf::column_view{left_indices_span};
  auto right_indices_col = cudf::column_view{right_indices_span};
  auto constexpr oob_policy = cudf::out_of_bounds_policy::NULLIFY;
  auto left_result =
      cudf::gather(left_input, left_indices_col, oob_policy, stream);
  auto right_result =
      cudf::gather(right_input, right_indices_col, oob_policy, stream);

  if (cudfDebugEnabled()) {
    std::cout << "Left result number of columns: " << left_result->num_columns()
              << std::endl;
    std::cout << "Right result number of columns: "
              << right_result->num_columns() << std::endl;
  }

  auto left_cols = left_result->release();
  auto right_cols = right_result->release();
  auto joined_cols =
      std::vector<std::unique_ptr<cudf::column>>(outputType_->names().size());
  for (int i = 0; i < left_column_output_indices_.size(); i++) {
    joined_cols[left_column_output_indices_[i]] = std::move(left_cols[i]);
  }
  for (int i = 0; i < right_column_output_indices_.size(); i++) {
    joined_cols[right_column_output_indices_[i]] = std::move(right_cols[i]);
  }
  auto cudf_output = std::make_unique<cudf::table>(std::move(joined_cols));
  stream.synchronize();

  input_.reset();
  finished_ = noMoreInput_;

  auto const size = cudf_output->num_rows();
  if (cudf_output->num_columns() == 0 or size == 0) {
    return nullptr;
  }
  return std::make_shared<CudfVector>(
      pool(), outputType_, size, std::move(cudf_output), stream);
}

exec::BlockingReason CudfHashJoinProbe::isBlocked(ContinueFuture* future) {
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
