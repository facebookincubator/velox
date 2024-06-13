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
#include "velox/exec/JoinBridge.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <cudf/join.hpp>
#include <cudf/copying.hpp>

#include <nvtx3/nvtx3.hpp>

#include "VeloxCudfInterop.h"
#include "CudfHashJoin.h"

namespace facebook::velox::cudf_velox {

CudfHashJoinNode::CudfHashJoinNode(
    const core::PlanNodeId& id,
    core::PlanNodePtr left,
    core::PlanNodePtr right)
    : PlanNode(id), sources_{std::move(left), std::move(right)} {}

const RowTypePtr& CudfHashJoinNode::outputType() const {
    // TODO similar to PlanBuilder::hashJoin()
    return sources_.front()->outputType();
}

const std::vector<core::PlanNodePtr>& CudfHashJoinNode::sources() const {
    return sources_;
}

std::string_view CudfHashJoinNode::name() const {
    return "CudfHashJoin";
}

void CudfHashJoinNode::addDetails(std::stringstream& /* stream */) const {}

void CudfHashJoinBridge::setHashTable(std::optional<CudfHashJoinBridge::hash_type> hashObject) {
    std::vector<ContinuePromise> promises;
    {
        std::lock_guard<std::mutex> l(mutex_);
        VELOX_CHECK(!hashObject_.has_value(), "CudfHashJoinBridge already has a hash table");
        hashObject_ = std::move(hashObject);
        promises = std::move(promises_);
    }
    notify(std::move(promises));
}

std::optional<CudfHashJoinBridge::hash_type> CudfHashJoinBridge::HashOrFuture(ContinueFuture* future) {
    std::lock_guard<std::mutex> l(mutex_);
    if (hashObject_.has_value()) {
        return std::move(hashObject_);
    }
    promises_.emplace_back("CudfHashJoinBridge::HashOrFuture");
    *future = promises_.back().getSemiFuture();
    return std::nullopt;
}

CudfHashJoinBuild::CudfHashJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const CudfHashJoinNode> joinNode)
    // TODO check outputType should be set or not?
    : exec::Operator(driverCtx,  nullptr, // joinNode->sources(),
    operatorId, joinNode->id(), "CudfHashJoinBuild") {}

void CudfHashJoinBuild::addInput(RowVectorPtr input) {
    // Queue inputs, process all at once.
    // TODO distribute work equally.
    auto inputSize = input->size();
    if (inputSize > 0) {
        inputs_.push_back(std::move(input));
    }
}

bool CudfHashJoinBuild::needsInput() const {
    return !noMoreInput_;
}

RowVectorPtr CudfHashJoinBuild::getOutput() {
    return nullptr;
}

void CudfHashJoinBuild::noMoreInput() {
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
        // numRows_ += build->numRows_;
        inputs_.insert(inputs_.end(), build->inputs_.begin(), build->inputs_.end());
    }
    // TODO build hash table
    auto tbl = to_cudf_table(inputs_[0]); // TODO how to process multiple inputs?
    // copy host to device table,
    // CudfHashJoinBridge::hash_type hashObject = 1;
    // TODO create hash table in device.
    // CudfHashJoinBridge::hash_type
    auto hashObject =
        std::make_shared<cudf::hash_join>(tbl->view(), cudf::null_equality::EQUAL);

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
    cudf_HashJoinBridge->setHashTable(std::make_optional(std::make_pair(std::move(tbl), std::move(hashObject))));
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
    std::shared_ptr<const CudfHashJoinNode> joinNode)
    : exec::Operator(driverCtx, nullptr, // joinNode->sources(),
    operatorId, joinNode->id(), "CudfHashJoinProbe") {}

bool CudfHashJoinProbe::needsInput() const {
    return !finished_ && input_ == nullptr;
}
void CudfHashJoinProbe::addInput(RowVectorPtr input) {
    input_ = std::move(input);
}

RowVectorPtr CudfHashJoinProbe::getOutput() {
    NVTX3_FUNC_RANGE();
    if (!input_) {
        return nullptr;
    }
    const auto inputSize = input_->size();
    if(!hashObject_.has_value()) {
        return nullptr;
    }
    // std::cout<<"here\n\n";
    // TODO convert input to cudf table
    auto tbl = to_cudf_table(input_);
    // TODO pass the input pool !!!
    RowVectorPtr output;
    // RowVectorPtr output;
    auto const [left_join_indices, right_join_indices] = hashObject_.value().second->inner_join(tbl->view());
    auto left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
    auto right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};
    auto left_input  = tbl->view();
    auto right_input = hashObject_.value().first->view();

    auto left_indices_col  = cudf::column_view{left_indices_span};
    auto right_indices_col = cudf::column_view{right_indices_span};
    auto constexpr oob_policy = cudf::out_of_bounds_policy::DONT_CHECK;
    auto left_result  = cudf::gather(left_input, left_indices_col, oob_policy);
    auto right_result = cudf::gather(right_input, right_indices_col, oob_policy);
    auto joined_cols = left_result->release();
    auto right_cols  = right_result->release();
    joined_cols.insert(joined_cols.end(),
                        std::make_move_iterator(right_cols.begin()),
                        std::make_move_iterator(right_cols.end()));
    auto cudf_output = std::make_unique<cudf::table>(std::move(joined_cols));
    // TODO convert output to RowVector
    if (cudf_output->num_columns() == 0 or cudf_output->num_rows() == 0) {
        output = nullptr;
    } else {
        output = to_velox_column(cudf_output->view(), input_->pool());
    }
    // auto output = input_;
    // auto output = std::make_shared<RowVector>(
    //     input_->pool(),
    //     input_->type(),
    //     input_->nulls(),
    //     std::min(20, inputSize-2),
    //     input_->children());
    // std::cout<<"there\n\n";
    input_.reset();
    finished_ = true;
    // printResults(output, std::cout);
    return output;
}

exec::BlockingReason CudfHashJoinProbe::isBlocked(ContinueFuture* future) {
    if (hashObject_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
    }

    auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());
    auto hashObject = std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge)
                    ->HashOrFuture(future);

    if (!hashObject.has_value()) {
    return exec::BlockingReason::kWaitForJoinBuild;
    }
    hashObject_ = std::move(hashObject);
    // remainingLimit_ = hashObject.value();

    return exec::BlockingReason::kNotBlocked;
}

bool CudfHashJoinProbe::isFinished() {
    return finished_ || (noMoreInput_ && input_ == nullptr);
}

std::unique_ptr<exec::Operator> CudfHashJoinBridgeTranslator::toOperator(exec::DriverCtx* ctx, int32_t id, const core::PlanNodePtr& node) {
    std::cout << "Calling CudfHashJoinBridgeTranslator::toOperator" << std::endl;
    if (auto joinNode = std::dynamic_pointer_cast<const CudfHashJoinNode>(node)) {
        return std::make_unique<CudfHashJoinProbe>(id, ctx, joinNode);
    }
    return nullptr;
}

std::unique_ptr<exec::JoinBridge> CudfHashJoinBridgeTranslator::toJoinBridge(const core::PlanNodePtr& node) {
    std::cout << "Calling CudfHashJoinBridgeTranslator::toJoinBridge" << std::endl;
    if (auto joinNode = std::dynamic_pointer_cast<const CudfHashJoinNode>(node)) {
        auto joinBridge = std::make_unique<CudfHashJoinBridge>();
        return joinBridge;
    }
    return nullptr;
}

exec::OperatorSupplier CudfHashJoinBridgeTranslator::toOperatorSupplier(const core::PlanNodePtr& node) {
    std::cout << "Calling CudfHashJoinBridgeTranslator::toOperatorSupplier" << std::endl;
    if (auto joinNode = std::dynamic_pointer_cast<const CudfHashJoinNode>(node)) {
        return [joinNode](int32_t operatorId, exec::DriverCtx* ctx) {
            return std::make_unique<CudfHashJoinBuild>(operatorId, ctx, joinNode);
        };
    }
    return nullptr;
}

} // namespace facebook::velox::cudf_velox