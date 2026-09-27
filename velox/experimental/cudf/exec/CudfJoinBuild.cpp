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

#include "velox/experimental/cudf/exec/CudfJoinBuild.h"

#include "velox/common/testutil/TestValue.h"
#include "velox/exec/Task.h"

#include <folly/ScopeGuard.h>

#include <iterator>
#include <typeinfo>
#include <utility>

namespace facebook::velox::cudf_velox {

CudfJoinBuild::CudfJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::PlanNode>& joinNode,
    const std::string& operatorName,
    NvtxMethodFlag nvtxMethods)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          nullptr,
          joinNode->id(),
          operatorName,
          nvtx3::rgb{65, 105, 225},
          nvtxMethods,
          std::nullopt,
          joinNode) {}

bool CudfJoinBuild::needsInput() const {
  return !noMoreInput_;
}

exec::BlockingReason CudfJoinBuild::isBlocked(ContinueFuture* future) {
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

void CudfJoinBuild::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  recordInputStats(*cudfInput);
  inputs_.push_back(std::move(cudfInput));
}

RowVectorPtr CudfJoinBuild::doGetOutput() {
  return nullptr;
}

void CudfJoinBuild::doNoMoreInput() {
  Operator::noMoreInput();

  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }

  SCOPE_EXIT {
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  for (auto& peer : peers) {
    auto* build =
        dynamic_cast<CudfJoinBuild*>(peer->findOperator(planNodeId()));
    VELOX_CHECK_NOT_NULL(build);
    VELOX_CHECK(
        typeid(*build) == typeid(*this),
        "Expected peer build type {}, got {}",
        typeid(*this).name(),
        typeid(*build).name());
    inputs_.insert(
        inputs_.end(),
        std::make_move_iterator(build->inputs_.begin()),
        std::make_move_iterator(build->inputs_.end()));
    build->inputs_.clear();

    auto retainedInputBatches = build->inputs_.size();
    common::testutil::TestValue::adjust(
        "facebook::velox::cudf_velox::CudfJoinBuild::doNoMoreInput::sourceDriverRetainedInputBatchesAfterTransfer",
        &retainedInputBatches);
  }

  buildAndPublish(std::exchange(inputs_, {}));
}

void CudfJoinBuild::doClose() {
  inputs_.clear();
  Operator::close();
}

} // namespace facebook::velox::cudf_velox
