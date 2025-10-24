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
#include "velox/exec/MemorySource.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/RowVectorSource.h"

namespace facebook::velox::exec {

MemorySource::MemorySource(
    int32_t operatorId,
    DriverCtx* driverCtx,
    std::shared_ptr<const core::MemorySourceNode> node)
    : SourceOperator(
          driverCtx,
          node->outputType(),
          operatorId,
          node->id(),
          "MemorySource"),
      node_(std::move(node)) {}

void MemorySource::initialize() {
  Operator::initialize();
  source_ = reinterpret_cast<RowVectorSource*>(node_->handler());
  VELOX_CHECK_NOT_NULL(source_);
  VELOX_CHECK_EQ(source_->name(), "LocalRowVectorSource");
}

RowVectorPtr MemorySource::getOutput() {
  VELOX_CHECK(!blockingFuture_.valid());
  blockingReason_ = BlockingReason::kNotBlocked;
  if (FOLLY_UNLIKELY(finished_)) {
    return nullptr;
  }

  RowVectorPtr output;
  blockingReason_ = source_->next(output, &blockingFuture_);
  if (output == nullptr) {
    finished_ = blockingReason_ == BlockingReason::kNotBlocked;
    VELOX_CHECK(finished_ != blockingFuture_.valid());
  }
  return output;
}

BlockingReason MemorySource::isBlocked(ContinueFuture* future) {
  if (blockingFuture_.valid()) {
    *future = std::move(blockingFuture_);
    return blockingReason_;
  }
  return BlockingReason::kNotBlocked;
}

bool MemorySource::isFinished() {
  return finished_;
}

} // namespace facebook::velox::exec
