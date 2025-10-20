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
#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/exec/RowVectorSource.h"

namespace facebook::velox::exec {

class MemorySource : public SourceOperator {
 public:
  MemorySource(
      int32_t operatorId,
      DriverCtx* driverCtx,
      std::shared_ptr<const core::MemorySourceNode> node);

  void initialize() override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 private:
  const std::shared_ptr<const core::MemorySourceNode> node_;
  RowVectorSource* source_;
  bool finished_{false};
  ContinueFuture blockingFuture_{ContinueFuture::makeEmpty()};
  BlockingReason blockingReason_{BlockingReason::kNotBlocked};
};

} // namespace facebook::velox::exec
