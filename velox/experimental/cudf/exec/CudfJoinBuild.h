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

#pragma once

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <memory>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Common input accumulation and peer coordination for cuDF join builds.
class CudfJoinBuild : public CudfOperatorBase {
 public:
  bool needsInput() const final;

  exec::BlockingReason isBlocked(ContinueFuture* future) final;

  bool isFinished() final;

 protected:
  CudfJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::PlanNode>& joinNode,
      const std::string& operatorName,
      NvtxMethodFlag nvtxMethods);

  void doAddInput(RowVectorPtr input) final;
  RowVectorPtr doGetOutput() final;
  void doNoMoreInput() final;
  void doClose() final;

  virtual void recordInputStats(const CudfVector& input) {}

  virtual void buildAndPublish(std::vector<CudfVectorPtr> inputs) = 0;

 private:
  std::vector<CudfVectorPtr> inputs_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

} // namespace facebook::velox::cudf_velox
