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

/// Shared lifecycle of the cuDF hash and nested loop join builds. Each build
/// driver accumulates its non-empty input batches. In noMoreInput(), the last
/// driver to arrive at the allPeersArrived() barrier moves the inputs of all
/// peer drivers into its own, releases the peers and calls buildAndPublish().
/// The other drivers wait at the barrier until they are released. getOutput()
/// always returns nullptr. A driver is finished once noMoreInput() was called
/// and it no longer waits at the barrier.
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

  // Records per-input statistics before accumulation; no-op by default.
  virtual void recordInputStats(const CudfVector& input) {}

  // Consumes accumulated inputs and publishes the build result to the bridge.
  // Called exactly once per split group, on the driver elected by
  // allPeersArrived(), with the inputs of every build driver. The peer drivers
  // have already been released, so implementations must not access them.
  virtual void buildAndPublish(std::vector<CudfVectorPtr> inputs) = 0;

 private:
  // Build inputs accumulated by this driver and transferred from peer drivers.
  std::vector<CudfVectorPtr> inputs_;

  // Waits for the last peer driver to finish collecting build inputs.
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

} // namespace facebook::velox::cudf_velox
