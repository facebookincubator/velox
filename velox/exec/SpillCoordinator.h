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

#include <folly/container/F14Set.h>
#include <stdint.h>
#include <functional>
#include "velox/core/PlanNode.h"

namespace facebook::velox::exec {

class Operator;

/// Used to coordinate the disk spill operation among multiple drivers of the
/// same node type within a task (split group). Take hash build for example,
/// when any one of the build driver tries to spill, then all the drivers will
/// stop execution for spilling, and one of them will be selected by the
/// coordinator to select a set of partitions to spill them for all the drivers.
/// The driver executions will be resumed after the coordinated spilling
/// finishes.
class SpillCoordinator {
 public:
  using SpillRunner = std::function<void(const std::vector<Operator*>& ops)>;

  SpillCoordinator(
      const std::string taskId,
      const uint32_t splitGroupId,
      const core::PlanNodeId& planNodeId)
      : taskId_(taskId), splitGroupId_(splitGroupId), planNodeId_(planNodeId){};

  void join(Operator* FOLLY_NONNULL op, SpillRunner spillRunner);

  bool hasPendingSpill() const {
    return hasPendingSpill_;
  }

  /// Invoked to request a new spill operation. The function returns true if the
  /// driver needs to wait for spill to run, otherwise the spill has been
  /// executed by this function and returns false.
  bool requestSpill(
      Operator* FOLLY_NONNULL op,
      ContinueFuture* FOLLY_NONNULL future);

  /// Invoked to check if a driver needs to wait for any pending spill to run.
  /// The function returns true if the driver needs to wait, otherwise it
  /// returns false. The latter is either because there is no pending spill or
  /// 'driver' is the last one to reaches to the spill barrier and has executed
  /// the spill as the coordinator.
  bool waitSpill(
      Operator* FOLLY_NONNULL op,
      ContinueFuture* FOLLY_NONNULL future);

  /// Invoked to indicate a driver won't trigger any spill request anymore and
  /// it will also not involve in the future spill coordination. The function
  /// will put this 'driver' into 'noMoreSpillOperators_'. If there is a pending
  /// spill request, then this function will also help to check if it is ready
  /// to run.
  void noMoreSpill(Operator* FOLLY_NONNULL op);

  void reset();

 private:
  // Return the number of spilling operators which excludes the operator won't
  // trigger spilling anymore.
  uint32_t numSpillingOperatorsLocked() const {
    return operators_.size() - noMoreSpillOperators_.size();
  }

  // Invoked to execute spill runner callback and resumes the driver executions
  // after that by fulfilling 'promises'.
  void runSpill(std::vector<ContinuePromise>& promises);

  bool waitSpillLocked(
      Operator* FOLLY_NONNULL op,
      std::vector<ContinuePromise>& promises,
      ContinueFuture* FOLLY_NONNULL future);

  const std::string taskId_;
  const uint32_t splitGroupId_;
  const core::PlanNodeId& planNodeId_;

  std::mutex mutex_;

  std::vector<Operator*> operators_;
  SpillRunner spillRunner_;

  // Indicates if there is a pending spill to run.
  std::atomic<bool> hasPendingSpill_{false};

  // Counts the number of operators to wait for the pending spill. Once the
  // counter matches the number of spilling operators, then the last wait driver
  // will act as the coordinator to run the spill in its driver thread.
  int numWaitingOperators_{0};

  // The operators won't trigger any spilling operations such as a hash build
  // driver which has finished the table build. It will be cleared by reset().
  folly::F14FastSet<Operator*> noMoreSpillOperators_;

  std::vector<ContinuePromise> promises_;
};

} // namespace facebook::velox::exec
