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

#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/runner/ExecutablePlan.h"

namespace facebook::velox::exec::test {

class DistributedPlanBuilder : public PlanBuilder {
 public:
  DistributedPlanBuilder(
      const ExecutablePlanOptions& options,
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
      memory::MemoryPool* pool = nullptr);
  DistributedPlanBuilder(DistributedPlanBuilder& parent);

  /// Returns the planned fragments. The builder will be empty after this.
  std::vector<ExecutableFragment> fragments();

  PlanBuilder& shuffle(
      const std::vector<std::string>& keys,
      int numPartitions,
      bool replicateNullsAndAny,
      const std::vector<std::string>& outputLayout = {}) override;

  core::PlanNodePtr shuffleResult(
      const std::vector<std::string>& keys,
      int numPartitions,
      bool replicateNullsAndAny,
      const std::vector<std::string>& outputLayout = {}) override;

 private:
  void newFragment();

  DistributedPlanBuilder* rootBuilder() {
    auto* parent = this;
    while (parent->parent_) {
      parent = parent->parent_;
    }
    return parent;
  }

  void gatherScans(const core::PlanNodePtr& plan);

  const ExecutablePlanOptions& options_;
  DistributedPlanBuilder* parent_{nullptr};

  //
  // Stack of outstanding builders. The last element is the immediately
  // enclosing one. When returning an ExchangeNode from returnShuffle, the stack
  // is used to establish the linkage between the fragment of the returning
  // builder and the fragment current in the calling builder. Only filled in the
  // root builder.
  std::vector<DistributedPlanBuilder*> stack_;

  // Fragment counter. Only used in root builder.
  int32_t taskCounter_{0};
  ExecutableFragment current_;
  std::vector<ExecutableFragment> fragments_;
};
} // namespace facebook::velox::exec::test
