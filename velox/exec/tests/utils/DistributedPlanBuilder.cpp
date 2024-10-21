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

#include "velox/exec/tests/utils/DistributedPlanBuilder.h"

namespace facebook::velox::exec::test {

DistributedPlanBuilder::DistributedPlanBuilder(
    const ExecutablePlanOptions& options,
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
    memory::MemoryPool* pool)
    : PlanBuilder(planNodeIdGenerator, pool), options_(options) {
  auto* root = rootBuilder();
  VELOX_CHECK(
      root->stack_.empty(),
      "Cannot make a top level DistributedPlanBuilder inside an open one");
  root->stack_.push_back(this);
  newFragment();
  current_.width = options_.numWorkers;
}

DistributedPlanBuilder::DistributedPlanBuilder(DistributedPlanBuilder& parent)
    : PlanBuilder(parent.planNodeIdGenerator(), parent.pool()),
      options_(parent.options_),
      parent_(&parent) {
  auto* root = rootBuilder();
  root->stack_.push_back(this);
  newFragment();
  current_.width = options_.numWorkers;
}

std::vector<ExecutableFragment> DistributedPlanBuilder::fragments() {
  newFragment();
  return std::move(fragments_);
}

void DistributedPlanBuilder::newFragment() {
  if (!current_.taskPrefix.empty()) {
    gatherScans(planNode_);
    current_.fragment = core::PlanFragment(std::move(planNode_));
    fragments_.push_back(std::move(current_));
  }
  current_ = ExecutableFragment();
  auto* root = rootBuilder();
  current_.taskPrefix =
      fmt::format("{}.{}", options_.queryId, root->taskCounter_++);
  planNode_ = nullptr;
}

PlanBuilder& DistributedPlanBuilder::shuffle(
    const std::vector<std::string>& keys,
    int numPartitions,
    bool replicateNullsAndAny,
    const std::vector<std::string>& outputLayout) {
  partitionedOutput(keys, numPartitions, replicateNullsAndAny, outputLayout);
  auto* output =
      dynamic_cast<const core::PartitionedOutputNode*>(planNode_.get());
  auto producerPrefix = current_.taskPrefix;
  newFragment();
  current_.width = numPartitions;
  exchange(output->outputType());
  auto* exchange = dynamic_cast<const core::ExchangeNode*>(planNode_.get());
  current_.inputStages.push_back(InputStage{exchange->id(), producerPrefix});
  return *this;
}

core::PlanNodePtr DistributedPlanBuilder::shuffleResult(
    const std::vector<std::string>& keys,
    int numPartitions,
    bool replicateNullsAndAny,
    const std::vector<std::string>& outputLayout) {
  partitionedOutput(keys, numPartitions, replicateNullsAndAny, outputLayout);
  auto* output =
      dynamic_cast<const core::PartitionedOutputNode*>(planNode_.get());
  auto producerPrefix = current_.taskPrefix;
  auto result = planNode_;
  newFragment();
  auto* root = rootBuilder();
  root->stack_.pop_back();
  auto* consumer = root->stack_.back();
  if (consumer->current_.width != 0) {
    VELOX_CHECK_EQ(
        numPartitions,
        consumer->current_.width,
        "The consumer width should match the producer fanout");
  } else {
    consumer->current_.width = numPartitions;
  }

  for (auto& fragment : fragments_) {
    root->fragments_.push_back(std::move(fragment));
  }
  exchange(output->outputType());
  auto* exchange = dynamic_cast<const core::ExchangeNode*>(planNode_.get());
  consumer->current_.inputStages.push_back(
      InputStage{exchange->id(), producerPrefix});
  return std::move(planNode_);
}

void DistributedPlanBuilder::gatherScans(const core::PlanNodePtr& plan) {
  if (auto scan = std::dynamic_pointer_cast<const core::TableScanNode>(plan)) {
    current_.scans.push_back(scan);
    return;
  }
  for (auto& in : plan->sources()) {
    gatherScans(in);
  }
}
} // namespace facebook::velox::exec::test
