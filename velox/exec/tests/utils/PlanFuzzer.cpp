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

#include "velox/exec/tests/utils/PlanFuzzer.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

DEFINE_int32(
    plan_fuzzer_encodings,
    facebook::velox::bits::lowMask(
        facebook::velox::exec::test::TestingRebatch::kNumEncodings),
    "Bit mask of encoding fuzzes to try");

namespace facebook::velox::exec::test {

namespace {
void addSplitVector(Task* task, const std::vector<Task::TaskSplit>& splits) {
  for (auto& split : splits) {
    auto splitCopy = split.split;
    task->addSplit(split.nodeId, std::move(splitCopy));
  }
}
} // namespace

core::PlanNodePtr fuzzOutput(
    const core::PlanNodePtr& node,
    core::PlanNodeIdGenerator& idGenerator) {
  return std::make_shared<TestingRebatchNode>(idGenerator.next(), node);
}

int32_t maxNodeId(const core::PlanNodePtr& node, std::vector<int32_t>& allIds) {
  auto& sources = node->sources();
  allIds.push_back(atoi(node->id().c_str()));
  if (sources.empty()) {
    return atoi(node->id().c_str());
  }
  int32_t max = atoi(node->id().c_str());
  for (auto& source : sources) {
    max = std::max<int32_t>(max, maxNodeId(source, allIds));
  }
  return max;
}

#define COPY_PLAN_CASE(className)                                             \
  else if (                                                                   \
      const className* node = dynamic_cast<const className*>(source.get())) { \
    return std::make_shared<className>(*node);                                \
  }

// Copies any non-leaf PlanNode.
core::PlanNodePtr copyNode(const core::PlanNodePtr& source) {
  if (0) {
  }
  COPY_PLAN_CASE(core::ProjectNode)
  COPY_PLAN_CASE(core::FilterNode)
  COPY_PLAN_CASE(core::FilterNode)
  COPY_PLAN_CASE(core::TableWriteNode)
  COPY_PLAN_CASE(core::AggregationNode)
  COPY_PLAN_CASE(core::GroupIdNode)
  COPY_PLAN_CASE(core::LocalPartitionNode)
  COPY_PLAN_CASE(core::PartitionedOutputNode)
  COPY_PLAN_CASE(core::HashJoinNode)
  COPY_PLAN_CASE(core::MergeJoinNode)
  COPY_PLAN_CASE(core::NestedLoopJoinNode)
  COPY_PLAN_CASE(core::OrderByNode)
  COPY_PLAN_CASE(core::TopNNode)
  COPY_PLAN_CASE(core::LimitNode)
  COPY_PLAN_CASE(core::UnnestNode)
  COPY_PLAN_CASE(core::EnforceSingleRowNode)
  COPY_PLAN_CASE(core::AssignUniqueIdNode)
  COPY_PLAN_CASE(core::WindowNode)
  else {
    VELOX_UNREACHABLE("Unsupported node type {}", source->toString(true, true));
  }
}

core::PlanNodePtr fuzzPlan(
    const core::PlanNodePtr& node,
    int32_t& counter,
    const SelectivityVector& fuzzable,
    core::PlanNodeIdGenerator& idGenerator) {
  auto& sources = node->sources();
  if (sources.empty()) {
    if (fuzzable.isValid(counter)) {
      ++counter;

      return fuzzOutput(node, idGenerator);
    }
    ++counter;
    return node;
  }
  std::vector<core::PlanNodePtr> newSources;
  for (auto& source : sources) {
    newSources.push_back(fuzzPlan(source, counter, fuzzable, idGenerator));
  }
  auto copy = copyNode(node);
  auto& copySources =
      const_cast<std::vector<core::PlanNodePtr>&>(copy->sources());
  copySources = newSources;
  if (fuzzable.isValid(counter)) {
    return fuzzOutput(copy, idGenerator);
  } else {
    return copy;
  }
}

void drillDownPlanFuzz(
    const CursorParameters& params,
    const std::vector<RowVectorPtr>& result,
    const std::vector<Task::TaskSplit>& previousSplits,
    const std::vector<int32_t>& allNodes) {}

void checkFuzzedPlans(
    const CursorParameters& params,
    const std::vector<RowVectorPtr>& result,
    const std::shared_ptr<Task>& referenceTask) {
  int32_t nodeCounter = 0;
  CursorParameters fuzzParams = params;
  std::vector<int32_t> allIds;
  int32_t maxNode = maxNodeId(params.planNode, allIds);
  SelectivityVector fuzzable(maxNode);
  core::PlanNodeIdGenerator idGenerator(maxNode + 1);
  fuzzParams.planNode =
      fuzzPlan(params.planNode, nodeCounter, fuzzable, idGenerator);
  auto previousSplits = referenceTask->moveRememberedSplits();
  bool firstTime = true;
  auto [cursor, fuzzResult] = readCursorOnly(fuzzParams, [&](auto task) {
    if (firstTime) {
      addSplitVector(task, previousSplits);
    }
    firstTime = false;
  });
  if (!assertEqualResults(result, fuzzResult)) {
    drillDownPlanFuzz(params, result, previousSplits, allIds);
  }
}

} // namespace facebook::velox::exec::test
