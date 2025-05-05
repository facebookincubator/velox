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

#include "velox/runner/tests/QueryReplayRunner.h"

#include "velox/exec/PartitionFunction.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"

namespace facebook::velox::runner {

const std::string QueryReplayRunner::kHiveConnectorId = "test-hive";

namespace {
std::shared_ptr<memory::MemoryPool> makeRootPool(const std::string& queryId) {
  static std::atomic_uint64_t poolId{0};
  return memory::memoryManager()->addRootPool(
      fmt::format("{}_{}", queryId, poolId++));
}

std::vector<VectorPtr> readCursor(
    std::shared_ptr<runner::LocalRunner> runner,
    memory::MemoryPool* pool) {
  // We'll check the result after tasks are deleted, so copy the result vectors
  // to 'pool' that has longer lifetime.
  std::vector<VectorPtr> result;
  while (auto rows = runner->next()) {
    result.push_back(BaseVector::copy(*rows, pool));
  }
  return result;
}
} // namespace

QueryReplayRunner::QueryReplayRunner(
    memory::MemoryPool* pool,
    TaskPrefixExtractor taskPrefixExtractor,
    int32_t width,
    int32_t maxDrivers)
    : pool_{pool},
      taskPrefixExtractor_{taskPrefixExtractor},
      width_{width},
      maxDrivers_{maxDrivers} {
  executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(maxDrivers_);

  Type::registerSerDe();
  core::PlanNode::registerSerDe();
  core::ITypedExpr::registerSerDe();
  exec::registerPartitionFunctionSerDe();

  exec::ExchangeSource::registerFactory(exec::test::createLocalExchangeSource);
}

std::shared_ptr<core::QueryCtx> QueryReplayRunner::makeQueryCtx(
    const std::string& queryId,
    std::shared_ptr<memory::MemoryPool> rootPool) {
  auto& config = config_;
  auto hiveConfig = hiveConfig_;
  std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
      connectorConfigs;
  connectorConfigs[kHiveConnectorId] =
      std::make_shared<config::ConfigBase>(std::move(hiveConfig));

  return core::QueryCtx::create(
      executor_.get(),
      core::QueryConfig(config),
      std::move(connectorConfigs),
      cache::AsyncDataCache::getInstance(),
      rootPool->shared_from_this(),
      nullptr,
      queryId);
}

std::vector<VectorPtr> QueryReplayRunner::run(
    const std::string& queryId,
    const std::vector<std::string>& serializedPlanFragments) {
  auto queryRootPool = makeRootPool(queryId);
  auto multiFragmentPlan = deserializePlan(queryId, serializedPlanFragments);
  auto splitSourceFactory = makeSplitSourceFactory(multiFragmentPlan); // todo
  auto localRunner = std::make_shared<LocalRunner>(
      std::move(multiFragmentPlan),
      makeQueryCtx(queryId, queryRootPool),
      splitSourceFactory);

  auto result = readCursor(localRunner, pool_);
  localRunner->waitForCompletion(kWaitTimeoutUs);

  return result;
}

namespace {
std::vector<std::string> getStringListFromJson(const folly::dynamic& json) {
  std::vector<std::string> result;
  result.resize(json.size());
  std::transform(
      json.begin(), json.end(), result.begin(), [](const folly::dynamic& json) {
        return json.getString();
      });
  return result;
}

void getScanNodesImpl(
    const core::PlanNodePtr& plan,
    std::vector<core::TableScanNodePtr>& result) {
  if (auto tableScan =
          std::dynamic_pointer_cast<const core::TableScanNode>(plan)) {
    result.push_back(tableScan);
    return;
  }
  for (const auto& child : plan->sources()) {
    getScanNodesImpl(child, result);
  }
}

// Return all table scan nodes in 'plan'.
std::vector<core::TableScanNodePtr> getScanNodes(
    const core::PlanNodePtr& plan) {
  std::vector<core::TableScanNodePtr> result;
  getScanNodesImpl(plan, result);
  return result;
}

// If 'plan' is a broadcast partitioned output node, return the number of
// broadcast destinations. Otherwise return 0.
int getNumBroadcastDestinations(const core::PlanNodePtr& plan) {
  if (auto partitionedOutput =
          std::dynamic_pointer_cast<const core::PartitionedOutputNode>(plan)) {
    if (partitionedOutput->isBroadcast()) {
      return partitionedOutput->numPartitions();
    }
  }
  return 0;
}

// Return true if 'node' is a gathering PartitionedOutput node.
bool isGatheringPartition(const core::PlanNodePtr& node) {
  if (auto partitionedOutput =
          std::dynamic_pointer_cast<const core::PartitionedOutputNode>(node)) {
    return partitionedOutput->keys().empty();
  }
  return false;
}

// Return a new plan tree with the same structure as 'plan' but with the number
// of partitions of the root PartitionedOutputNode updated to 'numPartitions'.
// This method throws if the root node of 'plan' is not a PartitionedOutputNode
// or if it's a gathering PartitionedOutputNode.
core::PlanNodePtr updateNumberOfPartitions(
    const core::PlanNodePtr& plan,
    int numPartitions) {
  auto partitionedOutput =
      std::dynamic_pointer_cast<const core::PartitionedOutputNode>(plan);
  VELOX_CHECK(partitionedOutput != nullptr);
  VELOX_CHECK(!partitionedOutput->keys().empty());
  return std::make_shared<core::PartitionedOutputNode>(
      partitionedOutput->id(),
      partitionedOutput->kind(),
      partitionedOutput->keys(),
      numPartitions,
      partitionedOutput->isReplicateNullsAndAny(),
      partitionedOutput->partitionFunctionSpecPtr(),
      partitionedOutput->outputType(),
      partitionedOutput->serdeKind(),
      partitionedOutput->sources()[0]);
}

struct PlanFragmentInfo {
  core::PlanNodePtr plan{nullptr};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      remoteTaskIdMap{};
  std::vector<core::TableScanNodePtr> scans{};
  int numBroadcastDestinations{0};
  int numWorkers{0};
};
} // namespace

MultiFragmentPlanPtr QueryReplayRunner::deserializePlan(
    const std::string& queryId,
    const std::vector<std::string>& serializedPlanFragments) {
  std::map<std::string, PlanFragmentInfo, std::greater<std::string>>
      planFragments;
  for (auto i = 0; i < serializedPlanFragments.size(); ++i) {
    auto json = folly::parseJson(serializedPlanFragments[i]);
    auto taskPrefix = taskPrefixExtractor_(json.at("task_id").getString());
    VELOX_CHECK_EQ(planFragments.count(taskPrefix), 0);

    auto jsonPlanFragment = json.at("plan_fragment");
    const auto plan =
        core::PlanNode::deserialize<core::PlanNode>(jsonPlanFragment, pool_);
    planFragments[taskPrefix].scans = getScanNodes(plan);
    planFragments[taskPrefix].numBroadcastDestinations =
        getNumBroadcastDestinations(plan);
    planFragments[taskPrefix].plan = plan;
  }

  for (auto i = 0; i < serializedPlanFragments.size(); ++i) {
    auto json = folly::parseJson(serializedPlanFragments[i]);
    auto taskPrefix = taskPrefixExtractor_(json.at("task_id").getString());
    auto jsonRemoteTaskIdMaps = json.at("remote_task_ids");
    std::unordered_map<std::string, std::unordered_set<std::string>>
        currentRemoteTaskIdMap;
    for (const auto& [planNodeId, remoteTaskIds] :
         jsonRemoteTaskIdMaps.items()) {
      auto remoteTaskIdList = getStringListFromJson(remoteTaskIds);
      std::unordered_set<std::string> remoteTaskIdPrefixSet;
      for (const auto& remoteTaskId : remoteTaskIdList) {
        auto remoteTaskPrefix = taskPrefixExtractor_(remoteTaskId);
        if (isGatheringPartition(planFragments[remoteTaskPrefix].plan)) {
          planFragments[taskPrefix].numWorkers = 1;
        } else {
          planFragments[taskPrefix].numWorkers = width_;
          planFragments[remoteTaskPrefix].plan = updateNumberOfPartitions(
              planFragments[remoteTaskPrefix].plan, width_);
        }
        remoteTaskIdPrefixSet.insert(remoteTaskPrefix);
      }
      currentRemoteTaskIdMap[planNodeId.getString()] =
          std::move(remoteTaskIdPrefixSet);
    }
    planFragments[taskPrefix].remoteTaskIdMap = currentRemoteTaskIdMap;
  }

  std::vector<ExecutableFragment> executableFragments;
  for (const auto& [taskPrefix, planFragmentInfo] : planFragments) {
    ExecutableFragment executableFragment{taskPrefix};
    executableFragment.width =
        (planFragmentInfo.numWorkers > 0) ? planFragmentInfo.numWorkers : 1;
    executableFragment.fragment =
        core::PlanFragment{planFragments[taskPrefix].plan};

    executableFragment.scans = planFragments[taskPrefix].scans;
    executableFragment.numBroadcastDestinations =
        planFragments[taskPrefix].numBroadcastDestinations;

    std::vector<InputStage> inputStages;
    const auto& remoteTaskIdMap = planFragments[taskPrefix].remoteTaskIdMap;
    for (const auto& [planNodeId, remoteTaskPrefixes] : remoteTaskIdMap) {
      for (const auto& remoteTaskPrefix : remoteTaskPrefixes) {
        inputStages.push_back(InputStage{planNodeId, remoteTaskPrefix}); // todo
      }
    }
    executableFragment.inputStages = inputStages;
    executableFragments.push_back(std::move(executableFragment));
  }

  MultiFragmentPlan::Options options{queryId, width_, maxDrivers_};
  return std::make_shared<MultiFragmentPlan>(
      std::move(executableFragments), std::move(options));
}

std::shared_ptr<runner::SplitSourceFactory>
QueryReplayRunner::makeSplitSourceFactory(
    const runner::MultiFragmentPlanPtr& plan) {
  // TODO: support TableScanNode.
  return nullptr;
}

} // namespace facebook::velox::runner
