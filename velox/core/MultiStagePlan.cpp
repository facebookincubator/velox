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

#include "velox/core/MultiStagePlan.h"

#include <limits>

#include <folly/CppAttributes.h>
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>

#include "velox/common/EnumDefine.h"
#include "velox/core/FixedPointPlanNodes.h"

namespace facebook::velox::core {

namespace {

const auto& stageTypeNames() {
  static const folly::F14FastMap<StageType, std::string_view> kNames = {
      {StageType::kSource, "SOURCE"},
      {StageType::kFixed, "FIXED"},
      {StageType::kSingle, "SINGLE"},
      {StageType::kCoordinator, "COORDINATOR"},
  };
  return kNames;
}

// Marks an index that is not set: a stage id not seen yet, or the consumer of
// the root.
constexpr size_t kNone = std::numeric_limits<size_t>::max();

} // namespace

VELOX_DEFINE_ENUM_NAME(StageType, stageTypeNames);

int32_t Stage::numTasks(int32_t numSourceTasks) const {
  if (numRemotePartitions.has_value()) {
    return numRemotePartitions.value();
  }
  return type == StageType::kSource ? numSourceTasks : 1;
}

namespace {

// Checks that 'stage's type is consistent with its numRemotePartitions.
void checkStageType(const Stage& stage) {
  const auto& numRemotePartitions = stage.numRemotePartitions;
  switch (stage.type) {
    case StageType::kFixed:
      VELOX_USER_CHECK(
          numRemotePartitions.has_value(),
          "kFixed stage must have numRemotePartitions set: {}",
          stage.id);
      break;
    case StageType::kSingle:
    case StageType::kCoordinator:
      VELOX_USER_CHECK(
          !numRemotePartitions.has_value(),
          "{} stage must not have numRemotePartitions set: {}",
          stage.type,
          stage.id);
      break;
    case StageType::kSource:
      break;
  }

  if (numRemotePartitions.has_value()) {
    VELOX_USER_CHECK_GT(
        numRemotePartitions.value(),
        0,
        "Stage numRemotePartitions must be positive: {}",
        stage.id);
  }
}

// Returns the node with 'id' in the tree under 'node', or null if none,
// including the initial plans of a fixed point, which read their exchanges once
// before the loop.
const PlanNode* FOLLY_NULLABLE
findConsumerNode(const PlanNode& node, const PlanNodeId& id) {
  if (node.id() == id) {
    return &node;
  }
  if (const auto* fixedPoint = node.as<FixedPointNode>()) {
    for (const auto& declaration : fixedPoint->stateDeclarations()) {
      if (declaration->initialPlan() != nullptr) {
        if (const auto* found =
                findConsumerNode(*declaration->initialPlan(), id)) {
          return found;
        }
      }
    }
  }
  for (const auto& source : node.sources()) {
    if (const auto* found = findConsumerNode(*source, id)) {
      return found;
    }
  }
  return nullptr;
}

// Checks that 'input' of 'consumer' names an ExchangeNode of it and a producer
// whose output it can read.
void checkInput(
    const Stage& consumer,
    const StageInput& input,
    const Stage& producer) {
  const auto* consumerNode = findConsumerNode(*consumer.plan, input.exchangeId);
  VELOX_USER_CHECK_NOT_NULL(
      consumerNode,
      "Exchange node not found: {}, stage: {}",
      input.exchangeId,
      consumer.id);
  VELOX_USER_CHECK(
      consumerNode->is<ExchangeNode>(),
      "Stage input must be an ExchangeNode: {}, stage: {}",
      input.exchangeId,
      consumer.id);

  const auto* partitionedOutput = producer.plan->as<PartitionedOutputNode>();
  VELOX_USER_CHECK_NOT_NULL(
      partitionedOutput,
      "Expected PartitionedOutputNode at root of producer stage: {}",
      producer.id);
  VELOX_USER_CHECK(
      consumerNode->outputType()->equivalent(*partitionedOutput->outputType()),
      "Exchange must read the schema its producer writes: {} reads {}, stage "
      "{} writes {}",
      input.exchangeId,
      consumerNode->outputType()->toString(),
      producer.id,
      partitionedOutput->outputType()->toString());

  if (partitionedOutput->isBroadcast() || partitionedOutput->isArbitrary() ||
      consumer.type == StageType::kSource) {
    return;
  }

  VELOX_USER_CHECK_EQ(
      partitionedOutput->numPartitions(),
      consumer.numRemotePartitions.value_or(1),
      "Partition count mismatch between producer {} and consumer {}",
      producer.id,
      consumer.id);
}

// Collects the ExchangeNodes under 'node' that a stage input can name: those
// findConsumerNode() finds.
void collectExchanges(
    const PlanNode& node,
    folly::F14FastSet<PlanNodeId>& exchangeIds) {
  if (node.is<ExchangeNode>()) {
    exchangeIds.insert(node.id());
  }
  if (const auto* fixedPoint = node.as<FixedPointNode>()) {
    for (const auto& declaration : fixedPoint->stateDeclarations()) {
      if (declaration->initialPlan() != nullptr) {
        collectExchanges(*declaration->initialPlan(), exchangeIds);
      }
    }
  }
  for (const auto& source : node.sources()) {
    collectExchanges(*source, exchangeIds);
  }
}

// Checks that an input of 'stage' names each of its ExchangeNodes, which
// nothing else feeds.
void checkExchangesFed(const Stage& stage) {
  folly::F14FastSet<PlanNodeId> exchangeIds;
  collectExchanges(*stage.plan, exchangeIds);
  for (const auto& input : stage.inputs) {
    exchangeIds.erase(input.exchangeId);
  }
  for (const auto& exchangeId : exchangeIds) {
    VELOX_USER_FAIL(
        "Exchange must be fed by a stage input: {}, stage: {}",
        exchangeId,
        stage.id);
  }
}

} // namespace

MultiStagePlan::MultiStagePlan(std::vector<Stage> stages)
    : stages_{std::move(stages)} {
  VELOX_USER_CHECK(!stages_.empty(), "Plan must have at least one stage");

  const auto numStages = stages_.size();
  indices_.assign(numStages, kNone);
  for (size_t i = 0; i < numStages; ++i) {
    const auto& stage = stages_[i];
    VELOX_USER_CHECK_GE(stage.id, 1, "Stage id must be positive");
    VELOX_USER_CHECK_LE(
        stage.id,
        static_cast<int32_t>(numStages),
        "Stage id exceeds the number of stages");
    auto& index = indices_[stage.id - 1];
    VELOX_USER_CHECK_EQ(index, kNone, "Duplicate stage id: {}", stage.id);
    index = i;
    VELOX_USER_CHECK_NOT_NULL(stage.plan, "Stage has no plan: {}", stage.id);
    checkStageType(stage);
  }

  consumers_.assign(numStages, kNone);
  for (size_t i = 0; i < numStages; ++i) {
    const auto& consumer = stages_[i];
    folly::F14FastSet<PlanNodeId> fedExchanges;
    for (const auto& input : consumer.inputs) {
      VELOX_USER_CHECK(
          fedExchanges.insert(input.exchangeId).second,
          "Exchange must be fed by one stage input: {}, stage: {}",
          input.exchangeId,
          consumer.id);
      const auto producerId = input.producerStageId;
      VELOX_USER_CHECK_GE(
          producerId,
          1,
          "Producer stage not found: {}, consumer: {}",
          producerId,
          consumer.id);
      VELOX_USER_CHECK_LE(
          producerId,
          static_cast<int32_t>(numStages),
          "Producer stage not found: {}, consumer: {}",
          producerId,
          consumer.id);
      VELOX_USER_CHECK_NE(
          producerId,
          consumer.id,
          "Stage cannot be its own producer: {}",
          consumer.id);
      auto& producerConsumer = consumers_[producerId - 1];
      VELOX_USER_CHECK_EQ(
          producerConsumer,
          kNone,
          "Producer stage read by multiple consumers: {}",
          producerId);
      producerConsumer = i;
      checkInput(consumer, input, stages_[indices_[producerId - 1]]);
    }
    checkExchangesFed(consumer);
  }

  std::optional<int32_t> rootId;
  for (int32_t id = 1; id <= static_cast<int32_t>(numStages); ++id) {
    if (consumers_[id - 1] == kNone) {
      VELOX_USER_CHECK(
          !rootId.has_value(),
          "Plan must have one stage that no stage reads: {}, {}",
          rootId.value(),
          id);
      rootId = id;
    }
  }
  VELOX_USER_CHECK(
      rootId.has_value(), "Plan must have one stage that no stage reads");
  rootIndex_ = indices_[rootId.value() - 1];

  // Stages that only read each other each have a consumer, yet never feed the
  // root.
  std::vector<size_t> pending{rootIndex_};
  std::vector<bool> reached(numStages, false);
  reached[rootIndex_] = true;
  while (!pending.empty()) {
    const auto& consumer = stages_[pending.back()];
    pending.pop_back();
    for (const auto& input : consumer.inputs) {
      const auto producer = indices_[input.producerStageId - 1];
      if (!reached[producer]) {
        reached[producer] = true;
        pending.push_back(producer);
      }
    }
  }
  for (size_t i = 0; i < numStages; ++i) {
    VELOX_USER_CHECK(
        reached[i], "Stage must feed the root stage: {}", stages_[i].id);
  }
}

const Stage& MultiStagePlan::stage(int32_t id) const {
  VELOX_CHECK_GE(id, 1);
  VELOX_CHECK_LE(id, static_cast<int32_t>(stages_.size()));
  return stages_[indices_[id - 1]];
}

const Stage* MultiStagePlan::consumer(int32_t id) const {
  VELOX_CHECK_GE(id, 1);
  VELOX_CHECK_LE(id, static_cast<int32_t>(stages_.size()));
  const auto index = consumers_[id - 1];
  return index == kNone ? nullptr : &stages_[index];
}

folly::dynamic MultiStagePlan::serialize() const {
  folly::dynamic stages = folly::dynamic::array;
  for (const auto& stage : stages_) {
    folly::dynamic obj = folly::dynamic::object;
    obj["id"] = stage.id;
    obj["type"] = std::string(StageTypeName::toName(stage.type));
    if (stage.numRemotePartitions.has_value()) {
      obj["numRemotePartitions"] = stage.numRemotePartitions.value();
    }
    obj["plan"] = stage.plan->serialize();
    folly::dynamic inputs = folly::dynamic::array;
    for (const auto& input : stage.inputs) {
      inputs.push_back(
          folly::dynamic::object("exchangeId", input.exchangeId)(
              "producerStageId", input.producerStageId));
    }
    obj["inputs"] = std::move(inputs);
    stages.push_back(std::move(obj));
  }
  return folly::dynamic::object("stages", std::move(stages));
}

// static
MultiStagePlan MultiStagePlan::deserialize(
    const folly::dynamic& obj,
    void* context) {
  return MultiStagePlan(deserializeStages(obj, context));
}

// static
std::vector<Stage> MultiStagePlan::deserializeStages(
    const folly::dynamic& obj,
    void* context) {
  std::vector<Stage> stages;
  for (const auto& stageObj : obj["stages"]) {
    Stage stage{
        .id = static_cast<int32_t>(stageObj["id"].asInt()),
        .type = StageTypeName::toStageType(stageObj["type"].asString()),
        .plan = ISerializable::deserialize<PlanNode>(stageObj["plan"], context),
    };
    if (stageObj.count("numRemotePartitions") != 0u) {
      stage.numRemotePartitions =
          static_cast<int32_t>(stageObj["numRemotePartitions"].asInt());
    }
    for (const auto& input : stageObj["inputs"]) {
      stage.inputs.push_back({
          .exchangeId = input["exchangeId"].asString(),
          .producerStageId =
              static_cast<int32_t>(input["producerStageId"].asInt()),
      });
    }
    stages.push_back(std::move(stage));
  }
  return stages;
}

} // namespace facebook::velox::core
