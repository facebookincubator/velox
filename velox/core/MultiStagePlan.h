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

#include <optional>
#include <vector>

#include <fmt/format.h>

#include "velox/common/EnumDeclare.h"
#include "velox/core/PlanNode.h"

namespace facebook::velox::core {

/// Describes how many tasks run a stage, and where.
enum class StageType {
  /// One task per partition of the stage's data source: as many as the runner
  /// decides from the splits of a query, one per worker in a fixed point,
  /// whose workers each hold a partition of every state entry.
  kSource,
  /// Exactly `numRemotePartitions` tasks.
  kFixed,
  /// Exactly 1 task, any worker.
  kSingle,
  /// Exactly 1 task, on coordinator.
  kCoordinator,
};

VELOX_DECLARE_ENUM_NAME(StageType);

/// Names an ExchangeNode of a stage and the stage that feeds it.
struct StageInput {
  /// Id of the ExchangeNode in the consuming stage.
  PlanNodeId exchangeId{};
  /// Id of the stage producing the exchange's input.
  int32_t producerStageId{0};
};

/// Describes one stage of a MultiStagePlan: a plan node that a runtime runs as
/// tasks, with how many tasks run it and which stages feed its exchanges. A
/// stage carries no per-task settings; a runtime builds the PlanFragment of
/// each task from 'plan'.
struct Stage {
  /// Identifies the stage within its plan. Ids run from 1 to the number of
  /// stages, in no relation to the order of the stages.
  int32_t id{0};

  /// How many tasks run the stage, and where.
  StageType type{StageType::kSource};

  /// Number of tasks the runtime must run for this stage. Required for
  /// kFixed. Optional for kSource, where it overrides the count the runtime
  /// would decide. std::nullopt for kSingle and kCoordinator. Splits tagged
  /// with a remotePartition in [0, numRemotePartitions) go to the task with
  /// that index.
  std::optional<int32_t> numRemotePartitions{};

  /// The plan node each task of the stage runs.
  PlanNodePtr plan{};

  /// The exchanges of 'plan' that other stages feed.
  std::vector<StageInput> inputs{};

  /// Returns the number of tasks that run this stage: 'numRemotePartitions'
  /// when set, 'numSourceTasks' -- the partitions of its data source -- for any
  /// other kSource, and 1 otherwise.
  int32_t numTasks(int32_t numSourceTasks) const;
};

/// A plan cut at its shuffles into stages, which a runtime runs as tasks and
/// wires through their exchanges. How many workers run a stage, where, and
/// over which transport its exchanges go are up to the runtime: a query runner
/// schedules the stages of a query, and a fixed point runs the stages of its
/// body on its workers once per iteration.
///
/// For example, a plan that aggregates across workers:
///
///   Stage 1, kSource:  TableScan -> partial agg -> PartitionedOutput(k, 4)
///   Stage 2, kFixed 4: Exchange e2 -> final agg -> PartitionedOutput(gather)
///   Stage 3, kSingle:  Exchange e3 -> result
///
///   MultiStagePlan plan({
///       {.id = 1, .type = StageType::kSource, .plan = partial},
///       {.id = 2, .type = StageType::kFixed, .numRemotePartitions = 4,
///        .plan = final, .inputs = {{"e2", 1}}},
///       {.id = 3, .type = StageType::kSingle, .plan = result,
///        .inputs = {{"e3", 2}}},
///   });
///   plan.root().id;  // 3
///
/// The root is the one stage that no stage reads; it produces the plan's
/// result. The constructor throws a user error unless:
///  - there is at least one stage, with ids running from 1 to the number of
///    stages;
///  - 'numRemotePartitions' is set, and positive, exactly for kFixed (kSource
///    may carry a positive hint);
///  - each input names an ExchangeNode of its stage and a producer other than
///    itself, whose plan's root is a PartitionedOutputNode writing the schema
///    the ExchangeNode reads;
///  - each ExchangeNode of a stage is named by exactly one of its inputs;
///  - each stage other than the root has exactly one consumer, and every stage
///    feeds the root;
///  - a partitioned producer has one partition per consumer task, unless the
///    consumer is kSource, whose task count the runtime decides.
class MultiStagePlan {
 public:
  explicit MultiStagePlan(std::vector<Stage> stages);

  /// Returns the stages in the order they were given.
  const std::vector<Stage>& stages() const {
    return stages_;
  }

  /// Returns the stage with 'id'.
  const Stage& stage(int32_t id) const;

  /// Returns the stage that produces the plan's result.
  const Stage& root() const {
    return stages_[rootIndex_];
  }

  /// Returns the stage that reads stage 'id', or nullptr for the root.
  const Stage* consumer(int32_t id) const;

  /// Serializes the stages with their plan nodes.
  folly::dynamic serialize() const;

  /// Rebuilds a plan from serialize()'s output.  'context' is passed to the
  /// plan nodes' deserialization.
  static MultiStagePlan deserialize(const folly::dynamic& obj, void* context);

  /// Rebuilds the stages of serialize()'s output without checking that they
  /// form a plan, for a caller that changes them before building one.
  /// 'context' is passed to the plan nodes' deserialization.
  static std::vector<Stage> deserializeStages(
      const folly::dynamic& obj,
      void* context);

 private:
  std::vector<Stage> stages_;

  // Index into 'stages_' of the stage with id i + 1.
  std::vector<size_t> indices_;

  // Index into 'stages_' of the consumer of the stage with id i + 1, or
  // SIZE_MAX for the root.
  std::vector<size_t> consumers_;

  size_t rootIndex_{0};
};

} // namespace facebook::velox::core

template <>
struct fmt::formatter<facebook::velox::core::StageType>
    : formatter<std::string_view> {
  auto format(facebook::velox::core::StageType type, format_context& ctx)
      const {
    return formatter<std::string_view>::format(
        facebook::velox::core::StageTypeName::toName(type), ctx);
  }
};
