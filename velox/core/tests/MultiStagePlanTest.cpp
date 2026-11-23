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

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::core {
namespace {

using exec::test::PlanBuilder;

class MultiStagePlanTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance({});
  }

  PlanBuilder startValues() {
    return PlanBuilder(idGenerator_)
        .values({BaseVector::create<RowVector>(
            ROW("a", BIGINT()), 1, pool_.get())});
  }

  PlanNodePtr values() {
    return startValues().planNode();
  }

  PlanNodePtr partitionedOutput(int32_t numPartitions) {
    return startValues().partitionedOutput({"a"}, numPartitions).planNode();
  }

  PlanNodePtr gatherOutput() {
    return startValues().partitionedOutput({}, 1).planNode();
  }

  PlanNodePtr broadcastOutput() {
    return startValues().partitionedOutputBroadcast().planNode();
  }

  PlanNodePtr exchange() {
    return PlanBuilder(idGenerator_)
        .exchange(
            ROW("a", BIGINT()),
            VectorSerde::kindName(VectorSerde::Kind::kPresto))
        .planNode();
  }

  // Returns a stage with 'id' whose plan is one exchange, fed by
  // 'producerId'.
  Stage reader(
      int32_t id,
      StageType type,
      std::optional<int32_t> numRemotePartitions,
      int32_t producerId) {
    Stage stage{
        .id = id,
        .type = type,
        .numRemotePartitions = numRemotePartitions,
        .plan = exchange(),
    };
    stage.inputs.push_back({stage.plan->id(), producerId});
    return stage;
  }

  // Returns a kSingle stage with 'id' that reads 'producerId' and gathers its
  // rows for another stage.
  Stage relay(int32_t id, int32_t producerId) {
    Stage stage{
        .id = id,
        .type = StageType::kSingle,
        .plan = PlanBuilder(idGenerator_)
                    .exchange(
                        ROW("a", BIGINT()),
                        VectorSerde::kindName(VectorSerde::Kind::kPresto))
                    .partitionedOutput({}, 1)
                    .planNode(),
    };
    stage.inputs.push_back({stage.plan->sources()[0]->id(), producerId});
    return stage;
  }

  // Returns a stage of 'type' with id 1 producing no output.
  Stage single(
      StageType type,
      std::optional<int32_t> numRemotePartitions = std::nullopt) {
    return Stage{
        .id = 1,
        .type = type,
        .numRemotePartitions = numRemotePartitions,
        .plan = values(),
    };
  }

  std::shared_ptr<PlanNodeIdGenerator> idGenerator_ =
      std::make_shared<PlanNodeIdGenerator>();

  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();
};

TEST_F(MultiStagePlanTest, numTasks) {
  EXPECT_EQ(single(StageType::kSource).numTasks(7), 7);
  EXPECT_EQ(single(StageType::kSource, 5).numTasks(7), 5);
  EXPECT_EQ(single(StageType::kFixed, 4).numTasks(7), 4);
  EXPECT_EQ(single(StageType::kSingle).numTasks(7), 1);
  EXPECT_EQ(single(StageType::kCoordinator).numTasks(7), 1);
}

TEST_F(MultiStagePlanTest, validPlan) {
  const MultiStagePlan plan({
      reader(2, StageType::kFixed, 4, 1),
      {.id = 1, .type = StageType::kSource, .plan = partitionedOutput(4)},
  });

  EXPECT_EQ(plan.root().id, 2);
  EXPECT_EQ(plan.stage(1).plan->name(), "PartitionedOutput");
  EXPECT_EQ(plan.consumer(1), &plan.root());
  EXPECT_EQ(plan.consumer(2), nullptr);
}

TEST_F(MultiStagePlanTest, empty) {
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({}), "Plan must have at least one stage");
}

TEST_F(MultiStagePlanTest, stageIds) {
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({{.id = 0, .plan = values()}}),
      "Stage id must be positive");
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({{.id = 2, .plan = values()}}),
      "Stage id exceeds the number of stages");
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({
          {.id = 1, .plan = gatherOutput()},
          {.id = 1, .plan = values()},
      }),
      "Duplicate stage id: 1");
  const Stage stage{.id = 1, .plan = nullptr};
  VELOX_ASSERT_USER_THROW(MultiStagePlan({stage}), "Stage has no plan: 1");
}

TEST_F(MultiStagePlanTest, numRemotePartitions) {
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({single(StageType::kFixed)}),
      "kFixed stage must have numRemotePartitions set: 1");
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({single(StageType::kSingle, 1)}),
      "SINGLE stage must not have numRemotePartitions set: 1");
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({single(StageType::kCoordinator, 1)}),
      "COORDINATOR stage must not have numRemotePartitions set: 1");
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({single(StageType::kFixed, 0)}),
      "Stage numRemotePartitions must be positive: 1");
  EXPECT_NO_THROW(MultiStagePlan({single(StageType::kSource, 5)}));
}

TEST_F(MultiStagePlanTest, missingProducer) {
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({reader(1, StageType::kSingle, std::nullopt, 7)}),
      "Producer stage not found: 7, consumer: 1");
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({reader(1, StageType::kSingle, std::nullopt, 0)}),
      "Producer stage not found: 0, consumer: 1");
}

TEST_F(MultiStagePlanTest, ownProducer) {
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({reader(1, StageType::kSingle, std::nullopt, 1)}),
      "Stage cannot be its own producer: 1");
}

TEST_F(MultiStagePlanTest, inputNotExchange) {
  Stage consumer{.id = 2, .type = StageType::kSingle, .plan = values()};
  consumer.inputs.push_back({consumer.plan->id(), 1});

  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({
          {.id = 1, .plan = gatherOutput()},
          consumer,
      }),
      "Stage input must be an ExchangeNode");

  consumer.inputs = {{"missing", 1}};
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({
          {.id = 1, .plan = gatherOutput()},
          consumer,
      }),
      "Exchange node not found: missing, stage: 2");
}

TEST_F(MultiStagePlanTest, producerNotPartitionedOutput) {
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({
          {.id = 1, .type = StageType::kSource, .plan = values()},
          reader(2, StageType::kSingle, std::nullopt, 1),
      }),
      "Expected PartitionedOutputNode at root of producer stage: 1");
}

TEST_F(MultiStagePlanTest, partitionCountMismatch) {
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({
          {.id = 1, .type = StageType::kSource, .plan = partitionedOutput(4)},
          reader(2, StageType::kFixed, 8, 1),
      }),
      "Partition count mismatch between producer 1 and consumer 2");
}

TEST_F(MultiStagePlanTest, broadcastSkipsWidthCheck) {
  EXPECT_NO_THROW(MultiStagePlan({
      {.id = 1, .type = StageType::kSource, .plan = broadcastOutput()},
      reader(2, StageType::kSingle, std::nullopt, 1),
  }));
}

TEST_F(MultiStagePlanTest, exchangeNotFed) {
  const auto plan = exchange();
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({{.id = 1, .type = StageType::kSingle, .plan = plan}}),
      fmt::format(
          "Exchange must be fed by a stage input: {}, stage: 1", plan->id()));
}

TEST_F(MultiStagePlanTest, exchangeSchemaMismatch) {
  Stage consumer{
      .id = 2,
      .type = StageType::kSingle,
      .plan = PlanBuilder(idGenerator_)
                  .exchange(
                      ROW("a", VARCHAR()),
                      VectorSerde::kindName(VectorSerde::Kind::kPresto))
                  .planNode(),
  };
  consumer.inputs.push_back({consumer.plan->id(), 1});

  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({
          {.id = 1, .type = StageType::kSource, .plan = gatherOutput()},
          consumer,
      }),
      "Exchange must read the schema its producer writes");
}

TEST_F(MultiStagePlanTest, exchangeFedTwice) {
  Stage consumer = reader(3, StageType::kSingle, std::nullopt, 1);
  consumer.inputs.push_back({consumer.plan->id(), 2});

  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({
          {.id = 1, .type = StageType::kSource, .plan = gatherOutput()},
          {.id = 2, .type = StageType::kSource, .plan = gatherOutput()},
          consumer,
      }),
      fmt::format(
          "Exchange must be fed by one stage input: {}, stage: 3",
          consumer.plan->id()));
}

TEST_F(MultiStagePlanTest, root) {
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({
          {.id = 1, .type = StageType::kSource, .plan = partitionedOutput(4)},
          {.id = 2,
           .type = StageType::kFixed,
           .numRemotePartitions = 4,
           .plan = values()},
      }),
      "Plan must have one stage that no stage reads: 1, 2");
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({relay(1, 2), relay(2, 1)}),
      "Plan must have one stage that no stage reads");
}

TEST_F(MultiStagePlanTest, producerReadByMultipleConsumers) {
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({
          {.id = 1, .type = StageType::kSource, .plan = partitionedOutput(4)},
          reader(2, StageType::kFixed, 4, 1),
          reader(3, StageType::kFixed, 4, 1),
      }),
      "Producer stage read by multiple consumers: 1");
}

TEST_F(MultiStagePlanTest, stagesFeedRoot) {
  // Stages 2 and 3 read each other and feed nothing else.
  VELOX_ASSERT_USER_THROW(
      MultiStagePlan({
          {.id = 1, .type = StageType::kSingle, .plan = values()},
          relay(2, 3),
          relay(3, 2),
      }),
      "Stage must feed the root stage: 2");
}

} // namespace
} // namespace facebook::velox::core
