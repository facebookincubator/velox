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

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <functional>

#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/connectors/Connector.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::memory::test {
namespace {

// Reports reclaimable bytes on an empty leaf, because parent reclaims skip
// leaves that report none, and passes each reclaim to 'onReclaim' so a test
// can see which leaves a reclaim reached.
class TestLeafReclaimer final : public exec::MemoryReclaimer {
 public:
  explicit TestLeafReclaimer(std::function<void(MemoryPool*)> onReclaim)
      : exec::MemoryReclaimer(0), onReclaim_(std::move(onReclaim)) {}

  bool reclaimableBytes(const MemoryPool& /*pool*/, uint64_t& bytes)
      const override {
    bytes = 1;
    return true;
  }

  uint64_t reclaim(
      MemoryPool* pool,
      uint64_t /*targetBytes*/,
      uint64_t /*maxWaitMs*/,
      MemoryReclaimer::Stats& /*stats*/) override {
    onReclaim_(pool);
    return 0;
  }

 private:
  const std::function<void(MemoryPool*)> onReclaim_;
};

// Uses the built-in query and task reclaimers, so arbitration on a custom
// root coordinates with the query and pauses the Task as it does for the CPU
// tree.
std::unique_ptr<MemoryReclaimer> makeExecutionReclaimer(
    const ReclaimerContext& context) {
  if (const auto* query = std::get_if<QueryReclaimerContext>(&context)) {
    return core::QueryCtx::MemoryReclaimer::create(
        query->queryCtx, query->pool);
  }
  if (const auto* task = std::get_if<TaskReclaimerContext>(&context)) {
    return exec::Task::MemoryReclaimer::create(
        task->task, task->priority, std::string(task->resourceTag));
  }
  return exec::MemoryReclaimer::create();
}

class CustomMemoryHierarchyTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    MemoryManager::testingSetInstance(MemoryManager::Options{});
  }

  void SetUp() override {
    vectorPool_ = memoryManager()->addLeafPool("test-vec");
  }

  void TearDown() override {
    for (const auto& task : tasks_) {
      task->requestCancel();
    }
    tasks_.clear();
  }

  std::shared_ptr<CustomMemoryResource> makeResource(
      const std::string& tag,
      CustomMemoryResource::ReclaimerFactory reclaimerFactory) {
    constexpr int64_t kCapacity{1L << 30};
    MemoryAllocator::Options options;
    options.capacity = kCapacity;
    return std::make_shared<CustomMemoryResource>(
        tag,
        std::make_shared<MallocAllocator>(options),
        MemoryArbitrator::create({}),
        std::move(reclaimerFactory),
        kCapacity);
  }

  // Builds a QueryCtx with one custom root per tag, attached as an extension
  // would. Roots come from the component overload because a resource-built
  // root already has a pool-context reclaimer, and setReclaimer cannot replace
  // it with the query reclaimer. The query reclaimer must be installed before
  // any Task exists, since a Task's custom pools can have reclaimers only if
  // the root has one. Resources go into an isolated per-query registry, where
  // Task resolves them, so tests do not contend on the global registry.
  std::shared_ptr<core::QueryCtx> buildQueryCtx(
      const std::vector<std::string>& tags,
      const std::string& queryId,
      const CustomMemoryResource::ReclaimerFactory& reclaimerFactory =
          makeExecutionReclaimer) {
    auto queryCtx = core::QueryCtx::Builder().queryId(queryId).build();
    auto registry = CustomMemoryResourceRegistry::createRegistry(nullptr);
    queryCtx->setRegistry<CustomMemoryResourceRegistry::Registry>(
        kCustomMemoryResourceRegistryKey, registry);
    for (const auto& tag : tags) {
      auto resource = makeResource(tag, reclaimerFactory);
      auto root = memoryManager()->addCustomRootPool(
          fmt::format("{}.{}", queryId, tag),
          resource->allocator(),
          resource->arbitrator(),
          resource->maxCapacity());
      // setReclaimer rejects nullptr, which a factory may return.
      if (auto reclaimer = resource->newReclaimer(
              QueryReclaimerContext{queryCtx.get(), root.get()})) {
        root->setReclaimer(std::move(reclaimer));
      }
      queryCtx->addCustomPool(tag, std::move(root));
      registry->insert(tag, std::move(resource));
    }
    return queryCtx;
  }

  // Returns a single-row Values plan. Vector data is allocated from
  // 'vectorPool_', which outlives every task built off this plan because
  // it is a fixture member destroyed after the test body.
  core::PlanFragment makePlan() {
    auto rowType = ROW({"a"}, {BIGINT()});
    auto rowVector =
        BaseVector::create<RowVector>(rowType, /*size=*/1, vectorPool_.get());
    return exec::test::PlanBuilder().values({rowVector}).planFragment();
  }

  std::shared_ptr<exec::Task> makeTask(
      const std::string& taskId,
      const std::shared_ptr<core::QueryCtx>& queryCtx,
      int32_t memoryArbitrationPriority = 0) {
    auto task = exec::Task::create(
        taskId,
        makePlan(),
        /*destination=*/0,
        queryCtx,
        exec::Task::ExecutionMode::kSerial,
        exec::Consumer{},
        memoryArbitrationPriority);
    // Track so TearDown can cancel it and release the Task<->Driver cycle.
    tasks_.push_back(task);
    return task;
  }

  // Returns the first child of 'pool' whose name matches 'name', or nullptr.
  static MemoryPool* findChild(MemoryPool* pool, const std::string& name) {
    MemoryPool* found = nullptr;
    pool->visitChildren([&](MemoryPool* child) {
      if (child->name() == name) {
        found = child;
        return false;
      }
      return true;
    });
    return found;
  }

  std::shared_ptr<MemoryPool> vectorPool_;

  // Tasks created via makeTask, cancelled in TearDown to break the
  // Task<->Driver cycle and release their pools.
  std::vector<std::shared_ptr<exec::Task>> tasks_;
};

// Task creation passes the Task, its arbitration priority and the resource
// tag, and installs exactly what the factory returns, so an extension can
// choose its own task-level reclaimer. Node pools keep the pool context.
TEST_F(CustomMemoryHierarchyTest, taskPoolUsesTaskContextReclaimer) {
  constexpr int32_t kPriority{7};
  exec::Task* contextTask{nullptr};
  MemoryReclaimer* taskReclaimer{nullptr};
  MemoryReclaimer* poolReclaimer{nullptr};
  auto queryCtx = buildQueryCtx(
      {"device"}, "task-context", [&](const ReclaimerContext& context) {
        auto reclaimer = makeExecutionReclaimer(context);
        if (const auto* taskContext =
                std::get_if<TaskReclaimerContext>(&context)) {
          contextTask = taskContext->task.get();
          EXPECT_EQ(taskContext->priority, kPriority);
          EXPECT_EQ(taskContext->resourceTag, "device");
          taskReclaimer = reclaimer.get();
        } else if (std::holds_alternative<PoolReclaimerContext>(context)) {
          poolReclaimer = reclaimer.get();
        }
        return reclaimer;
      });
  auto task = makeTask("t-context", queryCtx, kPriority);
  EXPECT_EQ(contextTask, task.get());

  auto* taskPool =
      findChild(queryCtx->customPool("device").get(), "task.t-context.device");
  ASSERT_NE(taskPool, nullptr);
  EXPECT_EQ(taskPool->reclaimer(), taskReclaimer);
  auto* nodePool = task->getOrAddCustomNodePool("device", "n0");
  EXPECT_EQ(nodePool->reclaimer(), poolReclaimer);
}

// A nullptr task-context result leaves the task pool without a reclaimer, so
// a resource can keep its task subtrees out of arbitration. Task must not
// substitute a default.
TEST_F(CustomMemoryHierarchyTest, nullTaskFactoryResultDoesNotFallBack) {
  // The root gets a reclaimer so that a substituted task reclaimer would be
  // accepted, and caught below, rather than rejected. Node pools need nullptr
  // too, since a child can have a reclaimer only if its parent does.
  auto queryCtx = buildQueryCtx(
      {"device"},
      "null-task",
      [](const ReclaimerContext& context) -> std::unique_ptr<MemoryReclaimer> {
        if (std::holds_alternative<QueryReclaimerContext>(context)) {
          return makeExecutionReclaimer(context);
        }
        return nullptr;
      });
  makeTask("t-null", queryCtx);

  auto* taskPool =
      findChild(queryCtx->customPool("device").get(), "task.t-null.device");
  ASSERT_NE(taskPool, nullptr);
  EXPECT_EQ(taskPool->reclaimer(), nullptr);
}

// A tag-bound Task::MemoryReclaimer pauses the Task and marks the query under
// arbitration as the CPU path does, but reclaims only its custom pool's
// subtree. Reaching the CPU tree would spill state the arbitrator did not
// target.
TEST_F(CustomMemoryHierarchyTest, taskReclaimerReclaimsOnlyItsCustomPool) {
  auto queryCtx = buildQueryCtx({"gpu"}, "selected");
  auto task = makeTask("t-selected", queryCtx);
  // Reclaiming from task->pool() instead of the gpu task pool would reach
  // this leaf.
  auto cpuLeaf = task->pool()->addLeafChild("cpu-sentinel");
  auto* gpuLeaf = task->addCustomOperatorPool(
      "gpu",
      "n0",
      exec::kUngroupedGroupId,
      /*pipelineId=*/0,
      /*driverId=*/0,
      "Project");
  std::vector<MemoryPool*> reclaimedLeaves;
  for (auto* leaf : {cpuLeaf.get(), gpuLeaf}) {
    leaf->setReclaimer(
        std::make_unique<TestLeafReclaimer>([&](MemoryPool* pool) {
          EXPECT_TRUE(task->pauseRequested());
          EXPECT_TRUE(queryCtx->testingUnderArbitration());
          reclaimedLeaves.push_back(pool);
        }));
  }

  MemoryReclaimer::Stats stats;
  // A nonzero wait bounds the pause, so a regression fails instead of hanging.
  queryCtx->customPool("gpu")->reclaim(
      /*targetBytes=*/0, /*maxWaitMs=*/1'000, stats);
  EXPECT_THAT(reclaimedLeaves, testing::ElementsAre(gpuLeaf));
}

TEST_F(CustomMemoryHierarchyTest, connectorContextUsesOperatorCustomPools) {
  for (const auto& tags :
       std::vector<std::vector<std::string>>{{}, {"cxl"}, {"gpu", "cxl"}}) {
    auto queryCtx =
        buildQueryCtx(tags, fmt::format("connector-{}", tags.size()));
    auto task =
        makeTask(fmt::format("connector-task-{}", tags.size()), queryCtx);
    exec::DriverCtx driverCtx(task, 3, 0, exec::kUngroupedGroupId, 0);
    exec::OperatorCtx first(&driverCtx, "scan0", 0, "TableScan");
    exec::OperatorCtx second(&driverCtx, "scan1", 1, "TableScan");
    auto firstConnector =
        first.createConnectorQueryCtx("test", "scan0", nullptr);
    auto secondConnector =
        second.createConnectorQueryCtx("test", "scan1", nullptr);

    EXPECT_EQ(firstConnector->memoryPool(), first.pool());
    EXPECT_EQ(secondConnector->memoryPool(), second.pool());
    EXPECT_EQ(firstConnector->customMemoryPool("missing"), nullptr);
    if (tags.empty()) {
      EXPECT_EQ(firstConnector->customMemoryPool("gpu"), nullptr);
      EXPECT_EQ(firstConnector->customMemoryPool("cxl"), nullptr);
    }
    for (const auto& tag : tags) {
      auto* firstPool = firstConnector->customMemoryPool(tag);
      auto* secondPool = secondConnector->customMemoryPool(tag);
      ASSERT_NE(firstPool, nullptr);
      ASSERT_NE(secondPool, nullptr);
      EXPECT_EQ(firstPool, first.customPool(tag));
      EXPECT_EQ(secondPool, second.customPool(tag));
      EXPECT_NE(firstPool, secondPool);
      EXPECT_EQ(firstPool->kind(), MemoryPool::Kind::kLeaf);
      EXPECT_EQ(firstPool->root(), queryCtx->customPool(tag).get());
      EXPECT_EQ(secondPool->root(), queryCtx->customPool(tag).get());
    }
  }
}

// Task construction creates 'task.<id>.<tag>' aggregate under each
// registered custom root.
TEST_F(CustomMemoryHierarchyTest, taskCreationMirrorsTaskPool) {
  auto queryCtx = buildQueryCtx({"gpu"}, "q1");
  auto task = makeTask("t1", queryCtx);

  auto gpuRoot = queryCtx->customPool("gpu");
  ASSERT_NE(gpuRoot, nullptr);
  auto* taskMirror = findChild(gpuRoot.get(), "task.t1.gpu");
  ASSERT_NE(taskMirror, nullptr);
  EXPECT_EQ(taskMirror->kind(), MemoryPool::Kind::kAggregate);
}

// Multiple tags produce independent mirror subtrees that do not bleed
// into one another.
TEST_F(CustomMemoryHierarchyTest, multipleTagsMirrorIndependently) {
  auto queryCtx = buildQueryCtx({"gpu", "cxl"}, "q2");
  auto task = makeTask("t2", queryCtx);

  EXPECT_NE(
      findChild(queryCtx->customPool("gpu").get(), "task.t2.gpu"), nullptr);
  EXPECT_NE(
      findChild(queryCtx->customPool("cxl").get(), "task.t2.cxl"), nullptr);
  EXPECT_EQ(
      findChild(queryCtx->customPool("gpu").get(), "task.t2.cxl"), nullptr);
  EXPECT_EQ(
      findChild(queryCtx->customPool("cxl").get(), "task.t2.gpu"), nullptr);
}

// With no custom pools registered, Task creation runs the default path
// only — no exceptions and the default subtree is unaffected.
TEST_F(CustomMemoryHierarchyTest, noCustomPoolsRegisteredIsHarmless) {
  auto queryCtx = buildQueryCtx({}, "q3");
  ASSERT_EQ(queryCtx->customPools().size(), 0);
  EXPECT_NO_THROW(makeTask("t3", queryCtx));
}

// getOrAddCustomNodePool creates the aggregate under the task mirror
// and is idempotent for repeated calls.
TEST_F(CustomMemoryHierarchyTest, getOrAddCustomNodePoolIsIdempotent) {
  auto queryCtx = buildQueryCtx({"gpu"}, "q4");
  auto task = makeTask("t4", queryCtx);

  auto* nodePool = task->getOrAddCustomNodePool("gpu", "n0");
  ASSERT_NE(nodePool, nullptr);
  EXPECT_EQ(nodePool->name(), "node.n0.gpu");
  EXPECT_EQ(nodePool->kind(), MemoryPool::Kind::kAggregate);
  EXPECT_EQ(nodePool->parent()->name(), "task.t4.gpu");

  EXPECT_EQ(task->getOrAddCustomNodePool("gpu", "n0"), nodePool);
}

// addCustomOperatorPool returns a fresh leaf parented to the node mirror
// for non-join operator types.
TEST_F(CustomMemoryHierarchyTest, addCustomOperatorPoolReturnsLeaf) {
  auto queryCtx = buildQueryCtx({"gpu"}, "q5");
  auto task = makeTask("t5", queryCtx);

  auto* leaf = task->addCustomOperatorPool(
      "gpu",
      "n0",
      exec::kUngroupedGroupId,
      /*pipelineId=*/0,
      /*driverId=*/0,
      "Project");
  ASSERT_NE(leaf, nullptr);
  EXPECT_EQ(leaf->name(), "op.n0.0.0.Project.gpu");
  EXPECT_EQ(leaf->kind(), MemoryPool::Kind::kLeaf);
  EXPECT_EQ(leaf->parent()->name(), "node.n0.gpu");
}

// HashBuild / HashProbe operator types route through the join-keyed
// node pool, mirroring the default getOrAddJoinNodePool path.
TEST_F(CustomMemoryHierarchyTest, hashJoinOperatorUsesJoinNodeKey) {
  auto queryCtx = buildQueryCtx({"gpu"}, "q6");
  auto task = makeTask("t6", queryCtx);

  auto* leaf = task->addCustomOperatorPool(
      "gpu",
      "n0",
      /*splitGroupId=*/7,
      /*pipelineId=*/0,
      /*driverId=*/0,
      std::string(exec::OperatorType::kHashBuild));
  ASSERT_NE(leaf, nullptr);
  EXPECT_EQ(leaf->parent()->name(), "node.n0[7].gpu");
}

// customNodePool returns the cached pool after creation, and nullptr for
// unknown tags or node ids.
TEST_F(CustomMemoryHierarchyTest, customNodePoolAccessor) {
  auto queryCtx = buildQueryCtx({"gpu"}, "q7");
  auto task = makeTask("t7", queryCtx);

  EXPECT_EQ(task->customNodePool("gpu", "n0"), nullptr);
  auto* nodePool = task->getOrAddCustomNodePool("gpu", "n0");
  EXPECT_EQ(task->customNodePool("gpu", "n0"), nodePool);
  EXPECT_EQ(task->customNodePool("missing-tag", "n0"), nullptr);
  EXPECT_EQ(task->customNodePool("gpu", "missing-node"), nullptr);
}

// Looking up a tag with no registered resource throws clearly during
// task creation. An isolated empty per-query registry is installed so the
// lookup never falls back to the process-global registry (which other
// tests in this process may have populated).
TEST_F(CustomMemoryHierarchyTest, taskCreationFailsWhenResourceMissing) {
  auto* manager = memoryManager();
  auto resource = makeResource("gpu", makeExecutionReclaimer);
  auto pool = manager->addCustomRootPool("q-missing.gpu", resource);
  auto queryCtx = core::QueryCtx::Builder()
                      .customPool("gpu", std::move(pool))
                      .queryId("q-missing")
                      .build();
  queryCtx->setRegistry<CustomMemoryResourceRegistry::Registry>(
      kCustomMemoryResourceRegistryKey,
      CustomMemoryResourceRegistry::createRegistry(nullptr));
  EXPECT_THROW(makeTask("t-missing", queryCtx), VeloxRuntimeError);
}

} // namespace
} // namespace facebook::velox::memory::test
