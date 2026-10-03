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
#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>
#include <functional>

#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Driver.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::memory::test {
namespace {

class TestTaskReclaimer final : public exec::MemoryReclaimer {
 public:
  explicit TestTaskReclaimer(int32_t priority)
      : exec::MemoryReclaimer(priority) {}
};

class TestLeafReclaimer final : public exec::MemoryReclaimer {
 public:
  explicit TestLeafReclaimer(std::function<uint64_t(MemoryPool*)> reclaim)
      : exec::MemoryReclaimer(0), reclaim_(std::move(reclaim)) {}

  bool reclaimableBytes(const MemoryPool& pool, uint64_t& bytes)
      const override {
    bytes = pool.usedBytes();
    return bytes != 0;
  }

  uint64_t reclaim(
      MemoryPool* pool,
      uint64_t,
      uint64_t,
      MemoryReclaimer::Stats&) override {
    return reclaim_(pool);
  }

 private:
  const std::function<uint64_t(MemoryPool*)> reclaim_;
};

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
      int64_t capacity = 1L << 30,
      bool useExecutionFactories = false) {
    MemoryAllocator::Options options;
    options.capacity = capacity;
    CustomMemoryResource::ExecutionReclaimerFactories factories;
    if (useExecutionFactories) {
      factories.query = core::QueryCtx::MemoryReclaimer::create;
      factories.task = exec::Task::MemoryReclaimer::create;
    }
    return std::make_shared<CustomMemoryResource>(
        tag,
        std::make_shared<MallocAllocator>(options),
        MemoryArbitrator::create({}),
        []() { return exec::MemoryReclaimer::create(); },
        capacity,
        std::move(factories));
  }

  // Models extension-owned setup; QueryCtx registration itself is passive.
  static void addResourcePool(
      core::QueryCtx& query,
      const CustomMemoryResource& resource,
      std::shared_ptr<MemoryPool> root) {
    if (resource.hasQueryReclaimerFactory() && root->reclaimer() == nullptr) {
      auto reclaimer = resource.newQueryReclaimer(&query, root.get());
      if (reclaimer != nullptr) {
        root->setReclaimer(std::move(reclaimer));
      }
    }
    query.addCustomPool(resource.tag(), std::move(root));
  }

  // Builds a QueryCtx with a custom root pool per tag, installs an isolated
  // per-query CustomMemoryResourceRegistry on the QueryCtx so tests do not
  // contend on the global registry, and inserts each backing resource into
  // it so Task can resolve it at construction time.
  std::shared_ptr<core::QueryCtx> buildQueryCtx(
      const std::vector<std::string>& tags,
      const std::string& queryId,
      bool useExecutionFactories = false) {
    auto* manager = memoryManager();
    auto queryCtx = core::QueryCtx::Builder().queryId(queryId).build();
    auto registry = CustomMemoryResourceRegistry::createRegistry(nullptr);
    queryCtx->setRegistry<CustomMemoryResourceRegistry::Registry>(
        kCustomMemoryResourceRegistryKey, registry);
    for (const auto& tag : tags) {
      auto resource = makeResource(tag, 1L << 30, useExecutionFactories);
      const auto name = fmt::format("{}.{}", queryId, tag);
      auto root = useExecutionFactories
          ? manager->addCustomRootPool(
                name,
                resource->allocator(),
                resource->arbitrator(),
                resource->maxCapacity())
          : manager->addCustomRootPool(name, resource);
      addResourcePool(*queryCtx, *resource, std::move(root));
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
      const std::shared_ptr<core::QueryCtx>& queryCtx) {
    auto task = exec::Task::create(
        taskId,
        makePlan(),
        /*destination=*/0,
        queryCtx,
        exec::Task::ExecutionMode::kSerial,
        exec::Consumer{});
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

TEST_F(
    CustomMemoryHierarchyTest,
    legacyCxlCodeUsesOriginalFactoryAtEveryLevel) {
  MemoryAllocator::Options options;
  options.capacity = 1L << 30;
  int calls{0};
  // Deliberately use only the original constructor, builder and factory API.
  auto resource = std::make_shared<CustomMemoryResource>(
      "cxl",
      std::make_shared<MallocAllocator>(options),
      MemoryArbitrator::create({}),
      [&]() {
        ++calls;
        return MemoryReclaimer::create(17);
      },
      options.capacity);
  auto root = memoryManager()->addCustomRootPool("legacy.cxl", resource);
  ASSERT_EQ(calls, 1);
  auto* original = root->reclaimer();
  auto query = core::QueryCtx::Builder()
                   .queryId("legacy-cxl")
                   .customPool("cxl", root)
                   .build();
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(root->reclaimer(), original);
  auto registry = CustomMemoryResourceRegistry::createRegistry(nullptr);
  registry->insert("cxl", resource);
  query->setRegistry<CustomMemoryResourceRegistry::Registry>(
      kCustomMemoryResourceRegistryKey, registry);
  auto task = makeTask("legacy-cxl-task", query);
  // Task creation also constructs the Values plan's node.0 pool.
  ASSERT_EQ(calls, 3);
  auto* taskPool = findChild(root.get(), "task.legacy-cxl-task.cxl");
  ASSERT_NE(taskPool, nullptr);
  ASSERT_NE(taskPool->reclaimer(), nullptr);
  EXPECT_EQ(taskPool->reclaimer()->priority(), 17);
  auto* node = task->getOrAddCustomNodePool("cxl", "n0");
  EXPECT_EQ(calls, 4);
  EXPECT_EQ(node->reclaimer()->priority(), 17);
  auto* join = task->getOrAddCustomJoinNodePool("cxl", "j0", 7);
  EXPECT_EQ(calls, 5);
  EXPECT_EQ(join->reclaimer()->priority(), 17);
}

TEST_F(
    CustomMemoryHierarchyTest,
    taskFactoryReceivesContextAndSelectsCustomType) {
  MemoryAllocator::Options options;
  options.capacity = 1L << 30;
  int queryCalls{0};
  int taskCalls{0};
  int legacyCalls{0};
  exec::Task* observedTask{nullptr};
  auto resource = std::make_shared<CustomMemoryResource>(
      "device",
      std::make_shared<MallocAllocator>(options),
      MemoryArbitrator::create({}),
      [&]() {
        ++legacyCalls;
        return MemoryReclaimer::create(3);
      },
      options.capacity,
      CustomMemoryResource::ExecutionReclaimerFactories{
          .query =
              [&](core::QueryCtx* query, MemoryPool* root) {
                ++queryCalls;
                return core::QueryCtx::MemoryReclaimer::create(query, root);
              },
          .task =
              [&](const std::shared_ptr<exec::Task>& task,
                  int64_t priority,
                  const std::string& tag) {
                ++taskCalls;
                observedTask = task.get();
                EXPECT_EQ(tag, "device");
                EXPECT_EQ(priority, 0);
                return std::make_unique<TestTaskReclaimer>(29);
              }});
  auto query = core::QueryCtx::Builder().queryId("mixed-factories").build();
  auto root = memoryManager()->addCustomRootPool(
      "mixed.device",
      resource->allocator(),
      resource->arbitrator(),
      resource->maxCapacity());
  addResourcePool(*query, *resource, root);
  auto registry = CustomMemoryResourceRegistry::createRegistry(nullptr);
  registry->insert("device", resource);
  query->setRegistry<CustomMemoryResourceRegistry::Registry>(
      kCustomMemoryResourceRegistryKey, registry);
  auto task = makeTask("mixed-task", query);
  EXPECT_EQ(queryCalls, 1);
  EXPECT_EQ(taskCalls, 1);
  EXPECT_EQ(legacyCalls, 1); // Values node.0.
  EXPECT_EQ(observedTask, task.get());
  auto* taskPool = findChild(root.get(), "task.mixed-task.device");
  ASSERT_NE(taskPool, nullptr);
  EXPECT_NE(dynamic_cast<TestTaskReclaimer*>(taskPool->reclaimer()), nullptr);
  EXPECT_EQ(taskPool->reclaimer()->priority(), 29);
  auto* node = task->getOrAddCustomNodePool("device", "n0");
  EXPECT_EQ(node->reclaimer()->priority(), 3);
  EXPECT_EQ(legacyCalls, 2);
}

TEST_F(CustomMemoryHierarchyTest, nullTaskFactoryResultDoesNotFallBack) {
  MemoryAllocator::Options options;
  options.capacity = 1L << 30;
  int legacyCalls{0};
  int taskCalls{0};
  auto resource = std::make_shared<CustomMemoryResource>(
      "device",
      std::make_shared<MallocAllocator>(options),
      MemoryArbitrator::create({}),
      [&]() {
        ++legacyCalls;
        return std::unique_ptr<MemoryReclaimer>{};
      },
      options.capacity,
      CustomMemoryResource::ExecutionReclaimerFactories{
          .query = core::QueryCtx::MemoryReclaimer::create,
          .task = [&](const std::shared_ptr<exec::Task>&,
                      int64_t,
                      const std::string&) {
            ++taskCalls;
            return std::unique_ptr<MemoryReclaimer>{};
          }});
  auto query = core::QueryCtx::Builder().queryId("null-task").build();
  auto root = memoryManager()->addCustomRootPool(
      "null.device",
      resource->allocator(),
      resource->arbitrator(),
      resource->maxCapacity());
  addResourcePool(*query, *resource, root);
  auto registry = CustomMemoryResourceRegistry::createRegistry(nullptr);
  registry->insert("device", resource);
  query->setRegistry<CustomMemoryResourceRegistry::Registry>(
      kCustomMemoryResourceRegistryKey, registry);
  auto task = makeTask("null-task", query);
  auto* taskPool = findChild(root.get(), "task.null-task.device");
  ASSERT_NE(taskPool, nullptr);
  EXPECT_EQ(taskPool->reclaimer(), nullptr);
  EXPECT_EQ(legacyCalls, 1);
  EXPECT_EQ(taskCalls, 1);
}

TEST_F(CustomMemoryHierarchyTest, builtInTaskReclaimsOnlySelectedResource) {
  auto query = buildQueryCtx({"gpu", "cxl"}, "selected", true);
  auto task = makeTask("selected-task", query);
  std::vector<MemoryPool*> leaves;
  std::vector<void*> allocations(3, nullptr);
  int calls[3]{0, 0, 0};
  auto cpuLeaf = task->pool()->addLeafChild("cpu-sentinel");
  leaves.push_back(cpuLeaf.get());
  for (const auto* tag : {"gpu", "cxl"}) {
    leaves.push_back(task->addCustomOperatorPool(
        tag, "n0", exec::kUngroupedGroupId, 0, 0, "Project"));
  }
  constexpr int64_t kBytes = 4096;
  auto cleanup = folly::makeGuard([&]() {
    for (size_t i = 0; i < leaves.size(); ++i) {
      if (allocations[i] != nullptr) {
        leaves[i]->free(allocations[i], kBytes);
      }
    }
  });
  for (size_t i = 0; i < leaves.size(); ++i) {
    leaves[i]->setReclaimer(
        std::make_unique<TestLeafReclaimer>(
            [&, i](MemoryPool* pool) -> uint64_t {
              EXPECT_TRUE(task->pauseRequested());
              EXPECT_TRUE(query->testingUnderArbitration());
              ++calls[i];
              pool->free(allocations[i], kBytes);
              allocations[i] = nullptr;
              return kBytes;
            }));
    allocations[i] = leaves[i]->allocate(kBytes);
  }
  MemoryReclaimer::Stats stats;
  {
    ScopedMemoryArbitrationContext context;
    EXPECT_EQ(query->customPool("gpu")->reclaim(kBytes, 1000, stats), kBytes);
  }
  EXPECT_EQ(calls[0], 0);
  EXPECT_EQ(calls[1], 1);
  EXPECT_EQ(calls[2], 0);
  EXPECT_EQ(leaves[0]->usedBytes(), kBytes);
  EXPECT_EQ(leaves[1]->usedBytes(), 0);
  EXPECT_EQ(leaves[2]->usedBytes(), kBytes);
  EXPECT_FALSE(task->pauseRequested());
  EXPECT_FALSE(query->testingUnderArbitration());
  EXPECT_EQ(task->taskStats().memoryReclaimCount, 1);
}

TEST_F(CustomMemoryHierarchyTest, explicitFactoriesUseExecutionHierarchy) {
  auto queryCtx = buildQueryCtx(
      {"gpu"}, "execution-reclaimers", /*useExecutionFactories=*/true);
  auto task = makeTask("execution-reclaimers-task", queryCtx);

  auto root = queryCtx->customPool("gpu");
  ASSERT_NE(root, nullptr);
  EXPECT_NE(root->reclaimer(), nullptr);

  auto* taskPool = findChild(root.get(), "task.execution-reclaimers-task.gpu");
  ASSERT_NE(taskPool, nullptr);
  EXPECT_NE(taskPool->reclaimer(), nullptr);

  auto* nodePool = task->getOrAddCustomNodePool("gpu", "n0");
  ASSERT_NE(nodePool, nullptr);
  EXPECT_NE(nodePool->reclaimer(), nullptr);

  auto* leaf = task->addCustomOperatorPool(
      "gpu", "n0", exec::kUngroupedGroupId, 0, 0, "Project");
  ASSERT_NE(leaf, nullptr);
  EXPECT_EQ(leaf->reclaimer(), nullptr)
      << "Custom operator leaves are owned by the accelerator operator base";
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
  auto resource = makeResource("gpu");
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
