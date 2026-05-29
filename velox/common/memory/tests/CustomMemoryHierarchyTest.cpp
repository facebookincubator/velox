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
#include <gtest/gtest.h>

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

class CustomMemoryHierarchyTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    MemoryManager::testingSetInstance(MemoryManager::Options{});
  }

  void SetUp() override {
    vectorPool_ = memoryManager()->addLeafPool("test-vec");
  }

  std::shared_ptr<CustomMemoryResource> makeResource(
      const std::string& tag,
      int64_t capacity = 1L << 30) {
    MemoryAllocator::Options options;
    options.capacity = capacity;
    return std::make_shared<CustomMemoryResource>(
        tag,
        std::make_shared<MallocAllocator>(options),
        MemoryArbitrator::create({}),
        []() { return MemoryReclaimer::create(0); },
        capacity);
  }

  // Builds a QueryCtx with a custom root pool per tag, installs an isolated
  // per-query CustomMemoryResourceRegistry on the QueryCtx so tests do not
  // contend on the global registry, and inserts each backing resource into
  // it so Task can resolve it at construction time.
  std::shared_ptr<core::QueryCtx> buildQueryCtx(
      const std::vector<std::string>& tags,
      const std::string& queryId) {
    auto* manager = memoryManager();
    auto builder = core::QueryCtx::Builder().queryId(queryId);
    std::vector<std::shared_ptr<CustomMemoryResource>> resources;
    for (const auto& tag : tags) {
      auto resource = makeResource(tag);
      builder.customPool(
          tag,
          manager->addCustomRootPool(
              fmt::format("{}.{}", queryId, tag), resource));
      resources.push_back(resource);
    }
    auto queryCtx = builder.build();
    auto registry = CustomMemoryResourceRegistry::createRegistry(nullptr);
    queryCtx->setRegistry<CustomMemoryResourceRegistry::Registry>(
        kCustomMemoryResourceRegistryKey, registry);
    for (size_t i = 0; i < tags.size(); ++i) {
      registry->insert(tags[i], resources[i]);
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
    return exec::Task::create(
        taskId,
        makePlan(),
        /*destination=*/0,
        queryCtx,
        exec::Task::ExecutionMode::kSerial,
        exec::Consumer{});
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
};

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
  auto task = makeTask("t3", queryCtx);
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
