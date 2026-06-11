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

#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/IOutputBufferManager.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/OutputBufferManagerRegistry.h"

namespace facebook::velox::exec {
namespace {

class MockOutputBufferManager : public IOutputBufferManager {
 public:
  int initCount{0};
  int updateBuffersCount{0};
  int updateDriversCount{0};
  int removeCount{0};
  int statsCount{0};

  void initializeTask(
      std::shared_ptr<Task> /*task*/,
      core::PartitionedOutputNode::Kind /*kind*/,
      int /*numDestinations*/,
      int /*numDrivers*/) override {
    ++initCount;
  }

  bool updateOutputBuffers(
      const std::string& /*taskId*/,
      int /*numBuffers*/,
      bool /*noMoreBuffers*/) override {
    ++updateBuffersCount;
    return true;
  }

  bool updateNumDrivers(const std::string& /*taskId*/, uint32_t /*num*/)
      override {
    ++updateDriversCount;
    return true;
  }

  void removeTask(const std::string& /*taskId*/) override {
    ++removeCount;
  }

  std::optional<OutputBuffer::Stats> stats(
      const std::string& /*taskId*/) override {
    ++statsCount;
    return std::nullopt;
  }

  double getUtilization(const std::string& /*taskId*/) override {
    return 0.0;
  }

  bool isOverutilized(const std::string& /*taskId*/) override {
    return false;
  }

  std::string toString(const std::string& /*taskId*/) override {
    return "mock";
  }
};

TEST(OutputBufferManagerRegistryTest, registryOperations) {
  OutputBufferManagerRegistry::unregisterAll();

  const int32_t numManagers = 5;
  for (int32_t i = 0; i < numManagers; i++) {
    auto mgr = std::make_shared<MockOutputBufferManager>();
    OutputBufferManagerRegistry::global().insert(
        fmt::format("mgr-{}", i), std::move(mgr));
  }

  for (int32_t i = 0; i < numManagers; i++) {
    EXPECT_NE(
        OutputBufferManagerRegistry::tryGet(fmt::format("mgr-{}", i)), nullptr);
  }
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("nonexistent"), nullptr);

  auto all = OutputBufferManagerRegistry::getAll();
  EXPECT_EQ(all.size(), numManagers);

  OutputBufferManagerRegistry::unregisterAll();
  EXPECT_TRUE(OutputBufferManagerRegistry::getAll().empty());
}

TEST(OutputBufferManagerRegistryTest, selfRegistration) {
  // Calling getInstanceRef() triggers self-registration into the global
  // registry, even if a prior test cleared it via unregisterAll().
  auto instance = OutputBufferManager::getInstanceRef();

  auto defaultMgr = OutputBufferManagerRegistry::tryGet(
      std::string(OutputBufferManagerRegistry::kDefaultId));
  EXPECT_NE(defaultMgr, nullptr);

  auto all = OutputBufferManagerRegistry::getAll();
  EXPECT_GE(all.size(), 1);

  bool foundDefault = false;
  for (auto& [key, _] : all) {
    if (key == OutputBufferManagerRegistry::kDefaultId) {
      foundDefault = true;
    }
  }
  EXPECT_TRUE(foundDefault);
}

class OutputBufferManagerRegistryFixture : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    OutputBufferManagerRegistry::unregisterAll();
  }

  void TearDown() override {
    OutputBufferManagerRegistry::unregisterAll();
  }
};

TEST_F(OutputBufferManagerRegistryFixture, queryScopedOverride) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert("obm", globalMgr);

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = OutputBufferManagerRegistry::create(
      &OutputBufferManagerRegistry::global());
  auto queryMgr = std::make_shared<MockOutputBufferManager>();
  queryRegistry->insert("obm", queryMgr);
  queryCtx->setRegistry(
      OutputBufferManagerRegistry::kRegistryKey, queryRegistry);

  EXPECT_EQ(OutputBufferManagerRegistry::tryGet(*queryCtx, "obm"), queryMgr);
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("obm"), globalMgr);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, queryScopedFallbackToGlobal) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert("obm", globalMgr);

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = OutputBufferManagerRegistry::create(
      &OutputBufferManagerRegistry::global());
  queryCtx->setRegistry(
      OutputBufferManagerRegistry::kRegistryKey, queryRegistry);

  EXPECT_EQ(OutputBufferManagerRegistry::tryGet(*queryCtx, "obm"), globalMgr);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, noQueryRegistryFallsBackToGlobal) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert("obm", globalMgr);

  auto queryCtx = core::QueryCtx::create();
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet(*queryCtx, "obm"), globalMgr);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, queryScopedUnregisterAll) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert("obm", globalMgr);

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = OutputBufferManagerRegistry::create(
      &OutputBufferManagerRegistry::global());
  queryRegistry->insert("obm", std::make_shared<MockOutputBufferManager>());
  queryCtx->setRegistry(
      OutputBufferManagerRegistry::kRegistryKey, queryRegistry);

  OutputBufferManagerRegistry::unregisterAll(*queryCtx);

  EXPECT_EQ(OutputBufferManagerRegistry::tryGet(*queryCtx, "obm"), globalMgr);
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("obm"), globalMgr);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, unregisterAllNoQueryRegistry) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert("obm", globalMgr);

  auto queryCtx = core::QueryCtx::create();
  OutputBufferManagerRegistry::unregisterAll(*queryCtx);

  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("obm"), globalMgr);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, queryScopedGetAll) {
  OutputBufferManagerRegistry::global().insert(
      "global-obm", std::make_shared<MockOutputBufferManager>());

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = OutputBufferManagerRegistry::create(
      &OutputBufferManagerRegistry::global());
  queryRegistry->insert(
      "query-obm", std::make_shared<MockOutputBufferManager>());
  queryCtx->setRegistry(
      OutputBufferManagerRegistry::kRegistryKey, queryRegistry);

  auto all = OutputBufferManagerRegistry::getAll(*queryCtx);
  EXPECT_EQ(all.size(), 2);

  auto globalOnly = OutputBufferManagerRegistry::getAll();
  EXPECT_EQ(globalOnly.size(), 1);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, lifecycleCallsDispatchedToAll) {
  auto mgr1 = std::make_shared<MockOutputBufferManager>();
  auto mgr2 = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert("mgr1", mgr1);
  OutputBufferManagerRegistry::global().insert("mgr2", mgr2);

  auto managers = OutputBufferManagerRegistry::getAll();
  ASSERT_EQ(managers.size(), 2);

  for (auto& [_, mgr] : managers) {
    mgr->updateOutputBuffers("task-1", 4, false);
    mgr->updateNumDrivers("task-1", 2);
    mgr->removeTask("task-1");
  }

  EXPECT_EQ(mgr1->updateBuffersCount, 1);
  EXPECT_EQ(mgr1->updateDriversCount, 1);
  EXPECT_EQ(mgr1->removeCount, 1);

  EXPECT_EQ(mgr2->updateBuffersCount, 1);
  EXPECT_EQ(mgr2->updateDriversCount, 1);
  EXPECT_EQ(mgr2->removeCount, 1);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, queryScopedOverrideWithGetAll) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert("obm", globalMgr);

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = OutputBufferManagerRegistry::create(
      &OutputBufferManagerRegistry::global());
  auto overrideMgr = std::make_shared<MockOutputBufferManager>();
  queryRegistry->insert("obm", overrideMgr);
  queryCtx->setRegistry(
      OutputBufferManagerRegistry::kRegistryKey, queryRegistry);

  auto queryManagers = OutputBufferManagerRegistry::getAll(*queryCtx);
  EXPECT_EQ(queryManagers.size(), 1);

  bool foundOverride = false;
  for (auto& [key, mgr] : queryManagers) {
    if (key == "obm") {
      EXPECT_EQ(mgr, overrideMgr);
      foundOverride = true;
    }
  }
  EXPECT_TRUE(foundOverride);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST(OutputBufferManagerRegistryTest, statsAdd) {
  OutputBuffer::Stats a(
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*_noMoreBuffers=*/false,
      /*_noMoreData=*/false,
      /*_finished=*/false,
      /*_bufferedBytes=*/100,
      /*_bufferedPages=*/10,
      /*_totalBytesSent=*/1000,
      /*_totalRowsSent=*/50,
      /*_totalPagesSent=*/5,
      /*_averageBufferTimeMs=*/200,
      /*_numTopBuffers=*/2,
      /*_buffersStats=*/{});

  OutputBuffer::Stats b(
      core::PartitionedOutputNode::Kind::kBroadcast,
      /*_noMoreBuffers=*/true,
      /*_noMoreData=*/true,
      /*_finished=*/true,
      /*_bufferedBytes=*/300,
      /*_bufferedPages=*/30,
      /*_totalBytesSent=*/3000,
      /*_totalRowsSent=*/150,
      /*_totalPagesSent=*/15,
      /*_averageBufferTimeMs=*/400,
      /*_numTopBuffers=*/6,
      /*_buffersStats=*/{});

  a.add(b);

  EXPECT_EQ(a.kind, core::PartitionedOutputNode::Kind::kBroadcast);
  EXPECT_TRUE(a.noMoreBuffers);
  EXPECT_TRUE(a.noMoreData);
  EXPECT_TRUE(a.finished);
  EXPECT_EQ(a.bufferedBytes, 400);
  EXPECT_EQ(a.bufferedPages, 40);
  EXPECT_EQ(a.totalBytesSent, 4000);
  EXPECT_EQ(a.totalRowsSent, 200);
  EXPECT_EQ(a.totalPagesSent, 20);
  EXPECT_EQ(a.numTopBuffers, 8);

  // Weighted average: (200*1000 + 400*3000) / (1000+3000) = 1400000/4000 = 350
  EXPECT_EQ(a.averageBufferTimeMs, 350);
}

TEST(OutputBufferManagerRegistryTest, statsAddZeroBytes) {
  OutputBuffer::Stats a(
      core::PartitionedOutputNode::Kind::kPartitioned,
      false,
      false,
      false,
      /*_bufferedBytes=*/0,
      /*_bufferedPages=*/0,
      /*_totalBytesSent=*/0,
      /*_totalRowsSent=*/0,
      /*_totalPagesSent=*/0,
      /*_averageBufferTimeMs=*/0,
      /*_numTopBuffers=*/0,
      /*_buffersStats=*/{});

  OutputBuffer::Stats b(
      core::PartitionedOutputNode::Kind::kPartitioned,
      false,
      false,
      false,
      /*_bufferedBytes=*/0,
      /*_bufferedPages=*/0,
      /*_totalBytesSent=*/0,
      /*_totalRowsSent=*/0,
      /*_totalPagesSent=*/0,
      /*_averageBufferTimeMs=*/0,
      /*_numTopBuffers=*/0,
      /*_buffersStats=*/{});

  a.add(b);
  EXPECT_EQ(a.averageBufferTimeMs, 0);
  EXPECT_EQ(a.totalBytesSent, 0);
}

TEST_F(OutputBufferManagerRegistryFixture, createWithNullParentIsolation) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert("obm", globalMgr);

  auto isolated = OutputBufferManagerRegistry::create(nullptr);
  EXPECT_EQ(isolated->find("obm"), nullptr);

  auto localMgr = std::make_shared<MockOutputBufferManager>();
  isolated->insert("local-obm", localMgr);
  EXPECT_EQ(isolated->find("local-obm"), localMgr);
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("local-obm"), nullptr);

  auto snap = isolated->snapshot();
  EXPECT_EQ(snap.size(), 1);
  EXPECT_EQ(snap[0].first, "local-obm");

  OutputBufferManagerRegistry::unregisterAll();
}

} // namespace
} // namespace facebook::velox::exec
