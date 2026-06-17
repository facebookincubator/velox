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
#include "velox/core/PlanFragment.h"
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
        fmt::format("mgr-{}", i),
        std::make_shared<OutputBufferManagerEntry>(
            OutputBufferManagerEntry{std::move(mgr), {}}));
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

TEST(OutputBufferManagerRegistryTest, selfRegistrationUsesHttpTransport) {
  // Calling getInstanceRef() triggers self-registration into the global
  // registry, even if a prior test cleared it via unregisterAll().
  auto instance = OutputBufferManager::getInstanceRef();

  auto httpMgr = OutputBufferManagerRegistry::tryGet(
      std::string(core::TransportKind::kHttp));
  EXPECT_EQ(httpMgr, instance);

  auto all = OutputBufferManagerRegistry::getAll();
  EXPECT_GE(all.size(), 1);

  bool foundHttp = false;
  for (auto& [key, _] : all) {
    if (key == core::TransportKind::kHttp) {
      foundHttp = true;
    }
  }
  EXPECT_TRUE(foundHttp);
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
  OutputBufferManagerRegistry::global().insert(
      "obm",
      std::make_shared<OutputBufferManagerEntry>(
          OutputBufferManagerEntry{globalMgr, {}}));

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = OutputBufferManagerRegistry::create(
      &OutputBufferManagerRegistry::global());
  auto queryMgr = std::make_shared<MockOutputBufferManager>();
  queryRegistry->insert(
      "obm",
      std::make_shared<OutputBufferManagerEntry>(
          OutputBufferManagerEntry{queryMgr, {}}));
  queryCtx->setRegistry(
      OutputBufferManagerRegistry::kRegistryKey, queryRegistry);

  EXPECT_EQ(OutputBufferManagerRegistry::tryGet(*queryCtx, "obm"), queryMgr);
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("obm"), globalMgr);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, containsIsPresenceOnly) {
  auto queryCtx = core::QueryCtx::create();
  EXPECT_FALSE(OutputBufferManagerRegistry::contains(*queryCtx, "obm"));

  OutputBufferManagerRegistry::global().insert(
      "obm",
      std::make_shared<OutputBufferManagerEntry>(OutputBufferManagerEntry{
          std::make_shared<MockOutputBufferManager>(), {}}));

  EXPECT_TRUE(OutputBufferManagerRegistry::contains(*queryCtx, "obm"));

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, availabilityPredicateGatesTryGet) {
  auto manager = std::make_shared<MockOutputBufferManager>();
  // Predicate denies availability for every query.
  OutputBufferManagerRegistry::global().insert(
      "obm",
      std::make_shared<OutputBufferManagerEntry>(OutputBufferManagerEntry{
          manager, [](const core::QueryCtx&) { return false; }}));

  auto queryCtx = core::QueryCtx::create();
  // Present (contains) but not available (tryGet) for this query.
  EXPECT_TRUE(OutputBufferManagerRegistry::contains(*queryCtx, "obm"));
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet(*queryCtx, "obm"), nullptr);
  // The global, query-less tryGet ignores the predicate (presence only).
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("obm"), manager);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, queryScopedFallbackToGlobal) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert(
      "obm",
      std::make_shared<OutputBufferManagerEntry>(
          OutputBufferManagerEntry{globalMgr, {}}));

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
  OutputBufferManagerRegistry::global().insert(
      "obm",
      std::make_shared<OutputBufferManagerEntry>(
          OutputBufferManagerEntry{globalMgr, {}}));

  auto queryCtx = core::QueryCtx::create();
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet(*queryCtx, "obm"), globalMgr);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, queryScopedUnregisterAll) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert(
      "obm",
      std::make_shared<OutputBufferManagerEntry>(
          OutputBufferManagerEntry{globalMgr, {}}));

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = OutputBufferManagerRegistry::create(
      &OutputBufferManagerRegistry::global());
  queryRegistry->insert(
      "obm",
      std::make_shared<OutputBufferManagerEntry>(OutputBufferManagerEntry{
          std::make_shared<MockOutputBufferManager>(), {}}));
  queryCtx->setRegistry(
      OutputBufferManagerRegistry::kRegistryKey, queryRegistry);

  OutputBufferManagerRegistry::unregisterAll(*queryCtx);

  EXPECT_EQ(OutputBufferManagerRegistry::tryGet(*queryCtx, "obm"), globalMgr);
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("obm"), globalMgr);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, unregisterAllNoQueryRegistry) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert(
      "obm",
      std::make_shared<OutputBufferManagerEntry>(
          OutputBufferManagerEntry{globalMgr, {}}));

  auto queryCtx = core::QueryCtx::create();
  OutputBufferManagerRegistry::unregisterAll(*queryCtx);

  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("obm"), globalMgr);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, queryScopedGetAll) {
  OutputBufferManagerRegistry::global().insert(
      "global-obm",
      std::make_shared<OutputBufferManagerEntry>(OutputBufferManagerEntry{
          std::make_shared<MockOutputBufferManager>(), {}}));

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = OutputBufferManagerRegistry::create(
      &OutputBufferManagerRegistry::global());
  queryRegistry->insert(
      "query-obm",
      std::make_shared<OutputBufferManagerEntry>(OutputBufferManagerEntry{
          std::make_shared<MockOutputBufferManager>(), {}}));
  queryCtx->setRegistry(
      OutputBufferManagerRegistry::kRegistryKey, queryRegistry);

  auto all = OutputBufferManagerRegistry::getAll(*queryCtx);
  EXPECT_EQ(all.size(), 2);

  auto globalOnly = OutputBufferManagerRegistry::getAll();
  EXPECT_EQ(globalOnly.size(), 1);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, queryScopedOverrideWithGetAll) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert(
      "obm",
      std::make_shared<OutputBufferManagerEntry>(
          OutputBufferManagerEntry{globalMgr, {}}));

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = OutputBufferManagerRegistry::create(
      &OutputBufferManagerRegistry::global());
  auto overrideMgr = std::make_shared<MockOutputBufferManager>();
  queryRegistry->insert(
      "obm",
      std::make_shared<OutputBufferManagerEntry>(
          OutputBufferManagerEntry{overrideMgr, {}}));
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

TEST_F(OutputBufferManagerRegistryFixture, createWithNullParentIsolation) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert(
      "obm",
      std::make_shared<OutputBufferManagerEntry>(
          OutputBufferManagerEntry{globalMgr, {}}));

  auto isolated = OutputBufferManagerRegistry::create(nullptr);
  EXPECT_EQ(isolated->find("obm"), nullptr);

  auto localMgr = std::make_shared<MockOutputBufferManager>();
  isolated->insert(
      "local-obm",
      std::make_shared<OutputBufferManagerEntry>(
          OutputBufferManagerEntry{localMgr, {}}));
  EXPECT_EQ(isolated->find("local-obm")->manager, localMgr);
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("local-obm"), nullptr);

  auto snap = isolated->snapshot();
  EXPECT_EQ(snap.size(), 1);
  EXPECT_EQ(snap[0].first, "local-obm");

  OutputBufferManagerRegistry::unregisterAll();
}

} // namespace
} // namespace facebook::velox::exec
