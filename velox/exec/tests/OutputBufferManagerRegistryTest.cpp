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

#include <functional>
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

// Wraps a manager in a registry entry with an optional availability predicate.
std::shared_ptr<OutputBufferManagerEntry> makeEntry(
    std::shared_ptr<IOutputBufferManager> manager,
    std::function<bool(const core::QueryCtx&)> isAvailable = nullptr) {
  return std::make_shared<OutputBufferManagerEntry>(
      std::move(manager), std::move(isAvailable));
}

TEST(OutputBufferManagerRegistryTest, registryOperations) {
  OutputBufferManagerRegistry::unregisterAll();

  const int32_t numManagers = 5;
  for (int32_t i = 0; i < numManagers; i++) {
    auto mgr = std::make_shared<MockOutputBufferManager>();
    OutputBufferManagerRegistry::global().insert(
        fmt::format("mgr-{}", i), makeEntry(std::move(mgr)));
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
  OutputBufferManagerRegistry::global().insert("obm", makeEntry(globalMgr));

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = OutputBufferManagerRegistry::create(
      &OutputBufferManagerRegistry::global());
  auto queryMgr = std::make_shared<MockOutputBufferManager>();
  queryRegistry->insert("obm", makeEntry(queryMgr));
  queryCtx->setRegistry(
      OutputBufferManagerRegistry::kRegistryKey, queryRegistry);

  EXPECT_EQ(OutputBufferManagerRegistry::tryGet(*queryCtx, "obm"), queryMgr);
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("obm"), globalMgr);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, queryScopedFallbackToGlobal) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert("obm", makeEntry(globalMgr));

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
  OutputBufferManagerRegistry::global().insert("obm", makeEntry(globalMgr));

  auto queryCtx = core::QueryCtx::create();
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet(*queryCtx, "obm"), globalMgr);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, queryScopedUnregisterAll) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert("obm", makeEntry(globalMgr));

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = OutputBufferManagerRegistry::create(
      &OutputBufferManagerRegistry::global());
  queryRegistry->insert(
      "obm", makeEntry(std::make_shared<MockOutputBufferManager>()));
  queryCtx->setRegistry(
      OutputBufferManagerRegistry::kRegistryKey, queryRegistry);

  OutputBufferManagerRegistry::unregisterAll(*queryCtx);

  EXPECT_EQ(OutputBufferManagerRegistry::tryGet(*queryCtx, "obm"), globalMgr);
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("obm"), globalMgr);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, unregisterAllNoQueryRegistry) {
  auto globalMgr = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::global().insert("obm", makeEntry(globalMgr));

  auto queryCtx = core::QueryCtx::create();
  OutputBufferManagerRegistry::unregisterAll(*queryCtx);

  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("obm"), globalMgr);

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(OutputBufferManagerRegistryFixture, queryScopedGetAll) {
  OutputBufferManagerRegistry::global().insert(
      "global-obm", makeEntry(std::make_shared<MockOutputBufferManager>()));

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = OutputBufferManagerRegistry::create(
      &OutputBufferManagerRegistry::global());
  queryRegistry->insert(
      "query-obm", makeEntry(std::make_shared<MockOutputBufferManager>()));
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
  OutputBufferManagerRegistry::global().insert("obm", makeEntry(globalMgr));

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = OutputBufferManagerRegistry::create(
      &OutputBufferManagerRegistry::global());
  auto overrideMgr = std::make_shared<MockOutputBufferManager>();
  queryRegistry->insert("obm", makeEntry(overrideMgr));
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
  OutputBufferManagerRegistry::global().insert("obm", makeEntry(globalMgr));

  auto isolated = OutputBufferManagerRegistry::create(nullptr);
  EXPECT_EQ(isolated->find("obm"), nullptr);

  auto localMgr = std::make_shared<MockOutputBufferManager>();
  isolated->insert("local-obm", makeEntry(localMgr));
  EXPECT_EQ(isolated->find("local-obm")->manager, localMgr);
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("local-obm"), nullptr);

  auto snap = isolated->snapshot();
  EXPECT_EQ(snap.size(), 1);
  EXPECT_EQ(snap[0].first, "local-obm");

  OutputBufferManagerRegistry::unregisterAll();
}

TEST_F(
    OutputBufferManagerRegistryFixture,
    availabilityPredicateGatesVisibility) {
  auto mgr = std::make_shared<MockOutputBufferManager>();
  // Registered once (capability present) but gated per query by the predicate.
  bool available = false;
  OutputBufferManagerRegistry::global().insert(
      "ucx", makeEntry(mgr, [&](const core::QueryCtx&) { return available; }));

  auto queryCtx = core::QueryCtx::create();

  // contains() reflects presence regardless of the predicate (capability
  // check).
  EXPECT_TRUE(OutputBufferManagerRegistry::contains("ucx"));
  EXPECT_FALSE(OutputBufferManagerRegistry::contains("nonexistent"));

  // Gated off: the per-query lookups hide the manager, while the predicate-free
  // global tryGet(id) still returns it.
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet(*queryCtx, "ucx"), nullptr);
  EXPECT_TRUE(OutputBufferManagerRegistry::getAll(*queryCtx).empty());
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet("ucx"), mgr);

  // Gated on: now visible per query.
  available = true;
  EXPECT_EQ(OutputBufferManagerRegistry::tryGet(*queryCtx, "ucx"), mgr);
  EXPECT_EQ(OutputBufferManagerRegistry::getAll(*queryCtx).size(), 1);

  OutputBufferManagerRegistry::unregisterAll();
}

} // namespace
} // namespace facebook::velox::exec
