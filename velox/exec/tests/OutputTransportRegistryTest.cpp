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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/core/PlanFragment.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/OutputTransportRegistry.h"

namespace facebook::velox::exec {
namespace {

using ::testing::Key;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class MockOutputBufferManager : public OutputBufferManager {
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
      int /*numDestinations*/,
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

  std::optional<OutputBufferStats> stats(
      const std::string& /*taskId*/) override {
    ++statsCount;
    return std::nullopt;
  }

  std::optional<double> getUtilization(const std::string& /*taskId*/) override {
    return 0.0;
  }

  std::optional<bool> isOverutilized(const std::string& /*taskId*/) override {
    return false;
  }

  std::string toString(const std::string& /*taskId*/) override {
    return "mock";
  }
};

std::shared_ptr<OutputTransportEntry> makeEntry(
    std::shared_ptr<OutputBufferManager> manager) {
  return std::make_shared<OutputTransportEntry>(
      std::move(manager),
      [](int32_t,
         DriverCtx*,
         const std::shared_ptr<const core::PartitionedOutputNode>&,
         bool) -> std::unique_ptr<Operator> { return nullptr; });
}

TEST(OutputTransportRegistryTest, registryOperations) {
  OutputTransportRegistry::unregisterAll();

  const int32_t numManagers = 5;
  for (int32_t i = 0; i < numManagers; i++) {
    auto manager = std::make_shared<MockOutputBufferManager>();
    OutputTransportRegistry::global().insert(
        fmt::format("manager-{}", i), makeEntry(std::move(manager)));
  }

  for (int32_t i = 0; i < numManagers; i++) {
    EXPECT_NE(
        OutputTransportRegistry::tryGet(fmt::format("manager-{}", i)), nullptr);
  }
  EXPECT_EQ(OutputTransportRegistry::tryGet("nonexistent"), nullptr);

  // getAll() also lists the always-available built-in in-memory default.
  auto all = OutputTransportRegistry::getAll();
  EXPECT_THAT(all, SizeIs(numManagers + 1));

  OutputTransportRegistry::unregisterAll();
  EXPECT_THAT(
      OutputTransportRegistry::getAll(),
      UnorderedElementsAre(Key(std::string(core::TransportKind::kInMemory))));
}

TEST(OutputTransportRegistryTest, defaultTransportResolves) {
  // The built-in in-memory transport is seeded into the registry and resolves
  // to the default manager singleton.
  auto instance = DefaultOutputBufferManager::getInstanceRef();

  auto defaultEntry = OutputTransportRegistry::tryGet(
      std::string(core::TransportKind::kInMemory));
  ASSERT_NE(defaultEntry, nullptr);
  EXPECT_EQ(defaultEntry->manager, instance);
}

class OutputTransportRegistryFixture : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    OutputTransportRegistry::unregisterAll();
  }

  void TearDown() override {
    OutputTransportRegistry::unregisterAll();
  }

  std::shared_ptr<core::QueryCtx> queryCtxWithRegistry(
      std::shared_ptr<OutputTransportRegistry::Registry> registry) {
    auto queryCtx = core::QueryCtx::create();
    queryCtx->setRegistry(
        OutputTransportRegistry::kRegistryKey, std::move(registry));
    return queryCtx;
  }
};

TEST_F(OutputTransportRegistryFixture, queryScopedResolution) {
  auto globalManager = std::make_shared<MockOutputBufferManager>();
  OutputTransportRegistry::global().insert("shared", makeEntry(globalManager));
  OutputTransportRegistry::global().insert(
      "global-only", makeEntry(globalManager));

  EXPECT_EQ(
      OutputTransportRegistry::tryGet(*core::QueryCtx::create(), "shared")
          ->manager,
      globalManager);

  auto queryManager = std::make_shared<MockOutputBufferManager>();
  auto queryRegistry =
      OutputTransportRegistry::create(&OutputTransportRegistry::global());
  queryRegistry->insert("shared", makeEntry(queryManager));
  auto queryCtx = queryCtxWithRegistry(queryRegistry);

  EXPECT_EQ(
      OutputTransportRegistry::tryGet(*queryCtx, "shared")->manager,
      queryManager);
  EXPECT_EQ(
      OutputTransportRegistry::tryGet(*queryCtx, "global-only")->manager,
      globalManager);
  EXPECT_EQ(OutputTransportRegistry::tryGet("shared")->manager, globalManager);
}

TEST_F(OutputTransportRegistryFixture, queryScopedUnregisterAll) {
  auto globalManager = std::make_shared<MockOutputBufferManager>();
  OutputTransportRegistry::global().insert("obm", makeEntry(globalManager));

  auto queryRegistry =
      OutputTransportRegistry::create(&OutputTransportRegistry::global());
  queryRegistry->insert(
      "obm", makeEntry(std::make_shared<MockOutputBufferManager>()));
  auto queryCtx = queryCtxWithRegistry(queryRegistry);

  OutputTransportRegistry::unregisterAll(*queryCtx);

  EXPECT_EQ(
      OutputTransportRegistry::tryGet(*queryCtx, "obm")->manager,
      globalManager);
  EXPECT_EQ(OutputTransportRegistry::tryGet("obm")->manager, globalManager);
}

TEST_F(OutputTransportRegistryFixture, queryScopedGetAll) {
  auto globalManager = std::make_shared<MockOutputBufferManager>();
  OutputTransportRegistry::global().insert(
      "global-only", makeEntry(globalManager));
  OutputTransportRegistry::global().insert("shared", makeEntry(globalManager));

  auto queryRegistry =
      OutputTransportRegistry::create(&OutputTransportRegistry::global());
  queryRegistry->insert(
      "query-only", makeEntry(std::make_shared<MockOutputBufferManager>()));
  queryRegistry->insert(
      "shared", makeEntry(std::make_shared<MockOutputBufferManager>()));
  auto queryCtx = queryCtxWithRegistry(queryRegistry);

  // getAll() also lists the always-available built-in in-memory default.
  const std::string inMemory{core::TransportKind::kInMemory};
  EXPECT_THAT(
      OutputTransportRegistry::getAll(*queryCtx),
      UnorderedElementsAre(
          Key("global-only"), Key("query-only"), Key("shared"), Key(inMemory)));
  EXPECT_THAT(
      OutputTransportRegistry::getAll(),
      UnorderedElementsAre(Key("global-only"), Key("shared"), Key(inMemory)));
}

TEST_F(OutputTransportRegistryFixture, isolatedQueryHasNoDefault) {
  // Isolation mode (create(nullptr)) has no parent fallback, so not even the
  // built-in default is visible; an isolated query must register every
  // transport it uses.
  auto queryCtx =
      queryCtxWithRegistry(OutputTransportRegistry::create(nullptr));

  EXPECT_EQ(
      OutputTransportRegistry::tryGet(
          *queryCtx, std::string(core::TransportKind::kInMemory)),
      nullptr);
}

} // namespace
} // namespace facebook::velox::exec
