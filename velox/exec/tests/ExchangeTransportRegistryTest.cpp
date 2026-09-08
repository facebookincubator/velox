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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/ExchangeTransportRegistry.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {
namespace {

using ::testing::Key;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

// Minimal control-plane-only client, standing in for a transport's client
// without needing an ExchangeSource or an executor.
class MockExchangeClient : public ExchangeClient {
 public:
  void addRemoteTaskId(const std::string& /*remoteTaskId*/) override {}

  void noMoreRemoteTasks() override {}

  void close() override {}

  folly::F14FastMap<std::string, RuntimeMetric> stats() override {
    return {};
  }

  std::string toString() const override {
    return "mock";
  }

  folly::dynamic toJson() const override {
    return folly::dynamic::object;
  }
};

// Client of a different transport: unrelated to MockExchangeClient, so a cast
// from one to the other must fail.
class UnrelatedExchangeClient : public ExchangeClient {
 public:
  void addRemoteTaskId(const std::string& /*remoteTaskId*/) override {}

  void noMoreRemoteTasks() override {}

  void close() override {}

  folly::F14FastMap<std::string, RuntimeMetric> stats() override {
    return {};
  }

  std::string toString() const override {
    return "unrelated";
  }

  folly::dynamic toJson() const override {
    return folly::dynamic::object;
  }
};

std::shared_ptr<MockExchangeClient> makeMockClient(
    const ExchangeClientContext& /*context*/) {
  return std::make_shared<MockExchangeClient>();
}

std::unique_ptr<Operator> buildNoOperator(
    int32_t /*operatorId*/,
    DriverCtx* /*ctx*/,
    const std::shared_ptr<const core::ExchangeNode>& /*node*/,
    const std::shared_ptr<MockExchangeClient>& /*client*/) {
  return nullptr;
}

std::shared_ptr<ExchangeTransportEntry> makeEntry() {
  return ExchangeTransportEntry::make<MockExchangeClient>(
      makeMockClient, buildNoOperator, buildNoOperator);
}

TEST(ExchangeTransportRegistryTest, registryOperations) {
  ExchangeTransportRegistry::unregisterAll();

  const int32_t numTransports = 5;
  for (int32_t i = 0; i < numTransports; i++) {
    ExchangeTransportRegistry::global().insert(
        fmt::format("transport-{}", i), makeEntry());
  }

  for (int32_t i = 0; i < numTransports; i++) {
    EXPECT_NE(
        ExchangeTransportRegistry::tryGet(fmt::format("transport-{}", i)),
        nullptr);
  }
  EXPECT_EQ(ExchangeTransportRegistry::tryGet("nonexistent"), nullptr);

  // getAll() also lists the always-available built-in in-memory default.
  EXPECT_THAT(ExchangeTransportRegistry::getAll(), SizeIs(numTransports + 1));

  ExchangeTransportRegistry::unregisterAll();
  EXPECT_THAT(
      ExchangeTransportRegistry::getAll(),
      UnorderedElementsAre(Key(std::string(core::TransportKind::kInMemory))));
}

TEST(ExchangeTransportRegistryTest, defaultTransportResolves) {
  // The built-in in-memory transport is seeded into the registry and pairs an
  // InMemoryExchangeClient factory with both exchange operator builders.
  auto defaultEntry = ExchangeTransportRegistry::tryGet(
      std::string(core::TransportKind::kInMemory));
  ASSERT_NE(defaultEntry, nullptr);
  EXPECT_TRUE(defaultEntry->makeClient != nullptr);
  EXPECT_TRUE(defaultEntry->makeExchangeOperator != nullptr);
  EXPECT_TRUE(defaultEntry->makeMergeExchangeOperator != nullptr);
}

TEST(ExchangeTransportRegistryTest, entryMakeRejectsNullHalves) {
  VELOX_ASSERT_THROW(
      ExchangeTransportEntry::make<MockExchangeClient>(
          nullptr, buildNoOperator),
      "Exchange transport client factory is null");
  VELOX_ASSERT_THROW(
      ExchangeTransportEntry::make<MockExchangeClient>(makeMockClient, nullptr),
      "Exchange transport operator builder is null");

  // A transport that cannot merge leaves the merge builder null; Task fails
  // fast when a MergeExchangeNode names it.
  auto entry = ExchangeTransportEntry::make<MockExchangeClient>(
      makeMockClient, buildNoOperator);
  ASSERT_NE(entry, nullptr);
  EXPECT_TRUE(entry->makeMergeExchangeOperator == nullptr);
}

TEST(ExchangeTransportRegistryTest, operatorBuilderChecksClientType) {
  // make<TClient>() binds the operator builder to the client type the
  // transport's own factory produces, so a client from another transport is
  // rejected rather than silently reinterpreted.
  bool built{false};
  auto entry = ExchangeTransportEntry::make<MockExchangeClient>(
      makeMockClient,
      [&built](
          int32_t,
          DriverCtx*,
          const std::shared_ptr<const core::ExchangeNode>&,
          const std::shared_ptr<MockExchangeClient>& client)
          -> std::unique_ptr<Operator> {
        EXPECT_NE(client, nullptr);
        built = true;
        return nullptr;
      });

  const core::QueryConfig queryConfig{
      std::unordered_map<std::string, std::string>{}};
  auto client = entry->makeClient(
      ExchangeClientContext{
          .taskId = "task",
          .destination = 0,
          .numberOfConsumers = 1,
          .maxExchangeBufferSize = 1 << 20,
          .minExchangeOutputBatchBytes = 0,
          .pool = nullptr,
          .executor = nullptr,
          .queryConfig = queryConfig});
  ASSERT_NE(client, nullptr);
  EXPECT_TRUE(
      entry->makeExchangeOperator(0, nullptr, nullptr, client) == nullptr);
  EXPECT_TRUE(built);

  VELOX_ASSERT_THROW(
      entry->makeExchangeOperator(
          0, nullptr, nullptr, std::make_shared<UnrelatedExchangeClient>()),
      "Exchange client was not created by this transport's client factory");
}

class ExchangeTransportRegistryFixture : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    ExchangeTransportRegistry::unregisterAll();
  }

  void TearDown() override {
    ExchangeTransportRegistry::unregisterAll();
  }

  std::shared_ptr<core::QueryCtx> queryCtxWithRegistry(
      std::shared_ptr<ExchangeTransportRegistry::Registry> registry) {
    auto queryCtx = core::QueryCtx::create();
    queryCtx->setRegistry(
        ExchangeTransportRegistry::kRegistryKey, std::move(registry));
    return queryCtx;
  }
};

TEST_F(ExchangeTransportRegistryFixture, queryScopedResolution) {
  auto globalEntry = makeEntry();
  ExchangeTransportRegistry::global().insert("shared", globalEntry);
  ExchangeTransportRegistry::global().insert("global-only", globalEntry);

  EXPECT_EQ(
      ExchangeTransportRegistry::tryGet(*core::QueryCtx::create(), "shared"),
      globalEntry);

  auto queryEntry = makeEntry();
  auto queryRegistry =
      ExchangeTransportRegistry::create(&ExchangeTransportRegistry::global());
  queryRegistry->insert("shared", queryEntry);
  auto queryCtx = queryCtxWithRegistry(queryRegistry);

  EXPECT_EQ(ExchangeTransportRegistry::tryGet(*queryCtx, "shared"), queryEntry);
  EXPECT_EQ(
      ExchangeTransportRegistry::tryGet(*queryCtx, "global-only"), globalEntry);
  EXPECT_EQ(ExchangeTransportRegistry::tryGet("shared"), globalEntry);
}

TEST_F(ExchangeTransportRegistryFixture, queryScopedUnregisterAll) {
  auto globalEntry = makeEntry();
  ExchangeTransportRegistry::global().insert("transport", globalEntry);

  auto queryRegistry =
      ExchangeTransportRegistry::create(&ExchangeTransportRegistry::global());
  queryRegistry->insert("transport", makeEntry());
  auto queryCtx = queryCtxWithRegistry(queryRegistry);

  ExchangeTransportRegistry::unregisterAll(*queryCtx);

  EXPECT_EQ(
      ExchangeTransportRegistry::tryGet(*queryCtx, "transport"), globalEntry);
  EXPECT_EQ(ExchangeTransportRegistry::tryGet("transport"), globalEntry);
}

TEST_F(ExchangeTransportRegistryFixture, queryScopedGetAll) {
  ExchangeTransportRegistry::global().insert("global-only", makeEntry());
  ExchangeTransportRegistry::global().insert("shared", makeEntry());

  auto queryRegistry =
      ExchangeTransportRegistry::create(&ExchangeTransportRegistry::global());
  queryRegistry->insert("query-only", makeEntry());
  queryRegistry->insert("shared", makeEntry());
  auto queryCtx = queryCtxWithRegistry(queryRegistry);

  // getAll() also lists the always-available built-in in-memory default.
  const std::string inMemory{core::TransportKind::kInMemory};
  EXPECT_THAT(
      ExchangeTransportRegistry::getAll(*queryCtx),
      UnorderedElementsAre(
          Key("global-only"), Key("query-only"), Key("shared"), Key(inMemory)));
  EXPECT_THAT(
      ExchangeTransportRegistry::getAll(),
      UnorderedElementsAre(Key("global-only"), Key("shared"), Key(inMemory)));
}

TEST_F(ExchangeTransportRegistryFixture, isolatedQueryHasNoDefault) {
  // Isolation mode (create(nullptr)) has no parent fallback, so not even the
  // built-in default is visible; an isolated query must register every
  // transport it uses.
  auto queryCtx =
      queryCtxWithRegistry(ExchangeTransportRegistry::create(nullptr));

  EXPECT_EQ(
      ExchangeTransportRegistry::tryGet(
          *queryCtx, std::string(core::TransportKind::kInMemory)),
      nullptr);
}

} // namespace
} // namespace facebook::velox::exec
