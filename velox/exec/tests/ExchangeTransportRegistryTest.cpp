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

#include <chrono>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>

#include <folly/ScopeGuard.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/synchronization/Baton.h>
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

// Keep entries immutable because tasks may share them.
static_assert(!std::is_copy_assignable_v<ExchangeTransportEntry>);
static_assert(!std::is_move_assignable_v<ExchangeTransportEntry>);

// Represents a control-plane-only client without an ExchangeSource or executor.
// Each 'Tag' produces an unrelated client type.
template <typename Tag>
class NoOpExchangeClient : public ExchangeClient {
 public:
  void addRemoteTaskId(std::string_view /*remoteTaskId*/) override {}

  void noMoreRemoteTasks() override {}

  void close() override {}

  folly::F14FastMap<std::string, RuntimeMetric> stats() const override {
    return {};
  }

  std::string toString() const override {
    return "no-op";
  }

  folly::dynamic toJson() const override {
    return folly::dynamic::object;
  }
};

using MockExchangeClient = NoOpExchangeClient<struct MockTag>;

using UnrelatedExchangeClient = NoOpExchangeClient<struct UnrelatedTag>;

class BlockingDestructionState {
 public:
  BlockingDestructionState(
      folly::Baton<>& destructionStarted,
      folly::Baton<>& continueDestruction)
      : destructionStarted_{destructionStarted},
        continueDestruction_{continueDestruction} {}

  BlockingDestructionState(const BlockingDestructionState&) = delete;
  BlockingDestructionState& operator=(const BlockingDestructionState&) = delete;
  BlockingDestructionState(BlockingDestructionState&&) = delete;
  BlockingDestructionState& operator=(BlockingDestructionState&&) = delete;

  ~BlockingDestructionState() {
    destructionStarted_.post();
    continueDestruction_.wait();
  }

 private:
  folly::Baton<>& destructionStarted_;
  folly::Baton<>& continueDestruction_;
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

class ExchangeTransportRegistryTest : public testing::Test {
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

TEST_F(ExchangeTransportRegistryTest, registryOperations) {
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

  // Account for the built-in in-memory transport.
  EXPECT_THAT(ExchangeTransportRegistry::getAll(), SizeIs(numTransports + 1));

  ExchangeTransportRegistry::unregisterAll();
  EXPECT_THAT(
      ExchangeTransportRegistry::getAll(),
      UnorderedElementsAre(Key(std::string(core::TransportKind::kInMemory))));
}

TEST_F(ExchangeTransportRegistryTest, defaultTransportResolves) {
  // The built-in entry supports both exchange operator types.
  auto defaultEntry = ExchangeTransportRegistry::tryGet(
      std::string(core::TransportKind::kInMemory));
  ASSERT_NE(defaultEntry, nullptr);
  EXPECT_TRUE(defaultEntry->makeClient != nullptr);
  EXPECT_TRUE(defaultEntry->makeExchangeOperator != nullptr);
  EXPECT_TRUE(defaultEntry->makeMergeExchangeOperator != nullptr);
}

TEST_F(ExchangeTransportRegistryTest, defaultTransportBufferSizeBoundary) {
  // In-memory clients store the buffer size as int64_t.
  auto defaultEntry = ExchangeTransportRegistry::tryGet(
      std::string(core::TransportKind::kInMemory));
  ASSERT_NE(defaultEntry, nullptr);
  const core::QueryConfig queryConfig{
      std::unordered_map<std::string, std::string>{}};
  auto pool = memory::memoryManager()->addLeafPool();
  folly::CPUThreadPoolExecutor executor(1);
  const auto makeContext = [&](uint64_t maxExchangeBufferSize) {
    return ExchangeClientContext{
        .taskId = "task",
        .destination = 0,
        .numberOfConsumers = 1,
        .maxExchangeBufferSize = maxExchangeBufferSize,
        .minExchangeOutputBatchBytes = 0,
        .pool = pool.get(),
        .executor = &executor,
        .queryConfig = queryConfig};
  };
  constexpr auto kMaxBufferSize =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

  auto client = defaultEntry->makeClient(makeContext(kMaxBufferSize));
  ASSERT_NE(client, nullptr);
  client->close();

  VELOX_ASSERT_USER_THROW(
      defaultEntry->makeClient(makeContext(kMaxBufferSize + 1)),
      core::QueryConfig::kMaxExchangeBufferSize);
}

TEST_F(ExchangeTransportRegistryTest, defaultTransportSurvivesReset) {
  const std::string inMemory{core::TransportKind::kInMemory};
  folly::Baton<> destructionStarted;
  folly::Baton<> continueDestruction;
  auto state = std::make_shared<BlockingDestructionState>(
      destructionStarted, continueDestruction);
  auto entry = ExchangeTransportEntry::make<MockExchangeClient>(
      [state](const ExchangeClientContext&) {
        return std::make_shared<MockExchangeClient>();
      },
      buildNoOperator);
  ExchangeTransportRegistry::global().insert("blocking", entry);
  state.reset();
  entry.reset();

  std::thread reset([] { ExchangeTransportRegistry::unregisterAll(); });
  bool destructionReleased{false};
  SCOPE_EXIT {
    if (!destructionReleased) {
      continueDestruction.post();
    }
    reset.join();
  };
  ASSERT_TRUE(destructionStarted.try_wait_for(std::chrono::seconds(5)));
  auto defaultEntry = ExchangeTransportRegistry::tryGet(inMemory);
  destructionReleased = true;
  continueDestruction.post();

  EXPECT_NE(defaultEntry, nullptr);
  EXPECT_THAT(
      ExchangeTransportRegistry::getAll(), UnorderedElementsAre(Key(inMemory)));
}

TEST_F(ExchangeTransportRegistryTest, entryMakeRejectsNullHalves) {
  VELOX_ASSERT_THROW(
      ExchangeTransportEntry::make<MockExchangeClient>(
          nullptr, buildNoOperator),
      "Exchange transport client factory is null");
  VELOX_ASSERT_THROW(
      ExchangeTransportEntry::make<MockExchangeClient>(makeMockClient, nullptr),
      "Exchange transport operator builder is null");

  // A null merge builder denotes an unsupported merge exchange.
  auto entry = ExchangeTransportEntry::make<MockExchangeClient>(
      makeMockClient, buildNoOperator);
  ASSERT_NE(entry, nullptr);
  EXPECT_TRUE(entry->makeMergeExchangeOperator == nullptr);
}

TEST_F(ExchangeTransportRegistryTest, operatorBuilderChecksClientType) {
  // Reject clients created by a different transport before calling the builder.
  bool built{false};
  bool builtMerge{false};
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
      },
      [&builtMerge](
          int32_t,
          DriverCtx*,
          const std::shared_ptr<const core::ExchangeNode>&,
          const std::shared_ptr<MockExchangeClient>& client)
          -> std::unique_ptr<Operator> {
        EXPECT_NE(client, nullptr);
        builtMerge = true;
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

  EXPECT_TRUE(
      entry->makeMergeExchangeOperator(0, nullptr, nullptr, client) == nullptr);
  EXPECT_TRUE(builtMerge);
  VELOX_ASSERT_THROW(
      entry->makeMergeExchangeOperator(
          0, nullptr, nullptr, std::make_shared<UnrelatedExchangeClient>()),
      "Exchange client was not created by this transport's client factory");
}

TEST_F(ExchangeTransportRegistryTest, queryScopedResolution) {
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

TEST_F(ExchangeTransportRegistryTest, queryScopedUnregisterAll) {
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

TEST_F(ExchangeTransportRegistryTest, queryScopedGetAll) {
  ExchangeTransportRegistry::global().insert("global-only", makeEntry());
  ExchangeTransportRegistry::global().insert("shared", makeEntry());

  auto queryRegistry =
      ExchangeTransportRegistry::create(&ExchangeTransportRegistry::global());
  queryRegistry->insert("query-only", makeEntry());
  queryRegistry->insert("shared", makeEntry());
  auto queryCtx = queryCtxWithRegistry(queryRegistry);

  const std::string inMemory{core::TransportKind::kInMemory};
  EXPECT_THAT(
      ExchangeTransportRegistry::getAll(*queryCtx),
      UnorderedElementsAre(
          Key("global-only"), Key("query-only"), Key("shared"), Key(inMemory)));
  EXPECT_THAT(
      ExchangeTransportRegistry::getAll(),
      UnorderedElementsAre(Key("global-only"), Key("shared"), Key(inMemory)));
}

TEST_F(ExchangeTransportRegistryTest, isolatedQueryHasNoDefault) {
  // An isolated registry does not inherit the built-in transport.
  auto queryCtx =
      queryCtxWithRegistry(ExchangeTransportRegistry::create(nullptr));

  EXPECT_EQ(
      ExchangeTransportRegistry::tryGet(
          *queryCtx, std::string(core::TransportKind::kInMemory)),
      nullptr);
}

} // namespace
} // namespace facebook::velox::exec
