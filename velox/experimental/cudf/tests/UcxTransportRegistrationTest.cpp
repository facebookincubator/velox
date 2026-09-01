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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/ucx-exchange/UcxExchange.h"
#include "velox/experimental/ucx-exchange/UcxExchangeClient.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/exec/Driver.h"
#include "velox/exec/ExchangeTransportRegistry.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/VectorStream.h"

#include <gtest/gtest.h>

namespace facebook::velox::exec::test {
namespace {

using core::TransportKind;

/// Verifies that the cuDF registration path advertises the UCX transport in
/// ExchangeTransportRegistry and OutputTransportRegistry, that the registered
/// entries build the UCX operators paired with their own client and manager,
/// and that those operators are left in place by the cuDF driver adapter
/// machinery.
class UcxTransportRegistrationTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    savedExchange_ = cudf_velox::CudfConfig::getInstance().exchange;
  }

  void TearDown() override {
    if (cudf_velox::cudfIsRegistered()) {
      cudf_velox::unregisterCudf();
    }
    cudf_velox::CudfConfig::getInstance().exchange = savedExchange_;
    // Drop the UCX registrations and restore the built-in in-memory defaults so
    // the next test starts from a known state.
    ExchangeTransportRegistry::unregisterAll();
    OutputTransportRegistry::unregisterAll();
    OperatorTestBase::TearDown();
  }

  // Runs the cuDF registration path with UCX exchange configured on or off.
  void registerCudfWithExchange(bool exchangeEnabled) {
    cudf_velox::CudfConfig::getInstance().exchange = exchangeEnabled;
    cudf_velox::registerCudf();
  }

  // Returns the plan fragment for an Exchange node reading over 'transport'.
  core::PlanFragment makeExchangePlan(std::string_view transport) {
    return PlanBuilder()
        .exchange(rowType_,
                  VectorSerde::kindName(VectorSerde::Kind::kPresto),
                  std::string{transport})
        .planFragment();
  }

  // Returns the plan fragment for a PartitionedOutput node writing over
  // 'transport'.
  core::PlanFragment makePartitionedOutputPlan(std::string_view transport) {
    auto vectors = makeRowVector(rowType_, 1);
    return PlanBuilder()
        .values({vectors})
        .partitionedOutput({"c0"},
                           4,
                           /*outputLayout=*/{},
                           /*serdeKind=*/"Presto",
                           std::string{transport})
        .planFragment();
  }

  std::shared_ptr<Task> makeTask(const std::string& taskId,
                                 core::PlanFragment fragment) {
    return Task::create(taskId,
                        std::move(fragment),
                        0,
                        core::QueryCtx::create(),
                        Task::ExecutionMode::kParallel);
  }

  // Returns a DriverCtx for driver 0 of 'task'.
  std::shared_ptr<DriverCtx> makeDriverCtx(std::shared_ptr<Task> task) {
    return std::make_shared<DriverCtx>(
        std::move(task), 0, 0, kUngroupedGroupId, 0);
  }

  RowTypePtr rowType_{ROW({"c0", "c1"}, {BIGINT(), VARCHAR()})};

 private:
  bool savedExchange_{false};
};

TEST_F(UcxTransportRegistrationTest, exchangeDisabledRegistersNoUcxTransport) {
  registerCudfWithExchange(false);

  const std::string ucx{TransportKind::kUcx};
  EXPECT_EQ(ExchangeTransportRegistry::tryGet(ucx), nullptr);
  EXPECT_EQ(OutputTransportRegistry::tryGet(ucx), nullptr);
}

TEST_F(UcxTransportRegistrationTest, exchangeEnabledRegistersUcxTransport) {
  registerCudfWithExchange(true);

  const std::string ucx{TransportKind::kUcx};
  auto exchangeEntry = ExchangeTransportRegistry::tryGet(ucx);
  ASSERT_NE(exchangeEntry, nullptr);
  EXPECT_TRUE(static_cast<bool>(exchangeEntry->makeClient));
  EXPECT_TRUE(static_cast<bool>(exchangeEntry->makeExchangeOperator));
  // UCX supports merge exchange by receiving and then sorting on the GPU.
  EXPECT_TRUE(static_cast<bool>(exchangeEntry->makeMergeExchangeOperator));

  auto outputEntry = OutputTransportRegistry::tryGet(ucx);
  ASSERT_NE(outputEntry, nullptr);
  EXPECT_TRUE(static_cast<bool>(outputEntry->makeOutputOperator));
  EXPECT_NE(std::dynamic_pointer_cast<ucx_exchange::UcxOutputQueueManager>(
                outputEntry->manager),
            nullptr);
}

TEST_F(UcxTransportRegistrationTest, inMemoryTransportIsUnaffected) {
  registerCudfWithExchange(true);

  const std::string inMemory{TransportKind::kInMemory};
  auto exchangeEntry = ExchangeTransportRegistry::tryGet(inMemory);
  ASSERT_NE(exchangeEntry, nullptr);
  // The stock in-memory transport still supports merge exchange.
  EXPECT_TRUE(static_cast<bool>(exchangeEntry->makeMergeExchangeOperator));

  auto outputEntry = OutputTransportRegistry::tryGet(inMemory);
  ASSERT_NE(outputEntry, nullptr);
  EXPECT_NE(std::dynamic_pointer_cast<DefaultOutputBufferManager>(
                outputEntry->manager),
            nullptr);
}

TEST_F(UcxTransportRegistrationTest, ucxEntryBuildsUcxExchange) {
  registerCudfWithExchange(true);

  auto plan = makeExchangePlan(TransportKind::kUcx);
  auto exchangeNode =
      std::dynamic_pointer_cast<const core::ExchangeNode>(plan.planNode);
  ASSERT_NE(exchangeNode, nullptr);
  auto task = makeTask("test-ucx-exchange-task", std::move(plan));
  auto driverCtx = makeDriverCtx(task);

  auto entry = ExchangeTransportRegistry::tryGet(*task->queryCtx(),
                                                 exchangeNode->transportKind());
  ASSERT_NE(entry, nullptr);

  auto client = entry->makeClient(ExchangeClientContext{
      .taskId = task->taskId(),
      .destination = task->destination(),
      .numberOfConsumers = 1,
      .maxExchangeBufferSize = 1 << 20,
      .minExchangeOutputBatchBytes = 0,
      .pool = pool(),
      .executor = executor_.get(),
      .queryConfig = task->queryCtx()->queryConfig()});
  ASSERT_NE(std::dynamic_pointer_cast<ucx_exchange::UcxExchangeClient>(client),
            nullptr);

  auto exchangeOperator =
      entry->makeExchangeOperator(0, driverCtx.get(), exchangeNode, client);
  auto* ucxExchange =
      dynamic_cast<ucx_exchange::UcxExchange*>(exchangeOperator.get());
  ASSERT_NE(ucxExchange, nullptr);

  // The cuDF driver adapter must leave the registry-built operator in place and
  // treat it as a GPU source, so that no conversion operator is spliced around
  // it and it is not reported as a failed replacement.
  auto& registry = cudf_velox::OperatorAdapterRegistry::getInstance();
  const auto* adapter = registry.findAdapter(exchangeOperator.get());
  ASSERT_NE(adapter, nullptr);
  EXPECT_TRUE(adapter->keepOperator());
  const auto properties = adapter->properties(
      exchangeOperator.get(), exchangeNode, driverCtx.get());
  EXPECT_TRUE(properties.canRunOnGPU);
  EXPECT_TRUE(properties.producesGpuOutput);
  EXPECT_FALSE(properties.acceptsGpuInput);

  exchangeOperator->close();
  client->close();
}

TEST_F(UcxTransportRegistrationTest, ucxEntryBuildsUcxPartitionedOutput) {
  registerCudfWithExchange(true);

  auto plan = makePartitionedOutputPlan(TransportKind::kUcx);
  auto outputNode =
      std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
          plan.planNode);
  ASSERT_NE(outputNode, nullptr);
  auto task = makeTask("test-ucx-partitioned-output-task", std::move(plan));
  auto driverCtx = makeDriverCtx(task);

  auto entry = OutputTransportRegistry::tryGet(*task->queryCtx(),
                                               outputNode->transportKind());
  ASSERT_NE(entry, nullptr);

  auto outputOperator = entry->makeOutputOperator(
      0, driverCtx.get(), outputNode, /*eagerFlush=*/false);
  auto* ucxOutput =
      dynamic_cast<ucx_exchange::UcxPartitionedOutput*>(outputOperator.get());
  ASSERT_NE(ucxOutput, nullptr);

  auto& registry = cudf_velox::OperatorAdapterRegistry::getInstance();
  const auto* adapter = registry.findAdapter(outputOperator.get());
  ASSERT_NE(adapter, nullptr);
  EXPECT_TRUE(adapter->keepOperator());
  const auto properties =
      adapter->properties(outputOperator.get(), outputNode, driverCtx.get());
  EXPECT_TRUE(properties.canRunOnGPU);
  EXPECT_TRUE(properties.acceptsGpuInput);
  EXPECT_FALSE(properties.producesGpuOutput);

  outputOperator->close();
}

TEST_F(UcxTransportRegistrationTest, ucxEntryRejectsForeignExchangeClient) {
  registerCudfWithExchange(true);

  auto plan = makeExchangePlan(TransportKind::kUcx);
  auto exchangeNode =
      std::dynamic_pointer_cast<const core::ExchangeNode>(plan.planNode);
  ASSERT_NE(exchangeNode, nullptr);
  auto task = makeTask("test-foreign-client-task", std::move(plan));
  auto driverCtx = makeDriverCtx(task);

  auto ucxEntry =
      ExchangeTransportRegistry::tryGet(std::string{TransportKind::kUcx});
  ASSERT_NE(ucxEntry, nullptr);
  auto inMemoryEntry =
      ExchangeTransportRegistry::tryGet(std::string{TransportKind::kInMemory});
  ASSERT_NE(inMemoryEntry, nullptr);

  // A client from another transport must not be accepted: the entry pairs the
  // operator with the client its own factory produces.
  auto foreignClient = inMemoryEntry->makeClient(ExchangeClientContext{
      .taskId = task->taskId(),
      .destination = task->destination(),
      .numberOfConsumers = 1,
      .maxExchangeBufferSize = 1 << 20,
      .minExchangeOutputBatchBytes = 0,
      .pool = pool(),
      .executor = executor_.get(),
      .queryConfig = task->queryCtx()->queryConfig()});
  ASSERT_NE(foreignClient, nullptr);
  VELOX_ASSERT_THROW(
      ucxEntry->makeExchangeOperator(
          0, driverCtx.get(), exchangeNode, foreignClient),
      "Exchange client was not created by this transport's client factory");

  foreignClient->close();
}

} // namespace
} // namespace facebook::velox::exec::test
