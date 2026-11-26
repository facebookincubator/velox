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
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>
#include <string>

namespace facebook::velox::ucx_exchange {
namespace {

bool isReady(const std::future<void>& future) {
  return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// An intra-node consumer of a producer that was removed early learns the
// reason, as a remote consumer does from the end message.
TEST(IntraNodeTransferRegistryTest, cancelledTaskReportsOutcome) {
  auto registry = IntraNodeTransferRegistry::getInstance();
  const std::string taskId{"registryCancelledWithError"};
  registry->cancelTask(
      taskId, ChannelOutcome::producerError("Injected producer failure"));

  const auto result =
      registry->poll({taskId, /*destination=*/0, /*sequenceNumber=*/0});
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->atEnd);
  EXPECT_EQ(result->data, nullptr);
  EXPECT_EQ(
      result->outcome,
      ChannelOutcome::producerError("Injected producer failure"));
  registry->clearCancelledTask(taskId);
}

// A producer removed after it sent all its output ends its stream cleanly.
TEST(IntraNodeTransferRegistryTest, cancelledFinishedTaskEndsCleanly) {
  auto registry = IntraNodeTransferRegistry::getInstance();
  const std::string taskId{"registryCancelledFinished"};
  registry->cancelTask(taskId, ChannelOutcome::clean());

  const auto result =
      registry->poll({taskId, /*destination=*/0, /*sequenceNumber=*/0});
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->atEnd);
  EXPECT_EQ(result->data, nullptr);
  EXPECT_EQ(result->outcome, ChannelOutcome::clean());
  registry->clearCancelledTask(taskId);
}

// The end a server publishes carries the outcome its entry decided.
TEST(IntraNodeTransferRegistryTest, publishedEndCarriesOutcome) {
  auto registry = IntraNodeTransferRegistry::getInstance();
  const IntraNodeTransferKey key{
      "registryPublishedError", /*destination=*/0, /*sequenceNumber=*/0};
  auto retrieved = registry->publish(
      key,
      nullptr,
      /*numRows=*/0,
      /*atEnd=*/true,
      ChannelOutcome::producerError("Injected producer failure"));
  EXPECT_FALSE(isReady(retrieved));

  const auto result = registry->poll(key);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->atEnd);
  EXPECT_EQ(result->data, nullptr);
  EXPECT_EQ(
      result->outcome,
      ChannelOutcome::producerError("Injected producer failure"));
  EXPECT_TRUE(isReady(retrieved));
  EXPECT_FALSE(registry->poll(key).has_value());
}

// A task is cancelled with the outcome its first removal decided, so that a
// later cancellation cannot turn a lost chunk into a clean end. The first
// cancellation drops the chunk that is published and not read, and releases
// its server.
TEST(IntraNodeTransferRegistryTest, firstCancellationWins) {
  auto registry = IntraNodeTransferRegistry::getInstance();
  const std::string taskId{"registryCancelledTwice"};
  const IntraNodeTransferKey key{
      taskId, /*destination=*/0, /*sequenceNumber=*/0};
  auto chunk = std::make_shared<cudf::packed_columns>();
  std::weak_ptr<cudf::packed_columns> weakChunk = chunk;
  auto retrieved = registry->publish(
      key,
      std::move(chunk),
      /*numRows=*/10,
      /*atEnd=*/false,
      ChannelOutcome::clean());

  registry->cancelTask(
      taskId, ChannelOutcome::producerError("Injected producer failure"));
  EXPECT_TRUE(isReady(retrieved));
  EXPECT_TRUE(weakChunk.expired());
  registry->cancelTask(taskId, ChannelOutcome::clean());

  const auto result = registry->poll(key);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->atEnd);
  EXPECT_EQ(result->data, nullptr);
  EXPECT_EQ(
      result->outcome,
      ChannelOutcome::producerError("Injected producer failure"));

  // A server that publishes after the cancellation does not wait.
  auto late = registry->publish(
      {taskId, /*destination=*/0, /*sequenceNumber=*/1},
      std::make_shared<cudf::packed_columns>(),
      /*numRows=*/10,
      /*atEnd=*/false,
      ChannelOutcome::clean());
  EXPECT_TRUE(isReady(late));
  registry->clearCancelledTask(taskId);
}

// A server whose destination is Released drops the chunk it published, and
// stops waiting for its source. The chunk is freed, and other destinations of
// the task keep theirs.
TEST(IntraNodeTransferRegistryTest, cancelTransferFulfilsPromise) {
  auto registry = IntraNodeTransferRegistry::getInstance();
  const std::string taskId{"registryCancelledTransfer"};
  const IntraNodeTransferKey key{
      taskId, /*destination=*/0, /*sequenceNumber=*/0};
  const IntraNodeTransferKey otherKey{
      taskId, /*destination=*/1, /*sequenceNumber=*/0};
  auto chunk = std::make_shared<cudf::packed_columns>();
  std::weak_ptr<cudf::packed_columns> weakChunk = chunk;
  auto retrieved = registry->publish(
      key,
      std::move(chunk),
      /*numRows=*/10,
      /*atEnd=*/false,
      ChannelOutcome::clean());
  auto otherRetrieved = registry->publish(
      otherKey,
      std::make_shared<cudf::packed_columns>(),
      /*numRows=*/20,
      /*atEnd=*/false,
      ChannelOutcome::clean());

  registry->cancelTransfer(key);
  EXPECT_TRUE(isReady(retrieved));
  EXPECT_TRUE(weakChunk.expired());
  EXPECT_FALSE(registry->poll(key).has_value());
  // A second cancellation of the same transfer does nothing.
  registry->cancelTransfer(key);

  EXPECT_FALSE(isReady(otherRetrieved));
  const auto other = registry->poll(otherKey);
  ASSERT_TRUE(other.has_value());
  EXPECT_FALSE(other->atEnd);
  EXPECT_NE(other->data, nullptr);
  EXPECT_EQ(other->numRows, 20);
  EXPECT_TRUE(isReady(otherRetrieved));
}

} // namespace
} // namespace facebook::velox::ucx_exchange
