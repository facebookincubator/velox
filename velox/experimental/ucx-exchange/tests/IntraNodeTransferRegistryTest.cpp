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

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <chrono>

#include "velox/common/testutil/TestValue.h"

namespace facebook::velox::ucx_exchange {
namespace {

using namespace std::chrono_literals;

TEST(IntraNodeTransferRegistryTest, cancelPublishedTransfer) {
  auto registry = IntraNodeTransferRegistry::getInstance();
  const IntraNodeTransferKey key{"cancel-published", 3, 7};
  registry->clearCancelledTask(key.taskId);

  auto retrieved =
      registry->publish(key, nullptr, /*numRows=*/0, /*atEnd=*/false);
  EXPECT_EQ(retrieved.wait_for(0s), std::future_status::timeout);

  registry->cancelTransfer(key);
  EXPECT_EQ(retrieved.wait_for(0s), std::future_status::ready);
  EXPECT_THROW(retrieved.get(), IntraNodeTransferCancelled);
  auto result = registry->poll(key);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->atEnd);
  EXPECT_EQ(result->data, nullptr);

  registry->clearCancelledTask(key.taskId);
  EXPECT_FALSE(registry->poll(key).has_value());
}

TEST(IntraNodeTransferRegistryTest, cancelWinsPublishRace) {
  auto registry = IntraNodeTransferRegistry::getInstance();
  const IntraNodeTransferKey key{"cancel-before-publish", 5, 11};
  registry->clearCancelledTask(key.taskId);

  registry->cancelTransfer(key);
  auto retrieved =
      registry->publish(key, nullptr, /*numRows=*/0, /*atEnd=*/false);
  EXPECT_EQ(retrieved.wait_for(0s), std::future_status::ready);
  EXPECT_THROW(retrieved.get(), IntraNodeTransferCancelled);

  registry->clearCancelledTask(key.taskId);
  EXPECT_FALSE(registry->poll(key).has_value());
}

#ifndef NDEBUG
TEST(IntraNodeTransferRegistryTest, cancellationDuringPublishStaysTerminal) {
  using common::testutil::TestValue;
  const bool testValuesEnabled = TestValue::enabled();
  TestValue::enable();
  SCOPE_EXIT {
    if (!testValuesEnabled) {
      TestValue::disable();
    }
  };

  auto registry = IntraNodeTransferRegistry::getInstance();
  const IntraNodeTransferKey key{"cancel-during-publish", 2, 4};
  registry->clearCancelledTask(key.taskId);
  SCOPE_EXIT {
    registry->clearCancelledTask(key.taskId);
  };

  std::shared_ptr<IntraNodeTransferEntry> entry;
  SCOPED_TESTVALUE_SET(
      "facebook::velox::ucx_exchange::IntraNodeTransferRegistry::publish",
      std::function<void(std::shared_ptr<IntraNodeTransferEntry>*)>(
          [&](auto* publishedEntry) {
            entry = *publishedEntry;
            registry->cancelTransfer(key);
          }));
  auto retrieved =
      registry->publish(key, nullptr, /*numRows=*/7, /*atEnd=*/false);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(retrieved.wait_for(0s), std::future_status::ready);
  EXPECT_THROW(retrieved.get(), IntraNodeTransferCancelled);
  EXPECT_TRUE(entry->retrievalSignaled);
  EXPECT_TRUE(entry->ready);
  EXPECT_TRUE(entry->atEnd);
  EXPECT_EQ(entry->numRows, 0);
  EXPECT_EQ(entry->data, nullptr);
}
#endif

} // namespace
} // namespace facebook::velox::ucx_exchange
