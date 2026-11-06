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
#pragma once

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

#include "velox/common/memory/MemoryPool.h"
#include "velox/exec/Split.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestData.h"

namespace facebook::velox::ucx_exchange {

/// Owns the process-wide UCX state every ucx_exchange_test case runs against:
/// the memory manager, the UcxOutputQueueManager and the Communicator with its
/// progress thread. It is registered once per binary, in
/// UcxExchangeTestBase.cpp, so the mock-driver and the Task-level cases share
/// one Communicator.
class UcxExchangeTestEnvironment : public testing::Environment {
 public:
  void SetUp() override;

  void TearDown() override;

  /// Returns the singleton output queue manager.
  static const std::shared_ptr<UcxOutputQueueManager>& queueManager() {
    return queueManager_;
  }

  /// Returns a task id prefix that no other case in this process uses.
  static std::string nextTaskPrefix() {
    return "t" + std::to_string(testCounter_.fetch_add(1)) + "_";
  }

  /// Returns the port a split URL advertises for the round trip to land on the
  /// Communicator.
  static int splitUrlPort() {
    return communicatorPort_ - kSplitUrlPortOffset;
  }

 private:
  static void initializeExchangeState();

  static void destroyExchangeState();

  // UcxExchangeSource computes the UCX port as the split URL's port + 3
  // (UcxExchangeSource.cpp), so remoteSplit() advertises this much below the
  // communicator's port for the round trip to land back on it.
  static constexpr int kSplitUrlPortOffset = 3;

  static constexpr auto kUnusedCoordinatorUrl =
      std::string_view("http://localhost:12345/bla");

  // Chosen once per process rather than hardcoded. The communicator opens a
  // listener on this port without address reuse, so two runs of this binary in
  // quick succession fail the second with "bind(0.0.0.0:21346) failed: Address
  // already in use" while the first port is still in TIME_WAIT.
  static uint16_t communicatorPort_;

  static std::shared_ptr<UcxOutputQueueManager> queueManager_;
  static std::shared_ptr<std::thread> communicatorThread_;
  static std::shared_ptr<Communicator> communicator_;
  static std::atomic<uint32_t> testCounter_;
};

/// Fixture shared by the cases in UcxExchangeTest.cpp and
/// UcxExchangeTaskTest.cpp.
class UcxExchangeTestBase : public testing::Test {
 protected:
  // Generate a unique task ID prefix for this test run to avoid collisions
  // between parametrized tests
  std::string getUniqueTaskPrefix() {
    return UcxExchangeTestEnvironment::nextTaskPrefix();
  }

  const std::shared_ptr<UcxOutputQueueManager>& queueManager() {
    return UcxExchangeTestEnvironment::queueManager();
  }

  // Get the row type based on the table type.
  facebook::velox::RowTypePtr getRowType(TableType tableType) {
    if (tableType == TableType::WIDE) {
      return WideTestTable::kRowType;
    }
    return UcxTestData::kTestRowType;
  }

  void SetUp() override;

  // Returns a split that makes the UcxExchange fetch partition 'partitionId' of
  // task 'taskId' over this process's Communicator.
  exec::Split remoteSplit(std::string_view taskId, int partitionId);

  std::shared_ptr<facebook::velox::memory::MemoryPool> pool_;
};

} // namespace facebook::velox::ucx_exchange
