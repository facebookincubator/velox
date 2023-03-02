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
#include "folly/experimental/EventCount.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/OrderBy.h"
#include "velox/exec/PartitionedOutputBufferManager.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Task.h"
#include "velox/exec/Values.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/Cursor.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::memory;

namespace facebook::velox::exec::test {
namespace {
static constexpr int64_t kCapacity = 512 << 20;
}
class SharedMemoryArbitrationTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    IMemoryManager::Options options;
    options.capacity = kCapacity;
    options.arbitratorKind = MemoryArbitrator::Kind::kShared;
    memoryManager_ = std::make_unique<MemoryManager>(options);
  }

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  std::shared_ptr<core::QueryCtx> newQueryCtx(int64_t memCapacity) {
    std::unordered_map<std::string, std::shared_ptr<Config>> configs;
    std::shared_ptr<MemoryPool> pool =
        memoryManager_->getPool(memCapacity, MemoryReclaimer::create());
    auto queryCtx = std::make_shared<core::QueryCtx>(
        driverExecutor_.get(),
        std::make_shared<core::MemConfig>(),
        configs,
        nullptr,
        std::move(pool));
    return queryCtx;
  }

  std::unique_ptr<MemoryManager> memoryManager_;
};

TEST_F(SharedMemoryArbitrationTest, crossQueries) {
  const int numBatches = 128;
  const int numRows = 1000;
  const int stringSize = 1024;
  std::vector<RowVectorPtr> batches;
  for (int i = 0; i < numBatches; ++i) {
    batches.push_back(makeRowVector(
        {makeFlatVector<int64_t>(numRows, [](auto row) { return row * 3; }),
         makeFlatVector<StringView>(numRows, [](auto row) {
           return StringView(std::string(stringSize, 'a' + row % 26));
         })}));
  }
  createDuckDbTable(batches);
  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  auto queryCtx = newQueryCtx(kCapacity / 2);

  folly::EventCount largeAllocationWait;
  auto largeAllocationWaitKey = largeAllocationWait.prepareWait();
  folly::EventCount orderByWait;
  auto orderByWaitKey = orderByWait.prepareWait();

  std::atomic<bool> injectValuesOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Values::getOutput",
      std::function<void(Values*)>(([&](Values* values) {
        if (!injectValuesOnce.exchange(false)) {
          return;
        }
        largeAllocationWait.wait(largeAllocationWaitKey);
        const int64_t allocationSize = kCapacity * 6 / 7;
        auto buffer = values->pool()->allocate(allocationSize);
        values->pool()->free(buffer, allocationSize);
        orderByWait.notify();
      })));

  std::atomic<bool> injectOrderByOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::OrderBy::addInput",
      std::function<void(OrderBy*)>(([&](const OrderBy* orderBy) {
        if (orderBy->pool()->currentBytes() < kCapacity / 6) {
          return;
        }
        if (!injectOrderByOnce.exchange(false)) {
          return;
        }
        largeAllocationWait.notify();
        orderByWait.wait(orderByWaitKey);
      })));

  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Task::requestPauseLocked",
      std::function<void(Task*)>(([&](Task* task) {
        orderByWait.notify();
      })));

  std::thread orderByThread([&]() {
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->path)
            .config(core::QueryConfig::kSpillEnabled, "true")
            .config(core::QueryConfig::kOrderBySpillEnabled, "true")
            .queryCtx(queryCtx)
            .plan(PlanBuilder()
                      .values(batches)
                      .orderBy({fmt::format("{} ASC NULLS LAST", "c0")}, false)
                      .planNode())
            .assertResults("SELECT * FROM tmp ORDER BY c0 ASC NULLS LAST");
  });

  std::thread valuesThread([&]() {
    auto task = AssertQueryBuilder(duckDbQueryRunner_)
                    .queryCtx(queryCtx)
                    .plan(PlanBuilder().values(batches).planNode())
                    .assertResults("SELECT * FROM tmp");
  });

  orderByThread.join();
  valuesThread.join();
}
} // namespace facebook::velox::exec::test
