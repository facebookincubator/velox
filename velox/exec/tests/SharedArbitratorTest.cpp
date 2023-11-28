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

#include <gtest/gtest.h>

#include <folly/Singleton.h>
#include <re2/re2.h>
#include <deque>

#include <functional>
#include <optional>
#include "folly/experimental/EventCount.h"
#include "folly/futures/Barrier.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/exec/Driver.h"
#include "velox/exec/HashBuild.h"
#include "velox/exec/HashJoinBridge.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/SharedArbitrator.h"
#include "velox/exec/TableWriter.h"
#include "velox/exec/Values.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

DECLARE_bool(velox_memory_leak_check_enabled);
DECLARE_bool(velox_suppress_memory_capacity_exceeding_error_message);

using namespace ::testing;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace facebook::velox::exec::test {
constexpr int64_t KB = 1024L;
constexpr int64_t MB = 1024L * KB;

constexpr uint64_t kMemoryCapacity = 512 * MB;
constexpr uint64_t kMemoryPoolInitCapacity = 16 * MB;
constexpr uint64_t kMemoryPoolTransferCapacity = 8 * MB;

struct TestAllocation {
  MemoryPool* pool{nullptr};
  void* buffer{nullptr};
  size_t size{0};

  size_t free() {
    const size_t freedBytes = size;
    if (pool == nullptr) {
      VELOX_CHECK_EQ(freedBytes, 0);
      return freedBytes;
    }
    VELOX_CHECK_GT(freedBytes, 0);
    pool->free(buffer, freedBytes);
    pool = nullptr;
    buffer = nullptr;
    size = 0;
    return freedBytes;
  }
};

// Custom node for the custom factory.
class FakeMemoryNode : public core::PlanNode {
 public:
  FakeMemoryNode(const core::PlanNodeId& id, core::PlanNodePtr input)
      : PlanNode(id), sources_{input} {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<std::shared_ptr<const PlanNode>>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "FakeMemoryNode";
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  std::vector<core::PlanNodePtr> sources_;
};

using AllocationCallback = std::function<TestAllocation(Operator* op)>;
// If return true, the caller will terminate execution and return early.
using ReclaimInjectionCallback = std::function<
    bool(MemoryPool* pool, uint64_t targetByte, MemoryReclaimer::Stats& stats)>;

// Custom operator for the custom factory.
class FakeMemoryOperator : public Operator {
 public:
  FakeMemoryOperator(
      DriverCtx* ctx,
      int32_t id,
      core::PlanNodePtr node,
      bool canReclaim,
      AllocationCallback allocationCb,
      ReclaimInjectionCallback reclaimCb)
      : Operator(ctx, node->outputType(), id, node->id(), "FakeMemoryNode"),
        canReclaim_(canReclaim),
        allocationCb_(std::move(allocationCb)),
        reclaimCb_(std::move(reclaimCb)) {}

  ~FakeMemoryOperator() override {
    clear();
  }

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
    if (allocationCb_ != nullptr) {
      TestAllocation allocation = allocationCb_(this);
      if (allocation.buffer != nullptr) {
        allocations_.push_back(allocation);
      }
      totalBytes_ += allocation.size;
    }
  }

  void noMoreInput() override {
    clear();
    Operator::noMoreInput();
  }

  RowVectorPtr getOutput() override {
    return std::move(input_);
  }

  BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr && allocations_.empty();
  }

  void close() override {
    clear();
    Operator::close();
  }

  bool canReclaim() const override {
    return canReclaim_;
  }

  void reclaim(uint64_t targetBytes, memory::MemoryReclaimer::Stats& stats)
      override {
    VELOX_CHECK(canReclaim());
    auto* driver = operatorCtx_->driver();
    VELOX_CHECK(!driver->state().isOnThread() || driver->state().isSuspended);
    VELOX_CHECK(driver->task()->pauseRequested());
    VELOX_CHECK_GT(targetBytes, 0);

    if (reclaimCb_ != nullptr && reclaimCb_(pool(), targetBytes, stats)) {
      return;
    }

    uint64_t bytesReclaimed{0};
    auto allocIt = allocations_.begin();
    while (allocIt != allocations_.end() &&
           ((targetBytes != 0) && (bytesReclaimed < targetBytes))) {
      bytesReclaimed += allocIt->size;
      totalBytes_ -= allocIt->size;
      pool()->free(allocIt->buffer, allocIt->size);
      allocIt = allocations_.erase(allocIt);
    }
    VELOX_CHECK_GE(totalBytes_, 0);
  }

 private:
  void clear() {
    for (auto& allocation : allocations_) {
      totalBytes_ -= allocation.free();
      VELOX_CHECK_GE(totalBytes_, 0);
    }
    allocations_.clear();
    VELOX_CHECK_EQ(totalBytes_, 0);
  }

  const bool canReclaim_;
  const AllocationCallback allocationCb_;
  const ReclaimInjectionCallback reclaimCb_{nullptr};

  std::atomic<size_t> totalBytes_{0};
  std::vector<TestAllocation> allocations_;
};

// Custom factory that creates fake memory operator.
class FakeMemoryOperatorFactory : public Operator::PlanNodeTranslator {
 public:
  FakeMemoryOperatorFactory() = default;

  std::unique_ptr<Operator> toOperator(
      DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override {
    if (std::dynamic_pointer_cast<const FakeMemoryNode>(node)) {
      return std::make_unique<FakeMemoryOperator>(
          ctx, id, node, canReclaim_, allocationCallback_, reclaimCallback_);
    }
    return nullptr;
  }

  std::optional<uint32_t> maxDrivers(const core::PlanNodePtr& node) override {
    if (std::dynamic_pointer_cast<const FakeMemoryNode>(node)) {
      return maxDrivers_;
    }
    return std::nullopt;
  }

  void setMaxDrivers(uint32_t maxDrivers) {
    maxDrivers_ = maxDrivers;
  }

  void setCanReclaim(bool canReclaim) {
    canReclaim_ = canReclaim;
  }

  void setAllocationCallback(AllocationCallback allocCb) {
    allocationCallback_ = std::move(allocCb);
  }

  void setReclaimCallback(ReclaimInjectionCallback reclaimCb) {
    reclaimCallback_ = std::move(reclaimCb);
  }

 private:
  bool canReclaim_{true};
  AllocationCallback allocationCallback_{nullptr};
  ReclaimInjectionCallback reclaimCallback_{nullptr};
  uint32_t maxDrivers_{1};
};

class FakeMemoryReclaimer : public exec::MemoryReclaimer {
 public:
  FakeMemoryReclaimer() = default;

  static std::unique_ptr<MemoryReclaimer> create() {
    return std::make_unique<FakeMemoryReclaimer>();
  }

  void enterArbitration() override {
    auto* driverThreadCtx = driverThreadContext();
    if (driverThreadCtx == nullptr) {
      return;
    }
    auto* driver = driverThreadCtx->driverCtx.driver;
    ASSERT_TRUE(driver != nullptr);
    if (driver->task()->enterSuspended(driver->state()) != StopReason::kNone) {
      VELOX_FAIL("Terminate detected when entering suspension");
    }
  }

  void leaveArbitration() noexcept override {
    auto* driverThreadCtx = driverThreadContext();
    if (driverThreadCtx == nullptr) {
      return;
    }
    auto* driver = driverThreadCtx->driverCtx.driver;
    ASSERT_TRUE(driver != nullptr);
    driver->task()->leaveSuspended(driver->state());
  }
};

class SharedArbitrationTest : public exec::test::HiveConnectorTestBase {
 protected:
  static void SetUpTestCase() {
    exec::test::HiveConnectorTestBase::SetUpTestCase();
    auto fakeOperatorFactory = std::make_unique<FakeMemoryOperatorFactory>();
    fakeOperatorFactory_ = fakeOperatorFactory.get();
    Operator::registerOperator(std::move(fakeOperatorFactory));
  }

  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    fakeOperatorFactory_->setCanReclaim(true);

    setupMemory();

    rowType_ = ROW(
        {{"c0", INTEGER()},
         {"c1", INTEGER()},
         {"c2", VARCHAR()},
         {"c3", VARCHAR()}});
    fuzzerOpts_.vectorSize = 1024;
    fuzzerOpts_.nullRatio = 0;
    fuzzerOpts_.stringVariableLength = false;
    fuzzerOpts_.stringLength = 1024;
    fuzzerOpts_.allowLazyVector = false;
    VectorFuzzer fuzzer(fuzzerOpts_, pool());
    vector_ = newVector();
    executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(32);
    numAddedPools_ = 0;
  }

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  void setupMemory(
      int64_t memoryCapacity = 0,
      uint64_t memoryPoolInitCapacity = kMemoryPoolInitCapacity,
      uint64_t memoryPoolTransferCapacity = kMemoryPoolTransferCapacity) {
    memoryCapacity = (memoryCapacity != 0) ? memoryCapacity : kMemoryCapacity;
    allocator_ = std::make_shared<MallocAllocator>(memoryCapacity);
    MemoryManagerOptions options;
    options.allocator = allocator_.get();
    options.capacity = allocator_->capacity();
    options.arbitratorKind = "SHARED";
    options.capacity = options.capacity;
    options.memoryPoolInitCapacity = memoryPoolInitCapacity;
    options.memoryPoolTransferCapacity = memoryPoolTransferCapacity;
    options.checkUsageLeak = true;
    options.arbitrationStateCheckCb = memoryArbitrationStateCheck;
    memoryManager_ = std::make_unique<MemoryManager>(options);
    ASSERT_EQ(memoryManager_->arbitrator()->kind(), "SHARED");
    arbitrator_ = static_cast<SharedArbitrator*>(memoryManager_->arbitrator());
    numAddedPools_ = 0;
  }

  RowVectorPtr newVector() {
    VectorFuzzer fuzzer(fuzzerOpts_, pool());
    return fuzzer.fuzzRow(rowType_);
  }

  std::vector<RowVectorPtr> newVectors(size_t vectorSize, size_t expectedSize) {
    VectorFuzzer::Options fuzzerOpts;
    fuzzerOpts.vectorSize = vectorSize;
    fuzzerOpts.stringVariableLength = false;
    fuzzerOpts.stringLength = 1024;
    fuzzerOpts.allowLazyVector = false;
    VectorFuzzer fuzzer(fuzzerOpts_, pool());
    uint64_t totalSize{0};
    std::vector<RowVectorPtr> vectors;
    while (totalSize < expectedSize) {
      vectors.push_back(fuzzer.fuzzInputRow(rowType_));
      totalSize += vectors.back()->estimateFlatSize();
    }
    return vectors;
  }

  std::shared_ptr<core::QueryCtx> newQueryCtx(
      int64_t memoryCapacity = kMaxMemory,
      std::unique_ptr<MemoryReclaimer>&& reclaimer = nullptr) {
    std::unordered_map<std::string, std::shared_ptr<Config>> configs;
    std::shared_ptr<MemoryPool> pool = memoryManager_->addRootPool(
        "",
        memoryCapacity,
        reclaimer != nullptr ? std::move(reclaimer)
                             : MemoryReclaimer::create());
    auto queryCtx = std::make_shared<core::QueryCtx>(
        executor_.get(),
        core::QueryConfig({}),
        configs,
        cache::AsyncDataCache::getInstance(),
        std::move(pool));
    ++numAddedPools_;
    return queryCtx;
  }

  core::PlanNodePtr hashJoinPlan(const std::vector<RowVectorPtr>& vectors) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values(vectors)
        .project({"c0", "c1", "c2"})
        .hashJoin(
            {"c0"},
            {"u1"},
            PlanBuilder(planNodeIdGenerator)
                .values(vectors)
                .project({"c0 AS u0", "c1 AS u1", "c2 AS u2"})
                .planNode(),
            "",
            {"c0", "c1", "c2"},
            core::JoinType::kInner)
        .planNode();
  }

  std::pair<RowVectorPtr, std::shared_ptr<Task>> runHashJoinTask(
      const std::vector<RowVectorPtr>& vectors,
      const std::shared_ptr<core::QueryCtx>& queryCtx,
      uint32_t numDrivers,
      bool enableSpilling,
      const RowVectorPtr& expectedResult = nullptr) {
    const auto plan = hashJoinPlan(vectors);
    RowVectorPtr result;
    std::shared_ptr<Task> task;
    if (enableSpilling) {
      const auto spillDirectory = exec::test::TempDirectoryPath::create();
      result = AssertQueryBuilder(plan)
                   .spillDirectory(spillDirectory->path)
                   .config(core::QueryConfig::kSpillEnabled, "true")
                   .config(core::QueryConfig::kJoinSpillEnabled, "true")
                   .queryCtx(queryCtx)
                   .maxDrivers(numDrivers)
                   .copyResults(pool(), task);
    } else {
      result = AssertQueryBuilder(plan)
                   .queryCtx(queryCtx)
                   .maxDrivers(numDrivers)
                   .copyResults(pool(), task);
    }
    if (expectedResult != nullptr) {
      assertEqualResults({result}, {expectedResult});
    }
    return {result, task};
  }

  core::PlanNodePtr aggregationPlan(const std::vector<RowVectorPtr>& vectors) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values(vectors)
        .singleAggregation({"c0", "c1"}, {"array_agg(c2)"})
        .planNode();
  }

  std::pair<RowVectorPtr, std::shared_ptr<Task>> runAggregateTask(
      const std::vector<RowVectorPtr>& vectors,
      const std::shared_ptr<core::QueryCtx>& queryCtx,
      bool enableSpilling,
      uint32_t numDrivers,
      const RowVectorPtr& expectedResult = nullptr) {
    const auto plan = aggregationPlan(vectors);
    RowVectorPtr result;
    std::shared_ptr<Task> task;
    if (enableSpilling) {
      const auto spillDirectory = exec::test::TempDirectoryPath::create();
      result = AssertQueryBuilder(plan)
                   .spillDirectory(spillDirectory->path)
                   .config(core::QueryConfig::kSpillEnabled, "true")
                   .config(core::QueryConfig::kAggregationSpillEnabled, "true")
                   .queryCtx(queryCtx)
                   .maxDrivers(numDrivers)
                   .copyResults(pool(), task);
    } else {
      result = AssertQueryBuilder(plan)
                   .queryCtx(queryCtx)
                   .maxDrivers(numDrivers)
                   .copyResults(pool(), task);
    }
    if (expectedResult != nullptr) {
      assertEqualResults({result}, {expectedResult});
    }
    return {result, task};
  }

  core::PlanNodePtr orderByPlan(const std::vector<RowVectorPtr>& vectors) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values(vectors)
        .project({"c0", "c1", "c2"})
        .orderBy({"c2 ASC NULLS LAST"}, false)
        .planNode();
  }

  std::pair<RowVectorPtr, std::shared_ptr<Task>> runOrderByTask(
      const std::vector<RowVectorPtr>& vectors,
      const std::shared_ptr<core::QueryCtx>& queryCtx,
      uint32_t numDrivers,
      bool enableSpilling,
      const RowVectorPtr& expectedResult = nullptr) {
    const auto plan = orderByPlan(vectors);
    RowVectorPtr result;
    std::shared_ptr<Task> task;
    if (enableSpilling) {
      const auto spillDirectory = exec::test::TempDirectoryPath::create();
      result = AssertQueryBuilder(plan)
                   .spillDirectory(spillDirectory->path)
                   .config(core::QueryConfig::kSpillEnabled, "true")
                   .config(core::QueryConfig::kOrderBySpillEnabled, "true")
                   .queryCtx(queryCtx)
                   .maxDrivers(numDrivers)
                   .copyResults(pool(), task);
    } else {
      result = AssertQueryBuilder(plan)
                   .queryCtx(queryCtx)
                   .maxDrivers(numDrivers)
                   .copyResults(pool(), task);
    }
    if (expectedResult != nullptr) {
      assertEqualResults({result}, {expectedResult});
    }
    return {result, task};
  }

  core::PlanNodePtr rowNumberPlan(const std::vector<RowVectorPtr>& vectors) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values(vectors)
        .rowNumber({"c0"}, 2, false)
        .project({"c0", "c1"})
        .planNode();
  }

  std::pair<RowVectorPtr, std::shared_ptr<Task>> runRowNumberTask(
      const std::vector<RowVectorPtr>& vectors,
      const std::shared_ptr<core::QueryCtx>& queryCtx,
      uint32_t numDrivers,
      bool enableSpilling,
      const RowVectorPtr& expectedResult = nullptr) {
    const auto plan = rowNumberPlan(vectors);
    RowVectorPtr result;
    std::shared_ptr<Task> task;
    if (enableSpilling) {
      const auto spillDirectory = exec::test::TempDirectoryPath::create();
      result = AssertQueryBuilder(plan)
                   .spillDirectory(spillDirectory->path)
                   .config(core::QueryConfig::kSpillEnabled, "true")
                   .config(core::QueryConfig::kRowNumberSpillEnabled, "true")
                   .queryCtx(queryCtx)
                   .maxDrivers(numDrivers)
                   .copyResults(pool(), task);
    } else {
      result = AssertQueryBuilder(plan)
                   .queryCtx(queryCtx)
                   .maxDrivers(numDrivers)
                   .copyResults(pool(), task);
    }
    if (expectedResult != nullptr) {
      assertEqualResults({result}, {expectedResult});
    }
    return {result, task};
  }

  core::PlanNodePtr topNPlan(const std::vector<RowVectorPtr>& vectors) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values(vectors)
        .project({"c1"})
        .topN({"c1 NULLS FIRST"}, 10, false)
        .planNode();
  }

  std::pair<RowVectorPtr, std::shared_ptr<Task>> runTopNTask(
      const std::vector<RowVectorPtr>& vectors,
      const std::shared_ptr<core::QueryCtx>& queryCtx,
      uint32_t numDrivers,
      bool enableSpilling,
      const RowVectorPtr& expectedResult = nullptr) {
    const auto plan = topNPlan(vectors);
    RowVectorPtr result;
    std::shared_ptr<Task> task;
    if (enableSpilling) {
      const auto spillDirectory = exec::test::TempDirectoryPath::create();
      result =
          AssertQueryBuilder(plan)
              .spillDirectory(spillDirectory->path)
              .config(core::QueryConfig::kSpillEnabled, "true")
              .config(core::QueryConfig::kTopNRowNumberSpillEnabled, "true")
              .queryCtx(queryCtx)
              .maxDrivers(numDrivers)
              .copyResults(pool(), task);
    } else {
      result = AssertQueryBuilder(plan)
                   .queryCtx(queryCtx)
                   .maxDrivers(numDrivers)
                   .copyResults(pool(), task);
    }
    if (expectedResult != nullptr) {
      assertEqualResults({result}, {expectedResult});
    }
    return {result, task};
  }

  core::PlanNodePtr writePlan(
      const std::vector<RowVectorPtr>& vectors,
      const std::string& outputDirPath) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values(vectors)
        .tableWrite(outputDirPath)
        .singleAggregation(
            {},
            {fmt::format("sum({})", TableWriteTraits::rowCountColumnName())})
        .planNode();
  }

  std::pair<RowVectorPtr, std::shared_ptr<Task>> runWriteTask(
      const std::vector<RowVectorPtr>& vectors,
      const std::shared_ptr<core::QueryCtx>& queryCtx,
      uint32_t numDrivers,
      bool enableSpilling,
      const RowVectorPtr& expectedResult = nullptr) {
    const auto outputDirectory = TempDirectoryPath::create();
    auto plan = writePlan(vectors, outputDirectory->path);
    RowVectorPtr result;
    std::shared_ptr<Task> task;
    if (enableSpilling) {
      const auto spillDirectory = exec::test::TempDirectoryPath::create();
      result =
          AssertQueryBuilder(plan)
              .spillDirectory(spillDirectory->path)
              .config(core::QueryConfig::kSpillEnabled, "true")
              .config(core::QueryConfig::kWriterSpillEnabled, "true")
              // Set 0 file writer flush threshold to always trigger flush in
              // test.
              .config(core::QueryConfig::kWriterFlushThresholdBytes, "0")
              // Set stripe size to extreme large to avoid writer internal
              // triggered flush.
              .connectorConfig(
                  kHiveConnectorId,
                  connector::hive::HiveConfig::kOrcWriterMaxStripeSize,
                  "1GB")
              .connectorConfig(
                  kHiveConnectorId,
                  connector::hive::HiveConfig::kOrcWriterMaxDictionaryMemory,
                  "1GB")
              .queryCtx(queryCtx)
              .maxDrivers(numDrivers)
              .copyResults(pool(), task);
    } else {
      result = AssertQueryBuilder(plan)
                   .queryCtx(queryCtx)
                   .maxDrivers(numDrivers)
                   .copyResults(pool(), task);
    }
    if (expectedResult != nullptr) {
      assertEqualResults({result}, {expectedResult});
    }
    return {result, task};
  }

  static inline FakeMemoryOperatorFactory* fakeOperatorFactory_;
  std::shared_ptr<MemoryAllocator> allocator_;
  std::unique_ptr<MemoryManager> memoryManager_;
  SharedArbitrator* arbitrator_;
  RowTypePtr rowType_;
  VectorFuzzer::Options fuzzerOpts_;
  RowVectorPtr vector_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
  std::atomic_uint64_t numAddedPools_{0};
};

DEBUG_ONLY_TEST_F(SharedArbitrationTest, reclaimFromOrderBy) {
  const int numVectors = 32;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);
  std::vector<bool> sameQueries = {false, true};
  for (bool sameQuery : sameQueries) {
    SCOPED_TRACE(fmt::format("sameQuery {}", sameQuery));
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
        newQueryCtx(kMemoryCapacity);
    std::shared_ptr<core::QueryCtx> orderByQueryCtx;
    if (sameQuery) {
      orderByQueryCtx = fakeMemoryQueryCtx;
    } else {
      orderByQueryCtx = newQueryCtx(kMemoryCapacity);
    }

    folly::EventCount fakeAllocationWait;
    auto fakeAllocationWaitKey = fakeAllocationWait.prepareWait();
    folly::EventCount taskPauseWait;
    auto taskPauseWaitKey = taskPauseWait.prepareWait();

    const auto orderByMemoryUsage = 32L << 20;
    const auto fakeAllocationSize = kMemoryCapacity - orderByMemoryUsage / 2;

    std::atomic<bool> injectAllocationOnce{true};
    fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
      if (!injectAllocationOnce.exchange(false)) {
        return TestAllocation{};
      }
      fakeAllocationWait.wait(fakeAllocationWaitKey);
      auto buffer = op->pool()->allocate(fakeAllocationSize);
      return TestAllocation{op->pool(), buffer, fakeAllocationSize};
    });

    std::atomic<bool> injectOrderByOnce{true};
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Driver::runInternal::addInput",
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "OrderBy") {
            return;
          }
          if (op->pool()->capacity() < orderByMemoryUsage) {
            return;
          }
          if (!injectOrderByOnce.exchange(false)) {
            return;
          }
          fakeAllocationWait.notify();
          // Wait for pause to be triggered.
          taskPauseWait.wait(taskPauseWaitKey);
        })));

    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Task::requestPauseLocked",
        std::function<void(Task*)>(
            ([&](Task* /*unused*/) { taskPauseWait.notify(); })));

    std::thread orderByThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .spillDirectory(spillDirectory->path)
              .config(core::QueryConfig::kSpillEnabled, "true")
              .config(core::QueryConfig::kOrderBySpillEnabled, "true")
              .queryCtx(orderByQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .orderBy({"c0 ASC NULLS LAST"}, false)
                        .planNode())
              .assertResults("SELECT * FROM tmp ORDER BY c0 ASC NULLS LAST");
      auto stats = task->taskStats().pipelineStats;
      ASSERT_GT(stats[0].operatorStats[1].spilledBytes, 0);
    });

    std::thread memThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(fakeMemoryQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .addNode([&](std::string id, core::PlanNodePtr input) {
                          return std::make_shared<FakeMemoryNode>(id, input);
                        })
                        .planNode())
              .assertResults("SELECT * FROM tmp");
    });

    orderByThread.join();
    memThread.join();
    waitForAllTasksToBeDeleted();
  }
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, reclaimFromEmptyOrderBy) {
  const int numVectors = 32;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);

  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  std::shared_ptr<core::QueryCtx> orderByQueryCtx =
      newQueryCtx(kMemoryCapacity);

  folly::EventCount fakeAllocationWait;
  auto fakeAllocationWaitKey = fakeAllocationWait.prepareWait();
  folly::EventCount taskPauseWait;
  auto taskPauseWaitKey = taskPauseWait.prepareWait();

  std::atomic<int> injectAllocations{0};
  fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
    const auto injectionCount = ++injectAllocations;
    if (injectionCount > 2) {
      return TestAllocation{};
    }
    if (injectionCount == 1) {
      return TestAllocation{
          op->pool(),
          op->pool()->allocate(kMemoryCapacity / 2),
          kMemoryCapacity / 2};
    }
    fakeAllocationWait.wait(fakeAllocationWaitKey);
    EXPECT_ANY_THROW(op->pool()->allocate(kMemoryCapacity));
    return TestAllocation{};
  });
  fakeOperatorFactory_->setCanReclaim(false);

  core::PlanNodeId orderByPlanNodeId;
  auto orderByPlan = PlanBuilder()
                         .values(vectors)
                         .orderBy({"c0 ASC NULLS LAST"}, false)
                         .capturePlanNodeId(orderByPlanNodeId)
                         .planNode();

  std::atomic<bool> injectDriverBlockOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal",
      std::function<void(Driver*)>(([&](Driver* driver) {
        Operator* op = driver->findOperator(orderByPlanNodeId);
        if (op == nullptr) {
          return;
        }
        if (op->operatorType() != "OrderBy") {
          return;
        }
        if (!injectDriverBlockOnce.exchange(false)) {
          return;
        }
        fakeAllocationWait.notify();
        // Wait for pause to be triggered.
        taskPauseWait.wait(taskPauseWaitKey);
      })));

  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Task::requestPauseLocked",
      std::function<void(Task*)>(
          ([&](Task* /*unused*/) { taskPauseWait.notify(); })));

  std::thread orderByThread([&]() {
    std::shared_ptr<Task> task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->path)
            .maxDrivers(1)
            .config(core::QueryConfig::kSpillEnabled, "true")
            .config(core::QueryConfig::kOrderBySpillEnabled, "true")
            .queryCtx(orderByQueryCtx)
            .plan(PlanBuilder()
                      .values(vectors)
                      .orderBy({"c0 ASC NULLS LAST"}, false)
                      .planNode())
            .assertResults("SELECT * FROM tmp ORDER BY c0 ASC NULLS LAST");
    // Verify no spill has been triggered.
    const auto stats = task->taskStats().pipelineStats;
    ASSERT_EQ(stats[0].operatorStats[1].spilledBytes, 0);
    ASSERT_EQ(stats[0].operatorStats[1].spilledPartitions, 0);
  });

  std::thread memThread([&]() {
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .queryCtx(orderByQueryCtx)
            .maxDrivers(1)
            .plan(PlanBuilder()
                      .values(vectors)
                      .addNode([&](std::string id, core::PlanNodePtr input) {
                        return std::make_shared<FakeMemoryNode>(id, input);
                      })
                      .planNode())
            .assertResults("SELECT * FROM tmp");
  });

  orderByThread.join();
  memThread.join();
  waitForAllTasksToBeDeleted();
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, reclaimToOrderBy) {
  const int numVectors = 32;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);
  std::vector<bool> sameQueries = {false, true};
  for (bool sameQuery : sameQueries) {
    SCOPED_TRACE(fmt::format("sameQuery {}", sameQuery));
    const auto oldStats = arbitrator_->stats();
    std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
        newQueryCtx(kMemoryCapacity);
    std::shared_ptr<core::QueryCtx> orderByQueryCtx;
    if (sameQuery) {
      orderByQueryCtx = fakeMemoryQueryCtx;
    } else {
      orderByQueryCtx = newQueryCtx(kMemoryCapacity);
    }

    folly::EventCount orderByWait;
    auto orderByWaitKey = orderByWait.prepareWait();
    folly::EventCount taskPauseWait;
    auto taskPauseWaitKey = taskPauseWait.prepareWait();

    const auto fakeAllocationSize = kMemoryCapacity - (32L << 20);

    std::atomic<bool> injectAllocationOnce{true};
    fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
      if (!injectAllocationOnce.exchange(false)) {
        return TestAllocation{};
      }
      auto buffer = op->pool()->allocate(fakeAllocationSize);
      orderByWait.notify();
      // Wait for pause to be triggered.
      taskPauseWait.wait(taskPauseWaitKey);
      return TestAllocation{op->pool(), buffer, fakeAllocationSize};
    });

    std::atomic<bool> injectOrderByOnce{true};
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Driver::runInternal::addInput",
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "OrderBy") {
            return;
          }
          if (!injectOrderByOnce.exchange(false)) {
            return;
          }
          orderByWait.wait(orderByWaitKey);
        })));

    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Task::requestPauseLocked",
        std::function<void(Task*)>(
            ([&](Task* /*unused*/) { taskPauseWait.notify(); })));

    std::thread orderByThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(orderByQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .orderBy({"c0 ASC NULLS LAST"}, false)
                        .planNode())
              .assertResults("SELECT * FROM tmp ORDER BY c0 ASC NULLS LAST");
    });

    std::thread memThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(fakeMemoryQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .addNode([&](std::string id, core::PlanNodePtr input) {
                          return std::make_shared<FakeMemoryNode>(id, input);
                        })
                        .planNode())
              .assertResults("SELECT * FROM tmp");
    });

    orderByThread.join();
    memThread.join();
    waitForAllTasksToBeDeleted();
    const auto newStats = arbitrator_->stats();
    ASSERT_GT(newStats.numReclaimedBytes, oldStats.numReclaimedBytes);
    ASSERT_GT(newStats.reclaimTimeUs, oldStats.reclaimTimeUs);
    ASSERT_EQ(arbitrator_->stats().numReserves, numAddedPools_);
  }
}

TEST_F(SharedArbitrationTest, reclaimFromCompletedOrderBy) {
  const int numVectors = 2;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);
  std::vector<bool> sameQueries = {false, true};
  for (bool sameQuery : sameQueries) {
    SCOPED_TRACE(fmt::format("sameQuery {}", sameQuery));
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
        newQueryCtx(kMemoryCapacity);
    std::shared_ptr<core::QueryCtx> orderByQueryCtx;
    if (sameQuery) {
      orderByQueryCtx = fakeMemoryQueryCtx;
    } else {
      orderByQueryCtx = newQueryCtx(kMemoryCapacity);
    }

    folly::EventCount fakeAllocationWait;
    auto fakeAllocationWaitKey = fakeAllocationWait.prepareWait();

    const auto fakeAllocationSize = kMemoryCapacity;

    std::atomic<bool> injectAllocationOnce{true};
    fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
      if (!injectAllocationOnce.exchange(false)) {
        return TestAllocation{};
      }
      fakeAllocationWait.wait(fakeAllocationWaitKey);
      auto buffer = op->pool()->allocate(fakeAllocationSize);
      return TestAllocation{op->pool(), buffer, fakeAllocationSize};
    });

    std::thread orderByThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(orderByQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .orderBy({"c0 ASC NULLS LAST"}, false)
                        .planNode())
              .assertResults("SELECT * FROM tmp ORDER BY c0 ASC NULLS LAST");
      waitForTaskCompletion(task.get());
      fakeAllocationWait.notify();
    });

    std::thread memThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(fakeMemoryQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .addNode([&](std::string id, core::PlanNodePtr input) {
                          return std::make_shared<FakeMemoryNode>(id, input);
                        })
                        .planNode())
              .assertResults("SELECT * FROM tmp");
    });

    orderByThread.join();
    memThread.join();
    waitForAllTasksToBeDeleted();
  }
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, reclaimFromAggregation) {
  const int numVectors = 32;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);
  std::vector<bool> sameQueries = {false, true};
  for (bool sameQuery : sameQueries) {
    SCOPED_TRACE(fmt::format("sameQuery {}", sameQuery));
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
        newQueryCtx(kMemoryCapacity);
    std::shared_ptr<core::QueryCtx> aggregationQueryCtx;
    if (sameQuery) {
      aggregationQueryCtx = fakeMemoryQueryCtx;
    } else {
      aggregationQueryCtx = newQueryCtx(kMemoryCapacity);
    }

    folly::EventCount fakeAllocationWait;
    auto fakeAllocationWaitKey = fakeAllocationWait.prepareWait();
    folly::EventCount taskPauseWait;
    auto taskPauseWaitKey = taskPauseWait.prepareWait();

    const auto aggregationMemoryUsage = 32L << 20;
    const auto fakeAllocationSize =
        kMemoryCapacity - aggregationMemoryUsage + 1;

    std::atomic<bool> injectAllocationOnce{true};
    fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
      if (!injectAllocationOnce.exchange(false)) {
        return TestAllocation{};
      }
      fakeAllocationWait.wait(fakeAllocationWaitKey);
      auto buffer = op->pool()->allocate(fakeAllocationSize);
      return TestAllocation{op->pool(), buffer, fakeAllocationSize};
    });

    std::atomic<bool> injectAggregationByOnce{true};
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Driver::runInternal::addInput",
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "Aggregation") {
            return;
          }
          if (op->pool()->capacity() < aggregationMemoryUsage) {
            return;
          }
          if (!injectAggregationByOnce.exchange(false)) {
            return;
          }
          fakeAllocationWait.notify();
          // Wait for pause to be triggered.
          taskPauseWait.wait(taskPauseWaitKey);
        })));

    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Task::requestPauseLocked",
        std::function<void(Task*)>(
            ([&](Task* /*unused*/) { taskPauseWait.notify(); })));

    std::thread aggregationThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .spillDirectory(spillDirectory->path)
              .config(core::QueryConfig::kSpillEnabled, "true")
              .config(core::QueryConfig::kAggregationSpillEnabled, "true")
              .queryCtx(aggregationQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .singleAggregation({"c0", "c1"}, {"array_agg(c2)"})
                        .planNode())
              .assertResults(
                  "SELECT c0, c1, array_agg(c2) FROM tmp GROUP BY c0, c1");
      auto stats = task->taskStats().pipelineStats;
      ASSERT_GT(stats[0].operatorStats[1].spilledBytes, 0);
    });

    std::thread memThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(fakeMemoryQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .addNode([&](std::string id, core::PlanNodePtr input) {
                          return std::make_shared<FakeMemoryNode>(id, input);
                        })
                        .planNode())
              .assertResults("SELECT * FROM tmp");
    });

    aggregationThread.join();
    memThread.join();
    waitForAllTasksToBeDeleted();
  }
}

TEST_F(SharedArbitrationTest, reclaimFromDistinctAggregation) {
  const uint64_t maxQueryCapacity = 20L << 20;
  std::vector<RowVectorPtr> vectors = newVectors(1024, maxQueryCapacity * 2);
  createDuckDbTable(vectors);
  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  core::PlanNodeId aggrNodeId;
  std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(maxQueryCapacity);
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .spillDirectory(spillDirectory->path)
                  .config(core::QueryConfig::kSpillEnabled, "true")
                  .config(core::QueryConfig::kAggregationSpillEnabled, "true")
                  .queryCtx(queryCtx)
                  .plan(PlanBuilder()
                            .values(vectors)
                            .singleAggregation({"c0"}, {})
                            .capturePlanNodeId(aggrNodeId)
                            .planNode())
                  .assertResults("SELECT distinct c0 FROM tmp");
  auto taskStats = exec::toPlanStats(task->taskStats());
  auto& planStats = taskStats.at(aggrNodeId);
  ASSERT_GT(planStats.spilledBytes, 0);
  task.reset();
  waitForAllTasksToBeDeleted();
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, reclaimFromAggregationOnNoMoreInput) {
  const int numVectors = 32;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);
  for (bool sameQuery : {false, true}) {
    SCOPED_TRACE(fmt::format("sameQuery {}", sameQuery));
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
        newQueryCtx(kMemoryCapacity);
    std::shared_ptr<core::QueryCtx> aggregationQueryCtx;
    if (sameQuery) {
      aggregationQueryCtx = fakeMemoryQueryCtx;
    } else {
      aggregationQueryCtx = newQueryCtx(kMemoryCapacity);
    }

    std::atomic<bool> fakeAllocationBlocked{true};
    folly::EventCount fakeAllocationWait;

    std::atomic<MemoryPool*> injectedPool{nullptr};

    std::atomic<bool> injectAllocationOnce{true};
    fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
      if (!injectAllocationOnce.exchange(false)) {
        return TestAllocation{};
      }
      fakeAllocationWait.await([&]() { return !fakeAllocationBlocked.load(); });
      EXPECT_TRUE(injectedPool != nullptr);
      const auto fakeAllocationSize =
          kMemoryCapacity - injectedPool.load()->reservedBytes() + 1;
      return TestAllocation{
          op->pool(),
          op->pool()->allocate(fakeAllocationSize),
          fakeAllocationSize};
    });

    folly::EventCount taskPauseWait;
    std::atomic<bool> taskPaused{false};
    std::atomic<bool> injectNoMoreInputOnce{true};
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Driver::runInternal::noMoreInput",
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "Aggregation") {
            return;
          }
          if (!injectNoMoreInputOnce.exchange(false)) {
            return;
          }
          injectedPool = op->pool();
          fakeAllocationBlocked = false;
          fakeAllocationWait.notifyAll();
          taskPauseWait.await([&]() { return taskPaused.load(); });
        })));

    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Task::requestPauseLocked",
        std::function<void(Task*)>(([&](Task* /*unused*/) {
          taskPaused = true;
          taskPauseWait.notifyAll();
        })));

    std::thread aggregationThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .spillDirectory(spillDirectory->path)
              .config(core::QueryConfig::kSpillEnabled, "true")
              .config(core::QueryConfig::kAggregationSpillEnabled, "true")
              .queryCtx(aggregationQueryCtx)
              .maxDrivers(1)
              .plan(PlanBuilder()
                        .values(vectors)
                        .singleAggregation({"c0", "c1"}, {"array_agg(c2)"})
                        .planNode())
              .assertResults(
                  "SELECT c0, c1, array_agg(c2) FROM tmp GROUP BY c0, c1");
      auto stats = task->taskStats().pipelineStats;
      ASSERT_GT(stats[0].operatorStats[1].spilledBytes, 0);
    });

    std::thread memThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(fakeMemoryQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .addNode([&](std::string id, core::PlanNodePtr input) {
                          return std::make_shared<FakeMemoryNode>(id, input);
                        })
                        .planNode())
              .assertResults("SELECT * FROM tmp");
    });

    aggregationThread.join();
    memThread.join();
    waitForAllTasksToBeDeleted();
  }
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, reclaimFromAggregationDuringOutput) {
  const int numVectors = 32;
  std::vector<RowVectorPtr> vectors;
  int numRows{0};
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
    numRows += vectors.back()->size();
  }
  createDuckDbTable(vectors);
  for (bool sameQuery : {false, true}) {
    SCOPED_TRACE(fmt::format("sameQuery {}", sameQuery));
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
        newQueryCtx(kMemoryCapacity);
    std::shared_ptr<core::QueryCtx> aggregationQueryCtx;
    if (sameQuery) {
      aggregationQueryCtx = fakeMemoryQueryCtx;
    } else {
      aggregationQueryCtx = newQueryCtx(kMemoryCapacity);
    }

    std::atomic<bool> fakeAllocationBlocked{true};
    folly::EventCount fakeAllocationWait;

    std::atomic<MemoryPool*> injectedPool{nullptr};

    std::atomic<bool> injectAllocationOnce{true};
    fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
      if (!injectAllocationOnce.exchange(false)) {
        return TestAllocation{};
      }
      fakeAllocationWait.await([&]() { return !fakeAllocationBlocked.load(); });
      EXPECT_TRUE(injectedPool != nullptr);
      const auto fakeAllocationSize =
          kMemoryCapacity - injectedPool.load()->reservedBytes() + 1;
      return TestAllocation{
          op->pool(),
          op->pool()->allocate(fakeAllocationSize),
          fakeAllocationSize};
    });

    folly::EventCount taskPauseWait;
    std::atomic<bool> taskPaused{false};
    std::atomic<int> injectGetOutputCount{0};
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Driver::runInternal::getOutput",
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "Aggregation") {
            return;
          }
          if (!op->testingNoMoreInput()) {
            return;
          }
          if (++injectGetOutputCount != 3) {
            return;
          }
          injectedPool = op->pool();
          fakeAllocationBlocked = false;
          fakeAllocationWait.notifyAll();
          taskPauseWait.await([&]() { return taskPaused.load(); });
        })));

    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Task::requestPauseLocked",
        std::function<void(Task*)>(([&](Task* /*unused*/) {
          taskPaused = true;
          taskPauseWait.notifyAll();
        })));

    std::thread aggregationThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .spillDirectory(spillDirectory->path)
              .config(core::QueryConfig::kSpillEnabled, "true")
              .config(core::QueryConfig::kAggregationSpillEnabled, "true")
              .config(
                  core::QueryConfig::kPreferredOutputBatchRows,
                  std::to_string(numRows / 10))
              .maxDrivers(1)
              .queryCtx(aggregationQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .singleAggregation({"c0", "c1"}, {"array_agg(c2)"})
                        .planNode())
              .assertResults(
                  "SELECT c0, c1, array_agg(c2) FROM tmp GROUP BY c0, c1");
      auto stats = task->taskStats().pipelineStats;
      ASSERT_GT(stats[0].operatorStats[1].spilledBytes, 0);
    });

    std::thread memThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(fakeMemoryQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .addNode([&](std::string id, core::PlanNodePtr input) {
                          return std::make_shared<FakeMemoryNode>(id, input);
                        })
                        .planNode())
              .assertResults("SELECT * FROM tmp");
    });

    aggregationThread.join();
    memThread.join();
    waitForAllTasksToBeDeleted();
  }
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, reclaimToAggregation) {
  const int numVectors = 32;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);
  std::vector<bool> sameQueries = {false, true};
  for (bool sameQuery : sameQueries) {
    SCOPED_TRACE(fmt::format("sameQuery {}", sameQuery));
    const auto oldStats = arbitrator_->stats();
    std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
        newQueryCtx(kMemoryCapacity);
    std::shared_ptr<core::QueryCtx> aggregationQueryCtx;
    if (sameQuery) {
      aggregationQueryCtx = fakeMemoryQueryCtx;
    } else {
      aggregationQueryCtx = newQueryCtx(kMemoryCapacity);
    }

    folly::EventCount aggregationWait;
    auto aggregationWaitKey = aggregationWait.prepareWait();
    folly::EventCount taskPauseWait;
    auto taskPauseWaitKey = taskPauseWait.prepareWait();

    const auto fakeAllocationSize = kMemoryCapacity - (32L << 20);

    std::atomic<bool> injectAllocationOnce{true};
    fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
      if (!injectAllocationOnce.exchange(false)) {
        return TestAllocation{};
      }
      auto buffer = op->pool()->allocate(fakeAllocationSize);
      aggregationWait.notify();
      // Wait for pause to be triggered.
      taskPauseWait.wait(taskPauseWaitKey);
      return TestAllocation{op->pool(), buffer, fakeAllocationSize};
    });

    std::atomic<bool> injectAggregationOnce{true};
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Driver::runInternal::addInput",
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "Aggregation") {
            return;
          }
          if (!injectAggregationOnce.exchange(false)) {
            return;
          }
          aggregationWait.wait(aggregationWaitKey);
        })));

    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Task::requestPauseLocked",
        std::function<void(Task*)>(
            ([&](Task* /*unused*/) { taskPauseWait.notify(); })));

    std::thread aggregationThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(aggregationQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .singleAggregation({"c0", "c1"}, {"array_agg(c2)"})
                        .planNode())
              .assertResults(
                  "SELECT c0, c1, array_agg(c2) FROM tmp GROUP BY c0, c1");
    });

    std::thread memThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(fakeMemoryQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .addNode([&](std::string id, core::PlanNodePtr input) {
                          return std::make_shared<FakeMemoryNode>(id, input);
                        })
                        .planNode())
              .assertResults("SELECT * FROM tmp");
    });

    aggregationThread.join();
    memThread.join();
    waitForAllTasksToBeDeleted();

    const auto newStats = arbitrator_->stats();
    ASSERT_GT(newStats.numReclaimedBytes, oldStats.numReclaimedBytes);
    ASSERT_GT(newStats.reclaimTimeUs, oldStats.reclaimTimeUs);
    ASSERT_EQ(newStats.numReserves, numAddedPools_);
  }
}

TEST_F(SharedArbitrationTest, reclaimFromCompletedAggregation) {
  const int numVectors = 2;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);
  std::vector<bool> sameQueries = {false, true};
  for (bool sameQuery : sameQueries) {
    SCOPED_TRACE(fmt::format("sameQuery {}", sameQuery));
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
        newQueryCtx(kMemoryCapacity);
    std::shared_ptr<core::QueryCtx> aggregationQueryCtx;
    if (sameQuery) {
      aggregationQueryCtx = fakeMemoryQueryCtx;
    } else {
      aggregationQueryCtx = newQueryCtx(kMemoryCapacity);
    }

    folly::EventCount fakeAllocationWait;
    auto fakeAllocationWaitKey = fakeAllocationWait.prepareWait();

    const auto fakeAllocationSize = kMemoryCapacity;

    std::atomic<bool> injectAllocationOnce{true};
    fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
      if (!injectAllocationOnce.exchange(false)) {
        return TestAllocation{};
      }
      fakeAllocationWait.wait(fakeAllocationWaitKey);
      auto buffer = op->pool()->allocate(fakeAllocationSize);
      return TestAllocation{op->pool(), buffer, fakeAllocationSize};
    });

    std::thread aggregationThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(aggregationQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .singleAggregation({"c0", "c1"}, {"array_agg(c2)"})
                        .planNode())
              .assertResults(
                  "SELECT c0, c1, array_agg(c2) FROM tmp GROUP BY c0, c1");
      waitForTaskCompletion(task.get());
      fakeAllocationWait.notify();
    });

    std::thread memThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(fakeMemoryQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .addNode([&](std::string id, core::PlanNodePtr input) {
                          return std::make_shared<FakeMemoryNode>(id, input);
                        })
                        .planNode())
              .assertResults("SELECT * FROM tmp");
    });

    aggregationThread.join();
    memThread.join();
    waitForAllTasksToBeDeleted();
  }
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, reclaimFromJoinBuilder) {
  const int numVectors = 32;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);
  std::vector<bool> sameQueries = {false, true};
  for (bool sameQuery : sameQueries) {
    SCOPED_TRACE(fmt::format("sameQuery {}", sameQuery));
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
        newQueryCtx(kMemoryCapacity);
    std::shared_ptr<core::QueryCtx> joinQueryCtx;
    if (sameQuery) {
      joinQueryCtx = fakeMemoryQueryCtx;
    } else {
      joinQueryCtx = newQueryCtx(kMemoryCapacity);
    }

    std::atomic_bool fakeAllocationReady{false};
    folly::EventCount fakeAllocationWait;
    std::atomic_bool taskPauseDone{false};
    folly::EventCount taskPauseWait;

    const auto joinMemoryUsage = 32L << 20;
    const auto fakeAllocationSize = kMemoryCapacity - joinMemoryUsage / 2;

    std::atomic<bool> injectAllocationOnce{true};
    fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
      if (!injectAllocationOnce.exchange(false)) {
        return TestAllocation{};
      }
      fakeAllocationWait.await([&]() { return fakeAllocationReady.load(); });
      auto buffer = op->pool()->allocate(fakeAllocationSize);
      return TestAllocation{op->pool(), buffer, fakeAllocationSize};
    });

    std::atomic<bool> injectAggregationByOnce{true};
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Driver::runInternal::addInput",
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "HashBuild") {
            return;
          }
          if (op->pool()->currentBytes() < joinMemoryUsage) {
            return;
          }
          if (!injectAggregationByOnce.exchange(false)) {
            return;
          }
          fakeAllocationReady.store(true);
          fakeAllocationWait.notifyAll();
          // Wait for pause to be triggered.
          taskPauseWait.await([&]() { return taskPauseDone.load(); });
        })));

    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Task::requestPauseLocked",
        std::function<void(Task*)>(([&](Task* /*unused*/) {
          taskPauseDone.store(true);
          taskPauseWait.notifyAll();
        })));

    // joinQueryCtx and fakeMemoryQueryCtx may be the same and thus share the
    // same underlying QueryConfig.  We apply the changes here instead of using
    // the AssertQueryBuilder to avoid a potential race condition caused by
    // writing the config in the join thread, and reading it in the memThread.
    std::unordered_map<std::string, std::string> config{
        {core::QueryConfig::kSpillEnabled, "true"},
        {core::QueryConfig::kJoinSpillEnabled, "true"},
        {core::QueryConfig::kJoinSpillPartitionBits, "2"},
    };
    joinQueryCtx->testingOverrideConfigUnsafe(std::move(config));

    std::thread aggregationThread([&]() {
      auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .spillDirectory(spillDirectory->path)
              .queryCtx(joinQueryCtx)
              .plan(PlanBuilder(planNodeIdGenerator)
                        .values(vectors)
                        .project({"c0 AS t0", "c1 AS t1", "c2 AS t2"})
                        .hashJoin(
                            {"t0"},
                            {"u0"},
                            PlanBuilder(planNodeIdGenerator)
                                .values(vectors)
                                .project({"c0 AS u0", "c1 AS u1", "c2 AS u2"})
                                .planNode(),
                            "",
                            {"t1"},
                            core::JoinType::kAnti)
                        .planNode())
              .assertResults(
                  "SELECT c1 FROM tmp WHERE c0 NOT IN (SELECT c0 FROM tmp)");
      auto stats = task->taskStats().pipelineStats;
      ASSERT_GT(stats[1].operatorStats[2].spilledBytes, 0);
    });

    std::thread memThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(fakeMemoryQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .addNode([&](std::string id, core::PlanNodePtr input) {
                          return std::make_shared<FakeMemoryNode>(id, input);
                        })
                        .planNode())
              .assertResults("SELECT * FROM tmp");
    });

    aggregationThread.join();
    memThread.join();
    waitForAllTasksToBeDeleted();
  }
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, reclaimToJoinBuilder) {
  const int numVectors = 32;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);
  std::vector<bool> sameQueries = {false, true};
  for (bool sameQuery : sameQueries) {
    SCOPED_TRACE(fmt::format("sameQuery {}", sameQuery));
    const auto oldStats = arbitrator_->stats();
    std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
        newQueryCtx(kMemoryCapacity);
    std::shared_ptr<core::QueryCtx> joinQueryCtx;
    if (sameQuery) {
      joinQueryCtx = fakeMemoryQueryCtx;
    } else {
      joinQueryCtx = newQueryCtx(kMemoryCapacity);
    }

    folly::EventCount joinWait;
    auto joinWaitKey = joinWait.prepareWait();
    folly::EventCount taskPauseWait;
    auto taskPauseWaitKey = taskPauseWait.prepareWait();

    const auto fakeAllocationSize = kMemoryCapacity - (32L << 20);

    std::atomic<bool> injectAllocationOnce{true};
    fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
      if (!injectAllocationOnce.exchange(false)) {
        return TestAllocation{};
      }
      auto buffer = op->pool()->allocate(fakeAllocationSize);
      joinWait.notify();
      // Wait for pause to be triggered.
      taskPauseWait.wait(taskPauseWaitKey);
      return TestAllocation{op->pool(), buffer, fakeAllocationSize};
    });

    std::atomic<bool> injectJoinOnce{true};
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Driver::runInternal::addInput",
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "HashBuild") {
            return;
          }
          if (!injectJoinOnce.exchange(false)) {
            return;
          }
          joinWait.wait(joinWaitKey);
        })));

    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Task::requestPauseLocked",
        std::function<void(Task*)>(
            ([&](Task* /*unused*/) { taskPauseWait.notify(); })));

    std::thread joinThread([&]() {
      auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(joinQueryCtx)
              .plan(PlanBuilder(planNodeIdGenerator)
                        .values(vectors)
                        .project({"c0 AS t0", "c1 AS t1", "c2 AS t2"})
                        .hashJoin(
                            {"t0"},
                            {"u0"},
                            PlanBuilder(planNodeIdGenerator)
                                .values(vectors)
                                .project({"c0 AS u0", "c1 AS u1", "c2 AS u2"})
                                .planNode(),
                            "",
                            {"t1"},
                            core::JoinType::kAnti)
                        .planNode())
              .assertResults(
                  "SELECT c1 FROM tmp WHERE c0 NOT IN (SELECT c0 FROM tmp)");
    });

    std::thread memThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(fakeMemoryQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .addNode([&](std::string id, core::PlanNodePtr input) {
                          return std::make_shared<FakeMemoryNode>(id, input);
                        })
                        .planNode())
              .assertResults("SELECT * FROM tmp");
    });

    joinThread.join();
    memThread.join();
    waitForAllTasksToBeDeleted();

    const auto newStats = arbitrator_->stats();
    ASSERT_GT(newStats.numReclaimedBytes, oldStats.numReclaimedBytes);
    ASSERT_GT(newStats.reclaimTimeUs, oldStats.reclaimTimeUs);
    ASSERT_EQ(arbitrator_->stats().numReserves, numAddedPools_);
  }
}

TEST_F(SharedArbitrationTest, reclaimFromCompletedJoinBuilder) {
  const int numVectors = 2;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);
  std::vector<bool> sameQueries = {false, true};
  for (bool sameQuery : sameQueries) {
    SCOPED_TRACE(fmt::format("sameQuery {}", sameQuery));
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    const uint64_t numCreatedTasks = Task::numCreatedTasks();
    std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
        newQueryCtx(kMemoryCapacity);
    std::shared_ptr<core::QueryCtx> joinQueryCtx;
    if (sameQuery) {
      joinQueryCtx = fakeMemoryQueryCtx;
    } else {
      joinQueryCtx = newQueryCtx(kMemoryCapacity);
    }

    folly::EventCount fakeAllocationWait;
    auto fakeAllocationWaitKey = fakeAllocationWait.prepareWait();

    const auto fakeAllocationSize = kMemoryCapacity;

    std::atomic<bool> injectAllocationOnce{true};
    fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
      if (!injectAllocationOnce.exchange(false)) {
        return TestAllocation{};
      }
      fakeAllocationWait.wait(fakeAllocationWaitKey);
      auto buffer = op->pool()->allocate(fakeAllocationSize);
      return TestAllocation{op->pool(), buffer, fakeAllocationSize};
    });

    std::thread joinThread([&]() {
      auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(joinQueryCtx)
              .plan(PlanBuilder(planNodeIdGenerator)
                        .values(vectors)
                        .project({"c0 AS t0", "c1 AS t1", "c2 AS t2"})
                        .hashJoin(
                            {"t0"},
                            {"u0"},
                            PlanBuilder(planNodeIdGenerator)
                                .values(vectors)
                                .project({"c0 AS u0", "c1 AS u1", "c2 AS u2"})
                                .planNode(),
                            "",
                            {"t1"},
                            core::JoinType::kAnti)
                        .planNode())
              .assertResults(
                  "SELECT c1 FROM tmp WHERE c0 NOT IN (SELECT c0 FROM tmp)");
      waitForTaskCompletion(task.get());
      task.reset();
      // Make sure the join query task has been destroyed.
      waitForAllTasksToBeDeleted(numCreatedTasks + 1, 3'000'000);
      fakeAllocationWait.notify();
    });

    std::thread memThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(fakeMemoryQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .addNode([&](std::string id, core::PlanNodePtr input) {
                          return std::make_shared<FakeMemoryNode>(id, input);
                        })
                        .planNode())
              .assertResults("SELECT * FROM tmp");
    });

    joinThread.join();
    memThread.join();
    waitForAllTasksToBeDeleted();
  }
}

DEBUG_ONLY_TEST_F(
    SharedArbitrationTest,
    reclaimFromJoinBuilderWithMultiDrivers) {
  const int numVectors = 32;
  std::vector<RowVectorPtr> vectors;
  fuzzerOpts_.vectorSize = 128;
  fuzzerOpts_.stringVariableLength = false;
  fuzzerOpts_.stringLength = 512;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  const int numDrivers = 4;
  const auto expectedResult =
      runHashJoinTask(vectors, nullptr, numDrivers, false).first;

  // Whether the tasks are run under the same query context.
  std::vector<bool> sameQueries = {false, true};
  for (bool sameQuery : sameQueries) {
    SCOPED_TRACE(fmt::format("sameQuery {}", sameQuery));
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
        newQueryCtx(kMemoryCapacity);
    std::shared_ptr<core::QueryCtx> joinQueryCtx;
    if (sameQuery) {
      joinQueryCtx = fakeMemoryQueryCtx;
    } else {
      joinQueryCtx = newQueryCtx(kMemoryCapacity);
    }
    // This is an experimentally calculated size that ensures the join will
    // spill. Can change as the join implementation evolves.
    const auto joinMemoryUsage = 11 * MB;
    const auto fakeAllocationSize = kMemoryCapacity - joinMemoryUsage;

    // Ensure a single allocation eats up the rest of the memory.
    std::atomic<bool> injectAllocationOnce{true};
    std::atomic<bool> fakeAllocationWaitFlag{true};
    folly::EventCount fakeAllocationWait;
    fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
      if (!injectAllocationOnce.exchange(false)) {
        return TestAllocation{};
      }
      fakeAllocationWait.await(
          [&]() { return !fakeAllocationWaitFlag.load(); });
      auto buffer = op->pool()->allocate(fakeAllocationSize);
      return TestAllocation{op->pool(), buffer, fakeAllocationSize};
    });

    // Ensure the build operators make enough progress to consume more than
    // `joinMemoryUsage` then wait for the fake allocation to eat up memory and
    // initiate a spill via the arbitrator.
    std::atomic<int> injectCount{0};
    folly::futures::Barrier builderBarrier(numDrivers);
    std::atomic<bool> taskPauseWaitFlag{true};
    folly::EventCount taskPauseWait;
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Driver::runInternal::addInput",
        std::function<void(Operator*)>(([&](Operator* op) {
          if (op->operatorType() != "HashBuild") {
            return;
          }
          // Make sure each hash build operator has reserved memory to avoid
          // trigger memory arbitration in test.
          if (static_cast<MemoryPoolImpl*>(op->pool())
                  ->testingMinReservationBytes() == 0) {
            return;
          }
          // Check all the hash build operators' memory usage instead of
          // individual operator.
          if (op->pool()->parent()->currentBytes() < joinMemoryUsage) {
            return;
          }
          if (++injectCount > numDrivers) {
            return;
          }
          auto future = builderBarrier.wait();
          if (future.wait().value()) {
            fakeAllocationWaitFlag = false;
            fakeAllocationWait.notifyAll();
          }
          // Wait for pause to be triggered.
          taskPauseWait.await([&]() { return !taskPauseWaitFlag.load(); });
        })));

    // This indicates that a spill has been initiated by the arbitrator.
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Task::requestPauseLocked",
        std::function<void(Task*)>([&](Task* /*unused*/) {
          taskPauseWaitFlag = false;
          taskPauseWait.notifyAll();
        }));

    // joinQueryCtx and fakeMemoryQueryCtx may be the same and thus share the
    // same underlying QueryConfig.  We apply the changes here instead of using
    // the AssertQueryBuilder to avoid a potential race condition caused by
    // writing the config in the join thread, and reading it in the memThread.
    std::unordered_map<std::string, std::string> config{
        {core::QueryConfig::kSpillEnabled, "true"},
        {core::QueryConfig::kJoinSpillEnabled, "true"},
        {core::QueryConfig::kJoinSpillPartitionBits, "2"}};
    joinQueryCtx->testingOverrideConfigUnsafe(std::move(config));

    std::thread joinThread([&]() {
      auto task = runHashJoinTask(
                      vectors, joinQueryCtx, numDrivers, true, expectedResult)
                      .second;
      auto stats = task->taskStats().pipelineStats;
      // Verify that spilling occured.
      ASSERT_GT(stats[1].operatorStats[2].spilledBytes, 0);
    });

    std::thread memThread([&]() {
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .queryCtx(fakeMemoryQueryCtx)
              .plan(PlanBuilder()
                        .values(vectors)
                        .addNode([&](std::string id, core::PlanNodePtr input) {
                          return std::make_shared<FakeMemoryNode>(id, input);
                        })
                        .planNode())
              .assertResults("SELECT * FROM tmp");
    });
    joinThread.join();
    memThread.join();
    waitForAllTasksToBeDeleted();
    ASSERT_GT(arbitrator_->stats().numRequests, 0);
  }
}

DEBUG_ONLY_TEST_F(
    SharedArbitrationTest,
    failedToReclaimFromHashJoinBuildersInNonReclaimableSection) {
  const int numVectors = 32;
  std::vector<RowVectorPtr> vectors;
  fuzzerOpts_.vectorSize = 128;
  fuzzerOpts_.stringVariableLength = false;
  fuzzerOpts_.stringLength = 512;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  const int numDrivers = 4;
  createDuckDbTable(vectors);
  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
      newQueryCtx(kMemoryCapacity);
  std::shared_ptr<core::QueryCtx> joinQueryCtx = newQueryCtx(kMemoryCapacity);

  std::atomic<bool> allocationWaitFlag{true};
  folly::EventCount allocationWait;
  std::atomic<bool> allocationDoneWaitFlag{true};
  folly::EventCount allocationDoneWait;

  const auto joinMemoryUsage = 8L << 20;
  const auto fakeAllocationSize = kMemoryCapacity - joinMemoryUsage / 3;

  std::atomic<bool> injectAllocationOnce{true};
  fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
    if (!injectAllocationOnce.exchange(false)) {
      return TestAllocation{};
    }
    allocationWait.await([&]() { return !allocationWaitFlag.load(); });
    EXPECT_ANY_THROW(op->pool()->allocate(fakeAllocationSize));
    allocationDoneWaitFlag = false;
    allocationDoneWait.notifyAll();
    return TestAllocation{};
  });

  std::atomic<bool> injectNonReclaimableSectionOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::common::memory::MemoryPoolImpl::allocateNonContiguous",
      std::function<void(memory::MemoryPoolImpl*)>(
          ([&](memory::MemoryPoolImpl* pool) {
            const std::string re(".*HashBuild");
            if (!RE2::FullMatch(pool->name(), re)) {
              return;
            }
            if (pool->parent()->currentBytes() < joinMemoryUsage) {
              return;
            }
            if (!injectNonReclaimableSectionOnce.exchange(false)) {
              return;
            }
            allocationWaitFlag = false;
            allocationWait.notifyAll();

            // Suspend the driver to simulate the arbitration.
            pool->reclaimer()->enterArbitration();
            allocationDoneWait.await(
                [&]() { return !allocationDoneWaitFlag.load(); });
            pool->reclaimer()->leaveArbitration();
          })));

  std::thread joinThread([&]() {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->path)
            .config(core::QueryConfig::kSpillEnabled, "true")
            .config(core::QueryConfig::kJoinSpillEnabled, "true")
            .config(core::QueryConfig::kJoinSpillPartitionBits, "2")
            .maxDrivers(numDrivers)
            .queryCtx(joinQueryCtx)
            .plan(PlanBuilder(planNodeIdGenerator)
                      .values(vectors, true)
                      .project({"c0 AS t0", "c1 AS t1", "c2 AS t2"})
                      .hashJoin(
                          {"t0"},
                          {"u1"},
                          PlanBuilder(planNodeIdGenerator)
                              .values(vectors, true)
                              .project({"c0 AS u0", "c1 AS u1", "c2 AS u2"})
                              .planNode(),
                          "",
                          {"t1"},
                          core::JoinType::kInner)
                      .planNode())
            .assertResults(
                "SELECT t.c1 FROM tmp as t, tmp AS u WHERE t.c0 == u.c1");
    // We expect the spilling is not triggered because of non-reclaimable
    // section.
    auto stats = task->taskStats().pipelineStats;
    ASSERT_EQ(stats[1].operatorStats[2].spilledBytes, 0);
  });

  std::thread memThread([&]() {
    AssertQueryBuilder(duckDbQueryRunner_)
        .queryCtx(fakeMemoryQueryCtx)
        .plan(PlanBuilder()
                  .values(vectors)
                  .addNode([&](std::string id, core::PlanNodePtr input) {
                    return std::make_shared<FakeMemoryNode>(id, input);
                  })
                  .planNode())
        .assertResults("SELECT * FROM tmp");
  });
  joinThread.join();
  memThread.join();
  waitForAllTasksToBeDeleted();
  ASSERT_EQ(arbitrator_->stats().numNonReclaimableAttempts, 2);
  ASSERT_EQ(arbitrator_->stats().numReserves, numAddedPools_);
}

DEBUG_ONLY_TEST_F(
    SharedArbitrationTest,
    reclaimFromJoinBuildWaitForTableBuild) {
  setupMemory(kMemoryCapacity, 0);
  const int numVectors = 32;
  std::vector<RowVectorPtr> vectors;
  fuzzerOpts_.vectorSize = 128;
  fuzzerOpts_.stringVariableLength = false;
  fuzzerOpts_.stringLength = 512;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  const int numDrivers = 4;
  createDuckDbTable(vectors);

  std::shared_ptr<core::QueryCtx> fakeMemoryQueryCtx =
      newQueryCtx(kMemoryCapacity);
  std::shared_ptr<core::QueryCtx> joinQueryCtx = newQueryCtx(kMemoryCapacity);

  folly::EventCount fakeAllocationWait;
  std::atomic<bool> fakeAllocationUnblock{false};
  std::atomic<bool> injectFakeAllocationOnce{true};

  fakeOperatorFactory_->setAllocationCallback([&](Operator* op) {
    if (!injectFakeAllocationOnce.exchange(false)) {
      return TestAllocation{};
    }
    fakeAllocationWait.await([&]() { return fakeAllocationUnblock.load(); });
    // Set the fake allocation size to trigger memory reclaim.
    const auto fakeAllocationSize = arbitrator_->stats().freeCapacityBytes +
        joinQueryCtx->pool()->freeBytes() + 1;
    return TestAllocation{
        op->pool(),
        op->pool()->allocate(fakeAllocationSize),
        fakeAllocationSize};
  });

  folly::futures::Barrier builderBarrier(numDrivers);
  folly::futures::Barrier pauseBarrier(numDrivers + 1);
  std::atomic<int> addInputInjectCount{0};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal",
      std::function<void(Driver*)>(([&](Driver* driver) {
        // Check if the driver is from join query.
        if (driver->task()->queryCtx()->pool()->name() !=
            joinQueryCtx->pool()->name()) {
          return;
        }
        // Check if the driver is from the pipeline with hash build.
        if (driver->driverCtx()->pipelineId != 1) {
          return;
        }
        if (++addInputInjectCount > numDrivers - 1) {
          return;
        }
        if (builderBarrier.wait().get()) {
          fakeAllocationUnblock = true;
          fakeAllocationWait.notifyAll();
        }
        // Wait for pause to be triggered.
        pauseBarrier.wait().get();
      })));

  std::atomic<bool> injectNoMoreInputOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal::noMoreInput",
      std::function<void(Operator*)>(([&](Operator* op) {
        if (op->operatorType() != "HashBuild") {
          return;
        }
        if (!injectNoMoreInputOnce.exchange(false)) {
          return;
        }
        if (builderBarrier.wait().get()) {
          fakeAllocationUnblock = true;
          fakeAllocationWait.notifyAll();
        }
        // Wait for pause to be triggered.
        pauseBarrier.wait().get();
      })));

  std::atomic<bool> injectPauseOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Task::requestPauseLocked",
      std::function<void(Task*)>([&](Task* /*unused*/) {
        if (!injectPauseOnce.exchange(false)) {
          return;
        }
        pauseBarrier.wait().get();
      }));

  // Verifies that we only trigger the hash build reclaim once.
  std::atomic<int> numHashBuildReclaims{0};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::HashBuild::reclaim",
      std::function<void(Operator*)>(
          ([&](Operator* /*unused*/) { ++numHashBuildReclaims; })));

  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  std::thread joinThread([&]() {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->path)
            .config(core::QueryConfig::kSpillEnabled, "true")
            .config(core::QueryConfig::kJoinSpillEnabled, "true")
            .config(core::QueryConfig::kJoinSpillPartitionBits, "2")
            // NOTE: set an extreme large value to avoid non-reclaimable
            // section in test.
            .config(core::QueryConfig::kSpillableReservationGrowthPct, "8000")
            .maxDrivers(numDrivers)
            .queryCtx(joinQueryCtx)
            .plan(PlanBuilder(planNodeIdGenerator)
                      .values(vectors, true)
                      .project({"c0 AS t0", "c1 AS t1", "c2 AS t2"})
                      .hashJoin(
                          {"t0"},
                          {"u1"},
                          PlanBuilder(planNodeIdGenerator)
                              .values(vectors, true)
                              .project({"c0 AS u0", "c1 AS u1", "c2 AS u2"})
                              .planNode(),
                          "",
                          {"t1"},
                          core::JoinType::kInner)
                      .planNode())
            .assertResults(
                "SELECT t.c1 FROM tmp as t, tmp AS u WHERE t.c0 == u.c1");
    // We expect the spilling triggered.
    auto stats = task->taskStats().pipelineStats;
    ASSERT_GT(stats[1].operatorStats[2].spilledBytes, 0);
  });

  std::thread memThread([&]() {
    AssertQueryBuilder(duckDbQueryRunner_)
        .queryCtx(fakeMemoryQueryCtx)
        .plan(PlanBuilder()
                  .values(vectors)
                  .addNode([&](std::string id, core::PlanNodePtr input) {
                    return std::make_shared<FakeMemoryNode>(id, input);
                  })
                  .planNode())
        .assertResults("SELECT * FROM tmp");
  });

  joinThread.join();
  memThread.join();

  // We only expect to reclaim from one hash build operator once.
  ASSERT_EQ(numHashBuildReclaims, 1);
  waitForAllTasksToBeDeleted();
}

DEBUG_ONLY_TEST_F(
    SharedArbitrationTest,
    arbitrationTriggeredDuringParallelJoinBuild) {
  const int numVectors = 2;
  std::vector<RowVectorPtr> vectors;
  // Build a large vector to trigger memory arbitration.
  fuzzerOpts_.vectorSize = 10'000;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);

  std::shared_ptr<core::QueryCtx> joinQueryCtx = newQueryCtx(kMemoryCapacity);

  // Make sure the parallel build has been triggered.
  std::atomic<bool> parallelBuildTriggered{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::HashTable::parallelJoinBuild",
      std::function<void(void*)>(
          [&](void*) { parallelBuildTriggered = true; }));

  // TODO: add driver context to test if the memory allocation is triggered in
  // driver context or not.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  AssertQueryBuilder(duckDbQueryRunner_)
      // Set very low table size threshold to trigger parallel build.
      .config(
          core::QueryConfig::kMinTableRowsForParallelJoinBuild,
          std::to_string(0))
      // Set multiple hash build drivers to trigger parallel build.
      .maxDrivers(4)
      .queryCtx(joinQueryCtx)
      .plan(PlanBuilder(planNodeIdGenerator)
                .values(vectors, true)
                .project({"c0 AS t0", "c1 AS t1", "c2 AS t2"})
                .hashJoin(
                    {"t0", "t1"},
                    {"u1", "u0"},
                    PlanBuilder(planNodeIdGenerator)
                        .values(vectors, true)
                        .project({"c0 AS u0", "c1 AS u1", "c2 AS u2"})
                        .planNode(),
                    "",
                    {"t1"},
                    core::JoinType::kInner)
                .planNode())
      .assertResults(
          "SELECT t.c1 FROM tmp as t, tmp AS u WHERE t.c0 == u.c1 AND t.c1 == u.c0");
  ASSERT_TRUE(parallelBuildTriggered);
  waitForAllTasksToBeDeleted();
}

DEBUG_ONLY_TEST_F(
    SharedArbitrationTest,
    arbitrationTriggeredByEnsureJoinTableFit) {
  setupMemory(kMemoryCapacity, 0);
  const int numVectors = 2;
  std::vector<RowVectorPtr> vectors;
  fuzzerOpts_.vectorSize = 10'000;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);

  std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(kMemoryCapacity);
  std::shared_ptr<core::QueryCtx> fakeCtx = newQueryCtx(kMemoryCapacity);
  auto fakePool = fakeCtx->pool()->addLeafChild(
      "fakePool", true, FakeMemoryReclaimer::create());
  std::vector<std::unique_ptr<TestAllocation>> injectAllocations;
  std::atomic<bool> injectAllocationOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::HashBuild::ensureTableFits",
      std::function<void(HashBuild*)>([&](HashBuild* buildOp) {
        // Inject the allocation once to ensure the merged table allocation will
        // trigger memory arbitration.
        if (!injectAllocationOnce.exchange(false)) {
          return;
        }
        auto* buildPool = buildOp->pool();
        // Free up available reservation from the leaf build memory pool.
        uint64_t injectAllocationSize = buildPool->availableReservation();
        injectAllocations.emplace_back(new TestAllocation{
            buildPool,
            buildPool->allocate(injectAllocationSize),
            injectAllocationSize});
        // Free up available memory from the system.
        injectAllocationSize = arbitrator_->stats().freeCapacityBytes +
            queryCtx->pool()->freeBytes();
        injectAllocations.emplace_back(new TestAllocation{
            fakePool.get(),
            fakePool->allocate(injectAllocationSize),
            injectAllocationSize});
      }));

  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::HashBuild::reclaim",
      std::function<void(Operator*)>([&](Operator* /*unused*/) {
        ASSERT_EQ(injectAllocations.size(), 2);
        for (auto& injectAllocation : injectAllocations) {
          injectAllocation->free();
        }
      }));

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .spillDirectory(spillDirectory->path)
          .config(core::QueryConfig::kSpillEnabled, "true")
          .config(core::QueryConfig::kJoinSpillEnabled, "true")
          .config(core::QueryConfig::kJoinSpillPartitionBits, "2")
          // Set multiple hash build drivers to trigger parallel build.
          .maxDrivers(4)
          .queryCtx(queryCtx)
          .plan(PlanBuilder(planNodeIdGenerator)
                    .values(vectors, true)
                    .project({"c0 AS t0", "c1 AS t1", "c2 AS t2"})
                    .hashJoin(
                        {"t0", "t1"},
                        {"u1", "u0"},
                        PlanBuilder(planNodeIdGenerator)
                            .values(vectors, true)
                            .project({"c0 AS u0", "c1 AS u1", "c2 AS u2"})
                            .planNode(),
                        "",
                        {"t1"},
                        core::JoinType::kInner)
                    .planNode())
          .assertResults(
              "SELECT t.c1 FROM tmp as t, tmp AS u WHERE t.c0 == u.c1 AND t.c1 == u.c0");
  task.reset();
  waitForAllTasksToBeDeleted();
  ASSERT_EQ(injectAllocations.size(), 2);
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, reclaimDuringJoinTableBuild) {
  setupMemory(kMemoryCapacity, 0);
  const int numVectors = 2;
  std::vector<RowVectorPtr> vectors;
  fuzzerOpts_.vectorSize = 10'000;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);

  std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(kMemoryCapacity);

  std::atomic<bool> blockTableBuildOpOnce{true};
  std::atomic<bool> tableBuildBlocked{false};
  folly::EventCount tableBuildBlockWait;
  std::atomic<bool> unblockTableBuild{false};
  folly::EventCount unblockTableBuildWait;
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::HashTable::parallelJoinBuild",
      std::function<void(MemoryPool*)>(([&](MemoryPool* pool) {
        if (!blockTableBuildOpOnce.exchange(false)) {
          return;
        }
        tableBuildBlocked = true;
        tableBuildBlockWait.notifyAll();
        unblockTableBuildWait.await([&]() { return unblockTableBuild.load(); });
        void* buffer = pool->allocate(kMemoryCapacity / 4);
        pool->free(buffer, kMemoryCapacity / 4);
      })));

  std::thread queryThread([&]() {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .spillDirectory(spillDirectory->path)
            .config(core::QueryConfig::kSpillEnabled, "true")
            .config(core::QueryConfig::kJoinSpillEnabled, "true")
            .config(core::QueryConfig::kJoinSpillPartitionBits, "2")
            // Set multiple hash build drivers to trigger parallel build.
            .maxDrivers(4)
            .queryCtx(queryCtx)
            .plan(PlanBuilder(planNodeIdGenerator)
                      .values(vectors, true)
                      .project({"c0 AS t0", "c1 AS t1", "c2 AS t2"})
                      .hashJoin(
                          {"t0", "t1"},
                          {"u1", "u0"},
                          PlanBuilder(planNodeIdGenerator)
                              .values(vectors, true)
                              .project({"c0 AS u0", "c1 AS u1", "c2 AS u2"})
                              .planNode(),
                          "",
                          {"t1"},
                          core::JoinType::kInner)
                      .planNode())
            .assertResults(
                "SELECT t.c1 FROM tmp as t, tmp AS u WHERE t.c0 == u.c1 AND t.c1 == u.c0");
  });

  tableBuildBlockWait.await([&]() { return tableBuildBlocked.load(); });

  folly::EventCount taskPauseWait;
  std::atomic<bool> taskPaused{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Task::requestPauseLocked",
      std::function<void(Task*)>(([&](Task* /*unused*/) {
        taskPaused = true;
        taskPauseWait.notifyAll();
      })));

  std::unique_ptr<TestAllocation> fakeAllocation;
  std::thread memThread([&]() {
    std::shared_ptr<core::QueryCtx> fakeCtx = newQueryCtx(kMemoryCapacity);
    auto fakePool = fakeCtx->pool()->addLeafChild("fakePool");
    const auto fakeAllocationSize = arbitrator_->stats().freeCapacityBytes +
        queryCtx->pool()->freeBytes() + 1;
    VELOX_ASSERT_THROW(
        fakePool->allocate(fakeAllocationSize), "Exceeded memory pool cap");
  });

  taskPauseWait.await([&]() { return taskPaused.load(); });

  unblockTableBuild = true;
  unblockTableBuildWait.notifyAll();

  memThread.join();
  queryThread.join();

  waitForAllTasksToBeDeleted();
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, driverInitTriggeredArbitration) {
  const int numVectors = 2;
  std::vector<RowVectorPtr> vectors;
  const int vectorSize = 100;
  fuzzerOpts_.vectorSize = vectorSize;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  const int expectedResultVectorSize = numVectors * vectorSize;
  const auto expectedVector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>(
           expectedResultVectorSize, [&](auto /*unused*/) { return 6; }),
       makeFlatVector<int64_t>(
           expectedResultVectorSize, [&](auto /*unused*/) { return 7; })});

  createDuckDbTable(vectors);
  setupMemory(kMemoryCapacity, 0);
  std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(kMemoryCapacity);
  ASSERT_EQ(queryCtx->pool()->capacity(), 0);
  ASSERT_EQ(queryCtx->pool()->maxCapacity(), kMemoryCapacity);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  AssertQueryBuilder(duckDbQueryRunner_)
      .config(core::QueryConfig::kSpillEnabled, "false")
      .queryCtx(queryCtx)
      .plan(PlanBuilder(planNodeIdGenerator, pool())
                .values(vectors)
                // Set filter projection to trigger memory allocation on
                // driver init.
                .project({"1+1+4 as t0", "1+3+3 as t1"})
                .planNode())
      .assertResults(expectedVector);
}

DEBUG_ONLY_TEST_F(
    SharedArbitrationTest,
    DISABLED_raceBetweenTaskTerminateAndReclaim) {
  setupMemory(kMemoryCapacity, 0);
  const int numVectors = 10;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);

  std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(kMemoryCapacity);
  ASSERT_EQ(queryCtx->pool()->capacity(), 0);

  // Allocate a large chunk of memory to trigger memory reclaim during the query
  // execution.
  auto fakeLeafPool = queryCtx->pool()->addLeafChild("fakeLeaf");
  const size_t fakeAllocationSize = kMemoryCapacity / 2;
  TestAllocation fakeAllocation{
      fakeLeafPool.get(),
      fakeLeafPool->allocate(fakeAllocationSize),
      fakeAllocationSize};

  // Set test injection to enforce memory arbitration based on the fake
  // allocation size and the total available memory.
  std::shared_ptr<Task> task;
  std::atomic<bool> injectAllocationOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Values::getOutput",
      std::function<void(const exec::Values*)>([&](const exec::Values* values) {
        if (!injectAllocationOnce.exchange(false)) {
          return;
        }
        task = values->testingOperatorCtx()->task();
        MemoryPool* pool = values->pool();
        VELOX_ASSERT_THROW(
            pool->allocate(kMemoryCapacity * 2 / 3),
            "Exceeded memory pool cap");
      }));

  // Set test injection to wait until the reclaim on hash aggregation operator
  // triggers.
  folly::EventCount opReclaimStartWait;
  std::atomic<bool> opReclaimStarted{false};
  folly::EventCount taskAbortWait;
  std::atomic<bool> taskAborted{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Operator::MemoryReclaimer::reclaim",
      std::function<void(MemoryPool*)>(([&](MemoryPool* pool) {
        const std::string re(".*Aggregation");
        if (!RE2::FullMatch(pool->name(), re)) {
          return;
        }
        opReclaimStarted = true;
        opReclaimStartWait.notifyAll();
        // Wait for task abort to happen before the actual memory reclaim.
        taskAbortWait.await([&]() { return taskAborted.load(); });
      })));

  const int numDrivers = 1;
  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  std::thread queryThread([&]() {
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(duckDbQueryRunner_)
            .queryCtx(queryCtx)
            .spillDirectory(spillDirectory->path)
            .config(core::QueryConfig::kSpillEnabled, "true")
            .config(core::QueryConfig::kJoinSpillEnabled, "true")
            .config(core::QueryConfig::kJoinSpillPartitionBits, "2")
            .maxDrivers(numDrivers)
            .plan(PlanBuilder()
                      .values(vectors)
                      .localPartition({"c0", "c1"})
                      .singleAggregation({"c0", "c1"}, {"array_agg(c2)"})
                      .localPartition(std::vector<std::string>{})
                      .planNode())
            .assertResults(
                "SELECT c0, c1, array_agg(c2) FROM tmp GROUP BY c0, c1"),
        "Aborted for external error");
  });

  // Wait for the reclaim on aggregation to be started before the task abort.
  opReclaimStartWait.await([&]() { return opReclaimStarted.load(); });
  ASSERT_TRUE(task != nullptr);
  task->requestAbort().wait();

  // Resume aggregation reclaim to execute.
  taskAborted = true;
  taskAbortWait.notifyAll();

  queryThread.join();
  fakeAllocation.free();
  task.reset();
  waitForAllTasksToBeDeleted();
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, raceBetweenMaybeReserveAndTaskAbort) {
  setupMemory(kMemoryCapacity, 0);
  const int numVectors = 10;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);

  auto queryCtx = newQueryCtx(kMemoryCapacity);
  ASSERT_EQ(queryCtx->pool()->capacity(), 0);

  // Create a fake query to hold some memory to trigger memory arbitration.
  auto fakeQueryCtx = newQueryCtx(kMemoryCapacity);
  auto fakeLeafPool = fakeQueryCtx->pool()->addLeafChild(
      "fakeLeaf", true, FakeMemoryReclaimer::create());
  TestAllocation fakeAllocation{
      fakeLeafPool.get(),
      fakeLeafPool->allocate(kMemoryCapacity / 3),
      kMemoryCapacity / 3};

  std::unique_ptr<TestAllocation> injectAllocation;
  std::atomic<bool> injectAllocationOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::common::memory::MemoryPoolImpl::maybeReserve",
      std::function<void(memory::MemoryPool*)>([&](memory::MemoryPool* pool) {
        if (!injectAllocationOnce.exchange(false)) {
          return;
        }
        // The injection memory allocation (with the given size) makes sure that
        // maybeReserve fails and abort this query itself.
        const size_t injectAllocationSize =
            pool->freeBytes() + arbitrator_->stats().freeCapacityBytes;
        injectAllocation.reset(new TestAllocation{
            fakeLeafPool.get(),
            fakeLeafPool->allocate(injectAllocationSize),
            injectAllocationSize});
      }));

  const int numDrivers = 1;
  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  std::thread queryThread([&]() {
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(duckDbQueryRunner_)
            .queryCtx(queryCtx)
            .spillDirectory(spillDirectory->path)
            .config(core::QueryConfig::kSpillEnabled, "true")
            .config(core::QueryConfig::kJoinSpillEnabled, "true")
            .config(core::QueryConfig::kJoinSpillPartitionBits, "2")
            .maxDrivers(numDrivers)
            .plan(PlanBuilder()
                      .values(vectors)
                      .localPartition({"c0", "c1"})
                      .singleAggregation({"c0", "c1"}, {"array_agg(c2)"})
                      .localPartition(std::vector<std::string>{})
                      .planNode())
            .copyResults(pool()),
        "Exceeded memory pool cap");
  });

  queryThread.join();
  fakeAllocation.free();
  injectAllocation->free();
  waitForAllTasksToBeDeleted();
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, asyncArbitratonFromNonDriverContext) {
  setupMemory(kMemoryCapacity, 0);
  const int numVectors = 10;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);
  std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(kMemoryCapacity);
  ASSERT_EQ(queryCtx->pool()->capacity(), 0);

  folly::EventCount aggregationAllocationWait;
  std::atomic<bool> aggregationAllocationOnce{true};
  folly::EventCount aggregationAllocationUnblockWait;
  std::atomic<bool> aggregationAllocationUnblocked{false};
  std::atomic<MemoryPool*> injectPool{nullptr};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::memory::MemoryPoolImpl::reserveThreadSafe",
      std::function<void(MemoryPool*)>(([&](MemoryPool* pool) {
        const std::string re(".*Aggregation");
        if (!RE2::FullMatch(pool->name(), re)) {
          return;
        }

        if (!aggregationAllocationOnce.exchange(false)) {
          return;
        }
        injectPool = pool;
        aggregationAllocationWait.notifyAll();

        aggregationAllocationUnblockWait.await(
            [&]() { return aggregationAllocationUnblocked.load(); });
      })));

  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  std::shared_ptr<Task> task;
  std::thread queryThread([&]() {
    task = AssertQueryBuilder(duckDbQueryRunner_)
               .queryCtx(queryCtx)
               .spillDirectory(spillDirectory->path)
               .config(core::QueryConfig::kSpillEnabled, "true")
               .config(core::QueryConfig::kJoinSpillEnabled, "true")
               .config(core::QueryConfig::kJoinSpillPartitionBits, "2")
               .plan(PlanBuilder()
                         .values(vectors)
                         .localPartition({"c0", "c1"})
                         .singleAggregation({"c0", "c1"}, {"array_agg(c2)"})
                         .localPartition(std::vector<std::string>{})
                         .planNode())
               .assertResults(
                   "SELECT c0, c1, array_agg(c2) FROM tmp GROUP BY c0, c1");
  });

  aggregationAllocationWait.await(
      [&]() { return !aggregationAllocationOnce.load(); });
  ASSERT_TRUE(injectPool != nullptr);

  // Trigger the memory arbitration with memory pool whose associated driver is
  // running on driver thread.
  const size_t fakeAllocationSize = arbitrator_->stats().freeCapacityBytes / 2;
  TestAllocation fakeAllocation = {
      injectPool.load(),
      injectPool.load()->allocate(fakeAllocationSize),
      fakeAllocationSize};

  aggregationAllocationUnblocked = true;
  aggregationAllocationUnblockWait.notifyAll();

  queryThread.join();
  fakeAllocation.free();

  task.reset();
  waitForAllTasksToBeDeleted();
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, tableWriteSpillUseMoreMemory) {
  const uint64_t memoryCapacity = 256 * MB;
  setupMemory(memoryCapacity);
  // Create a large number of vectors to trigger writer spill.
  fuzzerOpts_.vectorSize = 1000;
  fuzzerOpts_.stringLength = 2048;
  fuzzerOpts_.stringVariableLength = false;
  VectorFuzzer fuzzer(fuzzerOpts_, pool());
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 10; ++i) {
    vectors.push_back(fuzzer.fuzzInputRow(rowType_));
  }

  std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(memoryCapacity / 8);
  std::shared_ptr<core::QueryCtx> fakeQueryCtx = newQueryCtx(memoryCapacity);
  auto fakePool = fakeQueryCtx->pool()->addLeafChild(
      "fakePool", true, FakeMemoryReclaimer::create());
  TestAllocation injectedFakeAllocation{
      fakePool.get(),
      fakePool->allocate(memoryCapacity * 3 / 4),
      memoryCapacity * 3 / 4};

  void* allocatedBuffer;
  TestAllocation injectedWriterAllocation;
  SCOPED_TESTVALUE_SET(
      "facebook::velox::dwrf::Writer::flushInternal",
      std::function<void(dwrf::Writer*)>(([&](dwrf::Writer* writer) {
        ASSERT_TRUE(underMemoryArbitration());
        injectedFakeAllocation.free();
        auto& pool = writer->getContext().getMemoryPool(
            dwrf::MemoryUsageCategory::GENERAL);
        injectedWriterAllocation.pool = &pool;
        injectedWriterAllocation.size = memoryCapacity / 8;
        injectedWriterAllocation.buffer =
            pool.allocate(injectedWriterAllocation.size);
      })));

  // Free the extra fake memory allocations to make memory pool state consistent
  // at the end of test.
  std::atomic<bool> clearAllocationOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Task::setError",
      std::function<void(Task*)>(([&](Task* task) {
        if (!clearAllocationOnce.exchange(false)) {
          return;
        }
        ASSERT_EQ(injectedWriterAllocation.size, memoryCapacity / 8);
        injectedWriterAllocation.free();
      })));

  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  const auto outputDirectory = TempDirectoryPath::create();
  auto writerPlan = PlanBuilder()
                        .values(vectors)
                        .tableWrite(outputDirectory->path)
                        .planNode();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(duckDbQueryRunner_)
          .queryCtx(queryCtx)
          .maxDrivers(1)
          .spillDirectory(spillDirectory->path)
          .config(core::QueryConfig::kSpillEnabled, "true")
          .config(core::QueryConfig::kWriterSpillEnabled, "true")
          // Set 0 file writer flush threshold to always trigger flush in test.
          .config(core::QueryConfig::kWriterFlushThresholdBytes, "0")
          // Set stripe size to extreme large to avoid writer internal triggered
          // flush.
          .connectorConfig(
              kHiveConnectorId,
              connector::hive::HiveConfig::kOrcWriterMaxStripeSize,
              "1GB")
          .connectorConfig(
              kHiveConnectorId,
              connector::hive::HiveConfig::kOrcWriterMaxDictionaryMemory,
              "1GB")
          .plan(std::move(writerPlan))
          .copyResults(pool()),
      "Unexpected memory growth after memory reclaim");

  waitForAllTasksToBeDeleted();
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, tableWriteReclaimOnClose) {
  const uint64_t memoryCapacity = 512 * MB;
  setupMemory(memoryCapacity);
  // Create a large number of vectors to trigger writer spill.
  fuzzerOpts_.vectorSize = 1000;
  fuzzerOpts_.stringLength = 1024;
  fuzzerOpts_.stringVariableLength = false;
  VectorFuzzer fuzzer(fuzzerOpts_, pool());
  std::vector<RowVectorPtr> vectors;
  int numRows{0};
  for (int i = 0; i < 10; ++i) {
    vectors.push_back(fuzzer.fuzzInputRow(rowType_));
    numRows += vectors.back()->size();
  }

  std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(memoryCapacity);
  std::shared_ptr<core::QueryCtx> fakeQueryCtx = newQueryCtx(memoryCapacity);
  auto fakePool = fakeQueryCtx->pool()->addLeafChild(
      "fakePool", true, FakeMemoryReclaimer::create());

  std::atomic<bool> writerNoMoreInput{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal::noMoreInput",
      std::function<void(Operator*)>(([&](Operator* op) {
        if (op->operatorType() != "TableWrite") {
          return;
        }
        writerNoMoreInput = true;
      })));

  std::atomic<bool> maybeReserveInjectOnce{true};
  TestAllocation fakeAllocation;
  SCOPED_TESTVALUE_SET(
      "facebook::velox::common::memory::MemoryPoolImpl::maybeReserve",
      std::function<void(memory::MemoryPool*)>([&](memory::MemoryPool* pool) {
        if (!writerNoMoreInput) {
          return;
        }
        if (!maybeReserveInjectOnce.exchange(false)) {
          return;
        }
        // The injection memory allocation to cause maybeReserve on writer close
        // to trigger memory arbitration. The latter tries to reclaim memory
        // from this file writer.
        const size_t injectAllocationSize =
            pool->freeBytes() + arbitrator_->stats().freeCapacityBytes;
        fakeAllocation = TestAllocation{
            .pool = fakePool.get(),
            .buffer = fakePool->allocate(injectAllocationSize),
            .size = injectAllocationSize};
      }));

  SCOPED_TESTVALUE_SET(
      "facebook::velox::dwrf::Writer::flushStripe",
      std::function<void(dwrf::Writer*)>(
          [&](dwrf::Writer* writer) { fakeAllocation.free(); }));

  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  const auto outputDirectory = TempDirectoryPath::create();
  auto writerPlan =
      PlanBuilder()
          .values(vectors)
          .tableWrite(outputDirectory->path)
          .singleAggregation(
              {},
              {fmt::format("sum({})", TableWriteTraits::rowCountColumnName())})
          .planNode();

  AssertQueryBuilder(duckDbQueryRunner_)
      .queryCtx(queryCtx)
      .maxDrivers(1)
      .spillDirectory(spillDirectory->path)
      .config(core::QueryConfig::kSpillEnabled, "true")
      .config(core::QueryConfig::kWriterSpillEnabled, "true")
      // Set 0 file writer flush threshold to always trigger flush in test.
      .config(core::QueryConfig::kWriterFlushThresholdBytes, "0")
      // Set stripe size to extreme large to avoid writer internal triggered
      // flush.
      .connectorConfig(
          kHiveConnectorId,
          connector::hive::HiveConfig::kOrcWriterMaxStripeSize,
          "1GB")
      .connectorConfig(
          kHiveConnectorId,
          connector::hive::HiveConfig::kOrcWriterMaxDictionaryMemory,
          "1GB")
      .plan(std::move(writerPlan))
      .assertResults(fmt::format("SELECT {}", numRows));

  waitForAllTasksToBeDeleted();
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, tableFileWriteError) {
  const uint64_t memoryCapacity = 32 * MB;
  setupMemory(memoryCapacity);
  fuzzerOpts_.vectorSize = 1000;
  fuzzerOpts_.stringLength = 1024;
  fuzzerOpts_.stringVariableLength = false;
  VectorFuzzer fuzzer(fuzzerOpts_, pool());
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 10; ++i) {
    vectors.push_back(fuzzer.fuzzInputRow(rowType_));
  }

  std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(memoryCapacity);

  std::atomic<bool> injectWriterErrorOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::dwrf::Writer::write",
      std::function<void(dwrf::Writer*)>(([&](dwrf::Writer* writer) {
        auto& context = writer->getContext();
        auto& pool =
            context.getMemoryPool(dwrf::MemoryUsageCategory::OUTPUT_STREAM);
        if (static_cast<MemoryPoolImpl*>(&pool)->testingMinReservationBytes() ==
            0) {
          return;
        }
        if (!injectWriterErrorOnce.exchange(false)) {
          return;
        }
        VELOX_FAIL("inject writer error");
      })));

  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  const auto outputDirectory = TempDirectoryPath::create();
  auto writerPlan = PlanBuilder()
                        .values(vectors)
                        .tableWrite(outputDirectory->path)
                        .planNode();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(duckDbQueryRunner_)
          .queryCtx(queryCtx)
          .maxDrivers(1)
          .spillDirectory(spillDirectory->path)
          .config(core::QueryConfig::kSpillEnabled, "true")
          .config(core::QueryConfig::kWriterSpillEnabled, "true")
          // Set 0 file writer flush threshold to always reclaim memory from
          // file writer.
          .config(core::QueryConfig::kWriterFlushThresholdBytes, "0")
          // Set stripe size to extreme large to avoid writer internal triggered
          // flush.
          .connectorConfig(
              kHiveConnectorId,
              connector::hive::HiveConfig::kOrcWriterMaxStripeSize,
              "1GB")
          .connectorConfig(
              kHiveConnectorId,
              connector::hive::HiveConfig::kOrcWriterMaxDictionaryMemory,
              "1GB")
          .plan(std::move(writerPlan))
          .copyResults(pool()),
      "inject writer error");

  waitForAllTasksToBeDeleted();
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, runtimeStats) {
  const uint64_t memoryCapacity = 128 * MB;
  setupMemory(memoryCapacity);
  fuzzerOpts_.vectorSize = 1000;
  fuzzerOpts_.stringLength = 1024;
  fuzzerOpts_.stringVariableLength = false;
  VectorFuzzer fuzzer(fuzzerOpts_, pool());
  std::vector<RowVectorPtr> vectors;
  int numRows{0};
  for (int i = 0; i < 10; ++i) {
    vectors.push_back(fuzzer.fuzzInputRow(rowType_));
    numRows += vectors.back()->size();
  }

  std::atomic<int> outputCount{0};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Values::getOutput",
      std::function<void(const facebook::velox::exec::Values*)>(
          ([&](const facebook::velox::exec::Values* values) {
            if (outputCount++ != 5) {
              return;
            }
            const auto fakeAllocationSize =
                arbitrator_->stats().maxCapacityBytes -
                values->pool()->capacity() + 1;
            void* buffer = values->pool()->allocate(fakeAllocationSize);
            values->pool()->free(buffer, fakeAllocationSize);
          })));

  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  const auto outputDirectory = TempDirectoryPath::create();
  const auto queryCtx = newQueryCtx(memoryCapacity);
  auto writerPlan =
      PlanBuilder()
          .values(vectors)
          .tableWrite(outputDirectory->path)
          .singleAggregation(
              {},
              {fmt::format("sum({})", TableWriteTraits::rowCountColumnName())})
          .planNode();
  {
    const std::shared_ptr<Task> task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .queryCtx(queryCtx)
            .maxDrivers(1)
            .spillDirectory(spillDirectory->path)
            .config(core::QueryConfig::kSpillEnabled, "true")
            .config(core::QueryConfig::kWriterSpillEnabled, "true")
            // Set 0 file writer flush threshold to always trigger flush in
            // test.
            .config(core::QueryConfig::kWriterFlushThresholdBytes, "0")
            // Set stripe size to extreme large to avoid writer internal
            // triggered flush.
            .connectorConfig(
                kHiveConnectorId,
                connector::hive::HiveConfig::kOrcWriterMaxStripeSize,
                "1GB")
            .connectorConfig(
                kHiveConnectorId,
                connector::hive::HiveConfig::kOrcWriterMaxDictionaryMemory,
                "1GB")
            .plan(std::move(writerPlan))
            .assertResults(fmt::format("SELECT {}", numRows));

    auto stats = task->taskStats().pipelineStats.front().operatorStats;
    // TableWrite Operator's stripeSize runtime stats would be updated twice:
    // - Values Operator's memory allocation triggers TableWrite's memory
    // reclaim, which triggers data flush.
    // - TableWrite Operator's close would trigger flush.
    ASSERT_EQ(stats[1].runtimeStats["stripeSize"].count, 2);
    // Values Operator won't be set stripeSize in its runtimeStats.
    ASSERT_EQ(stats[0].runtimeStats["stripeSize"].count, 0);
  }
  waitForAllTasksToBeDeleted();
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, reclaimFromTableWriter) {
  VectorFuzzer::Options options;
  const int batchSize = 1'000;
  options.vectorSize = batchSize;
  options.stringVariableLength = false;
  options.stringLength = 1'000;
  VectorFuzzer fuzzer(options, pool());
  const int numBatches = 20;
  std::vector<RowVectorPtr> vectors;
  int numRows{0};
  for (int i = 0; i < numBatches; ++i) {
    numRows += batchSize;
    vectors.push_back(fuzzer.fuzzRow(rowType_));
  }
  createDuckDbTable(vectors);

  for (bool writerSpillEnabled : {false, true}) {
    {
      SCOPED_TRACE(fmt::format("writerSpillEnabled: {}", writerSpillEnabled));

      setupMemory(kMemoryCapacity, 0);

      std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(kMemoryCapacity);
      ASSERT_EQ(queryCtx->pool()->capacity(), 0);

      std::atomic<int> numInputs{0};
      SCOPED_TESTVALUE_SET(
          "facebook::velox::exec::Driver::runInternal::addInput",
          std::function<void(Operator*)>(([&](Operator* op) {
            if (op->operatorType() != "TableWrite") {
              return;
            }
            // We reclaim memory from table writer connector memory pool which
            // connects to the memory pools inside the hive connector.
            ASSERT_FALSE(op->canReclaim());
            if (++numInputs != numBatches) {
              return;
            }

            const auto fakeAllocationSize =
                arbitrator_->stats().maxCapacityBytes -
                op->pool()->parent()->reservedBytes();
            if (writerSpillEnabled) {
              auto* buffer = op->pool()->allocate(fakeAllocationSize);
              op->pool()->free(buffer, fakeAllocationSize);
            } else {
              VELOX_ASSERT_THROW(
                  op->pool()->allocate(fakeAllocationSize),
                  "Exceeded memory pool");
            }
          })));

      auto spillDirectory = exec::test::TempDirectoryPath::create();
      auto outputDirectory = TempDirectoryPath::create();
      auto writerPlan =
          PlanBuilder()
              .values(vectors)
              .tableWrite(outputDirectory->path)
              .project({TableWriteTraits::rowCountColumnName()})
              .singleAggregation(
                  {},
                  {fmt::format(
                      "sum({})", TableWriteTraits::rowCountColumnName())})
              .planNode();

      AssertQueryBuilder(duckDbQueryRunner_)
          .queryCtx(queryCtx)
          .maxDrivers(1)
          .spillDirectory(spillDirectory->path)
          .config(
              core::QueryConfig::kSpillEnabled,
              writerSpillEnabled ? "true" : "false")
          .config(
              core::QueryConfig::kWriterSpillEnabled,
              writerSpillEnabled ? "true" : "false")
          // Set 0 file writer flush threshold to always trigger flush in test.
          .config(core::QueryConfig::kWriterFlushThresholdBytes, "0")
          .plan(std::move(writerPlan))
          .assertResults(fmt::format("SELECT {}", numRows));

      ASSERT_EQ(arbitrator_->stats().numFailures, writerSpillEnabled ? 0 : 1);
      ASSERT_EQ(arbitrator_->stats().numNonReclaimableAttempts, 0);
      waitForAllTasksToBeDeleted(3'000'000);
    }
    ASSERT_EQ(arbitrator_->stats().numReserves, numAddedPools_);
    ASSERT_EQ(arbitrator_->stats().numReleases, numAddedPools_);
  }
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, reclaimFromSortTableWriter) {
  VectorFuzzer::Options options;
  const int batchSize = 1'000;
  options.vectorSize = batchSize;
  options.stringVariableLength = false;
  options.stringLength = 1'000;
  VectorFuzzer fuzzer(options, pool());
  const int numBatches = 20;
  std::vector<RowVectorPtr> vectors;
  int numRows{0};
  const auto partitionKeyVector = makeFlatVector<int32_t>(
      batchSize, [&](vector_size_t /*unused*/) { return 0; });
  for (int i = 0; i < numBatches; ++i) {
    numRows += batchSize;
    vectors.push_back(fuzzer.fuzzInputRow(rowType_));
    vectors.back()->childAt(0) = partitionKeyVector;
  }
  createDuckDbTable(vectors);

  for (bool writerSpillEnabled : {false, true}) {
    {
      SCOPED_TRACE(fmt::format("writerSpillEnabled: {}", writerSpillEnabled));

      setupMemory(kMemoryCapacity, 0);

      std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(kMemoryCapacity);
      ASSERT_EQ(queryCtx->pool()->capacity(), 0);

      const auto spillStats = common::globalSpillStats();

      std::atomic<int> numInputs{0};
      SCOPED_TESTVALUE_SET(
          "facebook::velox::exec::Driver::runInternal::addInput",
          std::function<void(Operator*)>(([&](Operator* op) {
            if (op->operatorType() != "TableWrite") {
              return;
            }
            // We reclaim memory from table writer connector memory pool which
            // connects to the memory pools inside the hive connector.
            ASSERT_FALSE(op->canReclaim());
            if (++numInputs != numBatches) {
              return;
            }

            const auto fakeAllocationSize =
                arbitrator_->stats().maxCapacityBytes -
                op->pool()->parent()->reservedBytes();
            if (writerSpillEnabled) {
              auto* buffer = op->pool()->allocate(fakeAllocationSize);
              op->pool()->free(buffer, fakeAllocationSize);
            } else {
              VELOX_ASSERT_THROW(
                  op->pool()->allocate(fakeAllocationSize),
                  "Exceeded memory pool");
            }
          })));

      auto spillDirectory = exec::test::TempDirectoryPath::create();
      auto outputDirectory = TempDirectoryPath::create();
      auto writerPlan =
          PlanBuilder()
              .values(vectors)
              .tableWrite(outputDirectory->path, {"c0"}, 4, {"c1"}, {"c2"})
              .project({TableWriteTraits::rowCountColumnName()})
              .singleAggregation(
                  {},
                  {fmt::format(
                      "sum({})", TableWriteTraits::rowCountColumnName())})
              .planNode();

      AssertQueryBuilder(duckDbQueryRunner_)
          .queryCtx(queryCtx)
          .maxDrivers(1)
          .spillDirectory(spillDirectory->path)
          .config(
              core::QueryConfig::kSpillEnabled,
              writerSpillEnabled ? "true" : "false")
          .config(
              core::QueryConfig::kWriterSpillEnabled,
              writerSpillEnabled ? "true" : "false")
          // Set 0 file writer flush threshold to always trigger flush in test.
          .config(core::QueryConfig::kWriterFlushThresholdBytes, "0")
          .plan(std::move(writerPlan))
          .assertResults(fmt::format("SELECT {}", numRows));

      ASSERT_EQ(arbitrator_->stats().numFailures, writerSpillEnabled ? 0 : 1);
      ASSERT_EQ(arbitrator_->stats().numNonReclaimableAttempts, 0);
      waitForAllTasksToBeDeleted(3'000'000);
      const auto updatedSpillStats = common::globalSpillStats();
      if (writerSpillEnabled) {
        ASSERT_GT(updatedSpillStats.spilledBytes, spillStats.spilledBytes);
        ASSERT_GT(
            updatedSpillStats.spilledPartitions, spillStats.spilledPartitions);
      } else {
        ASSERT_EQ(updatedSpillStats, spillStats);
      }
    }
    ASSERT_EQ(arbitrator_->stats().numReserves, numAddedPools_);
    ASSERT_EQ(arbitrator_->stats().numReleases, numAddedPools_);
  }
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, writerFlushThreshold) {
  VectorFuzzer::Options options;
  const int batchSize = 1'000;
  options.vectorSize = batchSize;
  options.stringVariableLength = false;
  options.stringLength = 1'000;
  VectorFuzzer fuzzer(options, pool());
  const int numBatches = 20;
  std::vector<RowVectorPtr> vectors;
  int numRows{0};
  for (int i = 0; i < numBatches; ++i) {
    numRows += batchSize;
    vectors.push_back(fuzzer.fuzzRow(rowType_));
  }
  createDuckDbTable(vectors);

  const std::vector<uint64_t> writerFlushThresholds{0, 1UL << 30};
  for (uint64_t writerFlushThreshold : writerFlushThresholds) {
    {
      SCOPED_TRACE(fmt::format(
          "writerFlushThreshold: {}", succinctBytes(writerFlushThreshold)));

      setupMemory(kMemoryCapacity, 0);

      std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(kMemoryCapacity);
      ASSERT_EQ(queryCtx->pool()->capacity(), 0);

      std::atomic<int> numInputs{0};
      SCOPED_TESTVALUE_SET(
          "facebook::velox::exec::Driver::runInternal::addInput",
          std::function<void(Operator*)>(([&](Operator* op) {
            if (op->operatorType() != "TableWrite") {
              return;
            }
            if (++numInputs != numBatches) {
              return;
            }

            const auto fakeAllocationSize =
                arbitrator_->stats().maxCapacityBytes -
                op->pool()->parent()->reservedBytes();
            if (writerFlushThreshold == 0) {
              auto* buffer = op->pool()->allocate(fakeAllocationSize);
              op->pool()->free(buffer, fakeAllocationSize);
            } else {
              // The injected memory allocation fail if we set very high memory
              // flush threshold.
              VELOX_ASSERT_THROW(
                  op->pool()->allocate(fakeAllocationSize),
                  "Exceeded memory pool");
            }
          })));

      auto spillDirectory = exec::test::TempDirectoryPath::create();
      auto outputDirectory = TempDirectoryPath::create();
      auto writerPlan =
          PlanBuilder()
              .values(vectors)
              .tableWrite(outputDirectory->path)
              .project({TableWriteTraits::rowCountColumnName()})
              .singleAggregation(
                  {},
                  {fmt::format(
                      "sum({})", TableWriteTraits::rowCountColumnName())})
              .planNode();

      AssertQueryBuilder(duckDbQueryRunner_)
          .queryCtx(queryCtx)
          .maxDrivers(1)
          .spillDirectory(spillDirectory->path)
          .config(core::QueryConfig::kSpillEnabled, "true")
          .config(core::QueryConfig::kWriterSpillEnabled, "true")
          .config(
              core::QueryConfig::kWriterFlushThresholdBytes,
              folly::to<std::string>(writerFlushThreshold))
          .plan(std::move(writerPlan))
          .assertResults(fmt::format("SELECT {}", numRows));

      ASSERT_EQ(
          arbitrator_->stats().numFailures, writerFlushThreshold == 0 ? 0 : 1);
      ASSERT_EQ(
          arbitrator_->stats().numNonReclaimableAttempts,
          writerFlushThreshold == 0 ? 0 : 1);
      waitForAllTasksToBeDeleted(3'000'000);
    }
    ASSERT_EQ(arbitrator_->stats().numReserves, numAddedPools_);
    ASSERT_EQ(arbitrator_->stats().numReleases, numAddedPools_);
  }
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, reclaimFromNonReclaimableTableWriter) {
  VectorFuzzer::Options options;
  const int batchSize = 1'000;
  options.vectorSize = batchSize;
  options.stringVariableLength = false;
  options.stringLength = 1'000;
  VectorFuzzer fuzzer(options, pool());
  const int numBatches = 20;
  std::vector<RowVectorPtr> vectors;
  int numRows{0};
  for (int i = 0; i < numBatches; ++i) {
    numRows += batchSize;
    vectors.push_back(fuzzer.fuzzRow(rowType_));
  }

  createDuckDbTable(vectors);

  setupMemory(kMemoryCapacity, 0);

  std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(kMemoryCapacity);
  ASSERT_EQ(queryCtx->pool()->capacity(), 0);

  std::atomic<bool> injectFakeAllocationOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::dwrf::Writer::write",
      std::function<void(dwrf::Writer*)>(([&](dwrf::Writer* writer) {
        if (!injectFakeAllocationOnce.exchange(false)) {
          return;
        }
        auto& pool = writer->getContext().getMemoryPool(
            dwrf::MemoryUsageCategory::GENERAL);
        const auto fakeAllocationSize =
            arbitrator_->stats().maxCapacityBytes - pool.reservedBytes();
        VELOX_ASSERT_THROW(
            pool.allocate(fakeAllocationSize), "Exceeded memory pool");
      })));

  auto outputDirectory = TempDirectoryPath::create();
  auto writerPlan =
      PlanBuilder()
          .values(vectors)
          .tableWrite(outputDirectory->path)
          .project({TableWriteTraits::rowCountColumnName()})
          .singleAggregation(
              {},
              {fmt::format("sum({})", TableWriteTraits::rowCountColumnName())})
          .planNode();

  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  AssertQueryBuilder(duckDbQueryRunner_)
      .queryCtx(queryCtx)
      .maxDrivers(1)
      .spillDirectory(spillDirectory->path)
      .config(core::QueryConfig::kSpillEnabled, "true")
      .config(core::QueryConfig::kWriterSpillEnabled, "true")
      // Set file writer flush threshold of zero to always trigger flush in
      // test.
      .config(core::QueryConfig::kWriterFlushThresholdBytes, "0")
      // Set large stripe and dictionary size thresholds to avoid writer
      // internal stripe flush.
      .connectorConfig(
          kHiveConnectorId,
          connector::hive::HiveConfig::kOrcWriterMaxStripeSize,
          "1GB")
      .connectorConfig(
          kHiveConnectorId,
          connector::hive::HiveConfig::kOrcWriterMaxDictionaryMemory,
          "1GB")
      .plan(std::move(writerPlan))
      .assertResults(fmt::format("SELECT {}", numRows));

  ASSERT_EQ(arbitrator_->stats().numFailures, 1);
  ASSERT_EQ(arbitrator_->stats().numNonReclaimableAttempts, 1);
  ASSERT_EQ(arbitrator_->stats().numReserves, numAddedPools_);
}

DEBUG_ONLY_TEST_F(
    SharedArbitrationTest,
    arbitrationFromTableWriterWithNoMoreInput) {
  VectorFuzzer::Options options;
  const int batchSize = 1'000;
  options.vectorSize = batchSize;
  options.stringVariableLength = false;
  options.stringLength = 1'000;
  VectorFuzzer fuzzer(options, pool());
  const int numBatches = 10;
  std::vector<RowVectorPtr> vectors;
  int numRows{0};
  for (int i = 0; i < numBatches; ++i) {
    numRows += batchSize;
    vectors.push_back(fuzzer.fuzzRow(rowType_));
  }

  createDuckDbTable(vectors);
  setupMemory(kMemoryCapacity, 0);

  std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(kMemoryCapacity);
  ASSERT_EQ(queryCtx->pool()->capacity(), 0);

  std::atomic<bool> writerNoMoreInput{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal::noMoreInput",
      std::function<void(Operator*)>(([&](Operator* op) {
        if (op->operatorType() != "TableWrite") {
          return;
        }
        writerNoMoreInput = true;
      })));

  std::atomic<bool> injectGetOutputOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal::getOutput",
      std::function<void(Operator*)>(([&](Operator* op) {
        if (op->operatorType() != "TableWrite") {
          return;
        }
        if (!writerNoMoreInput) {
          return;
        }
        if (!injectGetOutputOnce.exchange(false)) {
          return;
        }
        const auto fakeAllocationSize = arbitrator_->stats().maxCapacityBytes -
            op->pool()->parent()->reservedBytes();
        auto* buffer = op->pool()->allocate(fakeAllocationSize);
        op->pool()->free(buffer, fakeAllocationSize);
      })));

  auto outputDirectory = TempDirectoryPath::create();
  auto writerPlan =
      PlanBuilder()
          .values(vectors)
          .tableWrite(outputDirectory->path)
          .project({TableWriteTraits::rowCountColumnName()})
          .singleAggregation(
              {},
              {fmt::format("sum({})", TableWriteTraits::rowCountColumnName())})
          .planNode();

  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  AssertQueryBuilder(duckDbQueryRunner_)
      .queryCtx(queryCtx)
      .maxDrivers(1)
      .spillDirectory(spillDirectory->path)
      .config(core::QueryConfig::kSpillEnabled, "true")
      .config(core::QueryConfig::kWriterSpillEnabled, "true")
      // Set 0 file writer flush threshold to always trigger flush in test.
      .config(core::QueryConfig::kWriterFlushThresholdBytes, "0")
      // Set large stripe and dictionary size thresholds to avoid writer
      // internal stripe flush.
      .connectorConfig(
          kHiveConnectorId,
          connector::hive::HiveConfig::kOrcWriterMaxStripeSize,
          "1GB")
      .connectorConfig(
          kHiveConnectorId,
          connector::hive::HiveConfig::kOrcWriterMaxDictionaryMemory,
          "1GB")
      .plan(std::move(writerPlan))
      .assertResults(fmt::format("SELECT {}", numRows));

  ASSERT_EQ(arbitrator_->stats().numNonReclaimableAttempts, 0);
  ASSERT_EQ(arbitrator_->stats().numFailures, 0);
  ASSERT_GT(arbitrator_->stats().numReclaimedBytes, 0);
  ASSERT_EQ(arbitrator_->stats().numReserves, numAddedPools_);
}

DEBUG_ONLY_TEST_F(
    SharedArbitrationTest,
    reclaimFromNonReclaimableSortTableWriter) {
  VectorFuzzer::Options options;
  const int batchSize = 1'000;
  options.vectorSize = batchSize;
  options.stringVariableLength = false;
  options.stringLength = 1'000;
  VectorFuzzer fuzzer(options, pool());
  const int numBatches = 20;
  std::vector<RowVectorPtr> vectors;
  int numRows{0};
  const auto partitionKeyVector = makeFlatVector<int32_t>(
      batchSize, [&](vector_size_t /*unused*/) { return 0; });
  for (int i = 0; i < numBatches; ++i) {
    numRows += batchSize;
    vectors.push_back(fuzzer.fuzzInputRow(rowType_));
    vectors.back()->childAt(0) = partitionKeyVector;
  }

  createDuckDbTable(vectors);

  setupMemory(kMemoryCapacity, 0);

  std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(kMemoryCapacity);
  ASSERT_EQ(queryCtx->pool()->capacity(), 0);

  std::atomic<bool> injectFakeAllocationOnce{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::memory::MemoryPoolImpl::reserveThreadSafe",
      std::function<void(MemoryPool*)>(([&](MemoryPool* pool) {
        const std::string re(".*sort");
        if (!RE2::FullMatch(pool->name(), re)) {
          return;
        }
        const int writerMemoryUsage = 4L << 20;
        if (pool->parent()->reservedBytes() < writerMemoryUsage) {
          return;
        }
        if (!injectFakeAllocationOnce.exchange(false)) {
          return;
        }
        const auto fakeAllocationSize = arbitrator_->stats().maxCapacityBytes -
            pool->parent()->reservedBytes();
        VELOX_ASSERT_THROW(
            pool->allocate(fakeAllocationSize), "Exceeded memory pool");
      })));

  auto outputDirectory = TempDirectoryPath::create();
  auto writerPlan =
      PlanBuilder()
          .values(vectors)
          .tableWrite(outputDirectory->path, {"c0"}, 4, {"c1"}, {"c2"})
          .project({TableWriteTraits::rowCountColumnName()})
          .singleAggregation(
              {},
              {fmt::format("sum({})", TableWriteTraits::rowCountColumnName())})
          .planNode();

  const auto spillStats = common::globalSpillStats();
  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  AssertQueryBuilder(duckDbQueryRunner_)
      .queryCtx(queryCtx)
      .maxDrivers(1)
      .spillDirectory(spillDirectory->path)
      .config(core::QueryConfig::kSpillEnabled, "true")
      .config(core::QueryConfig::kWriterSpillEnabled, "true")
      // Set file writer flush threshold of zero to always trigger flush in
      // test.
      .config(core::QueryConfig::kWriterFlushThresholdBytes, "0")
      // Set large stripe and dictionary size thresholds to avoid writer
      // internal stripe flush.
      .connectorConfig(
          kHiveConnectorId,
          connector::hive::HiveConfig::kOrcWriterMaxStripeSize,
          "1GB")
      .connectorConfig(
          kHiveConnectorId,
          connector::hive::HiveConfig::kOrcWriterMaxDictionaryMemory,
          "1GB")
      .plan(std::move(writerPlan))
      .assertResults(fmt::format("SELECT {}", numRows));

  ASSERT_EQ(arbitrator_->stats().numFailures, 1);
  ASSERT_EQ(arbitrator_->stats().numNonReclaimableAttempts, 1);
  ASSERT_EQ(arbitrator_->stats().numReserves, numAddedPools_);
  const auto updatedSpillStats = common::globalSpillStats();
  ASSERT_EQ(updatedSpillStats, spillStats);
}

// This test is to reproduce a race condition that memory arbitrator tries to
// reclaim from a set of hash build operators in which the last hash build
// operator has finished.
DEBUG_ONLY_TEST_F(SharedArbitrationTest, raceBetweenRaclaimAndJoinFinish) {
  const int kMemoryCapacity = 512 << 20;
  setupMemory(kMemoryCapacity, 0);

  const int numVectors = 5;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);

  std::shared_ptr<core::QueryCtx> joinQueryCtx = newQueryCtx(kMemoryCapacity);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId planNodeId;
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values(vectors, false)
                  .project({"c0 AS t0", "c1 AS t1", "c2 AS t2"})
                  .hashJoin(
                      {"t0"},
                      {"u0"},
                      PlanBuilder(planNodeIdGenerator)
                          .values(vectors, true)
                          .project({"c0 AS u0", "c1 AS u1", "c2 AS u2"})
                          .planNode(),
                      "",
                      {"t1"},
                      core::JoinType::kAnti)
                  .capturePlanNodeId(planNodeId)
                  .planNode();

  std::atomic<bool> waitForBuildFinishFlag{true};
  folly::EventCount waitForBuildFinishEvent;
  std::atomic<Driver*> lastBuildDriver{nullptr};
  std::atomic<Task*> task{nullptr};
  std::atomic<bool> isLastBuildFirstChildPool{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::HashBuild::finishHashBuild",
      std::function<void(exec::HashBuild*)>([&](exec::HashBuild* buildOp) {
        lastBuildDriver = buildOp->testingOperatorCtx()->driver();
        // Checks if the last build memory pool is the first build pool in its
        // parent node pool. It is used to check the test result.
        int buildPoolIndex{0};
        buildOp->pool()->parent()->visitChildren([&](memory::MemoryPool* pool) {
          if (pool == buildOp->pool()) {
            return false;
          }
          if (isHashBuildMemoryPool(*pool)) {
            ++buildPoolIndex;
          }
          return true;
        });
        isLastBuildFirstChildPool = (buildPoolIndex == 0);
        task = lastBuildDriver.load()->task().get();
        waitForBuildFinishFlag = false;
        waitForBuildFinishEvent.notifyAll();
      }));

  std::atomic<bool> waitForReclaimFlag{true};
  folly::EventCount waitForReclaimEvent;
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal",
      std::function<void(Driver*)>([&](Driver* driver) {
        auto* op = driver->findOperator(planNodeId);
        if (op->operatorType() != "HashBuild" &&
            op->operatorType() != "HashProbe") {
          return;
        }

        // Suspend hash probe driver to wait for the test triggered reclaim to
        // finish.
        if (op->operatorType() == "HashProbe") {
          op->pool()->reclaimer()->enterArbitration();
          waitForReclaimEvent.await(
              [&]() { return !waitForReclaimFlag.load(); });
          op->pool()->reclaimer()->leaveArbitration();
        }

        // Check if we have reached to the last hash build operator or not. The
        // testvalue callback will set the last build driver.
        if (lastBuildDriver == nullptr) {
          return;
        }

        // Suspend all the remaining hash build drivers until the test triggered
        // reclaim finish.
        op->pool()->reclaimer()->enterArbitration();
        waitForReclaimEvent.await([&]() { return !waitForReclaimFlag.load(); });
        op->pool()->reclaimer()->leaveArbitration();
      }));

  const int numDrivers = 4;
  std::thread queryThread([&]() {
    const auto spillDirectory = exec::test::TempDirectoryPath::create();
    AssertQueryBuilder(plan, duckDbQueryRunner_)
        .maxDrivers(numDrivers)
        .queryCtx(joinQueryCtx)
        .spillDirectory(spillDirectory->path)
        .config(core::QueryConfig::kSpillEnabled, "true")
        .config(core::QueryConfig::kJoinSpillEnabled, "true")
        .assertResults(
            "SELECT c1 FROM tmp WHERE c0 NOT IN (SELECT c0 FROM tmp)");
  });

  // Wait for the last hash build operator to start building the hash table.
  waitForBuildFinishEvent.await([&] { return !waitForBuildFinishFlag.load(); });
  ASSERT_TRUE(lastBuildDriver != nullptr);
  ASSERT_TRUE(task != nullptr);

  // Wait until the last build driver gets removed from the task after finishes.
  while (task.load()->numFinishedDrivers() != 1) {
    bool foundLastBuildDriver{false};
    task.load()->testingVisitDrivers([&](Driver* driver) {
      if (driver == lastBuildDriver) {
        foundLastBuildDriver = true;
      }
    });
    if (!foundLastBuildDriver) {
      break;
    }
  }

  // Reclaim from the task, and we can't reclaim anything as we don't support
  // spill after hash table built.
  memory::MemoryReclaimer::Stats stats;
  const uint64_t oldCapacity = joinQueryCtx->pool()->capacity();
  task.load()->pool()->reclaim(1'000, stats);
  // If the last build memory pool is first child of its parent memory pool,
  // then memory arbitration (or join node memory pool) will reclaim from the
  // last build operator first which simply quits as the driver has gone. If
  // not, we expect to get numNonReclaimableAttempts from any one of the
  // remaining hash build operator.
  if (isLastBuildFirstChildPool) {
    ASSERT_EQ(stats.numNonReclaimableAttempts, 0);
  } else {
    ASSERT_EQ(stats.numNonReclaimableAttempts, 1);
  }
  // Make sure we don't leak memory capacity since we reclaim from task pool
  // directly.
  static_cast<MemoryPoolImpl*>(task.load()->pool())
      ->testingSetCapacity(oldCapacity);
  waitForReclaimFlag = false;
  waitForReclaimEvent.notifyAll();

  queryThread.join();

  waitForAllTasksToBeDeleted();
  ASSERT_EQ(arbitrator_->stats().numFailures, 0);
  ASSERT_EQ(arbitrator_->stats().numReclaimedBytes, 0);
  ASSERT_EQ(arbitrator_->stats().numReserves, numAddedPools_);
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, arbitrateMemoryFromOtherOperator) {
  setupMemory(kMemoryCapacity, 0);
  const int numVectors = 10;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  createDuckDbTable(vectors);

  for (bool sameDriver : {false, true}) {
    SCOPED_TRACE(fmt::format("sameDriver {}", sameDriver));
    std::shared_ptr<core::QueryCtx> queryCtx = newQueryCtx(kMemoryCapacity);
    ASSERT_EQ(queryCtx->pool()->capacity(), 0);

    std::atomic<bool> injectAllocationOnce{true};
    const int initialBufferLen = 1 << 20;
    std::atomic<void*> buffer{nullptr};
    std::atomic<MemoryPool*> bufferPool{nullptr};
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::Values::getOutput",
        std::function<void(const exec::Values*)>(
            [&](const exec::Values* values) {
              if (!injectAllocationOnce.exchange(false)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                return;
              }
              buffer = values->pool()->allocate(initialBufferLen);
              bufferPool = values->pool();
            }));
    std::atomic<bool> injectReallocateOnce{true};
    SCOPED_TESTVALUE_SET(
        "facebook::velox::common::memory::MemoryPoolImpl::allocateNonContiguous",
        std::function<void(memory::MemoryPoolImpl*)>(
            ([&](memory::MemoryPoolImpl* pool) {
              const std::string re(".*Aggregation");
              if (!RE2::FullMatch(pool->name(), re)) {
                return;
              }
              if (pool->root()->currentBytes() == 0) {
                return;
              }
              if (!injectReallocateOnce.exchange(false)) {
                return;
              }
              ASSERT_TRUE(buffer != nullptr);
              ASSERT_TRUE(bufferPool != nullptr);
              const int newLength =
                  kMemoryCapacity - bufferPool.load()->capacity() + 1;
              VELOX_ASSERT_THROW(
                  bufferPool.load()->reallocate(
                      buffer, initialBufferLen, newLength),
                  "Exceeded memory pool cap");
            })));

    std::shared_ptr<Task> task;
    std::thread queryThread([&]() {
      if (sameDriver) {
        task = AssertQueryBuilder(duckDbQueryRunner_)
                   .queryCtx(queryCtx)
                   .plan(PlanBuilder()
                             .values(vectors)
                             .singleAggregation({"c0", "c1"}, {"array_agg(c2)"})
                             .localPartition(std::vector<std::string>{})
                             .planNode())
                   .assertResults(
                       "SELECT c0, c1, array_agg(c2) FROM tmp GROUP BY c0, c1");
      } else {
        task = AssertQueryBuilder(duckDbQueryRunner_)
                   .queryCtx(queryCtx)
                   .plan(PlanBuilder()
                             .values(vectors)
                             .localPartition({"c0", "c1"})
                             .singleAggregation({"c0", "c1"}, {"array_agg(c2)"})
                             .planNode())
                   .assertResults(
                       "SELECT c0, c1, array_agg(c2) FROM tmp GROUP BY c0, c1");
      }
    });

    queryThread.join();
    ASSERT_TRUE(buffer != nullptr);
    ASSERT_TRUE(bufferPool != nullptr);
    bufferPool.load()->free(buffer, initialBufferLen);

    task.reset();
    waitForAllTasksToBeDeleted();
  }
}

DEBUG_ONLY_TEST_F(SharedArbitrationTest, joinBuildSpillError) {
  const int kMemoryCapacity = 32 << 20;
  // Set a small memory capacity to trigger spill.
  setupMemory(kMemoryCapacity, 0);

  const int numVectors = 16;
  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }

  std::shared_ptr<core::QueryCtx> joinQueryCtx = newQueryCtx(kMemoryCapacity);

  const int numDrivers = 4;
  std::atomic<int> numAppends{0};
  const std::string injectedErrorMsg("injected spillError");
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::SpillState::appendToPartition",
      std::function<void(exec::SpillState*)>([&](exec::SpillState* state) {
        if (++numAppends != numDrivers) {
          return;
        }
        VELOX_FAIL(injectedErrorMsg);
      }));

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto spillDirectory = exec::test::TempDirectoryPath::create();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values(vectors)
                  .project({"c0 AS t0", "c1 AS t1", "c2 AS t2"})
                  .hashJoin(
                      {"t0"},
                      {"u0"},
                      PlanBuilder(planNodeIdGenerator)
                          .values(vectors)
                          .project({"c0 AS u0", "c1 AS u1", "c2 AS u2"})
                          .planNode(),
                      "",
                      {"t1"},
                      core::JoinType::kAnti)
                  .planNode();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan)
          .queryCtx(joinQueryCtx)
          .spillDirectory(spillDirectory->path)
          .config(core::QueryConfig::kSpillEnabled, "true")
          .config(core::QueryConfig::kJoinSpillEnabled, "true")
          .copyResults(pool()),
      injectedErrorMsg);

  waitForAllTasksToBeDeleted();
  ASSERT_EQ(arbitrator_->stats().numFailures, 1);
  ASSERT_EQ(arbitrator_->stats().numReserves, numAddedPools_);
}

TEST_F(SharedArbitrationTest, concurrentArbitration) {
  // Tries to replicate an actual workload by concurrently running multiple
  // query shapes that support spilling (and hence can be forced to abort or
  // spill by the arbitrator). Also adds an element of randomness by randomly
  // keeping completed tasks alive (zombie tasks) hence holding on to some
  // memory. Ensures that arbitration is engaged under memory contention and
  // failed queries only have errors related to memory or arbitration.
  FLAGS_velox_suppress_memory_capacity_exceeding_error_message = true;
  const int numVectors = 8;
  std::vector<RowVectorPtr> vectors;
  fuzzerOpts_.vectorSize = 32;
  fuzzerOpts_.stringVariableLength = false;
  fuzzerOpts_.stringLength = 32;
  vectors.reserve(numVectors);
  for (int i = 0; i < numVectors; ++i) {
    vectors.push_back(newVector());
  }
  const int numDrivers = 4;
  const auto expectedWriteResult =
      runWriteTask(vectors, nullptr, numDrivers, false).first;
  const auto expectedJoinResult =
      runHashJoinTask(vectors, nullptr, numDrivers, false).first;
  const auto expectedOrderResult =
      runOrderByTask(vectors, nullptr, numDrivers, false).first;
  const auto expectedTopNResult =
      runTopNTask(vectors, nullptr, numDrivers, false).first;

  struct {
    uint64_t totalCapacity;
    uint64_t queryCapacity;

    std::string debugString() const {
      return fmt::format(
          "totalCapacity = {}, queryCapacity = {}.",
          succinctBytes(totalCapacity),
          succinctBytes(queryCapacity));
    }
  } testSettings[3] = {
      {16 * MB, 128 * MB}, {128 * MB, 16 * MB}, {128 * MB, 128 * MB}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto totalCapacity = testData.totalCapacity;
    auto queryCapacity = testData.queryCapacity;
    setupMemory(totalCapacity);

    std::mutex mutex;
    std::vector<std::shared_ptr<core::QueryCtx>> queries;
    std::deque<std::shared_ptr<Task>> zombieTasks;

    const int numThreads = 32;
    const int maxNumZombieTasks = 8;
    std::vector<std::thread> queryThreads;
    queryThreads.reserve(numThreads);
    for (int i = 0; i < numThreads; ++i) {
      queryThreads.emplace_back([&, i]() {
        std::shared_ptr<Task> task;
        try {
          auto queryCtx = newQueryCtx(queryCapacity);
          if (i == 0) {
            // Write task contains aggregate node, which does not support
            // multithread aggregation type resolver, so make sure it is built
            // in a single thread.
            task = runWriteTask(
                       vectors, queryCtx, numDrivers, true, expectedWriteResult)
                       .second;
          } else if ((i % 4) == 0) {
            task = runHashJoinTask(
                       vectors, queryCtx, numDrivers, true, expectedJoinResult)
                       .second;
          } else if ((i % 4) == 1) {
            task = runOrderByTask(
                       vectors, queryCtx, numDrivers, true, expectedOrderResult)
                       .second;
          } else {
            task = runTopNTask(
                       vectors, queryCtx, numDrivers, true, expectedTopNResult)
                       .second;
          }
        } catch (const VeloxException& e) {
          VELOX_CHECK(
              e.errorCode() == error_code::kMemCapExceeded.c_str() ||
              e.errorCode() == error_code::kMemAborted.c_str() ||
              e.errorCode() == error_code::kMemAllocError.c_str());
        }

        // TODO: Add RowNumber task after fixing its spiller bug.
        std::lock_guard<std::mutex> l(mutex);
        if (folly::Random().oneIn(3)) {
          zombieTasks.emplace_back(std::move(task));
        }
        while (zombieTasks.size() > maxNumZombieTasks) {
          zombieTasks.pop_front();
        }
      });
    }

    for (auto& queryThread : queryThreads) {
      queryThread.join();
    }
    ASSERT_GT(arbitrator_->stats().numRequests, 0);
  }
}

TEST_F(SharedArbitrationTest, reserveReleaseCounters) {
  for (int i = 0; i < 37; ++i) {
    folly::Random::DefaultGenerator rng(i);
    auto numRootPools = folly::Random::rand32(rng) % 11 + 3;
    std::vector<std::thread> threads;
    threads.reserve(numRootPools);
    std::mutex mutex;
    setupMemory(kMemoryCapacity, 0);
    {
      std::vector<std::shared_ptr<core::QueryCtx>> queries;
      queries.reserve(numRootPools);
      for (int j = 0; j < numRootPools; ++j) {
        threads.emplace_back([&]() {
          {
            std::lock_guard<std::mutex> l(mutex);
            auto oldNum = arbitrator_->stats().numReserves;
            queries.emplace_back(newQueryCtx());
            ASSERT_EQ(arbitrator_->stats().numReserves, oldNum + 1);
          }
        });
      }

      for (auto& queryThread : threads) {
        queryThread.join();
      }
      ASSERT_EQ(arbitrator_->stats().numReserves, numRootPools);
      ASSERT_EQ(arbitrator_->stats().numReleases, 0);
    }
    ASSERT_EQ(arbitrator_->stats().numReserves, numRootPools);
    ASSERT_EQ(arbitrator_->stats().numReleases, numRootPools);
  }
}
} // namespace facebook::velox::exec::test
