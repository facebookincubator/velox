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

#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/common/base/PeriodicStatsReporter.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/TaskMonitor.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/CompactRowSerializer.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/serializers/UnsafeRowSerializer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

DECLARE_bool(velox_memory_leak_check_enabled);
DECLARE_bool(velox_enable_memory_usage_track_in_default_memory_pool);

using namespace facebook::velox::common::testutil;
using namespace facebook::velox::memory;

namespace facebook::velox::exec::test {

OperatorTestBase::OperatorTestBase() {
  // Overloads the memory pools used by VectorTestBase to work with memory
  // arbitrator.
  rootPool_ = memory::memoryManager()->addRootPool(
      "", memory::kMaxMemory, exec::MemoryReclaimer::create());
  pool_ = rootPool_->addLeafChild("", true, exec::MemoryReclaimer::create());
  vectorMaker_ = velox::test::VectorMaker(pool_.get());

  parse::registerTypeResolver();
}

void OperatorTestBase::registerVectorSerde() {
  velox::serializer::presto::PrestoVectorSerde::registerVectorSerde();
}

OperatorTestBase::~OperatorTestBase() {
  // Wait for all the tasks to be deleted.
  exec::test::TaskMonitor::waitForAllTasksDeleted();
}

void OperatorTestBase::SetUpTestCase() {
  FLAGS_velox_enable_memory_usage_track_in_default_memory_pool = true;
  FLAGS_velox_memory_leak_check_enabled = true;
  memory::SharedArbitrator::registerFactory();
  resetMemory();
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  TestValue::enable();
}

void OperatorTestBase::TearDownTestCase() {
  asyncDataCache_->shutdown();
  TaskMonitor::waitForAllTasksDeleted();
  memory::SharedArbitrator::unregisterFactory();
}

void OperatorTestBase::setupMemory(
    int64_t allocatorCapacity,
    int64_t arbitratorCapacity,
    int64_t arbitratorReservedCapacity,
    int64_t memoryPoolInitCapacity,
    int64_t memoryPoolReservedCapacity,
    int64_t memoryPoolMinReclaimBytes,
    int64_t memoryPoolAbortCapacityLimit) {
  if (asyncDataCache_ != nullptr) {
    asyncDataCache_->clear();
    asyncDataCache_.reset();
  }
  MemoryManager::Options options;
  options.allocatorCapacity = allocatorCapacity;
  options.arbitratorCapacity = arbitratorCapacity;
  options.arbitratorKind = "SHARED";
  options.checkUsageLeak = true;
  options.arbitrationStateCheckCb = memoryArbitrationStateCheck;

  using ExtraConfig = SharedArbitrator::ExtraConfig;
  options.extraArbitratorConfigs = {
      {std::string(ExtraConfig::kReservedCapacity),
       folly::to<std::string>(arbitratorReservedCapacity) + "B"},
      {std::string(ExtraConfig::kMemoryPoolInitialCapacity),
       folly::to<std::string>(memoryPoolInitCapacity) + "B"},
      {std::string(ExtraConfig::kMemoryPoolReservedCapacity),
       folly::to<std::string>(memoryPoolReservedCapacity) + "B"},
      {std::string(ExtraConfig::kMemoryPoolMinReclaimBytes),
       folly::to<std::string>(memoryPoolMinReclaimBytes) + "B"},
      // For simplicity, we set the reclaim pct to 0, so that the tests will be
      // purely based on kMemoryPoolMinReclaimBytes.
      {std::string(ExtraConfig::kMemoryPoolMinReclaimPct), "0"},
      {std::string(ExtraConfig::kMemoryPoolAbortCapacityLimit),
       folly::to<std::string>(memoryPoolAbortCapacityLimit) + "B"},
      {std::string(ExtraConfig::kGlobalArbitrationEnabled), "true"},
  };

  memory::MemoryManager::testingSetInstance(options);
  asyncDataCache_ =
      cache::AsyncDataCache::create(memory::memoryManager()->allocator());
  cache::AsyncDataCache::setInstance(asyncDataCache_.get());
}

void OperatorTestBase::resetMemory() {
  OperatorTestBase::setupMemory(8L << 30, 6L << 30, 0, 512 << 20, 0, 0, 0);
}

void OperatorTestBase::SetUp() {
  if (!isRegisteredVectorSerde()) {
    this->registerVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde("Presto")) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde("CompactRow")) {
    serializer::CompactRowVectorSerde::registerNamedVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde("UnsafeRow")) {
    serializer::spark::UnsafeRowVectorSerde::registerNamedVectorSerde();
  }
  driverExecutor_ = std::make_unique<folly::CPUThreadPoolExecutor>(3);
  ioExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(3);
  PeriodicStatsReporter::Options options;
  options.allocator = memory::memoryManager()->allocator();
  options.allocatorStatsIntervalMs = 2'000;
  options.cache = asyncDataCache_.get();
  options.cacheStatsIntervalMs = 2'000;
  options.arbitrator = memory::memoryManager()->arbitrator();
  options.arbitratorStatsIntervalMs = 2'000;
  options.spillMemoryPool = memory::spillMemoryPool();
  options.spillStatsIntervalMs = 2'000;
  startPeriodicStatsReporter(options);
  testingStartLocalExchangeSource();
}

void OperatorTestBase::TearDown() {
  TaskMonitor::waitForAllTasksDeleted();
  stopPeriodicStatsReporter();
  executor_.reset();
  // There might be lingering exchange source on executor even after all tasks
  // are deleted. This can cause memory leak because exchange source holds
  // reference to memory pool. We need to make sure they are properly cleaned.
  testingShutdownLocalExchangeSource();
  pool_.reset();
  rootPool_.reset();
  resetMemory();
}

OperatorTestBase::FlatMapTestBatches OperatorTestBase::makeFlatMapTestBatches(
    int32_t numBatches,
    vector_size_t numRows) {
  using Entries = std::vector<std::pair<int32_t, std::optional<int64_t>>>;
  FlatMapTestBatches batches;
  for (int32_t batch = 0; batch < numBatches; ++batch) {
    std::vector<std::optional<Entries>> maps;
    for (vector_size_t row = 0; row < numRows; ++row) {
      if (row % 7 == 3) {
        maps.emplace_back(std::nullopt);
        continue;
      }
      Entries entries;
      for (int32_t key = batch; row % 11 != 5 && key < batch + 5; ++key) {
        if ((row + key) % 3 != 0) {
          entries.emplace_back(
              key,
              (row + key) % 5 == 0 ? std::nullopt
                                   : std::optional<int64_t>(row * 10 + key));
        }
      }
      maps.emplace_back(std::move(entries));
    }
    auto keys = makeFlatVector<int64_t>(numRows, [&](auto row) {
      return batch * numRows + (row * 37) % numRows;
    });
    auto map = makeNullableMapVector<int32_t, int64_t>(maps);
    batches.maps.push_back(makeRowVector({keys, map}));
    batches.flatMaps.push_back(
        makeRowVector({keys, velox::test::withFlatMapBase(map)}));
  }
  return batches;
}

TaskPtr OperatorTestBase::assertQuery(
    const core::PlanNodePtr& plan,
    const std::vector<std::shared_ptr<connector::ConnectorSplit>>&
        connectorSplits,
    const std::string& duckDbSql) {
  std::vector<exec::Split> splits;
  splits.reserve(connectorSplits.size());
  for (const auto& connectorSplit : connectorSplits) {
    splits.emplace_back(exec::Split(folly::copy(connectorSplit), -1));
  }

  return assertQuery(plan, std::move(splits), duckDbSql);
}

namespace {
/// Returns the plan node ID of the only leaf plan node. Throws if 'root' has
/// multiple leaf nodes.
core::PlanNodeId getOnlyLeafPlanNodeId(const core::PlanNodePtr& root) {
  const auto& sources = root->sources();
  if (sources.empty()) {
    return root->id();
  }

  VELOX_CHECK_EQ(1, sources.size());
  return getOnlyLeafPlanNodeId(sources[0]);
}
} // namespace

TaskPtr OperatorTestBase::assertQuery(
    const core::PlanNodePtr& plan,
    const std::string& duckDbSql) {
  return AssertQueryBuilder(plan, duckDbQueryRunner_).assertResults(duckDbSql);
}

TaskPtr OperatorTestBase::assertQuery(
    const core::PlanNodePtr& plan,
    const RowVectorPtr& expectedResults) {
  return AssertQueryBuilder(plan).assertResults(expectedResults);
}

TaskPtr OperatorTestBase::assertQuery(
    const core::PlanNodePtr& plan,
    std::vector<exec::Split>&& splits,
    const std::string& duckDbSql) {
  return AssertQueryBuilder(plan, duckDbQueryRunner_)
      .splits(getOnlyLeafPlanNodeId(plan), std::move(splits))
      .assertResults(duckDbSql);
}

core::TypedExprPtr OperatorTestBase::parseExpr(
    const std::string& sql,
    const RowTypePtr& rowType,
    const parse::ParseOptions& options) {
  auto untyped = parse::DuckSqlExpressionsParser(options).parseExpr(sql);
  return core::Expressions::inferTypes(untyped, rowType, pool_.get());
}

/*static*/ void OperatorTestBase::deleteTaskAndCheckSpillDirectory(
    TaskPtr& task) {
  const auto spillDirectoryStr = task->spillDirectory();
  // Nothing to do if there is no spilling directory was set.
  if (spillDirectoryStr.empty()) {
    return;
  }

  // Wait for the task to go.
  task.reset();
  TaskMonitor::waitForAllTasksDeleted();

  // If a spilling directory was set, ensure it was removed after the task is
  // gone.
  auto fs = filesystems::getFileSystem(spillDirectoryStr, nullptr);
  EXPECT_FALSE(fs->exists(spillDirectoryStr));
}

} // namespace facebook::velox::exec::test
