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
#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/core/QueryConfig.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

DEFINE_int32(width, 16, "Number of parties in shuffle");
DEFINE_int32(task_width, 4, "Number of threads in each task in shuffle");

DEFINE_int32(flat_batch_mb, 1, "MB in a 10k row flat batch.");
DEFINE_int64(exchange_buffer_mb, 32, "task-wide buffer in remote exchange");
DEFINE_int32(dict_pct, 0, "Percentage of columns wrapped in dictionary");
// Add the following definitions to allow Clion runs
DEFINE_bool(gtest_color, false, "");
DEFINE_string(gtest_filter, "*", "");

/// Benchmarks repartition/exchange with different batch sizes,
/// numbers of destinations and data type mixes.  Generates a plan
/// that 1. shuffles a constant input in each of n workers, sending
/// each partition to n consumers in the next stage. The consumers
/// count the rows and send the count to a final single task stage
/// that returns the sum of the counts. The sum is expected to be n *
/// number of rows in constant input.

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace {

struct ExchangeRunStats {
  int64_t wallUs = 0;
  PlanNodeStats partitionedOutputStats;
  PlanNodeStats exchangeStats;
};

void printExchangeStats(
    const std::string& datasetName,
    const std::string& modeName,
    const ExchangeRunStats& stats) {
  std::cout << "-----------------------------" << datasetName << " ("
            << modeName << ")-----------------------------" << std::endl;
  std::cout << "Wall Time (ms): " << succinctMicros(stats.wallUs) << std::endl;
  std::cout << "PartitionOutput: " << stats.partitionedOutputStats.toString()
            << std::endl;
  std::cout << "Exchange: " << stats.exchangeStats.toString() << std::endl;
}

class ExchangeBenchmark : public VectorTestBase {
 public:
  std::vector<RowVectorPtr> makeRows(
      RowTypePtr type,
      int32_t numVectors,
      int32_t rowsPerVector,
      int32_t dictPct = 0) {
    std::vector<RowVectorPtr> vectors;
    BufferPtr indices;
    for (int32_t i = 0; i < numVectors; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          BatchMaker::createBatch(type, rowsPerVector, *pool_));

      auto width = vector->childrenSize();
      for (auto child = 0; child < width; ++child) {
        if (100 * child / width > dictPct) {
          if (!indices) {
            indices = makeIndices(vector->size(), [&](auto i) { return i; });
          }
          vector->childAt(child) = BaseVector::wrapInDictionary(
              nullptr, indices, vector->size(), vector->childAt(child));
        }
      }
      vectors.push_back(vector);
    }
    return vectors;
  }

  void run(
      std::vector<RowVectorPtr>& vectors,
      int32_t width,
      int32_t taskWidth,
      bool useOptimizedPartitionedOutput,
      int64_t& wallUs,
      PlanNodeStats& partitionedOutputStats,
      PlanNodeStats& exchangeStats) {
    core::PlanNodePtr plan;
    core::PlanNodeId exchangeId;
    core::PlanNodeId leafPartitionedOutputId;
    core::PlanNodeId finalAggPartitionedOutputId;

    std::vector<std::shared_ptr<Task>> leafTasks;
    std::vector<std::shared_ptr<Task>> finalAggTasks;
    std::vector<exec::Split> finalAggSplits;

    RowVectorPtr expected;

    const auto startUs = getCurrentTimeMicro();
    BENCHMARK_SUSPEND {
      assert(!vectors.empty());
      configSettings_[core::QueryConfig::kMaxPartitionedOutputBufferSize] =
          fmt::format("{}", FLAGS_exchange_buffer_mb << 20);
      const auto iteration = ++iteration_;

      // leafPlan: PartitionedOutput/kPartitioned(1) <-- Values(0)
      std::vector<std::string> leafTaskIds;
      auto leafPlan = exec::test::PlanBuilder()
                          .values(vectors, true)
                          .partitionedOutput({"c0"}, width)
                          .capturePlanNodeId(leafPartitionedOutputId)
                          .planNode();

      for (int32_t counter = 0; counter < width; ++counter) {
        auto leafTaskId = makeTaskId(iteration, "leaf", counter);
        leafTaskIds.push_back(leafTaskId);
        auto leafTask = makeTask(leafTaskId, leafPlan, counter);
        leafTasks.push_back(leafTask);
        leafTask->start(taskWidth);
      }

      // finalAggPlan: PartitionedOutput/kPartitioned(2) <-- Agg/kSingle(1) <--
      // Exchange(0)
      std::vector<std::string> finalAggTaskIds;
      core::PlanNodePtr finalAggPlan =
          exec::test::PlanBuilder()
              .exchange(leafPlan->outputType(), "Presto")
              .capturePlanNodeId(exchangeId)
              .singleAggregation({}, {"count(1)"})
              .partitionedOutput({}, 1)
              .capturePlanNodeId(finalAggPartitionedOutputId)
              .planNode();

      for (int i = 0; i < width; i++) {
        auto taskId = makeTaskId(iteration, "final-agg", i);
        finalAggSplits.push_back(
            exec::Split(std::make_shared<exec::RemoteConnectorSplit>(taskId)));
        auto finalAggTask = makeTask(taskId, finalAggPlan, i);
        finalAggTasks.push_back(finalAggTask);
        finalAggTask->start(taskWidth);
        addRemoteSplits(finalAggTask, leafTaskIds);
      }

      expected = makeRowVector({makeFlatVector<int64_t>(1, [&](auto /*row*/) {
        return vectors.size() * vectors[0]->size() * width * taskWidth;
      })});

      // plan: Agg/kSingle(1) <-- Exchange (0)
      plan = exec::test::PlanBuilder()
                 .exchange(finalAggPlan->outputType(), "Presto")
                 .singleAggregation({}, {"sum(a0)"})
                 .planNode();
    };

    exec::test::AssertQueryBuilder(plan)
        .splits(finalAggSplits)
        .assertResults(expected);

    BENCHMARK_SUSPEND {
      wallUs = getCurrentTimeMicro() - startUs;
      std::vector<int64_t> taskWallMs;

      for (const auto& task : leafTasks) {
        const auto& taskStats = task->taskStats();
        taskWallMs.push_back(
            taskStats.executionEndTimeMs - taskStats.executionStartTimeMs);
        const auto& planStats = toPlanStats(taskStats);
        auto& taskPartitionedOutputStats =
            planStats.at(leafPartitionedOutputId);
        partitionedOutputStats += taskPartitionedOutputStats;
      }

      for (const auto& task : finalAggTasks) {
        const auto& taskStats = task->taskStats();
        taskWallMs.push_back(
            taskStats.executionEndTimeMs - taskStats.executionStartTimeMs);
        const auto& planStats = toPlanStats(taskStats);

        auto& taskPartitionedOutputStats =
            planStats.at(finalAggPartitionedOutputId);
        partitionedOutputStats += taskPartitionedOutputStats;

        auto& taskExchangeStats = planStats.at(exchangeId);
        exchangeStats += taskExchangeStats;
      }
    };
  }

 private:
  static constexpr int64_t kMaxMemory = 6UL << 30; // 6GB

  static std::string
  makeTaskId(int32_t iteration, const std::string& prefix, int num) {
    return fmt::format("local://{}-{}-{}", iteration, prefix, num);
  }

  std::shared_ptr<Task> makeTask(
      const std::string& taskId,
      std::shared_ptr<const core::PlanNode> planNode,
      int destination,
      Consumer consumer = nullptr,
      int64_t maxMemory = kMaxMemory) {
    auto configCopy = configSettings_;
    auto queryCtx = core::QueryCtx::create(
        executor_.get(), core::QueryConfig(std::move(configCopy)));
    queryCtx->testingOverrideMemoryPool(
        memory::memoryManager()->addRootPool(queryCtx->queryId(), maxMemory));
    core::PlanFragment planFragment{planNode};
    return Task::create(
        taskId,
        std::move(planFragment),
        destination,
        std::move(queryCtx),
        Task::ExecutionMode::kParallel,
        std::move(consumer));
  }

  void addRemoteSplits(
      std::shared_ptr<Task> task,
      const std::vector<std::string>& remoteTaskIds) {
    for (const auto& taskId : remoteTaskIds) {
      auto split =
          exec::Split(std::make_shared<RemoteConnectorSplit>(taskId), -1);
      task->addSplit("0", std::move(split));
    }
    task->noMoreSplits("0");
  }

  std::unordered_map<std::string, std::string> configSettings_;
  // Serial number to differentiate consecutive benchmark repeats.
  static int32_t iteration_;
};

int32_t ExchangeBenchmark::iteration_;

std::unique_ptr<ExchangeBenchmark> bm;

void runBenchmarks(bool optimizedPartitionedOutputEnabled = false) {
  std::vector<std::string> flatNames = {"c0"};
  std::vector<TypePtr> flatTypes = {BIGINT()};
  std::vector<TypePtr> typeSelection = {
      BOOLEAN(),
      TINYINT(),
      DECIMAL(20, 3),
      INTEGER(),
      BIGINT(),
      REAL(),
      DECIMAL(10, 2),
      DOUBLE(),
      VARCHAR()};

  int64_t flatSize = 0;
  // Add enough columns of different types to make a 10K row batch be
  // flat_batch_mb in flat size.
  while (flatSize * 10000 < static_cast<int64_t>(FLAGS_flat_batch_mb) << 20) {
    flatNames.push_back(fmt::format("c{}", flatNames.size()));
    assert(!flatNames.empty());
    flatTypes.push_back(typeSelection[flatTypes.size() % typeSelection.size()]);
    if (flatTypes.back()->isFixedWidth()) {
      flatSize += flatTypes.back()->cppSizeInBytes();
    } else {
      flatSize += 20;
    }
  }
  auto flatType = ROW(std::move(flatNames), std::move(flatTypes));

  auto structType = ROW(
      {{"c0", BIGINT()},
       {"r1",
        ROW(
            {{"k2", BIGINT()},
             {"r2",
              ROW(
                  {{"i1", BIGINT()},
                   {"i2", BIGINT()},
                   {"r3}, ROW({{s3", VARCHAR()},
                   {"i5", INTEGER()},
                   {"d5", DOUBLE()},
                   {"b5", BOOLEAN()},
                   {"a5", ARRAY(TINYINT())}})}})}});

  auto deepType = ROW(
      {{"c0", BIGINT()},
       {"long_array_val", ARRAY(ARRAY(BIGINT()))},
       {"array_val", ARRAY(VARCHAR())},
       {"struct_val", ROW({{"s_int", INTEGER()}, {"s_array", ARRAY(REAL())}})},
       {"map_val",
        MAP(VARCHAR(),
            MAP(BIGINT(),
                ROW({{"s2_int", INTEGER()}, {"s2_string", VARCHAR()}})))}});

  std::vector<RowVectorPtr> flat10k(
      bm->makeRows(flatType, 10, 10000, FLAGS_dict_pct));
  std::vector<RowVectorPtr> deep10k(
      bm->makeRows(deepType, 10, 10000, FLAGS_dict_pct));
  std::vector<RowVectorPtr> flat50(
      bm->makeRows(flatType, 2000, 50, FLAGS_dict_pct));
  std::vector<RowVectorPtr> deep50(
      bm->makeRows(deepType, 2000, 50, FLAGS_dict_pct));
  std::vector<RowVectorPtr> struct1k(
      bm->makeRows(structType, 100, 1000, FLAGS_dict_pct));

  std::vector<std::pair<std::string, std::vector<RowVectorPtr>*>> exchangeCases{
      {"Flat10K", &flat10k},
      {"Flat50", &flat50},
      {"Deep10K", &deep10k},
      {"Deep50", &deep50},
      {"Struct1K", &struct1k}};

  std::vector<ExchangeRunStats> normalPartitionedOutputStats(
      exchangeCases.size());
  std::vector<ExchangeRunStats> optimizedPartitionedOutputStats(
      exchangeCases.size());

  for (size_t i = 0; i < exchangeCases.size(); ++i) {
    const auto& name = exchangeCases[i].first;
    folly::addBenchmark(
        __FILE__,
        fmt::format("exchange{}_normalPartitionedOutput", name),
        [&, i]() {
          bm->run(
              *exchangeCases[i].second,
              FLAGS_width,
              FLAGS_task_width,
              false,
              normalPartitionedOutputStats[i].wallUs,
              normalPartitionedOutputStats[i].partitionedOutputStats,
              normalPartitionedOutputStats[i].exchangeStats);
          return 1;
        });
    if (optimizedPartitionedOutputEnabled) {
      folly::addBenchmark(
          __FILE__,
          fmt::format("exchange{}_optimizedPartitionedOutput", name),
          [&, i]() {
            bm->run(
                *exchangeCases[i].second,
                FLAGS_width,
                FLAGS_task_width,
                true,
                optimizedPartitionedOutputStats[i].wallUs,
                optimizedPartitionedOutputStats[i].partitionedOutputStats,
                optimizedPartitionedOutputStats[i].exchangeStats);
            return 1;
          });
    }
  }

  folly::runBenchmarks();

  for (size_t i = 0; i < exchangeCases.size(); ++i) {
    printExchangeStats(
        exchangeCases[i].first, "normal", normalPartitionedOutputStats[i]);
    if (optimizedPartitionedOutputEnabled) {
      printExchangeStats(
          exchangeCases[i].first,
          "optimized",
          optimizedPartitionedOutputStats[i]);
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  if (!isRegisteredNamedVectorSerde("Presto")) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }
  exec::ExchangeSource::registerFactory(exec::test::createLocalExchangeSource);

  bm = std::make_unique<ExchangeBenchmark>();
  runBenchmarks();
  bm.reset();

  return 0;
}
