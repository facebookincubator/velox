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

#include <algorithm>
#include <mutex>
#include <thread>

#include "velox/core/QueryConfig.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

DEFINE_int32(width, 16, "Number of drivers in each local exchange task");
DEFINE_int32(num_local_tasks, 8, "Number of concurrent local shuffles");
DEFINE_int32(num_local_repeat, 8, "Number of repeats of local exchange query");
DEFINE_int32(flat_batch_mb, 1, "MB in a 10k row flat batch.");
DEFINE_int64(
    local_exchange_buffer_mb,
    32,
    "task-wide buffer in local exchange");
DEFINE_int32(dict_pct, 0, "Percentage of columns wrapped in dictionary");
// Add the following definitions to allow Clion runs
DEFINE_bool(gtest_color, false, "");
DEFINE_string(gtest_filter, "*", "");

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace {

struct LocalPartitionWaitStats {
  int64_t totalProducerWaitMs = 0;
  int64_t totalConsumerWaitMs = 0;
  std::vector<RuntimeMetric> consumerWaitMs;
  std::vector<RuntimeMetric> producerWaitMs;
  std::vector<int64_t> wallMs;
};

void sortByMax(std::vector<RuntimeMetric>& metrics) {
  std::sort(
      metrics.begin(),
      metrics.end(),
      [](const RuntimeMetric& left, const RuntimeMetric& right) {
        return left.max > right.max;
      });
}

void sortByAndPrintMax(
    const char* title,
    int64_t total,
    std::vector<RuntimeMetric>& metrics) {
  sortByMax(metrics);
  VELOX_CHECK(!metrics.empty());
  std::cout << title << "\n Total " << succinctNanos(total)
            << "\n Max: " << metrics.front().toString()
            << "\n Median: " << metrics[metrics.size() / 2].toString()
            << "\n Min: " << metrics.back().toString() << std::endl;
}

class LocalExchangeBenchmark : public VectorTestBase {
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

  void runLocal(
      std::vector<RowVectorPtr>& vectors,
      int32_t taskWidth,
      int32_t numTasks,
      int64_t& localPartitionWallUs,
      PlanNodeStats& partitionedOutputStats,
      LocalPartitionWaitStats& localPartitionWaitStats) {
    VELOX_CHECK(!vectors.empty());

    core::PlanNodePtr plan;
    core::PlanNodeId localPartitionId1;
    core::PlanNodeId localPartitionId2;
    std::vector<std::shared_ptr<Task>> tasks;
    std::vector<std::thread> threads;

    RowVectorPtr expected;

    BENCHMARK_SUSPEND {
      std::vector<std::string> aggregates = {"count(1)"};
      auto& rowType = vectors[0]->type()->as<TypeKind::ROW>();
      for (auto i = 1; i < rowType.size(); ++i) {
        aggregates.push_back(fmt::format("checksum({})", rowType.nameOf(i)));
      }

      // plan: Agg/kSingle(4) <-- LocalPartition/Gather(3) <-- Agg/kGather(2)
      // <-- LocalPartition/kRepartition(1) <-- Values(0)
      plan = exec::test::PlanBuilder()
                 .values(vectors, true)
                 .localPartition({"c0"})
                 .capturePlanNodeId(localPartitionId1)
                 .singleAggregation({}, aggregates)
                 .localPartition(std::vector<std::string>{})
                 .capturePlanNodeId(localPartitionId2)
                 .singleAggregation({}, {"sum(a0)"})
                 .planNode();

      threads.reserve(numTasks);
      expected = makeRowVector({makeFlatVector<int64_t>(1, [&](auto /*row*/) {
        return vectors.size() * vectors[0]->size() * taskWidth;
      })});
    };

    const auto startMicros = getCurrentTimeMicro();
    std::mutex mutex;
    for (int32_t i = 0; i < numTasks; ++i) {
      threads.push_back(std::thread([&]() {
        for (auto repeat = 0; repeat < FLAGS_num_local_repeat; ++repeat) {
          auto task =
              exec::test::AssertQueryBuilder(plan)
                  .config(
                      core::QueryConfig::kMaxLocalExchangeBufferSize,
                      fmt::format("{}", FLAGS_local_exchange_buffer_mb << 20))
                  .maxDrivers(taskWidth)
                  .assertResults(expected);
          {
            std::lock_guard<std::mutex> l(mutex);
            tasks.push_back(task);
          }
        }
      }));
    }
    for (auto& thread : threads) {
      thread.join();
    }

    BENCHMARK_SUSPEND {
      localPartitionWallUs = getCurrentTimeMicro() - startMicros;

      std::vector<core::PlanNodeId> localPartitionNodeIds{
          localPartitionId1, localPartitionId2};

      localPartitionWaitStats.totalProducerWaitMs = 0;
      localPartitionWaitStats.totalConsumerWaitMs = 0;
      for (const auto& task : tasks) {
        const auto taskStats = task->taskStats();
        localPartitionWaitStats.wallMs.push_back(
            taskStats.executionEndTimeMs - taskStats.executionStartTimeMs);
        const auto planStats = toPlanStats(taskStats);

        for (const auto& nodeId : localPartitionNodeIds) {
          const auto planStatsIt = planStats.find(nodeId);
          if (planStatsIt == planStats.end()) {
            continue;
          }
          const auto& taskLocalPartitionStats = planStatsIt->second;
          partitionedOutputStats += taskLocalPartitionStats;

          const auto& runtimeStats = taskLocalPartitionStats.customStats;
          const auto producerWaitIt =
              runtimeStats.find("blockedWaitForProducerWallNanos");
          const auto consumerWaitIt =
              runtimeStats.find("blockedWaitForConsumerWallNanos");
          const RuntimeMetric producerWait =
              producerWaitIt == runtimeStats.end() ? RuntimeMetric{}
                                                   : producerWaitIt->second;
          const RuntimeMetric consumerWait =
              consumerWaitIt == runtimeStats.end() ? RuntimeMetric{}
                                                   : consumerWaitIt->second;
          localPartitionWaitStats.producerWaitMs.push_back(producerWait);
          localPartitionWaitStats.consumerWaitMs.push_back(consumerWait);
          localPartitionWaitStats.totalProducerWaitMs +=
              localPartitionWaitStats.producerWaitMs.back().sum;
          localPartitionWaitStats.totalConsumerWaitMs +=
              localPartitionWaitStats.consumerWaitMs.back().sum;
        }
      }
    };
  }
};

std::unique_ptr<LocalExchangeBenchmark> bm;

void runBenchmarks() {
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
    flatTypes.push_back(typeSelection[flatTypes.size() % typeSelection.size()]);
    if (flatTypes.back()->isFixedWidth()) {
      flatSize += flatTypes.back()->cppSizeInBytes();
    } else {
      flatSize += 20;
    }
  }
  auto flatType = ROW(std::move(flatNames), std::move(flatTypes));
  std::vector<RowVectorPtr> flat10k(
      bm->makeRows(flatType, 10, 10000, FLAGS_dict_pct));

  int64_t localPartitionWallUs;
  PlanNodeStats localPartitionStatsFlat10K;
  LocalPartitionWaitStats localPartitionWaitStats;
  folly::addBenchmark(__FILE__, "localFlat10k", [&]() {
    bm->runLocal(
        flat10k,
        FLAGS_width,
        FLAGS_num_local_tasks,
        localPartitionWallUs,
        localPartitionStatsFlat10K,
        localPartitionWaitStats);
    return 1;
  });

  folly::runBenchmarks();

  std::sort(
      localPartitionWaitStats.wallMs.begin(),
      localPartitionWaitStats.wallMs.end());
  VELOX_CHECK(!localPartitionWaitStats.wallMs.empty());

  std::cout
      << "--------------------------------LocalFlat10K-------------------------------"
      << std::endl;
  std::cout << "Wall Time (ms): " << "\n Total: "
            << succinctMicros(localPartitionWallUs)
            << "\n Max: " << localPartitionWaitStats.wallMs.back()
            << "\n Median: "
            << localPartitionWaitStats
                   .wallMs[localPartitionWaitStats.wallMs.size() / 2]
            << "\n Min: " << localPartitionWaitStats.wallMs.front()
            << std::endl;
  std::cout << "LocalPartition: " << localPartitionStatsFlat10K.toString()
            << std::endl;
  sortByAndPrintMax(
      "Producer Wait Time (ms)",
      localPartitionWaitStats.totalProducerWaitMs,
      localPartitionWaitStats.producerWaitMs);
  sortByAndPrintMax(
      "Consumer Wait Time (ms)",
      localPartitionWaitStats.totalConsumerWaitMs,
      localPartitionWaitStats.consumerWaitMs);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();

  bm = std::make_unique<LocalExchangeBenchmark>();
  runBenchmarks();
  bm.reset();

  return 0;
}
