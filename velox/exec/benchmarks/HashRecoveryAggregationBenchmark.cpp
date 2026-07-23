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
#include <iostream>

#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Spill.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/VectorStream.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

DEFINE_bool(
    bm_verbose_stats,
    false,
    "Print per-iteration spill stats for hash recovery aggregation benchmark.");

DECLARE_bool(velox_enable_memory_usage_track_in_default_memory_pool);

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

namespace facebook::velox::exec {
namespace {

class HashRecoveryAggregationBenchmark;

constexpr int64_t kAllocatorCapacityBytes = 8L << 30;
constexpr int64_t kArbitratorCapacityBytes = 6L << 30;
constexpr int64_t kQueryCapacityBytes = 6L << 30;

void initializeBenchmarkMemory() {
  // Align with OperatorTestBase::setupMemory so TestScopedSpillInjection can
  // trigger arbitration reclaim on operators with small working sets.
  FLAGS_velox_enable_memory_usage_track_in_default_memory_pool = true;
  memory::SharedArbitrator::registerFactory();

  memory::MemoryManager::Options options;
  options.allocatorCapacity = kAllocatorCapacityBytes;
  options.arbitratorCapacity = kArbitratorCapacityBytes;
  options.arbitratorKind = "SHARED";
  options.checkUsageLeak = false;
  options.arbitrationStateCheckCb = memoryArbitrationStateCheck;

  using ExtraConfig = memory::SharedArbitrator::ExtraConfig;
  options.extraArbitratorConfigs = {
      {std::string(ExtraConfig::kMemoryPoolInitialCapacity), "512MB"},
      {std::string(ExtraConfig::kMemoryPoolMinReclaimBytes), "0B"},
      {std::string(ExtraConfig::kMemoryPoolMinReclaimPct), "0"},
      {std::string(ExtraConfig::kGlobalArbitrationEnabled), "true"},
  };
  memory::MemoryManager::initialize(options);
}

void registerBenchmarkSerde() {
  if (!isRegisteredVectorSerde()) {
    serializer::presto::PrestoVectorSerde::registerVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde("Presto")) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }
}

struct BenchmarkCase {
  vector_size_t vectorSize;
  int32_t numVectors;
  int32_t numPartitionBits;
  // When true, each row gets a globally unique grouping key to model
  // high-cardinality final aggregation workloads.
  bool uniqueKeys;
};

struct RunStats {
  uint64_t totalUs{0};
  uint64_t inputNs{0};
  uint64_t outputNs{0};
  uint64_t finishNs{0};
  uint64_t spillSortNs{0};
  uint64_t spilledRows{0};
  uint64_t spilledBytes{0};
  uint64_t spillRuns{0};
};

HashRecoveryAggregationBenchmark& benchmarkInstance();

struct BenchmarkPlan {
  core::PlanNodePtr plan;
  core::PlanNodeId aggNodeId;
};

class HashRecoveryAggregationBenchmark : public VectorTestBase {
 public:
  void addBenchmarks(const BenchmarkCase& testCase) {
    addBenchmark(testCase, /*hashRecovery=*/false);
    addBenchmark(testCase, /*hashRecovery=*/true);
    BENCHMARK_DRAW_LINE();
  }

  // Release data-generation pools after benchmarks finish so the arbitrator can
  // tear down cleanly.
  void tearDown() {
    pool_.reset();
    rootPool_.reset();
    executor_.reset();
  }

  uint64_t runIteration(const BenchmarkCase& testCase, bool hashRecovery) {
    folly::BenchmarkSuspender suspender;
    const auto benchmarkPlan = makeBenchmarkPlan(testCase);
    suspender.dismiss();

    RunStats stats;
    run(testCase, hashRecovery, benchmarkPlan, stats);
    if (FLAGS_bm_verbose_stats) {
      std::cout << "Total " << succinctMicros(stats.totalUs) << " Input "
                << succinctNanos(stats.inputNs) << " Output "
                << succinctNanos(stats.outputNs) << " Finish "
                << succinctNanos(stats.finishNs) << " SpillSort "
                << succinctNanos(stats.spillSortNs) << " SpilledRows "
                << stats.spilledRows << " SpilledBytes "
                << succinctBytes(stats.spilledBytes) << " SpillRuns "
                << stats.spillRuns << std::endl;
    }
    return 1;
  }

 private:
  void addBenchmark(const BenchmarkCase& testCase, bool hashRecovery) {
    const auto totalRows =
        static_cast<int64_t>(testCase.vectorSize) * testCase.numVectors;
    const auto name = fmt::format(
        "{}_rows{}_vectors{}_parts{}_{}",
        hashRecovery ? "hash_recovery" : "sorted_merge",
        totalRows,
        testCase.numVectors,
        testCase.numPartitionBits,
        testCase.uniqueKeys ? "unique_keys" : "low_cardinality");

    folly::addBenchmark(__FILE__, name, [testCase, hashRecovery]() {
      return benchmarkInstance().runIteration(testCase, hashRecovery);
    });
  }

  std::vector<RowVectorPtr> makeData(const BenchmarkCase& testCase) {
    const auto rowType =
        ROW({"c0", "c1", "c2", "c3", "c4", "c5"},
            {BIGINT(), INTEGER(), BIGINT(), REAL(), DOUBLE(), VARCHAR()});

    VectorFuzzer::Options opts;
    opts.vectorSize = testCase.vectorSize;
    opts.nullRatio = 0;
    VectorFuzzer fuzzer(opts, pool());

    std::vector<RowVectorPtr> vectors;
    vectors.reserve(testCase.numVectors);
    for (auto i = 0; i < testCase.numVectors; ++i) {
      const auto rowOffset = static_cast<int64_t>(i) * testCase.vectorSize;
      std::vector<VectorPtr> children;
      children.push_back(
          makeFlatVector<int64_t>(testCase.vectorSize, [&](auto row) {
            const auto globalRow = rowOffset + row;
            return testCase.uniqueKeys ? globalRow : globalRow % 128;
          }));
      children.push_back(fuzzer.fuzzFlat(INTEGER()));
      children.push_back(fuzzer.fuzzFlat(BIGINT()));
      children.push_back(fuzzer.fuzzFlat(REAL()));
      children.push_back(fuzzer.fuzzFlat(DOUBLE()));
      children.push_back(fuzzer.fuzzFlat(VARCHAR()));
      vectors.push_back(makeRowVector(rowType->names(), children));
    }
    return vectors;
  }

  BenchmarkPlan makeBenchmarkPlan(const BenchmarkCase& testCase) {
    const auto data = makeData(testCase);
    core::PlanNodeId aggNodeId;
    const auto plan =
        PlanBuilder()
            .values(data)
            .singleAggregation(
                {"c0"}, {"sum(c2)", "max(c4)", "min(c5)", "count(c1)"})
            .capturePlanNodeId(aggNodeId)
            .planNode();
    return {plan, aggNodeId};
  }

  void run(
      const BenchmarkCase& testCase,
      bool hashRecovery,
      const BenchmarkPlan& benchmarkPlan,
      RunStats& stats) {
    const auto spillDir =
        common::testutil::TempDirectoryPath::create()->getPath();

    TestScopedSpillInjection scopedSpillInjection(100);
    const auto startUs = getCurrentTimeMicro();
    std::shared_ptr<Task> task;
    AssertQueryBuilder(benchmarkPlan.plan)
        .spillDirectory(spillDir)
        .maxDrivers(1)
        .maxQueryCapacity(kQueryCapacityBytes)
        .serialExecution(true)
        .config(core::QueryConfig::kSpillEnabled, true)
        .config(core::QueryConfig::kAggregationSpillEnabled, true)
        .config(
            core::QueryConfig::kAggregationSpillHashRecoveryEnabled,
            hashRecovery)
        .config(
            core::QueryConfig::kSpillNumPartitionBits,
            std::to_string(testCase.numPartitionBits))
        .config(core::QueryConfig::kMaxSpillLevel, "0")
        .countResults(task);
    stats.totalUs = getCurrentTimeMicro() - startUs;

    const auto planStats = toPlanStats(task->taskStats());
    const auto& aggStats = planStats.at(benchmarkPlan.aggNodeId);
    stats.inputNs = aggStats.addInputTiming.wallNanos;
    stats.outputNs = aggStats.getOutputTiming.wallNanos;
    stats.finishNs = aggStats.finishTiming.wallNanos;
    stats.spilledRows = aggStats.spilledRows;
    stats.spilledBytes = aggStats.spilledBytes;
    const auto spillRunsIt =
        aggStats.customStats.find(std::string(Operator::kSpillRuns));
    if (spillRunsIt != aggStats.customStats.end()) {
      stats.spillRuns = spillRunsIt->second.sum;
    }
    const auto spillSortIt =
        aggStats.customStats.find(std::string(Operator::kSpillSortTime));
    if (spillSortIt != aggStats.customStats.end()) {
      stats.spillSortNs = spillSortIt->second.sum;
    }

    VELOX_CHECK_GT(
        stats.spilledBytes,
        0,
        "Expected forced spill in benchmark run '{}'",
        hashRecovery ? "hash_recovery" : "sorted_merge");
    VELOX_CHECK_GT(
        stats.spillRuns, 0, "Expected at least one spill run in benchmark");
  }
};

HashRecoveryAggregationBenchmark& benchmarkInstance() {
  static HashRecoveryAggregationBenchmark instance;
  return instance;
}

} // namespace

HashRecoveryAggregationBenchmark& hashRecoveryAggregationBenchmark() {
  return benchmarkInstance();
}

} // namespace facebook::velox::exec

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  initializeBenchmarkMemory();
  registerBenchmarkSerde();
  aggregate::prestosql::registerAllAggregateFunctions();
  filesystems::registerLocalFileSystem();

  auto& benchmark = facebook::velox::exec::hashRecoveryAggregationBenchmark();
  // High-cardinality workloads with forced spill. Each case runs both the
  // sorted-merge recovery path and the hash-recovery path for comparison.
  benchmark.addBenchmarks(
      {.vectorSize = 4'096,
       .numVectors = 32,
       .numPartitionBits = 2,
       .uniqueKeys = true});
  benchmark.addBenchmarks(
      {.vectorSize = 8'192,
       .numVectors = 32,
       .numPartitionBits = 3,
       .uniqueKeys = true});
  benchmark.addBenchmarks(
      {.vectorSize = 4'096,
       .numVectors = 16,
       .numPartitionBits = 2,
       .uniqueKeys = false});

  folly::runBenchmarks();
  exec::test::waitForAllTasksToBeDeleted();
  benchmark.tearDown();
  memory::SharedArbitrator::unregisterFactory();
  return 0;
}
