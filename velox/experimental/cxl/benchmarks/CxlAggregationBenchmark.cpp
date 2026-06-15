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

// Benchmark for a grouping aggregation, selected with --query:
//
//   q18 (default)
//     SELECT l_orderkey FROM lineitem GROUP BY l_orderkey
//     HAVING SUM(l_quantity) > 312
//   q17
//     SELECT l_partkey, avg(l_quantity) FROM lineitem GROUP BY l_partkey
//   zipf
//     SELECT k, sum(v) FROM <synthesized> GROUP BY k
//
// The three differ in key arrival pattern, not in operator shape. dbgen emits
// lineitem clustered by orderkey, so q18 touches each group's 1-7 rows
// consecutively (~one cold probe per group, then cache hits). l_partkey is
// uniform-random across rows, so q17 probes each of its 200K-per-SF groups
// ~30 times scattered over the whole scan — the locality-adversarial case for
// any payload that has been demoted to CXL. zipf synthesizes a skewed key
// stream (--zipf_groups, --zipf_skew): a few hot groups take most updates in
// random arrival order.
//
// Each query runs across three memory-placement configurations, to measure
// whether building the aggregation in DRAM and relocating to CXL under
// pressure (the CxlHashAggregation operator) beats the alternatives:
//
//   --config=dram       Stock HashAggregation, DRAM pool capped below the group
//                       table, on-disk spill enabled. The "no CXL" competitor.
//   --config=dram_big   Stock HashAggregation, uncapped. The DRAM speed
//   ceiling.
//   --config=interleave Stock HashAggregation, uncapped; run the process under
//                       'numactl --interleave=0,<cxl_node>' so the OS stripes
//                       all pages across DRAM and CXL.
//   --config=cxl        CxlHashAggregation with a real CXL pool; DRAM pool
//   capped
//                       (same as 'dram') so the arbitrator relocates to CXL.
//
// Each config is meant to run as a separate process (the DriverAdapter that
// installs CxlHashAggregation is process-global and registered only for 'cxl',
// and the numactl policy differs per config). See run_cxl_benchmark.sh.

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include "velox/common/config/Config.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/connectors/tpch/TpchConnectorSplit.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/experimental/cxl/CxlHashAggregation.h"
#include "velox/experimental/cxl/CxlMemoryResource.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

DEFINE_string(
    config,
    "dram",
    "Placement configuration: dram | dram_big | interleave | cxl.");
DEFINE_string(
    query,
    "q18",
    "Grouping aggregation: q18 = GROUP BY l_orderkey (clustered keys, one "
    "cold probe per group) | q17 = GROUP BY l_partkey (random keys, ~30 "
    "scattered probes per group) | zipf = synthesized skewed keys "
    "(--zipf_groups, --zipf_skew; random arrival, hot groups).");
DEFINE_int64(
    zipf_groups,
    1'000'000,
    "Number of distinct grouping keys for --query=zipf.");
DEFINE_double(
    zipf_skew,
    1.0,
    "Zipf exponent for --query=zipf: rank r is drawn with probability "
    "proportional to 1/r^skew. 0 = uniform.");
DEFINE_double(scale_factor, 1.0, "TPC-H scale factor.");
DEFINE_int64(
    dram_limit_mb,
    48,
    "Query DRAM pool capacity in MB for the 'dram' and 'cxl' configs. Set "
    "below the group-table size to force spill / relocation.");
DEFINE_int32(
    cxl_numa_node,
    -1,
    "NUMA node id of the CXL device (required for --config=cxl).");
DEFINE_int64(
    cxl_capacity_mb,
    0,
    "CXL pool capacity in MB (required for --config=cxl; size it to the CXL "
    "device). The allocator pre-reserves this, so it must be bounded.");
DEFINE_int32(num_splits, 8, "Number of TPC-H scan splits.");
DEFINE_int32(
    pregen_drivers,
    32,
    "Number of parallel drivers for the untimed --pregen input scan. dbgen "
    "synthesis is the dominant pregen cost and is per-driver, so this speeds "
    "materialization without affecting the single-driver timed trials (the "
    "input is identical regardless of how many drivers built it). The scan "
    "uses 4x this many splits so each driver gets a few, smoothing dbgen's "
    "non-uniform per-split cost. Keep at or below the cores the process is "
    "pinned to (e.g. one NUMA node under 'numactl --cpunodebind').");
DEFINE_int32(num_trials, 5, "Number of measured trials.");
DEFINE_int32(warmup, 1, "Number of warmup trials to discard.");
DEFINE_string(spill_dir, "/tmp/cxl_bench_spill", "Spill directory for 'dram'.");
DEFINE_double(
    having_threshold,
    312.0,
    "HAVING SUM(l_quantity) > threshold (q18 only).");
DEFINE_bool(
    pregen,
    true,
    "Materialize the scanned lineitem columns once before the trials and feed "
    "them from a Values node, so per-trial time measures the aggregation "
    "rather than on-the-fly TPC-H row generation. Disable for scale factors "
    "whose two-column input does not fit in memory (roughly SF >= 100).");

using namespace facebook::velox;
using exec::test::PlanBuilder;

namespace {

constexpr std::string_view kConnectorId{"benchmark-tpch"};

// Per-trial measurements pulled from the finished task's stats.
struct TrialMetrics {
  uint64_t elapsedMs{0};
  uint64_t aggCpuNanos{0};
  uint64_t aggWallNanos{0};
  uint64_t aggBlockedNanos{0};
  uint64_t aggPeakBytes{0};
  // Rows produced by the aggregation operator = groups built, before the
  // HAVING filter cuts the output down.
  uint64_t numGroups{0};
  uint64_t scanIoNanos{0};
  uint64_t spilledBytes{0};
  uint64_t spilledRows{0};
  uint64_t spillWriteNanos{0};
  uint64_t spillReadNanos{0};
  int64_t migrations{0};
  int64_t resultRows{0};
  uint64_t checksum{0};
};

bool isCxlConfig() {
  return FLAGS_config == "cxl";
}

// Caps the query's DRAM pool for the configs that must feel memory pressure.
int64_t dramCapacityBytes() {
  if (FLAGS_config == "dram" || FLAGS_config == "cxl") {
    return FLAGS_dram_limit_mb << 20;
  }
  return memory::kMaxMemory;
}

// Returns the lineitem columns the query scans: the grouping key, then
// l_quantity.
std::vector<std::string> scanColumns() {
  if (FLAGS_query == "q17") {
    return {"l_partkey", "l_quantity"};
  }
  return {"l_orderkey", "l_quantity"};
}

// Appends the selected query's aggregation (and, for q18, the HAVING filter)
// to a source of (<grouping key>, <value>) rows.
PlanBuilder& addAggregation(PlanBuilder& builder) {
  if (FLAGS_query == "zipf") {
    return builder.singleAggregation({"k"}, {"sum(v) AS s"});
  }
  if (FLAGS_query == "q17") {
    return builder.singleAggregation(
        {"l_partkey"}, {"avg(l_quantity) AS avg_quantity"});
  }
  return builder
      .singleAggregation({"l_orderkey"}, {"sum(l_quantity) AS q"})
      // Cast the threshold so it is a DOUBLE literal regardless of how fmt
      // renders it (fmt prints 312.0 as "312", which would parse as BIGINT and
      // fail to match gt(DOUBLE, DOUBLE)).
      .filter(fmt::format("q > cast({} as double)", FLAGS_having_threshold));
}

core::PlanNodePtr buildScanPlan(core::PlanNodeId& scanId) {
  auto builder = PlanBuilder()
                     .tpchTableScan(
                         tpch::Table::TBL_LINEITEM,
                         scanColumns(),
                         FLAGS_scale_factor,
                         kConnectorId)
                     .capturePlanNodeId(scanId);
  return addAggregation(builder).planNode();
}

core::PlanNodePtr buildValuesPlan(const std::vector<RowVectorPtr>& input) {
  auto builder = PlanBuilder().values(input);
  return addAggregation(builder).planNode();
}

// Sums the output grouping-key column into a checksum so the three configs can
// be cross-checked for identical results (guards the silent-null class of
// bugs).
void accumulateResult(
    const std::vector<RowVectorPtr>& results,
    TrialMetrics& metrics) {
  for (const auto& result : results) {
    if (result == nullptr || result->size() == 0) {
      continue;
    }
    metrics.resultRows += result->size();
    // Decode so the checksum is correct regardless of the output vector's
    // encoding (flat, dictionary, constant).
    SelectivityVector rows(result->size());
    DecodedVector decoded(*result->childAt(0), rows);
    for (auto i = 0; i < result->size(); ++i) {
      if (!decoded.isNullAt(i)) {
        metrics.checksum += static_cast<uint64_t>(decoded.valueAt<int64_t>(i));
      }
    }
  }
}

uint64_t runtimeSum(
    const std::unordered_map<std::string, RuntimeMetric>& stats,
    std::string_view key) {
  const auto it = stats.find(std::string(key));
  return it == stats.end() ? 0 : static_cast<uint64_t>(it->second.sum);
}

void collectOperatorStats(const exec::TaskStats& stats, TrialMetrics& metrics) {
  metrics.elapsedMs = stats.executionEndTimeMs - stats.executionStartTimeMs;
  for (const auto& pipeline : stats.pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      // Matches stock "Aggregation" and our "CxlAggregation".
      if (op.operatorType.find("Aggregation") != std::string::npos) {
        metrics.aggCpuNanos += op.addInputTiming.cpuNanos +
            op.getOutputTiming.cpuNanos + op.finishTiming.cpuNanos;
        metrics.aggWallNanos += op.addInputTiming.wallNanos +
            op.getOutputTiming.wallNanos + op.finishTiming.wallNanos;
        metrics.aggBlockedNanos += op.blockedWallNanos;
        metrics.aggPeakBytes = std::max<uint64_t>(
            metrics.aggPeakBytes, op.memoryStats.peakTotalMemoryReservation);
        metrics.numGroups += op.outputPositions;
        metrics.spilledBytes += op.spilledBytes;
        metrics.spilledRows += op.spilledRows;
        metrics.spillWriteNanos +=
            runtimeSum(op.runtimeStats, "spillWriteWallNanos");
        metrics.spillReadNanos +=
            runtimeSum(op.runtimeStats, "spillReadWallNanos");
      } else if (op.operatorType == "TableScan") {
        metrics.scanIoNanos +=
            runtimeSum(op.runtimeStats, "dataSourceReadWallNanos");
      }
    }
  }
}

// Adds 'numSplits' TPC-H scan splits and signals no-more-splits, as required
// before the cursor can drain output. The split count also sets how the table
// is partitioned, so more splits means more, smaller scan units to spread over
// the drivers.
std::function<void(exec::TaskCursor*)> makeSplitAdder(
    const core::PlanNodeId& scanId,
    int32_t numSplits) {
  return [scanId, numSplits](exec::TaskCursor* cursor) {
    if (cursor->noMoreSplits()) {
      return;
    }
    auto& task = cursor->task();
    for (auto part = 0; part < numSplits; ++part) {
      task->addSplit(
          scanId,
          exec::Split(
              std::make_shared<connector::tpch::TpchConnectorSplit>(
                  std::string{kConnectorId}, numSplits, part)));
    }
    task->noMoreSplits(scanId);
    cursor->setNoMoreSplits();
  };
}

// Runs the lineitem scan once, untimed, and returns the materialized
// (<grouping key>, l_quantity) batches allocated from 'outputPool', which must
// outlive the trials.
std::vector<RowVectorPtr> pregenerateInput(
    const std::shared_ptr<memory::MemoryPool>& outputPool,
    folly::Executor* executor) {
  core::PlanNodeId scanId;
  auto plan = PlanBuilder()
                  .tpchTableScan(
                      tpch::Table::TBL_LINEITEM,
                      scanColumns(),
                      FLAGS_scale_factor,
                      kConnectorId)
                  .capturePlanNodeId(scanId)
                  .planNode();

  exec::CursorParameters params;
  params.planNode = plan;
  params.queryCtx = core::QueryCtx::Builder()
                        .executor(executor)
                        .queryId("cxl-bench-pregen")
                        .build();
  // Parallelize the untimed scan: dbgen synthesis dominates here and scales
  // with drivers. The trials stay single-driver; only this materialization is
  // sped up, and the output is identical regardless of driver count.
  params.maxDrivers = FLAGS_pregen_drivers;
  params.copyResult = true;
  params.outputPool = outputPool;

  const auto startMs = getCurrentTimeMs();
  auto [cursor, results] = exec::test::readCursor(
      params, makeSplitAdder(scanId, 4 * FLAGS_pregen_drivers));
  exec::test::waitForTaskCompletion(
      cursor->task().get(), /*maxWaitMicros=*/600'000'000);

  int64_t numRows = 0;
  int64_t numBytes = 0;
  for (const auto& batch : results) {
    numRows += batch->size();
    numBytes += batch->retainedSize();
  }
  VELOX_CHECK(!results.empty());
  // The batches must live in the caller's pool: the cursor (and any pool it
  // created internally) is destroyed when this function returns.
  VELOX_CHECK_EQ(results.front()->pool()->name(), outputPool->name());
  std::cout << fmt::format(
                   "pregenerated {} lineitem rows in {} batches, {:.1f} MB, "
                   "{} ms (excluded from trials)\n",
                   numRows,
                   results.size(),
                   numBytes / static_cast<double>(1 << 20),
                   getCurrentTimeMs() - startMs)
            << std::flush;
  return results;
}

// Synthesizes the --query=zipf input: (k BIGINT, v BIGINT) batches whose keys
// follow a Zipf distribution over --zipf_groups ranks and arrive in random
// order, allocated from 'outputPool', which must outlive the trials. The row
// count matches lineitem at the same scale factor so timings are comparable
// across queries. Per-batch seeds are fixed, so every config and trial
// aggregates identical data and the result checksum must match.
std::vector<RowVectorPtr> generateZipfInput(
    const std::shared_ptr<memory::MemoryPool>& outputPool) {
  const auto numRows = static_cast<int64_t>(6'001'215 * FLAGS_scale_factor);
  const int64_t numGroups = FLAGS_zipf_groups;
  const auto startMs = getCurrentTimeMs();

  // Cumulative distribution over ranks 1..numGroups, rank r weighted 1/r^skew.
  std::vector<double> cdf(numGroups);
  double total = 0;
  for (int64_t rank = 0; rank < numGroups; ++rank) {
    total += 1.0 / std::pow(rank + 1, FLAGS_zipf_skew);
    cdf[rank] = total;
  }
  for (auto& value : cdf) {
    value /= total;
  }

  const auto rowType = ROW({"k", "v"}, {BIGINT(), BIGINT()});
  constexpr int32_t kBatchSize = 4'096;
  std::vector<RowVectorPtr> batches;
  int64_t numBytes = 0;
  for (int64_t begin = 0; begin < numRows; begin += kBatchSize) {
    const auto size = static_cast<vector_size_t>(
        std::min<int64_t>(kBatchSize, numRows - begin));
    std::mt19937_64 rng(42 + begin);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    auto keys = BaseVector::create<FlatVector<int64_t>>(
        BIGINT(), size, outputPool.get());
    auto values = BaseVector::create<FlatVector<int64_t>>(
        BIGINT(), size, outputPool.get());
    for (vector_size_t i = 0; i < size; ++i) {
      const int64_t rank =
          std::upper_bound(cdf.begin(), cdf.end(), uniform(rng)) - cdf.begin();
      // Scatter ranks over the key space so hot keys are not adjacent. The odd
      // multiplier is invertible mod 2^64, so distinct ranks stay distinct.
      keys->set(
          i,
          static_cast<int64_t>(
              static_cast<uint64_t>(rank + 1) * 0x9E3779B97F4A7C15ULL));
      values->set(i, static_cast<int64_t>(uniform(rng) * 100));
    }
    auto batch = std::make_shared<RowVector>(
        outputPool.get(),
        rowType,
        nullptr,
        size,
        std::vector<VectorPtr>{std::move(keys), std::move(values)});
    numBytes += batch->retainedSize();
    batches.push_back(std::move(batch));
  }
  std::cout << fmt::format(
                   "generated {} zipf rows ({} groups, skew {}) in {} "
                   "batches, {:.1f} MB, {} ms (excluded from trials)\n",
                   numRows,
                   numGroups,
                   FLAGS_zipf_skew,
                   batches.size(),
                   numBytes / static_cast<double>(1 << 20),
                   getCurrentTimeMs() - startMs)
            << std::flush;
  return batches;
}

TrialMetrics runTrial(
    const core::PlanNodePtr& plan,
    const std::function<void(exec::TaskCursor*)>& addSplits,
    folly::Executor* executor,
    int32_t trial) {
  auto* manager = memory::memoryManager();
  const auto queryId = fmt::format("{}-{}", FLAGS_config, trial);

  // The CXL resource must outlive the pools that borrow its allocator, so it is
  // declared before the QueryCtx and the cursor.
  std::shared_ptr<memory::CustomMemoryResource> cxlResource;

  auto rootPool = manager->addRootPool(
      queryId, dramCapacityBytes(), memory::MemoryReclaimer::create());

  auto builder =
      core::QueryCtx::Builder().executor(executor).pool(rootPool).queryId(
          queryId);

  if (isCxlConfig()) {
    cxlResource = cxl::makeCxlMemoryResource(
        FLAGS_cxl_numa_node, FLAGS_cxl_capacity_mb << 20);
    auto cxlPool = manager->addCustomRootPool(queryId + ".cxl", cxlResource);
    builder.customPool(std::string{cxl::kCxlResourceTag}, std::move(cxlPool));
  }

  auto queryCtx = builder.build();
  if (isCxlConfig()) {
    auto registry =
        memory::CustomMemoryResourceRegistry::createRegistry(nullptr);
    queryCtx->setRegistry<memory::CustomMemoryResourceRegistry::Registry>(
        memory::kCustomMemoryResourceRegistryKey, registry);
    registry->insert(std::string{cxl::kCxlResourceTag}, cxlResource);
  }

  exec::CursorParameters params;
  params.planNode = plan;
  params.queryCtx = queryCtx;
  params.maxDrivers = 1;
  params.copyResult = true;
  if (FLAGS_config == "dram") {
    params.spillDirectory = FLAGS_spill_dir;
    params.queryConfigs[core::QueryConfig::kSpillEnabled] = "true";
    params.queryConfigs[core::QueryConfig::kAggregationSpillEnabled] = "true";
  }

  cxl::resetCxlHashAggregationCounters();
  auto [cursor, results] = exec::test::readCursor(params, addSplits);
  exec::test::waitForTaskCompletion(
      cursor->task().get(), /*maxWaitMicros=*/600'000'000);

  TrialMetrics metrics;
  collectOperatorStats(cursor->task()->taskStats(), metrics);
  metrics.migrations = cxl::numCxlPartitionsMigrated();
  accumulateResult(results, metrics);
  return metrics;
}

double medianMs(std::vector<uint64_t> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void report(const std::vector<TrialMetrics>& trials) {
  std::vector<uint64_t> elapsed;
  for (const auto& trial : trials) {
    elapsed.push_back(trial.elapsedMs);
  }
  const auto& last = trials.back();

  std::cout << "\n=== CXL aggregation benchmark ===\n";
  std::cout << fmt::format(
                   "config={} query={} scale_factor={} dram_limit_mb={} "
                   "splits={} trials={}",
                   FLAGS_config,
                   FLAGS_query,
                   FLAGS_scale_factor,
                   FLAGS_dram_limit_mb,
                   FLAGS_num_splits,
                   trials.size())
            << "\n";
  if (isCxlConfig()) {
    std::cout << fmt::format(
                     "cxl_numa_node={} cxl_capacity_mb={}",
                     FLAGS_cxl_numa_node,
                     FLAGS_cxl_capacity_mb)
              << "\n";
  }
  if (FLAGS_query == "zipf") {
    std::cout << fmt::format(
                     "zipf_groups={} zipf_skew={}",
                     FLAGS_zipf_groups,
                     FLAGS_zipf_skew)
              << "\n";
  }
  std::cout << fmt::format("median elapsed: {:.1f} ms\n", medianMs(elapsed));
  std::cout << fmt::format(
                   "aggregation: cpu={:.1f} ms wall={:.1f} ms blocked={:.1f} "
                   "ms peak={:.1f} MB\n",
                   last.aggCpuNanos / 1e6,
                   last.aggWallNanos / 1e6,
                   last.aggBlockedNanos / 1e6,
                   last.aggPeakBytes / static_cast<double>(1 << 20))
            << (FLAGS_pregen
                    ? std::string("input: pregenerated Values (untimed)\n")
                    : fmt::format(
                          "scan io: {:.1f} ms\n", last.scanIoNanos / 1e6));
  if (FLAGS_config == "dram") {
    std::cout << fmt::format(
        "spill: bytes={:.1f} MB rows={} write={:.1f} ms read={:.1f} ms\n",
        last.spilledBytes / static_cast<double>(1 << 20),
        last.spilledRows,
        last.spillWriteNanos / 1e6,
        last.spillReadNanos / 1e6);
  }
  if (isCxlConfig()) {
    std::cout << fmt::format("cxl relocations: {}\n", last.migrations);
    if (last.migrations == 0) {
      std::cout << "WARNING: no relocation fired; the DRAM cap did not trigger "
                   "the arbitrator. Lower --dram_limit_mb or verify the CXL "
                   "pool is set.\n";
    }
  }
  std::cout << fmt::format(
                   "result: groups built={} output rows={} checksum={}\n",
                   last.numGroups,
                   last.resultRows,
                   last.checksum)
            << std::flush;
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};

  if (FLAGS_query != "q18" && FLAGS_query != "q17" && FLAGS_query != "zipf") {
    LOG(ERROR) << "--query must be 'q18', 'q17' or 'zipf', got '" << FLAGS_query
               << "'.";
    return 1;
  }
  if (FLAGS_query == "zipf" && !FLAGS_pregen) {
    LOG(ERROR) << "--query=zipf has no scan source; it requires --pregen.";
    return 1;
  }
  if (FLAGS_query == "zipf" && FLAGS_zipf_groups <= 0) {
    LOG(ERROR) << "--query=zipf requires --zipf_groups > 0.";
    return 1;
  }
  if (isCxlConfig() && FLAGS_cxl_numa_node < 0) {
    LOG(ERROR) << "--config=cxl requires --cxl_numa_node to be set to the CXL "
                  "device's NUMA node id.";
    return 1;
  }
  if (isCxlConfig() && FLAGS_cxl_capacity_mb <= 0) {
    LOG(ERROR) << "--config=cxl requires --cxl_capacity_mb > 0; the CXL "
                  "allocator pre-reserves this capacity.";
    return 1;
  }

  memory::SharedArbitrator::registerFactory();
  memory::MemoryManager::Options options;
  options.allocatorCapacity = 64UL << 30;
  options.arbitratorKind = "SHARED";
  options.useMmapAllocator = true;
  memory::MemoryManager::testingSetInstance(options);

  filesystems::registerLocalFileSystem();
  if (!isRegisteredVectorSerde()) {
    serializer::presto::PrestoVectorSerde::registerVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde("Presto")) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();

  connector::tpch::TpchConnectorFactory factory;
  auto tpchConnector = factory.newConnector(
      std::string{kConnectorId},
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{}));
  connector::ConnectorRegistry::global().insert(
      tpchConnector->connectorId(), tpchConnector);

  // Install the operator swap only for the CXL config; the other configs must
  // run the stock HashAggregation.
  if (isCxlConfig()) {
    cxl::registerCxlHashAggregationAdapter();
  }

  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(
      std::thread::hardware_concurrency());

  // Either materialize the input once and aggregate from a Values node, or
  // stream from the connector and let per-trial time include row generation.
  // 'inputPool' is declared before 'input' and 'plan' because both hold
  // vectors whose buffers free into it on destruction; it must die last.
  std::shared_ptr<memory::MemoryPool> inputPool;
  std::vector<RowVectorPtr> input;
  core::PlanNodePtr plan;
  std::function<void(exec::TaskCursor*)> addSplits;
  try {
    if (FLAGS_pregen) {
      inputPool = memory::memoryManager()->addLeafPool("cxl-bench-input");
      input = FLAGS_query == "zipf"
          ? generateZipfInput(inputPool)
          : pregenerateInput(inputPool, executor.get());
      plan = buildValuesPlan(input);
      addSplits = [](exec::TaskCursor* cursor) { cursor->setNoMoreSplits(); };
    } else {
      core::PlanNodeId scanId;
      plan = buildScanPlan(scanId);
      addSplits = makeSplitAdder(scanId, FLAGS_num_splits);
    }

    for (auto i = 0; i < FLAGS_warmup; ++i) {
      runTrial(plan, addSplits, executor.get(), /*trial=*/-1 - i);
    }

    std::vector<TrialMetrics> trials;
    for (auto i = 0; i < FLAGS_num_trials; ++i) {
      trials.push_back(runTrial(plan, addSplits, executor.get(), i));
    }

    report(trials);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Benchmark config '" << FLAGS_config
               << "' failed: " << e.what();
    exec::test::waitForAllTasksToBeDeleted(/*maxWaitUs=*/30'000'000);
    return 1;
  }

  // Tasks are destroyed asynchronously on executor threads and hold the plan
  // — and with --pregen, the input vectors whose buffers free into
  // 'inputPool'. Wait for them so no straggler outlives the pool.
  exec::test::waitForAllTasksToBeDeleted(/*maxWaitUs=*/30'000'000);

  connector::ConnectorRegistry::global().erase(std::string{kConnectorId});
  return 0;
}
