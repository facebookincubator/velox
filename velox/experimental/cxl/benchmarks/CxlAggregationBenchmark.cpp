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

// Microbenchmark for a grouping aggregation over a synthesized Zipf key stream:
//
//   SELECT k, sum(v) FROM <synthesized> GROUP BY k
//
// Keys follow a Zipf distribution over --zipf_groups ranks (--zipf_skew), so a
// few hot groups take most updates in random arrival order — the hot/cold split
// that DRAM-to-CXL tiering targets. --scale_factor sizes the input (1 = ~1GB).
//
// The same input runs across memory-placement configurations to measure whether
// building the table in DRAM and relocating to CXL under pressure (the
// HashAggregation relocate path) beats the alternatives:
//
//   --config=dram       HashAggregation, DRAM pool capped at --dram_limit_mb,
//                       on-disk spill enabled. The "no CXL" competitor. Set a
//                       cap above the group table for the no-pressure DRAM
//                       speed ceiling.
//   --config=interleave HashAggregation, uncapped; run the process under
//                       'numactl --interleave=0,<cxl_node>' so the OS stripes
//                       pages across DRAM and CXL.
//   --config=cxl        HashAggregation with a real CXL tier pool registered on
//                       the query (customPool "cxl"); DRAM pool capped (same as
//                       'dram'). Under pressure reclaim() relocates the payload
//                       to CXL instead of disk spilling (either/or); if the CXL
//                       pool is exhausted the query fails rather than spilling.
//
// Each config is meant to run as a separate process (the numactl policy differs
// per config, and the 'cxl' config registers a per-query CXL tier pool). See
// run_cxl_benchmark.sh.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/Timer.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/GroupingSet.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
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
    "Placement configuration: dram | interleave | cxl.");
DEFINE_int64(zipf_groups, 1'000'000, "Number of distinct grouping keys.");
DEFINE_double(
    zipf_skew,
    1.0,
    "Zipf exponent: rank r is drawn with probability proportional to "
    "1/r^zipf_skew. 0 is uniform; higher is more skewed.");
DEFINE_double(
    scale_factor,
    1.0,
    "Input size: scale_factor = 1 is ~1GB of (key, value) data.");
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
DEFINE_int32(num_trials, 5, "Number of measured trials.");
DEFINE_int32(warmup, 1, "Number of warmup trials to discard.");
DEFINE_int32(
    num_drivers,
    1,
    "Local parallelism for the aggregation. 1 runs the serial single-stage "
    "plan; >1 splits the input across that many source pipelines, "
    "repartitions by key, and aggregates on that many drivers.");
DEFINE_string(
    spill_dir,
    "/tmp/cxl_bench_spill",
    "Spill directory for 'dram' and 'cxl'.");

using namespace facebook::velox;
using exec::test::PlanBuilder;

namespace {

// Counts GroupingSet::relocate() calls (DRAM -> CXL relocations) across a
// trial, incremented by a TestValue hook registered in main(). Reset before
// each trial.
std::atomic<int64_t> gRelocations{0};

// Per-trial measurements pulled from the finished task's stats.
struct TrialMetrics {
  uint64_t elapsedMs{0};
  uint64_t aggCpuNanos{0};
  uint64_t aggWallNanos{0};
  uint64_t aggBlockedNanos{0};
  uint64_t aggPeakBytes{0};
  // Rows produced by the aggregation operator = groups built.
  uint64_t numGroups{0};
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

// Splits 'input' batches round-robin into 'numChunks' disjoint groups so each
// source pipeline aggregates its own slice with no overlap or duplication.
std::vector<std::vector<RowVectorPtr>> splitInput(
    const std::vector<RowVectorPtr>& input,
    int32_t numChunks) {
  std::vector<std::vector<RowVectorPtr>> chunks(numChunks);
  for (auto i = 0; i < input.size(); ++i) {
    chunks[i % numChunks].push_back(input[i]);
  }
  return chunks;
}

core::PlanNodePtr buildPlan(const std::vector<RowVectorPtr>& input) {
  if (FLAGS_num_drivers <= 1) {
    return PlanBuilder()
        .values(input)
        .singleAggregation({"k"}, {"sum(v) AS s"})
        .planNode();
  }
  // Parallel plan: each disjoint input slice is read by its own (single-driver)
  // source pipeline, then 'localPartition' repartitions the rows by key so each
  // of the 'num_drivers' consumer drivers sees every row for the keys it owns
  // and runs a complete aggregation. Equivalent result to the serial plan; each
  // complete aggregation relocates to the CXL tier under pressure.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto chunks = splitInput(input, FLAGS_num_drivers);
  std::vector<core::PlanNodePtr> sources;
  sources.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    sources.push_back(
        PlanBuilder(planNodeIdGenerator).values(chunk).planNode());
  }
  return PlanBuilder(planNodeIdGenerator)
      .localPartition({"k"}, sources)
      .singleAggregation({"k"}, {"sum(v) AS s"})
      .planNode();
}

// Sums the output grouping-key column into a checksum so the configs can be
// cross-checked for identical results (guards the silent-null class of bugs).
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
      }
    }
  }
}

// Synthesizes the input: (k BIGINT, v BIGINT) batches whose keys follow a Zipf
// distribution over --zipf_groups ranks and arrive in random order, allocated
// from 'outputPool', which must outlive the trials. Per-batch seeds are fixed,
// so every config and trial sees identical input.
std::vector<RowVectorPtr> generateZipfInput(
    const std::shared_ptr<memory::MemoryPool>& outputPool) {
  // Size the input by target bytes: scale_factor = 1 is ~1GB of (k, v) data.
  constexpr int64_t kInputBytesPerRow = sizeof(int64_t) * 2;
  const auto numRows = static_cast<int64_t>(FLAGS_scale_factor * (1LL << 30)) /
      kInputBytesPerRow;
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
  const int64_t numBatches = (numRows + kBatchSize - 1) / kBatchSize;
  std::vector<RowVectorPtr> batches(numBatches);

  // Fill the independent batches in parallel. Each batch is seeded only by its
  // own row offset (42 + begin), so the data is bit-identical to a serial run
  // and every config still sees the same input; 'outputPool' is thread-safe and
  // each thread writes disjoint batch slots. This keeps generation (excluded
  // from the measured trials) from dominating wall time at high scale factors.
  const int64_t numThreads = std::min<int64_t>(
      numBatches, std::max<unsigned>(1u, std::thread::hardware_concurrency()));
  const int64_t batchesPerThread = (numBatches + numThreads - 1) / numThreads;
  std::vector<int64_t> threadBytes(numThreads, 0);
  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for (int64_t t = 0; t < numThreads; ++t) {
    const int64_t firstBatch = t * batchesPerThread;
    const int64_t lastBatch =
        std::min<int64_t>(firstBatch + batchesPerThread, numBatches);
    if (firstBatch >= lastBatch) {
      break;
    }
    threads.emplace_back([&, t, firstBatch, lastBatch]() {
      for (int64_t batch = firstBatch; batch < lastBatch; ++batch) {
        const int64_t begin = batch * kBatchSize;
        const auto size = static_cast<vector_size_t>(
            std::min<int64_t>(kBatchSize, numRows - begin));
        std::mt19937_64 rng(42 + begin);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        auto keys = BaseVector::create<FlatVector<int64_t>>(
            BIGINT(), size, outputPool.get());
        auto values = BaseVector::create<FlatVector<int64_t>>(
            BIGINT(), size, outputPool.get());
        for (vector_size_t i = 0; i < size; ++i) {
          const int64_t globalRow = begin + i;
          // Seed pass: the first 'numGroups' rows emit each group exactly once,
          // so the full key space is always populated and the group-table size
          // is held constant across skew (skew then varies only the access
          // concentration of the remaining rows, not how many groups exist).
          // Requires numRows >= numGroups, which the configured scale factors
          // satisfy. Remaining rows draw their rank from the Zipf CDF.
          const int64_t rank = globalRow < numGroups
              ? globalRow
              : std::upper_bound(cdf.begin(), cdf.end(), uniform(rng)) -
                  cdf.begin();
          // Scatter ranks over the key space so hot keys are not adjacent. The
          // odd multiplier is invertible mod 2^64, so distinct ranks stay
          // distinct.
          keys->set(
              i,
              static_cast<int64_t>(
                  static_cast<uint64_t>(rank + 1) * 0x9E3779B97F4A7C15ULL));
          values->set(i, static_cast<int64_t>(uniform(rng) * 100));
        }
        batches[batch] = std::make_shared<RowVector>(
            outputPool.get(),
            rowType,
            nullptr,
            size,
            std::vector<VectorPtr>{std::move(keys), std::move(values)});
        threadBytes[t] += batches[batch]->retainedSize();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  int64_t numBytes = 0;
  for (const auto bytes : threadBytes) {
    numBytes += bytes;
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
  params.maxDrivers = std::max(1, FLAGS_num_drivers);
  params.copyResult = true;

  if (FLAGS_config == "dram" || isCxlConfig()) {
    params.spillDirectory = FLAGS_spill_dir;
    params.queryConfigs[core::QueryConfig::kSpillEnabled] = "true";
    params.queryConfigs[core::QueryConfig::kAggregationSpillEnabled] = "true";
  }

  gRelocations.store(0, std::memory_order_relaxed);
  auto [cursor, results] = exec::test::readCursor(
      params, [](exec::TaskCursor* cursor) { cursor->setNoMoreSplits(); });
  exec::test::waitForTaskCompletion(
      cursor->task().get(), /*maxWaitMicros=*/600'000'000);

  TrialMetrics metrics;
  collectOperatorStats(cursor->task()->taskStats(), metrics);
  metrics.migrations = gRelocations.load(std::memory_order_relaxed);
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
                   "config={} scale_factor={} zipf_groups={} zipf_skew={} "
                   "dram_limit_mb={} trials={}",
                   FLAGS_config,
                   FLAGS_scale_factor,
                   FLAGS_zipf_groups,
                   FLAGS_zipf_skew,
                   FLAGS_dram_limit_mb,
                   trials.size())
            << "\n";
  if (isCxlConfig()) {
    std::cout << fmt::format(
                     "cxl_numa_node={} cxl_capacity_mb={}",
                     FLAGS_cxl_numa_node,
                     FLAGS_cxl_capacity_mb)
              << "\n";
  }
  std::cout << fmt::format("median elapsed: {:.1f} ms\n", medianMs(elapsed));
  std::cout << fmt::format(
      "aggregation: cpu={:.1f} ms wall={:.1f} ms blocked={:.1f} "
      "ms peak={:.1f} MB\n",
      last.aggCpuNanos / 1e6,
      last.aggWallNanos / 1e6,
      last.aggBlockedNanos / 1e6,
      last.aggPeakBytes / static_cast<double>(1 << 20));
  // On 'cxl' a non-zero figure means the CXL pool overflowed to disk.
  if (FLAGS_config == "dram" || isCxlConfig()) {
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

  if (FLAGS_zipf_groups <= 0) {
    LOG(ERROR) << "--zipf_groups must be > 0.";
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

  // The CXL config relocates inside HashAggregation when a "cxl" tier pool is
  // present on the query (configured per trial). Count relocations via a
  // TestValue hook for reporting.
  if (isCxlConfig()) {
    common::testutil::TestValue::enable();
    common::testutil::TestValue::set<exec::GroupingSet>(
        "facebook::velox::exec::GroupingSet::relocate", [](exec::GroupingSet*) {
          gRelocations.fetch_add(1, std::memory_order_relaxed);
        });
  }

  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(
      std::thread::hardware_concurrency());

  // 'inputPool' is declared before 'input' and 'plan' because both hold vectors
  // whose buffers free into it on destruction; it must die last.
  std::shared_ptr<memory::MemoryPool> inputPool;
  std::vector<RowVectorPtr> input;
  core::PlanNodePtr plan;
  try {
    inputPool = memory::memoryManager()->addLeafPool("cxl-bench-input");
    input = generateZipfInput(inputPool);
    plan = buildPlan(input);

    for (auto i = 0; i < FLAGS_warmup; ++i) {
      runTrial(plan, executor.get(), /*trial=*/-1 - i);
    }

    std::vector<TrialMetrics> trials;
    for (auto i = 0; i < FLAGS_num_trials; ++i) {
      trials.push_back(runTrial(plan, executor.get(), i));
    }

    report(trials);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Benchmark config '" << FLAGS_config
               << "' failed: " << e.what();
    exec::test::waitForAllTasksToBeDeleted(/*maxWaitUs=*/30'000'000);
    return 1;
  }

  // Tasks are destroyed asynchronously on executor threads and hold the plan
  // and the input vectors whose buffers free into 'inputPool'. Wait for them so
  // no straggler outlives the pool.
  exec::test::waitForAllTasksToBeDeleted(/*maxWaitUs=*/30'000'000);
  return 0;
}
