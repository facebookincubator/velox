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

/// Benchmark for cuDF hash join spilling analysis.
///
/// Two modes controlled by --preload:
///
///   --preload=false (default)
///     Full pipeline: TableScan → HashJoin → Aggregation.
///     Reads all parquet files at query time. Measures scan + join.
///
///   --preload=true
///     Pre-loads parquet data into GPU-resident CudfVectors at startup
///     (outside the timed path), then feeds them through a ValuesNode.
///     Measures join only — no scan, no CPU→GPU transfer in the hot path.
///     Use --repeat to control how many times the loaded data is replayed.
///
/// Usage:
///   # Full pipeline (scan + join):
///   ./velox_cudf_hashjoin_spill_benchmark \
///       --data_path=/data/tpch/pq_sf100_f64 --include_results
///
///   # Preloaded (join only, replay 4x):
///   ./velox_cudf_hashjoin_spill_benchmark \
///       --data_path=/data/tpch/pq_sf10_f64 --include_results \
///       --preload --repeat=4

#include "velox/experimental/cudf/benchmarks/CudfBenchmarkHelpers.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/dwio/common/ColumnSelector.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/tpch/gen/TpchGen.h"

DECLARE_string(data_path);
DECLARE_string(data_format);
DECLARE_string(preload);

DEFINE_int32(
    repeat,
    1,
    "Number of times to replay preloaded data through the ValuesNode. "
    "Only used when --preload=true. Increase to simulate larger datasets.");

DEFINE_int32(
    batch_size,
    1024 * 1024 * 1024,
    "Number of bytes per batch when reading parquet files during preload.");

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

/// Extends CudfTpchBenchmark with a purpose-built join-only plan.
///
/// Mode 1 (--preload=false): TableScan-based, reads from disk at query time.
/// Mode 2 (--preload=true): Pre-loads data into GPU CudfVectors, then uses
///   ValuesNode to feed pre-converted GPU data directly into the join.
class CudfHashJoinSpillBenchmark : public CudfTpchBenchmark {
 public:
  void runMain(std::ostream& out, RunStats& runStats) override {
    auto pool = memory::memoryManager()->addLeafPool();
    auto format = dwio::common::toFileFormat(FLAGS_data_format);

    auto lineitemStdCols =
        tpch::getTableSchema(tpch::Table::TBL_LINEITEM)->names();
    auto ordersStdCols = tpch::getTableSchema(tpch::Table::TBL_ORDERS)->names();

    auto lineitemInfo = cudf_velox::readTableInfo(
        "lineitem", FLAGS_data_path, lineitemStdCols, format, pool.get());
    auto ordersInfo = cudf_velox::readTableInfo(
        "orders", FLAGS_data_path, ordersStdCols, format, pool.get());

    VELOX_CHECK(
        !lineitemInfo.dataFiles.empty(), "No lineitem data files found");
    VELOX_CHECK(!ordersInfo.dataFiles.empty(), "No orders data files found");

    auto lineitemRowType =
        dwio::common::ColumnSelector(
            lineitemInfo.type,
            std::vector<std::string>{
                "l_orderkey",
                "l_extendedprice",
                "l_discount",
                "l_quantity",
                "l_partkey"})
            .buildSelectedReordered();
    auto ordersRowType =
        dwio::common::ColumnSelector(
            ordersInfo.type,
            std::vector<std::string>{"o_orderkey", "o_custkey", "o_totalprice"})
            .buildSelectedReordered();

    TpchPlan tpchPlan;

    if (FLAGS_preload != "off") {
      out << "=== cuDF Hash Join Spill Benchmark (preloaded) ===" << std::endl;
      out << "Data path: " << FLAGS_data_path << std::endl;
      out << "Repeat: " << FLAGS_repeat << std::endl;

      out << "Pre-loading lineitem (" << lineitemInfo.dataFiles.size()
          << " files) directly to GPU..." << std::endl;
      auto lineitemGpu = cudf_velox::readParquetIntoCudfVectors(
          lineitemInfo.dataFiles,
          lineitemRowType,
          lineitemInfo.fileColumnNames,
          pool.get(),
          FLAGS_batch_size);
      out << "Pre-loading orders (" << ordersInfo.dataFiles.size()
          << " files) directly to GPU..." << std::endl;
      auto ordersGpu = cudf_velox::readParquetIntoCudfVectors(
          ordersInfo.dataFiles,
          ordersRowType,
          ordersInfo.fileColumnNames,
          pool.get(),
          FLAGS_batch_size);

      int64_t lineitemRows = 0, ordersRows = 0;
      for (const auto& v : lineitemGpu)
        lineitemRows += v->size();
      for (const auto& v : ordersGpu)
        ordersRows += v->size();
      out << "Loaded lineitem: " << lineitemRows << " rows in "
          << lineitemGpu.size() << " batches" << std::endl;
      out << "Loaded orders: " << ordersRows << " rows in " << ordersGpu.size()
          << " batches" << std::endl;

      cudf_velox::registerGpuValuesAdapter();

      out << "Building preloaded plan (repeat=" << FLAGS_repeat << ")..."
          << std::endl;
      tpchPlan = buildPreloadedJoinPlan(
          lineitemGpu,
          ordersGpu,
          lineitemRowType,
          ordersRowType,
          FLAGS_repeat,
          pool.get());
    } else {
      out << "=== cuDF Hash Join Spill Benchmark (scan) ===" << std::endl;
      out << "Data path: " << FLAGS_data_path << std::endl;
      out << "Lineitem files: " << lineitemInfo.dataFiles.size() << std::endl;
      out << "Orders files: " << ordersInfo.dataFiles.size() << std::endl;

      tpchPlan = buildScanJoinPlan(
          lineitemInfo, ordersInfo, lineitemRowType, ordersRowType, pool.get());
    }

    out << "Executing..." << std::endl;
    auto [cursor, results] = run(tpchPlan, queryConfigs_);
    if (!cursor) {
      LOG(ERROR) << "Query terminated with error. Exiting";
      exit(1);
    }

    auto task = cursor->task();
    ensureTaskCompletion(task.get());

    if (FLAGS_include_results) {
      printResults(results, out);
      out << std::endl;
    }

    const auto stats = task->taskStats();
    int64_t rawInputBytes = 0;
    for (auto& pipeline : stats.pipelineStats) {
      auto& first = pipeline.operatorStats[0];
      if (first.operatorType == exec::OperatorType::kTableScan) {
        rawInputBytes += first.rawInputBytes;
      }
    }
    runStats.rawInputBytes = rawInputBytes;

    out << fmt::format(
               "Execution time: {}",
               succinctMillis(
                   stats.executionEndTimeMs - stats.executionStartTimeMs))
        << std::endl;
    out << fmt::format(
               "Splits total: {}, finished: {}",
               stats.numTotalSplits,
               stats.numFinishedSplits)
        << std::endl;
    out << exec::printPlanWithStats(
               *tpchPlan.plan, stats, FLAGS_include_custom_stats)
        << std::endl;
  }

 private:
  /// Builds the plan using TableScan nodes (original mode).
  TpchPlan buildScanJoinPlan(
      const cudf_velox::TableInfo& lineitemInfo,
      const cudf_velox::TableInfo& ordersInfo,
      const RowTypePtr& lineitemRowType,
      const RowTypePtr& ordersRowType,
      memory::MemoryPool* pool) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId ordersScanId;
    core::PlanNodeId lineitemScanId;

    auto ordersScan =
        PlanBuilder(planNodeIdGenerator, pool)
            .tableScan("orders", ordersRowType, ordersInfo.fileColumnNames, {})
            .captureScanNodeId(ordersScanId)
            .planNode();

    auto plan =
        PlanBuilder(planNodeIdGenerator, pool)
            .tableScan(
                "lineitem", lineitemRowType, lineitemInfo.fileColumnNames, {})
            .captureScanNodeId(lineitemScanId)
            .hashJoin(
                {"l_orderkey"},
                {"o_orderkey"},
                ordersScan,
                "",
                {"o_custkey",
                 "o_totalprice",
                 "l_extendedprice",
                 "l_discount",
                 "l_quantity"})
            .partialAggregation(
                {}, {"count(1) as cnt", "sum(l_extendedprice) as total_price"})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .planNode();

    TpchPlan tpchPlan;
    tpchPlan.plan = std::move(plan);
    tpchPlan.dataFiles[ordersScanId] = ordersInfo.dataFiles;
    tpchPlan.dataFiles[lineitemScanId] = lineitemInfo.dataFiles;
    tpchPlan.dataFileFormat = dwio::common::toFileFormat(FLAGS_data_format);
    return tpchPlan;
  }

  /// Builds the plan using ValuesNode with pre-loaded GPU CudfVectors.
  /// The CudfFromVelox passthrough ensures zero CPU→GPU transfer.
  TpchPlan buildPreloadedJoinPlan(
      const std::vector<RowVectorPtr>& lineitemGpu,
      const std::vector<RowVectorPtr>& ordersGpu,
      const RowTypePtr& lineitemRowType,
      const RowTypePtr& ordersRowType,
      int32_t repeatTimes,
      memory::MemoryPool* pool) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    auto ordersSide = PlanBuilder(planNodeIdGenerator, pool)
                          .values(ordersGpu, false, repeatTimes)
                          .planNode();

    auto plan =
        PlanBuilder(planNodeIdGenerator, pool)
            .values(lineitemGpu, false, repeatTimes)
            .hashJoin(
                {"l_orderkey"},
                {"o_orderkey"},
                ordersSide,
                "",
                {"o_custkey",
                 "o_totalprice",
                 "l_extendedprice",
                 "l_discount",
                 "l_quantity"})
            .partialAggregation(
                {}, {"count(1) as cnt", "sum(l_extendedprice) as total_price"})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .planNode();

    TpchPlan tpchPlan;
    tpchPlan.plan = std::move(plan);
    // No dataFiles needed — data comes from ValuesNode.
    tpchPlan.dataFileFormat = dwio::common::toFileFormat(FLAGS_data_format);
    return tpchPlan;
  }
};

int main(int argc, char** argv) {
  std::string kUsage("Benchmarks cuDF hash join for spilling analysis");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};
  benchmark = std::make_unique<CudfHashJoinSpillBenchmark>();
  tpchBenchmarkMain();
}
