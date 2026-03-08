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
/// Runs a minimal lineitem JOIN orders query designed to maximize GPU memory
/// pressure on both the build side (orders hash table) and probe side
/// (lineitem stream). Output is a single aggregate row (COUNT + SUM) so
/// results are minimal but not truncated.
///
/// Usage:
///   ./velox_cudf_hashjoin_spill_benchmark \
///       --data_path=/data/tpch/pq_sf100_f64 \
///       --include_results \
///       --cudf_memory_resource=pool \
///       --cudf_memory_percent=50

#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/ColumnSelector.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/tpch/gen/TpchGen.h"

#include <filesystem>

DECLARE_string(data_path);
DECLARE_string(data_format);

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace fs = std::filesystem;

namespace {

struct TableInfo {
  RowTypePtr type;
  std::vector<std::string> dataFiles;
  std::unordered_map<std::string, std::string> fileColumnNames;
};

/// Reads parquet schema from the first file in dataPath/tableName/ and maps
/// standard TPC-H column names to file column names by ordinal position.
TableInfo readTableInfo(
    const std::string& tableName,
    const std::string& dataPath,
    const std::vector<std::string>& standardColumns,
    dwio::common::FileFormat format,
    memory::MemoryPool* pool) {
  TableInfo info;

  const fs::path tablePath{dataPath + "/" + tableName};
  for (auto const& entry : fs::directory_iterator{tablePath}) {
    if (!entry.is_regular_file() ||
        entry.path().filename().c_str()[0] == '.') {
      continue;
    }

    if (info.dataFiles.empty()) {
      dwio::common::ReaderOptions readerOptions{pool};
      readerOptions.setFileFormat(format);
      auto readFile =
          filesystems::getFileSystem(entry.path().string(), nullptr)
              ->openFileForRead(entry.path().string());
      std::shared_ptr<ReadFile> sharedFile;
      sharedFile.reset(readFile.release());
      auto input =
          std::make_unique<dwio::common::BufferedInput>(sharedFile, *pool);
      auto reader = dwio::common::getReaderFactory(format)->createReader(
          std::move(input), readerOptions);
      auto fileType = reader->rowType();
      auto fileNames = fileType->names();

      VELOX_CHECK_GE(fileNames.size(), standardColumns.size());
      for (size_t i = 0; i < standardColumns.size(); ++i) {
        info.fileColumnNames[standardColumns[i]] = fileNames[i];
      }

      auto types = fileType->children();
      types.resize(standardColumns.size());
      auto colNames = standardColumns;
      info.type =
          std::make_shared<RowType>(std::move(colNames), std::move(types));
    }

    info.dataFiles.push_back(entry.path().string());
  }

  std::sort(info.dataFiles.begin(), info.dataFiles.end());
  return info;
}

RowTypePtr selectColumns(
    const TableInfo& table,
    const std::vector<std::string>& columns) {
  auto selector =
      std::make_shared<dwio::common::ColumnSelector>(table.type, columns);
  return selector->buildSelectedReordered();
}

} // namespace

/// Extends CudfTpchBenchmark with a purpose-built join-only plan.
/// The plan scans lineitem (probe) and orders (build), joins on
/// l_orderkey = o_orderkey, and aggregates to a single COUNT(*) + SUM row.
/// Extra columns from both sides are pulled through the join to inflate the
/// hash table and probe buffers, stressing GPU memory.
class CudfHashJoinSpillBenchmark : public CudfTpchBenchmark {
 public:
  void runMain(std::ostream& out, RunStats& runStats) override {
    auto pool = memory::memoryManager()->addLeafPool();
    auto format = dwio::common::toFileFormat(FLAGS_data_format);

    auto lineitemStdCols =
        tpch::getTableSchema(tpch::Table::TBL_LINEITEM)->names();
    auto ordersStdCols =
        tpch::getTableSchema(tpch::Table::TBL_ORDERS)->names();

    auto lineitemInfo = readTableInfo(
        "lineitem", FLAGS_data_path, lineitemStdCols, format, pool.get());
    auto ordersInfo = readTableInfo(
        "orders", FLAGS_data_path, ordersStdCols, format, pool.get());

    VELOX_CHECK(
        !lineitemInfo.dataFiles.empty(), "No lineitem data files found");
    VELOX_CHECK(!ordersInfo.dataFiles.empty(), "No orders data files found");

    out << "=== cuDF Hash Join Spill Benchmark ===" << std::endl;
    out << "Data path: " << FLAGS_data_path << std::endl;
    out << "Lineitem files: " << lineitemInfo.dataFiles.size() << std::endl;
    out << "Orders files: " << ordersInfo.dataFiles.size() << std::endl;

    auto tpchPlan = buildJoinPlan(lineitemInfo, ordersInfo, pool.get());

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
  /// Builds: SELECT count(*), sum(l_extendedprice) FROM lineitem l
  ///         JOIN orders o ON l.l_orderkey = o.o_orderkey
  ///
  /// Build side (orders): o_orderkey, o_custkey, o_totalprice
  ///   ~150M rows at SF100 -- forces a large GPU hash table
  ///
  /// Probe side (lineitem): l_orderkey, l_extendedprice, l_discount,
  ///   l_quantity, l_partkey
  ///   ~600M rows at SF100 -- large probe-side memory pressure
  ///
  /// Join output carries columns from both sides so they must be materialized
  /// in GPU memory. The final aggregation collapses everything to 1 row.
  TpchPlan buildJoinPlan(
      const TableInfo& lineitemInfo,
      const TableInfo& ordersInfo,
      memory::MemoryPool* pool) {
    std::vector<std::string> ordersCols = {
        "o_orderkey", "o_custkey", "o_totalprice"};
    auto ordersRowType = selectColumns(ordersInfo, ordersCols);

    std::vector<std::string> lineitemCols = {
        "l_orderkey",
        "l_extendedprice",
        "l_discount",
        "l_quantity",
        "l_partkey"};
    auto lineitemRowType = selectColumns(lineitemInfo, lineitemCols);

    auto planNodeIdGenerator =
        std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId ordersScanId;
    core::PlanNodeId lineitemScanId;

    auto ordersScan =
        PlanBuilder(planNodeIdGenerator, pool)
            .tableScan(
                "orders",
                ordersRowType,
                ordersInfo.fileColumnNames,
                {})
            .captureScanNodeId(ordersScanId)
            .planNode();

    auto plan =
        PlanBuilder(planNodeIdGenerator, pool)
            .tableScan(
                "lineitem",
                lineitemRowType,
                lineitemInfo.fileColumnNames,
                {})
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
                {},
                {"count(1) as cnt",
                 "sum(l_extendedprice) as total_price"})
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
};

int main(int argc, char** argv) {
  std::string kUsage(
      "Benchmarks cuDF hash join for spilling analysis.\n"
      "Run with --data_path=<tpch_dir> --include_results\n"
      "Use --cudf_memory_percent to control GPU memory limit for spill "
      "testing.\n"
      "Example: ./velox_cudf_hashjoin_spill_benchmark "
      "--data_path=/data/tpch/pq_sf100_f64 --include_results "
      "--cudf_memory_resource=pool --cudf_memory_percent=30");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};
  benchmark = std::make_unique<CudfHashJoinSpillBenchmark>();
  tpchBenchmarkMain();
}
