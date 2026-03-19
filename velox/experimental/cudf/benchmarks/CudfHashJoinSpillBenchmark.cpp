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

#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/ColumnSelector.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Values.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/tpch/gen/TpchGen.h"

#include <cudf/io/parquet.hpp>

#include <filesystem>

DECLARE_string(data_path);
DECLARE_string(data_format);

DEFINE_bool(
    preload,
    false,
    "Pre-load parquet data into GPU memory before the timed run. "
    "When true, the benchmark measures hash join only (no scan overhead).");

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
    if (!entry.is_regular_file() || entry.path().filename().c_str()[0] == '.') {
      continue;
    }

    if (info.dataFiles.empty()) {
      dwio::common::ReaderOptions readerOptions{pool};
      readerOptions.setFileFormat(format);
      auto readFile = filesystems::getFileSystem(entry.path().string(), nullptr)
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

/// Reads parquet files directly into GPU-resident CudfVectors using cudf's
/// native chunked parquet reader. No CPU intermediate — data goes straight
/// from disk to GPU memory. Column selection uses file column names; the
/// returned CudfVectors carry the standard column names from outputType.
std::vector<RowVectorPtr> readParquetIntoCudfVectors(
    const std::vector<std::string>& files,
    const RowTypePtr& outputType,
    const std::unordered_map<std::string, std::string>& fileColumnNames,
    memory::MemoryPool* pool) {
  // Map standard column names to file column names for projection.
  std::vector<std::string> fileColNames;
  for (size_t i = 0; i < outputType->size(); ++i) {
    const auto& stdName = outputType->nameOf(i);
    auto it = fileColumnNames.find(stdName);
    fileColNames.push_back(it != fileColumnNames.end() ? it->second : stdName);
  }

  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  auto mr = cudf_velox::get_output_mr();

  std::vector<RowVectorPtr> allBatches;

  for (const auto& filePath : files) {
    auto readerOptions = cudf::io::parquet_reader_options::builder(
                             cudf::io::source_info{filePath})
                             .build();
    readerOptions.set_column_names(fileColNames);

    auto reader = cudf::io::chunked_parquet_reader(
        FLAGS_batch_size, 0, readerOptions, stream, mr);

    while (reader.has_next()) {
      auto tableWithMetadata = reader.read_chunk();
      auto& tbl = tableWithMetadata.tbl;
      if (tbl && tbl->num_rows() > 0) {
        auto numRows = tbl->num_rows();
        allBatches.push_back(
            std::make_shared<cudf_velox::CudfVector>(
                pool, outputType, numRows, std::move(tbl), stream));
      }
    }
    stream.synchronize();
  }
  return allBatches;
}

/// Adapter that tells the cuDF operator replacement layer that the Values
/// operator already produces GPU-resident CudfVectors. When active, ToCudf
/// will NOT insert a CudfFromVelox converter after Values — the CudfVectors
/// flow directly into the next GPU operator.
///
/// The adapter inspects the ValuesNode at plan time: canRunOnGPU returns true
/// only when the node's vectors are actually CudfVectors. This makes it safe
/// to register globally — normal CPU Values usage is unaffected.
class GpuValuesAdapter : public cudf_velox::OperatorAdapter {
 public:
  GpuValuesAdapter() : OperatorAdapter("Values") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::Values*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    auto valuesNode =
        std::dynamic_pointer_cast<const core::ValuesNode>(planNode);
    if (!valuesNode || valuesNode->values().empty()) {
      return false;
    }
    return std::dynamic_pointer_cast<cudf_velox::CudfVector>(
               valuesNode->values()[0]) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  bool keepOperator() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {};
  }
};

void registerGpuValuesAdapter() {
  cudf_velox::OperatorAdapterRegistry::getInstance().registerAdapter(
      std::make_unique<GpuValuesAdapter>(), /*overwrite=*/true);
}

} // namespace

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

    auto lineitemInfo = readTableInfo(
        "lineitem", FLAGS_data_path, lineitemStdCols, format, pool.get());
    auto ordersInfo = readTableInfo(
        "orders", FLAGS_data_path, ordersStdCols, format, pool.get());

    VELOX_CHECK(
        !lineitemInfo.dataFiles.empty(), "No lineitem data files found");
    VELOX_CHECK(!ordersInfo.dataFiles.empty(), "No orders data files found");

    auto lineitemRowType = selectColumns(
        lineitemInfo,
        {"l_orderkey",
         "l_extendedprice",
         "l_discount",
         "l_quantity",
         "l_partkey"});
    auto ordersRowType =
        selectColumns(ordersInfo, {"o_orderkey", "o_custkey", "o_totalprice"});

    TpchPlan tpchPlan;

    if (FLAGS_preload) {
      out << "=== cuDF Hash Join Spill Benchmark (preloaded) ===" << std::endl;
      out << "Data path: " << FLAGS_data_path << std::endl;
      out << "Repeat: " << FLAGS_repeat << std::endl;

      out << "Pre-loading lineitem (" << lineitemInfo.dataFiles.size()
          << " files) directly to GPU..." << std::endl;
      auto lineitemGpu = readParquetIntoCudfVectors(
          lineitemInfo.dataFiles,
          lineitemRowType,
          lineitemInfo.fileColumnNames,
          pool.get());
      out << "Pre-loading orders (" << ordersInfo.dataFiles.size()
          << " files) directly to GPU..." << std::endl;
      auto ordersGpu = readParquetIntoCudfVectors(
          ordersInfo.dataFiles,
          ordersRowType,
          ordersInfo.fileColumnNames,
          pool.get());

      int64_t lineitemRows = 0, ordersRows = 0;
      for (const auto& v : lineitemGpu)
        lineitemRows += v->size();
      for (const auto& v : ordersGpu)
        ordersRows += v->size();
      out << "Loaded lineitem: " << lineitemRows << " rows in "
          << lineitemGpu.size() << " batches" << std::endl;
      out << "Loaded orders: " << ordersRows << " rows in " << ordersGpu.size()
          << " batches" << std::endl;

      registerGpuValuesAdapter();

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
      const TableInfo& lineitemInfo,
      const TableInfo& ordersInfo,
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
