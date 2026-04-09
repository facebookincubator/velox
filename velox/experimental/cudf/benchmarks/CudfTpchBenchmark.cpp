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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/benchmarks/CudfBenchmarkHelpers.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"
#include "velox/experimental/cudf/benchmarks/PreloadedScanOperator.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/tpch/gen/TpchGen.h"

#include <experimental/cudf/connectors/hive/CudfHiveConnector.h>

DECLARE_string(data_path);
DECLARE_string(data_format);
DECLARE_int64(max_coalesced_bytes);
DECLARE_string(max_coalesced_distance_bytes);
DECLARE_int32(parquet_prefetch_rowgroups);

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

DEFINE_bool(
    cudf_use_buffered_input,
    false,
    "Use buffered input for CudfHive connector (kUseBufferedInput).");

DEFINE_uint64(
    cudf_chunk_read_limit,
    0,
    "Output table chunk read limit for cudf::parquet_chunked_reader.");

DEFINE_uint64(
    cudf_pass_read_limit,
    0,
    "Pass read limit for cudf::parquet_chunked_reader.");

DEFINE_int32(
    cudf_gpu_batch_size_rows,
    100000,
    "Preferred output batch size in rows for cudf operators.");

DEFINE_string(
    cudf_memory_resource,
    "async",
    "Memory resource for cudf operators.");

DEFINE_int32(
    cudf_memory_percent,
    50,
    "Percentage of GPU memory to allocate for cudf operators.");

DEFINE_bool(velox_cudf_table_scan, true, "Enable cuDF table scan");

DEFINE_bool(cudf_debug_enabled, false, "Enable debug printing");

DEFINE_string(
    preload,
    "off",
    "Pre-load all TPC-H tables and serve from memory instead of disk. "
    "Values: off (default), gpu (read directly to GPU via cuDF), "
    "cpu (read to CPU RowVectors, converted to GPU on demand).");

DEFINE_int32(
    preload_batch_size,
    512 * 1024 * 1024,
    "Batch size in bytes when reading parquet during preload.");

void CudfTpchBenchmark::initialize() {
  TpchBenchmark::initialize();

  if (FLAGS_velox_cudf_table_scan) {
    connector::ConnectorRegistry::global().erase(
        facebook::velox::exec::test::kHiveConnectorId);

    // Add new values into the cudfHive configuration...
    auto cudfHiveConfigurationValues =
        std::unordered_map<std::string, std::string>();
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInput] =
            std::to_string(FLAGS_cudf_use_buffered_input);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxChunkReadLimit] =
            std::to_string(FLAGS_cudf_chunk_read_limit);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxPassReadLimit] =
            std::to_string(FLAGS_cudf_pass_read_limit);
    cudfHiveConfigurationValues[cudf_velox::connector::hive::CudfHiveConfig::
                                    kAllowMismatchedCudfHiveSchemas] =
        std::to_string(true);
    auto cudfHiveProperties = std::make_shared<const config::ConfigBase>(
        std::move(cudfHiveConfigurationValues));

    // Create cudfHive connector with config...
    cudf_velox::connector::hive::CudfHiveConnectorFactory cudfHiveFactory;
    auto cudfHiveConnector = cudfHiveFactory.newConnector(
        facebook::velox::exec::test::kHiveConnectorId,
        cudfHiveProperties,
        ioExecutor_.get());
    connector::ConnectorRegistry::global().insert(
        cudfHiveConnector->connectorId(), cudfHiveConnector);
  }

  cudf_velox::CudfConfig::getInstance().memoryResource =
      FLAGS_cudf_memory_resource;
  cudf_velox::CudfConfig::getInstance().memoryPercent =
      FLAGS_cudf_memory_percent;

  cudf_velox::CudfConfig::getInstance().debugEnabled = FLAGS_cudf_debug_enabled;
  // Enable cuDF operators
  cudf_velox::registerCudf();

  if (FLAGS_preload != "off") {
    ensurePreloaded();
  }

  // Add custom configs
  queryConfigs_[facebook::velox::cudf_velox::CudfFromVelox::kGpuBatchSizeRows] =
      std::to_string(FLAGS_cudf_gpu_batch_size_rows);
}

std::shared_ptr<config::ConfigBase>
CudfTpchBenchmark::makeConnectorProperties() {
  auto cfg = TpchBenchmark::makeConnectorProperties();
  using CudfHiveCfg = cudf_velox::connector::hive::CudfHiveConfig;

  // CuDF-specific properties.
  cfg->set(
      CudfHiveCfg::kMaxChunkReadLimit,
      std::to_string(FLAGS_cudf_chunk_read_limit));
  cfg->set(
      CudfHiveCfg::kMaxPassReadLimit,
      std::to_string(FLAGS_cudf_pass_read_limit));
  cfg->set(CudfHiveCfg::kAllowMismatchedCudfHiveSchemas, "true");

  return cfg;
}

std::vector<std::shared_ptr<connector::ConnectorSplit>>
CudfTpchBenchmark::listSplits(
    const std::string& path,
    int32_t numSplitsPerFile,
    const exec::test::TpchPlan& plan) {
  // TODO (dm): Figure out a way to enforce 1 split per file in
  // CudfHiveDataSource outside of this benchmark
  if (FLAGS_velox_cudf_table_scan) {
    // TODO (dm): Instead of this, we can maybe use
    // makeHiveConnectorSplits(vector<shared_ptr<TempFilePath>>&
    // filePaths)
    std::vector<std::shared_ptr<connector::ConnectorSplit>> result;
    auto temp = HiveConnectorTestBase::makeHiveConnectorSplits(
        path, 1, plan.dataFileFormat);
    for (auto& i : temp) {
      result.push_back(i);
    }
    return result;
  }

  return TpchBenchmark::listSplits(path, numSplitsPerFile, plan);
}

void CudfTpchBenchmark::ensurePreloaded() {
  if (preloaded_ || FLAGS_preload == "off") {
    return;
  }
  preloadPool_ = memory::memoryManager()->addLeafPool();
  auto* pool = preloadPool_.get();
  auto format = toFileFormat(FLAGS_data_format);

  static const std::vector<std::pair<std::string, tpch::Table>> kTables = {
      {"lineitem", tpch::Table::TBL_LINEITEM},
      {"orders", tpch::Table::TBL_ORDERS},
      {"customer", tpch::Table::TBL_CUSTOMER},
      {"part", tpch::Table::TBL_PART},
      {"partsupp", tpch::Table::TBL_PARTSUPP},
      {"supplier", tpch::Table::TBL_SUPPLIER},
      {"nation", tpch::Table::TBL_NATION},
      {"region", tpch::Table::TBL_REGION},
  };

  auto& store = cudf_velox::PreloadedTableStore::getInstance();
  for (const auto& [tableName, table] : kTables) {
    auto schema = tpch::getTableSchema(table);
    auto stdCols = schema->names();
    auto info = cudf_velox::readTableInfo(
        tableName, FLAGS_data_path, stdCols, format, pool);
    if (info.dataFiles.empty()) {
      continue;
    }
    auto gpuVectors = cudf_velox::readParquetIntoCudfVectors(
        info.dataFiles,
        info.type,
        info.fileColumnNames,
        pool,
        FLAGS_preload_batch_size);

    if (FLAGS_preload == "cpu") {
      auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
      auto mr = cudf_velox::get_output_mr();
      std::vector<RowVectorPtr> cpuVectors;
      cpuVectors.reserve(gpuVectors.size());
      for (auto& v : gpuVectors) {
        auto cudfVec = std::dynamic_pointer_cast<cudf_velox::CudfVector>(v);
        auto cpuRow = cudf_velox::with_arrow::toVeloxColumn(
            cudfVec->getTableView(), pool, info.type, stream, mr);
        cpuVectors.push_back(std::move(cpuRow));
      }
      stream.synchronize();
      store.store(tableName, std::move(cpuVectors));
    } else {
      store.store(tableName, std::move(gpuVectors));
    }
  }
  cudf_velox::registerPreloadedTableScanAdapter();
  preloaded_ = true;
}

void CudfTpchBenchmark::shutdown() {
  cudf_velox::PreloadedTableStore::getInstance().clear();
  preloadPool_.reset();
  cudf_velox::unregisterCudf();
  TpchBenchmark::shutdown();
}
