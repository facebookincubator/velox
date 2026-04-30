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
#include "velox/experimental/cudf/benchmarks/CudfTpcdsBenchmark.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/CudfTpcdsQueryBuilder.h"

#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

DECLARE_string(data_path);
DECLARE_string(data_format);
DECLARE_int64(max_coalesced_bytes);
DECLARE_string(max_coalesced_distance_bytes);
DECLARE_int32(parquet_prefetch_rowgroups);

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

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

DEFINE_bool(velox_cudf_table_scan, true, "Enable cuDF table scan");

DEFINE_string(
    cudf_properties,
    "",
    "Path to a properties file for CudfConfig. Each line should be key=value "
    "(e.g. cudf.memory_resource=async). See CudfConfig for available keys.");

void CudfTpcdsBenchmark::initQueryBuilder() {
  auto cudfBuilder =
      std::make_unique<cudf_velox::exec::test::CudfTpcdsQueryBuilder>(
          toFileFormat(FLAGS_data_format), ioExecutor_.get());
  cudfBuilder->enableCudf();
  cudfBuilder->initialize(FLAGS_data_path);
  queryBuilder_ = std::move(cudfBuilder);
}

void CudfTpcdsBenchmark::initialize() {
  // Must be set before TpcdsBenchmark::initialize(), which calls
  // initQueryBuilder() -> registerCudf(), which bakes the prefix into the
  // step-aware aggregation registry.
  auto& config = cudf_velox::CudfConfig::getInstance();
  config.functionNamePrefix = kPrestoFunctionNamespacePrefix;

  if (!FLAGS_cudf_properties.empty()) {
    config.initialize(cudf_velox::loadPropertiesFile(FLAGS_cudf_properties));
  }

  TpcdsBenchmark::initialize();

  const std::string prestoConnectorId{kPrestoHiveConnectorId};
  if (FLAGS_velox_cudf_table_scan) {
    // The base class registered a HiveConnector. Unregister it so the
    // CudfTpcdsQueryBuilder can register CudfHiveConnector under the plan's
    // connector ID instead. The query builder handles this when getQueryPlan()
    // is called.
    if (connector::ConnectorRegistry::tryGet(prestoConnectorId) != nullptr) {
      connector::ConnectorRegistry::global().erase(prestoConnectorId);
    }

    // Re-register with CuDF properties.
    auto properties = makeConnectorProperties();
    cudf_velox::connector::hive::CudfHiveConnectorFactory cudfHiveFactory;
    auto cudfHiveConnector = cudfHiveFactory.newConnector(
        prestoConnectorId, properties, ioExecutor_.get());
    connector::ConnectorRegistry::global().insert(
        cudfHiveConnector->connectorId(), cudfHiveConnector);
  }

  queryConfigs_[cudf_velox::CudfFromVelox::kGpuBatchSizeRows] =
      std::to_string(FLAGS_cudf_gpu_batch_size_rows);
}

std::shared_ptr<config::ConfigBase>
CudfTpcdsBenchmark::makeConnectorProperties() {
  auto cfg = TpcdsBenchmark::makeConnectorProperties();
  using CudfHiveCfg = cudf_velox::connector::hive::CudfHiveConfig;

  cfg->set(
      CudfHiveCfg::kMaxChunkReadLimit,
      std::to_string(FLAGS_cudf_chunk_read_limit));
  cfg->set(
      CudfHiveCfg::kMaxPassReadLimit,
      std::to_string(FLAGS_cudf_pass_read_limit));
  cfg->set(CudfHiveCfg::kAllowMismatchedCudfHiveSchemas, "true");

  return cfg;
}

void CudfTpcdsBenchmark::shutdown() {
  // CudfTpcdsQueryBuilder::shutdown() handles unregisterCudf().
  TpcdsBenchmark::shutdown();
}

int main(int argc, char** argv) {
  std::string kUsage(
      "This program benchmarks TPC-DS queries with CuDF GPU acceleration.\n"
      "  --data_path        Path to TPC-DS data directory\n"
      "  --plan_path        Path to plan JSON directory\n"
      "  --cudf_properties  Path to CudfConfig properties file\n");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};
  tpcdsBenchmark = std::make_unique<CudfTpcdsBenchmark>();
  tpcdsBenchmarkMain();
}
