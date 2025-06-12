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

#include "velox/experimental/cudf/connectors/parquet/ParquetConfig.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetConnector.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/ParquetConnectorTestBase.h"

#include "velox/benchmarks/tpch/TpchBenchmark.h"

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

class CudfTpchBenchmark : public TpchBenchmark {
 public:
  void initialize() override {
    TpchBenchmark::initialize();

    // Add new values into the parquet configuration...
    auto parquetConfigurationValues =
        std::unordered_map<std::string, std::string>();
    parquetConfigurationValues
        [cudf_velox::connector::parquet::ParquetConfig::kMaxChunkReadLimit] =
            std::to_string(FLAGS_cudf_chunk_read_limit);
    parquetConfigurationValues
        [cudf_velox::connector::parquet::ParquetConfig::kMaxPassReadLimit] =
            std::to_string(FLAGS_cudf_pass_read_limit);
    parquetConfigurationValues[cudf_velox::connector::parquet::ParquetConfig::
                                   kAllowMismatchedParquetSchemas] =
        std::to_string(true);
    auto parquetProperties = std::make_shared<const config::ConfigBase>(
        std::move(parquetConfigurationValues));

    // Create parquet connector with config...
    connector::registerConnectorFactory(
        std::make_shared<
            cudf_velox::connector::parquet::ParquetConnectorFactory>());
    auto parquetConnector =
        connector::getConnectorFactory(
            cudf_velox::connector::parquet::ParquetConnectorFactory::
                kParquetConnectorName)
            ->newConnector(
                cudf_velox::exec::test::kParquetConnectorId,
                parquetProperties,
                ioExecutor_.get());
    connector::registerConnector(parquetConnector);

    // Enable cuDF operators
    cudf_velox::registerCudf();

    queryConfigs
        [facebook::velox::cudf_velox::CudfFromVelox::kGpuBatchSizeRows] =
            std::to_string(FLAGS_cudf_gpu_batch_size_rows);
  }

  std::vector<std::shared_ptr<connector::ConnectorSplit>> listSplits(
      const std::string& path,
      int32_t numSplitsPerFile,
      const exec::test::TpchPlan& plan) override {
    if (facebook::velox::cudf_velox::cudfIsRegistered() &&
        facebook::velox::connector::getAllConnectors().count(
            cudf_velox::exec::test::kParquetConnectorId) > 0 &&
        facebook::velox::cudf_velox::cudfTableScanEnabled()) {
      std::vector<std::shared_ptr<connector::ConnectorSplit>> result;
      auto temp = cudf_velox::exec::test::ParquetConnectorTestBase::
          makeParquetConnectorSplits(path, 1);
      for (auto& i : temp) {
        result.push_back(i);
      }
      return result;
    }

    return TpchBenchmark::listSplits(path, numSplitsPerFile, plan);
  }

  void shutdown() override {
    cudf_velox::unregisterCudf();
    connector::unregisterConnector(cudf_velox::exec::test::kParquetConnectorId);
    connector::unregisterConnectorFactory(
        cudf_velox::connector::parquet::ParquetConnectorFactory::
            kParquetConnectorName);
    TpchBenchmark::shutdown();
  }
};

int main(int argc, char** argv) {
  std::string kUsage(
      "This program benchmarks TPC-H queries. Run 'velox_tpch_benchmark -helpon=TpchBenchmark' for available options.\n");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};
  benchmark = std::make_unique<CudfTpchBenchmark>();
  tpchBenchmarkMain();
}
