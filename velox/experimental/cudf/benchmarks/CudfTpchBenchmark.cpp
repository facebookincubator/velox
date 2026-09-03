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
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"
#include "velox/experimental/cudf/connectors/hive/CudfDecodedColumnCache.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include <experimental/cudf/connectors/hive/CudfHiveConnector.h>

DECLARE_int64(max_coalesced_bytes);
DECLARE_string(max_coalesced_distance_bytes);
DECLARE_int32(num_repeats);
DECLARE_int32(parquet_prefetch_rowgroups);
DECLARE_int32(run_query_verbose);

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
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

DEFINE_uint64(
    cudf_local_exchange_buffer_size,
    1UL << 30,
    "Maximum buffered bytes per local exchange before applying backpressure.");

DEFINE_bool(velox_cudf_table_scan, true, "Enable cuDF table scan");

DEFINE_bool(
    cudf_hive_use_buffered_input,
    true,
    "Use Velox BufferedInput for cuDF Hive reads.");

DEFINE_bool(
    cudf_hive_use_experimental_reader,
    false,
    "Use the cuDF experimental hybrid Parquet reader.");

DEFINE_bool(
    cudf_hive_use_decoded_column_cache,
    false,
    "Use the experimental process-lifetime decoded Parquet column cache. "
    "This also marks benchmark input files immutable.");

DEFINE_string(
    cudf_hive_decoded_column_cache_compression,
    "none",
    "Decoded column cache storage codec: none, column, or column-advanced.");

DEFINE_uint64(
    cudf_hive_decoded_column_cache_max_pinned_bytes,
    cudf_velox::connector::hive::CudfDecodedColumnCache::kMaxPinnedBytes,
    "Maximum bytes in the experimental decoded column cache pinned pool. The "
    "default remains 70 GiB.");

DEFINE_bool(
    cudf_benchmark_nvtx_query_ranges,
    false,
    "Wrap each verbose TPC-H query iteration in an NVTX range containing the "
    "query and iteration numbers.");

DEFINE_string(
    cudf_properties,
    "",
    "Path to a properties file for CudfConfig. Each line should be key=value "
    "(e.g. cudf.memory_resource=async). See CudfConfig for available keys.");

namespace {

class RepeatFlagRestorer {
 public:
  explicit RepeatFlagRestorer(int32_t value) : value_(value) {}

  ~RepeatFlagRestorer() {
    FLAGS_num_repeats = value_;
  }

 private:
  int32_t value_;
};

} // namespace

void CudfTpchBenchmark::initialize() {
  cudf_velox::connector::hive::CudfDecodedColumnCache::
      configureMaxPinnedBytes(
          FLAGS_cudf_hive_decoded_column_cache_max_pinned_bytes);

  if (!FLAGS_cudf_properties.empty()) {
    cudf_velox::CudfConfig::getInstance().initialize(
        cudf_velox::loadPropertiesFile(FLAGS_cudf_properties));
  }

  TpchBenchmark::initialize();

  if (FLAGS_velox_cudf_table_scan) {
    connector::ConnectorRegistry::global().erase(
        facebook::velox::exec::test::kHiveConnectorId);

    auto cudfHiveConfigurationValues =
        std::unordered_map<std::string, std::string>();
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxChunkReadLimit] =
            std::to_string(FLAGS_cudf_chunk_read_limit);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxPassReadLimit] =
            std::to_string(FLAGS_cudf_pass_read_limit);
    cudfHiveConfigurationValues[cudf_velox::connector::hive::CudfHiveConfig::
                                    kAllowMismatchedCudfHiveSchemas] =
        std::to_string(true);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInput] =
            std::to_string(FLAGS_cudf_hive_use_buffered_input);
    cudfHiveConfigurationValues[cudf_velox::connector::hive::CudfHiveConfig::
                                    kUseExperimentalCudfReader] =
        std::to_string(FLAGS_cudf_hive_use_experimental_reader);
    cudfHiveConfigurationValues[cudf_velox::connector::hive::CudfHiveConfig::
                                    kExperimentalDecodedColumnCacheEnabled] =
        std::to_string(FLAGS_cudf_hive_use_decoded_column_cache);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::
             kExperimentalDecodedColumnCacheCompression] =
            FLAGS_cudf_hive_decoded_column_cache_compression;
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kImmutableFiles] =
            std::to_string(FLAGS_cudf_hive_use_decoded_column_cache);
    auto cudfHiveProperties = std::make_shared<const config::ConfigBase>(
        std::move(cudfHiveConfigurationValues));

    cudf_velox::connector::hive::CudfHiveConnectorFactory cudfHiveFactory;
    auto cudfHiveConnector = cudfHiveFactory.newConnector(
        facebook::velox::exec::test::kHiveConnectorId,
        cudfHiveProperties,
        ioExecutor_.get());
    connector::ConnectorRegistry::global().insert(
        cudfHiveConnector->connectorId(), cudfHiveConnector);
  }

  cudf_velox::registerCudf();
  cudf_velox::registerPrestoFunctions(
      cudf_velox::CudfConfig::getInstance().functionNamePrefix);

  queryConfigs_[facebook::velox::cudf_velox::CudfFromVelox::kGpuBatchSizeRows] =
      std::to_string(FLAGS_cudf_gpu_batch_size_rows);
  queryConfigs_[core::QueryConfig::kMaxLocalExchangeBufferSize] =
      std::to_string(FLAGS_cudf_local_exchange_buffer_size);
}

void CudfTpchBenchmark::runMain(
    std::ostream& out,
    facebook::velox::RunStats& runStats) {
  if (not FLAGS_cudf_benchmark_nvtx_query_ranges) {
    TpchBenchmark::runMain(out, runStats);
    return;
  }

  VELOX_USER_CHECK_GT(
      FLAGS_run_query_verbose,
      0,
      "NVTX query iteration ranges require --run_query_verbose=<query>");
  VELOX_USER_CHECK_GT(
      FLAGS_num_repeats,
      0,
      "NVTX query iteration ranges require --num_repeats > 0");

  const auto numRepeats = FLAGS_num_repeats;
  RepeatFlagRestorer restoreRepeats{numRepeats};
  FLAGS_num_repeats = 1;
  for (int32_t iteration = 1; iteration <= numRepeats; ++iteration) {
    const auto label = fmt::format(
        "TPC-H Q{:02d} iteration {}/{} ({})",
        FLAGS_run_query_verbose,
        iteration,
        numRepeats,
        iteration == 1 ? "cold" : "hot");
    const auto color =
        iteration == 1 ? nvtx3::rgb{65, 105, 225} : nvtx3::rgb{34, 139, 34};
    const nvtx3::event_attributes attributes{
        label, color, nvtx3::payload{iteration}};
    const nvtx3::scoped_range_in<facebook::velox::cudf_velox::VeloxDomain>
        range{attributes};
    const auto before =
        cudf_velox::connector::hive::CudfDecodedColumnCache::instance().stats();
    TpchBenchmark::runMain(out, runStats);
    if (FLAGS_cudf_hive_use_decoded_column_cache) {
      const auto after =
          cudf_velox::connector::hive::CudfDecodedColumnCache::instance()
              .stats();
      const auto insertedUncompressed =
          after.insertedUncompressedBytes - before.insertedUncompressedBytes;
      const auto insertedStored =
          after.insertedStoredBytes - before.insertedStoredBytes;
      out << fmt::format(
          "decoded-cache iteration={} compression={} max_pinned_bytes={} "
          "pinned_bytes={} "
          "inserted_uncompressed_bytes={} inserted_stored_bytes={} "
          "compressed_ranges={} raw_ranges={} compression_attempts={} "
          "encode_ms={:.3f} restore_calls={} restored_stored_bytes={} "
          "restored_uncompressed_bytes={} decompress_ms={:.3f}\n",
          iteration,
          FLAGS_cudf_hive_decoded_column_cache_compression,
          after.maxPinnedBytes,
          after.pinnedBytes,
          insertedUncompressed,
          insertedStored,
          after.insertedCompressedRanges - before.insertedCompressedRanges,
          after.insertedRawRanges - before.insertedRawRanges,
          after.compressionAttempts - before.compressionAttempts,
          static_cast<double>(
              after.compressionEncodeNanos - before.compressionEncodeNanos) /
              1'000'000.0,
          after.restoreCalls - before.restoreCalls,
          after.restoredStoredBytes - before.restoredStoredBytes,
          after.restoredUncompressedBytes - before.restoredUncompressedBytes,
          static_cast<double>(
              after.decompressionNanos - before.decompressionNanos) /
              1'000'000.0);
    }
  }
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
  cfg->set(
      CudfHiveCfg::kUseBufferedInput,
      std::to_string(FLAGS_cudf_hive_use_buffered_input));
  cfg->set(
      CudfHiveCfg::kUseExperimentalCudfReader,
      std::to_string(FLAGS_cudf_hive_use_experimental_reader));
  cfg->set(
      CudfHiveCfg::kExperimentalDecodedColumnCacheEnabled,
      std::to_string(FLAGS_cudf_hive_use_decoded_column_cache));
  cfg->set(
      CudfHiveCfg::kExperimentalDecodedColumnCacheCompression,
      FLAGS_cudf_hive_decoded_column_cache_compression);
  cfg->set(
      CudfHiveCfg::kImmutableFiles,
      std::to_string(FLAGS_cudf_hive_use_decoded_column_cache));

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

void CudfTpchBenchmark::shutdown() {
  cudf_velox::unregisterCudf();
  TpchBenchmark::shutdown();
}

int main(int argc, char** argv) {
  std::string kUsage(
      "This program benchmarks TPC-H queries. Run 'velox_cudf_tpch_benchmark -helpon=TpchBenchmark' for available options.\n");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};
  benchmark = std::make_unique<CudfTpchBenchmark>();
  tpchBenchmarkMain();
}
