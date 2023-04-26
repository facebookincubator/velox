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

#include <folly/init/Init.h>
#include <vector>

#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/duckdb_reader/ParquetReader.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

DECLARE_int32(split_preload_per_driver);

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::parquet;

namespace facebook::velox::exec::test {
namespace {

class ParquetTpchTestBase : public testing::Test {
 public:
  ParquetTpchTestBase(ParquetReaderType parquetReaderType)
      : parquetReaderType_(parquetReaderType) {}

  // Set up a DuckDB instance for the entire suite and load TPC-H data with
  // scale factor 0.01.
  void SetUp() {
    if (duckDb_ == nullptr) {
      duckDb_ = std::make_shared<DuckDbQueryRunner>();
      constexpr double kTpchScaleFactor = 0.01;
      duckDb_->initializeTpch(kTpchScaleFactor);
    }
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();

    parse::registerTypeResolver();
    filesystems::registerLocalFileSystem();
    if (parquetReaderType_ == ParquetReaderType::DUCKDB) {
      FLAGS_split_preload_per_driver = 0;
    }
    registerParquetReaderFactory(parquetReaderType_);

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(kHiveConnectorId, nullptr);
    connector::registerConnector(hiveConnector);
    tempDirectory_ = TempDirectoryPath::create();
    saveTpchTablesAsParquet();
    tpchBuilder_.initialize(tempDirectory_->path);
  }

  void assertQuery(
      int queryId,
      std::optional<std::vector<uint32_t>> sortingKeys = {}) const {
    auto tpchPlan = tpchBuilder_.getQueryPlan(queryId);
    auto duckDbSql = duckDb_->getTpchQuery(queryId);
    auto task = assertQuery(tpchPlan, duckDbSql, sortingKeys);
  }

  void TearDown() {
    connector::unregisterConnector(kHiveConnectorId);
    unregisterParquetReaderFactory();
  }

 private:
  /// Write TPC-H tables as a Parquet file to temp directory in hive-style
  /// partition
  void saveTpchTablesAsParquet() {
    constexpr int kRowGroupSize = 10'000;
    const auto tableNames = tpchBuilder_.getTableNames();
    for (const auto& tableName : tableNames) {
      auto tableDirectory =
          fmt::format("{}/{}", tempDirectory_->path, tableName);
      fs::create_directory(tableDirectory);
      auto filePath = fmt::format("{}/file.parquet", tableDirectory);
      auto query = fmt::format(
          fmt::runtime(kDuckDbParquetWriteSQL_.at(tableName)),
          tableName,
          filePath,
          kRowGroupSize);
      duckDb_->execute(query);
    }
  }

  std::shared_ptr<Task> assertQuery(
      const TpchPlan& tpchPlan,
      const std::string& duckQuery,
      std::optional<std::vector<uint32_t>> sortingKeys) const {
    bool noMoreSplits = false;
    constexpr int kNumSplits = 10;
    constexpr int kNumDrivers = 4;
    auto addSplits = [&](Task* task) {
      if (!noMoreSplits) {
        for (const auto& entry : tpchPlan.dataFiles) {
          for (const auto& path : entry.second) {
            auto const splits = HiveConnectorTestBase::makeHiveConnectorSplits(
                path, kNumSplits, tpchPlan.dataFileFormat);
            for (const auto& split : splits) {
              task->addSplit(entry.first, Split(split));
            }
          }
          task->noMoreSplits(entry.first);
        }
      }
      noMoreSplits = true;
    };
    CursorParameters params;
    params.maxDrivers = kNumDrivers;
    params.planNode = tpchPlan.plan;

    return exec::test::assertQuery(
        params, addSplits, duckQuery, *duckDb_, sortingKeys);
  }

  const ParquetReaderType parquetReaderType_;
  std::shared_ptr<DuckDbQueryRunner> duckDb_ = nullptr;
  std::shared_ptr<TempDirectoryPath> tempDirectory_ = nullptr;
  TpchQueryBuilder tpchBuilder_ =
      TpchQueryBuilder(dwio::common::FileFormat::PARQUET);
  const std::unordered_map<std::string, std::string> kDuckDbParquetWriteSQL_ = {
      std::make_pair(
          "lineitem",
          R"(COPY (SELECT * FROM {})
         TO '{}'(FORMAT 'parquet', CODEC 'ZSTD', ROW_GROUP_SIZE {}))"),
      std::make_pair(
          "orders",
          R"(COPY (SELECT * FROM {})
         TO '{}' (FORMAT 'parquet', CODEC 'ZSTD', ROW_GROUP_SIZE {}))"),
      std::make_pair(
          "customer",
          R"(COPY (SELECT * FROM {})
         TO '{}' (FORMAT 'parquet', CODEC 'ZSTD', ROW_GROUP_SIZE {}))"),
      std::make_pair(
          "nation",
          R"(COPY (SELECT * FROM {})
          TO '{}' (FORMAT 'parquet', CODEC 'ZSTD', ROW_GROUP_SIZE {}))"),
      std::make_pair(
          "region",
          R"(COPY (SELECT * FROM {})
         TO '{}' (FORMAT 'parquet', CODEC 'ZSTD', ROW_GROUP_SIZE {}))"),
      std::make_pair(
          "part",
          R"(COPY (SELECT * FROM {})
         TO '{}' (FORMAT 'parquet', CODEC 'ZSTD', ROW_GROUP_SIZE {}))"),
      std::make_pair(
          "supplier",
          R"(COPY (SELECT * FROM {})
         TO '{}' (FORMAT 'parquet', CODEC 'ZSTD', ROW_GROUP_SIZE {}))"),
      std::make_pair(
          "partsupp",
          R"(COPY (SELECT * FROM {})
         TO '{}' (FORMAT 'parquet', CODEC 'ZSTD', ROW_GROUP_SIZE {}))")};
};

} // namespace
} // namespace facebook::velox::exec::test
