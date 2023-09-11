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
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/tpch/gen/TpchGen.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::parquet;

namespace facebook::velox::exec::test {
namespace {

static constexpr char const* kTpchConnectorId{"test-tpch"};

class ParquetTpchTestBase : public testing::Test,
                            public ::facebook::velox::test::VectorTestBase {
 public:
  ParquetTpchTestBase() {}

  // Setup a DuckDB instance for the entire suite and load TPC-H data with scale
  // factor 0.01.
  static void SetUpTestSuite() {
    if (duckDb_ == nullptr) {
      duckDb_ = std::make_shared<DuckDbQueryRunner>();
      constexpr double kTpchScaleFactor = 0.01;
      duckDb_->initializeTpch(kTpchScaleFactor);
    }
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();

    parse::registerTypeResolver();
    filesystems::registerLocalFileSystem();
    registerParquetReaderFactory();

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(kHiveConnectorId, nullptr);
    connector::registerConnector(hiveConnector);
    auto tpchConnector =
        connector::getConnectorFactory(
            connector::tpch::TpchConnectorFactory::kTpchConnectorName)
            ->newConnector(kTpchConnectorId, nullptr);
    connector::registerConnector(tpchConnector);
    tempDirectory_ = TempDirectoryPath::create();
    std::shared_ptr<memory::MemoryPool> rootPool{
        memory::defaultMemoryManager().addRootPool()};
    std::shared_ptr<memory::MemoryPool> pool{rootPool->addLeafChild("leaf")};
    saveTpchTablesAsParquet(pool.get());
    tpchBuilder_.initialize(tempDirectory_->path);
  }

  void assertQuery(
      int queryId,
      std::optional<std::vector<uint32_t>> sortingKeys = {}) const {
    auto tpchPlan = tpchBuilder_.getQueryPlan(queryId);
    auto duckDbSql = duckDb_->getTpchQuery(queryId);
    auto task = assertQuery(tpchPlan, duckDbSql, sortingKeys);
  }

  static void TearDownTestSuite() {
    connector::unregisterConnector(kHiveConnectorId);
    connector::unregisterConnector(kTpchConnectorId);
    unregisterParquetReaderFactory();
  }

 private:
  /// Write TPC-H tables as a Parquet file to temp directory in hive-style
  /// partition
  static void saveTpchTablesAsParquet(memory::MemoryPool* pool) {
    const auto tableNames = {
        tpch::Table::TBL_PART,
        tpch::Table::TBL_SUPPLIER,
        tpch::Table::TBL_PARTSUPP,
        tpch::Table::TBL_CUSTOMER,
        tpch::Table::TBL_ORDERS,
        tpch::Table::TBL_LINEITEM,
        tpch::Table::TBL_NATION,
        tpch::Table::TBL_REGION};

    for (const auto& tableName : tableNames) {
      auto tableDirectory =
          fmt::format("{}/{}", tempDirectory_->path, toTableName(tableName));
      fs::create_directory(tableDirectory);
      const auto outputRow = getTableSchema(tableName);
      auto names = outputRow->names();

      auto plan =
          PlanBuilder().tableScan(tableName, std::move(names), 0.01).planNode();
      auto split =
          exec::Split(std::make_shared<connector::tpch::TpchConnectorSplit>(
              kTpchConnectorId, 1, 0));

      auto results = AssertQueryBuilder(plan).splits({split}).copyResults(pool);

      auto tableHandle = std::make_shared<core::InsertTableHandle>(
          kHiveConnectorId,
          HiveConnectorTestBase::makeHiveInsertTableHandle(
              outputRow->names(),
              outputRow->children(),
              {},
              {},
              HiveConnectorTestBase::makeLocationHandle(
                  tableDirectory,
                  std::nullopt,
                  connector::hive::LocationHandle::TableType::kNew),
              dwio::common::FileFormat::PARQUET));

      auto scanPlan = PlanBuilder().values({results});
      plan = scanPlan
                 .tableWrite(
                     scanPlan.planNode()->outputType(),
                     outputRow->names(),
                     nullptr,
                     tableHandle,
                     false,
                     connector::CommitStrategy::kNoCommit)
                 .planNode();

      AssertQueryBuilder(plan).copyResults(pool);
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

  static std::shared_ptr<DuckDbQueryRunner> duckDb_;
  static std::shared_ptr<TempDirectoryPath> tempDirectory_;
  static TpchQueryBuilder tpchBuilder_;
};

std::shared_ptr<DuckDbQueryRunner> ParquetTpchTestBase::duckDb_ = nullptr;
std::shared_ptr<TempDirectoryPath> ParquetTpchTestBase::tempDirectory_ =
    nullptr;
TpchQueryBuilder ParquetTpchTestBase::tpchBuilder_ =
    TpchQueryBuilder(dwio::common::FileFormat::PARQUET);

} // namespace
} // namespace facebook::velox::exec::test
