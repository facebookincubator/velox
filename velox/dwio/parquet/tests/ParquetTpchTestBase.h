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
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/tpch/gen/TpchGen.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::parquet;

namespace facebook::velox::exec::test {
namespace {

class ParquetTpchTestBase : public testing::Test {
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
    auto duckDbSql = tpch::getQuery(queryId);
    auto task = assertQuery(tpchPlan, duckDbSql, sortingKeys);
  }

  static void TearDownTestSuite() {
    connector::unregisterConnector(kHiveConnectorId);
    unregisterParquetReaderFactory();
  }

 private:
  /// Write TPC-H tables as a Parquet file to temp directory in hive-style
  /// partition
  static void saveTpchTablesAsParquet(memory::MemoryPool* pool) {
    static constexpr auto tables = {
        tpch::Table::TBL_PART,
        tpch::Table::TBL_SUPPLIER,
        tpch::Table::TBL_PARTSUPP,
        tpch::Table::TBL_CUSTOMER,
        tpch::Table::TBL_ORDERS,
        tpch::Table::TBL_LINEITEM,
        tpch::Table::TBL_NATION,
        tpch::Table::TBL_REGION};

    for (const auto& table : tables) {
      auto tableName = std::string(toTableName(table));
      auto tableDirectory =
          fmt::format("{}/{}", tempDirectory_->path, tableName);

      auto tableSchema = velox::tpch::getTableSchema(table);
      auto tt = kDuckDbParquetWriteSQL_.at(tableName);
      auto query = fmt::format(
          fmt::runtime(kDuckDbParquetWriteSQL_.at(tableName)), tableName);
      // TODO: The name of API `executeOrdered` is confused, it do nothing
      // related to sorting, and should be updated with a more proper name.
      auto result = duckDb_->executeOrdered(query, tableSchema);
      auto ptr = velox::test::VectorMaker::rowVector(tableSchema, result, pool);
      result.size();

      auto tableHandle = std::make_shared<core::InsertTableHandle>(
          kHiveConnectorId,
          HiveConnectorTestBase::makeHiveInsertTableHandle(
              tableSchema->names(),
              tableSchema->children(),
              {},
              {},
              HiveConnectorTestBase::makeLocationHandle(
                  tableDirectory,
                  std::nullopt,
                  connector::hive::LocationHandle::TableType::kNew),
              dwio::common::FileFormat::PARQUET));

      auto scanPlan = PlanBuilder().values({ptr});
      auto plan = scanPlan
                      .tableWrite(
                          scanPlan.planNode()->outputType(),
                          tableSchema->names(),
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
  static std::unordered_map<std::string, const std::string>
      kDuckDbParquetWriteSQL_;
};

std::shared_ptr<DuckDbQueryRunner> ParquetTpchTestBase::duckDb_ = nullptr;
std::shared_ptr<TempDirectoryPath> ParquetTpchTestBase::tempDirectory_ =
    nullptr;
TpchQueryBuilder ParquetTpchTestBase::tpchBuilder_ =
    TpchQueryBuilder(dwio::common::FileFormat::PARQUET);

std::unordered_map<std::string, const std::string>
    ParquetTpchTestBase::kDuckDbParquetWriteSQL_ = {
        std::make_pair(
            "lineitem",
            R"(SELECT l_orderkey,
                      l_partkey,
                      l_suppkey,
                      l_linenumber,
                      L_QUANTITY::DOUBLE AS quantity,
                      L_EXTENDEDPRICE::DOUBLE AS extendedprice,
                      L_DISCOUNT::DOUBLE AS discount,
                      L_TAX::DOUBLE AS tax,
                      l_returnflag,
                      l_linestatus,
                      l_shipdate AS shipdate,
                      l_commitdate,
                      l_receiptdate,
                      l_shipinstruct,
                      l_shipmode,
                      l_comment
              FROM {})"),
        std::make_pair(
            "orders",
            R"(SELECT o_orderkey,
                      o_custkey,
                      o_orderstatus,
                      o_totalprice::DOUBLE as o_totalprice,
                      o_orderdate,
                      o_orderpriority,
                      o_clerk,
                      o_shippriority,
                      o_comment
              FROM {})"),
        std::make_pair(
            "customer",
            R"(SELECT c_custkey,
                      c_name,
                      c_address,
                      c_nationkey,
                      c_phone,
                      c_acctbal::DOUBLE as c_acctbal,
                      c_mktsegment,
                      c_comment
              FROM {})"),
        std::make_pair("nation", R"(SELECT * FROM {})"),
        std::make_pair("region", R"(SELECT * FROM {})"),
        std::make_pair(
            "part",
            R"(SELECT p_partkey,
                      p_name,
                      p_mfgr,
                      p_brand,
                      p_type,
                      p_size,
                      p_container,
                      p_retailprice::DOUBLE,
                      p_comment
                FROM {})"),
        std::make_pair(
            "supplier",
            R"(SELECT s_suppkey,
                      s_name,
                      s_address,
                      s_nationkey,
                      s_phone,
                      s_acctbal::DOUBLE,
                      s_comment
                FROM {})"),
        std::make_pair(
            "partsupp",
            R"(SELECT ps_partkey,
                      ps_suppkey,
                      ps_availqty,
                      ps_supplycost::DOUBLE as supplycost,
                      ps_comment
                FROM {})")};

} // namespace
} // namespace facebook::velox::exec::test
