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
#include "velox/duckdb/conversion/DuckConversion.h"
#include "velox/duckdb/conversion/DuckWrapper.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
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
  ParquetTpchTestBase() {}

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
    registerParquetReaderFactory();
    registerParquetWriterFactory();

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
  RowTypePtr getType(const DuckDBQueryResult& result) {
    std::vector<std::string> names = {};
    std::vector<TypePtr> types = {};
    for (auto i = 0; i < result->ColumnCount(); i++) {
      names.push_back(result->names[i]);
      types.push_back(duckdb::toVeloxType(result->types[i]));
    }
    return std::make_shared<RowType>(std::move(names), std::move(types));
  }

  std::vector<RowVectorPtr> getVector(
      const DuckDBQueryResult& result,
      const RowTypePtr& resultType) {
    std::vector<RowVectorPtr> rowVectors{};

    while (true) {
      auto currentChunk = result->Fetch();
      if (!currentChunk) {
        break;
      }
      currentChunk->Normalify();
      std::vector<VectorPtr> outputColumns;
      for (auto i = 0; i < result->ColumnCount(); i++) {
        auto vector = duckdb::toVeloxVector(
            currentChunk->size(),
            currentChunk->data[i],
            resultType->childAt(i),
            pool_.get());
        outputColumns.push_back(vector);
      }

      auto rowVector = std::make_shared<RowVector>(
          pool_.get(),
          resultType,
          BufferPtr(nullptr),
          currentChunk->size(),
          outputColumns);

      rowVectors.push_back(rowVector);
    }

    return rowVectors;
  }

  /// Write TPC-H tables as a Parquet file to temp directory in hive-style
  /// partition.
  void saveTpchTablesAsParquet() {
    constexpr int kRowGroupSize = 10'000;
    const auto tableNames = tpchBuilder_.getTableNames();
    for (const auto& tableName : tableNames) {
      auto tableDirectory =
          fmt::format("{}/{}", tempDirectory_->path, tableName);
      auto query =
          fmt::format(fmt::runtime(kDuckDbTpchSQL_.at(tableName)), tableName);

      auto res = duckDb_->execute(query);

      auto schema = getType(res);

      auto input = getVector(res, schema);
      auto plan = PlanBuilder()
                      .values(input)
                      .tableWrite(
                          schema,
                          schema->names(),
                          std::make_shared<core::InsertTableHandle>(
                              kHiveConnectorId,
                              HiveConnectorTestBase::makeHiveInsertTableHandle(
                                  schema->names(),
                                  schema->children(),
                                  {},
                                  HiveConnectorTestBase::makeLocationHandle(
                                      tableDirectory),
                                  dwio::common::FileFormat::PARQUET)))
                      .planNode();

      // Write the data and assert that all records from the TPC-H table have
      // been written.
      exec::test::assertQuery(
          plan, fmt::format("select count(*) from {}", tableName), *duckDb_);
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

  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::defaultMemoryManager().addRootPool("root")};
  std::shared_ptr<memory::MemoryPool> pool_{rootPool_->addLeafChild("leaf")};
  std::shared_ptr<DuckDbQueryRunner> duckDb_ = nullptr;
  std::shared_ptr<TempDirectoryPath> tempDirectory_ = nullptr;
  TpchQueryBuilder tpchBuilder_ =
      TpchQueryBuilder(dwio::common::FileFormat::PARQUET);
  const std::unordered_map<std::string, std::string> kDuckDbTpchSQL_ = {
      std::make_pair(
          "lineitem",
          R"(SELECT l_orderkey, l_partkey, l_suppkey, l_linenumber,
         l_quantity::DOUBLE as quantity, l_extendedprice::DOUBLE as extendedprice, l_discount::DOUBLE as discount,
         l_tax::DOUBLE as tax, l_returnflag, l_linestatus, l_shipdate AS shipdate, l_commitdate, l_receiptdate,
         l_shipinstruct, l_shipmode, l_comment FROM {})"),
      std::make_pair(
          "orders",
          R"(SELECT o_orderkey, o_custkey, o_orderstatus,
         o_totalprice::DOUBLE as o_totalprice,
         o_orderdate, o_orderpriority, o_clerk, o_shippriority, o_comment FROM {})"),
      std::make_pair(
          "customer",
          R"(SELECT c_custkey, c_name, c_address, c_nationkey, c_phone,
         c_acctbal::DOUBLE as c_acctbal, c_mktsegment, c_comment FROM {})"),
      std::make_pair("nation", R"(SELECT * FROM {})"),
      std::make_pair("region", R"(SELECT * FROM {})"),
      std::make_pair(
          "part",
          R"(SELECT p_partkey, p_name, p_mfgr, p_brand, p_type, p_size,
         p_container, p_retailprice::DOUBLE, p_comment FROM {})"),
      std::make_pair(
          "supplier",
          R"(SELECT s_suppkey, s_name, s_address, s_nationkey, s_phone,
         s_acctbal::DOUBLE, s_comment FROM {})"),
      std::make_pair(
          "partsupp",
          R"(SELECT ps_partkey, ps_suppkey, ps_availqty,
         ps_supplycost::DOUBLE as supplycost, ps_comment FROM {})")};
};

} // namespace
} // namespace facebook::velox::exec::test
