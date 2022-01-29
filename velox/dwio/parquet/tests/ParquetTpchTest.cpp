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

#include "velox/dwio/dwrf/test/utils/DataFiles.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/type/tests/FilterBuilder.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class ParquetTpchTest : public HiveConnectorTestBase {
 protected:
  // Setup a duckDB instance for the entire suite and load TPCH data with scale
  // factor 0.01.
  static void SetUpTestCase() {
    if (duckDb_ == nullptr) {
      duckDb_ = std::make_shared<DuckDbQueryRunner>();
      constexpr double tpchsf = 0.01;
      duckDb_->initTPCH(tpchsf);
    }
    functions::prestosql::registerAllFunctions();
  }

  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    parquet::registerParquetReaderFactory();
    tempDirectory_ = tempDirectory_ = exec::test::TempDirectoryPath::create();
  }

  void TearDown() override {
    parquet::unregisterParquetReaderFactory();
    HiveConnectorTestBase::TearDown();
  }

  int64_t date(std::string_view string_date) {
    Date date;
    parseTo(string_date, date);
    return date.days();
  }

  // For a given file at filePath, add splits equal to numSplits.
  std::vector<std::shared_ptr<connector::hive::HiveConnectorSplit>> makeSplits(
      const std::string& filePath,
      int64_t numSplits) {
    LocalReadFile lfs(filePath);
    const int fileSize = lfs.size();
    const int splitSize = fileSize / numSplits;
    std::vector<std::shared_ptr<connector::hive::HiveConnectorSplit>> splits;

    // Add all the splits except the last one.
    int i = 0;
    for (; i < numSplits - 1; i++) {
      auto split = makeHiveConnectorSplit(filePath, i * splitSize, splitSize);
      split->fileFormat = dwio::common::FileFormat::PARQUET;
      splits.push_back(std::move(split));
    }
    // Add the last split with the remaining as start.
    auto split = makeHiveConnectorSplit(filePath, i * splitSize);
    split->fileFormat = dwio::common::FileFormat::PARQUET;
    splits.push_back(std::move(split));
    return splits;
  }

  void addSplitsToTask(
      exec::Task* task,
      bool& noMoreSplits,
      const std::vector<std::string>& filePaths,
      int sourcePlanNodeId = 0,
      int64_t numSplits = 10) {
    if (!noMoreSplits) {
      for (const auto& filePath : filePaths) {
        auto const& splits = makeSplits(filePath, numSplits);
        for (const auto& split : splits) {
          task->addSplit(std::to_string(sourcePlanNodeId), exec::Split(split));
        }
      }
      task->noMoreSplits(std::to_string(sourcePlanNodeId));
      noMoreSplits = true;
    }
  }

  // Write the Lineitem TPCH table to a Parquet file and return the file
  // location.
  std::string writeDuckDBTPCHLineitemTableToParquet() {
    constexpr std::string_view tableName("lineitem");
    // Lineitem SF=0.01 has 60175 rows
    // Set the number of rows in a RowGroup so that the generated file contains
    // multiple RowGroups.
    constexpr int rowGroupNumRows = 15'000;
    const auto& filePath =
        fmt::format("{}/{}.parquet", tempDirectory_->path, tableName);
    // Convert decimal columns to double.
    const auto& query = fmt::format(
        "COPY (SELECT l_orderkey as orderkey, l_partkey as partkey, l_suppkey as suppkey, l_linenumber as linenumber, "
        "l_quantity::DOUBLE as quantity, l_extendedprice::DOUBLE as extendedprice, l_discount::DOUBLE as discount, "
        "l_tax::DOUBLE as tax, l_returnflag as returnflag, l_linestatus as linestatus, "
        "l_shipdate as shipdate, l_receiptdate as receiptdate, "
        "l_shipinstruct as shipinstruct, l_shipmode as shipmode, l_comment as comment "
        "FROM {}) TO '{}' (FORMAT 'parquet', ROW_GROUP_SIZE {})",
        tableName,
        filePath,
        rowGroupNumRows);
    duckDb_->execute(query);
    return filePath;
  }

  static std::shared_ptr<DuckDbQueryRunner> duckDb_;
  std::shared_ptr<exec::test::TempDirectoryPath> tempDirectory_;
};

std::shared_ptr<DuckDbQueryRunner> ParquetTpchTest::duckDb_ = nullptr;

TEST_F(ParquetTpchTest, tpchQ1) {
  const auto& lineitemFilePath = writeDuckDBTPCHLineitemTableToParquet();

  // TPCH Q1
  auto rowType = ROW(
      {"returnflag",
       "linestatus",
       "quantity",
       "extendedprice",
       "discount",
       "tax",
       "shipdate"},
      {VARCHAR(), VARCHAR(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DATE()});

  // shipdate <= '1998-09-02'
  auto filters =
      common::test::SubfieldFiltersBuilder()
          .add("shipdate", common::test::lessThanOrEqual(date("1998-09-02")))
          .build();

  const int sourcePlanNodeId = 10;
  CursorParameters params;
  params.maxDrivers = 4;
  params.numResultDrivers = 1;
  static const core::SortOrder kAscNullsLast(true, false);

  const auto stage1 =
      PlanBuilder(sourcePlanNodeId)
          .tableScan(
              rowType,
              makeTableHandle(std::move(filters)),
              allRegularColumns(rowType))
          .project(
              {"returnflag",
               "linestatus",
               "quantity",
               "extendedprice",
               "extendedprice * (1.0 - discount) AS sum_disc_price",
               "extendedprice * (1.0 - discount) * (1.0 + tax) AS sum_charge",
               "discount"})
          .partialAggregation(
              {0, 1},
              {"sum(quantity)",
               "sum(extendedprice)",
               "sum(sum_disc_price)",
               "sum(sum_charge)",
               "avg(quantity)",
               "avg(extendedprice)",
               "avg(discount)",
               "count(0)"})
          .planNode();

  auto plan = PlanBuilder(1)
                  .localPartition({}, {stage1})
                  .finalAggregation(
                      {0, 1},
                      {"sum(a0)",
                       "sum(a1)",
                       "sum(a2)",
                       "sum(a3)",
                       "avg(a4)",
                       "avg(a5)",
                       "avg(a6)",
                       "count(a7)"},
                      {DOUBLE(),
                       DOUBLE(),
                       DOUBLE(),
                       DOUBLE(),
                       DOUBLE(),
                       DOUBLE(),
                       DOUBLE(),
                       BIGINT()})
                  .orderBy({0, 1}, {kAscNullsLast, kAscNullsLast}, false)
                  // Additional step for double type result verification
                  .project(
                      {"returnflag",
                       "linestatus",
                       "round(a0, cast(2 as integer))",
                       "round(a1, cast(2 as integer))",
                       "round(a2, cast(2 as integer))",
                       "round(a3, cast(2 as integer))",
                       "round(a4, cast(2 as integer))",
                       "round(a5, cast(2 as integer))",
                       "round(a6, cast(2 as integer))",
                       "a7"})
                  .planNode();

  params.planNode = std::move(plan);
  bool noMoreSplits = false;
  auto duckDbSql = duckDb_->GetQuery(1);
  // Additional step for double type result verification.
  duckDbSql.pop_back(); // remove new line
  duckDbSql.pop_back(); // remove semi-colon
  const auto& duckDBTPCHQ1Rounded = fmt::format(
      "select l_returnflag, l_linestatus, round(sum_qty, 2), "
      "round(sum_base_price, 2), round(sum_disc_price, 2), round(sum_charge, 2), "
      "round(avg_qty, 2), round(avg_price, 2), round(avg_disc, 2),"
      "count_order from ({})",
      duckDbSql);
  auto task = exec::test::assertQuery(
      params,
      [&](exec::Task* task) {
        addSplitsToTask(
            task, noMoreSplits, {lineitemFilePath}, sourcePlanNodeId);
      },
      duckDBTPCHQ1Rounded,
      *duckDb_);

  const auto& stats = task->taskStats();
  // There should be two pipelines.
  ASSERT_EQ(2, stats.pipelineStats.size());
  // We used the default of 10 splits per file.
  ASSERT_EQ(10, stats.numFinishedSplits);
}

TEST_F(ParquetTpchTest, tpchQ6) {
  const auto& lineitemFilePath = writeDuckDBTPCHLineitemTableToParquet();

  auto rowType =
      ROW({"shipdate", "extendedprice", "quantity", "discount"},
          {DATE(), DOUBLE(), DOUBLE(), DOUBLE()});

  auto filters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "shipdate",
              common::test::between(date("1994-01-01"), date("1994-12-31")))
          .add("discount", common::test::betweenDouble(0.05, 0.07))
          .add("quantity", common::test::lessThanDouble(24.0))
          .build();

  const int sourcePlanNodeId = 4;
  CursorParameters params;
  params.maxDrivers = 4;
  params.numResultDrivers = 1;

  auto plan = PlanBuilder(10)
                  .localPartition(
                      {},
                      {PlanBuilder(sourcePlanNodeId)
                           .tableScan(
                               rowType,
                               makeTableHandle(std::move(filters)),
                               allRegularColumns(rowType))
                           .project({"extendedprice * discount"})
                           .partialAggregation({}, {"sum(p0)"})
                           .planNode()})
                  .finalAggregation({}, {"sum(a0)"}, {DOUBLE()})
                  // Additional step for double type result verification
                  .project({"round(a0, cast(2 as integer))"})
                  .planNode();

  params.planNode = std::move(plan);
  bool noMoreSplits = false;
  auto duckDbSql = duckDb_->GetQuery(6);
  // Additional step for double type result verification.
  duckDbSql.pop_back(); // remove new line
  duckDbSql.pop_back(); // remove semi-colon
  const auto& duckDBTPCHQ6Rounded =
      fmt::format("select round(revenue, 2) from ({})", duckDbSql);
  auto task = exec::test::assertQuery(
      params,
      [&](exec::Task* task) {
        addSplitsToTask(
            task, noMoreSplits, {lineitemFilePath}, sourcePlanNodeId);
      },
      duckDBTPCHQ6Rounded,
      *duckDb_);

  const auto& stats = task->taskStats();
  // There should be two pipelines
  ASSERT_EQ(2, stats.pipelineStats.size());
  // We used the default of 10 splits
  ASSERT_EQ(10, stats.numFinishedSplits);
}
