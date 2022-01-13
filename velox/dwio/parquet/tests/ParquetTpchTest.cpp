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
#include "velox/type/tests/FilterBuilder.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class ParquetTpchTest : public HiveConnectorTestBase {
 protected:
  using OperatorTestBase::assertQuery;

  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    parquet::registerParquetReaderFactory();
  }

  void TearDown() override {
    parquet::unregisterParquetReaderFactory();
    HiveConnectorTestBase::TearDown();
  }

  std::vector<std::string> getLineItemFilePaths() {
    auto fileNames = {
        "lineitem1.par", "lineitem2.par", "lineitem3.par", "lineitem4.par"};
    std::vector<std::string> filePaths;
    for (const auto& fileName : fileNames) {
      filePaths.push_back(facebook::velox::test::getDataFilePath(
          "velox/dwio/parquet/tests", fmt::format("tpch_tiny/{}", fileName)));
    }
    return filePaths;
  }

  std::shared_ptr<connector::hive::HiveConnectorSplit> makeSplit(
      const std::string& filePath) {
    auto split = makeHiveConnectorSplit(filePath);
    split->fileFormat = dwio::common::FileFormat::PARQUET;
    return split;
  }

  void addSplitsToTask(
      exec::Task* task,
      bool& noMoreSplits,
      const std::vector<std::string>& filePaths,
      int sourcePlanNodeId) {
    if (!noMoreSplits) {
      for (const auto& filePath : filePaths) {
        task->addSplit(
            std::to_string(sourcePlanNodeId), exec::Split(makeSplit(filePath)));
      }
      task->noMoreSplits(std::to_string(sourcePlanNodeId));
      noMoreSplits = true;
    }
  }
};

TEST_F(ParquetTpchTest, tpchQ1) {
  const auto& filePaths = getLineItemFilePaths();

  auto rowType =
      ROW({"returnflag",
           "linestatus",
           "quantity",
           "extendedprice",
           "discount",
           "tax",
           "shipdate"},
          {VARCHAR(),
           VARCHAR(),
           DOUBLE(),
           DOUBLE(),
           DOUBLE(),
           DOUBLE(),
           VARCHAR()});

  // date '1998-12-01' - interval '90' day <= shipdate <= '1998-12-01'
  auto filters =
      common::test::SubfieldFiltersBuilder()
          .add("shipdate", common::test::between("1998-09-02", "1998-12-01"))
          .build();

  const int sourcePlanNodeId = 10;
  static const core::SortOrder kAscNullsLast(true, false);

  const auto stage1 = PlanBuilder(sourcePlanNodeId)
                          .tableScan(
                              rowType,
                              makeTableHandle(std::move(filters)),
                              allRegularColumns(rowType))
                          .project(
                              {"returnflag",
                               "linestatus",
                               "quantity",
                               "extendedprice",
                               "extendedprice * (1.0 - discount)",
                               "extendedprice * (1.0 - discount) * (1.0 + tax)",
                               "discount"})
                          .partialAggregation(
                              {0, 1},
                              {"sum(quantity)",
                               "sum(extendedprice)",
                               "sum(p4)",
                               "sum(p5)",
                               "avg(quantity)",
                               "avg(extendedprice)",
                               "avg(discount)",
                               "count(0)"})
                          .planNode();

  auto plan = PlanBuilder(1)
                  .localPartition({0, 1}, {stage1})
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
                  .planNode();

  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 4;
  params.numResultDrivers = 1;

  bool noMoreSplits = false;
  auto result = readCursor(params, [&](auto* task) {
    addSplitsToTask(task, noMoreSplits, filePaths, sourcePlanNodeId);
  });

  EXPECT_EQ(1, result.second.size());
  auto output = materialize(result.second[0]);
  ASSERT_EQ(10, output[0].size());
  ASSERT_EQ(std::string("N"), std::string(output[0][0].value<StringView>()));
  ASSERT_EQ(std::string("O"), std::string(output[0][1].value<StringView>()));
  ASSERT_NEAR(22'943.0, output[0][2].value<double>(), 0.0001);
  ASSERT_NEAR(32'085'596.97999, output[0][3].value<double>(), 0.0001);
  ASSERT_NEAR(30'461'584.92660, output[0][4].value<double>(), 0.0001);
  ASSERT_NEAR(31'719'382.51849, output[0][5].value<double>(), 0.0001);
  ASSERT_NEAR(25.86583, output[0][6].value<double>(), 0.0001);
  ASSERT_NEAR(36'173.16457, output[0][7].value<double>(), 0.0001);
  ASSERT_NEAR(0.0500, output[0][8].value<double>(), 0.0001);
  ASSERT_EQ(887, output[0][9].value<int64_t>());

  const auto& stats = result.first->task()->taskStats();
  // There should be two pipelines
  ASSERT_EQ(2, stats.pipelineStats.size());
  ASSERT_EQ(4, stats.numFinishedSplits);
}

TEST_F(ParquetTpchTest, tpchQ6) {
  const auto& filePaths = getLineItemFilePaths();

  auto rowType =
      ROW({"shipdate", "extendedprice", "quantity", "discount"},
          {VARCHAR(), DOUBLE(), DOUBLE(), DOUBLE()});

  auto filters =
      common::test::SubfieldFiltersBuilder()
          .add("shipdate", common::test::between("1994-01-01", "1994-12-31"))
          .add("discount", common::test::betweenDouble(0.05, 0.07))
          .add("quantity", common::test::lessThanDouble(24.0))
          .build();

  auto plan = PlanBuilder(10)
                  .localPartition(
                      {},
                      {PlanBuilder(0)
                           .tableScan(
                               rowType,
                               makeTableHandle(std::move(filters)),
                               allRegularColumns(rowType))
                           .project({"extendedprice * discount"})
                           .partialAggregation({}, {"sum(p0)"})
                           .planNode()})
                  .finalAggregation({}, {"sum(a0)"}, {DOUBLE()})
                  .planNode();

  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 4;
  params.numResultDrivers = 1;

  bool noMoreSplits = false;
  auto result = readSingleValue(params, [&](auto* task) {
    addSplitsToTask(task, noMoreSplits, filePaths, 0);
  });

  ASSERT_NEAR(1'193'053.2253, result.value<double>(), 0.0001);
}
