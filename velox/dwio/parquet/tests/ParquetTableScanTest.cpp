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

static const std::string kWriter = "ParquetTableScanTest.Writer";

class ParquetTableScanTest : public HiveConnectorTestBase {
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

  std::string getExampleFilePath(const std::string& fileName) {
    return facebook::velox::test::getDataFilePath(
        "velox/dwio/parquet/tests", "examples/" + fileName);
  }

  std::shared_ptr<connector::hive::HiveConnectorSplit> makeSplit(
      const std::string& filePath) {
    auto split = makeHiveConnectorSplit(filePath);
    split->fileFormat = dwio::common::FileFormat::PARQUET;
    return split;
  }
};

TEST_F(ParquetTableScanTest, basic) {
  auto data = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>(20, [](auto row) { return row + 1; }),
          makeFlatVector<double>(20, [](auto row) { return row + 1; }),
      });
  createDuckDbTable({data});

  auto split = makeSplit(getExampleFilePath("sample.parquet"));

  auto rowType = ROW({"a", "b"}, {BIGINT(), DOUBLE()});
  auto plan = PlanBuilder().tableScan(rowType).planNode();

  assertQuery(plan, {split}, "SELECT * FROM tmp");

  // Add a filter on "a".
  auto filters =
      common::test::singleSubfieldFilter("a", common::test::lessThan(3));

  plan = PlanBuilder()
             .tableScan(
                 rowType,
                 makeTableHandle(std::move(filters)),
                 allRegularColumns(rowType))
             .planNode();

  assertQuery(plan, {split}, "SELECT * FROM tmp WHERE a < 3");

  // Add an aggregation.
  plan = PlanBuilder()
             .tableScan(rowType)
             .singleAggregation({}, {"min(a)", "max(b)"})
             .planNode();

  assertQuery(plan, {split}, "SELECT min(a), max(b) FROM tmp");
}

TEST_F(ParquetTableScanTest, tpchQ6) {
  auto fileNames = {
      "lineitem1.par", "lineitem2.par", "lineitem3.par", "lineitem4.par"};
  std::vector<std::string> filePaths;
  for (const auto& fileName : fileNames) {
    filePaths.push_back(facebook::velox::test::getDataFilePath(
        "velox/dwio/parquet/tests", fmt::format("tpch_tiny/{}", fileName)));
  }

  // TODO: Make sure to specify columns in the same order they appear in the
  // files. Specifying columns out-of-order breaks filter pushdown and generates
  // an error: INTERNAL Error: Invalid PhysicalType for GetTypeIdSize
  auto rowType =
      ROW({"quantity", "extendedprice", "discount", "shipdate"},
          {DOUBLE(), DOUBLE(), DOUBLE(), VARCHAR()});
  auto filters =
      common::test::SubfieldFiltersBuilder()
          .add("shipdate", common::test::between("1994-01-01", "1994-12-31"))
          .add("discount", common::test::betweenDouble(0.05, 0.07))
          .add("quantity", common::test::lessThanDouble(24.0))
          .build();

  auto plan = PlanBuilder(10)
                  .localPartition(
                      {},
                      {PlanBuilder()
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
    if (!noMoreSplits) {
      for (const auto& filePath : filePaths) {
        task->addSplit("0", exec::Split(makeSplit(filePath)));
      }
      task->noMoreSplits("0");
      noMoreSplits = true;
    }
  });

  ASSERT_NEAR(1'193'053.2253, result.value<double>(), 0.0001);
}
