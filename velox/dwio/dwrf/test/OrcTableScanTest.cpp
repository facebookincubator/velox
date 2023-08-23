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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;
using namespace facebook::velox::common::test;
using namespace facebook::velox::exec::test;

class OrcTableScanTest : public HiveConnectorTestBase {
 protected:
  using OperatorTestBase::assertQuery;

  void SetUp() {
    dwrf::registerOrcReaderFactory();
    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(kHiveConnectorId, nullptr);
    connector::registerConnector(hiveConnector);
  }

  std::string getExampleFilePath(const std::string& fileName) const {
    return facebook::velox::test::getDataFilePath(
        "velox/dwio/dwrf/test", "examples/" + fileName);
  }

  std::shared_ptr<connector::hive::HiveConnectorSplit> makeSplit(
      const std::string& filePath) const {
    return makeHiveConnectorSplits(
        filePath, 1, dwio::common::FileFormat::ORC)[0];
  }

  void loadData(const std::string& filePath) {
    splits_ = {makeSplit(filePath)};
    dwio::common::ReaderOptions readerOpts{pool_.get()};
    readerOpts.setFileFormat(dwio::common::FileFormat::ORC);
    auto reader = dwrf::DwrfReader::create(
        std::make_unique<dwio::common::BufferedInput>(
            std::make_shared<LocalReadFile>(filePath),
            readerOpts.getMemoryPool()),
        readerOpts);
    auto rowReader = reader->createRowReader();
    rowType_ = reader->rowType();
    VectorPtr batch = BaseVector::create(rowType_, 0, pool_.get());
    rowReader->next(100, batch);
    std::vector<RowVectorPtr> vectors{
        std::dynamic_pointer_cast<RowVector>(batch)};
    createDuckDbTable(vectors);
  }

  void assertSelectWithFilter(
      const std::vector<std::string>& subfieldFilters,
      const std::string& sql) {
    parse::ParseOptions options;
    options.parseDecimalAsDouble = false;
    auto plan = PlanBuilder(pool_.get())
                    .setParseOptions(options)
                    .tableScan(rowType_, subfieldFilters)
                    .planNode();
    assertQuery(plan, splits_, sql);
  }

  int64_t assertSelectWithAgg(
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::string& sql) {
    auto plan = PlanBuilder(pool_.get())
                    .tableScan(rowType_)
                    .singleAggregation(groupingKeys, aggregates)
                    .planNode();
    auto task = assertQuery(plan, splits_, sql);
    auto operatorStats = task->taskStats().pipelineStats[0].operatorStats;
    int64_t loadedToValueHook = 0;
    for (auto& operatorStat : operatorStats) {
      auto stats = operatorStat.runtimeStats;
      auto it = stats.find("loadedToValueHook");
      if (it != stats.end()) {
        loadedToValueHook += it->second.sum;
      }
    }
    return loadedToValueHook;
  }

  void testDecimalWithAgg(const std::string& filePath) {
    loadData(filePath);

    // Sum Aggregate
    auto loadedToValueHook = assertSelectWithAgg(
        {"a"}, {"sum(b)"}, "SELECT a, sum(b) FROM tmp GROUP BY a");
    // DecimalSumAggregate don't support function pushdown yet
    // So loadedToValueHook will be zero
    EXPECT_EQ(0, loadedToValueHook);

    // Min Max Aggregate
    loadedToValueHook = assertSelectWithAgg(
        {"a"}, {"max(b)"}, "SELECT a, max(b) FROM tmp GROUP BY a");
    EXPECT_EQ(100, loadedToValueHook);

    loadedToValueHook = assertSelectWithAgg(
        {"a"}, {"min(b)"}, "SELECT a, min(b) FROM tmp GROUP BY a");
    EXPECT_EQ(100, loadedToValueHook);
  }

 private:
  RowTypePtr rowType_;
  std::vector<std::shared_ptr<connector::ConnectorSplit>> splits_;
};

TEST_F(OrcTableScanTest, shortDecimalWithFilter) {
  std::string filePath = getExampleFilePath("short_decimal.orc");
  loadData(filePath);

  // IsNotNull
  std::string select = "SELECT * FROM tmp WHERE ";
  std::string filter = "b IS NOT NULL";
  assertSelectWithFilter({filter}, select + filter);

  // IsNull
  filter = "b IS NULL";
  assertSelectWithFilter({filter}, select + filter);

  // BigintRange
  filter = "b <= CAST(428493 as DECIMAL(10,2))";
  assertSelectWithFilter({filter}, select + filter);

  // BigintMultiRange
  filter =
      "b BETWEEN CAST(427531 as DECIMAL(10,2)) AND CAST(428493 as DECIMAL(10,2))";
  assertSelectWithFilter({filter}, select + filter);

  // NegatedBigintRange
  filter =
      "b NOT BETWEEN CAST(427531 as DECIMAL(10,2)) AND CAST(428493 as DECIMAL(10,2))";
  assertSelectWithFilter({filter}, select + filter);

  // BigintValues
  filter =
      "b IN (CAST(427531 as DECIMAL(10,2)), CAST(428493 as DECIMAL(10,2)))";
  assertSelectWithFilter({filter}, select + filter);

  // NegatedBigintValues
  filter =
      "b NOT IN (CAST(427531 as DECIMAL(10,2)), CAST(428493 as DECIMAL(10,2)))";
  assertSelectWithFilter({filter}, select + filter);
}

TEST_F(OrcTableScanTest, longDecimalWithFilter) {
  std::string filePath = getExampleFilePath("long_decimal.orc");
  loadData(filePath);

  // IsNotNull
  std::string select = "SELECT * FROM tmp WHERE ";
  std::string filter = "a IS NOT NULL";
  assertSelectWithFilter({filter}, select + filter);

  // IsNull
  filter = "a IS NULL";
  assertSelectWithFilter({filter}, select + filter);

  // HugeintRange
  filter = "a <= CAST(88892862 as DECIMAL(20,6))";
  assertSelectWithFilter({filter}, select + filter);

  // HugeintValues
  filter =
      "a IN (CAST(88892862 as DECIMAL(20,6)), CAST(87126468 as DECIMAL(20,6)))";
  assertSelectWithFilter({filter}, select + filter);
}

TEST_F(OrcTableScanTest, shortDecimalWithAgg) {
  std::string filePath = getExampleFilePath("short_decimal.orc");
  testDecimalWithAgg(filePath);
}

TEST_F(OrcTableScanTest, longDecimalWithAgg) {
  std::string filePath = getExampleFilePath("long_decimal.orc");
  testDecimalWithAgg(filePath);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
