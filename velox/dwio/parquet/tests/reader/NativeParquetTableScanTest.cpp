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

#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::parquet;

class NativeParquetTableScanTest : public HiveConnectorTestBase {
 protected:
  using OperatorTestBase::assertQuery;

  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    registerParquetReaderFactory(ParquetReaderType::NATIVE);
  }

  void TearDown() override {
    unregisterParquetReaderFactory();
    HiveConnectorTestBase::TearDown();
  }

  std::string getExampleFilePath(const std::string& fileName) {
    return facebook::velox::test::getDataFilePath(
        "velox/dwio/parquet/tests/duckdb_reader", "../examples/" + fileName);
  }

  std::shared_ptr<connector::hive::HiveConnectorSplit> makeSplit(
      const std::string& filePath) {
    return makeHiveConnectorSplits(
        filePath, 1, dwio::common::FileFormat::PARQUET)[0];
  }

 private:
  RowTypePtr rowType_;
  std::vector<std::shared_ptr<connector::ConnectorSplit>> splits_;
};

TEST_F(NativeParquetTableScanTest, agg) {
  auto filePath = getExampleFilePath("customer.parquet");
  auto split = makeSplit(filePath);

  auto rowType =
      ROW({"customer_id",
           "customer_first_name",
           "customer_last_name",
           "customer_preferred_cust_flag",
           "customer_birth_country",
           "customer_login",
           "customer_email_address",
           "dyear",
           "year_total"},
          {VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           INTEGER(),
           DOUBLE()});

  auto plan = PlanBuilder()
                  .tableScan(rowType)
                  .finalAggregation(
                      {"customer_id",
                       "customer_first_name",
                       "customer_last_name",
                       "customer_preferred_cust_flag",
                       "customer_birth_country",
                       "customer_login",
                       "customer_email_address",
                       "dyear"},
                      {"sum(year_total)"},
                      {DOUBLE()})
                  .planNode();

  // A fake DuckDb result only for test triggering.
  auto vectors = makeVectors(rowType, 1, 1);
  createDuckDbTable(vectors);

  assertQuery(
      plan,
      {split},
      "SELECT customer_id, customer_first_name, customer_last_name, customer_preferred_cust_flag, customer_birth_country, customer_login, customer_email_address, dyear, year_total from tmp");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
