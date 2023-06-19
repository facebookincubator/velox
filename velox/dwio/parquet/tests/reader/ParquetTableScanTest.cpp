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
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::parquet;

class ParquetTableScanTest : public HiveConnectorTestBase {
 protected:
  using OperatorTestBase::assertQuery;

  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    registerParquetReaderFactory(
        facebook::velox::parquet::ParquetReaderType::NATIVE);
  }

  void TearDown() override {
    unregisterParquetReaderFactory();
    HiveConnectorTestBase::TearDown();
  }

  void assertSelectWithFilter(
      std::vector<std::string>&& outputColumnNames,
      const std::vector<std::string>& subfieldFilters,
      const std::string& remainingFilter,
      const std::string& sql) {
    auto rowType = getRowType(std::move(outputColumnNames));
    parse::ParseOptions options;
    options.parseDecimalAsDouble = false;

    auto plan = PlanBuilder()
                    .setParseOptions(options)
                    .tableScan(rowType, subfieldFilters, remainingFilter)
                    .planNode();

    assertQuery(plan, splits_, sql);
  }

  void
  loadData(const std::string& filePath, RowTypePtr rowType, RowVectorPtr data) {
    splits_ = {makeSplit(filePath)};
    rowType_ = rowType;
    createDuckDbTable({data});
  }

  std::string getExampleFilePath(const std::string& fileName) {
    return facebook::velox::test::getDataFilePath(
        "velox/dwio/parquet/tests/reader", "../examples/" + fileName);
  }

  std::shared_ptr<connector::hive::HiveConnectorSplit> makeSplit(
      const std::string& filePath) {
    return makeHiveConnectorSplits(
        filePath, 1, dwio::common::FileFormat::PARQUET)[0];
  }

 private:
  RowTypePtr getRowType(std::vector<std::string>&& outputColumnNames) const {
    std::vector<TypePtr> types;
    for (auto colName : outputColumnNames) {
      types.push_back(rowType_->findChild(colName));
    }

    return ROW(std::move(outputColumnNames), std::move(types));
  }

  RowTypePtr rowType_;
  std::vector<std::shared_ptr<connector::ConnectorSplit>> splits_;
};

TEST_F(ParquetTableScanTest, decimalSubfieldFilter) {
  // decimal.parquet holds two columns (a: DECIMAL(5, 2), b: DECIMAL(20, 5)) and
  // 20 rows (10 rows per group). Data is in plain uncompressed format:
  //   a: [100.01 .. 100.20]
  //   b: [100000000000000.00001 .. 100000000000000.00020]
  std::vector<int64_t> unscaledShortValues(20);
  std::iota(unscaledShortValues.begin(), unscaledShortValues.end(), 10001);
  loadData(
      getExampleFilePath("decimal.parquet"),
      ROW({"a"}, {DECIMAL(5, 2)}),
      makeRowVector(
          {"a"},
          {
              makeShortDecimalFlatVector(unscaledShortValues, DECIMAL(5, 2)),
          }));

  assertSelectWithFilter(
      {"a"}, {"a < 100.07"}, "", "SELECT a FROM tmp WHERE a < 100.07");
  assertSelectWithFilter(
      {"a"}, {"a <= 100.07"}, "", "SELECT a FROM tmp WHERE a <= 100.07");
  assertSelectWithFilter(
      {"a"}, {"a > 100.07"}, "", "SELECT a FROM tmp WHERE a > 100.07");
  assertSelectWithFilter(
      {"a"}, {"a >= 100.07"}, "", "SELECT a FROM tmp WHERE a >= 100.07");
  assertSelectWithFilter(
      {"a"}, {"a = 100.07"}, "", "SELECT a FROM tmp WHERE a = 100.07");
  assertSelectWithFilter(
      {"a"},
      {"a BETWEEN 100.07 AND 100.12"},
      "",
      "SELECT a FROM tmp WHERE a BETWEEN 100.07 AND 100.12");

  VELOX_ASSERT_THROW(
      assertSelectWithFilter(
          {"a"}, {"a < 1000.7"}, "", "SELECT a FROM tmp WHERE a < 1000.7"),
      "Scalar function signature is not supported: lt(DECIMAL(5,2), DECIMAL(5,1))");
  VELOX_ASSERT_THROW(
      assertSelectWithFilter(
          {"a"}, {"a = 1000.7"}, "", "SELECT a FROM tmp WHERE a = 1000.7"),
      "Scalar function signature is not supported: eq(DECIMAL(5,2), DECIMAL(5,1))");
}

TEST_F(ParquetTableScanTest, timestampFilter) {
  // timestamp-int96.parquet holds one column (t: TIMESTAMP) and
  // 10 rows in one row group. Data is in SNAPPY compressed format.
  // The values are:
  // |t                  |
  // +-------------------+
  // |2015-06-01 19:34:56|
  // |2015-06-02 19:34:56|
  // |2001-02-03 03:34:06|
  // |1998-03-01 08:01:06|
  // |2022-12-23 03:56:01|
  // |1980-01-24 00:23:07|
  // |1999-12-08 13:39:26|
  // |2023-04-21 09:09:34|
  // |2000-09-12 22:36:29|
  // |2007-12-12 04:27:56|
  // +-------------------+
  auto vector = makeFlatVector<Timestamp>(
      {Timestamp(1433116800, 70496000000000),
       Timestamp(1433203200, 70496000000000),
       Timestamp(981158400, 12846000000000),
       Timestamp(888710400, 28866000000000),
       Timestamp(1671753600, 14161000000000),
       Timestamp(317520000, 1387000000000),
       Timestamp(944611200, 49166000000000),
       Timestamp(1682035200, 32974000000000),
       Timestamp(968716800, 81389000000000),
       Timestamp(1197417600, 16076000000000)});

  loadData(
      getExampleFilePath("timestamp-int96.parquet"),
      ROW({"t"}, {TIMESTAMP()}),
      makeRowVector(
          {"t"},
          {
              vector,
          }));

  assertSelectWithFilter({"t"}, {}, "", "SELECT t from tmp");
  assertSelectWithFilter(
      {"t"},
      {},
      "t < TIMESTAMP '2000-09-12 22:36:29'",
      "SELECT t from tmp where t < TIMESTAMP '2000-09-12 22:36:29'");
  assertSelectWithFilter(
      {"t"},
      {},
      "t <= TIMESTAMP '2000-09-12 22:36:29'",
      "SELECT t from tmp where t <= TIMESTAMP '2000-09-12 22:36:29'");
  assertSelectWithFilter(
      {"t"},
      {},
      "t > TIMESTAMP '1980-01-24 00:23:07'",
      "SELECT t from tmp where t > TIMESTAMP '1980-01-24 00:23:07'");
  assertSelectWithFilter(
      {"t"},
      {},
      "t >= TIMESTAMP '1980-01-24 00:23:07'",
      "SELECT t from tmp where t >= TIMESTAMP '1980-01-24 00:23:07'");
  assertSelectWithFilter(
      {"t"},
      {},
      "t == TIMESTAMP '2022-12-23 03:56:01'",
      "SELECT t from tmp where t == TIMESTAMP '2022-12-23 03:56:01'");
  VELOX_ASSERT_THROW(
      assertSelectWithFilter(
          {"t"},
          {"t < TIMESTAMP '2000-09-12 22:36:29'"},
          "",
          "SELECT t from tmp where t < TIMESTAMP '2000-09-12 22:36:29'"),
      "Unsupported expression for range filter: lt(ROW[\"t\"],cast \"2000-09-12 22:36:29\" as TIMESTAMP)");
}

// A fixed core dump issue.
TEST_F(ParquetTableScanTest, map) {
  auto vector = makeMapVector<StringView, StringView>({{{"name", "gluten"}}});

  loadData(
      getExampleFilePath("type1.parquet"),
      ROW({"map"}, {MAP(VARCHAR(), VARCHAR())}),
      makeRowVector(
          {"map"},
          {
              vector,
          }));

  assertSelectWithFilter({"map"}, {}, "", "SELECT map FROM tmp");
}

// Array reader result has missing result.
// TEST_F(ParquetTableScanTest, array) {
//   auto vector = makeArrayVector<int32_t>({{1, 2, 3}});

//   loadData(
//       getExampleFilePath("old-repeated-int.parquet"),
//       ROW({"repeatedInt"}, {ARRAY(INTEGER())}),
//       makeRowVector(
//           {"repeatedInt"},
//           {
//               vector,
//           }));

//   assertSelectWithFilter({"repeatedInt"}, {}, "", "SELECT repeatedInt FROM
//   tmp");
// }

// Failed unit test on Velox map reader.
// TEST_F(ParquetTableScanTest, nestedMapWithStruct) {
//   auto vector = makeArrayVector<int32_t>({{1, 2, 3}});

//   loadData(
//       getExampleFilePath("nested-map-with-struct.parquet"),
//       ROW({"_1"}, {MAP(ROW({"_1", "_2"}, {INTEGER(), VARCHAR()}),
//       VARCHAR())}), makeRowVector(
//           {"_1"},
//           {
//               vector,
//           }));

//   assertSelectWithFilter({"_1"}, {}, "", "SELECT _1");
// }

// A fixed core dump issue.
TEST_F(ParquetTableScanTest, singleRowStruct) {
  auto vector = makeArrayVector<int32_t>({{1, 2, 3}});
  loadData(
      getExampleFilePath("single-row-struct.parquet"),
      ROW({"s"}, {ROW({"a", "b"}, {BIGINT(), BIGINT()})}),
      makeRowVector(
          {"s"},
          {
              vector,
          }));

  assertSelectWithFilter({"s"}, {}, "", "SELECT (0, 1)");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
