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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/lance/LanceConnector.h"
#include "velox/connectors/lance/LanceConnectorSplit.h"
#include "velox/connectors/lance/LanceTableHandle.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::lance;
using namespace facebook::velox::exec::test;

// std::string rather than std::string_view: the connector registry and the
// handle constructors all take const std::string&.
static const std::string kLanceConnectorId = "lance-test";

class LanceConnectorTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    LanceConnectorFactory factory;
    auto lanceConnector = factory.newConnector(
        kLanceConnectorId,
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>()));
    connector::registerConnector(lanceConnector);
  }

  void TearDown() override {
    connector::unregisterConnector(kLanceConnectorId);
    OperatorTestBase::TearDown();
  }

  // Returns the path to the test_table1.lance dataset. The tests CMakeLists
  // copies the fixtures next to the test binary, and getDataFilePath resolves
  // relative to the working directory ctest runs the test in.
  static std::string getTestDatasetPath() {
    return test::getDataFilePath(
        "velox/connectors/lance/tests", "data/test_table1.lance");
  }

  // Build a TableScan plan node for the Lance connector.
  // tableName is the logical table name; the physical dataset path is provided
  // separately via the split. datasetVersion 0 reads the latest version and an
  // empty filter pushes no predicate.
  core::PlanNodePtr makeTableScanPlan(
      const RowTypePtr& outputType,
      const std::string& tableName,
      uint64_t datasetVersion = 0,
      const std::string& filter = "",
      std::unordered_map<std::string, std::string> storageOptions = {}) {
    connector::ColumnHandleMap assignments;
    for (const auto& name : outputType->names()) {
      assignments[name] = std::make_shared<LanceColumnHandle>(name);
    }
    return PlanBuilder()
        .startTableScan()
        .connectorId(kLanceConnectorId)
        .outputType(outputType)
        .tableHandle(
            std::make_shared<LanceTableHandle>(
                kLanceConnectorId,
                tableName,
                datasetVersion,
                std::move(storageOptions),
                filter))
        .assignments(std::move(assignments))
        .endTableScan()
        .planNode();
  }

  // Create a LanceConnectorSplit for the given dataset path.
  std::shared_ptr<LanceConnectorSplit> makeSplit(
      const std::string& datasetPath) {
    return std::make_shared<LanceConnectorSplit>(
        kLanceConnectorId, datasetPath);
  }

  // Scan the x column of test_table1 at the given version and return the
  // values read, which are distinct in every version of the fixture.
  std::set<int64_t> readXValues(
      uint64_t datasetVersion = 0,
      const std::string& filter = "") {
    auto plan = makeTableScanPlan(
        ROW({"x"}, {BIGINT()}), "test_table1", datasetVersion, filter);
    auto result = AssertQueryBuilder(plan)
                      .split(makeSplit(getTestDatasetPath()))
                      .copyResults(pool());

    auto xVector = result->childAt(0)->asFlatVector<int64_t>();
    EXPECT_NE(xVector, nullptr);

    std::set<int64_t> xValues;
    for (vector_size_t i = 0; i < result->size(); i++) {
      xValues.insert(xVector->valueAt(i));
    }
    return xValues;
  }
};

TEST_F(LanceConnectorTest, registrationWorks) {
  ASSERT_TRUE(connector::hasConnector(kLanceConnectorId));
}

TEST_F(LanceConnectorTest, tableHandleFields) {
  auto handle =
      std::make_shared<LanceTableHandle>(kLanceConnectorId, "test_table");
  ASSERT_EQ(handle->tableName(), "test_table");
  ASSERT_EQ(handle->name(), "test_table");
  ASSERT_EQ(handle->connectorId(), kLanceConnectorId);
  // Version 0 means latest, and nothing is pushed down by default.
  ASSERT_EQ(handle->datasetVersion(), 0);
  ASSERT_THAT(handle->storageOptions(), testing::IsEmpty());
  ASSERT_EQ(handle->filter(), "");
}

TEST_F(LanceConnectorTest, tableHandleVersionStorageOptionsAndFilter) {
  auto handle = std::make_shared<LanceTableHandle>(
      kLanceConnectorId,
      "test_table",
      /*datasetVersion=*/42,
      std::unordered_map<std::string, std::string>{
          {"region", "us-west-2"}, {"endpoint", "https://s3.example.com"}},
      /*filter=*/"x >= 2");
  ASSERT_EQ(handle->datasetVersion(), 42);
  ASSERT_THAT(
      handle->storageOptions(),
      testing::UnorderedElementsAre(
          std::pair<const std::string, std::string>{"region", "us-west-2"},
          std::pair<const std::string, std::string>{
              "endpoint", "https://s3.example.com"}));
  ASSERT_EQ(handle->filter(), "x >= 2");
}

TEST_F(LanceConnectorTest, tableHandleToString) {
  auto handle =
      std::make_shared<LanceTableHandle>(kLanceConnectorId, "my_table");
  ASSERT_EQ(
      handle->toString(),
      "LanceTableHandle [table: my_table, version: latest, storageOptions: 0, filter: <none>]");
}

// Storage option values can hold credentials, so toString must report only
// how many there are.
TEST_F(LanceConnectorTest, tableHandleToStringHidesStorageOptionValues) {
  auto handle = std::make_shared<LanceTableHandle>(
      kLanceConnectorId,
      "my_table",
      /*datasetVersion=*/7,
      std::unordered_map<std::string, std::string>{
          {"aws_secret_access_key", "super-secret"}},
      /*filter=*/"x == 1");
  const auto text = handle->toString();
  ASSERT_EQ(
      text,
      "LanceTableHandle [table: my_table, version: 7, storageOptions: 1, filter: x == 1]");
  ASSERT_THAT(text, testing::Not(testing::HasSubstr("super-secret")));
}

TEST_F(LanceConnectorTest, splitFields) {
  auto split = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId, "/tmp/test.lance");
  ASSERT_EQ(split->datasetPath, "/tmp/test.lance");
  ASSERT_EQ(split->connectorId, kLanceConnectorId);
  ASSERT_TRUE(split->fragmentIds.empty());
}

TEST_F(LanceConnectorTest, splitWithFragments) {
  std::vector<uint64_t> frags = {0, 2, 5};
  auto split = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId, "/tmp/test.lance", frags);
  ASSERT_EQ(split->datasetPath, "/tmp/test.lance");
  ASSERT_THAT(split->fragmentIds, testing::ElementsAre(0, 2, 5));
}

TEST_F(LanceConnectorTest, splitToString) {
  auto split = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId, "/data/my_table.lance");
  ASSERT_EQ(
      split->toString(),
      "[LanceConnectorSplit: connectorId lance-test, path /data/my_table.lance, fragments 0]");
}

TEST_F(LanceConnectorTest, columnHandleName) {
  LanceColumnHandle handle("my_column");
  ASSERT_EQ(handle.name(), "my_column");
}

// Reads all 4 columns from test_table1, verifies row count and values.
// The dataset has 2 fragments with 2 rows each (4 rows total).
// Row ordering is not guaranteed across fragments.
// Mirrors Java TestLanceFragmentPageSource.testFragmentScan.
TEST_F(LanceConnectorTest, fullScan) {
  auto datasetPath = getTestDatasetPath();
  auto outputType =
      ROW({"x", "y", "b", "c"}, {BIGINT(), BIGINT(), BIGINT(), BIGINT()});

  auto plan = makeTableScanPlan(outputType, "test_table1");

  auto result = AssertQueryBuilder(plan)
                    .split(makeSplit(datasetPath))
                    .copyResults(pool());

  // The connector reads all fragments (4 rows total).
  ASSERT_EQ(result->size(), 4);
  ASSERT_EQ(result->childrenSize(), 4);

  // Verify we got all 4 columns with the right type.
  auto xVector = result->childAt(0)->asFlatVector<int64_t>();
  auto yVector = result->childAt(1)->asFlatVector<int64_t>();
  auto bVector = result->childAt(2)->asFlatVector<int64_t>();
  auto cVector = result->childAt(3)->asFlatVector<int64_t>();
  ASSERT_NE(xVector, nullptr);
  ASSERT_NE(yVector, nullptr);
  ASSERT_NE(bVector, nullptr);
  ASSERT_NE(cVector, nullptr);

  // Collect x values and verify we got all expected values (order-independent).
  std::set<int64_t> xValues;
  for (vector_size_t i = 0; i < result->size(); i++) {
    xValues.insert(xVector->valueAt(i));
  }
  ASSERT_THAT(xValues, testing::UnorderedElementsAre(0, 1, 2, 3));
}

// Reads only columns b and x, verifies only 2 columns returned with correct
// values. Mirrors Java TestLanceFragmentPageSource.testColumnProjection.
TEST_F(LanceConnectorTest, columnProjection) {
  auto datasetPath = getTestDatasetPath();
  auto outputType = ROW({"b", "x"}, {BIGINT(), BIGINT()});

  auto plan = makeTableScanPlan(outputType, "test_table1");

  auto result = AssertQueryBuilder(plan)
                    .split(makeSplit(datasetPath))
                    .copyResults(pool());

  ASSERT_EQ(result->size(), 4);
  ASSERT_EQ(result->childrenSize(), 2);

  // Verify b and x values (order-independent).
  auto bVector = result->childAt(0)->asFlatVector<int64_t>();
  auto xVector = result->childAt(1)->asFlatVector<int64_t>();

  std::set<int64_t> bValues, xValues;
  for (vector_size_t i = 0; i < result->size(); i++) {
    bValues.insert(bVector->valueAt(i));
    xValues.insert(xVector->valueAt(i));
  }
  ASSERT_THAT(xValues, testing::UnorderedElementsAre(0, 1, 2, 3));
  // b column should have distinct values from the dataset.
  ASSERT_EQ(bValues.size(), 4);
}

// Reads columns c and x, verifies values including negative numbers.
// Mirrors Java TestLanceFragmentPageSource.testPartialColumnProjection.
TEST_F(LanceConnectorTest, partialColumnProjection) {
  auto datasetPath = getTestDatasetPath();
  auto outputType = ROW({"c", "x"}, {BIGINT(), BIGINT()});

  auto plan = makeTableScanPlan(outputType, "test_table1");

  auto result = AssertQueryBuilder(plan)
                    .split(makeSplit(datasetPath))
                    .copyResults(pool());

  ASSERT_EQ(result->size(), 4);
  ASSERT_EQ(result->childrenSize(), 2);

  // Verify c and x values (order-independent).
  auto cVector = result->childAt(0)->asFlatVector<int64_t>();
  auto xVector = result->childAt(1)->asFlatVector<int64_t>();

  std::set<int64_t> xValues;
  bool hasNegativeC = false;
  for (vector_size_t i = 0; i < result->size(); i++) {
    xValues.insert(xVector->valueAt(i));
    if (cVector->valueAt(i) < 0) {
      hasNegativeC = true;
    }
  }
  ASSERT_THAT(xValues, testing::UnorderedElementsAre(0, 1, 2, 3));
  // c column should contain negative values.
  ASSERT_TRUE(hasNegativeC);
}

// Reads a single column to test minimal projection.
TEST_F(LanceConnectorTest, singleColumnProjection) {
  auto datasetPath = getTestDatasetPath();
  auto outputType = ROW({"x"}, {BIGINT()});

  auto plan = makeTableScanPlan(outputType, "test_table1");

  auto result = AssertQueryBuilder(plan)
                    .split(makeSplit(datasetPath))
                    .copyResults(pool());

  ASSERT_EQ(result->size(), 4);
  ASSERT_EQ(result->childrenSize(), 1);

  auto xVector = result->childAt(0)->asFlatVector<int64_t>();
  std::set<int64_t> xValues;
  for (vector_size_t i = 0; i < result->size(); i++) {
    xValues.insert(xVector->valueAt(i));
  }
  ASSERT_THAT(xValues, testing::UnorderedElementsAre(0, 1, 2, 3));
}

// Version 6 is the latest version of the fixture, so requesting it explicitly
// must read exactly what version 0 (latest) reads.
TEST_F(LanceConnectorTest, timeTravelToLatestVersion) {
  ASSERT_THAT(
      readXValues(/*datasetVersion=*/6),
      testing::ContainerEq(readXValues(/*datasetVersion=*/0)));
}

// At least one older version must read differently from the latest one. This
// is what proves the version reaches lance_dataset_open: were it dropped,
// every version below would read identically.
TEST_F(LanceConnectorTest, timeTravelAcrossVersions) {
  const auto latest = readXValues(/*datasetVersion=*/0);

  bool anyVersionDiffers = false;
  for (uint64_t version = 1; version <= 6; ++version) {
    if (readXValues(version) != latest) {
      anyVersionDiffers = true;
    }
  }
  ASSERT_TRUE(anyVersionDiffers);
}

TEST_F(LanceConnectorTest, timeTravelToUnknownVersionFails) {
  auto plan = makeTableScanPlan(
      ROW({"x"}, {BIGINT()}), "test_table1", /*datasetVersion=*/9999);

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan)
          .split(makeSplit(getTestDatasetPath()))
          .copyResults(pool()),
      "Failed to open Lance dataset");
}

// x holds the distinct values 0..3, so a range predicate selects a known
// subset.
TEST_F(LanceConnectorTest, filterPushdownRange) {
  ASSERT_THAT(
      readXValues(/*datasetVersion=*/0, "x >= 2"),
      testing::UnorderedElementsAre(2, 3));
}

// lance-spark renders equality as "==" while presto-lance renders it as "=".
// Both dialects reach this connector, so both must be accepted.
TEST_F(LanceConnectorTest, filterPushdownEqualityInBothDialects) {
  ASSERT_THAT(
      readXValues(/*datasetVersion=*/0, "x == 1"),
      testing::UnorderedElementsAre(1));
  ASSERT_THAT(
      readXValues(/*datasetVersion=*/0, "x = 1"),
      testing::UnorderedElementsAre(1));
}

TEST_F(LanceConnectorTest, filterPushdownInList) {
  ASSERT_THAT(
      readXValues(/*datasetVersion=*/0, "x IN (0, 3)"),
      testing::UnorderedElementsAre(0, 3));
}

TEST_F(LanceConnectorTest, filterPushdownIsNotNull) {
  ASSERT_THAT(
      readXValues(/*datasetVersion=*/0, "x IS NOT NULL"),
      testing::UnorderedElementsAre(0, 1, 2, 3));
}

TEST_F(LanceConnectorTest, filterPushdownCombinedWithVersion) {
  ASSERT_THAT(
      readXValues(/*datasetVersion=*/6, "x >= 2"),
      testing::UnorderedElementsAre(2, 3));
}

// The predicate references x, which is not projected. presto-lance adds such
// filter columns to the scan projection because its JNI scanner needs them;
// if lance-c has the same requirement this test fails and LanceDataSource
// needs the same treatment.
TEST_F(LanceConnectorTest, filterPushdownOnNonOutputColumn) {
  auto plan = makeTableScanPlan(
      ROW({"y"}, {BIGINT()}), "test_table1", /*datasetVersion=*/0, "x >= 2");

  auto result = AssertQueryBuilder(plan)
                    .split(makeSplit(getTestDatasetPath()))
                    .copyResults(pool());

  ASSERT_EQ(result->size(), 2);
  ASSERT_EQ(result->childrenSize(), 1);
}

TEST_F(LanceConnectorTest, filterPushdownUnknownColumnFails) {
  auto plan = makeTableScanPlan(
      ROW({"x"}, {BIGINT()}),
      "test_table1",
      /*datasetVersion=*/0,
      "no_such_column > 1");

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan)
          .split(makeSplit(getTestDatasetPath()))
          .copyResults(pool()),
      "Failed to create scanner");
}

// Verifies that createDataSink throws NYI.
TEST_F(LanceConnectorTest, dataSinkNotSupported) {
  LanceConnectorFactory factory;
  auto connector = factory.newConnector(
      "lance-sink-test",
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));

  ASSERT_THROW(
      connector->createDataSink(
          ROW({"x"}, {BIGINT()}),
          nullptr,
          nullptr,
          connector::CommitStrategy::kNoCommit),
      VeloxRuntimeError);

  connector::unregisterConnector("lance-sink-test");
}
