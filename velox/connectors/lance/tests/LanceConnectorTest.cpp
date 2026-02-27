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
#include <set>

#include "velox/connectors/lance/LanceConnector.h"
#include "velox/connectors/lance/LanceConnectorSplit.h"
#include "velox/connectors/lance/LanceTableHandle.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::lance;
using namespace facebook::velox::exec::test;

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

  // Returns the path to the test_table1.lance dataset shipped with the tests.
  static std::string getTestDatasetPath() {
    // __FILE__ is .../velox/connectors/lance/tests/LanceConnectorTest.cpp
    std::string thisFile(__FILE__);
    auto dir = thisFile.substr(0, thisFile.rfind('/'));
    return dir + "/data/test_table1.lance";
  }

  // Build a TableScan plan node for the Lance connector.
  core::PlanNodePtr makeTableScanPlan(
      const RowTypePtr& outputType,
      const std::string& datasetPath) {
    connector::ColumnHandleMap assignments;
    for (const auto& name : outputType->names()) {
      assignments[name] = std::make_shared<LanceColumnHandle>(name);
    }
    return PlanBuilder()
        .startTableScan()
        .connectorId(kLanceConnectorId)
        .outputType(outputType)
        .tableHandle(
            std::make_shared<LanceTableHandle>(kLanceConnectorId, datasetPath))
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
};

TEST_F(LanceConnectorTest, registrationWorks) {
  ASSERT_TRUE(connector::hasConnector(kLanceConnectorId));
}

TEST_F(LanceConnectorTest, tableHandleFields) {
  auto handle = std::make_shared<LanceTableHandle>(
      kLanceConnectorId, "/tmp/test.lance");
  ASSERT_EQ(handle->datasetPath(), "/tmp/test.lance");
  ASSERT_EQ(handle->name(), "/tmp/test.lance");
  ASSERT_EQ(handle->connectorId(), kLanceConnectorId);
}

TEST_F(LanceConnectorTest, tableHandleToString) {
  auto handle = std::make_shared<LanceTableHandle>(
      kLanceConnectorId, "/data/my_table.lance");
  ASSERT_EQ(
      handle->toString(), "LanceTableHandle [path: /data/my_table.lance]");
}

TEST_F(LanceConnectorTest, splitFields) {
  auto split = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId, "/tmp/test.lance");
  ASSERT_EQ(split->datasetPath, "/tmp/test.lance");
  ASSERT_EQ(split->connectorId, kLanceConnectorId);
}

TEST_F(LanceConnectorTest, splitToString) {
  auto split = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId, "/data/my_table.lance");
  ASSERT_EQ(
      split->toString(),
      "[LanceConnectorSplit: connectorId lance-test, path /data/my_table.lance]");
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

  auto plan = makeTableScanPlan(outputType, datasetPath);

  auto result =
      AssertQueryBuilder(plan).split(makeSplit(datasetPath)).copyResults(pool());

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

  auto plan = makeTableScanPlan(outputType, datasetPath);

  auto result =
      AssertQueryBuilder(plan).split(makeSplit(datasetPath)).copyResults(pool());

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

  auto plan = makeTableScanPlan(outputType, datasetPath);

  auto result =
      AssertQueryBuilder(plan).split(makeSplit(datasetPath)).copyResults(pool());

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

  auto plan = makeTableScanPlan(outputType, datasetPath);

  auto result =
      AssertQueryBuilder(plan).split(makeSplit(datasetPath)).copyResults(pool());

  ASSERT_EQ(result->size(), 4);
  ASSERT_EQ(result->childrenSize(), 1);

  auto xVector = result->childAt(0)->asFlatVector<int64_t>();
  std::set<int64_t> xValues;
  for (vector_size_t i = 0; i < result->size(); i++) {
    xValues.insert(xVector->valueAt(i));
  }
  ASSERT_THAT(xValues, testing::UnorderedElementsAre(0, 1, 2, 3));
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
