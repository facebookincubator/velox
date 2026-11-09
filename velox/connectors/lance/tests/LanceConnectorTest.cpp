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
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TempDirectoryPath.h"
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

  // Returns the path to the vectors.lance dataset: 20 rows, id int64 0..19,
  // vec = FixedSizeList<float,4> [i, 0, 0, 0], tag = 'even'/'odd'. No vector
  // index, so Lance flat-searches and results are exact.
  static std::string getVectorDatasetPath() {
    return test::getDataFilePath(
        "velox/connectors/lance/tests", "data/vectors.lance");
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

  // Build a TableScan plan node for a Lance vector search (nearest query).
  // The physical dataset path is provided separately via the split.
  core::PlanNodePtr makeVectorSearchPlan(
      const RowTypePtr& outputType,
      const LanceVectorSearchParams& params,
      const std::string& filter = "") {
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
                "vectors",
                /*datasetVersion=*/0,
                std::unordered_map<std::string, std::string>{},
                filter,
                params))
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

TEST_F(LanceConnectorTest, tableHandleVectorSearch) {
  LanceVectorSearchParams params;
  params.column = "vec";
  params.queryVector = {0.3f, 0.0f, 0.0f, 0.0f};
  params.k = 3;
  params.metric = "l2";
  params.nprobes = 10;
  auto handle = std::make_shared<LanceTableHandle>(
      kLanceConnectorId,
      "vectors",
      /*datasetVersion=*/0,
      std::unordered_map<std::string, std::string>{},
      /*filter=*/"",
      params);
  ASSERT_TRUE(handle->vectorSearch().has_value());
  const auto& vs = handle->vectorSearch().value();
  ASSERT_EQ(vs.column, "vec");
  ASSERT_THAT(vs.queryVector, testing::ElementsAre(0.3f, 0.0f, 0.0f, 0.0f));
  ASSERT_EQ(vs.k, 3);
  ASSERT_EQ(vs.metric, "l2");
  ASSERT_EQ(vs.nprobes, std::optional<uint32_t>(10));
  ASSERT_EQ(vs.ef, std::nullopt);
  ASSERT_EQ(vs.refineFactor, std::nullopt);
}

TEST_F(LanceConnectorTest, tableHandleWithoutVectorSearch) {
  auto handle =
      std::make_shared<LanceTableHandle>(kLanceConnectorId, "test_table");
  ASSERT_FALSE(handle->vectorSearch().has_value());
}

// toString must summarize the search (the full query vector may be thousands
// of floats) and must not change the existing no-search rendering.
TEST_F(LanceConnectorTest, tableHandleToStringWithVectorSearch) {
  LanceVectorSearchParams params;
  params.column = "vec";
  params.queryVector = {0.3f, 0.0f, 0.0f, 0.0f};
  params.k = 3;
  params.metric = "l2";
  auto handle = std::make_shared<LanceTableHandle>(
      kLanceConnectorId,
      "vectors",
      /*datasetVersion=*/0,
      std::unordered_map<std::string, std::string>{},
      /*filter=*/"",
      params);
  ASSERT_EQ(
      handle->toString(),
      "LanceTableHandle [table: vectors, version: latest, storageOptions: 0, "
      "filter: <none>, vectorSearch: {column: vec, dim: 4, k: 3, metric: l2}]");
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

// k-NN over the 20-row fixture: exactly k rows, nearest ids first, and the
// synthesized _distance column carries the squared-L2 distances.
TEST_F(LanceConnectorTest, vectorSearchNearestL2) {
  LanceVectorSearchParams params;
  params.column = "vec";
  params.queryVector = {0.3f, 0.0f, 0.0f, 0.0f};
  params.k = 3;
  params.metric = "l2";

  auto outputType = ROW({"id", "_distance"}, {BIGINT(), REAL()});
  auto plan = makeVectorSearchPlan(outputType, params);

  auto result = AssertQueryBuilder(plan)
                    .split(makeSplit(getVectorDatasetPath()))
                    .copyResults(pool());

  ASSERT_EQ(result->size(), 3);
  auto idVector = result->childAt(0)->asFlatVector<int64_t>();
  auto distanceVector = result->childAt(1)->asFlatVector<float>();
  ASSERT_NE(idVector, nullptr);
  ASSERT_NE(distanceVector, nullptr);

  // Lance returns nearest-first: id 0 (0.3^2), id 1 (0.7^2), id 2 (1.7^2).
  EXPECT_EQ(idVector->valueAt(0), 0);
  EXPECT_EQ(idVector->valueAt(1), 1);
  EXPECT_EQ(idVector->valueAt(2), 2);
  EXPECT_NEAR(distanceVector->valueAt(0), 0.09f, 1e-4);
  EXPECT_NEAR(distanceVector->valueAt(1), 0.49f, 1e-4);
  EXPECT_NEAR(distanceVector->valueAt(2), 2.89f, 1e-4);
}

// Fragment-scoped k-NN: restrict the nearest scan to a single fragment via
// prefilter and verify the result contains only that fragment's rows.
// The vectors.lance fixture has 2 fragments; the test discovers which row IDs
// belong to the first fragment, computes the expected top-k from those rows,
// and asserts the fragment-restricted vector search returns exactly that.
TEST_F(LanceConnectorTest, vectorSearchFragmentScoped) {
  // Discover the fragment IDs in the vectors.lance fixture.
  auto dataset = ::lance::Dataset::open(getVectorDatasetPath());
  auto allFragmentIds = dataset.fragment_ids();
  ASSERT_GE(allFragmentIds.size(), 2u);

  // Pick the first fragment and discover which row IDs it contains by running
  // a plain (non-vector) scan restricted to that single fragment.
  const uint64_t targetFragment = allFragmentIds[0];
  auto scalarType = ROW({"id"}, {BIGINT()});
  auto scalarPlan = makeTableScanPlan(scalarType, "vectors");
  auto scalarSplit = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId,
      getVectorDatasetPath(),
      std::vector<uint64_t>{targetFragment});
  auto scalarResult =
      AssertQueryBuilder(scalarPlan).split(scalarSplit).copyResults(pool());
  ASSERT_GT(scalarResult->size(), 0);

  // Collect the row IDs belonging to the target fragment.
  auto scalarIdVec = scalarResult->childAt(0)->asFlatVector<int64_t>();
  std::vector<int64_t> fragmentRowIds;
  fragmentRowIds.reserve(scalarResult->size());
  for (vector_size_t i = 0; i < scalarResult->size(); ++i) {
    fragmentRowIds.push_back(scalarIdVec->valueAt(i));
  }

  // Sort by L2 distance to the query vector [0.3, 0, 0, 0].
  // vec = [id, 0, 0, 0], so squared-L2 distance = (id - 0.3)^2.
  std::sort(
      fragmentRowIds.begin(), fragmentRowIds.end(), [](int64_t a, int64_t b) {
        double da = (a - 0.3) * (a - 0.3);
        double db = (b - 0.3) * (b - 0.3);
        return da < db;
      });

  const uint32_t k = 3;
  const auto expectedCount =
      std::min(static_cast<size_t>(k), fragmentRowIds.size());

  // Run a fragment-scoped vector search: k-NN restricted to the target
  // fragment. With the prefilter path, Lance uses the fragment subset as a
  // pre-ranking row mask and returns the exact top-k within those rows.
  LanceVectorSearchParams params;
  params.column = "vec";
  params.queryVector = {0.3f, 0.0f, 0.0f, 0.0f};
  params.k = k;
  params.metric = "l2";

  auto outputType = ROW({"id", "_distance"}, {BIGINT(), REAL()});
  auto plan = makeVectorSearchPlan(outputType, params);
  auto split = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId,
      getVectorDatasetPath(),
      std::vector<uint64_t>{targetFragment});
  auto result = AssertQueryBuilder(plan).split(split).copyResults(pool());

  ASSERT_EQ(static_cast<size_t>(result->size()), expectedCount);
  auto idVector = result->childAt(0)->asFlatVector<int64_t>();
  auto distanceVector = result->childAt(1)->asFlatVector<float>();
  ASSERT_NE(idVector, nullptr);
  ASSERT_NE(distanceVector, nullptr);

  // Build a set of the target fragment's row IDs for membership checks.
  std::set<int64_t> fragmentIdSet(fragmentRowIds.begin(), fragmentRowIds.end());

  // Every returned row must belong to the target fragment, ids must match
  // the expected nearest order, and distances must be non-decreasing.
  for (size_t i = 0; i < expectedCount; ++i) {
    EXPECT_TRUE(fragmentIdSet.count(idVector->valueAt(i)))
        << "id " << idVector->valueAt(i) << " does not belong to fragment "
        << targetFragment;
    EXPECT_EQ(idVector->valueAt(i), fragmentRowIds[i]);
    if (i > 0) {
      EXPECT_LE(distanceVector->valueAt(i - 1), distanceVector->valueAt(i));
    }
  }
}

// dot metric negates the dot product, so the largest vectors are nearest:
// verifies the metric override actually reaches the Lance scanner.
TEST_F(LanceConnectorTest, vectorSearchNearestDotFlipsOrder) {
  LanceVectorSearchParams params;
  params.column = "vec";
  params.queryVector = {0.3f, 0.0f, 0.0f, 0.0f};
  params.k = 3;
  params.metric = "dot";

  auto outputType = ROW({"id", "_distance"}, {BIGINT(), REAL()});
  auto plan = makeVectorSearchPlan(outputType, params);

  auto result = AssertQueryBuilder(plan)
                    .split(makeSplit(getVectorDatasetPath()))
                    .copyResults(pool());

  ASSERT_EQ(result->size(), 3);
  auto idVector = result->childAt(0)->asFlatVector<int64_t>();
  auto distanceVector = result->childAt(1)->asFlatVector<float>();
  EXPECT_EQ(idVector->valueAt(0), 19);
  EXPECT_EQ(idVector->valueAt(1), 18);
  EXPECT_EQ(idVector->valueAt(2), 17);
  EXPECT_NEAR(distanceVector->valueAt(0), -5.7f, 1e-4);
}

// A split listing every fragment ID with prefilter enabled must produce
// the same results as a whole-dataset scan: the fragment mask covers all
// rows, so the top-k is identical.
TEST_F(LanceConnectorTest, vectorSearchAllFragmentsMatchesGlobal) {
  LanceVectorSearchParams params;
  params.column = "vec";
  params.queryVector = {0.3f, 0.0f, 0.0f, 0.0f};
  params.k = 3;
  params.metric = "l2";

  auto outputType = ROW({"id", "_distance"}, {BIGINT(), REAL()});
  auto plan = makeVectorSearchPlan(outputType, params);

  // Discover the fragment ids so the split lists them explicitly.
  auto dataset = ::lance::Dataset::open(getVectorDatasetPath());
  auto fragmentIds = dataset.fragment_ids();
  ASSERT_FALSE(fragmentIds.empty());

  auto split = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId, getVectorDatasetPath(), fragmentIds);
  auto result = AssertQueryBuilder(plan).split(split).copyResults(pool());

  ASSERT_EQ(result->size(), 3);
  auto idVector = result->childAt(0)->asFlatVector<int64_t>();
  EXPECT_EQ(idVector->valueAt(0), 0);
  EXPECT_EQ(idVector->valueAt(1), 1);
  EXPECT_EQ(idVector->valueAt(2), 2);
}

// Projecting non-vector data columns alongside _distance works; the vec
// column itself need not be selected.
TEST_F(LanceConnectorTest, vectorSearchProjectsDataColumns) {
  LanceVectorSearchParams params;
  params.column = "vec";
  params.queryVector = {0.3f, 0.0f, 0.0f, 0.0f};
  params.k = 2;
  params.metric = "l2";

  auto outputType =
      ROW({"tag", "id", "_distance"}, {VARCHAR(), BIGINT(), REAL()});
  auto plan = makeVectorSearchPlan(outputType, params);

  auto result = AssertQueryBuilder(plan)
                    .split(makeSplit(getVectorDatasetPath()))
                    .copyResults(pool());

  ASSERT_EQ(result->size(), 2);
  auto tagVector = result->childAt(0)->asFlatVector<StringView>();
  auto idVector = result->childAt(1)->asFlatVector<int64_t>();
  EXPECT_EQ(idVector->valueAt(0), 0);
  EXPECT_EQ(tagVector->valueAt(0), "even");
  EXPECT_EQ(idVector->valueAt(1), 1);
  EXPECT_EQ(tagVector->valueAt(1), "odd");
}

TEST_F(LanceConnectorTest, vectorSearchUnknownMetricFails) {
  LanceVectorSearchParams params;
  params.column = "vec";
  params.queryVector = {0.3f, 0.0f, 0.0f, 0.0f};
  params.k = 3;
  params.metric = "hamming";

  auto outputType = ROW({"id", "_distance"}, {BIGINT(), REAL()});
  auto plan = makeVectorSearchPlan(outputType, params);

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan)
          .split(makeSplit(getVectorDatasetPath()))
          .copyResults(pool()),
      "Unsupported Lance vector distance metric: hamming");
}

namespace {

// Formats a 16-byte UUID as the canonical 8-4-4-4-12 lowercase hex string,
// the form LanceConnector's parseSegmentUuids expects.
std::string formatUuid(const std::array<uint8_t, 16>& bytes) {
  static const char* kHexDigits = "0123456789abcdef";
  std::string out;
  out.reserve(36);
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) {
      out.push_back('-');
    }
    out.push_back(kHexDigits[bytes[i] >> 4]);
    out.push_back(kHexDigits[bytes[i] & 0x0f]);
  }
  return out;
}

// Copies the vectors.lance fixture into tempDir and builds a single-partition
// IVF_FLAT L2 index named "vec_idx" on the vec column. One partition on the
// 20-row fixture keeps the index exhaustive, so indexed results are exact.
// Returns the copied dataset's path.
std::string makeIndexedVectorsDataset(
    const std::string& sourcePath,
    const std::string& tempDir) {
  const auto datasetPath = tempDir + "/vectors_indexed.lance";
  std::filesystem::copy(
      sourcePath, datasetPath, std::filesystem::copy_options::recursive);

  auto dataset = ::lance::Dataset::open(datasetPath);
  LanceVectorIndexParams indexParams{};
  indexParams.index_type = LANCE_INDEX_IVF_FLAT;
  indexParams.metric = LANCE_METRIC_L2;
  indexParams.num_partitions = 1;
  dataset.create_vector_index("vec", indexParams, "vec_idx");
  return datasetPath;
}

} // namespace

// INDEXED split: a split carrying the index's segment UUIDs must return the
// same top-k as a whole-dataset nearest scan, because together the segments
// cover every row of the freshly indexed fixture copy.
TEST_F(LanceConnectorTest, vectorSearchIndexSegmentScoped) {
  auto tempDir = common::testutil::TempDirectoryPath::create();
  const auto datasetPath =
      makeIndexedVectorsDataset(getVectorDatasetPath(), tempDir->getPath());

  // Reopen the dataset to pick up the index commit, then fetch the segment
  // UUIDs the coordinator would route by.
  auto dataset = ::lance::Dataset::open(datasetPath);
  auto segments = dataset.index_segments("vec_idx");
  ASSERT_FALSE(segments.empty());
  std::vector<std::string> segmentUuids;
  segmentUuids.reserve(segments.size());
  for (const auto& segment : segments) {
    segmentUuids.push_back(formatUuid(segment));
  }

  LanceVectorSearchParams params;
  params.column = "vec";
  params.queryVector = {0.3f, 0.0f, 0.0f, 0.0f};
  params.k = 3;
  params.metric = "l2";

  auto outputType = ROW({"id", "_distance"}, {BIGINT(), REAL()});
  auto plan = makeVectorSearchPlan(outputType, params);

  auto split = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId,
      datasetPath,
      std::vector<uint64_t>{},
      segmentUuids,
      /*flatSearch=*/false);
  auto result = AssertQueryBuilder(plan).split(split).copyResults(pool());

  // The single segment covers all 20 rows, so the segment-scoped search is
  // exhaustive: same nearest-first result as vectorSearchNearestL2.
  ASSERT_EQ(result->size(), 3);
  auto idVector = result->childAt(0)->asFlatVector<int64_t>();
  auto distanceVector = result->childAt(1)->asFlatVector<float>();
  ASSERT_NE(idVector, nullptr);
  ASSERT_NE(distanceVector, nullptr);
  EXPECT_EQ(idVector->valueAt(0), 0);
  EXPECT_EQ(idVector->valueAt(1), 1);
  EXPECT_EQ(idVector->valueAt(2), 2);
  EXPECT_NEAR(distanceVector->valueAt(0), 0.09f, 1e-4);
  EXPECT_NEAR(distanceVector->valueAt(1), 0.49f, 1e-4);
  EXPECT_NEAR(distanceVector->valueAt(2), 2.89f, 1e-4);
}

// FLAT split: fragmentIds plus flatSearch=true brute-forces k-NN over just
// those fragments with the index disabled. On the unindexed fixture this must
// match the fragment-scoped prefilter result exactly.
TEST_F(LanceConnectorTest, vectorSearchFlatSplit) {
  auto dataset = ::lance::Dataset::open(getVectorDatasetPath());
  auto allFragmentIds = dataset.fragment_ids();
  ASSERT_GE(allFragmentIds.size(), 2u);

  // Discover the first fragment's row IDs with a plain fragment-scoped scan.
  const uint64_t targetFragment = allFragmentIds[0];
  auto scalarPlan = makeTableScanPlan(ROW({"id"}, {BIGINT()}), "vectors");
  auto scalarSplit = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId,
      getVectorDatasetPath(),
      std::vector<uint64_t>{targetFragment});
  auto scalarResult =
      AssertQueryBuilder(scalarPlan).split(scalarSplit).copyResults(pool());
  ASSERT_GT(scalarResult->size(), 0);

  auto scalarIdVec = scalarResult->childAt(0)->asFlatVector<int64_t>();
  std::vector<int64_t> fragmentRowIds;
  fragmentRowIds.reserve(scalarResult->size());
  for (vector_size_t i = 0; i < scalarResult->size(); ++i) {
    fragmentRowIds.push_back(scalarIdVec->valueAt(i));
  }

  // vec = [id, 0, 0, 0], so squared-L2 distance to [0.3, 0, 0, 0] is
  // (id - 0.3)^2; sort to get the expected nearest order.
  std::sort(
      fragmentRowIds.begin(), fragmentRowIds.end(), [](int64_t a, int64_t b) {
        return (a - 0.3) * (a - 0.3) < (b - 0.3) * (b - 0.3);
      });

  const uint32_t k = 3;
  const auto expectedCount =
      std::min(static_cast<size_t>(k), fragmentRowIds.size());

  LanceVectorSearchParams params;
  params.column = "vec";
  params.queryVector = {0.3f, 0.0f, 0.0f, 0.0f};
  params.k = k;
  params.metric = "l2";

  auto outputType = ROW({"id", "_distance"}, {BIGINT(), REAL()});
  auto plan = makeVectorSearchPlan(outputType, params);
  auto split = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId,
      getVectorDatasetPath(),
      std::vector<uint64_t>{targetFragment},
      std::vector<std::string>{},
      /*flatSearch=*/true);
  auto result = AssertQueryBuilder(plan).split(split).copyResults(pool());

  ASSERT_EQ(static_cast<size_t>(result->size()), expectedCount);
  auto idVector = result->childAt(0)->asFlatVector<int64_t>();
  auto distanceVector = result->childAt(1)->asFlatVector<float>();
  ASSERT_NE(idVector, nullptr);
  ASSERT_NE(distanceVector, nullptr);
  for (size_t i = 0; i < expectedCount; ++i) {
    EXPECT_EQ(idVector->valueAt(i), fragmentRowIds[i]);
    if (i > 0) {
      EXPECT_LE(distanceVector->valueAt(i - 1), distanceVector->valueAt(i));
    }
  }
}

// A segment UUID that is not in canonical 8-4-4-4-12 form must be rejected
// by the connector's parser before anything reaches Lance.
TEST_F(LanceConnectorTest, vectorSearchSegmentUuidMalformed) {
  LanceVectorSearchParams params;
  params.column = "vec";
  params.queryVector = {0.3f, 0.0f, 0.0f, 0.0f};
  params.k = 3;
  params.metric = "l2";

  auto outputType = ROW({"id", "_distance"}, {BIGINT(), REAL()});
  auto plan = makeVectorSearchPlan(outputType, params);

  auto wrongLength = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId,
      getVectorDatasetPath(),
      std::vector<uint64_t>{},
      std::vector<std::string>{"not-a-uuid"},
      /*flatSearch=*/false);
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).split(wrongLength).copyResults(pool()),
      "UUID must be 36 characters");

  auto nonHex = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId,
      getVectorDatasetPath(),
      std::vector<uint64_t>{},
      std::vector<std::string>{"zz0e8400-e29b-41d4-a716-446655440000"},
      /*flatSearch=*/false);
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).split(nonHex).copyResults(pool()),
      "Non-hex character in UUID");
}

// A well-formed UUID that does not name any index segment of the dataset must
// surface a Lance error rather than silently returning rows.
TEST_F(LanceConnectorTest, vectorSearchSegmentUuidUnknown) {
  auto tempDir = common::testutil::TempDirectoryPath::create();
  const auto datasetPath =
      makeIndexedVectorsDataset(getVectorDatasetPath(), tempDir->getPath());

  LanceVectorSearchParams params;
  params.column = "vec";
  params.queryVector = {0.3f, 0.0f, 0.0f, 0.0f};
  params.k = 3;
  params.metric = "l2";

  auto outputType = ROW({"id", "_distance"}, {BIGINT(), REAL()});
  auto plan = makeVectorSearchPlan(outputType, params);

  auto split = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId,
      datasetPath,
      std::vector<uint64_t>{},
      std::vector<std::string>{"00000000-0000-0000-0000-000000000000"},
      /*flatSearch=*/false);

  // The all-zero UUID parses fine but matches no segment; Lance rejects it.
  // Match the callLance prefix rather than Lance's exact wording, which is
  // not part of the connector's contract.
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).split(split).copyResults(pool()), "lance error");
}
