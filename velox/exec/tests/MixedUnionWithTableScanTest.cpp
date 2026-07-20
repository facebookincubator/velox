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

#include <gtest/gtest.h>
#include <set>
#include "velox/core/QueryConfig.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

/// Tests for MixedUnion operator with TableScan sources.
/// Similar to koski's TestLocalWarehouseReadSplit.cpp but exercises
/// velox execution directly.
class MixedUnionWithTableScanTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
  }

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  /// Helper to create a MixedUnionNode from multiple plan node sources.
  std::shared_ptr<core::MixedUnionNode> makeMixedUnionNode(
      std::vector<core::PlanNodePtr> sources,
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
    return std::make_shared<core::MixedUnionNode>(
        planNodeIdGenerator->next(), std::move(sources));
  }

  /// Helper to create a table scan plan node.
  core::PlanNodePtr makeTableScanNode(
      const RowTypePtr& rowType,
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
    return PlanBuilder(planNodeIdGenerator).tableScan(rowType).planNode();
  }

  /// Helper to write test data to files and return the file paths.
  std::vector<std::shared_ptr<TempFilePath>> writeTestData(
      const std::vector<RowVectorPtr>& vectors) {
    auto filePaths = makeFilePaths(vectors.size());
    for (size_t i = 0; i < vectors.size(); ++i) {
      writeToFile(filePaths[i]->getPath(), vectors[i]);
    }
    return filePaths;
  }

  RowTypePtr testSchema_{ROW({"c0", "c1"}, {BIGINT(), VARCHAR()})};
};

/// Basic test: Union of two table scans.
TEST_F(MixedUnionWithTableScanTest, basicUnionOfTableScans) {
  // Create test data for two tables.
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<StringView>({"a", "b", "c"}),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({4, 5, 6}),
      makeFlatVector<StringView>({"d", "e", "f"}),
  });

  // Write data to files.
  auto filePath1 = TempFilePath::create();
  auto filePath2 = TempFilePath::create();
  writeToFile(filePath1->getPath(), data1);
  writeToFile(filePath2->getPath(), data2);

  // Build plan with MixedUnion of two table scans.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = asRowType(data1->type());

  core::PlanNodeId tableScan1Id;
  auto source1 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan1Id)
                     .planNode();

  core::PlanNodeId tableScan2Id;
  auto source2 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan2Id)
                     .planNode();

  auto unionNode = makeMixedUnionNode({source1, source2}, planNodeIdGenerator);

  // Execute query with splits.
  auto result =
      AssertQueryBuilder(unionNode)
          .split(tableScan1Id, makeHiveConnectorSplit(filePath1->getPath()))
          .split(tableScan2Id, makeHiveConnectorSplit(filePath2->getPath()))
          .copyResults(pool());

  ASSERT_EQ(result->size(), 6);

  // Validate result content and row order.
  // MixedUnion preserves row order: data1 rows come first, then data2 rows.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6}),
      makeFlatVector<StringView>({"a", "b", "c", "d", "e", "f"}),
  });
  facebook::velox::test::assertEqualVectors(expected, result);
}

/// Test union of a single table scan (edge case).
TEST_F(MixedUnionWithTableScanTest, singleTableScanUnion) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<StringView>({"a", "b", "c"}),
  });

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = asRowType(data->type());

  core::PlanNodeId tableScanId;
  auto source = PlanBuilder(planNodeIdGenerator)
                    .tableScan(rowType)
                    .capturePlanNodeId(tableScanId)
                    .planNode();

  auto unionNode = makeMixedUnionNode({source}, planNodeIdGenerator);

  auto result =
      AssertQueryBuilder(unionNode)
          .split(tableScanId, makeHiveConnectorSplit(filePath->getPath()))
          .copyResults(pool());

  ASSERT_EQ(result->size(), 3);

  // Validate result content and row order.
  facebook::velox::test::assertEqualVectors(data, result);
}

/// Test union with multiple table scans (more than 2).
TEST_F(MixedUnionWithTableScanTest, multipleTableScanUnion) {
  constexpr int kNumSources = 5;

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<std::shared_ptr<TempFilePath>> filePaths;
  std::vector<core::PlanNodePtr> sources;
  std::vector<core::PlanNodeId> tableScanIds;

  auto rowType = ROW({"c0"}, {BIGINT()});

  for (int i = 0; i < kNumSources; ++i) {
    auto data = makeRowVector({
        makeFlatVector<int64_t>({i * 10, i * 10 + 1}),
    });

    auto filePath = TempFilePath::create();
    writeToFile(filePath->getPath(), data);
    filePaths.push_back(filePath);

    core::PlanNodeId tableScanId;
    auto source = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(tableScanId)
                      .planNode();
    sources.push_back(source);
    tableScanIds.push_back(tableScanId);
  }

  auto unionNode = makeMixedUnionNode(std::move(sources), planNodeIdGenerator);

  auto queryBuilder = AssertQueryBuilder(unionNode);
  for (int i = 0; i < kNumSources; ++i) {
    queryBuilder.split(
        tableScanIds[i], makeHiveConnectorSplit(filePaths[i]->getPath()));
  }

  auto result = queryBuilder.copyResults(pool());
  ASSERT_EQ(result->size(), kNumSources * 2);

  // Validate result content and row order.
  // MixedUnion preserves row order: source 0, 1, 2, 3, 4.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 10, 11, 20, 21, 30, 31, 40, 41}),
  });
  facebook::velox::test::assertEqualVectors(expected, result);
}

/// Test union with empty table scan source.
TEST_F(MixedUnionWithTableScanTest, emptyTableScanSource) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<StringView>({"a", "b", "c"}),
  });
  auto emptyData = makeRowVector({
      makeFlatVector<int64_t>({}),
      makeFlatVector<StringView>({}),
  });

  auto filePath1 = TempFilePath::create();
  auto filePath2 = TempFilePath::create();
  writeToFile(filePath1->getPath(), data);
  writeToFile(filePath2->getPath(), emptyData);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = asRowType(data->type());

  core::PlanNodeId tableScan1Id;
  auto source1 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan1Id)
                     .planNode();

  core::PlanNodeId tableScan2Id;
  auto source2 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan2Id)
                     .planNode();

  auto unionNode = makeMixedUnionNode({source1, source2}, planNodeIdGenerator);

  auto result =
      AssertQueryBuilder(unionNode)
          .split(tableScan1Id, makeHiveConnectorSplit(filePath1->getPath()))
          .split(tableScan2Id, makeHiveConnectorSplit(filePath2->getPath()))
          .copyResults(pool());

  // Should only get data from non-empty source.
  ASSERT_EQ(result->size(), 3);

  // Validate result content and row order.
  facebook::velox::test::assertEqualVectors(data, result);
}

/// Test union with filter pushed down to table scan.
TEST_F(MixedUnionWithTableScanTest, unionWithFilteredTableScans) {
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<StringView>({"a", "b", "c", "d", "e"}),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({6, 7, 8, 9, 10}),
      makeFlatVector<StringView>({"f", "g", "h", "i", "j"}),
  });

  auto filePath1 = TempFilePath::create();
  auto filePath2 = TempFilePath::create();
  writeToFile(filePath1->getPath(), data1);
  writeToFile(filePath2->getPath(), data2);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = asRowType(data1->type());

  // Build table scans with filters.
  core::PlanNodeId tableScan1Id;
  auto source1 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan1Id)
                     .filter("c0 > 2")
                     .planNode();

  core::PlanNodeId tableScan2Id;
  auto source2 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan2Id)
                     .filter("c0 < 9")
                     .planNode();

  auto unionNode = makeMixedUnionNode({source1, source2}, planNodeIdGenerator);

  auto result =
      AssertQueryBuilder(unionNode)
          .split(tableScan1Id, makeHiveConnectorSplit(filePath1->getPath()))
          .split(tableScan2Id, makeHiveConnectorSplit(filePath2->getPath()))
          .copyResults(pool());

  // source1: 3,4,5 (3 rows), source2: 6,7,8 (3 rows).
  ASSERT_EQ(result->size(), 6);

  // Validate result content and row order.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({3, 4, 5, 6, 7, 8}),
      makeFlatVector<StringView>({"c", "d", "e", "f", "g", "h"}),
  });
  facebook::velox::test::assertEqualVectors(expected, result);
}

/// Test union with projection after table scan.
TEST_F(MixedUnionWithTableScanTest, unionWithProjectedTableScans) {
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({4, 5, 6}),
      makeFlatVector<int64_t>({40, 50, 60}),
  });

  auto filePath1 = TempFilePath::create();
  auto filePath2 = TempFilePath::create();
  writeToFile(filePath1->getPath(), data1);
  writeToFile(filePath2->getPath(), data2);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = asRowType(data1->type());

  // Build table scans with projections (sum of columns).
  core::PlanNodeId tableScan1Id;
  auto source1 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan1Id)
                     .project({"c0 + c1 as sum"})
                     .planNode();

  core::PlanNodeId tableScan2Id;
  auto source2 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan2Id)
                     .project({"c0 + c1 as sum"})
                     .planNode();

  auto unionNode = makeMixedUnionNode({source1, source2}, planNodeIdGenerator);

  auto result =
      AssertQueryBuilder(unionNode)
          .split(tableScan1Id, makeHiveConnectorSplit(filePath1->getPath()))
          .split(tableScan2Id, makeHiveConnectorSplit(filePath2->getPath()))
          .copyResults(pool());

  ASSERT_EQ(result->size(), 6);

  // Validate result content and row order.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({11, 22, 33, 44, 55, 66}),
  });
  facebook::velox::test::assertEqualVectors(expected, result);
}

/// Test union followed by aggregation.
TEST_F(MixedUnionWithTableScanTest, unionWithAggregation) {
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 2}),
      makeFlatVector<int64_t>({40, 50, 60}),
  });

  auto filePath1 = TempFilePath::create();
  auto filePath2 = TempFilePath::create();
  writeToFile(filePath1->getPath(), data1);
  writeToFile(filePath2->getPath(), data2);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = asRowType(data1->type());

  core::PlanNodeId tableScan1Id;
  auto source1 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan1Id)
                     .planNode();

  core::PlanNodeId tableScan2Id;
  auto source2 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan2Id)
                     .planNode();

  auto unionNode = makeMixedUnionNode({source1, source2}, planNodeIdGenerator);

  // Add aggregation after union.
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .addNode([&](auto, auto) { return unionNode; })
                  .singleAggregation({"c0"}, {"sum(c1)"})
                  .planNode();

  auto result =
      AssertQueryBuilder(plan)
          .split(tableScan1Id, makeHiveConnectorSplit(filePath1->getPath()))
          .split(tableScan2Id, makeHiveConnectorSplit(filePath2->getPath()))
          .copyResults(pool());

  // Group 1: 10 + 20 + 40 = 70.
  // Group 2: 30 + 50 + 60 = 140.
  ASSERT_EQ(result->size(), 2);
}

/// Test union with multiple splits per table scan.
TEST_F(MixedUnionWithTableScanTest, unionWithMultipleSplitsPerSource) {
  auto data1a = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
  });
  auto data1b = makeRowVector({
      makeFlatVector<int64_t>({3, 4}),
  });
  auto data2a = makeRowVector({
      makeFlatVector<int64_t>({5, 6}),
  });
  auto data2b = makeRowVector({
      makeFlatVector<int64_t>({7, 8}),
  });

  auto filePath1a = TempFilePath::create();
  auto filePath1b = TempFilePath::create();
  auto filePath2a = TempFilePath::create();
  auto filePath2b = TempFilePath::create();
  writeToFile(filePath1a->getPath(), data1a);
  writeToFile(filePath1b->getPath(), data1b);
  writeToFile(filePath2a->getPath(), data2a);
  writeToFile(filePath2b->getPath(), data2b);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = asRowType(data1a->type());

  core::PlanNodeId tableScan1Id;
  auto source1 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan1Id)
                     .planNode();

  core::PlanNodeId tableScan2Id;
  auto source2 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan2Id)
                     .planNode();

  auto unionNode = makeMixedUnionNode({source1, source2}, planNodeIdGenerator);

  // Add multiple splits per table scan.
  auto result =
      AssertQueryBuilder(unionNode)
          .split(tableScan1Id, makeHiveConnectorSplit(filePath1a->getPath()))
          .split(tableScan1Id, makeHiveConnectorSplit(filePath1b->getPath()))
          .split(tableScan2Id, makeHiveConnectorSplit(filePath2a->getPath()))
          .split(tableScan2Id, makeHiveConnectorSplit(filePath2b->getPath()))
          .copyResults(pool());

  ASSERT_EQ(result->size(), 8);

  // Validate result content and row order.
  // MixedUnion preserves row order: data1a, data1b, data2a, data2b.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 5, 6, 3, 4, 7, 8}),
  });
  facebook::velox::test::assertEqualVectors(expected, result);
}

/// Test union with null values in data.
TEST_F(MixedUnionWithTableScanTest, unionWithNullValues) {
  auto data1 = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 3}),
      makeNullableFlatVector<StringView>({"a", "b", std::nullopt}),
  });
  auto data2 = makeRowVector({
      makeNullableFlatVector<int64_t>({std::nullopt, 5, 6}),
      makeNullableFlatVector<StringView>({std::nullopt, "e", "f"}),
  });

  auto filePath1 = TempFilePath::create();
  auto filePath2 = TempFilePath::create();
  writeToFile(filePath1->getPath(), data1);
  writeToFile(filePath2->getPath(), data2);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = asRowType(data1->type());

  core::PlanNodeId tableScan1Id;
  auto source1 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan1Id)
                     .planNode();

  core::PlanNodeId tableScan2Id;
  auto source2 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan2Id)
                     .planNode();

  auto unionNode = makeMixedUnionNode({source1, source2}, planNodeIdGenerator);

  auto result =
      AssertQueryBuilder(unionNode)
          .split(tableScan1Id, makeHiveConnectorSplit(filePath1->getPath()))
          .split(tableScan2Id, makeHiveConnectorSplit(filePath2->getPath()))
          .copyResults(pool());

  ASSERT_EQ(result->size(), 6);

  // Validate result content and row order, including nulls.
  auto expected = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 3, std::nullopt, 5, 6}),
      makeNullableFlatVector<StringView>(
          {"a", "b", std::nullopt, std::nullopt, "e", "f"}),
  });
  facebook::velox::test::assertEqualVectors(expected, result);
}

/// Test union with various data types.
TEST_F(MixedUnionWithTableScanTest, unionWithVariousTypes) {
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeFlatVector<double>({1.1, 2.2}),
      makeFlatVector<bool>({true, false}),
      makeFlatVector<StringView>({"hello", "world"}),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({3, 4}),
      makeFlatVector<double>({3.3, 4.4}),
      makeFlatVector<bool>({false, true}),
      makeFlatVector<StringView>({"foo", "bar"}),
  });

  auto filePath1 = TempFilePath::create();
  auto filePath2 = TempFilePath::create();
  writeToFile(filePath1->getPath(), data1);
  writeToFile(filePath2->getPath(), data2);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = asRowType(data1->type());

  core::PlanNodeId tableScan1Id;
  auto source1 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan1Id)
                     .planNode();

  core::PlanNodeId tableScan2Id;
  auto source2 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan2Id)
                     .planNode();

  auto unionNode = makeMixedUnionNode({source1, source2}, planNodeIdGenerator);

  auto result =
      AssertQueryBuilder(unionNode)
          .split(tableScan1Id, makeHiveConnectorSplit(filePath1->getPath()))
          .split(tableScan2Id, makeHiveConnectorSplit(filePath2->getPath()))
          .copyResults(pool());

  ASSERT_EQ(result->size(), 4);

  // Validate result content and row order.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
      makeFlatVector<double>({1.1, 2.2, 3.3, 4.4}),
      makeFlatVector<bool>({true, false, false, true}),
      makeFlatVector<StringView>({"hello", "world", "foo", "bar"}),
  });
  facebook::velox::test::assertEqualVectors(expected, result);
}

/// Test union with unequal source sizes.
TEST_F(MixedUnionWithTableScanTest, unionWithUnequalSourceSizes) {
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({6, 7}),
  });
  auto data3 = makeRowVector({
      makeFlatVector<int64_t>({8, 9, 10, 11, 12, 13, 14}),
  });

  auto filePath1 = TempFilePath::create();
  auto filePath2 = TempFilePath::create();
  auto filePath3 = TempFilePath::create();
  writeToFile(filePath1->getPath(), data1);
  writeToFile(filePath2->getPath(), data2);
  writeToFile(filePath3->getPath(), data3);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = asRowType(data1->type());

  core::PlanNodeId tableScan1Id;
  auto source1 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan1Id)
                     .planNode();

  core::PlanNodeId tableScan2Id;
  auto source2 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan2Id)
                     .planNode();

  core::PlanNodeId tableScan3Id;
  auto source3 = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(tableScan3Id)
                     .planNode();

  auto unionNode =
      makeMixedUnionNode({source1, source2, source3}, planNodeIdGenerator);

  auto result =
      AssertQueryBuilder(unionNode)
          .split(tableScan1Id, makeHiveConnectorSplit(filePath1->getPath()))
          .split(tableScan2Id, makeHiveConnectorSplit(filePath2->getPath()))
          .split(tableScan3Id, makeHiveConnectorSplit(filePath3->getPath()))
          .copyResults(pool());

  ASSERT_EQ(result->size(), 14);

  // Validate result content and row order.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}),
  });
  facebook::velox::test::assertEqualVectors(expected, result);
}

/// Test union with large data volume.
TEST_F(MixedUnionWithTableScanTest, unionWithLargeDataVolume) {
  constexpr int kRowsPerSource = 1000;
  constexpr int kNumSources = 3;

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<std::shared_ptr<TempFilePath>> filePaths;
  std::vector<core::PlanNodePtr> sources;
  std::vector<core::PlanNodeId> tableScanIds;

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), DOUBLE()});

  for (int s = 0; s < kNumSources; ++s) {
    auto data = makeRowVector({
        makeFlatVector<int64_t>(
            kRowsPerSource, [s](auto row) { return s * 10000 + row; }),
        makeFlatVector<double>(
            kRowsPerSource, [](auto row) { return row * 0.1; }),
    });

    auto filePath = TempFilePath::create();
    writeToFile(filePath->getPath(), data);
    filePaths.push_back(filePath);

    core::PlanNodeId tableScanId;
    auto source = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(tableScanId)
                      .planNode();
    sources.push_back(source);
    tableScanIds.push_back(tableScanId);
  }

  auto unionNode = makeMixedUnionNode(std::move(sources), planNodeIdGenerator);

  auto queryBuilder = AssertQueryBuilder(unionNode);
  for (int i = 0; i < kNumSources; ++i) {
    queryBuilder.split(
        tableScanIds[i], makeHiveConnectorSplit(filePaths[i]->getPath()));
  }

  auto result = queryBuilder.copyResults(pool());
  ASSERT_EQ(result->size(), kRowsPerSource * kNumSources);

  // Validate result content and row order.
  // MixedUnion preserves row order: source 0 rows, then source 1 rows, etc.
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(
          kRowsPerSource * kNumSources,
          [](auto row) {
            int source = row / 1000;
            int rowInSource = row % 1000;
            return source * 10000 + rowInSource;
          }),
      makeFlatVector<double>(
          kRowsPerSource * kNumSources,
          [](auto row) {
            int rowInSource = row % 1000;
            return rowInSource * 0.1;
          }),
  });
  facebook::velox::test::assertEqualVectors(expected, result);
}

/// Test that batchSizeHint on ConnectorSplit controls the TableScan read rate,
/// allowing split generators to produce rows at different rates per source.
/// This is the "automatic batch sizing at TableScan" option for MixedUnion:
/// the split generator stamps each split with a batch size proportional to its
/// share of the union, so sources with fewer splits naturally read fewer rows
/// per batch and both sides exhaust at approximately the same rate.
TEST_F(MixedUnionWithTableScanTest, batchSizeHintControlsReadRate) {
  // Source A: 500 rows. Simulates 5 splits worth of data → batchSizeHint = 100.
  // Source B: 200 rows. Simulates 2 splits worth of data → batchSizeHint = 40.
  // Ratio 5:2 means A should read 2.5x more rows per batch than B.
  constexpr int kRowsA = 500;
  constexpr int kRowsB = 200;
  constexpr int kBatchHintA = 100;
  constexpr int kBatchHintB = 40;

  auto dataA = makeRowVector({
      makeFlatVector<int64_t>(kRowsA, [](auto row) { return row; }),
  });
  auto dataB = makeRowVector({
      makeFlatVector<int64_t>(kRowsB, [](auto row) { return row + 10'000; }),
  });

  auto filePathA = TempFilePath::create();
  auto filePathB = TempFilePath::create();
  writeToFile(filePathA->getPath(), dataA);
  writeToFile(filePathB->getPath(), dataB);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = ROW({"c0"}, {BIGINT()});

  core::PlanNodeId scanAId;
  auto sourceA = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(scanAId)
                     .planNode();

  core::PlanNodeId scanBId;
  auto sourceB = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(scanBId)
                     .planNode();

  auto unionNode = makeMixedUnionNode({sourceA, sourceB}, planNodeIdGenerator);

  // Build splits with batchSizeHint set.
  auto splitA = HiveConnectorSplitBuilder(filePathA->getPath())
                    .batchSizeHint(kBatchHintA)
                    .build();
  auto splitB = HiveConnectorSplitBuilder(filePathB->getPath())
                    .batchSizeHint(kBatchHintB)
                    .build();

  // Use readCursor to collect individual output batches so we can inspect
  // per-batch composition.
  CursorParameters params;
  params.planNode = unionNode;
  params.maxDrivers = 1;
  params.serialExecution = true;

  auto [cursor, results] = readCursor(params, [&](TaskCursor* taskCursor) {
    if (taskCursor->noMoreSplits()) {
      return;
    }
    auto task = taskCursor->task();
    task->addSplit(scanAId, Split(splitA));
    task->addSplit(scanBId, Split(splitB));
    task->noMoreSplits(scanAId);
    task->noMoreSplits(scanBId);
    taskCursor->setNoMoreSplits();
  });

  // Verify all rows are present.
  std::set<int64_t> allValues;
  for (const auto& batch : results) {
    auto* col = batch->childAt(0)->asFlatVector<int64_t>();
    for (int i = 0; i < batch->size(); ++i) {
      allValues.insert(col->valueAt(i));
    }
  }
  EXPECT_EQ(allValues.size(), kRowsA + kRowsB);

  // Count rows from each source across all batches.
  int totalA = 0;
  int totalB = 0;
  for (const auto& batch : results) {
    auto* col = batch->childAt(0)->asFlatVector<int64_t>();
    for (int i = 0; i < batch->size(); ++i) {
      if (col->valueAt(i) < 10'000) {
        ++totalA;
      } else {
        ++totalB;
      }
    }
  }
  EXPECT_EQ(totalA, kRowsA);
  EXPECT_EQ(totalB, kRowsB);

  // Inspect batches that contain rows from both sources (mixed batches).
  // In such batches, the ratio of A-rows to B-rows should approximately
  // reflect the batchSizeHint ratio (100:40 = 2.5:1).
  int mixedBatchARows = 0;
  int mixedBatchBRows = 0;
  for (const auto& batch : results) {
    auto* col = batch->childAt(0)->asFlatVector<int64_t>();
    int batchA = 0, batchB = 0;
    for (int i = 0; i < batch->size(); ++i) {
      if (col->valueAt(i) < 10'000) {
        ++batchA;
      } else {
        ++batchB;
      }
    }
    if (batchA > 0 && batchB > 0) {
      mixedBatchARows += batchA;
      mixedBatchBRows += batchB;
    }
  }

  // If we have mixed batches, A should contribute more rows than B,
  // reflecting the 2.5:1 batch size hint ratio.
  if (mixedBatchARows > 0 && mixedBatchBRows > 0) {
    double observedRatio =
        static_cast<double>(mixedBatchARows) / mixedBatchBRows;
    double expectedRatio =
        static_cast<double>(kBatchHintA) / kBatchHintB; // 2.5
    // Allow generous tolerance — the exact ratio depends on file layout and
    // internal buffering. The key property is that A contributes more.
    EXPECT_GT(observedRatio, expectedRatio * 0.3)
        << "A should contribute proportionally more rows in mixed batches. "
        << "A=" << mixedBatchARows << " B=" << mixedBatchBRows
        << " observed ratio=" << observedRatio
        << " expected~=" << expectedRatio;
    EXPECT_GT(mixedBatchARows, mixedBatchBRows)
        << "A (hint=" << kBatchHintA
        << ") should have more rows than B (hint=" << kBatchHintB
        << ") in mixed batches";
  }
}

/// Test that batchSizeHint=0 (default) does not affect TableScan behavior —
/// it falls through to the normal dynamic batch sizing.
TEST_F(MixedUnionWithTableScanTest, zeroBatchSizeHintUsesDefault) {
  constexpr int kRows = 100;

  auto data = makeRowVector({
      makeFlatVector<int64_t>(kRows, folly::identity),
  });

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = ROW({"c0"}, {BIGINT()});

  core::PlanNodeId scanId;
  auto source = PlanBuilder(planNodeIdGenerator)
                    .tableScan(rowType)
                    .capturePlanNodeId(scanId)
                    .planNode();

  auto unionNode = makeMixedUnionNode({source}, planNodeIdGenerator);

  // Split with batchSizeHint=0 (default).
  auto split = HiveConnectorSplitBuilder(filePath->getPath()).build();
  ASSERT_EQ(split->batchSizeHint, 0);

  auto result =
      AssertQueryBuilder(unionNode).split(scanId, split).copyResults(pool());

  ASSERT_EQ(result->size(), kRows);

  auto* col = result->childAt(0)->asFlatVector<int64_t>();
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(col->valueAt(i), i);
  }
}

// Verify that batchSizeHint directly controls the per-batch row count produced
// by TableScan. With a single source through MixedUnion, each output batch
// should have exactly the hinted number of rows (except the final batch).
TEST_F(MixedUnionWithTableScanTest, batchSizeHintDirectlyControlsBatchSize) {
  constexpr int kRows = 500;
  constexpr int32_t kBatchHint = 50;

  auto data = makeRowVector({
      makeFlatVector<int64_t>(kRows, folly::identity),
  });

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = ROW({"c0"}, {BIGINT()});

  core::PlanNodeId scanId;
  auto source = PlanBuilder(planNodeIdGenerator)
                    .tableScan(rowType)
                    .capturePlanNodeId(scanId)
                    .planNode();

  auto unionNode = makeMixedUnionNode({source}, planNodeIdGenerator);

  auto split = HiveConnectorSplitBuilder(filePath->getPath())
                   .batchSizeHint(kBatchHint)
                   .build();

  CursorParameters params;
  params.planNode = unionNode;
  params.maxDrivers = 1;
  params.serialExecution = true;

  auto [cursor, results] = readCursor(params, [&](TaskCursor* taskCursor) {
    if (taskCursor->noMoreSplits()) {
      return;
    }
    auto task = taskCursor->task();
    task->addSplit(scanId, Split(split));
    task->noMoreSplits(scanId);
    taskCursor->setNoMoreSplits();
  });

  int totalRows = 0;
  for (const auto& batch : results) {
    totalRows += batch->size();
  }
  EXPECT_EQ(totalRows, kRows);

  // With 500 rows and hint=50, we expect at least 10 batches.
  ASSERT_GE(results.size(), kRows / kBatchHint);
  for (size_t i = 0; i + 1 < results.size(); ++i) {
    EXPECT_EQ(results[i]->size(), kBatchHint)
        << "Batch " << i << " should have " << kBatchHint << " rows";
  }
  EXPECT_LE(results.back()->size(), kBatchHint);
}

// Verify the priority order: batchSizeHint (split-level) takes precedence
// over outputBatchRowsOverride (query-level config). The hint is a specific
// signal from the split generator for proportional mixing; a generic
// query-level override must not silently destroy the ratio.
TEST_F(
    MixedUnionWithTableScanTest,
    batchSizeHintTakesPriorityOverOutputBatchRowsOverride) {
  constexpr int kRows = 500;
  constexpr int32_t kBatchHint = 50;
  constexpr uint32_t kOverride = 75;

  auto data = makeRowVector({
      makeFlatVector<int64_t>(kRows, folly::identity),
  });

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = ROW({"c0"}, {BIGINT()});

  core::PlanNodeId scanId;
  auto source = PlanBuilder(planNodeIdGenerator)
                    .tableScan(rowType)
                    .capturePlanNodeId(scanId)
                    .planNode();

  auto unionNode = makeMixedUnionNode({source}, planNodeIdGenerator);

  auto split = HiveConnectorSplitBuilder(filePath->getPath())
                   .batchSizeHint(kBatchHint)
                   .build();

  CursorParameters params;
  params.planNode = unionNode;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.queryConfigs[core::QueryConfig::kTableScanOutputBatchRowsOverride] =
      folly::to<std::string>(kOverride);

  auto [cursor, results] = readCursor(params, [&](TaskCursor* taskCursor) {
    if (taskCursor->noMoreSplits()) {
      return;
    }
    auto task = taskCursor->task();
    task->addSplit(scanId, Split(split));
    task->noMoreSplits(scanId);
    taskCursor->setNoMoreSplits();
  });

  int totalRows = 0;
  for (const auto& batch : results) {
    totalRows += batch->size();
  }
  EXPECT_EQ(totalRows, kRows);

  // Hint should win: batches should be kBatchHint, not kOverride.
  ASSERT_GE(results.size(), kRows / kBatchHint);
  for (size_t i = 0; i + 1 < results.size(); ++i) {
    EXPECT_EQ(results[i]->size(), kBatchHint)
        << "Batch " << i << " should use batchSizeHint (" << kBatchHint
        << "), not override (" << kOverride << ")";
  }
  EXPECT_LE(results.back()->size(), kBatchHint);
}

// Verify proportional mixing with three sources. Each source's batchSizeHint
// is proportional to its data volume, so mixed output batches should reflect
// the 3:2:1 ratio.
TEST_F(MixedUnionWithTableScanTest, threeSourceProportionalBatchSizeHints) {
  constexpr int kRowsA = 300;
  constexpr int kRowsB = 200;
  constexpr int kRowsC = 100;
  constexpr int32_t kHintA = 30;
  constexpr int32_t kHintB = 20;
  constexpr int32_t kHintC = 10;

  auto dataA = makeRowVector({
      makeFlatVector<int64_t>(kRowsA, [](auto row) { return row; }),
  });
  auto dataB = makeRowVector({
      makeFlatVector<int64_t>(kRowsB, [](auto row) { return row + 10'000; }),
  });
  auto dataC = makeRowVector({
      makeFlatVector<int64_t>(kRowsC, [](auto row) { return row + 20'000; }),
  });

  auto filePathA = TempFilePath::create();
  auto filePathB = TempFilePath::create();
  auto filePathC = TempFilePath::create();
  writeToFile(filePathA->getPath(), dataA);
  writeToFile(filePathB->getPath(), dataB);
  writeToFile(filePathC->getPath(), dataC);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = ROW({"c0"}, {BIGINT()});

  core::PlanNodeId scanAId;
  auto sourceA = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(scanAId)
                     .planNode();
  core::PlanNodeId scanBId;
  auto sourceB = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(scanBId)
                     .planNode();
  core::PlanNodeId scanCId;
  auto sourceC = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(scanCId)
                     .planNode();

  auto unionNode =
      makeMixedUnionNode({sourceA, sourceB, sourceC}, planNodeIdGenerator);

  auto splitA = HiveConnectorSplitBuilder(filePathA->getPath())
                    .batchSizeHint(kHintA)
                    .build();
  auto splitB = HiveConnectorSplitBuilder(filePathB->getPath())
                    .batchSizeHint(kHintB)
                    .build();
  auto splitC = HiveConnectorSplitBuilder(filePathC->getPath())
                    .batchSizeHint(kHintC)
                    .build();

  CursorParameters params;
  params.planNode = unionNode;
  params.maxDrivers = 1;
  params.serialExecution = true;

  auto [cursor, results] = readCursor(params, [&](TaskCursor* taskCursor) {
    if (taskCursor->noMoreSplits()) {
      return;
    }
    auto task = taskCursor->task();
    task->addSplit(scanAId, Split(splitA));
    task->addSplit(scanBId, Split(splitB));
    task->addSplit(scanCId, Split(splitC));
    task->noMoreSplits(scanAId);
    task->noMoreSplits(scanBId);
    task->noMoreSplits(scanCId);
    taskCursor->setNoMoreSplits();
  });

  // Verify all rows present.
  int totalA = 0, totalB = 0, totalC = 0;
  for (const auto& batch : results) {
    auto* col = batch->childAt(0)->asFlatVector<int64_t>();
    for (int i = 0; i < batch->size(); ++i) {
      auto value = col->valueAt(i);
      if (value < 10'000) {
        ++totalA;
      } else if (value < 20'000) {
        ++totalB;
      } else {
        ++totalC;
      }
    }
  }
  EXPECT_EQ(totalA, kRowsA);
  EXPECT_EQ(totalB, kRowsB);
  EXPECT_EQ(totalC, kRowsC);

  // In batches that contain rows from all three sources, check that
  // A > B > C, reflecting the 3:2:1 hint ratio.
  int mixedA = 0, mixedB = 0, mixedC = 0;
  for (const auto& batch : results) {
    auto* col = batch->childAt(0)->asFlatVector<int64_t>();
    int countA = 0, countB = 0, countC = 0;
    for (int i = 0; i < batch->size(); ++i) {
      auto value = col->valueAt(i);
      if (value < 10'000) {
        ++countA;
      } else if (value < 20'000) {
        ++countB;
      } else {
        ++countC;
      }
    }
    if (countA > 0 && countB > 0 && countC > 0) {
      mixedA += countA;
      mixedB += countB;
      mixedC += countC;
    }
  }

  if (mixedA > 0 && mixedB > 0 && mixedC > 0) {
    EXPECT_GT(mixedA, mixedB)
        << "A (hint=" << kHintA
        << ") should contribute more than B (hint=" << kHintB
        << ") in mixed batches";
    EXPECT_GT(mixedB, mixedC)
        << "B (hint=" << kHintB
        << ") should contribute more than C (hint=" << kHintC
        << ") in mixed batches";
  }
}

// Verify HiveConnectorSplitBuilder propagates batchSizeHint correctly.
TEST_F(
    MixedUnionWithTableScanTest,
    hiveConnectorSplitBuilderSetsBatchSizeHint) {
  auto splitWithHint = HiveConnectorSplitBuilder("file:///tmp/test.dwrf")
                           .batchSizeHint(42)
                           .build();
  EXPECT_EQ(splitWithHint->batchSizeHint, 42);

  auto splitWithZeroHint = HiveConnectorSplitBuilder("file:///tmp/test.dwrf")
                               .batchSizeHint(0)
                               .build();
  EXPECT_EQ(splitWithZeroHint->batchSizeHint, 0);

  // Default (no batchSizeHint call) should be 0.
  auto splitDefault =
      HiveConnectorSplitBuilder("file:///tmp/test.dwrf").build();
  EXPECT_EQ(splitDefault->batchSizeHint, 0);
}

// Verify that a small batchSizeHint (e.g., 5) produces many small batches,
// confirming the hint is respected even at low values.
TEST_F(MixedUnionWithTableScanTest, batchSizeHintWithSmallValue) {
  constexpr int kRows = 100;
  constexpr int32_t kBatchHint = 5;

  auto data = makeRowVector({
      makeFlatVector<int64_t>(kRows, folly::identity),
  });

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = ROW({"c0"}, {BIGINT()});

  core::PlanNodeId scanId;
  auto source = PlanBuilder(planNodeIdGenerator)
                    .tableScan(rowType)
                    .capturePlanNodeId(scanId)
                    .planNode();

  auto unionNode = makeMixedUnionNode({source}, planNodeIdGenerator);

  auto split = HiveConnectorSplitBuilder(filePath->getPath())
                   .batchSizeHint(kBatchHint)
                   .build();

  CursorParameters params;
  params.planNode = unionNode;
  params.maxDrivers = 1;
  params.serialExecution = true;

  auto [cursor, results] = readCursor(params, [&](TaskCursor* taskCursor) {
    if (taskCursor->noMoreSplits()) {
      return;
    }
    auto task = taskCursor->task();
    task->addSplit(scanId, Split(split));
    task->noMoreSplits(scanId);
    taskCursor->setNoMoreSplits();
  });

  int totalRows = 0;
  for (const auto& batch : results) {
    totalRows += batch->size();
  }
  EXPECT_EQ(totalRows, kRows);

  // With hint=5 and 100 rows, we expect at least 20 batches.
  EXPECT_GE(results.size(), kRows / kBatchHint);

  for (size_t i = 0; i + 1 < results.size(); ++i) {
    EXPECT_EQ(results[i]->size(), kBatchHint)
        << "Batch " << i << " should have " << kBatchHint << " rows";
  }
  EXPECT_LE(results.back()->size(), kBatchHint);

  // Verify data integrity — all values present.
  std::set<int64_t> allValues;
  for (const auto& batch : results) {
    auto* col = batch->childAt(0)->asFlatVector<int64_t>();
    for (int i = 0; i < batch->size(); ++i) {
      allValues.insert(col->valueAt(i));
    }
  }
  EXPECT_EQ(allValues.size(), kRows);
}

// Verify that each mixed output batch contains exactly the hinted number of
// rows from each source. Uses data sizes that are exact multiples of the
// hints so both sources exhaust in the same number of cycles, producing
// deterministic per-batch composition.
TEST_F(MixedUnionWithTableScanTest, perBatchMixRatioMatchesHints) {
  // 300/60 = 5 cycles for A, 100/20 = 5 cycles for B.
  // Every output batch should contain exactly 60 A-rows and 20 B-rows.
  constexpr int kRowsA = 300;
  constexpr int kRowsB = 100;
  constexpr int32_t kHintA = 60;
  constexpr int32_t kHintB = 20;

  auto dataA = makeRowVector({
      makeFlatVector<int64_t>(kRowsA, [](auto row) { return row; }),
  });
  auto dataB = makeRowVector({
      makeFlatVector<int64_t>(kRowsB, [](auto row) { return row + 10'000; }),
  });

  auto filePathA = TempFilePath::create();
  auto filePathB = TempFilePath::create();
  writeToFile(filePathA->getPath(), dataA);
  writeToFile(filePathB->getPath(), dataB);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto rowType = ROW({"c0"}, {BIGINT()});

  core::PlanNodeId scanAId;
  auto sourceA = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(scanAId)
                     .planNode();
  core::PlanNodeId scanBId;
  auto sourceB = PlanBuilder(planNodeIdGenerator)
                     .tableScan(rowType)
                     .capturePlanNodeId(scanBId)
                     .planNode();

  auto unionNode = makeMixedUnionNode({sourceA, sourceB}, planNodeIdGenerator);

  auto splitA = HiveConnectorSplitBuilder(filePathA->getPath())
                    .batchSizeHint(kHintA)
                    .build();
  auto splitB = HiveConnectorSplitBuilder(filePathB->getPath())
                    .batchSizeHint(kHintB)
                    .build();

  CursorParameters params;
  params.planNode = unionNode;
  params.maxDrivers = 1;
  params.serialExecution = true;

  auto [cursor, results] = readCursor(params, [&](TaskCursor* taskCursor) {
    if (taskCursor->noMoreSplits()) {
      return;
    }
    auto task = taskCursor->task();
    task->addSplit(scanAId, Split(splitA));
    task->addSplit(scanBId, Split(splitB));
    task->noMoreSplits(scanAId);
    task->noMoreSplits(scanBId);
    taskCursor->setNoMoreSplits();
  });

  // Both sources exhaust in 5 cycles, so we expect exactly 5 mixed batches.
  constexpr int kExpectedCycles = kRowsA / kHintA;
  static_assert(kRowsA / kHintA == kRowsB / kHintB);

  int totalRows = 0;
  int mixedBatchCount = 0;
  for (const auto& batch : results) {
    auto* col = batch->childAt(0)->asFlatVector<int64_t>();
    int countA = 0, countB = 0;
    for (int i = 0; i < batch->size(); ++i) {
      if (col->valueAt(i) < 10'000) {
        ++countA;
      } else {
        ++countB;
      }
    }
    totalRows += batch->size();

    if (countA > 0 && countB > 0) {
      ++mixedBatchCount;
      EXPECT_EQ(countA, kHintA) << "Mixed batch should contain exactly "
                                << kHintA << " rows from source A";
      EXPECT_EQ(countB, kHintB) << "Mixed batch should contain exactly "
                                << kHintB << " rows from source B";
      EXPECT_EQ(batch->size(), kHintA + kHintB)
          << "Mixed batch total should be " << kHintA + kHintB;
    }
  }

  EXPECT_EQ(totalRows, kRowsA + kRowsB);
  EXPECT_EQ(mixedBatchCount, kExpectedCycles)
      << "All " << kExpectedCycles
      << " output batches should be mixed (both sources exhaust together)";
}
