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
