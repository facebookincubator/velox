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

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempFilePath.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class MixedUnionTest : public OperatorTestBase {
 protected:
  std::shared_ptr<core::MixedUnionNode> makeUnionNode(
      std::vector<core::PlanNodePtr> sources,
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
    return std::make_shared<core::MixedUnionNode>(
        planNodeIdGenerator->next(), std::move(sources));
  }
};

TEST_F(MixedUnionTest, basicUnion) {
  auto data1 = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<std::string>({"a", "b", "c"}),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int32_t>({4, 5, 6}),
      makeFlatVector<std::string>({"d", "e", "f"}),
  });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto unionNode = makeUnionNode(
      {PlanBuilder(planNodeIdGenerator).values({data1}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({data2}).planNode()},
      planNodeIdGenerator);

  assertQuery(
      CursorParameters{
          .planNode = unionNode,
          .maxDrivers = 1,
      },
      "VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e'), (6, 'f')");
}

TEST_F(MixedUnionTest, singleSource) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<std::string>({"a", "b", "c"}),
  });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto unionNode = makeUnionNode(
      {PlanBuilder(planNodeIdGenerator).values({data}).planNode()},
      planNodeIdGenerator);

  assertQuery(
      CursorParameters{
          .planNode = unionNode,
          .maxDrivers = 1,
      },
      "VALUES (1, 'a'), (2, 'b'), (3, 'c')");
}

TEST_F(MixedUnionTest, emptyFirstSource) {
  auto emptyData = makeRowVector(
      {makeFlatVector<int32_t>({}), makeFlatVector<std::string>({})});
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<std::string>({"a", "b"}),
  });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto unionNode = makeUnionNode(
      {PlanBuilder(planNodeIdGenerator).values({emptyData}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({data}).planNode()},
      planNodeIdGenerator);

  assertQuery(
      CursorParameters{
          .planNode = unionNode,
          .maxDrivers = 1,
      },
      "VALUES (1, 'a'), (2, 'b')");
}

TEST_F(MixedUnionTest, emptyLastSource) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<std::string>({"a", "b"}),
  });
  auto emptyData = makeRowVector(
      {makeFlatVector<int32_t>({}), makeFlatVector<std::string>({})});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto unionNode = makeUnionNode(
      {PlanBuilder(planNodeIdGenerator).values({data}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({emptyData}).planNode()},
      planNodeIdGenerator);

  assertQuery(
      CursorParameters{
          .planNode = unionNode,
          .maxDrivers = 1,
      },
      "VALUES (1, 'a'), (2, 'b')");
}

TEST_F(MixedUnionTest, allEmptySources) {
  auto emptyData = makeRowVector(
      {makeFlatVector<int32_t>({}), makeFlatVector<std::string>({})});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto unionNode = makeUnionNode(
      {PlanBuilder(planNodeIdGenerator).values({emptyData}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({emptyData}).planNode()},
      planNodeIdGenerator);

  auto result = AssertQueryBuilder(unionNode).copyResults(pool());
  ASSERT_EQ(result->size(), 0);
}

TEST_F(MixedUnionTest, multipleInputs) {
  auto data1 = makeRowVector({
      makeFlatVector<int32_t>({1}),
      makeFlatVector<std::string>({"a"}),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int32_t>({2}),
      makeFlatVector<std::string>({"b"}),
  });
  auto data3 = makeRowVector({
      makeFlatVector<int32_t>({3}),
      makeFlatVector<std::string>({"c"}),
  });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto unionNode = makeUnionNode(
      {PlanBuilder(planNodeIdGenerator).values({data1}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({data2}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({data3}).planNode()},
      planNodeIdGenerator);

  assertQuery(
      CursorParameters{
          .planNode = unionNode,
          .maxDrivers = 1,
      },
      "VALUES (1, 'a'), (2, 'b'), (3, 'c')");
}

TEST_F(MixedUnionTest, manySources) {
  constexpr int kNumSources = 10;
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  std::vector<core::PlanNodePtr> sources;
  sources.reserve(kNumSources);
  for (int i = 0; i < kNumSources; ++i) {
    auto data = makeRowVector({
        makeFlatVector<int64_t>({i}),
    });
    sources.push_back(
        PlanBuilder(planNodeIdGenerator).values({data}).planNode());
  }

  auto unionNode = makeUnionNode(std::move(sources), planNodeIdGenerator);
  auto result = AssertQueryBuilder(unionNode).copyResults(pool());
  ASSERT_EQ(result->size(), kNumSources);
}

TEST_F(MixedUnionTest, unequalSourceSizes) {
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({6, 7}),
  });
  auto data3 = makeRowVector({
      makeFlatVector<int64_t>({8, 9, 10, 11, 12, 13, 14}),
  });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto unionNode = makeUnionNode(
      {PlanBuilder(planNodeIdGenerator).values({data1}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({data2}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({data3}).planNode()},
      planNodeIdGenerator);

  auto result = AssertQueryBuilder(unionNode).copyResults(pool());
  ASSERT_EQ(result->size(), 14);
}

TEST_F(MixedUnionTest, multipleBatchesPerSource) {
  // Test that union correctly handles the case where sources provide data
  // in a single batch each
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6}),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({7, 8, 9, 10, 11, 12}),
  });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto unionNode = makeUnionNode(
      {PlanBuilder(planNodeIdGenerator).values({data1}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({data2}).planNode()},
      planNodeIdGenerator);

  auto result = AssertQueryBuilder(unionNode).copyResults(pool());
  ASSERT_EQ(result->size(), 12);
}

TEST_F(MixedUnionTest, withNulls) {
  auto data1 = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 3}),
      makeNullableFlatVector<std::string>({"a", "b", std::nullopt}),
  });
  auto data2 = makeRowVector({
      makeNullableFlatVector<int64_t>({std::nullopt, 5, 6}),
      makeNullableFlatVector<std::string>({std::nullopt, "e", "f"}),
  });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto unionNode = makeUnionNode(
      {PlanBuilder(planNodeIdGenerator).values({data1}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({data2}).planNode()},
      planNodeIdGenerator);

  auto result = AssertQueryBuilder(unionNode).copyResults(pool());
  ASSERT_EQ(result->size(), 6);

  auto* col0 = result->childAt(0)->asFlatVector<int64_t>();
  EXPECT_TRUE(col0->isNullAt(1));
  EXPECT_TRUE(col0->isNullAt(3));

  auto* col1 = result->childAt(1)->asFlatVector<StringView>();
  EXPECT_TRUE(col1->isNullAt(2));
  EXPECT_TRUE(col1->isNullAt(3));
}

TEST_F(MixedUnionTest, variousTypes) {
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

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto unionNode = makeUnionNode(
      {PlanBuilder(planNodeIdGenerator).values({data1}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({data2}).planNode()},
      planNodeIdGenerator);

  auto result = AssertQueryBuilder(unionNode).copyResults(pool());
  ASSERT_EQ(result->size(), 4);

  auto* int64Col = result->childAt(0)->asFlatVector<int64_t>();
  EXPECT_EQ(int64Col->valueAt(0), 1);
  EXPECT_EQ(int64Col->valueAt(3), 4);

  auto* doubleCol = result->childAt(1)->asFlatVector<double>();
  EXPECT_DOUBLE_EQ(doubleCol->valueAt(0), 1.1);
  EXPECT_DOUBLE_EQ(doubleCol->valueAt(3), 4.4);
}

TEST_F(MixedUnionTest, largeDataVolume) {
  constexpr int kRowsPerSource = 1000;
  constexpr int kNumSources = 3;

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<core::PlanNodePtr> sources;
  sources.reserve(kNumSources);

  for (int s = 0; s < kNumSources; ++s) {
    auto data = makeRowVector({
        makeFlatVector<int64_t>(
            kRowsPerSource, [s](auto row) { return s * 10000 + row; }),
        makeFlatVector<double>(
            kRowsPerSource, [](auto row) { return row * 0.1; }),
    });
    sources.push_back(
        PlanBuilder(planNodeIdGenerator).values({data}).planNode());
  }

  auto unionNode = makeUnionNode(std::move(sources), planNodeIdGenerator);
  auto result = AssertQueryBuilder(unionNode).copyResults(pool());
  ASSERT_EQ(result->size(), kRowsPerSource * kNumSources);
}

TEST_F(MixedUnionTest, stats) {
  constexpr int kRowsPerSource = 100;
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>(kRowsPerSource, [](auto row) { return row; }),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>(
          kRowsPerSource, [](auto row) { return row + 100; }),
  });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId unionNodeId;
  auto unionNode = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{
          PlanBuilder(planNodeIdGenerator).values({data1}).planNode(),
          PlanBuilder(planNodeIdGenerator).values({data2}).planNode()});
  unionNodeId = unionNode->id();

  auto task = AssertQueryBuilder(unionNode).copyResults(pool());
  ASSERT_EQ(task->size(), kRowsPerSource * 2);
}

TEST_F(MixedUnionTest, withFilter) {
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({6, 7, 8, 9, 10}),
  });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto unionNode = makeUnionNode(
      {PlanBuilder(planNodeIdGenerator).values({data1}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({data2}).planNode()},
      planNodeIdGenerator);

  auto plan = PlanBuilder(planNodeIdGenerator)
                  .addNode([&](auto, auto) { return unionNode; })
                  .filter("c0 > 5")
                  .planNode();

  assertQuery(plan, "VALUES (6), (7), (8), (9), (10)");
}

TEST_F(MixedUnionTest, withProject) {
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeFlatVector<int64_t>({10, 20}),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({3, 4}),
      makeFlatVector<int64_t>({30, 40}),
  });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto unionNode = makeUnionNode(
      {PlanBuilder(planNodeIdGenerator).values({data1}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({data2}).planNode()},
      planNodeIdGenerator);

  auto plan = PlanBuilder(planNodeIdGenerator)
                  .addNode([&](auto, auto) { return unionNode; })
                  .project({"c0 + c1 as sum"})
                  .planNode();

  assertQuery(plan, "VALUES (11), (22), (33), (44)");
}

TEST_F(MixedUnionTest, withAggregation) {
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 2}),
      makeFlatVector<int64_t>({40, 50, 60}),
  });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto unionNode = makeUnionNode(
      {PlanBuilder(planNodeIdGenerator).values({data1}).planNode(),
       PlanBuilder(planNodeIdGenerator).values({data2}).planNode()},
      planNodeIdGenerator);

  auto plan = PlanBuilder(planNodeIdGenerator)
                  .addNode([&](auto, auto) { return unionNode; })
                  .singleAggregation({"c0"}, {"sum(c1)"})
                  .planNode();

  assertQuery(plan, "VALUES (1, 70), (2, 140)");
}

/// Test fixture for MixedUnion barrier execution tests using HiveConnector.
class MixedUnionBarrierTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
  }

  std::shared_ptr<core::MixedUnionNode> makeUnionNode(
      std::vector<core::PlanNodePtr> sources,
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
    return std::make_shared<core::MixedUnionNode>(
        planNodeIdGenerator->next(), std::move(sources));
  }

  std::vector<std::shared_ptr<TempFilePath>> writeDataToFiles(
      const std::vector<RowVectorPtr>& data) {
    std::vector<std::shared_ptr<TempFilePath>> files;
    files.reserve(data.size());
    for (const auto& vector : data) {
      auto file = TempFilePath::create();
      writeToFile(file->getPath(), vector);
      files.push_back(file);
    }
    return files;
  }
};

TEST_F(MixedUnionBarrierTest, supportsBarrier) {
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  core::PlanNodeId scanNodeId;
  core::PlanNodePtr unionNode;
  const auto plan = PlanBuilder(planNodeIdGenerator)
                        .tableScan(rowType)
                        .capturePlanNodeId(scanNodeId)
                        .addNode([&](const auto& id, auto source) {
                          return std::make_shared<core::MixedUnionNode>(
                              id,
                              std::vector<core::PlanNodePtr>{
                                  source,
                                  PlanBuilder(planNodeIdGenerator)
                                      .tableScan(rowType)
                                      .planNode()});
                        })
                        .capturePlanNode(unionNode)
                        .planNode();

  auto mixedUnionNode =
      std::dynamic_pointer_cast<const core::MixedUnionNode>(unionNode);
  ASSERT_TRUE(mixedUnionNode != nullptr);
  ASSERT_TRUE(mixedUnionNode->supportsBarrier());
}

TEST_F(MixedUnionBarrierTest, barrierExecution) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  // Create test data for two sources
  auto leftData = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row * 10; }),
  });
  auto rightData = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row + 100; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row * 10 + 1000; }),
  });

  // Write data to files
  auto leftFile = TempFilePath::create();
  writeToFile(leftFile->getPath(), leftData);
  auto rightFile = TempFilePath::create();
  writeToFile(rightFile->getPath(), rightData);

  // Build plan with MixedUnion
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  for (const auto hasBarrier : {false, true}) {
    SCOPED_TRACE(fmt::format("hasBarrier {}", hasBarrier));
    AssertQueryBuilder queryBuilder(plan);
    queryBuilder.barrierExecution(hasBarrier).serialExecution(true);
    queryBuilder.split(
        leftScanNodeId, makeHiveConnectorSplit(leftFile->getPath()));
    queryBuilder.split(
        rightScanNodeId, makeHiveConnectorSplit(rightFile->getPath()));

    std::shared_ptr<Task> task;
    auto result = queryBuilder.copyResults(pool(), task);
    ASSERT_EQ(result->size(), 200);

    const auto taskStats = task->taskStats();
    ASSERT_EQ(taskStats.numBarriers, hasBarrier ? 1 : 0);
  }
}

TEST_F(MixedUnionBarrierTest, barrierWithMultipleSplits) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  // Create test data for multiple splits on each side
  std::vector<std::shared_ptr<TempFilePath>> leftFiles;
  std::vector<std::shared_ptr<TempFilePath>> rightFiles;
  constexpr int kSplitsPerSide = 3;
  constexpr int kRowsPerSplit = 50;

  for (int i = 0; i < kSplitsPerSide; ++i) {
    auto leftData = makeRowVector({
        makeFlatVector<int32_t>(
            kRowsPerSplit, [i](auto row) { return i * 1000 + row; }),
        makeFlatVector<int64_t>(
            kRowsPerSplit, [i](auto row) { return i * 10000 + row * 10; }),
    });
    auto rightData = makeRowVector({
        makeFlatVector<int32_t>(
            kRowsPerSplit, [i](auto row) { return i * 1000 + row + 500; }),
        makeFlatVector<int64_t>(
            kRowsPerSplit,
            [i](auto row) { return i * 10000 + row * 10 + 5000; }),
    });

    auto leftFile = TempFilePath::create();
    writeToFile(leftFile->getPath(), leftData);
    leftFiles.push_back(leftFile);

    auto rightFile = TempFilePath::create();
    writeToFile(rightFile->getPath(), rightData);
    rightFiles.push_back(rightFile);
  }

  // Build plan
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  for (const auto hasBarrier : {false, true}) {
    SCOPED_TRACE(fmt::format("hasBarrier {}", hasBarrier));
    AssertQueryBuilder queryBuilder(plan);
    queryBuilder.barrierExecution(hasBarrier).serialExecution(true);

    for (const auto& file : leftFiles) {
      queryBuilder.split(
          leftScanNodeId, makeHiveConnectorSplit(file->getPath()));
    }
    for (const auto& file : rightFiles) {
      queryBuilder.split(
          rightScanNodeId, makeHiveConnectorSplit(file->getPath()));
    }

    auto result = queryBuilder.copyResults(pool());
    ASSERT_EQ(result->size(), kSplitsPerSide * kRowsPerSplit * 2);
  }
}

TEST_F(MixedUnionBarrierTest, barrierWithProject) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  auto leftData = makeRowVector({
      makeFlatVector<int32_t>(50, [](auto row) { return row; }),
      makeFlatVector<int64_t>(50, [](auto row) { return row * 2; }),
  });
  auto rightData = makeRowVector({
      makeFlatVector<int32_t>(50, [](auto row) { return row + 50; }),
      makeFlatVector<int64_t>(50, [](auto row) { return (row + 50) * 2; }),
  });

  auto leftFile = TempFilePath::create();
  writeToFile(leftFile->getPath(), leftData);
  auto rightFile = TempFilePath::create();
  writeToFile(rightFile->getPath(), rightData);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto unionNode = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  // Add a project on top of the union
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .addNode([&](const auto&, auto) { return unionNode; })
                  .project({"c0", "c1", "c0 + c1 as sum"})
                  .planNode();

  for (const auto hasBarrier : {false, true}) {
    SCOPED_TRACE(fmt::format("hasBarrier {}", hasBarrier));
    AssertQueryBuilder queryBuilder(plan);
    queryBuilder.barrierExecution(hasBarrier).serialExecution(true);
    queryBuilder.split(
        leftScanNodeId, makeHiveConnectorSplit(leftFile->getPath()));
    queryBuilder.split(
        rightScanNodeId, makeHiveConnectorSplit(rightFile->getPath()));

    auto result = queryBuilder.copyResults(pool());
    ASSERT_EQ(result->size(), 100);
    ASSERT_EQ(result->type()->size(), 3);
  }
}

TEST_F(MixedUnionBarrierTest, barrierWithFilter) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  auto leftData = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row * 10; }),
  });
  auto rightData = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row + 100; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row * 10 + 1000; }),
  });

  auto leftFile = TempFilePath::create();
  writeToFile(leftFile->getPath(), leftData);
  auto rightFile = TempFilePath::create();
  writeToFile(rightFile->getPath(), rightData);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto unionNode = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  // Add a filter on top of the union
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .addNode([&](const auto&, auto) { return unionNode; })
                  .filter("c0 >= 50 AND c0 < 150")
                  .planNode();

  for (const auto hasBarrier : {false, true}) {
    SCOPED_TRACE(fmt::format("hasBarrier {}", hasBarrier));
    AssertQueryBuilder queryBuilder(plan);
    queryBuilder.barrierExecution(hasBarrier).serialExecution(true);
    queryBuilder.split(
        leftScanNodeId, makeHiveConnectorSplit(leftFile->getPath()));
    queryBuilder.split(
        rightScanNodeId, makeHiveConnectorSplit(rightFile->getPath()));

    auto result = queryBuilder.copyResults(pool());
    // Left source: rows 50-99 (50 rows)
    // Right source: rows 100-149 (50 rows from 100-199 range, but capped at
    // 149)
    ASSERT_EQ(result->size(), 100);
  }
}

TEST_F(MixedUnionBarrierTest, barrierWithThreeSources) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  auto data1 = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row * 10; }),
  });
  auto data2 = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row + 100; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row * 10 + 1000; }),
  });
  auto data3 = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row + 200; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row * 10 + 2000; }),
  });

  auto file1 = TempFilePath::create();
  writeToFile(file1->getPath(), data1);
  auto file2 = TempFilePath::create();
  writeToFile(file2->getPath(), data2);
  auto file3 = TempFilePath::create();
  writeToFile(file3->getPath(), data3);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId scanNodeId1;
  core::PlanNodeId scanNodeId2;
  core::PlanNodeId scanNodeId3;

  auto scan1 = PlanBuilder(planNodeIdGenerator)
                   .tableScan(rowType)
                   .capturePlanNodeId(scanNodeId1)
                   .planNode();
  auto scan2 = PlanBuilder(planNodeIdGenerator)
                   .tableScan(rowType)
                   .capturePlanNodeId(scanNodeId2)
                   .planNode();
  auto scan3 = PlanBuilder(planNodeIdGenerator)
                   .tableScan(rowType)
                   .capturePlanNodeId(scanNodeId3)
                   .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{scan1, scan2, scan3});

  for (const auto hasBarrier : {false, true}) {
    SCOPED_TRACE(fmt::format("hasBarrier {}", hasBarrier));
    AssertQueryBuilder queryBuilder(plan);
    queryBuilder.barrierExecution(hasBarrier).serialExecution(true);
    queryBuilder.split(scanNodeId1, makeHiveConnectorSplit(file1->getPath()));
    queryBuilder.split(scanNodeId2, makeHiveConnectorSplit(file2->getPath()));
    queryBuilder.split(scanNodeId3, makeHiveConnectorSplit(file3->getPath()));

    auto result = queryBuilder.copyResults(pool());
    ASSERT_EQ(result->size(), 300);
  }
}

TEST_F(MixedUnionBarrierTest, barrierWithEmptySource) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  auto leftData = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row * 10; }),
  });
  // Empty right source
  auto rightData = makeRowVector({
      makeFlatVector<int32_t>({}),
      makeFlatVector<int64_t>({}),
  });

  auto leftFile = TempFilePath::create();
  writeToFile(leftFile->getPath(), leftData);
  auto rightFile = TempFilePath::create();
  writeToFile(rightFile->getPath(), rightData);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  for (const auto hasBarrier : {false, true}) {
    SCOPED_TRACE(fmt::format("hasBarrier {}", hasBarrier));
    AssertQueryBuilder queryBuilder(plan);
    queryBuilder.barrierExecution(hasBarrier).serialExecution(true);
    queryBuilder.split(
        leftScanNodeId, makeHiveConnectorSplit(leftFile->getPath()));
    queryBuilder.split(
        rightScanNodeId, makeHiveConnectorSplit(rightFile->getPath()));

    auto result = queryBuilder.copyResults(pool());
    ASSERT_EQ(result->size(), 100);
  }
}

TEST_F(MixedUnionBarrierTest, barrierWithUnequalSourceSizes) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  // Left source has significantly more data
  auto leftData = makeRowVector({
      makeFlatVector<int32_t>(1000, [](auto row) { return row; }),
      makeFlatVector<int64_t>(1000, [](auto row) { return row * 10; }),
  });
  // Right source has much less data
  auto rightData = makeRowVector({
      makeFlatVector<int32_t>(10, [](auto row) { return row + 1000; }),
      makeFlatVector<int64_t>(10, [](auto row) { return row * 10 + 10000; }),
  });

  auto leftFile = TempFilePath::create();
  writeToFile(leftFile->getPath(), leftData);
  auto rightFile = TempFilePath::create();
  writeToFile(rightFile->getPath(), rightData);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  for (const auto hasBarrier : {false, true}) {
    SCOPED_TRACE(fmt::format("hasBarrier {}", hasBarrier));
    AssertQueryBuilder queryBuilder(plan);
    queryBuilder.barrierExecution(hasBarrier).serialExecution(true);
    queryBuilder.split(
        leftScanNodeId, makeHiveConnectorSplit(leftFile->getPath()));
    queryBuilder.split(
        rightScanNodeId, makeHiveConnectorSplit(rightFile->getPath()));

    auto result = queryBuilder.copyResults(pool());
    ASSERT_EQ(result->size(), 1010);
  }
}

TEST_F(MixedUnionBarrierTest, barrierTaskStats) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  auto leftData = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row * 10; }),
  });
  auto rightData = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row + 100; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row * 10 + 1000; }),
  });

  auto leftFile = TempFilePath::create();
  writeToFile(leftFile->getPath(), leftData);
  auto rightFile = TempFilePath::create();
  writeToFile(rightFile->getPath(), rightData);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  for (const auto hasBarrier : {false, true}) {
    SCOPED_TRACE(fmt::format("hasBarrier {}", hasBarrier));
    AssertQueryBuilder queryBuilder(plan);
    queryBuilder.barrierExecution(hasBarrier).serialExecution(true);
    queryBuilder.split(
        leftScanNodeId, makeHiveConnectorSplit(leftFile->getPath()));
    queryBuilder.split(
        rightScanNodeId, makeHiveConnectorSplit(rightFile->getPath()));

    std::shared_ptr<Task> task;
    auto result = queryBuilder.copyResults(pool(), task);
    ASSERT_EQ(result->size(), 200);

    const auto taskStats = task->taskStats();
    ASSERT_EQ(taskStats.numBarriers, hasBarrier ? 1 : 0);
    ASSERT_EQ(taskStats.numFinishedSplits, 2);
  }
}

// Tests barrier with multiple splits per source. Each split triggers a separate
// barrier cycle, exercising the drain-reset logic in finishDrain(). After each
// cycle, sourcesDrained_ is reset so the next split group can proceed.
TEST_F(MixedUnionBarrierTest, barrierMultipleBarrierCycles) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  constexpr int kSplitsPerSide = 3;
  constexpr int kRowsPerSplit = 40;

  std::vector<std::shared_ptr<TempFilePath>> leftFiles;
  std::vector<std::shared_ptr<TempFilePath>> rightFiles;

  for (int i = 0; i < kSplitsPerSide; ++i) {
    auto leftData = makeRowVector({
        makeFlatVector<int32_t>(
            kRowsPerSplit, [i](auto row) { return i * 1000 + row; }),
        makeFlatVector<int64_t>(
            kRowsPerSplit, [i](auto row) { return i * 10000 + row * 10; }),
    });
    auto rightData = makeRowVector({
        makeFlatVector<int32_t>(
            kRowsPerSplit, [i](auto row) { return i * 1000 + row + 500; }),
        makeFlatVector<int64_t>(
            kRowsPerSplit,
            [i](auto row) { return i * 10000 + row * 10 + 5000; }),
    });

    auto leftFile = TempFilePath::create();
    writeToFile(leftFile->getPath(), leftData);
    leftFiles.push_back(leftFile);

    auto rightFile = TempFilePath::create();
    writeToFile(rightFile->getPath(), rightData);
    rightFiles.push_back(rightFile);
  }

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  AssertQueryBuilder queryBuilder(plan);
  queryBuilder.barrierExecution(true).serialExecution(true);

  for (const auto& file : leftFiles) {
    queryBuilder.split(leftScanNodeId, makeHiveConnectorSplit(file->getPath()));
  }
  for (const auto& file : rightFiles) {
    queryBuilder.split(
        rightScanNodeId, makeHiveConnectorSplit(file->getPath()));
  }

  std::shared_ptr<Task> task;
  auto result = queryBuilder.copyResults(pool(), task);
  ASSERT_EQ(result->size(), kSplitsPerSide * kRowsPerSplit * 2);

  const auto taskStats = task->taskStats();
  ASSERT_EQ(taskStats.numBarriers, kSplitsPerSide);
}

// Tests barrier where one source has very little data and likely finishes
// before the drain signal arrives. This exercises the code path in startDrain()
// and maybeFinishDrain() that sets sourcesDrained_[i] = true when
// sourcesFinished_[i] is already true.
TEST_F(MixedUnionBarrierTest, barrierWithSourceFinishedBeforeDrain) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  // Left source has a lot of data.
  auto leftData = makeRowVector({
      makeFlatVector<int32_t>(1000, [](auto row) { return row; }),
      makeFlatVector<int64_t>(1000, [](auto row) { return row * 10; }),
  });
  // Right source has very little data — likely finishes before drain starts.
  auto rightData = makeRowVector({
      makeFlatVector<int32_t>(1, [](auto /*row*/) { return 9999; }),
      makeFlatVector<int64_t>(1, [](auto /*row*/) { return 99990; }),
  });

  auto leftFile = TempFilePath::create();
  writeToFile(leftFile->getPath(), leftData);
  auto rightFile = TempFilePath::create();
  writeToFile(rightFile->getPath(), rightData);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  for (const auto hasBarrier : {false, true}) {
    SCOPED_TRACE(fmt::format("hasBarrier {}", hasBarrier));
    AssertQueryBuilder queryBuilder(plan);
    queryBuilder.barrierExecution(hasBarrier).serialExecution(true);
    queryBuilder.split(
        leftScanNodeId, makeHiveConnectorSplit(leftFile->getPath()));
    queryBuilder.split(
        rightScanNodeId, makeHiveConnectorSplit(rightFile->getPath()));

    auto result = queryBuilder.copyResults(pool());
    ASSERT_EQ(result->size(), 1001);
  }
}

// Tests barrier where all sources are empty. The drain cycle should complete
// immediately since all sources finish without producing data.
TEST_F(MixedUnionBarrierTest, barrierWithAllEmptySources) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  auto emptyData = makeRowVector({
      makeFlatVector<int32_t>({}),
      makeFlatVector<int64_t>({}),
  });

  auto leftFile = TempFilePath::create();
  writeToFile(leftFile->getPath(), emptyData);
  auto rightFile = TempFilePath::create();
  writeToFile(rightFile->getPath(), emptyData);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  for (const auto hasBarrier : {false, true}) {
    SCOPED_TRACE(fmt::format("hasBarrier {}", hasBarrier));
    AssertQueryBuilder queryBuilder(plan);
    queryBuilder.barrierExecution(hasBarrier).serialExecution(true);
    queryBuilder.split(
        leftScanNodeId, makeHiveConnectorSplit(leftFile->getPath()));
    queryBuilder.split(
        rightScanNodeId, makeHiveConnectorSplit(rightFile->getPath()));

    auto result = queryBuilder.copyResults(pool());
    ASSERT_EQ(result->size(), 0);
  }
}

// Tests barrier with three sources where one source is empty. Validates that
// the drain cycle correctly handles a mix of finished and active sources.
TEST_F(MixedUnionBarrierTest, barrierWithThreeSourcesOneEmpty) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  auto data1 = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row * 10; }),
  });
  auto emptyData = makeRowVector({
      makeFlatVector<int32_t>({}),
      makeFlatVector<int64_t>({}),
  });
  auto data3 = makeRowVector({
      makeFlatVector<int32_t>(50, [](auto row) { return row + 200; }),
      makeFlatVector<int64_t>(50, [](auto row) { return row * 10 + 2000; }),
  });

  auto file1 = TempFilePath::create();
  writeToFile(file1->getPath(), data1);
  auto file2 = TempFilePath::create();
  writeToFile(file2->getPath(), emptyData);
  auto file3 = TempFilePath::create();
  writeToFile(file3->getPath(), data3);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId scanNodeId1;
  core::PlanNodeId scanNodeId2;
  core::PlanNodeId scanNodeId3;

  auto scan1 = PlanBuilder(planNodeIdGenerator)
                   .tableScan(rowType)
                   .capturePlanNodeId(scanNodeId1)
                   .planNode();
  auto scan2 = PlanBuilder(planNodeIdGenerator)
                   .tableScan(rowType)
                   .capturePlanNodeId(scanNodeId2)
                   .planNode();
  auto scan3 = PlanBuilder(planNodeIdGenerator)
                   .tableScan(rowType)
                   .capturePlanNodeId(scanNodeId3)
                   .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{scan1, scan2, scan3});

  for (const auto hasBarrier : {false, true}) {
    SCOPED_TRACE(fmt::format("hasBarrier {}", hasBarrier));
    AssertQueryBuilder queryBuilder(plan);
    queryBuilder.barrierExecution(hasBarrier).serialExecution(true);
    queryBuilder.split(scanNodeId1, makeHiveConnectorSplit(file1->getPath()));
    queryBuilder.split(scanNodeId2, makeHiveConnectorSplit(file2->getPath()));
    queryBuilder.split(scanNodeId3, makeHiveConnectorSplit(file3->getPath()));

    auto result = queryBuilder.copyResults(pool());
    ASSERT_EQ(result->size(), 150);
  }
}

// Tests drain where one source finishes producing data very quickly (and is
// marked finished) while the other source is still producing large amounts of
// data. The finished source should be marked as drained immediately in
// startDrain() and maybeFinishDrain().
TEST_F(MixedUnionBarrierTest, drainOneSideFinishedOtherProducing) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  // Left source produces a lot of data in small batches.
  constexpr int kLeftBatches = 10;
  constexpr int kRowsPerBatch = 50;
  auto leftFile = TempFilePath::create();
  {
    auto leftData = makeRowVector({
        makeFlatVector<int32_t>(
            kLeftBatches * kRowsPerBatch, [](auto row) { return row; }),
        makeFlatVector<int64_t>(
            kLeftBatches * kRowsPerBatch, [](auto row) { return row * 10; }),
    });
    writeToFile(leftFile->getPath(), leftData);
  }

  // Right source is empty — finishes immediately.
  auto rightFile = TempFilePath::create();
  {
    auto rightData = makeRowVector({
        makeFlatVector<int32_t>({}),
        makeFlatVector<int64_t>({}),
    });
    writeToFile(rightFile->getPath(), rightData);
  }

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  AssertQueryBuilder queryBuilder(plan);
  queryBuilder.barrierExecution(true).serialExecution(true);
  queryBuilder.split(
      leftScanNodeId, makeHiveConnectorSplit(leftFile->getPath()));
  queryBuilder.split(
      rightScanNodeId, makeHiveConnectorSplit(rightFile->getPath()));

  std::shared_ptr<Task> task;
  auto result = queryBuilder.copyResults(pool(), task);
  ASSERT_EQ(result->size(), kLeftBatches * kRowsPerBatch);

  const auto taskStats = task->taskStats();
  ASSERT_EQ(taskStats.numBarriers, 1);
}

// Tests drain where both sources have produced some data but one is drained
// (signals no more data) while the other still has pending data buffered.
// This exercises the code path where pendingData_ is drained while one source
// has already signaled drained.
TEST_F(MixedUnionBarrierTest, drainWithPendingDataOnOneSide) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  // Both sources have data, but different amounts to create asymmetric
  // completion.
  auto leftData = makeRowVector({
      makeFlatVector<int32_t>(200, [](auto row) { return row; }),
      makeFlatVector<int64_t>(200, [](auto row) { return row * 10; }),
  });
  auto rightData = makeRowVector({
      makeFlatVector<int32_t>(50, [](auto row) { return row + 1000; }),
      makeFlatVector<int64_t>(50, [](auto row) { return row * 10 + 10000; }),
  });

  auto leftFile = TempFilePath::create();
  writeToFile(leftFile->getPath(), leftData);
  auto rightFile = TempFilePath::create();
  writeToFile(rightFile->getPath(), rightData);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  AssertQueryBuilder queryBuilder(plan);
  queryBuilder.barrierExecution(true).serialExecution(true);
  queryBuilder.split(
      leftScanNodeId, makeHiveConnectorSplit(leftFile->getPath()));
  queryBuilder.split(
      rightScanNodeId, makeHiveConnectorSplit(rightFile->getPath()));

  std::shared_ptr<Task> task;
  auto result = queryBuilder.copyResults(pool(), task);
  ASSERT_EQ(result->size(), 250);

  const auto taskStats = task->taskStats();
  ASSERT_EQ(taskStats.numBarriers, 1);
}

// Tests multiple barrier cycles where sources complete in different orders
// each cycle. This exercises the reset logic in finishDrain() ensuring
// sourcesDrained_ is properly cleared between cycles.
TEST_F(MixedUnionBarrierTest, multipleBarrierCyclesAlternatingCompletion) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  constexpr int kSplitsPerSide = 4;

  std::vector<std::shared_ptr<TempFilePath>> leftFiles;
  std::vector<std::shared_ptr<TempFilePath>> rightFiles;

  // Create splits with alternating sizes so completion order varies.
  for (int i = 0; i < kSplitsPerSide; ++i) {
    // Alternate which side has more data.
    int leftRows = (i % 2 == 0) ? 100 : 10;
    int rightRows = (i % 2 == 0) ? 10 : 100;

    auto leftData = makeRowVector({
        makeFlatVector<int32_t>(
            leftRows, [i](auto row) { return i * 1000 + row; }),
        makeFlatVector<int64_t>(
            leftRows, [i](auto row) { return i * 10000 + row * 10; }),
    });
    auto rightData = makeRowVector({
        makeFlatVector<int32_t>(
            rightRows, [i](auto row) { return i * 1000 + row + 500; }),
        makeFlatVector<int64_t>(
            rightRows, [i](auto row) { return i * 10000 + row * 10 + 5000; }),
    });

    auto leftFile = TempFilePath::create();
    writeToFile(leftFile->getPath(), leftData);
    leftFiles.push_back(leftFile);

    auto rightFile = TempFilePath::create();
    writeToFile(rightFile->getPath(), rightData);
    rightFiles.push_back(rightFile);
  }

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  AssertQueryBuilder queryBuilder(plan);
  queryBuilder.barrierExecution(true).serialExecution(true);

  for (const auto& file : leftFiles) {
    queryBuilder.split(leftScanNodeId, makeHiveConnectorSplit(file->getPath()));
  }
  for (const auto& file : rightFiles) {
    queryBuilder.split(
        rightScanNodeId, makeHiveConnectorSplit(file->getPath()));
  }

  std::shared_ptr<Task> task;
  auto result = queryBuilder.copyResults(pool(), task);
  // Each cycle: 2 splits alternate 100+10 and 10+100 = 110 rows per cycle.
  // Total: 4 cycles * 110 = 440 rows.
  ASSERT_EQ(result->size(), 440);

  const auto taskStats = task->taskStats();
  ASSERT_EQ(taskStats.numBarriers, kSplitsPerSide);
}

// Tests drain with three sources where they complete in different orders:
// first source finishes, second is drained, third still has pending data.
TEST_F(MixedUnionBarrierTest, drainThreeSourcesMixedStates) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  // Source 1: Empty (finishes immediately).
  auto data1 = makeRowVector({
      makeFlatVector<int32_t>({}),
      makeFlatVector<int64_t>({}),
  });
  // Source 2: Small amount (drains quickly).
  auto data2 = makeRowVector({
      makeFlatVector<int32_t>(20, [](auto row) { return row + 100; }),
      makeFlatVector<int64_t>(20, [](auto row) { return row * 10 + 1000; }),
  });
  // Source 3: Large amount (still producing when others complete).
  auto data3 = makeRowVector({
      makeFlatVector<int32_t>(500, [](auto row) { return row + 200; }),
      makeFlatVector<int64_t>(500, [](auto row) { return row * 10 + 2000; }),
  });

  auto file1 = TempFilePath::create();
  writeToFile(file1->getPath(), data1);
  auto file2 = TempFilePath::create();
  writeToFile(file2->getPath(), data2);
  auto file3 = TempFilePath::create();
  writeToFile(file3->getPath(), data3);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId scanNodeId1;
  core::PlanNodeId scanNodeId2;
  core::PlanNodeId scanNodeId3;

  auto scan1 = PlanBuilder(planNodeIdGenerator)
                   .tableScan(rowType)
                   .capturePlanNodeId(scanNodeId1)
                   .planNode();
  auto scan2 = PlanBuilder(planNodeIdGenerator)
                   .tableScan(rowType)
                   .capturePlanNodeId(scanNodeId2)
                   .planNode();
  auto scan3 = PlanBuilder(planNodeIdGenerator)
                   .tableScan(rowType)
                   .capturePlanNodeId(scanNodeId3)
                   .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{scan1, scan2, scan3});

  AssertQueryBuilder queryBuilder(plan);
  queryBuilder.barrierExecution(true).serialExecution(true);
  queryBuilder.split(scanNodeId1, makeHiveConnectorSplit(file1->getPath()));
  queryBuilder.split(scanNodeId2, makeHiveConnectorSplit(file2->getPath()));
  queryBuilder.split(scanNodeId3, makeHiveConnectorSplit(file3->getPath()));

  std::shared_ptr<Task> task;
  auto result = queryBuilder.copyResults(pool(), task);
  ASSERT_EQ(result->size(), 520);

  const auto taskStats = task->taskStats();
  ASSERT_EQ(taskStats.numBarriers, 1);
}

// Tests that the operator correctly handles rapid barrier cycles with minimal
// data. This ensures that state reset between cycles is complete and doesn't
// leave stale flags.
TEST_F(MixedUnionBarrierTest, rapidBarrierCyclesMinimalData) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  constexpr int kSplitsPerSide = 5;
  constexpr int kRowsPerSplit = 5;

  std::vector<std::shared_ptr<TempFilePath>> leftFiles;
  std::vector<std::shared_ptr<TempFilePath>> rightFiles;

  for (int i = 0; i < kSplitsPerSide; ++i) {
    auto leftData = makeRowVector({
        makeFlatVector<int32_t>(
            kRowsPerSplit, [i](auto row) { return i * 100 + row; }),
        makeFlatVector<int64_t>(
            kRowsPerSplit, [i](auto row) { return i * 1000 + row * 10; }),
    });
    auto rightData = makeRowVector({
        makeFlatVector<int32_t>(
            kRowsPerSplit, [i](auto row) { return i * 100 + row + 50; }),
        makeFlatVector<int64_t>(
            kRowsPerSplit, [i](auto row) { return i * 1000 + row * 10 + 500; }),
    });

    auto leftFile = TempFilePath::create();
    writeToFile(leftFile->getPath(), leftData);
    leftFiles.push_back(leftFile);

    auto rightFile = TempFilePath::create();
    writeToFile(rightFile->getPath(), rightData);
    rightFiles.push_back(rightFile);
  }

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto plan = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  AssertQueryBuilder queryBuilder(plan);
  queryBuilder.barrierExecution(true).serialExecution(true);

  for (const auto& file : leftFiles) {
    queryBuilder.split(leftScanNodeId, makeHiveConnectorSplit(file->getPath()));
  }
  for (const auto& file : rightFiles) {
    queryBuilder.split(
        rightScanNodeId, makeHiveConnectorSplit(file->getPath()));
  }

  std::shared_ptr<Task> task;
  auto result = queryBuilder.copyResults(pool(), task);
  ASSERT_EQ(result->size(), kSplitsPerSide * kRowsPerSplit * 2);

  const auto taskStats = task->taskStats();
  ASSERT_EQ(taskStats.numBarriers, kSplitsPerSide);
}

TEST_F(MixedUnionBarrierTest, barrierWithAggregation) {
  const auto rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});

  auto leftData = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row % 10; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row; }),
  });
  auto rightData = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row % 10; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row + 100; }),
  });

  auto leftFile = TempFilePath::create();
  writeToFile(leftFile->getPath(), leftData);
  auto rightFile = TempFilePath::create();
  writeToFile(rightFile->getPath(), rightData);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftScanNodeId;
  core::PlanNodeId rightScanNodeId;

  auto leftScan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType)
                      .capturePlanNodeId(leftScanNodeId)
                      .planNode();
  auto rightScan = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rowType)
                       .capturePlanNodeId(rightScanNodeId)
                       .planNode();

  auto unionNode = std::make_shared<core::MixedUnionNode>(
      planNodeIdGenerator->next(),
      std::vector<core::PlanNodePtr>{leftScan, rightScan});

  // Add a single aggregation on top of the union.
  // Note: Only non-barrier mode works with singleAggregation since hash
  // aggregation doesn't support barriers (only streaming aggregation does).
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .addNode([&](const auto&, auto) { return unionNode; })
                  .singleAggregation({"c0"}, {"sum(c1)"})
                  .planNode();

  // Aggregation doesn't support barriers, so only test non-barrier mode.
  AssertQueryBuilder queryBuilder(plan);
  queryBuilder.serialExecution(true);
  queryBuilder.split(
      leftScanNodeId, makeHiveConnectorSplit(leftFile->getPath()));
  queryBuilder.split(
      rightScanNodeId, makeHiveConnectorSplit(rightFile->getPath()));

  auto result = queryBuilder.copyResults(pool());
  // 10 distinct groups
  ASSERT_EQ(result->size(), 10);
}
