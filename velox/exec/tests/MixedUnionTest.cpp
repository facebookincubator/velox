/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

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
