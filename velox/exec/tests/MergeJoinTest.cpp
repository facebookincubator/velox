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

#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class MergeJoinTest : public OperatorTestBase {
 protected:
  static CursorParameters makeCursorParameters(
      const std::shared_ptr<const core::PlanNode>& planNode,
      uint32_t preferredOutputBatchSize) {
    auto queryCtx = core::QueryCtx::createForTest();
    queryCtx->setConfigOverridesUnsafe(
        {{core::QueryConfig::kCreateEmptyFiles, "true"}});

    CursorParameters params;
    params.planNode = planNode;
    params.queryCtx = core::QueryCtx::createForTest();
    params.queryCtx->setConfigOverridesUnsafe(
        {{core::QueryConfig::kPreferredOutputBatchSize,
          std::to_string(preferredOutputBatchSize)}});
    return params;
  }

  template <typename T>
  void testJoin(
      std::function<T(vector_size_t /*row*/)> leftKeyAt,
      std::function<T(vector_size_t /*row*/)> rightKeyAt) {
    // Single batch on the left and right sides of the join.
    {
      auto leftKeys = makeFlatVector<T>(1'234, leftKeyAt);
      auto rightKeys = makeFlatVector<T>(1'234, rightKeyAt);

      testJoin({leftKeys}, {rightKeys});
    }

    // Multiple batches on one side. Single batch on the other side.
    {
      std::vector<VectorPtr> leftKeys = {
          makeFlatVector<T>(1024, leftKeyAt),
          makeFlatVector<T>(
              1024, [&](auto row) { return leftKeyAt(1024 + row); }),
      };
      std::vector<VectorPtr> rightKeys = {makeFlatVector<T>(2048, rightKeyAt)};

      testJoin(leftKeys, rightKeys);

      // Swap left and right side keys.
      testJoin(rightKeys, leftKeys);
    }

    // Multiple batches on each side.
    {
      std::vector<VectorPtr> leftKeys = {
          makeFlatVector<T>(512, leftKeyAt),
          makeFlatVector<T>(
              1024, [&](auto row) { return leftKeyAt(512 + row); }),
          makeFlatVector<T>(
              16, [&](auto row) { return leftKeyAt(512 + 1024 + row); }),
      };
      std::vector<VectorPtr> rightKeys = {
          makeFlatVector<T>(123, rightKeyAt),
          makeFlatVector<T>(
              1024, [&](auto row) { return rightKeyAt(123 + row); }),
          makeFlatVector<T>(
              1234, [&](auto row) { return rightKeyAt(123 + 1024 + row); }),
      };

      testJoin(leftKeys, rightKeys);

      // Swap left and right side keys.
      testJoin(rightKeys, leftKeys);
    }
  }

  void testJoin(
      const std::vector<VectorPtr>& leftKeys,
      const std::vector<VectorPtr>& rightKeys) {
    std::vector<RowVectorPtr> left;
    left.reserve(leftKeys.size());
    vector_size_t startRow = 0;
    for (const auto& key : leftKeys) {
      auto payload = makeFlatVector<int32_t>(
          key->size(), [startRow](auto row) { return (startRow + row) * 10; });
      left.push_back(makeRowVector({key, payload}));
      startRow += key->size();
    }

    std::vector<RowVectorPtr> right;
    right.reserve(rightKeys.size());
    startRow = 0;
    for (const auto& key : rightKeys) {
      auto payload = makeFlatVector<int32_t>(
          key->size(), [startRow](auto row) { return (startRow + row) * 20; });
      right.push_back(makeRowVector({key, payload}));
      startRow += key->size();
    }

    createDuckDbTable("t", left);
    createDuckDbTable("u", right);

    // Test INNER join.
    auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
    auto plan = PlanBuilder(planNodeIdGenerator)
                    .values(left)
                    .mergeJoin(
                        {"c0"},
                        {"u_c0"},
                        PlanBuilder(planNodeIdGenerator)
                            .values(right)
                            .project({"c0 AS u_c0", "c1 AS u_c1"})
                            .planNode(),
                        "",
                        {"c0", "c1", "u_c1"},
                        core::JoinType::kInner)
                    .planNode();

    // Use very small output batch size.
    assertQuery(
        makeCursorParameters(plan, 16),
        "SELECT t.c0, t.c1, u.c1 FROM t, u WHERE t.c0 = u.c0");

    // Use regular output batch size.
    assertQuery(
        makeCursorParameters(plan, 1024),
        "SELECT t.c0, t.c1, u.c1 FROM t, u WHERE t.c0 = u.c0");

    // Use very large output batch size.
    assertQuery(
        makeCursorParameters(plan, 10'000),
        "SELECT t.c0, t.c1, u.c1 FROM t, u WHERE t.c0 = u.c0");

    // Test LEFT join.
    planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
    plan = PlanBuilder(planNodeIdGenerator)
               .values(left)
               .mergeJoin(
                   {"c0"},
                   {"u_c0"},
                   PlanBuilder(planNodeIdGenerator)
                       .values(right)
                       .project({"c0 as u_c0", "c1 as u_c1"})
                       .planNode(),
                   "",
                   {"c0", "c1", "u_c1"},
                   core::JoinType::kLeft)
               .planNode();

    // Use very small output batch size.
    assertQuery(
        makeCursorParameters(plan, 16),
        "SELECT t.c0, t.c1, u.c1 FROM t LEFT JOIN u ON t.c0 = u.c0");

    // Use regular output batch size.
    assertQuery(
        makeCursorParameters(plan, 1024),
        "SELECT t.c0, t.c1, u.c1 FROM t LEFT JOIN u ON t.c0 = u.c0");

    // Use very large output batch size.
    assertQuery(
        makeCursorParameters(plan, 10'000),
        "SELECT t.c0, t.c1, u.c1 FROM t LEFT JOIN u ON t.c0 = u.c0");
  }
};

TEST_F(MergeJoinTest, oneToOneAllMatch) {
  testJoin<int32_t>([](auto row) { return row; }, [](auto row) { return row; });
}

TEST_F(MergeJoinTest, someDontMatch) {
  testJoin<int32_t>(
      [](auto row) { return row % 5 == 0 ? row - 1 : row; },
      [](auto row) { return row % 7 == 0 ? row - 1 : row; });
}

TEST_F(MergeJoinTest, fewMatch) {
  testJoin<int32_t>(
      [](auto row) { return row * 5; }, [](auto row) { return row * 7; });
}

TEST_F(MergeJoinTest, duplicateMatch) {
  testJoin<int32_t>(
      [](auto row) { return row / 2; }, [](auto row) { return row / 3; });
}

TEST_F(MergeJoinTest, allRowsMatch) {
  std::vector<VectorPtr> leftKeys = {
      makeFlatVector<int32_t>(2, [](auto /* row */) { return 5; }),
      makeFlatVector<int32_t>(3, [](auto /* row */) { return 5; }),
      makeFlatVector<int32_t>(4, [](auto /* row */) { return 5; }),
  };
  std::vector<VectorPtr> rightKeys = {
      makeFlatVector<int32_t>(7, [](auto /* row */) { return 5; })};

  testJoin(leftKeys, rightKeys);

  testJoin(rightKeys, leftKeys);
}

TEST_F(MergeJoinTest, aggregationOverJoin) {
  auto left =
      makeRowVector({"t_c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto right = makeRowVector({"u_c0"}, {makeFlatVector<int32_t>({2, 4, 6})});

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values({left})
          .mergeJoin(
              {"t_c0"},
              {"u_c0"},
              PlanBuilder(planNodeIdGenerator).values({right}).planNode(),
              "",
              {"t_c0", "u_c0"},
              core::JoinType::kInner)
          .singleAggregation({}, {"count(1)"})
          .planNode();

  auto result = readSingleValue(plan);
  ASSERT_FALSE(result.isNull());
  ASSERT_EQ(2, result.value<int64_t>());
}

TEST_F(MergeJoinTest, nonFirstJoinKeys) {
  auto left = makeRowVector(
      {"t_data", "t_key"},
      {
          makeFlatVector<int32_t>({50, 40, 30, 20, 10}),
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
      });
  auto right = makeRowVector(
      {"u_data", "u_key"},
      {
          makeFlatVector<int32_t>({23, 22, 21}),
          makeFlatVector<int32_t>({2, 4, 6}),
      });

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values({left})
          .mergeJoin(
              {"t_key"},
              {"u_key"},
              PlanBuilder(planNodeIdGenerator).values({right}).planNode(),
              "",
              {"t_key", "t_data", "u_data"},
              core::JoinType::kInner)
          .planNode();

  assertQuery(plan, "VALUES (2, 40, 23), (4, 20, 22)");
}
