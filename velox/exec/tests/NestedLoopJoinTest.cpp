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
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/VectorTestUtil.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class NestedLoopJoinTest : public HiveConnectorTestBase {
 protected:
  RowTypePtr leftType_;
  RowTypePtr rightType_;
  std::vector<std::string> comparisons_;
  std::vector<core::JoinType> joinTypes_;
  std::string joinConditionStr_;
  std::string queryStr_;

  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    leftType_ = ROW({{"t0", BIGINT()}});
    rightType_ = ROW({{"u0", BIGINT()}});
    comparisons_ = {"=", "<", "<=", "<>"};
    joinTypes_ = {
        core::JoinType::kInner,
        core::JoinType::kLeft,
        core::JoinType::kRight,
        core::JoinType::kFull};
    joinConditionStr_ = "t0 {} u0";
    queryStr_ = "SELECT t0, u0 FROM t {} JOIN u ON t.t0 {} u.u0";
  }

  void setLeftType(RowTypePtr leftType) {
    leftType_ = leftType;
  }

  void setRightType(RowTypePtr rightType) {
    rightType_ = rightType;
  }

  void setComparisons(std::vector<std::string>&& comparisons) {
    comparisons_ = std::move(comparisons);
  }

  void setJoinConditionStr(std::string joinConditionStr) {
    joinConditionStr_ = std::move(joinConditionStr);
  }

  void setQueryStr(std::string queryStr) {
    queryStr_ = std::move(queryStr);
  }

  template <typename T>
  VectorPtr sequence(vector_size_t size, T start = 0) {
    return makeFlatVector<int32_t>(
        size, [start](auto row) { return start + row; });
  }

  template <typename T>
  VectorPtr lazySequence(vector_size_t size, T start = 0) {
    return vectorMaker_.lazyFlatVector<int32_t>(
        size, [start](auto row) { return start + row; });
  }

  void runTest(
      const std::vector<RowVectorPtr>& leftVectors,
      const std::vector<RowVectorPtr>& rightVectors,
      int32_t numDrivers = 1) {
    if (numDrivers == 1) {
      createDuckDbTable("t", leftVectors);
      createDuckDbTable("u", rightVectors);
    } else {
      auto allLeftVectors = makeCopies(leftVectors, numDrivers);
      auto allRightVectors = makeCopies(rightVectors, numDrivers);
      createDuckDbTable("t", allLeftVectors);
      createDuckDbTable("u", allRightVectors);
    }

    CursorParameters params;
    params.maxDrivers = numDrivers;
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    for (auto joinType : joinTypes_) {
      for (const auto& comparison : comparisons_) {
        SCOPED_TRACE(fmt::format(
            "maxDrivers:{} joinType:{} comparison:{}",
            std::to_string(numDrivers),
            joinTypeName(joinType),
            comparison));

        params.planNode = PlanBuilder(planNodeIdGenerator)
                              .values(leftVectors, numDrivers > 1)
                              .nestedLoopJoin(
                                  PlanBuilder(planNodeIdGenerator)
                                      .values(rightVectors, numDrivers > 1)
                                      .planNode(),
                                  fmt::format(joinConditionStr_, comparison),
                                  {"t0", "u0"},
                                  joinType)
                              .planNode();

        assertQuery(
            params, fmt::format(queryStr_, joinTypeName(joinType), comparison));
      }
    }
  }
};

TEST_F(NestedLoopJoinTest, basic) {
  auto leftVectors = makeBatches(20, 5, leftType_, pool_.get());
  auto rightVectors = makeBatches(18, 5, rightType_, pool_.get());
  runTest(leftVectors, rightVectors);
}

TEST_F(NestedLoopJoinTest, emptyLeft) {
  auto leftVectors = makeBatches(0, 5, leftType_, pool_.get());
  auto rightVectors = makeBatches(18, 5, rightType_, pool_.get());
  runTest(leftVectors, rightVectors);
}

TEST_F(NestedLoopJoinTest, emptyRight) {
  auto leftVectors = makeBatches(20, 5, leftType_, pool_.get());
  auto rightVectors = makeBatches(0, 5, rightType_, pool_.get());
  runTest(leftVectors, rightVectors);
}

TEST_F(NestedLoopJoinTest, basicCrossJoin) {
  auto leftVectors = {
      makeRowVector({sequence<int32_t>(10)}),
      makeRowVector({sequence<int32_t>(100, 10)}),
      makeRowVector({sequence<int32_t>(1'000, 10 + 100)}),
      makeRowVector({sequence<int32_t>(7, 10 + 100 + 1'000)}),
  };

  auto rightVectors = {
      makeRowVector({sequence<int32_t>(10)}),
      makeRowVector({sequence<int32_t>(100, 10)}),
      makeRowVector({sequence<int32_t>(1'000, 10 + 100)}),
      makeRowVector({sequence<int32_t>(11, 10 + 100 + 1'000)}),
  };

  createDuckDbTable("t", {leftVectors});
  createDuckDbTable("u", {rightVectors});

  // All x 13. Join output vectors contains multiple probe rows each.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto op = PlanBuilder(planNodeIdGenerator)
                .values({leftVectors})
                .nestedLoopJoin(
                    PlanBuilder(planNodeIdGenerator)
                        .values({rightVectors})
                        .filter("c0 < 13")
                        .project({"c0 AS u_c0"})
                        .planNode(),
                    {"c0", "u_c0"})
                .planNode();

  assertQuery(op, "SELECT * FROM t, u WHERE u.c0 < 13");

  // 13 x all. Join output vectors contains single probe row each.
  planNodeIdGenerator->reset();
  op = PlanBuilder(planNodeIdGenerator)
           .values({leftVectors})
           .filter("c0 < 13")
           .nestedLoopJoin(
               PlanBuilder(planNodeIdGenerator)
                   .values({rightVectors})
                   .project({"c0 AS u_c0"})
                   .planNode(),
               {"c0", "u_c0"})
           .planNode();

  assertQuery(op, "SELECT * FROM t, u WHERE t.c0 < 13");

  // All x 13. No columns on the build side.
  planNodeIdGenerator->reset();
  op = PlanBuilder(planNodeIdGenerator)
           .values({leftVectors})
           .nestedLoopJoin(
               PlanBuilder(planNodeIdGenerator)
                   .values({vectorMaker_.rowVector(ROW({}, {}), 13)})
                   .planNode(),
               {"c0"})
           .planNode();

  assertQuery(op, "SELECT t.* FROM t, (SELECT * FROM u LIMIT 13) u");

  // 13 x All. No columns on the build side.
  planNodeIdGenerator->reset();
  op = PlanBuilder(planNodeIdGenerator)
           .values({leftVectors})
           .filter("c0 < 13")
           .nestedLoopJoin(
               PlanBuilder(planNodeIdGenerator)
                   .values({vectorMaker_.rowVector(ROW({}, {}), 1121)})
                   .planNode(),
               {"c0"})
           .planNode();

  assertQuery(
      op,
      "SELECT t.* FROM (SELECT * FROM t WHERE c0 < 13) t, (SELECT * FROM u LIMIT 1121) u");

  // Empty build side.
  planNodeIdGenerator->reset();
  op = PlanBuilder(planNodeIdGenerator)
           .values({leftVectors})
           .nestedLoopJoin(
               PlanBuilder(planNodeIdGenerator)
                   .values({rightVectors})
                   .filter("c0 < 0")
                   .project({"c0 AS u_c0"})
                   .planNode(),
               {"c0", "u_c0"})
           .planNode();

  assertQueryReturnsEmptyResult(op);

  // Multi-threaded build side.
  planNodeIdGenerator->reset();
  CursorParameters params;
  params.maxDrivers = 4;
  params.planNode = PlanBuilder(planNodeIdGenerator)
                        .values({leftVectors})
                        .nestedLoopJoin(
                            PlanBuilder(planNodeIdGenerator, pool_.get())
                                .values({rightVectors}, true)
                                .filter("c0 in (10, 17)")
                                .project({"c0 AS u_c0"})
                                .planNode(),
                            {"c0", "u_c0"})
                        .limit(0, 100'000, false)
                        .planNode();

  OperatorTestBase::assertQuery(
      params,
      "SELECT * FROM t, (SELECT * FROM UNNEST (ARRAY[10, 17, 10, 17, 10, 17, 10, 17])) u");
}

TEST_F(NestedLoopJoinTest, lazyVectors) {
  auto leftVectors = {
      makeRowVector({lazySequence<int32_t>(10)}),
      makeRowVector({lazySequence<int32_t>(100, 10)}),
      makeRowVector({lazySequence<int32_t>(1'000, 10 + 100)}),
      makeRowVector({lazySequence<int32_t>(7, 10 + 100 + 1'000)}),
  };

  auto rightVectors = {
      makeRowVector({lazySequence<int32_t>(10)}),
      makeRowVector({lazySequence<int32_t>(100, 10)}),
      makeRowVector({lazySequence<int32_t>(1'000, 10 + 100)}),
      makeRowVector({lazySequence<int32_t>(11, 10 + 100 + 1'000)}),
  };

  createDuckDbTable("t", {makeRowVector({sequence<int32_t>(1117)})});
  createDuckDbTable("u", {makeRowVector({sequence<int32_t>(1121)})});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto op = PlanBuilder(planNodeIdGenerator)
                .values({leftVectors})
                .nestedLoopJoin(
                    PlanBuilder(planNodeIdGenerator)
                        .values({rightVectors})
                        .project({"c0 AS u_c0"})
                        .planNode(),
                    "c0+u_c0<100",
                    {"c0", "u_c0"},
                    core::JoinType::kFull)
                .planNode();

  assertQuery(op, "SELECT * FROM t FULL JOIN u ON t.c0 + u.c0 < 100");
}

// Test cross join with a build side that has rows, but no columns.
TEST_F(NestedLoopJoinTest, zeroColumnBuild) {
  auto leftVectors = {
      makeRowVector({sequence<int32_t>(10)}),
      makeRowVector({sequence<int32_t>(100, 10)}),
      makeRowVector({sequence<int32_t>(1'000, 10 + 100)}),
      makeRowVector({sequence<int32_t>(7, 10 + 100 + 1'000)}),
  };

  auto rightVectors = {
      makeRowVector({sequence<int32_t>(1)}),
      makeRowVector({sequence<int32_t>(4, 1)})};

  createDuckDbTable("t", {leftVectors});

  // Build side has > 1 row.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto op = PlanBuilder(planNodeIdGenerator)
                .values({leftVectors})
                .nestedLoopJoin(
                    PlanBuilder(planNodeIdGenerator)
                        .values({rightVectors})
                        .project({})
                        .planNode(),
                    {"c0"})
                .planNode();

  assertQuery(
      op, "SELECT t.* FROM t, (SELECT * FROM UNNEST (ARRAY[0, 1, 2, 3, 4])) u");

  // Build side has exactly 1 row.
  planNodeIdGenerator->reset();
  op = PlanBuilder(planNodeIdGenerator)
           .values({leftVectors})
           .nestedLoopJoin(
               PlanBuilder(planNodeIdGenerator)
                   .values({rightVectors})
                   .filter("c0 = 1")
                   .project({})
                   .planNode(),
               {"c0"})
           .planNode();

  assertQuery(op, "SELECT * FROM t");
}

TEST_F(NestedLoopJoinTest, parallel) {
  auto leftVectors = makeBatches(10, 5, leftType_, pool_.get());
  auto rightVectors = makeBatches(7, 5, rightType_, pool_.get());
  runTest(leftVectors, rightVectors, 5);
}

TEST_F(NestedLoopJoinTest, bigintArray) {
  auto leftVectors = makeBatches(1000, 5, leftType_, pool_.get());
  auto rightVectors = makeBatches(900, 5, rightType_, pool_.get());
  createDuckDbTable("t", leftVectors);
  createDuckDbTable("u", rightVectors);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values(leftVectors)
          .nestedLoopJoin(
              PlanBuilder(planNodeIdGenerator).values(rightVectors).planNode(),
              "t0 = u0",
              {"t0", "u0"},
              core::JoinType::kFull)
          .planNode();

  assertQuery(plan, "SELECT t0, u0 FROM t FULL JOIN u ON t.t0 = u.u0");
}

TEST_F(NestedLoopJoinTest, allTypes) {
  RowTypePtr leftType = ROW(
      {{"t0", BIGINT()},
       {"t1", VARCHAR()},
       {"t2", REAL()},
       {"t3", DOUBLE()},
       {"t4", INTEGER()},
       {"t5", SMALLINT()},
       {"t6", TINYINT()}});

  RowTypePtr rightType = ROW(
      {{"u0", BIGINT()},
       {"u1", VARCHAR()},
       {"u2", REAL()},
       {"u3", DOUBLE()},
       {"u4", INTEGER()},
       {"u5", SMALLINT()},
       {"u6", TINYINT()}});

  auto leftVectors = makeBatches(60, 5, leftType, pool_.get());
  auto rightVectors = makeBatches(50, 5, rightType, pool_.get());
  createDuckDbTable("t", leftVectors);
  createDuckDbTable("u", rightVectors);

  setLeftType(leftType);
  setRightType(rightType);
  setComparisons({"="});
  setJoinConditionStr(
      "t0 {0} u0 AND t1 {0} u1 AND t2 {0} u2 AND t3 {0} u3 AND t4 {0} u4 AND t5 {0} u5 AND t6 {0} u6");
  setQueryStr(
      "SELECT t0, u0 FROM t {0} JOIN u ON t.t0 {1} u0 AND t1 {1} u1 AND t2 {1} u2 AND t3 {1} u3 AND t4 {1} u4 AND t5 {1} u5 AND t6 {1} u6");
  runTest(leftVectors, rightVectors);
}
