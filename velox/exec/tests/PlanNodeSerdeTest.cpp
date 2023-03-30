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
#include "velox/exec/WindowFunction.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

namespace facebook::velox::exec::test {

class PlanNodeSerdeTest : public testing::Test,
                          public velox::test::VectorTestBase {
 protected:
  PlanNodeSerdeTest() {
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    parse::registerTypeResolver();

    Type::registerSerDe();
    core::PlanNode::registerSerDe();
    core::ITypedExpr::registerSerDe();
    PlanBuilder::registerSerDe();

    data_ = {makeRowVector({
        makeFlatVector<int64_t>({1, 2, 3}),
        makeFlatVector<int32_t>({10, 20, 30}),
        makeConstant(true, 3),
    })};
  }

  void testSerde(const core::PlanNodePtr& plan) {
    auto serialized = plan->serialize();

    auto copy =
        velox::ISerializable::deserialize<core::PlanNode>(serialized, pool());

    ASSERT_EQ(plan->toString(true, true), copy->toString(true, true));
  }

  std::vector<RowVectorPtr> data_;
};

TEST_F(PlanNodeSerdeTest, aggregation) {
  auto plan = PlanBuilder()
                  .values({data_})
                  .partialAggregation({"c0"}, {"count(1)", "sum(c1)"})
                  .finalAggregation()
                  .planNode();

  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, assignUniqueId) {
  auto plan = PlanBuilder().values({data_}).assignUniqueId().planNode();
  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, crossJoin) {
  auto left = makeRowVector(
      {"t0", "t1", "t2"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>({10, 20, 30}),
          makeFlatVector<bool>({true, true, false}),
      });

  auto right = makeRowVector(
      {"u0", "u1", "u2"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>({10, 20, 30}),
          makeFlatVector<bool>({true, true, false}),
      });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values({left})
          .crossJoin(
              PlanBuilder(planNodeIdGenerator).values({right}).planNode(),
              {"t0", "u1", "t2", "t1"})
          .planNode();
  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, enforceSingleRow) {
  auto plan = PlanBuilder().values({data_}).enforceSingleRow().planNode();
  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, exchange) {
  auto plan =
      PlanBuilder()
          .exchange(ROW({"a", "b", "c"}, {BIGINT(), DOUBLE(), VARCHAR()}))
          .planNode();
  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, filter) {
  auto plan = PlanBuilder().values({data_}).filter("c0 > 100").planNode();
  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, groupId) {
  auto plan = PlanBuilder()
                  .values({data_})
                  .groupId({{"c0"}, {"c0", "c1"}}, {"c2"})
                  .planNode();
  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, localPartition) {
  auto plan = PlanBuilder().values({data_}).localPartition({}).planNode();
  testSerde(plan);

  plan = PlanBuilder().values({data_}).localPartition({"c0", "c1"}).planNode();
  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, limit) {
  auto plan = PlanBuilder().values({data_}).limit(0, 10, true).planNode();
  testSerde(plan);

  plan = PlanBuilder().values({data_}).limit(0, 10, false).planNode();
  testSerde(plan);

  plan = PlanBuilder().values({data_}).limit(12, 10, false).planNode();
  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, mergeExchange) {
  auto plan = PlanBuilder()
                  .mergeExchange(
                      ROW({"a", "b", "c"}, {BIGINT(), DOUBLE(), VARCHAR()}),
                      {"a DESC", "b NULLS FIRST"})
                  .planNode();
  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, localMerge) {
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .localMerge(
              {"c0"},
              {
                  PlanBuilder(planNodeIdGenerator).values({data_}).planNode(),
                  PlanBuilder(planNodeIdGenerator).values({data_}).planNode(),
              })
          .planNode();

  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, mergeJoin) {
  auto probe = makeRowVector(
      {"t0", "t1", "t2"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>({10, 20, 30}),
          makeFlatVector<bool>({true, true, false}),
      });

  auto build = makeRowVector(
      {"u0", "u1", "u2"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>({10, 20, 30}),
          makeFlatVector<bool>({true, true, false}),
      });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values({probe})
          .mergeJoin(
              {"t0"},
              {"u0"},
              PlanBuilder(planNodeIdGenerator).values({build}).planNode(),
              "t1 > u1",
              {"t0", "t1", "u2", "t2"},
              core::JoinType::kInner)
          .planNode();

  testSerde(plan);

  plan = PlanBuilder(planNodeIdGenerator)
             .values({probe})
             .mergeJoin(
                 {"t0"},
                 {"u0"},
                 PlanBuilder(planNodeIdGenerator).values({build}).planNode(),
                 "", // no filter
                 {"t0", "t1", "u2", "t2"},
                 core::JoinType::kLeft)
             .planNode();

  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, orderBy) {
  auto plan = PlanBuilder()
                  .values({data_})
                  .orderBy({"c0", "c1 DESC NULLS FIRST"}, true)
                  .planNode();

  testSerde(plan);

  plan = PlanBuilder()
             .values({data_})
             .orderBy({"c0", "c1 DESC NULLS FIRST"}, false)
             .planNode();

  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, partitionedOutput) {
  auto plan =
      PlanBuilder().values({data_}).partitionedOutputBroadcast().planNode();
  testSerde(plan);

  plan = PlanBuilder().values({data_}).partitionedOutput({"c0"}, 50).planNode();
  testSerde(plan);

  plan = PlanBuilder()
             .values({data_})
             .partitionedOutput({"c0"}, 50, {"c1", {"c2"}, "c0"})
             .planNode();
  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, project) {
  auto plan = PlanBuilder()
                  .values({data_})
                  .project({"c0 * 10", "c0 + c1", "c1 > 0"})
                  .planNode();

  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, hashJoin) {
  auto probe = makeRowVector(
      {"t0", "t1", "t2"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>({10, 20, 30}),
          makeFlatVector<bool>({true, true, false}),
      });

  auto build = makeRowVector(
      {"u0", "u1", "u2"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>({10, 20, 30}),
          makeFlatVector<bool>({true, true, false}),
      });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values({probe})
          .hashJoin(
              {"t0"},
              {"u0"},
              PlanBuilder(planNodeIdGenerator).values({build}).planNode(),
              "t1 > u1",
              {"t0", "t1", "u2", "t2"},
              core::JoinType::kInner)
          .planNode();

  testSerde(plan);

  plan = PlanBuilder(planNodeIdGenerator)
             .values({probe})
             .hashJoin(
                 {"t0"},
                 {"u0"},
                 PlanBuilder(planNodeIdGenerator).values({build}).planNode(),
                 "", // no filter
                 {"t0", "t1", "u2", "t2"},
                 core::JoinType::kLeft)
             .planNode();

  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, topN) {
  auto plan = PlanBuilder().values({data_}).topN({"c0"}, 10, true).planNode();
  testSerde(plan);

  plan = PlanBuilder().values({data_}).topN({"c0 DESC"}, 10, false).planNode();
  testSerde(plan);

  plan = PlanBuilder()
             .values({data_})
             .topN({"c0", "c1 DESC NULLS FIRST"}, 10, true)
             .planNode();
  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, unnest) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeArrayVector<int32_t>({
          {1, 2},
          {3, 4, 5},
          {},
      }),
      makeMapVector<int64_t, int32_t>(
          {{{1, 10}, {2, 20}}, {{3, 30}, {4, 40}, {5, 50}}, {}}),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .unnest({"c0"}, {"c1", "c2"}, std::nullopt)
                  .planNode();
  testSerde(plan);

  plan =
      PlanBuilder().values({data}).unnest({"c0"}, {"c1"}, "ordinal").planNode();
  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, values) {
  auto plan = PlanBuilder().values({data_}).planNode();
  testSerde(plan);

  plan = PlanBuilder()
             .values({data_}, true /*parallelizable*/, 5 /*repeatTimes*/)
             .planNode();
  testSerde(plan);
}

TEST_F(PlanNodeSerdeTest, window) {
  auto plan =
      PlanBuilder()
          .values({data_})
          .window(
              {"sum(c0) over (partition by c1 order by c2 rows between 10 preceding and 5 following)"})
          .planNode();

  testSerde(plan);

  plan = PlanBuilder()
             .values({data_})
             .window({"sum(c0) over (partition by c1 order by c2)"})
             .planNode();

  testSerde(plan);
}

} // namespace facebook::velox::exec::test
