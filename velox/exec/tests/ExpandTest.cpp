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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::exec {
namespace {
class ExpandTest : public OperatorTestBase {
 public:
  RowVectorPtr makeRowVectorData(vector_size_t size) {
    return makeRowVector(
        {"k1", "k2", "a", "b"},
        {
            makeFlatVector<int64_t>(size, [](auto row) { return row % 11; }),
            makeFlatVector<int64_t>(size, [](auto row) { return row % 17; }),
            makeFlatVector<int64_t>(size, [](auto row) { return row; }),
            makeFlatVector<std::string>(
                size, [](auto row) { return std::string(row % 15, 'x'); }),
        });
  }
};

TEST_F(ExpandTest, complexConstant) {
  auto data = makeRowVectorData(3);
  auto children = data->children();
  auto arrayVector =
      makeArrayVector<int32_t>({{1, 2, 3}, {1, 2, 3}, {1, 2, 3}});
  children.push_back(arrayVector);
  children.push_back(makeAllNullArrayVector(3, INTEGER()));
  children.push_back(makeNullConstant(TypeKind::INTEGER, 3));
  auto expected = makeRowVector(children);

  auto plan = PlanBuilder(pool())
                  .values({data})
                  .expand(
                      {{"k1",
                        "k2",
                        "a",
                        "b",
                        "ARRAY[1, 2, 3] as c",
                        "null::integer[] as d",
                        "null::integer as e"}})
                  .planNode();

  assertQuery(plan, expected);
}

TEST_F(ExpandTest, groupingSets) {
  auto data = makeRowVectorData(1'000);

  createDuckDbTable({data});

  auto plan =
      PlanBuilder()
          .values({data})
          .expand(
              {{"k1",
                "null::bigint as k2",
                "a",
                "b",
                "0 as group_id_0",
                "0 as group_id_1"},
               {"k1", "null", "a", "b", "0", "1"},
               {"null", "k2", "a", "b", "1", "2"}})
          .singleAggregation(
              {"k1", "k2", "group_id_0", "group_id_1"},
              {"count(1) as count_1", "sum(a) as sum_a", "max(b) as max_b"})
          .project({"k1", "k2", "count_1", "sum_a", "max_b"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, count(1), sum(a), max(b) FROM tmp GROUP BY GROUPING SETS ((k1), (k1), (k2))");
}

TEST_F(ExpandTest, cube) {
  auto data = makeRowVectorData(1'000);

  createDuckDbTable({data});

  // Cube.
  auto plan =
      PlanBuilder()
          .values({data})
          .expand({
              {"k1", "k2", "a", "b", "0 as gid"},
              {"k1", "null", "a", "b", "1"},
              {"null", "k2", "a", "b", "2"},
              {"null", "null", "a", "b", "3"},
          })
          .singleAggregation(
              {"k1", "k2", "gid"},
              {"count(1) as count_1", "sum(a) as sum_a", "max(b) as max_b"})
          .project({"k1", "k2", "count_1", "sum_a", "max_b"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, count(1), sum(a), max(b) FROM tmp GROUP BY CUBE (k1, k2)");
}

TEST_F(ExpandTest, rollup) {
  auto data = makeRowVectorData(1'000);

  createDuckDbTable({data});

  // Rollup.
  auto plan =
      PlanBuilder()
          .values({data})
          .expand(
              {{"k1 as foo", "k2", "a", "b", "0 as gid"},
               {"k1", "null", "a", "b", "1"},
               {"null", "null", "a", "b", "2"}})
          .singleAggregation(
              {"foo", "k2", "gid"},
              {"count(1) as count_1", "sum(a) as sum_a", "max(b) as max_b"})
          .project({"foo", "k2", "count_1", "sum_a", "max_b"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, count(1), sum(a), max(b) FROM tmp GROUP BY ROLLUP (k1, k2)");
}

TEST_F(ExpandTest, countDistinct) {
  auto data = makeRowVectorData(1'000);

  createDuckDbTable({data});

  // count distinct.
  auto plan =
      PlanBuilder()
          .values({data})
          .expand({{"a", "null::varchar as b", "1 as gid"}, {"null", "b", "2"}})
          .singleAggregation({"a", "b", "gid"}, {})
          .singleAggregation({}, {"count(a) as count_a", "count(b) as count_b"})
          .planNode();

  assertQuery(plan, "SELECT count(distinct a), count(distinct b) FROM tmp");
}

TEST_F(ExpandTest, invalidUseCases) {
  auto data = makeRowVector(
      ROW({"k1", "k2", "a", "b"}, {BIGINT(), BIGINT(), BIGINT(), VARCHAR()}),
      10);

  VELOX_ASSERT_USER_THROW(
      PlanBuilder().values({data}).expand(
          {{"k1", "k1", "a", "b", "0 as gid"},
           {"k1", "null", "a", "b", "1"},
           {"null", "null", "a", "b", "2"}}),
      "Found duplicate column name in Expand plan node: k1.");

  VELOX_ASSERT_RUNTIME_THROW(
      PlanBuilder().values({data}).expand({}),
      "projections must not be empty.");
}

} // namespace
} // namespace facebook::velox::exec
