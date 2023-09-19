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
using namespace facebook::velox::exec::test;

class ExpandTest : public OperatorTestBase {};

TEST_F(ExpandTest, groupingSets) {
  vector_size_t size = 1'000;
  auto data = makeRowVector(
      {"k1", "k2", "a", "b"},
      {
          makeFlatVector<int64_t>(size, [](auto row) { return row % 11; }),
          makeFlatVector<int64_t>(size, [](auto row) { return row % 17; }),
          makeFlatVector<int64_t>(size, [](auto row) { return row; }),
          makeFlatVector<StringView>(
              size,
              [](auto row) {
                auto str = std::string(row % 12, 'x');
                return StringView(str);
              }),
      });

  createDuckDbTable({data});

  auto plan =
      PlanBuilder()
          .values({data})
          .expand(
              {{"k1", "", "a", "b", "0", "0"},
               {"k1", "", "a", "b", "0", "1"},
               {"", "k2", "a", "b", "1", "2"}})
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
  vector_size_t size = 1'000;
  auto data = makeRowVector(
      {"k1", "k2", "a", "b"},
      {
          makeFlatVector<int64_t>(size, [](auto row) { return row % 11; }),
          makeFlatVector<int64_t>(size, [](auto row) { return row % 17; }),
          makeFlatVector<int64_t>(size, [](auto row) { return row; }),
          makeFlatVector<StringView>(
              size,
              [](auto row) {
                auto str = std::string(row % 12, 'x');
                return StringView(str);
              }),
      });

  createDuckDbTable({data});

  // Cube.
  auto plan =
      PlanBuilder()
          .values({data})
          .expand({
              {"k1", "k2", "a", "b", "0"},
              {"k1", "", "a", "b", "1"},
              {"", "k2", "a", "b", "2"},
              {"", "", "a", "b", "3"},
          })
          .singleAggregation(
              {"k1", "k2", "group_id_0"},
              {"count(1) as count_1", "sum(a) as sum_a", "max(b) as max_b"})
          .project({"k1", "k2", "count_1", "sum_a", "max_b"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, count(1), sum(a), max(b) FROM tmp GROUP BY CUBE (k1, k2)");
}

TEST_F(ExpandTest, rollup) {
  vector_size_t size = 1'000;
  auto data = makeRowVector(
      {"k1", "k2", "a", "b"},
      {
          makeFlatVector<int64_t>(size, [](auto row) { return row % 11; }),
          makeFlatVector<int64_t>(size, [](auto row) { return row % 17; }),
          makeFlatVector<int64_t>(size, [](auto row) { return row; }),
          makeFlatVector<StringView>(
              size,
              [](auto row) {
                auto str = std::string(row % 12, 'x');
                return StringView(str);
              }),
      });

  createDuckDbTable({data});

  // Rollup.
  auto plan =
      PlanBuilder()
          .values({data})
          .expand(
              {{"k1", "k2", "a", "b", "0"},
               {"k1", "", "a", "b", "1"},
               {"", "", "a", "b", "2"}})
          .singleAggregation(
              {"k1", "k2", "group_id_0"},
              {"count(1) as count_1", "sum(a) as sum_a", "max(b) as max_b"})
          .project({"k1", "k2", "count_1", "sum_a", "max_b"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, count(1), sum(a), max(b) FROM tmp GROUP BY ROLLUP (k1, k2)");
}

TEST_F(ExpandTest, countDistinct) {
  vector_size_t size = 1'000;
  auto data = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>(size, [](auto row) { return row; }),
          makeFlatVector<StringView>(
              size,
              [](auto row) {
                auto str = std::string(row % 12, 'x');
                return StringView(str);
              }),
      });
  createDuckDbTable({data});

  // count distinct.
  auto plan =
      PlanBuilder()
          .values({data})
          .expand({{"a", "", "1"}, {"", "b", "2"}})
          .singleAggregation(
              {},
              {"count(distinct a) as count_a", "count(distinct b) as count_b"})
          .project({"count_a", "count_b"})
          .planNode();

  assertQuery(plan, "SELECT count(distinct a), count(distinct b) FROM tmp");
}
