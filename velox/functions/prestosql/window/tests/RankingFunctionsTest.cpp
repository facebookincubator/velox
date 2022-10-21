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
#include "velox/functions/prestosql/window/tests/WindowTestBase.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::window::test {

namespace {

struct TestParam {
  std::string rankFunction;
  std::string overClause;
};

const std::vector<std::string> kRankFunctions_ = {
    "cume_dist()",
    "dense_rank()",
    "percent_rank()",
    "rank()",
    "row_number()",
};

const std::vector<std::string> kOverClauses_ = {
    "partition by c0 order by c1",
    "partition by c1 order by c0",
    "partition by c0 order by c1 desc",
    "partition by c1 order by c0 desc",
    "partition by c0 order by c1 nulls first",
    "partition by c1 order by c0 nulls first",
    "partition by c0 order by c1 desc nulls first",
    "partition by c1 order by c0 desc nulls first",
    // No partition by clause.
    "order by c0, c1",
    "order by c1, c0",
    "order by c0 asc, c1 desc",
    "order by c1 asc, c0 desc",
    "order by c0 asc nulls first, c1 desc nulls first",
    "order by c1 asc nulls first, c0 desc nulls first",
    "order by c0 desc nulls first, c1 asc nulls first",
    "order by c1 desc nulls first, c0 asc nulls first",
    // No order by clause.
    "partition by c0, c1",
};

const std::vector<TestParam> getTestParams() {
  std::vector<TestParam> testParams;
  for (auto rankFunction : kRankFunctions_) {
    for (auto overClause : kOverClauses_) {
      testParams.push_back({rankFunction, overClause});
    }
  }
  return testParams;
}

class RankTest : public WindowTestBase,
                 public testing::WithParamInterface<TestParam> {
 protected:
  RankTest()
      : rankFunction_(GetParam().rankFunction),
        overClause_(GetParam().overClause) {}

  void testTwoColumnOverClauses(const RowVectorPtr& vectors) {
    WindowTestBase::testTwoColumnOverClauses(
        {vectors}, rankFunction_, overClause_);
  }

  void testWindowFunction(const std::vector<RowVectorPtr>& vectors) {
    WindowTestBase::testWindowFunction(vectors, rankFunction_, overClause_);
  }

  const std::string rankFunction_;
  const std::string overClause_;
};

TEST_P(RankTest, basic) {
  vector_size_t size = 1000;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row % 10; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 7; }),
  });

  testTwoColumnOverClauses({vectors});
}

TEST_P(RankTest, singlePartition) {
  // Test all input rows in a single partition.
  vector_size_t size = 1'000;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(
          size, [](auto row) { return row % 50; }, nullEvery(7)),
  });

  testTwoColumnOverClauses({vectors});
}

TEST_P(RankTest, singleRowPartitions) {
  vector_size_t size = 1000;
  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row; }),
  });

  testTwoColumnOverClauses({vectors});
}

TEST_P(RankTest, randomInput) {
  auto vectors = makeVectors(
      ROW({"c0", "c1", "c2", "c3"},
          {BIGINT(), SMALLINT(), INTEGER(), BIGINT()}),
      10,
      2,
      0.3);
  createDuckDbTable(vectors);

  testWindowFunction(vectors);
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    RankTest,
    RankTest,
    testing::ValuesIn(getTestParams()));

}; // namespace
}; // namespace facebook::velox::window::test
