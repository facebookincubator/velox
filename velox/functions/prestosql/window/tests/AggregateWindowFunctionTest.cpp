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
#include "velox/dwio/dwrf/test/utils/BatchMaker.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::window::test {

namespace {

class AggregateWindowFunctionTest : public OperatorTestBase {
 protected:
  void SetUp() {
    velox::window::registerWindowFunctions();
  }

  std::vector<RowVectorPtr> makeVectors(
      const std::shared_ptr<const RowType>& rowType,
      vector_size_t size,
      int numVectors) {
    std::vector<RowVectorPtr> vectors;
    for (int32_t i = 0; i < numVectors; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          velox::test::BatchMaker::createBatch(rowType, size, *pool_));
      vectors.push_back(vector);
    }
    return vectors;
  }

  std::shared_ptr<const RowType> rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {BIGINT(),
           SMALLINT(),
           INTEGER(),
           BIGINT(),
           REAL(),
           DOUBLE(),
           VARCHAR()})};
  folly::Random::DefaultGenerator rng_;
};

TEST_F(AggregateWindowFunctionTest, basicSum) {
  auto vectors = makeVectors(rowType_, 10, 1);
  createDuckDbTable(vectors);

  auto op = PlanBuilder()
                .values(vectors)
                .project({"c0 as c0", "c1 as c1", "c2 as c2"})
                .window({"c0"}, {"c1 asc nulls last"}, {"sum(c2) as sum_c2"})
                .orderBy({"c0 asc nulls last", "c1 asc nulls last"}, false)
                .planNode();
  assertQuery(
      op,
      "SELECT c0, c1, c2, sum(c2) over (partition by c0 order by c1) as sum_c2 FROM tmp order by c0, c1");
}

TEST_F(AggregateWindowFunctionTest, basicSum2) {
  vector_size_t size = 10;
  auto valueAtC0 = [](auto row) -> int32_t { return row % 5; };
  auto valueAtC1 = [](auto row) -> int32_t { return row % 5; };
  auto valueAtC2 = [](auto row) -> int32_t { return row; };

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, valueAtC0),
      makeFlatVector<int32_t>(size, valueAtC1),
      makeFlatVector<int32_t>(size, valueAtC2),
  });

  createDuckDbTable({vectors});

  auto op = PlanBuilder()
                .values({vectors})
                .project({"c0 as c0", "c1 as c1", "c2 as c2"})
                .window({"c0"}, {"c1 asc nulls last"}, {"sum(c2) as sum_c2"})
                .orderBy({"c0 asc nulls last", "c1 asc nulls last"}, false)
                .planNode();
  assertQuery(
      op,
      "SELECT c0, c1, c2, sum(c2) over (partition by c0 order by c1) as "
      "sum_c2 FROM tmp order by c0, c1");
}
}; // namespace
}; // namespace facebook::velox::window::test
