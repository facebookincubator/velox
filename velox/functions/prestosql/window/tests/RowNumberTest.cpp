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
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::window::test {

namespace {

class RowNumberTest : public OperatorTestBase {
 protected:
  void SetUp() {
    exec::test::OperatorTestBase::SetUp();
    velox::window::registerWindowFunctions();
  }

  std::vector<RowVectorPtr> makeVectors(
      const std::shared_ptr<const RowType>& rowType,
      vector_size_t size,
      int numVectors,
      std::function<bool(vector_size_t /*index*/)> isNullAt) {
    std::vector<RowVectorPtr> vectors;
    for (int32_t i = 0; i < numVectors; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          velox::test::BatchMaker::createBatch(
              rowType, size, *pool_, isNullAt));
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

TEST_F(RowNumberTest, basicRowNumber) {
  auto testWindowSql = [&](const RowVectorPtr& input,
                           const std::string& windowSql) -> void {
    VELOX_CHECK_GE(input->size(), 2);

    auto op = PlanBuilder()
                  .values({input})
                  .project({"c0 as c0", "c1 as c1"})
                  .window({windowSql})
                  .orderBy({"c0 asc nulls last", "c1 asc nulls last"}, false)
                  .planNode();
    assertQuery(
        op, "SELECT c0, c1, " + windowSql + " FROM tmp order by c0, c1");
  };

  auto basicTests = [&](const RowVectorPtr& vectors) -> void {
    testWindowSql(
        vectors,
        "row_number() over (partition by c0 order by c1) as row_number_partition");
    testWindowSql(
        vectors,
        "row_number() over (partition by c1 order by c0) as row_number_partition");
    testWindowSql(
        vectors,
        "row_number() over (partition by c0 order by c1 desc) as row_number_partition");
    testWindowSql(
        vectors,
        "row_number() over (partition by c1 order by c0 desc) as row_number_partition");

    // No partition clause
    testWindowSql(
        vectors, "row_number() over (order by c0, c1) as row_number_partition");
    testWindowSql(
        vectors, "row_number() over (order by c1, c0) as row_number_partition");
    testWindowSql(
        vectors,
        "row_number() over (order by c0 asc, c1 desc) as row_number_partition");
    testWindowSql(
        vectors,
        "row_number() over (order by c1 asc, c0 desc) as row_number_partition");

    // No order by clause
    testWindowSql(
        vectors,
        "row_number() over (partition by c0, c1) as row_number_partition");
  };
  vector_size_t size = 100;
  auto valueAtC0 = [](auto row) -> int32_t { return row % 5; };
  auto valueAtC1 = [](auto row) -> int32_t { return row % 7; };

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, valueAtC0),
      makeFlatVector<int32_t>(size, valueAtC1),
  });

  createDuckDbTable({vectors});
  basicTests(vectors);

  // Test all input in a single partition.
  size = 50;
  auto valueAtC01 = [](auto row) -> int32_t { return 1; };
  auto valueAtC11 = [](auto row) -> int32_t { return row; };

  vectors = makeRowVector({
      makeFlatVector<int32_t>(size, valueAtC0),
      makeFlatVector<int32_t>(size, valueAtC1),
  });

  createDuckDbTable({vectors});
  basicTests(vectors);
}

TEST_F(RowNumberTest, rowNumberRandomGen) {
  // Add nulls in the mix too.
  auto vectors = makeVectors(
      rowType_, 100, 100, [](vector_size_t i) { return i % 35 == 0; });
  createDuckDbTable(vectors);

  auto testWindowSql = [&](std::vector<RowVectorPtr>& input,
                           const std::string& windowSql) -> void {
    auto op = PlanBuilder()
                  .values(input)
                  .project({"c0 as c0", "c1 as c1", "c2 as c2", "c3 as c3"})
                  .window({windowSql})
                  .orderBy(
                      {"c0 asc nulls last",
                       "c1 asc nulls last",
                       "c2 asc nulls last",
                       "c3 asc nulls last"},
                      false)
                  .planNode();
    assertQuery(
        op,
        "SELECT c0, c1, c2, c3, " + windowSql +
            " FROM tmp order by c0, c1, c2, c3");
  };

  testWindowSql(
      vectors,
      "row_number() over (partition by c0 order by c1) as row_number_partition");
  testWindowSql(
      vectors,
      "row_number() over (partition by c1 order by c0) as row_number_partition");
  testWindowSql(
      vectors,
      "row_number() over (partition by c0 order by c1 desc) as row_number_partition");
  testWindowSql(
      vectors,
      "row_number() over (partition by c1 order by c0 desc) as row_number_partition");

  testWindowSql(
      vectors,
      "row_number() over (partition by c0, c1) as row_number_partition");
}

}; // namespace
}; // namespace facebook::velox::window::test
