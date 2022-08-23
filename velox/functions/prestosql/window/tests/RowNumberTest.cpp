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
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::window::test {

namespace {

class RowNumberTest : public OperatorTestBase {
 protected:
  void SetUp() {
    exec::test::OperatorTestBase::SetUp();
    velox::window::registerWindowFunctions();
  }

  std::vector<RowVectorPtr>
  makeVectors(const RowTypePtr& rowType, vector_size_t size, int numVectors) {
    std::vector<RowVectorPtr> vectors;
    VectorFuzzer::Options options;
    options.vectorSize = size;
    VectorFuzzer fuzzer(options, pool_.get(), 0);
    for (int32_t i = 0; i < numVectors; ++i) {
      auto vector =
          std::dynamic_pointer_cast<RowVector>(fuzzer.fuzzRow(rowType));
      vectors.push_back(vector);
    }
    return vectors;
  }

  void basicTests(const RowVectorPtr& vectors) {
    auto testWindowSql = [&](const RowVectorPtr& input,
                             const std::string& windowSql) -> void {
      VELOX_CHECK_GE(input->size(), 2);

      auto op = PlanBuilder().values({input}).window({windowSql}).planNode();
      assertQuery(op, "SELECT c0, c1, " + windowSql + " FROM tmp");
    };

    std::vector<std::string> overClauses = {
        "partition by c0 order by c1",
        "partition by c1 order by c0",
        "partition by c0 order by c1 desc",
        "partition by c1 order by c0 desc",
        // No partition by clause
        "order by c0, c1",
        "order by c1, c0",
        "order by c0 asc, c1 desc",
        "order by c1 asc, c0 desc",
        // No order by clause
        "partition by c0, c1",
    };

    for (const auto& overClause : overClauses) {
      testWindowSql(
          vectors,
          "row_number() over ( " + overClause + " ) as row_number_partition");
    }
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

TEST_F(RowNumberTest, basic) {
  vector_size_t size = 100;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(
          size, [](auto row) -> int32_t { return row % 5; }),
      makeFlatVector<int32_t>(
          size, [](auto row) -> int32_t { return row % 7; }),
  });

  createDuckDbTable({vectors});
  basicTests(vectors);
}

TEST_F(RowNumberTest, singlePartition) {
  // Test all input rows in a single partition. This data size would
  // need multiple input blocks.
  vector_size_t size = 1000;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) -> int32_t { return 1; }),
      makeFlatVector<int32_t>(size, [](auto row) -> int32_t { return row; }),
  });

  createDuckDbTable({vectors});
  basicTests(vectors);
}

TEST_F(RowNumberTest, randomGen) {
  auto vectors = makeVectors(rowType_, 10, 1);
  createDuckDbTable(vectors);

  auto testWindowSql = [&](std::vector<RowVectorPtr>& input,
                           const std::string& windowSql) -> void {
    auto op = PlanBuilder()
                  .values(input)
                  .project({"c0 as c0", "c1 as c1", "c2 as c2", "c3 as c3"})
                  .window({windowSql})
                  .planNode();
    assertQuery(op, "SELECT c0, c1, c2, c3, " + windowSql + " FROM tmp");
  };

  std::vector<std::string> overClauses = {
      "partition by c0 order by c1, c2, c3",
      "partition by c1 order by c0, c2, c3",
      "partition by c0 order by c1 desc, c2, c3",
      "partition by c1 order by c0 desc, c2, c3",
      "order by c0, c1, c2, c3",
      "partition by c0, c1, c2, c3",
  };

  for (const auto& overClause : overClauses) {
    testWindowSql(
        vectors,
        "row_number() over (" + overClause + " ) as row_number_partition");
  }
}

}; // namespace
}; // namespace facebook::velox::window::test
