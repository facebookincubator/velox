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

class WindowTestBase : public OperatorTestBase {
 protected:
  void SetUp() override {
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

  void testWindowSql(
      const std::vector<RowVectorPtr>& input,
      const std::string& function,
      const std::string& overClause) {
    auto windowSql = fmt::format("{}() over ({})", function, overClause);

    SCOPED_TRACE(windowSql);
    auto op = PlanBuilder().values(input).window({windowSql}).planNode();

    auto* rowType = dynamic_cast<const RowType*>(input[0]->type().get());
    std::string columnsString;
    for (const auto& name : rowType->names()) {
      columnsString = columnsString + name + ", ";
    }

    assertQuery(
        op, fmt::format("SELECT {} {} FROM tmp", columnsString, windowSql));
  };

  void testWindowClauses(
      const std::vector<RowVectorPtr>& input,
      const std::string& function,
      const std::vector<std::string> overClauses) {
    for (const auto& overClause : overClauses) {
      testWindowSql(input, function, overClause);
    }
  }

  void twoColumnTests(
      const std::vector<RowVectorPtr>& vectors,
      const std::string& windowFunction) {
    VELOX_CHECK_EQ(vectors[0]->childrenSize(), 2);
    std::vector<std::string> overClauses = {
        "partition by c0 order by c1",
        "partition by c1 order by c0",
        "partition by c0 order by c1 desc",
        "partition by c1 order by c0 desc",
        // No partition by clause.
        "order by c0, c1",
        "order by c1, c0",
        "order by c0 asc, c1 desc",
        "order by c1 asc, c0 desc",
        // No order by clause.
        "partition by c0, c1",
    };

    testWindowClauses(vectors, windowFunction, overClauses);
  }
};
}; // namespace facebook::velox::window::test
