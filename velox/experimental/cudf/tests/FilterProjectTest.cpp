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
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

namespace {

class CudfFilterProjectTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::registerCudf();
    rng_.seed(123);

    rowType_ = ROW({{"c0", INTEGER()}, {"c1", DOUBLE()}, {"c2", VARCHAR()}});
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  void testMultiplyOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with a multiply operation
    auto plan =
        PlanBuilder().values(input).project({"1.0 * c1 AS result"}).planNode();

    // Run the test
    runTest(plan, "SELECT 1.0 * c1 AS result FROM tmp");
  }

  void testDivideOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with a divide operation
    auto plan =
        PlanBuilder().values(input).project({"c0 / c1 AS result"}).planNode();

    // Run the test
    runTest(plan, "SELECT c0 / c1 AS result FROM tmp");
  }

  void testMultiplyAndMinusOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with a multiply and minus operation
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"c0 * (1.0 - c1) AS result"})
                    .planNode();

    // Run the test
    runTest(plan, "SELECT c0 * (1.0 - c1) AS result FROM tmp");
  }

  void testStringEqualOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with a string equal operation
    auto c2Value = input[0]
                       ->as<RowVector>()
                       ->childAt(2)
                       ->as<FlatVector<StringView>>()
                       ->valueAt(1)
                       .str();
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"c2 = '" + c2Value + "' AS result"})
                    .planNode();

    // Run the test
    runTest(plan, "SELECT c2 = '" + c2Value + "' AS result FROM tmp");
  }

  void testStringNotEqualOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with a string not equal operation
    auto c2Value = input[0]
                       ->as<RowVector>()
                       ->childAt(2)
                       ->as<FlatVector<StringView>>()
                       ->valueAt(1)
                       .str();
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"c2 <> '" + c2Value + "' AS result"})
                    .planNode();

    // Run the test
    runTest(plan, "SELECT c2 <> '" + c2Value + "' AS result FROM tmp");
  }

  void testAndOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with AND operation
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"c0 = 1 AND c1 = 2.0 AS result"})
                    .planNode();

    // Run the test
    runTest(plan, "SELECT c0 = 1 AND c1 = 2.0 AS result FROM tmp");
  }

  void testOrOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with OR operation
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"c0 = 1 OR c1 = 2.0 AS result"})
                    .planNode();

    // Run the test
    runTest(plan, "SELECT c0 = 1 OR c1 = 2.0 AS result FROM tmp");
  }

  void testYearFunction(const std::vector<RowVectorPtr>& input) {
    // Create a plan with YEAR function
    auto plan =
        PlanBuilder().values(input).project({"YEAR(c2) AS result"}).planNode();

    // Run the test
    runTest(plan, "SELECT YEAR(c2) AS result FROM tmp");
  }

  void testLengthFunction(const std::vector<RowVectorPtr>& input) {
    // Create a plan with LENGTH function
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"LENGTH(c2) AS result"})
                    .planNode();

    // Run the test
    runTest(plan, "SELECT LENGTH(c2) AS result FROM tmp");
  }

  void runTest(core::PlanNodePtr planNode, const std::string& duckDbSql) {
    SCOPED_TRACE("run without spilling");
    assertQuery(planNode, duckDbSql);
  }

  std::vector<RowVectorPtr> makeVectors(
      const RowTypePtr& rowType,
      int32_t numVectors,
      int32_t rowsPerVector) {
    std::vector<RowVectorPtr> vectors;
    for (int32_t i = 0; i < numVectors; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          facebook::velox::test::BatchMaker::createBatch(
              rowType, rowsPerVector, *pool_));
      vectors.push_back(vector);
    }
    return vectors;
  }

  folly::Random::DefaultGenerator rng_;
  RowTypePtr rowType_;
};

TEST_F(CudfFilterProjectTest, multiplyOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testMultiplyOperation(vectors);
}

TEST_F(CudfFilterProjectTest, divideOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testDivideOperation(vectors);
}

TEST_F(CudfFilterProjectTest, multiplyAndMinusOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testMultiplyAndMinusOperation(vectors);
}

TEST_F(CudfFilterProjectTest, stringEqualOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testStringEqualOperation(vectors);
}

TEST_F(CudfFilterProjectTest, stringNotEqualOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testStringNotEqualOperation(vectors);
}

TEST_F(CudfFilterProjectTest, andOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testAndOperation(vectors);
}

TEST_F(CudfFilterProjectTest, orOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testOrOperation(vectors);
}

TEST_F(CudfFilterProjectTest, lengthFunction) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testLengthFunction(vectors);
}

TEST_F(CudfFilterProjectTest, yearFunction) {
  // Update row type to use TIMESTAMP directly
  auto rowType =
      ROW({{"c0", INTEGER()}, {"c1", DOUBLE()}, {"c2", TIMESTAMP()}});

  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType, 2, batchSize);

  // Set timestamp values directly
  for (auto& vector : vectors) {
    auto timestampVector = vector->childAt(2)->asFlatVector<Timestamp>();
    for (vector_size_t i = 0; i < batchSize; ++i) {
      // Set to 2024-03-14 12:34:56
      Timestamp ts(1710415496, 0); // seconds, nanos
      timestampVector->set(i, ts);
    }
  }

  createDuckDbTable(vectors);
  testYearFunction(vectors);
}

} // namespace
