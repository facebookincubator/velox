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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

namespace {

template <typename T>
T getColValue(const std::vector<RowVectorPtr>& input, int col, int32_t index) {
  return input[0]->as<RowVector>()->childAt(col)->as<FlatVector<T>>()->valueAt(
      index);
}

class CudfFilterProjectTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
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

  void testLogicalShortCircuitWithLiterals(
      const std::vector<RowVectorPtr>& input) {
    // Constant false as first conjunct: LogicalFunction short-circuits to
    // false.
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"false AND (c0 = 1) AS result"})
                    .planNode();
    runTest(plan, "SELECT false AND (c0 = 1) AS result FROM tmp");

    // Constant true as first disjunct: LogicalFunction short-circuits to true.
    plan = PlanBuilder()
               .values(input)
               .project({"true OR (c0 = 1) AS result"})
               .planNode();
    runTest(plan, "SELECT true OR (c0 = 1) AS result FROM tmp");
  }

  void testLogicalAndOrLiteralsAndMixed(
      const std::vector<RowVectorPtr>& input) {
    // Literal-only (exercises LogicalFunction scalar/scalar and broadcast
    // paths).
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"true AND false AS r1", "true OR false AS r2"})
                    .planNode();
    runTest(plan, "SELECT true AND false AS r1, true OR false AS r2 FROM tmp");

    plan = PlanBuilder()
               .values(input)
               .project(
                   {"false AND true AS r1",
                    "false OR true AS r2",
                    "true AND true AS r3",
                    "false OR false AS r4"})
               .planNode();
    runTest(
        plan,
        "SELECT false AND true AS r1, false OR true AS r2, true AND true AS r3, false OR false AS r4 FROM tmp");

    // Literal on the left or right of a column predicate.
    plan = PlanBuilder()
               .values(input)
               .project(
                   {"(c0 = 1) AND true AS r1",
                    "true AND (c0 = 1) AS r2",
                    "(c0 = 1) OR false AS r3",
                    "false OR (c0 = 1) AS r4"})
               .planNode();
    runTest(
        plan,
        "SELECT (c0 = 1) AND true AS r1, true AND (c0 = 1) AS r2, (c0 = 1) OR false AS r3, false OR (c0 = 1) AS r4 FROM tmp");

    // Three-way mix: literals and columns interleaved.
    plan = PlanBuilder()
               .values(input)
               .project(
                   {"(c0 = 1) AND true AND (c1 = 2.0) AS r1",
                    "false OR (c0 = 1) OR (c1 = 2.0) AS r2"})
               .planNode();
    runTest(
        plan,
        "SELECT (c0 = 1) AND true AND (c1 = 2.0) AS r1, false OR (c0 = 1) OR (c1 = 2.0) AS r2 FROM tmp");
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

  void testCaseWhenOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with a CASE WHEN operation
    auto plan =
        PlanBuilder()
            .values(input)
            .project({"CASE WHEN c0 = 0 THEN 1.0 ELSE 0.0 END AS result"})
            .planNode();

    // Run the test
    runTest(
        plan,
        "SELECT CASE WHEN c0 = 0 THEN 1.0 ELSE 0.0 END AS result FROM tmp");
  }

  void testSubstrOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with a substr operation
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"substr(c2, 1, 3) AS result"})
                    .planNode();

    // Run the test
    runTest(plan, "SELECT substr(c2, 1, 3) AS result FROM tmp");
  }

  void testLikeOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with a like operation
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"c2 LIKE '%test%' AS result"})
                    .planNode();

    // Run the test
    runTest(plan, "SELECT c2 LIKE '%test%' AS result FROM tmp");
  }

  void testAddOperation(const std::vector<RowVectorPtr>& input) {
    auto plan =
        PlanBuilder().values(input).project({"c0 + c1 AS result"}).planNode();
    runTest(plan, "SELECT c0 + c1 AS result FROM tmp");

    plan =
        PlanBuilder().values(input).project({"1.0 + c1 AS result"}).planNode();
    runTest(plan, "SELECT 1.0 + c1 AS result FROM tmp");
  }

  void testSubtractOperation(const std::vector<RowVectorPtr>& input) {
    auto plan =
        PlanBuilder().values(input).project({"c0 - c1 AS result"}).planNode();
    runTest(plan, "SELECT c0 - c1 AS result FROM tmp");

    plan =
        PlanBuilder().values(input).project({"c1 - 1.0 AS result"}).planNode();
    runTest(plan, "SELECT c1 - 1.0 AS result FROM tmp");
  }

  void testMultiplyColumnColumn(const std::vector<RowVectorPtr>& input) {
    auto plan =
        PlanBuilder().values(input).project({"c0 * c1 AS result"}).planNode();
    runTest(plan, "SELECT c0 * c1 AS result FROM tmp");

    plan =
        PlanBuilder().values(input).project({"c1 * 2.0 AS result"}).planNode();
    runTest(plan, "SELECT c1 * 2.0 AS result FROM tmp");
  }

  void testDivideScalarVariants(const std::vector<RowVectorPtr>& input) {
    auto plan =
        PlanBuilder().values(input).project({"c1 / 2.0 AS result"}).planNode();
    runTest(plan, "SELECT c1 / 2.0 AS result FROM tmp");

    plan = PlanBuilder()
               .values(input)
               .project({"100.0 / c1 AS result"})
               .planNode();
    runTest(plan, "SELECT 100.0 / c1 AS result FROM tmp");
  }

  void testModuloOperation(const std::vector<RowVectorPtr>& input) {
    auto plan =
        PlanBuilder().values(input).project({"c0 % 10 AS result"}).planNode();
    runTest(plan, "SELECT c0 % 10 AS result FROM tmp");
  }

  void testLessThanOperation(const std::vector<RowVectorPtr>& input) {
    auto plan =
        PlanBuilder().values(input).project({"c0 < c1 AS result"}).planNode();
    runTest(plan, "SELECT c0 < c1 AS result FROM tmp");

    plan = PlanBuilder().values(input).project({"c0 < 1 AS result"}).planNode();
    runTest(plan, "SELECT c0 < 1 AS result FROM tmp");

    plan = PlanBuilder().values(input).project({"1 < c0 AS result"}).planNode();
    runTest(plan, "SELECT 1 < c0 AS result FROM tmp");
  }

  void testGreaterThanOperation(const std::vector<RowVectorPtr>& input) {
    auto plan =
        PlanBuilder().values(input).project({"c0 > c1 AS result"}).planNode();
    runTest(plan, "SELECT c0 > c1 AS result FROM tmp");

    plan = PlanBuilder().values(input).project({"c0 > 1 AS result"}).planNode();
    runTest(plan, "SELECT c0 > 1 AS result FROM tmp");

    plan = PlanBuilder().values(input).project({"1 > c0 AS result"}).planNode();
    runTest(plan, "SELECT 1 > c0 AS result FROM tmp");
  }

  void testLessThanEqualOperation(const std::vector<RowVectorPtr>& input) {
    auto plan =
        PlanBuilder().values(input).project({"c0 <= c1 AS result"}).planNode();
    runTest(plan, "SELECT c0 <= c1 AS result FROM tmp");

    plan =
        PlanBuilder().values(input).project({"c0 <= 1 AS result"}).planNode();
    runTest(plan, "SELECT c0 <= 1 AS result FROM tmp");

    plan =
        PlanBuilder().values(input).project({"1 <= c0 AS result"}).planNode();
    runTest(plan, "SELECT 1 <= c0 AS result FROM tmp");
  }

  void testGreaterThanEqualOperation(const std::vector<RowVectorPtr>& input) {
    auto plan =
        PlanBuilder().values(input).project({"c0 >= c1 AS result"}).planNode();
    runTest(plan, "SELECT c0 >= c1 AS result FROM tmp");

    plan =
        PlanBuilder().values(input).project({"c0 >= 1 AS result"}).planNode();
    runTest(plan, "SELECT c0 >= 1 AS result FROM tmp");

    plan =
        PlanBuilder().values(input).project({"1 >= c0 AS result"}).planNode();
    runTest(plan, "SELECT 1 >= c0 AS result FROM tmp");
  }

  void testEqualScalarLeft(const std::vector<RowVectorPtr>& input) {
    auto plan =
        PlanBuilder().values(input).project({"1 = c0 AS result"}).planNode();
    runTest(plan, "SELECT 1 = c0 AS result FROM tmp");
  }

  void testNotEqualScalarLeft(const std::vector<RowVectorPtr>& input) {
    auto plan =
        PlanBuilder().values(input).project({"1 <> c0 AS result"}).planNode();
    runTest(plan, "SELECT 1 <> c0 AS result FROM tmp");
  }

  void testNotOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with a NOT operation
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"NOT (c0 = 1) AS result"})
                    .planNode();

    // Run the test
    runTest(plan, "SELECT NOT (c0 = 1) AS result FROM tmp");
  }

  void testBetweenOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with a BETWEEN operation
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"c0 BETWEEN 1 AND 100 AS result"})
                    .planNode();

    // Run the test
    runTest(plan, "SELECT c0 BETWEEN 1 AND 100 AS result FROM tmp");
  }

  void testMultiInputAndOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with multiple AND operations
    auto c2Value = getColValue<StringView>(input, 2, 1).str();
    auto plan = PlanBuilder()
                    .values(input)
                    .project(
                        {"c0 > 1000 AND c0 < 20000 AND c2 = '" + c2Value +
                         "' AS result"})
                    .planNode();

    // Run the test
    runTest(
        plan,
        "SELECT c0 > 1000 AND c0 < 20000 AND c2 = '" + c2Value +
            "' AS result FROM tmp");
  }

  void testMultiInputOrOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with multiple OR operations
    auto c2Value = getColValue<StringView>(input, 2, 1).str();
    auto plan = PlanBuilder()
                    .values(input)
                    .project(
                        {"c0 > 16000 OR c0 < 8000 OR c1 = 2.0 OR c2 = '" +
                         c2Value + "' AS result"})
                    .planNode();

    // Run the test
    runTest(
        plan,
        "SELECT c0 > 16000 OR c0 < 8000 OR c1 = 2.0 OR c2 = '" + c2Value +
            "' AS result FROM tmp");
  }

  void testIntegerInOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with an IN operation for integers
    std::vector<int32_t> c0Values;
    for (int32_t i = 0; i < 5; i++) {
      c0Values.push_back(getColValue<int32_t>(input, 0, i));
    }
    std::string c0ValuesStr;
    for (size_t i = 0; i < c0Values.size(); ++i) {
      c0ValuesStr += std::to_string(c0Values[i]) + ",";
    }
    c0ValuesStr.pop_back();
    auto plan = PlanBuilder(pool_.get())
                    .values(input)
                    .project({"c0 IN (" + c0ValuesStr + ") AS result"})
                    .planNode();

    // Run the test
    runTest(plan, "SELECT c0 IN (" + c0ValuesStr + ") AS result FROM tmp");
  }

  void testDoubleInOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with an IN operation for doubles
    std::vector<double> c1Values;
    for (int32_t i = 0; i < 4; i++) {
      c1Values.push_back(getColValue<double>(input, 1, i));
    }
    std::string c1ValuesStr;
    for (size_t i = 0; i < c1Values.size(); ++i) {
      c1ValuesStr += std::to_string(c1Values[i]) + ",";
    }
    c1ValuesStr.pop_back();
    auto plan = PlanBuilder(pool_.get())
                    .values(input)
                    .project({"c1 IN (" + c1ValuesStr + ") AS result"})
                    .planNode();

    // Run the test
    runTest(plan, "SELECT c1 IN (" + c1ValuesStr + ") AS result FROM tmp");
  }

  void testStringInOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan with an IN operation for strings
    std::vector<StringView> c2Values;
    for (int32_t i = 0; i < 3; i++) {
      c2Values.push_back(getColValue<StringView>(input, 2, i));
    }
    std::string c2ValuesStr;
    for (size_t i = 0; i < c2Values.size(); ++i) {
      c2ValuesStr += "'" + c2Values[i].str() + "',";
    }
    c2ValuesStr.pop_back();
    auto plan = PlanBuilder(pool_.get())
                    .values(input)
                    .project({"c2 IN (" + c2ValuesStr + ") AS result"})
                    .planNode();

    // Run the test
    runTest(plan, "SELECT c2 IN (" + c2ValuesStr + ") AS result FROM tmp");
  }

  void testMixedInOperation(const std::vector<RowVectorPtr>& input) {
    // Create a plan that combines multiple IN operations
    auto plan =
        PlanBuilder(pool_.get())
            .values(input)
            .project(
                {"c0 IN (1, 2, 3) OR c1 IN (1.5, 2.5) OR c2 IN ('test1', 'test2') AS result"})
            .planNode();

    // Run the test
    runTest(
        plan,
        "SELECT c0 IN (1, 2, 3) OR c1 IN (1.5, 2.5) OR c2 IN ('test1', 'test2') AS result FROM tmp");
  }

  void testStringLiteralExpansion(const std::vector<RowVectorPtr>& input) {
    // Test VARCHAR literal as standalone expression (needs special handling)
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"'literal_value' AS result"})
                    .planNode();

    runTest(plan, "SELECT 'literal_value' AS result FROM tmp");
  }

  void testStringLiteralInComparison(const std::vector<RowVectorPtr>& input) {
    // Test VARCHAR literal within comparison (should work normally)
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"c2 = 'test_value' AS result"})
                    .planNode();

    runTest(plan, "SELECT c2 = 'test_value' AS result FROM tmp");
  }

  void testStringLowerOperation(const std::vector<RowVectorPtr>& input) {
    // Test VARCHAR lower
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"LOWER(c2) = 'test_value' AS result"})
                    .planNode();

    runTest(plan, "SELECT LOWER(c2) = 'test_value' AS result FROM tmp");
  }

  void testStringUpperOperation(const std::vector<RowVectorPtr>& input) {
    // Test VARCHAR upper
    auto plan = PlanBuilder()
                    .values(input)
                    .project({"UPPER(c2) = 'TEST_VALUE' AS result"})
                    .planNode();

    runTest(plan, "SELECT UPPER(c2) = 'TEST_VALUE' AS result FROM tmp");
  }

  void testMixedLiteralProjection(const std::vector<RowVectorPtr>& input) {
    // Test mixing standalone literals with expressions containing literals
    auto plan = PlanBuilder()
                    .values(input)
                    .project(
                        {"'standalone_string' AS str_literal",
                         "42 AS int_literal",
                         "c2 = 'comparison_string' AS bool_result"})
                    .planNode();

    runTest(
        plan,
        "SELECT 'standalone_string' AS str_literal, 42 AS int_literal, c2 = 'comparison_string' AS bool_result FROM tmp");
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

TEST_F(CudfFilterProjectTest, addOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testAddOperation(vectors);
}

TEST_F(CudfFilterProjectTest, subtractOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testSubtractOperation(vectors);
}

TEST_F(CudfFilterProjectTest, multiplyColumnColumn) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testMultiplyColumnColumn(vectors);
}

TEST_F(CudfFilterProjectTest, divideScalarVariants) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testDivideScalarVariants(vectors);
}

TEST_F(CudfFilterProjectTest, moduloOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testModuloOperation(vectors);
}

TEST_F(CudfFilterProjectTest, equalScalarLeft) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testEqualScalarLeft(vectors);
}

TEST_F(CudfFilterProjectTest, notEqualScalarLeft) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testNotEqualScalarLeft(vectors);
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

TEST_F(CudfFilterProjectTest, logicalShortCircuitWithLiterals) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testLogicalShortCircuitWithLiterals(vectors);
}

TEST_F(CudfFilterProjectTest, logicalAndOrLiteralsAndMixed) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testLogicalAndOrLiteralsAndMixed(vectors);
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

// The result mismatches.
TEST_F(CudfFilterProjectTest, caseWhenOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testCaseWhenOperation(vectors);
}

TEST_F(CudfFilterProjectTest, substrOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testSubstrOperation(vectors);
}

TEST_F(CudfFilterProjectTest, likeOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testLikeOperation(vectors);
}

TEST_F(CudfFilterProjectTest, lessThanOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testLessThanOperation(vectors);
}

TEST_F(CudfFilterProjectTest, greaterThanOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testGreaterThanOperation(vectors);
}

TEST_F(CudfFilterProjectTest, lessThanEqualOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testLessThanEqualOperation(vectors);
}

TEST_F(CudfFilterProjectTest, greaterThanEqualOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testGreaterThanEqualOperation(vectors);
}

TEST_F(CudfFilterProjectTest, notOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testNotOperation(vectors);
}

TEST_F(CudfFilterProjectTest, betweenOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testBetweenOperation(vectors);
}

TEST_F(CudfFilterProjectTest, multiInputAndOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testMultiInputAndOperation(vectors);
}

TEST_F(CudfFilterProjectTest, multiInputOrOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testMultiInputOrOperation(vectors);
}

TEST_F(CudfFilterProjectTest, integerInOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testIntegerInOperation(vectors);
}

TEST_F(CudfFilterProjectTest, doubleInOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testDoubleInOperation(vectors);
}

TEST_F(CudfFilterProjectTest, stringInOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testStringInOperation(vectors);
}

TEST_F(CudfFilterProjectTest, mixedInOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testMixedInOperation(vectors);
}

TEST_F(CudfFilterProjectTest, round) {
  auto data = makeRowVector({makeFlatVector<int64_t>({4123, 456789098})});
  parse::ParseOptions options;
  options.parseIntegerAsBigint = false;
  auto plan = PlanBuilder()
                  .setParseOptions(options)
                  .values({data})
                  .project({"round(c0, 2) as c1"})
                  .planNode();
  AssertQueryBuilder(plan).assertResults(data);
  plan = PlanBuilder()
             .setParseOptions(options)
             .values({data})
             .project({"round(c0) as c1"})
             .planNode();
  AssertQueryBuilder(plan).assertResults(data);

  plan = PlanBuilder()
             .setParseOptions(options)
             .values({data})
             .project({"round(c0, -3) as c1"})
             .planNode();
  auto expected = makeRowVector({makeFlatVector<int64_t>({4000, 456789000})});
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, roundDecimal) {
  parse::ParseOptions options;
  options.parseIntegerAsBigint = false;

  // Note that the underlying cudf::round_decimal function returns
  // a value with the specified scale, and rounds the internal integer
  // value accordingly, e.g. rounding 41.2389 to 2 decimal places
  // results in an internal integer value of 4124 with a scale of 2.
  //
  // When the specified scale is non-zero, Velox inserts extra casts
  // to restore the original scale. However, when the specified scale
  // is zero, Velox does NOT do that, and the result has a scale of 0.

  // Input values 41.2389 and -45.6789 as DECIMAL(10, 4).
  auto decimalData = makeRowVector(
      {makeFlatVector<int64_t>({412389, -456789}, DECIMAL(10, 4))});

  // Round to 2 decimal places.
  // Expected values are 41.24 and -45.68 as DECIMAL(10, 4).
  auto plan = PlanBuilder()
                  .setParseOptions(options)
                  .values({decimalData})
                  .project({"round(c0, 2) as c1"})
                  .planNode();
  auto decimalExpected = makeRowVector(
      {makeFlatVector<int64_t>({412400, -456800}, DECIMAL(10, 4))});
  AssertQueryBuilder(plan).assertResults(decimalExpected);

  // Round to 0 decimal places.
  // Expected values are 41.0 and -46.0 as DECIMAL(10, 0).
  plan = PlanBuilder()
             .setParseOptions(options)
             .values({decimalData})
             .project({"round(c0) as c1"})
             .planNode();
  decimalExpected =
      makeRowVector({makeFlatVector<int64_t>({41, -46}, DECIMAL(10, 0))});
  AssertQueryBuilder(plan).assertResults(decimalExpected);

  // Round to -1 decimal places.
  // Expected values are 40.0 and -50.0 as DECIMAL(10, 4).
  plan = PlanBuilder()
             .setParseOptions(options)
             .values({decimalData})
             .project({"round(c0, -1) as c1"})
             .planNode();
  decimalExpected = makeRowVector(
      {makeFlatVector<int64_t>({400000, -500000}, DECIMAL(10, 4))});
  AssertQueryBuilder(plan).assertResults(decimalExpected);
}

TEST_F(CudfFilterProjectTest, simpleFilter) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  // Create a plan with a simple filter
  auto plan = PlanBuilder()
                  .values(vectors)
                  .filter("c0 > 500")
                  .project({"c0", "c1", "c2"})
                  .planNode();

  // Run the test
  assertQuery(plan, "SELECT c0, c1, c2 FROM tmp WHERE c0 > 500");
}

TEST_F(CudfFilterProjectTest, filterWithProject) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  // Create a plan with filter and project
  auto plan =
      PlanBuilder()
          .values(vectors)
          .filter("c0 > 500")
          .project({"c0 + 2 as doubled", "c1 + 1.0 as incremented", "c2"})
          .planNode();

  // Run the test
  assertQuery(
      plan,
      "SELECT c0 + 2 as doubled, c1 + 1.0 as incremented, c2 FROM tmp WHERE c0 > 500");
}

TEST_F(CudfFilterProjectTest, complexFilter) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  // Create a plan with a complex filter condition
  auto plan = PlanBuilder()
                  .values(vectors)
                  .filter("c0 > 500 AND c1 < 0.5 AND c2 LIKE '%test%'")
                  .project({"c0", "c1", "c2"})
                  .planNode();

  // Run the test
  assertQuery(
      plan,
      "SELECT c0, c1, c2 FROM tmp WHERE c0 > 500 AND c1 < 0.5 AND c2 LIKE '%test%'");
}

TEST_F(CudfFilterProjectTest, filterWithNullValues) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);

  // Add some null values to the vectors
  for (auto& vector : vectors) {
    auto c0Vector = vector->childAt(0)->asFlatVector<int32_t>();
    auto c1Vector = vector->childAt(1)->asFlatVector<double>();
    for (vector_size_t i = 0; i < batchSize; i += 10) {
      c0Vector->setNull(i, true);
      c1Vector->setNull(i, true);
    }
  }

  createDuckDbTable(vectors);

  // Create a plan with filter that handles null values
  auto plan = PlanBuilder()
                  .values(vectors)
                  .filter("c0 IS NOT NULL AND c1 IS NOT NULL")
                  .project({"c0", "c1", "c2"})
                  .planNode();

  // Run the test
  assertQuery(
      plan,
      "SELECT c0, c1, c2 FROM tmp WHERE c0 IS NOT NULL AND c1 IS NOT NULL");
}

TEST_F(CudfFilterProjectTest, filterWithOrCondition) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  // Create a plan with OR condition in filter
  auto plan = PlanBuilder()
                  .values(vectors)
                  .filter("c0 > 500 OR c1 < 0.5")
                  .project({"c0", "c1", "c2"})
                  .planNode();

  // Run the test
  assertQuery(plan, "SELECT c0, c1, c2 FROM tmp WHERE c0 > 500 OR c1 < 0.5");
}

TEST_F(CudfFilterProjectTest, filterWithInCondition) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  // Create a plan with IN condition in filter
  auto plan = PlanBuilder(pool_.get())
                  .values(vectors)
                  .filter("c0 IN (100, 200, 300, 400, 500)")
                  .project({"c0", "c1", "c2"})
                  .planNode();

  // Run the test
  assertQuery(
      plan, "SELECT c0, c1, c2 FROM tmp WHERE c0 IN (100, 200, 300, 400, 500)");
}

TEST_F(CudfFilterProjectTest, filterWithBetweenCondition) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  // Create a plan with BETWEEN condition in filter
  auto plan = PlanBuilder()
                  .values(vectors)
                  .filter("c0 BETWEEN 100 AND 500")
                  .project({"c0", "c1", "c2"})
                  .planNode();

  // Run the test
  assertQuery(plan, "SELECT c0, c1, c2 FROM tmp WHERE c0 BETWEEN 100 AND 500");
}

TEST_F(CudfFilterProjectTest, filterWithStringOperations) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  // Create a plan with string operations in filter
  auto plan = PlanBuilder()
                  .values(vectors)
                  .filter("LENGTH(c2) > 5")
                  .project({"c0", "c1", "c2"})
                  .planNode();

  // Run the test
  assertQuery(plan, "SELECT c0, c1, c2 FROM tmp WHERE LENGTH(c2) > 5");
}

TEST_F(CudfFilterProjectTest, filterWithoutProject) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  // Create a plan with only filter (no projection)
  auto plan =
      PlanBuilder().values(vectors).filter("c0 > 500 AND c1 < 0.5").planNode();

  // Run the test - should return all columns without modification
  assertQuery(plan, "SELECT c0, c1, c2 FROM tmp WHERE c0 > 500 AND c1 < 0.5");
}

TEST_F(CudfFilterProjectTest, filterWithEmptyResult) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  // Create a plan with a filter that should return no rows
  auto plan = PlanBuilder()
                  .values(vectors)
                  .filter("c0 < 0 AND c0 > 1000") // Impossible condition
                  .planNode();

  // Run the test - should return empty result
  assertQuery(plan, "SELECT c0, c1, c2 FROM tmp WHERE c0 < 0 AND c0 > 1000");
}

TEST_F(CudfFilterProjectTest, stringLiteralExpansion) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testStringLiteralExpansion(vectors);
}

TEST_F(CudfFilterProjectTest, stringLiteralInComparison) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testStringLiteralInComparison(vectors);
}

TEST_F(CudfFilterProjectTest, stringLowerOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testStringLowerOperation(vectors);
}

TEST_F(CudfFilterProjectTest, stringUpperOperation) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testStringUpperOperation(vectors);
}

TEST_F(CudfFilterProjectTest, mixedLiteralProjection) {
  vector_size_t batchSize = 1000;
  auto vectors = makeVectors(rowType_, 2, batchSize);
  createDuckDbTable(vectors);

  testMixedLiteralProjection(vectors);
}

// This test checks for CudfExpression's ability to handle nested field
// references. However, to test this, we need to disable CPU fallback, otherwise
// the test could pass by using CPU, having not exercised the CudfExpression at
// all. But this test relies on row_constructor to construct the nested fields,
// which CudfExpression doesn't support. Disabling this until we have a better
// test
TEST_F(CudfFilterProjectTest, DISABLED_dereference) {
  auto rowType = ROW(
      {"c0", "c1", "c2", "c3"}, {BIGINT(), INTEGER(), SMALLINT(), DOUBLE()});
  auto vectors = makeVectors(rowType, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"row_constructor(c1, c2) AS c1_c2"})
                  .project({"c1_c2.c1", "c1_c2.c2"})
                  .planNode();
  assertQuery(plan, "SELECT c1, c2 FROM tmp");

  plan = PlanBuilder()
             .values(vectors)
             .project({"row_constructor(c1, c2) AS c1_c2"})
             .filter("c1_c2.c1 % 10 = 5")
             .project({"c1_c2.c1", "c1_c2.c2"})
             .planNode();
  assertQuery(plan, "SELECT c1, c2 FROM tmp WHERE c1 % 10 = 5");
}

TEST_F(CudfFilterProjectTest, cardinality) {
  auto input = makeArrayVector<int64_t>({{1, 2, 3}, {1, 2}, {}});
  auto data = makeRowVector({input});
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"cardinality(c0) AS result"})
                  .planNode();
  auto expected = makeRowVector({makeFlatVector<int64_t>({3, 2, 0})});
  AssertQueryBuilder(plan).assertResults({expected});
}

TEST_F(CudfFilterProjectTest, split) {
  auto input = makeFlatVector<std::string>(
      {"hello world", "hello world2", "hello hello"});
  auto data = makeRowVector({input});
  createDuckDbTable({data});
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"split(c0, 'hello', 2) AS result"})
                  .planNode();
  auto splitResults = AssertQueryBuilder(plan).copyResults(pool());

  auto calculatedSplitResults = makeRowVector({
      makeArrayVector<std::string>({
          {"", " world"},
          {"", " world2"},
          {"", " hello"},
      }),
  });
  facebook::velox::test::assertEqualVectors(
      splitResults, calculatedSplitResults);
}

TEST_F(CudfFilterProjectTest, cardinalityAndSplitOneByOne) {
  auto input = makeFlatVector<std::string>(
      {"hello world", "hello world2", "hello hello", "does not contain it"});
  auto data = makeRowVector({input});
  auto splitPlan = PlanBuilder()
                       .values({data})
                       .project({"split(c0, 'hello', 2) AS c0"})
                       .planNode();
  auto splitResults = AssertQueryBuilder(splitPlan).copyResults(pool());

  auto calculatedSplitResults = makeRowVector({
      makeArrayVector<std::string>({
          {"", " world"},
          {"", " world2"},
          {"", " hello"},
          {"does not contain it"},
      }),
  });
  facebook::velox::test::assertEqualVectors(
      splitResults, calculatedSplitResults);
  auto cardinalityPlan = PlanBuilder()
                             .values({splitResults})
                             .project({"cardinality(c0) AS result"})
                             .planNode();
  auto expected = makeRowVector({makeFlatVector<int64_t>({2, 2, 2, 1})});
  AssertQueryBuilder(cardinalityPlan).assertResults({expected});
}

// TODO: Requires a fix for the expression evaluator to handle function nesting.
TEST_F(CudfFilterProjectTest, cardinalityAndSplitFused) {
  auto input = makeFlatVector<std::string>(
      {"hello world", "hello world2", "hello hello", "does not contain it"});
  auto data = makeRowVector({input});
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"cardinality(split(c0, 'hello', 2)) AS c0"})
                  .planNode();
  auto expected = makeRowVector({makeFlatVector<int64_t>({2, 2, 2, 1})});
  AssertQueryBuilder(plan).assertResults({expected});
}

TEST_F(CudfFilterProjectTest, negativeSubstr) {
  auto input =
      makeFlatVector<std::string>({"hellobutlonghello", "secondstring"});
  auto data = makeRowVector({input});
  auto negativeSubstrPlan =
      PlanBuilder().values({data}).project({"substr(c0, -2) AS c0"}).planNode();
  auto negativeSubstrResults =
      AssertQueryBuilder(negativeSubstrPlan).copyResults(pool());

  auto calculatedNegativeSubstrResults = makeRowVector({
      makeFlatVector<std::string>({
          "lo",
          "ng",
      }),
  });
  facebook::velox::test::assertEqualVectors(
      negativeSubstrResults, calculatedNegativeSubstrResults);
}

TEST_F(CudfFilterProjectTest, negativeSubstrWithLength) {
  auto input =
      makeFlatVector<std::string>({"hellobutlonghello", "secondstring"});
  auto data = makeRowVector({input});
  auto negativeSubstrWithLengthPlan = PlanBuilder()
                                          .values({data})
                                          .project({"substr(c0, -6, 3) AS c0"})
                                          .planNode();
  auto negativeSubstrWithLengthResults =
      AssertQueryBuilder(negativeSubstrWithLengthPlan).copyResults(pool());

  auto calculatedNegativeSubstrWithLengthResults = makeRowVector({
      makeFlatVector<std::string>({
          "ghe",
          "str",
      }),
  });
  facebook::velox::test::assertEqualVectors(
      negativeSubstrWithLengthResults,
      calculatedNegativeSubstrWithLengthResults);
}

TEST_F(CudfFilterProjectTest, substrWithLength) {
  auto input =
      makeFlatVector<std::string>({"hellobutlonghello", "secondstring"});
  auto data = makeRowVector({input});
  auto substrPlan = PlanBuilder()
                        .values({data})
                        .project({"substr(c0, 1, 3) AS c0"})
                        .planNode();
  auto substrResults = AssertQueryBuilder(substrPlan).copyResults(pool());

  auto calculatedSubstrResults = makeRowVector({
      makeFlatVector<std::string>({
          "hel",
          "sec",
      }),
  });
  facebook::velox::test::assertEqualVectors(
      substrResults, calculatedSubstrResults);
}

TEST_F(CudfFilterProjectTest, coalesceColumnWithLiteral) {
  vector_size_t batchSize = 100;
  auto vectors = makeVectors(rowType_, 1, batchSize);
  // Introduce nulls into c0 to exercise replacement
  auto& vec = vectors[0];
  auto c0Vector = vec->childAt(0)->asFlatVector<int32_t>();
  for (vector_size_t i = 0; i < batchSize; i += 3) {
    c0Vector->setNull(i, true);
  }

  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"coalesce(c0, 5::INTEGER) AS result"})
                  .planNode();

  runTest(plan, "SELECT coalesce(c0, 5) AS result FROM tmp");
}

TEST_F(CudfFilterProjectTest, coalesceLiteralFirst) {
  vector_size_t batchSize = 100;
  auto vectors = makeVectors(rowType_, 1, batchSize);
  createDuckDbTable(vectors);

  // Literal appears before any column; result should be a constant column.
  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"coalesce(5::INTEGER, c0) AS result"})
                  .planNode();

  runTest(plan, "SELECT coalesce(5, c0) AS result FROM tmp");
}

TEST_F(CudfFilterProjectTest, coalesceStopsAtFirstLiteral) {
  // Use two integer columns to validate that columns after the literal are
  // ignored.
  auto rowType = ROW({{"c0", INTEGER()}, {"c1", INTEGER()}});
  auto vectors = makeVectors(rowType, 1, 50);
  // Make some c0 nulls so fallback engages.
  auto& vec = vectors[0];
  auto c0 = vec->childAt(0)->asFlatVector<int32_t>();
  for (vector_size_t i = 1; i < vec->size(); i += 4) {
    c0->setNull(i, true);
  }

  createDuckDbTable(vectors);

  // With expression coalesce(c0, 100, c1), rows with null c0 should become 100,
  // regardless of c1. This also verifies we stop at the first literal.
  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"coalesce(c0, 100::INTEGER, c1) AS result"})
                  .planNode();

  runTest(plan, "SELECT coalesce(c0, 100, c1) AS result FROM tmp");
}

TEST_F(CudfFilterProjectTest, switchExpr) {
  auto data = makeRowVector(
      {makeFlatVector<double>({45676567.78, 6789098767.90876, -2.34}),
       makeFlatVector<double>({123.4, 124.5, 1678})});
  auto plan =
      PlanBuilder()
          .values({data})
          .project(
              {"CASE WHEN c0 > 0.0 THEN c0 / c1 ELSE cast(null as double) END AS result"})
          .planNode();
  auto result = AssertQueryBuilder(plan).copyResults(pool());

  auto expected = makeRowVector({
      makeNullableFlatVector<double>(
          {45676567.78 / 123.4, 6789098767.90876 / 124.5, std::nullopt}),
  });
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfFilterProjectTest, greatestLeastAllColumns) {
  auto data = makeRowVector({
      makeFlatVector<double>({1.0, 5.0, -3.0}),
      makeFlatVector<double>({4.0, 2.0, -1.0}),
      makeFlatVector<double>({3.0, 3.0, -2.0}),
  });
  createDuckDbTable({data});
  auto plan =
      PlanBuilder()
          .values({data})
          .project({"greatest(c0, c1, c2) AS g", "least(c0, c1, c2) AS l"})
          .planNode();
  assertQuery(plan, "SELECT greatest(c0, c1, c2), least(c0, c1, c2) FROM tmp");
}

TEST_F(CudfFilterProjectTest, greatestLeastMixed) {
  auto data = makeRowVector({
      makeFlatVector<double>({1.0, 10.0, -3.0}),
      makeFlatVector<double>({4.0, 2.0, -1.0}),
  });
  createDuckDbTable({data});
  auto plan =
      PlanBuilder()
          .values({data})
          .project({"greatest(c0, 5.0, c1) AS g", "least(c0, 0.0, c1) AS l"})
          .planNode();
  assertQuery(
      plan, "SELECT greatest(c0, 5.0, c1), least(c0, 0.0, c1) FROM tmp");
}

TEST_F(CudfFilterProjectTest, greatestLeastAllLiterals) {
  auto data = makeRowVector({makeFlatVector<double>({0.0})});
  auto plan =
      PlanBuilder()
          .values({data})
          .project(
              {"greatest(1.0, 3.0, 2.0) AS g", "least(1.0, 3.0, 2.0) AS l"})
          .planNode();
  auto expected = makeRowVector({
      makeFlatVector<double>({3.0}),
      makeFlatVector<double>({1.0}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, greatestLeastWithNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<double>({1.0, std::nullopt, std::nullopt}),
      makeNullableFlatVector<double>({std::nullopt, 2.0, std::nullopt}),
      makeNullableFlatVector<double>({3.0, std::nullopt, std::nullopt}),
  });
  createDuckDbTable({data});
  auto plan =
      PlanBuilder()
          .values({data})
          .project({"greatest(c0, c1, c2) AS g", "least(c0, c1, c2) AS l"})
          .planNode();
  assertQuery(plan, "SELECT greatest(c0, c1, c2), least(c0, c1, c2) FROM tmp");
}

TEST_F(CudfFilterProjectTest, betweenDouble) {
  auto data = makeRowVector({
      makeFlatVector<double>({-1.0, 0.5, 1.5, 3.0}),
  });
  createDuckDbTable({data});
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"c0 BETWEEN 0.0 AND 2.0 AS result"})
                  .planNode();
  assertQuery(plan, "SELECT c0 BETWEEN 0.0 AND 2.0 AS result FROM tmp");
}

class CudfSimpleFilterProjectTest : public cudf_velox::CudfFunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    cudf_velox::registerCudf();
    cudf_velox::registerPrestoFunctions("");
  }

  static void TearDownTestCase() {
    cudf_velox::unregisterFunctions();
    cudf_velox::unregisterCudf();
  }

  template <typename T>
  ArrayVectorPtr makeArrayAccessNumericTestVector(
      const TypePtr& arrayType = ARRAY(CppToType<T>::create())) {
    return makeNullableArrayVector<T>(
        {
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)}},
            {{static_cast<T>(4), std::nullopt, static_cast<T>(6)}},
            std::nullopt,
            {{static_cast<T>(7), static_cast<T>(8)}},
        },
        arrayType);
  }

  void assertArrayAccessElementTypeMatchesCpu(
      const std::string& label,
      const ArrayVectorPtr& arrays) {
    SCOPED_TRACE(label);

    auto assertMatchesCpu = [&](const std::string& expression,
                                const RowVectorPtr& input) {
      SCOPED_TRACE(expression);
      assertExpressionMatchesCpu(expression, input, input->rowType());
    };

    // Presto array access currently registers INTEGER and BIGINT index
    // signatures. Exercise both variable-index paths for element_at and
    // subscript.
    auto integerIndices = makeFlatVector<int32_t>({2, 2, 1, 1});
    auto bigintIndices = makeFlatVector<int64_t>({2, 2, 1, 1});
    for (const auto& indices :
         std::vector<VectorPtr>{integerIndices, bigintIndices}) {
      auto input = makeRowVector({arrays, indices});
      assertMatchesCpu("element_at(c0, c1)", input);
      assertMatchesCpu("subscript(c0, c1)", input);
    }

    // Literal-index calls use the scalar-index cuDF extraction path. Null
    // literal indexes are handled separately from non-null literals.
    auto input = makeRowVector({arrays});
    for (const auto& expression : {
             "element_at(c0, cast(2 as integer))",
             "element_at(c0, cast(2 as bigint))",
             "subscript(c0, cast(2 as integer))",
             "subscript(c0, cast(2 as bigint))",
             "element_at(c0, cast(null as integer))",
             "element_at(c0, cast(null as bigint))",
             "subscript(c0, cast(null as integer))",
             "subscript(c0, cast(null as bigint))",
         }) {
      assertMatchesCpu(expression, input);
    }
  }

  void assertConstantArrayAccessMatchesCpu(
      const std::string& label,
      const std::string& arraySql) {
    SCOPED_TRACE(label);

    // A literal array is not passed as an input column to cuDF. These cases
    // cover the repeated-literal-array branch plus both Presto index types.
    auto integerIndices = makeFlatVector<int32_t>({1, 2, 3, 2});
    auto bigintIndices = makeFlatVector<int64_t>({1, 2, 3, 2});
    for (const auto& indices :
         std::vector<VectorPtr>{integerIndices, bigintIndices}) {
      auto input = makeRowVector({indices});
      for (const auto& functionName : {"element_at", "subscript"}) {
        auto expression = std::string(functionName) + "(" + arraySql + ", c0)";
        SCOPED_TRACE(expression);
        assertExpressionMatchesCpu(expression, input, input->rowType());
      }
    }

    // element_at is the Presto flavor that supports negative indexes from the
    // end of the array.
    auto negativeIndices = makeFlatVector<int64_t>({-1, -2, -3});
    auto input = makeRowVector({negativeIndices});
    auto expression = std::string("element_at(") + arraySql + ", c0)";
    SCOPED_TRACE(expression);
    assertExpressionMatchesCpu(expression, input, input->rowType());
  }
};

TEST_F(CudfSimpleFilterProjectTest, castToSmallInt) {
  auto castValue = evaluateOnce<int16_t, int32_t>("cast(c0 as smallint)", 12);
  EXPECT_EQ(castValue, 12);
  auto tryCast =
      evaluateOnce<int16_t, int32_t>("try_cast(c0 as smallint)", -214);
  EXPECT_EQ(tryCast, -214);
}

// Test unary math functions
TEST_F(CudfSimpleFilterProjectTest, unaryMathFunctions) {
  auto testUnaryFunction =
      [&](std::string expr, double input, double expected) {
        auto valueOpt = evaluateOnce<double, double>(expr, input);
        EXPECT_DOUBLE_EQ(valueOpt.value(), expected);
      };
  // Trigonometric functions (use 0.0 where stable/clean)
  testUnaryFunction("sin(c0)", 0.0, 0.0);
  testUnaryFunction("cos(c0)", 0.0, 1.0);
  testUnaryFunction("tan(c0)", 0.0, 0.0);

  // Inverse trig (use stable inputs)
  testUnaryFunction("asin(c0)", 0.0, 0.0);
  testUnaryFunction("acos(c0)", 1.0, 0.0);
  testUnaryFunction("atan(c0)", 0.0, 0.0);

  // Hyperbolic functions
  testUnaryFunction("cosh(c0)", 0.0, 1.0);
  testUnaryFunction("tanh(c0)", 0.0, 0.0);

  // Exponential / log / roots
  testUnaryFunction("exp(c0)", 0.0, 1.0);
  testUnaryFunction("ln(c0)", 1.0, 0.0);
  testUnaryFunction("sqrt(c0)", 4.0, 2.0);
  testUnaryFunction("cbrt(c0)", 8.0, 2.0);

  // Absolute value
  testUnaryFunction("abs(c0)", -5.5, 5.5);
}

TEST_F(CudfSimpleFilterProjectTest, elementAtArrayWithDictionaryElements) {
  auto elementsIndices = makeIndices({6, 5, 4, 3, 2, 1, 0});
  auto elements = wrapInDictionary(
      elementsIndices, makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6}));

  auto arrays = makeArrayVector({0, 3, 6}, elements);
  auto indices = makeFlatVector<int64_t>({3, -3, 1});
  auto input = makeRowVector({arrays, indices});

  assertExpressionMatchesCpu("element_at(c0, c1)", input, input->rowType());
}

TEST_F(CudfSimpleFilterProjectTest, elementAtArrayConstantInput) {
  auto arrays = makeArrayVector<int64_t>({
      {1, 2, 3, 4, 5, 6, 7},
      {1, 2, 7},
      {1, 2, 3, 5, 6, 7},
  });
  auto input = makeRowVector({arrays});

  assertExpressionMatchesCpu("element_at(c0, 2)", input, input->rowType());
  assertExpressionMatchesCpu("element_at(c0, -1)", input, input->rowType());

  auto variableSizeArrays = makeArrayVector<int64_t>({
      {1, 2, 6, 0},
      {1},
      {9, 100, 10043, 44, 22, 2},
      {-1, -10, 3},
  });
  auto variableSizeInput = makeRowVector({variableSizeArrays});

  assertExpressionMatchesCpu(
      "element_at(c0, -1)", variableSizeInput, variableSizeInput->rowType());
}

TEST_F(CudfSimpleFilterProjectTest, elementAtArrayVariableInput) {
  constexpr vector_size_t kArrayAccessVectorSize{1'000};

  auto fixedIndices = makeFlatVector<int64_t>(
      kArrayAccessVectorSize, [](vector_size_t row) { return 1 + row % 7; });

  auto fixedSizeAt = [](vector_size_t /*row*/) { return 7; };
  auto oneBasedValueAt = [](vector_size_t /*row*/, vector_size_t idx) {
    return 1 + idx;
  };
  auto fixedArrays = makeArrayVector<int64_t>(
      kArrayAccessVectorSize, fixedSizeAt, oneBasedValueAt);
  auto fixedInput = makeRowVector({fixedArrays, fixedIndices});

  assertExpressionMatchesCpu(
      "element_at(c0, c1)", fixedInput, fixedInput->rowType());

  auto mixedIndices =
      makeFlatVector<int64_t>(kArrayAccessVectorSize, [](vector_size_t row) {
        if (row % 7 == 0) {
          return 8;
        }
        if (row % 7 == 4) {
          return -3;
        }
        return 1 + row % 7;
      });

  auto variableSizeAt = [](vector_size_t row) { return row % 7 + 1; };
  auto zeroBasedValueAt = [](vector_size_t /*row*/, vector_size_t idx) {
    return idx;
  };
  auto mixedArrays = makeArrayVector<int64_t>(
      kArrayAccessVectorSize, variableSizeAt, zeroBasedValueAt);
  auto mixedInput = makeRowVector({mixedArrays, mixedIndices});

  assertExpressionMatchesCpu(
      "element_at(c0, c1)", mixedInput, mixedInput->rowType());
}

TEST_F(CudfSimpleFilterProjectTest, elementAtArrayVarcharVariableInput) {
  constexpr vector_size_t kArrayAccessVectorSize{1'000};

  auto indices = makeFlatVector<int64_t>(
      kArrayAccessVectorSize, [](vector_size_t row) { return 1 + row % 7; });

  auto arrays = makeArrayVector<std::string>(
      kArrayAccessVectorSize,
      [](vector_size_t /*row*/) { return 7; },
      [](vector_size_t /*row*/, vector_size_t idx) {
        return std::to_string(idx + 1);
      });
  auto input = makeRowVector({arrays, indices});

  assertExpressionMatchesCpu("element_at(c0, c1)", input, input->rowType());
}

TEST_F(CudfSimpleFilterProjectTest, elementAtArrayAllIndicesGreaterThanSize) {
  constexpr vector_size_t kArrayAccessVectorSize{1'000};

  auto indices = makeFlatVector<int64_t>(
      kArrayAccessVectorSize, [](vector_size_t /*row*/) { return 2; });
  auto arrays = makeArrayVector<int64_t>(
      kArrayAccessVectorSize,
      [](vector_size_t /*row*/) { return 1; },
      [](vector_size_t /*row*/, vector_size_t /*idx*/) { return 1; });
  auto input = makeRowVector({arrays, indices});

  assertExpressionMatchesCpu("element_at(c0, c1)", input, input->rowType());
}

TEST_F(CudfSimpleFilterProjectTest, elementAtArrayColumnIndexZeroFails) {
  constexpr vector_size_t kArrayAccessVectorSize{1'000};

  auto indices =
      makeFlatVector<int64_t>(kArrayAccessVectorSize, [](vector_size_t row) {
        if (row == 40) {
          return 0;
        }
        return 1 + row % 7;
      });
  auto arrays = makeArrayVector<int64_t>(
      kArrayAccessVectorSize,
      [](vector_size_t row) { return 2 + row % 7; },
      [](vector_size_t /*row*/, vector_size_t idx) { return 1 + idx; });

  VELOX_ASSERT_THROW(
      functions::test::FunctionBaseTest::evaluate(
          "element_at(c0, c1)", makeRowVector({arrays, indices})),
      "SQL array indices start at 1. Got 0.");
}

TEST_F(CudfSimpleFilterProjectTest, arrayAccessConstantIndex) {
  auto arrays = makeNullableArrayVector<int32_t>({
      {{10, 20, 30}},
      {{4, 5}},
      std::nullopt,
      {{7, std::nullopt, 9}},
  });

  auto input = makeRowVector({arrays});

  for (const auto& expression : {
           "element_at(c0, 2)",
           "element_at(c0, cast(2 as bigint))",
           "subscript(c0, 2)",
           "subscript(c0, cast(2 as bigint))",
       }) {
    SCOPED_TRACE(expression);
    assertExpressionMatchesCpu(expression, input, input->rowType());
  }
}

TEST_F(CudfSimpleFilterProjectTest, arrayAccessNullConstantIndex) {
  auto arrays = makeNullableArrayVector<int32_t>({
      {{10, 20, 30}},
      {{4, 5}},
      std::nullopt,
  });

  auto input = makeRowVector({arrays});

  for (const auto& expression : {
           "element_at(c0, cast(null as integer))",
           "element_at(c0, cast(null as bigint))",
           "subscript(c0, cast(null as integer))",
           "subscript(c0, cast(null as bigint))",
       }) {
    SCOPED_TRACE(expression);
    assertExpressionMatchesCpu(expression, input, input->rowType());
  }
}

TEST_F(CudfSimpleFilterProjectTest, arrayAccessSupportedScalarElementTypes) {
  // Cover all scalar ARRAY element types supported by Velox-cuDF conversion.
  assertArrayAccessElementTypeMatchesCpu(
      "tinyint", makeArrayAccessNumericTestVector<int8_t>());
  assertArrayAccessElementTypeMatchesCpu(
      "smallint", makeArrayAccessNumericTestVector<int16_t>());
  assertArrayAccessElementTypeMatchesCpu(
      "integer", makeArrayAccessNumericTestVector<int32_t>());
  assertArrayAccessElementTypeMatchesCpu(
      "bigint", makeArrayAccessNumericTestVector<int64_t>());
  assertArrayAccessElementTypeMatchesCpu(
      "real", makeArrayAccessNumericTestVector<float>());
  assertArrayAccessElementTypeMatchesCpu(
      "double", makeArrayAccessNumericTestVector<double>());
  assertArrayAccessElementTypeMatchesCpu(
      "date", makeArrayAccessNumericTestVector<int32_t>(ARRAY(DATE())));
  assertArrayAccessElementTypeMatchesCpu(
      "short decimal",
      makeArrayAccessNumericTestVector<int64_t>(ARRAY(DECIMAL(10, 2))));
  assertArrayAccessElementTypeMatchesCpu(
      "long decimal",
      makeArrayAccessNumericTestVector<int128_t>(ARRAY(DECIMAL(20, 4))));

  auto booleanArrays = makeNullableArrayVector<bool>({
      {{true, false, true}},
      {{false, std::nullopt, true}},
      std::nullopt,
      {{true, true}},
  });
  assertArrayAccessElementTypeMatchesCpu("boolean", booleanArrays);

  auto varcharArrays = makeNullableArrayVector<std::string>({
      {{"alpha", "beta", "gamma"}},
      {{"delta", std::nullopt, "zeta"}},
      std::nullopt,
      {{"eta", "theta"}},
  });
  assertArrayAccessElementTypeMatchesCpu("varchar", varcharArrays);

  auto varbinaryArrays = makeNullableArrayVector<std::string>(
      {
          {{"alpha", "beta", "gamma"}},
          {{"delta", std::nullopt, "zeta"}},
          std::nullopt,
          {{"eta", "theta"}},
      },
      ARRAY(VARBINARY()));
  assertArrayAccessElementTypeMatchesCpu("varbinary", varbinaryArrays);

  auto timestampArrays = makeNullableArrayVector<Timestamp>({
      {{Timestamp(1, 0), Timestamp(2, 0), Timestamp(3, 0)}},
      {{Timestamp(4, 0), std::nullopt, Timestamp(6, 0)}},
      std::nullopt,
      {{Timestamp(7, 0), Timestamp(8, 0)}},
  });
  assertArrayAccessElementTypeMatchesCpu("timestamp", timestampArrays);
}

TEST_F(CudfSimpleFilterProjectTest, arrayAccessSupportedComplexElementTypes) {
  // Cover complex ARRAY element types that cuDF list extraction can return.
  using OptionalIntArray = std::optional<std::vector<std::optional<int32_t>>>;
  using OptionalNestedArray = std::optional<std::vector<OptionalIntArray>>;

  std::vector<OptionalNestedArray> nestedArrayData = {
      std::vector<OptionalIntArray>{
          std::vector<std::optional<int32_t>>{1, 2},
          std::vector<std::optional<int32_t>>{3},
          std::vector<std::optional<int32_t>>{4, 5},
      },
      std::vector<OptionalIntArray>{
          std::vector<std::optional<int32_t>>{6},
          std::vector<std::optional<int32_t>>{7, 8},
          std::vector<std::optional<int32_t>>{9},
      },
      std::vector<OptionalIntArray>{
          std::vector<std::optional<int32_t>>{10},
          std::vector<std::optional<int32_t>>{11, 12},
      },
      std::vector<OptionalIntArray>{
          std::vector<std::optional<int32_t>>{13},
          std::vector<std::optional<int32_t>>{14, 15},
      },
  };
  assertArrayAccessElementTypeMatchesCpu(
      "array", makeNullableNestedArrayVector<int32_t>(nestedArrayData));

  auto rowArrays = makeArrayOfRowVector(
      ROW({"a", "b"}, {INTEGER(), VARCHAR()}),
      {
          {variant::row({1, "alpha"}),
           variant::row({2, "beta"}),
           variant::row({3, "gamma"})},
          {variant::row({4, "delta"}),
           variant::row({5, "epsilon"}),
           variant::row({6, "zeta"})},
          {variant::row({7, "eta"}), variant::row({8, "theta"})},
          {variant::row({9, "iota"}), variant::row({10, "kappa"})},
      });
  assertArrayAccessElementTypeMatchesCpu("row", rowArrays);
}

TEST_F(CudfSimpleFilterProjectTest, arrayAccessConstantArraySupportedTypes) {
  // Constant array literals take a different input-column path from array
  // columns, so repeat the supported element-type matrix here.
  assertConstantArrayAccessMatchesCpu(
      "tinyint",
      "array[cast(1 as tinyint), cast(2 as tinyint), cast(3 as tinyint)]");
  assertConstantArrayAccessMatchesCpu(
      "smallint",
      "array[cast(1 as smallint), cast(2 as smallint), cast(3 as smallint)]");
  assertConstantArrayAccessMatchesCpu(
      "integer",
      "array[cast(1 as integer), cast(2 as integer), cast(3 as integer)]");
  assertConstantArrayAccessMatchesCpu(
      "bigint",
      "array[cast(1 as bigint), cast(2 as bigint), cast(3 as bigint)]");
  assertConstantArrayAccessMatchesCpu(
      "real",
      "array[cast(1.25 as real), cast(2.5 as real), cast(3.75 as real)]");
  assertConstantArrayAccessMatchesCpu(
      "double",
      "array[cast(1.25 as double), "
      "cast(2.5 as double), "
      "cast(3.75 as double)]");
  assertConstantArrayAccessMatchesCpu("boolean", "array[true, false, true]");
  assertConstantArrayAccessMatchesCpu(
      "varchar", "array['alpha', 'beta', 'gamma']");
  assertConstantArrayAccessMatchesCpu(
      "varbinary",
      "array[cast('alpha' as varbinary), "
      "cast('beta' as varbinary), "
      "cast('gamma' as varbinary)]");
  assertConstantArrayAccessMatchesCpu(
      "date", "array[DATE '2020-01-01', DATE '2020-01-02', DATE '2020-01-03']");
  assertConstantArrayAccessMatchesCpu(
      "timestamp",
      "array[cast('2020-01-01 00:00:00' as timestamp), "
      "cast('2020-01-02 00:00:00' as timestamp), "
      "cast('2020-01-03 00:00:00' as timestamp)]");
  assertConstantArrayAccessMatchesCpu(
      "short decimal",
      "array[cast('1.00' as decimal(10, 2)), "
      "cast('2.00' as decimal(10, 2)), "
      "cast('3.00' as decimal(10, 2))]");
  assertConstantArrayAccessMatchesCpu(
      "long decimal",
      "array[cast('1.0000' as decimal(20, 4)), "
      "cast('2.0000' as decimal(20, 4)), "
      "cast('3.0000' as decimal(20, 4))]");
  assertConstantArrayAccessMatchesCpu(
      "array",
      "array_constructor("
      "array_constructor(1, 2), "
      "array_constructor(3), "
      "array_constructor(4, 5))");
  assertConstantArrayAccessMatchesCpu(
      "row",
      "array_constructor("
      "row_constructor(1, 'alpha'), "
      "row_constructor(2, 'beta'), "
      "row_constructor(3, 'gamma'))");
}

TEST_F(CudfSimpleFilterProjectTest, subscriptConstantArrayVariableIndex) {
  auto groupIds = makeFlatVector<int64_t>({0, 1, 2});
  auto input = makeRowVector({groupIds});

  assertExpressionMatchesCpu(
      "subscript(array[1, 1, 0], c0 + 1)", input, input->rowType());
  assertExpressionMatchesCpu(
      "subscript(array[1, 0, 0], c0 + 1)", input, input->rowType());
}

TEST_F(CudfSimpleFilterProjectTest, elementAtArrayVariableIndex) {
  auto arrays = makeNullableArrayVector<int32_t>({
      {{1, 2, 3}},
      {{4, std::nullopt, 6}},
      std::nullopt,
      {{7, 8}},
      {{9, 10, 11}},
  });
  auto integerIndices =
      makeNullableFlatVector<int32_t>({3, -2, std::nullopt, 4, -4});
  auto integerInput = makeRowVector({arrays, integerIndices});
  assertExpressionMatchesCpu(
      "element_at(c0, c1)", integerInput, integerInput->rowType());

  auto bigintIndices =
      makeNullableFlatVector<int64_t>({3, -2, std::nullopt, 4, -4});
  auto bigintInput = makeRowVector({arrays, bigintIndices});
  assertExpressionMatchesCpu(
      "element_at(c0, c1)", bigintInput, bigintInput->rowType());
}

TEST_F(CudfSimpleFilterProjectTest, arrayAccessVariableIndexSupportedTypes) {
  auto arrays = makeNullableArrayVector<int32_t>({
      {{1, 2, 3}},
      {{4, std::nullopt, 6}},
      std::nullopt,
      {{7, 8}},
      {{9, 10, 11}},
  });
  auto integerIndices = makeFlatVector<int32_t>({3, 2, 1, 2, 1});
  auto integerInput = makeRowVector({arrays, integerIndices});
  for (const auto& expression : {"element_at(c0, c1)", "subscript(c0, c1)"}) {
    SCOPED_TRACE(expression);
    assertExpressionMatchesCpu(
        expression, integerInput, integerInput->rowType());
  }

  auto bigintIndices = makeFlatVector<int64_t>({3, 2, 1, 2, 1});
  auto bigintInput = makeRowVector({arrays, bigintIndices});
  for (const auto& expression : {"element_at(c0, c1)", "subscript(c0, c1)"}) {
    SCOPED_TRACE(expression);
    assertExpressionMatchesCpu(expression, bigintInput, bigintInput->rowType());
  }
}

TEST_F(CudfSimpleFilterProjectTest, elementAtArrayNegativeConstantIndex) {
  auto arrays = makeNullableArrayVector<int32_t>({
      {{1, 2, 3}},
      {{4, std::nullopt, 6}},
      std::nullopt,
      {{7}},
  });
  auto input = makeRowVector({arrays});

  assertExpressionMatchesCpu("element_at(c0, -1)", input, input->rowType());
}

TEST_F(CudfSimpleFilterProjectTest, elementAtArrayNegativeVariableIndex) {
  auto arrays = makeNullableArrayVector<int32_t>({
      {{1, 2, 3}},
      {{4, std::nullopt, 6}},
      std::nullopt,
      {{7}},
  });

  auto integerIndices = makeFlatVector<int32_t>({-1, -1, -1, -1});
  auto integerInput = makeRowVector({arrays, integerIndices});
  assertExpressionMatchesCpu(
      "element_at(c0, c1)", integerInput, integerInput->rowType());

  auto bigintIndices = makeFlatVector<int64_t>({-1, -1, -1, -1});
  auto bigintInput = makeRowVector({arrays, bigintIndices});
  assertExpressionMatchesCpu(
      "element_at(c0, c1)", bigintInput, bigintInput->rowType());
}

TEST_F(CudfSimpleFilterProjectTest, arrayAccessRejectsInvalidIndex) {
  auto arrays = makeArrayVector<int32_t>({{1, 2, 3}});

  auto input = makeRowVector({arrays});
  VELOX_ASSERT_THROW(
      functions::test::FunctionBaseTest::evaluate("element_at(c0, 0)", input),
      "SQL array indices start at 1. Got 0.");

  VELOX_ASSERT_THROW(
      functions::test::FunctionBaseTest::evaluate("subscript(c0, -1)", input),
      "Array subscript index cannot be negative");

  VELOX_ASSERT_THROW(
      functions::test::FunctionBaseTest::evaluate("subscript(c0, 4)", input),
      "Array subscript index out of bounds");
}

TEST_F(CudfSimpleFilterProjectTest, elementAtReturnsNullForOutOfBoundsIndex) {
  auto arrays = makeArrayVector<int32_t>({{1, 2, 3}});
  auto input = makeRowVector({arrays});

  assertExpressionMatchesCpu("element_at(c0, 4)", input, input->rowType());
}

TEST_F(CudfSimpleFilterProjectTest, nullLogicalAnd) {
  const auto rowType = ROW({{"c0", BOOLEAN()}, {"c1", BOOLEAN()}});
  // All 9 combinations of {true, false, null} x {true, false, null}.
  auto c0 = makeNullableFlatVector<bool>({
      true,
      true,
      true,
      false,
      false,
      false,
      std::nullopt,
      std::nullopt,
      std::nullopt,
  });
  auto c1 = makeNullableFlatVector<bool>({
      true,
      false,
      std::nullopt,
      true,
      false,
      std::nullopt,
      true,
      false,
      std::nullopt,
  });
  auto input = makeRowVector({c0, c1});
  assertExpressionMatchesCpu("c0 AND c1", input, rowType);
}

TEST_F(CudfSimpleFilterProjectTest, nullLogicalOr) {
  const auto rowType = ROW({{"c0", BOOLEAN()}, {"c1", BOOLEAN()}});
  auto c0 = makeNullableFlatVector<bool>({
      true,
      true,
      true,
      false,
      false,
      false,
      std::nullopt,
      std::nullopt,
      std::nullopt,
  });
  auto c1 = makeNullableFlatVector<bool>({
      true,
      false,
      std::nullopt,
      true,
      false,
      std::nullopt,
      true,
      false,
      std::nullopt,
  });
  auto input = makeRowVector({c0, c1});
  assertExpressionMatchesCpu("c0 OR c1", input, rowType);
}

TEST_F(CudfSimpleFilterProjectTest, nullLogicalAndThreeArg) {
  const auto rowType =
      ROW({{"c0", BOOLEAN()}, {"c1", BOOLEAN()}, {"c2", BOOLEAN()}});
  // (true AND null) AND false -> false; (null AND true) AND false -> false.
  auto c0 = makeNullableFlatVector<bool>({true, std::nullopt, false});
  auto c1 = makeNullableFlatVector<bool>({std::nullopt, true, true});
  auto c2 = makeNullableFlatVector<bool>({false, false, true});
  auto input = makeRowVector({c0, c1, c2});
  assertExpressionMatchesCpu("c0 AND c1 AND c2", input, rowType);
}

TEST_F(CudfSimpleFilterProjectTest, nullLogicalOrThreeArg) {
  const auto rowType =
      ROW({{"c0", BOOLEAN()}, {"c1", BOOLEAN()}, {"c2", BOOLEAN()}});
  // (false OR null) OR true -> true; (null OR false) OR true -> true.
  auto c0 = makeNullableFlatVector<bool>({false, std::nullopt, std::nullopt});
  auto c1 = makeNullableFlatVector<bool>({std::nullopt, false, false});
  auto c2 = makeNullableFlatVector<bool>({true, true, true});
  auto input = makeRowVector({c0, c1, c2});
  assertExpressionMatchesCpu("c0 OR c1 OR c2", input, rowType);
}

TEST_F(CudfSimpleFilterProjectTest, logicalAndAllLiterals) {
  const auto rowType = ROW({{"c0", BOOLEAN()}});
  auto input = makeRowVector({makeFlatVector<bool>({true})});
  for (const auto& expr :
       {"true AND true",
        "true AND false",
        "false AND true",
        "false AND false"}) {
    assertExpressionMatchesCpu(expr, input, rowType);
  }
}

TEST_F(CudfSimpleFilterProjectTest, logicalOrAllLiterals) {
  const auto rowType = ROW({{"c0", BOOLEAN()}});
  auto input = makeRowVector({makeFlatVector<bool>({true})});
  for (const auto& expr :
       {"true OR true", "true OR false", "false OR true", "false OR false"}) {
    assertExpressionMatchesCpu(expr, input, rowType);
  }
}

TEST_F(CudfSimpleFilterProjectTest, logicalAndColumnWithLiteral) {
  const auto rowType = ROW({{"c0", BOOLEAN()}});
  auto c0 =
      makeNullableFlatVector<bool>({true, false, std::nullopt, std::nullopt});
  auto input = makeRowVector({c0});
  for (const auto& expr :
       {"c0 AND true", "true AND c0", "c0 AND false", "false AND c0"}) {
    assertExpressionMatchesCpu(expr, input, rowType);
  }
}

TEST_F(CudfSimpleFilterProjectTest, logicalOrColumnWithLiteral) {
  const auto rowType = ROW({{"c0", BOOLEAN()}});
  auto c0 =
      makeNullableFlatVector<bool>({true, false, std::nullopt, std::nullopt});
  auto input = makeRowVector({c0});
  for (const auto& expr :
       {"c0 OR true", "true OR c0", "c0 OR false", "false OR c0"}) {
    assertExpressionMatchesCpu(expr, input, rowType);
  }
}

TEST_F(CudfSimpleFilterProjectTest, logicalAndThreeArgLiteralsMixed) {
  const auto rowType = ROW({{"c0", BOOLEAN()}, {"c1", BOOLEAN()}});
  auto c0 = makeNullableFlatVector<bool>({true, false, std::nullopt});
  auto c1 = makeNullableFlatVector<bool>({false, true, true});
  auto input = makeRowVector({c0, c1});
  for (const auto& expr :
       {"c0 AND true AND c1",
        "true AND c0 AND c1",
        "c0 AND c1 AND false",
        "false AND c0 AND c1"}) {
    assertExpressionMatchesCpu(expr, input, rowType);
  }
}

TEST_F(CudfSimpleFilterProjectTest, logicalOrThreeArgLiteralsMixed) {
  const auto rowType = ROW({{"c0", BOOLEAN()}, {"c1", BOOLEAN()}});
  auto c0 = makeNullableFlatVector<bool>({false, true, std::nullopt});
  auto c1 = makeNullableFlatVector<bool>({false, false, true});
  auto input = makeRowVector({c0, c1});
  for (const auto& expr :
       {"c0 OR false OR c1",
        "false OR c0 OR c1",
        "c0 OR c1 OR true",
        "true OR c0 OR c1"}) {
    assertExpressionMatchesCpu(expr, input, rowType);
  }
}

TEST_F(CudfFilterProjectTest, andAndAndExpr) {
  auto data = makeRowVector(
      {makeFlatVector<int64_t>({100, 100, 100, 100}, DECIMAL(17, 2)),
       makeFlatVector<int64_t>({100, -100, 100, 100}, DECIMAL(17, 2)),
       makeFlatVector<int64_t>({100, -100, -100, 100}, DECIMAL(17, 2)),
       makeFlatVector<int64_t>({100, -100, -100, -100}, DECIMAL(17, 2))});
  auto plan =
      PlanBuilder()
          .values({data})
          .project(
              {"(c0 > CAST(0.0 AS DECIMAL(17, 2))) AND (c1 > CAST(0.0 AS DECIMAL(17, 2))) AND (c2 > CAST(0.0 AS DECIMAL(17, 2))) AND (c3 > CAST(0.0 AS DECIMAL(17, 2))) AS result"})
          .planNode();
  auto result = AssertQueryBuilder(plan).copyResults(pool());

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>({true, false, false, false}),
  });
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfFilterProjectTest, andAndAndWithDecimalDivideBelowExpr) {
  auto data = makeRowVector(
      {makeFlatVector<int64_t>({100, 100, 100, 100}, DECIMAL(17, 2)),
       makeFlatVector<int64_t>({100, -100, 100, 100}, DECIMAL(17, 2)),
       makeFlatVector<int64_t>({100, -100, -100, 100}, DECIMAL(17, 2)),
       makeFlatVector<int64_t>({100, -100, -100, -100}, DECIMAL(17, 2))});
  auto plan =
      PlanBuilder()
          .values({data})
          .project(
              {"(CAST((c0 / CAST(3.0 AS DECIMAL(17,2))) AS DECIMAL(17, 2)) > CAST(0.0 AS DECIMAL(17, 2))) AND (c1 > CAST(0.0 AS DECIMAL(17, 2))) AND (c2 > CAST(0.0 AS DECIMAL(17, 2))) AND (c3 > CAST(0.0 AS DECIMAL(17, 2))) AS result"})
          .planNode();
  auto result = AssertQueryBuilder(plan).copyResults(pool());

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>({true, false, false, false}),
  });
  facebook::velox::test::assertEqualVectors(expected, result);
}

} // namespace
