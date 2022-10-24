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
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::window::test {

namespace {

// This test class is for different variations of lead and lag functions
// parameterized by the over clause. Each function invocation tests the function
// over all possible frame clauses.
class LeadLagTest : public WindowTestBase {
 protected:
  LeadLagTest() : overClause_("") {}

  explicit LeadLagTest(const std::string& overClause)
      : overClause_(overClause) {}

  // This test has all important variations of the lead and lag functions
  // invocation to be tested per (dataset, partition, frame) clause combination.
  void testLeadLag(const std::vector<RowVectorPtr>& input) {
    // This is a basic test case to give the value of the first frame row.
    testWindowFunction(input, "lead(c0, 1)");
    testWindowFunction(input, "lag(c0, 1)");

    // This function invocation gets the column value of the 10th
    // frame row. Many tests have < 10 rows per partition. so the function
    // is expected to return null for such offsets.
    testWindowFunction(input, "lead(c0, 10)");
    testWindowFunction(input, "lag(c0, 10)");

    // This test gets the lead/lag offset from a column c2. The offsets could
    // be outside the partition also. The error cases for -ve offset values
    // are tested separately.
    testWindowFunction(input, "lead(c3, c2)");
    testWindowFunction(input, "lag(c3, c2)");
  }

  // This is for testing different output column types in the lead and lag
  // functions' column parameter.
  void testPrimitiveType(const TypePtr& type) {
    vector_size_t size = 25;
    auto vectors = makeRowVector({
        makeFlatVector<int32_t>(size, [](auto row) { return row % 5; }),
        makeFlatVector<int32_t>(size, [](auto row) { return row; }),
        makeFlatVector<int64_t>(size, [](auto row) { return row % 3 + 1; }),
        makeFlatVector<int64_t>(size, [](auto row) { return row % 3 + 1; }),
        // Note : The Fuzz vector can have null values.
        makeRandomInputVector(type, size, 0.3),
    });

    // Add c4 column in sort order in overClauses to impose a deterministic
    // output row order in the tests.
    auto newOverClause = overClause_ + ", c4";

    // The below tests cover lead/lag invocations with constant and column
    // arguments. The offsets could also give rows beyond the partition
    // returning null in those cases.
    WindowTestBase::testWindowFunction(
        {vectors}, "lead(c4, 7)", {newOverClause}, kFrameClauses);
    WindowTestBase::testWindowFunction(
        {vectors}, "lead(c4, c2)", {newOverClause}, kFrameClauses);

    WindowTestBase::testWindowFunction(
        {vectors}, "lag(c4, 7)", {newOverClause}, kFrameClauses);
    WindowTestBase::testWindowFunction(
        {vectors}, "lag(c4, c2)", {newOverClause}, kFrameClauses);
  }

 private:
  void testWindowFunction(
      const std::vector<RowVectorPtr>& input,
      const std::string& function) {
    WindowTestBase::testWindowFunction(
        input, function, {overClause_}, kFrameClauses);
  }

  const std::string overClause_;
};

class MultiLeadLagTest : public LeadLagTest,
                         public testing::WithParamInterface<std::string> {
 public:
  MultiLeadLagTest() : LeadLagTest(GetParam()) {}
};

// Tests lead/lag with data of a uniform distribution.
TEST_P(MultiLeadLagTest, basic) {
  testLeadLag({makeSimpleVector(50)});
}

// Tests lead/lag with a dataset with a single partition.
TEST_P(MultiLeadLagTest, singlePartition) {
  testLeadLag({makeSinglePartitionVector(50)});
}

// Tests lead/lag with a dataset with a single partition, but spread across
// 2 input vectors.
TEST_P(MultiLeadLagTest, multiInput) {
  testLeadLag({makeSinglePartitionVector(50), makeSinglePartitionVector(75)});
}

// Tests lead/lag with a dataset where all partitions have a single row.
TEST_P(MultiLeadLagTest, singleRowPartitions) {
  testLeadLag({makeSingleRowPartitionsVector((50))});
}

// Tests lead/lag functions with a randomly generated dataset.
TEST_P(MultiLeadLagTest, randomInput) {
  testLeadLag({makeRandomInputVector((50))});
}

// Tests lead/lag functions by projecting result columns of different types.
TEST_P(MultiLeadLagTest, integerValues) {
  testPrimitiveType(INTEGER());
}

TEST_P(MultiLeadLagTest, tinyintValues) {
  testPrimitiveType(TINYINT());
}

TEST_P(MultiLeadLagTest, smallintValues) {
  testPrimitiveType(SMALLINT());
}

TEST_P(MultiLeadLagTest, bigintValues) {
  testPrimitiveType(BIGINT());
}

TEST_P(MultiLeadLagTest, realValues) {
  testPrimitiveType(REAL());
}

TEST_P(MultiLeadLagTest, doubleValues) {
  testPrimitiveType(DOUBLE());
}

TEST_P(MultiLeadLagTest, varcharValues) {
  testPrimitiveType(VARCHAR());
}

TEST_P(MultiLeadLagTest, varbinaryValues) {
  testPrimitiveType(VARBINARY());
}

TEST_P(MultiLeadLagTest, timestampValues) {
  testPrimitiveType(TIMESTAMP());
}

TEST_P(MultiLeadLagTest, dateValues) {
  testPrimitiveType(DATE());
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    LeadLagTest,
    MultiLeadLagTest,
    testing::ValuesIn(std::vector<std::string>(kOverClauses)));

TEST_F(LeadLagTest, nullOffsets) {
  // Test that lead and lag with null offsets return rows with null value.
  vector_size_t size = 100;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
      makeFlatVector<int64_t>(
          size, [](auto row) { return row % 3 + 1; }, nullEvery(5)),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
  });

  WindowTestBase::testWindowFunction({vectors}, "lead(c0, c2)", kOverClauses);
  WindowTestBase::testWindowFunction({vectors}, "lag(c0, c2)", kOverClauses);
}

TEST_F(LeadLagTest, invalidOffsets) {
  vector_size_t size = 20;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
      makeFlatVector<int64_t>(size, [](auto row) { return (row % 5) - 2; }),
  });

  std::string overClause = "partition by c0 order by c1";
  std::string offsetError = "Offset must be at least 0";
  assertWindowFunctionError({vectors}, "lead(c0, -1)", overClause, offsetError);
  assertWindowFunctionError({vectors}, "lead(c0, c2)", overClause, offsetError);
  assertWindowFunctionError({vectors}, "lag(c0, -1)", overClause, offsetError);
  assertWindowFunctionError({vectors}, "lag(c0, c2)", overClause, offsetError);
}

TEST_F(LeadLagTest, invalidFrames) {
  vector_size_t size = 20;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
      makeFlatVector<int64_t>(size, [](auto row) { return row % 5; }),
  });

  std::string overClause = "partition by c0 order by c1";
  assertWindowFunctionError(
      {vectors},
      "lead(c0, 5)",
      overClause,
      "rows between 0 preceding and current row",
      "k in frame bounds must be at least 1");
  assertWindowFunctionError(
      {vectors},
      "lag(c0, 5)",
      overClause,
      "rows between c2 preceding and current row",
      "k in frame bounds must be at least 1");
}

}; // namespace
}; // namespace facebook::velox::window::test
