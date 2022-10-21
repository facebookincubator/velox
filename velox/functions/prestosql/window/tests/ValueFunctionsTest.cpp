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
#include "velox/functions/prestosql/window/tests/WindowTestBase.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::window::test {

namespace {

struct TestParam {
  std::string overClause;
  std::string frameClause;
};

const std::vector<std::string> kOverClauses_ = {
    "partition by c0 order by c1, c2",
    "partition by c1 order by c0, c2",
    "partition by c0 order by c1 desc, c2",
    "partition by c1 order by c0 desc, c2",
    "partition by c0 order by c1 nulls first, c2",
    "partition by c1 order by c0 nulls first, c2",
    "partition by c0 order by c1 desc nulls first, c2",
    "partition by c1 order by c0 desc nulls first, c2",
    // No partition by clause.
    "order by c0, c1, c2",
    "order by c1, c0, c2",
    "order by c0 asc, c1 desc, c2",
    "order by c1 asc, c0 desc, c2",
    "order by c0 asc nulls first, c1 desc nulls first, c2",
    "order by c1 asc nulls first, c0 desc nulls first, c2",
    "order by c0 desc nulls first, c1 asc nulls first, c2",
    "order by c1 desc nulls first, c0 asc nulls first, c2",
    // No order by clause.
    "partition by c0, c1, c2",
};

const std::vector<std::string> kFrameClauses_ = {
    "range unbounded preceding",
};

const std::vector<TestParam> getTestParams() {
  std::vector<TestParam> testParams;
  for (auto overClause : kOverClauses_) {
    for (auto frameClause : kFrameClauses_) {
      testParams.push_back({overClause, frameClause});
    }
  }
  return testParams;
}

class NthValueTest : public WindowTestBase,
                     public testing::WithParamInterface<TestParam> {
 protected:
  NthValueTest()
      : overClause_(GetParam().overClause),
        frameClause_(GetParam().frameClause) {}

  void testPrimitiveType(const TypePtr& type) {
    vector_size_t size = 100;
    auto vectors = makeRowVector({
        makeFlatVector<int32_t>(size, [](auto row) { return row % 5; }),
        makeFlatVector<int32_t>(size, [](auto row) { return row; }),
        makeFlatVector<int64_t>(size, [](auto row) { return row % 3 + 1; }),
        typeValues(type, size),
    });
    testTwoColumnOverClauses({vectors}, "nth_value(c3, c2)", overClause_, frameClause_);
    testTwoColumnOverClauses({vectors}, "nth_value(c3, 1)", overClause_, frameClause_);
    testTwoColumnOverClauses({vectors}, "nth_value(c3, 5)", overClause_, frameClause_);
  }

  const std::string overClause_;
  const std::string frameClause_;

 private:
  VectorPtr typeValues(const TypePtr& type, vector_size_t size) {
    VectorFuzzer::Options options;
    options.vectorSize = size;
    options.nullRatio = 0.2;
    options.useMicrosecondPrecisionTimestamp = true;
    VectorFuzzer fuzzer(options, pool_.get(), 0);
    return fuzzer.fuzz(type);
  }
};

TEST_P(NthValueTest, basic) {
  vector_size_t size = 100;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row % 5; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 7; }),
      makeFlatVector<int64_t>(size, [](auto row) { return row % 3 + 1; }),
  });

  testTwoColumnOverClauses({vectors}, "nth_value(c0, c2)", overClause_, frameClause_);
  testTwoColumnOverClauses({vectors}, "nth_value(c0, 1)", overClause_, frameClause_);
  testTwoColumnOverClauses({vectors}, "nth_value(c0, 5)", overClause_, frameClause_);
}

TEST_P(NthValueTest, singlePartition) {
  // Test all input rows in a single partition.
  vector_size_t size = 1'000;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
      makeFlatVector<int64_t>(size, [](auto row) { return row % 5 + 1; }),
  });

  testTwoColumnOverClauses({vectors}, "nth_value(c0, c2)", overClause_, frameClause_);
  testTwoColumnOverClauses({vectors}, "nth_value(c0, 1)", overClause_, frameClause_);
  testTwoColumnOverClauses({vectors}, "nth_value(c0, 25)", overClause_, frameClause_);
}

TEST_P(NthValueTest, integerValues) {
  testPrimitiveType(INTEGER());
}

TEST_P(NthValueTest, tinyintValues) {
  testPrimitiveType(TINYINT());
}

TEST_P(NthValueTest, smallintValues) {
  testPrimitiveType(SMALLINT());
}

TEST_P(NthValueTest, bigintValues) {
  testPrimitiveType(BIGINT());
}

TEST_P(NthValueTest, realValues) {
  testPrimitiveType(REAL());
}

TEST_P(NthValueTest, doubleValues) {
  testPrimitiveType(DOUBLE());
}

TEST_P(NthValueTest, varcharValues) {
  testPrimitiveType(VARCHAR());
}

TEST_P(NthValueTest, varbinaryValues) {
  testPrimitiveType(VARBINARY());
}

TEST_P(NthValueTest, timestampValues) {
  testPrimitiveType(TIMESTAMP());
}

TEST_P(NthValueTest, dateValues) {
  testPrimitiveType(DATE());
}

TEST_P(NthValueTest, nullOffsets) {
  // Test that nth_value with null offset returns rows with null value.
  vector_size_t size = 100;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
      makeFlatVector<int64_t>(
          size, [](auto row) { return row % 3 + 1; }, nullEvery(5)),
  });

  testTwoColumnOverClauses({vectors}, "nth_value(c0, c2)", overClause_, frameClause_);
}

TEST_P(NthValueTest, offsetValues) {
  // Test values for offset < 1.
  vector_size_t size = 100;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
      makeFlatVector<int64_t>(size, [](auto row) { return row % 5; }),
  });

  std::string overClause = "partition by c0 order by c1";
  std::string offsetError = "Offset must be at least 1";
  assertWindowFunctionError(
      {vectors}, "nth_value(c0, 0)", overClause, offsetError);
  assertWindowFunctionError(
      {vectors}, "nth_value(c0, -1)", overClause, offsetError);
  assertWindowFunctionError(
      {vectors}, "nth_value(c0, c2)", overClause, offsetError);
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    NthValueTest,
    NthValueTest,
    testing::ValuesIn(getTestParams()));

}; // namespace
}; // namespace facebook::velox::window::test
