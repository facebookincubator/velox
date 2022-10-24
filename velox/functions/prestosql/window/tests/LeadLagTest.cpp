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

class LeadLagTest : public WindowTestBase,
                    public testing::WithParamInterface<std::string> {
 protected:
  LeadLagTest() : function_(GetParam()) {}

  void testPrimitiveType(const TypePtr& type) {
    vector_size_t size = 100;
    auto vectors = makeRowVector({
        makeFlatVector<int32_t>(size, [](auto row) { return row % 5; }),
        makeFlatVector<int32_t>(size, [](auto row) { return row; }),
        makeFlatVector<int64_t>(size, [](auto row) { return row % 3 + 1; }),
        typeValues(type, size),
    });
    testTwoColumnOverClauses({vectors}, function_ + "(c3)");
    testTwoColumnOverClauses({vectors}, function_ + "(c3, c2)");
    testTwoColumnOverClauses({vectors}, function_ + "(c3, 1)");
    testTwoColumnOverClauses({vectors}, function_ + "(c3, 5)");
  }

  const std::string function_;

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

TEST_P(LeadLagTest, basic) {
  vector_size_t size = 100;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row % 5; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 7; }),
      makeFlatVector<int64_t>(size, [](auto row) { return row % 3 + 1; }),
  });

  testTwoColumnOverClauses({vectors}, function_ + "(c0, c2, c2)");
  testTwoColumnOverClauses({vectors}, function_ + "(c0, 1)");
  testTwoColumnOverClauses({vectors}, function_ + "(c0, 5)");
}

TEST_P(LeadLagTest, singlePartition) {
  // Test all input rows in a single partition.
  vector_size_t size = 1'000;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
      makeFlatVector<int64_t>(size, [](auto row) { return row % 5 + 1; }),
  });

  testTwoColumnOverClauses({vectors}, function_ + "(c0, c2)");
  testTwoColumnOverClauses({vectors}, function_ + "(c0, 1)");
  testTwoColumnOverClauses({vectors}, function_ + "(c0, 25)");
}

TEST_P(LeadLagTest, integerValues) {
  testPrimitiveType(INTEGER());
}

TEST_P(LeadLagTest, tinyintValues) {
  testPrimitiveType(TINYINT());
}

TEST_P(LeadLagTest, smallintValues) {
  testPrimitiveType(SMALLINT());
}

TEST_P(LeadLagTest, bigintValues) {
  testPrimitiveType(BIGINT());
}

TEST_P(LeadLagTest, realValues) {
  testPrimitiveType(REAL());
}

TEST_P(LeadLagTest, doubleValues) {
  testPrimitiveType(DOUBLE());
}

TEST_P(LeadLagTest, varcharValues) {
  testPrimitiveType(VARCHAR());
}

TEST_P(LeadLagTest, varbinaryValues) {
  testPrimitiveType(VARBINARY());
}

TEST_P(LeadLagTest, timestampValues) {
  testPrimitiveType(TIMESTAMP());
}

TEST_P(LeadLagTest, dateValues) {
  testPrimitiveType(DATE());
}

TEST_P(LeadLagTest, nullOffsets) {
  // Test that lead with null offset returns rows with null value.
  vector_size_t size = 100;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
      makeFlatVector<int64_t>(
          size, [](auto row) { return row % 3 + 1; }, nullEvery(5)),
  });

  testTwoColumnOverClauses({vectors}, function_ + "(c0, c2)");
}

TEST_P(LeadLagTest, offsetValues) {
  // Test values for offset < 1.
  vector_size_t size = 100;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
      makeFlatVector<int64_t>(size, [](auto row) { return row % 5 - 1; }),
  });

  std::string overClause = "partition by c0 order by c1";
  std::string offsetError = "Offset must be at least 0";
  assertWindowFunctionError(
      {vectors}, function_ + "(c0, 0)", overClause, offsetError);
  assertWindowFunctionError(
      {vectors}, function_ + "(c0, -1)", overClause, offsetError);
  assertWindowFunctionError(
      {vectors}, function_ + "(c0, c2)", overClause, offsetError);
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    LeadLagTest,
    LeadLagTest,
    testing::ValuesIn({std::string("lead"), std::string("lag")}));

}; // namespace
}; // namespace facebook::velox::window::test
