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

#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::window::test {

namespace {

class NthValueTest : public WindowTestBase {};

TEST_F(NthValueTest, basic) {
  vector_size_t size = 100;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row % 5; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 7; }),
      makeFlatVector<int64_t>(size, [](auto row) { return row % 3 + 1; }),
  });

  testTwoColumnOverClauses({vectors}, "nth_value(c0, c2)");
  testTwoColumnOverClauses({vectors}, "nth_value(c0, 1)");
  testTwoColumnOverClauses({vectors}, "nth_value(c0, 5)");
}

TEST_F(NthValueTest, singlePartition) {
  // Test all input rows in a single partition.
  vector_size_t size = 1'000;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
      makeFlatVector<int64_t>(size, [](auto row) { return row % 5 + 1; }),
  });

  testTwoColumnOverClauses({vectors}, "nth_value(c0, c2)");
  testTwoColumnOverClauses({vectors}, "nth_value(c0, 1)");
  testTwoColumnOverClauses({vectors}, "nth_value(c0, 25)");
}

TEST_F(NthValueTest, allPrimitiveTypes) {
  vector_size_t size = 100;

  VectorFuzzer::Options options;
  options.vectorSize = size;
  options.nullRatio = 0.2;
  options.useMicrosecondPrecisionTimestamp = true;
  VectorFuzzer fuzzer(options, pool_.get(), 0);

  // Test all primitive types of the column in the first parameter for
  // nth_value.
  std::vector<TypePtr> typesList = {
      INTEGER(),
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      VARCHAR(),
      VARBINARY(),
      TIMESTAMP(),
      DATE()};
  for (const auto& type : typesList) {
    auto vectors = makeRowVector({
        makeFlatVector<int32_t>(size, [](auto row) { return row % 5; }),
        makeFlatVector<int32_t>(size, [](auto row) { return row; }),
        makeFlatVector<int64_t>(size, [](auto row) { return row % 3 + 1; }),
        fuzzer.fuzz(type),
    });
    testTwoColumnOverClauses({vectors}, "nth_value(c3, c2)");
    testTwoColumnOverClauses({vectors}, "nth_value(c3, 1)");
    testTwoColumnOverClauses({vectors}, "nth_value(c3, 5)");
  }
}

TEST_F(NthValueTest, nullOffsets) {
  // Test that nth_value with null offset returns rows with null value.
  vector_size_t size = 100;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
      makeFlatVector<int64_t>(
          size, [](auto row) { return row % 3 + 1; }, nullEvery(5)),
  });

  testTwoColumnOverClauses({vectors}, "nth_value(c0, c2)");
}

TEST_F(NthValueTest, offsetValues) {
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

TEST_F(NthValueTest, randomInput) {
  auto vectors = makeVectors(
      ROW(
          {
              "c0",
              "c1",
              "c2",
              "c3",
          },
          {
              BIGINT(),
              SMALLINT(),
              INTEGER(),
              BIGINT(),
          }),
      10,
      2,
      0.5);
  createDuckDbTable(vectors);

  std::vector<std::string> overClauses = {
      "partition by c0 order by c1, c2, c3",
      "partition by c1 order by c0, c2, c3",
      "partition by c0 order by c1 desc, c2, c3",
      "partition by c1 order by c0 desc, c2, c3",
      "partition by c0 order by c1 desc nulls first, c2, c3",
      "partition by c1 order by c0 nulls first, c2, c3",
      "partition by c0 order by c1",
      "partition by c0 order by c2",
      "partition by c0 order by c3",
      "partition by c1 order by c0 desc",
      "partition by c0, c1 order by c2, c3",
      "partition by c0, c1 order by c2, c3 nulls first",
      "partition by c0, c1 order by c2",
      "partition by c0, c1 order by c2 nulls first",
      "partition by c0, c1 order by c2 desc",
      "partition by c0, c1 order by c2 desc nulls first",
      "order by c0, c1, c2, c3",
      "order by c0, c1 nulls first, c2, c3",
      "order by c0, c1 desc nulls first, c2, c3",
      "order by c0 nulls first, c1 nulls first, c2, c3",
      "order by c0 nulls first, c1 desc nulls first, c2, c3",
      "order by c0 desc nulls first, c1 nulls first, c2, c3",
      "order by c0 desc nulls first, c1 desc nulls first, c2, c3",
      "partition by c0, c1, c2, c3",
  };

  testWindowFunction(vectors, "nth_value(c0, 1)", overClauses);
  testWindowFunction(vectors, "nth_value(c1, 5)", overClauses);
  testWindowFunction(vectors, "nth_value(c2, 10)", overClauses);
  testWindowFunction(vectors, "nth_value(c3, 20)", overClauses);
}

}; // namespace
}; // namespace facebook::velox::window::test
