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

class FirstLastValueTest : public WindowTestBase,
                           public testing::WithParamInterface<std::string> {
 protected:
  FirstLastValueTest() : function_(GetParam()) {}

  void testPrimitiveType(const TypePtr& type) {
    vector_size_t size = 25;
    auto vectors = makeRowVector({
        makeFlatVector<int32_t>(size, [](auto row) { return row % 5; }),
        makeFlatVector<int32_t>(size, [](auto row) { return row; }),
        makeFlatVector<int64_t>(size, [](auto row) { return row % 3 + 1; }),
        typeValues(type, size),
    });
    WindowTestBase::testWindowFunction(
        {vectors}, function_, kFixedBasicOverClauses);
    WindowTestBase::testWindowFunction(
        {vectors}, function_, kFixedSortOrderBasedOverClauses);
  }

  RowVectorPtr makeBasicVectors(vector_size_t size) {
    return makeRowVector({
        makeFlatVector<int32_t>(
            size, [](auto row) -> int32_t { return row % 10; }),
        makeFlatVector<int32_t>(
            size, [](auto row) -> int32_t { return row % 7; }),
        makeFlatVector<int32_t>(size, [](auto row) -> int32_t { return row; }),
    });
  }

  RowVectorPtr makeSinglePartitionVector(vector_size_t size) {
    return makeRowVector({
        makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
        makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
        makeFlatVector<int32_t>(size, [](auto row) -> int32_t { return row; }),
    });
  }

  void testWindowFunction(
      const std::vector<RowVectorPtr>& vectors,
      const std::vector<std::string>& overClauses,
      const std::vector<std::string>& frameClauses = {""}) {
    WindowTestBase::testWindowFunction(
        vectors, function_, overClauses, frameClauses);
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

TEST_P(FirstLastValueTest, basic) {
  auto vectors = makeBasicVectors(50);

  testWindowFunction({vectors}, kFixedBasicOverClauses);
}

TEST_P(FirstLastValueTest, basicWithSortOrder) {
  auto vectors = makeBasicVectors(50);

  testWindowFunction({vectors}, kFixedSortOrderBasedOverClauses);
}

TEST_P(FirstLastValueTest, singlePartition) {
  auto vectors = makeSinglePartitionVector(100);

  testWindowFunction({vectors}, kFixedBasicOverClauses);
}

TEST_P(FirstLastValueTest, singlePartitionWithSortOrder) {
  auto vectors = makeSinglePartitionVector(500);

  testWindowFunction({vectors}, kFixedSortOrderBasedOverClauses);
}

TEST_P(FirstLastValueTest, multiInput) {
  auto vectors = makeSinglePartitionVector(250);
  auto doubleVectors = {vectors, vectors};

  testWindowFunction(doubleVectors, kFixedBasicOverClauses);
}

TEST_P(FirstLastValueTest, multiInputWithSortOrder) {
  auto vectors = makeSinglePartitionVector(250);
  auto doubleVectors = {vectors, vectors};

  testWindowFunction(doubleVectors, kFixedSortOrderBasedOverClauses);
}

TEST_P(FirstLastValueTest, integerValues) {
  testPrimitiveType(INTEGER());
}

TEST_P(FirstLastValueTest, tinyintValues) {
  testPrimitiveType(TINYINT());
}

TEST_P(FirstLastValueTest, smallintValues) {
  testPrimitiveType(SMALLINT());
}

TEST_P(FirstLastValueTest, bigintValues) {
  testPrimitiveType(BIGINT());
}

TEST_P(FirstLastValueTest, realValues) {
  testPrimitiveType(REAL());
}

TEST_P(FirstLastValueTest, doubleValues) {
  testPrimitiveType(DOUBLE());
}

TEST_P(FirstLastValueTest, varcharValues) {
  testPrimitiveType(VARCHAR());
}

TEST_P(FirstLastValueTest, varbinaryValues) {
  testPrimitiveType(VARBINARY());
}

TEST_P(FirstLastValueTest, timestampValues) {
  testPrimitiveType(TIMESTAMP());
}

TEST_P(FirstLastValueTest, dateValues) {
  testPrimitiveType(DATE());
}

TEST_P(FirstLastValueTest, nullOffsets) {
  // Test that nth_value with null offset returns rows with null value.
  vector_size_t size = 100;

  auto vectors = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 50; }),
      makeFlatVector<int64_t>(
          size, [](auto row) { return row % 3 + 1; }, nullEvery(5)),
  });

  testWindowFunction({vectors}, kFixedBasicOverClauses);
}

TEST_P(FirstLastValueTest, basicRangeFrames) {
  auto vectors = makeBasicVectors(50);

  testWindowFunction({vectors}, kFrameOverClauses, kRangeFrameClauses);
}

TEST_P(FirstLastValueTest, singlePartitionRangeFrames) {
  auto vectors = makeSinglePartitionVector(400);

  testWindowFunction({vectors}, kFrameOverClauses, kRangeFrameClauses);
}

TEST_P(FirstLastValueTest, multiInputRangeFrames) {
  auto vectors = makeSinglePartitionVector(200);
  auto doubleVectors = {vectors, vectors};

  testWindowFunction(doubleVectors, kFrameOverClauses, kRangeFrameClauses);
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    FirstLastValueTest,
    FirstLastValueTest,
    testing::ValuesIn(
        {std::string("first_value(c0)"),
         std::string("first_value(c1)"),
         std::string("first_value(c2)"),
         std::string("last_value(c0)"),
         std::string("last_value(c1)"),
         std::string("last_value(c2)")}));

}; // namespace
}; // namespace facebook::velox::window::test
