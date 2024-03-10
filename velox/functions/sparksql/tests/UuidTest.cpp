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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class UuidTest : public SparkFunctionBaseTest {
 protected:
  std::optional<std::string> uuid(int64_t seed, int32_t partitionIndex = 0) {
    setSparkPartitionId(partitionIndex);
    return evaluateOnce<std::string>(
        fmt::format("uuid({})", seed), makeRowVector(ROW({}), 1));
  }

  std::optional<std::string> uuidWithNullSeed(int32_t partitionIndex = 0) {
    setSparkPartitionId(partitionIndex);
    std::optional<int64_t> seed = std::nullopt;
    return evaluateOnce<std::string>("uuid(c0)", seed);
  }

  VectorPtr
  randWithBatchInput(int64_t seed, int32_t partitionIndex, int32_t batchSize) {
    setSparkPartitionId(partitionIndex);
    auto exprSet = compileExpression(fmt::format("uuid({})", seed), ROW({}));
    return evaluate(*exprSet, makeRowVector(ROW({}), batchSize));
  }
};

TEST_F(UuidTest, withSeed) {
  // With same default partitionIndex used, same seed always produces same
  // result.
  EXPECT_EQ(uuid(100, 1), uuid(100, 1));
  EXPECT_EQ(uuid(2, 1234), uuid(2, 1234));

  // Value below is the result of Spark implement with same seed.
  EXPECT_EQ(uuid(0, 0), std::string("8c7f0aac-97c4-4a2f-b716-a675d821ccc0"));

  velox::test::assertEqualVectors(
      randWithBatchInput(123, 1233, 100), randWithBatchInput(123, 1233, 100));
  velox::test::assertEqualVectors(
      randWithBatchInput(321, 1233, 33), randWithBatchInput(321, 1233, 33));

  EXPECT_NE(uuid(9, 1), uuid(100, 1));
  EXPECT_NE(uuid(9, 1), uuid(9, 2));
  EXPECT_NE(uuid(100, 1), uuid(99, 20));
}

TEST_F(UuidTest, withoutSeed) {
  EXPECT_EQ(uuid(0, 0), uuidWithNullSeed(0));
  EXPECT_EQ(uuid(0, 123), uuidWithNullSeed(123));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
