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

#include <cmath>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"

namespace facebook::velox::functions::aggregate::sparksql::test {
namespace {

// Big-endian helpers.
inline int64_t readBE64(const char*& buf) {
  uint64_t v = (static_cast<uint64_t>(static_cast<uint8_t>(buf[0])) << 56) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buf[1])) << 48) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buf[2])) << 40) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buf[3])) << 32) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buf[4])) << 24) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buf[5])) << 16) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buf[6])) << 8) |
      static_cast<uint64_t>(static_cast<uint8_t>(buf[7]));
  buf += 8;
  return static_cast<int64_t>(v);
}

inline int32_t readBE32(const char*& buf) {
  uint32_t v = (static_cast<uint32_t>(static_cast<uint8_t>(buf[0])) << 24) |
      (static_cast<uint32_t>(static_cast<uint8_t>(buf[1])) << 16) |
      (static_cast<uint32_t>(static_cast<uint8_t>(buf[2])) << 8) |
      static_cast<uint32_t>(static_cast<uint8_t>(buf[3]));
  buf += 4;
  return static_cast<int32_t>(v);
}

class CountMinSketchAggregateTest
    : public aggregate::test::AggregationTestBase {
 public:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerAggregateFunctions("");
  }

  // Parse a serialized sketch and return (depth, width, totalCount).
  std::tuple<int32_t, int32_t, int64_t> parseSketch(const StringView& sv) {
    const char* buf = sv.data();
    int32_t version = readBE32(buf);
    EXPECT_EQ(version, 1);
    int64_t totalCount = readBE64(buf);
    int32_t depth = readBE32(buf);
    int32_t width = readBE32(buf);
    return {depth, width, totalCount};
  }
};

TEST_F(CountMinSketchAggregateTest, bigintInput) {
  // Test with bigint input: VALUES (1), (2), (1) with eps=0.5, conf=0.5,
  // seed=1.
  // This matches the Spark doc example:
  // hex(count_min_sketch(col, 0.5d, 0.5d, 1)) =
  // 0000000100000000000000030000000100000004000000005D8D6AB9
  //   00000000000000000000000000000002000000000000000100000000
  //   00000000
  auto vectors = {makeRowVector({makeFlatVector<int64_t>({1, 2, 1})})};

  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, 0.5, 0.5, 1)"})
          .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultFlat = result->childAt(0)->asFlatVector<StringView>();
  ASSERT_FALSE(resultFlat->isNullAt(0));
  auto sv = resultFlat->valueAt(0);

  auto [depth, width, totalCount] = parseSketch(sv);
  EXPECT_EQ(depth, 1); // ceil(-log1p(-0.5) / log(2)) = 1
  EXPECT_EQ(width, 4); // ceil(2 / 0.5) = 4
  EXPECT_EQ(totalCount, 3);

  // Verify it matches the known Spark hex output.
  std::string expectedHex =
      "0000000100000000000000030000000100000004000000005D8D6AB9"
      "0000000000000000000000000000000200000000000000010000000000000000";
  std::string actualHex;
  for (size_t i = 0; i < sv.size(); ++i) {
    char hex[3];
    snprintf(hex, sizeof(hex), "%02X", static_cast<uint8_t>(sv.data()[i]));
    actualHex += hex;
  }
  EXPECT_EQ(actualHex, expectedHex);
}

TEST_F(CountMinSketchAggregateTest, integerInput) {
  // Test with integer input.
  auto vectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 1})})};

  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, 0.5, 0.5, 1)"})
          .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultFlat = result->childAt(0)->asFlatVector<StringView>();
  ASSERT_FALSE(resultFlat->isNullAt(0));
  auto sv = resultFlat->valueAt(0);

  auto [depth, width, totalCount] = parseSketch(sv);
  EXPECT_EQ(depth, 1);
  EXPECT_EQ(width, 4);
  EXPECT_EQ(totalCount, 3);
}

TEST_F(CountMinSketchAggregateTest, smallintInput) {
  auto vectors = {makeRowVector({makeFlatVector<int16_t>({1, 2, 3, 4, 5})})};

  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, 0.1, 0.9, 42)"})
          .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultFlat = result->childAt(0)->asFlatVector<StringView>();
  ASSERT_FALSE(resultFlat->isNullAt(0));
  auto sv = resultFlat->valueAt(0);

  auto [depth, width, totalCount] = parseSketch(sv);
  EXPECT_EQ(width, 20); // ceil(2 / 0.1) = 20
  EXPECT_GT(depth, 0);
  EXPECT_EQ(totalCount, 5);
}

TEST_F(CountMinSketchAggregateTest, varcharInput) {
  auto vectors = {
      makeRowVector({makeFlatVector<StringView>({"hello", "world", "hello"})})};

  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, 0.5, 0.5, 1)"})
          .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultFlat = result->childAt(0)->asFlatVector<StringView>();
  ASSERT_FALSE(resultFlat->isNullAt(0));
  auto sv = resultFlat->valueAt(0);

  auto [depth, width, totalCount] = parseSketch(sv);
  EXPECT_EQ(totalCount, 3);
}

TEST_F(CountMinSketchAggregateTest, varbinaryInput) {
  auto vectors = {makeRowVector({makeFlatVector<StringView>(
      {StringView("\x01\x02"), StringView("\x03\x04")}, VARBINARY())})};

  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, 0.5, 0.5, 1)"})
          .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultFlat = result->childAt(0)->asFlatVector<StringView>();
  ASSERT_FALSE(resultFlat->isNullAt(0));
  auto sv = resultFlat->valueAt(0);

  auto [depth, width, totalCount] = parseSketch(sv);
  EXPECT_EQ(totalCount, 2);
}

TEST_F(CountMinSketchAggregateTest, nullInputsSkipped) {
  // NULL values should be skipped (not counted).
  auto vectors = {makeRowVector({makeNullableFlatVector<int64_t>(
      {1, std::nullopt, 2, std::nullopt, 3})})};

  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, 0.5, 0.5, 1)"})
          .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultFlat = result->childAt(0)->asFlatVector<StringView>();
  ASSERT_FALSE(resultFlat->isNullAt(0));
  auto sv = resultFlat->valueAt(0);

  auto [depth, width, totalCount] = parseSketch(sv);
  EXPECT_EQ(totalCount, 3); // Only non-null values counted.
}

TEST_F(CountMinSketchAggregateTest, emptyInput) {
  auto vectors = {makeRowVector({makeFlatVector<int64_t>({})})};

  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, 0.5, 0.5, 1)"})
          .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultFlat = result->childAt(0)->asFlatVector<StringView>();
  // Spark's count_min_sketch is non-nullable: empty input produces a
  // serialized empty sketch, not null.
  ASSERT_FALSE(resultFlat->isNullAt(0));
  auto sv = resultFlat->valueAt(0);

  auto [depth, width, totalCount] = parseSketch(sv);
  EXPECT_EQ(depth, 1);
  EXPECT_EQ(width, 4);
  EXPECT_EQ(totalCount, 0);
}

TEST_F(CountMinSketchAggregateTest, emptyInputPartialToFinal) {
  // The empty sketch must also survive a partial->final split: the partial
  // step emits an empty sketch intermediate (dimensions known via the constant
  // arguments), which the final step merges back into an empty sketch.
  auto vectors = {makeRowVector({makeFlatVector<int64_t>({})})};

  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, 0.5, 0.5, 1)"})
          .planNode();
  auto expected = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  testAggregations(
      vectors, {}, {"count_min_sketch(c0, 0.5, 0.5, 1)"}, {expected});
}

TEST_F(CountMinSketchAggregateTest, groupBy) {
  auto vectors = {makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2, 2}), // grouping key
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}), // values
  })};

  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({"c0"}, {"count_min_sketch(c1, 0.5, 0.5, 1)"})
          .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  ASSERT_EQ(result->size(), 2);
  auto resultFlat = result->childAt(1)->asFlatVector<StringView>();

  for (vector_size_t i = 0; i < 2; ++i) {
    ASSERT_FALSE(resultFlat->isNullAt(i));
    auto sv = resultFlat->valueAt(i);
    auto [depth, width, totalCount] = parseSketch(sv);
    EXPECT_EQ(depth, 1);
    EXPECT_EQ(width, 4);
    // Group 1 has 2 values, group 2 has 3 values.
    // But we don't know the order, so just check both are valid.
    EXPECT_TRUE(totalCount == 2 || totalCount == 3);
  }
}

TEST_F(CountMinSketchAggregateTest, differentParameters) {
  // Test with different eps/confidence producing different sketch dimensions.
  auto vectors = {makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4, 5})})};

  // eps=0.1, confidence=0.99, seed=7
  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, 0.1, 0.99, 7)"})
          .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  auto resultFlat = result->childAt(0)->asFlatVector<StringView>();
  ASSERT_FALSE(resultFlat->isNullAt(0));
  auto sv = resultFlat->valueAt(0);
  auto [depth, width, totalCount] = parseSketch(sv);

  EXPECT_EQ(width, 20); // ceil(2/0.1)
  EXPECT_EQ(depth, 7); // ceil(-log1p(-0.99)/log(2))
  EXPECT_EQ(totalCount, 5);
}

TEST_F(CountMinSketchAggregateTest, bigintSeedInput) {
  // Integer literals are parsed as bigint by default; verify bigint seed works.
  auto vectors = {makeRowVector({makeFlatVector<int64_t>({1, 2, 3})})};

  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, 0.5, 0.5, 1)"})
          .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  auto resultFlat = result->childAt(0)->asFlatVector<StringView>();
  ASSERT_FALSE(resultFlat->isNullAt(0));
  auto sv = resultFlat->valueAt(0);
  auto [depth, width, totalCount] = parseSketch(sv);
  EXPECT_EQ(totalCount, 3);
}

TEST_F(CountMinSketchAggregateTest, partialToFinal) {
  // Verify partial->final aggregation produces correct results using
  // testAggregations which tests multiple plans.
  auto vectors = {makeRowVector({makeFlatVector<int64_t>({1, 2, 1})})};

  // Build expected result using single aggregation.
  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, 0.5, 0.5, 1)"})
          .planNode();
  auto expected = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  testAggregations(
      vectors, {}, {"count_min_sketch(c0, 0.5, 0.5, 1)"}, {expected});
}

TEST_F(CountMinSketchAggregateTest, nullParametersRejected) {
  // eps, confidence and seed must be non-null constants. A null constant is
  // rejected at initialization via setConstantInputs.
  std::vector<RowVectorPtr> data = {
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3})})};

  testFailingAggregations(
      data,
      {},
      {"count_min_sketch(c0, cast(null as double), 0.5, 1)"},
      "eps argument must not be null");
  testFailingAggregations(
      data,
      {},
      {"count_min_sketch(c0, 0.5, cast(null as double), 1)"},
      "confidence argument must not be null");
  testFailingAggregations(
      data,
      {},
      {"count_min_sketch(c0, 0.5, 0.5, cast(null as integer))"},
      "seed argument must not be null");
}

TEST_F(CountMinSketchAggregateTest, invalidParametersRejected) {
  // Out-of-range constants are rejected with a user error rather than
  // triggering undefined behavior or a division by zero.
  std::vector<RowVectorPtr> data = {
      makeRowVector({makeFlatVector<int64_t>({1, 2, 3})})};

  // Non-positive eps.
  testFailingAggregations(
      data, {}, {"count_min_sketch(c0, -0.5, 0.5, 1)"}, "eps must be positive");

  // eps so small that the derived width overflows int32.
  testFailingAggregations(
      data,
      {},
      {"count_min_sketch(c0, 1.0e-300, 0.5, 1)"},
      "count_min_sketch width out of range");

  // Confidence outside (0, 1).
  testFailingAggregations(
      data,
      {},
      {"count_min_sketch(c0, 0.5, 1.5, 1)"},
      "confidence must be less than 1.0");
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
