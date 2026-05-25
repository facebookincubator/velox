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

#include "velox/functions/sparksql/aggregates/BitmapConstructAggAggregate.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"
#include "velox/functions/sparksql/aggregates/tests/BitmapBuilder.h"

namespace facebook::velox::functions::aggregate::sparksql::test {
namespace {

class BitmapConstructAggAggregateTest
    : public aggregate::test::AggregationTestBase {
 public:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerAggregateFunctions("");
  }

  // Returns a single-row VARBINARY vector containing the bitmap.
  VectorPtr makeBitmapWithBits(const std::vector<int64_t>& positions) {
    return makeFlatVector({BitmapBuilder::fromBits(positions)}, VARBINARY());
  }
};

TEST_F(BitmapConstructAggAggregateTest, basic) {
  // Single group, several bit positions.
  auto input = makeRowVector({makeFlatVector<int64_t>({0, 1, 7, 8, 100})});
  auto expected = makeBitmapWithBits({0, 1, 7, 8, 100});

  testAggregations(
      {input}, {}, {"bitmap_construct_agg(c0)"}, {makeRowVector({expected})});
}

TEST_F(BitmapConstructAggAggregateTest, allNullInputReturnsZeroBitmap) {
  // Spark contract: nullable = false, so all-null input produces all-zeros.
  auto input = makeRowVector({makeNullableFlatVector<int64_t>(
      {std::nullopt, std::nullopt, std::nullopt})});
  auto expected = makeBitmapWithBits({});

  testAggregations(
      {input}, {}, {"bitmap_construct_agg(c0)"}, {makeRowVector({expected})});
}

TEST_F(BitmapConstructAggAggregateTest, emptyInputReturnsZeroBitmap) {
  auto input = makeRowVector({makeFlatVector<int64_t>({})});
  auto expected = makeBitmapWithBits({});

  testAggregations(
      {input}, {}, {"bitmap_construct_agg(c0)"}, {makeRowVector({expected})});
}

TEST_F(BitmapConstructAggAggregateTest, mixedNullAndValid) {
  // Mix of null and valid values.
  auto input = makeRowVector({makeNullableFlatVector<int64_t>(
      {std::nullopt, 5, std::nullopt, 10, std::nullopt})});
  auto expected = makeBitmapWithBits({5, 10});

  testAggregations(
      {input}, {}, {"bitmap_construct_agg(c0)"}, {makeRowVector({expected})});
}

TEST_F(BitmapConstructAggAggregateTest, groupBy) {
  // Test with group-by: two groups, including one all-null group.
  auto input = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2, 2}),
      makeNullableFlatVector<int64_t>(
          {0, 100, std::nullopt, std::nullopt, std::nullopt}),
  });

  // Group 1: bits 0 and 100 set. Group 2: all-zeros (all-null input).
  auto bitmap1 = BitmapBuilder::fromBits({0, 100});
  auto bitmap2 = BitmapBuilder::fromBits({});
  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector({bitmap1, bitmap2}, VARBINARY()),
  });

  testAggregations(
      {input}, {"c0"}, {"bitmap_construct_agg(c1)"}, {}, {expected});
}

TEST_F(BitmapConstructAggAggregateTest, boundaryPositions) {
  // Test min and max valid positions.
  auto input = makeRowVector({makeFlatVector<int64_t>({0, 32767})});
  auto expected = makeBitmapWithBits({0, 32767});

  testAggregations(
      {input}, {}, {"bitmap_construct_agg(c0)"}, {makeRowVector({expected})});
}

TEST_F(BitmapConstructAggAggregateTest, invalidPositionNegative) {
  auto input = makeRowVector({makeFlatVector<int64_t>({-1})});
  VELOX_ASSERT_THROW(
      testAggregations(
          {input},
          {},
          {"bitmap_construct_agg(c0)"},
          std::vector<RowVectorPtr>{}),
      "Bitmap position out of bounds");
}

TEST_F(BitmapConstructAggAggregateTest, invalidPositionTooLarge) {
  auto input = makeRowVector({makeFlatVector<int64_t>({32768})});
  VELOX_ASSERT_THROW(
      testAggregations(
          {input},
          {},
          {"bitmap_construct_agg(c0)"},
          std::vector<RowVectorPtr>{}),
      "Bitmap position out of bounds");
}

TEST_F(BitmapConstructAggAggregateTest, duplicatePositions) {
  // Duplicate positions should result in same bitmap as unique.
  auto input = makeRowVector({makeFlatVector<int64_t>({5, 5, 5, 10, 10})});
  auto expected = makeBitmapWithBits({5, 10});

  testAggregations(
      {input}, {}, {"bitmap_construct_agg(c0)"}, {makeRowVector({expected})});
}

TEST_F(BitmapConstructAggAggregateTest, partialAndFinalAggregation) {
  // Test that partial -> final merge works correctly.
  // Split data across multiple batches to exercise intermediate merge.
  auto batch1 = makeRowVector({makeFlatVector<int64_t>({0, 1, 2})});
  auto batch2 = makeRowVector({makeFlatVector<int64_t>({3, 4, 5})});
  auto expected = makeBitmapWithBits({0, 1, 2, 3, 4, 5});

  testAggregations(
      {batch1, batch2},
      {},
      {"bitmap_construct_agg(c0)"},
      {makeRowVector({expected})});
}

TEST_F(BitmapConstructAggAggregateTest, invalidIntermediateSizeTooSmall) {
  std::string tinyBitmap(1, '\xFF');
  auto input = makeRowVector(
      {makeFlatVector({tinyBitmap}, VARBINARY())});

  auto plan =
      exec::test::PlanBuilder()
          .values({input})
          .finalAggregation({}, {"bitmap_construct_agg(c0)"}, {{BIGINT()}})
          .planNode();

  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan).copyResults(pool()),
      "Unexpected intermediate bitmap size");
}

TEST_F(BitmapConstructAggAggregateTest, invalidIntermediateSizeTooLarge) {
  std::string bigBitmap(8192, '\x00');
  auto input = makeRowVector(
      {makeFlatVector({bigBitmap}, VARBINARY())});

  auto plan =
      exec::test::PlanBuilder()
          .values({input})
          .finalAggregation({}, {"bitmap_construct_agg(c0)"}, {{BIGINT()}})
          .planNode();

  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan).copyResults(pool()),
      "Unexpected intermediate bitmap size");
}

TEST_F(BitmapConstructAggAggregateTest, nullIntermediateInFinalAggregation) {
  // A null intermediate should be skipped; the result should still be a valid
  // all-zeros bitmap (non-null output).
  auto input = makeRowVector(
      {makeNullableFlatVector<StringView>({std::nullopt}, VARBINARY())});

  auto plan =
      exec::test::PlanBuilder()
          .values({input})
          .finalAggregation({}, {"bitmap_construct_agg(c0)"}, {{BIGINT()}})
          .planNode();

  auto expected = makeRowVector({makeBitmapWithBits({})});
  exec::test::AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(BitmapConstructAggAggregateTest, mergeMultipleIntermediates) {
  // Two batches with disjoint positions exercise the 64-bit OR combine logic.
  auto batch1 = makeRowVector({makeFlatVector<int64_t>({0})});
  auto batch2 = makeRowVector({makeFlatVector<int64_t>({15})});
  auto expected = makeBitmapWithBits({0, 15});

  testAggregations(
      {batch1, batch2},
      {},
      {"bitmap_construct_agg(c0)"},
      {makeRowVector({expected})});
}

TEST_F(BitmapConstructAggAggregateTest, mergeOverlappingIntermediates) {
  // Two batches with overlapping positions — OR produces union.
  auto batch1 = makeRowVector({makeFlatVector<int64_t>({0})});
  auto batch2 = makeRowVector({makeFlatVector<int64_t>({0, 1})});
  auto expected = makeBitmapWithBits({0, 1});

  testAggregations(
      {batch1, batch2},
      {},
      {"bitmap_construct_agg(c0)"},
      {makeRowVector({expected})});
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
