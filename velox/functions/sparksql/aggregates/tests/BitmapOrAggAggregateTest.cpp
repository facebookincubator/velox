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
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/BitmapConstructAggAggregate.h"
#include "velox/functions/sparksql/aggregates/Register.h"
#include "velox/functions/sparksql/aggregates/tests/BitmapBuilder.h"

namespace facebook::velox::functions::aggregate::sparksql::test {
namespace {

class BitmapOrAggAggregateTest : public aggregate::test::AggregationTestBase {
 public:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerAggregateFunctions("");
  }
};

TEST_F(BitmapOrAggAggregateTest, basic) {
  auto lowBytes = BitmapBuilder::fromBytes({{0, 0xF0}, {1, 0x0F}});
  auto highByte = BitmapBuilder::fromBytes({{0, 0x0F}, {2, 0xFF}});
  auto merged = BitmapBuilder::fromBytes({{0, 0xFF}, {1, 0x0F}, {2, 0xFF}});

  auto input =
      makeRowVector({BitmapBuilder::vector(pool(), {lowBytes, highByte})});
  auto expectedResult =
      makeRowVector({BitmapBuilder::vector(pool(), {merged})});

  testAggregations({input}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, allNullInputs) {
  auto input = makeRowVector({makeNullableFlatVector<StringView>(
      {std::nullopt, std::nullopt, std::nullopt}, VARBINARY())});

  auto expectedResult =
      makeRowVector({BitmapBuilder::vectorFromBits(pool(), {{}})});

  testAggregations({input}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, emptyInput) {
  auto input = makeRowVector(
      {BitmapBuilder::vector(pool(), std::vector<std::string>{})});

  auto expectedResult =
      makeRowVector({BitmapBuilder::vectorFromBits(pool(), {{}})});

  testAggregations({input}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, mixedNullAndNonNull) {
  auto bitmap = BitmapBuilder::fromBytes({{0, 0xAA}});

  auto input = makeRowVector({makeNullableFlatVector<StringView>(
      {StringView(bitmap), std::nullopt}, VARBINARY())});
  auto expectedResult =
      makeRowVector({BitmapBuilder::vector(pool(), {bitmap})});

  testAggregations({input}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, groupBy) {
  auto groupOneFirst = BitmapBuilder::fromBytes({{0, 0xF0}});
  auto groupOneSecond = BitmapBuilder::fromBytes({{0, 0x0F}});
  auto groupTwoBitmap = BitmapBuilder::fromBytes({{1, 0xFF}});

  auto input = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2}),
      BitmapBuilder::vector(
          pool(), {groupOneFirst, groupOneSecond, groupTwoBitmap}),
  });

  auto groupOneMerged = BitmapBuilder::fromBytes({{0, 0xFF}});
  auto expectedResult = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      BitmapBuilder::vector(pool(), {groupOneMerged, groupTwoBitmap}),
  });

  testAggregations({input}, {"c0"}, {"bitmap_or_agg(c1)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, invalidInputSize) {
  // Inputs not exactly 4096 bytes should trigger a check failure.
  auto shortBitmap = std::string(100, '\x01');
  auto input = makeRowVector({makeFlatVector({shortBitmap}, VARBINARY())});

  VELOX_ASSERT_THROW(
      testAggregations(
          {input}, {}, {"bitmap_or_agg(c0)"}, std::vector<RowVectorPtr>{}),
      "bitmap_or_agg expects exactly 4096 byte bitmaps");
}

TEST_F(BitmapOrAggAggregateTest, emptyBitmapRejected) {
  // A zero-length VARBINARY is invalid input, not silently skipped.
  auto emptyBitmap = std::string();
  auto input = makeRowVector({makeFlatVector({emptyBitmap}, VARBINARY())});

  VELOX_ASSERT_THROW(
      testAggregations(
          {input}, {}, {"bitmap_or_agg(c0)"}, std::vector<RowVectorPtr>{}),
      "bitmap_or_agg expects exactly 4096 byte bitmaps");
}

TEST_F(BitmapOrAggAggregateTest, mergeIntermediate) {
  // Two batches exercise partial -> final aggregation merge.
  auto firstPartial = BitmapBuilder::fromBytes({{0, 0xAA}, {100, 0x55}});
  auto secondPartial = BitmapBuilder::fromBytes({{0, 0x55}, {200, 0xFF}});
  auto merged = BitmapBuilder::fromBytes({{0, 0xFF}, {100, 0x55}, {200, 0xFF}});

  auto batch1 = makeRowVector({BitmapBuilder::vector(pool(), {firstPartial})});
  auto batch2 = makeRowVector({BitmapBuilder::vector(pool(), {secondPartial})});
  auto expectedResult =
      makeRowVector({BitmapBuilder::vector(pool(), {merged})});

  testAggregations(
      {batch1, batch2}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, invalidIntermediateSize) {
  // Invalid intermediate size should trigger a system check failure.
  std::string tinyBitmap(1, '\xFF');
  auto input = makeRowVector({makeFlatVector({tinyBitmap}, VARBINARY())});

  auto plan = exec::test::PlanBuilder()
                  .values({input})
                  .finalAggregation({}, {"bitmap_or_agg(c0)"}, {{VARBINARY()}})
                  .planNode();

  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan).copyResults(pool()),
      "Unexpected intermediate bitmap size");
}

TEST_F(BitmapOrAggAggregateTest, nullIntermediateInFinalAggregation) {
  // A null intermediate should be skipped; result is still all-zeros bitmap.
  auto input = makeRowVector(
      {makeNullableFlatVector<StringView>({std::nullopt}, VARBINARY())});

  auto plan = exec::test::PlanBuilder()
                  .values({input})
                  .finalAggregation({}, {"bitmap_or_agg(c0)"}, {{VARBINARY()}})
                  .planNode();

  auto expected = makeRowVector({BitmapBuilder::vectorFromBits(pool(), {{}})});
  exec::test::AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(BitmapOrAggAggregateTest, endToEndWithBitmapConstructAgg) {
  // Exercises the realistic workflow: bitmap_construct_agg produces bitmaps
  // that bitmap_or_agg then merges across groups.
  //
  // Simulates: SELECT bitmap_or_agg(bitmap) FROM (
  //   SELECT bucket, bitmap_construct_agg(position) as bitmap
  //   FROM data GROUP BY bucket
  // )
  auto input = makeRowVector({
      // bucket
      makeFlatVector<int32_t>({1, 1, 1, 2, 2, 2}),
      // position (bit positions to set in each bucket's bitmap)
      makeFlatVector<int64_t>({0, 7, 100, 200, 300, 7}),
  });

  // First stage: bitmap_construct_agg per bucket produces two 4096-byte
  // bitmaps. Second stage: bitmap_or_agg merges them.
  auto plan = exec::test::PlanBuilder()
                  .values({input})
                  .singleAggregation({"c0"}, {"bitmap_construct_agg(c1)"})
                  .singleAggregation({}, {"bitmap_or_agg(a0)"})
                  .planNode();

  // Expected: OR of bits {0, 7, 100} (bucket 1) and {200, 300, 7} (bucket 2)
  // = bits {0, 7, 100, 200, 300}
  auto expected = makeRowVector({makeFlatVector(
      {BitmapBuilder::fromBits({0, 7, 100, 200, 300})}, VARBINARY())});
  exec::test::AssertQueryBuilder(plan).assertResults(expected);
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
