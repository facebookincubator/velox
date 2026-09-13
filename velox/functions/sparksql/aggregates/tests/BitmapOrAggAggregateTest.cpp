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

#include <cstring>

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
  auto input = makeRowVector({BitmapBuilder::vectorFromBytes(
      pool(), {{{0, 0xF0}, {1, 0x0F}}, {{0, 0x0F}, {2, 0xFF}}})});
  auto expectedResult = makeRowVector({BitmapBuilder::vectorFromBytes(
      pool(), {{{0, 0xFF}, {1, 0x0F}, {2, 0xFF}}})});

  testAggregations({input}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, allNullInputs) {
  auto input =
      makeRowVector({BaseVector::createNullConstant(VARBINARY(), 3, pool())});

  auto expectedResult =
      makeRowVector({BitmapBuilder::vectorFromBits(pool(), {{}})});

  testAggregations({input}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, emptyInput) {
  auto input = makeRowVector({BitmapBuilder::vectorFromBits(pool(), {})});

  auto expectedResult =
      makeRowVector({BitmapBuilder::vectorFromBits(pool(), {{}})});

  testAggregations({input}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, mixedNullAndNonNull) {
  auto bitmap = BitmapBuilder::fromBytes({{0, 0xAA}});

  auto input = makeRowVector({makeNullableFlatVector<StringView>(
      {StringView(bitmap), std::nullopt}, VARBINARY())});
  auto expectedResult =
      makeRowVector({BitmapBuilder::vectorFromBytes(pool(), {{{0, 0xAA}}})});

  testAggregations({input}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, groupBy) {
  auto input = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2}),
      BitmapBuilder::vectorFromBytes(
          pool(), {{{0, 0xF0}}, {{0, 0x0F}}, {{1, 0xFF}}}),
  });

  auto expectedResult = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      BitmapBuilder::vectorFromBytes(pool(), {{{0, 0xFF}}, {{1, 0xFF}}}),
  });

  testAggregations({input}, {"c0"}, {"bitmap_or_agg(c1)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, variableLengthInput) {
  // Spark's bitmap_or_agg accepts bitmaps shorter than 4096 bytes (e.g. from
  // to_binary('...', 'hex')); only the provided bytes are merged and the rest
  // of the 4096-byte accumulator stays zero. Matches
  // BitmapExpressionUtils.bitmapMerge, which ORs min(input, buffer) bytes.
  auto shortBitmap = std::string(100, '\x01');
  auto input = makeRowVector({makeFlatVector({shortBitmap}, VARBINARY())});

  std::string expectedBitmap(kBitmapNumBytes, '\0');
  std::memcpy(expectedBitmap.data(), shortBitmap.data(), shortBitmap.size());
  auto expectedResult =
      makeRowVector({makeFlatVector({expectedBitmap}, VARBINARY())});

  testAggregations({input}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, overLengthInput) {
  // Inputs longer than 4096 bytes contribute only their first 4096 bytes; the
  // remainder is ignored, matching Spark's min(input, buffer) merge.
  auto longBitmap =
      std::string(kBitmapNumBytes, '\x0F') + std::string(1000, '\xFF');
  auto input = makeRowVector({makeFlatVector({longBitmap}, VARBINARY())});

  auto expectedBitmap = std::string(kBitmapNumBytes, '\x0F');
  auto expectedResult =
      makeRowVector({makeFlatVector({expectedBitmap}, VARBINARY())});

  testAggregations({input}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, emptyBitmapAccepted) {
  // A zero-length VARBINARY contributes no bits; the group still yields the
  // non-null all-zeros bitmap.
  auto emptyBitmap = std::string();
  auto input = makeRowVector({makeFlatVector({emptyBitmap}, VARBINARY())});

  auto expectedResult =
      makeRowVector({BitmapBuilder::vectorFromBits(pool(), {{}})});

  testAggregations({input}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, singleByteInput) {
  // Guards the exact regression: a 1-byte input such as to_binary('10', 'hex')
  // must merge rather than throw. Exercises the tail-only byte path (size < 8).
  auto oneByte = std::string(1, '\x10');
  auto input = makeRowVector({makeFlatVector({oneByte}, VARBINARY())});

  std::string expectedBitmap(kBitmapNumBytes, '\0');
  expectedBitmap[0] = '\x10';
  auto expectedResult =
      makeRowVector({makeFlatVector({expectedBitmap}, VARBINARY())});

  testAggregations({input}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, mixedLengthInputs) {
  // Bitmaps of different lengths in the same group must OR together without
  // clobbering bytes contributed by other rows.
  std::string shortA(3, '\0');
  shortA[0] = static_cast<char>(0xF0);
  shortA[2] = static_cast<char>(0x0F);
  std::string full = BitmapBuilder::fromBytes({{1, 0x0F}, {500, 0xFF}});
  std::string shortB(2, '\0');
  shortB[1] = static_cast<char>(0xFF);

  auto input =
      makeRowVector({makeFlatVector({shortA, full, shortB}, VARBINARY())});

  std::string expectedBitmap(kBitmapNumBytes, '\0');
  expectedBitmap[0] = static_cast<char>(0xF0);
  expectedBitmap[1] = static_cast<char>(0xFF);
  expectedBitmap[2] = static_cast<char>(0x0F);
  expectedBitmap[500] = static_cast<char>(0xFF);
  auto expectedResult =
      makeRowVector({makeFlatVector({expectedBitmap}, VARBINARY())});

  testAggregations({input}, {}, {"bitmap_or_agg(c0)"}, {expectedResult});
}

TEST_F(BitmapOrAggAggregateTest, mergeIntermediate) {
  // Two batches exercise partial -> final aggregation merge.
  auto batch1 = makeRowVector(
      {BitmapBuilder::vectorFromBytes(pool(), {{{0, 0xAA}, {100, 0x55}}})});
  auto batch2 = makeRowVector(
      {BitmapBuilder::vectorFromBytes(pool(), {{{0, 0x55}, {200, 0xFF}}})});
  auto expectedResult = makeRowVector({BitmapBuilder::vectorFromBytes(
      pool(), {{{0, 0xFF}, {100, 0x55}, {200, 0xFF}}})});

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
  auto input =
      makeRowVector({BaseVector::createNullConstant(VARBINARY(), 1, pool())});

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
