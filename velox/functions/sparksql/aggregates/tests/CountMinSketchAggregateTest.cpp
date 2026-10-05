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

#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <string>

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
  return std::bit_cast<int64_t>(v);
}

inline int32_t readBE32(const char*& buf) {
  uint32_t v = (static_cast<uint32_t>(static_cast<uint8_t>(buf[0])) << 24) |
      (static_cast<uint32_t>(static_cast<uint8_t>(buf[1])) << 16) |
      (static_cast<uint32_t>(static_cast<uint8_t>(buf[2])) << 8) |
      static_cast<uint32_t>(static_cast<uint8_t>(buf[3]));
  buf += 4;
  return static_cast<int32_t>(v);
}

void writeBE64(std::string& data, size_t offset, int64_t value) {
  uint64_t v = static_cast<uint64_t>(value);
  for (int32_t i = 0; i < 8; ++i) {
    data[offset + i] = static_cast<char>((v >> (56 - i * 8)) & uint64_t{0xFF});
  }
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

  // Serializes a global count_min_sketch(eps=0.5, confidence=0.5, seed=1) over
  // the single-column input and returns the raw sketch bytes. Used to assert
  // byte-for-byte parity of the narrower value types against bigint.
  std::string serializeSketch(const VectorPtr& column) {
    return serializeSketch(column, "count_min_sketch(c0, 0.5, 0.5, 1)");
  }

  // Serializes a sketch using the specified aggregate expression.
  std::string serializeSketch(
      const VectorPtr& column,
      const std::string& expression) {
    auto vectors = {makeRowVector({column})};
    auto planNode = exec::test::PlanBuilder(pool())
                        .values(vectors)
                        .singleAggregation({}, {expression})
                        .planNode();
    auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());
    EXPECT_EQ(result->size(), 1);
    auto resultFlat = result->childAt(0)->asFlatVector<StringView>();
    EXPECT_FALSE(resultFlat->isNullAt(0));
    auto sv = resultFlat->valueAt(0);
    return std::string(sv.data(), sv.size());
  }

  // Verifies that merging the supplied serialized sketches fails.
  void assertMergeFails(
      const std::vector<std::string>& sketches,
      const std::string& expectedError) {
    std::vector<StringView> views;
    views.reserve(sketches.size());
    for (const auto& sketch : sketches) {
      views.emplace_back(sketch);
    }

    auto input = makeFlatVector<StringView>(views, VARBINARY());
    auto planNode =
        exec::test::PlanBuilder(pool())
            .values({makeRowVector({input})})
            .singleAggregation(
                {}, {"count_min_sketch_merge_extract_varbinary(c0)"})
            .planNode();
    VELOX_ASSERT_THROW(
        exec::test::AssertQueryBuilder(planNode).copyResults(pool()),
        expectedError);
  }

  // Returns the uppercase hex encoding of a serialized sketch, matching the
  // format of Spark's hex(count_min_sketch(...)) reference output.
  std::string toHex(const StringView& sv) {
    std::string hex;
    for (size_t i = 0; i < sv.size(); ++i) {
      char byteHex[3];
      snprintf(
          byteHex, sizeof(byteHex), "%02X", static_cast<uint8_t>(sv.data()[i]));
      hex += byteHex;
    }
    return hex;
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

TEST_F(CountMinSketchAggregateTest, tinyintInput) {
  auto vectors = {makeRowVector({makeFlatVector<int8_t>({1, 2, 3, 4, 5})})};

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
  EXPECT_EQ(totalCount, 5);
}

TEST_F(CountMinSketchAggregateTest, integralTypeSignExtension) {
  // The tinyint/smallint/integer branches widen the value to int64 before
  // hashing, exactly as Spark widens TINYINT / SMALLINT / INT to long. A sketch
  // over the narrow type must therefore be byte-identical to the bigint sketch
  // of the same values. Cover negatives and the type min/max, where a missing
  // or wrong sign extension would diverge.
  EXPECT_EQ(
      serializeSketch(
          makeFlatVector<int8_t>(
              {int8_t(-128), int8_t(127), int8_t(-1), int8_t(0)})),
      serializeSketch(makeFlatVector<int64_t>({-128, 127, -1, 0})));

  EXPECT_EQ(
      serializeSketch(
          makeFlatVector<int16_t>(
              {int16_t(-32768), int16_t(32767), int16_t(-1), int16_t(0)})),
      serializeSketch(makeFlatVector<int64_t>({-32768, 32767, -1, 0})));

  EXPECT_EQ(
      serializeSketch(
          makeFlatVector<int32_t>(
              {std::numeric_limits<int32_t>::min(),
               std::numeric_limits<int32_t>::max(),
               -1,
               0})),
      serializeSketch(
          makeFlatVector<int64_t>(
              {std::numeric_limits<int32_t>::min(),
               std::numeric_limits<int32_t>::max(),
               -1,
               0})));
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

TEST_F(CountMinSketchAggregateTest, varcharGoldenParity) {
  // Byte-for-byte parity of the string (addBinary / murmurHash3) path against a
  // reference sketch produced by Spark's own
  // org.apache.spark.util.sketch.CountMinSketch (spark-sketch 4.1.1):
  //   CountMinSketch.create(0.5, 0.5, 42);
  //   addString("hello"); addString("world"); addString("hello");
  //   .toByteArray()
  // This locks Spark's hashUnsafeBytes murmur3 (signed-byte tail mixing),
  // double hashing and Math.abs(hash % width) bucketing, complementing the
  // long-path golden covered by the bigintInput test.
  auto vectors = {
      makeRowVector({makeFlatVector<StringView>({"hello", "world", "hello"})})};

  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, 0.5, 0.5, 42)"})
          .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultFlat = result->childAt(0)->asFlatVector<StringView>();
  ASSERT_FALSE(resultFlat->isNullAt(0));
  auto sv = resultFlat->valueAt(0);

  std::string expectedHex =
      "000000010000000000000003000000010000000400000000"
      "5D20CE9A00000000000000020000000000000000000000000000000100000000"
      "00000000";
  EXPECT_EQ(toHex(sv), expectedHex);
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

TEST_F(CountMinSketchAggregateTest, varbinaryMatchesVarchar) {
  // The varbinary branch shares the addBinary / murmurHash3 path with varchar,
  // so hashing the same raw bytes must yield a byte-identical sketch.
  std::vector<std::string> raw = {
      std::string("\x00\x01\x80\xFF", 4), std::string("hi", 2)};
  auto varbinary = makeFlatVector<StringView>(
      {StringView(raw[0]), StringView(raw[1])}, VARBINARY());
  auto varchar =
      makeFlatVector<StringView>({StringView(raw[0]), StringView(raw[1])});

  EXPECT_EQ(serializeSketch(varbinary), serializeSketch(varchar));
}

TEST_F(CountMinSketchAggregateTest, varbinarySignedHighByteTailGolden) {
  // Byte-for-byte parity against a Spark 4.1.1 CountMinSketch generated with:
  //   CountMinSketch.create(0.5, 0.5, 1)
  //   addBinary([0x80])
  //   addBinary([0x01, 0x80])
  //   addBinary([0x01, 0x02, 0xFF])
  // The one-, two-, and three-byte Murmur3 tails all contain a signed high
  // byte, locking Spark's signed-byte promotion behavior.
  std::vector<std::string> raw = {
      std::string("\x80", 1),
      std::string("\x01\x80", 2),
      std::string("\x01\x02\xFF", 3)};
  auto varbinary = makeFlatVector<StringView>(
      {StringView(raw[0]), StringView(raw[1]), StringView(raw[2])},
      VARBINARY());

  std::string expectedHex =
      "000000010000000000000003000000010000000400000000"
      "5D8D6AB900000000000000000000000000000000000000000000000200000000"
      "00000001";
  auto serialized = serializeSketch(varbinary);
  EXPECT_EQ(toHex(StringView(serialized)), expectedHex);
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

TEST_F(CountMinSketchAggregateTest, emptyInputCompanionFunctions) {
  auto vectors = {makeRowVector({makeFlatVector<int64_t>({})})};

  auto partialPlan =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch_partial(c0, 0.5, 0.5, 1)"})
          .planNode();
  auto partial =
      exec::test::AssertQueryBuilder(partialPlan).copyResults(pool());

  ASSERT_EQ(partial->size(), 1);
  auto partialFlat = partial->childAt(0)->asFlatVector<StringView>();
  ASSERT_FALSE(partialFlat->isNullAt(0));
  auto [partialDepth, partialWidth, partialTotalCount] =
      parseSketch(partialFlat->valueAt(0));
  EXPECT_EQ(partialDepth, 1);
  EXPECT_EQ(partialWidth, 4);
  EXPECT_EQ(partialTotalCount, 0);

  auto finalPlan = exec::test::PlanBuilder(pool())
                       .values({partial})
                       .singleAggregation(
                           {}, {"count_min_sketch_merge_extract_varbinary(a0)"})
                       .planNode();
  auto finalResult =
      exec::test::AssertQueryBuilder(finalPlan).copyResults(pool());

  ASSERT_EQ(finalResult->size(), 1);
  auto finalFlat = finalResult->childAt(0)->asFlatVector<StringView>();
  ASSERT_FALSE(finalFlat->isNullAt(0));
  auto [finalDepth, finalWidth, finalTotalCount] =
      parseSketch(finalFlat->valueAt(0));
  EXPECT_EQ(finalDepth, 1);
  EXPECT_EQ(finalWidth, 4);
  EXPECT_EQ(finalTotalCount, 0);
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
  auto keyFlat = result->childAt(0)->asFlatVector<int32_t>();
  auto resultFlat = result->childAt(1)->asFlatVector<StringView>();
  std::map<int32_t, int64_t> counts;

  for (vector_size_t i = 0; i < 2; ++i) {
    ASSERT_FALSE(resultFlat->isNullAt(i));
    auto sv = resultFlat->valueAt(i);
    auto [depth, width, totalCount] = parseSketch(sv);
    EXPECT_EQ(depth, 1);
    EXPECT_EQ(width, 4);
    ASSERT_TRUE(counts.emplace(keyFlat->valueAt(i), totalCount).second);
  }
  EXPECT_EQ(counts, (std::map<int32_t, int64_t>{{1, 2}, {2, 3}}));
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

TEST_F(CountMinSketchAggregateTest, seedInputTypes) {
  auto vectors = {makeRowVector({makeFlatVector<int64_t>({1, 2, 3})})};

  auto planNode = exec::test::PlanBuilder(pool())
                      .values(vectors)
                      .project(
                          {"c0",
                           "cast(1 as integer) as integer_seed",
                           "cast(1 as bigint) as bigint_seed"})
                      .singleAggregation(
                          {},
                          {
                              "count_min_sketch(c0, 0.5, 0.5, integer_seed)",
                              "count_min_sketch(c0, 0.5, 0.5, bigint_seed)",
                          })
                      .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto integerSeed = result->childAt(0)->asFlatVector<StringView>();
  auto bigintSeed = result->childAt(1)->asFlatVector<StringView>();
  ASSERT_FALSE(integerSeed->isNullAt(0));
  ASSERT_FALSE(bigintSeed->isNullAt(0));
  EXPECT_EQ(toHex(integerSeed->valueAt(0)), toHex(bigintSeed->valueAt(0)));
}

TEST_F(CountMinSketchAggregateTest, flatConstantParameters) {
  auto values = makeFlatVector<int64_t>({1, 2, 1});
  auto vectors = {makeRowVector(
      {values,
       makeFlatVector<double>({0.5, 0.5, 0.5}),
       makeFlatVector<double>({0.5, 0.5, 0.5}),
       makeFlatVector<int32_t>({1, 1, 1})})};

  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, c1, c2, c3)"})
          .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());
  auto resultFlat = result->childAt(0)->asFlatVector<StringView>();
  auto resultValue = resultFlat->valueAt(0);

  EXPECT_EQ(
      std::string(resultValue.data(), resultValue.size()),
      serializeSketch(values));
}

TEST_F(CountMinSketchAggregateTest, changedParametersAcrossBatchesRejected) {
  auto vectors = {
      makeRowVector(
          {makeFlatVector<int64_t>({1, 2}),
           makeFlatVector<double>({0.5, 0.5}),
           makeFlatVector<double>({0.5, 0.5}),
           makeFlatVector<int32_t>({1, 1})}),
      makeRowVector(
          {makeFlatVector<int64_t>({3, 4}),
           makeFlatVector<double>({0.25, 0.25}),
           makeFlatVector<double>({0.5, 0.5}),
           makeFlatVector<int32_t>({1, 1})}),
  };

  auto planNode =
      exec::test::PlanBuilder(pool())
          .values(vectors)
          .singleAggregation({}, {"count_min_sketch(c0, c1, c2, c3)"})
          .planNode();
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(planNode).copyResults(pool()),
      "eps argument must be constant for all input rows");
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

TEST_F(CountMinSketchAggregateTest, mergeCounterOverflowWraps) {
  // Spark uses Java long arithmetic for counters, which wraps at 64 bits.
  auto emptySketch = serializeSketch(makeFlatVector<int64_t>({}));
  auto maxSketch = emptySketch;
  auto oneSketch = emptySketch;

  // Version(4), totalCount(8), depth(4), width(4), hashA(8), table.
  constexpr size_t kTotalCountOffset = 4;
  constexpr size_t kFirstTableCountOffset = 28;
  writeBE64(maxSketch, kTotalCountOffset, std::numeric_limits<int64_t>::max());
  writeBE64(
      maxSketch, kFirstTableCountOffset, std::numeric_limits<int64_t>::max());
  writeBE64(oneSketch, kTotalCountOffset, 1);
  writeBE64(oneSketch, kFirstTableCountOffset, 1);

  auto input = makeFlatVector<StringView>(
      {StringView(maxSketch), StringView(oneSketch)}, VARBINARY());
  auto planNode = exec::test::PlanBuilder(pool())
                      .values({makeRowVector({input})})
                      .singleAggregation(
                          {}, {"count_min_sketch_merge_extract_varbinary(c0)"})
                      .planNode();
  auto result = exec::test::AssertQueryBuilder(planNode).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto resultFlat = result->childAt(0)->asFlatVector<StringView>();
  ASSERT_FALSE(resultFlat->isNullAt(0));
  auto serialized = resultFlat->valueAt(0);
  auto [depth, width, totalCount] = parseSketch(serialized);
  EXPECT_EQ(depth, 1);
  EXPECT_EQ(width, 4);
  EXPECT_EQ(totalCount, std::numeric_limits<int64_t>::min());

  const char* firstTableCount = serialized.data() + kFirstTableCountOffset;
  EXPECT_EQ(readBE64(firstTableCount), std::numeric_limits<int64_t>::min());
}

TEST_F(CountMinSketchAggregateTest, malformedMergeInputRejected) {
  auto values = makeFlatVector<int64_t>({1, 2, 3});
  auto sketch = serializeSketch(values);

  auto truncatedHeader = sketch.substr(0, 19);
  assertMergeFails(
      {truncatedHeader}, "CountMinSketch serialized data too small");

  auto truncatedTable = sketch.substr(0, sketch.size() - 1);
  assertMergeFails(
      {truncatedTable}, "CountMinSketch serialized data truncated");
}

TEST_F(CountMinSketchAggregateTest, incompatibleMergeInputRejected) {
  auto values = makeFlatVector<int64_t>({1, 2, 3});
  auto sketch = serializeSketch(values);
  auto differentDepth =
      serializeSketch(values, "count_min_sketch(c0, 0.5, 0.75, 1)");
  auto differentWidth =
      serializeSketch(values, "count_min_sketch(c0, 0.25, 0.5, 1)");
  auto differentHashSeed =
      serializeSketch(values, "count_min_sketch(c0, 0.5, 0.5, 2)");

  assertMergeFails(
      {sketch, differentDepth},
      "Cannot merge CountMinSketch of different depth");
  assertMergeFails(
      {sketch, differentWidth},
      "Cannot merge CountMinSketch of different width");
  assertMergeFails(
      {sketch, differentHashSeed},
      "Cannot merge CountMinSketch with different hash seeds");
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

  // eps that produces a valid width but an output larger than StringView can
  // represent.
  testFailingAggregations(
      data,
      {},
      {"count_min_sketch(c0, 7.450580596923828e-9, 0.5, 1)"},
      "count_min_sketch serialized size out of range");

  // Confidence outside (0, 1).
  testFailingAggregations(
      data,
      {},
      {"count_min_sketch(c0, 0.5, 1.5, 1)"},
      "confidence must be less than 1.0");
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
