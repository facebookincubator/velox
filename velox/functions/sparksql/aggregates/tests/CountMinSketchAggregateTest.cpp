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

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"

namespace facebook::velox::functions::aggregate::sparksql::test {
namespace {

// Big-endian helpers.
inline int64_t readBigEndianLong(const char*& buffer) {
  uint64_t unsignedValue =
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[0])) << 56) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[1])) << 48) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[2])) << 40) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[3])) << 32) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[4])) << 24) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[5])) << 16) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[6])) << 8) |
      static_cast<uint64_t>(static_cast<uint8_t>(buffer[7]));
  buffer += 8;
  return std::bit_cast<int64_t>(unsignedValue);
}

inline int32_t readBigEndianInt(const char*& buffer) {
  uint32_t unsignedValue =
      (static_cast<uint32_t>(static_cast<uint8_t>(buffer[0])) << 24) |
      (static_cast<uint32_t>(static_cast<uint8_t>(buffer[1])) << 16) |
      (static_cast<uint32_t>(static_cast<uint8_t>(buffer[2])) << 8) |
      static_cast<uint32_t>(static_cast<uint8_t>(buffer[3]));
  buffer += 4;
  return static_cast<int32_t>(unsignedValue);
}

void writeBigEndianLong(std::string& data, size_t offset, int64_t value) {
  uint64_t unsignedValue = static_cast<uint64_t>(value);
  for (int32_t i = 0; i < 8; ++i) {
    data[offset + i] =
        static_cast<char>((unsignedValue >> (56 - i * 8)) & uint64_t{0xFF});
  }
}

void writeBigEndianInt(std::string& data, size_t offset, int32_t value) {
  const auto unsignedValue = static_cast<uint32_t>(value);
  for (int32_t i = 0; i < 4; ++i) {
    data[offset + i] =
        static_cast<char>((unsignedValue >> (24 - i * 8)) & uint32_t{0xFF});
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
  std::tuple<int32_t, int32_t, int64_t> parseSketch(
      const StringView& serializedSketch) {
    const char* buffer = serializedSketch.data();
    int32_t version = readBigEndianInt(buffer);
    EXPECT_EQ(version, 1);
    int64_t totalCount = readBigEndianLong(buffer);
    int32_t depth = readBigEndianInt(buffer);
    int32_t width = readBigEndianInt(buffer);
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
    auto serializedSketch = resultFlat->valueAt(0);
    return std::string(serializedSketch.data(), serializedSketch.size());
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
  std::string toHex(const StringView& serializedSketch) {
    std::string hex;
    for (size_t i = 0; i < serializedSketch.size(); ++i) {
      char byteHex[3];
      snprintf(
          byteHex,
          sizeof(byteHex),
          "%02X",
          static_cast<uint8_t>(serializedSketch.data()[i]));
      hex += byteHex;
    }
    return hex;
  }
};

TEST_F(CountMinSketchAggregateTest, bigintInput) {
  // Match Spark's documented bigint example.
  auto serializedSketch = serializeSketch(makeFlatVector<int64_t>({1, 2, 1}));

  auto [depth, width, totalCount] = parseSketch(StringView(serializedSketch));
  EXPECT_EQ(depth, 1); // ceil(-log1p(-0.5) / log(2)) = 1
  EXPECT_EQ(width, 4); // ceil(2 / 0.5) = 4
  EXPECT_EQ(totalCount, 3);

  std::string expectedHex =
      "0000000100000000000000030000000100000004000000005D8D6AB9"
      "0000000000000000000000000000000200000000000000010000000000000000";
  EXPECT_EQ(toHex(StringView(serializedSketch)), expectedHex);

  // Match Spark 4.1.1 at depth 3 and width 7, including integral extremes and a
  // negative seed to exercise independent hash rows.
  auto extremeSketch = serializeSketch(
      makeFlatVector<int64_t>(
          {std::numeric_limits<int64_t>::min(),
           std::numeric_limits<int64_t>::max(),
           -1,
           0,
           1,
           1}),
      "count_min_sketch(c0, 0.3, 0.875, -7)");
  EXPECT_EQ(
      toHex(StringView(extremeSketch)),
      "0000000100000000000000060000000300000007000000002272954600000000"
      "736445C100000000562ACD160000000000000002000000000000000000000000"
      "0000000000000000000000000000000000000004000000000000000000000000"
      "0000000000000000000000040000000000000002000000000000000000000000"
      "0000000000000000000000000000000000000000000000000000000000000000"
      "0000000200000000000000000000000000000000000000000000000000000000"
      "0000000400000000000000000000000000000000");
}

TEST_F(CountMinSketchAggregateTest, integerInput) {
  auto serializedSketch = serializeSketch(makeFlatVector<int32_t>({1, 2, 1}));

  auto [depth, width, totalCount] = parseSketch(StringView(serializedSketch));
  EXPECT_EQ(depth, 1);
  EXPECT_EQ(width, 4);
  EXPECT_EQ(totalCount, 3);
}

TEST_F(CountMinSketchAggregateTest, smallintInput) {
  auto serializedSketch = serializeSketch(
      makeFlatVector<int16_t>({1, 2, 3, 4, 5}),
      "count_min_sketch(c0, 0.1, 0.9, 42)");

  auto [depth, width, totalCount] = parseSketch(StringView(serializedSketch));
  EXPECT_EQ(width, 20); // ceil(2 / 0.1) = 20
  EXPECT_GT(depth, 0);
  EXPECT_EQ(totalCount, 5);
}

TEST_F(CountMinSketchAggregateTest, tinyintInput) {
  auto serializedSketch =
      serializeSketch(makeFlatVector<int8_t>({1, 2, 3, 4, 5}));

  auto [depth, width, totalCount] = parseSketch(StringView(serializedSketch));
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
              {int16_t(-32'768), int16_t(32'767), int16_t(-1), int16_t(0)})),
      serializeSketch(makeFlatVector<int64_t>({-32'768, 32'767, -1, 0})));

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
  auto serializedSketch = resultFlat->valueAt(0);

  auto [depth, width, totalCount] = parseSketch(serializedSketch);
  EXPECT_EQ(totalCount, 3);
}

TEST_F(CountMinSketchAggregateTest, varcharGoldenParity) {
  // Match Spark 4.1.1 CountMinSketch.addString for one and three hash rows.
  // Multiple rows exercise the second Murmur3 hash in double hashing.
  auto column = makeFlatVector<StringView>({"hello", "world", "hello"});
  const std::vector<std::pair<std::string, std::string>> testCases = {
      {"count_min_sketch(c0, 0.5, 0.5, 42)",
       "000000010000000000000003000000010000000400000000"
       "5D20CE9A00000000000000020000000000000000000000000000000100000000"
       "00000000"},
      {"count_min_sketch(c0, 0.3, 0.875, 42)",
       "0000000100000000000000030000000300000007000000005D20CE9A00000000"
       "06FF457B000000005773DDF00000000000000000000000000000000000000000"
       "0000000000000000000000030000000000000000000000000000000000000000"
       "0000000000000000000000020000000000000001000000000000000000000000"
       "0000000000000000000000000000000000000000000000000000000000000000"
       "0000000000000000000000030000000000000000000000000000000000000000"
       "0000000000000000000000000000000000000000"},
  };
  for (const auto& [expression, expectedHex] : testCases) {
    SCOPED_TRACE(expression);
    auto serializedSketch = serializeSketch(column, expression);
    EXPECT_EQ(toHex(StringView(serializedSketch)), expectedHex);
  }
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
  auto serializedSketch = resultFlat->valueAt(0);

  auto [depth, width, totalCount] = parseSketch(serializedSketch);
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
  // Match Spark 4.1.1 CountMinSketch.addBinary for one and three hash rows.
  // The one-, two-, and three-byte Murmur3 tails contain signed high bytes.
  std::vector<std::string> raw = {
      std::string("\x80", 1),
      std::string("\x01\x80", 2),
      std::string("\x01\x02\xFF", 3)};
  auto varbinary = makeFlatVector<StringView>(
      {StringView(raw[0]), StringView(raw[1]), StringView(raw[2])},
      VARBINARY());

  const std::vector<std::pair<std::string, std::string>> testCases = {
      {"count_min_sketch(c0, 0.5, 0.5, 1)",
       "000000010000000000000003000000010000000400000000"
       "5D8D6AB900000000000000000000000000000000000000000000000200000000"
       "00000001"},
      {"count_min_sketch(c0, 0.3, 0.875, 1)",
       "0000000100000000000000030000000300000007000000005D8D6AB900000000"
       "0CDC4E6C00000000347D87370000000000000000000000000000000000000000"
       "0000000200000000000000000000000000000000000000000000000100000000"
       "0000000000000000000000000000000000000001000000000000000000000000"
       "0000000100000000000000000000000000000000000000000000000100000000"
       "0000000000000000000000010000000000000000000000000000000100000000"
       "0000000100000000000000000000000000000000"},
  };
  for (const auto& [expression, expectedHex] : testCases) {
    SCOPED_TRACE(expression);
    auto serializedSketch = serializeSketch(varbinary, expression);
    EXPECT_EQ(toHex(StringView(serializedSketch)), expectedHex);
  }
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
  auto serializedSketch = resultFlat->valueAt(0);

  auto [depth, width, totalCount] = parseSketch(serializedSketch);
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
  auto serializedSketch = resultFlat->valueAt(0);

  auto [depth, width, totalCount] = parseSketch(serializedSketch);
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

TEST_F(CountMinSketchAggregateTest, accumulatorRowAlignment) {
  auto aggregate = exec::Aggregate::create(
      "count_min_sketch",
      core::AggregationNode::Step::kSingle,
      {BIGINT(), DOUBLE(), DOUBLE(), INTEGER()},
      VARBINARY(),
      core::QueryConfig({}),
      {nullptr,
       makeConstant<double>(0.5, 1),
       makeConstant<double>(0.5, 1),
       makeConstant<int32_t>(1, 1)});
  exec::RowContainer groups(
      {SMALLINT()},
      /*nullableKeys=*/true,
      {exec::Accumulator(aggregate.get(), VARBINARY())},
      {},
      /*hasNext=*/false,
      /*isJoinBuild=*/false,
      /*hasProbedFlag=*/false,
      /*hasCountFlag=*/false,
      /*hasNormalizedKey=*/false,
      /*useListRowIndex=*/false,
      pool());

  const auto accumulatorOffset = groups.accumulatorColumnAt(0).offset();
  for (int32_t i = 0; i < 16; ++i) {
    SCOPED_TRACE(i);
    const auto* accumulator = groups.newRow() + accumulatorOffset;
    EXPECT_EQ(reinterpret_cast<uintptr_t>(accumulator) % alignof(int64_t), 0);
  }
}

TEST_F(CountMinSketchAggregateTest, destroyAfterPartialInitialization) {
  for (vector_size_t numInitialized : {0, 1, 2}) {
    SCOPED_TRACE(numInitialized);
    const std::vector<VectorPtr> arguments{
        makeFlatVector<int64_t>({11, 22}),
        makeConstant<double>(0.5, 2),
        makeConstant<double>(0.5, 2),
        makeConstant<int32_t>(1, 2),
    };
    auto aggregate = exec::Aggregate::create(
        "count_min_sketch",
        core::AggregationNode::Step::kSingle,
        {BIGINT(), DOUBLE(), DOUBLE(), INTEGER()},
        VARBINARY(),
        core::QueryConfig({}),
        {nullptr, arguments[1], arguments[2], arguments[3]});
    exec::RowContainer groups(
        {},
        /*nullableKeys=*/false,
        {exec::Accumulator(aggregate.get(), VARBINARY())},
        {},
        /*hasNext=*/false,
        /*isJoinBuild=*/false,
        /*hasProbedFlag=*/false,
        /*hasCountFlag=*/false,
        /*hasNormalizedKey=*/false,
        /*useListRowIndex=*/false,
        pool());

    const auto column = groups.accumulatorColumnAt(0);
    aggregate->setOffsets(
        column.offset(),
        column.nullByte(),
        column.nullMask(),
        column.initializedByte(),
        column.initializedMask(),
        groups.rowSizeOffset());
    aggregate->setAllocator(&groups.stringAllocator());
    std::array<char*, 2> rows{groups.newRow(), groups.newRow()};
    for (auto* row : rows) {
      // Prevent zeroed vector internals from hiding invalid destructor calls.
      std::memset(
          row + column.offset(), 0xA5, aggregate->accumulatorFixedWidthSize());
    }

    if (numInitialized > 0) {
      const std::array<vector_size_t, 2> indices{0, 1};
      aggregate->initializeNewGroups(
          rows.data(),
          folly::Range<const vector_size_t*>(indices.data(), numInitialized));
      aggregate->addRawInput(
          rows.data(), SelectivityVector(numInitialized), arguments, false);
      EXPECT_GT(groups.stringAllocator().currentBytes(), 0);
    }

    // Unlike clear(), eraseRows() always runs accumulator destruction.
    groups.eraseRows(folly::Range<char**>(rows.data(), rows.size()));
    EXPECT_TRUE(groups.stringAllocator().isEmpty());
  }
}

TEST_F(CountMinSketchAggregateTest, groupBy) {
  // Keep multiple batches so the harness also checks partial abandonment.
  auto vectors = {
      makeRowVector({
          makeFlatVector<int32_t>({1, 2, 1, 2, 3, 1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {std::numeric_limits<int64_t>::min(),
               42,
               -1,
               42,
               std::nullopt,
               0,
               std::nullopt,
               std::nullopt}),
          makeNullableFlatVector<StringView>(
              {StringView("\x80", 1),
               StringView("hello"),
               StringView("\x01\x80", 2),
               StringView("world"),
               std::nullopt,
               StringView(""),
               std::nullopt,
               std::nullopt},
              VARBINARY()),
      }),
      makeRowVector({
          makeFlatVector<int32_t>({2, 1, 3, 2, 1, 1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {7,
               std::numeric_limits<int64_t>::max(),
               std::nullopt,
               0,
               1,
               1,
               -7,
               std::nullopt}),
          makeNullableFlatVector<StringView>(
              {StringView("\x00\x01\x80\xFF", 4),
               StringView("\x01\x02\xFF", 3),
               std::nullopt,
               StringView(""),
               StringView("hello"),
               StringView("\x00\xFF", 2),
               StringView("\xFF\xFE\xFD\xFC", 4),
               std::nullopt},
              VARBINARY()),
      }),
      makeRowVector({
          makeFlatVector<int32_t>({1, 2, 1, 2, 3, 1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {std::numeric_limits<int64_t>::min(),
               42,
               std::nullopt,
               -1,
               std::nullopt,
               7,
               0,
               std::nullopt}),
          makeNullableFlatVector<StringView>(
              {StringView("\x80", 1),
               StringView("hello"),
               std::nullopt,
               StringView("\x00\x00", 2),
               std::nullopt,
               StringView("world"),
               StringView("\x81", 1),
               std::nullopt},
              VARBINARY()),
      }),
  };

  // Match complete Spark 4.1.1 sketches at depth 3, width 7, and seed -7,
  // including the empty sketch for the all-null group.
  const StringView emptySketchHex(
      "0000000100000000000000000000000300000007000000002272954600000000"
      "736445C100000000562ACD160000000000000000000000000000000000000000"
      "0000000000000000000000000000000000000000000000000000000000000000"
      "0000000000000000000000000000000000000000000000000000000000000000"
      "0000000000000000000000000000000000000000000000000000000000000000"
      "0000000000000000000000000000000000000000000000000000000000000000"
      "0000000000000000000000000000000000000000");
  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<StringView>({
          "0000000100000000000000080000000300000007000000002272954600000000"
          "736445C100000000562ACD160000000000000003000000000000000000000000"
          "0000000000000000000000000000000000000004000000000000000100000000"
          "0000000000000000000000050000000000000002000000000000000000000000"
          "0000000000000000000000000000000000000001000000000000000000000000"
          "0000000300000000000000010000000000000000000000000000000000000000"
          "0000000400000000000000000000000000000000",
          "0000000100000000000000080000000300000007000000002272954600000000"
          "736445C100000000562ACD160000000000000002000000000000000000000000"
          "0000000000000000000000010000000000000004000000000000000100000000"
          "0000000000000000000000050000000000000001000000000000000000000000"
          "0000000100000000000000000000000000000001000000000000000000000000"
          "0000000600000000000000010000000000000000000000000000000000000000"
          "0000000100000000000000000000000000000000",
          emptySketchHex,
      }),
      makeFlatVector<StringView>({
          "0000000100000000000000080000000300000007000000002272954600000000"
          "736445C100000000562ACD160000000000000001000000000000000000000000"
          "0000000200000000000000030000000000000000000000000000000200000000"
          "0000000000000000000000020000000000000003000000000000000000000000"
          "0000000100000000000000010000000000000000000000000000000100000000"
          "0000000100000000000000030000000000000001000000000000000100000000"
          "0000000200000000000000000000000000000000",
          "0000000100000000000000080000000300000007000000002272954600000000"
          "736445C100000000562ACD160000000000000001000000000000000000000000"
          "0000000200000000000000030000000000000001000000000000000100000000"
          "0000000000000000000000030000000000000002000000000000000200000000"
          "0000000000000000000000000000000000000001000000000000000000000000"
          "0000000200000000000000030000000000000000000000000000000100000000"
          "0000000100000000000000010000000000000000",
          emptySketchHex,
      }),
  });
  const std::vector<std::string> aggregates = {
      "count_min_sketch(c1, 0.3, 0.875, -7)",
      "count_min_sketch(c2, 0.3, 0.875, -7)",
  };
  const std::vector<std::string> projections = {
      "c0", "to_hex(a0)", "to_hex(a1)"};
  testAggregations(vectors, {"c0"}, aggregates, projections, {expected});
  testAggregationsWithCompanion(
      vectors,
      [](auto& /*builder*/) {},
      {"c0"},
      aggregates,
      {{BIGINT(), DOUBLE(), DOUBLE(), INTEGER()},
       {VARBINARY(), DOUBLE(), DOUBLE(), INTEGER()}},
      projections,
      {expected});
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
  auto serializedSketch = resultFlat->valueAt(0);
  auto [depth, width, totalCount] = parseSketch(serializedSketch);

  EXPECT_EQ(width, 20); // ceil(2/0.1)
  EXPECT_EQ(depth, 7); // ceil(-log1p(-0.99)/log(2))
  EXPECT_EQ(totalCount, 5);
}

TEST_F(CountMinSketchAggregateTest, seedInputTypes) {
  auto input = makeRowVector({
      makeNullableFlatVector<int8_t>({1, std::nullopt, 2, 1}),
      makeNullableFlatVector<int16_t>({1, std::nullopt, 2, 1}),
      makeNullableFlatVector<int32_t>({1, std::nullopt, 2, 1}),
      makeNullableFlatVector<int64_t>({1, std::nullopt, 2, 1}),
      makeNullableFlatVector<StringView>(
          {"\x80", std::nullopt, "\x01\x80", "\x01\x02\xFF"}),
      makeNullableFlatVector<StringView>(
          {"\x80", std::nullopt, "\x01\x80", "\x01\x02\xFF"}, VARBINARY()),
  });
  // Match independent Spark bytes for every value-type and seed-type pair.
  const StringView integralHex(
      "0000000100000000000000030000000100000004000000005D8D6AB9"
      "0000000000000000000000000000000200000000000000010000000000000000");
  const StringView binaryHex(
      "000000010000000000000003000000010000000400000000"
      "5D8D6AB900000000000000000000000000000000000000000000000200000000"
      "00000001");
  std::vector<VectorPtr> expectedColumns;
  for (const auto& hex :
       {integralHex,
        integralHex,
        integralHex,
        integralHex,
        binaryHex,
        binaryHex}) {
    expectedColumns.push_back(makeFlatVector<StringView>({hex}));
  }
  auto expected = makeRowVector(expectedColumns);

  for (const auto bigintSeed : {false, true}) {
    SCOPED_TRACE(bigintSeed);
    parse::ParseOptions options;
    options.parseIntegerAsBigint = bigintSeed;
    auto planNode = exec::test::PlanBuilder(pool())
                        .setParseOptions(options)
                        .values({input})
                        .singleAggregation(
                            {},
                            {"count_min_sketch(c0, 0.5, 0.5, 1)",
                             "count_min_sketch(c1, 0.5, 0.5, 1)",
                             "count_min_sketch(c2, 0.5, 0.5, 1)",
                             "count_min_sketch(c3, 0.5, 0.5, 1)",
                             "count_min_sketch(c4, 0.5, 0.5, 1)",
                             "count_min_sketch(c5, 0.5, 0.5, 1)"})
                        .project(
                            {"to_hex(a0)",
                             "to_hex(a1)",
                             "to_hex(a2)",
                             "to_hex(a3)",
                             "to_hex(a4)",
                             "to_hex(a5)"})
                        .planNode();
    exec::test::AssertQueryBuilder(planNode).assertResults(expected);
  }
}

TEST_F(CountMinSketchAggregateTest, columnParametersRejected) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"count_min_sketch(c0, c1, 0.5, 1)", "eps"},
      {"count_min_sketch(c0, 0.5, c2, 1)", "confidence"},
      {"count_min_sketch(c0, 0.5, 0.5, c3)", "seed"},
  };
  for (const auto constantEncoding : {false, true}) {
    SCOPED_TRACE(constantEncoding);
    for (const vector_size_t size : {0, 3}) {
      SCOPED_TRACE(size);
      // The streaming helper skips aggregation for fewer than two input rows.
      if (size < 2) {
        disableTestStreaming();
      } else {
        enableTestStreaming();
      }
      auto input = makeRowVector({
          makeFlatVector<int64_t>(size, [](auto row) { return row; }),
          makeFlatVector<double>(size, [](auto) { return 0.5; }),
          makeFlatVector<double>(size, [](auto) { return 0.5; }),
          makeFlatVector<int32_t>(size, [](auto) { return 1; }),
      });
      if (constantEncoding) {
        input->childAt(1) = makeConstant<double>(0.5, size);
        input->childAt(2) = makeConstant<double>(0.5, size);
        input->childAt(3) = makeConstant<int32_t>(1, size);
      }
      for (const auto& [aggregate, parameter] : cases) {
        SCOPED_TRACE(aggregate);
        const auto error =
            fmt::format("{} argument must be constant", parameter);
        testFailingAggregations({input}, {}, {aggregate}, error);
        if (size > 0) {
          auto planNode = exec::test::PlanBuilder(pool())
                              .values({input})
                              .window({aggregate + " OVER ()"})
                              .planNode();
          VELOX_ASSERT_THROW(
              exec::test::AssertQueryBuilder(planNode).copyResults(pool()),
              error);
        }
      }
    }
  }
}

TEST_F(CountMinSketchAggregateTest, windowEmptyFramesWithLiteralParameters) {
  const auto emptySketch = serializeSketch(makeFlatVector<int64_t>({}));
  const auto singleValueSketch = serializeSketch(makeFlatVector<int64_t>({1}));
  const auto emptySketchView = StringView(emptySketch);
  const auto singleValueSketchView = StringView(singleValueSketch);
  const vector_size_t numRows = 6;

  for (const auto bigintSeed : {false, true}) {
    SCOPED_TRACE(bigintSeed);
    parse::ParseOptions options;
    options.parseIntegerAsBigint = bigintSeed;
    auto input = makeRowVector({
        makeNullableFlatVector<int64_t>(
            {1, std::nullopt, 3, 1, std::nullopt, 3}),
        makeFlatVector<int32_t>({0, 1, 2, 0, 1, 2}),
        makeFlatVector<int32_t>({0, 0, 0, 1, 1, 1}),
    });
    auto expectedColumns = input->children();
    expectedColumns.push_back(
        makeFlatVector<StringView>(
            {emptySketchView,
             singleValueSketchView,
             emptySketchView,
             emptySketchView,
             singleValueSketchView,
             emptySketchView},
            VARBINARY()));
    expectedColumns.push_back(
        makeFlatVector<StringView>(
            {emptySketchView,
             singleValueSketchView,
             singleValueSketchView,
             emptySketchView,
             singleValueSketchView,
             singleValueSketchView},
            VARBINARY()));
    expectedColumns.push_back(
        makeFlatVector<StringView>(
            numRows,
            [&](auto) { return emptySketchView; },
            nullptr,
            VARBINARY()));
    auto expected = makeRowVector(expectedColumns);

    const std::vector<std::string> windowFunctions = {
        "count_min_sketch(c0, 0.5, 0.5, 1) OVER (PARTITION BY c2 ORDER BY c1 "
        "ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING)",
        "count_min_sketch(c0, 0.5, 0.5, 1) OVER (PARTITION BY c2 ORDER BY c1 "
        "ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)",
        "count_min_sketch(c0, 0.5, 0.5, 1) OVER (PARTITION BY c2 ORDER BY c1 "
        "ROWS BETWEEN 10 PRECEDING AND 9 PRECEDING)",
    };
    for (const auto streaming : {false, true}) {
      SCOPED_TRACE(streaming);
      for (const auto batchSize : {1, 1'024}) {
        SCOPED_TRACE(batchSize);
        exec::test::PlanBuilder builder(pool());
        builder.setParseOptions(options).values({input});
        if (streaming) {
          builder.streamingWindow(windowFunctions);
        } else {
          builder.window(windowFunctions);
        }
        exec::test::AssertQueryBuilder(builder.planNode())
            .config(core::QueryConfig::kPreferredOutputBatchRows, batchSize)
            .assertResults(expected);
      }
    }
  }
}

TEST_F(CountMinSketchAggregateTest, maskedGroupsProduceEmptySketches) {
  const auto makeInput = [&](int32_t group, bool mask) {
    return makeRowVector({
        makeFlatVector<int64_t>({group}),
        makeFlatVector<int32_t>({group}),
        makeFlatVector<bool>({mask}),
    });
  };
  const std::vector<RowVectorPtr> vectors = {
      makeInput(1, false),
      makeInput(0, true),
  };
  const std::vector<std::string> groupingKeys = {"c1"};
  const std::vector<std::string> aggregates = {
      "count_min_sketch(c0, 0.5, 0.5, 1)"};
  const std::vector<std::string> masks = {"c2"};

  auto globalMaskedPartialPlan = exec::test::PlanBuilder(pool())
                                     .values({vectors[0]})
                                     .partialAggregation({}, aggregates, masks)
                                     .planNode();
  auto globalMaskedPartial =
      exec::test::AssertQueryBuilder(globalMaskedPartialPlan)
          .copyResults(pool());
  ASSERT_EQ(globalMaskedPartial->size(), 1);
  auto globalMaskedSketch =
      globalMaskedPartial->childAt(0)->asFlatVector<StringView>();
  ASSERT_FALSE(globalMaskedSketch->isNullAt(0));
  auto [globalMaskedDepth, globalMaskedWidth, globalMaskedCount] =
      parseSketch(globalMaskedSketch->valueAt(0));
  EXPECT_EQ(globalMaskedDepth, 1);
  EXPECT_EQ(globalMaskedWidth, 4);
  EXPECT_EQ(globalMaskedCount, 0);

  auto maskedPartialPlan =
      exec::test::PlanBuilder(pool())
          .values({vectors[0]})
          .partialAggregation(groupingKeys, aggregates, masks)
          .planNode();
  auto maskedPartial =
      exec::test::AssertQueryBuilder(maskedPartialPlan).copyResults(pool());
  ASSERT_EQ(maskedPartial->size(), 1);
  auto maskedSketch = maskedPartial->childAt(1)->asFlatVector<StringView>();
  ASSERT_FALSE(maskedSketch->isNullAt(0));
  auto [maskedDepth, maskedWidth, maskedCount] =
      parseSketch(maskedSketch->valueAt(0));
  EXPECT_EQ(maskedDepth, 1);
  EXPECT_EQ(maskedWidth, 4);
  EXPECT_EQ(maskedCount, 0);

  auto streamingMaskedPartialPlan =
      exec::test::PlanBuilder(pool())
          .values({vectors[0]})
          .streamingAggregation(
              groupingKeys,
              aggregates,
              masks,
              core::AggregationNode::Step::kPartial,
              false)
          .planNode();
  auto streamingMaskedPartial =
      exec::test::AssertQueryBuilder(streamingMaskedPartialPlan)
          .copyResults(pool());
  ASSERT_EQ(streamingMaskedPartial->size(), 1);
  auto streamingMaskedSketch =
      streamingMaskedPartial->childAt(1)->asFlatVector<StringView>();
  ASSERT_FALSE(streamingMaskedSketch->isNullAt(0));
  auto [streamingDepth, streamingWidth, streamingCount] =
      parseSketch(streamingMaskedSketch->valueAt(0));
  EXPECT_EQ(streamingDepth, 1);
  EXPECT_EQ(streamingWidth, 4);
  EXPECT_EQ(streamingCount, 0);

  const auto* countMinSketchEntry =
      exec::getAggregateFunctionEntry("count_min_sketch");
  ASSERT_NE(countMinSketchEntry, nullptr);
  const auto sortedSignatures = countMinSketchEntry->signatures;
  const auto sortedFactory = countMinSketchEntry->factory;
  auto sortedMetadata = countMinSketchEntry->metadata;
  sortedMetadata.orderSensitive = true;
  exec::registerAggregateFunction(
      "sorted_count_min_sketch",
      sortedSignatures,
      sortedFactory,
      sortedMetadata,
      false,
      true);

  auto sortedMaskedPlan =
      exec::test::PlanBuilder(pool())
          .values({vectors[0]})
          .singleAggregation(
              groupingKeys,
              {"sorted_count_min_sketch(c0, 0.5, 0.5, 1 ORDER BY c0)"},
              masks)
          .planNode();
  auto sortedMasked =
      exec::test::AssertQueryBuilder(sortedMaskedPlan).copyResults(pool());
  ASSERT_EQ(sortedMasked->size(), 1);
  auto sortedMaskedSketch =
      sortedMasked->childAt(1)->asFlatVector<StringView>();
  ASSERT_FALSE(sortedMaskedSketch->isNullAt(0));
  auto [sortedDepth, sortedWidth, sortedCount] =
      parseSketch(sortedMaskedSketch->valueAt(0));
  EXPECT_EQ(sortedDepth, 1);
  EXPECT_EQ(sortedWidth, 4);
  EXPECT_EQ(sortedCount, 0);

  auto maskedCompanionPlan =
      exec::test::PlanBuilder(pool())
          .values({vectors[0]})
          .singleAggregation(
              groupingKeys,
              {"count_min_sketch_partial(c0, 0.5, 0.5, 1)"},
              masks)
          .planNode();
  auto maskedCompanion =
      exec::test::AssertQueryBuilder(maskedCompanionPlan).copyResults(pool());
  ASSERT_EQ(maskedCompanion->size(), 1);
  auto maskedCompanionSketch =
      maskedCompanion->childAt(1)->asFlatVector<StringView>();
  ASSERT_FALSE(maskedCompanionSketch->isNullAt(0));
  auto [companionDepth, companionWidth, companionCount] =
      parseSketch(maskedCompanionSketch->valueAt(0));
  EXPECT_EQ(companionDepth, 1);
  EXPECT_EQ(companionWidth, 4);
  EXPECT_EQ(companionCount, 0);

  auto expectedPlan = exec::test::PlanBuilder(pool())
                          .values(vectors)
                          .singleAggregation(groupingKeys, aggregates, masks)
                          .planNode();
  auto expected =
      exec::test::AssertQueryBuilder(expectedPlan).copyResults(pool());

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<core::PlanNodePtr> sources;
  for (const auto& vector : vectors) {
    sources.push_back(
        exec::test::PlanBuilder(planNodeIdGenerator)
            .values({vector})
            .partialAggregation(groupingKeys, aggregates, masks)
            .planNode());
  }
  auto partialPlan = exec::test::PlanBuilder(planNodeIdGenerator)
                         .localPartition(groupingKeys, sources)
                         .finalAggregation()
                         .planNode();

  assertQuery(partialPlan, expected);
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
  writeBigEndianLong(
      maxSketch, kTotalCountOffset, std::numeric_limits<int64_t>::max());
  writeBigEndianLong(
      maxSketch, kFirstTableCountOffset, std::numeric_limits<int64_t>::max());
  writeBigEndianLong(oneSketch, kTotalCountOffset, 1);
  writeBigEndianLong(oneSketch, kFirstTableCountOffset, 1);

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
  EXPECT_EQ(
      readBigEndianLong(firstTableCount), std::numeric_limits<int64_t>::min());
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

  constexpr size_t kVersionOffset = 0;
  constexpr size_t kDepthOffset = 12;
  constexpr size_t kWidthOffset = 16;
  for (const int32_t version : {-1, 0, 2}) {
    SCOPED_TRACE(version);
    auto invalidSketch = sketch;
    writeBigEndianInt(invalidSketch, kVersionOffset, version);
    assertMergeFails({invalidSketch}, "Unexpected CountMinSketch version");
  }

  for (const int32_t dimension : {-1, 0}) {
    SCOPED_TRACE(dimension);
    {
      auto invalidSketch = sketch;
      writeBigEndianInt(invalidSketch, kDepthOffset, dimension);
      assertMergeFails(
          {invalidSketch}, "CountMinSketch depth must be positive");
    }
    {
      auto invalidSketch = sketch;
      writeBigEndianInt(invalidSketch, kWidthOffset, dimension);
      assertMergeFails(
          {invalidSketch}, "CountMinSketch width must be positive");
    }
  }

  {
    auto invalidSketch = sketch;
    writeBigEndianInt(
        invalidSketch, kDepthOffset, std::numeric_limits<int32_t>::max());
    writeBigEndianInt(
        invalidSketch, kWidthOffset, std::numeric_limits<int32_t>::max());
    assertMergeFails(
        {invalidSketch},
        "CountMinSketch dimensions too large: 2147483647 x 2147483647");
  }
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
  testFailingAggregations(
      data,
      {},
      {"count_min_sketch(c0, 0.5, 0.5, cast(null as bigint))"},
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
