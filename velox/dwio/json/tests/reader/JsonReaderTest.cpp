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

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/File.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/json/RegisterJsonReader.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::json {
namespace {

class JsonReaderTest : public testing::Test, public test::VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    registerJsonReaderFactory();
  }

  void TearDown() override {
    unregisterJsonReaderFactory();
  }

  // Reads the entire input string through the JSON reader against the
  // given schema and returns the resulting RowVector. The input is
  // interpreted as JSON Lines (one record per newline).
  RowVectorPtr read(const std::string& input, const RowTypePtr& schema) {
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::JSON);

    dwio::common::ReaderOptions readerOptions{pool()};
    readerOptions.setFileSchema(schema);

    auto readFile = std::make_shared<InMemoryReadFile>(input);
    auto bufferedInput =
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool());
    auto reader =
        factory->createReader(std::move(bufferedInput), readerOptions);
    auto rowReader = reader->createRowReader(dwio::common::RowReaderOptions{});

    VectorPtr result;
    rowReader->next(1'000, result);
    return std::dynamic_pointer_cast<RowVector>(result);
  }
};

TEST_F(JsonReaderTest, factoryRegistration) {
  auto factory =
      dwio::common::getReaderFactory(dwio::common::FileFormat::JSON);
  ASSERT_NE(factory, nullptr);
  EXPECT_EQ(factory->fileFormat(), dwio::common::FileFormat::JSON);
}

TEST_F(JsonReaderTest, emptyFileReturnsZeroRows) {
  auto type = ROW({{"a", BIGINT()}});
  auto factory =
      dwio::common::getReaderFactory(dwio::common::FileFormat::JSON);

  dwio::common::ReaderOptions readerOptions{pool()};
  readerOptions.setFileSchema(type);

  auto readFile = std::make_shared<InMemoryReadFile>(std::string{});
  auto input =
      std::make_unique<dwio::common::BufferedInput>(readFile, *pool());
  auto reader = factory->createReader(std::move(input), readerOptions);
  auto rowReader = reader->createRowReader(dwio::common::RowReaderOptions{});

  VectorPtr result;
  EXPECT_EQ(rowReader->next(10, result), 0);
}

TEST_F(JsonReaderTest, parseSingleBigint) {
  auto schema = ROW({{"a", BIGINT()}});
  auto row = read("{\"a\":42}\n", schema);

  ASSERT_EQ(row->size(), 1);
  auto col = row->childAt(0)->asFlatVector<int64_t>();
  EXPECT_EQ(col->valueAt(0), 42);
}

TEST_F(JsonReaderTest, parseAllNumericTypes) {
  auto schema = ROW(
      {{"t", TINYINT()},
       {"s", SMALLINT()},
       {"i", INTEGER()},
       {"b", BIGINT()},
       {"r", REAL()},
       {"d", DOUBLE()}});
  auto row = read("{\"t\":1,\"s\":2,\"i\":3,\"b\":4,\"r\":1.5,\"d\":2.5}\n", schema);

  ASSERT_EQ(row->size(), 1);
  EXPECT_EQ(row->childAt(0)->asFlatVector<int8_t>()->valueAt(0), 1);
  EXPECT_EQ(row->childAt(1)->asFlatVector<int16_t>()->valueAt(0), 2);
  EXPECT_EQ(row->childAt(2)->asFlatVector<int32_t>()->valueAt(0), 3);
  EXPECT_EQ(row->childAt(3)->asFlatVector<int64_t>()->valueAt(0), 4);
  EXPECT_FLOAT_EQ(row->childAt(4)->asFlatVector<float>()->valueAt(0), 1.5f);
  EXPECT_DOUBLE_EQ(row->childAt(5)->asFlatVector<double>()->valueAt(0), 2.5);
}

TEST_F(JsonReaderTest, bigintFromString) {
  auto row = read("{\"a\":\"100\"}\n", ROW({{"a", BIGINT()}}));
  EXPECT_EQ(row->childAt(0)->asFlatVector<int64_t>()->valueAt(0), 100);
}

TEST_F(JsonReaderTest, bigintFromNonNumericString) {
  // Full-fail to 0, NOT partial-parse.
  // "12abc" -> 0, not 12.
  auto row = read(
      "{\"a\":\"abc\"}\n{\"a\":\"12abc\"}\n", ROW({{"a", BIGINT()}}));
  ASSERT_EQ(row->size(), 2);
  auto col = row->childAt(0)->asFlatVector<int64_t>();
  EXPECT_EQ(col->valueAt(0), 0);
  EXPECT_EQ(col->valueAt(1), 0);
}

TEST_F(JsonReaderTest, bigintFromBoolean) {
  auto row =
      read("{\"a\":true}\n{\"a\":false}\n", ROW({{"a", BIGINT()}}));
  ASSERT_EQ(row->size(), 2);
  auto col = row->childAt(0)->asFlatVector<int64_t>();
  EXPECT_EQ(col->valueAt(0), 1);
  EXPECT_EQ(col->valueAt(1), 0);
}

TEST_F(JsonReaderTest, bigintFromFloatTruncatesTowardZero) {
  // -3.7 -> -3 (truncate toward zero, NOT floor which would be -4).
  auto row = read("{\"a\":-3.7}\n", ROW({{"a", BIGINT()}}));
  EXPECT_EQ(row->childAt(0)->asFlatVector<int64_t>()->valueAt(0), -3);
}

TEST_F(JsonReaderTest, bigintOverflowWrapsUint64) {
  // 9223372036854775808 = 2^63: simdjson parses as unsigned_integer,
  // two's-complement wrap to int64 yields Long.MIN_VALUE.
  auto row =
      read("{\"a\":9223372036854775808}\n", ROW({{"a", BIGINT()}}));
  EXPECT_EQ(
      row->childAt(0)->asFlatVector<int64_t>()->valueAt(0),
      std::numeric_limits<int64_t>::min());
}

TEST_F(JsonReaderTest, bigintExtremeOverflowDiverges) {
  // 1e20 is outside [INT64_MIN, 2^64). v1 diverges from Presto/Jackson
  // here. The contract is only "does
  // not throw"; the exact wrap value is intentionally not asserted.
  EXPECT_NO_THROW(read("{\"a\":1e20}\n", ROW({{"a", BIGINT()}})));
}

TEST_F(JsonReaderTest, bigintFromEmptyString) {
  auto row = read("{\"a\":\"\"}\n", ROW({{"a", BIGINT()}}));
  EXPECT_EQ(row->childAt(0)->asFlatVector<int64_t>()->valueAt(0), 0);
}

TEST_F(JsonReaderTest, missingFieldYieldsNull) {
  // Field "b" is in the schema but absent from the JSON record.
  auto row = read("{\"a\":1}\n", ROW({{"a", BIGINT()}, {"b", BIGINT()}}));
  ASSERT_EQ(row->size(), 1);
  EXPECT_FALSE(row->childAt(0)->isNullAt(0));
  EXPECT_TRUE(row->childAt(1)->isNullAt(0));
}

TEST_F(JsonReaderTest, extraFieldIgnored) {
  // Field "extra" is in the JSON record but not in the schema. Should
  // be silently ignored.
  auto row = read("{\"a\":1,\"extra\":99}\n", ROW({{"a", BIGINT()}}));
  ASSERT_EQ(row->size(), 1);
  EXPECT_EQ(row->childAt(0)->asFlatVector<int64_t>()->valueAt(0), 1);
}

TEST_F(JsonReaderTest, tinyintRangeOverflow) {
  // 255 narrows to int8 via two's-complement wrap: 255 -> -1.
  auto row = read("{\"a\":255}\n", ROW({{"a", TINYINT()}}));
  EXPECT_EQ(row->childAt(0)->asFlatVector<int8_t>()->valueAt(0), -1);
}

TEST_F(JsonReaderTest, parseDoubleAndReal) {
  auto row = read(
      "{\"d\":3.14,\"r\":2.5}\n",
      ROW({{"d", DOUBLE()}, {"r", REAL()}}));
  EXPECT_DOUBLE_EQ(row->childAt(0)->asFlatVector<double>()->valueAt(0), 3.14);
  EXPECT_FLOAT_EQ(row->childAt(1)->asFlatVector<float>()->valueAt(0), 2.5f);
}

TEST_F(JsonReaderTest, parseString) {
  auto row = read("{\"a\":\"hello\"}\n", ROW({{"a", VARCHAR()}}));
  EXPECT_EQ(
      row->childAt(0)->asFlatVector<StringView>()->valueAt(0), "hello"_sv);
}

TEST_F(JsonReaderTest, parseBoolean) {
  auto row =
      read("{\"a\":true}\n{\"a\":false}\n", ROW({{"a", BOOLEAN()}}));
  ASSERT_EQ(row->size(), 2);
  auto col = row->childAt(0)->asFlatVector<bool>();
  EXPECT_TRUE(col->valueAt(0));
  EXPECT_FALSE(col->valueAt(1));
}

TEST_F(JsonReaderTest, booleanFromStringStrictLiteral) {
  // Only the exact lowercase "true" is true. "True"/"TRUE" are false —
  // case-sensitive per the probe.
  auto row = read(
      "{\"a\":\"true\"}\n{\"a\":\"True\"}\n{\"a\":\"TRUE\"}\n",
      ROW({{"a", BOOLEAN()}}));
  ASSERT_EQ(row->size(), 3);
  auto col = row->childAt(0)->asFlatVector<bool>();
  EXPECT_TRUE(col->valueAt(0));
  EXPECT_FALSE(col->valueAt(1));
  EXPECT_FALSE(col->valueAt(2));
}

TEST_F(JsonReaderTest, booleanFromNumberNonzero) {
  // Any nonzero number — including negatives — is true; 0 is false.
  auto row = read(
      "{\"a\":-1}\n{\"a\":2}\n{\"a\":0}\n", ROW({{"a", BOOLEAN()}}));
  ASSERT_EQ(row->size(), 3);
  auto col = row->childAt(0)->asFlatVector<bool>();
  EXPECT_TRUE(col->valueAt(0));
  EXPECT_TRUE(col->valueAt(1));
  EXPECT_FALSE(col->valueAt(2));
}

TEST_F(JsonReaderTest, varcharFromIntegerStringifies) {
  auto row = read("{\"a\":123}\n", ROW({{"a", VARCHAR()}}));
  EXPECT_EQ(
      row->childAt(0)->asFlatVector<StringView>()->valueAt(0), "123"_sv);
}

TEST_F(JsonReaderTest, varcharFromSimpleFloatStringifies) {
  auto row = read("{\"a\":123.45}\n", ROW({{"a", VARCHAR()}}));
  EXPECT_EQ(
      row->childAt(0)->asFlatVector<StringView>()->valueAt(0), "123.45"_sv);
}

TEST_F(JsonReaderTest, varcharFromNumberV1DivergenceDocumented) {
  // v1 emits the original lexeme rather than Presto's BigDecimal-canonical
  // form. The semantic value is preserved; the exact textual form may
  // differ (trailing zeros, scientific notation). We assert each output is
  // a valid string parseable back to the input number, NOT bit-for-bit
  // Presto parity. This is the documented "VARCHAR-from-number v1
  // divergence".
  auto row = read(
      "{\"a\":1.20}\n{\"a\":1e3}\n{\"a\":100.0}\n", ROW({{"a", VARCHAR()}}));
  ASSERT_EQ(row->size(), 3);
  auto col = row->childAt(0)->asFlatVector<StringView>();
  EXPECT_DOUBLE_EQ(folly::to<double>(std::string(col->valueAt(0))), 1.20);
  EXPECT_DOUBLE_EQ(folly::to<double>(std::string(col->valueAt(1))), 1e3);
  EXPECT_DOUBLE_EQ(folly::to<double>(std::string(col->valueAt(2))), 100.0);
}

TEST_F(JsonReaderTest, varcharFromBooleanStringifiesLowercase) {
  auto row =
      read("{\"a\":true}\n{\"a\":false}\n", ROW({{"a", VARCHAR()}}));
  ASSERT_EQ(row->size(), 2);
  auto col = row->childAt(0)->asFlatVector<StringView>();
  EXPECT_EQ(col->valueAt(0), "true"_sv);
  EXPECT_EQ(col->valueAt(1), "false"_sv);
}

TEST_F(JsonReaderTest, varcharFromNestedObjectReSerializes) {
  // Whitespace stripped, key order preserved (NOT canonicalized). Asserted
  // byte-for-byte against the probe's observed minified output.
  auto row =
      read("{\"a\":{\"a\"  :  1 , \"b\" : 2}}\n", ROW({{"a", VARCHAR()}}));
  EXPECT_EQ(
      row->childAt(0)->asFlatVector<StringView>()->valueAt(0),
      "{\"a\":1,\"b\":2}"_sv);
}

TEST_F(JsonReaderTest, varcharFromArrayReSerializes) {
  auto row = read("{\"a\":[1 ,  2,3]}\n", ROW({{"a", VARCHAR()}}));
  EXPECT_EQ(
      row->childAt(0)->asFlatVector<StringView>()->valueAt(0), "[1,2,3]"_sv);
}

TEST_F(JsonReaderTest, varcharStringDecodesEscapes) {
  // simdjson decodes the \/ escape; passthrough yields the bare slash.
  auto row =
      read("{\"a\":\"http:\\/\\/example.com\"}\n", ROW({{"a", VARCHAR()}}));
  EXPECT_EQ(
      row->childAt(0)->asFlatVector<StringView>()->valueAt(0),
      "http://example.com"_sv);
}

TEST_F(JsonReaderTest, varcharFromExplicitJsonNull) {
  // Unquoted null is SQL NULL.
  auto row = read("{\"a\":null}\n", ROW({{"a", VARCHAR()}}));
  ASSERT_EQ(row->size(), 1);
  EXPECT_TRUE(row->childAt(0)->isNullAt(0));
}

TEST_F(JsonReaderTest, varcharFromQuotedNullString) {
  // Quoted "null" is the 4-char string null, distinct from SQL NULL.
  auto row = read("{\"a\":\"null\"}\n", ROW({{"a", VARCHAR()}}));
  ASSERT_EQ(row->size(), 1);
  EXPECT_FALSE(row->childAt(0)->isNullAt(0));
  EXPECT_EQ(
      row->childAt(0)->asFlatVector<StringView>()->valueAt(0), "null"_sv);
}

TEST_F(JsonReaderTest, decimalPreservesPrecisionViaLexeme) {
  // 123456789.0123456789 has more significant digits than a double can hold
  // exactly. Routing through double would round the trailing digits away;
  // reading from the lexeme preserves them. DECIMAL(38, 10) scales by 10^10,
  // so the exact unscaled value is 1234567890123456789.
  auto row = read("{\"a\":123456789.0123456789}\n", ROW({{"a", DECIMAL(38, 10)}}));
  ASSERT_EQ(row->size(), 1);
  EXPECT_EQ(
      row->childAt(0)->asFlatVector<int128_t>()->valueAt(0),
      static_cast<int128_t>(1234567890123456789LL));
}

TEST_F(JsonReaderTest, decimalScaleAndPrecision) {
  // A short decimal (precision <= 18) is stored as int64. "3.14" at scale 4
  // scales to the unscaled value 31400. The string and number forms parse
  // identically since both go through the lexeme.
  auto row = read(
      "{\"a\":3.14,\"b\":\"3.14\"}\n",
      ROW({{"a", DECIMAL(10, 4)}, {"b", DECIMAL(10, 4)}}));
  ASSERT_EQ(row->size(), 1);
  EXPECT_EQ(row->childAt(0)->asFlatVector<int64_t>()->valueAt(0), 31400);
  EXPECT_EQ(row->childAt(1)->asFlatVector<int64_t>()->valueAt(0), 31400);
}

TEST_F(JsonReaderTest, varbinaryFromBase64) {
  // "SGVsbG8gV29ybGQ=" is the base64 encoding of "Hello World".
  auto row =
      read("{\"a\":\"SGVsbG8gV29ybGQ=\"}\n", ROW({{"a", VARBINARY()}}));
  ASSERT_EQ(row->size(), 1);
  EXPECT_EQ(
      row->childAt(0)->asFlatVector<StringView>()->valueAt(0),
      "Hello World"_sv);
}

TEST_F(JsonReaderTest, varbinaryInvalidBase64Throws) {
  // '@' is outside the base64 alphabet, so decoding must fail.
  VELOX_ASSERT_THROW(
      read("{\"a\":\"@@@@\"}\n", ROW({{"a", VARBINARY()}})),
      "Invalid base64 in VARBINARY column");
}

} // namespace
} // namespace facebook::velox::json
