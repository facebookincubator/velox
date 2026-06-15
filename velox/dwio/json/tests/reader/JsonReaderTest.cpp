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

#include <folly/compression/Compression.h>
#include <folly/compression/Zlib.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/File.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/json/RegisterJsonReader.h"
#include "velox/type/Timestamp.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::json {
namespace {

// Compresses data into a gzip-framed stream (.gz convention: zlib deflate with
// a 2^15 window and gzip header/trailer), matching what the reader's GZIP
// decompressor expects.
std::string gzipCompress(const std::string& data) {
  auto codec = folly::compression::zlib::getCodec(folly::compression::zlib::Options(
      folly::compression::zlib::Options::Format::GZIP));
  return codec->compress(folly::StringPiece(data));
}

// Compresses data into a raw deflate stream (.deflate convention: no zlib
// header, negative window bits), matching the reader's ZLIB decompressor.
std::string deflateCompress(const std::string& data) {
  auto codec = folly::compression::zlib::getCodec(folly::compression::zlib::Options(
      folly::compression::zlib::Options::Format::RAW));
  return codec->compress(folly::StringPiece(data));
}

// Compresses data with zstd (.zst convention), matching the reader's ZSTD
// decompressor.
std::string zstdCompress(const std::string& data) {
  auto codec =
      folly::compression::getCodec(folly::compression::CodecType::ZSTD);
  return codec->compress(folly::StringPiece(data));
}

class JsonReaderTest : public testing::Test, public test::VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    registerJsonReaderFactory();
    tempDir_ = common::testutil::TempDirectoryPath::create();
  }

  void TearDown() override {
    unregisterJsonReaderFactory();
  }

  // Writes bytes to a fresh file under the test's temp directory whose name
  // ends in extension (e.g. ".gz"), and returns its path. The extension drives
  // the reader's filename-based compression detection.
  std::string writeTempFile(
      const std::string& bytes,
      const std::string& extension) {
    auto path =
        fmt::format("{}/data{}{}", tempDir_->getPath(), fileCounter_++, extension);
    LocalWriteFile writeFile(path);
    writeFile.append(bytes);
    writeFile.close();
    return path;
  }

  // Reads an on-disk file through the JSON reader against schema and returns
  // the resulting RowVector. The file name's extension selects the codec; bytes
  // must already be encoded with that codec.
  RowVectorPtr readFromFile(
      const std::string& bytes,
      const std::string& extension,
      const RowTypePtr& schema) {
    auto path = writeTempFile(bytes, extension);
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::JSON);

    dwio::common::ReaderOptions readerOptions{pool()};
    readerOptions.setFileSchema(schema);

    auto readFile = std::make_shared<LocalReadFile>(path);
    auto bufferedInput =
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool());
    auto reader =
        factory->createReader(std::move(bufferedInput), readerOptions);
    auto rowReader = reader->createRowReader(dwio::common::RowReaderOptions{});

    VectorPtr result;
    rowReader->next(1'000, result);
    return std::dynamic_pointer_cast<RowVector>(result);
  }

  // Reads the entire input string through the JSON reader against the
  // given schema and returns the resulting RowVector. The input is
  // interpreted as JSON Lines (one record per newline). serDeOptions
  // supplies the temporal format strings used for DATE/TIMESTAMP columns.
  RowVectorPtr read(
      const std::string& input,
      const RowTypePtr& schema,
      const dwio::common::JsonSerDeOptions& serDeOptions = {}) {
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::JSON);

    dwio::common::ReaderOptions readerOptions{pool()};
    readerOptions.setFileSchema(schema);
    readerOptions.setJsonSerDeOptions(serDeOptions);

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

  // Reads the byte range [offset, offset + length) of the input through
  // the JSON reader and returns the resulting RowVector, or nullptr when
  // the split contains no whole records.
  RowVectorPtr readRange(
      const std::string& input,
      const RowTypePtr& schema,
      uint64_t offset,
      uint64_t length) {
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::JSON);

    dwio::common::ReaderOptions readerOptions{pool()};
    readerOptions.setFileSchema(schema);

    auto readFile = std::make_shared<InMemoryReadFile>(input);
    auto bufferedInput =
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool());
    auto reader =
        factory->createReader(std::move(bufferedInput), readerOptions);

    dwio::common::RowReaderOptions rowReaderOptions;
    rowReaderOptions.range(offset, length);
    auto rowReader = reader->createRowReader(rowReaderOptions);

    VectorPtr result;
    rowReader->next(1'000, result);
    return std::dynamic_pointer_cast<RowVector>(result);
  }

  // Collects the first (BIGINT) column of a read result into a vector,
  // treating a null result as no records.
  std::vector<int64_t> ids(const RowVectorPtr& row) {
    std::vector<int64_t> out;
    if (row == nullptr) {
      return out;
    }
    auto* col = row->childAt(0)->asFlatVector<int64_t>();
    for (vector_size_t i = 0; i < row->size(); ++i) {
      out.push_back(col->valueAt(i));
    }
    return out;
  }

  // Temp directory backing on-disk fixtures for compression tests. Lives for
  // the whole test so files written under it outlast each read.
  std::shared_ptr<common::testutil::TempDirectoryPath> tempDir_;

  // Distinguishes temp file names within a single test.
  int fileCounter_{0};
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

TEST_F(JsonReaderTest, parseDateDefaultFormat) {
  // Default Joda pattern is yyyy-MM-dd. 2021-03-15 is day 18701 since the
  // epoch; 1969-12-31 is -1, exercising pre-epoch floor division.
  auto row = read(
      "{\"a\":\"2021-03-15\"}\n{\"a\":\"1969-12-31\"}\n",
      ROW({{"a", DATE()}}));
  ASSERT_EQ(row->size(), 2);
  auto col = row->childAt(0)->asFlatVector<int32_t>();
  EXPECT_EQ(col->valueAt(0), 18701);
  EXPECT_EQ(col->valueAt(1), -1);
}

TEST_F(JsonReaderTest, parseDateCustomFormat) {
  dwio::common::JsonSerDeOptions options;
  options.dateFormat = "MM/dd/yyyy";
  auto row = read("{\"a\":\"03/15/2021\"}\n", ROW({{"a", DATE()}}), options);
  EXPECT_EQ(row->childAt(0)->asFlatVector<int32_t>()->valueAt(0), 18701);
}

TEST_F(JsonReaderTest, parseDateMalformedThrows) {
  // Unlike numeric coercion (which silently defaults), a temporal string the
  // format cannot parse is an error.
  VELOX_ASSERT_THROW(
      read("{\"a\":\"not-a-date\"}\n", ROW({{"a", DATE()}})),
      "Failed to parse DATE");
}

TEST_F(JsonReaderTest, parseTimestampDefaultFormat) {
  // Default Joda pattern is yyyy-MM-dd HH:mm:ss, interpreted as UTC.
  auto row =
      read("{\"a\":\"2021-03-15 12:34:56\"}\n", ROW({{"a", TIMESTAMP()}}));
  ASSERT_EQ(row->size(), 1);
  EXPECT_EQ(
      row->childAt(0)->asFlatVector<Timestamp>()->valueAt(0),
      Timestamp(1615811696, 0));
}

TEST_F(JsonReaderTest, parseTimestampWithTimezone) {
  // TZ behavior pinned here: a timezone token in the format consumes the
  // input's offset, and the wall-clock time is normalized to UTC. So
  // 12:34:56 at +05:00 stores as 07:34:56 UTC (seconds 1615793696). Inputs
  // without a timezone token are interpreted as UTC (see the default-format
  // test above).
  dwio::common::JsonSerDeOptions options;
  options.timestampFormat = "yyyy-MM-dd HH:mm:ss ZZ";
  auto row = read(
      "{\"a\":\"2021-03-15 12:34:56 +05:00\"}\n",
      ROW({{"a", TIMESTAMP()}}),
      options);
  EXPECT_EQ(
      row->childAt(0)->asFlatVector<Timestamp>()->valueAt(0),
      Timestamp(1615793696, 0));
}

TEST_F(JsonReaderTest, parseTimestampMalformedThrows) {
  VELOX_ASSERT_THROW(
      read("{\"a\":\"2021-13-99 99:99:99\"}\n", ROW({{"a", TIMESTAMP()}})),
      "Failed to parse TIMESTAMP");
}

TEST_F(JsonReaderTest, arrayOfBigint) {
  auto row = read("{\"a\":[1,2,3]}\n", ROW({{"a", ARRAY(BIGINT())}}));
  // Compare the whole row: next() leaves child vectors at their batch
  // capacity, so comparing the array child directly would mismatch on size.
  auto expected = makeRowVector({makeArrayVector<int64_t>({{1, 2, 3}})});
  test::assertEqualVectors(expected, row);
}

TEST_F(JsonReaderTest, arrayOfString) {
  auto row =
      read("{\"a\":[\"x\",\"y\",\"z\"]}\n", ROW({{"a", ARRAY(VARCHAR())}}));
  auto expected =
      makeRowVector({makeArrayVector<StringView>({{"x"_sv, "y"_sv, "z"_sv}})});
  test::assertEqualVectors(expected, row);
}

TEST_F(JsonReaderTest, arrayElementTypeMismatchCoerces) {
  // [1, "abc", 3] into ARRAY<BIGINT>: the element "abc" coerces to 0 (full-
  // fail to zero) rather than throwing or
  // producing a NULL element.
  auto row = read("{\"a\":[1,\"abc\",3]}\n", ROW({{"a", ARRAY(BIGINT())}}));
  auto expected = makeRowVector({makeArrayVector<int64_t>({{1, 0, 3}})});
  test::assertEqualVectors(expected, row);
}

TEST_F(JsonReaderTest, arrayNullProducesSqlNull) {
  // JSON null for the whole array is SQL NULL, not an empty array.
  auto row = read("{\"a\":null}\n", ROW({{"a", ARRAY(BIGINT())}}));
  ASSERT_EQ(row->size(), 1);
  EXPECT_TRUE(row->childAt(0)->isNullAt(0));
}

TEST_F(JsonReaderTest, arrayEmptyIsNotNull) {
  // JSON [] is a non-null array of cardinality 0, distinct from SQL NULL.
  auto row = read("{\"a\":[]}\n", ROW({{"a", ARRAY(BIGINT())}}));
  ASSERT_EQ(row->size(), 1);
  auto arrays = row->childAt(0)->as<ArrayVector>();
  EXPECT_FALSE(arrays->isNullAt(0));
  EXPECT_EQ(arrays->sizeAt(0), 0);
}

TEST_F(JsonReaderTest, arrayShapeMismatchScalarThrows) {
  // A scalar where an array is expected is a container-shape mismatch.
  VELOX_ASSERT_THROW(
      read("{\"a\":5}\n", ROW({{"a", ARRAY(BIGINT())}})),
      "expected array");
}

TEST_F(JsonReaderTest, arrayShapeMismatchObjectThrows) {
  // An object where an array is expected is a container-shape mismatch.
  VELOX_ASSERT_THROW(
      read("{\"a\":{\"k\":1}}\n", ROW({{"a", ARRAY(BIGINT())}})),
      "expected array");
}

TEST_F(JsonReaderTest, mapVarcharVarcharBaseline) {
  auto row = read(
      "{\"m\":{\"k1\":\"v1\",\"k2\":\"v2\"}}\n",
      ROW({{"m", MAP(VARCHAR(), VARCHAR())}}));
  auto expected = makeRowVector({makeMapVector<StringView, StringView>(
      {{{"k1"_sv, "v1"_sv}, {"k2"_sv, "v2"_sv}}})});
  test::assertEqualVectors(expected, row);
}

TEST_F(JsonReaderTest, mapVarcharVarcharStringifiesNonStrings) {
  // Non-string values stringify into VARCHAR per the leaf coercion rules:
  // numbers via lexeme, booleans as lowercase literals, objects re-serialized
  // minified with key order preserved.
  auto row = read(
      "{\"m\":{\"a\":1,\"b\":true,\"c\":{\"x\":1}}}\n",
      ROW({{"m", MAP(VARCHAR(), VARCHAR())}}));
  auto expected = makeRowVector({makeMapVector<StringView, StringView>(
      {{{"a"_sv, "1"_sv},
        {"b"_sv, "true"_sv},
        {"c"_sv, "{\"x\":1}"_sv}}})});
  test::assertEqualVectors(expected, row);
}

TEST_F(JsonReaderTest, mapVarcharBigintCoercesValues) {
  // Values coerce into BIGINT per the leaf table: "abc" -> 0, true -> 1,
  // -3.7 -> -3 (truncate toward zero).
  auto row = read(
      "{\"m\":{\"a\":\"abc\",\"b\":true,\"c\":-3.7}}\n",
      ROW({{"m", MAP(VARCHAR(), BIGINT())}}));
  auto expected = makeRowVector({makeMapVector<StringView, int64_t>(
      {{{"a"_sv, 0}, {"b"_sv, 1}, {"c"_sv, -3}}})});
  test::assertEqualVectors(expected, row);
}

TEST_F(JsonReaderTest, mapBadValueDoesNotDropRowOrEntry) {
  // A value that fails to coerce becomes 0 (full-fail to zero) — the entry is
  // kept and the row stays. Both keys remain.
  auto row = read(
      "{\"m\":{\"k1\":100,\"k2\":\"abc\"}}\n",
      ROW({{"m", MAP(VARCHAR(), BIGINT())}}));
  auto expected = makeRowVector({makeMapVector<StringView, int64_t>(
      {{{"k1"_sv, 100}, {"k2"_sv, 0}}})});
  test::assertEqualVectors(expected, row);
}

TEST_F(JsonReaderTest, mapKeysPassThroughCaseUnchanged) {
  // MAP keys are data, NOT schema — they are never case-folded. "K" and "k"
  // are two distinct keys (contrast with ROW field matching, which folds).
  auto row = read(
      "{\"m\":{\"K\":1,\"k\":2}}\n", ROW({{"m", MAP(VARCHAR(), BIGINT())}}));
  auto* map = row->childAt(0)->as<MapVector>();
  ASSERT_FALSE(map->isNullAt(0));
  EXPECT_EQ(map->sizeAt(0), 2);
  auto keys = map->mapKeys()->asFlatVector<StringView>();
  auto values = map->mapValues()->asFlatVector<int64_t>();
  const auto offset = map->offsetAt(0);
  EXPECT_EQ(keys->valueAt(offset), "K"_sv);
  EXPECT_EQ(values->valueAt(offset), 1);
  EXPECT_EQ(keys->valueAt(offset + 1), "k"_sv);
  EXPECT_EQ(values->valueAt(offset + 1), 2);
}

TEST_F(JsonReaderTest, mapDuplicateKeysLastWriteWinsInsertionOrderPreserved) {
  // {"a":1,"b":2,"a":3} -> {a=3, b=2} with "a" first, cardinality 2. A naive
  // append would produce a malformed MapVector with a duplicate "a" key.
  auto row = read(
      "{\"m\":{\"a\":1,\"b\":2,\"a\":3}}\n",
      ROW({{"m", MAP(VARCHAR(), BIGINT())}}));
  auto* map = row->childAt(0)->as<MapVector>();
  ASSERT_FALSE(map->isNullAt(0));
  EXPECT_EQ(map->sizeAt(0), 2);
  auto keys = map->mapKeys()->asFlatVector<StringView>();
  auto values = map->mapValues()->asFlatVector<int64_t>();
  const auto offset = map->offsetAt(0);
  EXPECT_EQ(keys->valueAt(offset), "a"_sv);
  EXPECT_EQ(values->valueAt(offset), 3);
  EXPECT_EQ(keys->valueAt(offset + 1), "b"_sv);
  EXPECT_EQ(values->valueAt(offset + 1), 2);
}

TEST_F(JsonReaderTest, mapDuplicateKeysCaseSensitive) {
  // {"k":1,"K":2} -> two distinct entries, cardinality 2. Distinct from ROW
  // case-insensitive matching: MAP keys are not folded.
  auto row = read(
      "{\"m\":{\"k\":1,\"K\":2}}\n", ROW({{"m", MAP(VARCHAR(), BIGINT())}}));
  auto* map = row->childAt(0)->as<MapVector>();
  EXPECT_EQ(map->sizeAt(0), 2);
  auto keys = map->mapKeys()->asFlatVector<StringView>();
  const auto offset = map->offsetAt(0);
  EXPECT_EQ(keys->valueAt(offset), "k"_sv);
  EXPECT_EQ(keys->valueAt(offset + 1), "K"_sv);
}

TEST_F(JsonReaderTest, mapJsonNullProducesSqlNull) {
  // JSON null for the whole map is SQL NULL, not an empty map.
  auto row =
      read("{\"m\":null}\n", ROW({{"m", MAP(VARCHAR(), BIGINT())}}));
  ASSERT_EQ(row->size(), 1);
  EXPECT_TRUE(row->childAt(0)->isNullAt(0));
}

TEST_F(JsonReaderTest, mapJsonEmptyObjectIsNotNull) {
  // JSON {} is a non-null map of cardinality 0, distinct from SQL NULL.
  auto row = read("{\"m\":{}}\n", ROW({{"m", MAP(VARCHAR(), BIGINT())}}));
  auto* map = row->childAt(0)->as<MapVector>();
  EXPECT_FALSE(map->isNullAt(0));
  EXPECT_EQ(map->sizeAt(0), 0);
}

TEST_F(JsonReaderTest, mapShapeMismatchScalarThrows) {
  // A scalar where an object is expected is a container-shape mismatch.
  VELOX_ASSERT_THROW(
      read("{\"m\":5}\n", ROW({{"m", MAP(VARCHAR(), BIGINT())}})),
      "expected object");
}

TEST_F(JsonReaderTest, mapShapeMismatchArrayThrows) {
  // An array where an object is expected is a container-shape mismatch.
  VELOX_ASSERT_THROW(
      read("{\"m\":[1,2]}\n", ROW({{"m", MAP(VARCHAR(), BIGINT())}})),
      "expected object");
}

TEST_F(JsonReaderTest, nestedRowBaseline) {
  auto schema =
      ROW({{"u", ROW({{"id", BIGINT()}, {"name", VARCHAR()}})}});
  auto row = read("{\"u\":{\"id\":123,\"name\":\"Alice\"}}\n", schema);
  auto inner = makeRowVector(
      {makeFlatVector<int64_t>({123}),
       makeFlatVector<StringView>({"Alice"_sv})});
  test::assertEqualVectors(makeRowVector({inner}), row);
}

TEST_F(JsonReaderTest, rowFieldMatchCaseInsensitive) {
  // {"X":1,"y":"hi"} matches ROW(x BIGINT, y VARCHAR) — field names are
  // folded for matching.
  auto schema = ROW({{"u", ROW({{"x", BIGINT()}, {"y", VARCHAR()}})}});
  auto row = read("{\"u\":{\"X\":1,\"y\":\"hi\"}}\n", schema);
  auto inner = makeRowVector(
      {makeFlatVector<int64_t>({1}), makeFlatVector<StringView>({"hi"_sv})});
  test::assertEqualVectors(makeRowVector({inner}), row);
}

TEST_F(JsonReaderTest, rowFieldLastWriteWins) {
  // {"X":1,"x":2} both fold to x; the later assignment wins.
  auto schema = ROW({{"u", ROW({{"x", BIGINT()}})}});
  auto row = read("{\"u\":{\"X\":1,\"x\":2}}\n", schema);
  auto inner = makeRowVector({makeFlatVector<int64_t>({2})});
  test::assertEqualVectors(makeRowVector({inner}), row);
}

TEST_F(JsonReaderTest, rowFieldNestedCaseInsensitive) {
  // Case-folding applies at every nesting level.
  auto schema = ROW(
      {{"u", ROW({{"profile", ROW({{"name", VARCHAR()}})}})}});
  auto row = read("{\"U\":{\"Profile\":{\"NAME\":\"Bob\"}}}\n", schema);
  auto profile = makeRowVector({makeFlatVector<StringView>({"Bob"_sv})});
  test::assertEqualVectors(makeRowVector({makeRowVector({profile})}), row);
}

TEST_F(JsonReaderTest, rowMissingFieldYieldsNull) {
  auto schema =
      ROW({{"u", ROW({{"id", BIGINT()}, {"name", VARCHAR()}})}});
  auto row = read("{\"u\":{\"id\":1}}\n", schema);
  auto* inner = row->childAt(0)->as<RowVector>();
  EXPECT_FALSE(inner->isNullAt(0));
  EXPECT_EQ(inner->childAt(0)->asFlatVector<int64_t>()->valueAt(0), 1);
  EXPECT_TRUE(inner->childAt(1)->isNullAt(0));
}

TEST_F(JsonReaderTest, rowExtraInnerFieldIgnored) {
  auto schema = ROW({{"u", ROW({{"id", BIGINT()}})}});
  auto row = read("{\"u\":{\"id\":1,\"extra\":99}}\n", schema);
  auto inner = makeRowVector({makeFlatVector<int64_t>({1})});
  test::assertEqualVectors(makeRowVector({inner}), row);
}

TEST_F(JsonReaderTest, rowExtraOuterFieldIgnored) {
  auto schema = ROW({{"u", ROW({{"id", BIGINT()}})}});
  auto row = read("{\"u\":{\"id\":1},\"junk\":2}\n", schema);
  auto inner = makeRowVector({makeFlatVector<int64_t>({1})});
  test::assertEqualVectors(makeRowVector({inner}), row);
}

TEST_F(JsonReaderTest, rowJsonNullProducesSqlNull) {
  // JSON null for the whole ROW is SQL NULL.
  auto schema = ROW({{"u", ROW({{"id", BIGINT()}})}});
  auto row = read("{\"u\":null}\n", schema);
  ASSERT_EQ(row->size(), 1);
  EXPECT_TRUE(row->childAt(0)->isNullAt(0));
}

TEST_F(JsonReaderTest, rowEmptyObjectAllFieldsNullButRowNotNull) {
  // {} is a non-null ROW whose every field is NULL — distinct from JSON null.
  auto schema =
      ROW({{"u", ROW({{"id", BIGINT()}, {"name", VARCHAR()}})}});
  auto row = read("{\"u\":{}}\n", schema);
  auto* inner = row->childAt(0)->as<RowVector>();
  EXPECT_FALSE(inner->isNullAt(0));
  EXPECT_TRUE(inner->childAt(0)->isNullAt(0));
  EXPECT_TRUE(inner->childAt(1)->isNullAt(0));
}

TEST_F(JsonReaderTest, rowFieldTypeMismatchCoerces) {
  // A leaf type mismatch coerces (per the leaf table), it does not throw:
  // "abc" into BIGINT -> 0.
  auto schema = ROW({{"u", ROW({{"x", BIGINT()}, {"y", VARCHAR()}})}});
  auto row = read("{\"u\":{\"x\":\"abc\",\"y\":\"hi\"}}\n", schema);
  auto inner = makeRowVector(
      {makeFlatVector<int64_t>({0}), makeFlatVector<StringView>({"hi"_sv})});
  test::assertEqualVectors(makeRowVector({inner}), row);
}

TEST_F(JsonReaderTest, rowShapeMismatchScalarThrows) {
  // A scalar where a ROW is expected is a container-shape mismatch.
  VELOX_ASSERT_THROW(
      read("{\"u\":5}\n", ROW({{"u", ROW({{"id", BIGINT()}})}})),
      "expected object");
}

TEST_F(JsonReaderTest, rowShapeMismatchArrayThrows) {
  // An array where a ROW is expected is a container-shape mismatch.
  VELOX_ASSERT_THROW(
      read("{\"u\":[1,2]}\n", ROW({{"u", ROW({{"id", BIGINT()}})}})),
      "expected object");
}

TEST_F(JsonReaderTest, arrayOfRow) {
  // ROW reachable through an ARRAY element exercises the recursive field-index
  // walk: the element ROW's field map must be built even though no top-level
  // column has that ROW type directly.
  auto schema =
      ROW({{"items", ARRAY(ROW({{"id", BIGINT()}, {"qty", BIGINT()}}))}});
  auto row = read(
      "{\"items\":[{\"id\":10,\"qty\":2},{\"id\":20,\"qty\":1}]}\n", schema);
  auto elements = makeRowVector(
      {makeFlatVector<int64_t>({10, 20}), makeFlatVector<int64_t>({2, 1})});
  auto expected = makeRowVector({makeArrayVector({0}, elements)});
  test::assertEqualVectors(expected, row);
}

TEST_F(JsonReaderTest, mapValueRow) {
  // ROW reachable through a MAP value exercises the same recursive walk.
  auto schema =
      ROW({{"m", MAP(VARCHAR(), ROW({{"n", BIGINT()}}))}});
  auto row = read("{\"m\":{\"a\":{\"n\":1}}}\n", schema);
  auto values = makeRowVector({makeFlatVector<int64_t>({1})});
  auto keys = makeFlatVector<StringView>({"a"_sv});
  auto expected = makeRowVector({makeMapVector({0}, keys, values)});
  test::assertEqualVectors(expected, row);
}

TEST_F(JsonReaderTest, splitFromOffsetZero) {
  auto schema = ROW({{"id", BIGINT()}});
  std::string file = "{\"id\":0}\n{\"id\":1}\n{\"id\":2}\n";
  EXPECT_EQ(
      ids(readRange(file, schema, 0, file.size())),
      (std::vector<int64_t>{0, 1, 2}));
}

TEST_F(JsonReaderTest, splitExhaustiveSweep) {
  auto schema = ROW({{"id", BIGINT()}});
  // Twenty records of varying length. "pad" is not in the schema, so it
  // is ignored — its only purpose is to shift record boundaries.
  std::string file;
  std::vector<int64_t> expected;
  for (int i = 0; i < 20; ++i) {
    file += "{\"id\":" + std::to_string(i) + ",\"pad\":\"" +
        std::string(i % 7, 'x') + "\"}\n";
    expected.push_back(i);
  }

  // Whole-file read sanity check.
  EXPECT_EQ(ids(readRange(file, schema, 0, file.size())), expected);

  // Partition the file into [0, p) and [p, fileSize) at every byte
  // offset p. The concatenation must equal the whole-file read with no
  // dropped or duplicated records. This is the only test that exercises
  // off-by-one errors at every byte position.
  for (uint64_t p = 1; p <= file.size(); ++p) {
    auto left = ids(readRange(file, schema, 0, p));
    auto right = ids(readRange(file, schema, p, file.size() - p));
    std::vector<int64_t> combined = left;
    combined.insert(combined.end(), right.begin(), right.end());
    EXPECT_EQ(combined, expected) << "split point " << p;
  }
}

TEST_F(JsonReaderTest, splitInsideMultibyteUtf8) {
  auto schema = ROW({{"id", BIGINT()}});
  // String value holds "ñ" (0xC3 0xB1) and "€" (0xE2 0x82 0xAC). None of
  // those bytes is 0x0A, so newline scanning must not be confused.
  std::string r0 = "{\"id\":0,\"s\":\"\xC3\xB1\xE2\x82\xAC\"}\n";
  std::string file = r0 + "{\"id\":1}\n";
  // Split one byte into the multibyte run of the first record.
  uint64_t p = r0.find("\xC3\xB1") + 1;
  auto left = ids(readRange(file, schema, 0, p));
  auto right = ids(readRange(file, schema, p, file.size() - p));
  std::vector<int64_t> combined = left;
  combined.insert(combined.end(), right.begin(), right.end());
  EXPECT_EQ(combined, (std::vector<int64_t>{0, 1}));
}

TEST_F(JsonReaderTest, splitInsideJsonStringLiteral) {
  auto schema = ROW({{"id", BIGINT()}});
  // The string value contains braces and an escaped newline (backslash
  // 'n', two bytes — not a real 0x0A line terminator).
  std::string r0 = "{\"id\":0,\"s\":\"a{b}c\\nd efgh\"}\n";
  std::string file = r0 + "{\"id\":1}\n";
  uint64_t p = r0.find("b}c");
  auto left = ids(readRange(file, schema, 0, p));
  auto right = ids(readRange(file, schema, p, file.size() - p));
  std::vector<int64_t> combined = left;
  combined.insert(combined.end(), right.begin(), right.end());
  EXPECT_EQ(combined, (std::vector<int64_t>{0, 1}));
}

TEST_F(JsonReaderTest, splitAtExactlyNewline) {
  auto schema = ROW({{"id", BIGINT()}});
  std::string file = "{\"id\":0}\n{\"id\":1}\n{\"id\":2}\n";
  // Split exactly at the newline ending the first record. That record
  // belongs to the left split; the right split skips from the newline
  // forward to the start of the second record.
  uint64_t nl = file.find('\n');
  auto left = ids(readRange(file, schema, 0, nl));
  auto right = ids(readRange(file, schema, nl, file.size() - nl));
  std::vector<int64_t> combined = left;
  combined.insert(combined.end(), right.begin(), right.end());
  EXPECT_EQ(combined, (std::vector<int64_t>{0, 1, 2}));
}

TEST_F(JsonReaderTest, splitPastEof) {
  auto schema = ROW({{"id", BIGINT()}});
  std::string file = "{\"id\":0}\n{\"id\":1}\n";
  // Offset beyond the end of the file yields no records.
  EXPECT_TRUE(ids(readRange(file, schema, file.size() + 10, 100)).empty());
  // Offset exactly at EOF yields no records.
  EXPECT_TRUE(ids(readRange(file, schema, file.size(), 100)).empty());
}

TEST_F(JsonReaderTest, splitWith20RecordsVaryingLength) {
  auto schema = ROW({{"id", BIGINT()}});
  std::string file;
  std::vector<int64_t> expected;
  for (int i = 0; i < 20; ++i) {
    file += "{\"id\":" + std::to_string(i) + ",\"pad\":\"" +
        std::string(i, 'x') + "\"}\n";
    expected.push_back(i);
  }
  // Three adjacent splits tiling the file end to end.
  uint64_t third = file.size() / 3;
  auto a = ids(readRange(file, schema, 0, third));
  auto b = ids(readRange(file, schema, third, third));
  auto c = ids(readRange(file, schema, 2 * third, file.size() - 2 * third));
  std::vector<int64_t> combined = a;
  combined.insert(combined.end(), b.begin(), b.end());
  combined.insert(combined.end(), c.begin(), c.end());
  EXPECT_EQ(combined, expected);
}

TEST_F(JsonReaderTest, splitFileWithLeadingBlankLineThrows) {
  auto schema = ROW({{"id", BIGINT()}});
  std::string file = "\n{\"id\":0}\n";
  VELOX_ASSERT_USER_THROW(readRange(file, schema, 0, file.size()), "empty row");
}

TEST_F(JsonReaderTest, splitFileWithBlankLineBetweenRecordsThrows) {
  auto schema = ROW({{"id", BIGINT()}});
  std::string file = "{\"id\":0}\n\n{\"id\":1}\n";
  VELOX_ASSERT_USER_THROW(readRange(file, schema, 0, file.size()), "empty row");
}

TEST_F(JsonReaderTest, splitFileWithTrailingBlankLineThrows) {
  auto schema = ROW({{"id", BIGINT()}});
  std::string file = "{\"id\":0}\n\n";
  VELOX_ASSERT_USER_THROW(readRange(file, schema, 0, file.size()), "empty row");
}

TEST_F(JsonReaderTest, splitFileWithWhitespaceOnlyLineThrows) {
  auto schema = ROW({{"id", BIGINT()}});
  std::string file = "{\"id\":0}\n   \n{\"id\":1}\n";
  VELOX_ASSERT_USER_THROW(readRange(file, schema, 0, file.size()), "empty row");
}

// Records exercising several leaf types so decompression is verified to feed
// the parser byte-identical input, not just simple scalars.
const std::string kCompressionJson =
    "{\"id\":0,\"name\":\"alice\",\"score\":1.5,\"ok\":true}\n"
    "{\"id\":1,\"name\":\"bob\",\"score\":-2.25,\"ok\":false}\n"
    "{\"id\":2,\"name\":\"carol\",\"score\":3.75,\"ok\":true}\n";

const RowTypePtr kCompressionSchema = ROW(
    {{"id", BIGINT()},
     {"name", VARCHAR()},
     {"score", DOUBLE()},
     {"ok", BOOLEAN()}});

TEST_F(JsonReaderTest, compressionGzipIdentical) {
  // A gzip-compressed file produces the same vector as the uncompressed bytes.
  auto expected = read(kCompressionJson, kCompressionSchema);
  auto actual = readFromFile(
      gzipCompress(kCompressionJson), ".gz", kCompressionSchema);
  test::assertEqualVectors(expected, actual);
}

TEST_F(JsonReaderTest, compressionDeflateIdentical) {
  // A raw-deflate (.deflate) file decompresses to the same vector.
  auto expected = read(kCompressionJson, kCompressionSchema);
  auto actual = readFromFile(
      deflateCompress(kCompressionJson), ".deflate", kCompressionSchema);
  test::assertEqualVectors(expected, actual);
}

TEST_F(JsonReaderTest, compressionZstdIdentical) {
  // A zstd-compressed (.zst) file decompresses to the same vector.
  auto expected = read(kCompressionJson, kCompressionSchema);
  auto actual = readFromFile(
      zstdCompress(kCompressionJson), ".zst", kCompressionSchema);
  test::assertEqualVectors(expected, actual);
}

TEST_F(JsonReaderTest, compressionLz4Throws) {
  // LZ4/LZO/Snappy carry framing the reader's decompressor does not support.
  // Detection is name-based and fails at reader creation, so the payload bytes
  // are irrelevant.
  auto schema = ROW({{"id", BIGINT()}});
  VELOX_ASSERT_THROW(
      readFromFile("{\"id\":0}\n", ".lz4", schema),
      "Unsupported compression extension");
}

TEST_F(JsonReaderTest, compressionLzoThrows) {
  auto schema = ROW({{"id", BIGINT()}});
  VELOX_ASSERT_THROW(
      readFromFile("{\"id\":0}\n", ".lzo", schema),
      "Unsupported compression extension");
}

TEST_F(JsonReaderTest, compressionSnappyThrows) {
  auto schema = ROW({{"id", BIGINT()}});
  VELOX_ASSERT_THROW(
      readFromFile("{\"id\":0}\n", ".snappy", schema),
      "Unsupported compression extension");
}

TEST_F(JsonReaderTest, compressionSplitReadsAllAtOffsetZeroNothingAfter) {
  // Compressed files are not byte-addressable, so they cannot be split. The
  // split starting at offset 0 reads the whole decompressed file; any later
  // split reads nothing. Without this carve-out the split's byte length (which
  // refers to compressed bytes) would truncate the decompressed stream.
  auto schema = ROW({{"id", BIGINT()}});
  std::string json = "{\"id\":0}\n{\"id\":1}\n{\"id\":2}\n";
  auto compressed = gzipCompress(json);
  auto path = writeTempFile(compressed, ".gz");

  auto factory =
      dwio::common::getReaderFactory(dwio::common::FileFormat::JSON);
  dwio::common::ReaderOptions readerOptions{pool()};
  readerOptions.setFileSchema(schema);

  auto makeReader = [&]() {
    auto readFile = std::make_shared<LocalReadFile>(path);
    auto bufferedInput =
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool());
    return factory->createReader(std::move(bufferedInput), readerOptions);
  };

  // Split at offset 0 covering only the first compressed byte still reads the
  // entire file.
  {
    dwio::common::RowReaderOptions rowReaderOptions;
    rowReaderOptions.range(0, 1);
    auto rowReader = makeReader()->createRowReader(rowReaderOptions);
    VectorPtr result;
    rowReader->next(1'000, result);
    EXPECT_EQ(
        ids(std::dynamic_pointer_cast<RowVector>(result)),
        (std::vector<int64_t>{0, 1, 2}));
  }

  // Any split that does not start at offset 0 reads nothing.
  {
    dwio::common::RowReaderOptions rowReaderOptions;
    rowReaderOptions.range(1, compressed.size());
    auto rowReader = makeReader()->createRowReader(rowReaderOptions);
    VectorPtr result;
    EXPECT_EQ(rowReader->next(1'000, result), 0);
  }
}

} // namespace
} // namespace facebook::velox::json
