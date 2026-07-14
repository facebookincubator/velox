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

#include "velox/common/Casts.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/Mutation.h"
#include "velox/dwio/parquet/reader/ParquetStatsContext.h"
#include "velox/dwio/parquet/reader/SemanticVersion.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"
#include "velox/dwio/parquet/thrift/ParquetThrift.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;

class ParquetReaderTest : public ParquetTestBase {
 public:
  void assertReadWithExpected(
      const std::string& fileName,
      const RowTypePtr& rowType,
      const RowVectorPtr& expected) {
    auto readerBundle = readerBuilder(fileName, rowType).build();
    assertReadWithReaderAndExpected(
        rowType, *readerBundle.rowReader, expected, *pool_);
  }

  void assertReadWithFilters(
      const std::string& fileName,
      const RowTypePtr& fileSchema,
      FilterMap filters,
      const RowVectorPtr& expected) {
    auto reader = createReader(fileName);
    assertReadWithReaderAndFilters(
        *reader, fileSchema, std::move(filters), expected);
  }

  // Builds ReaderOptions with the Parquet thrift footer-memory tracking
  // path enabled by a 1-byte threshold so any footer engages tracking.
  dwio::common::ReaderOptions makeThriftTrackingReaderOptions() {
    auto readerOptions = makeDefaultReaderOptions();
    auto parquetOptions = std::make_shared<ParquetReaderOptions>();
    parquetOptions->footerMemoryTrackingThreshold = 1;
    readerOptions.setFormatSpecificOptions(std::move(parquetOptions));
    return readerOptions;
  }
};

TEST_F(ParquetReaderTest, createFormatOptions) {
  config::ConfigBase connectorConfig({
      {std::string(ParquetConfig::kUseColumnNames), "true"},
      {std::string(ParquetConfig::kFooterSpeculativeIoSize), "99"},
      {std::string(ParquetConfig::kAllowInt32Narrowing), "false"},
      {std::string(ParquetConfig::kFooterMemoryTrackingThreshold), "99"},
  });
  config::ConfigBase session({
      {std::string(ParquetConfig::kFooterSpeculativeIoSizeSession), "2"},
      {std::string(ParquetConfig::kAllowInt32NarrowingSession), "true"},
      {std::string(ParquetConfig::kFooterMemoryTrackingThresholdSession), "1"},
  });

  ParquetReaderFactory factory;
  auto parquetOptions = checkedPointerCast<ParquetReaderOptions>(
      factory.createFormatOptions(connectorConfig, session));
  EXPECT_EQ(parquetOptions->columnMappingMode, ColumnMappingMode::kName);
  EXPECT_EQ(parquetOptions->footerSpeculativeIoSize, 2);
  EXPECT_TRUE(parquetOptions->allowInt32Narrowing);
  EXPECT_EQ(parquetOptions->footerMemoryTrackingThreshold, 1);
}

TEST_F(ParquetReaderTest, parseSample) {
  // sample.parquet holds two columns (a: BIGINT, b: DOUBLE) and
  // 20 rows (10 rows per group). Group offsets are 153 and 614.
  // Data is in plain uncompressed format:
  //   a: [1..20]
  //   b: [1.0..20.0]
  auto readerBundle = readerBuilder("sample.parquet", sampleSchema()).build();
  EXPECT_EQ(readerBundle.reader->numberOfRows(), 20ULL);

  auto type = readerBundle.reader->typeWithId();
  EXPECT_EQ(type->size(), 2ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::BIGINT);
  auto col1 = type->childAt(1);
  EXPECT_EQ(col1->type()->kind(), TypeKind::DOUBLE);
  EXPECT_EQ(type->childByName("a"), col0);
  EXPECT_EQ(type->childByName("b"), col1);

  auto expected = makeRowVector({
      makeFlatVector<int64_t>(20, [](auto row) { return row + 1; }),
      makeFlatVector<double>(20, [](auto row) { return row + 1; }),
  });

  assertReadWithReaderAndExpected(
      sampleSchema(), *readerBundle.rowReader, expected, *leafPool_);
}

TEST_F(ParquetReaderTest, parquetFieldIdColumnMappingNotImplemented) {
  auto readerOptions = makeDefaultReaderOptions();
  readerOptions.setColumnMappingMode(ColumnMappingMode::kParquetFieldId);

  VELOX_ASSERT_THROW(
      createReader("sample.parquet", readerOptions),
      "Parquet field ID column mapping is not implemented yet.");
}

TEST_F(ParquetReaderTest, parseEmptyNestedList) {
  // parse_empty_nested_list.parquet holds 1,000 rows of the following data:
  //
  // { "msg": { "a": { "b": [] } } }
  //
  // The create table SQL is:
  //
  // CREATE TABLE test (
  //   msg ROW(
  //     a ROW(
  //       b ARRAY<INTEGER>
  //     )
  //   )
  // )
  // WITH (
  //   format = 'PARQUET'
  // );
  //
  // All 1000 rows in one row group.
  // Data is in RLE_DICTIONARY Snappy format.
  auto schema = ROW("msg", ROW("a", ROW("b", ARRAY(INTEGER()))));
  auto readerBundle =
      readerBuilder("parse_empty_nested_list.parquet", schema).build();
  EXPECT_EQ(readerBundle.reader->numberOfRows(), 1000ULL);

  // Ensure the intact data structure
  auto type = readerBundle.reader->typeWithId();
  EXPECT_EQ(type->size(), 1ULL);
  auto colMsg = type->childAt(0);
  EXPECT_EQ(colMsg->type()->kind(), TypeKind::ROW);

  EXPECT_EQ(colMsg->size(), 1ULL);
  auto childMsgA = colMsg->childAt(0);
  EXPECT_EQ(childMsgA->type()->kind(), TypeKind::ROW);

  EXPECT_EQ(childMsgA->size(), 1ULL);
  auto childMsgAB = childMsgA->childAt(0);
  EXPECT_EQ(childMsgAB->type()->kind(), TypeKind::ARRAY);

  EXPECT_EQ(childMsgAB->size(), 1ULL);
  auto childMsgABElement = childMsgAB->childAt(0);
  EXPECT_EQ(childMsgABElement->type()->kind(), TypeKind::INTEGER);

  EXPECT_EQ(type->childByName("msg"), colMsg);
  EXPECT_EQ(colMsg->childByName("a"), childMsgA);
  EXPECT_EQ(childMsgA->childByName("b"), childMsgAB);

  // Ensure actual data can be read

  auto expected =
      makeRowVector({makeRowVector({makeRowVector({makeArrayVector<int32_t>(
          1000,
          // Create empty list
          [](auto) { return 0; },
          [](auto) { return 0; },
          [](auto) { return false; })})})});

  assertReadWithReaderAndExpected(
      schema, *readerBundle.rowReader, expected, *leafPool_);
}

TEST_F(ParquetReaderTest, parseUnannotatedList) {
  // unannotated_list.parquet has the following the schema
  // the list is defined without the middle layer
  // message ParquetSchema {
  //   optional group self (LIST) {
  //     repeated group self_tuple {
  //       optional int64 a;
  //       optional boolean b;
  //       required binary c (STRING);
  //     }
  //   }
  // }
  auto reader = createReader("unannotated_list.parquet");

  EXPECT_EQ(reader->numberOfRows(), 22ULL);

  auto type = reader->typeWithId();
  EXPECT_EQ(type->size(), 1ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::ARRAY);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(col0)->name_, "self");

  EXPECT_EQ(col0->size(), 1ULL);
  EXPECT_EQ(col0->childAt(0)->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(col0->childAt(0))
          ->name_,
      "dummy");

  EXPECT_EQ(col0->childAt(0)->childAt(0)->type()->kind(), TypeKind::BIGINT);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(
          col0->childAt(0)->childAt(0))
          ->name_,
      "a");

  EXPECT_EQ(col0->childAt(0)->childAt(1)->type()->kind(), TypeKind::BOOLEAN);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(
          col0->childAt(0)->childAt(1))
          ->name_,
      "b");

  EXPECT_EQ(col0->childAt(0)->childAt(2)->type()->kind(), TypeKind::VARCHAR);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(
          col0->childAt(0)->childAt(2))
          ->name_,
      "c");
}

TEST_F(ParquetReaderTest, parseUnannotatedMap) {
  // unannotated_map.parquet has the following the schema
  // the map is defined with a MAP_KEY_VALUE node
  // message hive_schema {
  // optional group test (MAP) {
  //   repeated group key_value (MAP_KEY_VALUE) {
  //       required binary key (STRING);
  //       optional int64 value;
  //   }
  //  }
  //}
  const std::string filename("unnotated_map.parquet");

  auto reader = createReader(filename);

  auto type = reader->typeWithId();
  EXPECT_EQ(type->size(), 1ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::MAP);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(col0)->name_, "test");

  EXPECT_EQ(col0->size(), 2ULL);
  EXPECT_EQ(col0->childAt(0)->type()->kind(), TypeKind::VARCHAR);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(col0->childAt(0))
          ->name_,
      "key");

  EXPECT_EQ(col0->childAt(1)->type()->kind(), TypeKind::BIGINT);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(col0->childAt(1))
          ->name_,
      "value");
}

TEST_F(ParquetReaderTest, parseLegacyListWithMultipleChildren) {
  // listmultiplechildren.parquet has the following the schema
  // message hive_schema {
  //  optional group test (LIST) {
  //    repeated group array {
  //      optional int64 a;
  //      optional boolean b;
  //      optional binary c (STRING);
  //    }
  //  }
  // }
  // Namely, node 'array' has >1 child
  const std::string filename("listmultiplechildren.parquet");

  auto reader = createReader(filename);

  auto type = reader->typeWithId();
  EXPECT_EQ(type->size(), 1ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::ARRAY);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(col0)->name_, "test");

  EXPECT_EQ(col0->size(), 1ULL);
  EXPECT_EQ(col0->childAt(0)->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(col0->childAt(0))
          ->name_,
      "dummy");

  EXPECT_EQ(col0->childAt(0)->childAt(0)->type()->kind(), TypeKind::BIGINT);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(
          col0->childAt(0)->childAt(0))
          ->name_,
      "a");

  EXPECT_EQ(col0->childAt(0)->childAt(1)->type()->kind(), TypeKind::BOOLEAN);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(
          col0->childAt(0)->childAt(1))
          ->name_,
      "b");

  EXPECT_EQ(col0->childAt(0)->childAt(2)->type()->kind(), TypeKind::VARCHAR);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(
          col0->childAt(0)->childAt(2))
          ->name_,
      "c");
}

TEST_F(ParquetReaderTest, parseArrayOfRowHiveReservedKeywords) {
  // array_of_row_hive_reserved_keywords.parquet was created using Hive's
  // built-in Parquet writer with the following schema:
  //  message hive_schema {
  //    optional int32 id;
  //    optional group items (LIST) {
  //      repeated group bag {
  //        optional group array_element {
  //          optional binary name (STRING);
  //          optional int32 quantity;
  //          optional double price;
  //        }
  //      }
  //    }
  //  }
  const std::string expectedVeloxType =
      "ROW<id:INTEGER,items:ARRAY<ROW<name:VARCHAR,quantity:INTEGER,price:DOUBLE>>>";
  auto reader = createReader("array_of_row_hive_reserved_keywords.parquet");
  EXPECT_EQ(reader->rowType()->toString(), expectedVeloxType);
  EXPECT_EQ(reader->numberOfRows(), 6ULL);

  auto type = reader->typeWithId();
  EXPECT_EQ(type->size(), 2ULL);

  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::INTEGER);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(col0)->name_, "id");

  auto col1 = type->childAt(1);
  EXPECT_EQ(col1->type()->kind(), TypeKind::ARRAY);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(col1)->name_, "items");
  EXPECT_EQ(col1->size(), 1ULL);

  auto arrayElement = col1->childAt(0);
  EXPECT_EQ(arrayElement->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(arrayElement)->name_,
      "array_element");
  EXPECT_EQ(arrayElement->size(), 3ULL);

  EXPECT_EQ(arrayElement->childAt(0)->type()->kind(), TypeKind::VARCHAR);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(
          arrayElement->childAt(0))
          ->name_,
      "name");

  EXPECT_EQ(arrayElement->childAt(1)->type()->kind(), TypeKind::INTEGER);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(
          arrayElement->childAt(1))
          ->name_,
      "quantity");

  EXPECT_EQ(arrayElement->childAt(2)->type()->kind(), TypeKind::DOUBLE);
  EXPECT_EQ(
      std::static_pointer_cast<const ParquetTypeWithId>(
          arrayElement->childAt(2))
          ->name_,
      "price");
}

TEST_F(ParquetReaderTest, parseSampleRange1) {
  auto readerBundle =
      readerBuilder("sample.parquet", sampleSchema()).byteRange(0, 200).build();
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(10, [](auto row) { return row + 1; }),
      makeFlatVector<double>(10, [](auto row) { return row + 1; }),
  });
  assertReadWithReaderAndExpected(
      sampleSchema(), *readerBundle.rowReader, expected, *leafPool_);
}

TEST_F(ParquetReaderTest, parseSampleRange2) {
  auto readerBundle = readerBuilder("sample.parquet", sampleSchema())
                          .byteRange(200, 500)
                          .build();
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(10, [](auto row) { return row + 11; }),
      makeFlatVector<double>(10, [](auto row) { return row + 11; }),
  });
  assertReadWithReaderAndExpected(
      sampleSchema(), *readerBundle.rowReader, expected, *leafPool_);
}

TEST_F(ParquetReaderTest, parseSampleEmptyRange) {
  auto readerBundle = readerBuilder("sample.parquet", sampleSchema())
                          .byteRange(300, 10)
                          .build();

  VectorPtr result;
  EXPECT_EQ(readerBundle.rowReader->next(1000, result), 0);
}

TEST_F(ParquetReaderTest, parseReadAsLowerCase) {
  // upper.parquet holds two columns (A: BIGINT, b: BIGINT) and
  // 2 rows.
  auto readerOptions = makeDefaultReaderOptions();
  auto outputRowType = ROW({"a", "b"}, {BIGINT(), BIGINT()});
  readerOptions.setFileSchema(outputRowType);
  readerOptions.setFileColumnNamesReadAsLowerCase(true);
  auto reader = createReader("upper.parquet", readerOptions);
  EXPECT_EQ(reader->numberOfRows(), 2ULL);

  auto type = reader->typeWithId();
  EXPECT_EQ(type->size(), 2ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::BIGINT);
  auto col1 = type->childAt(1);
  EXPECT_EQ(col1->type()->kind(), TypeKind::BIGINT);
  EXPECT_EQ(type->childByName("a"), col0);
  EXPECT_EQ(type->childByName("b"), col1);
}

TEST_F(ParquetReaderTest, parseRowMapArrayReadAsLowerCase) {
  // upper_complex.parquet holds one row of type
  // root
  //  |-- Cc: struct (nullable = true)
  //  |    |-- CcLong0: long (nullable = true)
  //  |    |-- CcMap1: map (nullable = true)
  //  |    |    |-- key: string
  //  |    |    |-- value: struct (valueContainsNull = true)
  //  |    |    |    |-- CcArray2: array (nullable = true)
  //  |    |    |    |    |-- element: struct (containsNull = true)
  //  |    |    |    |    |    |-- CcInt3: integer (nullable = true)
  // data
  // +-----------------------+
  // |Cc                     |
  // +-----------------------+
  // |{120, {key -> {[{1}]}}}|
  // +-----------------------+
  auto readerOptions = makeDefaultReaderOptions();
  readerOptions.setFileColumnNamesReadAsLowerCase(true);
  auto reader = createReader("upper_complex.parquet", readerOptions);

  EXPECT_EQ(reader->numberOfRows(), 1ULL);

  auto type = reader->typeWithId();
  EXPECT_EQ(type->size(), 1ULL);

  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(type->childByName("cc"), col0);

  auto col0_0 = col0->childAt(0);
  EXPECT_EQ(col0_0->type()->kind(), TypeKind::BIGINT);
  EXPECT_EQ(col0->childByName("cclong0"), col0_0);

  auto col0_1 = col0->childAt(1);
  EXPECT_EQ(col0_1->type()->kind(), TypeKind::MAP);
  EXPECT_EQ(col0->childByName("ccmap1"), col0_1);

  auto col0_1_0 = col0_1->childAt(0);
  EXPECT_EQ(col0_1_0->type()->kind(), TypeKind::VARCHAR);

  auto col0_1_1 = col0_1->childAt(1);
  EXPECT_EQ(col0_1_1->type()->kind(), TypeKind::ROW);

  auto col0_1_1_0 = col0_1_1->childAt(0);
  EXPECT_EQ(col0_1_1_0->type()->kind(), TypeKind::ARRAY);
  EXPECT_EQ(col0_1_1->childByName("ccarray2"), col0_1_1_0);

  auto col0_1_1_0_0 = col0_1_1_0->childAt(0);
  EXPECT_EQ(col0_1_1_0_0->type()->kind(), TypeKind::ROW);
  auto col0_1_1_0_0_0 = col0_1_1_0_0->childAt(0);
  EXPECT_EQ(col0_1_1_0_0_0->type()->kind(), TypeKind::INTEGER);
  EXPECT_EQ(col0_1_1_0_0->childByName("ccint3"), col0_1_1_0_0_0);
}

TEST_F(ParquetReaderTest, parseEmpty) {
  // empty.parquet holds two columns (a: BIGINT, b: DOUBLE) and
  // 0 rows.
  auto reader = createReader("empty.parquet");
  EXPECT_EQ(reader->numberOfRows(), 0ULL);

  auto type = reader->typeWithId();
  EXPECT_EQ(type->size(), 2ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::BIGINT);
  auto col1 = type->childAt(1);
  EXPECT_EQ(col1->type()->kind(), TypeKind::DOUBLE);
  EXPECT_EQ(type->childByName("a"), col0);
  EXPECT_EQ(type->childByName("b"), col1);
}

TEST_F(ParquetReaderTest, parseInt) {
  // int.parquet holds integer columns (int: INTEGER, bigint: BIGINT)
  // and 10 rows.
  // Data is in plain uncompressed format:
  //   int: [100 .. 109]
  //   bigint: [1000 .. 1009]
  auto readerBundle = readerBuilder("int.parquet", intSchema()).build();

  EXPECT_EQ(readerBundle.reader->numberOfRows(), 10ULL);

  auto type = readerBundle.reader->typeWithId();
  EXPECT_EQ(type->size(), 2ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::INTEGER);
  auto col1 = type->childAt(1);
  EXPECT_EQ(col1->type()->kind(), TypeKind::BIGINT);

  auto expected = makeRowVector({
      makeFlatVector<int32_t>(10, [](auto row) { return row + 100; }),
      makeFlatVector<int64_t>(10, [](auto row) { return row + 1000; }),
  });
  assertReadWithReaderAndExpected(
      intSchema(), *readerBundle.rowReader, expected, *leafPool_);
}

TEST_F(ParquetReaderTest, parseUnsignedInt1) {
  // uint.parquet holds unsigned integer columns (uint8: TINYINT, uint16:
  // SMALLINT, uint32: INTEGER, uint64: BIGINT) and 3 rows. Data is in plain
  // uncompressed format:
  //   uint8: [255, 3, 3]
  //   uint16: [65535, 2000, 3000]
  //   uint32: [4294967295, 2000000000, 3000000000]
  //   uint64: [18446744073709551615, 2000000000000000000, 3000000000000000000]
  auto rowType =
      ROW({"uint8", "uint16", "uint32", "uint64"},
          {TINYINT(), SMALLINT(), INTEGER(), BIGINT()});
  auto readerBundle = readerBuilder("uint.parquet", rowType).build();
  EXPECT_EQ(readerBundle.reader->numberOfRows(), 3ULL);
  auto type = readerBundle.reader->typeWithId();
  EXPECT_EQ(type->size(), 4ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::TINYINT);
  auto col1 = type->childAt(1);
  EXPECT_EQ(col1->type()->kind(), TypeKind::SMALLINT);
  auto col2 = type->childAt(2);
  EXPECT_EQ(col2->type()->kind(), TypeKind::INTEGER);
  auto col3 = type->childAt(3);
  EXPECT_EQ(col3->type()->kind(), TypeKind::BIGINT);

  auto expected = makeRowVector(
      {makeFlatVector<uint8_t>({255, 2, 3}),
       makeFlatVector<uint16_t>({65535, 2000, 3000}),
       makeFlatVector<uint32_t>({4294967295, 2000000000, 3000000000}),
       makeFlatVector<uint64_t>(
           {18446744073709551615ULL,
            2000000000000000000ULL,
            3000000000000000000ULL})});
  assertReadWithReaderAndExpected(
      rowType, *readerBundle.rowReader, expected, *pool_);
}

TEST_F(ParquetReaderTest, parseUnsignedInt2) {
  auto rowType =
      ROW({"uint8", "uint16", "uint32", "uint64"},
          {SMALLINT(), SMALLINT(), INTEGER(), BIGINT()});
  auto expected = makeRowVector(
      {makeFlatVector<uint16_t>({255, 2, 3}),
       makeFlatVector<uint16_t>({65535, 2000, 3000}),
       makeFlatVector<uint32_t>({4294967295, 2000000000, 3000000000}),
       makeFlatVector<uint64_t>(
           {18446744073709551615ULL,
            2000000000000000000ULL,
            3000000000000000000ULL})});
  assertReadWithExpected("uint.parquet", rowType, expected);
}

TEST_F(ParquetReaderTest, parseUnsignedInt3) {
  auto rowType =
      ROW({"uint8", "uint16", "uint32", "uint64"},
          {SMALLINT(), INTEGER(), INTEGER(), BIGINT()});
  auto expected = makeRowVector(
      {makeFlatVector<uint16_t>({255, 2, 3}),
       makeFlatVector<uint32_t>({65535, 2000, 3000}),
       makeFlatVector<uint32_t>({4294967295, 2000000000, 3000000000}),
       makeFlatVector<uint64_t>(
           {18446744073709551615ULL,
            2000000000000000000ULL,
            3000000000000000000ULL})});
  assertReadWithExpected("uint.parquet", rowType, expected);
}

TEST_F(ParquetReaderTest, parseUnsignedInt4) {
  auto rowType =
      ROW({"uint8", "uint16", "uint32", "uint64"},
          {SMALLINT(), INTEGER(), INTEGER(), DECIMAL(20, 0)});
  auto expected = makeRowVector(
      {makeFlatVector<uint16_t>({255, 2, 3}),
       makeFlatVector<uint32_t>({65535, 2000, 3000}),
       makeFlatVector<uint32_t>({4294967295, 2000000000, 3000000000}),
       makeFlatVector<uint128_t>(
           {18446744073709551615ULL,
            2000000000000000000ULL,
            3000000000000000000ULL})});
  assertReadWithExpected("uint.parquet", rowType, expected);
}

TEST_F(ParquetReaderTest, parseUnsignedInt5) {
  auto rowType =
      ROW({"uint8", "uint16", "uint32", "uint64"},
          {SMALLINT(), INTEGER(), BIGINT(), DECIMAL(20, 0)});
  auto expected = makeRowVector(
      {makeFlatVector<uint16_t>({255, 2, 3}),
       makeFlatVector<uint32_t>({65535, 2000, 3000}),
       makeFlatVector<uint64_t>({4294967295, 2000000000, 3000000000}),
       makeFlatVector<uint128_t>(
           {18446744073709551615ULL,
            2000000000000000000ULL,
            3000000000000000000ULL})});
  assertReadWithExpected("uint.parquet", rowType, expected);
}

TEST_F(ParquetReaderTest, parseDate) {
  // date.parquet holds a single column (date: DATE) and
  // 25 rows.
  // Data is in plain uncompressed format:
  //   date: [1969-12-27 .. 1970-01-20]
  auto readerBundle = readerBuilder("date.parquet", dateSchema()).build();

  EXPECT_EQ(readerBundle.reader->numberOfRows(), 25ULL);

  auto type = readerBundle.reader->typeWithId();
  EXPECT_EQ(type->size(), 1ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type(), DATE());
  EXPECT_EQ(type->childByName("date"), col0);

  auto expected = makeRowVector({
      makeFlatVector<int32_t>(25, [](auto row) { return row - 5; }),
  });
  assertReadWithReaderAndExpected(
      dateSchema(), *readerBundle.rowReader, expected, *leafPool_);
}

TEST_F(ParquetReaderTest, parseRowMapArray) {
  // sample.parquet holds one row of type (ROW(BIGINT c0, MAP(VARCHAR,
  // ARRAY(INTEGER)) c1) c)
  auto reader = createReader("row_map_array.parquet");

  EXPECT_EQ(reader->numberOfRows(), 1ULL);

  auto type = reader->typeWithId();
  EXPECT_EQ(type->size(), 1ULL);

  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(type->childByName("c"), col0);

  auto col0_0 = col0->childAt(0);
  EXPECT_EQ(col0_0->type()->kind(), TypeKind::BIGINT);
  EXPECT_EQ(col0->childByName("c0"), col0_0);

  auto col0_1 = col0->childAt(1);
  EXPECT_EQ(col0_1->type()->kind(), TypeKind::MAP);
  EXPECT_EQ(col0->childByName("c1"), col0_1);

  auto col0_1_0 = col0_1->childAt(0);
  EXPECT_EQ(col0_1_0->type()->kind(), TypeKind::VARCHAR);

  auto col0_1_1 = col0_1->childAt(1);
  EXPECT_EQ(col0_1_1->type()->kind(), TypeKind::ARRAY);

  auto col0_1_1_0 = col0_1_1->childAt(0);
  EXPECT_EQ(col0_1_1_0->type()->kind(), TypeKind::INTEGER);
}

TEST_F(ParquetReaderTest, projectNoColumns) {
  // This is the case for count(*).
  auto rowType = ROW({}, {});
  auto readerBundle =
      readerBuilder("sample.parquet", rowType).withScanSpecOnly().build();
  auto result = BaseVector::create(rowType, 1, leafPool_.get());
  constexpr int kBatchSize = 100;
  ASSERT_TRUE(readerBundle.rowReader->next(kBatchSize, result));
  EXPECT_EQ(result->size(), 10);
  ASSERT_TRUE(readerBundle.rowReader->next(kBatchSize, result));
  EXPECT_EQ(result->size(), 10);
  ASSERT_FALSE(readerBundle.rowReader->next(kBatchSize, result));
}

TEST_F(ParquetReaderTest, parseIntDecimal) {
  // decimal_dict.parquet two columns (a: DECIMAL(7,2), b: DECIMAL(14,2)) and
  // 6 rows.
  // The physical type of the decimal columns:
  //   a: int32
  //   b: int64
  // Data is in dictionary encoding:
  //   a: [11.11, 11.11, 22.22, 22.22, 33.33, 33.33]
  //   b: [11.11, 11.11, 22.22, 22.22, 33.33, 33.33]
  auto rowType = ROW({"a", "b"}, {DECIMAL(7, 2), DECIMAL(14, 2)});
  auto readerBundle =
      readerBuilder("decimal_dict.parquet", rowType).withScanSpecOnly().build();

  EXPECT_EQ(readerBundle.reader->numberOfRows(), 6ULL);

  auto type = readerBundle.reader->typeWithId();
  EXPECT_EQ(type->size(), 2ULL);
  auto col0 = type->childAt(0);
  auto col1 = type->childAt(1);
  EXPECT_EQ(col0->type()->kind(), TypeKind::BIGINT);
  EXPECT_EQ(col1->type()->kind(), TypeKind::BIGINT);

  int64_t expectValues[3] = {1111, 2222, 3333};
  auto result = BaseVector::create(rowType, 1, leafPool_.get());
  readerBundle.rowReader->next(6, result);
  EXPECT_EQ(result->size(), 6ULL);
  auto decimals = result->as<RowVector>();
  auto a = decimals->childAt(0)
               ->loadedVector()
               ->asFlatVector<int64_t>()
               ->rawValues();
  auto b = decimals->childAt(1)
               ->loadedVector()
               ->asFlatVector<int64_t>()
               ->rawValues();
  for (int i = 0; i < 3; i++) {
    int index = 2 * i;
    EXPECT_EQ(a[index], expectValues[i]);
    EXPECT_EQ(a[index + 1], expectValues[i]);
    EXPECT_EQ(b[index], expectValues[i]);
    EXPECT_EQ(b[index + 1], expectValues[i]);
  }
}

TEST_F(ParquetReaderTest, parseMapKeyValueAsMap) {
  // map_key_value.parquet holds a single map column (key: VARCHAR, b: BIGINT)
  // and 1 row that contains 8 map entries. It is with older version of Parquet
  // and uses MAP_KEY_VALUE instead of MAP as the map SchemaElement
  // converted_type. It has 5 SchemaElements in the schema, in the format of
  // schemaIdx: <repetition> <type> name (<converted type>):
  //
  // 0: REQUIRED BOOLEAN hive_schema (UTF8)
  // 1:   OPTIONAL BOOLEAN test (MAP_KEY_VALUE)
  // 2:     REPEATED BOOLEAN map (UTF8)
  // 3:       REQUIRED BYTE_ARRAY key (UTF8)
  // 4:       OPTIONAL INT64 value (UTF8)

  auto fileSchema =
      ROW("test", createType<TypeKind::MAP>({VARCHAR(), BIGINT()}));
  auto readerBundle =
      readerBuilder("map_key_value.parquet", fileSchema).build();
  EXPECT_EQ(readerBundle.reader->numberOfRows(), 1ULL);

  auto rowType = readerBundle.reader->typeWithId();
  EXPECT_EQ(rowType->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(rowType->size(), 1ULL);

  auto mapColumnType = rowType->childAt(0);
  EXPECT_EQ(mapColumnType->type()->kind(), TypeKind::MAP);

  auto mapKeyType = mapColumnType->childAt(0);
  EXPECT_EQ(mapKeyType->type()->kind(), TypeKind::VARCHAR);

  auto mapValueType = mapColumnType->childAt(1);
  EXPECT_EQ(mapValueType->type()->kind(), TypeKind::BIGINT);

  auto expected = makeRowVector({vectorMaker_.mapVector<std::string, int64_t>(
      {{{"0", 0},
        {"1", 1},
        {"2", 2},
        {"3", 3},
        {"4", 4},
        {"5", 5},
        {"6", 6},
        {"7", 7}}})});

  assertReadWithReaderAndExpected(
      fileSchema, *readerBundle.rowReader, expected, *leafPool_);
}

TEST_F(ParquetReaderTest, parseRowArrayTest) {
  // schema:
  //   optionalPrimitive:int
  //   requiredPrimitive:int
  //   repeatedPrimitive:array<int>
  //   optionalMessage:struct<someId:int>
  //   requiredMessage:struct<someId:int>
  //   repeatedMessage:array<struct<someId:int>>
  auto outputRowType =
      ROW({"optionalPrimitive",
           "requiredPrimitive",
           "repeatedPrimitive",
           "optionalMessage",
           "requiredMessage",
           "repeatedMessage"},
          {INTEGER(),
           INTEGER(),
           ARRAY(INTEGER()),
           ROW("someId", INTEGER()),
           ROW("someId", INTEGER()),
           ARRAY(ROW("someId", INTEGER()))});
  auto readerBundle =
      readerBuilder("proto-struct-with-array.parquet", outputRowType).build();
  EXPECT_EQ(readerBundle.reader->numberOfRows(), 1ULL);
  auto type = readerBundle.reader->typeWithId();
  EXPECT_EQ(type->size(), 6ULL);
  auto col6Type = type->childAt(5);
  EXPECT_EQ(col6Type->type()->kind(), TypeKind::ARRAY);
  auto col61Type = col6Type->childAt(0);
  EXPECT_EQ(col61Type->type()->kind(), TypeKind::ROW);

  VectorPtr result = BaseVector::create(outputRowType, 0, &*leafPool_);

  ASSERT_TRUE(readerBundle.rowReader->next(1, result));
  // data: 10, 9, <empty>, null, {9}, 2 elements starting at 0 {{9}, {10}}}
  auto structArray =
      result->as<RowVector>()->childAt(5)->loadedVector()->as<ArrayVector>();
  auto structEle = structArray->elements()
                       ->as<RowVector>()
                       ->childAt(0)
                       ->asFlatVector<int32_t>()
                       ->valueAt(0);
  EXPECT_EQ(structEle, 9);
}

TEST_F(ParquetReaderTest, readSampleBigintRangeFilter) {
  // Read sample.parquet with the int filter "a BETWEEN 16 AND 20".
  FilterMap filters;
  filters.insert({"a", exec::between(16, 20)});
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(5, [](auto row) { return row + 16; }),
      makeFlatVector<double>(5, [](auto row) { return row + 16; }),
  });
  assertReadWithFilters(
      "sample.parquet", sampleSchema(), std::move(filters), expected);
}

TEST_F(ParquetReaderTest, readSampleBigintValuesUsingBitmaskFilter) {
  // Read sample.parquet with the int filter "a in 16, 17, 18, 19, 20".
  std::vector<int64_t> values{16, 17, 18, 19, 20};
  auto bigintBitmaskFilter = std::make_unique<common::BigintValuesUsingBitmask>(
      16, 20, std::move(values), false);
  FilterMap filters;
  filters.insert({"a", std::move(bigintBitmaskFilter)});
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(5, [](auto row) { return row + 16; }),
      makeFlatVector<double>(5, [](auto row) { return row + 16; }),
  });
  assertReadWithFilters(
      "sample.parquet", sampleSchema(), std::move(filters), expected);
}

TEST_F(ParquetReaderTest, readSampleEqualFilter) {
  // Read sample.parquet with the int filter "a = 16".
  FilterMap filters;
  filters.insert({"a", exec::equal(16)});

  auto expected = makeRowVector({
      makeFlatVector<int64_t>(1, [](auto row) { return row + 16; }),
      makeFlatVector<double>(1, [](auto row) { return row + 16; }),
  });

  assertReadWithFilters(
      "sample.parquet", sampleSchema(), std::move(filters), expected);
}

TEST_F(ParquetReaderTest, dateFilters) {
  // Read date.parquet with the date filter "date BETWEEN 5 AND 14".
  FilterMap filters;
  filters.insert({"date", exec::between(5, 14)});

  auto expected = makeRowVector({
      makeFlatVector<int32_t>(10, [](auto row) { return row + 5; }),
  });

  assertReadWithFilters(
      "date.parquet", dateSchema(), std::move(filters), expected);
}

TEST_F(ParquetReaderTest, intMultipleFilters) {
  // Filter int BETWEEN 102 AND 120 AND bigint BETWEEN 900 AND 1006.
  FilterMap filters;
  filters.insert({"int", exec::between(102, 120)});
  filters.insert({"bigint", exec::between(900, 1006)});

  auto expected = makeRowVector({
      makeFlatVector<int32_t>(5, [](auto row) { return row + 102; }),
      makeFlatVector<int64_t>(5, [](auto row) { return row + 1002; }),
  });

  assertReadWithFilters(
      "int.parquet", intSchema(), std::move(filters), expected);
}

TEST_F(ParquetReaderTest, doubleFilters) {
  // Read sample.parquet with the double filter "b < 10.0".
  FilterMap filters;
  filters.insert({"b", exec::lessThanDouble(10.0)});

  auto expected = makeRowVector({
      makeFlatVector<int64_t>(9, [](auto row) { return row + 1; }),
      makeFlatVector<double>(9, [](auto row) { return row + 1; }),
  });

  assertReadWithFilters(
      "sample.parquet", sampleSchema(), std::move(filters), expected);
  filters.clear();

  // Test "b <= 10.0".
  filters.insert({"b", exec::lessThanOrEqualDouble(10.0)});
  expected = makeRowVector({
      makeFlatVector<int64_t>(10, [](auto row) { return row + 1; }),
      makeFlatVector<double>(10, [](auto row) { return row + 1; }),
  });
  assertReadWithFilters(
      "sample.parquet", sampleSchema(), std::move(filters), expected);
  filters.clear();

  // Test "b between 10.0 and 14.0".
  filters.insert({"b", exec::betweenDouble(10.0, 14.0)});
  expected = makeRowVector({
      makeFlatVector<int64_t>(5, [](auto row) { return row + 10; }),
      makeFlatVector<double>(5, [](auto row) { return row + 10; }),
  });
  assertReadWithFilters(
      "sample.parquet", sampleSchema(), std::move(filters), expected);
  filters.clear();

  // Test "b > 14.0".
  filters.insert({"b", exec::greaterThanDouble(14.0)});
  expected = makeRowVector({
      makeFlatVector<int64_t>(6, [](auto row) { return row + 15; }),
      makeFlatVector<double>(6, [](auto row) { return row + 15; }),
  });
  assertReadWithFilters(
      "sample.parquet", sampleSchema(), std::move(filters), expected);
  filters.clear();

  // Test "b >= 14.0".
  filters.insert({"b", exec::greaterThanOrEqualDouble(14.0)});
  expected = makeRowVector({
      makeFlatVector<int64_t>(7, [](auto row) { return row + 14; }),
      makeFlatVector<double>(7, [](auto row) { return row + 14; }),
  });
  assertReadWithFilters(
      "sample.parquet", sampleSchema(), std::move(filters), expected);
  filters.clear();
}

TEST_F(ParquetReaderTest, varcharFilters) {
  // Test "name < 'CANADA'".
  FilterMap filters;
  filters.insert({"name", exec::lessThan("CANADA")});

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2}),
      makeFlatVector<std::string>({"ALGERIA", "ARGENTINA", "BRAZIL"}),
      makeFlatVector<int64_t>({0, 1, 1}),
  });

  auto rowType =
      ROW({"nationkey", "name", "regionkey"}, {BIGINT(), VARCHAR(), BIGINT()});

  assertReadWithFilters(
      "nation.parquet", rowType, std::move(filters), expected);
  filters.clear();

  // Test "name <= 'CANADA'".
  filters.insert({"name", exec::lessThanOrEqual("CANADA")});
  expected = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2, 3}),
      makeFlatVector<std::string>({"ALGERIA", "ARGENTINA", "BRAZIL", "CANADA"}),
      makeFlatVector<int64_t>({0, 1, 1, 1}),
  });
  assertReadWithFilters(
      "nation.parquet", rowType, std::move(filters), expected);
  filters.clear();

  // Test "name > UNITED KINGDOM".
  filters.insert({"name", exec::greaterThan("UNITED KINGDOM")});
  expected = makeRowVector({
      makeFlatVector<int64_t>({21, 24}),
      makeFlatVector<std::string>({"VIETNAM", "UNITED STATES"}),
      makeFlatVector<int64_t>({2, 1}),
  });
  assertReadWithFilters(
      "nation.parquet", rowType, std::move(filters), expected);
  filters.clear();

  // Test "name >= 'UNITED KINGDOM'".
  filters.insert({"name", exec::greaterThanOrEqual("UNITED KINGDOM")});
  expected = makeRowVector({
      makeFlatVector<int64_t>({21, 23, 24}),
      makeFlatVector<std::string>(
          {"VIETNAM", "UNITED KINGDOM", "UNITED STATES"}),
      makeFlatVector<int64_t>({2, 3, 1}),
  });
  assertReadWithFilters(
      "nation.parquet", rowType, std::move(filters), expected);
  filters.clear();

  // Test "name = 'CANADA'".
  filters.insert({"name", exec::equal("CANADA")});
  expected = makeRowVector({
      makeFlatVector<int64_t>(1, [](auto row) { return row + 3; }),
      makeFlatVector<std::string>({"CANADA"}),
      makeFlatVector<int64_t>(1, [](auto row) { return row + 1; }),
  });
  assertReadWithFilters(
      "nation.parquet", rowType, std::move(filters), expected);
  filters.clear();

  // Test "name IN ('CANADA', 'UNITED KINGDOM')".
  filters.insert({"name", exec::in({std::string("CANADA"), "UNITED KINGDOM"})});
  expected = makeRowVector({
      makeFlatVector<int64_t>({3, 23}),
      makeFlatVector<std::string>({"CANADA", "UNITED KINGDOM"}),
      makeFlatVector<int64_t>({1, 3}),
  });
  assertReadWithFilters(
      "nation.parquet", rowType, std::move(filters), expected);
  filters.clear();

  // Test "name IN ('UNITED STATES', 'CANADA', 'INDIA', 'RUSSIA')".
  filters.insert(
      {"name",
       exec::in({std::string("UNITED STATES"), "INDIA", "CANADA", "RUSSIA"})});
  expected = makeRowVector({
      makeFlatVector<int64_t>({3, 8, 22, 24}),
      makeFlatVector<std::string>(
          {"CANADA", "INDIA", "RUSSIA", "UNITED STATES"}),
      makeFlatVector<int64_t>({1, 2, 3, 1}),
  });
  assertReadWithFilters(
      "nation.parquet", rowType, std::move(filters), expected);
  filters.clear();
}

TEST_F(ParquetReaderTest, readDifferentEncodingsWithFilter) {
  FilterMap filters;
  filters.insert({"n_1", exec::equal(1)});
  auto rowType = ROW({"n_0", "n_1", "n_2"}, {INTEGER(), INTEGER(), VARCHAR()});
  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 1, 2, 3, 4, 6, 9}),
      makeFlatVector<int32_t>(10, [](auto /*row*/) { return 1; }),
      makeNullableFlatVector<std::string>(
          {"A",
           "B",
           std::nullopt,
           std::nullopt,
           "A",
           "B",
           std::nullopt,
           std::nullopt,
           "F",
           std::nullopt}),
  });
  assertReadWithFilters(
      "different_encodings_with_filter.parquet",
      rowType,
      std::move(filters),
      expected);
}

// This test is to verify filterRowGroups() doesn't throw the fileOffset Velox
// check failure
TEST_F(ParquetReaderTest, filterRowGroups) {
  // decimal_no_ColumnMetadata.parquet has one columns a: DECIMAL(9,1). It
  // doesn't have ColumnMetaData, and rowGroups_[0].columns[0].file_offset is 0.
  auto rowType = ROW("_c0", DECIMAL(9, 1));
  auto readerBundle =
      readerBuilder("decimal_no_ColumnMetadata.parquet", rowType)
          .withScanSpecOnly()
          .build();

  EXPECT_EQ(readerBundle.reader->numberOfRows(), 10ULL);
}

TEST_F(ParquetReaderTest, shouldIgnoreStatsForParquetMRVersions) {
  SemanticVersion v181("parquet-mr", 1, 8, 1);
  ParquetStatsContext ctx181{std::optional<SemanticVersion>(v181)};
  EXPECT_TRUE(ctx181.shouldIgnoreStatistics(thrift::Type::BYTE_ARRAY))
      << "ParquetStatsContext(parquet-mr 1.8.1) should ignore string stats";

  SemanticVersion v182("parquet-mr", 1, 8, 2);
  ParquetStatsContext ctx182{std::optional<SemanticVersion>(v182)};
  EXPECT_FALSE(ctx182.shouldIgnoreStatistics(thrift::Type::BYTE_ARRAY))
      << "ParquetStatsContext(parquet-mr 1.8.2) should not ignore string stats";
}

// This test is to verify filterRowGroups() doesn't fail if offset is 0
TEST_F(ParquetReaderTest, filterRowGroupsWithZeroOffset) {
  auto rowType = ROW("IDX", INTEGER());
  auto readerBundle = readerBuilder("zero_offset_row_group.parquet", rowType)
                          .withScanSpecOnly()
                          .build();

  EXPECT_EQ(readerBundle.reader->numberOfRows(), 1L);
}

TEST_F(ParquetReaderTest, parseLongTagged) {
  // This is a case for long with annonation read
  auto reader = createReader("tagged_long.parquet");

  EXPECT_EQ(reader->numberOfRows(), 4ULL);

  auto type = reader->typeWithId();
  EXPECT_EQ(type->size(), 1ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::BIGINT);
  EXPECT_EQ(type->childByName("_c0"), col0);
}

TEST_F(ParquetReaderTest, preloadSmallFile) {
  const std::string sample(getExampleFilePath("sample.parquet"));

  auto file = std::make_shared<LocalReadFile>(sample);
  auto input = std::make_unique<BufferedInput>(file, *leafPool_);

  auto reader = std::make_unique<ParquetReader>(
      std::move(input), makeDefaultReaderOptions());

  auto rowReader = createRowReaderFromReader(*reader, sampleSchema());

  // Ensure the input is small parquet file.
  const auto fileSize = file->size();
  ASSERT_TRUE(
      fileSize <= dwio::common::ReaderOptions::kDefaultFilePreloadThreshold ||
      fileSize <= dwio::common::ReaderOptions::kDefaultFooterSpeculativeIoSize);

  // Check the whole file already loaded.
  ASSERT_EQ(file->bytesRead(), fileSize);

  // Reset bytes read to check for duplicate reads.
  file->resetBytesRead();

  constexpr int kBatchSize = 10;
  auto result = BaseVector::create(sampleSchema(), 1, leafPool_.get());
  while (rowReader->next(kBatchSize, result)) {
    // Check no duplicate reads.
    ASSERT_EQ(file->bytesRead(), 0);
  }
}

TEST_F(ParquetReaderTest, prefetchRowGroups) {
  auto rowType = ROW("id", BIGINT());
  const int numRowGroups = 4;

  auto readerOptions = makeDefaultReaderOptions();
  // Disable preload of file.
  readerOptions.setFilePreloadThreshold(0);

  // Test different number of prefetch row groups.
  // 2: Less than total number of row groups.
  // 4: Exactly as total number of row groups.
  // 10: More than total number of row groups.
  const std::vector<int> numPrefetchRowGroups{
      dwio::common::ReaderOptions::kDefaultPrefetchRowGroups, 2, 4, 10};
  for (auto numPrefetch : numPrefetchRowGroups) {
    readerOptions.setPrefetchRowGroups(numPrefetch);

    auto reader = createReader("multiple_row_groups.parquet", readerOptions);
    EXPECT_EQ(reader->fileMetaData().numRowGroups(), numRowGroups);

    auto rowReader = createRowReaderFromReader(*reader, rowType, false);
    auto parquetRowReader = dynamic_cast<ParquetRowReader*>(rowReader.get());

    constexpr int kBatchSize = 1000;
    auto result = BaseVector::create(rowType, kBatchSize, pool_.get());

    for (int i = 0; i < numRowGroups; i++) {
      if (i > 0) {
        // If it's not the first row group, check if the previous row group has
        // been evicted.
        EXPECT_FALSE(parquetRowReader->isRowGroupBuffered(i - 1));
      }
      EXPECT_TRUE(parquetRowReader->isRowGroupBuffered(i));
      if (i < numRowGroups - 1) {
        // If it's not the last row group, check if the configured number of
        // row groups have been prefetched.
        for (int j = 1; j <= numPrefetch && i + j < numRowGroups; j++) {
          EXPECT_TRUE(parquetRowReader->isRowGroupBuffered(i + j));
        }
      }

      // Read current row group.
      auto actualRows = parquetRowReader->next(kBatchSize, result);
      // kBatchSize should be large enough to hold the entire row group.
      EXPECT_LE(actualRows, kBatchSize);
      // Advance to the next row group.
      parquetRowReader->nextRowNumber();
    }
  }
}

TEST_F(ParquetReaderTest, testEmptyRowGroups) {
  // empty_row_groups.parquet contains empty row groups
  auto fileSchema = ROW("a", INTEGER());
  auto readerBundle =
      readerBuilder("empty_row_groups.parquet", fileSchema).build();
  EXPECT_EQ(readerBundle.reader->numberOfRows(), 5ULL);

  auto rowType = readerBundle.reader->typeWithId();
  EXPECT_EQ(rowType->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(rowType->size(), 1ULL);

  auto integerType = rowType->childAt(0);
  EXPECT_EQ(integerType->type()->kind(), TypeKind::INTEGER);

  auto expected = makeRowVector({makeFlatVector<int32_t>({0, 3, 3, 3, 3})});

  assertReadWithReaderAndExpected(
      fileSchema, *readerBundle.rowReader, expected, *leafPool_);
}

TEST_F(ParquetReaderTest, testEnumType) {
  // enum_type.parquet contains 1 column (ENUM) with 3 rows.
  auto fileSchema = ROW("test", VARCHAR());
  auto readerBundle = readerBuilder("enum_type.parquet", fileSchema).build();
  EXPECT_EQ(readerBundle.reader->numberOfRows(), 3ULL);

  auto rowType = readerBundle.reader->typeWithId();
  EXPECT_EQ(rowType->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(rowType->size(), 1ULL);

  EXPECT_EQ(rowType->childAt(0)->type()->kind(), TypeKind::VARCHAR);

  auto expected =
      makeRowVector({makeFlatVector<StringView>({"FOO", "BAR", "FOO"})});

  assertReadWithReaderAndExpected(
      fileSchema, *readerBundle.rowReader, expected, *leafPool_);
}

TEST_F(ParquetReaderTest, readVarbinaryFromFLBA) {
  const std::string filename("varbinary_flba.parquet");
  auto selectedType = ROW("flba_field", VARBINARY());
  auto readerBundle = readerBuilder(filename, selectedType).build();

  auto type = readerBundle.reader->typeWithId();
  EXPECT_EQ(type->size(), 8ULL);
  auto flbaCol =
      std::static_pointer_cast<const ParquetTypeWithId>(type->childAt(6));
  EXPECT_EQ(flbaCol->name_, "flba_field");
  EXPECT_EQ(flbaCol->parquetType_, thrift::Type::FIXED_LEN_BYTE_ARRAY);

  auto expected = std::string(1024, '*');
  VectorPtr result = BaseVector::create(selectedType, 0, &(*leafPool_));
  readerBundle.rowReader->next(1, result);
  EXPECT_EQ(
      expected,
      result->as<RowVector>()
          ->childAt(0)
          ->loadedVector()
          ->asFlatVector<StringView>()
          ->valueAt(0));
}

TEST_F(ParquetReaderTest, readBinaryAsStringFromNation) {
  const std::string filename("nation.parquet");

  auto readerOptions = makeDefaultReaderOptions();
  auto outputRowType =
      ROW({"nationkey", "name", "regionkey", "comment"},
          {BIGINT(), VARCHAR(), BIGINT(), VARCHAR()});

  readerOptions.setFileSchema(outputRowType);
  auto reader = createReader(filename, readerOptions);
  EXPECT_EQ(reader->numberOfRows(), 25ULL);
  auto rowType = reader->typeWithId();
  EXPECT_EQ(rowType->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(rowType->size(), 4ULL);
  EXPECT_EQ(rowType->childAt(1)->type()->kind(), TypeKind::VARCHAR);

  auto rowReader = createRowReaderFromReader(*reader, outputRowType);

  auto expected = std::string("ALGERIA");
  VectorPtr result = BaseVector::create(outputRowType, 0, &(*leafPool_));
  rowReader->next(1, result);
  auto nameVector = result->as<RowVector>()->childAt(1);
  ASSERT_TRUE(isLazyNotLoaded(*nameVector));
  EXPECT_EQ(
      expected,
      nameVector->loadedVector()->asFlatVector<StringView>()->valueAt(0));
}

TEST_F(ParquetReaderTest, readComplexType) {
  const std::string filename("complex_with_varchar_varbinary.parquet");

  auto readerOptions = makeDefaultReaderOptions();
  auto outputRowType =
      ROW({"a", "b", "c", "d"},
          {ARRAY(VARCHAR()),
           ARRAY(VARBINARY()),
           MAP(VARCHAR(), BIGINT()),
           MAP(VARBINARY(), BIGINT())});

  readerOptions.setFileSchema(outputRowType);
  auto reader = createReader(filename, readerOptions);
  EXPECT_EQ(reader->numberOfRows(), 1);
  auto rowType = reader->rowType();
  EXPECT_EQ(rowType->kind(), TypeKind::ROW);
  EXPECT_EQ(rowType->size(), 4);
  EXPECT_EQ(*rowType, *outputRowType);

  auto rowReader = createRowReaderFromReader(*reader, outputRowType);

  VectorPtr result = BaseVector::create(outputRowType, 0, &(*leafPool_));
  rowReader->next(1, result);
  auto aColVector = result->as<RowVector>()
                        ->childAt(0)
                        ->loadedVector()
                        ->as<ArrayVector>()
                        ->elements();
  EXPECT_EQ(aColVector->size(), 3);
  EXPECT_EQ(aColVector->encoding(), VectorEncoding::Simple::DICTIONARY);
  EXPECT_EQ(
      aColVector->asUnchecked<DictionaryVector<StringView>>()->valueAt(0).str(),
      "AAAA");

  auto cColVector =
      result->as<RowVector>()->childAt(2)->loadedVector()->as<MapVector>();
  auto mapKeys = cColVector->mapKeys();
  EXPECT_EQ(mapKeys->size(), 2);
  EXPECT_EQ(mapKeys->encoding(), VectorEncoding::Simple::DICTIONARY);
  EXPECT_EQ(
      mapKeys->asUnchecked<DictionaryVector<StringView>>()->valueAt(0).str(),
      "foo");
}

TEST_F(ParquetReaderTest, readFixedLenBinaryAsStringFromUuid) {
  const std::string filename("uuid.parquet");

  auto readerOptions = makeDefaultReaderOptions();
  auto outputRowType = ROW("uuid_field", VARCHAR());

  readerOptions.setFileSchema(outputRowType);
  auto reader = createReader(filename, readerOptions);
  EXPECT_EQ(reader->numberOfRows(), 3ULL);
  auto rowType = reader->typeWithId();
  EXPECT_EQ(rowType->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(rowType->size(), 1ULL);
  EXPECT_EQ(rowType->childAt(0)->type()->kind(), TypeKind::VARCHAR);

  auto rowReader = createRowReaderFromReader(*reader, outputRowType);

  auto expected = std::string("5468454a-363f-ccc8-7d0b-76072a75dfaa");
  VectorPtr result = BaseVector::create(outputRowType, 0, &(*leafPool_));
  rowReader->next(1, result);
  auto uuidVector = result->as<RowVector>()->childAt(0);
  ASSERT_TRUE(isLazyNotLoaded(*uuidVector));
  EXPECT_EQ(
      expected,
      uuidVector->loadedVector()->asFlatVector<StringView>()->valueAt(0));
}

TEST_F(ParquetReaderTest, testV2PageWithZeroMaxDefRep) {
  auto outputRowType = ROW("regionkey", BIGINT());
  auto readerBundle = readerBuilder("v2_page.parquet", outputRowType).build();
  EXPECT_EQ(readerBundle.reader->numberOfRows(), 5ULL);

  auto rowType = readerBundle.reader->typeWithId();
  EXPECT_EQ(rowType->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(rowType->size(), 1ULL);

  EXPECT_EQ(rowType->childAt(0)->type()->kind(), TypeKind::BIGINT);

  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})});

  assertReadWithReaderAndExpected(
      outputRowType, *readerBundle.rowReader, expected, *leafPool_);
}

TEST_F(ParquetReaderTest, readComplexTypeWithV2Page) {
  auto outputRowType =
      ROW({"nums", "props"}, {ARRAY(INTEGER()), MAP(VARCHAR(), INTEGER())});
  auto readerBundle =
      readerBuilder("complex_type_v2_page.parquet", outputRowType).build();
  EXPECT_EQ(readerBundle.reader->numberOfRows(), 1ULL);

  auto rowType = readerBundle.reader->typeWithId();
  EXPECT_EQ(rowType->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(rowType->size(), 2ULL);

  auto expected = makeRowVector(
      {makeArrayVectorFromJson<int32_t>({"[4 ,5]"}),
       makeMapVectorFromJson<std::string, int32_t>(
           {"{\"x\": 99, \"y\": 100}"})});
  assertReadWithReaderAndExpected(
      outputRowType, *readerBundle.rowReader, expected, *leafPool_);
}

TEST_F(ParquetReaderTest, arrayOfMapOfIntKeyArrayValue) {
  //  The Schema is of type
  //  message hive_schema {
  //    optional group test (LIST) {
  //      repeated group array (MAP) {
  //        repeated group key_value (MAP_KEY_VALUE) {
  //          required binary key (UTF8);
  //          optional group value (LIST) {
  //            repeated int32 array;
  //          }
  //        }
  //      }
  //    }
  //  }
  const std::string expectedVeloxType =
      "ROW<test:ARRAY<MAP<VARCHAR,ARRAY<INTEGER>>>>";
  auto rowType = ROW("test", ARRAY(MAP(VARCHAR(), ARRAY(INTEGER()))));
  auto readerBundle =
      readerBuilder("array_of_map_of_int_key_array_value.parquet", rowType)
          .withScanSpecOnly()
          .build();
  EXPECT_EQ(readerBundle.reader->rowType()->toString(), expectedVeloxType);
  auto type = readerBundle.reader->typeWithId();
  auto result = BaseVector::create(rowType, 10, leafPool_.get());
  constexpr int kBatchSize = 1000;
  while (readerBundle.rowReader->next(kBatchSize, result)) {
  }
}

TEST_F(ParquetReaderTest, arrayOfMapOfIntKeyStructValue) {
  //  The Schema is of type
  //   message hive_schema {
  //    optional group test (LIST) {
  //      repeated group array (MAP) {
  //        repeated group key_value (MAP_KEY_VALUE) {
  //          required int32 key;
  //          optional group value {
  //            optional binary stringfield (UTF8);
  //            optional int64 longfield;
  //          }
  //        }
  //      }
  //    }
  //  }
  const std::string expectedVeloxType =
      "ROW<test:ARRAY<MAP<INTEGER,ROW<stringfield:VARCHAR,longfield:BIGINT>>>>";
  auto reader = createReader("array_of_map_of_int_key_struct_value.parquet");
  EXPECT_EQ(reader->rowType()->toString(), expectedVeloxType);
  auto type = reader->typeWithId();
  auto rowType = reader->rowType();
  auto rowReader = createRowReaderFromReader(*reader, rowType, false);
  auto result = BaseVector::create(rowType, 10, leafPool_.get());
  constexpr int kBatchSize = 1000;
  while (rowReader->next(kBatchSize, result)) {
  }
}

TEST_F(ParquetReaderTest, struct_of_array_of_array) {
  //  The Schema is of type
  //  message hive_schema {
  //    optional group test {
  //      optional group stringarrayfield (LIST) {
  //        repeated group array (LIST) {
  //          repeated binary array (UTF8);
  //        }
  //      }
  //      optional group intarrayfield (LIST) {
  //        repeated group array (LIST) {
  //          repeated int32 array;
  //        }
  //      }
  //    }
  //  }
  const std::string expectedVeloxType =
      "ROW<test:ROW<stringarrayfield:ARRAY<ARRAY<VARCHAR>>,intarrayfield:ARRAY<ARRAY<INTEGER>>>>";
  auto rowType =
      ROW("test",
          ROW({"stringarrayfield", "intarrayfield"},
              {ARRAY(ARRAY(VARCHAR())), ARRAY(ARRAY(INTEGER()))}));
  auto readerBundle = readerBuilder("struct_of_array_of_array.parquet", rowType)
                          .withScanSpecOnly()
                          .build();
  auto type = readerBundle.reader->typeWithId();
  EXPECT_EQ(type->size(), 1ULL);
  EXPECT_EQ(readerBundle.reader->rowType()->toString(), expectedVeloxType);

  auto test_column = type->childAt(0);
  EXPECT_EQ(test_column->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(type->childByName("test"), test_column);

  // test_column has 2 children
  EXPECT_EQ(test_column->size(), 2ULL);
  // explore 1st child of test_column
  auto stringarrayfield_column = test_column->childAt(0);
  EXPECT_EQ(stringarrayfield_column->type()->kind(), TypeKind::ARRAY);

  // stringarrayfield_column column has 1 child
  EXPECT_EQ(stringarrayfield_column->size(), 1ULL);
  // explore 1st child of stringarrayfield_column
  auto array_column = stringarrayfield_column->childAt(0);
  EXPECT_EQ(array_column->type()->kind(), TypeKind::ARRAY);

  // array_column column has 1 child
  EXPECT_EQ(array_column->size(), 1ULL);
  // explore 1st child of array_column
  auto array_leaf_column = array_column->childAt(0);
  EXPECT_EQ(array_leaf_column->type()->kind(), TypeKind::VARCHAR);

  // explore 2nd child of test_column
  auto intarrayfield_column = test_column->childAt(1);
  EXPECT_EQ(intarrayfield_column->type()->kind(), TypeKind::ARRAY);
  EXPECT_EQ(test_column->childByName("intarrayfield"), intarrayfield_column);

  // intarrayfield_column column has 1 child
  EXPECT_EQ(intarrayfield_column->size(), 1ULL);
  // explore 1st child of intarrayfield_column
  auto array_column_for_intarrayfield = intarrayfield_column->childAt(0);
  EXPECT_EQ(array_column_for_intarrayfield->type()->kind(), TypeKind::ARRAY);

  // array_column_for_intarrayfield column has 1 child
  EXPECT_EQ(array_column_for_intarrayfield->size(), 1ULL);
  // explore 1st child
  auto array_leaf_column_for_intarrayfield =
      array_column_for_intarrayfield->childAt(0);
  EXPECT_EQ(
      array_leaf_column_for_intarrayfield->type()->kind(), TypeKind::INTEGER);

  auto result = BaseVector::create(rowType, 10, leafPool_.get());
  constexpr int kBatchSize = 1000;
  while (readerBundle.rowReader->next(kBatchSize, result)) {
  }
}

TEST_F(ParquetReaderTest, testLzoDataPage) {
  auto outputRowType = ROW(
      {"test"},
      {ROW({"intfield", "stringarrayfield"}, {INTEGER(), ARRAY(VARCHAR())})});
  auto readerBundle = readerBuilder("lzo.parquet", outputRowType).build();
  EXPECT_EQ(readerBundle.reader->numberOfRows(), 23'547ULL);

  VectorPtr result = BaseVector::create(outputRowType, 0, &*leafPool_);
  readerBundle.rowReader->next(23'547ULL, result);
  EXPECT_EQ(23'547ULL, result->size());
  auto* rowVector = result->asUnchecked<RowVector>();
  auto* values =
      rowVector->childAt(0)->loadedVector()->asUnchecked<RowVector>();
  auto* intField = values->childAt(0)->loadedVector()->asFlatVector<int32_t>();
  auto* stringArray = values->childAt(1)->loadedVector()->as<ArrayVector>();
  EXPECT_EQ(intField->valueAt(0), 1);
  EXPECT_EQ(intField->valueAt(23'546), 13);
  EXPECT_EQ(
      stringArray->elements()->asFlatVector<StringView>()->valueAt(0).str(),
      "0");
  EXPECT_EQ(
      stringArray->elements()
          ->asFlatVector<StringView>()
          ->valueAt(31'232)
          .str(),
      "31232");
}

TEST_F(ParquetReaderTest, testEmptyV2DataPage) {
  auto outputRowType = ROW("test", REAL());
  auto readerBundle =
      readerBuilder("empty_v2datapage.parquet", outputRowType).build();
  EXPECT_EQ(readerBundle.reader->numberOfRows(), 30001ULL);
  EXPECT_EQ(*(readerBundle.reader->typeWithId()->type()), *outputRowType);

  auto expected = makeRowVector({makeFlatVector<float>(
      30001, [](auto /*row*/) { return 1; }, nullEvery(1))});

  assertReadWithReaderAndExpected(
      outputRowType, *readerBundle.rowReader, expected, *leafPool_);
}

TEST_F(ParquetReaderTest, parquet251) {
  FilterMap filters;
  filters.insert({"str", exec::equal("2")});
  auto expected = makeRowVector({
      makeFlatVector<std::string>({"2"}),
  });
  auto rowType = ROW("str", VARCHAR());
  assertReadWithFilters(
      "parquet-251.parquet", rowType, std::move(filters), expected);
}

TEST_F(ParquetReaderTest, fileColumnVarcharToMetadataColumnMismatchTest) {
  auto readerOptions = makeDefaultReaderOptions();

  auto runVarcharColTest = [&](const TypePtr& requestedType) {
    // The type in the file is a BYTE_ARRAY resolving to VARCHAR.
    // The requested type must match with what is requested as otherwise:
    // - errors occur in the column readers
    // - SIGSEGVs can be encountered during partitioning and subsequent
    // operators following the table scan
    auto outputRowType =
        ROW({"nationkey", "name", "regionkey", "comment"},
            {BIGINT(), requestedType, BIGINT(), VARCHAR()});

    // Sets the metadata schema requested, for example from Hive, and not the
    // schema from the file.
    readerOptions.setFileSchema(outputRowType);
    VELOX_ASSERT_THROW(
        createReader("nation.parquet", readerOptions),
        fmt::format(
            "Converted type VARCHAR is not allowed for requested type {} for file column 'name'",
            requestedType->toString()));
  };

  auto types = std::vector<TypePtr>{
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      DECIMAL(10, 2),
      REAL(),
      DOUBLE(),
      TIMESTAMP(),
      DATE(),
      VARBINARY()};

  for (const auto& type : types) {
    runVarcharColTest(type);
  }
}

TEST_F(ParquetReaderTest, readerWithSchema) {
  // Create an in-memory writer.
  auto sink = std::make_unique<MemorySink>(
      1024 * 1024, dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto sinkPtr = sink.get();
  const auto data = makeRowVector(
      {makeFlatVector<int64_t>({1}),
       makeArrayVectorFromJson<int32_t>({"[4 ,5]"})});
  parquet::ParquetWriterOptions writerOptions;
  dwio::common::WriterOptions options;
  options.formatSpecificOptions =
      std::make_shared<parquet::ParquetWriterOptions>(writerOptions);

  // key, element are Parquet reserved keywords.
  // Ensure we handle them properly during the schema inference.
  auto schema = ROW({"key", "element"}, {BIGINT(), ARRAY(INTEGER())});

  auto writer = std::make_unique<facebook::velox::parquet::Writer>(
      std::move(sink), options, rootPool_, schema);
  writer->write(data);
  writer->close();

  // Create the reader.
  auto readerOptions = makeDefaultReaderOptions();
  readerOptions.setFileSchema(schema);
  std::string dataBuf(sinkPtr->data(), sinkPtr->size());
  auto file = std::make_shared<InMemoryReadFile>(std::move(dataBuf));
  auto buffer = std::make_unique<dwio::common::BufferedInput>(
      file, readerOptions.memoryPool());
  ParquetReader reader(std::move(buffer), readerOptions);

  EXPECT_EQ(reader.rowType()->toString(), schema->toString());
}

TEST_F(ParquetReaderTest, columnStatistics) {
  auto data = makeRowVector(
      {"a", "b", "c"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<double>({1.1, 2.2, 3.3, 4.4, 5.5}),
          makeFlatVector<std::string>({"aaa", "bbb", "ccc", "ddd", "eee"}),
      });

  auto* sink = write(data);
  auto reader = createReaderInMemory(*sink);
  const auto& schema = reader->typeWithId();

  // Root ROW type — no stats for non-leaf.
  EXPECT_EQ(reader->columnStatistics(schema->id()), nullptr);

  // Out of range.
  EXPECT_EQ(reader->columnStatistics(schema->maxId() + 1), nullptr);

  // BIGINT column.
  {
    auto stats = reader->columnStatistics(schema->childByName("a")->id());
    ASSERT_NE(stats, nullptr);
    EXPECT_EQ(stats->getNumberOfValues(), 5);
    EXPECT_FALSE(stats->hasNull().value());
    auto* intStats =
        dynamic_cast<dwio::common::IntegerColumnStatistics*>(stats.get());
    ASSERT_NE(intStats, nullptr);
    EXPECT_EQ(intStats->getMinimum(), 1);
    EXPECT_EQ(intStats->getMaximum(), 5);
  }

  // DOUBLE column.
  {
    auto stats = reader->columnStatistics(schema->childByName("b")->id());
    ASSERT_NE(stats, nullptr);
    EXPECT_EQ(stats->getNumberOfValues(), 5);
    EXPECT_FALSE(stats->hasNull().value());
    auto* doubleStats =
        dynamic_cast<dwio::common::DoubleColumnStatistics*>(stats.get());
    ASSERT_NE(doubleStats, nullptr);
    EXPECT_EQ(doubleStats->getMinimum(), 1.1);
    EXPECT_EQ(doubleStats->getMaximum(), 5.5);
  }

  // VARCHAR column.
  {
    auto stats = reader->columnStatistics(schema->childByName("c")->id());
    ASSERT_NE(stats, nullptr);
    EXPECT_EQ(stats->getNumberOfValues(), 5);
    EXPECT_FALSE(stats->hasNull().value());
    auto* stringStats =
        dynamic_cast<dwio::common::StringColumnStatistics*>(stats.get());
    ASSERT_NE(stringStats, nullptr);
    EXPECT_EQ(stringStats->getMinimum(), "aaa");
    EXPECT_EQ(stringStats->getMaximum(), "eee");
  }
}

TEST_F(ParquetReaderTest, columnStatisticsWithNulls) {
  auto data = makeRowVector(
      {"a"},
      {
          makeNullableFlatVector<int64_t>(
              {1, std::nullopt, 3, std::nullopt, 5}),
      });

  auto* sink = write(data);
  auto reader = createReaderInMemory(*sink);
  const auto& schema = reader->typeWithId();

  auto stats = reader->columnStatistics(schema->childByName("a")->id());
  ASSERT_NE(stats, nullptr);
  EXPECT_EQ(stats->getNumberOfValues(), 3);
  EXPECT_TRUE(stats->hasNull().value());
  auto* intStats =
      dynamic_cast<dwio::common::IntegerColumnStatistics*>(stats.get());
  ASSERT_NE(intStats, nullptr);
  EXPECT_EQ(intStats->getMinimum(), 1);
  EXPECT_EQ(intStats->getMaximum(), 5);
}

TEST_F(ParquetReaderTest, columnStatisticsMultipleRowGroups) {
  // Use a small flush size to force multiple row groups.
  dwio::common::WriterOptions writerOptions;
  writerOptions.memoryPool = rootPool_.get();
  writerOptions.flushPolicyFactory = []() {
    return std::make_unique<parquet::LambdaFlushPolicy>(
        /*rowsInRowGroup=*/5,
        /*bytesInRowGroup=*/1'024 * 1'024,
        []() { return false; });
  };

  auto data = makeRowVector(
      {"a"},
      {
          makeFlatVector<int64_t>({10, 20, 30, 40, 50, 1, 2, 3, 4, 5}),
      });

  auto* sink = write(data, writerOptions, parquet::ParquetWriterOptions{});
  auto reader = createReaderInMemory(*sink);

  // Verify we have multiple row groups.
  ASSERT_GT(reader->fileMetaData().numRowGroups(), 1);

  const auto& schema = reader->typeWithId();
  auto stats = reader->columnStatistics(schema->childByName("a")->id());
  ASSERT_NE(stats, nullptr);
  EXPECT_EQ(stats->getNumberOfValues(), 10);
  EXPECT_FALSE(stats->hasNull().value());
  auto* intStats =
      dynamic_cast<dwio::common::IntegerColumnStatistics*>(stats.get());
  ASSERT_NE(intStats, nullptr);
  // Global min/max across all row groups.
  EXPECT_EQ(intStats->getMinimum(), 1);
  EXPECT_EQ(intStats->getMaximum(), 50);
}

TEST_F(ParquetReaderTest, columnStatisticsTimestamp) {
  auto data = makeRowVector(
      {"ts"},
      {
          makeFlatVector<Timestamp>(
              {Timestamp::fromMicros(1'000'000),
               Timestamp::fromMicros(2'000'000),
               Timestamp::fromMicros(3'000'000),
               Timestamp::fromMicros(4'000'000),
               Timestamp::fromMicros(5'000'000)}),
      });

  ParquetWriterOptions writerOptions;
  writerOptions.parquetWriteTimestampUnit = TimestampPrecision::kMicroseconds;
  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  auto* sink = write(data, options, writerOptions);
  auto reader = createReaderInMemory(*sink);
  const auto& schema = reader->typeWithId();

  auto stats = reader->columnStatistics(schema->childByName("ts")->id());
  ASSERT_NE(stats, nullptr);
  EXPECT_EQ(stats->getNumberOfValues(), 5);
  EXPECT_FALSE(stats->hasNull().value());
  auto* tsStats =
      dynamic_cast<dwio::common::TimestampColumnStatistics*>(stats.get());
  ASSERT_NE(tsStats, nullptr);
  EXPECT_EQ(tsStats->getMinimum(), Timestamp::fromMicros(1'000'000));
  EXPECT_EQ(tsStats->getMaximum(), Timestamp::fromMicros(5'000'000));
}

TEST_F(ParquetReaderTest, timestampRowGroupPruning) {
  // Two row groups with non-overlapping timestamp ranges.
  auto batch1 = makeRowVector(
      {"ts"},
      {makeFlatVector<Timestamp>(
          {Timestamp::fromMicros(1'000'000),
           Timestamp::fromMicros(2'000'000),
           Timestamp::fromMicros(3'000'000)})});
  auto batch2 = makeRowVector(
      {"ts"},
      {makeFlatVector<Timestamp>(
          {Timestamp::fromMicros(10'000'000),
           Timestamp::fromMicros(11'000'000),
           Timestamp::fromMicros(12'000'000)})});

  ParquetWriterOptions writerOptions;
  writerOptions.parquetWriteTimestampUnit = TimestampPrecision::kMicroseconds;
  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  options.flushPolicyFactory = []() {
    return std::make_unique<parquet::LambdaFlushPolicy>(
        /*rowsInRowGroup=*/3,
        /*bytesInRowGroup=*/1'024 * 1'024,
        []() { return false; });
  };
  auto* sink = write({batch1, batch2}, options, writerOptions);
  auto reader = createReaderInMemory(*sink);
  ASSERT_EQ(reader->fileMetaData().numRowGroups(), 2);

  const auto rowType = ROW({"ts"}, {TIMESTAMP()});

  // Filter matches only the second row group.
  FilterMap filters;
  filters.insert(
      {"ts",
       std::make_unique<common::TimestampRange>(
           Timestamp::fromMicros(10'000'000),
           Timestamp::fromMicros(12'000'000),
           false)});
  auto expected = makeRowVector(
      {"ts"},
      {makeFlatVector<Timestamp>(
          {Timestamp::fromMicros(10'000'000),
           Timestamp::fromMicros(11'000'000),
           Timestamp::fromMicros(12'000'000)})});
  assertReadWithReaderAndFilters(
      *reader, rowType, std::move(filters), expected);
}

TEST_F(ParquetReaderTest, readTimeMillis) {
  // Write TIME data using the parquet writer.
  // The writer exports Velox TIME as Arrow time32 with milliseconds unit,
  // which maps to Parquet TIME_MILLIS (INT32).
  const auto rowType = ROW("time_col", TIME());

  auto data = makeRowVector(
      rowType->names(),
      {makeNullableFlatVector<int64_t>(
          {1, 1'000, 3'600'000, std::nullopt, 43'200'000, 86'399'999},
          TIME())});
  auto sink = write(data);

  auto readerBundle = readerBuilder(*sink, rowType).build();

  EXPECT_EQ(readerBundle.reader->numberOfRows(), 6ULL);

  auto type = readerBundle.reader->typeWithId();
  EXPECT_EQ(type->size(), 1ULL);
  auto col0 = type->childAt(0);
  EXPECT_TRUE(col0->type()->isTime());

  assertReadWithReaderAndExpected(
      rowType, *readerBundle.rowReader, data, *leafPool_);
}

TEST_F(ParquetReaderTest, readTimeMicros) {
  // Write TIME MICRO UTC data using the parquet writer.
  // The writer exports Velox TIME MICRO UTC as Arrow time64 with microseconds
  // unit, which maps to Parquet TIME_MICROS (INT64).
  const auto rowType = ROW("time_col", TIME_MICRO_UTC());

  auto data = makeRowVector(
      rowType->names(),
      {makeNullableFlatVector<int64_t>(
          {0,
           1,
           1'000,
           3'600'000'000,
           std::nullopt,
           43'200'000'000,
           86'399'999'999},
          TIME_MICRO_UTC())});
  auto sink = write(data);

  auto readerBundle = readerBuilder(*sink, rowType).build();

  EXPECT_EQ(readerBundle.reader->numberOfRows(), 7ULL);

  auto type = readerBundle.reader->typeWithId();
  EXPECT_EQ(type->size(), 1ULL);
  auto col0 = type->childAt(0);
  EXPECT_TRUE(col0->type()->isTime());
  EXPECT_TRUE(col0->type()->equivalent(*TIME_MICRO_UTC()));

  assertReadWithReaderAndExpected(
      rowType, *readerBundle.rowReader, data, *leafPool_);
}

TEST_F(ParquetReaderTest, readTimeWithMultipleColumns) {
  const auto rowType =
      ROW({"id", "time_col", "name"}, {INTEGER(), TIME(), VARCHAR()});

  auto data = makeRowVector(
      rowType->names(),
      {
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int64_t>(
              {0, 1'000, 3'600'000, 43'200'000, 86'399'999}, TIME()),
          makeFlatVector<StringView>({"a", "b", "c", "d", "e"}),
      });

  auto sink = write(data);

  auto readerBundle = readerBuilder(*sink, rowType).build();

  EXPECT_EQ(readerBundle.reader->numberOfRows(), 5ULL);

  auto type = readerBundle.reader->typeWithId();
  EXPECT_EQ(type->size(), 3ULL);
  EXPECT_EQ(type->childAt(0)->type()->kind(), TypeKind::INTEGER);
  EXPECT_TRUE(type->childAt(1)->type()->isTime());
  EXPECT_EQ(type->childAt(2)->type()->kind(), TypeKind::VARCHAR);

  assertReadWithReaderAndExpected(
      rowType, *readerBundle.rowReader, data, *leafPool_);
}

// Verifies that when the footer size exceeds the configured threshold, memory
// for the deserialized Thrift footer is reported to the pool when the row
// reader is created and released when the reader is destroyed.
TEST_F(ParquetReaderTest, thriftMemoryTracking) {
  auto readerOptions = makeThriftTrackingReaderOptions();
  const auto initialUsage = leafPool_->usedBytes();
  {
    auto reader = createReader("sample.parquet", readerOptions);
    EXPECT_EQ(reader->numberOfRows(), 20ULL);
    auto rowReaderOpts = makeRowReaderOpts(sampleSchema());
    rowReaderOpts.setScanSpec(makeScanSpec(sampleSchema()));
    auto rowReader = reader->createRowReader(rowReaderOpts);

    // Reported bytes must at least cover FileMetaData plus one ColumnChunk
    // per (row group * column); anything smaller would mean per-row-group
    // payload is not accounted for.
    const auto memoryAfterRowReader = leafPool_->usedBytes();
    const auto reportedBytes = memoryAfterRowReader - initialUsage;
    const auto numRowGroups = reader->fileMetaData().numRowGroups();
    EXPECT_GT(numRowGroups, 0);
    const auto numColumns = sampleSchema()->size();
    EXPECT_GE(
        reportedBytes,
        sizeof(thrift::FileMetaData) +
            numRowGroups * numColumns * sizeof(thrift::ColumnChunk));

    auto result = BaseVector::create(sampleSchema(), 0, leafPool_.get());
    rowReader->next(10, result);
    EXPECT_EQ(result->size(), 10);

    // Reading does not release any of the reported memory.
    EXPECT_GE(leafPool_->usedBytes(), memoryAfterRowReader);
  }

  // Destroying the reader returns memory to the pre-tracking baseline.
  EXPECT_EQ(leafPool_->usedBytes(), initialUsage);
}

// Verifies that the ParquetReader, not the RowReader, owns the tracked
// footer memory: a single external allocation is made when the reader
// is constructed, no further external allocations are made when row
// readers are created or destroyed (including a second row reader on
// the same ParquetReader), and the reservation is balanced by frees
// after the reader is destroyed.
TEST_F(ParquetReaderTest, thriftMemoryOwnedByReader) {
  auto readerOptions = makeThriftTrackingReaderOptions();
  const auto initialAllocs = leafPool_->stats().numExternalAllocs;
  const auto initialFrees = leafPool_->stats().numExternalFrees;
  {
    auto reader = createReader("sample.parquet", readerOptions);
    EXPECT_EQ(leafPool_->stats().numExternalAllocs, initialAllocs + 1);
    EXPECT_EQ(leafPool_->stats().numExternalFrees, initialFrees);

    {
      // Full-range row reader: no row groups skipped, no accounting change.
      auto rowReaderOpts = makeRowReaderOpts(sampleSchema());
      rowReaderOpts.setScanSpec(makeScanSpec(sampleSchema()));
      auto rowReader = reader->createRowReader(rowReaderOpts);
      EXPECT_EQ(leafPool_->stats().numExternalAllocs, initialAllocs + 1);
      EXPECT_EQ(leafPool_->stats().numExternalFrees, initialFrees);
    }
    EXPECT_EQ(leafPool_->stats().numExternalAllocs, initialAllocs + 1);
    EXPECT_EQ(leafPool_->stats().numExternalFrees, initialFrees);

    // A second row reader on the same ParquetReader does not allocate.
    // Range (0, 1) is out-of-file, so no row groups match — this also
    // sidesteps a pre-existing multi-row-reader prefetch limitation in
    // ParquetData::seekToRowGroup unrelated to footer accounting.
    auto emptyRangeRowReaderOpts = makeRowReaderOpts(sampleSchema());
    emptyRangeRowReaderOpts.setScanSpec(makeScanSpec(sampleSchema()));
    emptyRangeRowReaderOpts.range(0, 1);
    auto emptyRangeRowReader = reader->createRowReader(emptyRangeRowReaderOpts);
    EXPECT_EQ(leafPool_->stats().numExternalAllocs, initialAllocs + 1);
  }
  // Every external alloc is paired with at least one free across the
  // reader's lifetime (frees may include early releases from skipped
  // row groups in addition to the final release in ~ReaderBase).
  EXPECT_EQ(leafPool_->stats().numExternalAllocs, initialAllocs + 1);
  EXPECT_GE(leafPool_->stats().numExternalFrees, initialFrees + 1);
}

// Verifies that the estimated footer size is surfaced via the per-scan
// runtime stat, and appears in the runtime-metric map for operator
// stats consumers.
TEST_F(ParquetReaderTest, thriftMemoryRuntimeStat) {
  auto readerOptions = makeThriftTrackingReaderOptions();
  auto reader = createReader("sample.parquet", readerOptions);
  auto rowReaderOpts = makeRowReaderOpts(sampleSchema());
  rowReaderOpts.setScanSpec(makeScanSpec(sampleSchema()));
  auto rowReader = reader->createRowReader(rowReaderOpts);

  dwio::common::RuntimeStatistics stats;
  rowReader->updateRuntimeStats(stats);
  EXPECT_GT(stats.parquetFooterEstimatedBytes, 0);

  auto metrics = stats.toRuntimeMetricMap();
  ASSERT_TRUE(metrics.count("parquetFooterEstimatedBytes"));
  EXPECT_EQ(
      metrics["parquetFooterEstimatedBytes"].sum,
      stats.parquetFooterEstimatedBytes);
  EXPECT_EQ(
      metrics["parquetFooterEstimatedBytes"].unit,
      RuntimeCounter::Unit::kBytes);
}

// Verifies that without tracking the runtime stat stays at zero and
// the metric is not emitted in the metric map (kept off the operator
// stats panel for non-tracking scans).
TEST_F(ParquetReaderTest, thriftMemoryRuntimeStatAbsentWithoutTracking) {
  auto reader = createReader("sample.parquet");
  auto rowReaderOpts = makeRowReaderOpts(sampleSchema());
  rowReaderOpts.setScanSpec(makeScanSpec(sampleSchema()));
  auto rowReader = reader->createRowReader(rowReaderOpts);

  dwio::common::RuntimeStatistics stats;
  rowReader->updateRuntimeStats(stats);
  EXPECT_EQ(stats.parquetFooterEstimatedBytes, 0);

  auto metrics = stats.toRuntimeMetricMap();
  EXPECT_EQ(metrics.count("parquetFooterEstimatedBytes"), 0);
}

// Verifies that without setting the threshold the tracking path is
// disabled: no external allocation is reported, and ~ReaderBase has
// nothing to release.
TEST_F(ParquetReaderTest, thriftMemoryNotTrackedByDefault) {
  // Default threshold (uint64::max) leaves tracking disabled.
  const auto initialExternalAllocs = leafPool_->stats().numExternalAllocs;
  {
    auto reader = createReader("sample.parquet");
    auto rowReaderOpts = makeRowReaderOpts(sampleSchema());
    rowReaderOpts.setScanSpec(makeScanSpec(sampleSchema()));
    auto rowReader = reader->createRowReader(rowReaderOpts);
    EXPECT_EQ(leafPool_->stats().numExternalAllocs, initialExternalAllocs);
  }
  EXPECT_EQ(leafPool_->stats().numExternalAllocs, initialExternalAllocs);
  EXPECT_EQ(leafPool_->stats().numExternalFrees, 0);
}

// Verifies that when row groups are skipped via the scan range, the portion
// of the Thrift footer memory belonging to those row groups is released back
// to the pool when the row reader runs filterRowGroups.
TEST_F(ParquetReaderTest, thriftMemoryReleasedForSkippedRowGroups) {
  const auto rowType = ROW({"id"}, {BIGINT()});
  auto readerOptions = makeThriftTrackingReaderOptions();
  const auto initialUsage = leafPool_->usedBytes();
  {
    auto reader = createReader("multiple_row_groups.parquet", readerOptions);
    const auto numRowGroups = reader->fileMetaData().numRowGroups();
    ASSERT_GT(numRowGroups, 1);

    // Full footer footprint is reported at reader construction.
    const auto memoryAfterReader = leafPool_->usedBytes();
    EXPECT_GT(memoryAfterReader, initialUsage);

    // Scan range covers only row group 0; later groups are cleared.
    const auto secondRowGroupOffset =
        reader->fileMetaData().rowGroup(1).fileOffset();
    ASSERT_GT(secondRowGroupOffset, 0);

    RowReaderOptions rowReaderOpts;
    rowReaderOpts.setScanSpec(makeScanSpec(rowType));
    rowReaderOpts.range(0, secondRowGroupOffset);
    auto rowReader = reader->createRowReader(rowReaderOpts);

    // Cleared row groups' bytes are released; row group 0 still tracked.
    EXPECT_LT(leafPool_->usedBytes(), memoryAfterReader);
    EXPECT_GT(leafPool_->usedBytes(), initialUsage);
  }

  EXPECT_EQ(leafPool_->usedBytes(), initialUsage);
}
