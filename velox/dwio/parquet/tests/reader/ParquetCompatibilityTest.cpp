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
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"
#include "velox/type/Type.h"

using namespace facebook::velox;

namespace {

const auto kAllTypes =
    ROW({"id",
         "bool_col",
         "tinyint_col",
         "smallint_col",
         "int_col",
         "bigint_col",
         "float_col",
         "double_col",
         "date_string_col",
         "string_col",
         "timestamp_col"},
        {INTEGER(),
         BOOLEAN(),
         INTEGER(),
         INTEGER(),
         INTEGER(),
         BIGINT(),
         REAL(),
         DOUBLE(),
         VARBINARY(),
         VARBINARY(),
         TIMESTAMP()});

const auto kAllTypesTinyPages =
    ROW({"id",
         "bool_col",
         "tinyint_col",
         "smallint_col",
         "int_col",
         "bigint_col",
         "float_col",
         "double_col",
         "date_string_col",
         "string_col",
         "timestamp_col",
         "year",
         "month"},
        {INTEGER(),
         BOOLEAN(),
         TINYINT(),
         SMALLINT(),
         INTEGER(),
         BIGINT(),
         REAL(),
         DOUBLE(),
         VARCHAR(),
         VARCHAR(),
         TIMESTAMP(),
         INTEGER(),
         INTEGER()});

const auto kBadDataTypes =
    ROW({"boolean",
         "null",
         "uint8",
         "int8",
         "uint16",
         "int16",
         "uint32",
         "int32",
         "uint64",
         "int64",
         "float32",
         "float64",
         "string",
         "large_string",
         "timestamp_ms_gmt",
         "timestamp_ms_gmt_plus_2",
         "timestamp_ms_gmt_minus_0215",
         "timestamp_s_no_tz",
         "timestamp_us_no_tz",
         "timestamp_ns_no_tz",
         "time32_s",
         "time32_ms",
         "time64_us",
         "time64_ns",
         "date32",
         "date64",
         "binary",
         "large_binary",
         "fixed_size_binary",
         "decimal128",
         "decimal256",
         "list_boolean",
         "list_uint8",
         "list_int8",
         "list_uint16",
         "list_int16",
         "list_uint32",
         "list_int32",
         "list_uint64",
         "list_int64",
         "list_float32",
         "list_float64",
         "list_decimal128",
         "list_decimal256",
         "list_string",
         "list_large_string",
         "fixed_size_list_boolean",
         "fixed_size_list_uint8",
         "fixed_size_list_int8",
         "fixed_size_list_uint16",
         "fixed_size_list_int16",
         "fixed_size_list_uint32",
         "fixed_size_list_int32",
         "fixed_size_list_uint64",
         "fixed_size_list_int64",
         "fixed_size_list_float32",
         "fixed_size_list_float64",
         "fixed_size_list_string",
         "struct_field",
         "list_struct",
         "map_boolean",
         "map_uint8",
         "map_int8",
         "map_uint16",
         "map_int16",
         "map_uint32",
         "map_int32",
         "map_uint64",
         "map_int64",
         "map_float32",
         "map_float64",
         "map_decimal128",
         "map_decimal256",
         "map_string",
         "map_large_string",
         "map_list_string",
         "map_large_list_string",
         "map_fixed_size_list_string",
         "dict",
         "geometry"},
        {BOOLEAN(),
         INTEGER(),
         TINYINT(),
         TINYINT(),
         SMALLINT(),
         SMALLINT(),
         BIGINT(),
         INTEGER(),
         BIGINT(),
         BIGINT(),
         REAL(),
         DOUBLE(),
         VARCHAR(),
         VARCHAR(),
         TIMESTAMP(),
         TIMESTAMP(),
         TIMESTAMP(),
         TIMESTAMP(),
         TIMESTAMP(),
         TIMESTAMP(),
         TIMESTAMP(),
         TIMESTAMP(),
         TIMESTAMP(),
         TIMESTAMP(),
         TIMESTAMP(),
         TIMESTAMP(),
         VARBINARY(),
         VARBINARY(),
         VARCHAR(),
         DECIMAL(7, 3),
         DECIMAL(7, 3),
         ARRAY(BOOLEAN()),
         ARRAY(TINYINT()),
         ARRAY(TINYINT()),
         ARRAY(SMALLINT()),
         ARRAY(SMALLINT()),
         ARRAY(BIGINT()),
         ARRAY(INTEGER()),
         ARRAY(BIGINT()),
         ARRAY(BIGINT()),
         ARRAY(REAL()),
         ARRAY(DOUBLE()),
         ARRAY(DECIMAL(7, 3)),
         ARRAY(DECIMAL(7, 3)),
         ARRAY(VARCHAR()),
         ARRAY(VARCHAR()),
         ARRAY(BOOLEAN()),
         ARRAY(TINYINT()),
         ARRAY(TINYINT()),
         ARRAY(SMALLINT()),
         ARRAY(SMALLINT()),
         ARRAY(BIGINT()),
         ARRAY(INTEGER()),
         ARRAY(BIGINT()),
         ARRAY(BIGINT()),
         ARRAY(REAL()),
         ARRAY(DOUBLE()),
         ARRAY(VARCHAR()),
         ROW({"a", "b", "c", "h", "i"},
             {BIGINT(),
              DOUBLE(),
              ROW({"d", "f"}, {VARCHAR(), VARCHAR()}),
              ARRAY(BIGINT()),
              BIGINT()}),
         ARRAY(ROW({"a", "b", "c"}, {BIGINT(), DOUBLE(), DOUBLE()})),
         MAP(VARCHAR(), BOOLEAN()),
         MAP(VARCHAR(), TINYINT()),
         MAP(VARCHAR(), TINYINT()),
         MAP(VARCHAR(), SMALLINT()),
         MAP(VARCHAR(), SMALLINT()),
         MAP(VARCHAR(), BIGINT()),
         MAP(VARCHAR(), INTEGER()),
         MAP(VARCHAR(), BIGINT()),
         MAP(VARCHAR(), BIGINT()),
         MAP(VARCHAR(), REAL()),
         MAP(VARCHAR(), DOUBLE()),
         MAP(VARCHAR(), DECIMAL(7, 3)),
         MAP(VARCHAR(), DECIMAL(7, 3)),
         MAP(VARCHAR(), VARCHAR()),
         MAP(VARCHAR(), VARCHAR()),
         MAP(VARCHAR(), ARRAY(VARCHAR())),
         MAP(VARCHAR(), ARRAY(VARCHAR())),
         MAP(VARCHAR(), ARRAY(VARCHAR())),
         VARCHAR(),
         VARBINARY()});

const bool kBadData = true;

} // namespace

class ParquetCompatibilityTest : public parquet::ParquetTestBase {
 protected:
  void initReaderAndCheckSchema(
      const char* fileName,
      const uint64_t expectedRows,
      const std::shared_ptr<const RowType>& expectedTypes,
      const bool badData = false) {
    std::string relativePath = "../parquet-testing/";
    if (badData) {
      relativePath += "bad_";
    }

    const auto filePath = test::getDataFilePath(
        "velox/dwio/parquet/tests/reader", relativePath + "data/" + fileName);
    dwio::common::ReaderOptions readerOptions{leafPool_.get()};
    reader_ = createReader(filePath, readerOptions);
    EXPECT_EQ(reader_->numberOfRows(), expectedRows);
    EXPECT_TRUE(reader_->typeWithId()->type()->equivalent(*expectedTypes));
  };

 private:
  std::unique_ptr<parquet::ParquetReader> reader_;
};

TEST_F(ParquetCompatibilityTest, alltypesPlain) {
  initReaderAndCheckSchema("alltypes_plain.parquet", 8ULL, kAllTypes);
}

TEST_F(ParquetCompatibilityTest, alltypesDictionary) {
  initReaderAndCheckSchema("alltypes_dictionary.parquet", 2ULL, kAllTypes);
}

TEST_F(ParquetCompatibilityTest, alltypesTinyPagesPlain) {
  initReaderAndCheckSchema(
      "alltypes_tiny_pages_plain.parquet", 7300ULL, kAllTypesTinyPages);
}

TEST_F(ParquetCompatibilityTest, alltypesTinyPages) {
  initReaderAndCheckSchema(
      "alltypes_tiny_pages.parquet", 7300ULL, kAllTypesTinyPages);
}

TEST_F(ParquetCompatibilityTest, alltypesPlainSnappy) {
  initReaderAndCheckSchema("alltypes_plain.snappy.parquet", 2ULL, kAllTypes);
}

TEST_F(ParquetCompatibilityTest, binary) {
  initReaderAndCheckSchema(
      "binary.parquet", 12ULL, ROW({"foo"}, {VARBINARY()}));
}

TEST_F(ParquetCompatibilityTest, byteArrayDecimal) {
  initReaderAndCheckSchema(
      "byte_array_decimal.parquet", 24ULL, ROW({"value"}, {DECIMAL(4, 2)}));
}

TEST_F(ParquetCompatibilityTest, byteStreamSplitZstd) {
  initReaderAndCheckSchema(
      "byte_stream_split.zstd.parquet",
      300ULL,
      ROW({"f32", "f64"}, {REAL(), DOUBLE()}));
}

TEST_F(ParquetCompatibilityTest, byteStreamSplitExtendedGzip) {
  initReaderAndCheckSchema(
      "byte_stream_split_extended.gzip.parquet",
      200ULL,
      ROW({"float16_plain",
           "float16_byte_stream_split",
           "float_plain",
           "float_byte_stream_split",
           "double_plain",
           "double_byte_stream_split",
           "int32_plain",
           "int32_byte_stream_split",
           "int64_plain",
           "int64_byte_stream_split",
           "flba5_plain",
           "flba5_byte_stream_split",
           "decimal_plain",
           "decimal_byte_stream_split"},
          {VARBINARY(),
           VARBINARY(),
           REAL(),
           REAL(),
           DOUBLE(),
           DOUBLE(),
           INTEGER(),
           INTEGER(),
           BIGINT(),
           BIGINT(),
           VARBINARY(),
           VARBINARY(),
           DECIMAL(7, 3),
           DECIMAL(7, 3)}));
}

TEST_F(ParquetCompatibilityTest, columnChunkKeyValueMetadata) {
  initReaderAndCheckSchema(
      "column_chunk_key_value_metadata.parquet",
      0ULL,
      ROW({"column1", "column2"}, {INTEGER(), INTEGER()}));
}

TEST_F(ParquetCompatibilityTest, concatenatedGzipMembers) {
  initReaderAndCheckSchema(
      "concatenated_gzip_members.parquet",
      513ULL,
      ROW({"long_col"}, {BIGINT()}));
}

TEST_F(ParquetCompatibilityTest, dataIndexBloomEncodingStats) {
  initReaderAndCheckSchema(
      "data_index_bloom_encoding_stats.parquet",
      14ULL,
      ROW({"String"}, {VARCHAR()}));
}

TEST_F(ParquetCompatibilityTest, dataIndexBloomEncodingWithLength) {
  initReaderAndCheckSchema(
      "data_index_bloom_encoding_with_length.parquet",
      14ULL,
      ROW({"String"}, {VARCHAR()}));
}

TEST_F(ParquetCompatibilityTest, datapageV1CorruptChecksum) {
  initReaderAndCheckSchema(
      "datapage_v1-corrupt-checksum.parquet",
      5120ULL,
      ROW({"a", "b"}, {INTEGER(), INTEGER()}));
}

TEST_F(ParquetCompatibilityTest, datapageV1SnappyCompressedChecksum) {
  initReaderAndCheckSchema(
      "datapage_v1-snappy-compressed-checksum.parquet",
      5120ULL,
      ROW({"a", "b"}, {INTEGER(), INTEGER()}));
}

TEST_F(ParquetCompatibilityTest, datapageV1UncompressedChecksum) {
  initReaderAndCheckSchema(
      "datapage_v1-uncompressed-checksum.parquet",
      5120ULL,
      ROW({"a", "b"}, {INTEGER(), INTEGER()}));
}

TEST_F(ParquetCompatibilityTest, datapageV2Snappy) {
  initReaderAndCheckSchema(
      "datapage_v2.snappy.parquet",
      5ULL,
      ROW({"a", "b", "c", "d", "e"},
          {VARCHAR(), INTEGER(), DOUBLE(), BOOLEAN(), ARRAY(INTEGER())}));
}

TEST_F(ParquetCompatibilityTest, deltaBinaryPacked) {
  initReaderAndCheckSchema(
      "delta_binary_packed.parquet",
      200ULL,
      ROW({"bitwidth0",  "bitwidth1",  "bitwidth2",  "bitwidth3",  "bitwidth4",
           "bitwidth5",  "bitwidth6",  "bitwidth7",  "bitwidth8",  "bitwidth9",
           "bitwidth10", "bitwidth11", "bitwidth12", "bitwidth13", "bitwidth14",
           "bitwidth15", "bitwidth16", "bitwidth17", "bitwidth18", "bitwidth19",
           "bitwidth20", "bitwidth21", "bitwidth22", "bitwidth23", "bitwidth24",
           "bitwidth25", "bitwidth26", "bitwidth27", "bitwidth28", "bitwidth29",
           "bitwidth30", "bitwidth31", "bitwidth32", "bitwidth33", "bitwidth34",
           "bitwidth35", "bitwidth36", "bitwidth37", "bitwidth38", "bitwidth39",
           "bitwidth40", "bitwidth41", "bitwidth42", "bitwidth43", "bitwidth44",
           "bitwidth45", "bitwidth46", "bitwidth47", "bitwidth48", "bitwidth49",
           "bitwidth50", "bitwidth51", "bitwidth52", "bitwidth53", "bitwidth54",
           "bitwidth55", "bitwidth56", "bitwidth57", "bitwidth58", "bitwidth59",
           "bitwidth60", "bitwidth61", "bitwidth62", "bitwidth63", "bitwidth64",
           "int_value"},
          {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(),
           BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(),
           BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(),
           BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(),
           BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(),
           BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(),
           BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(),
           BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(),
           BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(),
           BIGINT(), BIGINT(), INTEGER()}));
}

TEST_F(ParquetCompatibilityTest, deltaByteArray) {
  initReaderAndCheckSchema(
      "delta_byte_array.parquet",
      1000ULL,
      ROW({"c_customer_id",
           "c_salutation",
           "c_first_name",
           "c_last_name",
           "c_preferred_cust_flag",
           "c_birth_country",
           "c_login",
           "c_email_address",
           "c_last_review_date"},
          {VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR()}));
}

TEST_F(ParquetCompatibilityTest, deltaEncodingOptionalColumn) {
  initReaderAndCheckSchema(
      "delta_encoding_optional_column.parquet",
      100ULL,
      ROW({"c_customer_sk",
           "c_current_cdemo_sk",
           "c_current_hdemo_sk",
           "c_current_addr_sk",
           "c_first_shipto_date_sk",
           "c_first_sales_date_sk",
           "c_birth_day",
           "c_birth_month",
           "c_birth_year",
           "c_customer_id",
           "c_salutation",
           "c_first_name",
           "c_last_name",
           "c_preferred_cust_flag",
           "c_birth_country",
           "c_email_address",
           "c_last_review_date"},
          {BIGINT(),
           BIGINT(),
           BIGINT(),
           BIGINT(),
           BIGINT(),
           BIGINT(),
           BIGINT(),
           BIGINT(),
           BIGINT(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR()}));
}

TEST_F(ParquetCompatibilityTest, deltaEncodingRequiredColumn) {
  initReaderAndCheckSchema(
      "delta_encoding_required_column.parquet",
      100ULL,
      ROW({"c_customer_sk:",
           "c_current_cdemo_sk:",
           "c_current_hdemo_sk:",
           "c_current_addr_sk:",
           "c_first_shipto_date_sk:",
           "c_first_sales_date_sk:",
           "c_birth_day:",
           "c_birth_month:",
           "c_birth_year:",
           "c_customer_id:",
           "c_salutation:",
           "c_first_name:",
           "c_last_name:",
           "c_preferred_cust_flag:",
           "c_birth_country:",
           "c_email_address:",
           "c_last_review_date:"},
          {INTEGER(),
           INTEGER(),
           INTEGER(),
           INTEGER(),
           INTEGER(),
           INTEGER(),
           INTEGER(),
           INTEGER(),
           INTEGER(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR()}));
}

TEST_F(ParquetCompatibilityTest, deltaLengthByteArray) {
  initReaderAndCheckSchema(
      "delta_length_byte_array.parquet", 1000ULL, ROW({"FRUIT"}, {VARCHAR()}));
}

TEST_F(ParquetCompatibilityTest, dictPageOffsetZero) {
  initReaderAndCheckSchema(
      "dict-page-offset-zero.parquet", 39ULL, ROW({"l_partkey"}, {INTEGER()}));
}

TEST_F(ParquetCompatibilityTest, fixedLengthByteArray) {
  initReaderAndCheckSchema(
      "fixed_length_byte_array.parquet",
      1000ULL,
      ROW({"flba_field"}, {VARBINARY()}));
}

TEST_F(ParquetCompatibilityTest, fixedLengthDecimal) {
  initReaderAndCheckSchema(
      "fixed_length_decimal.parquet", 24ULL, ROW({"value"}, {DECIMAL(25, 2)}));
}

TEST_F(ParquetCompatibilityTest, fixedLengthDecimalLegacy) {
  initReaderAndCheckSchema(
      "fixed_length_decimal_legacy.parquet",
      24ULL,
      ROW({"value"}, {DECIMAL(13, 2)}));
}

TEST_F(ParquetCompatibilityTest, float16NonzerosAndNans) {
  initReaderAndCheckSchema(
      "float16_nonzeros_and_nans.parquet", 8ULL, ROW({"x"}, {VARBINARY()}));
}

TEST_F(ParquetCompatibilityTest, float16ZerosAndNans) {
  initReaderAndCheckSchema(
      "float16_zeros_and_nans.parquet", 3ULL, ROW({"x"}, {VARBINARY()}));
}

TEST_F(ParquetCompatibilityTest, hadoopLz4Compressed) {
  initReaderAndCheckSchema(
      "hadoop_lz4_compressed.parquet",
      4ULL,
      ROW({"c0", "c1", "v11"}, {BIGINT(), VARBINARY(), DOUBLE()}));
}

TEST_F(ParquetCompatibilityTest, hadoopLz4CompressedLarger) {
  initReaderAndCheckSchema(
      "hadoop_lz4_compressed_larger.parquet",
      10000ULL,
      ROW({"a"}, {VARCHAR()}));
}

TEST_F(ParquetCompatibilityTest, incorrectMapSchema) {
  initReaderAndCheckSchema(
      "incorrect_map_schema.parquet",
      1ULL,
      ROW({"my_map"}, {MAP(VARCHAR(), VARCHAR())}));
}

TEST_F(ParquetCompatibilityTest, int32Decimal) {
  initReaderAndCheckSchema(
      "int32_decimal.parquet", 24ULL, ROW({"value"}, {DECIMAL(4, 2)}));
}

TEST_F(ParquetCompatibilityTest, int32WithNullPages) {
  initReaderAndCheckSchema(
      "int32_with_null_pages.parquet",
      1000ULL,
      ROW({"int32_field"}, {INTEGER()}));
}

TEST_F(ParquetCompatibilityTest, int64Decimal) {
  initReaderAndCheckSchema(
      "int64_decimal.parquet", 24ULL, ROW({"value"}, {DECIMAL(10, 2)}));
}

TEST_F(ParquetCompatibilityTest, largeStringMapBrotli) {
  initReaderAndCheckSchema(
      "large_string_map.brotli.parquet",
      2ULL,
      ROW({"arr"}, {MAP(VARCHAR(), INTEGER())}));
}

TEST_F(ParquetCompatibilityTest, listColumns) {
  initReaderAndCheckSchema(
      "list_columns.parquet",
      3ULL,
      ROW({"int64_list", "utf8_list"}, {ARRAY(BIGINT()), ARRAY(VARCHAR())}));
}

TEST_F(ParquetCompatibilityTest, lz4RawCompressed) {
  initReaderAndCheckSchema(
      "lz4_raw_compressed.parquet",
      4ULL,
      ROW({"c0", "c1", "v11"}, {BIGINT(), VARBINARY(), DOUBLE()}));
}

TEST_F(ParquetCompatibilityTest, lz4RawCompressedLarger) {
  initReaderAndCheckSchema(
      "lz4_raw_compressed_larger.parquet", 10000ULL, ROW({"a"}, {VARCHAR()}));
}

// FIXME(parquet): https://github.com/apache/parquet-format/issues/468
TEST_F(ParquetCompatibilityTest, DISABLED_mapNoValue) {
  initReaderAndCheckSchema(
      "map_no_value.parquet",
      3ULL,
      ROW({"my_map", "my_map_no_v", "my_list"},
          {MAP(INTEGER(), INTEGER()), ARRAY(INTEGER()), ARRAY(INTEGER())}));
}

TEST_F(ParquetCompatibilityTest, nanInStats) {
  initReaderAndCheckSchema(
      "nan_in_stats.parquet", 2ULL, ROW({"x"}, {DOUBLE()}));
}

TEST_F(ParquetCompatibilityTest, nationDictMalformed) {
  initReaderAndCheckSchema(
      "nation.dict-malformed.parquet",
      25ULL,
      ROW({"nation_key", "name", "region_key", "comment_col"},
          {INTEGER(), VARBINARY(), INTEGER(), VARBINARY()}));
}

TEST_F(ParquetCompatibilityTest, nestedListsSnappy) {
  initReaderAndCheckSchema(
      "nested_lists.snappy.parquet",
      3ULL,
      ROW({"a", "b"}, {ARRAY(ARRAY(ARRAY(VARCHAR()))), INTEGER()}));
}

TEST_F(ParquetCompatibilityTest, nestedMapsSnappy) {
  initReaderAndCheckSchema(
      "nested_maps.snappy.parquet",
      6ULL,
      ROW({"a", "b", "c"},
          {MAP(VARCHAR(), MAP(INTEGER(), BOOLEAN())), INTEGER(), DOUBLE()}));
}

TEST_F(ParquetCompatibilityTest, nestedStructsRust) {
  initReaderAndCheckSchema(
      "nested_structs.rust.parquet",
      1ULL,
      ROW({"roll_num",
           "PC_CUR",
           "CVA_2012",
           "CVA_2016",
           "BIA_3",
           "BIA_4",
           "ACTUAL_FRONTAGE",
           "ACTUAL_DEPTH",
           "ACTUAL_LOT_SIZE",
           "GLA",
           "SOURCE_GLA",
           "IPS_GLA",
           "GLA_ALL",
           "bia",
           "EFFECTIVE_LOT_SIZE",
           "effective_lot_area",
           "EFFECTIVE_FRONTAGE",
           "EFFECTIVE_DEPTH",
           "rw_area_tot",
           "effective_lot_sqft",
           "dup",
           "nonCTXT",
           "vacantland",
           "parkingbillboard",
           "cvalte10",
           "condootherhotel",
           "calculated_lot_size",
           "calculated_efflot_size",
           "missingsite",
           "missinggla",
           "missingsitegla",
           "actual_lot_size_sqft",
           "lotsize_sqft",
           "count",
           "ul_observation_date",
           "ul_tz_offset_minutes_ul_observation_date"},
          {ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {DOUBLE(), DOUBLE(), DOUBLE(), BIGINT(), DOUBLE(), DOUBLE()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {TIMESTAMP(),
                TIMESTAMP(),
                TIMESTAMP(),
                BIGINT(),
                TIMESTAMP(),
                TIMESTAMP()}),
           ROW({"min", "max", "mean", "count", "sum", "variance"},
               {BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT(), BIGINT()})}));
}

TEST_F(ParquetCompatibilityTest, nonHadoopLz4Compressed) {
  initReaderAndCheckSchema(
      "non_hadoop_lz4_compressed.parquet",
      4ULL,
      ROW({"c0", "c1", "v11"}, {BIGINT(), VARBINARY(), DOUBLE()}));
}

TEST_F(ParquetCompatibilityTest, nonnullableImpala) {
  initReaderAndCheckSchema(
      "nonnullable.impala.parquet",
      1ULL,
      ROW({"ID",
           "Int_Array",
           "int_array_array",
           "Int_Map",
           "int_map_array",
           "nested_Struct"},
          {BIGINT(),
           ARRAY(INTEGER()),
           ARRAY(ARRAY(INTEGER())),
           MAP(VARCHAR(), INTEGER()),
           ARRAY(MAP(VARCHAR(), INTEGER())),
           ROW({"a", "B", "c", "G"},
               {INTEGER(),
                ARRAY(INTEGER()),
                ROW({"D"},
                    {ARRAY(ARRAY(ROW({"e", "f"}, {INTEGER(), VARCHAR()})))}),
                MAP(VARCHAR(),
                    ROW({"h"}, {ROW({"i"}, {ARRAY(DOUBLE())})}))})}));
}

TEST_F(ParquetCompatibilityTest, nullList) {
  initReaderAndCheckSchema(
      "null_list.parquet", 1ULL, ROW({"emptylist"}, {ARRAY(INTEGER())}));
}

TEST_F(ParquetCompatibilityTest, nullableImpala) {
  initReaderAndCheckSchema(
      "nullable.impala.parquet",
      7ULL,
      ROW({"id",
           "int_array",
           "int_array_Array",
           "int_map",
           "int_Map_Array",
           "nested_struct"},
          {BIGINT(),
           ARRAY(INTEGER()),
           ARRAY(ARRAY(INTEGER())),
           MAP(VARCHAR(), INTEGER()),
           ARRAY(MAP(VARCHAR(), INTEGER())),
           ROW({"A", "b", "C", "g"},
               {INTEGER(),
                ARRAY(INTEGER()),
                ROW({"d"},
                    {ARRAY(ARRAY(ROW({"E", "F"}, {INTEGER(), VARCHAR()})))}),
                MAP(VARCHAR(),
                    ROW({"H"}, {ROW({"i"}, {ARRAY(DOUBLE())})}))})}));
}

TEST_F(ParquetCompatibilityTest, nullsSnappy) {
  initReaderAndCheckSchema(
      "nulls.snappy.parquet",
      8ULL,
      ROW({"b_struct"}, {ROW({"b_c_int"}, {INTEGER()})}));
}

TEST_F(ParquetCompatibilityTest, oldListStructure) {
  initReaderAndCheckSchema(
      "old_list_structure.parquet",
      1ULL,
      ROW({"a"}, {ARRAY(ARRAY(INTEGER()))}));
}

TEST_F(ParquetCompatibilityTest, overflowI16PageCnt) {
  initReaderAndCheckSchema(
      "overflow_i16_page_cnt.parquet", 40000ULL, ROW({"inc"}, {BOOLEAN()}));
}

TEST_F(ParquetCompatibilityTest, plainDictUncompressedChecksum) {
  initReaderAndCheckSchema(
      "plain-dict-uncompressed-checksum.parquet",
      1000ULL,
      ROW({"long_field", "binary_field"}, {BIGINT(), VARBINARY()}));
}

// NOTE(parquet): this file actually contains 6 rows.
TEST_F(ParquetCompatibilityTest, repeatedNoAnnotation) {
  initReaderAndCheckSchema(
      "repeated_no_annotation.parquet",
      0ULL,
      ROW({"id", "phoneNumbers"},
          {INTEGER(),
           ROW({"phone"},
               {ARRAY(ROW({"number", "kind"}, {BIGINT(), VARCHAR()}))})}));
}

TEST_F(ParquetCompatibilityTest, repeatedPrimitiveNoList) {
  initReaderAndCheckSchema(
      "repeated_primitive_no_list.parquet",
      4ULL,
      ROW({"Int32_list", "String_list", "group_of_lists"},
          {ARRAY(INTEGER()),
           ARRAY(VARCHAR()),
           ROW({"Int32_list_in_group", "String_list_in_group"},
               {ARRAY(INTEGER()), ARRAY(VARCHAR())})}));
}

TEST_F(ParquetCompatibilityTest, rleDictSnappyChecksum) {
  initReaderAndCheckSchema(
      "rle-dict-snappy-checksum.parquet",
      1000ULL,
      ROW({"long_field", "binary_field"}, {BIGINT(), VARBINARY()}));
}

TEST_F(ParquetCompatibilityTest, rleDictUncompressedCorruptChecksum) {
  initReaderAndCheckSchema(
      "rle-dict-uncompressed-corrupt-checksum.parquet",
      1000ULL,
      ROW({"long_field", "binary_field"}, {BIGINT(), VARBINARY()}));
}

TEST_F(ParquetCompatibilityTest, rleBooleanEncoding) {
  initReaderAndCheckSchema(
      "rle_boolean_encoding.parquet",
      68ULL,
      ROW({"datatype_boolean"}, {BOOLEAN()}));
}

TEST_F(ParquetCompatibilityTest, singleNan) {
  initReaderAndCheckSchema(
      "single_nan.parquet", 1ULL, ROW({"mycol"}, {DOUBLE()}));
}

TEST_F(ParquetCompatibilityTest, sortColumns) {
  initReaderAndCheckSchema(
      "sort_columns.parquet", 6ULL, ROW({"a", "b"}, {BIGINT(), VARCHAR()}));
}

// NOTE(parquet): Disabled due to unsupported converted types
// thrift::ConvertedType::TIME_MILLIS and thrift::ConvertedType::TIME_MICROS.
TEST_F(ParquetCompatibilityTest, DISABLED_badArrowGh41317) {
  initReaderAndCheckSchema(
      "ARROW-GH-41317.parquet", 5ULL, kBadDataTypes, kBadData);
}

// NOTE(parquet): Disabled due to unsupported converted types
// thrift::ConvertedType::TIME_MILLIS and thrift::ConvertedType::TIME_MICROS.
TEST_F(ParquetCompatibilityTest, DISABLED_badArrowGh41321) {
  initReaderAndCheckSchema(
      "ARROW-GH-41321.parquet", 5ULL, kBadDataTypes, kBadData);
}

TEST_F(ParquetCompatibilityTest, badArrowGh43605) {
  initReaderAndCheckSchema(
      "ARROW-GH-43605.parquet",
      21186ULL,
      ROW({"min_fl"}, {SMALLINT()}),
      kBadData);
}

TEST_F(ParquetCompatibilityTest, badArrowRsGh6229Dictheader) {
  initReaderAndCheckSchema(
      "ARROW-RS-GH-6229-DICTHEADER.parquet",
      25ULL,
      ROW({"nation_key", "name", "region_key", "comment_col"},
          {INTEGER(), VARBINARY(), INTEGER(), VARBINARY()}),
      kBadData);
}

TEST_F(ParquetCompatibilityTest, badArrowRsGh6229Levels) {
  initReaderAndCheckSchema(
      "ARROW-RS-GH-6229-LEVELS.parquet",
      1ULL,
      ROW({"outer"}, {ARRAY(ROW({"c"}, {INTEGER()}))}),
      kBadData);
}

TEST_F(ParquetCompatibilityTest, badParquet1481) {
  EXPECT_THROW(
      initReaderAndCheckSchema(
          "PARQUET-1481.parquet", -1ULL, ROW({}, {}), kBadData),
      VeloxRuntimeError);
}
