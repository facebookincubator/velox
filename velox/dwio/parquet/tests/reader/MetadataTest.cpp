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

#include "velox/dwio/parquet/reader/Metadata.h"

#include <gtest/gtest.h>
#include "velox/dwio/parquet/thrift/ParquetThrift.h"

namespace facebook::velox::parquet {

namespace {

// Long enough to defeat the small-string optimization on libstdc++,
// libc++, and MSVC, so heap accounting is exercised.
const std::string kLongString(64, 'x');

size_t estimateColumnSize(const thrift::ColumnChunk& column) {
  return ColumnChunkMetaDataPtr(&column).estimateColumnMetadataSize();
}

size_t estimateFileSize(const thrift::FileMetaData& metadata) {
  return FileMetaDataPtr(&metadata).estimateFileMetadataSize();
}

} // namespace

// An empty ColumnChunk has no heap-backed payloads; the estimate must
// equal sizeof(thrift::ColumnChunk) and never double-count inline
// sub-structs like ColumnMetaData or ColumnCryptoMetaData.
TEST(MetadataTest, emptyColumnChunkIsSizeofOnly) {
  thrift::ColumnChunk column;
  EXPECT_EQ(estimateColumnSize(column), sizeof(thrift::ColumnChunk));
}

// file_path is a heap-allocated string on ColumnChunk; setting it must
// add exactly the string capacity.
TEST(MetadataTest, columnChunkAccountsFilePath) {
  thrift::ColumnChunk column;
  column.file_path() = kLongString;
  EXPECT_EQ(
      estimateColumnSize(column),
      sizeof(thrift::ColumnChunk) + column.file_path()->capacity());
}

// SSO strings live inline in the std::string object, which is already
// covered by sizeof(ColumnChunk). The estimator must not add their
// capacity on top.
TEST(MetadataTest, columnChunkSkipsSsoFilePath) {
  thrift::ColumnChunk column;
  // Short string fits in SSO buffer on all supported standard libraries.
  column.file_path() = "short";
  EXPECT_EQ(estimateColumnSize(column), sizeof(thrift::ColumnChunk));
}

// encrypted_column_metadata is a heap-allocated string on ColumnChunk.
TEST(MetadataTest, columnChunkAccountsEncryptedColumnMetadata) {
  thrift::ColumnChunk column;
  column.encrypted_column_metadata() = kLongString;
  EXPECT_EQ(
      estimateColumnSize(column),
      sizeof(thrift::ColumnChunk) +
          column.encrypted_column_metadata()->capacity());
}

// meta_data.encodings is a heap-allocated vector; each element is a
// fixed-size enum.
TEST(MetadataTest, columnChunkAccountsEncodings) {
  thrift::ColumnChunk column;
  column.meta_data().ensure().encodings()->push_back(thrift::Encoding::PLAIN);
  column.meta_data()->encodings()->push_back(thrift::Encoding::RLE);
  EXPECT_EQ(
      estimateColumnSize(column),
      sizeof(thrift::ColumnChunk) + 2 * sizeof(thrift::Encoding));
}

// meta_data.path_in_schema is a heap-allocated vector of strings; the
// vector buffer plus the heap part of each string must be counted.
TEST(MetadataTest, columnChunkAccountsPathInSchema) {
  thrift::ColumnChunk column;
  column.meta_data().ensure().path_in_schema()->push_back(kLongString);
  column.meta_data()->path_in_schema()->push_back(kLongString);
  size_t expected = sizeof(thrift::ColumnChunk) +
      column.meta_data()->path_in_schema()->size() * sizeof(std::string);
  for (const auto& path : *column.meta_data()->path_in_schema()) {
    expected += path.capacity();
  }
  EXPECT_EQ(estimateColumnSize(column), expected);
}

// meta_data.key_value_metadata: each KeyValue holds two heap strings.
TEST(MetadataTest, columnChunkAccountsKeyValueMetadata) {
  thrift::ColumnChunk column;
  thrift::KeyValue kv;
  kv.key() = kLongString;
  kv.value() = kLongString;
  column.meta_data().ensure().key_value_metadata().ensure().push_back(kv);
  const auto& keyValueMetadata = *column.meta_data()->key_value_metadata();
  size_t expected = sizeof(thrift::ColumnChunk) +
      keyValueMetadata.size() * sizeof(thrift::KeyValue) +
      keyValueMetadata[0].key()->capacity() +
      keyValueMetadata[0].value()->capacity();
  EXPECT_EQ(estimateColumnSize(column), expected);
}

// Statistics is an optional member of ColumnMetaData; only its heap-backed
// min/max strings should be added. The struct itself must not be
// double-counted on top of sizeof(ColumnChunk).
TEST(MetadataTest, columnChunkAccountsStatisticsHeapOnly) {
  thrift::ColumnChunk column;
  auto& stats = column.meta_data().ensure().statistics().ensure();
  stats.min() = kLongString;
  stats.max() = kLongString;
  stats.min_value() = kLongString;
  stats.max_value() = kLongString;
  size_t expected = sizeof(thrift::ColumnChunk) + stats.min()->capacity() +
      stats.max()->capacity() + stats.min_value()->capacity() +
      stats.max_value()->capacity();
  EXPECT_EQ(estimateColumnSize(column), expected);
}

// Statistics with no isset must contribute nothing beyond sizeof(CC).
TEST(MetadataTest, columnChunkSkipsStatisticsWhenNotSet) {
  thrift::ColumnChunk column;
  EXPECT_EQ(estimateColumnSize(column), sizeof(thrift::ColumnChunk));
}

// An empty FileMetaData estimate must equal sizeof(thrift::FileMetaData)
// and never double-count inline sub-structs.
TEST(MetadataTest, emptyFileMetaDataIsSizeofOnly) {
  thrift::FileMetaData metadata;
  EXPECT_EQ(estimateFileSize(metadata), sizeof(thrift::FileMetaData));
}

// Schema is a heap vector of SchemaElement; each element's name string
// must be counted on top of the inline element bytes.
TEST(MetadataTest, fileMetaDataAccountsSchema) {
  thrift::FileMetaData metadata;
  thrift::SchemaElement element;
  element.name() = kLongString;
  metadata.schema()->push_back(element);
  metadata.schema()->push_back(element);
  size_t expected = sizeof(thrift::FileMetaData) +
      metadata.schema()->size() * sizeof(thrift::SchemaElement) +
      (*metadata.schema())[0].name()->capacity() +
      (*metadata.schema())[1].name()->capacity();
  EXPECT_EQ(estimateFileSize(metadata), expected);
}

// Each RowGroup's columns vector and its element heap payloads must
// flow into the file-level estimate.
TEST(MetadataTest, fileMetaDataAccountsRowGroupColumns) {
  thrift::FileMetaData metadata;
  thrift::RowGroup rowGroup;
  thrift::ColumnChunk column;
  column.file_path() = kLongString;
  rowGroup.columns()->push_back(column);
  rowGroup.columns()->push_back(column);
  metadata.row_groups()->push_back(rowGroup);
  size_t expected = sizeof(thrift::FileMetaData) +
      metadata.row_groups()->size() * sizeof(thrift::RowGroup);
  for (const auto& group : *metadata.row_groups()) {
    for (const auto& col : *group.columns()) {
      expected += estimateColumnSize(col);
    }
  }
  EXPECT_EQ(estimateFileSize(metadata), expected);
}

// created_by is a heap string on FileMetaData.
TEST(MetadataTest, fileMetaDataAccountsCreatedBy) {
  thrift::FileMetaData metadata;
  metadata.created_by() = kLongString;
  EXPECT_EQ(
      estimateFileSize(metadata),
      sizeof(thrift::FileMetaData) + metadata.created_by()->capacity());
}

// key_value_metadata uses the same per-element accounting as on
// ColumnChunk.
TEST(MetadataTest, fileMetaDataAccountsKeyValueMetadata) {
  thrift::FileMetaData metadata;
  thrift::KeyValue kv;
  kv.key() = kLongString;
  kv.value() = kLongString;
  metadata.key_value_metadata().ensure().push_back(kv);
  const auto& keyValueMetadata = *metadata.key_value_metadata();
  size_t expected = sizeof(thrift::FileMetaData) +
      keyValueMetadata.size() * sizeof(thrift::KeyValue) +
      keyValueMetadata[0].key()->capacity() +
      keyValueMetadata[0].value()->capacity();
  EXPECT_EQ(estimateFileSize(metadata), expected);
}

// column_orders is a fixed-size element vector; no string payloads.
TEST(MetadataTest, fileMetaDataAccountsColumnOrders) {
  thrift::FileMetaData metadata;
  metadata.column_orders().ensure().emplace_back();
  metadata.column_orders()->emplace_back();
  EXPECT_EQ(
      estimateFileSize(metadata),
      sizeof(thrift::FileMetaData) +
          metadata.column_orders()->size() * sizeof(thrift::ColumnOrder));
}

// footer_signing_key_metadata is a heap string on FileMetaData.
TEST(MetadataTest, fileMetaDataAccountsFooterSigningKey) {
  thrift::FileMetaData metadata;
  metadata.footer_signing_key_metadata() = kLongString;
  EXPECT_EQ(
      estimateFileSize(metadata),
      sizeof(thrift::FileMetaData) +
          metadata.footer_signing_key_metadata()->capacity());
}

} // namespace facebook::velox::parquet
