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

// Adapted from Apache Arrow.

#include "velox/dwio/parquet/writer/arrow/Metadata.h"

#include <gtest/gtest.h>

#include "arrow/util/key_value_metadata.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/writer/arrow/FileWriter.h"
#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"

namespace facebook::velox::parquet::arrow {
namespace metadata {

using namespace facebook::velox::common::testutil;

// Helper function for generating table metadata.
std::unique_ptr<FileMetaData> generateTableMetaData(
    const SchemaDescriptor& schema,
    const std::shared_ptr<WriterProperties>& props,
    const int64_t& nrows,
    EncodedStatistics statsInt,
    EncodedStatistics statsFloat) {
  auto fBuilder = FileMetaDataBuilder::make(&schema, props);
  auto rg1Builder = fBuilder->appendRowGroup();
  // Write the metadata.
  // Rowgroup1 metadata.
  auto col1Builder = rg1Builder->nextColumnChunk();
  auto col2Builder = rg1Builder->nextColumnChunk();
  // Column metadata.
  std::map<Encoding::type, int32_t> dictEncodingStats(
      {{Encoding::kRleDictionary, 1}});
  std::map<Encoding::type, int32_t> dataEncodingStats(
      {{Encoding::kPlain, 1}, {Encoding::kRle, 1}});
  statsInt.setIsSigned(true);
  col1Builder->setStatistics(statsInt);
  statsFloat.setIsSigned(true);
  col2Builder->setStatistics(statsFloat);
  col1Builder->finish(
      nrows / 2,
      4,
      0,
      10,
      512,
      600,
      true,
      false,
      dictEncodingStats,
      dataEncodingStats);
  col2Builder->finish(
      nrows / 2,
      24,
      0,
      30,
      512,
      600,
      true,
      false,
      dictEncodingStats,
      dataEncodingStats);

  rg1Builder->setNumRows(nrows / 2);
  rg1Builder->finish(1024);

  // Rowgroup2 metadata.
  auto rg2Builder = fBuilder->appendRowGroup();
  col1Builder = rg2Builder->nextColumnChunk();
  col2Builder = rg2Builder->nextColumnChunk();
  // Column metadata.
  col1Builder->setStatistics(statsInt);
  col2Builder->setStatistics(statsFloat);
  col1Builder->finish(
      nrows / 2,
      0,
      0,
      10,
      512,
      600,
      false,
      false,
      dictEncodingStats,
      dataEncodingStats);
  col2Builder->finish(
      nrows / 2,
      16,
      0,
      26,
      512,
      600,
      true,
      false,
      dictEncodingStats,
      dataEncodingStats);

  rg2Builder->setNumRows(nrows / 2);
  rg2Builder->finish(1024);

  // Return the metadata accessor.
  return fBuilder->finish();
}

void assertEncodings(
    const ColumnChunkMetaData& data,
    const std::set<Encoding::type>& expected) {
  std::set<Encoding::type> encodings(
      data.encodings().begin(), data.encodings().end());
  ASSERT_EQ(encodings, expected);
}

TEST(Metadata, TestBuildAccess) {
  schema::NodeVector fields;
  schema::NodePtr root;
  SchemaDescriptor schema;

  WriterProperties::Builder propBuilder;

  std::shared_ptr<WriterProperties> props =
      propBuilder.version(ParquetVersion::PARQUET_2_6)->build();

  fields.push_back(schema::int32("int_col", Repetition::kRequired));
  fields.push_back(schema::floatType("float_col", Repetition::kRequired));
  root = schema::GroupNode::make("schema", Repetition::kRepeated, fields);
  schema.init(root);

  int64_t nrows = 1000;
  int32_t intMin = 100, intMax = 200;
  EncodedStatistics statsInt;
  statsInt.setNullCount(0)
      .setDistinctCount(nrows)
      .setMin(std::string(reinterpret_cast<const char*>(&intMin), 4))
      .setMax(std::string(reinterpret_cast<const char*>(&intMax), 4));
  EncodedStatistics statsFloat;
  float floatMin = 100.100f, floatMax = 200.200f;
  statsFloat.setNullCount(0)
      .setDistinctCount(nrows)
      .setMin(std::string(reinterpret_cast<const char*>(&floatMin), 4))
      .setMax(std::string(reinterpret_cast<const char*>(&floatMax), 4));

  // Generate the metadata.
  auto fAccessor =
      generateTableMetaData(schema, props, nrows, statsInt, statsFloat);

  std::string fAccessorSerializedMetadata = fAccessor->serializeToString();
  uint32_t expectedLen =
      static_cast<uint32_t>(fAccessorSerializedMetadata.length());

  // Decoded_len is an in-out parameter.
  uint32_t decodedLen = expectedLen;
  auto fAccessorCopy =
      FileMetaData::make(fAccessorSerializedMetadata.data(), &decodedLen);

  // Check that all of the serialized data is consumed.
  ASSERT_EQ(expectedLen, decodedLen);

  // Run this block twice, one for f_accessor, one for f_accessor_copy.
  // To make sure SerializedMetadata was deserialized correctly.
  std::vector<FileMetaData*> fAccessors = {
      fAccessor.get(), fAccessorCopy.get()};
  for (int loopIndex = 0; loopIndex < 2; loopIndex++) {
    // File metadata.
    ASSERT_EQ(nrows, fAccessors[loopIndex]->numRows());
    ASSERT_LE(0, static_cast<int>(fAccessors[loopIndex]->size()));
    ASSERT_EQ(2, fAccessors[loopIndex]->numRowGroups());
    ASSERT_EQ(ParquetVersion::PARQUET_2_6, fAccessors[loopIndex]->version());
    ASSERT_TRUE(
        fAccessors[loopIndex]->createdBy().find(DEFAULT_CREATED_BY) !=
        std::string::npos);
    ASSERT_EQ(3, fAccessors[loopIndex]->numSchemaElements());

    // Row group1 metadata.
    auto rg1Accessor = fAccessors[loopIndex]->rowGroup(0);
    ASSERT_EQ(2, rg1Accessor->numColumns());
    ASSERT_EQ(nrows / 2, rg1Accessor->numRows());
    ASSERT_EQ(1024, rg1Accessor->totalByteSize());
    ASSERT_EQ(1024, rg1Accessor->totalCompressedSize());
    EXPECT_EQ(
        rg1Accessor->fileOffset(),
        rg1Accessor->columnChunk(0)->dictionaryPageOffset());

    auto rg1Column1 = rg1Accessor->columnChunk(0);
    auto rg1Column2 = rg1Accessor->columnChunk(1);
    ASSERT_EQ(true, rg1Column1->isStatsSet());
    ASSERT_EQ(true, rg1Column2->isStatsSet());
    ASSERT_EQ(statsFloat.min(), rg1Column2->statistics()->encodeMin());
    ASSERT_EQ(statsFloat.max(), rg1Column2->statistics()->encodeMax());
    ASSERT_EQ(statsInt.min(), rg1Column1->statistics()->encodeMin());
    ASSERT_EQ(statsInt.max(), rg1Column1->statistics()->encodeMax());
    ASSERT_EQ(0, rg1Column1->statistics()->nullCount());
    ASSERT_EQ(0, rg1Column2->statistics()->nullCount());
    ASSERT_EQ(nrows, rg1Column1->statistics()->distinctCount());
    ASSERT_EQ(nrows, rg1Column2->statistics()->distinctCount());
    ASSERT_EQ(DEFAULT_COMPRESSION_TYPE, rg1Column1->compression());
    ASSERT_EQ(DEFAULT_COMPRESSION_TYPE, rg1Column2->compression());
    ASSERT_EQ(nrows / 2, rg1Column1->numValues());
    ASSERT_EQ(nrows / 2, rg1Column2->numValues());
    {
      std::set<Encoding::type> encodings{
          Encoding::kRle, Encoding::kRleDictionary, Encoding::kPlain};
      assertEncodings(*rg1Column1, encodings);
    }
    {
      std::set<Encoding::type> encodings{
          Encoding::kRle, Encoding::kRleDictionary, Encoding::kPlain};
      assertEncodings(*rg1Column2, encodings);
    }
    ASSERT_EQ(512, rg1Column1->totalCompressedSize());
    ASSERT_EQ(512, rg1Column2->totalCompressedSize());
    ASSERT_EQ(600, rg1Column1->totalUncompressedSize());
    ASSERT_EQ(600, rg1Column2->totalUncompressedSize());
    ASSERT_EQ(4, rg1Column1->dictionaryPageOffset());
    ASSERT_EQ(24, rg1Column2->dictionaryPageOffset());
    ASSERT_EQ(10, rg1Column1->dataPageOffset());
    ASSERT_EQ(30, rg1Column2->dataPageOffset());
    ASSERT_EQ(3, rg1Column1->encodingStats().size());
    ASSERT_EQ(3, rg1Column2->encodingStats().size());

    auto rg2Accessor = fAccessors[loopIndex]->rowGroup(1);
    ASSERT_EQ(2, rg2Accessor->numColumns());
    ASSERT_EQ(nrows / 2, rg2Accessor->numRows());
    ASSERT_EQ(1024, rg2Accessor->totalByteSize());
    ASSERT_EQ(1024, rg2Accessor->totalCompressedSize());
    EXPECT_EQ(
        rg2Accessor->fileOffset(),
        rg2Accessor->columnChunk(0)->dataPageOffset());

    auto rg2Column1 = rg2Accessor->columnChunk(0);
    auto rg2Column2 = rg2Accessor->columnChunk(1);
    ASSERT_EQ(true, rg2Column1->isStatsSet());
    ASSERT_EQ(true, rg2Column2->isStatsSet());
    ASSERT_EQ(statsFloat.min(), rg2Column2->statistics()->encodeMin());
    ASSERT_EQ(statsFloat.max(), rg2Column2->statistics()->encodeMax());
    ASSERT_EQ(statsInt.min(), rg1Column1->statistics()->encodeMin());
    ASSERT_EQ(statsInt.max(), rg1Column1->statistics()->encodeMax());
    ASSERT_EQ(0, rg2Column1->statistics()->nullCount());
    ASSERT_EQ(0, rg2Column2->statistics()->nullCount());
    ASSERT_EQ(nrows, rg2Column1->statistics()->distinctCount());
    ASSERT_EQ(nrows, rg2Column2->statistics()->distinctCount());
    ASSERT_EQ(nrows / 2, rg2Column1->numValues());
    ASSERT_EQ(nrows / 2, rg2Column2->numValues());
    ASSERT_EQ(DEFAULT_COMPRESSION_TYPE, rg2Column1->compression());
    ASSERT_EQ(DEFAULT_COMPRESSION_TYPE, rg2Column2->compression());
    {
      std::set<Encoding::type> encodings{Encoding::kRle, Encoding::kPlain};
      assertEncodings(*rg2Column1, encodings);
    }
    {
      std::set<Encoding::type> encodings{
          Encoding::kRle, Encoding::kRleDictionary, Encoding::kPlain};
      assertEncodings(*rg2Column2, encodings);
    }
    ASSERT_EQ(512, rg2Column1->totalCompressedSize());
    ASSERT_EQ(512, rg2Column2->totalCompressedSize());
    ASSERT_EQ(600, rg2Column1->totalUncompressedSize());
    ASSERT_EQ(600, rg2Column2->totalUncompressedSize());
    EXPECT_FALSE(rg2Column1->hasDictionaryPage());
    ASSERT_EQ(0, rg2Column1->dictionaryPageOffset());
    ASSERT_EQ(16, rg2Column2->dictionaryPageOffset());
    ASSERT_EQ(10, rg2Column1->dataPageOffset());
    ASSERT_EQ(26, rg2Column2->dataPageOffset());
    ASSERT_EQ(2, rg2Column1->encodingStats().size());
    ASSERT_EQ(3, rg2Column2->encodingStats().size());

    // Test FileMetaData::set_file_path.
    ASSERT_TRUE(rg2Column1->filePath().empty());
    fAccessors[loopIndex]->setFilePath("/foo/bar/bar.parquet");
    ASSERT_EQ("/foo/bar/bar.parquet", rg2Column1->filePath());
  }

  // Test AppendRowGroups.
  auto fAccessor2 =
      generateTableMetaData(schema, props, nrows, statsInt, statsFloat);
  fAccessor->appendRowGroups(*fAccessor2);
  ASSERT_EQ(4, fAccessor->numRowGroups());
  ASSERT_EQ(nrows * 2, fAccessor->numRows());
  ASSERT_LE(0, static_cast<int>(fAccessor->size()));
  ASSERT_EQ(ParquetVersion::PARQUET_2_6, fAccessor->version());
  ASSERT_TRUE(
      fAccessor->createdBy().find(DEFAULT_CREATED_BY) != std::string::npos);
  ASSERT_EQ(3, fAccessor->numSchemaElements());

  // Test AppendRowGroups from self (ARROW-13654)
  fAccessor->appendRowGroups(*fAccessor);
  ASSERT_EQ(8, fAccessor->numRowGroups());
  ASSERT_EQ(nrows * 4, fAccessor->numRows());
  ASSERT_EQ(3, fAccessor->numSchemaElements());

  // Test Subset.
  auto fAccessor1 = fAccessor->subset({2, 3});
  ASSERT_TRUE(fAccessor1->equals(*fAccessor2));

  fAccessor1 = fAccessor2->subset({0});
  fAccessor1->appendRowGroups(*fAccessor->subset({0}));
  ASSERT_TRUE(fAccessor1->equals(*fAccessor->subset({2, 0})));
}

TEST(Metadata, TestV1Version) {
  // PARQUET-839.
  schema::NodeVector fields;
  schema::NodePtr root;
  SchemaDescriptor schema;

  WriterProperties::Builder propBuilder;

  std::shared_ptr<WriterProperties> props =
      propBuilder.version(ParquetVersion::PARQUET_1_0)->build();

  fields.push_back(schema::int32("int_col", Repetition::kRequired));
  fields.push_back(schema::floatType("float_col", Repetition::kRequired));
  root = schema::GroupNode::make("schema", Repetition::kRepeated, fields);
  schema.init(root);

  auto fBuilder = FileMetaDataBuilder::make(&schema, props);

  // Read the metadata.
  auto fAccessor = fBuilder->finish();

  // File metadata.
  ASSERT_EQ(ParquetVersion::PARQUET_1_0, fAccessor->version());
}

TEST(Metadata, TestKeyValueMetadata) {
  schema::NodeVector fields;
  schema::NodePtr root;
  SchemaDescriptor schema;

  WriterProperties::Builder propBuilder;

  std::shared_ptr<WriterProperties> props =
      propBuilder.version(ParquetVersion::PARQUET_1_0)->build();

  fields.push_back(schema::int32("int_col", Repetition::kRequired));
  fields.push_back(schema::floatType("float_col", Repetition::kRequired));
  root = schema::GroupNode::make("schema", Repetition::kRepeated, fields);
  schema.init(root);

  auto kvmeta = std::make_shared<KeyValueMetadata>();
  kvmeta->Append("test_key", "test_value");

  auto fBuilder = FileMetaDataBuilder::make(&schema, props);

  // Read the metadata.
  auto fAccessor = fBuilder->finish(kvmeta);

  // Key value metadata.
  ASSERT_TRUE(fAccessor->keyValueMetadata());
  EXPECT_TRUE(fAccessor->keyValueMetadata()->Equals(*kvmeta));
}

TEST(Metadata, TestAddKeyValueMetadata) {
  schema::NodeVector fields;
  fields.push_back(schema::int32("int_col", Repetition::kRequired));
  auto schema = std::static_pointer_cast<schema::GroupNode>(
      schema::GroupNode::make("schema", Repetition::kRequired, fields));

  auto kvMeta = std::make_shared<KeyValueMetadata>();
  kvMeta->Append("test_key_1", "test_value_1");
  kvMeta->Append("test_key_2", "test_value_2_");

  auto sink = createOutputStream();
  auto writerProps = WriterProperties::Builder().disableDictionary()->build();
  auto fileWriter = ParquetFileWriter::open(sink, schema, writerProps, kvMeta);

  // Key value metadata that will be added to the file.
  auto kvMetaAdded = std::make_shared<KeyValueMetadata>();
  kvMetaAdded->Append("test_key_2", "test_value_2");
  kvMetaAdded->Append("test_key_3", "test_value_3");

  fileWriter->addKeyValueMetadata(kvMetaAdded);
  fileWriter->close();

  // Throw if appending key value metadata to closed file.
  auto kvMetaIgnored = std::make_shared<KeyValueMetadata>();
  kvMetaIgnored->Append("test_key_4", "test_value_4");
  EXPECT_THROW(
      fileWriter->addKeyValueMetadata(kvMetaIgnored), ParquetException);

  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());

  // Write the buffer to a temp file path.
  auto filePath = TempFilePath::create();
  test::writeToFile(filePath, buffer);
  memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool =
      memory::memoryManager()->addRootPool("MetadataTest");
  std::shared_ptr<facebook::velox::memory::MemoryPool> leafPool =
      rootPool->addLeafChild("MetadataTest");
  dwio::common::ReaderOptions readerOptions{leafPool.get()};
  auto input = std::make_unique<dwio::common::BufferedInput>(
      std::make_shared<LocalReadFile>(filePath->getPath()),
      readerOptions.memoryPool());
  auto reader =
      std::make_unique<ParquetReader>(std::move(input), readerOptions);
  ASSERT_EQ(3, reader->fileMetaData().keyValueMetadataSize());
  // Verify keys that were added before file writer was closed are present.
  for (int i = 1; i <= 3; ++i) {
    auto index = std::to_string(i);
    auto value =
        reader->fileMetaData().keyValueMetadataValue("test_key_" + index);
    EXPECT_EQ("test_value_" + index, value);
  }
  // Verify keys that were added after file writer was closed are not present.
  EXPECT_FALSE(reader->fileMetaData().keyValueMetadataContains("test_key_4"));
  ASSERT_EQ(
      CREATED_BY_VERSION + std::string(" version ") + VELOX_VERSION,
      reader->fileMetaData().createdBy());
}

// TODO: disabled as they require Arrow parquet data dir.
/*
TEST(Metadata, TestHasBloomFilter) {
  std::string dir_string(test::get_data_dir());
  std::string path = dir_string + "/data_index_bloom_encoding_stats.parquet";
  auto reader = ParquetFileReader::OpenFile(path, false);
  auto file_metadata = reader->metadata();
  ASSERT_EQ(1, file_metadata->num_row_groups());
  auto row_group_metadata = file_metadata->RowGroup(0);
  ASSERT_EQ(1, row_group_metadata->num_columns());
  auto col_chunk_metadata = row_group_metadata->ColumnChunk(0);
  auto bloom_filter_offset = col_chunk_metadata->bloom_filter_offset();
  ASSERT_TRUE(bloom_filter_offset.has_value());
  ASSERT_EQ(192, bloom_filter_offset);
}

TEST(Metadata, TestReadPageIndex) {
  std::string dir_string(test::get_data_dir());
  std::string path = dir_string + "/alltypes_tiny_pages.parquet";
  auto reader = ParquetFileReader::OpenFile(path, false);
  auto file_metadata = reader->metadata();
  ASSERT_EQ(1, file_metadata->num_row_groups());
  auto row_group_metadata = file_metadata->RowGroup(0);
  ASSERT_EQ(13, row_group_metadata->num_columns());
  std::vector<int64_t> ci_offsets = {323583, 327502, 328009, 331928, 335847,
                                     339766, 350345, 354264, 364843, 384342,
                                     -1,     386473, 390392};
  std::vector<int32_t> ci_lengths = {3919,  507,   3919, 3919, 3919, 10579,
3919, 10579, 19499, 2131, -1,   3919, 3919}; std::vector<int64_t> oi_offsets =
{394311, 397814, 398637, 401888, 405139, 408390, 413670, 416921, 422201, 431936,
                                     435457, 446002, 449253};
  std::vector<int32_t> oi_lengths = {3503, 823,  3251, 3251,  3251, 5280, 3251,
                                     5280, 9735, 3521, 10545, 3251, 3251};
  for (int i = 0; i < row_group_metadata->num_columns(); ++i) {
    auto col_chunk_metadata = row_group_metadata->ColumnChunk(i);
    auto ci_location = col_chunk_metadata->GetColumnIndexLocation();
    if (i == 10) {
      // Column_id 10 does not have column index.
      ASSERT_FALSE(ci_location.has_value());
    } else {
      ASSERT_TRUE(ci_location.has_value());
    }
    if (ci_location.has_value()) {
      ASSERT_EQ(ci_offsets.at(i), ci_location->offset);
      ASSERT_EQ(ci_lengths.at(i), ci_location->length);
    }
    auto oi_location = col_chunk_metadata->GetOffsetIndexLocation();
    ASSERT_TRUE(oi_location.has_value());
    ASSERT_EQ(oi_offsets.at(i), oi_location->offset);
    ASSERT_EQ(oi_lengths.at(i), oi_location->length);
    ASSERT_FALSE(col_chunk_metadata->bloom_filter_offset().has_value());
  }
}
*/

TEST(Metadata, TestSortingColumns) {
  schema::NodeVector fields;
  fields.push_back(schema::int32("sort_col", Repetition::kRequired));
  fields.push_back(schema::int32("int_col", Repetition::kRequired));

  auto schema = std::static_pointer_cast<schema::GroupNode>(
      schema::GroupNode::make("schema", Repetition::kRequired, fields));

  std::vector<SortingColumn> sortingColumns;
  {
    SortingColumn sortingColumn;
    sortingColumn.columnIdx = 0;
    sortingColumn.descending = false;
    sortingColumn.nullsFirst = false;
    sortingColumns.push_back(sortingColumn);
  }

  auto createdBy = CREATED_BY_VERSION + std::string(" version 1.0");
  auto sink = createOutputStream();
  auto writerProps = WriterProperties::Builder()
                         .disableDictionary()
                         ->setSortingColumns(sortingColumns)
                         ->createdBy(createdBy)
                         ->build();

  EXPECT_EQ(sortingColumns, writerProps->sortingColumns());

  auto fileWriter = ParquetFileWriter::open(sink, schema, writerProps);

  auto rowGroupWriter = fileWriter->appendBufferedRowGroup();
  rowGroupWriter->close();
  fileWriter->close();

  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());

  // Write the buffer to a temp file path.
  auto filePath = TempFilePath::create();
  test::writeToFile(filePath, buffer);
  memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool =
      memory::memoryManager()->addRootPool("MetadataTest");
  std::shared_ptr<facebook::velox::memory::MemoryPool> leafPool =
      rootPool->addLeafChild("MetadataTest");
  dwio::common::ReaderOptions readerOptions{leafPool.get()};
  auto input = std::make_unique<dwio::common::BufferedInput>(
      std::make_shared<LocalReadFile>(filePath->getPath()),
      readerOptions.memoryPool());
  auto reader =
      std::make_unique<ParquetReader>(std::move(input), readerOptions);
  ASSERT_EQ(1, reader->fileMetaData().numRowGroups());
  auto rowGroup = reader->fileMetaData().rowGroup(0);
  EXPECT_EQ(sortingColumns[0].columnIdx, rowGroup.sortingColumnIdx(0));
  EXPECT_EQ(sortingColumns[0].descending, rowGroup.sortingColumnDescending(0));
  EXPECT_EQ(sortingColumns[0].nullsFirst, rowGroup.sortingColumnNullsFirst(0));
  ASSERT_EQ(createdBy, reader->fileMetaData().createdBy());
}

TEST(ApplicationVersion, Basics) {
  ApplicationVersion version("parquet-mr version 1.7.9");
  ApplicationVersion version1("parquet-mr version 1.8.0");
  ApplicationVersion version2("parquet-cpp version 1.0.0");
  ApplicationVersion version3("");
  ApplicationVersion version4(
      "parquet-mr version 1.5.0ab-cdh5.5.0+cd (build abcd)");
  ApplicationVersion version5("parquet-mr");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ(1, version.version.major);
  ASSERT_EQ(7, version.version.minor);
  ASSERT_EQ(9, version.version.patch);

  ASSERT_EQ("parquet-cpp", version2.application_);
  ASSERT_EQ(1, version2.version.major);
  ASSERT_EQ(0, version2.version.minor);
  ASSERT_EQ(0, version2.version.patch);

  ASSERT_EQ("parquet-mr", version4.application_);
  ASSERT_EQ("abcd", version4.build_);
  ASSERT_EQ(1, version4.version.major);
  ASSERT_EQ(5, version4.version.minor);
  ASSERT_EQ(0, version4.version.patch);
  ASSERT_EQ("ab", version4.version.unknown);
  ASSERT_EQ("cdh5.5.0", version4.version.preRelease);
  ASSERT_EQ("cd", version4.version.buildInfo);

  ASSERT_EQ("parquet-mr", version5.application_);
  ASSERT_EQ(0, version5.version.major);
  ASSERT_EQ(0, version5.version.minor);
  ASSERT_EQ(0, version5.version.patch);

  ASSERT_EQ(true, version.versionLt(version1));

  EncodedStatistics stats;
  ASSERT_FALSE(
      version1.hasCorrectStatistics(Type::kInt96, stats, SortOrder::kUnknown));
  ASSERT_TRUE(
      version.hasCorrectStatistics(Type::kInt32, stats, SortOrder::kSigned));
  ASSERT_FALSE(version.hasCorrectStatistics(
      Type::kByteArray, stats, SortOrder::kSigned));
  ASSERT_TRUE(version1.hasCorrectStatistics(
      Type::kByteArray, stats, SortOrder::kSigned));
  ASSERT_FALSE(version1.hasCorrectStatistics(
      Type::kByteArray, stats, SortOrder::kUnsigned));
  ASSERT_TRUE(version3.hasCorrectStatistics(
      Type::kFixedLenByteArray, stats, SortOrder::kSigned));

  // Check that the old stats are correct if min and max are the same.
  // Regardless of sort order.
  EncodedStatistics statsStr;
  statsStr.setMin("a").setMax("b");
  ASSERT_FALSE(version1.hasCorrectStatistics(
      Type::kByteArray, statsStr, SortOrder::kUnsigned));
  statsStr.setMax("a");
  ASSERT_TRUE(version1.hasCorrectStatistics(
      Type::kByteArray, statsStr, SortOrder::kUnsigned));

  // Check that the same holds true for ints.
  int32_t intMin = 100, intMax = 200;
  EncodedStatistics statsInt;
  statsInt.setMin(std::string(reinterpret_cast<const char*>(&intMin), 4))
      .setMax(std::string(reinterpret_cast<const char*>(&intMax), 4));
  ASSERT_FALSE(version1.hasCorrectStatistics(
      Type::kByteArray, statsInt, SortOrder::kUnsigned));
  statsInt.setMax(std::string(reinterpret_cast<const char*>(&intMin), 4));
  ASSERT_TRUE(version1.hasCorrectStatistics(
      Type::kByteArray, statsInt, SortOrder::kUnsigned));
}

TEST(ApplicationVersion, empty) {
  ApplicationVersion version("");

  ASSERT_EQ("", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(0, version.version.major);
  ASSERT_EQ(0, version.version.minor);
  ASSERT_EQ(0, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("", version.version.buildInfo);
}

TEST(ApplicationVersion, NoVersion) {
  ApplicationVersion version("parquet-mr (build abcd)");

  ASSERT_EQ("parquet-mr (build abcd)", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(0, version.version.major);
  ASSERT_EQ(0, version.version.minor);
  ASSERT_EQ(0, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("", version.version.buildInfo);
}

TEST(ApplicationVersion, VersionEmpty) {
  ApplicationVersion version("parquet-mr version ");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(0, version.version.major);
  ASSERT_EQ(0, version.version.minor);
  ASSERT_EQ(0, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("", version.version.buildInfo);
}

TEST(ApplicationVersion, VersionNoMajor) {
  ApplicationVersion version("parquet-mr version .");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(0, version.version.major);
  ASSERT_EQ(0, version.version.minor);
  ASSERT_EQ(0, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("", version.version.buildInfo);
}

TEST(ApplicationVersion, VersionInvalidMajor) {
  ApplicationVersion version("parquet-mr version x1");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(0, version.version.major);
  ASSERT_EQ(0, version.version.minor);
  ASSERT_EQ(0, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("", version.version.buildInfo);
}

TEST(ApplicationVersion, VersionMajorOnly) {
  ApplicationVersion version("parquet-mr version 1");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(1, version.version.major);
  ASSERT_EQ(0, version.version.minor);
  ASSERT_EQ(0, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("", version.version.buildInfo);
}

TEST(ApplicationVersion, VersionNoMinor) {
  ApplicationVersion version("parquet-mr version 1.");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(1, version.version.major);
  ASSERT_EQ(0, version.version.minor);
  ASSERT_EQ(0, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("", version.version.buildInfo);
}

TEST(ApplicationVersion, VersionMajorMinorOnly) {
  ApplicationVersion version("parquet-mr version 1.7");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(1, version.version.major);
  ASSERT_EQ(7, version.version.minor);
  ASSERT_EQ(0, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("", version.version.buildInfo);
}

TEST(ApplicationVersion, VersionInvalidMinor) {
  ApplicationVersion version("parquet-mr version 1.x7");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(1, version.version.major);
  ASSERT_EQ(0, version.version.minor);
  ASSERT_EQ(0, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("", version.version.buildInfo);
}

TEST(ApplicationVersion, VersionNoPatch) {
  ApplicationVersion version("parquet-mr version 1.7.");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(1, version.version.major);
  ASSERT_EQ(7, version.version.minor);
  ASSERT_EQ(0, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("", version.version.buildInfo);
}

TEST(ApplicationVersion, VersionInvalidPatch) {
  ApplicationVersion version("parquet-mr version 1.7.x9");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(1, version.version.major);
  ASSERT_EQ(7, version.version.minor);
  ASSERT_EQ(0, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("", version.version.buildInfo);
}

TEST(ApplicationVersion, VersionNoUnknown) {
  ApplicationVersion version("parquet-mr version 1.7.9-cdh5.5.0+cd");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(1, version.version.major);
  ASSERT_EQ(7, version.version.minor);
  ASSERT_EQ(9, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("cdh5.5.0", version.version.preRelease);
  ASSERT_EQ("cd", version.version.buildInfo);
}

TEST(ApplicationVersion, VersionNoPreRelease) {
  ApplicationVersion version("parquet-mr version 1.7.9ab+cd");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(1, version.version.major);
  ASSERT_EQ(7, version.version.minor);
  ASSERT_EQ(9, version.version.patch);
  ASSERT_EQ("ab", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("cd", version.version.buildInfo);
}

TEST(ApplicationVersion, VersionNoUnknownNoPreRelease) {
  ApplicationVersion version("parquet-mr version 1.7.9+cd");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(1, version.version.major);
  ASSERT_EQ(7, version.version.minor);
  ASSERT_EQ(9, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("cd", version.version.buildInfo);
}

TEST(ApplicationVersion, VersionNoUnknownBuildInfoPreRelease) {
  ApplicationVersion version("parquet-mr version 1.7.9+cd-cdh5.5.0");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("", version.build_);
  ASSERT_EQ(1, version.version.major);
  ASSERT_EQ(7, version.version.minor);
  ASSERT_EQ(9, version.version.patch);
  ASSERT_EQ("", version.version.unknown);
  ASSERT_EQ("", version.version.preRelease);
  ASSERT_EQ("cd-cdh5.5.0", version.version.buildInfo);
}

TEST(ApplicationVersion, FullWithSpaces) {
  ApplicationVersion version(
      " parquet-mr \t version \v 1.5.3ab-cdh5.5.0+cd \r (build \n abcd \f) ");

  ASSERT_EQ("parquet-mr", version.application_);
  ASSERT_EQ("abcd", version.build_);
  ASSERT_EQ(1, version.version.major);
  ASSERT_EQ(5, version.version.minor);
  ASSERT_EQ(3, version.version.patch);
  ASSERT_EQ("ab", version.version.unknown);
  ASSERT_EQ("cdh5.5.0", version.version.preRelease);
  ASSERT_EQ("cd", version.version.buildInfo);
}

} // namespace metadata
} // namespace facebook::velox::parquet::arrow
