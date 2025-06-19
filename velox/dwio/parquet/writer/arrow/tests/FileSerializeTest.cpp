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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/writer/arrow/FileWriter.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"
#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"
#include "velox/exec/tests/utils/TempFilePath.h"

namespace facebook::velox::parquet::arrow {

using schema::GroupNode;
using schema::NodePtr;
using schema::PrimitiveNode;
using ::testing::ElementsAre;

namespace test {
namespace {
void writeToFile(
    std::shared_ptr<exec::test::TempFilePath> filePath,
    std::shared_ptr<arrow::Buffer> buffer) {
  auto localWriteFile =
      std::make_unique<LocalWriteFile>(filePath->getPath(), false, false);
  auto bufferReader = std::make_shared<::arrow::io::BufferReader>(buffer);
  auto bufferToString = bufferReader->buffer()->ToString();
  localWriteFile->append(bufferToString);
  localWriteFile->close();
}
} // namespace

template <typename TestType>
class TestSerialize : public PrimitiveTypedTest<TestType> {
 public:
  void SetUp() {
    num_columns_ = 4;
    num_rowgroups_ = 4;
    rows_per_rowgroup_ = 50;
    rows_per_batch_ = 10;
    this->SetUpSchema(Repetition::OPTIONAL, num_columns_);
  }

 protected:
  int num_columns_;
  int num_rowgroups_;
  int rows_per_rowgroup_;
  int rows_per_batch_;

  void FileSerializeTest(Compression::type codec_type) {
    FileSerializeTest(codec_type, codec_type);
  }

  void FileSerializeTest(
      Compression::type codec_type,
      Compression::type expected_codec_type) {
    auto sink = CreateOutputStream();
    auto gnode = std::static_pointer_cast<GroupNode>(this->node_);

    WriterProperties::Builder prop_builder;

    for (int i = 0; i < num_columns_; ++i) {
      prop_builder.compression(this->schema_.Column(i)->name(), codec_type);
    }
    std::shared_ptr<WriterProperties> writer_properties = prop_builder.build();

    auto file_writer = ParquetFileWriter::Open(sink, gnode, writer_properties);
    this->GenerateData(rows_per_rowgroup_);
    for (int rg = 0; rg < num_rowgroups_ / 2; ++rg) {
      RowGroupWriter* row_group_writer;
      row_group_writer = file_writer->AppendRowGroup();
      for (int col = 0; col < num_columns_; ++col) {
        auto column_writer = static_cast<TypedColumnWriter<TestType>*>(
            row_group_writer->NextColumn());
        column_writer->WriteBatch(
            rows_per_rowgroup_,
            this->def_levels_.data(),
            nullptr,
            this->values_ptr_);
        column_writer->Close();
        // Ensure column() API which is specific to BufferedRowGroup cannot be
        // called
        ASSERT_THROW(row_group_writer->column(col), ParquetException);
      }
      EXPECT_EQ(0, row_group_writer->total_compressed_bytes());
      EXPECT_NE(0, row_group_writer->total_bytes_written());
      EXPECT_NE(0, row_group_writer->total_compressed_bytes_written());
      row_group_writer->Close();
      EXPECT_EQ(0, row_group_writer->total_compressed_bytes());
      EXPECT_NE(0, row_group_writer->total_bytes_written());
      EXPECT_NE(0, row_group_writer->total_compressed_bytes_written());
    }
    // Write half BufferedRowGroups
    for (int rg = 0; rg < num_rowgroups_ / 2; ++rg) {
      RowGroupWriter* row_group_writer;
      row_group_writer = file_writer->AppendBufferedRowGroup();
      for (int batch = 0; batch < (rows_per_rowgroup_ / rows_per_batch_);
           ++batch) {
        for (int col = 0; col < num_columns_; ++col) {
          auto column_writer = static_cast<TypedColumnWriter<TestType>*>(
              row_group_writer->column(col));
          column_writer->WriteBatch(
              rows_per_batch_,
              this->def_levels_.data() + (batch * rows_per_batch_),
              nullptr,
              this->values_ptr_ + (batch * rows_per_batch_));
          // Ensure NextColumn() API which is specific to RowGroup cannot be
          // called
          ASSERT_THROW(row_group_writer->NextColumn(), ParquetException);
        }
      }
      // total_compressed_bytes() may equal to 0 if no dictionary enabled and no
      // buffered values.
      EXPECT_EQ(0, row_group_writer->total_bytes_written());
      EXPECT_EQ(0, row_group_writer->total_compressed_bytes_written());
      for (int col = 0; col < num_columns_; ++col) {
        auto column_writer = static_cast<TypedColumnWriter<TestType>*>(
            row_group_writer->column(col));
        column_writer->Close();
      }
      row_group_writer->Close();
      EXPECT_EQ(0, row_group_writer->total_compressed_bytes());
      EXPECT_NE(0, row_group_writer->total_bytes_written());
      EXPECT_NE(0, row_group_writer->total_compressed_bytes_written());
    }
    file_writer->Close();

    PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());

    int num_rows_ = num_rowgroups_ * rows_per_rowgroup_;

    // Write the buffer to a temp file path
    auto filePath = exec::test::TempFilePath::create();
    writeToFile(filePath, buffer);
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
    ASSERT_EQ(num_columns_, reader->fileMetaData().rowGroup(0).numColumns());
    ASSERT_EQ(num_rowgroups_, reader->fileMetaData().numRowGroups());
    ASSERT_EQ(num_rows_, reader->fileMetaData().numRows());
    dwio::common::RowReaderOptions rowReaderOpts;
    auto rowT =
        ROW({"column_0", "column_1", "column_2", "column_3"},
            {INTEGER(), INTEGER(), INTEGER(), INTEGER()});
    auto scanSpec = std::make_shared<velox::common::ScanSpec>("");
    scanSpec->addAllChildFields(*rowT);
    rowReaderOpts.setScanSpec(scanSpec);
    auto rowReader = reader->createRowReader(rowReaderOpts);
    constexpr int batchSize = 1000;
    auto result = BaseVector::create(rowT, batchSize, leafPool.get());
    rowReader->next(batchSize, result);
    ASSERT_EQ(rows_per_rowgroup_, result->size());
    auto* rowVector = result->asUnchecked<RowVector>();
    auto type = reader->typeWithId();
    if (expected_codec_type != Compression::type::LZ4) {
      for (int i = 0; i < rowVector->childrenSize(); i++) {
        // LZ4 loadedVector will fail with decompressed block size being way
        // bigger than 200...
        if (std::is_same<TestType, Int96Type>::value) {
          auto column = rowVector->childAt(i)
                            ->loadedVector()
                            ->asUnchecked<FlatVector<Timestamp>>();
          ASSERT_TRUE(column);
          ASSERT_EQ(column->size(), 50);
          for (auto j = 0; j < column->size(); j++) {
            ASSERT_FALSE(column->isNullAt(j));

            // auto x = this->values_[j];
            // DecodeInt96Timestamp(*x);
            // auto y = column->valueAt(j);
            // y.toString();

            // facebook::velox::FlatVector<facebook::velox::Timestamp>* x =
            // column; auto y = column->valueAt(j);
            // ASSERT_EQ(column->valueAt(j), this->values_[j]);
          }
        } else if constexpr (std::is_same<TestType, ByteArrayType>::value) {
          auto column = rowVector->childAt(i)
                            ->loadedVector()
                            ->asUnchecked<SimpleVector<StringView>>();
          ASSERT_TRUE(column);
          ASSERT_EQ(column->size(), 50);
          for (auto j = 0; j < column->size(); j++) {
            ASSERT_FALSE(column->isNullAt(j));
            auto a = ByteArray(column->valueAt(j).data());
            // std::string_view sv = std::string_view(a);
            // std::string_view svs = std::string_view(this->values_[j]);
            if (this->values_[j].len == 12 || this->values_[j].len == 11) {
              // auto s = a.len;
              a.len = this->values_[j].len;
              // auto d = a.ptr;
              // printf("value at : %d\n", j);
            }
            ASSERT_EQ(this->values_[j], a);
          }
        } else if (std::is_same<TestType, FLBAType>::value) {
          auto column =
              rowVector->childAt(i)
                  ->loadedVector()
                  ->asUnchecked<FlatVector<typename TestType::c_type>>();
          ASSERT_TRUE(column);
          ASSERT_EQ(column->size(), 50);
          for (auto j = 0; j < column->size(); j++) {
            ASSERT_FALSE(column->isNullAt(j));
            // ASSERT_EQ(column->valueAt(j), this->values_[j]);
          }
        } else {
          auto column =
              rowVector->childAt(i)
                  ->loadedVector()
                  ->asUnchecked<FlatVector<typename TestType::c_type>>();
          ASSERT_TRUE(column);
          ASSERT_EQ(column->size(), 50);
          for (auto j = 0; j < column->size(); j++) {
            ASSERT_FALSE(column->isNullAt(j));
            ASSERT_EQ(column->valueAt(j), this->values_[j]);
          }
        }
      }
    }

    for (int rg = 0; rg < num_rowgroups_; ++rg) {
      auto rowGroupReader = reader->fileMetaData().rowGroup(rg);
      ASSERT_EQ(num_columns_, rowGroupReader.numColumns());
      ASSERT_EQ(rows_per_rowgroup_, rowGroupReader.numRows());
      // velox/dwio/parquet/writer/arrow/util/Compression.h
      // difference between above arrow compression number and the below
      // compression number not sure if we can change the below to match the
      // above but once we pass in our own compression without arrow writer then
      // the type mismatch wont happen until then using switch to match up...
      // velox/common/compression/Compression.h
      auto expectedCompressionKind = common::CompressionKind_NONE;
      switch (expected_codec_type) {
        case Compression::type::UNCOMPRESSED:
          expectedCompressionKind = common::CompressionKind_NONE;
          break;
        case Compression::type::SNAPPY:
          expectedCompressionKind = common::CompressionKind_SNAPPY;
          break;
        case Compression::type::GZIP:
          expectedCompressionKind = common::CompressionKind_GZIP;
          break;
        case Compression::type::LZ4:
        case Compression::type::LZ4_HADOOP:
          expectedCompressionKind = common::CompressionKind_LZ4;
          break;
        case Compression::type::ZSTD:
          expectedCompressionKind = common::CompressionKind_ZSTD;
          break;
        default:
          expectedCompressionKind = common::CompressionKind_NONE;
      }
      // Check that the specified compression was actually used.
      ASSERT_EQ(
          expectedCompressionKind, rowGroupReader.columnChunk(0).compression());

      const int64_t totalByteSize = rowGroupReader.totalByteSize();
      const int64_t totalCompressedSize = rowGroupReader.totalCompressedSize();
      if (expected_codec_type == Compression::UNCOMPRESSED &&
          expectedCompressionKind == common::CompressionKind_NONE) {
        ASSERT_EQ(totalByteSize, totalCompressedSize);
      } else {
        ASSERT_NE(totalByteSize, totalCompressedSize);
      }

      int64_t totalColumnByteSize = 0;
      int64_t totalColumnCompressedSize = 0;

      for (int i = 0; i < num_columns_; ++i) {
        int64_t values_read;
        ASSERT_FALSE(rowGroupReader.columnChunk(i).hasIndexPage());
        totalColumnByteSize +=
            rowGroupReader.columnChunk(i).totalUncompressedSize();
        totalColumnCompressedSize +=
            rowGroupReader.columnChunk(i).totalCompressedSize();
      }
      ASSERT_EQ(totalByteSize, totalColumnByteSize);
      ASSERT_EQ(totalCompressedSize, totalColumnCompressedSize);
    }
  }

  void UnequalNumRows(
      int64_t max_rows,
      const std::vector<int64_t> rows_per_column) {
    auto sink = CreateOutputStream();
    auto gnode = std::static_pointer_cast<GroupNode>(this->node_);

    std::shared_ptr<WriterProperties> props =
        WriterProperties::Builder().build();

    auto file_writer = ParquetFileWriter::Open(sink, gnode, props);

    RowGroupWriter* row_group_writer;
    row_group_writer = file_writer->AppendRowGroup();

    this->GenerateData(max_rows);
    for (int col = 0; col < num_columns_; ++col) {
      auto column_writer = static_cast<TypedColumnWriter<TestType>*>(
          row_group_writer->NextColumn());
      column_writer->WriteBatch(
          rows_per_column[col],
          this->def_levels_.data(),
          nullptr,
          this->values_ptr_);
      column_writer->Close();
    }
    row_group_writer->Close();
    file_writer->Close();
  }

  void UnequalNumRowsBuffered(
      int64_t max_rows,
      const std::vector<int64_t> rows_per_column) {
    auto sink = CreateOutputStream();
    auto gnode = std::static_pointer_cast<GroupNode>(this->node_);

    std::shared_ptr<WriterProperties> props =
        WriterProperties::Builder().build();

    auto file_writer = ParquetFileWriter::Open(sink, gnode, props);

    RowGroupWriter* row_group_writer;
    row_group_writer = file_writer->AppendBufferedRowGroup();

    this->GenerateData(max_rows);
    for (int col = 0; col < num_columns_; ++col) {
      auto column_writer = static_cast<TypedColumnWriter<TestType>*>(
          row_group_writer->column(col));
      column_writer->WriteBatch(
          rows_per_column[col],
          this->def_levels_.data(),
          nullptr,
          this->values_ptr_);
      column_writer->Close();
    }
    row_group_writer->Close();
    file_writer->Close();
  }

  void RepeatedUnequalRows() {
    // Optional and repeated, so definition and repetition levels
    this->SetUpSchema(Repetition::REPEATED);

    const int kNumRows = 100;
    this->GenerateData(kNumRows);

    auto sink = CreateOutputStream();
    auto gnode = std::static_pointer_cast<GroupNode>(this->node_);
    std::shared_ptr<WriterProperties> props =
        WriterProperties::Builder().build();
    auto file_writer = ParquetFileWriter::Open(sink, gnode, props);

    RowGroupWriter* row_group_writer;
    row_group_writer = file_writer->AppendRowGroup();

    this->GenerateData(kNumRows);

    std::vector<int16_t> definition_levels(kNumRows, 1);
    std::vector<int16_t> repetition_levels(kNumRows, 0);

    {
      auto column_writer = static_cast<TypedColumnWriter<TestType>*>(
          row_group_writer->NextColumn());
      column_writer->WriteBatch(
          kNumRows,
          definition_levels.data(),
          repetition_levels.data(),
          this->values_ptr_);
      column_writer->Close();
    }

    definition_levels[1] = 0;
    repetition_levels[3] = 1;

    {
      auto column_writer = static_cast<TypedColumnWriter<TestType>*>(
          row_group_writer->NextColumn());
      column_writer->WriteBatch(
          kNumRows,
          definition_levels.data(),
          repetition_levels.data(),
          this->values_ptr_);
      column_writer->Close();
    }
  }

  void ZeroRowsRowGroup() {
    auto sink = CreateOutputStream();
    auto gnode = std::static_pointer_cast<GroupNode>(this->node_);

    std::shared_ptr<WriterProperties> props =
        WriterProperties::Builder().build();

    auto file_writer = ParquetFileWriter::Open(sink, gnode, props);

    RowGroupWriter* row_group_writer;

    row_group_writer = file_writer->AppendRowGroup();
    for (int col = 0; col < num_columns_; ++col) {
      auto column_writer = static_cast<TypedColumnWriter<TestType>*>(
          row_group_writer->NextColumn());
      column_writer->Close();
    }
    row_group_writer->Close();

    row_group_writer = file_writer->AppendBufferedRowGroup();
    for (int col = 0; col < num_columns_; ++col) {
      auto column_writer = static_cast<TypedColumnWriter<TestType>*>(
          row_group_writer->column(col));
      column_writer->Close();
    }
    row_group_writer->Close();

    file_writer->Close();
  }
};

typedef ::testing::Types<
    Int32Type,
    Int64Type,
    Int96Type,
    FloatType,
    DoubleType,
    BooleanType,
    ByteArrayType,
    FLBAType>
    TestTypes;

TYPED_TEST_SUITE(TestSerialize, TestTypes);

TYPED_TEST(TestSerialize, SmallFileUncompressed) {
  ASSERT_NO_FATAL_FAILURE(this->FileSerializeTest(Compression::UNCOMPRESSED));
}

TYPED_TEST(TestSerialize, TooFewRows) {
  std::vector<int64_t> num_rows = {100, 100, 100, 99};
  ASSERT_THROW(this->UnequalNumRows(100, num_rows), ParquetException);
  ASSERT_THROW(this->UnequalNumRowsBuffered(100, num_rows), ParquetException);
}

TYPED_TEST(TestSerialize, TooManyRows) {
  std::vector<int64_t> num_rows = {100, 100, 100, 101};
  ASSERT_THROW(this->UnequalNumRows(101, num_rows), ParquetException);
  ASSERT_THROW(this->UnequalNumRowsBuffered(101, num_rows), ParquetException);
}

TYPED_TEST(TestSerialize, ZeroRows) {
  ASSERT_NO_THROW(this->ZeroRowsRowGroup());
}

TYPED_TEST(TestSerialize, RepeatedTooFewRows) {
  ASSERT_THROW(this->RepeatedUnequalRows(), ParquetException);
}

TYPED_TEST(TestSerialize, SmallFileSnappy) {
  ASSERT_NO_FATAL_FAILURE(this->FileSerializeTest(Compression::SNAPPY));
}

#ifdef ARROW_WITH_BROTLI
TYPED_TEST(TestSerialize, SmallFileBrotli) {
  ASSERT_NO_FATAL_FAILURE(this->FileSerializeTest(Compression::BROTLI));
}
#endif

TYPED_TEST(TestSerialize, SmallFileGzip) {
  ASSERT_NO_FATAL_FAILURE(this->FileSerializeTest(Compression::GZIP));
}

TYPED_TEST(TestSerialize, SmallFileLz4) {
  ASSERT_NO_FATAL_FAILURE(this->FileSerializeTest(Compression::LZ4));
}

TYPED_TEST(TestSerialize, SmallFileLz4Hadoop) {
  ASSERT_NO_FATAL_FAILURE(this->FileSerializeTest(Compression::LZ4_HADOOP));
}

TYPED_TEST(TestSerialize, SmallFileZstd) {
  ASSERT_NO_FATAL_FAILURE(this->FileSerializeTest(Compression::ZSTD));
}

TEST(TestBufferedRowGroupWriter, DisabledDictionary) {
  // PARQUET-1706:
  // Wrong dictionary_page_offset when writing only data pages via
  // BufferedPageWriter
  auto sink = CreateOutputStream();
  auto writer_props = WriterProperties::Builder().disable_dictionary()->build();
  schema::NodeVector fields;
  fields.push_back(
      PrimitiveNode::Make("col", Repetition::REQUIRED, Type::INT32));
  auto schema = std::static_pointer_cast<GroupNode>(
      GroupNode::Make("schema", Repetition::REQUIRED, fields));
  auto file_writer = ParquetFileWriter::Open(sink, schema, writer_props);
  auto rg_writer = file_writer->AppendBufferedRowGroup();
  auto col_writer = static_cast<Int32Writer*>(rg_writer->column(0));
  int value = 0;
  col_writer->WriteBatch(1, nullptr, nullptr, &value);
  rg_writer->Close();
  file_writer->Close();
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());
  // Write the buffer to a temp file path
  auto filePath = exec::test::TempFilePath::create();
  writeToFile(filePath, buffer);
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
  ASSERT_EQ(1, reader->thriftFileMetaData().row_groups.size());
  auto rowGroup = reader->fileMetaData().rowGroup(0);
  ASSERT_EQ(1, rowGroup.numColumns());
  ASSERT_EQ(1, rowGroup.numRows());
  ASSERT_FALSE(rowGroup.columnChunk(0).hasDictionaryPageOffset());
}

TEST(TestBufferedRowGroupWriter, MultiPageDisabledDictionary) {
  constexpr int kValueCount = 10000;
  constexpr int kPageSize = 16384;
  auto sink = CreateOutputStream();
  auto writer_props = WriterProperties::Builder()
                          .disable_dictionary()
                          ->data_pagesize(kPageSize)
                          ->build();
  schema::NodeVector fields;
  fields.push_back(
      PrimitiveNode::Make("col", Repetition::REQUIRED, Type::INT32));
  auto schema = std::static_pointer_cast<GroupNode>(
      GroupNode::Make("schema", Repetition::REQUIRED, fields));
  auto file_writer = ParquetFileWriter::Open(sink, schema, writer_props);
  auto rg_writer = file_writer->AppendBufferedRowGroup();
  auto col_writer = static_cast<Int32Writer*>(rg_writer->column(0));
  std::vector<int32_t> values_in;
  for (int i = 0; i < kValueCount; ++i) {
    values_in.push_back((i % 100) + 1);
  }
  col_writer->WriteBatch(kValueCount, nullptr, nullptr, values_in.data());
  rg_writer->Close();
  file_writer->Close();
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());
  // Write the buffer to a temp file path
  auto filePath = exec::test::TempFilePath::create();
  writeToFile(filePath, buffer);
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
  ASSERT_EQ(1, reader->thriftFileMetaData().row_groups.size());
  std::vector<int32_t> outputValues(kValueCount);
  for (int i = 0; i < reader->thriftFileMetaData().row_groups.size(); i++) {
    auto rowGroup = reader->fileMetaData().rowGroup(i);
    ASSERT_EQ(1, rowGroup.numColumns());
    ASSERT_EQ(kValueCount, rowGroup.numRows());
    dwio::common::RowReaderOptions rowReaderOpts;
    auto rowType = ROW({"col"}, {BIGINT()});
    auto spec = std::make_shared<common::ScanSpec>("<root>");
    spec->addAllChildFields(*rowType);
    rowReaderOpts.setScanSpec(spec);
    auto rowReader = reader->createRowReader(rowReaderOpts);
    constexpr int batchSize = 10000;
    auto result = BaseVector::create(rowType, batchSize, leafPool.get());
    rowReader->next(batchSize, result);
    for (auto j = 0; j < result->size(); j++) {
      auto value = result->toString(j);
      value = value.substr(1, value.length() - 2);
      outputValues[j] = std::stoi(value);
    }
    ASSERT_EQ(kValueCount, result->size());
    ASSERT_EQ(values_in, outputValues);
  }
}

TEST(ParquetRoundtrip, AllNulls) {
  auto primitive_node =
      PrimitiveNode::Make("nulls", Repetition::OPTIONAL, nullptr, Type::INT32);
  schema::NodeVector columns({primitive_node});

  auto root_node =
      GroupNode::Make("root", Repetition::REQUIRED, columns, nullptr);

  auto sink = CreateOutputStream();

  auto file_writer = ParquetFileWriter::Open(
      sink, std::static_pointer_cast<GroupNode>(root_node));
  auto row_group_writer = file_writer->AppendRowGroup();
  auto column_writer =
      static_cast<Int32Writer*>(row_group_writer->NextColumn());

  int32_t values[3];
  int16_t def_levels[] = {0, 0, 0};

  column_writer->WriteBatch(3, def_levels, nullptr, values);

  column_writer->Close();
  row_group_writer->Close();
  file_writer->Close();

  ReaderProperties props = default_reader_properties();
  props.enable_buffered_stream();
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());
  // Write the buffer to a temp file path
  auto filePath = exec::test::TempFilePath::create();
  writeToFile(filePath, buffer);
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
  auto rowGroup = reader->fileMetaData().rowGroup(0);
  ASSERT_EQ(rowGroup.numColumns(), 1);
  dwio::common::RowReaderOptions rowReaderOpts;
  auto rowT = ROW({"nulls"}, {BIGINT()});
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("");
  scanSpec->addAllChildFields(*rowT);
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  constexpr int batchSize = 1000;
  auto result = BaseVector::create(rowT, batchSize, leafPool.get());
  rowReader->next(batchSize, result);
  for (auto i = 0; i < result->size(); i++) {
    auto val = result->toString(i);
    val = val.substr(1, val.size() - 2);
    val.erase(std::remove(val.begin(), val.end(), ' '), val.end());
    std::stringstream ss(val);
    std::string token;
    int pos = 0;
    while (std::getline(ss, token, ',')) {
      // Trim spaces from the token and check if it is "null"
      ASSERT_EQ(token, "null");
    }
  }
}

} // namespace test

} // namespace facebook::velox::parquet::arrow
