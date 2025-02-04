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

#include "arrow/testing/gtest_compat.h"

#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"
#include "velox/dwio/parquet/writer/arrow/ColumnWriter.h"
#include "velox/dwio/parquet/writer/arrow/FileWriter.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/tests/ColumnReader.h"
#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"
#include "velox/exec/tests/utils/TempFilePath.h"

namespace facebook::velox::parquet::arrow {

using schema::GroupNode;
using schema::NodePtr;
using schema::PrimitiveNode;
using ::testing::ElementsAre;

namespace test {

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

    // Write the buffer to a temp file
    auto temp_file = exec::test::TempFilePath::create();
    auto file_path = temp_file->getPath();
    auto local_write_file =
        std::make_unique<LocalWriteFile>(file_path, false, false);
    auto buffer_reader = std::make_shared<::arrow::io::BufferReader>(buffer);
    auto buffer_to_string = buffer_reader->buffer()->ToString();
    local_write_file->append(buffer_to_string);
    local_write_file->close();
    memory::MemoryManager::testingSetInstance({});
    std::shared_ptr<facebook::velox::memory::MemoryPool> root_pool;
    std::shared_ptr<facebook::velox::memory::MemoryPool> leaf_pool;
    root_pool = memory::memoryManager()->addRootPool("FileSerializeTest");
    leaf_pool = root_pool->addLeafChild("FileSerializeTest");
    dwio::common::ReaderOptions readerOptions{leaf_pool.get()};
    auto input = std::make_unique<dwio::common::BufferedInput>(
        std::make_shared<LocalReadFile>(file_path), readerOptions.memoryPool());
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
    constexpr int kBatchSize = 1000;
    auto result = BaseVector::create(rowT, kBatchSize, leaf_pool.get());
    auto actualRows = rowReader->next(kBatchSize, result);
    std::vector<typename TestType::c_type> col0;
    std::vector<typename TestType::c_type> col1;
    std::vector<typename TestType::c_type> col2;
    std::vector<typename TestType::c_type> col3;
    ASSERT_EQ(rows_per_rowgroup_, result->size());
    // auto a = result->as<RowVector>()->type();
    auto* rowVector = result->asUnchecked<RowVector>();
    // auto* col_0 =
    // rowVector->childAt(0)->loadedVector()->asUnchecked<FlatVector<typename
    // TestType::c_type>>(); auto x = col_0->rawValues();
    // ASSERT_EQ(this->values_, x);
    // auto sz = col_0->size();
    // auto* col_1 =
    // rowVector->childAt(0)->loadedVector()->asUnchecked<FlatVector<int32_t>>();
    // auto* col_2 =
    // rowVector->childAt(0)->loadedVector()->asUnchecked<FlatVector<int32_t>>();
    // auto* col_3 =
    // rowVector->childAt(0)->loadedVector()->asUnchecked<FlatVector<int32_t>>();
    // int32_t value = col_0->valueAt(0);  // Get the value at index 0
    // int32_t valu = col_1->valueAt(0);
    // int32_t vale = col_2->valueAt(0);
    // int32_t valez = col_3->valueAt(0);
    auto type = reader->typeWithId();
    if (expected_codec_type != Compression::type::LZ4) {
      for (int i = 0; i < rowVector->childrenSize(); i++) {
        // LZ4 loadedVector will fail with decompressed block size being way
        // bigger than 200...
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

    for (int rg = 0; rg < num_rowgroups_; ++rg) {
      auto rg_reader_t = reader->fileMetaData().rowGroup(rg);
      ASSERT_EQ(num_columns_, rg_reader_t.numColumns());
      ASSERT_EQ(rows_per_rowgroup_, rg_reader_t.numRows());
      // velox/dwio/parquet/writer/arrow/util/Compression.h
      // diffference between above arrow compression number and the below
      // compression number not sure if we can change the below to match the
      // above but once we pass in our own compression without arrow writer then
      // the type mismatch wont happen until then using switch to match up...
      // velox/common/compression/Compression.h
      auto expected_compression_kind = common::CompressionKind_NONE;
      switch (expected_codec_type) {
        case Compression::type::UNCOMPRESSED:
          expected_compression_kind = common::CompressionKind_NONE;
          break;
        case Compression::type::SNAPPY:
          expected_compression_kind = common::CompressionKind_SNAPPY;
          break;
        case Compression::type::GZIP:
          expected_compression_kind = common::CompressionKind_GZIP;
          break;
        case Compression::type::LZ4:
        case Compression::type::LZ4_HADOOP:
          expected_compression_kind = common::CompressionKind_LZ4;
          break;
        case Compression::type::ZSTD:
          expected_compression_kind = common::CompressionKind_ZSTD;
          break;
        default:
          expected_compression_kind = common::CompressionKind_NONE;
      }
      // Check that the specified compression was actually used.
      ASSERT_EQ(
          expected_compression_kind, rg_reader_t.columnChunk(0).compression());

      const int64_t total_byte_size_new = rg_reader_t.totalByteSize();
      const int64_t total_compressed_size_new =
          rg_reader_t.totalCompressedSize();
      if (expected_codec_type == Compression::UNCOMPRESSED &&
          expected_compression_kind == common::CompressionKind_NONE) {
        ASSERT_EQ(total_byte_size_new, total_compressed_size_new);
      } else {
        ASSERT_NE(total_byte_size_new, total_compressed_size_new);
      }

      int64_t total_column_byte_size_new = 0;
      int64_t total_column_compressed_size_new = 0;

      for (int i = 0; i < num_columns_; ++i) {
        int64_t values_read;
        ASSERT_FALSE(rg_reader_t.columnChunk(i).hasIndexPage());
        total_column_byte_size_new +=
            rg_reader_t.columnChunk(i).totalUncompressedSize();
        total_column_compressed_size_new +=
            rg_reader_t.columnChunk(i).totalCompressedSize();
      }
      ASSERT_EQ(total_byte_size_new, total_column_byte_size_new);
      ASSERT_EQ(total_compressed_size_new, total_column_compressed_size_new);
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
    // Int96Type,
    FloatType,
    DoubleType,
    BooleanType>
    // ByteArrayType,
    // FLBAType>
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
  // Write the buffer to a temp file
  auto tempFile = exec::test::TempFilePath::create();
  auto file_path = tempFile->getPath();
  auto local_write_file =
      std::make_unique<LocalWriteFile>(file_path, false, false);
  auto buffer_reader = std::make_shared<::arrow::io::BufferReader>(buffer);
  auto buffer_to_string = buffer_reader->buffer()->ToString();
  local_write_file->append(buffer_to_string);
  local_write_file->close();
  memory::MemoryManager::testingSetInstance({});
  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<facebook::velox::memory::MemoryPool> leafPool_;
  rootPool_ = memory::memoryManager()->addRootPool("FileSerializeTest");
  leafPool_ = rootPool_->addLeafChild("FileSerializeTest");
  dwio::common::ReaderOptions readerOptions{leafPool_.get()};
  auto input = std::make_unique<dwio::common::BufferedInput>(
      std::make_shared<LocalReadFile>(file_path), readerOptions.memoryPool());
  auto reader =
      std::make_unique<ParquetReader>(std::move(input), readerOptions);
  ASSERT_EQ(1, reader->thriftFileMetaData().row_groups.size());
  auto row_group = reader->fileMetaData().rowGroup(0);
  ASSERT_EQ(1, row_group.numColumns());
  ASSERT_EQ(1, row_group.numRows());
  ASSERT_FALSE(row_group.columnChunk(0).hasDictionaryPageOffset());
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
  // Write the buffer to a temp file
  auto tempFile = exec::test::TempFilePath::create();
  auto file_path = tempFile->getPath();
  auto local_write_file =
      std::make_unique<LocalWriteFile>(file_path, false, false);
  auto buffer_reader = std::make_shared<::arrow::io::BufferReader>(buffer);
  auto buffer_to_string = buffer_reader->buffer()->ToString();
  local_write_file->append(buffer_to_string);
  local_write_file->close();
  memory::MemoryManager::testingSetInstance({});
  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<facebook::velox::memory::MemoryPool> leafPool_;
  rootPool_ = memory::memoryManager()->addRootPool("FileSerializeTest");
  leafPool_ = rootPool_->addLeafChild("FileSerializeTest");
  dwio::common::ReaderOptions readerOptions{leafPool_.get()};
  auto input = std::make_unique<dwio::common::BufferedInput>(
      std::make_shared<LocalReadFile>(file_path), readerOptions.memoryPool());
  auto reader =
      std::make_unique<ParquetReader>(std::move(input), readerOptions);
  ASSERT_EQ(1, reader->thriftFileMetaData().row_groups.size());
  std::vector<int32_t> values_out(kValueCount);
  for (int r = 0; r < reader->thriftFileMetaData().row_groups.size(); ++r) {
    auto row_group = reader->fileMetaData().rowGroup(r);
    ASSERT_EQ(1, row_group.numColumns());
    ASSERT_EQ(kValueCount, row_group.numRows());
    dwio::common::RowReaderOptions rowReaderOpts;
    auto rowType = ROW({"col"}, {BIGINT()});
    auto spec = std::make_shared<common::ScanSpec>("<root>");
    spec->addAllChildFields(*rowType);
    rowReaderOpts.setScanSpec(spec);
    auto rowReader = reader->createRowReader(rowReaderOpts);
    constexpr int kBatchSize = 10000;
    auto result = BaseVector::create(rowType, kBatchSize, leafPool_.get());
    rowReader->next(kBatchSize, result);
    for (auto i = 0; i < result->size(); i++) {
      auto val_string = result->toString(i);
      val_string = val_string.substr(1, val_string.length() - 2);
      values_out[i] = std::stoi(val_string);
    }
    ASSERT_EQ(kValueCount, result->size());
    ASSERT_EQ(values_in, values_out);
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
  // Write the buffer to a temp file
  auto tempFile = exec::test::TempFilePath::create();
  auto file_path = tempFile->getPath();
  auto local_write_file =
      std::make_unique<LocalWriteFile>(file_path, false, false);
  auto buffer_reader = std::make_shared<::arrow::io::BufferReader>(buffer);
  auto buffer_to_string = buffer_reader->buffer()->ToString();
  local_write_file->append(buffer_to_string);
  local_write_file->close();
  memory::MemoryManager::testingSetInstance({});
  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<facebook::velox::memory::MemoryPool> leafPool_;
  rootPool_ = memory::memoryManager()->addRootPool("FileSerializeTest");
  leafPool_ = rootPool_->addLeafChild("FileSerializeTest");
  dwio::common::ReaderOptions readerOptions{leafPool_.get()};
  auto input = std::make_unique<dwio::common::BufferedInput>(
      std::make_shared<LocalReadFile>(file_path), readerOptions.memoryPool());
  auto reader =
      std::make_unique<ParquetReader>(std::move(input), readerOptions);
  auto rg = reader->fileMetaData().rowGroup(0);
  dwio::common::RowReaderOptions rowReaderOpts;
  auto rowT = ROW({"nulls"}, {BIGINT()});
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("");
  scanSpec->addAllChildFields(*rowT);
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  constexpr int kBatchSize = 1000;
  auto result = BaseVector::create(rowT, kBatchSize, leafPool_.get());
  auto actualRows = rowReader->next(kBatchSize, result);
  for (auto i = 0; i < result->size(); i++) {
    auto val = result->toString(i);
    val = val.substr(1, val.size() - 2);
    val.erase(std::remove(val.begin(), val.end(), ' '), val.end());
    std::stringstream ss(val);
    std::string token;
    int pos = 0;
    while (std::getline(ss, token, ',')) {
      // Trim spaces from the token and check if it is "null"
      if (token != "null") {
        VELOX_FAIL(
            fmt::format("value was not null but instead it was ", token));
      }
    }
  }
}

} // namespace test

} // namespace facebook::velox::parquet::arrow
