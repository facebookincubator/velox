/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/parquet/writer/arrow/ColumnWriter.h"
#include "velox/dwio/parquet/writer/arrow/FileWriter.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"
#include "velox/dwio/parquet/writer/arrow/tests/ColumnReader.h"
#include "velox/dwio/parquet/writer/arrow/tests/FileReader.h"
#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"

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
    numColumns_ = 4;
    numRowgroups_ = 4;
    rowsPerRowgroup_ = 50;
    rowsPerBatch_ = 10;
    this->setUpSchema(Repetition::kOptional, numColumns_);
  }

 protected:
  int numColumns_;
  int numRowgroups_;
  int rowsPerRowgroup_;
  int rowsPerBatch_;

  void fileSerializeTest(Compression::type codecType) {
    fileSerializeTest(codecType, codecType);
  }

  void fileSerializeTest(
      Compression::type codecType,
      Compression::type expectedCodecType) {
    auto sink = createOutputStream();
    auto gnode = std::static_pointer_cast<GroupNode>(this->node_);

    WriterProperties::Builder propBuilder;

    for (int i = 0; i < numColumns_; ++i) {
      propBuilder.compression(this->schema_.column(i)->name(), codecType);
    }
    std::shared_ptr<WriterProperties> WriterProperties = propBuilder.build();

    auto fileWriter = ParquetFileWriter::open(sink, gnode, WriterProperties);
    this->generateData(rowsPerRowgroup_);
    for (int rg = 0; rg < numRowgroups_ / 2; ++rg) {
      RowGroupWriter* rowGroupWriter;
      rowGroupWriter = fileWriter->appendRowGroup();
      for (int col = 0; col < numColumns_; ++col) {
        auto columnWriter = static_cast<TypedColumnWriter<TestType>*>(
            rowGroupWriter->nextColumn());
        columnWriter->writeBatch(
            rowsPerRowgroup_,
            this->defLevels_.data(),
            nullptr,
            this->valuesPtr_);
        columnWriter->close();
        // Ensure column() API which is specific to BufferedRowGroup cannot be.
        // Called.
        ASSERT_THROW(rowGroupWriter->column(col), ParquetException);
      }
      EXPECT_EQ(0, rowGroupWriter->totalCompressedBytes());
      EXPECT_NE(0, rowGroupWriter->totalBytesWritten());
      EXPECT_NE(0, rowGroupWriter->totalCompressedBytesWritten());
      rowGroupWriter->close();
      EXPECT_EQ(0, rowGroupWriter->totalCompressedBytes());
      EXPECT_NE(0, rowGroupWriter->totalBytesWritten());
      EXPECT_NE(0, rowGroupWriter->totalCompressedBytesWritten());
    }
    // Write half BufferedRowGroups.
    for (int rg = 0; rg < numRowgroups_ / 2; ++rg) {
      RowGroupWriter* rowGroupWriter;
      rowGroupWriter = fileWriter->appendBufferedRowGroup();
      for (int batch = 0; batch < (rowsPerRowgroup_ / rowsPerBatch_); ++batch) {
        for (int col = 0; col < numColumns_; ++col) {
          auto columnWriter = static_cast<TypedColumnWriter<TestType>*>(
              rowGroupWriter->column(col));
          columnWriter->writeBatch(
              rowsPerBatch_,
              this->defLevels_.data() + (batch * rowsPerBatch_),
              nullptr,
              this->valuesPtr_ + (batch * rowsPerBatch_));
          // Ensure NextColumn() API which is specific to RowGroup cannot be.
          // Called.
          ASSERT_THROW(rowGroupWriter->nextColumn(), ParquetException);
        }
      }
      // Total_compressed_bytes() may equal to 0 if no dictionary enabled and
      // no. Buffered values.
      EXPECT_EQ(0, rowGroupWriter->totalBytesWritten());
      EXPECT_EQ(0, rowGroupWriter->totalCompressedBytesWritten());
      for (int col = 0; col < numColumns_; ++col) {
        auto columnWriter = static_cast<TypedColumnWriter<TestType>*>(
            rowGroupWriter->column(col));
        columnWriter->close();
      }
      rowGroupWriter->close();
      EXPECT_EQ(0, rowGroupWriter->totalCompressedBytes());
      EXPECT_NE(0, rowGroupWriter->totalBytesWritten());
      EXPECT_NE(0, rowGroupWriter->totalCompressedBytesWritten());
    }
    fileWriter->close();

    PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());

    int numRows_ = numRowgroups_ * rowsPerRowgroup_;

    auto source = std::make_shared<::arrow::io::BufferReader>(buffer);
    auto fileReader = ParquetFileReader::open(source);
    ASSERT_EQ(numColumns_, fileReader->metadata()->numColumns());
    ASSERT_EQ(numRowgroups_, fileReader->metadata()->numRowGroups());
    ASSERT_EQ(numRows_, fileReader->metadata()->numRows());

    for (int rg = 0; rg < numRowgroups_; ++rg) {
      auto rgReader = fileReader->rowGroup(rg);
      auto rgMetadata = rgReader->metadata();
      ASSERT_EQ(numColumns_, rgMetadata->numColumns());
      ASSERT_EQ(rowsPerRowgroup_, rgMetadata->numRows());
      // Check that the specified compression was actually used.
      ASSERT_EQ(expectedCodecType, rgMetadata->columnChunk(0)->compression());

      const int64_t totalByteSize = rgMetadata->totalByteSize();
      const int64_t totalCompressedSize = rgMetadata->totalCompressedSize();
      if (expectedCodecType == Compression::UNCOMPRESSED) {
        ASSERT_EQ(totalByteSize, totalCompressedSize);
      } else {
        ASSERT_NE(totalByteSize, totalCompressedSize);
      }

      int64_t totalColumnByteSize = 0;
      int64_t totalColumnCompressedSize = 0;

      for (int i = 0; i < numColumns_; ++i) {
        int64_t valuesRead;
        ASSERT_FALSE(rgMetadata->columnChunk(i)->hasIndexPage());
        totalColumnByteSize +=
            rgMetadata->columnChunk(i)->totalUncompressedSize();
        totalColumnCompressedSize +=
            rgMetadata->columnChunk(i)->totalCompressedSize();

        std::vector<int16_t> defLevelsOut(rowsPerRowgroup_);
        std::vector<int16_t> repLevelsOut(rowsPerRowgroup_);
        auto colReader = std::static_pointer_cast<TypedColumnReader<TestType>>(
            rgReader->column(i));
        this->setupValuesOut(rowsPerRowgroup_);
        colReader->readBatch(
            rowsPerRowgroup_,
            defLevelsOut.data(),
            repLevelsOut.data(),
            this->valuesOutPtr_,
            &valuesRead);
        this->syncValuesOut();
        ASSERT_EQ(rowsPerRowgroup_, valuesRead);
        ASSERT_EQ(this->values_, this->valuesOut_);
        ASSERT_EQ(this->defLevels_, defLevelsOut);
      }

      ASSERT_EQ(totalByteSize, totalColumnByteSize);
      ASSERT_EQ(totalCompressedSize, totalColumnCompressedSize);
    }
  }

  void unequalNumRows(
      int64_t maxRows,
      const std::vector<int64_t> rowsPerColumn) {
    auto sink = createOutputStream();
    auto gnode = std::static_pointer_cast<GroupNode>(this->node_);

    std::shared_ptr<WriterProperties> props =
        WriterProperties::Builder().build();

    auto fileWriter = ParquetFileWriter::open(sink, gnode, props);

    RowGroupWriter* rowGroupWriter;
    rowGroupWriter = fileWriter->appendRowGroup();

    this->generateData(maxRows);
    for (int col = 0; col < numColumns_; ++col) {
      auto columnWriter = static_cast<TypedColumnWriter<TestType>*>(
          rowGroupWriter->nextColumn());
      columnWriter->writeBatch(
          rowsPerColumn[col],
          this->defLevels_.data(),
          nullptr,
          this->valuesPtr_);
      columnWriter->close();
    }
    rowGroupWriter->close();
    fileWriter->close();
  }

  void unequalNumRowsBuffered(
      int64_t maxRows,
      const std::vector<int64_t> rowsPerColumn) {
    auto sink = createOutputStream();
    auto gnode = std::static_pointer_cast<GroupNode>(this->node_);

    std::shared_ptr<WriterProperties> props =
        WriterProperties::Builder().build();

    auto fileWriter = ParquetFileWriter::open(sink, gnode, props);

    RowGroupWriter* rowGroupWriter;
    rowGroupWriter = fileWriter->appendBufferedRowGroup();

    this->generateData(maxRows);
    for (int col = 0; col < numColumns_; ++col) {
      auto columnWriter = static_cast<TypedColumnWriter<TestType>*>(
          rowGroupWriter->column(col));
      columnWriter->writeBatch(
          rowsPerColumn[col],
          this->defLevels_.data(),
          nullptr,
          this->valuesPtr_);
      columnWriter->close();
    }
    rowGroupWriter->close();
    fileWriter->close();
  }

  void repeatedUnequalRows() {
    // Optional and repeated, so definition and repetition levels.
    this->setUpSchema(Repetition::kRepeated);

    const int kNumRows = 100;
    this->generateData(kNumRows);

    auto sink = createOutputStream();
    auto gnode = std::static_pointer_cast<GroupNode>(this->node_);
    std::shared_ptr<WriterProperties> props =
        WriterProperties::Builder().build();
    auto fileWriter = ParquetFileWriter::open(sink, gnode, props);

    RowGroupWriter* rowGroupWriter;
    rowGroupWriter = fileWriter->appendRowGroup();

    this->generateData(kNumRows);

    std::vector<int16_t> definitionLevels(kNumRows, 1);
    std::vector<int16_t> repetitionLevels(kNumRows, 0);

    {
      auto columnWriter = static_cast<TypedColumnWriter<TestType>*>(
          rowGroupWriter->nextColumn());
      columnWriter->writeBatch(
          kNumRows,
          definitionLevels.data(),
          repetitionLevels.data(),
          this->valuesPtr_);
      columnWriter->close();
    }

    definitionLevels[1] = 0;
    repetitionLevels[3] = 1;

    {
      auto columnWriter = static_cast<TypedColumnWriter<TestType>*>(
          rowGroupWriter->nextColumn());
      columnWriter->writeBatch(
          kNumRows,
          definitionLevels.data(),
          repetitionLevels.data(),
          this->valuesPtr_);
      columnWriter->close();
    }
  }

  void zeroRowsRowGroup() {
    auto sink = createOutputStream();
    auto gnode = std::static_pointer_cast<GroupNode>(this->node_);

    std::shared_ptr<WriterProperties> props =
        WriterProperties::Builder().build();

    auto fileWriter = ParquetFileWriter::open(sink, gnode, props);

    RowGroupWriter* rowGroupWriter;

    rowGroupWriter = fileWriter->appendRowGroup();
    for (int col = 0; col < numColumns_; ++col) {
      auto columnWriter = static_cast<TypedColumnWriter<TestType>*>(
          rowGroupWriter->nextColumn());
      columnWriter->close();
    }
    rowGroupWriter->close();

    rowGroupWriter = fileWriter->appendBufferedRowGroup();
    for (int col = 0; col < numColumns_; ++col) {
      auto columnWriter = static_cast<TypedColumnWriter<TestType>*>(
          rowGroupWriter->column(col));
      columnWriter->close();
    }
    rowGroupWriter->close();

    fileWriter->close();
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
  ASSERT_NO_FATAL_FAILURE(this->fileSerializeTest(Compression::UNCOMPRESSED));
}

TYPED_TEST(TestSerialize, TooFewRows) {
  std::vector<int64_t> numRows = {100, 100, 100, 99};
  ASSERT_THROW(this->unequalNumRows(100, numRows), ParquetException);
  ASSERT_THROW(this->unequalNumRowsBuffered(100, numRows), ParquetException);
}

TYPED_TEST(TestSerialize, TooManyRows) {
  std::vector<int64_t> numRows = {100, 100, 100, 101};
  ASSERT_THROW(this->unequalNumRows(101, numRows), ParquetException);
  ASSERT_THROW(this->unequalNumRowsBuffered(101, numRows), ParquetException);
}

TYPED_TEST(TestSerialize, ZeroRows) {
  ASSERT_NO_THROW(this->zeroRowsRowGroup());
}

TYPED_TEST(TestSerialize, RepeatedTooFewRows) {
  ASSERT_THROW(this->repeatedUnequalRows(), ParquetException);
}

TYPED_TEST(TestSerialize, SmallFileSnappy) {
  ASSERT_NO_FATAL_FAILURE(this->fileSerializeTest(Compression::SNAPPY));
}

#ifdef ARROW_WITH_BROTLI
TYPED_TEST(TestSerialize, SmallFileBrotli) {
  ASSERT_NO_FATAL_FAILURE(this->fileSerializeTest(Compression::BROTLI));
}
#endif

TYPED_TEST(TestSerialize, SmallFileGzip) {
  ASSERT_NO_FATAL_FAILURE(this->fileSerializeTest(Compression::GZIP));
}

TYPED_TEST(TestSerialize, SmallFileLz4) {
  ASSERT_NO_FATAL_FAILURE(this->fileSerializeTest(Compression::LZ4));
}

TYPED_TEST(TestSerialize, SmallFileLz4Hadoop) {
  ASSERT_NO_FATAL_FAILURE(this->fileSerializeTest(Compression::LZ4_HADOOP));
}

TYPED_TEST(TestSerialize, SmallFileZstd) {
  ASSERT_NO_FATAL_FAILURE(this->fileSerializeTest(Compression::ZSTD));
}

TEST(TestBufferedRowGroupWriter, DisabledDictionary) {
  // PARQUET-1706:
  // Wrong dictionary_page_offset when writing only data pages via.
  // BufferedPageWriter.
  auto sink = createOutputStream();
  auto writerProps = WriterProperties::Builder().disableDictionary()->build();
  schema::NodeVector fields;
  fields.push_back(
      PrimitiveNode::make("col", Repetition::kRequired, Type::kInt32));
  auto schema = std::static_pointer_cast<GroupNode>(
      GroupNode::make("schema", Repetition::kRequired, fields));
  auto fileWriter = ParquetFileWriter::open(sink, schema, writerProps);
  auto rgWriter = fileWriter->appendBufferedRowGroup();
  auto colWriter = static_cast<Int32Writer*>(rgWriter->column(0));
  int value = 0;
  colWriter->writeBatch(1, nullptr, nullptr, &value);
  rgWriter->close();
  fileWriter->close();
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());

  auto source = std::make_shared<::arrow::io::BufferReader>(buffer);
  auto fileReader = ParquetFileReader::open(source);
  ASSERT_EQ(1, fileReader->metadata()->numRowGroups());
  auto rgReader = fileReader->rowGroup(0);
  ASSERT_EQ(1, rgReader->metadata()->numColumns());
  ASSERT_EQ(1, rgReader->metadata()->numRows());
  ASSERT_FALSE(rgReader->metadata()->columnChunk(0)->hasDictionaryPage());
}

TEST(TestBufferedRowGroupWriter, MultiPageDisabledDictionary) {
  constexpr int kValueCount = 10000;
  constexpr int kPageSize = 16384;
  auto sink = createOutputStream();
  auto writerProps = WriterProperties::Builder()
                         .disableDictionary()
                         ->dataPagesize(kPageSize)
                         ->build();
  schema::NodeVector fields;
  fields.push_back(
      PrimitiveNode::make("col", Repetition::kRequired, Type::kInt32));
  auto schema = std::static_pointer_cast<GroupNode>(
      GroupNode::make("schema", Repetition::kRequired, fields));
  auto fileWriter = ParquetFileWriter::open(sink, schema, writerProps);
  auto rgWriter = fileWriter->appendBufferedRowGroup();
  auto colWriter = static_cast<Int32Writer*>(rgWriter->column(0));
  std::vector<int32_t> valuesIn;
  for (int i = 0; i < kValueCount; ++i) {
    valuesIn.push_back((i % 100) + 1);
  }
  colWriter->writeBatch(kValueCount, nullptr, nullptr, valuesIn.data());
  rgWriter->close();
  fileWriter->close();
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());

  auto source = std::make_shared<::arrow::io::BufferReader>(buffer);
  auto fileReader = ParquetFileReader::open(source);
  auto fileMetadata = fileReader->metadata();
  ASSERT_EQ(1, fileReader->metadata()->numRowGroups());
  std::vector<int32_t> valuesOut(kValueCount);
  for (int r = 0; r < fileMetadata->numRowGroups(); ++r) {
    auto rgReader = fileReader->rowGroup(r);
    ASSERT_EQ(1, rgReader->metadata()->numColumns());
    ASSERT_EQ(kValueCount, rgReader->metadata()->numRows());
    int64_t totalValuesRead = 0;
    std::shared_ptr<ColumnReader> colReader;
    ASSERT_NO_THROW(colReader = rgReader->column(0));
    Int32Reader* int32Reader = static_cast<Int32Reader*>(colReader.get());
    int64_t vn = kValueCount;
    int32_t* vx = valuesOut.data();
    while (int32Reader->hasNext()) {
      int64_t valuesRead;
      int32Reader->readBatch(vn, nullptr, nullptr, vx, &valuesRead);
      vn -= valuesRead;
      vx += valuesRead;
      totalValuesRead += valuesRead;
    }
    ASSERT_EQ(kValueCount, totalValuesRead);
    ASSERT_EQ(valuesIn, valuesOut);
  }
}

TEST(ParquetRoundtrip, AllNulls) {
  auto primitiveNode = PrimitiveNode::make(
      "nulls", Repetition::kOptional, nullptr, Type::kInt32);
  schema::NodeVector columns({primitiveNode});

  auto rootNode =
      GroupNode::make("root", Repetition::kRequired, columns, nullptr);

  auto sink = createOutputStream();

  auto fileWriter = ParquetFileWriter::open(
      sink, std::static_pointer_cast<GroupNode>(rootNode));
  auto rowGroupWriter = fileWriter->appendRowGroup();
  auto columnWriter = static_cast<Int32Writer*>(rowGroupWriter->nextColumn());

  int32_t values[3];
  int16_t defLevels[] = {0, 0, 0};

  columnWriter->writeBatch(3, defLevels, nullptr, values);

  columnWriter->close();
  rowGroupWriter->close();
  fileWriter->close();

  ReaderProperties props = defaultReaderProperties();
  props.enableBufferedStream();
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());

  auto source = std::make_shared<::arrow::io::BufferReader>(buffer);
  auto fileReader = ParquetFileReader::open(source, props);
  auto RowGroupReader = fileReader->rowGroup(0);
  auto ColumnReader =
      std::static_pointer_cast<Int32Reader>(RowGroupReader->column(0));

  int64_t valuesRead;
  defLevels[0] = -1;
  defLevels[1] = -1;
  defLevels[2] = -1;
  ColumnReader->readBatch(3, defLevels, nullptr, values, &valuesRead);
  EXPECT_THAT(defLevels, ElementsAre(0, 0, 0));
}

} // namespace test

} // namespace facebook::velox::parquet::arrow
