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

template <typename TestType>
RowTypePtr createRowBasedOnType() {
  auto rowT =
      ROW({"column_0", "column_1", "column_2", "column_3"},
          {INTEGER(), INTEGER(), INTEGER(), INTEGER()});
  if constexpr (std::is_same<TestType, Int64Type>::value) {
    rowT =
        ROW({"column_0", "column_1", "column_2", "column_3"},
            {BIGINT(), BIGINT(), BIGINT(), BIGINT()});
  } else if constexpr (std::is_same<TestType, Int96Type>::value) {
    rowT =
        ROW({"column_0", "column_1", "column_2", "column_3"},
            {TIMESTAMP(), TIMESTAMP(), TIMESTAMP(), TIMESTAMP()});
  } else if constexpr (std::is_same<TestType, FloatType>::value) {
    rowT =
        ROW({"column_0", "column_1", "column_2", "column_3"},
            {REAL(), REAL(), REAL(), REAL()});
  } else if constexpr (std::is_same<TestType, DoubleType>::value) {
    rowT =
        ROW({"column_0", "column_1", "column_2", "column_3"},
            {DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE()});
  } else if constexpr (std::is_same<TestType, BooleanType>::value) {
    rowT =
        ROW({"column_0", "column_1", "column_2", "column_3"},
            {BOOLEAN(), BOOLEAN(), BOOLEAN(), BOOLEAN()});
  } else if constexpr (
      std::is_same<TestType, ByteArrayType>::value ||
      std::is_same<TestType, FLBAType>::value) {
    rowT =
        ROW({"column_0", "column_1", "column_2", "column_3"},
            {VARBINARY(), VARBINARY(), VARBINARY(), VARBINARY()});
  }
  return rowT;
}
} // namespace test

template <typename TestType>
class TestSerialize : public test::PrimitiveTypedTest<TestType> {
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
    propBuilder.disableDictionary();

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

    // Write the buffer to a temp file path
    auto filePath = exec::test::TempFilePath::create();
    test::writeToFile(filePath, buffer);
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool =
        memory::memoryManager()->addRootPool("FileSerializeTest");
    std::shared_ptr<facebook::velox::memory::MemoryPool> leafPool =
        rootPool->addLeafChild("FileSerializeTest");
    dwio::common::RowReaderOptions rowReaderOpts;
    auto rowT = test::createRowBasedOnType<TestType>();
    auto scanSpec = std::make_shared<velox::common::ScanSpec>("");
    scanSpec->addAllChildFields(*rowT);
    rowReaderOpts.setScanSpec(scanSpec);
    rowReaderOpts.setRequestedType(rowT);
    dwio::common::ReaderOptions readerOptions{leafPool.get()};
    readerOptions.setScanSpec(scanSpec);
    auto input = std::make_unique<dwio::common::BufferedInput>(
        std::make_shared<LocalReadFile>(filePath->getPath()),
        readerOptions.memoryPool());
    auto reader =
        std::make_unique<ParquetReader>(std::move(input), readerOptions);
    ASSERT_EQ(numColumns_, reader->fileMetaData().rowGroup(0).numColumns());
    ASSERT_EQ(numRowgroups_, reader->fileMetaData().numRowGroups());
    ASSERT_EQ(numRows_, reader->fileMetaData().numRows());
    auto rowReader = reader->createRowReader(rowReaderOpts);
    constexpr int batchSize = 1000;
    auto result = BaseVector::create(rowT, batchSize, leafPool.get());
    rowReader->next(batchSize, result);
    ASSERT_EQ(rowsPerRowgroup_, result->size());
    auto* rowVector = result->template as<RowVector>();
    auto type = reader->typeWithId();
    // Compression type LZ4 causes loadedVector() to fail with decompressed
    // block size being way bigger than 200...
    for (int i = 0; i < rowVector->childrenSize(); i++) {
      if constexpr (std::is_same<TestType, Int96Type>::value) {
        auto column = rowVector->childAt(i)
                          ->loadedVector()
                          ->template asUnchecked<FlatVector<Timestamp>>();
        ASSERT_TRUE(column);
        ASSERT_EQ(column->size(), 50);
        for (auto j = 0; j < column->size(); j++) {
          ASSERT_FALSE(column->isNullAt(j));
          ASSERT_EQ(
              column->valueAt(j).getSeconds(),
              int96GetSeconds(this->values_[j]));
        }
      } else if constexpr (std::is_same<TestType, ByteArrayType>::value) {
        auto column = rowVector->childAt(i)
                          ->loadedVector()
                          ->template asUnchecked<FlatVector<StringView>>();
        ASSERT_TRUE(column);
        ASSERT_EQ(column->size(), 50);
        for (auto j = 0; j < column->size(); j++) {
          ASSERT_FALSE(column->isNullAt(j));
          auto inputValue = StringView{
              reinterpret_cast<const char*>(this->values_[j].ptr),
              static_cast<int32_t>(this->values_[j].len)};
          ASSERT_EQ(inputValue, column->valueAt(j));
        }
      } else if constexpr (std::is_same<TestType, FLBAType>::value) {
        auto column = rowVector->childAt(i)
                          ->loadedVector()
                          ->template asUnchecked<FlatVector<StringView>>();
        ASSERT_TRUE(column);
        ASSERT_EQ(column->size(), 50);
        for (auto j = 0; j < column->size(); j++) {
          ASSERT_FALSE(column->isNullAt(j));
          auto inputValue = StringView{
              reinterpret_cast<const char*>(this->values_[j].ptr), FLBA_LENGTH};
          ASSERT_EQ(inputValue, column->valueAt(j));
        }
      } else {
        auto column =
            rowVector->childAt(i)
                ->loadedVector()
                ->template asUnchecked<FlatVector<typename TestType::CType>>();
        ASSERT_TRUE(column);
        ASSERT_EQ(column->size(), 50);
        for (auto j = 0; j < column->size(); j++) {
          ASSERT_FALSE(column->isNullAt(j));
          ASSERT_EQ(column->valueAt(j), this->values_[j]);
        }
      }
    }

    for (int rg = 0; rg < numRowgroups_; ++rg) {
      auto rowGroupReader = reader->fileMetaData().rowGroup(rg);
      ASSERT_EQ(numColumns_, rowGroupReader.numColumns());
      ASSERT_EQ(rowsPerRowgroup_, rowGroupReader.numRows());
      // There is a difference between
      // velox/dwio/parquet/writer/arrow/util/Compression.h compression number
      // and velox/common/compression/Compression.h compression number. Once we
      // pass in our own compression without arrow writer then the type mismatch
      // wont happen.
      auto expectedCompressionKind = common::CompressionKind_NONE;
      switch (expectedCodecType) {
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
          expectedCompressionKind = common::CompressionKind_LZ4;
          break;
        case Compression::type::LZ4_HADOOP:
          expectedCompressionKind = common::CompressionKind_LZ4_HADOOP;
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
      if (expectedCodecType == Compression::UNCOMPRESSED &&
          expectedCompressionKind == common::CompressionKind_NONE) {
        ASSERT_EQ(totalByteSize, totalCompressedSize);
      } else {
        ASSERT_NE(totalByteSize, totalCompressedSize);
      }

      int64_t totalColumnByteSize = 0;
      int64_t totalColumnCompressedSize = 0;

      for (int i = 0; i < numColumns_; ++i) {
        int64_t valuesRead;
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
  // Write the buffer to a temp file path
  auto filePath = exec::test::TempFilePath::create();
  test::writeToFile(filePath, buffer);
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
  // Write the buffer to a temp file path
  auto filePath = exec::test::TempFilePath::create();
  test::writeToFile(filePath, buffer);
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
    ASSERT_EQ(valuesIn, outputValues);
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
  // Write the buffer to a temp file path
  auto filePath = exec::test::TempFilePath::create();
  test::writeToFile(filePath, buffer);
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

} // namespace test

} // namespace facebook::velox::parquet::arrow
