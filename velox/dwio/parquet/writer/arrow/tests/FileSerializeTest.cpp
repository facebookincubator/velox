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

#include <cstring>

#include <gtest/gtest.h>

#include "velox/dwio/parquet/writer/arrow/FileWriter.h"
#include "velox/dwio/parquet/writer/arrow/tests/ParquetTestFile.h"
#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::parquet::arrow {

using schema::GroupNode;
using schema::PrimitiveNode;

using velox::test::assertEqualVectors;

namespace test {

// Returns the Velox row type the reader produces for a file of 'numColumns'
// columns of TestType, one field per column named as setUpSchema() names it.
template <typename TestType>
RowTypePtr createRowType(int numColumns) {
  std::vector<std::string> names;
  names.reserve(numColumns);
  for (int i = 0; i < numColumns; ++i) {
    names.push_back(testColumnName(i));
  }
  if constexpr (std::is_same_v<TestType, Int96Type>) {
    return ROW(std::move(names), TIMESTAMP());
  } else if constexpr (
      std::is_same_v<TestType, ByteArrayType> ||
      std::is_same_v<TestType, FLBAType>) {
    return ROW(std::move(names), VARBINARY());
  } else {
    return ROW(std::move(names), CppToType<typename TestType::CType>::create());
  }
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

  // Builds the values the reader is expected to return for the 'numRows' rows
  // of the file starting at 'firstRow'. Every row group holds the same
  // values_, and every column of a row group holds the same values, so row n
  // of the file holds values_[n % rowsPerRowgroup_] in every column.
  RowVectorPtr expectedRows(
      velox::test::VectorMaker& vectorMaker,
      const RowTypePtr& rowType,
      int64_t firstRow,
      vector_size_t numRows) {
    auto valueIndex = [&](vector_size_t row) {
      return (firstRow + row) % rowsPerRowgroup_;
    };

    VectorPtr column;
    if constexpr (std::is_same_v<TestType, Int96Type>) {
      // INT96 stores nanoseconds of day in the low 8 bytes and the Julian day
      // in the high 4. The reader hands both to Timestamp::fromDaysAndNanos(),
      // which converts the Julian day to the Unix epoch, then truncates to the
      // precision requested in the reader options, milliseconds by default.
      column =
          vectorMaker.flatVector<Timestamp>(numRows, [&](vector_size_t row) {
            const auto& int96 = this->values_[valueIndex(row)];
            uint64_t nanosOfDay{0};
            std::memcpy(&nanosOfDay, int96.value, sizeof(nanosOfDay));
            return Timestamp::fromDaysAndNanos(
                       static_cast<int32_t>(int96.value[2]),
                       static_cast<int64_t>(nanosOfDay))
                .toPrecision(TimestampPrecision::kMilliseconds);
          });
    } else if constexpr (std::is_same_v<TestType, ByteArrayType>) {
      column = vectorMaker.flatVector<StringView>(
          numRows,
          [&](vector_size_t row) {
            const auto& value = this->values_[valueIndex(row)];
            return StringView{
                reinterpret_cast<const char*>(value.ptr),
                static_cast<int32_t>(value.len)};
          },
          nullptr,
          VARBINARY());
    } else if constexpr (std::is_same_v<TestType, FLBAType>) {
      column = vectorMaker.flatVector<StringView>(
          numRows,
          [&](vector_size_t row) {
            return StringView{
                reinterpret_cast<const char*>(
                    this->values_[valueIndex(row)].ptr),
                FLBA_LENGTH};
          },
          nullptr,
          VARBINARY());
    } else {
      column = vectorMaker.flatVector<typename TestType::CType>(
          numRows,
          [&](vector_size_t row) { return this->values_[valueIndex(row)]; });
    }

    // Every column holds the same values, so the same vector can back all of
    // them. assertEqualVectors() only reads the expected side.
    std::vector<VectorPtr> children(numColumns_, column);
    return vectorMaker.rowVector(rowType->names(), children);
  }

  // Reads 'rowReader' to EOF and asserts that every row of every column
  // matches the values that were written.
  void assertRoundtrip(
      dwio::common::RowReader& rowReader,
      const RowTypePtr& rowType,
      memory::MemoryPool* pool) {
    velox::test::VectorMaker vectorMaker{pool};
    constexpr int kBatchSize = 1000;
    auto result = BaseVector::create(rowType, 0, pool);
    int64_t rowsReadSoFar = 0;
    while (const auto rowsRead = rowReader.next(kBatchSize, result)) {
      ASSERT_EQ(rowsRead, result->size());
      ASSERT_NO_FATAL_FAILURE(assertEqualVectors(
          expectedRows(
              vectorMaker,
              rowType,
              rowsReadSoFar,
              static_cast<vector_size_t>(rowsRead)),
          result));
      rowsReadSoFar += rowsRead;
    }
    ASSERT_EQ(
        static_cast<int64_t>(numRowgroups_) * rowsPerRowgroup_, rowsReadSoFar);
  }

  void fileSerializeTest(Compression::type codecType) {
    fileSerializeTest(codecType, codecType);
  }

  void fileSerializeTest(
      Compression::type codecType,
      Compression::type expectedCodecType) {
    auto sink = createOutputStream();
    auto gnode = std::static_pointer_cast<GroupNode>(this->node_);

    WriterProperties::Builder propBuilder;
    if constexpr (std::is_same_v<TestType, FLBAType>) {
      // PageReader::prepareDictionary() handles FIXED_LEN_BYTE_ARRAY only for
      // decimal, so the reader cannot read a dictionary-encoded FLBA column as
      // VARBINARY. Every other type is written with the default properties,
      // which enable dictionary encoding.
      // TODO: Drop this once the reader supports it.
      propBuilder.disableDictionary();
    }
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

    const int64_t numRows =
        static_cast<int64_t>(numRowgroups_) * rowsPerRowgroup_;

    auto rowType = test::createRowType<TestType>(numColumns_);
    auto scanSpec = std::make_shared<velox::common::ScanSpec>("");
    scanSpec->addAllChildFields(*rowType);
    dwio::common::RowReaderOptions rowReaderOpts;
    rowReaderOpts.setScanSpec(scanSpec);
    rowReaderOpts.setRequestedType(rowType);

    auto file = test::ParquetTestFile::open(buffer, "FileSerializeTest");
    auto& reader = file.reader();
    ASSERT_EQ(numColumns_, reader.fileMetaData().rowGroup(0).numColumns());
    ASSERT_EQ(numRowgroups_, reader.fileMetaData().numRowGroups());
    ASSERT_EQ(numRows, reader.fileMetaData().numRows());

    auto rowReader = reader.createRowReader(rowReaderOpts);
    ASSERT_NO_FATAL_FAILURE(
        assertRoundtrip(*rowReader, rowType, file.leafPool()));

    for (int rg = 0; rg < numRowgroups_; ++rg) {
      auto rowGroupReader = reader.fileMetaData().rowGroup(rg);
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
          FAIL() << "The Velox reader has no CompressionKind for codec: "
                 << util::Codec::getCodecAsString(expectedCodecType);
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
  auto file = test::ParquetTestFile::open(buffer, "DisabledDictionary");
  auto& reader = file.reader();
  ASSERT_EQ(1, reader.fileMetaData().numRowGroups());
  auto rowGroup = reader.fileMetaData().rowGroup(0);
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
  auto rowType = ROW({"col"}, {INTEGER()});
  auto scanSpec = std::make_shared<common::ScanSpec>("");
  scanSpec->addAllChildFields(*rowType);
  dwio::common::RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(scanSpec);
  rowReaderOpts.setRequestedType(rowType);

  auto file =
      test::ParquetTestFile::open(buffer, "MultiPageDisabledDictionary");
  auto& reader = file.reader();
  ASSERT_EQ(1, reader.fileMetaData().numRowGroups());
  auto rowGroup = reader.fileMetaData().rowGroup(0);
  ASSERT_EQ(1, rowGroup.numColumns());
  ASSERT_EQ(kValueCount, rowGroup.numRows());

  velox::test::VectorMaker vectorMaker{file.leafPool()};
  auto expected = vectorMaker.rowVector(
      {"col"}, {vectorMaker.flatVector<int32_t>(valuesIn)});

  auto rowReader = reader.createRowReader(rowReaderOpts);
  auto result = BaseVector::create(rowType, 0, file.leafPool());
  const auto rowsRead = rowReader->next(kValueCount, result);
  ASSERT_EQ(kValueCount, rowsRead);
  ASSERT_EQ(kValueCount, result->size());
  assertEqualVectors(expected, result);
  // The single row group is fully consumed.
  ASSERT_EQ(0, rowReader->next(kValueCount, result));
}

TEST(ParquetRoundtrip, AllNulls) {
  constexpr int kNumRows = 3;
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

  int32_t values[kNumRows];
  int16_t defLevels[kNumRows] = {0, 0, 0};

  columnWriter->writeBatch(kNumRows, defLevels, nullptr, values);

  columnWriter->close();
  rowGroupWriter->close();
  fileWriter->close();

  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());
  auto rowType = ROW({"nulls"}, {INTEGER()});
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("");
  scanSpec->addAllChildFields(*rowType);
  dwio::common::RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(scanSpec);
  rowReaderOpts.setRequestedType(rowType);

  auto file = test::ParquetTestFile::open(buffer, "AllNulls");
  auto& reader = file.reader();
  auto rowGroup = reader.fileMetaData().rowGroup(0);
  ASSERT_EQ(1, rowGroup.numColumns());
  ASSERT_EQ(kNumRows, rowGroup.numRows());

  velox::test::VectorMaker vectorMaker{file.leafPool()};
  auto expected = vectorMaker.rowVector(
      {"nulls"}, {vectorMaker.allNullFlatVector<int32_t>(kNumRows)});

  auto rowReader = reader.createRowReader(rowReaderOpts);
  auto result = BaseVector::create(rowType, 0, file.leafPool());
  ASSERT_EQ(kNumRows, rowReader->next(kNumRows, result));
  ASSERT_EQ(kNumRows, result->size());
  assertEqualVectors(expected, result);
  ASSERT_EQ(0, rowReader->next(kNumRows, result));
}

} // namespace facebook::velox::parquet::arrow
