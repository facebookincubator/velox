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

#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "arrow/io/buffered.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_builders.h"

#include "velox/dwio/parquet/writer/arrow/ColumnWriter.h"
#include "velox/dwio/parquet/writer/arrow/FileWriter.h"
#include "velox/dwio/parquet/writer/arrow/Metadata.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/dwio/parquet/writer/arrow/Statistics.h"
#include "velox/dwio/parquet/writer/arrow/ThriftInternal.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"
#include "velox/dwio/parquet/writer/arrow/tests/ColumnReader.h"
#include "velox/dwio/parquet/writer/arrow/tests/FileReader.h"
#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"

namespace bit_util = arrow::bit_util;

namespace facebook::velox::parquet::arrow {

using arrow::internal::kJulianEpochOffsetDays;
using schema::GroupNode;
using schema::NodePtr;
using schema::PrimitiveNode;
using util::Codec;

namespace test {

// The default size used in most tests.
const int SMALL_SIZE = 100;
#ifdef PARQUET_VALGRIND
// Larger size to test some corner cases, only used in some specific cases.
const int LARGE_SIZE = 10000;
// Very large size to test dictionary fallback.
const int VERY_LARGE_SIZE = 40000;
// Reduced dictionary page size to use for testing dictionary fallback with.
// Valgrind.
const int64_t DICTIONARY_PAGE_SIZE = 1024;
#else
// Larger size to test some corner cases, only used in some specific cases.
const int LARGE_SIZE = 100000;
// Very large size to test dictionary fallback.
const int VERY_LARGE_SIZE = 400000;
// Dictionary page size to use for testing dictionary fallback.
const int64_t DICTIONARY_PAGE_SIZE = 1024 * 1024;
#endif

template <typename TestType>
class TestPrimitiveWriter : public PrimitiveTypedTest<TestType> {
 public:
  void SetUp() {
    this->setupValuesOut(SMALL_SIZE);
    writerProperties_ = defaultWriterProperties();
    definitionLevelsOut_.resize(SMALL_SIZE);
    repetitionLevelsOut_.resize(SMALL_SIZE);

    this->setUpSchema(Repetition::kRequired);

    descr_ = this->schema_.column(0);
  }

  Type::type typeNum() {
    return TestType::typeNum;
  }

  void buildReader(
      int64_t numRows,
      Compression::type compression = Compression::UNCOMPRESSED,
      bool pageChecksumVerify = false) {
    ASSERT_OK_AND_ASSIGN(auto buffer, sink_->Finish());
    auto source = std::make_shared<::arrow::io::BufferReader>(buffer);
    ReaderProperties readerProperties;
    readerProperties.setPageChecksumVerification(pageChecksumVerify);
    std::unique_ptr<PageReader> pageReader = PageReader::open(
        std::move(source), numRows, compression, readerProperties);
    reader_ = std::static_pointer_cast<TypedColumnReader<TestType>>(
        ColumnReader::make(this->descr_, std::move(pageReader)));
  }

  std::shared_ptr<TypedColumnWriter<TestType>> buildWriter(
      int64_t outputSize = SMALL_SIZE,
      const ColumnProperties& columnProps = ColumnProperties(),
      const ParquetVersion::type version = ParquetVersion::PARQUET_1_0,
      bool enableChecksum = false) {
    sink_ = createOutputStream();
    WriterProperties::Builder wpBuilder;
    wpBuilder.version(version);
    if (columnProps.encoding() == Encoding::kPlainDictionary ||
        columnProps.encoding() == Encoding::kRleDictionary) {
      wpBuilder.enableDictionary();
      wpBuilder.dictionaryPagesizeLimit(DICTIONARY_PAGE_SIZE);
    } else {
      wpBuilder.disableDictionary();
      wpBuilder.encoding(columnProps.encoding());
    }
    if (enableChecksum) {
      wpBuilder.enablePageChecksum();
    }
    wpBuilder.maxStatisticsSize(columnProps.maxStatisticsSize());
    writerProperties_ = wpBuilder.build();

    metadata_ =
        ColumnChunkMetaDataBuilder::make(writerProperties_, this->descr_);
    std::unique_ptr<PageWriter> pager = PageWriter::open(
        sink_,
        columnProps.compression(),
        Codec::useDefaultCompressionLevel(),
        metadata_.get(),
        /* row_group_ordinal */ -1,
        /* column_chunk_ordinal*/ -1,
        ::arrow::default_memory_pool(),
        /* buffered_row_group */ false,
        /* header_encryptor */ NULLPTR,
        /* data_encryptor */ NULLPTR,
        enableChecksum);
    std::shared_ptr<ColumnWriter> writer = ColumnWriter::make(
        metadata_.get(), std::move(pager), writerProperties_.get());
    return std::static_pointer_cast<TypedColumnWriter<TestType>>(writer);
  }

  void readColumn(
      Compression::type compression = Compression::UNCOMPRESSED,
      bool pageChecksumVerify = false) {
    buildReader(
        static_cast<int64_t>(this->valuesOut_.size()),
        compression,
        pageChecksumVerify);
    reader_->readBatch(
        static_cast<int>(this->valuesOut_.size()),
        definitionLevelsOut_.data(),
        repetitionLevelsOut_.data(),
        this->valuesOutPtr_,
        &valuesRead_);
    this->syncValuesOut();
  }

  void readColumnFully(
      Compression::type compression = Compression::UNCOMPRESSED,
      bool pageChecksumVerify = false);

  void testRequiredWithEncoding(Encoding::type encoding) {
    return testRequiredWithSettings(
        encoding, Compression::UNCOMPRESSED, false, false);
  }

  void testRequiredWithSettings(
      Encoding::type encoding,
      Compression::type compression,
      bool enableDictionary,
      bool enableStatistics,
      int64_t numRows = SMALL_SIZE,
      int compressionLevel = Codec::useDefaultCompressionLevel(),
      bool enableChecksum = false) {
    this->generateData(numRows);

    this->writeRequiredWithSettings(
        encoding,
        compression,
        enableDictionary,
        enableStatistics,
        compressionLevel,
        numRows,
        enableChecksum);
    ASSERT_NO_FATAL_FAILURE(
        this->readAndCompare(compression, numRows, enableChecksum));

    this->writeRequiredWithSettingsSpaced(
        encoding,
        compression,
        enableDictionary,
        enableStatistics,
        numRows,
        compressionLevel,
        enableChecksum);
    ASSERT_NO_FATAL_FAILURE(
        this->readAndCompare(compression, numRows, enableChecksum));
  }

  void testDictionaryFallbackEncoding(ParquetVersion::type version) {
    this->generateData(VERY_LARGE_SIZE);
    ColumnProperties columnProperties;
    columnProperties.setDictionaryEnabled(true);

    if (version == ParquetVersion::PARQUET_1_0) {
      columnProperties.setEncoding(Encoding::kPlainDictionary);
    } else {
      columnProperties.setEncoding(Encoding::kRleDictionary);
    }

    auto writer = this->buildWriter(VERY_LARGE_SIZE, columnProperties, version);

    writer->writeBatch(
        this->values_.size(), nullptr, nullptr, this->valuesPtr_);
    writer->close();

    // Read all rows so we are sure that also the non-dictionary pages are read.
    // Correctly.
    this->setupValuesOut(VERY_LARGE_SIZE);
    this->readColumnFully();
    ASSERT_EQ(VERY_LARGE_SIZE, this->valuesRead_);
    this->values_.resize(VERY_LARGE_SIZE);
    ASSERT_EQ(this->values_, this->valuesOut_);
    std::vector<Encoding::type> encodingsVector = this->metadataEncodings();
    std::set<Encoding::type> encodings(
        encodingsVector.cbegin(), encodingsVector.cend());

    if (this->typeNum() == Type::kBoolean) {
      // Dictionary encoding is not allowed for boolean type.
      // There are 2 encodings (PLAIN, RLE) in a non dictionary encoding case.
      std::set<Encoding::type> expected({Encoding::kPlain, Encoding::kRle});
      ASSERT_EQ(encodings, expected);
    } else if (version == ParquetVersion::PARQUET_1_0) {
      // There are 3 encodings (PLAIN_DICTIONARY, PLAIN, RLE) in a fallback
      // case. For version 1.0.
      std::set<Encoding::type> expected(
          {Encoding::kPlainDictionary, Encoding::kPlain, Encoding::kRle});
      ASSERT_EQ(encodings, expected);
    } else {
      // There are 3 encodings (RLE_DICTIONARY, PLAIN, RLE) in a fallback case.
      // For version 2.0.
      std::set<Encoding::type> expected(
          {Encoding::kRleDictionary, Encoding::kPlain, Encoding::kRle});
      ASSERT_EQ(encodings, expected);
    }

    std::vector<PageEncodingStats> encodingStats =
        this->metadataEncodingStats();
    if (this->typeNum() == Type::kBoolean) {
      ASSERT_EQ(encodingStats[0].encoding, Encoding::kPlain);
      ASSERT_EQ(encodingStats[0].pageType, PageType::kDataPage);
    } else if (version == ParquetVersion::PARQUET_1_0) {
      std::vector<Encoding::type> expected(
          {Encoding::kPlainDictionary,
           Encoding::kPlain,
           Encoding::kPlainDictionary});
      ASSERT_EQ(encodingStats[0].encoding, expected[0]);
      ASSERT_EQ(encodingStats[0].pageType, PageType::kDictionaryPage);
      for (size_t i = 1; i < encodingStats.size(); i++) {
        ASSERT_EQ(encodingStats[i].encoding, expected[i]);
        ASSERT_EQ(encodingStats[i].pageType, PageType::kDataPage);
      }
    } else {
      std::vector<Encoding::type> expected(
          {Encoding::kPlain, Encoding::kPlain, Encoding::kRleDictionary});
      ASSERT_EQ(encodingStats[0].encoding, expected[0]);
      ASSERT_EQ(encodingStats[0].pageType, PageType::kDictionaryPage);
      for (size_t i = 1; i < encodingStats.size(); i++) {
        ASSERT_EQ(encodingStats[i].encoding, expected[i]);
        ASSERT_EQ(encodingStats[i].pageType, PageType::kDataPage);
      }
    }
  }

  void writeRequiredWithSettings(
      Encoding::type encoding,
      Compression::type compression,
      bool enableDictionary,
      bool enableStatistics,
      int compressionLevel,
      int64_t numRows,
      bool enableChecksum) {
    ColumnProperties columnProperties(
        encoding, compression, enableDictionary, enableStatistics);
    columnProperties.setCompressionLevel(compressionLevel);
    std::shared_ptr<TypedColumnWriter<TestType>> writer = this->buildWriter(
        numRows, columnProperties, ParquetVersion::PARQUET_1_0, enableChecksum);
    writer->writeBatch(
        this->values_.size(), nullptr, nullptr, this->valuesPtr_);
    // The behaviour should be independent from the number of Close() calls.
    writer->close();
    writer->close();
  }

  void writeRequiredWithSettingsSpaced(
      Encoding::type encoding,
      Compression::type compression,
      bool enableDictionary,
      bool enableStatistics,
      int64_t numRows,
      int compressionLevel,
      bool enableChecksum) {
    std::vector<uint8_t> validBits(
        ::arrow::bit_util::BytesForBits(
            static_cast<uint32_t>(this->values_.size())) +
            1,
        255);
    ColumnProperties columnProperties(
        encoding, compression, enableDictionary, enableStatistics);
    columnProperties.setCompressionLevel(compressionLevel);
    std::shared_ptr<TypedColumnWriter<TestType>> writer = this->buildWriter(
        numRows, columnProperties, ParquetVersion::PARQUET_1_0, enableChecksum);
    writer->writeBatchSpaced(
        this->values_.size(),
        nullptr,
        nullptr,
        validBits.data(),
        0,
        this->valuesPtr_);
    // The behaviour should be independent from the number of Close() calls.
    writer->close();
    writer->close();
  }

  void readAndCompare(
      Compression::type compression,
      int64_t numRows,
      bool pageChecksumVerify) {
    this->setupValuesOut(numRows);
    this->readColumnFully(compression, pageChecksumVerify);
    auto Comparator = makeComparator<TestType>(this->descr_);
    for (size_t i = 0; i < this->values_.size(); i++) {
      if (Comparator->compare(this->values_[i], this->valuesOut_[i]) ||
          Comparator->compare(this->valuesOut_[i], this->values_[i])) {
        ARROW_SCOPED_TRACE("i = ", i);
      }
      ASSERT_FALSE(Comparator->compare(this->values_[i], this->valuesOut_[i]));
      ASSERT_FALSE(Comparator->compare(this->valuesOut_[i], this->values_[i]));
    }
    ASSERT_EQ(this->values_, this->valuesOut_);
  }

  int64_t metadataNumValues() {
    // Metadata accessor must be created lazily.
    // This is because the ColumnChunkMetaData semantics dictate the metadata.
    // Object is complete (no changes to the metadata buffer can be made after.
    // instantiation)
    auto metadataAccessor =
        ColumnChunkMetaData::make(metadata_->Contents(), this->descr_);
    return metadataAccessor->numValues();
  }

  bool metadataIsStatsSet() {
    // Metadata accessor must be created lazily.
    // This is because the ColumnChunkMetaData semantics dictate the metadata.
    // Object is complete (no changes to the metadata buffer can be made after.
    // instantiation)
    ApplicationVersion appVersion(this->writerProperties_->createdBy());
    auto metadataAccessor = ColumnChunkMetaData::make(
        metadata_->Contents(),
        this->descr_,
        defaultReaderProperties(),
        &appVersion);
    return metadataAccessor->isStatsSet();
  }

  std::pair<bool, bool> metadataStatsHasMinMax() {
    // Metadata accessor must be created lazily.
    // This is because the ColumnChunkMetaData semantics dictate the metadata.
    // Object is complete (no changes to the metadata buffer can be made after.
    // instantiation)
    ApplicationVersion appVersion(this->writerProperties_->createdBy());
    auto metadataAccessor = ColumnChunkMetaData::make(
        metadata_->Contents(),
        this->descr_,
        defaultReaderProperties(),
        &appVersion);
    auto encodedStats = metadataAccessor->statistics()->encode();
    return {encodedStats.hasMin, encodedStats.hasMax};
  }

  std::vector<Encoding::type> metadataEncodings() {
    // Metadata accessor must be created lazily.
    // This is because the ColumnChunkMetaData semantics dictate the metadata.
    // Object is complete (no changes to the metadata buffer can be made after.
    // instantiation)
    auto metadataAccessor =
        ColumnChunkMetaData::make(metadata_->Contents(), this->descr_);
    return metadataAccessor->encodings();
  }

  std::vector<PageEncodingStats> metadataEncodingStats() {
    // Metadata accessor must be created lazily.
    // This is because the ColumnChunkMetaData semantics dictate the metadata.
    // Object is complete (no changes to the metadata buffer can be made after.
    // instantiation)
    auto metadataAccessor =
        ColumnChunkMetaData::make(metadata_->Contents(), this->descr_);
    return metadataAccessor->encodingStats();
  }

 protected:
  int64_t valuesRead_;
  // Keep the reader alive as for ByteArray the lifetime of the ByteArray.
  // Content is bound to the reader.
  std::shared_ptr<TypedColumnReader<TestType>> reader_;

  std::vector<int16_t> definitionLevelsOut_;
  std::vector<int16_t> repetitionLevelsOut_;

  const ColumnDescriptor* descr_;

 private:
  std::unique_ptr<ColumnChunkMetaDataBuilder> metadata_;
  std::shared_ptr<::arrow::io::BufferOutputStream> sink_;
  std::shared_ptr<WriterProperties> writerProperties_;
  std::vector<std::vector<uint8_t>> dataBuffer_;
};

template <typename TestType>
void TestPrimitiveWriter<TestType>::readColumnFully(
    Compression::type compression,
    bool pageChecksumVerify) {
  int64_t totalValues = static_cast<int64_t>(this->valuesOut_.size());
  buildReader(totalValues, compression, pageChecksumVerify);
  valuesRead_ = 0;
  while (valuesRead_ < totalValues) {
    int64_t valuesReadRecently = 0;
    reader_->readBatch(
        static_cast<int>(this->valuesOut_.size()) -
            static_cast<int>(valuesRead_),
        definitionLevelsOut_.data() + valuesRead_,
        repetitionLevelsOut_.data() + valuesRead_,
        this->valuesOutPtr_ + valuesRead_,
        &valuesReadRecently);
    valuesRead_ += valuesReadRecently;
  }
  this->syncValuesOut();
}

template <>
void TestPrimitiveWriter<Int96Type>::readAndCompare(
    Compression::type compression,
    int64_t numRows,
    bool pageChecksumVerify) {
  this->setupValuesOut(numRows);
  this->readColumnFully(compression, pageChecksumVerify);

  auto Comparator = makeComparator<Int96Type>(Type::kInt96, SortOrder::kSigned);
  for (size_t i = 0; i < this->values_.size(); i++) {
    if (Comparator->compare(this->values_[i], this->valuesOut_[i]) ||
        Comparator->compare(this->valuesOut_[i], this->values_[i])) {
      ARROW_SCOPED_TRACE("i = ", i);
    }
    ASSERT_FALSE(Comparator->compare(this->values_[i], this->valuesOut_[i]));
    ASSERT_FALSE(Comparator->compare(this->valuesOut_[i], this->values_[i]));
  }
  ASSERT_EQ(this->values_, this->valuesOut_);
}

template <>
void TestPrimitiveWriter<FLBAType>::readColumnFully(
    Compression::type compression,
    bool pageChecksumVerify) {
  int64_t totalValues = static_cast<int64_t>(this->valuesOut_.size());
  buildReader(totalValues, compression, pageChecksumVerify);
  this->dataBuffer_.clear();

  valuesRead_ = 0;
  while (valuesRead_ < totalValues) {
    int64_t valuesReadRecently = 0;
    reader_->readBatch(
        static_cast<int>(this->valuesOut_.size()) -
            static_cast<int>(valuesRead_),
        definitionLevelsOut_.data() + valuesRead_,
        repetitionLevelsOut_.data() + valuesRead_,
        this->valuesOutPtr_ + valuesRead_,
        &valuesReadRecently);

    // Copy contents of the pointers.
    std::vector<uint8_t> data(valuesReadRecently * this->descr_->typeLength());
    uint8_t* dataPtr = data.data();
    for (int64_t i = 0; i < valuesReadRecently; i++) {
      memcpy(
          dataPtr + this->descr_->typeLength() * i,
          this->valuesOut_[i + valuesRead_].ptr,
          this->descr_->typeLength());
      this->valuesOut_[i + valuesRead_].ptr =
          dataPtr + this->descr_->typeLength() * i;
    }
    dataBuffer_.emplace_back(std::move(data));

    valuesRead_ += valuesReadRecently;
  }
  this->syncValuesOut();
}

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

TYPED_TEST_SUITE(TestPrimitiveWriter, TestTypes);

using TestValuesWriterInt32Type = TestPrimitiveWriter<Int32Type>;
using TestValuesWriterInt64Type = TestPrimitiveWriter<Int64Type>;
using TestByteArrayValuesWriter = TestPrimitiveWriter<ByteArrayType>;
using TestFixedLengthByteArrayValuesWriter = TestPrimitiveWriter<FLBAType>;
using TestTimestampValuesWriter = TestPrimitiveWriter<Int96Type>;

TYPED_TEST(TestPrimitiveWriter, RequiredPlain) {
  this->testRequiredWithEncoding(Encoding::kPlain);
}

TYPED_TEST(TestPrimitiveWriter, RequiredDictionary) {
  this->testRequiredWithEncoding(Encoding::kPlainDictionary);
}

/*
TYPED_TEST(TestPrimitiveWriter, RequiredRLE) {
  this->TestRequiredWithEncoding(Encoding::RLE);
}

TYPED_TEST(TestPrimitiveWriter, RequiredBitPacked) {
  this->TestRequiredWithEncoding(Encoding::BIT_PACKED);
}
*/

TEST_F(TestValuesWriterInt32Type, RequiredDeltaBinaryPacked) {
  this->testRequiredWithEncoding(Encoding::kDeltaBinaryPacked);
}

TEST_F(TestValuesWriterInt64Type, RequiredDeltaBinaryPacked) {
  this->testRequiredWithEncoding(Encoding::kDeltaBinaryPacked);
}

TEST_F(TestByteArrayValuesWriter, RequiredDeltaLengthByteArray) {
  this->testRequiredWithEncoding(Encoding::kDeltaLengthByteArray);
}

/*
TYPED_TEST(TestByteArrayValuesWriter, RequiredDeltaByteArray) {
  this->TestRequiredWithEncoding(Encoding::DELTA_BYTE_ARRAY);
}

TEST_F(TestFixedLengthByteArrayValuesWriter, RequiredDeltaByteArray) {
  this->TestRequiredWithEncoding(Encoding::DELTA_BYTE_ARRAY);
}
*/

TYPED_TEST(TestPrimitiveWriter, RequiredRLEDictionary) {
  this->testRequiredWithEncoding(Encoding::kRleDictionary);
}

TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithStats) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::UNCOMPRESSED, false, true, LARGE_SIZE);
}

TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithSnappyCompression) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::SNAPPY, false, false, LARGE_SIZE);
}

TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithStatsAndSnappyCompression) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::SNAPPY, false, true, LARGE_SIZE);
}

#ifdef ARROW_WITH_BROTLI
TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithBrotliCompression) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::BROTLI, false, false, LARGE_SIZE);
}

TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithBrotliCompressionAndLevel) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::BROTLI, false, false, LARGE_SIZE, 10);
}

TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithStatsAndBrotliCompression) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::BROTLI, false, true, LARGE_SIZE);
}

#endif

TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithGzipCompression) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::GZIP, false, false, LARGE_SIZE);
}

TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithGzipCompressionAndLevel) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::GZIP, false, false, LARGE_SIZE, 10);
}

TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithStatsAndGzipCompression) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::GZIP, false, true, LARGE_SIZE);
}

TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithLz4Compression) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::LZ4, false, false, LARGE_SIZE);
}

TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithStatsAndLz4Compression) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::LZ4, false, true, LARGE_SIZE);
}

TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithZstdCompression) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::ZSTD, false, false, LARGE_SIZE);
}

TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithZstdCompressionAndLevel) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::ZSTD, false, false, LARGE_SIZE, 6);
}

TYPED_TEST(TestPrimitiveWriter, RequiredPlainWithStatsAndZstdCompression) {
  this->testRequiredWithSettings(
      Encoding::kPlain, Compression::ZSTD, false, true, LARGE_SIZE);
}

TYPED_TEST(TestPrimitiveWriter, Optional) {
  // Optional and non-repeated, with definition levels.
  // But no repetition levels.
  this->setUpSchema(Repetition::kOptional);

  this->generateData(SMALL_SIZE);
  std::vector<int16_t> definitionLevels(SMALL_SIZE, 1);
  definitionLevels[1] = 0;

  auto writer = this->buildWriter();
  writer->writeBatch(
      this->values_.size(), definitionLevels.data(), nullptr, this->valuesPtr_);
  writer->close();

  // PARQUET-703.
  ASSERT_EQ(100, this->metadataNumValues());

  this->readColumn();
  ASSERT_EQ(99, this->valuesRead_);
  this->valuesOut_.resize(99);
  this->values_.resize(99);
  ASSERT_EQ(this->values_, this->valuesOut_);
}

TYPED_TEST(TestPrimitiveWriter, OptionalSpaced) {
  // Optional and non-repeated, with definition levels.
  // But no repetition levels.
  this->setUpSchema(Repetition::kOptional);

  this->generateData(SMALL_SIZE);
  std::vector<int16_t> definitionLevels(SMALL_SIZE, 1);
  std::vector<uint8_t> validBits(
      ::arrow::bit_util::BytesForBits(SMALL_SIZE), 255);

  definitionLevels[SMALL_SIZE - 1] = 0;
  ::arrow::bit_util::ClearBit(validBits.data(), SMALL_SIZE - 1);
  definitionLevels[1] = 0;
  ::arrow::bit_util::ClearBit(validBits.data(), 1);

  auto writer = this->buildWriter();
  writer->writeBatchSpaced(
      this->values_.size(),
      definitionLevels.data(),
      nullptr,
      validBits.data(),
      0,
      this->valuesPtr_);
  writer->close();

  // PARQUET-703.
  ASSERT_EQ(100, this->metadataNumValues());

  this->readColumn();
  ASSERT_EQ(98, this->valuesRead_);
  this->valuesOut_.resize(98);
  this->values_.resize(99);
  this->values_.erase(this->values_.begin() + 1);
  ASSERT_EQ(this->values_, this->valuesOut_);
}

TYPED_TEST(TestPrimitiveWriter, Repeated) {
  // Optional and repeated, so definition and repetition levels.
  this->setUpSchema(Repetition::kRepeated);

  this->generateData(SMALL_SIZE);
  std::vector<int16_t> definitionLevels(SMALL_SIZE, 1);
  definitionLevels[1] = 0;
  std::vector<int16_t> repetitionLevels(SMALL_SIZE, 0);

  auto writer = this->buildWriter();
  writer->writeBatch(
      this->values_.size(),
      definitionLevels.data(),
      repetitionLevels.data(),
      this->valuesPtr_);
  writer->close();

  this->readColumn();
  ASSERT_EQ(SMALL_SIZE - 1, this->valuesRead_);
  this->valuesOut_.resize(SMALL_SIZE - 1);
  this->values_.resize(SMALL_SIZE - 1);
  ASSERT_EQ(this->values_, this->valuesOut_);
}

TYPED_TEST(TestPrimitiveWriter, RequiredLargeChunk) {
  this->generateData(LARGE_SIZE);

  // Test case 1: required and non-repeated, so no definition or repetition.
  // Levels.
  auto writer = this->buildWriter(LARGE_SIZE);
  writer->writeBatch(this->values_.size(), nullptr, nullptr, this->valuesPtr_);
  writer->close();

  // Just read the first SMALL_SIZE rows to ensure we could read it back in.
  this->readColumn();
  ASSERT_EQ(SMALL_SIZE, this->valuesRead_);
  this->values_.resize(SMALL_SIZE);
  ASSERT_EQ(this->values_, this->valuesOut_);
}

// Test cases for dictionary fallback encoding.
TYPED_TEST(TestPrimitiveWriter, dictionaryfallbackversion10) {
  this->testDictionaryFallbackEncoding(ParquetVersion::PARQUET_1_0);
}

TYPED_TEST(TestPrimitiveWriter, dictionaryfallbackversion20) {
  this->testDictionaryFallbackEncoding(ParquetVersion::PARQUET_2_4);
  this->testDictionaryFallbackEncoding(ParquetVersion::PARQUET_2_6);
}

TEST(TestWriter, NullValuesBuffer) {
  std::shared_ptr<::arrow::io::BufferOutputStream> sink = createOutputStream();

  const auto itemNode = schema::PrimitiveNode::make(
      "item",
      Repetition::kRequired,
      LogicalType::intType(32, true),
      Type::kInt32);
  const auto listNode =
      schema::GroupNode::make("list", Repetition::kRepeated, {itemNode});
  const auto columnNode = schema::GroupNode::make(
      "array_of_ints_column",
      Repetition::kOptional,
      {listNode},
      LogicalType::list());
  const auto schemaNode =
      schema::GroupNode::make("schema", Repetition::kRequired, {columnNode});

  auto fileWriter = ParquetFileWriter::open(
      sink, std::dynamic_pointer_cast<schema::GroupNode>(schemaNode));
  auto groupWriter = fileWriter->appendRowGroup();
  auto columnWriter = groupWriter->nextColumn();
  auto typedWriter = dynamic_cast<Int32Writer*>(columnWriter);

  const int64_t numValues = 1;
  const int16_t defLevels[] = {0};
  const int16_t repLevels[] = {0};
  const uint8_t validBits[] = {0};
  const int64_t validBitsOffset = 0;
  const int32_t* values = nullptr;

  typedWriter->writeBatchSpaced(
      numValues, defLevels, repLevels, validBits, validBitsOffset, values);
}

TYPED_TEST(TestPrimitiveWriter, RequiredPlainChecksum) {
  this->testRequiredWithSettings(
      Encoding::kPlain,
      Compression::UNCOMPRESSED,
      /* enable_dictionary */ false,
      false,
      SMALL_SIZE,
      Codec::useDefaultCompressionLevel(),
      /* enable_checksum */ true);
}

TYPED_TEST(TestPrimitiveWriter, RequiredDictChecksum) {
  this->testRequiredWithSettings(
      Encoding::kPlain,
      Compression::UNCOMPRESSED,
      /* enable_dictionary */ true,
      false,
      SMALL_SIZE,
      Codec::useDefaultCompressionLevel(),
      /* enable_checksum */ true);
}

// PARQUET-719.
// Test case for NULL values.
TEST_F(TestValuesWriterInt32Type, OptionalNullValueChunk) {
  this->setUpSchema(Repetition::kOptional);

  this->generateData(LARGE_SIZE);

  std::vector<int16_t> definitionLevels(LARGE_SIZE, 0);
  std::vector<int16_t> repetitionLevels(LARGE_SIZE, 0);

  auto writer = this->buildWriter(LARGE_SIZE);
  // All values being written are NULL.
  writer->writeBatch(
      this->values_.size(),
      definitionLevels.data(),
      repetitionLevels.data(),
      nullptr);
  writer->close();

  // Just read the first SMALL_SIZE rows to ensure we could read it back in.
  this->readColumn();
  ASSERT_EQ(0, this->valuesRead_);
}

// PARQUET-764.
// Correct bitpacking for boolean write at non-byte boundaries.
using TestBooleanValuesWriter = TestPrimitiveWriter<BooleanType>;
TEST_F(TestBooleanValuesWriter, AlternateBooleanValues) {
  this->setUpSchema(Repetition::kRequired);
  auto writer = this->buildWriter();
  for (int i = 0; i < SMALL_SIZE; i++) {
    bool value = (i % 2 == 0) ? true : false;
    writer->writeBatch(1, nullptr, nullptr, &value);
  }
  writer->close();
  this->readColumn();
  for (int i = 0; i < SMALL_SIZE; i++) {
    ASSERT_EQ((i % 2 == 0) ? true : false, this->valuesOut_[i]) << i;
  }
}

// PARQUET-979.
// Prevent writing large MIN, MAX stats.
TEST_F(TestByteArrayValuesWriter, OmitStats) {
  int minLen = 1024 * 4;
  int maxLen = 1024 * 8;
  this->setUpSchema(Repetition::kRequired);
  auto writer = this->buildWriter();

  values_.resize(SMALL_SIZE);
  initWideByteArrayValues(
      SMALL_SIZE, this->values_, this->buffer_, minLen, maxLen);
  writer->writeBatch(SMALL_SIZE, nullptr, nullptr, this->values_.data());
  writer->close();

  auto hasMinMax = this->metadataStatsHasMinMax();
  ASSERT_FALSE(hasMinMax.first);
  ASSERT_FALSE(hasMinMax.second);
}

// PARQUET-1405.
// Prevent writing large stats in the DataPageHeader.
TEST_F(TestByteArrayValuesWriter, OmitDataPageStats) {
  int minLen = static_cast<int>(std::pow(10, 7));
  int maxLen = static_cast<int>(std::pow(10, 7));
  this->setUpSchema(Repetition::kRequired);
  ColumnProperties columnProperties;
  columnProperties.setStatisticsEnabled(false);
  auto writer = this->buildWriter(SMALL_SIZE, columnProperties);

  values_.resize(1);
  initWideByteArrayValues(1, this->values_, this->buffer_, minLen, maxLen);
  writer->writeBatch(1, nullptr, nullptr, this->values_.data());
  writer->close();

  ASSERT_NO_THROW(this->readColumn());
}

TEST_F(TestByteArrayValuesWriter, LimitStats) {
  int minLen = 1024 * 4;
  int maxLen = 1024 * 8;
  this->setUpSchema(Repetition::kRequired);
  ColumnProperties columnProperties;
  columnProperties.setMaxStatisticsSize(static_cast<size_t>(maxLen));
  auto writer = this->buildWriter(SMALL_SIZE, columnProperties);

  values_.resize(SMALL_SIZE);
  initWideByteArrayValues(
      SMALL_SIZE, this->values_, this->buffer_, minLen, maxLen);
  writer->writeBatch(SMALL_SIZE, nullptr, nullptr, this->values_.data());
  writer->close();

  ASSERT_TRUE(this->metadataIsStatsSet());
}

TEST_F(TestByteArrayValuesWriter, CheckDefaultStats) {
  this->setUpSchema(Repetition::kRequired);
  auto writer = this->buildWriter();
  this->generateData(SMALL_SIZE);

  writer->writeBatch(SMALL_SIZE, nullptr, nullptr, this->valuesPtr_);
  writer->close();

  ASSERT_TRUE(this->metadataIsStatsSet());
}

TEST_F(TestTimestampValuesWriter, TimestampConventions) {
  auto testTimestamp =
      [](int64_t seconds, int32_t dayDelta, uint64_t nanoseconds) {
        Int96 actual;
        arrow::internal::secondsToImpalaTimestamp(seconds, &actual);
        auto julianDay = kJulianEpochOffsetDays + dayDelta;
#if ARROW_LITTLE_ENDIAN
        Int96 expected = {
            {static_cast<uint32_t>(nanoseconds),
             static_cast<uint32_t>(nanoseconds >> 32),
             static_cast<uint32_t>(julianDay)}};
#else
        Int96 expected = {
            {static_cast<uint32_t>(nanoseconds >> 32),
             static_cast<uint32_t>(nanoseconds),
             static_cast<uint32_t>(julianDay)}};
#endif
        EXPECT_EQ(actual, expected);
      };

  // Positive timestamps.
  testTimestamp(0, 0, 0);
  testTimestamp(1, 0, 1'000'000'000);
  testTimestamp(3600, 0, 3'600'000'000'000);
  testTimestamp(86400, 1, 0);
  testTimestamp(1767229259, 20454, 3'659'000'000'000);

  // Negative timestamps.
  testTimestamp(-1, -1, 86'399'000'000'000);
  testTimestamp(-3600, -1, 23LL * 3600 * 1'000'000'000);
  testTimestamp(-86400, -1, 0);
  testTimestamp(-86401, -2, 86'399'000'000'000);
}

TEST(TestColumnWriter, RepeatedListsUpdateSpacedBug) {
  // In ARROW-3930 we discovered a bug when writing from Arrow when we had data.
  // That looks like this:
  //
  // [Null, [0, 1, null, 2, 3, 4, null]].

  // Create schema.
  NodePtr item = schema::int32("item"); // optional item
  NodePtr List(
      GroupNode::make(
          "b", Repetition::kRepeated, {item}, ConvertedType::kList));
  NodePtr bag(
      GroupNode::make("bag", Repetition::kOptional, {List})); // optional list
  std::vector<NodePtr> fields = {bag};
  NodePtr root = GroupNode::make("schema", Repetition::kRepeated, fields);

  SchemaDescriptor schema;
  schema.init(root);

  auto sink = createOutputStream();
  auto props = WriterProperties::Builder().build();

  auto metadata = ColumnChunkMetaDataBuilder::make(props, schema.column(0));
  std::unique_ptr<PageWriter> pager = PageWriter::open(
      sink,
      Compression::UNCOMPRESSED,
      Codec::useDefaultCompressionLevel(),
      metadata.get());
  std::shared_ptr<ColumnWriter> writer =
      ColumnWriter::make(metadata.get(), std::move(pager), props.get());
  auto typedWriter =
      std::static_pointer_cast<TypedColumnWriter<Int32Type>>(writer);

  std::vector<int16_t> defLevels = {1, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3};
  std::vector<int16_t> repLevels = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Write the values into uninitialized memory.
  ASSERT_OK_AND_ASSIGN(auto valuesBuffer, ::arrow::AllocateBuffer(64));
  memcpy(valuesBuffer->mutable_data(), values.data(), 13 * sizeof(int32_t));
  auto valuesData = reinterpret_cast<const int32_t*>(valuesBuffer->data());

  std::shared_ptr<Buffer> validBits;
  ASSERT_OK_AND_ASSIGN(
      validBits,
      ::arrow::internal::BytesToBits({1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1}));

  // Valgrind will warn about out of bounds access into def_levels_data.
  typedWriter->writeBatchSpaced(
      14, defLevels.data(), repLevels.data(), validBits->data(), 0, valuesData);
  writer->close();
}

void generateLevels(
    int minRepeatFactor,
    int maxRepeatFactor,
    int maxLevel,
    std::vector<int16_t>& inputLevels) {
  // For each repetition count up to max_repeat_factor.
  for (int repeat = minRepeatFactor; repeat <= maxRepeatFactor; repeat++) {
    // Repeat count increases by a factor of 2 for every iteration.
    int repeatCount = (1 << repeat);
    // Generate levels for repetition count up to the maximum level.
    int16_t value = 0;
    int bwidth = 0;
    while (value <= maxLevel) {
      for (int i = 0; i < repeatCount; i++) {
        inputLevels.push_back(value);
      }
      value = static_cast<int16_t>((2 << bwidth) - 1);
      bwidth++;
    }
  }
}

void encodeLevels(
    Encoding::type encoding,
    int16_t maxLevel,
    int numLevels,
    const int16_t* inputLevels,
    std::vector<uint8_t>& bytes) {
  LevelEncoder encoder;
  int levelsCount = 0;
  bytes.resize(2 * numLevels);
  ASSERT_EQ(2 * numLevels, static_cast<int>(bytes.size()));
  // Encode levels.
  if (encoding == Encoding::kRle) {
    // Leave space to write the rle length value.
    encoder.init(
        encoding,
        maxLevel,
        numLevels,
        bytes.data() + sizeof(int32_t),
        static_cast<int>(bytes.size()));

    levelsCount = encoder.encode(numLevels, inputLevels);
    (reinterpret_cast<int32_t*>(bytes.data()))[0] = encoder.len();
  } else {
    encoder.init(
        encoding,
        maxLevel,
        numLevels,
        bytes.data(),
        static_cast<int>(bytes.size()));
    levelsCount = encoder.encode(numLevels, inputLevels);
  }
  ASSERT_EQ(numLevels, levelsCount);
}

void verifyDecodingLevels(
    Encoding::type encoding,
    int16_t maxLevel,
    std::vector<int16_t>& inputLevels,
    std::vector<uint8_t>& bytes) {
  LevelDecoder decoder;
  int levelsCount = 0;
  std::vector<int16_t> outputLevels;
  int numLevels = static_cast<int>(inputLevels.size());

  outputLevels.resize(numLevels);
  ASSERT_EQ(numLevels, static_cast<int>(outputLevels.size()));

  // Decode levels and test with multiple decode calls.
  decoder.setData(
      encoding,
      maxLevel,
      numLevels,
      bytes.data(),
      static_cast<int32_t>(bytes.size()));
  int decodeCount = 4;
  int numInnerLevels = numLevels / decodeCount;
  // Try multiple decoding on a single SetData call.
  for (int ct = 0; ct < decodeCount; ct++) {
    int Offset = ct * numInnerLevels;
    levelsCount = decoder.decode(numInnerLevels, outputLevels.data());
    ASSERT_EQ(numInnerLevels, levelsCount);
    for (int i = 0; i < numInnerLevels; i++) {
      EXPECT_EQ(inputLevels[i + Offset], outputLevels[i]);
    }
  }
  // Check the remaining levels.
  int numLevelsCompleted = decodeCount * (numLevels / decodeCount);
  int numRemainingLevels = numLevels - numLevelsCompleted;
  if (numRemainingLevels > 0) {
    levelsCount = decoder.decode(numRemainingLevels, outputLevels.data());
    ASSERT_EQ(numRemainingLevels, levelsCount);
    for (int i = 0; i < numRemainingLevels; i++) {
      EXPECT_EQ(inputLevels[i + numLevelsCompleted], outputLevels[i]);
    }
  }
  // Test zero Decode values.
  ASSERT_EQ(0, decoder.decode(1, outputLevels.data()));
}

void verifyDecodingMultipleSetData(
    Encoding::type encoding,
    int16_t maxLevel,
    std::vector<int16_t>& inputLevels,
    std::vector<std::vector<uint8_t>>& bytes) {
  LevelDecoder decoder;
  int levelsCount = 0;
  std::vector<int16_t> outputLevels;

  // Decode levels and test with multiple SetData calls.
  int setdataCount = static_cast<int>(bytes.size());
  int numLevels = static_cast<int>(inputLevels.size()) / setdataCount;
  outputLevels.resize(numLevels);
  // Try multiple SetData.
  for (int ct = 0; ct < setdataCount; ct++) {
    int Offset = ct * numLevels;
    ASSERT_EQ(numLevels, static_cast<int>(outputLevels.size()));
    decoder.setData(
        encoding,
        maxLevel,
        numLevels,
        bytes[ct].data(),
        static_cast<int32_t>(bytes[ct].size()));
    levelsCount = decoder.decode(numLevels, outputLevels.data());
    ASSERT_EQ(numLevels, levelsCount);
    for (int i = 0; i < numLevels; i++) {
      EXPECT_EQ(inputLevels[i + Offset], outputLevels[i]);
    }
  }
}

// Test levels with maximum bit-width from 1 to 8.
// Increase the repetition count for each iteration by a factor of 2.
TEST(TestLevels, TestLevelsDecodeMultipleBitWidth) {
  int minRepeatFactor = 0;
  int maxRepeatFactor = 7; // 128
  int maxBitWidth = 8;
  std::vector<int16_t> inputLevels;
  std::vector<uint8_t> bytes;
  Encoding::type encodings[2] = {Encoding::kRle, Encoding::kBitPacked};

  // For each encoding.
  for (int encode = 0; encode < 2; encode++) {
    Encoding::type encoding = encodings[encode];
    // BIT_PACKED requires a sequence of at least 8.
    if (encoding == Encoding::kBitPacked)
      minRepeatFactor = 3;
    // For each maximum bit-width.
    for (int bitWidth = 1; bitWidth <= maxBitWidth; bitWidth++) {
      // Find the maximum level for the current bit_width.
      int16_t maxLevel = static_cast<int16_t>((1 << bitWidth) - 1);
      // Generate levels.
      generateLevels(minRepeatFactor, maxRepeatFactor, maxLevel, inputLevels);
      ASSERT_NO_FATAL_FAILURE(encodeLevels(
          encoding,
          maxLevel,
          static_cast<int>(inputLevels.size()),
          inputLevels.data(),
          bytes));
      ASSERT_NO_FATAL_FAILURE(
          verifyDecodingLevels(encoding, maxLevel, inputLevels, bytes));
      inputLevels.clear();
    }
  }
}

// Test multiple decoder SetData calls.
TEST(TestLevels, TestLevelsDecodeMultipleSetData) {
  int minRepeatFactor = 3;
  int maxRepeatFactor = 7; // 128
  int bitWidth = 8;
  int16_t maxLevel = static_cast<int16_t>((1 << bitWidth) - 1);
  std::vector<int16_t> inputLevels;
  std::vector<std::vector<uint8_t>> bytes;
  Encoding::type encodings[2] = {Encoding::kRle, Encoding::kBitPacked};
  generateLevels(minRepeatFactor, maxRepeatFactor, maxLevel, inputLevels);
  int numLevels = static_cast<int>(inputLevels.size());
  int setdataFactor = 8;
  int splitLevelSize = numLevels / setdataFactor;
  bytes.resize(setdataFactor);

  // For each encoding.
  for (int encode = 0; encode < 2; encode++) {
    Encoding::type encoding = encodings[encode];
    for (int rf = 0; rf < setdataFactor; rf++) {
      int Offset = rf * splitLevelSize;
      ASSERT_NO_FATAL_FAILURE(encodeLevels(
          encoding,
          maxLevel,
          splitLevelSize,
          reinterpret_cast<int16_t*>(inputLevels.data()) + Offset,
          bytes[rf]));
    }
    ASSERT_NO_FATAL_FAILURE(
        verifyDecodingMultipleSetData(encoding, maxLevel, inputLevels, bytes));
  }
}

TEST(TestLevelEncoder, MinimumBufferSize) {
  // PARQUET-676, PARQUET-698.
  const int kNumToEncode = 1024;

  std::vector<int16_t> levels;
  for (int i = 0; i < kNumToEncode; ++i) {
    if (i % 9 == 0) {
      levels.push_back(0);
    } else {
      levels.push_back(1);
    }
  }

  std::vector<uint8_t> output(
      LevelEncoder::maxBufferSize(Encoding::kRle, 1, kNumToEncode));

  LevelEncoder encoder;
  encoder.init(
      Encoding::kRle,
      1,
      kNumToEncode,
      output.data(),
      static_cast<int>(output.size()));
  int encodeCount = encoder.encode(kNumToEncode, levels.data());

  ASSERT_EQ(kNumToEncode, encodeCount);
}

TEST(TestLevelEncoder, MinimumBufferSize2) {
  // PARQUET-708.
  // Test the worst case for bit_width=2 consisting of.
  // LiteralRun(size=8)
  // RepeatedRun(size=8)
  // LiteralRun(size=8)
  // ...
  const int kNumToEncode = 1024;

  std::vector<int16_t> levels;
  for (int i = 0; i < kNumToEncode; ++i) {
    // This forces a literal run of 00000001.
    // Followed by eight 1s.
    if ((i % 16) < 7) {
      levels.push_back(0);
    } else {
      levels.push_back(1);
    }
  }

  for (int16_t bitWidth = 1; bitWidth <= 8; bitWidth++) {
    std::vector<uint8_t> output(
        LevelEncoder::maxBufferSize(Encoding::kRle, bitWidth, kNumToEncode));

    LevelEncoder encoder;
    encoder.init(
        Encoding::kRle,
        bitWidth,
        kNumToEncode,
        output.data(),
        static_cast<int>(output.size()));
    int encodeCount = encoder.encode(kNumToEncode, levels.data());

    ASSERT_EQ(kNumToEncode, encodeCount);
  }
}

TEST(TestColumnWriter, WriteDataPageV2Header) {
  auto sink = createOutputStream();
  auto schema = std::static_pointer_cast<GroupNode>(GroupNode::make(
      "schema",
      Repetition::kRequired,
      {
          schema::int32("required", Repetition::kRequired),
          schema::int32("optional", Repetition::kOptional),
          schema::int32("repeated", Repetition::kRepeated),
      }));
  auto properties = WriterProperties::Builder()
                        .disableDictionary()
                        ->dataPageVersion(ParquetDataPageVersion::V2)
                        ->build();
  auto fileWriter = ParquetFileWriter::open(sink, schema, properties);
  auto rgWriter = fileWriter->appendRowGroup();

  constexpr int32_t numRows = 100;

  auto requiredWriter = static_cast<Int32Writer*>(rgWriter->nextColumn());
  for (int32_t i = 0; i < numRows; i++) {
    requiredWriter->writeBatch(1, nullptr, nullptr, &i);
  }

  // Write a null value at every other row.
  auto optionalWriter = static_cast<Int32Writer*>(rgWriter->nextColumn());
  for (int32_t i = 0; i < numRows; i++) {
    int16_t definitionLevel = i % 2 == 0 ? 1 : 0;
    optionalWriter->writeBatch(1, &definitionLevel, nullptr, &i);
  }

  // Each row has repeated twice.
  auto repeatedWriter = static_cast<Int32Writer*>(rgWriter->nextColumn());
  for (int i = 0; i < 2 * numRows; i++) {
    int32_t value = i * 1000;
    int16_t definitionLevel = 1;
    int16_t repetitionLevel = i % 2 == 0 ? 1 : 0;
    repeatedWriter->writeBatch(1, &definitionLevel, &repetitionLevel, &value);
  }

  ASSERT_NO_THROW(fileWriter->close());
  ASSERT_OK_AND_ASSIGN(auto buffer, sink->Finish());
  auto fileReader = ParquetFileReader::open(
      std::make_shared<::arrow::io::BufferReader>(buffer),
      defaultReaderProperties());
  auto metadata = fileReader->metadata();
  ASSERT_EQ(1, metadata->numRowGroups());
  auto rowGroupReader = fileReader->rowGroup(0);

  // Verify required column.
  {
    auto pageReader = rowGroupReader->getColumnPageReader(0);
    auto page = pageReader->nextPage();
    ASSERT_NE(page, nullptr);
    auto dataPage = std::static_pointer_cast<DataPageV2>(page);
    EXPECT_EQ(numRows, dataPage->numRows());
    EXPECT_EQ(numRows, dataPage->numValues());
    EXPECT_EQ(0, dataPage->numNulls());
    EXPECT_EQ(pageReader->nextPage(), nullptr);
  }

  // Verify optional column.
  {
    auto pageReader = rowGroupReader->getColumnPageReader(1);
    auto page = pageReader->nextPage();
    ASSERT_NE(page, nullptr);
    auto dataPage = std::static_pointer_cast<DataPageV2>(page);
    EXPECT_EQ(numRows, dataPage->numRows());
    EXPECT_EQ(numRows, dataPage->numValues());
    EXPECT_EQ(numRows / 2, dataPage->numNulls());
    EXPECT_EQ(pageReader->nextPage(), nullptr);
  }

  // Verify repeated column.
  {
    auto pageReader = rowGroupReader->getColumnPageReader(2);
    auto page = pageReader->nextPage();
    ASSERT_NE(page, nullptr);
    auto dataPage = std::static_pointer_cast<DataPageV2>(page);
    EXPECT_EQ(numRows, dataPage->numRows());
    EXPECT_EQ(numRows * 2, dataPage->numValues());
    EXPECT_EQ(0, dataPage->numNulls());
    EXPECT_EQ(pageReader->nextPage(), nullptr);
  }
}

// The test below checks that data page v2 changes on record boundaries for.
// all repetition types (i.e. required, optional, and repeated)
TEST(TestColumnWriter, WriteDataPagesChangeOnRecordBoundaries) {
  auto sink = createOutputStream();
  auto schema = std::static_pointer_cast<GroupNode>(GroupNode::make(
      "schema",
      Repetition::kRequired,
      {schema::int32("required", Repetition::kRequired),
       schema::int32("optional", Repetition::kOptional),
       schema::int32("repeated", Repetition::kRepeated)}));
  // Write at most 11 levels per batch.
  constexpr int64_t batchSize = 11;
  auto properties =
      WriterProperties::Builder()
          .disableDictionary()
          ->dataPageVersion(ParquetDataPageVersion::V2)
          ->writeBatchSize(batchSize)
          ->dataPagesize(1) /* every page size check creates a new page */
          ->build();
  auto fileWriter = ParquetFileWriter::open(sink, schema, properties);
  auto rgWriter = fileWriter->appendRowGroup();

  constexpr int32_t numLevels = 100;
  const std::vector<int32_t> values(numLevels, 1024);
  std::array<int16_t, numLevels> defLevels;
  std::array<int16_t, numLevels> repLevels;
  for (int32_t i = 0; i < numLevels; i++) {
    defLevels[i] = i % 2 == 0 ? 1 : 0;
    repLevels[i] = i % 2 == 0 ? 0 : 1;
  }

  auto requiredWriter = static_cast<Int32Writer*>(rgWriter->nextColumn());
  requiredWriter->writeBatch(numLevels, nullptr, nullptr, values.data());

  // Write a null value at every other row.
  auto optionalWriter = static_cast<Int32Writer*>(rgWriter->nextColumn());
  optionalWriter->writeBatch(
      numLevels, defLevels.data(), nullptr, values.data());

  // Each row has repeated twice.
  auto repeatedWriter = static_cast<Int32Writer*>(rgWriter->nextColumn());
  repeatedWriter->writeBatch(
      numLevels, defLevels.data(), repLevels.data(), values.data());
  repeatedWriter->writeBatch(
      numLevels, defLevels.data(), repLevels.data(), values.data());

  ASSERT_NO_THROW(fileWriter->close());
  ASSERT_OK_AND_ASSIGN(auto buffer, sink->Finish());
  auto fileReader = ParquetFileReader::open(
      std::make_shared<::arrow::io::BufferReader>(buffer),
      defaultReaderProperties());
  auto metadata = fileReader->metadata();
  ASSERT_EQ(1, metadata->numRowGroups());
  auto rowGroupReader = fileReader->rowGroup(0);

  // Check if pages are changed on record boundaries.
  constexpr int numColumns = 3;
  const std::array<int64_t, numColumns> expectedNumPages = {10, 10, 19};
  for (int i = 0; i < numColumns; ++i) {
    auto pageReader = rowGroupReader->getColumnPageReader(i);
    int64_t numRows = 0;
    int64_t numPages = 0;
    std::shared_ptr<Page> page;
    while ((page = pageReader->nextPage()) != nullptr) {
      auto dataPage = std::static_pointer_cast<DataPageV2>(page);
      if (i < 2) {
        EXPECT_EQ(dataPage->numValues(), dataPage->numRows());
      } else {
        // Make sure repeated column has 2 values per row and not span multiple.
        // Pages.
        EXPECT_EQ(dataPage->numValues(), 2 * dataPage->numRows());
      }
      numRows += dataPage->numRows();
      numPages++;
    }
    EXPECT_EQ(numLevels, numRows);
    EXPECT_EQ(expectedNumPages[i], numPages);
  }
}

// The test below checks that data page v2 changes on record boundaries for.
// Repeated columns with small batches.
TEST(TestColumnWriter, WriteDataPagesChangeOnRecordBoundariesWithSmallBatches) {
  auto sink = createOutputStream();
  auto schema = std::static_pointer_cast<GroupNode>(GroupNode::make(
      "schema",
      Repetition::kRequired,
      {schema::int32("tiny_repeat", Repetition::kRepeated),
       schema::int32("small_repeat", Repetition::kRepeated),
       schema::int32("medium_repeat", Repetition::kRepeated),
       schema::int32("large_repeat", Repetition::kRepeated)}));

  // The batch_size is large enough so each WriteBatch call checks page size at.
  // Most once.
  constexpr int64_t batchSize = std::numeric_limits<int64_t>::max();
  auto properties =
      WriterProperties::Builder()
          .disableDictionary()
          ->dataPageVersion(ParquetDataPageVersion::V2)
          ->writeBatchSize(batchSize)
          ->dataPagesize(1) /* every page size check creates a new page */
          ->build();
  auto fileWriter = ParquetFileWriter::open(sink, schema, properties);
  auto rgWriter = fileWriter->appendRowGroup();

  constexpr int32_t numCols = 4;
  constexpr int64_t numRows = 400;
  constexpr int64_t numLevels = 100;
  constexpr std::array<int64_t, numCols> numLevelsPerRowByCol = {
      1, 50, 99, 150};

  // All values are not null and fixed to 1024 for simplicity.
  const std::vector<int32_t> values(numLevels, 1024);
  const std::vector<int16_t> defLevels(numLevels, 1);
  std::vector<int16_t> repLevels(numLevels, 0);

  for (int32_t i = 0; i < numCols; ++i) {
    auto writer = static_cast<Int32Writer*>(rgWriter->nextColumn());
    const auto numLevelsPerRow = numLevelsPerRowByCol[i];
    int64_t numRowsWritten = 0;
    int64_t numLevelsWrittenCurrRow = 0;
    while (numRowsWritten < numRows) {
      int32_t numLevelsToWrite = 0;
      while (numLevelsToWrite < numLevels) {
        if (numLevelsWrittenCurrRow == 0) {
          // A new record.
          repLevels[numLevelsToWrite++] = 0;
        } else {
          repLevels[numLevelsToWrite++] = 1;
        }

        if (++numLevelsWrittenCurrRow == numLevelsPerRow) {
          // Current row has enough levels.
          numLevelsWrittenCurrRow = 0;
          if (++numRowsWritten == numRows) {
            // Enough rows have been written.
            break;
          }
        }
      }

      writer->writeBatch(
          numLevelsToWrite, defLevels.data(), repLevels.data(), values.data());
    }
  }

  ASSERT_NO_THROW(fileWriter->close());
  ASSERT_OK_AND_ASSIGN(auto buffer, sink->Finish());
  auto fileReader = ParquetFileReader::open(
      std::make_shared<::arrow::io::BufferReader>(buffer),
      defaultReaderProperties());
  auto metadata = fileReader->metadata();
  ASSERT_EQ(1, metadata->numRowGroups());
  auto rowGroupReader = fileReader->rowGroup(0);

  // Check if pages are changed on record boundaries.
  const std::array<int64_t, numCols> expectNumPagesByCol = {5, 201, 397, 201};
  const std::array<int64_t, numCols> expectNumRows1stPageByCol = {99, 1, 1, 1};
  const std::array<int64_t, numCols> expectNumVals1stPageByCol = {
      99, 50, 99, 150};
  for (int32_t i = 0; i < numCols; ++i) {
    auto pageReader = rowGroupReader->getColumnPageReader(i);
    int64_t numRowsRead = 0;
    int64_t numPagesRead = 0;
    int64_t numValuesRead = 0;
    std::shared_ptr<Page> page;
    while ((page = pageReader->nextPage()) != nullptr) {
      auto dataPage = std::static_pointer_cast<DataPageV2>(page);
      numValuesRead += dataPage->numValues();
      numRowsRead += dataPage->numRows();
      if (numPagesRead++ == 0) {
        EXPECT_EQ(expectNumRows1stPageByCol[i], dataPage->numRows());
        EXPECT_EQ(expectNumVals1stPageByCol[i], dataPage->numValues());
      }
    }
    EXPECT_EQ(numRows, numRowsRead);
    EXPECT_EQ(expectNumPagesByCol[i], numPagesRead);
    EXPECT_EQ(numLevelsPerRowByCol[i] * numRows, numValuesRead);
  }
}

class ColumnWriterTestSizeEstimated : public ::testing::Test {
 public:
  void SetUp() {
    sink_ = createOutputStream();
    node_ = std::static_pointer_cast<GroupNode>(GroupNode::make(
        "schema",
        Repetition::kRequired,
        {
            schema::int32("required", Repetition::kRequired),
        }));
    std::vector<schema::NodePtr> fields;
    schemaDescriptor_ = std::make_unique<SchemaDescriptor>();
    schemaDescriptor_->init(node_);
  }

  std::shared_ptr<Int32Writer> buildWriter(
      Compression::type compression,
      bool buffered,
      bool enableDictionary = false) {
    auto Builder = WriterProperties::Builder();
    Builder.disableDictionary()
        ->compression(compression)
        ->dataPagesize(100 * sizeof(int));
    if (enableDictionary) {
      Builder.enableDictionary();
    } else {
      Builder.disableDictionary();
    }
    writerProperties_ = Builder.build();
    metadata_ = ColumnChunkMetaDataBuilder::make(
        writerProperties_, schemaDescriptor_->column(0));

    std::unique_ptr<PageWriter> pager = PageWriter::open(
        sink_,
        compression,
        Codec::useDefaultCompressionLevel(),
        metadata_.get(),
        /* row_group_ordinal */ -1,
        /* column_chunk_ordinal*/ -1,
        ::arrow::default_memory_pool(),
        /* buffered_row_group */ buffered,
        /* header_encryptor */ NULLPTR,
        /* data_encryptor */ NULLPTR,
        /* enable_checksum */ false);
    return std::static_pointer_cast<Int32Writer>(ColumnWriter::make(
        metadata_.get(), std::move(pager), writerProperties_.get()));
  }

  std::shared_ptr<::arrow::io::BufferOutputStream> sink_;
  std::shared_ptr<GroupNode> node_;
  std::unique_ptr<SchemaDescriptor> schemaDescriptor_;

  std::shared_ptr<WriterProperties> writerProperties_;
  std::unique_ptr<ColumnChunkMetaDataBuilder> metadata_;
};

TEST_F(ColumnWriterTestSizeEstimated, NonBuffered) {
  auto requiredWriter =
      this->buildWriter(Compression::UNCOMPRESSED, /* buffered*/ false);
  // Write half page, page will not be flushed after loop.
  for (int32_t i = 0; i < 50; i++) {
    requiredWriter->writeBatch(1, nullptr, nullptr, &i);
  }
  // Page not flushed, check size.
  EXPECT_EQ(0, requiredWriter->totalBytesWritten());
  EXPECT_EQ(0, requiredWriter->totalCompressedBytes()); // unbuffered
  EXPECT_EQ(0, requiredWriter->totalCompressedBytesWritten());
  // Write half page, page be flushed after loop.
  for (int32_t i = 0; i < 50; i++) {
    requiredWriter->writeBatch(1, nullptr, nullptr, &i);
  }
  // Page flushed, check size.
  EXPECT_LT(400, requiredWriter->totalBytesWritten());
  EXPECT_EQ(0, requiredWriter->totalCompressedBytes());
  EXPECT_LT(400, requiredWriter->totalCompressedBytesWritten());

  // Test after closed.
  int64_t writtenSize = requiredWriter->close();
  EXPECT_EQ(0, requiredWriter->totalCompressedBytes());
  EXPECT_EQ(writtenSize, requiredWriter->totalBytesWritten());
  // Uncompressed writer should be equal.
  EXPECT_EQ(writtenSize, requiredWriter->totalCompressedBytesWritten());
}

TEST_F(ColumnWriterTestSizeEstimated, Buffered) {
  auto requiredWriter =
      this->buildWriter(Compression::UNCOMPRESSED, /* buffered*/ true);
  // Write half page, page will not be flushed after loop.
  for (int32_t i = 0; i < 50; i++) {
    requiredWriter->writeBatch(1, nullptr, nullptr, &i);
  }
  // Page not flushed, check size.
  EXPECT_EQ(0, requiredWriter->totalBytesWritten());
  EXPECT_EQ(0, requiredWriter->totalCompressedBytes()); // buffered
  EXPECT_EQ(0, requiredWriter->totalCompressedBytesWritten());
  // Write half page, page be flushed after loop.
  for (int32_t i = 0; i < 50; i++) {
    requiredWriter->writeBatch(1, nullptr, nullptr, &i);
  }
  // Page flushed, check size.
  EXPECT_LT(400, requiredWriter->totalBytesWritten());
  EXPECT_EQ(0, requiredWriter->totalCompressedBytes());
  EXPECT_LT(400, requiredWriter->totalCompressedBytesWritten());

  // Test after closed.
  int64_t writtenSize = requiredWriter->close();
  EXPECT_EQ(0, requiredWriter->totalCompressedBytes());
  EXPECT_EQ(writtenSize, requiredWriter->totalBytesWritten());
  // Uncompressed writer should be equal.
  EXPECT_EQ(writtenSize, requiredWriter->totalCompressedBytesWritten());
}

TEST_F(ColumnWriterTestSizeEstimated, NonBufferedDictionary) {
  auto requiredWriter =
      this->buildWriter(Compression::UNCOMPRESSED, /* buffered*/ false, true);
  // For dict, keep all values equal.
  int32_t dictValue = 1;
  for (int32_t i = 0; i < 50; i++) {
    requiredWriter->writeBatch(1, nullptr, nullptr, &dictValue);
  }
  // Page not flushed, check size.
  EXPECT_EQ(0, requiredWriter->totalBytesWritten());
  EXPECT_EQ(0, requiredWriter->totalCompressedBytes());
  EXPECT_EQ(0, requiredWriter->totalCompressedBytesWritten());
  // Write a huge batch to trigger page flush.
  for (int32_t i = 0; i < 50000; i++) {
    requiredWriter->writeBatch(1, nullptr, nullptr, &dictValue);
  }
  // Page flushed, check size.
  EXPECT_EQ(0, requiredWriter->totalBytesWritten());
  EXPECT_LT(400, requiredWriter->totalCompressedBytes());
  EXPECT_EQ(0, requiredWriter->totalCompressedBytesWritten());

  requiredWriter->close();

  // Test after closed.
  int64_t writtenSize = requiredWriter->close();
  EXPECT_EQ(0, requiredWriter->totalCompressedBytes());
  EXPECT_EQ(writtenSize, requiredWriter->totalBytesWritten());
  // Uncompressed writer should be equal.
  EXPECT_EQ(writtenSize, requiredWriter->totalCompressedBytesWritten());
}

TEST_F(ColumnWriterTestSizeEstimated, BufferedCompression) {
  auto requiredWriter = this->buildWriter(Compression::SNAPPY, true);

  // Write half page.
  for (int32_t i = 0; i < 50; i++) {
    requiredWriter->writeBatch(1, nullptr, nullptr, &i);
  }
  // Page not flushed, check size.
  EXPECT_EQ(0, requiredWriter->totalBytesWritten());
  EXPECT_EQ(0, requiredWriter->totalCompressedBytes()); // buffered
  EXPECT_EQ(0, requiredWriter->totalCompressedBytesWritten());
  for (int32_t i = 0; i < 50; i++) {
    requiredWriter->writeBatch(1, nullptr, nullptr, &i);
  }
  // Page flushed, check size.
  EXPECT_LT(400, requiredWriter->totalBytesWritten());
  EXPECT_EQ(0, requiredWriter->totalCompressedBytes());
  EXPECT_LT(
      requiredWriter->totalCompressedBytesWritten(),
      requiredWriter->totalBytesWritten());

  // Test after closed.
  int64_t writtenSize = requiredWriter->close();
  EXPECT_EQ(0, requiredWriter->totalCompressedBytes());
  EXPECT_EQ(writtenSize, requiredWriter->totalBytesWritten());
  EXPECT_GT(writtenSize, requiredWriter->totalCompressedBytesWritten());
}

TEST(TestColumnWriter, WriteDataPageV2HeaderNullCount) {
  auto sink = createOutputStream();
  auto listType = GroupNode::make(
      "list",
      Repetition::kRepeated,
      {schema::int32("elem", Repetition::kOptional)});
  auto schema = std::static_pointer_cast<GroupNode>(GroupNode::make(
      "schema",
      Repetition::kRequired,
      {
          schema::int32("non_null", Repetition::kOptional),
          schema::int32("half_null", Repetition::kOptional),
          schema::int32("all_null", Repetition::kOptional),
          GroupNode::make("half_null_list", Repetition::kOptional, {listType}),
          GroupNode::make("half_empty_list", Repetition::kOptional, {listType}),
          GroupNode::make(
              "half_list_of_null", Repetition::kOptional, {listType}),
          GroupNode::make("all_single_list", Repetition::kOptional, {listType}),
      }));
  auto properties = WriterProperties::Builder()
                        /* Use V2 data page to read null_count from header */
                        .dataPageVersion(ParquetDataPageVersion::V2)
                        /* Disable stats to test null_count is properly set */
                        ->disableStatistics()
                        ->disableDictionary()
                        ->build();
  auto fileWriter = ParquetFileWriter::open(sink, schema, properties);
  auto rgWriter = fileWriter->appendRowGroup();

  constexpr int32_t numRows = 10;
  constexpr int32_t numCols = 7;
  const std::vector<std::vector<int16_t>> defLevelsByCol = {
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 3, 0, 3, 0, 3, 0, 3, 0, 3},
      {1, 3, 1, 3, 1, 3, 1, 3, 1, 3},
      {2, 3, 2, 3, 2, 3, 2, 3, 2, 3},
      {3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
  };
  const std::vector<int16_t> refLevels(numRows, 0);
  const std::vector<int32_t> values(numRows, 123);
  const std::vector<int64_t> expectNullCountByCol = {0, 5, 10, 5, 5, 5, 0};

  for (int32_t i = 0; i < numCols; ++i) {
    auto writer = static_cast<Int32Writer*>(rgWriter->nextColumn());
    writer->writeBatch(
        numRows,
        defLevelsByCol[i].data(),
        i >= 3 ? refLevels.data() : nullptr,
        values.data());
  }

  ASSERT_NO_THROW(fileWriter->close());
  ASSERT_OK_AND_ASSIGN(auto buffer, sink->Finish());
  auto fileReader = ParquetFileReader::open(
      std::make_shared<::arrow::io::BufferReader>(buffer),
      defaultReaderProperties());
  auto metadata = fileReader->metadata();
  ASSERT_EQ(1, metadata->numRowGroups());
  auto rowGroupReader = fileReader->rowGroup(0);

  std::shared_ptr<Page> page;
  for (int32_t i = 0; i < numCols; ++i) {
    auto pageReader = rowGroupReader->getColumnPageReader(i);
    int64_t numNullsRead = 0;
    int64_t numRowsRead = 0;
    int64_t numValuesRead = 0;
    while ((page = pageReader->nextPage()) != nullptr) {
      auto dataPage = std::static_pointer_cast<DataPageV2>(page);
      numNullsRead += dataPage->numNulls();
      numRowsRead += dataPage->numRows();
      numValuesRead += dataPage->numValues();
    }
    EXPECT_EQ(expectNullCountByCol[i], numNullsRead);
    EXPECT_EQ(numRows, numRowsRead);
    EXPECT_EQ(numRows, numValuesRead);
  }
}

} // namespace test
} // namespace facebook::velox::parquet::arrow
