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

#include <gtest/gtest.h>

#include "velox/common/base/RandomUtil.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/dwio/common/Mutation.h"
#include "velox/dwio/dwrf/common/Common.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/dwrf/test/OrcTest.h"
#include "velox/type/fbhive/HiveTypeParser.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox::dwio::common;
using namespace facebook::velox::type::fbhive;
using namespace facebook::velox;
using namespace facebook::velox::dwrf;
using namespace facebook::velox::test;

namespace {
class OrcReaderTest : public testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  std::shared_ptr<io::IoStatistics> dataIoStats_ =
      std::make_shared<io::IoStatistics>();
  std::shared_ptr<io::IoStatistics> metadataIoStats_ =
      std::make_shared<io::IoStatistics>();
};

inline std::string getExamplesFilePath(const std::string& fileName) {
  return test::getDataFilePath("velox/dwio/orc/test", "examples/" + fileName);
}

} // namespace

TEST_F(OrcReaderTest, testOrcReaderSimple) {
  const std::string simpleTest(
      getExamplesFilePath("TestStringDictionary.testRowIndex.orc"));
  dwio::common::ReaderOptions readerOpts{pool(), nullptr};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  // To make DwrfReader reads ORC file, setFileFormat to FileFormat::ORC
  readerOpts.setFileFormat(dwio::common::FileFormat::ORC);
  auto reader = DwrfReader::create(
      createFileBufferedInput(simpleTest, readerOpts.memoryPool()), readerOpts);

  RowReaderOptions rowReaderOptions;
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr batch;
  const std::string stringPrefix{"row "};
  size_t rowNumber = 0;
  while (rowReader->next(500, batch)) {
    auto rowVector = batch->as<RowVector>();
    auto strings = rowVector->childAt(0)->as<SimpleVector<StringView>>();
    for (size_t i = 0; i < rowVector->size(); ++i) {
      std::stringstream stream;
      stream << std::setfill('0') << std::setw(6) << rowNumber;
      EXPECT_EQ(stringPrefix + stream.str(), strings->valueAt(i).str());
      rowNumber++;
    }
  }
  EXPECT_EQ(rowNumber, 32768);
}
TEST_F(OrcReaderTest, testOrcReaderComplexTypes) {
  const std::string icebergOrc(getExamplesFilePath("complextypes_iceberg.orc"));
  const std::shared_ptr<const RowType> expectedType =
      std::dynamic_pointer_cast<const RowType>(HiveTypeParser().parse("struct<\
     id:bigint,int_array:array<int>,int_array_array:array<array<int>>,\
     int_map:map<string,int>,int_map_array:array<map<string,int>>,\
     nested_struct:struct<\
       a:int,b:array<int>,c:struct<\
         d:array<array<struct<\
           e:int,f:string>>>>,\
         g:map<string,struct<\
           h:struct<\
             i:array<double>>>>>>"));
  dwio::common::ReaderOptions readerOpts{pool(), nullptr};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setFileFormat(dwio::common::FileFormat::ORC);
  auto reader = DwrfReader::create(
      createFileBufferedInput(icebergOrc, readerOpts.memoryPool()), readerOpts);
  auto rowType = reader->rowType();
  EXPECT_TRUE(rowType->equivalent(*expectedType));

  RowReaderOptions rowReaderOptions;
  auto rowReader = reader->createRowReader(rowReaderOptions);
  VectorPtr batch;

  while (rowReader->next(500, batch)) {
    auto rowVector = batch->as<RowVector>();
    std::cout << rowVector->toString() << std::endl;
  }
}

TEST_F(OrcReaderTest, testOrcReaderVarchar) {
  const std::string varcharOrc(getExamplesFilePath("orc_index_int_string.orc"));
  dwio::common::ReaderOptions readerOpts{pool(), nullptr};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setFileFormat(dwio::common::FileFormat::ORC);
  auto reader = DwrfReader::create(
      createFileBufferedInput(varcharOrc, readerOpts.memoryPool()), readerOpts);

  RowReaderOptions rowReaderOptions;
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr batch;
  int counter = 0;
  while (rowReader->next(500, batch)) {
    auto rowVector = batch->as<RowVector>();
    auto ints = rowVector->childAt(0)->as<SimpleVector<int32_t>>();
    auto strings = rowVector->childAt(1)->as<SimpleVector<StringView>>();
    for (size_t i = 0; i < rowVector->size(); ++i) {
      counter++;
      EXPECT_EQ(counter, ints->valueAt(i));
      std::stringstream stream;
      stream << counter;
      if (counter < 1000) {
        stream << "a";
      }
      EXPECT_EQ(stream.str(), strings->valueAt(i).str());
    }
  }
  EXPECT_EQ(counter, 6000);
}

TEST_F(OrcReaderTest, testOrcReaderDate) {
  const std::string dateOrc(
      getExamplesFilePath("TestOrcFile.testDate1900.orc"));
  dwio::common::ReaderOptions readerOpts{pool(), nullptr};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setFileFormat(dwio::common::FileFormat::ORC);
  auto reader = DwrfReader::create(
      createFileBufferedInput(dateOrc, readerOpts.memoryPool()), readerOpts);

  RowReaderOptions rowReaderOptions;
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr batch;
  int year = 1900;
  while (rowReader->next(1000, batch)) {
    auto rowVector = batch->as<RowVector>();
    auto dates = rowVector->childAt(1)->as<SimpleVector<int32_t>>();

    std::stringstream stream;
    stream << year << "-12-25";
    EXPECT_EQ(stream.str(), DATE()->toString(dates->valueAt(0)));

    for (size_t i = 1; i < rowVector->size(); ++i) {
      EXPECT_EQ(dates->valueAt(0), dates->valueAt(i));
    }

    year++;
  }
}

// create table orc_types_test (
//    "a" integer,
//    "b" bigint,
//    "c" tinyint,
//    "d" smallint,
//    "e" real,
//    "f" double,
//    "g" varchar,
//    "h" boolean,
//    "i" decimal(38,6),
//    "j" decimal(9,2),
//    "k" date,
//    "l" timestamp,
//    "m" array(varchar(100)),
//    "n" map(varchar(20), bigint),
//    "o" ROW(x BIGINT, y DOUBLE)
// ) with (format = 'ORC');
TEST_F(OrcReaderTest, testOrcReadAllType) {
  const std::string dateOrc(getExamplesFilePath("orc_all_type.orc"));
  dwio::common::ReaderOptions readerOpts{pool(), nullptr};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setFileFormat(dwio::common::FileFormat::ORC);
  auto reader = DwrfReader::create(
      createFileBufferedInput(dateOrc, readerOpts.memoryPool()), readerOpts);

  RowReaderOptions rowReaderOptions;
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr batch;
  while (rowReader->next(500, batch)) {
    auto rowVector = batch->as<RowVector>();
    auto integerCol = rowVector->childAt(0)->as<SimpleVector<int32_t>>();
    auto bigintCol = rowVector->childAt(1)->as<SimpleVector<int64_t>>();
    auto tinyintCol = rowVector->childAt(2)->as<SimpleVector<int8_t>>();
    auto smallintCol = rowVector->childAt(3)->as<SimpleVector<int16_t>>();
    auto realCol = rowVector->childAt(4)->as<SimpleVector<float>>();
    auto doubleCol = rowVector->childAt(5)->as<SimpleVector<double>>();
    auto varcharCol = rowVector->childAt(6)->as<SimpleVector<StringView>>();
    auto booleanCol = rowVector->childAt(7)->as<SimpleVector<bool>>();
    auto longDecimalCol = rowVector->childAt(8)->as<SimpleVector<int128_t>>();
    auto shortDecimalCol = rowVector->childAt(9)->as<SimpleVector<int64_t>>();
    auto dateCol = rowVector->childAt(10)->as<SimpleVector<int32_t>>();
    auto timestampCol = rowVector->childAt(11)->as<SimpleVector<Timestamp>>();
    auto arrayCol = rowVector->childAt(12)->as<ArrayVector>();
    auto mapCol = rowVector->childAt(13)->as<MapVector>();
    auto structCol = rowVector->childAt(14)->as<RowVector>();

    EXPECT_EQ(1, rowVector->size());
    EXPECT_EQ(integerCol->valueAt(0), 111);
    EXPECT_EQ(bigintCol->valueAt(0), 1111);
    EXPECT_EQ(tinyintCol->valueAt(0), 127);
    EXPECT_EQ(smallintCol->valueAt(0), 11);
    EXPECT_EQ(realCol->valueAt(0), static_cast<float>(1.1));
    EXPECT_EQ(doubleCol->valueAt(0), static_cast<double>(1.12));
    EXPECT_EQ(varcharCol->valueAt(0), "velox");
    EXPECT_EQ(booleanCol->valueAt(0), false);

    auto longDecimalType = rowVector->type()->childAt(8);
    auto shortDecimalType = rowVector->type()->childAt(9);
    EXPECT_EQ(
        DecimalUtil::toString(longDecimalCol->valueAt(0), longDecimalType),
        "1242141234.123456");
    EXPECT_EQ(
        DecimalUtil::toString(shortDecimalCol->valueAt(0), shortDecimalType),
        "321423.21");

    EXPECT_EQ(dateCol->valueAt(0), DATE()->toDays("2023-08-18"));
    EXPECT_EQ(
        timestampCol->valueAt(0),
        util::fromTimestampString(
            "2023-08-18 08:12:23.000", util::TimestampParseMode::kPrestoCast)
            .value());

    auto arrayElements = arrayCol->elements()->as<SimpleVector<StringView>>();
    EXPECT_EQ(arrayElements->size(), 3);
    EXPECT_EQ(arrayElements->toString(0, 3, ",", false), "aaaa,BBBB,velox");

    auto mapKeys = mapCol->mapKeys()->as<SimpleVector<StringView>>();
    auto mapValues = mapCol->mapValues()->as<SimpleVector<int64_t>>();
    EXPECT_EQ(mapKeys->size(), 2);
    EXPECT_EQ(mapKeys->size(), mapValues->size());
    EXPECT_EQ(mapCol->toString(0, 2, ",", false), "{foo => 1, bar => 2}");

    EXPECT_EQ(structCol->size(), 1);
    EXPECT_EQ(structCol->type()->toString(), "ROW<x:BIGINT,y:DOUBLE>");
    EXPECT_EQ(structCol->toString(0, 2, ",", false), "{1, 2}");
  }
}

TEST_F(OrcReaderTest, testOrcRlev2) {
  google::InstallFailureSignalHandler();
  const std::string dateOrc(getExamplesFilePath("rlev2.orc"));
  auto schema =
      ROW({"id", "price", "name"}, {BIGINT(), DECIMAL(7, 2), VARCHAR()});
  auto spec = std::make_shared<common::ScanSpec>("<root>");
  spec->addAllChildFields(*schema);

  dwio::common::ReaderOptions readerOpts{pool(), spec};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setFileFormat(dwio::common::FileFormat::ORC);

  auto reader = DwrfReader::create(
      createFileBufferedInput(dateOrc, readerOpts.memoryPool()), readerOpts);

  RowReaderOptions rowReaderOptions;
  auto rowReader = reader->createRowReader(rowReaderOptions);

  auto batch = BaseVector::create(schema, 0, &readerOpts.memoryPool());
  while (rowReader->next(500, batch)) {
    auto rowVector = batch->as<RowVector>();
    auto idCol =
        rowVector->childAt(0)->loadedVector()->as<SimpleVector<int64_t>>();
    auto priceCol =
        rowVector->childAt(1)->loadedVector()->as<SimpleVector<int64_t>>();
    auto nameCol =
        rowVector->childAt(2)->loadedVector()->as<SimpleVector<StringView>>();

    EXPECT_EQ(5, rowVector->size());
    EXPECT_EQ(idCol->valueAt(0), 1);

    auto priceColType = rowVector->type()->childAt(1);
    EXPECT_EQ(
        DecimalUtil::toString(priceCol->valueAt(0), priceColType), "111.11");
    EXPECT_EQ(nameCol->valueAt(0), "AAAA");
  }
}
class OrcFileDescription {
 public:
  std::string filename;
  std::string json;
  std::string typeString;
  std::string formatVersion;
  std::string softwareVersion;
  uint64_t rowCount;
  uint64_t contentLength;
  uint64_t stripeCount;
  common::CompressionKind compression;
  size_t compressionSize;
  uint64_t rowIndexStride;
  std::map<std::string, std::string> userMeta;

  OrcFileDescription(
      const std::string& _filename,
      const std::string& _json,
      const std::string& _typeString,
      const std::string& _version,
      const std::string& _softwareVersion,
      uint64_t _rowCount,
      uint64_t _contentLength,
      uint64_t _stripeCount,
      common::CompressionKind _compression,
      size_t _compressionSize,
      uint64_t _rowIndexStride,
      const std::map<std::string, std::string>& _meta)
      : filename(_filename),
        json(_json),
        typeString(_typeString),
        formatVersion(_version),
        softwareVersion(_softwareVersion),
        rowCount(_rowCount),
        contentLength(_contentLength),
        stripeCount(_stripeCount),
        compression(_compression),
        compressionSize(_compressionSize),
        rowIndexStride(_rowIndexStride),
        userMeta(_meta) {}

  friend std::ostream& operator<<(
      std::ostream& stream,
      OrcFileDescription const& obj);
};

std::ostream& operator<<(std::ostream& stream, OrcFileDescription const& obj) {
  stream << obj.filename;
  return stream;
}

class OrcReaderTestP : public testing::TestWithParam<OrcFileDescription>,
                       public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  inline std::string getExamplesFilePath(const std::string& fileName) {
    return test::getDataFilePath("velox/dwio/orc/test", "examples/" + fileName);
  }

  inline std::string getExpectedFilePath(const std::string& fileName) {
    return test::getDataFilePath(
        "velox/dwio/orc/test", "examples/expected/" + fileName);
  }

  std::string getFilename() {
    return test::getDataFilePath(
        "velox/dwio/orc/test", "examples/" + GetParam().filename);
  }

  std::string getJsonFilename() {
    return test::getDataFilePath(
        "velox/dwio/orc/test", "examples/expected/" + GetParam().json);
  }

  std::shared_ptr<io::IoStatistics> dataIoStats_ =
      std::make_shared<io::IoStatistics>();
  std::shared_ptr<io::IoStatistics> metadataIoStats_ =
      std::make_shared<io::IoStatistics>();
};

TEST_P(
    OrcReaderTestP,
    dwrfReaderFetchesOrcMetadataExpectCorrectFooterAndMetadata) {
  const std::string dateOrc(getFilename());
  dwio::common::ReaderOptions readerOpts{pool(), nullptr};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setFileFormat(dwio::common::FileFormat::ORC);
  auto reader = DwrfReader::create(
      createFileBufferedInput(dateOrc, readerOpts.memoryPool()), readerOpts);

  const std::shared_ptr<const RowType> expectedType =
      std::dynamic_pointer_cast<const RowType>(
          HiveTypeParser().parse(GetParam().typeString));
  auto rowType = reader->rowType();
  EXPECT_TRUE(rowType->equivalent(*expectedType));

  EXPECT_EQ(GetParam().compression, reader->getCompression());
  EXPECT_EQ(GetParam().compressionSize, reader->getCompressionBlockSize());
  EXPECT_EQ(GetParam().stripeCount, reader->getNumberOfStripes());
  EXPECT_EQ(GetParam().rowCount, reader->getFooter().numberOfRows());
  EXPECT_EQ(GetParam().rowIndexStride, reader->getFooter().rowIndexStride());
  EXPECT_EQ(GetParam().contentLength, reader->getFooter().contentLength());
  EXPECT_EQ(GetParam().userMeta.size(), reader->getMetadataKeys().size());

  RowReaderOptions rowReaderOptions;
  auto rowReader = reader->createRowReader(rowReaderOptions);

  for (std::map<std::string, std::string>::const_iterator itr =
           GetParam().userMeta.cbegin();
       itr != GetParam().userMeta.cend();
       ++itr) {
    ASSERT_EQ(true, reader->hasMetadataValue(itr->first));
    std::string val = reader->getMetadataValue(itr->first);
    EXPECT_EQ(itr->second, val);
  }
}

TEST_P(OrcReaderTestP, dwrfRowReaderReadAllColumnTypesExpectedRowDataRead) {
  // Create schema and scan spec
  std::string schemaString = GetParam().typeString;
  auto type = HiveTypeParser().parse(schemaString);
  auto schema = std::dynamic_pointer_cast<const RowType>(type);
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addAllChildFields(*schema);

  const std::string dateOrc(getFilename());
  dwio::common::ReaderOptions readerOpts{pool(), scanSpec};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setFileFormat(dwio::common::FileFormat::ORC);

  auto reader = DwrfReader::create(
      createFileBufferedInput(dateOrc, readerOpts.memoryPool()), readerOpts);

  RowReaderOptions rowReaderOptions;
  auto rowReader = reader->createRowReader(rowReaderOptions);

  size_t rowCount = 0;
  auto batch = BaseVector::create(schema, 0, &readerOpts.memoryPool());
  while (rowReader->next(1024, batch)) {
    auto rowVector = batch->as<RowVector>();
    for (auto i = 0; i < rowVector->size(); ++i) {
      // TODO: Validate against all JSON data.  Currently the
      // vectorRow->toString(i) does not output the same stringified line format
      // as ColumnPrinter used to create json files.
      rowCount++;
    }
  }
  EXPECT_EQ(GetParam().rowCount, rowCount);
}

INSTANTIATE_TEST_SUITE_P(
    OrcReaderTestsP,
    OrcReaderTestP,
    testing::Values(
        OrcFileDescription(
            "TestOrcFile.columnProjection.orc",
            "TestOrcFile.columnProjection.jsn.gz",
            "struct<int1:int,string1:string>",
            "0.12",
            "ORC Java",
            21000,
            428406,
            5,
            common::CompressionKind::CompressionKind_NONE,
            262144,
            1000,
            std::map<std::string, std::string>()),
        OrcFileDescription(
            "TestOrcFile.testWithoutIndex.orc",
            "TestOrcFile.testWithoutIndex.jsn.gz",
            "struct<int1:int,string1:string>",
            "0.12",
            "ORC Java",
            50000,
            214643,
            10,
            common::CompressionKind::CompressionKind_SNAPPY,
            1000,
            0,
            std::map<std::string, std::string>())),
    [](const testing::TestParamInfo<OrcReaderTestP::ParamType>& info) {
      auto filename = info.param.filename;
      filename.erase(
          std::remove_if(
              filename.begin(),
              filename.end(),
              [](unsigned char c) { return !std::isalnum(c); }),
          filename.end());
      return filename;
    });

// A file without a row index (10 stripes of 5,000 rows, row index stride 0)
// reads the same through the selective reader, including from the middle of a
// stripe.
TEST_F(OrcReaderTest, selectiveReadWithoutRowIndex) {
  auto createReader = [&](const std::shared_ptr<common::ScanSpec>& scanSpec) {
    dwio::common::ReaderOptions readerOpts{pool(), scanSpec};
    readerOpts.setFileFormat(dwio::common::FileFormat::ORC);
    return DwrfReader::create(
        createFileBufferedInput(
            getExamplesFilePath("TestOrcFile.testWithoutIndex.orc"),
            readerOpts.memoryPool()),
        readerOpts);
  };
  auto reader = createReader(nullptr);
  ASSERT_EQ(reader->getFooter().rowIndexStride(), 0);
  const auto schema = reader->rowType();

  auto read = [&](DwrfReader& fileReader, const RowReaderOptions& options) {
    auto rowReader = fileReader.createRowReader(options);
    auto result = BaseVector::create<RowVector>(schema, 0, pool());
    VectorPtr batch = BaseVector::create(schema, 0, pool());
    while (rowReader->next(700, batch) > 0) {
      const auto offset = result->size();
      result->resize(offset + batch->size());
      result->copy(batch.get(), offset, 0, batch->size());
    }
    return result;
  };
  RowReaderOptions options;
  const auto expected = read(*reader, options);
  ASSERT_EQ(expected->size(), 50'000);
  auto spec = std::make_shared<common::ScanSpec>("<root>");
  spec->addAllChildFields(*schema);
  auto selectiveReader = createReader(spec);
  test::assertEqualVectors(expected, read(*selectiveReader, options));

  // Starting mid-stripe seeks within the stripe.
  options.setSplitRowRange(
      SplitRowRange{.firstRow = 12'345, .numRows = 20'000});
  test::assertEqualVectors(
      expected->slice(12'345, 20'000), read(*selectiveReader, options));
}

namespace {

// Reads ORC files with a split row range and compares the rows with the
// same rows of a full read.
class OrcSplitRowRangeTest : public OrcReaderTest,
                             public testing::WithParamInterface<bool> {
 protected:
  static constexpr uint64_t kAll{std::numeric_limits<uint64_t>::max()};

  void open(const std::string& fileName) {
    fileName_ = fileName;
    reader_ = createReader(nullptr);
    schema_ = reader_->rowType();
    if (GetParam()) {
      reader_ = createReader(makeScanSpec());
    }
    expected_ = read(RowReaderOptions{});
    ASSERT_EQ(expected_->size(), reader_->numberOfRows().value());
  }

  std::shared_ptr<common::ScanSpec> makeScanSpec() const {
    auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
    scanSpec->addAllChildFields(*schema_);
    return scanSpec;
  }

  // Opens the file, sampling its rows with 'randomSkip' if set.
  std::unique_ptr<DwrfReader> createReader(
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      std::shared_ptr<random::RandomSkipTracker> randomSkip = nullptr) {
    dwio::common::ReaderOptions readerOpts{pool(), scanSpec};
    readerOpts.setFileFormat(dwio::common::FileFormat::ORC);
    readerOpts.setRandomSkip(std::move(randomSkip));
    return DwrfReader::create(
        createFileBufferedInput(
            getExamplesFilePath(fileName_), readerOpts.memoryPool()),
        readerOpts);
  }

  // Reads all rows in batches of 700 into one vector.
  RowVectorPtr read(const RowReaderOptions& options) {
    auto result = BaseVector::create<RowVector>(schema_, 0, pool());
    append(*reader_->createRowReader(options), nullptr, *result);
    return result;
  }

  // Reads all rows of 'rowReader' in batches of 700, applying 'mutation', and
  // appends them to 'result'.
  void append(
      RowReader& rowReader,
      const dwio::common::Mutation* mutation,
      RowVector& result) {
    VectorPtr batch = BaseVector::create(schema_, 0, pool());
    while (rowReader.next(700, batch, mutation) > 0) {
      const auto offset = result.size();
      result.resize(offset + batch->size());
      result.copy(batch.get(), offset, 0, batch->size());
    }
  }

  // Reads 'range' of the split that starts at stripe 'firstStripe' and checks
  // the rows against the full read.
  void verify(SplitRowRange range, uint32_t firstStripe = 0) {
    SCOPED_TRACE(
        fmt::format(
            "firstStripe {} firstRow {} numRows {}",
            firstStripe,
            range.firstRow,
            range.numRows));
    const auto& footer = reader_->getFooter();
    uint64_t splitStartRow = 0;
    for (uint32_t i = 0; i < firstStripe; ++i) {
      splitStartRow += footer.stripes(i).numberOfRows();
    }
    const uint64_t numRows = expected_->size();
    const uint64_t begin = std::min(splitStartRow + range.firstRow, numRows);
    const uint64_t end =
        range.numRows >= numRows - begin ? numRows : begin + range.numRows;

    RowReaderOptions options;
    options.setSplitRowRange(range);
    options.range(footer.stripes(firstStripe).offset(), kAll / 2);
    auto actual = read(options);
    ASSERT_EQ(actual->size(), end - begin);
    for (vector_size_t i = 0; i < actual->size(); ++i) {
      ASSERT_TRUE(actual->equalValueAt(expected_.get(), i, begin + i))
          << "row " << begin + i << ": " << actual->toString(vector_size_t(i))
          << " vs " << expected_->toString(vector_size_t(begin + i));
    }
  }

  // Reads 'range' twice with one sampler, like two splits of a scan, and
  // checks that the kept rows are those that trying each row of the range in
  // turn would keep.
  void verifySampling(SplitRowRange range, double sampleRate, uint32_t seed) {
    SCOPED_TRACE(
        fmt::format(
            "firstRow {} numRows {} sampleRate {} seed {}",
            range.firstRow,
            range.numRows,
            sampleRate,
            seed));
    random::setSeed(seed);
    auto randomSkip = std::make_shared<random::RandomSkipTracker>(sampleRate);
    auto reader = createReader(makeScanSpec(), randomSkip);
    RowReaderOptions options;
    options.setSplitRowRange(range);
    dwio::common::Mutation mutation;
    mutation.randomSkip = randomSkip.get();
    auto actual = BaseVector::create<RowVector>(schema_, 0, pool());
    for (int i = 0; i < 2; ++i) {
      append(*reader->createRowReader(options), &mutation, *actual);
    }

    random::setSeed(seed);
    random::RandomSkipTracker expectedSkip(sampleRate);
    const uint64_t numRows = expected_->size();
    const uint64_t begin = std::min(range.firstRow, numRows);
    const uint64_t end = begin + std::min(range.numRows, numRows - begin);
    std::vector<vector_size_t> expectedRows;
    for (int i = 0; i < 2; ++i) {
      for (auto row = begin; row < end; ++row) {
        if (expectedSkip.testOne()) {
          expectedRows.push_back(row);
        }
      }
    }
    ASSERT_EQ(actual->size(), expectedRows.size());
    for (vector_size_t i = 0; i < actual->size(); ++i) {
      ASSERT_TRUE(actual->equalValueAt(expected_.get(), i, expectedRows[i]))
          << "row " << expectedRows[i];
    }
  }

  std::string fileName_;
  std::unique_ptr<DwrfReader> reader_;
  RowTypePtr schema_;
  RowVectorPtr expected_;
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    OrcSplitRowRangeTests,
    OrcSplitRowRangeTest,
    testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) {
      return info.param ? "selective" : "nonSelective";
    });

// 5 stripes of 5,000, 5,000, 5,000, 5,000 and 1,000 rows, with a row index
// stride of 1,000.
TEST_P(OrcSplitRowRangeTest, withRowIndex) {
  open("TestOrcFile.columnProjection.orc");
  ASSERT_EQ(reader_->getFooter().stripesSize(), 5);
  ASSERT_EQ(reader_->getFooter().rowIndexStride(), 1'000);
  verify({0, kAll});
  verify({1, 2});
  // Starts past the first strides and crosses a stride boundary.
  verify({2'500, 800});
  // Crosses stripes.
  verify({4'900, 5'200});
  verify({20'500, kAll});
  verify({20'999, 1});
  verify({21'000, 1});
  verify({3, 0});
  // Counts rows from the split's first stripe.
  verify({0, 2}, 1);
  verify({3'500, 3'000}, 2);
  verify({6'000, kAll}, 3);
  verify({6'000, 1}, 3);
}

// 10 stripes of 5,000 rows and no row index.
TEST_P(OrcSplitRowRangeTest, withoutRowIndex) {
  open("TestOrcFile.testWithoutIndex.orc");
  ASSERT_EQ(reader_->getFooter().stripesSize(), 10);
  ASSERT_EQ(reader_->getFooter().rowIndexStride(), 0);
  verify({2'500, 800});
  verify({4'900, 5'200});
  verify({12'345, 20'000});
  verify({49'999, kAll});
  verify({3'500, 3'000}, 4);
}

// The low sample rates skip whole stripes, including the range's partial first
// and last stripes.
TEST_P(OrcSplitRowRangeTest, randomSampling) {
  if (!GetParam()) {
    GTEST_SKIP() << "Only the selective reader samples rows";
  }
  for (const auto& fileName :
       {"TestOrcFile.columnProjection.orc",
        "TestOrcFile.testWithoutIndex.orc"}) {
    open(fileName);
    for (const SplitRowRange range :
         {SplitRowRange{0, kAll},
          SplitRowRange{2'500, 800},
          SplitRowRange{4'900, 5'200},
          SplitRowRange{20'500, kAll}}) {
      for (const double sampleRate : {0.1, 0.001, 0.0002}) {
        for (uint32_t seed = 1; seed <= 20; ++seed) {
          verifySampling(range, sampleRate, seed);
        }
      }
    }
  }
}

// One stripe of 32,768 strings with a row index stride of 4,096.
TEST_P(OrcSplitRowRangeTest, oneStripe) {
  open("TestStringDictionary.testRowIndex.orc");
  ASSERT_EQ(reader_->getFooter().stripesSize(), 1);
  verify({4'095, 2});
  verify({10'000, 10'000});
  verify({30'000, kAll});
}
