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

#include <arrow/type.h>
#include <folly/init/Init.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/core/QueryCtx.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/dwio/parquet/reader/PageReader.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/exec/tests/utils/Cursor.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

namespace {

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::parquet;

class ParquetWriterTest : public ParquetTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
    testutil::TestValue::enable();
    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(
                kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);
    parquet::registerParquetWriterFactory();
  }

  std::unique_ptr<RowReader> createRowReaderWithSchema(
      const std::unique_ptr<Reader> reader,
      const RowTypePtr& rowType) {
    auto rowReaderOpts = getReaderOpts(rowType);
    auto scanSpec = makeScanSpec(rowType);
    rowReaderOpts.setScanSpec(scanSpec);
    auto rowReader = reader->createRowReader(rowReaderOpts);
    return rowReader;
  };

  std::unique_ptr<facebook::velox::parquet::ParquetReader> createReaderInMemory(
      const dwio::common::MemorySink& sink,
      const dwio::common::ReaderOptions& opts) {
    std::string data(sink.data(), sink.size());
    return std::make_unique<facebook::velox::parquet::ParquetReader>(
        std::make_unique<dwio::common::BufferedInput>(
            std::make_shared<InMemoryReadFile>(std::move(data)),
            opts.memoryPool()),
        opts);
  };

  facebook::velox::parquet::thrift::PageType::type getDataPageVersion(
      const dwio::common::MemorySink* sinkPtr,
      const facebook::velox::parquet::ColumnChunkMetaDataPtr& colChunkPtr) {
    std::string_view sinkData(sinkPtr->data(), sinkPtr->size());
    auto readFile = std::make_shared<InMemoryReadFile>(sinkData);
    auto file = std::make_shared<ReadFileInputStream>(std::move(readFile));
    auto inputStream = std::make_unique<SeekableFileInputStream>(
        std::move(file),
        colChunkPtr.dataPageOffset(),
        150,
        *leafPool_,
        LogType::TEST);
    auto pageReader = std::make_unique<PageReader>(
        std::move(inputStream),
        *leafPool_,
        colChunkPtr.compression(),
        colChunkPtr.totalCompressedSize());
    return pageReader->readPageHeader().type;
  };

  inline static const std::string kHiveConnectorId = "test-hive";
};

std::vector<CompressionKind> params = {
    CompressionKind::CompressionKind_NONE,
    CompressionKind::CompressionKind_SNAPPY,
    CompressionKind::CompressionKind_ZSTD,
    CompressionKind::CompressionKind_LZ4,
    CompressionKind::CompressionKind_GZIP,
};

TEST_F(ParquetWriterTest, compression) {
  auto schema =
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {INTEGER(),
           DOUBLE(),
           BIGINT(),
           INTEGER(),
           BIGINT(),
           INTEGER(),
           DOUBLE()});
  const int64_t kRows = 10'000;
  const auto data = makeRowVector({
      makeFlatVector<int32_t>(kRows, [](auto row) { return row + 5; }),
      makeFlatVector<double>(kRows, [](auto row) { return row - 10; }),
      makeFlatVector<int64_t>(kRows, [](auto row) { return row - 15; }),
      makeFlatVector<uint32_t>(kRows, [](auto row) { return row + 20; }),
      makeFlatVector<uint64_t>(kRows, [](auto row) { return row + 25; }),
      makeFlatVector<int32_t>(kRows, [](auto row) { return row + 30; }),
      makeFlatVector<double>(kRows, [](auto row) { return row - 25; }),
  });

  // Create an in-memory writer
  auto sink = std::make_unique<MemorySink>(
      200 * 1024 * 1024,
      dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto sinkPtr = sink.get();
  facebook::velox::parquet::WriterOptions writerOptions;
  writerOptions.memoryPool = leafPool_.get();
  writerOptions.compressionKind = CompressionKind::CompressionKind_SNAPPY;

  const auto& fieldNames = schema->names();

  for (int i = 0; i < params.size(); i++) {
    writerOptions.columnCompressionsMap[fieldNames[i]] = params[i];
  }

  auto writer = std::make_unique<facebook::velox::parquet::Writer>(
      std::move(sink), writerOptions, rootPool_, schema);
  writer->write(data);
  writer->close();

  dwio::common::ReaderOptions readerOptions{leafPool_.get()};
  auto reader = createReaderInMemory(*sinkPtr, readerOptions);

  ASSERT_EQ(reader->numberOfRows(), kRows);
  ASSERT_EQ(*reader->rowType(), *schema);

  for (int i = 0; i < params.size(); i++) {
    EXPECT_EQ(
        reader->fileMetaData().rowGroup(0).columnChunk(i).compression(),
        (i < params.size()) ? params[i]
                            : CompressionKind::CompressionKind_SNAPPY);
  }

  auto rowReader = createRowReaderWithSchema(std::move(reader), schema);
  assertReadWithReaderAndExpected(schema, *rowReader, data, *leafPool_);
};

TEST_F(ParquetWriterTest, datapageVersionV1) {
  auto schema = ROW({"c0"}, {INTEGER()});
  const int64_t kRows = 1;
  const auto data = makeRowVector({
      makeFlatVector<int32_t>(kRows, [](auto row) { return 987; }),
  });

  // Create an in-memory writer
  auto sink = std::make_unique<MemorySink>(
      200 * 1024 * 1024,
      dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto sinkPtr = sink.get();
  facebook::velox::parquet::WriterOptions writerOptions;
  writerOptions.memoryPool = leafPool_.get();
  writerOptions.parquetDataPageVersion =
      facebook::velox::parquet::arrow::ParquetDataPageVersion::V1;

  auto writer = std::make_unique<facebook::velox::parquet::Writer>(
      std::move(sink), writerOptions, rootPool_, schema);
  writer->write(data);
  writer->close();

  dwio::common::ReaderOptions readerOptions{leafPool_.get()};
  auto reader = createReaderInMemory(*sinkPtr, readerOptions);
  auto dataPageVersion = getDataPageVersion(
      sinkPtr, reader->fileMetaData().rowGroup(0).columnChunk(0));
  ASSERT_EQ(dataPageVersion, thrift::PageType::type::DATA_PAGE);
};

TEST_F(ParquetWriterTest, datapageVersionV2) {
  auto schema = ROW({"c0"}, {INTEGER()});
  const int64_t kRows = 1;
  const auto data = makeRowVector({
      makeFlatVector<int32_t>(kRows, [](auto row) { return 987; }),
  });

  // Create an in-memory writer
  auto sink = std::make_unique<MemorySink>(
      200 * 1024 * 1024,
      dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto sinkPtr = sink.get();
  facebook::velox::parquet::WriterOptions writerOptions;
  writerOptions.memoryPool = leafPool_.get();
  writerOptions.parquetDataPageVersion =
      facebook::velox::parquet::arrow::ParquetDataPageVersion::V1;

  auto writer = std::make_unique<facebook::velox::parquet::Writer>(
      std::move(sink), writerOptions, rootPool_, schema);
  writer->write(data);
  writer->close();

  dwio::common::ReaderOptions readerOptions{leafPool_.get()};
  auto reader = createReaderInMemory(*sinkPtr, readerOptions);
  auto dataPageVersion = getDataPageVersion(
      sinkPtr, reader->fileMetaData().rowGroup(0).columnChunk(0));
  ASSERT_EQ(dataPageVersion, thrift::PageType::type::DATA_PAGE_V2);
};

DEBUG_ONLY_TEST_F(ParquetWriterTest, unitFromWriterOptions) {
  SCOPED_TESTVALUE_SET(
      "facebook::velox::parquet::Writer::write",
      std::function<void(const ::arrow::Schema*)>(
          ([&](const ::arrow::Schema* arrowSchema) {
            const auto tsType =
                std::dynamic_pointer_cast<::arrow::TimestampType>(
                    arrowSchema->field(0)->type());
            ASSERT_EQ(tsType->unit(), ::arrow::TimeUnit::MICRO);
          })));

  const auto data = makeRowVector({makeFlatVector<Timestamp>(
      10'000, [](auto row) { return Timestamp(row, row); })});
  parquet::WriterOptions writerOptions;
  writerOptions.memoryPool = leafPool_.get();
  writerOptions.parquetWriteTimestampUnit = TimestampUnit::kMicro;

  // Create an in-memory writer.
  auto sink = std::make_unique<MemorySink>(
      200 * 1024 * 1024,
      dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto writer = std::make_unique<parquet::Writer>(
      std::move(sink), writerOptions, rootPool_, ROW({"c0"}, {TIMESTAMP()}));
  writer->write(data);
  writer->close();
};

#ifdef VELOX_ENABLE_PARQUET
DEBUG_ONLY_TEST_F(ParquetWriterTest, unitFromHiveConfig) {
  SCOPED_TESTVALUE_SET(
      "facebook::velox::parquet::Writer::write",
      std::function<void(const ::arrow::Schema*)>(
          ([&](const ::arrow::Schema* arrowSchema) {
            const auto tsType =
                std::dynamic_pointer_cast<::arrow::TimestampType>(
                    arrowSchema->field(0)->type());
            ASSERT_EQ(tsType->unit(), ::arrow::TimeUnit::MICRO);
          })));

  const auto data = makeRowVector({makeFlatVector<Timestamp>(
      10'000, [](auto row) { return Timestamp(row, row); })});
  const auto outputDirectory = TempDirectoryPath::create();

  auto writerOptions = std::make_shared<parquet::WriterOptions>();
  writerOptions->parquetWriteTimestampUnit = TimestampUnit::kMicro;

  const auto plan = PlanBuilder()
                        .values({data})
                        .tableWrite(
                            outputDirectory->getPath(),
                            dwio::common::FileFormat::PARQUET,
                            {},
                            writerOptions)
                        .planNode();
  AssertQueryBuilder(plan).copyResults(pool_.get());
}
#endif

} // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
