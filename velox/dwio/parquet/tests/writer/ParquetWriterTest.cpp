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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;

class ParquetWriterTest : public ParquetTestBase {
 protected:
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
    std::string_view data(sink.data(), sink.size());
    return std::make_unique<facebook::velox::parquet::ParquetReader>(
        std::make_unique<dwio::common::BufferedInput>(
            std::make_shared<InMemoryReadFile>(data), opts.getMemoryPool()),
        opts);
  };

  template <bool flatten>
  void testConstantVector() {
    auto schema = ROW({"c0"}, {INTEGER()});
    int32_t value = (int32_t)123'456;
    int32_t size = 100;
    auto data = makeRowVector({makeConstant(value, size)});
    auto filePath =
        fs::path(fmt::format("{}/test_constant.txt", tempPath_->path));
    auto sink = createSink(filePath.string());
    auto sinkPtr = sink.get();

    auto parquetOptions = getWriterOpts();
    EXPECT_TRUE(parquetOptions.flattenConstantVector);

    if constexpr (!flatten) {
      parquetOptions.flattenConstantVector = false;
    }

    auto writer = createWriter(std::move(sink), parquetOptions, schema);
    writer->write(data);

    if constexpr (!flatten) {
      try {
        writer->close();
        FAIL() << "Expected a ParquetException";
      } catch (const parquet::arrow::ParquetException& pe) {
        EXPECT_TRUE(
            std::string(pe.what()).find(
                "NotImplemented: Unhandled type for Arrow to Parquet schema conversion: run_end_encoded<run_ends: int32, values: int32>") !=
            std::string::npos);
      }
    } else {
      writer->close();

      dwio::common::ReaderOptions readerOptions{leafPool_.get()};
      auto reader = createReader(filePath.string(), readerOptions);

      ASSERT_EQ(reader->numberOfRows(), size);
      auto type = reader->typeWithId();
      auto col0 = type->childAt(0);
      EXPECT_EQ(type->size(), 1ULL);
      EXPECT_EQ(col0->type()->kind(), TypeKind::INTEGER);

      auto expected = makeRowVector({makeFlatVector<int32_t>(
          size, [&value](auto row) { return value; })});
      auto rowReader = createRowReaderWithSchema(std::move(reader), schema);
      assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
    }
  };
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
  writerOptions.compression = CompressionKind::CompressionKind_SNAPPY;

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
}

TEST_F(ParquetWriterTest, flattenConstantVector) {
  testConstantVector<true>();
}

TEST_F(ParquetWriterTest, constantVectorReeError) {
  testConstantVector<false>();
}
