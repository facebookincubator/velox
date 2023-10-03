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

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::dwio::parquet;
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
};

class SupportedParquetWriterCompressionTest
    : public ParquetWriterTest,
      public testing::WithParamInterface<CompressionKind> {
 public:
  SupportedParquetWriterCompressionTest() : compressionKind_(GetParam()) {}
  static std::vector<CompressionKind> getTestParams() {
    static std::vector<CompressionKind> params = {
        CompressionKind::CompressionKind_NONE,
        CompressionKind::CompressionKind_SNAPPY,
    };
    return params;
  }

 protected:
  CompressionKind compressionKind_;
};

TEST_P(SupportedParquetWriterCompressionTest, compression) {
  auto schema = ROW({"c0", "c1"}, {INTEGER(), DOUBLE()});
  auto data = makeRowVector({
      makeFlatVector<int32_t>(10, [](auto row) { return row + 5; }),
      makeFlatVector<double>(10, [](auto row) { return row + 10; }),
  });
  auto filePath = fs::path(fmt::format("{}/test_close.txt", tempPath_->path));
  auto [writer, sinkPtr] = createWriterWithSinkPtr(
      filePath.string(),
      [&]() {
        return std::make_unique<LambdaFlushPolicy>(
            kRowsInRowGroup, kBytesInRowGroup, [&]() { return false; });
      },
      compressionKind_);

  writer->write(data);

  writer->close();

  dwio::common::ReaderOptions readerOptions{leafPool_.get()};

  auto reader = createReader(filePath.string(), readerOptions);

  ASSERT_EQ(reader->numberOfRows(), 10ULL);

  auto type = reader->typeWithId();
  auto col0 = type->childAt(0);
  EXPECT_EQ(type->size(), 2ULL);
  EXPECT_EQ(col0->type()->kind(), TypeKind::INTEGER);
  auto col1 = type->childAt(1);
  EXPECT_EQ(col1->type()->kind(), TypeKind::DOUBLE);

  auto fileMetaData = reader->getFileMetaData();
  auto fileCompressionKind = getCompressionKindFromFileMetaData(fileMetaData);
  EXPECT_EQ(fileCompressionKind, compressionKind_);

  auto rowReader = createRowReaderWithSchema(std::move(reader), schema);

  assertReadExpected(schema, *rowReader, data, *leafPool_);
};

VELOX_INSTANTIATE_TEST_SUITE_P(
    SupportedParquetWriterCompressionTestSuite,
    SupportedParquetWriterCompressionTest,
    testing::ValuesIn(SupportedParquetWriterCompressionTest::getTestParams()));
    
