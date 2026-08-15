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

#include <gtest/gtest.h>

#include <string>

#include "arrow/buffer.h"
#include "arrow/io/memory.h"

#include "velox/dwio/parquet/writer/arrow/Properties.h"

namespace facebook::velox::parquet::arrow {

using schema::ColumnPath;

namespace test {

TEST(TestReaderProperties, Basics) {
  ReaderProperties props;

  ASSERT_EQ(props.bufferSize(), kDefaultBufferSize);
  ASSERT_FALSE(props.isBufferedStreamEnabled());
  ASSERT_FALSE(props.pageChecksumVerification());
}

TEST(TestWriterProperties, Basics) {
  std::shared_ptr<WriterProperties> props = WriterProperties::Builder().build();

  ASSERT_EQ(kDefaultDataPageSize, props->dataPagesize());
  ASSERT_EQ(
      DEFAULT_DICTIONARY_PAGE_SIZE_LIMIT, props->dictionaryPagesizeLimit());
  ASSERT_EQ(ParquetVersion::PARQUET_2_6, props->version());
  ASSERT_EQ(ParquetDataPageVersion::V1, props->dataPageVersion());
  ASSERT_FALSE(props->pageChecksumEnabled());
}

TEST(TestWriterProperties, AdvancedHandling) {
  WriterProperties::Builder Builder;
  Builder.compression("gzip", Compression::GZIP);
  Builder.compression("zstd", Compression::ZSTD);
  Builder.compression(Compression::SNAPPY);
  Builder.encoding(Encoding::kDeltaBinaryPacked);
  Builder.encoding("delta-length", Encoding::kDeltaLengthByteArray);
  Builder.dataPageVersion(ParquetDataPageVersion::V2);
  std::shared_ptr<WriterProperties> props = Builder.build();

  ASSERT_EQ(
      Compression::GZIP, props->compression(ColumnPath::fromDotString("gzip")));
  ASSERT_EQ(
      Compression::ZSTD, props->compression(ColumnPath::fromDotString("zstd")));
  ASSERT_EQ(
      Compression::SNAPPY,
      props->compression(ColumnPath::fromDotString("delta-length")));
  ASSERT_EQ(
      Encoding::kDeltaBinaryPacked,
      props->encoding(ColumnPath::fromDotString("gzip")));
  ASSERT_EQ(
      Encoding::kDeltaLengthByteArray,
      props->encoding(ColumnPath::fromDotString("delta-length")));
  ASSERT_EQ(ParquetDataPageVersion::V2, props->dataPageVersion());
}

TEST(TestReaderProperties, GetStreamInsufficientData) {
  // ARROW-6058.
  std::string data = "shorter than expected";
  auto buf = std::make_shared<Buffer>(data);
  auto reader = std::make_shared<::arrow::io::BufferReader>(buf);

  ReaderProperties props;
  try {
    ARROW_UNUSED(props.getStream(reader, 12, 15));
    FAIL() << "No exception raised";
  } catch (const ParquetException& e) {
    std::string exWhat =
        ("Tried reading 15 bytes starting at position 12"
         " from file but only got 9");
    ASSERT_EQ(exWhat, e.what());
  }
}

} // namespace test
} // namespace facebook::velox::parquet::arrow
