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

  ASSERT_EQ(props.buffer_size(), kDefaultBufferSize);
  ASSERT_FALSE(props.is_buffered_stream_enabled());
  ASSERT_FALSE(props.page_checksum_verification());
}

TEST(TestWriterProperties, Basics) {
  std::shared_ptr<WriterProperties> props = WriterProperties::Builder().build();

  ASSERT_EQ(kDefaultDataPageSize, props->data_pagesize());
  ASSERT_EQ(
      DEFAULT_DICTIONARY_PAGE_SIZE_LIMIT, props->dictionary_pagesize_limit());
  ASSERT_EQ(ParquetVersion::PARQUET_2_6, props->version());
  ASSERT_EQ(ParquetDataPageVersion::V1, props->data_page_version());
  ASSERT_FALSE(props->page_checksum_enabled());
}

TEST(TestWriterProperties, AdvancedHandling) {
  WriterProperties::Builder builder;
  builder.compression("gzip", Compression::GZIP);
  builder.compression("zstd", Compression::ZSTD);
  builder.compression(Compression::SNAPPY);
  builder.encoding(Encoding::DELTA_BINARY_PACKED);
  builder.encoding("delta-length", Encoding::DELTA_LENGTH_BYTE_ARRAY);
  builder.data_page_version(ParquetDataPageVersion::V2);
  std::shared_ptr<WriterProperties> props = builder.build();

  ASSERT_EQ(
      Compression::GZIP, props->compression(ColumnPath::FromDotString("gzip")));
  ASSERT_EQ(
      Compression::ZSTD, props->compression(ColumnPath::FromDotString("zstd")));
  ASSERT_EQ(
      Compression::SNAPPY,
      props->compression(ColumnPath::FromDotString("delta-length")));
  ASSERT_EQ(
      Encoding::DELTA_BINARY_PACKED,
      props->encoding(ColumnPath::FromDotString("gzip")));
  ASSERT_EQ(
      Encoding::DELTA_LENGTH_BYTE_ARRAY,
      props->encoding(ColumnPath::FromDotString("delta-length")));
  ASSERT_EQ(ParquetDataPageVersion::V2, props->data_page_version());
}

TEST(TestReaderProperties, GetStreamInsufficientData) {
  // ARROW-6058
  std::string data = "shorter than expected";
  auto buf = std::make_shared<Buffer>(data);
  auto reader = std::make_shared<::arrow::io::BufferReader>(buf);

  ReaderProperties props;
  try {
    ARROW_UNUSED(props.GetStream(reader, 12, 15));
    FAIL() << "No exception raised";
  } catch (const ParquetException& e) {
    std::string ex_what =
        ("Tried reading 15 bytes starting at position 12"
         " from file but only got 9");
    ASSERT_EQ(ex_what, e.what());
  }
}

} // namespace test
} // namespace facebook::velox::parquet::arrow
