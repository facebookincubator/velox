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
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/velox/ChunkedStream.h"
#include "velox/dwio/nimble/velox/ChunkedStreamWriter.h"

using namespace facebook;

namespace {

class TestStreamLoader : public nimble::StreamLoader {
 public:
  explicit TestStreamLoader(std::string stream) : stream_{std::move(stream)} {}
  const std::string_view getStream() const override {
    return stream_;
  }

 private:
  const std::string stream_;
};

} // namespace

TEST(ChunkedStreamWriterTest, uncompressedEncoding) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  nimble::ChunkedStreamWriter writer{buffer};

  std::string data = "hello world";
  auto segments = writer.encode(data);

  // Should produce exactly 2 segments: header and data.
  ASSERT_EQ(segments.size(), 2);

  // Parse the header: uint32_t size + CompressionType byte.
  const char* pos = segments[0].data();
  auto size = nimble::encoding::read<uint32_t>(pos);
  auto compressionType = nimble::encoding::read<nimble::CompressionType>(pos);

  EXPECT_EQ(size, data.size());
  EXPECT_EQ(compressionType, nimble::CompressionType::Uncompressed);

  // Data segment should point to the original data.
  EXPECT_EQ(segments[1], data);
}

TEST(ChunkedStreamWriterTest, headerFormatIs5Bytes) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  nimble::ChunkedStreamWriter writer{buffer};

  auto segments = writer.encode("test");
  constexpr size_t expectedHeaderSize =
      sizeof(uint32_t) + sizeof(nimble::CompressionType);
  EXPECT_EQ(segments[0].size(), expectedHeaderSize);
  EXPECT_EQ(expectedHeaderSize, 5);
}

TEST(ChunkedStreamWriterTest, zstdCompressionWithCompressibleData) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  nimble::CompressionParams params{
      .type = nimble::CompressionType::Zstd, .zstdLevel = 3};
  nimble::ChunkedStreamWriter writer{buffer, params};

  // Highly compressible data: long string of repeated characters.
  std::string data(1000, 'a');
  auto segments = writer.encode(data);

  ASSERT_EQ(segments.size(), 2);

  const char* pos = segments[0].data();
  auto compressedSize = nimble::encoding::read<uint32_t>(pos);
  auto compressionType = nimble::encoding::read<nimble::CompressionType>(pos);

  EXPECT_EQ(compressionType, nimble::CompressionType::Zstd);
  EXPECT_LT(compressedSize, data.size());
}

TEST(ChunkedStreamWriterTest, zstdFallbackToUncompressed) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  nimble::CompressionParams params{
      .type = nimble::CompressionType::Zstd, .zstdLevel = 3};
  nimble::ChunkedStreamWriter writer{buffer, params};

  // Small random data that won't compress well.
  std::string data;
  data.resize(16);
  for (int i = 0; i < 16; ++i) {
    data[i] = static_cast<char>(i * 17 + 31);
  }
  auto segments = writer.encode(data);

  ASSERT_EQ(segments.size(), 2);

  const char* pos = segments[0].data();
  nimble::encoding::read<uint32_t>(pos); // skip size
  auto compressionType = nimble::encoding::read<nimble::CompressionType>(pos);

  // With small incompressible data, zstd should fall back to uncompressed.
  EXPECT_EQ(compressionType, nimble::CompressionType::Uncompressed);
}

TEST(ChunkedStreamWriterTest, lowAcceptRatioRejectsCompression) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  // A tiny accept ratio demands near-total savings, so even highly compressible
  // data is rejected and stored uncompressed.
  nimble::CompressionParams params{
      .type = nimble::CompressionType::Zstd,
      .zstdLevel = 3,
      .acceptRatio = 0.01f};
  nimble::ChunkedStreamWriter writer{buffer, params};

  std::string data(1000, 'a');
  auto segments = writer.encode(data);

  ASSERT_EQ(segments.size(), 2);

  const char* pos = segments[0].data();
  nimble::encoding::read<uint32_t>(pos); // skip size
  auto compressionType = nimble::encoding::read<nimble::CompressionType>(pos);

  EXPECT_EQ(compressionType, nimble::CompressionType::Uncompressed);
}

TEST(ChunkedStreamWriterTest, highAcceptRatioKeepsCompression) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  // A permissive accept ratio keeps any size reduction.
  nimble::CompressionParams params{
      .type = nimble::CompressionType::Zstd,
      .zstdLevel = 3,
      .acceptRatio = 1.0f};
  nimble::ChunkedStreamWriter writer{buffer, params};

  std::string data(1000, 'a');
  auto segments = writer.encode(data);

  ASSERT_EQ(segments.size(), 2);

  const char* pos = segments[0].data();
  auto compressedSize = nimble::encoding::read<uint32_t>(pos);
  auto compressionType = nimble::encoding::read<nimble::CompressionType>(pos);

  EXPECT_EQ(compressionType, nimble::CompressionType::Zstd);
  EXPECT_LT(compressedSize, data.size());
}

TEST(ChunkedStreamWriterTest, lz4CompressionWithCompressibleData) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  nimble::CompressionParams params{.type = nimble::CompressionType::Lz4};
  nimble::ChunkedStreamWriter writer{buffer, params};

  // Highly compressible data: long string of repeated characters.
  std::string data(1000, 'a');
  auto segments = writer.encode(data);

  ASSERT_EQ(segments.size(), 2);

  const char* pos = segments[0].data();
  auto compressedSize = nimble::encoding::read<uint32_t>(pos);
  auto compressionType = nimble::encoding::read<nimble::CompressionType>(pos);

  EXPECT_EQ(compressionType, nimble::CompressionType::Lz4);
  EXPECT_LT(compressedSize, data.size());
}

TEST(ChunkedStreamWriterTest, lz4FallbackToUncompressed) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  nimble::CompressionParams params{.type = nimble::CompressionType::Lz4};
  nimble::ChunkedStreamWriter writer{buffer, params};

  // Small random data that won't compress well.
  std::string data;
  data.resize(16);
  for (int i = 0; i < 16; ++i) {
    data[i] = static_cast<char>(i * 17 + 31);
  }
  auto segments = writer.encode(data);

  ASSERT_EQ(segments.size(), 2);

  const char* pos = segments[0].data();
  nimble::encoding::read<uint32_t>(pos); // skip size
  auto compressionType = nimble::encoding::read<nimble::CompressionType>(pos);

  // With small incompressible data, lz4 should fall back to uncompressed.
  EXPECT_EQ(compressionType, nimble::CompressionType::Uncompressed);
}

TEST(ChunkedStreamWriterTest, lowAcceptRatioRejectsCompressionLz4) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  // The accept ratio applies to lz4 too: a tiny ratio rejects even highly
  // compressible data.
  nimble::CompressionParams params{
      .type = nimble::CompressionType::Lz4, .acceptRatio = 0.01f};
  nimble::ChunkedStreamWriter writer{buffer, params};

  std::string data(1000, 'a');
  auto segments = writer.encode(data);

  ASSERT_EQ(segments.size(), 2);

  const char* pos = segments[0].data();
  nimble::encoding::read<uint32_t>(pos); // skip size
  auto compressionType = nimble::encoding::read<nimble::CompressionType>(pos);

  EXPECT_EQ(compressionType, nimble::CompressionType::Uncompressed);
}

TEST(ChunkedStreamWriterTest, unsupportedCompressionTypeThrows) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  // Only Uncompressed, Zstd and Lz4 are supported; zstrong (MetaInternal) is
  // intentionally excluded (dominated by Zstd) and must be rejected.
  nimble::CompressionParams params{
      .type = nimble::CompressionType::MetaInternal};
  NIMBLE_ASSERT_THROW(
      nimble::ChunkedStreamWriter(buffer, params),
      "Unsupported chunked stream compression type");
}

TEST(ChunkedStreamWriterTest, emptyChunk) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  nimble::ChunkedStreamWriter writer{buffer};

  auto segments = writer.encode("");

  ASSERT_EQ(segments.size(), 2);

  const char* pos = segments[0].data();
  auto size = nimble::encoding::read<uint32_t>(pos);
  EXPECT_EQ(size, 0);
}

TEST(ChunkedStreamWriterTest, roundTrip) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  nimble::ChunkedStreamWriter writer{buffer};

  std::string data = "round trip test data with some content";
  auto segments = writer.encode(data);

  // Concatenate all segments into a single stream.
  std::string stream;
  for (const auto& seg : segments) {
    stream += seg;
  }

  auto loader = std::make_unique<TestStreamLoader>(std::move(stream));
  nimble::InMemoryChunkedStream reader{*pool, std::move(loader)};

  ASSERT_TRUE(reader.hasNext());
  EXPECT_EQ(
      nimble::CompressionType::Uncompressed, reader.peekCompressionType());
  auto chunk = reader.nextChunk();
  EXPECT_EQ(chunk, data);
  EXPECT_FALSE(reader.hasNext());
}

TEST(ChunkedStreamWriterTest, roundTripCompressed) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  nimble::CompressionParams params{
      .type = nimble::CompressionType::Zstd, .zstdLevel = 3};
  nimble::ChunkedStreamWriter writer{buffer, params};

  // Compressible data.
  std::string data(500, 'x');
  data += "some variety to make it interesting";
  auto segments = writer.encode(data);

  std::string stream;
  for (const auto& seg : segments) {
    stream += seg;
  }

  auto loader = std::make_unique<TestStreamLoader>(std::move(stream));
  nimble::InMemoryChunkedStream reader{*pool, std::move(loader)};

  ASSERT_TRUE(reader.hasNext());
  EXPECT_EQ(nimble::CompressionType::Zstd, reader.peekCompressionType());
  auto chunk = reader.nextChunk();
  EXPECT_EQ(chunk, data);
  EXPECT_FALSE(reader.hasNext());
}
