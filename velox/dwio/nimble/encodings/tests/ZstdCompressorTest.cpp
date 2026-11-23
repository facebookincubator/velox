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
#include "velox/dwio/nimble/compression/ZstdCompressor.h"
#include <gtest/gtest.h>
#include <cstring>
#include <random>
#include "velox/buffer/BufferPool.h"
#include "velox/common/memory/Memory.h"

using namespace facebook::nimble;

namespace {

class TestCompressionPolicy : public CompressionPolicy {
 public:
  explicit TestCompressionPolicy(uint64_t minCompressionSize = 0) {
    compressionInfo_ = {
        .compressionType = CompressionType::Zstd,
        .minCompressionSize = minCompressionSize};
    compressionInfo_.parameters.zstd.compressionLevel = 3;
  }

  CompressionConfig config() const override {
    return compressionInfo_;
  }

  bool shouldAccept(
      CompressionType /* compressionType */,
      uint64_t uncompressedSize,
      uint64_t compressedSize) const override {
    return compressedSize < uncompressedSize;
  }

 private:
  CompressionConfig compressionInfo_;
};

// Always declines, to verify the compressor honors shouldAccept().
class RejectingCompressionPolicy : public CompressionPolicy {
 public:
  CompressionConfig config() const override {
    CompressionConfig config{.compressionType = CompressionType::Zstd};
    config.parameters.zstd.compressionLevel = 3;
    return config;
  }

  bool shouldAccept(
      CompressionType /* compressionType */,
      uint64_t /* uncompressedSize */,
      uint64_t /* compressedSize */) const override {
    return false;
  }
};

} // namespace

TEST(ZstdCompressorTest, compressionType) {
  ZstdCompressor compressor;
  EXPECT_EQ(CompressionType::Zstd, compressor.compressionType());
}

TEST(ZstdCompressorTest, roundTrip) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  ZstdCompressor compressor;
  TestCompressionPolicy policy;

  // Create compressible data (repeated pattern)
  std::vector<char> original(1024);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<char>(i % 10);
  }
  std::string_view input(original.data(), original.size());

  auto result = compressor.compress(*pool, input, DataType::Int8, 0, policy);
  EXPECT_EQ(CompressionType::Zstd, result.compressionType);
  ASSERT_TRUE(result.buffer.has_value());

  std::string_view compressed(result.buffer->data(), result.buffer->size());

  auto decompressed = compressor.uncompress(
      *pool, CompressionType::Zstd, DataType::Int8, compressed);
  ASSERT_EQ(original.size(), decompressed->size());
  EXPECT_EQ(
      0,
      std::memcmp(original.data(), decompressed->as<char>(), original.size()));
}

TEST(ZstdCompressorTest, uncompressedSize) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  ZstdCompressor compressor;
  TestCompressionPolicy policy;

  std::vector<char> original(512);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<char>(i % 7);
  }
  std::string_view input(original.data(), original.size());

  auto result = compressor.compress(*pool, input, DataType::Int32, 0, policy);
  ASSERT_EQ(CompressionType::Zstd, result.compressionType);
  ASSERT_TRUE(result.buffer.has_value());

  std::string_view compressed(result.buffer->data(), result.buffer->size());
  auto size = compressor.uncompressedSize(compressed);
  ASSERT_TRUE(size.has_value());
  EXPECT_EQ(original.size(), size.value());
}

TEST(ZstdCompressorTest, incompressibleData) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  ZstdCompressor compressor;
  TestCompressionPolicy policy;

  // Generate random data that doesn't compress well
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<char> randomData(256);
  for (auto& byte : randomData) {
    byte = static_cast<char>(dist(rng));
  }
  std::string_view input(randomData.data(), randomData.size());

  auto result = compressor.compress(*pool, input, DataType::Int8, 0, policy);
  // Random data should not compress well; compressor returns Uncompressed
  EXPECT_EQ(CompressionType::Uncompressed, result.compressionType);
  EXPECT_FALSE(result.buffer.has_value());
}

TEST(ZstdCompressorTest, declinedByPolicy) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  ZstdCompressor compressor;
  RejectingCompressionPolicy policy;

  // Highly compressible data that would otherwise be kept.
  std::vector<char> original(1024);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<char>(i % 10);
  }
  std::string_view input(original.data(), original.size());

  auto result = compressor.compress(*pool, input, DataType::Int8, 0, policy);
  // The compressor honors shouldAccept() and declines despite compressibility.
  EXPECT_EQ(CompressionType::Uncompressed, result.compressionType);
  EXPECT_FALSE(result.buffer.has_value());
}

TEST(ZstdCompressorTest, roundTripVariousDataTypes) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  ZstdCompressor compressor;
  TestCompressionPolicy policy;

  // Test with int32 data (repeating pattern to ensure compressibility)
  std::vector<int32_t> int32Data(256, 42);
  std::string_view input(
      reinterpret_cast<const char*>(int32Data.data()),
      int32Data.size() * sizeof(int32_t));

  auto result = compressor.compress(*pool, input, DataType::Int32, 0, policy);
  EXPECT_EQ(CompressionType::Zstd, result.compressionType);
  ASSERT_TRUE(result.buffer.has_value());

  std::string_view compressed(result.buffer->data(), result.buffer->size());
  auto decompressed = compressor.uncompress(
      *pool, CompressionType::Zstd, DataType::Int8, compressed);
  ASSERT_EQ(input.size(), decompressed->size());
  EXPECT_EQ(
      0, std::memcmp(input.data(), decompressed->as<char>(), input.size()));
}

TEST(ZstdCompressorTest, minCompressionSizeSkipsSmallData) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  ZstdCompressor compressor;

  std::vector<char> data(50, 'a');
  std::string_view input(data.data(), data.size());

  // Use CompressionEncoder with a minCompressionSize larger than data
  TestCompressionPolicy policy(data.size() + 1);
  CompressionEncoder<int8_t> encoder{*pool, policy, DataType::Int8, input};
  EXPECT_EQ(CompressionType::Uncompressed, encoder.compressionType());
}

TEST(ZstdCompressorTest, allocateBufferFromPool) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  facebook::velox::BufferPool bufferPool{
      facebook::velox::BufferPool::kDefaultCapacity};

  // Pool is empty — should fall back to MemoryPool allocation.
  auto buf1 = allocateBuffer(*pool, &bufferPool, 100);
  ASSERT_NE(buf1, nullptr);
  EXPECT_GE(buf1->capacity(), 100);
  EXPECT_EQ(buf1->size(), 100);

  // Return the buffer to the pool.
  bufferPool.release(std::move(buf1));
  EXPECT_EQ(bufferPool.size(), 1);

  // Now the pool has a buffer — should reuse it.
  auto buf2 = allocateBuffer(*pool, &bufferPool, 50);
  ASSERT_NE(buf2, nullptr);
  EXPECT_GE(buf2->capacity(), 100);
  EXPECT_EQ(buf2->size(), 50);
  EXPECT_EQ(bufferPool.size(), 0);
}

TEST(ZstdCompressorTest, allocateBufferNullPool) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();

  // Null pool — should allocate from MemoryPool.
  auto buf = allocateBuffer(*pool, nullptr, 200);
  ASSERT_NE(buf, nullptr);
  EXPECT_GE(buf->capacity(), 200);
  EXPECT_EQ(buf->size(), 200);
}

TEST(ZstdCompressorTest, uncompressWithBufferPool) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  ZstdCompressor compressor;
  TestCompressionPolicy policy;
  facebook::velox::BufferPool bufferPool{
      facebook::velox::BufferPool::kDefaultCapacity};

  std::vector<char> original(1024);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<char>(i % 10);
  }
  std::string_view input(original.data(), original.size());

  auto result = compressor.compress(*pool, input, DataType::Int8, 0, policy);
  ASSERT_TRUE(result.buffer.has_value());
  std::string_view compressed(result.buffer->data(), result.buffer->size());

  // Decompress with BufferPool — first call allocates fresh.
  auto decompressed = compressor.uncompress(
      *pool, CompressionType::Zstd, DataType::Int8, compressed, &bufferPool);
  ASSERT_EQ(original.size(), decompressed->size());
  EXPECT_EQ(
      0,
      std::memcmp(original.data(), decompressed->as<char>(), original.size()));

  // Return buffer to pool, then decompress again — should reuse.
  bufferPool.release(std::move(decompressed));
  EXPECT_EQ(bufferPool.size(), 1);

  auto decompressed2 = compressor.uncompress(
      *pool, CompressionType::Zstd, DataType::Int8, compressed, &bufferPool);
  EXPECT_EQ(bufferPool.size(), 0);
  ASSERT_EQ(original.size(), decompressed2->size());
  EXPECT_EQ(
      0,
      std::memcmp(original.data(), decompressed2->as<char>(), original.size()));
}

TEST(ZstdCompressorTest, bufferPoolReuseAcrossMultipleCycles) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  ZstdCompressor compressor;
  TestCompressionPolicy policy;
  facebook::velox::BufferPool bufferPool{
      facebook::velox::BufferPool::kDefaultCapacity};

  std::vector<char> original(512);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<char>(i % 13);
  }
  std::string_view input(original.data(), original.size());

  auto result = compressor.compress(*pool, input, DataType::Int32, 0, policy);
  ASSERT_TRUE(result.buffer.has_value());
  std::string_view compressed(result.buffer->data(), result.buffer->size());

  // Simulate multiple encoding lifecycles: decompress → use → release → repeat.
  for (int cycle = 0; cycle < 5; ++cycle) {
    auto buf = compressor.uncompress(
        *pool, CompressionType::Zstd, DataType::Int32, compressed, &bufferPool);
    ASSERT_EQ(original.size(), buf->size());
    EXPECT_EQ(
        0, std::memcmp(original.data(), buf->as<char>(), original.size()));
    bufferPool.release(std::move(buf));
  }
  EXPECT_EQ(bufferPool.size(), 1);
}
