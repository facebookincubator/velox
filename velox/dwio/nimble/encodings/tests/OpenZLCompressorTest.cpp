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
#include "velox/dwio/nimble/compression/OpenZLCompressor.h"
#include <gtest/gtest.h>
#include <cstring>
#include <random>
#include <vector>
#include "velox/buffer/BufferPool.h"
#include "velox/common/memory/Memory.h"

using namespace facebook::nimble;

namespace {

class TestOpenZLCompressionPolicy : public CompressionPolicy {
 public:
  explicit TestOpenZLCompressionPolicy(uint64_t minCompressionSize = 0) {
    compressionConfig_ = {
        .compressionType = CompressionType::OpenZL,
        .minCompressionSize = minCompressionSize};
  }

  CompressionConfig config() const override {
    return compressionConfig_;
  }

  bool shouldAccept(
      CompressionType /* compressionType */,
      uint64_t uncompressedSize,
      uint64_t compressedSize) const override {
    return compressedSize < uncompressedSize;
  }

 private:
  CompressionConfig compressionConfig_;
};

// Builds a serial byte buffer from a numeric vector for feeding into the
// compressor.
template <typename T>
std::string_view asBytes(const std::vector<T>& values) {
  return std::string_view(
      reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
}

// Round-trips compressible numeric data through OpenZL and asserts the bytes
// come back identical.
template <typename T>
void roundTripNumeric(DataType dataType, const std::vector<T>& values) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  OpenZLCompressor compressor;
  TestOpenZLCompressionPolicy policy;

  auto input = asBytes(values);
  auto result = compressor.compress(*pool, input, dataType, 0, policy);
  ASSERT_EQ(CompressionType::OpenZL, result.compressionType);
  ASSERT_TRUE(result.buffer.has_value());

  std::string_view compressed(result.buffer->data(), result.buffer->size());
  auto decompressed = compressor.uncompress(
      *pool, CompressionType::OpenZL, dataType, compressed);
  ASSERT_EQ(input.size(), decompressed->size());
  EXPECT_EQ(
      0, std::memcmp(input.data(), decompressed->as<char>(), input.size()));
}

} // namespace

TEST(OpenZLCompressorTest, CompressionType) {
  OpenZLCompressor compressor;
  EXPECT_EQ(CompressionType::OpenZL, compressor.compressionType());
}

TEST(OpenZLCompressorTest, RoundTripInt8) {
  std::vector<int8_t> values(4096);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<int8_t>(i % 10);
  }
  roundTripNumeric(DataType::Int8, values);
}

TEST(OpenZLCompressorTest, RoundTripInt32) {
  // A narrow, locally-correlated range that exercises range-pack and delta.
  std::vector<int32_t> values(4096);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<int32_t>(1000 + (i % 50));
  }
  roundTripNumeric(DataType::Int32, values);
}

TEST(OpenZLCompressorTest, RoundTripInt64) {
  std::vector<int64_t> values(4096);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<int64_t>(i / 4);
  }
  roundTripNumeric(DataType::Int64, values);
}

TEST(OpenZLCompressorTest, RoundTripFloat) {
  std::vector<float> values(4096);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i % 16);
  }
  roundTripNumeric(DataType::Float, values);
}

TEST(OpenZLCompressorTest, RoundTripSerialInput) {
  // Feed raw serial bytes tagged as Int8 (single-byte path -> zstd backend).
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  OpenZLCompressor compressor;
  TestOpenZLCompressionPolicy policy;

  std::vector<char> original(2048);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<char>(i % 7);
  }
  std::string_view input(original.data(), original.size());

  auto result = compressor.compress(*pool, input, DataType::Int8, 0, policy);
  ASSERT_EQ(CompressionType::OpenZL, result.compressionType);
  ASSERT_TRUE(result.buffer.has_value());

  std::string_view compressed(result.buffer->data(), result.buffer->size());
  auto decompressed = compressor.uncompress(
      *pool, CompressionType::OpenZL, DataType::Int8, compressed);
  ASSERT_EQ(original.size(), decompressed->size());
  EXPECT_EQ(
      0,
      std::memcmp(original.data(), decompressed->as<char>(), original.size()));
}

TEST(OpenZLCompressorTest, UncompressedSize) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  OpenZLCompressor compressor;
  TestOpenZLCompressionPolicy policy;

  std::vector<int32_t> values(2048, 42);
  auto input = asBytes(values);

  auto result = compressor.compress(*pool, input, DataType::Int32, 0, policy);
  ASSERT_EQ(CompressionType::OpenZL, result.compressionType);
  ASSERT_TRUE(result.buffer.has_value());

  std::string_view compressed(result.buffer->data(), result.buffer->size());
  auto size = compressor.uncompressedSize(compressed);
  ASSERT_TRUE(size.has_value());
  EXPECT_EQ(input.size(), size.value());
}

TEST(OpenZLCompressorTest, IncompressibleData) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  OpenZLCompressor compressor;
  TestOpenZLCompressionPolicy policy;

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<char> randomData(256);
  for (auto& byte : randomData) {
    byte = static_cast<char>(dist(rng));
  }
  std::string_view input(randomData.data(), randomData.size());

  auto result = compressor.compress(*pool, input, DataType::Int8, 0, policy);
  // Random data should not compress; the policy rejects it -> Uncompressed.
  EXPECT_EQ(CompressionType::Uncompressed, result.compressionType);
  EXPECT_FALSE(result.buffer.has_value());
}

TEST(OpenZLCompressorTest, MinCompressionSizeSkipsSmallData) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();

  std::vector<char> data(50, 'a');
  std::string_view input(data.data(), data.size());

  // minCompressionSize larger than the data -> CompressionEncoder skips it.
  TestOpenZLCompressionPolicy policy(data.size() + 1);
  CompressionEncoder<int8_t> encoder{*pool, policy, DataType::Int8, input};
  EXPECT_EQ(CompressionType::Uncompressed, encoder.compressionType());
}

TEST(OpenZLCompressorTest, UncompressWithBufferPool) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  OpenZLCompressor compressor;
  TestOpenZLCompressionPolicy policy;
  facebook::velox::BufferPool bufferPool{
      facebook::velox::BufferPool::kDefaultCapacity};

  std::vector<int32_t> values(2048);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<int32_t>(i % 20);
  }
  auto input = asBytes(values);

  auto result = compressor.compress(*pool, input, DataType::Int32, 0, policy);
  ASSERT_TRUE(result.buffer.has_value());
  std::string_view compressed(result.buffer->data(), result.buffer->size());

  // First decompress allocates fresh from the pool path.
  auto decompressed = compressor.uncompress(
      *pool, CompressionType::OpenZL, DataType::Int32, compressed, &bufferPool);
  ASSERT_EQ(input.size(), decompressed->size());
  EXPECT_EQ(
      0, std::memcmp(input.data(), decompressed->as<char>(), input.size()));

  // Return to the pool, then decompress again -> should reuse.
  bufferPool.release(std::move(decompressed));
  EXPECT_EQ(bufferPool.size(), 1);

  auto decompressed2 = compressor.uncompress(
      *pool, CompressionType::OpenZL, DataType::Int32, compressed, &bufferPool);
  EXPECT_EQ(bufferPool.size(), 0);
  ASSERT_EQ(input.size(), decompressed2->size());
  EXPECT_EQ(
      0, std::memcmp(input.data(), decompressed2->as<char>(), input.size()));
}
