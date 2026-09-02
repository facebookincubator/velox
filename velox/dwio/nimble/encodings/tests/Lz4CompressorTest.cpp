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
#include "velox/dwio/nimble/compression/Lz4Compressor.h"
#include <gtest/gtest.h>
#include <cstring>
#include <random>
#include "velox/common/memory/Memory.h"

using namespace facebook::nimble;

namespace {

class TestLz4CompressionPolicy : public CompressionPolicy {
 public:
  explicit TestLz4CompressionPolicy(uint64_t minCompressionSize = 0) {
    compressionInfo_ = {
        .compressionType = CompressionType::Lz4,
        .minCompressionSize = minCompressionSize};
    compressionInfo_.parameters.lz4.accelerationLevel = 1;
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
class RejectingLz4CompressionPolicy : public CompressionPolicy {
 public:
  CompressionConfig config() const override {
    CompressionConfig config{.compressionType = CompressionType::Lz4};
    config.parameters.lz4.accelerationLevel = 1;
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

TEST(Lz4CompressorTest, compressionType) {
  Lz4Compressor compressor;
  EXPECT_EQ(CompressionType::Lz4, compressor.compressionType());
}

TEST(Lz4CompressorTest, roundTrip) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Lz4Compressor compressor;
  TestLz4CompressionPolicy policy;

  std::vector<char> original(1024);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<char>(i % 10);
  }
  std::string_view input(original.data(), original.size());

  auto result = compressor.compress(*pool, input, DataType::Int8, 0, policy);
  EXPECT_EQ(CompressionType::Lz4, result.compressionType);
  ASSERT_TRUE(result.buffer.has_value());

  std::string_view compressed(result.buffer->data(), result.buffer->size());

  auto decompressed = compressor.uncompress(
      *pool, CompressionType::Lz4, DataType::Int8, compressed);
  ASSERT_EQ(original.size(), decompressed->size());
  EXPECT_EQ(
      0,
      std::memcmp(original.data(), decompressed->as<char>(), original.size()));
}

TEST(Lz4CompressorTest, uncompressedSize) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Lz4Compressor compressor;
  TestLz4CompressionPolicy policy;

  std::vector<char> original(512);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<char>(i % 7);
  }
  std::string_view input(original.data(), original.size());

  auto result = compressor.compress(*pool, input, DataType::Int32, 0, policy);
  ASSERT_EQ(CompressionType::Lz4, result.compressionType);
  ASSERT_TRUE(result.buffer.has_value());

  std::string_view compressed(result.buffer->data(), result.buffer->size());
  auto size = compressor.uncompressedSize(compressed);
  ASSERT_TRUE(size.has_value());
  EXPECT_EQ(original.size(), size.value());
}

TEST(Lz4CompressorTest, incompressibleData) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Lz4Compressor compressor;
  TestLz4CompressionPolicy policy;

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<char> randomData(256);
  for (auto& byte : randomData) {
    byte = static_cast<char>(dist(rng));
  }
  std::string_view input(randomData.data(), randomData.size());

  auto result = compressor.compress(*pool, input, DataType::Int8, 0, policy);
  EXPECT_EQ(CompressionType::Uncompressed, result.compressionType);
  EXPECT_FALSE(result.buffer.has_value());
}

TEST(Lz4CompressorTest, declinedByPolicy) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Lz4Compressor compressor;
  RejectingLz4CompressionPolicy policy;

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

TEST(Lz4CompressorTest, roundTripVariousDataTypes) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Lz4Compressor compressor;
  TestLz4CompressionPolicy policy;

  std::vector<int32_t> int32Data(256, 42);
  std::string_view input(
      reinterpret_cast<const char*>(int32Data.data()),
      int32Data.size() * sizeof(int32_t));

  auto result = compressor.compress(*pool, input, DataType::Int32, 0, policy);
  EXPECT_EQ(CompressionType::Lz4, result.compressionType);
  ASSERT_TRUE(result.buffer.has_value());

  std::string_view compressed(result.buffer->data(), result.buffer->size());
  auto decompressed = compressor.uncompress(
      *pool, CompressionType::Lz4, DataType::Int8, compressed);
  ASSERT_EQ(input.size(), decompressed->size());
  EXPECT_EQ(
      0, std::memcmp(input.data(), decompressed->as<char>(), input.size()));
}

TEST(Lz4CompressorTest, minCompressionSizeSkipsSmallData) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Lz4Compressor compressor;

  std::vector<char> data(50, 'a');
  std::string_view input(data.data(), data.size());

  TestLz4CompressionPolicy policy(data.size() + 1);
  CompressionEncoder<int8_t> encoder{*pool, policy, DataType::Int8, input};
  EXPECT_EQ(CompressionType::Uncompressed, encoder.compressionType());
}
