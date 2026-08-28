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

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/index/KeyChunkDecoder.h"
#include "velox/dwio/nimble/writer/ChunkedStreamWriter.h"

namespace facebook::nimble::index {
namespace {

using velox::dwio::common::SeekableArrayInputStream;

class KeyChunkDecoderTest : public ::testing::TestWithParam<EncodingType> {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addRootPool("KeyChunkDecoderTest");
    leafPool_ = pool_->addLeafChild("leaf");
  }

  std::string encodeChunk(
      const std::vector<std::string_view>& keys,
      CompressionType compressionType = CompressionType::Uncompressed) {
    Buffer encodingBuffer{*leafPool_};
    auto policy =
        std::make_unique<ManualEncodingSelectionPolicy<std::string_view>>(
            std::vector<std::pair<EncodingType, float>>{{GetParam(), 1.0}},
            CompressionOptions{},
            std::nullopt);
    auto encoded = EncodingFactory::encode<std::string_view>(
        std::move(policy), keys, encodingBuffer);

    ChunkedStreamWriter writer{
        encodingBuffer, CompressionParams{.type = compressionType}};
    auto segments = writer.encode(encoded);
    std::string result;
    for (const auto& segment : segments) {
      result.append(segment.data(), segment.size());
    }
    return result;
  }

  // blockSize caps how many bytes each SeekableArrayInputStream::Next() returns
  // (0 means the whole buffer in one call). decodeKeyChunk branches on whether
  // the first Next() window already holds the whole chunk, so a small blockSize
  // splits the data across windows and forces its multi-buffer copy path.
  std::unique_ptr<SeekableArrayInputStream> makeStream(
      const std::string& data,
      uint64_t blockSize = 0) {
    return std::make_unique<SeekableArrayInputStream>(
        reinterpret_cast<const unsigned char*>(data.data()),
        data.size(),
        blockSize);
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
  velox::BufferPtr dataBuffer_;
};

TEST_P(KeyChunkDecoderTest, decodeAndMaterialize) {
  std::vector<std::string_view> keys = {"apple", "banana", "cherry"};
  auto chunkData = encodeChunk(keys);

  auto result = decodeKeyChunk(makeStream(chunkData), *leafPool_, dataBuffer_);
  ASSERT_NE(result, nullptr);
  ASSERT_NE(result->encoding, nullptr);
  EXPECT_EQ(result->encoding->rowCount(), 3);

  auto materialized = result->encoding->materialize(0, 3);
  EXPECT_EQ(materialized[0], "apple");
  EXPECT_EQ(materialized[1], "banana");
  EXPECT_EQ(materialized[2], "cherry");
}

TEST_P(KeyChunkDecoderTest, seekAndSelectiveMaterialize) {
  std::vector<std::string_view> keys = {"aaa", "bbb", "ccc", "ddd", "eee"};
  auto chunkData = encodeChunk(keys);

  auto result = decodeKeyChunk(makeStream(chunkData), *leafPool_, dataBuffer_);
  ASSERT_NE(result, nullptr);
  ASSERT_NE(result->encoding, nullptr);

  auto pos = result->encoding->seek("ccc", /*inclusive=*/true);
  ASSERT_TRUE(pos.has_value());
  EXPECT_EQ(pos.value(), 2);

  auto entries = result->encoding->materialize(2, 2);
  EXPECT_EQ(entries[0], "ccc");
  EXPECT_EQ(entries[1], "ddd");
}

TEST_P(KeyChunkDecoderTest, singleEntry) {
  std::vector<std::string_view> keys = {"only"};
  auto chunkData = encodeChunk(keys);

  auto result = decodeKeyChunk(makeStream(chunkData), *leafPool_, dataBuffer_);
  ASSERT_NE(result, nullptr);
  ASSERT_NE(result->encoding, nullptr);
  EXPECT_EQ(result->encoding->rowCount(), 1);

  auto materialized = result->encoding->materialize(0, 1);
  EXPECT_EQ(materialized[0], "only");
}

TEST_P(KeyChunkDecoderTest, materializeAfterMove) {
  std::vector<std::string_view> keys = {"foo", "bar", "baz"};
  auto chunkData = encodeChunk(keys);

  std::shared_ptr<DecodedKeyChunk> moved;
  {
    auto decoded =
        decodeKeyChunk(makeStream(chunkData), *leafPool_, dataBuffer_);
    moved = std::move(decoded);
  }

  ASSERT_NE(moved->encoding, nullptr);

  auto materialized = moved->encoding->materialize(0, 3);
  EXPECT_FALSE(moved->stringBuffers.empty());
  EXPECT_EQ(materialized[0], "foo");
  EXPECT_EQ(materialized[1], "bar");
  EXPECT_EQ(materialized[2], "baz");
}

TEST_P(KeyChunkDecoderTest, materializeAfterMoveAssign) {
  std::vector<std::string_view> keys1 = {"alpha", "beta"};
  std::vector<std::string_view> keys2 = {"gamma", "delta", "epsilon"};
  auto chunkData1 = encodeChunk(keys1);
  auto chunkData2 = encodeChunk(keys2);

  auto holder = decodeKeyChunk(makeStream(chunkData1), *leafPool_, dataBuffer_);
  holder = decodeKeyChunk(makeStream(chunkData2), *leafPool_, dataBuffer_);

  ASSERT_NE(holder->encoding, nullptr);
  EXPECT_EQ(holder->encoding->rowCount(), 3);

  auto materialized = holder->encoding->materialize(0, 3);
  EXPECT_FALSE(holder->stringBuffers.empty());
  EXPECT_EQ(materialized[0], "gamma");
  EXPECT_EQ(materialized[1], "delta");
  EXPECT_EQ(materialized[2], "epsilon");
}

TEST_P(KeyChunkDecoderTest, roundTripsCompressedKeyChunk) {
  // Many sorted keys sharing a long prefix so the encoded blob is compressible
  // and ChunkedStreamWriter actually emits a Zstd chunk header (it falls back
  // to Uncompressed when compression would not shrink the payload).
  const std::string prefix(64, 'a');
  std::vector<std::string> storage;
  std::vector<std::string_view> keys;
  storage.reserve(2000);
  keys.reserve(2000);
  for (int i = 0; i < 2000; ++i) {
    storage.push_back(fmt::format("{}{:05d}", prefix, i));
    keys.emplace_back(storage.back());
  }

  for (const auto compressionType :
       {CompressionType::Zstd, CompressionType::Lz4}) {
    SCOPED_TRACE(toString(compressionType));
    const auto chunkData = encodeChunk(keys, compressionType);

    // Sanity-check the setup: the chunk header's compression-type byte (the
    // last byte of the 5-byte header) must actually say the requested type,
    // otherwise the test is not exercising the decompression path.
    ASSERT_GE(chunkData.size(), kChunkHeaderSize);
    ASSERT_EQ(
        static_cast<uint8_t>(chunkData[kChunkHeaderSize - 1]),
        static_cast<uint8_t>(compressionType));

    // Exercise both compressed-payload release paths. blockSize 0 delivers the
    // whole chunk in the first Next() (zero-copy branch: releases dataStream);
    // blockSize 64 delivers it in 64-byte windows, so the first window is
    // smaller than the payload and decode stitches across windows (multi-buffer
    // branch: releases the staging copy from stringBuffers).
    for (const uint64_t blockSize : {uint64_t{0}, uint64_t{64}}) {
      SCOPED_TRACE(fmt::format("blockSize={}", blockSize));
      auto result = decodeKeyChunk(
          makeStream(chunkData, blockSize), *leafPool_, dataBuffer_);
      ASSERT_NE(result, nullptr);
      ASSERT_NE(result->encoding, nullptr);
      ASSERT_EQ(
          result->encoding->rowCount(), static_cast<uint32_t>(keys.size()));

      auto materialized =
          result->encoding->materialize(0, static_cast<uint32_t>(keys.size()));
      for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_EQ(materialized[i], keys[i]);
      }
    }
  }
}

TEST_P(KeyChunkDecoderTest, rejectsUnsupportedChunkCompression) {
  std::vector<std::string_view> keys = {"apple", "banana", "cherry"};
  auto chunkData = encodeChunk(keys); // uncompressed chunk
  ASSERT_GE(chunkData.size(), kChunkHeaderSize);
  // Rewrite the header's compression-type byte to an unsupported type.
  chunkData[kChunkHeaderSize - 1] =
      static_cast<char>(CompressionType::MetaInternal);

  NIMBLE_ASSERT_THROW(
      decodeKeyChunk(makeStream(chunkData), *leafPool_, dataBuffer_),
      "Unsupported key chunk compression type");
}

INSTANTIATE_TEST_SUITE_P(
    EncodingTypes,
    KeyChunkDecoderTest,
    ::testing::Values(EncodingType::Trivial, EncodingType::Prefix),
    [](const ::testing::TestParamInfo<EncodingType>& info) {
      return info.param == EncodingType::Trivial ? "Trivial" : "Prefix";
    });

} // namespace
} // namespace facebook::nimble::index
