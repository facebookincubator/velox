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

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/common/base/VeloxException.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/compression/Compression.h"
#include "velox/common/compression/Lz4Compression.h"

namespace facebook::velox::common {

namespace {
const std::shared_ptr<CodecOptions> kDefaultCodecOptions =
    std::make_shared<CodecOptions>();

struct TestParams {
  CompressionKind compressionKind;
  std::shared_ptr<CodecOptions> codecOptions;

  explicit TestParams(
      common::CompressionKind compressionKind,
      std::shared_ptr<CodecOptions> codecOptions = kDefaultCodecOptions)
      : compressionKind(compressionKind),
        codecOptions(std::move(codecOptions)) {}
};

std::vector<uint8_t> makeRandomData(size_t n) {
  std::vector<uint8_t> data(n);
  std::default_random_engine engine(42);
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  std::generate(data.begin(), data.end(), [&]() { return dist(engine); });
  return data;
}

std::vector<uint8_t> makeCompressibleData(size_t size) {
  std::string baseData = "The quick brown fox jumps over the lazy dog";
  auto repeats = static_cast<int32_t>(1 + size / baseData.size());

  std::vector<uint8_t> data(baseData.size() * repeats);
  for (int i = 0; i < repeats; ++i) {
    std::memcpy(
        data.data() + i * baseData.size(), baseData.data(), baseData.size());
  }
  data.resize(size);
  return data;
}

std::function<uint64_t()> makeRandomInputSize() {
  std::default_random_engine engine(42);
  std::uniform_int_distribution<uint64_t> sizeDistribution(10, 40);
  return [=]() mutable -> uint64_t { return sizeDistribution(engine); };
}

// Check roundtrip of one-shot compression and decompression functions.
void checkCodecRoundtrip(
    Codec* c1,
    Codec* c2,
    const std::vector<uint8_t>& data) {
  auto maxCompressedLen =
      static_cast<size_t>(c1->maxCompressedLength(data.size()));
  std::vector<uint8_t> compressed(maxCompressedLen);
  std::vector<uint8_t> decompressed(data.size());

  // Compress with codec c1.
  auto compressedSize = c1->compress(
      data.data(), data.size(), compressed.data(), maxCompressedLen);
  compressed.resize(compressedSize);

  // Decompress with codec c2.
  auto decompressedSize = c2->decompress(
      compressed.data(),
      compressed.size(),
      decompressed.data(),
      decompressed.size());

  ASSERT_EQ(data, decompressed);
  ASSERT_EQ(data.size(), decompressedSize);
}

// Use same codec for both compression and decompression.
void checkCodecRoundtrip(
    const std::unique_ptr<Codec>& codec,
    const std::vector<uint8_t>& data) {
  checkCodecRoundtrip(codec.get(), codec.get(), data);
}

// Compress with codec c1 and decompress with codec c2.
void checkCodecRoundtrip(
    const std::unique_ptr<Codec>& c1,
    const std::unique_ptr<Codec>& c2,
    const std::vector<uint8_t>& data) {
  checkCodecRoundtrip(c1.get(), c2.get(), data);
}

void streamingCompress(
    const std::shared_ptr<StreamingCompressor>& compressor,
    const std::vector<uint8_t>& uncompressed,
    std::vector<uint8_t>& compressed) {
  const uint8_t* input = uncompressed.data();
  uint64_t remaining = uncompressed.size();
  uint64_t compressedSize = 0;
  compressed.resize(10);
  bool doFlush = false;

  // Generate small random input buffer size.
  auto randomInputSize = makeRandomInputSize();

  // Continue decompressing until consuming all compressed data .
  while (remaining > 0) {
    // Feed a small amount each time.
    auto inputLength = std::min(remaining, randomInputSize());
    auto outputLength = compressed.size() - compressedSize;
    uint8_t* output = compressed.data() + compressedSize;

    // Compress once.
    auto compressResult =
        compressor->compress(input, inputLength, output, outputLength);
    ASSERT_LE(compressResult.bytesRead, inputLength);
    ASSERT_LE(compressResult.bytesWritten, outputLength);

    // Update result.
    compressedSize += compressResult.bytesWritten;
    input += compressResult.bytesRead;
    remaining -= compressResult.bytesRead;

    // Grow compressed buffer if it's too small.
    if (compressResult.outputTooSmall) {
      compressed.resize(compressed.capacity() * 2);
    }

    // Once every two iterations, do a flush.
    if (doFlush) {
      StreamingCompressor::FlushResult flushResult;
      do {
        outputLength = compressed.size() - compressedSize;
        output = compressed.data() + compressedSize;
        flushResult = compressor->flush(output, outputLength);
        ASSERT_LE(flushResult.bytesWritten, outputLength);
        compressedSize += flushResult.bytesWritten;
        if (flushResult.outputTooSmall) {
          compressed.resize(compressed.capacity() * 2);
        }
      } while (flushResult.outputTooSmall);
    }
    doFlush = !doFlush;
  }

  // End the compressed stream.
  StreamingCompressor::EndResult endResult;
  do {
    int64_t outputLength = compressed.size() - compressedSize;
    uint8_t* output = compressed.data() + compressedSize;
    endResult = compressor->end(output, outputLength);
    ASSERT_LE(endResult.bytesWritten, outputLength);
    compressedSize += endResult.bytesWritten;
    if (endResult.outputTooSmall) {
      compressed.resize(compressed.capacity() * 2);
    }
  } while (endResult.outputTooSmall);
  compressed.resize(compressedSize);
}

void streamingDecompress(
    const std::shared_ptr<StreamingDecompressor>& decompressor,
    const std::vector<uint8_t>& compressed,
    std::vector<uint8_t>& decompressed) {
  const uint8_t* input = compressed.data();
  uint64_t remaining = compressed.size();
  uint64_t decompressedSize = 0;
  decompressed.resize(10);

  // Generate small random input buffer size.
  auto ramdomInputSize = makeRandomInputSize();

  // Continue decompressing until finishes.
  while (!decompressor->isFinished()) {
    // Feed a small amount each time.
    auto inputLength = std::min(remaining, ramdomInputSize());
    auto outputLength = decompressed.size() - decompressedSize;
    uint8_t* output = decompressed.data() + decompressedSize;

    // Decompress once.
    auto result =
        decompressor->decompress(input, inputLength, output, outputLength);
    ASSERT_LE(result.bytesRead, inputLength);
    ASSERT_LE(result.bytesWritten, outputLength);
    ASSERT_TRUE(
        result.outputTooSmall || result.bytesWritten > 0 ||
        result.bytesRead > 0)
        << "Decompression not progressing anymore";

    // Update result.
    decompressedSize += result.bytesWritten;
    input += result.bytesRead;
    remaining -= result.bytesRead;

    // Grow decompressed buffer if it's too small.
    if (result.outputTooSmall) {
      decompressed.resize(decompressed.capacity() * 2);
    }
  }
  ASSERT_TRUE(decompressor->isFinished());
  ASSERT_EQ(remaining, 0);
  decompressed.resize(decompressedSize);
}

// Check the streaming compressor against one-shot decompression.
void checkStreamingCompressor(Codec* codec, const std::vector<uint8_t>& data) {
  // Run streaming compression.
  std::vector<uint8_t> compressed;
  streamingCompress(codec->makeStreamingCompressor(), data, compressed);

  // Check decompressing the compressed data.
  std::vector<uint8_t> decompressed(data.size());
  ASSERT_NO_THROW(codec->decompress(
      compressed.data(),
      compressed.size(),
      decompressed.data(),
      decompressed.size()));
  ASSERT_EQ(data, decompressed);
}

// Check the streaming decompressor against one-shot compression.
void checkStreamingDecompressor(
    Codec* codec,
    const std::vector<uint8_t>& data) {
  // Create compressed data.
  auto maxCompressedLen = codec->maxCompressedLength(data.size());
  std::vector<uint8_t> compressed(maxCompressedLen);
  auto compressedSize = codec->compress(
      data.data(), data.size(), compressed.data(), maxCompressedLen);
  compressed.resize(compressedSize);

  // Run streaming decompression.
  std::vector<uint8_t> decompressed;
  streamingDecompress(
      codec->makeStreamingDecompressor(), compressed, decompressed);

  // Check the decompressed data.
  ASSERT_EQ(data.size(), decompressed.size());
  ASSERT_EQ(data, decompressed);
}

// Check the streaming compressor and decompressor together.
void checkStreamingRoundtrip(
    const std::shared_ptr<StreamingCompressor>& compressor,
    const std::shared_ptr<StreamingDecompressor>& decompressor,
    const std::vector<uint8_t>& data) {
  std::vector<uint8_t> compressed;
  streamingCompress(compressor, data, compressed);
  std::vector<uint8_t> decompressed;
  streamingDecompress(decompressor, compressed, decompressed);
  ASSERT_EQ(data, decompressed);
}

void checkStreamingRoundtrip(Codec* codec, const std::vector<uint8_t>& data) {
  checkStreamingRoundtrip(
      codec->makeStreamingCompressor(),
      codec->makeStreamingDecompressor(),
      data);
}
} // namespace

class CompressionTest : public testing::Test {};

TEST_F(CompressionTest, testCompressionNames) {
  EXPECT_EQ("none", compressionKindToString(CompressionKind_NONE));
  EXPECT_EQ("zlib", compressionKindToString(CompressionKind_ZLIB));
  EXPECT_EQ("snappy", compressionKindToString(CompressionKind_SNAPPY));
  EXPECT_EQ("lzo", compressionKindToString(CompressionKind_LZO));
  EXPECT_EQ("lz4", compressionKindToString(CompressionKind_LZ4));
  EXPECT_EQ("zstd", compressionKindToString(CompressionKind_ZSTD));
  EXPECT_EQ("gzip", compressionKindToString(CompressionKind_GZIP));
  EXPECT_EQ(
      "unknown - 99",
      compressionKindToString(static_cast<CompressionKind>(99)));
}

TEST_F(CompressionTest, compressionKindToCodec) {
  ASSERT_EQ(
      folly::io::CodecType::NO_COMPRESSION,
      compressionKindToCodec(CompressionKind::CompressionKind_NONE)->type());
  ASSERT_EQ(
      folly::io::CodecType::LZ4,
      compressionKindToCodec(CompressionKind::CompressionKind_LZ4)->type());
  EXPECT_THROW(
      compressionKindToCodec(CompressionKind::CompressionKind_LZO),
      facebook::velox::VeloxException);
}

TEST_F(CompressionTest, stringToCompressionKind) {
  EXPECT_EQ(stringToCompressionKind("none"), CompressionKind_NONE);
  EXPECT_EQ(stringToCompressionKind("zlib"), CompressionKind_ZLIB);
  EXPECT_EQ(stringToCompressionKind("snappy"), CompressionKind_SNAPPY);
  EXPECT_EQ(stringToCompressionKind("lzo"), CompressionKind_LZO);
  EXPECT_EQ(stringToCompressionKind("lz4"), CompressionKind_LZ4);
  EXPECT_EQ(stringToCompressionKind("zstd"), CompressionKind_ZSTD);
  EXPECT_EQ(stringToCompressionKind("gzip"), CompressionKind_GZIP);
  VELOX_ASSERT_THROW(
      stringToCompressionKind("bz2"), "Not support compression kind bz2");
}

class CodecTest : public ::testing::TestWithParam<TestParams> {
 protected:
  static CompressionKind getCompressionKind() {
    return GetParam().compressionKind;
  }

  static const CodecOptions& getCodecOptions() {
    return *GetParam().codecOptions;
  }

  static std::unique_ptr<Codec> makeCodec() {
    return Codec::create(getCompressionKind(), getCodecOptions());
  }
};

TEST_P(CodecTest, specifyCompressionLevel) {
  std::vector<uint8_t> data = makeRandomData(2000);
  const auto kind = getCompressionKind();
  if (!Codec::isAvailable(kind)) {
    // Support for this codec hasn't been built.
    VELOX_ASSERT_THROW(
        Codec::create(kind, kUseDefaultCompressionLevel),
        "Support for codec '" + compressionKindToString(kind) +
            "' not implemented.");
    return;
  }
  auto compressionLevels = {
      Codec::defaultCompressionLevel(kind),
      Codec::minimumCompressionLevel(kind),
      Codec::maximumCompressionLevel(kind)};
  for (auto compressionLevel : compressionLevels) {
    auto codec = Codec::create(kind, compressionLevel);
    checkCodecRoundtrip(codec, data);
  }
}

TEST_P(CodecTest, getUncompressedLength) {
  auto codec = makeCodec();
  auto inputLength = 100;
  auto input = makeRandomData(inputLength);
  std::vector<uint8_t> compressed(codec->maxCompressedLength(input.size()));
  auto compressedLength = codec->compress(
      input.data(), inputLength, compressed.data(), compressed.size());
  compressed.resize(compressedLength);

  if (Codec::supportsGetUncompressedLength(getCompressionKind())) {
    ASSERT_EQ(
        codec->getUncompressedLength(compressedLength, compressed.data()),
        inputLength);
  } else {
    ASSERT_EQ(
        codec->getUncompressedLength(compressedLength, compressed.data()),
        std::nullopt);
  }
}

TEST_P(CodecTest, codecRoundtrip) {
  auto codec = makeCodec();
  for (int dataSize : {0, 10, 10000, 100000}) {
    checkCodecRoundtrip(codec, makeRandomData(dataSize));
    checkCodecRoundtrip(codec, makeCompressibleData(dataSize));
  }
}

TEST_P(CodecTest, streamingCompressor) {
  if (!Codec::supportsStreamingCompression(getCompressionKind())) {
    return;
  }

  for (auto dataSize : {0, 10, 10000, 100000}) {
    auto codec = makeCodec();
    checkStreamingCompressor(codec.get(), makeRandomData(dataSize));
    checkStreamingCompressor(codec.get(), makeCompressibleData(dataSize));
  }
}

TEST_P(CodecTest, streamingDecompressor) {
  if (!Codec::supportsStreamingCompression(getCompressionKind())) {
    return;
  }

  for (auto dataSize : {0, 10, 10000, 100000}) {
    auto codec = makeCodec();
    checkStreamingDecompressor(codec.get(), makeRandomData(dataSize));
    checkStreamingDecompressor(codec.get(), makeCompressibleData(dataSize));
  }
}

TEST_P(CodecTest, streamingRoundtrip) {
  if (!Codec::supportsStreamingCompression(getCompressionKind())) {
    return;
  }

  for (auto dataSize : {0, 10, 10000, 100000}) {
    auto codec = makeCodec();
    checkStreamingRoundtrip(codec.get(), makeRandomData(dataSize));
    checkStreamingRoundtrip(codec.get(), makeCompressibleData(dataSize));
  }
}

TEST_P(CodecTest, streamingDecompressorReuse) {
  if (!Codec::supportsStreamingCompression(getCompressionKind())) {
    return;
  }

  auto codec = makeCodec();
  auto decompressor = codec->makeStreamingDecompressor();
  checkStreamingRoundtrip(
      codec->makeStreamingCompressor(), decompressor, makeRandomData(100));

  // StreamingDecompressor::reset() should allow reusing decompressor for a
  // new stream.
  decompressor->reset();
  checkStreamingRoundtrip(
      codec->makeStreamingCompressor(), decompressor, makeRandomData(200));
}

INSTANTIATE_TEST_SUITE_P(
    TestLz4Frame,
    CodecTest,
    ::testing::Values(TestParams{
        CompressionKind::CompressionKind_LZ4,
        std::make_shared<Lz4CodecOptions>(Lz4CodecOptions::kLz4Frame)}));

INSTANTIATE_TEST_SUITE_P(
    TestLz4Raw,
    CodecTest,
    ::testing::Values(TestParams{
        CompressionKind::CompressionKind_LZ4,
        std::make_shared<Lz4CodecOptions>(Lz4CodecOptions::kLz4Raw)}));

INSTANTIATE_TEST_SUITE_P(
    TestLz4Hadoop,
    CodecTest,
    ::testing::Values(TestParams{
        CompressionKind::CompressionKind_LZ4,
        std::make_shared<Lz4CodecOptions>(Lz4CodecOptions::kLz4Hadoop)}));

TEST(CodecLZ4HadoopTest, compatibility) {
  // LZ4 Hadoop codec should be able to read back LZ4 raw blocks.
  auto c1 = Codec::create(
      CompressionKind::CompressionKind_LZ4,
      Lz4CodecOptions{Lz4CodecOptions::kLz4Raw});
  auto c2 = Codec::create(
      CompressionKind::CompressionKind_LZ4,
      Lz4CodecOptions{Lz4CodecOptions::kLz4Hadoop});

  for (auto dataSize : {0, 10, 10000, 100000}) {
    checkCodecRoundtrip(c1, c2, makeRandomData(dataSize));
  }
}
} // namespace facebook::velox::common
