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

#include "velox/common/base/VeloxException.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/compression/v2/Compression.h"
#include "velox/common/compression/v2/GzipCompression.h"

namespace facebook::velox::common {

namespace {

const std::shared_ptr<CodecOptions> kDefaultCodecOptions =
    std::make_shared<CodecOptions>();

struct TestParam {
  CompressionKind compressionKind;
  std::shared_ptr<CodecOptions> codecOptions;

  TestParam(
      common::CompressionKind compressionKind,
      std::shared_ptr<CodecOptions> codecOptions = kDefaultCodecOptions)
      : compressionKind(compressionKind), codecOptions(codecOptions) {}
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
      data.size(), data.data(), maxCompressedLen, compressed.data());
  compressed.resize(compressedSize);

  // Decompress with codec c2.
  auto decompressedSize = c2->decompress(
      compressed.size(),
      compressed.data(),
      decompressed.size(),
      decompressed.data());

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
    const std::shared_ptr<Compressor>& compressor,
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
        compressor->compress(inputLength, input, outputLength, output);
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
      Compressor::FlushResult flushResult;
      do {
        outputLength = compressed.size() - compressedSize;
        output = compressed.data() + compressedSize;
        flushResult = compressor->flush(outputLength, output);
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
  Compressor::EndResult endResult;
  do {
    int64_t output_len = compressed.size() - compressedSize;
    uint8_t* output = compressed.data() + compressedSize;
    endResult = compressor->end(output_len, output);
    ASSERT_LE(endResult.bytesWritten, output_len);
    compressedSize += endResult.bytesWritten;
    if (endResult.outputTooSmall) {
      compressed.resize(compressed.capacity() * 2);
    }
  } while (endResult.outputTooSmall);
  compressed.resize(compressedSize);
}

void streamingDecompress(
    const std::shared_ptr<Decompressor>& decompressor,
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
        decompressor->decompress(inputLength, input, outputLength, output);
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

// Check the streaming compressor against one-shot decompression
void checkStreamingCompressor(Codec* codec, const std::vector<uint8_t>& data) {
  // Run streaming compression.
  std::vector<uint8_t> compressed;
  streamingCompress(codec->makeCompressor(), data, compressed);
  // Check decompressing the compressed data.
  std::vector<uint8_t> decompressed(data.size());
  ASSERT_NO_THROW(codec->decompress(
      compressed.size(),
      compressed.data(),
      decompressed.size(),
      decompressed.data()));
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
      data.size(), data.data(), maxCompressedLen, compressed.data());
  compressed.resize(compressedSize);
  // Run streaming decompression.
  std::vector<uint8_t> decompressed;
  streamingDecompress(codec->makeDecompressor(), compressed, decompressed);
  // Check the decompressed data.
  ASSERT_EQ(data.size(), decompressed.size());
  ASSERT_EQ(data, decompressed);
}

// Check the streaming compressor and decompressor together.
void checkStreamingRoundtrip(
    const std::shared_ptr<Compressor>& compressor,
    const std::shared_ptr<Decompressor>& decompressor,
    const std::vector<uint8_t>& data) {
  std::vector<uint8_t> compressed;
  streamingCompress(compressor, data, compressed);
  std::vector<uint8_t> decompressed;
  streamingDecompress(decompressor, compressed, decompressed);
  ASSERT_EQ(data, decompressed);
}

void checkStreamingRoundtrip(Codec* codec, const std::vector<uint8_t>& data) {
  checkStreamingRoundtrip(
      codec->makeCompressor(), codec->makeDecompressor(), data);
}
} // namespace

class CodecTest : public ::testing::TestWithParam<TestParam> {
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
        Codec::create(kind, Codec::useDefaultCompressionLevel()),
        "Support for codec '" + compressionKindToString(kind) +
            "' not implemented.");
  } else if (!Codec::supportsCompressionLevel(kind)) {
    VELOX_ASSERT_THROW(
        Codec::create(kind, 1),
        fmt::format(
            "Codec '{}' doesn't support setting a compression level.",
            compressionKindToString(kind)));
  } else {
    auto codec = Codec::create(kind, Codec::minimumCompressionLevel(kind));
    checkCodecRoundtrip(codec, data);
  }
}

TEST_P(CodecTest, getUncompressedLength) {
  auto codec = makeCodec();
  // Test non-empty input.
  {
    auto inputLength = 100;
    auto input = makeRandomData(inputLength);
    std::vector<uint8_t> compressed(codec->maxCompressedLength(input.size()));
    auto compressedLength = codec->compress(
        inputLength, input.data(), compressed.size(), compressed.data());
    compressed.resize(compressedLength);

    if (Codec::supportsGetUncompressedLength(getCompressionKind())) {
      auto uncompressedLength =
          codec->getUncompressedLength(compressedLength, compressed.data());
      ASSERT_EQ(
          codec->getUncompressedLength(compressedLength, compressed.data()),
          inputLength);
      ASSERT_EQ(
          codec->getUncompressedLength(
              compressedLength, compressed.data(), inputLength),
          inputLength);
      ASSERT_EQ(
          codec->getUncompressedLength(
              compressedLength, compressed.data(), std::nullopt),
          inputLength);
      VELOX_ASSERT_THROW(
          codec->getUncompressedLength(
              compressedLength, compressed.data(), inputLength + 1),
          fmt::format("Invalid uncompressed length: {}", inputLength + 1));
    } else {
      ASSERT_EQ(
          codec->getUncompressedLength(input.size(), input.data()),
          std::nullopt);
      ASSERT_EQ(
          codec->getUncompressedLength(
              input.size(), input.data(), std::nullopt),
          std::nullopt);
      ASSERT_EQ(codec->getUncompressedLength(input.size(), input.data(), 0), 0);
      ASSERT_EQ(codec->getUncompressedLength(input.size(), input.data(), 2), 2);
    }
  }
  // Test empty input.
  {
    std::vector<uint8_t> input{};
    ASSERT_EQ(codec->getUncompressedLength(0, input.data(), 0), 0);
    ASSERT_EQ(codec->getUncompressedLength(0, input.data(), std::nullopt), 0);
    VELOX_ASSERT_THROW(
        codec->getUncompressedLength(0, input.data(), 1),
        fmt::format("Invalid uncompressed length: {}", 1));
  }
}

TEST_P(CodecTest, codecRoundtrip) {
  auto codec = makeCodec();
  for (int dataSize : {0, 10, 10000, 100000}) {
    checkCodecRoundtrip(codec, makeRandomData(dataSize));
    checkCodecRoundtrip(codec, makeCompressibleData(dataSize));
  }
}

TEST_P(CodecTest, minMaxCompressionLevel) {
  auto type = getCompressionKind();
  auto codec = makeCodec();
  auto notSupportCompressionLevel = [](CompressionKind kind) {
    return fmt::format(
        "Codec '{}' doesn't support setting a compression level.",
        compressionKindToString(kind));
  };

  if (Codec::supportsCompressionLevel(type)) {
    auto minLevel = Codec::minimumCompressionLevel(type);
    auto maxLevel = Codec::maximumCompressionLevel(type);
    auto defaultLevel = Codec::defaultCompressionLevel(type);
    ASSERT_NE(minLevel, Codec::useDefaultCompressionLevel());
    ASSERT_NE(maxLevel, Codec::useDefaultCompressionLevel());
    ASSERT_NE(defaultLevel, Codec::useDefaultCompressionLevel());
    ASSERT_LT(minLevel, maxLevel);
    ASSERT_EQ(minLevel, codec->minimumCompressionLevel());
    ASSERT_EQ(maxLevel, codec->maximumCompressionLevel());
    ASSERT_GE(defaultLevel, minLevel);
    ASSERT_LE(defaultLevel, maxLevel);
  } else {
    VELOX_ASSERT_THROW(
        Codec::minimumCompressionLevel(type), notSupportCompressionLevel(type));
    VELOX_ASSERT_THROW(
        Codec::maximumCompressionLevel(type), notSupportCompressionLevel(type));
    VELOX_ASSERT_THROW(
        Codec::defaultCompressionLevel(type), notSupportCompressionLevel(type));
    ASSERT_EQ(
        codec->minimumCompressionLevel(), Codec::useDefaultCompressionLevel());
    ASSERT_EQ(
        codec->maximumCompressionLevel(), Codec::useDefaultCompressionLevel());
    ASSERT_EQ(
        codec->defaultCompressionLevel(), Codec::useDefaultCompressionLevel());
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
  auto decompressor = codec->makeDecompressor();
  checkStreamingRoundtrip(
      codec->makeCompressor(), decompressor, makeRandomData(100));

  // Decompressor::reset() should allow reusing decompressor for a new stream.
  decompressor->reset();
  checkStreamingRoundtrip(
      codec->makeCompressor(), decompressor, makeRandomData(200));
}

std::vector<TestParam> getGzipTestParams() {
  std::vector<TestParam> params;
  for (auto windowBits : {kGzipDefaultWindowBits, kGzip4KBWindowBits}) {
    for (auto format : {GzipFormat::kGzip, GzipFormat::kDeflate}) {
      params.emplace_back(
          CompressionKind::CompressionKind_ZLIB,
          std::make_shared<GzipCodecOptions>(
              kUseDefaultCompressionLevel, format, windowBits));
    }
    params.emplace_back(
        CompressionKind::CompressionKind_ZLIB,
        std::make_shared<GzipCodecOptions>(
            kUseDefaultCompressionLevel, GzipFormat::kZlib, windowBits));
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(
    TestLZ4Frame,
    CodecTest,
    ::testing::Values(CompressionKind::CompressionKind_LZ4));
INSTANTIATE_TEST_SUITE_P(
    TestLZ4Raw,
    CodecTest,
    ::testing::Values(CompressionKind::CompressionKind_LZ4RAW));
INSTANTIATE_TEST_SUITE_P(
    TestLZ4Hadoop,
    CodecTest,
    ::testing::Values(CompressionKind::CompressionKind_LZ4HADOOP));
INSTANTIATE_TEST_SUITE_P(
    TestGzip,
    CodecTest,
    ::testing::ValuesIn(getGzipTestParams()));

TEST(CodecLZ4HadoopTest, compatibility) {
  // LZ4 Hadoop codec should be able to read back LZ4 raw blocks.
  auto c1 = Codec::create(CompressionKind::CompressionKind_LZ4RAW);
  auto c2 = Codec::create(CompressionKind::CompressionKind_LZ4HADOOP);

  for (auto dataSize : {0, 10, 10000, 100000}) {
    checkCodecRoundtrip(c1, c2, makeRandomData(dataSize));
  }
}
} // namespace facebook::velox::common
