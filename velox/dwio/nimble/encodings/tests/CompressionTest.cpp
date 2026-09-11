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
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Constants.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

#include <vector>

using namespace facebook;
using namespace facebook::nimble::test;

namespace facebook::nimble::test {

class TestCompressionPolicy : public nimble::CompressionPolicy {
 public:
  explicit TestCompressionPolicy(
      nimble::CompressionType compressionType,
      uint64_t minCompressionSize) {
    EXPECT_TRUE(
        compressionType == nimble::CompressionType::Zstd ||
        compressionType == nimble::CompressionType::MetaInternal ||
        compressionType == nimble::CompressionType::Lz4);

    compressionInfo_ = {
        .compressionType = compressionType,
        .minCompressionSize = minCompressionSize};

    compressionInfo_.parameters.zstd.compressionLevel = 3;
    compressionInfo_.parameters.lz4.accelerationLevel = 1;
    compressionInfo_.parameters.metaInternal.compressionLevel = 4;
    compressionInfo_.parameters.metaInternal.decompressionLevel = 2;
  }

  nimble::CompressionConfig config() const override {
    return compressionInfo_;
  }

  virtual bool shouldAccept(
      nimble::CompressionType /* compressionType */,
      uint64_t uncompressedSize,
      uint64_t compressedSize) const override {
    return compressedSize < uncompressedSize;
  }

 private:
  nimble::CompressionConfig compressionInfo_;
};

template <typename T>
void assertMinCompressibleSizeMetaInternal(
    const nimble::CompressionType compressionType,
    const uint32_t expectedMinCompressibleBytes) {
  const auto pool =
      facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  bool hitUncompressedBetter = false;
  bool hitCompressedBetter = false;
  const uint32_t itemSize = std::is_same<T, std::string>::value ? 1 : sizeof(T);

  for (uint32_t i = 1; i < 100; ++i) {
    const uint32_t uncompressedSize = itemSize * i;

    // assumption is that the data with all bytes being equal is the best case
    // for compression
    std::vector<char> data(uncompressedSize);

    TestCompressionPolicy compressionPolicy{compressionType, 0};
    nimble::CompressionEncoder<T> compressionEncoder{
        *pool,
        compressionPolicy,
        nimble::TypeTraits<T>::dataType,
        {data.data(), uncompressedSize}};

    // ZStd compressor returns uncompressed data if the input is too small,
    // MetInternal returns compressed data even if the input is too small.
    const bool expectCompressedBetter =
        uncompressedSize >= expectedMinCompressibleBytes &&
        compressionEncoder.compressionType() == compressionType;
    const auto compressedSize = compressionEncoder.getSize();
    EXPECT_TRUE(
        compressionEncoder.compressionType() == compressionType ||
        compressionEncoder.compressionType() == CompressionType::Uncompressed);

    if (expectCompressedBetter) {
      EXPECT_GT(uncompressedSize, compressedSize);
      hitCompressedBetter = true;
    } else {
      EXPECT_LE(uncompressedSize, compressedSize);
      hitUncompressedBetter = true;
    }
  }

  EXPECT_TRUE(hitUncompressedBetter);
  EXPECT_TRUE(hitCompressedBetter);
}
} // namespace facebook::nimble::test

template <typename C>
class CompressionTests : public ::testing::Test {};

#define TYPES int8_t, int16_t, int32_t, int64_t, double, float, std::string
using TestTypes = ::testing::Types<TYPES>;

TYPED_TEST_CASE(CompressionTests, TestTypes);

TYPED_TEST(CompressionTests, minCompressibleSizeMetaInternal) {
#ifdef DISABLE_META_INTERNAL_COMPRESSOR
  GTEST_SKIP() << "The Meta internal compressor has no OSS implementation.";
#endif
  using T = TypeParam;
  assertMinCompressibleSizeMetaInternal<T>(
      nimble::CompressionType::MetaInternal,
      nimble::kMetaInternalMinCompressionSize);
}

TYPED_TEST(CompressionTests, minCompressibleSizeZstd) {
  using T = TypeParam;
  assertMinCompressibleSizeMetaInternal<T>(
      nimble::CompressionType::Zstd, nimble::kZstdMinCompressionSize);
}

TYPED_TEST(CompressionTests, minCompressibleSizeLz4) {
  using T = TypeParam;
  assertMinCompressibleSizeMetaInternal<T>(
      nimble::CompressionType::Lz4, nimble::kLz4MinCompressionSize);
}

TEST(CompressionTests, verifyDefaultMinCompressionSize) {
  nimble::CompressionOptions compressionOptions{};
  EXPECT_EQ(
      compressionOptions.internalMinCompressionSize,
      nimble::kMetaInternalMinCompressionSize);
  EXPECT_EQ(
      compressionOptions.zstdMinCompressionSize,
      nimble::kZstdMinCompressionSize);
  EXPECT_EQ(
      compressionOptions.lz4MinCompressionSize, nimble::kLz4MinCompressionSize);
}

TEST(CompressionTests, noCompressionPolicy) {
  nimble::NoCompressionPolicy policy;

  EXPECT_EQ(
      policy.config().compressionType, nimble::CompressionType::Uncompressed);
  EXPECT_FALSE(policy.shouldAccept(
      nimble::CompressionType::Zstd,
      /*uncompressedSize=*/100,
      /*compressedSize=*/1));
}

TEST(CompressionTests, configuredCompressionPolicy) {
  auto checkCompressionType = [](nimble::CompressionType type) {
    nimble::ConfiguredCompressionPolicy policy{
        nimble::CompressionOptions{.compressionType = type},
        nimble::EncodingType::FixedBitWidth};
    EXPECT_EQ(policy.config().compressionType, type);
  };
  checkCompressionType(nimble::CompressionType::Uncompressed);
  checkCompressionType(nimble::CompressionType::Zstd);
  checkCompressionType(nimble::CompressionType::Lz4);
  checkCompressionType(nimble::CompressionType::OpenZL);
  checkCompressionType(nimble::CompressionType::MetaInternal);
}

TEST(CompressionTests, configuredCompressionPolicyUsesCompressionOptions) {
  nimble::CompressionOptions options{
      .compressionType = nimble::CompressionType::Zstd,
      .zstdMinCompressionSize = 123,
      .zstdCompressionLevel = 7,
  };

  nimble::ConfiguredCompressionPolicy policy{
      options, nimble::EncodingType::FixedBitWidth};
  const auto compression = policy.config();

  EXPECT_EQ(compression.compressionType, nimble::CompressionType::Zstd);
  EXPECT_EQ(compression.minCompressionSize, 123);
  EXPECT_EQ(compression.parameters.zstd.compressionLevel, 7);
}

TEST(CompressionTests, configuredCompressionPolicyAcceptRatioOverride) {
  nimble::ConfiguredCompressionPolicy blockBitPackingPolicy{
      nimble::CompressionOptions{}, nimble::EncodingType::BlockBitPacking};

  EXPECT_FALSE(blockBitPackingPolicy.shouldAccept(
      nimble::CompressionType::MetaInternal,
      /*uncompressedSize=*/100,
      /*compressedSize=*/80));
  EXPECT_TRUE(blockBitPackingPolicy.shouldAccept(
      nimble::CompressionType::MetaInternal,
      /*uncompressedSize=*/100,
      /*compressedSize=*/60));

  nimble::ConfiguredCompressionPolicy trivialPolicy{
      nimble::CompressionOptions{}, nimble::EncodingType::Trivial};

  EXPECT_TRUE(trivialPolicy.shouldAccept(
      nimble::CompressionType::MetaInternal,
      /*uncompressedSize=*/100,
      /*compressedSize=*/80));
}

TEST(CompressionTests, minCompresssionSizeIsApplied) {
#ifdef DISABLE_META_INTERNAL_COMPRESSOR
  GTEST_SKIP() << "The Meta internal compressor has no OSS implementation.";
#endif
  const auto pool =
      facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  const auto compressionType = nimble::CompressionType::MetaInternal;
  const uint32_t uncompressedSize = 100;
  std::vector<char> data(uncompressedSize, 0);

  {
    // make minCompressionSize slightly smaller than data to apply compression
    TestCompressionPolicy policy{compressionType, uncompressedSize - 1};
    nimble::CompressionEncoder<std::string> encoder{
        *pool,
        policy,
        nimble::DataType::String,
        {data.data(), uncompressedSize}};
    EXPECT_EQ(encoder.compressionType(), compressionType);
    EXPECT_GT(uncompressedSize, encoder.getSize());
  }

  {
    // make minCompressionSize same as the data size to apply compression
    TestCompressionPolicy policy{compressionType, uncompressedSize};
    nimble::CompressionEncoder<std::string> encoder{
        *pool,
        policy,
        nimble::DataType::String,
        {data.data(), uncompressedSize}};
    EXPECT_EQ(encoder.compressionType(), compressionType);
    EXPECT_GT(uncompressedSize, encoder.getSize());
  }

  {
    // make minCompressionSize slightly larger than data to skip compression
    TestCompressionPolicy policy{compressionType, uncompressedSize + 1};
    nimble::CompressionEncoder<std::string> encoder{
        *pool,
        policy,
        nimble::DataType::String,
        {data.data(), uncompressedSize}};
    EXPECT_EQ(encoder.compressionType(), nimble::CompressionType::Uncompressed);
  }
}
