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
#include <cstring>
#include <string>
#include "folly/Random.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

namespace {

struct EncodingTestAccess : nimble::Encoding {
  using nimble::Encoding::serializePrefix;
  using nimble::Encoding::serializePrefixSize;
};

} // namespace

template <bool UseVarint>
struct FBWTestConfig {
  static constexpr bool useVarint = UseVarint;
};

template <typename Config>
class FixedBitWidthEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  nimble::Vector<uint32_t> toVector(std::initializer_list<uint32_t> l) {
    nimble::Vector<uint32_t> v{pool_.get()};
    v.insert(v.end(), l.begin(), l.end());
    return v;
  }

  std::unique_ptr<nimble::Encoding> createEncoding(
      const nimble::Vector<uint32_t>& values,
      const nimble::Encoding::Options& options = {}) {
    return nimble::test::Encoder<nimble::FixedBitWidthEncoding<uint32_t>>::
        createEncoding(
            *buffer_,
            values,
            stringBufferFactory(),
            nimble::CompressionType::Uncompressed,
            options);
  }

  std::string encodeByteRounded(
      const nimble::Vector<uint32_t>& values,
      const nimble::Encoding::Options& options = {}) {
    constexpr uint8_t kByteRoundedBits{8};
    constexpr uint32_t kBaseline{0};
    const auto payloadSize =
        nimble::FixedBitArray::bufferSize(values.size(), kByteRoundedBits);
    const auto encodingSize = EncodingTestAccess::serializePrefixSize(
                                  values.size(), options.useVarintRowCount) +
        sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint8_t) + payloadSize;
    std::string serialized(encodingSize, '\0');
    char* pos = serialized.data();
    EncodingTestAccess::serializePrefix(
        nimble::EncodingType::FixedBitWidth,
        nimble::TypeTraits<uint32_t>::dataType,
        values.size(),
        options.useVarintRowCount,
        pos);
    nimble::encoding::writeChar(
        static_cast<char>(nimble::CompressionType::Uncompressed), pos);
    nimble::encoding::write(kBaseline, pos);
    nimble::encoding::writeChar(kByteRoundedBits, pos);
    std::memset(pos, 0, payloadSize);
    nimble::FixedBitArray fixedBitArray(pos, kByteRoundedBits);
    fixedBitArray.bulkSetWithBaseline(
        /*start=*/0, values.size(), values.data(), kBaseline);
    return serialized;
  }

  std::function<void*(uint32_t)> stringBufferFactory() {
    return [this](uint32_t totalLength) {
      auto& buf = stringBuffers_.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
      return buf->template asMutable<void>();
    };
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
  std::vector<velox::BufferPtr> stringBuffers_;
};

using TestTypes = ::testing::Types<FBWTestConfig<false>, FBWTestConfig<true>>;

TYPED_TEST_CASE(FixedBitWidthEncodingTest, TestTypes);

TYPED_TEST(FixedBitWidthEncodingTest, serializeThenDeserialize) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto encoding = this->createEncoding(values, options);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::FixedBitWidth);
  EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<uint32_t>::dataType);
  EXPECT_EQ(encoding->rowCount(), 10);

  nimble::Vector<uint32_t> result(this->pool_.get(), 10);
  encoding->materialize(10, result.data());
  for (uint32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(
    FixedBitWidthEncodingTest,
    usesByteRoundedBitsByDefaultAndExactBitsWhenEnabled) {
  auto values = this->toVector(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52});
  nimble::Encoding::Options defaultOptions;
  defaultOptions.useVarintRowCount = TypeParam::useVarint;
  nimble::Encoding::Options exactOptions;
  exactOptions.useVarintRowCount = TypeParam::useVarint;
  exactOptions.fixedBitWidthUseExactBits = true;

  nimble::Buffer defaultBuffer{*this->pool_};
  const auto defaultSerialized =
      nimble::test::Encoder<nimble::FixedBitWidthEncoding<uint32_t>>::encode(
          defaultBuffer,
          values,
          nimble::CompressionType::Uncompressed,
          defaultOptions);

  const auto defaultPrefixSize = nimble::EncodingPrefix::prefixSize(
      defaultSerialized, TypeParam::useVarint);
  EXPECT_EQ(
      static_cast<uint8_t>(
          defaultSerialized[defaultPrefixSize + sizeof(uint8_t) + 4]),
      8);

  nimble::Buffer exactBuffer{*this->pool_};
  const auto exactSerialized =
      nimble::test::Encoder<nimble::FixedBitWidthEncoding<uint32_t>>::encode(
          exactBuffer,
          values,
          nimble::CompressionType::Uncompressed,
          exactOptions);

  const auto exactPrefixSize =
      nimble::EncodingPrefix::prefixSize(exactSerialized, TypeParam::useVarint);
  EXPECT_EQ(
      static_cast<uint8_t>(
          exactSerialized[exactPrefixSize + sizeof(uint8_t) + 4]),
      6);

  auto encoding = nimble::EncodingFactory{defaultOptions}.create(
      *this->pool_, defaultSerialized, this->stringBufferFactory());
  std::vector<uint32_t> result(values.size());
  encoding->materialize(values.size(), result.data());
  EXPECT_EQ(result, std::vector<uint32_t>(values.begin(), values.end()));
}

TYPED_TEST(FixedBitWidthEncodingTest, decodesByteRoundedPayload) {
  auto values = this->toVector(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52});
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  const auto serialized = this->encodeByteRounded(values, options);

  const auto prefixSize =
      nimble::EncodingPrefix::prefixSize(serialized, TypeParam::useVarint);
  EXPECT_EQ(
      static_cast<uint8_t>(serialized[prefixSize + sizeof(uint8_t) + 4]), 8);

  auto encoding = nimble::EncodingFactory{options}.create(
      *this->pool_, serialized, this->stringBufferFactory());
  std::vector<uint32_t> result(values.size());
  encoding->materialize(values.size(), result.data());
  EXPECT_EQ(result, std::vector<uint32_t>(values.begin(), values.end()));
}

TYPED_TEST(FixedBitWidthEncodingTest, slice) {
  struct Range {
    uint32_t offset;
    uint32_t length;
  };

  for (const bool useExactBits : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useExactBits=" << useExactBits);
    nimble::Encoding::Options options{
        .useVarintRowCount = TypeParam::useVarint};
    options.fixedBitWidthUseExactBits = useExactBits;

    nimble::Vector<uint32_t> values{this->pool_.get()};
    for (uint32_t i = 0; i < 32; ++i) {
      values.push_back(1000 + (i * 7) % 64);
    }

    nimble::Buffer sourceBuffer{*this->pool_};
    const auto encoded =
        nimble::test::Encoder<nimble::FixedBitWidthEncoding<uint32_t>>::encode(
            sourceBuffer,
            values,
            nimble::CompressionType::Uncompressed,
            options);

    for (const auto range :
         {Range{/*offset=*/0, /*length=*/5},
          Range{/*offset=*/1, /*length=*/7},
          Range{/*offset=*/4, /*length=*/8},
          Range{/*offset=*/27, /*length=*/5}}) {
      SCOPED_TRACE(
          testing::Message()
          << "offset=" << range.offset << " length=" << range.length);
      nimble::Buffer sliceBuffer{*this->pool_};
      const auto sliced = nimble::EncodingFactory::slice(
          encoded, range.offset, range.length, sliceBuffer, options);

      EXPECT_EQ(
          nimble::EncodingPrefix::encodingType(sliced),
          nimble::EncodingType::FixedBitWidth);
      EXPECT_EQ(
          nimble::EncodingPrefix::readRowCount(
              sliced, options.useVarintRowCount),
          range.length);

      auto encoding = nimble::EncodingFactory{options}.create(
          *this->pool_, sliced, this->stringBufferFactory());
      std::vector<uint32_t> result(range.length);
      encoding->materialize(range.length, result.data());

      const std::vector<uint32_t> expected{
          values.begin() + range.offset,
          values.begin() + range.offset + range.length};
      EXPECT_EQ(result, expected);
    }
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, rejectsZeroLengthSlice) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  nimble::Vector<uint32_t> values{this->pool_.get()};
  for (uint32_t i = 0; i < 32; ++i) {
    values.push_back(1000 + i);
  }

  nimble::Buffer sourceBuffer{*this->pool_};
  const auto encoded =
      nimble::test::Encoder<nimble::FixedBitWidthEncoding<uint32_t>>::encode(
          sourceBuffer, values, nimble::CompressionType::Uncompressed, options);

  nimble::Buffer sliceBuffer{*this->pool_};
  NIMBLE_ASSERT_THROW(
      nimble::EncodingFactory::slice(
          encoded,
          /*offset=*/0,
          /*length=*/0,
          sliceBuffer,
          options),
      "");
}

TYPED_TEST(FixedBitWidthEncodingTest, sliceCompressedSource) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  nimble::Vector<uint32_t> values{this->pool_.get()};
  for (uint32_t i = 0; i < 512; ++i) {
    values.push_back(1000 + i % 128);
  }

  nimble::Buffer sourceBuffer{*this->pool_};
  const auto encoded =
      nimble::test::Encoder<nimble::FixedBitWidthEncoding<uint32_t>>::encode(
          sourceBuffer, values, nimble::CompressionType::Zstd, options);

  constexpr uint32_t kOffset{17};
  constexpr uint32_t kLength{65};
  nimble::Buffer sliceBuffer{*this->pool_};
  const auto sliced = nimble::EncodingFactory::slice(
      encoded, kOffset, kLength, sliceBuffer, options);

  const auto prefixSize =
      nimble::EncodingPrefix::prefixSize(sliced, TypeParam::useVarint);
  EXPECT_EQ(
      static_cast<nimble::CompressionType>(
          static_cast<uint8_t>(sliced[prefixSize])),
      nimble::CompressionType::Uncompressed);

  auto encoding = nimble::EncodingFactory{options}.create(
      *this->pool_, sliced, this->stringBufferFactory());
  std::vector<uint32_t> result(kLength);
  encoding->materialize(kLength, result.data());

  const std::vector<uint32_t> expected{
      values.begin() + kOffset, values.begin() + kOffset + kLength};
  EXPECT_EQ(result, expected);
}

TYPED_TEST(FixedBitWidthEncodingTest, singleValue) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({42});
  auto encoding = this->createEncoding(values, options);

  EXPECT_EQ(encoding->rowCount(), 1);

  nimble::Vector<uint32_t> result(this->pool_.get(), 1);
  encoding->materialize(1, result.data());
  EXPECT_EQ(result[0], 42u);
}

TYPED_TEST(FixedBitWidthEncodingTest, allZeros) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({0, 0, 0, 0, 0});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 5);
  encoding->materialize(5, result.data());
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(result[i], 0u) << "index " << i;
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, allSameNonZero) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({100, 100, 100, 100});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 4);
  encoding->materialize(4, result.data());
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(result[i], 100u) << "index " << i;
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, largeValues) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  uint32_t maxVal = std::numeric_limits<uint32_t>::max();
  uint32_t halfMax = maxVal / 2;
  auto values = this->toVector({maxVal, halfMax, 0, halfMax, maxVal});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 5);
  encoding->materialize(5, result.data());
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, skipThenMaterialize) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({10, 20, 30, 40, 50, 60, 70, 80});
  auto encoding = this->createEncoding(values, options);

  encoding->skip(3);

  nimble::Vector<uint32_t> result(this->pool_.get(), 3);
  encoding->materialize(3, result.data());
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_EQ(result[i], values[i + 3]) << "index " << i;
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, resetAndRematerialize) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({5, 15, 25, 35, 45});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result1(this->pool_.get(), 5);
  encoding->materialize(5, result1.data());

  encoding->reset();

  nimble::Vector<uint32_t> result2(this->pool_.get(), 5);
  encoding->materialize(5, result2.data());

  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(result1[i], result2[i]) << "index " << i;
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, incrementalMaterialize) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 10);
  encoding->materialize(4, result.data());
  encoding->materialize(3, result.data() + 4);
  encoding->materialize(3, result.data() + 7);

  for (uint32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, materializeZeroRows) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({10, 20, 30});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 3);
  encoding->materialize(0, result.data());

  encoding->materialize(3, result.data());
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, narrowBitWidth) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({0, 1, 0, 1, 0, 1, 0, 1});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 8);
  encoding->materialize(8, result.data());
  for (uint32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, baselineOffset) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({1000, 1001, 1002, 1003, 1004});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 5);
  encoding->materialize(5, result.data());
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, fuzz) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int trial = 0; trial < 20; ++trial) {
    uint32_t rowCount = folly::Random::rand32(1, 500, rng);
    uint32_t maxBits = folly::Random::rand32(1, 32, rng);
    uint32_t mask = maxBits >= 32 ? std::numeric_limits<uint32_t>::max()
                                  : (uint32_t{1} << maxBits) - 1;
    uint32_t baseline = folly::Random::rand32(rng) & ~mask;

    nimble::Vector<uint32_t> values{this->pool_.get()};
    for (uint32_t i = 0; i < rowCount; ++i) {
      values.push_back(baseline + (folly::Random::rand32(rng) & mask));
    }

    auto encoding = this->createEncoding(values, options);

    EXPECT_EQ(encoding->rowCount(), rowCount);

    nimble::Vector<uint32_t> result(this->pool_.get(), rowCount);
    encoding->materialize(rowCount, result.data());

    for (uint32_t i = 0; i < rowCount; ++i) {
      EXPECT_EQ(result[i], values[i]) << "trial " << trial << " index " << i;
    }
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, withZstdCompression) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  nimble::Vector<uint32_t> values{this->pool_.get()};
  for (uint32_t i = 0; i < 200; ++i) {
    values.push_back(i);
  }

  auto encoding = nimble::test::
      Encoder<nimble::FixedBitWidthEncoding<uint32_t>>::createEncoding(
          *this->buffer_,
          values,
          this->stringBufferFactory(),
          nimble::CompressionType::Zstd,
          options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 200);
  encoding->materialize(200, result.data());
  for (uint32_t i = 0; i < 200; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, largeDataset) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  nimble::Vector<uint32_t> values{this->pool_.get()};
  for (uint32_t i = 0; i < 10000; ++i) {
    values.push_back(1000 + (i % 256));
  }

  auto encoding = this->createEncoding(values, options);
  EXPECT_EQ(encoding->rowCount(), 10000);

  nimble::Vector<uint32_t> result(this->pool_.get(), 10000);
  encoding->materialize(10000, result.data());
  for (uint32_t i = 0; i < 10000; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, wideRange) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector(
      {0,
       1,
       255,
       256,
       65535,
       65536,
       16777215,
       16777216,
       std::numeric_limits<uint32_t>::max()});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), values.size());
  encoding->materialize(values.size(), result.data());
  for (uint32_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(FixedBitWidthEncodingTest, uint64MaterializeWideBitWidths) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto roundTripAfterSkip = [&](const std::vector<uint64_t>& values) {
    nimble::Vector<uint64_t> nimbleValues{
        this->pool_.get(), values.begin(), values.end()};
    auto encoding = nimble::test::
        Encoder<nimble::FixedBitWidthEncoding<uint64_t>>::createEncoding(
            *this->buffer_,
            nimbleValues,
            this->stringBufferFactory(),
            nimble::CompressionType::Uncompressed,
            options);

    constexpr uint32_t skipCount = 3;
    encoding->skip(skipCount);

    std::vector<uint64_t> result(values.size() - skipCount);
    encoding->materialize(static_cast<uint32_t>(result.size()), result.data());

    const std::vector<uint64_t> expected{
        values.begin() + skipCount, values.end()};
    EXPECT_EQ(result, expected);
  };

  roundTripAfterSkip(
      {100,
       101,
       102,
       103,
       100 + (uint64_t{1} << 40) - 1,
       100 + (uint64_t{1} << 39),
       100 + 17});
  roundTripAfterSkip(
      {17,
       18,
       19,
       20,
       17 + (uint64_t{1} << 58) - 1,
       17 + (uint64_t{1} << 57),
       17 + 42});
  roundTripAfterSkip(
      {29,
       30,
       31,
       32,
       29 + (uint64_t{1} << 60) - 1,
       29 + (uint64_t{1} << 59),
       29 + 99});
  roundTripAfterSkip(
      {0,
       1,
       2,
       3,
       std::numeric_limits<uint64_t>::max(),
       uint64_t{1} << 63,
       42});
}

TYPED_TEST(FixedBitWidthEncodingTest, skipToEnd) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({10, 20, 30, 40, 50});
  auto encoding = this->createEncoding(values, options);

  encoding->skip(4);

  nimble::Vector<uint32_t> result(this->pool_.get(), 1);
  encoding->materialize(1, result.data());
  EXPECT_EQ(result[0], 50u);
}
