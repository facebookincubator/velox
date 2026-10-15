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
#include "velox/dwio/nimble/encodings/EliasFanoEncoding.h"

#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

class EliasFanoEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  nimble::Vector<T> toVector(const std::vector<T>& input) {
    nimble::Vector<T> values{pool_.get()};
    values.reserve(input.size());
    for (const auto value : input) {
      values.push_back(value);
    }
    return values;
  }

  template <typename T>
  std::unique_ptr<nimble::Encoding> createEncoding(
      const std::vector<T>& input) {
    auto values = toVector(input);
    return nimble::test::Encoder<nimble::EliasFanoEncoding<T>>::createEncoding(
        *buffer_, values, nullptr);
  }

  template <typename T>
  void roundTrip(const std::vector<T>& input) {
    auto encoding = createEncoding(input);
    std::vector<T> output(input.size());
    encoding->materialize(static_cast<uint32_t>(input.size()), output.data());

    EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::EliasFano);
    EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<T>::dataType);
    EXPECT_EQ(encoding->rowCount(), static_cast<uint32_t>(input.size()));
    EXPECT_EQ(output, input);
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

TEST_F(EliasFanoEncodingTest, roundTripUnsignedValues) {
  roundTrip<uint32_t>({1, 2, 3, 7, 9, 12, 13, 15, 20, 100});
  roundTrip<uint64_t>({0, 0, 0, 5, 5, 9, 1'000, 2'000, 2'000, 4'000});
  roundTrip<uint16_t>({2, 4, 4, 4, 9, 12, 30});
  roundTrip<uint8_t>({0, 1, 1, 2, 3, 8});
}

TEST_F(EliasFanoEncodingTest, roundTripSignedValues) {
  roundTrip<int32_t>({-5, -3, -3, -1, 0, 2, 9, 10});
  roundTrip<int64_t>({-100, -100, -7, -1, 0, 1, 1'000'000});
  roundTrip<int16_t>({-10, -5, -5, 0, 2, 4});
  roundTrip<int8_t>({-8, -2, -2, 0, 5, 9});
}

TEST_F(EliasFanoEncodingTest, roundTripSignedBoundaries) {
  roundTrip<int8_t>({
      std::numeric_limits<int8_t>::min(),
      -1,
      0,
      std::numeric_limits<int8_t>::max(),
  });
  roundTrip<int16_t>({
      std::numeric_limits<int16_t>::min(),
      -1,
      0,
      std::numeric_limits<int16_t>::max(),
  });
  roundTrip<int32_t>({
      std::numeric_limits<int32_t>::min(),
      -1,
      0,
      std::numeric_limits<int32_t>::max(),
  });
  roundTrip<int64_t>({
      std::numeric_limits<int64_t>::min(),
      -1,
      0,
      std::numeric_limits<int64_t>::max(),
  });
}

TEST_F(EliasFanoEncodingTest, roundTripSingleAndRepeatedValues) {
  roundTrip<uint64_t>({std::numeric_limits<uint64_t>::max()});
  roundTrip<int64_t>({-7, -7, -7, -7});
}

TEST_F(EliasFanoEncodingTest, resetSkipAndMaterialize) {
  const std::vector<uint32_t> input{0, 1, 2, 10, 11, 12, 100, 101, 102};
  auto encoding = createEncoding(input);

  encoding->reset();
  encoding->skip(0);
  encoding->skip(4);
  std::vector<uint32_t> output(3);
  encoding->materialize(static_cast<uint32_t>(output.size()), output.data());
  EXPECT_EQ(output, (std::vector<uint32_t>{11, 12, 100}));

  encoding->reset();
  output.resize(input.size());
  encoding->materialize(static_cast<uint32_t>(output.size()), output.data());
  EXPECT_EQ(output, input);
}

TEST_F(EliasFanoEncodingTest, supportsRandomAccessAndValueLowerBound) {
  const std::vector<uint64_t> input{10, 20, 20, 40};
  const auto encoding = createEncoding(input);
  const auto* eliasFano =
      dynamic_cast<const nimble::EliasFanoEncoding<uint64_t>*>(encoding.get());
  ASSERT_NE(eliasFano, nullptr);

  EXPECT_EQ(eliasFano->valueAt(0), 10);
  EXPECT_EQ(eliasFano->valueAt(3), 40);
  EXPECT_EQ(eliasFano->lowerBound(0), 0);
  EXPECT_EQ(eliasFano->lowerBound(10), 0);
  EXPECT_EQ(eliasFano->lowerBound(19), 1);
  EXPECT_EQ(eliasFano->lowerBound(20), 1);
  EXPECT_EQ(eliasFano->lowerBound(21), 3);
  EXPECT_EQ(eliasFano->lowerBound(41), input.size());
}

TEST_F(EliasFanoEncodingTest, absoluteReadsPreserveSequentialPosition) {
  const std::vector<uint64_t> input{10, 20, 20, 40, 80};
  auto encoding = createEncoding(input);
  auto* eliasFano =
      dynamic_cast<nimble::EliasFanoEncoding<uint64_t>*>(encoding.get());
  ASSERT_NE(eliasFano, nullptr);

  std::vector<uint64_t> absoluteOutput(3);
  eliasFano->materializeAt(
      /*offset=*/1,
      static_cast<uint32_t>(absoluteOutput.size()),
      absoluteOutput.data());
  EXPECT_EQ(absoluteOutput, (std::vector<uint64_t>{20, 20, 40}));

  std::vector<uint64_t> sequentialOutput(2);
  encoding->materialize(
      static_cast<uint32_t>(sequentialOutput.size()), sequentialOutput.data());
  EXPECT_EQ(sequentialOutput, (std::vector<uint64_t>{10, 20}));

  eliasFano->materializeAt(
      static_cast<uint32_t>(input.size()), /*rowCount=*/0, nullptr);
  encoding->materialize(/*rowCount=*/0, nullptr);
  NIMBLE_ASSERT_THROW(
      eliasFano->materializeAt(
          input.size(), /*rowCount=*/1, absoluteOutput.data()),
      "Reading past end of encoding.");
  NIMBLE_ASSERT_THROW(
      eliasFano->valueAt(input.size()), "Reading past end of encoding.");
}

TEST_F(EliasFanoEncodingTest, slicesWithoutGenericMaterialization) {
  const std::vector<int64_t> input{
      std::numeric_limits<int64_t>::min(), -7, -7, 0, 4, 100, 1'000};
  const auto values = toVector(input);
  const auto encoded =
      nimble::test::Encoder<nimble::EliasFanoEncoding<int64_t>>::encode(
          *buffer_, values);

  const std::vector<std::pair<uint32_t, uint32_t>> ranges{
      {0, 1}, {1, 5}, {3, 1}, {6, 1}, {0, input.size()}};
  for (const auto& [offset, length] : ranges) {
    SCOPED_TRACE(
        ::testing::Message() << "offset=" << offset << ", length=" << length);
    nimble::Buffer sliceBuffer{*pool_};
    const auto sliced = nimble::EliasFanoEncoding<int64_t>::slice(
        encoded, offset, length, sliceBuffer);
    auto encoding = nimble::EncodingFactory{}.create(
        *pool_, sliced, /*stringBufferFactory=*/nullptr);
    std::vector<int64_t> output(length);
    encoding->materialize(length, output.data());

    EXPECT_EQ(
        output,
        std::vector<int64_t>(
            input.begin() + offset, input.begin() + offset + length));
  }

  nimble::Buffer sliceBuffer{*pool_};
  NIMBLE_ASSERT_THROW(
      nimble::EliasFanoEncoding<int64_t>::slice(
          encoded, /*offset=*/0, /*length=*/0, sliceBuffer),
      "Cannot slice zero rows.");
}

TEST_F(EliasFanoEncodingTest, factoryRoundTrip) {
  const std::vector<uint64_t> input{10, 10, 12, 30, 1'000, 1'001};
  auto values = toVector(input);
  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<uint64_t>>(
          std::vector<std::pair<nimble::EncodingType, float>>{
              {nimble::EncodingType::EliasFano, 1.0}},
          std::nullopt,
          std::nullopt);
  const auto encoded = nimble::EncodingFactory::encode<uint64_t>(
      std::move(policy), values, *buffer_);
  auto encoding = nimble::EncodingFactory{}.create(
      *pool_, encoded, /*stringBufferFactory=*/nullptr);
  std::vector<uint64_t> output(input.size());
  encoding->materialize(static_cast<uint32_t>(input.size()), output.data());

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::EliasFano);
  EXPECT_EQ(output, input);
  EXPECT_NE(encoding->debugString(0).find("lowerBits="), std::string::npos);
}

TEST_F(EliasFanoEncodingTest, factoryRoundTripSignedLogicalOrder) {
  const std::vector<int64_t> input{
      std::numeric_limits<int64_t>::min(),
      -7,
      -7,
      0,
      9,
      std::numeric_limits<int64_t>::max()};
  auto values = toVector(input);
  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<int64_t>>(
          std::vector<std::pair<nimble::EncodingType, float>>{
              {nimble::EncodingType::EliasFano, 1.0}},
          std::nullopt,
          std::nullopt);
  const auto encoded = nimble::EncodingFactory::encode<int64_t>(
      std::move(policy), values, *buffer_);
  auto encoding = nimble::EncodingFactory{}.create(
      *pool_, encoded, /*stringBufferFactory=*/nullptr);
  std::vector<int64_t> output(input.size());
  encoding->materialize(static_cast<uint32_t>(input.size()), output.data());

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::EliasFano);
  EXPECT_EQ(output, input);
}

TEST_F(EliasFanoEncodingTest, estimateMatchesEncodedSize) {
  for (const uint32_t rowCount : {1, 127, 128, 255, 256, 257}) {
    std::vector<uint64_t> input(rowCount);
    for (uint32_t row{0}; row < rowCount; ++row) {
      input[row] = row / 2;
    }
    const auto values = toVector(input);
    const auto physicalValues =
        std::span<const uint64_t>{values.data(), values.size()};
    const auto statistics =
        nimble::Statistics<uint64_t>::create(physicalValues);
    for (const bool useVarintRowCount : {false, true}) {
      SCOPED_TRACE(
          ::testing::Message() << "rowCount=" << rowCount
                               << ", useVarintRowCount=" << useVarintRowCount);
      const nimble::Encoding::Options options{
          .useVarintRowCount = useVarintRowCount};
      const auto estimate = nimble::EliasFanoEncoding<uint64_t>::estimateSize(
          physicalValues, statistics, options);
      ASSERT_TRUE(estimate.has_value());
      buffer_->reset();
      const auto encoded =
          nimble::test::Encoder<nimble::EliasFanoEncoding<uint64_t>>::encode(
              *buffer_, values, nimble::CompressionType::Uncompressed, options);
      EXPECT_EQ(*estimate, encoded.size());
    }
  }
}

TEST_F(EliasFanoEncodingTest, rejectsSerializedSizeAboveUint32) {
  constexpr uint64_t kPayloadOffset{24};
  constexpr uint64_t kPaddingSize{7};
  constexpr uint64_t kMaxPayloadSize{
      std::numeric_limits<uint32_t>::max() - kPayloadOffset - kPaddingSize};

  EXPECT_EQ(
      nimble::detail::elias_fano::trySerializedSize(
          kPayloadOffset, kMaxPayloadSize, kPaddingSize),
      std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(
      nimble::detail::elias_fano::trySerializedSize(
          kPayloadOffset, kMaxPayloadSize + 1, kPaddingSize),
      std::nullopt);
  EXPECT_EQ(
      nimble::detail::elias_fano::trySerializedSize(
          kPayloadOffset, std::numeric_limits<uint64_t>::max(), kPaddingSize),
      std::nullopt);
}

TEST_F(EliasFanoEncodingTest, estimateRejectsEmptyAndUnsortedValues) {
  const auto emptyStatistics =
      nimble::Statistics<uint64_t>::create(std::span<const uint64_t>{});
  EXPECT_EQ(
      nimble::EliasFanoEncoding<uint64_t>::estimateSize({}, emptyStatistics),
      std::nullopt);

  const std::vector<uint32_t> input{1, 5, 4, 8};
  const auto values = toVector(input);
  const auto physicalValues = std::span<const uint32_t>{
      reinterpret_cast<const uint32_t*>(values.data()), values.size()};
  const auto statistics = nimble::Statistics<uint32_t>::create(physicalValues);
  EXPECT_EQ(
      nimble::EliasFanoEncoding<uint32_t>::estimateSize(
          physicalValues, statistics),
      std::nullopt);
}

TEST_F(EliasFanoEncodingTest, rejectsTruncatedHeader) {
  const auto values = toVector<uint64_t>({1, 2, 3});
  const auto encoded =
      nimble::test::Encoder<nimble::EliasFanoEncoding<uint64_t>>::encode(
          *buffer_, values);
  std::string malformed{encoded};
  constexpr size_t kTruncatedSize = nimble::EncodingPrefix::kFixedPrefixSize +
      sizeof(uint64_t) + sizeof(uint8_t) + sizeof(uint32_t) - 1;
  malformed.resize(kTruncatedSize);

  NIMBLE_ASSERT_FILE_THROW(
      nimble::EncodingFactory{}.create(
          *pool_, malformed, /*stringBufferFactory=*/nullptr),
      "Truncated EliasFano header.");
}

TEST_F(EliasFanoEncodingTest, rejectsInvalidLowerBitCount) {
  const auto values = toVector<uint64_t>({1, 2, 3});
  const auto encoded =
      nimble::test::Encoder<nimble::EliasFanoEncoding<uint64_t>>::encode(
          *buffer_, values);
  std::string malformed{encoded};
  constexpr size_t kLowerBitsOffset =
      nimble::EncodingPrefix::kFixedPrefixSize + sizeof(uint64_t);
  malformed[kLowerBitsOffset] = 57;

  NIMBLE_ASSERT_FILE_THROW(
      nimble::EncodingFactory{}.create(
          *pool_, malformed, /*stringBufferFactory=*/nullptr),
      "Invalid EliasFano lower bit count.");
}

TEST_F(EliasFanoEncodingTest, rejectsInsufficientUpperBits) {
  const auto values = toVector<uint64_t>({1, 2, 3});
  const auto encoded =
      nimble::test::Encoder<nimble::EliasFanoEncoding<uint64_t>>::encode(
          *buffer_, values);
  std::string malformed{encoded};
  char* upperBytes = malformed.data() +
      nimble::EncodingPrefix::kFixedPrefixSize + sizeof(uint64_t) +
      sizeof(uint8_t);
  nimble::encoding::writeUint32(0, upperBytes);

  NIMBLE_ASSERT_FILE_THROW(
      nimble::EncodingFactory{}.create(
          *pool_, malformed, /*stringBufferFactory=*/nullptr),
      "Invalid EliasFano upper bit count.");
}

TEST_F(EliasFanoEncodingTest, rejectsMissingUpperBits) {
  const auto values = toVector<uint64_t>({0, 0, 0});
  const auto encoded =
      nimble::test::Encoder<nimble::EliasFanoEncoding<uint64_t>>::encode(
          *buffer_, values);
  std::string malformed{encoded};
  const auto upperByteOffset =
      malformed.size() - folly::compression::kUpperTrailingBytes - 1;
  malformed[upperByteOffset] = 0;

  NIMBLE_ASSERT_FILE_THROW(
      nimble::EncodingFactory{}.create(
          *pool_, malformed, /*stringBufferFactory=*/nullptr),
      "Invalid EliasFano upper bits.");
}

TEST_F(EliasFanoEncodingTest, alignsAndReadsFollyPointerTables) {
  std::vector<uint64_t> input(600);
  for (size_t i{0}; i < input.size(); ++i) {
    input[i] = i / 2;
  }
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "useVarint=" << useVarint);
    buffer_->reset();
    buffer_->reserve(1);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    const auto encoded =
        nimble::test::Encoder<nimble::EliasFanoEncoding<uint64_t>>::encode(
            *buffer_,
            toVector(input),
            nimble::CompressionType::Uncompressed,
            options);
    const auto prefixSize = nimble::EncodingPrefix::serializedSize(
        static_cast<uint32_t>(input.size()), useVarint);
    const auto headerEndOffset =
        prefixSize + sizeof(uint64_t) + sizeof(uint8_t) + sizeof(uint32_t);
    const auto payloadOffset =
        (headerEndOffset + alignof(uint64_t) - 1) & ~(alignof(uint64_t) - 1);

    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(encoded.data()) % alignof(uint64_t), 0);
    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(encoded.data() + payloadOffset) %
            alignof(uint64_t),
        0);
    for (size_t offset{headerEndOffset}; offset < payloadOffset; ++offset) {
      SCOPED_TRACE(::testing::Message() << "offset=" << offset);
      EXPECT_EQ(encoded[offset], 0);
    }

    auto encoding = nimble::EncodingFactory{}.create(
        *pool_, encoded, /*stringBufferFactory=*/nullptr, options);
    std::vector<uint64_t> output(input.size());
    encoding->materialize(static_cast<uint32_t>(input.size()), output.data());
    EXPECT_EQ(output, input);

    const auto* eliasFano =
        dynamic_cast<const nimble::EliasFanoEncoding<uint64_t>*>(
            encoding.get());
    ASSERT_NE(eliasFano, nullptr);
    for (const uint32_t row : {127, 128, 255, 256, 511, 512}) {
      EXPECT_EQ(eliasFano->valueAt(row), input[row]);
      EXPECT_EQ(eliasFano->lowerBound(input[row]), row - row % 2);
    }
  }
}

TEST_F(EliasFanoEncodingTest, encodeRejectsUnsortedValues) {
  const std::vector<int32_t> input{-1, -2};
  NIMBLE_ASSERT_THROW(
      createEncoding(input),
      "EliasFano requires non-empty non-decreasing values.");
}

TEST_F(EliasFanoEncodingTest, encodeRejectsEmptyValues) {
  NIMBLE_ASSERT_THROW(
      createEncoding(std::vector<uint64_t>{}),
      "EliasFano requires non-empty non-decreasing values.");
}
