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
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <random>

#include "velox/buffer/BufferPool.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

using namespace ::facebook;

namespace facebook::nimble::test {

template <typename T>
class SimdForBitpackEncodingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  }

  std::unique_ptr<EncodingSelectionPolicy<T>> createSelectionPolicy() {
    EncodingLayout layout{
        EncodingType::SimdForBitpack, {}, CompressionType::Uncompressed};
    return std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
        std::move(layout),
        CompressionOptions{},
        encodingSelectionPolicyCreator_);
  }

  std::unique_ptr<Encoding> encodeAndCreate(const std::vector<T>& values) {
    encodedStorage_ = encodeValues(values);
    return EncodingFactory().create(
        *pool_, {encodedStorage_.data(), encodedStorage_.size()}, nullptr);
  }

  std::vector<char> encodeValues(
      const std::vector<T>& values,
      const Encoding::Options& options = {}) {
    Buffer buffer{*pool_};
    auto encoded = EncodingFactory::encode<T>(
        createSelectionPolicy(),
        std::span<const T>{values.data(), values.size()},
        buffer,
        options);
    return {encoded.begin(), encoded.end()};
  }

  void expectMalformedDirect(
      std::string_view encoded,
      std::string_view expectedMessage,
      const Encoding::Options& options = {}) {
    NIMBLE_ASSERT_THROW(
        (SimdForBitpackEncoding<T>{*pool_, encoded, nullptr, options}),
        expectedMessage);
  }

  void expectMalformedFactory(
      std::string_view encoded,
      std::string_view expectedMessage,
      const Encoding::Options& options = {}) {
    NIMBLE_ASSERT_THROW(
        (EncodingFactory(options).create(*pool_, encoded, nullptr)),
        expectedMessage);
  }

  void expectMalformed(
      std::string_view encoded,
      std::string_view expectedMessage,
      const Encoding::Options& options = {}) {
    expectMalformedDirect(encoded, expectedMessage, options);
    expectMalformedFactory(encoded, expectedMessage, options);
  }

  void expectMalformed(
      const std::vector<char>& encoded,
      std::string_view expectedMessage,
      const Encoding::Options& options = {}) {
    expectMalformed({encoded.data(), encoded.size()}, expectedMessage, options);
  }

  size_t bitWidthOffset(
      const std::vector<char>& encoded,
      const Encoding::Options& options = {}) {
    return EncodingPrefix::prefixSize(
               {encoded.data(), encoded.size()}, options.useVarintRowCount) +
        sizeof(typename TypeTraits<T>::physicalType);
  }

  size_t firstGroupRowsOffset(
      const std::vector<char>& encoded,
      const Encoding::Options& options = {}) {
    return bitWidthOffset(encoded, options) + sizeof(uint8_t);
  }

  void roundTripAndExpect(const std::vector<T>& values) {
    auto encoding = encodeAndCreate(values);
    EXPECT_EQ(encoding->encodingType(), EncodingType::SimdForBitpack);
    EXPECT_EQ(encoding->dataType(), TypeTraits<T>::dataType);
    EXPECT_EQ(encoding->rowCount(), values.size());

    std::vector<T> decoded(values.size());
    encoding->materialize(static_cast<uint32_t>(values.size()), decoded.data());
    for (size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(decoded[i], values[i]) << "mismatch at index " << i;
    }
  }

  uint32_t firstGroupRows(
      std::string_view encoded,
      const Encoding::Options& options) {
    const char* pos = encoded.data() +
        EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
    pos += sizeof(typename TypeTraits<T>::physicalType);
    ++pos;
    return varint::readVarint32(&pos);
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::vector<char> encodedStorage_;
  ManualEncodingSelectionPolicyFactory manualPolicyFactory_;
  EncodingSelectionPolicyCreator encodingSelectionPolicyCreator_ =
      [this](DataType dataType) {
        return manualPolicyFactory_.createPolicy(dataType);
      };
};

using SimdForBitpackTypes =
    ::testing::Types<uint32_t, uint64_t, uint8_t, uint16_t>;
TYPED_TEST_SUITE(SimdForBitpackEncodingTest, SimdForBitpackTypes);

TYPED_TEST(SimdForBitpackEncodingTest, singleElement) {
  this->roundTripAndExpect({TypeParam{42}});
}

TYPED_TEST(SimdForBitpackEncodingTest, allSameValues) {
  std::vector<TypeParam> values(64, TypeParam{7});
  this->roundTripAndExpect(values);
}

TYPED_TEST(SimdForBitpackEncodingTest, exactlyOneGroup) {
  std::vector<TypeParam> values;
  values.reserve(32);
  for (uint32_t i = 0; i < 32; ++i) {
    values.push_back(static_cast<TypeParam>(100 + i));
  }
  this->roundTripAndExpect(values);
}

TYPED_TEST(SimdForBitpackEncodingTest, partialLastGroup) {
  for (uint32_t n : {1u, 15u, 31u, 33u, 63u, 65u, 100u}) {
    SCOPED_TRACE(fmt::format("n={}", n));
    std::vector<TypeParam> values;
    values.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      values.push_back(static_cast<TypeParam>(10 + i % 50));
    }
    this->roundTripAndExpect(values);
  }
}

TYPED_TEST(SimdForBitpackEncodingTest, bitWidthZero) {
  std::vector<TypeParam> values(100, TypeParam{42});
  auto encoding = this->encodeAndCreate(values);
  std::vector<TypeParam> decoded(100);
  encoding->materialize(100, decoded.data());
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_EQ(decoded[i], TypeParam{42});
  }
}

TYPED_TEST(SimdForBitpackEncodingTest, fullBitWidthRange) {
  std::vector<TypeParam> values;
  values.reserve(100);
  for (uint32_t i = 0; i < 100; ++i) {
    values.push_back(static_cast<TypeParam>(i));
  }
  values.push_back(std::numeric_limits<TypeParam>::max());
  this->roundTripAndExpect(values);
}

TYPED_TEST(SimdForBitpackEncodingTest, rejectsInvalidFirstGroupRows) {
  std::vector<TypeParam> values(16);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<TypeParam>(i);
  }
  const auto encoded = this->encodeValues(values);
  const auto firstGroupRowsOffset = this->firstGroupRowsOffset(encoded);

  for (const auto invalidFirstGroupRows : {uint8_t{0}, uint8_t{17}}) {
    auto malformed = encoded;
    malformed[firstGroupRowsOffset] = static_cast<char>(invalidFirstGroupRows);
    this->expectMalformed(
        malformed, "Invalid SimdForBitpack first group row count.");
  }

  auto malformed = encoded;
  malformed[firstGroupRowsOffset] = 33;
  this->expectMalformed(
      malformed, "Invalid SimdForBitpack first group row count.");
}

TYPED_TEST(SimdForBitpackEncodingTest, rejectsInvalidBitWidth) {
  const auto encoded = this->encodeValues(std::vector<TypeParam>{1, 2, 3});
  auto malformed = encoded;
  malformed[this->bitWidthOffset(encoded)] =
      static_cast<char>(sizeof(TypeParam) * 8 + 1);
  this->expectMalformed(
      malformed, "SimdForBitpack bit width exceeds physical type size.");
}

TYPED_TEST(SimdForBitpackEncodingTest, rejectsTruncatedFixedHeader) {
  auto malformed = this->encodeValues(std::vector<TypeParam>{1, 2, 3});
  malformed.resize(EncodingPrefix::kFixedPrefixSize + sizeof(TypeParam) - 1);
  this->expectMalformed(malformed, "Truncated SimdForBitpack header.");
}

TYPED_TEST(SimdForBitpackEncodingTest, rejectsEveryTruncatedArtifactSize) {
  std::vector<TypeParam> values(64);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<TypeParam>(i);
  }
  const auto encoded = this->encodeValues(values);
  const std::string storage{encoded.data(), encoded.size()};

  for (size_t truncatedSize = 0; truncatedSize < storage.size();
       ++truncatedSize) {
    SCOPED_TRACE(testing::Message() << "truncatedSize=" << truncatedSize);
    const std::string_view truncated{storage.data(), truncatedSize};
    this->expectMalformedDirect(truncated, "");
    // EncodingFactory dispatches on the common type bytes before constructing
    // an encoding. Its prefix safety below two bytes belongs to a separate
    // common factory safety change.
    if (truncatedSize >= EncodingPrefix::kRowCountOffset) {
      this->expectMalformedFactory(truncated, "");
    }
  }
}

TYPED_TEST(SimdForBitpackEncodingTest, rejectsMalformedRowCountVarint) {
  const Encoding::Options options{.useVarintRowCount = true};
  std::vector<TypeParam> values(300);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<TypeParam>(i);
  }
  const auto encoded = this->encodeValues(values, options);
  const auto headerOffset = EncodingPrefix::prefixSize(
      {encoded.data(), encoded.size()}, /*useVarint=*/true);

  this->expectMalformed(
      std::string_view{
          encoded.data(), EncodingPrefix::kRowCountOffset + size_t{1}},
      "Truncated SimdForBitpack varint.",
      options);

  const auto replaceRowCount = [&](std::initializer_list<char> rowCount) {
    std::vector<char> malformed{
        encoded.begin(), encoded.begin() + EncodingPrefix::kRowCountOffset};
    malformed.insert(malformed.end(), rowCount);
    malformed.insert(
        malformed.end(), encoded.begin() + headerOffset, encoded.end());
    return malformed;
  };

  for (const auto& malformed :
       {replaceRowCount(
            {static_cast<char>(0x80),
             static_cast<char>(0x80),
             static_cast<char>(0x80),
             static_cast<char>(0x80),
             static_cast<char>(0x80),
             0}),
        replaceRowCount(
            {static_cast<char>(0x80),
             static_cast<char>(0x80),
             static_cast<char>(0x80),
             static_cast<char>(0x80),
             static_cast<char>(0x10)}),
        replaceRowCount(
            {static_cast<char>(0xac), static_cast<char>(0x82), 0})}) {
    this->expectMalformed(
        malformed, "Overlong SimdForBitpack varint.", options);
  }
}

TYPED_TEST(SimdForBitpackEncodingTest, rejectsInvalidPackedPayloadSize) {
  std::vector<TypeParam> values(64);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<TypeParam>(i);
  }
  const auto encoded = this->encodeValues(values);

  auto truncated = encoded;
  truncated.pop_back();
  this->expectMalformed(
      truncated, "Invalid SimdForBitpack packed payload size.");

  auto extended = encoded;
  extended.push_back(0);
  this->expectMalformed(
      extended, "Invalid SimdForBitpack packed payload size.");
}

TYPED_TEST(SimdForBitpackEncodingTest, sliceRejectsTruncatedPackedPayload) {
  std::vector<TypeParam> values(64);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<TypeParam>(i);
  }
  auto malformed = this->encodeValues(values);
  malformed.pop_back();

  Buffer sliceBuffer{*this->pool_};
  NIMBLE_ASSERT_THROW(
      (SimdForBitpackEncoding<TypeParam>::slice(
          {malformed.data(), malformed.size()},
          /*offset=*/0,
          /*length=*/1,
          sliceBuffer)),
      "Invalid SimdForBitpack packed payload size.");
}

TYPED_TEST(SimdForBitpackEncodingTest, rejectsMalformedFirstGroupRowsVarint) {
  const auto encoded = this->encodeValues(std::vector<TypeParam>{1, 2, 3});
  const auto firstGroupRowsOffset = this->firstGroupRowsOffset(encoded);

  std::vector<char> truncated{
      encoded.begin(), encoded.begin() + firstGroupRowsOffset};
  truncated.push_back(static_cast<char>(0x80));
  this->expectMalformed(truncated, "Truncated SimdForBitpack varint.");

  std::vector<char> overlong{
      encoded.begin(), encoded.begin() + firstGroupRowsOffset};
  overlong.insert(
      overlong.end(),
      {static_cast<char>(0x80),
       static_cast<char>(0x80),
       static_cast<char>(0x80),
       static_cast<char>(0x80),
       static_cast<char>(0x80),
       0});
  overlong.insert(
      overlong.end(),
      encoded.begin() + firstGroupRowsOffset + 1,
      encoded.end());
  this->expectMalformed(overlong, "Overlong SimdForBitpack varint.");
}

TYPED_TEST(SimdForBitpackEncodingTest, nonZeroBaseline) {
  std::vector<TypeParam> values;
  values.reserve(200);
  const TypeParam base =
      static_cast<TypeParam>(std::numeric_limits<TypeParam>::max() / 2);
  for (uint32_t i = 0; i < 200; ++i) {
    values.push_back(static_cast<TypeParam>(base + i % 30));
  }
  this->roundTripAndExpect(values);
}

TYPED_TEST(SimdForBitpackEncodingTest, skipAndMaterialize) {
  std::vector<TypeParam> values;
  values.reserve(300);
  for (uint32_t i = 0; i < 300; ++i) {
    values.push_back(static_cast<TypeParam>(50 + (i % 100)));
  }
  auto encoding = this->encodeAndCreate(values);

  encoding->reset();
  std::vector<TypeParam> chunk(50);
  encoding->materialize(50, chunk.data());
  for (uint32_t i = 0; i < 50; ++i) {
    EXPECT_EQ(chunk[i], values[i]);
  }

  encoding->skip(75);
  encoding->materialize(50, chunk.data());
  for (uint32_t i = 0; i < 50; ++i) {
    EXPECT_EQ(chunk[i], values[125 + i]) << "i=" << i;
  }

  encoding->skip(75);
  encoding->materialize(50, chunk.data());
  for (uint32_t i = 0; i < 50; ++i) {
    EXPECT_EQ(chunk[i], values[250 + i]) << "i=" << i;
  }

  encoding->reset();
  std::vector<TypeParam> all(values.size());
  encoding->materialize(static_cast<uint32_t>(values.size()), all.data());
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(all[i], values[i]) << "after-reset i=" << i;
  }
}

TYPED_TEST(SimdForBitpackEncodingTest, skipAcrossGroupBoundary) {
  std::vector<TypeParam> values;
  values.reserve(128);
  for (uint32_t i = 0; i < 128; ++i) {
    values.push_back(static_cast<TypeParam>(i % 64));
  }
  auto encoding = this->encodeAndCreate(values);

  encoding->skip(30);
  std::vector<TypeParam> chunk(4);
  encoding->materialize(4, chunk.data());
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(chunk[i], values[30 + i]) << "i=" << i;
  }
}

TYPED_TEST(SimdForBitpackEncodingTest, slice) {
  using T = TypeParam;

  struct Range {
    uint32_t offset;
    uint32_t length;
  };

  std::vector<T> values;
  values.reserve(300);
  for (uint32_t i = 0; i < 300; ++i) {
    values.push_back(static_cast<T>(1000 + (i % 127)));
  }
  const auto valueCount = static_cast<uint32_t>(values.size());

  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
    const Encoding::Options options{.useVarintRowCount = useVarint};
    Buffer buffer{*this->pool_};
    const auto encoded = EncodingFactory::encode<T>(
        this->createSelectionPolicy(),
        std::span<const T>{values.data(), values.size()},
        buffer,
        options);

    for (const auto range :
         {Range{/*offset=*/0, /*length=*/1},
          Range{/*offset=*/17, /*length=*/7},
          Range{/*offset=*/1, /*length=*/37},
          Range{/*offset=*/31, /*length=*/64},
          Range{/*offset=*/32, /*length=*/64},
          Range{/*offset=*/63, /*length=*/2},
          Range{/*offset=*/valueCount - 43, /*length=*/43}}) {
      SCOPED_TRACE(
          fmt::format("offset={}, length={}", range.offset, range.length));
      Buffer sliceBuffer{*this->pool_};
      const auto sliced = SimdForBitpackEncoding<T>::slice(
          encoded, range.offset, range.length, sliceBuffer, options);

      SimdForBitpackEncoding<T> encoding{
          *this->pool_, sliced, nullptr, options};
      EXPECT_EQ(encoding.encodingType(), EncodingType::SimdForBitpack);
      EXPECT_EQ(encoding.dataType(), TypeTraits<T>::dataType);
      EXPECT_EQ(encoding.rowCount(), range.length);

      std::vector<T> output(range.length);
      encoding.materialize(range.length, output.data());
      EXPECT_EQ(
          output,
          std::vector<T>(
              values.begin() + range.offset,
              values.begin() + range.offset + range.length));
    }
  }
}

TYPED_TEST(SimdForBitpackEncodingTest, sliceFirstGroupRows) {
  using T = TypeParam;

  std::vector<T> values;
  values.reserve(160);
  for (uint32_t i = 0; i < 160; ++i) {
    values.push_back(static_cast<T>(1000 + (i % 127)));
  }

  struct Case {
    uint32_t offset;
    uint32_t length;
    uint32_t expectedFirstGroupRows;
  };

  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
    const Encoding::Options options{.useVarintRowCount = useVarint};
    Buffer buffer{*this->pool_};
    const auto encoded = EncodingFactory::encode<T>(
        this->createSelectionPolicy(),
        std::span<const T>{values.data(), values.size()},
        buffer,
        options);

    for (const auto testCase :
         {Case{/*offset=*/0, /*length=*/64, /*expectedFirstGroupRows=*/32},
          Case{/*offset=*/17, /*length=*/64, /*expectedFirstGroupRows=*/15},
          Case{/*offset=*/31, /*length=*/33, /*expectedFirstGroupRows=*/1},
          Case{/*offset=*/63, /*length=*/2, /*expectedFirstGroupRows=*/1},
          Case{/*offset=*/145, /*length=*/15, /*expectedFirstGroupRows=*/15}}) {
      SCOPED_TRACE(
          fmt::format(
              "offset={}, length={}", testCase.offset, testCase.length));
      Buffer sliceBuffer{*this->pool_};
      const auto sliced = SimdForBitpackEncoding<T>::slice(
          encoded, testCase.offset, testCase.length, sliceBuffer, options);
      EXPECT_EQ(
          this->firstGroupRows(sliced, options),
          testCase.expectedFirstGroupRows);
    }
  }
}

TYPED_TEST(SimdForBitpackEncodingTest, slicedFirstGroupReadsThroughView) {
  using T = TypeParam;

  struct Range {
    uint32_t offset;
    uint32_t length;
  };

  std::vector<T> values;
  values.reserve(160);
  for (uint32_t i = 0; i < 160; ++i) {
    values.push_back(static_cast<T>(1000 + (i % 127)));
  }

  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
    const Encoding::Options options{.useVarintRowCount = useVarint};
    Buffer buffer{*this->pool_};
    const auto encoded = EncodingFactory::encode<T>(
        this->createSelectionPolicy(),
        std::span<const T>{values.data(), values.size()},
        buffer,
        options);

    for (const auto range :
         {Range{/*offset=*/17, /*length=*/7},
          Range{/*offset=*/31, /*length=*/33},
          Range{/*offset=*/63, /*length=*/34},
          Range{/*offset=*/145, /*length=*/15}}) {
      SCOPED_TRACE(
          fmt::format("offset={}, length={}", range.offset, range.length));
      Buffer sliceBuffer{*this->pool_};
      const auto sliced = SimdForBitpackEncoding<T>::slice(
          encoded, range.offset, range.length, sliceBuffer, options);

      const std::vector<T> expected{
          values.begin() + range.offset,
          values.begin() + range.offset + range.length};
      SimdForBitpackEncoding<T> encoding{
          *this->pool_, sliced, nullptr, options};
      std::vector<T> encodingOutput(range.length);
      encoding.materialize(range.length, encodingOutput.data());
      EXPECT_EQ(encodingOutput, expected);

      auto view = createEncodingView(sliced, this->pool_.get(), options);
      ASSERT_NE(view, nullptr);

      std::vector<T> viewOutput(range.length);
      view->read(/*offset=*/0, range.length, viewOutput.data());
      EXPECT_EQ(viewOutput, expected);

      T value;
      view->readAt(/*offset=*/0, &value);
      EXPECT_EQ(value, values[range.offset]);
      view->readAt(range.length - 1, &value);
      EXPECT_EQ(value, values[range.offset + range.length - 1]);
    }
  }
}

TYPED_TEST(SimdForBitpackEncodingTest, invalidSliceRange) {
  using T = TypeParam;

  std::vector<T> values;
  values.reserve(16);
  for (uint32_t i = 0; i < 16; ++i) {
    values.push_back(static_cast<T>(i + 1));
  }

  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
    const Encoding::Options options{.useVarintRowCount = useVarint};
    Buffer buffer{*this->pool_};
    const auto encoded = EncodingFactory::encode<T>(
        this->createSelectionPolicy(),
        std::span<const T>{values.data(), values.size()},
        buffer,
        options);

    Buffer sliceBuffer{*this->pool_};
    NIMBLE_ASSERT_THROW(
        SimdForBitpackEncoding<T>::slice(
            encoded,
            /*offset=*/0,
            /*length=*/0,
            sliceBuffer,
            options),
        "Cannot slice zero rows.");
    NIMBLE_ASSERT_THROW(
        SimdForBitpackEncoding<T>::slice(
            encoded,
            /*offset=*/values.size() + 1,
            /*length=*/1,
            sliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        SimdForBitpackEncoding<T>::slice(
            encoded,
            /*offset=*/values.size() - 1,
            /*length=*/2,
            sliceBuffer,
            options),
        "");
  }
}

TEST(SimdForBitpackBufferPoolTest, sliceDoesNotUseScratchBuffer) {
  using Value = uint64_t;
  using Encoding = SimdForBitpackEncoding<Value>;

  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Buffer encodeBuffer{*pool};
  std::vector<Value> input;
  input.reserve(300);
  for (uint32_t i = 0; i < 300; ++i) {
    input.push_back(static_cast<Value>(1000 + (i % 127)));
  }

  Vector<Value> values{pool.get()};
  values.insert(values.end(), input.data(), input.data() + input.size());
  EncodingLayout layout{
      EncodingType::SimdForBitpack, {}, CompressionType::Uncompressed};
  ManualEncodingSelectionPolicyFactory manualFactory;
  EncodingSelectionPolicyCreator creator = [&manualFactory](DataType dataType) {
    return manualFactory.createPolicy(dataType);
  };
  auto policy = std::make_unique<ReplayedEncodingSelectionPolicy<Value>>(
      std::move(layout), CompressionOptions{}, creator);
  const auto encoded = EncodingFactory::encode<Value>(
      std::move(policy),
      std::span<const Value>{values.data(), values.size()},
      encodeBuffer);
  const std::string encodedStorage{encoded};

  velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};
  const Encoding::Options options{.bufferPool = &bufferPool};
  Buffer sliceBuffer{*pool};
  const auto sliced = Encoding::slice(
      encodedStorage,
      /*offset=*/17,
      /*length=*/128,
      sliceBuffer,
      options);

  EXPECT_EQ(bufferPool.size(), 0);

  Encoding encoding{*pool, sliced, nullptr, options};
  std::vector<Value> output(128);
  encoding.materialize(static_cast<uint32_t>(output.size()), output.data());
  EXPECT_EQ(
      output, std::vector<Value>(input.begin() + 17, input.begin() + 17 + 128));
}

TYPED_TEST(SimdForBitpackEncodingTest, debugString) {
  std::vector<TypeParam> values;
  values.reserve(8);
  for (uint32_t i = 0; i < 8; ++i) {
    values.push_back(static_cast<TypeParam>(i + 1));
  }
  auto encoding = this->encodeAndCreate(values);
  const std::string debug = encoding->debugString();
  EXPECT_NE(debug.find("SimdForBitpack"), std::string::npos);
  EXPECT_NE(debug.find("bitWidth="), std::string::npos);
}

class SimdForBitpackFuzzerTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  }

  template <typename T>
  void runFuzzer(uint32_t seed, uint32_t numIterations) {
    std::mt19937 rng(seed);

    ManualEncodingSelectionPolicyFactory manualFactory;
    EncodingSelectionPolicyCreator creator =
        [&manualFactory](DataType dataType) {
          return manualFactory.createPolicy(dataType);
        };

    for (uint32_t iter = 0; iter < numIterations; ++iter) {
      SCOPED_TRACE(fmt::format("iteration {}", iter));

      std::uniform_int_distribution<uint32_t> sizeDist(1, 1000);
      const uint32_t numValues = sizeDist(rng);

      std::uniform_int_distribution<uint32_t> valueDist(
          0, sizeof(T) >= 4 ? (1u << 20) : std::numeric_limits<T>::max());
      std::vector<T> values;
      values.reserve(numValues);
      const T offset = static_cast<T>(1000);
      for (uint32_t i = 0; i < numValues; ++i) {
        values.push_back(static_cast<T>(offset + valueDist(rng)));
      }

      Buffer buffer{*pool_};
      EncodingLayout layout{
          EncodingType::SimdForBitpack, {}, CompressionType::Uncompressed};
      auto policy = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
          std::move(layout), CompressionOptions{}, creator);
      auto encoded = EncodingFactory::encode<T>(
          std::move(policy),
          std::span<const T>{values.data(), values.size()},
          buffer);
      std::vector<char> storage(encoded.begin(), encoded.end());
      auto encoding = EncodingFactory().create(
          *pool_, {storage.data(), storage.size()}, nullptr);

      ASSERT_EQ(encoding->encodingType(), EncodingType::SimdForBitpack);
      ASSERT_EQ(encoding->rowCount(), values.size());
      std::vector<T> decoded(values.size());
      encoding->materialize(
          static_cast<uint32_t>(values.size()), decoded.data());
      for (size_t i = 0; i < values.size(); ++i) {
        ASSERT_EQ(decoded[i], values[i])
            << "iter=" << iter << " i=" << i << " size=" << numValues;
      }

      encoding->reset();
      const uint32_t sliceOffset =
          std::uniform_int_distribution<uint32_t>{0, numValues - 1}(rng);
      const uint32_t sliceLength = std::uniform_int_distribution<uint32_t>{
          1, numValues - sliceOffset}(rng);
      Buffer sliceBuffer{*pool_};
      const auto sliced = SimdForBitpackEncoding<T>::slice(
          {storage.data(), storage.size()},
          sliceOffset,
          sliceLength,
          sliceBuffer);
      SimdForBitpackEncoding<T> slicedEncoding{
          *pool_, sliced, nullptr, Encoding::Options{}};
      std::vector<T> slicedOutput(sliceLength);
      slicedEncoding.materialize(sliceLength, slicedOutput.data());
      EXPECT_EQ(
          slicedOutput,
          std::vector<T>(
              values.begin() + sliceOffset,
              values.begin() + sliceOffset + sliceLength));

      encoding->reset();
      std::uniform_int_distribution<uint32_t> stepDist(0, numValues);
      uint32_t cursor = 0;
      while (cursor < numValues) {
        const uint32_t remaining = numValues - cursor;
        const uint32_t skipCount = stepDist(rng) % (remaining + 1);
        if (skipCount > 0) {
          encoding->skip(skipCount);
          cursor += skipCount;
        }
        if (cursor >= numValues) {
          break;
        }
        const uint32_t matCount =
            std::max<uint32_t>(1, stepDist(rng) % (numValues - cursor + 1));
        std::vector<T> chunk(matCount);
        encoding->materialize(matCount, chunk.data());
        for (uint32_t i = 0; i < matCount; ++i) {
          ASSERT_EQ(chunk[i], values[cursor + i])
              << "iter=" << iter << " cursor=" << cursor << " i=" << i;
        }
        cursor += matCount;
      }
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(SimdForBitpackFuzzerTest, fuzzerUint8) {
  runFuzzer<uint8_t>(/*seed=*/11111, /*numIterations=*/30);
}

TEST_F(SimdForBitpackFuzzerTest, fuzzerUint16) {
  runFuzzer<uint16_t>(/*seed=*/22222, /*numIterations=*/30);
}

TEST_F(SimdForBitpackFuzzerTest, fuzzerUint32) {
  runFuzzer<uint32_t>(/*seed=*/33333, /*numIterations=*/30);
}

TEST_F(SimdForBitpackFuzzerTest, fuzzerUint64) {
  runFuzzer<uint64_t>(/*seed=*/44444, /*numIterations=*/30);
}

TEST_F(SimdForBitpackFuzzerTest, fuzzerInt32) {
  runFuzzer<int32_t>(/*seed=*/55555, /*numIterations=*/30);
}

TEST_F(SimdForBitpackFuzzerTest, fuzzerInt64) {
  runFuzzer<int64_t>(/*seed=*/66666, /*numIterations=*/30);
}

TEST_F(SimdForBitpackFuzzerTest, encodeRejectsEmpty) {
  Buffer buffer{*pool_};
  EncodingLayout layout{
      EncodingType::SimdForBitpack, {}, CompressionType::Uncompressed};
  ManualEncodingSelectionPolicyFactory manualFactory;
  EncodingSelectionPolicyCreator creator = [&manualFactory](DataType dataType) {
    return manualFactory.createPolicy(dataType);
  };
  auto policy = std::make_unique<ReplayedEncodingSelectionPolicy<uint32_t>>(
      std::move(layout), CompressionOptions{}, creator);
  std::vector<uint32_t> empty;
  NIMBLE_ASSERT_THROW(
      EncodingFactory::encode<uint32_t>(
          std::move(policy),
          std::span<const uint32_t>{empty.data(), empty.size()},
          buffer),
      "SimdForBitpack encoding cannot be used with 0 rows.");
}

} // namespace facebook::nimble::test
