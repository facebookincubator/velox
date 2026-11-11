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

#include "velox/dwio/nimble/encodings/views/ALPRDEncodingView.h"

#include <gmock/gmock.h>
#include <bit>
#include <limits>
#include <random>

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/tests/EncodingViewTestUtils.h"

namespace facebook::nimble::test {
namespace {

class ALPRDEncodingViewTest : public EncodingViewTest {
 protected:
  template <typename T>
  std::string_view encode(
      const Vector<T>& values,
      EncodingType mainChild,
      EncodingType exceptionChild,
      const Encoding::Options& options) {
    const auto childLayout = [](EncodingType type) {
      if (type == EncodingType::RLE || type == EncodingType::Dictionary) {
        return EncodingLayout{
            type,
            {},
            CompressionType::Uncompressed,
            {std::nullopt, std::nullopt}};
      }
      return EncodingLayout{type, {}, CompressionType::Uncompressed};
    };
    // Varint supports the 32/64-bit children, but not uint16 codes/highs.
    const auto codes = childLayout(
        mainChild == EncodingType::Varint ? EncodingType::Trivial : mainChild);
    const auto right = childLayout(mainChild);
    const auto positions = childLayout(exceptionChild);
    const auto high = childLayout(
        exceptionChild == EncodingType::Varint ? EncodingType::Trivial
                                               : exceptionChild);
    auto policy = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
        EncodingLayout{
            EncodingType::ALPRD,
            {},
            CompressionType::Uncompressed,
            {codes, right, positions, high}},
        std::nullopt,
        [](DataType type) {
          return ManualEncodingSelectionPolicyFactory(
                     {{EncodingType::Trivial, 1.0}}, std::nullopt)
              .createPolicy(type);
        });
    return EncodingFactory::encode<T>(
        std::move(policy), values, *buffer_, options);
  }

  template <typename T>
  void checkReads(
      const Vector<T>& values,
      std::string_view encoded,
      Encoding::Options options) {
    using Physical = typename TypeTraits<T>::physicalType;
    options.cacheEncodingView = cacheEncodingView();
    auto view = createEncodingView(encoded, pool_.get(), options);
    ASSERT_EQ(view->encodingType(), EncodingType::ALPRD);
    ASSERT_EQ(view->dataType(), TypeTraits<T>::dataType);
    ASSERT_EQ(view->rowCount(), values.size());
    Vector<Physical> expected{pool_.get(), values.size()};
    EncodingFactory(options)
        .create(*pool_, encoded, nullptr)
        ->materialize(expected.size(), expected.data());
    for (uint32_t i = 0; i < values.size(); ++i) {
      Physical actual;
      view->readAt(i, &actual);
      EXPECT_EQ(actual, expected[i]) << i;
      EXPECT_EQ(actual, std::bit_cast<Physical>(values[i])) << i;
    }
    const uint32_t numRows = values.size();
    for (uint32_t offset : {0, 1, 127, 255, 256, 257}) {
      expectRangeRead(*view, values, offset, numRows - offset);
    }
    expectRangeRead(*view, values, numRows, 0);
    expectRangeRead(*view, values, 0, 0);
    const std::vector<uint32_t> indices{
        numRows - 1,
        0,
        0,
        1,
        2,
        3,
        257,
        256,
        127,
        128,
        numRows - 1,
    };
    expectIndexedRead(*view, values, indices);
    expectSelectedRead(*view, values, indices);
    expectIndexedRead(*view, values, {});
    Physical output;
    NIMBLE_ASSERT_THROW(view->readAt(numRows, &output), "");
    NIMBLE_ASSERT_THROW(view->read(numRows, 1, &output), "");
    NIMBLE_ASSERT_THROW(view->read(numRows + 1, 0, &output), "");
  }

  template <typename T>
  void checkBitPatterns() {
    SCOPED_TRACE(fmt::format("dataType={}", TypeTraits<T>::dataType));
    using Physical = typename TypeTraits<T>::physicalType;
    auto values = makeVector<T>({
        T{0},
        -T{0},
        std::numeric_limits<T>::infinity(),
        -std::numeric_limits<T>::infinity(),
        std::numeric_limits<T>::quiet_NaN(),
        std::numeric_limits<T>::signaling_NaN(),
        std::numeric_limits<T>::denorm_min(),
        std::numeric_limits<T>::min(),
        std::numeric_limits<T>::max(),
        std::bit_cast<T>(
            std::bit_cast<Physical>(std::numeric_limits<T>::quiet_NaN()) | 37),
    });
    std::mt19937_64 random{19'043};
    for (uint32_t i = 0; i < 4'099; ++i) {
      values.push_back(std::bit_cast<T>(static_cast<Physical>(random())));
    }
    for (const auto useVarint : {false, true}) {
      const Encoding::Options options{.useVarintRowCount = useVarint};
      for (const auto child :
           {EncodingType::Trivial,
            EncodingType::FixedBitWidth,
            EncodingType::RLE,
            EncodingType::Dictionary}) {
        SCOPED_TRACE(fmt::format("varint={} child={}", useVarint, child));
        const auto encoded = encode(values, child, child, options);
        checkReads(values, encoded, options);
      }
    }
  }

  template <typename T>
  void checkNoExceptions() {
    SCOPED_TRACE(fmt::format("dataType={}", TypeTraits<T>::dataType));
    Vector<T> values{pool_.get(), 1'024};
    std::fill(values.begin(), values.end(), T{1.25});
    for (const auto child :
         {EncodingType::Constant, EncodingType::FixedBitWidth}) {
      const auto encoded = encode(values, child, child, {});
      EXPECT_EQ(ALPRDEncodingBase::readMetadata(encoded, {}).exceptionCount, 0);
      checkReads(values, encoded, {});
    }
  }

  template <typename T>
  void checkVarintChildViews() {
    SCOPED_TRACE(fmt::format("dataType={}", TypeTraits<T>::dataType));
    Vector<T> values{pool_.get(), 4'096};
    std::fill(values.begin(), values.end(), T{1.25});
    values.back() = std::numeric_limits<T>::infinity();
    // The final value is outside the training sample and becomes an exception.
    const auto encoded =
        encode(values, EncodingType::FixedBitWidth, EncodingType::Varint, {});
    ASSERT_EQ(ALPRDEncodingBase::readMetadata(encoded, {}).exceptionCount, 1);
    checkReads(values, encoded, {});
    const auto varintMain =
        encode(values, EncodingType::Varint, EncodingType::Trivial, {});
    checkReads(values, varintMain, {});
  }

  template <typename T>
  void checkNullable() {
    SCOPED_TRACE(fmt::format("dataType={}", TypeTraits<T>::dataType));
    using Physical = typename TypeTraits<T>::physicalType;
    const auto values = makeVector<T>(
        {T{0}, -T{0}, std::numeric_limits<T>::signaling_NaN(), T{2.5}});
    for (const auto useVarint : {false, true}) {
      Encoding::Options options{.useVarintRowCount = useVarint};
      const auto data = encode(
          values, EncodingType::FixedBitWidth, EncodingType::Trivial, options);
      const auto notNulls =
          makeVector<bool>({false, true, true, false, true, false, true});
      const auto nulls = Encoder<TrivialEncoding<bool>>::encode(
          *buffer_, notNulls, CompressionType::Uncompressed, options);
      const auto encoded = NullableEncoding<T>::encodeNullable(
          notNulls.size(), data, nulls, *buffer_, options);
      options.cacheEncodingView = cacheEncodingView();
      auto view = createEncodingView(encoded, pool_.get(), options);
      const std::array<uint32_t, 7> indices{4, 0, 1, 2, 4, 5, 6};
      std::array<Physical, 7> output{};
      std::vector<uint32_t> nullIndices;
      EXPECT_EQ(
          view->read(
              indices,
              [&](uint32_t index) { nullIndices.push_back(index); },
              output.data()),
          5);
      EXPECT_THAT(nullIndices, ::testing::ElementsAre(1, 5));
      EXPECT_THAT(
          output,
          ::testing::ElementsAre(
              std::bit_cast<Physical>(values[2]),
              0,
              std::bit_cast<Physical>(values[0]),
              std::bit_cast<Physical>(values[1]),
              std::bit_cast<Physical>(values[2]),
              0,
              std::bit_cast<Physical>(values[3])));
    }
  }
};

TEST_P(ALPRDEncodingViewTest, bitPatterns) {
  checkBitPatterns<float>();
  checkBitPatterns<double>();
}

TEST_P(ALPRDEncodingViewTest, noExceptions) {
  checkNoExceptions<float>();
  checkNoExceptions<double>();
}

TEST_P(ALPRDEncodingViewTest, varintChildViews) {
  checkVarintChildViews<float>();
  checkVarintChildViews<double>();
}

TEST_P(ALPRDEncodingViewTest, nullable) {
  checkNullable<float>();
  checkNullable<double>();
}

TEST_P(ALPRDEncodingViewTest, concurrent) {
  expectConcurrentReads<ALPRDEncoding<float>>(
      randomAlpData<float>(19'043), randomizedPositions(19'220));
  expectConcurrentReads<ALPRDEncoding<double>>(
      randomAlpData<double>(19'043), randomizedPositions(19'220));
}

INSTANTIATE_ENCODING_VIEW_TEST_SUITE(ALPRDEncodingViewTest);

} // namespace
} // namespace facebook::nimble::test
