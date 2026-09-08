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

#include "velox/dwio/nimble/encodings/tests/EncodingViewTestUtils.h"

#include <gtest/gtest.h>

#include <tuple>
#include <utility>
#include <vector>

#include "fmt/core.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/BitRangeSplitEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, supportsEncodingViewMatchesViewableEncodingSet) {
  const std::vector<nimble::EncodingType> supportedEncodings{
      nimble::EncodingType::Constant,
      nimble::EncodingType::Trivial,
      nimble::EncodingType::MainlyConstant,
      nimble::EncodingType::ALP,
      nimble::EncodingType::FixedBitWidth,
      nimble::EncodingType::Dictionary,
      nimble::EncodingType::SparseBool,
      nimble::EncodingType::RLE,
      nimble::EncodingType::FOR,
      nimble::EncodingType::DeltaBlock,
      nimble::EncodingType::EliasFano,
      nimble::EncodingType::Huffman,
      nimble::EncodingType::PFOR,
      nimble::EncodingType::SimdForBitpack,
      nimble::EncodingType::SubIntSplit,
      nimble::EncodingType::BitRangeSplit,
      nimble::EncodingType::BlockBitPacking};
  for (const auto encodingType : supportedEncodings) {
    SCOPED_TRACE(fmt::format("encodingType={}", encodingType));
    EXPECT_TRUE(nimble::supportsEncodingView(encodingType));
  }

  const std::vector<nimble::EncodingType> unsupportedEncodings{
      nimble::EncodingType::Sentinel,
      nimble::EncodingType::Nullable,
      nimble::EncodingType::Varint,
      nimble::EncodingType::Delta,
      nimble::EncodingType::Prefix,
      nimble::EncodingType::FrequencyPartition,
      nimble::EncodingType::Fsst,
      nimble::EncodingType::SharedDictionary};
  for (const auto encodingType : unsupportedEncodings) {
    SCOPED_TRACE(fmt::format("encodingType={}", encodingType));
    EXPECT_FALSE(nimble::supportsEncodingView(encodingType));
  }
}

TEST_F(EncodingViewTest, rejectsCompressedTrivialEncoding) {
  const nimble::Encoding::Options options;
  nimble::Vector<int32_t> values{pool_.get()};
  for (auto i = 0; i < 4096; ++i) {
    values.push_back(i % 17);
  }
  auto serialized =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Zstd, options);

  NIMBLE_ASSERT_THROW(
      nimble::createEncodingView(serialized, pool_.get(), options),
      "EncodingView does not support compressed Trivial streams");
}

TEST_F(EncodingViewTest, parsesArbitraryBitRangeSplitBoundaries) {
  const auto sections =
      nimble::detail::BitRangeSplitEncodingBase::parseSections(
          "0-12;13-46;47-63", /*physicalBits=*/64);
  std::vector<std::tuple<uint8_t, uint8_t, uint8_t>> actual;
  actual.reserve(sections.size());
  for (const auto& section : sections) {
    actual.emplace_back(section.bitStart, section.bitEnd, section.storageBytes);
  }
  EXPECT_EQ(
      actual,
      (std::vector<std::tuple<uint8_t, uint8_t, uint8_t>>{
          {0, 12, 2},
          {13, 46, 8},
          {47, 63, 4},
      }));

  NIMBLE_ASSERT_THROW(
      nimble::detail::BitRangeSplitEncodingBase::parseSections(
          "0-12;14-63", /*physicalBits=*/64),
      "not contiguous");
  NIMBLE_ASSERT_THROW(
      nimble::detail::BitRangeSplitEncodingBase::parseSections(
          "start-12;13-63", /*physicalBits=*/64),
      "Invalid BitRangeSplit range config");
}

TEST_F(EncodingViewTest, readsConfigurableBitRangeSplitSections) {
  const std::vector<uint64_t> values{
      0x299aff8ca62b0001,
      0x299be75117820001,
      0x0999c1e5b8460001,
      0x299f100bfa830002,
  };
  const nimble::EncodingSelectionPolicyCreator policyCreator =
      [](nimble::DataType dataType)
      -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    return nimble::ManualEncodingSelectionPolicyFactory{}.createPolicy(
        dataType);
  };
  std::vector<std::pair<std::string, uint8_t>> rangeConfigs{
      {"0-63", 1},
      {"0-31;32-63", 2},
      {"0-15;16-58;59-63", 3},
      {"0-7;8-23;24-47;48-63", 4},
  };
  std::string singleBitRanges;
  for (uint8_t bit{0}; bit < 64; ++bit) {
    if (!singleBitRanges.empty()) {
      singleBitRanges.push_back(';');
    }
    singleBitRanges += fmt::format("{}-{}", bit, bit);
  }
  rangeConfigs.emplace_back(std::move(singleBitRanges), 64);

  for (const auto& [rangesConfig, numSections] : rangeConfigs) {
    SCOPED_TRACE(fmt::format("sectionCount={}", numSections));
    const nimble::EncodingLayout layout{
        nimble::EncodingType::BitRangeSplit,
        nimble::EncodingLayout::Config{{
            {std::string(
                 nimble::BitRangeSplitEncoding<uint64_t>::kRangesConfigKey),
             rangesConfig},
        }},
        nimble::CompressionType::Uncompressed,
        std::vector<std::optional<const nimble::EncodingLayout>>(numSections)};
    const auto encoded = nimble::EncodingFactory::encode<uint64_t>(
        std::make_unique<nimble::ReplayedEncodingSelectionPolicy<uint64_t>>(
            layout, std::nullopt, policyCreator),
        values,
        *buffer_,
        {});

    auto decoded = nimble::EncodingFactory().create(
        *pool_, encoded, nullptr, nimble::Encoding::Options{});
    std::vector<uint64_t> materialized(values.size());
    decoded->materialize(
        static_cast<uint32_t>(values.size()), materialized.data());
    EXPECT_EQ(materialized, values);

    auto view = nimble::createEncodingView(encoded, pool_.get());
    const std::vector<uint32_t> indices{3, 0, 2, 2};
    std::vector<uint64_t> actual(indices.size());
    view->readAt(indices, actual.data());
    EXPECT_EQ(
        actual,
        (std::vector<uint64_t>{values[3], values[0], values[2], values[2]}));
  }
}

TEST_F(EncodingViewTest, rejectsInvalidBitRangeSplitConfig) {
  const std::vector<uint64_t> values{1, 2, 3, 4};
  const nimble::EncodingSelectionPolicyCreator policyCreator =
      [](nimble::DataType dataType)
      -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    return nimble::ManualEncodingSelectionPolicyFactory{}.createPolicy(
        dataType);
  };
  const auto encode = [&](nimble::EncodingLayout::Config config) {
    const nimble::EncodingLayout layout{
        nimble::EncodingType::BitRangeSplit,
        std::move(config),
        nimble::CompressionType::Uncompressed,
        std::vector<std::optional<const nimble::EncodingLayout>>(2)};
    return nimble::EncodingFactory::encode<uint64_t>(
        std::make_unique<nimble::ReplayedEncodingSelectionPolicy<uint64_t>>(
            layout, std::nullopt, policyCreator),
        values,
        *buffer_,
        {});
  };

  NIMBLE_ASSERT_THROW(
      encode({}), "requires the 'bit-range-split.ranges' encoding config");
  NIMBLE_ASSERT_THROW(
      encode(
          nimble::EncodingLayout::Config{{
              {std::string(
                   nimble::BitRangeSplitEncoding<uint64_t>::kRangesConfigKey),
               "0-15;17-63"},
          }}),
      "not contiguous");
}

TEST_F(EncodingViewTest, readsNestedBitRangeSplitSections) {
  const std::vector<uint64_t> values{
      0x299aff8ca62b0001,
      0x299be75117820001,
      0x0999c1e5b8460001,
      0x299f100bfa830002,
  };
  constexpr std::string_view kRangesConfig = "0-15;16-58;59-63";
  constexpr uint8_t kNumSections = 3;
  const nimble::EncodingLayout layout{
      nimble::EncodingType::BitRangeSplit,
      nimble::EncodingLayout::Config{{
          {std::string(
               nimble::BitRangeSplitEncoding<uint64_t>::kRangesConfigKey),
           std::string(kRangesConfig)},
      }},
      nimble::CompressionType::Uncompressed,
      std::vector<std::optional<const nimble::EncodingLayout>>(kNumSections)};
  const nimble::EncodingSelectionPolicyCreator policyCreator =
      [](nimble::DataType dataType)
      -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    return nimble::ManualEncodingSelectionPolicyFactory{}.createPolicy(
        dataType);
  };
  const auto encoded = nimble::EncodingFactory::encode<uint64_t>(
      std::make_unique<nimble::ReplayedEncodingSelectionPolicy<uint64_t>>(
          layout, std::nullopt, policyCreator),
      values,
      *buffer_,
      {});

  const auto captured = nimble::EncodingLayoutCapture::capture(encoded, {});
  EXPECT_EQ(captured.encodingType(), nimble::EncodingType::BitRangeSplit);
  EXPECT_EQ(captured.childrenCount(), kNumSections);
  EXPECT_EQ(
      captured.config().get(
          std::string(
              nimble::BitRangeSplitEncoding<uint64_t>::kRangesConfigKey)),
      kRangesConfig);

  auto decoded = nimble::EncodingFactory().create(
      *pool_, encoded, nullptr, nimble::Encoding::Options{});
  std::vector<uint64_t> materialized(values.size());
  decoded->materialize(
      static_cast<uint32_t>(values.size()), materialized.data());
  EXPECT_EQ(materialized, values);

  auto view = nimble::createEncodingView(encoded, pool_.get());
  const std::vector<uint32_t> indices{3, 0, 2, 2};
  std::vector<uint64_t> actual(indices.size());
  view->readAt(indices, actual.data());
  EXPECT_EQ(
      actual,
      (std::vector<uint64_t>{values[3], values[0], values[2], values[2]}));
}

TEST_F(EncodingViewTest, rejectsVarintEncoding) {
  const nimble::Encoding::Options options;
  const auto values = makeVector({10, 11, 12, 1000, 1001});
  auto serialized =
      nimble::test::Encoder<nimble::VarintEncoding<int32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Uncompressed, options);

  NIMBLE_ASSERT_THROW(
      nimble::createEncodingView(serialized, pool_.get(), options),
      "Varint does not support EncodingView");
}

TEST_F(EncodingViewTest, rejectsUnsupportedEncodingTypes) {
  const std::vector<std::pair<nimble::EncodingType, nimble::DataType>>
      unsupportedEncodings{
          {nimble::EncodingType::Sentinel, nimble::DataType::Int32},
          {nimble::EncodingType::Nullable, nimble::DataType::Int32},
          {nimble::EncodingType::Delta, nimble::DataType::Int32},
          {nimble::EncodingType::Prefix, nimble::DataType::String},
          {nimble::EncodingType::FrequencyPartition, nimble::DataType::Uint32},
          {nimble::EncodingType::Fsst, nimble::DataType::String},
      };

  const nimble::Encoding::Options options;
  for (const auto& [encodingType, dataType] : unsupportedEncodings) {
    SCOPED_TRACE(fmt::format("encodingType={}", encodingType));
    std::string serialized(nimble::EncodingPrefix::kFixedPrefixSize, 0);
    char* pos = serialized.data();
    nimble::encoding::writeChar(static_cast<char>(encodingType), pos);
    nimble::encoding::writeChar(static_cast<char>(dataType), pos);
    nimble::encoding::writeUint32(/*value=*/1, pos);

    NIMBLE_ASSERT_THROW(
        nimble::createEncodingView(serialized, pool_.get(), options),
        "does not support EncodingView");
  }
}

TEST_F(EncodingViewTest, rejectsUnsupportedDataType) {
  const nimble::Encoding::Options options;
  std::string serialized(nimble::EncodingPrefix::kFixedPrefixSize, 0);
  serialized[nimble::EncodingPrefix::kEncodingTypeOffset] =
      static_cast<char>(nimble::EncodingType::Trivial);
  serialized[nimble::EncodingPrefix::kDataTypeOffset] =
      static_cast<char>(nimble::DataType::Undefined);

  NIMBLE_ASSERT_THROW(
      nimble::createEncodingView(serialized, pool_.get(), options),
      "does not support EncodingView");
}

TEST_F(EncodingViewTest, rejectsFixedBitWidthWithNonNumericDataType) {
  const nimble::Encoding::Options options;
  std::string serialized(nimble::EncodingPrefix::kFixedPrefixSize, 0);
  serialized[nimble::EncodingPrefix::kEncodingTypeOffset] =
      static_cast<char>(nimble::EncodingType::FixedBitWidth);
  serialized[nimble::EncodingPrefix::kDataTypeOffset] =
      static_cast<char>(nimble::DataType::String);

  NIMBLE_ASSERT_USER_THROW(
      nimble::createEncodingView(serialized, pool_.get(), options),
      "FixedBitWidth encoding should not be selected for non-numeric data types");
}
