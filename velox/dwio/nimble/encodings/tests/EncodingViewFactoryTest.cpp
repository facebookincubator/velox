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

#include <array>
#include <tuple>
#include <utility>
#include <vector>

#include "fmt/core.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/BitRangeSplitEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitBoundaries.h"

using namespace facebook;

using EncodingViewFactoryTest = nimble::test::EncodingViewTest;

TEST_P(
    EncodingViewFactoryTest,
    supportsEncodingViewMatchesViewableEncodingSet) {
  const std::vector<nimble::EncodingType> supportedEncodings{
      nimble::EncodingType::Sentinel,
      nimble::EncodingType::Nullable,
      nimble::EncodingType::Constant,
      nimble::EncodingType::Trivial,
      nimble::EncodingType::MainlyConstant,
      nimble::EncodingType::ALP,
      nimble::EncodingType::ALPRD,
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
      nimble::EncodingType::BitRangeSplit,
      nimble::EncodingType::BlockBitPacking,
      nimble::EncodingType::SubIntSplit,
      nimble::EncodingType::Varint,
      nimble::EncodingType::Fsst};
  for (const auto encodingType : supportedEncodings) {
    SCOPED_TRACE(fmt::format("encodingType={}", encodingType));
    EXPECT_TRUE(nimble::supportsEncodingView(encodingType));
  }

  const std::vector<nimble::EncodingType> unsupportedEncodings{
      nimble::EncodingType::Delta,
      nimble::EncodingType::Prefix,
      nimble::EncodingType::FrequencyPartition,
      nimble::EncodingType::SharedDictionary};
  for (const auto encodingType : unsupportedEncodings) {
    SCOPED_TRACE(fmt::format("encodingType={}", encodingType));
    EXPECT_FALSE(nimble::supportsEncodingView(encodingType));
  }
}

TEST_P(EncodingViewFactoryTest, readsNullableEncoding) {
  const auto values = makeVector({10, 30, 50});
  const auto nonNulls = makeVector({true, false, true, false, true});
  const std::array<int32_t, 5> expectedValues{10, 0, 30, 0, 50};
  const std::array<bool, 5> expectedNulls{false, true, false, true, false};

  for (const auto useVarint : {false, true}) {
    SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
    const nimble::Encoding::Options options{
        .useVarintRowCount = useVarint,
        .cacheEncodingView = cacheEncodingView(),
    };
    const auto verify = [&](std::string_view serialized) {
      auto view = nimble::createEncodingView(serialized, pool_.get(), options);
      ASSERT_EQ(view->rowCount(), static_cast<uint32_t>(expectedValues.size()));
      std::array<bool, 5> actualNulls{};
      std::array<int32_t, 5> actualValues{};
      const std::array<nimble::RowRange, 1> ranges{
          nimble::RowRange{0, static_cast<uint32_t>(expectedValues.size())}};
      const auto numNonNull = view->read(
          ranges,
          [&](uint32_t outputIndex) { actualNulls[outputIndex] = true; },
          actualValues.data());
      EXPECT_EQ(numNonNull, 3);
      EXPECT_EQ(actualValues, expectedValues);
      EXPECT_EQ(actualNulls, expectedNulls);
    };

    verify(
        nimble::test::Encoder<nimble::NullableEncoding<int32_t>>::
            encodeNullable(
                *buffer_,
                values,
                nonNulls,
                nimble::CompressionType::Uncompressed,
                options));
  }
}

TEST_P(EncodingViewFactoryTest, readsCompressedTrivialEncoding) {
  const nimble::Encoding::Options options{
      .cacheEncodingView = cacheEncodingView()};
  nimble::Vector<int32_t> values{pool_.get()};
  for (auto i = 0; i < 4'096; ++i) {
    values.push_back(i % 17);
  }
  auto serialized =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Zstd, options);

  auto view = nimble::createEncodingView(serialized, pool_.get(), options);
  ASSERT_EQ(view->rowCount(), values.size());
  for (uint32_t row{0}; row < values.size(); ++row) {
    SCOPED_TRACE(row);
    int32_t value{0};
    view->readAt(row, &value);
    EXPECT_EQ(value, row % 17);
  }
}

TEST_P(EncodingViewFactoryTest, boolViewsCountValuesBefore) {
  // Mostly-false and mostly-true streams make SparseBool store either value as
  // the sparse one.
  struct Pattern {
    const char* name;
    bool (*value)(uint32_t row);
  };
  const std::vector<Pattern> patterns = {
      {"mostlyFalse", [](uint32_t row) { return row % 37 == 5; }},
      {"mostlyTrue", [](uint32_t row) { return row % 37 != 5; }},
      {"alternating", [](uint32_t row) { return row % 2 == 0; }},
      {"runs", [](uint32_t row) { return (row / 50) % 2 == 0; }},
      {"allFalse", [](uint32_t) { return false; }},
  };

  const auto check = [&]<typename Encoding>(const char* encodingName) {
    for (const auto& pattern : patterns) {
      for (const uint32_t numRows : {1U, 64U, 200U, 9'000U}) {
        SCOPED_TRACE(
            fmt::format(
                "encoding={}, pattern={}, numRows={}",
                encodingName,
                pattern.name,
                numRows));
        nimble::Vector<bool> values{pool_.get()};
        for (uint32_t row = 0; row < numRows; ++row) {
          values.push_back(pattern.value(row));
        }
        const auto serialized =
            nimble::test::Encoder<Encoding>::encode(*buffer_, values);
        const auto view = nimble::detail::createTypedEncodingView<bool>(
            serialized,
            pool_.get(),
            nimble::Encoding::Options{
                .cacheEncodingView = cacheEncodingView()});
        ASSERT_EQ(view->rowCount(), numRows);

        // Checks a sample of the end rows when the stream spans several
        // blocks of the default count.
        const uint32_t stride = numRows > 1'000 ? 499 : 1;
        uint32_t numTrue{0};
        for (uint32_t end = 0; end <= numRows; ++end) {
          if (end % stride == 0 || end == numRows) {
            EXPECT_EQ(view->count(end, /*value=*/true), numTrue);
            EXPECT_EQ(view->count(end, /*value=*/false), end - numTrue);
          }
          if (end < numRows) {
            numTrue += values[end] ? 1 : 0;
          }
        }
        std::vector<char> range(numRows);
        view->read(0, numRows, reinterpret_cast<bool*>(range.data()));
        for (uint32_t row = 0; row < numRows; ++row) {
          EXPECT_EQ(static_cast<bool>(range[row]), values[row]);
          EXPECT_EQ(view->readAt(row), values[row]);
        }
      }
    }
  };
  check.operator()<nimble::SparseBoolEncoding>("SparseBool");
  check.operator()<nimble::TrivialEncoding<bool>>("Trivial");
  check.operator()<nimble::RLEEncoding<bool>>("RLE");
}

TEST_P(EncodingViewFactoryTest, countsIntegerValues) {
  constexpr uint32_t kNumRows{3'000};
  nimble::Vector<uint32_t> values{pool_.get()};
  for (uint32_t row = 0; row < kNumRows; ++row) {
    values.push_back(row % 7);
  }
  const auto serialized =
      nimble::test::Encoder<nimble::TrivialEncoding<uint32_t>>::encode(
          *buffer_, values);
  const auto view = nimble::detail::createTypedEncodingView<uint32_t>(
      serialized,
      pool_.get(),
      nimble::Encoding::Options{.cacheEncodingView = cacheEncodingView()});

  // The rows span several blocks of the default count, so the sampled ends fall
  // inside blocks and on their edges. The expected count grows by one for each
  // written row that equals the value.
  uint32_t numThrees{0};
  for (uint32_t end = 0; end <= kNumRows; ++end) {
    if (end % 499 == 0 || end == kNumRows) {
      SCOPED_TRACE(fmt::format("end={}", end));
      EXPECT_EQ(view->count(end, /*value=*/3), numThrees);
    }
    if (end < kNumRows) {
      numThrees += values[end] == 3 ? 1 : 0;
    }
  }

  // An end past the row count is rejected.
  NIMBLE_ASSERT_THROW(
      view->count(kNumRows + 1, /*value=*/3), "(3001 vs. 3000)");
}

TEST_P(EncodingViewFactoryTest, sumsIntegerValues) {
  constexpr uint32_t kNumRows{3'000};
  nimble::Vector<uint32_t> values{pool_.get()};
  std::vector<uint64_t> sumsBefore{0};
  for (uint32_t row = 0; row < kNumRows; ++row) {
    values.push_back(row % 13 + 1);
    sumsBefore.emplace_back(sumsBefore.back() + values[row]);
  }
  const auto serialized =
      nimble::test::Encoder<nimble::TrivialEncoding<uint32_t>>::encode(
          *buffer_, values);
  const auto view = nimble::detail::createTypedEncodingView<uint32_t>(
      serialized,
      pool_.get(),
      nimble::Encoding::Options{.cacheEncodingView = cacheEncodingView()});

  {
    // The ranges start and end on both sides of a block boundary, and some are
    // empty.
    const std::vector<std::pair<uint32_t, uint32_t>> ranges{
        {0, 0},
        {5, 5},
        {0, 1},
        {1'023, 1'025},
        {1'024, 2'048},
        {7, 2'999},
        {0, kNumRows},
        {kNumRows, kNumRows}};
    for (const auto& [begin, end] : ranges) {
      SCOPED_TRACE(fmt::format("begin={}, end={}", begin, end));
      // The sums before the two ends were built while writing the values, so
      // their difference is the sum of the range.
      EXPECT_EQ(view->sum(begin, end), sumsBefore[end] - sumsBefore[begin]);
    }
    // A range that ends before it begins, and one that ends past the row
    // count, are rejected.
    NIMBLE_ASSERT_THROW(view->sum(5, 4), "(5 vs. 4)");
    NIMBLE_ASSERT_THROW(view->sum(0, kNumRows + 1), "(3001 vs. 3000)");
  }

  {
    // The indices are unordered and repeated.
    const std::vector<uint32_t> indices{
        2'500, 0, 7, 2'500, 1'024, kNumRows, 1'023};
    std::vector<uint64_t> actual(indices.size());
    view->sum(indices, actual.data());
    // Each result is the sum before its index, in the order the indices were
    // given, although the view reads the rows once in ascending order.
    std::vector<uint64_t> expected;
    expected.reserve(indices.size());
    for (const auto index : indices) {
      expected.emplace_back(sumsBefore[index]);
    }
    EXPECT_EQ(actual, expected);

    // An index past the row count is rejected.
    const std::vector<uint32_t> pastRowCount{kNumRows + 1};
    NIMBLE_ASSERT_THROW(
        view->sum(pastRowCount, actual.data()), "(3001 vs. 3000)");
  }
}

TEST_P(EncodingViewFactoryTest, rejectsNonBoolStreamAsBoolView) {
  nimble::Vector<int32_t> values{pool_.get()};
  for (int32_t row = 0; row < 10; ++row) {
    values.push_back(row);
  }
  const auto serialized =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);
  NIMBLE_ASSERT_THROW(
      nimble::detail::createTypedEncodingView<bool>(
          serialized,
          pool_.get(),
          nimble::Encoding::Options{.cacheEncodingView = cacheEncodingView()}),
      "A bool encoding view requires a bool stream.");
}

TEST_P(EncodingViewFactoryTest, parsesArbitraryBitRangeSplitBoundaries) {
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

TEST_P(EncodingViewFactoryTest, readsSubIntSplitEncoding) {
  const std::vector<uint64_t> values{
      0x299aff8ca62b0001,
      0x299be75117820001,
      0x0999c1e5b8460001,
      0x299f100bfa830002,
  };
  const std::vector<nimble::subintsplit::SectionPlan> segments{
      {.bitStart = 0, .bitEnd = 15},
      {.bitStart = 16, .bitEnd = 58},
      {.bitStart = 59, .bitEnd = 63},
  };
  std::vector<std::optional<const nimble::EncodingLayout>> children(
      segments.size());
  nimble::EncodingLayout layout{
      nimble::EncodingType::SubIntSplit,
      nimble::EncodingLayout::Config{
          nimble::subintsplit::makePreserveSplitConfig(segments)},
      nimble::CompressionType::Uncompressed,
      std::move(children)};
  const nimble::EncodingSelectionPolicyCreator leafPolicyCreator =
      [](nimble::DataType dataType)
      -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    auto readFactors = nimble::ManualEncodingSelectionPolicyFactory::
        defaultEncodingReadFactors();
    readFactors.erase(
        std::remove_if(
            readFactors.begin(),
            readFactors.end(),
            [](const auto& factor) {
              return factor.first == nimble::EncodingType::Varint;
            }),
        readFactors.end());
    return nimble::ManualEncodingSelectionPolicyFactory{
        std::move(readFactors), std::nullopt}
        .createPolicy(dataType);
  };
  const auto encoded = nimble::EncodingFactory::encode<uint64_t>(
      std::make_unique<nimble::ReplayedEncodingSelectionPolicy<uint64_t>>(
          std::move(layout), std::nullopt, leafPolicyCreator),
      values,
      *buffer_);

  const auto captured = nimble::EncodingLayoutCapture::capture(encoded, {});
  EXPECT_EQ(captured.encodingType(), nimble::EncodingType::SubIntSplit);
  EXPECT_EQ(captured.childrenCount(), segments.size());
  EXPECT_EQ(
      captured.config().get(
          std::string(nimble::subintsplit::kSplitBoundariesConfigKey)),
      "0-15;16-58;59-63");

  auto view = nimble::createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});
  EXPECT_EQ(view->rowCount(), values.size());
  for (uint32_t row{0}; row < values.size(); ++row) {
    uint64_t actual{0};
    view->readAt(row, &actual);
    EXPECT_EQ(actual, values[row]);
  }

  const std::vector<uint32_t> indices{3, 0, 2, 2};
  std::vector<uint64_t> actual(indices.size());
  view->readAt(indices, actual.data());
  EXPECT_EQ(
      actual,
      (std::vector<uint64_t>{values[3], values[0], values[2], values[2]}));
}

TEST_P(EncodingViewFactoryTest, readsConfigurableBitRangeSplitSections) {
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

    auto view = nimble::createEncodingView(
        encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});
    const std::vector<uint32_t> indices{3, 0, 2, 2};
    std::vector<uint64_t> actual(indices.size());
    view->readAt(indices, actual.data());
    EXPECT_EQ(
        actual,
        (std::vector<uint64_t>{values[3], values[0], values[2], values[2]}));
  }
}

TEST_P(EncodingViewFactoryTest, rejectsInvalidBitRangeSplitConfig) {
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

TEST_P(EncodingViewFactoryTest, readsNestedBitRangeSplitSections) {
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

  auto view = nimble::createEncodingView(
      encoded, pool_.get(), {.cacheEncodingView = cacheEncodingView()});
  const std::vector<uint32_t> indices{3, 0, 2, 2};
  std::vector<uint64_t> actual(indices.size());
  view->readAt(indices, actual.data());
  EXPECT_EQ(
      actual,
      (std::vector<uint64_t>{values[3], values[0], values[2], values[2]}));
}

TEST_P(EncodingViewFactoryTest, rejectsVarintWithSmallDataType) {
  const nimble::Encoding::Options options{
      .cacheEncodingView = cacheEncodingView()};
  std::string serialized(nimble::EncodingPrefix::kFixedPrefixSize, 0);
  char* pos = serialized.data();
  nimble::encoding::writeChar(
      static_cast<char>(nimble::EncodingType::Varint), pos);
  nimble::encoding::writeChar(static_cast<char>(nimble::DataType::Int16), pos);
  nimble::encoding::writeUint32(/*value=*/1, pos);

  NIMBLE_ASSERT_THROW(
      nimble::createEncodingView(serialized, pool_.get(), options),
      "Varint encoding only supports 32- and 64-bit numeric data types");
}

TEST_P(EncodingViewFactoryTest, rejectsUnsupportedEncodingTypes) {
  const std::vector<std::pair<nimble::EncodingType, nimble::DataType>>
      unsupportedEncodings{
          {nimble::EncodingType::Delta, nimble::DataType::Int32},
          {nimble::EncodingType::Prefix, nimble::DataType::String},
          {nimble::EncodingType::FrequencyPartition, nimble::DataType::Uint32},
      };

  const nimble::Encoding::Options options{
      .cacheEncodingView = cacheEncodingView()};
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

TEST_P(EncodingViewFactoryTest, rejectsUnsupportedDataType) {
  const nimble::Encoding::Options options{
      .cacheEncodingView = cacheEncodingView()};
  std::string serialized(nimble::EncodingPrefix::kFixedPrefixSize, 0);
  serialized[nimble::EncodingPrefix::kEncodingTypeOffset] =
      static_cast<char>(nimble::EncodingType::Trivial);
  serialized[nimble::EncodingPrefix::kDataTypeOffset] =
      static_cast<char>(nimble::DataType::Undefined);

  NIMBLE_ASSERT_THROW(
      nimble::createEncodingView(serialized, pool_.get(), options),
      "does not support EncodingView");
}

TEST_P(EncodingViewFactoryTest, rejectsFixedBitWidthWithNonNumericDataType) {
  const nimble::Encoding::Options options{
      .cacheEncodingView = cacheEncodingView()};
  std::string serialized(nimble::EncodingPrefix::kFixedPrefixSize, 0);
  serialized[nimble::EncodingPrefix::kEncodingTypeOffset] =
      static_cast<char>(nimble::EncodingType::FixedBitWidth);
  serialized[nimble::EncodingPrefix::kDataTypeOffset] =
      static_cast<char>(nimble::DataType::String);

  NIMBLE_ASSERT_USER_THROW(
      nimble::createEncodingView(serialized, pool_.get(), options),
      "FixedBitWidth encoding should not be selected for non-numeric data types");
}

INSTANTIATE_ENCODING_VIEW_TEST_SUITE(EncodingViewFactoryTest);
