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
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <string>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;
using namespace facebook::nimble;

namespace {

Encoding::Options exactBitWidthOptions() {
  Encoding::Options options;
  options.fixedBitWidthUseExactBits = true;
  return options;
}

Encoding::Options byteRoundedBitWidthOptions() {
  Encoding::Options options;
  options.fixedBitWidthUseExactBits = false;
  return options;
}

template <typename T>
uint64_t fixedBitWidthEstimate(
    uint64_t rowCount,
    uint64_t minValue,
    uint64_t maxValue,
    bool roundBitWidthToByte) {
  auto bitWidth = velox::bits::bitsRequired(maxValue - minValue);
  if (roundBitWidthToByte) {
    bitWidth = velox::bits::roundUp(bitWidth, 8);
  }
  return EncodingPrefix::kFixedPrefixSize + 2 + sizeof(T) +
      velox::bits::nbytes(bitWidth * rowCount);
}

} // namespace

class EncodingSizeEstimationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  }

  template <typename T>
  void verifySimdForBitpackEstimateVsActualSize() {
    using Est = detail::EncodingSizeEstimation<T>;

    const T min = std::numeric_limits<T>::min();
    const T max = std::numeric_limits<T>::max();
    const T base = static_cast<T>(max / 4);
    const std::vector<std::vector<T>> testCases = {
        std::vector<T>(1000, min),
        std::vector<T>(1000, max),
        {min, max},
        [&] {
          std::vector<T> data;
          data.reserve(1000);
          for (uint32_t i = 0; i < 1000; ++i) {
            data.push_back(static_cast<T>(base + i % 50));
          }
          return data;
        }(),
    };

    for (const auto& data : testCases) {
      SCOPED_TRACE(static_cast<uint64_t>(max));
      SCOPED_TRACE(static_cast<uint64_t>(data.front()));
      auto stats = Statistics<T>::create(data);
      const uint32_t numValues = static_cast<uint32_t>(data.size());

      auto estimated = Est::estimateSize(
          EncodingType::SimdForBitpack, numValues, stats, defaultOptions_);
      ASSERT_TRUE(estimated.has_value());

      nimble::Buffer buffer{*pool_};
      auto encoded =
          nimble::test::Encoder<nimble::SimdForBitpackEncoding<T>>::encode(
              buffer, nimble::Vector<T>(pool_.get(), data.begin(), data.end()));
      const uint64_t actualSize = encoded.size();

      EXPECT_GT(estimated.value(), actualSize / 2)
          << "estimate too low: " << estimated.value() << " vs actual "
          << actualSize;
      EXPECT_LT(estimated.value(), actualSize * 2)
          << "estimate too high: " << estimated.value() << " vs actual "
          << actualSize;
    }
  }

  template <typename T>
  void verifyBlockBitPackingEstimateVsActualSize() {
    std::vector<T> data;
    data.reserve(2050);
    for (uint32_t i = 0; i < 1024; ++i) {
      data.push_back(static_cast<T>(1000 + i % 4));
    }
    for (uint32_t i = 0; i < 1024; ++i) {
      data.push_back(static_cast<T>(1'000'000 + i % 8));
    }
    data.push_back(std::numeric_limits<T>::min());
    data.push_back(std::numeric_limits<T>::max());

    const auto values = std::span<const T>{data.data(), data.size()};
    const auto estimated = BlockBitPackingEncoding<T>::estimateSize(values);
    ASSERT_TRUE(estimated.has_value());

    nimble::Buffer buffer{*pool_};
    auto encoded =
        nimble::test::Encoder<nimble::BlockBitPackingEncoding<T>>::encode(
            buffer, nimble::Vector<T>(pool_.get(), data.begin(), data.end()));

    // estimateSize() estimates the per-block metadata as Trivial sub-encodings,
    // while the actual encoding routes that metadata through nested encoding
    // selection (whose size is data-dependent), so the estimate is an
    // approximation rather than exact. Allow a 2x band, matching the other
    // estimate checks.
    EXPECT_GT(estimated.value(), encoded.size() / 2)
        << "estimate too low: " << estimated.value() << " vs actual "
        << encoded.size();
    EXPECT_LT(estimated.value(), encoded.size() * 2)
        << "estimate too high: " << estimated.value() << " vs actual "
        << encoded.size();
  }

  template <typename T>
  void verifyAlpEstimateVsActualSize() {
    using physicalType = typename TypeTraits<T>::physicalType;
    using Est = detail::EncodingSizeEstimation<T>;
    SCOPED_TRACE(toString(TypeTraits<T>::dataType));

    std::vector<T> data;
    data.reserve(512);
    for (int32_t i = 0; i < 512; ++i) {
      data.push_back(static_cast<T>((i % 33) - 16) / static_cast<T>(4));
    }

    const auto physicalValues =
        EncodingPhysicalType<T>::asEncodingPhysicalTypeSpan(
            std::span<const T>{data.data(), data.size()});
    auto stats = Statistics<physicalType>::create(physicalValues);

    EXPECT_FALSE(
        Est::estimateSize(
            EncodingType::ALP, data.size(), stats, defaultOptions_)
            .has_value());

    const auto estimated = Est::estimateSize(
        EncodingType::ALP, physicalValues, stats, defaultOptions_);
    ASSERT_TRUE(estimated.has_value());

    auto policy = std::make_unique<ManualEncodingSelectionPolicy<T>>(
        std::vector<std::pair<EncodingType, float>>{
            {EncodingType::FixedBitWidth, 1.0},
        },
        std::nullopt,
        std::nullopt);
    EncodingSelection<physicalType> selection{
        {.encodingType = EncodingType::ALP},
        Statistics<physicalType>::create(physicalValues),
        std::move(policy)};

    Buffer buffer{*pool_};
    const auto encoded = ALPEncoding<T>::encode(
        selection, physicalValues, buffer, defaultOptions_);

    EXPECT_GT(estimated.value(), encoded.size() / 2)
        << "estimate too low: " << estimated.value() << " vs actual "
        << encoded.size();
    EXPECT_LT(estimated.value(), encoded.size() * 2)
        << "estimate too high: " << estimated.value() << " vs actual "
        << encoded.size();
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  const Encoding::Options defaultOptions_;
};

TEST_F(EncodingSizeEstimationTest, numericConstant) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data(100, 42);
  auto stats = Statistics<uint32_t>::create(data);

  auto size =
      Est::estimateSize(EncodingType::Constant, data, stats, defaultOptions_);
  ASSERT_TRUE(size.has_value());
  EXPECT_GT(size.value(), 0);
  EXPECT_LT(size.value(), 100 * sizeof(uint32_t));
}

TEST_F(EncodingSizeEstimationTest, numericConstantNonConstantData) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data = {1, 2, 3};
  auto stats = Statistics<uint32_t>::create(data);

  auto size =
      Est::estimateSize(EncodingType::Constant, 3, stats, defaultOptions_);
  EXPECT_FALSE(size.has_value());
}

TEST_F(EncodingSizeEstimationTest, numericTrivial) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data = {1, 2, 3, 4, 5};
  auto stats = Statistics<uint32_t>::create(data);

  auto size =
      Est::estimateSize(EncodingType::Trivial, 5, stats, defaultOptions_);
  ASSERT_TRUE(size.has_value());
  EXPECT_GE(size.value(), 5 * sizeof(uint32_t));
}

TEST_F(EncodingSizeEstimationTest, numericFixedBitWidth) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data = {100, 101, 102, 103, 104};
  auto stats = Statistics<uint32_t>::create(data);

  auto size =
      Est::estimateSize(EncodingType::FixedBitWidth, 5, stats, defaultOptions_);
  ASSERT_TRUE(size.has_value());
  EXPECT_LT(size.value(), 5 * sizeof(uint32_t));
}

// Dense/high-cardinality data (most rows uncommon): the other-values stream is
// priced from full-column statistics via the cheap FixedBitWidth over-estimate,
// so MainlyConstant is over-priced relative to the flat encodings.
TEST_F(EncodingSizeEstimationTest, mainlyConstantOverPricedForDenseData) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data(300, 7); // 30% common, 70% uncommon & distinct
  data.reserve(1000);
  for (uint32_t i = 0; i < 700; ++i) {
    data.push_back(1000 + i);
  }
  const auto stats = Statistics<uint32_t>::create(data);

  const size_t uncommonCount = 700;
  const auto mc = Est::estimateSize(
      EncodingType::MainlyConstant, data.size(), stats, defaultOptions_);
  const auto fixedBitWidth = Est::estimateSize(
      EncodingType::FixedBitWidth, uncommonCount, stats, defaultOptions_);
  const auto trivial = Est::estimateSize(
      EncodingType::Trivial, uncommonCount, stats, defaultOptions_);

  ASSERT_TRUE(mc.has_value());
  ASSERT_TRUE(fixedBitWidth.has_value());
  ASSERT_TRUE(trivial.has_value());
  // Priced as the flat other-values stream plus the is-common bitmap, so the
  // estimate exceeds the flat other-values alone.
  EXPECT_GT(mc.value(), std::min(fixedBitWidth.value(), trivial.value()));
}

TEST_F(EncodingSizeEstimationTest, fixedBitWidthEstimateUsesExactBits) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  const std::vector<uint32_t> data = {100, 101, 102, 103, 104};
  auto stats = Statistics<uint32_t>::create(data);
  Encoding::Options options;
  options.fixedBitWidthUseExactBits = true;

  const auto size = Est::estimateSize(
      EncodingType::FixedBitWidth, data.size(), stats, options);

  ASSERT_TRUE(size.has_value());
  EXPECT_EQ(
      size.value(),
      fixedBitWidthEstimate<uint32_t>(
          data.size(), stats.min(), stats.max(), false));
  EXPECT_LT(
      size.value(),
      fixedBitWidthEstimate<uint32_t>(
          data.size(), stats.min(), stats.max(), true));
}

TEST_F(EncodingSizeEstimationTest, floatingDictionaryEstimateUsesNestedAlp) {
  auto verify = [&](auto typeTag) {
    using T = decltype(typeTag);
    using physicalType = typename TypeTraits<T>::physicalType;
    using Est = detail::EncodingSizeEstimation<T>;
    SCOPED_TRACE(toString(TypeTraits<T>::dataType));

    std::vector<T> data;
    data.reserve(4096);
    for (int32_t i = 0; i < 4096; ++i) {
      data.push_back(static_cast<T>((i % 1024) - 512) / static_cast<T>(100));
    }
    const auto physicalValues =
        EncodingPhysicalType<T>::asEncodingPhysicalTypeSpan(
            std::span<const T>{data.data(), data.size()});
    const auto stats = Statistics<physicalType>::create(physicalValues);

    auto nestedAlpOptions = defaultOptions_;
    nestedAlpOptions.allowNestedAlpSelection = true;
    const auto defaultEstimate = Est::estimateSize(
        EncodingType::Dictionary,
        physicalValues.size(),
        stats,
        defaultOptions_);
    const auto nestedAlpEstimate = Est::estimateSize(
        EncodingType::Dictionary,
        physicalValues.size(),
        stats,
        nestedAlpOptions);

    ASSERT_TRUE(defaultEstimate.has_value());
    ASSERT_TRUE(nestedAlpEstimate.has_value());
    EXPECT_LT(nestedAlpEstimate.value(), defaultEstimate.value());
  };

  verify(float{});
  verify(double{});
}

TEST_F(EncodingSizeEstimationTest, floatingRleEstimateUsesNestedAlp) {
  auto verify = [&](auto typeTag) {
    using T = decltype(typeTag);
    using physicalType = typename TypeTraits<T>::physicalType;
    using Est = detail::EncodingSizeEstimation<T>;
    SCOPED_TRACE(toString(TypeTraits<T>::dataType));

    std::vector<T> data;
    data.reserve(4096);
    for (int32_t i = 0; i < 1024; ++i) {
      const auto value = static_cast<T>((i % 257) - 128) / static_cast<T>(100);
      data.push_back(value);
      data.push_back(value);
      data.push_back(value);
      data.push_back(value);
    }
    const auto physicalValues =
        EncodingPhysicalType<T>::asEncodingPhysicalTypeSpan(
            std::span<const T>{data.data(), data.size()});
    const auto stats = Statistics<physicalType>::create(physicalValues);

    auto nestedAlpOptions = defaultOptions_;
    nestedAlpOptions.allowNestedAlpSelection = true;
    const auto defaultEstimate = Est::estimateSize(
        EncodingType::RLE, physicalValues, stats, defaultOptions_);
    const auto nestedAlpEstimate = Est::estimateSize(
        EncodingType::RLE, physicalValues, stats, nestedAlpOptions);

    ASSERT_TRUE(defaultEstimate.has_value());
    ASSERT_TRUE(nestedAlpEstimate.has_value());
    EXPECT_LT(nestedAlpEstimate.value(), defaultEstimate.value());
  };

  verify(float{});
  verify(double{});
}

TEST_F(EncodingSizeEstimationTest, nestedAlpSizeEligibility) {
  const std::vector<float> logicalValues = {1.25F, 2.5F, 3.75F};
  const auto physicalValues =
      EncodingPhysicalType<float>::asEncodingPhysicalTypeSpan(logicalValues);

  EXPECT_FALSE(
      detail::nestedAlpSize<float>(physicalValues, defaultOptions_)
          .has_value());

  auto enabledOptions = defaultOptions_;
  enabledOptions.allowNestedAlpSelection = true;
  EXPECT_EQ(
      detail::nestedAlpSize<float>(physicalValues, enabledOptions),
      ALPEncoding<float>::estimateSize(physicalValues, enabledOptions));
}

TEST_F(EncodingSizeEstimationTest, uniqueValuesNestedAlpSizeEligibility) {
  const std::vector<float> logicalValues = {1.25F, 1.25F, 2.5F, 3.75F};
  const auto physicalValues =
      EncodingPhysicalType<float>::asEncodingPhysicalTypeSpan(logicalValues);
  const auto statistics = Statistics<uint32_t>::create(physicalValues);
  const auto& uniqueCounts = statistics.uniqueCounts().value();

  EXPECT_FALSE(
      detail::uniqueValuesNestedAlpSize<float>(uniqueCounts, defaultOptions_)
          .has_value());

  auto enabledOptions = defaultOptions_;
  enabledOptions.allowNestedAlpSelection = true;
  EXPECT_TRUE(
      detail::uniqueValuesNestedAlpSize<float>(uniqueCounts, enabledOptions)
          .has_value());

  const auto emptyStatistics =
      Statistics<uint32_t>::create(std::span<const uint32_t>{});
  EXPECT_FALSE(
      detail::uniqueValuesNestedAlpSize<float>(
          emptyStatistics.uniqueCounts().value(), enabledOptions)
          .has_value());
}

TEST_F(EncodingSizeEstimationTest, numericDictionary) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
  auto stats = Statistics<uint32_t>::create(data);

  auto size =
      Est::estimateSize(EncodingType::Dictionary, 10, stats, defaultOptions_);
  ASSERT_TRUE(size.has_value());
  EXPECT_GT(size.value(), 0);
}

TEST_F(EncodingSizeEstimationTest, numericMainlyConstant) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data(100, 42);
  data[50] = 99;
  auto stats = Statistics<uint32_t>::create(data);

  auto size = Est::estimateSize(
      EncodingType::MainlyConstant, 100, stats, defaultOptions_);
  ASSERT_TRUE(size.has_value());
  EXPECT_LT(size.value(), 100 * sizeof(uint32_t));
}

TEST_F(EncodingSizeEstimationTest, numericRle) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data;
  for (int i = 0; i < 50; ++i) {
    data.push_back(1);
  }
  for (int i = 0; i < 50; ++i) {
    data.push_back(2);
  }
  auto stats = Statistics<uint32_t>::create(data);

  auto size = Est::estimateSize(EncodingType::RLE, 100, stats, defaultOptions_);
  ASSERT_TRUE(size.has_value());
  EXPECT_LT(size.value(), 100 * sizeof(uint32_t));
}

TEST_F(EncodingSizeEstimationTest, numericVarint32) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data = {0, 1, 127, 128, 255, 256, 1000};
  auto stats = Statistics<uint32_t>::create(data);

  auto size =
      Est::estimateSize(EncodingType::Varint, 7, stats, defaultOptions_);
  ASSERT_TRUE(size.has_value());
  EXPECT_GT(size.value(), 0);
}

TEST_F(EncodingSizeEstimationTest, numericVarint64) {
  using Est = detail::EncodingSizeEstimation<uint64_t>;

  std::vector<uint64_t> data = {0, 1, 127, 128, 16384, 1000000};
  auto stats = Statistics<uint64_t>::create(data);

  auto size =
      Est::estimateSize(EncodingType::Varint, 6, stats, defaultOptions_);
  ASSERT_TRUE(size.has_value());
  EXPECT_GT(size.value(), 0);
}

TEST_F(EncodingSizeEstimationTest, numericUnsupportedEncoding) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data = {1, 2, 3};
  auto stats = Statistics<uint32_t>::create(data);

  auto size =
      Est::estimateSize(EncodingType::SparseBool, 3, stats, defaultOptions_);
  EXPECT_FALSE(size.has_value());
}

TEST_F(EncodingSizeEstimationTest, numericEstimateSizeDispatches) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data = {1, 2, 3, 4, 5};
  auto stats = Statistics<uint32_t>::create(data);

  auto size = Est::estimateSize(
      EncodingType::Trivial, data.size(), stats, defaultOptions_);
  ASSERT_TRUE(size.has_value());

  auto sizeFromValues = Est::estimateSize(
      EncodingType::Trivial,
      std::span<const uint32_t>{data},
      stats,
      defaultOptions_);
  EXPECT_EQ(sizeFromValues, size);
}

TEST_F(EncodingSizeEstimationTest, alpEstimateRequiresValuesAndMatchesActual) {
  verifyAlpEstimateVsActualSize<float>();
  verifyAlpEstimateVsActualSize<double>();
}

TEST_F(EncodingSizeEstimationTest, fixedBitWidthEstimateUsesValueRange) {
  Encoding::Options options;
  options.fixedBitWidthUseExactBits = true;
  const auto sizeFromZero = FixedBitWidthEncoding<uint32_t>::estimateSize(
      /*rowCount=*/100,
      /*minValue=*/0,
      /*maxValue=*/4,
      options);
  const auto sizeWithBaseline = FixedBitWidthEncoding<uint32_t>::estimateSize(
      /*rowCount=*/100,
      /*minValue=*/100,
      /*maxValue=*/104,
      options);
  EXPECT_EQ(sizeFromZero, sizeWithBaseline);

  const auto exactBitWidthSize = FixedBitWidthEncoding<uint32_t>::estimateSize(
      /*rowCount=*/100,
      /*minValue=*/0,
      /*maxValue=*/5,
      options);
  EXPECT_EQ(
      exactBitWidthSize,
      fixedBitWidthEstimate<uint32_t>(
          /*rowCount=*/100, /*minValue=*/0, /*maxValue=*/5, false));
  EXPECT_LT(
      exactBitWidthSize,
      fixedBitWidthEstimate<uint32_t>(
          /*rowCount=*/100, /*minValue=*/0, /*maxValue=*/5, true));
}

TEST_F(EncodingSizeEstimationTest, fixedBitWidthEstimateOptionCanRoundToByte) {
  std::vector<uint32_t> data(100);
  for (uint32_t i = 0; i < data.size(); ++i) {
    data[i] = i % 8;
  }
  const auto stats = Statistics<uint32_t>::create(data);
  Encoding::Options exactOptions;
  exactOptions.fixedBitWidthUseExactBits = true;
  Encoding::Options roundedOptions;
  roundedOptions.fixedBitWidthUseExactBits = false;

  const auto exactSize = FixedBitWidthEncoding<uint32_t>::estimateSize(
      data.size(), stats, exactOptions);
  const auto roundedSize = FixedBitWidthEncoding<uint32_t>::estimateSize(
      data.size(), stats, roundedOptions);

  EXPECT_EQ(
      exactSize,
      fixedBitWidthEstimate<uint32_t>(
          data.size(), stats.min(), stats.max(), false));
  EXPECT_EQ(
      roundedSize,
      fixedBitWidthEstimate<uint32_t>(
          data.size(), stats.min(), stats.max(), true));
  EXPECT_LT(exactSize, roundedSize);
}

TEST_F(
    EncodingSizeEstimationTest,
    fixedBitWidthEstimateOptionPropagatesToNestedChildren) {
  const auto exactOptions = exactBitWidthOptions();
  const auto roundedOptions = byteRoundedBitWidthOptions();

  std::vector<uint32_t> dictionaryData(100);
  for (uint32_t i = 0; i < dictionaryData.size(); ++i) {
    dictionaryData[i] = i % 8;
  }
  const auto dictionaryStats = Statistics<uint32_t>::create(dictionaryData);

  std::vector<uint32_t> rleData;
  for (uint32_t run = 0; run < 100; ++run) {
    rleData.insert(rleData.end(), run % 8 + 1, run % 2);
  }
  const auto rleStats = Statistics<uint32_t>::create(rleData);

  std::vector<uint32_t> mainlyConstantData(80, 42);
  for (uint32_t i = 0; i < 50; ++i) {
    mainlyConstantData.push_back(i % 8);
  }
  const auto mainlyConstantStats =
      Statistics<uint32_t>::create(mainlyConstantData);

  std::array<bool, 100> sparseBoolData;
  sparseBoolData.fill(true);
  for (size_t i = 0; i < sparseBoolData.size(); i += 10) {
    sparseBoolData[i] = false;
  }
  const auto sparseBoolStats = Statistics<bool>::create(sparseBoolData);

  std::vector<std::string> strings;
  std::vector<std::string_view> stringViews;
  strings.reserve(100);
  stringViews.reserve(100);
  for (uint32_t i = 0; i < 100; ++i) {
    strings.emplace_back(i % 8 + 1, static_cast<char>('a' + (i % 8)));
    stringViews.push_back(strings.back());
  }
  const auto stringStats = Statistics<std::string_view>::create(stringViews);

  struct TestCase {
    std::string name;
    uint64_t exactSize;
    uint64_t roundedSize;
  };

  for (const auto& testCase : std::vector<TestCase>{
           {
               .name = "Dictionary indices",
               .exactSize = DictionaryEncoding<uint32_t>::estimateSize(
                   dictionaryData.size(), dictionaryStats, exactOptions),
               .roundedSize = DictionaryEncoding<uint32_t>::estimateSize(
                   dictionaryData.size(), dictionaryStats, roundedOptions),
           },
           {
               .name = "RLE run lengths and values",
               .exactSize = RLEEncoding<uint32_t>::estimateSize(
                   rleData.size(), rleStats, exactOptions),
               .roundedSize = RLEEncoding<uint32_t>::estimateSize(
                   rleData.size(), rleStats, roundedOptions),
           },
           {
               .name = "MainlyConstant uncommon values and sparse bitmap",
               .exactSize = MainlyConstantEncoding<uint32_t>::estimateSize(
                   mainlyConstantData.size(),
                   mainlyConstantStats,
                   exactOptions),
               .roundedSize = MainlyConstantEncoding<uint32_t>::estimateSize(
                   mainlyConstantData.size(),
                   mainlyConstantStats,
                   roundedOptions),
           },
           {
               .name = "SparseBool indices",
               .exactSize = SparseBoolEncoding::estimateSize(
                   sparseBoolData.size(), sparseBoolStats, exactOptions),
               .roundedSize = SparseBoolEncoding::estimateSize(
                   sparseBoolData.size(), sparseBoolStats, roundedOptions),
           },
           {
               .name = "Trivial string lengths",
               .exactSize = TrivialEncoding<std::string_view>::estimateSize(
                   stringViews.size(), stringStats, exactOptions),
               .roundedSize = TrivialEncoding<std::string_view>::estimateSize(
                   stringViews.size(), stringStats, roundedOptions),
           },
           {
               .name = "Dictionary string indices and alphabet lengths",
               .exactSize = DictionaryEncoding<std::string_view>::estimateSize(
                   stringViews.size(), stringStats, exactOptions),
               .roundedSize =
                   DictionaryEncoding<std::string_view>::estimateSize(
                       stringViews.size(), stringStats, roundedOptions),
           },
           {
               .name = "FSST compressed lengths",
               .exactSize = FsstEncoding::estimateSize(
                   stringViews.size(), stringStats, exactOptions),
               .roundedSize = FsstEncoding::estimateSize(
                   stringViews.size(), stringStats, roundedOptions),
           },
       }) {
    SCOPED_TRACE(testCase.name);
    EXPECT_LT(testCase.exactSize, testCase.roundedSize);
  }
}

TEST_F(EncodingSizeEstimationTest, encodingEstimateHelpersReturnSizes) {
  const std::vector<uint32_t> numericData = {1, 1, 1, 2, 3, 3};
  auto numericStats = Statistics<uint32_t>::create(numericData);
  const std::array<bool, 4> boolData = {true, true, false, true};
  auto boolStats = Statistics<bool>::create(boolData);

  EXPECT_GT(TrivialEncoding<uint32_t>::estimateSize(numericData.size()), 0u);
  const std::vector<uint32_t> constantData = {1, 1};
  EXPECT_GT(
      ConstantEncoding<uint32_t>::estimateSize(
          std::span<const uint32_t>{constantData.data(), constantData.size()},
          Statistics<uint32_t>::create(constantData),
          Encoding::Options{})
          .value(),
      0u);
  EXPECT_GT(
      FixedBitWidthEncoding<uint32_t>::estimateSize(
          numericData.size(), numericStats, Encoding::Options{}),
      0u);
  EXPECT_GT(
      DictionaryEncoding<uint32_t>::estimateSize(
          numericData.size(), numericStats),
      0u);
  EXPECT_GT(
      RLEEncoding<uint32_t>::estimateSize(numericData.size(), numericStats),
      0u);
  EXPECT_GT(VarintEncoding<uint32_t>::estimateSize(numericStats), 0u);
  EXPECT_GT(
      BlockBitPackingEncoding<uint32_t>::estimateSize(
          std::span<const uint32_t>{numericData.data(), numericData.size()}),
      0u);
  EXPECT_GT(
      MainlyConstantEncoding<uint32_t>::estimateSize(
          numericData.size(), numericStats),
      0u);
  EXPECT_GT(SparseBoolEncoding::estimateSize(boolData.size(), boolStats), 0u);
}

TEST_F(
    EncodingSizeEstimationTest,
    mainlyConstantStringEstimateUsesUncommonLengthRange) {
  const std::string common(100, 'x');
  const std::vector<std::string_view> data = {
      common,
      common,
      common,
      common,
      common,
      "aa",
      "bbb",
  };
  auto stats = Statistics<std::string_view>::create(data);

  const uint64_t expectedSize = EncodingPrefix::kFixedPrefixSize +
      2 * sizeof(uint32_t) + common.size() + sizeof(uint32_t) +
      TrivialEncoding<std::string_view>::estimateSize(
                                    /*rowCount=*/2,
                                    /*totalStringsLength=*/5,
                                    /*minLength=*/2,
                                    /*maxLength=*/3) +
      SparseBoolEncoding::estimateSize(data.size(), /*exceptionCount=*/2);

  EXPECT_EQ(
      MainlyConstantEncoding<std::string_view>::estimateSize(
          data.size(), stats),
      expectedSize);
}

TEST_F(EncodingSizeEstimationTest, sparseBoolEstimateIncludesSentinelIndex) {
  Encoding::Options options;
  options.fixedBitWidthUseExactBits = false;
  const uint64_t indicesSize = FixedBitWidthEncoding<uint32_t>::estimateSize(
      /*rowCount=*/3,
      /*minValue=*/0,
      /*maxValue=*/10,
      options);
  const uint64_t expectedSize =
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t) + indicesSize;

  EXPECT_EQ(
      SparseBoolEncoding::estimateSize(
          /*rowCount=*/10, /*exceptionCount=*/2),
      expectedSize);
}

TEST_F(EncodingSizeEstimationTest, trivialSmallerThanDictionaryForUnique) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data;
  for (uint32_t i = 0; i < 100; ++i) {
    data.push_back(i * 43'000'000u);
  }
  auto stats = Statistics<uint32_t>::create(data);

  auto trivialSize =
      Est::estimateSize(EncodingType::Trivial, 100, stats, defaultOptions_);
  auto dictSize =
      Est::estimateSize(EncodingType::Dictionary, 100, stats, defaultOptions_);

  ASSERT_TRUE(trivialSize.has_value());
  ASSERT_TRUE(dictSize.has_value());
  EXPECT_LT(trivialSize.value(), dictSize.value());
}

TEST_F(EncodingSizeEstimationTest, dictionarySmallerForHighRepetition) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data;
  for (int i = 0; i < 1000; ++i) {
    data.push_back(i % 3);
  }
  auto stats = Statistics<uint32_t>::create(data);

  auto trivialSize =
      Est::estimateSize(EncodingType::Trivial, 1000, stats, defaultOptions_);
  auto dictSize =
      Est::estimateSize(EncodingType::Dictionary, 1000, stats, defaultOptions_);

  ASSERT_TRUE(trivialSize.has_value());
  ASSERT_TRUE(dictSize.has_value());
  EXPECT_LT(dictSize.value(), trivialSize.value());
}

TEST_F(EncodingSizeEstimationTest, constantSmallestForConstantData) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data(1000, 42);
  auto stats = Statistics<uint32_t>::create(data);

  auto constantSize =
      Est::estimateSize(EncodingType::Constant, data, stats, defaultOptions_);
  auto trivialSize =
      Est::estimateSize(EncodingType::Trivial, 1000, stats, defaultOptions_);

  ASSERT_TRUE(constantSize.has_value());
  ASSERT_TRUE(trivialSize.has_value());
  EXPECT_LT(constantSize.value(), trivialSize.value());
}

TEST_F(EncodingSizeEstimationTest, rleSmallForLongRuns) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data;
  for (int i = 0; i < 500; ++i) {
    data.push_back(1);
  }
  for (int i = 0; i < 500; ++i) {
    data.push_back(2);
  }
  auto stats = Statistics<uint32_t>::create(data);

  auto rleSize =
      Est::estimateSize(EncodingType::RLE, 1000, stats, defaultOptions_);
  auto trivialSize =
      Est::estimateSize(EncodingType::Trivial, 1000, stats, defaultOptions_);

  ASSERT_TRUE(rleSize.has_value());
  ASSERT_TRUE(trivialSize.has_value());
  EXPECT_LT(rleSize.value(), trivialSize.value());
}

TEST_F(EncodingSizeEstimationTest, fbwSmallForNarrowRange) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data;
  for (uint32_t i = 0; i < 1000; ++i) {
    data.push_back(1000 + (i % 4));
  }
  auto stats = Statistics<uint32_t>::create(data);

  auto fbwSize = Est::estimateSize(
      EncodingType::FixedBitWidth, 1000, stats, defaultOptions_);
  auto trivialSize =
      Est::estimateSize(EncodingType::Trivial, 1000, stats, defaultOptions_);

  ASSERT_TRUE(fbwSize.has_value());
  ASSERT_TRUE(trivialSize.has_value());
  EXPECT_LT(fbwSize.value(), trivialSize.value());
}

TEST_F(EncodingSizeEstimationTest, uint64AllEncodings) {
  using Est = detail::EncodingSizeEstimation<uint64_t>;

  std::vector<uint64_t> data;
  for (uint64_t i = 0; i < 100; ++i) {
    data.push_back(i);
  }
  auto stats = Statistics<uint64_t>::create(data);

  ASSERT_TRUE(
      Est::estimateSize(EncodingType::Trivial, 100, stats, defaultOptions_)
          .has_value());
  ASSERT_TRUE(
      Est::estimateSize(
          EncodingType::FixedBitWidth, 100, stats, defaultOptions_)
          .has_value());
  ASSERT_TRUE(
      Est::estimateSize(EncodingType::Dictionary, 100, stats, defaultOptions_)
          .has_value());
  ASSERT_TRUE(
      Est::estimateSize(EncodingType::RLE, 100, stats, defaultOptions_)
          .has_value());
  ASSERT_TRUE(
      Est::estimateSize(EncodingType::Varint, 100, stats, defaultOptions_)
          .has_value());
}

TEST_F(EncodingSizeEstimationTest, pforEstimateVsActualSize) {
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  // ~90% values in [0, 100], ~10% outliers near max — Pfor's sweet spot.
  std::vector<uint32_t> data;
  for (uint32_t i = 0; i < 900; ++i) {
    data.push_back(i % 100);
  }
  for (uint32_t i = 0; i < 100; ++i) {
    data.push_back(1'000'000 + i);
  }
  auto stats = Statistics<uint32_t>::create(data);
  const uint32_t numValues = static_cast<uint32_t>(data.size());

  auto estimated =
      Est::estimateSize(EncodingType::PFOR, numValues, stats, defaultOptions_);
  ASSERT_TRUE(estimated.has_value());

  // Actually encode and measure real size.
  nimble::Buffer buffer{*pool_};
  auto encoded = nimble::test::Encoder<nimble::PFOREncoding<uint32_t>>::encode(
      buffer, nimble::Vector<uint32_t>(pool_.get(), data.begin(), data.end()));
  const uint64_t actualSize = encoded.size();

  // Estimate should be in the same ballpark as actual — within 2x.
  EXPECT_GT(estimated.value(), actualSize / 2)
      << "estimate too low: " << estimated.value() << " vs actual "
      << actualSize;
  EXPECT_LT(estimated.value(), actualSize * 2)
      << "estimate too high: " << estimated.value() << " vs actual "
      << actualSize;
}

TEST_F(EncodingSizeEstimationTest, simdForBitpackEstimateVsActualSize) {
  verifySimdForBitpackEstimateVsActualSize<uint8_t>();
  verifySimdForBitpackEstimateVsActualSize<uint16_t>();
  verifySimdForBitpackEstimateVsActualSize<uint32_t>();
  verifySimdForBitpackEstimateVsActualSize<uint64_t>();
}

TEST_F(EncodingSizeEstimationTest, blockBitPackingEstimateVsActualSize) {
  verifyBlockBitPackingEstimateVsActualSize<uint32_t>();
  verifyBlockBitPackingEstimateVsActualSize<uint64_t>();
}

TEST_F(
    EncodingSizeEstimationTest,
    blockBitPackingStatisticsOverloadMatchesSpanOverload) {
  std::vector<uint32_t> data;
  data.reserve(2050);
  for (uint32_t i = 0; i < 1024; ++i) {
    data.push_back(1000 + i % 4);
  }
  for (uint32_t i = 0; i < 1024; ++i) {
    data.push_back(1'000'000 + i % 8);
  }
  data.push_back(std::numeric_limits<uint32_t>::min());
  data.push_back(std::numeric_limits<uint32_t>::max());

  const auto spanEstimate = BlockBitPackingEncoding<uint32_t>::estimateSize(
      std::span<const uint32_t>{data.data(), data.size()});
  auto stats = Statistics<uint32_t>::create(data);
  const auto statsEstimate =
      BlockBitPackingEncoding<uint32_t>::estimateSize(stats);

  ASSERT_TRUE(spanEstimate.has_value());
  ASSERT_TRUE(statsEstimate.has_value());
  EXPECT_EQ(spanEstimate.value(), statsEstimate.value());
}

TEST_F(
    EncodingSizeEstimationTest,
    blockBitPackingSkipEncodingForFullRangeBlock) {
  std::vector<uint32_t> data;
  data.reserve(2048);
  // Block 0: narrow range (should pack)
  for (uint32_t i = 0; i < 1024; ++i) {
    data.push_back(i % 4);
  }
  // Block 1: full 32-bit range (should skip encoding)
  for (uint32_t i = 0; i < 1024; ++i) {
    data.push_back(i == 0 ? 0 : std::numeric_limits<uint32_t>::max());
  }

  auto stats = Statistics<uint32_t>::create(data);
  const auto estimate = BlockBitPackingEncoding<uint32_t>::estimateSize(stats);
  ASSERT_TRUE(estimate.has_value());

  // Block 1 should fall back to raw: 1024 * 4 = 4096 bytes.
  // Block 0 packs at 2 bits: much smaller.
  // Total should be less than 2 * 1024 * 4 (if both were raw).
  const auto bothRaw = 2 * 1024 * sizeof(uint32_t);
  EXPECT_LT(estimate.value(), bothRaw + 200);
  EXPECT_GT(estimate.value(), 4096);
}

// TODO: Add perBlockTighteningReducesMainlyConstantEstimate and
// perBlockTighteningReducesDictionaryEstimate tests when per-block
// tightening is wired up for MainlyConstant and Dictionary encodings.

TEST_F(EncodingSizeEstimationTest, customBlockSizeAffectsEstimate) {
  // With block size 512, we get 4 blocks instead of 2 (for 2048 values).
  // Each smaller block has a tighter range → different estimate.
  std::vector<uint32_t> data;
  data.reserve(2048);
  for (uint32_t i = 0; i < 2048; ++i) {
    data.push_back(i);
  }

  auto stats = Statistics<uint32_t>::create(data);

  const auto defaultBlockEstimate =
      BlockBitPackingEncoding<uint32_t>::estimateSize(stats);
  const auto smallBlockEstimate =
      BlockBitPackingEncoding<uint32_t>::estimateSize(stats, /*blockSize=*/512);
  ASSERT_TRUE(defaultBlockEstimate.has_value());
  ASSERT_TRUE(smallBlockEstimate.has_value());

  // Smaller blocks → tighter per-block ranges → smaller packed data.
  // But more blocks → more metadata overhead.
  // The estimates should differ.
  EXPECT_NE(defaultBlockEstimate.value(), smallBlockEstimate.value());

  // With sequential data [0..2047], smaller blocks have tighter ranges.
  // Block size 512: 4 blocks, each spanning 512 values (9 bits).
  // Block size 1024: 2 blocks, each spanning 1024 values (10 bits).
  // Smaller blocks should pack tighter despite more metadata.
  EXPECT_LT(smallBlockEstimate.value(), defaultBlockEstimate.value());

  // Calling with the original block size again should return the same result
  // (verifies cache invalidation round-trips correctly).
  const auto defaultBlockEstimateAgain =
      BlockBitPackingEncoding<uint32_t>::estimateSize(stats);
  ASSERT_TRUE(defaultBlockEstimateAgain.has_value());
  EXPECT_EQ(defaultBlockEstimate.value(), defaultBlockEstimateAgain.value());
}

TEST_F(
    EncodingSizeEstimationTest,
    blockBitPackingStatisticsOverloadMultipleTypes) {
  // Verify Statistics-based and span-based estimateSize match for all unsigned
  // physical types, not just uint32_t.
  auto verifyMatch = [](auto typeTag) {
    using T = decltype(typeTag);
    std::vector<T> data;
    data.reserve(2048);
    for (uint32_t i = 0; i < 2048; ++i) {
      data.push_back(static_cast<T>(i % 200));
    }

    const auto spanEstimate = BlockBitPackingEncoding<T>::estimateSize(
        std::span<const T>{data.data(), data.size()});
    auto stats = Statistics<T>::create(data);
    const auto statsEstimate = BlockBitPackingEncoding<T>::estimateSize(stats);

    ASSERT_TRUE(spanEstimate.has_value());
    ASSERT_TRUE(statsEstimate.has_value());
    EXPECT_EQ(spanEstimate.value(), statsEstimate.value())
        << "Mismatch for type size " << sizeof(T);
  };

  verifyMatch(uint8_t{});
  verifyMatch(uint16_t{});
  verifyMatch(uint32_t{});
  verifyMatch(uint64_t{});
}

TEST_F(EncodingSizeEstimationTest, blockBitPackingConstantData) {
  // All values identical: every block has range=0, bw=0, packedSize=0.
  // Estimate should be dominated by metadata.
  std::vector<uint32_t> data(2048, 42);

  auto stats = Statistics<uint32_t>::create(data);
  const auto estimate = BlockBitPackingEncoding<uint32_t>::estimateSize(stats);
  ASSERT_TRUE(estimate.has_value());

  const auto rawSize = 2048 * sizeof(uint32_t);
  EXPECT_LT(estimate.value(), rawSize / 2);
  EXPECT_GT(estimate.value(), 0);
}

TEST_F(EncodingSizeEstimationTest, blockBitPackingPartialLastBlock) {
  // 1500 values: 1 full block (1024) + 1 partial block (476).
  // Verifies BlockStatsAccumulator correctly flushes the remainder.
  std::vector<uint32_t> data;
  data.reserve(1500);
  for (uint32_t i = 0; i < 1500; ++i) {
    data.push_back(i % 16);
  }

  auto stats = Statistics<uint32_t>::create(data);
  const auto statsEstimate =
      BlockBitPackingEncoding<uint32_t>::estimateSize(stats);
  const auto spanEstimate = BlockBitPackingEncoding<uint32_t>::estimateSize(
      std::span<const uint32_t>{data.data(), data.size()});

  ASSERT_TRUE(statsEstimate.has_value());
  ASSERT_TRUE(spanEstimate.has_value());
  EXPECT_EQ(spanEstimate.value(), statsEstimate.value());
  EXPECT_GT(statsEstimate.value(), 0);
}

TEST_F(EncodingSizeEstimationTest, blockBitPackingSingleValue) {
  // Edge case: 1 value.
  std::vector<uint32_t> data = {12345};

  auto stats = Statistics<uint32_t>::create(data);
  const auto statsEstimate =
      BlockBitPackingEncoding<uint32_t>::estimateSize(stats);
  const auto spanEstimate = BlockBitPackingEncoding<uint32_t>::estimateSize(
      std::span<const uint32_t>{data.data(), data.size()});

  ASSERT_TRUE(statsEstimate.has_value());
  ASSERT_TRUE(spanEstimate.has_value());
  EXPECT_EQ(spanEstimate.value(), statsEstimate.value());
}

TEST_F(
    EncodingSizeEstimationTest,
    blockBitPackingEstimateViaEncodingSizeEstimation) {
  // Verify BlockBitPacking is wired up in EncodingSizeEstimation dispatcher.
  using Est = detail::EncodingSizeEstimation<uint32_t>;

  std::vector<uint32_t> data;
  data.reserve(2048);
  for (uint32_t i = 0; i < 2048; ++i) {
    data.push_back(i % 100);
  }
  auto stats = Statistics<uint32_t>::create(data);

  auto estimate = Est::estimateSize(
      EncodingType::BlockBitPacking, 2048, stats, defaultOptions_);
  ASSERT_TRUE(estimate.has_value());
  EXPECT_GT(estimate.value(), 0);

  // Verify it matches the direct call with default block size.
  const auto directEstimate =
      BlockBitPackingEncoding<uint32_t>::estimateSize(stats);
  ASSERT_TRUE(directEstimate.has_value());
  EXPECT_EQ(estimate.value(), directEstimate.value());

  // Verify options.blockBitPackingBlockSize is threaded through.
  Encoding::Options options;
  options.blockBitPackingBlockSize = 512;
  auto estimateSmallBlock =
      Est::estimateSize(EncodingType::BlockBitPacking, 2048, stats, options);
  ASSERT_TRUE(estimateSmallBlock.has_value());
  EXPECT_NE(estimateSmallBlock.value(), estimate.value());
}

TEST_F(
    EncodingSizeEstimationTest,
    blockBitPackingEstimateSkippedWhenBlockCountExceedsIndexLimit) {
  // BlockBitPacking indexes its blocks with a uint16_t, so encode() rejects a
  // stream with more than kMaxBlockCount blocks. estimateSize() must return
  // nullopt in that case so encoding selection skips it rather than picking it
  // and letting encode() hard-fail. blockSize = 1 makes numBlocks == rowCount,
  // so the boundary is reachable without a huge stream.
  constexpr uint16_t kBlockSize = 1;
  constexpr uint32_t kMaxBlocks =
      BlockBitPackingEncoding<uint32_t>::kMaxBlockCount;

  auto makeData = [](uint32_t rowCount) {
    std::vector<uint32_t> data(rowCount);
    for (uint32_t i = 0; i < rowCount; ++i) {
      data[i] = i % 100;
    }
    return data;
  };

  // At the limit: exactly kMaxBlocks blocks still fit the uint16_t index.
  const auto atLimit = makeData(kMaxBlocks);
  const auto atLimitStats = Statistics<uint32_t>::create(atLimit);
  EXPECT_TRUE(
      BlockBitPackingEncoding<uint32_t>::estimateSize(
          std::span<const uint32_t>{atLimit.data(), atLimit.size()}, kBlockSize)
          .has_value());
  EXPECT_TRUE(
      BlockBitPackingEncoding<uint32_t>::estimateSize(atLimitStats, kBlockSize)
          .has_value());

  // One block past the limit overflows the index: both overloads and the
  // selection dispatcher must skip the encoding.
  const auto overLimit = makeData(kMaxBlocks + 1);
  const auto overLimitStats = Statistics<uint32_t>::create(overLimit);
  EXPECT_FALSE(
      BlockBitPackingEncoding<uint32_t>::estimateSize(
          std::span<const uint32_t>{overLimit.data(), overLimit.size()},
          kBlockSize)
          .has_value());
  EXPECT_FALSE(
      BlockBitPackingEncoding<uint32_t>::estimateSize(
          overLimitStats, kBlockSize)
          .has_value());

  Encoding::Options options;
  options.blockBitPackingBlockSize = kBlockSize;
  EXPECT_FALSE(
      detail::EncodingSizeEstimation<uint32_t>::estimateSize(
          EncodingType::BlockBitPacking,
          static_cast<size_t>(overLimit.size()),
          overLimitStats,
          options)
          .has_value());
}
