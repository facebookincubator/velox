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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <random>

#include "velox/dwio/nimble/encodings/ALPRDEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/NestedAlpSizeEstimation.h"
#include "velox/dwio/nimble/encodings/selection/tests/RandomEncodingSelectionPolicy.h"

namespace facebook::nimble {
namespace {

struct ALPRDSelectionTestAccess : ALPRDEncodingBase {
  using ALPRDEncodingBase::calculateSerializedSize;
  using ALPRDEncodingBase::selectAndDecompose;
};

using ReadFactors = std::vector<std::pair<EncodingType, float>>;

// Counts child policy creation during selection and writing.
template <typename T>
class CountingManualPolicy : public ManualEncodingSelectionPolicy<T> {
 public:
  explicit CountingManualPolicy(EncodingType encodingType)
      : ManualEncodingSelectionPolicy<T>{
            {{encodingType, 1}},
            std::nullopt,
            std::nullopt} {}

  uint32_t numNestedPolicies{0};

 protected:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier identifier,
      DataType dataType) override {
    ++numNestedPolicies;
    return ManualEncodingSelectionPolicy<T>::createImpl(
        parentEncodingType, identifier, dataType);
  }
};

// Supplies custom size and cost through the normal child selection loop.
template <typename T>
class CustomCandidatePolicy : public ManualEncodingSelectionPolicy<T> {
 public:
  using physicalType = typename TypeTraits<T>::physicalType;

  CustomCandidatePolicy(
      uint64_t size,
      std::optional<double> cost,
      uint32_t& numEstimates,
      float readFactor = 1)
      : ManualEncodingSelectionPolicy<
            T>{{{EncodingType::Trivial, readFactor}}, std::nullopt, std::nullopt},
        size_{size},
        cost_{cost},
        numEstimates_{numEstimates} {}

 protected:
  std::optional<EncodingSelectionResult> estimateCandidateSize(
      EncodingType encodingType,
      std::span<const physicalType>,
      const Statistics<physicalType>&,
      size_t,
      const Encoding::Options&) override {
    ++numEstimates_;
    return EncodingSelectionResult{
        .encodingType = encodingType,
        .estimatedSize = size_,
        .estimatedCost = cost_,
    };
  }

 private:
  const uint64_t size_;
  const std::optional<double> cost_;
  uint32_t& numEstimates_;
};

template <typename T>
class CustomChildrenPolicy : public ManualEncodingSelectionPolicy<T> {
 public:
  CustomChildrenPolicy(uint64_t childSize, double childCost)
      : ManualEncodingSelectionPolicy<
            T>{{{EncodingType::ALPRD, 1}, {EncodingType::Trivial, 1}}, std::nullopt, std::nullopt},
        childSize_{childSize},
        childCost_{childCost} {}

  uint32_t numChildEstimates{0};

 protected:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType,
      NestedEncodingIdentifier identifier,
      DataType dataType) override {
    UNIQUE_PTR_FACTORY(
        dataType,
        CustomCandidatePolicy,
        childSize_ + identifier,
        childCost_ + identifier,
        numChildEstimates);
  }

 private:
  const uint64_t childSize_;
  const double childCost_;
};

ReadFactors candidates() {
  // Equal weights test size selection independently of deployment tuning.
  return {
      {EncodingType::Constant, 1},
      {EncodingType::Trivial, 1},
      {EncodingType::FixedBitWidth, 1},
      {EncodingType::MainlyConstant, 1},
      {EncodingType::Dictionary, 1},
      {EncodingType::RLE, 1},
      {EncodingType::ALP, 1},
      {EncodingType::ALPRD, 1},
  };
}

template <typename T, bool UseVarint>
struct SelectionConfig {
  using Value = T;
  static constexpr bool useVarint = UseVarint;
};

template <typename Config>
class ALPRDSelectionTest : public ::testing::Test {
 protected:
  using T = typename Config::Value;
  using PhysicalType = typename TypeTraits<T>::physicalType;

  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<Buffer>(*pool_);
  }

  std::unique_ptr<EncodingSelectionPolicy<T>> policy(
      ReadFactors factors,
      std::optional<ReadFactors> nested) {
    return std::make_unique<ManualEncodingSelectionPolicy<T>>(
        std::move(factors), std::nullopt, std::nullopt, std::move(nested));
  }

  std::vector<PhysicalType> makeValues(uint32_t numRows, uint32_t numPrefixes) {
    std::mt19937_64 random{73};
    constexpr auto kShift = sizeof(T) * 8 - 16;
    const auto mask = (PhysicalType{1} << kShift) - 1;
    std::vector<PhysicalType> values(numRows);
    for (auto& value : values) {
      // Widely separated high prefixes make ordinary FOR ineffective while
      // the low parts retain the full mantissa entropy.
      const PhysicalType high = 0x3f00 + 0x1000 * (random() % numPrefixes);
      value = (high << kShift) | (random() & mask);
    }
    return values;
  }

  EncodingSelectionResult select(
      std::span<const PhysicalType> values,
      ReadFactors factors,
      std::optional<ReadFactors> nested) {
    auto selectionPolicy = policy(std::move(factors), std::move(nested));
    return selectionPolicy->select(
        values, Statistics<PhysicalType>::create(values), options_);
  }

  std::string_view encode(
      std::span<const PhysicalType> values,
      std::unique_ptr<EncodingSelectionPolicy<T>> selectionPolicy) {
    ScopedVector<T> logical{values.size(), pool_.get(), options_.bufferPool};
    for (size_t i = 0; i < values.size(); ++i) {
      logical[i] = std::bit_cast<T>(values[i]);
    }
    return EncodingFactory::encode<T>(
        std::move(selectionPolicy), logical, *buffer_, options_);
  }

  void check(std::string_view encoded, std::span<const PhysicalType> values) {
    auto decoder = EncodingFactory(options_).create(*pool_, encoded, nullptr);
    ScopedVector<PhysicalType> actual{
        values.size(), pool_.get(), options_.bufferPool};
    std::fill(actual.begin(), actual.end(), 0);
    decoder->materialize(values.size(), actual.data());
    EXPECT_THAT(
        (std::span<const PhysicalType>{actual}),
        ::testing::ElementsAreArray(values));
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<Buffer> buffer_;
  Encoding::Options options_{.useVarintRowCount = Config::useVarint};
};

using Configs = ::testing::Types<
    SelectionConfig<float, false>,
    SelectionConfig<double, false>,
    SelectionConfig<float, true>,
    SelectionConfig<double, true>>;
TYPED_TEST_SUITE(ALPRDSelectionTest, Configs);

TYPED_TEST(ALPRDSelectionTest, emptyInputAndInvalidTargetCount) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  const std::span<const PhysicalType> empty;
  const auto emptyStatistics = Statistics<PhysicalType>::create(empty);
  CountingManualPolicy<T> policy{EncodingType::ALPRD};
  EXPECT_FALSE(ALPRDEncoding<T>::estimateSize(empty, this->options_));
  const auto selected = policy.select(empty, emptyStatistics, this->options_);
  EXPECT_EQ(selected.encodingType, EncodingType::Trivial);
  EXPECT_EQ(selected.estimatedSize, 0);
  EXPECT_EQ(selected.estimatedCost, 0);
  EXPECT_THROW(
      policy.select(empty, emptyStatistics, 1, this->options_),
      NimbleInternalError);
  const auto values = this->makeValues(2, 1);
  EXPECT_THROW(
      policy.select(
          values, Statistics<PhysicalType>::create(values), 1, this->options_),
      NimbleInternalError);
  EXPECT_EQ(policy.numNestedPolicies, 0);
}

TYPED_TEST(ALPRDSelectionTest, emptyCandidatesPreserveUncompressedFallback) {
  using PhysicalType = typename TestFixture::PhysicalType;
  const std::array<PhysicalType, 4> values{0, 1, 2, 3};
  const auto statistics = Statistics<PhysicalType>::create(values);
  for (const bool withCompression : {false, true}) {
    std::optional<CompressionOptions> compression;
    if (withCompression) {
      compression = CompressionOptions{};
    }
    ManualEncodingSelectionPolicy<PhysicalType> policy{
        {}, compression, std::nullopt};
    for (const uint32_t numValues : {4, 65'536}) {
      const auto result =
          policy.select(values, statistics, numValues, this->options_);
      ASSERT_TRUE(result.has_value());
      EXPECT_EQ(result->encodingType, EncodingType::Trivial);
      ASSERT_TRUE(result->estimatedSize.has_value());
      ASSERT_TRUE(result->estimatedCost.has_value());
      EXPECT_DOUBLE_EQ(*result->estimatedCost, *result->estimatedSize);
      auto compressionPolicy = result->compressionPolicyFactory();
      ASSERT_NE(compressionPolicy, nullptr);
      EXPECT_EQ(
          dynamic_cast<ConfiguredCompressionPolicy*>(compressionPolicy.get()),
          nullptr);
    }
  }
}

TEST(ALPRDSelectionProjectionTest, emptyStringCandidatesOnlyEstimateFullInput) {
  const std::array<std::string_view, 2> values{"a", "longer"};
  const auto statistics = Statistics<std::string_view>::create(values);
  ManualEncodingSelectionPolicy<std::string_view> policy{
      {}, std::nullopt, std::nullopt};
  const auto full = policy.select(values, statistics, Encoding::Options{});
  EXPECT_EQ(full.encodingType, EncodingType::Trivial);
  EXPECT_TRUE(full.estimatedSize.has_value());
  EXPECT_FALSE(policy.select(values, statistics, 65'536, Encoding::Options{}));
}

TEST(ALPRDSelectionProjectionTest, supportedProjectionAndMixedCandidates) {
  const std::array<uint32_t, 4> values{1, 1, 1, 1};
  const auto statistics = Statistics<uint32_t>::create(values);
  for (const auto encodingType :
       {EncodingType::Constant,
        EncodingType::Trivial,
        EncodingType::FixedBitWidth,
        EncodingType::SimdForBitpack}) {
    ManualEncodingSelectionPolicy<uint32_t> policy{
        {{EncodingType::RLE, 0.000001}, {encodingType, 2}},
        std::nullopt,
        std::nullopt};
    const auto selected =
        policy.select(values, statistics, 65'536, Encoding::Options{});
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(selected->encodingType, encodingType);
    ASSERT_TRUE(selected->estimatedSize.has_value());
    ASSERT_TRUE(selected->estimatedCost.has_value());
    EXPECT_DOUBLE_EQ(*selected->estimatedCost, 2 * *selected->estimatedSize);
  }
}

TEST(
    ALPRDSelectionProjectionTest,
    customCandidateEstimatesUseNormalSelectionLoop) {
  const std::array<uint32_t, 2> values{1, 2};
  const auto statistics = Statistics<uint32_t>::create(values);
  uint32_t numEstimates = 0;
  CustomCandidatePolicy<uint32_t> policy{17, 3, numEstimates, 7};
  for (const uint32_t numValues : {2, 65'536}) {
    const auto selected =
        policy.select(values, statistics, numValues, Encoding::Options{});
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(selected->estimatedSize, 17);
    EXPECT_EQ(selected->estimatedCost, 3);
  }
  EXPECT_EQ(numEstimates, 2);
  CustomCandidatePolicy<uint32_t> sizeOnly{17, std::nullopt, numEstimates, 2};
  for (const uint32_t numValues : {2, 65'536}) {
    const auto selected =
        sizeOnly.select(values, statistics, numValues, Encoding::Options{});
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(selected->estimatedSize, 17);
    EXPECT_EQ(selected->estimatedCost, 34);
  }
  EXPECT_EQ(numEstimates, 4);
}

TYPED_TEST(ALPRDSelectionTest, alprdEstimationScoresEachChild) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  for (const uint32_t numRows : {1, 512, 1'024, 1'025, 65'536}) {
    for (const bool withExceptions : {false, true}) {
      SCOPED_TRACE(numRows);
      SCOPED_TRACE(withExceptions);
      auto values = this->makeValues(numRows, 1);
      if (withExceptions) {
        values[0] = std::numeric_limits<PhysicalType>::max();
      }
      const auto children =
          ALPRDSelectionTestAccess::selectAndDecompose<PhysicalType>(
              values, this->options_);
      for (const auto encodingType : {EncodingType::ALP, EncodingType::ALPRD}) {
        CountingManualPolicy<T> policy{encodingType};
        const auto selected = policy.select(
            values, Statistics<PhysicalType>::create(values), this->options_);
        ASSERT_EQ(selected.encodingType, encodingType);
        ASSERT_TRUE(selected.estimatedSize);
        ASSERT_TRUE(selected.estimatedCost);
        EXPECT_EQ(
            policy.numNestedPolicies,
            encodingType == EncodingType::ALPRD
                ? (children.exceptionCount == 0 ? 2 : 4)
                : 0);
      }
    }
  }
}

TYPED_TEST(ALPRDSelectionTest, customChildEstimatesDetermineSizeAndRanking) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  for (const uint32_t numRows : {512, 65'536}) {
    SCOPED_TRACE(numRows);
    const auto values = this->makeValues(numRows, 2);
    const auto statistics = Statistics<PhysicalType>::create(values);
    const auto children =
        ALPRDSelectionTestAccess::selectAndDecompose<PhysicalType>(
            values, this->options_);
    const uint32_t numChildren = children.exceptionCount == 0 ? 2 : 4;
    const uint64_t positionsSize = children.exceptionCount == 0 ? 0 : 12;
    const uint64_t highsSize = children.exceptionCount == 0 ? 0 : 13;
    CustomChildrenPolicy<T> cheapChildren{10, 3};
    const auto selected =
        cheapChildren.select(values, statistics, this->options_);
    ASSERT_EQ(selected.encodingType, EncodingType::ALPRD);
    EXPECT_EQ(cheapChildren.numChildEstimates, numChildren);
    const auto expectedSize = ALPRDSelectionTestAccess::calculateSerializedSize(
        children.parameters,
        numRows,
        children.exceptionCount,
        {10, 11, positionsSize, highsSize},
        this->options_);
    EXPECT_EQ(selected.estimatedSize, expectedSize);
    EXPECT_EQ(selected.estimatedCost, expectedSize);
    CustomChildrenPolicy<T> estimatorPolicy{10, 3};
    const auto estimate =
        ALPRDEncoding<T>::estimateSize(estimatorPolicy, values, this->options_);
    ASSERT_TRUE(estimate.has_value());
    EXPECT_EQ(estimate, expectedSize);

    // Child costs choose child encodings; they are not added to the parent
    // score when the selected child sizes are unchanged.
    CustomChildrenPolicy<T> expensiveChildren{10, 1'000'000};
    const auto expensive =
        expensiveChildren.select(values, statistics, this->options_);
    EXPECT_EQ(expensiveChildren.numChildEstimates, numChildren);
    EXPECT_EQ(expensive.encodingType, EncodingType::ALPRD);
    EXPECT_EQ(expensive.estimatedSize, expectedSize);
    EXPECT_EQ(expensive.estimatedCost, expectedSize);

    CustomChildrenPolicy<T> largeChildren{1'000'000, 3};
    const auto rejected =
        largeChildren.select(values, statistics, this->options_);
    EXPECT_EQ(largeChildren.numChildEstimates, numChildren);
    EXPECT_EQ(rejected.encodingType, EncodingType::Trivial);
  }
}

TYPED_TEST(ALPRDSelectionTest, alprdScoringUsesBoundedSample) {
  using PhysicalType = typename TestFixture::PhysicalType;
  const auto values = this->makeValues(4 * ALPRDEncodingBase::kSampleSize, 2);
  const auto children =
      ALPRDSelectionTestAccess::selectAndDecompose<PhysicalType>(
          values, this->options_);
  EXPECT_EQ(children.rowCount, values.size());
  EXPECT_EQ(children.codes.size(), ALPRDEncodingBase::kSampleSize);
  EXPECT_EQ(children.rightParts.size(), ALPRDEncodingBase::kSampleSize);
  EXPECT_LE(children.exceptionPositions.size(), ALPRDEncodingBase::kSampleSize);
  EXPECT_LE(children.exceptionHighParts.size(), ALPRDEncodingBase::kSampleSize);
  EXPECT_EQ(
      children.exceptionCount,
      (uint64_t{children.exceptionPositions.size()} * values.size() +
       ALPRDEncodingBase::kSampleSize - 1) /
          ALPRDEncodingBase::kSampleSize);
}

TYPED_TEST(ALPRDSelectionTest, rootFactorWeightsSelectedChildSizes) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  const auto values = this->makeValues(512, 2);
  const auto statistics = Statistics<PhysicalType>::create(values);
  const auto score = [&](float rootFactor) {
    ManualEncodingSelectionPolicy<T> policy{
        {{EncodingType::ALPRD, rootFactor}},
        std::nullopt,
        std::nullopt,
        ReadFactors{{EncodingType::FixedBitWidth, 1}}};
    return policy.select(values, statistics, this->options_);
  };
  const auto factorOne = score(1);
  const auto factorTwo = score(2);
  ASSERT_EQ(factorOne.encodingType, EncodingType::ALPRD);
  ASSERT_EQ(factorTwo.encodingType, EncodingType::ALPRD);
  ASSERT_TRUE(factorOne.estimatedCost.has_value());
  ASSERT_TRUE(factorTwo.estimatedCost.has_value());
  EXPECT_EQ(factorTwo.estimatedSize, factorOne.estimatedSize);
  EXPECT_DOUBLE_EQ(
      factorTwo.estimatedCost.value(), 2 * factorOne.estimatedCost.value());
  EXPECT_DOUBLE_EQ(
      factorOne.estimatedCost.value(), factorOne.estimatedSize.value());
  ManualEncodingSelectionPolicy<T> expensiveRoot{
      {{EncodingType::ALPRD, 10'000}, {EncodingType::Trivial, 1}},
      std::nullopt,
      std::nullopt,
      ReadFactors{{EncodingType::FixedBitWidth, 1}}};
  EXPECT_EQ(
      expensiveRoot.select(values, statistics, this->options_).encodingType,
      EncodingType::Trivial);
}

TYPED_TEST(ALPRDSelectionTest, childReadFactorsChooseSizesForParent) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  for (const uint32_t numRows : {512, 65'536}) {
    SCOPED_TRACE(numRows);
    const auto values = this->makeValues(numRows, 2);
    const auto statistics = Statistics<PhysicalType>::create(values);
    const auto children =
        ALPRDSelectionTestAccess::selectAndDecompose<PhysicalType>(
            values, this->options_);
    if (numRows == 65'536) {
      ASSERT_GT(children.exceptionCount, 0);
    }
    std::array<uint32_t, 2> projectedPositions{0, numRows - 1};
    const std::span<const uint32_t> positions =
        children.exceptionPositions.size() < children.exceptionCount
        ? std::span<const uint32_t>{projectedPositions}
        : std::span<const uint32_t>{children.exceptionPositions};
    for (const auto preferred :
         {EncodingType::Trivial, EncodingType::FixedBitWidth}) {
      SCOPED_TRACE(preferred);
      ReadFactors nested{
          {EncodingType::Trivial,
           preferred == EncodingType::Trivial ? 1.0f : 10'000.0f},
          {EncodingType::FixedBitWidth,
           preferred == EncodingType::FixedBitWidth ? 1.0f : 10'000.0f}};
      ManualEncodingSelectionPolicy<T> policy{
          {{EncodingType::ALPRD, 2}}, std::nullopt, std::nullopt, nested};
      const auto childSize = [&]<typename ChildT>(
                                 std::span<const ChildT> sample,
                                 uint32_t numValues) {
        return detail::EncodingSizeEstimation<ChildT>::estimateSize(
                   preferred,
                   numValues,
                   Statistics<ChildT>::create(sample),
                   this->options_)
            .value();
      };
      const auto expectedSize =
          ALPRDSelectionTestAccess::calculateSerializedSize(
              children.parameters,
              numRows,
              children.exceptionCount,
              {childSize(std::span<const uint16_t>{children.codes}, numRows),
               childSize(
                   std::span<const PhysicalType>{children.rightParts}, numRows),
               children.exceptionCount == 0
                   ? 0
                   : childSize(positions, children.exceptionCount),
               children.exceptionCount == 0
                   ? 0
                   : childSize(
                         std::span<const uint16_t>{children.exceptionHighParts},
                         children.exceptionCount)},
              this->options_);
      const auto selected = policy.select(values, statistics, this->options_);
      ASSERT_EQ(selected.encodingType, EncodingType::ALPRD);
      EXPECT_EQ(selected.estimatedSize, expectedSize);
      EXPECT_EQ(selected.estimatedCost, 2 * expectedSize);
    }
  }
}

TYPED_TEST(ALPRDSelectionTest, constantChildHeadersAreNotScaledWithRows) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  constexpr uint32_t kNumRows = 65'536;
  const std::vector<PhysicalType> values(
      kNumRows, std::bit_cast<PhysicalType>(T{1.25}));
  ManualEncodingSelectionPolicy<T> policy{
      {{EncodingType::ALPRD, 1}},
      std::nullopt,
      std::nullopt,
      ReadFactors{{EncodingType::Constant, 1}}};
  const auto result = policy.select(
      values, Statistics<PhysicalType>::create(values), this->options_);
  ASSERT_EQ(result.encodingType, EncodingType::ALPRD);
  ASSERT_TRUE(result.estimatedSize.has_value());
  // ALPRD metadata and both Constant children remain O(1). Linear sample
  // extrapolation would incorrectly multiply their headers by 64 here.
  EXPECT_LT(result.estimatedSize.value(), 256);
}

TYPED_TEST(ALPRDSelectionTest, projectedChildrenUseSafeSelectedEstimator) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  constexpr uint32_t kNumRows = 65'536;
  const auto values = this->makeValues(kNumRows, 2);
  const auto children =
      ALPRDSelectionTestAccess::selectAndDecompose<PhysicalType>(
          values, this->options_);

  const auto codesSize = detail::EncodingSizeEstimation<uint16_t>::estimateSize(
      EncodingType::FixedBitWidth,
      children.rowCount,
      Statistics<uint16_t>::create(children.codes),
      this->options_);
  const auto rightPartsSize =
      detail::EncodingSizeEstimation<PhysicalType>::estimateSize(
          EncodingType::FixedBitWidth,
          children.rowCount,
          Statistics<PhysicalType>::create(children.rightParts),
          this->options_);
  ASSERT_FALSE(children.exceptionPositions.empty());
  std::array<uint32_t, 2> projectedExceptionPositions{};
  std::span<const uint32_t> exceptionPositionValues =
      children.exceptionPositions;
  if (children.exceptionPositions.size() < children.exceptionCount) {
    projectedExceptionPositions = {0, children.rowCount - 1};
    exceptionPositionValues = projectedExceptionPositions;
  }
  const auto exceptionPositionsSize =
      detail::EncodingSizeEstimation<uint32_t>::estimateSize(
          EncodingType::FixedBitWidth,
          children.exceptionCount,
          Statistics<uint32_t>::create(exceptionPositionValues),
          this->options_);
  const auto exceptionHighPartsSize =
      detail::EncodingSizeEstimation<uint16_t>::estimateSize(
          EncodingType::FixedBitWidth,
          children.exceptionCount,
          Statistics<uint16_t>::create(children.exceptionHighParts),
          this->options_);
  ASSERT_TRUE(codesSize.has_value());
  ASSERT_TRUE(rightPartsSize.has_value());
  ASSERT_TRUE(exceptionPositionsSize.has_value());
  ASSERT_TRUE(exceptionHighPartsSize.has_value());
  const auto expectedSize = ALPRDSelectionTestAccess::calculateSerializedSize(
      children.parameters,
      children.rowCount,
      children.exceptionCount,
      {codesSize.value(),
       rightPartsSize.value(),
       exceptionPositionsSize.value(),
       exceptionHighPartsSize.value()},
      this->options_);

  ManualEncodingSelectionPolicy<T> policy{
      {{EncodingType::ALPRD, 1}},
      std::nullopt,
      std::nullopt,
      ReadFactors{{EncodingType::FixedBitWidth, 1}}};
  const auto result = policy.select(
      values, Statistics<PhysicalType>::create(values), this->options_);
  ASSERT_EQ(result.encodingType, EncodingType::ALPRD);
  EXPECT_EQ(result.estimatedSize, expectedSize);
}

TYPED_TEST(
    ALPRDSelectionTest,
    projectedSelectionRejectsIncompatibleCandidates) {
  using PhysicalType = typename TestFixture::PhysicalType;
  constexpr uint32_t kNumRows{65'536};
  const std::array<PhysicalType, 4> values{0, 1, 2, 3};
  const auto statistics = Statistics<PhysicalType>::create(values);
  for (const bool withCompression : {false, true}) {
    SCOPED_TRACE(withCompression);
    std::optional<CompressionOptions> compressionOptions;
    if (withCompression) {
      compressionOptions = CompressionOptions{};
    }
    ManualEncodingSelectionPolicy<PhysicalType> policy{
        ReadFactors{{EncodingType::SparseBool, 1}},
        compressionOptions,
        std::nullopt};
    const auto projected =
        policy.select(values, statistics, kNumRows, this->options_);
    EXPECT_FALSE(projected.has_value());

    const auto fullInput = policy.select(values, statistics, this->options_);
    EXPECT_EQ(fullInput.encodingType, EncodingType::Trivial);
    EXPECT_FALSE(fullInput.estimatedSize.has_value());
    EXPECT_FALSE(fullInput.estimatedCost.has_value());
  }
}

TYPED_TEST(ALPRDSelectionTest, skipsUnprojectableSampledChildren) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  const auto values = this->makeValues(65'536, 2);
  for (const auto childEncoding :
       {EncodingType::RLE,
        EncodingType::Dictionary,
        EncodingType::MainlyConstant,
        EncodingType::Varint,
        EncodingType::PFOR,
        EncodingType::BlockBitPacking}) {
    SCOPED_TRACE(childEncoding);
    ManualEncodingSelectionPolicy<T> policy{
        {{EncodingType::ALPRD, 1}},
        std::nullopt,
        std::nullopt,
        ReadFactors{{childEncoding, 1}}};
    const auto result = policy.select(
        values, Statistics<PhysicalType>::create(values), this->options_);
    EXPECT_EQ(result.encodingType, EncodingType::Trivial);
  }
}

TEST(
    ALPRDSelectionProjectionTest,
    rejectsUnscaledFrequencyAndStringStatistics) {
  ManualEncodingSelectionPolicy<std::string_view> stringPolicy{
      {{EncodingType::Trivial, 1}}, std::nullopt, std::nullopt};
  const std::array<uint32_t, 4> numericValues{1, 1, 2, 3};
  const auto numericStatistics = Statistics<uint32_t>::create(numericValues);
  for (const auto encodingType :
       {EncodingType::Dictionary, EncodingType::MainlyConstant}) {
    ManualEncodingSelectionPolicy<uint32_t> numericPolicy{
        {{encodingType, 1}}, std::nullopt, std::nullopt};
    const auto projected = numericPolicy.select(
        numericValues, numericStatistics, 65'536, Encoding::Options{});
    EXPECT_FALSE(projected.has_value());
    EXPECT_TRUE(
        detail::EncodingSizeEstimation<uint32_t>::estimateSize(
            numericPolicy,
            encodingType,
            numericValues,
            numericStatistics,
            Encoding::Options{}));
  }

  const std::array<std::string_view, 2> stringValues{"a", "longer"};
  const auto projected = stringPolicy.select(
      stringValues,
      Statistics<std::string_view>::create(stringValues),
      65'536,
      Encoding::Options{});
  EXPECT_FALSE(projected.has_value());
}

TYPED_TEST(ALPRDSelectionTest, unsupportedProjectionSkipsChildEstimation) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  const auto values = this->makeValues(512, 2);
  CountingManualPolicy<T> policy{EncodingType::ALPRD};
  const auto result = policy.select(
      values, Statistics<PhysicalType>::create(values), 65'536, this->options_);
  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(policy.numNestedPolicies, 0);
}

TYPED_TEST(
    ALPRDSelectionTest,
    projectedSelectionSkipsUnsupportedCandidateBeforeChoosing) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  const auto values = this->makeValues(65'536, 2);
  ManualEncodingSelectionPolicy<T> policy{
      {{EncodingType::ALPRD, 1}},
      std::nullopt,
      std::nullopt,
      ReadFactors{
          {EncodingType::RLE, 0.000001}, {EncodingType::FixedBitWidth, 1}}};
  const auto result = policy.select(
      values, Statistics<PhysicalType>::create(values), this->options_);
  EXPECT_EQ(result.encodingType, EncodingType::ALPRD);
}

TEST(ALPRDSelectionProjectionTest, fullInputSelectionStillUsesRle) {
  std::vector<uint32_t> values(1'024, 1);
  std::fill(values.begin() + 512, values.end(), 2);
  ManualEncodingSelectionPolicy<uint32_t> policy{
      ReadFactors{{EncodingType::RLE, 1}, {EncodingType::Trivial, 1}},
      std::nullopt,
      std::nullopt};
  const auto result = policy.select(
      values, Statistics<uint32_t>::create(values), Encoding::Options{});
  EXPECT_EQ(result.encodingType, EncodingType::RLE);
}

TYPED_TEST(ALPRDSelectionTest, writingCreatesOnePolicyPerChild) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  for (bool withExceptions : {false, true}) {
    SCOPED_TRACE(withExceptions);
    auto values = this->makeValues(1'024, 1);
    if (withExceptions) {
      values[3] = std::numeric_limits<PhysicalType>::max();
    }
    auto policy =
        std::make_unique<CountingManualPolicy<T>>(EncodingType::ALPRD);
    auto* counter = policy.get();
    EncodingSelection<PhysicalType> selection{
        {.encodingType = EncodingType::ALPRD},
        Statistics<PhysicalType>::create(values),
        std::move(policy)};
    const auto encoded = ALPRDEncoding<T>::encode(
        selection, values, *this->buffer_, this->options_);
    const auto metadata =
        ALPRDEncodingBase::readMetadata(encoded, this->options_);
    EXPECT_EQ(metadata.exceptionCount > 0, withExceptions);
    EXPECT_EQ(counter->numNestedPolicies, metadata.childrenCount());
    this->check(encoded, values);
  }
}

TYPED_TEST(ALPRDSelectionTest, estimatesByteRoundedAndExactBitChildren) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  constexpr uint32_t kNumRows = 1'024;
  constexpr uint8_t kRightBits = sizeof(T) * 8 - 16;
  constexpr PhysicalType kMask = (PhysicalType{1} << kRightBits) - 1;
  ScopedVector<PhysicalType> values{
      kNumRows, this->pool_.get(), this->options_.bufferPool};
  for (uint32_t i = 0; i < kNumRows; ++i) {
    const PhysicalType high = i % 2 == 0 ? 0x3f00 : 0x4f00;
    values[i] = (high << kRightBits) | (i % 4 < 2 ? 0 : kMask);
  }
  for (auto exactBits : {false, true}) {
    SCOPED_TRACE(exactBits);
    this->options_.fixedBitWidthUseExactBits = exactBits;
    const auto parameters = ALPRDEncodingBase::selectParameters<PhysicalType>(
        values, this->options_);
    EXPECT_EQ(parameters.rightBitWidth, kRightBits);
    EXPECT_EQ(parameters.dictionarySize, 2);
    const auto estimate =
        ALPRDEncoding<T>::estimateSize(values, this->options_);
    ASSERT_TRUE(estimate);
    // The code child has 1,024 payload bytes by default and 128 with exact
    // bits. Both FBW estimates keep their fixed headers and omit padding.
    // The ALPRD header alone follows the row-count prefix option here.
    const uint64_t byteRoundedSize = sizeof(T) == 4 ? 3'111 : 7'211;
    EXPECT_EQ(
        *estimate,
        byteRoundedSize - (exactBits ? 896 : 0) -
            (this->options_.useVarintRowCount ? 2 : 0));
    const auto encoded = this->encode(
        values, this->policy({{EncodingType::ALPRD, 1}}, std::nullopt));
    this->check(encoded, values);
  }
}

TYPED_TEST(ALPRDSelectionTest, projectsOneSampleExceptionToDistinctPositions) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  constexpr uint8_t kRightBits = sizeof(T) * 8 - 16;
  constexpr PhysicalType kMask = (PhysicalType{1} << kRightBits) - 1;
  ScopedVector<PhysicalType> sample{
      1'024, this->pool_.get(), this->options_.bufferPool};
  for (uint32_t i = 0; i < sample.size(); ++i) {
    const PhysicalType high = i == 3 ? 0xff00 : 0x3f00;
    sample[i] = (high << kRightBits) | (i % 2 == 0 ? 0 : kMask);
  }
  const auto parameters =
      ALPRDEncodingBase::selectParameters<PhysicalType>(sample, this->options_);
  ASSERT_EQ(parameters.dictionarySize, 1);
  ASSERT_EQ(parameters.rightBitWidth, kRightBits);
  // One observed exception represents 64 distinct positions in 65,536 rows.
  // Byte-rounded FBW estimates 140 bytes for their uint32 child, rather than
  // treating the one sampled position as a constant across the target stream.
  const auto estimate = ALPRDEncodingBase::estimateSize<PhysicalType>(
      sample, 65'536, this->options_);
  ASSERT_TRUE(estimate);
  const uint64_t fixedPrefixSize = sizeof(T) == 4 ? 131'258 : 393'406;
  EXPECT_EQ(
      *estimate, fixedPrefixSize - (this->options_.useVarintRowCount ? 5 : 0));
  const auto encoded = this->encode(
      sample, this->policy({{EncodingType::ALPRD, 1}}, std::nullopt));
  EXPECT_EQ(
      ALPRDEncodingBase::readMetadata(encoded, this->options_).exceptionCount,
      1);
  this->check(encoded, sample);
}

TYPED_TEST(
    ALPRDSelectionTest,
    projectedExceptionChildrenRespectValueSemantics) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  constexpr uint32_t kNumRows = 65'536;
  constexpr uint8_t kRightBits = sizeof(T) * 8 - 16;
  constexpr PhysicalType kMask = (PhysicalType{1} << kRightBits) - 1;
  std::vector<PhysicalType> values(kNumRows);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = (PhysicalType{0x3f00} << kRightBits) | (i % 2 == 0 ? 0 : kMask);
  }
  const auto sampledException =
      detail::NestedAlpSizeEstimation::sampledRowIndex(
          3, ALPRDEncodingBase::kSampleSize, kNumRows);
  values[sampledException] = (PhysicalType{0xff00} << kRightBits) | kMask;
  const auto children =
      ALPRDSelectionTestAccess::selectAndDecompose<PhysicalType>(
          values, this->options_);
  ASSERT_EQ(children.exceptionPositions.size(), 1);
  ASSERT_GT(children.exceptionCount, 1);
  ManualEncodingSelectionPolicy<T> policy{
      {{EncodingType::ALPRD, 1}},
      std::nullopt,
      std::nullopt,
      ReadFactors{
          {EncodingType::Constant, 1}, {EncodingType::FixedBitWidth, 1}}};
  const auto result =
      ALPRDEncoding<T>::estimateSize(policy, values, this->options_);
  ASSERT_TRUE(result.has_value());
  const std::array<uint32_t, 2> positions{0, kNumRows - 1};
  auto positionsPolicy = policy.template create<uint32_t>(
      EncodingType::ALPRD, EncodingIdentifiers::ALPRD::ExceptionPositions);
  const auto selectedPositions =
      static_cast<EncodingSelectionPolicy<uint32_t>*>(positionsPolicy.get())
          ->select(
              positions,
              Statistics<uint32_t>::create(positions),
              children.exceptionCount,
              this->options_);
  auto highsPolicy = policy.template create<uint16_t>(
      EncodingType::ALPRD, EncodingIdentifiers::ALPRD::ExceptionHighParts);
  const auto selectedHighs =
      static_cast<EncodingSelectionPolicy<uint16_t>*>(highsPolicy.get())
          ->select(
              children.exceptionHighParts,
              Statistics<uint16_t>::create(children.exceptionHighParts),
              children.exceptionCount,
              this->options_);
  ASSERT_TRUE(selectedPositions);
  ASSERT_TRUE(selectedHighs);
  EXPECT_EQ(selectedPositions->encodingType, EncodingType::FixedBitWidth);
  EXPECT_EQ(selectedHighs->encodingType, EncodingType::Constant);
}

TYPED_TEST(ALPRDSelectionTest, projectedExceptionPositionsUseFullRowRange) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  constexpr uint32_t kNumRows = 65'536;
  constexpr uint8_t kRightBits = sizeof(T) * 8 - 16;
  constexpr PhysicalType kMask = (PhysicalType{1} << kRightBits) - 1;

  const auto makeValues = [&](std::array<uint32_t, 3> sampleIndexes) {
    std::vector<PhysicalType> values(kNumRows);
    for (uint32_t i = 0; i < values.size(); ++i) {
      values[i] =
          (PhysicalType{0x3f00} << kRightBits) | (i % 2 == 0 ? 0 : kMask);
    }
    for (const auto sampleIndex : sampleIndexes) {
      const auto row = detail::NestedAlpSizeEstimation::sampledRowIndex(
          sampleIndex, ALPRDEncodingBase::kSampleSize, kNumRows);
      values[row] = (PhysicalType{0xff00} << kRightBits) | kMask;
    }
    return values;
  };
  const auto score = [&](const std::vector<PhysicalType>& values) {
    ManualEncodingSelectionPolicy<T> policy{
        {{EncodingType::ALPRD, 1}},
        CompressionOptions{},
        std::nullopt,
        ReadFactors{
            {EncodingType::Constant, 1}, {EncodingType::FixedBitWidth, 1}}};
    return policy.select(
        values, Statistics<PhysicalType>::create(values), this->options_);
  };

  const auto clustered = makeValues({1, 2, 3});
  const auto dispersed = makeValues({1, 512, 1'022});
  const auto clusteredChildren =
      ALPRDSelectionTestAccess::selectAndDecompose<PhysicalType>(
          clustered, this->options_);
  ASSERT_EQ(clusteredChildren.exceptionPositions.size(), 3);
  ASSERT_GT(
      clusteredChildren.exceptionCount,
      clusteredChildren.exceptionPositions.size());

  const auto clusteredScore = score(clustered);
  const auto dispersedScore = score(dispersed);
  ASSERT_EQ(clusteredScore.encodingType, EncodingType::ALPRD);
  ASSERT_EQ(dispersedScore.encodingType, EncodingType::ALPRD);
  EXPECT_EQ(clusteredScore.estimatedSize, dispersedScore.estimatedSize);
  EXPECT_EQ(clusteredScore.estimatedCost, dispersedScore.estimatedCost);
}

TYPED_TEST(ALPRDSelectionTest, constantAndTargetRowBoundaries) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  for (uint32_t numRows : {1, 127, 128, 1'024, 65'536}) {
    SCOPED_TRACE(numRows);
    const std::array<PhysicalType, 1> sample{
        std::bit_cast<PhysicalType>(T{1.25})};
    const auto estimate = ALPRDEncodingBase::estimateSize<PhysicalType>(
        sample, numRows, this->options_);
    ASSERT_TRUE(estimate);
    // The outer encoding and two Constant children each carry one prefix.
    // The remaining bytes hold split metadata, one dictionary entry, two
    // lengths, and the two constant values.
    const auto prefixSize = EncodingPrefix::serializedSize(
        numRows, this->options_.useVarintRowCount);
    EXPECT_EQ(*estimate, 3 * prefixSize + 9 + sizeof(PhysicalType));
  }
  const auto values = this->makeValues(1'024, 2);
  const auto estimate = ALPRDEncodingBase::estimateSize<PhysicalType>(
      values, std::numeric_limits<uint32_t>::max(), this->options_);
  ASSERT_TRUE(estimate);
  EXPECT_GT(*estimate, std::numeric_limits<uint32_t>::max());
  EXPECT_LT(
      *estimate, uint64_t{std::numeric_limits<uint32_t>::max()} * sizeof(T));
  EXPECT_THROW(
      ALPRDEncodingBase::estimateSize<PhysicalType>(values, 1, this->options_),
      NimbleInternalError);
}

TYPED_TEST(ALPRDSelectionTest, selectsBySizeAndReadFactor) {
  const auto values = this->makeValues(1'024, 2);
  const auto result = this->select(values, candidates(), std::nullopt);
  EXPECT_EQ(result.encodingType, EncodingType::ALPRD);
  ASSERT_TRUE(result.estimatedSize);
  const auto encoded =
      this->encode(values, this->policy(candidates(), std::nullopt));
  EXPECT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::ALPRD);
  this->check(encoded, values);

  auto favorTrivial = candidates();
  const auto trivial = std::find_if(
      favorTrivial.begin(), favorTrivial.end(), [](const auto& entry) {
        return entry.first == EncodingType::Trivial;
      });
  ASSERT_NE(trivial, favorTrivial.end());
  trivial->second = 0.01;
  EXPECT_NE(
      this->select(values, favorTrivial, std::nullopt).encodingType,
      EncodingType::ALPRD);
  favorTrivial.pop_back();
  EXPECT_NE(
      this->select(values, favorTrivial, std::nullopt).encodingType,
      EncodingType::ALPRD);
}

TYPED_TEST(ALPRDSelectionTest, keepsBetterExistingCandidates) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  std::vector<PhysicalType> decimals(1'024);
  for (uint32_t i = 0; i < decimals.size(); ++i) {
    decimals[i] = std::bit_cast<PhysicalType>(static_cast<T>(i % 997) / 100);
  }
  std::mt19937_64 random{73};
  auto randomBits = decimals;
  for (auto& value : randomBits) {
    value = random();
  }
  for (const auto& values :
       {decimals, randomBits, this->makeValues(1'024, 1)}) {
    const auto result = this->select(values, candidates(), std::nullopt);
    EXPECT_NE(result.encodingType, EncodingType::ALPRD);
    const auto encoded =
        this->encode(values, this->policy(candidates(), std::nullopt));
    EXPECT_NE(EncodingPrefix::encodingType(encoded), EncodingType::ALPRD);
    this->check(encoded, values);
  }
}

TYPED_TEST(ALPRDSelectionTest, estimatesSelectedChildEncodings) {
  const auto values = this->makeValues(512, 2);
  const auto packed = this->select(
      values,
      {{EncodingType::ALPRD, 1}},
      ReadFactors{{EncodingType::FixedBitWidth, 1}});
  const auto trivial = this->select(
      values,
      {{EncodingType::ALPRD, 1}},
      ReadFactors{{EncodingType::Trivial, 1}});
  ASSERT_TRUE(packed.estimatedSize);
  ASSERT_TRUE(trivial.estimatedSize);
  EXPECT_LT(*packed.estimatedSize, *trivial.estimatedSize);
  std::optional<ALPRDEncodingBase::Parameters> trained;
  std::optional<uint64_t> packedSize;
  std::optional<uint64_t> trivialSize;
  for (auto child : {EncodingType::FixedBitWidth, EncodingType::Trivial}) {
    const auto encoded = this->encode(
        values,
        this->policy({{EncodingType::ALPRD, 1}}, ReadFactors{{child, 1}}));
    // Generic child estimators deliberately approximate serialization prefix
    // sizes. Validate that their ranking follows the actual complete trees,
    // rather than requiring a byte-exact estimate.
    if (child == EncodingType::FixedBitWidth) {
      packedSize = encoded.size();
    } else {
      trivialSize = encoded.size();
    }
    const auto layout = EncodingLayoutCapture::capture(encoded, this->options_);
    EXPECT_EQ(layout.encodingType(), EncodingType::ALPRD);
    EXPECT_EQ(layout.child(0)->encodingType(), child);
    EXPECT_EQ(layout.child(1)->encodingType(), child);
    const auto parameters =
        ALPRDEncodingBase::readMetadata(encoded, this->options_).parameters;
    if (trained) {
      EXPECT_EQ(parameters.rightBitWidth, trained->rightBitWidth);
      EXPECT_EQ(parameters.dictionarySize, trained->dictionarySize);
      EXPECT_EQ(parameters.dictionary, trained->dictionary);
    } else {
      trained = parameters;
    }
    this->check(encoded, values);
  }
  ASSERT_TRUE(packedSize.has_value());
  ASSERT_TRUE(trivialSize.has_value());
  EXPECT_LT(packedSize.value(), trivialSize.value());
}

TYPED_TEST(ALPRDSelectionTest, dictionaryEstimateUsesConfiguredFloatAlphabet) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  this->options_.fixedBitWidthUseExactBits = true;
  const ReadFactors factors{
      {EncodingType::Dictionary, 1}, {EncodingType::Trivial, 0.6}};
  for (auto encodingType : {EncodingType::ALP, EncodingType::ALPRD}) {
    SCOPED_TRACE(encodingType);
    auto alphabet = this->makeValues(1'024, 2);
    if (encodingType == EncodingType::ALP) {
      for (uint32_t i = 0; i < alphabet.size(); ++i) {
        alphabet[i] = std::bit_cast<PhysicalType>(
            static_cast<T>(static_cast<int32_t>(i) - 512) / 100);
      }
    }
    auto values = alphabet;
    values.insert(values.end(), alphabet.begin(), alphabet.end());
    for (bool enableAlphabetEncoding : {false, true}) {
      SCOPED_TRACE(enableAlphabetEncoding);
      ReadFactors nested{
          {EncodingType::Constant, 1},
          {EncodingType::Trivial, 1},
          {EncodingType::FixedBitWidth, 1},
      };
      if (enableAlphabetEncoding) {
        nested.emplace_back(encodingType, 1);
      }
      const auto selected = this->select(values, factors, nested);
      EXPECT_EQ(
          selected.encodingType,
          enableAlphabetEncoding ? EncodingType::Dictionary
                                 : EncodingType::Trivial);
      const auto encoded = this->encode(values, this->policy(factors, nested));
      EXPECT_EQ(EncodingPrefix::encodingType(encoded), selected.encodingType);
      if (enableAlphabetEncoding) {
        const auto layout =
            EncodingLayoutCapture::capture(encoded, this->options_);
        ASSERT_EQ(layout.encodingType(), EncodingType::Dictionary);
        EXPECT_EQ(layout.child(0)->encodingType(), encodingType);
      }
      this->check(encoded, values);
    }
  }
}

TYPED_TEST(ALPRDSelectionTest, rleEstimateUsesConfiguredFloatRunValues) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  this->options_.fixedBitWidthUseExactBits = true;
  const ReadFactors factors{
      {EncodingType::RLE, 1}, {EncodingType::Trivial, 0.4}};
  for (auto encodingType : {EncodingType::ALP, EncodingType::ALPRD}) {
    SCOPED_TRACE(encodingType);
    auto runValues = this->makeValues(1'024, 2);
    if (encodingType == EncodingType::ALP) {
      for (uint32_t i = 0; i < runValues.size(); ++i) {
        runValues[i] = std::bit_cast<PhysicalType>(
            static_cast<T>(static_cast<int32_t>(i) - 512) / 100);
      }
    }
    std::vector<PhysicalType> values;
    values.reserve(2 * runValues.size());
    for (auto value : runValues) {
      values.push_back(value);
      values.push_back(value);
    }
    for (bool enableRunValueEncoding : {false, true}) {
      SCOPED_TRACE(enableRunValueEncoding);
      ReadFactors nested{
          {EncodingType::Constant, 1},
          {EncodingType::Trivial, 1},
          {EncodingType::FixedBitWidth, 1},
      };
      if (enableRunValueEncoding) {
        nested.emplace_back(encodingType, 1);
      }
      const auto selected = this->select(values, factors, nested);
      EXPECT_EQ(
          selected.encodingType,
          enableRunValueEncoding ? EncodingType::RLE : EncodingType::Trivial);
      const auto encoded = this->encode(values, this->policy(factors, nested));
      EXPECT_EQ(EncodingPrefix::encodingType(encoded), selected.encodingType);
      if (enableRunValueEncoding) {
        const auto layout =
            EncodingLayoutCapture::capture(encoded, this->options_);
        ASSERT_EQ(layout.encodingType(), EncodingType::RLE);
        EXPECT_EQ(layout.child(1)->encodingType(), encodingType);
      }
      this->check(encoded, values);
    }
  }
}

TYPED_TEST(ALPRDSelectionTest, nestedSelectionUsesOnlyConfiguredCandidates) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  std::array<PhysicalType, 128> values{};
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = std::bit_cast<PhysicalType>(static_cast<T>(i + 1) / 4);
  }
  const auto statistics = Statistics<PhysicalType>::create(values);
  const std::array<std::pair<EncodingType, NestedEncodingIdentifier>, 3>
      valueChildren{
          std::pair{
              EncodingType::Dictionary,
              EncodingIdentifiers::Dictionary::Alphabet,
          },
          std::pair{
              EncodingType::RLE,
              EncodingIdentifiers::RunLength::RunValues,
          },
          std::pair{
              EncodingType::MainlyConstant,
              EncodingIdentifiers::MainlyConstant::OtherValues,
          },
      };
  for (const auto& [parent, nestedIdentifier] : valueChildren) {
    SCOPED_TRACE(toString(parent));
    for (const auto& nestedCandidates :
         {ReadFactors{},
          ReadFactors{{EncodingType::ALP, 1}},
          ReadFactors{{EncodingType::ALPRD, 1}},
          ReadFactors{{EncodingType::ALP, 1}, {EncodingType::ALPRD, 1}}}) {
      SCOPED_TRACE(::testing::PrintToString(nestedCandidates));
      auto parentPolicy = this->policy({{parent, 1}}, nestedCandidates);
      auto nestedPolicy =
          parentPolicy->template create<T>(parent, nestedIdentifier);
      auto& typedPolicy =
          static_cast<ManualEncodingSelectionPolicy<T>&>(*nestedPolicy);
      const auto selected =
          typedPolicy.select(values, statistics, this->options_);
      if (nestedCandidates.empty()) {
        EXPECT_EQ(selected.encodingType, EncodingType::Trivial);
      } else {
        EXPECT_THAT(
            nestedCandidates,
            ::testing::Contains(
                ::testing::Pair(selected.encodingType, ::testing::_)));
      }
    }
  }
}

TYPED_TEST(ALPRDSelectionTest, preservesChildPolicyScoring) {
  using PhysicalType = typename TestFixture::PhysicalType;
  const std::array<PhysicalType, 4> values{0, 1, 0, 1};
  const auto statistics = Statistics<PhysicalType>::create(values);
  constexpr uint64_t kFixedBitWidthSize = sizeof(PhysicalType) == 4 ? 16 : 20;
  constexpr uint64_t kTrivialSize = sizeof(PhysicalType) == 4 ? 23 : 39;
  EXPECT_EQ(
      FixedBitWidthEncoding<PhysicalType>::estimateSize(
          values.size(), statistics, this->options_),
      kFixedBitWidthSize);
  EXPECT_EQ(
      TrivialEncoding<PhysicalType>::estimateSize(values.size()), kTrivialSize);
  const auto trivialReadFactor =
      static_cast<float>(kFixedBitWidthSize + 3) / kTrivialSize;
  const ReadFactors factors{
      {EncodingType::Trivial, trivialReadFactor},
      {EncodingType::FixedBitWidth, 1},
  };
  auto childPolicy =
      std::make_unique<ManualEncodingSelectionPolicy<PhysicalType>>(
          factors, std::nullopt, std::nullopt);
  EXPECT_FALSE(childPolicy->hasAlpLikeCandidates());
  const auto selected = childPolicy->select(values, statistics, this->options_);
  ASSERT_EQ(selected.encodingType, EncodingType::FixedBitWidth);
  EXPECT_EQ(selected.estimatedSize, kFixedBitWidthSize);

  const auto logicalSelection = this->select(values, factors, std::nullopt);
  EXPECT_EQ(logicalSelection.encodingType, selected.encodingType);
  EXPECT_EQ(logicalSelection.estimatedSize, selected.estimatedSize);
}

TYPED_TEST(ALPRDSelectionTest, estimatesLargeInputFromBoundedSample) {
  using PhysicalType = typename TestFixture::PhysicalType;
  const auto values = this->makeValues(65'536, 2);
  for (auto exactBits : {false, true}) {
    SCOPED_TRACE(exactBits);
    this->options_.fixedBitWidthUseExactBits = exactBits;
    const auto estimate = ALPRDEncodingBase::estimateSize<PhysicalType>(
        values, values.size(), this->options_);
    ASSERT_TRUE(estimate);
    // Compare with the scalar encodings modeled by the heuristic. A policy
    // that excludes Constant can produce larger constant child streams.
    const auto encoded = this->encode(
        values,
        this->policy(
            {{EncodingType::ALPRD, 1}},
            ReadFactors{
                {EncodingType::Constant, 1},
                {EncodingType::FixedBitWidth, 1},
                {EncodingType::Trivial, 1},
            }));
    EXPECT_NEAR(*estimate, encoded.size(), encoded.size() * 0.02);
    this->check(encoded, values);
  }
}

TYPED_TEST(ALPRDSelectionTest, estimatesPeriodicInput) {
  using PhysicalType = typename TestFixture::PhysicalType;
  using T = typename TestFixture::T;
  auto values = this->makeValues(65'536, 2);
  for (uint32_t i = 0; i < values.size(); i += 64) {
    values[i] = 0;
  }
  const ReadFactors children{
      {EncodingType::Constant, 1},
      {EncodingType::FixedBitWidth, 1},
      {EncodingType::Trivial, 1},
  };
  auto policy = this->policy(children, std::nullopt);
  const auto estimate = ALPRDEncodingBase::estimateSize<PhysicalType>(
      values, values.size(), this->options_);
  ASSERT_TRUE(estimate);
  EncodingSelection<PhysicalType> selection{
      {.encodingType = EncodingType::ALPRD},
      Statistics<PhysicalType>::create(values),
      std::move(policy)};
  const auto encoded = ALPRDEncoding<T>::encode(
      selection, values, *this->buffer_, this->options_);
  EXPECT_NEAR(*estimate, encoded.size(), encoded.size() * 0.1);
  this->check(encoded, values);
}

TYPED_TEST(ALPRDSelectionTest, prefersDictionaryForRepeatingAlphabet) {
  const auto alphabet = this->makeValues(512, 2);
  std::vector<typename TestFixture::PhysicalType> values(65'536);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = alphabet[i % alphabet.size()];
  }
  const auto encoded =
      this->encode(values, this->policy(candidates(), std::nullopt));
  EXPECT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Dictionary);
  this->check(encoded, values);
}

TYPED_TEST(ALPRDSelectionTest, selectsNestedFloatingPointValues) {
  const auto alphabet = this->makeValues(512, 2);
  for (auto parent :
       {EncodingType::Dictionary,
        EncodingType::RLE,
        EncodingType::MainlyConstant}) {
    SCOPED_TRACE(parent);
    std::vector<typename TestFixture::PhysicalType> values;
    values.reserve(4'096);
    for (uint32_t i = 0; i < 4'096; ++i) {
      values.push_back(
          parent == EncodingType::MainlyConstant
              ? (i % 8 == 0 ? alphabet[i / 8] : 0)
              : alphabet[(parent == EncodingType::RLE ? i / 8 : i) % 512]);
    }
    const auto result = this->select(
        values, {{parent, 1}, {EncodingType::Trivial, 1}}, candidates());
    EXPECT_EQ(result.encodingType, parent);
    const auto encoded = this->encode(
        values,
        this->policy({{parent, 1}, {EncodingType::Trivial, 1}}, candidates()));
    const auto layout = EncodingLayoutCapture::capture(encoded, this->options_);
    EXPECT_EQ(layout.encodingType(), parent);
    const auto slot = parent == EncodingType::Dictionary ? 0 : 1;
    ASSERT_TRUE(layout.child(slot));
    EXPECT_EQ(layout.child(slot)->encodingType(), EncodingType::ALPRD);
    this->check(encoded, values);
  }
}

TYPED_TEST(ALPRDSelectionTest, nullableSelectionAndReplay) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  const auto alphabet = this->makeValues(512, 2);
  for (auto parent :
       {EncodingType::ALPRD,
        EncodingType::Dictionary,
        EncodingType::RLE,
        EncodingType::MainlyConstant}) {
    SCOPED_TRACE(parent);
    const uint32_t numValues = parent == EncodingType::ALPRD ? 512 : 4'096;
    std::vector<T> values;
    std::vector<PhysicalType> expected(2 * numValues, 0);
    ScopedVector<bool> notNulls{
        expected.size(), this->pool_.get(), this->options_.bufferPool};
    std::fill(notNulls.begin(), notNulls.end(), false);
    for (uint32_t i = 0; i < numValues; ++i) {
      const auto bits = parent == EncodingType::MainlyConstant
          ? (i % 8 == 0 ? alphabet[i / 8] : 0)
          : alphabet[(parent == EncodingType::RLE ? i / 8 : i) % 512];
      values.push_back(std::bit_cast<T>(bits));
      expected[2 * i] = bits;
      notNulls[2 * i] = true;
    }
    std::unique_ptr<EncodingSelectionPolicy<T>> selectionPolicy;
    if (parent == EncodingType::ALPRD) {
      selectionPolicy = this->policy(candidates(), std::nullopt);
    } else {
      // Bind the container to verify ALPRD selection for its floating-point
      // value child under the Nullable wrapper.
      selectionPolicy = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
          EncodingLayout{
              parent,
              {},
              CompressionType::Uncompressed,
              {std::nullopt, std::nullopt}},
          std::nullopt,
          [](DataType type) {
            return ManualEncodingSelectionPolicyFactory{
                candidates(), std::nullopt}
                .createPolicy(type);
          });
    }
    const auto encoded = EncodingFactory::encodeNullable<T>(
        std::move(selectionPolicy),
        values,
        notNulls,
        *this->buffer_,
        this->options_);
    const auto layout = EncodingLayoutCapture::capture(encoded, this->options_);
    ASSERT_EQ(layout.encodingType(), parent);
    if (parent != EncodingType::ALPRD) {
      const auto slot = parent == EncodingType::Dictionary ? 0 : 1;
      ASSERT_TRUE(layout.child(slot));
      EXPECT_EQ(layout.child(slot)->encodingType(), EncodingType::ALPRD);
    }
    this->check(encoded, expected);
    auto replay = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
        layout, std::nullopt, [](DataType type) {
          return ManualEncodingSelectionPolicyFactory{
              candidates(), std::nullopt}
              .createPolicy(type);
        });
    const auto replayed = EncodingFactory::encodeNullable<T>(
        std::move(replay), values, notNulls, *this->buffer_, this->options_);
    EXPECT_EQ(
        EncodingLayoutCapture::capture(replayed, this->options_).encodingType(),
        parent);
    this->check(replayed, expected);
  }
}

TYPED_TEST(ALPRDSelectionTest, replayLimitsFallbackToUnspecifiedChildren) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  const auto values = this->makeValues(257, 2);
  ScopedVector<T> logical{
      values.size(), this->pool_.get(), this->options_.bufferPool};
  ScopedVector<bool> notNulls{
      2 * values.size(), this->pool_.get(), this->options_.bufferPool};
  std::vector<PhysicalType> expected(notNulls.size(), 0);
  for (uint32_t i = 0; i < values.size(); ++i) {
    logical[i] = std::bit_cast<T>(values[i]);
    notNulls[2 * i] = true;
    notNulls[2 * i + 1] = false;
    expected[2 * i] = values[i];
  }

  const EncodingLayout trivial{
      EncodingType::Trivial, {}, CompressionType::Uncompressed};
  const EncodingLayout packed{
      EncodingType::FixedBitWidth, {}, CompressionType::Uncompressed};
  const std::vector<std::pair<EncodingLayout, std::vector<DataType>>> layouts{
      {trivial, {DataType::Bool}},
      {packed, {DataType::Bool}},
      {{EncodingType::Dictionary,
        {},
        CompressionType::Uncompressed,
        {trivial, packed}},
       {DataType::Bool}},
      {{EncodingType::RLE,
        {},
        CompressionType::Uncompressed,
        {packed, trivial}},
       {DataType::Bool}},
      {{EncodingType::MainlyConstant,
        {},
        CompressionType::Uncompressed,
        {std::nullopt, trivial}},
       {DataType::Bool, DataType::Bool}},
      {{EncodingType::RLE,
        {},
        CompressionType::Uncompressed,
        {packed,
         EncodingLayout{
             EncodingType::Dictionary,
             {},
             CompressionType::Uncompressed,
             {trivial, std::nullopt}}}},
       {DataType::Uint32, DataType::Bool}},
  };
  for (const auto& [layout, expectedTypes] : layouts) {
    SCOPED_TRACE(layout.encodingType());
    std::vector<DataType> fallbackTypes;
    auto replay = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
        layout, std::nullopt, [&](DataType type) {
          const auto index = fallbackTypes.size();
          fallbackTypes.push_back(type);
          NIMBLE_CHECK_LT(
              index, expectedTypes.size(), "Unexpected fallback call.");
          NIMBLE_CHECK_EQ(
              type, expectedTypes[index], "Unexpected fallback type.");
          return ManualEncodingSelectionPolicyFactory{
              {{EncodingType::Trivial, 1}}, std::nullopt}
              .createPolicy(type);
        });
    const auto encoded = EncodingFactory::encodeNullable<T>(
        std::move(replay), logical, notNulls, *this->buffer_, this->options_);
    EXPECT_THAT(fallbackTypes, ::testing::ElementsAreArray(expectedTypes));
    const auto childOffset =
        EncodingPrefix::prefixSize(encoded, this->options_.useVarintRowCount) +
        sizeof(uint32_t);
    EXPECT_EQ(
        EncodingPrefix::dataType(encoded.substr(childOffset)),
        TypeTraits<PhysicalType>::dataType);
    EXPECT_EQ(
        EncodingLayoutCapture::capture(encoded, this->options_).encodingType(),
        layout.encodingType());
    this->check(encoded, expected);
  }
}

TYPED_TEST(ALPRDSelectionTest, replaySelectsAlpCodecsForUnspecifiedValues) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  const auto values = this->makeValues(257, 2);
  ScopedVector<T> logical{
      values.size(), this->pool_.get(), this->options_.bufferPool};
  ScopedVector<bool> notNulls{
      2 * values.size(), this->pool_.get(), this->options_.bufferPool};
  std::vector<PhysicalType> expected(notNulls.size(), 0);
  for (uint32_t i = 0; i < values.size(); ++i) {
    logical[i] = std::bit_cast<T>(values[i]);
    notNulls[2 * i] = true;
    notNulls[2 * i + 1] = false;
    expected[2 * i] = values[i];
  }
  const EncodingLayout trivial{
      EncodingType::Trivial, {}, CompressionType::Uncompressed};
  for (auto parent :
       {EncodingType::Dictionary,
        EncodingType::RLE,
        EncodingType::MainlyConstant}) {
    SCOPED_TRACE(parent);
    const uint8_t valueChild = parent == EncodingType::Dictionary ? 0 : 1;
    for (auto child : {EncodingType::ALP, EncodingType::ALPRD}) {
      SCOPED_TRACE(child);
      std::vector<std::optional<const EncodingLayout>> children{
          trivial, trivial};
      children[valueChild].reset();
      auto replay = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
          EncodingLayout{
              parent, {}, CompressionType::Uncompressed, std::move(children)},
          std::nullopt,
          [child](DataType type) {
            return ManualEncodingSelectionPolicyFactory{
                {{EncodingType::Trivial, 1'000'000}, {child, 1}},
                std::nullopt,
                ManualEncodingSelectionPolicyFactory::
                    defaultEncodingReadFactors()}
                .createPolicy(type);
          });
      const auto encoded = EncodingFactory::encodeNullable<T>(
          std::move(replay), logical, notNulls, *this->buffer_, this->options_);
      const auto layout =
          EncodingLayoutCapture::capture(encoded, this->options_);
      EXPECT_EQ(layout.encodingType(), parent);
      ASSERT_TRUE(layout.child(valueChild));
      EXPECT_EQ(layout.child(valueChild)->encodingType(), child);
      this->check(encoded, expected);
    }
  }
}

TYPED_TEST(ALPRDSelectionTest, replayRetainsLayoutAndRetrainsSplit) {
  using T = typename TestFixture::T;
  auto values = this->makeValues(512, 2);
  const auto encoded =
      this->encode(values, this->policy(candidates(), std::nullopt));
  auto layout = EncodingLayoutCapture::capture(encoded, this->options_);
  ASSERT_EQ(layout.encodingType(), EncodingType::ALPRD);
  // The new input favors an existing codec. Replaying a captured ALPRD layout
  // still honors that binding while rebuilding its data-dependent dictionary.
  values = this->makeValues(512, 1);
  EXPECT_NE(
      this->select(values, candidates(), std::nullopt).encodingType,
      EncodingType::ALPRD);
  auto replay = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
      std::move(layout), std::nullopt, [](DataType type) {
        return ManualEncodingSelectionPolicyFactory{candidates(), std::nullopt}
            .createPolicy(type);
      });
  const auto replayed = this->encode(values, std::move(replay));
  EXPECT_EQ(EncodingPrefix::encodingType(replayed), EncodingType::ALPRD);
  this->check(replayed, values);
}

TYPED_TEST(ALPRDSelectionTest, randomPolicyExercisesAlprd) {
  using T = typename TestFixture::T;
  const auto values = this->makeValues(127, 2);
  bool found{false};
  for (uint64_t seed = 0; seed < 16; ++seed) {
    auto policy = std::make_unique<testing::RandomEncodingSelectionPolicy<T>>(
        seed,
        std::vector<EncodingType>{EncodingType::Trivial, EncodingType::ALPRD},
        std::nullopt);
    const auto encoded = this->encode(values, std::move(policy));
    found |= EncodingPrefix::encodingType(encoded) == EncodingType::ALPRD;
    this->check(encoded, values);
  }
  EXPECT_TRUE(found);
}

TEST(ALPRDSelectionProjectionTest, randomChildrenRetainSize) {
  const std::array<uint16_t, 2> values{1, 2};
  const auto statistics = Statistics<uint16_t>::create(values);
  testing::RandomEncodingSelectionPolicy<double> parent{
      73,
      {EncodingType::ALPRD, EncodingType::Trivial, EncodingType::RLE},
      std::nullopt};
  auto child = parent.create<uint16_t>(
      EncodingType::ALPRD, EncodingIdentifiers::ALPRD::Codes);
  auto* policy = static_cast<EncodingSelectionPolicy<uint16_t>*>(child.get());
  for (const bool useVarint : {false, true}) {
    const Encoding::Options options{.useVarintRowCount = useVarint};
    const auto full = policy->select(values, statistics, options);
    ASSERT_TRUE(full.estimatedSize);
    EXPECT_FALSE(full.estimatedCost.has_value());
    for (const size_t numRows : {128, 65'536}) {
      SCOPED_TRACE(numRows);
      const auto projected =
          policy->select(values, statistics, numRows, options);
      ASSERT_TRUE(projected);
      EXPECT_EQ(projected->encodingType, EncodingType::Trivial);
      EXPECT_EQ(
          projected->estimatedSize,
          EncodingPrefix::kFixedPrefixSize + 1 + numRows * sizeof(uint16_t));
      EXPECT_FALSE(projected->estimatedCost.has_value());
      const auto repeated =
          policy->select(values, statistics, numRows, options);
      ASSERT_TRUE(repeated);
      EXPECT_EQ(repeated->encodingType, projected->encodingType);
      EXPECT_EQ(repeated->estimatedSize, projected->estimatedSize);
    }
  }
}

TEST(ALPRDSelectionProjectionTest, randomConstantUsesTargetRowCount) {
  const std::array<uint16_t, 1> values{7};
  testing::RandomEncodingSelectionPolicy<uint16_t> policy{
      73, {EncodingType::Constant}, std::nullopt};
  for (const bool useVarint : {false, true}) {
    const Encoding::Options options{.useVarintRowCount = useVarint};
    const auto projected = policy.select(
        values, Statistics<uint16_t>::create(values), 65'536, options);
    ASSERT_TRUE(projected);
    EXPECT_EQ(projected->encodingType, EncodingType::Constant);
    EXPECT_EQ(
        projected->estimatedSize,
        EncodingPrefix::serializedSize(65'536, useVarint) + sizeof(uint16_t));
    EXPECT_FALSE(projected->estimatedCost.has_value());
  }
}

TEST(ALPRDSelectionProjectionTest, randomFallbackAndUnsupportedProjection) {
  const std::array<uint16_t, 2> values{1, 2};
  const auto statistics = Statistics<uint16_t>::create(values);
  const Encoding::Options options;
  testing::RandomEncodingSelectionPolicy<uint16_t> emptyCandidates{73, {}};
  const auto full = emptyCandidates.select(values, statistics, options);
  EXPECT_EQ(full.encodingType, EncodingType::Trivial);
  ASSERT_TRUE(full.estimatedSize);
  EXPECT_FALSE(full.estimatedCost.has_value());
  EXPECT_NE(
      dynamic_cast<NoCompressionPolicy*>(full.compressionPolicyFactory().get()),
      nullptr);
  const auto projected =
      emptyCandidates.select(values, statistics, 65'536, options);
  ASSERT_TRUE(projected);
  EXPECT_EQ(projected->encodingType, EncodingType::Trivial);
  ASSERT_TRUE(projected->estimatedSize);
  EXPECT_GT(*projected->estimatedSize, *full.estimatedSize);
  EXPECT_FALSE(projected->estimatedCost.has_value());
  const std::span<const uint16_t> noValues;
  const auto empty = emptyCandidates.select(
      noValues, Statistics<uint16_t>::create(noValues), options);
  EXPECT_EQ(empty.estimatedSize, 0);
  EXPECT_FALSE(empty.estimatedCost.has_value());

  testing::RandomEncodingSelectionPolicy<uint16_t> unsupported{
      73, {EncodingType::RLE}, std::nullopt};
  const auto unsupportedProjection =
      unsupported.select(values, statistics, 65'536, options);
  ASSERT_TRUE(unsupportedProjection);
  EXPECT_EQ(unsupportedProjection->encodingType, EncodingType::Trivial);
  EXPECT_EQ(unsupportedProjection->estimatedSize, projected->estimatedSize);
  EXPECT_EQ(unsupportedProjection->estimatedCost, projected->estimatedCost);
  EXPECT_EQ(
      unsupported.select(values, statistics, options).encodingType,
      EncodingType::RLE);
  testing::RandomEncodingSelectionPolicy<uint16_t> incompatible{
      73, {EncodingType::Constant, EncodingType::Varint}, std::nullopt};
  const auto incompatibleProjection =
      incompatible.select(values, statistics, 65'536, options);
  ASSERT_TRUE(incompatibleProjection);
  EXPECT_EQ(incompatibleProjection->encodingType, EncodingType::Trivial);
  EXPECT_EQ(incompatibleProjection->estimatedSize, projected->estimatedSize);
  EXPECT_EQ(incompatibleProjection->estimatedCost, projected->estimatedCost);
  const auto fallback = incompatible.select(values, statistics, options);
  EXPECT_EQ(fallback.encodingType, EncodingType::Trivial);
  EXPECT_EQ(fallback.estimatedSize, full.estimatedSize);
  EXPECT_EQ(fallback.estimatedCost, full.estimatedCost);
}

TYPED_TEST(
    ALPRDSelectionTest,
    randomPolicyEstimatesBoundedChildrenAndRoundTrips) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  constexpr uint32_t kNumRows = 65'536;
  constexpr uint8_t kRightBits = sizeof(T) * 8 - 16;
  const PhysicalType mask = (PhysicalType{1} << kRightBits) - 1;
  auto values = this->makeValues(kNumRows, 1);
  const auto exceptionRow = detail::NestedAlpSizeEstimation::sampledRowIndex(
      3, ALPRDEncodingBase::kSampleSize, kNumRows);
  values[exceptionRow] = (PhysicalType{0xff00} << kRightBits) | mask;
  const auto children =
      ALPRDSelectionTestAccess::selectAndDecompose<PhysicalType>(
          values, this->options_);
  ASSERT_LT(children.codes.size(), kNumRows);
  ASSERT_EQ(children.exceptionPositions.size(), 1);
  ASSERT_GT(children.exceptionCount, 1);
  for (const bool withCompression : {false, true}) {
    SCOPED_TRACE(withCompression);
    auto policy = std::make_unique<testing::RandomEncodingSelectionPolicy<T>>(
        73,
        std::vector<EncodingType>{EncodingType::ALPRD},
        withCompression
            ? std::optional<CompressionOptions>{CompressionOptions{}}
            : std::nullopt);
    const auto selected = policy->select(
        values, Statistics<PhysicalType>::create(values), this->options_);
    ASSERT_EQ(selected.encodingType, EncodingType::ALPRD);
    ASSERT_TRUE(selected.estimatedSize);
    EXPECT_GT(*selected.estimatedSize, 0);
    EXPECT_FALSE(selected.estimatedCost.has_value());
    const auto encoded = this->encode(values, std::move(policy));
    EXPECT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::ALPRD);
    EXPECT_EQ(
        ALPRDEncodingBase::readMetadata(encoded, this->options_).exceptionCount,
        1);
    this->check(encoded, values);
  }
}

TYPED_TEST(ALPRDSelectionTest, randomPolicyPreservesFullInputChoices) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  const auto values = this->makeValues(65'536, 1);
  const auto statistics = Statistics<PhysicalType>::create(values);
  const std::vector<EncodingType> candidates{
      EncodingType::ALPRD, EncodingType::RLE};
  ASSERT_TRUE(ALPRDEncoding<T>::estimateSize(values, this->options_));
  for (const bool withCompression : {false, true}) {
    SCOPED_TRACE(withCompression);
    bool checkedChildren = false;
    for (uint64_t seed = 0; seed < 32; ++seed) {
      SCOPED_TRACE(seed);
      const auto selectionSeed = folly::hash::hash_combine(
          seed, folly::hash::hash_range(values.begin(), values.end()));
      std::mt19937_64 generator{selectionSeed};
      std::uniform_int_distribution<size_t> distribution(
          0, candidates.size() - 1);
      const auto expected = candidates[distribution(generator)];
      auto policy = std::make_unique<testing::RandomEncodingSelectionPolicy<T>>(
          seed,
          candidates,
          withCompression
              ? std::optional<CompressionOptions>{CompressionOptions{}}
              : std::nullopt);
      const auto selected = policy->select(values, statistics, this->options_);
      ASSERT_EQ(selected.encodingType, expected);
      ASSERT_TRUE(selected.estimatedSize);
      EXPECT_FALSE(selected.estimatedCost.has_value());
      EXPECT_EQ(
          dynamic_cast<ConfiguredCompressionPolicy*>(
              selected.compressionPolicyFactory().get()) != nullptr,
          withCompression);
      if (expected == EncodingType::ALPRD && !checkedChildren) {
        const auto encoded = this->encode(values, std::move(policy));
        const auto metadata =
            ALPRDEncodingBase::readMetadata(encoded, this->options_);
        EXPECT_EQ(
            EncodingPrefix::encodingType(metadata.children[0]),
            EncodingType::RLE);
        EXPECT_EQ(
            EncodingPrefix::encodingType(metadata.children[1]),
            EncodingType::RLE);
        this->check(encoded, values);
        checkedChildren = true;
      }
    }
    EXPECT_TRUE(checkedChildren);
  }
}

TEST(ALPRDSelectionConfigurationTest, optInAndFloatingPointEligibility) {
  const auto defaults =
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
  EXPECT_THAT(
      defaults,
      ::testing::Not(
          ::testing::Contains(
              ::testing::Pair(EncodingType::ALPRD, ::testing::_))));
  EXPECT_THAT(
      ManualEncodingSelectionPolicyFactory::parseEncodingReadFactors(
          "ALP=1;ALPRD=1;Trivial=1"),
      ::testing::ElementsAre(
          std::pair{EncodingType::ALP, 1.0f},
          std::pair{EncodingType::ALPRD, 1.0f},
          std::pair{EncodingType::Trivial, 1.0f}));
  const std::vector<uint64_t> values{1, 2, 3};
  const auto statistics = Statistics<uint64_t>::create(values);
  ManualEncodingSelectionPolicy<uint64_t> integerPolicy{
      {{EncodingType::ALPRD, 1}}, std::nullopt, std::nullopt};
  EXPECT_FALSE(
      detail::EncodingSizeEstimation<uint64_t>::estimateSize(
          integerPolicy, EncodingType::ALPRD, values, statistics, {}));
  ManualEncodingSelectionPolicy<double> floatingPointPolicy{
      {{EncodingType::ALPRD, 1}}, std::nullopt, std::nullopt};
  EXPECT_FALSE(
      detail::EncodingSizeEstimation<double>::estimateSize(
          floatingPointPolicy,
          EncodingType::ALPRD,
          std::span<const uint64_t>{},
          Statistics<uint64_t>::create({}),
          {}));
}

TEST(ALPRDSelectionConfigurationTest, inheritsAndFiltersCandidates) {
  ManualEncodingSelectionPolicy<double> policy(
      candidates(), std::nullopt, std::nullopt);
  auto alpChild = policy.create<double>(
      EncodingType::ALP, EncodingIdentifiers::ALP::ExceptionValues);
  const auto& alpCandidates =
      static_cast<ManualEncodingSelectionPolicy<double>&>(*alpChild)
          .candidateEncodingReadFactors();
  EXPECT_THAT(
      alpCandidates,
      ::testing::Contains(::testing::Pair(EncodingType::ALPRD, 1)));
  EXPECT_THAT(
      alpCandidates,
      ::testing::Not(
          ::testing::Contains(
              ::testing::Pair(EncodingType::ALP, ::testing::_))));
  auto alprdChild = policy.create<uint64_t>(
      EncodingType::ALPRD, EncodingIdentifiers::ALPRD::RightParts);
  const auto& alprdCandidates =
      static_cast<ManualEncodingSelectionPolicy<uint64_t>&>(*alprdChild)
          .candidateEncodingReadFactors();
  EXPECT_THAT(
      alprdCandidates,
      ::testing::Contains(::testing::Pair(EncodingType::ALP, 1)));
  EXPECT_THAT(
      alprdCandidates,
      ::testing::Not(
          ::testing::Contains(
              ::testing::Pair(EncodingType::ALPRD, ::testing::_))));
}

} // namespace
} // namespace facebook::nimble
