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
#include <bit>
#include <random>

#include "velox/dwio/nimble/encodings/ALPRDEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/tests/RandomEncodingSelectionPolicy.h"

namespace facebook::nimble {
namespace {

using ReadFactors = std::vector<std::pair<EncodingType, float>>;

// Counts child policy creation to keep estimation separate from child writing.
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
    for (auto useLegacy : {false, true}) {
      SCOPED_TRACE(useLegacy);
      // Legacy containers use fixed prefixes. The shared ALPRD decoder accepts
      // both formats; compact-prefix container trees use the native factory.
      if (useLegacy && Config::useVarint &&
          EncodingPrefix::encodingType(encoded) != EncodingType::ALPRD) {
        continue;
      }
      auto decoder = useLegacy
          ? legacy::EncodingFactory(options_).create(*pool_, encoded, nullptr)
          : EncodingFactory(options_).create(*pool_, encoded, nullptr);
      ScopedVector<PhysicalType> actual{
          values.size(), pool_.get(), options_.bufferPool};
      std::fill(actual.begin(), actual.end(), 0);
      decoder->materialize(values.size(), actual.data());
      EXPECT_THAT(
          (std::span<const PhysicalType>{actual}),
          ::testing::ElementsAreArray(values));
    }
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

TYPED_TEST(ALPRDSelectionTest, estimationDoesNotCreateChildPolicies) {
  using T = typename TestFixture::T;
  using PhysicalType = typename TestFixture::PhysicalType;
  const auto values = this->makeValues(512, 2);
  for (auto encodingType : {EncodingType::ALP, EncodingType::ALPRD}) {
    SCOPED_TRACE(encodingType);
    CountingManualPolicy<T> policy{encodingType};
    const auto selected = policy.select(
        values, Statistics<PhysicalType>::create(values), this->options_);
    ASSERT_EQ(selected.encodingType, encodingType);
    ASSERT_TRUE(selected.estimatedSize);
    EXPECT_EQ(policy.numNestedPolicies, 0);
  }
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

  auto penalized = candidates();
  penalized.back().second = 10;
  EXPECT_NE(
      this->select(values, penalized, std::nullopt).encodingType,
      EncodingType::ALPRD);
  penalized.pop_back();
  EXPECT_NE(
      this->select(values, penalized, std::nullopt).encodingType,
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

TYPED_TEST(ALPRDSelectionTest, estimatesIndependentlyOfWriterCandidates) {
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
  EXPECT_EQ(*packed.estimatedSize, *trivial.estimatedSize);
  std::optional<ALPRDEncodingBase::Parameters> trained;
  for (auto child : {EncodingType::FixedBitWidth, EncodingType::Trivial}) {
    const auto encoded = this->encode(
        values,
        this->policy({{EncodingType::ALPRD, 1}}, ReadFactors{{child, 1}}));
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
  for (auto allowNestedAlpSelection : {false, true}) {
    SCOPED_TRACE(allowNestedAlpSelection);
    this->options_.allowNestedAlpSelection = allowNestedAlpSelection;
    for (const auto& [parent, nestedIdentifier] : valueChildren) {
      SCOPED_TRACE(toString(parent));
      for (const auto& nestedCandidates :
           {ReadFactors{},
            ReadFactors{{EncodingType::ALP, 1}},
            ReadFactors{{EncodingType::ALPRD, 1}},
            ReadFactors{{EncodingType::ALP, 1}, {EncodingType::ALPRD, 1}}}) {
        auto parentPolicy = this->policy({{parent, 1}}, nestedCandidates);
        auto nestedPolicy =
            parentPolicy->template create<T>(parent, nestedIdentifier);
        auto& typedPolicy =
            static_cast<ManualEncodingSelectionPolicy<T>&>(*nestedPolicy);
        const auto selected =
            typedPolicy.select(values, statistics, this->options_);
        if (nestedCandidates.empty()) {
          // The legacy option must not turn an empty candidate list into ALP.
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
  EXPECT_FALSE(childPolicy->hasAlpOrAlprdCandidates());
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
  EXPECT_FALSE(
      detail::EncodingSizeEstimation<uint64_t>::estimateSize(
          EncodingType::ALPRD, values, statistics, {}));
  EXPECT_FALSE(
      detail::EncodingSizeEstimation<double>::estimateSize(
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
