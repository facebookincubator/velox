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
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleCompare.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

using namespace facebook;

template <typename T>
struct DictionaryTestValues;

template <>
struct DictionaryTestValues<uint32_t> {
  static std::vector<uint32_t> values() {
    return {10, 20, 10, 30, 40, 20, 30};
  }
};

template <>
struct DictionaryTestValues<double> {
  static std::vector<double> values() {
    return {1.25, 2.5, 1.25, -3.75, 8.0, 2.5, -3.75};
  }
};

template <>
struct DictionaryTestValues<std::string_view> {
  static std::vector<std::string_view> values() {
    return {"alpha", "beta", "alpha", "gamma", "delta", "beta", "gamma"};
  }
};

template <typename T>
class DictionaryEncodingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("DictionaryEncodingTest");
    pool_ = rootPool_->addLeafChild("DictionaryEncodingTestLeaf");
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  nimble::Vector<T> makeValues() {
    nimble::Vector<T> values{pool_.get()};
    const auto source = DictionaryTestValues<T>::values();
    for (const auto value : source) {
      values.push_back(value);
    }
    return values;
  }

  // Encodes `input` as a dictionary with `options`, checks that it reads back,
  // and returns the alphabet in stored order.
  template <typename V>
  std::vector<V> encodeAndReadAlphabet(
      const std::vector<V>& input,
      const nimble::Encoding::Options& options) {
    const nimble::Vector<V> values{pool_.get(), input.begin(), input.end()};
    const auto encoded =
        nimble::test::Encoder<nimble::DictionaryEncoding<V>>::encode(
            *buffer_, values, nimble::CompressionType::Uncompressed, options);
    nimble::DictionaryEncoding<V> encoding{
        *pool_, encoded, /*stringBufferFactory=*/nullptr, options};
    std::vector<V> output(input.size());
    encoding.materialize(input.size(), output.data());
    EXPECT_EQ(output, input);
    std::vector<V> alphabet;
    alphabet.reserve(encoding.dictionarySize());
    for (uint32_t i = 0; i < encoding.dictionarySize(); ++i) {
      alphabet.push_back(*static_cast<const V*>(encoding.dictionaryEntry(i)));
    }
    return alphabet;
  }

  // Encodes `input` with the encoding selection over `candidates` and checks
  // that it reads back.
  std::string encodeWithCandidates(
      const std::vector<uint32_t>& input,
      std::vector<std::pair<nimble::EncodingType, float>> candidates,
      bool sortDictionaryAlphabet) {
    auto policy =
        std::make_unique<nimble::ManualEncodingSelectionPolicy<uint32_t>>(
            std::move(candidates),
            /*compressionOptions=*/std::nullopt,
            /*identifier=*/std::nullopt);
    std::string encoded{nimble::EncodingFactory::encode<uint32_t>(
        std::move(policy),
        {input.data(), input.size()},
        *buffer_,
        {.sortDictionaryAlphabet = sortDictionaryAlphabet})};
    auto encoding = nimble::EncodingFactory{}.create(
        *pool_, encoded, /*stringBufferFactory=*/nullptr);
    std::vector<uint32_t> output(input.size());
    encoding->materialize(input.size(), output.data());
    EXPECT_EQ(output, input);
    return encoded;
  }

  void* stringBufferFactory(uint32_t totalLength) {
    auto& stringBuffer = stringBuffers_.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
    return stringBuffer->asMutable<void>();
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
  std::vector<velox::BufferPtr> stringBuffers_;
};

using DictionaryEncodingTypes =
    ::testing::Types<uint32_t, double, std::string_view>;

namespace {

// Returns `numRows` values over `numDistinct` random 32-bit values, each of
// which appears at least once, in random order.
std::vector<uint32_t> randomValues(uint32_t numDistinct, uint32_t numRows) {
  std::mt19937 random{/* seed */ 7};
  std::vector<uint32_t> values(numDistinct);
  for (auto& value : values) {
    value = static_cast<uint32_t>(random());
  }
  while (values.size() < numRows) {
    values.push_back(values[random() % numDistinct]);
  }
  std::shuffle(values.begin(), values.end(), random);
  return values;
}

// Returns the top-level encoding type of `encoded`.
nimble::EncodingType rootEncodingType(std::string_view encoded) {
  return nimble::EncodingLayoutCapture::capture(encoded, {}).encodingType();
}

// Returns the encoding type of the alphabet of `encoded`, which must be a
// dictionary.
nimble::EncodingType alphabetEncodingType(std::string_view encoded) {
  return nimble::EncodingLayoutCapture::capture(encoded, {})
      .child(nimble::EncodingIdentifiers::Dictionary::Alphabet)
      ->encodingType();
}

} // namespace

TYPED_TEST_CASE(DictionaryEncodingTest, DictionaryEncodingTypes);

TYPED_TEST(DictionaryEncodingTest, slicePreservesAlphabetAndMaterializesRange) {
  using DataType = TypeParam;

  struct Range {
    uint32_t offset;
    uint32_t length;
  };

  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    const auto values = this->makeValues();
    const auto encoded =
        nimble::test::Encoder<nimble::DictionaryEncoding<DataType>>::encode(
            *this->buffer_,
            values,
            nimble::CompressionType::Uncompressed,
            options);

    nimble::DictionaryEncoding<DataType> source{
        *this->pool_,
        encoded,
        [this](uint32_t totalLength) {
          return this->stringBufferFactory(totalLength);
        },
        options};
    const auto sourceDictionarySize = source.dictionarySize();

    for (const auto range :
         {Range{/*offset=*/0, /*length=*/1},
          Range{/*offset=*/1, /*length=*/3},
          Range{/*offset=*/2, /*length=*/4},
          Range{/*offset=*/6, /*length=*/1}}) {
      SCOPED_TRACE(
          testing::Message()
          << "offset=" << range.offset << ", length=" << range.length);
      nimble::Buffer sliceBuffer{*this->pool_};
      const auto sliced = nimble::DictionaryEncoding<DataType>::slice(
          encoded, range.offset, range.length, sliceBuffer, options);
      nimble::DictionaryEncoding<DataType> encoding{
          *this->pool_,
          sliced,
          [this](uint32_t totalLength) {
            return this->stringBufferFactory(totalLength);
          },
          options};

      EXPECT_EQ(encoding.encodingType(), nimble::EncodingType::Dictionary);
      EXPECT_EQ(encoding.dataType(), nimble::TypeTraits<DataType>::dataType);
      EXPECT_EQ(encoding.rowCount(), range.length);
      EXPECT_EQ(encoding.dictionarySize(), sourceDictionarySize);

      nimble::Vector<DataType> result{this->pool_.get(), range.length};
      encoding.materialize(range.length, result.data());
      for (uint32_t i = 0; i < range.length; ++i) {
        EXPECT_TRUE(
            nimble::NimbleCompare<DataType>::equals(
                result[i], values[range.offset + i]));
      }
    }
  }
}

TYPED_TEST(DictionaryEncodingTest, sortDictionaryAlphabet) {
  using DataType = TypeParam;
  if constexpr (std::is_same_v<DataType, std::string_view>) {
    GTEST_SKIP() << "String alphabets are not sorted.";
  } else {
    // First occurrence orders the alphabet 40, 10, 30, 20 (or 8.0, -3.75,
    // 2.5, 1.25), so a sorted alphabet differs from it.
    std::vector<DataType> input;
    if constexpr (std::is_integral_v<DataType>) {
      input = {40, 10, 30, 10, 20, 40, 30};
    } else {
      input = {8.0, -3.75, 2.5, -3.75, 1.25, 8.0, 2.5};
    }
    const auto firstOccurrence =
        this->encodeAndReadAlphabet(input, {.sortDictionaryAlphabet = false});
    const auto sorted =
        this->encodeAndReadAlphabet(input, {.sortDictionaryAlphabet = true});
    if constexpr (std::is_integral_v<DataType>) {
      EXPECT_THAT(firstOccurrence, testing::ElementsAre(40, 10, 30, 20));
      EXPECT_THAT(sorted, testing::ElementsAre(10, 20, 30, 40));
    } else {
      // Floating-point alphabets keep the first-occurrence order.
      EXPECT_EQ(sorted, firstOccurrence);
    }
  }
}

TYPED_TEST(DictionaryEncodingTest, sortDictionaryAlphabetSignedAndWide) {
  using DataType = TypeParam;
  if constexpr (!std::is_same_v<DataType, uint32_t>) {
    GTEST_SKIP() << "Covers other integer types, once.";
  } else {
    const nimble::Encoding::Options options{.sortDictionaryAlphabet = true};

    // The alphabet sorts by physical value, which stores a negative value as
    // a large unsigned one, so negative values follow the others.
    {
      const std::vector<int32_t> input{5, -3, 7, -3, 0, 5};
      EXPECT_THAT(
          this->encodeAndReadAlphabet(input, options),
          testing::ElementsAre(0, 5, 7, -3));
    }
    {
      constexpr int64_t kLarge{int64_t{1} << 40};
      const std::vector<int64_t> input{kLarge, -2, 3, -2, kLarge};
      EXPECT_THAT(
          this->encodeAndReadAlphabet(input, options),
          testing::ElementsAre(3, kLarge, -2));
    }
    {
      constexpr uint64_t kMax{std::numeric_limits<uint64_t>::max()};
      constexpr uint64_t kHighBit{uint64_t{1} << 63};
      const std::vector<uint64_t> input{kMax, 1, kHighBit, 1};
      EXPECT_THAT(
          this->encodeAndReadAlphabet(input, options),
          testing::ElementsAre(1, kHighBit, kMax));
    }
  }
}

TYPED_TEST(DictionaryEncodingTest, sortDictionaryAlphabetEdgeCases) {
  using DataType = TypeParam;
  if constexpr (!std::is_same_v<DataType, uint32_t>) {
    GTEST_SKIP() << "Covers integer alphabets.";
  } else {
    const nimble::Encoding::Options options{.sortDictionaryAlphabet = true};

    // A single-entry alphabet has nothing to sort.
    {
      const std::vector<uint32_t> input{7, 7, 7};
      EXPECT_THAT(
          this->encodeAndReadAlphabet(input, options), testing::ElementsAre(7));
    }

    // Every value distinct, as in an id column.
    {
      const std::vector<uint32_t> input{30, 10, 40, 20};
      EXPECT_THAT(
          this->encodeAndReadAlphabet(input, options),
          testing::ElementsAre(10, 20, 30, 40));
    }
  }
}

TYPED_TEST(DictionaryEncodingTest, sortedAlphabetSelectsAscendingEncoding) {
  using DataType = TypeParam;
  if constexpr (!std::is_same_v<DataType, uint32_t>) {
    GTEST_SKIP() << "Covers integer alphabets.";
  } else {
    // Wide values with repeats.
    const auto input = randomValues(/*numDistinct=*/1'000, /*numRows=*/4'000);
    const std::vector<std::pair<nimble::EncodingType, float>> candidates{
        {nimble::EncodingType::Trivial, 1.0},
        {nimble::EncodingType::FixedBitWidth, 1.0},
        {nimble::EncodingType::Dictionary, 1.0},
        {nimble::EncodingType::DeltaBlock, 1.0},
        {nimble::EncodingType::EliasFano, 1.0},
    };
    const auto firstOccurrence = this->encodeWithCandidates(
        input, candidates, /*sortDictionaryAlphabet=*/false);
    const auto sorted = this->encodeWithCandidates(
        input, candidates, /*sortDictionaryAlphabet=*/true);
    ASSERT_EQ(
        rootEncodingType(firstOccurrence), nimble::EncodingType::Dictionary);
    ASSERT_EQ(rootEncodingType(sorted), nimble::EncodingType::Dictionary);
    EXPECT_THAT(
        alphabetEncodingType(firstOccurrence),
        testing::AnyOf(
            nimble::EncodingType::Trivial,
            nimble::EncodingType::FixedBitWidth));
    EXPECT_THAT(
        alphabetEncodingType(sorted),
        testing::AnyOf(
            nimble::EncodingType::DeltaBlock, nimble::EncodingType::EliasFano));
    EXPECT_LT(sorted.size(), firstOccurrence.size());
  }
}

TYPED_TEST(
    DictionaryEncodingTest,
    sortedAlphabetEstimateNeedsEliasFanoCandidate) {
  using DataType = TypeParam;
  if constexpr (!std::is_same_v<DataType, uint32_t>) {
    GTEST_SKIP() << "Covers integer alphabets.";
  } else {
    // 24,000 distinct values over 40,000 rows, with 16-bit indices. Unsorted,
    // the alphabet takes 32 bits per entry and Dictionary is larger than
    // FixedBitWidth. Sorted, EliasFano stores the alphabet in about 20 bits per
    // entry, so only sorting makes Dictionary the smallest.
    const auto input = randomValues(/*numDistinct=*/24'000, /*numRows=*/40'000);
    using Candidates = std::vector<std::pair<nimble::EncodingType, float>>;
    const Candidates baseCandidates{
        {nimble::EncodingType::Trivial, 1.0},
        {nimble::EncodingType::FixedBitWidth, 1.0},
        {nimble::EncodingType::Dictionary, 1.0},
    };
    // Encodes `input` over the base candidates plus `extraCandidate`.
    const auto encode = [&](std::optional<nimble::EncodingType> extraCandidate,
                            bool sortDictionaryAlphabet) {
      Candidates candidates{baseCandidates};
      if (extraCandidate.has_value()) {
        candidates.emplace_back(*extraCandidate, 1.0);
      }
      return this->encodeWithCandidates(
          input, std::move(candidates), sortDictionaryAlphabet);
    };

    // Require EliasFano before using its size estimate. DeltaBlock needs the
    // actual sorted values to estimate size; PartitionedEliasFano declines
    // a single ascending run.
    for (const auto extraCandidate :
         {std::optional<nimble::EncodingType>{},
          std::optional{nimble::EncodingType::DeltaBlock},
          std::optional{nimble::EncodingType::PartitionedEliasFano}}) {
      SCOPED_TRACE(
          fmt::format(
              "extraCandidate={}",
              extraCandidate.has_value() ? toString(*extraCandidate) : "none"));
      const auto firstOccurrence =
          encode(extraCandidate, /*sortDictionaryAlphabet=*/false);
      const auto sorted =
          encode(extraCandidate, /*sortDictionaryAlphabet=*/true);
      EXPECT_NE(
          rootEncodingType(firstOccurrence), nimble::EncodingType::Dictionary);
      EXPECT_EQ(rootEncodingType(sorted), rootEncodingType(firstOccurrence));
      EXPECT_EQ(sorted.size(), firstOccurrence.size());
    }

    // With EliasFano, sorting picks Dictionary.
    {
      const auto firstOccurrence = encode(
          nimble::EncodingType::EliasFano, /*sortDictionaryAlphabet=*/false);
      const auto sorted = encode(
          nimble::EncodingType::EliasFano, /*sortDictionaryAlphabet=*/true);
      EXPECT_NE(
          rootEncodingType(firstOccurrence), nimble::EncodingType::Dictionary);
      ASSERT_EQ(rootEncodingType(sorted), nimble::EncodingType::Dictionary);
      EXPECT_EQ(alphabetEncodingType(sorted), nimble::EncodingType::EliasFano);
      EXPECT_LT(sorted.size(), firstOccurrence.size());
    }
  }
}

TYPED_TEST(DictionaryEncodingTest, invalidSliceRange) {
  using DataType = TypeParam;

  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    const auto values = this->makeValues();
    const auto encoded =
        nimble::test::Encoder<nimble::DictionaryEncoding<DataType>>::encode(
            *this->buffer_,
            values,
            nimble::CompressionType::Uncompressed,
            options);

    nimble::Buffer sliceBuffer{*this->pool_};
    NIMBLE_ASSERT_THROW(
        nimble::DictionaryEncoding<DataType>::slice(
            encoded,
            /*offset=*/0,
            /*length=*/0,
            sliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        nimble::DictionaryEncoding<DataType>::slice(
            encoded,
            /*offset=*/values.size() + 1,
            /*length=*/1,
            sliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        nimble::DictionaryEncoding<DataType>::slice(
            encoded,
            /*offset=*/values.size() - 1,
            /*length=*/2,
            sliceBuffer,
            options),
        "");
  }
}
