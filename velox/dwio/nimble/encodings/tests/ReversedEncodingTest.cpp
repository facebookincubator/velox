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

#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/EncodingSliceFactory.h"
#include "velox/dwio/nimble/encodings/ReversedEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

using namespace facebook;

namespace {

using ReadFactors = std::vector<std::pair<nimble::EncodingType, float>>;

// Returns 'numRuns' strictly descending runs of 'minLength' to 'maxLength'
// rows, each starting near the top of T or at its middle, such as list values
// sorted in descending order.
template <typename T>
std::vector<T> makeDescendingRuns(
    uint32_t numRuns,
    uint32_t minLength,
    uint32_t maxLength,
    uint64_t seed) {
  std::mt19937_64 random{seed};
  std::vector<T> values;
  for (uint32_t run = 0; run < numRuns; ++run) {
    const auto length = minLength + random() % (maxLength - minLength + 1);
    auto value = static_cast<T>(
        std::numeric_limits<T>::max() - static_cast<T>(random() % 100));
    if (random() % 2 == 0) {
      value = static_cast<T>(std::numeric_limits<T>::max() / 2);
    }
    for (uint32_t row = 0; row < length; ++row) {
      values.push_back(value);
      value = static_cast<T>(value - 1 - static_cast<T>(random() % 2));
    }
  }
  return values;
}

} // namespace

class ReversedEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  nimble::Vector<T> toVector(const std::vector<T>& input) {
    return nimble::Vector<T>{pool_.get(), input.begin(), input.end()};
  }

  // Encodes 'input' with a policy that may pick from 'readFactors'.
  template <typename T>
  std::string_view encodeWithPolicy(
      const std::vector<T>& input,
      const ReadFactors& readFactors,
      const nimble::Encoding::Options& options = {}) {
    auto policy = std::make_unique<nimble::ManualEncodingSelectionPolicy<T>>(
        readFactors,
        /*compressionOptions=*/std::nullopt,
        /*identifier=*/std::nullopt);
    return nimble::EncodingFactory::encode<T>(
        std::move(policy), std::span<const T>{input}, *buffer_, options);
  }

  // Checks materialize, skip, reset and the view of 'encoded' against 'input'.
  template <typename T>
  void checkReads(
      std::string_view encoded,
      const std::vector<T>& input,
      const nimble::Encoding::Options& options = {}) {
    using PhysicalType = typename nimble::TypeTraits<T>::physicalType;
    const auto numRows = static_cast<uint32_t>(input.size());
    auto encoding = nimble::EncodingFactory{options}.create(
        *pool_, encoded, /*stringBufferFactory=*/nullptr);
    ASSERT_EQ(encoding->encodingType(), nimble::EncodingType::Reversed);
    ASSERT_EQ(encoding->rowCount(), numRows);

    std::vector<T> output(numRows);
    encoding->materialize(numRows, output.data());
    ASSERT_EQ(output, input);

    encoding->reset();
    const uint32_t numSkipped = numRows / 3;
    encoding->skip(numSkipped);
    std::vector<T> rest(numRows - numSkipped);
    encoding->materialize(numRows - numSkipped, rest.data());
    const std::vector<T> expectedRest{input.begin() + numSkipped, input.end()};
    ASSERT_EQ(rest, expectedRest);

    auto view = nimble::createEncodingView(encoded, pool_.get(), options);
    const auto& typedView =
        dynamic_cast<const nimble::TypedEncodingView<T>&>(*view);
    std::vector<T> viewRange(numRows - numSkipped);
    typedView.read(
        numSkipped,
        numRows - numSkipped,
        reinterpret_cast<PhysicalType*>(viewRange.data()));
    ASSERT_EQ(viewRange, expectedRest);
    const std::vector<uint32_t> indices{numRows - 1, 0, numRows / 2, 0};
    std::vector<T> viewValues(indices.size());
    typedView.readAt(indices, viewValues.data());
    for (size_t i = 0; i < indices.size(); ++i) {
      EXPECT_EQ(viewValues[i], input[indices[i]]) << "at index " << i;
    }
  }

  template <typename T>
  void testRoundTrip() {
    SCOPED_TRACE(fmt::format("{}", nimble::TypeTraits<T>::dataType));
    const auto input = makeDescendingRuns<T>(
        /*numRuns=*/20, /*minLength=*/2, /*maxLength=*/40, /*seed=*/1);
    for (const bool useVarint : {false, true}) {
      SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
      const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
      const auto encoded =
          nimble::test::Encoder<nimble::ReversedEncoding<T>>::encode(
              *buffer_,
              toVector(input),
              nimble::CompressionType::Uncompressed,
              options);
      checkReads(encoded, input, options);
    }
  }

  template <typename T>
  void testExtremeValues() {
    SCOPED_TRACE(fmt::format("{}", nimble::TypeTraits<T>::dataType));
    std::vector<T> input{std::numeric_limits<T>::max(), 1, 0};
    if constexpr (std::numeric_limits<T>::is_signed) {
      input.push_back(-1);
      input.push_back(std::numeric_limits<T>::min());
    }
    const auto numRows = static_cast<uint32_t>(input.size());
    for (const bool useVarint : {false, true}) {
      SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
      const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
      const auto encoded =
          nimble::test::Encoder<nimble::ReversedEncoding<T>>::encode(
              *buffer_,
              toVector(input),
              nimble::CompressionType::Uncompressed,
              options);
      checkReads(encoded, input, options);
      for (uint32_t offset = 0; offset < numRows; ++offset) {
        for (uint32_t length = 1; offset + length <= numRows; ++length) {
          SCOPED_TRACE(fmt::format("offset={}, length={}", offset, length));
          nimble::Buffer sliceBuffer{*pool_};
          const auto sliced = nimble::EncodingSliceFactory::slice(
              encoded, offset, length, sliceBuffer, options);
          checkReads(
              sliced,
              std::vector<T>(
                  input.begin() + offset, input.begin() + offset + length),
              options);
        }
      }
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

TEST_F(ReversedEncodingTest, roundTripAllTypes) {
  testRoundTrip<int8_t>();
  testRoundTrip<uint8_t>();
  testRoundTrip<int16_t>();
  testRoundTrip<uint16_t>();
  testRoundTrip<int32_t>();
  testRoundTrip<uint32_t>();
  testRoundTrip<int64_t>();
  testRoundTrip<uint64_t>();
}

TEST_F(ReversedEncodingTest, roundTripExtremeValues) {
  testExtremeValues<int8_t>();
  testExtremeValues<uint8_t>();
  testExtremeValues<int16_t>();
  testExtremeValues<uint16_t>();
  testExtremeValues<int32_t>();
  testExtremeValues<uint32_t>();
  testExtremeValues<int64_t>();
  testExtremeValues<uint64_t>();
}

TEST_F(ReversedEncodingTest, mapsDocumentedExample) {
  using Encoding = nimble::ReversedEncoding<int8_t>;
  static_assert(Encoding::kMask == 0x7F);
  static_assert(nimble::ReversedEncoding<uint32_t>::kMask == 0xFFFFFFFF);
  const std::vector<int8_t> input{5, 1, -1, -3};
  std::vector<uint8_t> stored(input.begin(), input.end());
  Encoding::flip(stored.data(), stored.size());
  EXPECT_EQ(stored, (std::vector<uint8_t>{0x7A, 0x7E, 0x80, 0x82}));
  // The mapping is its own inverse.
  Encoding::flip(stored.data(), stored.size());
  EXPECT_EQ(std::vector<int8_t>(stored.begin(), stored.end()), input);
}

TEST_F(ReversedEncodingTest, selectsEliasFanoAcrossZero) {
  // A signed stream that descends across zero ascends in the unsigned order of
  // the nested stream, so EliasFano applies to all of it.
  std::vector<int32_t> input(1'000);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = 500 - static_cast<int32_t>(i);
  }
  const auto encoded = encodeWithPolicy<int32_t>(
      input,
      {{nimble::EncodingType::FixedBitWidth, 1.0},
       {nimble::EncodingType::EliasFano, 1.0},
       {nimble::EncodingType::Reversed, 1.0}});
  ASSERT_EQ(
      nimble::EncodingPrefix::encodingType(encoded),
      nimble::EncodingType::Reversed);
  const auto child = encoded.substr(
      nimble::EncodingPrefix::prefixSize(encoded, /*useVarint=*/false));
  EXPECT_EQ(
      nimble::EncodingPrefix::encodingType(child),
      nimble::EncodingType::EliasFano);
  EXPECT_EQ(nimble::EncodingPrefix::dataType(child), nimble::DataType::Uint32);
  checkReads(encoded, input);
}

TEST_F(ReversedEncodingTest, selectsPartitionedEliasFanoForDescendingRuns) {
  const auto input = makeDescendingRuns<uint32_t>(
      /*numRuns=*/200, /*minLength=*/64, /*maxLength=*/200, /*seed=*/3);
  const auto encoded = encodeWithPolicy<uint32_t>(
      input,
      {{nimble::EncodingType::FixedBitWidth, 1.0},
       {nimble::EncodingType::PartitionedEliasFano, 1.0},
       {nimble::EncodingType::Reversed, 1.0}});
  ASSERT_EQ(
      nimble::EncodingPrefix::encodingType(encoded),
      nimble::EncodingType::Reversed);
  const auto child = encoded.substr(
      nimble::EncodingPrefix::prefixSize(encoded, /*useVarint=*/false));
  EXPECT_EQ(
      nimble::EncodingPrefix::encodingType(child),
      nimble::EncodingType::PartitionedEliasFano);
  checkReads(encoded, input);
}

TEST_F(ReversedEncodingTest, estimatesOnlyOrderedNestedEncodings) {
  // Descending lists of small positive values, each ending in a -1 sentinel.
  // The stored values lie in one narrow range, but Reversed estimates only
  // the ordered encodings, so FixedBitWidth stores the values themselves.
  std::vector<int32_t> sentinels;
  for (int32_t list = 0; list < 100; ++list) {
    for (int32_t value = 1'000; value > 0; value -= 10) {
      sentinels.push_back(value);
    }
    sentinels.push_back(-1);
  }
  const auto encoded = encodeWithPolicy<int32_t>(
      sentinels,
      {{nimble::EncodingType::FixedBitWidth, 1.0},
       {nimble::EncodingType::Reversed, 1.0}});
  EXPECT_EQ(
      nimble::EncodingPrefix::encodingType(encoded),
      nimble::EncodingType::FixedBitWidth);
}

TEST_F(ReversedEncodingTest, estimatesOnlyMostlyDescendingValues) {
  const auto estimate = [](const std::vector<uint32_t>& input,
                           const ReadFactors& readFactors) {
    return nimble::ReversedEncoding<uint32_t>::estimateSize(
        input,
        nimble::Statistics<uint32_t>::create(input),
        /*options=*/{},
        readFactors);
  };
  const ReadFactors eliasFano{{nimble::EncodingType::EliasFano, 1.0}};

  std::vector<uint32_t> descending(1'000);
  std::vector<uint32_t> ascending(1'000);
  for (uint32_t i = 0; i < descending.size(); ++i) {
    descending[i] = 5'000 - i * 3;
    ascending[i] = i * 3;
  }
  EXPECT_FALSE(estimate({}, eliasFano).has_value());
  EXPECT_FALSE(estimate(ascending, eliasFano).has_value());
  EXPECT_FALSE(
      estimate(std::vector<uint32_t>(1'000, 7), eliasFano).has_value());
  EXPECT_FALSE(
      estimate(descending, {{nimble::EncodingType::FixedBitWidth, 1.0}})
          .has_value());

  // The estimate is the prefix plus the nested estimate of the stored values.
  std::vector<uint32_t> stored{descending};
  nimble::ReversedEncoding<uint32_t>::flip(stored.data(), stored.size());
  const std::span<const uint32_t> storedValues{stored};
  const auto nestedSize = nimble::EliasFanoEncoding<uint32_t>::estimateSize(
      storedValues, nimble::Statistics<uint32_t>::create(storedValues));
  ASSERT_TRUE(nestedSize.has_value());
  EXPECT_EQ(
      estimate(descending, eliasFano),
      nimble::EncodingPrefix::serializedSize(
          static_cast<uint32_t>(descending.size()),
          /*useVarint=*/false) +
          *nestedSize);

  // Descending runs mostly descend, though the stream as a whole does not.
  const auto runs = makeDescendingRuns<uint32_t>(
      /*numRuns=*/200, /*minLength=*/64, /*maxLength=*/200, /*seed=*/4);
  const ReadFactors partitionedEliasFano{
      {nimble::EncodingType::PartitionedEliasFano, 1.0}};
  EXPECT_TRUE(estimate(runs, partitionedEliasFano).has_value());

  // 'reversedMinDescendingRatio' raises how much more often the values must
  // descend than ascend.
  const auto estimateWithRatio = [&](double minDescendingRatio) {
    return nimble::ReversedEncoding<uint32_t>::estimateSize(
        runs,
        nimble::Statistics<uint32_t>::create(runs),
        {.reversedMinDescendingRatio = minDescendingRatio},
        partitionedEliasFano);
  };
  const auto order = nimble::Statistics<uint32_t>::create(runs)
                         .template adjacentOrder<uint32_t>();
  ASSERT_GT(order.numAscending, 0);
  const auto ratio = static_cast<double>(order.numDescending) /
      static_cast<double>(order.numAscending);
  EXPECT_TRUE(estimateWithRatio(ratio * 0.99).has_value());
  EXPECT_FALSE(estimateWithRatio(ratio * 1.01).has_value());
}

TEST_F(ReversedEncodingTest, slicesEveryRange) {
  const auto input = makeDescendingRuns<int64_t>(
      /*numRuns=*/4, /*minLength=*/5, /*maxLength=*/10, /*seed=*/5);
  const auto numRows = static_cast<uint32_t>(input.size());
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    const auto encoded =
        nimble::test::Encoder<nimble::ReversedEncoding<int64_t>>::encode(
            *buffer_,
            toVector(input),
            nimble::CompressionType::Uncompressed,
            options);
    for (uint32_t offset = 0; offset < numRows; ++offset) {
      for (uint32_t length = 1; offset + length <= numRows; ++length) {
        SCOPED_TRACE(fmt::format("offset={}, length={}", offset, length));
        nimble::Buffer sliceBuffer{*pool_};
        const auto sliced = nimble::EncodingSliceFactory::slice(
            encoded, offset, length, sliceBuffer, options);
        ASSERT_EQ(
            nimble::EncodingPrefix::encodingType(sliced),
            nimble::EncodingType::Reversed);
        auto encoding = nimble::EncodingFactory{options}.create(
            *pool_, sliced, /*stringBufferFactory=*/nullptr);
        std::vector<int64_t> output(length);
        encoding->materialize(length, output.data());
        ASSERT_EQ(
            output,
            std::vector<int64_t>(
                input.begin() + offset, input.begin() + offset + length));
      }
    }
  }
}

TEST_F(ReversedEncodingTest, sliceRejectsInvalidRange) {
  const auto encoded =
      nimble::test::Encoder<nimble::ReversedEncoding<uint32_t>>::encode(
          *buffer_, toVector<uint32_t>({9, 7, 5, 3}));
  NIMBLE_ASSERT_THROW(
      nimble::ReversedEncoding<uint32_t>::slice(encoded, 5, 1, *buffer_),
      "Slice starts past the last row.");
  NIMBLE_ASSERT_THROW(
      nimble::ReversedEncoding<uint32_t>::slice(encoded, 2, 3, *buffer_),
      "Slice ends past the last row.");
  NIMBLE_ASSERT_THROW(
      nimble::ReversedEncoding<uint32_t>::slice(encoded, 1, 0, *buffer_),
      "Cannot slice zero rows.");
}

TEST_F(ReversedEncodingTest, rejectsMalformedStreams) {
  // Wraps 'child' in a Reversed prefix of 'dataType' and 'rowCount'.
  const auto wrap =
      [](nimble::DataType dataType, uint32_t rowCount, std::string_view child) {
        std::string data(nimble::EncodingPrefix::kFixedPrefixSize, '\0');
        char* pos = data.data();
        nimble::EncodingPrefix::serialize(
            nimble::EncodingType::Reversed,
            dataType,
            rowCount,
            /*useVarint=*/false,
            pos);
        data.append(child);
        return data;
      };
  const std::string child{
      nimble::test::Encoder<nimble::TrivialEncoding<uint32_t>>::encode(
          *buffer_, toVector<uint32_t>({1, 2, 3}))};
  const auto open = [&](const std::string& data) {
    return nimble::EncodingFactory{}.create(
        *pool_, data, /*stringBufferFactory=*/nullptr);
  };

  // The nested stream of an Int64 stream must hold Uint64 values.
  NIMBLE_ASSERT_FILE_THROW(
      open(wrap(nimble::DataType::Int64, 3, child)),
      "Reversed nested stream has the wrong data type.");
  NIMBLE_ASSERT_FILE_THROW(
      open(wrap(nimble::DataType::Uint32, 4, child)),
      "Reversed nested stream has the wrong row count.");
  NIMBLE_ASSERT_FILE_THROW(
      open(wrap(nimble::DataType::Uint32, 3, /*child=*/"")),
      "Truncated encoding prefix.");
  EXPECT_NO_THROW(open(wrap(nimble::DataType::Int32, 3, child)));

  // Views check the nested stream the same way.
  const auto view = [&](const std::string& data) {
    return nimble::createEncodingView(data, pool_.get(), {});
  };
  NIMBLE_ASSERT_FILE_THROW(
      view(wrap(nimble::DataType::Int64, 3, child)),
      "Reversed nested stream has the wrong data type.");
  NIMBLE_ASSERT_FILE_THROW(
      view(wrap(nimble::DataType::Uint32, 4, child)),
      "Reversed nested stream has the wrong row count.");
  NIMBLE_ASSERT_FILE_THROW(
      view(wrap(nimble::DataType::Uint32, 3, /*child=*/"")),
      "Truncated encoding prefix.");
  EXPECT_NO_THROW(view(wrap(nimble::DataType::Int32, 3, child)));
}
