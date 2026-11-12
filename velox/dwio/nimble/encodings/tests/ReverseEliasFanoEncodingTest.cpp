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
#include "velox/dwio/nimble/encodings/ReverseEliasFanoEncoding.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

namespace {

// Supplies the nulls of a read range to IntegralReadWithVisitor. The streams
// here have no nulls.
class TestReader {
 public:
  const uint64_t* rawNullsInReadRange() const {
    return nullptr;
  }

  bool returnReaderNulls() const {
    return false;
  }
};

// Collects the values of the selected rows that readWithVisitor() decodes.
template <typename T>
class IntegralReadWithVisitor {
 public:
  using DataType = T;
  using Extract = std::nullptr_t;

  explicit IntegralReadWithVisitor(std::vector<velox::vector_size_t> rows)
      : rows_{std::move(rows)} {}

  TestReader& reader() {
    return reader_;
  }

  velox::vector_size_t currentRow() const {
    return rows_[rowIndex_];
  }

  void process(T value, bool& atEnd) {
    values_.push_back(value);
    addRowIndex(1);
    atEnd = this->atEnd();
  }

  void processNull(bool& atEnd) {
    addRowIndex(1);
    atEnd = this->atEnd();
  }

  bool allowNulls() const {
    return false;
  }

  void addRowIndex(velox::vector_size_t count) {
    rowIndex_ += count;
  }

  void addNumValues(velox::vector_size_t /*count*/) {}

  bool atEnd() const {
    return rowIndex_ >= static_cast<velox::vector_size_t>(rows_.size());
  }

  const std::vector<T>& values() const {
    return values_;
  }

 private:
  TestReader reader_;
  std::vector<velox::vector_size_t> rows_;
  velox::vector_size_t rowIndex_{0};
  std::vector<T> values_;
};

class ReverseEliasFanoEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  nimble::Vector<T> toVector(const std::vector<T>& input) {
    nimble::Vector<T> values{pool_.get()};
    values.insert(values.end(), input.data(), input.data() + input.size());
    return values;
  }

  template <typename T>
  std::string_view encode(
      const std::vector<T>& input,
      const nimble::Encoding::Options& options = {}) {
    return nimble::test::Encoder<nimble::ReverseEliasFanoEncoding<T>>::encode(
        *buffer_,
        toVector(input),
        nimble::CompressionType::Uncompressed,
        options);
  }

  template <typename T>
  std::unique_ptr<nimble::Encoding> createEncoding(
      const std::vector<T>& input) {
    return nimble::test::Encoder<nimble::ReverseEliasFanoEncoding<T>>::
        createEncoding(*buffer_, toVector(input), nullptr);
  }

  // Checks sequential, absolute and single-row reads of 'input' after a round
  // trip.
  template <typename T>
  void roundTrip(const std::vector<T>& input) {
    using PhysicalType = typename nimble::TypeTraits<T>::physicalType;
    auto encoding = createEncoding(input);
    const auto rowCount = static_cast<uint32_t>(input.size());
    EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::ReverseEliasFano);
    EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<T>::dataType);
    EXPECT_EQ(encoding->rowCount(), rowCount);

    std::vector<T> output(input.size());
    encoding->materialize(rowCount, output.data());
    EXPECT_EQ(output, input);

    const auto& reverseEliasFano =
        dynamic_cast<const nimble::ReverseEliasFanoEncoding<T>&>(*encoding);
    std::vector<T> absoluteOutput(input.size());
    reverseEliasFano.materializeAt(
        /*offset=*/0,
        rowCount,
        reinterpret_cast<PhysicalType*>(absoluteOutput.data()));
    EXPECT_EQ(absoluteOutput, input);
    for (uint32_t row = 0; row < rowCount; ++row) {
      EXPECT_EQ(static_cast<T>(reverseEliasFano.valueAt(row)), input[row])
          << "row=" << row;
    }
  }

  // Checks that slicing 'input' at each of 'ranges' keeps the sliced rows, with
  // fixed and varint row counts.
  template <typename T>
  void expectSlices(
      const std::vector<T>& input,
      const std::vector<std::pair<uint32_t, uint32_t>>& ranges) {
    for (const bool useVarintRowCount : {false, true}) {
      const nimble::Encoding::Options options{
          .useVarintRowCount = useVarintRowCount};
      const auto encoded = encode(input, options);
      for (const auto& [offset, length] : ranges) {
        SCOPED_TRACE(
            fmt::format(
                "useVarintRowCount={}, offset={}, length={}",
                useVarintRowCount,
                offset,
                length));
        nimble::Buffer sliceBuffer{*pool_};
        const auto sliced = nimble::ReverseEliasFanoEncoding<T>::slice(
            encoded, offset, length, sliceBuffer, options);
        auto encoding = nimble::EncodingFactory{options}.create(
            *pool_, sliced, /*stringBufferFactory=*/nullptr);
        EXPECT_EQ(
            encoding->encodingType(), nimble::EncodingType::ReverseEliasFano);
        EXPECT_EQ(encoding->rowCount(), length);
        std::vector<T> output(length);
        encoding->materialize(length, output.data());
        EXPECT_EQ(
            output,
            std::vector<T>(
                input.begin() + offset, input.begin() + offset + length));
      }
    }
  }

  // Round trips and slices random non-increasing values of T: values spread
  // over the whole type, and dense values with many repeats.
  template <typename T>
  void roundTripRandom(uint32_t seed) {
    std::mt19937_64 random{seed};
    for (const uint32_t rowCount : {1, 2, 127, 128, 129, 1'000}) {
      for (const bool dense : {false, true}) {
        SCOPED_TRACE(
            ::testing::Message()
            << "rowCount=" << rowCount << ", dense=" << dense);
        std::vector<T> input(rowCount);
        for (auto& value : input) {
          value = dense ? static_cast<T>(random() % (rowCount / 2 + 1))
                        : static_cast<T>(random());
        }
        std::sort(input.begin(), input.end(), std::greater<>{});
        roundTrip(input);

        std::uniform_int_distribution<uint32_t> offsetDistribution{
            0, rowCount - 1};
        std::vector<std::pair<uint32_t, uint32_t>> ranges{{0, rowCount}};
        for (int i = 0; i < 4; ++i) {
          const auto offset = offsetDistribution(random);
          std::uniform_int_distribution<uint32_t> lengthDistribution{
              1, rowCount - offset};
          ranges.emplace_back(offset, lengthDistribution(random));
        }
        expectSlices(input, ranges);
      }
    }
  }

  // Encodes 'input' through the encoding factory with a policy that offers
  // 'candidates'.
  template <typename T>
  std::string_view encodeWithCandidates(
      const std::vector<T>& input,
      std::vector<std::pair<nimble::EncodingType, float>> candidates) {
    auto policy = std::make_unique<nimble::ManualEncodingSelectionPolicy<T>>(
        std::move(candidates), std::nullopt, std::nullopt);
    return nimble::EncodingFactory::encode<T>(
        std::move(policy), input, *buffer_);
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

TEST_F(ReverseEliasFanoEncodingTest, roundTripUnsignedValues) {
  roundTrip<uint32_t>({100, 20, 15, 13, 12, 9, 7, 3, 2, 1});
  roundTrip<uint64_t>({4'000, 2'000, 2'000, 1'000, 9, 5, 5, 0, 0, 0});
  roundTrip<uint16_t>({30, 12, 9, 4, 4, 4, 2});
  roundTrip<uint8_t>({8, 3, 2, 1, 1, 0});
}

TEST_F(ReverseEliasFanoEncodingTest, roundTripSignedValues) {
  roundTrip<int32_t>({10, 9, 2, 0, -1, -3, -3, -5});
  roundTrip<int64_t>({1'000'000, 1, 0, -1, -7, -100, -100});
  roundTrip<int16_t>({4, 2, 0, -5, -5, -10});
  roundTrip<int8_t>({9, 5, 0, -2, -2, -8});
}

TEST_F(ReverseEliasFanoEncodingTest, roundTripBoundaries) {
  roundTrip<int8_t>({
      std::numeric_limits<int8_t>::max(),
      0,
      -1,
      std::numeric_limits<int8_t>::min(),
  });
  roundTrip<int16_t>({
      std::numeric_limits<int16_t>::max(),
      0,
      -1,
      std::numeric_limits<int16_t>::min(),
  });
  roundTrip<int32_t>({
      std::numeric_limits<int32_t>::max(),
      0,
      -1,
      std::numeric_limits<int32_t>::min(),
  });
  roundTrip<int64_t>({
      std::numeric_limits<int64_t>::max(),
      0,
      -1,
      std::numeric_limits<int64_t>::min(),
  });
  roundTrip<uint8_t>({std::numeric_limits<uint8_t>::max(), 1, 0});
  roundTrip<uint16_t>({std::numeric_limits<uint16_t>::max(), 1, 0});
  roundTrip<uint32_t>({std::numeric_limits<uint32_t>::max(), 1, 0});
  roundTrip<uint64_t>({std::numeric_limits<uint64_t>::max(), 1, 0});
}

TEST_F(ReverseEliasFanoEncodingTest, roundTripSingleAndRepeatedValues) {
  roundTrip<uint64_t>({std::numeric_limits<uint64_t>::max()});
  roundTrip<uint64_t>({0});
  roundTrip<int64_t>({-7, -7, -7, -7});
}

TEST_F(ReverseEliasFanoEncodingTest, roundTripRandomValues) {
  roundTripRandom<int8_t>(/*seed=*/1);
  roundTripRandom<uint8_t>(/*seed=*/2);
  roundTripRandom<int16_t>(/*seed=*/3);
  roundTripRandom<uint16_t>(/*seed=*/4);
  roundTripRandom<int32_t>(/*seed=*/5);
  roundTripRandom<uint32_t>(/*seed=*/6);
  roundTripRandom<int64_t>(/*seed=*/7);
  roundTripRandom<uint64_t>(/*seed=*/8);
}

TEST_F(ReverseEliasFanoEncodingTest, storesComplementsInEliasFanoChild) {
  {
    const auto encoded = encode<uint8_t>({9, 7, 4, 2});
    auto child = nimble::EncodingFactory{}.create(
        *pool_,
        encoded.substr(nimble::EncodingPrefix::kFixedPrefixSize),
        /*stringBufferFactory=*/nullptr);
    EXPECT_EQ(child->encodingType(), nimble::EncodingType::EliasFano);
    EXPECT_EQ(child->dataType(), nimble::DataType::Uint8);
    std::vector<uint8_t> stored(child->rowCount());
    child->materialize(child->rowCount(), stored.data());
    EXPECT_THAT(stored, ::testing::ElementsAre(246, 248, 251, 253));
  }

  // Signed values are complemented in their order-preserving form, which
  // flips the sign bit.
  {
    const auto encoded = encode<int8_t>({1, 0, -1, -128});
    auto child = nimble::EncodingFactory{}.create(
        *pool_,
        encoded.substr(nimble::EncodingPrefix::kFixedPrefixSize),
        /*stringBufferFactory=*/nullptr);
    EXPECT_EQ(child->encodingType(), nimble::EncodingType::EliasFano);
    EXPECT_EQ(child->dataType(), nimble::DataType::Uint8);
    std::vector<uint8_t> stored(child->rowCount());
    child->materialize(child->rowCount(), stored.data());
    EXPECT_THAT(stored, ::testing::ElementsAre(126, 127, 128, 255));
  }
}

TEST_F(ReverseEliasFanoEncodingTest, resetSkipAndMaterialize) {
  const std::vector<uint32_t> input{102, 101, 100, 12, 11, 10, 2, 1, 0};
  auto encoding = createEncoding(input);

  encoding->reset();
  encoding->skip(0);
  encoding->skip(4);
  std::vector<uint32_t> output(3);
  encoding->materialize(static_cast<uint32_t>(output.size()), output.data());
  EXPECT_THAT(output, ::testing::ElementsAre(11, 10, 2));

  encoding->reset();
  output.resize(input.size());
  encoding->materialize(static_cast<uint32_t>(output.size()), output.data());
  EXPECT_EQ(output, input);
}

TEST_F(ReverseEliasFanoEncodingTest, absoluteReadsPreserveSequentialPosition) {
  const std::vector<uint64_t> input{80, 40, 20, 20, 10};
  auto encoding = createEncoding(input);
  const auto& reverseEliasFano =
      dynamic_cast<const nimble::ReverseEliasFanoEncoding<uint64_t>&>(
          *encoding);

  std::vector<uint64_t> absoluteOutput(3);
  reverseEliasFano.materializeAt(
      /*offset=*/1,
      static_cast<uint32_t>(absoluteOutput.size()),
      absoluteOutput.data());
  EXPECT_THAT(absoluteOutput, ::testing::ElementsAre(40, 20, 20));
  EXPECT_EQ(reverseEliasFano.valueAt(4), 10);

  std::vector<uint64_t> sequentialOutput(2);
  encoding->materialize(
      static_cast<uint32_t>(sequentialOutput.size()), sequentialOutput.data());
  EXPECT_THAT(sequentialOutput, ::testing::ElementsAre(80, 40));

  NIMBLE_ASSERT_THROW(
      reverseEliasFano.materializeAt(
          input.size(), /*rowCount=*/1, absoluteOutput.data()),
      "Reading past end of encoding.");
  NIMBLE_ASSERT_THROW(
      reverseEliasFano.valueAt(input.size()), "Reading past end of encoding.");
}

TEST_F(ReverseEliasFanoEncodingTest, readWithVisitorSelectsRows) {
  std::vector<int64_t> input;
  input.reserve(300);
  int64_t value{1'000};
  for (uint32_t i = 0; i < 300; ++i) {
    value -= i % 11;
    input.push_back(value);
  }
  auto encoding = createEncoding(input);
  auto& reverseEliasFano =
      dynamic_cast<nimble::ReverseEliasFanoEncoding<int64_t>&>(*encoding);

  for (const auto& rows : std::vector<std::vector<velox::vector_size_t>>{
           {0, 1, 2, 3, 4, 5, 6, 7},
           {0, 127, 128, 129, 255, 299},
       }) {
    SCOPED_TRACE(fmt::format("rows={}", ::testing::PrintToString(rows)));
    encoding->reset();
    IntegralReadWithVisitor<int64_t> visitor{rows};
    nimble::ReadWithVisitorParams params;
    params.numScanned = 0;

    reverseEliasFano.readWithVisitor(visitor, params);

    std::vector<int64_t> expected;
    expected.reserve(rows.size());
    for (const auto row : rows) {
      expected.push_back(input[row]);
    }
    EXPECT_THAT(visitor.values(), ::testing::ElementsAreArray(expected));
  }
}

TEST_F(ReverseEliasFanoEncodingTest, slices) {
  expectSlices<int64_t>(
      {1'000, 100, 4, 0, -7, -7, std::numeric_limits<int64_t>::min()},
      {{0, 1}, {1, 5}, {3, 1}, {6, 1}, {0, 7}});

  const auto encoded = encode<int64_t>({3, 2, 1});
  nimble::Buffer sliceBuffer{*pool_};
  NIMBLE_ASSERT_THROW(
      nimble::ReverseEliasFanoEncoding<int64_t>::slice(
          encoded, /*offset=*/0, /*length=*/0, sliceBuffer),
      "Cannot slice zero rows.");
}

TEST_F(ReverseEliasFanoEncodingTest, factoryRoundTrip) {
  const std::vector<int64_t> input{
      std::numeric_limits<int64_t>::max(),
      9,
      0,
      -7,
      -7,
      std::numeric_limits<int64_t>::min()};
  const auto encoded = encodeWithCandidates(
      input, {{nimble::EncodingType::ReverseEliasFano, 1.0}});
  auto encoding = nimble::EncodingFactory{}.create(
      *pool_, encoded, /*stringBufferFactory=*/nullptr);
  std::vector<int64_t> output(input.size());
  encoding->materialize(static_cast<uint32_t>(input.size()), output.data());

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::ReverseEliasFano);
  EXPECT_EQ(output, input);
  const auto debugString = encoding->debugString(0);
  EXPECT_NE(debugString.find("child:"), std::string::npos);
  EXPECT_NE(debugString.find("lowerBits="), std::string::npos);
}

TEST_F(ReverseEliasFanoEncodingTest, selectionPrefersDescendingValues) {
  std::vector<uint32_t> descending(1'000);
  for (uint32_t row = 0; row < descending.size(); ++row) {
    descending[row] = 1'000'000 - row * 3;
  }
  const std::vector<std::pair<nimble::EncodingType, float>> candidates{
      {nimble::EncodingType::Trivial, 1.0},
      {nimble::EncodingType::ReverseEliasFano, 1.0},
  };

  EXPECT_EQ(
      nimble::EncodingPrefix::encodingType(
          encodeWithCandidates(descending, candidates)),
      nimble::EncodingType::ReverseEliasFano);

  // One rise makes the stream not non-increasing. Its first value still
  // exceeds its last, so only the order check keeps the encoding out.
  auto bumped = descending;
  std::swap(bumped[500], bumped[501]);
  EXPECT_EQ(
      nimble::EncodingPrefix::encodingType(
          encodeWithCandidates(bumped, candidates)),
      nimble::EncodingType::Trivial);
}

TEST_F(ReverseEliasFanoEncodingTest, estimateMatchesEncodedSize) {
  for (const uint32_t rowCount : {1, 127, 128, 255, 256, 257}) {
    std::vector<int64_t> input(rowCount);
    for (uint32_t row = 0; row < rowCount; ++row) {
      input[row] = 100 - static_cast<int64_t>(row / 2);
    }
    const auto values = toVector(input);
    const auto physicalValues = std::span<const uint64_t>{
        reinterpret_cast<const uint64_t*>(values.data()), values.size()};
    const auto statistics =
        nimble::Statistics<uint64_t>::create(physicalValues);
    for (const bool useVarintRowCount : {false, true}) {
      SCOPED_TRACE(
          ::testing::Message() << "rowCount=" << rowCount
                               << ", useVarintRowCount=" << useVarintRowCount);
      const nimble::Encoding::Options options{
          .useVarintRowCount = useVarintRowCount};
      const auto estimate =
          nimble::ReverseEliasFanoEncoding<int64_t>::estimateSize(
              physicalValues, statistics, options);
      ASSERT_TRUE(estimate.has_value());
      buffer_->reset();
      EXPECT_EQ(*estimate, encode(input, options).size());
    }
  }
}

TEST_F(ReverseEliasFanoEncodingTest, estimateRejectsEmptyAscendingAndUnsorted) {
  const auto emptyStatistics =
      nimble::Statistics<uint64_t>::create(std::span<const uint64_t>{});
  EXPECT_EQ(
      nimble::ReverseEliasFanoEncoding<uint64_t>::estimateSize(
          {}, emptyStatistics),
      std::nullopt);

  for (const auto& input : std::vector<std::vector<uint32_t>>{
           {1, 2},
           {8, 4, 5, 1},
       }) {
    SCOPED_TRACE(fmt::format("input={}", ::testing::PrintToString(input)));
    const std::span<const uint32_t> values{input};
    EXPECT_EQ(
        nimble::ReverseEliasFanoEncoding<uint32_t>::estimateSize(
            values, nimble::Statistics<uint32_t>::create(values)),
        std::nullopt);
  }

  // Unsigned -1 ascends past 0, but signed -1 descends from it.
  const std::vector<uint32_t> signedDescending{
      0, static_cast<uint32_t>(-1), static_cast<uint32_t>(-2)};
  const std::span<const uint32_t> values{signedDescending};
  const auto statistics = nimble::Statistics<uint32_t>::create(values);
  EXPECT_EQ(
      nimble::ReverseEliasFanoEncoding<uint32_t>::estimateSize(
          values, statistics),
      std::nullopt);
  EXPECT_TRUE(
      nimble::ReverseEliasFanoEncoding<int32_t>::estimateSize(
          values, statistics)
          .has_value());
}

TEST_F(ReverseEliasFanoEncodingTest, encodeRejectsAscendingValues) {
  NIMBLE_ASSERT_THROW(
      createEncoding(std::vector<int32_t>{-2, -1}),
      "ReverseEliasFano requires non-empty non-increasing values.");
}

TEST_F(ReverseEliasFanoEncodingTest, encodeRejectsEmptyValues) {
  NIMBLE_ASSERT_THROW(
      createEncoding(std::vector<uint64_t>{}),
      "ReverseEliasFano requires non-empty non-increasing values.");
}

TEST_F(ReverseEliasFanoEncodingTest, rejectsMalformedChild) {
  const std::string encoded{encode<int32_t>({3, 2, 1})};
  constexpr auto kChildOffset = nimble::EncodingPrefix::kFixedPrefixSize;

  {
    auto malformed = encoded;
    malformed[kChildOffset + nimble::EncodingPrefix::kEncodingTypeOffset] =
        static_cast<char>(nimble::EncodingType::Trivial);
    NIMBLE_ASSERT_FILE_THROW(
        nimble::EncodingFactory{}.create(
            *pool_, malformed, /*stringBufferFactory=*/nullptr),
        "ReverseEliasFano child must be an EliasFano stream.");
  }

  {
    auto malformed = encoded;
    malformed[kChildOffset + nimble::EncodingPrefix::kDataTypeOffset] =
        static_cast<char>(nimble::DataType::Int32);
    NIMBLE_ASSERT_FILE_THROW(
        nimble::EncodingFactory{}.create(
            *pool_, malformed, /*stringBufferFactory=*/nullptr),
        "ReverseEliasFano child must hold the unsigned physical type.");
  }

  {
    auto malformed = encoded;
    char* position = malformed.data() + kChildOffset +
        nimble::EncodingPrefix::kRowCountOffset;
    nimble::encoding::writeUint32(2, position);
    NIMBLE_ASSERT_FILE_THROW(
        nimble::EncodingFactory{}.create(
            *pool_, malformed, /*stringBufferFactory=*/nullptr),
        "ReverseEliasFano child row count must match the stream.");
  }

  {
    const auto truncated = encoded.substr(0, kChildOffset + 3);
    NIMBLE_ASSERT_FILE_THROW(
        nimble::EncodingFactory{}.create(
            *pool_, truncated, /*stringBufferFactory=*/nullptr),
        "Truncated ReverseEliasFano child prefix.");
  }
}

} // namespace
