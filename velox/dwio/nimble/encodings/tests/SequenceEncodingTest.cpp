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
#include "velox/dwio/nimble/encodings/SequenceEncoding.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

class SequenceEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  // Returns 'numRows' values that start at 'base' and advance by 'step',
  // wrapping like T. Adds one step at a time, so it does not share the
  // encoding's arithmetic.
  template <typename T>
  nimble::Vector<T> makeSequence(T base, T step, uint32_t numRows) {
    using UnsignedType = std::make_unsigned_t<T>;
    nimble::Vector<T> values{pool_.get()};
    values.reserve(numRows);
    auto value = static_cast<UnsignedType>(base);
    for (uint32_t row = 0; row < numRows; ++row) {
      values.push_back(static_cast<T>(value));
      value =
          static_cast<UnsignedType>(value + static_cast<UnsignedType>(step));
    }
    return values;
  }

  template <typename T>
  nimble::Vector<T> makeVector(std::initializer_list<T> values) {
    nimble::Vector<T> result{pool_.get()};
    result.insert(result.end(), values.begin(), values.end());
    return result;
  }

  template <typename T>
  std::string_view encode(
      const nimble::Vector<T>& values,
      const nimble::Encoding::Options& options = {}) {
    return nimble::test::Encoder<nimble::SequenceEncoding<T>>::encode(
        *buffer_, values, nimble::CompressionType::Uncompressed, options);
  }

  std::unique_ptr<nimble::Encoding> createEncoding(
      std::string_view encoded,
      const nimble::Encoding::Options& options = {}) {
    return nimble::EncodingFactory{options}.create(
        *pool_, encoded, /*stringBufferFactory=*/nullptr);
  }

  template <typename T>
  std::vector<T> materialize(nimble::Encoding& encoding, uint32_t numRows) {
    std::vector<T> output(numRows);
    encoding.materialize(numRows, output.data());
    return output;
  }

  // Encodes each progression with and without a varint row count, and checks
  // the encoded size and every decoded row.
  template <typename T>
  void testProgressions() {
    constexpr auto kMin = std::numeric_limits<T>::min();
    constexpr auto kMax = std::numeric_limits<T>::max();
    struct TestCase {
      T base;
      T step;
      uint32_t numRows;

      std::string debugString() const {
        return fmt::format(
            "type {}, base {}, step {}, numRows {}",
            nimble::TypeTraits<T>::dataType,
            base,
            step,
            numRows);
      }
    };
    const std::vector<TestCase> testCases{
        // Row ids. Wraps around many times for the 8-bit types.
        {T{0}, T{1}, 1'000},
        // Steps down by 3.
        {T{5}, static_cast<T>(-3), 100},
        // Wraps past the maximum.
        {static_cast<T>(kMax - 2), T{1}, 10},
        // A step as wide as the type.
        {kMin, kMax, 50},
        // A constant stream has step 0.
        {T{42}, T{0}, 7},
        // Any one value and any two values form a progression.
        {T{9}, T{4}, 1},
        {T{100}, static_cast<T>(-57), 2},
    };
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      const auto values =
          makeSequence<T>(testCase.base, testCase.step, testCase.numRows);
      for (const bool useVarint : {false, true}) {
        SCOPED_TRACE(fmt::format("useVarint {}", useVarint));
        const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
        const auto encoded = encode(values, options);
        // The prefix, then the base and the step in the physical type.
        const auto dataOffset =
            nimble::EncodingPrefix::serializedSize(testCase.numRows, useVarint);
        EXPECT_EQ(encoded.size(), dataOffset + 2 * sizeof(T));
        EXPECT_EQ(
            nimble::SequenceEncoding<T>::readBase(encoded, dataOffset),
            static_cast<std::make_unsigned_t<T>>(testCase.base));
        // One row has no step to keep, so it stores step 0.
        const T expectedStep = testCase.numRows > 1 ? testCase.step : T{0};
        EXPECT_EQ(
            nimble::SequenceEncoding<T>::readStep(encoded, dataOffset),
            static_cast<std::make_unsigned_t<T>>(expectedStep));

        auto encoding = createEncoding(encoded, options);
        EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Sequence);
        EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<T>::dataType);
        EXPECT_EQ(encoding->rowCount(), testCase.numRows);
        EXPECT_EQ(
            materialize<T>(*encoding, testCase.numRows),
            std::vector<T>(values.begin(), values.end()));
      }
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

TEST_F(SequenceEncodingTest, roundTripsEveryIntegerType) {
  testProgressions<int8_t>();
  testProgressions<uint8_t>();
  testProgressions<int16_t>();
  testProgressions<uint16_t>();
  testProgressions<int32_t>();
  testProgressions<uint32_t>();
  testProgressions<int64_t>();
  testProgressions<uint64_t>();
}

TEST_F(SequenceEncodingTest, skipsAndMaterializesInPieces) {
  const auto values = makeSequence<int64_t>(-20, 7, 100);
  const std::vector<int64_t> expected(values.begin(), values.end());
  auto encoding = createEncoding(encode(values));

  encoding->skip(15);
  EXPECT_EQ(
      materialize<int64_t>(*encoding, 10),
      std::vector<int64_t>(expected.begin() + 15, expected.begin() + 25));
  encoding->skip(70);
  EXPECT_EQ(
      materialize<int64_t>(*encoding, 5),
      std::vector<int64_t>(expected.begin() + 95, expected.end()));

  std::vector<int64_t> output(1);
  NIMBLE_ASSERT_THROW(
      encoding->materialize(1, output.data()), "Reading past end of encoding.");
  NIMBLE_ASSERT_THROW(encoding->skip(1), "Skipping past end of encoding.");

  encoding->reset();
  EXPECT_EQ(
      materialize<int64_t>(*encoding, 3),
      std::vector<int64_t>(expected.begin(), expected.begin() + 3));
}

TEST_F(SequenceEncodingTest, encodeRejectsOtherStreams) {
  // A replayed layout falls back to encoding selection on these errors.
  NIMBLE_ASSERT_USER_THROW(
      encode(makeVector<uint32_t>({1, 2, 4})),
      "Sequence encoding requires an arithmetic progression.");
  NIMBLE_ASSERT_USER_THROW(
      encode(makeVector<int64_t>({})), "Sequence encoding cannot be empty.");
}

TEST_F(SequenceEncodingTest, estimatesSizeOfProgressionsOnly) {
  const auto estimateSize = [](const std::vector<uint64_t>& values,
                               bool useVarint) {
    return nimble::SequenceEncoding<int64_t>::estimateSize(
        values,
        nimble::Statistics<uint64_t>::create(values),
        {.useVarintRowCount = useVarint});
  };
  std::vector<uint64_t> progression(1'000);
  for (uint64_t row = 0; row < progression.size(); ++row) {
    progression[row] = 3 + 2 * row;
  }
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(fmt::format("useVarint {}", useVarint));
    EXPECT_EQ(
        estimateSize(progression, useVarint),
        nimble::EncodingPrefix::serializedSize(1'000, useVarint) +
            2 * sizeof(int64_t));
  }

  progression.back() = 0;
  EXPECT_EQ(estimateSize(progression, /*useVarint=*/false), std::nullopt);
  EXPECT_EQ(estimateSize({}, /*useVarint=*/false), std::nullopt);
}

TEST_F(SequenceEncodingTest, selectionPicksSequenceForProgressionsOnly) {
  const auto readFactors =
      nimble::ManualEncodingSelectionPolicyFactory::parseEncodingReadFactors(
          "Trivial=1.0;FixedBitWidth=1.0;Sequence=1.0");
  const auto encodeWithSelection = [&](const std::vector<int64_t>& values) {
    auto policy =
        std::make_unique<nimble::ManualEncodingSelectionPolicy<int64_t>>(
            readFactors, std::nullopt, std::nullopt);
    return nimble::EncodingFactory::encode<int64_t>(
        std::move(policy), values, *buffer_);
  };

  constexpr uint32_t kNumRows{10'000};
  std::vector<int64_t> rowIds(kNumRows);
  for (uint32_t row = 0; row < kNumRows; ++row) {
    rowIds[row] = 1'000'000 + int64_t{row};
  }
  {
    const auto encoded = encodeWithSelection(rowIds);
    EXPECT_EQ(
        nimble::EncodingPrefix::encodingType(encoded),
        nimble::EncodingType::Sequence);
    auto encoding = createEncoding(encoded);
    EXPECT_EQ(materialize<int64_t>(*encoding, kNumRows), rowIds);
  }

  // One value off the progression makes every row cost bits again.
  rowIds[5'000] = 0;
  {
    const auto encoded = encodeWithSelection(rowIds);
    EXPECT_NE(
        nimble::EncodingPrefix::encodingType(encoded),
        nimble::EncodingType::Sequence);
    auto encoding = createEncoding(encoded);
    EXPECT_EQ(materialize<int64_t>(*encoding, kNumRows), rowIds);
  }

  // The default read factors leave the experimental encoding out.
  const auto defaults = nimble::ManualEncodingSelectionPolicyFactory::
      defaultEncodingReadFactors();
  for (const auto& [encodingType, _] : defaults) {
    EXPECT_NE(encodingType, nimble::EncodingType::Sequence);
  }
}

TEST_F(SequenceEncodingTest, slicesKeepTheStep) {
  const auto values = makeSequence<int32_t>(1'000, -9, 64);
  const auto encoded = encode(values);
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(fmt::format("useVarint {}", useVarint));
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    const auto source = encode(values, options);
    for (const auto& [offset, length] :
         std::vector<std::pair<uint32_t, uint32_t>>{
             {0, 64}, {0, 1}, {63, 1}, {17, 30}}) {
      SCOPED_TRACE(fmt::format("offset {}, length {}", offset, length));
      const auto sliced = nimble::SequenceEncoding<int32_t>::slice(
          source, offset, length, *buffer_, options);
      auto encoding = createEncoding(sliced, options);
      EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Sequence);
      EXPECT_EQ(encoding->rowCount(), length);
      EXPECT_EQ(
          materialize<int32_t>(*encoding, length),
          std::vector<int32_t>(
              values.begin() + offset, values.begin() + offset + length));
    }
  }

  NIMBLE_ASSERT_THROW(
      nimble::SequenceEncoding<int32_t>::slice(
          encoded, /*offset=*/10, /*length=*/0, *buffer_),
      "Cannot slice zero rows.");
  NIMBLE_ASSERT_THROW(
      nimble::SequenceEncoding<int32_t>::slice(
          encoded, /*offset=*/60, /*length=*/5, *buffer_),
      "Slice ends past the last row.");
}

TEST_F(SequenceEncodingTest, rejectsWrongDataSize) {
  const auto encoded = encode(makeSequence<int64_t>(0, 1, 10));
  const std::string truncated{encoded.substr(0, encoded.size() - 1)};
  NIMBLE_ASSERT_FILE_THROW(
      createEncoding(truncated), "Unexpected sequence encoding size.");
  const std::string padded = std::string{encoded} + '\0';
  NIMBLE_ASSERT_FILE_THROW(
      createEncoding(padded), "Unexpected sequence encoding size.");
}

TEST_F(SequenceEncodingTest, debugString) {
  auto encoding = createEncoding(encode(makeSequence<int16_t>(-5, -2, 4)));
  EXPECT_EQ(
      encoding->debugString(), "Sequence<Int16> rowCount=4 base=-5 step=-2");
}
