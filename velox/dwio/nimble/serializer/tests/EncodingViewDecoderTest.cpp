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

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/serializer/EncodingViewDecoder.h"

namespace facebook::nimble {
namespace {

// Builds ordered, disjoint ranges with randomized gaps and lengths.
std::vector<RowRange> makeRandomRanges(uint32_t rowCount, std::mt19937& rng) {
  std::vector<RowRange> ranges;
  uint32_t row = static_cast<uint32_t>(rng() % std::min<uint32_t>(rowCount, 8));
  while (row < rowCount) {
    const auto maxLength = std::min<uint32_t>(rowCount - row, 16);
    const uint32_t length{1 + static_cast<uint32_t>(rng() % maxLength)};
    const auto end = row + length;
    ranges.emplace_back(row, end);
    row = std::min<uint32_t>(rowCount, end + static_cast<uint32_t>(rng() % 8));
  }
  return ranges;
}

class EncodingViewDecoderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool(
        "encoding_view_decoder_test");
  }

  template <typename T>
  std::string encode(std::span<const T> values) {
    Buffer buffer{*pool_};
    auto policy = std::make_unique<ManualEncodingSelectionPolicy<T>>(
        std::vector<std::pair<EncodingType, float>>{
            {EncodingType::Trivial, 1.0}},
        CompressionOptions{},
        std::nullopt);
    const auto encoded =
        EncodingFactory::encode<T>(std::move(policy), values, buffer);
    return std::string{encoded};
  }

  template <typename T>
  std::string encodeNullable(const std::vector<std::optional<T>>& values) {
    std::vector<T> nonNullValues;
    Vector<bool> isNonNull{pool_.get(), values.size()};
    for (size_t i{0}; i < values.size(); ++i) {
      isNonNull[i] = values[i].has_value();
      if (values[i].has_value()) {
        nonNullValues.push_back(*values[i]);
      }
    }

    Buffer buffer{*pool_};
    auto policy = std::make_unique<ManualEncodingSelectionPolicy<T>>(
        std::vector<std::pair<EncodingType, float>>{
            {EncodingType::Trivial, 1.0}},
        CompressionOptions{},
        std::nullopt);
    const auto encoded = EncodingFactory::encodeNullable<T>(
        std::move(policy), nonNullValues, isNonNull, buffer);
    return std::string{encoded};
  }

  std::unique_ptr<EncodingViewDecoder> makeDecoder(std::string_view stream) {
    return std::make_unique<EncodingViewDecoder>(
        stream,
        pool_.get(),
        /*bufferPool=*/nullptr,
        [this](std::string_view encoded) {
          return createEncodingView(encoded, pool_.get());
        });
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(EncodingViewDecoderTest, readsSelectedNullableRows) {
  const auto stream =
      encodeNullable<int64_t>({10, std::nullopt, 12, 13, std::nullopt, 15});
  auto decoder = makeDecoder(stream);
  const std::array<uint32_t, 4> rows{1, 2, 4, 5};
  std::array<int64_t, 4> output{};
  std::array<uint64_t, 1> outputNulls{};
  std::vector<velox::BufferPtr> stringBuffers;

  const auto numNonNulls = decoder->read(
      rows,
      DataType::Int64,
      output.data(),
      [&outputNulls]() { return outputNulls.data(); },
      stringBuffers);

  EXPECT_EQ(numNonNulls, 2);
  EXPECT_EQ(output, (std::array<int64_t, 4>{0, 12, 0, 15}));
  std::array<bool, 4> actualNotNulls{};
  for (size_t i{0}; i < actualNotNulls.size(); ++i) {
    actualNotNulls[i] = velox::bits::isBitSet(outputNulls.data(), i);
  }
  EXPECT_EQ(actualNotNulls, (std::array<bool, 4>{false, true, false, true}));
  EXPECT_TRUE(stringBuffers.empty());
}

TEST_F(EncodingViewDecoderTest, readsNullableRanges) {
  const auto stream =
      encodeNullable<int64_t>({10, std::nullopt, 12, 13, std::nullopt, 15});
  auto decoder = makeDecoder(stream);
  const std::array<RowRange, 2> ranges{{{0, 2}, {4, 6}}};
  std::array<int64_t, 4> output{};
  std::array<uint64_t, 1> outputNulls{};
  std::vector<velox::BufferPtr> stringBuffers;

  const auto numNonNulls = decoder->read(
      ranges,
      DataType::Int64,
      output.data(),
      [&outputNulls]() { return outputNulls.data(); },
      stringBuffers);

  EXPECT_EQ(numNonNulls, 2);
  EXPECT_EQ(output, (std::array<int64_t, 4>{10, 0, 0, 15}));
  std::array<bool, 4> actualNotNulls{};
  for (size_t i{0}; i < actualNotNulls.size(); ++i) {
    actualNotNulls[i] = velox::bits::isBitSet(outputNulls.data(), i);
  }
  EXPECT_EQ(actualNotNulls, (std::array<bool, 4>{true, false, false, true}));
  EXPECT_TRUE(stringBuffers.empty());
}

TEST_F(EncodingViewDecoderTest, readsRandomNullableRanges) {
  constexpr uint32_t kSeed{2'718'281};
  std::mt19937 rng{kSeed};
  for (uint32_t iteration{0}; iteration < 100; ++iteration) {
    SCOPED_TRACE(
        ::testing::Message()
        << "seed=" << kSeed << ", iteration=" << iteration);
    const uint32_t rowCount{1 + static_cast<uint32_t>(rng() % 256)};
    const uint32_t nullModulo{1 + static_cast<uint32_t>(rng() % 8)};
    std::vector<std::optional<int64_t>> rows;
    rows.reserve(rowCount);
    for (uint32_t row{0}; row < rowCount; ++row) {
      rows.push_back(
          rng() % nullModulo == 0
              ? std::nullopt
              : std::optional<int64_t>{static_cast<int64_t>(rng())});
    }
    const auto ranges = makeRandomRanges(rowCount, rng);
    std::vector<std::optional<int64_t>> expected;
    for (const auto& range : ranges) {
      expected.insert(
          expected.end(),
          rows.begin() + range.startRow,
          rows.begin() + range.endRow);
    }
    const auto stream = encodeNullable<int64_t>(rows);
    auto decoder = makeDecoder(stream);
    std::vector<int64_t> output(expected.size(), -1);
    std::vector<uint64_t> outputNulls(
        velox::bits::nwords(expected.size()), ~uint64_t{0});
    std::vector<velox::BufferPtr> stringBuffers;

    const auto numNonNulls = decoder->read(
        ranges,
        DataType::Int64,
        output.data(),
        [&outputNulls]() { return outputNulls.data(); },
        stringBuffers);

    std::vector<std::optional<int64_t>> actual;
    actual.reserve(output.size());
    for (size_t i{0}; i < output.size(); ++i) {
      actual.push_back(
          velox::bits::isBitSet(outputNulls.data(), i)
              ? std::optional<int64_t>{output[i]}
              : std::nullopt);
    }
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(
        numNonNulls,
        static_cast<uint32_t>(std::count_if(
            expected.begin(), expected.end(), [](const auto& value) {
              return value.has_value();
            })));
    EXPECT_TRUE(stringBuffers.empty());
  }
}

TEST_F(EncodingViewDecoderTest, retainsSelectedStrings) {
  const auto stream = encode<std::string_view>(
      std::vector<std::string_view>{"first string", "unused", "third string"});
  const std::array<uint32_t, 2> rows{0, 2};
  std::array<std::string_view, 2> output{};
  std::vector<velox::BufferPtr> stringBuffers;

  {
    auto decoder = makeDecoder(stream);
    EXPECT_EQ(
        decoder->read(
            rows,
            DataType::String,
            output.data(),
            /*getOutputNulls=*/nullptr,
            stringBuffers),
        rows.size());
  }

  EXPECT_EQ(
      output,
      (std::array<std::string_view, 2>{"first string", "third string"}));
  ASSERT_EQ(stringBuffers.size(), 1);
  const std::string_view retainedContent{
      stringBuffers[0]->as<char>(), stringBuffers[0]->size()};
  EXPECT_EQ(retainedContent, "first stringthird string");
}

TEST_F(EncodingViewDecoderTest, readsSequentialRowsAfterSkipAndReset) {
  const auto stream = encodeNullable<int64_t>({10, std::nullopt, 12, 13});
  auto decoder = makeDecoder(stream);
  std::array<int64_t, 2> output{};
  std::array<uint64_t, 1> outputNulls{};
  std::vector<velox::BufferPtr> stringBuffers;

  decoder->skip(1);
  EXPECT_EQ(
      decoder->next(
          2,
          output.data(),
          [&outputNulls]() { return outputNulls.data(); },
          stringBuffers),
      1);

  EXPECT_THAT(output, ::testing::ElementsAre(0, 12));
  EXPECT_FALSE(velox::bits::isBitSet(outputNulls.data(), 0));
  EXPECT_TRUE(velox::bits::isBitSet(outputNulls.data(), 1));

  decoder->reset();
  EXPECT_EQ(
      decoder->next(
          1,
          output.data(),
          [&outputNulls]() { return outputNulls.data(); },
          stringBuffers),
      1);
  EXPECT_EQ(output[0], 10);
}

TEST_F(EncodingViewDecoderTest, scattersSequentialRows) {
  const auto stream = encodeNullable<int64_t>({10, std::nullopt, 12});
  auto decoder = makeDecoder(stream);
  std::array<int64_t, 5> output{};
  std::array<uint64_t, 1> outputNulls{};
  std::array<uint64_t, 1> scatterBits{};
  velox::bits::setBit(scatterBits.data(), 1);
  velox::bits::setBit(scatterBits.data(), 3);
  velox::bits::setBit(scatterBits.data(), 4);
  const velox::bits::Bitmap scatterBitmap{scatterBits.data(), output.size()};
  std::vector<velox::BufferPtr> stringBuffers;

  EXPECT_EQ(
      decoder->next(
          3,
          output.data(),
          [&outputNulls]() { return outputNulls.data(); },
          stringBuffers,
          &scatterBitmap),
      2);

  std::vector<std::optional<int64_t>> actual;
  actual.reserve(output.size());
  for (size_t i{0}; i < output.size(); ++i) {
    actual.push_back(
        velox::bits::isBitSet(outputNulls.data(), i)
            ? std::optional<int64_t>{output[i]}
            : std::nullopt);
  }
  EXPECT_THAT(
      actual,
      ::testing::ElementsAre(std::nullopt, 10, std::nullopt, std::nullopt, 12));
}

TEST_F(EncodingViewDecoderTest, mixesRandomReadsSkipsAndResets) {
  constexpr uint32_t kSeed{1'618'033};
  constexpr uint32_t kNumRows{257};
  std::mt19937 rng{kSeed};
  std::vector<std::optional<int64_t>> values;
  values.reserve(kNumRows);
  for (uint32_t row{0}; row < kNumRows; ++row) {
    values.push_back(
        row % 7 == 0 ? std::nullopt
                     : std::optional<int64_t>{static_cast<int64_t>(rng())});
  }
  const auto stream = encodeNullable<int64_t>(values);
  auto decoder = makeDecoder(stream);
  uint32_t nextRow{0};

  const auto readActual = [](std::span<const int64_t> output,
                             const uint64_t* outputNulls) {
    std::vector<std::optional<int64_t>> actual;
    actual.reserve(output.size());
    for (size_t i{0}; i < output.size(); ++i) {
      actual.push_back(
          velox::bits::isBitSet(outputNulls, i)
              ? std::optional<int64_t>{output[i]}
              : std::nullopt);
    }
    return actual;
  };

  for (uint32_t iteration{0}; iteration < 100; ++iteration) {
    SCOPED_TRACE(
        ::testing::Message()
        << "seed=" << kSeed << ", iteration=" << iteration);
    decoder->reset();
    nextRow = 0;
    for (uint32_t operation{0}; operation < 32; ++operation) {
      if (operation % 5 == 0) {
        decoder->reset();
        nextRow = 0;
        continue;
      }

      if (operation % 3 == 0) {
        const uint32_t startRow{static_cast<uint32_t>(rng() % kNumRows)};
        const uint32_t numRows{
            1 +
            static_cast<uint32_t>(rng() % std::min(16u, kNumRows - startRow))};
        const std::array<RowRange, 1> ranges{{{startRow, startRow + numRows}}};
        std::vector<int64_t> output(numRows);
        std::vector<uint64_t> outputNulls(
            velox::bits::nwords(numRows), ~uint64_t{0});
        std::vector<velox::BufferPtr> stringBuffers;

        const auto numNonNulls = decoder->read(
            ranges,
            DataType::Int64,
            output.data(),
            [&outputNulls]() { return outputNulls.data(); },
            stringBuffers);

        const std::vector<std::optional<int64_t>> expected(
            values.begin() + startRow, values.begin() + startRow + numRows);
        EXPECT_THAT(
            readActual(output, outputNulls.data()),
            ::testing::ElementsAreArray(expected));
        EXPECT_EQ(
            numNonNulls,
            static_cast<uint32_t>(std::count_if(
                expected.begin(), expected.end(), [](const auto& value) {
                  return value.has_value();
                })));
        continue;
      }

      const auto remainingRows = kNumRows - nextRow;
      if (remainingRows == 0) {
        continue;
      }
      const uint32_t numRows{
          1 + static_cast<uint32_t>(rng() % std::min(16u, remainingRows))};
      if (operation % 3 == 1) {
        decoder->skip(numRows);
        nextRow += numRows;
        continue;
      }

      const uint32_t outputSize{numRows * 2 + 1};
      std::vector<uint64_t> scatterBits(
          velox::bits::nwords(outputSize), uint64_t{0});
      std::vector<uint32_t> outputRows;
      outputRows.reserve(numRows);
      for (uint32_t i{0}; i < numRows; ++i) {
        const uint32_t outputRow{i * 2 + static_cast<uint32_t>(rng() & 1)};
        outputRows.push_back(outputRow);
        velox::bits::setBit(scatterBits.data(), outputRow);
      }
      const velox::bits::Bitmap scatterBitmap{scatterBits.data(), outputSize};
      std::vector<int64_t> output(outputSize);
      std::vector<uint64_t> outputNulls(
          velox::bits::nwords(outputSize), ~uint64_t{0});
      std::vector<velox::BufferPtr> stringBuffers;

      const auto numNonNulls = decoder->next(
          numRows,
          output.data(),
          [&outputNulls]() { return outputNulls.data(); },
          stringBuffers,
          &scatterBitmap);

      std::vector<std::optional<int64_t>> expected(outputSize, std::nullopt);
      for (uint32_t i{0}; i < numRows; ++i) {
        expected[outputRows[i]] = values[nextRow + i];
      }
      EXPECT_THAT(
          readActual(output, outputNulls.data()),
          ::testing::ElementsAreArray(expected));
      EXPECT_EQ(
          numNonNulls,
          static_cast<uint32_t>(std::count_if(
              expected.begin(), expected.end(), [](const auto& value) {
                return value.has_value();
              })));
      nextRow += numRows;
    }
  }
}

} // namespace
} // namespace facebook::nimble
