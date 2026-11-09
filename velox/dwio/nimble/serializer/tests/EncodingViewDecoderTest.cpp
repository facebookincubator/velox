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
#include <limits>
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
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
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

// Pairs decoded values with their null flags. A cleared bit in 'notNulls'
// marks a null.
std::vector<std::optional<int64_t>> withNulls(
    std::span<const int64_t> values,
    const uint64_t* notNulls) {
  std::vector<std::optional<int64_t>> result;
  result.reserve(values.size());
  for (size_t i{0}; i < values.size(); ++i) {
    result.push_back(
        velox::bits::isBitSet(notNulls, i) ? std::optional<int64_t>{values[i]}
                                           : std::nullopt);
  }
  return result;
}

// Returns whether the view behind each of 'views' is destroyed.
std::vector<bool> isExpired(
    const std::vector<std::weak_ptr<const EncodingView>>& views) {
  std::vector<bool> result;
  result.reserve(views.size());
  for (const auto& view : views) {
    result.push_back(view.expired());
  }
  return result;
}

// Returns a Constant encoding of 'numRows' copies of 'value'. Its size does not
// depend on 'numRows', so it can hold any number of rows.
std::string encodeConstant(uint32_t numRows, int64_t value) {
  std::string encoded(EncodingPrefix::kFixedPrefixSize + sizeof(int64_t), '\0');
  char* pos = encoded.data();
  EncodingPrefix::serialize(
      EncodingType::Constant,
      DataType::Int64,
      numRows,
      /*useVarint=*/false,
      pos);
  encoding::write<int64_t>(value, pos);
  return encoded;
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

  // Splits 'values' at random rows into 'numChunks' chunks and encodes each
  // one. Chunks can be empty.
  template <typename T>
  std::vector<std::string> encodeNullableChunks(
      const std::vector<std::optional<T>>& values,
      uint32_t numChunks,
      std::mt19937& rng) {
    std::vector<size_t> splitRows{0, values.size()};
    for (uint32_t i{1}; i < numChunks; ++i) {
      splitRows.push_back(rng() % (values.size() + 1));
    }
    std::sort(splitRows.begin(), splitRows.end());
    std::vector<std::string> chunks;
    chunks.reserve(numChunks);
    for (size_t i{0}; i < numChunks; ++i) {
      chunks.push_back(
          encodeNullable<T>(std::vector<std::optional<T>>{
              values.begin() + splitRows[i],
              values.begin() + splitRows[i + 1]}));
    }
    return chunks;
  }

  std::unique_ptr<EncodingViewDecoder> makeDecoder(
      std::span<const std::string_view> chunks) {
    return std::make_unique<EncodingViewDecoder>(
        chunks,
        pool_.get(),
        /*bufferPool=*/nullptr,
        [this](std::string_view encoded) {
          return createEncodingView(encoded, pool_.get());
        });
  }

  std::unique_ptr<EncodingViewDecoder> makeDecoder(std::string_view stream) {
    return makeDecoder(std::span{&stream, 1});
  }

  // Returns a factory that records every view it creates in 'views'.
  EncodingViewDecoder::EncodingViewFactory makeTrackingFactory(
      std::vector<std::weak_ptr<const EncodingView>>& views) {
    return [this, &views](std::string_view encoded) {
      std::shared_ptr<const EncodingView> view =
          createEncodingView(encoded, pool_.get());
      views.push_back(view);
      return view;
    };
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
    const uint32_t numChunks{1 + static_cast<uint32_t>(rng() % 4)};
    SCOPED_TRACE(::testing::Message() << "numChunks=" << numChunks);
    const auto chunks = encodeNullableChunks<int64_t>(rows, numChunks, rng);
    const std::vector<std::string_view> chunkViews{
        chunks.begin(), chunks.end()};
    auto decoder = makeDecoder(chunkViews);
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

TEST_F(EncodingViewDecoderTest, readsRandomNullableRows) {
  constexpr uint32_t kSeed{1'414'213};
  std::mt19937 rng{kSeed};
  for (uint32_t iteration{0}; iteration < 100; ++iteration) {
    SCOPED_TRACE(
        ::testing::Message()
        << "seed=" << kSeed << ", iteration=" << iteration);
    const uint32_t rowCount{1 + static_cast<uint32_t>(rng() % 256)};
    std::vector<std::optional<int64_t>> values;
    values.reserve(rowCount);
    for (uint32_t row{0}; row < rowCount; ++row) {
      values.push_back(
          rng() % 5 == 0 ? std::nullopt
                         : std::optional<int64_t>{static_cast<int64_t>(rng())});
    }
    const uint32_t numChunks{1 + static_cast<uint32_t>(rng() % 4)};
    SCOPED_TRACE(::testing::Message() << "numChunks=" << numChunks);
    const auto chunks = encodeNullableChunks<int64_t>(values, numChunks, rng);
    const std::vector<std::string_view> chunkViews{
        chunks.begin(), chunks.end()};
    auto decoder = makeDecoder(chunkViews);

    // Picks rows in random order, with repeats.
    const uint32_t numReadRows{1 + static_cast<uint32_t>(rng() % 64)};
    std::vector<uint32_t> rows;
    std::vector<std::optional<int64_t>> expected;
    for (uint32_t i{0}; i < numReadRows; ++i) {
      rows.push_back(static_cast<uint32_t>(rng() % rowCount));
      expected.push_back(values[rows.back()]);
    }
    std::vector<int64_t> output(numReadRows, -1);
    std::vector<uint64_t> outputNulls(
        velox::bits::nwords(numReadRows), ~uint64_t{0});
    std::vector<velox::BufferPtr> stringBuffers;

    const auto numNonNulls = decoder->read(
        rows,
        DataType::Int64,
        output.data(),
        [&outputNulls]() { return outputNulls.data(); },
        stringBuffers);

    EXPECT_EQ(withNulls(output, outputNulls.data()), expected);
    EXPECT_EQ(
        numNonNulls,
        static_cast<uint32_t>(std::count_if(
            expected.begin(), expected.end(), [](const auto& value) {
              return value.has_value();
            })));
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

TEST_F(EncodingViewDecoderTest, remainingRowsTracksSequentialCursor) {
  const std::array<int64_t, 4> values{10, 11, 12, 13};
  const auto stream = encode<int64_t>(values);
  auto decoder = makeDecoder(stream);
  std::array<int64_t, 1> output{};
  std::vector<velox::BufferPtr> stringBuffers;

  EXPECT_EQ(decoder->remainingRows(), 4);
  decoder->skip(2);
  EXPECT_EQ(decoder->remainingRows(), 2);
  EXPECT_EQ(
      decoder->next(
          1,
          output.data(),
          /*getOutputNulls=*/nullptr,
          stringBuffers),
      1);
  EXPECT_EQ(decoder->remainingRows(), 1);
  decoder->skip(1);
  EXPECT_EQ(decoder->remainingRows(), 0);
}

TEST_F(EncodingViewDecoderTest, rebindReplacesChunkAndRewindsCursor) {
  const auto first = encode<int64_t>(std::vector<int64_t>{10, 11, 12, 13});
  const auto second = encode<int64_t>(std::vector<int64_t>{20, 21, 22});
  auto decoder = makeDecoder(first);
  std::array<int64_t, 2> output{};
  std::array<uint64_t, 1> outputNulls{};
  std::vector<velox::BufferPtr> stringBuffers;
  const auto getOutputNulls = [&outputNulls]() { return outputNulls.data(); };
  decoder->next(2, output.data(), getOutputNulls, stringBuffers);
  ASSERT_EQ(output, (std::array<int64_t, 2>{10, 11}));
  ASSERT_EQ(decoder->remainingRows(), 2);

  decoder->rebind(
      std::array<std::string_view, 1>{second}, /*bufferPool=*/nullptr);

  EXPECT_EQ(decoder->remainingRows(), 3);
  decoder->next(2, output.data(), getOutputNulls, stringBuffers);
  EXPECT_EQ(output, (std::array<int64_t, 2>{20, 21}));
}

TEST_F(EncodingViewDecoderTest, rebindRecyclesScratchInTheNewBufferPool) {
  const auto first = encodeNullable<int64_t>({10, std::nullopt, 12});
  const auto second = encodeNullable<int64_t>({20, 21, std::nullopt});
  velox::BufferPool firstPool{velox::BufferPool::kDefaultCapacity};
  velox::BufferPool secondPool{velox::BufferPool::kDefaultCapacity};
  EncodingViewDecoder decoder{
      std::array<std::string_view, 1>{first},
      pool_.get(),
      &firstPool,
      [this](std::string_view encoded) {
        return createEncodingView(encoded, pool_.get());
      }};
  std::array<int64_t, 5> output{};
  std::array<uint64_t, 1> outputNulls{};
  std::array<uint64_t, 1> scatterBits{};
  velox::bits::setBit(scatterBits.data(), 1);
  velox::bits::setBit(scatterBits.data(), 3);
  velox::bits::setBit(scatterBits.data(), 4);
  const velox::bits::Bitmap scatterBitmap{scatterBits.data(), output.size()};
  std::vector<velox::BufferPtr> stringBuffers;
  // Scatters the 3 rows of the bound chunk into 5 output positions, which
  // decodes them into scratch buffers first.
  const auto scatterNext = [&]() {
    decoder.next(
        3,
        output.data(),
        [&outputNulls]() { return outputNulls.data(); },
        stringBuffers,
        &scatterBitmap);
  };

  scatterNext();
  const auto numScratchBuffers = firstPool.size();
  ASSERT_GT(numScratchBuffers, 0);
  ASSERT_EQ(secondPool.size(), 0);

  // The reads of the new chunks take their scratch buffers from the pool of
  // the new binding, not from the pool that the decoder was built with.
  decoder.rebind(std::array<std::string_view, 1>{second}, &secondPool);
  scatterNext();
  EXPECT_EQ(secondPool.size(), numScratchBuffers);
  EXPECT_EQ(firstPool.size(), numScratchBuffers);
}

TEST_F(EncodingViewDecoderTest, releaseDropsViewsUntilRebound) {
  const auto first = encode<int64_t>(std::vector<int64_t>{10, 11, 12});
  const auto second = encode<int64_t>(std::vector<int64_t>{20, 21});
  std::vector<std::weak_ptr<const EncodingView>> views;
  EncodingViewDecoder decoder{
      std::array<std::string_view, 2>{first, second},
      pool_.get(),
      /*bufferPool=*/nullptr,
      makeTrackingFactory(views)};
  ASSERT_THAT(isExpired(views), ::testing::ElementsAre(false, false));

  decoder.release();
  EXPECT_THAT(isExpired(views), ::testing::ElementsAre(true, true));

  decoder.rebind(
      std::array<std::string_view, 1>{second}, /*bufferPool=*/nullptr);
  EXPECT_THAT(isExpired(views), ::testing::ElementsAre(true, true, false));
  EXPECT_EQ(decoder.remainingRows(), 2);
}

TEST_F(EncodingViewDecoderTest, failedRebindReleasesViews) {
  const auto first = encode<int64_t>(std::vector<int64_t>{10, 11, 12});
  const auto second = encode<int64_t>(std::vector<int64_t>{20, 21});
  const auto otherType = encode<int32_t>(std::vector<int32_t>{30});
  std::vector<std::weak_ptr<const EncodingView>> views;
  EncodingViewDecoder decoder{
      std::array<std::string_view, 1>{first},
      pool_.get(),
      /*bufferPool=*/nullptr,
      makeTrackingFactory(views)};

  // The last chunk fails after the views of the earlier chunks are created.
  NIMBLE_ASSERT_FILE_THROW(
      decoder.rebind(
          std::array<std::string_view, 3>{first, second, otherType},
          /*bufferPool=*/nullptr),
      "Encoding chunks of a stream must store the same data type");
  EXPECT_THAT(isExpired(views), ::testing::ElementsAre(true, true, true, true));

  decoder.rebind(
      std::array<std::string_view, 1>{second}, /*bufferPool=*/nullptr);
  EXPECT_EQ(decoder.remainingRows(), 2);
}

TEST_F(EncodingViewDecoderTest, rebindUsesConstructorFactory) {
  const auto first = encode<int64_t>(std::vector<int64_t>{10, 11, 12});
  const auto second = encode<int64_t>(std::vector<int64_t>{20, 21});
  uint32_t numViews{0};
  EncodingViewDecoder decoder{
      std::array<std::string_view, 1>{first},
      pool_.get(),
      /*bufferPool=*/nullptr,
      [&](std::string_view encoded) {
        ++numViews;
        return createEncodingView(encoded, pool_.get());
      }};
  ASSERT_EQ(numViews, 1);

  decoder.rebind(
      std::array<std::string_view, 2>{second, first}, /*bufferPool=*/nullptr);
  decoder.rebind(
      std::array<std::string_view, 1>{first}, /*bufferPool=*/nullptr);

  EXPECT_EQ(numViews, 4);
}

TEST_F(EncodingViewDecoderTest, readsRangesAcrossChunks) {
  const auto first = encode<int64_t>(std::vector<int64_t>{10, 11, 12});
  const auto second = encode<int64_t>(std::vector<int64_t>{20, 21});
  const auto third = encode<int64_t>(std::vector<int64_t>{30, 31, 32});
  const std::array<std::string_view, 3> chunks{first, second, third};
  auto decoder = makeDecoder(chunks);

  // Stream rows 0-2 are in the first chunk, 3-4 in the second and 5-7 in the
  // third. The second range crosses two chunk boundaries and covers the whole
  // second chunk.
  const std::array<RowRange, 3> ranges{{{0, 1}, {2, 6}, {7, 8}}};
  std::array<int64_t, 6> output{};
  std::vector<velox::BufferPtr> stringBuffers;

  EXPECT_EQ(
      decoder->read(
          ranges,
          DataType::Int64,
          output.data(),
          /*getOutputNulls=*/nullptr,
          stringBuffers),
      6);
  EXPECT_THAT(output, ::testing::ElementsAre(10, 12, 20, 21, 30, 32));
}

TEST_F(EncodingViewDecoderTest, readsUnorderedRowsAcrossChunks) {
  const auto first = encode<int64_t>(std::vector<int64_t>{10, 11, 12});
  const auto second = encode<int64_t>(std::vector<int64_t>{20, 21});
  const auto third = encode<int64_t>(std::vector<int64_t>{30, 31, 32});
  const std::array<std::string_view, 3> chunks{first, second, third};
  auto decoder = makeDecoder(chunks);
  const std::array<uint32_t, 6> rows{6, 0, 3, 4, 2, 7};
  std::array<int64_t, 6> output{};
  std::vector<velox::BufferPtr> stringBuffers;

  EXPECT_EQ(
      decoder->read(
          rows,
          DataType::Int64,
          output.data(),
          /*getOutputNulls=*/nullptr,
          stringBuffers),
      6);
  EXPECT_THAT(output, ::testing::ElementsAre(31, 10, 20, 21, 12, 32));
}

TEST_F(EncodingViewDecoderTest, readsNullsFromSomeChunks) {
  const std::array<RowRange, 1> ranges{{{1, 4}}};
  std::vector<velox::BufferPtr> stringBuffers;

  // Only the later chunk has a null. The rows read from the earlier chunk
  // must still read as non-null.
  {
    const auto first = encode<int64_t>(std::vector<int64_t>{10, 11});
    const auto second = encodeNullable<int64_t>({std::nullopt, 21});
    const std::array<std::string_view, 2> chunks{first, second};
    auto decoder = makeDecoder(chunks);
    std::array<int64_t, 3> output{};
    std::array<uint64_t, 1> outputNulls{};

    EXPECT_EQ(
        decoder->read(
            ranges,
            DataType::Int64,
            output.data(),
            [&outputNulls]() { return outputNulls.data(); },
            stringBuffers),
        2);
    EXPECT_THAT(
        withNulls(output, outputNulls.data()),
        ::testing::ElementsAre(11, std::nullopt, 21));
  }

  // Only the earlier chunk has a null. The rows read from the later chunk must
  // still read as non-null.
  {
    const auto first = encodeNullable<int64_t>({10, std::nullopt});
    const auto second = encode<int64_t>(std::vector<int64_t>{20, 21});
    const std::array<std::string_view, 2> chunks{first, second};
    auto decoder = makeDecoder(chunks);
    std::array<int64_t, 3> output{};
    std::array<uint64_t, 1> outputNulls{};

    EXPECT_EQ(
        decoder->read(
            ranges,
            DataType::Int64,
            output.data(),
            [&outputNulls]() { return outputNulls.data(); },
            stringBuffers),
        2);
    EXPECT_THAT(
        withNulls(output, outputNulls.data()),
        ::testing::ElementsAre(std::nullopt, 20, 21));
  }
}

TEST_F(EncodingViewDecoderTest, retainsStringsAcrossChunks) {
  const auto first = encode<std::string_view>(
      std::vector<std::string_view>{"first string", "unused"});
  const auto second =
      encode<std::string_view>(std::vector<std::string_view>{"second string"});
  const std::array<std::string_view, 2> chunks{first, second};
  const std::array<uint32_t, 2> rows{0, 2};
  std::array<std::string_view, 2> output{};
  std::vector<velox::BufferPtr> stringBuffers;

  {
    auto decoder = makeDecoder(chunks);
    EXPECT_EQ(
        decoder->read(
            rows,
            DataType::String,
            output.data(),
            /*getOutputNulls=*/nullptr,
            stringBuffers),
        rows.size());
  }

  EXPECT_THAT(output, ::testing::ElementsAre("first string", "second string"));
  ASSERT_EQ(stringBuffers.size(), 1);
  const std::string_view retainedContent{
      stringBuffers[0]->as<char>(), stringBuffers[0]->size()};
  EXPECT_EQ(retainedContent, "first stringsecond string");
}

TEST_F(EncodingViewDecoderTest, readsSequentialRowsAcrossChunks) {
  const auto first = encodeNullable<int64_t>({10, std::nullopt, 12});
  const auto second = encode<int64_t>(std::vector<int64_t>{20, 21});
  const std::array<std::string_view, 2> chunks{first, second};
  auto decoder = makeDecoder(chunks);
  std::vector<velox::BufferPtr> stringBuffers;

  // Reads stream rows 2-3, which cross the chunk boundary.
  {
    std::array<int64_t, 2> output{};
    std::array<uint64_t, 1> outputNulls{};
    decoder->skip(2);
    EXPECT_EQ(
        decoder->next(
            2,
            output.data(),
            [&outputNulls]() { return outputNulls.data(); },
            stringBuffers),
        2);
    EXPECT_THAT(output, ::testing::ElementsAre(12, 20));
  }

  // Scatters stream rows 1-3 to output positions 0, 2 and 3.
  {
    std::array<int64_t, 5> output{};
    std::array<uint64_t, 1> outputNulls{};
    std::array<uint64_t, 1> scatterBits{};
    velox::bits::setBit(scatterBits.data(), 0);
    velox::bits::setBit(scatterBits.data(), 2);
    velox::bits::setBit(scatterBits.data(), 3);
    const velox::bits::Bitmap scatterBitmap{scatterBits.data(), output.size()};
    decoder->reset();
    decoder->skip(1);
    EXPECT_EQ(
        decoder->next(
            3,
            output.data(),
            [&outputNulls]() { return outputNulls.data(); },
            stringBuffers,
            &scatterBitmap),
        2);
    EXPECT_THAT(
        withNulls(output, outputNulls.data()),
        ::testing::ElementsAre(
            std::nullopt, std::nullopt, 12, 20, std::nullopt));
  }
}

TEST_F(EncodingViewDecoderTest, remainingRowsTracksChunks) {
  const auto first = encode<int64_t>(std::vector<int64_t>{10, 11, 12});
  const auto second = encode<int64_t>(std::vector<int64_t>{20, 21});
  const std::array<std::string_view, 2> chunks{first, second};
  auto decoder = makeDecoder(chunks);
  std::array<int64_t, 1> output{};
  std::vector<velox::BufferPtr> stringBuffers;

  EXPECT_EQ(decoder->remainingRows(), 3);
  decoder->skip(2);
  EXPECT_EQ(decoder->remainingRows(), 1);
  decoder->skip(1);
  EXPECT_EQ(decoder->remainingRows(), 2);
  EXPECT_EQ(
      decoder->next(
          1, output.data(), /*getOutputNulls=*/nullptr, stringBuffers),
      1);
  EXPECT_EQ(output[0], 20);
  EXPECT_EQ(decoder->remainingRows(), 1);
  decoder->skip(1);
  EXPECT_EQ(decoder->remainingRows(), 0);
}

TEST_F(EncodingViewDecoderTest, skipsEmptyChunks) {
  const auto first = encode<int64_t>(std::vector<int64_t>{10, 11});
  const auto empty = encode<int64_t>(std::vector<int64_t>{});
  const auto last = encode<int64_t>(std::vector<int64_t>{20});
  const std::array<std::string_view, 3> chunks{first, empty, last};
  auto decoder = makeDecoder(chunks);
  std::vector<velox::BufferPtr> stringBuffers;

  {
    const std::array<RowRange, 1> ranges{{{1, 3}}};
    std::array<int64_t, 2> output{};
    decoder->read(
        ranges,
        DataType::Int64,
        output.data(),
        /*getOutputNulls=*/nullptr,
        stringBuffers);
    EXPECT_THAT(output, ::testing::ElementsAre(11, 20));
  }

  {
    const std::array<uint32_t, 2> rows{2, 0};
    std::array<int64_t, 2> output{};
    decoder->read(
        rows,
        DataType::Int64,
        output.data(),
        /*getOutputNulls=*/nullptr,
        stringBuffers);
    EXPECT_THAT(output, ::testing::ElementsAre(20, 10));
  }

  decoder->skip(2);
  EXPECT_EQ(decoder->remainingRows(), 1);
}

TEST_F(EncodingViewDecoderTest, rebindChangesChunkCount) {
  const auto single = encode<int64_t>(std::vector<int64_t>{1, 2});
  const auto first = encode<int64_t>(std::vector<int64_t>{10, 11});
  const auto second = encode<int64_t>(std::vector<int64_t>{20});
  const auto third = encode<int64_t>(std::vector<int64_t>{30, 31});
  auto decoder = makeDecoder(single);
  std::vector<velox::BufferPtr> stringBuffers;

  decoder->rebind(
      std::array<std::string_view, 3>{first, second, third},
      /*bufferPool=*/nullptr);
  {
    const std::array<RowRange, 1> ranges{{{1, 4}}};
    std::array<int64_t, 3> output{};
    decoder->read(
        ranges,
        DataType::Int64,
        output.data(),
        /*getOutputNulls=*/nullptr,
        stringBuffers);
    EXPECT_THAT(output, ::testing::ElementsAre(11, 20, 30));
    EXPECT_EQ(decoder->remainingRows(), 2);
  }

  decoder->rebind(
      std::array<std::string_view, 1>{single}, /*bufferPool=*/nullptr);
  {
    const std::array<RowRange, 1> ranges{{{0, 2}}};
    std::array<int64_t, 2> output{};
    decoder->read(
        ranges,
        DataType::Int64,
        output.data(),
        /*getOutputNulls=*/nullptr,
        stringBuffers);
    EXPECT_THAT(output, ::testing::ElementsAre(1, 2));
    EXPECT_EQ(decoder->remainingRows(), 2);
  }
}

TEST_F(EncodingViewDecoderTest, rejectsInvalidChunks) {
  NIMBLE_ASSERT_THROW(
      makeDecoder(std::span<const std::string_view>{}),
      "EncodingViewDecoder requires at least one chunk");

  const auto first = encode<int64_t>(std::vector<int64_t>{10, 11});
  const auto second = encode<int32_t>(std::vector<int32_t>{20});
  NIMBLE_ASSERT_FILE_THROW(
      makeDecoder(std::array<std::string_view, 2>{first, second}),
      "Encoding chunks of a stream must store the same data type");
}

TEST_F(EncodingViewDecoderTest, limitsStreamRowsToUint32) {
  constexpr auto kMaxRows = std::numeric_limits<uint32_t>::max();
  const auto first = encodeConstant(kMaxRows - 1, /*value=*/1);
  const auto last = encodeConstant(1, /*value=*/2);

  {
    auto decoder = makeDecoder(std::array<std::string_view, 2>{first, last});
    const std::array<RowRange, 1> ranges{{{kMaxRows - 2, kMaxRows}}};
    std::array<int64_t, 2> output{};
    std::vector<velox::BufferPtr> stringBuffers;
    decoder->read(
        ranges,
        DataType::Int64,
        output.data(),
        /*getOutputNulls=*/nullptr,
        stringBuffers);
    EXPECT_THAT(output, ::testing::ElementsAre(1, 2));
  }

  NIMBLE_ASSERT_FILE_THROW(
      makeDecoder(std::array<std::string_view, 3>{first, last, last}),
      "Stream rows must fit in uint32_t");
}

TEST_F(EncodingViewDecoderTest, rejectsReadsBeyondChunks) {
  const auto first = encode<int64_t>(std::vector<int64_t>{10, 11, 12});
  const auto second = encode<int64_t>(std::vector<int64_t>{20, 21});
  const std::array<std::string_view, 2> chunks{first, second};
  auto decoder = makeDecoder(chunks);
  std::array<int64_t, 2> output{};
  std::vector<velox::BufferPtr> stringBuffers;

  const std::array<RowRange, 1> ranges{{{4, 6}}};
  NIMBLE_ASSERT_THROW(
      decoder->read(
          ranges,
          DataType::Int64,
          output.data(),
          /*getOutputNulls=*/nullptr,
          stringBuffers),
      "Read rows must be within the stream");

  const std::array<uint32_t, 2> rows{0, 5};
  NIMBLE_ASSERT_THROW(
      decoder->read(
          rows,
          DataType::Int64,
          output.data(),
          /*getOutputNulls=*/nullptr,
          stringBuffers),
      "Read rows must be within the stream");
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
    const uint32_t numChunks{1 + static_cast<uint32_t>(rng() % 4)};
    SCOPED_TRACE(::testing::Message() << "numChunks=" << numChunks);
    const auto chunks = encodeNullableChunks<int64_t>(values, numChunks, rng);
    const std::vector<std::string_view> chunkViews{
        chunks.begin(), chunks.end()};
    auto decoder = makeDecoder(chunkViews);
    uint32_t nextRow{0};
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
