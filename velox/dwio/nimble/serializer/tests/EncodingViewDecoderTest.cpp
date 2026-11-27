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
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/views/DecodedStringBuffers.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/serializer/EncodingViewDecoder.h"

namespace facebook::nimble {
namespace {

// Returns true if all of 'value' lies in one of 'buffers'.
bool liesIn(
    std::string_view value,
    const std::vector<velox::BufferPtr>& buffers) {
  return std::any_of(buffers.begin(), buffers.end(), [&](const auto& buffer) {
    const auto* begin = buffer->template as<char>();
    return value.data() >= begin &&
        value.data() + value.size() <= begin + buffer->size();
  });
}

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

// Returns compression options that leave payloads uncompressed, so that the
// views read their strings from the encoded bytes.
CompressionOptions uncompressedOptions() {
  CompressionOptions options;
  options.compressionType = CompressionType::Uncompressed;
  return options;
}

// Returns a string of its own for every row, which lies in no buffer, and
// passes a buffer that does not hold it to the addStringBuffer of its reads:
// a view that breaks the contract of EncodingView::read().
class OutsideBufferView final : public TypedEncodingView<std::string_view> {
 public:
  using TypedEncodingView<std::string_view>::read;

  // Takes the rows of 'data', an encoding of strings.
  OutsideBufferView(std::string_view data, velox::memory::MemoryPool* pool)
      : TypedEncodingView<std::string_view>{data, pool, Encoding::Options{}},
        passedBuffer_{velox::AlignedBuffer::allocate<char>(1, pool)} {}

  uint32_t read(
      std::span<const RowRange> ranges,
      const std::function<void(uint32_t)>& /*setNull*/,
      const std::function<void(velox::BufferPtr)>& addStringBuffer,
      void* output) const final {
    const auto numRows = this->checkReadRanges(ranges);
    auto* values = static_cast<std::string_view*>(output);
    std::fill(values, values + numRows, std::string_view{value_});
    if (addStringBuffer != nullptr) {
      addStringBuffer(passedBuffer_);
    }
    return numRows;
  }

 private:
  std::string_view readTypedAt(uint32_t /*index*/) const final {
    return value_;
  }

  void readPhysical(
      uint32_t /*offset*/,
      uint32_t length,
      std::string_view* output) const final {
    std::fill(output, output + length, std::string_view{value_});
  }

  // The string of every row.
  const std::string value_{"a string outside the passed buffer"};

  // The buffer that each read passes.
  const velox::BufferPtr passedBuffer_;
};

// Returns 'numStrings' strings that FSST compresses.
std::vector<std::string> makeFsstStrings(uint32_t numStrings) {
  std::vector<std::string> strings;
  strings.reserve(numStrings);
  for (uint32_t i{0}; i < numStrings; ++i) {
    strings.push_back(fmt::format("repeated fsst prefix value {:06d}", i));
  }
  return strings;
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
    return encode<T>(values, EncodingType::Trivial, CompressionOptions{});
  }

  // Encodes 'values' with the selection policy limited to 'encodingType'
  // and payloads left uncompressed, so that the views read their strings from
  // the encoded bytes.
  template <typename T>
  std::string encodeUncompressed(
      std::span<const T> values,
      EncodingType encodingType) {
    CompressionOptions compressionOptions;
    compressionOptions.compressionType = CompressionType::Uncompressed;
    return encode<T>(values, encodingType, compressionOptions);
  }

  template <typename T>
  std::string encode(
      std::span<const T> values,
      EncodingType encodingType,
      const CompressionOptions& compressionOptions) {
    Buffer buffer{*pool_};
    auto policy = std::make_unique<ManualEncodingSelectionPolicy<T>>(
        std::vector<std::pair<EncodingType, float>>{{encodingType, 1.0}},
        compressionOptions,
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

  // Encodes 'values' with the selection policy limited to 'encodingType' and
  // with 'compressionOptions', or, when a value is null, as Nullable, with the
  // non-null values encoded so.
  std::string encodeStrings(
      std::span<const std::optional<std::string_view>> values,
      EncodingType encodingType,
      const CompressionOptions& compressionOptions) {
    std::vector<std::string_view> nonNullValues;
    Vector<bool> isNonNull{pool_.get(), values.size()};
    for (size_t i{0}; i < values.size(); ++i) {
      isNonNull[i] = values[i].has_value();
      if (values[i].has_value()) {
        nonNullValues.push_back(*values[i]);
      }
    }
    auto encodedValues = encode<std::string_view>(
        nonNullValues, encodingType, compressionOptions);
    if (nonNullValues.size() == values.size()) {
      return encodedValues;
    }
    const auto encodedNulls =
        encode<bool>(std::span<const bool>{isNonNull.data(), isNonNull.size()});
    Buffer buffer{*pool_};
    return std::string{NullableEncoding<std::string_view>::encodeNullable(
        static_cast<uint32_t>(values.size()),
        encodedValues,
        encodedNulls,
        buffer)};
  }

  // Copies 'bytes' into a new buffer, which can own the chunks of a decoder.
  velox::BufferPtr copyToBuffer(std::string_view bytes) {
    auto buffer =
        velox::AlignedBuffer::allocate<char>(bytes.size(), pool_.get());
    std::copy(bytes.begin(), bytes.end(), buffer->asMutable<char>());
    return buffer;
  }

  // Creates the view of 'encoded' with 'options' and 'inputStringBuffer', as
  // the factory of a StreamReader creates it. The test keeps
  // 'inputStringBuffer' alive while the view is bound.
  std::shared_ptr<const EncodingView> makeView(
      std::string_view encoded,
      velox::Buffer* inputStringBuffer,
      Encoding::Options options) {
    options.inputStringBuffer = inputStringBuffer;
    return createEncodingView(encoded, pool_.get(), options);
  }

  std::unique_ptr<EncodingViewDecoder> makeDecoder(
      std::span<const std::string_view> chunks) {
    return makeDecoder(chunks, /*chunkBufferHolders=*/{});
  }

  std::unique_ptr<EncodingViewDecoder> makeDecoder(std::string_view stream) {
    return makeDecoder(std::span{&stream, 1});
  }

  // Returns a decoder over 'chunks' whose buffers 'chunkBufferHolders' hold.
  std::unique_ptr<EncodingViewDecoder> makeDecoder(
      std::span<const std::string_view> chunks,
      std::span<const velox::BufferPtr> chunkBufferHolders) {
    return std::make_unique<EncodingViewDecoder>(
        chunks,
        chunkBufferHolders,
        pool_.get(),
        /*bufferPool=*/nullptr,
        [this](std::string_view encoded, velox::Buffer* inputStringBuffer) {
          return makeView(encoded, inputStringBuffer, Encoding::Options{});
        });
  }

  // Returns a decoder whose factory creates views with 'options', and keeps
  // them in 'views' too, so that a test can ask them what they hold.
  std::unique_ptr<EncodingViewDecoder> makeDecoder(
      std::span<const std::string_view> chunks,
      std::span<const velox::BufferPtr> chunkBufferHolders,
      const Encoding::Options& options,
      std::vector<std::shared_ptr<const EncodingView>>& views) {
    return std::make_unique<EncodingViewDecoder>(
        chunks,
        chunkBufferHolders,
        pool_.get(),
        /*bufferPool=*/nullptr,
        [this, options, &views](
            std::string_view encoded, velox::Buffer* inputStringBuffer) {
          auto view = makeView(encoded, inputStringBuffer, options);
          views.push_back(view);
          return view;
        });
  }

  // Returns the options of the cached views of the chunks of one stream, which
  // decode strings into buffers that they share. The test keeps the buffers
  // until it ends.
  Encoding::Options sharedBufferOptions() {
    Encoding::Options options{.cacheEncodingView = true};
    options.decodedStringBuffers =
        decodedStringBuffers_
            .emplace_back(std::make_unique<DecodedStringBuffers>(pool_.get()))
            .get();
    return options;
  }

  // Returns a factory that records every view it creates in 'views'.
  EncodingViewDecoder::EncodingViewFactory makeTrackingFactory(
      std::vector<std::weak_ptr<const EncodingView>>& views) {
    return [this, &views](
               std::string_view encoded, velox::Buffer* inputStringBuffer) {
      auto view = makeView(encoded, inputStringBuffer, Encoding::Options{});
      views.push_back(view);
      return view;
    };
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;

  // The buffers of sharedBufferOptions(), which the views do not own. Declared
  // after pool_, so that they free their last buffers before the pool goes.
  std::vector<std::unique_ptr<DecodedStringBuffers>> decodedStringBuffers_;
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
      std::array<std::string_view, 1>{second},
      /*chunkBufferHolders=*/{},
      /*bufferPool=*/nullptr);

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
      /*chunkBufferHolders=*/{},
      pool_.get(),
      &firstPool,
      [this](std::string_view encoded, velox::Buffer* inputStringBuffer) {
        return makeView(encoded, inputStringBuffer, Encoding::Options{});
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
  decoder.rebind(
      std::array<std::string_view, 1>{second},
      /*chunkBufferHolders=*/{},
      &secondPool);
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
      /*chunkBufferHolders=*/{},
      pool_.get(),
      /*bufferPool=*/nullptr,
      makeTrackingFactory(views)};
  ASSERT_THAT(isExpired(views), ::testing::ElementsAre(false, false));

  decoder.release();
  EXPECT_THAT(isExpired(views), ::testing::ElementsAre(true, true));

  decoder.rebind(
      std::array<std::string_view, 1>{second},
      /*chunkBufferHolders=*/{},
      /*bufferPool=*/nullptr);
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
      /*chunkBufferHolders=*/{},
      pool_.get(),
      /*bufferPool=*/nullptr,
      makeTrackingFactory(views)};

  // The last chunk fails after the views of the earlier chunks are created.
  NIMBLE_ASSERT_FILE_THROW(
      decoder.rebind(
          std::array<std::string_view, 3>{first, second, otherType},
          /*chunkBufferHolders=*/{},
          /*bufferPool=*/nullptr),
      "Encoding chunks of a stream must store the same data type");
  EXPECT_THAT(isExpired(views), ::testing::ElementsAre(true, true, true, true));

  decoder.rebind(
      std::array<std::string_view, 1>{second},
      /*chunkBufferHolders=*/{},
      /*bufferPool=*/nullptr);
  EXPECT_EQ(decoder.remainingRows(), 2);
}

TEST_F(EncodingViewDecoderTest, rebindUsesConstructorFactory) {
  const auto first = encode<int64_t>(std::vector<int64_t>{10, 11, 12});
  const auto second = encode<int64_t>(std::vector<int64_t>{20, 21});
  uint32_t numViews{0};
  EncodingViewDecoder decoder{
      std::array<std::string_view, 1>{first},
      /*chunkBufferHolders=*/{},
      pool_.get(),
      /*bufferPool=*/nullptr,
      [&](std::string_view encoded, velox::Buffer* inputStringBuffer) {
        ++numViews;
        return makeView(encoded, inputStringBuffer, Encoding::Options{});
      }};
  ASSERT_EQ(numViews, 1);

  decoder.rebind(
      std::array<std::string_view, 2>{second, first},
      /*chunkBufferHolders=*/{},
      /*bufferPool=*/nullptr);
  decoder.rebind(
      std::array<std::string_view, 1>{first},
      /*chunkBufferHolders=*/{},
      /*bufferPool=*/nullptr);

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
  const auto first = encodeUncompressed<std::string_view>(
      std::vector<std::string_view>{"first string", "unused"},
      EncodingType::Trivial);
  const auto second = encodeUncompressed<std::string_view>(
      std::vector<std::string_view>{"second string"}, EncodingType::Trivial);
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

  // No buffer holds the chunks, so the view of each copies its strings into a
  // buffer that it passes to the read.
  EXPECT_THAT(output, ::testing::ElementsAre("first string", "second string"));
  ASSERT_EQ(stringBuffers.size(), 2);
  EXPECT_EQ(
      (std::string_view{
          stringBuffers[0]->as<char>(), stringBuffers[0]->size()}),
      "first string");
  EXPECT_EQ(
      (std::string_view{
          stringBuffers[1]->as<char>(), stringBuffers[1]->size()}),
      "second string");
}

TEST_F(EncodingViewDecoderTest, keepsStringsInChunkBufferHolders) {
  const std::vector<std::string_view> values{
      "first string",
      "second string",
      "third string",
      "fourth string",
      "fifth string",
      "sixth string",
      "seventh string",
      "eighth string",
  };
  const auto stream =
      encodeUncompressed<std::string_view>(values, EncodingType::Trivial);
  const auto holder = copyToBuffer(stream);
  const std::array<velox::BufferPtr, 1> holders{holder};
  const std::array<std::string_view, 1> chunks{
      std::string_view{holder->as<char>(), holder->size()}};
  {
    // Reads half of the rows of the chunk, so the output keeps the holder.
    std::array<std::string_view, 4> output{};
    std::vector<velox::BufferPtr> stringBuffers;
    {
      auto decoder = makeDecoder(chunks, holders);
      const std::array<RowRange, 1> ranges{{{2, 6}}};
      EXPECT_EQ(
          decoder->read(
              ranges,
              DataType::String,
              output.data(),
              /*getOutputNulls=*/nullptr,
              stringBuffers),
          4);
    }
    EXPECT_THAT(
        output,
        ::testing::ElementsAre(
            "third string", "fourth string", "fifth string", "sixth string"));
    EXPECT_THAT(stringBuffers, ::testing::ElementsAre(holder));
    for (const auto value : output) {
      EXPECT_TRUE(liesIn(value, {holder})) << value;
    }
  }
  {
    // Reads half of the rows of the chunk by index, so the output keeps the
    // holder whatever the order of the rows.
    struct TestParam {
      std::string testName;
      std::array<uint32_t, 4> rows;
      std::array<std::string_view, 4> expected;

      std::string debugString() const {
        return testName;
      }
    };
    const std::vector<TestParam> testSettings{
        {"sorted",
         {2, 3, 4, 5},
         {"third string", "fourth string", "fifth string", "sixth string"}},
        {"reversed",
         {5, 4, 3, 2},
         {"sixth string", "fifth string", "fourth string", "third string"}},
    };
    for (const auto& testData : testSettings) {
      SCOPED_TRACE(testData.debugString());
      std::array<std::string_view, 4> output{};
      std::vector<velox::BufferPtr> stringBuffers;
      {
        auto decoder = makeDecoder(chunks, holders);
        EXPECT_EQ(
            decoder->read(
                testData.rows,
                DataType::String,
                output.data(),
                /*getOutputNulls=*/nullptr,
                stringBuffers),
            4);
      }
      EXPECT_THAT(output, ::testing::ElementsAreArray(testData.expected));
      EXPECT_THAT(stringBuffers, ::testing::ElementsAre(holder));
    }
  }
  {
    // Reads one row of eight, too few to keep the chunk alive for it.
    std::array<std::string_view, 1> output{};
    std::vector<velox::BufferPtr> stringBuffers;
    auto decoder = makeDecoder(chunks, holders);
    const std::array<uint32_t, 1> rows{5};
    EXPECT_EQ(
        decoder->read(
            rows,
            DataType::String,
            output.data(),
            /*getOutputNulls=*/nullptr,
            stringBuffers),
        1);
    EXPECT_THAT(output, ::testing::ElementsAre("sixth string"));
    ASSERT_EQ(stringBuffers.size(), 1);
    EXPECT_NE(stringBuffers[0], holder);
    EXPECT_TRUE(liesIn(output[0], stringBuffers));
  }
}

TEST_F(EncodingViewDecoderTest, keepsDecompressedStringsAfterTheDecoder) {
  constexpr int kNumStrings{500};
  std::vector<std::string> strings;
  strings.reserve(kNumStrings);
  for (int i = 0; i < kNumStrings; ++i) {
    strings.push_back(fmt::format("compressible string value {:04d}", i % 10));
  }
  const std::vector<std::string_view> values{strings.begin(), strings.end()};
  const auto stream = encode<std::string_view>(values);
  std::vector<std::string_view> output(values.size());
  std::vector<velox::BufferPtr> stringBuffers;
  {
    // No buffer holds the chunk, and the view's decompressed payload holds
    // the strings, which the view passes to the read, so the output holds the
    // payload, with no copy.
    std::vector<std::shared_ptr<const EncodingView>> views;
    const std::array<std::string_view, 1> chunks{stream};
    auto decoder = makeDecoder(
        chunks, /*chunkBufferHolders=*/{}, Encoding::Options{}, views);
    const std::array<RowRange, 1> ranges{
        {{0, static_cast<uint32_t>(values.size())}}};
    EXPECT_EQ(
        decoder->read(
            ranges,
            DataType::String,
            output.data(),
            /*getOutputNulls=*/nullptr,
            stringBuffers),
        values.size());

    // Another read of the view passes the same buffer, which a copy would
    // not be.
    ASSERT_EQ(views.size(), 1);
    std::vector<std::string_view> again(values.size());
    std::vector<velox::BufferPtr> payloads;
    views[0]->read(
        ranges,
        [](uint32_t) {},
        [&payloads](velox::BufferPtr buffer) {
          payloads.push_back(std::move(buffer));
        },
        again.data());
    ASSERT_EQ(payloads.size(), 1);
    EXPECT_THAT(stringBuffers, ::testing::ElementsAre(payloads[0]));
  }

  EXPECT_THAT(output, ::testing::ElementsAreArray(values));
  for (const auto value : output) {
    EXPECT_TRUE(liesIn(value, stringBuffers)) << value;
  }
}

TEST_F(EncodingViewDecoderTest, copiesStringsOfChunksWithoutBufferHolder) {
  const auto first = encodeUncompressed<std::string_view>(
      std::vector<std::string_view>{"first string", "second string"},
      EncodingType::Trivial);
  const auto second = encodeUncompressed<std::string_view>(
      std::vector<std::string_view>{"third string", "fourth string"},
      EncodingType::Trivial);

  // Only the first chunk has a buffer holder. The second stays in a
  // std::string, so its view copies its strings into a buffer that it passes.
  const auto firstHolder = copyToBuffer(first);
  const std::array<velox::BufferPtr, 2> holders{firstHolder, nullptr};
  const std::array<std::string_view, 2> chunks{
      std::string_view{firstHolder->as<char>(), firstHolder->size()}, second};
  std::array<std::string_view, 4> output{};
  std::vector<velox::BufferPtr> stringBuffers;
  {
    auto decoder = makeDecoder(chunks, holders);
    const std::array<RowRange, 1> ranges{{{0, 4}}};
    EXPECT_EQ(
        decoder->read(
            ranges,
            DataType::String,
            output.data(),
            /*getOutputNulls=*/nullptr,
            stringBuffers),
        4);
  }

  EXPECT_THAT(
      output,
      ::testing::ElementsAre(
          "first string", "second string", "third string", "fourth string"));
  ASSERT_EQ(stringBuffers.size(), 2);
  EXPECT_EQ(stringBuffers[0], firstHolder);
  EXPECT_EQ(
      (std::string_view{
          stringBuffers[1]->as<char>(), stringBuffers[1]->size()}),
      "third stringfourth string");
  for (const auto value : output) {
    EXPECT_TRUE(liesIn(value, stringBuffers)) << value;
  }
}

TEST_F(EncodingViewDecoderTest, retainsStringsOfEveryEncoding) {
  constexpr uint32_t kNumRows{600};
  const auto strings = makeFsstStrings(kNumRows);
  std::vector<std::string> compressibleStrings;
  compressibleStrings.reserve(10);
  for (int i = 0; i < 10; ++i) {
    compressibleStrings.push_back(
        fmt::format("compressible string value {:04d}", i));
  }
  const std::array<std::string_view, 4> fewStrings{
      "alpha", "", "gamma value", "delta value"};

  // Returns valueOf(row) for each row.
  const auto makeValues =
      [&](const std::function<std::optional<std::string_view>(uint32_t)>&
              valueOf) {
        std::vector<std::optional<std::string_view>> values;
        values.reserve(kNumRows);
        for (uint32_t row{0}; row < kNumRows; ++row) {
          values.push_back(valueOf(row));
        }
        return values;
      };
  const auto distinctValues = makeValues([&](uint32_t row) {
    return row % 7 == 3 ? std::string_view{} : std::string_view{strings[row]};
  });

  struct TestParam {
    std::string testName;
    std::vector<std::optional<std::string_view>> values;
    EncodingType encodingType;
    bool compressed{false};
    bool cacheEncodingView{false};

    // Whether the views return strings of the bytes of their chunks, rather
    // than strings that they decompress or decode.
    bool readsChunkBytes{false};

    // Buffers that a read passes for each chunk when no buffer holds the
    // chunk: the copies of its strings, or the buffer that its view
    // decompressed or decoded them into.
    size_t numBuffersPerChunk{1};

    std::string debugString() const {
      return testName;
    }
  };
  const std::vector<TestParam> testSettings{
      {.testName = "trivial",
       .values = distinctValues,
       .encodingType = EncodingType::Trivial,
       .readsChunkBytes = true},
      {.testName = "compressed trivial",
       .values = makeValues([&](uint32_t row) {
         return std::string_view{compressibleStrings[row % 10]};
       }),
       .encodingType = EncodingType::Trivial,
       .compressed = true},
      {.testName = "constant",
       .values = makeValues([](uint32_t) { return "constant value"; }),
       .encodingType = EncodingType::Constant,
       .readsChunkBytes = true},
      {.testName = "mainly constant",
       .values = makeValues([&](uint32_t row) {
         return row % 10 == 7 ? std::string_view{strings[row]}
                              : std::string_view{"common value"};
       }),
       .encodingType = EncodingType::MainlyConstant,
       .readsChunkBytes = true,

       // The copy of the other values, and the copy of the common value.
       .numBuffersPerChunk = 2},
      {.testName = "dictionary",
       .values = makeValues([&](uint32_t row) { return fewStrings[row % 4]; }),
       .encodingType = EncodingType::Dictionary,
       .readsChunkBytes = true},
      {.testName = "rle",
       .values =
           makeValues([&](uint32_t row) { return fewStrings[row / 25 % 4]; }),
       .encodingType = EncodingType::RLE,
       .readsChunkBytes = true},
      {.testName = "nullable trivial",
       .values =
           makeValues([&](uint32_t row) -> std::optional<std::string_view> {
             return row % 5 == 0 ? std::nullopt : distinctValues[row];
           }),
       .encodingType = EncodingType::Trivial,
       .readsChunkBytes = true},
      {.testName = "fsst",
       .values = distinctValues,
       .encodingType = EncodingType::Fsst},
      {.testName = "cached fsst",
       .values = distinctValues,
       .encodingType = EncodingType::Fsst,
       .cacheEncodingView = true},
      {.testName = "nullable fsst",
       .values =
           makeValues([&](uint32_t row) -> std::optional<std::string_view> {
             return row % 5 == 0 ? std::nullopt : distinctValues[row];
           }),
       .encodingType = EncodingType::Fsst},
  };

  // Reads two ranges across the chunks, or every other row from the last to
  // the first. Each selects enough rows that the read holds the buffers of
  // its strings instead of copying them.
  const std::array<RowRange, 2> ranges{{{0, 250}, {260, kNumRows}}};
  std::vector<uint32_t> rangeRows;
  for (const auto& range : ranges) {
    for (auto row = range.startRow; row < range.endRow; ++row) {
      rangeRows.push_back(row);
    }
  }
  std::vector<uint32_t> rows;
  for (uint32_t i{0}; i < kNumRows / 2; ++i) {
    rows.push_back(kNumRows - 1 - 2 * i);
  }

  for (const auto& testData : testSettings) {
    const auto compressionOptions =
        testData.compressed ? CompressionOptions{} : uncompressedOptions();
    const std::array<std::string, 2> encodedChunks{
        encodeStrings(
            std::span{testData.values}.first(kNumRows / 2),
            testData.encodingType,
            compressionOptions),
        encodeStrings(
            std::span{testData.values}.subspan(kNumRows / 2),
            testData.encodingType,
            compressionOptions)};
    const bool nullable = std::any_of(
        testData.values.begin(), testData.values.end(), [](const auto& value) {
          return !value.has_value();
        });
    for (const auto& encoded : encodedChunks) {
      ASSERT_EQ(
          createEncodingView(encoded, pool_.get())->encodingType(),
          nullable ? EncodingType::Nullable : testData.encodingType)
          << testData.debugString();
    }
    for (const bool holdsChunks : {false, true}) {
      for (const bool readsRows : {false, true}) {
        SCOPED_TRACE(
            fmt::format(
                "{}, holdsChunks {}, readsRows {}",
                testData.debugString(),
                holdsChunks,
                readsRows));
        const auto& selectedRows = readsRows ? rows : rangeRows;
        std::vector<std::string_view> output(selectedRows.size());
        std::vector<uint64_t> outputNulls(
            velox::bits::nwords(selectedRows.size()), ~uint64_t{0});
        std::vector<velox::BufferPtr> stringBuffers;
        uint32_t numNonNulls{0};
        {
          std::vector<velox::BufferPtr> holders;
          std::vector<std::string_view> chunks;
          for (const auto& encoded : encodedChunks) {
            holders.push_back(copyToBuffer(encoded));
            chunks.emplace_back(
                holders.back()->as<char>(), holders.back()->size());
          }
          {
            std::vector<std::shared_ptr<const EncodingView>> views;
            auto decoder = makeDecoder(
                chunks,
                holdsChunks ? std::span<const velox::BufferPtr>{holders}
                            : std::span<const velox::BufferPtr>{},
                Encoding::Options{
                    .cacheEncodingView = testData.cacheEncodingView},
                views);
            const auto getOutputNulls = [&outputNulls]() {
              return outputNulls.data();
            };
            numNonNulls = readsRows ? decoder->read(
                                          rows,
                                          DataType::String,
                                          output.data(),
                                          getOutputNulls,
                                          stringBuffers)
                                    : decoder->read(
                                          ranges,
                                          DataType::String,
                                          output.data(),
                                          getOutputNulls,
                                          stringBuffers);
          }
          if (testData.readsChunkBytes && holdsChunks) {
            EXPECT_THAT(
                stringBuffers,
                ::testing::UnorderedElementsAre(holders[0], holders[1]));
          } else {
            // Holds the buffers of each chunk, which no holder holds.
            EXPECT_EQ(stringBuffers.size(), 2 * testData.numBuffersPerChunk);
            EXPECT_THAT(
                stringBuffers, ::testing::Not(::testing::Contains(holders[0])));
            EXPECT_THAT(
                stringBuffers, ::testing::Not(::testing::Contains(holders[1])));
          }
        }

        // The strings stay valid after the decoder, its views and the buffers
        // of the chunks are gone.
        uint32_t numExpectedNonNulls{0};
        for (size_t i{0}; i < selectedRows.size(); ++i) {
          const auto row = selectedRows[i];
          const auto& expected = testData.values[row];
          if (!expected.has_value()) {
            EXPECT_FALSE(velox::bits::isBitSet(outputNulls.data(), i))
                << "row " << row;
            continue;
          }
          ++numExpectedNonNulls;
          EXPECT_EQ(output[i], *expected) << "row " << row;
          EXPECT_TRUE(expected->empty() || liesIn(output[i], stringBuffers))
              << "row " << row;
        }
        EXPECT_EQ(numNonNulls, numExpectedNonNulls);
      }
    }
  }
}

TEST_F(EncodingViewDecoderTest, rejectsInvalidChunkBufferHolders) {
  const auto holder = copyToBuffer(
      encodeUncompressed<std::string_view>(
          std::vector<std::string_view>{"a string"}, EncodingType::Trivial));
  const std::array<std::string_view, 1> chunks{
      std::string_view{holder->as<char>(), holder->size()}};
  {
    const std::array<velox::BufferPtr, 2> holders{holder, holder};
    NIMBLE_ASSERT_USER_THROW(
        makeDecoder(chunks, holders),
        "Chunk buffer holders must align with the chunks");
  }
  {
    // Holds the chunk but its first byte.
    const std::array<velox::BufferPtr, 1> holders{velox::Buffer::slice<char>(
        holder, /*offset=*/1, holder->size() - 1, pool_.get())};
    NIMBLE_ASSERT_USER_THROW(
        makeDecoder(chunks, holders),
        "A chunk buffer holder must not start after its chunk");
  }
  {
    // Holds the chunk but its last byte.
    const std::array<velox::BufferPtr, 1> holders{velox::Buffer::slice<char>(
        holder, /*offset=*/0, holder->size() - 1, pool_.get())};
    NIMBLE_ASSERT_USER_THROW(
        makeDecoder(chunks, holders),
        "A chunk buffer holder must not end before its chunk");
  }
  {
    // A new binding checks its holders too.
    auto decoder = makeDecoder(chunks);
    const std::array<velox::BufferPtr, 2> holders{holder, holder};
    NIMBLE_ASSERT_USER_THROW(
        decoder->rebind(chunks, holders, /*bufferPool=*/nullptr),
        "Chunk buffer holders must align with the chunks");
  }
}

DEBUG_ONLY_TEST_F(
    EncodingViewDecoderTest,
    rejectsStringsOutsideThePassedBuffers) {
  const auto stream = encodeUncompressed<std::string_view>(
      std::vector<std::string_view>{"first string", "second string"},
      EncodingType::Trivial);
  EncodingViewDecoder decoder{
      std::array<std::string_view, 1>{stream},
      /*chunkBufferHolders=*/{},
      pool_.get(),
      /*bufferPool=*/nullptr,
      [this](std::string_view encoded, velox::Buffer* /*inputStringBuffer*/) {
        return std::make_shared<OutsideBufferView>(encoded, pool_.get());
      }};
  const std::array<RowRange, 1> ranges{{{0, 2}}};
  std::array<std::string_view, 2> output{};
  std::vector<velox::BufferPtr> stringBuffers;
  NIMBLE_ASSERT_THROW(
      decoder.read(
          ranges,
          DataType::String,
          output.data(),
          /*getOutputNulls=*/nullptr,
          stringBuffers),
      "A view must pass buffers that hold every string it returns");
}

TEST_F(EncodingViewDecoderTest, keepsFsstStringsInDecodedBuffers) {
  // Few enough strings that the view decodes them into its first buffer, of
  // 32 KB.
  constexpr int kNumStrings{600};
  std::vector<std::string> strings;
  strings.reserve(kNumStrings);
  for (int i = 0; i < kNumStrings; ++i) {
    strings.push_back(fmt::format("repeated fsst prefix value {:06d}", i));
  }
  const std::vector<std::string_view> values{strings.begin(), strings.end()};
  const auto stream =
      encodeUncompressed<std::string_view>(values, EncodingType::Fsst);
  ASSERT_EQ(
      createEncodingView(stream, pool_.get())->encodingType(),
      EncodingType::Fsst);
  std::vector<std::string_view> output(values.size());
  std::vector<velox::BufferPtr> stringBuffers;
  {
    auto decoder = makeDecoder(stream);
    const std::array<RowRange, 1> ranges{
        {{0, static_cast<uint32_t>(values.size())}}};
    EXPECT_EQ(
        decoder->read(
            ranges,
            DataType::String,
            output.data(),
            /*getOutputNulls=*/nullptr,
            stringBuffers),
        values.size());
  }

  // The values stay valid after the decoder and its views are gone. The view
  // decodes every string of the read into one buffer, which it passes.
  EXPECT_THAT(output, ::testing::ElementsAreArray(values));
  ASSERT_EQ(stringBuffers.size(), 1);
  for (const auto value : output) {
    EXPECT_TRUE(liesIn(value, stringBuffers)) << value;
  }
}

TEST_F(EncodingViewDecoderTest, holdsBufferThatFsstViewsDecodeReadInto) {
  // Few enough strings that those of both chunks fit in the first buffer, of
  // 32 KB.
  constexpr uint32_t kNumStrings{600};
  const auto strings = makeFsstStrings(kNumStrings);
  const std::vector<std::string_view> values{strings.begin(), strings.end()};
  const auto first = encodeUncompressed<std::string_view>(
      std::span{values}.first(360), EncodingType::Fsst);
  const auto second = encodeUncompressed<std::string_view>(
      std::span{values}.subspan(360), EncodingType::Fsst);
  const std::array<std::string_view, 2> chunks{first, second};
  const auto options = sharedBufferOptions();
  {
    // Reads every row. The views decode the strings into the buffer they
    // share for the read alone, which the output holds.
    std::vector<std::string_view> output(values.size());
    std::vector<velox::BufferPtr> stringBuffers;
    {
      std::vector<std::shared_ptr<const EncodingView>> views;
      auto decoder =
          makeDecoder(chunks, /*chunkBufferHolders=*/{}, options, views);
      const std::array<RowRange, 1> ranges{{{0, kNumStrings}}};
      EXPECT_EQ(
          decoder->read(
              ranges,
              DataType::String,
              output.data(),
              /*getOutputNulls=*/nullptr,
              stringBuffers),
          values.size());
    }
    EXPECT_THAT(output, ::testing::ElementsAreArray(values));
    ASSERT_EQ(stringBuffers.size(), 1);
    for (const auto value : output) {
      EXPECT_TRUE(liesIn(value, stringBuffers)) << value;
    }
  }
  {
    // Reads one row in eight, too few to hold buffers for it, so the output
    // copies the strings.
    std::vector<uint32_t> rows;
    for (uint32_t row{0}; row < kNumStrings; row += 8) {
      rows.push_back(row);
    }
    std::vector<std::string_view> output(rows.size());
    std::vector<velox::BufferPtr> stringBuffers;
    {
      std::vector<std::shared_ptr<const EncodingView>> views;
      auto decoder =
          makeDecoder(chunks, /*chunkBufferHolders=*/{}, options, views);
      EXPECT_EQ(
          decoder->read(
              rows,
              DataType::String,
              output.data(),
              /*getOutputNulls=*/nullptr,
              stringBuffers),
          rows.size());
    }
    uint64_t numValueBytes{0};
    for (size_t i{0}; i < rows.size(); ++i) {
      EXPECT_EQ(output[i], values[rows[i]]) << "row " << rows[i];
      numValueBytes += output[i].size();
    }
    ASSERT_EQ(stringBuffers.size(), 1);
    EXPECT_EQ(stringBuffers[0]->size(), numValueBytes);
  }
}

TEST_F(EncodingViewDecoderTest, holdsBuffersOfTrivialAndCachedFsstChunks) {
  const auto strings = makeFsstStrings(1'000);
  const std::vector<std::string_view> values{strings.begin(), strings.end()};
  const auto trivial = encodeUncompressed<std::string_view>(
      std::span{values}.first(400), EncodingType::Trivial);
  const auto fsst = encodeUncompressed<std::string_view>(
      std::span{values}.subspan(400), EncodingType::Fsst);
  const auto holder = copyToBuffer(trivial);
  const std::array<velox::BufferPtr, 2> holders{holder, nullptr};
  const std::array<std::string_view, 2> chunks{
      std::string_view{holder->as<char>(), holder->size()}, fsst};
  std::vector<std::string_view> output(values.size());
  std::vector<velox::BufferPtr> stringBuffers;
  {
    std::vector<std::shared_ptr<const EncodingView>> views;
    auto decoder = makeDecoder(chunks, holders, sharedBufferOptions(), views);
    const std::array<RowRange, 1> ranges{
        {{0, static_cast<uint32_t>(values.size())}}};
    EXPECT_EQ(
        decoder->read(
            ranges,
            DataType::String,
            output.data(),
            /*getOutputNulls=*/nullptr,
            stringBuffers),
        values.size());
  }

  // The Trivial view returns strings of the chunk's bytes and passes the
  // holder of the chunk, and the FSST view passes the buffer it decoded into.
  EXPECT_THAT(output, ::testing::ElementsAreArray(values));
  ASSERT_EQ(stringBuffers.size(), 2);
  EXPECT_EQ(stringBuffers[0], holder);
  for (const auto value : output) {
    EXPECT_TRUE(liesIn(value, stringBuffers)) << value;
  }
}

TEST_F(EncodingViewDecoderTest, dropsStringStateAfterEachRead) {
  const auto strings = makeFsstStrings(1'000);
  const std::vector<std::string_view> values{strings.begin(), strings.end()};
  const auto trivial = encodeUncompressed<std::string_view>(
      std::span{values}.first(400), EncodingType::Trivial);
  const auto fsst = encodeUncompressed<std::string_view>(
      std::span{values}.subspan(400), EncodingType::Fsst);
  const auto holder = copyToBuffer(trivial);
  const std::array<velox::BufferPtr, 2> holders{holder, nullptr};
  const std::array<std::string_view, 2> chunks{
      std::string_view{holder->as<char>(), holder->size()}, fsst};
  std::vector<std::shared_ptr<const EncodingView>> views;
  auto decoder = makeDecoder(chunks, holders, sharedBufferOptions(), views);
  {
    // The Trivial view passes the holder of its chunk, and the FSST view the
    // buffer it decoded into.
    std::vector<std::string_view> output(values.size());
    std::vector<velox::BufferPtr> stringBuffers;
    const std::array<RowRange, 1> ranges{
        {{0, static_cast<uint32_t>(values.size())}}};
    EXPECT_EQ(
        decoder->read(
            ranges,
            DataType::String,
            output.data(),
            /*getOutputNulls=*/nullptr,
            stringBuffers),
        values.size());
    EXPECT_THAT(output, ::testing::ElementsAreArray(values));
    ASSERT_EQ(stringBuffers.size(), 2);
    EXPECT_EQ(stringBuffers[0], holder);
  }
  {
    // A read of the FSST rows alone holds the buffers that it decodes into,
    // without the holder of the Trivial chunk that the previous read held.
    const auto fsstValues = std::span{values}.subspan(400);
    std::vector<std::string_view> output(fsstValues.size());
    std::vector<velox::BufferPtr> stringBuffers;
    const std::array<RowRange, 1> ranges{
        {{400, static_cast<uint32_t>(values.size())}}};
    EXPECT_EQ(
        decoder->read(
            ranges,
            DataType::String,
            output.data(),
            /*getOutputNulls=*/nullptr,
            stringBuffers),
        fsstValues.size());
    EXPECT_THAT(output, ::testing::ElementsAreArray(fsstValues));
    EXPECT_THAT(stringBuffers, ::testing::Not(::testing::Contains(holder)));
    for (const auto value : output) {
      EXPECT_TRUE(liesIn(value, stringBuffers)) << value;
    }
  }
}

TEST_F(EncodingViewDecoderTest, holdsBufferThatNullableFsstViewsDecodeInto) {
  // Few enough rows that their strings fit in the first buffer, of 32 KB.
  constexpr uint32_t kNumRows{800};
  const auto strings = makeFsstStrings(kNumRows);
  std::vector<std::optional<std::string_view>> values;
  values.reserve(kNumRows);
  for (uint32_t row{0}; row < kNumRows; ++row) {
    values.push_back(
        row % 5 == 0 ? std::nullopt
                     : std::optional<std::string_view>{strings[row]});
  }
  const auto stream =
      encodeStrings(values, EncodingType::Fsst, uncompressedOptions());
  const std::array<std::string_view, 1> chunks{stream};
  std::vector<std::string_view> output(kNumRows);
  std::vector<uint64_t> outputNulls(velox::bits::nwords(kNumRows));
  std::vector<velox::BufferPtr> stringBuffers;
  {
    // The Nullable view passes the read's addStringBuffer to its FSST child.
    std::vector<std::shared_ptr<const EncodingView>> views;
    auto decoder = makeDecoder(
        chunks, /*chunkBufferHolders=*/{}, sharedBufferOptions(), views);
    const std::array<RowRange, 1> ranges{{{0, kNumRows}}};
    EXPECT_EQ(
        decoder->read(
            ranges,
            DataType::String,
            output.data(),
            [&outputNulls]() { return outputNulls.data(); },
            stringBuffers),
        kNumRows - kNumRows / 5);
  }
  for (uint32_t row{0}; row < kNumRows; ++row) {
    ASSERT_EQ(velox::bits::isBitSet(outputNulls.data(), row), row % 5 != 0)
        << "row " << row;
    if (values[row].has_value()) {
      EXPECT_EQ(output[row], *values[row]) << "row " << row;
    }
  }
  ASSERT_EQ(stringBuffers.size(), 1);
  for (const auto value : output) {
    EXPECT_TRUE(value.empty() || liesIn(value, stringBuffers)) << value;
  }
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
      /*chunkBufferHolders=*/{},
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
      std::array<std::string_view, 1>{single},
      /*chunkBufferHolders=*/{},
      /*bufferPool=*/nullptr);
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
