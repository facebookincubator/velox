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

#include "velox/dwio/nimble/velox/selective/ChunkedDecoder.h"

#include "velox/common/base/Nulls.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/dwio/common/DirectBufferedInput.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingUtils.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/EncodingLayoutTestHelper.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestBase.h"
#include "velox/dwio/nimble/velox/ChunkedStreamWriter.h"

#include <folly/executors/IOThreadPoolExecutor.h>
#include <gtest/gtest.h>

namespace facebook::nimble {

using namespace facebook::velox;

class ChunkedDecoderTestHelper {
 public:
  explicit ChunkedDecoderTestHelper(ChunkedDecoder* decoder)
      : decoder_(decoder) {
    NIMBLE_CHECK_NOT_NULL(decoder_);
  }

  bool ensureInput(int size) {
    return decoder_->ensureInput(size);
  }

  std::string_view inputData() const {
    return std::string_view(decoder_->inputData_, decoder_->inputSize_);
  }

  void advanceInputData(int size) {
    decoder_->inputData_ += size;
    decoder_->inputSize_ -= size;
  }

  bool fromInputBuffer() const {
    return decoder_->fromInputBuffer();
  }

  int64_t inputBufferCapacity() const {
    return decoder_->inputBuffer_ ? decoder_->inputBuffer_->capacity() : 0;
  }

  const char* inputBufferStart() const {
    return decoder_->inputBuffer_ ? decoder_->inputBuffer_->as<char>()
                                  : nullptr;
  }

  const char* inputDataPtr() const {
    return decoder_->inputData_;
  }

 private:
  ChunkedDecoder* const decoder_;
};

namespace {

class ChunkedDecoderTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    leafPool_ = memory::MemoryManager::getInstance()->addLeafPool();
  }

  MemoryPool& pool() {
    return *leafPool_;
  }

  const EncodingFactory& encodingFactory() const {
    return encodingFactory_;
  }

 protected:
  std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};

 private:
  std::shared_ptr<MemoryPool> leafPool_;
  EncodingFactory encodingFactory_;
};

TEST_F(ChunkedDecoderTest, bufferedInput) {
  auto fileIoStats = std::make_shared<velox::io::IoStatistics>();
  constexpr size_t kFileSize = 1 << 12;
  constexpr size_t kLoadQuantum = 1 << 4;

  // Set up a file that has different content every load quantum for
  // the direct input stream.
  std::string fileContent{};
  size_t filePos = 0;
  size_t segmentLength;
  for (size_t segmentIdx = 0; filePos < kFileSize; ++segmentIdx) {
    segmentLength = std::min(kLoadQuantum, kFileSize - filePos);
    fileContent.append(segmentLength, 'a' + segmentIdx);
    filePos += segmentLength;
  }

  auto file = std::make_shared<InMemoryReadFile>(fileContent);
  dwio::common::ReaderOptions readerOpts{&pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setLoadQuantum(kLoadQuantum);
  auto executor = std::make_unique<folly::IOThreadPoolExecutor>(10, 10);
  auto input = std::make_unique<dwio::common::DirectBufferedInput>(
      file,
      dwio::common::MetricsLog::voidLog(),
      StringIdLease{},
      nullptr,
      StringIdLease{},
      fileIoStats,
      nullptr,
      executor.get(),
      readerOpts);

  auto chunkedDecoder = std::make_unique<nimble::ChunkedDecoder>(
      input->read(0, kFileSize, velox::dwio::common::LogType::TEST),
      nullptr,
      false,
      &encodingFactory(),
      &pool());
  ChunkedDecoderTestHelper helper(chunkedDecoder.get());
  helper.ensureInput(kFileSize);
  ASSERT_EQ(helper.inputData(), fileContent);
}

TEST_F(ChunkedDecoderTest, ensureInput) {
  // Test that ensureInput correctly buffers data when reading in small chunks.
  // SeekableArrayInputStream with block_size=1 simulates reading one byte at a
  // time.
  std::string data = "abcdefgh";
  ChunkedDecoder decoder(
      std::make_unique<dwio::common::SeekableArrayInputStream>(
          data.data(), data.size(), /*block_size=*/1),
      nullptr,
      false,
      &encodingFactory(),
      &pool());
  ChunkedDecoderTestHelper helper(&decoder);
  auto checkNext = [&](const std::string& expected) {
    helper.ensureInput(expected.size());
    ASSERT_GE(helper.inputData().size(), expected.size());
    ASSERT_EQ(helper.inputData().substr(0, expected.size()), expected);
    helper.advanceInputData(expected.size());
  };
  checkNext("ab");
  checkNext("c");
  checkNext("de");
  checkNext("f");
  checkNext("gh");
  ASSERT_TRUE(helper.inputData().empty());
  ASSERT_FALSE(helper.ensureInput(1));
}

// Verify that ensureInput correctly appends data when inputData_ is mid-buffer
// (the memmove-skip optimization in prepareInputBuffer fires).
// Regression test: the old code used inputBuffer_->asMutable<char>() +
// inputSize_ for the memcpy destination, which is wrong when inputData_ doesn't
// point to the start of inputBuffer_. The fix uses
// const_cast<char*>(inputData_) + inputSize_.
TEST_F(ChunkedDecoderTest, ensureInputMemmoveSkip) {
  // Build 32 bytes of deterministic data.
  std::string data(32, '\0');
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = 'a' + (i % 26);
  }

  // block_size = 3: Next() returns 3 bytes at a time, forcing ensureInput to
  // loop and accumulate data in inputBuffer_.
  ChunkedDecoder decoder(
      std::make_unique<dwio::common::SeekableArrayInputStream>(
          data.data(), data.size(), /*block_size=*/3),
      /*streamIndex=*/nullptr,
      /*decodeValuesWithNulls=*/false,
      &encodingFactory(),
      &pool());
  ChunkedDecoderTestHelper helper(&decoder);

  // Step 1: Fill the internal buffer with 9 bytes ("abcdefghi").
  // This forces inputBuffer_ allocation; Velox AlignedBuffer rounds capacity
  // up (typically to 64 bytes), so there is plenty of trailing space.
  helper.ensureInput(9);
  ASSERT_EQ(helper.inputData().substr(0, 9), data.substr(0, 9));

  // Step 2: Consume 3 bytes. inputData_ advances to bufStart+3, inputSize_=6.
  // Now inputData_ no longer points to the start of inputBuffer_.
  helper.advanceInputData(3);
  ASSERT_EQ(helper.inputData(), data.substr(3, 6));

  // Step 3: Request 9 bytes total (have 6, need 3 more from the stream).
  // prepareInputBuffer sees inputData_ mid-buffer with enough trailing space
  // and skips the memmove. Then Next() returns 3 more bytes that must be
  // appended right after the existing 6 bytes — i.e. at inputData_+6
  // (= bufStart+9), NOT at bufStart+6.
  // With the old bug, the memcpy destination was bufStart+inputSize_ =
  // bufStart+6, which overwrote the tail of the existing data, corrupting it.
  helper.ensureInput(9);
  ASSERT_EQ(helper.inputData().substr(0, 9), data.substr(3, 9));
}

// Test the memmove compaction path: inputData_ is mid-buffer and there is NOT
// enough trailing space, so prepareInputBuffer must memmove data to the front.
TEST_F(ChunkedDecoderTest, ensureInputMemmoveCompaction) {
  // Use a large enough dataset to exhaust trailing space.
  // AlignedBuffer typically rounds up to 64 bytes.
  const int kDataSize = 256;
  std::string data(kDataSize, '\0');
  for (int i = 0; i < kDataSize; ++i) {
    data[i] = 'a' + (i % 26);
  }

  // block_size = 10: Next() returns 10 bytes at a time.
  ChunkedDecoder decoder(
      std::make_unique<dwio::common::SeekableArrayInputStream>(
          data.data(), data.size(), /*block_size=*/10),
      /*streamIndex=*/nullptr,
      /*decodeValuesWithNulls=*/false,
      &encodingFactory(),
      &pool());
  ChunkedDecoderTestHelper helper(&decoder);

  // Step 1: Fill buffer with 30 bytes.
  helper.ensureInput(30);
  ASSERT_EQ(helper.inputData().substr(0, 30), data.substr(0, 30));
  const auto capacity = helper.inputBufferCapacity();
  ASSERT_TRUE(helper.fromInputBuffer());

  // Step 2: Consume most of the buffer, leaving just 2 bytes.
  helper.advanceInputData(28);
  ASSERT_EQ(helper.inputData(), data.substr(28, 2));

  // Step 3: Request more data than trailing space allows.
  // inputData_ is at bufStart+28, inputSize_=2, capacity ~64.
  // We request capacity bytes total — this exceeds trailing space,
  // forcing memmove compaction to the front.
  const auto requestSize = static_cast<int>(capacity - 20);
  helper.ensureInput(requestSize);
  ASSERT_EQ(
      helper.inputData().substr(0, requestSize), data.substr(28, requestSize));
  // After compaction, inputData_ should be at the start of the buffer.
  ASSERT_EQ(helper.inputDataPtr(), helper.inputBufferStart());
}

// Test the reallocation path: the requested size exceeds inputBuffer_ capacity,
// forcing allocation of a new larger buffer.
TEST_F(ChunkedDecoderTest, ensureInputReallocation) {
  const int kDataSize = 256;
  std::string data(kDataSize, '\0');
  for (int i = 0; i < kDataSize; ++i) {
    data[i] = 'a' + (i % 26);
  }

  // block_size = 5: Next() returns 5 bytes at a time.
  ChunkedDecoder decoder(
      std::make_unique<dwio::common::SeekableArrayInputStream>(
          data.data(), data.size(), /*block_size=*/5),
      /*streamIndex=*/nullptr,
      /*decodeValuesWithNulls=*/false,
      &encodingFactory(),
      &pool());
  ChunkedDecoderTestHelper helper(&decoder);

  // Step 1: Fill buffer with 10 bytes to get initial allocation.
  helper.ensureInput(10);
  ASSERT_EQ(helper.inputData().substr(0, 10), data.substr(0, 10));
  const auto initialCapacity = helper.inputBufferCapacity();
  ASSERT_GT(initialCapacity, 0);

  // Step 2: Consume 5 bytes.
  helper.advanceInputData(5);
  ASSERT_EQ(helper.inputData(), data.substr(5, 5));

  // Step 3: Request much more than the current buffer capacity.
  // This forces reallocation: a new larger buffer is allocated,
  // existing data is copied, and inputData_ is reset to the new buffer start.
  const auto bigRequest = static_cast<int>(initialCapacity + 50);
  helper.ensureInput(bigRequest);
  ASSERT_EQ(
      helper.inputData().substr(0, bigRequest), data.substr(5, bigRequest));
  // Buffer should have grown.
  ASSERT_GT(helper.inputBufferCapacity(), initialCapacity);
  // After reallocation, inputData_ should be at the start of the new buffer.
  ASSERT_EQ(helper.inputDataPtr(), helper.inputBufferStart());
}

// Test fixture for ChunkedDecoder data operations with parameterized
// stream index support.
class ChunkedDecoderDataTest : public index::test::ClusterIndexTestBase,
                               public testing::WithParamInterface<bool> {
 protected:
  using Stream = index::test::ClusterIndexTestBase::Stream;
  using KeyStream = index::test::ClusterIndexTestBase::KeyStream;
  using Stripe = index::test::ClusterIndexTestBase::Stripe;
  using IndexBuffers = index::test::ClusterIndexTestBase::IndexBuffers;

  void SetUp() override {}

  const EncodingFactory& encodingFactory() const {
    return encodingFactory_;
  }

  bool useStreamIndex() const {
    return GetParam();
  }

  // Encodes integer values into a chunked stream format.
  // Returns the encoded stream data and chunk metadata (row counts and
  // offsets).
  struct ChunkInfo {
    uint32_t rowCount;
    uint32_t streamOffset;
  };

  template <typename T>
  std::pair<std::string, std::vector<ChunkInfo>> encodeChunkedStream(
      const std::vector<std::vector<T>>& chunks,
      CompressionType compressionType = CompressionType::Uncompressed) {
    Buffer buffer{*pool_};
    std::string streamData;
    std::vector<ChunkInfo> chunkInfos;

    uint32_t currentOffset = 0;
    for (const auto& chunk : chunks) {
      // Encode each chunk using TrivialEncoding
      auto encodedChunk = encodeValues<T>(chunk, buffer);

      // Write chunk using ChunkedStreamWriter
      CompressionParams compressionParams{.type = compressionType};
      if (compressionType == CompressionType::Zstd) {
        compressionParams.zstdLevel = 3;
      }
      ChunkedStreamWriter writer{buffer, compressionParams};
      auto segments = writer.encode(encodedChunk);

      for (const auto& segment : segments) {
        streamData += segment;
      }

      chunkInfos.push_back({
          .rowCount = static_cast<uint32_t>(chunk.size()),
          .streamOffset = currentOffset,
      });
      currentOffset = streamData.size();
    }

    return {streamData, chunkInfos};
  }

  static void verifyChunkCompressionTypes(
      std::string_view streamData,
      size_t chunkCount,
      CompressionType compressionType) {
    const char* position = streamData.data();
    const char* const end = streamData.data() + streamData.size();
    for (size_t index = 0; index < chunkCount; ++index) {
      const auto [length, actualCompressionType] = readChunkHeader(position);
      EXPECT_EQ(actualCompressionType, compressionType);
      position += length;
      ASSERT_LE(position, end);
    }
    EXPECT_EQ(position, end);
  }

  template <typename T>
  std::unique_ptr<EncodingSelectionPolicy<T>> createEncodingSelectionPolicy(
      EncodingType encodingType) {
    std::vector<std::optional<const EncodingLayout>> children;
    if (encodingType == EncodingType::Nullable) {
      // Nullable encoding needs child encodings for:
      // 0: nulls bitmap (bool)
      // 1: non-null values encoding
      children.emplace_back(EncodingLayout(
          EncodingType::Trivial, {}, CompressionType::Uncompressed));
      children.emplace_back(EncodingLayout(
          EncodingType::Trivial, {}, CompressionType::Uncompressed));
    }
    EncodingLayout layout(
        encodingType, {}, CompressionType::Uncompressed, std::move(children));
    return std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
        std::move(layout),
        CompressionOptions{},
        [](DataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
          return nullptr;
        });
  }

  template <typename T>
  std::string_view encodeValues(const std::vector<T>& values, Buffer& buffer) {
    using physicalType = typename TypeTraits<T>::physicalType;
    EncodingSelectionResult selectionResult{
        .encodingType = EncodingType::Trivial};
    std::span<const physicalType> valuesSpan(
        reinterpret_cast<const physicalType*>(values.data()), values.size());
    Statistics<physicalType> stats =
        Statistics<physicalType>::create(valuesSpan);
    auto policy = createEncodingSelectionPolicy<T>(EncodingType::Trivial);
    EncodingSelection<T> selection{
        std::move(selectionResult), std::move(stats), std::move(policy)};
    return TrivialEncoding<T>::encode(selection, valuesSpan, buffer);
  }

  // Enum for null configurations in test cases.
  enum class NullConfig {
    NoNulls, // All values are non-null
    SomeNulls, // Some values are null (alternating pattern)
    AllNulls // All values are null
  };

  static std::string nullConfigName(NullConfig config) {
    switch (config) {
      case NullConfig::NoNulls:
        return "no_nulls";
      case NullConfig::SomeNulls:
        return "some_nulls";
      case NullConfig::AllNulls:
        return "all_nulls";
    }
    NIMBLE_UNREACHABLE();
  }

  // Encodes nullable integer values into a chunked stream format.
  // Uses std::optional<T> where std::nullopt represents null values.
  // Returns the encoded stream data and chunk metadata.
  template <typename T>
  std::pair<std::string, std::vector<ChunkInfo>> encodeNullableChunkedStream(
      const std::vector<std::vector<std::optional<T>>>& chunks) {
    Buffer buffer{*pool_};
    std::string streamData;
    std::vector<ChunkInfo> chunkInfos;

    uint32_t currentOffset = 0;
    for (const auto& chunk : chunks) {
      // Encode each chunk using NullableEncoding
      auto encodedChunk = encodeNullableValues<T>(chunk, buffer);

      // Write chunk using ChunkedStreamWriter
      ChunkedStreamWriter writer{
          buffer, {.type = CompressionType::Uncompressed}};
      auto segments = writer.encode(encodedChunk);

      for (const auto& segment : segments) {
        streamData += segment;
      }

      chunkInfos.push_back({
          .rowCount = static_cast<uint32_t>(chunk.size()),
          .streamOffset = currentOffset,
      });
      currentOffset = streamData.size();
    }

    return {streamData, chunkInfos};
  }

  // Encodes nullable values using std::optional<T>.
  // std::nullopt indicates a null value, otherwise the value is present.
  template <typename T>
  std::string_view encodeNullableValues(
      const std::vector<std::optional<T>>& values,
      Buffer& buffer) {
    using physicalType = typename TypeTraits<T>::physicalType;

    // Extract non-null values and build nulls bitmap
    // In Nimble: true = non-null, false = null
    // Use Vector<bool> instead of std::vector<bool> to work with std::span
    std::vector<physicalType> nonNullValues;
    Vector<bool> nulls{pool_.get()};
    nulls.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      nulls[i] = values[i].has_value();
      if (values[i].has_value()) {
        nonNullValues.push_back(
            reinterpret_cast<const physicalType&>(values[i].value()));
      }
    }

    EncodingSelectionResult selectionResult{
        .encodingType = EncodingType::Nullable};
    std::span<const physicalType> nonNullValuesSpan(
        nonNullValues.data(), nonNullValues.size());
    Statistics<physicalType> stats =
        Statistics<physicalType>::create(nonNullValuesSpan);
    auto policy = createEncodingSelectionPolicy<T>(EncodingType::Nullable);
    EncodingSelection<T> selection{
        std::move(selectionResult), std::move(stats), std::move(policy)};
    std::span<const bool> nullsSpan(nulls.data(), nulls.size());
    return NullableEncoding<T>::encodeNullable(
        selection, nonNullValuesSpan, nullsSpan, buffer);
  }

  // Generates test data based on NullConfig.
  // Returns a vector of std::optional<T> where std::nullopt represents null.
  // startIndex is the global index offset for value generation.
  template <typename T>
  static std::vector<std::optional<T>> generateNullableData(
      NullConfig config,
      uint32_t size,
      uint32_t startIndex = 0) {
    std::vector<std::optional<T>> data(size);
    switch (config) {
      case NullConfig::NoNulls:
        for (uint32_t i = 0; i < size; ++i) {
          data[i] = static_cast<T>(startIndex + i);
        }
        break;
      case NullConfig::SomeNulls:
        // Alternating pattern: odd global indices have values, even are null
        for (uint32_t i = 0; i < size; ++i) {
          const uint32_t globalIndex = startIndex + i;
          if (globalIndex % 2 == 0) {
            data[i] = std::nullopt;
          } else {
            data[i] = static_cast<T>(globalIndex);
          }
        }
        break;
      case NullConfig::AllNulls:
        for (uint32_t i = 0; i < size; ++i) {
          data[i] = std::nullopt;
        }
        break;
    }
    return data;
  }

  // Generates expected nulls based on NullConfig for a given range.
  // Returns a vector of booleans where true = non-null, false = null.
  static std::vector<bool> generateExpectedNulls(
      NullConfig config,
      uint32_t startIndex,
      uint32_t count) {
    std::vector<bool> nulls(count);
    switch (config) {
      case NullConfig::NoNulls:
        std::fill(nulls.begin(), nulls.end(), velox::bits::kNotNull);
        break;
      case NullConfig::SomeNulls:
        // Matches generateNullableData: odd indices are non-null
        for (uint32_t i = 0; i < count; ++i) {
          nulls[i] = (startIndex + i) % 2 != 0 ? velox::bits::kNotNull
                                               : velox::bits::kNull;
        }
        break;
      case NullConfig::AllNulls:
        std::fill(nulls.begin(), nulls.end(), velox::bits::kNull);
        break;
    }
    return nulls;
  }

  // Generates expected values from allValues starting at startIndex.
  // For non-null positions, returns the actual value; for null positions,
  // returns 0.
  template <typename T>
  static std::vector<T> generateExpectedValuesWithNulls(
      const std::vector<std::optional<T>>& allValues,
      uint32_t startIndex,
      uint32_t count) {
    std::vector<T> values(count);
    for (uint32_t i = 0; i < count; ++i) {
      const auto& val = allValues[startIndex + i];
      values[i] = val.has_value() ? val.value() : 0;
    }
    return values;
  }

  // Creates a stream index from chunk metadata for parameterized tests.
  // Returns nullptr when useStreamIndex() is false.
  std::shared_ptr<index::StreamIndex> createTestStreamIndex(
      const std::vector<ChunkInfo>& chunkInfos) {
    if (!useStreamIndex()) {
      return nullptr;
    }
    return createTestStreamIndexInternal(chunkInfos);
  }

  // Creates a stream index from chunk metadata (internal implementation).
  // Always returns a valid stream index regardless of test parameter.
  std::shared_ptr<index::StreamIndex> createTestStreamIndexInternal(
      const std::vector<ChunkInfo>& chunkInfos) {
    // Build stripe index data from chunk infos
    std::vector<int32_t> chunkRows;
    std::vector<uint32_t> chunkOffsets;
    uint32_t accumulatedRows = 0;
    for (const auto& info : chunkInfos) {
      accumulatedRows += info.rowCount;
      chunkRows.push_back(info.rowCount);
      chunkOffsets.push_back(info.streamOffset);
    }

    std::vector<std::string> indexColumns = {"col1"};
    std::string minKey = "aaa";
    std::vector<Stripe> stripes = {
        {.streams =
             {{.numChunks = static_cast<uint32_t>(chunkInfos.size()),
               .chunkRows = chunkRows,
               .chunkOffsets = chunkOffsets}},
         .keyStream = {
             .streamOffset = 0,
             .streamSize = 100,
             .stream =
                 {.numChunks = 1,
                  .chunkRows = {static_cast<int32_t>(accumulatedRows)},
                  .chunkOffsets = {0}},
             .chunkKeys = {"zzz"}}}};
    std::vector<int> stripeGroups = {1};

    testIndexBuffers_ =
        createTestClusterIndex(indexColumns, minKey, stripes, stripeGroups);
    testChunkStats_ = createChunkStats(testIndexBuffers_, 0);
    return testChunkStats_->createStreamIndex(0, 0, /*streamSize=*/1000);
  }

 private:
  IndexBuffers testIndexBuffers_;
  std::shared_ptr<index::ChunkStatsGroup> testChunkStats_;
  EncodingFactory encodingFactory_;
};

TEST_P(ChunkedDecoderDataTest, skipTwoChunks) {
  struct TestCase {
    std::string name;
    uint32_t skipCount;
    // All values to encode (split into two chunks). Expected values/nulls
    // after skip are simply allValues[skipCount:].
    std::vector<std::optional<uint32_t>> allValues;
    // Expected error message when skip fails. Empty string means no error
    // expected. For skip with index vs without index, we may have different
    // error messages.
    std::string expectedErrorMessageWithIndex;
    std::string expectedErrorMessageWithoutIndex;
  };

  // Test with 100 values
  constexpr uint32_t kNumValues = 100;

  std::vector<TestCase> testCases;

  // Generate test cases for all combinations of skip positions and null
  // configurations
  for (auto nullConfig :
       //{NullConfig::NoNulls, NullConfig::SomeNulls, NullConfig::AllNulls}) {
       {NullConfig::SomeNulls}) {
    const auto configName = nullConfigName(nullConfig);
    auto allValues = generateNullableData<uint32_t>(nullConfig, kNumValues);

    testCases.push_back(
        {.name = "skip_0_" + configName,
         .skipCount = 0,
         .allValues = allValues,
         .expectedErrorMessageWithIndex = "",
         .expectedErrorMessageWithoutIndex = ""});
    testCases.push_back(
        {.name = "skip_50_" + configName,
         .skipCount = 50,
         .allValues = allValues,
         .expectedErrorMessageWithIndex = "",
         .expectedErrorMessageWithoutIndex = ""});
    testCases.push_back(
        {.name = "skip_second_last_" + configName,
         .skipCount = kNumValues - 2,
         .allValues = allValues,
         .expectedErrorMessageWithIndex = "",
         .expectedErrorMessageWithoutIndex = ""});
    testCases.push_back(
        {.name = "skip_last_" + configName,
         .skipCount = kNumValues - 1,
         .allValues = allValues,
         .expectedErrorMessageWithIndex = "",
         .expectedErrorMessageWithoutIndex = ""});
    testCases.push_back(
        {.name = "skip_all_" + configName,
         .skipCount = kNumValues,
         .allValues = allValues,
         .expectedErrorMessageWithIndex = "",
         .expectedErrorMessageWithoutIndex = ""});
    // Skip beyond available values - expect error
    testCases.push_back(
        {.name = "skip_beyond_" + configName,
         .skipCount = kNumValues + 1,
         .allValues = allValues,
         .expectedErrorMessageWithIndex =
             "Cannot skip beyond end of stream in stream 0",
         .expectedErrorMessageWithoutIndex = "Failed to read chunk header"});
    testCases.push_back(
        {.name = "skip_beyond_" + configName,
         .skipCount = kNumValues + 10,
         .allValues = allValues,
         .expectedErrorMessageWithIndex =
             "Cannot skip beyond end of stream in stream 0",
         .expectedErrorMessageWithoutIndex = "Failed to read chunk header"});
  }

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);

    // Encode stream from allValues in two chunks so that createStreamIndex
    // returns a valid stream index (single-chunk streams return nullptr).
    const auto mid = testCase.allValues.size() / 2;
    std::vector<std::optional<uint32_t>> chunk1(
        testCase.allValues.begin(), testCase.allValues.begin() + mid);
    std::vector<std::optional<uint32_t>> chunk2(
        testCase.allValues.begin() + mid, testCase.allValues.end());
    auto [streamData, chunkInfos] =
        encodeNullableChunkedStream<uint32_t>({chunk1, chunk2});

    auto streamIndex = createTestStreamIndex(chunkInfos);
    ChunkedDecoder decoder(
        std::make_unique<dwio::common::SeekableArrayInputStream>(
            streamData.data(), streamData.size()),
        streamIndex,
        false,
        &encodingFactory(),
        pool_.get());

    // Check if we expect an error
    const auto& expectedErrorMessage = useStreamIndex()
        ? testCase.expectedErrorMessageWithIndex
        : testCase.expectedErrorMessageWithoutIndex;
    if (!expectedErrorMessage.empty()) {
      NIMBLE_ASSERT_THROW(
          decoder.skip(testCase.skipCount), expectedErrorMessage);
      continue;
    }

    // Skip values (including skip 0 to test that path)
    decoder.skip(testCase.skipCount);

    // Read remaining values (including 0 remaining to test that path)
    const uint32_t remainingCount =
        testCase.allValues.size() - testCase.skipCount;

    // Use decodeNullable for nullable data
    std::vector<uint64_t> nullBits(
        (remainingCount + 63) / 64, velox::bits::kNotNull64);
    std::vector<uint32_t> result(remainingCount);
    decoder.decodeNullable(
        nullBits.data(), result.data(), remainingCount, nullptr);

    // Verify against allValues[skipCount:]
    for (uint32_t i = 0; i < remainingCount; ++i) {
      const auto& expected = testCase.allValues[testCase.skipCount + i];
      const bool actualNonNull = velox::bits::isBitSet(nullBits.data(), i);
      const bool expectedNonNull = expected.has_value();

      EXPECT_EQ(actualNonNull, expectedNonNull)
          << "Null mismatch at position " << i;

      if (expectedNonNull) {
        EXPECT_EQ(result[i], expected.value())
            << "Value mismatch at position " << i;
      }
    }
  }
}

TEST_P(ChunkedDecoderDataTest, readsCompressedChunks) {
  constexpr uint32_t kFirstChunkValue = 42;
  constexpr uint32_t kSecondChunkValue = 7;
  const std::vector<std::vector<uint32_t>> chunks{
      std::vector<uint32_t>(512, kFirstChunkValue),
      std::vector<uint32_t>(384, kSecondChunkValue),
  };

  for (auto compressionType : {CompressionType::Zstd, CompressionType::Lz4}) {
    SCOPED_TRACE(toString(compressionType));
    auto [streamData, chunkInfos] =
        encodeChunkedStream<uint32_t>(chunks, compressionType);
    verifyChunkCompressionTypes(streamData, chunkInfos.size(), compressionType);

    ChunkedDecoder decoder(
        std::make_unique<dwio::common::SeekableArrayInputStream>(
            streamData.data(), streamData.size()),
        createTestStreamIndex(chunkInfos),
        false,
        &encodingFactory(),
        pool_.get());

    std::vector<int32_t> result(chunks[0].size() + chunks[1].size());
    decoder.nextIndices(result.data(), result.size(), nullptr);

    std::vector<int32_t> expected;
    expected.insert(expected.end(), chunks[0].size(), kFirstChunkValue);
    expected.insert(expected.end(), chunks[1].size(), kSecondChunkValue);
    EXPECT_EQ(result, expected);
  }
}

TEST_P(ChunkedDecoderDataTest, skipMultipleChunks) {
  struct TestCase {
    std::string name;
    uint32_t skipCount;
    uint32_t readCount;
    // Expected error message when skip fails (empty means no error)
    std::string expectedErrorMessageWithIndex;
    std::string expectedErrorMessageWithoutIndex;
  };

  // Chunk sizes: 50, 60, 40 (total 150)
  constexpr std::array<uint32_t, 3> kChunkSizes = {50, 60, 40};
  constexpr uint32_t kTotalValues = 150;

  std::vector<TestCase> testCases = {
      // Skip 0 values
      {.name = "skip_0",
       .skipCount = 0,
       .readCount = 10,
       .expectedErrorMessageWithIndex = "",
       .expectedErrorMessageWithoutIndex = ""},
      // Skip within first chunk
      {.name = "skip_within_first_chunk",
       .skipCount = 30,
       .readCount = 10,
       .expectedErrorMessageWithIndex = "",
       .expectedErrorMessageWithoutIndex = ""},
      // Skip to exact chunk boundary (end of chunk 0)
      {.name = "skip_to_chunk_boundary",
       .skipCount = 50,
       .readCount = 10,
       .expectedErrorMessageWithIndex = "",
       .expectedErrorMessageWithoutIndex = ""},
      // Skip across first chunk into second
      {.name = "skip_across_first_chunk",
       .skipCount = 80,
       .readCount = 40,
       .expectedErrorMessageWithIndex = "",
       .expectedErrorMessageWithoutIndex = ""},
      // Skip across two chunks (lands in third chunk)
      {.name = "skip_across_two_chunks",
       .skipCount = 120,
       .readCount = 20,
       .expectedErrorMessageWithIndex = "",
       .expectedErrorMessageWithoutIndex = ""},
      // Skip to last value
      {.name = "skip_to_last_value",
       .skipCount = kTotalValues - 1,
       .readCount = 1,
       .expectedErrorMessageWithIndex = "",
       .expectedErrorMessageWithoutIndex = ""},
      // Skip all values
      {.name = "skip_all",
       .skipCount = kTotalValues,
       .readCount = 0,
       .expectedErrorMessageWithIndex = "",
       .expectedErrorMessageWithoutIndex = ""},
      // Skip beyond available values
      {.name = "skip_beyond",
       .skipCount = kTotalValues + 1,
       .readCount = 0,
       .expectedErrorMessageWithIndex =
           "Cannot skip beyond end of stream in stream 0",
       .expectedErrorMessageWithoutIndex = "Failed to read chunk header"},
      // Skip way beyond available values
      {.name = "skip_way_beyond",
       .skipCount = kTotalValues + 100,
       .readCount = 0,
       .expectedErrorMessageWithIndex =
           "Cannot skip beyond end of stream in stream 0",
       .expectedErrorMessageWithoutIndex = "Failed to read chunk header"},
  };

  for (auto nullConfig : {NullConfig::NoNulls, NullConfig::SomeNulls}) {
    const auto configName = nullConfigName(nullConfig);

    // Generate chunks with the specified null config
    // Chunk 0: rows 0-49, Chunk 1: rows 50-109, Chunk 2: rows 110-149
    std::vector<std::vector<std::optional<uint32_t>>> chunks;
    std::vector<std::optional<uint32_t>> allValues;
    uint32_t globalIndex = 0;
    for (auto chunkSize : kChunkSizes) {
      auto chunkData =
          generateNullableData<uint32_t>(nullConfig, chunkSize, globalIndex);
      allValues.insert(allValues.end(), chunkData.begin(), chunkData.end());
      chunks.push_back(std::move(chunkData));
      globalIndex += chunkSize;
    }

    for (const auto& testCase : testCases) {
      SCOPED_TRACE(configName + "_" + testCase.name);

      auto [streamData, chunkInfos] =
          encodeNullableChunkedStream<uint32_t>(chunks);

      auto streamIndex = createTestStreamIndex(chunkInfos);
      ChunkedDecoder decoder(
          std::make_unique<dwio::common::SeekableArrayInputStream>(
              streamData.data(), streamData.size()),
          streamIndex,
          false,
          &encodingFactory(),
          pool_.get());

      // Check if we expect an error
      const auto& expectedErrorMessage = useStreamIndex()
          ? testCase.expectedErrorMessageWithIndex
          : testCase.expectedErrorMessageWithoutIndex;
      if (!expectedErrorMessage.empty()) {
        NIMBLE_ASSERT_THROW(
            decoder.skip(testCase.skipCount), expectedErrorMessage);
        continue;
      }

      decoder.skip(testCase.skipCount);

      if (testCase.readCount == 0) {
        continue;
      }

      // Read and verify values
      std::vector<uint64_t> nullBits(
          (testCase.readCount + 63) / 64, velox::bits::kNotNull64);
      std::vector<uint32_t> result(testCase.readCount);
      decoder.decodeNullable(
          nullBits.data(), result.data(), testCase.readCount, nullptr);

      // Verify against allValues[skipCount:]
      for (uint32_t i = 0; i < testCase.readCount; ++i) {
        const auto& expected = allValues[testCase.skipCount + i];
        const bool actualNonNull = velox::bits::isBitSet(nullBits.data(), i);
        const bool expectedNonNull = expected.has_value();

        EXPECT_EQ(actualNonNull, expectedNonNull)
            << "Null mismatch at position " << i;

        if (expectedNonNull) {
          EXPECT_EQ(result[i], expected.value())
              << "Value mismatch at position " << i;
        }
      }
    }
  }
}

// Test skip within current chunk optimization in skipWithIndex
TEST_P(ChunkedDecoderDataTest, skipWithinCurrentChunk) {
  // Create test data: single chunk with 100 values
  std::vector<uint32_t> values(100);
  std::iota(values.begin(), values.end(), 0);

  auto [streamData, chunkInfos] =
      encodeChunkedStream<uint32_t>(std::vector<std::vector<uint32_t>>{values});

  auto streamIndex = createTestStreamIndex(chunkInfos);
  ChunkedDecoder decoder(
      std::make_unique<dwio::common::SeekableArrayInputStream>(
          streamData.data(), streamData.size()),
      streamIndex,
      false,
      &encodingFactory(),
      pool_.get());

  // Read 10 values first

  // Read 10 values first to load the chunk and set remainingValues_
  std::vector<int32_t> result1(10);
  decoder.nextIndices(result1.data(), 10, nullptr);
  std::vector<int32_t> expected1(10);
  std::iota(expected1.begin(), expected1.end(), 0);
  EXPECT_EQ(result1, expected1);

  // Skip 20 values within the same chunk (tests skipWithIndex optimization)
  decoder.skip(20);

  // Read remaining 70 values
  std::vector<int32_t> result2(70);
  decoder.nextIndices(result2.data(), 70, nullptr);
  std::vector<int32_t> expected2(70);
  std::iota(expected2.begin(), expected2.end(), 30);
  EXPECT_EQ(result2, expected2);
}

TEST_P(ChunkedDecoderDataTest, skipEntireStream) {
  // Create test data: 2 chunks with 30, 20 uint32 values
  std::vector<std::vector<uint32_t>> chunks;
  uint32_t value = 0;
  for (int size : {30, 20}) {
    std::vector<uint32_t> chunk(size);
    std::iota(chunk.begin(), chunk.end(), value);
    value += size;
    chunks.push_back(std::move(chunk));
  }

  auto [streamData, chunkInfos] = encodeChunkedStream<uint32_t>(chunks);

  auto streamIndex = createTestStreamIndex(chunkInfos);
  ChunkedDecoder decoder(
      std::make_unique<dwio::common::SeekableArrayInputStream>(
          streamData.data(), streamData.size()),
      streamIndex,
      false,
      &encodingFactory(),
      pool_.get());

  // Skip all 50 values
  decoder.skip(50);

  // Try to read after skipping entire stream - expect throw
  std::vector<int32_t> result(1);
  NIMBLE_ASSERT_THROW(
      decoder.nextIndices(result.data(), 1, nullptr),
      "Failed to read chunk header");
}

TEST_P(ChunkedDecoderDataTest, skipAndReadMixed) {
  // Create test data: 4 chunks with varying sizes
  std::vector<std::vector<uint32_t>> chunks;
  uint32_t value = 0;
  for (int size : {25, 35, 40, 50}) {
    std::vector<uint32_t> chunk(size);
    std::iota(chunk.begin(), chunk.end(), value);
    value += size;
    chunks.push_back(std::move(chunk));
  }

  auto [streamData, chunkInfos] = encodeChunkedStream<uint32_t>(chunks);

  auto streamIndex = createTestStreamIndex(chunkInfos);
  ChunkedDecoder decoder(
      std::make_unique<dwio::common::SeekableArrayInputStream>(
          streamData.data(), streamData.size()),
      streamIndex,
      false,
      &encodingFactory(),
      pool_.get());

  // Read 10 values
  std::vector<int32_t> result1(10);
  decoder.nextIndices(result1.data(), 10, nullptr);
  std::vector<int32_t> expected1(10);
  std::iota(expected1.begin(), expected1.end(), 0);
  EXPECT_EQ(result1, expected1);

  // Skip 50 values (from position 10 to 60)
  decoder.skip(50);

  // Read 20 values (from position 60)
  std::vector<int32_t> result2(20);
  decoder.nextIndices(result2.data(), 20, nullptr);
  std::vector<int32_t> expected2(20);
  std::iota(expected2.begin(), expected2.end(), 60);
  EXPECT_EQ(result2, expected2);

  // Skip 30 values (from position 80 to 110)
  decoder.skip(30);

  // Read remaining 40 values (from position 110 to 150)
  std::vector<int32_t> result3(40);
  decoder.nextIndices(result3.data(), 40, nullptr);
  std::vector<int32_t> expected3(40);
  std::iota(expected3.begin(), expected3.end(), 110);
  EXPECT_EQ(result3, expected3);
}

// Test skip with index using simple values to verify index-based skip is
// triggered. This test creates multiple chunks and verifies that skipping
// across chunks or within chunks correctly calls skipWithIndex.
DEBUG_ONLY_TEST_P(ChunkedDecoderDataTest, skipChunkWithIndexCheck) {
  // Create 4 chunks with known values:
  // Chunk 0: [0, 1, 2, 3, 4] (5 values, rows 0-4)
  // Chunk 1: [5, 6, 7, 8, 9] (5 values, rows 5-9)
  // Chunk 2: [10, 11, 12, 13, 14] (5 values, rows 10-14)
  // Chunk 3: [15, 16, 17, 18, 19] (5 values, rows 15-19)
  // Total: 20 values
  std::vector<std::vector<uint32_t>> chunks = {
      {0, 1, 2, 3, 4},
      {5, 6, 7, 8, 9},
      {10, 11, 12, 13, 14},
      {15, 16, 17, 18, 19},
  };

  auto [streamData, chunkInfos] = encodeChunkedStream<uint32_t>(chunks);

  auto streamIndex = createTestStreamIndex(chunkInfos);
  ChunkedDecoder decoder(
      std::make_unique<dwio::common::SeekableArrayInputStream>(
          streamData.data(), streamData.size()),
      streamIndex,
      false,
      &encodingFactory(),
      pool_.get());

  // Use TestValue to track skipWithIndex calls
  uint32_t skipWithIndexCount = 0;
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::ChunkedDecoder::skipWithIndex",
      std::function<void(ChunkedDecoder*)>(
          [&](ChunkedDecoder*) { ++skipWithIndexCount; }));

  // Test scenario 1: Skip across first chunk (0->6, crosses chunk 0 to chunk 1)
  // This should trigger skipWithIndex when index is set
  decoder.skip(6);
  if (useStreamIndex()) {
    EXPECT_EQ(skipWithIndexCount, 1);
  } else {
    EXPECT_EQ(skipWithIndexCount, 0);
  }

  // Verify we're at value 6
  {
    std::vector<int32_t> result(1);
    decoder.nextIndices(result.data(), 1, nullptr);
    EXPECT_EQ(result[0], 6);
  }
  // Now at row 7

  // Test scenario 2: Skip within current chunk (7->9, stays in chunk 1)
  // This should still call skipWithIndex (but it optimizes internally)
  decoder.skip(2);
  if (useStreamIndex()) {
    EXPECT_EQ(skipWithIndexCount, 2);
  } else {
    EXPECT_EQ(skipWithIndexCount, 0);
  }

  // Verify we're at value 9
  {
    std::vector<int32_t> result(1);
    decoder.nextIndices(result.data(), 1, nullptr);
    EXPECT_EQ(result[0], 9);
  }
  // Now at row 10 (start of chunk 2)

  // Test scenario 3: Skip across multiple chunks (10->17, crosses chunk 2 to
  // chunk 3)
  decoder.skip(7);
  if (useStreamIndex()) {
    EXPECT_EQ(skipWithIndexCount, 3);
  } else {
    EXPECT_EQ(skipWithIndexCount, 0);
  }

  // Verify we're at value 17
  {
    std::vector<int32_t> result(1);
    decoder.nextIndices(result.data(), 1, nullptr);
    EXPECT_EQ(result[0], 17);
  }
  // Now at row 18

  // Test scenario 4: Skip to end of stream (18->20, lands at chunk boundary)
  decoder.skip(2);
  if (useStreamIndex()) {
    EXPECT_EQ(skipWithIndexCount, 4);
  } else {
    EXPECT_EQ(skipWithIndexCount, 0);
  }

  // Test scenario 5: Skip beyond available values - expect throw
  // Error messages differ between with/without index
  if (useStreamIndex()) {
    NIMBLE_ASSERT_THROW(decoder.skip(1), "Cannot skip beyond end of stream");
  } else {
    NIMBLE_ASSERT_THROW(decoder.skip(1), "Failed to read chunk header");
  }
}

INSTANTIATE_TEST_SUITE_P(
    ChunkedDecoderDataTests,
    ChunkedDecoderDataTest,
    testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) {
      return info.param ? "WithStreamIndex" : "WithoutStreamIndex";
    });

// Fuzzer test for ChunkedDecoder with randomized operations.
// Uses two decoders (one with index, one without) and verifies they produce
// the same results for identical operations.
// Tests various combinations of:
// - Null configurations: all nulls, some nulls, no nulls
// - Chunk configurations: single chunk, multiple chunks
// - Operations: interleaved skip and read (multiple read APIs)
TEST_F(ChunkedDecoderDataTest, fuzzer) {
  constexpr uint32_t kNumIterations = 100;
  constexpr uint32_t kMinTotalRows = 10;
  constexpr uint32_t kMaxTotalRows = 1'000;
  constexpr uint32_t kMinChunks = 1;
  constexpr uint32_t kMaxChunks = 5;
  constexpr uint32_t kMinOpsPerIteration = 3;
  constexpr uint32_t kMaxOpsPerIteration = 10;

  std::mt19937 rng(42); // Fixed seed for reproducibility

  // Distribution for null configurations
  std::uniform_int_distribution<int> nullConfigDist(0, 2);
  // Distribution for number of chunks
  std::uniform_int_distribution<uint32_t> numChunksDist(kMinChunks, kMaxChunks);
  // Distribution for total rows
  std::uniform_int_distribution<uint32_t> totalRowsDist(
      kMinTotalRows, kMaxTotalRows);
  // Distribution for number of operations
  std::uniform_int_distribution<uint32_t> numOpsDist(
      kMinOpsPerIteration, kMaxOpsPerIteration);
  // Distribution for operation type: 0=skip, 1=decodeNullable, 2=nextIndices
  std::uniform_int_distribution<int> opTypeDist(0, 2);

  for (uint32_t iteration = 0; iteration < kNumIterations; ++iteration) {
    // Generate test configuration
    const auto nullConfig = static_cast<NullConfig>(nullConfigDist(rng));
    const uint32_t numChunks = numChunksDist(rng);
    const uint32_t totalRows = totalRowsDist(rng);

    SCOPED_TRACE(
        fmt::format(
            "iteration={} nullConfig={} numChunks={} totalRows={}",
            iteration,
            nullConfigName(nullConfig),
            numChunks,
            totalRows));

    // Generate chunk sizes that sum to totalRows
    std::vector<uint32_t> chunkSizes(numChunks);
    {
      uint32_t remainingRows = totalRows;
      for (uint32_t i = 0; i < numChunks - 1; ++i) {
        // Leave at least 1 row for remaining chunks
        const uint32_t maxChunkSize =
            remainingRows - (numChunks - i - 1); // Leave room for other chunks
        std::uniform_int_distribution<uint32_t> chunkSizeDist(1, maxChunkSize);
        chunkSizes[i] = chunkSizeDist(rng);
        remainingRows -= chunkSizes[i];
      }
      chunkSizes[numChunks - 1] = remainingRows;
    }

    // Generate test data
    std::vector<std::vector<std::optional<uint32_t>>> chunks;
    std::vector<std::optional<uint32_t>> allValues;
    uint32_t rowOffset = 0;
    for (uint32_t i = 0; i < numChunks; ++i) {
      auto chunkData =
          generateNullableData<uint32_t>(nullConfig, chunkSizes[i], rowOffset);
      allValues.insert(allValues.end(), chunkData.begin(), chunkData.end());
      chunks.push_back(std::move(chunkData));
      rowOffset += chunkSizes[i];
    }

    // Encode the data
    auto [streamData, chunkInfos] =
        encodeNullableChunkedStream<uint32_t>(chunks);

    // Create stream index for the decoder with index
    auto streamIndex = createTestStreamIndexInternal(chunkInfos);

    // Create two decoders: one with index, one without
    ChunkedDecoder decoderWithIndex(
        std::make_unique<dwio::common::SeekableArrayInputStream>(
            streamData.data(), streamData.size()),
        streamIndex,
        false,
        &encodingFactory(),
        pool_.get());

    ChunkedDecoder decoderWithoutIndex(
        std::make_unique<dwio::common::SeekableArrayInputStream>(
            streamData.data(), streamData.size()),
        nullptr,
        false,
        &encodingFactory(),
        pool_.get());

    // Generate and execute random operations on both decoders
    const uint32_t numOps = numOpsDist(rng);
    uint32_t currentPosition = 0;

    for (uint32_t opIdx = 0; opIdx < numOps; ++opIdx) {
      const uint32_t remainingRows = totalRows - currentPosition;
      if (remainingRows == 0) {
        break;
      }

      const int opType = opTypeDist(rng);
      std::uniform_int_distribution<uint32_t> countDist(1, remainingRows);
      const uint32_t count = countDist(rng);

      SCOPED_TRACE(
          fmt::format(
              "op={} opType={} count={} currentPosition={} remainingRows={}",
              opIdx,
              opType == 0 ? "skip"
                          : (opType == 1 ? "decodeNullable" : "nextIndices"),
              count,
              currentPosition,
              remainingRows));

      if (opType == 0) {
        // Skip operation - apply to both decoders
        decoderWithIndex.skip(count);
        decoderWithoutIndex.skip(count);
        currentPosition += count;
      } else if (opType == 1 || nullConfig != NullConfig::NoNulls) {
        // decodeNullable operation - apply to both decoders and compare
        std::vector<uint64_t> nullBitsWithIndex(
            (count + 63) / 64, velox::bits::kNotNull64);
        std::vector<uint32_t> resultWithIndex(count);
        decoderWithIndex.decodeNullable(
            nullBitsWithIndex.data(), resultWithIndex.data(), count, nullptr);

        std::vector<uint64_t> nullBitsWithoutIndex(
            (count + 63) / 64, velox::bits::kNotNull64);
        std::vector<uint32_t> resultWithoutIndex(count);
        decoderWithoutIndex.decodeNullable(
            nullBitsWithoutIndex.data(),
            resultWithoutIndex.data(),
            count,
            nullptr);

        // Compare results from both decoders
        for (uint32_t i = 0; i < count; ++i) {
          const bool nullWithIndex =
              velox::bits::isBitSet(nullBitsWithIndex.data(), i);
          const bool nullWithoutIndex =
              velox::bits::isBitSet(nullBitsWithoutIndex.data(), i);

          EXPECT_EQ(nullWithIndex, nullWithoutIndex)
              << "Null mismatch between decoders at position " << i
              << " (global " << (currentPosition + i) << ")";

          if (nullWithIndex) {
            EXPECT_EQ(resultWithIndex[i], resultWithoutIndex[i])
                << "Value mismatch between decoders at position " << i
                << " (global " << (currentPosition + i) << ")";
          }

          // Also verify against expected values
          const auto& expected = allValues[currentPosition + i];
          const bool expectedNonNull = expected.has_value();
          EXPECT_EQ(nullWithIndex, expectedNonNull)
              << "Null mismatch with expected at position " << i << " (global "
              << (currentPosition + i) << ")";

          if (expectedNonNull) {
            EXPECT_EQ(resultWithIndex[i], expected.value())
                << "Value mismatch with expected at position " << i
                << " (global " << (currentPosition + i) << ")";
          }
        }
        currentPosition += count;
      } else {
        // nextIndices operation for non-null data - apply to both decoders
        std::vector<int32_t> resultWithIndex(count);
        decoderWithIndex.nextIndices(resultWithIndex.data(), count, nullptr);

        std::vector<int32_t> resultWithoutIndex(count);
        decoderWithoutIndex.nextIndices(
            resultWithoutIndex.data(), count, nullptr);

        // Compare results from both decoders
        EXPECT_EQ(resultWithIndex, resultWithoutIndex)
            << "nextIndices results differ between decoders";

        // Also verify against expected values
        for (uint32_t i = 0; i < count; ++i) {
          const auto& expected = allValues[currentPosition + i];
          EXPECT_TRUE(expected.has_value())
              << "Expected non-null at position " << i;
          EXPECT_EQ(static_cast<uint32_t>(resultWithIndex[i]), expected.value())
              << "Value mismatch with expected at position " << i << " (global "
              << (currentPosition + i) << ")";
        }
        currentPosition += count;
      }
    }
  }
}

TEST_P(ChunkedDecoderDataTest, ensureLoadedAndAccessors) {
  auto [streamData, chunkInfos] =
      encodeChunkedStream<uint32_t>({{1, 2, 3, 4, 5}});
  auto streamIndex = createTestStreamIndex(chunkInfos);
  auto input = std::make_unique<velox::dwio::common::SeekableArrayInputStream>(
      streamData.data(), streamData.size());
  ChunkedDecoder decoder(
      std::move(input),
      streamIndex,
      false,
      &encodingFactory(),
      pool_.get(),
      true);

  EXPECT_EQ(decoder.currentEncoding(), nullptr);
  EXPECT_EQ(decoder.remainingValues(), 0);

  decoder.ensureLoaded();

  ASSERT_NE(decoder.currentEncoding(), nullptr);
  EXPECT_EQ(decoder.remainingValues(), 5);
  EXPECT_FALSE(decoder.dictionaryConvertible());
}

TEST_P(ChunkedDecoderDataTest, skipChunkBoundaryCallback) {
  auto [streamData, chunkInfos] =
      encodeChunkedStream<uint32_t>({{1, 2, 3}, {4, 5, 6}});
  auto streamIndex = createTestStreamIndex(chunkInfos);
  auto input = std::make_unique<velox::dwio::common::SeekableArrayInputStream>(
      streamData.data(), streamData.size());
  ChunkedDecoder decoder(
      std::move(input), streamIndex, false, &encodingFactory(), pool_.get());

  decoder.ensureLoaded();

  int callbackCount = 0;
  // Skip past chunk 1 (3 values) into chunk 2, triggering the callback.
  decoder.skip(4, [&] {
    ++callbackCount;
    return true;
  });
  EXPECT_EQ(callbackCount, 1);
}

TEST_P(ChunkedDecoderDataTest, hasMoreChunks) {
  auto [streamData, chunkInfos] =
      encodeChunkedStream<uint32_t>({{1, 2, 3}, {4, 5}});
  auto streamIndex = createTestStreamIndex(chunkInfos);
  auto input = std::make_unique<velox::dwio::common::SeekableArrayInputStream>(
      streamData.data(), streamData.size());
  ChunkedDecoder decoder(
      std::move(input), streamIndex, false, &encodingFactory(), pool_.get());

  // Before any load, there are chunks available.
  decoder.ensureLoaded();
  EXPECT_EQ(decoder.remainingValues(), 3);

  // Skip all of chunk 1. Chunk 2 is not yet loaded but available.
  decoder.skip(3);
  EXPECT_EQ(decoder.remainingValues(), 0);

  // ensureLoaded reloads when exhausted and more chunks exist.
  decoder.ensureLoaded();
  EXPECT_EQ(decoder.remainingValues(), 2);

  // Skip all of chunk 2. No more chunks available.
  decoder.skip(2);
  EXPECT_EQ(decoder.remainingValues(), 0);

  // ensureLoaded is a no-op when no more chunks exist.
  decoder.ensureLoaded();
  EXPECT_EQ(decoder.remainingValues(), 0);
}

TEST_P(ChunkedDecoderDataTest, ensureLoadedReloadsExhaustedChunk) {
  auto [streamData, chunkInfos] =
      encodeChunkedStream<uint32_t>({{1, 2, 3}, {4, 5, 6}});
  auto streamIndex = createTestStreamIndex(chunkInfos);
  auto input = std::make_unique<velox::dwio::common::SeekableArrayInputStream>(
      streamData.data(), streamData.size());
  ChunkedDecoder decoder(
      std::move(input), streamIndex, false, &encodingFactory(), pool_.get());

  decoder.ensureLoaded();
  EXPECT_EQ(decoder.remainingValues(), 3);

  // Consume chunk 1.
  decoder.skip(3);
  EXPECT_EQ(decoder.remainingValues(), 0);

  // ensureLoaded reloads chunk 2.
  decoder.ensureLoaded();
  EXPECT_EQ(decoder.remainingValues(), 3);
}

// Verifies that ensureLoaded fires its onChunkBoundary callback when
// reloading an exhausted chunk, allowing the caller to invalidate cached
// state (e.g., dictionary alphabet).
TEST_P(ChunkedDecoderDataTest, ensureLoadedFiresCallbackOnReload) {
  auto [streamData, chunkInfos] =
      encodeChunkedStream<uint32_t>({{1, 2, 3}, {4, 5, 6}});
  auto streamIndex = createTestStreamIndex(chunkInfos);
  auto input = std::make_unique<velox::dwio::common::SeekableArrayInputStream>(
      streamData.data(), streamData.size());
  ChunkedDecoder decoder(
      std::move(input), streamIndex, false, &encodingFactory(), pool_.get());

  int callbackCount = 0;
  auto onChunkBoundary = [&] {
    ++callbackCount;
    return true;
  };

  // First load fires the callback.
  decoder.ensureLoaded(onChunkBoundary);
  EXPECT_EQ(callbackCount, 1);
  EXPECT_EQ(decoder.remainingValues(), 3);

  // ensureLoaded with remaining > 0 does not fire.
  decoder.ensureLoaded(onChunkBoundary);
  EXPECT_EQ(callbackCount, 1);

  // Consume chunk 1.
  decoder.skip(3);
  EXPECT_EQ(decoder.remainingValues(), 0);

  // ensureLoaded reloads chunk 2, firing the callback.
  decoder.ensureLoaded(onChunkBoundary);
  EXPECT_EQ(callbackCount, 2);
  EXPECT_EQ(decoder.remainingValues(), 3);
}

// DictionaryEnc with StringTrivialEnc for the alphabet, suitable for
// encoding std::string_view data.
DictionaryEnc stringDictEnc() {
  return DictionaryEnc{.alphabet = StringTrivialEnc{}};
}

class ChunkedDecoderDictTest : public ChunkedDecoderTest {
 protected:
  // Encodes string values into a chunked stream using the given encoding
  // layout. Returns the encoded stream data.
  std::string encodeStringChunkedStream(
      const std::vector<std::vector<std::string_view>>& chunks,
      const EncodingLayout& layout) {
    Buffer buffer{pool()};
    std::string streamData;

    for (const auto& chunk : chunks) {
      // The creator must outlive the policy because
      // ReplayedEncodingSelectionPolicy stores it by reference.
      EncodingSelectionPolicyCreator creator =
          [](DataType type) -> std::unique_ptr<EncodingSelectionPolicyBase> {
        UNIQUE_PTR_FACTORY(type, TrivialNestedPolicy);
      };
      auto policy =
          std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
              layout, CompressionOptions{}, creator);
      auto encoded = EncodingFactory::encode<std::string_view>(
          std::move(policy),
          std::span<const std::string_view>(chunk.data(), chunk.size()),
          buffer);

      ChunkedStreamWriter writer{
          buffer, {.type = CompressionType::Uncompressed}};
      for (const auto& segment : writer.encode(encoded)) {
        streamData += segment;
      }
    }
    return streamData;
  }

  // Encodes nullable string values into a chunked stream.
  std::string encodeNullableStringChunkedStream(
      const std::vector<std::vector<std::optional<std::string_view>>>& chunks,
      const EncodingLayout& dataLayout) {
    Buffer buffer{pool()};
    std::string streamData;

    for (const auto& chunk : chunks) {
      std::vector<std::string_view> nonNullValues;
      Vector<bool> nulls(&pool());
      nulls.resize(chunk.size());
      for (size_t i = 0; i < chunk.size(); ++i) {
        nulls[i] = chunk[i].has_value();
        if (chunk[i].has_value()) {
          nonNullValues.push_back(*chunk[i]);
        }
      }

      // Pass just the data layout; ReplayedEncodingSelectionPolicy::
      // selectNullable() will automatically wrap it in a Nullable layout.
      // The creator must outlive the policy because
      // ReplayedEncodingSelectionPolicy stores it by reference.
      EncodingSelectionPolicyCreator creator =
          [](DataType type) -> std::unique_ptr<EncodingSelectionPolicyBase> {
        UNIQUE_PTR_FACTORY(type, TrivialNestedPolicy);
      };
      auto policy =
          std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
              dataLayout, CompressionOptions{}, creator);

      auto encoded = EncodingFactory::encodeNullable<std::string_view>(
          std::move(policy),
          std::span<const std::string_view>(
              nonNullValues.data(), nonNullValues.size()),
          std::span<const bool>(nulls.data(), nulls.size()),
          buffer);

      ChunkedStreamWriter writer{
          buffer, {.type = CompressionType::Uncompressed}};
      for (const auto& segment : writer.encode(encoded)) {
        streamData += segment;
      }
    }
    return streamData;
  }

  // Creates a ChunkedDecoder from encoded string stream data.
  ChunkedDecoder createStringDecoder(const std::string& streamData) {
    return ChunkedDecoder(
        std::make_unique<dwio::common::SeekableArrayInputStream>(
            streamData.data(), streamData.size()),
        /*streamIndex=*/nullptr,
        /*decodeValuesWithNulls=*/false,
        &encodingFactory(),
        &pool(),
        /*stringDecoderZeroCopy=*/true);
  }
};

// Verify dictionaryConvertible() returns true for Dictionary encoding and
// eagerly loads the first chunk when encoding_ is null.
// Verifies dictionaryConvertible() for standalone (non-nested) encodings:
// Dictionary (true), Nullable→Dictionary (true), and Trivial (false).
TEST_F(ChunkedDecoderDictTest, dictionaryConvertibleLeafEncodings) {
  // Dictionary encoding.
  {
    SCOPED_TRACE("Dictionary");
    std::vector<std::string_view> data = {"apple", "banana", "apple", "cherry"};
    auto streamData = encodeStringChunkedStream({data}, stringDictEnc());
    auto decoder = createStringDecoder(streamData);
    decoder.ensureLoaded();
    EXPECT_TRUE(decoder.dictionaryConvertible());
    EXPECT_EQ(decoder.currentEncoding()->dataType(), DataType::String);
  }

  // Nullable→Dictionary.
  {
    SCOPED_TRACE("Nullable→Dictionary");
    std::vector<std::optional<std::string_view>> data = {
        "apple", std::nullopt, "banana", "apple"};
    auto streamData =
        encodeNullableStringChunkedStream({data}, stringDictEnc());
    auto decoder = createStringDecoder(streamData);
    decoder.ensureLoaded();
    EXPECT_TRUE(decoder.dictionaryConvertible());
    EXPECT_EQ(
        decoder.currentEncoding()->encodingType(), EncodingType::Nullable);
  }

  // Trivial — not dictionary-convertible.
  {
    SCOPED_TRACE("Trivial");
    std::vector<std::string_view> data = {"hello", "world"};
    auto streamData = encodeStringChunkedStream({data}, StringTrivialEnc{});
    auto decoder = createStringDecoder(streamData);
    decoder.ensureLoaded();
    EXPECT_FALSE(decoder.dictionaryConvertible());
  }
}

// Verifies dictionaryConvertible() for nested encodings:
// MainlyConstant→Dictionary (true) and Nullable→MainlyConstant→Dictionary
// (true).
TEST_F(ChunkedDecoderDictTest, dictionaryConvertibleNestedEncodings) {
  // MainlyConstant→Dictionary.
  {
    SCOPED_TRACE("MainlyConstant→Dictionary");
    std::vector<std::string_view> data;
    data.reserve(100);
    for (int i = 0; i < 100; ++i) {
      data.emplace_back("common");
    }
    data[10] = "rare";
    data[20] = "rare";
    auto streamData = encodeStringChunkedStream(
        {data}, MainlyConstantEnc{.otherValues = stringDictEnc()});
    auto decoder = createStringDecoder(streamData);
    decoder.ensureLoaded();
    EXPECT_TRUE(decoder.dictionaryConvertible());
  }

  // Nullable→MainlyConstant→Dictionary.
  {
    SCOPED_TRACE("Nullable→MainlyConstant→Dictionary");
    std::vector<std::optional<std::string_view>> data;
    data.reserve(100);
    for (int i = 0; i < 100; ++i) {
      data.emplace_back("common");
    }
    data[10] = "rare";
    data[20] = "rare";
    data[30] = std::nullopt;
    data[60] = std::nullopt;
    auto streamData = encodeNullableStringChunkedStream(
        {data}, MainlyConstantEnc{.otherValues = stringDictEnc()});
    auto decoder = createStringDecoder(streamData);
    decoder.ensureLoaded();
    EXPECT_TRUE(decoder.dictionaryConvertible());
    EXPECT_EQ(
        decoder.currentEncoding()->encodingType(), EncodingType::Nullable);
  }
}

// Verifies dictionaryConvertible() is re-evaluated per chunk and that
// multi-chunk navigation works correctly.
TEST_F(ChunkedDecoderDictTest, dictionaryConvertibleAcrossChunks) {
  std::vector<std::string_view> chunk1 = {"a", "b", "c", "a"};
  std::vector<std::string_view> chunk2 = {"x", "y", "z", "x", "y"};

  auto streamData =
      encodeStringChunkedStream({chunk1, chunk2}, stringDictEnc());

  auto decoder = createStringDecoder(streamData);

  // First chunk is dictionary.
  decoder.ensureLoaded();
  EXPECT_TRUE(decoder.dictionaryConvertible());
  EXPECT_EQ(decoder.remainingValues(), 4);

  // Calling again with remaining > 0 returns cached result.
  EXPECT_TRUE(decoder.dictionaryConvertible());

  // Skip 3 values within chunk 1.
  decoder.skip(3);
  EXPECT_EQ(decoder.remainingValues(), 1);

  // Skip past the chunk boundary into chunk 2.
  decoder.skip(2);
  EXPECT_EQ(decoder.remainingValues(), 4);
  // Second chunk is also dictionary — re-evaluated after chunk load.
  EXPECT_TRUE(decoder.dictionaryConvertible());
}

// Verifies that alphabet string_views built from the encoding's dictionary
// point into valid string buffers allocated during loadNextChunk.
TEST_F(ChunkedDecoderDictTest, stringBuffersAfterChunkLoad) {
  std::vector<std::string_view> data = {"hello", "world", "hello"};
  auto streamData = encodeStringChunkedStream({data}, stringDictEnc());

  auto decoder = createStringDecoder(streamData);
  decoder.ensureLoaded();
  ASSERT_TRUE(decoder.dictionaryConvertible());

  auto alphabet = buildEncodingDictionaryAlphabet<std::string_view>(
      decoder.currentEncoding());
  ASSERT_EQ(alphabet.size(), 2);
  std::set<std::string> entries(alphabet.begin(), alphabet.end());
  EXPECT_TRUE(entries.count("hello"));
  EXPECT_TRUE(entries.count("world"));
}

} // namespace

} // namespace facebook::nimble
