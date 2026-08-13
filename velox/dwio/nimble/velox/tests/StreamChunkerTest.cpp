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

#include <utility>

#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/StreamData.h"
#include "velox/dwio/nimble/writer/StreamChunker.h"

namespace facebook::nimble {
template <typename T>
void populateData(Vector<T>& vec, const std::vector<T>& data) {
  vec.reserve(data.size());
  for (const auto& item : data) {
    vec.push_back(item);
  }
}

void populateStringData(
    StringBuffer& stringBuffer,
    const std::vector<std::string_view>& testData) {
  uint64_t extraMemory = 0;
  for (const auto& str : testData) {
    extraMemory += str.size();
  }
  auto& buffer = stringBuffer.buffer;
  auto& mutableLengths = stringBuffer.lengths;
  buffer.reserve(buffer.size() + extraMemory);
  mutableLengths.reserve(mutableLengths.size() + testData.size());
  for (const auto& str : testData) {
    buffer.insert(buffer.end(), str.begin(), str.end());
    mutableLengths.push_back(str.size());
  }
}

template <typename T>
std::vector<T> toVector(std::string_view data) {
  const T* chunkData = reinterpret_cast<const T*>(data.data());
  return std::vector<T>(chunkData, chunkData + data.size() / sizeof(T));
}

// Expected chunk result structure
template <typename T>
struct ExpectedChunk {
  std::vector<T> chunkData;
  std::vector<bool> chunkNonNulls = {};
  bool hasNulls = false;
  size_t extraMemory = 0;
};

class StreamChunkerTestsBase : public ::testing::Test {
 protected:
  // Helper method to validate chunk results against expected data
  template <typename T>
  void validateChunk(
      StreamData& stream,
      std::unique_ptr<StreamChunker> chunker,
      const std::vector<ExpectedChunk<T>>& expectedChunks,
      const ExpectedChunk<T>& expectedRetainedData) {
    // Compaction before chunking should have no effect.
    if (folly::Random::rand32() % 2 == 0) {
      chunker->compact();
    }

    int chunkIndex = 0;
    while (const auto chunkView = chunker->next()) {
      ASSERT_LT(chunkIndex, expectedChunks.size());
      ASSERT_TRUE(chunkView.has_value());
      const auto chunkData = toVector<T>(chunkView->data());
      const auto& expectedChunk = expectedChunks[chunkIndex];
      const auto& expectedChunkData = expectedChunk.chunkData;
      const auto& expectedChunkNonNulls = expectedChunk.chunkNonNulls;
      EXPECT_THAT(chunkData, ::testing::ElementsAreArray(expectedChunkData));
      EXPECT_THAT(
          chunkView->nonNulls(),
          ::testing::ElementsAreArray(expectedChunkNonNulls));
      EXPECT_EQ(chunkView->hasNulls(), expectedChunk.hasNulls);
      // Null count is precomputed by the chunker: false entries in the bitmap,
      // or -- for the null-only stream -- false entries in the validity
      // payload.
      uint64_t expectedNulls = std::count(
          expectedChunkNonNulls.begin(), expectedChunkNonNulls.end(), false);
      if constexpr (std::is_same_v<T, bool>) {
        if (expectedChunkNonNulls.empty() &&
            dynamic_cast<NullsStreamData*>(&stream) != nullptr) {
          expectedNulls = std::count(
              expectedChunkData.begin(), expectedChunkData.end(), false);
        }
      }
      EXPECT_EQ(chunkView->numNulls(), expectedNulls);
      ++chunkIndex;
    }
    ASSERT_EQ(chunkIndex, expectedChunks.size());

    // Erases processed data to reclaim memory.
    chunker->compact();

    // Validate buffer is properly compacted to only retain expected data
    size_t expectedStreamDataSize = 0;
    if (auto* nullableContentStringStreamData =
            dynamic_cast<NullableContentStringStreamData*>(&stream)) {
      if constexpr (std::is_same_v<T, std::string_view>) {
        const auto [buffer, lengths] =
            nullableContentStringStreamData->mutableData();

        std::string expectedBufferContent;
        std::vector<uint64_t> expectedLength;
        for (const auto str : expectedRetainedData.chunkData) {
          expectedBufferContent += str;
          expectedLength.push_back(str.size());
        }
        EXPECT_EQ(
            std::string(buffer.begin(), buffer.end()), expectedBufferContent);
        EXPECT_THAT(
            std::vector<uint64_t>(lengths.begin(), lengths.end()),
            ::testing::ElementsAreArray(expectedLength));
        expectedStreamDataSize = expectedBufferContent.size() +
            expectedLength.size() * sizeof(std::string_view);
      } else {
        NIMBLE_UNREACHABLE(
            "NullableContentStringStreamData can only be called on string data");
      }
    } else {
      EXPECT_THAT(
          toVector<T>(stream.data()),
          ::testing::ElementsAreArray(expectedRetainedData.chunkData));
      expectedStreamDataSize =
          sizeof(T) * expectedRetainedData.chunkData.size();
    }
    EXPECT_THAT(
        stream.nonNulls(),
        ::testing::ElementsAreArray(expectedRetainedData.chunkNonNulls));
    EXPECT_EQ(stream.hasNulls(), expectedRetainedData.hasNulls);
    const uint64_t expectedRetainedMemoryUsed = expectedStreamDataSize +
        expectedRetainedData.chunkNonNulls.size() +
        expectedRetainedData.extraMemory;
    EXPECT_EQ(stream.memoryUsed(), expectedRetainedMemoryUsed);
  }

  static void SetUpTestCase() {
    velox::memory::SharedArbitrator::registerFactory();
    velox::memory::MemoryManager::Options options;
    options.arbitratorKind = "SHARED";
    velox::memory::MemoryManager::testingSetInstance(options);
  }

  static void TearDownTestCase() {
    // registerFactory() throws if the kind is already registered, and every
    // suite in this binary runs SetUpTestCase. Pairing them leaves the
    // process-global registry as each suite found it, matching
    // OperatorTestBase::TearDownTestCase.
    velox::memory::SharedArbitrator::unregisterFactory();
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("default_root");
    leafPool_ = rootPool_->addLeafChild("default_leaf");
    schemaBuilder_ = std::make_unique<SchemaBuilder>();
    inputBufferGrowthPolicy_ =
        DefaultInputBufferGrowthPolicy::withDefaultRanges();
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
  std::unique_ptr<SchemaBuilder> schemaBuilder_;
  std::unique_ptr<InputBufferGrowthPolicy> inputBufferGrowthPolicy_;
};

TEST_F(StreamChunkerTestsBase, getStreamChunker) {
  // Ensure a chunker can be created for all types.
#define TEST_STREAM_CHUNKER_FOR_TYPE(scalarKind, T)                    \
  {                                                                    \
    const auto scalarTypeBuilder =                                     \
        schemaBuilder_->createScalarTypeBuilder(scalarKind);           \
    auto descriptor = &scalarTypeBuilder->scalarDescriptor();          \
    ContentStreamData<T> stream(                                       \
        *leafPool_, *descriptor, *inputBufferGrowthPolicy_);           \
    getStreamChunker(stream, {.minChunkSize = 4, .maxChunkSize = 30}); \
  }

  EXPECT_NO_THROW(
      TEST_STREAM_CHUNKER_FOR_TYPE(facebook::nimble::ScalarKind::Bool, bool));
  EXPECT_NO_THROW(
      TEST_STREAM_CHUNKER_FOR_TYPE(facebook::nimble::ScalarKind::Int8, int8_t));
  EXPECT_NO_THROW(TEST_STREAM_CHUNKER_FOR_TYPE(
      facebook::nimble::ScalarKind::Int16, int16_t));
  EXPECT_NO_THROW(TEST_STREAM_CHUNKER_FOR_TYPE(
      facebook::nimble::ScalarKind::Int32, int32_t));
  EXPECT_NO_THROW(TEST_STREAM_CHUNKER_FOR_TYPE(
      facebook::nimble::ScalarKind::UInt32, uint32_t));
  EXPECT_NO_THROW(TEST_STREAM_CHUNKER_FOR_TYPE(
      facebook::nimble::ScalarKind::Int64, int64_t));
  EXPECT_NO_THROW(
      TEST_STREAM_CHUNKER_FOR_TYPE(facebook::nimble::ScalarKind::Float, float));
  EXPECT_NO_THROW(TEST_STREAM_CHUNKER_FOR_TYPE(
      facebook::nimble::ScalarKind::Double, double));
  EXPECT_NO_THROW(TEST_STREAM_CHUNKER_FOR_TYPE(
      facebook::nimble::ScalarKind::String, std::string_view));
  EXPECT_NO_THROW(TEST_STREAM_CHUNKER_FOR_TYPE(
      facebook::nimble::ScalarKind::Binary, std::string_view));
  EXPECT_THROW(
      TEST_STREAM_CHUNKER_FOR_TYPE(
          facebook::nimble::ScalarKind::UInt64, uint64_t),
      facebook::nimble::NimbleInternalError);
#undef TEST_STREAM_CHUNKER_FOR_TYPE
}

TEST_F(StreamChunkerTestsBase, shouldOmitNullStream) {
  const auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Bool);
  const auto descriptor = &scalarTypeBuilder->scalarDescriptor();

  // No Omit Case: Has nulls and non nulls size greater than minChunkSize.
  {
    NullsStreamData nullsStream(
        *leafPool_, *descriptor, *inputBufferGrowthPolicy_);
    std::vector<bool> nonNullsTestData = {true, false};
    nullsStream.ensureAdditionalNullsCapacity(
        /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsTestData.size()));
    auto& nonNulls = nullsStream.mutableNonNulls();
    populateData(nonNulls, nonNullsTestData);
    EXPECT_FALSE(
        detail::shouldOmitNullStream(
            nullsStream,
            /*minChunkSize=*/1,
            /*isFirstChunk=*/false));
  }

  // No validity bitmap.
  {
    NullsStreamData nullsStream(
        *leafPool_, *descriptor, *inputBufferGrowthPolicy_);
    nullsStream.ensureAdditionalNullsCapacity(
        /*mayHaveNulls=*/false, /*additionalSize=*/2);

    EXPECT_TRUE(
        detail::shouldOmitNullStream(
            nullsStream,
            /*minChunkSize=*/1,
            /*isFirstChunk=*/true));
    EXPECT_FALSE(
        detail::shouldOmitNullStream(
            nullsStream,
            /*minChunkSize=*/1,
            /*isFirstChunk=*/false));
  }

  // Allocated validity bitmap, but all values are non-null.
  {
    NullsStreamData nullsStream(
        *leafPool_, *descriptor, *inputBufferGrowthPolicy_);
    std::vector<bool> nonNullsTestData = {true, true};
    nullsStream.ensureAdditionalNullsCapacity(
        /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsTestData.size()));
    auto& nonNulls = nullsStream.mutableNonNulls();
    populateData(nonNulls, nonNullsTestData);

    EXPECT_TRUE(
        detail::shouldOmitNullStream(
            nullsStream,
            /*minChunkSize=*/1,
            /*isFirstChunk=*/true));
    EXPECT_FALSE(
        detail::shouldOmitNullStream(
            nullsStream,
            /*minChunkSize=*/1,
            /*isFirstChunk=*/false));
  }

  // Has nulls.
  {
    NullsStreamData nullsStream(
        *leafPool_, *descriptor, *inputBufferGrowthPolicy_);
    std::vector<bool> nonNullsTestData = {true, false};
    nullsStream.ensureAdditionalNullsCapacity(
        /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsTestData.size()));
    auto& nonNulls = nullsStream.mutableNonNulls();
    populateData(nonNulls, nonNullsTestData);

    // No omit stream when size above minChunkSize.
    EXPECT_FALSE(
        detail::shouldOmitNullStream(
            nullsStream,
            /*minChunkSize=*/1,
            /*isFirstChunk=*/true));

    // Omit stream when size below minChunkSize.
    EXPECT_TRUE(
        detail::shouldOmitNullStream(
            nullsStream,
            /*minChunkSize=*/10,
            /*isFirstChunk=*/true));
  }
}

TEST_F(StreamChunkerTestsBase, shouldOmitDataStream) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Bool);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  NullsStreamData nullsStream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);

  // Setup test data with some nulls
  std::vector<bool> nonNullsTestData = {
      true, false, true, true, false, true, false, true, false, false};
  nullsStream.ensureAdditionalNullsCapacity(
      /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsTestData.size()));
  auto& nonNulls = nullsStream.mutableNonNulls();
  populateData(nonNulls, nonNullsTestData);

  scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  descriptor = &scalarTypeBuilder->scalarDescriptor();
  NullableContentStreamData<int32_t> nullableContentStream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);
  std::vector<int32_t> testData = {1, 2, 3, 4, 5};
  // Populate data using ensureAdditionalNullsCapacity and manual data insertion
  nullableContentStream.ensureAdditionalNullsCapacity(
      /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsTestData.size()));
  populateData(nullableContentStream.mutableData(), testData);
  populateData(nullableContentStream.mutableNonNulls(), nonNullsTestData);

  // Nullable Content Stream Basic Test
  {
    // Non-empty stream with size above minChunkSize
    EXPECT_FALSE(
        detail::shouldOmitDataStream(
            nullableContentStream.data().size(),
            /*minChunkSize=*/4,
            nullableContentStream.nonNulls().empty(),
            /*isFirstChunk=*/false));

    // Non-empty stream with size below minChunkSize
    EXPECT_TRUE(
        detail::shouldOmitDataStream(
            nullableContentStream.data().size(),
            /*minChunkSize=*/100,
            nullableContentStream.nonNulls().empty(),

            /*isFirstChunk=*/true));

    // Non-empty stream with size below minChunkSize
    // and non-empty written stream content
    EXPECT_FALSE(
        detail::shouldOmitDataStream(
            nullableContentStream.data().size(),
            /*minChunkSize=*/100,
            nullableContentStream.nonNulls().empty(),
            /*isFirstChunk=*/false));
  }
}

TEST_F(StreamChunkerTestsBase, ensureShouldOmitStreamIsRespected) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Bool);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  NullsStreamData nullsStream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);

  // Setup test data with some nulls
  std::vector<bool> nonNullsTestData = {
      true, false, true, true, false, true, false, true, false, false};
  nullsStream.ensureAdditionalNullsCapacity(
      /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsTestData.size()));
  auto& nonNulls = nullsStream.mutableNonNulls();
  populateData(nonNulls, nonNullsTestData);
  //   NullsStreamChunker respects omitStream
  {
    // Test case where omitStream returns true - chunker should return nullopt
    auto chunker = getStreamChunker(
        nullsStream,
        {
            .minChunkSize = 11, // Set high minChunkSize to trigger omitStream
            .maxChunkSize = 4,
            .ensureFullChunks = false,
            .isFirstChunk = true,
        });

    // Validate that the correct chunker type was created
    ASSERT_NE(dynamic_cast<NullsStreamChunker*>(chunker.get()), nullptr);

    // Should return nullopt because omitStream should return true
    auto result = chunker->next();
    EXPECT_FALSE(result.has_value());

    // Test case where omitStream returns false
    auto chunker2 = getStreamChunker(
        nullsStream,
        {
            .minChunkSize = 4, // Set reasonable minChunkSize
            .maxChunkSize = 4,
            .ensureFullChunks = false,
        });

    // Should return valid chunks because omitStream should return false
    auto result2 = chunker2->next();
    EXPECT_TRUE(result2.has_value());
  }

  scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  descriptor = &scalarTypeBuilder->scalarDescriptor();
  NullableContentStreamData<int32_t> nullableContentStream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);
  std::vector<int32_t> testData = {1, 2, 3, 4, 5};
  // Populate data using ensureAdditionalNullsCapacity and manual data
  // insertion
  nullableContentStream.ensureAdditionalNullsCapacity(
      /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsTestData.size()));
  populateData(nullableContentStream.mutableData(), testData);
  populateData(nullableContentStream.mutableNonNulls(), nonNullsTestData);
  //   NullableContentStreamChunker respects omitStream
  {
    // Test case where omitStream returns true - chunker should return nullopt
    auto chunker = getStreamChunker(
        nullableContentStream,
        {
            .minChunkSize = 100, // Set high minChunkSize to trigger omitStream
            .maxChunkSize = 20,
            .ensureFullChunks = false,
            .isFirstChunk = true,
        });

    // Validate that the correct chunker type was created
    ASSERT_NE(
        dynamic_cast<NullableContentStreamChunker<int32_t>*>(chunker.get()),
        nullptr);

    // Should return nullopt because omitStream should return true
    auto result = chunker->next();
    EXPECT_FALSE(result.has_value());

    // Test case where omitStream returns false
    auto chunker2 = getStreamChunker(
        nullableContentStream,
        {
            .minChunkSize = 4, // Set reasonable minChunkSize
            .maxChunkSize = 20,
            .ensureFullChunks = false,
        });

    // Should return valid chunks because omitStream should return false
    auto result2 = chunker2->next();
    EXPECT_TRUE(result2.has_value());
  }
}

TEST_F(StreamChunkerTestsBase, contentStreamChunker) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  StreamChunkerOptions options{.minChunkSize = 4, .maxChunkSize = 30};

  // ContentStreamData can be used to create a ContentStreamChunker.
  ContentStreamData<int32_t> stream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);
  ASSERT_NE(
      dynamic_cast<ContentStreamChunker<int32_t>*>(
          getStreamChunker(stream, options).get()),
      nullptr);

  // NullableContentStreamData can be used to create a ContentStreamChunker.
  NullableContentStreamData<int32_t> nullableStream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);
  auto streamChunker = getStreamChunker(nullableStream, options);
  auto contentStreamChunker = dynamic_cast<
      ContentStreamChunker<int32_t, NullableContentStreamData<int32_t>>*>(
      streamChunker.get());
  ASSERT_NE(contentStreamChunker, nullptr);
}

TEST_F(StreamChunkerTestsBase, contentStreamIntChunking) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  ContentStreamData<int32_t> stream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);

  uint64_t maxChunkSize = 18; // 4.5 elements
  uint64_t minChunkSize = 2; // 0 elements
  // Test 1: Ensure Full Chunks. Min Size is Ignored.
  {
    std::vector<int32_t> testData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto& data = stream.mutableData();
    populateData(data, testData);
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = true,
        });
    // Validate that the correct chunker type was created
    ASSERT_NE(
        dynamic_cast<ContentStreamChunker<int32_t>*>(chunker.get()), nullptr);
    std::vector<ExpectedChunk<int32_t>> expectedChunks = {
        {.chunkData = {1, 2, 3, 4}}, // First chunk: 4 elements
        {.chunkData = {5, 6, 7, 8}} // Second chunk: 4 elements
        // Remaining 2 elements won't be chunked due to minChunkSize
    };
    // Expected retained data after compaction
    ExpectedChunk<int32_t> expectedRetainedData{.chunkData = {9, 10}};
    validateChunk<int32_t>(
        stream, std::move(chunker), expectedChunks, expectedRetainedData);
  }

  // When ensureFullChunks = false.
  {
    // Test 2: Nothing is returned when chunk is below minChunkSize
    minChunkSize = 12; // 3 elements
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = false,
        });
    // Validate that the correct chunker type was created
    ExpectedChunk<int32_t> expectedRetainedData{.chunkData = {9, 10}};
    validateChunk<int32_t>(
        stream, std::move(chunker), {}, expectedRetainedData);

    // Test 3: Everything is returned when chunk is above minChunkSize
    minChunkSize = 2;
    auto chunker2 = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = false,
        });
    // Validate that the correct chunker type was created
    std::vector<ExpectedChunk<int32_t>> expectedChunks = {
        {.chunkData = {9, 10}}};
    validateChunk<int32_t>(stream, std::move(chunker2), expectedChunks, {});
  }

  // Test 4: We can reuse a stream post compaction.
  {
    auto& data = stream.mutableData();
    std::vector<int32_t> testData = {7, 8, 9, 10, 11};
    populateData(data, testData);

    maxChunkSize = 12; // 3 elements
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = false,
        });
    // Validate that the correct chunker type was created
    ASSERT_NE(
        dynamic_cast<ContentStreamChunker<int32_t>*>(chunker.get()), nullptr);

    std::vector<ExpectedChunk<int32_t>> expectedChunks = {
        {.chunkData = {7, 8, 9}}, // First chunk: 3 elements
        {.chunkData = {10, 11}}, // Second chunk: 2 elements
    };
    // No data retained after processing all chunks
    validateChunk<int32_t>(stream, std::move(chunker), expectedChunks, {});
  }
}

TEST_F(StreamChunkerTestsBase, contentStreamStringChunking) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::String);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  ContentStreamData<std::string_view> stream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);
  uint64_t maxChunkSize = 55;
  uint64_t minChunkSize = 1;
  // Test 1: Ensure Full Chunks. Min Size is Ignored.
  {
    std::vector<std::string_view> testData = {
        "short", "a_longer_string", "x", "medium_size", "tiny"};
    auto& data = stream.mutableData();
    populateData(data, testData);

    // Calculate extra memory for string content
    for (const auto& str : testData) {
      stream.extraMemory() += str.size();
    }

    // Total string content sizes:
    // "short"(5) + "a_longer_string"(15) + "x"(1) + "medium_size"(11) +
    // "tiny"(4) = 36 bytes
    ASSERT_EQ(stream.extraMemory(), 36);
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = true,
        });
    std::vector<ExpectedChunk<std::string_view>> expectedChunks = {
        {.chunkData = {"short", "a_longer_string"},
         .chunkNonNulls = {},
         .hasNulls = false,
         .extraMemory =
             20}, // "short"(5) + "a_longer_string"(15) = 20 bytes extra
        {.chunkData = {"x", "medium_size"},
         .chunkNonNulls = {},
         .hasNulls = false,
         .extraMemory = 12} // "x"(1) + "medium_size"(11) = 12 bytes extra
        // "tiny"(4) remains due to minChunkSize constraint
    };

    ExpectedChunk<std::string_view> expectedRetainedData{
        .chunkData = {"tiny"},
        .chunkNonNulls = {},
        .hasNulls = false,
        .extraMemory = 4}; // "tiny"(4) = 4 bytes extra
    validateChunk<std::string_view>(
        stream, std::move(chunker), expectedChunks, expectedRetainedData);
  }

  // When ensureFullChunks = false
  {
    // Test 2: Nothing is returned when chunk is below minChunkSize
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = 30,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = false,
        });

    ExpectedChunk<std::string_view> lastChunk = {
        .chunkData = {"tiny"},
        .chunkNonNulls = {},
        .hasNulls = false,
        .extraMemory = 4 // "tiny"(4) = 4 bytes extra
    };
    validateChunk<std::string_view>(stream, std::move(chunker), {}, lastChunk);

    // Test 3: Everything is returned when chunk is above minChunkSize
    auto chunker2 = getStreamChunker(
        stream,
        {
            .minChunkSize = 0,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = false,
        });
    validateChunk<std::string_view>(
        stream, std::move(chunker2), {lastChunk}, {});
  }

  // Test 4: We can reuse a stream post compaction.
  {
    auto& data = stream.mutableData();
    stream.extraMemory() = 0; // Reset extra memory
    std::vector<std::string_view> testData = {
        "hello", "world", "hello", "world"};
    populateData(data, testData);

    // Calculate extra memory for string content
    for (const auto& str : testData) {
      stream.extraMemory() += str.size();
    }

    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = 0,
            .maxChunkSize = 42,
            .ensureFullChunks = true,
        });

    std::vector<ExpectedChunk<std::string_view>> expectedChunks = {
        {.chunkData = {"hello", "world"},
         .chunkNonNulls = {},
         .hasNulls = false,
         .extraMemory = 10}, // "hello"(5) + "world"(5) = 10 bytes extra
        {.chunkData = {"hello", "world"},
         .chunkNonNulls = {},
         .hasNulls = false,
         .extraMemory = 10}, // "hello"(5) + "world"(5) = 10 bytes extra
    };
    validateChunk<std::string_view>(
        stream, std::move(chunker), expectedChunks, {});
  }

  // Test 5: Single string exceeds maxChunkSize.
  {
    std::vector<std::string_view> testData = {
        "a", "short", "really really large string", "small"};
    auto& data = stream.mutableData();
    populateData(data, testData);

    // Size of "really really large string" and string view minus 1.
    maxChunkSize = 24 + sizeof(std::string_view);

    // Calculate extra memory for string content.
    for (const auto& str : testData) {
      stream.extraMemory() += str.size();
    }

    // "a"(1), "short"(5), "really really large string"(25), "small"(5) = 37
    // bytes of extra memory.
    ASSERT_EQ(stream.extraMemory(), 37);
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = true,
        });
    validateChunk<std::string_view>(
        stream,
        std::move(chunker),
        {{.chunkData = {"a", "short"}, .extraMemory = 6},
         {.chunkData = {"really really large string"}, .extraMemory = 25}},
        {.chunkData = {"small"}, .extraMemory = 5});
  }
}

TEST_F(StreamChunkerTestsBase, nullsStreamChunking) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Bool);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  NullsStreamData stream(*leafPool_, *descriptor, *inputBufferGrowthPolicy_);

  // Setup test data with some nulls
  std::vector<bool> testData = {
      true, false, true, true, false, true, false, true, false, false};
  stream.ensureAdditionalNullsCapacity(
      /*mayHaveNulls=*/true, static_cast<uint32_t>(testData.size()));
  auto& nonNulls = stream.mutableNonNulls();
  populateData(nonNulls, testData);

  // size of data is 10 * sizeof(bool) = 10 bytes
  ASSERT_EQ(stream.memoryUsed(), 10);
  EXPECT_EQ(stream.data().size(), 0);

  // 4 bytes (arbitrary chunk size)
  uint32_t maxChunkSize = 4;
  uint32_t minChunkSize = 1;

  // Test 1: Ensure Full Chunks. Min Size is Ignored.
  {
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = true,
        });
    // Validate that the correct chunker type was created
    ASSERT_NE(dynamic_cast<NullsStreamChunker*>(chunker.get()), nullptr);

    std::vector<ExpectedChunk<bool>> expectedChunks = {
        {.chunkData = {true, false, true, true},
         .chunkNonNulls = {},
         .hasNulls = false},
        {.chunkData = {false, true, false, true},
         .chunkNonNulls = {},
         .hasNulls = false}};
    // Expected retained data after compaction (no data, but non-nulls remain)
    ExpectedChunk<bool> expectedRetainedData{
        .chunkData = {}, .chunkNonNulls = {false, false}, .hasNulls = true};
    validateChunk<bool>(
        stream, std::move(chunker), expectedChunks, expectedRetainedData);
  }

  // When ensureFullChunks = false
  {
    // Test 2: Nothing is returned when chunk is below minChunkSize
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = 3,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = false,
        });

    validateChunk<bool>(
        stream,
        std::move(chunker),
        {},
        {.chunkNonNulls = {false, false}, .hasNulls = true});

    // Test 3: Everything is returned when chunk is above minChunkSize
    auto chunker2 = getStreamChunker(
        stream,
        {
            .minChunkSize = 1,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = false,
        });
    // No data retained after processing all chunks
    validateChunk<bool>(
        stream, std::move(chunker2), {{.chunkData = {false, false}}}, {});
  }

  // Test 4: We can reuse a stream post compaction.
  {
    testData = {
        true, false, true, true, false, true, false, true, false, false};
    stream.ensureAdditionalNullsCapacity(
        /*mayHaveNulls=*/true, static_cast<uint32_t>(testData.size()));
    nonNulls = stream.mutableNonNulls();
    populateData(nonNulls, testData);
    auto memoryUsed = stream.memoryUsed();
    // When maxChunkSize is equal to available memory, chunk is returned.
    auto chunker2 = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = memoryUsed,
            .ensureFullChunks = true,
        });
    auto expectedChunks = {ExpectedChunk<bool>{
        .chunkData = testData, .chunkNonNulls = {}, .hasNulls = false}};
    // No data retained after processing all chunks
    validateChunk<bool>(stream, std::move(chunker2), expectedChunks, {});
  }
}

TEST_F(StreamChunkerTestsBase, nullableContentStreamIntChunking) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  NullableContentStreamData<int32_t> stream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);

  // Setup test data with some nulls
  std::vector<int32_t> testData = {1, 2, 3, 4, 5, 6, 7};
  std::vector<bool> nonNullsData = {
      true, false, true, true, false, true, false, true, true, true};

  // Populate data using ensureAdditionalNullsCapacity and manual data
  // insertion
  stream.ensureAdditionalNullsCapacity(
      /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsData.size()));
  populateData(stream.mutableData(), testData);
  populateData(stream.mutableNonNulls(), nonNullsData);

  // 7 * sizeof(int32_t) + 10 * sizeof(bool)) = 38
  ASSERT_EQ(stream.memoryUsed(), 38);

  // Test 1: Ensure Full Chunks. Min Size is Ignored
  uint64_t maxChunkSize = 17;
  uint64_t minChunkSize = 1;
  {
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = true,
        });

    std::vector<ExpectedChunk<int32_t>> expectedChunks = {
        {.chunkData = {1, 2, 3},
         .chunkNonNulls = {true, false, true, true, false},
         .hasNulls = true},
        {.chunkData = {4, 5, 6},
         .chunkNonNulls = {true, false, true, true},
         .hasNulls = true}};
    // Expected retained data after compaction
    ExpectedChunk<int32_t> expectedRetainedData{
        .chunkData = {7}, .chunkNonNulls = {true}, .hasNulls = true};
    validateChunk<int32_t>(
        stream, std::move(chunker), expectedChunks, expectedRetainedData);
  }

  // When ensureFullChunks = false
  {
    // Test 2: Nothing is returned when chunk is below minChunkSize
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = 80,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = false,
        });
    // Validate that the correct chunker type was created
    ASSERT_NE(
        dynamic_cast<NullableContentStreamChunker<int32_t>*>(chunker.get()),
        nullptr);
    // Nothing is returned
    validateChunk<int32_t>(
        stream,
        std::move(chunker),
        {},
        {.chunkData = {7}, .chunkNonNulls = {true}, .hasNulls = true});
    // Test 3: Everything is returned when chunk is above minChunkSize
    auto chunker2 = getStreamChunker(
        stream,
        {
            .minChunkSize = 1,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = false,
        });
    // Everything is returned.
    validateChunk<int32_t>(
        stream,
        std::move(chunker2),
        {{.chunkData = {7}, .hasNulls = false}},
        {});
  }

  // Test 4: We can reuse a stream post compaction, while properly handling
  // retained memory.
  {
    std::vector<int32_t> newTestData = {10, 11, 12, 13, 14};
    std::vector<bool> test3NonNullsData = {true, true, true};
    stream.ensureAdditionalNullsCapacity(
        /*mayHaveNulls=*/false,
        static_cast<uint32_t>(test3NonNullsData.size()));
    auto& data = stream.mutableData();
    auto& nonNulls = stream.mutableNonNulls();
    populateData(data, newTestData);
    populateData(nonNulls, test3NonNullsData);

    minChunkSize = sizeof(int32_t) * 3;
    auto chunker = getStreamChunker(
        stream,
        {.minChunkSize = minChunkSize,
         .maxChunkSize = minChunkSize,
         .ensureFullChunks = true});
    // NullableContentStreamData as ContentStreamData when mayHaveNulls=false
    auto contentStreamChunker = dynamic_cast<
        ContentStreamChunker<int32_t, NullableContentStreamData<int32_t>>*>(
        chunker.get());
    ASSERT_NE(contentStreamChunker, nullptr);
    std::vector<ExpectedChunk<int32_t>> expectedChunks = {
        {.chunkData = {10, 11, 12}}};
    ExpectedChunk<int32_t> retainedData = {.chunkData = {13, 14}};
    validateChunk<int32_t>(
        stream, std::move(chunker), expectedChunks, retainedData);
    ASSERT_EQ(stream.mutableNonNulls().size(), 0);
    // After materialization, mutable non-nulls should reflect leftover data
    stream.materialize();
    ASSERT_EQ(stream.mutableNonNulls().size(), 2);
  }
}

TEST_F(StreamChunkerTestsBase, nullableContentStreamStringChunking) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::String);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  NullableContentStreamData<std::string_view> stream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);

  // Setup test data with some nulls
  std::vector<std::string_view> testData = {
      "hello", "world", "test", "string", "chunk", "data", "example"};
  std::vector<bool> nonNullsData = {
      true, false, true, true, false, true, false, true, true, true, false};

  // Populate data using ensureAdditionalNullsCapacity and manual data
  // insertion
  stream.ensureAdditionalNullsCapacity(
      /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsData.size()));
  auto& data = stream.mutableData();
  auto& nonNulls = stream.mutableNonNulls();
  populateData(data, testData);
  populateData(nonNulls, nonNullsData);

  // Set extra memory for string overhead
  // number of strings: 7
  // size of string view data pointers = 7 * 8 = 56 bytes
  // size of string view data size = 7 * 8 = 56 bytes
  // size of nulls = 11 * 1 = 11 bytes
  // total size of stored string data = 36 bytes
  // total size of stream data = 56 + 56 + 11 + 36 = 159 bytes
  for (const auto& entry : testData) {
    stream.extraMemory() += entry.size();
  }
  ASSERT_EQ(stream.memoryUsed(), 159);

  // Test 1: Not last chunk
  {
    const uint64_t maxChunkSize = 72;
    const uint64_t minChunkSize = 50;
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = true,
        });

    std::vector<ExpectedChunk<std::string_view>> expectedChunks = {
        {.chunkData = {"hello", "world", "test"},
         .chunkNonNulls = {true, false, true, true, false},
         .hasNulls = true,
         .extraMemory =
             14}, // "hello"(5) + "world"(5) + "test"(4) = 14 bytes extra
        {.chunkData = {"string", "chunk", "data"},
         .chunkNonNulls = {true, false, true, true},
         .hasNulls = true,
         .extraMemory = 15}
        // "string"(6) + "chunk"(5) + "data"(4) = 14 bytes extra
    };
    // Expected retained data after compaction
    ExpectedChunk<std::string_view> expectedRetainedData{
        .chunkData = {"example"},
        .chunkNonNulls = {true, false},
        .hasNulls = true,
        .extraMemory = 7};
    validateChunk<std::string_view>(
        stream, std::move(chunker), expectedChunks, expectedRetainedData);
  }

  // When ensureFullChunks = false
  {
    // Test 2: Nothing is returned when chunk is below minChunkSize
    const uint64_t maxChunkSize = 35;
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = stream.memoryUsed() + 1,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = false,
        });

    ExpectedChunk<std::string_view> lastChunk{
        .chunkData = {"example"},
        .chunkNonNulls = {true, false},
        .hasNulls = true,
        .extraMemory = 7 // "example"(7) = 7 bytes extra
    };

    // Nothing is returned
    validateChunk<std::string_view>(
        stream, std::move(chunker), {}, {lastChunk});

    // Test 3: Everything is returned when chunk is above minChunkSize
    auto chunker2 = getStreamChunker(
        stream,
        {
            .minChunkSize = stream.memoryUsed(),
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = false,
        });
    validateChunk<std::string_view>(
        stream, std::move(chunker2), {lastChunk}, {});
  }

  // Test 4: We can reuse a stream post compaction, while properly handling
  // retained memory.
  {
    std::vector<std::string_view> smallTestData = {"extra_data", "test"};
    std::vector<bool> smallNonNullsData = {true, true};
    stream.ensureAdditionalNullsCapacity(
        /*mayHaveNulls=*/false,
        static_cast<uint32_t>(smallNonNullsData.size()));
    auto& smallData = stream.mutableData();
    auto& smallNonNulls = stream.mutableNonNulls();
    populateData(smallData, smallTestData);
    populateData(smallNonNulls, smallNonNullsData);

    // Reset extra memory for new test data
    stream.extraMemory() = 0;
    for (const auto& entry : smallTestData) {
      stream.extraMemory() += entry.size();
    }

    const uint64_t minChunkSize =
        smallTestData.at(0).size() + sizeof(std::string_view);
    const uint64_t maxChunkSize =
        minChunkSize; // Exactly what is needed to fit just the first entry
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = true,
        });

    // NullableContentStreamData as ContentStreamData when mayHaveNulls=false
    auto contentStreamChunker = dynamic_cast<ContentStreamChunker<
        std::string_view,
        NullableContentStreamData<std::string_view>>*>(chunker.get());
    ASSERT_NE(contentStreamChunker, nullptr);
    validateChunk<std::string_view>(
        stream,
        std::move(chunker),
        {{.chunkData = {"extra_data"}}},
        {.chunkData = {"test"}, .extraMemory = 4});
    ASSERT_EQ(stream.mutableNonNulls().size(), 0);
    // After materialization, mutable non-nulls should reflect leftover data
    stream.materialize();
    ASSERT_EQ(stream.mutableNonNulls().size(), 1);
    stream.reset();
  }

  // Test 5: Single string exceeds maxChunkSize.
  {
    testData = {"a", "short", "really really large string", "small"};
    populateData(data, testData);

    nonNullsData = {true, false, true, true, false, true, false};
    stream.ensureAdditionalNullsCapacity(
        /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsData.size()));
    populateData(nonNulls, nonNullsData);

    const uint64_t minChunkSize = 1;
    // Size of "really really large string", string view, and null minus 1.
    const uint64_t maxChunkSize = 24 + sizeof(std::string_view) + sizeof(bool);

    // Calculate extra memory for string content.
    for (const auto& str : testData) {
      stream.extraMemory() += str.size();
    }

    // "a"(1), "short"(5), "really really large string"(25), "small"(5) =
    // 37 bytes of extra memory.
    ASSERT_EQ(stream.extraMemory(), 37);
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = true,
        });
    validateChunk<std::string_view>(
        stream,
        std::move(chunker),
        {{.chunkData = {"a", "short"},
          .chunkNonNulls = {true, false, true},
          .hasNulls = true,
          .extraMemory = 6},
         {.chunkData = {"really really large string"}, .extraMemory = 25}},
        {.chunkData = {"small"},
         .chunkNonNulls = {false, true, false},
         .hasNulls = true,
         .extraMemory = 5});
  }
}

TEST_F(StreamChunkerTestsBase, nullableContentStringStreamChunking) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::String);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  NullableContentStringStreamData stream(
      *leafPool_,
      *descriptor,
      *inputBufferGrowthPolicy_, /* lengthsGrowthPolicy */
      *inputBufferGrowthPolicy_ /* bufferGrowthPolicy */);

  // Setup test data with some nulls
  std::vector<std::string_view> testData = {
      "hello", "world", "test", "string", "chunk", "data", "example"};
  std::vector<bool> nonNullsData = {
      true, false, true, true, false, true, false, true, true, true, false};

  stream.ensureAdditionalNullsCapacity(
      /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsData.size()));
  auto& nonNulls = stream.mutableNonNulls();
  populateData(nonNulls, nonNullsData);

  auto data = stream.mutableData();
  populateStringData(data, testData);

  // number of strings: 7
  // size of string buffer length = 7 * 16 = 112 bytes
  // size of nulls = 11 * 1 = 11 bytes
  // total number of characters = 36 bytes
  // total size of stream data = 112 + 11 + 36 = 159 bytes
  ASSERT_EQ(stream.memoryUsed(), 159);

  // Test 1: Not last chunk
  {
    const uint64_t maxChunkSize = 72;
    const uint64_t minChunkSize = 50;
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = true,
        });

    std::vector<ExpectedChunk<std::string_view>> expectedChunks = {
        {.chunkData = {"hello", "world", "test"},
         .chunkNonNulls = {true, false, true, true, false},
         .hasNulls = true},
        {.chunkData = {"string", "chunk", "data"},
         .chunkNonNulls = {true, false, true, true},
         .hasNulls = true}};
    // Expected retained data after compaction
    ExpectedChunk<std::string_view> expectedRetainedData{
        .chunkData = {"example"},
        .chunkNonNulls = {true, false},
        .hasNulls = true};
    validateChunk<std::string_view>(
        stream, std::move(chunker), expectedChunks, expectedRetainedData);
  }

  // When ensureFullChunks = false
  {
    // Test 2: Nothing is returned when chunk is below minChunkSize
    const uint64_t maxChunkSize = 35;
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = stream.memoryUsed() + 1,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = false,
        });

    ExpectedChunk<std::string_view> lastChunk{
        .chunkData = {"example"},
        .chunkNonNulls = {true, false},
        .hasNulls = true};

    // Nothing is returned
    validateChunk<std::string_view>(
        stream, std::move(chunker), {}, {lastChunk});

    // Test 3: Everything is returned when chunk is above minChunkSize
    auto chunker2 = getStreamChunker(
        stream,
        {
            .minChunkSize = stream.memoryUsed(),
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = false,
        });
    validateChunk<std::string_view>(
        stream, std::move(chunker2), {lastChunk}, {});
  }

  // Test 4: We can reuse a stream post compaction.
  {
    std::vector<std::string_view> smallTestData = {"extra_data", "test"};
    std::vector<bool> smallNonNullsData = {true, true};
    stream.ensureAdditionalNullsCapacity(
        /*mayHaveNulls=*/false,
        static_cast<uint32_t>(smallNonNullsData.size()));
    auto smallData = stream.mutableData();
    populateStringData(smallData, smallTestData);

    auto& smallNonNulls = stream.mutableNonNulls();
    populateData(smallNonNulls, smallNonNullsData);

    // Exactly what is needed to fit just the first entry
    const uint64_t minChunkSize =
        smallTestData.at(0).size() + sizeof(std::string_view);
    const uint64_t maxChunkSize = minChunkSize;
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = true,
        });

    validateChunk<std::string_view>(
        stream,
        std::move(chunker),
        {{.chunkData = {"extra_data"}}},
        {.chunkData = {"test"}});
    ASSERT_EQ(stream.mutableNonNulls().size(), 0);
    // After materialization, mutable non-nulls should reflect leftover data
    stream.materialize();
    ASSERT_EQ(stream.mutableNonNulls().size(), 1);
    stream.reset();
  }

  // Test 5: Single string exceeds maxChunkSize.
  {
    testData = {"a", "short", "really really large string", "small"};
    populateStringData(data, testData);

    nonNullsData = {true, false, true, true, false, true, false};
    stream.ensureAdditionalNullsCapacity(
        /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsData.size()));
    populateData(nonNulls, nonNullsData);

    const uint64_t minChunkSize = 1;
    // Size of "really really large string", string view, and null minus 1.
    const uint64_t maxChunkSize = 24 + sizeof(std::string_view) + sizeof(bool);
    auto chunker = getStreamChunker(
        stream,
        {
            .minChunkSize = minChunkSize,
            .maxChunkSize = maxChunkSize,
            .ensureFullChunks = true,
        });
    validateChunk<std::string_view>(
        stream,
        std::move(chunker),
        {{.chunkData = {"a", "short"},
          .chunkNonNulls = {true, false, true},
          .hasNulls = true},
         {.chunkData = {"really really large string"}, .extraMemory = 25}},
        {.chunkData = {"small"},
         .chunkNonNulls = {false, true, false},
         .hasNulls = true});
  }
}

// Tests to verify rowCount is set properly for chunked StreamDataViews.
TEST_F(StreamChunkerTestsBase, contentStreamChunkerRowCount) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  ContentStreamData<int32_t> stream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);

  // Add 10 elements.
  std::vector<int32_t> testData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto& data = stream.mutableData();
  populateData(data, testData);

  // Max chunk size = 16 bytes = 4 int32_t elements.
  auto chunker = getStreamChunker(
      stream,
      {
          .minChunkSize = 1,
          .maxChunkSize = 16,
          .ensureFullChunks = false,
      });

  // Verify rowCount for each chunk.
  std::vector<uint32_t> expectedRowCounts = {4, 4, 2};
  size_t chunkIndex = 0;
  while (const auto chunkView = chunker->next()) {
    ASSERT_LT(chunkIndex, expectedRowCounts.size());
    EXPECT_EQ(chunkView->rowCount(), expectedRowCounts[chunkIndex])
        << "Chunk " << chunkIndex << " has incorrect rowCount";
    // Verify data size matches rowCount * sizeof(T).
    EXPECT_EQ(
        chunkView->data().size(), chunkView->rowCount() * sizeof(int32_t));
    ++chunkIndex;
  }
  EXPECT_EQ(chunkIndex, expectedRowCounts.size());
}

TEST_F(StreamChunkerTestsBase, contentStreamStringChunkerRowCount) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::String);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  ContentStreamData<std::string_view> stream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);

  // Add strings with varying sizes.
  std::vector<std::string_view> testData = {
      "a", "bb", "ccc", "dddd", "eeeee", "ffffff"};
  auto& data = stream.mutableData();
  populateData(data, testData);
  for (const auto& str : testData) {
    stream.extraMemory() += str.size();
  }

  // Max chunk size that fits ~2-3 strings per chunk.
  auto chunker = getStreamChunker(
      stream,
      {
          .minChunkSize = 1,
          .maxChunkSize = 40,
          .ensureFullChunks = false,
      });

  // Verify rowCount matches the number of string_views in each chunk.
  uint32_t totalRowCount = 0;
  while (const auto chunkView = chunker->next()) {
    EXPECT_GT(chunkView->rowCount(), 0);
    // For strings, data size = rowCount * sizeof(string_view).
    EXPECT_EQ(
        chunkView->data().size(),
        chunkView->rowCount() * sizeof(std::string_view));
    totalRowCount += chunkView->rowCount();
  }
  EXPECT_EQ(totalRowCount, testData.size());
}

TEST_F(StreamChunkerTestsBase, nullsStreamChunkerRowCount) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Bool);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  NullsStreamData stream(*leafPool_, *descriptor, *inputBufferGrowthPolicy_);

  // Add 10 null indicators.
  std::vector<bool> testData = {
      true, false, true, true, false, true, false, true, false, false};
  stream.ensureAdditionalNullsCapacity(
      /*mayHaveNulls=*/true, static_cast<uint32_t>(testData.size()));
  auto& nonNulls = stream.mutableNonNulls();
  populateData(nonNulls, testData);

  // Max chunk size = 4 bytes = 4 bool elements.
  auto chunker = getStreamChunker(
      stream,
      {
          .minChunkSize = 1,
          .maxChunkSize = 4,
          .ensureFullChunks = false,
      });

  // Verify rowCount for each chunk.
  std::vector<uint32_t> expectedRowCounts = {4, 4, 2};
  size_t chunkIndex = 0;
  while (const auto chunkView = chunker->next()) {
    ASSERT_LT(chunkIndex, expectedRowCounts.size());
    EXPECT_EQ(chunkView->rowCount(), expectedRowCounts[chunkIndex])
        << "Chunk " << chunkIndex << " has incorrect rowCount";
    // For nulls stream, data size = rowCount (1 byte per bool).
    EXPECT_EQ(chunkView->data().size(), chunkView->rowCount());
    ++chunkIndex;
  }
  EXPECT_EQ(chunkIndex, expectedRowCounts.size());
}

TEST_F(StreamChunkerTestsBase, nullableContentStreamChunkerRowCount) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  NullableContentStreamData<int32_t> stream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);

  // Add data with nulls: 10 rows, 7 non-null values.
  std::vector<int32_t> testData = {1, 2, 3, 4, 5, 6, 7};
  std::vector<bool> nonNullsData = {
      true, false, true, true, false, true, false, true, true, true};
  stream.ensureAdditionalNullsCapacity(
      /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsData.size()));
  populateData(stream.mutableData(), testData);
  populateData(stream.mutableNonNulls(), nonNullsData);

  // Max chunk size = 17 bytes (fits ~4 rows with mixed nulls).
  auto chunker = getStreamChunker(
      stream,
      {
          .minChunkSize = 1,
          .maxChunkSize = 17,
          .ensureFullChunks = false,
      });

  // Verify rowCount includes both null and non-null rows.
  uint32_t totalRowCount = 0;
  while (const auto chunkView = chunker->next()) {
    EXPECT_GT(chunkView->rowCount(), 0);
    // rowCount should match nonNulls size when nulls are present.
    if (chunkView->hasNulls()) {
      EXPECT_EQ(chunkView->rowCount(), chunkView->nonNulls().size());
    }
    totalRowCount += chunkView->rowCount();
  }
  EXPECT_EQ(totalRowCount, nonNullsData.size());
}

TEST_F(StreamChunkerTestsBase, nullableContentStreamStringChunkerRowCount) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::String);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  NullableContentStreamData<std::string_view> stream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);

  // Add data with nulls.
  std::vector<std::string_view> testData = {"hello", "world", "test", "data"};
  std::vector<bool> nonNullsData = {true, false, true, true, false, true};
  stream.ensureAdditionalNullsCapacity(
      /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsData.size()));
  auto& data = stream.mutableData();
  auto& nonNulls = stream.mutableNonNulls();
  populateData(data, testData);
  populateData(nonNulls, nonNullsData);
  for (const auto& str : testData) {
    stream.extraMemory() += str.size();
  }

  auto chunker = getStreamChunker(
      stream,
      {
          .minChunkSize = 1,
          .maxChunkSize = 50,
          .ensureFullChunks = false,
      });

  // Verify rowCount for nullable string stream.
  uint32_t totalRowCount = 0;
  while (const auto chunkView = chunker->next()) {
    EXPECT_GT(chunkView->rowCount(), 0);
    if (chunkView->hasNulls()) {
      EXPECT_EQ(chunkView->rowCount(), chunkView->nonNulls().size());
    }
    totalRowCount += chunkView->rowCount();
  }
  EXPECT_EQ(totalRowCount, nonNullsData.size());
}

TEST_F(StreamChunkerTestsBase, nullableContentStringStreamChunkerRowCount) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::String);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  NullableContentStringStreamData stream(
      *leafPool_,
      *descriptor,
      *inputBufferGrowthPolicy_, /* lengthsGrowthPolicy */
      *inputBufferGrowthPolicy_ /* bufferGrowthPolicy */);

  // Add data with nulls.
  std::vector<std::string_view> testData = {"hello", "world", "test", "data"};
  std::vector<bool> nonNullsData = {true, false, true, true, false, true};
  stream.ensureAdditionalNullsCapacity(
      /*mayHaveNulls=*/true, static_cast<uint32_t>(nonNullsData.size()));
  auto data = stream.mutableData();
  populateStringData(data, testData);

  auto& nonNulls = stream.mutableNonNulls();
  populateData(nonNulls, nonNullsData);

  auto chunker = getStreamChunker(
      stream,
      {
          .minChunkSize = 1,
          .maxChunkSize = 50,
          .ensureFullChunks = false,
      });

  // Verify rowCount for nullable string stream.
  uint32_t totalRowCount = 0;
  while (const auto chunkView = chunker->next()) {
    EXPECT_GT(chunkView->rowCount(), 0);
    if (chunkView->hasNulls()) {
      EXPECT_EQ(chunkView->rowCount(), chunkView->nonNulls().size());
    }
    totalRowCount += chunkView->rowCount();
  }
  EXPECT_EQ(totalRowCount, nonNullsData.size());
}

TEST_F(StreamChunkerTestsBase, singleElementChunkRowCount) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int64);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  ContentStreamData<int64_t> stream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);

  // Add single element.
  stream.mutableData().push_back(42);

  auto chunker = getStreamChunker(
      stream,
      {
          .minChunkSize = 1,
          .maxChunkSize = 100,
          .ensureFullChunks = false,
      });

  auto chunkView = chunker->next();
  ASSERT_TRUE(chunkView.has_value());
  EXPECT_EQ(chunkView->rowCount(), 1);
  EXPECT_EQ(chunkView->data().size(), sizeof(int64_t));

  // No more chunks.
  EXPECT_FALSE(chunker->next().has_value());
}

TEST_F(StreamChunkerTestsBase, rowCountAcrossChunks) {
  auto scalarTypeBuilder =
      schemaBuilder_->createScalarTypeBuilder(ScalarKind::Int32);
  auto descriptor = &scalarTypeBuilder->scalarDescriptor();
  ContentStreamData<int32_t> stream(
      *leafPool_, *descriptor, *inputBufferGrowthPolicy_);

  // Add 100 elements.
  for (int32_t i = 0; i < 100; ++i) {
    stream.mutableData().push_back(i);
  }

  // Small chunk size to create many chunks.
  auto chunker = getStreamChunker(
      stream,
      {
          .minChunkSize = 1,
          .maxChunkSize = 20, // 5 int32_t per chunk
          .ensureFullChunks = false,
      });

  // Verify total rowCount across all chunks equals original data size.
  uint32_t totalRowCount = 0;
  uint32_t chunkCount = 0;
  while (const auto chunkView = chunker->next()) {
    EXPECT_GT(chunkView->rowCount(), 0);
    EXPECT_LE(chunkView->rowCount(), 5); // Max 5 per chunk.
    totalRowCount += chunkView->rowCount();
    ++chunkCount;
  }
  EXPECT_EQ(totalRowCount, 100);
  EXPECT_EQ(chunkCount, 20); // 100 / 5 = 20 chunks.
}

} // namespace facebook::nimble
