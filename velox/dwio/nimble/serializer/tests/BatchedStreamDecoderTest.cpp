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

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "folly/container/F14Map.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/serializer/BatchedStreamDecoder.h"
#include "velox/dwio/nimble/serializer/Options.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/serializer/StreamDataParser.h"
#include "velox/dwio/nimble/serializer/StreamDataWriter.h"
#include "velox/dwio/nimble/velox/HybridFlatMap.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

using namespace facebook::nimble;

namespace {

// One serialized batch, split into its physical streams. Payloads are owned as
// `std::string` so the `string_view`s handed to `addBatch()` stay valid for as
// long as the batch does — `StreamDataParser` hands out views into either the
// serialized blob or its own scratch buffers, both of which are gone by the
// time the decoder runs.
struct SerializedBatch {
  uint32_t rowCount{0};
  SerializationVersion version{SerializationVersion::kSerialization};
  bool streamEncodingUsesVarintRowCount{true};
  folly::F14FastMap<uint32_t, std::string> streams;

  bool hasStream(uint32_t offset) const {
    return streams.contains(offset);
  }

  std::string_view stream(uint32_t offset) const {
    const auto it = streams.find(offset);
    NIMBLE_CHECK(it != streams.end(), "No stream at offset {}", offset);
    return it->second;
  }
};

struct SerializedInput {
  std::vector<SerializedBatch> batches;
  std::shared_ptr<const Type> schema;
};

class BatchedStreamDecoderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    facebook::velox::memory::MemoryManager::testingSetInstance(
        facebook::velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ = facebook::velox::memory::memoryManager()->addRootPool(
        "batched_stream_decoder_test");
    pool_ = rootPool_->addLeafChild("leaf");
  }

  static SerializerOptions serializerOptions() {
    return SerializerOptions{
        .version = SerializationVersion::kSerialization,
        .streamIndicesEncodingType = EncodingType::Trivial,
        .streamSizesEncodingType = EncodingType::Trivial,
    };
  }

  // Serializes each vector as its own batch and splits it into per-stream
  // payloads, mirroring what `Deserializer::appendStreamSegments` feeds the
  // decoder. The schema is read after every batch so dynamically discovered
  // FlatMap keys are included.
  SerializedInput serializeBatches(
      const facebook::velox::TypePtr& rowType,
      const std::vector<facebook::velox::VectorPtr>& batches,
      SerializerOptions options) {
    Serializer serializer{std::move(options), rowType, pool_.get()};
    SerializedInput input;
    for (const auto& batch : batches) {
      const std::string blob{
          serializer.serialize(batch, OrderedRanges::of(0, batch->size()))};
      DeserializerOptions deserializerOptions{};
      serde::StreamDataParser parser{pool_.get()};
      SerializedBatch collected;
      collected.rowCount = parser.initialize(blob);
      collected.version = parser.version();
      collected.streamEncodingUsesVarintRowCount =
          parser.streamEncodingUsesVarintRowCount();
      parser.iterateStreams([&](uint32_t offset, std::string_view streamData) {
        collected.streams.emplace(offset, std::string{streamData});
      });
      input.batches.emplace_back(std::move(collected));
    }
    input.schema =
        SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
    return input;
  }

  // Single INTEGER column. `nullEvery` of 0 means no nulls.
  facebook::velox::VectorPtr makeIntBatch(
      const facebook::velox::TypePtr& rowType,
      int32_t firstValue,
      facebook::velox::vector_size_t numRows,
      int32_t nullEvery = 0) {
    auto values = facebook::velox::BaseVector::create(
        facebook::velox::INTEGER(), numRows, pool_.get());
    auto* flat = values->asFlatVector<int32_t>();
    for (facebook::velox::vector_size_t i = 0; i < numRows; ++i) {
      if (nullEvery > 0 && i % nullEvery == 0) {
        flat->setNull(i, true);
      } else {
        flat->set(i, firstValue + i);
      }
    }
    return std::make_shared<facebook::velox::RowVector>(
        pool_.get(),
        rowType,
        /*nulls=*/nullptr,
        numRows,
        std::vector<facebook::velox::VectorPtr>{values});
  }

  // ROW(id BIGINT, flat_map MAP(VARCHAR, DOUBLE)) where row i carries exactly
  // the keys in `keysByRow[i]`.
  facebook::velox::VectorPtr makeFlatMapBatch(
      const facebook::velox::TypePtr& rowType,
      const std::vector<std::vector<std::string>>& keysByRow) {
    namespace vx = facebook::velox;
    const auto numRows = static_cast<vx::vector_size_t>(keysByRow.size());
    vx::vector_size_t totalEntries = 0;
    for (const auto& keys : keysByRow) {
      totalEntries += static_cast<vx::vector_size_t>(keys.size());
    }

    auto ids = vx::BaseVector::create(vx::BIGINT(), numRows, pool_.get());
    auto mapKeys =
        vx::BaseVector::create(vx::VARCHAR(), totalEntries, pool_.get());
    auto mapValues =
        vx::BaseVector::create(vx::DOUBLE(), totalEntries, pool_.get());
    for (vx::vector_size_t i = 0; i < numRows; ++i) {
      ids->asFlatVector<int64_t>()->set(i, i);
    }
    vx::vector_size_t idx = 0;
    for (vx::vector_size_t row = 0; row < numRows; ++row) {
      for (const auto& key : keysByRow[row]) {
        mapKeys->asFlatVector<vx::StringView>()->set(idx, vx::StringView(key));
        mapValues->asFlatVector<double>()->set(idx, row * 10.0 + idx);
        ++idx;
      }
    }

    auto mapVector = std::make_shared<vx::MapVector>(
        pool_.get(),
        vx::MAP(vx::VARCHAR(), vx::DOUBLE()),
        nullptr,
        numRows,
        vx::allocateOffsets(numRows, pool_.get()),
        vx::allocateSizes(numRows, pool_.get()),
        mapKeys,
        mapValues);
    auto* rawOffsets =
        mapVector->mutableOffsets(numRows)->asMutable<vx::vector_size_t>();
    auto* rawSizes =
        mapVector->mutableSizes(numRows)->asMutable<vx::vector_size_t>();
    vx::vector_size_t offset = 0;
    for (vx::vector_size_t i = 0; i < numRows; ++i) {
      rawOffsets[i] = offset;
      rawSizes[i] = static_cast<vx::vector_size_t>(keysByRow[i].size());
      offset += rawSizes[i];
    }

    return std::make_shared<vx::RowVector>(
        pool_.get(),
        rowType,
        nullptr,
        numRows,
        std::vector<vx::VectorPtr>{ids, mapVector});
  }

  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<facebook::velox::memory::MemoryPool> pool_;
};

// Feeds every batch's payload for `streamOffset` into the decoder, advancing
// `startRow` the way the Deserializer does across a run of batches.
void addBatches(
    BatchedStreamDecoder& decoder,
    const SerializedInput& input,
    uint32_t streamOffset) {
  uint32_t startRow = 0;
  for (const auto& batch : input.batches) {
    if (batch.hasStream(streamOffset)) {
      decoder.addBatch(
          startRow,
          batch.stream(streamOffset),
          /*legacyHeaderless=*/false,
          batch.streamEncodingUsesVarintRowCount);
    }
    startRow += batch.rowCount;
  }
}

std::vector<int32_t> readInts(BatchedStreamDecoder& decoder, uint32_t count) {
  std::vector<int32_t> output(count);
  std::vector<facebook::velox::BufferPtr> stringBuffers;
  decoder.next(count, output.data(), /*getOutputNulls=*/nullptr, stringBuffers);
  return output;
}

std::vector<int32_t> expectedInts(int32_t firstValue, int32_t count) {
  std::vector<int32_t> expected(count);
  for (int32_t i = 0; i < count; ++i) {
    expected[i] = firstValue + i;
  }
  return expected;
}

constexpr uint32_t kNoBufferPool = 0;

TEST_F(BatchedStreamDecoderTest, rejectsSelectedReads) {
  auto rowType = facebook::velox::ROW({{"c0", facebook::velox::INTEGER()}});
  auto input = serializeBatches(
      rowType, {makeIntBatch(rowType, 0, 1)}, serializerOptions());
  BatchedStreamDecoder decoder{
      input.schema->asRow().childAt(0).get(),
      /*isInMapStream=*/false,
      kNoBufferPool,
      pool_.get()};
  const std::array<uint32_t, 1> rows{0};
  std::array<int32_t, 1> output{};
  std::vector<facebook::velox::BufferPtr> stringBuffers;

  NIMBLE_ASSERT_THROW(
      decoder.read(
          rows,
          DataType::Int32,
          output.data(),
          /*getOutputNulls=*/nullptr,
          stringBuffers),
      "BatchedStreamDecoder does not support selective row decoding");
}

TEST_F(BatchedStreamDecoderTest, hybridFlatMapGroupStreamsRoundTrip) {
  SchemaBuilder schemaBuilder;
  auto hybridMap =
      schemaBuilder.createHybridFlatMapTypeBuilder(ScalarKind::String);
  hybridMap->addGroup(
      0, {"A", "B"}, schemaBuilder.createScalarTypeBuilder(ScalarKind::String));
  hybridMap->addGroup(
      HybridFlatMap::kDefaultGroupId,
      {},
      schemaBuilder.createScalarTypeBuilder(ScalarKind::String));

  SchemaSerializer schemaSerializer;
  const auto schema = SchemaDeserializer::deserialize(
      schemaSerializer.serialize(schemaBuilder));
  const auto& group = schema->asHybridFlatMap().groupAt(0);

  const auto options = serializerOptions();
  const auto encodeStrings = [&](const std::vector<std::string>& values) {
    std::vector<std::string_view> views;
    views.reserve(values.size());
    for (const auto& value : values) {
      views.push_back(value);
    }
    const std::string_view raw{
        reinterpret_cast<const char*>(views.data()),
        views.size() * sizeof(std::string_view)};
    Buffer buffer{*pool_};
    return std::string{serde::detail::encodeScalar<std::string>(
        options,
        ScalarKind::String,
        raw,
        *pool_,
        buffer,
        /*encodingLayout=*/nullptr,
        options.encodingOptions)};
  };
  const auto decodeStrings = [&](const StreamDescriptor& descriptor,
                                 std::string_view encoded,
                                 size_t count) {
    ScalarType type{descriptor};
    BatchedStreamDecoder decoder{
        &type, /*isInMapStream=*/false, kNoBufferPool, pool_.get()};
    decoder.addBatch(
        /*startRow=*/0,
        encoded,
        /*legacyHeaderless=*/false,
        options.encodingOptions.useVarintRowCount);
    std::vector<std::string_view> decodedViews(count);
    std::vector<facebook::velox::BufferPtr> stringBuffers;
    EXPECT_EQ(
        decoder.next(
            static_cast<uint32_t>(count),
            decodedViews.data(),
            /*getOutputNulls=*/nullptr,
            stringBuffers),
        count);
    std::vector<std::string> decoded;
    decoded.reserve(count);
    for (const auto value : decodedViews) {
      decoded.emplace_back(value);
    }
    return decoded;
  };

  const std::vector<std::string> expectedKeys{"A", "B"};
  const auto encodedKeys = encodeStrings(expectedKeys);
  const auto decodedKeys =
      decodeStrings(group.keyDescriptor, encodedKeys, expectedKeys.size());
  EXPECT_EQ(decodedKeys, expectedKeys);

  const std::array<bool, 6> expectedInMap{
      true, false, false, true, false, true};
  const std::string_view rawInMap{
      reinterpret_cast<const char*>(expectedInMap.data()),
      expectedInMap.size() * sizeof(bool)};
  Buffer inMapBuffer{*pool_};
  const std::string encodedInMap{serde::detail::encodeScalar<std::string>(
      options,
      ScalarKind::Bool,
      rawInMap,
      *pool_,
      inMapBuffer,
      /*encodingLayout=*/nullptr,
      options.encodingOptions)};
  ScalarType inMapType{group.inMapDescriptor};
  BatchedStreamDecoder inMapDecoder{
      &inMapType, /*isInMapStream=*/true, kNoBufferPool, pool_.get()};
  inMapDecoder.addBatch(
      /*startRow=*/0,
      encodedInMap,
      /*legacyHeaderless=*/false,
      options.encodingOptions.useVarintRowCount);
  std::vector<uint8_t> decodedInMap(expectedInMap.size());
  std::vector<facebook::velox::BufferPtr> stringBuffers;
  EXPECT_EQ(
      inMapDecoder.next(
          decodedInMap.size(),
          decodedInMap.data(),
          /*getOutputNulls=*/nullptr,
          stringBuffers),
      expectedInMap.size());
  EXPECT_EQ(
      std::vector<bool>(decodedInMap.begin(), decodedInMap.end()),
      std::vector<bool>(expectedInMap.begin(), expectedInMap.end()));

  const std::vector<std::string> expectedValues{"x", "u", "v"};
  const auto encodedValues = encodeStrings(expectedValues);
  const auto decodedValues = decodeStrings(
      group.valueType->asScalar().scalarDescriptor(),
      encodedValues,
      expectedValues.size());
  EXPECT_EQ(decodedValues, expectedValues);

  std::vector<std::vector<std::pair<std::string, std::string>>> rows(3);
  size_t valueIndex{0};
  for (size_t keyIndex = 0; keyIndex < decodedKeys.size(); ++keyIndex) {
    for (size_t row = 0; row < rows.size(); ++row) {
      if (decodedInMap[keyIndex * rows.size() + row]) {
        rows[row].emplace_back(
            decodedKeys[keyIndex], decodedValues[valueIndex]);
        ++valueIndex;
      }
    }
  }
  const std::vector<std::vector<std::pair<std::string, std::string>>>
      expectedRows{
          {{"A", "x"}, {"B", "u"}},
          {},
          {{"B", "v"}},
      };
  EXPECT_EQ(rows, expectedRows);

  const auto& defaultGroup = schema->asHybridFlatMap().defaultGroup();
  const std::vector<std::string> expectedDefaultKeys{"new:X", "new:Y"};
  const auto encodedDefaultKeys = encodeStrings(expectedDefaultKeys);
  EXPECT_EQ(
      decodeStrings(
          defaultGroup.keyDescriptor,
          encodedDefaultKeys,
          expectedDefaultKeys.size()),
      expectedDefaultKeys);
}

TEST_F(BatchedStreamDecoderTest, hybridFlatMapOmittedNullsAreAllNonNull) {
  SchemaBuilder schemaBuilder;
  auto hybridMap =
      schemaBuilder.createHybridFlatMapTypeBuilder(ScalarKind::String);
  hybridMap->addGroup(
      0,
      {"configured"},
      schemaBuilder.createScalarTypeBuilder(ScalarKind::Int64));
  hybridMap->addGroup(
      HybridFlatMap::kDefaultGroupId,
      {},
      schemaBuilder.createScalarTypeBuilder(ScalarKind::Int64));
  const auto schema = SchemaReader::getSchema(schemaBuilder.schemaNodes());

  BatchedStreamDecoder decoder{
      schema.get(), /*isInMapStream=*/false, kNoBufferPool, pool_.get()};
  decoder.skip(2);

  std::vector<uint8_t> output(4, 0);
  std::vector<facebook::velox::BufferPtr> stringBuffers;
  EXPECT_EQ(
      decoder.next(
          output.size(),
          output.data(),
          /*getOutputNulls=*/nullptr,
          stringBuffers),
      4);
  EXPECT_EQ(output, std::vector<uint8_t>(4, 1));
}

TEST_F(BatchedStreamDecoderTest, nextStitchesSegmentsFromMultipleBatches) {
  auto rowType = facebook::velox::ROW({{"c0", facebook::velox::INTEGER()}});
  auto input = serializeBatches(
      rowType,
      {makeIntBatch(rowType, 0, 4),
       makeIntBatch(rowType, 4, 3),
       makeIntBatch(rowType, 7, 5)},
      serializerOptions());

  const auto* valueType = input.schema->asRow().childAt(0).get();
  const auto valueOffset = valueType->asScalar().scalarDescriptor().offset();
  BatchedStreamDecoder decoder{
      valueType, /*isInMapStream=*/false, kNoBufferPool, pool_.get()};
  addBatches(decoder, input, valueOffset);

  EXPECT_EQ(readInts(decoder, 12), expectedInts(0, 12));
}

TEST_F(BatchedStreamDecoderTest, nextReadsStreamRowCountEncodingCombinations) {
  auto rowType = facebook::velox::ROW({{"c0", facebook::velox::INTEGER()}});
  auto schemaInput = serializeBatches(
      rowType, {makeIntBatch(rowType, 0, 1)}, serializerOptions());
  const auto* valueType = schemaInput.schema->asRow().childAt(0).get();
  const std::array<std::vector<int32_t>, 3> batches{{
      {0, 1, 2, 3},
      {4, 5, 6},
      {7, 8, 9, 10, 11},
  }};

  // Tablet batches independently choose the row-count prefix encoding. Verify
  // the decoder preserves each segment's mode while stitching mixed batches.
  constexpr std::array<std::array<bool, 3>, 8> rowCountEncodingCombinations{{
      {false, false, false},
      {false, false, true},
      {false, true, false},
      {false, true, true},
      {true, false, false},
      {true, false, true},
      {true, true, false},
      {true, true, true},
  }};
  for (const auto& useVarintRowCount : rowCountEncodingCombinations) {
    SCOPED_TRACE(
        "useVarintRowCount=[" + std::to_string(useVarintRowCount[0]) + "," +
        std::to_string(useVarintRowCount[1]) + "," +
        std::to_string(useVarintRowCount[2]) + "]");

    BatchedStreamDecoder decoder{
        valueType, /*isInMapStream=*/false, kNoBufferPool, pool_.get()};
    std::vector<std::unique_ptr<Buffer>> encodedBuffers;
    std::array<std::string, 3> encodedSegments;
    uint32_t startRow{0};
    for (size_t i = 0; i < batches.size(); ++i) {
      auto buffer = std::make_unique<Buffer>(*pool_);
      const std::string_view data{
          reinterpret_cast<const char*>(batches[i].data()),
          batches[i].size() * sizeof(int32_t)};
      encodedSegments[i] = serde::detail::encodeScalar<std::string>(
          serializerOptions(),
          ScalarKind::Int32,
          data,
          *pool_,
          *buffer,
          /*encodingLayout=*/nullptr,
          Encoding::Options{
              .useVarintRowCount = useVarintRowCount[i],
          });
      encodedBuffers.push_back(std::move(buffer));
      decoder.addBatch(
          startRow,
          encodedSegments[i],
          /*legacyHeaderless=*/false,
          useVarintRowCount[i]);
      startRow += static_cast<uint32_t>(batches[i].size());
    }

    EXPECT_EQ(readInts(decoder, 12), expectedInts(0, 12));
  }
}

TEST_F(BatchedStreamDecoderTest, nextResumesMidSegmentAcrossCalls) {
  auto rowType = facebook::velox::ROW({{"c0", facebook::velox::INTEGER()}});
  auto input = serializeBatches(
      rowType,
      {makeIntBatch(rowType, 0, 4),
       makeIntBatch(rowType, 4, 3),
       makeIntBatch(rowType, 7, 5)},
      serializerOptions());

  const auto* valueType = input.schema->asRow().childAt(0).get();
  const auto valueOffset = valueType->asScalar().scalarDescriptor().offset();
  BatchedStreamDecoder decoder{
      valueType, /*isInMapStream=*/false, kNoBufferPool, pool_.get()};
  addBatches(decoder, input, valueOffset);

  // Both reads straddle a batch boundary, so each one resumes a partially
  // consumed segment.
  EXPECT_EQ(readInts(decoder, 5), expectedInts(0, 5));
  EXPECT_EQ(readInts(decoder, 7), expectedInts(5, 7));
}

TEST_F(BatchedStreamDecoderTest, skipAdvancesAcrossSegmentBoundary) {
  auto rowType = facebook::velox::ROW({{"c0", facebook::velox::INTEGER()}});
  auto input = serializeBatches(
      rowType,
      {makeIntBatch(rowType, 0, 4),
       makeIntBatch(rowType, 4, 3),
       makeIntBatch(rowType, 7, 5)},
      serializerOptions());

  const auto* valueType = input.schema->asRow().childAt(0).get();
  const auto valueOffset = valueType->asScalar().scalarDescriptor().offset();
  BatchedStreamDecoder decoder{
      valueType, /*isInMapStream=*/false, kNoBufferPool, pool_.get()};
  addBatches(decoder, input, valueOffset);

  // Consumes all of batch 1 and part of batch 2.
  decoder.skip(6);

  EXPECT_EQ(readInts(decoder, 6), expectedInts(6, 6));
}

TEST_F(BatchedStreamDecoderTest, skipWithinSegmentThenRead) {
  auto rowType = facebook::velox::ROW({{"c0", facebook::velox::INTEGER()}});
  auto input = serializeBatches(
      rowType, {makeIntBatch(rowType, 0, 8)}, serializerOptions());

  const auto* valueType = input.schema->asRow().childAt(0).get();
  const auto valueOffset = valueType->asScalar().scalarDescriptor().offset();
  BatchedStreamDecoder decoder{
      valueType, /*isInMapStream=*/false, kNoBufferPool, pool_.get()};
  addBatches(decoder, input, valueOffset);

  decoder.skip(3);

  EXPECT_EQ(readInts(decoder, 5), expectedInts(3, 5));
}

TEST_F(BatchedStreamDecoderTest, nextStitchesNullBitmapAcrossSegments) {
  auto rowType = facebook::velox::ROW({{"c0", facebook::velox::INTEGER()}});
  // Batch 1 is all non-null, batch 2 nulls every other row. The all-non-null
  // prefix has to be back-filled as non-null once the second segment turns
  // null handling on.
  auto input = serializeBatches(
      rowType,
      {makeIntBatch(rowType, 0, 4),
       makeIntBatch(rowType, 100, 4, /*nullEvery=*/2)},
      serializerOptions());

  const auto* valueType = input.schema->asRow().childAt(0).get();
  const auto valueOffset = valueType->asScalar().scalarDescriptor().offset();
  BatchedStreamDecoder decoder{
      valueType, /*isInMapStream=*/false, kNoBufferPool, pool_.get()};
  addBatches(decoder, input, valueOffset);

  constexpr uint32_t kRows = 8;
  std::vector<int32_t> output(kRows);
  std::vector<uint64_t> nulls(facebook::velox::bits::nwords(kRows), 0);
  std::vector<facebook::velox::BufferPtr> stringBuffers;
  const auto nonNullCount = decoder.next(
      kRows,
      output.data(),
      [&]() -> void* { return nulls.data(); },
      stringBuffers);

  // Rows 4 and 6 are the nulls contributed by batch 2.
  const std::vector<bool> expectedNonNull = {
      true, true, true, true, false, true, false, true};
  std::vector<bool> actualNonNull(kRows);
  for (uint32_t i = 0; i < kRows; ++i) {
    actualNonNull[i] = facebook::velox::bits::isBitSet(nulls.data(), i);
  }
  EXPECT_EQ(actualNonNull, expectedNonNull);
  EXPECT_EQ(nonNullCount, 6);
}

TEST_F(BatchedStreamDecoderTest, denseReadReconstructsOmittedRowNullStream) {
  auto rowType = facebook::velox::ROW({{"c0", facebook::velox::INTEGER()}});
  auto input = serializeBatches(
      rowType, {makeIntBatch(rowType, 0, 6)}, serializerOptions());

  // The writer omits an all-non-null Row nulls stream entirely, so the
  // decoder gets no segments and must reconstruct all-true.
  const auto rowNullsOffset = input.schema->asRow().nullsDescriptor().offset();
  ASSERT_FALSE(input.batches[0].hasStream(rowNullsOffset));

  BatchedStreamDecoder decoder{
      input.schema.get(),
      /*isInMapStream=*/false,
      kNoBufferPool,
      pool_.get()};

  constexpr uint32_t kRows = 6;
  std::vector<bool> expected(kRows, true);
  std::vector<uint8_t> output(kRows, 0);
  std::vector<facebook::velox::BufferPtr> stringBuffers;
  const auto nonNullCount = decoder.next(
      kRows, output.data(), /*getOutputNulls=*/nullptr, stringBuffers);

  std::vector<bool> actual(output.begin(), output.end());
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(nonNullCount, kRows);
}

TEST_F(BatchedStreamDecoderTest, skipOnOmittedRowNullStreamAdvancesCursor) {
  auto rowType = facebook::velox::ROW({{"c0", facebook::velox::INTEGER()}});
  auto input = serializeBatches(
      rowType, {makeIntBatch(rowType, 0, 6)}, serializerOptions());

  BatchedStreamDecoder decoder{
      input.schema.get(),
      /*isInMapStream=*/false,
      kNoBufferPool,
      pool_.get()};

  // With no segments queued there is nothing to advance; skip must not throw
  // and the following read still reconstructs all-true.
  decoder.skip(2);

  constexpr uint32_t kRows = 4;
  std::vector<uint8_t> output(kRows, 0);
  std::vector<facebook::velox::BufferPtr> stringBuffers;
  const auto nonNullCount = decoder.next(
      kRows, output.data(), /*getOutputNulls=*/nullptr, stringBuffers);

  std::vector<bool> actual(output.begin(), output.end());
  EXPECT_EQ(actual, std::vector<bool>(kRows, true));
  EXPECT_EQ(nonNullCount, kRows);
}

TEST_F(BatchedStreamDecoderTest, clearRestoresDecoderForReuse) {
  auto rowType = facebook::velox::ROW({{"c0", facebook::velox::INTEGER()}});
  auto input = serializeBatches(
      rowType,
      {makeIntBatch(rowType, 0, 4), makeIntBatch(rowType, 4, 4)},
      serializerOptions());

  const auto* valueType = input.schema->asRow().childAt(0).get();
  const auto valueOffset = valueType->asScalar().scalarDescriptor().offset();
  BatchedStreamDecoder decoder{
      valueType, /*isInMapStream=*/false, kNoBufferPool, pool_.get()};

  addBatches(decoder, input, valueOffset);
  EXPECT_EQ(readInts(decoder, 8), expectedInts(0, 8));

  decoder.clear();
  addBatches(decoder, input, valueOffset);

  // Row cursor and segment cursor are both back at zero.
  EXPECT_EQ(readInts(decoder, 8), expectedInts(0, 8));
}

TEST_F(BatchedStreamDecoderTest, nextWithZeroCountDecodesNothing) {
  auto rowType = facebook::velox::ROW({{"c0", facebook::velox::INTEGER()}});
  auto input = serializeBatches(
      rowType, {makeIntBatch(rowType, 0, 4)}, serializerOptions());

  const auto* valueType = input.schema->asRow().childAt(0).get();
  const auto valueOffset = valueType->asScalar().scalarDescriptor().offset();
  BatchedStreamDecoder decoder{
      valueType, /*isInMapStream=*/false, kNoBufferPool, pool_.get()};
  addBatches(decoder, input, valueOffset);

  std::vector<facebook::velox::BufferPtr> stringBuffers;
  EXPECT_EQ(
      decoder.next(0, nullptr, /*getOutputNulls=*/nullptr, stringBuffers), 0);

  // The segment cursor did not move.
  EXPECT_EQ(readInts(decoder, 4), expectedInts(0, 4));
}

TEST_F(BatchedStreamDecoderTest, addBatchRejectsEmptySegment) {
  auto rowType = facebook::velox::ROW({{"c0", facebook::velox::INTEGER()}});
  auto input = serializeBatches(
      rowType, {makeIntBatch(rowType, 0, 4)}, serializerOptions());

  BatchedStreamDecoder decoder{
      input.schema->asRow().childAt(0).get(),
      /*isInMapStream=*/false,
      kNoBufferPool,
      pool_.get()};

  NIMBLE_ASSERT_THROW(
      decoder.addBatch(
          0,
          std::string_view{},
          /*legacyHeaderless=*/false,
          /*streamEncodingUsesVarintRowCount=*/true),
      "Physical stream segment must be non-empty");
}

// --- FlatMap in-map streams ---

class BatchedStreamDecoderInMapTest : public BatchedStreamDecoderTest {
 protected:
  static SerializerOptions flatMapOptions() {
    auto options = serializerOptions();
    options.flatMapColumns = {{"flat_map", {}}};
    return options;
  }

  static facebook::velox::TypePtr flatMapRowType() {
    return facebook::velox::ROW({
        {"id", facebook::velox::BIGINT()},
        {"flat_map",
         facebook::velox::MAP(
             facebook::velox::VARCHAR(), facebook::velox::DOUBLE())},
    });
  }

  // In-map decoders are constructed with the PARENT FlatMap type, not the
  // child value type — see Deserializer::createDeserializersForType.
  static const Type* flatMapType(const SerializedInput& input) {
    return input.schema->asRow().childAt(1).get();
  }

  static std::vector<bool> readInMap(
      BatchedStreamDecoder& decoder,
      uint32_t count) {
    std::vector<uint8_t> output(count, 0xFF);
    std::vector<facebook::velox::BufferPtr> stringBuffers;
    decoder.next(
        count, output.data(), /*getOutputNulls=*/nullptr, stringBuffers);
    return std::vector<bool>(output.begin(), output.end());
  }
};

TEST_F(BatchedStreamDecoderInMapTest, inMapReadFillsGapForAllPresentBatches) {
  auto rowType = flatMapRowType();
  auto input = serializeBatches(
      rowType,
      {makeFlatMapBatch(
           rowType, std::vector<std::vector<std::string>>(4, {"a"})),
       makeFlatMapBatch(
           rowType, std::vector<std::vector<std::string>>(3, {"a"}))},
      flatMapOptions());

  const auto& flatMap = flatMapType(input)->asFlatMap();
  const auto inMapOffset = flatMap.inMapDescriptorAt(0).offset();
  // Key "a" is in every row of both batches, so the writer omits the in-map
  // stream and the reader has to reconstruct it from presence ranges alone.
  ASSERT_FALSE(input.batches[0].hasStream(inMapOffset));
  ASSERT_FALSE(input.batches[1].hasStream(inMapOffset));

  BatchedStreamDecoder decoder{
      flatMapType(input), /*isInMapStream=*/true, kNoBufferPool, pool_.get()};
  decoder.addPresentInMapBatch(/*startRow=*/0, /*rowCount=*/4);
  decoder.addPresentInMapBatch(/*startRow=*/4, /*rowCount=*/3);

  EXPECT_EQ(readInMap(decoder, 7), std::vector<bool>(7, true));
}

TEST_F(BatchedStreamDecoderInMapTest, inMapReadLeavesUnrecordedRowsAbsent) {
  auto rowType = flatMapRowType();
  auto input = serializeBatches(
      rowType,
      {makeFlatMapBatch(
          rowType, std::vector<std::vector<std::string>>(4, {"a"}))},
      flatMapOptions());

  BatchedStreamDecoder decoder{
      flatMapType(input), /*isInMapStream=*/true, kNoBufferPool, pool_.get()};
  // Only rows [1, 3) are recorded as present; everything else defaults
  // absent.
  decoder.addPresentInMapBatch(/*startRow=*/1, /*rowCount=*/2);

  const std::vector<bool> expected = {false, true, true, false, false};
  EXPECT_EQ(readInMap(decoder, 5), expected);
}

TEST_F(
    BatchedStreamDecoderInMapTest,
    inMapReadMixesPhysicalAndPresentSegments) {
  auto rowType = flatMapRowType();
  // Batch 1: key "b" only on even rows, so its in-map stream is written.
  // Batch 2: key "b" everywhere, so its in-map stream is omitted.
  std::vector<std::vector<std::string>> mixed(4);
  for (size_t i = 0; i < mixed.size(); ++i) {
    mixed[i].emplace_back("a");
    if (i % 2 == 0) {
      mixed[i].emplace_back("b");
    }
  }
  auto input = serializeBatches(
      rowType,
      {makeFlatMapBatch(rowType, mixed),
       makeFlatMapBatch(
           rowType, std::vector<std::vector<std::string>>(3, {"a", "b"}))},
      flatMapOptions());

  const auto& flatMap = flatMapType(input)->asFlatMap();
  ASSERT_EQ(flatMap.childrenCount(), 2);
  const auto inMapOffsetB = flatMap.inMapDescriptorAt(1).offset();
  ASSERT_TRUE(input.batches[0].hasStream(inMapOffsetB));
  ASSERT_FALSE(input.batches[1].hasStream(inMapOffsetB));

  BatchedStreamDecoder decoder{
      flatMapType(input), /*isInMapStream=*/true, kNoBufferPool, pool_.get()};
  decoder.addBatch(
      /*startRow=*/0,
      input.batches[0].stream(inMapOffsetB),
      /*legacyHeaderless=*/false,
      input.batches[0].streamEncodingUsesVarintRowCount);
  decoder.addPresentInMapBatch(/*startRow=*/4, /*rowCount=*/3);

  const std::vector<bool> expected = {
      true, false, true, false, true, true, true};
  EXPECT_EQ(readInMap(decoder, 7), expected);
}

TEST_F(
    BatchedStreamDecoderInMapTest,
    nullBarrierPresentSegmentCoversWholeRead) {
  auto rowType = flatMapRowType();
  auto input = serializeBatches(
      rowType,
      {makeFlatMapBatch(
          rowType, std::vector<std::vector<std::string>>(4, {"a"}))},
      flatMapOptions());

  BatchedStreamDecoder decoder{
      flatMapType(input), /*isInMapStream=*/true, kNoBufferPool, pool_.get()};
  // The parameterless overload stores a sentinel end row because the read's
  // extent is not known until decode time.
  decoder.addPresentInMapBatch();

  EXPECT_EQ(readInMap(decoder, 5), std::vector<bool>(5, true));
}

TEST_F(BatchedStreamDecoderInMapTest, skipInMapAdvancesPresenceSegments) {
  auto rowType = flatMapRowType();
  auto input = serializeBatches(
      rowType,
      {makeFlatMapBatch(
          rowType, std::vector<std::vector<std::string>>(4, {"a"}))},
      flatMapOptions());

  BatchedStreamDecoder decoder{
      flatMapType(input), /*isInMapStream=*/true, kNoBufferPool, pool_.get()};
  decoder.addPresentInMapBatch(/*startRow=*/0, /*rowCount=*/3);
  decoder.addPresentInMapBatch(/*startRow=*/5, /*rowCount=*/2);

  // Skips past the first presence segment and the absent rows behind it.
  decoder.skip(5);

  EXPECT_EQ(readInMap(decoder, 2), std::vector<bool>(2, true));
}

TEST_F(
    BatchedStreamDecoderInMapTest,
    addPresentInMapBatchRejectsNonInMapStream) {
  auto rowType = flatMapRowType();
  auto input = serializeBatches(
      rowType,
      {makeFlatMapBatch(
          rowType, std::vector<std::vector<std::string>>(2, {"a"}))},
      flatMapOptions());

  BatchedStreamDecoder decoder{
      flatMapType(input),
      /*isInMapStream=*/false,
      kNoBufferPool,
      pool_.get()};

  NIMBLE_ASSERT_THROW(
      decoder.addPresentInMapBatch(0, 2), "Expected FlatMap in-map stream");
}

} // namespace
