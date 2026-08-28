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
#include <zstd.h>

#include <cstdint>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <fmt/core.h>

#include "folly/container/F14Map.h"

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/Projector.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/serializer/StreamDataWriter.h"
#include "velox/dwio/nimble/serializer/StreamSlicer.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;

namespace facebook::nimble::serde {
namespace {

std::string iobufToString(const folly::IOBuf& buf) {
  std::string result;
  result.reserve(buf.computeChainDataLength());
  for (const auto range : buf) {
    result.append(reinterpret_cast<const char*>(range.data()), range.size());
  }
  return result;
}

std::string makeUncompressedChunk(std::string_view data) {
  std::string output(kChunkHeaderSize, '\0');
  auto* pos = output.data();
  writeChunkHeader(
      static_cast<uint32_t>(data.size()), CompressionType::Uncompressed, pos);
  output.append(data);
  return output;
}

std::string makeZstdChunk(std::string_view data) {
  const auto maxCompressedSize = ZSTD_compressBound(data.size());
  std::string compressed(sizeof(uint32_t) + maxCompressedSize, '\0');
  auto* compressedData = compressed.data();
  encoding::writeUint32(data.size(), compressedData);
  const auto compressedSize = ZSTD_compress(
      compressedData, maxCompressedSize, data.data(), data.size(), 1);
  NIMBLE_CHECK(!ZSTD_isError(compressedSize));
  compressed.resize(sizeof(uint32_t) + compressedSize);

  std::string output(kChunkHeaderSize, '\0');
  auto* pos = output.data();
  writeChunkHeader(
      static_cast<uint32_t>(compressed.size()), CompressionType::Zstd, pos);
  output.append(compressed);
  return output;
}

void writeTabletTrailer(
    const std::vector<uint32_t>& streamSizes,
    std::string& output) {
  std::vector<uint32_t> streamIds;
  std::vector<uint32_t> streamSizeIndices;
  std::vector<uint32_t> uniqueStreamSizes;
  for (size_t i = 0; i < streamSizes.size(); ++i) {
    if (streamSizes[i] == 0) {
      continue;
    }
    streamIds.emplace_back(static_cast<uint32_t>(i));
    streamSizeIndices.emplace_back(
        static_cast<uint32_t>(uniqueStreamSizes.size()));
    uniqueStreamSizes.emplace_back(streamSizes[i]);
  }
  detail::writeTrailer(
      streamIds,
      streamSizeIndices,
      uniqueStreamSizes,
      EncodingType::Trivial,
      EncodingType::Trivial,
      EncodingType::Trivial,
      output);
}

enum class SlicerPayloadKind {
  kSerialization,
  kProjection,
  kTabletUncompressed,
  kTabletZstd,
};

std::string toString(SlicerPayloadKind kind) {
  switch (kind) {
    case SlicerPayloadKind::kSerialization:
      return "Serialization";
    case SlicerPayloadKind::kProjection:
      return "Projection";
    case SlicerPayloadKind::kTabletUncompressed:
      return "TabletUncompressed";
    case SlicerPayloadKind::kTabletZstd:
      return "TabletZstd";
  }
}

struct SlicerPayload {
  SlicerPayloadKind kind;
  std::string data;
  std::shared_ptr<const nimble::Type> schema;
};

enum class RawStreamKind {
  kSerialization,
  kProjection,
  kTablet,
  kTabletUncompressedChunk,
  kTabletZstdChunk,
};

SerializationVersion streamVersion(RawStreamKind kind) {
  switch (kind) {
    case RawStreamKind::kSerialization:
      return SerializationVersion::kSerialization;
    case RawStreamKind::kProjection:
      return SerializationVersion::kProjection;
    case RawStreamKind::kTablet:
    case RawStreamKind::kTabletUncompressedChunk:
    case RawStreamKind::kTabletZstdChunk:
      return SerializationVersion::kTablet;
  }
}

std::string toString(RawStreamKind kind) {
  switch (kind) {
    case RawStreamKind::kSerialization:
      return "Serialization";
    case RawStreamKind::kProjection:
      return "Projection";
    case RawStreamKind::kTablet:
      return "Tablet";
    case RawStreamKind::kTabletUncompressedChunk:
      return "TabletUncompressedChunk";
    case RawStreamKind::kTabletZstdChunk:
      return "TabletZstdChunk";
  }
}

bool streamHasChunkHeader(RawStreamKind kind) {
  return kind == RawStreamKind::kTabletUncompressedChunk ||
      kind == RawStreamKind::kTabletZstdChunk;
}

class StreamSlicerTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ = memory::memoryManager()->addRootPool("stream_slicer_root");
    pool_ = memory::memoryManager()->addLeafPool("stream_slicer_leaf");
  }

  template <typename T>
  FlatVectorPtr<T> makeFlatVector(const std::vector<T>& values) {
    auto vector = BaseVector::create<FlatVector<T>>(
        CppToType<T>::create(), values.size(), pool_.get());
    for (size_t i = 0; i < values.size(); ++i) {
      vector->set(i, values[i]);
    }
    return vector;
  }

  RowVectorPtr makeRowVector(
      const std::vector<std::string>& names,
      const std::vector<VectorPtr>& children) {
    std::vector<TypePtr> types;
    types.reserve(children.size());
    for (const auto& child : children) {
      types.push_back(child->type());
    }
    return std::make_shared<RowVector>(
        pool_.get(),
        ROW(std::vector<std::string>(names), std::move(types)),
        nullptr,
        children.empty() ? 0 : children.front()->size(),
        children);
  }

  ArrayVectorPtr makeArrayVector(
      const std::vector<vector_size_t>& sizes,
      const VectorPtr& elements) {
    auto offsets =
        AlignedBuffer::allocate<vector_size_t>(sizes.size(), pool_.get());
    auto sizesBuffer =
        AlignedBuffer::allocate<vector_size_t>(sizes.size(), pool_.get());
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizesBuffer->asMutable<vector_size_t>();
    vector_size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
      rawOffsets[i] = offset;
      rawSizes[i] = sizes[i];
      offset += sizes[i];
    }
    return std::make_shared<ArrayVector>(
        pool_.get(),
        ARRAY(elements->type()),
        nullptr,
        sizes.size(),
        offsets,
        sizesBuffer,
        elements);
  }

  MapVectorPtr makeMapVector(
      const std::vector<vector_size_t>& sizes,
      const VectorPtr& keys,
      const VectorPtr& values) {
    auto offsets =
        AlignedBuffer::allocate<vector_size_t>(sizes.size(), pool_.get());
    auto sizesBuffer =
        AlignedBuffer::allocate<vector_size_t>(sizes.size(), pool_.get());
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizesBuffer->asMutable<vector_size_t>();
    vector_size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
      rawOffsets[i] = offset;
      rawSizes[i] = sizes[i];
      offset += sizes[i];
    }
    return std::make_shared<MapVector>(
        pool_.get(),
        MAP(keys->type(), values->type()),
        nullptr,
        sizes.size(),
        offsets,
        sizesBuffer,
        keys,
        values);
  }

  std::pair<std::string, std::shared_ptr<const nimble::Type>> serialize(
      const VectorPtr& vector,
      const TypePtr& type,
      std::optional<EncodingType> encodingType = std::nullopt,
      std::optional<CompressionOptions> compressionOptions = std::nullopt,
      folly::F14FastMap<std::string, std::set<std::string>> flatMapColumns =
          {}) {
    auto options = SerializerOptions{
        .version = SerializationVersion::kSerialization,
        .flatMapColumns = std::move(flatMapColumns),
    };
    if (encodingType.has_value()) {
      auto factory = std::make_shared<ManualEncodingSelectionPolicyFactory>(
          std::vector<std::pair<EncodingType, float>>{
              {encodingType.value(), 1.0}},
          std::move(compressionOptions));
      options.encodingSelectionPolicyCreator = [factory](DataType dataType) {
        return factory->createPolicy(dataType);
      };
    }
    Serializer serializer{std::move(options), type, pool_.get()};
    auto data =
        serializer.serialize(vector, OrderedRanges::of(0, vector->size()));
    auto schema =
        SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
    return {std::string(data), std::move(schema)};
  }

  std::string encodeTabletIntStream(const std::vector<int32_t>& values) {
    return encodeIntStream(values, /*useVarintRowCount=*/false);
  }

  std::string encodeIntStream(
      const std::vector<int32_t>& values,
      bool useVarintRowCount) {
    Buffer buffer{*pool_};
    ManualEncodingSelectionPolicyFactory factory{
        {{EncodingType::Trivial, 1.0}}, /*compressionOptions=*/std::nullopt};
    auto policyBase = factory.createPolicy(DataType::Int32);
    auto policy = std::unique_ptr<EncodingSelectionPolicy<int32_t>>(
        static_cast<EncodingSelectionPolicy<int32_t>*>(policyBase.release()));
    const auto encoded = EncodingFactory::encode<int32_t>(
        std::move(policy),
        std::span<const int32_t>{values.data(), values.size()},
        buffer,
        Encoding::Options{.useVarintRowCount = useVarintRowCount});
    return std::string(encoded);
  }

  std::string makeTabletPayload(
      uint32_t rowCount,
      uint32_t streamId,
      std::string_view encodedStream,
      bool compressStream = false) {
    auto header = createTabletChunkHeader({
        .rowCount = rowCount,
        .streamHasChunkHeader = true,
        .rowRange = RowRange{0, rowCount},
    });
    std::string output(
        reinterpret_cast<const char*>(header.data()), header.length());
    const auto chunkedStream = compressStream
        ? makeZstdChunk(encodedStream)
        : makeUncompressedChunk(encodedStream);
    output.append(chunkedStream);
    std::vector<uint32_t> streamSizes(streamId + 1, 0);
    streamSizes[streamId] = static_cast<uint32_t>(chunkedStream.size());
    writeTabletTrailer(streamSizes, output);
    return output;
  }

  std::pair<std::string, std::shared_ptr<const nimble::Type>> projectAllColumns(
      std::string_view serialized,
      std::shared_ptr<const nimble::Type> schema,
      const std::vector<std::string>& columnNames) {
    std::vector<Subfield> subfields;
    subfields.reserve(columnNames.size());
    for (const auto& name : columnNames) {
      subfields.emplace_back(name);
    }
    Projector projector{schema, subfields, pool_.get(), Projector::Options{}};
    return {
        iobufToString(projector.project(serialized)),
        projector.projectedSchema(),
    };
  }

  SlicerPayload makePayload(
      const std::string& serialized,
      const std::shared_ptr<const nimble::Type>& schema,
      const std::vector<std::string>& columnNames,
      SlicerPayloadKind kind) {
    switch (kind) {
      case SlicerPayloadKind::kSerialization:
        return {
            .kind = kind,
            .data = serialized,
            .schema = schema,
        };
      case SlicerPayloadKind::kProjection: {
        auto [projected, projectedSchema] =
            projectAllColumns(serialized, schema, columnNames);
        return {
            .kind = kind,
            .data = std::move(projected),
            .schema = std::move(projectedSchema),
        };
      }
      case SlicerPayloadKind::kTabletUncompressed:
      case SlicerPayloadKind::kTabletZstd:
        NIMBLE_FAIL("Unsupported payload kind: {}", toString(kind));
    }
  }

  VectorPtr deserialize(
      std::string_view data,
      const std::shared_ptr<const nimble::Type>& schema) {
    Deserializer deserializer{schema, pool_.get(), {.hasHeader = true}};
    VectorPtr output;
    deserializer.deserialize(data, output);
    return output;
  }

  template <typename T>
  std::vector<T> readColumn(const VectorPtr& vector) {
    auto* row = vector->as<RowVector>();
    NIMBLE_CHECK_NOT_NULL(row);
    auto* values = row->childAt(0)->as<FlatVector<T>>();
    NIMBLE_CHECK_NOT_NULL(values);
    std::vector<T> result;
    result.reserve(values->size());
    for (vector_size_t i = 0; i < values->size(); ++i) {
      result.push_back(values->valueAt(i));
    }
    return result;
  }

  void expectIntColumn(
      const VectorPtr& vector,
      const std::vector<int32_t>& values,
      uint32_t offset,
      uint32_t length) {
    const std::vector<int32_t> expected{
        values.begin() + offset, values.begin() + offset + length};
    EXPECT_EQ(readColumn<int32_t>(vector), expected);
  }

  static size_t inputStreamBytes(
      const std::vector<std::string_view>& inputStreams) {
    size_t bytes{0};
    for (const auto stream : inputStreams) {
      bytes += stream.size();
    }
    return bytes;
  }

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> pool_;
};

class StreamSlicerPayloadVersionTest
    : public StreamSlicerTest,
      public ::testing::WithParamInterface<SlicerPayloadKind> {};

class StreamSlicerPayloadApiTest
    : public StreamSlicerTest,
      public ::testing::WithParamInterface<SlicerPayloadKind> {};

class StreamSlicerRawStreamApiTest
    : public StreamSlicerTest,
      public ::testing::WithParamInterface<bool> {};

TEST_P(StreamSlicerPayloadVersionTest, slicesScalarPayload) {
  const auto payloadKind = GetParam();
  const bool isTabletPayload =
      payloadKind == SlicerPayloadKind::kTabletUncompressed ||
      payloadKind == SlicerPayloadKind::kTabletZstd;
  auto type = ROW({{"id", INTEGER()}});
  const std::vector<int32_t> values{10, 20, 30, 40, 50};
  auto input = makeRowVector({"id"}, {makeFlatVector<int32_t>(values)});
  auto [serialized, schema] = serialize(input, type);

  std::string payload;
  std::shared_ptr<const nimble::Type> sliceSchema;
  switch (payloadKind) {
    case SlicerPayloadKind::kSerialization:
      payload = serialized;
      sliceSchema = schema;
      break;
    case SlicerPayloadKind::kProjection:
      std::tie(payload, sliceSchema) =
          projectAllColumns(serialized, schema, {"id"});
      break;
    case SlicerPayloadKind::kTabletUncompressed:
    case SlicerPayloadKind::kTabletZstd: {
      const auto streamId =
          schema->asRow().childAt(0)->asScalar().scalarDescriptor().offset();
      payload = makeTabletPayload(
          static_cast<uint32_t>(values.size()),
          streamId,
          encodeTabletIntStream(values),
          /*compressStream=*/payloadKind == SlicerPayloadKind::kTabletZstd);
      sliceSchema = schema;
      break;
    }
  }

  StreamSlicer slicer{sliceSchema, pool_.get(), StreamSlicer::Options{}};
  if (isTabletPayload) {
    NIMBLE_ASSERT_THROW(
        slicer.slice(payload, /*offset=*/1, /*length=*/3),
        "Unsupported StreamSlicer input version");
    return;
  }

  auto sliced =
      iobufToString(slicer.slice(payload, /*offset=*/1, /*length=*/3));
  const char* pos = sliced.data();
  const auto header =
      readSerializationHeader(pos, sliced.data() + sliced.size(), true);
  EXPECT_EQ(header.version, SerializationVersion::kProjection);
  EXPECT_EQ(header.rowCount, 3);
  EXPECT_TRUE(header.flags.streamEncodingUsesVarintRowCount);
  EXPECT_FALSE(header.rowRange.has_value());

  auto output = deserialize(sliced, sliceSchema);
  const std::vector<int32_t> expected{values.begin() + 1, values.begin() + 4};
  EXPECT_EQ(readColumn<int32_t>(output), expected);
}

INSTANTIATE_TEST_SUITE_P(
    StreamSlicerTest,
    StreamSlicerPayloadVersionTest,
    ::testing::Values(
        SlicerPayloadKind::kSerialization,
        SlicerPayloadKind::kProjection,
        SlicerPayloadKind::kTabletUncompressed,
        SlicerPayloadKind::kTabletZstd),
    [](const auto& info) { return toString(info.param); });

TEST_P(StreamSlicerPayloadApiTest, fuzzesRandomSchemasAndNullableData) {
  constexpr uint32_t kSeed{12345};
  constexpr uint32_t kIterations{32};
  constexpr uint32_t kMinSliceRows{8};
  const auto payloadKind = GetParam();
  const std::vector<TypePtr> scalarTypes{
      BOOLEAN(), INTEGER(), BIGINT(), REAL(), DOUBLE(), VARCHAR()};
  VectorFuzzer fuzzer(
      {
          .vectorSize = 32,
          .nullRatio = 0.2,
          .useRandomNullPattern = true,
          .containerHasNulls = true,
          .stringLength = 20,
          .stringVariableLength = true,
          .containerLength = 5,
          .containerVariableLength = true,
          .complexElementsMaxSize = 1'000,
          .allowSlice = false,
          .allowConstantVector = false,
          .allowDictionaryVector = false,
      },
      pool_.get(),
      kSeed);
  std::mt19937 rng{kSeed};

  for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
    const auto randomRowType = fuzzer.randRowType(scalarTypes, /*maxDepth=*/3);
    auto rowNames = randomRowType->names();
    rowNames.emplace_back("flat_map");
    std::vector<TypePtr> rowChildren;
    std::vector<VectorPtr> inputChildren;
    rowChildren.reserve(randomRowType->size() + 1);
    inputChildren.reserve(randomRowType->size() + 1);
    for (size_t i = 0; i < randomRowType->size(); ++i) {
      const auto& childType = randomRowType->childAt(i);
      rowChildren.emplace_back(childType);
      inputChildren.emplace_back(fuzzer.fuzz(childType));
    }
    const auto flatMapValueType = fuzzer.randType(scalarTypes, /*maxDepth=*/2);
    rowChildren.emplace_back(MAP(INTEGER(), flatMapValueType));
    inputChildren.emplace_back(
        fuzzer.fuzzFlatMap(INTEGER(), flatMapValueType, /*size=*/32));
    const auto rowType = ROW(std::move(rowNames), std::move(rowChildren));
    auto input = std::make_shared<RowVector>(
        pool_.get(),
        rowType,
        nullptr,
        /*size=*/32,
        std::move(inputChildren));
    const auto rowCount = static_cast<uint32_t>(input->size());
    const uint32_t offset = rng() % (rowCount - kMinSliceRows + 1);
    const uint32_t length =
        kMinSliceRows + (rng() % (rowCount - offset - kMinSliceRows + 1));
    input->childAt(0)->setNull(offset, true);
    SCOPED_TRACE(
        fmt::format(
            "seed={} iteration={} type={} offset={} length={}",
            kSeed,
            iteration,
            rowType->toString(),
            offset,
            length));

    auto [serialized, schema] = serialize(
        input,
        rowType,
        /*encodingType=*/std::nullopt,
        /*compressionOptions=*/std::nullopt,
        /*flatMapColumns=*/{{"flat_map", {}}});
    auto projectionSubfields = randomRowType->names();
    const auto& flatMap =
        schema->asRow().childAt(randomRowType->size())->asFlatMap();
    ASSERT_GT(flatMap.childrenCount(), 0);
    for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
      projectionSubfields.emplace_back(
          fmt::format("flat_map[{}]", flatMap.nameAt(i)));
    }
    auto payload =
        makePayload(serialized, schema, projectionSubfields, payloadKind);
    SCOPED_TRACE(toString(payload.kind));
    StreamSlicer slicer{payload.schema, pool_.get(), StreamSlicer::Options{}};
    auto sliced = iobufToString(slicer.slice(payload.data, offset, length));
    auto output = deserialize(sliced, payload.schema);
    ASSERT_EQ(output->size(), length);
    for (uint32_t i = 0; i < length; ++i) {
      EXPECT_TRUE(input->equalValueAt(output.get(), offset + i, i));
    }
  }
}

TEST_P(StreamSlicerRawStreamApiTest, fuzzesRawStreamApi) {
  const bool useOutputBuffer = GetParam();
  std::mt19937 rng{67890};
  auto type = ROW({{"id", INTEGER()}});

  for (uint32_t iteration = 0; iteration < 48; ++iteration) {
    SCOPED_TRACE(iteration);
    const uint32_t rowCount = 1 + (rng() % 32);
    std::vector<int32_t> values;
    values.reserve(rowCount);
    for (uint32_t i = 0; i < rowCount; ++i) {
      values.emplace_back(static_cast<int32_t>(rng() % 2000) - 1000);
    }
    const uint32_t offset = rng() % rowCount;
    const uint32_t length = 1 + (rng() % (rowCount - offset));

    auto input = makeRowVector({"id"}, {makeFlatVector<int32_t>(values)});
    auto [_, schema] = serialize(input, type);
    const auto streamId =
        schema->asRow().childAt(0)->asScalar().scalarDescriptor().offset();

    for (const auto kind :
         {RawStreamKind::kSerialization,
          RawStreamKind::kProjection,
          RawStreamKind::kTablet,
          RawStreamKind::kTabletUncompressedChunk,
          RawStreamKind::kTabletZstdChunk}) {
      SCOPED_TRACE(toString(kind));
      const auto version = streamVersion(kind);
      const auto useVarintRowCount = !isTabletVersion(version);
      const auto encodedStream = encodeIntStream(values, useVarintRowCount);
      const auto storedStream = kind == RawStreamKind::kTabletUncompressedChunk
          ? makeUncompressedChunk(encodedStream)
          : kind == RawStreamKind::kTabletZstdChunk
          ? makeZstdChunk(encodedStream)
          : encodedStream;
      std::vector<std::string_view> inputStreams(streamId + 1);
      inputStreams[streamId] = storedStream;

      const auto options = kind == RawStreamKind::kProjection
          ? StreamSlicer::Options{}
          : StreamSlicer::Options{
                .streamVersion = version,
                .streamHasChunkHeader = streamHasChunkHeader(kind),
                .streamsUseVarintRowCount = useVarintRowCount,
            };
      StreamSlicer slicer{schema, pool_.get(), options};
      std::optional<Buffer> outputBuffer;
      if (useOutputBuffer) {
        outputBuffer.emplace(*pool_, inputStreamBytes(inputStreams));
      }
      auto sliced = slicer.slice(
          inputStreams,
          offset,
          length,
          outputBuffer.has_value() ? &outputBuffer.value() : nullptr);
      EXPECT_EQ(sliced.data.computeChainDataLength() == 0, useOutputBuffer);
      ASSERT_LT(streamId, sliced.streams.size());
      ASSERT_FALSE(sliced.streams[streamId].empty());

      auto encoding =
          EncodingFactory{
              Encoding::Options{.useVarintRowCount = useVarintRowCount}}
              .create(*pool_, sliced.streams[streamId], nullptr);
      std::vector<int32_t> actual(length);
      encoding->materialize(length, actual.data());
      const std::vector<int32_t> expected{
          values.begin() + offset, values.begin() + offset + length};
      EXPECT_EQ(actual, expected);
    }
  }
}

TEST_F(StreamSlicerTest, ownsSlicedStreamOutputBuffers) {
  auto writeStreams = [&](Buffer& buffer) {
    return std::vector<std::string_view>{
        buffer.writeString("abc"),
        {},
        buffer.writeString("def"),
    };
  };

  {
    Buffer buffer{*pool_};
    const auto streams = writeStreams(buffer);
    auto body = StreamSlicer::takeOwnershipAsIOBuf(streams, buffer);
    EXPECT_EQ(iobufToString(body), "abcdef");
  }

  {
    Buffer buffer{*pool_};
    const auto streams = writeStreams(buffer);
    auto chunks = std::make_shared<std::vector<velox::BufferPtr>>(
        buffer.transferBuffers());
    auto body = StreamSlicer::takeOwnershipAsIOBuf(
        streams, std::shared_ptr<const void>{std::move(chunks)});
    EXPECT_EQ(iobufToString(body), "abcdef");
  }

  {
    Buffer buffer{*pool_};
    auto body = StreamSlicer::takeOwnershipAsIOBuf(
        std::vector<std::string_view>{}, buffer);
    EXPECT_EQ(body.computeChainDataLength(), 0);
  }
}

TEST_F(StreamSlicerTest, fuzzesSlicedStreamOutputBufferOwnership) {
  constexpr uint32_t kSeed{0x136a7329};
  constexpr uint32_t kIterations{128};
  std::mt19937 rng{kSeed};
  std::uniform_int_distribution<size_t> numStreamsDistribution{0, 32};
  std::uniform_int_distribution<size_t> streamSizeDistribution{0, 256};
  std::uniform_int_distribution<int> byteDistribution{0, 255};

  auto makeBody = [&](bool useSharedOwner,
                      std::string& expected) -> folly::IOBuf {
    Buffer buffer{*pool_, /*initialChunkSize=*/64};
    std::vector<std::string_view> streams;
    const auto numStreams = numStreamsDistribution(rng);
    streams.reserve(numStreams);
    expected.clear();
    for (size_t i{0}; i < numStreams; ++i) {
      const auto streamSize = streamSizeDistribution(rng);
      if (streamSize == 0) {
        streams.emplace_back();
        continue;
      }
      std::string payload(streamSize, '\0');
      for (char& byte : payload) {
        byte = static_cast<char>(byteDistribution(rng));
      }
      streams.emplace_back(buffer.writeString(payload));
      expected.append(payload);
    }

    if (useSharedOwner) {
      auto chunks = std::make_shared<std::vector<velox::BufferPtr>>(
          buffer.transferBuffers());
      return StreamSlicer::takeOwnershipAsIOBuf(
          streams, std::shared_ptr<const void>{std::move(chunks)});
    }
    return StreamSlicer::takeOwnershipAsIOBuf(streams, buffer);
  };

  for (uint32_t iteration{0}; iteration < kIterations; ++iteration) {
    for (const bool useSharedOwner : {false, true}) {
      SCOPED_TRACE(
          fmt::format(
              "seed={} iteration={} useSharedOwner={}",
              kSeed,
              iteration,
              useSharedOwner));
      std::string expected;
      auto body = makeBody(useSharedOwner, expected);
      EXPECT_EQ(iobufToString(body), expected);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    OutputStorage,
    StreamSlicerRawStreamApiTest,
    ::testing::Bool(),
    [](const auto& info) {
      return info.param ? "CallerOutputBuffer" : "OwnedOutputBuffer";
    });

TEST_P(StreamSlicerPayloadApiTest, slicesFlatMap) {
  auto type = ROW({{"features", MAP(INTEGER(), INTEGER())}});
  auto features = makeMapVector(
      {2, 1, 0, 1, 2},
      makeFlatVector<int32_t>({1, 2, 2, 1, 1, 2}),
      makeFlatVector<int32_t>({10, 20, 21, 13, 14, 24}));
  features->setNull(2, true);
  auto input = makeRowVector({"features"}, {features});

  SerializerOptions options{
      .version = SerializationVersion::kSerialization,
      .flatMapColumns = {{"features", {}}},
  };
  Serializer serializer{options, type, pool_.get()};
  const std::string serialized{
      serializer.serialize(input, OrderedRanges::of(0, input->size()))};
  auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());

  auto payload = makePayload(
      serialized, schema, {"features[1]", "features[2]"}, GetParam());
  StreamSlicer slicer{payload.schema, pool_.get(), StreamSlicer::Options{}};
  auto sliced =
      iobufToString(slicer.slice(payload.data, /*offset=*/1, /*length=*/3));

  const char* pos = sliced.data();
  const auto header =
      readSerializationHeader(pos, sliced.data() + sliced.size(), true);
  EXPECT_TRUE(header.flags.requiresNullBarrier);

  auto output = deserialize(sliced, payload.schema);
  ASSERT_EQ(output->size(), 3);
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(input->equalValueAt(output.get(), i + 1, i));
  }
}

TEST_P(StreamSlicerPayloadApiTest, slicesEmptyArrayElementRange) {
  auto type = ROW({{"items", ARRAY(INTEGER())}});
  auto items =
      makeArrayVector({2, 0, 3}, makeFlatVector<int32_t>({10, 11, 20, 21, 22}));
  auto input = makeRowVector({"items"}, {items});
  auto [serialized, schema] = serialize(input, type);
  auto payload = makePayload(serialized, schema, {"items"}, GetParam());

  StreamSlicer slicer{payload.schema, pool_.get(), StreamSlicer::Options{}};
  auto sliced = iobufToString(slicer.slice(payload.data, /*offset=*/1, 1));
  auto output = deserialize(sliced, payload.schema);

  ASSERT_EQ(output->size(), 1);
  auto* row = output->as<RowVector>();
  ASSERT_NE(row, nullptr);
  auto* arrays = row->childAt(0)->as<ArrayVector>();
  ASSERT_NE(arrays, nullptr);
  EXPECT_EQ(arrays->sizeAt(0), 0);
}

TEST_P(StreamSlicerPayloadApiTest, returnsInputForFullPayloadSlice) {
  auto type = ROW({{"id", INTEGER()}});
  auto input =
      makeRowVector({"id"}, {makeFlatVector<int32_t>({10, 20, 30, 40, 50})});
  auto [serialized, schema] = serialize(input, type);
  auto payload = makePayload(serialized, schema, {"id"}, GetParam());

  StreamSlicer slicer{payload.schema, pool_.get(), StreamSlicer::Options{}};
  auto sliced = iobufToString(slicer.slice(
      payload.data, /*offset=*/0, static_cast<uint32_t>(input->size())));

  EXPECT_EQ(sliced, payload.data);
  auto output = deserialize(sliced, payload.schema);
  const std::vector<int32_t> expected{10, 20, 30, 40, 50};
  EXPECT_EQ(readColumn<int32_t>(output), expected);
}

TEST_P(StreamSlicerPayloadApiTest, slicesForcedIntegerEncodings) {
  const std::vector<std::pair<EncodingType, std::vector<int32_t>>> cases{
      {EncodingType::RLE, {10, 10, 20, 20, 20, 30, 40}},
      {EncodingType::Dictionary, {10, 20, 10, 30, 20, 40, 10}},
      {EncodingType::FixedBitWidth, {10, 11, 12, 13, 14, 15, 16}},
      {EncodingType::Varint, {10, 11, 12, 13, 14, 15, 16}},
      {EncodingType::Delta, {10, 11, 13, 16, 20, 25, 31}},
      {EncodingType::Constant, {10, 10, 10, 10, 10, 10, 10}},
      {EncodingType::MainlyConstant, {10, 10, 10, 20, 10, 10, 30}},
      {EncodingType::BlockBitPacking, {10, 11, 12, 13, 14, 15, 16}},
      {EncodingType::PFOR, {10, 11, 12, 13, 14, 15, 16}},
      {EncodingType::SimdForBitpack, {10, 11, 12, 13, 14, 15, 16}},
      {EncodingType::Huffman, {10, 11, 10, 12, 11, 10, 13}},
  };

  auto type = ROW({{"id", INTEGER()}});
  for (const auto& [encodingType, values] : cases) {
    SCOPED_TRACE(toString(encodingType));
    auto input = makeRowVector({"id"}, {makeFlatVector<int32_t>(values)});
    auto [serialized, schema] = serialize(
        input, type, encodingType, /*compressionOptions=*/std::nullopt);
    auto payload = makePayload(serialized, schema, {"id"}, GetParam());

    StreamSlicer slicer{payload.schema, pool_.get(), StreamSlicer::Options{}};
    auto sliced = iobufToString(slicer.slice(payload.data, 2, 3));
    auto output = deserialize(sliced, payload.schema);

    const std::vector<int32_t> expected{values.begin() + 2, values.begin() + 5};
    EXPECT_EQ(readColumn<int32_t>(output), expected);
  }
}

TEST_P(StreamSlicerPayloadApiTest, slicesAlpEncoding) {
  auto type = ROW({{"value", REAL()}});
  const std::vector<float> values{
      1.25f, 1.50f, 1.75f, 2.00f, 2.25f, 2.50f, 2.75f};
  auto input = makeRowVector({"value"}, {makeFlatVector<float>(values)});
  auto [serialized, schema] = serialize(
      input, type, EncodingType::ALP, /*compressionOptions=*/std::nullopt);
  auto payload = makePayload(serialized, schema, {"value"}, GetParam());

  StreamSlicer slicer{payload.schema, pool_.get(), StreamSlicer::Options{}};
  auto sliced = iobufToString(slicer.slice(payload.data, 2, 3));
  auto output = deserialize(sliced, payload.schema);

  const std::vector<float> expected{values.begin() + 2, values.begin() + 5};
  EXPECT_EQ(readColumn<float>(output), expected);
}

TEST_P(StreamSlicerPayloadApiTest, slicesCompressedTrivialStream) {
  auto type = ROW({{"id", INTEGER()}});
  std::vector<int32_t> values(1024, 7);
  for (size_t i = 0; i < values.size(); i += 128) {
    values[i] = i;
  }
  auto input = makeRowVector({"id"}, {makeFlatVector<int32_t>(values)});
  auto compressionOptions = CompressionOptions{
      .compressionAcceptRatio = 1.0,
      .compressionType = CompressionType::Zstd,
      .zstdMinCompressionSize = 0,
  };
  auto [serialized, schema] = serialize(
      input, type, EncodingType::Trivial, std::move(compressionOptions));
  auto payload = makePayload(serialized, schema, {"id"}, GetParam());

  StreamSlicer slicer{payload.schema, pool_.get(), StreamSlicer::Options{}};
  auto sliced = iobufToString(slicer.slice(payload.data, 127, 4));
  auto output = deserialize(sliced, payload.schema);

  const std::vector<int32_t> expected{
      values.begin() + 127, values.begin() + 131};
  EXPECT_EQ(readColumn<int32_t>(output), expected);
}

TEST_P(StreamSlicerPayloadApiTest, slicesArrayChildRange) {
  auto elements = makeFlatVector<int32_t>({10, 11, 20, 21, 22, 30});
  auto arrays = makeArrayVector({2, 0, 3, 1}, elements);
  auto type = ROW({{"items", ARRAY(INTEGER())}});
  auto input = makeRowVector({"items"}, {arrays});
  auto [serialized, schema] = serialize(input, type);
  auto payload = makePayload(serialized, schema, {"items"}, GetParam());

  StreamSlicer slicer{payload.schema, pool_.get(), StreamSlicer::Options{}};
  auto sliced = iobufToString(slicer.slice(payload.data, 2, 2));
  auto output = deserialize(sliced, payload.schema);

  ASSERT_EQ(output->size(), 2);
  auto* row = output->as<RowVector>();
  ASSERT_NE(row, nullptr);
  auto* result = row->childAt(0)->as<ArrayVector>();
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->sizeAt(0), 3);
  EXPECT_EQ(result->sizeAt(1), 1);
  auto* values = result->elements()->as<FlatVector<int32_t>>();
  ASSERT_NE(values, nullptr);
  EXPECT_EQ(values->valueAt(0), 20);
  EXPECT_EQ(values->valueAt(1), 21);
  EXPECT_EQ(values->valueAt(2), 22);
  EXPECT_EQ(values->valueAt(3), 30);
}

INSTANTIATE_TEST_SUITE_P(
    AcceptedPayload,
    StreamSlicerPayloadApiTest,
    ::testing::Values(
        SlicerPayloadKind::kSerialization,
        SlicerPayloadKind::kProjection),
    [](const auto& info) { return toString(info.param); });

TEST_F(StreamSlicerTest, rejectsZeroLengthSlice) {
  auto type = ROW({{"id", INTEGER()}});
  auto input = makeRowVector({"id"}, {makeFlatVector<int32_t>({1, 2, 3})});
  auto [serialized, schema] = serialize(input, type);

  StreamSlicer slicer{schema, pool_.get(), StreamSlicer::Options{}};
  NIMBLE_ASSERT_THROW(
      slicer.slice(serialized, 1, 0), "Slice length must be positive");
  NIMBLE_ASSERT_THROW(
      slicer.slice(std::vector<std::string_view>{}, 1, 0),
      "Slice length must be positive");
}

TEST_F(StreamSlicerTest, rejectsLegacyFormats) {
  auto type = ROW({{"id", INTEGER()}});
  auto input = makeRowVector({"id"}, {makeFlatVector<int32_t>({1})});
  auto [_, schema] = serialize(input, type);
  for (const auto version :
       {SerializationVersion::kLegacy,
        SerializationVersion::kLegacyCompact,
        SerializationVersion::kLegacySerialization}) {
    SCOPED_TRACE(toString(version));
    StreamSlicer slicer{
        schema,
        pool_.get(),
        StreamSlicer::Options{
            .streamVersion = version,
            .streamHasChunkHeader = false,
            .streamsUseVarintRowCount = false,
        }};
    const std::string payload{static_cast<char>(version)};
    NIMBLE_ASSERT_THROW(
        slicer.slice(payload, /*offset=*/0, /*length=*/1),
        "Unsupported StreamSlicer input version");
    NIMBLE_ASSERT_THROW(
        slicer.slice(
            std::vector<std::string_view>{},
            /*offset=*/0,
            /*length=*/1),
        "StreamSlicer raw streams must be kSerialization, kProjection, or "
        "kTablet encoded");
  }
}

TEST_F(StreamSlicerTest, rejectsFixedRowCountOptionForNonTabletFormats) {
  auto type = ROW({{"id", INTEGER()}});
  auto input = makeRowVector({"id"}, {makeFlatVector<int32_t>({1})});
  auto [_, schema] = serialize(input, type);
  for (const auto version : {
           SerializationVersion::kSerialization,
           SerializationVersion::kProjection,
       }) {
    SCOPED_TRACE(toString(version));
    StreamSlicer slicer{
        schema,
        pool_.get(),
        StreamSlicer::Options{
            .streamVersion = version,
            .streamHasChunkHeader = false,
            .streamsUseVarintRowCount = false,
        }};
    NIMBLE_ASSERT_THROW(
        slicer.slice(
            std::vector<std::string_view>{},
            /*offset=*/0,
            /*length=*/1),
        "Non-tablet streams must use varint row counts");
  }
}

} // namespace
} // namespace facebook::nimble::serde
