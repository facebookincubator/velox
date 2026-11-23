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
#include <utility>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/HybridFlatMap.h"
#include "velox/dwio/nimble/common/SchemaBuilder.h"
#include "velox/dwio/nimble/common/SchemaGenerated.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/serializer/StreamDataParser.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/type/Type.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble {
namespace {

// The parameterized tests run once per Serializer writer version.
class SerializerTest : public ::testing::TestWithParam<SerializationVersion> {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("serializer_test");
    pool_ = rootPool_->addLeafChild("leaf");
    vectorMaker_ = std::make_unique<velox::test::VectorMaker>(pool_.get());
  }

  static velox::RowTypePtr mapRowType() {
    return velox::ROW(
        {{"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())}});
  }

  // Builds rows whose columns compress well: a constant column, repeated
  // strings, nulls, and nested types. Encoding selection writes some of them
  // with encodings other than Trivial.
  velox::RowVectorPtr makeRepetitiveRows() {
    constexpr velox::vector_size_t kNumRows{1'000};
    return vectorMaker_->rowVector(
        {"constant", "sequence", "label", "nullable", "array", "map"},
        {
            vectorMaker_->flatVector<int32_t>(
                kNumRows, [](auto /*row*/) { return 7; }),
            vectorMaker_->flatVector<int64_t>(
                kNumRows, [](auto row) { return row; }),
            vectorMaker_->flatVector<velox::StringView>(
                kNumRows,
                [](auto row) {
                  return velox::StringView(
                      row % 2 == 0 ? "the first repeated label"
                                   : "the second repeated label");
                }),
            vectorMaker_->flatVector<int32_t>(
                kNumRows,
                [](auto row) { return row; },
                velox::test::VectorMaker::nullEvery(3)),
            vectorMaker_->arrayVector<int32_t>(
                kNumRows,
                [](auto row) { return row % 4; },
                [](auto row) { return row; }),
            vectorMaker_->mapVector<int32_t, double>(
                kNumRows,
                [](auto row) { return row % 3; },
                [](auto row) { return row % 5; },
                [](auto row) { return row * 0.5; }),
        });
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
};

TEST_F(SerializerTest, hybridFlatMapRequiresSerializationFormat) {
  const SerializerOptions options{
      .version = SerializationVersion::kProjection,
      .hybridFlatMapColumns =
          {{"features",
            HybridFlatMap{
                ScalarKind::Int32,
                {{.groupId = 0, .groupKeys = {"1"}},
                 {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}}},
                HybridFlatMap::KeyPresenceMode::kBitmap,
            }}},
  };

  NIMBLE_ASSERT_THROW(
      std::make_unique<Serializer>(options, mapRowType(), pool_.get()),
      "Serializer writes must use kSerialization or kBlob");
}

TEST_P(SerializerTest, validatesHybridFlatMapColumnConfiguration) {
  const auto type = mapRowType();

  NIMBLE_ASSERT_USER_THROW(
      std::make_unique<Serializer>(
          SerializerOptions{
              .version = GetParam(),
              .hybridFlatMapColumns =
                  {{"missing",
                    HybridFlatMap{
                        ScalarKind::Int32,
                        {{.groupId = 0, .groupKeys = {"1"}},
                         {.groupId = HybridFlatMap::kDefaultGroupId,
                          .groupKeys = {}}},
                        HybridFlatMap::KeyPresenceMode::kBitmap,
                    }}},
          },
          type,
          pool_.get()),
      "Hybrid FlatMap column 'missing' does not exist");

  NIMBLE_ASSERT_USER_THROW(
      std::make_unique<Serializer>(
          SerializerOptions{
              .version = GetParam(),
              .flatMapColumns = {{"features", {}}},
              .hybridFlatMapColumns =
                  {{"features",
                    HybridFlatMap{
                        ScalarKind::Int32,
                        {{.groupId = 0, .groupKeys = {"1"}},
                         {.groupId = HybridFlatMap::kDefaultGroupId,
                          .groupKeys = {}}},
                        HybridFlatMap::KeyPresenceMode::kBitmap,
                    }}},
          },
          type,
          pool_.get()),
      "cannot use both FlatMap and hybrid FlatMap options");

  const auto scalarType = velox::ROW({{"features", velox::BIGINT()}});
  NIMBLE_ASSERT_USER_THROW(
      std::make_unique<Serializer>(
          SerializerOptions{
              .version = GetParam(),
              .hybridFlatMapColumns =
                  {{"features",
                    HybridFlatMap{
                        ScalarKind::Int32,
                        {{.groupId = 0, .groupKeys = {"1"}},
                         {.groupId = HybridFlatMap::kDefaultGroupId,
                          .groupKeys = {}}},
                        HybridFlatMap::KeyPresenceMode::kBitmap,
                    }}},
          },
          scalarType,
          pool_.get()),
      "Hybrid FlatMap column 'features' must be a MAP");
}

TEST_P(SerializerTest, hybridFlatMapSerializesDefaultFeature) {
  const SerializerOptions options{
      .version = GetParam(),
      .hybridFlatMapColumns =
          {{"features",
            HybridFlatMap{
                ScalarKind::Int32,
                {{.groupId = 0, .groupKeys = {"1"}},
                 {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}}},
                HybridFlatMap::KeyPresenceMode::kBitmap,
            }}},
  };

  Serializer serializer{options, mapRowType(), pool_.get()};
  const auto input = vectorMaker_->rowVector(
      {"features"},
      {vectorMaker_->mapVector<int32_t, double>({{{1, 1.0}, {99, 99.0}}})});
  EXPECT_NO_THROW(
      serializer.serialize(input, OrderedRanges::of(0, input->size())));
}

TEST_P(SerializerTest, hybridFlatMapGroupKeysLayout) {
  // Returns the encoding of the integer keys in the schema of `options`, or
  // nullopt when the schema writes them as strings.
  const auto groupKeysEncoding =
      [&](const SerializerOptions& options) -> std::optional<EncodingType> {
    Serializer serializer{options, mapRowType(), pool_.get()};
    const auto nodes = serializer.schemaBuilder().schemaNodes();
    const auto& serialized = nodes[1].attributes().at(0).second;
    const auto* encoded =
        flatbuffers::GetRoot<serialization::HybridFlatMap>(serialized.data())
            ->encoded_group_keys();
    if (encoded == nullptr) {
      return std::nullopt;
    }
    return EncodingPrefix::encodingType(
        {reinterpret_cast<const char*>(encoded->data()), encoded->size()});
  };
  SerializerOptions options{
      .version = GetParam(),
      .hybridFlatMapColumns =
          {{"features",
            HybridFlatMap{
                ScalarKind::Int32,
                {{.groupId = 0, .groupKeys = {"1", "2", "3"}},
                 {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}}},
                HybridFlatMap::KeyPresenceMode::kKeyList,
            }}},
  };
  EXPECT_EQ(groupKeysEncoding(options), std::nullopt);

  for (const auto encodingType :
       {EncodingType::FixedBitWidth,
        EncodingType::Varint,
        EncodingType::EliasFano}) {
    SCOPED_TRACE(toString(encodingType));
    options.hybridFlatMapGroupKeysLayout.emplace(
        encodingType, EncodingLayout::Config{}, CompressionType::Uncompressed);
    EXPECT_EQ(groupKeysEncoding(options), encodingType);
  }
}

TEST_F(SerializerTest, failedSerializeRestoresBufferAndDropsRows) {
  // The policy creator fails the first stream encoding, which happens after the
  // field writers took the rows of the batch and the header was written.
  auto failNextEncoding = std::make_shared<bool>(true);
  SerializerOptions options;
  options.encodingSelectionPolicyCreator =
      [failNextEncoding,
       defaultCreator =
           defaultEncodingSelectionPolicyCreator()](DataType dataType) {
        if (std::exchange(*failNextEncoding, false)) {
          NIMBLE_USER_FAIL("Injected encoding failure");
        }
        return defaultCreator(dataType);
      };
  const auto failedBatch = vectorMaker_->rowVector(
      {"id"}, {vectorMaker_->flatVector<int64_t>({1, 2, 3})});
  const auto nextBatch = vectorMaker_->rowVector(
      {"id"}, {vectorMaker_->flatVector<int64_t>({4, 5})});
  Serializer serializer{options, failedBatch->type(), pool_.get()};
  const std::string prefix{"prefix"};
  std::string buffer{prefix};

  NIMBLE_ASSERT_USER_THROW(
      serializer.serialize(
          failedBatch, OrderedRanges::of(0, failedBatch->size()), buffer),
      "Injected encoding failure");
  EXPECT_EQ(buffer, prefix);

  serializer.serialize(
      nextBatch, OrderedRanges::of(0, nextBatch->size()), buffer);
  ASSERT_TRUE(buffer.starts_with(prefix));
  const Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get()};
  velox::VectorPtr output;
  deserializer.deserialize(
      std::string_view{buffer}.substr(prefix.size()), output);
  velox::test::assertEqualVectors(nextBatch, output);
}

TEST_P(SerializerTest, abortedPoolThrowsFromSerialize) {
  const auto input = vectorMaker_->rowVector(
      {"id"}, {vectorMaker_->flatVector<int64_t>({1, 2, 3})});
  Serializer serializer{
      SerializerOptions{.version = GetParam()}, input->type(), pool_.get()};

  // An aborted pool fails every allocation that grows its reservation.
  // serialize() fails at the first one, before it appends a byte, and must
  // throw: its cleanup on failure must not allocate.
  try {
    NIMBLE_USER_FAIL("Injected pool abort");
  } catch (const NimbleUserError&) {
    rootPool_->abort(std::current_exception());
  }
  std::string buffer;

  NIMBLE_ASSERT_USER_THROW(
      serializer.serialize(input, OrderedRanges::of(0, input->size()), buffer),
      "Injected pool abort");
  EXPECT_TRUE(buffer.empty());
}

TEST_P(SerializerTest, failedFlatMapBatchLeavesNoValuesBehind) {
  Serializer serializer{
      SerializerOptions{
          .version = GetParam(), .flatMapColumns = {{"features", {"1", "5"}}}},
      mapRowType(),
      pool_.get()};

  // Key 9 is not a declared key, so the batch fails after the writers of keys
  // 5 and 1 took their values.
  const auto failedBatch = vectorMaker_->rowVector(
      {"features"},
      {vectorMaker_->mapVector<int32_t, double>(
          {{{5, 0.5}, {1, 1.0}, {9, 9.0}}})});
  const auto nextBatch = vectorMaker_->rowVector(
      {"features"},
      {vectorMaker_->mapVector<int32_t, double>({{{1, 10.0}, {5, 50.0}}})});

  NIMBLE_ASSERT_USER_THROW(
      serializer.serialize(
          failedBatch, OrderedRanges::of(0, failedBatch->size())),
      "Unknown flatmap key");
  const std::string serialized{
      serializer.serialize(nextBatch, OrderedRanges::of(0, nextBatch->size()))};

  const Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get()};
  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);
  velox::test::assertEqualVectors(nextBatch, output);
}

// Returns the encoding tree of a serialized stream. For a stream with nulls,
// the capture skips the Nullable wrapper and its nulls child, and returns the
// tree of the non-null values.
EncodingLayout captureLayout(std::string_view stream) {
  return EncodingLayoutCapture::capture(stream, {.useVarintRowCount = true});
}

// Expects the encoding that 'layout' captures, and every encoding nested in
// it, to be Trivial.
void expectTrivialLayout(const EncodingLayout& layout) {
  ASSERT_EQ(layout.encodingType(), EncodingType::Trivial);
  for (uint8_t childIndex = 0; childIndex < layout.childrenCount();
       ++childIndex) {
    const auto& childLayout = layout.child(childIndex);
    if (childLayout.has_value()) {
      ASSERT_NO_FATAL_FAILURE(expectTrivialLayout(*childLayout));
    }
  }
}

// Expects every encoding in 'stream' to be Trivial, except the Nullable
// wrapper of a stream with nulls. The wrapper is parsed by hand, because
// captureLayout() drops its nulls child.
void expectTrivialStream(std::string_view stream) {
  if (EncodingPrefix::encodingType(stream) != EncodingType::Nullable) {
    ASSERT_NO_FATAL_FAILURE(expectTrivialLayout(captureLayout(stream)));
    return;
  }

  // [prefix][valuesSize:u32][values][nulls]
  const char* pos =
      stream.data() + EncodingPrefix::prefixSize(stream, /*useVarint=*/true);
  const auto valuesSize = encoding::readUint32(pos);
  ASSERT_NO_FATAL_FAILURE(
      expectTrivialLayout(captureLayout({pos, valuesSize})));
  const char* nulls = pos + valuesSize;
  ASSERT_NO_FATAL_FAILURE(expectTrivialLayout(captureLayout(
      {nulls, static_cast<size_t>(stream.data() + stream.size() - nulls)})));
}

// Returns the serialization header of 'serialized'.
serde::SerializationHeader readHeader(std::string_view serialized) {
  const char* pos = serialized.data();
  return serde::readSerializationHeader(
      pos, serialized.data() + serialized.size());
}

TEST_F(SerializerTest, blobWritesOnlyTrivialStreams) {
  const auto input = makeRepetitiveRows();

  // kSerialization writes some of these streams with other encodings, so the
  // checks below are not vacuous.
  {
    Serializer serializer{SerializerOptions{}, input->type(), pool_.get()};
    const std::string serialized{
        serializer.serialize(input, OrderedRanges::of(0, input->size()))};
    serde::StreamDataParser parser{pool_.get()};
    parser.initialize(serialized);
    uint32_t numNonTrivialStreams{0};
    parser.iterateStreams([&](uint32_t /*offset*/, std::string_view stream) {
      if (captureLayout(stream).encodingType() != EncodingType::Trivial) {
        ++numNonTrivialStreams;
      }
    });
    EXPECT_GT(numNonTrivialStreams, 0);
  }

  Serializer serializer{
      SerializerOptions{.version = SerializationVersion::kBlob},
      input->type(),
      pool_.get()};
  const std::string serialized{
      serializer.serialize(input, OrderedRanges::of(0, input->size()))};

  const auto header = readHeader(serialized);
  EXPECT_EQ(header.version, SerializationVersion::kBlob);
  EXPECT_EQ(header.rowCount, input->size());
  EXPECT_EQ(header.blobCompression, CompressionType::Lz4);

  serde::StreamDataParser parser{pool_.get()};
  parser.initialize(serialized);
  uint32_t numStreams{0};
  parser.iterateStreams([&](uint32_t streamOffset, std::string_view stream) {
    SCOPED_TRACE(fmt::format("streamOffset={}", streamOffset));
    ASSERT_NO_FATAL_FAILURE(expectTrivialStream(stream));
    ++numStreams;
  });

  // constant, sequence, label and nullable write one stream each, the array
  // writes lengths and elements, and the map lengths, keys and values. The
  // row writes no nulls stream, because no row is null.
  EXPECT_EQ(numStreams, 9);
}

TEST_F(SerializerTest, blobCompressesEachStreamWithCompressionOptions) {
  const auto input = makeRepetitiveRows();
  Serializer serializer{
      SerializerOptions{
          .version = SerializationVersion::kBlob,
          .compressionOptions =
              CompressionOptions{
                  .compressionAcceptRatio = 1.0F,
                  .compressionType = CompressionType::Zstd,
                  .zstdMinCompressionSize = 0},
          .blobCompressionOptions =
              {.compressionType = CompressionType::Uncompressed},
      },
      input->type(),
      pool_.get()};
  const std::string serialized{
      serializer.serialize(input, OrderedRanges::of(0, input->size()))};
  EXPECT_EQ(
      readHeader(serialized).blobCompression, CompressionType::Uncompressed);

  serde::StreamDataParser parser{pool_.get()};
  parser.initialize(serialized);
  uint32_t numStreams{0};
  parser.iterateStreams([&](uint32_t streamOffset, std::string_view stream) {
    SCOPED_TRACE(fmt::format("streamOffset={}", streamOffset));
    EXPECT_EQ(captureLayout(stream).compressionType(), CompressionType::Zstd);
    ++numStreams;
  });

  // The same 9 streams as in blobWritesOnlyTrivialStreams.
  EXPECT_EQ(numStreams, 9);
}

TEST_F(SerializerTest, blobAppendsToNonEmptyBuffer) {
  const auto input = makeRepetitiveRows();
  Serializer serializer{
      SerializerOptions{.version = SerializationVersion::kBlob},
      input->type(),
      pool_.get()};
  const std::string prefix{"prefix"};
  std::string buffer{prefix};

  serializer.serialize(input, OrderedRanges::of(0, input->size()), buffer);

  ASSERT_TRUE(buffer.starts_with(prefix));
  const auto serialized = std::string_view{buffer}.substr(prefix.size());
  EXPECT_EQ(readHeader(serialized).blobCompression, CompressionType::Lz4);
  const Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get()};
  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);
  velox::test::assertEqualVectors(input, output);
}

TEST_F(SerializerTest, blobRejectsEncodingLayoutTree) {
  const SerializerOptions options{
      .version = SerializationVersion::kBlob,
      .encodingLayoutTree = EncodingLayoutTree{Kind::Row, {}, ""},
  };

  NIMBLE_ASSERT_USER_THROW(
      std::make_unique<Serializer>(options, mapRowType(), pool_.get()),
      "kBlob does not support encodingLayoutTree");
}

TEST_F(SerializerTest, blobRejectsUnsupportedBlobCompression) {
  const SerializerOptions options{
      .version = SerializationVersion::kBlob,
      .blobCompressionOptions =
          {.compressionType = static_cast<CompressionType>(99)},
  };

  NIMBLE_ASSERT_USER_THROW(
      std::make_unique<Serializer>(options, mapRowType(), pool_.get()),
      "kBlob blob compression must be Lz4, Zstd, MetaInternal, OpenZL or Uncompressed");
}

INSTANTIATE_TEST_SUITE_P(
    WriterVersions,
    SerializerTest,
    ::testing::Values(
        SerializationVersion::kSerialization,
        SerializationVersion::kBlob),
    [](const auto& info) { return toString(info.param); });

} // namespace
} // namespace facebook::nimble
