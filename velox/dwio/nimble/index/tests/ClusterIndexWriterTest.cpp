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
#include <fmt/ranges.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/buffer/Buffer.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/PrefixEncoding.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/index/ClusterIndexWriter.h"
#include "velox/dwio/nimble/index/IndexConfig.h"
#include "velox/dwio/nimble/index/KeyChunkDecoder.h"
#include "velox/dwio/nimble/index/SortOrder.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestUtils.h"
#include "velox/dwio/nimble/tablet/MetadataInput.h"
#include "velox/serializers/KeyEncoder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble::index::test {
namespace {

velox::BufferPtr toBufferPtr(
    std::string_view data,
    velox::memory::MemoryPool* pool) {
  auto buffer = velox::AlignedBuffer::allocate<char>(data.size(), pool);
  std::memcpy(buffer->asMutable<char>(), data.data(), data.size());
  return buffer;
}

class ClusterIndexWriterTest : public testing::Test,
                               public velox::test::VectorTestBase {
 public:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

 protected:
  static constexpr std::string_view kCol0{"col0"};
  static constexpr std::string_view kCol1{"col1"};
  static constexpr std::string_view kCol2{"col2"};

  static std::shared_ptr<const IndexConfig> buildIndexConfig(
      std::vector<std::string> columns,
      EncodingLayout encodingLayout =
          EncodingLayout{
              EncodingType::Prefix,
              {},
              CompressionType::Uncompressed},
      std::vector<SortOrder> sortOrders = {},
      bool enforceKeyOrder = false,
      bool noDuplicateKey = false,
      uint64_t maxRowsPerKeyChunk = 10'000,
      CompressionType keyChunkCompressionType = CompressionType::Uncompressed) {
    auto builder = ClusterIndexConfigBuilder{}
                       .withKeyColumns(std::move(columns))
                       .withEncodingLayout(std::move(encodingLayout))
                       .withSortOrders(std::move(sortOrders))
                       .withEnforceKeyOrder(enforceKeyOrder)
                       .withMaxRowsPerKeyChunk(maxRowsPerKeyChunk)
                       .withKeyChunkCompressionType(keyChunkCompressionType);
    if (noDuplicateKey) {
      builder.withNoDuplicateKey(true);
    }
    return builder.build();
  }

  static std::unique_ptr<ClusterIndexWriter> createWriter(
      const std::shared_ptr<const IndexConfig>& config,
      const velox::TypePtr& type,
      velox::memory::MemoryPool* pool) {
    return ClusterIndexWriter::create(*config, type, pool);
  }

  void SetUp() override {
    type_ = velox::ROW(
        {{std::string(kCol0), velox::VARCHAR()},
         {std::string(kCol1), velox::INTEGER()},
         {std::string(kCol2), velox::VARCHAR()}});
  }

  std::unique_ptr<velox::serializer::KeyEncoder> createKeyEncoder() const {
    const auto keyType = velox::ROW({{std::string(kCol1), velox::INTEGER()}});
    return velox::serializer::KeyEncoder::create(
        {std::string(kCol1)},
        keyType,
        {velox::core::SortOrder{true, false}},
        pool_.get());
  }

  static std::shared_ptr<const IndexConfig> indexConfig(
      EncodingType encodingType = EncodingType::Prefix) {
    EncodingLayout layout = encodingType == EncodingType::Prefix
        ? EncodingLayout{EncodingType::Prefix, {}, CompressionType::Zstd}
        : EncodingLayout{
              EncodingType::Trivial,
              {},
              CompressionType::Zstd,
              {EncodingLayout{
                  EncodingType::Trivial, {}, CompressionType::Uncompressed}}};
    return buildIndexConfig({std::string(kCol1)}, std::move(layout));
  }

  static std::shared_ptr<const IndexConfig> orderedIndexConfig(
      bool noDuplicateKey) {
    return buildIndexConfig(
        {std::string(kCol1)},
        EncodingLayout{
            EncodingType::Trivial,
            {},
            CompressionType::Uncompressed,
            {EncodingLayout{
                EncodingType::Trivial, {}, CompressionType::Uncompressed}}},
        {},
        true,
        noDuplicateKey);
  }

  velox::RowTypePtr type_;
};

TEST_F(ClusterIndexWriterTest, createWithValidConfig) {
  auto type = velox::ROW({{"col0", velox::INTEGER()}});
  for (auto compressionType :
       {CompressionType::Uncompressed,
        CompressionType::Zstd,
        CompressionType::MetaInternal}) {
    for (auto encodingType : {EncodingType::Prefix, EncodingType::Trivial}) {
      SCOPED_TRACE(
          fmt::format(
              "compressionType: {}, encodingType: {}",
              compressionType,
              toString(encodingType)));
      EncodingLayout layout = encodingType == EncodingType::Prefix
          ? EncodingLayout{EncodingType::Prefix, {}, compressionType}
          : EncodingLayout{
                EncodingType::Trivial,
                {},
                compressionType,
                {EncodingLayout{
                    EncodingType::Trivial, {}, CompressionType::Uncompressed}}};
      auto config = ClusterIndexConfigBuilder{}
                        .withKeyColumns({"col0"})
                        .withEncodingLayout(std::move(layout))
                        .build();
      auto writer = createWriter(config, type_, pool_.get());
      EXPECT_NE(writer, nullptr);
    }
  }
}

TEST_F(ClusterIndexWriterTest, createWithInvalidConfig) {
  auto type = velox::ROW({{"col0", velox::INTEGER()}});

  {
    SCOPED_TRACE("non-existent column");
    auto config =
        ClusterIndexConfigBuilder{}
            .withKeyColumns({"non_existent_col"})
            .withEncodingLayout(
                EncodingLayout{
                    EncodingType::Trivial, {}, CompressionType::Uncompressed})
            .build();
    VELOX_ASSERT_USER_THROW(
        createWriter(config, type, pool_.get()), "Field not found");
  }

  {
    SCOPED_TRACE("non-supported encoding");
    auto config = ClusterIndexConfigBuilder{}
                      .withKeyColumns({"col0"})
                      .withEncodingLayout(
                          EncodingLayout{
                              EncodingType::FixedBitWidth,
                              {},
                              CompressionType::Uncompressed})
                      .build();
    NIMBLE_ASSERT_THROW(
        createWriter(config, type, pool_.get()),
        "Index key stream only supports Prefix or Trivial encoding");
  }

  {
    SCOPED_TRACE("non-supported key chunk compression");
    auto config =
        ClusterIndexConfigBuilder{}
            .withKeyColumns({"col0"})
            .withKeyChunkCompressionType(CompressionType::MetaInternal)
            .build();
    NIMBLE_ASSERT_THROW(
        createWriter(config, type, pool_.get()),
        "Index key chunk compression only supports Uncompressed, Zstd, or Lz4");
  }
}

TEST_F(ClusterIndexWriterTest, keyChunkCompressionProducesRequestedChunkCodec) {
  // Sorted VARCHAR keys sharing a long common prefix so the encoded key stream
  // is highly compressible and ChunkedStreamWriter emits the requested codec's
  // chunk header (it falls back to Uncompressed when compression would not
  // shrink the payload).
  constexpr int kNumRows = 4000;
  const std::string prefix(64, 'a');
  std::vector<std::string> col0Storage(kNumRows);
  std::vector<velox::StringView> col0(kNumRows);
  std::vector<int32_t> col1(kNumRows);
  std::vector<velox::StringView> col2(kNumRows);
  for (int i = 0; i < kNumRows; ++i) {
    col0Storage[i] = fmt::format("{}{:05d}", prefix, i);
    col0[i] = velox::StringView(col0Storage[i]);
    col1[i] = i;
    col2[i] = velox::StringView("v");
  }
  auto batch = makeRowVector(
      {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
      {makeFlatVector<velox::StringView>(col0),
       makeFlatVector<int32_t>(col1),
       makeFlatVector<velox::StringView>(col2)});

  // Decoded keys from the uncompressed run; every codec must decode to these.
  std::vector<std::string> referenceKeys;
  for (const auto compressionType :
       {CompressionType::Uncompressed,
        CompressionType::Zstd,
        CompressionType::Lz4}) {
    SCOPED_TRACE(toString(compressionType));
    auto config = buildIndexConfig(
        {std::string(kCol0)},
        EncodingLayout{EncodingType::Prefix, {}, CompressionType::Uncompressed},
        {},
        false,
        false,
        10'000,
        compressionType);
    auto writer = createWriter(config, type_, pool_.get());
    ASSERT_NE(writer, nullptr);
    writer->write(batch);

    std::string keyStreamData;
    WriteDataFn writeDataFn = [&](const std::vector<std::string_view>& segments)
        -> std::pair<uint64_t, uint32_t> {
      const uint64_t offset = keyStreamData.size();
      uint32_t total = 0;
      for (const auto& segment : segments) {
        keyStreamData.append(segment.data(), segment.size());
        total += segment.size();
      }
      return {offset, total};
    };
    uint32_t nextSectionId = 0;
    CreateMetadataSectionFn createMetadataFn = [&](std::string_view) {
      return MetadataSection{nextSectionId++, 0, CompressionType::Uncompressed};
    };
    writer->flush(writeDataFn, createMetadataFn);

    // The first chunk's header compression-type byte (the last byte of the
    // 5-byte header) reflects the configured codec.
    ASSERT_GE(keyStreamData.size(), kChunkHeaderSize);
    EXPECT_EQ(
        static_cast<uint8_t>(keyStreamData[kChunkHeaderSize - 1]),
        static_cast<uint8_t>(compressionType));

    // Decode the writer's actual output and confirm every codec round-trips to
    // the same keys as the uncompressed run: compression must not alter
    // content.
    velox::BufferPtr decodeBuffer;
    auto decoded = decodeKeyChunk(
        std::make_unique<velox::dwio::common::SeekableArrayInputStream>(
            keyStreamData.data(), keyStreamData.size()),
        *pool_,
        decodeBuffer);
    ASSERT_NE(decoded, nullptr);
    ASSERT_NE(decoded->encoding, nullptr);
    ASSERT_EQ(decoded->encoding->rowCount(), static_cast<uint32_t>(kNumRows));
    auto materialized =
        decoded->encoding->materialize(0, static_cast<uint32_t>(kNumRows));
    if (compressionType == CompressionType::Uncompressed) {
      referenceKeys = std::move(materialized);
    } else {
      EXPECT_EQ(materialized, referenceKeys);
    }
  }
}

TEST_F(ClusterIndexWriterTest, rejectNullKeys) {
  // Test that null values in key columns are rejected
  {
    SCOPED_TRACE("null in key column throws");
    auto config = indexConfig();
    auto writer = createWriter(config, type_, pool_.get());

    auto batch = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"a", "b", "c"}),
         makeNullableFlatVector<int32_t>({1, std::nullopt, 3}),
         makeFlatVector<velox::StringView>({"d", "e", "f"})});

    NIMBLE_ASSERT_USER_THROW(
        writer->write(batch),
        "Null value not allowed in index key column at index 1: found 1 null(s)");
  }

  // Test multiple nulls in key column
  {
    SCOPED_TRACE("multiple nulls in key column");
    auto config = indexConfig();
    auto writer = createWriter(config, type_, pool_.get());

    auto batch = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"a", "b", "c", "d", "e"}),
         makeNullableFlatVector<int32_t>(
             {std::nullopt, 2, std::nullopt, 4, std::nullopt}),
         makeFlatVector<velox::StringView>({"f", "g", "h", "i", "j"})});

    NIMBLE_ASSERT_USER_THROW(
        writer->write(batch),
        "Null value not allowed in index key column at index 1: found 3 null(s)");
  }

  // Test no nulls passes validation
  {
    SCOPED_TRACE("no nulls in key column passes");
    auto config = indexConfig();
    auto writer = createWriter(config, type_, pool_.get());

    auto batch = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"a", "b", "c"}),
         makeFlatVector<int32_t>({1, 2, 3}),
         makeFlatVector<velox::StringView>({"d", "e", "f"})});

    EXPECT_NO_THROW(writer->write(batch));
  }
}

TEST_F(ClusterIndexWriterTest, multiColumnWithDifferentSortOrders) {
  const auto type = velox::ROW({
      {"col0", velox::INTEGER()},
      {"col1", velox::VARCHAR()},
      {"col2", velox::BIGINT()},
  });

  {
    SCOPED_TRACE("valid: empty sort orders uses default");
    auto config = buildIndexConfig({"col0", "col1"});
    auto writer = createWriter(config, type, pool_.get());
    EXPECT_NE(writer, nullptr);
  }

  {
    SCOPED_TRACE("invalid: sort orders size mismatch (fewer)");
    auto config = buildIndexConfig(
        {"col0", "col1", "col2"},
        EncodingLayout{EncodingType::Prefix, {}, CompressionType::Uncompressed},
        {SortOrder{.ascending = true}, SortOrder{.ascending = false}});
    NIMBLE_ASSERT_THROW(
        createWriter(config, type, pool_.get()),
        "Cluster index columns and sort orders must have the same size");
  }

  {
    SCOPED_TRACE("invalid: sort orders size mismatch (more)");
    auto config = buildIndexConfig(
        {"col0", "col1"},
        EncodingLayout{EncodingType::Prefix, {}, CompressionType::Uncompressed},
        {SortOrder{.ascending = true},
         SortOrder{.ascending = false},
         SortOrder{.ascending = true}});
    NIMBLE_ASSERT_THROW(
        createWriter(config, type, pool_.get()),
        "Cluster index columns and sort orders must have the same size");
  }

  // Data test: write data with multiple columns and different sort orders,
  // then verify the encoded keys match what KeyEncoder produces.
  {
    SCOPED_TRACE("data test: write and verify encoded keys");
    const std::vector<SortOrder> sortOrders = {
        SortOrder{.ascending = true},
        SortOrder{.ascending = false},
        SortOrder{.ascending = true}};
    const std::vector<velox::core::SortOrder> veloxSortOrders = {
        velox::core::kAscNullsLast,
        velox::core::kDescNullsLast,
        velox::core::kAscNullsLast};

    auto config = buildIndexConfig(
        {"col0", "col1", "col2"},
        EncodingLayout{EncodingType::Prefix, {}, CompressionType::Zstd},
        sortOrders);

    auto writer = createWriter(config, type, pool_.get());
    ASSERT_NE(writer, nullptr);

    // Create input data with 3 rows.
    auto batch = makeRowVector(
        {"col0", "col1", "col2"},
        {makeFlatVector<int32_t>({1, 2, 3}),
         makeFlatVector<velox::StringView>({"a", "b", "c"}),
         makeFlatVector<int64_t>({100, 200, 300})});

    writer->write(batch);
  }

  // Data test: verify that different sort orders produce different encoded
  // keys for the same input data.
  {
    SCOPED_TRACE("data test: different sort orders produce different keys");
    const auto singleColType = velox::ROW({{"col0", velox::INTEGER()}});

    // Create input data.
    auto batch = makeRowVector({"col0"}, {makeFlatVector<int32_t>({1, 2, 3})});

    // Write with ascending order.
    auto configAsc = buildIndexConfig(
        {"col0"},
        EncodingLayout{EncodingType::Prefix, {}, CompressionType::Zstd},
        {SortOrder{.ascending = true}});
    auto writerAsc = createWriter(configAsc, singleColType, pool_.get());
    writerAsc->write(batch);

    // Write with descending order.
    auto configDesc = buildIndexConfig(
        {"col0"},
        EncodingLayout{EncodingType::Prefix, {}, CompressionType::Zstd},
        {SortOrder{.ascending = false}});
    auto writerDesc = createWriter(configDesc, singleColType, pool_.get());
    writerDesc->write(batch);
  }
}

TEST_F(ClusterIndexWriterTest, emptyWrite) {
  auto config = indexConfig(EncodingType::Trivial);
  auto writer = createWriter(config, type_, pool_.get());
  ASSERT_NE(writer, nullptr);
}

TEST_F(ClusterIndexWriterTest, writeEmptyVectorOnly) {
  auto config = indexConfig(EncodingType::Trivial);
  auto writer = createWriter(config, type_, pool_.get());
  ASSERT_NE(writer, nullptr);

  // Write a single empty vector.
  auto emptyBatch = makeRowVector(
      {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
      {makeFlatVector<velox::StringView>({}),
       makeFlatVector<int32_t>({}),
       makeFlatVector<velox::StringView>({})});

  writer->write(emptyBatch);
}

// Test writing empty vectors at different positions: beginning, middle, and
// end.
TEST_F(ClusterIndexWriterTest, writeEmptyVectorAtDifferentPositions) {
  auto config = indexConfig(EncodingType::Trivial);
  auto writer = createWriter(config, type_, pool_.get());
  ASSERT_NE(writer, nullptr);

  auto emptyBatch = makeRowVector(
      {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
      {makeFlatVector<velox::StringView>({}),
       makeFlatVector<int32_t>({}),
       makeFlatVector<velox::StringView>({})});

  // Write empty vector at the beginning.
  writer->write(emptyBatch);

  // Write first non-empty vector.
  auto batch1 = makeRowVector(
      {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
      {makeFlatVector<velox::StringView>({"a", "b"}),
       makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<velox::StringView>({"c", "d"})});
  writer->write(batch1);

  // Write empty vector in the middle.
  writer->write(emptyBatch);

  // Write second non-empty vector.
  auto batch2 = makeRowVector(
      {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
      {makeFlatVector<velox::StringView>({"e", "f", "g"}),
       makeFlatVector<int32_t>({3, 4, 5}),
       makeFlatVector<velox::StringView>({"h", "i", "j"})});
  writer->write(batch2);

  // Write empty vector at the end.
  writer->write(emptyBatch);
}

TEST_F(ClusterIndexWriterTest, enforceKeyOrderInvalidWithinBatch) {
  for (bool enforceKeyOrder : {true, false}) {
    SCOPED_TRACE(fmt::format("enforceKeyOrder: {}", enforceKeyOrder));
    auto config = buildIndexConfig(
        {std::string(kCol1)},
        EncodingLayout{
            EncodingType::Trivial,
            {},
            CompressionType::Uncompressed,
            {EncodingLayout{
                EncodingType::Trivial, {}, CompressionType::Uncompressed}}},
        {},
        enforceKeyOrder,
        enforceKeyOrder);
    auto writer = createWriter(config, type_, pool_.get());
    ASSERT_NE(writer, nullptr);

    auto batch = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"a", "b", "c", "d", "e"}),
         makeFlatVector<int32_t>({5, 4, 3, 2, 1}),
         makeFlatVector<velox::StringView>({"f", "g", "h", "i", "j"})});

    if (enforceKeyOrder) {
      NIMBLE_ASSERT_USER_THROW(
          writer->write(batch),
          "Encoded keys must be in strictly ascending order (duplicates are not allowed)");
    } else {
      EXPECT_NO_THROW(writer->write(batch));
    }
  }
}

TEST_F(ClusterIndexWriterTest, enforceKeyOrderInvalidAcrossBatches) {
  for (bool enforceKeyOrder : {true, false}) {
    SCOPED_TRACE(fmt::format("enforceKeyOrder: {}", enforceKeyOrder));
    auto config = buildIndexConfig(
        {std::string(kCol1)},
        EncodingLayout{
            EncodingType::Trivial,
            {},
            CompressionType::Uncompressed,
            {EncodingLayout{
                EncodingType::Trivial, {}, CompressionType::Uncompressed}}},
        {},
        enforceKeyOrder,
        enforceKeyOrder);
    auto writer = createWriter(config, type_, pool_.get());
    ASSERT_NE(writer, nullptr);

    auto batch1 = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"a", "b", "c"}),
         makeFlatVector<int32_t>({1, 2, 3}),
         makeFlatVector<velox::StringView>({"d", "e", "f"})});
    auto batch2 = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"g", "h", "i"}),
         makeFlatVector<int32_t>({2, 3, 4}),
         makeFlatVector<velox::StringView>({"j", "k", "l"})});

    writer->write(batch1);
    if (enforceKeyOrder) {
      NIMBLE_ASSERT_USER_THROW(
          writer->write(batch2),
          "Encoded keys must be in strictly ascending order (duplicates are not allowed)");
    } else {
      EXPECT_NO_THROW(writer->write(batch2));
    }
  }
}

TEST_F(ClusterIndexWriterTest, enforceKeyOrderDuplicateKeys) {
  // When enforceKeyOrder=true and noDuplicateKey=true, duplicate keys should
  // be rejected.
  auto config = buildIndexConfig(
      {std::string(kCol1)},
      EncodingLayout{
          EncodingType::Trivial,
          {},
          CompressionType::Uncompressed,
          {EncodingLayout{
              EncodingType::Trivial, {}, CompressionType::Uncompressed}}},
      {},
      true,
      true);
  auto writer = createWriter(config, type_, pool_.get());
  ASSERT_NE(writer, nullptr);

  // Batch with duplicate keys (1, 1, 2, 3, 3)
  auto batch = makeRowVector(
      {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
      {makeFlatVector<velox::StringView>({"a", "b", "c", "d", "e"}),
       makeFlatVector<int32_t>({1, 1, 2, 3, 3}),
       makeFlatVector<velox::StringView>({"f", "g", "h", "i", "j"})});

  NIMBLE_ASSERT_USER_THROW(
      writer->write(batch),
      "Encoded keys must be in strictly ascending order (duplicates are not allowed)");
}

TEST_F(ClusterIndexWriterTest, noDuplicateKey) {
  // noDuplicateKey only affects order validation when enforceKeyOrder is set.
  {
    auto config = buildIndexConfig(
        {std::string(kCol1)},
        EncodingLayout{
            EncodingType::Trivial,
            {},
            CompressionType::Uncompressed,
            {EncodingLayout{
                EncodingType::Trivial, {}, CompressionType::Uncompressed}}},
        {},
        false,
        true);
    auto writer = createWriter(config, type_, pool_.get());
    ASSERT_NE(writer, nullptr);

    auto batch = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"a", "b", "c"}),
         makeFlatVector<int32_t>({2, 1, 1}),
         makeFlatVector<velox::StringView>({"d", "e", "f"})});
    EXPECT_NO_THROW(writer->write(batch));
  }

  // When enforceKeyOrder=true and noDuplicateKey=true, duplicate keys
  // are rejected
  {
    auto config = orderedIndexConfig(true);
    auto writer = createWriter(config, type_, pool_.get());
    ASSERT_NE(writer, nullptr);

    // Batch with duplicate keys (1, 1, 2, 3, 3) - should fail
    auto batch = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"a", "b", "c", "d", "e"}),
         makeFlatVector<int32_t>({1, 1, 2, 3, 3}),
         makeFlatVector<velox::StringView>({"f", "g", "h", "i", "j"})});

    NIMBLE_ASSERT_USER_THROW(
        writer->write(batch),
        "Encoded keys must be in strictly ascending order (duplicates are not allowed)");
  }

  // Test 4: When enforceKeyOrder=true and noDuplicateKey=true, duplicates
  // across batches are rejected
  {
    auto config = orderedIndexConfig(true);
    auto writer = createWriter(config, type_, pool_.get());
    ASSERT_NE(writer, nullptr);

    auto batch1 = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"a", "b", "c"}),
         makeFlatVector<int32_t>({1, 2, 3}),
         makeFlatVector<velox::StringView>({"d", "e", "f"})});

    // Second batch starts with same key as last key of first batch
    auto batch2 = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"g", "h", "i"}),
         makeFlatVector<int32_t>({3, 4, 5}),
         makeFlatVector<velox::StringView>({"j", "k", "l"})});

    writer->write(batch1);
    NIMBLE_ASSERT_USER_THROW(
        writer->write(batch2),
        "Encoded keys must be in strictly ascending order (duplicates are not allowed)");
  }

  // Test 5: When enforceKeyOrder=true and noDuplicateKey=true, strictly
  // ascending keys succeed
  {
    auto config = orderedIndexConfig(true);
    auto writer = createWriter(config, type_, pool_.get());
    ASSERT_NE(writer, nullptr);

    auto batch1 = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"a", "b", "c"}),
         makeFlatVector<int32_t>({1, 2, 3}),
         makeFlatVector<velox::StringView>({"d", "e", "f"})});

    auto batch2 = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"g", "h", "i"}),
         makeFlatVector<int32_t>({4, 5, 6}),
         makeFlatVector<velox::StringView>({"j", "k", "l"})});

    EXPECT_NO_THROW(writer->write(batch1));
    EXPECT_NO_THROW(writer->write(batch2));
  }

  // Test 6: When enforceKeyOrder=true but noDuplicateKey=false, duplicates are
  // allowed
  {
    auto config = orderedIndexConfig(false);
    auto writer = createWriter(config, type_, pool_.get());
    ASSERT_NE(writer, nullptr);

    // Batch with duplicate keys (1, 1, 2, 3, 3) - should succeed
    auto batch = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"a", "b", "c", "d", "e"}),
         makeFlatVector<int32_t>({1, 1, 2, 3, 3}),
         makeFlatVector<velox::StringView>({"f", "g", "h", "i", "j"})});

    EXPECT_NO_THROW(writer->write(batch));
  }

  // Test 7: When enforceKeyOrder=true but noDuplicateKey=false, out-of-order
  // keys are still rejected
  {
    auto config = orderedIndexConfig(false);
    auto writer = createWriter(config, type_, pool_.get());
    ASSERT_NE(writer, nullptr);

    // Batch with out-of-order keys (5, 4, 3, 2, 1) - should fail
    auto batch = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>({"a", "b", "c", "d", "e"}),
         makeFlatVector<int32_t>({5, 4, 3, 2, 1}),
         makeFlatVector<velox::StringView>({"f", "g", "h", "i", "j"})});

    NIMBLE_ASSERT_USER_THROW(
        writer->write(batch), "Encoded keys must be in ascending order");
  }
}

struct ClusterIndexWriterDataTestParam {
  EncodingType encodingType;
  SortOrder sortOrder;
  std::optional<uint32_t> prefixRestartInterval;

  std::string toString() const {
    const std::string encodingName =
        encodingType == EncodingType::Prefix ? "Prefix" : "Trivial";
    const std::string sortName = sortOrder.ascending ? "Asc" : "Desc";
    std::string result = encodingName + "_" + sortName;
    if (prefixRestartInterval.has_value()) {
      result += "_restart" + std::to_string(prefixRestartInterval.value());
    }
    return result;
  }
};

class ClusterIndexWriterDataTest
    : public ClusterIndexWriterTest,
      public ::testing::WithParamInterface<ClusterIndexWriterDataTestParam> {
 protected:
  void SetUp() override {
    ClusterIndexWriterTest::SetUp();
    velox::VectorFuzzer::Options opts;
    opts.nullRatio = 0;
    fuzzer_ = std::make_unique<velox::VectorFuzzer>(opts, pool_.get());
  }

  void TearDown() override {
    fuzzer_ = nullptr;
    ClusterIndexWriterTest::TearDown();
  }

  EncodingType encodingType() const {
    return GetParam().encodingType;
  }

  SortOrder sortOrder() const {
    return GetParam().sortOrder;
  }

  std::unique_ptr<velox::serializer::KeyEncoder> createKeyEncoderWithSortOrder()
      const {
    const auto keyType = velox::ROW({{std::string(kCol1), velox::INTEGER()}});
    return velox::serializer::KeyEncoder::create(
        {std::string(kCol1)},
        keyType,
        {sortOrder().toVeloxSortOrder()},
        pool_.get());
  }

  std::shared_ptr<const IndexConfig> indexConfigWithSortOrder(
      EncodingType encodingType = EncodingType::Prefix) const {
    EncodingLayout layout{
        EncodingType::Trivial, {}, CompressionType::Uncompressed};
    if (encodingType == EncodingType::Prefix) {
      EncodingLayout::Config config;
      if (GetParam().prefixRestartInterval.has_value()) {
        config = EncodingLayout::Config{
            {{std::string(PrefixEncoding::kRestartIntervalConfigKey),
              std::to_string(GetParam().prefixRestartInterval.value())}}};
      }
      layout = EncodingLayout{
          EncodingType::Prefix, std::move(config), CompressionType::Zstd};
    } else {
      layout = EncodingLayout{
          EncodingType::Trivial,
          {},
          CompressionType::Zstd,
          {EncodingLayout{
              EncodingType::Trivial, {}, CompressionType::Uncompressed}}};
    }
    return buildIndexConfig(
        {std::string(kCol1)}, std::move(layout), {sortOrder()});
  }

  // Creates a row vector with the given index column values and
  // fuzzer-generated non-index columns.
  velox::RowVectorPtr makeInput(const std::vector<int32_t>& indexValues) {
    auto opts = fuzzer_->getOptions();
    opts.vectorSize = indexValues.size();
    fuzzer_->setOptions(opts);
    auto fuzzed = fuzzer_->fuzzInputFlatRow(type_);
    return makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {fuzzed->childAt(0),
         makeFlatVector<int32_t>(indexValues),
         fuzzed->childAt(2)});
  }

  // Creates a row vector containing only the key column for KeyEncoder.
  velox::RowVectorPtr makeKeyInput(int32_t keyValue) {
    return makeRowVector(
        {std::string(kCol1)}, {makeFlatVector<int32_t>({keyValue})});
  }

  std::unique_ptr<velox::VectorFuzzer> fuzzer_;
};

TEST_P(ClusterIndexWriterDataTest, writeAndFinishStripeNonChunked) {
  auto config = indexConfigWithSortOrder(encodingType());
  auto keyEncoder = createKeyEncoderWithSortOrder();

  struct {
    std::vector<std::vector<int32_t>> batches;

    std::string debugString() const {
      std::vector<std::string> batchStrs;
      batchStrs.reserve(batches.size());
      for (const auto& batch : batches) {
        batchStrs.push_back(fmt::format("[{}]", fmt::join(batch, ", ")));
      }
      return fmt::format("batches: [{}]", fmt::join(batchStrs, ", "));
    }

    size_t totalRows() const {
      size_t total = 0;
      for (const auto& batch : batches) {
        total += batch.size();
      }
      return total;
    }
  } testCases[] = {
      // Single batch, single row
      {{{1}}},
      // Single batch, multiple rows
      {{{1, 2, 3, 4, 5}}},
      // Two batches
      {{{1, 2, 3}, {4, 5, 6}}},
      // Three batches
      {{{1, 2}, {3, 4}, {5, 6}}},
      // Multiple batches with varying sizes
      {{{1}, {2, 3}, {4, 5, 6}, {7, 8, 9, 10}}},
      // Large single batch
      {{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}},
      // Many small batches
      {{{1}, {2}, {3}, {4}, {5}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    auto writer = createWriter(config, type_, pool_.get());
    ASSERT_NE(writer, nullptr);

    for (const auto& batchData : testCase.batches) {
      writer->write(makeInput(batchData));
    }
  }
}

TEST_P(ClusterIndexWriterDataTest, writeAndNewStripeMultiBatch) {
  auto config = indexConfig(encodingType());

  struct {
    std::vector<std::vector<int32_t>> batches;

    std::string debugString() const {
      std::vector<std::string> batchStrs;
      batchStrs.reserve(batches.size());
      for (const auto& batch : batches) {
        batchStrs.push_back(fmt::format("[{}]", fmt::join(batch, ", ")));
      }
      return fmt::format("batches: [{}]", fmt::join(batchStrs, ", "));
    }
  } testCases[] = {
      // Single batch with multiple rows.
      {{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}},
      // Multiple batches.
      {{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}},
      // Many small batches.
      {{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    auto writer = createWriter(config, type_, pool_.get());
    ASSERT_NE(writer, nullptr);

    for (const auto& batchData : testCase.batches) {
      writer->write(makeInput(batchData));
    }
  }
}

TEST_P(ClusterIndexWriterDataTest, multiColumnIndex) {
  auto type = velox::ROW({
      {"col0", velox::INTEGER()},
      {"col1", velox::VARCHAR()},
  });
  EncodingLayout layout = encodingType() == EncodingType::Prefix
      ? EncodingLayout{EncodingType::Prefix, {}, CompressionType::Zstd}
      : EncodingLayout{
            EncodingType::Trivial,
            {},
            CompressionType::Zstd,
            {EncodingLayout{
                EncodingType::Trivial, {}, CompressionType::Uncompressed}}};
  auto config = buildIndexConfig({"col0", "col1"}, std::move(layout));
  auto writer = createWriter(config, type, pool_.get());

  auto batch = makeRowVector(
      {"col0", "col1"},
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<velox::StringView>({"a", "b", "c"})});

  writer->write(batch);
}

TEST_P(ClusterIndexWriterDataTest, close) {
  auto config = indexConfig(encodingType());
  auto writer = createWriter(config, type_, pool_.get());
  ASSERT_NE(writer, nullptr);

  writer->write(makeInput({1, 2, 3}));
  uint32_t nextSectionId = 0;
  CreateMetadataSectionFn createMetadataFn = [&](std::string_view) {
    return MetadataSection{nextSectionId++, 0, CompressionType::Uncompressed};
  };
  WriteDataFn noopWriteDataFn = [](const std::vector<std::string_view>&)
      -> std::pair<uint64_t, uint32_t> { return {0, 0}; };
  EXPECT_FALSE(writer->close(noopWriteDataFn, createMetadataFn).has_value());
}

TEST_P(ClusterIndexWriterDataTest, writeAfterClose) {
  auto config = indexConfig(encodingType());
  auto writer = createWriter(config, type_, pool_.get());
  ASSERT_NE(writer, nullptr);

  writer->write(makeInput({1, 2, 3}));
  uint32_t nextSectionId = 0;
  CreateMetadataSectionFn createMetadataFn = [&](std::string_view) {
    return MetadataSection{nextSectionId++, 0, CompressionType::Uncompressed};
  };
  WriteDataFn noopWriteDataFn = [](const std::vector<std::string_view>&)
      -> std::pair<uint64_t, uint32_t> { return {0, 0}; };
  EXPECT_FALSE(writer->close(noopWriteDataFn, createMetadataFn).has_value());

  NIMBLE_ASSERT_THROW(
      writer->write(makeInput({4, 5, 6})), "IndexWriter has been closed");
  NIMBLE_ASSERT_THROW(
      writer->close(noopWriteDataFn, createMetadataFn),
      "close() already called");
}

INSTANTIATE_TEST_SUITE_P(
    ClusterIndexWriterDataTestSuite,
    ClusterIndexWriterDataTest,
    ::testing::Values(
        ClusterIndexWriterDataTestParam{
            EncodingType::Prefix,
            SortOrder{.ascending = true},
            std::nullopt},
        ClusterIndexWriterDataTestParam{
            EncodingType::Prefix,
            SortOrder{.ascending = false},
            std::nullopt},
        ClusterIndexWriterDataTestParam{
            EncodingType::Trivial,
            SortOrder{.ascending = true},
            std::nullopt},
        ClusterIndexWriterDataTestParam{
            EncodingType::Trivial,
            SortOrder{.ascending = false},
            std::nullopt},
        ClusterIndexWriterDataTestParam{
            EncodingType::Prefix,
            SortOrder{.ascending = false},
            1},
        ClusterIndexWriterDataTestParam{
            EncodingType::Prefix,
            SortOrder{.ascending = false},
            1024}),
    [](const ::testing::TestParamInfo<ClusterIndexWriterDataTestParam>& info) {
      return info.param.toString();
    });

TEST_F(ClusterIndexWriterTest, enforceKeyOrderWithEmptyKeys) {
  // Empty strings are valid encoded key values. Verify ordering is enforced
  // correctly when keys include empty strings.
  auto config = buildIndexConfig(
      {std::string(kCol0)},
      EncodingLayout{
          EncodingType::Trivial,
          {},
          CompressionType::Uncompressed,
          {EncodingLayout{
              EncodingType::Trivial, {}, CompressionType::Uncompressed}}},
      {},
      true);
  auto writer = createWriter(config, type_, pool_.get());
  ASSERT_NE(writer, nullptr);

  // Batch with empty string key followed by non-empty keys.
  auto batch1 = makeRowVector(
      {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
      {makeFlatVector<velox::StringView>({"", "a", "b"}),
       makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<velox::StringView>({"x", "y", "z"})});
  EXPECT_NO_THROW(writer->write(batch1));

  // Second batch starting with empty string should fail (out of order
  // relative to last key "b" from previous batch).
  auto batch2 = makeRowVector(
      {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
      {makeFlatVector<velox::StringView>({"", "c", "d"}),
       makeFlatVector<int32_t>({4, 5, 6}),
       makeFlatVector<velox::StringView>({"u", "v", "w"})});
  NIMBLE_ASSERT_USER_THROW(
      writer->write(batch2), "Encoded keys must be in ascending order");
}

TEST_F(ClusterIndexWriterTest, enforceKeyOrderAllEmptyKeys) {
  // All keys are empty strings — should be accepted when duplicates are
  // allowed.
  auto config = buildIndexConfig(
      {std::string(kCol0)},
      EncodingLayout{
          EncodingType::Trivial,
          {},
          CompressionType::Uncompressed,
          {EncodingLayout{
              EncodingType::Trivial, {}, CompressionType::Uncompressed}}},
      {},
      true);
  auto writer = createWriter(config, type_, pool_.get());
  ASSERT_NE(writer, nullptr);

  auto batch = makeRowVector(
      {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
      {makeFlatVector<velox::StringView>({"", "", ""}),
       makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<velox::StringView>({"x", "y", "z"})});
  EXPECT_NO_THROW(writer->write(batch));

  // Second batch with all empty keys — should work across batch boundary
  // since lastEncodedKey_ is empty string and duplicates are allowed.
  EXPECT_NO_THROW(writer->write(batch));
}

class ClusterIndexWriterChunkTest
    : public ClusterIndexWriterTest,
      public ::testing::WithParamInterface<uint64_t> {};

TEST_P(ClusterIndexWriterChunkTest, maxRowsPerKeyChunk) {
  const uint64_t maxRowsPerKeyChunk = GetParam();
  constexpr uint32_t kNumRows = 100;

  auto config = buildIndexConfig(
      {std::string(kCol1)},
      EncodingLayout{EncodingType::Prefix, {}, CompressionType::Uncompressed},
      {SortOrder{.ascending = true}},
      true,
      false,
      maxRowsPerKeyChunk);
  auto writer = createWriter(config, type_, pool_.get());
  ASSERT_NE(writer, nullptr);

  // Write kNumRows sorted integer keys.
  std::vector<std::string> col0Strs(kNumRows);
  std::vector<std::string> col2Strs(kNumRows);
  std::vector<velox::StringView> col0Vals(kNumRows);
  std::vector<int32_t> col1Vals(kNumRows);
  std::vector<velox::StringView> col2Vals(kNumRows);
  for (uint32_t i = 0; i < kNumRows; ++i) {
    col0Strs[i] = fmt::format("s{:04d}", i);
    col0Vals[i] = velox::StringView(col0Strs[i]);
    col1Vals[i] = static_cast<int32_t>(i);
    col2Strs[i] = fmt::format("v{:04d}", i);
    col2Vals[i] = velox::StringView(col2Strs[i]);
  }
  auto batch = makeRowVector(
      {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
      {makeFlatVector<velox::StringView>(col0Vals),
       makeFlatVector<int32_t>(col1Vals),
       makeFlatVector<velox::StringView>(col2Vals)});
  writer->write(batch);

  // Flush partition: captures key stream data and metadata.
  std::string keyStreamData;
  std::vector<std::string> partitionMetadata;
  auto writeDataFn =
      [&keyStreamData](const std::vector<std::string_view>& segments)
      -> std::pair<uint64_t, uint32_t> {
    const uint64_t offset = keyStreamData.size();
    uint32_t totalSize = 0;
    for (const auto& segment : segments) {
      keyStreamData.append(segment);
      totalSize += segment.size();
    }
    return {offset, totalSize};
  };
  auto createMetadataFn =
      [&partitionMetadata](std::string_view metadata) -> MetadataSection {
    partitionMetadata.emplace_back(metadata);
    return MetadataSection(
        0,
        metadata.size(),
        CompressionType::Uncompressed,
        static_cast<uint32_t>(metadata.size()));
  };
  writer->flush(writeDataFn, createMetadataFn);

  // Close the writer and capture the root index payload.
  WriteDataFn noopWriteDataFn2 = [](const std::vector<std::string_view>&)
      -> std::pair<uint64_t, uint32_t> { return {0, 0}; };
  auto descriptor = writer->close(noopWriteDataFn2, createMetadataFn);
  ASSERT_TRUE(descriptor.has_value());
  const auto& rootIndexData = partitionMetadata.back();

  ASSERT_FALSE(rootIndexData.empty());
  ASSERT_EQ(partitionMetadata.size(), 2);

  // Load ClusterIndex from serialized data and verify layout.
  Section rootSection{MetadataBuffer(
      MetadataBuffer::decompress(
          toBufferPtr(rootIndexData, pool_.get()),
          CompressionType::Uncompressed,
          pool_.get()))};
  auto metadataFile =
      std::make_shared<velox::InMemoryReadFile>(partitionMetadata[0]);
  auto ioStats = std::make_shared<velox::io::IoStatistics>();
  auto metadataInput = MetadataInput::create(
      metadataFile.get(),
      MetadataInput::Options{.pool = pool_.get(), .ioStats = ioStats});
  auto dataInput = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(keyStreamData), *pool_);

  auto clusterIndex = test::ClusterIndexTestHelper::create(
      std::move(rootSection),
      pool_.get(),
      std::move(metadataInput),
      std::move(dataInput));

  // Verify partition count and layout.
  EXPECT_EQ(clusterIndex->numPartitions(), 1);

  const auto layout = clusterIndex->layout(/*detail=*/true);
  EXPECT_EQ(layout.numPartitions, 1);
  EXPECT_EQ(layout.indexColumns.size(), 1);
  EXPECT_EQ(layout.indexColumns[0], std::string(kCol1));
  ASSERT_EQ(layout.partitions.size(), 1);

  const auto& partition = layout.partitions[0];
  EXPECT_EQ(partition.numRows, kNumRows);
  EXPECT_GT(partition.metadataSizeBytes, 0);
  EXPECT_GT(partition.keyStreamRegion.length, 0);

  // Verify chunk count. encodeKeyChunks absorbs tail when remaining < 2 *
  // maxRows, so expected chunks = floor(numRows / maxRows) when tail is
  // small enough to absorb.
  const uint32_t expectedChunks = maxRowsPerKeyChunk == 0
      ? 1
      : std::max(1u, static_cast<uint32_t>(kNumRows / maxRowsPerKeyChunk));
  EXPECT_EQ(partition.numChunks, expectedChunks)
      << "maxRowsPerKeyChunk=" << maxRowsPerKeyChunk;

  // Verify lookups using minKey/maxKey from the index.
  velox::serializer::EncodedKeyBounds firstBounds{
      .lowerKey = std::string(clusterIndex->minKey()),
      .upperKey = std::nullopt};
  auto firstResult = clusterIndex->lookup(
      IndexLookup::LookupRequest::rangeScan({firstBounds}));
  ASSERT_EQ(firstResult.size(), 1);
  ASSERT_EQ(firstResult[0].size(), 1);
  EXPECT_EQ(firstResult[0][0].startRow, 0);

  velox::serializer::EncodedKeyBounds lastBounds{
      .lowerKey = std::string(clusterIndex->maxKey()),
      .upperKey = std::nullopt};
  auto lastResult =
      clusterIndex->lookup(IndexLookup::LookupRequest::rangeScan({lastBounds}));
  ASSERT_EQ(lastResult.size(), 1);
  ASSERT_EQ(lastResult[0].size(), 1);
  EXPECT_EQ(lastResult[0][0].startRow, kNumRows - 1);
  EXPECT_EQ(lastResult[0][0].endRow, kNumRows);
}

INSTANTIATE_TEST_SUITE_P(
    ChunkGranularity,
    ClusterIndexWriterChunkTest,
    ::testing::Values(0, 1, 32, 1'024),
    [](const ::testing::TestParamInfo<uint64_t>& info) {
      return fmt::format("maxRowsPerKeyChunk_{}", info.param);
    });

} // namespace
} // namespace facebook::nimble::index::test
