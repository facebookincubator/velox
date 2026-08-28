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

#include "velox/dwio/nimble/index/NimbleIndexProjector.h"

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/SchemaBuilder.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/SchemaUtils.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/reader/BatchReader.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletReaderCache.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/tests/SchemaUtils.h"
#include "velox/dwio/nimble/writer/Writer.h"

#include "folly/Random.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/file/IoUringReader.h"
#include "velox/common/file/LocalFile.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/serializers/KeyEncoder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble::test {

using namespace velox;
using namespace velox::dwio::common;
using index::ClusterIndexConfigBuilder;
using Subfield = velox::common::Subfield;

namespace {

// Test-local mirrors of `rocks::readResultRowRange` and
// `rocks::readResultResumeKey`.
// The dwio/nimble layer cannot depend on rocks/nimble (the dependency runs
// the other way — rocks/nimble's NimbleTable depends on the projector), so
// these tests cannot include `rocks/nimble/NimbleTable.h`. Both helpers call
// the same underlying `serde::readTabletChunkHeader` as the production rocks::
// helpers, so behavior is in lockstep.
serde::TabletChunkHeader readEmbeddedTabletChunkHeader(
    const folly::IOBuf& slice) {
  const char* pos = reinterpret_cast<const char*>(slice.data());
  const char* end = pos + slice.length();
  return serde::readTabletChunkHeader(pos, end);
}

RowRange readEmbeddedRowRange(const folly::IOBuf& slice) {
  return readEmbeddedTabletChunkHeader(slice).rowRange;
}

std::optional<std::string> readEmbeddedResumeKey(const folly::IOBuf& slice) {
  return readEmbeddedTabletChunkHeader(slice).resumeKey;
}

// Coalesces the chunk slice IOBuf chain into a single contiguous buffer for
// passing to Deserializer. The Deserializer reads the per-slice header
// natively (consuming the row range and resume-key fields) and honors the
// row range at decode time via skip+next, so the downstream deserialized
// vector contains only the rows in `[rowRange.startRow, rowRange.endRow)`
// — no over-fetch.
folly::IOBuf coalesceChunkSlice(const folly::IOBuf& slice) {
  return slice.cloneCoalescedAsValue();
}

} // namespace

struct TestParam {
  bool cacheMetadata;
  bool pinMetadata;

  std::string debugString() const {
    return fmt::format(
        "cacheMetadata={}, pinMetadata={}", cacheMetadata, pinMetadata);
  }
};

TEST(NimbleTypeProjectionTest, buildsScalarProjection) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({
          {"id", NIMBLE_BIGINT()},
          {"value", NIMBLE_INTEGER()},
      }));
  auto nimbleSchema = SchemaReader::getSchema(schemaBuilder.schemaNodes());

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("value");
  auto projection =
      buildProjectedNimbleType(nimbleSchema.get(), projectedSubfields);

  ASSERT_NE(projection.nimbleType, nullptr);
  ASSERT_EQ(projection.nimbleType->kind(), Kind::Row);
  const auto& projectedRow = projection.nimbleType->asRow();
  ASSERT_EQ(projectedRow.childrenCount(), 1);
  EXPECT_EQ(projectedRow.nameAt(0), "value");
  EXPECT_EQ(projectedRow.childAt(0)->kind(), Kind::Scalar);
  EXPECT_EQ(projection.streamOffsets, std::vector<uint32_t>({2, 1}));
  EXPECT_EQ(
      projection.rowOrFlatMapNullStreams, std::vector<bool>({true, false}));
}

TEST(NimbleTypeProjectionTest, buildsFlatMapProjectionWithMissingKey) {
  SchemaBuilder schemaBuilder;
  test::FlatMapChildAdder featuresAdder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({
          {"features", NIMBLE_FLATMAP(String, NIMBLE_INTEGER(), featuresAdder)},
      }));
  featuresAdder.addChild("a");
  featuresAdder.addChild("c");
  auto nimbleSchema = SchemaReader::getSchema(schemaBuilder.schemaNodes());

  std::vector<Subfield> projectedSubfields;
  projectedSubfields.emplace_back("features[\"a\"]");
  projectedSubfields.emplace_back("features[\"missing\"]");
  auto projection =
      buildProjectedNimbleType(nimbleSchema.get(), projectedSubfields);

  ASSERT_NE(projection.nimbleType, nullptr);
  ASSERT_EQ(projection.nimbleType->kind(), Kind::Row);
  const auto& projectedRow = projection.nimbleType->asRow();
  ASSERT_EQ(projectedRow.childrenCount(), 1);
  ASSERT_EQ(projectedRow.childAt(0)->kind(), Kind::FlatMap);
  const auto& projectedFlatMap = projectedRow.childAt(0)->asFlatMap();
  ASSERT_EQ(projectedFlatMap.childrenCount(), 2);
  EXPECT_EQ(projectedFlatMap.nameAt(0), "a");
  EXPECT_EQ(projectedFlatMap.nameAt(1), "missing");
  EXPECT_EQ(
      projection.streamOffsets,
      std::vector<uint32_t>({1, 0, 2, 3, UINT32_MAX, UINT32_MAX}));
  EXPECT_EQ(
      projection.rowOrFlatMapNullStreams,
      std::vector<bool>({true, true, false, false, false, false}));
}

TEST(NimbleTypeProjectionTest, rejectsEmptyProjection) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({
          {"value", NIMBLE_INTEGER()},
      }));
  auto nimbleSchema = SchemaReader::getSchema(schemaBuilder.schemaNodes());

  const std::vector<Subfield> projectedSubfields;
  NIMBLE_ASSERT_THROW(
      buildProjectedNimbleType(nimbleSchema.get(), projectedSubfields),
      "projectedSubfields must not be empty");
}

class NimbleIndexProjectorTest : public ::testing::TestWithParam<TestParam> {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    rootPool_ =
        memory::memoryManager()->addRootPool("NimbleIndexProjectorTest");
    leafPool_ = rootPool_->addLeafChild("leaf");
    vectorMaker_ = std::make_unique<velox::test::VectorMaker>(leafPool_.get());
    ioExecutor_ = std::make_shared<folly::CPUThreadPoolExecutor>(4);
  }

  void TearDown() override {
    vectorMaker_.reset();
    leafPool_.reset();
    rootPool_.reset();
    ioExecutor_.reset();
    TabletReaderCache::testingReset();
  }

  // Writes sorted data with an index on the key column.
  void writeData(
      const std::vector<RowVectorPtr>& batches,
      const std::vector<std::string>& indexColumns,
      const folly::F14FastMap<std::string, std::set<std::string>>&
          flatMapColumns = {},
      uint64_t stripeSize = 4 << 10 /* 4KB */,
      bool enableStreamDeduplication = true,
      bool compactRowCountEncoding = false,
      CompressionType chunkCompressionType = CompressionType::Uncompressed,
      bool skipConstantFlatMapInMapStreams = false,
      bool enableChunking = true) {
    sinkData_.clear();
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);

    WriterOptions options;
    options.enableChunking = enableChunking;
    options.flatMapColumns = flatMapColumns;
    options.enableStreamDeduplication = enableStreamDeduplication;
    options.experimentalCompactRowCountEncoding = compactRowCountEncoding;
    options.chunkCompression = {
        .type = chunkCompressionType,
        // Retain compressed chunks when a small fuzzed stream grows slightly.
        .acceptRatio = 10.0f,
    };
    options.skipConstantFlatMapInMapStreams = skipConstantFlatMapInMapStreams;
    options.clusterIndexConfig =
        ClusterIndexConfigBuilder{}
            .withKeyColumns(indexColumns)
            .withSortOrders(
                std::vector<SortOrder>(
                    indexColumns.size(), SortOrder{.ascending = true}))
            .withEnforceKeyOrder(true)
            .withNoDuplicateKey(true)
            .build();

    options.flushPolicyFactory = [stripeSize]() {
      return std::make_unique<LambdaFlushPolicy>(
          /*shouldFlush*/
          [stripeSize](const StripeProgress& progress) {
            return progress.stripeRawSize >= stripeSize;
          });
    };
    auto rowType = asRowType(batches[0]->type());
    Writer writer(
        rowType, std::move(writeFile), *rootPool_, std::move(options));
    for (const auto& batch : batches) {
      writer.write(batch);
    }
    writer.close();
    TabletReaderCache::testingReset();
    tabletReaderCache_.reset();
    keyEncoder_.reset();
  }

 public:
  static std::vector<TestParam> getTestParams() {
    std::vector<TestParam> params;
    for (bool cacheMetadata : {false, true}) {
      for (bool pinMetadata : {false, true}) {
        params.push_back({cacheMetadata, pinMetadata});
      }
    }
    return params;
  }

 protected:
  velox::FileHandle makeFileHandle() {
    auto readFile =
        std::make_shared<InMemoryReadFile>(std::string_view(sinkData_));
    return makeFileHandle(std::move(readFile), "test_file");
  }

  velox::FileHandle makeFileHandle(
      std::shared_ptr<velox::ReadFile> readFile,
      std::string_view id) {
    return velox::FileHandle{
        std::move(readFile),
        velox::StringIdLease(velox::fileIds(), id),
        velox::StringIdLease(velox::fileIds(), id)};
  }

  std::unique_ptr<NimbleIndexProjector> createProjector(
      const std::vector<Subfield>& projectedSubfields,
      bool setDataIoStats = true,
      bool setMetadataIoStats = true,
      bool setIndexIoStats = true) {
    return createProjectorWithFileHandle(
        projectedSubfields,
        makeFileHandle(),
        setDataIoStats,
        setMetadataIoStats,
        setIndexIoStats);
  }

  std::unique_ptr<NimbleIndexProjector> createProjectorWithFileHandle(
      const std::vector<Subfield>& projectedSubfields,
      velox::FileHandle fileHandle,
      bool setDataIoStats = true,
      bool setMetadataIoStats = true,
      bool setIndexIoStats = true) {
    dataIoStats_ = std::make_shared<velox::io::IoStatistics>();
    metadataIoStats_ = std::make_shared<velox::io::IoStatistics>();
    indexIoStats_ = std::make_shared<velox::io::IoStatistics>();
    dwio::common::ReaderOptions readerOptions(leafPool_.get());
    if (setDataIoStats) {
      readerOptions.setDataIoStats(dataIoStats_);
    }
    if (setMetadataIoStats) {
      readerOptions.setMetadataIoStats(metadataIoStats_);
    }
    if (setIndexIoStats) {
      readerOptions.setIndexIoStats(indexIoStats_);
    }
    readerOptions.setFileFormat(FileFormat::NIMBLE);
    readerOptions.setLoadClusterIndex(true);
    readerOptions.setCacheData(false);
    readerOptions.setCacheMetadata(GetParam().cacheMetadata);
    readerOptions.setPinMetadata(GetParam().pinMetadata);
    readerOptions.setFilePreloadThreshold(0);
    readerOptions.setIOExecutor(ioExecutor_);
    ensureTabletReaderCache();
    return NimbleIndexProjector::create(
        *tabletReaderCache_, fileHandle, projectedSubfields, readerOptions);
  }

  VectorPtr deserializeProjectedBatches(
      const NimbleIndexProjector& projector,
      const std::vector<std::string_view>& batches) {
    DeserializerOptions deserOptions;
    deserOptions.hasHeader = true;
    Deserializer deserializer(
        projector.projectedNimbleType(), leafPool_.get(), deserOptions);
    VectorPtr output;
    deserializer.deserialize(batches, output);
    return output;
  }

  VectorPtr deserializeProjectedSlice(
      const NimbleIndexProjector& projector,
      const folly::IOBuf& slice) {
    auto coalesced = coalesceChunkSlice(slice);
    std::vector<std::string_view> batches{std::string_view(
        reinterpret_cast<const char*>(coalesced.data()), coalesced.length())};
    return deserializeProjectedBatches(projector, batches);
  }

  // Stripe rows the `makeWindowedSlice` key bounds select.
  static constexpr uint32_t kWindowStart = 30;
  static constexpr uint32_t kWindowEnd = 60;
  static constexpr uint32_t kWindowRows = kWindowEnd - kWindowStart;

  // A projected slice whose header window is [kWindowStart, kWindowEnd).
  // Owns the projector and the projector result because the slice borrows
  // from both.
  struct WindowedSlice {
    std::unique_ptr<NimbleIndexProjector> projector;
    NimbleIndexProjector::Result result;
    // The full stripe's `value` column, for building expectations.
    std::vector<int32_t> values;

    const folly::IOBuf& slice() const {
      return result.responses[0].slices[0];
    }
  };

  // Writes a 100-row stripe keyed on `key`, projects `value`, and returns
  // the single slice the key bounds window to [kWindowStart, kWindowEnd).
  WindowedSlice makeWindowedSlice() {
    auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
    constexpr int kStripeRows = 100;
    std::vector<int64_t> keys(kStripeRows);
    std::vector<int32_t> values(kStripeRows);
    for (int i = 0; i < kStripeRows; ++i) {
      keys[i] = i;
      values[i] = i * 10;
    }
    auto batch = vectorMaker_->rowVector(
        {"key", "value"},
        {vectorMaker_->flatVector<int64_t>(keys),
         vectorMaker_->flatVector<int32_t>(values)});
    writeData({batch}, {"key"});
    std::vector<Subfield> subfields;
    subfields.emplace_back("value");
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, kWindowStart, kWindowEnd);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    auto result = projector->project(request, {});
    return {std::move(projector), std::move(result), std::move(values)};
  }

  // Asserts the slice really carries the expected header window, so the
  // composition tests cannot pass vacuously on an unwindowed slice.
  void expectWindow(const WindowedSlice& windowed) {
    ASSERT_EQ(windowed.result.responses[0].slices.size(), 1);
    const auto window = readEmbeddedRowRange(windowed.slice());
    ASSERT_EQ(window.startRow, kWindowStart);
    ASSERT_EQ(window.endRow, kWindowEnd);
  }

  // Same, but narrows within the slice's header window using the per-batch
  // rowRanges overload. `rowRanges` are relative to that window.
  VectorPtr deserializeProjectedSlice(
      const NimbleIndexProjector& projector,
      const folly::IOBuf& slice,
      const std::vector<RowRange>& rowRanges) {
    auto coalesced = coalesceChunkSlice(slice);
    std::vector<std::string_view> batches{std::string_view(
        reinterpret_cast<const char*>(coalesced.data()), coalesced.length())};
    DeserializerOptions deserOptions;
    deserOptions.hasHeader = true;
    Deserializer deserializer(
        projector.projectedNimbleType(), leafPool_.get(), deserOptions);
    VectorPtr output;
    deserializer.deserialize(batches, rowRanges, output);
    return output;
  }

  // Compares decoded output rows with an expected vector. The Deserializer
  // honors the header's row range via skip+next at decode time, so output
  // contains exactly the requested rows starting at index 0.
  // `assertEqualVectors` internally asserts equal size.
  void expectVectorRows(const VectorPtr& output, const VectorPtr& expected) {
    ASSERT_NE(output, nullptr);
    ASSERT_NE(expected, nullptr);
    velox::test::assertEqualVectors(expected, output);
  }

  // Creates encoded key bounds for a point lookup on a single int64 key.
  velox::serializer::EncodedKeyBounds makePointLookup(
      const RowTypePtr& keyType,
      const std::vector<std::string>& indexColumns,
      int64_t key) {
    if (keyEncoder_ == nullptr) {
      std::vector<velox::core::SortOrder> sortOrders(
          indexColumns.size(), velox::core::SortOrder{true, false});
      keyEncoder_ = velox::serializer::KeyEncoder::create(
          indexColumns, keyType, sortOrders, leafPool_.get());
    }

    auto keyVector = vectorMaker_->rowVector(
        indexColumns, {vectorMaker_->flatVector<int64_t>({key})});

    velox::serializer::IndexBounds bounds;
    bounds.indexColumns = indexColumns;
    bounds.set(
        velox::serializer::IndexBound{keyVector, true},
        velox::serializer::IndexBound{keyVector, true});

    auto encoded = keyEncoder_->encodeIndexBounds(bounds);
    EXPECT_EQ(encoded.size(), 1);
    return encoded[0];
  }

  // Creates encoded key bounds for a range scan on a single int64 key.
  // The range is [lowerKey, upperKey) (lower inclusive, upper exclusive).
  velox::serializer::EncodedKeyBounds makeRangeLookup(
      const RowTypePtr& keyType,
      const std::vector<std::string>& indexColumns,
      int64_t lowerKey,
      int64_t upperKey) {
    if (keyEncoder_ == nullptr) {
      std::vector<velox::core::SortOrder> sortOrders(
          indexColumns.size(), velox::core::SortOrder{true, false});
      keyEncoder_ = velox::serializer::KeyEncoder::create(
          indexColumns, keyType, sortOrders, leafPool_.get());
    }

    auto lowerVector = vectorMaker_->rowVector(
        indexColumns, {vectorMaker_->flatVector<int64_t>({lowerKey})});
    auto upperVector = vectorMaker_->rowVector(
        indexColumns, {vectorMaker_->flatVector<int64_t>({upperKey})});

    velox::serializer::IndexBounds bounds;
    bounds.indexColumns = indexColumns;
    bounds.set(
        velox::serializer::IndexBound{lowerVector, true},
        velox::serializer::IndexBound{upperVector, false});

    auto encoded = keyEncoder_->encodeIndexBounds(bounds);
    EXPECT_EQ(encoded.size(), 1);
    return encoded[0];
  }

  // Writes deterministic data with keys 0..numBatches*rowsPerBatch-1 across
  // multiple stripes (one stripe per batch).
  void writeResumeKeyTestData(int rowsPerBatch = 100, int numBatches = 10) {
    std::vector<RowVectorPtr> batches;
    for (int b = 0; b < numBatches; ++b) {
      std::vector<int64_t> keys(rowsPerBatch);
      std::vector<int32_t> values(rowsPerBatch);
      for (int i = 0; i < rowsPerBatch; ++i) {
        keys[i] = b * rowsPerBatch + i;
        values[i] = keys[i] * 10;
      }
      batches.emplace_back(vectorMaker_->rowVector(
          {"key", "value"},
          {vectorMaker_->flatVector<int64_t>(keys),
           vectorMaker_->flatVector<int32_t>(values)}));
    }
    // stripeSize=1 forces each batch into its own stripe.
    writeData(batches, {"key"}, {}, /*stripeSize=*/1);
  }

  std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<io::IoStatistics> indexIoStats_{
      std::make_shared<io::IoStatistics>()};

  void ensureTabletReaderCache() {
    if (tabletReaderCache_ == nullptr) {
      TabletReaderCache::Options opts;
      opts.numShards = 2;
      opts.maxEntries = 100;
      opts.executor = std::make_shared<folly::CPUThreadPoolExecutor>(4);
      tabletReaderCache_ = std::make_unique<TabletReaderCache>(opts);
    }
  }

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
  std::shared_ptr<folly::CPUThreadPoolExecutor> ioExecutor_;
  std::string sinkData_;
  std::unique_ptr<velox::serializer::KeyEncoder> keyEncoder_;
  std::unique_ptr<TabletReaderCache> tabletReaderCache_;
};

TEST_P(NimbleIndexProjectorTest, basicColumnProjection) {
  // Schema: key (int64, sorted), col_a (int32), col_b (varchar)
  auto rowType =
      ROW({"key", "col_a", "col_b"}, {BIGINT(), INTEGER(), VARCHAR()});

  // Generate sorted data.
  const int numRows = 200;
  std::vector<int64_t> keys(numRows);
  std::vector<int32_t> colA(numRows);
  std::vector<std::string> colB(numRows);
  for (int i = 0; i < numRows; ++i) {
    keys[i] = i * 10; // 0, 10, 20, ...
    colA[i] = i * 100;
    colB[i] = fmt::format("value_{}", i);
  }

  std::vector<StringView> colBViews;
  colBViews.reserve(numRows);
  for (const auto& s : colB) {
    colBViews.emplace_back(s);
  }
  auto batch = vectorMaker_->rowVector(
      {"key", "col_a", "col_b"},
      {vectorMaker_->flatVector<int64_t>(keys),
       vectorMaker_->flatVector<int32_t>(colA),
       vectorMaker_->flatVector(colBViews)});

  writeData({batch}, {"key"});

  std::vector<Subfield> subfields;
  subfields.emplace_back("col_a");
  subfields.emplace_back("col_b");
  auto projector = createProjector(subfields);

  // Point lookup for key=50 (row index 5).
  auto pointBounds = makePointLookup(rowType, {"key"}, 50);
  NimbleIndexProjector::Request request;
  request.keyBounds = {pointBounds};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 1);
  const auto& response = result.responses[0];
  ASSERT_FALSE(response.slices.empty());
  EXPECT_FALSE(response.resumeKey.has_value());

  const auto& slice = response.slices[0];
  const auto rowRange = readEmbeddedRowRange(slice);
  ASSERT_GT(rowRange.numRows(), 0);

  // Coalesce the chunk slice IOBuf chain into a contiguous buffer that the
  // Deserializer can read end-to-end.
  auto coalesced = coalesceChunkSlice(slice);
  const std::string_view projectedData{
      reinterpret_cast<const char*>(coalesced.data()), coalesced.length()};

  DeserializerOptions deserOptions;
  deserOptions.hasHeader = true;
  // Deserialize using nimble Deserializer with nimble schema.
  auto projectedVeloxType = ROW({"col_a", "col_b"}, {INTEGER(), VARCHAR()});
  auto projectedNimbleSchema = convertToNimbleType(*projectedVeloxType);
  Deserializer deserializer(
      projectedNimbleSchema, leafPool_.get(), deserOptions);

  VectorPtr deserialized;
  deserializer.deserialize(projectedData, deserialized);
  ASSERT_NE(deserialized, nullptr);

  auto* rowResult = deserialized->as<RowVector>();
  ASSERT_NE(rowResult, nullptr);
  // The Deserializer honors the header's row range at decode time, so the
  // output contains exactly the requested rows starting at index 0.
  ASSERT_EQ(rowResult->size(), rowRange.numRows());
  auto* colAResult = rowResult->childAt(0)->as<FlatVector<int32_t>>();
  ASSERT_NE(colAResult, nullptr);
  // key=50 -> colA=500 (single-row point lookup).
  EXPECT_EQ(colAResult->valueAt(0), 500);
}

// Every IOBuf node in a result slice chain should be managed (own/refcount its
// own buffer) so the chain is self-contained. This currently fails when the
// projected streams are non-contiguous in the read buffer, because packStripe
// wraps the additional run(s) with folly::IOBuf::wrapBuffer (unmanaged). It
// asserts the invariant we are moving toward.
TEST_P(NimbleIndexProjectorTest, resultSlicesAreManaged) {
  // 4 columns; project a NON-adjacent subset (col_a, col_c) so the skipped
  // col_b stream sits between them in the read buffer, forcing a second run.
  auto rowType =
      ROW({"key", "col_a", "col_b", "col_c"},
          {BIGINT(), INTEGER(), INTEGER(), INTEGER()});

  const int numRows = 200;
  std::vector<int64_t> keys(numRows);
  std::vector<int32_t> colA(numRows);
  std::vector<int32_t> colB(numRows);
  std::vector<int32_t> colC(numRows);
  for (int i = 0; i < numRows; ++i) {
    keys[i] = i * 10;
    colA[i] = i * 100;
    colB[i] = i * 1000;
    colC[i] = i * 7;
  }
  auto batch = vectorMaker_->rowVector(
      {"key", "col_a", "col_b", "col_c"},
      {vectorMaker_->flatVector<int64_t>(keys),
       vectorMaker_->flatVector<int32_t>(colA),
       vectorMaker_->flatVector<int32_t>(colB),
       vectorMaker_->flatVector<int32_t>(colC)});

  writeData({batch}, {"key"}, /*flatMapColumns=*/{}, /*stripeSize=*/1 << 10);

  // Skip col_b: its stream lands between col_a's and col_c's in the read
  // buffer, so packStripe cannot extend a single contiguous run.
  std::vector<Subfield> subfields;
  subfields.emplace_back("col_a");
  subfields.emplace_back("col_c");
  auto projector = createProjector(subfields);

  // Range scan over all keys hits every stripe.
  auto rangeBounds = makeRangeLookup(rowType, {"key"}, 0, numRows * 10);
  NimbleIndexProjector::Request request;
  request.keyBounds = {rangeBounds};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 1);
  const auto& response = result.responses[0];
  ASSERT_FALSE(response.slices.empty());

  for (size_t s = 0; s < response.slices.size(); ++s) {
    const auto& slice = response.slices[s];
    EXPECT_TRUE(slice.isManaged())
        << "slice " << s << "/" << response.slices.size()
        << " has an unmanaged (wrapBuffer) IOBuf node";
  }
}

TEST_P(NimbleIndexProjectorTest, emptyResult) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  // Keys: 0, 10, 20, ..., 90
  const int numRows = 10;
  std::vector<int64_t> keys(numRows);
  std::vector<int32_t> values(numRows);
  for (int i = 0; i < numRows; ++i) {
    keys[i] = i * 10;
    values[i] = i;
  }

  auto batch = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>(keys),
       vectorMaker_->flatVector<int32_t>(values)});

  writeData({batch}, {"key"});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // Lookup key=1000, which doesn't exist (beyond max).
  auto bounds = makePointLookup(rowType, {"key"}, 1000);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 1);
  EXPECT_TRUE(result.responses[0].slices.empty());
  EXPECT_FALSE(result.responses[0].resumeKey.has_value());

  // All stats should be zero for a miss.
  EXPECT_EQ(projector->stats().numReadStripes, 0);
  EXPECT_EQ(projector->stats().numReadRows, 0);
  EXPECT_EQ(projector->stats().numProjectedRows, 0);
  EXPECT_EQ(dataIoStats_->rawBytesRead(), 0);
  EXPECT_EQ(dataIoStats_->rawOverreadBytes(), 0);
  EXPECT_EQ(dataIoStats_->read().count(), 0);
  EXPECT_EQ(dataIoStats_->readGap().count(), 0);
}

TEST_P(NimbleIndexProjectorTest, multipleRequests) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  const int numRows = 100;
  std::vector<int64_t> keys(numRows);
  std::vector<int32_t> values(numRows);
  for (int i = 0; i < numRows; ++i) {
    keys[i] = i;
    values[i] = i * 10;
  }

  auto batch = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>(keys),
       vectorMaker_->flatVector<int32_t>(values)});

  writeData({batch}, {"key"});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // Multiple point lookups.
  auto bounds0 = makePointLookup(rowType, {"key"}, 5);
  auto bounds1 = makePointLookup(rowType, {"key"}, 50);
  auto bounds2 = makePointLookup(rowType, {"key"}, 95);

  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds0, bounds1, bounds2};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 3);
  uint64_t totalBytes = 0;
  for (const auto& response : result.responses) {
    for (const auto& slice : response.slices) {
      totalBytes += slice.computeChainDataLength();
    }
  }
  EXPECT_GT(totalBytes, 0);

  // All requests should have results (no misses).
  for (const auto& response : result.responses) {
    EXPECT_FALSE(response.slices.empty());
    EXPECT_FALSE(response.resumeKey.has_value());
  }
}

TEST_P(NimbleIndexProjectorTest, overlappingRequestsShareBody) {
  // Two range scans that hit the same single stripe with different row ranges.
  // Verify that the per-request kTablet chunk slices share the body+trailer
  // IOBuf bytes via cloneOne() and carry distinct per-slice header nodes.
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  const int numRows = 100;
  std::vector<int64_t> keys(numRows);
  std::vector<int32_t> values(numRows);
  for (int i = 0; i < numRows; ++i) {
    keys[i] = i;
    values[i] = i * 10;
  }
  auto batch = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>(keys),
       vectorMaker_->flatVector<int32_t>(values)});

  // Single stripe with all 100 rows (default 4KB stripe size easily fits).
  writeData({batch}, {"key"});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // Two overlapping range scans into the same stripe.
  auto bounds0 = makeRangeLookup(rowType, {"key"}, 10, 40); // 30 rows
  auto bounds1 = makeRangeLookup(rowType, {"key"}, 25, 70); // 45 rows

  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds0, bounds1};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 2);
  ASSERT_EQ(result.responses[0].slices.size(), 1);
  ASSERT_EQ(result.responses[1].slices.size(), 1);

  const auto& slice0 = result.responses[0].slices[0];
  const auto& slice1 = result.responses[1].slices[0];

  // Each chunk slice is a 2-node chain: per-slice header + shared body+trailer.
  ASSERT_TRUE(slice0.isChained());
  ASSERT_TRUE(slice1.isChained());

  // Header nodes (first node in each chunk slice) must be DISTINCT
  // allocations since they carry per-request row range bytes.
  EXPECT_NE(slice0.data(), slice1.data());

  // Body nodes (second node in each chunk slice) must point to the same
  // underlying memory because they are cloneOne() copies of the shared
  // per-stripe body+trailer.
  EXPECT_EQ(slice0.prev()->data(), slice1.prev()->data());
  EXPECT_EQ(slice0.prev()->length(), slice1.prev()->length());

  // Embedded row ranges match the (stripe-relative) request ranges.
  const auto range0 = readEmbeddedRowRange(slice0);
  const auto range1 = readEmbeddedRowRange(slice1);
  EXPECT_EQ(range0, RowRange(10, 40));
  EXPECT_EQ(range1, RowRange(25, 70));

  // Neither response is truncated, so neither slice carries a resume key.
  EXPECT_FALSE(readEmbeddedResumeKey(slice0).has_value());
  EXPECT_FALSE(readEmbeddedResumeKey(slice1).has_value());

  // The shared body dominates each chunk slice size — assert that
  // computeChainDataLength reflects header + body and that the body is
  // refcount-shared.
  EXPECT_GT(slice0.computeChainDataLength(), slice0.length());
  EXPECT_GT(slice0.prev()->length(), 0u);
}

TEST_P(NimbleIndexProjectorTest, slicesStripeBasedOnOverfetchRatio) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  constexpr int numRows = 200;
  std::vector<int64_t> keys(numRows);
  std::vector<int32_t> values(numRows);
  for (int i = 0; i < numRows; ++i) {
    keys[i] = i;
    values[i] = i * 10;
  }
  auto batch = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>(keys),
       vectorMaker_->flatVector<int32_t>(values)});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  struct TestCase {
    double maxOverfetchRowsRatio;
    uint32_t expectedRowCount;
    RowRange expectedRowRange;
    bool expectedStreamHasChunkHeader;
    uint32_t expectedNumSlicedStripes;
  };
  const std::vector<TestCase> testCases = {
      {0.0, 100, RowRange(0, 100), false, 1},
      {0.4, 100, RowRange(0, 100), false, 1},
      {0.5, numRows, RowRange(50, 150), true, 0},
      {0.6, numRows, RowRange(50, 150), true, 0},
      {1.0, numRows, RowRange(50, 150), true, 0},
  };
  const std::vector<int32_t> expectedValues(
      values.begin() + 50, values.begin() + 150);
  const auto expected = vectorMaker_->rowVector(
      {"value"}, {vectorMaker_->flatVector<int32_t>(expectedValues)});

  for (const bool compactRowCountEncoding : {false, true}) {
    SCOPED_TRACE(
        ::testing::Message()
        << "compactRowCountEncoding=" << compactRowCountEncoding);
    writeData(
        {batch},
        {"key"},
        {},
        /*stripeSize=*/1 << 20,
        /*enableStreamDeduplication=*/true,
        compactRowCountEncoding);

    for (const auto& testCase : testCases) {
      SCOPED_TRACE(
          ::testing::Message()
          << "maxOverfetchRowsRatio=" << testCase.maxOverfetchRowsRatio);
      auto projector = createProjector(subfields);
      NimbleIndexProjector::Request request;
      request.keyBounds = {makeRangeLookup(rowType, {"key"}, 50, 150)};
      NimbleIndexProjector::Options options;
      options.maxOverfetchRowsRatio = testCase.maxOverfetchRowsRatio;
      auto result = projector->project(request, options);

      ASSERT_EQ(result.responses.size(), 1);
      ASSERT_EQ(result.responses[0].slices.size(), 1);
      const auto header =
          readEmbeddedTabletChunkHeader(result.responses[0].slices[0]);
      EXPECT_EQ(header.rowCount, testCase.expectedRowCount);
      EXPECT_EQ(header.rowRange, testCase.expectedRowRange);
      EXPECT_EQ(
          header.streamEncodingUsesVarintRowCount, compactRowCountEncoding);
      EXPECT_EQ(
          header.streamHasChunkHeader, testCase.expectedStreamHasChunkHeader);
      EXPECT_EQ(projector->stats().numReadStripes, 1);
      EXPECT_EQ(
          projector->stats().numSlicedStripes,
          testCase.expectedNumSlicedStripes);
      expectVectorRows(
          deserializeProjectedSlice(*projector, result.responses[0].slices[0]),
          expected);
    }
  }
}

TEST_P(NimbleIndexProjectorTest, slicesPartialStripesAroundFullStripe) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  writeResumeKeyTestData(/*rowsPerBatch=*/100, /*numBatches=*/3);

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  NimbleIndexProjector::Request request;
  request.keyBounds = {makeRangeLookup(rowType, {"key"}, 10, 290)};
  NimbleIndexProjector::Options options;
  options.maxOverfetchRowsRatio = 0.0;
  auto result = projector->project(request, options);

  ASSERT_EQ(result.responses.size(), 1);
  ASSERT_EQ(result.responses[0].slices.size(), 3);

  const auto firstHeader =
      readEmbeddedTabletChunkHeader(result.responses[0].slices[0]);
  const auto middleHeader =
      readEmbeddedTabletChunkHeader(result.responses[0].slices[1]);
  const auto lastHeader =
      readEmbeddedTabletChunkHeader(result.responses[0].slices[2]);
  EXPECT_EQ(firstHeader.rowCount, 90);
  EXPECT_EQ(firstHeader.rowRange, RowRange(0, 90));
  EXPECT_FALSE(firstHeader.streamHasChunkHeader);
  EXPECT_EQ(middleHeader.rowCount, 100);
  EXPECT_EQ(middleHeader.rowRange, RowRange(0, 100));
  EXPECT_TRUE(middleHeader.streamHasChunkHeader);
  EXPECT_EQ(lastHeader.rowCount, 90);
  EXPECT_EQ(lastHeader.rowRange, RowRange(0, 90));
  EXPECT_FALSE(lastHeader.streamHasChunkHeader);

  const auto& stats = projector->stats();
  EXPECT_EQ(stats.numReadStripes, 3);
  EXPECT_EQ(stats.numSlicedStripes, 2);
}

TEST_P(NimbleIndexProjectorTest, slicedRequestsShareShiftedBody) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  constexpr int numRows = 100;
  std::vector<int64_t> keys(numRows);
  std::vector<int32_t> values(numRows);
  for (int i = 0; i < numRows; ++i) {
    keys[i] = i;
    values[i] = i * 10;
  }
  auto batch = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>(keys),
       vectorMaker_->flatVector<int32_t>(values)});
  writeData({batch}, {"key"}, {}, /*stripeSize=*/1 << 20);

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  struct TestCase {
    RowRange firstRequest;
    RowRange secondRequest;
    uint32_t expectedPackedRowCount;
    RowRange expectedFirstRange;
    RowRange expectedSecondRange;
  };
  const std::vector<TestCase> testCases = {
      {
          RowRange(10, 40),
          RowRange(25, 70),
          60,
          RowRange(0, 30),
          RowRange(15, 60),
      },
      {
          RowRange(10, 80),
          RowRange(30, 50),
          70,
          RowRange(0, 70),
          RowRange(20, 40),
      },
      {
          RowRange(60, 80),
          RowRange(10, 30),
          70,
          RowRange(50, 70),
          RowRange(0, 20),
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        ::testing::Message()
        << "firstRequest=[" << testCase.firstRequest.startRow << ", "
        << testCase.firstRequest.endRow << ") secondRequest=["
        << testCase.secondRequest.startRow << ", "
        << testCase.secondRequest.endRow << ")");
    auto projector = createProjector(subfields);
    NimbleIndexProjector::Request request;
    request.keyBounds = {
        makeRangeLookup(
            rowType,
            {"key"},
            testCase.firstRequest.startRow,
            testCase.firstRequest.endRow),
        makeRangeLookup(
            rowType,
            {"key"},
            testCase.secondRequest.startRow,
            testCase.secondRequest.endRow),
    };
    NimbleIndexProjector::Options options;
    options.maxOverfetchRowsRatio = 0.0;
    auto result = projector->project(request, options);

    ASSERT_EQ(result.responses.size(), 2);
    ASSERT_EQ(result.responses[0].slices.size(), 1);
    ASSERT_EQ(result.responses[1].slices.size(), 1);

    const auto& slice0 = result.responses[0].slices[0];
    const auto& slice1 = result.responses[1].slices[0];
    EXPECT_EQ(slice0.prev()->data(), slice1.prev()->data());
    EXPECT_EQ(
        readEmbeddedTabletChunkHeader(slice0).rowCount,
        testCase.expectedPackedRowCount);
    EXPECT_EQ(
        readEmbeddedTabletChunkHeader(slice1).rowCount,
        testCase.expectedPackedRowCount);
    EXPECT_EQ(readEmbeddedRowRange(slice0), testCase.expectedFirstRange);
    EXPECT_EQ(readEmbeddedRowRange(slice1), testCase.expectedSecondRange);

    const std::vector<int32_t> expectedValues0(
        values.begin() + testCase.firstRequest.startRow,
        values.begin() + testCase.firstRequest.endRow);
    const auto expected0 = vectorMaker_->rowVector(
        {"value"}, {vectorMaker_->flatVector<int32_t>(expectedValues0)});
    expectVectorRows(deserializeProjectedSlice(*projector, slice0), expected0);

    const std::vector<int32_t> expectedValues1(
        values.begin() + testCase.secondRequest.startRow,
        values.begin() + testCase.secondRequest.endRow);
    const auto expected1 = vectorMaker_->rowVector(
        {"value"}, {vectorMaker_->flatVector<int32_t>(expectedValues1)});
    expectVectorRows(deserializeProjectedSlice(*projector, slice1), expected1);
  }
}

TEST_P(NimbleIndexProjectorTest, projectsFlatMapWithoutConstantInMapStream) {
  constexpr vector_size_t kNumRows{64};
  constexpr uint32_t kRequestStart{10};
  constexpr uint32_t kRequestEnd{30};
  auto rowType = ROW({"key", "features"}, {BIGINT(), MAP(VARCHAR(), BIGINT())});
  auto batch = vectorMaker_->rowVector(
      {"key", "features"},
      {vectorMaker_->flatVector<int64_t>(
           kNumRows, [](auto row) { return static_cast<int64_t>(row); }),
       vectorMaker_->mapVector<StringView, int64_t>(
           kNumRows,
           /*sizeAt=*/[](auto /*row*/) { return 1; },
           /*keyAt=*/
           [](auto /*row*/, auto /*mapIndex*/) { return StringView{"always"}; },
           /*valueAt=*/
           [](auto row, auto /*mapIndex*/) {
             return static_cast<int64_t>(row * 100);
           })});

  writeData(
      {batch},
      {"key"},
      {{"features", {"always"}}},
      /*stripeSize=*/1 << 20,
      /*enableStreamDeduplication=*/true,
      /*compactRowCountEncoding=*/false,
      CompressionType::Uncompressed,
      /*skipConstantFlatMapInMapStreams=*/true,
      // TODO: Enable chunking here once chunked writer skips constant FlatMap
      // in-map streams.
      /*enableChunking=*/false);

  auto readFile =
      std::make_shared<InMemoryReadFile>(std::string_view(sinkData_));
  auto tablet = TabletReader::create(
      readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
  ASSERT_EQ(tablet->stripeCount(), 1);
  BatchReader schemaReader(readFile.get(), *leafPool_);
  const auto& flatMap = schemaReader.schema()->asRow().childAt(1)->asFlatMap();
  ASSERT_EQ(flatMap.childrenCount(), 1);
  ASSERT_EQ(flatMap.nameAt(0), "always");
  const auto stripe = tablet->stripeIdentifier(0);
  const auto inMapStream = flatMap.inMapDescriptorAt(0).offset();
  EXPECT_TRUE(
      inMapStream >= tablet->streamCount(stripe) ||
      tablet->streamSize(stripe, inMapStream) == 0);

  std::vector<Subfield> subfields;
  subfields.emplace_back("features[\"always\"]");
  const auto expected = vectorMaker_->rowVector(
      {"features"},
      {vectorMaker_->mapVector<StringView, int64_t>(
          kRequestEnd - kRequestStart,
          /*sizeAt=*/[](auto /*row*/) { return 1; },
          /*keyAt=*/
          [](auto /*row*/, auto /*mapIndex*/) { return StringView{"always"}; },
          /*valueAt=*/
          [](auto row, auto /*mapIndex*/) {
            return static_cast<int64_t>((row + kRequestStart) * 100);
          })});

  for (const bool sliceStripe : {false, true}) {
    SCOPED_TRACE(fmt::format("sliceStripe={}", sliceStripe));
    auto projector = createProjector(subfields);
    NimbleIndexProjector::Request request;
    request.keyBounds = {
        makeRangeLookup(rowType, {"key"}, kRequestStart, kRequestEnd)};
    NimbleIndexProjector::Options options;
    options.maxOverfetchRowsRatio = sliceStripe ? 0.0 : 1.0;
    auto result = projector->project(request, options);

    ASSERT_EQ(result.responses.size(), 1);
    ASSERT_EQ(result.responses[0].slices.size(), 1);
    const auto& slice = result.responses[0].slices[0];
    const auto header = readEmbeddedTabletChunkHeader(slice);
    EXPECT_EQ(
        header.rowCount, sliceStripe ? kRequestEnd - kRequestStart : kNumRows);
    EXPECT_EQ(
        header.rowRange,
        sliceStripe ? RowRange(0, kRequestEnd - kRequestStart)
                    : RowRange(kRequestStart, kRequestEnd));
    EXPECT_EQ(header.streamHasChunkHeader, !sliceStripe);
    expectVectorRows(deserializeProjectedSlice(*projector, slice), expected);
  }
}

TEST_P(NimbleIndexProjectorTest, preservesStreamRowCountEncoding) {
  constexpr vector_size_t kNumRows{64};
  constexpr uint32_t kRequestStart{10};
  constexpr uint32_t kRequestEnd{30};
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  auto batch = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>(
           kNumRows, [](auto row) { return static_cast<int64_t>(row); }),
       vectorMaker_->flatVector<int32_t>(
           kNumRows, [](auto row) { return row * 10; })});
  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  const auto expected = vectorMaker_->rowVector(
      {"value"},
      {vectorMaker_->flatVector<int32_t>(
          kRequestEnd - kRequestStart,
          [](auto row) { return (row + kRequestStart) * 10; })});

  for (const bool streamsUseVarintRowCount : {false, true}) {
    SCOPED_TRACE(
        fmt::format("streamsUseVarintRowCount={}", streamsUseVarintRowCount));
    writeData(
        {batch},
        {"key"},
        {},
        /*stripeSize=*/1 << 20,
        /*enableStreamDeduplication=*/true,
        /*compactRowCountEncoding=*/streamsUseVarintRowCount);

    for (const bool sliceStripe : {false, true}) {
      SCOPED_TRACE(fmt::format("sliceStripe={}", sliceStripe));
      auto projector = createProjector(subfields);
      NimbleIndexProjector::Request request;
      request.keyBounds = {
          makeRangeLookup(rowType, {"key"}, kRequestStart, kRequestEnd)};
      NimbleIndexProjector::Options options;
      options.maxOverfetchRowsRatio = sliceStripe ? 0.0 : 1.0;
      auto result = projector->project(request, options);

      ASSERT_EQ(result.responses.size(), 1);
      ASSERT_EQ(result.responses[0].slices.size(), 1);
      const auto& slice = result.responses[0].slices[0];
      const auto header = readEmbeddedTabletChunkHeader(slice);
      EXPECT_EQ(
          header.streamEncodingUsesVarintRowCount, streamsUseVarintRowCount);
      EXPECT_EQ(
          header.rowCount,
          sliceStripe ? kRequestEnd - kRequestStart : kNumRows);
      EXPECT_EQ(
          header.rowRange,
          sliceStripe ? RowRange(0, kRequestEnd - kRequestStart)
                      : RowRange(kRequestStart, kRequestEnd));
      EXPECT_EQ(header.streamHasChunkHeader, !sliceStripe);
      expectVectorRows(deserializeProjectedSlice(*projector, slice), expected);
    }
  }
}

TEST_P(NimbleIndexProjectorTest, fuzzesFlatMapPartialStripeProjection) {
  constexpr vector_size_t kRows{256};
  constexpr int kIterations{4};
  const auto rowType =
      ROW({"key", "features", "payload"},
          {BIGINT(), MAP(VARCHAR(), BIGINT()), VARCHAR()});
  const folly::F14FastMap<std::string, std::set<std::string>> flatMapColumns = {
      {"features", {"always", "never", "sparse"}}};

  const auto seed = folly::Random::rand32();
  LOG(INFO) << "fuzzesFlatMapPartialStripeProjection seed: " << seed;
  folly::detail::DefaultGenerator random{seed};

  for (int iteration = 0; iteration < kIterations; ++iteration) {
    const auto rangeStart =
        8 + static_cast<uint32_t>(folly::Random::rand32(random) % 24);
    const std::array<RowRange, 3> requestRanges = {
        RowRange{rangeStart, rangeStart + 64},
        RowRange{rangeStart + 16, rangeStart + 96},
        RowRange{rangeStart + 80, rangeStart + 128},
    };
    SCOPED_TRACE(
        fmt::format(
            "seed={} iteration={} rangeStart={}", seed, iteration, rangeStart));

    std::vector<bool> sparsePresent(kRows);
    std::vector<bool> alwaysValueNull(kRows);
    std::vector<bool> sparseValueNull(kRows);
    for (vector_size_t row = 0; row < kRows; ++row) {
      sparsePresent[row] = folly::Random::oneIn(2, random);
      alwaysValueNull[row] = folly::Random::oneIn(5, random);
      sparseValueNull[row] = folly::Random::oneIn(4, random);
    }
    sparsePresent[rangeStart] = false;
    sparsePresent[rangeStart + 1] = true;
    alwaysValueNull[rangeStart + 2] = true;
    sparseValueNull[rangeStart + 1] = true;
    vector_size_t numEntries{kRows};
    for (const bool isPresent : sparsePresent) {
      numEntries += isPresent;
    }

    auto offsets = allocateOffsets(kRows, leafPool_.get());
    auto sizes = allocateSizes(kRows, leafPool_.get());
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    auto mapKeys = BaseVector::create<FlatVector<StringView>>(
        VARCHAR(), numEntries, leafPool_.get());
    auto mapValues = BaseVector::create<FlatVector<int64_t>>(
        BIGINT(), numEntries, leafPool_.get());
    vector_size_t entry{0};
    for (vector_size_t row = 0; row < kRows; ++row) {
      rawOffsets[row] = entry;
      rawSizes[row] = sparsePresent[row] ? 2 : 1;
      mapKeys->set(entry, StringView{"always"});
      mapValues->set(entry, row * 100);
      mapValues->setNull(entry, alwaysValueNull[row]);
      ++entry;
      if (sparsePresent[row]) {
        mapKeys->set(entry, StringView{"sparse"});
        mapValues->set(entry, row * 100 + 1);
        mapValues->setNull(entry, sparseValueNull[row]);
        ++entry;
      }
    }
    ASSERT_EQ(entry, numEntries);

    auto features = std::make_shared<MapVector>(
        leafPool_.get(),
        MAP(VARCHAR(), BIGINT()),
        nullptr,
        kRows,
        offsets,
        sizes,
        mapKeys,
        mapValues);
    VectorFuzzer fuzzer(
        {
            .vectorSize = kRows,
            .nullRatio = 0.25,
            .useRandomNullPattern = true,
            .stringLength = 256,
            .stringVariableLength = false,
        },
        leafPool_.get(),
        folly::Random::rand32(random));
    auto payload = fuzzer.fuzzFlat(VARCHAR(), kRows);
    payload->setNull(rangeStart + 3, true);
    const std::string compressiblePayload(256, 'p');
    auto* payloadValues = payload->as<FlatVector<StringView>>();
    ASSERT_NE(payloadValues, nullptr);
    for (vector_size_t row = 0; row < kRows; ++row) {
      if (!payloadValues->isNullAt(row)) {
        payloadValues->set(row, StringView{compressiblePayload});
      }
    }
    auto batch = vectorMaker_->rowVector(
        {"key", "features", "payload"},
        {vectorMaker_->flatVector<int64_t>(
             kRows, [](auto row) { return static_cast<int64_t>(row); }),
         features,
         payload});

    for (const bool skipConstantInMapStreams : {false, true}) {
      for (const bool compressChunks : {false, true}) {
        SCOPED_TRACE(
            fmt::format(
                "skipConstantInMapStreams={} compressChunks={}",
                skipConstantInMapStreams,
                compressChunks));
        const auto chunkCompressionType = compressChunks
            ? CompressionType::Zstd
            : CompressionType::Uncompressed;
        writeData(
            {batch},
            {"key"},
            flatMapColumns,
            /*stripeSize=*/1 << 20,
            /*enableStreamDeduplication=*/true,
            /*compactRowCountEncoding=*/false,
            chunkCompressionType,
            skipConstantInMapStreams);

        auto readFile =
            std::make_shared<InMemoryReadFile>(std::string_view(sinkData_));
        auto tablet = TabletReader::create(
            readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
        ASSERT_EQ(tablet->stripeCount(), 1);
        const auto stripe = tablet->stripeIdentifier(0);
        BatchReader schemaReader(readFile.get(), *leafPool_);
        const auto& root = schemaReader.schema()->asRow();
        const auto& flatMap = root.childAt(1)->asFlatMap();
        const auto findFlatMapKey = [&](std::string_view key) {
          for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
            if (flatMap.nameAt(i) == key) {
              return i;
            }
          }
          NIMBLE_FAIL("Missing FlatMap key {}", key);
        };
        const auto streamIsPresent = [&](uint32_t streamId) {
          return streamId < tablet->streamCount(stripe) &&
              tablet->streamSize(stripe, streamId) > 0;
        };
        EXPECT_EQ(
            streamIsPresent(
                flatMap.inMapDescriptorAt(findFlatMapKey("never")).offset()),
            false);
        EXPECT_TRUE(streamIsPresent(
            flatMap.inMapDescriptorAt(findFlatMapKey("sparse")).offset()));

        std::vector<uint32_t> streamIds(tablet->streamCount(stripe));
        std::iota(streamIds.begin(), streamIds.end(), 0);
        auto streams = tablet->load(stripe, streamIds);
        bool sawCompressedChunk{false};
        for (const auto& stream : streams) {
          if (stream == nullptr) {
            continue;
          }
          const auto streamData = stream->getStream();
          const auto* position = streamData.data();
          const auto* const end = position + streamData.size();
          while (position < end) {
            const auto chunkHeader = readChunkHeader(position);
            ASSERT_LE(chunkHeader.length, end - position);
            if (chunkHeader.compressionType == CompressionType::Zstd) {
              sawCompressedChunk = true;
            } else {
              EXPECT_EQ(
                  chunkHeader.compressionType, CompressionType::Uncompressed);
            }
            position += chunkHeader.length;
          }
          EXPECT_EQ(position, end);
        }
        EXPECT_EQ(sawCompressedChunk, compressChunks);

        std::vector<Subfield> subfields;
        subfields.emplace_back("features[\"always\"]");
        subfields.emplace_back("features[\"never\"]");
        subfields.emplace_back("features[\"sparse\"]");
        subfields.emplace_back("payload");
        auto projector = createProjector(subfields);
        NimbleIndexProjector::Request request;
        for (const auto& requestRange : requestRanges) {
          request.keyBounds.push_back(makeRangeLookup(
              rowType, {"key"}, requestRange.startRow, requestRange.endRow));
        }
        NimbleIndexProjector::Options options;
        options.maxOverfetchRowsRatio = 0.0;
        auto result = projector->project(request, options);

        ASSERT_EQ(result.responses.size(), requestRanges.size());
        const RowRange packRange{
            requestRanges.front().startRow, requestRanges.back().endRow};
        const folly::IOBuf* sharedBody{nullptr};
        for (size_t requestIndex = 0; requestIndex < requestRanges.size();
             ++requestIndex) {
          const auto& response = result.responses[requestIndex];
          ASSERT_EQ(response.slices.size(), 1);
          const auto& slice = response.slices[0];
          ASSERT_TRUE(slice.isChained());
          if (sharedBody == nullptr) {
            sharedBody = slice.prev();
          } else {
            EXPECT_EQ(slice.prev()->data(), sharedBody->data());
          }

          const auto header = readEmbeddedTabletChunkHeader(slice);
          EXPECT_EQ(header.rowCount, packRange.numRows());
          const RowRange expectedRowRange{
              requestRanges[requestIndex].startRow - packRange.startRow,
              requestRanges[requestIndex].endRow - packRange.startRow};
          EXPECT_EQ(header.rowRange, expectedRowRange);
          EXPECT_FALSE(header.streamHasChunkHeader);

          auto output = deserializeProjectedSlice(*projector, slice);
          ASSERT_NE(output, nullptr);
          ASSERT_EQ(output->size(), requestRanges[requestIndex].numRows());
          const auto* outputRow = output->as<RowVector>();
          ASSERT_NE(outputRow, nullptr);
          const auto* outputFeatures = outputRow->childAt(0)->as<MapVector>();
          const auto* outputPayload =
              outputRow->childAt(1)->as<FlatVector<StringView>>();
          ASSERT_NE(outputFeatures, nullptr);
          ASSERT_NE(outputPayload, nullptr);
          const auto* outputMapKeys =
              outputFeatures->mapKeys()->as<FlatVector<StringView>>();
          const auto* outputMapValues =
              outputFeatures->mapValues()->as<FlatVector<int64_t>>();
          ASSERT_NE(outputMapKeys, nullptr);
          ASSERT_NE(outputMapValues, nullptr);
          const auto* inputPayload = payload->as<FlatVector<StringView>>();
          ASSERT_NE(inputPayload, nullptr);

          for (vector_size_t outputIndex = 0; outputIndex < output->size();
               ++outputIndex) {
            const auto inputIndex = static_cast<vector_size_t>(
                requestRanges[requestIndex].startRow + outputIndex);
            SCOPED_TRACE(
                fmt::format(
                    "requestIndex={} outputIndex={} inputIndex={}",
                    requestIndex,
                    outputIndex,
                    inputIndex));
            EXPECT_EQ(
                outputPayload->isNullAt(outputIndex),
                inputPayload->isNullAt(inputIndex));
            if (!inputPayload->isNullAt(inputIndex)) {
              EXPECT_EQ(
                  outputPayload->valueAt(outputIndex),
                  inputPayload->valueAt(inputIndex));
            }

            bool sawAlways{false};
            bool sawSparse{false};
            const auto mapOffset = outputFeatures->offsetAt(outputIndex);
            const auto mapSize = outputFeatures->sizeAt(outputIndex);
            for (vector_size_t mapIndex = 0; mapIndex < mapSize; ++mapIndex) {
              const auto outputEntry = mapOffset + mapIndex;
              const auto key = outputMapKeys->valueAt(outputEntry).str();
              if (key == "always") {
                sawAlways = true;
                EXPECT_EQ(
                    outputMapValues->isNullAt(outputEntry),
                    alwaysValueNull[inputIndex]);
                if (!alwaysValueNull[inputIndex]) {
                  EXPECT_EQ(
                      outputMapValues->valueAt(outputEntry), inputIndex * 100);
                }
              } else if (key == "sparse") {
                sawSparse = true;
                EXPECT_TRUE(sparsePresent[inputIndex]);
                EXPECT_EQ(
                    outputMapValues->isNullAt(outputEntry),
                    sparseValueNull[inputIndex]);
                if (!sparseValueNull[inputIndex]) {
                  EXPECT_EQ(
                      outputMapValues->valueAt(outputEntry),
                      inputIndex * 100 + 1);
                }
              } else {
                ADD_FAILURE() << "Unexpected FlatMap key: " << key;
              }
            }
            EXPECT_TRUE(sawAlways);
            EXPECT_EQ(sawSparse, sparsePresent[inputIndex]);
          }
        }
      }
    }
  }
}

TEST_P(NimbleIndexProjectorTest, deduplicatedProjectedStreamsReadOnce) {
  auto rowType =
      ROW({"key", "dup_a", "dup_b"}, {BIGINT(), INTEGER(), INTEGER()});

  constexpr int numRows = 1'024;
  std::vector<int64_t> keys(numRows);
  std::vector<int32_t> values(numRows);
  for (int i = 0; i < numRows; ++i) {
    keys[i] = i;
    values[i] = i % 17;
  }

  auto batch = vectorMaker_->rowVector(
      {"key", "dup_a", "dup_b"},
      {vectorMaker_->flatVector<int64_t>(keys),
       vectorMaker_->flatVector<int32_t>(values),
       vectorMaker_->flatVector<int32_t>(values)});

  writeData(
      {batch},
      {"key"},
      {},
      /*stripeSize=*/1 << 20,
      /*enableStreamDeduplication=*/true);

  std::vector<Subfield> subfields;
  subfields.emplace_back("dup_a");
  subfields.emplace_back("dup_b");
  auto projector = createProjector(subfields);

  auto bounds = makeRangeLookup(rowType, {"key"}, 0, numRows);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 1);
  ASSERT_EQ(result.responses[0].slices.size(), 1);
  EXPECT_EQ(projector->stats().numReadStripes, 1);
  EXPECT_EQ(dataIoStats_->read().count(), 1);
  // dup_a and dup_b carry identical data, so the source tablet aliases their
  // streams to one extent. The packed output is now deduplicated too: the body
  // holds a single physical copy that both logical streams reference via the
  // kTablet dedup trailer, so the output stays well under the ~2x size it would
  // be if each logical stream carried its own copy.
  EXPECT_LT(projector->stats().numOutputBytes, dataIoStats_->rawBytesRead() * 2)
      << "Deduplicated output must hold a single physical copy of the aliased "
         "stream, not two";

  // The duplicate streams are reported through IoStatistics when the data read
  // reaches DirectDataInput. With pinned metadata enabled, the read may be
  // served from pinned stream bodies, so duplicate read counters are not a
  // stable signal.
  if (!GetParam().pinMetadata) {
    EXPECT_GT(dataIoStats_->duplicateReadRegions(), 0u)
        << "dup_a and dup_b should be detected as duplicate read regions";
    EXPECT_GT(dataIoStats_->duplicateReadBytes(), 0u);
  }

  // Both aliased logical streams must still decode correctly from the single
  // shared body copy: the kTablet dedup trailer resolves both slots to the same
  // body extent.
  auto expected = vectorMaker_->rowVector(
      {"dup_a", "dup_b"},
      {vectorMaker_->flatVector<int32_t>(values),
       vectorMaker_->flatVector<int32_t>(values)});
  expectVectorRows(
      deserializeProjectedSlice(*projector, result.responses[0].slices[0]),
      expected);
}

TEST_P(NimbleIndexProjectorTest, projectedNullableDataDeserializes) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  constexpr int numRows = 64;
  auto batch = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>(
           numRows, [](auto row) { return static_cast<int64_t>(row); }),
       vectorMaker_->flatVector<int32_t>(
           numRows,
           [](auto row) { return static_cast<int32_t>(row * 10); },
           [](auto row) { return row % 5 == 0; })});

  writeData({batch}, {"key"});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  auto bounds = makeRangeLookup(rowType, {"key"}, 10, 30);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 1);
  ASSERT_EQ(result.responses[0].slices.size(), 1);

  auto expected = vectorMaker_->rowVector(
      {"value"},
      {vectorMaker_->flatVector<int32_t>(
          20,
          [](auto row) { return static_cast<int32_t>((row + 10) * 10); },
          [](auto row) { return (row + 10) % 5 == 0; })});
  expectVectorRows(
      deserializeProjectedSlice(*projector, result.responses[0].slices[0]),
      expected);
}

TEST_P(NimbleIndexProjectorTest, projectedTabletNullBarrierFlag) {
  constexpr vector_size_t kRows = 6;
  auto nestedValueType = ROW({{"score", INTEGER()}});
  auto nestedRowType = ROW({
      {"key", BIGINT()},
      {"id", BIGINT()},
      {"profile", nestedValueType},
      {"nullable_score", INTEGER()},
  });

  auto makeNestedBatch = [&](bool hasRowNulls) {
    auto ids = vectorMaker_->flatVector<int64_t>(
        kRows, [](auto row) { return static_cast<int64_t>(row); });
    auto profile = std::make_shared<RowVector>(
        leafPool_.get(),
        nestedValueType,
        nullptr,
        kRows,
        std::vector<VectorPtr>{vectorMaker_->flatVector<int32_t>(
            kRows, [](auto row) { return static_cast<int32_t>(row * 10); })});
    if (hasRowNulls) {
      profile->setNull(2, true);
    }

    auto nullableScore = vectorMaker_->flatVector<int32_t>(
        kRows,
        [](auto row) { return static_cast<int32_t>(1000 + row); },
        [](auto row) { return row == 4; });

    return vectorMaker_->rowVector(
        {"key", "id", "profile", "nullable_score"},
        {vectorMaker_->flatVector<int64_t>(
             kRows, [](auto row) { return static_cast<int64_t>(row); }),
         ids,
         profile,
         nullableScore});
  };

  auto flatMapRowType =
      ROW({{"key", BIGINT()}, {"features", MAP(VARCHAR(), DOUBLE())}});
  auto makeFlatMapBatch = [&](bool hasValueNulls, bool hasFlatMapNulls) {
    auto offsets = allocateOffsets(kRows, leafPool_.get());
    auto sizes = allocateSizes(kRows, leafPool_.get());
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    for (vector_size_t row = 0; row < kRows; ++row) {
      rawOffsets[row] = row;
      rawSizes[row] = 1;
    }
    auto featureValues = vectorMaker_->flatVector<double>(
        kRows, [](auto row) { return static_cast<double>(100 + row); });
    if (hasValueNulls) {
      featureValues->setNull(2, true);
    }
    auto features = std::make_shared<MapVector>(
        leafPool_.get(),
        MAP(VARCHAR(), DOUBLE()),
        nullptr,
        kRows,
        offsets,
        sizes,
        vectorMaker_->flatVector<StringView>(
            kRows, [](auto /*row*/) { return StringView("a"); }),
        featureValues);
    if (hasFlatMapNulls) {
      features->setNull(3, true);
    }

    return vectorMaker_->rowVector(
        {"key", "features"},
        {vectorMaker_->flatVector<int64_t>(
             kRows, [](auto row) { return static_cast<int64_t>(row); }),
         features});
  };

  auto assertBarrier =
      [&](const RowVectorPtr& batch,
          const RowTypePtr& rowType,
          std::string_view subfield,
          const folly::F14FastMap<std::string, std::set<std::string>>&
              flatMapColumns,
          bool expectedRequiresNullBarrier) {
        writeData(
            {batch},
            {"key"},
            flatMapColumns,
            /*stripeSize=*/1 << 20);
        std::vector<Subfield> subfields;
        subfields.emplace_back(std::string{subfield});
        auto projector = createProjector(subfields);
        auto bounds = makeRangeLookup(rowType, {"key"}, 0, kRows);
        NimbleIndexProjector::Request request;
        request.keyBounds = {bounds};
        auto result = projector->project(request, {});

        ASSERT_EQ(result.responses.size(), 1);
        ASSERT_EQ(result.responses[0].slices.size(), 1);
        EXPECT_EQ(
            readEmbeddedTabletChunkHeader(result.responses[0].slices[0])
                .requiresNullBarrier,
            expectedRequiresNullBarrier);
      };

  const auto nestedNoNulls = makeNestedBatch(false);
  const auto nestedHasNulls = makeNestedBatch(true);
  assertBarrier(nestedNoNulls, nestedRowType, "profile.score", {}, false);
  assertBarrier(nestedNoNulls, nestedRowType, "nullable_score", {}, false);
  assertBarrier(nestedHasNulls, nestedRowType, "id", {}, false);
  assertBarrier(nestedHasNulls, nestedRowType, "nullable_score", {}, false);
  assertBarrier(nestedHasNulls, nestedRowType, "profile.score", {}, true);

  const auto flatMapNoNulls = makeFlatMapBatch(false, false);
  const auto flatMapValueNulls = makeFlatMapBatch(true, false);
  const auto flatMapHasNulls = makeFlatMapBatch(false, true);
  assertBarrier(
      flatMapNoNulls,
      flatMapRowType,
      "features[\"a\"]",
      {{"features", {}}},
      false);
  assertBarrier(
      flatMapValueNulls,
      flatMapRowType,
      "features[\"a\"]",
      {{"features", {}}},
      false);
  assertBarrier(
      flatMapHasNulls,
      flatMapRowType,
      "features[\"a\"]",
      {{"features", {}}},
      true);
}

TEST_P(
    NimbleIndexProjectorTest,
    projectedTabletMixedNullBarrierChunksPreserveNulls) {
  constexpr vector_size_t kRowsPerBatch = 5;
  auto nestedType = ROW({{"score", INTEGER()}});
  auto rowType = ROW({
      {"key", BIGINT()},
      {"nested", nestedType},
  });

  auto makeBatch = [&](int64_t keyBase,
                       int32_t valueBase,
                       std::optional<vector_size_t> nestedNullRow) {
    auto nested = std::make_shared<RowVector>(
        leafPool_.get(),
        nestedType,
        nullptr,
        kRowsPerBatch,
        std::vector<VectorPtr>{vectorMaker_->flatVector<int32_t>(
            kRowsPerBatch, [valueBase](auto row) {
              return static_cast<int32_t>(valueBase + row);
            })});
    if (nestedNullRow.has_value()) {
      nested->setNull(*nestedNullRow, true);
    }
    return vectorMaker_->rowVector(
        {"key", "nested"},
        {vectorMaker_->flatVector<int64_t>(
             kRowsPerBatch, [keyBase](auto row) { return keyBase + row; }),
         nested});
  };

  auto first = makeBatch(0, 100, std::nullopt);
  auto second = makeBatch(5, 200, 2);
  auto third = makeBatch(10, 300, std::nullopt);
  writeData({first, second, third}, {"key"}, {}, /*stripeSize=*/1);

  std::vector<Subfield> subfields;
  subfields.emplace_back("nested.score");
  auto projector = createProjector(subfields);
  auto bounds = makeRangeLookup(rowType, {"key"}, 0, 15);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 1);
  ASSERT_EQ(result.responses[0].slices.size(), 3);
  EXPECT_FALSE(readEmbeddedTabletChunkHeader(result.responses[0].slices[0])
                   .requiresNullBarrier);
  EXPECT_TRUE(readEmbeddedTabletChunkHeader(result.responses[0].slices[1])
                  .requiresNullBarrier);
  EXPECT_FALSE(readEmbeddedTabletChunkHeader(result.responses[0].slices[2])
                   .requiresNullBarrier);

  std::vector<folly::IOBuf> owned;
  std::vector<std::string_view> batches;
  owned.reserve(result.responses[0].slices.size());
  batches.reserve(result.responses[0].slices.size());
  for (const auto& slice : result.responses[0].slices) {
    owned.push_back(coalesceChunkSlice(slice));
    batches.emplace_back(
        reinterpret_cast<const char*>(owned.back().data()),
        owned.back().length());
  }
  auto output = deserializeProjectedBatches(*projector, batches);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->size(), 15);

  auto* nested = output->as<RowVector>()->childAt(0)->as<RowVector>();
  ASSERT_NE(nested, nullptr);
  auto* scores = nested->childAt(0)->as<FlatVector<int32_t>>();
  ASSERT_NE(scores, nullptr);
  for (vector_size_t row = 0; row < 15; ++row) {
    if (row == 7) {
      EXPECT_TRUE(nested->isNullAt(row));
      continue;
    }
    EXPECT_FALSE(nested->isNullAt(row));
    const auto batchBase = row < 5 ? 100 : (row < 10 ? 200 : 300);
    EXPECT_EQ(scores->valueAt(row), batchBase + row % kRowsPerBatch);
  }
}

TEST_P(
    NimbleIndexProjectorTest,
    projectedTabletUnselectedNullColumnDoesNotRequireNullBarrier) {
  constexpr vector_size_t kRows = 6;
  auto skippedType = ROW({{"score", INTEGER()}});
  auto rowType = ROW({
      {"key", BIGINT()},
      {"kept", INTEGER()},
      {"skipped", skippedType},
  });

  auto skipped = std::make_shared<RowVector>(
      leafPool_.get(),
      skippedType,
      nullptr,
      kRows,
      std::vector<VectorPtr>{vectorMaker_->flatVector<int32_t>(
          kRows, [](auto row) { return static_cast<int32_t>(100 + row); })});
  skipped->setNull(1, true);
  skipped->setNull(4, true);
  auto batch = vectorMaker_->rowVector(
      {"key", "kept", "skipped"},
      {vectorMaker_->flatVector<int64_t>(
           kRows, [](auto row) { return static_cast<int64_t>(row); }),
       vectorMaker_->flatVector<int32_t>(
           kRows, [](auto row) { return static_cast<int32_t>(10 + row); }),
       skipped});

  writeData({batch}, {"key"}, {}, /*stripeSize=*/1 << 20);

  std::vector<Subfield> subfields;
  subfields.emplace_back("kept");
  auto projector = createProjector(subfields);
  auto bounds = makeRangeLookup(rowType, {"key"}, 0, kRows);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 1);
  ASSERT_EQ(result.responses[0].slices.size(), 1);
  EXPECT_FALSE(readEmbeddedTabletChunkHeader(result.responses[0].slices[0])
                   .requiresNullBarrier);

  auto output =
      deserializeProjectedSlice(*projector, result.responses[0].slices[0]);
  ASSERT_EQ(output->size(), kRows);
  auto* kept = output->as<RowVector>()->childAt(0)->as<FlatVector<int32_t>>();
  ASSERT_NE(kept, nullptr);
  for (vector_size_t row = 0; row < kRows; ++row) {
    EXPECT_FALSE(kept->isNullAt(row));
    EXPECT_EQ(kept->valueAt(row), 10 + row);
  }
}

TEST_P(NimbleIndexProjectorTest, projectedComplexNullFuzzer) {
  constexpr vector_size_t kRowsPerBatch = 8;
  constexpr int kNumBatches = 4;
  constexpr vector_size_t kRows = kRowsPerBatch * kNumBatches;
  constexpr int kIterations = 8;
  constexpr vector_size_t kEntriesPerMapRow = 2;

  auto rowType = ROW({
      {"key", BIGINT()},
      {"profile",
       ROW({
           {"score", INTEGER()},
           {"details",
            ROW({
                {"rank", BIGINT()},
                {"quality", DOUBLE()},
            })},
       })},
      {"activity",
       ROW({
           {"clicks", INTEGER()},
           {"label", VARCHAR()},
           {"inner",
            ROW({
                {"flag", BOOLEAN()},
                {"weight", DOUBLE()},
            })},
       })},
      {"attrs", MAP(VARCHAR(), DOUBLE())},
      {"flatAttrs", MAP(VARCHAR(), DOUBLE())},
  });

  struct RowLeafPath {
    std::string path;
    std::vector<std::string_view> names;
  };
  struct FlatMapLeafPath {
    std::string path;
    std::string_view key;
  };

  const std::vector<RowLeafPath> rowLeafPaths = {
      {"profile.score", {"profile", "score"}},
      {"profile.details.rank", {"profile", "details", "rank"}},
      {"profile.details.quality", {"profile", "details", "quality"}},
      {"activity.clicks", {"activity", "clicks"}},
      {"activity.label", {"activity", "label"}},
      {"activity.inner.flag", {"activity", "inner", "flag"}},
      {"activity.inner.weight", {"activity", "inner", "weight"}},
  };
  const std::vector<FlatMapLeafPath> flatMapLeafPaths = {
      {"flatAttrs[\"a\"]", "a"},
      {"flatAttrs[\"b\"]", "b"},
  };

  auto makeFlatInputRow = [&](VectorFuzzer& fuzzer) {
    std::vector<VectorPtr> children;
    children.reserve(rowType->size());
    for (size_t child = 0; child < rowType->size(); ++child) {
      children.push_back(
          fuzzer.fuzzFlat(rowType->childAt(child), kRowsPerBatch));
    }
    return std::make_shared<RowVector>(
        leafPool_.get(), rowType, nullptr, kRowsPerBatch, std::move(children));
  };

  auto childIndex = [](const RowVector* row, std::string_view name) {
    const auto& type = row->type()->asRow();
    for (size_t i = 0; i < type.size(); ++i) {
      if (type.nameOf(i) == name) {
        return i;
      }
    }
    NIMBLE_FAIL("Missing child {}", name);
  };

  auto leafVector = [&](const RowVector* root,
                        const RowLeafPath& path) -> const BaseVector* {
    const RowVector* row = root;
    for (size_t i = 0; i + 1 < path.names.size(); ++i) {
      row = row->childAt(childIndex(row, path.names[i]))->as<RowVector>();
    }
    return row->childAt(childIndex(row, path.names.back())).get();
  };

  auto pathIsNull = [&](const RowVector* root,
                        const RowLeafPath& path,
                        vector_size_t rowIndex) {
    const RowVector* row = root;
    if (row->isNullAt(rowIndex)) {
      return true;
    }
    for (size_t i = 0; i + 1 < path.names.size(); ++i) {
      row = row->childAt(childIndex(row, path.names[i]))->as<RowVector>();
      if (row->isNullAt(rowIndex)) {
        return true;
      }
    }
    return row->childAt(childIndex(row, path.names.back()))->isNullAt(rowIndex);
  };

  auto expectLeafEqual = [&](const RowVector* expectedRoot,
                             vector_size_t expectedRow,
                             const RowVector* actualRoot,
                             vector_size_t actualRow,
                             const RowLeafPath& path) {
    const auto expectedNull = pathIsNull(expectedRoot, path, expectedRow);
    EXPECT_EQ(pathIsNull(actualRoot, path, actualRow), expectedNull)
        << "path=" << path.path << " expectedRow=" << expectedRow
        << " actualRow=" << actualRow;
    if (expectedNull) {
      return;
    }

    const auto* expected = leafVector(expectedRoot, path);
    const auto* actual = leafVector(actualRoot, path);
    switch (expected->type()->kind()) {
      case TypeKind::BOOLEAN:
        EXPECT_EQ(
            expected->as<FlatVector<bool>>()->valueAt(expectedRow),
            actual->as<FlatVector<bool>>()->valueAt(actualRow));
        break;
      case TypeKind::INTEGER:
        EXPECT_EQ(
            expected->as<FlatVector<int32_t>>()->valueAt(expectedRow),
            actual->as<FlatVector<int32_t>>()->valueAt(actualRow));
        break;
      case TypeKind::BIGINT:
        EXPECT_EQ(
            expected->as<FlatVector<int64_t>>()->valueAt(expectedRow),
            actual->as<FlatVector<int64_t>>()->valueAt(actualRow));
        break;
      case TypeKind::DOUBLE:
        EXPECT_EQ(
            expected->as<FlatVector<double>>()->valueAt(expectedRow),
            actual->as<FlatVector<double>>()->valueAt(actualRow));
        break;
      case TypeKind::VARCHAR:
        EXPECT_EQ(
            expected->as<FlatVector<StringView>>()->valueAt(expectedRow),
            actual->as<FlatVector<StringView>>()->valueAt(actualRow));
        break;
      default:
        NIMBLE_FAIL(
            "Unexpected projected type {}", expected->type()->toString());
    }
  };

  auto makeStringDoubleMap = [&](bool hasContainerNulls, double valueBase) {
    const auto numEntries = kRowsPerBatch * kEntriesPerMapRow;
    auto offsets = allocateOffsets(kRowsPerBatch, leafPool_.get());
    auto sizes = allocateSizes(kRowsPerBatch, leafPool_.get());
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    for (vector_size_t row = 0; row < kRowsPerBatch; ++row) {
      rawOffsets[row] = row * kEntriesPerMapRow;
      rawSizes[row] = kEntriesPerMapRow;
    }

    auto keys = BaseVector::create<FlatVector<StringView>>(
        VARCHAR(), numEntries, leafPool_.get());
    auto values = BaseVector::create<FlatVector<double>>(
        DOUBLE(), numEntries, leafPool_.get());
    for (vector_size_t row = 0; row < kRowsPerBatch; ++row) {
      for (vector_size_t mapIndex = 0; mapIndex < kEntriesPerMapRow;
           ++mapIndex) {
        const auto entry = row * kEntriesPerMapRow + mapIndex;
        keys->set(entry, StringView(mapIndex == 0 ? "a" : "b"));
        values->set(entry, valueBase + row * 10 + mapIndex);
      }
    }
    values->setNull(4 * kEntriesPerMapRow, true);
    auto map = std::make_shared<MapVector>(
        leafPool_.get(),
        MAP(VARCHAR(), DOUBLE()),
        nullptr,
        kRowsPerBatch,
        offsets,
        sizes,
        keys,
        values);
    if (hasContainerNulls) {
      map->setNull(5, true);
    }
    return map;
  };

  auto findStringKeyEntry =
      [&](const MapVector* map,
          vector_size_t row,
          std::string_view key) -> std::optional<vector_size_t> {
    if (map->isNullAt(row)) {
      return std::nullopt;
    }
    const auto* keys = map->mapKeys()->as<FlatVector<StringView>>();
    const auto offset = map->offsetAt(row);
    const auto size = map->sizeAt(row);
    for (vector_size_t i = 0; i < size; ++i) {
      const auto entry = offset + i;
      if (keys->valueAt(entry).str() == key) {
        return entry;
      }
    }
    return std::nullopt;
  };

  auto flatMapFieldVector =
      [&](const RowVector* root,
          const FlatMapLeafPath& path,
          vector_size_t row) -> std::pair<const BaseVector*, vector_size_t> {
    const auto* map =
        root->childAt(childIndex(root, "flatAttrs"))->as<MapVector>();
    const auto entry = findStringKeyEntry(map, row, path.key);
    NIMBLE_CHECK(entry.has_value(), "Expected FlatMap key in test data");
    return {map->mapValues().get(), *entry};
  };

  auto flatMapFieldIsNull = [&](const RowVector* root,
                                const FlatMapLeafPath& path,
                                vector_size_t row) {
    if (root->isNullAt(row)) {
      return true;
    }
    const auto* map =
        root->childAt(childIndex(root, "flatAttrs"))->as<MapVector>();
    const auto entry = findStringKeyEntry(map, row, path.key);
    if (!entry.has_value()) {
      return true;
    }
    return map->mapValues()->isNullAt(*entry);
  };

  auto expectFlatMapLeafEqual = [&](const RowVector* expectedRoot,
                                    vector_size_t expectedRow,
                                    const RowVector* actualRoot,
                                    vector_size_t actualRow,
                                    const FlatMapLeafPath& path) {
    const auto expectedNull =
        flatMapFieldIsNull(expectedRoot, path, expectedRow);
    EXPECT_EQ(flatMapFieldIsNull(actualRoot, path, actualRow), expectedNull)
        << "path=" << path.path << " expectedRow=" << expectedRow
        << " actualRow=" << actualRow;
    if (expectedNull) {
      return;
    }

    const auto [expected, expectedEntry] =
        flatMapFieldVector(expectedRoot, path, expectedRow);
    const auto [actual, actualEntry] =
        flatMapFieldVector(actualRoot, path, actualRow);
    EXPECT_EQ(
        expected->as<FlatVector<double>>()->valueAt(expectedEntry),
        actual->as<FlatVector<double>>()->valueAt(actualEntry));
  };

  auto expectRegularMapEqual = [&](const RowVector* expectedRoot,
                                   vector_size_t expectedRow,
                                   const RowVector* actualRoot,
                                   vector_size_t actualRow) {
    const auto* expectedMap =
        expectedRoot->childAt(childIndex(expectedRoot, "attrs"))
            ->as<MapVector>();
    const auto* actualMap =
        actualRoot->childAt(childIndex(actualRoot, "attrs"))->as<MapVector>();
    const auto expectedNull = expectedRoot->isNullAt(expectedRow) ||
        expectedMap->isNullAt(expectedRow);
    const auto actualNull =
        actualRoot->isNullAt(actualRow) || actualMap->isNullAt(actualRow);
    EXPECT_EQ(actualNull, expectedNull)
        << "expectedRow=" << expectedRow << " actualRow=" << actualRow;
    if (expectedNull) {
      return;
    }
    EXPECT_TRUE(actualMap->equalValueAt(expectedMap, actualRow, expectedRow))
        << "expectedRow=" << expectedRow << " actualRow=" << actualRow;
  };

  const auto seed = folly::Random::rand32();
  LOG(INFO) << "projectedComplexNullFuzzer seed: " << seed;
  folly::detail::DefaultGenerator rng{seed};

  for (int iteration = 0; iteration < kIterations; ++iteration) {
    SCOPED_TRACE(fmt::format("iteration {}", iteration));

    std::vector<size_t> selectedRowIndices;
    auto addRowIndex = [&](size_t index) {
      for (const auto selected : selectedRowIndices) {
        if (selected == index) {
          return;
        }
      }
      selectedRowIndices.push_back(index);
    };
    addRowIndex(0);
    for (size_t i = 0; i < rowLeafPaths.size(); ++i) {
      if (folly::Random::oneIn(2, rng)) {
        addRowIndex(i);
      }
    }
    if (selectedRowIndices.empty()) {
      addRowIndex(folly::Random::rand32(rng) % rowLeafPaths.size());
    }
    bool hasTwoLevelNestedProjection = false;
    for (const auto index : selectedRowIndices) {
      hasTwoLevelNestedProjection |= rowLeafPaths[index].names.size() > 2;
    }
    if (!hasTwoLevelNestedProjection) {
      addRowIndex(1 + folly::Random::rand32(rng) % 2);
    }

    std::vector<std::string_view> selectedFlatMapKeys = {"a"};
    if (folly::Random::oneIn(2, rng)) {
      selectedFlatMapKeys.emplace_back("b");
    }

    std::vector<Subfield> subfields;
    std::vector<const RowLeafPath*> selectedRowPaths;
    std::vector<const FlatMapLeafPath*> selectedFlatMapPaths;
    subfields.emplace_back("attrs");
    for (const auto index : selectedRowIndices) {
      subfields.emplace_back(rowLeafPaths[index].path);
      selectedRowPaths.push_back(&rowLeafPaths[index]);
    }
    for (const auto& path : flatMapLeafPaths) {
      for (const auto selectedKey : selectedFlatMapKeys) {
        if (path.key == selectedKey) {
          subfields.emplace_back(path.path);
          selectedFlatMapPaths.push_back(&path);
          break;
        }
      }
    }

    std::vector<RowVectorPtr> inputs;
    inputs.reserve(kNumBatches);
    for (int batchIndex = 0; batchIndex < kNumBatches; ++batchIndex) {
      const auto hasContainerNulls = batchIndex % 2 == 1;
      VectorFuzzer fuzzer(
          {
              .vectorSize = kRowsPerBatch,
              .nullRatio = 0,
              .useRandomNullPattern = true,
              .stringLength = 12,
              .stringVariableLength = true,
              .containerLength = 3,
              .containerVariableLength = true,
              .normalizeMapKeys = true,
          },
          leafPool_.get(),
          folly::Random::rand32(rng));
      auto fuzzed = makeFlatInputRow(fuzzer);
      std::vector<VectorPtr> children;
      children.reserve(rowType->size());
      for (size_t child = 0; child < rowType->size(); ++child) {
        children.push_back(fuzzed->childAt(child));
      }
      const auto keyBase = static_cast<int64_t>(batchIndex) * kRowsPerBatch;
      children[childIndex(fuzzed.get(), "key")] =
          vectorMaker_->flatVector<int64_t>(
              kRowsPerBatch, [keyBase](auto row) { return keyBase + row; });
      children[childIndex(fuzzed.get(), "attrs")] =
          makeStringDoubleMap(hasContainerNulls, 1000 + batchIndex * 100);
      children[childIndex(fuzzed.get(), "flatAttrs")] =
          makeStringDoubleMap(hasContainerNulls, 2000 + batchIndex * 100);
      auto input = std::make_shared<RowVector>(
          leafPool_.get(),
          rowType,
          nullptr,
          kRowsPerBatch,
          std::move(children));
      auto* profile =
          input->childAt(childIndex(input.get(), "profile"))->as<RowVector>();
      profile->childAt(childIndex(profile, "score"))->setNull(1, true);
      if (hasContainerNulls) {
        auto* details =
            profile->childAt(childIndex(profile, "details"))->as<RowVector>();
        details->setNull(2, true);
        auto* activity = input->childAt(childIndex(input.get(), "activity"))
                             ->as<RowVector>();
        auto* inner =
            activity->childAt(childIndex(activity, "inner"))->as<RowVector>();
        inner->setNull(3, true);
      }
      inputs.push_back(std::move(input));
    }

    writeData(inputs, {"key"}, {{"flatAttrs", {}}}, /*stripeSize=*/1);

    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, kRows);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    auto result = projector->project(request, {});

    ASSERT_EQ(result.responses.size(), 1);
    ASSERT_EQ(result.responses[0].slices.size(), kNumBatches);
    for (int batchIndex = 0; batchIndex < kNumBatches; ++batchIndex) {
      const auto& slice = result.responses[0].slices[batchIndex];
      EXPECT_EQ(readEmbeddedRowRange(slice), RowRange(0, kRowsPerBatch));
      EXPECT_EQ(
          readEmbeddedTabletChunkHeader(slice).requiresNullBarrier,
          batchIndex % 2 == 1);
    }

    std::vector<folly::IOBuf> owned;
    std::vector<std::string_view> batches;
    owned.reserve(result.responses[0].slices.size());
    batches.reserve(result.responses[0].slices.size());
    for (const auto& slice : result.responses[0].slices) {
      owned.push_back(coalesceChunkSlice(slice));
      batches.emplace_back(
          reinterpret_cast<const char*>(owned.back().data()),
          owned.back().length());
    }

    const auto output = deserializeProjectedBatches(*projector, batches);
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->size(), kRows);
    const auto* outputRow = output->as<RowVector>();
    ASSERT_NE(outputRow, nullptr);

    for (int batchIndex = 0; batchIndex < kNumBatches; ++batchIndex) {
      const auto* input = inputs[batchIndex].get();
      const auto outputOffset = batchIndex * kRowsPerBatch;
      for (vector_size_t row = 0; row < kRowsPerBatch; ++row) {
        const auto outputRowIndex = outputOffset + row;
        SCOPED_TRACE(
            fmt::format(
                "batch {} row {} outputRow {}",
                batchIndex,
                row,
                outputRowIndex));
        expectRegularMapEqual(input, row, outputRow, outputRowIndex);
        for (const auto* path : selectedRowPaths) {
          expectLeafEqual(input, row, outputRow, outputRowIndex, *path);
        }
        for (const auto* path : selectedFlatMapPaths) {
          expectFlatMapLeafEqual(input, row, outputRow, outputRowIndex, *path);
        }
      }
    }
  }
}

// Round-trips a single-batch deserialize over a key range. The Deserializer
// honors the per-batch row range at decode time (via skip+next), so the
// decoded output contains exactly `upperKey_ - lowerKey_` rows with values
// that match the source stripe's in-range subset. Used by the four
// positional range tests below (start / middle / end / full).
//
// Defined as a macro (rather than a function) so it can access the protected
// fixture members from inside the TEST_P body.
#define RUN_DESERIALIZER_RANGE_TEST(lowerKey_, upperKey_)                      \
  do {                                                                         \
    auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});               \
    const int numRows = 100;                                                   \
    std::vector<int64_t> keys(numRows);                                        \
    std::vector<int32_t> values(numRows);                                      \
    for (int i = 0; i < numRows; ++i) {                                        \
      keys[i] = i;                                                             \
      values[i] = i * 10;                                                      \
    }                                                                          \
    auto batch = vectorMaker_->rowVector(                                      \
        {"key", "value"},                                                      \
        {vectorMaker_->flatVector<int64_t>(keys),                              \
         vectorMaker_->flatVector<int32_t>(values)});                          \
    writeData({batch}, {"key"});                                               \
    std::vector<Subfield> subfields;                                           \
    subfields.emplace_back("value");                                           \
    auto projector = createProjector(subfields);                               \
    auto bounds = makeRangeLookup(rowType, {"key"}, (lowerKey_), (upperKey_)); \
    NimbleIndexProjector::Request request;                                     \
    request.keyBounds = {bounds};                                              \
    auto result = projector->project(request, {});                             \
    ASSERT_EQ(result.responses[0].slices.size(), 1);                           \
    const auto expectedOffset = static_cast<size_t>(lowerKey_);                \
    const auto expectedEnd = static_cast<size_t>(upperKey_);                   \
    std::vector<int32_t> expectedValues(                                       \
        values.begin() + expectedOffset, values.begin() + expectedEnd);        \
    auto expected = vectorMaker_->rowVector(                                   \
        {"value"}, {vectorMaker_->flatVector<int32_t>(expectedValues)});       \
    expectVectorRows(                                                          \
        deserializeProjectedSlice(*projector, result.responses[0].slices[0]),  \
        expected);                                                             \
  } while (0)

TEST_P(NimbleIndexProjectorTest, deserializerRangeAtStart) {
  // Range covers the first 30 rows of a 100-row stripe: [0, 30).
  RUN_DESERIALIZER_RANGE_TEST(0, 30);
}

TEST_P(NimbleIndexProjectorTest, deserializerRangeInMiddle) {
  // Range covers a 30-row window strictly inside the stripe: [30, 60).
  RUN_DESERIALIZER_RANGE_TEST(30, 60);
}

TEST_P(NimbleIndexProjectorTest, deserializerRangeAtEnd) {
  // Range covers the last 30 rows of a 100-row stripe: [70, 100).
  RUN_DESERIALIZER_RANGE_TEST(70, 100);
}

TEST_P(NimbleIndexProjectorTest, deserializerRangeFullStripe) {
  // Range spans the entire stripe: [0, 100). Validates round-trip on a
  // full-stripe row range (boundary case where the decoded output equals
  // the source stripe verbatim).
  RUN_DESERIALIZER_RANGE_TEST(0, 100);
}

#undef RUN_DESERIALIZER_RANGE_TEST

// Barrier + windowed kTablet + caller rowRange together. The projector marks a
// slice `requiresNullBarrier` whenever a Row/FlatMap null stream is physically
// present, so this combination is production-shaped rather than synthetic.
TEST_P(NimbleIndexProjectorTest, deserializerRowRangeOnBarrierWindowedSlice) {
  auto nestedType = ROW({"b"}, {INTEGER()});
  auto rowType = ROW({"key", "s"}, {BIGINT(), nestedType});
  constexpr int kStripeRows = 100;
  std::vector<int64_t> keys(kStripeRows);
  std::vector<int32_t> bs(kStripeRows);
  for (int i = 0; i < kStripeRows; ++i) {
    keys[i] = i;
    bs[i] = i * 10;
  }
  auto nested = std::make_shared<RowVector>(
      leafPool_.get(),
      nestedType,
      nullptr,
      kStripeRows,
      std::vector<VectorPtr>{vectorMaker_->flatVector<int32_t>(bs)});
  // Real nulls inside the projected window make the slice a barrier.
  for (int row : {32, 41, 55}) {
    nested->setNull(row, true);
  }
  auto batch = std::make_shared<RowVector>(
      leafPool_.get(),
      rowType,
      nullptr,
      kStripeRows,
      std::vector<VectorPtr>{vectorMaker_->flatVector<int64_t>(keys), nested});
  writeData({batch}, {"key"});

  std::vector<Subfield> subfields;
  subfields.emplace_back("s");
  auto projector = createProjector(subfields);
  auto bounds = makeRangeLookup(rowType, {"key"}, kWindowStart, kWindowEnd);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  auto result = projector->project(request, {});
  ASSERT_EQ(result.responses[0].slices.size(), 1);
  const auto& slice = result.responses[0].slices[0];
  const auto header = readEmbeddedTabletChunkHeader(slice);
  ASSERT_EQ(header.rowRange.startRow, kWindowStart);
  ASSERT_EQ(header.rowRange.endRow, kWindowEnd);
  // Without this the test would silently cover only the ordinary path.
  ASSERT_TRUE(header.requiresNullBarrier);

  // Caller [0, 10) narrows the [30, 60) window to stripe rows [30, 40), which
  // spans the null at row 32. Compare against the unranged decode sliced the
  // same way, so nulls and values are both checked.
  auto full = deserializeProjectedSlice(*projector, slice);
  ASSERT_EQ(full->size(), kWindowRows);
  auto expected = full->slice(0, 10);
  expectVectorRows(
      deserializeProjectedSlice(*projector, slice, {RowRange{0, 10}}),
      expected);
}

// A caller rowRange composes with the header window instead of replacing it:
// it is relative to the window start, so it can only narrow.
TEST_P(NimbleIndexProjectorTest, deserializerRowRangeNarrowsHeaderWindow) {
  auto windowed = makeWindowedSlice();
  ASSERT_NO_FATAL_FAILURE(expectWindow(windowed));

  // [0, 10) is relative to the window, so it selects stripe rows [30, 40)
  // (values 300..390). Replacing the window would have yielded [0, 10).
  std::vector<int32_t> expectedValues(
      windowed.values.begin() + kWindowStart,
      windowed.values.begin() + kWindowStart + 10);
  auto expected = vectorMaker_->rowVector(
      {"value"}, {vectorMaker_->flatVector<int32_t>(expectedValues)});
  expectVectorRows(
      deserializeProjectedSlice(
          *windowed.projector, windowed.slice(), {RowRange{0, 10}}),
      expected);
}

// The full window is addressable; one row past it is not.
TEST_P(NimbleIndexProjectorTest, deserializerRowRangeAcceptsWholeWindow) {
  auto windowed = makeWindowedSlice();
  ASSERT_NO_FATAL_FAILURE(expectWindow(windowed));

  std::vector<int32_t> expectedValues(
      windowed.values.begin() + kWindowStart,
      windowed.values.begin() + kWindowEnd);
  auto expected = vectorMaker_->rowVector(
      {"value"}, {vectorMaker_->flatVector<int32_t>(expectedValues)});
  expectVectorRows(
      deserializeProjectedSlice(
          *windowed.projector, windowed.slice(), {RowRange{0, kWindowRows}}),
      expected);
}

// The window, not the stripe, bounds a caller rowRange. Every one of these
// fits inside the 100-row stripe, so validating against the batch rowCount
// (the pre-composition contract) would have accepted them all and silently
// returned rows outside the window the slice was scoped to.
TEST_P(NimbleIndexProjectorTest, deserializerRowRangeRejectsRangePastWindow) {
  auto windowed = makeWindowedSlice();
  ASSERT_NO_FATAL_FAILURE(expectWindow(windowed));

  const std::vector<RowRange> pastWindow{
      // One row too many, starting at the window start.
      RowRange{0, kWindowRows + 1},
      // Starts inside the window but runs past its end.
      RowRange{kWindowRows - 5, kWindowRows + 5},
      // Starts at the first row past the window.
      RowRange{kWindowRows, kWindowRows + 1},
      // Entirely beyond the window.
      RowRange{kWindowRows + 10, kWindowRows + 20},
  };
  for (const auto& range : pastWindow) {
    SCOPED_TRACE(range.toString());
    NIMBLE_ASSERT_THROW(
        deserializeProjectedSlice(
            *windowed.projector, windowed.slice(), {range}),
        "rowRange endRow exceeds the rows this batch exposes");
  }
}

TEST_P(NimbleIndexProjectorTest, deserializerMultiBatchWithRowRange) {
  // Each input batch's embedded row range is applied per-segment by the
  // Deserializer; the output is the concatenation of each batch's in-range
  // rows. Same chunk slice passed twice yields 2 × range.numRows() output
  // rows, with values[10..39] repeated.
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  const int numRows = 100;
  std::vector<int64_t> keys(numRows);
  std::vector<int32_t> values(numRows);
  for (int i = 0; i < numRows; ++i) {
    keys[i] = i;
    values[i] = i * 10;
  }
  auto batch = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>(keys),
       vectorMaker_->flatVector<int32_t>(values)});
  writeData({batch}, {"key"});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  auto bounds = makeRangeLookup(rowType, {"key"}, 10, 40);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  auto result = projector->project(request, {});
  ASSERT_EQ(result.responses[0].slices.size(), 1);

  auto coalesced = result.responses[0].slices[0].cloneCoalescedAsValue();
  auto bytes = std::string_view(
      reinterpret_cast<const char*>(coalesced.data()), coalesced.length());

  DeserializerOptions deserOptions;
  deserOptions.hasHeader = true;
  Deserializer deserializer(
      projector->projectedNimbleType(), leafPool_.get(), deserOptions);

  std::vector<std::string_view> multiBatch{bytes, bytes};
  VectorPtr output;
  deserializer.deserialize(multiBatch, output);

  // Each batch's [10, 40) range is honored at decode time, so the output
  // is 2 × 30 = 60 rows — values[10..39] repeated.
  ASSERT_EQ(output->size(), 60u);
  auto* rowVec = output->as<velox::RowVector>();
  auto* valueVec = rowVec->childAt(0)->as<velox::FlatVector<int32_t>>();
  for (uint32_t i = 0; i < 30; ++i) {
    const int32_t expected = static_cast<int32_t>((10 + i) * 10);
    EXPECT_EQ(valueVec->valueAt(i), expected) << "batch=0 i=" << i;
    EXPECT_EQ(valueVec->valueAt(30 + i), expected) << "batch=1 i=" << i;
  }
}

TEST_P(NimbleIndexProjectorTest, deserializerMultiBatchMixedRanges) {
  // Multi-batch deserialize where each batch carries a different positional
  // row range: start [0, 20), middle [30, 50), end [80, 100), and full
  // [0, 100). Output is the concatenation of in-range rows from each batch,
  // preserving order.
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  const int numRows = 100;
  std::vector<int64_t> keys(numRows);
  std::vector<int32_t> values(numRows);
  for (int i = 0; i < numRows; ++i) {
    keys[i] = i;
    values[i] = i * 10;
  }
  auto batch = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>(keys),
       vectorMaker_->flatVector<int32_t>(values)});
  writeData({batch}, {"key"});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // One projector call with four independent range requests; each produces
  // its own chunk slice (one per request × stripe intersection).
  NimbleIndexProjector::Request request;
  request.keyBounds = {
      makeRangeLookup(rowType, {"key"}, 0, 20),
      makeRangeLookup(rowType, {"key"}, 30, 50),
      makeRangeLookup(rowType, {"key"}, 80, 100),
      makeRangeLookup(rowType, {"key"}, 0, 100),
  };
  auto result = projector->project(request, {});
  ASSERT_EQ(result.responses.size(), 4);

  // Coalesce each response's chunk slice into a contiguous buffer the
  // deserializer can read from. The owned IOBufs must outlive the
  // string_views below — keep them in a parallel vector.
  std::vector<folly::IOBuf> ownedSlices;
  ownedSlices.reserve(4);
  std::vector<std::string_view> batches;
  batches.reserve(4);
  for (auto& response : result.responses) {
    ASSERT_EQ(response.slices.size(), 1u);
    ownedSlices.push_back(response.slices[0].cloneCoalescedAsValue());
    batches.emplace_back(
        reinterpret_cast<const char*>(ownedSlices.back().data()),
        ownedSlices.back().length());
  }

  DeserializerOptions deserOptions;
  deserOptions.hasHeader = true;
  Deserializer deserializer(
      projector->projectedNimbleType(), leafPool_.get(), deserOptions);
  VectorPtr output;
  deserializer.deserialize(batches, output);

  // Each batch is decoded to its embedded row range at ingest, so the
  // concatenated output is 20 + 20 + 20 + 100 = 160 rows.
  ASSERT_EQ(output->size(), 160u);
  auto* rowVec = output->as<velox::RowVector>();
  auto* valueVec = rowVec->childAt(0)->as<velox::FlatVector<int32_t>>();
  // Batch 0: range [0, 20) → values[0..19] at output offset 0.
  for (uint32_t i = 0; i < 20; ++i) {
    EXPECT_EQ(valueVec->valueAt(i), static_cast<int32_t>(i * 10))
        << "batch=0 i=" << i;
  }
  // Batch 1: range [30, 50) → values[30..49] at output offset 20.
  for (uint32_t i = 0; i < 20; ++i) {
    EXPECT_EQ(valueVec->valueAt(20 + i), static_cast<int32_t>((30 + i) * 10))
        << "batch=1 i=" << i;
  }
  // Batch 2: range [80, 100) → values[80..99] at output offset 40.
  for (uint32_t i = 0; i < 20; ++i) {
    EXPECT_EQ(valueVec->valueAt(40 + i), static_cast<int32_t>((80 + i) * 10))
        << "batch=2 i=" << i;
  }
  // Batch 3: range [0, 100) → all 100 values at output offset 60.
  for (uint32_t i = 0; i < 100; ++i) {
    EXPECT_EQ(valueVec->valueAt(60 + i), static_cast<int32_t>(i * 10))
        << "batch=3 i=" << i;
  }
}

TEST_P(NimbleIndexProjectorTest, deserializerMultiBatchMixedVersions) {
  // Mix a kTablet chunk slice (projector output, narrow row range) with
  // a kLegacyCompact chunk (Serializer output, no row range) in one deserialize
  // call. Verifies the no-range branch of the per-batch loop: when
  // batchRanges[i] is nullopt, numRowsInBatch falls back to batchRows
  // (the full kLegacyCompact batch).
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  const int numRows = 100;
  std::vector<int64_t> keys(numRows);
  std::vector<int32_t> values(numRows);
  for (int i = 0; i < numRows; ++i) {
    keys[i] = i;
    values[i] = i * 10;
  }
  auto batch = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>(keys),
       vectorMaker_->flatVector<int32_t>(values)});
  writeData({batch}, {"key"});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // kTablet batch: range [10, 40) = 30 rows.
  auto bounds = makeRangeLookup(rowType, {"key"}, 10, 40);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  auto result = projector->project(request, {});
  ASSERT_EQ(result.responses[0].slices.size(), 1u);
  auto tabletOwned = result.responses[0].slices[0].cloneCoalescedAsValue();

  // kLegacyCompact batch: serialize a 5-row Row<INTEGER> vector matching the
  // projected schema. The Serializer is constructed against the same velox
  // type the Deserializer will use, so the streams encode in lockstep.
  auto projectedVeloxType =
      convertToVeloxType(*projector->projectedNimbleType());
  std::vector<int32_t> kLegacyCompactValues = {1000, 2000, 3000, 4000, 5000};
  auto kLegacyCompactRow = vectorMaker_->rowVector(
      {"value"}, {vectorMaker_->flatVector<int32_t>(kLegacyCompactValues)});
  SerializerOptions serOptions{
      .version = SerializationVersion::kSerialization,
  };
  Serializer ser{
      serOptions,
      projectedVeloxType,
      leafPool_.get(),
  };
  auto kLegacyCompactBytes = ser.serialize(
      kLegacyCompactRow,
      OrderedRanges::of(0, static_cast<uint32_t>(kLegacyCompactValues.size())));

  DeserializerOptions deserOptions;
  deserOptions.hasHeader = true;
  Deserializer deserializer(
      projector->projectedNimbleType(), leafPool_.get(), deserOptions);

  std::vector<std::string_view> batches{
      std::string_view(
          reinterpret_cast<const char*>(tabletOwned.data()),
          tabletOwned.length()),
      kLegacyCompactBytes,
  };
  VectorPtr output;
  deserializer.deserialize(batches, output);

  // kTablet batch is honored to its [10, 40) range = 30 rows; kLegacyCompact
  // batch has no rowRange and contributes its full 5 rows. Concatenation =
  // 35 rows: values[10..39] followed by kLegacyCompactValues[0..4].
  ASSERT_EQ(output->size(), 35u);
  auto* rowVec = output->as<velox::RowVector>();
  auto* valueVec = rowVec->childAt(0)->as<velox::FlatVector<int32_t>>();
  for (uint32_t i = 0; i < 30; ++i) {
    EXPECT_EQ(valueVec->valueAt(i), static_cast<int32_t>((10 + i) * 10))
        << "kTablet i=" << i;
  }
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(valueVec->valueAt(30 + i), kLegacyCompactValues[i])
        << "kLegacyCompact i=" << i;
  }
}

TEST_P(NimbleIndexProjectorTest, stats) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  // Write 4 batches of 10 rows each. The flush policy flushes every batch,
  // producing exactly 4 stripes with deterministic row counts.
  const int numBatches = 4;
  const int rowsPerBatch = 10;
  const int numRows = numBatches * rowsPerBatch;
  std::vector<RowVectorPtr> batches;
  for (int b = 0; b < numBatches; ++b) {
    std::vector<int64_t> keys(rowsPerBatch);
    std::vector<int32_t> values(rowsPerBatch);
    for (int i = 0; i < rowsPerBatch; ++i) {
      keys[i] = b * rowsPerBatch + i;
      values[i] = keys[i] * 10;
    }
    batches.emplace_back(vectorMaker_->rowVector(
        {"key", "value"},
        {vectorMaker_->flatVector<int64_t>(keys),
         vectorMaker_->flatVector<int32_t>(values)}));
  }

  // Flush every batch to get exactly numBatches stripes.
  writeData(batches, {"key"}, {}, /*stripeSize=*/1);

  std::vector<Subfield> subs;
  subs.emplace_back("value");

  // Empty request should throw.
  {
    auto proj = createProjector(subs);
    NimbleIndexProjector::Request request;
    NIMBLE_ASSERT_THROW(
        proj->project(request, {}), "keyBounds must not be empty");
  }

  // Single point lookup: key=15 is in stripe 1 (rows 10..19).
  {
    auto proj = createProjector(subs);
    auto bounds = makePointLookup(rowType, {"key"}, 15);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    proj->project(request, {});

    EXPECT_EQ(proj->stats().numReadStripes, 1);
    EXPECT_EQ(proj->stats().numReadRows, rowsPerBatch);
    EXPECT_EQ(proj->stats().numProjectedRows, 1);

    // Timings should be non-zero for a hit.
    EXPECT_GT(proj->stats().lookupTiming.count, 0);
    EXPECT_GT(proj->stats().lookupTiming.wallNanos, 0);
    EXPECT_GT(proj->stats().prepareTiming.wallNanos, 0);
    EXPECT_GT(proj->stats().scanTiming.wallNanos, 0);
    EXPECT_GT(proj->stats().projectionTiming.wallNanos, 0);

    // IO stats should be non-zero for a hit.
    EXPECT_GT(dataIoStats_->rawBytesRead(), 0);
    EXPECT_GT(dataIoStats_->read().count(), 0);
  }

  // Two point lookups in first and last stripes.
  {
    auto proj = createProjector(subs);
    auto bounds0 = makePointLookup(rowType, {"key"}, 0);
    auto bounds1 = makePointLookup(rowType, {"key"}, numRows - 1);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds0, bounds1};
    proj->project(request, {});

    EXPECT_EQ(proj->stats().numReadStripes, 2);
    EXPECT_EQ(
        proj->stats().numReadRows, static_cast<uint64_t>(2 * rowsPerBatch));
    EXPECT_EQ(proj->stats().numProjectedRows, 2);

    // IO stats should reflect more data read than single lookup.
    EXPECT_GT(dataIoStats_->rawBytesRead(), 0);
  }

  // Point lookup for a missing key (beyond max) should yield zero stats.
  {
    auto proj = createProjector(subs);
    auto bounds = makePointLookup(rowType, {"key"}, numRows + 1'000);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    proj->project(request, {});

    EXPECT_EQ(proj->stats().numReadStripes, 0);
    EXPECT_EQ(proj->stats().numReadRows, 0);
    EXPECT_EQ(proj->stats().numProjectedRows, 0);
    EXPECT_EQ(dataIoStats_->rawBytesRead(), 0);
  }
}

TEST_P(NimbleIndexProjectorTest, localReadModeStats) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  constexpr int numBatches = 4;
  constexpr int rowsPerBatch = 128;
  constexpr int numRows = numBatches * rowsPerBatch;
  std::vector<RowVectorPtr> inputBatches;
  std::vector<int32_t> expectedValues(numRows);
  inputBatches.reserve(numBatches);
  for (int batch = 0; batch < numBatches; ++batch) {
    std::vector<int64_t> keys(rowsPerBatch);
    std::vector<int32_t> values(rowsPerBatch);
    for (int row = 0; row < rowsPerBatch; ++row) {
      const auto key = batch * rowsPerBatch + row;
      keys[row] = key;
      values[row] = key * 10;
      expectedValues[key] = values[row];
    }
    inputBatches.emplace_back(vectorMaker_->rowVector(
        {"key", "value"},
        {vectorMaker_->flatVector<int64_t>(keys),
         vectorMaker_->flatVector<int32_t>(values)}));
  }
  writeData(inputBatches, {"key"}, {}, /*stripeSize=*/1);

  auto tempFile = velox::common::testutil::TempFilePath::create();
  {
    velox::LocalWriteFile writeFile(
        tempFile->getPath(),
        /*shouldCreateParentDirectories=*/false,
        /*shouldThrowOnFileAlreadyExists=*/false);
    writeFile.append(sinkData_);
    writeFile.close();
  }

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  struct TestCase {
    const char* name;
    bool bufferIo;
    bool useIoUring;
  };
  std::vector<TestCase> testCases = {
      {"buffered", /*bufferIo=*/true, /*useIoUring=*/false},
      {"direct", /*bufferIo=*/false, /*useIoUring=*/false},
  };
  // LocalReadFile rejects a useIoUring request outright when io_uring is
  // unavailable, which is the case whenever folly was built without liburing.
  // Velox neither declares nor installs liburing, so a stock OSS build never
  // has it. The cache assertions below are expressed in terms of
  // testCases.size(), so the mode is dropped rather than skipped in the loop.
  if (velox::IoUringReader::available()) {
    testCases.push_back(
        {"directIoUring", /*bufferIo=*/false, /*useIoUring=*/true});
  }

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    velox::ThreadLocalIoUringReader::testingClear();
    uint64_t numIoUringReaders{0};
    EXPECT_EQ(velox::getIoUringReaderStats(numIoUringReaders).readCalls, 0);
    EXPECT_EQ(numIoUringReaders, 0);

    auto tabletReadFile = std::make_shared<velox::LocalReadFile>(
        tempFile->getPath(),
        /*executor=*/nullptr,
        /*bufferIo=*/true,
        /*useIoUring=*/false);
    auto dataReadFile = std::make_shared<velox::LocalReadFile>(
        tempFile->getPath(),
        /*executor=*/nullptr,
        testCase.bufferIo,
        testCase.useIoUring);
    const auto fileName = dataReadFile->getName();
    dwio::common::ReaderOptions tabletReaderOptions(leafPool_.get());
    tabletReaderOptions.setFileFormat(FileFormat::NIMBLE);
    tabletReaderOptions.setLoadClusterIndex(true);
    tabletReaderOptions.setCacheData(false);
    tabletReaderOptions.setCacheMetadata(GetParam().cacheMetadata);
    tabletReaderOptions.setPinMetadata(GetParam().pinMetadata);
    tabletReaderOptions.setIOExecutor(ioExecutor_);
    ensureTabletReaderCache();
    (void)tabletReaderCache_->get(
        tabletReadFile, TabletReader::configureOptions(tabletReaderOptions));
    auto projector = createProjectorWithFileHandle(
        subfields, makeFileHandle(dataReadFile, fileName));
    const auto fileBytesBeforeProject = dataReadFile->bytesRead();

    NimbleIndexProjector::Request request;
    request.keyBounds = {makeRangeLookup(rowType, {"key"}, 0, numRows)};
    auto result = projector->project(request, {});

    ASSERT_EQ(result.responses.size(), 1);
    ASSERT_EQ(result.responses[0].slices.size(), numBatches);
    std::vector<folly::IOBuf> ownedSlices;
    std::vector<std::string_view> serializedBatches;
    ownedSlices.reserve(numBatches);
    serializedBatches.reserve(numBatches);
    for (const auto& slice : result.responses[0].slices) {
      const auto rowRange = readEmbeddedRowRange(slice);
      EXPECT_EQ(rowRange, RowRange(0, rowsPerBatch));
      ownedSlices.push_back(slice.cloneCoalescedAsValue());
      serializedBatches.emplace_back(
          reinterpret_cast<const char*>(ownedSlices.back().data()),
          ownedSlices.back().length());
    }

    auto expected = vectorMaker_->rowVector(
        {"value"}, {vectorMaker_->flatVector<int32_t>(expectedValues)});
    expectVectorRows(
        deserializeProjectedBatches(*projector, serializedBatches), expected);

    EXPECT_EQ(projector->stats().numReadStripes, numBatches);
    EXPECT_EQ(projector->stats().numReadRows, numRows);
    EXPECT_EQ(projector->stats().numProjectedRows, numRows);
    EXPECT_GT(projector->stats().numOutputBytes, 0);

    const auto physicalBytes = dataIoStats_->read().sum();
    const auto payloadBytes = dataIoStats_->rawBytesRead();
    EXPECT_GT(dataIoStats_->read().count(), 0);
    EXPECT_GT(payloadBytes, 0);
    EXPECT_EQ(
        dataReadFile->bytesRead() - fileBytesBeforeProject, physicalBytes);
    EXPECT_EQ(dataIoStats_->rawOverreadBytes(), physicalBytes - payloadBytes);
    EXPECT_GE(physicalBytes, payloadBytes);
    uint64_t alignment{0};
    EXPECT_EQ(dataReadFile->directIo(alignment), !testCase.bufferIo);
    if (!testCase.bufferIo) {
      EXPECT_EQ(physicalBytes % alignment, 0);
      EXPECT_GE(physicalBytes, payloadBytes);
    } else {
      EXPECT_EQ(alignment, 1);
    }
    uint64_t tabletAlignment{0};
    EXPECT_FALSE(tabletReadFile->directIo(tabletAlignment));
    EXPECT_EQ(tabletAlignment, 1);

    const auto ioUringStats = velox::getIoUringReaderStats(numIoUringReaders);
    if (testCase.useIoUring) {
      EXPECT_EQ(numIoUringReaders, 1);
      EXPECT_EQ(ioUringStats.readCalls, 1);
      EXPECT_EQ(ioUringStats.regions, dataIoStats_->read().count());
      EXPECT_EQ(ioUringStats.batches, 1);
      EXPECT_EQ(ioUringStats.minRegionsPerRead, dataIoStats_->read().count());
      EXPECT_EQ(ioUringStats.maxRegionsPerRead, dataIoStats_->read().count());
      EXPECT_EQ(ioUringStats.minBatchSize, dataIoStats_->read().count());
      EXPECT_EQ(ioUringStats.maxBatchSize, dataIoStats_->read().count());
    } else {
      EXPECT_EQ(numIoUringReaders, 0);
      EXPECT_EQ(ioUringStats.readCalls, 0);
      EXPECT_EQ(ioUringStats.regions, 0);
      EXPECT_EQ(ioUringStats.batches, 0);
    }
  }
  velox::ThreadLocalIoUringReader::testingClear();
  EXPECT_EQ(tabletReaderCache_->stats().numElements, 1);
  EXPECT_EQ(tabletReaderCache_->stats().numLookups, 2 * testCases.size());
  EXPECT_EQ(tabletReaderCache_->stats().numHits, 2 * testCases.size() - 1);
}

TEST_P(NimbleIndexProjectorTest, statsToString) {
  NimbleIndexProjector::Stats stats;
  stats.numReadStripes = 3;
  stats.numSlicedStripes = 1;
  stats.numReadRows = 1'000;
  stats.numProjectedRows = 800;
  stats.numOutputBytes = 8'192;
  EXPECT_EQ(
      stats.toString(),
      "Stats(numReadStripes=3, numSlicedStripes=1, "
      "slicedStripePct=33.33%, numReadRows=1000, numProjectedRows=800, "
      "numOutputBytes=8.00KB, "
      "lookupTiming=[count: 0, wallTime: 0ns, cpuTime: 0ns], "
      "prepareTiming=[count: 0, wallTime: 0ns, cpuTime: 0ns], "
      "scanTiming=[count: 0, wallTime: 0ns, cpuTime: 0ns], "
      "projectionTiming=[count: 0, wallTime: 0ns, cpuTime: 0ns])");
}

TEST_P(NimbleIndexProjectorTest, flatMapProjection) {
  // Schema: key (int64, sorted), features (MAP<VARCHAR, BIGINT> as FlatMap).
  // FlatMap keys in the file: "a", "b", "c", "d", "e".
  // Project keys in non-alphabetical order: ["d", "b", "e"] to verify that
  // projectedNimbleType() stream ordering matches the serialized data.
  auto rowType = ROW({"key", "features"}, {BIGINT(), MAP(VARCHAR(), BIGINT())});

  const int numRows = 200;
  const std::vector<std::string> mapKeys = {"a", "b", "c", "d", "e"};
  const auto numMapKeys = static_cast<vector_size_t>(mapKeys.size());

  std::vector<StringView> mapKeyViews;
  mapKeyViews.reserve(mapKeys.size());
  for (const auto& k : mapKeys) {
    mapKeyViews.emplace_back(k);
  }

  auto batch = vectorMaker_->rowVector(
      {"key", "features"},
      {vectorMaker_->flatVector<int64_t>(
           numRows, [](auto row) { return row * 10; }),
       vectorMaker_->mapVector<StringView, int64_t>(
           numRows,
           /*sizeAt*/ [&](auto /*row*/) { return numMapKeys; },
           /*keyAt*/
           [&](auto /*row*/, auto mapIndex) { return mapKeyViews[mapIndex]; },
           /*valueAt*/
           [](auto row, auto mapIndex) {
             return static_cast<int64_t>(row * 100 + mapIndex);
           })});

  writeData({batch}, {"key"}, {{"features", {}}});

  // Project keys in non-alphabetical order to exercise FlatMap sorting.
  std::vector<Subfield> subfields;
  subfields.emplace_back("features[\"d\"]");
  subfields.emplace_back("features[\"b\"]");
  subfields.emplace_back("features[\"e\"]");
  auto projector = createProjector(subfields);

  // Point lookup for key=50 (row index 5).
  auto pointBounds = makePointLookup(rowType, {"key"}, 50);
  NimbleIndexProjector::Request request;
  request.keyBounds = {pointBounds};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 1);
  const auto& response = result.responses[0];
  ASSERT_FALSE(response.slices.empty());

  const auto& slice = response.slices[0];
  const auto rowRange = readEmbeddedRowRange(slice);
  ASSERT_GT(rowRange.numRows(), 0);

  // Deserialize using the projector's projected nimble type which accounts
  // for FlatMap key sorting (keys sorted alphabetically: "b", "d", "e").
  auto coalesced = coalesceChunkSlice(slice);
  DeserializerOptions deserOptions;
  deserOptions.hasHeader = true;
  Deserializer deserializer(
      projector->projectedNimbleType(), leafPool_.get(), deserOptions);

  VectorPtr deserialized;
  deserializer.deserialize(
      std::string_view(
          reinterpret_cast<const char*>(coalesced.data()), coalesced.length()),
      deserialized);
  ASSERT_NE(deserialized, nullptr);

  auto* rowResult = deserialized->as<RowVector>();
  ASSERT_NE(rowResult, nullptr);
  ASSERT_EQ(rowResult->childrenSize(), 1);

  // FlatMap is deserialized as a MapVector. The Deserializer honors the
  // header's row range at decode time, so the single point-lookup row
  // lands at output index 0.
  auto* mapResult = rowResult->childAt(0)->as<MapVector>();
  ASSERT_NE(mapResult, nullptr);
  ASSERT_EQ(mapResult->size(), rowRange.numRows());

  auto* mapKeyResults = mapResult->mapKeys()->as<FlatVector<StringView>>();
  auto* mapValues = mapResult->mapValues()->as<FlatVector<int64_t>>();
  ASSERT_NE(mapKeyResults, nullptr);
  ASSERT_NE(mapValues, nullptr);

  // Collect key-value pairs at the target row.
  const auto mapOffset = mapResult->offsetAt(0);
  const auto mapSize = mapResult->sizeAt(0);

  // row=5: key=50, features[k] = 5*100 + index_of(k).
  std::map<std::string, int64_t> kvPairs;
  for (vector_size_t i = 0; i < mapSize; ++i) {
    kvPairs[std::string(mapKeyResults->valueAt(mapOffset + i))] =
        mapValues->valueAt(mapOffset + i);
  }

  // Verify projected keys: "b"(idx=1), "d"(idx=3), "e"(idx=4).
  ASSERT_EQ(kvPairs.size(), 3);
  EXPECT_EQ(kvPairs["b"], 501); // 5*100 + 1
  EXPECT_EQ(kvPairs["d"], 503); // 5*100 + 3
  EXPECT_EQ(kvPairs["e"], 504); // 5*100 + 4
}

TEST_P(NimbleIndexProjectorTest, flatMapProjectionWithNarrowRowRange) {
  // Schema: key (int64, sorted), features (MAP<VARCHAR, BIGINT> as FlatMap).
  // Range scan over keys [50, 100) — 5 rows starting at source row index 5
  // (since keys are row*10). The Deserializer honors the header's row
  // range at decode time, so the output is exactly 5 rows.
  auto rowType = ROW({"key", "features"}, {BIGINT(), MAP(VARCHAR(), BIGINT())});

  const int numRows = 200;
  const std::vector<std::string> mapKeys = {"a", "b", "c"};
  const auto numMapKeys = static_cast<vector_size_t>(mapKeys.size());

  std::vector<StringView> mapKeyViews;
  mapKeyViews.reserve(mapKeys.size());
  for (const auto& k : mapKeys) {
    mapKeyViews.emplace_back(k);
  }

  auto batch = vectorMaker_->rowVector(
      {"key", "features"},
      {vectorMaker_->flatVector<int64_t>(
           numRows, [](auto row) { return row * 10; }),
       vectorMaker_->mapVector<StringView, int64_t>(
           numRows,
           /*sizeAt*/ [&](auto /*row*/) { return numMapKeys; },
           /*keyAt*/
           [&](auto /*row*/, auto mapIndex) { return mapKeyViews[mapIndex]; },
           /*valueAt*/
           [](auto row, auto mapIndex) {
             return static_cast<int64_t>(row * 100 + mapIndex);
           })});

  writeData({batch}, {"key"}, {{"features", {}}});

  std::vector<Subfield> subfields;
  subfields.emplace_back("features[\"a\"]");
  subfields.emplace_back("features[\"c\"]");
  auto projector = createProjector(subfields);

  // Range [50, 100) → rows 5..9 (5 rows).
  auto bounds = makeRangeLookup(rowType, {"key"}, 50, 100);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  auto result = projector->project(request, {});
  ASSERT_EQ(result.responses.size(), 1);
  ASSERT_EQ(result.responses[0].slices.size(), 1u);

  const auto& slice = result.responses[0].slices[0];
  const auto rowRange = readEmbeddedRowRange(slice);
  ASSERT_EQ(rowRange.numRows(), 5u);

  auto coalesced = coalesceChunkSlice(slice);
  DeserializerOptions deserOptions;
  deserOptions.hasHeader = true;
  Deserializer deserializer(
      projector->projectedNimbleType(), leafPool_.get(), deserOptions);

  VectorPtr deserialized;
  deserializer.deserialize(
      std::string_view(
          reinterpret_cast<const char*>(coalesced.data()), coalesced.length()),
      deserialized);
  ASSERT_NE(deserialized, nullptr);
  ASSERT_EQ(deserialized->size(), 5u);

  auto* rowResult = deserialized->as<RowVector>();
  ASSERT_NE(rowResult, nullptr);
  auto* mapResult = rowResult->childAt(0)->as<MapVector>();
  ASSERT_NE(mapResult, nullptr);
  ASSERT_EQ(mapResult->size(), 5u);

  auto* mapKeyResults = mapResult->mapKeys()->as<FlatVector<StringView>>();
  auto* mapValues = mapResult->mapValues()->as<FlatVector<int64_t>>();
  ASSERT_NE(mapKeyResults, nullptr);
  ASSERT_NE(mapValues, nullptr);

  // Verify each output row corresponds to source rows 5..9: both projected
  // FlatMap keys "a" (idx 0) and "c" (idx 2) are present with values
  // sourceRow*100 + index.
  for (vector_size_t outRow = 0; outRow < 5; ++outRow) {
    const int64_t sourceRow = 5 + outRow;
    const auto mapOffset = mapResult->offsetAt(outRow);
    const auto mapSize = mapResult->sizeAt(outRow);
    ASSERT_EQ(mapSize, 2) << "outRow=" << outRow;
    std::map<std::string, int64_t> kv;
    for (vector_size_t k = 0; k < mapSize; ++k) {
      kv[std::string(mapKeyResults->valueAt(mapOffset + k))] =
          mapValues->valueAt(mapOffset + k);
    }
    EXPECT_EQ(kv["a"], sourceRow * 100 + 0) << "outRow=" << outRow;
    EXPECT_EQ(kv["c"], sourceRow * 100 + 2) << "outRow=" << outRow;
  }
}

TEST_P(NimbleIndexProjectorTest, flatMapProjectionWithInterleavedMissingKeys) {
  auto rowType = ROW({"key", "features"}, {BIGINT(), MAP(VARCHAR(), BIGINT())});

  auto keyOrdinal = [](std::string_view key) -> int64_t {
    if (key == "a") {
      return 1;
    }
    if (key == "b") {
      return 2;
    }
    if (key == "c") {
      return 3;
    }
    NIMBLE_FAIL("Unexpected key {}", key);
  };

  std::vector<std::vector<std::vector<std::string>>> keysByBatch{
      {{"a", "b"}, {"a"}, {"b"}, {}, {"a", "b"}},
      {{"b"}, {"b", "c"}, {"c"}, {}, {"b"}},
      {{"a", "c"}, {"a"}, {"c"}, {}, {"a", "c"}},
      {{"a", "b", "c"}, {"b"}, {"a"}, {}, {"c"}},
  };

  std::vector<RowVectorPtr> batches;
  std::vector<std::map<std::string, int64_t>> expectedRows;
  batches.reserve(keysByBatch.size());
  for (size_t batchIndex = 0; batchIndex < keysByBatch.size(); ++batchIndex) {
    const auto& keysByRow = keysByBatch[batchIndex];
    const auto numRows = static_cast<vector_size_t>(keysByRow.size());
    const auto keyBase = static_cast<int64_t>(batchIndex * keysByRow.size());

    vector_size_t numEntries = 0;
    for (const auto& rowKeys : keysByRow) {
      numEntries += static_cast<vector_size_t>(rowKeys.size());
    }

    auto featureKeys = BaseVector::create<FlatVector<StringView>>(
        VARCHAR(), numEntries, leafPool_.get());
    auto featureValues = BaseVector::create<FlatVector<int64_t>>(
        BIGINT(), numEntries, leafPool_.get());

    auto offsets = allocateOffsets(numRows, leafPool_.get());
    auto sizes = allocateSizes(numRows, leafPool_.get());
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();

    vector_size_t entry = 0;
    for (vector_size_t row = 0; row < numRows; ++row) {
      rawOffsets[row] = entry;
      rawSizes[row] = static_cast<vector_size_t>(keysByRow[row].size());
      const auto globalRow = keyBase + row;
      auto& expected = expectedRows.emplace_back();
      for (const auto& key : keysByRow[row]) {
        featureKeys->set(entry, StringView(key));
        const auto value = globalRow * 100 + keyOrdinal(key);
        featureValues->set(entry, value);
        expected.emplace(key, value);
        ++entry;
      }
    }

    auto features = std::make_shared<MapVector>(
        leafPool_.get(),
        MAP(VARCHAR(), BIGINT()),
        nullptr,
        numRows,
        offsets,
        sizes,
        featureKeys,
        featureValues);
    batches.emplace_back(vectorMaker_->rowVector(
        {"key", "features"},
        {vectorMaker_->flatVector<int64_t>(
             numRows, [keyBase](auto row) { return keyBase + row; }),
         features}));
  }

  writeData(batches, {"key"}, {{"features", {}}}, /*stripeSize=*/1);

  std::vector<Subfield> subfields;
  subfields.emplace_back("features[\"a\"]");
  subfields.emplace_back("features[\"b\"]");
  subfields.emplace_back("features[\"c\"]");
  auto projector = createProjector(subfields);

  auto bounds = makeRangeLookup(
      rowType, {"key"}, 0, static_cast<int64_t>(expectedRows.size()));
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 1);
  ASSERT_EQ(result.responses[0].slices.size(), keysByBatch.size());

  std::vector<folly::IOBuf> owned;
  std::vector<std::string_view> serializedBatches;
  owned.reserve(result.responses[0].slices.size());
  serializedBatches.reserve(result.responses[0].slices.size());
  for (const auto& slice : result.responses[0].slices) {
    owned.push_back(coalesceChunkSlice(slice));
    serializedBatches.emplace_back(
        reinterpret_cast<const char*>(owned.back().data()),
        owned.back().length());
  }

  auto output = deserializeProjectedBatches(*projector, serializedBatches);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->size(), expectedRows.size());

  auto* rowResult = output->as<RowVector>();
  ASSERT_NE(rowResult, nullptr);
  auto* features = rowResult->childAt(0)->as<MapVector>();
  ASSERT_NE(features, nullptr);
  auto* featureKeys = features->mapKeys()->as<FlatVector<StringView>>();
  auto* featureValues = features->mapValues()->as<FlatVector<int64_t>>();
  ASSERT_NE(featureKeys, nullptr);
  ASSERT_NE(featureValues, nullptr);

  for (vector_size_t row = 0; row < output->size(); ++row) {
    SCOPED_TRACE(fmt::format("row {}", row));
    std::map<std::string, int64_t> actual;
    const auto offset = features->offsetAt(row);
    const auto size = features->sizeAt(row);
    for (vector_size_t i = 0; i < size; ++i) {
      const auto entry = offset + i;
      actual.emplace(
          std::string(featureKeys->valueAt(entry)),
          featureValues->valueAt(entry));
    }
    EXPECT_EQ(actual, expectedRows[row]);
  }
}

TEST_P(NimbleIndexProjectorTest, flatMapFullColumnProjection) {
  // Full FlatMap projection (no key subscripts) is not supported —
  // buildProjectedNimbleType requires explicit key selection for FlatMap.
  auto rowType = ROW({"key", "features"}, {BIGINT(), MAP(VARCHAR(), BIGINT())});

  const int numRows = 200;
  const std::vector<std::string> mapKeys = {"a", "b", "c", "d", "e"};
  const auto numMapKeys = static_cast<vector_size_t>(mapKeys.size());

  std::vector<StringView> mapKeyViews;
  mapKeyViews.reserve(mapKeys.size());
  for (const auto& k : mapKeys) {
    mapKeyViews.emplace_back(k);
  }

  auto batch = vectorMaker_->rowVector(
      {"key", "features"},
      {vectorMaker_->flatVector<int64_t>(
           numRows, [](auto row) { return row * 10; }),
       vectorMaker_->mapVector<StringView, int64_t>(
           numRows,
           /*sizeAt*/ [&](auto /*row*/) { return numMapKeys; },
           /*keyAt*/
           [&](auto /*row*/, auto mapIndex) { return mapKeyViews[mapIndex]; },
           /*valueAt*/
           [](auto row, auto mapIndex) {
             return static_cast<int64_t>(row * 100 + mapIndex);
           })});

  writeData({batch}, {"key"}, {{"features", {}}});

  // Project entire column (no key subscripts) — should fail.
  std::vector<Subfield> subfields;
  subfields.emplace_back("features");
  NIMBLE_ASSERT_THROW(
      createProjector(subfields),
      "Cannot project entire FlatMap column without key subscripts");
}

TEST_P(NimbleIndexProjectorTest, flatMapKeyProjectionSchemaComparison) {
  // Verify that buildProjectedNimbleType (nimble-type overload) produces
  // FlatMap with keys sorted alphabetically for key-level projection.
  auto rowType = ROW({"key", "features"}, {BIGINT(), MAP(VARCHAR(), BIGINT())});

  const int numRows = 200;
  const std::vector<std::string> mapKeys = {"a", "b", "c", "d", "e"};
  const auto numMapKeys = static_cast<vector_size_t>(mapKeys.size());

  std::vector<StringView> mapKeyViews;
  mapKeyViews.reserve(mapKeys.size());
  for (const auto& k : mapKeys) {
    mapKeyViews.emplace_back(k);
  }

  auto batch = vectorMaker_->rowVector(
      {"key", "features"},
      {vectorMaker_->flatVector<int64_t>(
           numRows, [](auto row) { return row * 10; }),
       vectorMaker_->mapVector<StringView, int64_t>(
           numRows,
           /*sizeAt*/ [&](auto /*row*/) { return numMapKeys; },
           /*keyAt*/
           [&](auto /*row*/, auto mapIndex) { return mapKeyViews[mapIndex]; },
           /*valueAt*/
           [](auto row, auto mapIndex) {
             return static_cast<int64_t>(row * 100 + mapIndex);
           })});

  writeData({batch}, {"key"}, {{"features", {}}});

  // Project keys in non-alphabetical order: ["d", "b"].
  std::vector<Subfield> subfields;
  subfields.emplace_back("features[\"d\"]");
  subfields.emplace_back("features[\"b\"]");
  auto projector = createProjector(subfields);

  // Projected schema should be FlatMap with keys sorted alphabetically.
  const auto& buildSchema = projector->projectedNimbleType();
  ASSERT_TRUE(buildSchema->isRow());
  ASSERT_EQ(buildSchema->asRow().childrenCount(), 1);

  const auto& buildFlatMap = buildSchema->asRow().childAt(0)->asFlatMap();
  ASSERT_EQ(buildFlatMap.childrenCount(), 2);

  // Keys should be sorted alphabetically: "b", "d".
  EXPECT_EQ(buildFlatMap.nameAt(0), "b");
  EXPECT_EQ(buildFlatMap.nameAt(1), "d");

  // Verify schema decodes the projected data correctly.
  auto pointBounds = makePointLookup(rowType, {"key"}, 50);
  NimbleIndexProjector::Request request;
  request.keyBounds = {pointBounds};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 1);
  ASSERT_FALSE(result.responses[0].slices.empty());

  const auto& slice = result.responses[0].slices[0];
  auto coalesced = coalesceChunkSlice(slice);
  auto data = std::string_view(
      reinterpret_cast<const char*>(coalesced.data()), coalesced.length());

  DeserializerOptions deserOptions;
  deserOptions.hasHeader = true;
  Deserializer deserializer(buildSchema, leafPool_.get(), deserOptions);
  VectorPtr output;
  deserializer.deserialize(data, output);
  ASSERT_NE(output, nullptr);
  auto* row = output->as<RowVector>();
  auto* map = row->childAt(0)->as<MapVector>();
  ASSERT_NE(map, nullptr);
  EXPECT_GE(map->size(), 1);
}

TEST_P(NimbleIndexProjectorTest, flatMapIntKeyProjection) {
  // Schema: key (int64, sorted), features (MAP<INT, BIGINT> as FlatMap).
  // Tests LongSubscript handling in resolveSubfield: "features[3]" produces
  // a LongSubscript which is converted to string key "3" for FlatMap lookup.
  auto rowType = ROW({"key", "features"}, {BIGINT(), MAP(INTEGER(), BIGINT())});

  const int numRows = 200;
  const int numMapKeys = 5;
  auto batch = vectorMaker_->rowVector(
      {"key", "features"},
      {vectorMaker_->flatVector<int64_t>(
           numRows, [](auto row) { return row * 10; }),
       vectorMaker_->mapVector<int32_t, int64_t>(
           numRows,
           /*sizeAt*/ [&](auto /*row*/) { return numMapKeys; },
           /*keyAt*/
           [](auto /*row*/, auto mapIndex) {
             return static_cast<int32_t>(mapIndex);
           },
           /*valueAt*/
           [](auto row, auto mapIndex) {
             return static_cast<int64_t>(row * 100 + mapIndex);
           })});

  writeData({batch}, {"key"}, {{"features", {}}});

  // Project integer keys in non-sorted order: [3, 1, 4].
  std::vector<Subfield> subfields;
  subfields.emplace_back("features[3]");
  subfields.emplace_back("features[1]");
  subfields.emplace_back("features[4]");
  auto projector = createProjector(subfields);

  // Point lookup for key=50 (row index 5).
  auto pointBounds = makePointLookup(rowType, {"key"}, 50);
  NimbleIndexProjector::Request request;
  request.keyBounds = {pointBounds};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 1);
  const auto& response = result.responses[0];
  ASSERT_FALSE(response.slices.empty());

  const auto& slice = response.slices[0];
  const auto rowRange = readEmbeddedRowRange(slice);
  ASSERT_GT(rowRange.numRows(), 0);

  // Deserialize using the projector's projected nimble type.
  auto coalesced = coalesceChunkSlice(slice);
  DeserializerOptions deserOptions;
  deserOptions.hasHeader = true;
  Deserializer deserializer(
      projector->projectedNimbleType(), leafPool_.get(), deserOptions);

  VectorPtr deserialized;
  deserializer.deserialize(
      std::string_view(
          reinterpret_cast<const char*>(coalesced.data()), coalesced.length()),
      deserialized);
  ASSERT_NE(deserialized, nullptr);

  auto* rowResult = deserialized->as<RowVector>();
  ASSERT_NE(rowResult, nullptr);
  ASSERT_EQ(rowResult->childrenSize(), 1);

  auto* mapResult = rowResult->childAt(0)->as<MapVector>();
  ASSERT_NE(mapResult, nullptr);
  // Sliced output contains exactly the point-lookup row.
  ASSERT_EQ(mapResult->size(), rowRange.numRows());

  auto* mapKeyResults = mapResult->mapKeys()->as<FlatVector<int32_t>>();
  auto* mapValues = mapResult->mapValues()->as<FlatVector<int64_t>>();
  ASSERT_NE(mapKeyResults, nullptr);
  ASSERT_NE(mapValues, nullptr);

  const auto mapOffset = mapResult->offsetAt(0);
  const auto mapSize = mapResult->sizeAt(0);

  // row=5: key=50, features[k] = 5*100 + k.
  std::map<int32_t, int64_t> kvPairs;
  for (vector_size_t i = 0; i < mapSize; ++i) {
    kvPairs[mapKeyResults->valueAt(mapOffset + i)] =
        mapValues->valueAt(mapOffset + i);
  }

  // Verify projected keys: 1, 3, 4.
  ASSERT_EQ(kvPairs.size(), 3);
  EXPECT_EQ(kvPairs[1], 501); // 5*100 + 1
  EXPECT_EQ(kvPairs[3], 503); // 5*100 + 3
  EXPECT_EQ(kvPairs[4], 504); // 5*100 + 4
}

TEST_P(NimbleIndexProjectorTest, flatMapMissingKeys) {
  // Verify that missing FlatMap keys are emitted as synthetic placeholder
  // children in the projected schema so cross-schema-version reads decode
  // missing keys as null columns instead of silently shifting offsets.
  auto rowType = ROW({"key", "features"}, {BIGINT(), MAP(VARCHAR(), BIGINT())});

  const int numRows = 200;
  const std::vector<std::string> mapKeys = {"a", "b", "c"};
  const auto numMapKeys = static_cast<vector_size_t>(mapKeys.size());

  std::vector<StringView> mapKeyViews;
  mapKeyViews.reserve(mapKeys.size());
  for (const auto& k : mapKeys) {
    mapKeyViews.emplace_back(k);
  }

  auto batch = vectorMaker_->rowVector(
      {"key", "features"},
      {vectorMaker_->flatVector<int64_t>(
           numRows, [](auto row) { return row * 10; }),
       vectorMaker_->mapVector<StringView, int64_t>(
           numRows,
           /*sizeAt*/ [&](auto /*row*/) { return numMapKeys; },
           /*keyAt*/
           [&](auto /*row*/, auto mapIndex) { return mapKeyViews[mapIndex]; },
           /*valueAt*/
           [](auto row, auto mapIndex) {
             return static_cast<int64_t>(row * 100 + mapIndex);
           })});

  writeData({batch}, {"key"}, {{"features", {}}});

  // Project mix of existing ("a", "c") and missing ("x", "z") keys.
  {
    std::vector<Subfield> subfields;
    subfields.emplace_back("features[\"a\"]");
    subfields.emplace_back("features[\"x\"]");
    subfields.emplace_back("features[\"c\"]");
    subfields.emplace_back("features[\"z\"]");
    auto projector = createProjector(subfields);

    // Projected schema contains all requested keys in alphabetical order
    // ("a", "c", "x", "z"). Real keys ("a", "c") carry source data while
    // synthetic keys ("x", "z") hold placeholder slots that decode to nulls.
    const auto& schema = projector->projectedNimbleType();
    ASSERT_TRUE(schema->isRow());
    const auto& flatMap = schema->asRow().childAt(0)->asFlatMap();
    ASSERT_EQ(flatMap.childrenCount(), 4);
    EXPECT_EQ(flatMap.nameAt(0), "a");
    EXPECT_EQ(flatMap.nameAt(1), "c");
    EXPECT_EQ(flatMap.nameAt(2), "x");
    EXPECT_EQ(flatMap.nameAt(3), "z");

    // Decode the projected slice and verify per-row values. Source row r's
    // map values are `r * 100 + mapIndex` for mapIndex in {0, 1, 2} mapping
    // to keys {"a", "b", "c"} respectively. The decoded map for the target
    // row must contain only the present projected keys ("a" → r*100+0,
    // "c" → r*100+2) — "x" and "z" are missing in source and therefore not
    // present in the decoded MapVector entries (gap-fill renders their inMap
    // as all-false, producing no key entry).
    auto pointBounds = makePointLookup(rowType, {"key"}, 50);
    NimbleIndexProjector::Request request;
    request.keyBounds = {pointBounds};
    auto result = projector->project(request, {});
    ASSERT_EQ(result.responses.size(), 1);
    ASSERT_FALSE(result.responses[0].slices.empty());

    const auto& slice = result.responses[0].slices[0];
    const auto rowRange = readEmbeddedRowRange(slice);
    ASSERT_GT(rowRange.numRows(), 0);

    auto coalesced = coalesceChunkSlice(slice);
    DeserializerOptions deserOptions;
    deserOptions.hasHeader = true;
    Deserializer deserializer(
        projector->projectedNimbleType(), leafPool_.get(), deserOptions);
    VectorPtr deserialized;
    deserializer.deserialize(
        std::string_view(
            reinterpret_cast<const char*>(coalesced.data()),
            coalesced.length()),
        deserialized);
    ASSERT_NE(deserialized, nullptr);

    auto* mapResult =
        deserialized->as<RowVector>()->childAt(0)->as<MapVector>();
    ASSERT_NE(mapResult, nullptr);
    ASSERT_EQ(mapResult->size(), rowRange.numRows());

    auto* keysVec = mapResult->mapKeys()->as<FlatVector<StringView>>();
    auto* valsVec = mapResult->mapValues()->as<FlatVector<int64_t>>();
    ASSERT_NE(keysVec, nullptr);
    ASSERT_NE(valsVec, nullptr);

    const auto mapOffset = mapResult->offsetAt(0);
    const auto mapSize = mapResult->sizeAt(0);
    std::map<std::string, int64_t> kvPairs;
    for (vector_size_t i = 0; i < mapSize; ++i) {
      kvPairs[keysVec->valueAt(mapOffset + i).str()] =
          valsVec->valueAt(mapOffset + i);
    }
    // Only the present-key entries appear; "x" and "z" are absent (their
    // placeholder slots produced all-false in-map → no map entry).
    EXPECT_EQ(2, kvPairs.size());
    EXPECT_EQ(500, kvPairs["a"]); // row 5 * 100 + mapIndex 0
    EXPECT_EQ(502, kvPairs["c"]); // row 5 * 100 + mapIndex 2
    EXPECT_EQ(0, kvPairs.count("x"));
    EXPECT_EQ(0, kvPairs.count("z"));
  }

  // All requested keys missing — projection still succeeds. The FlatMap
  // contains alphabetically-sorted placeholder children for every requested
  // key, all decoded as null columns by the deserializer's gap-fill.
  {
    std::vector<Subfield> subfields;
    subfields.emplace_back("features[\"x\"]");
    subfields.emplace_back("features[\"y\"]");
    auto projector = createProjector(subfields);
    const auto& schema = projector->projectedNimbleType();
    ASSERT_TRUE(schema->isRow());
    const auto& flatMap = schema->asRow().childAt(0)->asFlatMap();
    ASSERT_EQ(flatMap.childrenCount(), 2);
    EXPECT_EQ(flatMap.nameAt(0), "x");
    EXPECT_EQ(flatMap.nameAt(1), "y");
  }
}

TEST_P(NimbleIndexProjectorTest, flatMapMissingKeyDeserializesAsNullField) {
  auto rowType = ROW({"key", "features"}, {BIGINT(), MAP(VARCHAR(), BIGINT())});

  constexpr int numRows = 64;
  const std::vector<std::string> mapKeys = {"a", "b", "c"};
  const auto numMapKeys = static_cast<vector_size_t>(mapKeys.size());

  std::vector<StringView> mapKeyViews;
  mapKeyViews.reserve(mapKeys.size());
  for (const auto& key : mapKeys) {
    mapKeyViews.emplace_back(key);
  }

  auto batch = vectorMaker_->rowVector(
      {"key", "features"},
      {vectorMaker_->flatVector<int64_t>(
           numRows, [](auto row) { return static_cast<int64_t>(row); }),
       vectorMaker_->mapVector<StringView, int64_t>(
           numRows,
           /*sizeAt*/ [&](auto /*row*/) { return numMapKeys; },
           /*keyAt*/
           [&](auto /*row*/, auto mapIndex) { return mapKeyViews[mapIndex]; },
           /*valueAt*/
           [](auto row, auto mapIndex) {
             return static_cast<int64_t>(row * 100 + mapIndex);
           })});

  writeData({batch}, {"key"}, {{"features", {}}});

  std::vector<Subfield> subfields;
  subfields.emplace_back("features[\"a\"]");
  subfields.emplace_back("features[\"x\"]");
  auto projector = createProjector(subfields);

  auto bounds = makeRangeLookup(rowType, {"key"}, 10, 20);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  auto result = projector->project(request, {});

  ASSERT_EQ(result.responses.size(), 1);
  ASSERT_EQ(result.responses[0].slices.size(), 1);

  auto coalesced = coalesceChunkSlice(result.responses[0].slices[0]);
  DeserializerOptions deserOptions{
      .hasHeader = true,
      .outputType = ROW({"features"}, {ROW({"a", "x"}, {BIGINT(), BIGINT()})})};
  Deserializer deserializer(
      projector->projectedNimbleType(), leafPool_.get(), deserOptions);

  VectorPtr output;
  deserializer.deserialize(
      std::string_view(
          reinterpret_cast<const char*>(coalesced.data()), coalesced.length()),
      output);

  ASSERT_NE(output, nullptr);
  // Sliced output has 10 rows for the [10, 20) range.
  ASSERT_EQ(output->size(), 10);
  auto* features = output->as<RowVector>()->childAt(0)->as<RowVector>();
  ASSERT_NE(features, nullptr);
  ASSERT_EQ(features->childrenSize(), 2);
  EXPECT_EQ(features->type()->asRow().nameOf(0), "a");
  EXPECT_EQ(features->type()->asRow().nameOf(1), "x");

  const auto* presentKey = features->childAt(0)->asFlatVector<int64_t>();
  const auto* missingKey = features->childAt(1)->asFlatVector<int64_t>();
  ASSERT_NE(presentKey, nullptr);
  ASSERT_NE(missingKey, nullptr);

  for (vector_size_t outRow = 0; outRow < 10; ++outRow) {
    const int64_t sourceRow = 10 + outRow;
    SCOPED_TRACE(fmt::format("outRow={} sourceRow={}", outRow, sourceRow));
    EXPECT_FALSE(presentKey->isNullAt(outRow));
    EXPECT_EQ(presentKey->valueAt(outRow), sourceRow * 100);
    EXPECT_TRUE(missingKey->isNullAt(outRow));
  }
}

TEST_P(NimbleIndexProjectorTest, featureReorderingStorageReads) {
  // Schema: key (int64, sorted), features (MAP<INT, BIGINT> as FlatMap).
  // Write with 10 FlatMap keys (0-9), project 3 keys {7, 3, 1}.
  // With feature reordering, projected streams are adjacent on disk and require
  // fewer storage reads when coalesce distance is small.
  auto rowType = ROW({"key", "features"}, {BIGINT(), MAP(INTEGER(), BIGINT())});

  const int numRows = 1'000;
  const int numMapKeys = 10;
  const std::vector<int64_t> projectedKeys = {7, 3, 1};

  auto makeBatch = [&]() {
    return vectorMaker_->rowVector(
        {"key", "features"},
        {vectorMaker_->flatVector<int64_t>(
             numRows, [](auto row) { return static_cast<int64_t>(row); }),
         vectorMaker_->mapVector<int32_t, int64_t>(
             numRows,
             /*sizeAt*/ [&](auto /*row*/) { return numMapKeys; },
             /*keyAt*/
             [](auto /*row*/, auto mapIndex) {
               return static_cast<int32_t>(mapIndex);
             },
             /*valueAt*/
             [](auto row, auto mapIndex) {
               return static_cast<int64_t>(row * 100 + mapIndex);
             })});
  };

  // Write data with optional feature reordering.
  auto writeFile = [&](bool enableReordering,
                       StripeGroup::EncodingLayout metadataFormat =
                           StripeGroup::EncodingLayout::kRaw) {
    sinkData_.clear();
    auto file = std::make_unique<InMemoryWriteFile>(&sinkData_);

    WriterOptions options;
    options.enableChunking = true;
    options.flatMapColumns = {{"features", {}}};
    options.experimentalStripeGroupEncodingLayout = metadataFormat;

    options.clusterIndexConfig =
        ClusterIndexConfigBuilder{}
            .withKeyColumns({"key"})
            .withSortOrders({SortOrder{.ascending = true}})
            .withEnforceKeyOrder(true)
            .withNoDuplicateKey(true)
            .build();

    if (enableReordering) {
      // Ordinal 1 = "features" column (after "key").
      options.featureReordering =
          std::vector<std::tuple<size_t, std::vector<int64_t>>>{
              {1, projectedKeys}};
    }

    auto batch = makeBatch();
    Writer writer(
        batch->type(), std::move(file), *rootPool_, std::move(options));
    writer.write(batch);
    writer.close();
  };

  // Create projector with custom coalesce distance.
  auto makeProjector = [&](int32_t maxCoalesceDistance) {
    std::vector<Subfield> subfields;
    subfields.reserve(projectedKeys.size());
    for (auto key : projectedKeys) {
      subfields.emplace_back(fmt::format("features[{}]", key));
    }

    dataIoStats_ = std::make_shared<velox::io::IoStatistics>();
    metadataIoStats_ = std::make_shared<velox::io::IoStatistics>();
    indexIoStats_ = std::make_shared<velox::io::IoStatistics>();
    auto fileHandle = makeFileHandle();
    dwio::common::ReaderOptions readerOptions(leafPool_.get());
    readerOptions.setDataIoStats(dataIoStats_);
    readerOptions.setMetadataIoStats(metadataIoStats_);
    readerOptions.setIndexIoStats(indexIoStats_);
    readerOptions.setFileFormat(FileFormat::NIMBLE);
    readerOptions.setLoadClusterIndex(true);
    readerOptions.setMaxCoalesceDistance(maxCoalesceDistance);
    readerOptions.setCacheData(false);
    readerOptions.setIOExecutor(ioExecutor_);
    readerOptions.setFilePreloadThreshold(0);

    TabletReaderCache::testingReset();
    ensureTabletReaderCache();
    return NimbleIndexProjector::create(
        *tabletReaderCache_, fileHandle, subfields, readerOptions);
  };

  // Part 1: Verify stream adjacency on disk with feature reordering, for each
  // stripe-group metadata format.
  for (auto metadataFormat :
       {StripeGroup::EncodingLayout::kRaw,
        StripeGroup::EncodingLayout::kStreamMajor}) {
    SCOPED_TRACE(
        fmt::format("metadataFormat={}", static_cast<int>(metadataFormat)));
    writeFile(/*enableReordering=*/true, metadataFormat);
    auto readFile =
        std::make_shared<InMemoryReadFile>(std::string_view(sinkData_));
    auto tablet = TabletReader::create(
        readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
    ASSERT_GE(tablet->stripeCount(), 1);

    auto stripeId = tablet->stripeIdentifier(0);
    const auto streamCount = tablet->streamCount(stripeId);
    std::vector<TabletReader::StreamLocation> streamLocations(streamCount);
    tablet->streamLocations(stripeId, streamLocations);

    BatchReader reader(readFile.get(), *leafPool_);
    const auto& flatMap = reader.schema()->asRow().childAt(1)->asFlatMap();

    std::unordered_map<std::string, uint32_t> keyToValueStreamId;
    for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
      keyToValueStreamId[flatMap.nameAt(i)] =
          flatMap.childAt(i)->asScalar().scalarDescriptor().offset();
    }

    auto diskPosition = [&](int64_t key) -> uint32_t {
      return streamLocations[keyToValueStreamId.at(folly::to<std::string>(key))]
          .offset;
    };

    // Verify projected keys appear in reordering order on disk.
    for (size_t i = 1; i < projectedKeys.size(); ++i) {
      EXPECT_LT(
          diskPosition(projectedKeys[i - 1]), diskPosition(projectedKeys[i]))
          << "Key " << projectedKeys[i - 1] << " should appear before key "
          << projectedKeys[i] << " on disk";
    }

    // Verify projected keys' value streams are contiguous (adjacent on disk).
    for (size_t i = 1; i < projectedKeys.size(); ++i) {
      auto prevStreamId =
          keyToValueStreamId.at(folly::to<std::string>(projectedKeys[i - 1]));
      auto currStreamId =
          keyToValueStreamId.at(folly::to<std::string>(projectedKeys[i]));
      EXPECT_EQ(
          streamLocations[prevStreamId].offset +
              streamLocations[prevStreamId].size,
          streamLocations[currStreamId].offset)
          << "Key " << projectedKeys[i - 1]
          << " value stream should be adjacent to key " << projectedKeys[i];
    }
  }

  // Part 2: Compare storage reads with and without feature reordering.
  struct ReorderParam {
    bool enableReordering;
    int32_t maxCoalesceDistance;
    std::string debugString() const {
      return fmt::format(
          "enableReordering={}, maxCoalesceDistance={}",
          enableReordering,
          maxCoalesceDistance);
    }
  };

  std::vector<ReorderParam> testSettings = {
      {false, 0},
      {true, 0},
      {false, 512 << 10},
      {true, 512 << 10},
  };

  // Map from debugString to numStorageReads.
  std::map<std::string, uint64_t> storageReads;

  for (const auto& param : testSettings) {
    SCOPED_TRACE(param.debugString());

    writeFile(param.enableReordering);
    auto projector = makeProjector(param.maxCoalesceDistance);

    // Point lookup for key in the middle.
    auto bounds = makePointLookup(rowType, {"key"}, numRows / 2);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    auto result = projector->project(request, {});

    ASSERT_EQ(result.responses.size(), 1);
    ASSERT_FALSE(result.responses[0].slices.empty());

    storageReads[param.debugString()] = dataIoStats_->read().count();

    if (param.maxCoalesceDistance == 0 && !param.enableReordering) {
      EXPECT_GT(dataIoStats_->readGap().count(), 0)
          << "Projecting non-adjacent streams should produce gaps";
    }
  }

  // Verify all configurations produced valid results with non-zero reads.
  for (const auto& [key, reads] : storageReads) {
    SCOPED_TRACE(key);
    EXPECT_GT(reads, 0) << "Expected non-zero storage reads";
  }
}

TEST_P(NimbleIndexProjectorTest, readGapTracking) {
  // Schema with many columns: projecting a sparse subset creates gaps between
  // the projected stream regions on disk.
  auto rowType =
      ROW({"key", "a", "b", "c", "d", "e"},
          {BIGINT(), INTEGER(), INTEGER(), INTEGER(), INTEGER(), INTEGER()});

  const int numRows = 500;
  auto batch = vectorMaker_->rowVector(
      {"key", "a", "b", "c", "d", "e"},
      {vectorMaker_->flatVector<int64_t>(
           numRows, [](auto i) { return static_cast<int64_t>(i); }),
       vectorMaker_->flatVector<int32_t>(
           numRows, [](auto i) { return i * 10; }),
       vectorMaker_->flatVector<int32_t>(
           numRows, [](auto i) { return i * 20; }),
       vectorMaker_->flatVector<int32_t>(
           numRows, [](auto i) { return i * 30; }),
       vectorMaker_->flatVector<int32_t>(
           numRows, [](auto i) { return i * 40; }),
       vectorMaker_->flatVector<int32_t>(
           numRows, [](auto i) { return i * 50; })});

  writeData({batch}, {"key"});

  // Project only "a" and "e" (skipping b, c, d) — creates gaps.
  {
    std::vector<Subfield> subfields;
    subfields.emplace_back("a");
    subfields.emplace_back("e");
    auto projector = createProjector(subfields);

    auto bounds = makePointLookup(rowType, {"key"}, numRows / 2);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    projector->project(request, {});

    EXPECT_GT(dataIoStats_->readGap().count(), 0)
        << "Projecting non-adjacent columns should produce gaps";
    EXPECT_GT(dataIoStats_->readGap().min(), 0);
  }

  // Project all value columns. All-true root null streams are omitted, so the
  // projected value streams are physically contiguous.
  {
    std::vector<Subfield> subfields;
    subfields.emplace_back("a");
    subfields.emplace_back("b");
    subfields.emplace_back("c");
    subfields.emplace_back("d");
    subfields.emplace_back("e");
    auto projector = createProjector(subfields);

    auto bounds = makePointLookup(rowType, {"key"}, numRows / 2);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    projector->project(request, {});

    EXPECT_EQ(dataIoStats_->readGap().count(), 0)
        << "Projecting all value columns should not produce gaps";
  }
}

TEST_P(NimbleIndexProjectorTest, resumeKeyOnTruncation) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 10 stripes x 100 rows = 1000 rows, keys 0..999.
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // Range scan [0, 500) with maxRows=50. Since maxRows is a soft limit
  // at stripe boundaries and each stripe has 100 rows, the first stripe
  // (100 rows) exceeds the limit but is fully included.
  auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  NimbleIndexProjector::Options options;
  options.maxRows = 50;
  options.needResumeKey = true;
  auto result = projector->project(request, options);

  ASSERT_EQ(result.responses.size(), 1);
  const auto& response = result.responses[0];
  ASSERT_FALSE(response.slices.empty());

  // Soft limit: first stripe (100 rows) included fully.
  uint64_t totalRows = 0;
  for (const auto& slice : response.slices) {
    totalRows += readEmbeddedRowRange(slice).numRows();
  }
  EXPECT_EQ(totalRows, 100);

  // Resume key must be set since the request spans more stripes.
  ASSERT_TRUE(response.resumeKey.has_value());
  EXPECT_FALSE(response.resumeKey->empty());

  // The resume key is embedded only on the last slice; earlier slices carry
  // an empty resume key marker.
  const auto numSlices = response.slices.size();
  for (size_t i = 0; i + 1 < numSlices; ++i) {
    EXPECT_FALSE(readEmbeddedResumeKey(response.slices[i]).has_value())
        << "non-last slice " << i << " unexpectedly carries a resume key";
  }
  const auto embeddedResumeKey = readEmbeddedResumeKey(response.slices.back());
  ASSERT_TRUE(embeddedResumeKey.has_value());
  EXPECT_EQ(*embeddedResumeKey, *response.resumeKey);
}

TEST_P(NimbleIndexProjectorTest, resumeKeyNotSetWithoutTruncation) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  // Case 1: maxRows = 0 (unlimited).
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 50);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    auto result = projector->project(request, {});
    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_FALSE(result.responses[0].resumeKey.has_value());
  }

  // Case 2: maxRows > total matching rows.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 50);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRows = 10000;
    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_FALSE(result.responses[0].resumeKey.has_value());
  }

  // Case 3: Point lookup (always returns <=1 row per key).
  {
    auto projector = createProjector(subfields);
    auto bounds = makePointLookup(rowType, {"key"}, 50);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRows = 10;
    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_FALSE(result.responses[0].resumeKey.has_value());
  }
}

TEST_P(NimbleIndexProjectorTest, softLimitAtStripeBoundary) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 10 stripes x 100 rows.
  writeResumeKeyTestData(/*rowsPerBatch=*/100, /*numBatches=*/10);

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  // maxRows=99: below stripe size, but soft limit includes entire first stripe.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRows = 99;
    options.needResumeKey = true;
    auto result = projector->project(request, options);

    ASSERT_EQ(result.responses.size(), 1);
    uint64_t totalRows = 0;
    for (const auto& slice : result.responses[0].slices) {
      totalRows += readEmbeddedRowRange(slice).numRows();
    }
    EXPECT_EQ(totalRows, 100);
    EXPECT_TRUE(result.responses[0].resumeKey.has_value());
  }

  // maxRows=100: exactly one stripe, stops after it.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRows = 100;
    options.needResumeKey = true;
    auto result = projector->project(request, options);

    ASSERT_EQ(result.responses.size(), 1);
    uint64_t totalRows = 0;
    for (const auto& slice : result.responses[0].slices) {
      totalRows += readEmbeddedRowRange(slice).numRows();
    }
    EXPECT_EQ(totalRows, 100);
    EXPECT_TRUE(result.responses[0].resumeKey.has_value());
  }

  // maxRows=101: exceeds first stripe, processes second stripe too (soft
  // limit). Second stripe has 100 rows, total = 200.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRows = 101;
    options.needResumeKey = true;
    auto result = projector->project(request, options);

    ASSERT_EQ(result.responses.size(), 1);
    uint64_t totalRows = 0;
    for (const auto& slice : result.responses[0].slices) {
      totalRows += readEmbeddedRowRange(slice).numRows();
    }
    EXPECT_EQ(totalRows, 200);
    EXPECT_TRUE(result.responses[0].resumeKey.has_value());
  }
}

TEST_P(NimbleIndexProjectorTest, resumeKeyPagination) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 10 stripes x 100 rows = 1000 rows, keys 0..999.
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  const int64_t rangeStart = 0;
  const int64_t rangeEnd = 1000;
  // maxRows=150 with 100-row stripes: each page gets 2 stripes (200 rows)
  // because 1st stripe (100) is under limit, 2nd stripe (200 total) exceeds
  // the soft limit and stops.
  const uint64_t maxRows = 150;

  uint64_t totalRowsRead = 0;
  auto currentBounds = makeRangeLookup(rowType, {"key"}, rangeStart, rangeEnd);
  int pageCount = 0;

  while (true) {
    auto projector = createProjector(subfields);
    NimbleIndexProjector::Request request;
    request.keyBounds = {currentBounds};
    NimbleIndexProjector::Options options;
    options.maxRows = maxRows;
    options.needResumeKey = true;
    auto result = projector->project(request, options);

    ASSERT_EQ(result.responses.size(), 1);
    const auto& response = result.responses[0];

    uint64_t pageRows = 0;
    for (const auto& slice : response.slices) {
      pageRows += readEmbeddedRowRange(slice).numRows();
    }
    EXPECT_GT(pageRows, 0) << "Page " << pageCount << " returned no rows";
    totalRowsRead += pageRows;
    ++pageCount;

    if (!response.resumeKey.has_value()) {
      break;
    }

    currentBounds = velox::serializer::EncodedKeyBounds{
        .lowerKey = response.resumeKey.value(),
        .upperKey = currentBounds.upperKey};
    ASSERT_LT(pageCount, 100) << "Pagination loop exceeded safety limit";
  }

  // All rows should have been read across all pages.
  EXPECT_EQ(totalRowsRead, static_cast<uint64_t>(rangeEnd - rangeStart));
}

TEST_P(NimbleIndexProjectorTest, maxRowsTotalAcrossRequests) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 10 stripes x 100 rows = 1000 rows, keys 0..999.
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // Two range scans: request 0 spans stripes 0-4, request 1 spans stripe 9.
  // maxRows=50 (soft limit): first stripe (100 rows) exceeds limit, stops.
  // Request 0 gets 100 rows from stripe 0, request 1 gets nothing (stripe 9
  // is never reached).
  auto bounds0 = makeRangeLookup(rowType, {"key"}, 0, 500);
  auto bounds1 = makeRangeLookup(rowType, {"key"}, 900, 1000);

  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds0, bounds1};
  NimbleIndexProjector::Options options;
  options.maxRows = 50;
  options.needResumeKey = true;
  auto result = projector->project(request, options);

  ASSERT_EQ(result.responses.size(), 2);

  // Request 0: gets full first stripe (100 rows), has resume key.
  {
    const auto& response = result.responses[0];
    uint64_t totalRows = 0;
    for (const auto& slice : response.slices) {
      totalRows += readEmbeddedRowRange(slice).numRows();
    }
    EXPECT_EQ(totalRows, 100);
    EXPECT_TRUE(response.resumeKey.has_value());
  }

  // Request 1: stripe 9 was never processed, resume key = original lower key.
  {
    const auto& response = result.responses[1];
    EXPECT_TRUE(response.slices.empty());
    ASSERT_TRUE(response.resumeKey.has_value());
    EXPECT_EQ(response.resumeKey.value(), bounds1.lowerKey);
  }
}

TEST_P(NimbleIndexProjectorTest, noResumeKeyWhenRequestFullySatisfied) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 10 stripes x 100 rows = 1000 rows, keys 0..999.
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // Request 0: [0, 100) — exactly one stripe, fully satisfied.
  // Request 1: [0, 500) — spans 5 stripes, will be truncated.
  // maxRows=100: first stripe covers both, request 0 ends within it.
  auto bounds0 = makeRangeLookup(rowType, {"key"}, 0, 100);
  auto bounds1 = makeRangeLookup(rowType, {"key"}, 0, 500);

  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds0, bounds1};
  NimbleIndexProjector::Options options;
  options.maxRows = 100;
  options.needResumeKey = true;
  auto result = projector->project(request, options);

  ASSERT_EQ(result.responses.size(), 2);

  // Request 0: fully satisfied within stripe 0 — no resume key.
  {
    const auto& response = result.responses[0];
    ASSERT_FALSE(response.slices.empty());
    EXPECT_FALSE(response.resumeKey.has_value());
  }

  // Request 1: spans beyond stripe 0, truncated — has resume key.
  {
    const auto& response = result.responses[1];
    ASSERT_FALSE(response.slices.empty());
    EXPECT_TRUE(response.resumeKey.has_value());
  }
}

TEST_P(NimbleIndexProjectorTest, maxRowsExceedsTotal) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 10 stripes x 100 rows = 1000 rows.
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // maxRows larger than total matching rows — no truncation.
  auto bounds = makeRangeLookup(rowType, {"key"}, 0, 300);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  NimbleIndexProjector::Options options;
  options.maxRows = 10000;
  auto result = projector->project(request, options);

  ASSERT_EQ(result.responses.size(), 1);
  uint64_t totalRows = 0;
  for (const auto& slice : result.responses[0].slices) {
    totalRows += readEmbeddedRowRange(slice).numRows();
  }
  EXPECT_EQ(totalRows, 300);
  EXPECT_FALSE(result.responses[0].resumeKey.has_value());
}

TEST_P(NimbleIndexProjectorTest, maxRowsZeroMeansUnlimited) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  auto bounds = makeRangeLookup(rowType, {"key"}, 0, 1000);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  NimbleIndexProjector::Options options;
  options.maxRows = 0;
  auto result = projector->project(request, options);

  ASSERT_EQ(result.responses.size(), 1);
  uint64_t totalRows = 0;
  for (const auto& slice : result.responses[0].slices) {
    totalRows += readEmbeddedRowRange(slice).numRows();
  }
  EXPECT_EQ(totalRows, 1000);
  EXPECT_FALSE(result.responses[0].resumeKey.has_value());
}

TEST_P(NimbleIndexProjectorTest, maxRowsSingleRowStripes) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 10 stripes x 1 row each.
  writeResumeKeyTestData(/*rowsPerBatch=*/1, /*numBatches=*/10);

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // maxRows=3: should process exactly 3 stripes (3 rows).
  auto bounds = makeRangeLookup(rowType, {"key"}, 0, 10);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  NimbleIndexProjector::Options options;
  options.maxRows = 3;
  options.needResumeKey = true;
  auto result = projector->project(request, options);

  ASSERT_EQ(result.responses.size(), 1);
  uint64_t totalRows = 0;
  for (const auto& slice : result.responses[0].slices) {
    totalRows += readEmbeddedRowRange(slice).numRows();
  }
  EXPECT_EQ(totalRows, 3);
  EXPECT_TRUE(result.responses[0].resumeKey.has_value());
}

TEST_P(NimbleIndexProjectorTest, maxRowsPerRequestClipping) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  // Stripe-aligned: 200 rows = exactly 2 stripes. No resume key.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRowsPerRequest = 200;
    auto result = projector->project(request, options);

    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_FALSE(result.responses[0].resumeKey.has_value());
    EXPECT_EQ(projector->stats().numProjectedRows, 200);
  }

  // Mid-stripe hard cut: 150 rows = 1 full stripe (100) + 50 from stripe 2.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRowsPerRequest = 150;
    auto result = projector->project(request, options);

    ASSERT_EQ(result.responses.size(), 1);
    const auto& response = result.responses[0];
    EXPECT_FALSE(response.resumeKey.has_value());
    EXPECT_EQ(projector->stats().numProjectedRows, 150);
    ASSERT_EQ(response.slices.size(), 2);
    EXPECT_EQ(readEmbeddedRowRange(response.slices[0]).numRows(), 100);
    EXPECT_EQ(readEmbeddedRowRange(response.slices[1]).numRows(), 50);
  }
}

TEST_P(NimbleIndexProjectorTest, maxRowsPerRequestIndependentPruning) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // Request 0: [0, 500) spans stripes 0-4.
  // Request 1: [200, 800) spans stripes 2-7.
  // maxRowsPerRequest=100: each gets exactly 100 rows, no resume key.
  auto bounds0 = makeRangeLookup(rowType, {"key"}, 0, 500);
  auto bounds1 = makeRangeLookup(rowType, {"key"}, 200, 800);

  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds0, bounds1};
  NimbleIndexProjector::Options options;
  options.maxRowsPerRequest = 100;
  auto result = projector->project(request, options);

  ASSERT_EQ(result.responses.size(), 2);

  {
    const auto& response = result.responses[0];
    uint64_t totalRows = 0;
    for (const auto& slice : response.slices) {
      totalRows += readEmbeddedRowRange(slice).numRows();
    }
    EXPECT_EQ(totalRows, 100);
    EXPECT_FALSE(response.resumeKey.has_value());
  }

  {
    const auto& response = result.responses[1];
    uint64_t totalRows = 0;
    for (const auto& slice : response.slices) {
      totalRows += readEmbeddedRowRange(slice).numRows();
    }
    EXPECT_EQ(totalRows, 100);
    EXPECT_FALSE(response.resumeKey.has_value());
  }
}

TEST_P(NimbleIndexProjectorTest, maxBytesTruncation) {
  writeResumeKeyTestData();
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  // First, read one stripe without limits to learn the per-stripe byte size.
  uint64_t bytesPerStripe;
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 100);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    auto result = projector->project(request, {});
    ASSERT_EQ(result.responses[0].slices.size(), 1);
    bytesPerStripe = projector->stats().numOutputBytes;
    ASSERT_GT(bytesPerStripe, 0);
  }

  // Range [0, 500) spans 5 stripes. Set byte limit to allow ~2 stripes.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxBytes = bytesPerStripe * 2;
    options.needResumeKey = true;

    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    const auto& response = result.responses[0];

    // Should be truncated with a resume key.
    ASSERT_TRUE(response.resumeKey.has_value());

    // Soft byte limit — at most 3 stripes (planning uses raw stream sizes
    // which underestimate serialized output, so one extra stripe may fit).
    EXPECT_LE(projector->stats().numReadStripes, 3);
    EXPECT_LT(projector->stats().numReadStripes, 5);
  }
}

TEST_P(NimbleIndexProjectorTest, maxBytesNoTruncation) {
  writeResumeKeyTestData();
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  // maxBytes=0 (unlimited) -> no truncation, no resume key.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 100, 300);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    auto result = projector->project(request, {});
    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_FALSE(result.responses[0].resumeKey.has_value());
  }

  // maxBytes larger than total data -> no truncation, no resume key.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 100, 300);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxBytes = std::numeric_limits<uint64_t>::max();
    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_FALSE(result.responses[0].resumeKey.has_value());
  }
}

TEST_P(NimbleIndexProjectorTest, maxBytesPagination) {
  writeResumeKeyTestData();
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  // Learn per-stripe byte size.
  uint64_t bytesPerStripe;
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 100);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    auto result = projector->project(request, {});
    ASSERT_EQ(result.responses[0].slices.size(), 1);
    bytesPerStripe = projector->stats().numOutputBytes;
  }

  // Paginate through the full range using byte limits.
  auto paginate =
      [&](int64_t lowerKey, int64_t upperKey, uint64_t maxBytes) -> uint64_t {
    auto currentBounds = makeRangeLookup(rowType, {"key"}, lowerKey, upperKey);
    uint64_t totalReadRows = 0;
    int iterations = 0;
    const int maxIterations = 100;

    while (iterations < maxIterations) {
      ++iterations;
      auto projector = createProjector(subfields);
      NimbleIndexProjector::Request request;
      request.keyBounds = {currentBounds};
      NimbleIndexProjector::Options options;
      options.maxBytes = maxBytes;
      options.needResumeKey = true;

      auto result = projector->project(request, options);
      EXPECT_EQ(result.responses.size(), 1);
      totalReadRows += projector->stats().numProjectedRows;

      const auto& response = result.responses[0];
      if (!response.resumeKey.has_value()) {
        break;
      }
      currentBounds = velox::serializer::EncodedKeyBounds{
          .lowerKey = response.resumeKey.value(),
          .upperKey = currentBounds.upperKey};
    }
    EXPECT_LT(iterations, maxIterations);
    return totalReadRows;
  };

  EXPECT_EQ(paginate(0, 1000, bytesPerStripe), 1000);
  EXPECT_EQ(paginate(0, 1000, bytesPerStripe * 2), 1000);
  EXPECT_EQ(paginate(0, 1000, bytesPerStripe * 3), 1000);
  EXPECT_EQ(paginate(0, 1000, bytesPerStripe * 100), 1000);
  EXPECT_EQ(paginate(150, 750, bytesPerStripe), 600);
  EXPECT_EQ(paginate(150, 750, bytesPerStripe * 2), 600);
}

TEST_P(NimbleIndexProjectorTest, maxBytesAndMaxRowsInteraction) {
  writeResumeKeyTestData();
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  // Learn per-stripe byte size.
  uint64_t bytesPerStripe;
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 100);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    auto result = projector->project(request, {});
    ASSERT_EQ(result.responses[0].slices.size(), 1);
    bytesPerStripe = projector->stats().numOutputBytes;
  }

  // Range [0, 500) = 5 stripes, 500 rows.
  auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);

  // Case 1: Global row limit is tighter (50 rows < 3 stripes of bytes).
  {
    auto projector = createProjector(subfields);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRows = 50;
    options.maxBytes = bytesPerStripe * 3;
    options.needResumeKey = true;

    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    ASSERT_TRUE(result.responses[0].resumeKey.has_value());
    EXPECT_EQ(projector->stats().numReadStripes, 1);
    EXPECT_EQ(projector->stats().numReadRows, 100);
  }

  // Case 2: Byte limit is tighter (1 stripe of bytes < 500 rows).
  // Soft byte limit — planning uses raw stream sizes, so may include
  // one extra stripe beyond the serialized-byte budget.
  {
    auto projector = createProjector(subfields);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRows = 500;
    options.maxBytes = bytesPerStripe;
    options.needResumeKey = true;

    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    ASSERT_TRUE(result.responses[0].resumeKey.has_value());
    EXPECT_LE(projector->stats().numReadStripes, 2);
    EXPECT_LT(projector->stats().numReadStripes, 5);
  }
}

TEST_P(NimbleIndexProjectorTest, maxRowsPerRequestAndGlobalMaxRows) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  // Per-request limit is tighter: 100 rows/request vs 500 global.
  // Request is fulfilled (100 rows), no resume key from maxRowsPerRequest.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRowsPerRequest = 100;
    options.maxRows = 500;

    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_FALSE(result.responses[0].resumeKey.has_value());
    EXPECT_EQ(projector->stats().numProjectedRows, 100);
  }

  // Global limit is tighter: 50 rows global vs 300 rows/request.
  // Global limit triggers a resume key (needResumeKey set; stripe-boundary
  // soft limit = 100 rows).
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRowsPerRequest = 300;
    options.maxRows = 50;
    options.needResumeKey = true;

    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_TRUE(result.responses[0].resumeKey.has_value());
    EXPECT_EQ(projector->stats().numReadStripes, 1);
    EXPECT_EQ(projector->stats().numReadRows, 100);
  }

  // Per-request hard limit cuts mid-stripe: 50 rows/request < 100-row stripe.
  // No resume key — request is fulfilled with 50 rows.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRowsPerRequest = 50;
    options.maxRows = 500;

    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_FALSE(result.responses[0].resumeKey.has_value());
    EXPECT_EQ(projector->stats().numProjectedRows, 50);
  }
}

TEST_P(NimbleIndexProjectorTest, maxRowsPerRequestStripeAlignedSetsResumeKeys) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  auto bounds0 = makeRangeLookup(rowType, {"key"}, 0, 500);
  auto bounds1 = makeRangeLookup(rowType, {"key"}, 200, 800);

  // maxRowsPerRequest=100 lands exactly on the stripe boundary; needResumeKey
  // makes each request resume from the next stripe that still contains it.
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds0, bounds1};
  NimbleIndexProjector::Options options;
  options.maxRowsPerRequest = 100;
  options.needResumeKey = true;
  auto result = projector->project(request, options);

  ASSERT_EQ(result.responses.size(), 2);
  EXPECT_EQ(projector->stats().numReadStripes, 2);
  EXPECT_EQ(projector->stats().numProjectedRows, 200);

  for (const auto& response : result.responses) {
    ASSERT_EQ(response.slices.size(), 1);
    EXPECT_EQ(readEmbeddedRowRange(response.slices[0]).numRows(), 100);
    ASSERT_TRUE(response.resumeKey.has_value());
    EXPECT_EQ(readEmbeddedResumeKey(response.slices[0]), response.resumeKey);
  }
}

TEST_P(NimbleIndexProjectorTest, needResumeKeyGatesResumeKey) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 10 stripes x 100 rows.
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);

  // maxRows truncates after the first stripe, but needResumeKey is unset, so
  // neither the response nor the last slice carries a resume key.
  {
    auto projector = createProjector(subfields);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRows = 50;
    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    const auto& response = result.responses[0];
    ASSERT_FALSE(response.slices.empty());
    EXPECT_FALSE(response.resumeKey.has_value());
    EXPECT_FALSE(readEmbeddedResumeKey(response.slices.back()).has_value());
  }

  // Same truncation with needResumeKey set -> resume key present.
  {
    auto projector = createProjector(subfields);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRows = 50;
    options.needResumeKey = true;
    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_TRUE(result.responses[0].resumeKey.has_value());
  }

  // maxRowsPerRequest cuts the request mid-stripe; without needResumeKey the
  // request is returned capped with no resume key.
  {
    auto projector = createProjector(subfields);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRowsPerRequest = 150;
    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_FALSE(result.responses[0].resumeKey.has_value());
    EXPECT_EQ(projector->stats().numProjectedRows, 150);
  }

  // Same per-request cap with needResumeKey set -> resume key present.
  {
    auto projector = createProjector(subfields);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRowsPerRequest = 150;
    options.needResumeKey = true;
    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_TRUE(result.responses[0].resumeKey.has_value());
    EXPECT_EQ(projector->stats().numProjectedRows, 150);
  }
}

TEST_P(NimbleIndexProjectorTest, limitsTruncateButEmitNoResumeKeyWhenDisabled) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 10 stripes x 100 rows = 1000 rows, keys 0..999.
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  // maxRows still truncates at the stripe boundary (100 of the 500 matching
  // rows, one stripe), but with needResumeKey unset no resume key is produced.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRows = 50;
    options.needResumeKey = false;
    auto result = projector->project(request, options);

    ASSERT_EQ(result.responses.size(), 1);
    const auto& response = result.responses[0];
    uint64_t totalRows = 0;
    for (const auto& slice : response.slices) {
      totalRows += readEmbeddedRowRange(slice).numRows();
    }
    EXPECT_EQ(totalRows, 100);
    EXPECT_EQ(projector->stats().numReadStripes, 1);
    ASSERT_FALSE(response.slices.empty());
    EXPECT_FALSE(response.resumeKey.has_value());
    EXPECT_FALSE(readEmbeddedResumeKey(response.slices.back()).has_value());
  }

  // maxRowsPerRequest still hard-clips the request to 150 rows, but with
  // needResumeKey unset no resume key is produced.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRowsPerRequest = 150;
    options.needResumeKey = false;
    auto result = projector->project(request, options);

    ASSERT_EQ(result.responses.size(), 1);
    const auto& response = result.responses[0];
    EXPECT_EQ(projector->stats().numProjectedRows, 150);
    ASSERT_FALSE(response.slices.empty());
    EXPECT_FALSE(response.resumeKey.has_value());
    EXPECT_FALSE(readEmbeddedResumeKey(response.slices.back()).has_value());
  }

  // maxBytes still truncates the stripes read, but with needResumeKey unset no
  // resume key is produced.
  {
    uint64_t bytesPerStripe = 0;
    {
      auto probe = createProjector(subfields);
      auto bounds = makeRangeLookup(rowType, {"key"}, 0, 100);
      NimbleIndexProjector::Request request;
      request.keyBounds = {bounds};
      auto result = probe->project(request, {});
      bytesPerStripe = probe->stats().numOutputBytes;
      ASSERT_GT(bytesPerStripe, 0);
    }

    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxBytes = bytesPerStripe * 2;
    options.needResumeKey = false;
    auto result = projector->project(request, options);

    ASSERT_EQ(result.responses.size(), 1);
    const auto& response = result.responses[0];
    // Soft byte limit truncates before all 5 stripes are read.
    EXPECT_LT(projector->stats().numReadStripes, 5);
    ASSERT_FALSE(response.slices.empty());
    EXPECT_FALSE(response.resumeKey.has_value());
    EXPECT_FALSE(readEmbeddedResumeKey(response.slices.back()).has_value());
  }
}

TEST_P(NimbleIndexProjectorTest, maxRowsPerRequestResumeKeyPagination) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 10 stripes x 100 rows = 1000 rows, keys 0..999.
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  // maxRowsPerRequest=150 cuts mid-stripe. With needResumeKey, the row-precise
  // resume key lets the caller paginate the full range with no gaps or overlap:
  // every page returns exactly 150 rows until the final tail page.
  auto currentBounds = makeRangeLookup(rowType, {"key"}, 0, 1000);
  uint64_t totalRows = 0;
  int pages = 0;
  while (pages < 100) {
    ++pages;
    auto projector = createProjector(subfields);
    NimbleIndexProjector::Request request;
    request.keyBounds = {currentBounds};
    NimbleIndexProjector::Options options;
    options.maxRowsPerRequest = 150;
    options.needResumeKey = true;
    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    const auto& response = result.responses[0];

    uint64_t pageRows = 0;
    for (const auto& slice : response.slices) {
      pageRows += readEmbeddedRowRange(slice).numRows();
    }
    totalRows += pageRows;
    if (!response.resumeKey.has_value()) {
      break;
    }
    EXPECT_EQ(pageRows, 150) << "page " << pages;
    currentBounds = velox::serializer::EncodedKeyBounds{
        .lowerKey = response.resumeKey.value(),
        .upperKey = currentBounds.upperKey};
  }
  EXPECT_LT(pages, 100);
  EXPECT_EQ(totalRows, 1000);
}

TEST_P(
    NimbleIndexProjectorTest,
    maxRowsPerRequestMultiRequestMidStripeClipping) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 10 stripes x 100 rows.
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // Request 0: [0, 300) spans stripes 0-2.
  // Request 1: [0, 300) same range.
  // maxRowsPerRequest=150: both get 100 (stripe 0) + 50 (clipped stripe 1).
  auto bounds0 = makeRangeLookup(rowType, {"key"}, 0, 300);
  auto bounds1 = makeRangeLookup(rowType, {"key"}, 0, 300);

  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds0, bounds1};
  NimbleIndexProjector::Options options;
  options.maxRowsPerRequest = 150;
  auto result = projector->project(request, options);

  ASSERT_EQ(result.responses.size(), 2);
  for (int i = 0; i < 2; ++i) {
    SCOPED_TRACE(fmt::format("request {}", i));
    const auto& response = result.responses[i];
    uint64_t totalRows = 0;
    for (const auto& slice : response.slices) {
      totalRows += readEmbeddedRowRange(slice).numRows();
    }
    EXPECT_EQ(totalRows, 150);
    EXPECT_FALSE(response.resumeKey.has_value());
    ASSERT_EQ(response.slices.size(), 2);
    EXPECT_EQ(readEmbeddedRowRange(response.slices[0]).numRows(), 100);
    EXPECT_EQ(readEmbeddedRowRange(response.slices[1]).numRows(), 50);
  }
}

TEST_P(NimbleIndexProjectorTest, maxRowsPerRequestSingleRow) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 10 stripes x 100 rows.
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // maxRowsPerRequest=1: each request gets exactly 1 row, clipped from the
  // first row of the first matching stripe.
  auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds};
  NimbleIndexProjector::Options options;
  options.maxRowsPerRequest = 1;
  auto result = projector->project(request, options);

  ASSERT_EQ(result.responses.size(), 1);
  const auto& response = result.responses[0];
  EXPECT_FALSE(response.resumeKey.has_value());
  EXPECT_EQ(projector->stats().numProjectedRows, 1);
  ASSERT_EQ(response.slices.size(), 1);
  EXPECT_EQ(readEmbeddedRowRange(response.slices[0]).numRows(), 1);
}

TEST_P(NimbleIndexProjectorTest, maxRowsLargeScale) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 50 stripes x 200 rows = 10,000 rows.
  writeResumeKeyTestData(/*rowsPerBatch=*/200, /*numBatches=*/50);

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  // Paginate through all 10,000 rows with maxRows=500 (soft limit).
  // Each page gets at least 500 rows (soft limit includes the full stripe
  // that crosses the threshold, so ~600 rows per page with 200-row stripes).
  auto paginate = [&](uint64_t maxRows) -> uint64_t {
    auto currentBounds = makeRangeLookup(rowType, {"key"}, 0, 10000);
    uint64_t totalReadRows = 0;
    int pages = 0;
    while (pages < 100) {
      ++pages;
      auto projector = createProjector(subfields);
      NimbleIndexProjector::Request request;
      request.keyBounds = {currentBounds};
      NimbleIndexProjector::Options options;
      options.maxRows = maxRows;
      options.needResumeKey = true;
      auto result = projector->project(request, options);
      EXPECT_EQ(result.responses.size(), 1);
      totalReadRows += projector->stats().numProjectedRows;
      if (!result.responses[0].resumeKey.has_value()) {
        break;
      }
      currentBounds = velox::serializer::EncodedKeyBounds{
          .lowerKey = result.responses[0].resumeKey.value(),
          .upperKey = currentBounds.upperKey};
    }
    EXPECT_LT(pages, 100);
    return totalReadRows;
  };

  EXPECT_EQ(paginate(500), 10000);
  EXPECT_EQ(paginate(1000), 10000);
  EXPECT_EQ(paginate(5000), 10000);
}

TEST_P(NimbleIndexProjectorTest, maxRowsMultiRequestMixedSatisfaction) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 10 stripes x 100 rows.
  writeResumeKeyTestData();

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");
  auto projector = createProjector(subfields);

  // Request 0: [0, 50) — within stripe 0, fully satisfied.
  // Request 1: [0, 100) — exactly stripe 0, fully satisfied.
  // Request 2: [0, 500) — spans 5 stripes, truncated by maxRows.
  // Request 3: [950, 1000) — stripe 9, never reached.
  // maxRows=100: processes stripe 0, then stops.
  auto bounds0 = makeRangeLookup(rowType, {"key"}, 0, 50);
  auto bounds1 = makeRangeLookup(rowType, {"key"}, 0, 100);
  auto bounds2 = makeRangeLookup(rowType, {"key"}, 0, 500);
  auto bounds3 = makeRangeLookup(rowType, {"key"}, 950, 1000);

  NimbleIndexProjector::Request request;
  request.keyBounds = {bounds0, bounds1, bounds2, bounds3};
  NimbleIndexProjector::Options options;
  options.maxRows = 100;
  options.needResumeKey = true;
  auto result = projector->project(request, options);

  ASSERT_EQ(result.responses.size(), 4);

  // Request 0: 50 rows, fully satisfied, no resume key.
  EXPECT_FALSE(result.responses[0].slices.empty());
  EXPECT_FALSE(result.responses[0].resumeKey.has_value());

  // Request 1: 100 rows, fully satisfied (ends at stripe boundary), no resume.
  EXPECT_FALSE(result.responses[1].slices.empty());
  EXPECT_FALSE(result.responses[1].resumeKey.has_value());

  // Request 2: 100 rows from stripe 0, has data in stripes 1-4, resume key.
  EXPECT_FALSE(result.responses[2].slices.empty());
  EXPECT_TRUE(result.responses[2].resumeKey.has_value());

  // Request 3: stripe 9 never processed, resume key = original lower key.
  EXPECT_TRUE(result.responses[3].slices.empty());
  ASSERT_TRUE(result.responses[3].resumeKey.has_value());
  EXPECT_EQ(result.responses[3].resumeKey.value(), bounds3.lowerKey);
}

TEST_P(NimbleIndexProjectorTest, maxBytesAndMaxRowsPerRequestInteraction) {
  writeResumeKeyTestData();
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  // Learn per-stripe byte size.
  uint64_t bytesPerStripe;
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 100);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    auto result = projector->project(request, {});
    ASSERT_EQ(result.responses[0].slices.size(), 1);
    bytesPerStripe = projector->stats().numOutputBytes;
  }

  // maxRowsPerRequest clips to 150 rows (mid-stripe), maxBytes allows 5
  // stripes. maxRowsPerRequest is tighter.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRowsPerRequest = 150;
    options.maxBytes = bytesPerStripe * 5;

    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_FALSE(result.responses[0].resumeKey.has_value());
    EXPECT_EQ(projector->stats().numProjectedRows, 150);
  }

  // maxBytes allows ~1 stripe of bytes, maxRowsPerRequest allows 300.
  // maxBytes is tighter — sets resume key.
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 500);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    NimbleIndexProjector::Options options;
    options.maxRowsPerRequest = 300;
    options.maxBytes = bytesPerStripe;
    options.needResumeKey = true;

    auto result = projector->project(request, options);
    ASSERT_EQ(result.responses.size(), 1);
    EXPECT_TRUE(result.responses[0].resumeKey.has_value());
    EXPECT_LE(projector->stats().numReadStripes, 2);
    EXPECT_LT(projector->stats().numReadStripes, 5);
  }
}

TEST_P(NimbleIndexProjectorTest, maxBytesLargeScale) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  // 50 stripes x 200 rows = 10,000 rows.
  writeResumeKeyTestData(/*rowsPerBatch=*/200, /*numBatches=*/50);

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  // Learn per-stripe byte size.
  uint64_t bytesPerStripe;
  {
    auto projector = createProjector(subfields);
    auto bounds = makeRangeLookup(rowType, {"key"}, 0, 200);
    NimbleIndexProjector::Request request;
    request.keyBounds = {bounds};
    auto result = projector->project(request, {});
    ASSERT_EQ(result.responses[0].slices.size(), 1);
    bytesPerStripe = projector->stats().numOutputBytes;
  }

  // Paginate all 10,000 rows with byte limit of ~3 stripes per page.
  auto currentBounds = makeRangeLookup(rowType, {"key"}, 0, 10000);
  uint64_t totalReadRows = 0;
  int pages = 0;
  while (pages < 100) {
    ++pages;
    auto projector = createProjector(subfields);
    NimbleIndexProjector::Request request;
    request.keyBounds = {currentBounds};
    NimbleIndexProjector::Options options;
    options.maxBytes = bytesPerStripe * 3;
    options.needResumeKey = true;
    auto result = projector->project(request, options);
    EXPECT_EQ(result.responses.size(), 1);
    totalReadRows += projector->stats().numProjectedRows;
    if (!result.responses[0].resumeKey.has_value()) {
      break;
    }
    currentBounds = velox::serializer::EncodedKeyBounds{
        .lowerKey = result.responses[0].resumeKey.value(),
        .upperKey = currentBounds.upperKey};
  }
  EXPECT_LT(pages, 100);
  EXPECT_EQ(totalReadRows, 10000);
}

TEST_P(NimbleIndexProjectorTest, requiresIoStats) {
  auto batch = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>({1, 2, 3}),
       vectorMaker_->flatVector<int32_t>({10, 20, 30})});
  writeData({batch}, {"key"});

  std::vector<Subfield> subfields;
  subfields.emplace_back("value");

  struct TestCase {
    bool setDataIoStats;
    bool setMetadataIoStats;
    bool setIndexIoStats;
    std::string expectedMessage;
  };

  const std::vector<TestCase> testCases = {
      {false,
       true,
       true,
       "NimbleIndexProjector requires ReaderOptions::dataIoStats to be set"},
      {true,
       false,
       true,
       "NimbleIndexProjector requires ReaderOptions::metadataIoStats to be "
       "set"},
      {true,
       true,
       false,
       "NimbleIndexProjector requires ReaderOptions::indexIoStats to be set"},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.expectedMessage);
    NIMBLE_ASSERT_THROW(
        createProjector(
            subfields,
            /*setDataIoStats=*/testCase.setDataIoStats,
            /*setMetadataIoStats=*/testCase.setMetadataIoStats,
            /*setIndexIoStats=*/testCase.setIndexIoStats),
        testCase.expectedMessage);
  }
}

INSTANTIATE_TEST_CASE_P(
    AllCacheParams,
    NimbleIndexProjectorTest,
    ::testing::ValuesIn(NimbleIndexProjectorTest::getTestParams()),
    [](const ::testing::TestParamInfo<TestParam>& info) {
      return fmt::format(
          "cacheMeta{}_pin{}",
          info.param.cacheMetadata ? "On" : "Off",
          info.param.pinMetadata ? "On" : "Off");
    });

} // namespace facebook::nimble::test
