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

#include "velox/dwio/nimble/velox/index/tests/NimbleIndexTestBase.h"

#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/ScopeGuard.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Timeout.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestUtils.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/serializer/StreamDataParser.h"
#include "velox/dwio/nimble/serializer/StreamReader.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/FileProperties.h"
#include "velox/dwio/nimble/tablet/FilePropertiesGenerated.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/TabletReaderCache.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/velox/BatchReader.h"
#include "velox/dwio/nimble/velox/ChunkedStreamWriter.h"
#include "velox/dwio/nimble/velox/HybridFlatMap.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"
#include "velox/dwio/nimble/velox/SharedDictionaryConfig.h"
#include "velox/dwio/nimble/velox/index/NimbleIterator.h"
#include "velox/dwio/nimble/velox/index/StripeStreams.h"
#include "velox/dwio/nimble/velox/tests/SchemaUtils.h"
#include "velox/dwio/nimble/velox/tests/SharedDictionaryTestUtils.h"
#include "velox/dwio/nimble/writer/Writer.h"

#include "folly/FileUtil.h"
#include "folly/Random.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/file/IoUringReader.h"
#include "velox/common/file/LocalFile.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/row/CompactRow.h"
#include "velox/row/UnsafeRowFast.h"
#include "velox/serializers/KeyEncoder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble::test {

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

std::string nextTestOutputRowFormat(std::string_view prefix) {
  static std::atomic<std::size_t> nextFormatId{0};
  return fmt::format(
      "{}_{}", prefix, nextFormatId.fetch_add(1, std::memory_order_relaxed));
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

HybridFlatMap makeHybridFlatMapConfig(
    const std::vector<std::vector<std::string>>& explicitGroups) {
  HybridFlatMap hybridMap;
  hybridMap.groups.reserve(explicitGroups.size() + 1);
  for (uint32_t groupId = 0; groupId < explicitGroups.size(); ++groupId) {
    hybridMap.groups.push_back(
        {.groupId = groupId, .groupKeys = explicitGroups[groupId]});
  }
  hybridMap.groups.push_back(
      {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}});
  return hybridMap;
}

NimbleIndexProjector::RowProjectionResult projectRows(
    NimbleIndexProjector& projector,
    const NimbleIndexProjector::Request& request,
    const NimbleIndexProjector::Options& options) {
  return folly::coro::blockingWait(projector.co_projectRows(request, options));
}

void NimbleIndexProjectorTestBase::SetUpTestCase() {
  memory::MemoryManager::testingSetInstance({});
}

void NimbleIndexProjectorTestBase::SetUp() {
  rootPool_ = memory::memoryManager()->addRootPool("NimbleIndexProjectorTest");
  leafPool_ = rootPool_->addLeafChild("leaf");
  vectorMaker_ = std::make_unique<velox::test::VectorMaker>(leafPool_.get());
  ioExecutor_ = std::make_shared<folly::CPUThreadPoolExecutor>(4);
}

void NimbleIndexProjectorTestBase::TearDown() {
  vectorMaker_.reset();
  leafPool_.reset();
  rootPool_.reset();
  ioExecutor_.reset();
  TabletReaderCache::testingReset();
}

RowTypePtr NimbleIndexProjectorTestBase::singleFeatureRowType() {
  return ROW({{"key", BIGINT()}, {"features", MAP(VARCHAR(), DOUBLE())}});
}

RowVectorPtr NimbleIndexProjectorTestBase::singleFeatureBatch(
    int64_t keyBase,
    StringView featureKey) {
  auto offsets = allocateOffsets(kFeatureRowsPerBatch, leafPool_.get());
  auto sizes = allocateSizes(kFeatureRowsPerBatch, leafPool_.get());
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  auto* rawSizes = sizes->asMutable<vector_size_t>();
  for (vector_size_t row = 0; row < kFeatureRowsPerBatch; ++row) {
    rawOffsets[row] = row;
    rawSizes[row] = 1;
  }
  auto features = std::make_shared<MapVector>(
      leafPool_.get(),
      MAP(VARCHAR(), DOUBLE()),
      nullptr,
      kFeatureRowsPerBatch,
      offsets,
      sizes,
      vectorMaker_->flatVector<StringView>(
          kFeatureRowsPerBatch, [&](auto /*row*/) { return featureKey; }),
      vectorMaker_->flatVector<double>(kFeatureRowsPerBatch, [](auto row) {
        return static_cast<double>(row);
      }));
  return vectorMaker_->rowVector(
      {"key", "features"},
      {vectorMaker_->flatVector<int64_t>(
           kFeatureRowsPerBatch, [keyBase](auto row) { return keyBase + row; }),
       features});
}

EncodingSelectionPolicyCreator
NimbleIndexProjectorTestBase::encodingViewSelectionPolicy() {
  auto readFactors =
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
  std::erase_if(readFactors, [](const auto& entry) {
    return !supportsEncodingView(entry.first);
  });
  // Nested policies inherit these filtered candidates.
  return [encodingFactory = ManualEncodingSelectionPolicyFactory{
              std::move(readFactors)}](DataType dataType) {
    return encodingFactory.createPolicy(dataType);
  };
}

void NimbleIndexProjectorTestBase::writeData(
    const std::vector<RowVectorPtr>& batches,
    const std::vector<std::string>& indexColumns,
    const folly::F14FastMap<std::string, std::set<std::string>>& flatMapColumns,
    uint64_t stripeSize,
    bool enableStreamDeduplication,
    bool compactRowCountEncoding,
    CompressionType chunkCompressionType,
    bool skipConstantFlatMapInMapStreams,
    bool enableChunking,
    bool enableStreamChecksums,
    StripeGroup::EncodingLayout stripeGroupEncodingLayout,
    std::optional<uint32_t> metadataFlushThreshold) {
  WriterOptions options;
  options.encodingSelectionPolicyCreator = encodingViewSelectionPolicy();
  options.metadataFlushThreshold = metadataFlushThreshold;
  options.enableStreamChecksums = enableStreamChecksums;
  options.experimentalStripeGroupEncodingLayout = stripeGroupEncodingLayout;
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
  options.clusterIndexConfig = makeClusterIndexConfig(indexColumns);

  options.flushPolicyFactory = [stripeSize]() {
    return std::make_unique<LambdaFlushPolicy>(
        /*shouldFlush*/
        [stripeSize](const StripeProgress& progress) {
          return progress.stripeRawSize >= stripeSize;
        });
  };
  writeBatches(batches, std::move(options));
}

std::shared_ptr<const index::IndexConfig>
NimbleIndexProjectorTestBase::makeClusterIndexConfig(
    const std::vector<std::string>& indexColumns) {
  return ClusterIndexConfigBuilder{}
      .withKeyColumns(indexColumns)
      .withSortOrders(
          std::vector<SortOrder>(
              indexColumns.size(), SortOrder{.ascending = true}))
      .withEnforceKeyOrder(true)
      .withNoDuplicateKey(true)
      .build();
}

void NimbleIndexProjectorTestBase::writeBatches(
    const std::vector<RowVectorPtr>& batches,
    WriterOptions options) {
  sinkData_.clear();
  auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
  auto rowType = asRowType(batches[0]->type());
  Writer writer(rowType, std::move(writeFile), *rootPool_, std::move(options));
  for (const auto& batch : batches) {
    writer.write(batch);
  }
  writer.close();
  resetReaders();
}

void NimbleIndexProjectorTestBase::setFile(std::string data) {
  sinkData_ = std::move(data);
  resetReaders();
}

void NimbleIndexProjectorTestBase::resetReaders() {
  TabletReaderCache::testingReset();
  tabletReaderCache_.reset();
  keyEncoder_.reset();
}

std::shared_ptr<TabletReader> NimbleIndexProjectorTestBase::openSinkTablet() {
  auto readFile =
      std::make_shared<InMemoryReadFile>(std::string_view(sinkData_));
  dwio::common::ReaderOptions tabletReaderOptions(leafPool_.get());
  tabletReaderOptions.setFileFormat(FileFormat::NIMBLE);
  return TabletReader::create(
      readFile,
      leafPool_.get(),
      TabletReader::configureOptions(tabletReaderOptions));
}

std::vector<TestParam> NimbleIndexProjectorTestBase::getTestParams() {
  std::vector<TestParam> params;
  for (bool cacheMetadata : {false, true}) {
    for (bool pinMetadata : {false, true}) {
      params.push_back({cacheMetadata, pinMetadata});
    }
  }
  return params;
}

void NimbleIndexProjectorTestBase::corruptProjectedStreams() {
  corruptStripe0Streams([](const TabletReader::StreamMetadata& location) {
    return location.size / 2;
  });
}

void NimbleIndexProjectorTestBase::corruptStreamTrailers() {
  const auto checksumOffset =
      openSinkTablet()->properties().streamTrailerLayout().checksumOffset();
  ASSERT_TRUE(checksumOffset.has_value());
  corruptStripe0Streams(
      [&](const TabletReader::StreamMetadata& location) -> uint64_t {
        return location.size + *checksumOffset;
      });
}

size_t NimbleIndexProjectorTestBase::countDeduplicatedStreams() {
  auto tablet = openSinkTablet();
  const auto stripe = tablet->stripeIdentifier(0);
  std::vector<TabletReader::StreamMetadata> locations(
      tablet->streamCount(stripe));
  tablet->streamLocations(stripe, locations);
  std::set<std::pair<uint32_t, uint32_t>> unique;
  size_t nonEmpty{0};
  for (const auto& location : locations) {
    if (location.size == 0) {
      continue;
    }
    ++nonEmpty;
    unique.emplace(location.offset, location.size);
  }
  return nonEmpty - unique.size();
}

void NimbleIndexProjectorTestBase::setFirstStreamTrailerFieldKind(
    uint8_t kind) {
  auto tablet = openSinkTablet();
  const auto& sections = tablet->optionalSections();
  const auto it = sections.find(std::string{kPropertiesSection});
  ASSERT_NE(it, sections.end());
  ASSERT_EQ(it->second.compressionType(), CompressionType::Uncompressed);
  const auto* properties = flatbuffers::GetRoot<serialization::FileProperties>(
      sinkData_.data() + it->second.offset());
  ASSERT_NE(properties->stream_trailer(), nullptr);
  ASSERT_GT(properties->stream_trailer()->size(), 0);

  // `kind` is the first byte of the field struct.
  const auto kindOffset =
      reinterpret_cast<const char*>(properties->stream_trailer()->Get(0)) -
      sinkData_.data();
  sinkData_[kindOffset] = static_cast<char>(kind);
}

velox::FileHandle NimbleIndexProjectorTestBase::makeFileHandle() {
  auto readFile =
      std::make_shared<InMemoryReadFile>(std::string_view(sinkData_));
  return makeFileHandle(std::move(readFile), "test_file");
}

velox::FileHandle NimbleIndexProjectorTestBase::makeFileHandle(
    std::shared_ptr<velox::ReadFile> readFile,
    std::string_view id) {
  return velox::FileHandle{
      std::move(readFile),
      velox::StringIdLease(velox::fileIds(), id),
      velox::StringIdLease(velox::fileIds(), id)};
}

std::unique_ptr<NimbleIndexProjector>
NimbleIndexProjectorTestBase::createProjector(
    const std::vector<Subfield>& projectedSubfields,
    bool setDataIoStats,
    bool setMetadataIoStats,
    bool setIndexIoStats) {
  return createProjectorWithFileHandle(
      projectedSubfields,
      makeFileHandle(),
      setDataIoStats,
      setMetadataIoStats,
      setIndexIoStats);
}

std::unique_ptr<NimbleIterator> NimbleIndexProjectorTestBase::createRowIterator(
    const std::vector<Subfield>& projectedSubfields,
    const NimbleIterator::Options& options) {
  return createIterator(
      NimbleIterator::createRowIterator, projectedSubfields, options);
}

std::unique_ptr<NimbleIterator>
NimbleIndexProjectorTestBase::createBatchIterator(
    const std::vector<Subfield>& projectedSubfields,
    const NimbleIterator::Options& options) {
  return createIterator(
      NimbleIterator::createBatchIterator, projectedSubfields, options);
}

std::unique_ptr<NimbleIndexProjector>
NimbleIndexProjectorTestBase::createProjectorWithFileHandle(
    const std::vector<Subfield>& projectedSubfields,
    const velox::FileHandle& fileHandle,
    bool setDataIoStats,
    bool setMetadataIoStats,
    bool setIndexIoStats,
    bool cacheData,
    bool verifyStreamChecksums) {
  dwio::common::ReaderOptions readerOptions(leafPool_.get());
  configureReaderOptions(
      readerOptions,
      setDataIoStats,
      setMetadataIoStats,
      setIndexIoStats,
      cacheData,
      verifyStreamChecksums);
  ensureTabletReaderCache();
  return NimbleIndexProjector::create(
      *tabletReaderCache_, fileHandle, projectedSubfields, readerOptions);
}

void NimbleIndexProjectorTestBase::configureReaderOptions(
    dwio::common::ReaderOptions& readerOptions,
    bool setDataIoStats,
    bool setMetadataIoStats,
    bool setIndexIoStats,
    bool cacheData,
    bool verifyStreamChecksums) {
  dataIoStats_ = std::make_shared<velox::io::IoStatistics>();
  metadataIoStats_ = std::make_shared<velox::io::IoStatistics>();
  indexIoStats_ = std::make_shared<velox::io::IoStatistics>();
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
  readerOptions.setVerifyStreamChecksums(verifyStreamChecksums);
  readerOptions.setCacheData(cacheData);
  if (cacheData) {
    readerOptions.setCache(velox::checkedNotNull(dataCache()));
  }
  const auto testParam = currentTestParam();
  readerOptions.setCacheMetadata(testParam.cacheMetadata);
  readerOptions.setPinMetadata(testParam.pinMetadata);
  readerOptions.setFilePreloadThreshold(0);
  readerOptions.setIOExecutor(ioExecutor_);
}

VectorPtr NimbleIndexProjectorTestBase::deserializeProjectedBatches(
    const NimbleIndexProjector& projector,
    const std::vector<std::string_view>& batches) {
  DeserializerOptions deserOptions;
  Deserializer deserializer(
      projector.projectedNimbleType(), leafPool_.get(), deserOptions);
  VectorPtr output;
  deserializer.deserialize(batches, output);
  return output;
}

VectorPtr NimbleIndexProjectorTestBase::deserializeProjectedSlice(
    const NimbleIndexProjector& projector,
    const folly::IOBuf& slice) {
  auto coalesced = coalesceChunkSlice(slice);
  std::vector<std::string_view> batches{std::string_view(
      reinterpret_cast<const char*>(coalesced.data()), coalesced.length())};
  return deserializeProjectedBatches(projector, batches);
}

NimbleIndexProjectorTestBase::WindowedSlice
NimbleIndexProjectorTestBase::makeWindowedSlice() {
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
  auto result = projector->projectStreams(request, {});
  return {std::move(projector), std::move(result), std::move(values)};
}

void NimbleIndexProjectorTestBase::expectWindow(const WindowedSlice& windowed) {
  ASSERT_EQ(windowed.result.responses[0].slices.size(), 1);
  const auto window = readEmbeddedRowRange(windowed.slice());
  ASSERT_EQ(window.startRow, kWindowStart);
  ASSERT_EQ(window.endRow, kWindowEnd);
}

VectorPtr NimbleIndexProjectorTestBase::deserializeProjectedSlice(
    const NimbleIndexProjector& projector,
    const folly::IOBuf& slice,
    const std::vector<RowRange>& rowRanges) {
  auto coalesced = coalesceChunkSlice(slice);
  std::vector<std::string_view> batches{std::string_view(
      reinterpret_cast<const char*>(coalesced.data()), coalesced.length())};
  DeserializerOptions deserOptions;
  Deserializer deserializer(
      projector.projectedNimbleType(), leafPool_.get(), deserOptions);
  VectorPtr output;
  deserializer.deserialize(batches, rowRanges, output);
  return output;
}

void NimbleIndexProjectorTestBase::expectVectorRows(
    const VectorPtr& output,
    const VectorPtr& expected) {
  ASSERT_NE(output, nullptr);
  ASSERT_NE(expected, nullptr);
  velox::test::assertEqualVectors(expected, output);
}

velox::serializer::EncodedKeyBounds
NimbleIndexProjectorTestBase::makePointLookup(
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

velox::serializer::EncodedKeyBounds
NimbleIndexProjectorTestBase::makeRangeLookup(
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

void NimbleIndexProjectorTestBase::writeResumeKeyTestData(
    int rowsPerBatch,
    int numBatches) {
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

void NimbleIndexProjectorTestBase::ensureTabletReaderCache() {
  if (tabletReaderCache_ == nullptr) {
    TabletReaderCache::Options opts;
    opts.numShards = 2;
    opts.maxEntries = 100;
    opts.executor = std::make_shared<folly::CPUThreadPoolExecutor>(4);
    tabletReaderCache_ = std::make_unique<TabletReaderCache>(opts);
  }
}

velox::cache::AsyncDataCache* NimbleIndexProjectorTestBase::dataCache() const {
  return nullptr;
}

void NimbleIndexProjectorTest::SetUpTestCase() {
  NimbleIndexProjectorTestBase::SetUpTestCase();
  dataCache_ = velox::cache::AsyncDataCache::create(
      velox::memory::memoryManager()->allocator());
}

void NimbleIndexProjectorTest::TearDownTestCase() {
  dataCache_->shutdown();
  dataCache_.reset();
}

void NimbleIndexProjectorTest::TearDown() {
  NimbleIndexProjectorTestBase::TearDown();
  dataCache_->clear();
}

TestParam NimbleIndexProjectorTest::currentTestParam() const {
  return GetParam();
}

velox::cache::AsyncDataCache* NimbleIndexProjectorTest::dataCache() const {
  return dataCache_.get();
}

} // namespace facebook::nimble::test
