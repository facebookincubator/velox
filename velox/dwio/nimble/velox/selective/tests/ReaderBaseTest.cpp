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

#include "velox/dwio/nimble/velox/selective/ReaderBase.h"

#include <gtest/gtest.h>

#include <limits>
#include <optional>
#include <set>
#include <vector>

#include "folly/executors/CPUThreadPoolExecutor.h"
#include "velox/common/file/File.h"
#include "velox/common/file/tests/TestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/tablet/FileLayout.h"
#include "velox/dwio/nimble/tablet/TabletReaderCache.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

#include <fmt/core.h>

using namespace facebook::nimble;
using namespace facebook;

namespace {

enum class CreationMode { kDirect, kCached };

int32_t sharedDictionaryValue(velox::vector_size_t row) {
  return row % 2 == 0 ? 0 : std::numeric_limits<int32_t>::max();
}

} // namespace

class ReaderBaseTest : public ::testing::TestWithParam<CreationMode> {
 protected:
  static inline const velox::RowTypePtr kSchema =
      velox::ROW({"a", "b"}, {velox::BIGINT(), velox::VARCHAR()});

  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool("test");
    executor_ = std::make_shared<folly::CPUThreadPoolExecutor>(4);
    fileData_ = writeTestFile();
  }

  void TearDown() override {
    TabletReaderCache::testingReset();
  }

  std::string writeTestFile() {
    std::string file;
    velox::test::VectorMaker vectorMaker(pool_.get());
    auto vector = vectorMaker.rowVector(
        {"a", "b"},
        {vectorMaker.flatVector<int64_t>(200, [](auto i) { return i; }),
         vectorMaker.flatVector<velox::StringView>(200, [](auto i) {
           return velox::StringView::makeInline("val" + std::to_string(i));
         })});

    WriterOptions writerOptions;
    auto writer = std::make_unique<Writer>(
        kSchema,
        std::make_unique<velox::InMemoryWriteFile>(&file),
        *pool_,
        std::move(writerOptions));
    writer->write(vector);
    writer->close();
    return file;
  }

  velox::dwio::common::ReaderOptions makeReaderOptions() {
    velox::dwio::common::ReaderOptions opts(pool_.get());
    opts.setDataIoStats(std::make_shared<velox::io::IoStatistics>());
    opts.setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
    opts.setIndexIoStats(std::make_shared<velox::io::IoStatistics>());
    return opts;
  }

  std::shared_ptr<ReaderBase> createReaderBase() {
    return createReaderBase(fileData_);
  }

  std::shared_ptr<ReaderBase> createReaderBase(const std::string& fileData) {
    auto readFile = std::make_shared<velox::InMemoryReadFile>(fileData);
    auto readerOpts = makeReaderOptions();

    switch (GetParam()) {
      case CreationMode::kDirect: {
        auto input = std::make_unique<velox::dwio::common::BufferedInput>(
            readFile, *pool_);
        return ReaderBase::create(std::move(input), readerOpts);
      }
      case CreationMode::kCached: {
        ensureCache();
        auto tabletOpts = TabletReader::configureOptions(readerOpts);
        auto cached = cache_->get(readFile, tabletOpts);

        auto input = std::make_unique<velox::dwio::common::BufferedInput>(
            readFile, *pool_);
        return ReaderBase::create(std::move(input), cached, readerOpts);
      }
    }
    VELOX_UNREACHABLE();
  }

  std::string writeSharedDictionaryFile(SharedDictionaryScope scope) {
    std::string file;
    velox::test::VectorMaker vectorMaker(pool_.get());
    auto vector = vectorMaker.rowVector(
        {"shared", "plain"},
        {vectorMaker.mapVector<int64_t, int32_t>(
             2'000,
             [](auto /*row*/) { return 1; },
             [](auto /*idx*/) { return 10; },
             sharedDictionaryValue),
         vectorMaker.flatVector<int32_t>(
             2'000, [](auto row) { return row + 10'000; })});

    WriterOptions writerOptions;
    writerOptions.maxStreamChunkRawSize = 512;
    writerOptions.minStreamChunkRawSize = 1;
    writerOptions.encodingSelectionPolicyCreator = [](DataType dataType) {
      std::vector<std::pair<EncodingType, float>> readFactors;
      if (dataType == DataType::Uint32) {
        readFactors = {{EncodingType::FixedBitWidth, 1.0}};
      } else if (dataType == DataType::Int32) {
        readFactors = {
            {EncodingType::Trivial, 1.0}, {EncodingType::Dictionary, 1.0}};
      } else {
        readFactors = {{EncodingType::Trivial, 1.0}};
      }
      return ManualEncodingSelectionPolicyFactory{
          std::move(readFactors), /*compressionOptions=*/std::nullopt}
          .createPolicy(dataType);
    };
    writerOptions.flatMapColumns.emplace("shared", std::set<std::string>{});
    writerOptions.experimentalSharedDictionaryEncoding =
        SharedDictionaryEncodingConfig::builder()
            .addFlatmapValueDictionary(
                "shared",
                /*key=*/10,
                SharedDictionaryConfig{.scope = scope})
            .build();

    auto writer = std::make_unique<Writer>(
        vector->type(),
        std::make_unique<velox::InMemoryWriteFile>(&file),
        *pool_,
        std::move(writerOptions));
    writer->write(vector);
    writer->close();
    return file;
  }

  uint32_t sharedDictionaryStreamId(const ReaderBase& reader) const {
    const auto& flatMap =
        reader.nimbleSchema()->asRow().childAt(0)->asFlatMap();
    auto child = flatMap.findChild("10");
    NIMBLE_CHECK(child.has_value());
    return flatMap.childAt(child.value())
        ->asScalar()
        .scalarDescriptor()
        .offset();
  }

  void ensureCache() {
    if (cache_ == nullptr) {
      TabletReaderCache::Options cacheOpts;
      cacheOpts.numShards = 2;
      cacheOpts.maxEntries = 10;
      cacheOpts.executor = executor_;
      cache_ = std::make_unique<TabletReaderCache>(cacheOpts);
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
  std::string fileData_;
  std::unique_ptr<TabletReaderCache> cache_;
};

// A ReaderBase built through the cache has to keep the cache ENTRY alive, not
// just the tablet. The entry owns the IoStatistics the tablet writes its
// metadata and index reads into, and retiring it hands those totals off and
// stops watching them -- so an entry retiring under a live reader would leave
// the rest of its IO reported by nobody. Holding only tablet() would not keep
// the entry alive, because the tablet has its own reference to the statistics.
class ReaderBaseCacheLifetimeTest : public ReaderBaseTest {
 protected:
  // InMemoryReadFile reports the same name for every instance, and the cache
  // keys on the name, so distinct entries need distinct names.
  class NamedReadFile : public velox::InMemoryReadFile {
   public:
    NamedReadFile(std::string name, std::string_view data)
        : velox::InMemoryReadFile(data), name_{std::move(name)} {}

    std::string getName() const override {
      return name_;
    }

   private:
    const std::string name_;
  };
};

TEST_F(ReaderBaseCacheLifetimeTest, evictionDoesNotRetireEntryUnderLiveReader) {
  bool released = false;
  TabletReaderCache::Options cacheOpts;
  cacheOpts.numShards = 1;
  // One slot, so opening a second file evicts the first.
  cacheOpts.maxEntries = 1;
  cacheOpts.executor = executor_;
  cacheOpts.onCreate = [](const CachedTabletReader&) {};
  cacheOpts.onRelease = [&](const CachedTabletReader&) { released = true; };
  // Outlives every entry: a TabletReader frees its metadata buffers back to a
  // pool the cache owns, so an entry must never outlive the cache.
  TabletReaderCache cache(cacheOpts);

  auto readerOpts = makeReaderOptions();
  auto readFile = std::make_shared<NamedReadFile>("file0", fileData_);
  std::shared_ptr<ReaderBase> reader;
  {
    auto cached =
        cache.get(readFile, TabletReader::configureOptions(readerOpts));
    auto input =
        std::make_unique<velox::dwio::common::BufferedInput>(readFile, *pool_);
    reader = ReaderBase::create(std::move(input), cached, readerOpts);
  }
  EXPECT_FALSE(released);

  // Evicts file0. The cache's reference was the only other one, so unless the
  // reader owns the entry this retires it while the reader is still usable.
  auto other = std::make_shared<NamedReadFile>("file1", fileData_);
  auto evicting = cache.get(other, TabletReader::configureOptions(readerOpts));
  EXPECT_FALSE(released)
      << "the reader must own the entry, not just the tablet";
  // Still usable, which is the point: more stripe reads can happen through it.
  EXPECT_GT(reader->tablet().stripeCount(), 0);

  reader.reset();
  EXPECT_TRUE(released);
}

TEST_P(ReaderBaseTest, basic) {
  auto reader = createReaderBase();

  EXPECT_TRUE(reader->fileSchema()->equivalent(*kSchema));
  EXPECT_EQ(reader->fileSchema()->size(), 2);
  EXPECT_EQ(reader->fileSchema()->nameOf(0), "a");
  EXPECT_EQ(reader->fileSchema()->nameOf(1), "b");

  ASSERT_NE(reader->nimbleSchema(), nullptr);
  EXPECT_TRUE(reader->nimbleSchema()->isRow());
  EXPECT_EQ(reader->nimbleSchema()->asRow().childrenCount(), 2);

  EXPECT_EQ(reader->tablet().stripeCount(), 1);
  EXPECT_EQ(reader->tablet().tabletRowCount(), 200);
  EXPECT_EQ(reader->pool(), pool_.get());

  const auto& schemaWithId = reader->fileSchemaWithId();
  ASSERT_NE(schemaWithId, nullptr);
  EXPECT_EQ(schemaWithId->size(), 2);
  EXPECT_EQ(reader->fileSchemaWithId().get(), schemaWithId.get());
}

TEST_P(ReaderBaseTest, equivalentAcrossModes) {
  auto readFile = std::make_shared<velox::InMemoryReadFile>(fileData_);
  auto readerOpts = makeReaderOptions();

  auto inputDirect =
      std::make_unique<velox::dwio::common::BufferedInput>(readFile, *pool_);
  auto direct = ReaderBase::create(std::move(inputDirect), readerOpts);

  ensureCache();
  auto tabletOpts = TabletReader::configureOptions(readerOpts);
  auto cached = cache_->get(readFile, tabletOpts);

  auto inputCached =
      std::make_unique<velox::dwio::common::BufferedInput>(readFile, *pool_);
  auto fromCache =
      ReaderBase::create(std::move(inputCached), cached, readerOpts);

  EXPECT_TRUE(direct->fileSchema()->equivalent(*fromCache->fileSchema()));
  EXPECT_EQ(
      direct->nimbleSchema()->asRow().childrenCount(),
      fromCache->nimbleSchema()->asRow().childrenCount());
  EXPECT_EQ(direct->tablet().stripeCount(), fromCache->tablet().stripeCount());
  EXPECT_EQ(
      direct->tablet().tabletRowCount(), fromCache->tablet().tabletRowCount());
}

TEST_P(ReaderBaseTest, locateStreams) {
  auto reader = createReaderBase();
  StripeStreams streams(reader);
  streams.setStripe(0);

  const auto numStreams = reader->nimbleSchema()->asRow().childrenCount() + 1;

  struct TestCase {
    std::vector<uint32_t> streamIds;
    std::string debugString() const {
      return fmt::format("streamIds=[{}]", fmt::join(streamIds, ","));
    }
  };

  std::vector<uint32_t> allStreamIds;
  allStreamIds.reserve(numStreams);
  for (uint32_t i = 0; i < numStreams; ++i) {
    allStreamIds.push_back(i);
  }

  std::vector<TestCase> testCases = {
      {allStreamIds},
      {{9999}},
      {{}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    auto locations = streams.locateStreams(testCase.streamIds);
    ASSERT_EQ(locations.size(), testCase.streamIds.size());

    for (size_t i = 0; i < locations.size(); ++i) {
      SCOPED_TRACE(fmt::format("streamId={}", testCase.streamIds[i]));
      if (locations[i].has_value()) {
        EXPECT_EQ(locations[i]->streamId, testCase.streamIds[i]);
        EXPECT_GT(locations[i]->region.length, 0);
      }
    }
  }
}

TEST_P(ReaderBaseTest, fileDictionaryAlphabetLoader) {
  const auto fileData = writeSharedDictionaryFile(SharedDictionaryScope::File);
  auto reader = createReaderBase(fileData);
  StripeStreams streams(reader);
  streams.setStripe(0);

  const auto& row = reader->nimbleSchema()->asRow();
  const auto sharedStreamId = sharedDictionaryStreamId(*reader);
  const auto plainStreamId =
      row.childAt(1)->asScalar().scalarDescriptor().offset();

  EXPECT_TRUE(reader->tablet().hasFileOrExternalDictionaries());
  EXPECT_FALSE(
      reader->tablet().stripeDictionaryStreamId(sharedStreamId).has_value());
  const auto expectedAlphabet =
      reader->tablet().resolveDictionaryAlphabet(sharedStreamId);
  ASSERT_NE(expectedAlphabet, nullptr);
  ASSERT_EQ(
      reader->tablet().resolveDictionaryAlphabet(sharedStreamId).get(),
      expectedAlphabet.get());

  auto loader = streams.dictionaryAlphabetLoader(sharedStreamId);
  ASSERT_TRUE(loader);
  EXPECT_EQ(loader(), expectedAlphabet);
  EXPECT_EQ(loader(), expectedAlphabet);

  EXPECT_EQ(reader->tablet().resolveDictionaryAlphabet(plainStreamId), nullptr);
  auto plainLoader = streams.dictionaryAlphabetLoader(plainStreamId);
  EXPECT_FALSE(plainLoader);
}

TEST_P(ReaderBaseTest, stripeDictionaryAlphabetLoader) {
  const auto fileData =
      writeSharedDictionaryFile(SharedDictionaryScope::Stripe);
  auto reader = createReaderBase(fileData);
  StripeStreams streams(reader);
  streams.setStripe(0);

  const auto& row = reader->nimbleSchema()->asRow();
  const auto sharedStreamId = sharedDictionaryStreamId(*reader);
  const auto plainStreamId =
      row.childAt(1)->asScalar().scalarDescriptor().offset();

  EXPECT_TRUE(reader->tablet().hasStripeDictionaries());
  const auto stripeDictionaryStreamId =
      reader->tablet().stripeDictionaryStreamId(sharedStreamId);
  ASSERT_TRUE(stripeDictionaryStreamId.has_value());
  ASSERT_TRUE(streams.hasStream(
      static_cast<int32_t>(stripeDictionaryStreamId.value())));

  auto valueInput = streams.enqueue(sharedStreamId);
  ASSERT_NE(valueInput, nullptr);
  streams.load();

  auto loader = streams.dictionaryAlphabetLoader(sharedStreamId);
  ASSERT_TRUE(loader);
  const auto alphabet = loader();
  ASSERT_NE(alphabet, nullptr);
  EXPECT_EQ(alphabet->entryCount(), 2);
  EXPECT_EQ(alphabet->physicalValueAt<int32_t>(0), 0);
  EXPECT_EQ(
      alphabet->physicalValueAt<int32_t>(1),
      std::numeric_limits<int32_t>::max());
  EXPECT_FALSE(streams.dictionaryAlphabetLoader(sharedStreamId));

  auto plainLoader = streams.dictionaryAlphabetLoader(plainStreamId);
  EXPECT_FALSE(plainLoader);
}

INSTANTIATE_TEST_CASE_P(
    AllCreationModes,
    ReaderBaseTest,
    ::testing::Values(CreationMode::kDirect, CreationMode::kCached),
    [](const ::testing::TestParamInfo<CreationMode>& info) {
      return info.param == CreationMode::kDirect ? "direct" : "cached";
    });

class StripeStreamsMultiStripeTest
    : public ::testing::TestWithParam<StripeGroup::EncodingLayout> {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool("test");
    fileData_ = writeMultiStripeFile();
  }

  std::string writeMultiStripeFile() {
    std::string file;
    velox::test::VectorMaker vectorMaker(pool_.get());

    WriterOptions writerOptions;
    writerOptions.experimentalStripeGroupEncodingLayout = GetParam();
    writerOptions.flushPolicyFactory = []() {
      return std::make_unique<LambdaFlushPolicy>(
          [](const StripeProgress&) { return true; });
    };

    auto writer = std::make_unique<Writer>(
        kSchema,
        std::make_unique<velox::InMemoryWriteFile>(&file),
        *pool_,
        std::move(writerOptions));

    for (int stripe = 0; stripe < 3; ++stripe) {
      const auto offset = stripe * 100;
      auto vector = vectorMaker.rowVector(
          {"a", "b"},
          {vectorMaker.flatVector<int64_t>(
               100, [offset](auto i) { return offset + i; }),
           vectorMaker.flatVector<velox::StringView>(100, [offset](auto i) {
             return velox::StringView::makeInline(
                 "v" + std::to_string(offset + i));
           })});
      writer->write(vector);
    }
    writer->close();
    return file;
  }

  static inline const velox::RowTypePtr kSchema =
      velox::ROW({"a", "b"}, {velox::BIGINT(), velox::VARCHAR()});

  velox::dwio::common::ReaderOptions makeReaderOptions() {
    velox::dwio::common::ReaderOptions opts(pool_.get());
    opts.setDataIoStats(dataIoStats_);
    opts.setMetadataIoStats(metadataIoStats_);
    opts.setIndexIoStats(indexIoStats_);
    return opts;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::string fileData_;
  const std::shared_ptr<velox::io::IoStatistics> dataIoStats_{
      std::make_shared<velox::io::IoStatistics>()};
  const std::shared_ptr<velox::io::IoStatistics> metadataIoStats_{
      std::make_shared<velox::io::IoStatistics>()};
  const std::shared_ptr<velox::io::IoStatistics> indexIoStats_{
      std::make_shared<velox::io::IoStatistics>()};
};

TEST_P(StripeStreamsMultiStripeTest, locateStreams) {
  auto readFile = std::make_shared<velox::InMemoryReadFile>(fileData_);
  auto opts = makeReaderOptions();

  auto input =
      std::make_unique<velox::dwio::common::BufferedInput>(readFile, *pool_);
  auto reader = ReaderBase::create(std::move(input), opts);
  const auto& tablet = reader->tablet();
  ASSERT_EQ(tablet.stripeCount(), 3);

  const auto layout = FileLayout::create(readFile, pool_.get());
  ASSERT_EQ(layout.stripesInfo.size(), 3);

  StripeStreams streams(reader);
  std::vector<uint32_t> streamIds = {1, 2};

  for (uint32_t stripe = 0; stripe < 3; ++stripe) {
    SCOPED_TRACE(fmt::format("stripe={}", stripe));
    streams.setStripe(stripe);
    auto locations = streams.locateStreams(streamIds);
    ASSERT_EQ(locations.size(), streamIds.size());

    const auto stripeId = tablet.stripeIdentifier(stripe);
    const auto stripeOffset = tablet.stripeOffset(stripe);
    const auto& stripeInfo = layout.stripesInfo[stripe];

    for (size_t i = 0; i < locations.size(); ++i) {
      SCOPED_TRACE(fmt::format("streamId={}", streamIds[i]));
      ASSERT_TRUE(locations[i].has_value());
      EXPECT_EQ(locations[i]->streamId, streamIds[i]);
      EXPECT_EQ(
          locations[i]->region.offset,
          stripeOffset + tablet.streamOffset(stripeId, streamIds[i]));
      EXPECT_EQ(
          locations[i]->region.length,
          tablet.streamSize(stripeId, streamIds[i]));
      EXPECT_GE(locations[i]->region.offset, stripeInfo.offset);
      EXPECT_LE(
          locations[i]->region.offset + locations[i]->region.length,
          stripeInfo.offset + stripeInfo.size);
    }
  }
}

TEST_P(StripeStreamsMultiStripeTest, preloadCollapsesStripeReads) {
  constexpr uint32_t kStripeCount = 3;

  auto preadsReadingAllStripes = [&](uint64_t preloadThreshold) {
    auto countingFile =
        std::make_shared<velox::tests::utils::CountingReadFile>(fileData_);
    auto opts = makeReaderOptions();
    opts.setFilePreloadThreshold(preloadThreshold);
    auto input = std::make_unique<velox::dwio::common::BufferedInput>(
        countingFile, *pool_);
    auto reader = ReaderBase::create(std::move(input), opts);
    EXPECT_EQ(reader->tablet().stripeCount(), kStripeCount);

    StripeStreams streams(reader);
    const auto numStreams = reader->nimbleSchema()->asRow().childrenCount() + 1;
    for (uint32_t stripe = 0; stripe < kStripeCount; ++stripe) {
      streams.setStripe(stripe);
      std::vector<std::unique_ptr<velox::dwio::common::SeekableInputStream>>
          enqueued;
      for (uint32_t streamId = 0; streamId < numStreams; ++streamId) {
        if (streams.hasStream(streamId)) {
          enqueued.push_back(streams.enqueue(streamId));
        }
      }
      streams.load();
    }
    return countingFile->numReads();
  };

  EXPECT_EQ(preadsReadingAllStripes(fileData_.size() + 1), 1);
  EXPECT_GE(preadsReadingAllStripes(0), kStripeCount);
}

INSTANTIATE_TEST_CASE_P(
    MetadataFormats,
    StripeStreamsMultiStripeTest,
    ::testing::Values(
        StripeGroup::EncodingLayout::kRaw,
        StripeGroup::EncodingLayout::kStreamMajor),
    [](const ::testing::TestParamInfo<StripeGroup::EncodingLayout>& info) {
      return info.param == StripeGroup::EncodingLayout::kRaw ? "raw"
                                                             : "streamMajor";
    });
