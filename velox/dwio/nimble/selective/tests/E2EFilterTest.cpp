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

#include <folly/executors/CPUThreadPoolExecutor.h>

#include <utility>

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/dwio/common/CachedBufferedInput.h"
#include "velox/dwio/common/tests/utils/E2EFilterTestBase.h"
#include "velox/dwio/nimble/common/ChunkedStream.h"
#include "velox/dwio/nimble/common/SchemaSerialization.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble {
namespace {

using namespace facebook::velox;
using index::ClusterIndexConfigBuilder;

using E2EFilterTestParams =
    std::tuple<bool, bool, bool, bool, bool, bool, bool>;

struct E2EFilterTestParam {
  bool indexEnabled;
  bool skipConstantFlatMapInMapStreams;
  bool enableChunkIndex;
  bool stringDecoderZeroCopy;
  bool pinMetadata;
  bool enableCache;
  bool nimblePreserveDictionaryEncoding;

  explicit E2EFilterTestParam(const E2EFilterTestParams& t)
      : indexEnabled{std::get<0>(t)},
        skipConstantFlatMapInMapStreams{std::get<1>(t)},
        enableChunkIndex{std::get<2>(t)},
        stringDecoderZeroCopy{std::get<3>(t)},
        pinMetadata{std::get<4>(t)},
        enableCache{std::get<5>(t)},
        nimblePreserveDictionaryEncoding{std::get<6>(t)} {}

  std::string debugString() const {
    return fmt::format(
        "index_{}_skipConstantInMap_{}_chunkIndex_{}_stringDecoderZeroCopy_{}_pinMetadata_{}_cache_{}_nimblePreserveDict_{}",
        indexEnabled,
        skipConstantFlatMapInMapStreams,
        enableChunkIndex,
        stringDecoderZeroCopy,
        pinMetadata,
        enableCache,
        nimblePreserveDictionaryEncoding);
  }
};

// Applies the multi-chunk writer config used by the shared write path so that
// standalone tests (which build their own writer) also produce multiple chunks
// per stream within a stripe, exercising the cross-chunk reader resume paths.
// Note: chunks are emitted at write() boundaries, so a single write() still
// yields one chunk — callers that write a single batch must split it.
void applyMultiChunkOptions(WriterOptions& options) {
  options.enableChunking = true;
  options.minStreamChunkRawSize = 0;
  options.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
}

// Writes `batch` as `numChunks` row-slices so the multi-chunk options above
// emit a separate chunk per slice (a single write() would yield one chunk).
void writeInChunks(
    Writer& writer,
    const RowVectorPtr& batch,
    int numChunks = 3) {
  const vector_size_t total = batch->size();
  const vector_size_t per =
      std::max<vector_size_t>(1, (total + numChunks - 1) / numChunks);
  for (vector_size_t offset = 0; offset < total; offset += per) {
    writer.write(
        std::dynamic_pointer_cast<RowVector>(
            batch->slice(offset, std::min(per, total - offset))));
  }
}

class E2EFilterTest
    : public dwio::common::E2EFilterTestBase,
      public ::testing::WithParamInterface<E2EFilterTestParams> {
 protected:
  static void SetUpTestCase() {
    E2EFilterTestBase::SetUpTestCase();
    registerSelectiveNimbleReaderFactory();
  }

  void SetUp() override {
    E2EFilterTestBase::SetUp();
    batchSize_ = 2003;
    batchCount_ = 5;
    testRowGroupSkip_ = false;
    dataIoStats_ = std::make_shared<io::IoStatistics>();
    metadataIoStats_ = std::make_shared<io::IoStatistics>();
    indexIoStats_ = std::make_shared<io::IoStatistics>();
    if (param().enableCache) {
      allocator_ = std::make_shared<memory::MallocAllocator>(
          memory::MemoryAllocator::Options{
              .capacity = 512 << 20, .reservationByteLimit = 0});
      cache_ = cache::AsyncDataCache::create(allocator_.get());
      scanTracker_ = std::make_shared<cache::ScanTracker>(
          "testTracker", nullptr, 256 << 10);
      ioExecutor_ = std::make_unique<folly::CPUThreadPoolExecutor>(10);
    }
  }

  void TearDown() override {
    ioExecutor_.reset();
    if (cache_ != nullptr) {
      cache_->shutdown();
    }
    cache_.reset();
    allocator_.reset();
  }

  E2EFilterTestParam param() const {
    return E2EFilterTestParam{GetParam()};
  }

  virtual bool indexEnabled() const {
    return param().indexEnabled;
  }

  bool skipConstantFlatMapInMapStreams() const {
    return param().skipConstantFlatMapInMapStreams;
  }

  void setUpRowReaderOptions(
      dwio::common::RowReaderOptions& opts,
      const std::shared_ptr<dwio::common::ScanSpec>& spec) override {
    opts.setScanSpec(spec);
    opts.setTimestampPrecision(TimestampPrecision::kNanoseconds);
    opts.setIndexEnabled(indexEnabled());
    opts.setStringDecoderZeroCopy(param().stringDecoderZeroCopy);
    opts.setNimblePreserveDictionaryEncoding(
        param().nimblePreserveDictionaryEncoding);
  }

  void testWithTypes(
      const std::string& columns,
      std::function<void()> customize,
      bool wrapInStruct,
      const std::vector<std::string>& filterable,
      int32_t numCombinations,
      bool withRecursiveNulls = true,
      std::optional<std::vector<std::pair<EncodingType, float>>>
          encodingFactors = std::nullopt) {
    // Select index columns if index is enabled.
    std::vector<std::string> indexColumns;
    if (indexEnabled()) {
      indexColumns = selectIndexColumns(columns, wrapInStruct);
    }

    // Set encoding factors for this test run.
    encodingFactors_ = encodingFactors;

    if (!withRecursiveNulls) {
      testScenario(
          columns,
          customize,
          wrapInStruct,
          filterable,
          numCombinations,
          /*withRecursiveNulls=*/false,
          indexColumns);
      encodingFactors_.reset();
      return;
    }

    testScenario(
        columns,
        customize,
        wrapInStruct,
        filterable,
        numCombinations,
        /*withRecursiveNulls=*/true,
        indexColumns);
    auto noNullCustomize = [&] {
      customize();
      makeNotNull();
    };
    testScenario(
        columns,
        noNullCustomize,
        wrapInStruct,
        filterable,
        numCombinations,
        /*withRecursiveNulls=*/true,
        indexColumns);

    // Clear encoding factors after test run.
    encodingFactors_.reset();
  }

  // Selects eligible index columns (integer types) from the schema.
  // Returns 1-2 randomly selected columns.
  std::vector<std::string> selectIndexColumns(
      const std::string& columns,
      bool wrapInStruct) {
    auto rowType =
        velox::test::DataSetBuilder::makeRowType(columns, wrapInStruct);
    std::vector<std::string> eligibleColumns;
    for (column_index_t i = 0; i < static_cast<column_index_t>(rowType->size());
         ++i) {
      const auto& childType = rowType->childAt(i);
      // Only top-level integer and varchar types are eligible for indexing.
      if (childType->kind() == TypeKind::TINYINT ||
          childType->kind() == TypeKind::SMALLINT ||
          childType->kind() == TypeKind::INTEGER ||
          childType->kind() == TypeKind::BIGINT ||
          childType->kind() == TypeKind::VARCHAR) {
        eligibleColumns.push_back(rowType->nameOf(i));
      }
    }

    if (eligibleColumns.empty()) {
      return {};
    }

    // Randomly select 1-2 columns.
    std::mt19937 gen(seed_);
    std::shuffle(eligibleColumns.begin(), eligibleColumns.end(), gen);
    const size_t numColumns =
        std::min(eligibleColumns.size(), static_cast<size_t>(1 + gen() % 2));
    eligibleColumns.resize(numColumns);
    return eligibleColumns;
  }

  void testRunLengthDictionaryWithTypes(
      const std::string& columns,
      std::function<void()> customize,
      bool wrapInStruct,
      const std::vector<std::string>& filterable,
      int32_t numCombinations,
      bool withRecursiveNulls,
      int64_t dictionaryGenSeed) {
    constexpr size_t kMaxRunLength = 10;
    if (withRecursiveNulls) {
      testRunLengthDictionaryScenario(
          columns,
          customize,
          wrapInStruct,
          filterable,
          numCombinations,
          kMaxRunLength,
          /*withRecursiveNulls=*/true,
          dictionaryGenSeed);
      auto noNullCustomize = [&] {
        customize();
        makeNotNull();
      };
      testRunLengthDictionaryScenario(
          columns,
          noNullCustomize,
          wrapInStruct,
          filterable,
          numCombinations,
          kMaxRunLength,
          /*withRecursiveNulls=*/true,
          dictionaryGenSeed);
    } else {
      testRunLengthDictionaryScenario(
          columns,
          customize,
          wrapInStruct,
          filterable,
          numCombinations,
          kMaxRunLength,
          /*withRecursiveNulls=*/false,
          dictionaryGenSeed);
    }
  }

  void writeToMemory(
      const TypePtr& type,
      const std::vector<RowVectorPtr>& batches,
      bool forRowGroupSkip,
      const std::vector<std::string>& indexColumns = {}) override {
    VELOX_CHECK(!forRowGroupSkip);
    // Clear cached IO data from previous writes to avoid stale cache entries.
    if (cache_ != nullptr) {
      cache_->clear();
    }
    rowType_ = asRowType(type);
    writeSchema_ = rowType_;
    WriterOptions options;
    options.skipConstantFlatMapInMapStreams = skipConstantFlatMapInMapStreams();
    options.enableChunking = true;
    options.enableChunkIndex = param().enableChunkIndex;
    // Ensure the chunk flush policy is honored for small test data by removing
    // the minimum chunk size threshold. Without this, small writes get merged
    // into a single chunk regardless of the flush policy.
    options.minStreamChunkRawSize = 0;

    // Force a specific encoding via layout tree, or use biased factors.
    if (forcedEncodingType_.has_value()) {
      auto encodingType = forcedEncodingType_.value();
      std::vector<EncodingLayoutTree> children;
      for (column_index_t col = 0;
           col < static_cast<column_index_t>(rowType_->size());
           ++col) {
        EncodingLayout layout{
            encodingType,
            {},
            CompressionType::Uncompressed,
            {EncodingLayout{
                 EncodingType::Trivial, {}, CompressionType::Uncompressed},
             EncodingLayout{
                 EncodingType::Trivial, {}, CompressionType::Uncompressed},
             EncodingLayout{
                 EncodingType::Trivial, {}, CompressionType::Uncompressed}}};
        children.push_back(
            EncodingLayoutTree{
                Kind::Scalar,
                {{0, std::move(layout)}},
                std::string(rowType_->nameOf(col))});
      }
      options.encodingLayoutTree.emplace(
          Kind::Row,
          std::unordered_map<
              EncodingLayoutTree::StreamIdentifier,
              EncodingLayout>{},
          "",
          std::move(children));
    } else if (encodingFactors_.has_value()) {
      // Use custom encoding factors if set.
      // Capture by value to avoid dangling reference.
      auto factors = encodingFactors_.value();
      options.encodingSelectionPolicyCreator = [factors](DataType dataType) {
        ManualEncodingSelectionPolicyFactory factory(factors);
        return factory.createPolicy(dataType);
      };
    }

    auto i = 0;
    options.flushPolicyFactory = [&] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[&](const StripeProgress&) { return (i++ % 3 == 2); },
          /*chunkLambda=*/
          [&](const StripeProgress&) { return (i++ % 3 == 2); });
    };
    if (!flatMapColumns_.empty()) {
      setUpFlatMapColumns();
      options.flatMapColumns = flatMapColumns_;
    }
    if (!deduplicatedArrayColumns_.empty()) {
      options.dictionaryArrayColumns = deduplicatedArrayColumns_;
    }
    if (!deduplicatedMapColumns_.empty()) {
      options.deduplicatedMapColumns = deduplicatedMapColumns_;
    }
    // Configure index if index columns are specified.
    if (!indexColumns.empty()) {
      options.clusterIndexConfig = ClusterIndexConfigBuilder{}
                                       .withKeyColumns(indexColumns)
                                       .withEnforceKeyOrder(true)
                                       .build();
    }
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    // Branch on whether we need forced dictionary encoding, since
    // EncodingLayoutTree is not move-assignable due to const members.
    if (useForcedDictionaryEncoding_) {
      // Need to set encodingLayoutTree at construction time.
      options.encodingLayoutTree.emplace(
          buildForcedDictionaryEncodingLayoutTree());
    }
    if (useForcedFsstEncoding_) {
      options.fsstCompressionTargetRatio = 10.0;
      options.encodingLayoutTree.emplace(buildForcedFsstEncodingLayoutTree());
    }
    Writer writer(
        writeSchema_, std::move(writeFile), *rootPool_, std::move(options));
    for (auto& batch : batches) {
      writer.write(batch);
    }
    writer.close();
  }

  std::unique_ptr<dwio::common::Reader> makeReader(
      const dwio::common::ReaderOptions& opts,
      std::unique_ptr<dwio::common::BufferedInput> input) final {
    auto readerOpts = opts;
    readerOpts.setLoadClusterIndex(true);
    readerOpts.setPinMetadata(param().pinMetadata);
    readerOpts.setCacheMetadata(param().enableCache);
    readerOpts.setIndexIoStats(indexIoStats_);
    if (param().enableCache) {
      auto readFile = std::make_shared<InMemoryReadFile>(sinkData_);
      auto& ids = fileIds();
      StringIdLease fileId(
          ids, fmt::format("testFile_{}", reinterpret_cast<uintptr_t>(this)));
      StringIdLease groupId(ids, "testGroup");
      io::ReaderOptions cacheReaderOpts(&readerOpts.memoryPool());
      cacheReaderOpts.setDataIoStats(dataIoStats_);
      cacheReaderOpts.setMetadataIoStats(metadataIoStats_);
      input = std::make_unique<dwio::common::CachedBufferedInput>(
          std::move(readFile),
          dwio::common::MetricsLog::voidLog(),
          std::move(fileId),
          cache_.get(),
          scanTracker_,
          std::move(groupId),
          dataIoStats_,
          nullptr,
          ioExecutor_.get(),
          cacheReaderOpts);
    }
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    return factory->createReader(std::move(input), readerOpts);
  }

  folly::F14FastMap<std::string, std::set<std::string>> flatMapColumns_;
  folly::F14FastSet<std::string> deduplicatedArrayColumns_;
  folly::F14FastSet<std::string> deduplicatedMapColumns_;
  // When true, forces dictionary encoding for string columns via
  // EncodingLayoutTree. This is useful for testing nested encoding scenarios
  // where dictionary is not at the top level (e.g., Nullable -> Dictionary).
  bool useForcedDictionaryEncoding_{false};

  bool useForcedFsstEncoding_{false};

  // Builds an encoding layout tree that forces dictionary encoding for string
  // columns. When data has nulls, the writer will wrap this with nullable
  // encoding, creating a Nullable -> Dictionary nesting.
  EncodingLayoutTree buildForcedDictionaryEncodingLayoutTree() {
    std::vector<EncodingLayoutTree> children;
    for (column_index_t i = 0;
         i < static_cast<column_index_t>(rowType_->size());
         ++i) {
      const auto& childType = rowType_->childAt(i);
      const auto& childName = rowType_->nameOf(i);

      if (childType->isVarchar() || childType->isVarbinary()) {
        // Dictionary has two sub-encodings: Alphabet (id=0) and Indices (id=1).
        // Specify nullopt for both to let the writer decide their encodings.
        // Without these children, the layout replay fails with
        // "Sub-encoding identifier out of range" when the writer creates
        // Dictionary's sub-encoding selection policies.
        children.push_back(
            EncodingLayoutTree{
                Kind::Scalar,
                {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                  EncodingLayout{
                      EncodingType::Dictionary,
                      {},
                      // Required by EncodingLayout constructor; does not
                      // override the encoding selection policy's choice.
                      CompressionType::Uncompressed,
                      {
                          // Alphabet (id=0): let writer decide
                          std::nullopt,
                          // Indices (id=1): let writer decide
                          std::nullopt,
                      }}}},
                std::string(childName)});
      } else {
        children.push_back(
            EncodingLayoutTree{Kind::Scalar, {}, std::string(childName)});
      }
    }
    return EncodingLayoutTree{Kind::Row, {}, "", std::move(children)};
  }

  EncodingLayoutTree buildForcedFsstEncodingLayoutTree() {
    std::vector<EncodingLayoutTree> children;
    for (column_index_t i = 0;
         i < static_cast<column_index_t>(rowType_->size());
         ++i) {
      const auto& childType = rowType_->childAt(i);
      const auto& childName = rowType_->nameOf(i);

      if (childType->isVarchar() || childType->isVarbinary()) {
        children.push_back(
            EncodingLayoutTree{
                Kind::Scalar,
                {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                  EncodingLayout{
                      EncodingType::Fsst,
                      {},
                      CompressionType::Uncompressed,
                      {EncodingLayout{
                          EncodingType::Trivial,
                          {},
                          CompressionType::Uncompressed}}}}},
                std::string(childName)});
      } else {
        children.push_back(
            EncodingLayoutTree{Kind::Scalar, {}, std::string(childName)});
      }
    }
    return EncodingLayoutTree{Kind::Row, {}, "", std::move(children)};
  }

  EncodingLayoutTree buildForcedMainlyConstantDictionaryEncodingLayoutTree() {
    std::vector<EncodingLayoutTree> children;
    for (column_index_t i = 0;
         i < static_cast<column_index_t>(rowType_->size());
         ++i) {
      const auto& childType = rowType_->childAt(i);
      const auto& childName = rowType_->nameOf(i);

      if (childType->isVarchar() || childType->isVarbinary()) {
        // CompressionType::Uncompressed is required by EncodingLayout
        // constructor; does not override the encoding selection policy.
        EncodingLayout dictLayout{
            EncodingType::Dictionary,
            {},
            CompressionType::Uncompressed,
            {std::nullopt, std::nullopt}};
        children.push_back(
            EncodingLayoutTree{
                Kind::Scalar,
                {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                  EncodingLayout{
                      EncodingType::MainlyConstant,
                      {},
                      CompressionType::Uncompressed,
                      {std::nullopt, std::move(dictLayout)}}}},
                std::string(childName)});
      } else {
        children.push_back(
            EncodingLayoutTree{Kind::Scalar, {}, std::string(childName)});
      }
    }
    return EncodingLayoutTree{Kind::Row, {}, "", std::move(children)};
  }

  /// Builds a layout tree that forces string columns into RLE→Dictionary
  /// encoding to exercise the RLE dictionary vector path in E2E tests.
  EncodingLayoutTree buildForcedRleDictionaryEncodingLayoutTree() {
    std::vector<EncodingLayoutTree> children;
    for (column_index_t i = 0;
         i < static_cast<column_index_t>(rowType_->size());
         ++i) {
      const auto& childType = rowType_->childAt(i);
      const auto& childName = rowType_->nameOf(i);

      if (childType->isVarchar() || childType->isVarbinary()) {
        EncodingLayout dictLayout{
            EncodingType::Dictionary,
            {},
            CompressionType::Uncompressed,
            {std::nullopt, std::nullopt}};
        children.push_back(
            EncodingLayoutTree{
                Kind::Scalar,
                {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                  EncodingLayout{
                      EncodingType::RLE,
                      {},
                      CompressionType::Uncompressed,
                      {std::nullopt, std::move(dictLayout)}}}},
                std::string(childName)});
      } else {
        children.push_back(
            EncodingLayoutTree{Kind::Scalar, {}, std::string(childName)});
      }
    }
    return EncodingLayoutTree{Kind::Row, {}, "", std::move(children)};
  }

  void verifyDictionaryVectorReturned(
      EncodingLayoutTree layoutTree,
      const std::vector<std::string>& stringData) {
    auto type = asRowType(rowType_);
    const auto kRowCount = static_cast<vector_size_t>(stringData.size());

    auto stringVector =
        velox::test::VectorMaker(leafPool_.get())
            .flatVector<velox::StringView>(kRowCount, [&](auto row) {
              return velox::StringView(stringData[row]);
            });
    auto longVector = velox::test::VectorMaker(leafPool_.get())
                          .flatVector<int64_t>(kRowCount, [](auto row) {
                            return static_cast<int64_t>(row);
                          });

    auto batch = std::make_shared<RowVector>(
        leafPool_.get(),
        type,
        nullptr,
        kRowCount,
        std::vector<VectorPtr>{stringVector, longVector});

    {
      auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
      WriterOptions writerOptions;
      writerOptions.encodingLayoutTree.emplace(std::move(layoutTree));
      applyMultiChunkOptions(writerOptions);
      Writer writer(
          type, std::move(writeFile), *rootPool_, std::move(writerOptions));
      writeInChunks(writer, batch);
      writer.close();
    }

    auto input = std::make_unique<velox::dwio::common::BufferedInput>(
        std::make_shared<velox::InMemoryReadFile>(sinkData_),
        *leafPool_,
        velox::dwio::common::MetricsLog::voidLog());

    auto dataIoStats = std::make_shared<velox::io::IoStatistics>();
    auto metadataIoStats = std::make_shared<velox::io::IoStatistics>();
    velox::dwio::common::ReaderOptions readerOpts(leafPool_.get());
    readerOpts.setDataIoStats(dataIoStats);
    readerOpts.setMetadataIoStats(metadataIoStats);
    auto reader = makeReader(readerOpts, std::move(input));
    auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
    scanSpec->addAllChildFields(*type);
    velox::dwio::common::RowReaderOptions rowReaderOptions;
    rowReaderOptions.setScanSpec(scanSpec);
    rowReaderOptions.setStringDecoderZeroCopy(param().stringDecoderZeroCopy);
    rowReaderOptions.setNimblePreserveDictionaryEncoding(
        param().stringDecoderZeroCopy);
    auto rowReader = reader->createRowReader(rowReaderOptions);

    auto result = BaseVector::create(type, 0, leafPool_.get());
    size_t totalRows = 0;
    size_t globalRow = 0;
    while (rowReader->next(1000, result)) {
      auto* rowResult = result->as<RowVector>();
      ASSERT_NE(rowResult, nullptr);
      totalRows += rowResult->size();

      auto stringChild = rowResult->childAt(0)->loadedVector();
      if (param().stringDecoderZeroCopy) {
        ASSERT_EQ(stringChild->encoding(), VectorEncoding::Simple::DICTIONARY)
            << "Expected DICTIONARY encoding for string column";
      }

      DecodedVector decoded(*stringChild);
      for (size_t i = 0; i < rowResult->size(); ++i) {
        ASSERT_EQ(decoded.valueAt<StringView>(i).str(), stringData[globalRow])
            << "Mismatch at row=" << globalRow;
        ++globalRow;
      }
    }

    EXPECT_EQ(totalRows, kRowCount);
  }

  // Optional encoding factors for ManualEncodingSelectionPolicyFactory.
  // When set, writeToMemory will use these factors instead of the default
  // encoding selection policy. Clear after test to avoid affecting other tests.
  std::optional<std::vector<std::pair<EncodingType, float>>> encodingFactors_;

  // Optional forced encoding type. When set, writeToMemory builds an
  // EncodingLayoutTree that forces every scalar column to use this encoding.
  // Takes precedence over encodingFactors_.
  std::optional<EncodingType> forcedEncodingType_;

  // Cache infrastructure (only initialized when enableCache is true).
  std::shared_ptr<memory::MallocAllocator> allocator_;
  std::shared_ptr<cache::AsyncDataCache> cache_;
  std::shared_ptr<io::IoStatistics> dataIoStats_;
  std::shared_ptr<io::IoStatistics> metadataIoStats_;
  std::shared_ptr<io::IoStatistics> indexIoStats_;
  std::shared_ptr<cache::ScanTracker> scanTracker_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> ioExecutor_;

  // Generates encoding factors that strongly favor a specific encoding type.
  // Encodings with lower read factors are preferred (factor represents CPU
  // cost). The favored encoding gets a low factor (0.1), while others get high
  // factors (100.0). Only includes encodings in the pool; omitting encodings
  // from the pool prevents them from being selected. Trivial encoding is always
  // included as a fallback.
  static std::vector<std::pair<EncodingType, float>> biasedEncodingFactors(
      EncodingType favored) {
    std::vector<std::pair<EncodingType, float>> factors;

    // Trivial is always needed as a fallback when other encodings don't apply.
    // Use a high factor to discourage it unless necessary.
    factors.emplace_back(EncodingType::Trivial, 100.0);

    // Add the favored encoding with a low factor (strongly preferred).
    if (favored != EncodingType::Trivial) {
      factors.emplace_back(favored, 0.1);
    } else {
      // If Trivial is favored, just use it with low factor.
      factors[0].second = 0.1;
    }

    // For certain encodings, we need to include additional encodings in the
    // pool to handle edge cases (e.g., constant values).
    switch (favored) {
      case EncodingType::FixedBitWidth:
      case EncodingType::RLE:
      case EncodingType::Varint:
        // These may fall back to Constant for constant data.
        factors.emplace_back(EncodingType::Constant, 100.0);
        break;
      case EncodingType::MainlyConstant:
        // MainlyConstant needs Constant for the common value.
        factors.emplace_back(EncodingType::Constant, 100.0);
        break;
      case EncodingType::Delta:
        // Delta uses child encodings for deltas, restatements, and
        // isRestatements that may fall back to Constant.
        factors.emplace_back(EncodingType::Constant, 100.0);
        break;
      case EncodingType::SubIntSplit:
        // SubIntSplit may fall back to Constant for fully-constant data.
        factors.emplace_back(EncodingType::Constant, 100.0);
        break;
      default:
        break;
    }

    return factors;
  }

  // Verifies that all scalar columns in the file written to sinkData_ use
  // the expected encoding type. Accounts for nullable wrapping.
  void verifyColumnEncodingsOnDisk(EncodingType expectedEncodingType) {
    auto readFile = std::make_shared<InMemoryReadFile>(sinkData_);
    auto& pool = *leafPool_;
    auto tablet = TabletReader::create(
        readFile, &pool, test::makeTestTabletOptions(&pool));
    auto section = tablet->loadOptionalSection(std::string(kSchemaSection));
    ASSERT_TRUE(section.has_value());
    auto schema = SchemaDeserializer::deserialize(section->content().data());

    for (uint32_t col = 0; col < schema->asRow().childrenCount(); ++col) {
      auto& childNode = schema->asRow().childAt(col)->asScalar();
      for (auto i = 0; i < tablet->stripeCount(); ++i) {
        auto stripeIdentifier = tablet->stripeIdentifier(i);
        std::vector<uint32_t> identifiers{
            childNode.scalarDescriptor().offset()};
        auto streams = tablet->load(stripeIdentifier, identifiers);

        InMemoryChunkedStream chunkedStream{pool, std::move(streams[0])};
        if (!chunkedStream.hasNext()) {
          continue;
        }
        auto capture = EncodingLayoutCapture::capture(
            chunkedStream.nextChunk(), Encoding::Options{});
        // The top-level encoding may be Nullable (wrapping the data encoding)
        // or the data encoding directly.
        if (capture.encodingType() == EncodingType::Nullable) {
          ASSERT_TRUE(capture.child(1).has_value())
              << "Column " << col << " stripe " << i;
          EXPECT_EQ(expectedEncodingType, capture.child(1)->encodingType())
              << "Column " << col << " stripe " << i;
        } else {
          EXPECT_EQ(expectedEncodingType, capture.encodingType())
              << "Column " << col << " stripe " << i;
        }
      }
    }
  }

  void verifyColumnEncodingOnDisk(
      const std::string& columnName,
      EncodingType expectedEncodingType) {
    auto readFile = std::make_shared<InMemoryReadFile>(sinkData_);
    auto& pool = *leafPool_;
    auto tablet = TabletReader::create(
        readFile, &pool, test::makeTestTabletOptions(&pool));
    auto section = tablet->loadOptionalSection(std::string(kSchemaSection));
    ASSERT_TRUE(section.has_value());
    auto schema = SchemaDeserializer::deserialize(section->content().data());

    std::optional<column_index_t> column;
    for (column_index_t col = 0;
         col < static_cast<column_index_t>(rowType_->size());
         ++col) {
      if (rowType_->nameOf(col) == columnName) {
        column = col;
        break;
      }
    }
    ASSERT_TRUE(column.has_value()) << columnName;

    auto& childNode = schema->asRow().childAt(*column)->asScalar();
    for (auto i = 0; i < tablet->stripeCount(); ++i) {
      auto stripeIdentifier = tablet->stripeIdentifier(i);
      std::vector<uint32_t> identifiers{childNode.scalarDescriptor().offset()};
      auto streams = tablet->load(stripeIdentifier, identifiers);

      InMemoryChunkedStream chunkedStream{pool, std::move(streams[0])};
      if (!chunkedStream.hasNext()) {
        continue;
      }
      auto capture = EncodingLayoutCapture::capture(
          chunkedStream.nextChunk(), Encoding::Options{});
      if (capture.encodingType() == EncodingType::Nullable) {
        ASSERT_TRUE(capture.child(1).has_value())
            << "Column " << columnName << " stripe " << i;
        EXPECT_EQ(expectedEncodingType, capture.child(1)->encodingType())
            << "Column " << columnName << " stripe " << i;
      } else {
        EXPECT_EQ(expectedEncodingType, capture.encodingType())
            << "Column " << columnName << " stripe " << i;
      }
    }
  }

 private:
  void setUpFlatMapColumns() {
    auto rowTypeWithId = dwio::common::TypeWithId::create(rowType_);
    auto names = rowType_->names();
    auto types = rowType_->children();
    for (int i = 0; i < rowType_->size(); ++i) {
      if (!flatMapColumns_.contains(rowType_->nameOf(i))) {
        continue;
      }
      auto& type = rowType_->childAt(i);
      if (!type->isRow()) {
        continue;
      }
      types[i] = MAP(VARCHAR(), type->childAt(0));
    }
    writeSchema_ = ROW(std::move(names), std::move(types));
  }

  RowTypePtr writeSchema_;
};

TEST_P(E2EFilterTest, byte) {
  testWithTypes(
      "tiny_val:tinyint,"
      "bool_val:boolean,"
      "long_val:bigint,"
      "tiny_null:bigint",
      [&]() { makeAllNulls("tiny_null"); },
      true,
      {"tiny_val", "bool_val", "tiny_null"},
      20);
}

TEST_P(E2EFilterTest, integer) {
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint,"
      "long_null:bigint",
      [&] { makeAllNulls("long_null"); },
      true,
      {"short_val", "int_val", "long_val"},
      20);
}

// Biased variant that forces FixedBitWidth encoding selection.
TEST_P(E2EFilterTest, integerBiasedFixedBitWidth) {
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint",
      [&]() {
        makeIntDistribution<int16_t>(
            "short_val",
            /*min=*/10,
            /*max=*/1000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
        makeIntDistribution<int32_t>(
            "int_val",
            /*min=*/100,
            /*max=*/100000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
        makeIntDistribution<int64_t>(
            "long_val",
            /*min=*/1000,
            /*max=*/1000000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
      },
      /*wrapInStruct=*/false,
      {"short_val", "int_val", "long_val"},
      /*numCombinations=*/20,
      /*withRecursiveNulls=*/true,
      biasedEncodingFactors(EncodingType::FixedBitWidth));
}

// Biased variant that forces BlockBitPacking encoding selection.
TEST_P(E2EFilterTest, integerBiasedBlockBitPacking) {
  testWithTypes(
      "tiny_val:tinyint,"
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint",
      [&]() {
        makeIntDistribution<int8_t>(
            "tiny_val",
            /*min=*/1,
            /*max=*/15,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
        makeIntDistribution<int16_t>(
            "short_val",
            /*min=*/10,
            /*max=*/1000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
        makeIntDistribution<int32_t>(
            "int_val",
            /*min=*/100,
            /*max=*/100000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
        makeIntDistribution<int64_t>(
            "long_val",
            /*min=*/1000,
            /*max=*/1000000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
      },
      /*wrapInStruct=*/false,
      {"tiny_val", "short_val", "int_val", "long_val"},
      /*numCombinations=*/20,
      /*withRecursiveNulls=*/true,
      biasedEncodingFactors(EncodingType::BlockBitPacking));
}

// Biased variant that forces Trivial encoding selection.
TEST_P(E2EFilterTest, integerBiasedTrivial) {
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint",
      [&]() {
        makeIntDistribution<int16_t>(
            "short_val",
            /*min=*/10,
            /*max=*/1000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
        makeIntDistribution<int32_t>(
            "int_val",
            /*min=*/100,
            /*max=*/100000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
        makeIntDistribution<int64_t>(
            "long_val",
            /*min=*/1000,
            /*max=*/1000000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
      },
      /*wrapInStruct=*/false,
      {"short_val", "int_val", "long_val"},
      /*numCombinations=*/20,
      /*withRecursiveNulls=*/true,
      biasedEncodingFactors(EncodingType::Trivial));
}

TEST_P(E2EFilterTest, integerLowCardinality) {
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint",
      [&]() {
        makeIntDistribution<int64_t>(
            "long_val",
            10, // min
            100, // max
            22, // repeats
            19, // rareFrequency
            -9999, // rareMin
            10000000000, // rareMax
            true); // keepNulls
        makeIntDistribution<int32_t>(
            "int_val",
            10, // min
            100, // max
            22, // repeats
            19, // rareFrequency
            -9999, // rareMin
            100000000, // rareMax
            false); // keepNulls
        makeIntDistribution<int16_t>(
            "short_val",
            10, // min
            100, // max
            22, // repeats
            19, // rareFrequency
            -999, // rareMin
            30000, // rareMax
            true); // keepNulls
      },
      true,
      {"short_val", "int_val", "long_val"},
      20);
}

TEST_P(E2EFilterTest, integerRle) {
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint",
      [&] {
        makeIntRle<int16_t>("short_val");
        makeIntRle<int32_t>("int_val");
        makeIntRle<int64_t>("long_val");
        makeIntRle<int16_t>("struct_val.short_val");
        makeIntRle<int32_t>("struct_val.int_val");
        makeIntRle<int64_t>("struct_val.long_val");
      },
      true,
      {"short_val", "int_val", "long_val"},
      20);
}

// Biased variant that forces RLE encoding selection.
TEST_P(E2EFilterTest, integerRleBiased) {
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint",
      [&]() {
        makeIntRle<int16_t>("short_val");
        makeIntRle<int32_t>("int_val");
        makeIntRle<int64_t>("long_val");
        makeIntRle<int16_t>("struct_val.short_val");
        makeIntRle<int32_t>("struct_val.int_val");
        makeIntRle<int64_t>("struct_val.long_val");
      },
      /*wrapInStruct=*/true,
      {"short_val", "int_val", "long_val"},
      /*numCombinations=*/20,
      /*withRecursiveNulls=*/true,
      biasedEncodingFactors(EncodingType::RLE));
}

TEST_P(E2EFilterTest, integerRleCustomSeed4049269257) {
  if (common::testutil::useRandomSeed()) {
    return;
  }
  seed_ = 4049269257;
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint",
      [&] {
        makeIntRle<int16_t>("short_val");
        makeIntRle<int32_t>("int_val");
        makeIntRle<int64_t>("long_val");
        makeIntRle<int16_t>("struct_val.short_val");
        makeIntRle<int32_t>("struct_val.int_val");
        makeIntRle<int64_t>("struct_val.long_val");
      },
      true,
      {"short_val", "int_val", "long_val"},
      20);
}

TEST_P(E2EFilterTest, integerRleCustomSeed583694982) {
  if (common::testutil::useRandomSeed()) {
    return;
  }
  seed_ = 583694982;
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint",
      [&] {
        makeIntRle<int16_t>("short_val");
        makeIntRle<int32_t>("int_val");
        makeIntRle<int64_t>("long_val");
        makeIntRle<int16_t>("struct_val.short_val");
        makeIntRle<int32_t>("struct_val.int_val");
        makeIntRle<int64_t>("struct_val.long_val");
      },
      true,
      {"short_val", "int_val", "long_val"},
      20);
}

TEST_P(E2EFilterTest, integerRleCustomSeed2518626933) {
  if (common::testutil::useRandomSeed()) {
    return;
  }
  seed_ = 2518626933;
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint",
      [&] {
        makeIntRle<int16_t>("short_val");
        makeIntRle<int32_t>("int_val");
        makeIntRle<int64_t>("long_val");
        makeIntRle<int16_t>("struct_val.short_val");
        makeIntRle<int32_t>("struct_val.int_val");
        makeIntRle<int64_t>("struct_val.long_val");
      },
      true,
      {"short_val", "int_val", "long_val"},
      20);
}

TEST_P(E2EFilterTest, mainlyConstant) {
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint",
      [&] {
        makeIntMainlyConstant<int16_t>("short_val");
        makeIntMainlyConstant<int32_t>("int_val");
        makeIntMainlyConstant<int64_t>("long_val");
        makeIntMainlyConstant<int16_t>("struct_val.short_val");
        makeIntMainlyConstant<int32_t>("struct_val.int_val");
        makeIntMainlyConstant<int64_t>("struct_val.long_val");
      },
      true,
      {"short_val", "int_val", "long_val"},
      20);
}

// Biased variant that forces MainlyConstant encoding selection.
TEST_P(E2EFilterTest, mainlyConstantBiased) {
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint",
      [&]() {
        makeIntMainlyConstant<int16_t>("short_val");
        makeIntMainlyConstant<int32_t>("int_val");
        makeIntMainlyConstant<int64_t>("long_val");
      },
      /*wrapInStruct=*/false,
      {"short_val", "int_val", "long_val"},
      /*numCombinations=*/20,
      /*withRecursiveNulls=*/true,
      biasedEncodingFactors(EncodingType::MainlyConstant));
}

// Forces Delta encoding via encoding layout tree and verifies it on disk.
TEST_P(E2EFilterTest, integerBiasedDelta) {
  forcedEncodingType_ = EncodingType::Delta;
  testWithTypes(
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint",
      [&]() {
        makeIntDistribution<int16_t>(
            "short_val",
            /*min=*/10,
            /*max=*/1000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
        makeIntDistribution<int32_t>(
            "int_val",
            /*min=*/100,
            /*max=*/100000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
        makeIntDistribution<int64_t>(
            "long_val",
            /*min=*/1000,
            /*max=*/1000000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
      },
      /*wrapInStruct=*/false,
      {"short_val", "int_val", "long_val"},
      /*numCombinations=*/20,
      /*withRecursiveNulls=*/true);
  forcedEncodingType_.reset();
  // Verify the on-disk encoding is Delta (sinkData_ holds the last written
  // file from testWithTypes).
  verifyColumnEncodingsOnDisk(EncodingType::Delta);
}

TEST_P(E2EFilterTest, integerForcedHuffman) {
  forcedEncodingType_ = EncodingType::Huffman;
  testWithTypes(
      "tiny_val:tinyint,"
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint",
      [&]() {
        makeIntDistribution<int8_t>("tiny_val", -8, 7, 20, 0, 0, 0, false);
        makeIntDistribution<int16_t>("short_val", -32, 31, 20, 0, 0, 0, false);
        makeIntDistribution<int32_t>("int_val", -128, 127, 20, 0, 0, 0, false);
        makeIntDistribution<int64_t>("long_val", -512, 511, 20, 0, 0, 0, false);
      },
      /*wrapInStruct=*/false,
      {"tiny_val", "short_val", "int_val", "long_val"},
      /*numCombinations=*/20,
      /*withRecursiveNulls=*/true);
  forcedEncodingType_.reset();
  verifyColumnEncodingsOnDisk(EncodingType::Huffman);
}

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
// Biased variant that forces SubIntSplit encoding selection.
// SubIntSplit only applies to 32- and 64-bit types; smallint is excluded.
TEST_P(E2EFilterTest, integerBiasedSubIntSplit) {
  testWithTypes(
      "int_val:int,"
      "long_val:bigint",
      [&]() {
        makeIntDistribution<int32_t>(
            "int_val",
            /*min=*/100,
            /*max=*/100000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
        makeIntDistribution<int64_t>(
            "long_val",
            /*min=*/1000,
            /*max=*/1000000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
      },
      /*wrapInStruct=*/true,
      {"int_val", "long_val"},
      /*numCombinations=*/20,
      /*withRecursiveNulls=*/true,
      biasedEncodingFactors(EncodingType::SubIntSplit));
}

// Tests data with bit-structured distributions where SubIntSplit naturally
// wins: values concentrated in the low-bit range of a wider type, so the
// upper bit sections collapse to Constant encoding.
TEST_P(E2EFilterTest, integerSubIntSplitBitStructured) {
  testWithTypes(
      "int_val:int,"
      "long_val:bigint",
      [&]() {
        makeIntDistribution<int32_t>(
            "int_val",
            /*min=*/0,
            /*max=*/65535, // only 16 low bits vary; upper 16 are zero
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
        makeIntDistribution<int64_t>(
            "long_val",
            /*min=*/0,
            /*max=*/4294967295, // only 32 low bits vary; upper 32 are zero
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
      },
      /*wrapInStruct=*/false,
      {"int_val", "long_val"},
      /*numCombinations=*/20,
      /*withRecursiveNulls=*/true);
}

// Biased variant that forces SubIntSplit encoding for float/double columns.
// SubIntSplit encodes float/double via their uint32/uint64 IEEE 754 bit
// patterns; quantized data produces sections of differing entropy.
TEST_P(E2EFilterTest, floatBiasedSubIntSplit) {
  testWithTypes(
      "float_val:float,"
      "double_val:double",
      [&]() {
        makeQuantizedFloat<float>("float_val", 100, true);
        makeQuantizedFloat<double>("double_val", 100, true);
      },
      /*wrapInStruct=*/false,
      {"float_val", "double_val"},
      /*numCombinations=*/20,
      /*withRecursiveNulls=*/true,
      biasedEncodingFactors(EncodingType::SubIntSplit));
}
#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
// Forces SubIntSplit via EncodingLayoutTree and verifies the on-disk encoding
// matches. Schema uses only 32/64-bit types (SubIntSplit requires
// sizeof(T)>=4).
TEST_P(E2EFilterTest, integerForcedSubIntSplit) {
  forcedEncodingType_ = EncodingType::SubIntSplit;
  testWithTypes(
      "int_val:int,"
      "long_val:bigint",
      [&]() {
        makeIntDistribution<int32_t>(
            "int_val",
            /*min=*/100,
            /*max=*/100000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
        makeIntDistribution<int64_t>(
            "long_val",
            /*min=*/1000,
            /*max=*/1000000,
            /*repeats=*/1,
            /*rareFrequency=*/0,
            /*rareMin=*/0,
            /*rareMax=*/0,
            /*keepNulls=*/false);
      },
      /*wrapInStruct=*/false,
      {"int_val", "long_val"},
      /*numCombinations=*/20,
      /*withRecursiveNulls=*/true);
  forcedEncodingType_.reset();
  verifyColumnEncodingsOnDisk(EncodingType::SubIntSplit);
}
#endif

TEST_P(E2EFilterTest, float) {
  testWithTypes(
      "float_val:float,"
      "double_val:double,"
      "long_val:bigint,"
      "float_null:float",
      [&]() { makeAllNulls("float_null"); },
      true,
      {"float_val", "double_val", "float_null"},
      20);
}

TEST_P(E2EFilterTest, string) {
  testWithTypes(
      "string_val:string,"
      "string_val_2:string",
      [&]() {
        makeStringUnique("string_val");
        makeStringUnique("string_val_2");
      },
      true,
      {"string_val", "string_val_2"},
      20);
}

TEST_P(E2EFilterTest, stringDictionary) {
  testWithTypes(
      "string_val:string,"
      "string_val_2:string",
      [&]() {
        makeStringDistribution("string_val", 100, true, false);
        makeStringDistribution("string_val_2", 170, false, true);
      },
      true,
      {"string_val", "string_val_2"},
      20);
}

TEST_P(E2EFilterTest, fsstStringFilterPushdown) {
  if (!param().stringDecoderZeroCopy) {
    GTEST_SKIP() << "FSST string decoding requires zero-copy string decoder.";
  }

  useForcedFsstEncoding_ = true;
  testWithTypes(
      "string_val:string,"
      "string_val_2:string,"
      "long_val:bigint",
      [&]() {
        makeStringDistribution("string_val", 50, true, true);
        makeStringDistribution("string_val_2", 80, true, false);
      },
      /*wrapInStruct=*/false,
      {"string_val"},
      /*numCombinations=*/5,
      /*withRecursiveNulls=*/false);
  verifyColumnEncodingOnDisk("string_val", EncodingType::Fsst);
  useForcedFsstEncoding_ = false;
}

TEST_P(E2EFilterTest, listAndMapNoRecursiveNulls) {
  int numCombinations = 20;
#if !defined(NDEBUG) || defined(TSAN_BUILD)
  // The test is running slow under dev/debug and TSAN build; reduce the number
  // of combinations to avoid timeout.
  numCombinations = 2;
#endif
  testWithTypes(
      "long_val:bigint,"
      "long_val_2:bigint,"
      "int_val:int,"
      "simple_array_val:array<int>,"
      "array_val:array<struct<array_member: array<int>, float_val:float,long_val:bigint, string_val:string>>"
      "map_val:map<bigint,struct<nested_map: map<int, int>>>",
      [&]() {},
      true,
      {"long_val",
       "long_val_2",
       "int_val",
       "simple_array_val",
       "array_val",
       "map_val"},
      numCombinations,
      /*withRecursiveNulls=*/false);
}

TEST_P(E2EFilterTest, listAndMap) {
  int numCombinations = 20;
#if !defined(NDEBUG) || defined(TSAN_BUILD)
  // The test is running slow under dev/debug and TSAN build; reduce the number
  // of combinations to avoid timeout.
  numCombinations = 2;
#endif
  testWithTypes(
      "long_val:bigint,"
      "long_val_2:bigint,"
      "int_val:int,"
      "simple_array_val:array<int>,"
      "array_val:array<struct<array_member: array<int>, float_val:float,long_val:bigint, string_val:string>>"
      "map_val:map<bigint,struct<nested_map: map<int, int>>>",
      [&]() {},
      true,
      {"long_val",
       "long_val_2",
       "int_val",
       "simple_array_val",
       "array_val",
       "map_val"},
      numCombinations,
      /*withRecursiveNulls=*/true);
}

TEST_P(E2EFilterTest, listAndMapSmallReads) {
  int numCombinations = 20;
#if !defined(NDEBUG) || defined(TSAN_BUILD)
  // The test is running slow under dev/debug and TSAN build; reduce the number
  // of combinations to avoid timeout.
  numCombinations = 2;
#endif
  setReadSize(50);

  testWithTypes(
      "long_val:bigint,"
      "long_val_2:bigint,"
      "int_val:int,"
      "simple_array_val:array<int>,"
      "array_val:array<struct<array_member: array<int>, float_val:float,long_val:bigint, string_val:string>>"
      "map_val:map<bigint,struct<nested_map: map<int, int>>>",
      [&]() {},
      true,
      {"long_val",
       "long_val_2",
       "int_val",
       "simple_array_val",
       "array_val",
       "map_val"},
      numCombinations,
      /*withRecursiveNulls=*/true);
}

TEST_P(E2EFilterTest, deduplicatedArrayNoRecursiveNulls) {
  int numCombinations = 20;
#if !defined(NDEBUG) || defined(TSAN_BUILD)
  // The test is running slow under dev/debug and TSAN build; reduce the number
  // of combinations to avoid timeout.
  numCombinations = 2;
#endif
  deduplicatedArrayColumns_ = {"array_val"};
  testWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "array_val:array<int>",
      [&]() {},
      true,
      {"long_val", "int_val", "array_val"},
      numCombinations,
      /*withRecursiveNulls=*/false);

  std::mt19937 gen{seed_};
  testRunLengthDictionaryWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "array_val:array<int>",
      [&]() {},
      true,
      {"long_val", "int_val", "array_val"},
      numCombinations,
      /*withRecursiveNulls=*/false,
      folly::Random::rand64(gen));
}

TEST_P(E2EFilterTest, deduplicatedArray) {
  int numCombinations = 20;
#if !defined(NDEBUG) || defined(TSAN_BUILD)
  // The test is running slow under dev/debug and TSAN build; reduce the number
  // of combinations to avoid timeout.
  numCombinations = 2;
#endif
  deduplicatedArrayColumns_ = {"array_val"};
  testWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "array_val:array<int>",
      [&]() {},
      true,
      {"long_val", "int_val", "array_val"},
      numCombinations,
      /*withRecursiveNulls=*/true);

  std::mt19937 gen{seed_};
  testRunLengthDictionaryWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "array_val:array<int>",
      [&]() {},
      true,
      {"long_val", "int_val", "array_val"},
      numCombinations,
      /*withRecursiveNulls=*/true,
      folly::Random::rand64(gen));
}

// Pruning is only generated in struct_val.array_val in the normal case if
// filter is present on top level array_val, but struct_val.array_val is not
// deduplicated, so pruning is untested in normal case.
TEST_P(E2EFilterTest, deduplicatedArraySubfieldPruning) {
  int numCombinations = 20;
#if !defined(NDEBUG) || defined(TSAN_BUILD)
  // The test is running slow under dev/debug and TSAN build; reduce the number
  // of combinations to avoid timeout.
  numCombinations = 2;
#endif
  deduplicatedArrayColumns_ = {"array_val"};
  testWithTypes(
      "long_val:bigint,"
      "array_val:array<int>",
      [&]() {},
      false,
      {"long_val"},
      numCombinations,
      /*withRecursiveNulls=*/true);
  std::mt19937 gen{seed_};
  testRunLengthDictionaryWithTypes(
      "long_val:bigint,"
      "array_val:array<int>",
      [&]() {},
      false,
      {"long_val"},
      numCombinations,
      /*withRecursiveNulls=*/true,
      folly::Random::rand64(gen));
}

TEST_P(E2EFilterTest, deduplicatedArraySmallReads) {
  int numCombinations = 20;
#if !defined(NDEBUG) || defined(TSAN_BUILD)
  // The test is running slow under dev/debug and TSAN build; reduce the number
  // of combinations to avoid timeout.
  numCombinations = 2;
#endif
  setReadSize(50);

  deduplicatedArrayColumns_ = {"array_val"};
  testWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "array_val:array<int>",
      [&]() {},
      true,
      {"long_val", "int_val", "array_val"},
      numCombinations,
      /*withRecursiveNulls=*/true);

  std::mt19937 gen{seed_};
  testRunLengthDictionaryWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "array_val:array<int>",
      [&]() {},
      true,
      {"long_val", "int_val", "array_val"},
      numCombinations,
      /*withRecursiveNulls=*/true,
      folly::Random::rand64(gen));
}

// Small skips that don't reach the next dictionary run.
TEST_P(E2EFilterTest, deduplicatedArrayCustomSeedConnectedSkips) {
  if (common::testutil::useRandomSeed()) {
    return;
  }
  seed_ = 80108952;

  int numCombinations = 20;
#if !defined(NDEBUG) || defined(TSAN_BUILD)
  // The test is running slow under dev/debug and TSAN build; reduce the number
  // of combinations to avoid timeout.
  numCombinations = 2;
#endif
  deduplicatedArrayColumns_ = {"array_val"};

  testWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "array_val:array<int>",
      [&]() {},
      true,
      {"long_val", "int_val", "array_val"},
      numCombinations,
      /*withRecursiveNulls=*/true);

  testRunLengthDictionaryWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "array_val:array<int>",
      [&]() {},
      true,
      {"long_val", "int_val", "array_val"},
      numCombinations,
      /*withRecursiveNulls=*/true,
      /*dictionaryGenSeed=*/1729375660);
}

// Small reads that don't reach the next dictionary run and uses the cached
// last run only.
TEST_P(E2EFilterTest, deduplicatedArrayCustomSeedCachedRunOnly) {
  if (common::testutil::useRandomSeed()) {
    return;
  }
  seed_ = 41071487;

  int numCombinations = 20;
#if !defined(NDEBUG) || defined(TSAN_BUILD)
  // The test is running slow under dev/debug and TSAN build; reduce the number
  // of combinations to avoid timeout.
  numCombinations = 2;
#endif
  deduplicatedArrayColumns_ = {"array_val"};

  testWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "array_val:array<int>",
      [&]() {},
      true,
      {"long_val", "int_val", "array_val"},
      numCombinations,
      /*withRecursiveNulls=*/true);

  testRunLengthDictionaryWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "array_val:array<int>",
      [&]() {},
      true,
      {"long_val", "int_val", "array_val"},
      numCombinations,
      /*withRecursiveNulls=*/true,
      /*dictionaryGenSeed=*/1729403307);
}

TEST_P(E2EFilterTest, deduplicatedMapNoRecursiveNulls) {
  int numCombinations = 20;
#if !defined(NDEBUG) || defined(TSAN_BUILD)
  // The test is running slow under dev/debug and TSAN build; reduce the number
  // of combinations to avoid timeout.
  numCombinations = 2;
#endif
  deduplicatedMapColumns_ = {"map_val"};
  testWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "map_val:map<int, int>",
      [&]() {},
      true,
      {"long_val", "int_val", "map_val"},
      numCombinations,
      /*withRecursiveNulls=*/false);

  std::mt19937 gen{seed_};
  testRunLengthDictionaryWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "map_val:map<int, int>",
      [&]() {},
      true,
      {"long_val", "int_val", "map_val"},
      numCombinations,
      /*withRecursiveNulls=*/false,
      folly::Random::rand64(gen));
}

TEST_P(E2EFilterTest, deduplicatedMap) {
  int numCombinations = 20;
#if !defined(NDEBUG) || defined(TSAN_BUILD)
  // The test is running slow under dev/debug and TSAN build; reduce the number
  // of combinations to avoid timeout.
  numCombinations = 2;
#endif
  deduplicatedMapColumns_ = {"map_val"};
  testWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "map_val:map<int, int>",
      [&]() {},
      true,
      {"long_val", "int_val", "map_val"},
      numCombinations,
      /*withRecursiveNulls=*/true);

  std::mt19937 gen{seed_};
  testRunLengthDictionaryWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "map_val:map<int, int>",
      [&]() {},
      true,
      {"long_val", "int_val", "map_val"},
      numCombinations,
      /*withRecursiveNulls=*/true,
      folly::Random::rand64(gen));
}

// Regression for the deduplicated-map seekTo desync. Reproducing it requires
// the run-length-dictionary path: wrapping the batch so consecutive rows are
// true duplicates produces long same-offset dedup runs. Under an IS NULL /
// IS NOT NULL filter on map_val (-> readsNullsOnly), seekTo() uses skipNulls(),
// which advances the nulls stream but freezes the deduplicated lengths decoder;
// read over a long same-offset run, the stale length mismatches the real one,
// tripping "None empty shared prefix is not allowed" in
// prepareDeduplicatedStates. (Plain testWithTypes generates runs too short to
// expose this -- hence the run-length-dictionary wrapping here.)
TEST_P(E2EFilterTest, deduplicatedMapRunLengthDictSeekToDesync) {
  if (common::testutil::useRandomSeed()) {
    return;
  }
  seed_ = 2237282808;

  deduplicatedMapColumns_ = {"map_val"};
  // The base data comes from a persistent RNG that advances per makeDataset()
  // call, so the testWithTypes() calls below must run first to reach the same
  // data state that the run-length-dictionary path then desyncs on. Keep this
  // sequence identical to the deduplicatedMap fuzz test.
  testWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "map_val:map<int, int>",
      [&]() {},
      true,
      {"long_val", "int_val", "map_val"},
      20,
      /*withRecursiveNulls=*/true);
  std::mt19937 gen{seed_};
  testRunLengthDictionaryWithTypes(
      "long_val:bigint,"
      "int_val:int,"
      "map_val:map<int, int>",
      [&]() {},
      true,
      {"long_val", "int_val", "map_val"},
      20,
      /*withRecursiveNulls=*/true,
      /*dictionaryGenSeed=*/folly::Random::rand64(gen));
}

// Pruning is only generated in struct_val.map_val in the normal case if filter
// is present on top level map_val, but struct_val.map_val is not deduplicated,
// so pruning is untested in normal case.
TEST_P(E2EFilterTest, deduplicatedMapSubfieldPruning) {
  int numCombinations = 20;
#if !defined(NDEBUG) || defined(TSAN_BUILD)
  // The test is running slow under dev/debug and TSAN build; reduce the number
  // of combinations to avoid timeout.
  numCombinations = 2;
#endif
  deduplicatedMapColumns_ = {"map_val"};
  testWithTypes(
      "long_val:bigint,"
      "map_val:map<int, int>",
      [&]() {},
      false,
      {"long_val"},
      numCombinations,
      /*withRecursiveNulls=*/true);
  std::mt19937 gen{seed_};
  testRunLengthDictionaryWithTypes(
      "long_val:bigint,"
      "map_val:map<int, int>",
      [&]() {},
      false,
      {"long_val"},
      numCombinations,
      /*withRecursiveNulls=*/true,
      folly::Random::rand64(gen));
}

TEST_P(E2EFilterTest, nullCompactRanges) {
  // Makes a dataset with nulls at the beginning. Tries different
  // filter combinations on progressively larger batches. tests for a
  // bug in null compaction where null bits past end of nulls buffer
  // were compacted while there actually were no nulls.
  readSizes_ = {10, 100, 1000, 2500, 2500, 2500};
  testWithTypes(
      "tiny_val:tinyint,"
      "bool_val:boolean,"
      "long_val:bigint,"
      "tiny_null:bigint",
      [&]() { makeNotNull(500); },
      true,
      {"tiny_val", "bool_val", "long_val", "tiny_null"},
      20);
}

TEST_P(E2EFilterTest, lazyStruct) {
  testWithTypes(
      "long_val:bigint,"
      "outer_struct: struct<nested1:bigint, "
      "inner_struct: struct<nested2: bigint>>",
      [&]() {},
      true,
      {"long_val"},
      10);
}

TEST_P(E2EFilterTest, filterStruct) {
#ifdef TSAN_BUILD
  // The test is running slow under TSAN; reduce the number of combinations to
  // avoid timeout.
  constexpr int kNumCombinations = 10;
#else
  constexpr int kNumCombinations = 40;
#endif
  // The data has a struct member with one second level struct
  // column. Both structs have a column that gets filtered 'nestedxxx'
  // and one that does not 'dataxxx'.
  testWithTypes(
      "long_val:bigint,"
      "outer_struct: struct<nested1:bigint, "
      "  data1: string, "
      "  inner_struct: struct<nested2: bigint, data2: smallint>>",
      [&]() {},
      true,
      {"long_val",
       "outer_struct.inner_struct",
       "outer_struct.nested1",
       "outer_struct.inner_struct.nested2"},
      kNumCombinations);
}

// Enable once the writer support struct as flat map.
TEST_P(E2EFilterTest, DISABLED_flatMapAsStruct) {
  constexpr auto kColumns =
      "long_val:bigint,"
      "long_vals:struct<v1:bigint,v2:bigint,v3:bigint>,"
      "struct_vals:struct<nested1:struct<v1:bigint, v2:float>,nested2:struct<v1:bigint, v2:float>>";
  flatMapColumns_ = {{"long_vals", {}}, {"struct_vals", {}}};
  testWithTypes(kColumns, [] {}, false, {"long_val"}, 10);
}

TEST_P(E2EFilterTest, flatMapScalar) {
  constexpr auto kColumns =
      "long_val:bigint,"
      "long_vals:map<tinyint,bigint>,"
      "string_vals:map<string,string>";
  flatMapColumns_ = {{"long_vals", {}}, {"string_vals", {}}};
  auto customize = [this] {
    dataSetBuilder_->makeUniformMapKeys(common::Subfield("string_vals"));
    dataSetBuilder_->makeMapStringValues(common::Subfield("string_vals"));
  };
  int numCombinations = 5;
#if defined(__has_feature)
#if __has_feature(thread_sanitizer) || __has_feature(__address_sanitizer__)
  numCombinations = 1;
#endif
#endif
  testWithTypes(
      kColumns, customize, false, {"long_val", "long_vals"}, numCombinations);
}

TEST_P(E2EFilterTest, flatMapComplexNoRecursiveNulls) {
  constexpr auto kColumns =
      "long_val:bigint,"
      "struct_vals:map<varchar,struct<v1:bigint, v2:float>>,"
      "array_vals:map<tinyint,array<int>>";
  flatMapColumns_ = {{"struct_vals", {}}, {"array_vals", {}}};
  auto customize = [this] {
    dataSetBuilder_->makeUniformMapKeys(common::Subfield("struct_vals"));
  };
  int numCombinations = 5;
#if defined(__has_feature)
#if __has_feature(thread_sanitizer) || __has_feature(__address_sanitizer__)
  numCombinations = 1;
#endif
#endif
#if !defined(NDEBUG)
  numCombinations = 1;
#endif
  testWithTypes(
      kColumns,
      customize,
      false,
      {"long_val"},
      numCombinations,
      /*withRecursiveNulls=*/false);
}

TEST_P(E2EFilterTest, flatMapComplex) {
  constexpr auto kColumns =
      "long_val:bigint,"
      "struct_vals:map<varchar,struct<v1:bigint, v2:float>>,"
      "array_vals:map<tinyint,array<int>>";
  flatMapColumns_ = {{"struct_vals", {}}, {"array_vals", {}}};
  auto customize = [this] {
    dataSetBuilder_->makeUniformMapKeys(common::Subfield("struct_vals"));
  };
  int numCombinations = 5;
#if defined(__has_feature)
#if __has_feature(thread_sanitizer) || __has_feature(__address_sanitizer__)
  numCombinations = 1;
#endif
#endif
#if !defined(NDEBUG)
  numCombinations = 1;
#endif
  testWithTypes(
      kColumns,
      customize,
      false,
      {"long_val"},
      numCombinations,
      /*withRecursiveNulls=*/true);
}

TEST_P(E2EFilterTest, mutationCornerCases) {
  testMutationCornerCases();
}

TEST_P(E2EFilterTest, timestamp) {
  testWithTypes(
      "timestamp_val:timestamp,"
      "long_val:bigint,"
      "timestamp_null:timestamp",
      [&]() { makeAllNulls("timestamp_null"); },
      true,
      {"timestamp_val", "long_val"},
      20);
}

TEST_P(E2EFilterTest, timestampNullFilter) {
  testWithTypes(
      "timestamp_val:timestamp,"
      "long_val:bigint,"
      "timestamp_null:timestamp",
      [&]() { makeAllNulls("timestamp_null"); },
      true,
      {"timestamp_null"},
      20);
}

TEST_P(E2EFilterTest, timestampNoNulls) {
  testWithTypes(
      "timestamp_val:timestamp,"
      "long_val:bigint",
      [&]() { makeNotNull(); },
      true,
      {"long_val", "timestamp_val"},
      20,
      /*withRecursiveNulls=*/false);
}

TEST_P(E2EFilterTest, timestampMultiple) {
  testWithTypes(
      "ts1:timestamp,"
      "ts2:timestamp,"
      "ts3:timestamp",
      [&]() { makeAllNulls("ts3"); },
      true,
      {"ts1", "ts2"},
      20);
}

TEST_P(E2EFilterTest, timestampArray) {
  testWithTypes(
      "long_val:bigint,"
      "timestamp_array:array<timestamp>",
      [&]() {},
      true,
      {"long_val", "timestamp_array"},
      20);
}

TEST_P(E2EFilterTest, timestampMap) {
  testWithTypes(
      "long_val:bigint,"
      "timestamp_map:map<int, timestamp>",
      [&]() {},
      true,
      {"long_val", "timestamp_map"},
      20);
}

TEST_P(E2EFilterTest, timestampInStruct) {
  testWithTypes(
      "long_val:bigint,"
      "struct_with_ts:struct<ts:timestamp, val:bigint>",
      [&]() {},
      true,
      {"long_val", "struct_with_ts"},
      20);
}

TEST_P(E2EFilterTest, timestampNestedArray) {
  testWithTypes(
      "long_val:bigint,"
      "nested_ts_array:array<array<timestamp>>",
      [&]() {},
      true,
      {"long_val"},
      20);
}

TEST_P(E2EFilterTest, timestampNestedStruct) {
  testWithTypes(
      "long_val:bigint,"
      "outer:struct<inner:struct<ts:timestamp>>",
      [&]() {},
      true,
      {"long_val", "outer"},
      20);
}

TEST_P(E2EFilterTest, timestampNestedMap) {
  testWithTypes(
      "long_val:bigint,"
      "nested_ts_map:map<int, map<int, timestamp>>",
      [&]() {},
      true,
      {"long_val"},
      20);
}

TEST_P(E2EFilterTest, timestampArrayInMap) {
  testWithTypes(
      "long_val:bigint,"
      "map_of_ts_array:map<int, array<timestamp>>",
      [&]() {},
      true,
      {"long_val"},
      20);
}

TEST_P(E2EFilterTest, timestampMapInArray) {
  testWithTypes(
      "long_val:bigint,"
      "array_of_ts_map:array<map<int, timestamp>>",
      [&]() {},
      true,
      {"long_val"},
      20);
}

TEST_P(E2EFilterTest, timestampStructInArray) {
  testWithTypes(
      "long_val:bigint,"
      "array_of_ts_struct:array<struct<ts:timestamp, val:bigint>>",
      [&]() {},
      true,
      {"long_val"},
      20);
}

TEST_P(E2EFilterTest, timestampArrayInStruct) {
  testWithTypes(
      "long_val:bigint,"
      "struct_with_ts_array:struct<ts_array:array<timestamp>, val:bigint>",
      [&]() {},
      true,
      {"long_val"},
      20);
}

INSTANTIATE_TEST_SUITE_P(
    E2EFilterTests,
    E2EFilterTest,
    testing::Combine(
        testing::Bool(),
        testing::Bool(),
        testing::Bool(),
        testing::Bool(),
        testing::Bool(),
        testing::Bool(),
        testing::Bool()),
    [](const ::testing::TestParamInfo<E2EFilterTestParams>& info) {
      return E2EFilterTestParam{info.param}.debugString();
    });

} // namespace

// Test that PrefixEncoding's readWithVisitor works correctly when reading
// sorted string data through the selective reader. This test derives from
// E2EFilterTest and overrides writeToMemory to explicitly specify
// PrefixEncoding for the string column.
class PrefixEncodingE2ETest : public E2EFilterTest {
 protected:
  bool indexEnabled() const override {
    return false;
  }

  void setUpRowReaderOptions(
      dwio::common::RowReaderOptions& opts,
      const std::shared_ptr<dwio::common::ScanSpec>& spec) override {
    opts.setScanSpec(spec);
    opts.setTimestampPrecision(TimestampPrecision::kNanoseconds);
    opts.setIndexEnabled(false);
    // PrefixEncoding requires string buffer optimization for correct
    // string_view lifecycle management in readWithVisitor.
    opts.setStringDecoderZeroCopy(true);
    opts.setNimblePreserveDictionaryEncoding(
        param().nimblePreserveDictionaryEncoding);
  }

  void writeToMemory(
      const TypePtr& type,
      const std::vector<RowVectorPtr>& batches,
      bool forRowGroupSkip,
      const std::vector<std::string>& /*indexColumns*/ = {}) override {
    VELOX_CHECK(!forRowGroupSkip);
    // Clear cached IO data from previous writes to avoid stale cache entries.
    if (cache_ != nullptr) {
      cache_->clear();
    }
    rowType_ = asRowType(type);

    // Create encoding layout tree with prefix encoding for string columns
    std::vector<EncodingLayoutTree> children;
    for (column_index_t i = 0;
         i < static_cast<column_index_t>(rowType_->size());
         ++i) {
      const auto& childType = rowType_->childAt(i);
      const auto& childName = rowType_->nameOf(i);

      if (childType->isVarchar() || childType->isVarbinary()) {
        // Use prefix encoding for string columns
        children.push_back(
            EncodingLayoutTree{
                Kind::Scalar,
                {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                  EncodingLayout{
                      EncodingType::Prefix,
                      {},
                      CompressionType::Uncompressed}}},
                std::string(childName)});
      } else {
        // For all other types, let the writer decide
        children.push_back(
            EncodingLayoutTree{Kind::Scalar, {}, std::string(childName)});
      }
    }

    EncodingLayoutTree encodingLayoutTree{
        Kind::Row, {}, "", std::move(children)};

    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions options;
    options.encodingLayoutTree.emplace(std::move(encodingLayoutTree));
    applyMultiChunkOptions(options);
    Writer writer(
        rowType_, std::move(writeFile), *rootPool_, std::move(options));
    for (auto& batch : batches) {
      writer.write(batch);
    }
    writer.close();
  }
};

INSTANTIATE_TEST_SUITE_P(
    PrefixEncodingE2ETests,
    PrefixEncodingE2ETest,
    testing::Combine(
        testing::Bool(),
        testing::Bool(),
        testing::Bool(),
        testing::Bool(),
        testing::Bool(),
        testing::Bool(),
        testing::Bool()),
    [](const ::testing::TestParamInfo<E2EFilterTestParams>& info) {
      return E2EFilterTestParam{info.param}.debugString();
    });

TEST_P(PrefixEncodingE2ETest, withoutFilterOnString) {
  for (bool withRecursiveNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("withRecursiveNulls={}", withRecursiveNulls));
    testWithTypes(
        "string_val:string,"
        "long_val:bigint",
        [&]() {
          // Make string values unique and sorted to benefit from prefix
          // encoding
          makeStringUnique("string_val");
        },
        /*wrapInStruct=*/false,
        /*filterable=*/{"long_val"},
        /*numCombinations=*/5,
        withRecursiveNulls);
  }
}

TEST_P(PrefixEncodingE2ETest, filterOnString) {
  // Test with filter on both string and integer columns.
  // This tests the skip functionality in readWithVisitor.
  for (bool withRecursiveNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("withRecursiveNulls={}", withRecursiveNulls));
    testWithTypes(
        "string_val:string,"
        "string_val_2:string,"
        "long_val:bigint",
        [&]() {
          makeStringUnique("string_val");
          makeStringUnique("string_val_2");
        },
        /*wrapInStruct=*/false,
        {"string_val", "long_val"},
        /*numCombinations=*/5,
        withRecursiveNulls);
  }
}

// Test for cross-stripe reading of string data with filters.
// This tests that dictionary column visitors work correctly when reading
// across multiple stripes, where the dictionary may change between stripes.
TEST_P(E2EFilterTest, stringDictionaryAcrossStripes) {
  // Use frequent stripe flushes to maximize cross-stripe reads.
  flushEveryNBatches_ = 1;
  for (bool withRecursiveNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("withRecursiveNulls={}", withRecursiveNulls));
    testWithTypes(
        "string_val:string,"
        "string_val_2:string,"
        "long_val:bigint",
        [&]() {
          makeStringDistribution("string_val", 100, true, false);
          makeStringDistribution("string_val_2", 170, false, true);
        },
        /*wrapInStruct=*/false,
        {"string_val", "long_val"},
        /*numCombinations=*/5,
        withRecursiveNulls);
  }
}

TEST_P(E2EFilterTest, stringUniqueAcrossStripes) {
  // Use frequent stripe flushes to maximize cross-stripe reads.
  flushEveryNBatches_ = 1;
  for (bool withRecursiveNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("withRecursiveNulls={}", withRecursiveNulls));
    testWithTypes(
        "string_val:string,"
        "string_val_2:string,"
        "long_val:bigint",
        [&]() {
          makeStringUnique("string_val");
          makeStringUnique("string_val_2");
        },
        /*wrapInStruct=*/false,
        {"string_val", "long_val"},
        /*numCombinations=*/5,
        withRecursiveNulls);
  }
}

// Test for nested encodings where dictionary encoding is NOT at the top level.
// This tests scenarios like Nullable -> Dictionary, which may have different
// behavior than Dictionary at the top level.
TEST_P(E2EFilterTest, stringDictionaryNestedUnderNullable) {
  // Force dictionary encoding for string columns via encoding layout.
  useForcedDictionaryEncoding_ = true;
  for (bool withRecursiveNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("withRecursiveNulls={}", withRecursiveNulls));
    testWithTypes(
        "string_val:string,"
        "string_val_2:string,"
        "long_val:bigint",
        [&]() {
          makeStringDistribution("string_val", 100, true, false);
          makeStringDistribution("string_val_2", 170, false, true);
        },
        /*wrapInStruct=*/false,
        {"string_val", "long_val"},
        /*numCombinations=*/5,
        withRecursiveNulls);
  }
}

TEST_P(E2EFilterTest, filterOnNestedDictionaryString) {
  // Force dictionary encoding for string columns via encoding layout.
  useForcedDictionaryEncoding_ = true;
  for (bool withRecursiveNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("withRecursiveNulls={}", withRecursiveNulls));
    testWithTypes(
        "string_val:string,"
        "string_val_2:string,"
        "long_val:bigint",
        [&]() {
          makeStringDistribution("string_val", 50, true, true);
          makeStringDistribution("string_val_2", 80, true, false);
        },
        /*wrapInStruct=*/false,
        {"string_val", "string_val_2", "long_val"},
        /*numCombinations=*/5,
        withRecursiveNulls);
  }
}

// Test combining cross-stripe reading with nested dictionary encodings.
// This is the most challenging scenario for dictionary column visitors.
TEST_P(E2EFilterTest, stringDictionaryCrossStripeWithNestedEncoding) {
  // Use both frequent stripe flushes and forced dictionary encoding.
  flushEveryNBatches_ = 1;
  useForcedDictionaryEncoding_ = true;
  for (bool withRecursiveNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("withRecursiveNulls={}", withRecursiveNulls));
    testWithTypes(
        "string_val:string,"
        "string_val_2:string,"
        "long_val:bigint",
        [&]() {
          makeStringDistribution("string_val", 100, true, false);
          makeStringDistribution("string_val_2", 170, false, true);
        },
        /*wrapInStruct=*/false,
        {"string_val", "long_val"},
        /*numCombinations=*/5,
        withRecursiveNulls);
  }
}

TEST_P(E2EFilterTest, stringFilterOnBothColumnsCrossStripeNested) {
  // Use both frequent stripe flushes and forced dictionary encoding.
  flushEveryNBatches_ = 1;
  useForcedDictionaryEncoding_ = true;
  for (bool withRecursiveNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("withRecursiveNulls={}", withRecursiveNulls));
    testWithTypes(
        "string_val:string,"
        "string_val_2:string,"
        "long_val:bigint",
        [&]() {
          makeStringDistribution("string_val", 50, true, true);
          makeStringDistribution("string_val_2", 80, true, false);
        },
        /*wrapInStruct=*/false,
        {"string_val", "string_val_2", "long_val"},
        /*numCombinations=*/5,
        withRecursiveNulls);
  }
}

// Test for nested encoding where dictionary is not at the top level.
// This test creates a MainlyConstant → Dictionary nesting by:
// 1. Creating data where most values are a constant (triggers MainlyConstant)
// 2. Forcing the otherValues_ child to use Dictionary encoding via
// EncodingLayoutTree
//
// This scenario is challenging because dictionaryEnabled() must handle
// deeper nesting like MainlyConstant→Dictionary, not just direct Dictionary
// or Nullable→Dictionary.
TEST_P(E2EFilterTest, nestedMainlyConstantWithDictionary) {
  // Create data where most values are a constant string.
  // ~90% of values will be the constant, ~10% will be "other" values.
  auto type = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});

  const size_t kRowCount = 1000;
  std::vector<std::string> stringVals;
  std::vector<int64_t> longVals;
  stringVals.reserve(kRowCount);
  longVals.reserve(kRowCount);

  const std::string kConstantValue = "MOSTLY_CONSTANT_VALUE";
  // For the "other" values, use low cardinality to encourage dictionary
  // encoding for the otherValues_ child of MainlyConstant.
  const std::vector<std::string> otherValues = {
      "OTHER_A", "OTHER_B", "OTHER_C", "OTHER_D", "OTHER_E"};

  for (size_t i = 0; i < kRowCount; ++i) {
    if (i % 10 == 0) {
      // ~10% of rows have "other" values (low cardinality for dictionary).
      stringVals.push_back(otherValues[i % otherValues.size()]);
    } else {
      // ~90% of rows have the constant value.
      stringVals.push_back(kConstantValue);
    }
    longVals.push_back(i);
  }

  auto stringVector =
      velox::test::VectorMaker(leafPool_.get())
          .flatVector<velox::StringView>(kRowCount, [&](auto row) {
            return velox::StringView(stringVals[row]);
          });
  auto longVector = velox::test::VectorMaker(leafPool_.get())
                        .flatVector<int64_t>(
                            kRowCount, [&](auto row) { return longVals[row]; });

  auto batch = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kRowCount,
      std::vector<VectorPtr>{stringVector, longVector});

  // Build EncodingLayoutTree that forces MainlyConstant with Dictionary child
  // for otherValues_.
  // The encoding hierarchy we want:
  //   Row
  //     -> Scalar (string_val): MainlyConstant
  //         -> OtherValues (child 1): Dictionary
  //             -> Alphabet (child 0): let writer decide
  //             -> Indices (child 1): let writer decide
  //     -> Scalar (long_val): default
  EncodingLayoutTree encodingLayoutTree{
      Kind::Row,
      {},
      "",
      {
          EncodingLayoutTree{
              Kind::Scalar,
              {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                EncodingLayout{
                    EncodingType::MainlyConstant,
                    {},
                    CompressionType::Uncompressed,
                    {
                        // Child 0: IsCommon (let the writer decide)
                        std::nullopt,
                        // Child 1: OtherValues -> Dictionary
                        // Dictionary encoding needs children for alphabet and
                        // indices.
                        EncodingLayout{
                            EncodingType::Dictionary,
                            {},
                            CompressionType::Uncompressed,
                            {
                                // Alphabet (id=0): let writer decide
                                std::nullopt,
                                // Indices (id=1): let writer decide
                                std::nullopt,
                            }},
                    }}}},
              "string_val"},
          EncodingLayoutTree{Kind::Scalar, {}, "long_val"},
      }};

  // Write the data with the forced encoding layout.
  auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
  WriterOptions writerOptions;
  writerOptions.encodingLayoutTree.emplace(std::move(encodingLayoutTree));
  applyMultiChunkOptions(writerOptions);
  Writer writer(type, std::move(writeFile), *rootPool_, writerOptions);
  writeInChunks(writer, batch);
  writer.close();

  // Read back and verify the data is correct.
  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());

  velox::dwio::common::ReaderOptions readerOpts(leafPool_.get());
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  // Read all rows and verify.
  auto result = BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(1000, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();

    // Use DecodedVector to handle both flat and dictionary-encoded vectors.
    // The string column might be wrapped in a DictionaryVector when using
    // dictionary encoding path.
    velox::DecodedVector decodedString(*rowResult->childAt(0));
    velox::DecodedVector decodedLong(*rowResult->childAt(1));

    for (size_t i = 0; i < rowResult->size(); ++i) {
      auto rowIdx = decodedLong.valueAt<int64_t>(i);
      std::string expected;
      if (rowIdx % 10 == 0) {
        expected = otherValues[rowIdx % otherValues.size()];
      } else {
        expected = kConstantValue;
      }
      ASSERT_EQ(decodedString.valueAt<velox::StringView>(i).str(), expected)
          << "Mismatch at row=" << rowIdx;
    }
  }

  EXPECT_EQ(totalRows, kRowCount);
}

// Writes string data with a forced encoding layout, reads back, and verifies
// dictionary vector return and data correctness.
TEST_P(E2EFilterTest, dictionaryVectorReturned) {
  const vector_size_t kRowCount = 1000;
  std::mt19937 gen(42);
  const int numDistinct = 5 + gen() % 20;
  std::vector<std::string> alphabet(numDistinct);
  for (int i = 0; i < numDistinct; ++i) {
    auto len = 3 + gen() % 30;
    alphabet[i].resize(len);
    for (auto& c : alphabet[i]) {
      c = 'a' + gen() % 26;
    }
  }
  std::vector<std::string> stringData(kRowCount);
  for (vector_size_t i = 0; i < kRowCount; ++i) {
    stringData[i] = alphabet[gen() % numDistinct];
  }

  rowType_ = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});
  verifyDictionaryVectorReturned(
      buildForcedDictionaryEncodingLayoutTree(), stringData);
}

// Regression test: the FlatVector backing the dictionary alphabet must hold
// the decoder's string buffers so that non-inline StringViews point into
// the vector's own buffers. FlatVector::validate() (debug mode) checks this
// invariant. Non-inline strings (>12 bytes) are the only ones affected since
// inline strings are stored directly in the StringView.
TEST_P(E2EFilterTest, dictionaryVectorNonInlineAlphabet) {
  const vector_size_t kRowCount = 1000;
  std::vector<std::string> stringData(kRowCount);
  for (vector_size_t i = 0; i < kRowCount; ++i) {
    stringData[i] = fmt::format("LONG_DICTIONARY_VALUE_NUMBER_{}", i % 10);
  }

  rowType_ = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});
  verifyDictionaryVectorReturned(
      buildForcedDictionaryEncodingLayoutTree(), stringData);
}

TEST_P(E2EFilterTest, mainlyConstantDictionaryVectorReturned) {
  const vector_size_t kRowCount = 1000;
  std::mt19937 gen(123);
  const int numOther = 3 + gen() % 10;
  std::vector<std::string> otherAlphabet(numOther);
  for (int i = 0; i < numOther; ++i) {
    auto len = 3 + gen() % 20;
    otherAlphabet[i].resize(len);
    for (auto& c : otherAlphabet[i]) {
      c = 'a' + gen() % 26;
    }
  }
  auto commonLen = 20 + gen() % 40;
  std::string commonValue(commonLen, 'x' + gen() % 3);

  std::vector<std::string> stringData(kRowCount);
  for (vector_size_t i = 0; i < kRowCount; ++i) {
    stringData[i] =
        (gen() % 10 == 0) ? otherAlphabet[gen() % numOther] : commonValue;
  }

  rowType_ = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});
  verifyDictionaryVectorReturned(
      buildForcedMainlyConstantDictionaryEncodingLayoutTree(), stringData);
}

// Fuzz test: exercises MC→Dict readIndicesWithVisitor with randomized data
// shapes. Runs 10000 iterations with varying row counts, alphabet sizes,
// common percentages, and null rates to catch subtle scattering/composition
// bugs in materializeIndices and readIndicesWithVisitor.
TEST_P(E2EFilterTest, fuzzMainlyConstantDictionaryVector) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "fuzzMainlyConstantDictionaryVector seed: " << seed;
  std::mt19937 rng(seed);

  const int kRowCount = 50 + rng() % 500;
  const int numOther = 1 + rng() % 8;
  const int commonPct = 50 + rng() % 45;
  const bool withNulls = rng() % 3 == 0;

  SCOPED_TRACE(
      fmt::format(
          "seed={} rows={} numOther={} commonPct={} withNulls={}",
          seed,
          kRowCount,
          numOther,
          commonPct,
          withNulls));

  std::vector<std::string> otherAlphabet(numOther);
  for (int j = 0; j < numOther; ++j) {
    auto len = 1 + rng() % 30;
    otherAlphabet[j].resize(len);
    for (auto& c : otherAlphabet[j]) {
      c = 'a' + rng() % 26;
    }
  }
  // Use a prefix that won't appear in otherAlphabet (which uses 'a'-'z').
  auto commonLen = 5 + rng() % 50;
  std::string commonValue(commonLen, '0' + rng() % 3);

  std::vector<std::string> stringData(kRowCount);
  for (int i = 0; i < kRowCount; ++i) {
    stringData[i] = (static_cast<int>(rng() % 100) < commonPct)
        ? commonValue
        : otherAlphabet[rng() % numOther];
  }
  // Ensure at least one non-common value at a non-null position so the Dict
  // child isn't empty. With nullEvery(7), position i is null when i % 7 == 0.
  {
    bool hasNonCommonAtNonNull = false;
    for (int i = 0; i < kRowCount; ++i) {
      if ((!withNulls || i % 7 != 0) && stringData[i] != commonValue) {
        hasNonCommonAtNonNull = true;
        break;
      }
    }
    if (!hasNonCommonAtNonNull) {
      // Place a non-common value at a position that won't be null.
      int pos = 1;
      while (withNulls && pos % 7 == 0) {
        ++pos;
      }
      stringData[pos] = otherAlphabet[0];
    }
  }

  rowType_ = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});

  auto type = asRowType(rowType_);
  auto stringVector =
      velox::test::VectorMaker(leafPool_.get())
          .flatVector<velox::StringView>(
              kRowCount,
              [&](auto row) { return velox::StringView(stringData[row]); },
              withNulls ? velox::test::VectorMaker::nullEvery(7) : nullptr);
  auto longVector =
      velox::test::VectorMaker(leafPool_.get())
          .flatVector<int64_t>(kRowCount, [](auto row) { return row; });

  auto batch = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kRowCount,
      std::vector<VectorPtr>{stringVector, longVector});

  sinkData_.clear();
  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.encodingLayoutTree.emplace(
        buildForcedMainlyConstantDictionaryEncodingLayoutTree());
    applyMultiChunkOptions(writerOptions);
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writeInChunks(writer, batch);
    writer.close();
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());

  velox::dwio::common::ReaderOptions readerOpts(leafPool_.get());
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(param().stringDecoderZeroCopy);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(
      param().stringDecoderZeroCopy);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  auto result = BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  size_t globalRow = 0;
  while (rowReader->next(kRowCount, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();

    auto stringChild = rowResult->childAt(0)->loadedVector();
    DecodedVector decoded(*stringChild);
    for (size_t i = 0; i < rowResult->size(); ++i) {
      if (withNulls && globalRow % 7 == 0) {
        ASSERT_TRUE(decoded.isNullAt(i))
            << "Expected null at row=" << globalRow;
      } else {
        ASSERT_FALSE(decoded.isNullAt(i))
            << "Unexpected null at row=" << globalRow;
        ASSERT_EQ(decoded.valueAt<StringView>(i).str(), stringData[globalRow])
            << "Mismatch at row=" << globalRow;
      }
      ++globalRow;
    }
  }
  EXPECT_EQ(totalRows, kRowCount);
}

TEST_P(E2EFilterTest, fuzzRleDictionaryVector) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "fuzzRleDictionaryVector seed: " << seed;
  std::mt19937 rng(seed);

  const int kRowCount = 50 + rng() % 500;
  const int numDistinct = 2 + rng() % 8;
  const int maxRunLength = 2 + rng() % 20;
  const bool withNulls = rng() % 3 == 0;

  SCOPED_TRACE(
      fmt::format(
          "seed={} rows={} numDistinct={} maxRunLength={} withNulls={}",
          seed,
          kRowCount,
          numDistinct,
          maxRunLength,
          withNulls));

  std::vector<std::string> alphabet(numDistinct);
  for (int j = 0; j < numDistinct; ++j) {
    auto len = 1 + rng() % 30;
    alphabet[j].resize(len);
    for (auto& c : alphabet[j]) {
      c = 'a' + rng() % 26;
    }
  }

  std::vector<std::string> stringData;
  stringData.reserve(kRowCount);
  while (static_cast<int>(stringData.size()) < kRowCount) {
    auto value = alphabet[rng() % numDistinct];
    auto runLen =
        std::min<int>(1 + rng() % maxRunLength, kRowCount - stringData.size());
    for (int j = 0; j < runLen; ++j) {
      stringData.push_back(value);
    }
  }

  rowType_ = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});

  auto type = asRowType(rowType_);
  auto stringVector =
      velox::test::VectorMaker(leafPool_.get())
          .flatVector<velox::StringView>(
              kRowCount,
              [&](auto row) { return velox::StringView(stringData[row]); },
              withNulls ? velox::test::VectorMaker::nullEvery(7) : nullptr);
  auto longVector =
      velox::test::VectorMaker(leafPool_.get())
          .flatVector<int64_t>(kRowCount, [](auto row) { return row; });

  auto batch = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kRowCount,
      std::vector<VectorPtr>{stringVector, longVector});

  sinkData_.clear();
  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.encodingLayoutTree.emplace(
        buildForcedRleDictionaryEncodingLayoutTree());
    applyMultiChunkOptions(writerOptions);
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writeInChunks(writer, batch);
    writer.close();
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());

  velox::dwio::common::ReaderOptions readerOpts(leafPool_.get());
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(param().stringDecoderZeroCopy);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(
      param().stringDecoderZeroCopy);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  auto result = BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  size_t globalRow = 0;
  while (rowReader->next(kRowCount, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();

    auto stringChild = rowResult->childAt(0)->loadedVector();
    DecodedVector decoded(*stringChild);
    for (size_t i = 0; i < rowResult->size(); ++i) {
      if (withNulls && globalRow % 7 == 0) {
        ASSERT_TRUE(decoded.isNullAt(i))
            << "Expected null at row=" << globalRow;
      } else {
        ASSERT_FALSE(decoded.isNullAt(i))
            << "Unexpected null at row=" << globalRow;
        ASSERT_EQ(decoded.valueAt<StringView>(i).str(), stringData[globalRow])
            << "Mismatch at row=" << globalRow;
      }
      ++globalRow;
    }
  }
  EXPECT_EQ(totalRows, kRowCount);
}

// Constant string column — all values identical. The encoding factory picks
// ConstantEncoding which is dictionary-enabled, so the reader should return
// a DictionaryVector with a 1-element alphabet.
TEST_P(E2EFilterTest, constantStringDictionaryVector) {
  constexpr int kRowCount = 500;
  const std::string constantValue = "always_the_same_value";

  rowType_ = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});
  auto type = asRowType(rowType_);
  auto stringVector =
      velox::test::VectorMaker(leafPool_.get())
          .flatVector<velox::StringView>(kRowCount, [&](auto /*row*/) {
            return velox::StringView(constantValue);
          });
  auto longVector =
      velox::test::VectorMaker(leafPool_.get())
          .flatVector<int64_t>(kRowCount, [](auto row) { return row; });

  auto batch = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kRowCount,
      std::vector<VectorPtr>{stringVector, longVector});

  sinkData_.clear();
  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions options;
    applyMultiChunkOptions(options);
    Writer writer(type, std::move(writeFile), *rootPool_, std::move(options));
    writeInChunks(writer, batch);
    writer.close();
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());

  velox::dwio::common::ReaderOptions readerOpts(leafPool_.get());
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(param().stringDecoderZeroCopy);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(
      param().nimblePreserveDictionaryEncoding);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  auto result = BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(kRowCount, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();

    auto stringChild = rowResult->childAt(0)->loadedVector();
    DecodedVector decoded(*stringChild);
    for (size_t i = 0; i < rowResult->size(); ++i) {
      ASSERT_FALSE(decoded.isNullAt(i));
      ASSERT_EQ(decoded.valueAt<StringView>(i).str(), constantValue);
    }
  }
  EXPECT_EQ(totalRows, kRowCount);
}

// Constant string with nulls — exercises Nullable→Constant path.
TEST_P(E2EFilterTest, constantStringWithNullsDictionaryVector) {
  constexpr int kRowCount = 500;
  const std::string constantValue = "nullable_constant";

  rowType_ = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});
  auto type = asRowType(rowType_);
  auto stringVector =
      velox::test::VectorMaker(leafPool_.get())
          .flatVector<velox::StringView>(
              kRowCount,
              [&](auto /*row*/) { return velox::StringView(constantValue); },
              velox::test::VectorMaker::nullEvery(7));
  auto longVector =
      velox::test::VectorMaker(leafPool_.get())
          .flatVector<int64_t>(kRowCount, [](auto row) { return row; });

  auto batch = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kRowCount,
      std::vector<VectorPtr>{stringVector, longVector});

  sinkData_.clear();
  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions options;
    applyMultiChunkOptions(options);
    Writer writer(type, std::move(writeFile), *rootPool_, std::move(options));
    writeInChunks(writer, batch);
    writer.close();
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());

  velox::dwio::common::ReaderOptions readerOpts(leafPool_.get());
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(param().stringDecoderZeroCopy);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(
      param().nimblePreserveDictionaryEncoding);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  auto result = BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  size_t globalRow = 0;
  while (rowReader->next(kRowCount, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();

    auto stringChild = rowResult->childAt(0)->loadedVector();
    DecodedVector decoded(*stringChild);
    for (size_t i = 0; i < rowResult->size(); ++i) {
      if (globalRow % 7 == 0) {
        ASSERT_TRUE(decoded.isNullAt(i))
            << "Expected null at row=" << globalRow;
      } else {
        ASSERT_FALSE(decoded.isNullAt(i));
        ASSERT_EQ(decoded.valueAt<StringView>(i).str(), constantValue);
      }
      ++globalRow;
    }
  }
  EXPECT_EQ(totalRows, kRowCount);
}

// Test for cross-chunk encoding changes where encoding type varies between
// chunks. This is a challenging scenario for dictionary column visitors because
// the first chunk may use dictionary encoding while subsequent chunks may use
// trivial encoding (or vice versa).
//
// The test creates data where:
// - Batch 0: Highly repetitive strings -> likely dictionary encoding
// - Batch 1: Unique strings -> likely trivial encoding
// - Batch 2: Highly repetitive strings -> likely dictionary encoding
// - Batch 3: Unique strings -> likely trivial encoding
// - ... (8 batches total)
//
// Each batch is 200 rows and flushed as a separate chunk. The reader requests
// 1000 rows per next() call, forcing each read to span ~5 chunks with
// different encodings.
TEST_P(E2EFilterTest, crossChunkEncodingChange) {
  // Use frequent chunk/stripe flushes to create multiple chunks with different
  // data characteristics.
  flushEveryNBatches_ = 1;

  // Create custom batches with alternating encoding characteristics.
  // Use small batches (200 rows) so that next(1000) spans multiple chunks.
  auto type =
      ROW({"string_val", "string_val_2", "long_val"},
          {VARCHAR(), VARCHAR(), BIGINT()});

  const size_t kRowsPerBatch = 200;
  const int kNumBatches = 8;
  std::vector<RowVectorPtr> batches;

  for (int batchIdx = 0; batchIdx < kNumBatches; ++batchIdx) {
    std::vector<std::string> stringVals;
    std::vector<std::string> stringVals2;
    std::vector<int64_t> longVals;

    stringVals.reserve(kRowsPerBatch);
    stringVals2.reserve(kRowsPerBatch);
    longVals.reserve(kRowsPerBatch);

    if (batchIdx % 2 == 0) {
      // Even batches: highly repetitive strings -> dictionary encoding.
      // Use very low cardinality to ensure dictionary encoding is chosen.
      for (size_t i = 0; i < kRowsPerBatch; ++i) {
        stringVals.push_back(fmt::format("DICT_VAL_{}", i % 5));
        stringVals2.push_back(fmt::format("DICT2_VAL_{}", i % 3));
        longVals.push_back(batchIdx * kRowsPerBatch + i);
      }
    } else {
      // Odd batches: unique strings -> trivial encoding.
      // Use unique values to prevent dictionary encoding.
      for (size_t i = 0; i < kRowsPerBatch; ++i) {
        stringVals.push_back(
            fmt::format("UNIQUE_VAL_BATCH{}_ROW{}_EXTRA_PADDING", batchIdx, i));
        stringVals2.push_back(
            fmt::format("UNIQUE2_VAL_BATCH{}_ROW{}_MORE_PADDING", batchIdx, i));
        longVals.push_back(batchIdx * kRowsPerBatch + i);
      }
    }

    auto stringVector =
        velox::test::VectorMaker(leafPool_.get())
            .flatVector<velox::StringView>(kRowsPerBatch, [&](auto row) {
              return velox::StringView(stringVals[row]);
            });
    auto stringVector2 =
        velox::test::VectorMaker(leafPool_.get())
            .flatVector<velox::StringView>(kRowsPerBatch, [&](auto row) {
              return velox::StringView(stringVals2[row]);
            });
    auto longVector = velox::test::VectorMaker(leafPool_.get())
                          .flatVector<int64_t>(kRowsPerBatch, [&](auto row) {
                            return longVals[row];
                          });

    batches.push_back(
        std::make_shared<RowVector>(
            leafPool_.get(),
            type,
            nullptr,
            kRowsPerBatch,
            std::vector<VectorPtr>{stringVector, stringVector2, longVector}));
  }

  // Write the data to memory.
  writeToMemory(type, batches, /*forRowGroupSkip=*/false);

  // Read back and verify the data is correct.
  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());

  velox::dwio::common::ReaderOptions crossChunkReaderOpts(leafPool_.get());
  crossChunkReaderOpts.setDataIoStats(dataIoStats_);
  crossChunkReaderOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(crossChunkReaderOpts, std::move(input));
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  // Read with batch size larger than chunk size, forcing cross-chunk reads.
  // Each chunk has 200 rows, so next(1000) should span ~5 chunks.
  auto result = BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(1000, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();

    // Verify the string values are correct.
    // Use DecodedVector to handle both flat and dictionary-encoded vectors.
    velox::DecodedVector decodedString(*rowResult->childAt(0));
    velox::DecodedVector decodedString2(*rowResult->childAt(1));
    velox::DecodedVector decodedLong(*rowResult->childAt(2));

    for (size_t i = 0; i < rowResult->size(); ++i) {
      auto globalRow = decodedLong.valueAt<int64_t>(i);
      auto batchIdx = globalRow / kRowsPerBatch;
      auto rowInBatch = globalRow % kRowsPerBatch;

      std::string expectedStr;
      std::string expectedStr2;
      if (batchIdx % 2 == 0) {
        expectedStr = fmt::format("DICT_VAL_{}", rowInBatch % 5);
        expectedStr2 = fmt::format("DICT2_VAL_{}", rowInBatch % 3);
      } else {
        expectedStr = fmt::format(
            "UNIQUE_VAL_BATCH{}_ROW{}_EXTRA_PADDING", batchIdx, rowInBatch);
        expectedStr2 = fmt::format(
            "UNIQUE2_VAL_BATCH{}_ROW{}_MORE_PADDING", batchIdx, rowInBatch);
      }

      ASSERT_EQ(decodedString.valueAt<velox::StringView>(i).str(), expectedStr)
          << "Mismatch at globalRow=" << globalRow;
      ASSERT_EQ(
          decodedString2.valueAt<velox::StringView>(i).str(), expectedStr2)
          << "Mismatch at globalRow=" << globalRow;
    }
  }

  EXPECT_EQ(totalRows, kNumBatches * kRowsPerBatch);
}

// --- Column extraction pushdown tests ---
// These tests write data to Nimble format, configure ScanSpec with extraction
// transforms, and verify the reader applies them correctly.

class NimbleExtractionTest : public E2EFilterTest {
 protected:
  struct ReadResult {
    VectorPtr result;
    std::unique_ptr<dwio::common::Reader> reader;
    std::unique_ptr<dwio::common::RowReader> rowReader;
  };

  // Write data to Nimble in-memory and read with a configured ScanSpec.
  // configureColSpec is invoked with the column's ScanSpec to apply
  // extraction settings (setExtractionType, setExtractionFieldIndex,
  // setFilter, setConstantValue on children, etc.).
  ReadResult readWithTransform(
      const RowVectorPtr& data,
      const std::string& columnName,
      std::function<void(common::ScanSpec*)> configureColSpec) {
    // Write.
    rowType_ = asRowType(data->type());
    WriterOptions options;
    applyMultiChunkOptions(options);
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    Writer writer(
        rowType_, std::move(writeFile), *rootPool_, std::move(options));
    writeInChunks(writer, data);
    writer.close();

    // Read.
    auto ioStats = std::make_shared<io::IoStatistics>();
    dwio::common::ReaderOptions readerOpts(leafPool_.get());
    readerOpts.setDataIoStats(ioStats);
    readerOpts.setMetadataIoStats(ioStats);
    auto input = std::make_unique<dwio::common::BufferedInput>(
        std::make_shared<InMemoryReadFile>(sinkData_), readerOpts.memoryPool());

    ReadResult rr;
    rr.reader = makeReader(readerOpts, std::move(input));

    auto spec = std::make_shared<common::ScanSpec>("<root>");
    spec->addAllChildFields(*rowType_);
    if (configureColSpec) {
      configureColSpec(spec->childByName(columnName));
    }

    dwio::common::RowReaderOptions rowReaderOpts;
    setUpRowReaderOptions(rowReaderOpts, spec);
    rr.rowReader = rr.reader->createRowReader(rowReaderOpts);

    rr.result = BaseVector::create(rowType_, 0, leafPool_.get());
    rr.rowReader->next(data->size(), rr.result);
    return rr;
  }
};

TEST_P(NimbleExtractionTest, mapKeysExtraction) {
  // Write MAP(VARCHAR, BIGINT), apply MapKeys transform -> ARRAY(VARCHAR).
  velox::test::VectorMaker vm(leafPool_.get());
  auto mapVector = vm.mapVector<StringView, int64_t>(
      {{{"a", 1}, {"b", 2}}, {{"c", 3}}, {{"d", 4}, {"e", 5}, {"f", 6}}});
  auto data = vm.rowVector({"col"}, {mapVector});

  // Reader handles MapKeys natively — no transform needed.
  auto rr = readWithTransform(data, "col", [](common::ScanSpec* colSpec) {
    colSpec->setExtractionType(common::ScanSpec::ExtractionType::kKeys);
  });

  auto* row = rr.result->as<RowVector>();
  ASSERT_EQ(row->size(), 3);
  auto* arr = row->childAt(0)->loadedVector()->as<ArrayVector>();
  ASSERT_EQ(arr->sizeAt(0), 2);
  ASSERT_EQ(arr->sizeAt(1), 1);
  ASSERT_EQ(arr->sizeAt(2), 3);
}

TEST_P(NimbleExtractionTest, sizeExtraction) {
  // Write MAP(VARCHAR, BIGINT), apply Size transform -> BIGINT.
  velox::test::VectorMaker vm(leafPool_.get());
  auto mapVector = vm.mapVector<StringView, int64_t>(
      {{{"a", 1}, {"b", 2}, {"c", 3}}, {{"d", 4}}});
  auto data = vm.rowVector({"col"}, {mapVector});

  // Reader handles Size natively — no transform needed.
  auto rr = readWithTransform(data, "col", [](common::ScanSpec* colSpec) {
    colSpec->setExtractionType(common::ScanSpec::ExtractionType::kSize);
  });

  auto* row = rr.result->as<RowVector>();
  ASSERT_EQ(row->size(), 2);
  auto* sizes = row->childAt(0)->loadedVector()->as<FlatVector<int64_t>>();
  ASSERT_EQ(sizes->valueAt(0), 3);
  ASSERT_EQ(sizes->valueAt(1), 1);
}

TEST_P(NimbleExtractionTest, mapValuesExtraction) {
  // Write MAP(VARCHAR, BIGINT), apply MapValues transform -> ARRAY(BIGINT).
  velox::test::VectorMaker vm(leafPool_.get());
  auto mapVector = vm.mapVector<StringView, int64_t>(
      {{{"a", 10}}, {{"b", 20}, {"c", 30}}, {{"d", 40}}});
  auto data = vm.rowVector({"col"}, {mapVector});

  // Reader handles MapValues natively — no transform needed.
  auto rr = readWithTransform(data, "col", [](common::ScanSpec* colSpec) {
    colSpec->setExtractionType(common::ScanSpec::ExtractionType::kValues);
  });

  auto* row = rr.result->as<RowVector>();
  ASSERT_EQ(row->size(), 3);
  auto* arr = row->childAt(0)->loadedVector()->as<ArrayVector>();
  ASSERT_EQ(arr->sizeAt(0), 1);
  ASSERT_EQ(arr->sizeAt(1), 2);
  ASSERT_EQ(arr->sizeAt(2), 1);
  auto* elements = arr->elements()->as<FlatVector<int64_t>>();
  ASSERT_EQ(elements->valueAt(0), 10);
  ASSERT_EQ(elements->valueAt(1), 20);
  ASSERT_EQ(elements->valueAt(2), 30);
  ASSERT_EQ(elements->valueAt(3), 40);
}

TEST_P(NimbleExtractionTest, mapKeyFilterExtraction) {
  // Write MAP(VARCHAR, BIGINT), apply MapKeyFilter for keys {"a","b"}.
  velox::test::VectorMaker vm(leafPool_.get());
  auto mapVector = vm.mapVector<StringView, int64_t>(
      {{{"a", 1}, {"b", 2}, {"c", 3}}, {{"a", 10}, {"d", 40}}});
  auto data = vm.rowVector({"col"}, {mapVector});

  // MapKeyFilter is implemented as an IN filter on the map keys ScanSpec
  // (type-preserving — MAP stays MAP).
  auto rr = readWithTransform(data, "col", [](common::ScanSpec* colSpec) {
    auto* keysSpec = colSpec->childByName(common::ScanSpec::kMapKeysFieldName);
    keysSpec->setFilter(
        std::make_unique<common::BytesValues>(
            std::vector<std::string>{"a", "b"}, /*nullAllowed=*/false));
  });

  auto* row = rr.result->as<RowVector>();
  ASSERT_EQ(row->size(), 2);
  auto* filteredMap = row->childAt(0)->loadedVector()->as<MapVector>();
  // Row 0: {"a":1, "b":2} kept, "c" filtered out.
  ASSERT_EQ(filteredMap->sizeAt(0), 2);
  // Row 1: {"a":10} kept, "d" filtered out.
  ASSERT_EQ(filteredMap->sizeAt(1), 1);
}

TEST_P(NimbleExtractionTest, structFieldExtraction) {
  // Write ROW(x: INT, y: VARCHAR), extract just field "x".
  velox::test::VectorMaker vm(leafPool_.get());
  auto structVector = vm.rowVector(
      {"x", "y"},
      {vm.flatVector<int32_t>({10, 20, 30}),
       vm.flatVector<StringView>({"aa", "bb", "cc"})});
  auto data = vm.rowVector({"col"}, {structVector});

  // kField on struct: extract field index 0 ("x"); mark "y" constant null.
  auto rr = readWithTransform(data, "col", [&](common::ScanSpec* colSpec) {
    colSpec->setExtractionType(common::ScanSpec::ExtractionType::kField);
    colSpec->setExtractionFieldIndex(0);
    colSpec->childByName("y")->setConstantValue(
        BaseVector::createNullConstant(VARCHAR(), 1, leafPool_.get()));
  });

  auto* row = rr.result->as<RowVector>();
  ASSERT_EQ(row->size(), 3);
  auto* xField = row->childAt(0)->loadedVector()->as<FlatVector<int32_t>>();
  ASSERT_EQ(xField->valueAt(0), 10);
  ASSERT_EQ(xField->valueAt(1), 20);
  ASSERT_EQ(xField->valueAt(2), 30);
}

TEST_P(NimbleExtractionTest, arraySizeExtraction) {
  // Write ARRAY(BIGINT), apply Size transform -> BIGINT.
  velox::test::VectorMaker vm(leafPool_.get());
  auto arrayVector = vm.arrayVector<int64_t>({{1, 2, 3}, {4}, {5, 6}});
  auto data = vm.rowVector({"col"}, {arrayVector});

  // Reader handles Size natively — no transform needed.
  auto rr = readWithTransform(data, "col", [](common::ScanSpec* colSpec) {
    colSpec->setExtractionType(common::ScanSpec::ExtractionType::kSize);
  });

  auto* row = rr.result->as<RowVector>();
  ASSERT_EQ(row->size(), 3);
  auto* sizes = row->childAt(0)->loadedVector()->as<FlatVector<int64_t>>();
  ASSERT_EQ(sizes->valueAt(0), 3);
  ASSERT_EQ(sizes->valueAt(1), 1);
  ASSERT_EQ(sizes->valueAt(2), 2);
}

TEST_P(NimbleExtractionTest, mapValuesStructFieldExtraction) {
  // Write MAP(VARCHAR, ROW(x: INT, y: INT)), apply
  // [MapValues, AE, StructField("x")] -> ARRAY(INT).
  velox::test::VectorMaker vm(leafPool_.get());
  auto keys = vm.flatVector<StringView>({"a", "b", "c"});
  auto structValues = vm.rowVector(
      {"x", "y"},
      {vm.flatVector<int32_t>({10, 20, 30}),
       vm.flatVector<int32_t>({100, 200, 300})});
  auto mapVector = vm.mapVector({0, 2}, keys, structValues);
  auto data = vm.rowVector({"col"}, {mapVector});

  // Reader handles MapValues via kValues and StructField("x") via kField
  // on the values struct — no remaining transform needed.
  auto rr = readWithTransform(data, "col", [&](common::ScanSpec* colSpec) {
    colSpec->setExtractionType(common::ScanSpec::ExtractionType::kValues);
    auto* valuesSpec =
        colSpec->childByName(common::ScanSpec::kMapValuesFieldName);
    valuesSpec->setExtractionType(common::ScanSpec::ExtractionType::kField);
    valuesSpec->setExtractionFieldIndex(0);
    valuesSpec->childByName("y")->setConstantValue(
        BaseVector::createNullConstant(INTEGER(), 1, leafPool_.get()));
  });

  auto* row = rr.result->as<RowVector>();
  ASSERT_EQ(row->size(), 2);
  auto* arr = row->childAt(0)->loadedVector()->as<ArrayVector>();
  ASSERT_EQ(arr->sizeAt(0), 2);
  ASSERT_EQ(arr->sizeAt(1), 1);
  auto* elements = arr->elements()->as<FlatVector<int32_t>>();
  ASSERT_EQ(elements->valueAt(0), 10);
  ASSERT_EQ(elements->valueAt(1), 20);
  ASSERT_EQ(elements->valueAt(2), 30);
}

INSTANTIATE_TEST_SUITE_P(
    NimbleExtractionTests,
    NimbleExtractionTest,
    ::testing::Values(
        E2EFilterTestParams{false, false, false, false, false, false, false},
        E2EFilterTestParams{false, false, false, false, false, true, false}));

// Exercises multi-chunk dictionary reads where each chunk has a distinct
// alphabet. All batches are dictionary-encoded and placed in a single stripe
// (each batch = one chunk). The reader requests 1000 rows per next() call,
// forcing the dictionary read path to merge alphabets across 5 chunks.
//
// This test guards against two bugs found in the original implementation:
// 1. Chunk-transition detection via pointer comparison was unreliable (the
//    allocator reused the same address after destroying the old encoding).
// 2. The onChunkLoad_ callback cleared the accumulated mergedAlphabet_
//    during the read, destroying previously merged alphabet entries.
TEST_P(E2EFilterTest, multiChunkDictionaryWithDifferentAlphabets) {
  auto type = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});
  rowType_ = asRowType(type);

  const size_t kRowsPerBatch = 200;
  const int kNumBatches = 8;
  std::vector<RowVectorPtr> batches;

  for (int batchIdx = 0; batchIdx < kNumBatches; ++batchIdx) {
    std::vector<std::string> stringStorage(kRowsPerBatch);
    std::vector<int64_t> longVals(kRowsPerBatch);

    // Each batch has a unique alphabet prefix so multi-chunk merging is
    // exercised (identical alphabets would trivially work).
    for (size_t i = 0; i < kRowsPerBatch; ++i) {
      stringStorage[i] = fmt::format("BATCH{}_VAL_{}", batchIdx, i % 5);
      longVals[i] = batchIdx * kRowsPerBatch + i;
    }

    auto stringVector =
        velox::test::VectorMaker(leafPool_.get())
            .flatVector<velox::StringView>(kRowsPerBatch, [&](auto row) {
              return velox::StringView(stringStorage[row]);
            });
    auto longVector = velox::test::VectorMaker(leafPool_.get())
                          .flatVector<int64_t>(kRowsPerBatch, [&](auto row) {
                            return longVals[row];
                          });

    batches.push_back(
        std::make_shared<RowVector>(
            leafPool_.get(),
            type,
            nullptr,
            kRowsPerBatch,
            std::vector<VectorPtr>{stringVector, longVector}));
  }

  // Write directly with flushLambda=false (no stripe flush → single stripe)
  // and chunkLambda=true (each batch = one chunk). This creates multiple
  // dictionary-encoded chunks within a single stripe.
  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    writerOptions.encodingLayoutTree.emplace(
        buildForcedDictionaryEncodingLayoutTree());
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    for (auto& batch : batches) {
      writer.write(batch);
    }
    writer.close();
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());

  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(param().stringDecoderZeroCopy);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(
      param().stringDecoderZeroCopy);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  // Read with batch size (1000) > chunk size (200), forcing cross-chunk reads.
  VectorPtr result = velox::BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(1000, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();

    auto stringChild = rowResult->childAt(0)->loadedVector();
    if (param().stringDecoderZeroCopy) {
      ASSERT_EQ(stringChild->encoding(), VectorEncoding::Simple::DICTIONARY)
          << "Expected DICTIONARY encoding for string column, got "
          << stringChild->encoding();
    }

    velox::DecodedVector decodedString(*stringChild);
    velox::DecodedVector decodedLong(*rowResult->childAt(1));

    for (size_t i = 0; i < rowResult->size(); ++i) {
      auto globalRow = decodedLong.valueAt<int64_t>(i);
      auto batchIdx = globalRow / kRowsPerBatch;
      auto rowInBatch = globalRow % kRowsPerBatch;
      auto expected = fmt::format("BATCH{}_VAL_{}", batchIdx, rowInBatch % 5);
      ASSERT_EQ(decodedString.valueAt<velox::StringView>(i).str(), expected)
          << "Mismatch at globalRow=" << globalRow;
    }
  }

  EXPECT_EQ(totalRows, kNumBatches * kRowsPerBatch);
}

// Regression test for two bugs in the dictionary reactive fallback path:
//   Bug 1: expandDictionaryToFlat was called after mergedAlphabet_.clear(),
//          causing out-of-bounds access when expanding dictionary indices.
//   Bug 2: The flat fallback's visitor added 0-based row numbers to
//          outputRows_ without biasing by the number of rows consumed during
//          the dictionary phase. This causes wrong cross-column row mapping.
//
// Both bugs require the reactive fallback: the string column starts with
// dictionary-compatible chunks but encounters a non-dictionary chunk mid-read.
// Bug 2 further requires hasFilter()=true so that useOutputRows()=true.
//
// The test verifies cross-column consistency: long_val encodes the original
// row position, so we can reconstruct the expected string_val from it. If
// Bug 2 causes wrong outputRows_, long_val maps to the wrong row, and the
// reconstructed string_val won't match the actual string_val.
TEST_P(E2EFilterTest, reactiveEncodingFallbackOutputRowsBias) {
  auto type = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});
  rowType_ = asRowType(type);

  const size_t kRowsPerBatch = 200;
  const int kNumBatches = 4;
  std::vector<RowVectorPtr> batches;

  for (int batchIdx = 0; batchIdx < kNumBatches; ++batchIdx) {
    std::vector<std::string> stringStorage(kRowsPerBatch);
    std::vector<int64_t> longVals(kRowsPerBatch);

    for (size_t i = 0; i < kRowsPerBatch; ++i) {
      longVals[i] = batchIdx * kRowsPerBatch + i;
      if (batchIdx % 2 == 0) {
        stringStorage[i] = fmt::format("DICT_VAL_{}", i % 5);
      } else {
        stringStorage[i] = fmt::format("UNIQUE_BATCH{}_ROW{}_PAD", batchIdx, i);
      }
    }

    auto stringVector =
        velox::test::VectorMaker(leafPool_.get())
            .flatVector<velox::StringView>(kRowsPerBatch, [&](auto row) {
              return velox::StringView(stringStorage[row]);
            });
    auto longVector = velox::test::VectorMaker(leafPool_.get())
                          .flatVector<int64_t>(kRowsPerBatch, [&](auto row) {
                            return longVals[row];
                          });

    batches.push_back(
        std::make_shared<RowVector>(
            leafPool_.get(),
            type,
            nullptr,
            kRowsPerBatch,
            std::vector<VectorPtr>{stringVector, longVector}));
  }

  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    for (auto& batch : batches) {
      writer.write(batch);
    }
    writer.close();
  }

  // Accept values from both dictionary and non-dictionary chunks so the
  // reactive fallback produces output rows from both phases.
  std::vector<std::string> accepted = {
      "DICT_VAL_0", "UNIQUE_BATCH1_ROW0_PAD", "UNIQUE_BATCH3_ROW0_PAD"};
  std::set<std::string> acceptedSet(accepted.begin(), accepted.end());

  size_t expectedRows = 0;
  for (int batchIdx = 0; batchIdx < kNumBatches; ++batchIdx) {
    for (size_t i = 0; i < kRowsPerBatch; ++i) {
      std::string val;
      if (batchIdx % 2 == 0) {
        val = fmt::format("DICT_VAL_{}", i % 5);
      } else {
        val = fmt::format("UNIQUE_BATCH{}_ROW{}_PAD", batchIdx, i);
      }
      if (acceptedSet.count(val)) {
        ++expectedRows;
      }
    }
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());

  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  scanSpec->childByName("string_val")
      ->setFilter(
          std::make_unique<velox::common::BytesValues>(accepted, false));
  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(param().stringDecoderZeroCopy);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr result = velox::BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(1000, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();

    velox::DecodedVector decodedString(*rowResult->childAt(0)->loadedVector());
    velox::DecodedVector decodedLong(*rowResult->childAt(1)->loadedVector());

    for (size_t i = 0; i < rowResult->size(); ++i) {
      auto str = decodedString.valueAt<velox::StringView>(i).str();
      auto longVal = decodedLong.valueAt<int64_t>(i);

      ASSERT_TRUE(acceptedSet.count(str))
          << "Value didn't pass filter: " << str;

      // Verify cross-column consistency. long_val encodes the original row
      // position, from which we can reconstruct the expected string_val.
      auto batchIdx = longVal / kRowsPerBatch;
      auto rowInBatch = longVal % kRowsPerBatch;
      std::string expectedStr;
      if (batchIdx % 2 == 0) {
        expectedStr = fmt::format("DICT_VAL_{}", rowInBatch % 5);
      } else {
        expectedStr =
            fmt::format("UNIQUE_BATCH{}_ROW{}_PAD", batchIdx, rowInBatch);
      }
      ASSERT_EQ(str, expectedStr)
          << "Cross-column row mapping mismatch at result row " << i
          << ", long_val=" << longVal << " (batch " << batchIdx << ", row "
          << rowInBatch << ")";
    }
  }

  EXPECT_EQ(totalRows, expectedRows);
}

// Exercises multi-chunk dictionary reads with different alphabets AND a filter
// on the string column. This validates that:
// 1. updateDictionaryScanState() correctly re-populates the filter cache as
//    chunks
//    are merged (new alphabet entries must be kUnknown, not stale).
// 2. DictionaryColumnVisitor::refreshState() correctly re-syncs after
//    chunk boundary alphabet growth.
// 3. Index offsetting (alphabetOffset) interacts correctly with filter cache
//    lookups — the visitor uses merged indices, not per-chunk indices.
TEST_P(E2EFilterTest, multiChunkDictionaryWithFilter) {
  auto type = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});
  rowType_ = asRowType(type);

  const size_t kRowsPerBatch = 200;
  const int kNumBatches = 8;
  std::vector<RowVectorPtr> batches;

  for (int batchIdx = 0; batchIdx < kNumBatches; ++batchIdx) {
    std::vector<std::string> stringStorage(kRowsPerBatch);
    std::vector<int64_t> longVals(kRowsPerBatch);

    for (size_t i = 0; i < kRowsPerBatch; ++i) {
      stringStorage[i] = fmt::format("BATCH{}_VAL_{}", batchIdx, i % 5);
      longVals[i] = batchIdx * kRowsPerBatch + i;
    }

    auto stringVector =
        velox::test::VectorMaker(leafPool_.get())
            .flatVector<velox::StringView>(kRowsPerBatch, [&](auto row) {
              return velox::StringView(stringStorage[row]);
            });
    auto longVector = velox::test::VectorMaker(leafPool_.get())
                          .flatVector<int64_t>(kRowsPerBatch, [&](auto row) {
                            return longVals[row];
                          });

    batches.push_back(
        std::make_shared<RowVector>(
            leafPool_.get(),
            type,
            nullptr,
            kRowsPerBatch,
            std::vector<VectorPtr>{stringVector, longVector}));
  }

  // Single stripe, each batch = one chunk, forced dictionary encoding.
  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    writerOptions.encodingLayoutTree.emplace(
        buildForcedDictionaryEncodingLayoutTree());
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    for (auto& batch : batches) {
      writer.write(batch);
    }
    writer.close();
  }

  // Apply a BytesValues filter that matches entries from multiple chunks.
  // BATCH0_VAL_2 is in chunk 0, BATCH3_VAL_1 is in chunk 3, BATCH7_VAL_4 is
  // in chunk 7 — the filter cache must be correctly extended at each boundary.
  std::vector<std::string> accepted = {
      "BATCH0_VAL_2", "BATCH3_VAL_1", "BATCH7_VAL_4"};
  std::set<std::string> acceptedSet(accepted.begin(), accepted.end());

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());

  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  scanSpec->childByName("string_val")
      ->setFilter(
          std::make_unique<velox::common::BytesValues>(accepted, false));
  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(param().stringDecoderZeroCopy);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr result = velox::BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(1000, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();

    velox::DecodedVector decodedString(*rowResult->childAt(0)->loadedVector());
    for (size_t i = 0; i < rowResult->size(); ++i) {
      auto str = decodedString.valueAt<velox::StringView>(i).str();
      ASSERT_TRUE(acceptedSet.count(str))
          << "Unexpected value passed filter: " << str;
    }
  }

  // Each accepted value appears in exactly one batch with kRowsPerBatch/5
  // repetitions (200/5 = 40 rows each).
  EXPECT_EQ(totalRows, accepted.size() * (kRowsPerBatch / 5));
}

// Regression: filter-only dictionary string column drops rows due to stale
// null state across batches. For filter-only columns, prepareRead skips
// prepareNulls, so anyNulls_/returnReaderNulls_ leak from the previous batch.
// Stale anyNulls_=true causes the null reconciliation to be skipped, and
// resultNulls() returns a stale buffer instead of the fresh nullsInReadRange_,
// corrupting filterDictionaryIndices output.
TEST_P(E2EFilterTest, filterOnlyDictionaryStringColumn) {
  if (!param().stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  auto type = ROW({"filter_col", "select_col"}, {VARCHAR(), VARCHAR()});
  rowType_ = asRowType(type);

  // 800 rows total, read with batch_size=50 → 16 batches.
  // Null positions differ per row (every 7th row is null), so the stale
  // resultNulls_ from batch N has nulls at positions that don't match
  // batch N+1's actual nulls, corrupting filter evaluation.
  constexpr size_t kTotalRows = 800;
  velox::test::VectorMaker maker(leafPool_.get());

  std::vector<std::string> filterStorage(kTotalRows);
  std::vector<std::string> selectStorage(kTotalRows);
  for (size_t i = 0; i < kTotalRows; ++i) {
    filterStorage[i] = fmt::format("FILTER_{}", i % 20);
    selectStorage[i] = fmt::format("SELECT_{}", i % 10);
  }

  auto filterVector = maker.flatVector<velox::StringView>(
      kTotalRows,
      [&](auto row) { return velox::StringView(filterStorage[row]); },
      velox::test::VectorMaker::nullEvery(7));
  auto selectVector = maker.flatVector<velox::StringView>(
      kTotalRows,
      [&](auto row) { return velox::StringView(selectStorage[row]); });

  auto batch = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kTotalRows,
      std::vector<VectorPtr>{filterVector, selectVector});

  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.encodingLayoutTree.emplace(
        buildForcedDictionaryEncodingLayoutTree());
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writer.write(batch);
    writer.close();
  }

  std::vector<std::string> accepted = {"FILTER_3", "FILTER_7"};

  // Compute expected row count.
  size_t expectedRows = 0;
  for (size_t i = 0; i < kTotalRows; ++i) {
    if (i % 7 == 0)
      continue;
    auto val = fmt::format("FILTER_{}", i % 20);
    if (val == "FILTER_3" || val == "FILTER_7") {
      ++expectedRows;
    }
  }

  // Read with filter_col as filter-only (not projected), small batch size
  // to force multiple next() calls and trigger stale state across batches.
  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());
  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));

  auto outputType = ROW({"select_col"}, {VARCHAR()});
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outputType);
  scanSpec->getOrCreateChild(velox::common::Subfield("filter_col"))
      ->setFilter(
          std::make_unique<velox::common::BytesValues>(accepted, false));

  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(true);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr result = velox::BaseVector::create(outputType, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(50, result)) {
    totalRows += result->as<RowVector>()->size();
  }

  EXPECT_EQ(totalRows, expectedRows)
      << "Filter-only dictionary column produced wrong row count";
}

// Regression: reading a dictionary-encoded string field inside a struct with
// a high null rate (~99%) crashes with "Failed to read chunk header" when the
// dict path is enabled and the column is read through the struct reader with
// struct-level incomingNulls. The crash is specific to aggregation-like access
// patterns where the struct is loaded lazily at the top level.
TEST_P(E2EFilterTest, dictionaryStringInSparseNullableStruct) {
  if (!param().stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr size_t kTotalRows = 20000;
  constexpr size_t kStructNonNullCount = 20;

  auto innerType = ROW({"report_id"}, {VARCHAR()});
  auto type = ROW({"select_col", "page_info"}, {BIGINT(), innerType});
  rowType_ = asRowType(type);

  velox::test::VectorMaker maker(leafPool_.get());

  auto selectVector = maker.flatVector<int64_t>(
      kTotalRows, [](auto row) { return static_cast<int64_t>(row); });

  std::vector<std::string> reportIdStorage(kStructNonNullCount);
  for (size_t i = 0; i < kStructNonNullCount; ++i) {
    reportIdStorage[i] = fmt::format("report-{}", i);
  }

  auto reportIdVector = maker.flatVector<velox::StringView>(
      kStructNonNullCount,
      [&](auto row) { return velox::StringView(reportIdStorage[row]); });

  auto innerStruct = std::make_shared<RowVector>(
      leafPool_.get(),
      innerType,
      nullptr,
      kStructNonNullCount,
      std::vector<VectorPtr>{reportIdVector});

  auto structNulls =
      velox::AlignedBuffer::allocate<bool>(kTotalRows, leafPool_.get());
  auto* rawStructNulls = structNulls->asMutable<uint64_t>();
  velox::bits::fillBits(rawStructNulls, 0, kTotalRows, velox::bits::kNull);
  size_t nonNullIndex = 0;
  for (size_t i = 0; i < kTotalRows && nonNullIndex < kStructNonNullCount;
       i += kTotalRows / kStructNonNullCount) {
    velox::bits::setBit(rawStructNulls, i, true);
    ++nonNullIndex;
  }

  auto indices = velox::AlignedBuffer::allocate<vector_size_t>(
      kTotalRows, leafPool_.get());
  auto* rawIndices = indices->asMutable<vector_size_t>();
  vector_size_t innerIdx = 0;
  for (size_t i = 0; i < kTotalRows; ++i) {
    if (velox::bits::isBitSet(rawStructNulls, i)) {
      rawIndices[i] = innerIdx++;
    } else {
      rawIndices[i] = 0;
    }
  }

  auto pageInfoVector = BaseVector::wrapInDictionary(
      structNulls, indices, kTotalRows, innerStruct);

  auto batch = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kTotalRows,
      std::vector<VectorPtr>{selectVector, pageInfoVector});

  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    auto layoutTree = EncodingLayoutTree{
        Kind::Row,
        {},
        "",
        {EncodingLayoutTree{Kind::Scalar, {}, "select_col"},
         EncodingLayoutTree{
             Kind::Row,
             {},
             "page_info",
             {EncodingLayoutTree{
                 Kind::Scalar,
                 {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                   EncodingLayout{
                       EncodingType::Dictionary,
                       {},
                       CompressionType::Uncompressed,
                       {std::nullopt, std::nullopt}}}},
                 "report_id"}}}}};
    writerOptions.encodingLayoutTree.emplace(std::move(layoutTree));
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writer.write(batch);
    writer.close();
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());
  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));

  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);

  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(true);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  // Read using the same pattern as count(page_info.report_id): the struct
  // column is read lazily, and accessing the inner field triggers lazy
  // loading through ColumnLoader, which calls readWithTiming with
  // incomingNulls from the struct reader.
  VectorPtr result = velox::BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  size_t totalNonNull = 0;
  constexpr size_t kBatchSize = 1000;
  while (rowReader->next(kBatchSize, result)) {
    auto* rowVector = result->as<RowVector>();
    totalRows += rowVector->size();
    auto pageInfo = rowVector->childAt(1);
    // Trigger lazy loading — this calls ColumnLoader::loadInternal which
    // calls the struct reader's advanceFieldReader + readWithTiming with
    // incomingNulls = structReader->nulls().
    pageInfo->loadedVector();
    for (vector_size_t i = 0; i < pageInfo->size(); ++i) {
      if (!pageInfo->isNullAt(i)) {
        ++totalNonNull;
      }
    }
  }

  EXPECT_EQ(totalRows, kTotalRows);
  EXPECT_EQ(totalNonNull, kStructNonNullCount);
}

// Same as above but with multiple struct children matching the production
// table's page_info schema, and simulates the aggregation access pattern
// where count(struct.field) accesses only one child of the struct.
TEST_P(E2EFilterTest, dictionaryStringInSparseNullableStructMultiChild) {
  if (!param().stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr size_t kTotalRows = 20000;
  constexpr size_t kStructNonNullCount = 20;

  auto innerType =
      ROW({"oncall", "tier", "report_id", "report_uid"},
          {VARCHAR(), VARCHAR(), VARCHAR(), VARCHAR()});
  auto type = ROW({"select_col", "page_info"}, {BIGINT(), innerType});
  rowType_ = asRowType(type);

  velox::test::VectorMaker maker(leafPool_.get());

  auto selectVector = maker.flatVector<int64_t>(
      kTotalRows, [](auto row) { return static_cast<int64_t>(row); });

  // oncall and tier are all-null within non-null struct instances.
  auto oncallVector = BaseVector::createNullConstant(
      VARCHAR(), kStructNonNullCount, leafPool_.get());
  auto tierVector = BaseVector::createNullConstant(
      VARCHAR(), kStructNonNullCount, leafPool_.get());

  std::vector<std::string> ridStorage(kStructNonNullCount);
  std::vector<std::string> ruidStorage(kStructNonNullCount);
  for (size_t i = 0; i < kStructNonNullCount; ++i) {
    ridStorage[i] = fmt::format("report-{}", i);
    ruidStorage[i] = fmt::format("uid-{}", i);
  }

  auto ridVector = maker.flatVector<velox::StringView>(
      kStructNonNullCount,
      [&](auto row) { return velox::StringView(ridStorage[row]); });
  auto ruidVector = maker.flatVector<velox::StringView>(
      kStructNonNullCount,
      [&](auto row) { return velox::StringView(ruidStorage[row]); });

  auto innerStruct = std::make_shared<RowVector>(
      leafPool_.get(),
      innerType,
      nullptr,
      kStructNonNullCount,
      std::vector<VectorPtr>{oncallVector, tierVector, ridVector, ruidVector});

  auto structNulls =
      velox::AlignedBuffer::allocate<bool>(kTotalRows, leafPool_.get());
  auto* rawStructNulls = structNulls->asMutable<uint64_t>();
  velox::bits::fillBits(rawStructNulls, 0, kTotalRows, velox::bits::kNull);
  size_t nonNullIndex = 0;
  for (size_t i = 0; i < kTotalRows && nonNullIndex < kStructNonNullCount;
       i += kTotalRows / kStructNonNullCount) {
    velox::bits::setBit(rawStructNulls, i, true);
    ++nonNullIndex;
  }

  auto indices = velox::AlignedBuffer::allocate<vector_size_t>(
      kTotalRows, leafPool_.get());
  auto* rawIndices = indices->asMutable<vector_size_t>();
  vector_size_t innerIdx = 0;
  for (size_t i = 0; i < kTotalRows; ++i) {
    if (velox::bits::isBitSet(rawStructNulls, i)) {
      rawIndices[i] = innerIdx++;
    } else {
      rawIndices[i] = 0;
    }
  }

  auto pageInfoVector = BaseVector::wrapInDictionary(
      structNulls, indices, kTotalRows, innerStruct);

  auto batch = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kTotalRows,
      std::vector<VectorPtr>{selectVector, pageInfoVector});

  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    auto layoutTree = EncodingLayoutTree{
        Kind::Row,
        {},
        "",
        {EncodingLayoutTree{Kind::Scalar, {}, "select_col"},
         EncodingLayoutTree{
             Kind::Row,
             {},
             "page_info",
             {EncodingLayoutTree{
                  Kind::Scalar,
                  {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                    EncodingLayout{
                        EncodingType::Dictionary,
                        {},
                        CompressionType::Uncompressed,
                        {std::nullopt, std::nullopt}}}},
                  "oncall"},
              EncodingLayoutTree{
                  Kind::Scalar,
                  {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                    EncodingLayout{
                        EncodingType::Dictionary,
                        {},
                        CompressionType::Uncompressed,
                        {std::nullopt, std::nullopt}}}},
                  "tier"},
              EncodingLayoutTree{
                  Kind::Scalar,
                  {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                    EncodingLayout{
                        EncodingType::Dictionary,
                        {},
                        CompressionType::Uncompressed,
                        {std::nullopt, std::nullopt}}}},
                  "report_id"},
              EncodingLayoutTree{
                  Kind::Scalar,
                  {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                    EncodingLayout{
                        EncodingType::Dictionary,
                        {},
                        CompressionType::Uncompressed,
                        {std::nullopt, std::nullopt}}}},
                  "report_uid"}}}}};
    writerOptions.encodingLayoutTree.emplace(std::move(layoutTree));
    applyMultiChunkOptions(writerOptions);
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writeInChunks(writer, batch, 4);
    writer.close();
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());
  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));

  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);

  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(true);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr result = velox::BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  size_t totalNonNull = 0;
  constexpr size_t kBatchSize = 1000;
  while (rowReader->next(kBatchSize, result)) {
    auto* rowVector = result->as<RowVector>();
    totalRows += rowVector->size();
    auto pageInfo = rowVector->childAt(1);
    pageInfo->loadedVector();
    for (vector_size_t i = 0; i < pageInfo->size(); ++i) {
      if (!pageInfo->isNullAt(i)) {
        ++totalNonNull;
      }
    }
  }

  EXPECT_EQ(totalRows, kTotalRows);
  EXPECT_EQ(totalNonNull, kStructNonNullCount);
}

// Regression: when a filter-only dictionary read spans a dict chunk followed
// by a non-dict chunk (abandon path), the dict portion's rows are read with
// the filter suppressed but never filtered post-hoc — filterDictionaryIndices
// is skipped because abandonDictionary=true. The flat continuation only
// filters the remaining rows, so the dict portion passes unfiltered.
TEST_P(E2EFilterTest, filterOnlyDictAbandonDropsFilter) {
  if (!param().stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  auto type = ROW({"filter_col", "select_col"}, {VARCHAR(), VARCHAR()});
  rowType_ = asRowType(type);

  velox::test::VectorMaker maker(leafPool_.get());

  // Chunk 1: low cardinality → dictionary encoding.
  constexpr size_t kChunk1Rows = 500;
  std::vector<std::string> filter1(kChunk1Rows);
  std::vector<std::string> select1(kChunk1Rows);
  for (size_t i = 0; i < kChunk1Rows; ++i) {
    filter1[i] = fmt::format("EVENT_{}", i % 10);
    select1[i] = fmt::format("VAL_{}", i % 5);
  }
  auto chunk1 = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kChunk1Rows,
      std::vector<VectorPtr>{
          maker.flatVector<velox::StringView>(
              kChunk1Rows,
              [&](auto row) { return velox::StringView(filter1[row]); }),
          maker.flatVector<velox::StringView>(kChunk1Rows, [&](auto row) {
            return velox::StringView(select1[row]);
          })});

  // Chunk 2: all unique → trivial encoding (non-dict, forces abandon).
  constexpr size_t kChunk2Rows = 300;
  std::vector<std::string> filter2(kChunk2Rows);
  std::vector<std::string> select2(kChunk2Rows);
  for (size_t i = 0; i < kChunk2Rows; ++i) {
    filter2[i] =
        fmt::format("UNIQUE_EVENT_{}_padding_to_avoid_inline_string", i);
    select2[i] = fmt::format("UNIQUE_VAL_{}", i);
  }
  auto chunk2 = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kChunk2Rows,
      std::vector<VectorPtr>{
          maker.flatVector<velox::StringView>(
              kChunk2Rows,
              [&](auto row) { return velox::StringView(filter2[row]); }),
          maker.flatVector<velox::StringView>(kChunk2Rows, [&](auto row) {
            return velox::StringView(select2[row]);
          })});

  // Write with chunking: each write = one chunk.
  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writer.write(chunk1);
    writer.write(chunk2);
    writer.close();
  }

  // Filter that matches some dict entries but no trivial entries.
  std::vector<std::string> accepted = {"EVENT_3", "EVENT_7"};
  std::set<std::string> acceptedSet(accepted.begin(), accepted.end());

  // Expected: only matching rows from chunk 1 (chunk 2 has unique values,
  // none match).
  size_t expectedRows = 0;
  for (size_t i = 0; i < kChunk1Rows; ++i) {
    if (acceptedSet.count(filter1[i])) {
      ++expectedRows;
    }
  }

  // Read with filter-only column, batch size that spans both chunks.
  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());
  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));

  auto outputType = ROW({"select_col"}, {VARCHAR()});
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outputType);
  scanSpec->getOrCreateChild(velox::common::Subfield("filter_col"))
      ->setFilter(
          std::make_unique<velox::common::BytesValues>(accepted, false));

  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(true);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  // Batch size 800 spans chunk 1 (500 rows, dict) + chunk 2 (300 rows,
  // non-dict), forcing the abandon path mid-batch.
  VectorPtr result = velox::BaseVector::create(outputType, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(800, result)) {
    totalRows += result->as<RowVector>()->size();
  }

  EXPECT_EQ(totalRows, expectedRows)
      << "Dict abandon path dropped filter on dict portion";
}

// Regression: when all non-null values in a dictionary-encoded string field
// inside a mostly-null struct are consumed in earlier batches, later batches
// contain only null rows. readDictionaryIndicesImpl unconditionally calls
// loadNextChunk when remainingValues_==0, even when no more chunks exist,
// crashing with "Failed to read chunk header". The fix checks the null bitmap
// to verify all remaining rows are null before skipping the chunk load.
TEST_P(E2EFilterTest, dictionaryChunkExhaustedAllNullTail) {
  if (!param().stringDecoderZeroCopy ||
      !param().nimblePreserveDictionaryEncoding) {
    GTEST_SKIP()
        << "Dictionary path requires stringDecoderZeroCopy and nimblePreserveDict";
  }

  // 20,000 rows with only 20 struct-non-null. All non-null values are the
  // same string → ConstantEncoding with rowCount=20. The non-null positions
  // cluster in the first quarter so later batches are entirely null.
  constexpr size_t kTotalRows = 20000;
  constexpr size_t kNonNullCount = 20;

  auto type =
      ROW({"select_col", "info"}, {BIGINT(), ROW({{"val"}}, {VARCHAR()})});
  rowType_ = asRowType(type);

  velox::test::VectorMaker maker(leafPool_.get());

  auto selectVector = maker.flatVector<int64_t>(
      kTotalRows, [](auto row) { return static_cast<int64_t>(row); });

  auto valVector = maker.flatVector<velox::StringView>(
      kNonNullCount,
      [](auto /*row*/) { return velox::StringView("constant-val"); });

  auto innerType = ROW({{"val"}}, {VARCHAR()});
  auto innerStruct = std::make_shared<RowVector>(
      leafPool_.get(),
      innerType,
      nullptr,
      kNonNullCount,
      std::vector<VectorPtr>{valVector});

  auto structNulls =
      velox::AlignedBuffer::allocate<bool>(kTotalRows, leafPool_.get());
  auto* rawStructNulls = structNulls->asMutable<uint64_t>();
  velox::bits::fillBits(rawStructNulls, 0, kTotalRows, velox::bits::kNull);
  for (size_t i = 0; i < kNonNullCount; ++i) {
    velox::bits::setBit(rawStructNulls, i * 250, true);
  }

  auto indices = velox::AlignedBuffer::allocate<vector_size_t>(
      kTotalRows, leafPool_.get());
  auto* rawIndices = indices->asMutable<vector_size_t>();
  vector_size_t innerIdx = 0;
  for (size_t i = 0; i < kTotalRows; ++i) {
    rawIndices[i] = velox::bits::isBitSet(rawStructNulls, i) ? innerIdx++ : 0;
  }

  auto infoVector = BaseVector::wrapInDictionary(
      structNulls, indices, kTotalRows, innerStruct);

  auto batch = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kTotalRows,
      std::vector<VectorPtr>{selectVector, infoVector});

  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writer.write(batch);
    writer.close();
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());
  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));

  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);

  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(true);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr result = velox::BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  size_t totalNonNull = 0;
  constexpr size_t kBatchSize = 1000;
  while (rowReader->next(kBatchSize, result)) {
    auto* rowVector = result->as<RowVector>();
    totalRows += rowVector->size();
    auto info = rowVector->childAt(1);
    info->loadedVector();
    for (vector_size_t i = 0; i < info->size(); ++i) {
      if (!info->isNullAt(i)) {
        ++totalNonNull;
        auto innerRow = info->wrappedVector()->as<RowVector>();
        auto wrappedIdx = info->wrappedIndex(i);
        auto val = innerRow->childAt(0)->as<SimpleVector<velox::StringView>>();
        ASSERT_FALSE(val->isNullAt(wrappedIdx));
        EXPECT_EQ(val->valueAt(wrappedIdx), velox::StringView("constant-val"));
      }
    }
  }

  EXPECT_EQ(totalRows, kTotalRows);
  EXPECT_EQ(totalNonNull, kNonNullCount);
}

namespace {
// Builds a {select_col, info:{val}} batch of `totalRows` rows with the `info`
// struct non-null only at `nonNullRows` (values from `values`), else null. Each
// batch, written as its own chunk, forms one value-stream chunk.
RowVectorPtr makeSparseDictStructBatch(
    velox::memory::MemoryPool* pool,
    const RowTypePtr& type,
    const RowTypePtr& innerType,
    int64_t globalRowBase,
    size_t totalRows,
    const std::vector<size_t>& nonNullRows,
    const std::vector<std::string>& values) {
  VELOX_CHECK_EQ(nonNullRows.size(), values.size());
  velox::test::VectorMaker maker(pool);
  auto selectVector = maker.flatVector<int64_t>(
      static_cast<vector_size_t>(totalRows),
      [&](auto row) { return globalRowBase + static_cast<int64_t>(row); });

  auto valVector = maker.flatVector<velox::StringView>(
      static_cast<vector_size_t>(values.size()),
      [&](auto row) { return velox::StringView(values[row]); });
  auto innerStruct = std::make_shared<RowVector>(
      pool,
      innerType,
      nullptr,
      values.size(),
      std::vector<VectorPtr>{valVector});

  auto structNulls = velox::AlignedBuffer::allocate<bool>(totalRows, pool);
  auto* rawStructNulls = structNulls->asMutable<uint64_t>();
  velox::bits::fillBits(
      rawStructNulls,
      0,
      static_cast<vector_size_t>(totalRows),
      velox::bits::kNull);
  auto indices = velox::AlignedBuffer::allocate<vector_size_t>(totalRows, pool);
  auto* rawIndices = indices->asMutable<vector_size_t>();
  std::fill(rawIndices, rawIndices + totalRows, 0);
  for (size_t i = 0; i < nonNullRows.size(); ++i) {
    velox::bits::setBit(rawStructNulls, nonNullRows[i], true);
    rawIndices[nonNullRows[i]] = static_cast<vector_size_t>(i);
  }
  auto infoVector = BaseVector::wrapInDictionary(
      structNulls, indices, static_cast<vector_size_t>(totalRows), innerStruct);
  return std::make_shared<RowVector>(
      pool,
      type,
      nullptr,
      totalRows,
      std::vector<VectorPtr>{selectVector, infoVector});
}
} // namespace

// Two dictionary-compatible value chunks (each a single repeated string ->
// ConstantEncoding) separated by a large all-null gap, plus an all-null tail
// after the last chunk. The struct null stream (20,000 rows) and the child
// value stream (20 values in two chunks) are separate ChunkedDecoders, so once
// a chunk's values are consumed the reader must decide whether to load the next
// chunk from the null bitmap, not from remaining values alone. This is the
// non-tail-specific counterpart to dictionaryChunkExhaustedAllNullTail: the gap
// sits *between* two chunks, and non-nulls are spread across read batches.
TEST_P(E2EFilterTest, dictionaryChunkGapBetweenChunks) {
  if (!param().stringDecoderZeroCopy ||
      !param().nimblePreserveDictionaryEncoding) {
    GTEST_SKIP()
        << "Dictionary path requires stringDecoderZeroCopy and nimblePreserveDict";
  }

  constexpr size_t kRowsPerWrite = 10000;
  constexpr size_t kValsPerChunk = 10;
  constexpr size_t kSpacing = 300; // spread non-nulls across read batches
  constexpr size_t kBatchSize = 1000;

  auto type =
      ROW({"select_col", "info"}, {BIGINT(), ROW({{"val"}}, {VARCHAR()})});
  rowType_ = asRowType(type);
  auto innerType = ROW({{"val"}}, {VARCHAR()});

  std::vector<size_t> chunkRows;
  chunkRows.reserve(kValsPerChunk);
  for (size_t i = 0; i < kValsPerChunk; ++i) {
    chunkRows.push_back(i * kSpacing);
  }
  std::vector<std::string> chunkAVals(kValsPerChunk, "chunk-a-val");
  std::vector<std::string> chunkBVals(kValsPerChunk, "chunk-b-val");

  auto batchA = makeSparseDictStructBatch(
      leafPool_.get(),
      type,
      innerType,
      /*globalRowBase=*/0,
      kRowsPerWrite,
      chunkRows,
      chunkAVals);
  auto batchB = makeSparseDictStructBatch(
      leafPool_.get(),
      type,
      innerType,
      /*globalRowBase=*/kRowsPerWrite,
      kRowsPerWrite,
      chunkRows,
      chunkBVals);

  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writer.write(batchA);
    writer.write(batchB);
    writer.close();
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());
  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));

  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(true);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  // Non-null iff the global row falls on a chunk's spacing grid; value depends
  // on which write (chunk) the row belongs to.
  auto expectedValue = [&](int64_t globalRow) -> std::optional<std::string> {
    const int64_t local = globalRow < static_cast<int64_t>(kRowsPerWrite)
        ? globalRow
        : globalRow - static_cast<int64_t>(kRowsPerWrite);
    if (local % static_cast<int64_t>(kSpacing) == 0 &&
        local / static_cast<int64_t>(kSpacing) <
            static_cast<int64_t>(kValsPerChunk)) {
      return globalRow < static_cast<int64_t>(kRowsPerWrite) ? "chunk-a-val"
                                                             : "chunk-b-val";
    }
    return std::nullopt;
  };

  VectorPtr result = velox::BaseVector::create(type, 0, leafPool_.get());
  size_t rowsSeen = 0;
  size_t totalNonNull = 0;
  while (rowReader->next(kBatchSize, result)) {
    auto* rowVector = result->as<RowVector>();
    auto info = rowVector->childAt(1);
    info->loadedVector();
    for (vector_size_t i = 0; i < info->size(); ++i) {
      const auto globalRow = static_cast<int64_t>(rowsSeen + i);
      const auto expected = expectedValue(globalRow);
      if (expected.has_value()) {
        ASSERT_FALSE(info->isNullAt(i)) << "row " << globalRow;
        ++totalNonNull;
        auto innerRow = info->wrappedVector()->as<RowVector>();
        auto wrappedIdx = info->wrappedIndex(i);
        auto val = innerRow->childAt(0)->as<SimpleVector<velox::StringView>>();
        ASSERT_FALSE(val->isNullAt(wrappedIdx)) << "row " << globalRow;
        EXPECT_EQ(val->valueAt(wrappedIdx).str(), *expected)
            << "row " << globalRow;
      } else {
        ASSERT_TRUE(info->isNullAt(i)) << "row " << globalRow;
      }
    }
    rowsSeen += rowVector->size();
  }

  EXPECT_EQ(rowsSeen, 2 * kRowsPerWrite);
  EXPECT_EQ(totalNonNull, 2 * kValsPerChunk);
}

// Gap-between-chunks layout where the second chunk is non-dictionary (distinct
// padded values -> Trivial), forcing the reactive dictionary->flat abandon on
// crossing into it. Covers the all-null gap/tail load decision under abandon.
TEST_P(E2EFilterTest, dictionaryChunkGapWithAbandon) {
  if (!param().stringDecoderZeroCopy ||
      !param().nimblePreserveDictionaryEncoding) {
    GTEST_SKIP()
        << "Dictionary path requires stringDecoderZeroCopy and nimblePreserveDict";
  }

  constexpr size_t kRowsPerWrite = 10000;
  constexpr size_t kValsPerChunk = 10;
  constexpr size_t kSpacing = 300;
  constexpr size_t kBatchSize = 1000;

  auto type =
      ROW({"select_col", "info"}, {BIGINT(), ROW({{"val"}}, {VARCHAR()})});
  rowType_ = asRowType(type);
  auto innerType = ROW({{"val"}}, {VARCHAR()});

  std::vector<size_t> chunkRows;
  chunkRows.reserve(kValsPerChunk);
  for (size_t i = 0; i < kValsPerChunk; ++i) {
    chunkRows.push_back(i * kSpacing);
  }
  // Chunk A: dictionary-compatible constant. Chunk B: all distinct padded
  // strings -> Trivial (non-dictionary) -> abandon on cross into it.
  std::vector<std::string> chunkAVals(kValsPerChunk, "chunk-a-val");
  std::vector<std::string> chunkBVals;
  chunkBVals.reserve(kValsPerChunk);
  for (size_t i = 0; i < kValsPerChunk; ++i) {
    chunkBVals.push_back(fmt::format("chunk-b-unique-value-{:03d}-padding", i));
  }

  auto batchA = makeSparseDictStructBatch(
      leafPool_.get(),
      type,
      innerType,
      /*globalRowBase=*/0,
      kRowsPerWrite,
      chunkRows,
      chunkAVals);
  auto batchB = makeSparseDictStructBatch(
      leafPool_.get(),
      type,
      innerType,
      /*globalRowBase=*/kRowsPerWrite,
      kRowsPerWrite,
      chunkRows,
      chunkBVals);

  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writer.write(batchA);
    writer.write(batchB);
    writer.close();
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());
  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));

  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(true);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  auto expectedValue = [&](int64_t globalRow) -> std::optional<std::string> {
    if (globalRow < static_cast<int64_t>(kRowsPerWrite)) {
      if (globalRow % static_cast<int64_t>(kSpacing) == 0 &&
          globalRow / static_cast<int64_t>(kSpacing) <
              static_cast<int64_t>(kValsPerChunk)) {
        return std::string("chunk-a-val");
      }
      return std::nullopt;
    }
    const int64_t local = globalRow - static_cast<int64_t>(kRowsPerWrite);
    if (local % static_cast<int64_t>(kSpacing) == 0 &&
        local / static_cast<int64_t>(kSpacing) <
            static_cast<int64_t>(kValsPerChunk)) {
      return chunkBVals[local / static_cast<int64_t>(kSpacing)];
    }
    return std::nullopt;
  };

  VectorPtr result = velox::BaseVector::create(type, 0, leafPool_.get());
  size_t rowsSeen = 0;
  size_t totalNonNull = 0;
  while (rowReader->next(kBatchSize, result)) {
    auto* rowVector = result->as<RowVector>();
    auto info = rowVector->childAt(1);
    info->loadedVector();
    for (vector_size_t i = 0; i < info->size(); ++i) {
      const auto globalRow = static_cast<int64_t>(rowsSeen + i);
      const auto expected = expectedValue(globalRow);
      if (expected.has_value()) {
        ASSERT_FALSE(info->isNullAt(i)) << "row " << globalRow;
        ++totalNonNull;
        auto innerRow = info->wrappedVector()->as<RowVector>();
        auto wrappedIdx = info->wrappedIndex(i);
        auto val = innerRow->childAt(0)->as<SimpleVector<velox::StringView>>();
        ASSERT_FALSE(val->isNullAt(wrappedIdx)) << "row " << globalRow;
        EXPECT_EQ(val->valueAt(wrappedIdx).str(), *expected)
            << "row " << globalRow;
      } else {
        ASSERT_TRUE(info->isNullAt(i)) << "row " << globalRow;
      }
    }
    rowsSeen += rowVector->size();
  }

  EXPECT_EQ(rowsSeen, 2 * kRowsPerWrite);
  EXPECT_EQ(totalNonNull, 2 * kValsPerChunk);
}

// Exercises the reactive fallback path (dictionary → flat mid-batch) with an
// active filter on the string column. The test creates alternating dictionary
// and trivial chunks. When the reader encounters a trivial chunk mid-read, it
// must:
// 1. Correctly expand already-filtered dictionary indices to flat StringViews
//    via expandDictionaryToFlat().
// 2. Continue reading remaining rows as flat strings with the filter still
//    applied.
// 3. Produce correct filtered output combining both phases.
TEST_P(E2EFilterTest, crossChunkEncodingChangeWithFilter) {
  auto type = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});
  rowType_ = asRowType(type);

  const size_t kRowsPerBatch = 200;
  const int kNumBatches = 8;
  std::vector<RowVectorPtr> batches;

  // Accumulate all expected string values for verification.
  std::vector<std::string> allStringVals;

  for (int batchIdx = 0; batchIdx < kNumBatches; ++batchIdx) {
    std::vector<std::string> stringVals;
    std::vector<int64_t> longVals;
    stringVals.reserve(kRowsPerBatch);
    longVals.reserve(kRowsPerBatch);

    if (batchIdx % 2 == 0) {
      // Even batches: repetitive → dictionary encoding.
      for (size_t i = 0; i < kRowsPerBatch; ++i) {
        stringVals.push_back(fmt::format("DICT_VAL_{}", i % 5));
        longVals.push_back(batchIdx * kRowsPerBatch + i);
      }
    } else {
      // Odd batches: unique → trivial encoding.
      for (size_t i = 0; i < kRowsPerBatch; ++i) {
        stringVals.push_back(
            fmt::format("UNIQUE_VAL_BATCH{}_ROW{}_EXTRA_PADDING", batchIdx, i));
        longVals.push_back(batchIdx * kRowsPerBatch + i);
      }
    }

    allStringVals.insert(
        allStringVals.end(), stringVals.begin(), stringVals.end());

    auto stringVector =
        velox::test::VectorMaker(leafPool_.get())
            .flatVector<velox::StringView>(kRowsPerBatch, [&](auto row) {
              return velox::StringView(stringVals[row]);
            });
    auto longVector = velox::test::VectorMaker(leafPool_.get())
                          .flatVector<int64_t>(kRowsPerBatch, [&](auto row) {
                            return longVals[row];
                          });

    batches.push_back(
        std::make_shared<RowVector>(
            leafPool_.get(),
            type,
            nullptr,
            kRowsPerBatch,
            std::vector<VectorPtr>{stringVector, longVector}));
  }

  // Write with per-batch chunking, single stripe.
  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    for (auto& batch : batches) {
      writer.write(batch);
    }
    writer.close();
  }

  // Filter: accept "DICT_VAL_0" (from dictionary chunks) and anything
  // starting with "UNIQUE_VAL_BATCH1" (from trivial chunk 1). Use BytesRange
  // to catch the unique values: ["UNIQUE_VAL_BATCH1", "UNIQUE_VAL_BATCH1~").
  // We test with BytesValues for the dictionary value and verify correctness
  // across the encoding transition.
  std::vector<std::string> accepted = {"DICT_VAL_0"};
  // Also include one unique value to verify the flat path works with filter.
  accepted.push_back("UNIQUE_VAL_BATCH1_ROW0_EXTRA_PADDING");
  std::set<std::string> acceptedSet(accepted.begin(), accepted.end());

  // Compute expected count.
  size_t expectedRows = 0;
  for (const auto& val : allStringVals) {
    if (acceptedSet.count(val)) {
      ++expectedRows;
    }
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());

  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  scanSpec->childByName("string_val")
      ->setFilter(
          std::make_unique<velox::common::BytesValues>(accepted, false));
  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(param().stringDecoderZeroCopy);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr result = velox::BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(1000, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();

    velox::DecodedVector decodedString(*rowResult->childAt(0)->loadedVector());
    for (size_t i = 0; i < rowResult->size(); ++i) {
      auto str = decodedString.valueAt<velox::StringView>(i).str();
      ASSERT_TRUE(acceptedSet.count(str))
          << "Unexpected value passed filter: " << str;
    }
  }

  EXPECT_EQ(totalRows, expectedRows);
}

// Companion to filterOnlyDictAbandonDropsFilter with nulls in the dictionary
// (filter) chunk: exercises the null-realignment branch of
// filterDictionaryIndices on the abandon path — a non-null-accepting filter
// must drop the null rows while filtering the dict portion before it is
// expanded to flat.
TEST_P(E2EFilterTest, crossChunkAbandonWithNullsAndFilter) {
  if (!param().stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  auto type = ROW({"filter_col", "select_col"}, {VARCHAR(), VARCHAR()});
  rowType_ = asRowType(type);

  velox::test::VectorMaker maker(leafPool_.get());

  // Chunk 1: low cardinality with periodic nulls → nullable dictionary.
  constexpr size_t kChunk1Rows = 500;
  constexpr size_t kNullEvery = 7;
  std::vector<std::string> filter1Storage(kChunk1Rows);
  std::vector<std::optional<velox::StringView>> filter1(kChunk1Rows);
  std::vector<std::string> select1(kChunk1Rows);
  for (size_t i = 0; i < kChunk1Rows; ++i) {
    select1[i] = fmt::format("VAL_{}", i % 5);
    if (i % kNullEvery == 0) {
      filter1[i] = std::nullopt;
    } else {
      filter1Storage[i] = fmt::format("EVENT_{}", i % 10);
      filter1[i] = velox::StringView(filter1Storage[i]);
    }
  }
  auto chunk1 = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kChunk1Rows,
      std::vector<VectorPtr>{
          maker.flatVectorNullable<velox::StringView>(filter1),
          maker.flatVector<velox::StringView>(kChunk1Rows, [&](auto row) {
            return velox::StringView(select1[row]);
          })});

  // Chunk 2: all unique → trivial encoding, forcing abandon.
  constexpr size_t kChunk2Rows = 300;
  std::vector<std::string> filter2(kChunk2Rows);
  std::vector<std::string> select2(kChunk2Rows);
  for (size_t i = 0; i < kChunk2Rows; ++i) {
    filter2[i] =
        fmt::format("UNIQUE_EVENT_{}_padding_to_avoid_inline_string", i);
    select2[i] = fmt::format("UNIQUE_VAL_{}", i);
  }
  auto chunk2 = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kChunk2Rows,
      std::vector<VectorPtr>{
          maker.flatVector<velox::StringView>(
              kChunk2Rows,
              [&](auto row) { return velox::StringView(filter2[row]); }),
          maker.flatVector<velox::StringView>(kChunk2Rows, [&](auto row) {
            return velox::StringView(select2[row]);
          })});

  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writer.write(chunk1);
    writer.write(chunk2);
    writer.close();
  }

  // Non-null-accepting filter: null rows must be dropped, and only matching
  // non-null dict rows from chunk 1 survive.
  std::vector<std::string> accepted = {"EVENT_3", "EVENT_7"};
  std::set<std::string> acceptedSet(accepted.begin(), accepted.end());
  size_t expectedRows = 0;
  for (size_t i = 0; i < kChunk1Rows; ++i) {
    if (filter1[i] && acceptedSet.count(filter1[i]->str())) {
      ++expectedRows;
    }
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());
  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));

  auto outputType = ROW({"select_col"}, {VARCHAR()});
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outputType);
  scanSpec->getOrCreateChild(velox::common::Subfield("filter_col"))
      ->setFilter(
          std::make_unique<velox::common::BytesValues>(accepted, false));

  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(true);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr result = velox::BaseVector::create(outputType, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(800, result)) {
    totalRows += result->as<RowVector>()->size();
  }

  EXPECT_EQ(totalRows, expectedRows)
      << "Nullable dict abandon path dropped or miscounted filtered rows";
}

// Projected variant of filterOnlyDictAbandonDropsFilter: the dictionary/filter
// column is also read out, so the abandon path must produce correctly filtered
// values (not just the right count) once the dict portion is expanded to flat.
TEST_P(E2EFilterTest, crossChunkAbandonProjectedDictFilter) {
  if (!param().stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  auto type = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});
  rowType_ = asRowType(type);

  velox::test::VectorMaker maker(leafPool_.get());

  // Chunk 1: 10 distinct values → dictionary-encoded.
  constexpr size_t kChunk1Rows = 500;
  std::vector<std::string> string1(kChunk1Rows);
  for (size_t i = 0; i < kChunk1Rows; ++i) {
    string1[i] = fmt::format("EVENT_{}", i % 10);
  }
  auto chunk1 = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kChunk1Rows,
      std::vector<VectorPtr>{
          maker.flatVector<velox::StringView>(
              kChunk1Rows,
              [&](auto row) { return velox::StringView(string1[row]); }),
          maker.flatVector<int64_t>(kChunk1Rows, [&](auto row) {
            return static_cast<int64_t>(row);
          })});

  // Chunk 2: all unique → trivial encoding, forcing abandon.
  constexpr size_t kChunk2Rows = 300;
  std::vector<std::string> string2(kChunk2Rows);
  for (size_t i = 0; i < kChunk2Rows; ++i) {
    string2[i] = fmt::format("UNIQUE_{}_padding_to_avoid_inline_string", i);
  }
  auto chunk2 = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kChunk2Rows,
      std::vector<VectorPtr>{
          maker.flatVector<velox::StringView>(
              kChunk2Rows,
              [&](auto row) { return velox::StringView(string2[row]); }),
          maker.flatVector<int64_t>(kChunk2Rows, [&](auto row) {
            return static_cast<int64_t>(kChunk1Rows + row);
          })});

  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writer.write(chunk1);
    writer.write(chunk2);
    writer.close();
  }

  std::vector<std::string> accepted = {"EVENT_3", "EVENT_7"};
  std::set<std::string> acceptedSet(accepted.begin(), accepted.end());
  size_t expectedRows = 0;
  for (const auto& s : string1) {
    if (acceptedSet.count(s)) {
      ++expectedRows;
    }
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());
  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));

  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  scanSpec->childByName("string_val")
      ->setFilter(
          std::make_unique<velox::common::BytesValues>(accepted, false));

  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(true);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr result = velox::BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(800, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();
    velox::DecodedVector decodedString(*rowResult->childAt(0)->loadedVector());
    for (size_t i = 0; i < rowResult->size(); ++i) {
      auto str = decodedString.valueAt<velox::StringView>(i).str();
      ASSERT_TRUE(acceptedSet.count(str))
          << "Row leaked past filter on abandon path: " << str;
    }
  }

  EXPECT_EQ(totalRows, expectedRows)
      << "Projected dict abandon path dropped filter on dict portion";
}

// Projected + nullable variant: the dict/filter column has periodic nulls and
// is also read out. The non-null-accepting filter must drop nulls, filter the
// dict portion, and expand it to flat with the output null bitmap realigned.
TEST_P(E2EFilterTest, crossChunkAbandonProjectedNullableDictFilter) {
  if (!param().stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  auto type = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});
  rowType_ = asRowType(type);

  velox::test::VectorMaker maker(leafPool_.get());

  // Chunk 1: 10 distinct values with periodic nulls → nullable dictionary.
  constexpr size_t kChunk1Rows = 500;
  constexpr size_t kNullEvery = 7;
  std::vector<std::string> string1Storage(kChunk1Rows);
  std::vector<std::optional<velox::StringView>> string1(kChunk1Rows);
  for (size_t i = 0; i < kChunk1Rows; ++i) {
    if (i % kNullEvery == 0) {
      string1[i] = std::nullopt;
    } else {
      string1Storage[i] = fmt::format("EVENT_{}", i % 10);
      string1[i] = velox::StringView(string1Storage[i]);
    }
  }
  auto chunk1 = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kChunk1Rows,
      std::vector<VectorPtr>{
          maker.flatVectorNullable<velox::StringView>(string1),
          maker.flatVector<int64_t>(kChunk1Rows, [&](auto row) {
            return static_cast<int64_t>(row);
          })});

  // Chunk 2: all unique → trivial encoding, forcing abandon.
  constexpr size_t kChunk2Rows = 300;
  std::vector<std::string> string2(kChunk2Rows);
  for (size_t i = 0; i < kChunk2Rows; ++i) {
    string2[i] = fmt::format("UNIQUE_{}_padding_to_avoid_inline_string", i);
  }
  auto chunk2 = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kChunk2Rows,
      std::vector<VectorPtr>{
          maker.flatVector<velox::StringView>(
              kChunk2Rows,
              [&](auto row) { return velox::StringView(string2[row]); }),
          maker.flatVector<int64_t>(kChunk2Rows, [&](auto row) {
            return static_cast<int64_t>(kChunk1Rows + row);
          })});

  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writer.write(chunk1);
    writer.write(chunk2);
    writer.close();
  }

  std::vector<std::string> accepted = {"EVENT_3", "EVENT_7"};
  std::set<std::string> acceptedSet(accepted.begin(), accepted.end());
  size_t expectedRows = 0;
  for (size_t i = 0; i < kChunk1Rows; ++i) {
    if (string1[i] && acceptedSet.count(string1[i]->str())) {
      ++expectedRows;
    }
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());
  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));

  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  scanSpec->childByName("string_val")
      ->setFilter(
          std::make_unique<velox::common::BytesValues>(accepted, false));

  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(true);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr result = velox::BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(800, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();
    velox::DecodedVector decodedString(*rowResult->childAt(0)->loadedVector());
    for (size_t i = 0; i < rowResult->size(); ++i) {
      ASSERT_FALSE(decodedString.isNullAt(i))
          << "Non-null-accepting filter emitted a null row on abandon path";
      auto str = decodedString.valueAt<velox::StringView>(i).str();
      ASSERT_TRUE(acceptedSet.count(str))
          << "Row leaked past filter on abandon path: " << str;
    }
  }

  EXPECT_EQ(totalRows, expectedRows)
      << "Projected nullable dict abandon path dropped or miscounted rows";
}

// Sparse variant: a more-selective prior filter on `k` makes the dictionary
// column `str` read over a non-contiguous (sparse) row set. When that sparse
// read spans a dict chunk into a non-dict chunk, filterDictionaryIndices runs
// on the abandon path with isDense=false — exercising the sparse null source
// and the sparse rows[0, numValues_) mapping without subranging `rows`.
TEST_P(E2EFilterTest, crossChunkAbandonSparseDictFilter) {
  if (!param().stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  auto type = ROW({"k", "str"}, {BIGINT(), VARCHAR()});
  rowType_ = asRowType(type);

  velox::test::VectorMaker maker(leafPool_.get());

  // Chunk 1: `str` low cardinality → dictionary. `k = i % 11`, so the very
  // selective `k == 0` filter (~9%, applied before the ~12.5% `str` filter)
  // sparsifies `str`'s read over non-contiguous rows.
  constexpr size_t kChunk1Rows = 500;
  std::vector<std::string> str1(kChunk1Rows);
  for (size_t i = 0; i < kChunk1Rows; ++i) {
    str1[i] = fmt::format("EVENT_{}", i % 10);
  }
  auto chunk1 = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kChunk1Rows,
      std::vector<VectorPtr>{
          maker.flatVector<int64_t>(
              kChunk1Rows,
              [](auto row) { return static_cast<int64_t>(row % 11); }),
          maker.flatVector<velox::StringView>(kChunk1Rows, [&](auto row) {
            return velox::StringView(str1[row]);
          })});

  // Chunk 2: unique → trivial encoding, forcing abandon.
  constexpr size_t kChunk2Rows = 300;
  std::vector<std::string> str2(kChunk2Rows);
  for (size_t i = 0; i < kChunk2Rows; ++i) {
    str2[i] = fmt::format("UNIQUE_{}_padding_to_avoid_inline_string", i);
  }
  auto chunk2 = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kChunk2Rows,
      std::vector<VectorPtr>{
          maker.flatVector<int64_t>(
              kChunk2Rows,
              [](auto row) { return static_cast<int64_t>(row % 11); }),
          maker.flatVector<velox::StringView>(kChunk2Rows, [&](auto row) {
            return velox::StringView(str2[row]);
          })});

  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writer.write(chunk1);
    writer.write(chunk2);
    writer.close();
  }

  std::vector<std::string> accepted = {"EVENT_3", "EVENT_7"};
  std::set<std::string> acceptedSet(accepted.begin(), accepted.end());
  // Expected: chunk-1 rows with k == 0 (i % 11 == 0) whose str matches.
  size_t expectedRows = 0;
  for (size_t i = 0; i < kChunk1Rows; ++i) {
    if (i % 11 == 0 && acceptedSet.count(str1[i])) {
      ++expectedRows;
    }
  }
  ASSERT_GT(expectedRows, 0u);

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());
  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));

  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  scanSpec->childByName("k")->setFilter(
      std::make_unique<velox::common::BigintRange>(
          0, 0, /*nullAllowed=*/false));
  scanSpec->childByName("str")->setFilter(
      std::make_unique<velox::common::BytesValues>(accepted, false));

  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(true);
  rowReaderOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr result = velox::BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(800, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();
    velox::DecodedVector decodedStr(*rowResult->childAt(1)->loadedVector());
    for (size_t i = 0; i < rowResult->size(); ++i) {
      auto s = decodedStr.valueAt<velox::StringView>(i).str();
      ASSERT_TRUE(acceptedSet.count(s))
          << "Row leaked past filter on sparse abandon path: " << s;
    }
  }

  EXPECT_EQ(totalRows, expectedRows)
      << "Sparse dict abandon path dropped or miscounted filtered rows";
}

// Exercises the IsNull filter guard: when a null-accepting filter (e.g. IsNull)
// is combined with NullableEncoding wrapping Dictionary, the reader must bypass
// the dictionary path. NullableEncoding::materializeNullsForVisitor writes
// encoding-level nulls to the bitmap but doesn't call processNull() for
// individual null rows, so null-accepting filters would produce incorrect
// results if the dictionary path were used.
TEST_P(E2EFilterTest, isNullFilterWithNullableDictionary) {
  auto type = ROW({"string_val", "long_val"}, {VARCHAR(), BIGINT()});
  rowType_ = asRowType(type);

  const size_t kRowCount = 1000;
  std::vector<std::string> stringStorage;
  std::vector<std::optional<velox::StringView>> stringVals;
  std::vector<int64_t> longVals;
  stringStorage.reserve(kRowCount);
  stringVals.reserve(kRowCount);
  longVals.reserve(kRowCount);

  size_t expectedNullCount = 0;
  for (size_t i = 0; i < kRowCount; ++i) {
    longVals.push_back(i);
    if (i % 7 == 0) {
      // ~14% nulls.
      stringVals.push_back(std::nullopt);
      stringStorage.push_back("");
      ++expectedNullCount;
    } else {
      stringStorage.push_back(fmt::format("VAL_{}", i % 10));
      stringVals.push_back(velox::StringView(stringStorage.back()));
    }
  }

  auto stringVector = velox::test::VectorMaker(leafPool_.get())
                          .flatVectorNullable<velox::StringView>(stringVals);
  auto longVector = velox::test::VectorMaker(leafPool_.get())
                        .flatVector<int64_t>(
                            kRowCount, [&](auto row) { return longVals[row]; });

  auto batch = std::make_shared<RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      kRowCount,
      std::vector<VectorPtr>{stringVector, longVector});

  // Write with forced dictionary encoding. With nulls, the writer will produce
  // Nullable → Dictionary.
  {
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.encodingLayoutTree.emplace(
        buildForcedDictionaryEncodingLayoutTree());
    Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writer.write(batch);
    writer.close();
  }

  auto input = std::make_unique<velox::dwio::common::BufferedInput>(
      std::make_shared<velox::InMemoryReadFile>(sinkData_),
      *leafPool_,
      velox::dwio::common::MetricsLog::voidLog());

  velox::dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto reader = makeReader(readerOpts, std::move(input));
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
  scanSpec->addAllChildFields(*type);
  scanSpec->childByName("string_val")
      ->setFilter(std::make_unique<velox::common::IsNull>());
  velox::dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setStringDecoderZeroCopy(param().stringDecoderZeroCopy);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  VectorPtr result = velox::BaseVector::create(type, 0, leafPool_.get());
  size_t totalRows = 0;
  while (rowReader->next(kRowCount, result)) {
    auto* rowResult = result->as<RowVector>();
    ASSERT_NE(rowResult, nullptr);
    totalRows += rowResult->size();

    // Every returned row should have a null string value.
    auto* stringChild = rowResult->childAt(0).get();
    for (size_t i = 0; i < rowResult->size(); ++i) {
      ASSERT_TRUE(stringChild->isNullAt(i))
          << "Expected null at result row " << i;
    }
  }

  EXPECT_EQ(totalRows, expectedNullCount);
}

} // namespace facebook::nimble
