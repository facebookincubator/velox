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

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/common/base/ConcurrentRuntimeStatWriter.h"
#include "velox/common/base/RandomUtil.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/CachedBufferedInput.h"
#include "velox/dwio/common/DirectBufferedInput.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/common/tests/utils/E2EFilterTestBase.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/PrefixEncoding.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/type/Filter.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble::test {

using namespace velox;
using namespace velox::common;
using namespace velox::dwio::common;
using index::ClusterIndexConfigBuilder;

namespace {

// Builds a name for prefix encoding with optional restart interval suffix.
std::string makePrefixEncodingName(
    std::string_view baseName,
    std::optional<uint32_t> prefixRestartInterval) {
  std::string name(baseName);
  if (prefixRestartInterval.has_value()) {
    name += "_restart" + std::to_string(prefixRestartInterval.value());
  }
  return name;
}

// Creates an EncodingLayout for index tests with the given parameters.
EncodingLayout makeIndexEncodingLayout(
    EncodingType encodingType,
    CompressionType compressionType,
    std::optional<uint32_t> prefixRestartInterval) {
  std::vector<std::optional<const EncodingLayout>> children;
  if (encodingType == EncodingType::Trivial) {
    // Trivial encoding for string data needs a child encoding for lengths.
    children.emplace_back(
        EncodingLayout{
            EncodingType::Trivial, {}, CompressionType::Uncompressed});
  }
  EncodingLayout::Config config;
  if (prefixRestartInterval.has_value()) {
    config = EncodingLayout::Config{
        {{std::string(PrefixEncoding::kRestartIntervalConfigKey),
          std::to_string(prefixRestartInterval.value())}}};
  }
  return EncodingLayout{
      encodingType, std::move(config), compressionType, std::move(children)};
}

} // namespace

// Defines the encoding layout configuration for index tests.
struct IndexEncodingParam {
  std::string name;
  EncodingType encodingType;
  CompressionType compressionType;
  SortOrder sortOrder;
  std::optional<uint32_t> prefixRestartInterval;
  bool enableChunkIndex{false};
  bool pinMetadata{false};
  bool enableCache{false};

  static IndexEncodingParam prefix(
      std::optional<uint32_t> prefixRestartInterval = std::nullopt,
      bool enableChunkIndex = false,
      bool pinMetadata = false,
      bool enableCache = false) {
    return IndexEncodingParam{
        makePrefixEncodingName("Prefix", prefixRestartInterval) +
            (enableChunkIndex ? "_ChunkIndex" : "") +
            (pinMetadata ? "_PinMetadata" : "") + (enableCache ? "_Cache" : ""),
        EncodingType::Prefix,
        CompressionType::Uncompressed,
        SortOrder{.ascending = true},
        prefixRestartInterval,
        enableChunkIndex,
        pinMetadata,
        enableCache};
  }

  static IndexEncodingParam trivialZstd(
      bool enableChunkIndex = false,
      bool pinMetadata = false,
      bool enableCache = false) {
    return IndexEncodingParam{
        std::string("TrivialZstd") + (enableChunkIndex ? "_ChunkIndex" : "") +
            (pinMetadata ? "_PinMetadata" : "") + (enableCache ? "_Cache" : ""),
        EncodingType::Trivial,
        CompressionType::Zstd,
        SortOrder{.ascending = true},
        std::nullopt,
        enableChunkIndex,
        pinMetadata,
        enableCache};
  }

  static IndexEncodingParam prefixDesc(
      std::optional<uint32_t> prefixRestartInterval = std::nullopt,
      bool enableChunkIndex = false,
      bool pinMetadata = false,
      bool enableCache = false) {
    return IndexEncodingParam{
        makePrefixEncodingName("PrefixDesc", prefixRestartInterval) +
            (enableChunkIndex ? "_ChunkIndex" : "") +
            (pinMetadata ? "_PinMetadata" : "") + (enableCache ? "_Cache" : ""),
        EncodingType::Prefix,
        CompressionType::Uncompressed,
        SortOrder{.ascending = false},
        prefixRestartInterval,
        enableChunkIndex,
        pinMetadata,
        enableCache};
  }

  static IndexEncodingParam trivialZstdDesc(
      bool enableChunkIndex = false,
      bool pinMetadata = false,
      bool enableCache = false) {
    return IndexEncodingParam{
        std::string("TrivialZstdDesc") +
            (enableChunkIndex ? "_ChunkIndex" : "") +
            (pinMetadata ? "_PinMetadata" : "") + (enableCache ? "_Cache" : ""),
        EncodingType::Trivial,
        CompressionType::Zstd,
        SortOrder{.ascending = false},
        std::nullopt,
        enableChunkIndex,
        pinMetadata,
        enableCache};
  }

  EncodingLayout makeEncodingLayout() const {
    return makeIndexEncodingLayout(
        encodingType, compressionType, prefixRestartInterval);
  }
};

class E2EIndexTestBase : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
    registerSelectiveNimbleReaderFactory();
  }

  void SetUp() override {
    rootPool_ = memory::memoryManager()->addRootPool("E2EIndexTest");
    leafPool_ = rootPool_->addLeafChild("E2EIndexTestLeaf");
    vectorMaker_ = std::make_unique<velox::test::VectorMaker>(leafPool_.get());
    dataIoStats_ = std::make_shared<io::IoStatistics>();
    metadataIoStats_ = std::make_shared<io::IoStatistics>();
    indexIoStats_ = std::make_shared<io::IoStatistics>();
    scanTracker_ =
        std::make_shared<cache::ScanTracker>("testTracker", nullptr, 256 << 10);
    ioExecutor_ = std::make_unique<folly::CPUThreadPoolExecutor>(10);
  }

  void setUpCache() {
    allocator_ = std::make_shared<memory::MallocAllocator>(
        memory::MemoryAllocator::Options{
            .capacity = 512 << 20, .reservationByteLimit = 0});
    cache_ = cache::AsyncDataCache::create(allocator_.get());
  }

  void tearDownCache() {
    ioExecutor_.reset();
    if (cache_ != nullptr) {
      cache_->shutdown();
    }
    cache_.reset();
    allocator_.reset();
  }

  void TearDown() override {
    tearDownCache();
    vectorMaker_.reset();
    leafPool_.reset();
    rootPool_.reset();
  }

  // Recursively adds child specs for nested types.
  void addChildSpecs(ScanSpec* spec, const TypePtr& type) {
    switch (type->kind()) {
      case TypeKind::ROW: {
        auto rowType = std::dynamic_pointer_cast<const velox::RowType>(type);
        for (size_t i = 0; i < rowType->size(); ++i) {
          auto* childSpec = spec->addField(
              rowType->nameOf(i), static_cast<column_index_t>(i));
          addChildSpecs(childSpec, rowType->childAt(i));
        }
        break;
      }
      case TypeKind::ARRAY: {
        auto* childSpec = spec->addField(ScanSpec::kArrayElementsFieldName, 0);
        addChildSpecs(childSpec, type->childAt(0));
        break;
      }
      case TypeKind::MAP: {
        auto* keySpec = spec->addField(ScanSpec::kMapKeysFieldName, 0);
        addChildSpecs(keySpec, type->childAt(0));
        auto* valueSpec = spec->addField(ScanSpec::kMapValuesFieldName, 1);
        addChildSpecs(valueSpec, type->childAt(1));
        break;
      }
      default:
        // Leaf types (primitives) don't need child specs.
        break;
    }
  }

  // Creates a ScanSpec with the given filters.
  std::unique_ptr<ScanSpec> createScanSpec(
      const RowTypePtr& rowType,
      const std::unordered_map<std::string, std::unique_ptr<Filter>>& filters) {
    auto scanSpec = std::make_unique<ScanSpec>("root");
    for (size_t i = 0; i < rowType->size(); ++i) {
      auto* childSpec = scanSpec->addField(
          rowType->nameOf(i), static_cast<column_index_t>(i));
      auto it = filters.find(rowType->nameOf(i));
      if (it != filters.end()) {
        childSpec->setFilter(it->second->clone());
      }
      // Recursively add child specs for nested types.
      addChildSpecs(childSpec, rowType->childAt(i));
    }
    return scanSpec;
  }

  // Writes data to an in-memory file with the given index columns.
  // indexColumns must not be empty.
  // Uses small stripe size (16KB) and small chunk size (4KB) to generate
  // multiple stripes and write groups for better test coverage.
  void writeData(
      const std::vector<RowVectorPtr>& batches,
      const std::vector<std::string>& indexColumns,
      std::optional<EncodingLayout> encodingLayout = std::nullopt,
      SortOrder sortOrder = SortOrder{.ascending = true},
      bool noDuplicateKey = true,
      bool enableChunkIndex = false) {
    ASSERT_FALSE(indexColumns.empty()) << "indexColumns must not be empty";

    sinkData_.clear();
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);

    WriterOptions options;
    options.enableChunking = true;
    options.enableChunkIndex = enableChunkIndex;
    auto clusterIndexConfig =
        ClusterIndexConfigBuilder{}
            .withKeyColumns(indexColumns)
            .withSortOrders(
                std::vector<SortOrder>(indexColumns.size(), sortOrder))
            .withEnforceKeyOrder(true)
            .withNoDuplicateKey(noDuplicateKey);
    if (encodingLayout.has_value()) {
      clusterIndexConfig.withEncodingLayout(std::move(encodingLayout).value());
    }
    options.clusterIndexConfig = clusterIndexConfig.build();

    // Use small stripe and chunk sizes to generate multiple stripes and write
    // groups for better index test coverage.
    constexpr uint64_t kSmallStripeSize = 16 << 10; // 16KB
    constexpr uint64_t kSmallChunkSize = 4 << 10; // 4KB
    options.flushPolicyFactory = [=]() {
      return std::make_unique<LambdaFlushPolicy>(
          [](const StripeProgress& progress) {
            return progress.stripeRawSize >= kSmallStripeSize;
          },
          [](const StripeProgress& progress) {
            return progress.stripeRawSize >= kSmallChunkSize;
          });
    };
    // Set minimum stream chunk size to allow small chunks.
    options.minStreamChunkRawSize = kSmallChunkSize;

    auto rowType = asRowType(batches[0]->type());
    Writer writer(
        rowType, std::move(writeFile), *rootPool_, std::move(options));
    for (const auto& batch : batches) {
      writer.write(batch);
    }
    writer.close();
  }

  // Creates a reader for the written data with optional random skip.
  virtual std::unique_ptr<Reader> createReader(
      double sampleRate = 1.0,
      uint32_t randomSeed = 0) {
    auto readFile =
        std::make_shared<InMemoryReadFile>(std::string_view(sinkData_));
    dwio::common::ReaderOptions readerOptions(leafPool_.get());
    readerOptions.setDataIoStats(dataIoStats_);
    readerOptions.setMetadataIoStats(metadataIoStats_);
    readerOptions.setIndexIoStats(indexIoStats_);
    readerOptions.setFileFormat(FileFormat::NIMBLE);
    readerOptions.setLoadClusterIndex(true);
    setUpReaderOptions(readerOptions);

    // Set up random skip if sample rate is less than 1.0.
    if (sampleRate < 1.0) {
      random::setSeed(randomSeed);
      auto randomSkip = std::make_shared<random::RandomSkipTracker>(sampleRate);
      readerOptions.setRandomSkip(randomSkip);
    }

    std::unique_ptr<BufferedInput> input;
    auto& ids = fileIds();
    const auto readerId = readerIdCounter_++;
    StringIdLease fileId(ids, fmt::format("testFile_{}", readerId));
    StringIdLease groupId(ids, fmt::format("testGroup_{}", readerId));
    io::ReaderOptions ioReaderOpts(&readerOptions.memoryPool());
    ioReaderOpts.setDataIoStats(dataIoStats_);
    ioReaderOpts.setMetadataIoStats(metadataIoStats_);
    if (cache_ != nullptr) {
      input = std::make_unique<CachedBufferedInput>(
          readFile,
          MetricsLog::voidLog(),
          std::move(fileId),
          cache_.get(),
          scanTracker_,
          std::move(groupId),
          dataIoStats_,
          nullptr,
          ioExecutor_.get(),
          ioReaderOpts);
    } else {
      input = std::make_unique<DirectBufferedInput>(
          readFile,
          MetricsLog::voidLog(),
          std::move(fileId),
          scanTracker_,
          std::move(groupId),
          dataIoStats_,
          nullptr,
          ioExecutor_.get(),
          ioReaderOpts);
    }
    auto factory = getReaderFactory(FileFormat::NIMBLE);
    return factory->createReader(std::move(input), readerOptions);
  }

  // Override to customize reader options (e.g., pinMetadata).
  virtual void setUpReaderOptions(dwio::common::ReaderOptions& /*opts*/) {}

  // Reads all rows with the given filters and optionally disables index.
  std::vector<RowVectorPtr> readWithFilters(
      Reader& reader,
      const RowTypePtr& requestedType,
      const std::unordered_map<std::string, std::unique_ptr<Filter>>& filters,
      bool indexEnabled,
      int64_t& numIndexFilterConversions) {
    RowReaderOptions rowReaderOptions;
    rowReaderOptions.setRequestedType(requestedType);

    // Clone filters because createScanSpec modifies them.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filtersCopy;
    for (const auto& [name, filter] : filters) {
      filtersCopy[name] = filter->clone();
    }
    auto scanSpec = createScanSpec(requestedType, filtersCopy);
    rowReaderOptions.setScanSpec(std::move(scanSpec));

    // Set index enabled option.
    rowReaderOptions.setIndexEnabled(indexEnabled);

    // Set up a stat writer to capture runtime stats.
    ConcurrentRuntimeStatWriter statWriter;
    RuntimeStatWriterScopeGuard guard(&statWriter);

    auto rowReader = reader.createRowReader(rowReaderOptions);

    std::vector<RowVectorPtr> results;
    VectorPtr result = BaseVector::create(requestedType, 1, leafPool_.get());
    while (rowReader->next(1'000, result) > 0) {
      // Materialize lazy loaded vectors before storing the result.
      result->loadedVector();
      results.push_back(std::dynamic_pointer_cast<RowVector>(result));
    }

    // Get the number of index filter conversions from runtime stats captured
    // by our writer.
    const auto stats = statWriter.runtimeStats();
    const auto it =
        stats.find(std::string(RowReader::kNumIndexFilterConversions));
    numIndexFilterConversions = it != stats.end() ? it->second.sum : 0;

    return results;
  }

  // Compares two sets of results for equality.
  void verifyResultsEqual(
      const std::vector<RowVectorPtr>& withIndex,
      const std::vector<RowVectorPtr>& withoutIndex) {
    // Flatten both results into single vectors for comparison.
    std::vector<VectorPtr> withIndexFlat;
    std::vector<VectorPtr> withoutIndexFlat;

    size_t totalRowsWithIndex = 0;
    size_t totalRowsWithoutIndex = 0;

    for (const auto& batch : withIndex) {
      totalRowsWithIndex += batch->size();
      withIndexFlat.push_back(batch);
    }
    for (const auto& batch : withoutIndex) {
      totalRowsWithoutIndex += batch->size();
      withoutIndexFlat.push_back(batch);
    }

    ASSERT_EQ(totalRowsWithIndex, totalRowsWithoutIndex)
        << "Row count mismatch: with index = " << totalRowsWithIndex
        << ", without index = " << totalRowsWithoutIndex;

    // Compare row by row.
    size_t idxWithIndex = 0;
    size_t rowWithIndex = 0;
    size_t idxWithoutIndex = 0;
    size_t rowWithoutIndex = 0;

    for (size_t i = 0; i < totalRowsWithIndex; ++i) {
      // Advance to next batch if needed.
      while (rowWithIndex >= withIndex[idxWithIndex]->size()) {
        rowWithIndex = 0;
        ++idxWithIndex;
      }
      while (rowWithoutIndex >= withoutIndex[idxWithoutIndex]->size()) {
        rowWithoutIndex = 0;
        ++idxWithoutIndex;
      }

      const auto& batchWithIndex = withIndex[idxWithIndex];
      const auto& batchWithoutIndex = withoutIndex[idxWithoutIndex];

      ASSERT_TRUE(batchWithIndex->equalValueAt(
          batchWithoutIndex.get(), rowWithIndex, rowWithoutIndex))
          << "Row " << i << " differs between with-index and without-index";

      ++rowWithIndex;
      ++rowWithoutIndex;
    }
  }

  // Generates sorted key column data.
  template <typename T>
  VectorPtr makeSortedFlatVector(size_t size, T startValue, T step) {
    std::vector<T> values(size);
    for (size_t i = 0; i < size; ++i) {
      values[i] = startValue + static_cast<T>(i) * step;
    }
    return vectorMaker_->flatVector(values);
  }

  // Generates a complex column using VectorFuzzer.
  VectorPtr
  makeFuzzedComplexVector(const TypePtr& type, size_t size, size_t seed = 0) {
    VectorFuzzer::Options options;
    options.vectorSize = size;
    options.nullRatio = 0.1;
    VectorFuzzer fuzzer(options, leafPool_.get(), seed);
    return fuzzer.fuzz(type);
  }

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
  std::string sinkData_;
  uint32_t readerIdCounter_{0};

  // Cache infrastructure (only initialized when enableCache is true).
  std::shared_ptr<memory::MallocAllocator> allocator_;
  std::shared_ptr<cache::AsyncDataCache> cache_;
  std::shared_ptr<io::IoStatistics> dataIoStats_;
  std::shared_ptr<io::IoStatistics> metadataIoStats_;
  std::shared_ptr<io::IoStatistics> indexIoStats_;
  std::shared_ptr<cache::ScanTracker> scanTracker_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> ioExecutor_;
};

class E2EIndexTest : public E2EIndexTestBase,
                     public ::testing::WithParamInterface<IndexEncodingParam> {
 protected:
  void SetUp() override {
    E2EIndexTestBase::SetUp();
    if (GetParam().enableCache) {
      setUpCache();
    }
  }

  void setUpReaderOptions(dwio::common::ReaderOptions& opts) override {
    opts.setPinMetadata(GetParam().pinMetadata);
  }

  // Returns the encoding layout from the test parameter.
  EncodingLayout getEncodingLayout() const {
    return GetParam().makeEncodingLayout();
  }

  // Returns true if the sort order is ascending.
  bool isAscending() const {
    return GetParam().sortOrder.ascending;
  }

  // Generates sorted key column data based on the test parameter's sort order.
  template <typename T>
  VectorPtr makeSortedFlatVector(size_t size, T startValue, T step) {
    std::vector<T> values(size);
    if (isAscending()) {
      for (size_t i = 0; i < size; ++i) {
        values[i] = startValue + static_cast<T>(i) * step;
      }
    } else {
      // For descending order, generate values in reverse.
      for (size_t i = 0; i < size; ++i) {
        values[i] = startValue + static_cast<T>(size - 1 - i) * step;
      }
    }
    return vectorMaker_->flatVector(values);
  }

  // Generates sorted key data using the test parameter's sort order.
  // Uses the shared generateStrictSortedVector() or generateSortedVector()
  // utility from E2EFilterTestBase.
  // keyType: The type of the key column (BIGINT, DOUBLE, REAL, TIMESTAMP,
  //          VARCHAR)
  // numRows: Total number of rows to generate
  // rowsPerBatch: Number of rows per batch
  // startValue: The starting value for the monotonically increasing sequence
  // withDuplicates: If true, generates 1-3 random duplicates for each key
  std::vector<RowVectorPtr> generateKeyData(
      const TypePtr& keyType,
      size_t numRows,
      size_t rowsPerBatch,
      int64_t startValue = 0,
      bool withDuplicates = false) {
    const bool ascending = isAscending();
    std::vector<RowVectorPtr> batches;
    const size_t numBatches = numRows / rowsPerBatch;

    for (size_t batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
      const size_t globalRowOffset = batchIdx * rowsPerBatch;
      VectorPtr keyVector;
      if (withDuplicates) {
        keyVector = dwio::common::E2EFilterTestBase::generateSortedVector(
            keyType,
            rowsPerBatch,
            globalRowOffset,
            startValue,
            numRows,
            ascending,
            leafPool_.get());
      } else {
        keyVector = dwio::common::E2EFilterTestBase::generateStrictSortedVector(
            keyType,
            rowsPerBatch,
            globalRowOffset,
            startValue,
            numRows,
            ascending,
            leafPool_.get());
      }
      auto dataVector = makeFuzzedComplexVector(
          ARRAY(INTEGER()), rowsPerBatch, batchIdx * 12345);
      auto nestedVector = makeFuzzedComplexVector(
          ROW({{"nested", MAP(INTEGER(), VARCHAR())}}),
          rowsPerBatch,
          batchIdx * 54321);
      batches.push_back(vectorMaker_->rowVector(
          {"key", "data", "nested"}, {keyVector, dataVector, nestedVector}));
    }
    return batches;
  }

  // Writes data using the test parameter's encoding layout and sort order.
  void writeDataWithParam(
      const std::vector<RowVectorPtr>& batches,
      const std::vector<std::string>& indexColumns,
      bool noDuplicateKey = true) {
    writeData(
        batches,
        indexColumns,
        getEncodingLayout(),
        GetParam().sortOrder,
        noDuplicateKey,
        GetParam().enableChunkIndex);
  }
};

// Test with single bigint key column covering various boundary conditions.
TEST_P(E2EIndexTest, singleBigintKey) {
  // Key values range from kMinKey to kMinKey + kNumRows - 1 (for unique keys).
  // With duplicates, there are fewer unique keys but the same total row count.
  constexpr int64_t kMinKey = 1'000;
  constexpr size_t kNumRows = 10'000;
  constexpr size_t kRowsPerBatch = 1'000;
  const int64_t kMaxKey = kMinKey + kNumRows - 1;

  struct TestCase {
    std::string name;
    int64_t filterLower;
    int64_t filterUpper;
    bool withDuplicates;

    std::string debugString() const {
      return fmt::format(
          "name={}, filterLower={}, filterUpper={}, withDuplicates={}",
          name,
          filterLower,
          filterUpper,
          withDuplicates);
    }
  };

  const int64_t kNumericMin = std::numeric_limits<int64_t>::min();
  const int64_t kNumericMax = std::numeric_limits<int64_t>::max();

  // Base filter configurations to test.
  std::vector<std::tuple<std::string, int64_t, int64_t>> filterConfigs = {
      {"pointLookupMiddle", 5'000, 5'000},
      {"pointLookupAtMin", kMinKey, kMinKey},
      {"pointLookupAtMax", kMaxKey, kMaxKey},
      {"rangeMiddle", 3'000, 7'000},
      {"rangeFromMin", kMinKey, 5'000},
      {"rangeToMax", 5'000, kMaxKey},
      {"fullRange", kMinKey, kMaxKey},
      {"lowerBelowMin", 0, 5'000},
      {"upperAboveMax", 5'000, kMaxKey + 1'000},
      {"bothBoundsAboveMax", kMaxKey + 100, kMaxKey + 200},
      {"bothBoundsBelowMin", 0, 500},
      {"pointLookupBelowMin", 500, 500},
      {"pointLookupAboveMax", kMaxKey + 100, kMaxKey + 100},
      // Numeric limits test cases.
      {"lowerAtNumericMin", kNumericMin, 5'000},
      {"upperAtNumericMax", 5'000, kNumericMax},
      {"fullNumericRange", kNumericMin, kNumericMax},
  };

  // Generate test cases for both with and without duplicates.
  // Group by withDuplicates to minimize data regeneration.
  std::vector<TestCase> testCases;
  for (bool withDuplicates : {false, true}) {
    for (const auto& [name, lower, upper] : filterConfigs) {
      std::string testName = withDuplicates ? name + "WithDuplicates" : name;
      testCases.push_back({testName, lower, upper, withDuplicates});
    }
  }

  auto rowType =
      ROW({"key", "data", "nested"},
          {BIGINT(),
           ARRAY(INTEGER()),
           ROW({{"nested", MAP(INTEGER(), VARCHAR())}})});

  // Track current duplicate mode to regenerate data only when needed.
  std::optional<bool> currentWithDuplicates;

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    // Regenerate data only when duplicate mode changes.
    if (!currentWithDuplicates.has_value() ||
        currentWithDuplicates.value() != testCase.withDuplicates) {
      auto batches = generateKeyData(
          BIGINT(), kNumRows, kRowsPerBatch, kMinKey, testCase.withDuplicates);
      // noDuplicateKey should be false when data has duplicates
      writeDataWithParam(batches, {"key"}, !testCase.withDuplicates);
      currentWithDuplicates = testCase.withDuplicates;
    }

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["key"] = std::make_unique<BigintRange>(
        testCase.filterLower, testCase.filterUpper, false);

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    EXPECT_EQ(numConversionsWithIndex, 1);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

// Test with single double key column covering various boundary conditions.
TEST_P(E2EIndexTest, singleDoubleKey) {
  // Key values range from kMinKey to kMinKey + kNumRows - 1 (for unique keys).
  // With duplicates, there are fewer unique keys but the same total row count.
  constexpr double kMinKey = 1'000.0;
  constexpr size_t kNumRows = 10'000;
  constexpr size_t kRowsPerBatch = 1'000;
  const double kMaxKey = kMinKey + kNumRows - 1;

  struct TestCase {
    std::string name;
    double filterLower;
    double filterUpper;
    bool withDuplicates;

    std::string debugString() const {
      return fmt::format(
          "name={}, filterLower={}, filterUpper={}, withDuplicates={}",
          name,
          filterLower,
          filterUpper,
          withDuplicates);
    }
  };

  const double kNumericMin = std::numeric_limits<double>::lowest();
  const double kNumericMax = std::numeric_limits<double>::max();
  const double kNegInfinity = -std::numeric_limits<double>::infinity();
  const double kPosInfinity = std::numeric_limits<double>::infinity();

  // Base filter configurations to test.
  std::vector<std::tuple<std::string, double, double>> filterConfigs = {
      {"pointLookupMiddle", 5'000.0, 5'000.0},
      {"pointLookupAtMin", kMinKey, kMinKey},
      {"pointLookupAtMax", kMaxKey, kMaxKey},
      {"rangeMiddle", 3'000.0, 7'000.0},
      {"rangeFromMin", kMinKey, 5'000.0},
      {"rangeToMax", 5'000.0, kMaxKey},
      {"fullRange", kMinKey, kMaxKey},
      {"lowerBelowMin", 0.0, 5'000.0},
      {"upperAboveMax", 5'000.0, kMaxKey + 1'000.0},
      {"bothBoundsAboveMax", kMaxKey + 100.0, kMaxKey + 200.0},
      {"bothBoundsBelowMin", 0.0, 500.0},
      {"pointLookupBelowMin", 500.0, 500.0},
      {"pointLookupAboveMax", kMaxKey + 100.0, kMaxKey + 100.0},
      // Numeric limits test cases.
      {"lowerAtNumericMin", kNumericMin, 5'000.0},
      {"upperAtNumericMax", 5'000.0, kNumericMax},
      {"fullNumericRange", kNumericMin, kNumericMax},
      // Infinity test cases.
      {"lowerAtNegInfinity", kNegInfinity, 5'000.0},
      {"upperAtPosInfinity", 5'000.0, kPosInfinity},
      {"fullInfinityRange", kNegInfinity, kPosInfinity},
  };

  // Generate test cases for both with and without duplicates.
  // Group by withDuplicates to minimize data regeneration.
  std::vector<TestCase> testCases;
  for (bool withDuplicates : {false, true}) {
    for (const auto& [name, lower, upper] : filterConfigs) {
      std::string testName = withDuplicates ? name + "WithDuplicates" : name;
      testCases.push_back({testName, lower, upper, withDuplicates});
    }
  }

  auto rowType =
      ROW({"key", "data", "nested"},
          {DOUBLE(),
           ARRAY(INTEGER()),
           ROW({{"nested", MAP(INTEGER(), VARCHAR())}})});

  // Track current duplicate mode to regenerate data only when needed.
  std::optional<bool> currentWithDuplicates;

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    // Regenerate data only when duplicate mode changes.
    if (!currentWithDuplicates.has_value() ||
        currentWithDuplicates.value() != testCase.withDuplicates) {
      auto batches = generateKeyData(
          DOUBLE(), kNumRows, kRowsPerBatch, kMinKey, testCase.withDuplicates);
      // noDuplicateKey should be false when data has duplicates
      writeDataWithParam(batches, {"key"}, !testCase.withDuplicates);
      currentWithDuplicates = testCase.withDuplicates;
    }

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["key"] = std::make_unique<DoubleRange>(
        testCase.filterLower,
        false,
        false,
        testCase.filterUpper,
        false,
        false,
        false);

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    EXPECT_EQ(numConversionsWithIndex, 1);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

// Test with single float key column covering various boundary conditions.
TEST_P(E2EIndexTest, singleFloatKey) {
  // Key values range from kMinKey to kMinKey + kNumRows - 1 (for unique keys).
  // With duplicates, there are fewer unique keys but the same total row count.
  constexpr float kMinKey = 1'000.0f;
  constexpr size_t kNumRows = 10'000;
  constexpr size_t kRowsPerBatch = 1'000;
  const float kMaxKey = kMinKey + kNumRows - 1;

  struct TestCase {
    std::string name;
    float filterLower;
    float filterUpper;
    bool withDuplicates;

    std::string debugString() const {
      return fmt::format(
          "name={}, filterLower={}, filterUpper={}, withDuplicates={}",
          name,
          filterLower,
          filterUpper,
          withDuplicates);
    }
  };

  const float kNumericMin = std::numeric_limits<float>::lowest();
  const float kNumericMax = std::numeric_limits<float>::max();
  const float kNegInfinity = -std::numeric_limits<float>::infinity();
  const float kPosInfinity = std::numeric_limits<float>::infinity();

  // Base filter configurations to test.
  std::vector<std::tuple<std::string, float, float>> filterConfigs = {
      {"pointLookupMiddle", 5'000.0f, 5'000.0f},
      {"pointLookupAtMin", kMinKey, kMinKey},
      {"pointLookupAtMax", kMaxKey, kMaxKey},
      {"rangeMiddle", 3'000.0f, 7'000.0f},
      {"rangeFromMin", kMinKey, 5'000.0f},
      {"rangeToMax", 5'000.0f, kMaxKey},
      {"fullRange", kMinKey, kMaxKey},
      {"lowerBelowMin", 0.0f, 5'000.0f},
      {"upperAboveMax", 5'000.0f, kMaxKey + 1'000.0f},
      {"bothBoundsAboveMax", kMaxKey + 100.0f, kMaxKey + 200.0f},
      {"bothBoundsBelowMin", 0.0f, 500.0f},
      {"pointLookupBelowMin", 500.0f, 500.0f},
      {"pointLookupAboveMax", kMaxKey + 100.0f, kMaxKey + 100.0f},
      // Numeric limits test cases.
      {"lowerAtNumericMin", kNumericMin, 5'000.0f},
      {"upperAtNumericMax", 5'000.0f, kNumericMax},
      {"fullNumericRange", kNumericMin, kNumericMax},
      // Infinity test cases.
      {"lowerAtNegInfinity", kNegInfinity, 5'000.0f},
      {"upperAtPosInfinity", 5'000.0f, kPosInfinity},
      {"fullInfinityRange", kNegInfinity, kPosInfinity},
  };

  // Generate test cases for both with and without duplicates.
  // Group by withDuplicates to minimize data regeneration.
  std::vector<TestCase> testCases;
  for (bool withDuplicates : {false, true}) {
    for (const auto& [name, lower, upper] : filterConfigs) {
      std::string testName = withDuplicates ? name + "WithDuplicates" : name;
      testCases.push_back({testName, lower, upper, withDuplicates});
    }
  }

  auto rowType = ROW(
      {"key", "data", "nested"},
      {REAL(), ARRAY(INTEGER()), ROW({{"nested", MAP(INTEGER(), VARCHAR())}})});

  // Track current duplicate mode to regenerate data only when needed.
  std::optional<bool> currentWithDuplicates;

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    // Regenerate data only when duplicate mode changes.
    if (!currentWithDuplicates.has_value() ||
        currentWithDuplicates.value() != testCase.withDuplicates) {
      auto batches = generateKeyData(
          REAL(), kNumRows, kRowsPerBatch, kMinKey, testCase.withDuplicates);
      // noDuplicateKey should be false when data has duplicates
      writeDataWithParam(batches, {"key"}, !testCase.withDuplicates);
      currentWithDuplicates = testCase.withDuplicates;
    }

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["key"] = std::make_unique<FloatRange>(
        testCase.filterLower,
        false,
        false,
        testCase.filterUpper,
        false,
        false,
        false);

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    EXPECT_EQ(numConversionsWithIndex, 1);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

// Test with single timestamp key column covering various boundary conditions.
TEST_P(E2EIndexTest, singleTimestampKey) {
  // Key values range from kMinKey to kMinKey + kNumRows - 1 seconds (for unique
  // keys). With duplicates, there are fewer unique keys but the same total row
  // count.
  constexpr int64_t kMinKeySeconds = 1'000'000;
  constexpr size_t kNumRows = 10'000;
  constexpr size_t kRowsPerBatch = 1'000;
  const int64_t kMaxKeySeconds = kMinKeySeconds + kNumRows - 1;

  struct TestCase {
    std::string name;
    Timestamp filterLower;
    Timestamp filterUpper;
    bool withDuplicates;

    std::string debugString() const {
      return fmt::format(
          "name={}, filterLower=({}, {}), filterUpper=({}, {}), withDuplicates={}",
          name,
          filterLower.getSeconds(),
          filterLower.getNanos(),
          filterUpper.getSeconds(),
          filterUpper.getNanos(),
          withDuplicates);
    }
  };

  const Timestamp kMinKey(kMinKeySeconds, 0);
  const Timestamp kMaxKey(kMaxKeySeconds, 0);
  const Timestamp kNumericMin = Timestamp::minMillis();
  const Timestamp kNumericMax = Timestamp::maxMillis();

  // Base filter configurations to test.
  std::vector<std::tuple<std::string, Timestamp, Timestamp>> filterConfigs = {
      {"pointLookupMiddle", Timestamp(1'005'000, 0), Timestamp(1'005'000, 0)},
      {"pointLookupAtMin", kMinKey, kMinKey},
      {"pointLookupAtMax", kMaxKey, kMaxKey},
      {"rangeMiddle", Timestamp(1'003'000, 0), Timestamp(1'007'000, 0)},
      {"rangeFromMin", kMinKey, Timestamp(1'005'000, 0)},
      {"rangeToMax", Timestamp(1'005'000, 0), kMaxKey},
      {"fullRange", kMinKey, kMaxKey},
      {"lowerBelowMin", Timestamp(0, 0), Timestamp(1'005'000, 0)},
      {"upperAboveMax",
       Timestamp(1'005'000, 0),
       Timestamp(kMaxKeySeconds + 1'000, 0)},
      {"bothBoundsAboveMax",
       Timestamp(kMaxKeySeconds + 100, 0),
       Timestamp(kMaxKeySeconds + 200, 0)},
      {"bothBoundsBelowMin", Timestamp(0, 0), Timestamp(500, 0)},
      {"pointLookupBelowMin", Timestamp(500, 0), Timestamp(500, 0)},
      {"pointLookupAboveMax",
       Timestamp(kMaxKeySeconds + 100, 0),
       Timestamp(kMaxKeySeconds + 100, 0)},
      // Numeric limits test cases.
      {"lowerAtNumericMin", kNumericMin, Timestamp(1'005'000, 0)},
      {"upperAtNumericMax", Timestamp(1'005'000, 0), kNumericMax},
      {"fullNumericRange", kNumericMin, kNumericMax},
  };

  // Generate test cases for both with and without duplicates.
  // Group by withDuplicates to minimize data regeneration.
  std::vector<TestCase> testCases;
  for (bool withDuplicates : {false, true}) {
    for (const auto& [name, lower, upper] : filterConfigs) {
      std::string testName = withDuplicates ? name + "WithDuplicates" : name;
      testCases.push_back({testName, lower, upper, withDuplicates});
    }
  }

  auto rowType =
      ROW({"key", "data", "nested"},
          {TIMESTAMP(),
           ARRAY(INTEGER()),
           ROW({{"nested", MAP(INTEGER(), VARCHAR())}})});

  // Track current duplicate mode to regenerate data only when needed.
  std::optional<bool> currentWithDuplicates;

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    // Regenerate data only when duplicate mode changes.
    if (!currentWithDuplicates.has_value() ||
        currentWithDuplicates.value() != testCase.withDuplicates) {
      auto batches = generateKeyData(
          TIMESTAMP(),
          kNumRows,
          kRowsPerBatch,
          kMinKeySeconds,
          testCase.withDuplicates);
      // noDuplicateKey should be false when data has duplicates
      writeDataWithParam(batches, {"key"}, !testCase.withDuplicates);
      currentWithDuplicates = testCase.withDuplicates;
    }

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["key"] = std::make_unique<TimestampRange>(
        testCase.filterLower, testCase.filterUpper, false);

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    EXPECT_EQ(numConversionsWithIndex, 1);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

// Test with single boolean key column.
// Boolean keys naturally have duplicates since there are only two possible
// values (true/false).
TEST_P(E2EIndexTest, singleBoolKey) {
  constexpr size_t kNumRows = 2'000;
  constexpr size_t kRowsPerBatch = 200;

  struct TestCase {
    std::string name;
    bool filterValue;

    std::string debugString() const {
      return fmt::format("name={}, filterValue={}", name, filterValue);
    }
  };

  // Test both possible boolean filter values.
  std::vector<TestCase> testCases = {
      {"filterFalse", false},
      {"filterTrue", true},
  };

  auto rowType = ROW({"key", "data"}, {BOOLEAN(), ARRAY(INTEGER())});

  // Generate sorted boolean keys based on sort order.
  // Ascending: false first, then true.
  // Descending: true first, then false.
  const bool ascending = GetParam().sortOrder.ascending;
  std::vector<RowVectorPtr> batches;
  for (size_t batchIdx = 0; batchIdx < kNumRows / kRowsPerBatch; ++batchIdx) {
    std::vector<bool> keyValues(kRowsPerBatch);
    for (size_t i = 0; i < kRowsPerBatch; ++i) {
      size_t globalRow = batchIdx * kRowsPerBatch + i;
      // First half of rows get the "smaller" value, second half get the
      // "larger" value. For ascending: false < true. For descending: true <
      // false.
      bool isSecondHalf = globalRow >= kNumRows / 2;
      keyValues[i] = ascending ? isSecondHalf : !isSecondHalf;
    }
    auto keyVector = vectorMaker_->flatVector(keyValues);
    auto dataVector = makeFuzzedComplexVector(
        ARRAY(INTEGER()), kRowsPerBatch, batchIdx * 12345);
    batches.push_back(
        vectorMaker_->rowVector({"key", "data"}, {keyVector, dataVector}));
  }

  // Boolean keys naturally have duplicates, so noDuplicateKey = false
  writeDataWithParam(batches, {"key"}, false);

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["key"] = std::make_unique<BoolValue>(testCase.filterValue, false);

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    EXPECT_EQ(numConversionsWithIndex, 1);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

// Test with single varchar key column covering various boundary conditions.
TEST_P(E2EIndexTest, singleVarcharKey) {
  // Key values are formatted strings "key_00001000" to "key_00010999" (for
  // unique keys). With duplicates, there are fewer unique keys but the same
  // total row count.
  constexpr int64_t kMinKeyNum = 1'000;
  constexpr size_t kNumRows = 10'000;
  constexpr size_t kRowsPerBatch = 1'000;
  const int64_t kMaxKeyNum = kMinKeyNum + kNumRows - 1;

  // Helper to format key number to string.
  auto formatKey = [](int64_t num) { return fmt::format("key_{:08d}", num); };

  const std::string kMinKey = formatKey(kMinKeyNum);
  const std::string kMaxKey = formatKey(kMaxKeyNum);

  struct TestCase {
    std::string name;
    std::string filterLower;
    std::string filterUpper;
    bool isPointLookup;
    bool withDuplicates;

    std::string debugString() const {
      return fmt::format(
          "name={}, filterLower={}, filterUpper={}, isPointLookup={}, withDuplicates={}",
          name,
          filterLower,
          filterUpper,
          isPointLookup,
          withDuplicates);
    }
  };

  // Base filter configurations to test.
  // For point lookups, we use BytesValues filter.
  // For range queries, we use BytesRange filter.
  std::vector<std::tuple<std::string, std::string, std::string, bool>>
      filterConfigs = {
          // Point lookups using BytesValues.
          {"pointLookupMiddle", formatKey(5'000), formatKey(5'000), true},
          {"pointLookupAtMin", kMinKey, kMinKey, true},
          {"pointLookupAtMax", kMaxKey, kMaxKey, true},
          {"pointLookupBelowMin", formatKey(500), formatKey(500), true},
          {"pointLookupAboveMax",
           formatKey(kMaxKeyNum + 100),
           formatKey(kMaxKeyNum + 100),
           true},
          // Range queries using BytesRange.
          {"rangeMiddle", formatKey(3'000), formatKey(7'000), false},
          {"rangeFromMin", kMinKey, formatKey(5'000), false},
          {"rangeToMax", formatKey(5'000), kMaxKey, false},
          {"fullRange", kMinKey, kMaxKey, false},
          {"lowerBelowMin", formatKey(0), formatKey(5'000), false},
          {"upperAboveMax",
           formatKey(5'000),
           formatKey(kMaxKeyNum + 1'000),
           false},
          {"bothBoundsAboveMax",
           formatKey(kMaxKeyNum + 100),
           formatKey(kMaxKeyNum + 200),
           false},
          {"bothBoundsBelowMin", formatKey(0), formatKey(500), false},
          // String boundary test cases.
          {"lowerAtEmptyString", "", formatKey(5'000), false},
          {"prefixRange", "key_0000", "key_0001", false},
          {"singleCharRange", "k", "l", false},
      };

  // Generate test cases for both with and without duplicates.
  // Group by withDuplicates to minimize data regeneration.
  std::vector<TestCase> testCases;
  // for (bool withDuplicates : {false, true}) {
  for (bool withDuplicates : {false}) {
    for (const auto& [name, lower, upper, isPoint] : filterConfigs) {
      std::string testName = withDuplicates ? name + "WithDuplicates" : name;
      testCases.push_back({testName, lower, upper, isPoint, withDuplicates});
    }
  }

  auto rowType =
      ROW({"key", "data", "nested"},
          {VARCHAR(),
           ARRAY(INTEGER()),
           ROW({{"nested", MAP(INTEGER(), VARCHAR())}})});

  // Track current duplicate mode to regenerate data only when needed.
  std::optional<bool> currentWithDuplicates;

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    // Regenerate data only when duplicate mode changes.
    if (!currentWithDuplicates.has_value() ||
        currentWithDuplicates.value() != testCase.withDuplicates) {
      auto batches = generateKeyData(
          VARCHAR(),
          kNumRows,
          kRowsPerBatch,
          kMinKeyNum,
          testCase.withDuplicates);
      // noDuplicateKey should be false when data has duplicates
      writeDataWithParam(batches, {"key"}, !testCase.withDuplicates);
      currentWithDuplicates = testCase.withDuplicates;
    }

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    if (testCase.isPointLookup) {
      // Use BytesValues for point lookups.
      filters["key"] = std::make_unique<BytesValues>(
          std::vector<std::string>{testCase.filterLower}, false);
    } else {
      // Use BytesRange for range queries.
      filters["key"] = std::make_unique<BytesRange>(
          testCase.filterLower,
          false,
          false,
          testCase.filterUpper,
          false,
          false,
          false);
    }

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    // For descending order, VARCHAR index filter conversion is not supported.
    const int64_t expectedConversions = isAscending() ? 1 : 0;
    EXPECT_EQ(numConversionsWithIndex, expectedConversions);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

TEST_P(E2EIndexTest, twoKeyColumnsPointAndRange) {
  constexpr size_t kNumRows = 5'000;
  constexpr size_t kRowsPerBatch = 500;

  auto rowType =
      ROW({"key1", "key2", "data"},
          {BIGINT(), VARCHAR(), ROW({{"nested", ARRAY(INTEGER())}})});

  const bool ascending = isAscending();
  std::vector<RowVectorPtr> batches;
  for (size_t batchIdx = 0; batchIdx < kNumRows / kRowsPerBatch; ++batchIdx) {
    std::vector<int64_t> key1Values(kRowsPerBatch);
    std::vector<std::string> key2Values(kRowsPerBatch);
    for (size_t i = 0; i < kRowsPerBatch; ++i) {
      const size_t globalIdx = batchIdx * kRowsPerBatch + i;
      const size_t effectiveIdx =
          ascending ? globalIdx : (kNumRows - 1 - globalIdx);
      // key1: each value repeated 10 times.
      key1Values[i] = effectiveIdx / 10;
      // key2: sorted within each key1 group.
      key2Values[i] = fmt::format("val_{:05d}", effectiveIdx % 10);
    }
    auto key1Vector = vectorMaker_->flatVector(key1Values);
    auto key2Vector = vectorMaker_->flatVector(key2Values);

    auto dataVector = makeFuzzedComplexVector(
        ROW({{"nested", ARRAY(INTEGER())}}), kRowsPerBatch, batchIdx * 11111);
    batches.push_back(vectorMaker_->rowVector(
        {"key1", "key2", "data"}, {key1Vector, key2Vector, dataVector}));
  }

  writeDataWithParam(batches, {"key1", "key2"});

  // Point lookup on key1 (= 100), range on key2.
  std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
  filters["key1"] = std::make_unique<BigintRange>(100, 100, false);
  filters["key2"] = std::make_unique<BytesRange>(
      "val_00002", false, false, "val_00008", false, false, false);

  auto reader = createReader();

  int64_t numConversionsWithIndex = 0;
  auto resultsWithIndex = readWithFilters(
      *reader,
      rowType,
      filters,
      /*indexEnabled=*/true,
      numConversionsWithIndex);

  int64_t numConversionsWithoutIndex = 0;
  auto resultsWithoutIndex = readWithFilters(
      *reader,
      rowType,
      filters,
      /*indexEnabled=*/false,
      numConversionsWithoutIndex);

  verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

  // For ascending order, both key columns should be converted (2 conversions).
  // For descending order, VARCHAR index filter conversion is not supported,
  // so only key1 (BIGINT) is converted (1 conversion).
  const int64_t expectedConversions = ascending ? 2 : 1;
  EXPECT_EQ(numConversionsWithIndex, expectedConversions);
  EXPECT_EQ(numConversionsWithoutIndex, 0);
}

// Test with two bigint key columns covering all filter-to-index conversion
// combinations. This verifies that:
// - Point + Point: both columns converted (2 conversions)
// - Point + Range: both columns converted (2 conversions)
// - Range + Point: only first column converted (1 conversion)
// - Range + Range: only first column converted (1 conversion)
// - No filter on key1 + filter on key2: no conversion (0 conversions)
// - Filter gap (key1 and key3 but not key2): only key1 converted (1 conversion)
// Includes nested complex types in the data column.
TEST_P(E2EIndexTest, twoBigintKeysFilterCombinations) {
  constexpr size_t kNumRows = 5'000;
  constexpr size_t kRowsPerBatch = 500;

  // Generate data with two bigint keys.
  // key1 has duplicates (each value repeated 10 times), key2 is sorted within
  // each key1 group.
  auto rowType =
      ROW({"key1", "key2", "data", "nested"},
          {BIGINT(),
           BIGINT(),
           ARRAY(MAP(INTEGER(), VARCHAR())),
           ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}})});

  const bool ascending = isAscending();
  std::vector<RowVectorPtr> batches;
  for (size_t batchIdx = 0; batchIdx < kNumRows / kRowsPerBatch; ++batchIdx) {
    std::vector<int64_t> key1Values(kRowsPerBatch);
    std::vector<int64_t> key2Values(kRowsPerBatch);
    for (size_t i = 0; i < kRowsPerBatch; ++i) {
      const size_t globalIdx = batchIdx * kRowsPerBatch + i;
      const size_t effectiveIdx =
          ascending ? globalIdx : (kNumRows - 1 - globalIdx);
      // key1: 0, 0, ..., 0 (10x), 1, 1, ..., 1 (10x), etc.
      key1Values[i] = effectiveIdx / 10;
      // key2: 0, 1, 2, ..., 9, 0, 1, 2, ..., 9, etc. (sorted within key1 group)
      key2Values[i] = effectiveIdx % 10;
    }
    auto key1Vector = vectorMaker_->flatVector(key1Values);
    auto key2Vector = vectorMaker_->flatVector(key2Values);
    auto dataVector = makeFuzzedComplexVector(
        ARRAY(MAP(INTEGER(), VARCHAR())), kRowsPerBatch, batchIdx * 33333);
    auto nestedVector = makeFuzzedComplexVector(
        ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}}),
        kRowsPerBatch,
        batchIdx * 44444);
    batches.push_back(vectorMaker_->rowVector(
        {"key1", "key2", "data", "nested"},
        {key1Vector, key2Vector, dataVector, nestedVector}));
  }

  writeDataWithParam(batches, {"key1", "key2"});

  struct TestCase {
    std::string name;
    std::unique_ptr<Filter> key1Filter;
    std::unique_ptr<Filter> key2Filter;
    int64_t expectedConversions;

    std::string debugString() const {
      return fmt::format(
          "name={}, expectedConversions={}, key1Filter={}, key2Filter={}",
          name,
          expectedConversions,
          key1Filter ? key1Filter->serialize() : "null",
          key2Filter ? key2Filter->serialize() : "null");
    }
  };

  std::vector<TestCase> testCases;

  // Point + Point: both columns converted.
  testCases.push_back(
      {"pointPoint",
       std::make_unique<BigintRange>(100, 100, false),
       std::make_unique<BigintRange>(5, 5, false),
       2});

  // Point + Range: both columns converted.
  testCases.push_back(
      {"pointRange",
       std::make_unique<BigintRange>(100, 100, false),
       std::make_unique<BigintRange>(2, 8, false),
       2});

  // Range + Point: only first column converted (range stops extraction).
  testCases.push_back(
      {"rangePoint",
       std::make_unique<BigintRange>(50, 150, false),
       std::make_unique<BigintRange>(5, 5, false),
       1});

  // Range + Range: only first column converted (range stops extraction).
  testCases.push_back(
      {"rangeRange",
       std::make_unique<BigintRange>(50, 150, false),
       std::make_unique<BigintRange>(2, 8, false),
       1});

  // No filter on key1, filter on key2: no conversion.
  testCases.push_back(
      {"noFilterOnKey1",
       nullptr,
       std::make_unique<BigintRange>(5, 5, false),
       0});

  // Only filter on key1 (point): 1 conversion.
  testCases.push_back(
      {"onlyKey1Point",
       std::make_unique<BigintRange>(100, 100, false),
       nullptr,
       1});

  // Only filter on key1 (range): 1 conversion.
  testCases.push_back(
      {"onlyKey1Range",
       std::make_unique<BigintRange>(50, 150, false),
       nullptr,
       1});

  for (auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    if (testCase.key1Filter) {
      filters["key1"] = testCase.key1Filter->clone();
    }
    if (testCase.key2Filter) {
      filters["key2"] = testCase.key2Filter->clone();
    }

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    EXPECT_EQ(numConversionsWithIndex, testCase.expectedConversions);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

// Test with three bigint key columns to verify filter gap behavior.
// When there's a gap in the filter chain (filter on key1 and key3, but not
// key2), extraction stops at the missing filter on key2.
// Includes nested complex types in the data column.
TEST_P(E2EIndexTest, threeBigintKeysFilterGap) {
  constexpr size_t kNumRows = 3'000;
  constexpr size_t kRowsPerBatch = 300;

  auto rowType =
      ROW({"key1", "key2", "key3", "data", "nested"},
          {BIGINT(),
           BIGINT(),
           BIGINT(),
           ARRAY(MAP(INTEGER(), VARCHAR())),
           ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}})});

  const bool ascending = isAscending();
  std::vector<RowVectorPtr> batches;
  for (size_t batchIdx = 0; batchIdx < kNumRows / kRowsPerBatch; ++batchIdx) {
    std::vector<int64_t> key1Values(kRowsPerBatch);
    std::vector<int64_t> key2Values(kRowsPerBatch);
    std::vector<int64_t> key3Values(kRowsPerBatch);
    for (size_t i = 0; i < kRowsPerBatch; ++i) {
      const size_t globalIdx = batchIdx * kRowsPerBatch + i;
      const size_t effectiveIdx =
          ascending ? globalIdx : (kNumRows - 1 - globalIdx);
      // key1: each value repeated 100 times.
      key1Values[i] = effectiveIdx / 100;
      // key2: each value repeated 10 times within key1 group.
      key2Values[i] = (effectiveIdx / 10) % 10;
      // key3: unique within key1+key2 group.
      key3Values[i] = effectiveIdx % 10;
    }
    auto key1Vector = vectorMaker_->flatVector(key1Values);
    auto key2Vector = vectorMaker_->flatVector(key2Values);
    auto key3Vector = vectorMaker_->flatVector(key3Values);
    auto dataVector = makeFuzzedComplexVector(
        ARRAY(MAP(INTEGER(), VARCHAR())), kRowsPerBatch, batchIdx * 55555);
    auto nestedVector = makeFuzzedComplexVector(
        ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}}),
        kRowsPerBatch,
        batchIdx * 66666);
    batches.push_back(vectorMaker_->rowVector(
        {"key1", "key2", "key3", "data", "nested"},
        {key1Vector, key2Vector, key3Vector, dataVector, nestedVector}));
  }

  writeDataWithParam(batches, {"key1", "key2", "key3"});

  struct TestCase {
    std::string name;
    std::unique_ptr<Filter> key1Filter;
    std::unique_ptr<Filter> key2Filter;
    std::unique_ptr<Filter> key3Filter;
    int64_t expectedConversions;

    std::string debugString() const {
      return fmt::format(
          "name={}, expectedConversions={}, key1Filter={}, key2Filter={}, key3Filter={}",
          name,
          expectedConversions,
          key1Filter ? key1Filter->toString() : "null",
          key2Filter ? key2Filter->toString() : "null",
          key3Filter ? key3Filter->toString() : "null");
    }
  };

  std::vector<TestCase> testCases;

  // Point + Point + Point: all three columns converted.
  testCases.push_back(
      {"pointPointPoint",
       std::make_unique<BigintRange>(10, 10, false),
       std::make_unique<BigintRange>(5, 5, false),
       std::make_unique<BigintRange>(3, 3, false),
       3});

  // Point + Point + Range: all three columns converted.
  testCases.push_back(
      {"pointPointRange",
       std::make_unique<BigintRange>(10, 10, false),
       std::make_unique<BigintRange>(5, 5, false),
       std::make_unique<BigintRange>(1, 8, false),
       3});

  // Point + Range + Point: only first two columns converted (range stops).
  testCases.push_back(
      {"pointRangePoint",
       std::make_unique<BigintRange>(10, 10, false),
       std::make_unique<BigintRange>(2, 7, false),
       std::make_unique<BigintRange>(3, 3, false),
       2});

  // Range + Point + Point: only first column converted (range stops).
  testCases.push_back(
      {"rangePointPoint",
       std::make_unique<BigintRange>(5, 15, false),
       std::make_unique<BigintRange>(5, 5, false),
       std::make_unique<BigintRange>(3, 3, false),
       1});

  // Filter gap: point on key1 + no filter on key2 + point on key3.
  // Only key1 converted because of the gap at key2.
  testCases.push_back(
      {"gapAtKey2",
       std::make_unique<BigintRange>(10, 10, false),
       nullptr,
       std::make_unique<BigintRange>(3, 3, false),
       1});

  // No filter on key1: no conversion even with filters on key2 and key3.
  testCases.push_back(
      {"noFilterOnKey1",
       nullptr,
       std::make_unique<BigintRange>(5, 5, false),
       std::make_unique<BigintRange>(3, 3, false),
       0});

  for (auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    if (testCase.key1Filter) {
      filters["key1"] = testCase.key1Filter->clone();
    }
    if (testCase.key2Filter) {
      filters["key2"] = testCase.key2Filter->clone();
    }
    if (testCase.key3Filter) {
      filters["key3"] = testCase.key3Filter->clone();
    }

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    EXPECT_EQ(numConversionsWithIndex, testCase.expectedConversions);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

// Test with two double key columns covering all filter-to-index conversion
// combinations. Includes nested complex types in the data column.
TEST_P(E2EIndexTest, twoDoubleKeysFilterCombinations) {
  constexpr size_t kNumRows = 5'000;
  constexpr size_t kRowsPerBatch = 500;

  auto rowType =
      ROW({"key1", "key2", "data", "nested"},
          {DOUBLE(),
           DOUBLE(),
           ARRAY(MAP(INTEGER(), VARCHAR())),
           ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}})});

  const bool ascending = isAscending();
  std::vector<RowVectorPtr> batches;
  for (size_t batchIdx = 0; batchIdx < kNumRows / kRowsPerBatch; ++batchIdx) {
    std::vector<double> key1Values(kRowsPerBatch);
    std::vector<double> key2Values(kRowsPerBatch);
    for (size_t i = 0; i < kRowsPerBatch; ++i) {
      const size_t globalIdx = batchIdx * kRowsPerBatch + i;
      const size_t effectiveIdx =
          ascending ? globalIdx : (kNumRows - 1 - globalIdx);
      // key1: 0.0, 0.0, ..., 0.0 (10x), 1.0, 1.0, ..., 1.0 (10x), etc.
      key1Values[i] = static_cast<double>(effectiveIdx / 10);
      // key2: 0.0, 1.0, 2.0, ..., 9.0 (sorted within key1 group)
      key2Values[i] = static_cast<double>(effectiveIdx % 10);
    }
    auto key1Vector = vectorMaker_->flatVector(key1Values);
    auto key2Vector = vectorMaker_->flatVector(key2Values);
    auto dataVector = makeFuzzedComplexVector(
        ARRAY(MAP(INTEGER(), VARCHAR())), kRowsPerBatch, batchIdx * 111);
    auto nestedVector = makeFuzzedComplexVector(
        ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}}),
        kRowsPerBatch,
        batchIdx * 222);
    batches.push_back(vectorMaker_->rowVector(
        {"key1", "key2", "data", "nested"},
        {key1Vector, key2Vector, dataVector, nestedVector}));
  }

  writeDataWithParam(batches, {"key1", "key2"});

  struct TestCase {
    std::string name;
    std::unique_ptr<Filter> key1Filter;
    std::unique_ptr<Filter> key2Filter;
    int64_t expectedConversions;

    std::string debugString() const {
      return fmt::format(
          "name={}, expectedConversions={}, key1Filter={}, key2Filter={}",
          name,
          expectedConversions,
          key1Filter ? key1Filter->toString() : "null",
          key2Filter ? key2Filter->toString() : "null");
    }
  };

  std::vector<TestCase> testCases;

  // Point + Point: both columns converted.
  testCases.push_back(
      {"pointPoint",
       std::make_unique<DoubleRange>(
           100.0, false, false, 100.0, false, false, false),
       std::make_unique<DoubleRange>(
           5.0, false, false, 5.0, false, false, false),
       2});

  // Point + Range: both columns converted.
  testCases.push_back(
      {"pointRange",
       std::make_unique<DoubleRange>(
           100.0, false, false, 100.0, false, false, false),
       std::make_unique<DoubleRange>(
           2.0, false, false, 8.0, false, false, false),
       2});

  // Range + Point: only first column converted (range stops extraction).
  testCases.push_back(
      {"rangePoint",
       std::make_unique<DoubleRange>(
           50.0, false, false, 150.0, false, false, false),
       std::make_unique<DoubleRange>(
           5.0, false, false, 5.0, false, false, false),
       1});

  // Range + Range: only first column converted (range stops extraction).
  testCases.push_back(
      {"rangeRange",
       std::make_unique<DoubleRange>(
           50.0, false, false, 150.0, false, false, false),
       std::make_unique<DoubleRange>(
           2.0, false, false, 8.0, false, false, false),
       1});

  // No filter on key1, filter on key2: no conversion.
  testCases.push_back(
      {"noFilterOnKey1",
       nullptr,
       std::make_unique<DoubleRange>(
           5.0, false, false, 5.0, false, false, false),
       0});

  // Only filter on key1 (point): 1 conversion.
  testCases.push_back(
      {"onlyKey1Point",
       std::make_unique<DoubleRange>(
           100.0, false, false, 100.0, false, false, false),
       nullptr,
       1});

  // Only filter on key1 (range): 1 conversion.
  testCases.push_back(
      {"onlyKey1Range",
       std::make_unique<DoubleRange>(
           50.0, false, false, 150.0, false, false, false),
       nullptr,
       1});

  for (auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    if (testCase.key1Filter) {
      filters["key1"] = testCase.key1Filter->clone();
    }
    if (testCase.key2Filter) {
      filters["key2"] = testCase.key2Filter->clone();
    }

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    EXPECT_EQ(numConversionsWithIndex, testCase.expectedConversions);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

// Test with two float key columns covering all filter-to-index conversion
// combinations. Includes nested complex types in the data column.
TEST_P(E2EIndexTest, twoFloatKeysFilterCombinations) {
  constexpr size_t kNumRows = 5'000;
  constexpr size_t kRowsPerBatch = 500;

  auto rowType =
      ROW({"key1", "key2", "data", "nested"},
          {REAL(),
           REAL(),
           ARRAY(MAP(INTEGER(), VARCHAR())),
           ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}})});

  const bool ascending = isAscending();
  std::vector<RowVectorPtr> batches;
  for (size_t batchIdx = 0; batchIdx < kNumRows / kRowsPerBatch; ++batchIdx) {
    std::vector<float> key1Values(kRowsPerBatch);
    std::vector<float> key2Values(kRowsPerBatch);
    for (size_t i = 0; i < kRowsPerBatch; ++i) {
      const size_t globalIdx = batchIdx * kRowsPerBatch + i;
      const size_t effectiveIdx =
          ascending ? globalIdx : (kNumRows - 1 - globalIdx);
      // key1: 0.0f, 0.0f, ..., 0.0f (10x), 1.0f, 1.0f, ..., 1.0f (10x), etc.
      key1Values[i] = static_cast<float>(effectiveIdx / 10);
      // key2: 0.0f, 1.0f, 2.0f, ..., 9.0f (sorted within key1 group)
      key2Values[i] = static_cast<float>(effectiveIdx % 10);
    }
    auto key1Vector = vectorMaker_->flatVector(key1Values);
    auto key2Vector = vectorMaker_->flatVector(key2Values);
    auto dataVector = makeFuzzedComplexVector(
        ARRAY(MAP(INTEGER(), VARCHAR())), kRowsPerBatch, batchIdx * 333);
    auto nestedVector = makeFuzzedComplexVector(
        ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}}),
        kRowsPerBatch,
        batchIdx * 444);
    batches.push_back(vectorMaker_->rowVector(
        {"key1", "key2", "data", "nested"},
        {key1Vector, key2Vector, dataVector, nestedVector}));
  }

  writeDataWithParam(batches, {"key1", "key2"});

  struct TestCase {
    std::string name;
    std::unique_ptr<Filter> key1Filter;
    std::unique_ptr<Filter> key2Filter;
    int64_t expectedConversions;

    std::string debugString() const {
      return fmt::format(
          "name={}, expectedConversions={}, key1Filter={}, key2Filter={}",
          name,
          expectedConversions,
          key1Filter ? key1Filter->toString() : "null",
          key2Filter ? key2Filter->toString() : "null");
    }
  };

  std::vector<TestCase> testCases;

  // Point + Point: both columns converted.
  testCases.push_back(
      {"pointPoint",
       std::make_unique<FloatRange>(
           100.0f, false, false, 100.0f, false, false, false),
       std::make_unique<FloatRange>(
           5.0f, false, false, 5.0f, false, false, false),
       2});

  // Point + Range: both columns converted.
  testCases.push_back(
      {"pointRange",
       std::make_unique<FloatRange>(
           100.0f, false, false, 100.0f, false, false, false),
       std::make_unique<FloatRange>(
           2.0f, false, false, 8.0f, false, false, false),
       2});

  // Range + Point: only first column converted (range stops extraction).
  testCases.push_back(
      {"rangePoint",
       std::make_unique<FloatRange>(
           50.0f, false, false, 150.0f, false, false, false),
       std::make_unique<FloatRange>(
           5.0f, false, false, 5.0f, false, false, false),
       1});

  // Range + Range: only first column converted (range stops extraction).
  testCases.push_back(
      {"rangeRange",
       std::make_unique<FloatRange>(
           50.0f, false, false, 150.0f, false, false, false),
       std::make_unique<FloatRange>(
           2.0f, false, false, 8.0f, false, false, false),
       1});

  // No filter on key1, filter on key2: no conversion.
  testCases.push_back(
      {"noFilterOnKey1",
       nullptr,
       std::make_unique<FloatRange>(
           5.0f, false, false, 5.0f, false, false, false),
       0});

  // Only filter on key1 (point): 1 conversion.
  testCases.push_back(
      {"onlyKey1Point",
       std::make_unique<FloatRange>(
           100.0f, false, false, 100.0f, false, false, false),
       nullptr,
       1});

  // Only filter on key1 (range): 1 conversion.
  testCases.push_back(
      {"onlyKey1Range",
       std::make_unique<FloatRange>(
           50.0f, false, false, 150.0f, false, false, false),
       nullptr,
       1});

  for (auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    if (testCase.key1Filter) {
      filters["key1"] = testCase.key1Filter->clone();
    }
    if (testCase.key2Filter) {
      filters["key2"] = testCase.key2Filter->clone();
    }

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    EXPECT_EQ(numConversionsWithIndex, testCase.expectedConversions);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

// Test with two boolean key columns covering filter-to-index conversion
// combinations. Note: Boolean only supports point lookups (BoolValue filter),
// not range filters. Includes nested complex types in the data column.
// Two boolean keys have exactly 4 unique combinations when strictly ordered:
// (false, false), (false, true), (true, false), (true, true).
TEST_P(E2EIndexTest, twoBoolKeysFilterCombinations) {
  constexpr size_t kNumRows = 4'000;
  constexpr size_t kRowsPerBatch = 400;

  const auto rowType =
      ROW({"key1", "key2", "data", "nested"},
          {BOOLEAN(),
           BOOLEAN(),
           ARRAY(MAP(INTEGER(), VARCHAR())),
           ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}})});

  // Generate sorted boolean keys based on sort order.
  // Ascending: key1 starts false then true; key2 sorted within key1 groups
  // Descending: key1 starts true then false; key2 sorted within key1 groups
  const bool ascending = GetParam().sortOrder.ascending;
  std::vector<RowVectorPtr> batches;
  for (size_t batchIdx = 0; batchIdx < kNumRows / kRowsPerBatch; ++batchIdx) {
    std::vector<bool> key1Values(kRowsPerBatch);
    std::vector<bool> key2Values(kRowsPerBatch);
    for (size_t i = 0; i < kRowsPerBatch; ++i) {
      size_t globalIdx = batchIdx * kRowsPerBatch + i;
      // key1: first half vs second half (ascending: false then true)
      bool isSecondHalfKey1 = globalIdx >= kNumRows / 2;
      key1Values[i] = ascending ? isSecondHalfKey1 : !isSecondHalfKey1;
      // key2: sorted within key1 group (first half false, second half true)
      const size_t halfSize = kNumRows / 4;
      bool isSecondHalfKey2;
      if (globalIdx < kNumRows / 2) {
        isSecondHalfKey2 = (globalIdx % (kNumRows / 2)) >= halfSize;
      } else {
        isSecondHalfKey2 =
            ((globalIdx - kNumRows / 2) % (kNumRows / 2)) >= halfSize;
      }
      key2Values[i] = ascending ? isSecondHalfKey2 : !isSecondHalfKey2;
    }
    auto key1Vector = vectorMaker_->flatVector(key1Values);
    auto key2Vector = vectorMaker_->flatVector(key2Values);
    auto dataVector = makeFuzzedComplexVector(
        ARRAY(MAP(INTEGER(), VARCHAR())), kRowsPerBatch, batchIdx * 555);
    auto nestedVector = makeFuzzedComplexVector(
        ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}}),
        kRowsPerBatch,
        batchIdx * 666);
    batches.push_back(vectorMaker_->rowVector(
        {"key1", "key2", "data", "nested"},
        {key1Vector, key2Vector, dataVector, nestedVector}));
  }

  // Boolean keys naturally have duplicates, so noDuplicateKey = false
  writeDataWithParam(batches, {"key1", "key2"}, false);

  struct TestCase {
    std::string name;
    std::unique_ptr<Filter> key1Filter;
    std::unique_ptr<Filter> key2Filter;
    int64_t expectedConversions;

    std::string debugString() const {
      return fmt::format(
          "name={}, expectedConversions={}, key1Filter={}, key2Filter={}",
          name,
          expectedConversions,
          key1Filter ? key1Filter->toString() : "null",
          key2Filter ? key2Filter->toString() : "null");
    }
  };

  std::vector<TestCase> testCases;
  // Point + Point (false, false): both columns converted.
  testCases.push_back(
      {"pointPointFalseFalse",
       std::make_unique<BoolValue>(false, false),
       std::make_unique<BoolValue>(false, false),
       2});
  // Point + Point (false, true): both columns converted.
  testCases.push_back(
      {"pointPointFalseTrue",
       std::make_unique<BoolValue>(false, false),
       std::make_unique<BoolValue>(true, false),
       2});
  // Point + Point (true, false): both columns converted.
  testCases.push_back(
      {"pointPointTrueFalse",
       std::make_unique<BoolValue>(true, false),
       std::make_unique<BoolValue>(false, false),
       2});

  // Point + Point (true, true): both columns converted.
  testCases.push_back(
      {"pointPointTrueTrue",
       std::make_unique<BoolValue>(true, false),
       std::make_unique<BoolValue>(true, false),
       2});

  // No filter on key1, filter on key2: no conversion.
  testCases.push_back(
      {"noFilterOnKey1", nullptr, std::make_unique<BoolValue>(true, false), 0});

  // Only filter on key1: 1 conversion.
  testCases.push_back(
      {"onlyKey1", std::make_unique<BoolValue>(false, false), nullptr, 1});

  for (auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    if (testCase.key1Filter) {
      filters["key1"] = testCase.key1Filter->clone();
    }
    if (testCase.key2Filter) {
      filters["key2"] = testCase.key2Filter->clone();
    }

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    EXPECT_EQ(numConversionsWithIndex, testCase.expectedConversions);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

// Test with two varchar key columns covering all filter-to-index conversion
// combinations. Includes nested complex types in the data column.
TEST_P(E2EIndexTest, twoVarcharKeysFilterCombinations) {
  constexpr size_t kNumRows = 5'000;
  constexpr size_t kRowsPerBatch = 500;

  const auto rowType =
      ROW({"key1", "key2", "data", "nested"},
          {VARCHAR(),
           VARCHAR(),
           ARRAY(MAP(INTEGER(), VARCHAR())),
           ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}})});

  auto formatKey1 = [](size_t idx) { return fmt::format("k1_{:05d}", idx); };
  auto formatKey2 = [](size_t idx) { return fmt::format("k2_{:05d}", idx); };

  const bool ascending = isAscending();

  std::vector<RowVectorPtr> batches;
  for (size_t batchIdx = 0; batchIdx < kNumRows / kRowsPerBatch; ++batchIdx) {
    std::vector<std::string> key1Values(kRowsPerBatch);
    std::vector<std::string> key2Values(kRowsPerBatch);
    for (size_t i = 0; i < kRowsPerBatch; ++i) {
      const size_t globalIdx = batchIdx * kRowsPerBatch + i;
      const size_t effectiveIdx =
          ascending ? globalIdx : (kNumRows - 1 - globalIdx);
      // key1: k1_00000 (10x), k1_00001 (10x), etc.
      key1Values[i] = formatKey1(effectiveIdx / 10);
      // key2: k2_00000, k2_00001, ..., k2_00009 (sorted within key1 group)
      key2Values[i] = formatKey2(effectiveIdx % 10);
    }

    auto key1Vector = vectorMaker_->flatVector(key1Values);
    auto key2Vector = vectorMaker_->flatVector(key2Values);
    auto dataVector = makeFuzzedComplexVector(
        ARRAY(MAP(INTEGER(), VARCHAR())), kRowsPerBatch, batchIdx * 777);
    auto nestedVector = makeFuzzedComplexVector(
        ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}}),
        kRowsPerBatch,
        batchIdx * 888);
    batches.push_back(vectorMaker_->rowVector(
        {"key1", "key2", "data", "nested"},
        {key1Vector, key2Vector, dataVector, nestedVector}));
  }

  writeDataWithParam(batches, {"key1", "key2"});

  struct TestCase {
    std::string name;
    std::unique_ptr<Filter> key1Filter;
    std::unique_ptr<Filter> key2Filter;
    // Expected conversions for ascending order.
    // For descending order, VARCHAR index filter conversion is not supported,
    // so expectedConversions is always 0.
    int64_t expectedConversions;

    std::string debugString() const {
      return fmt::format(
          "name={}, expectedConversions={}, key1Filter={}, key2Filter={}",
          name,
          expectedConversions,
          key1Filter ? key1Filter->serialize() : "null",
          key2Filter ? key2Filter->serialize() : "null");
    }
  };

  std::vector<TestCase> testCases;

  // Point + Point (using BytesValues): both columns converted.
  testCases.push_back(
      {"pointPoint",
       std::make_unique<BytesValues>(
           std::vector<std::string>{formatKey1(100)}, false),
       std::make_unique<BytesValues>(
           std::vector<std::string>{formatKey2(5)}, false),
       2});

  // Point + Range: both columns converted.
  testCases.push_back(
      {"pointRange",
       std::make_unique<BytesValues>(
           std::vector<std::string>{formatKey1(100)}, false),
       std::make_unique<BytesRange>(
           formatKey2(2), false, false, formatKey2(8), false, false, false),
       2});

  // Range + Point: only first column converted (range stops extraction).
  testCases.push_back(
      {"rangePoint",
       std::make_unique<BytesRange>(
           formatKey1(50), false, false, formatKey1(150), false, false, false),
       std::make_unique<BytesValues>(
           std::vector<std::string>{formatKey2(5)}, false),
       1});

  // Range + Range: only first column converted (range stops extraction).
  testCases.push_back(
      {"rangeRange",
       std::make_unique<BytesRange>(
           formatKey1(50), false, false, formatKey1(150), false, false, false),
       std::make_unique<BytesRange>(
           formatKey2(2), false, false, formatKey2(8), false, false, false),
       1});

  // No filter on key1, filter on key2: no conversion.
  testCases.push_back(
      {"noFilterOnKey1",
       nullptr,
       std::make_unique<BytesValues>(
           std::vector<std::string>{formatKey2(5)}, false),
       0});

  // Only filter on key1 (point): 1 conversion.
  testCases.push_back(
      {"onlyKey1Point",
       std::make_unique<BytesValues>(
           std::vector<std::string>{formatKey1(100)}, false),
       nullptr,
       1});

  // Only filter on key1 (range): 1 conversion.
  testCases.push_back(
      {"onlyKey1Range",
       std::make_unique<BytesRange>(
           formatKey1(50), false, false, formatKey1(150), false, false, false),
       nullptr,
       1});

  for (auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    if (testCase.key1Filter) {
      filters["key1"] = testCase.key1Filter->clone();
    }
    if (testCase.key2Filter) {
      filters["key2"] = testCase.key2Filter->clone();
    }

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    // For descending order, VARCHAR index filter conversion is not supported.
    const int64_t expectedConversions =
        ascending ? testCase.expectedConversions : 0;
    EXPECT_EQ(numConversionsWithIndex, expectedConversions);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

// Test with two timestamp key columns covering all filter-to-index conversion
// combinations. Includes nested complex types in the data column to ensure
// index filtering works correctly with complex nested data.
TEST_P(E2EIndexTest, twoTimestampKeysFilterCombinations) {
  constexpr size_t kNumRows = 5'000;
  constexpr size_t kRowsPerBatch = 500;
  constexpr int64_t kBaseSeconds = 1'000'000;

  // Include nested complex types similar to single key tests.
  auto rowType =
      ROW({"key1", "key2", "data", "nested"},
          {TIMESTAMP(),
           TIMESTAMP(),
           ARRAY(MAP(INTEGER(), VARCHAR())),
           ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}})});

  const bool ascending = isAscending();
  std::vector<RowVectorPtr> batches;
  for (size_t batchIdx = 0; batchIdx < kNumRows / kRowsPerBatch; ++batchIdx) {
    std::vector<Timestamp> key1Values(kRowsPerBatch);
    std::vector<Timestamp> key2Values(kRowsPerBatch);
    for (size_t i = 0; i < kRowsPerBatch; ++i) {
      const size_t globalIdx = batchIdx * kRowsPerBatch + i;
      const size_t effectiveIdx =
          ascending ? globalIdx : (kNumRows - 1 - globalIdx);
      // key1: base, base, ..., base (10x), base+1, base+1, ..., base+1 (10x)
      key1Values[i] = Timestamp(kBaseSeconds + effectiveIdx / 10, 0);
      // key2: 0, 1, 2, ..., 9 seconds (sorted within key1 group)
      key2Values[i] = Timestamp(effectiveIdx % 10, 0);
    }
    auto key1Vector = vectorMaker_->flatVector(key1Values);
    auto key2Vector = vectorMaker_->flatVector(key2Values);
    auto dataVector = makeFuzzedComplexVector(
        ARRAY(MAP(INTEGER(), VARCHAR())), kRowsPerBatch, batchIdx * 555);
    auto nestedVector = makeFuzzedComplexVector(
        ROW({{"inner", MAP(VARCHAR(), ARRAY(BIGINT()))}}),
        kRowsPerBatch,
        batchIdx * 666);
    batches.push_back(vectorMaker_->rowVector(
        {"key1", "key2", "data", "nested"},
        {key1Vector, key2Vector, dataVector, nestedVector}));
  }

  writeDataWithParam(batches, {"key1", "key2"});

  struct TestCase {
    std::string name;
    std::unique_ptr<Filter> key1Filter;
    std::unique_ptr<Filter> key2Filter;
    int64_t expectedConversions;

    std::string debugString() const {
      return fmt::format(
          "name={}, expectedConversions={}, key1Filter={}, key2Filter={}",
          name,
          expectedConversions,
          key1Filter ? key1Filter->toString() : "null",
          key2Filter ? key2Filter->toString() : "null");
    }
  };

  std::vector<TestCase> testCases;

  Timestamp key1Point(kBaseSeconds + 100, 0);
  Timestamp key1RangeLower(kBaseSeconds + 50, 0);
  Timestamp key1RangeUpper(kBaseSeconds + 150, 0);
  Timestamp key2Point(5, 0);
  Timestamp key2RangeLower(2, 0);
  Timestamp key2RangeUpper(8, 0);

  // Point + Point: both columns converted.
  testCases.push_back(
      {"pointPoint",
       std::make_unique<TimestampRange>(key1Point, key1Point, false),
       std::make_unique<TimestampRange>(key2Point, key2Point, false),
       2});

  // Point + Range: both columns converted.
  testCases.push_back(
      {"pointRange",
       std::make_unique<TimestampRange>(key1Point, key1Point, false),
       std::make_unique<TimestampRange>(key2RangeLower, key2RangeUpper, false),
       2});

  // Range + Point: only first column converted (range stops extraction).
  testCases.push_back(
      {"rangePoint",
       std::make_unique<TimestampRange>(key1RangeLower, key1RangeUpper, false),
       std::make_unique<TimestampRange>(key2Point, key2Point, false),
       1});

  // Range + Range: only first column converted (range stops extraction).
  testCases.push_back(
      {"rangeRange",
       std::make_unique<TimestampRange>(key1RangeLower, key1RangeUpper, false),
       std::make_unique<TimestampRange>(key2RangeLower, key2RangeUpper, false),
       1});

  // No filter on key1, filter on key2: no conversion.
  testCases.push_back(
      {"noFilterOnKey1",
       nullptr,
       std::make_unique<TimestampRange>(key2Point, key2Point, false),
       0});

  // Only filter on key1 (point): 1 conversion.
  testCases.push_back(
      {"onlyKey1Point",
       std::make_unique<TimestampRange>(key1Point, key1Point, false),
       nullptr,
       1});

  // Only filter on key1 (range): 1 conversion.
  testCases.push_back(
      {"onlyKey1Range",
       std::make_unique<TimestampRange>(key1RangeLower, key1RangeUpper, false),
       nullptr,
       1});

  for (auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    if (testCase.key1Filter) {
      filters["key1"] = testCase.key1Filter->clone();
    }
    if (testCase.key2Filter) {
      filters["key2"] = testCase.key2Filter->clone();
    }

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    EXPECT_EQ(numConversionsWithIndex, testCase.expectedConversions);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    E2EIndexTestSuite,
    E2EIndexTest,
    ::testing::Values(
        IndexEncodingParam::prefix(),
        IndexEncodingParam::prefix(1),
        IndexEncodingParam::prefix(1024),
        IndexEncodingParam::trivialZstd(),
        IndexEncodingParam::prefixDesc(),
        IndexEncodingParam::prefixDesc(1),
        IndexEncodingParam::prefixDesc(1024),
        IndexEncodingParam::trivialZstdDesc(),
        // With chunk index enabled.
        IndexEncodingParam::prefix(std::nullopt, true),
        IndexEncodingParam::trivialZstd(true),
        IndexEncodingParam::prefixDesc(std::nullopt, true),
        IndexEncodingParam::trivialZstdDesc(true),
        // With pinMetadata enabled.
        IndexEncodingParam::prefix(std::nullopt, false, true),
        IndexEncodingParam::prefixDesc(std::nullopt, false, true),
        IndexEncodingParam::prefix(std::nullopt, true, true),
        IndexEncodingParam::prefixDesc(std::nullopt, true, true),
        // With cache enabled.
        IndexEncodingParam::prefix(std::nullopt, false, false, true),
        IndexEncodingParam::prefixDesc(std::nullopt, false, false, true),
        // With cache and pinMetadata enabled.
        IndexEncodingParam::prefix(std::nullopt, false, true, true),
        IndexEncodingParam::prefixDesc(std::nullopt, false, true, true)),
    [](const ::testing::TestParamInfo<IndexEncodingParam>& info) {
      return info.param.name;
    });

// Fuzzer test that generates random wide and nested types with index columns
// at various positions. Tests different filter combinations including point
// lookups, ranges, and filters on non-index columns.

// Parameter struct for fuzzer tests including seed and sort order.
struct FuzzerTestParam {
  uint32_t seed;
  SortOrder sortOrder;
  EncodingType encodingType;
  CompressionType compressionType;
  std::optional<uint32_t> restartInterval;

  static FuzzerTestParam make(
      uint32_t seed,
      SortOrder sortOrder,
      EncodingType encodingType,
      CompressionType compressionType,
      std::optional<uint32_t> restartInterval = std::nullopt) {
    return FuzzerTestParam{
        seed, sortOrder, encodingType, compressionType, restartInterval};
  }

  bool isAscending() const {
    return sortOrder.ascending;
  }

  EncodingLayout makeEncodingLayout() const {
    return makeIndexEncodingLayout(
        encodingType, compressionType, restartInterval);
  }

  std::string name() const {
    std::string result = fmt::format("seed_{}", seed);
    result += isAscending() ? "_asc" : "_desc";
    result += encodingType == EncodingType::Prefix ? "_prefix" : "_trivial";
    if (compressionType == CompressionType::Zstd) {
      result += "_zstd";
    }
    if (restartInterval.has_value()) {
      result += fmt::format("_restart{}", restartInterval.value());
    }
    return result;
  }
};

class E2EIndexFuzzerTest
    : public E2EIndexTestBase,
      public ::testing::WithParamInterface<FuzzerTestParam> {
 protected:
  // Supported index column types.
  static constexpr std::array<TypeKind, 8> kIndexableTypes = {
      TypeKind::BOOLEAN,
      TypeKind::TINYINT,
      TypeKind::SMALLINT,
      TypeKind::INTEGER,
      TypeKind::BIGINT,
      TypeKind::REAL,
      TypeKind::DOUBLE,
      TypeKind::VARCHAR,
  };

  // Complex types for non-index columns.
  std::vector<TypePtr> getComplexTypes() {
    return {
        ARRAY(INTEGER()),
        ARRAY(VARCHAR()),
        MAP(INTEGER(), VARCHAR()),
        MAP(VARCHAR(), BIGINT()),
        ROW({{"nested_int", INTEGER()}, {"nested_str", VARCHAR()}}),
        ROW(
            {{"inner_array", ARRAY(DOUBLE())},
             {"inner_map", MAP(VARCHAR(), INTEGER())}}),
        ARRAY(ROW({{"x", INTEGER()}, {"y", DOUBLE()}})),
        MAP(VARCHAR(), ARRAY(INTEGER())),
    };
  }

  // Generates a random indexable type.
  TypePtr randomIndexableType(std::mt19937& rng) {
    std::uniform_int_distribution<size_t> dist(0, kIndexableTypes.size() - 1);
    return createScalarType(kIndexableTypes[dist(rng)]);
  }

  // Generates a random complex type.
  TypePtr randomComplexType(std::mt19937& rng) {
    auto complexTypes = getComplexTypes();
    std::uniform_int_distribution<size_t> dist(0, complexTypes.size() - 1);
    return complexTypes[dist(rng)];
  }

  // Generates sorted key column data.
  // For types with limited ranges (BOOLEAN, TINYINT, SMALLINT), values will
  // have duplicates since the number of rows exceeds the type's range. For
  // ascending order: values increase (0, 1, 2, ...) For descending order:
  // values decrease (numRows-1, numRows-2, ...)
  VectorPtr generateSortedKeyColumn(
      const TypePtr& type,
      size_t numRows,
      size_t batchOffset,
      size_t totalRows,
      bool ascending) {
    const auto effectiveOffset = [&](size_t i) -> size_t {
      const size_t globalIdx = batchOffset + i;
      return ascending ? globalIdx : (totalRows - 1 - globalIdx);
    };

    switch (type->kind()) {
      case TypeKind::BOOLEAN: {
        // Boolean only has 2 values, so we split rows into two halves.
        // Ascending: first half = false, second half = true.
        // Descending: first half = true, second half = false.
        std::vector<bool> values(numRows);
        for (size_t i = 0; i < numRows; ++i) {
          size_t globalIdx = batchOffset + i;
          bool isSecondHalf = globalIdx >= totalRows / 2;
          values[i] = ascending ? isSecondHalf : !isSecondHalf;
        }
        return vectorMaker_->flatVector(values);
      }
      case TypeKind::TINYINT: {
        constexpr int8_t kModulo = 126;
        std::vector<int8_t> values(numRows);
        for (size_t i = 0; i < numRows; ++i) {
          values[i] = static_cast<int8_t>(effectiveOffset(i) % kModulo);
        }
        return vectorMaker_->flatVector(values);
      }
      case TypeKind::SMALLINT: {
        constexpr int16_t kModulo = 32766;
        std::vector<int16_t> values(numRows);
        for (size_t i = 0; i < numRows; ++i) {
          values[i] = static_cast<int16_t>(effectiveOffset(i) % kModulo);
        }
        return vectorMaker_->flatVector(values);
      }
      case TypeKind::INTEGER: {
        std::vector<int32_t> values(numRows);
        for (size_t i = 0; i < numRows; ++i) {
          values[i] = static_cast<int32_t>(effectiveOffset(i));
        }
        return vectorMaker_->flatVector(values);
      }
      case TypeKind::BIGINT: {
        std::vector<int64_t> values(numRows);
        for (size_t i = 0; i < numRows; ++i) {
          values[i] = static_cast<int64_t>(effectiveOffset(i));
        }
        return vectorMaker_->flatVector(values);
      }
      case TypeKind::REAL: {
        std::vector<float> values(numRows);
        for (size_t i = 0; i < numRows; ++i) {
          values[i] = static_cast<float>(effectiveOffset(i));
        }
        return vectorMaker_->flatVector(values);
      }
      case TypeKind::DOUBLE: {
        std::vector<double> values(numRows);
        for (size_t i = 0; i < numRows; ++i) {
          values[i] = static_cast<double>(effectiveOffset(i));
        }
        return vectorMaker_->flatVector(values);
      }
      case TypeKind::VARCHAR: {
        std::vector<std::string> values(numRows);
        for (size_t i = 0; i < numRows; ++i) {
          values[i] = fmt::format("key_{:08d}", effectiveOffset(i));
        }
        return vectorMaker_->flatVector(values);
      }
      default:
        VELOX_UNREACHABLE("Unsupported type: {}", type->toString());
    }
  }

  // Generates a filter for the given type and filter kind.
  enum class FilterKind { POINT, RANGE, NONE };

  std::unique_ptr<Filter> generateFilter(
      const TypePtr& type,
      FilterKind kind,
      size_t totalRows,
      std::mt19937& rng) {
    if (kind == FilterKind::NONE) {
      return nullptr;
    }

    // Calculate effective range based on type to match data generation.
    // Data values are ((batchOffset + i) / dupFactor) % modulo for wrapped
    // types.
    size_t effectiveRange;
    switch (type->kind()) {
      case TypeKind::TINYINT:
        effectiveRange = 127; // Matches modulo in generateSortedKeyColumn.
        break;
      case TypeKind::SMALLINT:
        effectiveRange = 32000; // Matches modulo in generateSortedKeyColumn.
        break;
      default:
        effectiveRange = totalRows;
        break;
    }

    // Pick a value in the middle range for point lookups.
    // Pick a range covering ~20% of data for range filters.
    const size_t midPoint = effectiveRange / 2;
    const size_t rangeStart = effectiveRange / 4;
    const size_t rangeEnd = effectiveRange * 3 / 4;

    switch (type->kind()) {
      case TypeKind::BOOLEAN: {
        // Boolean only supports point lookup.
        std::uniform_int_distribution<int> dist(0, 1);
        return std::make_unique<BoolValue>(dist(rng) == 1, false);
      }
      case TypeKind::TINYINT: {
        if (kind == FilterKind::POINT) {
          auto val = static_cast<int64_t>(midPoint);
          return std::make_unique<BigintRange>(val, val, false);
        } else {
          return std::make_unique<BigintRange>(
              static_cast<int64_t>(rangeStart),
              static_cast<int64_t>(rangeEnd),
              false);
        }
      }
      case TypeKind::SMALLINT: {
        if (kind == FilterKind::POINT) {
          auto val = static_cast<int64_t>(midPoint);
          return std::make_unique<BigintRange>(val, val, false);
        } else {
          return std::make_unique<BigintRange>(
              static_cast<int64_t>(rangeStart),
              static_cast<int64_t>(rangeEnd),
              false);
        }
      }
      case TypeKind::INTEGER: {
        if (kind == FilterKind::POINT) {
          auto val = static_cast<int64_t>(midPoint);
          return std::make_unique<BigintRange>(val, val, false);
        } else {
          return std::make_unique<BigintRange>(
              static_cast<int64_t>(rangeStart),
              static_cast<int64_t>(rangeEnd),
              false);
        }
      }
      case TypeKind::BIGINT: {
        if (kind == FilterKind::POINT) {
          auto val = static_cast<int64_t>(midPoint);
          return std::make_unique<BigintRange>(val, val, false);
        } else {
          return std::make_unique<BigintRange>(
              static_cast<int64_t>(rangeStart),
              static_cast<int64_t>(rangeEnd),
              false);
        }
      }
      case TypeKind::REAL: {
        if (kind == FilterKind::POINT) {
          auto val = static_cast<float>(midPoint);
          return std::make_unique<FloatRange>(
              val, false, false, val, false, false, false);
        } else {
          return std::make_unique<FloatRange>(
              static_cast<float>(rangeStart),
              false,
              false,
              static_cast<float>(rangeEnd),
              false,
              false,
              false);
        }
      }
      case TypeKind::DOUBLE: {
        if (kind == FilterKind::POINT) {
          auto val = static_cast<double>(midPoint);
          return std::make_unique<DoubleRange>(
              val, false, false, val, false, false, false);
        } else {
          return std::make_unique<DoubleRange>(
              static_cast<double>(rangeStart),
              false,
              false,
              static_cast<double>(rangeEnd),
              false,
              false,
              false);
        }
      }
      case TypeKind::VARCHAR: {
        if (kind == FilterKind::POINT) {
          auto val = fmt::format("key_{:08d}", midPoint);
          return std::make_unique<BytesValues>(
              std::vector<std::string>{val}, false);
        } else {
          return std::make_unique<BytesRange>(
              fmt::format("key_{:08d}", rangeStart),
              false,
              false,
              fmt::format("key_{:08d}", rangeEnd),
              false,
              false,
              false);
        }
      }
      default:
        return nullptr;
    }
  }

  // Checks if a filter is a point filter (single value lookup).
  bool isPointFilter(const Filter* filter) {
    if (filter == nullptr) {
      return false;
    }
    switch (filter->kind()) {
      case velox::common::FilterKind::kBoolValue:
        return true;
      case velox::common::FilterKind::kBytesValues: {
        const auto* bytesValues = filter->as<BytesValues>();
        return bytesValues->values().size() == 1;
      }
      case velox::common::FilterKind::kBigintRange:
        return filter->as<BigintRange>()->isSingleValue();
      case velox::common::FilterKind::kDoubleRange: {
        const auto* range = filter->as<DoubleRange>();
        return !range->lowerUnbounded() && !range->upperUnbounded() &&
            !range->lowerExclusive() && !range->upperExclusive() &&
            range->lower() == range->upper();
      }
      case velox::common::FilterKind::kFloatRange: {
        const auto* range = filter->as<FloatRange>();
        return !range->lowerUnbounded() && !range->upperUnbounded() &&
            !range->lowerExclusive() && !range->upperExclusive() &&
            range->lower() == range->upper();
      }
      case velox::common::FilterKind::kTimestampRange:
        return filter->as<TimestampRange>()->isSingleValue();
      case velox::common::FilterKind::kBytesRange:
        return filter->as<BytesRange>()->isSingleValue();
      default:
        return false;
    }
  }

  // Checks if a filter is a range filter (non-point range).
  bool isRangeFilter(const Filter* filter) {
    if (filter == nullptr) {
      return false;
    }
    switch (filter->kind()) {
      case velox::common::FilterKind::kBigintRange:
      case velox::common::FilterKind::kDoubleRange:
      case velox::common::FilterKind::kFloatRange:
      case velox::common::FilterKind::kTimestampRange:
      case velox::common::FilterKind::kBytesRange:
        return !isPointFilter(filter);
      default:
        return false;
    }
  }

  // Generates a random filter kind.
  FilterKind randomFilterKind(std::mt19937& rng, bool allowNone = true) {
    std::uniform_int_distribution<int> dist(0, allowNone ? 2 : 1);
    int val = dist(rng);
    if (val == 0) {
      return FilterKind::POINT;
    }
    if (val == 1) {
      return FilterKind::RANGE;
    }
    return FilterKind::NONE;
  }
};

TEST_P(E2EIndexFuzzerTest, randomSchemaAndFilters) {
  const auto& param = GetParam();
  const uint32_t seed = param.seed;
  const bool ascending = param.isAscending();
  std::mt19937 rng(seed);

  // Configuration.
  constexpr size_t kNumRows = 20'000;
  constexpr size_t kRowsPerBatch = 500;
  constexpr size_t kMinColumns = 5;
  constexpr size_t kMaxColumns = 10;
  constexpr size_t kMinIndexColumns = 1;
  constexpr size_t kMaxIndexColumns = 3;

  // Generate random schema.
  std::uniform_int_distribution<size_t> numColsDist(kMinColumns, kMaxColumns);
  std::uniform_int_distribution<size_t> numIndexColsDist(
      kMinIndexColumns, kMaxIndexColumns);

  const size_t numColumns = numColsDist(rng);
  const size_t numIndexColumns = std::min(numIndexColsDist(rng), numColumns);

  // Randomly select positions for index columns.
  std::vector<size_t> allPositions(numColumns);
  std::iota(allPositions.begin(), allPositions.end(), 0);
  std::shuffle(allPositions.begin(), allPositions.end(), rng);

  std::vector<size_t> indexPositions(
      allPositions.begin(), allPositions.begin() + numIndexColumns);
  std::sort(indexPositions.begin(), indexPositions.end());

  std::set<size_t> indexPositionSet(
      indexPositions.begin(), indexPositions.end());

  // Generate column types and names.
  std::vector<std::string> columnNames;
  std::vector<TypePtr> columnTypes;
  std::vector<std::string> indexColumnNames;

  for (size_t i = 0; i < numColumns; ++i) {
    std::string name = fmt::format("col_{}", i);
    columnNames.push_back(name);

    if (indexPositionSet.count(i) > 0) {
      // Index column - must be indexable type.
      columnTypes.push_back(randomIndexableType(rng));
      indexColumnNames.push_back(name);
    } else {
      // Non-index column - can be complex type.
      std::uniform_int_distribution<int> typeChoice(0, 1);
      if (typeChoice(rng) == 0) {
        columnTypes.push_back(randomComplexType(rng));
      } else {
        columnTypes.push_back(randomIndexableType(rng));
      }
    }
  }

  const auto rowType = ROW(std::move(columnNames), std::move(columnTypes));

  SCOPED_TRACE(
      fmt::format(
          "seed={}, numColumns={}, numIndexColumns={}, indexColumns=[{}], rowType={}",
          seed,
          numColumns,
          numIndexColumns,
          fmt::join(indexColumnNames, ", "),
          rowType->toString()));

  // Generate data.
  std::vector<RowVectorPtr> batches;
  for (size_t batchIdx = 0; batchIdx < kNumRows / kRowsPerBatch; ++batchIdx) {
    std::vector<VectorPtr> columns;

    for (size_t colIdx = 0; colIdx < rowType->size(); ++colIdx) {
      const auto& colType = rowType->childAt(colIdx);

      if (indexPositionSet.count(colIdx) > 0) {
        // Generate sorted key column with strictly ordered unique values.
        columns.push_back(generateSortedKeyColumn(
            colType,
            kRowsPerBatch,
            batchIdx * kRowsPerBatch,
            kNumRows,
            ascending));
      } else {
        // Generate fuzzed non-index column.
        columns.push_back(makeFuzzedComplexVector(
            colType, kRowsPerBatch, batchIdx * 12345 + colIdx * 67890));
      }
    }

    batches.push_back(vectorMaker_->rowVector(rowType->names(), columns));
  }

  writeData(
      batches, indexColumnNames, param.makeEncodingLayout(), param.sortOrder);
  // Generate multiple filter combinations.
  constexpr int kNumFilterIterations = 10;

  for (int filterIter = 0; filterIter < kNumFilterIterations; ++filterIter) {
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;

    // Generate filters for index columns.
    // First index column must have a filter for any conversion to happen.
    int expectedConversions{0};
    bool rangeEncountered{false};

    for (size_t i = 0; i < indexColumnNames.size(); ++i) {
      const auto& colName = indexColumnNames[i];
      const auto& colType = rowType->findChild(colName);

      FilterKind kind;
      if (i == 0) {
        // First index column - always have a filter (50% chance each type).
        std::uniform_int_distribution<int> firstFilterDist(0, 1);
        kind =
            firstFilterDist(rng) == 0 ? FilterKind::POINT : FilterKind::RANGE;
      } else {
        // Subsequent index columns - 70% chance of filter.
        std::uniform_int_distribution<int> hasFilterDist(0, 9);
        if (hasFilterDist(rng) < 7) {
          kind = randomFilterKind(rng, /*allowNone=*/false);
        } else {
          kind = FilterKind::NONE;
        }
      }

      auto filter = generateFilter(colType, kind, kNumRows, rng);
      if (filter) {
        // Count expected conversions.
        // Note: For descending order, VARCHAR filter conversion is not
        // supported.
        if (!rangeEncountered) {
          const auto& colType = rowType->findChild(colName);
          const bool isVarchar = colType->kind() == TypeKind::VARCHAR;
          if (isVarchar && !ascending) {
            // VARCHAR with descending order stops the conversion chain.
            break;
          }
          ++expectedConversions;
          if (isRangeFilter(filter.get())) {
            rangeEncountered = true;
          }
        }
        filters[colName] = std::move(filter);
      } else {
        break;
      }
    }

    // Optionally add filters on non-index columns (should not affect index).
    std::uniform_int_distribution<int> nonIndexFilterDist(0, 2);
    for (size_t colIdx = 0; colIdx < rowType->size(); ++colIdx) {
      if (indexPositionSet.count(colIdx) > 0) {
        continue;
      }

      const auto& colName = rowType->nameOf(colIdx);
      const auto& colType = rowType->childAt(colIdx);

      // Only add filters to scalar non-index columns.
      if (!colType->isPrimitiveType()) {
        continue;
      }

      if (nonIndexFilterDist(rng) == 0) {
        auto filter = generateFilter(colType, FilterKind::RANGE, kNumRows, rng);
        if (filter) {
          filters[colName] = std::move(filter);
        }
      }
    }

    // Build filter description for SCOPED_TRACE.
    std::string filterDesc;
    for (const auto& [name, filter] : filters) {
      if (!filterDesc.empty()) {
        filterDesc += ", ";
      }
      filterDesc += fmt::format("{}={}", name, filter->toString());
    }

    SCOPED_TRACE(
        fmt::format(
            "filterIter={}, expectedConversions={}, filters=[{}]",
            filterIter,
            expectedConversions,
            filterDesc));

    // Test without random skip first.
    {
      auto reader = createReader();

      int64_t numConversionsWithIndex = 0;
      auto resultsWithIndex = readWithFilters(
          *reader,
          rowType,
          filters,
          /*indexEnabled=*/true,
          numConversionsWithIndex);

      int64_t numConversionsWithoutIndex = 0;
      auto resultsWithoutIndex = readWithFilters(
          *reader,
          rowType,
          filters,
          /*indexEnabled=*/false,
          numConversionsWithoutIndex);

      verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

      EXPECT_EQ(numConversionsWithIndex, expectedConversions);
      EXPECT_EQ(numConversionsWithoutIndex, 0);
    }

    // Test with random skip enabled (50% sample rate).
    // This verifies that random skip tracker is properly updated when rows
    // are skipped due to index bounds.
    {
      constexpr double kSampleRate = 0.5;
      const uint32_t randomSeed = seed + filterIter;

      SCOPED_TRACE(
          fmt::format(
              "randomSkip: sampleRate={}, seed={}", kSampleRate, randomSeed));

      auto readerWithIndex = createReader(kSampleRate, randomSeed);
      int64_t numConversionsWithIndex = 0;
      auto resultsWithIndex = readWithFilters(
          *readerWithIndex,
          rowType,
          filters,
          /*indexEnabled=*/true,
          numConversionsWithIndex);

      auto readerWithoutIndex = createReader(kSampleRate, randomSeed);
      int64_t numConversionsWithoutIndex = 0;
      auto resultsWithoutIndex = readWithFilters(
          *readerWithoutIndex,
          rowType,
          filters,
          /*indexEnabled=*/false,
          numConversionsWithoutIndex);

      verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

      EXPECT_EQ(numConversionsWithIndex, expectedConversions);
      EXPECT_EQ(numConversionsWithoutIndex, 0);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    E2EIndexFuzzerTestSuite,
    E2EIndexFuzzerTest,
    ::testing::Values(
        // Ascending order with Prefix encoding.
        FuzzerTestParam::make(
            42,
            SortOrder{.ascending = true},
            EncodingType::Prefix,
            CompressionType::Uncompressed),
        FuzzerTestParam::make(
            123,
            SortOrder{.ascending = true},
            EncodingType::Prefix,
            CompressionType::Uncompressed),
        FuzzerTestParam::make(
            456,
            SortOrder{.ascending = true},
            EncodingType::Prefix,
            CompressionType::Uncompressed),
        // Ascending order with Trivial encoding and Zstd compression.
        FuzzerTestParam::make(
            42,
            SortOrder{.ascending = true},
            EncodingType::Trivial,
            CompressionType::Zstd),
        FuzzerTestParam::make(
            123,
            SortOrder{.ascending = true},
            EncodingType::Trivial,
            CompressionType::Zstd),
        // Descending order with Prefix encoding.
        // Note: VARCHAR filter conversion is not supported for descending
        // order.
        FuzzerTestParam::make(
            42,
            SortOrder{.ascending = false},
            EncodingType::Prefix,
            CompressionType::Uncompressed),
        FuzzerTestParam::make(
            123,
            SortOrder{.ascending = false},
            EncodingType::Prefix,
            CompressionType::Uncompressed),
        // Descending order with Trivial encoding and Zstd compression.
        FuzzerTestParam::make(
            42,
            SortOrder{.ascending = false},
            EncodingType::Trivial,
            CompressionType::Zstd),
        // Prefix encoding with custom restart intervals.
        FuzzerTestParam::make(
            42,
            SortOrder{.ascending = true},
            EncodingType::Prefix,
            CompressionType::Uncompressed,
            1),
        FuzzerTestParam::make(
            42,
            SortOrder{.ascending = true},
            EncodingType::Prefix,
            CompressionType::Uncompressed,
            16),
        FuzzerTestParam::make(
            42,
            SortOrder{.ascending = true},
            EncodingType::Prefix,
            CompressionType::Uncompressed,
            1024)),
    [](const ::testing::TestParamInfo<FuzzerTestParam>& info) {
      return info.param.name();
    });

// Test with filters that are not eligible for index conversion.
// These include unsupported filter types, filters that allow nulls, and
// filters on non-index columns.
TEST_P(E2EIndexTest, filtersNotEligibleForIndexConversion) {
  constexpr size_t kNumRows = 2'000;
  constexpr size_t kRowsPerBatch = 200;

  const bool ascending = isAscending();

  // Test unsupported filter types and filters allowing nulls.
  {
    struct TestCase {
      std::string name;
      std::unique_ptr<Filter> filter;

      std::string debugString() const {
        return fmt::format("name={}", name);
      }
    };

    auto rowType = ROW({"key", "data"}, {BIGINT(), ARRAY(INTEGER())});

    std::vector<RowVectorPtr> batches;
    for (size_t batchIdx = 0; batchIdx < kNumRows / kRowsPerBatch; ++batchIdx) {
      std::vector<int64_t> keyValues(kRowsPerBatch);
      for (size_t i = 0; i < kRowsPerBatch; ++i) {
        const size_t globalIdx = batchIdx * kRowsPerBatch + i;
        const size_t effectiveIdx =
            ascending ? globalIdx : (kNumRows - 1 - globalIdx);
        keyValues[i] = static_cast<int64_t>(effectiveIdx);
      }
      auto keyVector = vectorMaker_->flatVector(keyValues);
      auto dataVector = makeFuzzedComplexVector(
          ARRAY(INTEGER()), kRowsPerBatch, batchIdx * 99999);
      batches.push_back(
          vectorMaker_->rowVector({"key", "data"}, {keyVector, dataVector}));
    }

    writeDataWithParam(batches, {"key"});

    std::vector<TestCase> testCases;
    // IsNotNull filter is NOT supported for index conversion.
    testCases.push_back({"isNotNull", std::make_unique<IsNotNull>()});
    // Filter with nullAllowed = true is NOT supported.
    testCases.push_back(
        {"rangeWithNullAllowed",
         std::make_unique<BigintRange>(100, 500, true)});

    for (auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());

      std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
      filters["key"] = testCase.filter->clone();

      auto reader = createReader();

      int64_t numConversionsWithIndex = 0;
      auto resultsWithIndex = readWithFilters(
          *reader,
          rowType,
          filters,
          /*indexEnabled=*/true,
          numConversionsWithIndex);

      int64_t numConversionsWithoutIndex = 0;
      auto resultsWithoutIndex = readWithFilters(
          *reader,
          rowType,
          filters,
          /*indexEnabled=*/false,
          numConversionsWithoutIndex);

      verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

      EXPECT_EQ(numConversionsWithIndex, 0);
      EXPECT_EQ(numConversionsWithoutIndex, 0);
    }
  }

  // Test filter on non-index column.
  {
    SCOPED_TRACE("filterOnNonIndexColumn");

    auto rowType =
        ROW({"key", "value", "data"}, {BIGINT(), BIGINT(), ARRAY(INTEGER())});

    std::vector<RowVectorPtr> batches;
    for (size_t batchIdx = 0; batchIdx < kNumRows / kRowsPerBatch; ++batchIdx) {
      std::vector<int64_t> keyValues(kRowsPerBatch);
      for (size_t i = 0; i < kRowsPerBatch; ++i) {
        const size_t globalIdx = batchIdx * kRowsPerBatch + i;
        const size_t effectiveIdx =
            ascending ? globalIdx : (kNumRows - 1 - globalIdx);
        keyValues[i] = static_cast<int64_t>(effectiveIdx);
      }
      auto keyVector = vectorMaker_->flatVector(keyValues);
      std::vector<int64_t> valueValues(kRowsPerBatch);
      for (size_t i = 0; i < kRowsPerBatch; ++i) {
        valueValues[i] = i % 100;
      }
      auto valueVector = vectorMaker_->flatVector(valueValues);
      auto dataVector = makeFuzzedComplexVector(
          ARRAY(INTEGER()), kRowsPerBatch, batchIdx * 14141);
      batches.push_back(vectorMaker_->rowVector(
          {"key", "value", "data"}, {keyVector, valueVector, dataVector}));
    }

    writeDataWithParam(batches, {"key"});

    // Filter on "value" which is NOT an index column.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["value"] = std::make_unique<BigintRange>(50, 60, false);

    auto reader = createReader();

    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *reader,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    EXPECT_EQ(numConversionsWithIndex, 0);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }

  // Test VARCHAR with descending order - not convertible for both point lookup
  // and range scan.
  if (!ascending) {
    SCOPED_TRACE("varcharDescendingNotConvertible");

    auto rowType = ROW({"key", "data"}, {VARCHAR(), ARRAY(INTEGER())});

    std::vector<RowVectorPtr> batches;
    for (size_t batchIdx = 0; batchIdx < kNumRows / kRowsPerBatch; ++batchIdx) {
      std::vector<std::string> keyValues(kRowsPerBatch);
      for (size_t i = 0; i < kRowsPerBatch; ++i) {
        const size_t globalIdx = batchIdx * kRowsPerBatch + i;
        const size_t effectiveIdx = kNumRows - 1 - globalIdx;
        keyValues[i] = fmt::format("key_{:08d}", effectiveIdx);
      }
      auto keyVector = vectorMaker_->flatVector(keyValues);
      auto dataVector = makeFuzzedComplexVector(
          ARRAY(INTEGER()), kRowsPerBatch, batchIdx * 77777);
      batches.push_back(
          vectorMaker_->rowVector({"key", "data"}, {keyVector, dataVector}));
    }

    writeDataWithParam(batches, {"key"});

    struct TestCase {
      std::string name;
      std::unique_ptr<Filter> filter;

      std::string debugString() const {
        return fmt::format("name={}", name);
      }
    };

    std::vector<TestCase> testCases;
    // Point lookup (BytesValues) with descending order is NOT supported.
    testCases.push_back(
        {"pointLookupDescending",
         std::make_unique<BytesValues>(
             std::vector<std::string>{"key_00000100", "key_00000200"}, false)});
    // Range scan (BytesRange) with descending order is NOT supported.
    testCases.push_back(
        {"rangeScanDescending",
         std::make_unique<BytesRange>(
             "key_00000100",
             false,
             false,
             "key_00000500",
             false,
             false,
             false)});

    for (auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());

      std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
      filters["key"] = testCase.filter->clone();

      auto reader = createReader();

      int64_t numConversionsWithIndex = 0;
      auto resultsWithIndex = readWithFilters(
          *reader,
          rowType,
          filters,
          /*indexEnabled=*/true,
          numConversionsWithIndex);

      int64_t numConversionsWithoutIndex = 0;
      auto resultsWithoutIndex = readWithFilters(
          *reader,
          rowType,
          filters,
          /*indexEnabled=*/false,
          numConversionsWithoutIndex);

      verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

      EXPECT_EQ(numConversionsWithIndex, 0);
      EXPECT_EQ(numConversionsWithoutIndex, 0);
    }
  }
}

// Test that random skip (random sampling) produces the same results with and
// without cluster index enabled.
// without cluster index enabled. This verifies that the random skip tracker
// is properly updated when rows are skipped due to index bounds.
TEST_P(E2EIndexTest, randomSkipWithIndex) {
  constexpr int64_t kMinKey = 1'000;
  constexpr size_t kNumRows = 10'000;
  constexpr size_t kRowsPerBatch = 1'000;

  auto rowType =
      ROW({"key", "data", "nested"},
          {BIGINT(),
           ARRAY(INTEGER()),
           ROW({{"nested", MAP(INTEGER(), VARCHAR())}})});

  // Use the shared generateKeyData() utility with the startValue being kMinKey.
  auto batches = generateKeyData(BIGINT(), kNumRows, kRowsPerBatch, kMinKey);
  writeDataWithParam(batches, {"key"});

  struct TestCase {
    std::string name;
    int64_t filterLower;
    int64_t filterUpper;
    double sampleRate;
    uint32_t randomSeed;

    std::string debugString() const {
      return fmt::format(
          "name={}, filterLower={}, filterUpper={}, sampleRate={}, randomSeed={}",
          name,
          filterLower,
          filterUpper,
          sampleRate,
          randomSeed);
    }
  };

  std::vector<TestCase> testCases = {
      // Various sample rates with middle range filter.
      {"sampleRate50pctMiddleRange", 3'000, 7'000, 0.5, 42},
      {"sampleRate25pctMiddleRange", 3'000, 7'000, 0.25, 42},
      {"sampleRate75pctMiddleRange", 3'000, 7'000, 0.75, 42},
      {"sampleRate10pctMiddleRange", 3'000, 7'000, 0.1, 42},

      // Different filter ranges with 50% sample rate.
      {"sampleRate50pctFromMin", kMinKey, 5'000, 0.5, 42},
      {"sampleRate50pctToMax", 5'000, kMinKey + kNumRows - 1, 0.5, 42},
      {"sampleRate50pctFullRange", kMinKey, kMinKey + kNumRows - 1, 0.5, 42},
      {"sampleRate50pctNarrowRange", 4'500, 5'500, 0.5, 42},

      // Point lookup with sampling (edge case).
      {"sampleRate50pctPointLookup", 5'000, 5'000, 0.5, 42},

      // Different random seeds to ensure determinism.
      {"sampleRate50pctSeed123", 3'000, 7'000, 0.5, 123},
      {"sampleRate50pctSeed456", 3'000, 7'000, 0.5, 456},
      {"sampleRate50pctSeed789", 3'000, 7'000, 0.5, 789},

      // Very low sample rate (more skipping).
      {"sampleRate5pctMiddleRange", 3'000, 7'000, 0.05, 42},
      {"sampleRate1pctMiddleRange", 3'000, 7'000, 0.01, 42},

      // Very high sample rate (less skipping).
      {"sampleRate95pctMiddleRange", 3'000, 7'000, 0.95, 42},
      {"sampleRate99pctMiddleRange", 3'000, 7'000, 0.99, 42},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["key"] = std::make_unique<BigintRange>(
        testCase.filterLower, testCase.filterUpper, false);

    // Read with index enabled and random skip.
    // Create a new reader for each read since random skip is configured in
    // ReaderOptions.
    auto readerWithIndex =
        createReader(testCase.sampleRate, testCase.randomSeed);
    int64_t numConversionsWithIndex = 0;
    auto resultsWithIndex = readWithFilters(
        *readerWithIndex,
        rowType,
        filters,
        /*indexEnabled=*/true,
        numConversionsWithIndex);

    // Read without index but with same random skip configuration.
    // Using the same seed ensures deterministic comparison.
    auto readerWithoutIndex =
        createReader(testCase.sampleRate, testCase.randomSeed);
    int64_t numConversionsWithoutIndex = 0;
    auto resultsWithoutIndex = readWithFilters(
        *readerWithoutIndex,
        rowType,
        filters,
        /*indexEnabled=*/false,
        numConversionsWithoutIndex);

    // Results should be identical regardless of index usage.
    verifyResultsEqual(resultsWithIndex, resultsWithoutIndex);

    // Index should be used when enabled.
    EXPECT_EQ(numConversionsWithIndex, 1);
    EXPECT_EQ(numConversionsWithoutIndex, 0);
  }
}

// Test that filters converted to index bounds are properly restored after each
// row reader completes. This is important for multiple split execution modes
// where the scan spec is shared across split readers.
TEST_P(E2EIndexTest, filterRestorationAcrossMultipleSplits) {
  constexpr int64_t kMinKey = 1'000;
  constexpr size_t kNumRows = 10'000;
  constexpr size_t kRowsPerBatch = 1'000;

  auto rowType =
      ROW({"key", "data", "nested"},
          {BIGINT(),
           ARRAY(INTEGER()),
           ROW({{"nested", MAP(INTEGER(), VARCHAR())}})});

  // Use the shared generateKeyData() utility with the startValue being kMinKey.
  auto batches = generateKeyData(BIGINT(), kNumRows, kRowsPerBatch, kMinKey);
  writeDataWithParam(batches, {"key"});

  // Create a shared scan spec with a filter that will be converted to index
  // bounds.
  auto scanSpec = std::make_shared<ScanSpec>("root");
  for (size_t i = 0; i < rowType->size(); ++i) {
    auto* childSpec =
        scanSpec->addField(rowType->nameOf(i), static_cast<column_index_t>(i));
    addChildSpecs(childSpec, rowType->childAt(i));
  }
  scanSpec->childByName("key")->setFilter(
      std::make_unique<BigintRange>(3'000, 7'000, false));

  auto* keySpec = scanSpec->childByName("key");
  auto reader = createReader();

  RowReaderOptions rowReaderOptions;
  rowReaderOptions.setRequestedType(rowType);
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setIndexEnabled(true);

  auto rowReader = reader->createRowReader(rowReaderOptions);

  // Verify filter is removed after row reader init.
  EXPECT_EQ(keySpec->filter(), nullptr);

  // Read all data to completion.
  VectorPtr result = BaseVector::create(rowType, 1, leafPool_.get());
  while (rowReader->next(1'000, result) > 0) {
  }

  // Verify filter is restored after row reader completes.
  ASSERT_NE(keySpec->filter(), nullptr);
  EXPECT_EQ(keySpec->filter()->kind(), FilterKind::kBigintRange);
}

// Verifies that createIndexReader() throws a clear error (not SIGSEGV) when
// the Nimble file has no cluster index — both with data and with an empty file.
TEST_F(E2EIndexTestBase, createIndexReaderWithoutClusterIndex) {
  auto rowType = ROW({"a", "b"}, {VARCHAR(), INTEGER()});
  auto batch = vectorMaker_->rowVector(
      {"a", "b"},
      {vectorMaker_->flatVector<StringView>({"foo", "bar", "baz"}),
       vectorMaker_->flatVector<int32_t>({1, 2, 3})});

  // Test with both non-empty and empty files.
  for (bool writeRows : {true, false}) {
    SCOPED_TRACE(fmt::format("writeRows={}", writeRows));
    sinkData_.clear();
    auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
    WriterOptions options;
    Writer writer(
        rowType, std::move(writeFile), *rootPool_, std::move(options));
    if (writeRows) {
      writer.write(batch);
    }
    writer.close();

    auto reader = createReader();
    RowReaderOptions rowReaderOpts;
    rowReaderOpts.setRequestedType(rowType);
    auto scanSpec = createScanSpec(rowType, {});
    rowReaderOpts.setScanSpec(std::move(scanSpec));

    // Files without cluster index return nullptr instead of throwing.
    EXPECT_EQ(reader->createIndexReader(rowReaderOpts), nullptr);
  }
}

TEST_F(E2EIndexTestBase, createIndexReaderRejectsHiddenKeyReadAccess) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), INTEGER()});
  auto storedType = ROW({"value"}, {INTEGER()});
  auto batch = vectorMaker_->rowVector(
      {"key", "value"},
      {vectorMaker_->flatVector<int64_t>({1, 2, 3}),
       vectorMaker_->flatVector<int32_t>({10, 20, 30})});

  sinkData_.clear();
  auto writeFile = std::make_unique<InMemoryWriteFile>(&sinkData_);
  WriterOptions options;
  options.clusterIndexConfig =
      ClusterIndexConfigBuilder{}
          .withKeyColumns({"key"})
          .withSortOrders({SortOrder{.ascending = true}})
          .withEnforceKeyOrder(true)
          .build();
  options.experimentalOmitClusterIndexKeyColumnStorage = true;

  Writer writer(rowType, std::move(writeFile), *rootPool_, std::move(options));
  writer.write(batch);
  writer.close();

  // Hidden key columns are still usable for cluster index bounds.
  {
    auto reader = createReader();
    RowReaderOptions rowReaderOpts;
    rowReaderOpts.setRequestedType(storedType);
    rowReaderOpts.setScanSpec(createScanSpec(storedType, {}));
    auto indexReader = reader->createIndexReader(rowReaderOpts);
    ASSERT_NE(indexReader, nullptr);

    auto bound = vectorMaker_->rowVector(
        {"key"}, {vectorMaker_->flatVector<int64_t>({2})});
    serializer::IndexBounds bounds;
    bounds.indexColumns = {"key"};
    bounds.set(
        serializer::IndexBound{bound, /*inclusive=*/true},
        serializer::IndexBound{bound, /*inclusive=*/true});
    indexReader->startLookup(bounds, {});

    ASSERT_TRUE(indexReader->hasNext());
    auto result = indexReader->next(10);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->output->size(), 1);
    EXPECT_EQ(
        result->output->childAt(0)->asFlatVector<int32_t>()->valueAt(0), 20);
    EXPECT_FALSE(indexReader->hasNext());
  }

  struct Scenario {
    std::string name;
    RowTypePtr requestedType;
    bool remainingFilter;
    std::string expectedMessage;
  };
  const std::vector<Scenario> scenarios{
      {
          // Hidden key columns cannot be read from normal data streams.
          .name = "projection",
          .requestedType = rowType,
          .remainingFilter = false,
          .expectedMessage =
              "Cluster index key column 'key' cannot be projected",
      },
      {
          // Remaining filters require normal data streams; index bounds are
          // the supported hidden-key filter path.
          .name = "remainingFilter",
          .requestedType = storedType,
          .remainingFilter = true,
          .expectedMessage =
              "Cluster index key column 'key' cannot be used as a remaining scan filter",
      },
  };

  for (const auto& scenario : scenarios) {
    SCOPED_TRACE(fmt::format("scenario={}", scenario.name));
    auto reader = createReader();
    RowReaderOptions rowReaderOpts;
    rowReaderOpts.setRequestedType(scenario.requestedType);
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    if (scenario.remainingFilter) {
      filters["key"] = std::make_unique<BigintRange>(1, 3, false);
    }
    rowReaderOpts.setScanSpec(createScanSpec(rowType, filters));

    NIMBLE_ASSERT_THROW(
        reader->createIndexReader(rowReaderOpts), scenario.expectedMessage);
  }
}

// TODO: add schema revolution tests like column renaming to make sure selective
// reader can handle properly for the index columns.
} // namespace facebook::nimble::test
