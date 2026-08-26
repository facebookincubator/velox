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

#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <span>

#include <fmt/format.h>

#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileHandle.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/file/tests/TestUtils.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/dwio/common/CachedBufferedInput.h"
#include "velox/dwio/common/DirectBufferedInput.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/common/TypeUtils.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/velox/ChunkedStream.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/dwio/nimble/velox/SharedDictionaryConfig.h"
#include "velox/dwio/nimble/velox/VeloxReader.h"
#include "velox/dwio/nimble/velox/tests/SharedDictionaryTestUtils.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

using test::makeSharedDictionaryInput;
using test::SharedDictionarySource;
using test::SharedDictionaryTestResolver;
using test::sharedDictionaryValueUniverse;
using test::sharedDictionaryWriterOptions;
using test::writeWithRandomStripes;

struct NullableArrayData {
  std::optional<std::vector<std::optional<int64_t>>> value;

  NullableArrayData(std::nullopt_t) : value(std::nullopt) {}

  NullableArrayData(std::initializer_list<int64_t> values)
      : value(std::in_place) {
    value->reserve(values.size());
    for (auto element : values) {
      value->emplace_back(element);
    }
  }

  operator std::optional<std::vector<std::optional<int64_t>>>() const {
    return value;
  }
};

struct NullableMapData {
  std::optional<std::vector<std::pair<int64_t, std::optional<int64_t>>>> value;

  NullableMapData(std::nullopt_t) : value(std::nullopt) {}

  NullableMapData(std::initializer_list<std::pair<int64_t, int64_t>> values)
      : value(std::in_place) {
    value->reserve(values.size());
    for (const auto& [key, mapped] : values) {
      value->emplace_back(key, mapped);
    }
  }

  operator std::optional<
      std::vector<std::pair<int64_t, std::optional<int64_t>>>>() const {
    return value;
  }
};

enum FilterType { kNone, kKeep, kDrop };
auto format_as(FilterType filterType) {
  return fmt::underlying(filterType);
}

struct TestParam {
  bool stringDecoderZeroCopy;
  bool enableCache;
  bool pinMetadata{false};

  std::string debugString() const {
    return fmt::format(
        "passStringBuffers_{}_cache_{}_pinMetadata_{}",
        stringDecoderZeroCopy,
        enableCache,
        pinMetadata);
  }
};

// This test suite covers the basic and mostly single batch test cases, as well
// as some corner cases that are hard to cover in randomized tests.  We rely on
// E2EFilterTest for more comprehensive tests with multi stripes and multi
// filters.
class SelectiveNimbleReaderTest
    : public ::testing::Test,
      public velox::test::VectorTestBase,
      public ::testing::WithParamInterface<TestParam> {
 public:
  static std::vector<TestParam> getTestParams() {
    std::vector<TestParam> params;
    for (bool passStringBuffers : {false, true}) {
      for (bool cache : {false, true}) {
        for (bool pin : {false, true}) {
          params.emplace_back(TestParam{passStringBuffers, cache, pin});
        }
      }
    }
    return params;
  }

 protected:
  static void SetUpTestCase() {
    if (!memory::MemoryManager::testInstance()) {
      memory::MemoryManager::testingSetInstance(
          velox::memory::MemoryManager::Options{});
    }
    registerSelectiveNimbleReaderFactory();
  }

  static void TearDownTestCase() {
    unregisterSelectiveNimbleReaderFactory();
  }

  void SetUp() override {
    scanTracker_ =
        std::make_shared<cache::ScanTracker>("testTracker", nullptr, 256 << 10);
    ioExecutor_ = std::make_unique<folly::CPUThreadPoolExecutor>(10);
    if (GetParam().enableCache) {
      allocator_ = std::make_shared<memory::MallocAllocator>(
          memory::MemoryAllocator::Options{
              .capacity = 512 << 20, .reservationByteLimit = 0});
      cache_ = cache::AsyncDataCache::create(allocator_.get());
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

  memory::MemoryPool* rootPool() {
    return rootPool_.get();
  }

  bool stringDecoderZeroCopy() const {
    return GetParam().stringDecoderZeroCopy;
  }

  struct Readers {
    std::unique_ptr<dwio::common::Reader> reader;
    std::unique_ptr<dwio::common::RowReader> rowReader;
  };

  Readers makeReaders(
      const RowVectorPtr& expected,
      const std::string& file,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      bool stringDecoderZeroCopy,
      bool preserveFlatMapsInMemory = false,
      bool lazyColumnIo = false,
      std::shared_ptr<const ExternalDictionaryResolver>
          externalDictionaryResolver = nullptr) {
    auto readFile = std::make_shared<InMemoryReadFile>(file);
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions options(pool());
    options.setDataIoStats(dataIoStats_);
    options.setMetadataIoStats(metadataIoStats_);
    options.setScanSpec(scanSpec);
    options.setPinMetadata(GetParam().pinMetadata);
    options.setCacheMetadata(GetParam().enableCache);
    options.setFilePreloadThreshold(0);
    if (externalDictionaryResolver != nullptr) {
      auto nimbleOptions = std::make_shared<NimbleReaderOptions>();
      nimbleOptions->externalDictionaryResolver =
          std::move(externalDictionaryResolver);
      options.setFormatSpecificOptions(std::move(nimbleOptions));
    }
    std::unique_ptr<dwio::common::BufferedInput> input;
    auto& ids = fileIds();
    const auto readerId = readerIdCounter_++;
    StringIdLease fileId(ids, fmt::format("testFile_{}", readerId));
    StringIdLease groupId(ids, fmt::format("testGroup_{}", readerId));
    io::ReaderOptions ioReaderOpts(pool());
    ioReaderOpts.setDataIoStats(dataIoStats_);
    ioReaderOpts.setMetadataIoStats(metadataIoStats_);
    if (cache_ != nullptr) {
      input = std::make_unique<dwio::common::CachedBufferedInput>(
          readFile,
          dwio::common::MetricsLog::voidLog(),
          std::move(fileId),
          cache_.get(),
          scanTracker_,
          std::move(groupId),
          dataIoStats_,
          nullptr,
          ioExecutor_.get(),
          ioReaderOpts);
    } else {
      input = std::make_unique<dwio::common::DirectBufferedInput>(
          readFile,
          dwio::common::MetricsLog::voidLog(),
          std::move(fileId),
          scanTracker_,
          std::move(groupId),
          dataIoStats_,
          nullptr,
          ioExecutor_.get(),
          ioReaderOpts);
    }
    Readers readers;
    readers.reader = factory->createReader(std::move(input), options);
    EXPECT_EQ(readers.reader->numberOfRows(), expected->size());
    auto type = asRowType(expected->type());
    dwio::common::typeutils::checkTypeCompatibility(
        *readers.reader->rowType(), *type);
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    rowOptions.setRequestedType(type);
    rowOptions.setPreserveFlatMapsInMemory(preserveFlatMapsInMemory);
    rowOptions.setStringDecoderZeroCopy(stringDecoderZeroCopy);
    rowOptions.setLazyColumnIo(lazyColumnIo);
    readers.rowReader = readers.reader->createRowReader(rowOptions);
    return readers;
  }

  Readers makeReaders(
      const RowVectorPtr& input,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      bool stringDecoderZeroCopy) {
    return makeReaders(
        input,
        test::createNimbleFile(*rootPool(), input),
        scanSpec,
        stringDecoderZeroCopy);
  }

  uint32_t readerIdCounter_{0};

  // Cache infrastructure (only initialized when enableCache is true).
  std::shared_ptr<memory::MallocAllocator> allocator_;
  std::shared_ptr<cache::AsyncDataCache> cache_;
  std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<io::IoStatistics> indexIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<cache::ScanTracker> scanTracker_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> ioExecutor_;

  template <typename F>
  void validate(
      const RowVector& input,
      dwio::common::RowReader& rowReader,
      int batchSize,
      F&& filter) {
    validate(input, rowReader, batchSize, -1, std::forward<F>(filter));
  }

  template <typename F>
  void validate(
      const RowVector& input,
      dwio::common::RowReader& rowReader,
      int batchSize,
      int dropColumn,
      F&& filter) {
    auto result =
        BaseVector::create(rowType(input.type(), dropColumn), 0, pool());
    int numScanned = 0;
    int i = 0;
    while (numScanned < input.size()) {
      numScanned += rowReader.next(batchSize, result);
      result->validate();
      for (int j = 0; j < result->size(); ++j) {
        for (;;) {
          ASSERT_LT(i, input.size());
          if (filter(i)) {
            break;
          }
          VLOG(1) << i << ": " << input.toString(i);
          ++i;
        }
        VLOG(1) << i << ": " << input.toString(i) << " " << j << ": "
                << result->toString(j);
        if (dropColumn < 0) {
          ASSERT_TRUE(result->equalValueAt(&input, j, i));
        } else {
          auto* resultRow = result->asUnchecked<RowVector>();
          for (int k = 0, kk = 0; k < resultRow->childrenSize(); ++k) {
            if (k != dropColumn) {
              auto& expected = input.childAt(k);
              auto& actual = resultRow->childAt(kk++);
              ASSERT_TRUE(actual->equalValueAt(expected.get(), j, i));
            }
          }
        }
        ++i;
      }
    }
    while (i < input.size()) {
      VLOG(1) << i << ": " << input.toString(i);
      ASSERT_FALSE(filter(i));
      ++i;
    }
    ASSERT_EQ(numScanned, input.size());
    ASSERT_EQ(0, rowReader.next(1, result));
  }

  template <typename MakeData, typename ValidationFilter>
  void runTestCase(
      MakeData&& makeData,
      const common::Filter& filter,
      int batchSize,
      ValidationFilter&& validationFilter,
      bool stringDecoderZeroCopy,
      const std::function<VectorPtr(bool)>& makeExpectedData = nullptr) {
    for (bool hasNulls : {false, true}) {
      auto data = makeData(hasNulls);
      auto input = makeRowVector({data, data});
      auto file = test::createNimbleFile(*rootPool(), input);
      RowVectorPtr expected;
      if (makeExpectedData) {
        auto expectedData = makeExpectedData(hasNulls);
        expected = makeRowVector({expectedData, expectedData});
      } else {
        expected = input;
      }
      for (auto filterType : {kNone, kKeep, kDrop}) {
        SCOPED_TRACE(
            fmt::format("hasNulls={} filterType={}", hasNulls, filterType));
        int dropColumn = -1;
        auto scanSpec = std::make_shared<common::ScanSpec>("root");
        scanSpec->addAllChildFields(*expected->type());
        if (filterType != kNone) {
          common::ScanSpec* c0 = scanSpec->childByName("c0");
          c0->setFilter(filter.clone());
          if (filterType == kDrop) {
            c0->setProjectOut(false);
            c0->setChannel(common::ScanSpec::kNoChannel);
            scanSpec->childByName("c1")->setChannel(0);
            dropColumn = 0;
          }
        }
        auto readers =
            makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
        validate(
            *expected, *readers.rowReader, batchSize, dropColumn, [&](auto i) {
              return filterType == kNone || validationFilter(data, i);
            });
      }
    }
  }

  // Verifies that the first scalar column in the file uses the expected
  // encoding type on disk. For nullable columns, the data stream is at
  // child offset 1 (offset 0 is the nulls stream).
  void verifyEncodingOnDisk(
      const std::string& file,
      EncodingType expectedEncodingType,
      bool isNullable = true) {
    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
    auto tablet = TabletReader::create(
        readFile, pool(), test::makeTestTabletOptions(pool()));
    auto section = tablet->loadOptionalSection(std::string(kSchemaSection));
    ASSERT_TRUE(section.has_value());
    auto schema = SchemaDeserializer::deserialize(section->content().data());
    auto& scalarNode = schema->asRow().childAt(0)->asScalar();

    for (auto i = 0; i < tablet->stripeCount(); ++i) {
      auto stripeIdentifier = tablet->stripeIdentifier(i);
      std::vector<uint32_t> identifiers{scalarNode.scalarDescriptor().offset()};
      auto streams = tablet->load(stripeIdentifier, identifiers);

      InMemoryChunkedStream chunkedStream{*pool(), std::move(streams[0])};
      ASSERT_TRUE(chunkedStream.hasNext());
      auto capture = EncodingLayoutCapture::capture(
          chunkedStream.nextChunk(), Encoding::Options{});
      if (isNullable) {
        // Nullable encoding wraps the data encoding. The data child is at
        // index 1 (index 0 is the nulls bool stream).
        ASSERT_EQ(EncodingType::Nullable, capture.encodingType())
            << "Stripe " << i;
        ASSERT_TRUE(capture.child(1).has_value()) << "Stripe " << i;
        EXPECT_EQ(expectedEncodingType, capture.child(1)->encodingType())
            << "Stripe " << i;
      } else {
        EXPECT_EQ(expectedEncodingType, capture.encodingType())
            << "Stripe " << i;
      }
    }
  }

  void checkArrayWithOffsets(
      const std::vector<std::optional<std::vector<std::optional<int64_t>>>>&
          data,
      const std::vector<bool>& filter,
      const std::vector<int>& readSizes,
      bool stringDecoderZeroCopy,
      std::optional<int> maxArrayElementsCount = std::nullopt,
      bool filterAfterRead = false) {
    RowVectorPtr vector;
    if (filter.empty()) {
      vector = makeRowVector({makeNullableArrayVector<int64_t>(data)});
    } else if (filterAfterRead) {
      vector = makeRowVector({
          makeNullableArrayVector<int64_t>(data),
          makeRowVector(
              {makeConstant<int8_t>(0, data.size())},
              [&](auto i) { return !filter[i]; }),
      });
    } else {
      vector = makeRowVector({
          makeNullableArrayVector<int64_t>(data),
          makeFlatVector<bool>(filter),
      });
    }
    auto rowType = asRowType(vector->type());
    auto file = test::createNimbleFile(
        *rootPool(), vector, {.dictionaryArrayColumns = {"c0"}});
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    if (!filter.empty()) {
      if (filterAfterRead) {
        scanSpec->childByName("c0")->setFilter(
            std::make_unique<common::IsNotNull>());
        scanSpec->childByName("c1")->setFilter(
            std::make_unique<common::IsNotNull>());
      } else {
        scanSpec->childByName("c1")->setFilter(
            std::make_unique<common::BoolValue>(true, false));
      }
    }
    if (maxArrayElementsCount.has_value()) {
      scanSpec->childByName("c0")->setMaxArrayElementsCount(
          *maxArrayElementsCount);
    }
    auto readers = makeReaders(vector, file, scanSpec, stringDecoderZeroCopy);
    auto result = BaseVector::create(rowType, 0, pool());
    int totalScanned = 0;
    std::vector<bool> selected(data.size(), true);
    if (!filter.empty()) {
      selected = filter;
    }
    for (int readSize : readSizes) {
      ASSERT_EQ(readers.rowReader->next(readSize, result), readSize);
      auto begin = selected.begin() + totalScanned;
      ASSERT_EQ(result->size(), std::accumulate(begin, begin + readSize, 0));
      auto& c0 = result->loadedVector()->asChecked<RowVector>()->childAt(0);
      int resultIndex = 0;
      int wrappedIndex = -1;
      int offset = 0;
      int lastSize = 0;
      int lastExpected = -1;
      for (int i = 0; i < readSize; ++i) {
        if (!selected[totalScanned + i]) {
          continue;
        }
        auto& expected = data[totalScanned + i];
        if (!expected.has_value()) {
          ASSERT_TRUE(c0->isNullAt(resultIndex));
        } else {
          if (lastExpected < 0 || data[lastExpected] != expected) {
            ++wrappedIndex;
            lastExpected = totalScanned + i;
            offset += lastSize;
          }
          folly::Range<const std::optional<int64_t>*> expectedRange;
          if (maxArrayElementsCount.has_value()) {
            expectedRange = {
                expected->data(),
                std::min<size_t>(
                    expected->size(), maxArrayElementsCount.value())};
          } else {
            expectedRange = *expected;
          }
          ASSERT_FALSE(c0->isNullAt(resultIndex));
          ASSERT_EQ(c0->wrappedIndex(resultIndex), wrappedIndex);
          auto* alphabet = c0->wrappedVector()->asChecked<ArrayVector>();
          ASSERT_EQ(alphabet->offsetAt(wrappedIndex), offset);
          ASSERT_EQ(alphabet->sizeAt(wrappedIndex), expectedRange.size());
          lastSize = alphabet->sizeAt(wrappedIndex);
          auto* elements =
              alphabet->elements()->asChecked<FlatVector<int64_t>>();
          for (int j = 0; j < expectedRange.size(); ++j) {
            if (expectedRange[j].has_value()) {
              ASSERT_FALSE(elements->isNullAt(offset + j));
              ASSERT_EQ(elements->valueAt(offset + j), expectedRange[j]);
            } else {
              ASSERT_TRUE(elements->isNullAt(offset + j));
            }
          }
        }
        ++resultIndex;
      }
      totalScanned += readSize;
    }
    ASSERT_EQ(totalScanned, data.size());
    ASSERT_EQ(readers.rowReader->next(1, result), 0);
  }

  void checkArrayWithOffsets(
      std::initializer_list<NullableArrayData> data,
      const std::vector<bool>& filter,
      const std::vector<int>& readSizes,
      bool stringDecoderZeroCopy,
      std::optional<int> maxArrayElementsCount = std::nullopt,
      bool filterAfterRead = false) {
    std::vector<std::optional<std::vector<std::optional<int64_t>>>> converted;
    converted.reserve(data.size());
    for (const auto& item : data) {
      converted.push_back(item);
    }
    checkArrayWithOffsets(
        converted,
        filter,
        readSizes,
        stringDecoderZeroCopy,
        maxArrayElementsCount,
        filterAfterRead);
  }

  void checkSlidingWindowMap(
      const std::vector<std::optional<
          std::vector<std::pair<int64_t, std::optional<int64_t>>>>>& data,
      const std::vector<bool>& rowFilter,
      const common::Filter* keyFilter,
      const std::vector<int>& readSizes,
      bool stringDecoderZeroCopy,
      bool filterAfterRead = false) {
    RowVectorPtr vector;
    if (rowFilter.empty()) {
      vector = makeRowVector({makeNullableMapVector<int64_t>(data)});
    } else if (filterAfterRead) {
      vector = makeRowVector({
          makeNullableMapVector<int64_t>(data),
          makeRowVector(
              {makeConstant<int8_t>(0, data.size())},
              [&](auto i) { return !rowFilter[i]; }),
      });
    } else {
      vector = makeRowVector({
          makeNullableMapVector<int64_t>(data),
          makeFlatVector<bool>(rowFilter),
      });
    }
    auto rowType = asRowType(vector->type());
    auto file = test::createNimbleFile(
        *rootPool(), vector, {.deduplicatedMapColumns = {"c0"}});
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    if (!rowFilter.empty()) {
      if (filterAfterRead) {
        scanSpec->childByName("c0")->setFilter(
            std::make_unique<common::IsNotNull>());
        scanSpec->childByName("c1")->setFilter(
            std::make_unique<common::IsNotNull>());
      } else {
        scanSpec->childByName("c1")->setFilter(
            std::make_unique<common::BoolValue>(true, false));
      }
    }
    if (keyFilter) {
      scanSpec->childByName("c0")
          ->childByName(common::ScanSpec::kMapKeysFieldName)
          ->setFilter(keyFilter->clone());
    }
    auto readers = makeReaders(vector, file, scanSpec, stringDecoderZeroCopy);
    auto result = BaseVector::create(rowType, 0, pool());
    int totalScanned = 0;
    std::vector<bool> selected(data.size(), true);
    if (!rowFilter.empty()) {
      selected = rowFilter;
    }
    for (int readSize : readSizes) {
      ASSERT_EQ(readers.rowReader->next(readSize, result), readSize);
      auto begin = selected.begin() + totalScanned;
      ASSERT_EQ(result->size(), std::accumulate(begin, begin + readSize, 0));
      auto& c0 = result->loadedVector()->asChecked<RowVector>()->childAt(0);
      int resultIndex = 0;
      int wrappedIndex = -1;
      int offset = 0;
      int lastSize = 0;
      int lastExpected = -1;
      for (int i = 0; i < readSize; ++i) {
        if (!selected[totalScanned + i]) {
          continue;
        }
        auto& expected = data[totalScanned + i];
        if (!expected.has_value()) {
          ASSERT_TRUE(c0->isNullAt(resultIndex));
        } else {
          if (lastExpected < 0 || data[lastExpected] != expected) {
            ++wrappedIndex;
            lastExpected = totalScanned + i;
            offset += lastSize;
          }
          std::vector<int> selectedKeys;
          if (keyFilter) {
            for (int j = 0; j < expected->size(); ++j) {
              if (keyFilter->testInt64((*expected)[j].first)) {
                selectedKeys.push_back(j);
              }
            }
          } else {
            selectedKeys.resize(expected->size());
            std::iota(selectedKeys.begin(), selectedKeys.end(), 0);
          }
          ASSERT_FALSE(c0->isNullAt(resultIndex));
          ASSERT_EQ(c0->wrappedIndex(resultIndex), wrappedIndex);
          auto* alphabet = c0->wrappedVector()->asChecked<MapVector>();
          ASSERT_EQ(alphabet->offsetAt(wrappedIndex), offset);
          ASSERT_EQ(alphabet->sizeAt(wrappedIndex), selectedKeys.size());
          lastSize = alphabet->sizeAt(wrappedIndex);
          auto* keys = alphabet->mapKeys()->asChecked<FlatVector<int64_t>>();
          auto* values =
              alphabet->mapValues()->asChecked<FlatVector<int64_t>>();
          for (int j = 0; j < selectedKeys.size(); ++j) {
            auto [expectedKey, expectedValue] = (*expected)[selectedKeys[j]];
            ASSERT_FALSE(keys->isNullAt(offset + j));
            ASSERT_EQ(keys->valueAt(offset + j), expectedKey);
            if (expectedValue.has_value()) {
              ASSERT_FALSE(values->isNullAt(offset + j));
              ASSERT_EQ(values->valueAt(offset + j), *expectedValue);
            } else {
              ASSERT_TRUE(values->isNullAt(offset + j));
            }
          }
        }
        ++resultIndex;
      }
      totalScanned += readSize;
    }
    ASSERT_EQ(totalScanned, data.size());
    ASSERT_EQ(readers.rowReader->next(1, result), 0);
  }

  void checkSlidingWindowMap(
      std::initializer_list<NullableMapData> data,
      const std::vector<bool>& rowFilter,
      const common::Filter* keyFilter,
      const std::vector<int>& readSizes,
      bool stringDecoderZeroCopy,
      bool filterAfterRead = false) {
    std::vector<
        std::optional<std::vector<std::pair<int64_t, std::optional<int64_t>>>>>
        converted;
    converted.reserve(data.size());
    for (const auto& item : data) {
      converted.push_back(item);
    }
    checkSlidingWindowMap(
        converted,
        rowFilter,
        keyFilter,
        readSizes,
        stringDecoderZeroCopy,
        filterAfterRead);
  }

 private:
  static RowTypePtr rowType(const TypePtr& type, int dropColumn) {
    if (dropColumn < 0) {
      return asRowType(type);
    }
    auto& rowType = type->asRow();
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    for (int k = 0; k < rowType.size(); ++k) {
      if (k != dropColumn) {
        names.push_back(rowType.nameOf(k));
        types.push_back(rowType.childAt(k));
      }
    }
    return ROW(std::move(names), std::move(types));
  }
};

// This case covers
//   - Dense + No nulls
//   - Sparse + No nulls
//   - Sparse + Nulls
TEST_P(SelectiveNimbleReaderTest, basic) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto c0 = makeFlatVector<int64_t>(1009, folly::identity);
  auto* rawC0 = c0->mutableRawValues();
  std::default_random_engine rng(42);
  std::shuffle(rawC0, rawC0 + 809, rng);
  setNulls(c0, nullEvery(11));
  auto input = makeRowVector({
      c0,
      makeFlatVector<int64_t>(1009, folly::identity, nullEvery(17)),
      makeRowVector({c0}, nullEvery(13)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(0, 502, false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  auto estimatedRowSize = readers.rowReader->estimatedRowSize();
  ASSERT_TRUE(estimatedRowSize.has_value());
  ASSERT_EQ(*estimatedRowSize, 21);
  validate(*input, *readers.rowReader, 101, [&](auto i) {
    return !c0->isNullAt(i) && rawC0[i] <= 502;
  });
}

TEST_P(SelectiveNimbleReaderTest, currentStripe) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto input = makeRowVector({
      makeFlatVector<int64_t>(200, folly::identity),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  EXPECT_EQ(readers.rowReader->currentStripe(), 0u);
}

TEST_P(SelectiveNimbleReaderTest, readsWithoutMetadataIoStats) {
  // Velox's connector path (HiveDataSource) builds ReaderOptions without
  // metadata IO stats, while TabletReader requires them. Reading has to work
  // anyway, so that this factory is usable on its own rather than only behind
  // a wrapper that fills them in.
  auto input = makeRowVector({
      makeFlatVector<int64_t>(200, folly::identity),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());

  const auto file = test::createNimbleFile(*rootPool(), input);
  dwio::common::ReaderOptions options(pool());
  options.setScanSpec(scanSpec);
  ASSERT_EQ(options.metadataIoStats(), nullptr);

  auto reader = SelectiveNimbleReaderFactory().createReader(
      std::make_unique<dwio::common::BufferedInput>(
          std::make_shared<InMemoryReadFile>(file), *pool()),
      options);
  EXPECT_EQ(reader->numberOfRows(), input->size());

  dwio::common::RowReaderOptions rowOptions;
  rowOptions.setScanSpec(scanSpec);
  rowOptions.setRequestedType(asRowType(input->type()));
  auto rowReader = reader->createRowReader(rowOptions);
  validate(*input, *rowReader, 101, [](auto /*i*/) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, denseWithNulls) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto input = makeRowVector({
      makeRowVector(
          {makeFlatVector<int64_t>(103, folly::identity)}, nullEvery(11)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  auto result = BaseVector::create(input->type(), 0, pool());
  validate(*input, *readers.rowReader, 7, [](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, sharedDictionary) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  const auto valueUniverse = sharedDictionaryValueUniverse();

  for (const bool nullableData : {false, true}) {
    for (const auto scope :
         {SharedDictionaryScope::Stripe,
          SharedDictionaryScope::File,
          SharedDictionaryScope::External}) {
      const uint32_t dictionaryId =
          scope == SharedDictionaryScope::Stripe ? 0 : 17;
      std::shared_ptr<const ExternalDictionaryResolver> resolver;
      if (scope == SharedDictionaryScope::External) {
        resolver = std::make_shared<SharedDictionaryTestResolver>(
            std::vector<std::pair<uint32_t, std::vector<int32_t>>>{
                {dictionaryId, valueUniverse}},
            pool());
      }

      const std::vector<SharedDictionarySource> sources{{
          .columnName = "c0",
          .dictionaryKey = 10,
          .scope = scope,
          .dictionaryId = dictionaryId,
      }};
      auto input = makeSharedDictionaryInput(
          pool(),
          /*rowCount=*/2'000,
          sources,
          valueUniverse,
          nullableData);
      auto scanSpec = std::make_shared<common::ScanSpec>("root");
      scanSpec->addAllChildFields(*input->type());
      const auto file = test::createNimbleFile(
          *rootPool(),
          input,
          sharedDictionaryWriterOptions(
              sources, {.externalDictionaryResolver = resolver}));

      for (const bool lazyColumnIo : {false, true}) {
        SCOPED_TRACE(
            fmt::format(
                "scope={}, nullableData={}, lazyColumnIo={}",
                scope,
                nullableData,
                lazyColumnIo));
        auto readers = makeReaders(
            input,
            file,
            scanSpec,
            stringDecoderZeroCopy,
            /*preserveFlatMapsInMemory=*/false,
            lazyColumnIo,
            resolver);
        if (!stringDecoderZeroCopy) {
          NIMBLE_ASSERT_THROW(
              validate(
                  *input,
                  *readers.rowReader,
                  /*batchSize=*/127,
                  [](auto /*row*/) { return true; }),
              "Shared dictionary encoding requires non-legacy encoding "
              "dispatch.");
          continue;
        }
        validate(
            *input,
            *readers.rowReader,
            /*batchSize=*/127,
            [](auto /*row*/) { return true; });
      }
    }
  }
}

TEST_P(SelectiveNimbleReaderTest, sharedDictionaryRandomizedSourcesAndStripes) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  constexpr uint32_t kSeed{0x51A9D1C7};
  SCOPED_TRACE(fmt::format("seed={}", kSeed));

  const auto valueUniverse = sharedDictionaryValueUniverse();
  const std::vector<SharedDictionarySource> sources{
      {
          .columnName = "stripe",
          .dictionaryKey = 10,
          .scope = SharedDictionaryScope::Stripe,
          .dictionaryId = 0,
      },
      {
          .columnName = "file",
          .dictionaryKey = 20,
          .scope = SharedDictionaryScope::File,
          .dictionaryId = 7,
      },
      {
          .columnName = "external",
          .dictionaryKey = 30,
          .scope = SharedDictionaryScope::External,
          .dictionaryId = 17,
      },
  };
  auto resolver = std::make_shared<SharedDictionaryTestResolver>(
      std::vector<std::pair<uint32_t, std::vector<int32_t>>>{
          {sources.back().dictionaryId, valueUniverse}},
      pool());
  auto input = makeSharedDictionaryInput(
      pool(),
      /*rowCount=*/1'024,
      sources,
      valueUniverse,
      /*nullableData=*/true);
  const auto file = writeWithRandomStripes(
      rootPool(),
      input,
      sharedDictionaryWriterOptions(
          sources, {.externalDictionaryResolver = resolver}),
      kSeed);

  {
    auto readFile = std::make_shared<InMemoryReadFile>(file);
    auto tabletOptions = test::makeTestTabletOptions(pool());
    tabletOptions.externalDictionaryResolver = resolver;
    auto tablet = TabletReader::create(readFile, pool(), tabletOptions);
    EXPECT_GT(tablet->stripeCount(), 1);
  }

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  for (const bool lazyColumnIo : {false, true}) {
    SCOPED_TRACE(fmt::format("lazyColumnIo={}", lazyColumnIo));
    auto readers = makeReaders(
        input,
        file,
        scanSpec,
        stringDecoderZeroCopy,
        /*preserveFlatMapsInMemory=*/false,
        lazyColumnIo,
        resolver);
    if (!stringDecoderZeroCopy) {
      NIMBLE_ASSERT_THROW(
          validate(
              *input,
              *readers.rowReader,
              /*batchSize=*/89,
              [](auto /*row*/) { return true; }),
          "Shared dictionary encoding requires non-legacy encoding dispatch.");
      continue;
    }
    validate(
        *input,
        *readers.rowReader,
        /*batchSize=*/89,
        [](auto /*row*/) { return true; });
  }
}

TEST_P(SelectiveNimbleReaderTest, denseMostlyNulls) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto isNull = [](auto i) { return i != 53; };
  auto input = makeRowVector({
      makeRowVector({makeFlatVector<int64_t>(103, folly::identity)}, isNull),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 11, [](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, sparseMostlyNulls) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto c0 = makeFlatVector<int64_t>(101, folly::identity);
  auto input = makeRowVector({
      c0,
      makeRowVector({c0}, [](auto i) { return i % 17 != 0; }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(47, 100, true));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 13, [&](auto i) {
    return c0->isNullAt(i) || i >= 47;
  });
}

TEST_P(SelectiveNimbleReaderTest, allNulls) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  std::vector<vector_size_t> offsets(104);
  std::iota(offsets.begin(), offsets.end(), 0);
  auto input = makeRowVector({
      makeConstant<int64_t>(std::nullopt, 103),
      BaseVector::createNullConstant(ROW({"c0"}, {BIGINT()}), 103, pool()),
      BaseVector::createNullConstant(ARRAY(BIGINT()), 103, pool()),
      BaseVector::createNullConstant(MAP(BIGINT(), BIGINT()), 103, pool()),
      makeMapVector<int32_t, int32_t>(
          103,
          [](auto i) { return i % 2 == 0 ? 1 : 0; },
          [](auto) { return 1; },
          [](auto) { return 42; },
          {},
          [](auto) { return true; }),
      makeMapVector(
          offsets,
          BaseVector::createNullConstant(INTEGER(), 103, pool()),
          BaseVector::createNullConstant(INTEGER(), 103, pool())),
  });
  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c4", {}}};
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers =
      makeReaders(input, fileContent, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 11, [](auto) { return true; });

  scanSpec->resetCachedValues(false);
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNull>());
  readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 11, [](auto) { return true; });

  scanSpec->resetCachedValues(false);
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNotNull>());
  readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 11, [](auto) { return false; });
}

TEST_P(SelectiveNimbleReaderTest, multiChunkNulls) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto chunk1 = makeRowVector({
      makeFlatVector<StringView>(
          2, [](auto) { return "foo"; }, [](auto i) { return i == 1; }),
      makeConstant<bool>(true, 2),
  });
  auto chunk2 = makeRowVector({
      makeFlatVector<StringView>(
          10,
          [](auto i) { return i == 0 ? "foo" : "bar"; },
          [](auto i) { return i >= 2; }),
      makeFlatVector<bool>(10, [](auto i) { return i != 7; }),
  });
  auto chunk3 = makeRowVector({
      makeConstant<StringView>(StringView("quux"), 10),
      makeConstant<bool>(true, 10),
  });
  std::vector<std::pair<EncodingType, float>> readFactors = {
      {EncodingType::MainlyConstant, 1.0},
      {EncodingType::Constant, 1.0},
  };
  ManualEncodingSelectionPolicyFactory encodingFactory(readFactors);
  WriterOptions options;
  options.encodingSelectionPolicyCreator = [&](DataType dataType) {
    return encodingFactory.createPolicy(dataType);
  };
  options.enableChunking = true;
  options.minStreamChunkRawSize = 0;
  options.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2, chunk3}, options, false);
  auto& input = chunk1;
  input->append(chunk2.get());
  input->append(chunk3.get());
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c1")->setFilter(
      std::make_unique<common::BoolValue>(true, true));
  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  validate(
      *input, *readers.rowReader, input->size(), [](auto i) { return i != 9; });
}

TEST_P(SelectiveNimbleReaderTest, filterIsNull) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto c0 = makeFlatVector<int64_t>(11, folly::identity, nullEvery(2));
  auto input = makeRowVector({makeRowVector({c0}, nullEvery(3))});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->childByName("c0")->setFilter(
      std::make_unique<common::IsNull>());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 5, [&](auto i) {
    return i % 2 == 0 || i % 3 == 0;
  });
}

TEST_P(SelectiveNimbleReaderTest, multiChunkInt16RowSetOverBoundary) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto chunk1 = makeRowVector({
      makeFlatVector<int16_t>(10, folly::identity),
  });
  auto chunk2 = makeRowVector({
      makeFlatVector<int16_t>(3, folly::identity),
  });
  std::vector<std::pair<EncodingType, float>> readFactors;
  ManualEncodingSelectionPolicyFactory encodingFactory(readFactors);
  WriterOptions options;
  options.encodingSelectionPolicyCreator = [&](DataType dataType) {
    return encodingFactory.createPolicy(dataType);
  };
  options.enableChunking = true;
  options.minStreamChunkRawSize = 0;
  options.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file =
      test::createNimbleFile(*rootPool(), {chunk1, chunk2}, options, false);
  auto& input = chunk1;
  input->append(chunk2.get());
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNotNull>());
  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  validate(
      *input, *readers.rowReader, input->size(), [](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, strings) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  const std::string longPrefix(17, 'x');
  auto c0 = makeFlatVector<std::string>(
      13,
      [&](auto i) {
        return i % 2 == 0 ? std::to_string(i)
                          : fmt::format("{}{}", longPrefix, i);
      },
      nullEvery(5));
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 3, [](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, bools) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto input = makeRowVector({
      makeFlatVector<bool>(
          67, [](auto i) { return i % 3 != 0; }, nullEvery(7)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BoolValue>(true, true));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 13, [&](auto i) {
    return i % 7 == 0 || i % 3 != 0;
  });
}

TEST_P(SelectiveNimbleReaderTest, floats) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeFlatVector<double>(
            101,
            [](auto i) { return sin(i); },
            hasNulls ? nullEvery(11) : nullptr);
      },
      common::FloatingPointRange<double>(
          -INFINITY, true, false, 0.5, false, false, false),
      23,
      [&](auto& data, auto i) { return !data->isNullAt(i) && sin(i) <= 0.5; },
      stringDecoderZeroCopy);
}

TEST_P(SelectiveNimbleReaderTest, rle) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeFlatVector<double>(
            101,
            [](auto i) { return sin(i / 13); },
            hasNulls ? nullEvery(17) : nullptr);
      },
      common::FloatingPointRange<double>(
          -INFINITY, true, false, 0.5, false, false, false),
      23,
      [&](auto& data, auto i) {
        return !data->isNullAt(i) && sin(i / 13) <= 0.5;
      },
      stringDecoderZeroCopy);
}

TEST_P(SelectiveNimbleReaderTest, byteRle) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeFlatVector<int8_t>(
            101,
            [](auto i) { return i / 13; },
            hasNulls ? nullEvery(11) : nullptr);
      },
      common::BigintRange(4, 6, false),
      23,
      [&](auto& data, auto i) {
        return !data->isNullAt(i) && 4 <= i / 13 && i / 13 <= 6;
      },
      stringDecoderZeroCopy);
}

// Pushes down a BIGINT IN-list filter (createBigintValues). The batch reader
// covers IN-lists, but the selective path only exercised BigintRange; migration
// workloads use IN heavily, so cover it here.
TEST_P(SelectiveNimbleReaderTest, bigintInList) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  const std::vector<int64_t> inList{3, 17, 42, 88, 500};
  auto inValue = [&](int64_t v) {
    return std::find(inList.begin(), inList.end(), v) != inList.end();
  };
  runTestCase(
      [&](bool hasNulls) {
        return makeFlatVector<int64_t>(
            1009, folly::identity, hasNulls ? nullEvery(11) : nullptr);
      },
      *common::createBigintValues(inList, /*nullAllowed=*/false),
      101,
      [&](auto& data, auto i) { return !data->isNullAt(i) && inValue(i); },
      stringDecoderZeroCopy);
}

// Pushes down a VARCHAR IN-list filter (BytesValues). Strings were only
// read-verified before; equality/IN pushdown on VARCHAR was untested.
TEST_P(SelectiveNimbleReaderTest, stringInList) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  static constexpr std::array<const char*, 5> kValues{
      "aaa", "bbb", "ccc", "ddd", "eee"};
  const std::vector<std::string> inList{"aaa", "ccc", "eee"};
  runTestCase(
      [&](bool hasNulls) {
        return makeFlatVector<std::string>(
            101,
            [](auto i) { return std::string(kValues[i % kValues.size()]); },
            hasNulls ? nullEvery(11) : nullptr);
      },
      common::BytesValues(inList, /*nullAllowed=*/false),
      23,
      [&](auto& data, auto i) {
        // kValues at even offsets (aaa/ccc/eee) are in the IN-list.
        return !data->isNullAt(i) && (i % kValues.size()) % 2 == 0;
      },
      stringDecoderZeroCopy);
}

// Pushes down a VARCHAR bounded range filter (BytesRange), i.e. WHERE col
// BETWEEN 'bbb' AND 'ddd'.
TEST_P(SelectiveNimbleReaderTest, stringRange) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  static constexpr std::array<const char*, 5> kValues{
      "aaa", "bbb", "ccc", "ddd", "eee"};
  runTestCase(
      [&](bool hasNulls) {
        return makeFlatVector<std::string>(
            101,
            [](auto i) { return std::string(kValues[i % kValues.size()]); },
            hasNulls ? nullEvery(11) : nullptr);
      },
      common::BytesRange(
          "bbb",
          /*lowerUnbounded=*/false,
          /*lowerExclusive=*/false,
          "ddd",
          /*upperUnbounded=*/false,
          /*upperExclusive=*/false,
          /*nullAllowed=*/false),
      23,
      [&](auto& data, auto i) {
        // Keeps bbb/ccc/ddd (offsets 1..3).
        const auto offset = i % kValues.size();
        return !data->isNullAt(i) && offset >= 1 && offset <= 3;
      },
      stringDecoderZeroCopy);
}

// Pushes down a REAL (float32) range filter. The double range path is covered
// by `floats`; REAL is a distinct decoder path used by migration workloads.
TEST_P(SelectiveNimbleReaderTest, realRange) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeFlatVector<float>(
            101,
            [](auto i) { return static_cast<float>(sin(i)); },
            hasNulls ? nullEvery(11) : nullptr);
      },
      common::FloatingPointRange<float>(
          -INFINITY, true, false, 0.5f, false, false, false),
      23,
      [&](auto& data, auto i) {
        return !data->isNullAt(i) && static_cast<float>(sin(i)) <= 0.5f;
      },
      stringDecoderZeroCopy);
}

TEST_P(SelectiveNimbleReaderTest, rleString) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto c0 = makeFlatVector<std::string>(
      101, [](auto i) { return std::to_string(sin(i / 13)); }, nullEvery(17));
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 23, [](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, rleBool) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto c0 =
      makeFlatVector<bool>(67, [](auto i) { return i >= 31; }, nullEvery(17));
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 23, [](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, mainlyConstant) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  const std::string longPrefix(17, 'x');
  auto makeData = [&](bool hasNulls) {
    return makeFlatVector<std::string>(
        101,
        [&](auto i) {
          return i % 11 != 0 ? "common" : fmt::format("{}{}", longPrefix, i);
        },
        hasNulls ? nullEvery(17) : nullptr);
  };
  {
    SCOPED_TRACE("Keep common value");
    runTestCase(
        makeData,
        common::BytesValues(
            {"common", fmt::format("{}{}", longPrefix, 11)}, false),
        23,
        [&](auto& data, auto i) {
          return !data->isNullAt(i) && (i % 11 != 0 || i == 11);
        },
        stringDecoderZeroCopy);
  }
  {
    SCOPED_TRACE("Drop common value");
    runTestCase(
        makeData,
        common::BytesValues({fmt::format("{}{}", longPrefix, 11)}, false),
        23,
        [&](auto& data, auto i) { return !data->isNullAt(i) && i == 11; },
        stringDecoderZeroCopy);
  }
}

TEST_P(SelectiveNimbleReaderTest, dictionary) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  const std::string alphabet[] = {"foo", "bar", "quux"};
  auto c0 = makeFlatVector<std::string>(
      1009, [&](auto i) { return alphabet[i % 3]; }, nullEvery(17));
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 23, [](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, smallDictionaryValue) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto c0 = makeFlatVector<int8_t>(257, folly::identity);
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  std::vector<std::pair<EncodingType, float>> readFactors = {
      {EncodingType::MainlyConstant, 1.0},
      {EncodingType::Dictionary, 1.0},
  };
  ManualEncodingSelectionPolicyFactory encodingFactory(readFactors);
  WriterOptions options;
  options.encodingSelectionPolicyCreator = [&](DataType dataType) {
    return encodingFactory.createPolicy(dataType);
  };
  auto readers = makeReaders(
      input,
      test::createNimbleFile(*rootPool(), input, options),
      scanSpec,
      stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 257, [](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, constant) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto c0 = makeFlatVector<std::string>(
      101, [&](auto) { return "foo"; }, nullEvery(17));
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 23, [](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, array) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeArrayVector<double>(
            13,
            [](auto i) { return 1 + i % 5; },
            [](auto j) { return sin(j); },
            hasNulls ? nullEvery(7) : nullptr);
      },
      common::IsNotNull(),
      5,
      [](auto& data, auto i) { return !data->isNullAt(i); },
      stringDecoderZeroCopy);
}

TEST_P(SelectiveNimbleReaderTest, arrayPruning) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto input = makeRowVector({
      makeArrayVector<int64_t>({{1, 2, 3, 4, 5}, {1, 2}}),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setMaxArrayElementsCount(3);
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  auto expected = makeRowVector({
      makeArrayVector<int64_t>({{1, 2, 3}, {1, 2}}),
  });
  validate(*expected, *readers.rowReader, 23, [](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, arrayElementFilter) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto input = makeRowVector({
      makeArrayVector<int64_t>({{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")
      ->childByName(common::ScanSpec::kArrayElementsFieldName)
      ->setFilter(std::make_unique<common::BigintRange>(4, 7, false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  auto expected = makeRowVector({makeArrayVector<int64_t>({
      {4, 5},
      {6, 7},
  })});
  validate(*expected, *readers.rowReader, 23, [](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, map) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeMapVector<int64_t, double>(
            13,
            [](auto i) { return 1 + i % 5; },
            [](auto j) { return j; },
            [](auto j) { return sin(j); },
            hasNulls ? nullEvery(7) : nullptr);
      },
      common::IsNotNull(),
      5,
      [](auto& data, auto i) { return !data->isNullAt(i); },
      stringDecoderZeroCopy);
}

TEST_P(SelectiveNimbleReaderTest, mapKeyFilter) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto input = makeRowVector({makeMapVector<int64_t, double>({
      {{1, 0.5}, {2, 1.0}, {3, 1.5}, {4, 2.0}, {5, 2.5}},
      {{6, 3.0}, {7, 3.5}, {20, 10.0}},
  })});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")
      ->childByName(common::ScanSpec::kMapKeysFieldName)
      ->setFilter(std::make_unique<common::BigintRange>(4, 10, false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  auto expected = makeRowVector({makeMapVector<int64_t, double>({
      {{4, 2.0}, {5, 2.5}},
      {{6, 3.0}, {7, 3.5}},
  })});
  validate(*expected, *readers.rowReader, 23, [](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, mapValueFilter) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto input = makeRowVector({makeMapVector<int64_t, double>({
      {{1, 0.5}, {2, 1.0}, {3, 1.5}, {4, 2.0}, {5, 2.5}},
      {{6, 3.0}, {7, 3.5}, {20, 10.0}},
  })});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")
      ->childByName(common::ScanSpec::kMapValuesFieldName)
      ->setFilter(
          std::make_unique<common::DoubleRange>(
              2.0, false, false, 5.0, false, true, false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  auto expected = makeRowVector({makeMapVector<int64_t, double>({
      {{4, 2.0}, {5, 2.5}},
      {{6, 3.0}, {7, 3.5}},
  })});
  validate(*expected, *readers.rowReader, 23, [](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, estimatedRowSize) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  constexpr int kSize = 13;
  auto input = makeRowVector({
      makeFlatVector<bool>(
          kSize, [](auto i) { return i % 3 != 0; }, nullEvery(5)),
      makeMapVector<int8_t, std::string>(
          kSize,
          [](auto i) { return 1 + i % 5; },
          [](auto j) { return j % 128; },
          [](auto j) { return std::to_string(j); },
          nullEvery(7)),
      makeMapVector<int32_t, double>(
          kSize,
          [](auto i) { return 1 + i % 5; },
          [](auto j) { return j % 6; },
          [](auto j) { return sin(j); },
          nullEvery(7)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c2", {}}};
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto readers =
      makeReaders(input, fileContent, scanSpec, stringDecoderZeroCopy);
  auto estimatedRowSize = readers.rowReader->estimatedRowSize();
  ASSERT_TRUE(estimatedRowSize.has_value());
  ASSERT_EQ(*estimatedRowSize, 37);
  {
    SCOPED_TRACE("Read flatmap as struct");
    auto c2Type =
        ROW({"1", "2", "3", "6", "4", "5"},
            {DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE()});
    auto outputType =
        ROW({"c0", "c1", "c2"}, {BOOLEAN(), MAP(TINYINT(), VARCHAR()), c2Type});
    auto structScanSpec = std::make_shared<common::ScanSpec>("root");
    structScanSpec->addAllChildFields(*outputType);
    structScanSpec->childByName("c2")->setFlatMapAsStruct(true);
    auto readFile = std::make_shared<InMemoryReadFile>(fileContent);
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions options(pool());
    options.setDataIoStats(dataIoStats_);
    options.setMetadataIoStats(metadataIoStats_);
    options.setScanSpec(structScanSpec);
    auto reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
        options);
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(structScanSpec);
    auto rowReader = readers.reader->createRowReader(rowOptions);
    auto structEstimatedRowSize = rowReader->estimatedRowSize();
    ASSERT_TRUE(structEstimatedRowSize.has_value());
    ASSERT_EQ(*structEstimatedRowSize, 37);
    auto result = BaseVector::create(outputType, 0, pool());
    int numRows = 0;
    while (rowReader->next(11, result) > 0) {
      auto* rowResult = result->asUnchecked<RowVector>();
      ASSERT_TRUE(result->type()->childAt(2)->isRow());
      ASSERT_TRUE(rowResult->childAt(2)->type()->isRow());
      auto* c2 =
          rowResult->childAt(2)->loadedVector()->asUnchecked<RowVector>();
      ASSERT_EQ(c2->childrenSize(), 6);
      ASSERT_TRUE(c2->childAt(3)->isConstantEncoding());
      ASSERT_TRUE(c2->childAt(3)->isNullAt(0));
      numRows += result->size();
    }
    ASSERT_EQ(numRows, kSize);
  }
}

TEST_P(SelectiveNimbleReaderTest, estimatedRowSizeNullableString) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto input = makeRowVector({
      makeNullableFlatVector<std::string>({{"foo"}, std::nullopt, {"bar"}}),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  auto estimatedRowSize = readers.rowReader->estimatedRowSize();
  ASSERT_TRUE(estimatedRowSize.has_value());
  ASSERT_EQ(*estimatedRowSize, 2);
  validate(*input, *readers.rowReader, 3, [&](auto) { return true; });
}

// Verifies that estimatedRowSize only counts projected columns, not all
// columns in the file.
TEST_P(SelectiveNimbleReaderTest, estimatedRowSizePartialProjection) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  constexpr int kSize = 100;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<int32_t>(kSize, folly::identity),
      makeFlatVector<double>(kSize, [](auto i) { return i * 1.5; }),
  });

  auto fileContent = test::createNimbleFile(*rootPool(), input);

  // Project all 3 columns: int64 (8) + int32 (4) + double (8) = 20 per row.
  {
    auto allSpec = std::make_shared<common::ScanSpec>("root");
    allSpec->addAllChildFields(*input->type());
    auto readers =
        makeReaders(input, fileContent, allSpec, stringDecoderZeroCopy);
    auto allSize = readers.rowReader->estimatedRowSize();
    ASSERT_TRUE(allSize.has_value());
    ASSERT_EQ(*allSize, 20);
  }

  // Project only c0 (int64): expect 8 per row.
  {
    auto partialType = ROW({"c0"}, {BIGINT()});
    auto partialSpec = std::make_shared<common::ScanSpec>("root");
    partialSpec->addAllChildFields(*partialType);
    auto readFile = std::make_shared<InMemoryReadFile>(fileContent);
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions options(pool());
    options.setDataIoStats(dataIoStats_);
    options.setMetadataIoStats(metadataIoStats_);
    options.setScanSpec(partialSpec);
    auto reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
        options);
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(partialSpec);
    auto rowReader = reader->createRowReader(rowOptions);
    auto partialSize = rowReader->estimatedRowSize();
    ASSERT_TRUE(partialSize.has_value());
    ASSERT_EQ(*partialSize, 8);
  }
}

// Verifies estimatedRowSize for nested types (array, map) uses the
// already-rolled-up column stats without double-counting.
TEST_P(SelectiveNimbleReaderTest, estimatedRowSizeNestedTypes) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  constexpr int kSize = 20;

  // Array<int64_t> with 3 elements each.
  // The array column's rolled-up logicalSize includes element sizes + null
  // overhead. With 20 non-null rows and 3 int64 elements each:
  // element data = 20 * 3 * 8 = 480 bytes, plus null overhead.
  {
    auto input = makeRowVector({
        makeArrayVector<int64_t>(
            kSize, [](auto) { return 3; }, [](auto j) { return j; }),
    });
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
    auto estimatedRowSize = readers.rowReader->estimatedRowSize();
    ASSERT_TRUE(estimatedRowSize.has_value());
    // At least 24 bytes per row (3 elements * 8 bytes each).
    ASSERT_GE(*estimatedRowSize, 24);
    // Should not be wildly over-estimated (no double-counting).
    ASSERT_LE(*estimatedRowSize, 48);
  }

  // Map<int32_t, int64_t> with 2 entries each.
  // key data = 20 * 2 * 4 = 160, value data = 20 * 2 * 8 = 320.
  {
    auto input = makeRowVector({
        makeMapVector<int32_t, int64_t>(
            kSize,
            [](auto) { return 2; },
            [](auto j) { return j; },
            [](auto j) { return j * 10; }),
    });
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
    auto estimatedRowSize = readers.rowReader->estimatedRowSize();
    ASSERT_TRUE(estimatedRowSize.has_value());
    // At least 24 bytes per row (2 * (4 + 8)).
    ASSERT_GE(*estimatedRowSize, 24);
    ASSERT_LE(*estimatedRowSize, 48);
  }
}

// Verifies estimatedRowSize handles partial nested projection on a nested ROW.
// Projects only one subfield of a nested struct, ensuring we don't count the
// entire struct's rolled-up logicalSize.
TEST_P(SelectiveNimbleReaderTest, estimatedRowSizeNestedRowPartialProjection) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  constexpr int kSize = 100;

  // Schema: ROW{a: BIGINT, b: ROW{x: INT, y: DOUBLE}}
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeRowVector({
          makeFlatVector<int32_t>(kSize, folly::identity),
          makeFlatVector<double>(kSize, [](auto i) { return i * 1.5; }),
      }),
  });

  auto fileContent = test::createNimbleFile(*rootPool(), input);

  // Project all: a(8) + b.x(4) + b.y(8) + null overhead = ~20+ per row.
  {
    auto allSpec = std::make_shared<common::ScanSpec>("root");
    allSpec->addAllChildFields(*input->type());
    auto readers =
        makeReaders(input, fileContent, allSpec, stringDecoderZeroCopy);
    auto allSize = readers.rowReader->estimatedRowSize();
    ASSERT_TRUE(allSize.has_value());
    auto fullSize = *allSize;
    ASSERT_GE(fullSize, 20);
  }

  // Project only a(8) + b.x(4): should be less than full projection.
  {
    auto partialType = ROW({"c0", "c1"}, {BIGINT(), ROW({"c0"}, {INTEGER()})});
    auto partialSpec = std::make_shared<common::ScanSpec>("root");
    partialSpec->addAllChildFields(*partialType);
    auto readFile = std::make_shared<InMemoryReadFile>(fileContent);
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions options(pool());
    options.setDataIoStats(dataIoStats_);
    options.setMetadataIoStats(metadataIoStats_);
    options.setScanSpec(partialSpec);
    auto reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
        options);
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(partialSpec);
    auto rowReader = reader->createRowReader(rowOptions);
    auto partialSize = rowReader->estimatedRowSize();
    ASSERT_TRUE(partialSize.has_value());
    // Should be roughly 12 (8 + 4) plus some null overhead, but significantly
    // less than the full projection.
    ASSERT_GE(*partialSize, 12);
    ASSERT_LE(*partialSize, 20);
  }
}

// Verifies estimatedRowSize with nullable nested types includes null overhead
// from intermediate container nodes.
TEST_P(SelectiveNimbleReaderTest, estimatedRowSizeNullableNested) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  constexpr int kSize = 100;

  // Nested ROW with 50% nulls — the null overhead should be reflected.
  auto input = makeRowVector({
      makeRowVector(
          {makeFlatVector<int64_t>(kSize, folly::identity)}, nullEvery(2)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  auto estimatedRowSize = readers.rowReader->estimatedRowSize();
  ASSERT_TRUE(estimatedRowSize.has_value());
  // With 50% nulls on the outer ROW, the non-null rows contribute int64 (8)
  // data, plus null overhead at the ROW level.
  ASSERT_GE(*estimatedRowSize, 1);
}

// Verifies that estimatedRowSize falls back gracefully when vectorized stats
// are not available in the file.
TEST_P(SelectiveNimbleReaderTest, estimatedRowSizeNoStats) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  constexpr int kSize = 100;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
  });

  // Write file without vectorized stats by using an older writer config.
  WriterOptions writerOptions;
  writerOptions.enableVectorizedStats = false;
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers =
      makeReaders(input, fileContent, scanSpec, stringDecoderZeroCopy);
  // The fallback may or may not return a value depending on whether
  // estimateMaterializedSize succeeds, but it should not crash.
  ASSERT_NO_THROW(readers.rowReader->estimatedRowSize());
}

// Lazy FlatMap child uses stream-based estimate, not RowSizeTracker
// fallback.
TEST_P(SelectiveNimbleReaderTest, estimatedRowSizeLazyColumn) {
  const bool passStringBuffersFromDecoder = GetParam().stringDecoderZeroCopy;
  constexpr int kSize = 200;

  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeMapVector<int32_t, int32_t>(
          kSize,
          [](auto) { return 10; }, // 10 keys per row
          [](auto i) { return i; },
          [](auto i) { return i * 100; }),
  });

  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c1", {}}};
  writerOptions.enableVectorizedStats = false;
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(0, kSize, false));

  auto readers = makeReaders(
      input,
      fileContent,
      scanSpec,
      passStringBuffersFromDecoder,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);

  auto estimatedRowSize = readers.rowReader->estimatedRowSize();
  // With lazy columns, estimateMaterializedSize returns false and the
  // fallback (1MB default or tracked row size) is used.
  ASSERT_TRUE(estimatedRowSize.has_value());

  validate(*input, *readers.rowReader, kSize, [](auto) { return true; });
}

// Map-key-only filter must not cause all children to be lazy.
TEST_P(SelectiveNimbleReaderTest, estimatedRowSizeMapKeyFilterOnly) {
  const bool passStringBuffersFromDecoder = GetParam().stringDecoderZeroCopy;
  constexpr int kSize = 200;

  auto input = makeRowVector({
      makeMapVector<int32_t, int32_t>(
          kSize,
          [](auto) { return 5; },
          [](auto i) { return i; },
          [](auto i) { return i * 10; }),
      makeMapVector<int32_t, int32_t>(
          kSize,
          [](auto) { return 3; },
          [](auto i) { return i + 100; },
          [](auto i) { return i * 20; }),
  });

  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c0", {}}, {"c1", {}}};
  writerOptions.enableVectorizedStats = false;
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")
      ->childByName(common::ScanSpec::kMapKeysFieldName)
      ->setFilter(std::make_unique<common::BigintRange>(0, 2, false));

  auto readers = makeReaders(
      input,
      fileContent,
      scanSpec,
      passStringBuffersFromDecoder,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);

  auto estimatedRowSize = readers.rowReader->estimatedRowSize();
  // Both FlatMap columns are lazy (map-key filters are invisible to
  // hasFilter()). estimateMaterializedSize returns false (no eager children),
  // falling through to RowSizeTracker.
  ASSERT_TRUE(estimatedRowSize.has_value());
  ASSERT_GT(*estimatedRowSize, 0);
}

// Lazy VARCHAR MAP column: estimate doesn't crash on string decoder,
// laziness is preserved (data reads correctly after estimate).
TEST_P(SelectiveNimbleReaderTest, estimatedRowSizeLazyStringColumn) {
  const bool passStringBuffersFromDecoder = GetParam().stringDecoderZeroCopy;
  constexpr int kSize = 200;

  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeMapVector<StringView, StringView>(
          kSize,
          [](auto) { return 5; },
          [](auto i) {
            return StringView::makeInline(fmt::format("key_{}", i));
          },
          [](auto i) {
            return StringView::makeInline(fmt::format("val_{}", i));
          }),
  });

  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c1", {}}};
  writerOptions.enableVectorizedStats = false;
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(0, kSize, false));

  auto readers = makeReaders(
      input,
      fileContent,
      scanSpec,
      passStringBuffersFromDecoder,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);

  auto estimatedRowSize = readers.rowReader->estimatedRowSize();
  // With lazy columns, estimateMaterializedSize returns false and the
  // fallback (1MB default or tracked row size) is used.
  ASSERT_TRUE(estimatedRowSize.has_value());

  validate(*input, *readers.rowReader, kSize, [](auto) { return true; });
}

// All columns lazy (no scalar projected): estimateMaterializedSize returns
// false gracefully when rowCount=0, doesn't crash.
TEST_P(SelectiveNimbleReaderTest, estimatedRowSizeAllLazy) {
  const bool passStringBuffersFromDecoder = GetParam().stringDecoderZeroCopy;
  constexpr int kSize = 200;

  auto input = makeRowVector({
      makeMapVector<int32_t, int32_t>(
          kSize,
          [](auto) { return 5; },
          [](auto i) { return i; },
          [](auto i) { return i * 10; }),
  });

  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c0", {}}};
  writerOptions.enableVectorizedStats = false;
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());

  auto readers = makeReaders(
      input,
      fileContent,
      scanSpec,
      passStringBuffersFromDecoder,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);

  // Should not crash even though all columns are lazy and rowCount=0.
  ASSERT_NO_THROW(readers.rowReader->estimatedRowSize());

  validate(*input, *readers.rowReader, kSize, [](auto) { return true; });
}

// Lazy VARCHAR MAP with vectorized stats enabled and disabled.
TEST_P(SelectiveNimbleReaderTest, estimatedRowSizeLazyStringWithStats) {
  const bool passStringBuffersFromDecoder = GetParam().stringDecoderZeroCopy;
  constexpr int kSize = 200;

  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeMapVector<StringView, StringView>(
          kSize,
          [](auto) { return 5; },
          [](auto i) {
            return StringView::makeInline(fmt::format("key_{}", i));
          },
          [](auto i) {
            return StringView::makeInline(fmt::format("val_{}", i));
          }),
  });

  for (bool enableStats : {false, true}) {
    SCOPED_TRACE(fmt::format("enableVectorizedStats={}", enableStats));
    WriterOptions writerOptions;
    writerOptions.flatMapColumns = {{"c1", {}}};
    writerOptions.enableVectorizedStats = enableStats;
    auto fileContent =
        test::createNimbleFile(*rootPool(), input, writerOptions);

    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::BigintRange>(0, kSize, false));

    auto readers = makeReaders(
        input,
        fileContent,
        scanSpec,
        passStringBuffersFromDecoder,
        /*preserveFlatMapsInMemory=*/false,
        /*lazyColumnIo=*/true);

    auto estimatedRowSize = readers.rowReader->estimatedRowSize();
    ASSERT_TRUE(estimatedRowSize.has_value());
    ASSERT_GT(*estimatedRowSize, 0);

    validate(*input, *readers.rowReader, kSize, [](auto) { return true; });
  }
}

TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsLastRunFilteredOut) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {NullableArrayData{1}, NullableArrayData{}, NullableArrayData{}},
      {false, true, true},
      {1, 1, 1},
      stringDecoderZeroCopy);
}

TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsLastRunNotLoaded) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {std::vector<std::optional<int64_t>>{1},
       std::nullopt,
       std::vector<std::optional<int64_t>>{1}},
      {false, true, true},
      {1, 1, 1},
      stringDecoderZeroCopy);
}

TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsNoSeekBackward) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {std::vector<std::optional<int64_t>>{1},
       std::nullopt,
       std::vector<std::optional<int64_t>>{1}},
      {},
      {1, 1, 1},
      stringDecoderZeroCopy);
}

TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsLastRunResize) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {NullableArrayData{1},
       NullableArrayData{1},
       NullableArrayData{},
       NullableArrayData{}},
      {},
      {1, 2, 1},
      stringDecoderZeroCopy);
}

// Exercises the dense fast path in makeNestedRowSet and setAlphabet: all rows
// are selected, selectedIndices_ is dense, and nestedRows_ is built via iota.
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsDenseAllSelected) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {{{1, 2}}, {{3, 4}}, {{5, 6}}, {{7, 8}}}, {}, {4}, stringDecoderZeroCopy);
}

// Dense path across multiple batches: nestedRows built via iota on each batch.
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsDenseMultiBatch) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {std::vector<std::optional<int64_t>>{1},
       std::vector<std::optional<int64_t>>{2},
       std::vector<std::optional<int64_t>>{3},
       std::vector<std::optional<int64_t>>{4},
       std::vector<std::optional<int64_t>>{5},
       std::vector<std::optional<int64_t>>{6}},
      {},
      {2, 2, 2},
      stringDecoderZeroCopy);
}

// Dense path with nulls interspersed: nulls are not part of the alphabet, so
// selectedIndices_ remains dense for the non-null entries.
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsDenseWithNulls) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {{{1, 2}}, std::nullopt, {{3, 4}}, std::nullopt, {{5, 6}}},
      {},
      {5},
      stringDecoderZeroCopy);
}

// Dense path with empty arrays: empty arrays still occupy alphabet entries.
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsDenseWithEmpties) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {std::vector<std::optional<int64_t>>{},
       std::vector<std::optional<int64_t>>{1},
       std::vector<std::optional<int64_t>>{},
       std::vector<std::optional<int64_t>>{2},
       std::vector<std::optional<int64_t>>{}},
      {},
      {5},
      stringDecoderZeroCopy);
}

// When subfield pruning is enabled, the dense fast path in makeNestedRowSet is
// disabled (nestedRowsAllSelected is false), exercising the non-dense path even
// though all rows are selected.
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsDenseWithSubfieldPruning) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {{{1, 2, 3}}, {{4, 5, 6}}, {{7, 8, 9}}},
      {},
      {3},
      stringDecoderZeroCopy,
      2);
}

// Filter drops first rows, making selectedIndices_ sparse.  Exercises the
// non-dense (original) path in both makeNestedRowSet and setAlphabet, as a
// contrast to the dense cases above.
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsSparseFilter) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {{{1, 2}}, {{3, 4}}, {{5, 6}}, {{7, 8}}},
      {false, true, false, true},
      {2, 2},
      stringDecoderZeroCopy);
}

// --- Batch boundary interaction tests for the dense fast path ---
// These tests exercise the interplay between startFromLastRun_, loadLastRun_,
// copyLastRun_, and skipLastRun across batch boundaries, ensuring the dense
// iota path produces correct results.

// Run continuation across batches: the same deduplicated array value appears at
// the end of batch 1 and start of batch 2, triggering startFromLastRun_=true
// and copyLastRun_=true in batch 2.
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsDenseContinuedRun) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {{{1, 2}}, {{1, 2}}, {{3, 4}}, {{3, 4}}},
      {},
      {2, 2},
      stringDecoderZeroCopy);
}

// Single-row batches through a dense deduplicated sequence: every batch
// boundary triggers the last-run cache. Forces copyLastRun_ or skipLastRun
// transitions on every read.
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsDenseSingleRowBatches) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {std::vector<std::optional<int64_t>>{1},
       std::vector<std::optional<int64_t>>{2},
       std::vector<std::optional<int64_t>>{3},
       std::vector<std::optional<int64_t>>{4},
       std::vector<std::optional<int64_t>>{5}},
      {},
      {1, 1, 1, 1, 1},
      stringDecoderZeroCopy);
}

// Single-row batches with duplicate values: adjacent rows share the same
// alphabet entry, so the run spans batch boundaries repeatedly.
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsDenseSingleRowDuplicates) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {std::vector<std::optional<int64_t>>{1},
       std::vector<std::optional<int64_t>>{1},
       std::vector<std::optional<int64_t>>{2},
       std::vector<std::optional<int64_t>>{2},
       std::vector<std::optional<int64_t>>{3},
       std::vector<std::optional<int64_t>>{3}},
      {},
      {1, 1, 1, 1, 1, 1},
      stringDecoderZeroCopy);
}

// Last run unloaded then new run: batch 1 reads row 0, the alphabet has
// entries for rows 0 and 1 but row 1 exceeds the batch's maxRow, so
// loadLastRun_ is set to false. Batch 2 starts a different run
// (startFromLastRun_=false), so skipLastRun=true, disabling the dense path.
// Batch 3 continues normally. This tests the transition through skipLastRun.
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsDenseSkipLastRunTransition) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {{{1, 2}}, {{3, 4}}, {{5, 6}}, {{7, 8}}, {{9, 10}}},
      {},
      {1, 1, 1, 1, 1},
      stringDecoderZeroCopy);
}

// Uneven batch sizes splitting deduplicated runs at different points. Tests
// that the dense path and last-run cache interact correctly when batches are
// of varying sizes.
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsDenseUnevenBatches) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {std::vector<std::optional<int64_t>>{1},
       std::vector<std::optional<int64_t>>{2},
       std::vector<std::optional<int64_t>>{2},
       std::vector<std::optional<int64_t>>{3},
       std::vector<std::optional<int64_t>>{3},
       std::vector<std::optional<int64_t>>{3},
       std::vector<std::optional<int64_t>>{4}},
      {},
      {1, 3, 2, 1},
      stringDecoderZeroCopy);
}

// Null at batch boundary: batch 1 ends with a non-null, batch 2 starts with
// a null. Nulls are not in the alphabet, so this tests that the dense path
// handles the boundary when the alphabet is smaller than the row count.
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsDenseNullAtBatchBoundary) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {std::vector<std::optional<int64_t>>{1},
       std::vector<std::optional<int64_t>>{2},
       std::nullopt,
       std::vector<std::optional<int64_t>>{3},
       std::vector<std::optional<int64_t>>{4}},
      {},
      {2, 1, 2},
      stringDecoderZeroCopy);
}

// Dense multi-element arrays with run continuation: tests that offsets and
// sizes are computed correctly in setAlphabet when a multi-element array run
// spans a batch boundary.
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsDenseMultiElementContinued) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {{{1, 2, 3}}, {{1, 2, 3}}, {{4, 5}}, {{4, 5}}, {{6, 7, 8, 9}}},
      {},
      {2, 2, 1},
      stringDecoderZeroCopy);
}

TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsCopyLastRunAfterSkip) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {std::vector<std::optional<int64_t>>{1},
       std::vector<std::optional<int64_t>>{1},
       std::vector<std::optional<int64_t>>{2},
       std::nullopt,
       std::vector<std::optional<int64_t>>{3}},
      {true, false, false, true, true},
      {1, 2, 1, 1},
      stringDecoderZeroCopy);
}

TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsSubfieldPruning) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {std::vector<std::optional<int64_t>>{1, 2},
       std::vector<std::optional<int64_t>>{1, 2},
       std::vector<std::optional<int64_t>>{1},
       std::nullopt,
       std::vector<std::optional<int64_t>>{1, 2, 3}},
      {},
      {1, 1, 3},
      stringDecoderZeroCopy,
      1);
}

TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsLastRunFilteredOutAfterRead) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkArrayWithOffsets(
      {std::vector<std::optional<int64_t>>{1},
       std::vector<std::optional<int64_t>>{1}},
      {false, true},
      {1, 1},
      stringDecoderZeroCopy,
      std::nullopt,
      true);
}

TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsReuseNullResult) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto vector = makeRowVector({
      std::make_shared<MapVector>(
          pool(),
          MAP(BIGINT(), ARRAY(BIGINT())),
          nullptr,
          4,
          makeIndices({0, 2, 4, 6}),
          makeIndices({2, 2, 2, 2}),
          makeFlatVector<int64_t>({1, 2, 1, 2, 1, 2, 1, 2}),
          makeNullableArrayVector<int64_t>({
              {std::nullopt, std::nullopt},
              {std::optional(3)},
              {std::nullopt, std::nullopt},
              {std::optional(3)},
              {std::nullopt},
              {std::optional(3)},
              {std::nullopt},
              {std::optional(4)},
          })),
  });
  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c0", {}}};
  writerOptions.dictionaryArrayColumns = {"c0"};
  auto fileContent = test::createNimbleFile(*rootPool(), vector, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*vector->type());
  auto readers =
      makeReaders(vector, fileContent, scanSpec, stringDecoderZeroCopy);
  validate(*vector, *readers.rowReader, 2, [](auto) { return true; });
}

/*
TEST_P(SelectiveNimbleReaderTest, arrayWithOffsetsLastRowSetLifeCycle) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  std::vector<std::optional<std::vector<std::optional<int64_t>>>> c0, c1, c2;
  // First batch, one row after filtering, and allocate outputRows_ with a
  // smaller size.
  for (int i = 0; i < 16; ++i) {
    c0.emplace_back(std::nullopt);
    c1.push_back(NullableArrayData{{1}});
    c2.push_back(NullableArrayData{{}});
  }
  c0.push_back(NullableArrayData{{}});
  c1.push_back(NullableArrayData{{1}});
  c2.push_back(NullableArrayData{{}});
  // Second batch, all filtered out by c2, but c1 reads some value without
  // calling getValues.
  c0.emplace_back(std::nullopt);
  // Add a null to force setComplexNulls to read last row set before batch 3.
  c1.emplace_back(std::nullopt);
  c2.push_back(NullableArrayData{{}});
  for (int i = 0; i < 15; ++i) {
    c0.emplace_back(std::nullopt);
    c1.push_back(NullableArrayData{{2}});
    c2.push_back(NullableArrayData{{}});
  }
  c0.push_back(NullableArrayData{{}});
  c1.push_back(NullableArrayData{{2}});
  c2.emplace_back(std::nullopt);
  // Third batch, nothing is filtered out, and force outputRows_ buffer
  // reallocation.
  for (int i = 0; i < 17; ++i) {
    c0.push_back(NullableArrayData{{}});
    c1.push_back(NullableArrayData{{2}});
    c2.push_back(NullableArrayData{{}});
  }
  auto vector = makeRowVector({
      makeNullableArrayVector<int64_t>(c0),
      makeRowVector(
          {"c1c0"},
          {makeNullableArrayVector<int64_t>(c1)},
          // Add some nulls so it is not lazy.
          [](auto i) { return i == 0; }),
      makeNullableArrayVector<int64_t>(c2),
  });
  WriterOptions writerOptions;
  writerOptions.dictionaryArrayColumns = {"c1"};
  auto fileContent = test::createNimbleFile(*rootPool(), vector, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*vector->type());
  scanSpec->childByName("c0")->setFilter(std::make_shared<common::IsNotNull>());
  scanSpec->childByName("c1")->setFilter(std::make_shared<common::IsNotNull>());
  scanSpec->childByName("c2")->setFilter(std::make_shared<common::IsNotNull>());
  scanSpec->disableStatsBasedFilterReorder();
  auto readers =
      makeReaders(vector, fileContent, scanSpec, stringDecoderZeroCopy);
  validate(*vector, *readers.rowReader, 17, [](auto i) {
    return i == 16 || i >= 34;
  });
}
*/

TEST_P(SelectiveNimbleReaderTest, slidingWindowMapSubfieldPruning) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  common::BigintRange keyFilter(2, 2, false);
  checkSlidingWindowMap(
      {
          {{{1, {1}}, {2, {2}}}},
          {{{1, {1}}, {2, {2}}}},
          {{{2, {3}}}},
          {{{2, {3}}}},
          std::nullopt,
          {{{1, {4}}, {2, {5}}, {3, {6}}}},
      },
      {},
      &keyFilter,
      {1, 1, 2, 2},
      stringDecoderZeroCopy);
}

TEST_P(SelectiveNimbleReaderTest, slidingWindowMapLengthDedup) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkSlidingWindowMap(
      {
          {{{1, {2}}}},
          std::vector<std::pair<int64_t, std::optional<int64_t>>>{},
          {{{3, {4}}}},
          std::vector<std::pair<int64_t, std::optional<int64_t>>>{},
          {{{5, {6}}}},
      },
      {},
      nullptr,
      {3, 1, 1},
      stringDecoderZeroCopy);
}

// Dense path for deduplicated maps: all rows selected, no filter, single batch.
TEST_P(SelectiveNimbleReaderTest, slidingWindowMapDenseAllSelected) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkSlidingWindowMap(
      {
          {{{1, {10}}, {2, {20}}}},
          {{{3, {30}}}},
          {{{4, {40}}, {5, {50}}}},
      },
      {},
      nullptr,
      {3},
      stringDecoderZeroCopy);
}

// Dense path for deduplicated maps across multiple batches.
TEST_P(SelectiveNimbleReaderTest, slidingWindowMapDenseMultiBatch) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkSlidingWindowMap(
      {
          {{{1, {10}}}},
          {{{2, {20}}}},
          {{{3, {30}}}},
          {{{4, {40}}}},
      },
      {},
      nullptr,
      {2, 2},
      stringDecoderZeroCopy);
}

// Dense path with nulls: selectedIndices_ stays dense for non-null entries.
TEST_P(SelectiveNimbleReaderTest, slidingWindowMapDenseWithNulls) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkSlidingWindowMap(
      {
          {{{1, {10}}}},
          std::nullopt,
          {{{2, {20}}}},
          std::nullopt,
          {{{3, {30}}}},
      },
      {},
      nullptr,
      {5},
      stringDecoderZeroCopy);
}

// Sparse filter on deduplicated maps: exercises the non-dense fallback.
TEST_P(SelectiveNimbleReaderTest, slidingWindowMapSparseFilter) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkSlidingWindowMap(
      {
          {{{1, {10}}}},
          {{{2, {20}}}},
          {{{3, {30}}}},
          {{{4, {40}}}},
      },
      {true, false, true, false},
      nullptr,
      {2, 2},
      stringDecoderZeroCopy);
}

TEST_P(SelectiveNimbleReaderTest, slidingWindowMapLastRunFilteredOutAfterRead) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  checkSlidingWindowMap(
      {{{{1, {2}}}}, {{{1, {2}}}}},
      {false, true},
      nullptr,
      {1, 1},
      stringDecoderZeroCopy,
      true);
}

TEST_P(SelectiveNimbleReaderTest, schemaEvolutionRealToDouble) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeFlatVector<float>(
            101,
            [](auto i) { return sin(i); },
            hasNulls ? nullEvery(11) : nullptr);
      },
      common::DoubleRange(-INFINITY, true, false, 0.5, false, false, false),
      23,
      [&](auto& data, auto i) { return !data->isNullAt(i) && sin(i) <= 0.5; },
      stringDecoderZeroCopy,
      [&](bool hasNulls) {
        return makeFlatVector<double>(
            101,
            [](auto i) { return static_cast<float>(sin(i)); },
            hasNulls ? nullEvery(11) : nullptr);
      });
}

TEST_P(SelectiveNimbleReaderTest, schemaEvolutionBoolToTinyint) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeFlatVector<bool>(
            101,
            [](auto i) { return i % 3 == 0; },
            hasNulls ? nullEvery(11) : nullptr);
      },
      common::BigintRange(1, 100, false),
      23,
      [&](auto& data, auto i) { return !data->isNullAt(i) && i % 3 == 0; },
      stringDecoderZeroCopy,
      [&](bool hasNulls) {
        return makeFlatVector<int8_t>(
            101,
            [](auto i) { return i % 3 == 0; },
            hasNulls ? nullEvery(11) : nullptr);
      });
}

TEST_P(SelectiveNimbleReaderTest, schemaEvolutionBoolToBigint) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeFlatVector<bool>(
            101,
            [](auto i) { return i % 3 == 0; },
            hasNulls ? nullEvery(11) : nullptr);
      },
      common::BigintRange(1, 100, false),
      23,
      [&](auto& data, auto i) { return !data->isNullAt(i) && i % 3 == 0; },
      stringDecoderZeroCopy,
      [&](bool hasNulls) {
        return makeFlatVector<int64_t>(
            101,
            [](auto i) { return i % 3 == 0; },
            hasNulls ? nullEvery(11) : nullptr);
      });
}

TEST_P(SelectiveNimbleReaderTest, schemaEvolutionTinyintToBigint) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeFlatVector<int8_t>(
            101, [](auto i) { return i; }, hasNulls ? nullEvery(11) : nullptr);
      },
      common::BigintRange(20, 80, false),
      23,
      [&](auto& data, auto i) {
        return !data->isNullAt(i) && 20 <= i && i <= 80;
      },
      stringDecoderZeroCopy,
      [&](bool hasNulls) {
        return makeFlatVector<int64_t>(
            101, [](auto i) { return i; }, hasNulls ? nullEvery(11) : nullptr);
      });
}

TEST_P(SelectiveNimbleReaderTest, schemaEvolutionIntegerToBigint) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeFlatVector<int32_t>(
            101, [](auto i) { return i; }, hasNulls ? nullEvery(11) : nullptr);
      },
      common::BigintRange(20, 80, false),
      23,
      [&](auto& data, auto i) {
        return !data->isNullAt(i) && 20 <= i && i <= 80;
      },
      stringDecoderZeroCopy,
      [&](bool hasNulls) {
        return makeFlatVector<int64_t>(
            101, [](auto i) { return i; }, hasNulls ? nullEvery(11) : nullptr);
      });
}

TEST_P(SelectiveNimbleReaderTest, schemaEvolutionAddPrimitiveField) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeRowVector(
            {
                makeFlatVector<int32_t>(101, folly::identity),
            },
            hasNulls ? nullEvery(11) : nullptr);
      },
      common::IsNotNull(),
      23,
      [&](auto& data, auto i) { return !data->isNullAt(i); },
      stringDecoderZeroCopy,
      [&](bool hasNulls) {
        return makeRowVector(
            {
                makeFlatVector<int32_t>(101, folly::identity),
                makeNullConstant(TypeKind::BIGINT, 101),
            },
            hasNulls ? nullEvery(11) : nullptr);
      });
}

TEST_P(SelectiveNimbleReaderTest, schemaEvolutionAddComplexField) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeRowVector(
            {
                makeFlatVector<int32_t>(101, folly::identity),
            },
            hasNulls ? nullEvery(11) : nullptr);
      },
      common::IsNotNull(),
      23,
      [&](auto& data, auto i) { return !data->isNullAt(i); },
      stringDecoderZeroCopy,
      [&](bool hasNulls) {
        return makeRowVector(
            {
                makeFlatVector<int32_t>(101, folly::identity),
                BaseVector::createNullConstant(ARRAY(BIGINT()), 101, pool()),
            },
            hasNulls ? nullEvery(11) : nullptr);
      });
}

TEST_P(SelectiveNimbleReaderTest, schemaEvolutionAddNestedStruct) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeRowVector(
            {
                makeFlatVector<int32_t>(101, folly::identity),
                makeRowVector({
                    makeFlatVector<int32_t>(101, folly::identity),
                }),
            },
            hasNulls ? nullEvery(11) : nullptr);
      },
      common::IsNotNull(),
      23,
      [&](auto& data, auto i) { return !data->isNullAt(i); },
      stringDecoderZeroCopy,
      [&](bool hasNulls) {
        return makeRowVector(
            {
                makeFlatVector<int32_t>(101, folly::identity),
                makeRowVector({
                    makeFlatVector<int32_t>(101, folly::identity),
                    makeNullConstant(TypeKind::REAL, 101),
                }),
                BaseVector::createNullConstant(
                    ROW({"c0"}, {REAL()}), 101, pool()),
            },
            hasNulls ? nullEvery(11) : nullptr);
      });
}

TEST_P(SelectiveNimbleReaderTest, schemaEvolutionFilterOnMissingSubfield) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  // Toplevel missing fields are checked and the whole file is potentially
  // skipped in SplitReader::filterOnStats.  Maybe we should check subfields
  // there as well to avoid unnecessary IO.  For now let's make sure subfield is
  // handled in file reader.
  auto input = makeRowVector({
      makeFlatVector<int32_t>(101, folly::identity),
  });
  auto file = test::createNimbleFile(*rootPool(), input);
  auto expected = makeRowVector({
      makeFlatVector<int32_t>(101, folly::identity),
      BaseVector::createNullConstant(ROW({"c1c0"}, {BIGINT()}), 101, pool()),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*expected->type());
  auto* c1 = scanSpec->childByName("c1");
  c1->setConstantValue(
      BaseVector::createNullConstant(ROW({"c1c0"}, {BIGINT()}), 1, pool()));
  auto* c1c0 = c1->childByName("c1c0");
  c1c0->setFilter(std::make_unique<common::BigintRange>(0, 50, false));
  auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
  validate(*expected, *readers.rowReader, 101, [&](auto) { return false; });
  c1c0->setFilter(std::make_unique<common::BigintRange>(0, 50, true));
  readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
  validate(*expected, *readers.rowReader, 101, [&](auto) { return true; });
}

TEST_P(SelectiveNimbleReaderTest, nativeFlatMap) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  // Roundtrip test that takes a flat map vector and writes it to a file as a
  // storage flat map. Then reads the storage flat map as a flat map vector, and
  // compares with the initial input.
  auto testRoundtrip = [&](const FlatMapVectorPtr& inputFlatMap) {
    auto input = makeRowVector({inputFlatMap, inputFlatMap->toMapVector()});

    WriterOptions writerOptions;
    writerOptions.flatMapColumns = {{"c0", {}}};
    auto fileContent =
        test::createNimbleFile(*rootPool(), input, writerOptions);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    auto readers =
        makeReaders(input, fileContent, scanSpec, stringDecoderZeroCopy, true);

    // We invert the order on purpose to cross-compare MapVector with
    // FlatMapVectors; logically they are the same.
    auto expected = makeRowVector({inputFlatMap->toMapVector(), inputFlatMap});
    validate(*expected, *readers.rowReader, inputFlatMap->size(), [](auto) {
      return true;
    });
  };

  testRoundtrip(makeFlatMapVector<int32_t, int32_t>({}));
  testRoundtrip(makeFlatMapVector<int32_t, int32_t>({{}}));
  testRoundtrip(
      makeFlatMapVector<int16_t, float>({
          {{1, 300}},
          {{1, 400}},
          {{2, 20}},
          {{2, 30}},
          {{2, 40}},
          {{2, 50}},
          {{2, 60}},
      }));
  testRoundtrip(
      makeFlatMapVector<int16_t, float>({
          {},
          {{1, 1.9}, {2, 2.1}, {0, 3.12}},
          {{127, 0.12}},
      }));

  testRoundtrip(
      makeFlatMapVector<int16_t, float>({
          {},
          {{1, 1.9}, {2, 2.1}, {0, 3.12}},
          {{127, 0.12}},
      }));

  testRoundtrip(
      makeFlatMapVector<StringView, StringView>({
          {{"a", "a1"}},
          {{"b", "b1"}},
          {{"c", "c1"}},
          {{"d", "d1"}},
      }));

  testRoundtrip(
      makeNullableFlatMapVector<int32_t, int32_t>({
          {{{101, 1}, {102, 2}, {103, 3}}},
          {{{105, 0}, {106, 0}}},
          {std::nullopt},
          {{{101, 11}, {103, 13}, {105, std::nullopt}}},
          {{{101, 1}, {102, 2}, {103, 3}}},
      }));

  auto constructFlatMap = [&](VectorPtr keys) {
    return std::make_shared<FlatMapVector>(
        pool(),
        MAP(INTEGER(), INTEGER()),
        nullptr,
        3,
        keys,
        std::vector<VectorPtr>{
            makeFlatVector<int32_t>({1, 2, 3}),
            makeFlatVector<int32_t>({4, 5, 6}),
            makeFlatVector<int32_t>({7, 8, 9})},
        std::vector<BufferPtr>{nullptr, nullptr, nullptr});
  };

  // Dictionary wrapped keys
  {
    testRoundtrip(constructFlatMap(
        BaseVector::wrapInDictionary(
            nullptr,
            makeIndices({0, 1, 2}),
            3,
            makeFlatVector<int32_t>({4, 5, 6}))));

    testRoundtrip(constructFlatMap(
        BaseVector::wrapInDictionary(
            nullptr,
            makeIndices({2, 0, 1}),
            3,
            makeFlatVector<int32_t>({4, 5, 6}))));

    // Difference in cardinality
    testRoundtrip(constructFlatMap(
        BaseVector::wrapInDictionary(
            nullptr,
            makeIndices({1, 3, 5}),
            3,
            makeFlatVector<int32_t>({4, 5, 6, 7, 8, 9}))));

    NIMBLE_ASSERT_THROW(
        testRoundtrip(constructFlatMap(
            BaseVector::wrapInDictionary(
                nullptr,
                makeIndices({2, 1, 1}),
                3,
                makeFlatVector<int32_t>({4, 5, 6})))),
        "FlatMapVector keys are not distinct.");
  }

  // Constant wrapped keys
  {
    NIMBLE_ASSERT_THROW(
        testRoundtrip(constructFlatMap(
            BaseVector::wrapInConstant(
                3, 0, makeFlatVector<int32_t>({1, 2, 3})))),
        "FlatMapVector keys are not distinct.");
  }

  // Scan spec with map key filter
  {
    auto inputFlatMap = makeFlatMapVector<int64_t, int64_t>({
        {{{1, 100}, {5, 500}, {10, 1000}, {15, 1500}, {20, 2000}}},
        {{{2, 200}, {8, 800}, {12, 1200}}},
        {{{3, 300}, {7, 700}, {18, 1800}, {25, 2500}}},
    });
    auto input = makeRowVector({inputFlatMap, inputFlatMap->toMapVector()});

    WriterOptions writerOptions;
    writerOptions.flatMapColumns = {{"c0", {}}};
    auto fileContent =
        test::createNimbleFile(*rootPool(), input, writerOptions, true);

    // Test with map key filter [5, 12]
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    scanSpec->childByName("c0")
        ->childByName(common::ScanSpec::kMapKeysFieldName)
        ->setFilter(std::make_unique<common::BigintRange>(5, 12, false));
    scanSpec->childByName("c1")
        ->childByName(common::ScanSpec::kMapKeysFieldName)
        ->setFilter(std::make_unique<common::BigintRange>(5, 12, false));
    auto readers = makeReaders(input, fileContent, scanSpec, true);

    // Expected output after filtering: only keys in [5, 12] remain
    auto expectedFlatMap = makeFlatMapVector<int64_t, int64_t>({
        {{{5, 500}, {10, 1000}}},
        {{{8, 800}, {12, 1200}}},
        {{{7, 700}}},
    });
    auto expected =
        makeRowVector({expectedFlatMap->toMapVector(), expectedFlatMap});
    validate(*expected, *readers.rowReader, inputFlatMap->size(), [](auto) {
      return true;
    });
  }
}

TEST_P(SelectiveNimbleReaderTest, mapAsStruct) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto input = makeRowVector({
      makeMapVector<int32_t, int64_t>({{{1, 4}, {2, 5}}, {{1, 6}, {3, 7}}}),
  });
  auto outType = ROW({"c0"}, {ROW({"3", "1"}, BIGINT())});
  auto spec = std::make_shared<common::ScanSpec>("<root>");
  spec->addAllChildFields(*outType);
  spec->childByName("c0")->setFlatMapAsStruct(true);
  auto readers = makeReaders(input, spec, stringDecoderZeroCopy);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, batch), 2);
  auto expected = makeRowVector({
      makeRowVector(
          {"3", "1"},
          {
              makeNullableFlatVector<int64_t>({std::nullopt, 7}),
              makeFlatVector<int64_t>({4, 6}),
          }),
  });
  velox::test::assertEqualVectors(expected, batch);
}

TEST_P(SelectiveNimbleReaderTest, mapAsStructFilterAfterRead) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto input = makeRowVector({
      makeMapVector<int32_t, int64_t>({{{1, 4}, {2, 5}}, {}, {{1, 6}, {3, 7}}}),
      makeRowVector(
          {makeConstant<int64_t>(0, 3)}, [](auto i) { return i == 0; }),
  });
  auto outType =
      ROW({"c0", "c1"}, {ROW({"3", "1"}, BIGINT()), ROW({"c0"}, BIGINT())});
  auto spec = std::make_shared<common::ScanSpec>("<root>");
  spec->addAllChildFields(*outType);
  auto* c0Spec = spec->childByName("c0");
  c0Spec->setFlatMapAsStruct(true);
  c0Spec->setFilter(std::make_shared<common::IsNotNull>());
  spec->childByName("c1")->setFilter(std::make_shared<common::IsNotNull>());
  auto readers = makeReaders(input, spec, stringDecoderZeroCopy);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, batch), 3);
  auto expected = makeRowVector({
      makeRowVector(
          {"3", "1"},
          {
              makeNullableFlatVector<int64_t>({std::nullopt, 7}),
              makeNullableFlatVector<int64_t>({std::nullopt, 6}),
          }),
      makeRowVector({makeConstant<int64_t>(0, 2)}),
  });
  velox::test::assertEqualVectors(expected, batch);
}

TEST_P(SelectiveNimbleReaderTest, mapAsStructAllEmpty) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto input = makeRowVector({makeMapVector<int32_t, int64_t>({{}, {}})});
  auto outType = ROW({"c0"}, {ROW({"1"}, BIGINT())});
  auto spec = std::make_shared<common::ScanSpec>("<root>");
  spec->addAllChildFields(*outType);
  spec->childByName("c0")->setFlatMapAsStruct(true);
  auto readers = makeReaders(input, spec, stringDecoderZeroCopy);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, batch), 2);
  auto expected = makeRowVector({
      makeRowVector({"1"}, {makeNullConstant(TypeKind::BIGINT, 2)}),
  });
  velox::test::assertEqualVectors(expected, batch);
}

TEST_P(SelectiveNimbleReaderTest, mapAsStructAllNulls) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto input = makeRowVector(
      {makeNullableMapVector<int32_t, int64_t>({std::nullopt, std::nullopt})});
  auto outType = ROW({"c0"}, {ROW({"1"}, BIGINT())});
  auto spec = std::make_shared<common::ScanSpec>("<root>");
  spec->addAllChildFields(*outType);
  spec->childByName("c0")->setFlatMapAsStruct(true);
  auto readers = makeReaders(input, spec, stringDecoderZeroCopy);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, batch), 2);
  auto expected = makeRowVector({
      BaseVector::createNullConstant(ROW({"1"}, BIGINT()), 2, pool()),
  });
  velox::test::assertEqualVectors(expected, batch);
}

TEST_P(SelectiveNimbleReaderTest, columnDecodeMetrics) {
  GTEST_SKIP() << "Per column decode counters moved from RuntimeStats to "
                  "SplitStats, which readers own privately and do not expose. "
                  "Re-enable once Velox provides caller access.";
  const int numRows = 100'000;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(numRows, [](auto i) { return i * 7 + 13; }),
      makeFlatVector<std::string>(
          numRows, [](auto i) { return std::string(20, 'a' + (i % 26)); }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*asRowType(input->type()));
  nimble::CompressionOptions comprOpts;
  comprOpts.compressionAcceptRatio = 100.0f;
  comprOpts.compressionAcceptRatioOverrides = {};
  nimble::ManualEncodingSelectionPolicyFactory encodingFactory(
      {{{nimble::EncodingType::Trivial, 1.0}}}, comprOpts);
  nimble::WriterOptions writerOptions;
  writerOptions.encodingSelectionPolicyCreator = [&](nimble::DataType dt) {
    return encodingFactory.createPolicy(dt);
  };
  auto file =
      test::createNimbleFile(*rootPool(), input, std::move(writerOptions));
  auto readFile = std::make_shared<InMemoryReadFile>(file);
  auto factory =
      dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
  dwio::common::ReaderOptions options(pool());
  options.setDataIoStats(dataIoStats_);
  options.setMetadataIoStats(metadataIoStats_);
  options.setScanSpec(scanSpec);
  auto reader = factory->createReader(
      std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
      options);
  dwio::common::RowReaderOptions rowOptions;
  rowOptions.setScanSpec(scanSpec);
  rowOptions.setRequestedType(asRowType(input->type()));
  // zeroCopy=true uses nimble::EncodingFactory (not legacy), which forwards
  // Encoding::Options including decodingStats to encoding constructors.
  rowOptions.setStringDecoderZeroCopy(true);
  rowOptions.setCollectColumnStats(true);
  rowOptions.setEagerFirstStripeLoad(true);
  auto rowReader = reader->createRowReader(rowOptions);

  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  uint64_t totalRows = 0;
  while (auto n = rowReader->next(1'000, result)) {
    totalRows += n;
    auto* row = result->as<RowVector>();
    for (auto i = 0; i < row->childrenSize(); ++i) {
      row->childAt(i)->loadedVector();
    }
  }
  EXPECT_EQ(totalRows, numRows) << "should read all rows";

  // The per column decode and decompress assertions that used to live here
  // read counters off RuntimeStatistics::columnReaderStats. Velox moved those
  // counters onto SplitStats, which every reader owns privately and none
  // exposes, so there is no supported way to reach them from a caller. The
  // skip above records that this coverage is currently unavailable.
}

// Tests for FixedBitWidthEncoding fast path.
// The fast path is used for 4-byte integral types without filters or hooks.

TEST_P(SelectiveNimbleReaderTest, fixedBitWidthFastPathSameType) {
  // Tests that the fast path works correctly for same-type reads (int32 →
  // int32). This exercises the memcpy branch in bulkScan.
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  for (bool hasNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("hasNulls={}", hasNulls));
    auto data = makeFlatVector<int32_t>(
        1000, [](auto i) { return i * 7; }, hasNulls ? nullEvery(17) : nullptr);
    auto input = makeRowVector({data});
    auto file = test::createNimbleFile(*rootPool(), input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
    // Read in a single large batch to exercise the dense fast path.
    validate(*input, *readers.rowReader, 1000, -1, [](auto) { return true; });
  }
}

TEST_P(SelectiveNimbleReaderTest, fixedBitWidthFastPathUpcast) {
  // Tests that the fast path works correctly for upcast reads (int32 →
  // int64). This exercises the upcast loop branch in bulkScan.
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  for (bool hasNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("hasNulls={}", hasNulls));
    // Write as int32.
    auto writeData = makeFlatVector<int32_t>(
        1000, [](auto i) { return i * 7; }, hasNulls ? nullEvery(17) : nullptr);
    auto input = makeRowVector({writeData});
    auto file = test::createNimbleFile(*rootPool(), input);

    // Read as int64 (schema evolution / upcast).
    auto readData = makeFlatVector<int64_t>(
        1000, [](auto i) { return i * 7; }, hasNulls ? nullEvery(17) : nullptr);
    auto expected = makeRowVector({readData});
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*expected->type());
    auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
    // Read in a single large batch to exercise the dense fast path.
    validate(
        *expected, *readers.rowReader, 1000, -1, [](auto) { return true; });
  }
}

TEST_P(SelectiveNimbleReaderTest, fixedBitWidthFastPathMultipleBatches) {
  // Tests that the fast path works correctly with multiple batches.
  // This exercises the row tracking and position management.
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  for (bool hasNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("hasNulls={}", hasNulls));
    auto data = makeFlatVector<int32_t>(
        500,
        [](auto i) { return i * 3 + 1; },
        hasNulls ? nullEvery(11) : nullptr);
    auto input = makeRowVector({data});
    auto file = test::createNimbleFile(*rootPool(), input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
    // Read in small batches to test batch boundary handling.
    validate(*input, *readers.rowReader, 37, -1, [](auto) { return true; });
  }
}

TEST_P(SelectiveNimbleReaderTest, fixedBitWidthFastPathLargeValues) {
  // Tests that the fast path handles large values correctly.
  // This verifies baseline handling in the encoding.
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  for (bool hasNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("hasNulls={}", hasNulls));
    auto data = makeFlatVector<int32_t>(
        200,
        [](auto i) { return static_cast<int32_t>(1'000'000'000 + i * 100); },
        hasNulls ? nullEvery(13) : nullptr);
    auto input = makeRowVector({data});
    auto file = test::createNimbleFile(*rootPool(), input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
    validate(*input, *readers.rowReader, 200, -1, [](auto) { return true; });
  }
}

TEST_P(SelectiveNimbleReaderTest, fixedBitWidthFastPathNegativeValues) {
  // Tests that the fast path handles negative values correctly.
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  for (bool hasNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("hasNulls={}", hasNulls));
    auto data = makeFlatVector<int32_t>(
        300,
        [](auto i) { return static_cast<int32_t>(i) - 150; },
        hasNulls ? nullEvery(19) : nullptr);
    auto input = makeRowVector({data});
    auto file = test::createNimbleFile(*rootPool(), input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
    validate(*input, *readers.rowReader, 300, -1, [](auto) { return true; });
  }
}

TEST_P(SelectiveNimbleReaderTest, fixedBitWidthFastPathUpcastNegative) {
  // Tests that the fast path handles upcast with negative values correctly.
  // This verifies sign extension works properly.
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  for (bool hasNulls : {false, true}) {
    SCOPED_TRACE(fmt::format("hasNulls={}", hasNulls));
    // Write as int32.
    auto writeData = makeFlatVector<int32_t>(
        200,
        [](auto i) { return static_cast<int32_t>(i) - 100; },
        hasNulls ? nullEvery(11) : nullptr);
    auto input = makeRowVector({writeData});
    auto file = test::createNimbleFile(*rootPool(), input);

    // Read as int64.
    auto readData = makeFlatVector<int64_t>(
        200,
        [](auto i) { return static_cast<int64_t>(i) - 100; },
        hasNulls ? nullEvery(11) : nullptr);
    auto expected = makeRowVector({readData});
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*expected->type());
    auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
    validate(*expected, *readers.rowReader, 200, -1, [](auto) { return true; });
  }
}

TEST_P(SelectiveNimbleReaderTest, pinMetadata) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto data = makeFlatVector<int64_t>(100, [](auto i) { return i; });
  auto input = makeRowVector({data});
  auto file = test::createNimbleFile(*rootPool(), input);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());

  for (bool enableCache : {false, true}) {
    SCOPED_TRACE(fmt::format("enableCache {}", enableCache));

    auto readFile = std::make_shared<InMemoryReadFile>(file);
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions options(pool());
    options.setDataIoStats(dataIoStats_);
    options.setMetadataIoStats(metadataIoStats_);
    options.setScanSpec(scanSpec);
    options.setPinMetadata(true);
    if (enableCache) {
      options.setCacheMetadata(true);
    }
    Readers readers;
    readers.reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
        options);
    auto type = asRowType(input->type());
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    rowOptions.setRequestedType(type);
    rowOptions.setStringDecoderZeroCopy(stringDecoderZeroCopy);
    readers.rowReader = readers.reader->createRowReader(rowOptions);
    validate(*input, *readers.rowReader, 100, [](auto) { return true; });
  }
}

// Verifies that within the same TabletReader, the second VeloxReader pass
// does not re-read metadata. The weak-pointer cache retains entries while the
// tablet is alive, so metadata is always reused regardless of pinMetadata,
// cacheMetadata, or whether cache infrastructure is provided.
TEST_P(SelectiveNimbleReaderTest, metadataReuseWithSameReader) {
  if (!GetParam().enableCache) {
    GTEST_SKIP() << "Only applicable when cache is enabled";
  }

  struct TestCase {
    bool pinMetadata;
    bool cacheMetadata;
    bool provideCache;
    bool expectRamHit;

    std::string debugString() const {
      return fmt::format(
          "pinMetadata={}, cacheMetadata={}, provideCache={}, "
          "expectRamHit={}",
          pinMetadata,
          cacheMetadata,
          provideCache,
          expectRamHit);
    }
  };

  std::vector<TestCase> testCases = {
      {true, true, true, true},
      {true, false, true, false},
      {false, true, true, true},
      {false, false, true, false},
      {false, true, false, false},
      {false, false, false, false},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    auto data = makeFlatVector<int64_t>(100, [](auto i) { return i; });
    auto input = makeRowVector({data});
    auto file = test::createNimbleFile(*rootPool(), input);

    auto delegate = std::make_shared<InMemoryReadFile>(file);
    auto trackingFile =
        std::make_shared<::facebook::nimble::testing::TrackingReadFile>(
            delegate);

    auto selector = std::make_shared<dwio::common::ColumnSelector>(
        std::dynamic_pointer_cast<const velox::RowType>(input->type()));

    TabletReader::Options tabletOptions;
    tabletOptions.pinMetadata = testCase.pinMetadata;
    tabletOptions.cacheMetadata = testCase.cacheMetadata;
    io::ReaderOptions ioOpts(pool());
    ioOpts.setDataIoStats(dataIoStats_);
    ioOpts.setMetadataIoStats(metadataIoStats_);
    ioOpts.setIndexIoStats(indexIoStats_);
    tabletOptions.ioOptions = ioOpts;
    std::unique_ptr<FileHandle> fileHandle;
    if (testCase.provideCache) {
      auto& ids = fileIds();
      auto fileIdStr = fmt::format(
          "sameReaderSelectiveTest_pin{}_cache{}_provide{}",
          testCase.pinMetadata,
          testCase.cacheMetadata,
          testCase.provideCache);
      fileHandle = std::make_unique<FileHandle>();
      fileHandle->file = trackingFile;
      fileHandle->uuid = StringIdLease(ids, fileIdStr);
      fileHandle->groupId = StringIdLease(ids, fileIdStr + "_group");
      tabletOptions.cache = cache_.get();
      tabletOptions.fileHandle = fileHandle.get();
    }

    auto tablet = TabletReader::create(trackingFile, pool(), tabletOptions);

    nimble::VeloxReadParams readParams;
    readParams.decodingExecutor =
        std::make_shared<folly::CPUThreadPoolExecutor>(1);

    auto reader1 = std::make_unique<nimble::VeloxReader>(
        tablet, *pool(), selector, readParams);
    VectorPtr result;
    ASSERT_TRUE(reader1->next(100, result));
    ASSERT_EQ(result->size(), 100);
    ASSERT_FALSE(reader1->next(1, result));

    const auto stripeGroupsMeta = tablet->stripeGroupsMetadata();
    ASSERT_EQ(stripeGroupsMeta.size(), 1);
    const auto metadataBoundary = stripeGroupsMeta[0].offset();

    ASSERT_GT(trackingFile->maxReadOffset(), metadataBoundary)
        << "First pass should read metadata from the file";

    const auto ramHitsAfterFirstPass = metadataIoStats_->ramHit().count();

    // Second pass: new VeloxReader on the same TabletReader. Metadata is
    // always reused within the same tablet regardless of settings.
    trackingFile->resetMaxReadOffset();
    auto reader2 = std::make_unique<nimble::VeloxReader>(
        tablet, *pool(), selector, readParams);
    ASSERT_TRUE(reader2->next(100, result));
    ASSERT_EQ(result->size(), 100);
    ASSERT_FALSE(reader2->next(1, result));

    ASSERT_LE(trackingFile->maxReadOffset(), metadataBoundary)
        << "Second pass should not re-read metadata. "
        << "maxReadOffset=" << trackingFile->maxReadOffset()
        << " metadataBoundary=" << metadataBoundary;

    if (testCase.expectRamHit) {
      ASSERT_GT(metadataIoStats_->ramHit().count(), ramHitsAfterFirstPass)
          << "Second pass should have RAM cache hits";
    } else {
      ASSERT_EQ(metadataIoStats_->ramHit().count(), ramHitsAfterFirstPass)
          << "Second pass should have no new RAM cache hits";
    }
  }
}

// Tests metadata reuse across separate TabletReader instances. Only
// cacheMetadata persists across TabletReader opens via AsyncDataCache.
// pinMetadata only holds strong references within the same TabletReader.
TEST_P(SelectiveNimbleReaderTest, metadataReuseCrossReaders) {
  if (!GetParam().enableCache) {
    GTEST_SKIP() << "Only applicable when cache is enabled";
  }

  struct TestCase {
    bool pinMetadata;
    bool cacheMetadata;
    bool provideCache;
    bool expectCrossReaderReuse;
    bool expectRamHit;

    std::string debugString() const {
      return fmt::format(
          "pinMetadata={}, cacheMetadata={}, provideCache={}, "
          "expectCrossReaderReuse={}, expectRamHit={}",
          pinMetadata,
          cacheMetadata,
          provideCache,
          expectCrossReaderReuse,
          expectRamHit);
    }
  };

  std::vector<TestCase> testCases = {
      {true, true, true, true, true},
      {true, false, true, false, false},
      {false, true, true, true, true},
      {false, false, true, false, false},
      // cacheMetadata is set but cache is not provided — silently falls back
      // to direct IO, so cross-reader reuse and RAM hits are not expected.
      {false, true, false, false, false},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    auto data = makeFlatVector<int64_t>(100, [](auto i) { return i; });
    auto input = makeRowVector({data});
    auto file = test::createNimbleFile(*rootPool(), input);

    auto delegate = std::make_shared<InMemoryReadFile>(file);
    auto trackingFile =
        std::make_shared<::facebook::nimble::testing::TrackingReadFile>(
            delegate);

    auto selector = std::make_shared<dwio::common::ColumnSelector>(
        std::dynamic_pointer_cast<const velox::RowType>(input->type()));

    auto& ids = fileIds();
    auto fileIdStr = fmt::format(
        "crossReaderSelectiveTest_pin{}_cache{}",
        testCase.pinMetadata,
        testCase.cacheMetadata);
    StringIdLease fileId(ids, fileIdStr);
    StringIdLease groupId(ids, fileIdStr + "_group");

    auto makeTablet =
        [&](const std::shared_ptr<io::IoStatistics>& metadataStats,
            std::unique_ptr<FileHandle>& fileHandleOut) {
          TabletReader::Options tabletOptions;
          tabletOptions.pinMetadata = testCase.pinMetadata;
          tabletOptions.cacheMetadata = testCase.cacheMetadata;
          io::ReaderOptions ioOpts(pool());
          ioOpts.setDataIoStats(dataIoStats_);
          ioOpts.setMetadataIoStats(metadataStats);
          ioOpts.setIndexIoStats(indexIoStats_);
          tabletOptions.ioOptions = ioOpts;
          if (testCase.provideCache) {
            fileHandleOut = std::make_unique<FileHandle>();
            fileHandleOut->file = trackingFile;
            fileHandleOut->uuid = fileId;
            fileHandleOut->groupId = groupId;
            tabletOptions.cache = cache_.get();
            tabletOptions.fileHandle = fileHandleOut.get();
          }
          return TabletReader::create(trackingFile, pool(), tabletOptions);
        };

    // First open + first pass.
    auto metadataIoStats1 = std::make_shared<io::IoStatistics>();
    std::unique_ptr<FileHandle> fh1;
    auto tablet1 = makeTablet(metadataIoStats1, fh1);

    const auto stripeGroupsMeta = tablet1->stripeGroupsMetadata();
    ASSERT_EQ(stripeGroupsMeta.size(), 1);
    const auto metadataBoundary = stripeGroupsMeta[0].offset();

    nimble::VeloxReadParams readParams;
    readParams.decodingExecutor =
        std::make_shared<folly::CPUThreadPoolExecutor>(1);
    auto reader1 = std::make_unique<nimble::VeloxReader>(
        tablet1, *pool(), selector, readParams);
    VectorPtr result;
    ASSERT_TRUE(reader1->next(100, result));
    ASSERT_EQ(result->size(), 100);
    ASSERT_FALSE(reader1->next(1, result));

    ASSERT_GT(trackingFile->maxReadOffset(), metadataBoundary)
        << "First pass should read metadata from the file";
    ASSERT_GT(metadataIoStats1->rawBytesRead(), 0)
        << "First pass should record metadata IO";

    // Cross-reader: destroy everything and open a new TabletReader.
    reader1.reset();
    tablet1.reset();
    fh1.reset();
    trackingFile->resetMaxReadOffset();

    auto metadataIoStats2 = std::make_shared<io::IoStatistics>();
    std::unique_ptr<FileHandle> fh2;
    auto tablet2 = makeTablet(metadataIoStats2, fh2);
    auto reader3 = std::make_unique<nimble::VeloxReader>(
        tablet2, *pool(), selector, readParams);
    ASSERT_TRUE(reader3->next(100, result));
    ASSERT_EQ(result->size(), 100);

    if (testCase.expectCrossReaderReuse) {
      ASSERT_LE(trackingFile->maxReadOffset(), metadataBoundary)
          << "Cross-reader should not re-read metadata. "
          << "maxReadOffset=" << trackingFile->maxReadOffset()
          << " metadataBoundary=" << metadataBoundary;
      ASSERT_EQ(metadataIoStats2->rawBytesRead(), 0)
          << "Cross-reader should not record any metadata IO";
    } else {
      ASSERT_GT(trackingFile->maxReadOffset(), metadataBoundary)
          << "Cross-reader should re-read metadata. "
          << "maxReadOffset=" << trackingFile->maxReadOffset()
          << " metadataBoundary=" << metadataBoundary;
      ASSERT_GT(metadataIoStats2->rawBytesRead(), 0)
          << "Cross-reader should record metadata IO";
    }

    if (testCase.expectRamHit) {
      ASSERT_GT(metadataIoStats2->ramHit().count(), 0)
          << "Should have RAM cache hits";
    } else {
      ASSERT_EQ(metadataIoStats2->ramHit().count(), 0)
          << "Should have no RAM cache hits";
    }
  }
}

// ---------------------------------------------------------------------------
// DeltaEncoding: monotonically increasing data with filter
// ---------------------------------------------------------------------------
TEST_P(SelectiveNimbleReaderTest, delta) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  runTestCase(
      [&](bool hasNulls) {
        return makeFlatVector<int64_t>(
            101,
            [](auto i) { return i * 3; },
            hasNulls ? nullEvery(11) : nullptr);
      },
      common::BigintRange(50, 200, false),
      23,
      [&](auto& data, auto i) {
        return !data->isNullAt(i) && data->valueAt(i) >= 50 &&
            data->valueAt(i) <= 200;
      },
      stringDecoderZeroCopy);
}

// ---------------------------------------------------------------------------
// HuffmanEncoding: filtered reads across random-access checkpoints
// ---------------------------------------------------------------------------
TEST_P(SelectiveNimbleReaderTest, huffmanFilteredAcrossCheckpoints) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto c0 = makeFlatVector<int64_t>(1025, [](auto i) {
    return i % 10 == 0 ? static_cast<int64_t>(100 + i % 7)
                       : static_cast<int64_t>(i % 4);
  });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(102, 104, false));

  auto file = test::createNimbleFile(
      *rootPool(),
      input,
      {.encodingLayoutTree = EncodingLayoutTree{
           Kind::Row,
           {},
           "",
           {{Kind::Scalar,
             {{0,
               EncodingLayout{
                   EncodingType::Huffman, {}, CompressionType::Uncompressed}}},
             ""}}}});
  verifyEncodingOnDisk(file, EncodingType::Huffman, /*isNullable=*/false);

  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 73, [&](auto i) {
    return c0->valueAt(i) >= 102 && c0->valueAt(i) <= 104;
  });
}

// ---------------------------------------------------------------------------
// DeltaEncoding: forced via encoding layout
// ---------------------------------------------------------------------------
TEST_P(SelectiveNimbleReaderTest, deltaForcedEncoding) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  // Monotonically increasing data ideal for delta encoding.
  auto c0 =
      makeFlatVector<int64_t>(501, [](auto i) { return i * 7; }, nullEvery(13));
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(100, 2000, false));
  auto file = test::createNimbleFile(
      *rootPool(),
      input,
      {.encodingLayoutTree = EncodingLayoutTree{
           Kind::Row,
           {},
           "",
           {{Kind::Scalar,
             {{0,
               EncodingLayout{
                   EncodingType::Delta,
                   {},
                   CompressionType::Uncompressed,
                   {EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed},
                    EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed},
                    EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed}}}}},
             ""}}}});
  verifyEncodingOnDisk(file, EncodingType::Delta, /*isNullable=*/false);
  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 101, [&](auto i) {
    return !c0->isNullAt(i) && c0->valueAt(i) >= 100 && c0->valueAt(i) <= 2000;
  });
}

// ---------------------------------------------------------------------------
// DeltaEncoding: data with restatements (non-monotonic)
// ---------------------------------------------------------------------------
TEST_P(SelectiveNimbleReaderTest, deltaWithRestatements) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  // Data that resets periodically, causing restatements in delta encoding.
  auto c0 = makeFlatVector<int64_t>(
      301,
      [](auto i) { return static_cast<int64_t>((i % 50) * 2 + (i / 50) * 10); },
      nullEvery(17));
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(20, 80, false));
  auto file = test::createNimbleFile(
      *rootPool(),
      input,
      {.encodingLayoutTree = EncodingLayoutTree{
           Kind::Row,
           {},
           "",
           {{Kind::Scalar,
             {{0,
               EncodingLayout{
                   EncodingType::Delta,
                   {},
                   CompressionType::Uncompressed,
                   {EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed},
                    EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed},
                    EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed}}}}},
             ""}}}});
  verifyEncodingOnDisk(file, EncodingType::Delta, /*isNullable=*/false);
  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 50, [&](auto i) {
    return !c0->isNullAt(i) && c0->valueAt(i) >= 20 && c0->valueAt(i) <= 80;
  });
}

// ---------------------------------------------------------------------------
// DeltaEncoding: int32_t data (verifies non-int64 types work through the
// selective reader with Delta encoding).
// ---------------------------------------------------------------------------
TEST_P(SelectiveNimbleReaderTest, deltaInt32) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto c0 =
      makeFlatVector<int32_t>(401, [](auto i) { return i * 3; }, nullEvery(11));
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(50, 800, false));
  auto file = test::createNimbleFile(
      *rootPool(),
      input,
      {.encodingLayoutTree = EncodingLayoutTree{
           Kind::Row,
           {},
           "",
           {{Kind::Scalar,
             {{0,
               EncodingLayout{
                   EncodingType::Delta,
                   {},
                   CompressionType::Uncompressed,
                   {EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed},
                    EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed},
                    EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed}}}}},
             ""}}}});
  verifyEncodingOnDisk(file, EncodingType::Delta, /*isNullable=*/false);
  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 80, [&](auto i) {
    return !c0->isNullAt(i) && c0->valueAt(i) >= 50 && c0->valueAt(i) <= 800;
  });
}

// ---------------------------------------------------------------------------
// DeltaEncoding: no filter applied — reads all rows from a delta-encoded file.
// ---------------------------------------------------------------------------
TEST_P(SelectiveNimbleReaderTest, deltaNoFilter) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto c0 = makeFlatVector<int64_t>(200, [](auto i) { return i * 5; });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto file = test::createNimbleFile(
      *rootPool(),
      input,
      {.encodingLayoutTree = EncodingLayoutTree{
           Kind::Row,
           {},
           "",
           {{Kind::Scalar,
             {{0,
               EncodingLayout{
                   EncodingType::Delta,
                   {},
                   CompressionType::Uncompressed,
                   {EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed},
                    EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed},
                    EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed}}}}},
             ""}}}});
  verifyEncodingOnDisk(file, EncodingType::Delta, /*isNullable=*/false);
  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 50, [](auto) { return true; });
}

// ---------------------------------------------------------------------------
// DeltaEncoding: two columns both using delta encoding with different data
// patterns, verifying multi-column delta support in selective reader.
// ---------------------------------------------------------------------------
TEST_P(SelectiveNimbleReaderTest, deltaTwoColumns) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto c0 = makeFlatVector<int64_t>(300, [](auto i) { return i * 7; });
  auto c1 = makeFlatVector<int32_t>(
      300, [](auto i) { return static_cast<int32_t>((i % 30) * 2); });
  auto input = makeRowVector({c0, c1});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(100, 1500, false));
  EncodingLayout deltaLayout{
      EncodingType::Delta,
      {},
      CompressionType::Uncompressed,
      {EncodingLayout{EncodingType::Trivial, {}, CompressionType::Uncompressed},
       EncodingLayout{EncodingType::Trivial, {}, CompressionType::Uncompressed},
       EncodingLayout{
           EncodingType::Trivial, {}, CompressionType::Uncompressed}}};
  auto file = test::createNimbleFile(
      *rootPool(),
      input,
      {.encodingLayoutTree = EncodingLayoutTree{
           Kind::Row,
           {},
           "",
           {{Kind::Scalar, {{0, deltaLayout}}, ""},
            {Kind::Scalar, {{0, deltaLayout}}, ""}}}});
  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 60, [&](auto i) {
    return c0->valueAt(i) >= 100 && c0->valueAt(i) <= 1500;
  });
}

// ---------------------------------------------------------------------------
// DeltaEncoding: sawtooth pattern with small batch reads to exercise
// chunk boundary handling.
// ---------------------------------------------------------------------------
TEST_P(SelectiveNimbleReaderTest, deltaSawtoothSmallBatches) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();
  auto c0 = makeFlatVector<int64_t>(
      200, [](auto i) { return static_cast<int64_t>((i % 20) * 5); });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto file = test::createNimbleFile(
      *rootPool(),
      input,
      {.encodingLayoutTree = EncodingLayoutTree{
           Kind::Row,
           {},
           "",
           {{Kind::Scalar,
             {{0,
               EncodingLayout{
                   EncodingType::Delta,
                   {},
                   CompressionType::Uncompressed,
                   {EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed},
                    EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed},
                    EncodingLayout{
                        EncodingType::Trivial,
                        {},
                        CompressionType::Uncompressed}}}}},
             ""}}}});
  verifyEncodingOnDisk(file, EncodingType::Delta, /*isNullable=*/false);
  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  // Small batch size (7) to exercise partial-read boundaries.
  validate(*input, *readers.rowReader, 7, [](auto) { return true; });
}

// Verifies columnStatistics returns IntegerColumnStatistics for BIGINT columns
// with correct value count and null status.
TEST_P(SelectiveNimbleReaderTest, columnStatisticsInteger) {
  constexpr int kSize = 100;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, [](auto i) { return i * 3; }),
  });
  WriterOptions writerOptions;
  writerOptions.enableVectorizedStats = true;
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers =
      makeReaders(input, fileContent, scanSpec, this->stringDecoderZeroCopy());

  // Column 0 is the root ROW -- column 1 is the first child (c0).
  auto stats = readers.reader->columnStatistics(1);
  ASSERT_NE(stats, nullptr);
  ASSERT_TRUE(stats->getNumberOfValues().has_value());
  EXPECT_EQ(stats->getNumberOfValues().value(), kSize);
  ASSERT_TRUE(stats->hasNull().has_value());
  EXPECT_FALSE(stats->hasNull().value());
  auto* intStats =
      dynamic_cast<dwio::common::IntegerColumnStatistics*>(stats.get());
  ASSERT_NE(intStats, nullptr);
  ASSERT_TRUE(intStats->getMinimum().has_value());
  EXPECT_EQ(intStats->getMinimum().value(), 0);
  ASSERT_TRUE(intStats->getMaximum().has_value());
  EXPECT_EQ(intStats->getMaximum().value(), 297);
}

// Verifies columnStatistics returns DoubleColumnStatistics for DOUBLE columns.
TEST_P(SelectiveNimbleReaderTest, columnStatisticsDouble) {
  constexpr int kSize = 50;
  auto input = makeRowVector({
      makeFlatVector<double>(kSize, [](auto i) { return i * 1.5; }),
  });
  WriterOptions writerOptions;
  writerOptions.enableVectorizedStats = true;
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers =
      makeReaders(input, fileContent, scanSpec, this->stringDecoderZeroCopy());

  auto stats = readers.reader->columnStatistics(1);
  ASSERT_NE(stats, nullptr);
  auto* fpStats =
      dynamic_cast<dwio::common::DoubleColumnStatistics*>(stats.get());
  ASSERT_NE(fpStats, nullptr);
  ASSERT_TRUE(stats->getNumberOfValues().has_value());
  EXPECT_EQ(stats->getNumberOfValues().value(), kSize);
  ASSERT_TRUE(fpStats->getMinimum().has_value());
  EXPECT_DOUBLE_EQ(fpStats->getMinimum().value(), 0.0);
  ASSERT_TRUE(fpStats->getMaximum().has_value());
  EXPECT_DOUBLE_EQ(fpStats->getMaximum().value(), 73.5);
}

// Verifies columnStatistics returns StringColumnStatistics for VARCHAR columns.
TEST_P(SelectiveNimbleReaderTest, columnStatisticsString) {
  auto input = makeRowVector({
      makeFlatVector<std::string>({"apple", "banana", "cherry"}),
  });
  WriterOptions writerOptions;
  writerOptions.enableVectorizedStats = true;
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers =
      makeReaders(input, fileContent, scanSpec, this->stringDecoderZeroCopy());

  auto stats = readers.reader->columnStatistics(1);
  ASSERT_NE(stats, nullptr);
  auto* strStats =
      dynamic_cast<dwio::common::StringColumnStatistics*>(stats.get());
  ASSERT_NE(strStats, nullptr);
  ASSERT_TRUE(stats->getNumberOfValues().has_value());
  EXPECT_EQ(stats->getNumberOfValues().value(), 3);
  ASSERT_TRUE(strStats->getMinimum().has_value());
  EXPECT_EQ(strStats->getMinimum().value(), "apple");
  ASSERT_TRUE(strStats->getMaximum().has_value());
  EXPECT_EQ(strStats->getMaximum().value(), "cherry");
}

// Verifies columnStatistics returns nullptr for out-of-range indices and
// returns base ColumnStatistics for the root ROW column.
TEST_P(SelectiveNimbleReaderTest, columnStatisticsOutOfRange) {
  constexpr int kSize = 10;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  WriterOptions writerOptions;
  writerOptions.enableVectorizedStats = true;
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers =
      makeReaders(input, fileContent, scanSpec, this->stringDecoderZeroCopy());

  // Out-of-range index returns nullptr.
  EXPECT_EQ(readers.reader->columnStatistics(999), nullptr);

  // Column 0 is the root ROW -- should return base ColumnStatistics.
  auto rootStats = readers.reader->columnStatistics(0);
  ASSERT_NE(rootStats, nullptr);
  ASSERT_TRUE(rootStats->getNumberOfValues().has_value());
}

// Verifies columnStatistics returns nullptr when vectorized stats are absent.
TEST_P(SelectiveNimbleReaderTest, columnStatisticsNoStats) {
  constexpr int kSize = 10;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  WriterOptions writerOptions;
  writerOptions.enableVectorizedStats = false;
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers =
      makeReaders(input, fileContent, scanSpec, this->stringDecoderZeroCopy());

  EXPECT_EQ(readers.reader->columnStatistics(0), nullptr);
  EXPECT_EQ(readers.reader->columnStatistics(1), nullptr);
}

// Verifies columnStatistics correctly reports hasNull for nullable columns.
TEST_P(SelectiveNimbleReaderTest, columnStatisticsWithNulls) {
  constexpr int kSize = 100;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity, nullEvery(5)),
  });
  WriterOptions writerOptions;
  writerOptions.enableVectorizedStats = true;
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers =
      makeReaders(input, fileContent, scanSpec, this->stringDecoderZeroCopy());

  auto stats = readers.reader->columnStatistics(1);
  ASSERT_NE(stats, nullptr);
  ASSERT_TRUE(stats->hasNull().has_value());
  EXPECT_TRUE(stats->hasNull().value());
}

// Verifies columnStatistics for an all-null column: hasNull is true and
// min/max are absent.
TEST_P(SelectiveNimbleReaderTest, columnStatisticsAllNull) {
  constexpr int kSize = 50;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(
          kSize, folly::identity, [](auto) { return true; }),
  });
  WriterOptions writerOptions;
  writerOptions.enableVectorizedStats = true;
  auto fileContent = test::createNimbleFile(*rootPool(), input, writerOptions);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers =
      makeReaders(input, fileContent, scanSpec, this->stringDecoderZeroCopy());

  auto stats = readers.reader->columnStatistics(1);
  ASSERT_NE(stats, nullptr);
  ASSERT_TRUE(stats->hasNull().has_value());
  EXPECT_TRUE(stats->hasNull().value());
  auto* intStats =
      dynamic_cast<dwio::common::IntegerColumnStatistics*>(stats.get());
  ASSERT_NE(intStats, nullptr);
  EXPECT_EQ(intStats->getMinimum(), std::nullopt);
  EXPECT_EQ(intStats->getMaximum(), std::nullopt);
}

TEST_P(SelectiveNimbleReaderTest, extractionTransformMapKeys) {
  // Write a MAP(VARCHAR, BIGINT) column, read with a MapKeys extraction
  // transform applied via ScanSpec.
  auto mapVector = makeMapVector<StringView, int64_t>(
      {{{"a", 1}, {"b", 2}}, {{"c", 3}}, {{"d", 4}, {"e", 5}, {"f", 6}}});
  auto input = makeRowVector({"col"}, {mapVector});
  auto file = test::createNimbleFile(*rootPool(), input);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());

  scanSpec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kKeys);

  auto readers =
      makeReaders(input, file, scanSpec, GetParam().stringDecoderZeroCopy);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, result), 3);
  auto* row = result->as<RowVector>();
  auto* resultArray = row->childAt(0)->loadedVector()->as<ArrayVector>();
  ASSERT_EQ(resultArray->size(), 3);
  ASSERT_EQ(resultArray->sizeAt(0), 2);
  ASSERT_EQ(resultArray->sizeAt(1), 1);
  ASSERT_EQ(resultArray->sizeAt(2), 3);
}

TEST_P(SelectiveNimbleReaderTest, extractionTransformSize) {
  // Write a MAP(VARCHAR, BIGINT) column, read with a Size extraction.
  auto mapVector = makeMapVector<StringView, int64_t>(
      {{{"a", 1}, {"b", 2}, {"c", 3}}, {{"d", 4}}});
  auto input = makeRowVector({"col"}, {mapVector});
  auto file = test::createNimbleFile(*rootPool(), input);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());

  scanSpec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kSize);

  auto readers =
      makeReaders(input, file, scanSpec, GetParam().stringDecoderZeroCopy);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, result), 2);
  auto* row = result->as<RowVector>();
  auto* sizes = row->childAt(0)->loadedVector()->as<FlatVector<int64_t>>();
  ASSERT_EQ(sizes->size(), 2);
  ASSERT_EQ(sizes->valueAt(0), 3);
  ASSERT_EQ(sizes->valueAt(1), 1);
}

TEST_P(SelectiveNimbleReaderTest, extractionTransformMapValuesStructField) {
  // Write MAP(VARCHAR, ROW(x: INT, y: INT)), apply
  // [MapValues, AE, StructField("x")] -> ARRAY(INT).
  // The reader handles this natively via kValues on the map and kField on
  // the values struct, so no post-read transform is needed.
  auto keys = makeFlatVector<StringView>({"a", "b", "c"});
  auto structValues = makeRowVector(
      {"x", "y"},
      {makeFlatVector<int32_t>({10, 20, 30}),
       makeFlatVector<int32_t>({100, 200, 300})});
  auto mapVector = makeMapVector({0, 2}, keys, structValues);
  auto input = makeRowVector({"col"}, {mapVector});
  auto file = test::createNimbleFile(*rootPool(), input);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());

  // Configure ScanSpec directly: kValues on the map, then kField on the
  // values struct (extracting field "x" at index 0).  Mark "y" constant
  // null so it is not read.
  auto* colSpec = scanSpec->childByName("col");
  colSpec->setExtractionType(common::ScanSpec::ExtractionType::kValues);
  auto* valuesSpec =
      colSpec->childByName(common::ScanSpec::kMapValuesFieldName);
  valuesSpec->setExtractionType(common::ScanSpec::ExtractionType::kField);
  valuesSpec->setExtractionFieldIndex(0);
  valuesSpec->childByName("y")->setConstantValue(
      BaseVector::createNullConstant(INTEGER(), 1, pool()));

  auto readers =
      makeReaders(input, file, scanSpec, GetParam().stringDecoderZeroCopy);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, result), 2);
  auto* row = result->as<RowVector>();
  auto* resultArray = row->childAt(0)->loadedVector()->as<ArrayVector>();
  ASSERT_EQ(resultArray->size(), 2);
  ASSERT_EQ(resultArray->sizeAt(0), 2);
  ASSERT_EQ(resultArray->sizeAt(1), 1);
  auto* elements = resultArray->elements()->as<FlatVector<int32_t>>();
  ASSERT_EQ(elements->valueAt(0), 10);
  ASSERT_EQ(elements->valueAt(1), 20);
  ASSERT_EQ(elements->valueAt(2), 30);
}

TEST_P(SelectiveNimbleReaderTest, extractionSizeResultVectorReuse) {
  // Verify the FlatVector<int64_t> result is reused across batches.
  constexpr int kNumRows = 200;
  std::vector<std::string> keyStrs(kNumRows * 2);
  for (int i = 0; i < kNumRows * 2; ++i) {
    keyStrs[i] = std::to_string(i);
  }
  auto keys = makeFlatVector<StringView>(
      kNumRows * 2, [&](auto i) { return StringView(keyStrs[i]); });
  auto values = makeFlatVector<int64_t>(kNumRows * 2, folly::identity);
  std::vector<vector_size_t> offsets(kNumRows);
  for (int i = 0; i < kNumRows; ++i) {
    offsets[i] = i * 2;
  }
  auto mapVector = makeMapVector(offsets, keys, values);
  auto input = makeRowVector({"col"}, {mapVector});
  auto file = test::createNimbleFile(*rootPool(), input);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kSize);

  auto readers =
      makeReaders(input, file, scanSpec, GetParam().stringDecoderZeroCopy);
  auto result = BaseVector::create(input->type(), 0, pool());

  // Read first batch.
  ASSERT_GT(readers.rowReader->next(50, result), 0);
  auto* row = result->as<RowVector>();
  auto* child = row->childAt(0)->loadedVector();
  ASSERT_TRUE(child->type()->isBigint());
  auto* firstBatchPtr = child;

  // Read second batch — FlatVector should be the same object.
  ASSERT_GT(readers.rowReader->next(50, result), 0);
  row = result->as<RowVector>();
  child = row->childAt(0)->loadedVector();
  ASSERT_EQ(child, firstBatchPtr)
      << "FlatVector result should be reused across batches for Size extraction";

  auto* sizes = child->as<FlatVector<int64_t>>();
  for (int i = 0; i < sizes->size(); ++i) {
    ASSERT_EQ(sizes->valueAt(i), 2);
  }
}

TEST_P(SelectiveNimbleReaderTest, extractionTransformMapValues) {
  // Write a MAP(VARCHAR, BIGINT) column, read with a MapValues extraction
  // transform applied via ScanSpec.
  auto mapVector = makeMapVector<StringView, int64_t>(
      {{{"a", 10}}, {{"b", 20}, {"c", 30}}, {{"d", 40}}});
  auto input = makeRowVector({"col"}, {mapVector});
  auto file = test::createNimbleFile(*rootPool(), input);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());

  scanSpec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kValues);

  auto readers =
      makeReaders(input, file, scanSpec, GetParam().stringDecoderZeroCopy);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, result), 3);
  auto* row = result->as<RowVector>();
  auto* resultArray = row->childAt(0)->loadedVector()->as<ArrayVector>();
  ASSERT_EQ(resultArray->size(), 3);
  ASSERT_EQ(resultArray->sizeAt(0), 1);
  ASSERT_EQ(resultArray->sizeAt(1), 2);
  ASSERT_EQ(resultArray->sizeAt(2), 1);
  auto* elements = resultArray->elements()->as<FlatVector<int64_t>>();
  ASSERT_EQ(elements->valueAt(0), 10);
  ASSERT_EQ(elements->valueAt(1), 20);
  ASSERT_EQ(elements->valueAt(2), 30);
  ASSERT_EQ(elements->valueAt(3), 40);
}

TEST_P(SelectiveNimbleReaderTest, extractionTransformMapKeyFilter) {
  // Write a MAP(VARCHAR, BIGINT) column, apply MapKeyFilter extraction
  // to keep only selected keys.
  auto mapVector = makeMapVector<StringView, int64_t>(
      {{{"a", 1}, {"b", 2}, {"c", 3}}, {{"a", 10}, {"d", 40}}});
  auto input = makeRowVector({"col"}, {mapVector});
  auto file = test::createNimbleFile(*rootPool(), input);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());

  // MapKeyFilter is implemented as an IN filter on the map keys ScanSpec
  // (type-preserving — MAP stays MAP).
  auto* colSpec = scanSpec->childByName("col");
  auto* keysSpec = colSpec->childByName(common::ScanSpec::kMapKeysFieldName);
  keysSpec->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"a", "b"}, /*nullAllowed=*/false));

  auto readers =
      makeReaders(input, file, scanSpec, GetParam().stringDecoderZeroCopy);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, result), 2);
  auto* row = result->as<RowVector>();
  auto* filteredMap = row->childAt(0)->loadedVector()->as<MapVector>();
  ASSERT_EQ(filteredMap->size(), 2);
  // Row 0: {"a":1, "b":2} kept, "c" filtered out.
  ASSERT_EQ(filteredMap->sizeAt(0), 2);
  // Row 1: {"a":10} kept, "d" filtered out.
  ASSERT_EQ(filteredMap->sizeAt(1), 1);
}

TEST_P(SelectiveNimbleReaderTest, extractionTransformStructField) {
  // Write a ROW(x: INT, y: VARCHAR) column, extract just field "x".
  auto structVector = makeRowVector(
      {"x", "y"},
      {makeFlatVector<int32_t>({10, 20, 30}),
       makeFlatVector<StringView>({"aa", "bb", "cc"})});
  auto input = makeRowVector({"col"}, {structVector});
  auto file = test::createNimbleFile(*rootPool(), input);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());

  // kField on struct: extract field index 0 ("x"); mark "y" constant null.
  auto* colSpec = scanSpec->childByName("col");
  colSpec->setExtractionType(common::ScanSpec::ExtractionType::kField);
  colSpec->setExtractionFieldIndex(0);
  colSpec->childByName("y")->setConstantValue(
      BaseVector::createNullConstant(VARCHAR(), 1, pool()));

  auto readers =
      makeReaders(input, file, scanSpec, GetParam().stringDecoderZeroCopy);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, result), 3);
  auto* row = result->as<RowVector>();
  auto* xField = row->childAt(0)->loadedVector()->as<FlatVector<int32_t>>();
  ASSERT_EQ(xField->size(), 3);
  ASSERT_EQ(xField->valueAt(0), 10);
  ASSERT_EQ(xField->valueAt(1), 20);
  ASSERT_EQ(xField->valueAt(2), 30);
}

TEST_P(SelectiveNimbleReaderTest, extractionTransformArraySize) {
  // Write an ARRAY(BIGINT) column, read with Size extraction.
  auto arrayVector = makeArrayVector<int64_t>({{1, 2, 3}, {4}, {5, 6}});
  auto input = makeRowVector({"col"}, {arrayVector});
  auto file = test::createNimbleFile(*rootPool(), input);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());

  scanSpec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kSize);

  auto readers =
      makeReaders(input, file, scanSpec, GetParam().stringDecoderZeroCopy);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, result), 3);
  auto* row = result->as<RowVector>();
  auto* sizes = row->childAt(0)->loadedVector()->as<FlatVector<int64_t>>();
  ASSERT_EQ(sizes->size(), 3);
  ASSERT_EQ(sizes->valueAt(0), 3);
  ASSERT_EQ(sizes->valueAt(1), 1);
  ASSERT_EQ(sizes->valueAt(2), 2);
}

TEST_P(SelectiveNimbleReaderTest, extractionDeduplicatedArraySize) {
  // Write an ARRAY(BIGINT) column using the deduplicated (ArrayWithOffsets)
  // encoding path, then verify Size extraction works correctly.
  auto arrayVector = makeArrayVector<int64_t>({{1, 2}, {3}, {4, 5, 6}, {1, 2}});
  auto input = makeRowVector({"col"}, {arrayVector});
  auto file = test::createNimbleFile(
      *rootPool(), input, {.dictionaryArrayColumns = {"col"}});

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kSize);

  auto readers =
      makeReaders(input, file, scanSpec, GetParam().stringDecoderZeroCopy);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, result), 4);
  auto* row = result->as<RowVector>();
  auto* sizes = row->childAt(0)->loadedVector()->as<FlatVector<int64_t>>();
  ASSERT_EQ(sizes->size(), 4);
  ASSERT_EQ(sizes->valueAt(0), 2);
  ASSERT_EQ(sizes->valueAt(1), 1);
  ASSERT_EQ(sizes->valueAt(2), 3);
  ASSERT_EQ(sizes->valueAt(3), 2);
}

TEST_P(SelectiveNimbleReaderTest, extractionDeduplicatedMapSize) {
  // Write a MAP(BIGINT, BIGINT) column using the deduplicated
  // (SlidingWindowMap) encoding path, then verify Size extraction works.
  auto keys = makeFlatVector<int64_t>({1, 2, 3, 4});
  auto values = makeFlatVector<int64_t>({10, 20, 30, 40});
  auto mapVector = makeMapVector({0, 3}, keys, values);
  auto input = makeRowVector({"col"}, {mapVector});
  auto file = test::createNimbleFile(
      *rootPool(), input, {.deduplicatedMapColumns = {"col"}});

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kSize);

  auto readers =
      makeReaders(input, file, scanSpec, GetParam().stringDecoderZeroCopy);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(10, result), 2);
  auto* row = result->as<RowVector>();
  auto* sizes = row->childAt(0)->loadedVector()->as<FlatVector<int64_t>>();
  ASSERT_EQ(sizes->size(), 2);
  ASSERT_EQ(sizes->valueAt(0), 3);
  ASSERT_EQ(sizes->valueAt(1), 1);
}

TEST_P(SelectiveNimbleReaderTest, readGapTracking) {
  const bool stringDecoderZeroCopy = this->stringDecoderZeroCopy();

  // Schema with many columns — projecting a sparse subset creates gaps.
  auto input = makeRowVector(
      {"a", "b", "c", "d", "e"},
      {makeFlatVector<int32_t>(500, [](auto i) { return i * 10; }),
       makeFlatVector<int32_t>(500, [](auto i) { return i * 20; }),
       makeFlatVector<int32_t>(500, [](auto i) { return i * 30; }),
       makeFlatVector<int32_t>(500, [](auto i) { return i * 40; }),
       makeFlatVector<int32_t>(500, [](auto i) { return i * 50; })});
  auto file = test::createNimbleFile(*rootPool(), input);

  struct TestCase {
    std::vector<std::string> projectedColumns;
    bool expectGaps;
    std::string debugString() const {
      return fmt::format(
          "columns=[{}], expectGaps={}",
          folly::join(",", projectedColumns),
          expectGaps);
    }
  };

  std::vector<TestCase> testCases = {
      {{"a", "e"}, true},
      {{"a", "b", "c", "d", "e"}, false},
  };

  const auto& rowType = asRowType(input->type());
  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    dataIoStats_ = std::make_shared<io::IoStatistics>();
    metadataIoStats_ = std::make_shared<io::IoStatistics>();
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    for (const auto& col : testCase.projectedColumns) {
      scanSpec->addField(col, rowType->getChildIdx(col));
    }
    auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);

    VectorPtr result = BaseVector::create(rowType, 0, pool());
    while (readers.rowReader->next(1'000, result)) {
    }

    if (testCase.expectGaps) {
      EXPECT_GT(dataIoStats_->readGap().count(), 0);
      EXPECT_GT(dataIoStats_->readGap().sum(), 0);
      EXPECT_GT(dataIoStats_->readGap().min(), 0);
    } else {
      EXPECT_EQ(dataIoStats_->readGap().count(), 0);
    }
  }
}

INSTANTIATE_TEST_CASE_P(
    SelectiveNimbleReaderTestSuite,
    SelectiveNimbleReaderTest,
    ::testing::ValuesIn(SelectiveNimbleReaderTest::getTestParams()),
    [](const ::testing::TestParamInfo<TestParam>& info) {
      return info.param.debugString();
    });

class SmallFilePreloadTest : public ::testing::Test,
                             public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    if (!memory::MemoryManager::testInstance()) {
      memory::MemoryManager::testingSetInstance(
          velox::memory::MemoryManager::Options{});
    }
    registerSelectiveNimbleReaderFactory();
  }

  static void TearDownTestCase() {
    unregisterSelectiveNimbleReaderFactory();
  }
};

TEST_F(SmallFilePreloadTest, preloadReducesPreadCalls) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>(500, folly::identity),
      makeFlatVector<int64_t>(500, [](auto i) { return i * 10; }),
  });
  auto nimbleFile = test::createNimbleFile(*rootPool_, input);
  ASSERT_LE(nimbleFile.size(), 8 << 20);

  auto factory =
      dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);

  auto preadsWithThreshold = [&](uint64_t preloadThreshold) {
    auto countingFile =
        std::make_shared<velox::tests::utils::CountingReadFile>(nimbleFile);
    auto bufferedInput =
        std::make_unique<dwio::common::BufferedInput>(countingFile, *pool());
    dwio::common::ReaderOptions options(pool());
    auto ioStats = std::make_shared<io::IoStatistics>();
    options.setDataIoStats(ioStats);
    options.setMetadataIoStats(ioStats);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    options.setScanSpec(scanSpec);
    options.setFilePreloadThreshold(preloadThreshold);

    auto reader = factory->createReader(std::move(bufferedInput), options);
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    auto rowReader = reader->createRowReader(rowOptions);
    VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
    EXPECT_EQ(rowReader->next(1'000, result), 500);
    velox::test::assertEqualVectors(input, result);
    return countingFile->numReads();
  };

  EXPECT_EQ(preadsWithThreshold(nimbleFile.size() + 1), 1);
  EXPECT_GT(preadsWithThreshold(0), 1);
}

TEST_F(SmallFilePreloadTest, preloadAttributesSingleReadToData) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>(500, folly::identity),
      makeFlatVector<int64_t>(500, [](auto i) { return i * 10; }),
  });
  auto nimbleFile = test::createNimbleFile(*rootPool_, input);
  ASSERT_LE(nimbleFile.size(), 8 << 20);

  auto factory =
      dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);

  auto dataStats = std::make_shared<io::IoStatistics>();
  auto metadataStats = std::make_shared<io::IoStatistics>();

  auto countingFile =
      std::make_shared<velox::tests::utils::CountingReadFile>(nimbleFile);
  auto bufferedInput =
      std::make_unique<dwio::common::BufferedInput>(countingFile, *pool());
  dwio::common::ReaderOptions options(pool());
  options.setDataIoStats(dataStats);
  options.setMetadataIoStats(metadataStats);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  options.setScanSpec(scanSpec);
  options.setFilePreloadThreshold(nimbleFile.size() + 1);

  auto reader = factory->createReader(std::move(bufferedInput), options);
  dwio::common::RowReaderOptions rowOptions;
  rowOptions.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowOptions);
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  ASSERT_EQ(rowReader->next(1'000, result), 500);
  velox::test::assertEqualVectors(input, result);

  EXPECT_EQ(countingFile->numReads(), 1);
  EXPECT_EQ(dataStats->rawBytesRead(), nimbleFile.size());
}

// Preload must skip when a cache is present, else it bypasses the data cache.
TEST_F(SmallFilePreloadTest, cachePresentSkipsPreload) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>(500, folly::identity),
      makeFlatVector<int64_t>(500, [](auto i) { return i * 10; }),
  });
  auto nimbleFile = test::createNimbleFile(*rootPool_, input);
  ASSERT_LE(nimbleFile.size(), 8 << 20);

  auto allocator = std::make_shared<memory::MallocAllocator>(
      memory::MemoryAllocator::Options{
          .capacity = 512 << 20, .reservationByteLimit = 0});
  auto cache = cache::AsyncDataCache::create(allocator.get());
  auto scanTracker =
      std::make_shared<cache::ScanTracker>("preloadGate", nullptr, 256 << 10);
  folly::CPUThreadPoolExecutor ioExecutor(1);
  auto ioStats = std::make_shared<io::IoStatistics>();

  auto countingFile =
      std::make_shared<velox::tests::utils::CountingReadFile>(nimbleFile);

  auto factory =
      dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
  {
    auto& ids = fileIds();
    io::ReaderOptions ioReaderOpts(pool());
    ioReaderOpts.setDataIoStats(ioStats);
    auto cachedInput = std::make_unique<dwio::common::CachedBufferedInput>(
        countingFile,
        dwio::common::MetricsLog::voidLog(),
        StringIdLease(ids, "preloadGateFile"),
        cache.get(),
        scanTracker,
        StringIdLease(ids, "preloadGateGroup"),
        ioStats,
        nullptr,
        &ioExecutor,
        ioReaderOpts);

    dwio::common::ReaderOptions options(pool());
    options.setDataIoStats(ioStats);
    options.setMetadataIoStats(ioStats);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    options.setScanSpec(scanSpec);
    options.setFilePreloadThreshold(nimbleFile.size() + 1);
    options.setCache(cache.get());

    auto reader = factory->createReader(std::move(cachedInput), options);
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    auto rowReader = reader->createRowReader(rowOptions);
    VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
    ASSERT_EQ(rowReader->next(1'000, result), 500);
    velox::test::assertEqualVectors(input, result);
  }
  cache->shutdown();

  EXPECT_GT(countingFile->numReads(), 1);
}

} // namespace
} // namespace facebook::nimble
