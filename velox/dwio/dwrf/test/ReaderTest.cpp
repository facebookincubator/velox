/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "folly/Random.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/executors/IOThreadPoolExecutor.h"
#include "folly/lang/Assume.h"
#include "folly/synchronization/Baton.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/ExtractionUtils.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/dwio/common/ExecutorBarrier.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/dwio/dwrf/common/Common.h"
#include "velox/dwio/dwrf/common/DwrfRuntimeStats.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/dwrf/test/OrcTest.h"
#include "velox/dwio/dwrf/test/utils/E2EWriterTestUtil.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/type/fbhive/HiveTypeParser.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <fmt/core.h>
#include <array>
#include <future>
#include <memory>
#include <numeric>

#include "velox/common/io/IoStatistics.h"

namespace facebook::velox::dwrf {
namespace {

using namespace ::testing;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::type::fbhive;
using namespace facebook::velox::test;

const std::string& getStructFile() {
  static const std::string structFile_ = getExampleFilePath("struct.orc");
  return structFile_;
}

const std::string& getFMSmallFile() {
  static const std::string fmSmallFile_ = getExampleFilePath("fm_small.orc");
  return fmSmallFile_;
}

const std::string& getFMLargeFile() {
  static const std::string fmLargeFile_ = getExampleFilePath("fm_large.orc");
  return fmLargeFile_;
}

// RowType for fmSmallFile and fmLargeFile
const std::shared_ptr<const RowType>& getFlatmapSchema() {
  static const std::shared_ptr<const RowType> schema_ =
      std::dynamic_pointer_cast<const RowType>(HiveTypeParser().parse("struct<\
         id:int,\
     map1:map<int, array<float>>,\
     map2:map<string, map<smallint,bigint>>,\
     map3:map<int,int>,\
     map4:map<int,struct<field1:int,field2:float,field3:string>>,\
     memo:string>"));
  return schema_;
}

std::vector<common::Subfield> makeSubfields(
    const std::vector<std::string>& paths) {
  std::vector<common::Subfield> subfields;
  subfields.reserve(paths.size());
  for (auto& path : paths) {
    subfields.emplace_back(path);
  }
  return subfields;
}

folly::F14FastMap<std::string, std::vector<const common::Subfield*>>
groupSubfields(const std::vector<common::Subfield>& subfields) {
  folly::F14FastMap<std::string, std::vector<const common::Subfield*>> grouped;
  for (auto& subfield : subfields) {
    auto& name =
        static_cast<const common::Subfield::NestedField&>(*subfield.path()[0])
            .name();
    grouped[name].push_back(&subfield);
  }
  return grouped;
}

class TestReaderP
    : public testing::TestWithParam</* parallel decoding = */ bool>,
      public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    facebook::velox::common::testutil::TestValue::enable();
  }

  folly::Executor* executor() {
    if (GetParam() && !executor_) {
      std::make_shared<folly::CPUThreadPoolExecutor>(
          getDecodingParallelismFactor());
    }
    return executor_.get();
  }

  size_t getDecodingParallelismFactor() {
    return GetParam() ? 2 : 0;
  }

 private:
  std::unique_ptr<folly::Executor> executor_;

 protected:
  std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
};

class TestReader : public testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    facebook::velox::common::testutil::TestValue::enable();
  }

  std::vector<VectorPtr> createBatches(
      const std::vector<std::vector<int32_t>>& values) {
    std::vector<VectorPtr> batches;
    for (const auto& value : values) {
      auto vector = makeFlatVector<int32_t>(value);
      auto rowVector = makeRowVector({vector});
      batches.push_back(rowVector);
    }
    return batches;
  }

  std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
};

TEST_F(TestReader, testWriterVersions) {
  EXPECT_EQ("original", writerVersionToString(ORIGINAL));
  EXPECT_EQ("dwrf-4.9", writerVersionToString(DWRF_4_9));
  EXPECT_EQ("dwrf-5.0", writerVersionToString(DWRF_5_0));
  EXPECT_EQ("dwrf-6.0", writerVersionToString(DWRF_6_0));
  EXPECT_EQ(
      "future - 99", writerVersionToString(static_cast<WriterVersion>(99)));
}

TEST_F(TestReader, currentStripe) {
  dwio::common::ReaderOptions readerOpts{pool()};
  auto reader = DwrfReader::create(
      createFileBufferedInput(getFMSmallFile(), readerOpts.memoryPool()),
      readerOpts);
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*getFlatmapSchema()));
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(getFlatmapSchema(), 0, pool());
  ASSERT_GT(rowReader->next(1000, batch), 0);
  EXPECT_LT(rowReader->currentStripe(), reader->getNumberOfStripes());
}

// This relies on schema and data inside of our fm_small and fm_large orc files,
// and is not composeable with other schema/datas
void verifyFlatMapReading(
    DwrfRowReader* rowReader,
    const int32_t seeks[],
    const int32_t expectedBatchSize[],
    const int32_t numBatches) {
  auto expectNoNullKeys = [](const auto* keys) {
    for (vector_size_t i = 0; i < keys->size(); ++i) {
      EXPECT_FALSE(keys->isNullAt(i)) << "Null map key at index " << i;
    }
  };
  VectorPtr batch = BaseVector::create(
      rowReader->type(), 0, &rowReader->getReader().memoryPool());
  int32_t batchId = 0;
  do {
    // for every read, it seek to the specified row
    if (seeks[batchId] > 0) {
      rowReader->seekToRow(seeks[batchId]);
    }

    bool result = rowReader->next(1000, batch);
    if (!result) {
      break;
    }

    // verify current batch
    auto root = batch->as<RowVector>();
    EXPECT_EQ(root->childrenSize(), 6);
    // 4 stripes -> 4 batches
    EXPECT_EQ(root->size(), expectedBatchSize[batchId++]);

    // try to read first map as map<int, list<float>>
    auto map1 = root->childAt(1)->loadedVector()->as<MapVector>();
    auto map1KeyInt = map1->mapKeys()->as<SimpleVector<int32_t>>();
    auto map1ValueList = map1->mapValues();

    // print all the map vector based on offsets
    EXPECT_EQ(map1KeyInt->size(), map1ValueList->size());
    expectNoNullKeys(map1KeyInt);

    // try to verify map2 as map<string, map<smallint, bigint>>
    auto map2 = root->childAt(2)->loadedVector()->as<MapVector>();
    auto map2Key = map2->mapKeys();
    FlatVectorPtr<StringView> map2KeyString =
        std::dynamic_pointer_cast<FlatVector<StringView>>(map2Key);
    auto map2ValueMap = map2->mapValues();
    EXPECT_EQ(map2KeyString->size(), map2ValueMap->size());
    expectNoNullKeys(map2KeyString.get());

    // data - map2 always has string keys "key-1" and "key-nullable"
    // key-1 always has value map {1:1}
    // value of "key-nullable" is either null or map {1:1}
    for (int32_t i = 0; i < map2->size(); ++i) {
      int64_t start = map2->offsetAt(i);
      int64_t end = start + map2->sizeAt(i);

      // map2 has at least key-1 and key-nullable
      EXPECT_GE(end - start, 2);

      // go through all the keys
      int32_t found = 0;
      while (start < end) {
        std::string keyStr = map2KeyString->valueAt(start).str();
        start++;

        if (keyStr == "key-1" || keyStr == "key-nullable") {
          found++;
        }
      }

      // these two keys should always present
      EXPECT_EQ(found, 2);
    }

    // try to verify map3 as map<int, int>
    auto map3 = root->childAt(3)->loadedVector()->as<MapVector>();
    auto map3KeyInt = map3->mapKeys()->as<SimpleVector<int32_t>>();
    auto map3ValueInt = map3->mapValues()->as<SimpleVector<int32_t>>();

    EXPECT_EQ(map3KeyInt->size(), map3ValueInt->size());
    expectNoNullKeys(map3KeyInt);

    // try to verify map4 as
    // map<int,struct<field1:int,field2:float,field3:string>>
    auto map4 = root->childAt(4)->loadedVector()->as<MapVector>();
    auto map4KeyInt = map4->mapKeys()->as<SimpleVector<int32_t>>();
    auto map4ValueStruct = map4->mapValues();

    EXPECT_EQ(map4KeyInt->size(), map4ValueStruct->size());
    expectNoNullKeys(map4KeyInt);

    // data - map4 always has 9 keys [0-8]
    // each key maps the a internal struct with all fields the same value as key
    EXPECT_EQ(map4->size() * 9, map4KeyInt->size());
  } while (true);

  // number of batches should match
  EXPECT_EQ(batchId, numBatches);
}

// schema of flat map sample file
// struct {
//   id int,
//   map1 map<int, array<float>>,
//   map2 map<varchar, map<smallint, bigint>>,
//   map3 map<int, int>,
//   map4 map<int, struct<field1 int, field2 float, field3 varchar>>,
//   memo varchar
// }
void verifyFlatMapReading(
    memory::MemoryPool* pool,
    const std::string& file,
    const int32_t seeks[],
    const int32_t expectedBatchSize[],
    const int32_t numBatches,
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats,
    const std::vector<uint32_t>& expectedPrefetchRowSizes = {},
    const std::vector<bool>& shouldTryPrefetch = {}) {
  dwio::common::ReaderOptions readerOpts{pool};
  readerOpts.setDataIoStats(dataIoStats);
  readerOpts.setMetadataIoStats(metadataIoStats);

  /* If an extra sanity check is desired you can uncomment the 2 below lines and
   * re-run */
  // readerOpts.setFooterSpeculativeIoSize(257);
  // readerOpts.setFilePreloadThreshold(0);

  RowReaderOptions rowReaderOpts;
  rowReaderOpts.select(std::make_shared<ColumnSelector>(getFlatmapSchema()));
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*getFlatmapSchema()));
  auto reader = DwrfReader::create(
      createFileBufferedInput(file, readerOpts.memoryPool()), readerOpts);
  auto rowReaderOwner = reader->createRowReader(rowReaderOpts);
  auto rowReader = dynamic_cast<DwrfRowReader*>(rowReaderOwner.get());

  verifyFlatMapReading(rowReader, seeks, expectedBatchSize, numBatches);
}

/*
Verifies contents of dict_encoded_strings.orc
schema:
struct {
 int_column int,
 string_column string
 string_column_2 string
}
*/
void verifyCachedIndexStreamReads(
    DwrfRowReader* rowReader,
    uint32_t firstStripe,
    uint32_t pastLastStripe) {
  VectorPtr batch;

  if (firstStripe == 0 && pastLastStripe > 0) {
    // Stripe 1
    ASSERT_TRUE(rowReader->next(100, batch));
    auto root = batch->as<RowVector>();
    EXPECT_EQ(root->childrenSize(), 4);
    auto stringCol1 = root->childAt(1)->as<SimpleVector<StringView>>();
    auto stringCol2 = root->childAt(2)->as<SimpleVector<StringView>>();

    for (int i = 0; i < 50; i++) {
      ASSERT_EQ(stringCol1->valueAt(i), "baz");
      ASSERT_EQ(stringCol2->valueAt(i), "abcdefghijklmnop");
    }

    ASSERT_EQ(stringCol1->valueAt(50), "zax");
    ASSERT_EQ(stringCol2->valueAt(50), "unique");

    ASSERT_EQ(stringCol1->valueAt(51), "zax");
    ASSERT_EQ(stringCol2->valueAt(51), "different");

    ASSERT_EQ(stringCol1->valueAt(52), "zax");
    ASSERT_EQ(stringCol2->valueAt(52), "special");

    for (int i = 53; i < 100; i++) {
      ASSERT_EQ(stringCol1->valueAt(i), "baz");
      ASSERT_EQ(stringCol2->valueAt(i), "abcdefghijklmnop");
    }
  }

  if (firstStripe <= 1 && pastLastStripe > 1) {
    // // Stripe 2
    ASSERT_TRUE(rowReader->next(100, batch));
    auto root = batch->as<RowVector>();
    EXPECT_EQ(root->childrenSize(), 4);
    auto stringCol1 = root->childAt(1)->as<SimpleVector<StringView>>();
    auto stringCol2 = root->childAt(2)->as<SimpleVector<StringView>>();

    for (int i = 0; i < 50; i++) {
      ASSERT_EQ(stringCol1->valueAt(i), "ee");
      ASSERT_EQ(stringCol2->valueAt(i), "pomelo");
    }

    ASSERT_EQ(stringCol1->valueAt(50), "craz");
    ASSERT_EQ(stringCol2->valueAt(50), "unique");

    ASSERT_EQ(stringCol1->valueAt(51), "doop");
    ASSERT_EQ(stringCol2->valueAt(51), "different");

    ASSERT_EQ(stringCol1->valueAt(52), "hello");
    ASSERT_EQ(stringCol2->valueAt(52), "special");

    for (int i = 53; i < 100; i++) {
      ASSERT_EQ(stringCol1->valueAt(i), "baz");
      ASSERT_EQ(stringCol2->valueAt(i), "pomelo");
    }
  }

  if (firstStripe <= 2 && pastLastStripe > 2) {
    // Stripe 3
    ASSERT_TRUE(rowReader->next(100, batch));
    auto root = batch->as<RowVector>();
    ASSERT_EQ(root->size(), 3);
    EXPECT_EQ(root->childrenSize(), 4);
    auto stringCol1 = root->childAt(1)->as<SimpleVector<StringView>>();
    auto stringCol2 = root->childAt(2)->as<SimpleVector<StringView>>();

    ASSERT_EQ(stringCol1->valueAt(0), "craz");
    ASSERT_EQ(stringCol2->valueAt(0), "dog");

    ASSERT_EQ(stringCol1->valueAt(1), "doop");
    ASSERT_EQ(stringCol2->valueAt(1), "cat");

    ASSERT_EQ(stringCol1->valueAt(2), "hello");
    ASSERT_EQ(stringCol2->valueAt(2), "chicken");
  }
}

class TestFlatMapReader : public testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
};

TEST_F(TestFlatMapReader, testReadFlatMapEmptyMap) {
  const std::string emptyFile(getExampleFilePath("empty_flatmap.orc"));

  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  RowReaderOptions rowReaderOpts;
  std::shared_ptr<const RowType> emptyFileType =
      std::dynamic_pointer_cast<const RowType>(HiveTypeParser().parse("struct<\
         id:int,\
     mapcol:map<int,int>>"));
  rowReaderOpts.select(std::make_shared<ColumnSelector>(emptyFileType));
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*emptyFileType));
  auto reader = DwrfReader::create(
      createFileBufferedInput(emptyFile, readerOpts.memoryPool()), readerOpts);
  auto rowReaderOwner = reader->createRowReader(rowReaderOpts);
  auto rowReader = dynamic_cast<DwrfRowReader*>(rowReaderOwner.get());
  VectorPtr batch = BaseVector::create(emptyFileType, 0, pool());
  rowReader->next(1, batch);
  auto root = batch->as<RowVector>();

  auto map = root->childAt(1)->loadedVector()->as<MapVector>();
  auto mapKeyInt = map->mapKeys()->as<SimpleVector<int32_t>>();
  auto mapValueInt = map->mapValues()->as<SimpleVector<int32_t>>();

  EXPECT_EQ(0, mapKeyInt->size());
  EXPECT_EQ(0, mapValueInt->size());
  EXPECT_EQ(mapKeyInt->getNullCount().has_value(), false);
}

TEST_F(TestFlatMapReader, testStringKeyLifeCycle) {
  VectorPtr batch = BaseVector::create(getFlatmapSchema(), 0, pool());
  dwio::common::ReaderOptions readerOptions{pool()};
  readerOptions.setDataIoStats(dataIoStats_);
  readerOptions.setMetadataIoStats(metadataIoStats_);

  std::shared_ptr<MapVector> map2;
  FlatVectorPtr<StringView> map2KeyString;
  FlatVectorPtr<StringView> rowFieldString;
  {
    RowReaderOptions rowReaderOptions;
    rowReaderOptions.setScanSpec(makeAllFieldsScanSpec(*getFlatmapSchema()));

    auto reader = DwrfReader::create(
        createFileBufferedInput(getFMSmallFile(), readerOptions.memoryPool()),
        readerOptions);
    auto rowReader = reader->createRowReader(rowReaderOptions);
    rowReader->next(100, batch);

    // Load the fields used below while their column readers are still alive.
    auto root = batch->as<RowVector>();
    map2 = std::dynamic_pointer_cast<MapVector>(
        BaseVector::loadedVectorShared(root->childAt(2)));
    map2KeyString = std::dynamic_pointer_cast<FlatVector<StringView>>(
        BaseVector::loadedVectorShared(map2->mapKeys()));
    auto map4 = root->childAt(4)->loadedVector()->as<MapVector>();
    auto rowField =
        map4->mapValues()->wrappedVector()->as<RowVector>()->childAt(2);
    rowFieldString = std::dynamic_pointer_cast<FlatVector<StringView>>(
        BaseVector::loadedVectorShared(rowField));
  }

  // data - map2 always has string keys "key-1" and "key-nullable"
  // key-1 always has value map {1:1}
  // value of "key-nullable" is either null or map {1:1}
  for (int32_t i = 0; i < map2->size(); ++i) {
    int64_t start = map2->offsetAt(i);
    int64_t end = start + map2->sizeAt(i);

    // map2 has at least key-1 and key-nullable
    EXPECT_GE(end - start, 2);

    // go through all the keys
    int32_t found = 0;
    while (start < end) {
      auto keyStr = map2KeyString->valueAt(start++).str();
      if (keyStr == "key-1" || keyStr == "key-nullable") {
        found++;
      }
    }

    // these two keys should always be present
    EXPECT_EQ(found, 2);
  }

  // try to verify map4 as
  // map<int,struct<field1:int,field2:float,field3:string>>
  ASSERT_GT(rowFieldString->size(), 0);
  ASSERT_GE(rowFieldString->valueAt(0).str().size(), 0);
}

// Disabled because DwrfRowReader::skip does not update read offset for
// selective column reader.
TEST_F(TestFlatMapReader, DISABLED_testReadFlatMapSampleSmallSkips) {
  // batch size is set as 1000 in reading
  const std::array<int32_t, 4> seeks{100, 700, 0, 0};
  const std::array<int32_t, 3> expectedBatchSize{200, 200, 100};
  verifyFlatMapReading(
      pool(),
      getFMSmallFile(),
      seeks.data(),
      expectedBatchSize.data(),
      expectedBatchSize.size(),
      dataIoStats_,
      metadataIoStats_);
}

TEST_F(TestFlatMapReader, testReadFlatMapSampleSmall) {
  // batch size is set as 1000 in reading
  std::array<int32_t, 5> seeks;
  seeks.fill(0);
  const std::array<int32_t, 4> expectedBatchSize{300, 300, 300, 100};
  verifyFlatMapReading(
      pool(),
      getFMSmallFile(),
      seeks.data(),
      expectedBatchSize.data(),
      expectedBatchSize.size(),
      dataIoStats_,
      metadataIoStats_);
}

TEST_F(TestFlatMapReader, testReadFlatMapSampleLarge) {
  // batch size is set as 1000 in reading
  // 3000 per stripe
  std::array<int32_t, 11> seeks;
  seeks.fill(0);
  // batch size is set as 1000 in reading
  const std::array<int32_t, 10> expectedBatchSize{
      1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
  verifyFlatMapReading(
      pool(),
      getFMLargeFile(),
      seeks.data(),
      expectedBatchSize.data(),
      expectedBatchSize.size(),
      dataIoStats_,
      metadataIoStats_);
}

class TestFlatMapReaderFlatLayout
    : public TestWithParam<std::tuple<bool, size_t>>,
      public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
};

TEST_P(TestFlatMapReaderFlatLayout, testCompare) {
  dwio::common::ReaderOptions readerOptions{pool()};
  readerOptions.setDataIoStats(dataIoStats_);
  readerOptions.setMetadataIoStats(metadataIoStats_);
  auto reader = DwrfReader::create(
      createFileBufferedInput(getFMSmallFile(), readerOptions.memoryPool()),
      readerOptions);
  RowReaderOptions rowReaderOptions;
  auto param = GetParam();
  rowReaderOptions.setScanSpec(makeAllFieldsScanSpec(*getFlatmapSchema()));
  rowReaderOptions.setReturnFlatVector(false);
  auto rowReader = reader->createRowReader(rowReaderOptions);
  rowReaderOptions.setReturnFlatVector(true);
  auto rowReader2 = reader->createRowReader(rowReaderOptions);

  VectorPtr vector1 = BaseVector::create(reader->rowType(), 0, pool());
  auto size = std::get<1>(param);
  while (rowReader->next(size, vector1) > 0) {
    VectorPtr vector2 = BaseVector::create(reader->rowType(), 0, pool());
    rowReader2->next(size, vector2);
    ASSERT_EQ(vector1->size(), vector2->size());
    VectorPtr comp1 = vector1;
    VectorPtr comp2 = vector2;
    for (auto i = 0; i < vector1->size(); ++i) {
      ASSERT_TRUE(comp1->equalValueAt(comp2.get(), i, i)) << i;
    }
  }
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    FlatMapReaderFlatLayoutTests,
    TestFlatMapReaderFlatLayout,
    Combine(Bool(), Values(1, 100)));

TEST_F(TestReader, testReadFlatMapWithKeyFilters) {
  // batch size is set as 1000 in reading
  // file has schema: a int, b struct<a:int, b:float, c:string>, c float
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  RowReaderOptions rowReaderOpts;
  // set map key filter for map1 we only need key=1, and map2 only key-1
  auto cs = std::make_shared<ColumnSelector>(
      getFlatmapSchema(),
      std::vector<std::string>{"map1#[1]", "map2#[\"key-1\"]"});
  rowReaderOpts.select(cs);
  auto reader = DwrfReader::create(
      createFileBufferedInput(getFMSmallFile(), readerOpts.memoryPool()),
      readerOpts);
  auto scanSpec = makeAllFieldsScanSpec(*getFlatmapSchema());
  scanSpec->childByName("map1")
      ->childByName(common::ScanSpec::kMapKeysFieldName)
      ->setFilter(common::createBigintValues({1}, false));
  scanSpec->childByName("map2")
      ->childByName(common::ScanSpec::kMapKeysFieldName)
      ->setFilter(
          std::make_unique<common::BytesValues>(
              std::vector<std::string>{"key-1"}, false));
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(getFlatmapSchema(), 0, pool());

  do {
    bool result = rowReader->next(1000, batch);
    if (!result) {
      break;
    }

    // verify current batch
    auto root = batch->as<RowVector>();

    // verify map1
    {
      auto map1 = root->childAt(1)->loadedVector()->as<MapVector>();
      auto map1KeyInt = map1->mapKeys()->as<SimpleVector<int32_t>>();

      // every key value should be 1
      EXPECT_GT(map1KeyInt->size(), 0);
      for (int32_t i = 0; i < map1KeyInt->size(); ++i) {
        // every key should be just 1
        EXPECT_EQ(map1KeyInt->valueAt(i), 1);
      }
    }

    // verify map2
    {
      auto map2 = root->childAt(2)->loadedVector()->as<MapVector>();
      auto map2KeyString = map2->mapKeys()->as<SimpleVector<StringView>>();

      // every key value should be key-1
      EXPECT_GT(map2KeyString->size(), 0);
      for (int32_t i = 0; i < map2KeyString->size(); ++i) {
        // every key should be just 1
        EXPECT_EQ(map2KeyString->valueAt(i).str(), "key-1");
      }
    }
  } while (true);
}

TEST_F(TestReader, testReadFlatMapWithKeyRejectList) {
  // batch size is set as 1000 in reading
  // file has schema: a int, b struct<a:int, b:float, c:string>, c float
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  RowReaderOptions rowReaderOpts;
  auto cs = std::make_shared<ColumnSelector>(
      getFlatmapSchema(), std::vector<std::string>{"map1#[\"!2\",\"!3\"]"});
  rowReaderOpts.select(cs);
  auto reader = DwrfReader::create(
      createFileBufferedInput(getFMSmallFile(), readerOpts.memoryPool()),
      readerOpts);
  auto scanSpec = makeAllFieldsScanSpec(*getFlatmapSchema());
  scanSpec->childByName("map1")
      ->childByName(common::ScanSpec::kMapKeysFieldName)
      ->setFilter(std::make_unique<common::NegatedBigintRange>(2, 3, false));
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(getFlatmapSchema(), 0, pool());

  const std::unordered_set<int32_t> map1RejectList{2, 3};

  do {
    bool result = rowReader->next(1000, batch);
    if (!result) {
      break;
    }

    // verify current batch
    auto root = batch->as<RowVector>();

    // verify map1
    {
      auto map1 = root->childAt(1)->loadedVector()->as<MapVector>();
      auto map1KeyInt = map1->mapKeys()->as<SimpleVector<int32_t>>();

      // every key value should be 1
      EXPECT_GT(map1KeyInt->size(), 0);
      for (int32_t i = 0; i < map1KeyInt->size(); ++i) {
        // These keys should not exist
        EXPECT_TRUE(map1RejectList.count(map1KeyInt->valueAt(i)) == 0);
      }
    }
  } while (true);
}

// Disabled because keySelectionCallback is ignored in
// SelectiveStructColumnReader.
TEST_F(TestReader, DISABLED_testStatsCallbackFiredWithFiltering) {
  RowReaderOptions rowReaderOpts;
  // Apply feature projection
  auto cs = std::make_shared<ColumnSelector>(
      getFlatmapSchema(), std::vector<std::string>{"map2#[\"key-1\"]"});
  rowReaderOpts.select(cs);

  uint64_t totalKeyStreamsAggregate = 0;
  uint64_t selectedKeyStreamsAggregate = 0;

  rowReaderOpts.setKeySelectionCallback(
      [&totalKeyStreamsAggregate, &selectedKeyStreamsAggregate](
          facebook::velox::dwio::common::flatmap::FlatMapKeySelectionStats
              keySelectionStats) {
        totalKeyStreamsAggregate += keySelectionStats.totalKeys;
        selectedKeyStreamsAggregate += keySelectionStats.selectedKeys;
      });

  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);

  auto reader = DwrfReader::create(
      createFileBufferedInput(getFMSmallFile(), readerOpts.memoryPool()),
      readerOpts);
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addFieldRecursively("map2", *getFlatmapSchema()->childAt(2), 2);
  scanSpec->childByName("map2")
      ->childByName(common::ScanSpec::kMapKeysFieldName)
      ->setFilter(
          std::make_unique<common::BytesValues>(
              std::vector<std::string>{"key-1"}, false));
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(getFlatmapSchema(), 0, pool());

  do {
    bool result = rowReader->next(1000, batch);
    if (!result) {
      break;
    }
  } while (true);

  // Features were projected, so we expect selected keys > total keys
  EXPECT_EQ(totalKeyStreamsAggregate, 16);
  EXPECT_EQ(selectedKeyStreamsAggregate, 4);
}

TEST_F(TestReader, testBlockedIoCallbackFiredBlocking) {
  RowReaderOptions rowReaderOpts;
  std::optional<uint64_t> metricToIncrement;

  rowReaderOpts.setBlockedOnIoCallback(
      [&metricToIncrement](
          std::chrono::high_resolution_clock::duration blockedTime) {
        const auto blockedTimeMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(blockedTime)
                .count();
        if (metricToIncrement) {
          *metricToIncrement += blockedTimeMs;
        } else {
          metricToIncrement = blockedTimeMs;
        }
      });
  rowReaderOpts.setEagerFirstStripeLoad(false);

  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);

  auto reader = DwrfReader::create(
      createFileBufferedInput(getFMLargeFile(), readerOpts.memoryPool()),
      readerOpts);
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*getFlatmapSchema()));
  auto rowReader = reader->createRowReader(rowReaderOpts);
  // We didn't preload first stripe, so we expect metric to not be populated yet
  EXPECT_EQ(metricToIncrement, std::nullopt);
  VectorPtr batch = BaseVector::create(getFlatmapSchema(), 0, pool());

  auto lastMetric = metricToIncrement;
  do {
    bool result = rowReader->next(1000, batch);
    // Stripes in fm_large are 1000 rows, so we expect reading 1000 rows to load
    // a new stripe and increment the metric
    EXPECT_GE(metricToIncrement, lastMetric);
    lastMetric = metricToIncrement;
    if (!result) {
      break;
    }
  } while (true);

  // Reading stripes that were prefetched should not affect metric
  EXPECT_GE(metricToIncrement, 0);
}

TEST_F(TestReader, DISABLED_testBlockedIoCallbackFiredNonBlocking) {
  RowReaderOptions rowReaderOpts;
  std::optional<uint64_t> metricToIncrement;

  rowReaderOpts.setBlockedOnIoCallback(
      [&metricToIncrement](
          std::chrono::high_resolution_clock::duration blockedTime) {
        const auto blockedTimeMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(blockedTime)
                .count();
        if (metricToIncrement) {
          *metricToIncrement += blockedTimeMs;
        } else {
          metricToIncrement = blockedTimeMs;
        }
      });
  rowReaderOpts.setEagerFirstStripeLoad(false);

  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);

  auto reader = DwrfReader::create(
      createFileBufferedInput(getFMLargeFile(), readerOpts.memoryPool()),
      readerOpts);
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*getFlatmapSchema()));
  auto rowReader = reader->createRowReader(rowReaderOpts);
  EXPECT_EQ(metricToIncrement, std::nullopt);
  VectorPtr batch = BaseVector::create(getFlatmapSchema(), 0, pool());

  auto units = rowReader->prefetchUnits().value();

  // Blocking prefetch all stripes
  for (auto& unit : units) {
    unit.prefetch();
    // Since these are not blocking a read, metric should not be incremented.
    EXPECT_EQ(metricToIncrement, std::nullopt);
  }

  do {
    bool result = rowReader->next(1000, batch);
    if (!result) {
      break;
    }
  } while (true);

  // Reading prefetched stripes should not increment metric, but it should set
  // the metric from nullopt to 0, indicating we've hit the read path
  EXPECT_EQ(metricToIncrement, 0);
}

TEST_F(TestReader, DISABLED_testBlockedIoCallbackFiredWithFirstStripeLoad) {
  RowReaderOptions rowReaderOpts;
  std::optional<uint64_t> metricToIncrement;

  rowReaderOpts.setBlockedOnIoCallback(
      [&metricToIncrement](
          std::chrono::high_resolution_clock::duration blockedTime) {
        const auto blockedTimeMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(blockedTime)
                .count();
        if (metricToIncrement) {
          *metricToIncrement += blockedTimeMs;
        } else {
          metricToIncrement = blockedTimeMs;
        }
      });

  rowReaderOpts.setEagerFirstStripeLoad(true);

  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);

  auto reader = DwrfReader::create(
      createFileBufferedInput(getFMLargeFile(), readerOpts.memoryPool()),
      readerOpts);
  EXPECT_EQ(metricToIncrement, std::nullopt);
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*getFlatmapSchema()));
  auto rowReader = reader->createRowReader(rowReaderOpts);
  // Expect metric has now been populated, due to the initial blocking IO of
  // loadCurrentStripe()
  EXPECT_GE(metricToIncrement, 0);
  auto metricAfterFirstStripe = metricToIncrement;
  VectorPtr batch = BaseVector::create(getFlatmapSchema(), 0, pool());

  auto units = rowReader->prefetchUnits().value();
  EXPECT_EQ(metricToIncrement, metricAfterFirstStripe);

  // Blocking prefetch all stripes
  for (auto& unit : units) {
    unit.prefetch();
    // Since these are not blocking a read, metric should not be incremented
    EXPECT_EQ(metricToIncrement, metricAfterFirstStripe);
  }

  do {
    bool result = rowReader->next(1000, batch);
    if (!result) {
      break;
    }
  } while (true);

  // Reading prefetched stripes should not affect metric
  EXPECT_EQ(metricToIncrement, metricAfterFirstStripe);
}

TEST_F(TestReader, testEstimatedSize) {
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  {
    auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
    scanSpec->addFieldRecursively("map2", *getFlatmapSchema()->childAt(2), 2);
    readerOpts.setScanSpec(scanSpec);
    auto reader = DwrfReader::create(
        createFileBufferedInput(getFMSmallFile(), readerOpts.memoryPool()),
        readerOpts);
    RowReaderOptions rowReaderOpts;
    rowReaderOpts.setScanSpec(scanSpec);
    auto rowReader = reader->createRowReader(rowReaderOpts);
    ASSERT_EQ(rowReader->estimatedRowSize(), 67);
  }

  {
    auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
    scanSpec->addFieldRecursively("id", *getFlatmapSchema()->childAt(0), 0);
    readerOpts.setScanSpec(scanSpec);
    auto reader = DwrfReader::create(
        createFileBufferedInput(getFMSmallFile(), readerOpts.memoryPool()),
        readerOpts);
    RowReaderOptions rowReaderOpts;
    rowReaderOpts.setScanSpec(scanSpec);
    auto rowReader = reader->createRowReader(rowReaderOpts);
    ASSERT_EQ(rowReader->estimatedRowSize(), 4);
  }
}

TEST_F(TestReader, testSubfieldEstimatedSize) {
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  std::shared_ptr<const RowType> schema =
      std::dynamic_pointer_cast<const RowType>(HiveTypeParser().parse("struct<\
              a:int,\
              b:struct<\
                  a:int,\
                  b:float,\
                  c:string>,\
              c:float>"));

  std::shared_ptr<const RowType> outputType =
      std::dynamic_pointer_cast<const RowType>(HiveTypeParser().parse("struct<\
              a:int,\
              b:struct<\
                  a:int,\
                  b:float,\
                  c:string>>"));
  // estimation with subfield filtering
  auto subfields = makeSubfields({"a", "b.b"});
  folly::F14FastMap<std::string, std::vector<const common::Subfield*>>
      subfieldsByName = groupSubfields(subfields);
  auto scanSpec = velox::connector::hive::makeScanSpec(
      outputType, subfieldsByName, {}, {}, schema, {}, {}, {}, true, pool());
  readerOpts.setScanSpec(scanSpec);

  auto reader = DwrfReader::create(
      createFileBufferedInput(getStructFile(), readerOpts.memoryPool()),
      readerOpts);
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(scanSpec);

  auto rowReader = reader->createRowReader(rowReaderOpts);
  ASSERT_EQ(rowReader->estimatedRowSize(), 8);

  // estimation with full struct field selection
  dwio::common::ReaderOptions readerOpts2{pool()};
  readerOpts2.setDataIoStats(dataIoStats_);
  readerOpts2.setMetadataIoStats(metadataIoStats_);
  auto subfields2 = makeSubfields({"a", "b"});
  folly::F14FastMap<std::string, std::vector<const common::Subfield*>>
      subfields2ByName = groupSubfields(subfields2);
  auto scanSpec2 = velox::connector::hive::makeScanSpec(
      outputType, subfields2ByName, {}, {}, schema, {}, {}, {}, true, pool());
  readerOpts2.setScanSpec(scanSpec2);

  auto reader2 = DwrfReader::create(
      createFileBufferedInput(getStructFile(), readerOpts2.memoryPool()),
      readerOpts2);
  RowReaderOptions rowReaderOpts2;
  rowReaderOpts2.setScanSpec(scanSpec2);

  auto rowReader2 = reader2->createRowReader(rowReaderOpts2);
  ASSERT_EQ(rowReader2->estimatedRowSize(), 15);
}

// Disabled because keySelectionCallback is ignored in
// SelectiveStructColumnReader.
TEST_F(TestReader, DISABLED_testStatsCallbackFiredWithoutFiltering) {
  RowReaderOptions rowReaderOpts;
  // Don't apply feature projection here
  auto cs = std::make_shared<ColumnSelector>(
      getFlatmapSchema(), std::vector<std::string>{"map2"});
  rowReaderOpts.select(cs);

  uint64_t totalKeyStreamsAggregate = 0;
  uint64_t selectedKeyStreamsAggregate = 0;

  rowReaderOpts.setKeySelectionCallback(
      [&totalKeyStreamsAggregate, &selectedKeyStreamsAggregate](
          facebook::velox::dwio::common::flatmap::FlatMapKeySelectionStats
              keySelectionStats) {
        totalKeyStreamsAggregate += keySelectionStats.totalKeys;
        selectedKeyStreamsAggregate += keySelectionStats.selectedKeys;
      });

  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);

  auto reader = DwrfReader::create(
      createFileBufferedInput(getFMSmallFile(), readerOpts.memoryPool()),
      readerOpts);
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addFieldRecursively("map2", *getFlatmapSchema()->childAt(2), 2);
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(getFlatmapSchema(), 0, pool());

  do {
    bool result = rowReader->next(1000, batch);
    if (!result) {
      break;
    }
  } while (true);

  // No features were projected, so we expect selected keys == total keys
  EXPECT_EQ(totalKeyStreamsAggregate, 16);
  EXPECT_EQ(selectedKeyStreamsAggregate, 16);
}

namespace {

void verifyMapColumnEqual(
    MapVector* mapVector,
    RowVector* rowVector,
    int32_t key,
    vector_size_t childOffset) {
  const auto& key1ValueVector = rowVector->childAt(childOffset);
  const auto& keyVector = mapVector->mapKeys()->as<SimpleVector<int32_t>>();
  const auto& valueVector = mapVector->mapValues();
  for (uint64_t i = 0; i < mapVector->size(); ++i) {
    if (mapVector->isNullAt(i)) {
      EXPECT_TRUE(key1ValueVector->isNullAt(i));
    } else {
      bool found = false;
      for (uint64_t j = mapVector->offsetAt(i);
           j < mapVector->offsetAt(i) + mapVector->sizeAt(i);
           ++j) {
        if (keyVector->valueAt(j) == key) {
          EXPECT_EQ(valueVector->compare(key1ValueVector.get(), j, i), 0);
          found = true;
          break;
        }
      }

      if (!found) {
        EXPECT_TRUE(key1ValueVector->isNullAt(i));
      }
    }
  }
}

void verifyFlatmapStructEncoding(
    memory::MemoryPool* pool,
    const std::string& filename,
    const std::vector<int32_t>& keysAsFields,
    const std::vector<int32_t>& keysToSelect,
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats,
    size_t batchSize = 1000) {
  dwio::common::ReaderOptions readerOpts{pool};
  readerOpts.setDataIoStats(dataIoStats);
  readerOpts.setMetadataIoStats(metadataIoStats);
  auto reader = DwrfReader::create(
      createFileBufferedInput(filename, readerOpts.memoryPool()), readerOpts);

  const std::string projectedColumn = "map1";
  const vector_size_t projectedColumnIndex = 1;
  RowReaderOptions rowReaderOpts;
  auto mapSpec = std::make_shared<common::ScanSpec>("<root>");
  mapSpec->addFieldRecursively(
      projectedColumn,
      *getFlatmapSchema()->childAt(projectedColumnIndex),
      projectedColumnIndex);
  if (!keysToSelect.empty()) {
    mapSpec->childByName(projectedColumn)
        ->childByName(common::ScanSpec::kMapKeysFieldName)
        ->setFilter(
            common::createBigintValues(
                std::vector<int64_t>(keysToSelect.begin(), keysToSelect.end()),
                false));
  }
  rowReaderOpts.setScanSpec(mapSpec);
  auto mapEncodingReader = reader->createRowReader(rowReaderOpts);

  std::vector<std::string> keyNames;
  keyNames.reserve(keysAsFields.size());
  for (auto key : keysAsFields) {
    keyNames.push_back(folly::to<std::string>(key));
  }
  auto mapAsStructType =
      ROW(keyNames, std::vector<TypePtr>(keysAsFields.size(), ARRAY(REAL())));
  auto structTypes = getFlatmapSchema()->children();
  structTypes[projectedColumnIndex] = mapAsStructType;
  auto structType = ROW(getFlatmapSchema()->names(), structTypes);

  auto structSpec = std::make_shared<common::ScanSpec>("<root>");
  auto* map1Spec = structSpec->addField(projectedColumn, projectedColumnIndex);
  map1Spec->setFlatMapAsStruct(true);
  for (size_t i = 0; i < keysAsFields.size(); ++i) {
    auto* keySpec = map1Spec->addFieldRecursively(
        keyNames[i], *mapAsStructType->childAt(i), i);
    if (!keysToSelect.empty() &&
        std::find(keysToSelect.begin(), keysToSelect.end(), keysAsFields[i]) ==
            keysToSelect.end()) {
      keySpec->setConstantValue(
          BaseVector::createNullConstant(mapAsStructType->childAt(i), 1, pool));
    }
  }
  RowReaderOptions structReaderOpts;
  structReaderOpts.setScanSpec(structSpec);
  auto structEncodingReader = reader->createRowReader(structReaderOpts);

  const auto compare = [&]() {
    VectorPtr batchMap = BaseVector::create(getFlatmapSchema(), 0, pool);
    VectorPtr batchStruct = BaseVector::create(structType, 0, pool);

    do {
      bool resultMap = mapEncodingReader->next(batchSize, batchMap);
      bool resultStruct = structEncodingReader->next(batchSize, batchStruct);

      EXPECT_EQ(resultMap, resultStruct);
      if (!resultMap) {
        break;
      }

      // verify current batch
      auto rowMapEncoding = batchMap->as<RowVector>();
      auto rowStructEncoding = batchStruct->as<RowVector>();

      EXPECT_EQ(rowMapEncoding->size(), rowStructEncoding->size());

      for (size_t i = 0; i < keysAsFields.size(); ++i) {
        verifyMapColumnEqual(
            rowMapEncoding->childAt(projectedColumnIndex)
                ->loadedVector()
                ->as<MapVector>(),
            rowStructEncoding->childAt(projectedColumnIndex)
                ->loadedVector()
                ->as<RowVector>(),
            keysAsFields[i],
            i);
      }
    } while (true);
  };
  compare();
}
} // namespace

TEST_F(TestReader, testFlatmapAsStructSmall) {
  verifyFlatmapStructEncoding(
      pool(),
      getFMSmallFile(),
      {1, 2, 3, 4, 5, -99999999 /* does not exist */},
      {} /* no key filtering */,
      dataIoStats_,
      metadataIoStats_);
}

TEST_F(TestReader, testFlatmapAsStructSmallEmptyInmap) {
  verifyFlatmapStructEncoding(
      pool(),
      getFMSmallFile(),
      {1, 2, 3, 4, 5, -99999999 /* does not exist */},
      {} /* no key filtering */,
      dataIoStats_,
      metadataIoStats_,
      2);
}

TEST_F(TestReader, testFlatmapAsStructLarge) {
  verifyFlatmapStructEncoding(
      pool(),
      getFMSmallFile(),
      {1, 2, 3, 4, 5, -99999999 /* does not exist */},
      {} /* no key filtering */,
      dataIoStats_,
      metadataIoStats_);
}

TEST_F(TestReader, testFlatmapAsStructWithKeyProjection) {
  verifyFlatmapStructEncoding(
      pool(),
      getFMSmallFile(),
      {1, 2, 3, 4, 5, -99999999 /* does not exist */},
      {3, 5} /* select only these to read */,
      dataIoStats_,
      metadataIoStats_);
}

TEST_F(TestReader, testFlatmapAsStructRequiringKeyList) {
  const std::unordered_map<uint32_t, std::vector<std::string>> emptyKeys = {
      {0, {}}};
  RowReaderOptions rowReaderOpts;
  EXPECT_THROW(
      rowReaderOpts.setFlatmapNodeIdsAsStruct(emptyKeys), VeloxException);
}

// TODO: replace with mock
// Disabled because DwrfReader set `isRoot` to true for SelectiveDwrfReader,
// which cause the isChildMissing() returns false and failed when calling
// fileType_->childByName().
TEST_F(TestReader, DISABLED_testMismatchSchemaMoreFields) {
  // file has schema: a int, b struct<a:int, b:float, c:string>, c float
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  RowReaderOptions rowReaderOpts;
  std::shared_ptr<const RowType> requestedType =
      std::dynamic_pointer_cast<const RowType>(HiveTypeParser().parse(
          "struct<a:int,b:struct<a:int,b:float,c:string>,c:float,d:string>"));
  auto reader = DwrfReader::create(
      createFileBufferedInput(getStructFile(), readerOpts.memoryPool()),
      readerOpts);
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addFieldRecursively("b", *requestedType->childAt(1), 1);
  scanSpec->addFieldRecursively("c", *requestedType->childAt(2), 2);
  scanSpec->addFieldRecursively("d", *requestedType->childAt(3), 3);
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(requestedType, 0, pool());
  // Keep the input shared so the selective reader creates a fresh result.
  VectorPtr holder = batch;
  rowReader->next(1, batch);

  {
    auto root = std::dynamic_pointer_cast<RowVector>(batch);
    EXPECT_EQ(4, root->childrenSize());
    EXPECT_EQ(1, root->size());
    // Column 3 should be filled with NULLs
    EXPECT_LT(0, root->childAt(3)->getNullCount().value_or(0));
    EXPECT_TRUE(root->childAt(3)->isNullAt(0));
    // Column 0 should be null since it's not selected
    EXPECT_FALSE(root->childAt(0));
  }
}

TEST_F(TestReader, testMismatchSchemaFewerFields) {
  // file has schema: a int, b struct<a:int, b:float, c:string>, c float
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  RowReaderOptions rowReaderOpts;
  std::shared_ptr<const RowType> requestedType =
      std::dynamic_pointer_cast<const RowType>(HiveTypeParser().parse(
          "struct<a:int,b:struct<a:int,b:float,c:string>>"));
  auto reader = DwrfReader::create(
      createFileBufferedInput(getStructFile(), readerOpts.memoryPool()),
      readerOpts);
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addFieldRecursively("b", *requestedType->childAt(1), 1);
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(requestedType, 0, pool());
  // Keep the input shared so the selective reader creates a fresh result.
  VectorPtr holder = batch;
  rowReader->next(1, batch);

  {
    auto root = std::dynamic_pointer_cast<RowVector>(batch);
    EXPECT_EQ(2, root->childrenSize());
    EXPECT_EQ(1, root->size());

    // Column 0 should be null since it's not selected
    EXPECT_FALSE(root->childAt(0));
  }
}

TEST_F(TestReader, testMismatchSchemaNestedMoreFields) {
  // file has schema: a int, b struct<a:int, b:float>, c float
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  RowReaderOptions rowReaderOpts;
  std::shared_ptr<const RowType> requestedType =
      std::dynamic_pointer_cast<const RowType>(HiveTypeParser().parse(
          "struct<a:int,b:struct<a:int,b:float,c:string,d:binary>,c:float>"));
  LOG(INFO) << requestedType->toString();
  auto reader = DwrfReader::create(
      createFileBufferedInput(getStructFile(), readerOpts.memoryPool()),
      readerOpts);
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  auto* bSpec = scanSpec->addField("b", 1);
  const auto& bType = requestedType->childAt(1)->asRow();
  bSpec->addFieldRecursively("b", *bType.childAt(1), 1);
  bSpec->addFieldRecursively("c", *bType.childAt(2), 2);
  bSpec->addFieldRecursively("d", *bType.childAt(3), 3);
  scanSpec->addFieldRecursively("c", *requestedType->childAt(2), 2);
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(requestedType, 0, pool());
  // Keep the input shared so the selective reader creates a fresh result.
  VectorPtr holder = batch;
  rowReader->next(1, batch);

  {
    auto root = std::dynamic_pointer_cast<RowVector>(batch);
    EXPECT_EQ(3, root->childrenSize());

    auto* nested = root->childAt(1)->loadedVector()->as<RowVector>();
    EXPECT_EQ(4, nested->childrenSize());
    EXPECT_EQ(1, nested->size());

    // Column 3 should be filled with NULLs
    EXPECT_EQ(1, nested->childAt(3)->getNullCount().value_or(0));
    EXPECT_TRUE(nested->childAt(3)->isNullAt(0));

    // Column 0 should be null since it's not selected
    EXPECT_FALSE(nested->childAt(0));

    // float column should be selected and not null
    auto* fv = root->childAt(2)->loadedVector()->as<FlatVector<float>>();
    EXPECT_EQ(1, fv->size());
    EXPECT_EQ(0, fv->getNullCount().value_or(0));
  }
}

TEST_F(TestReader, testMismatchSchemaNestedFewerFields) {
  // file has schema: a int, b struct<a:int, b:float>, c float
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  RowReaderOptions rowReaderOpts;
  std::shared_ptr<const RowType> requestedType =
      std::dynamic_pointer_cast<const RowType>(HiveTypeParser().parse(
          "struct<a:int,b:struct<a:int,b:float>,c:float>"));
  auto reader = DwrfReader::create(
      createFileBufferedInput(getStructFile(), readerOpts.memoryPool()),
      readerOpts);
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  auto* bSpec = scanSpec->addField("b", 1);
  bSpec->addFieldRecursively(
      "b", *requestedType->childAt(1)->asRow().childAt(1), 1);
  scanSpec->addFieldRecursively("c", *requestedType->childAt(2), 2);
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(requestedType, 0, pool());
  // Keep the input shared so the selective reader creates a fresh result.
  VectorPtr holder = batch;
  rowReader->next(1, batch);

  {
    auto root = std::dynamic_pointer_cast<RowVector>(batch);
    EXPECT_EQ(3, root->childrenSize());

    auto* nested = root->childAt(1)->loadedVector()->as<RowVector>();
    EXPECT_EQ(2, nested->childrenSize());
    EXPECT_EQ(1, nested->size());

    // Column 0 should have size 0 since it's not selected
    EXPECT_FALSE(nested->childAt(0));

    // float column should be selected and not null
    auto* fv = root->childAt(2)->loadedVector()->as<FlatVector<float>>();
    EXPECT_EQ(1, fv->size());
    EXPECT_EQ(0, fv->getNullCount().value_or(0));
  }
}

TEST_F(TestReader, testMismatchSchemaIncompatibleNotSelected) {
  // file has schema: a int, b struct<a:int, b:float>, c float
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  RowReaderOptions rowReaderOpts;
  std::shared_ptr<const RowType> requestedType =
      std::dynamic_pointer_cast<const RowType>(HiveTypeParser().parse(
          "struct<a:float,b:struct<a:string,b:float>,c:int>"));
  auto reader = DwrfReader::create(
      createFileBufferedInput(getStructFile(), readerOpts.memoryPool()),
      readerOpts);
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  auto* bSpec = scanSpec->addField("b", 1);
  bSpec->addFieldRecursively(
      "b", *requestedType->childAt(1)->asRow().childAt(1), 1);
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(requestedType, 0, pool());
  // Keep the input shared so the selective reader creates a fresh result.
  VectorPtr holder = batch;
  rowReader->next(1, batch);

  {
    auto root = std::dynamic_pointer_cast<RowVector>(batch);
    EXPECT_EQ(3, root->childrenSize());

    auto* nested = root->childAt(1)->loadedVector()->as<RowVector>();
    EXPECT_EQ(2, nested->childrenSize());
    EXPECT_EQ(1, nested->size());

    // Column 0 should have size 0 since it's not selected
    EXPECT_FALSE(nested->childAt(0));
    // Column 1 should be selected and not null
    EXPECT_EQ(nested->childAt(1)->size(), 1);
    EXPECT_EQ(0, nested->childAt(1)->getNullCount().value_or(0));

    // Columns not selected should have nullptr
    EXPECT_FALSE(root->childAt(0));
    EXPECT_FALSE(root->childAt(2));
  }
}

TEST_F(TestReader, testMismatchSchemaIncompatible) {
  MockStripeStreams streams;

  // set getEncoding
  proto::ColumnEncoding directEncoding;
  directEncoding.set_kind(proto::ColumnEncoding_Kind_DIRECT);
  EXPECT_CALL(streams, getEncodingProxy(_))
      .WillRepeatedly(Return(&directEncoding));

  std::shared_ptr<const RowType> rowType =
      std::dynamic_pointer_cast<const RowType>(
          HiveTypeParser().parse("struct<col0:int>"));

  auto types = folly::make_array<std::string>("float", "smallint");
  EncodingKey root(0, 0);
  for (auto& t : types) {
    std::shared_ptr<const RowType> reqType =
        std::dynamic_pointer_cast<const RowType>(
            HiveTypeParser().parse(fmt::format("struct<col0:{}>", t)));
    EXPECT_THROW(
        ColumnSelector cs(reqType, rowType), facebook::velox::VeloxUserError);
  }
}

namespace {
// Writes 'fileData' (whose type is the physical file schema) to a DWRF file,
// then opens it requesting 'tableSchema' with 'columnMappingMode' and reads all
// columns back. Returns the reader (so the caller can inspect rowType()) and
// the fully-read result vector. In ColumnMappingMode::kName the reader renames
// the file columns to the table schema by position only when the file's
// physical names are all Hive placeholders; otherwise it keeps the file's
// physical names. In ColumnMappingMode::kPosition it always renames by
// position.
std::pair<std::unique_ptr<DwrfReader>, RowVectorPtr> readWithColumnMapping(
    memory::MemoryPool& pool,
    const RowVectorPtr& fileData,
    const RowTypePtr& tableSchema,
    dwio::common::ColumnMappingMode columnMappingMode) {
  auto sink = std::make_unique<MemorySink>(
      16 * 1024 * 1024, dwio::common::FileSink::Options{.pool = &pool});
  auto* sinkPtr = sink.get();
  // Writer owns sink. Keep the writer alive until the data is copied out,
  // otherwise destroying the writer frees the sink and sinkPtr dangles.
  auto writer = E2EWriterTestUtil::writeData(
      std::move(sink),
      asRowType(fileData->type()),
      {fileData},
      std::make_shared<dwrf::Config>());

  dwio::common::ReaderOptions readerOpts(&pool);
  readerOpts.setColumnMappingMode(columnMappingMode);
  readerOpts.setFileSchema(tableSchema);
  std::string data(sinkPtr->data(), sinkPtr->size());
  auto reader = std::make_unique<DwrfReader>(
      readerOpts,
      std::make_unique<BufferedInput>(
          std::make_shared<InMemoryReadFile>(std::move(data)),
          readerOpts.memoryPool()));

  // Read all columns using the reader's (possibly renamed) schema.
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*reader->rowType()));
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr result = BaseVector::create(reader->rowType(), 0, &pool);
  rowReader->next(fileData->size(), result);
  result->loadedVector();
  return {std::move(reader), std::dynamic_pointer_cast<RowVector>(result)};
}
} // namespace

// File written by old Hive with placeholder names (_col0, _col1). In name-based
// mapping the requested names (id, name) are absent from the file, so a plain
// by-name read would find nothing. Because every physical name is a Hive
// placeholder, the reader renames the file columns to the table schema by
// position, and the data reads back under the requested names.
TEST_F(TestReader, columnMappingPositionalFallbackForHivePlaceholders) {
  auto fileData = makeRowVector(
      {"_col0", "_col1"},
      {makeFlatVector<int32_t>({7, 8}),
       makeFlatVector<StringView>({"a", "b"})});
  auto tableSchema = ROW({"id", "name"}, {INTEGER(), VARCHAR()});
  auto [reader, result] = readWithColumnMapping(
      *pool(), fileData, tableSchema, dwio::common::ColumnMappingMode::kName);

  // File columns were renamed to the table schema by position.
  EXPECT_TRUE(*reader->rowType() == *tableSchema);
  ASSERT_TRUE(result != nullptr);
  auto expected = makeRowVector(
      {makeFlatVector<int32_t>({7, 8}),
       makeFlatVector<StringView>({"a", "b"})});
  assertEqualVectors(expected, result);
}

// File has real physical names that do not match the requested names. In
// name-based mapping (not all placeholders) the reader must NOT rename by
// position; the file's physical names are preserved so a downstream by-name
// match binds to the real columns.
TEST_F(TestReader, columnMappingRealNamesMappedByName) {
  auto fileData = makeRowVector(
      {"uid", "label"},
      {makeFlatVector<int32_t>({7, 8}),
       makeFlatVector<StringView>({"a", "b"})});
  auto tableSchema = ROW({"id", "name"}, {INTEGER(), VARCHAR()});
  auto [reader, result] = readWithColumnMapping(
      *pool(), fileData, tableSchema, dwio::common::ColumnMappingMode::kName);

  // File columns are NOT renamed; the physical names are preserved.
  EXPECT_EQ(reader->rowType()->nameOf(0), "uid");
  EXPECT_EQ(reader->rowType()->nameOf(1), "label");
  ASSERT_TRUE(result != nullptr);
  auto expected = makeRowVector(
      {makeFlatVector<int32_t>({7, 8}),
       makeFlatVector<StringView>({"a", "b"})});
  assertEqualVectors(expected, result);
}

// File has real physical names but the caller requested position-based mapping
// (e.g. Spark's orc.force.positional.evolution, expressed as kPosition). The
// reader renames the file columns to the table schema by position, and the data
// reads back under the requested names.
TEST_F(TestReader, columnMappingRealNamesMappedByPosition) {
  auto fileData = makeRowVector(
      {"uid", "label"},
      {makeFlatVector<int32_t>({7, 8}),
       makeFlatVector<StringView>({"a", "b"})});
  auto tableSchema = ROW({"id", "name"}, {INTEGER(), VARCHAR()});
  auto [reader, result] = readWithColumnMapping(
      *pool(),
      fileData,
      tableSchema,
      dwio::common::ColumnMappingMode::kPosition);

  // File columns were renamed to the table schema by position.
  EXPECT_TRUE(*reader->rowType() == *tableSchema);
  ASSERT_TRUE(result != nullptr);
  auto expected = makeRowVector(
      {makeFlatVector<int32_t>({7, 8}),
       makeFlatVector<StringView>({"a", "b"})});
  assertEqualVectors(expected, result);
}

TEST_F(TestReader, fileColumnNamesReadAsLowerCase) {
  // upper.orc holds one columns (Bool_Val: BOOLEAN, b: BIGINT)
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setFileColumnNamesReadAsLowerCase(true);
  auto reader = DwrfReader::create(
      createFileBufferedInput(
          getExampleFilePath("upper.orc"), readerOpts.memoryPool()),
      readerOpts);
  auto type = reader->typeWithId();
  auto col0 = type->childAt(0);
  EXPECT_EQ(type->childByName("bool_val"), col0);
}

TEST_F(TestReader, fileColumnNamesReadAsLowerCaseComplexStruct) {
  // upper_complex.orc holds type
  // Cc:struct<CcLong0:bigint,CcMap1:map<string,struct<CcArray2:array<struct<CcInt3:int>>>>>
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setFileColumnNamesReadAsLowerCase(true);
  auto reader = DwrfReader::create(
      createFileBufferedInput(
          getExampleFilePath("upper_complex.orc"), readerOpts.memoryPool()),
      readerOpts);
  auto type = reader->typeWithId();

  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::ROW);
  EXPECT_EQ(type->childByName("cc"), col0);

  auto col0_0 = col0->childAt(0);
  EXPECT_EQ(col0_0->type()->kind(), TypeKind::BIGINT);
  EXPECT_EQ(col0->childByName("cclong0"), col0_0);

  auto col0_1 = col0->childAt(1);
  EXPECT_EQ(col0_1->type()->kind(), TypeKind::MAP);
  EXPECT_EQ(col0->childByName("ccmap1"), col0_1);

  auto col0_1_0 = col0_1->childAt(0);
  EXPECT_EQ(col0_1_0->type()->kind(), TypeKind::VARCHAR);

  auto col0_1_1 = col0_1->childAt(1);
  EXPECT_EQ(col0_1_1->type()->kind(), TypeKind::ROW);

  auto col0_1_1_0 = col0_1_1->childAt(0);
  EXPECT_EQ(col0_1_1_0->type()->kind(), TypeKind::ARRAY);
  EXPECT_EQ(col0_1_1->childByName("ccarray2"), col0_1_1_0);

  auto col0_1_1_0_0 = col0_1_1_0->childAt(0);
  EXPECT_EQ(col0_1_1_0_0->type()->kind(), TypeKind::ROW);
  auto col0_1_1_0_0_0 = col0_1_1_0_0->childAt(0);
  EXPECT_EQ(col0_1_1_0_0_0->type()->kind(), TypeKind::INTEGER);
  EXPECT_EQ(col0_1_1_0_0->childByName("ccint3"), col0_1_1_0_0_0);
}

TEST_F(TestReader, testStripeSizeCallback) {
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setFilePreloadThreshold(0);
  readerOpts.setFooterSpeculativeIoSize(17);
  RowReaderOptions rowReaderOpts;

  std::shared_ptr<const RowType> requestedType = std::dynamic_pointer_cast<
      const RowType>(HiveTypeParser().parse(
      "struct<int_column:int,string_column:string,string_column_2:string,ds:string>"));
  rowReaderOpts.select(std::make_shared<ColumnSelector>(requestedType));
  rowReaderOpts.setEagerFirstStripeLoad(false);
  uint16_t stripeCount = 0;
  int numCalls = 0;
  rowReaderOpts.setStripeCountCallback([&](uint16_t count) {
    stripeCount += count;
    ++numCalls;
  });

  auto reader = DwrfReader::create(
      createFileBufferedInput(
          getExampleFilePath("dict_encoded_strings.orc"),
          readerOpts.memoryPool()),
      readerOpts);
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*requestedType));
  auto rowReaderOwner = reader->createRowReader(rowReaderOpts);
  EXPECT_EQ(stripeCount, 3);
  EXPECT_EQ(numCalls, 1);
}

TEST_F(TestReader, testStripeSizeCallbackLimitsOneStripe) {
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setFilePreloadThreshold(0);
  readerOpts.setFooterSpeculativeIoSize(17);
  RowReaderOptions rowReaderOpts;

  std::shared_ptr<const RowType> requestedType = std::dynamic_pointer_cast<
      const RowType>(HiveTypeParser().parse(
      "struct<int_column:int,string_column:string,string_column_2:string,ds:string>"));
  rowReaderOpts.select(std::make_shared<ColumnSelector>(requestedType));
  rowReaderOpts.setEagerFirstStripeLoad(false);
  rowReaderOpts.range(600, 600);
  uint16_t stripeCount = 0;
  int numCalls = 0;
  rowReaderOpts.setStripeCountCallback([&](uint16_t count) {
    stripeCount += count;
    ++numCalls;
  });

  auto reader = DwrfReader::create(
      createFileBufferedInput(
          getExampleFilePath("dict_encoded_strings.orc"),
          readerOpts.memoryPool()),
      readerOpts);
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*requestedType));
  auto rowReaderOwner = reader->createRowReader(rowReaderOpts);
  EXPECT_EQ(stripeCount, 1);
  EXPECT_EQ(numCalls, 1);
}

TEST_F(TestReader, testStripeSizeCallbackLimitsTwoStripe) {
  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setFilePreloadThreshold(0);
  readerOpts.setFooterSpeculativeIoSize(17);
  RowReaderOptions rowReaderOpts;

  std::shared_ptr<const RowType> requestedType = std::dynamic_pointer_cast<
      const RowType>(HiveTypeParser().parse(
      "struct<int_column:int,string_column:string,string_column_2:string,ds:string>"));
  rowReaderOpts.select(std::make_shared<ColumnSelector>(requestedType));
  rowReaderOpts.setEagerFirstStripeLoad(false);
  rowReaderOpts.range(0, 600);
  uint16_t stripeCount = 0;
  int numCalls = 0;
  rowReaderOpts.setStripeCountCallback([&](uint16_t count) {
    stripeCount += count;
    ++numCalls;
  });

  auto reader = DwrfReader::create(
      createFileBufferedInput(
          getExampleFilePath("dict_encoded_strings.orc"),
          readerOpts.memoryPool()),
      readerOpts);
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*requestedType));
  auto rowReaderOwner = reader->createRowReader(rowReaderOpts);
  EXPECT_EQ(stripeCount, 2);
  EXPECT_EQ(numCalls, 1);
}

TEST_P(TestReaderP, testUpcastBoolean) {
  MockStripeStreams streams;

  // set getEncoding
  proto::ColumnEncoding directEncoding;
  directEncoding.set_kind(proto::ColumnEncoding_Kind_DIRECT);
  EXPECT_CALL(streams, getEncodingProxy(_))
      .WillRepeatedly(Return(&directEncoding));

  // set getStream
  EXPECT_CALL(streams, getStreamProxy(_, proto::Stream_Kind_PRESENT, false))
      .WillRepeatedly(Return(nullptr));

  // [0, 1] * 52 = 104 booleans/bits or 13 bytes
  // 0,1 encoded in a byte is 0101 0101 ->0x55
  // ByteRLE - Repeat->10 (13-MINIMUM_REPEAT), Value - 0x55
  auto data = folly::make_array<char>(10, 0x55);
  EXPECT_CALL(streams, getStreamProxy(1, proto::Stream_Kind_DATA, true))
      .WillRepeatedly(
          Return(new SeekableArrayInputStream(data.data(), data.size())));

  // create the row type
  std::shared_ptr<const RowType> rowType =
      std::dynamic_pointer_cast<const RowType>(
          HiveTypeParser().parse("struct<col0:boolean>"));
  std::shared_ptr<const RowType> reqType =
      std::dynamic_pointer_cast<const RowType>(
          HiveTypeParser().parse("struct<col0:int>"));
  ColumnSelector cs(reqType, rowType);
  EXPECT_CALL(streams, getColumnSelectorProxy()).WillRepeatedly(Return(&cs));
  memory::AllocationPool allocPool(pool());
  StreamLabels labels(allocPool);
  std::unique_ptr<ColumnReader> reader = ColumnReader::build(
      TypeWithId::create(reqType),
      TypeWithId::create(rowType),
      streams,
      labels,
      executor(),
      getDecodingParallelismFactor());

  VectorPtr batch;
  reader->next(104, batch);

  auto lv = std::dynamic_pointer_cast<FlatVector<int32_t>>(
      std::dynamic_pointer_cast<RowVector>(batch)->childAt(0));

  for (size_t i = 0; i < batch->size(); ++i) {
    EXPECT_EQ(lv->valueAt(i), i % 2);
  }
}

TEST_P(TestReaderP, testUpcastIntDirect) {
  MockStripeStreams streams;

  // set getEncoding
  proto::ColumnEncoding directEncoding;
  directEncoding.set_kind(proto::ColumnEncoding_Kind_DIRECT);
  EXPECT_CALL(streams, getEncodingProxy(_))
      .WillRepeatedly(Return(&directEncoding));

  // set getStream
  EXPECT_CALL(streams, getStreamProxy(_, proto::Stream_Kind_PRESENT, false))
      .WillRepeatedly(Return(nullptr));

  // [0..99]
  std::array<char, 100> data;
  std::iota(data.begin(), data.end(), 0);
  EXPECT_CALL(streams, getStreamProxy(1, proto::Stream_Kind_DATA, true))
      .WillRepeatedly(
          Return(new SeekableArrayInputStream(data.data(), data.size())));

  // create the row type
  std::shared_ptr<const RowType> rowType =
      std::dynamic_pointer_cast<const RowType>(
          HiveTypeParser().parse("struct<col0:int>"));
  std::shared_ptr<const RowType> reqType =
      std::dynamic_pointer_cast<const RowType>(
          HiveTypeParser().parse("struct<col0:bigint>"));

  ColumnSelector cs(reqType, rowType);
  EXPECT_CALL(streams, getColumnSelectorProxy()).WillRepeatedly(Return(&cs));
  memory::AllocationPool allocPool(pool());
  StreamLabels labels(allocPool);
  std::unique_ptr<ColumnReader> reader = ColumnReader::build(
      TypeWithId::create(reqType),
      TypeWithId::create(rowType),
      streams,
      labels,
      executor(),
      getDecodingParallelismFactor());

  VectorPtr batch;
  reader->next(100, batch);

  auto lv = std::dynamic_pointer_cast<FlatVector<int64_t>>(
      std::dynamic_pointer_cast<RowVector>(batch)->childAt(0));
  for (size_t i = 0; i < batch->size(); ++i) {
    // bytes in the stream are zig-zag decoded on read
    // so zigzag::decode i to match the value.
    EXPECT_EQ(lv->valueAt(i), zigZagDecode(i));
  }
}

TEST_P(TestReaderP, testUpcastIntDict) {
  MockStripeStreams streams;

  // set getEncoding
  proto::ColumnEncoding directEncoding;
  directEncoding.set_kind(proto::ColumnEncoding_Kind_DIRECT);
  EXPECT_CALL(streams, getEncodingProxy(_))
      .WillRepeatedly(Return(&directEncoding));

  const size_t DICT_SIZE = 100;
  proto::ColumnEncoding dictEncoding;
  dictEncoding.set_kind(proto::ColumnEncoding_Kind_DICTIONARY);
  dictEncoding.set_dictionarysize(DICT_SIZE);
  EXPECT_CALL(streams, getEncodingProxy(1))
      .WillRepeatedly(Return(&dictEncoding));

  // set getStream
  EXPECT_CALL(streams, getStreamProxy(_, proto::Stream_Kind_PRESENT, false))
      .WillRepeatedly(Return(nullptr));

  EXPECT_CALL(
      streams, getStreamProxy(1, proto::Stream_Kind_IN_DICTIONARY, false))
      .WillRepeatedly(Return(nullptr));

  // [0..99] RLE encoded, is length = 100 (subtract -3 minimum repeat, 97 =
  // 0x61), delta - 1, start - 0
  auto data = folly::make_array<char>(0x61, 0x01, 0x00);
  EXPECT_CALL(streams, getStreamProxy(1, proto::Stream_Kind_DATA, true))
      .WillRepeatedly(
          Return(new SeekableArrayInputStream(data.data(), data.size())));

  EXPECT_CALL(streams, genMockDictDataSetter(1, 0))
      .WillRepeatedly(Return([](BufferPtr& buffer, MemoryPool* pool) {
        buffer = AlignedBuffer::allocate<int64_t>(1024, pool);
        setSequence<int64_t>(buffer, 0, 100);
      }));

  // create the row type
  std::shared_ptr<const RowType> rowType =
      std::dynamic_pointer_cast<const RowType>(
          HiveTypeParser().parse("struct<col0:int>"));
  std::shared_ptr<const RowType> reqType =
      std::dynamic_pointer_cast<const RowType>(
          HiveTypeParser().parse("struct<col0:bigint>"));
  ColumnSelector cs(reqType, rowType);
  EXPECT_CALL(streams, getColumnSelectorProxy()).WillRepeatedly(Return(&cs));
  memory::AllocationPool allocPool(pool());
  StreamLabels labels(allocPool);
  std::unique_ptr<ColumnReader> reader = ColumnReader::build(
      TypeWithId::create(reqType),
      TypeWithId::create(rowType),
      streams,
      labels,
      executor(),
      getDecodingParallelismFactor());

  VectorPtr batch;
  reader->next(100, batch);

  auto lv = std::dynamic_pointer_cast<FlatVector<int64_t>>(
      std::dynamic_pointer_cast<RowVector>(batch)->childAt(0));
  for (size_t i = 0; i < batch->size(); ++i) {
    EXPECT_EQ(lv->valueAt(i), i);
  }
}

TEST_P(TestReaderP, testUpcastFloat) {
  MockStripeStreams streams;

  // set getEncoding
  proto::ColumnEncoding directEncoding;
  directEncoding.set_kind(proto::ColumnEncoding_Kind_DIRECT);
  EXPECT_CALL(streams, getEncodingProxy(_))
      .WillRepeatedly(Return(&directEncoding));

  // set getStream
  EXPECT_CALL(streams, getStreamProxy(_, proto::Stream_Kind_PRESENT, false))
      .WillRepeatedly(Return(nullptr));

  // [0..99]
  std::array<char, 100 * 4> data;
  size_t pos = 0;
  for (size_t i = 0; i < 100; ++i) {
    auto val = static_cast<float>(i);
    auto intPtr = reinterpret_cast<int32_t*>(&val);
    for (size_t j = 0; j < sizeof(int32_t); ++j) {
      data.data()[pos++] = static_cast<char>((*intPtr >> (8 * j)) & 0xff);
    }
  }
  EXPECT_CALL(streams, getStreamProxy(1, proto::Stream_Kind_DATA, true))
      .WillRepeatedly(
          Return(new SeekableArrayInputStream(data.data(), data.size())));

  // create the row type
  std::shared_ptr<const RowType> rowType =
      std::dynamic_pointer_cast<const RowType>(
          HiveTypeParser().parse("struct<col0:float>"));
  std::shared_ptr<const RowType> reqType =
      std::dynamic_pointer_cast<const RowType>(
          HiveTypeParser().parse("struct<col0:double>"));
  ColumnSelector cs(reqType, rowType);
  EXPECT_CALL(streams, getColumnSelectorProxy()).WillRepeatedly(Return(&cs));
  memory::AllocationPool allocPool(pool());
  StreamLabels labels(allocPool);
  std::unique_ptr<ColumnReader> reader = ColumnReader::build(
      TypeWithId::create(reqType),
      TypeWithId::create(rowType),
      streams,
      labels,
      executor(),
      getDecodingParallelismFactor());

  VectorPtr batch;
  reader->next(100, batch);

  auto lv = std::dynamic_pointer_cast<FlatVector<double>>(
      std::dynamic_pointer_cast<RowVector>(batch)->childAt(0));
  for (size_t i = 0; i < batch->size(); ++i) {
    EXPECT_EQ(lv->valueAt(i), static_cast<double>(i));
  }
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    TestReaderSerialDecoding,
    TestReaderP,
    Values(false));

VELOX_INSTANTIATE_TEST_SUITE_P(
    TestReaderParallelDecoding,
    TestReaderP,
    Values(true));

TEST_F(TestReader, testEmptyFile) {
  MemorySink sink{1024, {.pool = pool()}};
  DataBufferHolder holder{*pool(), 1024, 0, DEFAULT_PAGE_GROW_RATIO, &sink};
  facebook::velox::dwio::common::BufferedOutputStream output{holder};

  proto::Footer footer;
  footer.set_numberofrows(0);
  auto type = footer.add_types();
  type->set_kind(proto::Type_Kind::Type_Kind_STRUCT);

  footer.SerializeToZeroCopyStream(&output);
  output.flush();
  auto footerLen = sink.size();

  proto::PostScript ps;
  ps.set_footerlength(footerLen);
  ps.set_compression(proto::CompressionKind::NONE);

  ps.SerializeToZeroCopyStream(&output);
  output.flush();
  auto psLen = static_cast<uint8_t>(sink.size() - footerLen);

  DataBuffer<char> buf{*pool(), 1};
  buf.data()[0] = psLen;
  sink.write(std::move(buf));
  std::string data(sink.data(), sink.size());
  auto input = std::make_unique<BufferedInput>(
      std::make_shared<InMemoryReadFile>(std::move(data)), *pool());

  dwio::common::ReaderOptions readerOpts{pool()};
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  RowReaderOptions rowReaderOpts;

  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = DwrfReader::create(std::move(input), readerOpts)
                       ->createRowReader(rowReaderOpts);
  VectorPtr batch;
  EXPECT_FALSE(rowReader->next(1, batch));
  EXPECT_FALSE(batch);
}

TEST_F(TestReader, testFooterWrapper) {
  proto::Footer impl;
  FooterWrapper wrapper(&impl);
  EXPECT_FALSE(wrapper.hasNumberOfRows());
  impl.set_numberofrows(0);
  ASSERT_TRUE(wrapper.hasNumberOfRows());
  EXPECT_EQ(wrapper.numberOfRows(), 0);
}

TEST_F(TestReader, testOrcAndDwrfRowIndexStride) {
  // orc footer
  proto::orc::Footer orcFooter;
  FooterWrapper orcFooterWrapper(&orcFooter);
  EXPECT_FALSE(orcFooterWrapper.hasRowIndexStride());
  orcFooter.set_rowindexstride(100);
  ASSERT_TRUE(orcFooterWrapper.hasRowIndexStride());
  EXPECT_EQ(orcFooterWrapper.rowIndexStride(), 100);

  // dwrf footer
  proto::Footer dwrfFooter;
  FooterWrapper dwrfFooterWrapper(&dwrfFooter);
  EXPECT_FALSE(dwrfFooterWrapper.hasRowIndexStride());
  dwrfFooter.set_rowindexstride(100);
  ASSERT_TRUE(dwrfFooterWrapper.hasRowIndexStride());
  EXPECT_EQ(dwrfFooterWrapper.rowIndexStride(), 100);
}
namespace {

/*
 * Verifies that row numbers are equal to values in first column
 */
void verifyRowNumbers(
    RowReader& rowReader,
    memory::MemoryPool* pool,
    int expectedNumRows,
    bool explicitRowNumber = false) {
  auto result = explicitRowNumber
      ? BaseVector::create(
            ROW({{"c0", INTEGER()}, {"$row_number", BIGINT()}}), 0, pool)
      : BaseVector::create(ROW({{"c0", INTEGER()}}), 0, pool);
  int numRows = 0;
  while (rowReader.next(10, result) > 0) {
    auto* rowVector = result->asUnchecked<RowVector>();
    ASSERT_EQ(2, rowVector->childrenSize());
    ASSERT_EQ(
        rowVector->type()->asRow().nameOf(1),
        explicitRowNumber ? "$row_number" : "");
    DecodedVector values(*rowVector->childAt(0));
    DecodedVector rowNumbers(*rowVector->childAt(1));
    for (size_t i = 0; i < rowVector->size(); ++i) {
      ASSERT_EQ(values.valueAt<int32_t>(i), rowNumbers.valueAt<int64_t>(i));
    }
    numRows += result->size();
  }
  ASSERT_EQ(numRows, expectedNumRows);
}

std::pair<std::unique_ptr<dwrf::Writer>, std::unique_ptr<DwrfReader>>
createWriterReader(
    const std::vector<VectorPtr>& batches,
    memory::MemoryPool* pool,
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats,
    const std::shared_ptr<dwrf::Config>& config =
        std::make_shared<dwrf::Config>(),
    std::function<std::unique_ptr<DWRFFlushPolicy>()> flushPolicy =
        E2EWriterTestUtil::simpleFlushPolicyFactory(true)) {
  auto sink =
      std::make_unique<MemorySink>(1 << 20, FileSink::Options{.pool = pool});
  auto* sinkPtr = sink.get();
  auto writer = E2EWriterTestUtil::writeData(
      std::move(sink),
      asRowType(batches[0]->type()),
      batches,
      config,
      std::move(flushPolicy));
  std::string data(sinkPtr->data(), sinkPtr->size());
  auto input = std::make_unique<BufferedInput>(
      std::make_shared<InMemoryReadFile>(std::move(data)), *pool);
  dwio::common::ReaderOptions readerOpts(pool);
  readerOpts.setDataIoStats(dataIoStats);
  readerOpts.setMetadataIoStats(metadataIoStats);
  readerOpts.setFileFormat(FileFormat::DWRF);
  auto reader = DwrfReader::create(std::move(input), readerOpts);
  return std::make_pair(std::move(writer), std::move(reader));
}

struct FlatMapReaderWithKeyFilter {
  std::unique_ptr<dwrf::Writer> writer;
  std::unique_ptr<DwrfReader> reader;
  std::unique_ptr<dwio::common::RowReader> rowReader;
  RowTypePtr schema;
  VectorPtr batch;
};

FlatMapReaderWithKeyFilter createFlatMapReaderWithKeyFilter(
    const std::vector<VectorPtr>& inputs,
    std::shared_ptr<common::Filter> keyFilter,
    memory::MemoryPool* pool,
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats) {
  auto config = std::make_shared<dwrf::Config>();
  config->set(dwrf::Config::FLATTEN_MAP, true);
  config->set(dwrf::Config::MAP_FLAT_COLS, {0});

  auto [writer, reader] =
      createWriterReader(inputs, pool, dataIoStats, metadataIoStats, config);
  auto schema = asRowType(inputs.front()->type());
  auto scanSpec = makeAllFieldsScanSpec(*schema);
  scanSpec->childByName("c0")
      ->childByName(common::ScanSpec::kMapKeysFieldName)
      ->setFilter(std::move(keyFilter));

  RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setPreserveFlatMapsInMemory(true);
  auto rowReader = reader->createRowReader(rowReaderOptions);
  auto batch = BaseVector::create(schema, 0, pool);
  return {
      std::move(writer),
      std::move(reader),
      std::move(rowReader),
      std::move(schema),
      std::move(batch)};
}

} // namespace

TEST_F(TestReader, setRowNumberColumnInfo) {
  std::vector<std::vector<int32_t>> integerValues{
      {0, 1, 2, 3, 4},
      {5, 6, 7},
      {8},
      {},
      {9, 10, 11, 12, 13, 14, 15},
  };
  auto batches = createBatches(integerValues);
  auto schema = asRowType(batches[0]->type());
  auto [writer, reader] =
      createWriterReader(batches, pool(), dataIoStats_, metadataIoStats_);

  auto spec = makeAllFieldsScanSpec(*schema);
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  RowNumberColumnInfo rowNumberColumnInfo;
  rowNumberColumnInfo.insertPosition = 1;
  rowNumberColumnInfo.name = "";
  rowReaderOpts.setRowNumberColumnInfo(rowNumberColumnInfo);
  {
    SCOPED_TRACE("Selective no filter");
    auto rowReader = reader->createRowReader(rowReaderOpts);
    verifyRowNumbers(*rowReader, pool(), 16);
  }
  spec->childByName("c0")->setFilter(
      common::createBigintValues({1, 4, 5, 7, 11, 14}, false));
  spec->resetCachedValues(true);
  {
    SCOPED_TRACE("Selective with filter");
    auto rowReader = reader->createRowReader(rowReaderOpts);
    verifyRowNumbers(*rowReader, pool(), 6);
  }
}

TEST_F(TestReader, reuseRowNumberColumn) {
  std::vector<std::vector<int32_t>> integerValues{{0, 1, 2, 3, 4}};
  auto batches = createBatches(integerValues);
  auto schema = asRowType(batches[0]->type());
  auto [writer, reader] =
      createWriterReader(batches, pool(), dataIoStats_, metadataIoStats_);

  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*schema));
  RowNumberColumnInfo rowNumberColumnInfo;
  rowNumberColumnInfo.insertPosition = 1;
  rowNumberColumnInfo.name = "";
  rowReaderOpts.setRowNumberColumnInfo(rowNumberColumnInfo);
  {
    SCOPED_TRACE("Reuse passed in");
    auto rowReader = reader->createRowReader(rowReaderOpts);
    auto result =
        BaseVector::create(ROW({{"c0", INTEGER()}, {"", BIGINT()}}), 0, pool());
    auto* rowNum = result->asUnchecked<RowVector>()->childAt(1).get();
    ASSERT_EQ(rowReader->next(3, result), 3);
    ASSERT_EQ(rowNum, result->asUnchecked<RowVector>()->childAt(1).get());
  }
  {
    SCOPED_TRACE("Reuse generated");
    auto rowReader = reader->createRowReader(rowReaderOpts);
    auto result = BaseVector::create(ROW({{"c0", INTEGER()}}), 0, pool());
    ASSERT_EQ(rowReader->next(3, result), 3);
    auto* rowNum = result->asUnchecked<RowVector>()->childAt(1).get();
    ASSERT_EQ(rowReader->next(3, result), 2);
    ASSERT_EQ(rowNum, result->asUnchecked<RowVector>()->childAt(1).get());
  }
  {
    SCOPED_TRACE("No reuse passed in");
    auto rowReader = reader->createRowReader(rowReaderOpts);
    auto result =
        BaseVector::create(ROW({{"c0", INTEGER()}, {"", BIGINT()}}), 0, pool());
    auto rowNum = result->asUnchecked<RowVector>()->childAt(1);
    ASSERT_EQ(rowReader->next(3, result), 3);
    ASSERT_NE(rowNum.get(), result->asUnchecked<RowVector>()->childAt(1).get());
  }
  {
    SCOPED_TRACE("No reuse generated");
    auto rowReader = reader->createRowReader(rowReaderOpts);
    auto result = BaseVector::create(ROW({{"c0", INTEGER()}}), 0, pool());
    ASSERT_EQ(rowReader->next(3, result), 3);
    auto rowNum = result->asUnchecked<RowVector>()->childAt(1);
    ASSERT_EQ(rowReader->next(3, result), 2);
    ASSERT_NE(rowNum.get(), result->asUnchecked<RowVector>()->childAt(1).get());
  }
  {
    SCOPED_TRACE("No reuse type mismatch");
    auto rowReader = reader->createRowReader(rowReaderOpts);
    auto result = BaseVector::create(
        ROW({{"c0", INTEGER()}, {"", INTEGER()}}), 0, pool());
    auto rowNum = result->asUnchecked<RowVector>()->childAt(1);
    ASSERT_EQ(rowReader->next(3, result), 3);
    ASSERT_NE(rowNum.get(), result->asUnchecked<RowVector>()->childAt(1).get());
  }
}

TEST_F(TestReader, explicitRowNumberColumn) {
  const std::vector<std::vector<int32_t>> integerValues{
      {0, 1, 2, 3, 4},
      {5, 6, 7},
      {8},
      {},
      {9, 10, 11, 12, 13, 14, 15},
  };
  auto batches = createBatches(integerValues);
  auto [writer, reader] =
      createWriterReader(batches, pool(), dataIoStats_, metadataIoStats_);
  auto spec = std::make_shared<common::ScanSpec>("<root>");
  spec->addField("c0", 0);
  spec->addField("$row_number", 1)
      ->setColumnType(common::ScanSpec::ColumnType::kRowIndex);
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  {
    SCOPED_TRACE("Selective no filter");
    auto rowReader = reader->createRowReader(rowReaderOpts);
    verifyRowNumbers(*rowReader, pool(), 16, true);
  }
  spec->childByName("c0")->setFilter(
      common::createBigintValues({1, 4, 5, 7, 11, 14}, false));
  spec->resetCachedValues(true);
  {
    SCOPED_TRACE("Selective with filter");
    auto rowReader = reader->createRowReader(rowReaderOpts);
    verifyRowNumbers(*rowReader, pool(), 6, true);
  }
}

TEST_F(TestReader, failToReuseReaderNulls) {
  auto c0 = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>(11, folly::identity),
          makeFlatVector<int64_t>(
              11, folly::identity, [](auto i) { return i % 3 == 0; }),
      });
  // Set a null so that the children will not be loaded lazily.
  bits::setNull(c0->mutableRawNulls(), 10);
  auto data = makeRowVector({
      c0,
      makeRowVector({"c"}, {makeFlatVector<int64_t>(11, folly::identity)}),
  });
  auto schema = asRowType(data->type());
  auto [writer, reader] =
      createWriterReader({data}, pool(), dataIoStats_, metadataIoStats_);
  auto spec = makeAllFieldsScanSpec(*schema);
  spec->childByName("c0")->childByName("a")->setFilter(
      std::make_unique<common::BigintRange>(
          0, std::numeric_limits<int64_t>::max(), false));
  spec->childByName("c1")->childByName("c")->setFilter(
      std::make_unique<common::BigintRange>(0, 4, false));
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  auto result = BaseVector::create(schema, 0, pool());
  ASSERT_EQ(rowReader->next(10, result), 10);
  ASSERT_EQ(result->size(), 5);
  for (int i = 0; i < result->size(); ++i) {
    ASSERT_TRUE(result->equalValueAt(data.get(), i, i)) << result->toString(i);
  }
}

TEST_F(TestReader, readFlatMapsSomeEmpty) {
  // Test reading a flat map where the key filter means that some maps are
  // empty.
  auto keys = makeFlatVector(
      std::vector<int64_t>{
          1,
          2,
          3,
          4,
          5,
          6, // map 1 has more than just the selected keys.
          1,
          2,
          3, // map 2 has only selected keys.
          4,
          5,
          6, // map 3 has no selected keys.
          1,
          2,
          5,
          6 // map 4 has some selected keys.
      });
  auto values = makeFlatVector<int64_t>(16, folly::identity);
  auto maps =
      makeMapVector(std::vector<vector_size_t>{0, 6, 9, 12, 16}, keys, values);
  auto row = makeRowVector({"a"}, {maps});

  // Set up the config so that the maps are flattened.
  std::shared_ptr<dwrf::Config> config = std::make_shared<dwrf::Config>();
  config->set(dwrf::Config::FLATTEN_MAP, true);
  config->set(dwrf::Config::MAP_FLAT_COLS, {0});

  auto [writer, reader] =
      createWriterReader({row}, pool(), dataIoStats_, metadataIoStats_, config);

  auto schema = asRowType(row->type());
  auto spec = makeAllFieldsScanSpec(*schema);
  spec->childByName("a")
      ->childByName(common::ScanSpec::kMapKeysFieldName)
      ->setFilter(common::createBigintValues({1, 2, 3}, false));
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);

  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(schema, 0, pool());

  ASSERT_TRUE(rowReader->next(4, batch));
  auto rowVector = batch->as<RowVector>();
  auto resultMaps = rowVector->childAt(0)->loadedVector()->as<MapVector>();
  ASSERT_EQ(resultMaps->size(), 4);
  auto resultKeys = resultMaps->mapKeys()->as<SimpleVector<int64_t>>();
  auto resultValues = resultMaps->mapValues()->as<SimpleVector<int64_t>>();

  auto validate = [&](vector_size_t index,
                      vector_size_t expectedSize,
                      const std::unordered_set<int64_t>& expectedKeys,
                      const std::unordered_set<int64_t>& expectedValues) {
    ASSERT_FALSE(resultMaps->isNullAt(index));

    vector_size_t offset = resultMaps->offsetAt(index);
    vector_size_t size = resultMaps->sizeAt(index);
    ASSERT_EQ(size, expectedSize);

    std::unordered_set<int64_t> keySet;
    std::unordered_set<int64_t> valueSet;
    for (int i = offset; i < offset + size; i++) {
      keySet.insert(resultKeys->valueAt(i));
      valueSet.insert(resultValues->valueAt(i));
    }

    EXPECT_EQ(keySet, expectedKeys);
    EXPECT_EQ(valueSet, expectedValues);
  };

  validate(0, 3, {1, 2, 3}, {0, 1, 2});
  validate(1, 3, {1, 2, 3}, {6, 7, 8});
  validate(2, 0, {}, {});
  validate(3, 2, {1, 2}, {12, 13});
}

TEST_F(TestReader, readFlatMapsWithNullMaps) {
  // Test reading a flat map where the key filter means that some maps are
  // empty.
  auto keys =
      makeFlatVector<int64_t>(16, [](vector_size_t row) { return row % 4; });
  auto values = makeFlatVector<int64_t>(16, folly::identity);
  auto maps = makeMapVector(
      std::vector<vector_size_t>{0, 4, 4, 8, 8, 12, 12, 16, 16},
      keys,
      values,
      {1, 3, 5, 7});
  auto row = makeRowVector({"a"}, {maps});

  // Set up the config so that the maps are flattened.
  std::shared_ptr<dwrf::Config> config = std::make_shared<dwrf::Config>();
  config->set(dwrf::Config::FLATTEN_MAP, true);
  config->set(dwrf::Config::MAP_FLAT_COLS, {0});

  auto [writer, reader] =
      createWriterReader({row}, pool(), dataIoStats_, metadataIoStats_, config);

  auto schema = asRowType(row->type());
  auto spec = makeAllFieldsScanSpec(*schema);
  spec->childByName("a")
      ->childByName(common::ScanSpec::kMapKeysFieldName)
      ->setFilter(common::createBigintValues({1, 2, 3}, false));
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);

  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(schema, 0, pool());

  ASSERT_TRUE(rowReader->next(8, batch));
  auto rowVector = batch->as<RowVector>();
  auto resultMaps = rowVector->childAt(0)->loadedVector()->as<MapVector>();
  ASSERT_EQ(resultMaps->size(), 8);
  auto resultKeys = resultMaps->mapKeys()->as<SimpleVector<int64_t>>();
  auto resultValues = resultMaps->mapValues()->as<SimpleVector<int64_t>>();

  for (int mapIndex = 0; mapIndex < 8; mapIndex++) {
    if (mapIndex % 2 != 0) {
      ASSERT_TRUE(resultMaps->isNullAt(mapIndex));
    } else {
      ASSERT_FALSE(resultMaps->isNullAt(mapIndex));

      vector_size_t offset = resultMaps->offsetAt(mapIndex);
      vector_size_t size = resultMaps->sizeAt(mapIndex);
      ASSERT_EQ(size, 3);

      std::unordered_set<int64_t> keySet;
      std::unordered_set<int64_t> valueSet;
      for (int i = offset; i < offset + size; i++) {
        keySet.insert(resultKeys->valueAt(i));
        valueSet.insert(resultValues->valueAt(i));
      }

      EXPECT_EQ(keySet, (std::unordered_set<int64_t>{1, 2, 3}));
      EXPECT_EQ(
          valueSet,
          (std::unordered_set<int64_t>{
              4 * mapIndex / 2 + 1,
              4 * mapIndex / 2 + 2,
              4 * mapIndex / 2 + 3}));
    }
  }
}

TEST_F(TestReader, readFlatMapsAsFlatMaps) {
  auto testRoundTrip = [&](const FlatMapVectorPtr& flatMap) {
    auto input = makeRowVector({flatMap->toMapVector()});

    std::shared_ptr<dwrf::Config> config = std::make_shared<dwrf::Config>();
    config->set(dwrf::Config::FLATTEN_MAP, true);
    config->set(dwrf::Config::MAP_FLAT_COLS, {0});

    auto [writer, reader] = createWriterReader(
        {input}, pool(), dataIoStats_, metadataIoStats_, config);

    auto schema = asRowType(input->type());

    RowReaderOptions rowReaderOpts;
    rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*schema));
    rowReaderOpts.setPreserveFlatMapsInMemory(true);

    auto rowReader = reader->createRowReader(rowReaderOpts);
    VectorPtr batch = BaseVector::create(schema, 0, pool());

    rowReader->next(flatMap->size(), batch);
    auto rowVector = batch->as<RowVector>();
    auto resultMaps = rowVector->childAt(0);

    assertEqualVectors(flatMap, resultMaps);
  };

  testRoundTrip(
      makeFlatMapVector<int16_t, float>({
          {},
          {{1, 1.9}, {2, 2.1}, {0, 3.12}},
          {{127, 0.12}},
      }));

  testRoundTrip(
      makeFlatMapVector<StringView, StringView>({
          {{"a", "a1"}},
          {{"b", "b1"}},
          {{"c", "c1"}},
          {{"d", "d1"}},
      }));

  testRoundTrip(
      makeNullableFlatMapVector<int32_t, int32_t>({
          {{{101, 1}, {102, 2}, {103, 3}}},
          {{{105, 0}, {106, 0}}},
          {std::nullopt},
          {{{101, 11}, {103, 13}, {105, std::nullopt}}},
          {{{101, 1}, {102, 2}, {103, 3}}},
      }));

  testRoundTrip(
      makeFlatMapVector<int64_t, int64_t>(
          {{{0, 0}, {1, 1}, {2, 2}, {3, 3}},
           {{0, 4}, {1, 5}, {2, 6}, {3, 7}},
           {{0, 8}, {1, 9}, {2, 10}, {3, 11}},
           {{0, 12}, {1, 13}, {2, 14}, {3, 15}}}));
}

TEST_F(TestReader, readFlatMapsAsFlatMapsWithKeyFilter) {
  // Reading a flat map with preserveFlatMapsInMemory=true must honor a
  // requested-key filter and project only the selected keys into the output
  // FlatMapVector.
  auto flatMap = makeFlatMapVector<int64_t, int64_t>({
      {{0, 0}, {1, 1}, {2, 2}, {3, 3}},
      {{0, 4}, {1, 5}, {2, 6}, {3, 7}},
      {{0, 8}, {1, 9}, {2, 10}, {3, 11}},
  });
  auto input = makeRowVector({flatMap->toMapVector()});
  auto result = createFlatMapReaderWithKeyFilter(
      {input},
      common::createBigintValues({1, 2}, false),
      pool(),
      dataIoStats_,
      metadataIoStats_);

  ASSERT_EQ(
      result.rowReader->next(flatMap->size(), result.batch), flatMap->size());
  auto rowVector = result.batch->as<RowVector>();
  auto resultFlatMap =
      rowVector->childAt(0)->loadedVector()->as<FlatMapVector>();
  ASSERT_TRUE(resultFlatMap);

  // Only the selected keys are projected out.
  auto distinctKeys =
      resultFlatMap->distinctKeys()->as<SimpleVector<int64_t>>();
  std::unordered_set<int64_t> keySet;
  for (auto i = 0; i < distinctKeys->size(); ++i) {
    keySet.insert(distinctKeys->valueAt(i));
  }
  EXPECT_EQ(keySet, (std::unordered_set<int64_t>{1, 2}));

  auto expected = makeFlatMapVector<int64_t, int64_t>({
      {{1, 1}, {2, 2}},
      {{1, 5}, {2, 6}},
      {{1, 9}, {2, 10}},
  });
  assertEqualVectors(expected->toMapVector(), resultFlatMap->toMapVector());
}

TEST_F(TestReader, readFlatMapsAsFlatMapsWithStringKeyFilter) {
  // Key pruning in the preserving path must also work for string keys.
  auto flatMap = makeFlatMapVector<StringView, int64_t>({
      {{"a", 1}, {"b", 2}, {"c", 3}},
      {{"a", 4}, {"b", 5}, {"c", 6}},
  });
  auto input = makeRowVector({flatMap->toMapVector()});
  auto result = createFlatMapReaderWithKeyFilter(
      {input},
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"a", "c"}, false),
      pool(),
      dataIoStats_,
      metadataIoStats_);

  ASSERT_EQ(
      result.rowReader->next(flatMap->size(), result.batch), flatMap->size());
  auto resultFlatMap = result.batch->as<RowVector>()
                           ->childAt(0)
                           ->loadedVector()
                           ->as<FlatMapVector>();
  ASSERT_TRUE(resultFlatMap);

  auto expected = makeFlatMapVector<StringView, int64_t>({
      {{"a", 1}, {"c", 3}},
      {{"a", 4}, {"c", 6}},
  });
  assertEqualVectors(expected->toMapVector(), resultFlatMap->toMapVector());
}

TEST_F(TestReader, readFlatMapsAsFlatMapsKeyFilterExcludesAllKeys) {
  // A key filter that matches no key in the stripe yields N empty maps, not
  // zero rows.
  auto flatMap = makeFlatMapVector<int64_t, int64_t>({
      {{0, 0}, {1, 1}},
      {{0, 2}, {1, 3}},
  });
  auto input = makeRowVector({flatMap->toMapVector()});
  auto result = createFlatMapReaderWithKeyFilter(
      {input},
      common::createBigintValues({99}, false),
      pool(),
      dataIoStats_,
      metadataIoStats_);

  ASSERT_EQ(
      result.rowReader->next(flatMap->size(), result.batch), flatMap->size());
  auto resultFlatMap = result.batch->as<RowVector>()
                           ->childAt(0)
                           ->loadedVector()
                           ->as<FlatMapVector>();
  ASSERT_TRUE(resultFlatMap);
  EXPECT_EQ(resultFlatMap->distinctKeys()->size(), 0);

  auto resultMaps = resultFlatMap->toMapVector();
  ASSERT_EQ(resultMaps->size(), 2);
  for (vector_size_t row = 0; row < resultMaps->size(); ++row) {
    EXPECT_FALSE(resultMaps->isNullAt(row));
    EXPECT_EQ(resultMaps->sizeAt(row), 0);
  }
}

TEST_F(TestReader, readFlatMapsAsFlatMapsWithKeyFilterAndNullMaps) {
  // Key pruning must compose with null maps: null rows stay null and non-null
  // rows are pruned to the requested keys.
  auto flatMap = makeNullableFlatMapVector<int64_t, int64_t>({
      {{{0, 0}, {1, 1}, {2, 2}, {3, 3}}},
      {std::nullopt},
      {{{0, 4}, {1, 5}, {2, 6}, {3, 7}}},
      {std::nullopt},
  });
  auto input = makeRowVector({flatMap->toMapVector()});
  auto result = createFlatMapReaderWithKeyFilter(
      {input},
      common::createBigintValues({1, 2, 3}, false),
      pool(),
      dataIoStats_,
      metadataIoStats_);

  ASSERT_EQ(
      result.rowReader->next(flatMap->size(), result.batch), flatMap->size());
  auto resultFlatMap = result.batch->as<RowVector>()
                           ->childAt(0)
                           ->loadedVector()
                           ->as<FlatMapVector>();
  ASSERT_TRUE(resultFlatMap);

  auto expected = makeNullableFlatMapVector<int64_t, int64_t>({
      {{{1, 1}, {2, 2}, {3, 3}}},
      {std::nullopt},
      {{{1, 5}, {2, 6}, {3, 7}}},
      {std::nullopt},
  });
  assertEqualVectors(expected->toMapVector(), resultFlatMap->toMapVector());
}

TEST_F(TestReader, readFlatMapsAsFlatMapsMultiStripeWithKeyFilter) {
  // The key filter must prune every stripe, not just the first. The ScanSpec
  // is shared across stripes, so a fix that consumes/clears the filter would
  // silently stop pruning after stripe 1.
  auto stripe1 = makeRowVector({makeMapVector<int64_t, int64_t>({
      {{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}},
      {{0, 20}, {1, 21}, {2, 22}, {3, 23}, {4, 24}},
  })});
  auto stripe2 = makeRowVector({makeMapVector<int64_t, int64_t>({
      {{0, 30}, {1, 31}, {2, 32}},
      {{0, 40}, {1, 41}, {2, 42}},
      {{0, 50}, {1, 51}, {2, 52}},
  })});

  // simpleFlushPolicyFactory(true) produces one stripe per batch.
  auto result = createFlatMapReaderWithKeyFilter(
      {stripe1, stripe2},
      common::createBigintValues({1, 2}, false),
      pool(),
      dataIoStats_,
      metadataIoStats_);
  ASSERT_EQ(result.reader->getNumberOfStripes(), 2);

  uint64_t totalRows = 0;
  // Read one row at a time to exercise repeated lazy materialization at
  // non-zero reader offsets as well as the stripe transition.
  while (result.rowReader->next(1, result.batch) > 0) {
    auto resultMaps = result.batch->as<RowVector>()
                          ->childAt(0)
                          ->loadedVector()
                          ->as<FlatMapVector>()
                          ->toMapVector();
    auto* resultKeys = resultMaps->mapKeys()->as<SimpleVector<int64_t>>();
    for (vector_size_t row = 0; row < resultMaps->size(); ++row) {
      std::unordered_set<int64_t> keySet;
      const auto offset = resultMaps->offsetAt(row);
      for (vector_size_t i = 0; i < resultMaps->sizeAt(row); ++i) {
        keySet.insert(resultKeys->valueAt(offset + i));
      }
      EXPECT_EQ(keySet, (std::unordered_set<int64_t>{1, 2}));
    }
    totalRows += result.batch->size();
  }
  EXPECT_EQ(totalRows, 5); // 2 rows from stripe 1 + 3 rows from stripe 2.
}

// Regression test: reading a multi-stripe flatmap file with
// preserveFlatMapsInMemory=true used to crash when stripes had different
// key sets. The ScanSpec accumulated stale children across stripes, causing
// out-of-range access on the reader's children_ vector.
TEST_F(TestReader, readFlatMapMultiStripeDifferentKeys) {
  // Stripe 1: keys {0, 1, 2, 3, 4} — 5 keys.
  auto stripe1 = makeRowVector({makeMapVector<int32_t, float>(
      {{{0, 1.0f}, {1, 2.0f}, {2, 3.0f}, {3, 4.0f}, {4, 5.0f}},
       {{0, 6.0f}, {1, 7.0f}, {2, 8.0f}, {3, 9.0f}, {4, 10.0f}}})});

  // Stripe 2: keys {0, 1, 2} — fewer keys than stripe 1.
  auto stripe2 = makeRowVector({makeMapVector<int32_t, float>(
      {{{0, 11.0f}, {1, 12.0f}, {2, 13.0f}},
       {{0, 14.0f}, {1, 15.0f}, {2, 16.0f}},
       {{0, 17.0f}, {1, 18.0f}, {2, 19.0f}}})});

  auto config = std::make_shared<dwrf::Config>();
  config->set(dwrf::Config::FLATTEN_MAP, true);
  config->set(dwrf::Config::MAP_FLAT_COLS, {0});

  // simpleFlushPolicyFactory(true) produces one stripe per batch.
  auto [writer, reader] = createWriterReader(
      {stripe1, stripe2}, pool(), dataIoStats_, metadataIoStats_, config);
  ASSERT_EQ(reader->getNumberOfStripes(), 2);

  auto schema = asRowType(stripe1->type());

  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*schema));
  rowReaderOpts.setPreserveFlatMapsInMemory(true);

  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(schema, 0, pool());

  uint64_t totalRows = 0;
  while (rowReader->next(100, batch) > 0) {
    auto* rowVec = batch->as<RowVector>();
    // Trigger lazy loading — this is the pattern that used to crash.
    for (column_index_t i = 0; i < rowVec->childrenSize(); ++i) {
      auto& child = rowVec->childAt(i);
      child = BaseVector::loadedVectorShared(child);
    }
    totalRows += batch->size();
  }
  EXPECT_EQ(totalRows, 5); // 2 rows from stripe 1 + 3 rows from stripe 2.
}

TEST_F(TestReader, readStructWithWholeBatchFiltered) {
  // Test reading a struct with a pushdown filter that filters out all rows
  // for a certain batch.
  auto rowType = ROW({"a"}, {BIGINT()});
  const vector_size_t vectorSize = 20;
  const vector_size_t batchSize = 10;
  std::vector<VectorPtr> children{makeFlatVector<int64_t>(
      vectorSize,
      folly::identity,
      // In the first batch, the parent Rows will all be null.
      [&](auto i) { return i < batchSize; })};

  BufferPtr nulls = AlignedBuffer::allocate<bool>(vectorSize, pool());
  uint64_t* rawNulls = nulls->asMutable<uint64_t>();
  memset(rawNulls, bits::kNotNullByte, nulls->capacity());
  // Mark the Row as null in the first batch.
  for (int i = 0; i < batchSize; ++i) {
    bits::setNull(rawNulls, i, true);
  }

  auto c0 =
      std::make_shared<RowVector>(pool(), rowType, nulls, vectorSize, children);
  auto row = makeRowVector({"c0"}, {c0});

  auto [writer, reader] =
      createWriterReader({row}, pool(), dataIoStats_, metadataIoStats_);

  auto schema = asRowType(row->type());
  auto spec = makeAllFieldsScanSpec(*schema);
  // Create a filter that will filter out all rows in the first batch.
  spec->childByName("c0")->setFilter(std::make_unique<common::IsNotNull>());
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);

  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(schema, 0, pool());

  ASSERT_TRUE(rowReader->next(batchSize, batch));
  // Confirm that all rows were filtered out.
  ASSERT_EQ(batch->size(), 0);
  ASSERT_TRUE(rowReader->next(batchSize, batch));
  // None of the rows should be filtered out in the second batch.
  ASSERT_EQ(batch->size(), 10);
  // Validate that we successfully read the batch.
  auto rowVector = batch->as<RowVector>();
  auto resultRows = rowVector->childAt(0)->loadedVector()->as<RowVector>();
  ASSERT_EQ(resultRows->size(), batchSize);

  for (int i = 0; i < batchSize; ++i) {
    ASSERT_FALSE(resultRows->isNullAt(i));
  }

  auto resultValues =
      resultRows->childAt(0)->loadedVector()->as<FlatVector<int64_t>>();
  ASSERT_EQ(resultValues->size(), batchSize);

  for (int i = 0; i < batchSize; ++i) {
    ASSERT_FALSE(resultValues->isNullAt(i));
    ASSERT_EQ(resultValues->valueAt(i), i + 10);
  }
}

TEST_F(TestReader, readStringDictionaryAsFlat) {
  std::vector<std::string> dictionary;
  for (int i = 0; i < 26; ++i) {
    dictionary.emplace_back(20 + i, 'a' + i);
  }
  auto indices = allocateIndices(200, pool());
  auto* rawIndices = indices->asMutable<vector_size_t>();
  for (int i = 0; i < 200; ++i) {
    rawIndices[i] = i % dictionary.size();
  }
  auto batch = makeRowVector({
      BaseVector::wrapInDictionary(
          nullptr, indices, 200, makeFlatVector(dictionary)),
  });
  auto [writer, reader] = createWriterReader(
      {batch},
      pool(),
      dataIoStats_,
      metadataIoStats_,
      std::make_shared<dwrf::Config>(),
      // The always true flush policy would disable dictionary encoding at least
      // for first batch.
      E2EWriterTestUtil::simpleFlushPolicyFactory(false));
  auto rowType = reader->rowType();
  auto spec = makeAllFieldsScanSpec(*rowType);
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  auto actual = BaseVector::create(rowType, 0, pool());
  ASSERT_EQ(rowReader->next(20, actual), 20);
  ASSERT_EQ(actual->size(), 20);
  auto* c0 = actual->as<RowVector>()->childAt(0)->loadedVector();
  ASSERT_EQ(c0->encoding(), VectorEncoding::Simple::DICTIONARY);
  ASSERT_TRUE(c0->valueVector()->isFlatEncoding());
  ASSERT_EQ(c0->valueVector()->size(), dictionary.size());
  dwio::common::RuntimeStats stats;
  rowReader->updateRuntimeStats(stats);
  const auto metricName =
      std::string(DwrfRuntimeStats::kFlattenStringDictionaryValues);
  ASSERT_FALSE(stats.columnStats.at(1)
                   .at(FileFormat::DWRF)
                   .columnMetrics.contains(metricName));
  spec->childByName("c0")->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"aaaaaaaaaaaaaaaaaaaa"}, false));
  spec->resetCachedValues(true);
  rowReader = reader->createRowReader(rowReaderOpts);
  ASSERT_EQ(rowReader->next(20, actual), 20);
  ASSERT_EQ(actual->size(), 1);
  ASSERT_TRUE(actual->as<RowVector>()->childAt(0)->isFlatEncoding());
  stats = dwio::common::RuntimeStats();
  rowReader->updateRuntimeStats(stats);
  ASSERT_TRUE(stats.columnStats.at(1)
                  .at(FileFormat::DWRF)
                  .columnMetrics.contains(metricName));
  ASSERT_EQ(
      stats.columnStats.at(1)
          .at(FileFormat::DWRF)
          .columnMetrics.at(metricName)
          .sum,
      1);
}

// A primitive subfield is missing in file, and result is not reused.
TEST_F(TestReader, missingSubfieldsNoResultReusing) {
  constexpr int kSize = 10;
  auto batch = makeRowVector({
      makeRowVector({
          makeFlatVector<int64_t>(kSize, folly::identity),
      }),
  });
  auto [writer, reader] =
      createWriterReader({batch}, pool(), dataIoStats_, metadataIoStats_);
  auto schema = ROW({{"c0", ROW({{"c0", BIGINT()}, {"c1", VARCHAR()}})}});
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*schema));
  auto rowReader = reader->createRowReader(rowReaderOpts);
  auto actual = BaseVector::create(schema, 0, pool());
  // Hold a second reference to result so it cannot be reused.
  auto actual2 = actual;
  ASSERT_EQ(rowReader->next(1024, actual), 10);
  auto expected = makeRowVector({
      makeRowVector({
          makeFlatVector<int64_t>(kSize, folly::identity),
          BaseVector::createNullConstant(VARCHAR(), kSize, pool()),
      }),
  });
  assertEqualVectors(expected, actual);
}

// Ensure there is enough data before switching to fast path.
TEST_F(TestReader, selectiveStringDirectFastPath) {
  auto genStr = [](auto i) {
    return i == 0 ? "x" : i < 8 ? "" : "xxxxxxxxxxx";
  };
  auto batch = makeRowVector({
      makeFlatVector<int64_t>(17, [](auto i) { return i != 8; }),
      makeFlatVector<StringView>(17, genStr),
  });
  auto [writer, reader] =
      createWriterReader({batch}, pool(), dataIoStats_, metadataIoStats_);
  auto schema = asRowType(batch->type());
  auto spec = makeAllFieldsScanSpec(*schema);
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  spec->childByName("c0")->setFilter(common::createBigintValues({1}, false));
  auto rowReader = reader->createRowReader(rowReaderOpts);
  auto actual = BaseVector::create(schema, 0, pool());
  ASSERT_EQ(rowReader->next(1024, actual), batch->size());
  auto expected = makeRowVector({
      makeConstant<int64_t>(1, 16),
      makeFlatVector<StringView>(16, genStr),
  });
  assertEqualVectors(expected, actual);
}

TEST_F(TestReader, selectiveStringDirect) {
  auto genStr = [](auto i) {
    static const std::string s(2048, 'x');
    return i == 0 || i == 8 ? s.c_str() : "";
  };
  auto batch = makeRowVector({
      makeFlatVector<int64_t>(17, [](auto i) { return i != 15; }),
      makeFlatVector<StringView>(17, genStr),
  });
  auto [writer, reader] =
      createWriterReader({batch}, pool(), dataIoStats_, metadataIoStats_);
  auto schema = asRowType(batch->type());
  auto spec = makeAllFieldsScanSpec(*schema);
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  spec->childByName("c0")->setFilter(common::createBigintValues({1}, false));
  auto rowReader = reader->createRowReader(rowReaderOpts);
  auto actual = BaseVector::create(schema, 0, pool());
  ASSERT_EQ(rowReader->next(1024, actual), batch->size());
  auto expected = makeRowVector({
      makeConstant<int64_t>(1, 16),
      makeFlatVector<StringView>(16, genStr),
  });
  assertEqualVectors(expected, actual);
}

TEST_F(TestReader, selectiveFlatMapFastPathAllInlinedStringKeys) {
  auto maps = makeMapVector<std::string, int64_t>(
      {{{"a", 0}, {"b", 0}}, {{"a", 1}, {"b", 1}}});
  auto row = makeRowVector({"c0"}, {maps});
  auto config = std::make_shared<dwrf::Config>();
  config->set(dwrf::Config::FLATTEN_MAP, true);
  config->set(dwrf::Config::MAP_FLAT_COLS, {0});
  auto [writer, reader] =
      createWriterReader({row}, pool(), dataIoStats_, metadataIoStats_, config);
  auto schema = asRowType(row->type());
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*schema));
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(schema, 0, pool());
  ASSERT_EQ(rowReader->next(10, batch), 2);
  assertEqualVectors(batch, row);
}

TEST_F(TestReader, skipLongString) {
  // c0 in long_string.dwrf has 25 rows of 200,000,000 character long strings,
  // whose values are repeated 'a' to 'y' respectively.
  auto input = std::make_unique<BufferedInput>(
      std::make_shared<LocalReadFile>(getExampleFilePath("long_string.dwrf")),
      *pool());
  dwio::common::ReaderOptions readerOpts(pool());
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  readerOpts.setFileFormat(FileFormat::DWRF);
  auto reader = DwrfReader::create(std::move(input), readerOpts);
  auto spec = std::make_shared<common::ScanSpec>("<root>");
  spec->addField("c0", 0);
  spec->getOrCreateChild("c1")->setFilter(
      std::make_unique<common::BoolValue>(true, false));
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  VectorPtr batch = BaseVector::create(ROW({"c0"}, {VARCHAR()}), 0, pool());
  auto validate = [](const VectorPtr& batch) {
    ASSERT_EQ(batch->size(), 1);
    auto string = batch->asChecked<RowVector>()
                      ->childAt(0)
                      ->loadedVector()
                      ->asChecked<SimpleVector<StringView>>()
                      ->valueAt(0);
    ASSERT_EQ(string.size(), 200'000'000);
    for (char c : string) {
      ASSERT_EQ(c, 'y');
    }
  };
  {
    SCOPED_TRACE("Skip");
    auto rowReader = reader->createRowReader(rowReaderOpts);
    ASSERT_EQ(rowReader->next(24, batch), 24);
    ASSERT_EQ(batch->size(), 0);
    ASSERT_EQ(rowReader->next(2, batch), 1);
    validate(batch);
  }
  {
    SCOPED_TRACE("Filter");
    auto rowReader = reader->createRowReader(rowReaderOpts);
    ASSERT_EQ(rowReader->next(26, batch), 25);
    validate(batch);
  }
}

TEST_F(TestReader, mapAsStruct) {
  auto row = makeRowVector({
      makeMapVector<int32_t, int64_t>({{{1, 4}, {2, 5}}, {{1, 6}, {3, 7}}}),
  });
  auto [writer, reader] =
      createWriterReader({row}, pool(), dataIoStats_, metadataIoStats_);
  auto outType = ROW({"c0"}, {ROW({"3", "1"}, BIGINT())});
  auto spec = makeAllFieldsScanSpec(*outType);
  spec->childByName("c0")->setFlatMapAsStruct(true);
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(rowReader->next(10, batch), 2);
  auto expected = makeRowVector({
      makeRowVector(
          {"3", "1"},
          {
              makeNullableFlatVector<int64_t>({std::nullopt, 7}),
              makeFlatVector<int64_t>({4, 6}),
          }),
  });
  assertEqualVectors(expected, batch);
}

TEST_F(TestReader, mapAsStructFilterAfterRead) {
  auto row = makeRowVector({
      makeMapVector<int32_t, int64_t>({{{1, 4}, {2, 5}}, {}, {{1, 6}, {3, 7}}}),
      makeRowVector(
          {makeConstant<int64_t>(0, 3)}, [](auto i) { return i == 0; }),
  });
  auto [writer, reader] =
      createWriterReader({row}, pool(), dataIoStats_, metadataIoStats_);
  auto outType =
      ROW({"c0", "c1"}, {ROW({"3", "1"}, BIGINT()), ROW({"c0"}, BIGINT())});
  auto spec = makeAllFieldsScanSpec(*outType);
  auto* c0Spec = spec->childByName("c0");
  c0Spec->setFlatMapAsStruct(true);
  c0Spec->setFilter(std::make_shared<common::IsNotNull>());
  spec->childByName("c1")->setFilter(std::make_shared<common::IsNotNull>());
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(rowReader->next(10, batch), 3);
  auto expected = makeRowVector({
      makeRowVector(
          {"3", "1"},
          {
              makeNullableFlatVector<int64_t>({std::nullopt, 7}),
              makeNullableFlatVector<int64_t>({std::nullopt, 6}),
          }),
      makeRowVector({makeConstant<int64_t>(0, 2)}),
  });
  assertEqualVectors(expected, batch);
}

TEST_F(TestReader, mapAsStructNullChildrenWithReuse) {
  // Reproduces the SIGSEGV fixed by initializing null children in
  // SelectiveMapAsStructColumnReader::getValues. A filter on the
  // flat-map-as-struct column routes the parent struct reader to call
  // getValues() directly. The result is not singly referenced (a second
  // reference is held, as a real downstream consumer would), so prepareResult()
  // reallocates it via fillRowVectorChildren(), which leaves the
  // flat-map-as-struct's non-ROW children as nullptr. getValues() must
  // initialize those children rather than dereference them.
  //
  // This is identical to mapAsStructFilterAfterRead except for the held
  // reference that forces the reallocation path.
  auto row = makeRowVector({
      makeMapVector<int32_t, int64_t>({{{1, 4}, {2, 5}}, {}, {{1, 6}, {3, 7}}}),
      makeRowVector(
          {makeConstant<int64_t>(0, 3)}, [](auto i) { return i == 0; }),
  });
  auto [writer, reader] =
      createWriterReader({row}, pool(), dataIoStats_, metadataIoStats_);
  auto outType =
      ROW({"c0", "c1"}, {ROW({"3", "1"}, BIGINT()), ROW({"c0"}, BIGINT())});
  auto spec = makeAllFieldsScanSpec(*outType);
  auto* c0Spec = spec->childByName("c0");
  c0Spec->setFlatMapAsStruct(true);
  c0Spec->setFilter(std::make_shared<common::IsNotNull>());
  spec->childByName("c1")->setFilter(std::make_shared<common::IsNotNull>());
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  // Hold a second reference so the result is not singly referenced; this forces
  // prepareResult() down the fillRowVectorChildren() path, producing a
  // flat-map-as-struct child whose non-ROW children are null.
  VectorPtr holder = batch;
  ASSERT_EQ(rowReader->next(10, batch), 3);
  auto expected = makeRowVector({
      makeRowVector(
          {"3", "1"},
          {
              makeNullableFlatVector<int64_t>({std::nullopt, 7}),
              makeNullableFlatVector<int64_t>({std::nullopt, 6}),
          }),
      makeRowVector({makeConstant<int64_t>(0, 2)}),
  });
  assertEqualVectors(expected, batch);
}

TEST_F(TestReader, mapAsStructAllEmpty) {
  auto row = makeRowVector({makeMapVector<int32_t, int64_t>({{}, {}})});
  auto [writer, reader] =
      createWriterReader({row}, pool(), dataIoStats_, metadataIoStats_);
  auto outType = ROW({"c0"}, {ROW({"1"}, BIGINT())});
  auto spec = makeAllFieldsScanSpec(*outType);
  spec->childByName("c0")->setFlatMapAsStruct(true);
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(rowReader->next(10, batch), 2);
  auto expected = makeRowVector({
      makeRowVector({"1"}, {makeNullConstant(TypeKind::BIGINT, 2)}),
  });
  assertEqualVectors(expected, batch);
}

// A map-as-struct read that projects only a small subset of the keys present in
// the (regular) map on disk. The reader pushes an IN filter over the projected
// keys onto the map key sub-reader so the element reader never decodes values
// for unprojected keys. The output must be identical to decoding the whole map
// and dropping unprojected keys afterward -- including rows that are missing a
// projected key, rows whose keys are all unprojected, empty maps, and null
// maps.
TEST_F(TestReader, mapAsStructKeySubsetExtraction) {
  auto row = makeRowVector({
      makeMapVectorFromJson<int32_t, int64_t>({
          "{1: 10, 2: 20, 3: 30, 4: 40, 5: 50}", // all keys present
          "{2: 21, 4: 41}", // no projected key present
          "{1: 12, 3: 32}", // both projected keys present
          "{}", // empty map
          "null", // null map
      }),
  });
  auto [writer, reader] =
      createWriterReader({row}, pool(), dataIoStats_, metadataIoStats_);
  // Project only keys "3" and "1"; keys 2, 4, 5 are unprojected and must be
  // filtered out of the element decode.
  auto outType = ROW({"c0"}, {ROW({"3", "1"}, BIGINT())});
  auto spec = makeAllFieldsScanSpec(*outType);
  spec->childByName("c0")->setFlatMapAsStruct(true);
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  VectorPtr batch = BaseVector::create(outType, 0, pool());
  ASSERT_EQ(rowReader->next(10, batch), 5);
  // Rows missing a projected key and empty maps yield a non-null struct with
  // null children; a null map yields a null struct row (setComplexNulls).
  auto expected = makeRowVector({
      makeRowVector(
          {"3", "1"},
          {
              makeNullableFlatVector<int64_t>(
                  {30, std::nullopt, 32, std::nullopt, std::nullopt}),
              makeNullableFlatVector<int64_t>(
                  {10, std::nullopt, 12, std::nullopt, std::nullopt}),
          },
          /*isNullAt=*/[](vector_size_t row) { return row == 4; }),
  });
  assertEqualVectors(expected, batch);
}

// Verify DwrfRowReader can be destroyed while ParallelUnitLoader async load()
// are in progress. This regression test ensures that:
// 1. ParallelUnitLoader destructor doesn't wait for async load() operations
// 2. Async load() from DwrfUnit can still function after ParallelUnitLoader
// destruction and DwrfRowReader destruction, which means all dependencies in
// DwrfUnit remain valid (eg ReaderBase)
//
// If a future change adds an unsafe raw pointer to DwrfUnit's dependencies that
// would be freed by ParallelUnitLoader or DwrfRowReader's destruction, this
// test may crash due to use-after-free.
DEBUG_ONLY_TEST_F(TestReader, asyncLoadSurvivesReaderDestruction) {
  const int kNumStripes = 2;
  const int kRowsPerStripe = 100;
  std::vector<VectorPtr> batches;
  batches.reserve(kNumStripes);
  for (int stripe = 0; stripe < kNumStripes; ++stripe) {
    batches.push_back(makeRowVector({
        makeFlatVector<int64_t>(
            kRowsPerStripe,
            [stripe](auto row) { return stripe * kRowsPerStripe + row; }),
    }));
  }

  // Write the DWRF file - force each batch into its own stripe
  auto config = std::make_shared<dwrf::Config>();

  auto sink =
      std::make_unique<MemorySink>(1 << 20, FileSink::Options{.pool = pool()});
  auto* sinkPtr = sink.get();
  auto writer = E2EWriterTestUtil::writeData(
      std::move(sink),
      asRowType(batches[0]->type()),
      batches,
      config,
      // Force flush after each batch to create separate stripes
      E2EWriterTestUtil::simpleFlushPolicyFactory(true));

  std::string data(sinkPtr->data(), sinkPtr->size());
  auto input = std::make_unique<BufferedInput>(
      std::make_shared<InMemoryReadFile>(std::move(data)), *pool());

  std::atomic<int> asyncLoadsStarted{0};
  std::atomic<int> asyncLoadsCompleted{0};
  folly::Baton<> readerDestroyed;

  SCOPED_TESTVALUE_SET(
      "facebook::velox::dwio::common::ParallelUnitLoader::load",
      std::function<void(void*)>([&](void*) {
        // Only block the second stripe (index 1) - let the first stripe load
        // normally so rowReader->next() can complete
        // fetch_add returns the value before increment: 0 for first, 1 for
        // second, etc.
        if (asyncLoadsStarted.fetch_add(1) == 1) {
          // Block here until reader is destroyed
          readerDestroyed.wait();
        }
        asyncLoadsCompleted.fetch_add(1);
      }));

  auto ioExecutor = std::make_shared<folly::IOThreadPoolExecutor>(2);

  // Make sure ReaderOptions and DwrfRowReader are freed after {} scope
  {
    dwio::common::ReaderOptions readerOpts(pool());
    readerOpts.setDataIoStats(dataIoStats_);
    readerOpts.setMetadataIoStats(metadataIoStats_);
    readerOpts.setFileFormat(FileFormat::DWRF);
    auto reader = DwrfReader::create(std::move(input), readerOpts);

    // Enable parallel unit load
    RowReaderOptions rowReaderOpts;
    rowReaderOpts.setParallelUnitLoadCount(2);
    rowReaderOpts.setIOExecutor(ioExecutor.get());
    rowReaderOpts.setScanSpec(makeAllFieldsScanSpec(*reader->rowType()));
    auto rowReader = reader->createRowReader(rowReaderOpts);

    VectorPtr batch = BaseVector::create(reader->rowType(), 0, pool());
    rowReader->next(50, batch); // Read first stripe

    auto start = std::chrono::steady_clock::now();
    rowReader.reset();
    auto duration = std::chrono::steady_clock::now() - start;
    // Verify destruction was fast (didn't wait for async operations)
    EXPECT_LT(duration, std::chrono::seconds(1))
        << "Destruction should not wait for async loads";
  }

  // Now signal that reader is destroyed
  readerDestroyed.post();

  // Wait for async loads to complete
  int maxWaitMs = 2000;
  int waitedMs = 0;
  while (asyncLoadsCompleted.load() < 2 && waitedMs < maxWaitMs) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    waitedMs += 100;
  }

  // Verify that both async loads completed successfully after reader
  // destruction This proves the fix works: async operations can complete even
  // after DwrfRowReader is destroyed because LoadUnit is captured as shared_ptr
  EXPECT_EQ(asyncLoadsCompleted.load(), 2)
      << "Both async loads should complete even after reader destruction. "
      << "If this fails, it means async operations are being cancelled or "
      << "crashing after DwrfRowReader destruction, indicating unsafe pointers.";

  // Clean up
  ioExecutor->join();
}

TEST_F(TestReader, extractionTransformMapKeys) {
  // Write a MAP(VARCHAR, BIGINT) column, read with a ScanSpec transform
  // that applies MapKeys extraction at the reader level.
  auto keys = makeFlatVector<StringView>({"a", "b", "c", "d"});
  auto values = makeFlatVector<int64_t>({1, 2, 3, 4});
  auto mapVector = makeMapVector({0, 2}, keys, values);
  auto data = makeRowVector({"col"}, {mapVector});
  auto schema = asRowType(data->type());
  auto [writer, reader] =
      createWriterReader({data}, pool(), dataIoStats_, metadataIoStats_);

  auto spec = makeAllFieldsScanSpec(*schema);

  spec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kKeys);

  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);

  auto result = BaseVector::create(schema, 0, pool());
  ASSERT_EQ(rowReader->next(10, result), 2);
  auto* row = result->as<RowVector>();
  auto* resultArray = row->childAt(0)->loadedVector()->as<ArrayVector>();
  ASSERT_EQ(resultArray->size(), 2);
  ASSERT_EQ(resultArray->sizeAt(0), 2);
  ASSERT_EQ(resultArray->sizeAt(1), 2);
}

TEST_F(TestReader, extractionTransformAfterScanSpecReorder) {
  constexpr vector_size_t kNumRows = 12;
  auto maps = makeMapVector<int64_t, int64_t>(
      kNumRows,
      [](auto row) { return row % 3; },
      [](auto index) { return index; },
      [](auto index) { return index * 10; },
      [](auto row) { return row % 5 == 0; });
  auto data = makeRowVector(
      {"constant", "maps", "plain", "id"},
      {makeFlatVector<int64_t>(kNumRows, folly::identity),
       maps,
       makeFlatVector<int64_t>(kNumRows, [](auto row) { return 100 + row; }),
       makeFlatVector<int64_t>(kNumRows, folly::identity)});
  auto [writer, reader] =
      createWriterReader({data}, pool(), dataIoStats_, metadataIoStats_);

  auto spec = makeAllFieldsScanSpec(*asRowType(data->type()));
  spec->childByName("constant")
      ->setConstantValue(BaseVector::createNullConstant(BIGINT(), 1, pool()));
  spec->childByName("id")->setFilter(
      std::make_unique<common::BigintRange>(1, 10, false));

  using connector::hive::applyExtractionChain;
  using connector::hive::ExtractionPathElement;
  using connector::hive::ExtractionPathElementPtr;
  using connector::hive::ExtractionStep;
  const std::vector<ExtractionPathElementPtr> keysChain = {
      ExtractionPathElement::simple(ExtractionStep::kMapKeys)};
  const std::vector<ExtractionPathElementPtr> sizeChain = {
      ExtractionPathElement::simple(ExtractionStep::kSize)};
  const auto outputType = ROW({"keys", "size"}, {ARRAY(BIGINT()), BIGINT()});
  auto* mapsSpec = spec->childByName("maps");
  int transformCalls = 0;
  mapsSpec->setTransform(
      [&](const VectorPtr& input, memory::MemoryPool* memoryPool) -> VectorPtr {
        ++transformCalls;
        return std::make_shared<RowVector>(
            memoryPool,
            outputType,
            nullptr,
            input->size(),
            std::vector<VectorPtr>{
                applyExtractionChain(input, keysChain, memoryPool),
                applyExtractionChain(input, sizeChain, memoryPool)});
      },
      outputType);

  RowReaderOptions options;
  options.setScanSpec(spec);
  auto rowReader = reader->createRowReader(options);
  auto result = BaseVector::create(data->type(), 0, pool());
  DecodedVector expectedKeys(*maps->mapKeys());
  int batches = 0;
  int totalRows = 0;
  while (rowReader->next(3, result)) {
    auto* row = result->as<RowVector>();
    // Constants have no reader, and filtering reorders the ScanSpecs. Neither
    // output channels nor ScanSpec positions are reader indices.
    ASSERT_NE(mapsSpec->subscript(), mapsSpec->channel());
    ASSERT_NE(spec->children().at(mapsSpec->subscript()).get(), mapsSpec);
    auto& mapResult = row->childAt(1);
    ASSERT_TRUE(mapResult->isLazy());
    ASSERT_TRUE(mapResult->type()->equivalent(*outputType));
    ASSERT_FALSE(mapResult->as<LazyVector>()->supportsHook());
    ASSERT_EQ(transformCalls, batches);
    auto* extracted = mapResult->loadedVector()->as<RowVector>();
    ASSERT_NE(extracted, nullptr);
    auto* keys = extracted->childAt(0)->as<ArrayVector>();
    ASSERT_NE(keys, nullptr);
    DecodedVector sizes(*extracted->childAt(1));
    DecodedVector actualKeys(*keys->elements());
    DecodedVector ids(*row->childAt(3));
    ASSERT_TRUE(row->childAt(2)->isLazy());
    ASSERT_TRUE(row->childAt(2)->as<LazyVector>()->supportsHook());
    DecodedVector plain(*row->childAt(2));
    for (vector_size_t i = 0; i < row->size(); ++i) {
      const auto sourceRow = ids.valueAt<int64_t>(i);
      EXPECT_TRUE(row->childAt(0)->isNullAt(i));
      EXPECT_EQ(plain.valueAt<int64_t>(i), 100 + sourceRow);
      EXPECT_EQ(keys->isNullAt(i), maps->isNullAt(sourceRow));
      EXPECT_EQ(sizes.isNullAt(i), maps->isNullAt(sourceRow));
      if (maps->isNullAt(sourceRow)) {
        continue;
      }
      EXPECT_EQ(sizes.valueAt<int64_t>(i), maps->sizeAt(sourceRow));
      ASSERT_EQ(keys->sizeAt(i), maps->sizeAt(sourceRow));
      for (vector_size_t j = 0; j < keys->sizeAt(i); ++j) {
        EXPECT_EQ(
            actualKeys.valueAt<int64_t>(keys->offsetAt(i) + j),
            expectedKeys.valueAt<int64_t>(maps->offsetAt(sourceRow) + j));
      }
    }
    totalRows += row->size();
    ++batches;
    EXPECT_EQ(transformCalls, batches);
    mapResult->loadedVector();
    EXPECT_EQ(transformCalls, batches);
  }
  EXPECT_EQ(totalRows, 10);
  EXPECT_EQ(batches, 4);
}

TEST_F(TestReader, extractionTransformSize) {
  // Write a MAP(VARCHAR, BIGINT) column, read with a Size extraction.
  auto keys = makeFlatVector<StringView>({"a", "b", "c"});
  auto values = makeFlatVector<int64_t>({1, 2, 3});
  auto mapVector = makeMapVector({0, 1}, keys, values);
  auto data = makeRowVector({"col"}, {mapVector});
  auto schema = asRowType(data->type());
  auto [writer, reader] =
      createWriterReader({data}, pool(), dataIoStats_, metadataIoStats_);

  auto spec = makeAllFieldsScanSpec(*schema);

  spec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kSize);

  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);

  auto result = BaseVector::create(schema, 0, pool());
  ASSERT_EQ(rowReader->next(10, result), 2);
  auto* row = result->as<RowVector>();
  auto* sizes = row->childAt(0)->loadedVector()->as<FlatVector<int64_t>>();
  ASSERT_EQ(sizes->size(), 2);
  ASSERT_EQ(sizes->valueAt(0), 1);
  ASSERT_EQ(sizes->valueAt(1), 2);
}

TEST_F(TestReader, extractionMapKeySizeWithSeek) {
  // Repro for the kSize+MAP+no-deltaUpdate skip() bug.  When kSize is
  // configured on a MAP column without a delta update, neither keyReader_
  // nor elementReader_ is created.  Forward seekToRow() after a prior read
  // triggers SelectiveMapColumnReaderBase::skip() which currently fails
  // because it requires at least one child reader to read the length stream.
  constexpr int kNumRows = 100;
  std::vector<std::string> keyStrs(kNumRows * 2);
  for (int i = 0; i < kNumRows * 2; ++i) {
    keyStrs[i] = "k_" + std::to_string(i);
  }
  auto keys = makeFlatVector<StringView>(
      kNumRows * 2, [&](auto i) { return StringView(keyStrs[i]); });
  auto values = makeFlatVector<int64_t>(kNumRows * 2, folly::identity);
  std::vector<vector_size_t> offsets(kNumRows);
  for (int i = 0; i < kNumRows; ++i) {
    offsets[i] = i * 2;
  }
  auto mapVector = makeMapVector(offsets, keys, values);
  auto data = makeRowVector({"col"}, {mapVector});
  auto schema = asRowType(data->type());
  auto [writer, reader] =
      createWriterReader({data}, pool(), dataIoStats_, metadataIoStats_);

  auto spec = makeAllFieldsScanSpec(*schema);
  spec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kSize);

  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);

  // Read & materialize the first 20 rows.
  auto result = BaseVector::create(schema, 0, pool());
  ASSERT_EQ(rowReader->next(20, result), 20);
  result->as<RowVector>()->childAt(0)->loadedVector();

  // seekToRow advances the map reader from offset 20 to 50, calling skip()
  // for the 30-row gap.  Without the fix, this VELOX_FAILs at
  // SelectiveMapColumnReaderBase::skip with "repeated reader with no
  // children".
  ASSERT_NO_THROW(dynamic_cast<DwrfRowReader*>(rowReader.get())->seekToRow(50));
}

TEST_F(TestReader, extractionTransformMapValuesStructField) {
  // MAP(VARCHAR, ROW(x: INT, y: INT)) with chain
  // [MapValues, ArrayElements, StructField("x")] -> ARRAY(INT).
  // The reader handles this natively via kValues on the map and kField on
  // the values struct, so no post-read transform is needed.
  auto keys = makeFlatVector<StringView>({"a", "b", "c"});
  auto structValues = makeRowVector(
      {"x", "y"},
      {makeFlatVector<int32_t>({10, 20, 30}),
       makeFlatVector<int32_t>({100, 200, 300})});
  auto mapVector = makeMapVector({0, 2}, keys, structValues);
  auto data = makeRowVector({"col"}, {mapVector});
  auto schema = asRowType(data->type());
  auto [writer, reader] =
      createWriterReader({data}, pool(), dataIoStats_, metadataIoStats_);

  auto spec = makeAllFieldsScanSpec(*schema);

  using connector::hive::applyExtractionChain;
  using connector::hive::configureExtractionScanSpec;
  using connector::hive::ExtractionPathElement;
  using connector::hive::ExtractionPathElementPtr;
  using connector::hive::ExtractionStep;
  using connector::hive::NamedExtraction;

  // Configure extraction on the "col" spec with the sub-chain starting
  // from the MAP type: [MapValues, ArrayElements, StructField("x")].
  auto mapType = schema->childAt(0);
  std::vector<NamedExtraction> colExtractions = {
      {"out",
       {ExtractionPathElement::simple(ExtractionStep::kMapValues),
        ExtractionPathElement::simple(ExtractionStep::kArrayElements),
        ExtractionPathElement::structField("x")},
       INTEGER()}};
  auto* colSpec = spec->childByName("col");
  configureExtractionScanSpec(mapType, colExtractions, *colSpec, pool());

  // Set a full-chain transform for delta update fallback.
  auto fullChain = colExtractions[0].chain;
  colSpec->setTransform(
      [fullChain](const VectorPtr& input, memory::MemoryPool* p) -> VectorPtr {
        return applyExtractionChain(input, fullChain, p);
      },
      ARRAY(INTEGER()));

  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);

  auto result = BaseVector::create(schema, 0, pool());
  ASSERT_EQ(rowReader->next(10, result), 2);
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

TEST_F(TestReader, extractionSizeResultVectorReuse) {
  // Verify the FlatVector<int64_t> result is reused across batches for
  // Size extraction (no per-batch allocation).
  // Write multiple batches to produce multiple reads.
  constexpr int kNumRows = 200;
  std::vector<std::string> keyStrs(kNumRows * 2);
  for (int i = 0; i < kNumRows * 2; ++i) {
    keyStrs[i] = std::to_string(i);
  }
  auto keys = makeFlatVector<StringView>(
      kNumRows * 2, [&](auto i) { return StringView(keyStrs[i]); });
  auto values = makeFlatVector<int64_t>(kNumRows * 2, folly::identity);
  // Each row has 2 map entries.
  std::vector<vector_size_t> offsets(kNumRows);
  for (int i = 0; i < kNumRows; ++i) {
    offsets[i] = i * 2;
  }
  auto mapVector = makeMapVector(offsets, keys, values);
  auto data = makeRowVector({"col"}, {mapVector});
  auto schema = asRowType(data->type());
  auto [writer, reader] =
      createWriterReader({data}, pool(), dataIoStats_, metadataIoStats_);

  auto spec = makeAllFieldsScanSpec(*schema);
  spec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kSize);

  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);

  auto result = BaseVector::create(schema, 0, pool());

  // Read first batch.
  ASSERT_GT(rowReader->next(50, result), 0);
  auto* row = result->as<RowVector>();
  auto* child = row->childAt(0)->loadedVector();
  ASSERT_TRUE(child->type()->isBigint());
  auto* firstBatchPtr = child;

  // Read second batch — the FlatVector should be the same object.
  ASSERT_GT(rowReader->next(50, result), 0);
  row = result->as<RowVector>();
  child = row->childAt(0)->loadedVector();
  ASSERT_EQ(child, firstBatchPtr)
      << "FlatVector result should be reused across batches for Size extraction";

  // Verify values are correct (each map has 2 entries).
  auto* sizes = child->as<FlatVector<int64_t>>();
  for (int i = 0; i < sizes->size(); ++i) {
    ASSERT_EQ(sizes->valueAt(i), 2);
  }
}

TEST_F(TestReader, extractionMapKeysMultipleBatches) {
  // Verify MapKeys extraction produces correct results across multiple
  // batches.
  constexpr int kNumRows = 200;
  std::vector<std::string> keyStrs(kNumRows * 3);
  for (int i = 0; i < kNumRows * 3; ++i) {
    keyStrs[i] = std::to_string(i);
  }
  auto keys = makeFlatVector<StringView>(
      kNumRows * 3, [&](auto i) { return StringView(keyStrs[i]); });
  auto values = makeFlatVector<int64_t>(kNumRows * 3, folly::identity);
  std::vector<vector_size_t> offsets(kNumRows);
  for (int i = 0; i < kNumRows; ++i) {
    offsets[i] = i * 3;
  }
  auto mapVector = makeMapVector(offsets, keys, values);
  auto data = makeRowVector({"col"}, {mapVector});
  auto schema = asRowType(data->type());
  auto [writer, reader] =
      createWriterReader({data}, pool(), dataIoStats_, metadataIoStats_);

  auto spec = makeAllFieldsScanSpec(*schema);
  spec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kKeys);

  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);

  auto result = BaseVector::create(schema, 0, pool());

  // Read first batch.
  ASSERT_GT(rowReader->next(50, result), 0);
  auto* row = result->as<RowVector>();
  auto* child = row->childAt(0)->loadedVector();
  ASSERT_TRUE(child->type()->isArray());
  auto* arr = child->as<ArrayVector>();
  for (int i = 0; i < arr->size(); ++i) {
    ASSERT_EQ(arr->sizeAt(i), 3);
  }

  // Read second batch — verify correctness.
  ASSERT_GT(rowReader->next(50, result), 0);
  row = result->as<RowVector>();
  child = row->childAt(0)->loadedVector();
  arr = child->as<ArrayVector>();
  for (int i = 0; i < arr->size(); ++i) {
    ASSERT_EQ(arr->sizeAt(i), 3);
  }
}

TEST_F(TestReader, extractionMapKeysIoReduction) {
  // Verify that MapKeys extraction produces correct results on a large
  // dataset.  The extraction pushdown skips decoding the value stream
  // (seekTo instead of readWithTiming) — this saves CPU but DWRF coalesced
  // I/O reads the full stripe from storage regardless.  Decode-level savings
  // are validated by the extractionTransformMapKeys test which verifies the
  // reader's seekTo behavior.
  constexpr int kNumRows = 10'000;
  std::vector<std::string> keyStrs(kNumRows * 2);
  for (int i = 0; i < kNumRows * 2; ++i) {
    keyStrs[i] = "key_" + std::to_string(i);
  }
  auto keys = makeFlatVector<StringView>(
      kNumRows * 2, [&](auto i) { return StringView(keyStrs[i]); });
  auto values = makeFlatVector<int64_t>(kNumRows * 2, folly::identity);
  std::vector<vector_size_t> offsets(kNumRows);
  for (int i = 0; i < kNumRows; ++i) {
    offsets[i] = i * 2;
  }
  auto mapVector = makeMapVector(offsets, keys, values);
  auto data = makeRowVector({"col"}, {mapVector});
  auto schema = asRowType(data->type());
  auto [writer, reader] =
      createWriterReader({data}, pool(), dataIoStats_, metadataIoStats_);

  auto spec = makeAllFieldsScanSpec(*schema);
  spec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kKeys);

  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(spec);
  auto rowReader = reader->createRowReader(rowReaderOpts);

  auto result = BaseVector::create(schema, 0, pool());
  uint64_t totalRows = 0;
  while (auto batch = rowReader->next(1'000, result)) {
    totalRows += batch;
    auto* row = result->as<RowVector>();
    auto* arr = row->childAt(0)->loadedVector()->as<ArrayVector>();
    ASSERT_EQ(arr->size(), batch);
    for (vector_size_t i = 0; i < arr->size(); ++i) {
      ASSERT_EQ(arr->sizeAt(i), 2) << "Row " << i << " should have 2 keys";
    }
  }
  ASSERT_EQ(totalRows, kNumRows);

  // Validate IO reduction: MapKeys extraction should read fewer bytes than
  // a full scan because the values stream is not requested.
  auto sink =
      std::make_unique<MemorySink>(1 << 20, FileSink::Options{.pool = pool()});
  auto* sinkPtr = sink.get();
  auto writerObj = E2EWriterTestUtil::writeData(
      std::move(sink),
      schema,
      {data},
      std::make_shared<dwrf::Config>(),
      E2EWriterTestUtil::simpleFlushPolicyFactory(true));
  std::string fileData(sinkPtr->data(), sinkPtr->size());

  auto runWithSpec =
      [&](const std::shared_ptr<common::ScanSpec>& scanSpec) -> uint64_t {
    auto ioStats = std::make_shared<io::IoStatistics>();
    // Disable coalescing (maxMergeDistance=0) so each stream is read
    // separately and rawBytesRead reflects actual stream-level I/O.
    // Disable file preloading so small files don't bypass per-stream IO.
    auto input = std::make_unique<BufferedInput>(
        std::make_shared<InMemoryReadFile>(fileData),
        *pool(),
        MetricsLog::voidLog(),
        ioStats.get(),
        /*ioStats=*/nullptr,
        /*maxMergeDistance=*/0);
    dwio::common::ReaderOptions readerOpts(pool());
    readerOpts.setDataIoStats(ioStats);
    readerOpts.setMetadataIoStats(ioStats);
    readerOpts.setFileFormat(FileFormat::DWRF);
    readerOpts.setFilePreloadThreshold(0);
    auto rdr = DwrfReader::create(std::move(input), readerOpts);
    RowReaderOptions rro;
    rro.setScanSpec(scanSpec);
    auto rr = rdr->createRowReader(rro);
    auto res = BaseVector::create(schema, 0, pool());
    while (rr->next(kNumRows, res) > 0) {
    }
    return ioStats->rawBytesRead();
  };

  auto fullSpec = makeAllFieldsScanSpec(*schema);
  auto fullBytes = runWithSpec(fullSpec);

  auto extSpec = makeAllFieldsScanSpec(*schema);
  extSpec->childByName("col")->setExtractionType(
      common::ScanSpec::ExtractionType::kKeys);
  auto extBytes = runWithSpec(extSpec);

  ASSERT_GT(fullBytes, 0);
  ASSERT_LT(extBytes, fullBytes)
      << "Extraction: " << extBytes << ", Full: " << fullBytes;
}

TEST_F(TestReader, extractionNestedChainScanSpec) {
  // Test that a nested extraction chain recursively configures ALL levels
  // of the ScanSpec, so no post-read transform is needed.
  //
  // Schema: ROW(a: MAP(VARCHAR, ROW(x: INT, y: ARRAY(BIGINT))), b: INT)
  // Chain: [StructField("a"), MapValues, ArrayElements, StructField("y"), Size]
  //
  // Expected ScanSpec at each level:
  //   ROOT ROW: "b" pruned (constant null), "a" not pruned
  //   "a" MAP: ExtractionType::kValues
  //   values ROW: "x" pruned (constant null), "y" not pruned
  //   "y" ARRAY: ExtractionType::kSize

  auto innerStructType = ROW({{"x", INTEGER()}, {"y", ARRAY(BIGINT())}});
  auto mapType = MAP(VARCHAR(), innerStructType);
  auto schema = ROW({{"a", mapType}, {"b", INTEGER()}});

  // Build the ScanSpec.
  auto spec = makeAllFieldsScanSpec(*schema);

  // Build extraction.
  using connector::hive::configureExtractionScanSpec;
  using connector::hive::ExtractionPathElement;
  using connector::hive::ExtractionPathElementPtr;
  using connector::hive::ExtractionStep;
  using connector::hive::NamedExtraction;

  std::vector<NamedExtraction> extractions = {
      {"out",
       {ExtractionPathElement::structField("a"),
        ExtractionPathElement::simple(ExtractionStep::kMapValues),
        ExtractionPathElement::simple(ExtractionStep::kArrayElements),
        ExtractionPathElement::structField("y"),
        ExtractionPathElement::simple(ExtractionStep::kSize)},
       ARRAY(BIGINT())}};

  configureExtractionScanSpec(schema, extractions, *spec, pool());

  // Level 1 (ROOT ROW): "b" pruned, "a" not pruned.
  auto* bSpec = spec->childByName("b");
  ASSERT_NE(bSpec, nullptr);
  ASSERT_TRUE(bSpec->isConstant());

  auto* aSpec = spec->childByName("a");
  ASSERT_NE(aSpec, nullptr);
  ASSERT_FALSE(aSpec->isConstant());

  // Level 2 ("a" MAP): ExtractionType::kValues.
  ASSERT_EQ(aSpec->extractionType(), common::ScanSpec::ExtractionType::kValues);

  // Level 3 (values ROW): "x" pruned, "y" not pruned.
  auto* valuesSpec = aSpec->childByName(common::ScanSpec::kMapValuesFieldName);
  ASSERT_NE(valuesSpec, nullptr);

  auto* xSpec = valuesSpec->childByName("x");
  ASSERT_NE(xSpec, nullptr);
  ASSERT_TRUE(xSpec->isConstant());

  auto* ySpec = valuesSpec->childByName("y");
  ASSERT_NE(ySpec, nullptr);
  ASSERT_FALSE(ySpec->isConstant());

  // Level 4 ("y" ARRAY): ExtractionType::kSize.
  ASSERT_EQ(ySpec->extractionType(), common::ScanSpec::ExtractionType::kSize);

  // The values struct has kField extraction for "y" (the only needed field).
  ASSERT_EQ(
      valuesSpec->extractionType(), common::ScanSpec::ExtractionType::kField);

  // Verify kField targets the correct field index.
  ASSERT_EQ(valuesSpec->extractionFieldIndex(), 1);

  // Write test data and verify the reader produces correct results.
  //
  // Each row i has map: {"k" -> {x: i*10, y: [0..i]}}
  // So y has (i+1) elements.
  constexpr int kNumRows = 10;
  std::vector<StringView> allKeys;
  std::vector<int32_t> allX;
  std::vector<std::vector<int64_t>> allY;
  for (int i = 0; i < kNumRows; ++i) {
    allKeys.emplace_back("k");
    allX.push_back(i * 10);
    std::vector<int64_t> yValues;
    for (int j = 0; j <= i; ++j) {
      yValues.push_back(i * 100 + j);
    }
    allY.push_back(std::move(yValues));
  }

  auto keys = makeFlatVector<StringView>(allKeys);
  auto xFlat = makeFlatVector<int32_t>(allX);
  auto yArray = makeArrayVector<int64_t>(allY);
  auto rowValues = makeRowVector({"x", "y"}, {xFlat, yArray});

  // Each row has exactly 1 map entry.
  std::vector<vector_size_t> mapOffsets(kNumRows);
  std::iota(mapOffsets.begin(), mapOffsets.end(), 0);
  auto map = makeMapVector(mapOffsets, keys, rowValues);
  auto bCol = makeFlatVector<int32_t>(kNumRows, folly::identity);
  auto batch = makeRowVector({"a", "b"}, {map, bCol});

  auto [writer, reader] =
      createWriterReader({batch}, pool(), dataIoStats_, metadataIoStats_);

  // Configure the ScanSpec for reading.  In production, the extraction is
  // applied to the column's spec (not the root), so we configure "a"'s
  // spec directly with the sub-chain after StructField("a").
  using connector::hive::applyExtractionChain;
  auto readSpec = makeAllFieldsScanSpec(*schema);
  auto* readASpec = readSpec->childByName("a");

  // Sub-chain starting from the MAP type: [MapValues, AE, SF("y"), Size].
  std::vector<NamedExtraction> aExtractions = {
      {"out",
       {ExtractionPathElement::simple(ExtractionStep::kMapValues),
        ExtractionPathElement::simple(ExtractionStep::kArrayElements),
        ExtractionPathElement::structField("y"),
        ExtractionPathElement::simple(ExtractionStep::kSize)},
       ARRAY(BIGINT())}};
  configureExtractionScanSpec(mapType, aExtractions, *readASpec, pool());

  // Prune "b" as constant null (mimic HiveDataSource behavior).
  auto* readBSpec = readSpec->childByName("b");
  readBSpec->setConstantValue(
      BaseVector::createNullConstant(INTEGER(), 1, pool()));

  // Set a full-chain transform (used as fallback for delta updates).
  auto fullChain = aExtractions[0].chain;
  readASpec->setTransform(
      [fullChain](const VectorPtr& input, memory::MemoryPool* p) -> VectorPtr {
        return applyExtractionChain(input, fullChain, p);
      },
      BIGINT());

  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(readSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);

  auto result = BaseVector::create(schema, 0, pool());
  ASSERT_TRUE(rowReader->next(kNumRows, result));
  ASSERT_EQ(result->size(), kNumRows);

  // The reader with kValues extraction on "a" produces an ArrayVector.
  // The values struct reader has kField extraction for "y", so it
  // produces the "y" field directly (BIGINT from kSize).  No remaining
  // transform is needed.  So elements are BIGINT.
  auto* resultRow = result->as<RowVector>();
  ASSERT_NE(resultRow, nullptr);

  // "b" should be null (pruned).
  auto* bResult = resultRow->childAt(1).get();
  ASSERT_TRUE(bResult->isConstantEncoding());

  // "a" should be an ArrayVector (from kValues extraction).
  auto* aResult = resultRow->childAt(0)->loadedVector()->as<ArrayVector>();
  ASSERT_NE(aResult, nullptr);

  // Each element is BIGINT (size of y array).
  auto* elements = aResult->elements()->asFlatVector<int64_t>();
  ASSERT_NE(elements, nullptr);
  for (int i = 0; i < kNumRows; ++i) {
    ASSERT_EQ(aResult->sizeAt(i), 1) << "Row " << i << " should have 1 entry";
    // y array had (i+1) elements, so size should be (i+1).
    ASSERT_EQ(elements->valueAt(aResult->offsetAt(i)), i + 1)
        << "Incorrect size at row " << i;
  }

  // Validate IO reduction: nested extraction should read fewer bytes than
  // a full scan because "b" column, map keys, "x" field, and y array
  // elements are all skipped.
  constexpr int kLargeNumRows = 1'000;
  std::vector<StringView> largeKeys;
  std::vector<int32_t> largeX;
  std::vector<std::vector<int64_t>> largeY;
  for (int i = 0; i < kLargeNumRows; ++i) {
    largeKeys.emplace_back("k");
    largeX.push_back(i * 10);
    std::vector<int64_t> yVals(10);
    std::iota(yVals.begin(), yVals.end(), i * 100);
    largeY.push_back(std::move(yVals));
  }
  auto largeKeysVec = makeFlatVector<StringView>(largeKeys);
  auto largeXVec = makeFlatVector<int32_t>(largeX);
  auto largeYVec = makeArrayVector<int64_t>(largeY);
  auto largeRowValues = makeRowVector({"x", "y"}, {largeXVec, largeYVec});
  std::vector<vector_size_t> largeMapOffsets(kLargeNumRows);
  std::iota(largeMapOffsets.begin(), largeMapOffsets.end(), 0);
  auto largeMap = makeMapVector(largeMapOffsets, largeKeysVec, largeRowValues);
  auto largeBCol = makeFlatVector<int32_t>(kLargeNumRows, folly::identity);
  auto largeBatch = makeRowVector({"a", "b"}, {largeMap, largeBCol});

  auto largeSink =
      std::make_unique<MemorySink>(1 << 20, FileSink::Options{.pool = pool()});
  auto* largeSinkPtr = largeSink.get();
  auto largeWriter = E2EWriterTestUtil::writeData(
      std::move(largeSink),
      schema,
      {largeBatch},
      std::make_shared<dwrf::Config>(),
      E2EWriterTestUtil::simpleFlushPolicyFactory(true));
  std::string largeFileData(largeSinkPtr->data(), largeSinkPtr->size());

  auto runLargeWithSpec =
      [&](const std::shared_ptr<common::ScanSpec>& scanSpec) -> uint64_t {
    auto ioStats = std::make_shared<io::IoStatistics>();
    // Disable coalescing so each stream is read separately.
    // Disable file preloading so small files don't bypass per-stream IO.
    auto input = std::make_unique<BufferedInput>(
        std::make_shared<InMemoryReadFile>(largeFileData),
        *pool(),
        MetricsLog::voidLog(),
        ioStats.get(),
        /*ioStats=*/nullptr,
        /*maxMergeDistance=*/0);
    dwio::common::ReaderOptions readerOpts(pool());
    readerOpts.setDataIoStats(ioStats);
    readerOpts.setMetadataIoStats(ioStats);
    readerOpts.setFileFormat(FileFormat::DWRF);
    readerOpts.setFilePreloadThreshold(0);
    auto rdr = DwrfReader::create(std::move(input), readerOpts);
    RowReaderOptions rro;
    rro.setScanSpec(scanSpec);
    auto rr = rdr->createRowReader(rro);
    auto res = BaseVector::create(schema, 0, pool());
    while (rr->next(kLargeNumRows, res) > 0) {
    }
    return ioStats->rawBytesRead();
  };

  // Full scan: read all columns.
  auto fullSpec2 = makeAllFieldsScanSpec(*schema);
  auto fullBytes = runLargeWithSpec(fullSpec2);

  // Extraction scan with nested pushdown.
  auto extSpec2 = makeAllFieldsScanSpec(*schema);
  configureExtractionScanSpec(schema, extractions, *extSpec2, pool());
  extSpec2->childByName("b")->setConstantValue(
      BaseVector::createNullConstant(INTEGER(), 1, pool()));
  auto extBytes = runLargeWithSpec(extSpec2);

  ASSERT_GT(fullBytes, 0);
  ASSERT_LT(extBytes, fullBytes)
      << "Nested extraction: " << extBytes << ", Full: " << fullBytes;
}

} // namespace
} // namespace facebook::velox::dwrf
