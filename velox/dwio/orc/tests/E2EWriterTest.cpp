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

#include <folly/Random.h>
#include <random>
#include "velox/common/base/SpillConfig.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/tests/SharedArbitratorTestUtil.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/dwio/dwrf/common/Config.h"
#include "velox/dwio/dwrf/reader/ColumnReader.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/dwrf/test/OrcTest.h"
#include "velox/dwio/dwrf/test/utils/E2EWriterTestUtil.h"
#include "velox/type/fbhive/HiveTypeParser.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace ::testing;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::test;
using namespace facebook::velox::type::fbhive;
using namespace facebook::velox;
using facebook::velox::memory::MemoryPool;
using folly::Random;

constexpr uint64_t kSizeMB = 1024UL * 1024UL;

namespace {
class E2EWriterTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    TestValue::enable();
    memory::MemoryManager::testingSetInstance({});
  }

  E2EWriterTest() {
    rootPool_ = memory::memoryManager()->addRootPool("E2EWriterTest");
    leafPool_ = rootPool_->addLeafChild("leaf");
  }

  static std::unique_ptr<dwrf::DwrfReader> createReader(
      const MemorySink& sink,
      const dwio::common::ReaderOptions& opts) {
    std::string_view data(sink.data(), sink.size());
    return std::make_unique<dwrf::DwrfReader>(
        opts,
        std::make_unique<BufferedInput>(
            std::make_shared<InMemoryReadFile>(data), opts.memoryPool()));
  }

  std::shared_ptr<MemoryPool> rootPool_;
  std::shared_ptr<MemoryPool> leafPool_;
};

VectorPtr createRowVector(
    facebook::velox::memory::MemoryPool* pool,
    const std::shared_ptr<const Type>& type,
    size_t batchSize,
    const VectorPtr& child) {
  return std::make_shared<RowVector>(
      pool,
      type,
      BufferPtr(nullptr),
      batchSize,
      std::vector<VectorPtr>{child},
      /*nullCount=*/0);
}

TEST_F(E2EWriterTest, E2E) {
  const size_t batchCount = 4;
  // Start with a size larger than stride to cover splitting into
  // strides. Continue with smaller size for faster test.
  size_t batchSize = 1100;

  HiveTypeParser parser;
  auto type = parser.parse(
      "struct<"
      "bool_val:boolean,"
      "byte_val:tinyint,"
      "short_val:smallint,"
      "int_val:int,"
      "long_val:bigint,"
      "float_val:float,"
      "double_val:double,"
      "string_val:string,"
      "binary_val:binary,"
      "date_val:date,"
      "short_decimal_val:decimal(9,5),"
      "long_decimal_val:decimal(30,15),"
      "timestamp_val:timestamp,"
      "array_val:array<float>,"
      "map_val:map<int,double>,"
      "map_val:map<bigint,double>," /* this is column 15 */
      "map_val:map<bigint,map<string, int>>," /* this is column 16 */
      "struct_val:struct<a:float,b:double>"
      ">");

  auto config = std::make_shared<dwrf::Config>();
  config->set(dwrf::Config::ROW_INDEX_STRIDE, static_cast<uint32_t>(1000));
  config->set(dwrf::Config::FLATTEN_MAP, false);

  std::vector<VectorPtr> batches;
  for (size_t i = 0; i < batchCount; ++i) {
    batches.push_back(
        BatchMaker::createBatch(type, batchSize, *leafPool_, nullptr, i));
    batchSize = 200;
  }

  dwrf::E2EWriterTestUtil::testWriter(
      *leafPool_,
      type,
      batches,
      1,
      1,
      config,
      /*flushPolicyFactory=*/nullptr,
      /*layoutPlannerFactory=*/nullptr,
      /*memoryBudget=*/std::numeric_limits<int64_t>::max(),
      /*verifyContent=*/true,
      FileFormat::ORC);
}

TEST_F(E2EWriterTest, PartialStride) {
  auto type = ROW({"bool_val"}, {INTEGER()});
  FileFormat fileFormat = FileFormat::ORC;

  size_t batchSize = 1'000;

  auto config = std::make_shared<dwrf::Config>();
  auto sink = std::make_unique<MemorySink>(
      2 * 1024 * 1024,
      dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto sinkPtr = sink.get();

  dwrf::WriterOptions options;
  options.config = config;
  options.schema = type;
  options.memoryPool = rootPool_.get();
  dwrf::Writer writer{std::move(sink), options, fileFormat};

  auto nulls = allocateNulls(batchSize, leafPool_.get());
  auto* nullsPtr = nulls->asMutable<uint64_t>();
  size_t nullCount = 0;

  auto values = AlignedBuffer::allocate<int32_t>(batchSize, leafPool_.get());
  auto* valuesPtr = values->asMutable<int32_t>();

  for (size_t i = 0; i < batchSize; ++i) {
    if ((i & 1) == 0) {
      bits::clearNull(nullsPtr, i);
      valuesPtr[i] = i;
    } else {
      bits::setNull(nullsPtr, i);
      nullCount++;
    }
  }

  auto batch = createRowVector(
      leafPool_.get(),
      type,
      batchSize,
      std::make_shared<FlatVector<int32_t>>(
          leafPool_.get(),
          type->childAt(0),
          nulls,
          batchSize,
          values,
          std::vector<BufferPtr>()));

  writer.write(batch);
  writer.close();

  dwio::common::ReaderOptions readerOpts{leafPool_.get()};
  readerOpts.setFileFormat(fileFormat);
  RowReaderOptions rowReaderOpts;
  auto reader = createReader(*sinkPtr, readerOpts);
  ASSERT_EQ(
      batchSize - nullCount, reader->columnStatistics(1)->getNumberOfValues())
      << reader->columnStatistics(1)->toString();
  ASSERT_EQ(true, reader->columnStatistics(1)->hasNull().value());
}

TEST_F(E2EWriterTest, OversizeRows) {
  auto pool = facebook::velox::memory::memoryManager()->addLeafPool();

  HiveTypeParser parser;
  auto type = parser.parse(
      "struct<"
      "map_val:map<string, map<string, map<string, map<string, string>>>>,"
      "list_val:array<array<array<array<string>>>>,"
      "struct_val:struct<"
      "map_val_field_1:map<string, map<string, map<string, map<string, string>>>>,"
      "list_val_field_1:array<array<array<array<string>>>>,"
      "list_val_field_2:array<array<array<array<string>>>>,"
      "map_val_field_2:map<string, map<string, map<string, map<string, string>>>>"
      ">,"
      ">");
  auto config = std::make_shared<dwrf::Config>();
  config->set(dwrf::Config::DISABLE_LOW_MEMORY_MODE, true);
  config->set(dwrf::Config::STRIPE_SIZE, 10 * kSizeMB);
  config->set(
      dwrf::Config::RAW_DATA_SIZE_PER_BATCH, folly::to<uint64_t>(20 * 1024UL));

  // Retained bytes in vector: 44704
  auto singleBatch = dwrf::E2EWriterTestUtil::generateBatches(
      type, 1, 1, /*seed=*/1411367325, *pool);

  dwrf::E2EWriterTestUtil::testWriter(
      *pool,
      type,
      singleBatch,
      1,
      1,
      config,
      /*flushPolicyFactory=*/nullptr,
      /*layoutPlannerFactory=*/nullptr,
      /*memoryBudget=*/std::numeric_limits<int64_t>::max(),
      false,
      FileFormat::ORC);
}

TEST_F(E2EWriterTest, OversizeBatches) {
  auto pool = facebook::velox::memory::memoryManager()->addLeafPool();

  HiveTypeParser parser;
  auto type = parser.parse(
      "struct<"
      "bool_val:boolean,"
      "byte_val:tinyint,"
      "float_val:float,"
      "double_val:double,"
      ">");
  auto config = std::make_shared<dwrf::Config>();
  config->set(dwrf::Config::DISABLE_LOW_MEMORY_MODE, true);
  config->set(dwrf::Config::STRIPE_SIZE, 10 * kSizeMB);

  // Test splitting a gigantic batch.
  auto singleBatch = dwrf::E2EWriterTestUtil::generateBatches(
      type, 1, 10000000, /*seed=*/1411367325, *pool);
  // A gigantic batch is split into 10 stripes.
  dwrf::E2EWriterTestUtil::testWriter(
      *pool,
      type,
      singleBatch,
      10,
      10,
      config,
      /*flushPolicyFactory=*/nullptr,
      /*layoutPlannerFactory=*/nullptr,
      /*memoryBudget=*/std::numeric_limits<int64_t>::max(),
      false,
      FileFormat::ORC);

  // Test splitting multiple huge batches.
  auto batches = dwrf::E2EWriterTestUtil::generateBatches(
      type, 3, 5000000, /*seed=*/1411367325, *pool);
  // 3 gigantic batches are split into 15~16 stripes.
  dwrf::E2EWriterTestUtil::testWriter(
      *pool,
      type,
      batches,
      15,
      16,
      config,
      /*flushPolicyFactory=*/nullptr,
      /*layoutPlannerFactory=*/nullptr,
      /*memoryBudget=*/std::numeric_limits<int64_t>::max(),
      false,
      FileFormat::ORC);
}

TEST_F(E2EWriterTest, OverflowLengthIncrements) {
  auto pool = facebook::velox::memory::memoryManager()->addLeafPool();

  HiveTypeParser parser;
  auto type = parser.parse(
      "struct<"
      "struct_val:struct<bigint_val:bigint>"
      ">");
  auto config = std::make_shared<dwrf::Config>();
  config->set(dwrf::Config::DISABLE_LOW_MEMORY_MODE, true);
  config->set(dwrf::Config::STRIPE_SIZE, 10 * kSizeMB);
  config->set(
      dwrf::Config::RAW_DATA_SIZE_PER_BATCH,
      folly::to<uint64_t>(500 * 1024UL * 1024UL));

  const size_t batchSize = 1024;

  auto nulls =
      AlignedBuffer::allocate<char>(bits::nbytes(batchSize), pool.get());
  auto* nullsPtr = nulls->asMutable<uint64_t>();
  for (size_t i = 0; i < batchSize; ++i) {
    // Only the first element is non-null
    bits::setNull(nullsPtr, i, i != 0);
  }

  // Bigint column
  VectorMaker maker{pool.get()};
  auto child = maker.flatVector<int64_t>(std::vector<int64_t>{1UL});

  std::vector<VectorPtr> children{child};
  auto rowVec = std::make_shared<RowVector>(
      pool.get(),
      type->childAt(0),
      nulls,
      batchSize,
      children,
      /*nullCount=*/batchSize - 1);

  // Retained bytes in vector: 192, which is much less than 1024
  auto vec = std::make_shared<RowVector>(
      pool.get(),
      type,
      BufferPtr{},
      batchSize,
      std::vector<VectorPtr>{rowVec},
      /*nullCount=*/0);

  dwrf::E2EWriterTestUtil::testWriter(
      *pool,
      type,
      {vec},
      1,
      1,
      config,
      /*flushPolicyFactory=*/nullptr,
      /*layoutPlannerFactory=*/nullptr,
      /*memoryBudget=*/std::numeric_limits<int64_t>::max(),
      false,
      FileFormat::ORC);
}

void testWriter(
    MemoryPool& pool,
    const std::shared_ptr<const Type>& type,
    size_t batchCount,
    std::function<VectorPtr()> generator,
    const std::shared_ptr<dwrf::Config> config =
        std::make_shared<dwrf::Config>()) {
  std::vector<VectorPtr> batches;
  for (auto i = 0; i < batchCount; ++i) {
    batches.push_back(generator());
  }
  dwrf::E2EWriterTestUtil::testWriter(
      pool,
      type,
      batches,
      1,
      1,
      config,
      /*flushPolicyFactory=*/nullptr,
      /*layoutPlannerFactory=*/nullptr,
      /*memoryBudget=*/std::numeric_limits<int64_t>::max(),
      true,
      FileFormat::ORC);
};

TEST_F(E2EWriterTest, fuzzSimple) {
  auto pool = memory::memoryManager()->addLeafPool();
  auto type = ROW({
      {"bool_val", BOOLEAN()},
      {"byte_val", TINYINT()},
      {"short_val", SMALLINT()},
      {"int_val", INTEGER()},
      {"long_val", BIGINT()},
      {"float_val", REAL()},
      {"double_val", DOUBLE()},
      {"string_val", VARCHAR()},
      {"binary_val", VARBINARY()},
      {"ts_val", TIMESTAMP()},
  });
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;

  // Small batches creates more edge cases.
  size_t batchSize = 10;
  VectorFuzzer noNulls(
      {
          .vectorSize = batchSize,
          .nullRatio = 0,
          .stringLength = 20,
          .stringVariableLength = true,
      },
      pool.get(),
      seed);

  VectorFuzzer hasNulls{
      {
          .vectorSize = batchSize,
          .nullRatio = 0.05,
          .stringLength = 10,
          .stringVariableLength = true,
      },
      pool.get(),
      seed};

  auto iterations = 20;
  auto batches = 20;
  for (auto i = 0; i < iterations; ++i) {
    testWriter(
        *pool, type, batches, [&]() { return noNulls.fuzzInputRow(type); });
    testWriter(
        *pool, type, batches, [&]() { return hasNulls.fuzzInputRow(type); });
  }
}

TEST_F(E2EWriterTest, fuzzComplex) {
  auto pool = memory::memoryManager()->addLeafPool();
  auto type = ROW({
      {"array", ARRAY(REAL())},
      {"map", MAP(INTEGER(), DOUBLE())},
      {"row",
       ROW({
           {"a", REAL()},
           {"b", INTEGER()},
       })},
      {"nested",
       ARRAY(ROW({
           {"a", INTEGER()},
           {"b", MAP(REAL(), REAL())},
       }))},
  });
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;

  // Small batches creates more edge cases.
  size_t batchSize = 10;
  VectorFuzzer noNulls(
      {
          .vectorSize = batchSize,
          .nullRatio = 0,
          .stringLength = 20,
          .stringVariableLength = true,
          .containerLength = 5,
          .containerVariableLength = true,
      },
      pool.get(),
      seed);

  VectorFuzzer hasNulls{
      {
          .vectorSize = batchSize,
          .nullRatio = 0.05,
          .stringLength = 10,
          .stringVariableLength = true,
          .containerLength = 5,
          .containerVariableLength = true,
      },
      pool.get(),
      seed};

  auto iterations = 20;
  auto batches = 20;
  for (auto i = 0; i < iterations; ++i) {
    testWriter(
        *pool, type, batches, [&]() { return noNulls.fuzzInputRow(type); });
    testWriter(
        *pool, type, batches, [&]() { return hasNulls.fuzzInputRow(type); });
  }
}
} // namespace
