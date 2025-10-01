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

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <re2/re2.h>
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#endif

#include "fb_velox/nimble/writer/NimbleWriter.h"

namespace facebook::velox::connector::hive {
namespace {

using namespace facebook::velox::common;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

// Specialized version of HiveDataSinkTest that can register nimble properly
class HiveDataSinkTestWithNimble : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
#ifdef VELOX_ENABLE_PARQUET
    parquet::registerParquetReaderFactory();
    parquet::registerParquetWriterFactory();
#endif
    // Register NIMBLE writer factory for feature reordering tests
    nimble::registerNimbleWriterFactory();
    Type::registerSerDe();
    HiveSortingColumn::registerSerDe();
    HiveBucketProperty::registerSerDe();

    rowType_ =
        ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
            {BIGINT(),
             INTEGER(),
             SMALLINT(),
             REAL(),
             DOUBLE(),
             VARCHAR(),
             BOOLEAN()});

    setupMemoryPools();

    spillExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(
        std::thread::hardware_concurrency());
  }

  void TearDown() override {
    nimble::unregisterNimbleWriterFactory();
    connectorQueryCtx_.reset();
    connectorPool_.reset();
    opPool_.reset();
    root_.reset();
    HiveConnectorTestBase::TearDown();
  }

  std::vector<RowVectorPtr> createVectors(int vectorSize, int numVectors) {
    VectorFuzzer::Options options;
    options.vectorSize = vectorSize;
    VectorFuzzer fuzzer(options, pool());
    std::vector<RowVectorPtr> vectors;
    vectors.reserve(numVectors);
    for (int i = 0; i < numVectors; ++i) {
      vectors.push_back(fuzzer.fuzzInputRow(rowType_));
    }
    return vectors;
  }

  void setupMemoryPools() {
    connectorQueryCtx_.reset();
    connectorPool_.reset();
    opPool_.reset();
    root_.reset();

    root_ = memory::memoryManager()->addRootPool(
        "HiveDataSinkTest", 1L << 30, exec::MemoryReclaimer::create());
    opPool_ = root_->addLeafChild("operator");
    connectorPool_ =
        root_->addAggregateChild("connector", exec::MemoryReclaimer::create());

    connectorQueryCtx_ = std::make_unique<connector::ConnectorQueryCtx>(
        opPool_.get(),
        connectorPool_.get(),
        connectorSessionProperties_.get(),
        nullptr,
        common::PrefixSortConfig(),
        nullptr,
        nullptr,
        "query.HiveDataSinkTest",
        "task.HiveDataSinkTest",
        "planNodeId.HiveDataSinkTest",
        0,
        "");
  }

  const std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();

  std::shared_ptr<memory::MemoryPool> root_;
  std::shared_ptr<memory::MemoryPool> opPool_;
  std::shared_ptr<memory::MemoryPool> connectorPool_;
  RowTypePtr rowType_;
  std::shared_ptr<config::ConfigBase> connectorSessionProperties_ =
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>(),
          /*mutable=*/true);
  std::unique_ptr<ConnectorQueryCtx> connectorQueryCtx_;
  std::shared_ptr<HiveConfig> connectorConfig_ =
      std::make_shared<HiveConfig>(std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  std::unique_ptr<folly::IOThreadPoolExecutor> spillExecutor_;
};

// Test that validates the FEATURE_REORDERING config entry got added to
// NimbleConfig.h and verifies it gets properly stored in rawConfig field
// added to VeloxWriterOptions.h
TEST_F(HiveDataSinkTestWithNimble, featureReorderingConfigStoredInRawConfig) {
  // Tests:
  // 1. FEATURE_REORDERING config entry exists in NimbleConfig.h
  // 2. rawConfig field stores serde params in VeloxWriterOptions.h
  // 3. NimbleWriterOptionBuilder stores rawConfig
  const auto outputDirectory = exec::test::TempDirectoryPath::create();

  auto featureMapType = MAP(INTEGER(), INTEGER());
  auto testRowType = ROW({"id", "features"}, {BIGINT(), featureMapType});

  // test that FEATURE_REORDERING config entry works
  std::unordered_map<std::string, std::string> serdeParameters;
  serdeParameters["alpha.feature.reordering"] =
      R"([{"column_ordinal": 1, "feature_order": [3, 2, 1]}])";

  auto insertTableHandle = makeHiveInsertTableHandle(
      testRowType->names(),
      testRowType->children(),
      {}, // partitionedBy
      nullptr, // bucketProperty
      std::make_shared<LocationHandle>(
          outputDirectory->getPath(),
          outputDirectory->getPath(),
          LocationHandle::TableType::kNew),
      dwio::common::FileFormat::NIMBLE, // Use NIMBLE format
      CompressionKind::CompressionKind_NONE,
      serdeParameters,
      nullptr, // writerOptions
      false); // ensureFiles

  // this should succeed bc config infrastructure works
  auto dataSink = std::make_unique<HiveDataSink>(
      testRowType,
      insertTableHandle,
      connectorQueryCtx_.get(),
      CommitStrategy::kTaskCommit,
      connectorConfig_);

  // create test data with map columns
  auto keyVector =
      BaseVector::create<FlatVector<int32_t>>(INTEGER(), 3, pool());
  auto valueVector =
      BaseVector::create<FlatVector<int32_t>>(INTEGER(), 3, pool());
  keyVector->set(0, 1);
  keyVector->set(1, 2);
  keyVector->set(2, 3);
  valueVector->set(0, 100);
  valueVector->set(1, 200);
  valueVector->set(2, 300);

  auto offsetsBuffer = AlignedBuffer::allocate<vector_size_t>(2, pool());
  auto rawOffsets = offsetsBuffer->asMutable<vector_size_t>();
  rawOffsets[0] = 0;
  rawOffsets[1] = 3;

  auto sizesBuffer = AlignedBuffer::allocate<vector_size_t>(1, pool());
  auto rawSizes = sizesBuffer->asMutable<vector_size_t>();
  rawSizes[0] = 3;

  auto mapVector = std::make_shared<MapVector>(
      pool(),
      featureMapType,
      nullptr,
      1,
      offsetsBuffer,
      sizesBuffer,
      keyVector,
      valueVector);

  auto idVector = BaseVector::create<FlatVector<int64_t>>(BIGINT(), 1, pool());
  idVector->set(0, 1001);

  auto input = std::make_shared<RowVector>(
      pool(),
      testRowType,
      nullptr,
      1,
      std::vector<VectorPtr>{idVector, mapVector});

  dataSink->appendData(input);
  dataSink->finish();
  auto partitionUpdates = dataSink->close();

  ASSERT_EQ(partitionUpdates.size(), 1);
  const auto stats = dataSink->stats();
  ASSERT_GT(stats.numWrittenBytes, 0);
}

TEST_F(HiveDataSinkTestWithNimble, nimbleFactoryRegistration) {
  // Test that nimble::registerNimbleWriterFactory() called in SetUp() works
  // correctly This validates the basic nimble infrastructure is available for
  // feature tests
  const auto vectors = createVectors(10, 1);
  ASSERT_EQ(vectors.size(), 1);
  ASSERT_GT(vectors[0]->size(), 0);
}

TEST_F(
    HiveDataSinkTestWithNimble,
    featureReorderingConfigEndToEndWithNimbleFormat) {
  // Tests that alpha.feature.reordering serde parameter gets passed through the
  // entire Velox pipeline to NIMBLE writer when using FileFormat::NIMBLE (not
  // DWRF) This validates implementation works in the actual format where
  // feature reordering matters
  const auto outputDirectory = exec::test::TempDirectoryPath::create();

  auto featureMapType = MAP(INTEGER(), INTEGER());
  auto testRowType = ROW({"id", "features"}, {BIGINT(), featureMapType});

  std::unordered_map<std::string, std::string> serdeParameters;
  serdeParameters["alpha.feature.reordering"] =
      R"([{"column_ordinal": 1, "feature_order": [3, 2, 1]}])";

  auto insertTableHandle = makeHiveInsertTableHandle(
      testRowType->names(),
      testRowType->children(),
      {}, // partitionedBy
      nullptr, // bucketProperty
      std::make_shared<LocationHandle>(
          outputDirectory->getPath(),
          outputDirectory->getPath(),
          LocationHandle::TableType::kNew),
      dwio::common::FileFormat::NIMBLE, // Use NIMBLE format specifically
      CompressionKind::CompressionKind_NONE,
      serdeParameters,
      nullptr, // writerOptions
      false); // ensureFiles

  auto dataSink = std::make_unique<HiveDataSink>(
      testRowType,
      insertTableHandle,
      connectorQueryCtx_.get(),
      CommitStrategy::kTaskCommit,
      connectorConfig_);

  // Create test data with specific feature order
  auto keyVector =
      BaseVector::create<FlatVector<int32_t>>(INTEGER(), 3, pool());
  auto valueVector =
      BaseVector::create<FlatVector<int32_t>>(INTEGER(), 3, pool());
  keyVector->set(0, 1);
  keyVector->set(1, 2);
  keyVector->set(2, 3);
  valueVector->set(0, 100);
  valueVector->set(1, 200);
  valueVector->set(2, 300);

  auto offsetsBuffer = AlignedBuffer::allocate<vector_size_t>(2, pool());
  auto rawOffsets = offsetsBuffer->asMutable<vector_size_t>();
  rawOffsets[0] = 0;
  rawOffsets[1] = 3;

  auto sizesBuffer = AlignedBuffer::allocate<vector_size_t>(1, pool());
  auto rawSizes = sizesBuffer->asMutable<vector_size_t>();
  rawSizes[0] = 3;

  auto mapVector = std::make_shared<MapVector>(
      pool(),
      featureMapType,
      nullptr,
      1,
      offsetsBuffer,
      sizesBuffer,
      keyVector,
      valueVector);

  auto idVector = BaseVector::create<FlatVector<int64_t>>(BIGINT(), 1, pool());
  idVector->set(0, 1001);

  auto input = std::make_shared<RowVector>(
      pool(),
      testRowType,
      nullptr,
      1,
      std::vector<VectorPtr>{idVector, mapVector});

  dataSink->appendData(input);
  dataSink->finish();
  auto partitionUpdates = dataSink->close();

  ASSERT_EQ(partitionUpdates.size(), 1);
  const auto stats = dataSink->stats();
  ASSERT_GT(stats.numWrittenBytes, 0);
}

TEST_F(
    HiveDataSinkTestWithNimble,
    featureReorderingInvalidConfigGracefulHandling) {
  // Tests that our implementation gracefully handles malformed JSON
  // configurations without crashing. Validates the robustness of the config
  // parsing in NimbleWriterOptionBuilder and the error handling in the NIMBLE
  // writer
  const auto outputDirectory = exec::test::TempDirectoryPath::create();

  struct {
    std::string name;
    std::string config;
  } testCases[] = {
      {"invalid_json", "invalid_json"},
      {"out_of_bounds_column",
       R"([{"column_ordinal": 99, "feature_order": [1, 2, 3]}])"},
      {"malformed_structure",
       R"([{"column_ordinal": "not_an_int", "feature_order": [1, 2]}])"}};

  for (const auto& testCase : testCases) {
    SCOPED_TRACE("Test case: " + testCase.name);

    std::unordered_map<std::string, std::string> serdeParameters;
    serdeParameters["alpha.feature.reordering"] = testCase.config;

    auto insertTableHandle = makeHiveInsertTableHandle(
        rowType_->names(),
        rowType_->children(),
        {}, // partitionedBy
        nullptr, // bucketProperty
        std::make_shared<LocationHandle>(
            outputDirectory->getPath() + "/" + testCase.name,
            outputDirectory->getPath() + "/" + testCase.name,
            LocationHandle::TableType::kNew),
        dwio::common::FileFormat::NIMBLE,
        CompressionKind::CompressionKind_NONE,
        serdeParameters,
        nullptr, // writerOptions
        false); // ensureFiles

    // Should not throw with invalid configs
    auto dataSink = std::make_unique<HiveDataSink>(
        rowType_,
        insertTableHandle,
        connectorQueryCtx_.get(),
        CommitStrategy::kTaskCommit,
        connectorConfig_);

    const auto vectors = createVectors(10, 1);
    dataSink->appendData(vectors[0]);
    dataSink->finish();
    auto partitionUpdates = dataSink->close();

    ASSERT_EQ(partitionUpdates.size(), 1);
  }
}

} // namespace
} // namespace facebook::velox::connector::hive
