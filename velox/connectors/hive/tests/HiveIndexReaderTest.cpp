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

#include "dwio/nimble/index/IndexConfig.h"
#include "dwio/nimble/velox/VeloxWriter.h"
#include "dwio/nimble/velox/selective/SelectiveNimbleReader.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/HiveIndexReader.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::connector::hive {
namespace {

using namespace facebook::velox::exec::test;

/// Tests for HiveIndexReader's getAdaptedOutputType() flatmap-as-struct
/// support.
class HiveIndexReaderTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    nimble::registerSelectiveNimbleReaderFactory();
    filesystems::registerLocalFileSystem();
  }

  void TearDown() override {
    // Reset all objects that hold memory pool references before the base
    // TearDown resets the MemoryManager.
    indexReader_.reset();
    connectorQueryCtx_.reset();
    queryCtx_.reset();
    connectorSessionProperties_.reset();
    hiveConfig_.reset();
    scanSpec_.reset();
    tableHandle_.reset();
    nimble::unregisterSelectiveNimbleReaderFactory();
    HiveConnectorTestBase::TearDown();
  }

  /// Writes a Nimble file with flat map columns and cluster index.
  /// The data schema is:
  ///   (key_col: BIGINT, flatmap_col: MAP(INTEGER, BIGINT))
  /// Index is on key_col (ascending).
  void writeNimbleFile(
      const std::string& filePath,
      const std::vector<RowVectorPtr>& vectors) {
    auto writeFile = std::make_unique<LocalWriteFile>(filePath, true, false);

    nimble::IndexConfig indexConfig{
        .columns = {"key_col"},
        .sortOrders = {nimble::SortOrder{.ascending = true}},
        .enforceKeyOrder = true,
    };
    indexConfig.encodingLayout = nimble::EncodingLayout{
        nimble::EncodingType::Prefix,
        {},
        nimble::CompressionType::Uncompressed};

    nimble::VeloxWriterOptions writerOptions{
        .indexConfig = std::move(indexConfig),
    };
    writerOptions.flatMapColumns.insert("flatmap_col");

    nimble::VeloxWriter writer(
        dataType_, std::move(writeFile), *pool_, std::move(writerOptions));
    for (const auto& vector : vectors) {
      writer.write(vector);
    }
    writer.close();
  }

  /// Writes a Nimble file with two flat map columns and cluster index.
  void writeNimbleFileMultiFlatMap(
      const std::string& filePath,
      const std::vector<RowVectorPtr>& vectors) {
    auto writeFile = std::make_unique<LocalWriteFile>(filePath, true, false);

    nimble::IndexConfig indexConfig{
        .columns = {"key_col"},
        .sortOrders = {nimble::SortOrder{.ascending = true}},
        .enforceKeyOrder = true,
    };
    indexConfig.encodingLayout = nimble::EncodingLayout{
        nimble::EncodingType::Prefix,
        {},
        nimble::CompressionType::Uncompressed};

    nimble::VeloxWriterOptions writerOptions{
        .indexConfig = std::move(indexConfig),
    };
    writerOptions.flatMapColumns.insert("flatmap_col1");
    writerOptions.flatMapColumns.insert("flatmap_col2");

    nimble::VeloxWriter writer(
        multiDataType_, std::move(writeFile), *pool_, std::move(writerOptions));
    for (const auto& vector : vectors) {
      writer.write(vector);
    }
    writer.close();
  }

  /// Creates a HiveIndexReader with the given output type.
  std::unique_ptr<HiveIndexReader> createIndexReader(
      const std::string& filePath,
      const RowTypePtr& outputType,
      const RowTypePtr& dataColumns) {
    auto split = HiveConnectorSplitBuilder(filePath)
                     .connectorId(kHiveConnectorId)
                     .fileFormat(dwio::common::FileFormat::NIMBLE)
                     .build();

    auto tableHandle = std::make_shared<HiveTableHandle>(
        kHiveConnectorId,
        "test_table",
        common::SubfieldFilters{},
        nullptr,
        dataColumns,
        std::vector<std::string>{"key_col"});

    auto scanSpec = makeScanSpec(
        outputType,
        /*outputSubfields=*/{},
        /*subfieldFilters=*/{},
        dataColumns,
        /*partitionKeys=*/{},
        /*infoColumns=*/{},
        /*specialColumns=*/{},
        /*disableStatsBasedFilterReorder=*/false,
        pool_.get());

    auto hiveConfig =
        std::make_shared<HiveConfig>(std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{}));

    auto connectorSessionProperties = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{});

    auto queryCtx = core::QueryCtx::create(executor_.get());
    auto connectorQueryCtx = std::make_unique<ConnectorQueryCtx>(
        pool_.get(),
        pool_.get(),
        connectorSessionProperties.get(),
        nullptr,
        common::PrefixSortConfig(),
        std::make_unique<exec::SimpleExpressionEvaluator>(
            queryCtx.get(), pool_.get()),
        nullptr,
        "query.test",
        "task.test",
        "planNodeId.test",
        0,
        "");

    // Store objects that must outlive the reader.
    queryCtx_ = std::move(queryCtx);
    connectorQueryCtx_ = std::move(connectorQueryCtx);
    connectorSessionProperties_ = std::move(connectorSessionProperties);
    hiveConfig_ = std::move(hiveConfig);
    scanSpec_ = std::move(scanSpec);
    tableHandle_ = std::move(tableHandle);

    auto probeType = ROW({"probe_key"}, {BIGINT()});
    auto lookupConditions = std::vector<core::IndexLookupConditionPtr>{
        std::make_shared<core::EqualIndexLookupCondition>(
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "key_col"),
            std::make_shared<core::FieldAccessTypedExpr>(
                BIGINT(), "probe_key")),
    };

    auto ioStats = std::make_shared<io::IoStatistics>();
    auto fsStats = std::make_shared<IoStats>();

    return std::make_unique<HiveIndexReader>(
        std::vector<std::shared_ptr<const HiveConnectorSplit>>{split},
        tableHandle_,
        connectorQueryCtx_.get(),
        hiveConfig_,
        scanSpec_,
        lookupConditions,
        probeType,
        outputType,
        ioStats,
        fsStats,
        fileHandleFactory_.get(),
        executor_.get());
  }

  /// Makes a probe request vector for index lookup.
  RowVectorPtr makeProbeVector(const std::vector<int64_t>& keys) {
    auto probeType = ROW({"probe_key"}, {BIGINT()});
    auto keyVector = makeFlatVector<int64_t>(keys);
    return makeRowVector(probeType->names(), {keyVector});
  }

  // Data schema: (key_col: BIGINT, flatmap_col: MAP(INTEGER, BIGINT)).
  RowTypePtr dataType_ =
      ROW({"key_col", "flatmap_col"}, {BIGINT(), MAP(INTEGER(), BIGINT())});

  // Multi flat map data schema.
  RowTypePtr multiDataType_ =
      ROW({"key_col", "flatmap_col1", "flatmap_col2"},
          {BIGINT(), MAP(INTEGER(), BIGINT()), MAP(INTEGER(), BIGINT())});

  // Stored state for reader lifetime management.
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<ConnectorQueryCtx> connectorQueryCtx_;
  std::shared_ptr<config::ConfigBase> connectorSessionProperties_;
  std::shared_ptr<HiveConfig> hiveConfig_;
  std::shared_ptr<common::ScanSpec> scanSpec_;
  std::shared_ptr<const HiveTableHandle> tableHandle_;

  std::unique_ptr<FileHandleFactory> fileHandleFactory_ =
      std::make_unique<FileHandleFactory>(
          std::make_unique<SimpleLRUCache<FileHandleKey, FileHandle>>(1000),
          std::make_unique<FileHandleGenerator>());

  // Index reader stored here so it can be cleaned up in TearDown.
  std::unique_ptr<HiveIndexReader> indexReader_;
};

/// Verifies that HiveIndexReader constructs successfully when outputType has
/// ROW for flatmap columns (flatmap-as-struct). The getAdaptedOutputType()
/// method adapts ROW back to MAP for the underlying reader. Without the fix
/// in D96684816, this would crash during construction because the reader
/// would receive mismatched MAP vs ROW types.
TEST_F(HiveIndexReaderTest, flatMapAsStructConstruction) {
  auto tempFile = TempFilePath::create();

  // Write data with sorted key_col and flatmap_col as MAP(INTEGER, BIGINT).
  auto data = makeRowVector(
      dataType_->names(),
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeMapVector<int32_t, int64_t>(
              {{{1, 10}, {2, 20}}, {{1, 100}}, {{3, 300}}}),
      });
  writeNimbleFile(tempFile->getPath(), {data});

  // Output type requests flatmap_col as ROW (struct with key names).
  auto structOutputType =
      ROW({"key_col", "flatmap_col"},
          {BIGINT(), ROW({"1", "2"}, {BIGINT(), BIGINT()})});

  // Construction should succeed — getAdaptedOutputType() adapts ROW→MAP.
  indexReader_ =
      createIndexReader(tempFile->getPath(), structOutputType, dataType_);
  ASSERT_NE(indexReader_, nullptr);

  // Verify the scanSpec has isFlatMapAsStruct set on flatmap_col.
  auto* flatmapSpec = scanSpec_->childByName("flatmap_col");
  ASSERT_NE(flatmapSpec, nullptr);
  ASSERT_TRUE(flatmapSpec->isFlatMapAsStruct());
}

/// Verifies flatmap-as-struct construction with multiple flat map columns.
TEST_F(HiveIndexReaderTest, flatMapAsStructMultipleColumns) {
  auto tempFile = TempFilePath::create();

  auto data = makeRowVector(
      multiDataType_->names(),
      {
          makeFlatVector<int64_t>({1, 2}),
          makeMapVector<int32_t, int64_t>({{{1, 10}}, {{2, 20}}}),
          makeMapVector<int32_t, int64_t>({{{5, 50}}, {{6, 60}}}),
      });
  writeNimbleFileMultiFlatMap(tempFile->getPath(), {data});

  // Both MAP columns read as struct.
  auto structOutputType =
      ROW({"key_col", "flatmap_col1", "flatmap_col2"},
          {BIGINT(), ROW({"1"}, {BIGINT()}), ROW({"5"}, {BIGINT()})});

  indexReader_ =
      createIndexReader(tempFile->getPath(), structOutputType, multiDataType_);
  ASSERT_NE(indexReader_, nullptr);

  // Verify both columns have isFlatMapAsStruct set.
  ASSERT_TRUE(scanSpec_->childByName("flatmap_col1")->isFlatMapAsStruct());
  ASSERT_TRUE(scanSpec_->childByName("flatmap_col2")->isFlatMapAsStruct());
}

/// Verifies that when outputType uses MAP (no struct adaptation), no
/// flatmap-as-struct flag is set and the reader reads data correctly.
TEST_F(HiveIndexReaderTest, noAdaptationWhenMapOutputType) {
  auto tempFile = TempFilePath::create();

  auto data = makeRowVector(
      dataType_->names(),
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeMapVector<int32_t, int64_t>({{{1, 10}}, {{2, 20}}, {{3, 30}}}),
      });
  writeNimbleFile(tempFile->getPath(), {data});

  // Output type matches data type — MAP, no flatmap-as-struct.
  indexReader_ = createIndexReader(tempFile->getPath(), dataType_, dataType_);
  ASSERT_NE(indexReader_, nullptr);

  // Verify flatmap_col does NOT have isFlatMapAsStruct set.
  auto* flatmapSpec = scanSpec_->childByName("flatmap_col");
  ASSERT_NE(flatmapSpec, nullptr);
  ASSERT_FALSE(flatmapSpec->isFlatMapAsStruct());

  // Perform a lookup for key_col = 2 and verify results.
  auto probe = makeProbeVector({2});
  HiveIndexReader::Request request{probe};
  indexReader_->startLookup(request);
  ASSERT_TRUE(indexReader_->hasNext());
  auto result = indexReader_->next(1024);
  ASSERT_NE(result, nullptr);
  ASSERT_NE(result->output, nullptr);
  ASSERT_GT(result->output->size(), 0);
}

/// Verifies that requesting a missing key in MAP output type still works.
TEST_F(HiveIndexReaderTest, mapOutputLookupMissingKey) {
  auto tempFile = TempFilePath::create();

  auto data = makeRowVector(
      dataType_->names(),
      {
          makeFlatVector<int64_t>({1, 2}),
          makeMapVector<int32_t, int64_t>({{{1, 10}}, {{2, 20}}}),
      });
  writeNimbleFile(tempFile->getPath(), {data});

  indexReader_ = createIndexReader(tempFile->getPath(), dataType_, dataType_);
  ASSERT_NE(indexReader_, nullptr);

  // Lookup key_col = 99 which doesn't exist — should return no results.
  auto probe = makeProbeVector({99});
  HiveIndexReader::Request request{probe};
  indexReader_->startLookup(request);
  // No results expected for non-existent key.
  bool hasResults = false;
  while (indexReader_->hasNext()) {
    auto result = indexReader_->next(1024);
    if (result && result->output && result->output->size() > 0) {
      hasResults = true;
    }
  }
  ASSERT_FALSE(hasResults);
}

/// Verifies end-to-end flatmap-as-struct reading through HiveIndexReader.
/// The output type has ROW for the flatmap column, and the reader should
/// produce struct output with the correct values.
TEST_F(HiveIndexReaderTest, flatMapAsStructRead) {
  auto tempFile = TempFilePath::create();

  auto data = makeRowVector(
      dataType_->names(),
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeMapVector<int32_t, int64_t>(
              {{{1, 10}, {2, 20}}, {{1, 100}}, {{3, 300}}}),
      });
  writeNimbleFile(tempFile->getPath(), {data});

  // Output type requests flatmap_col as ROW (struct).
  auto structOutputType =
      ROW({"key_col", "flatmap_col"},
          {BIGINT(), ROW({"1", "2"}, {BIGINT(), BIGINT()})});

  indexReader_ =
      createIndexReader(tempFile->getPath(), structOutputType, dataType_);
  ASSERT_NE(indexReader_, nullptr);

  // Lookup key_col = 2 and verify struct output.
  auto probe = makeProbeVector({2});
  HiveIndexReader::Request request{probe};
  indexReader_->startLookup(request);
  ASSERT_TRUE(indexReader_->hasNext());
  auto result = indexReader_->next(1024);
  ASSERT_NE(result, nullptr);
  ASSERT_NE(result->output, nullptr);
  ASSERT_GT(result->output->size(), 0);

  // Verify the output type is ROW (struct), not MAP.
  auto flatmapChild = result->output->childAt(1);
  ASSERT_TRUE(flatmapChild->type()->isRow());
}

} // namespace
} // namespace facebook::velox::connector::hive
