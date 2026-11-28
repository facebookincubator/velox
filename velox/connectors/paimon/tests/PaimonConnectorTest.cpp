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
#include "velox/connectors/paimon/PaimonConnector.h"

#include <fmt/ranges.h>
#include <folly/init/Init.h>
#include <folly/lang/Bits.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <random>
#include <type_traits>

#include "velox/common/base/RandomUtil.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/LocalFile.h"
#include "velox/common/file/PlainUserNameTokenProvider.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/process/ThreadDebugInfo.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/iceberg/DeletionVectorWriter.h"
#include "velox/connectors/paimon/PaimonColumnHandle.h"
#include "velox/connectors/paimon/PaimonConnectorSplit.h"
#include "velox/connectors/paimon/PaimonRowKind.h"
#include "velox/connectors/paimon/PaimonTableHandle.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/orc/reader/OrcReader.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TaskMonitor.h"
#include "velox/expression/Expr.h"
#include "velox/vector/DecodedVector.h"

#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#endif

#ifdef VELOX_ENABLE_NIMBLE
#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/writer/Writer.h"
#endif

namespace facebook::velox::connector::paimon {
namespace {

using hive::HiveConfig;
using hive::HiveConnector;
using hive::HiveTableHandle;

static_assert(std::is_base_of_v<connector::FileConnector, PaimonConnector>);
static_assert(!std::is_base_of_v<HiveConnector, PaimonConnector>);

static const std::string kPaimonConnectorId = "test-paimon";

class TableValidationFileSystem : public tests::utils::FaultyFileSystem {
 public:
  TableValidationFileSystem() : FaultyFileSystem(nullptr) {}

  std::string_view extractPath(std::string_view path) const override {
    return path.substr(std::string_view("paimon-table-test:").size());
  }

  std::unique_ptr<ReadFile> openFileForRead(
      std::string_view path,
      const filesystems::FileOptions& options) override {
    ++openedFiles;
    if (options.tokenProvider == nullptr) {
      ++numOpensWithoutTokenProvider;
    }
    return FaultyFileSystem::openFileForRead(path, options);
  }

  std::atomic<int32_t> openedFiles{0};

  // Opens without a token provider and getFileSystem() calls without a
  // connector config. Data files get both.
  std::atomic_int32_t numOpensWithoutTokenProvider{0};
  std::atomic_int32_t numResolvesWithoutConfig{0};
};

class PaimonConnectorTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    Type::registerSerDe();
    common::Filter::registerSerDe();
    core::ITypedExpr::registerSerDe();
    core::PlanNode::registerSerDe();
    PaimonConnector::registerSerDe();
    orc::registerOrcReaderFactory();
    auto config = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{});
    auto connector =
        PaimonConnectorFactory().newConnector(kPaimonConnectorId, config);
    ConnectorRegistry::global().insert(connector->connectorId(), connector);
#ifdef VELOX_ENABLE_NIMBLE
    nimble::registerSelectiveNimbleReaderFactory();
#endif
#ifdef VELOX_ENABLE_PARQUET
    parquet::registerParquetReaderFactory();
#endif
  }

  ConnectorQueryCtx::Builder queryContextBuilder(core::QueryCtx* queryCtx) {
    ConnectorQueryCtx::Builder builder;
    builder.operatorPool(pool())
        .connectorPool(rootPool_.get())
        .expressionEvaluator(
            std::make_unique<exec::SimpleExpressionEvaluator>(queryCtx, pool()))
        .queryId("PaimonConnectorTest")
        .taskId("PaimonConnectorTest")
        .planNodeId("scan");
    return builder;
  }

  void TearDown() override {
    orc::unregisterOrcReaderFactory();
#ifdef VELOX_ENABLE_NIMBLE
    nimble::unregisterSelectiveNimbleReaderFactory();
#endif
#ifdef VELOX_ENABLE_PARQUET
    parquet::unregisterParquetReaderFactory();
#endif
    ConnectorRegistry::global().erase(kPaimonConnectorId);
    HiveConnectorTestBase::TearDown();
  }

  static std::vector<dwio::common::FileFormat> fileFormats() {
    std::vector<dwio::common::FileFormat> formats{
        dwio::common::FileFormat::DWRF};
#ifdef VELOX_ENABLE_NIMBLE
    formats.push_back(dwio::common::FileFormat::NIMBLE);
#endif
#ifdef VELOX_ENABLE_PARQUET
    formats.push_back(dwio::common::FileFormat::PARQUET);
#endif
    return formats;
  }

  void writeFile(
      const std::string& path,
      const RowVectorPtr& input,
      dwio::common::FileFormat format) {
    if (format == dwio::common::FileFormat::DWRF) {
      writeToFile(path, input);
      return;
    }
#ifdef VELOX_ENABLE_NIMBLE
    if (format == dwio::common::FileFormat::NIMBLE) {
      nimble::Writer writer(
          input->rowType(),
          std::make_unique<LocalWriteFile>(path, false, false),
          *pool(),
          nimble::WriterOptions{});
      writer.write(input);
      writer.close();
      return;
    }
#endif
#ifdef VELOX_ENABLE_PARQUET
    if (format == dwio::common::FileFormat::PARQUET) {
      dwio::common::WriterOptions options;
      options.memoryPool = rootPool_.get();
      parquet::Writer writer(
          std::make_unique<dwio::common::WriteFileSink>(
              std::make_unique<LocalWriteFile>(path, false, false), path),
          options,
          input->rowType());
      writer.write(input);
      writer.close();
      return;
    }
#endif
    VELOX_FAIL("Unsupported test file format");
  }

  // Writes each of 'stripes' as its own stripe, so that readers can skip
  // stripes one at a time.
  void writeStripes(
      const std::string& path,
      const std::vector<RowVectorPtr>& stripes,
      dwio::common::FileFormat format) {
    if (format == dwio::common::FileFormat::DWRF) {
      std::vector<uint64_t> rowsPerStripe;
      rowsPerStripe.reserve(stripes.size());
      for (const auto& stripe : stripes) {
        rowsPerStripe.emplace_back(stripe->size());
      }
      writeToFile(
          path, stripes, std::make_shared<dwrf::Config>(), [&rowsPerStripe] {
            return std::make_unique<dwrf::RowsPerStripeFlushPolicy>(
                rowsPerStripe);
          });
      return;
    }
#ifdef VELOX_ENABLE_NIMBLE
    if (format == dwio::common::FileFormat::NIMBLE) {
      nimble::Writer writer(
          asRowType(stripes.front()->type()),
          std::make_unique<LocalWriteFile>(path, false, false),
          *pool(),
          nimble::WriterOptions{});
      for (const auto& stripe : stripes) {
        writer.write(stripe);
        writer.flush();
      }
      writer.close();
      return;
    }
#endif
    VELOX_FAIL("Unsupported test file format");
  }

  static const std::shared_ptr<TableValidationFileSystem>&
  validationFileSystem() {
    static const auto kFileSystem =
        std::make_shared<TableValidationFileSystem>();
    static const bool kRegistered = [] {
      filesystems::registerFileSystem(
          [](std::string_view path) {
            return path.starts_with("paimon-table-test:");
          },
          [](std::shared_ptr<const config::ConfigBase> config,
             std::string_view) {
            if (config == nullptr) {
              ++kFileSystem->numResolvesWithoutConfig;
            }
            return kFileSystem;
          });
      return true;
    }();
    VELOX_CHECK(kRegistered);
    return kFileSystem;
  }

  void testTakeoverRuntimeStats(int fileCount, bool existingStats) {
    auto queryCtx = core::QueryCtx::create();
    auto config = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{});
    // Keep the context and all other borrowed inputs alive to isolate the
    // ownership of runtimeStats during DataSource takeover.
    auto context = ConnectorQueryCtx::Builder()
                       .operatorPool(pool())
                       .connectorPool(rootPool_.get())
                       .sessionProperties(config.get())
                       .expressionEvaluator(
                           std::make_unique<exec::SimpleExpressionEvaluator>(
                               queryCtx.get(), pool()))
                       .queryId("PaimonConnectorTest")
                       .taskId("PaimonConnectorTest")
                       .planNodeId("scan")
                       .build();
    const auto type = ROW({{"c0", BIGINT()}});
    auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
    auto table = makePaimonTableHandle(type);
    auto columns = makePaimonColumnHandles(type);
    auto makeSource = [&] {
      return connector->createDataSource(type, table, columns, context.get());
    };
    auto processedFiles = [](DataSource& source) {
      auto stats = source.getRuntimeStats();
      auto it = stats.find("processedSplits");
      return it == stats.end() ? 0 : it->second.sum;
    };
    for (auto format : fileFormats()) {
      SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
      const int firstFile = existingStats ? 1 : 0;
      auto files = makeFilePaths(firstFile + fileCount);
      std::vector<std::string> paths;
      std::vector<RowVectorPtr> inputs;
      for (int i = 0; i < files.size(); ++i) {
        paths.push_back(files[i]->getPath());
        inputs.push_back(
            makeRowVector({makeFlatVector<int64_t>({10 * i + 1, 10 * i + 2})}));
        writeFile(paths.back(), inputs.back(), format);
      }
      auto source = makeSource();
      auto readFiles = [&](int begin, int count) {
        ContinueFuture future = ContinueFuture::makeEmpty();
        for (int i = begin; i < begin + count; ++i) {
          auto result = source->next(100, future);
          ASSERT_TRUE(result.has_value());
          ASSERT_FALSE(future.valid());
          ASSERT_TRUE(*result);
          test::assertEqualVectors(inputs[i], *result);
        }
        auto result = source->next(100, future);
        ASSERT_TRUE(result.has_value());
        ASSERT_FALSE(future.valid());
        EXPECT_EQ(*result, nullptr);
      };
      if (existingStats) {
        source->addSplit(makePaimonSplit(
            {paths.front()}, PaimonTableType::kAppendOnly, format));
        readFiles(0, 1);
      }
      ASSERT_EQ(processedFiles(*source), firstFile);
      auto preloaded = makeSource();
      preloaded->addSplit(makePaimonSplit(
          std::vector<std::string>(paths.begin() + firstFile, paths.end()),
          PaimonTableType::kAppendOnly,
          format));
      // Paimon opens even the first file lazily. No file preparation has
      // occurred when the donor is destroyed by setFromDataSource().
      ASSERT_EQ(processedFiles(*preloaded), 0);
      source->setFromDataSource(std::move(preloaded));
      ASSERT_EQ(preloaded, nullptr);
      EXPECT_EQ(processedFiles(*source), firstFile);
      readFiles(firstFile, fileCount);
      EXPECT_EQ(processedFiles(*source), firstFile + fileCount);
      EXPECT_EQ(source->getCompletedRows(), 2 * (firstFile + fileCount));
    }
  }

  static std::vector<int32_t> fieldIds(const RowTypePtr& rowType) {
    std::vector<int32_t> ids;
    for (column_index_t i = 0; i < rowType->size(); ++i) {
      ids.push_back(i + 10);
    }
    return ids;
  }

  static std::shared_ptr<PaimonTableHandle> makePaimonTableHandle(
      const RowTypePtr& rowType,
      common::SubfieldFilters subfieldFilters = {},
      const core::TypedExprPtr& remainingFilter = nullptr,
      const std::vector<std::string>& partitionNames = {},
      std::vector<connector::FileColumnHandlePtr> filterColumns = {},
      std::unordered_map<std::string, std::string> options = {},
      PaimonTableType tableType = PaimonTableType::kAppendOnly,
      std::vector<std::string> primaryKeys = {}) {
    return std::make_shared<PaimonTableHandle>(
        kPaimonConnectorId,
        "paimon_table",
        17,
        rowType,
        fieldIds(rowType),
        tableType,
        partitionNames,
        std::move(options),
        std::move(subfieldFilters),
        remainingFilter,
        std::move(filterColumns),
        /*sampleRate=*/1.0,
        /*dbName=*/"",
        std::move(primaryKeys));
  }

  /// Creates column assignments for all columns as regular columns.
  static connector::ColumnHandleMap makePaimonColumnHandles(
      const RowTypePtr& rowType) {
    connector::ColumnHandleMap assignments;
    assignments.reserve(rowType->size());
    for (uint32_t i = 0; i < rowType->size(); ++i) {
      const auto& name = rowType->nameOf(i);
      assignments[name] = std::make_shared<PaimonColumnHandle>(
          name,
          i + 10,
          connector::FileColumnHandle::ColumnType::kRegular,
          rowType->childAt(i));
    }
    return assignments;
  }

  /// Builds a table scan plan using the Paimon connector.
  core::PlanNodePtr makePaimonScanPlan(
      const RowTypePtr& outputType,
      const RowTypePtr& dataColumns = nullptr,
      common::SubfieldFilters subfieldFilters = {},
      const core::TypedExprPtr& remainingFilter = nullptr,
      PaimonTableType tableType = PaimonTableType::kAppendOnly,
      std::unordered_map<std::string, std::string> options = {},
      std::vector<std::string> primaryKeys = {}) {
    return makePaimonScan(
               outputType,
               dataColumns,
               std::move(subfieldFilters),
               remainingFilter,
               tableType,
               std::move(options),
               std::move(primaryKeys))
        .planNode();
  }

  /// Returns a PlanBuilder whose plan is a Paimon table scan, for adding nodes
  /// on top of the scan.
  exec::test::PlanBuilder makePaimonScan(
      const RowTypePtr& outputType,
      const RowTypePtr& dataColumns = nullptr,
      common::SubfieldFilters subfieldFilters = {},
      const core::TypedExprPtr& remainingFilter = nullptr,
      PaimonTableType tableType = PaimonTableType::kAppendOnly,
      std::unordered_map<std::string, std::string> options = {},
      std::vector<std::string> primaryKeys = {}) {
    const auto rowType = dataColumns ? dataColumns : outputType;
    const auto tableHandle = makePaimonTableHandle(
        rowType,
        std::move(subfieldFilters),
        remainingFilter,
        /*partitionNames=*/{},
        /*filterColumns=*/{},
        std::move(options),
        tableType,
        std::move(primaryKeys));
    auto assignments =
        makePaimonColumnHandles(dataColumns ? dataColumns : outputType);
    return exec::test::PlanBuilder()
        .startTableScan()
        .connectorId(kPaimonConnectorId)
        .outputType(outputType)
        .tableHandle(tableHandle)
        .assignments(assignments)
        .endTableScan();
  }

  /// Builds a scan of a primary-key table keyed by 'id'.
  core::PlanNodePtr makePrimaryKeyScanPlan(
      const RowTypePtr& outputType,
      const RowTypePtr& rowType,
      std::unordered_map<std::string, std::string> options = {},
      common::SubfieldFilters subfieldFilters = {},
      const core::TypedExprPtr& remainingFilter = nullptr) {
    return makePaimonScanPlan(
        outputType,
        rowType,
        std::move(subfieldFilters),
        remainingFilter,
        PaimonTableType::kPrimaryKey,
        std::move(options),
        /*primaryKeys=*/{"id"});
  }

  /// Creates a PaimonConnectorSplit of the files at 'filePaths', all in
  /// 'format'.
  std::shared_ptr<PaimonConnectorSplit> makePaimonSplit(
      const std::vector<std::string>& filePaths,
      PaimonTableType tableType = PaimonTableType::kAppendOnly,
      dwio::common::FileFormat format = dwio::common::FileFormat::DWRF,
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {},
      bool rawConvertible = true) {
    PaimonConnectorSplitBuilder builder(
        kPaimonConnectorId, /*snapshotId=*/1, tableType);
    for (const auto& filePath : filePaths) {
      builder.addFile(filePath, format, /*fileSize=*/0);
    }
    for (const auto& [key, value] : partitionKeys) {
      builder.partitionKey(key, value);
    }
    builder.rawConvertible(rawConvertible);
    return builder.build();
  }

  std::shared_ptr<PaimonConnectorSplit> makePaimonSplit(
      const std::vector<PaimonDataFile>& dataFiles,
      PaimonTableType tableType = PaimonTableType::kAppendOnly,
      bool rawConvertible = true) {
    return std::make_shared<PaimonConnectorSplit>(
        kPaimonConnectorId,
        /*snapshotId=*/1,
        tableType,
        dataFiles,
        std::unordered_map<std::string, std::optional<std::string>>{},
        std::nullopt,
        rawConvertible);
  }

  // Writes 'rows' of a table keyed by 'id' to 'path' in the layout of Paimon
  // primary-key files, [_KEY_id, _SEQUENCE_NUMBER, _VALUE_KIND, table
  // columns], and returns its data file. Rows are inserts unless 'rowKinds'
  // says otherwise.
  PaimonDataFile writePrimaryKeyFile(
      const std::string& path,
      const RowVectorPtr& rows,
      const std::vector<int64_t>& sequenceNumbers,
      dwio::common::FileFormat format,
      const std::vector<PaimonRowKind>& rowKinds = {}) {
    VELOX_CHECK_EQ(sequenceNumbers.size(), rows->size());
    std::vector<int8_t> valueKinds(
        rows->size(), static_cast<int8_t>(PaimonRowKind::kInsert));
    if (!rowKinds.empty()) {
      VELOX_CHECK_EQ(rowKinds.size(), rows->size());
      std::transform(
          rowKinds.begin(), rowKinds.end(), valueKinds.begin(), [](auto kind) {
            return static_cast<int8_t>(kind);
          });
    }
    std::vector<std::string> names{
        "_KEY_id", "_SEQUENCE_NUMBER", "_VALUE_KIND"};
    std::vector<VectorPtr> columns{
        rows->childAt("id"),
        makeFlatVector<int64_t>(sequenceNumbers),
        makeFlatVector<int8_t>(valueKinds)};
    const auto& rowType = rows->rowType();
    names.insert(names.end(), rowType->names().begin(), rowType->names().end());
    columns.insert(
        columns.end(), rows->children().begin(), rows->children().end());
    writeFile(path, makeRowVector(names, columns), format);

    PaimonDataFile dataFile;
    dataFile.path = path;
    dataFile.fileFormat = format;
    dataFile.rowCount = rows->size();
    dataFile.minSequenceNumber =
        *std::min_element(sequenceNumbers.begin(), sequenceNumbers.end());
    dataFile.maxSequenceNumber =
        *std::max_element(sequenceNumbers.begin(), sequenceNumbers.end());
    return dataFile;
  }

  // Writes deletion vectors into one index file as Paimon Java does: a version
  // byte, then a [length][magic][bitmap][CRC] frame per vector. Version 1
  // frames hold a 32-bit RoaringBitmap and version 2 frames a Roaring64
  // bitmap. Returns the deletion file of each vector.
  std::vector<PaimonDeletionFile> writeDeletionVectors(
      const std::vector<std::vector<int64_t>>& positions,
      int32_t version) {
    std::vector<std::string> bitmaps;
    std::vector<uint64_t> cardinalities;
    for (const auto& filePositions : positions) {
      hive::iceberg::DeletionVectorWriter writer;
      writer.addDeletedPositions(filePositions);
      cardinalities.emplace_back(writer.numDistinctPositions());
      auto bitmap = writer.serialize();
      if (version == 1) {
        // Positions below 2^32 form a single Roaring64 group. Dropping the
        // group count and key leaves its 32-bit RoaringBitmap.
        bitmap = bitmap.substr(sizeof(uint64_t) + sizeof(uint32_t));
      }
      bitmaps.emplace_back(std::move(bitmap));
    }
    auto deletionFiles = writeDeletionVectorFrames(bitmaps, version);
    for (size_t i = 0; i < deletionFiles.size(); ++i) {
      deletionFiles[i].cardinality = cardinalities[i];
    }
    return deletionFiles;
  }

  // Frames serialized bitmaps, valid or not, as writeDeletionVectors() does.
  // Returns the deletion file of each vector, without cardinality.
  std::vector<PaimonDeletionFile> writeDeletionVectorFrames(
      const std::vector<std::string>& bitmaps,
      int32_t version) {
    VELOX_CHECK(version == 1 || version == 2);
    const auto appendInt = [](std::string& data, uint32_t value, bool big) {
      value = big ? folly::Endian::big(value) : folly::Endian::little(value);
      data.append(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    auto filePath = exec::test::TempFilePath::create();
    std::string data(1, '\1');
    std::vector<PaimonDeletionFile> deletionFiles;
    for (const auto& bitmap : bitmaps) {
      const auto offset = data.size();
      const auto frameLength =
          static_cast<uint32_t>(sizeof(uint32_t) + bitmap.size());
      appendInt(data, frameLength, /*big=*/true);
      if (version == 1) {
        appendInt(data, 1'581'511'376, /*big=*/true);
      } else {
        appendInt(data, 1'681'511'377, /*big=*/false);
      }
      data += bitmap;
      // Paimon Java skips the CRC on read, so leave it zero.
      appendInt(data, 0, /*big=*/true);
      deletionFiles.emplace_back(
          filePath->getPath(),
          offset,
          version == 1 ? frameLength : frameLength + 8,
          /*cardinality=*/std::nullopt);
    }
    {
      std::ofstream out(filePath->getPath(), std::ios::binary);
      out.write(data.data(), static_cast<std::streamsize>(data.size()));
    }
    deletionVectorFiles_.push_back(std::move(filePath));
    return deletionFiles;
  }

  // Returns the handle of an append-only table that samples each row with
  // probability 'sampleRate'.
  static std::shared_ptr<PaimonTableHandle> makeSampledTableHandle(
      const RowTypePtr& rowType,
      double sampleRate) {
    return std::make_shared<PaimonTableHandle>(
        kPaimonConnectorId,
        "paimon_table",
        17,
        rowType,
        fieldIds(rowType),
        PaimonTableType::kAppendOnly,
        /*partitionKeys=*/std::vector<std::string>{},
        /*options=*/std::unordered_map<std::string, std::string>{},
        /*subfieldFilters=*/common::SubfieldFilters{},
        /*remainingFilter=*/nullptr,
        /*filterColumnHandles=*/std::vector<connector::FileColumnHandlePtr>{},
        sampleRate);
  }

  std::vector<std::shared_ptr<exec::test::TempFilePath>> deletionVectorFiles_;
};

TEST_F(PaimonConnectorTest, connectorRegistration) {
  auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  ASSERT_NE(connector, nullptr);
  ASSERT_NE(connector->connectorConfig(), nullptr);
  EXPECT_TRUE(connector->canAddDynamicFilter());
  EXPECT_FALSE(connector->supportsSplitPreload());
  EXPECT_FALSE(connector->supportsIndexLookup());
  VELOX_ASSERT_THROW(
      connector->createDataSink(
          ROW({}), nullptr, nullptr, CommitStrategy::kNoCommit),
      "Paimon connector does not support writes");
}

TEST_F(PaimonConnectorTest, connectorFactory) {
  PaimonConnectorFactory factory;
  EXPECT_EQ(
      std::string(PaimonConnectorFactory::kPaimonConnectorName), "paimon");

  auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {PaimonConfig::kEnableFileHandleCache, "true"},
          {PaimonConfig::kNumCacheFileHandles, "500"}});

  auto connector = factory.newConnector("test-paimon-2", config);
  ASSERT_NE(connector, nullptr);

  PaimonConfig paimonConfig(connector->connectorConfig());
  EXPECT_TRUE(paimonConfig.isFileHandleCacheEnabled());
  EXPECT_EQ(paimonConfig.numCacheFileHandles(), 500);
  auto* paimonConnector = dynamic_cast<PaimonConnector*>(connector.get());
  ASSERT_NE(paimonConnector, nullptr);
  EXPECT_EQ(paimonConnector->fileHandleCacheStats().maxSize, 500);
}

TEST_F(PaimonConnectorTest, configProvider) {
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {PaimonConfig::kCacheMetadata, "true"}});
  auto connector =
      PaimonConnectorFactory().newConnector("paimon-config", config);
  const auto* provider = connector->configProvider();
  ASSERT_NE(provider, nullptr);
  const auto properties = provider->properties();
  EXPECT_EQ(properties.size(), PaimonConfig::registeredProperties().size());
  const auto cacheMetadata = std::find_if(
      properties.begin(), properties.end(), [](const auto& property) {
        return property.name == PaimonConfig::kCacheMetadataSession;
      });
  ASSERT_NE(cacheMetadata, properties.end());
  EXPECT_EQ(cacheMetadata->defaultValue, "true");
  EXPECT_EQ(provider->normalize("cache_metadata", "false"), "false");
  EXPECT_NE(
      std::find_if(
          properties.begin(),
          properties.end(),
          [](const auto& property) {
            return property.name ==
                PaimonConfig::kOrcFooterSpeculativeIoSizeSession;
          }),
      properties.end());
  EXPECT_EQ(
      std::find_if(
          properties.begin(),
          properties.end(),
          [](const auto& property) {
            return property.name ==
                HiveConfig::kInsertExistingPartitionsBehaviorSession;
          }),
      properties.end());
}

// E2E test: read a single file from an append-only Paimon split.
TEST_F(PaimonConnectorTest, takeoverRuntimeStatsBeforeFirstFile) {
  testTakeoverRuntimeStats(1, false);
}

TEST_F(PaimonConnectorTest, takeoverRuntimeStatsAcrossFiles) {
  testTakeoverRuntimeStats(2, false);
}

TEST_F(PaimonConnectorTest, takeoverRuntimeStatsAccumulate) {
  testTakeoverRuntimeStats(2, true);
}

TEST_F(PaimonConnectorTest, appendOnlySingleFile) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto vectors = makeVectors(rowType, 1, 100);

  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);

  auto split = makePaimonSplit({filePaths[0]->getPath()});
  auto plan = makePaimonScanPlan(rowType);

  exec::test::AssertQueryBuilder(plan).split(split).assertResults(vectors);
}

TEST_F(PaimonConnectorTest, ordinaryOrcTableScan) {
  const auto rowType = ROW({{"int1", INTEGER()}, {"string1", VARCHAR()}});
  const auto plan = makePaimonScan(rowType).limit(0, 3, false).planNode();
  const auto path = velox::test::getDataFilePath(
      "velox/dwio/orc/test", "examples/TestOrcFile.columnProjection.orc");
  const auto expected = makeRowVector(
      {"int1", "string1"},
      {makeFlatVector<int32_t>({-1155869325, 431529176, 1761283695}),
       makeFlatVector<std::string>(
           {"bb2c72394b1ab9f8", "e6c5459001105f17", "7f2fc3e9dacc5ecd"})});
  exec::test::AssertQueryBuilder(plan)
      .split(makePaimonSplit(
          {path}, PaimonTableType::kAppendOnly, dwio::common::FileFormat::ORC))
      .assertResults(expected);
}

// E2E test: read multiple files from a single append-only Paimon split.
TEST_F(PaimonConnectorTest, appendOnlyMultipleFiles) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto vectors1 = makeVectors(rowType, 1, 100);
  auto vectors2 = makeVectors(rowType, 1, 50);

  auto filePaths = makeFilePaths(2);
  writeToFile(filePaths[0]->getPath(), vectors1);
  writeToFile(filePaths[1]->getPath(), vectors2);

  auto split =
      makePaimonSplit({filePaths[0]->getPath(), filePaths[1]->getPath()});
  auto plan = makePaimonScanPlan(rowType);

  // Expected result is all rows from both files.
  std::vector<RowVectorPtr> expected;
  expected.insert(expected.end(), vectors1.begin(), vectors1.end());
  expected.insert(expected.end(), vectors2.begin(), vectors2.end());

  exec::test::AssertQueryBuilder(plan).split(split).assertResults(expected);
}

// Paimon names each file after its format, and the files of a split can differ
// in format, e.g. with 'file.format.per.level'. Each file is read in its own
// format.
TEST_F(PaimonConnectorTest, mixedFileFormats) {
  const auto formats = fileFormats();
  if (formats.size() < 2) {
    GTEST_SKIP() << "Needs two file formats";
  }
  const auto rowType = ROW({{"c0", BIGINT()}, {"c1", VARCHAR()}});
  const auto firstInput = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({1, 2}),
       makeFlatVector<std::string>({"a", "b"})});
  const auto secondInput = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({3, 4, 5}),
       makeFlatVector<std::string>({"c", "d", "e"})});
  const auto plan = makePaimonScanPlan(rowType);

  struct TestParam {
    dwio::common::FileFormat firstFileFormat;
    dwio::common::FileFormat secondFileFormat;

    std::string debugString() const {
      return fmt::format(
          "first file format {}, second file format {}",
          firstFileFormat,
          secondFileFormat);
    }
  };
  std::vector<TestParam> testSettings;
  for (auto format : formats) {
    for (auto otherFormat : formats) {
      if (format != otherFormat) {
        testSettings.push_back({format, otherFormat});
      }
    }
  }
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const auto filePaths = makeFilePaths(2);
    writeFile(filePaths[0]->getPath(), firstInput, testData.firstFileFormat);
    writeFile(filePaths[1]->getPath(), secondInput, testData.secondFileFormat);

    std::vector<PaimonDataFile> dataFiles(2);
    dataFiles[0].path = filePaths[0]->getPath();
    dataFiles[0].fileFormat = testData.firstFileFormat;
    dataFiles[1].path = filePaths[1]->getPath();
    dataFiles[1].fileFormat = testData.secondFileFormat;

    exec::test::AssertQueryBuilder(plan)
        .split(makePaimonSplit(dataFiles))
        .assertResults({firstInput, secondInput});
  }
}

TEST_F(PaimonConnectorTest, tableSchemaTypesAndBatchOwnership) {
  const auto input = makeRowVector(
      {"flag", "number", "id", "score", "text"},
      {makeNullableFlatVector<bool>({true, false, std::nullopt, true}),
       makeFlatVector<int32_t>({1, 2, 3, 4}),
       makeFlatVector<int64_t>({11, 12, 13, 14}),
       makeNullableFlatVector<double>({1.5, 2.5, std::nullopt, 4.5}),
       makeNullableFlatVector<std::string>(
           {"alpha", "", std::nullopt, "omega"})});
  const auto rowType = input->rowType();
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto table = makePaimonTableHandle(rowType);
  const auto plan = makePaimonScanPlan(rowType);
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    for (const auto& file : files) {
      writeFile(file->getPath(), input, format);
    }
    auto split = makePaimonSplit(
        {files[0]->getPath(), files[1]->getPath()},
        PaimonTableType::kAppendOnly,
        format);
    split = PaimonConnectorSplit::create(split->serialize());
    exec::test::AssertQueryBuilder(copy).split(split).assertResults(
        {input, input});
    const auto countPlan = exec::test::PlanBuilder()
                               .startTableScan()
                               .connectorId(kPaimonConnectorId)
                               .outputType(ROW({}))
                               .tableHandle(table)
                               .assignments({})
                               .endTableScan()
                               .singleAggregation({}, {"count(1)"})
                               .planNode();
    exec::test::AssertQueryBuilder(countPlan).split(split).assertSingleResult(
        int64_t{8});

    auto source = ConnectorRegistry::tryGet(kPaimonConnectorId)
                      ->createDataSource(
                          rowType,
                          table,
                          makePaimonColumnHandles(rowType),
                          context.get());
    source->addSplit(split);
    ContinueFuture future = ContinueFuture::makeEmpty();
    std::vector<RowVectorPtr> heldOutputs;
    std::vector<RowVectorPtr> expectedOutputs;
    while (true) {
      auto result = source->next(1, future);
      ASSERT_TRUE(result.has_value());
      ASSERT_FALSE(future.valid());
      if (!*result) {
        break;
      }
      for (auto& child : (*result)->children()) {
        child->loadedVector();
      }
      auto expected = BaseVector::create(rowType, (*result)->size(), pool());
      expected->copy(result->get(), 0, 0, (*result)->size());
      expectedOutputs.push_back(std::static_pointer_cast<RowVector>(expected));
      heldOutputs.push_back(std::move(*result));
    }
    EXPECT_EQ(source->getCompletedRows(), 8);
    const auto stats = source->getRuntimeStats();
    source.reset();
    for (size_t i = 0; i < heldOutputs.size(); ++i) {
      test::assertEqualVectors(expectedOutputs[i], heldOutputs[i]);
    }
    EXPECT_EQ(stats.at("processedSplits").sum, 2);
  }
}

// Verify that empty splits (no data files) are rejected by
// PaimonConnectorSplit.
TEST_F(PaimonConnectorTest, rejectsEmptySplit) {
  VELOX_ASSERT_THROW(
      makePaimonSplit(std::vector<std::string>{}),
      "PaimonConnectorSplit requires non-empty dataFiles");
}

// E2E test: read with partition keys from an append-only Paimon split.
TEST_F(PaimonConnectorTest, appendOnlyWithPartitionKeys) {
  // Data file columns (not including partition column).
  auto dataRowType = ROW({"c0"}, {BIGINT()});
  auto vectors = makeVectors(dataRowType, 1, 100);

  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);

  // The output includes both the data column and the partition column.
  auto outputType = ROW({"c0", "p0"}, {BIGINT(), VARCHAR()});
  auto tableHandle = makePaimonTableHandle(
      outputType, /*subfieldFilters=*/{}, /*remainingFilter=*/nullptr, {"p0"});

  connector::ColumnHandleMap assignments;
  assignments["c0"] = std::make_shared<PaimonColumnHandle>(
      "c0", 10, connector::FileColumnHandle::ColumnType::kRegular, BIGINT());
  assignments["p0"] = std::make_shared<PaimonColumnHandle>(
      "p0",
      11,
      connector::FileColumnHandle::ColumnType::kPartitionKey,
      VARCHAR());

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(kPaimonConnectorId)
                  .outputType(outputType)
                  .tableHandle(tableHandle)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  auto split = makePaimonSplit(
      {filePaths[0]->getPath()},
      PaimonTableType::kAppendOnly,
      dwio::common::FileFormat::DWRF,
      {{"p0", std::optional<std::string>("2024-01-01")}});

  // Build expected output with the partition value filled in.
  auto expectedC0 = vectors[0]->childAt(0);
  auto numRows = vectors[0]->size();
  auto expectedP0 =
      BaseVector::createConstant(VARCHAR(), "2024-01-01", numRows, pool());

  auto expected = makeRowVector({"c0", "p0"}, {expectedC0, expectedP0});

  exec::test::AssertQueryBuilder(plan).split(split).assertResults({expected});
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  exec::test::AssertQueryBuilder(copy).split(split).assertResults({expected});
}

TEST_F(PaimonConnectorTest, physicalPartitionColumnsUseSplitConstants) {
  const auto rowType = ROW({{"id", BIGINT()}, {"p", VARCHAR()}});
  const auto table = makePaimonTableHandle(rowType, {}, nullptr, {"p"});
  const ColumnHandleMap assignments{
      {"id",
       std::make_shared<PaimonColumnHandle>(
           "id",
           10,
           connector::FileColumnHandle::ColumnType::kRegular,
           BIGINT())},
      {"p",
       std::make_shared<PaimonColumnHandle>(
           "p",
           11,
           connector::FileColumnHandle::ColumnType::kPartitionKey,
           VARCHAR())}};
  const auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(kPaimonConnectorId)
                        .outputType(rowType)
                        .tableHandle(table)
                        .assignments(assignments)
                        .endTableScan()
                        .planNode();
  const auto input = makeRowVector(
      {"id", "p"},
      {makeFlatVector<int64_t>({1, 2}),
       makeNullableFlatVector<std::string>({"physical-copy", std::nullopt})});
  const auto expected = makeRowVector(
      {"id", "p"},
      {input->childAt(0),
       makeFlatVector<std::string>({"split-value", "split-value"})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    writeFile(files[0]->getPath(), input, format);
    exec::test::AssertQueryBuilder(plan)
        .split(makePaimonSplit(
            {files[0]->getPath()},
            PaimonTableType::kAppendOnly,
            format,
            {{"p", "split-value"}}))
        .assertResults(expected);
  }
}

TEST_F(PaimonConnectorTest, leadingPhysicalPartitionColumn) {
  // Paimon data files store partition columns at their table positions, so a
  // leading partition column must not shift the data columns after it.
  const auto rowType =
      ROW({{"p", VARCHAR()}, {"id", BIGINT()}, {"value", DOUBLE()}});
  const auto table = makePaimonTableHandle(rowType, {}, nullptr, {"p"});
  const ColumnHandleMap assignments{
      {"p",
       std::make_shared<PaimonColumnHandle>(
           "p",
           10,
           connector::FileColumnHandle::ColumnType::kPartitionKey,
           VARCHAR())},
      {"id",
       std::make_shared<PaimonColumnHandle>(
           "id",
           11,
           connector::FileColumnHandle::ColumnType::kRegular,
           BIGINT())},
      {"value",
       std::make_shared<PaimonColumnHandle>(
           "value",
           12,
           connector::FileColumnHandle::ColumnType::kRegular,
           DOUBLE())}};
  const auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(kPaimonConnectorId)
                        .outputType(rowType)
                        .tableHandle(table)
                        .assignments(assignments)
                        .endTableScan()
                        .planNode();
  const auto input = makeRowVector(
      {"p", "id", "value"},
      {makeFlatVector<std::string>({"physical-copy", "physical-copy"}),
       makeFlatVector<int64_t>({1, 2}),
       makeFlatVector<double>({1.5, 2.5})});
  const auto expected = makeRowVector(
      {"p", "id", "value"},
      {makeFlatVector<std::string>({"split-value", "split-value"}),
       input->childAt(1),
       input->childAt(2)});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    writeFile(files[0]->getPath(), input, format);
    exec::test::AssertQueryBuilder(plan)
        .split(makePaimonSplit(
            {files[0]->getPath()},
            PaimonTableType::kAppendOnly,
            format,
            {{"p", "split-value"}}))
        .assertResults(expected);
  }
}

TEST_F(PaimonConnectorTest, tableHandleFiltersAndProjection) {
  const auto rowType = ROW({{"id", BIGINT()}, {"value", BIGINT()}});
  const auto input = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<int64_t>({0, 11, 2, 13, 15})});
  const auto remainingFilter = parseExpr("value % 2 = 1", rowType);
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("id"),
      std::make_unique<common::BigintRange>(2, 4, false));
  auto tableHandle = makePaimonTableHandle(
      rowType,
      std::move(filters),
      remainingFilter,
      {},
      std::vector<connector::FileColumnHandlePtr>{
          std::make_shared<PaimonColumnHandle>(
              "value",
              11,
              connector::FileColumnHandle::ColumnType::kRegular,
              BIGINT())});
  const auto outputType = ROW({{"selected_id", BIGINT()}});
  connector::ColumnHandleMap assignments{
      {"selected_id",
       std::make_shared<PaimonColumnHandle>(
           "id",
           10,
           connector::FileColumnHandle::ColumnType::kRegular,
           BIGINT())}};
  const auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(kPaimonConnectorId)
                        .outputType(outputType)
                        .tableHandle(tableHandle)
                        .assignments(assignments)
                        .endTableScan()
                        .planNode();
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  const auto expected =
      makeRowVector({"selected_id"}, {makeFlatVector<int64_t>({2, 4})});
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto paths = makeFilePaths(1);
    writeFile(paths[0]->getPath(), input, format);
    const auto split = makePaimonSplit(
        {paths[0]->getPath()}, PaimonTableType::kAppendOnly, format);
    exec::test::AssertQueryBuilder(plan).split(split).assertResults(expected);
    exec::test::AssertQueryBuilder(copy).split(split).assertResults(expected);
    auto source = ConnectorRegistry::tryGet(kPaimonConnectorId)
                      ->createDataSource(
                          outputType, tableHandle, assignments, context.get());
    source->addSplit(split);
    // TableScan supplies cumulative static and dynamic conditions.
    const common::BigintRange dynamicFilter(4, 5, false);
    source->addDynamicFilter(
        0,
        source->getFilters()
            ->at(common::Subfield("id"))
            ->mergeWith(&dynamicFilter));
    ContinueFuture future = ContinueFuture::makeEmpty();
    auto result = source->next(100, future);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(*result);
    test::assertEqualVectors(
        makeRowVector({"selected_id"}, {makeFlatVector<int64_t>({4})}),
        *result);
  }
}

TEST_F(PaimonConnectorTest, fileScanCapabilitiesWithPaimonTable) {
  const auto rowType = ROW({{"id", BIGINT()}, {"value", BIGINT()}});
  const auto input = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({0, 1, 2, 3}),
       makeFlatVector<int64_t>({11, 12, 13, 14})});
  std::unordered_map<std::string, std::string> options;
  options["write-buffer-size"] = "256 mb";
  options["scan.mode"] = "latest";
  options["bucket"] = "8";
  options["file.format"] = "avro";
  // A native table option is not a File reader's skip-header instruction.
  options["skip.header.line.count"] = "1";
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("id"), std::make_unique<common::BoolValue>(true, false));
  const auto filterColumn = std::make_shared<PaimonColumnHandle>(
      "value", 11, connector::FileColumnHandle::ColumnType::kRegular, BIGINT());
  const auto table = makePaimonTableHandle(
      rowType,
      std::move(filters),
      parseExpr("value % 2 = 1", rowType),
      {},
      {filterColumn, filterColumn},
      std::move(options));
  const auto outputType = ROW({{"selected_id", BIGINT()}});
  const auto makePlan = [&](bool postProcess) {
    std::function<void(VectorPtr&)> processor;
    if (postProcess) {
      processor = [&](VectorPtr& column) {
        const DecodedVector decoded(*column);
        column =
            makeFlatVector<int64_t>(column->size(), [&](vector_size_t row) {
              return decoded.valueAt<int64_t>(row) + 100;
            });
      };
    }
    std::vector<common::Subfield> requiredSubfields;
    requiredSubfields.emplace_back("id");
    return exec::test::PlanBuilder()
        .startTableScan()
        .connectorId(kPaimonConnectorId)
        .outputType(outputType)
        .tableHandle(table)
        .assignments(
            {{"selected_id",
              std::make_shared<PaimonColumnHandle>(
                  "id",
                  10,
                  connector::FileColumnHandle::ColumnType::kRegular,
                  BIGINT(),
                  std::move(requiredSubfields),
                  std::vector<connector::NamedExtraction>{},
                  /*partitionDateValueDaysSinceEpoch=*/false,
                  std::move(processor))}})
        .endTableScan()
        .planNode();
  };
  const auto plan = makePlan(false);
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  const auto processedPlan = makePlan(true);
  const auto expected =
      makeRowVector({"selected_id"}, {makeFlatVector<int64_t>({2, 2})});
  const auto processedExpected =
      makeRowVector({"selected_id"}, {makeFlatVector<int64_t>({102, 102})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    std::vector<std::string> paths;
    for (const auto& file : files) {
      writeFile(file->getPath(), input, format);
      paths.push_back(file->getPath());
    }
    const auto split =
        makePaimonSplit(paths, PaimonTableType::kAppendOnly, format);
    for (const auto& scan : {plan, copy}) {
      exec::test::AssertQueryBuilder(scan).split(split).assertResults(expected);
    }
    exec::test::AssertQueryBuilder(processedPlan)
        .maxDrivers(1)
        .split(split)
        .assertResults(processedExpected);
  }

  // This local scan keeps the literal name in the Subfield path elements.
  const auto literalType = ROW({{"a.b", BIGINT()}});
  const auto literalInput =
      makeRowVector({"a.b"}, {makeFlatVector<int64_t>({1, 2})});
  std::vector<common::Subfield> requiredSubfields;
  requiredSubfields.push_back(std::move(*common::Subfield::create("a.b")));
  const auto literalPlan =
      exec::test::PlanBuilder()
          .startTableScan()
          .connectorId(kPaimonConnectorId)
          .outputType(literalType)
          .tableHandle(makePaimonTableHandle(literalType))
          .assignments(
              {{"a.b",
                std::make_shared<PaimonColumnHandle>(
                    "a.b",
                    10,
                    connector::FileColumnHandle::ColumnType::kRegular,
                    BIGINT(),
                    std::move(requiredSubfields))}})
          .endTableScan()
          .planNode();
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    writeFile(files[0]->getPath(), literalInput, format);
    exec::test::AssertQueryBuilder(literalPlan)
        .split(makePaimonSplit(
            {files[0]->getPath()}, PaimonTableType::kAppendOnly, format))
        .assertResults(literalInput);
  }
}

TEST_F(PaimonConnectorTest, partitionFilterOnlyConstants) {
  const auto rowType = ROW({{"id", BIGINT()}, {"p", VARCHAR()}});
  const auto input = makeRowVector({"id"}, {makeFlatVector<int64_t>({7, 8})});
  const auto partitionColumn = std::make_shared<PaimonColumnHandle>(
      "p",
      11,
      connector::FileColumnHandle::ColumnType::kPartitionKey,
      VARCHAR());
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    writeFile(files[0]->getPath(), input, format);
    for (const std::optional<std::string>& partition :
         {std::optional<std::string>{}, std::optional<std::string>{""}}) {
      const auto table = makePaimonTableHandle(
          rowType,
          common::SubfieldFilters{},
          parseExpr("p is null or length(p) = 0", rowType),
          {"p"},
          std::vector<connector::FileColumnHandlePtr>{partitionColumn});
      const auto plan =
          exec::test::PlanBuilder()
              .startTableScan()
              .connectorId(kPaimonConnectorId)
              .outputType(ROW({{"id", BIGINT()}}))
              .tableHandle(table)
              .assignments(makePaimonColumnHandles(ROW({{"id", BIGINT()}})))
              .endTableScan()
              .planNode();
      const auto split = makePaimonSplit(
          {files[0]->getPath()},
          PaimonTableType::kAppendOnly,
          format,
          {{"p", partition}});
      exec::test::AssertQueryBuilder(plan).split(split).assertResults(input);
      const auto copy =
          ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
      exec::test::AssertQueryBuilder(copy).split(split).assertResults(input);
    }
    common::SubfieldFilters filters;
    filters.emplace(common::Subfield("p"), std::make_unique<common::IsNull>());
    const auto table = makePaimonTableHandle(
        rowType,
        std::move(filters),
        nullptr,
        {"p"},
        std::vector<connector::FileColumnHandlePtr>{partitionColumn});
    const auto plan =
        exec::test::PlanBuilder()
            .startTableScan()
            .connectorId(kPaimonConnectorId)
            .outputType(ROW({{"id", BIGINT()}}))
            .tableHandle(table)
            .assignments(makePaimonColumnHandles(ROW({{"id", BIGINT()}})))
            .endTableScan()
            .planNode();
    const auto nullSplit = makePaimonSplit(
        {files[0]->getPath()},
        PaimonTableType::kAppendOnly,
        format,
        {{"p", std::nullopt}});
    const auto emptyStringSplit = makePaimonSplit(
        {files[0]->getPath()},
        PaimonTableType::kAppendOnly,
        format,
        {{"p", ""}});
    exec::test::AssertQueryBuilder(plan).split(nullSplit).assertResults(input);
    exec::test::AssertQueryBuilder(plan)
        .split(emptyStringSplit)
        .assertEmptyResults();
  }
}

TEST_F(PaimonConnectorTest, fileExpressionPipelineWithLambda) {
  const auto rowType = ROW({{"id", BIGINT()}, {"value", BIGINT()}});
  const auto input = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({10, 20, 30})});
  // The lambda captures 'value', which is needed by the filter but is not
  // included in the scan output.
  const auto table = makePaimonTableHandle(
      rowType, {}, parseExpr("any_match(array[20], x -> x = value)", rowType));
  const auto plan =
      exec::test::PlanBuilder()
          .startTableScan()
          .connectorId(kPaimonConnectorId)
          .outputType(ROW({{"id", BIGINT()}}))
          .tableHandle(table)
          .assignments(makePaimonColumnHandles(ROW({{"id", BIGINT()}})))
          .endTableScan()
          .planNode();
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  const auto expected =
      makeRowVector({"id"}, {makeFlatVector<int64_t>({2, 2})});
  for (auto format : fileFormats()) {
    const auto files = makeFilePaths(2);
    for (const auto& file : files) {
      writeFile(file->getPath(), input, format);
    }
    const auto split = makePaimonSplit(
        {files[0]->getPath(), files[1]->getPath()},
        PaimonTableType::kAppendOnly,
        format);
    for (const auto& scan : {plan, copy}) {
      exec::test::AssertQueryBuilder(scan).split(split).assertResults(expected);
    }
  }
}

TEST_F(PaimonConnectorTest, dynamicFilterAndEmptyFiles) {
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto rowType = ROW({{"a", BIGINT()}, {"b", DOUBLE()}});
  const auto table = makePaimonTableHandle(rowType);
  const auto input = makeRowVector(
      {"a", "b"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<double>({1.5, 2.5, 3.5})});
  const auto empty = makeRowVector(
      {"a", "b"}, {makeFlatVector<int64_t>({}), makeFlatVector<double>({})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(4);
    auto firstEmpty = files[0]->getPath();
    auto nextEmpty = files[2]->getPath();
#ifdef VELOX_ENABLE_PARQUET
    if (format == dwio::common::FileFormat::PARQUET) {
      firstEmpty = velox::test::getDataFilePath(
          "velox/dwio/parquet/tests", "examples/empty.parquet");
      nextEmpty = firstEmpty;
    } else
#endif
    {
      writeFile(firstEmpty, empty, format);
      writeFile(nextEmpty, empty, format);
    }
    writeFile(files[1]->getPath(), input, format);
    writeFile(files[3]->getPath(), input, format);
    auto source = ConnectorRegistry::tryGet(kPaimonConnectorId)
                      ->createDataSource(
                          rowType,
                          table,
                          makePaimonColumnHandles(rowType),
                          context.get());
    source->addSplit(makePaimonSplit(
        {firstEmpty, files[1]->getPath(), nextEmpty, files[3]->getPath()},
        PaimonTableType::kAppendOnly,
        format));
    ContinueFuture future = ContinueFuture::makeEmpty();
    auto first = source->next(100, future);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(*first);
    test::assertEqualVectors(input, *first);
    source->addDynamicFilter(
        0, std::make_shared<common::BigintRange>(2, 2, false));
    auto second = source->next(100, future);
    ASSERT_TRUE(second.has_value());
    ASSERT_TRUE(*second);
    test::assertEqualVectors(
        makeRowVector(
            {"a", "b"},
            {makeFlatVector<int64_t>({2}), makeFlatVector<double>({2.5})}),
        *second);
    auto end = source->next(100, future);
    ASSERT_TRUE(end.has_value());
    EXPECT_EQ(*end, nullptr);
    EXPECT_EQ(source->getCompletedRows(), 6);
    const auto stats = source->getRuntimeStats();
    EXPECT_EQ(stats.at("processedSplits").sum, 2);
    EXPECT_EQ(
        source->getRuntimeStats().at("processedSplits").sum,
        stats.at("processedSplits").sum);
  }
}

TEST_F(PaimonConnectorTest, tableValidationBeforeFileOpen) {
  const auto fileSystem = validationFileSystem();
  fileSystem->openedFiles = 0;

  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto rowType = ROW({{"id", BIGINT()}, {"p", VARCHAR()}});
  const auto tableHandle =
      makePaimonTableHandle(rowType, {}, nullptr, /*partitionNames=*/{"p"});
  connector::ColumnHandleMap assignments{
      {"id",
       std::make_shared<PaimonColumnHandle>(
           "id",
           10,
           connector::FileColumnHandle::ColumnType::kRegular,
           BIGINT())},
      {"p",
       std::make_shared<PaimonColumnHandle>(
           "p",
           11,
           connector::FileColumnHandle::ColumnType::kPartitionKey,
           VARCHAR())}};
  auto source =
      ConnectorRegistry::tryGet(kPaimonConnectorId)
          ->createDataSource(
              ROW({{"id", BIGINT()}}), tableHandle, assignments, context.get());
  const auto files = makeFilePaths(1);
  const auto input = makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 2})});
  writeToFile(files[0]->getPath(), input);
  const std::string path = "paimon-table-test:" + files[0]->getPath();
  const auto makeSplit = [&](PaimonTableType type, const auto& partitions) {
    return makePaimonSplit(
        {path}, type, dwio::common::FileFormat::DWRF, partitions);
  };
  const std::unordered_map<std::string, std::optional<std::string>> good{
      {"p", std::nullopt}};
  VELOX_ASSERT_THROW(
      source->addSplit(makeSplit(PaimonTableType::kPrimaryKey, good)),
      "Paimon split table type does not match table handle");
  EXPECT_EQ(fileSystem->openedFiles, 0);

  VELOX_ASSERT_THROW(
      source->addSplit(makePaimonSplit(
          {path},
          PaimonTableType::kAppendOnly,
          dwio::common::FileFormat::DWRF,
          good,
          false)),
      "Paimon data-evolution column merging is not implemented yet: paimon_table");
  EXPECT_EQ(fileSystem->openedFiles, 0);

  auto dataSplit = makeSplit(PaimonTableType::kAppendOnly, good)->serialize();
  dataSplit["dataFiles"][0]["level"] = 1;
  source->addSplit(PaimonConnectorSplit::create(dataSplit));
  ContinueFuture future = ContinueFuture::makeEmpty();
  auto result = source->next(100, future);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(*result);
  test::assertEqualVectors(input, *result);
  EXPECT_GT(fileSystem->openedFiles, 0);
  auto end = source->next(100, future);
  ASSERT_TRUE(end.has_value());
  EXPECT_EQ(*end, nullptr);

  // Partition strings are converted when the scan binds a constant.
  const auto numericTable = makePaimonTableHandle(
      ROW({{"id", BIGINT()}, {"p", INTEGER()}}), {}, nullptr, {"p"});
  const auto numericColumn = std::make_shared<PaimonColumnHandle>(
      "p",
      11,
      connector::FileColumnHandle::ColumnType::kPartitionKey,
      INTEGER());
  auto numericSource = ConnectorRegistry::tryGet(kPaimonConnectorId)
                           ->createDataSource(
                               ROW({{"p", INTEGER()}}),
                               numericTable,
                               {{"p", numericColumn}},
                               context.get());
  const auto numericFiles = makeFilePaths(1);
  writeToFile(numericFiles[0]->getPath(), input);
  const std::string numericPath =
      "paimon-table-test:" + numericFiles[0]->getPath();
  const auto openedFiles = fileSystem->openedFiles.load();
  numericSource->addSplit(makePaimonSplit(
      {numericPath},
      PaimonTableType::kAppendOnly,
      dwio::common::FileFormat::DWRF,
      std::unordered_map<std::string, std::optional<std::string>>{
          {"p", "not-an-integer"}}));
  EXPECT_EQ(fileSystem->openedFiles, openedFiles);
  EXPECT_THROW(numericSource->next(100, future), std::exception);
  EXPECT_GT(fileSystem->openedFiles, openedFiles);
}

TEST_F(PaimonConnectorTest, queryValidationBeforeFileOpen) {
  const auto fileSystem = validationFileSystem();
  fileSystem->openedFiles = 0;
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto rowType = ROW({{"id", BIGINT()}, {"p", VARCHAR()}});
  const auto table = makePaimonTableHandle(rowType, {}, nullptr, {"p"});
  const auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  const auto idColumns = makePaimonColumnHandles(ROW({{"id", BIGINT()}}));
  VELOX_ASSERT_THROW(
      connector->createDataSource(
          ROW({{"missing", BIGINT()}}), table, idColumns, context.get()),
      "ColumnHandle is missing for output column: missing");
  VELOX_ASSERT_THROW(
      connector->createDataSource(
          ROW({{"id", BIGINT()}}),
          table,
          {{"id", idColumns.at("id")}, {"alias", idColumns.at("id")}},
          context.get()),
      "Cannot map from same table column to different outputs in table scan; a project node should be used instead: id");
  const auto requiredColumn = [&](const common::Subfield& path) {
    std::vector<common::Subfield> subfields;
    subfields.push_back(path.clone());
    return std::make_shared<PaimonColumnHandle>(
        "id",
        10,
        connector::FileColumnHandle::ColumnType::kRegular,
        BIGINT(),
        std::move(subfields));
  };
  for (const auto& path : {"missing", "p"}) {
    SCOPED_TRACE(path);
    VELOX_ASSERT_USER_THROW(
        connector->createDataSource(
            ROW({{"id", BIGINT()}}),
            table,
            {{"id", requiredColumn(common::Subfield(path))}},
            context.get()),
        "Required subfield does not match column name");
  }
  EXPECT_NE(
      connector->createDataSource(
          ROW({{"id", BIGINT()}}),
          table,
          {{"id", requiredColumn(common::Subfield("id"))}},
          context.get()),
      nullptr);
  {
    auto wire = table->serialize();
    wire["tableType"] = "PRIMARY_KEY";
    const auto primaryKeyTable =
        ISerializable::deserialize<PaimonTableHandle>(wire);
    EXPECT_NE(
        connector->createDataSource(
            ROW("id", BIGINT()), primaryKeyTable, idColumns, context.get()),
        nullptr);
  }
  const auto widerType = ROW({{"id", BIGINT()}, {"unused", VARBINARY()}});
  const auto widerTable = makePaimonTableHandle(widerType);
  EXPECT_NE(
      connector->createDataSource(
          ROW({{"id", BIGINT()}}), widerTable, idColumns, context.get()),
      nullptr);
  const auto binaryType = ROW({{"unused", VARBINARY()}});
  EXPECT_NE(
      connector->createDataSource(
          binaryType,
          widerTable,
          {{"unused",
            std::make_shared<PaimonColumnHandle>(
                "unused",
                11,
                connector::FileColumnHandle::ColumnType::kRegular,
                VARBINARY())}},
          context.get()),
      nullptr);
  const auto foreignTable = std::make_shared<PaimonTableHandle>(
      "other-connector",
      table->name(),
      table->schemaId(),
      table->rowType(),
      table->fieldIds(),
      table->tableType(),
      table->partitionKeys(),
      table->options());
  VELOX_ASSERT_USER_THROW(
      connector->createDataSource(
          ROW({{"id", BIGINT()}}), foreignTable, idColumns, context.get()),
      "Paimon table handle connector does not match connector instance");
  for (const auto& condition : {"p is null", "length(p) = 0"}) {
    SCOPED_TRACE(condition);
    const auto filteredTable = makePaimonTableHandle(
        rowType, {}, parseExpr(condition, rowType), {"p"});
    const ColumnHandleMap columns{
        {"id", idColumns.at("id")},
        {"p",
         std::make_shared<PaimonColumnHandle>(
             "p",
             11,
             connector::FileColumnHandle::ColumnType::kPartitionKey,
             VARCHAR())}};
    EXPECT_NE(
        connector->createDataSource(
            ROW({{"id", BIGINT()}}), filteredTable, columns, context.get()),
        nullptr);
    const auto filterColumnTable = makePaimonTableHandle(
        rowType,
        common::SubfieldFilters{},
        parseExpr(condition, rowType),
        {"p"},
        std::vector<connector::FileColumnHandlePtr>{
            std::static_pointer_cast<const connector::FileColumnHandle>(
                columns.at("p"))});
    EXPECT_NE(
        connector->createDataSource(
            ROW({{"id", BIGINT()}}),
            filterColumnTable,
            idColumns,
            context.get()),
        nullptr);
  }
  for (const auto* option :
       {connector::FileConfig::kIgnoreMissingFilesSession,
        connector::FileConfig::kFileColumnNamesReadAsLowerCaseSession}) {
    SCOPED_TRACE(option);
    const auto session = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{{option, "true"}});
    auto optionContext = queryContextBuilder(queryCtx.get())
                             .sessionProperties(session.get())
                             .build();
    EXPECT_NE(
        connector->createDataSource(
            ROW({{"id", BIGINT()}}), table, idColumns, optionContext.get()),
        nullptr);
  }
  EXPECT_EQ(fileSystem->openedFiles, 0);
}

TEST_F(PaimonConnectorTest, rejectsWrongFormatAndMissingFiles) {
  const auto rowType = ROW({{"id", BIGINT()}, {"value", BIGINT()}});
  const auto plan = makePaimonScanPlan(ROW({{"id", BIGINT()}}), rowType);
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    const auto input = makeRowVector(
        {"id", "value"},
        {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({2})});
    writeFile(files[0]->getPath(), input, format);
    const auto wrongFormat = format == dwio::common::FileFormat::DWRF
        ? dwio::common::FileFormat::PARQUET
        : dwio::common::FileFormat::DWRF;
    EXPECT_THROW(
        exec::test::AssertQueryBuilder(plan)
            .split(makePaimonSplit(
                {files[0]->getPath()},
                PaimonTableType::kAppendOnly,
                wrongFormat))
            .copyResults(pool()),
        std::exception);
  }
  const auto missing = makePaimonSplit({"/missing-paimon-file"});
  EXPECT_THROW(
      exec::test::AssertQueryBuilder(plan).split(missing).copyResults(pool()),
      std::exception);
}

TEST_F(PaimonConnectorTest, ordinaryColumnsIgnoreFeatureOptions) {
  const auto rowType = ROW({{"id", BIGINT()}});
  const std::vector<std::unordered_map<std::string, std::string>> optionSets{
      {},
      {{"deletion-vectors.enabled", "true"}},
      {{"data-evolution.enabled", "true"}, {"row-tracking.enabled", "true"}},
      {{"data-file.thin-mode", "true"}},
      {{"row-tracking.enabled", "true"}},
      {{"row-tracking.enabled", "false"}}};
  const auto expected =
      makeRowVector({"id"}, {makeFlatVector<int64_t>({4, 5})});
  const auto physical = makeRowVector(
      {"id", "_ROW_ID", "_SEQUENCE_NUMBER"},
      {expected->childAt(0),
       makeFlatVector<int64_t>({100, 101}),
       makeFlatVector<int64_t>({7, 7})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    for (const auto& input : {expected, physical}) {
      const auto files = makeFilePaths(1);
      writeFile(files[0]->getPath(), input, format);
      const auto split = makePaimonSplit(
          {files[0]->getPath()}, PaimonTableType::kAppendOnly, format);
      for (const auto& options : optionSets) {
        SCOPED_TRACE(
            options.empty()
                ? "no options"
                : options.begin()->first + "=" + options.begin()->second);
        const auto table =
            makePaimonTableHandle(rowType, {}, nullptr, {}, {}, options);
        const auto plan = exec::test::PlanBuilder()
                              .startTableScan()
                              .connectorId(kPaimonConnectorId)
                              .outputType(rowType)
                              .tableHandle(table)
                              .assignments(makePaimonColumnHandles(rowType))
                              .endTableScan()
                              .planNode();
        const auto copy = ISerializable::deserialize<core::PlanNode>(
            plan->serialize(), pool());
        for (const auto& scan : {plan, copy}) {
          exec::test::AssertQueryBuilder(scan)
              .connectorSessionProperty(
                  kPaimonConnectorId,
                  connector::FileConfig::kIgnoreMissingFilesSession,
                  "true")
              .split(split)
              .assertResults(expected);
        }
      }
    }
  }
}

TEST_F(PaimonConnectorTest, taskCancellationAndLimit) {
  const auto input = makeRowVector(
      {"id"}, {makeFlatVector<int64_t>(10000, [](auto row) { return row; })});
  const auto plan = makePaimonScanPlan(input->rowType());
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    writeFile(files[0]->getPath(), input, format);
    const auto split = makePaimonSplit(
        {files[0]->getPath()}, PaimonTableType::kAppendOnly, format);
    exec::CursorParameters params;
    params.planNode = plan;
    // Keep the scan in progress after the first batch, even if the producer
    // runs ahead of the cursor.
    params.bufferedBytes = 1024;
    params.queryConfigs[core::QueryConfig::kMaxOutputBatchRows] = "1024";
    auto cursor = exec::TaskCursor::create(params);
    cursor->task()->addSplit(plan->id(), exec::Split(split));
    cursor->task()->noMoreSplits(plan->id());
    ASSERT_TRUE(cursor->moveNext());
    ASSERT_TRUE(cursor->task()->isRunning());
    cursor->task()->requestCancel().wait();
    ASSERT_EQ(cursor->task()->state(), exec::TaskState::kCanceled);
    EXPECT_THROW(cursor->moveNext(), folly::OperationCancelled);
    cursor.reset();
    exec::test::TaskMonitor::waitForAllTasksDeleted();
    const auto limitPlan =
        makePaimonScan(input->rowType()).limit(0, 1, false).planNode();
    exec::test::AssertQueryBuilder(limitPlan).split(split).assertTypeAndNumRows(
        input->type(), 1);
    exec::test::TaskMonitor::waitForAllTasksDeleted();
  }
}

TEST_F(PaimonConnectorTest, takeoverPreparedSplit) {
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto rowType = ROW({{"id", BIGINT()}});
  const auto columns = makePaimonColumnHandles(rowType);
  const auto table = makePaimonTableHandle(rowType);
  const auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  auto makeSource = [&](const auto& handle) {
    return connector->createDataSource(rowType, handle, columns, context.get());
  };
  auto source = makeSource(table);
  const auto copy =
      ISerializable::deserialize<PaimonTableHandle>(table->serialize());
  auto preloaded = makeSource(copy);
  const auto paths = makeFilePaths(1);
  const auto input = makeRowVector({"id"}, {makeFlatVector<int64_t>({5, 6})});
  writeToFile(paths[0]->getPath(), input);
  preloaded->addSplit(makePaimonSplit({paths[0]->getPath()}));
  source->setFromDataSource(std::move(preloaded));
  ContinueFuture future = ContinueFuture::makeEmpty();
  auto result = source->next(100, future);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(*result);
  test::assertEqualVectors(input, *result);
}

TEST_F(PaimonConnectorTest, columnHandlesRoundTrip) {
  const auto rowType =
      ROW({{"value", BIGINT()}, {"id", BIGINT()}, {"p", VARCHAR()}});
  const auto id = std::make_shared<PaimonColumnHandle>(
      "id", 7, connector::FileColumnHandle::ColumnType::kRegular, BIGINT());
  const auto value = std::make_shared<PaimonColumnHandle>(
      "value", 41, connector::FileColumnHandle::ColumnType::kRegular, BIGINT());
  const auto partition = std::make_shared<PaimonColumnHandle>(
      "p",
      99,
      connector::FileColumnHandle::ColumnType::kPartitionKey,
      VARCHAR());
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("value"),
      std::make_unique<common::BigintRange>(10, 30, false));
  const auto table = std::make_shared<PaimonTableHandle>(
      kPaimonConnectorId,
      "orders",
      23,
      rowType,
      std::vector<int32_t>{41, 7, 99},
      PaimonTableType::kAppendOnly,
      std::vector<std::string>{"p"},
      std::unordered_map<std::string, std::string>{},
      std::move(filters),
      parseExpr("id < 3 and p = 'keep'", rowType),
      std::vector<connector::FileColumnHandlePtr>{value, partition});
  const auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(kPaimonConnectorId)
                        .outputType(ROW({{"selected_id", BIGINT()}}))
                        .tableHandle(table)
                        .assignments({{"selected_id", id}})
                        .endTableScan()
                        .planNode();
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  const auto input = makeRowVector(
      {"value", "id"},
      {makeFlatVector<int64_t>({10, 20, 30}),
       makeFlatVector<int64_t>({1, 2, 3})});
  const auto expected =
      makeRowVector({"selected_id"}, {makeFlatVector<int64_t>({1, 2, 1, 2})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    for (const auto& file : files) {
      writeFile(file->getPath(), input, format);
    }
    const auto split = makePaimonSplit(
        {files[0]->getPath(), files[1]->getPath()},
        PaimonTableType::kAppendOnly,
        format,
        {{"p", "keep"}});
    for (const auto& scan : {plan, copy}) {
      exec::test::AssertQueryBuilder(scan).split(split).assertResults(expected);
    }
  }
}

TEST_F(PaimonConnectorTest, nestedColumnHandleRoundTrip) {
  const auto nestedType = ROW({{"keep", BIGINT()}, {"unused", VARCHAR()}});
  const auto rowType = ROW({{"r", nestedType}});
  const auto table = makePaimonTableHandle(rowType);
  std::vector<common::Subfield> paths;
  paths.emplace_back("r.keep");
  const auto column = std::make_shared<PaimonColumnHandle>(
      "r",
      10,
      connector::FileColumnHandle::ColumnType::kRegular,
      nestedType,
      std::move(paths));
  const auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(kPaimonConnectorId)
                        .outputType(rowType)
                        .tableHandle(table)
                        .assignments({{"r", column}})
                        .endTableScan()
                        .project({"r.keep"})
                        .planNode();
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  const auto values = makeFlatVector<int64_t>({4, 9});
  const auto input = makeRowVector(
      {"r"},
      {makeRowVector(
          {"keep", "unused"},
          {values, makeFlatVector<std::string>({"a", "b"})})});
  const auto expected = makeRowVector({values});
  for (auto format : fileFormats()) {
    const auto files = makeFilePaths(1);
    writeFile(files[0]->getPath(), input, format);
    for (const auto& scan : {plan, copy}) {
      exec::test::AssertQueryBuilder(scan)
          .split(makePaimonSplit(
              {files[0]->getPath()}, PaimonTableType::kAppendOnly, format))
          .assertResults(expected);
    }
  }
}

TEST_F(PaimonConnectorTest, extractedColumnHandleRoundTrip) {
  const auto mapType = MAP(VARCHAR(), BIGINT());
  const auto table = makePaimonTableHandle(ROW({{"m", mapType}}));
  const auto column = std::make_shared<PaimonColumnHandle>(
      "m",
      10,
      connector::FileColumnHandle::ColumnType::kRegular,
      mapType,
      std::vector<common::Subfield>{},
      std::vector<connector::NamedExtraction>{
          {"size",
           {connector::ExtractionPathElement::simple(
               connector::ExtractionStep::kSize)},
           BIGINT()}});
  const auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(kPaimonConnectorId)
                        .outputType(ROW({{"n", BIGINT()}}))
                        .tableHandle(table)
                        .assignments({{"n", column}})
                        .endTableScan()
                        .planNode();
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  const auto input = makeRowVector(
      {"m"},
      {makeMapVector<std::string, int64_t>(
          {{{"a", 1}, {"b", 2}}, {{"c", 3}}, {}})});
  const auto expected =
      makeRowVector({"n"}, {makeFlatVector<int64_t>({2, 1, 0})});
  for (auto format : fileFormats()) {
    const auto files = makeFilePaths(1);
    writeFile(files[0]->getPath(), input, format);
    for (const auto& scan : {plan, copy}) {
      auto query = exec::test::AssertQueryBuilder(scan);
      query.split(makePaimonSplit(
          {files[0]->getPath()}, PaimonTableType::kAppendOnly, format));
      if (format == dwio::common::FileFormat::PARQUET) {
        VELOX_ASSERT_THROW(
            query.assertResults(expected),
            "Parquet reader does not support extraction pushdown");
      } else {
        query.assertResults(expected);
      }
    }
  }
}

TEST_F(PaimonConnectorTest, remainingFilterLambdaRoundTrip) {
  const auto rowType = ROW({{"id", BIGINT()}, {"value", BIGINT()}});

  // The lambda parameter 'id' shadows the table column, while 'value' is
  // a captured, unprojected table column.
  const auto shadowing = makePaimonTableHandle(
      rowType,
      {},
      parseExpr("any_match(array[20], id -> id = value)", rowType));
  const auto plan =
      exec::test::PlanBuilder()
          .startTableScan()
          .connectorId(kPaimonConnectorId)
          .outputType(ROW({{"id", BIGINT()}}))
          .tableHandle(shadowing)
          .assignments(makePaimonColumnHandles(ROW({{"id", BIGINT()}})))
          .endTableScan()
          .planNode();
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  const auto input = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  const auto expected = makeRowVector({"id"}, {makeFlatVector<int64_t>({2})});
  const auto files = makeFilePaths(1);
  writeToFile(files[0]->getPath(), input);
  for (const auto& scan : {plan, copy}) {
    exec::test::AssertQueryBuilder(scan)
        .split(makePaimonSplit({files[0]->getPath()}))
        .assertResults(expected);
  }
}

TEST_F(PaimonConnectorTest, subfieldFilterRoundTrip) {
  const auto rowType = ROW({{"id", BIGINT()}, {"r", ROW({{"x", BIGINT()}})}});
  const auto outputType = ROW({{"id", BIGINT()}});
  const auto input = makeRowVector(
      {"id", "r"},
      {makeFlatVector<int64_t>({1, 2}),
       makeRowVector(
           {"x"}, {makeNullableFlatVector<int64_t>({std::nullopt, 20})})});
  const auto expected = makeRowVector({"id"}, {makeFlatVector<int64_t>({1})});
  const auto files = makeFilePaths(1);
  writeToFile(files[0]->getPath(), input);

  for (bool useRemainingFilter : {false, true}) {
    SCOPED_TRACE(useRemainingFilter);
    common::SubfieldFilters filters;
    core::TypedExprPtr remainingFilter;
    if (useRemainingFilter) {
      remainingFilter = parseExpr("r.x is null", rowType);
    } else {
      filters.emplace(
          common::Subfield("r.x"), std::make_unique<common::IsNull>());
    }
    const auto table =
        makePaimonTableHandle(rowType, std::move(filters), remainingFilter);
    const auto plan = exec::test::PlanBuilder()
                          .startTableScan()
                          .connectorId(kPaimonConnectorId)
                          .outputType(outputType)
                          .tableHandle(table)
                          .assignments(makePaimonColumnHandles(outputType))
                          .endTableScan()
                          .planNode();
    const auto copy =
        ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
    for (const auto& scan : {plan, copy}) {
      exec::test::AssertQueryBuilder(scan)
          .split(makePaimonSplit({files[0]->getPath()}))
          .assertResults(expected);
    }
  }
}

TEST_F(PaimonConnectorTest, partitionDateEncodingRoundTrip) {
  const auto rowType = ROW({{"id", BIGINT()}, {"day", DATE()}});
  const auto table = makePaimonTableHandle(rowType, {}, nullptr, {"day"});
  const auto column = std::make_shared<PaimonColumnHandle>(
      "day",
      11,
      connector::FileColumnHandle::ColumnType::kPartitionKey,
      DATE(),
      std::vector<common::Subfield>{},
      std::vector<connector::NamedExtraction>{},
      /*partitionDateValueDaysSinceEpoch=*/true);
  const auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(kPaimonConnectorId)
                        .outputType(ROW({{"day", DATE()}}))
                        .tableHandle(table)
                        .assignments({{"day", column}})
                        .endTableScan()
                        .planNode();
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  const auto input = makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 2})});
  const auto expected =
      makeRowVector({"day"}, {makeFlatVector<int32_t>({1234, 1234}, DATE())});
  const auto files = makeFilePaths(1);
  writeToFile(files[0]->getPath(), input);
  for (const auto& scan : {plan, copy}) {
    exec::test::AssertQueryBuilder(scan)
        .split(makePaimonSplit(
            {files[0]->getPath()},
            PaimonTableType::kAppendOnly,
            dwio::common::FileFormat::DWRF,
            {{"day", "1234"}}))
        .assertResults(expected);
  }
}

TEST_F(PaimonConnectorTest, requiresPaimonTableHandle) {
  const auto rowType = ROW({{"id", BIGINT()}});
  const auto columns = makePaimonColumnHandles(rowType);
  const auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto fileSystem = validationFileSystem();
  fileSystem->openedFiles = 0;
  try {
    connector->createDataSource(rowType, nullptr, columns, context.get());
    FAIL() << "Expected a null table handle check failure";
  } catch (const VeloxRuntimeError& exception) {
    EXPECT_EQ(exception.failingExpression(), "tableHandle != nullptr");
  }
  const auto tableHandle = std::make_shared<HiveTableHandle>(
      kPaimonConnectorId,
      "paimon_table",
      common::SubfieldFilters{},
      nullptr,
      rowType);
  VELOX_ASSERT_THROW(
      connector->createDataSource(rowType, tableHandle, columns, context.get()),
      "Failed to cast from");
  const auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(kPaimonConnectorId)
                        .outputType(rowType)
                        .tableHandle(tableHandle)
                        .assignments(columns)
                        .endTableScan()
                        .planNode();
  for (auto tableType :
       {PaimonTableType::kAppendOnly, PaimonTableType::kPrimaryKey}) {
    SCOPED_TRACE(fmt::format("table type {}", tableType));
    const auto split = makePaimonSplit(
        {"paimon-table-test:/not-opened"},
        tableType,
        dwio::common::FileFormat::DWRF);
    VELOX_ASSERT_THROW(
        exec::test::AssertQueryBuilder(plan).split(split).copyResults(pool()),
        "Failed to cast from");
    exec::test::TaskMonitor::waitForAllTasksDeleted();
  }
  EXPECT_EQ(fileSystem->openedFiles, 0);
}

// Append-only splits that are not raw convertible come from data evolution,
// whose files must be merged by column.
TEST_F(PaimonConnectorTest, rejectsAppendOnlyNonRawConvertible) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto vectors = makeVectors(rowType, 1, 10);

  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);

  auto split = makePaimonSplit(
      {filePaths[0]->getPath()},
      PaimonTableType::kAppendOnly,
      dwio::common::FileFormat::DWRF,
      /*partitionKeys=*/{},
      /*rawConvertible=*/false);

  auto plan = makePaimonScanPlan(rowType);

  VELOX_ASSERT_THROW_CODE(
      exec::test::AssertQueryBuilder(plan).split(split).copyResults(pool()),
      error_code::kNotImplemented,
      "Paimon data-evolution column merging is not implemented yet: paimon_table");
}

// A PK split cannot bypass the append table definition, even when Raw.
TEST_F(PaimonConnectorTest, rejectsPrimaryKeySplits) {
  const auto rowType = ROW({"c0"}, {BIGINT()});
  const auto plan = makePaimonScanPlan(rowType);
  const auto fileSystem = validationFileSystem();
  fileSystem->openedFiles = 0;
  for (auto rawConvertible : {true, false}) {
    SCOPED_TRACE(rawConvertible);
    const auto split = makePaimonSplit(
        {"paimon-table-test:/not-opened"},
        PaimonTableType::kPrimaryKey,
        dwio::common::FileFormat::DWRF,
        /*partitionKeys=*/{},
        rawConvertible);
    VELOX_ASSERT_USER_THROW(
        exec::test::AssertQueryBuilder(plan).split(split).copyResults(pool()),
        "Paimon split table type does not match table handle");
    exec::test::TaskMonitor::waitForAllTasksDeleted();
  }
  EXPECT_EQ(fileSystem->openedFiles, 0);
}

// Changelog files serve streaming reads, so a scan rejects them before opening
// any file.
TEST_F(PaimonConnectorTest, rejectsChangelogFiles) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto plan = makePaimonScanPlan(
      rowType,
      /*dataColumns=*/nullptr,
      /*subfieldFilters=*/{},
      /*remainingFilter=*/nullptr,
      PaimonTableType::kPrimaryKey);
  const auto fileSystem = validationFileSystem();
  fileSystem->openedFiles = 0;
  PaimonDataFile changelogFile;
  changelogFile.path = "paimon-table-test:/changelog-not-opened";
  changelogFile.fileFormat = dwio::common::FileFormat::DWRF;
  changelogFile.type = PaimonDataFile::Type::kChangelog;
  changelogFile.source = PaimonDataFile::Source::kCompact;
  VELOX_ASSERT_USER_THROW(
      exec::test::AssertQueryBuilder(plan)
          .split(makePaimonSplit({changelogFile}, PaimonTableType::kPrimaryKey))
          .copyResults(pool()),
      "Paimon reader does not support changelog files: paimon-table-test:/changelog-not-opened");
  EXPECT_EQ(fileSystem->openedFiles, 0);
}

// Primary-key files lead with the key, sequence number and row kind columns,
// and thin-mode files drop the key columns. A raw read binds the table columns
// by name, so a filter on 'id' reads the table column, not '_KEY_id'.
TEST_F(PaimonConnectorTest, primaryKeyFileLayout) {
  const auto rowType = ROW({{"id", BIGINT()}, {"value", VARCHAR()}});
  const auto ids = makeFlatVector<int64_t>({1, 2, 3});
  const auto keys = makeFlatVector<int64_t>({-1, -2, -3});
  const auto values = makeFlatVector<std::string>({"a", "b", "c"});
  const auto sequenceNumbers = makeFlatVector<int64_t>({7, 8, 9});
  const auto valueKinds = makeFlatVector<int8_t>({0, 0, 2});
  struct TestParam {
    std::string layout;
    RowVectorPtr input;

    std::string debugString() const {
      return layout;
    }
  };
  const std::vector<TestParam> testSettings{
      {"full",
       makeRowVector(
           {"_KEY_id", "_SEQUENCE_NUMBER", "_VALUE_KIND", "id", "value"},
           {keys, sequenceNumbers, valueKinds, ids, values})},
      {"thin",
       makeRowVector(
           {"_SEQUENCE_NUMBER", "_VALUE_KIND", "id", "value"},
           {sequenceNumbers, valueKinds, ids, values})},
  };
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("id"),
      std::make_unique<common::BigintRange>(2, 3, false));
  const auto plan = makePaimonScanPlan(
      ROW("value", VARCHAR()),
      rowType,
      std::move(filters),
      /*remainingFilter=*/nullptr,
      PaimonTableType::kPrimaryKey);
  const auto expected =
      makeRowVector({makeFlatVector<std::string>({"b", "c"})});
  for (const auto& testData : testSettings) {
    for (auto format : fileFormats()) {
      SCOPED_TRACE(
          fmt::format(
              "{}, format {}",
              testData.debugString(),
              dwio::common::FileFormatName::toName(format)));
      const auto files = makeFilePaths(1);
      writeFile(files[0]->getPath(), testData.input, format);
      exec::test::AssertQueryBuilder(plan)
          .split(makePaimonSplit(
              {files[0]->getPath()}, PaimonTableType::kPrimaryKey, format))
          .assertResults({expected});
    }
  }
}

// E2E test: read a partitioned primary-key table through raw splits that went
// through serialization, as a worker receives them. With deletion vectors
// enabled, Paimon reads the files of a bucket above level 0 without merging,
// and the deletion vector of each file drops the rows whose keys newer files
// hold. Files written before 'data-file.thin-mode' was set keep their _KEY_
// columns, so the files of a split can differ in layout.
TEST_F(PaimonConnectorTest, primaryKeyMultipleSplits) {
  // The primary key is (dt, id) and dt is the partition column, so files key
  // their rows by 'id' alone.
  const auto rowType =
      ROW({{"dt", VARCHAR()}, {"id", BIGINT()}, {"value", VARCHAR()}});
  struct KeyValueFile {
    std::vector<int64_t> ids;
    std::vector<std::string> values;
    // _VALUE_KIND of each row: 0 for INSERT, 2 for UPDATE_AFTER.
    std::vector<int8_t> valueKinds;
    // _SEQUENCE_NUMBER of the first row. Each later row takes the next number.
    int64_t minSequenceNumber;
    int32_t level;
    // Whether the file lacks the _KEY_id column.
    bool thin;
    // Rows whose keys newer files hold.
    std::vector<int64_t> deletedPositions;
  };
  // Bucket 0 of a partition, which one split reads.
  struct Bucket {
    std::string dt;
    std::vector<KeyValueFile> files;
  };
  const std::vector<Bucket> buckets{
      {.dt = "2026-11-21",
       .files =
           {{.ids = {1, 2, 3, 4},
             .values = {"a1", "b1", "c1", "d1"},
             .valueKinds = {0, 0, 0, 0},
             .minSequenceNumber = 1,
             .level = 5,
             .thin = false,
             .deletedPositions = {1, 3}},
            {.ids = {2, 4, 5},
             .values = {"b2", "d2", "e1"},
             .valueKinds = {2, 2, 0},
             .minSequenceNumber = 5,
             .level = 4,
             .thin = true,
             .deletedPositions = {}}}},
      {.dt = "2026-11-22",
       .files =
           {{.ids = {2, 3, 4},
             .values = {"g1", "h1", "i1"},
             .valueKinds = {0, 0, 0},
             .minSequenceNumber = 1,
             .level = 5,
             .thin = false,
             .deletedPositions = {2}},
            {.ids = {1, 4},
             .values = {"f1", "i2"},
             .valueKinds = {0, 2},
             .minSequenceNumber = 4,
             .level = 1,
             .thin = false,
             .deletedPositions = {}}}},
  };
  // The latest row of each key with 2 <= id <= 4.
  const auto expected = makeRowVector(
      {"dt", "id", "value"},
      {makeFlatVector<std::string>(
           {"2026-11-21",
            "2026-11-21",
            "2026-11-21",
            "2026-11-22",
            "2026-11-22",
            "2026-11-22"}),
       makeFlatVector<int64_t>({3, 2, 4, 2, 3, 4}),
       makeFlatVector<std::string>({"c1", "b2", "d2", "g1", "h1", "i2"})});

  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("id"),
      std::make_unique<common::BigintRange>(2, 4, false));
  const auto table = makePaimonTableHandle(
      rowType,
      std::move(filters),
      /*remainingFilter=*/nullptr,
      /*partitionNames=*/{"dt"},
      /*filterColumns=*/{},
      {{"deletion-vectors.enabled", "true"}, {"data-file.thin-mode", "true"}},
      PaimonTableType::kPrimaryKey);
  auto assignments = makePaimonColumnHandles(rowType);
  assignments["dt"] = std::make_shared<PaimonColumnHandle>(
      "dt",
      10,
      connector::FileColumnHandle::ColumnType::kPartitionKey,
      VARCHAR());
  const auto plan = ISerializable::deserialize<core::PlanNode>(
      exec::test::PlanBuilder()
          .startTableScan()
          .connectorId(kPaimonConnectorId)
          .outputType(rowType)
          .tableHandle(table)
          .assignments(assignments)
          .endTableScan()
          .planNode()
          ->serialize(),
      pool());
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    std::vector<std::shared_ptr<exec::test::TempFilePath>> filePaths;
    std::vector<std::shared_ptr<connector::ConnectorSplit>> splits;
    for (const auto& bucket : buckets) {
      std::vector<PaimonDataFile> dataFiles;
      for (const auto& file : bucket.files) {
        const auto numRows = static_cast<vector_size_t>(file.ids.size());
        std::vector<std::string> names{
            "_SEQUENCE_NUMBER", "_VALUE_KIND", "dt", "id", "value"};
        std::vector<VectorPtr> columns{
            makeFlatVector<int64_t>(
                numRows,
                [&](auto row) { return file.minSequenceNumber + row; }),
            makeFlatVector<int8_t>(file.valueKinds),
            makeFlatVector<std::string>(
                std::vector<std::string>(numRows, bucket.dt)),
            makeFlatVector<int64_t>(file.ids),
            makeFlatVector<std::string>(file.values)};
        if (!file.thin) {
          names.insert(names.begin(), "_KEY_id");
          columns.insert(columns.begin(), makeFlatVector<int64_t>(file.ids));
        }
        filePaths.push_back(exec::test::TempFilePath::create());
        writeFile(
            filePaths.back()->getPath(), makeRowVector(names, columns), format);

        PaimonDataFile dataFile;
        dataFile.path = filePaths.back()->getPath();
        dataFile.fileFormat = format;
        dataFile.rowCount = numRows;
        dataFile.level = file.level;
        dataFile.minSequenceNumber = file.minSequenceNumber;
        dataFile.maxSequenceNumber = file.minSequenceNumber + numRows - 1;
        dataFile.source = PaimonDataFile::Source::kCompact;
        if (!file.deletedPositions.empty()) {
          dataFile.deletionFile =
              writeDeletionVectors({file.deletedPositions}, /*version=*/1)[0];
        }
        dataFiles.push_back(std::move(dataFile));
      }
      const PaimonConnectorSplit split(
          kPaimonConnectorId,
          /*snapshotId=*/3,
          PaimonTableType::kPrimaryKey,
          dataFiles,
          {{"dt", bucket.dt}},
          /*tableBucketNumber=*/0);
      splits.push_back(PaimonConnectorSplit::create(split.serialize()));
    }
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(
        {expected});
  }
}

// Paimon packs the deletion vectors of several data files into one index
// file, in either frame version, and append-only tables use them too.
TEST_F(PaimonConnectorTest, deletionVectorFrames) {
  const auto rowType = ROW("c0", BIGINT());
  const auto first = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})});
  const auto second = makeRowVector({makeFlatVector<int64_t>({10, 11, 12})});
  const auto plan = makePaimonScanPlan(rowType);
  const auto expected = makeRowVector({makeFlatVector<int64_t>({0, 2, 4, 11})});
  for (int32_t version : {1, 2}) {
    const auto deletionFiles = writeDeletionVectors({{1, 3}, {0, 2}}, version);
    ASSERT_GT(deletionFiles[1].offset, deletionFiles[0].offset);
    for (auto format : fileFormats()) {
      SCOPED_TRACE(
          fmt::format(
              "version {}, format {}",
              version,
              dwio::common::FileFormatName::toName(format)));
      const auto files = makeFilePaths(2);
      writeFile(files[0]->getPath(), first, format);
      writeFile(files[1]->getPath(), second, format);
      std::vector<PaimonDataFile> dataFiles(2);
      dataFiles[0].path = files[0]->getPath();
      dataFiles[0].fileFormat = format;
      dataFiles[0].rowCount = first->size();
      dataFiles[0].deletionFile = deletionFiles[0];
      dataFiles[1].path = files[1]->getPath();
      dataFiles[1].fileFormat = format;
      dataFiles[1].rowCount = second->size();
      dataFiles[1].deletionFile = deletionFiles[1];
      exec::test::AssertQueryBuilder(plan)
          .split(makePaimonSplit(dataFiles))
          .assertResults({expected});
    }
  }
}

// Paimon 0.9 records no cardinality for its deletion vectors.
TEST_F(PaimonConnectorTest, deletionVectorWithoutCardinality) {
  const auto rowType = ROW("c0", BIGINT());
  const auto input = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})});
  const auto files = makeFilePaths(1);
  writeToFile(files[0]->getPath(), input);
  const auto plan = makePaimonScanPlan(rowType);
  const auto expected = makeRowVector({makeFlatVector<int64_t>({0, 2, 4})});
  for (int32_t version : {1, 2}) {
    SCOPED_TRACE(fmt::format("version {}", version));
    const auto deletionFile = writeDeletionVectors({{1, 3}}, version)[0];
    PaimonDataFile dataFile;
    dataFile.path = files[0]->getPath();
    dataFile.fileFormat = dwio::common::FileFormat::DWRF;
    dataFile.rowCount = input->size();
    dataFile.deletionFile = PaimonDeletionFile(
        deletionFile.path,
        deletionFile.offset,
        deletionFile.length,
        /*cardinality=*/std::nullopt);
    exec::test::AssertQueryBuilder(plan)
        .split(makePaimonSplit({dataFile}))
        .assertResults({expected});
  }
}

TEST_F(PaimonConnectorTest, deletionVectorFrameErrors) {
  const auto rowType = ROW("c0", BIGINT());
  const auto input = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  const auto files = makeFilePaths(1);
  writeToFile(files[0]->getPath(), input);
  const auto plan = makePaimonScanPlan(rowType);
  const auto scan = [&](const PaimonDeletionFile& deletionFile) {
    PaimonDataFile dataFile;
    dataFile.path = files[0]->getPath();
    dataFile.fileFormat = dwio::common::FileFormat::DWRF;
    dataFile.rowCount = input->size();
    dataFile.deletionFile = deletionFile;
    exec::test::AssertQueryBuilder(plan)
        .split(makePaimonSplit({dataFile}))
        .copyResults(pool());
  };
  for (int32_t version : {1, 2}) {
    SCOPED_TRACE(fmt::format("version {}", version));
    const auto deletionFile = writeDeletionVectors({{1}}, version)[0];
    VELOX_ASSERT_THROW(
        scan(PaimonDeletionFile(
            deletionFile.path,
            deletionFile.offset,
            deletionFile.length + 1,
            deletionFile.cardinality)),
        "Paimon deletion-vector frame length does not match the deletion file length");
    exec::test::TaskMonitor::waitForAllTasksDeleted();
    VELOX_ASSERT_THROW(
        scan(PaimonDeletionFile(
            deletionFile.path,
            deletionFile.offset,
            deletionFile.length,
            /*cardinality=*/2)),
        fmt::format(
            "(1 vs. 2) Paimon deletion-vector cardinality mismatch: {}",
            deletionFile.path));
    exec::test::TaskMonitor::waitForAllTasksDeleted();
  }

  // A version byte, then a frame of length 8 whose magic number 0x12345678
  // matches neither frame version.
  const std::string frame("\1\0\0\0\x08\x12\x34\x56\x78\0\0\0\0\0\0\0\0", 17);
  const auto badMagicFile = exec::test::TempFilePath::create();
  {
    std::ofstream out(badMagicFile->getPath(), std::ios::binary);
    out.write(frame.data(), static_cast<std::streamsize>(frame.size()));
  }
  VELOX_ASSERT_THROW(
      scan(PaimonDeletionFile(
          badMagicFile->getPath(),
          /*offset=*/1,
          /*length=*/8,
          /*cardinality=*/1)),
      "Paimon deletion vector has an unknown magic number: 305419896");
}

// Corrupt bitmaps fail the scan rather than read past the bitmap or delete
// rows of another container.
TEST_F(PaimonConnectorTest, deletionVectorBitmapErrors) {
  const auto rowType = ROW("c0", BIGINT());
  const auto input = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  const auto files = makeFilePaths(1);
  writeToFile(files[0]->getPath(), input);
  const auto plan = makePaimonScanPlan(rowType);
  const auto scan = [&](const std::string& bitmap) {
    PaimonDataFile dataFile;
    dataFile.path = files[0]->getPath();
    dataFile.fileFormat = dwio::common::FileFormat::DWRF;
    dataFile.rowCount = input->size();
    dataFile.deletionFile =
        writeDeletionVectorFrames({bitmap}, /*version=*/1)[0];
    exec::test::AssertQueryBuilder(plan)
        .split(makePaimonSplit({dataFile}))
        .copyResults(pool());
  };
  const auto append = [](std::string& data, auto value) {
    value = folly::Endian::little(value);
    data.append(reinterpret_cast<const char*>(&value), sizeof(value));
  };

  // A bitmap without run containers that claims 2^30 containers, whose 4-byte
  // headers overflow a 32-bit size: the cookie, the container count and one
  // key and cardinality.
  std::string tooManyContainers;
  append(tooManyContainers, uint32_t{12'346});
  append(tooManyContainers, uint32_t{1} << 30);
  append(tooManyContainers, uint32_t{0});
  VELOX_ASSERT_THROW(
      scan(tooManyContainers),
      "Roaring bitmap container count exceeds the 16-bit key space: 1073741824");
  exec::test::TaskMonitor::waitForAllTasksDeleted();

  // A bitmap of one run container whose run of 2 values starts at 65535: the
  // cookie, the run flags, the key, the cardinality minus one, the number of
  // runs, and the start and length minus one of the run.
  std::string runPastContainer;
  append(runPastContainer, uint32_t{12'347});
  append(runPastContainer, uint8_t{1});
  append(runPastContainer, uint16_t{0});
  append(runPastContainer, uint16_t{1});
  append(runPastContainer, uint16_t{1});
  append(runPastContainer, uint16_t{65'535});
  append(runPastContainer, uint16_t{1});
  VELOX_ASSERT_THROW(
      scan(runPastContainer), "Roaring run extends past its container.");
}

// Batches of different sizes share one delete bitmap, so the deletes of a
// batch must not leak into later batches or files.
TEST_F(PaimonConnectorTest, deletionVectorAcrossBatches) {
  const auto rowType = ROW("c0", BIGINT());
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  const auto table = makePaimonTableHandle(rowType);
  const auto columns = makePaimonColumnHandles(rowType);
  const auto firstInput = makeRowVector(
      {makeFlatVector<int64_t>(108, [](auto row) { return row; })});
  const auto secondInput = makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return 1'000 + row; })});
  // Deletes row 1 of the first 8-row batch and row 50 of the next 100 rows.
  const auto deletionFile = writeDeletionVectors({{1, 58}}, /*version=*/1)[0];
  const auto keptRows = [](int64_t begin, int64_t end, int64_t deleted) {
    std::vector<int64_t> rows;
    for (auto row = begin; row < end; ++row) {
      if (row != deleted) {
        rows.push_back(row);
      }
    }
    return rows;
  };
  const std::vector<RowVectorPtr> expectedBatches{
      makeRowVector({makeFlatVector<int64_t>(keptRows(0, 8, 1))}),
      makeRowVector({makeFlatVector<int64_t>(keptRows(8, 108, 58))}),
      secondInput,
  };
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    writeFile(files[0]->getPath(), firstInput, format);
    writeFile(files[1]->getPath(), secondInput, format);
    std::vector<PaimonDataFile> dataFiles(2);
    dataFiles[0].path = files[0]->getPath();
    dataFiles[0].fileFormat = format;
    dataFiles[0].rowCount = firstInput->size();
    dataFiles[0].deletionFile = deletionFile;
    dataFiles[1].path = files[1]->getPath();
    dataFiles[1].fileFormat = format;
    dataFiles[1].rowCount = secondInput->size();

    auto source =
        connector->createDataSource(rowType, table, columns, context.get());
    source->addSplit(makePaimonSplit(dataFiles));
    ContinueFuture future = ContinueFuture::makeEmpty();
    const std::vector<uint64_t> batchSizes{8, 100, 100};
    for (size_t i = 0; i < batchSizes.size(); ++i) {
      auto result = source->next(batchSizes[i], future);
      ASSERT_TRUE(result.has_value());
      ASSERT_NE(*result, nullptr);
      test::assertEqualVectors(expectedBatches[i], *result);
    }
    auto end = source->next(100, future);
    ASSERT_TRUE(end.has_value());
    EXPECT_EQ(*end, nullptr);
  }
}

// Samples the rows that deletion vectors keep, in every file of a split. The
// first file is sampled with a deletion bitmap and the second without one.
TEST_F(PaimonConnectorTest, randomSample) {
  random::setSeed(42);
  constexpr int64_t kNumFileRows{10'000};
  constexpr double kSampleRate{0.5};
  const auto firstInput = makeRowVector(
      {makeFlatVector<int64_t>(kNumFileRows, [](auto row) { return row; })});
  const auto secondInput = makeRowVector({makeFlatVector<int64_t>(
      kNumFileRows, [](auto row) { return kNumFileRows + row; })});
  // Keeps every fourth row of the first file.
  std::vector<int64_t> deletedPositions;
  for (int64_t row = 0; row < kNumFileRows; ++row) {
    if (row % 4 != 0) {
      deletedPositions.emplace_back(row);
    }
  }
  const auto deletionFile =
      writeDeletionVectors({deletedPositions}, /*version=*/1)[0];
  const auto rowType = ROW("c0", BIGINT());
  const auto plan =
      exec::test::PlanBuilder()
          .startTableScan()
          .connectorId(kPaimonConnectorId)
          .outputType(rowType)
          .tableHandle(makeSampledTableHandle(rowType, kSampleRate))
          .assignments(makePaimonColumnHandles(rowType))
          .endTableScan()
          .planNode();
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    writeFile(files[0]->getPath(), firstInput, format);
    writeFile(files[1]->getPath(), secondInput, format);
    std::vector<PaimonDataFile> dataFiles(2);
    dataFiles[0].path = files[0]->getPath();
    dataFiles[0].fileFormat = format;
    dataFiles[0].rowCount = kNumFileRows;
    dataFiles[0].deletionFile = deletionFile;
    dataFiles[1].path = files[1]->getPath();
    dataFiles[1].fileFormat = format;
    dataFiles[1].rowCount = kNumFileRows;

    const auto result = exec::test::AssertQueryBuilder(plan)
                            .split(makePaimonSplit(dataFiles))
                            .copyResults(pool());
    const auto* values = result->childAt(0)->as<SimpleVector<int64_t>>();
    int64_t numFirstFileRows{0};
    for (vector_size_t i = 0; i < result->size(); ++i) {
      const auto value = values->valueAt(i);
      if (i > 0) {
        ASSERT_LT(values->valueAt(i - 1), value);
      }
      if (value < kNumFileRows) {
        ASSERT_EQ(value % 4, 0) << "Deleted row " << value;
        ++numFirstFileRows;
      }
    }
    // Allows 10%, which is 5 standard deviations for the first file and 10 for
    // the second.
    const double expectedFirstFileRows = kSampleRate * kNumFileRows / 4;
    const double expectedSecondFileRows = kSampleRate * kNumFileRows;
    EXPECT_NEAR(
        numFirstFileRows, expectedFirstFileRows, 0.1 * expectedFirstFileRows);
    EXPECT_NEAR(
        result->size() - numFirstFileRows,
        expectedSecondFileRows,
        0.1 * expectedSecondFileRows);
  }
}

// Sampling skips whole stripes without reading them, so the deletion bitmap of
// the next batch must start at the first row after the skipped stripes.
TEST_F(PaimonConnectorTest, randomSampleSkipsStripes) {
  random::setSeed(42);
  constexpr int32_t kNumStripes{1'000};
  constexpr int32_t kNumStripeRows{100};
  constexpr int64_t kNumRows{kNumStripes * kNumStripeRows};
  // Skips a stripe with probability (1 - 0.02)^100, about 13%.
  constexpr double kSampleRate{0.02};
  std::vector<RowVectorPtr> stripes;
  stripes.reserve(kNumStripes);
  for (int32_t stripe = 0; stripe < kNumStripes; ++stripe) {
    stripes.emplace_back(makeRowVector(
        {makeFlatVector<int64_t>(kNumStripeRows, [stripe](auto row) {
          return stripe * kNumStripeRows + row;
        })}));
  }
  // Deletes a pseudo-random half of the rows, so that a deletion bitmap
  // applied at the wrong row lets deleted rows through.
  std::mt19937 rng(7);
  std::vector<bool> deleted(kNumRows);
  std::vector<int64_t> deletedPositions;
  for (int64_t row = 0; row < kNumRows; ++row) {
    deleted[row] = rng() % 2 == 0;
    if (deleted[row]) {
      deletedPositions.emplace_back(row);
    }
  }
  const auto deletionFile =
      writeDeletionVectors({deletedPositions}, /*version=*/1)[0];

  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  const auto rowType = ROW("c0", BIGINT());
  const auto table = makeSampledTableHandle(rowType, kSampleRate);
  const auto columns = makePaimonColumnHandles(rowType);
  // Parquet readers sample row by row and skip no row groups.
  std::vector<dwio::common::FileFormat> formats{dwio::common::FileFormat::DWRF};
#ifdef VELOX_ENABLE_NIMBLE
  formats.emplace_back(dwio::common::FileFormat::NIMBLE);
#endif
  for (auto format : formats) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    writeStripes(files[0]->getPath(), stripes, format);
    PaimonDataFile dataFile;
    dataFile.path = files[0]->getPath();
    dataFile.fileFormat = format;
    dataFile.rowCount = kNumRows;
    dataFile.deletionFile = deletionFile;

    auto source =
        connector->createDataSource(rowType, table, columns, context.get());
    source->addSplit(makePaimonSplit({dataFile}));
    std::vector<int64_t> sampledRows;
    ContinueFuture future = ContinueFuture::makeEmpty();
    for (;;) {
      const auto result = source->next(kNumRows, future);
      ASSERT_TRUE(result.has_value());
      if (*result == nullptr) {
        break;
      }
      DecodedVector values(*(*result)->childAt(0));
      for (vector_size_t i = 0; i < (*result)->size(); ++i) {
        sampledRows.emplace_back(values.valueAt<int64_t>(i));
      }
    }
    // The rows of skipped stripes are never scanned.
    EXPECT_LT(source->getCompletedRows(), kNumRows);
    for (size_t i = 0; i < sampledRows.size(); ++i) {
      if (i > 0) {
        ASSERT_LT(sampledRows[i - 1], sampledRows[i]);
      }
      ASSERT_FALSE(deleted[sampledRows[i]]) << "Deleted row " << sampledRows[i];
    }
    // Allows 20%, which is 6 standard deviations.
    const double expectedNumRows = kSampleRate *
        (kNumRows - static_cast<int64_t>(deletedPositions.size()));
    EXPECT_NEAR(sampledRows.size(), expectedNumRows, 0.2 * expectedNumRows);
  }
}

// A deletion file opens like a data file, with the connector config and the
// query's token provider.
TEST_F(PaimonConnectorTest, deletionVectorFileOptions) {
  const auto fileSystem = validationFileSystem();
  fileSystem->openedFiles = 0;
  fileSystem->numOpensWithoutTokenProvider = 0;
  fileSystem->numResolvesWithoutConfig = 0;

  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context =
      queryContextBuilder(queryCtx.get())
          .sessionProperties(config.get())
          .tokenProvider(
              std::make_shared<filesystems::PlainUserNameTokenProvider>(
                  "paimon"))
          .build();
  const auto rowType = ROW("c0", BIGINT());
  const auto input = makeRowVector({makeFlatVector<int64_t>({0, 1, 2})});
  const auto files = makeFilePaths(1);
  writeToFile(files[0]->getPath(), input);
  PaimonDataFile dataFile;
  dataFile.path = files[0]->getPath();
  dataFile.fileFormat = dwio::common::FileFormat::DWRF;
  dataFile.rowCount = input->size();
  dataFile.deletionFile = writeDeletionVectors({{1}}, /*version=*/1)[0];
  dataFile.deletionFile->path =
      "paimon-table-test:" + dataFile.deletionFile->path;

  auto source = ConnectorRegistry::tryGet(kPaimonConnectorId)
                    ->createDataSource(
                        rowType,
                        makePaimonTableHandle(rowType),
                        makePaimonColumnHandles(rowType),
                        context.get());
  source->addSplit(makePaimonSplit({dataFile}));
  ContinueFuture future = ContinueFuture::makeEmpty();
  auto result = source->next(100, future);
  ASSERT_TRUE(result.has_value());
  ASSERT_NE(*result, nullptr);
  test::assertEqualVectors(
      makeRowVector({makeFlatVector<int64_t>({0, 2})}), *result);
  EXPECT_EQ(fileSystem->openedFiles, 1);
  EXPECT_EQ(fileSystem->numOpensWithoutTokenProvider, 0);
  EXPECT_EQ(fileSystem->numResolvesWithoutConfig, 0);
}

// Sequence numbers, not levels or the order of files in a split, decide the
// latest row of a key.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadLatestWins) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto older = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({10, 20, 30})});
  const auto newer = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({2, 4}), makeFlatVector<int64_t>({200, 400})});
  const auto plan = makePrimaryKeyScanPlan(ROW("value", BIGINT()), rowType);
  const auto expected =
      makeRowVector({makeFlatVector<int64_t>({10, 200, 30, 400})});
  for (auto format : fileFormats()) {
    for (bool newerFirst : {false, true}) {
      SCOPED_TRACE(
          fmt::format(
              "format {}, newerFirst {}",
              dwio::common::FileFormatName::toName(format),
              newerFirst));
      const auto files = makeFilePaths(2);
      auto olderFile =
          writePrimaryKeyFile(files[0]->getPath(), older, {1, 2, 3}, format);
      olderFile.level = 1;
      const auto newerFile =
          writePrimaryKeyFile(files[1]->getPath(), newer, {4, 5}, format);
      const auto dataFiles = newerFirst
          ? std::vector<PaimonDataFile>{newerFile, olderFile}
          : std::vector<PaimonDataFile>{olderFile, newerFile};
      exec::test::AssertQueryBuilder(plan)
          .split(makePaimonSplit(
              dataFiles,
              PaimonTableType::kPrimaryKey,
              /*rawConvertible=*/false))
          .assertResults({expected});
    }
  }
}

// Rows of a key can tie on sequence number, for example when two writers of a
// bucket start from the same sequence number. The order of the files in the
// split then decides, so deduplicate returns the row of the later file and
// first-row the row of the earlier one. Sequence numbers that differ decide
// whatever the file order.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadSequenceNumberTie) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto first = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  const auto second = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({11, 21})});
  struct TestParam {
    std::string mergeEngine;
    bool secondFileFirst;
    std::vector<int64_t> expectedValues;

    std::string debugString() const {
      return fmt::format(
          "mergeEngine {}, secondFileFirst {}", mergeEngine, secondFileFirst);
    }
  };
  const std::vector<TestParam> testSettings{
      {"deduplicate", false, {11, 21}},
      {"deduplicate", true, {10, 21}},
      {"first-row", false, {10, 20}},
      {"first-row", true, {11, 20}},
  };
  for (auto format : fileFormats()) {
    const auto files = makeFilePaths(2);
    // Key 1 has sequence number 5 in both files, and key 2 is newer in the
    // second file.
    const auto firstFile =
        writePrimaryKeyFile(files[0]->getPath(), first, {5, 6}, format);
    const auto secondFile =
        writePrimaryKeyFile(files[1]->getPath(), second, {5, 7}, format);
    for (const auto& testData : testSettings) {
      SCOPED_TRACE(
          fmt::format(
              "{}, format {}",
              testData.debugString(),
              dwio::common::FileFormatName::toName(format)));
      const auto dataFiles = testData.secondFileFirst
          ? std::vector<PaimonDataFile>{secondFile, firstFile}
          : std::vector<PaimonDataFile>{firstFile, secondFile};
      exec::test::AssertQueryBuilder(
          makePrimaryKeyScanPlan(
              ROW("value", BIGINT()),
              rowType,
              {{"merge-engine", testData.mergeEngine}}))
          .split(makePaimonSplit(
              dataFiles,
              PaimonTableType::kPrimaryKey,
              /*rawConvertible=*/false))
          .assertResults({makeRowVector(
              {makeFlatVector<int64_t>(testData.expectedValues)})});
    }
  }
}

// Keys of several columns merge rows only when every key column matches. Key
// reads of one row at a time split the rows of a key across batches, with
// strings too long to inline.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadCompositeKeys) {
  const auto rowType =
      ROW({{"name", VARCHAR()}, {"id", BIGINT()}, {"value", BIGINT()}});
  const std::string firstName{"a key name too long to inline"};
  const std::string secondName{"another key name too long to inline"};
  // Files sort their rows by the primary keys, 'name' and then 'id'.
  const auto older = makeRowVector(
      {"name", "id", "value"},
      {makeFlatVector<std::string>({firstName, firstName, secondName}),
       makeFlatVector<int64_t>({1, 2, 1}),
       makeFlatVector<int64_t>({10, 20, 30})});
  const auto newer = makeRowVector(
      {"name", "id", "value"},
      {makeFlatVector<std::string>({firstName, secondName, secondName}),
       makeFlatVector<int64_t>({2, 1, 2}),
       makeFlatVector<int64_t>({21, 31, 40})});
  const auto plan = makePaimonScanPlan(
      rowType,
      rowType,
      /*subfieldFilters=*/{},
      /*remainingFilter=*/nullptr,
      PaimonTableType::kPrimaryKey,
      /*options=*/{},
      /*primaryKeys=*/{"name", "id"});
  const auto expected = makeRowVector(
      {"name", "id", "value"},
      {makeFlatVector<std::string>(
           {firstName, firstName, secondName, secondName}),
       makeFlatVector<int64_t>({1, 2, 1, 2}),
       makeFlatVector<int64_t>({10, 21, 31, 40})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    const std::vector<PaimonDataFile> dataFiles{
        writePrimaryKeyFile(files[0]->getPath(), older, {1, 2, 3}, format),
        writePrimaryKeyFile(files[1]->getPath(), newer, {4, 5, 6}, format)};
    exec::test::AssertQueryBuilder(plan)
        .config(core::QueryConfig::kTableScanOutputBatchRowsOverride, "1")
        .split(makePaimonSplit(
            dataFiles,
            PaimonTableType::kPrimaryKey,
            /*rawConvertible=*/false))
        .assertResults({expected});
  }
}

// DWRF and Parquet readers can read file column names as lowercase, and the
// merge still finds the sequence number and row kind columns.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadLowerCaseColumnNames) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto older = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  const auto newer = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({2}), makeFlatVector<int64_t>({200})});
  const auto plan = makePrimaryKeyScanPlan(rowType, rowType);
  const auto expected = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 200})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    const std::vector<PaimonDataFile> dataFiles{
        writePrimaryKeyFile(files[0]->getPath(), older, {1, 2}, format),
        writePrimaryKeyFile(files[1]->getPath(), newer, {3}, format)};
    exec::test::AssertQueryBuilder(plan)
        .connectorSessionProperty(
            kPaimonConnectorId,
            connector::FileConfig::kFileColumnNamesReadAsLowerCaseSession,
            "true")
        .split(makePaimonSplit(
            dataFiles,
            PaimonTableType::kPrimaryKey,
            /*rawConvertible=*/false))
        .assertResults({expected});
  }
}

// A key whose latest row is a DELETE or UPDATE_BEFORE returns nothing, as
// Paimon Java drops such rows after merging.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadRetracts) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto older = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2, 3, 4}),
       makeFlatVector<int64_t>({10, 20, 30, 40})});
  // Key 5 has only a DELETE row.
  const auto newer = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({2, 3, 4, 5}),
       makeFlatVector<int64_t>({200, 30, 40, 50})});
  const auto plan = makePrimaryKeyScanPlan(rowType, rowType);
  const auto expected = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 200})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    const std::vector<PaimonDataFile> dataFiles{
        writePrimaryKeyFile(files[0]->getPath(), older, {1, 2, 3, 4}, format),
        writePrimaryKeyFile(
            files[1]->getPath(),
            newer,
            {5, 6, 7, 8},
            format,
            {PaimonRowKind::kUpdateAfter,
             PaimonRowKind::kDelete,
             PaimonRowKind::kUpdateBefore,
             PaimonRowKind::kDelete})};
    exec::test::AssertQueryBuilder(plan)
        .split(makePaimonSplit(
            dataFiles,
            PaimonTableType::kPrimaryKey,
            /*rawConvertible=*/false))
        .assertResults({expected});
  }
}

// 'ignore-delete', or else an engine-specific fallback, makes merging skip
// DELETE and UPDATE_BEFORE rows.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadIgnoreDelete) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto older = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  // Key 2 has an UPDATE_BEFORE row and key 3 only a DELETE row.
  const auto retracts = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({2, 3}), makeFlatVector<int64_t>({21, 30})});
  struct TestParam {
    std::string name;
    std::unordered_map<std::string, std::string> options;
    std::vector<int64_t> expectedValues;

    std::string debugString() const {
      return name;
    }
  };
  const std::vector<TestParam> testSettings{
      {"default", {}, {10}},
      {"ignore-delete", {{"ignore-delete", "true"}}, {10, 20}},
      {"deduplicate.ignore-delete",
       {{"deduplicate.ignore-delete", "TRUE"}},
       {10, 20}},
      {"ignore-delete over fallback",
       {{"ignore-delete", "false"}, {"deduplicate.ignore-delete", "true"}},
       {10}},
  };
  for (auto format : fileFormats()) {
    const auto files = makeFilePaths(2);
    const std::vector<PaimonDataFile> dataFiles{
        writePrimaryKeyFile(files[0]->getPath(), older, {1, 2}, format),
        writePrimaryKeyFile(
            files[1]->getPath(),
            retracts,
            {3, 4},
            format,
            {PaimonRowKind::kUpdateBefore, PaimonRowKind::kDelete})};
    for (const auto& testData : testSettings) {
      SCOPED_TRACE(
          fmt::format(
              "{}, format {}",
              testData.debugString(),
              dwio::common::FileFormatName::toName(format)));
      exec::test::AssertQueryBuilder(
          makePrimaryKeyScanPlan(
              ROW("value", BIGINT()), rowType, testData.options))
          .split(makePaimonSplit(
              dataFiles,
              PaimonTableType::kPrimaryKey,
              /*rawConvertible=*/false))
          .assertResults({makeRowVector(
              {makeFlatVector<int64_t>(testData.expectedValues)})});
    }
  }
}

// 'sequence.field' orders the rows of a key before sequence numbers do, field
// by field, in 'sequence.field.sort-order' with nulls first either way, as in
// Paimon Java. A primary key in it orders nothing, as it is equal within a key.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadSequenceField) {
  const auto rowType = ROW({"id", "ts", "value"}, BIGINT());
  const auto older = makeRowVector(
      {"id", "ts", "value"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({5, 7, 4}),
       makeFlatVector<int64_t>({10, 20, 30})});
  const auto newer = makeRowVector(
      {"id", "ts", "value"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeNullableFlatVector<int64_t>({3, std::nullopt, 4}),
       makeFlatVector<int64_t>({11, 21, 31})});
  struct TestParam {
    std::string name;
    std::unordered_map<std::string, std::string> options;
    std::vector<int64_t> expectedValues;

    std::string debugString() const {
      return name;
    }
  };
  const std::vector<TestParam> testSettings{
      {"sequence numbers", {}, {11, 21, 31}},
      {"ascending", {{"sequence.field", "ts"}}, {10, 20, 31}},
      {"descending",
       {{"sequence.field", "ts"}, {"sequence.field.sort-order", "descending"}},
       {11, 20, 31}},
      {"primary key", {{"sequence.field", "id"}}, {11, 21, 31}},
      // 'value' orders the rows of key 3, which tie on 'ts'.
      {"two fields",
       {{"sequence.field", "ts,value"},
        {"sequence.field.sort-order", "descending"}},
       {11, 20, 30}},
  };
  for (auto format : fileFormats()) {
    const auto files = makeFilePaths(2);
    const std::vector<PaimonDataFile> dataFiles{
        writePrimaryKeyFile(files[0]->getPath(), older, {1, 2, 3}, format),
        writePrimaryKeyFile(files[1]->getPath(), newer, {4, 5, 6}, format)};
    for (const auto& testData : testSettings) {
      SCOPED_TRACE(
          fmt::format(
              "{}, format {}",
              testData.debugString(),
              dwio::common::FileFormatName::toName(format)));
      exec::test::AssertQueryBuilder(
          makePrimaryKeyScanPlan(
              ROW("value", BIGINT()), rowType, testData.options))
          .split(makePaimonSplit(
              dataFiles,
              PaimonTableType::kPrimaryKey,
              /*rawConvertible=*/false))
          .assertResults({makeRowVector(
              {makeFlatVector<int64_t>(testData.expectedValues)})});
    }
  }
}

// Tables that look up keys in compaction merge the rows of a key in level 0
// with only the first row of the lowest level above 0, as Paimon Java's
// LookupMergeFunction does. A sequence field can order a row of a higher level
// after one of a lower level, which shows the difference.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadLookup) {
  const auto rowType = ROW({"id", "ts", "value"}, BIGINT());
  const auto levelTwo = makeRowVector(
      {"id", "ts", "value"},
      {makeFlatVector<int64_t>({1, 2, 4, 6}),
       makeFlatVector<int64_t>({30, 10, 1, 70}),
       makeFlatVector<int64_t>({120, 220, 420, 620})});
  const auto levelOne = makeRowVector(
      {"id", "ts", "value"},
      {makeFlatVector<int64_t>({1, 3, 4, 6}),
       makeFlatVector<int64_t>({20, 25, 2, 60}),
       makeFlatVector<int64_t>({110, 310, 410, 610})});
  const auto levelZero = makeRowVector(
      {"id", "ts", "value"},
      {makeFlatVector<int64_t>({2, 3}),
       makeFlatVector<int64_t>({15, 5}),
       makeFlatVector<int64_t>({200, 300})});
  // Flushed after levelZero, so its rows have larger sequence numbers.
  const auto newerLevelZero = makeRowVector(
      {"id", "ts", "value"},
      {makeFlatVector<int64_t>({2, 4}),
       makeFlatVector<int64_t>({5, 3}),
       makeFlatVector<int64_t>({230, 430})});
  struct TestParam {
    std::unordered_map<std::string, std::string> options;
    std::vector<int64_t> expectedValues;

    std::string debugString() const {
      std::vector<std::string> entries;
      entries.reserve(options.size());
      for (const auto& [key, value] : options) {
        entries.push_back(fmt::format("{}={}", key, value));
      }
      std::sort(entries.begin(), entries.end());
      return fmt::format("options [{}]", fmt::join(entries, ", "));
    }
  };
  // The lookup merge keeps the row of the lowest level above 0 at its position
  // in merge order among the level-0 rows:
  // - Key 1 orders level 1 before level 2, so the lookup merge keeps the row
  //   of level 1.
  // - Key 2 orders the row of level 2 between two level-0 rows, so the later
  //   level-0 row wins.
  // - Key 3 orders the row of level 1 after the level-0 row, so it wins.
  // - Key 4 ends with a level-0 DELETE, which drops the key unless
  //   'ignore-delete' skips it and leaves the row of level 1.
  // - Key 6 orders the DELETE of level 1 before the insert of level 2, so the
  //   lookup merge drops the key, also with 'ignore-delete'.
  // First-row tables look up keys too, but Paimon Java does not wrap their
  // merge function, so they return the first row of each key by sequence
  // number.
  const std::vector<TestParam> testSettings{
      {{{"sequence.field", "ts"}}, {120, 200, 310, 620}},
      {{{"sequence.field", "ts"}, {"ignore-delete", "true"}},
       {120, 200, 310, 410, 620}},
      {{{"sequence.field", "ts"}, {"changelog-producer", "lookup"}},
       {110, 200, 310}},
      {{{"sequence.field", "ts"}, {"deletion-vectors.enabled", "true"}},
       {110, 200, 310}},
      {{{"sequence.field", "ts"}, {"force-lookup", "true"}}, {110, 200, 310}},
      {{{"sequence.field", "ts"},
        {"force-lookup", "true"},
        {"ignore-delete", "true"}},
       {110, 200, 310, 410}},
      {{{"merge-engine", "first-row"}, {"ignore-delete", "true"}},
       {120, 220, 310, 420, 620}},
  };
  for (auto format : fileFormats()) {
    const auto files = makeFilePaths(4);
    auto levelTwoFile = writePrimaryKeyFile(
        files[0]->getPath(), levelTwo, {1, 2, 3, 4}, format);
    levelTwoFile.level = 2;
    auto levelOneFile = writePrimaryKeyFile(
        files[1]->getPath(),
        levelOne,
        {5, 6, 7, 8},
        format,
        {PaimonRowKind::kInsert,
         PaimonRowKind::kInsert,
         PaimonRowKind::kInsert,
         PaimonRowKind::kDelete});
    levelOneFile.level = 1;
    const auto levelZeroFile =
        writePrimaryKeyFile(files[2]->getPath(), levelZero, {9, 10}, format);
    const auto newerLevelZeroFile = writePrimaryKeyFile(
        files[3]->getPath(),
        newerLevelZero,
        {11, 12},
        format,
        {PaimonRowKind::kInsert, PaimonRowKind::kDelete});
    for (const auto& testData : testSettings) {
      SCOPED_TRACE(
          fmt::format(
              "{}, format {}",
              testData.debugString(),
              dwio::common::FileFormatName::toName(format)));
      exec::test::AssertQueryBuilder(
          makePrimaryKeyScanPlan(
              ROW("value", BIGINT()), rowType, testData.options))
          .split(makePaimonSplit(
              {levelTwoFile, levelOneFile, levelZeroFile, newerLevelZeroFile},
              PaimonTableType::kPrimaryKey,
              /*rawConvertible=*/false))
          .assertResults({makeRowVector(
              {makeFlatVector<int64_t>(testData.expectedValues)})});
    }
  }
}

// The first-row merge engine returns the first row of a key. A DELETE or
// UPDATE_BEFORE row fails the merge of a key with other rows unless
// 'ignore-delete' is set, while a key with only such a row returns nothing, as
// in Paimon Java.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadFirstRow) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto older = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  // Key 4 has only a DELETE row.
  const auto newer = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 3, 4}),
       makeFlatVector<int64_t>({11, 30, 40})});
  const auto retract = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({12})});
  const auto outputType = ROW("value", BIGINT());
  const auto plan = makePrimaryKeyScanPlan(
      outputType, rowType, {{"merge-engine", "first-row"}});
  const auto ignoreDeletePlan = makePrimaryKeyScanPlan(
      outputType,
      rowType,
      {{"merge-engine", "FIRST-ROW"}, {"first-row.ignore-delete", "true"}});
  const auto expected = makeRowVector({makeFlatVector<int64_t>({10, 20, 30})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(3);
    const auto olderFile =
        writePrimaryKeyFile(files[0]->getPath(), older, {1, 2}, format);
    const auto newerFile = writePrimaryKeyFile(
        files[1]->getPath(),
        newer,
        {3, 4, 5},
        format,
        {PaimonRowKind::kInsert,
         PaimonRowKind::kInsert,
         PaimonRowKind::kDelete});
    const auto retractFile = writePrimaryKeyFile(
        files[2]->getPath(), retract, {6}, format, {PaimonRowKind::kDelete});
    const auto makeSplit = [&](const std::vector<PaimonDataFile>& dataFiles) {
      return makePaimonSplit(
          dataFiles,
          PaimonTableType::kPrimaryKey,
          /*rawConvertible=*/false);
    };

    exec::test::AssertQueryBuilder(plan)
        .split(makeSplit({olderFile, newerFile}))
        .assertResults({expected});
    VELOX_ASSERT_USER_THROW(
        exec::test::AssertQueryBuilder(plan)
            .split(makeSplit({olderFile, newerFile, retractFile}))
            .copyResults(pool()),
        fmt::format(
            "Paimon first-row merge engine cannot merge a DELETE or "
            "UPDATE_BEFORE row unless 'ignore-delete' is true: {}",
            retractFile.path));
    exec::test::TaskMonitor::waitForAllTasksDeleted();
    exec::test::AssertQueryBuilder(ignoreDeletePlan)
        .split(makeSplit({olderFile, newerFile, retractFile}))
        .assertResults({expected});
  }
}

// Filters apply to merged rows, so an older row of a key that passes a filter
// does not return when the latest row fails it.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadFilters) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto older = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({10, 20, 30})});
  const auto newer = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({11, 21})});
  const auto expectedRow = [&](int64_t id, int64_t value) {
    return makeRowVector(
        {"id", "value"},
        {makeFlatVector<int64_t>({id}), makeFlatVector<int64_t>({value})});
  };
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    const auto split = makePaimonSplit(
        {writePrimaryKeyFile(files[0]->getPath(), older, {1, 2, 3}, format),
         writePrimaryKeyFile(files[1]->getPath(), newer, {4, 5}, format)},
        PaimonTableType::kPrimaryKey,
        /*rawConvertible=*/false);
    {
      common::SubfieldFilters filters;
      filters.emplace(
          common::Subfield("value"),
          std::make_unique<common::BigintRange>(0, 20, false));
      exec::test::AssertQueryBuilder(
          makePrimaryKeyScanPlan(rowType, rowType, {}, std::move(filters)))
          .split(split)
          .assertResults({expectedRow(1, 11)});
    }
    {
      common::SubfieldFilters filters;
      filters.emplace(
          common::Subfield("id"),
          std::make_unique<common::BigintRange>(2, 2, false));
      exec::test::AssertQueryBuilder(
          makePrimaryKeyScanPlan(rowType, rowType, {}, std::move(filters)))
          .split(split)
          .assertResults({expectedRow(2, 21)});
    }
    exec::test::AssertQueryBuilder(
        makePrimaryKeyScanPlan(
            rowType, rowType, {}, {}, parseExpr("value % 10 = 0", rowType)))
        .split(split)
        .assertResults({expectedRow(3, 30)});
  }
}

// A dynamic filter applies to merged rows like any other filter.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadDynamicFilter) {
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto table = makePaimonTableHandle(
      rowType,
      /*subfieldFilters=*/{},
      /*remainingFilter=*/nullptr,
      /*partitionNames=*/{},
      /*filterColumns=*/{},
      /*options=*/{},
      PaimonTableType::kPrimaryKey,
      /*primaryKeys=*/{"id"});
  const auto older = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({10, 20, 30})});
  const auto newer = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({11, 21})});
  const auto expected = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({11})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    auto source = ConnectorRegistry::tryGet(kPaimonConnectorId)
                      ->createDataSource(
                          rowType,
                          table,
                          makePaimonColumnHandles(rowType),
                          context.get());
    source->addSplit(makePaimonSplit(
        {writePrimaryKeyFile(files[0]->getPath(), older, {1, 2, 3}, format),
         writePrimaryKeyFile(files[1]->getPath(), newer, {4, 5}, format)},
        PaimonTableType::kPrimaryKey,
        /*rawConvertible=*/false));
    source->addDynamicFilter(
        1, std::make_shared<common::BigintRange>(0, 20, false));
    std::vector<RowVectorPtr> results;
    ContinueFuture future = ContinueFuture::makeEmpty();
    for (;;) {
      auto result = source->next(100, future);
      ASSERT_TRUE(result.has_value());
      if (*result == nullptr) {
        break;
      }
      results.push_back(
          std::static_pointer_cast<RowVector>(BaseVector::copy(**result)));
    }
    ASSERT_TRUE(exec::test::assertEqualResults({expected}, results));
  }
}

// Rows a deletion vector deletes take no part in merging, so an older row of a
// key returns when the deletion vector deletes the latest one. The key reads
// open deletion files like data files, with the connector config and the
// query's token provider, and the data reads do not open them again.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadDeletionVector) {
  const auto fileSystem = validationFileSystem();
  fileSystem->openedFiles = 0;
  fileSystem->numOpensWithoutTokenProvider = 0;
  fileSystem->numResolvesWithoutConfig = 0;
  const auto queryCtx =
      core::QueryCtx::Builder()
          .executor(driverExecutor_.get())
          .tokenProvider(
              std::make_shared<filesystems::PlainUserNameTokenProvider>(
                  "paimon"))
          .build();
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto older = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({10, 20, 30})});
  const auto newer = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({2, 3}), makeFlatVector<int64_t>({21, 31})});
  // Deletes key 1 from the older file and key 3 from the newer one.
  auto deletionFiles = writeDeletionVectors({{0}, {1}}, /*version=*/1);
  for (auto& deletionFile : deletionFiles) {
    deletionFile.path = "paimon-table-test:" + deletionFile.path;
  }
  const auto plan = makePrimaryKeyScanPlan(rowType, rowType);
  const auto expected = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({2, 3}), makeFlatVector<int64_t>({21, 30})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    auto olderFile =
        writePrimaryKeyFile(files[0]->getPath(), older, {1, 2, 3}, format);
    olderFile.deletionFile = deletionFiles[0];
    auto newerFile =
        writePrimaryKeyFile(files[1]->getPath(), newer, {4, 5}, format);
    newerFile.deletionFile = deletionFiles[1];
    exec::test::AssertQueryBuilder(plan)
        .queryCtx(queryCtx)
        .split(makePaimonSplit(
            {olderFile, newerFile},
            PaimonTableType::kPrimaryKey,
            /*rawConvertible=*/false))
        .assertResults({expected});
  }
  // Each format opens the two deletion files once.
  EXPECT_EQ(
      fileSystem->openedFiles, static_cast<int32_t>(2 * fileFormats().size()));
  EXPECT_EQ(fileSystem->numOpensWithoutTokenProvider, 0);
  EXPECT_EQ(fileSystem->numResolvesWithoutConfig, 0);
}

// The merge reads past a file that is empty or whose deletion vector deletes
// every row.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadEmptyAndDeletedFiles) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto older = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  // Newer rows of both keys, which the deletion vector of their file deletes.
  const auto deleted = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({11, 21})});
  const auto deletionFile = writeDeletionVectors({{0, 1}}, /*version=*/1)[0];
  const auto empty = makeRowVector(
      {"_KEY_id", "_SEQUENCE_NUMBER", "_VALUE_KIND", "id", "value"},
      {makeFlatVector<int64_t>({}),
       makeFlatVector<int64_t>({}),
       makeFlatVector<int8_t>({}),
       makeFlatVector<int64_t>({}),
       makeFlatVector<int64_t>({})});
  const auto plan = makePrimaryKeyScanPlan(rowType, rowType);
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(3);
    PaimonDataFile emptyFile;
    emptyFile.path = files[0]->getPath();
    emptyFile.fileFormat = format;
    emptyFile.rowCount = 0;
#ifdef VELOX_ENABLE_PARQUET
    // An empty Parquet file from the Velox test data.
    if (format == dwio::common::FileFormat::PARQUET) {
      emptyFile.path = velox::test::getDataFilePath(
          "velox/dwio/parquet/tests", "examples/empty.parquet");
    } else
#endif
    {
      writeFile(emptyFile.path, empty, format);
    }
    auto deletedFile =
        writePrimaryKeyFile(files[1]->getPath(), deleted, {3, 4}, format);
    deletedFile.deletionFile = deletionFile;
    exec::test::AssertQueryBuilder(plan)
        .split(makePaimonSplit(
            {emptyFile,
             deletedFile,
             writePrimaryKeyFile(files[2]->getPath(), older, {1, 2}, format)},
            PaimonTableType::kPrimaryKey,
            /*rawConvertible=*/false))
        .assertResults({older});
  }
}

// A single file needs merging when it has DELETE rows.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadSingleFile) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto input = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({10, 20, 30})});
  const auto plan = makePrimaryKeyScanPlan(rowType, rowType);
  const auto expected = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 3}), makeFlatVector<int64_t>({10, 30})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    exec::test::AssertQueryBuilder(plan)
        .split(makePaimonSplit(
            {writePrimaryKeyFile(
                files[0]->getPath(),
                input,
                {1, 2, 3},
                format,
                {PaimonRowKind::kInsert,
                 PaimonRowKind::kDelete,
                 PaimonRowKind::kUpdateAfter})},
            PaimonTableType::kPrimaryKey,
            /*rawConvertible=*/false))
        .assertResults({expected});
  }
}

// Files of old Paimon versions do not record their number of DELETE and
// UPDATE_BEFORE rows. Paimon Java merges a rawConvertible split of such files
// anyway, which drops those rows.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadUnknownDeleteRowCount) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto input = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({10, 20, 30})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    {
      const auto files = makeFilePaths(1);
      auto dataFile = writePrimaryKeyFile(
          files[0]->getPath(),
          input,
          {1, 2, 3},
          format,
          {PaimonRowKind::kInsert,
           PaimonRowKind::kDelete,
           PaimonRowKind::kUpdateAfter});
      dataFile.level = 1;
      dataFile.deleteRowCount = std::nullopt;
      exec::test::AssertQueryBuilder(makePrimaryKeyScanPlan(rowType, rowType))
          .split(makePaimonSplit({dataFile}, PaimonTableType::kPrimaryKey))
          .assertResults({makeRowVector(
              {"id", "value"},
              {makeFlatVector<int64_t>({1, 3}),
               makeFlatVector<int64_t>({10, 30})})});
    }

    // A partial-update table reads such a split raw only while every file
    // records the count.
    {
      const auto files = makeFilePaths(1);
      auto dataFile =
          writePrimaryKeyFile(files[0]->getPath(), input, {1, 2, 3}, format);
      dataFile.level = 1;
      const auto plan = makePrimaryKeyScanPlan(
          rowType, rowType, {{"merge-engine", "partial-update"}});
      exec::test::AssertQueryBuilder(plan)
          .split(makePaimonSplit({dataFile}, PaimonTableType::kPrimaryKey))
          .assertResults({input});
      dataFile.deleteRowCount = std::nullopt;
      VELOX_ASSERT_THROW(
          exec::test::AssertQueryBuilder(plan)
              .split(makePaimonSplit({dataFile}, PaimonTableType::kPrimaryKey))
              .copyResults(pool()),
          "Paimon merge-on-read does not support the merge engine: "
          "partial-update");
      exec::test::TaskMonitor::waitForAllTasksDeleted();
    }
  }
}

// Paimon plans a primary-key split as rawConvertible only if no file records
// DELETE or UPDATE_BEFORE rows, so a split with such a file fails before any
// read. Append-only files have no such rows, and their count is ignored, as in
// Paimon Java.
TEST_F(PaimonConnectorTest, rawConvertibleSplitWithDeleteRowCount) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto input = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    {
      const auto files = makeFilePaths(1);
      auto dataFile = writePrimaryKeyFile(
          files[0]->getPath(),
          input,
          {1, 2},
          format,
          {PaimonRowKind::kInsert, PaimonRowKind::kDelete});
      dataFile.level = 1;
      dataFile.deleteRowCount = 1;
      VELOX_ASSERT_USER_THROW(
          exec::test::AssertQueryBuilder(
              makePrimaryKeyScanPlan(rowType, rowType))
              .split(makePaimonSplit({dataFile}, PaimonTableType::kPrimaryKey))
              .copyResults(pool()),
          fmt::format(
              "(1 vs. 0) Paimon rawConvertible split must not have DELETE or "
              "UPDATE_BEFORE rows: {}",
              dataFile.path));
      exec::test::TaskMonitor::waitForAllTasksDeleted();
    }

    {
      const auto files = makeFilePaths(1);
      writeFile(files[0]->getPath(), input, format);
      PaimonDataFile dataFile;
      dataFile.path = files[0]->getPath();
      dataFile.fileFormat = format;
      dataFile.rowCount = input->size();
      dataFile.deleteRowCount = 1;
      exec::test::AssertQueryBuilder(makePaimonScanPlan(rowType))
          .split(makePaimonSplit({dataFile}, PaimonTableType::kAppendOnly))
          .assertResults({input});
    }
  }
}

// Keys span the batches of the key reads, and the batches of the data reads
// span the selected rows of several files.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadAcrossBatches) {
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto table = makePaimonTableHandle(
      rowType,
      /*subfieldFilters=*/{},
      /*remainingFilter=*/nullptr,
      /*partitionNames=*/{},
      /*filterColumns=*/{},
      /*options=*/{},
      PaimonTableType::kPrimaryKey,
      /*primaryKeys=*/{"id"});
  // File 'step - 1' holds the keys below 50 divisible by 'step', with values
  // and sequence numbers that grow with 'step'. The last file deletes its even
  // keys.
  constexpr int64_t kNumKeys = 50;
  constexpr int32_t kNumFiles = 3;
  std::vector<RowVectorPtr> inputs;
  std::vector<std::vector<int64_t>> sequenceNumbers;
  std::vector<std::vector<PaimonRowKind>> rowKinds;
  for (int64_t step = 1; step <= kNumFiles; ++step) {
    std::vector<int64_t> ids;
    std::vector<int64_t> values;
    sequenceNumbers.emplace_back();
    rowKinds.emplace_back();
    for (int64_t id = 0; id < kNumKeys; id += step) {
      ids.push_back(id);
      values.push_back(1'000 * (step - 1) + id);
      sequenceNumbers.back().push_back(100 * (step - 1) + id);
      rowKinds.back().push_back(
          step == kNumFiles && id % 2 == 0 ? PaimonRowKind::kDelete
                                           : PaimonRowKind::kInsert);
    }
    inputs.push_back(makeRowVector(
        {"id", "value"},
        {makeFlatVector<int64_t>(ids), makeFlatVector<int64_t>(values)}));
  }
  std::vector<int64_t> expectedIds;
  std::vector<int64_t> expectedValues;
  for (int64_t id = 0; id < kNumKeys; ++id) {
    if (id % 3 == 0 && id % 2 == 0) {
      continue;
    }
    const int64_t step = id % 3 == 0 ? 3 : (id % 2 == 0 ? 2 : 1);
    expectedIds.push_back(id);
    expectedValues.push_back(1'000 * (step - 1) + id);
  }
  const auto expected = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>(expectedIds),
       makeFlatVector<int64_t>(expectedValues)});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(kNumFiles);
    std::vector<PaimonDataFile> dataFiles;
    dataFiles.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      dataFiles.push_back(writePrimaryKeyFile(
          files[i]->getPath(),
          inputs[i],
          sequenceNumbers[i],
          format,
          rowKinds[i]));
    }
    auto source = ConnectorRegistry::tryGet(kPaimonConnectorId)
                      ->createDataSource(
                          rowType,
                          table,
                          makePaimonColumnHandles(rowType),
                          context.get());
    source->addSplit(makePaimonSplit(
        dataFiles,
        PaimonTableType::kPrimaryKey,
        /*rawConvertible=*/false));
    std::vector<RowVectorPtr> results;
    ContinueFuture future = ContinueFuture::makeEmpty();
    for (;;) {
      auto result = source->next(7, future);
      ASSERT_TRUE(result.has_value());
      if (*result == nullptr) {
        break;
      }
      ASSERT_LE((*result)->size(), 7);
      results.push_back(
          std::static_pointer_cast<RowVector>(BaseVector::copy(**result)));
    }
    ASSERT_TRUE(exec::test::assertEqualResults({expected}, results));
  }
}

// The merge fails on a data file whose rows differ from the row count in its
// metadata, which sizes the row selection of the file. A count below the rows
// fails on the batch that passes it, and a count above at the end of the file.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadRowCountMismatch) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto older = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  const auto newer = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({2, 3}), makeFlatVector<int64_t>({21, 30})});
  const auto expected = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({10, 21, 30})});
  struct TestParam {
    uint64_t rowCount;

    std::string debugString() const {
      return fmt::format("rowCount {}", rowCount);
    }
  };
  // Row counts below and above the 2 rows of the newer file.
  const std::vector<TestParam> testSettings{{0}, {1}, {3}, {10'000}};
  const auto plan = makePrimaryKeyScanPlan(rowType, rowType);
  const auto makeSplit = [this](const std::vector<PaimonDataFile>& dataFiles) {
    return makePaimonSplit(
        dataFiles, PaimonTableType::kPrimaryKey, /*rawConvertible=*/false);
  };
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    const auto olderFile =
        writePrimaryKeyFile(files[0]->getPath(), older, {1, 2}, format);
    const auto newerFile =
        writePrimaryKeyFile(files[1]->getPath(), newer, {3, 4}, format);
    exec::test::AssertQueryBuilder(plan)
        .split(makeSplit({olderFile, newerFile}))
        .assertResults({expected});
    for (const auto& testData : testSettings) {
      SCOPED_TRACE(testData.debugString());
      auto mismatchedFile = newerFile;
      mismatchedFile.rowCount = testData.rowCount;
      VELOX_ASSERT_USER_THROW(
          exec::test::AssertQueryBuilder(plan)
              .split(makeSplit({olderFile, mismatchedFile}))
              .copyResults(pool()),
          fmt::format(
              "(2 vs. {}) Paimon data file row count must match its "
              "metadata: {}",
              testData.rowCount,
              newerFile.path));
      exec::test::TaskMonitor::waitForAllTasksDeleted();
    }
  }
}

// Samples the latest rows of the keys. The key reads see every row, so a key
// whose latest row is not sampled returns no row rather than an older one.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadSample) {
  random::setSeed(42);
  constexpr int64_t kNumKeys{10'000};
  constexpr double kSampleRate{0.5};
  constexpr int64_t kNewerValueOffset{1'000'000};
  // The older file holds every key, and the newer file updates the even keys.
  std::vector<int64_t> olderSequenceNumbers;
  std::vector<int64_t> newerIds;
  std::vector<int64_t> newerValues;
  std::vector<int64_t> newerSequenceNumbers;
  for (int64_t id = 0; id < kNumKeys; ++id) {
    olderSequenceNumbers.push_back(1 + id);
    if (id % 2 == 0) {
      newerIds.push_back(id);
      newerValues.push_back(kNewerValueOffset + id);
      newerSequenceNumbers.push_back(1 + kNumKeys + id);
    }
  }
  const auto older = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>(kNumKeys, [](auto row) { return row; }),
       makeFlatVector<int64_t>(kNumKeys, [](auto row) { return row; })});
  const auto newer = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>(newerIds),
       makeFlatVector<int64_t>(newerValues)});
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto plan =
      exec::test::PlanBuilder()
          .startTableScan()
          .connectorId(kPaimonConnectorId)
          .outputType(rowType)
          .tableHandle(
              std::make_shared<PaimonTableHandle>(
                  kPaimonConnectorId,
                  "paimon_table",
                  17,
                  rowType,
                  fieldIds(rowType),
                  PaimonTableType::kPrimaryKey,
                  /*partitionKeys=*/std::vector<std::string>{},
                  /*options=*/std::unordered_map<std::string, std::string>{},
                  /*subfieldFilters=*/common::SubfieldFilters{},
                  /*remainingFilter=*/nullptr,
                  /*filterColumnHandles=*/
                  std::vector<connector::FileColumnHandlePtr>{},
                  kSampleRate,
                  /*dbName=*/"",
                  /*primaryKeys=*/std::vector<std::string>{"id"}))
          .assignments(makePaimonColumnHandles(rowType))
          .endTableScan()
          .planNode();
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    const auto result =
        exec::test::AssertQueryBuilder(plan)
            .split(makePaimonSplit(
                {writePrimaryKeyFile(
                     files[0]->getPath(), older, olderSequenceNumbers, format),
                 writePrimaryKeyFile(
                     files[1]->getPath(), newer, newerSequenceNumbers, format)},
                PaimonTableType::kPrimaryKey,
                /*rawConvertible=*/false))
            .copyResults(pool());
    const auto* ids = result->childAt(0)->as<SimpleVector<int64_t>>();
    const auto* values = result->childAt(1)->as<SimpleVector<int64_t>>();
    std::vector<bool> returned(kNumKeys, false);
    int64_t numEvenKeys{0};
    for (vector_size_t i = 0; i < result->size(); ++i) {
      const auto id = ids->valueAt(i);
      ASSERT_FALSE(returned[id]) << "Key " << id << " returned twice";
      returned[id] = true;
      // The newer file holds the latest row of an even key.
      const auto latestValue = id % 2 == 0 ? kNewerValueOffset + id : id;
      ASSERT_EQ(values->valueAt(i), latestValue) << "Key " << id;
      numEvenKeys += id % 2 == 0 ? 1 : 0;
    }
    // Samples the latest rows in both files. Allows 10% for each half of the
    // keys, which is 7 standard deviations.
    const double expectedNumKeys = kSampleRate * kNumKeys / 2;
    EXPECT_NEAR(numEvenKeys, expectedNumKeys, 0.1 * expectedNumKeys);
    EXPECT_NEAR(
        result->size() - numEvenKeys, expectedNumKeys, 0.1 * expectedNumKeys);
  }
}

// Primary keys may include partition columns, which merging skips since the
// rows of a split share their partition. So may 'sequence.field'.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadPartitionedTable) {
  const auto rowType =
      ROW({{"p", VARCHAR()}, {"id", BIGINT()}, {"value", BIGINT()}});
  auto assignments = makePaimonColumnHandles(rowType);
  assignments["p"] = std::make_shared<PaimonColumnHandle>(
      "p",
      10,
      connector::FileColumnHandle::ColumnType::kPartitionKey,
      VARCHAR());
  struct TestParam {
    std::string name;
    std::unordered_map<std::string, std::string> options;

    std::string debugString() const {
      return name;
    }
  };
  const std::vector<TestParam> testSettings{
      {"partition primary key", {}},
      {"partition sequence field", {{"sequence.field", "p"}}},
  };
  const auto older = makeRowVector(
      {"p", "id", "value"},
      {makeFlatVector<std::string>({"x", "x"}),
       makeFlatVector<int64_t>({1, 2}),
       makeFlatVector<int64_t>({10, 20})});
  const auto newer = makeRowVector(
      {"p", "id", "value"},
      {makeFlatVector<std::string>({"x"}),
       makeFlatVector<int64_t>({2}),
       makeFlatVector<int64_t>({21})});
  const auto expected = makeRowVector(
      {"p", "id", "value"},
      {makeFlatVector<std::string>({"x", "x"}),
       makeFlatVector<int64_t>({1, 2}),
       makeFlatVector<int64_t>({10, 21})});
  for (auto format : fileFormats()) {
    const auto files = makeFilePaths(2);
    const auto split = std::make_shared<PaimonConnectorSplit>(
        kPaimonConnectorId,
        /*snapshotId=*/1,
        PaimonTableType::kPrimaryKey,
        std::vector<PaimonDataFile>{
            writePrimaryKeyFile(files[0]->getPath(), older, {1, 2}, format),
            writePrimaryKeyFile(files[1]->getPath(), newer, {3}, format)},
        std::unordered_map<std::string, std::optional<std::string>>{{"p", "x"}},
        /*tableBucketNumber=*/std::nullopt,
        /*rawConvertible=*/false);
    for (const auto& testData : testSettings) {
      SCOPED_TRACE(
          fmt::format(
              "{}, format {}",
              testData.debugString(),
              dwio::common::FileFormatName::toName(format)));
      const auto table = makePaimonTableHandle(
          rowType,
          /*subfieldFilters=*/{},
          /*remainingFilter=*/nullptr,
          /*partitionNames=*/{"p"},
          /*filterColumns=*/{},
          testData.options,
          PaimonTableType::kPrimaryKey,
          /*primaryKeys=*/{"p", "id"});
      const auto plan = exec::test::PlanBuilder()
                            .startTableScan()
                            .connectorId(kPaimonConnectorId)
                            .outputType(rowType)
                            .tableHandle(table)
                            .assignments(assignments)
                            .endTableScan()
                            .planNode();
      exec::test::AssertQueryBuilder(plan).split(split).assertResults(
          {expected});
    }
  }
}

// One scan reads both raw and merge-on-read splits of a table.
TEST_F(PaimonConnectorTest, primaryKeyMixedRawAndMergeSplits) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto older = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  const auto newer = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({2}), makeFlatVector<int64_t>({21})});
  const auto compacted = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({3, 4}), makeFlatVector<int64_t>({30, 40})});
  const auto plan = makePrimaryKeyScanPlan(rowType, rowType);
  const auto expected = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2, 3, 4}),
       makeFlatVector<int64_t>({10, 21, 30, 40})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(3);
    const auto mergeSplit = makePaimonSplit(
        {writePrimaryKeyFile(files[0]->getPath(), older, {1, 2}, format),
         writePrimaryKeyFile(files[1]->getPath(), newer, {3}, format)},
        PaimonTableType::kPrimaryKey,
        /*rawConvertible=*/false);
    const auto rawSplit = makePaimonSplit(
        {writePrimaryKeyFile(files[2]->getPath(), compacted, {4, 5}, format)},
        PaimonTableType::kPrimaryKey);
    exec::test::AssertQueryBuilder(plan)
        .splits({mergeSplit, rawSplit})
        .assertResults({expected});
  }
}

// The aggregation merge engine aggregates each column over the rows of a key in
// merge order and retracts DELETE and UPDATE_BEFORE rows, as Paimon Java's
// AggregateMergeFunction does. A key with one row returns the row unmerged
// unless it is a DELETE or UPDATE_BEFORE row.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadAggregation) {
  const auto rowType = ROW(
      {{"id", BIGINT()},
       {"total", BIGINT()},
       {"low", DOUBLE()},
       {"name", VARCHAR()}});
  const auto older = makeRowVector(
      {"id", "total", "low", "name"},
      {makeFlatVector<int64_t>({1, 2, 3, 5}),
       makeFlatVector<int64_t>({10, 20, 30, 50}),
       makeFlatVector<double>({1.5, 2.5, 3.5, 5.5}),
       makeNullableFlatVector<std::string>({"a", "b", std::nullopt, "e"})});
  // Key 6 has only a DELETE row.
  const auto newer = makeRowVector(
      {"id", "total", "low", "name"},
      {makeFlatVector<int64_t>({1, 2, 4, 5, 6}),
       makeFlatVector<int64_t>({1, 2, 4, 5, 6}),
       makeFlatVector<double>({0.5, 9.5, 4.5, 0.25, 6.5}),
       makeNullableFlatVector<std::string>(
           {std::nullopt, "B", "d", "E", "f"})});
  const auto makeExpected =
      [&](const std::vector<int64_t>& ids,
          const std::vector<int64_t>& totals,
          const std::vector<std::optional<double>>& lows,
          const std::vector<std::optional<std::string>>& names) {
        return makeRowVector(
            {"id", "total", "low", "name"},
            {makeFlatVector<int64_t>(ids),
             makeFlatVector<int64_t>(totals),
             makeNullableFlatVector<double>(lows),
             makeNullableFlatVector<std::string>(names)});
      };
  struct TestParam {
    std::unordered_map<std::string, std::string> options;
    RowVectorPtr expected;

    std::string debugString() const {
      std::vector<std::string> entries;
      entries.reserve(options.size());
      for (const auto& [key, value] : options) {
        entries.push_back(fmt::format("{}={}", key, value));
      }
      std::sort(entries.begin(), entries.end());
      return fmt::format("options [{}]", fmt::join(entries, ", "));
    }
  };
  // 'name' takes the default 'last_non_null_value'. With
  // 'aggregation.remove-record-on-delete', the DELETE of key 5 removes it.
  const std::vector<TestParam> testSettings{
      {{{"merge-engine", "aggregation"},
        {"fields.total.aggregate-function", "sum"},
        {"fields.low.aggregate-function", "min"},
        {"fields.low.ignore-retract", "true"}},
       makeExpected(
           {1, 2, 3, 4, 5},
           {11, 18, 30, 4, 45},
           {0.5, 2.5, 3.5, 4.5, 5.5},
           {"a", std::nullopt, std::nullopt, "d", std::nullopt})},
      {{{"merge-engine", "aggregation"},
        {"fields.total.aggregate-function", "sum"},
        {"fields.low.aggregate-function", "last_value"},
        {"aggregation.remove-record-on-delete", "true"}},
       makeExpected(
           {1, 2, 3, 4},
           {11, 18, 30, 4},
           {0.5, std::nullopt, 3.5, 4.5},
           {"a", std::nullopt, std::nullopt, "d"})},
  };
  for (auto format : fileFormats()) {
    const auto files = makeFilePaths(2);
    const auto split = makePaimonSplit(
        {writePrimaryKeyFile(files[0]->getPath(), older, {1, 2, 3, 4}, format),
         writePrimaryKeyFile(
             files[1]->getPath(),
             newer,
             {5, 6, 7, 8, 9},
             format,
             {PaimonRowKind::kInsert,
              PaimonRowKind::kUpdateBefore,
              PaimonRowKind::kInsert,
              PaimonRowKind::kDelete,
              PaimonRowKind::kDelete})},
        PaimonTableType::kPrimaryKey,
        /*rawConvertible=*/false);
    for (const auto& testData : testSettings) {
      SCOPED_TRACE(
          fmt::format(
              "{}, format {}",
              testData.debugString(),
              dwio::common::FileFormatName::toName(format)));
      exec::test::AssertQueryBuilder(
          makePrimaryKeyScanPlan(rowType, rowType, testData.options))
          .split(split)
          .assertResults({testData.expected});
    }
  }
}

// Aggregation tables that look up keys in compaction aggregate the rows of a
// key in level 0 with only the first row of the lowest level above 0, as Paimon
// Java's LookupMergeFunction does. Rows a deletion vector deletes take no
// part.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadAggregationLookup) {
  const auto rowType = ROW({"id", "total"}, BIGINT());
  const auto levelTwo = makeRowVector(
      {"id", "total"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({100, 200, 300})});
  const auto levelOne = makeRowVector(
      {"id", "total"},
      {makeFlatVector<int64_t>({1, 3, 4}),
       makeFlatVector<int64_t>({10, 30, 40})});
  const auto levelZero = makeRowVector(
      {"id", "total"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({1, 2})});
  // Deletes key 4 from level 1.
  const auto deletionFiles = writeDeletionVectors({{2}}, /*version=*/1);
  struct TestParam {
    std::unordered_map<std::string, std::string> options;
    std::vector<int64_t> expectedTotals;

    std::string debugString() const {
      std::vector<std::string> entries;
      entries.reserve(options.size());
      for (const auto& [key, value] : options) {
        entries.push_back(fmt::format("{}={}", key, value));
      }
      std::sort(entries.begin(), entries.end());
      return fmt::format("options [{}]", fmt::join(entries, ", "));
    }
  };
  // Key 3 has rows in levels 1 and 2 only, so the lookup merge aggregates the
  // row of level 1 alone.
  const std::vector<TestParam> testSettings{
      {{{"merge-engine", "aggregation"},
        {"fields.total.aggregate-function", "sum"}},
       {111, 202, 330}},
      {{{"merge-engine", "aggregation"},
        {"fields.total.aggregate-function", "sum"},
        {"deletion-vectors.enabled", "true"}},
       {11, 202, 30}},
      {{{"merge-engine", "aggregation"},
        {"fields.total.aggregate-function", "sum"},
        {"changelog-producer", "lookup"}},
       {11, 202, 30}},
  };
  for (auto format : fileFormats()) {
    const auto files = makeFilePaths(3);
    auto levelTwoFile =
        writePrimaryKeyFile(files[0]->getPath(), levelTwo, {1, 2, 3}, format);
    levelTwoFile.level = 2;
    auto levelOneFile =
        writePrimaryKeyFile(files[1]->getPath(), levelOne, {4, 5, 6}, format);
    levelOneFile.level = 1;
    levelOneFile.deletionFile = deletionFiles[0];
    const auto levelZeroFile =
        writePrimaryKeyFile(files[2]->getPath(), levelZero, {7, 8}, format);
    for (const auto& testData : testSettings) {
      SCOPED_TRACE(
          fmt::format(
              "{}, format {}",
              testData.debugString(),
              dwio::common::FileFormatName::toName(format)));
      exec::test::AssertQueryBuilder(
          makePrimaryKeyScanPlan(rowType, rowType, testData.options))
          .split(makePaimonSplit(
              {levelTwoFile, levelOneFile, levelZeroFile},
              PaimonTableType::kPrimaryKey,
              /*rawConvertible=*/false))
          .assertResults({makeRowVector(
              {"id", "total"},
              {makeFlatVector<int64_t>({1, 2, 3}),
               makeFlatVector<int64_t>(testData.expectedTotals)})});
    }
  }
}

// Filters apply to aggregated rows, and a scan can filter on a column it does
// not return.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadAggregationFilters) {
  const auto rowType = ROW({"id", "total"}, BIGINT());
  const std::unordered_map<std::string, std::string> options{
      {"merge-engine", "aggregation"},
      {"fields.total.aggregate-function", "sum"}};
  const auto older = makeRowVector(
      {"id", "total"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({10, 20, 30})});
  const auto newer = makeRowVector(
      {"id", "total"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 1})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    const auto split = makePaimonSplit(
        {writePrimaryKeyFile(files[0]->getPath(), older, {1, 2, 3}, format),
         writePrimaryKeyFile(
             files[1]->getPath(),
             newer,
             {4, 5},
             format,
             {PaimonRowKind::kInsert, PaimonRowKind::kUpdateBefore})},
        PaimonTableType::kPrimaryKey,
        /*rawConvertible=*/false);
    // Merged totals are 20, 19 and 30. No row of key 1 passes alone.
    {
      common::SubfieldFilters filters;
      filters.emplace(
          common::Subfield("total"),
          std::make_unique<common::BigintRange>(20, 29, false));
      exec::test::AssertQueryBuilder(
          makePrimaryKeyScanPlan(
              ROW("id", BIGINT()), rowType, options, std::move(filters)))
          .split(split)
          .assertResults({makeRowVector({makeFlatVector<int64_t>({1})})});
    }
    exec::test::AssertQueryBuilder(makePrimaryKeyScanPlan(
                                       ROW("id", BIGINT()),
                                       rowType,
                                       options,
                                       {},
                                       parseExpr("total % 10 = 9", rowType)))
        .split(split)
        .assertResults({makeRowVector({makeFlatVector<int64_t>({2})})});
  }
}

// Keys span the batches of the file reads and of the output.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadAggregationAcrossBatches) {
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto rowType = ROW({"id", "total"}, BIGINT());
  const auto table = makePaimonTableHandle(
      rowType,
      /*subfieldFilters=*/{},
      /*remainingFilter=*/nullptr,
      /*partitionNames=*/{},
      /*filterColumns=*/{},
      {{"merge-engine", "aggregation"},
       {"fields.total.aggregate-function", "sum"}},
      PaimonTableType::kPrimaryKey,
      /*primaryKeys=*/{"id"});
  // File 'step - 1' holds the keys below 50 divisible by 'step', with totals
  // and sequence numbers that grow with 'step'. The last file retracts its
  // even keys.
  constexpr int64_t kNumKeys = 50;
  constexpr int32_t kNumFiles = 3;
  std::vector<RowVectorPtr> inputs;
  std::vector<std::vector<int64_t>> sequenceNumbers;
  std::vector<std::vector<PaimonRowKind>> rowKinds;
  for (int64_t step = 1; step <= kNumFiles; ++step) {
    std::vector<int64_t> ids;
    std::vector<int64_t> totals;
    sequenceNumbers.emplace_back();
    rowKinds.emplace_back();
    for (int64_t id = 0; id < kNumKeys; id += step) {
      ids.push_back(id);
      totals.push_back(1'000 * (step - 1) + id);
      sequenceNumbers.back().push_back(100 * (step - 1) + id);
      rowKinds.back().push_back(
          step == kNumFiles && id % 2 == 0 ? PaimonRowKind::kUpdateBefore
                                           : PaimonRowKind::kInsert);
    }
    inputs.push_back(makeRowVector(
        {"id", "total"},
        {makeFlatVector<int64_t>(ids), makeFlatVector<int64_t>(totals)}));
  }
  std::vector<int64_t> expectedIds;
  std::vector<int64_t> expectedTotals;
  for (int64_t id = 0; id < kNumKeys; ++id) {
    int64_t total = id;
    if (id % 2 == 0) {
      total += 1'000 + id;
    }
    if (id % 3 == 0) {
      total += (id % 2 == 0 ? -1 : 1) * (2'000 + id);
    }
    expectedIds.push_back(id);
    expectedTotals.push_back(total);
  }
  const auto expected = makeRowVector(
      {"id", "total"},
      {makeFlatVector<int64_t>(expectedIds),
       makeFlatVector<int64_t>(expectedTotals)});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(kNumFiles);
    std::vector<PaimonDataFile> dataFiles;
    dataFiles.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      dataFiles.push_back(writePrimaryKeyFile(
          files[i]->getPath(),
          inputs[i],
          sequenceNumbers[i],
          format,
          rowKinds[i]));
    }
    auto source = ConnectorRegistry::tryGet(kPaimonConnectorId)
                      ->createDataSource(
                          rowType,
                          table,
                          makePaimonColumnHandles(rowType),
                          context.get());
    source->addSplit(makePaimonSplit(
        dataFiles, PaimonTableType::kPrimaryKey, /*rawConvertible=*/false));
    std::vector<RowVectorPtr> results;
    ContinueFuture future = ContinueFuture::makeEmpty();
    for (;;) {
      auto result = source->next(7, future);
      ASSERT_TRUE(result.has_value());
      if (*result == nullptr) {
        break;
      }
      ASSERT_LE((*result)->size(), 7);
      results.push_back(
          std::static_pointer_cast<RowVector>(BaseVector::copy(**result)));
    }
    ASSERT_TRUE(exec::test::assertEqualResults({expected}, results));
  }
}

// A key that only other files hold can fall between two keys that one file
// holds in consecutive rows. Its merged row then lies between the copies of
// those rows in the output.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadAggregationInterleavedKeys) {
  const auto rowType = ROW({"id", "total"}, BIGINT());
  const auto outerKeys = makeRowVector(
      {"id", "total"},
      {makeFlatVector<int64_t>({1, 3}), makeFlatVector<int64_t>({10, 30})});
  const auto middleKey = makeRowVector(
      {"id", "total"},
      {makeFlatVector<int64_t>({2}), makeFlatVector<int64_t>({20})});
  const auto newerMiddleKey = makeRowVector(
      {"id", "total"},
      {makeFlatVector<int64_t>({2}), makeFlatVector<int64_t>({5})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(3);
    exec::test::AssertQueryBuilder(
        makePrimaryKeyScanPlan(
            rowType,
            rowType,
            {{"merge-engine", "aggregation"},
             {"fields.total.aggregate-function", "sum"}}))
        .split(makePaimonSplit(
            {writePrimaryKeyFile(
                 files[0]->getPath(), outerKeys, {1, 2}, format),
             writePrimaryKeyFile(files[1]->getPath(), middleKey, {3}, format),
             writePrimaryKeyFile(
                 files[2]->getPath(), newerMiddleKey, {4}, format)},
            PaimonTableType::kPrimaryKey,
            /*rawConvertible=*/false))
        .assertResults({makeRowVector(
            {"id", "total"},
            {makeFlatVector<int64_t>({1, 2, 3}),
             makeFlatVector<int64_t>({10, 25, 30})})});
  }
}

// Partition columns of an aggregated split come from the split, and a filter on
// a partition column skips the split. A raw split read after an aggregated one
// reads its columns from its file again.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadAggregationPartitionedTable) {
  const auto rowType =
      ROW({{"p", VARCHAR()}, {"id", BIGINT()}, {"total", BIGINT()}});
  auto assignments = makePaimonColumnHandles(rowType);
  assignments["p"] = std::make_shared<PaimonColumnHandle>(
      "p",
      10,
      connector::FileColumnHandle::ColumnType::kPartitionKey,
      VARCHAR());
  const auto makePlan = [&](common::SubfieldFilters filters) {
    return exec::test::PlanBuilder()
        .startTableScan()
        .connectorId(kPaimonConnectorId)
        .outputType(rowType)
        .tableHandle(makePaimonTableHandle(
            rowType,
            std::move(filters),
            /*remainingFilter=*/nullptr,
            /*partitionNames=*/{"p"},
            /*filterColumns=*/{},
            {{"merge-engine", "aggregation"},
             {"fields.total.aggregate-function", "sum"}},
            PaimonTableType::kPrimaryKey,
            /*primaryKeys=*/{"p", "id"}))
        .assignments(assignments)
        .endTableScan()
        .planNode();
  };
  const auto older = makeRowVector(
      {"p", "id", "total"},
      {makeFlatVector<std::string>({"x", "x"}),
       makeFlatVector<int64_t>({1, 2}),
       makeFlatVector<int64_t>({10, 20})});
  const auto newer = makeRowVector(
      {"p", "id", "total"},
      {makeFlatVector<std::string>({"x"}),
       makeFlatVector<int64_t>({2}),
       makeFlatVector<int64_t>({5})});
  const auto compacted = makeRowVector(
      {"p", "id", "total"},
      {makeFlatVector<std::string>({"y", "y"}),
       makeFlatVector<int64_t>({1, 3}),
       makeFlatVector<int64_t>({100, 300})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(3);
    const auto mergeSplit = std::make_shared<PaimonConnectorSplit>(
        kPaimonConnectorId,
        /*snapshotId=*/1,
        PaimonTableType::kPrimaryKey,
        std::vector<PaimonDataFile>{
            writePrimaryKeyFile(files[0]->getPath(), older, {1, 2}, format),
            writePrimaryKeyFile(files[1]->getPath(), newer, {3}, format)},
        std::unordered_map<std::string, std::optional<std::string>>{{"p", "x"}},
        /*tableBucketNumber=*/std::nullopt,
        /*rawConvertible=*/false);
    const auto rawSplit = std::make_shared<PaimonConnectorSplit>(
        kPaimonConnectorId,
        /*snapshotId=*/1,
        PaimonTableType::kPrimaryKey,
        std::vector<PaimonDataFile>{writePrimaryKeyFile(
            files[2]->getPath(), compacted, {4, 5}, format)},
        std::unordered_map<std::string, std::optional<std::string>>{{"p", "y"}},
        /*tableBucketNumber=*/std::nullopt,
        /*rawConvertible=*/true);
    exec::test::AssertQueryBuilder(makePlan({}))
        .splits({mergeSplit, rawSplit})
        .assertResults({makeRowVector(
            {"p", "id", "total"},
            {makeFlatVector<std::string>({"x", "x", "y", "y"}),
             makeFlatVector<int64_t>({1, 2, 1, 3}),
             makeFlatVector<int64_t>({10, 25, 100, 300})})});
    common::SubfieldFilters filters;
    filters.emplace(
        common::Subfield("p"),
        std::make_unique<common::BytesValues>(
            std::vector<std::string>{"y"}, false));
    exec::test::AssertQueryBuilder(makePlan(std::move(filters)))
        .splits({mergeSplit, rawSplit})
        .assertResults({makeRowVector(
            {"p", "id", "total"},
            {makeFlatVector<std::string>({"y", "y"}),
             makeFlatVector<int64_t>({1, 3}),
             makeFlatVector<int64_t>({100, 300})})});
  }
}

// Merging reads the primary keys, _SEQUENCE_NUMBER and _VALUE_KIND, so it fails
// on a file that lacks one of them or has a null in one.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadRequiresKeyAndSystemColumns) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto ids = makeFlatVector<int64_t>({1, 2});
  const auto sequenceNumbers = makeFlatVector<int64_t>({1, 2});
  const auto valueKinds = makeFlatVector<int8_t>({0, 0});
  const auto values = makeFlatVector<int64_t>({10, 20});
  struct TestParam {
    std::string name;
    // Column the error names.
    std::string column;
    RowVectorPtr input;

    std::string debugString() const {
      return name;
    }
  };
  const std::vector<TestParam> testSettings{
      // A missing primary key would merge every row into one key.
      {"missing id",
       "id",
       makeRowVector(
           {"_SEQUENCE_NUMBER", "_VALUE_KIND", "value"},
           {sequenceNumbers, valueKinds, values})},
      {"missing _SEQUENCE_NUMBER",
       "_SEQUENCE_NUMBER",
       makeRowVector(
           {"_VALUE_KIND", "id", "value"}, {valueKinds, ids, values})},
      {"missing _VALUE_KIND",
       "_VALUE_KIND",
       makeRowVector(
           {"_SEQUENCE_NUMBER", "id", "value"},
           {sequenceNumbers, ids, values})},
      {"null id",
       "id",
       makeRowVector(
           {"_SEQUENCE_NUMBER", "_VALUE_KIND", "id", "value"},
           {sequenceNumbers,
            valueKinds,
            makeNullableFlatVector<int64_t>({1, std::nullopt}),
            values})},
      {"null _SEQUENCE_NUMBER",
       "_SEQUENCE_NUMBER",
       makeRowVector(
           {"_SEQUENCE_NUMBER", "_VALUE_KIND", "id", "value"},
           {makeNullableFlatVector<int64_t>({1, std::nullopt}),
            valueKinds,
            ids,
            values})},
      {"null _VALUE_KIND",
       "_VALUE_KIND",
       makeRowVector(
           {"_SEQUENCE_NUMBER", "_VALUE_KIND", "id", "value"},
           {sequenceNumbers,
            makeNullableFlatVector<int8_t>({0, std::nullopt}),
            ids,
            values})},
  };
  const auto plan = makePrimaryKeyScanPlan(rowType, rowType);
  for (const auto& testData : testSettings) {
    for (auto format : fileFormats()) {
      SCOPED_TRACE(
          fmt::format(
              "{}, format {}",
              testData.debugString(),
              dwio::common::FileFormatName::toName(format)));
      const auto files = makeFilePaths(1);
      writeFile(files[0]->getPath(), testData.input, format);
      PaimonDataFile dataFile;
      dataFile.path = files[0]->getPath();
      dataFile.fileFormat = format;
      dataFile.rowCount = testData.input->size();
      VELOX_ASSERT_USER_THROW(
          exec::test::AssertQueryBuilder(plan)
              .split(makePaimonSplit(
                  {dataFile},
                  PaimonTableType::kPrimaryKey,
                  /*rawConvertible=*/false))
              .copyResults(pool()),
          fmt::format(
              "Paimon primary-key file must have a non-null {} column: {}",
              testData.column,
              files[0]->getPath()));
      exec::test::TaskMonitor::waitForAllTasksDeleted();
    }
  }
}

// A _VALUE_KIND value that is not a Paimon row kind fails the merge.
TEST_F(PaimonConnectorTest, primaryKeyMergeOnReadUnknownValueKind) {
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto input = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  const auto plan = makePrimaryKeyScanPlan(rowType, rowType);
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    const auto dataFile = writePrimaryKeyFile(
        files[0]->getPath(),
        input,
        {1, 2},
        format,
        {PaimonRowKind::kInsert, static_cast<PaimonRowKind>(9)});
    VELOX_ASSERT_THROW(
        exec::test::AssertQueryBuilder(plan)
            .split(makePaimonSplit(
                {dataFile},
                PaimonTableType::kPrimaryKey,
                /*rawConvertible=*/false))
            .copyResults(pool()),
        "Unknown PaimonRowKind value: 9");
    exec::test::TaskMonitor::waitForAllTasksDeleted();
  }
}

// Merge-on-read needs the primary keys and supported merge options, which a
// scan checks before opening any file. Raw splits read with any merge options.
TEST_F(PaimonConnectorTest, mergeOnReadValidationBeforeFileOpen) {
  const auto fileSystem = validationFileSystem();
  fileSystem->openedFiles = 0;
  const auto rowType = ROW({"id", "value"}, BIGINT());
  const auto split = makePaimonSplit(
      {"paimon-table-test:/not-opened"},
      PaimonTableType::kPrimaryKey,
      dwio::common::FileFormat::DWRF,
      /*partitionKeys=*/{},
      /*rawConvertible=*/false);
  struct TestParam {
    std::pair<std::string, std::string> option;
    std::vector<std::string> primaryKeys;
    std::string error;

    std::string debugString() const {
      return fmt::format(
          "option {}={}, primaryKeys {}",
          option.first,
          option.second,
          primaryKeys.size());
    }
  };
  const std::vector<TestParam> testSettings{
      {{"merge-engine", "deduplicate"},
       {},
       "Paimon table handle must have primary keys for merge-on-read: "
       "paimon_table"},
      {{"merge-engine", "partial-update"},
       {"id"},
       "Paimon merge-on-read does not support the merge engine: "
       "partial-update"},
      {{"sequence.snapshot-ordering", "true"},
       {"id"},
       "Paimon merge-on-read does not support the table option: "
       "sequence.snapshot-ordering"},
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const auto plan = makePaimonScanPlan(
        rowType,
        /*dataColumns=*/nullptr,
        /*subfieldFilters=*/{},
        /*remainingFilter=*/nullptr,
        PaimonTableType::kPrimaryKey,
        {testData.option},
        testData.primaryKeys);
    VELOX_ASSERT_THROW(
        exec::test::AssertQueryBuilder(plan).split(split).copyResults(pool()),
        testData.error);
    exec::test::TaskMonitor::waitForAllTasksDeleted();
  }
  EXPECT_EQ(fileSystem->openedFiles, 0);

  const auto files = makeFilePaths(1);
  const auto input = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  const auto dataFile = writePrimaryKeyFile(
      files[0]->getPath(), input, {1, 2}, dwio::common::FileFormat::DWRF);
  exec::test::AssertQueryBuilder(makePrimaryKeyScanPlan(
                                     rowType,
                                     rowType,
                                     {{"merge-engine", "partial-update"},
                                      {"sequence.snapshot-ordering", "true"}}))
      .split(makePaimonSplit({dataFile}, PaimonTableType::kPrimaryKey))
      .assertResults({input});
}

// E2E test: multiple splits each with multiple files.
TEST_F(PaimonConnectorTest, appendOnlyMultipleSplits) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});

  // Split 1 with 2 files.
  auto vectors1a = makeVectors(rowType, 1, 50);
  auto vectors1b = makeVectors(rowType, 1, 30);
  // Split 2 with 1 file.
  auto vectors2a = makeVectors(rowType, 1, 40);

  auto filePaths = makeFilePaths(3);
  writeToFile(filePaths[0]->getPath(), vectors1a);
  writeToFile(filePaths[1]->getPath(), vectors1b);
  writeToFile(filePaths[2]->getPath(), vectors2a);

  auto split1 =
      makePaimonSplit({filePaths[0]->getPath(), filePaths[1]->getPath()});
  auto split2 = makePaimonSplit({filePaths[2]->getPath()});

  auto plan = makePaimonScanPlan(rowType);

  std::vector<RowVectorPtr> expected;
  expected.insert(expected.end(), vectors1a.begin(), vectors1a.end());
  expected.insert(expected.end(), vectors1b.begin(), vectors1b.end());
  expected.insert(expected.end(), vectors2a.begin(), vectors2a.end());

  exec::test::AssertQueryBuilder(plan)
      .splits({split1, split2})
      .assertResults(expected);
}

} // namespace
} // namespace facebook::velox::connector::paimon

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  facebook::velox::process::addDefaultFatalSignalHandler();
  folly::Init init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
