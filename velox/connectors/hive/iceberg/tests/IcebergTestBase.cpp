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

#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

#include <filesystem>

#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergConfig.h"
#include "velox/connectors/hive/iceberg/IcebergConnector.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/IcebergTableHandle.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::connector::hive::iceberg::test {

const std::string kIcebergConnectorId{"test-iceberg"};

void IcebergTestBase::SetUp() {
  HiveConnectorTestBase::SetUp();
#ifdef VELOX_ENABLE_PARQUET
  parquet::registerParquetReaderFactory();
  parquet::registerParquetWriterFactory();
#endif
  Type::registerSerDe();

  // Register IcebergConnector.
  IcebergConnectorFactory icebergFactory;
  auto icebergConnector = icebergFactory.newConnector(
      kIcebergConnectorId,
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()),
      ioExecutor_.get());
  ConnectorRegistry::global().insert(
      icebergConnector->connectorId(), icebergConnector);

  connectorSessionProperties_ = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>(), true);

  hiveConfig_ =
      std::make_shared<HiveConfig>(std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));

  icebergConfig_ =
      std::make_shared<IcebergConfig>(std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{
              {IcebergConfig::kFunctionPrefixConfig,
               IcebergConfig::kDefaultFunctionPrefix}}));

  setupMemoryPools();

  fuzzerOptions_.vectorSize = 100;
  fuzzerOptions_.nullRatio = 0.1;
  fuzzer_ = std::make_unique<VectorFuzzer>(fuzzerOptions_, opPool_.get(), 1);
}

void IcebergTestBase::TearDown() {
  fuzzer_.reset();
  connectorQueryCtx_.reset();
  connectorPool_.reset();
  opPool_.reset();
  root_.reset();
  queryCtx_.reset();
  ConnectorRegistry::global().erase(kIcebergConnectorId);
  HiveConnectorTestBase::TearDown();
}

void IcebergTestBase::setupMemoryPools() {
  root_.reset();
  opPool_.reset();
  connectorPool_.reset();
  connectorQueryCtx_.reset();
  queryCtx_.reset();

  root_ = memory::memoryManager()->addRootPool(
      "IcebergTest", 1L << 30, exec::MemoryReclaimer::create());
  opPool_ = root_->addLeafChild("operator");
  connectorPool_ =
      root_->addAggregateChild("connector", exec::MemoryReclaimer::create());

  recreateConnectorQueryCtx(/*sessionTimezone=*/"", false);
}

void IcebergTestBase::recreateConnectorQueryCtx(
    const std::string& sessionTimezone,
    bool adjustTimestampToTimezone) {
  connectorQueryCtx_.reset();
  queryCtx_.reset();

  queryCtx_ = core::QueryCtx::create(nullptr, core::QueryConfig({}));
  auto expressionEvaluator = std::make_unique<exec::SimpleExpressionEvaluator>(
      queryCtx_.get(), opPool_.get());

  connectorQueryCtx_ = std::make_unique<ConnectorQueryCtx>(
      opPool_.get(),
      connectorPool_.get(),
      connectorSessionProperties_.get(),
      nullptr,
      common::PrefixSortConfig(),
      std::move(expressionEvaluator),
      nullptr,
      "query.IcebergTest",
      "task.IcebergTest",
      "planNodeId.IcebergTest",
      0,
      sessionTimezone,
      adjustTimestampToTimezone);
}

std::vector<RowVectorPtr> IcebergTestBase::createTestData(
    RowTypePtr rowType,
    int32_t numBatches,
    vector_size_t rowsPerBatch,
    double nullRatio) {
  std::vector<RowVectorPtr> vectors;
  vectors.reserve(numBatches);

  fuzzerOptions_.nullRatio = nullRatio;
  fuzzerOptions_.allowDictionaryVector = false;
  fuzzerOptions_.timestampPrecision =
      fuzzer::FuzzerTimestampPrecision::kMilliSeconds;
  fuzzer_->setOptions(fuzzerOptions_);

  for (auto i = 0; i < numBatches; ++i) {
    vectors.push_back(fuzzer_->fuzzRow(rowType, rowsPerBatch, false));
  }

  return vectors;
}

void IcebergTestBase::setConnectorSessionProperty(
    const std::string& key,
    const std::string& value) {
  VELOX_CHECK_NOT_NULL(connectorSessionProperties_);
  connectorSessionProperties_->set(key, value);
}

std::shared_ptr<IcebergPartitionSpec> IcebergTestBase::createPartitionSpec(
    const RowTypePtr& rowType,
    const std::vector<PartitionField>& partitionFields) {
  std::vector<IcebergPartitionSpec::Field> fields;
  for (const auto& partitionField : partitionFields) {
    fields.push_back(
        IcebergPartitionSpec::Field{
            rowType->nameOf(partitionField.id),
            rowType->childAt(partitionField.id),
            partitionField.type,
            partitionField.parameter});
  }

  return fields.empty() ? nullptr
                        : std::make_shared<IcebergPartitionSpec>(1, fields);
}

namespace {

parquet::ParquetFieldId makeField(const TypePtr& type, int32_t& fieldId) {
  const int32_t currentId = fieldId++;
  std::vector<parquet::ParquetFieldId> children;
  children.reserve(type->size());
  for (auto i = 0; i < type->size(); ++i) {
    children.push_back(makeField(type->childAt(i), fieldId));
  }
  return parquet::ParquetFieldId{currentId, children};
}

void addColumnHandles(
    const RowTypePtr& rowType,
    const std::vector<PartitionField>& partitionFields,
    std::vector<IcebergColumnHandlePtr>& columnHandles) {
  std::unordered_set<int32_t> partitionColumnIds;
  for (const auto& field : partitionFields) {
    partitionColumnIds.insert(field.id);
  }

  int32_t fieldId = 1;
  columnHandles.reserve(rowType->size());
  for (auto i = 0; i < rowType->size(); ++i) {
    const auto& columnName = rowType->nameOf(i);
    const auto& type = rowType->childAt(i);
    auto field = makeField(type, fieldId);
    columnHandles.push_back(
        std::make_shared<const IcebergColumnHandle>(
            columnName,
            partitionColumnIds.contains(i)
                ? FileColumnHandle::ColumnType::kPartitionKey
                : FileColumnHandle::ColumnType::kRegular,
            type,
            field));
  }
}

} // namespace

IcebergInsertTableHandlePtr IcebergTestBase::createInsertTableHandle(
    const RowTypePtr& rowType,
    const std::string& outputDirectoryPath,
    const std::vector<PartitionField>& partitionFields) {
  std::vector<IcebergColumnHandlePtr> columnHandles;
  addColumnHandles(rowType, partitionFields, columnHandles);

  auto locationHandle = std::make_shared<LocationHandle>(
      outputDirectoryPath,
      outputDirectoryPath,
      LocationHandle::TableType::kNew);

  auto partitionSpec = createPartitionSpec(rowType, partitionFields);

  return std::make_shared<const IcebergInsertTableHandle>(
      /*inputColumns=*/columnHandles,
      locationHandle,
      /*tableStorageFormat=*/fileFormat_,
      partitionSpec,
      /*compressionKind=*/common::CompressionKind::CompressionKind_ZSTD);
}

std::shared_ptr<IcebergDataSink> IcebergTestBase::createDataSink(
    const RowTypePtr& rowType,
    const std::string& outputDirectoryPath,
    const std::vector<PartitionField>& partitionFields) {
  auto tableHandle =
      createInsertTableHandle(rowType, outputDirectoryPath, partitionFields);
  return std::make_shared<IcebergDataSink>(
      rowType,
      tableHandle,
      connectorQueryCtx_.get(),
      CommitStrategy::kNoCommit,
      hiveConfig_,
      icebergConfig_);
}

std::shared_ptr<IcebergDataSink> IcebergTestBase::createDataSinkAndAppendData(
    const std::vector<RowVectorPtr>& vectors,
    const std::string& dataPath,
    const std::vector<PartitionField>& partitionFields) {
  VELOX_CHECK(!vectors.empty(), "vectors cannot be empty");

  auto rowType = vectors.front()->rowType();
  auto dataSink = createDataSink(rowType, dataPath, partitionFields);

  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }
  EXPECT_TRUE(dataSink->finish());
  return dataSink;
}

std::vector<std::string> IcebergTestBase::listFiles(
    const std::string& dirPath) {
  std::vector<std::string> files;
  if (!std::filesystem::exists(dirPath)) {
    return files;
  }

  for (auto& dirEntry :
       std::filesystem::recursive_directory_iterator(dirPath)) {
    if (dirEntry.is_regular_file()) {
      files.push_back(dirEntry.path().string());
    }
  }
  return files;
}

std::unordered_map<std::string, std::optional<std::string>>
IcebergTestBase::extractPartitionKeys(const std::string& filePath) {
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys;

  std::vector<std::string> pathComponents;
  folly::split("/", filePath, pathComponents);
  for (const auto& component : pathComponents) {
    if (component.find('=') != std::string::npos) {
      std::vector<std::string> keys;
      folly::split('=', component, keys);
      if (keys.size() == 2) {
        if (keys[1] == "null") {
          partitionKeys[keys[0]] = std::nullopt;
        } else {
          partitionKeys[keys[0]] = keys[1];
        }
      }
    }
  }

  return partitionKeys;
}

std::vector<std::shared_ptr<ConnectorSplit>>
IcebergTestBase::createSplitsForDirectory(const std::string& directory) {
  std::vector<std::shared_ptr<ConnectorSplit>> splits;

  auto files = listFiles(directory);
  for (const auto& filePath : files) {
    auto partitionKeys = extractPartitionKeys(filePath);

    const auto file = filesystems::getFileSystem(filePath, nullptr)
                          ->openFileForRead(filePath);
    splits.push_back(IcebergSplitBuilder(filePath)
                         .connectorId(kIcebergConnectorId)
                         .fileFormat(fileFormat_)
                         .length(file->size())
                         .partitionKeys(partitionKeys)
                         .build());
  }

  return splits;
}

uint64_t IcebergTestBase::getFileSize(const std::string& path) {
  return filesystems::getFileSystem(path, nullptr)
      ->openFileForRead(path)
      ->size();
}

std::vector<std::shared_ptr<ConnectorSplit>> IcebergTestBase::makeIcebergSplits(
    const std::string& dataFilePath,
    const std::vector<IcebergDeleteFile>& deleteFiles,
    const std::unordered_map<std::string, std::optional<std::string>>&
        partitionKeys,
    uint32_t splitCount,
    const std::unordered_map<std::string, std::string>& infoColumns,
    int64_t dataSequenceNumber,
    const std::unordered_map<int32_t, std::optional<std::string>>&
        identityPartitionKeys) {
  VELOX_CHECK_GT(splitCount, 0);
  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  const auto fileSize = getFileSize(dataFilePath);
  const auto splitSize = fileSize / splitCount;
  splits.reserve(splitCount);

  for (auto i = 0; i < splitCount; ++i) {
    splits.emplace_back(IcebergSplitBuilder(dataFilePath)
                            .connectorId(kIcebergConnectorId)
                            .fileFormat(fileFormat_)
                            .start(i * splitSize)
                            .length(splitSize)
                            .partitionKeys(partitionKeys)
                            .deleteFiles(deleteFiles)
                            .infoColumns(infoColumns)
                            .dataSequenceNumber(dataSequenceNumber)
                            .identityPartitionKeys(identityPartitionKeys)
                            .build());
  }

  return splits;
}

std::shared_ptr<ConnectorSplit>
IcebergTestBase::makeIcebergSplitWithInfoColumns(
    const std::string& dataFilePath,
    const std::unordered_map<std::string, std::string>& infoColumns,
    const std::vector<IcebergDeleteFile>& deleteFiles,
    int64_t dataSequenceNumber) {
  auto splits = makeIcebergSplits(
      dataFilePath, deleteFiles, {}, 1, infoColumns, dataSequenceNumber);
  VELOX_CHECK_EQ(splits.size(), 1);
  return splits.front();
}

std::shared_ptr<common::testutil::TempFilePath> IcebergTestBase::writeDataFile(
    const std::vector<RowVectorPtr>& data) {
  auto file = common::testutil::TempFilePath::create();
  writeToFile(file->getPath(), data);
  return file;
}

std::shared_ptr<common::testutil::TempFilePath>
IcebergTestBase::writeDwrfFileWithFieldIds(
    const std::vector<RowVectorPtr>& data,
    const std::vector<int32_t>& icebergFieldIds) {
  VELOX_CHECK(!data.empty());
  const uint32_t numCols = data[0]->type()->size();
  VELOX_CHECK_EQ(icebergFieldIds.size(), numCols);

  // Build schemaAttributes: DWRF pre-order node 0 is the root struct (no
  // iceberg.id); nodes 1..numCols are the top-level columns.
  std::unordered_map<uint32_t, std::vector<std::pair<std::string, std::string>>>
      attrs;
  for (uint32_t i = 0; i < numCols; ++i) {
    attrs[i + 1] = {{"iceberg.id", std::to_string(icebergFieldIds[i])}};
  }

  auto file = common::testutil::TempFilePath::create();
  auto fs = filesystems::getFileSystem(file->getPath(), {});
  auto writeFile = fs->openFileForWrite(
      file->getPath(),
      {.shouldCreateParentDirectories = true,
       .shouldThrowOnFileAlreadyExists = false});
  auto sink = std::make_unique<dwio::common::WriteFileSink>(
      std::move(writeFile), file->getPath());
  dwio::common::WriterOptions writerOptions;
  auto dwrfOptions = std::make_shared<dwrf::DwrfWriterOptions>();
  dwrfOptions->schemaAttributes = std::move(attrs);
  writerOptions.formatSpecificOptions = dwrfOptions;
  writerOptions.schema = data[0]->type();
  auto childPool =
      rootPool_->addAggregateChild("writeDwrfFileWithFieldIds.writer");
  writerOptions.memoryPool = childPool.get();
  dwrf::Writer writer{std::move(sink), writerOptions};
  for (const auto& batch : data) {
    writer.write(batch);
  }
  writer.close();
  return file;
}

#ifdef VELOX_ENABLE_PARQUET
std::shared_ptr<common::testutil::TempFilePath>
IcebergTestBase::writeParquetFile(
    const std::vector<RowVectorPtr>& data,
    const std::vector<int32_t>& icebergFieldIds) {
  VELOX_CHECK(!data.empty());
  auto file = common::testutil::TempFilePath::create();
  auto writeFile =
      std::make_unique<LocalWriteFile>(file->getPath(), true, false);
  auto sink = std::make_unique<dwio::common::WriteFileSink>(
      std::move(writeFile), file->getPath());
  dwio::common::WriterOptions writerOptions;
  writerOptions.memoryPool = rootPool_.get();
  parquet::ParquetWriterOptions parquetOptions;
  if (!icebergFieldIds.empty()) {
    VELOX_CHECK_EQ(icebergFieldIds.size(), data[0]->type()->size());
    parquetOptions.parquetFieldIds.reserve(icebergFieldIds.size());
    for (int32_t id : icebergFieldIds) {
      parquetOptions.parquetFieldIds.push_back(parquet::ParquetFieldId{id, {}});
    }
  }
  writerOptions.formatSpecificOptions =
      std::make_shared<parquet::ParquetWriterOptions>(
          std::move(parquetOptions));
  auto writer = std::make_unique<parquet::Writer>(
      std::move(sink), writerOptions, asRowType(data[0]->type()));
  for (const auto& batch : data) {
    writer->write(batch);
  }
  writer->close();
  return file;
}
#endif // VELOX_ENABLE_PARQUET

core::PlanNodePtr IcebergTestBase::makeIcebergTableScanPlan(
    const RowTypePtr& outputType,
    const RowTypePtr& dataColumns,
    const std::vector<int32_t>& dataColumnFieldIds,
    const std::vector<std::string>& subfieldFilters,
    const std::string& remainingFilter) {
  VELOX_CHECK_NOT_NULL(dataColumns);

  // Build IcebergColumnHandle assignments for each output-projected column.
  // The Iceberg field ID is taken from dataColumnFieldIds when available,
  // otherwise it defaults to the 1-based ordinal position in dataColumns.
  connector::ColumnHandleMap assignments;
  assignments.reserve(outputType->size());
  for (uint32_t i = 0; i < outputType->size(); ++i) {
    const auto& name = outputType->nameOf(i);
    const auto& type = outputType->childAt(i);
    auto tableIdx = dataColumns->getChildIdxIfExists(name);
    VELOX_CHECK(
        tableIdx.has_value(),
        "Output column '{}' not found in dataColumns.",
        name);
    const int32_t fieldId = !dataColumnFieldIds.empty()
        ? dataColumnFieldIds[*tableIdx]
        : static_cast<int32_t>(*tableIdx + 1);
    assignments.emplace(
        name,
        std::make_shared<IcebergColumnHandle>(
            name,
            FileColumnHandle::ColumnType::kRegular,
            type,
            parquet::ParquetFieldId{fieldId, {}}));
  }

  // Build filter-only IcebergColumnHandles for columns referenced by pushed-
  // down filters but absent from the output projection. These are needed so
  // buildIcebergHandleByName() can resolve their Iceberg field IDs and
  // configureEqualityDeleteColumns() can promote them to projected columns
  // when they also serve as equality-delete keys.
  std::vector<HiveColumnHandlePtr> filterHandles;
  if (!subfieldFilters.empty() || !remainingFilter.empty()) {
    for (uint32_t i = 0; i < dataColumns->size(); ++i) {
      const auto& name = dataColumns->nameOf(i);
      if (assignments.count(name)) {
        continue; // Already in the output projection.
      }
      // Include this column as a filter handle if any subfield filter names it.
      bool usedInFilter = std::any_of(
          subfieldFilters.begin(),
          subfieldFilters.end(),
          [&name](const std::string& f) {
            return f.find(name) != std::string::npos;
          });
      // Also include it if it appears in the remainingFilter expression.
      if (!usedInFilter && !remainingFilter.empty()) {
        usedInFilter = remainingFilter.find(name) != std::string::npos;
      }
      if (!usedInFilter) {
        continue;
      }
      const auto& type = dataColumns->childAt(i);
      const int32_t fieldId = !dataColumnFieldIds.empty()
          ? dataColumnFieldIds[i]
          : static_cast<int32_t>(i + 1);
      filterHandles.push_back(
          std::make_shared<IcebergColumnHandle>(
              name,
              FileColumnHandle::ColumnType::kRegular,
              type,
              parquet::ParquetFieldId{fieldId, {}}));
    }
  }

  return exec::test::PlanBuilder()
      .startTableScan(kIcebergConnectorId)
      .outputType(outputType)
      .dataColumns(dataColumns)
      .subfieldFilters(subfieldFilters)
      .remainingFilter(remainingFilter)
      .dataColumnFieldIds(dataColumnFieldIds)
      .filterColumnHandles(std::move(filterHandles))
      .assignments(assignments)
      .endTableScan()
      .planNode();
}

core::PlanNodePtr IcebergTestBase::makeIcebergTableScanPlan(
    const RowTypePtr& rowType) {
  return makeIcebergTableScanPlan(rowType, rowType);
}

ColumnHandleMap IcebergTestBase::makeColumnHandles(
    const RowTypePtr& rowType,
    const std::unordered_set<int>& partitionIndices) {
  ColumnHandleMap assignments;
  for (auto i = 0; i < rowType->size(); ++i) {
    const auto& columnName = rowType->nameOf(i);
    const auto& columnType = rowType->childAt(i);
    const auto columnHandleType = partitionIndices.contains(i)
        ? FileColumnHandle::ColumnType::kPartitionKey
        : FileColumnHandle::ColumnType::kRegular;
    assignments.insert(
        {columnName,
         std::make_shared<HiveColumnHandle>(
             columnName,
             columnHandleType,
             columnType,
             columnType,
             std::vector<common::Subfield>{})});
  }

  return assignments;
}

RowTypePtr IcebergTestBase::makeChangelogOutputType(
    const RowTypePtr& dataType) {
  return ROW(
      {"operation", "ordinal", "snapshotid", "rowdata"},
      {VARCHAR(), BIGINT(), BIGINT(), dataType});
}

ColumnHandleMap IcebergTestBase::makeChangelogColumnHandles(
    const RowTypePtr& dataType) {
  ColumnHandleMap handles;
  handles["operation"] = std::make_shared<IcebergColumnHandle>(
      "operation",
      IcebergColumnHandle::ColumnType::kRegular,
      VARCHAR(),
      parquet::ParquetFieldId{1});
  handles["ordinal"] = std::make_shared<IcebergColumnHandle>(
      "ordinal",
      IcebergColumnHandle::ColumnType::kRegular,
      BIGINT(),
      parquet::ParquetFieldId{2});
  handles["snapshotid"] = std::make_shared<IcebergColumnHandle>(
      "snapshotid",
      IcebergColumnHandle::ColumnType::kRegular,
      BIGINT(),
      parquet::ParquetFieldId{3});
  handles["rowdata"] = std::make_shared<IcebergColumnHandle>(
      "rowdata",
      IcebergColumnHandle::ColumnType::kRegular,
      dataType,
      parquet::ParquetFieldId{4});
  return handles;
}

std::unordered_map<std::string, IcebergColumnHandlePtr>
IcebergTestBase::makeDataColumnHandles(const RowTypePtr& dataType) {
  std::unordered_map<std::string, IcebergColumnHandlePtr> handles;
  int32_t fieldId = 1;
  for (size_t i = 0; i < dataType->size(); ++i) {
    const auto& name = dataType->nameOf(i);
    const auto& type = dataType->childAt(i);
    handles[name] = std::make_shared<IcebergColumnHandle>(
        name,
        IcebergColumnHandle::ColumnType::kRegular,
        type,
        parquet::ParquetFieldId{fieldId++});
  }
  return handles;
}

std::shared_ptr<IcebergTableHandle> IcebergTestBase::makeChangelogTableHandle(
    const RowTypePtr& dataType) {
  return std::make_shared<IcebergTableHandle>(
      kIcebergConnectorId,
      "test_table",
      common::SubfieldFilters{},
      nullptr,
      dataType,
      std::vector<std::string>{},
      std::unordered_map<std::string, std::string>{},
      std::vector<IcebergColumnHandlePtr>{},
      1.0,
      "",
      std::vector<int32_t>{},
      true,
      makeDataColumnHandles(dataType));
}

} // namespace facebook::velox::connector::hive::iceberg::test
