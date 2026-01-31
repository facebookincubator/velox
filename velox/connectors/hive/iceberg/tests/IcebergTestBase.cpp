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

#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergConnector.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
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
  registerConnector(icebergConnector);

  connectorSessionProperties_ = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>(), true);

  connectorConfig_ =
      std::make_shared<HiveConfig>(std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));

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
  unregisterConnector(kIcebergConnectorId);
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
      "");
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
                ? HiveColumnHandle::ColumnType::kPartitionKey
                : HiveColumnHandle::ColumnType::kRegular,
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
      connectorConfig_,
      std::string(kDefaultIcebergFunctionPrefix));
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
    splits.push_back(
        std::make_shared<HiveIcebergSplit>(
            kIcebergConnectorId,
            filePath,
            fileFormat_,
            0,
            file->size(),
            partitionKeys,
            std::nullopt,
            std::unordered_map<std::string, std::string>{},
            nullptr,
            /*cacheable=*/true,
            std::vector<IcebergDeleteFile>()));
  }

  return splits;
}

} // namespace facebook::velox::connector::hive::iceberg::test
