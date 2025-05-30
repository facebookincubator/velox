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

#include <regex>

#include "velox/connectors/hive/iceberg/PartitionSpec.h"

namespace facebook::velox::connector::hive::iceberg::test {
void IcebergTestBase::SetUp() {
  HiveConnectorTestBase::SetUp();
  parquet::registerParquetReaderFactory();
  parquet::registerParquetWriterFactory();
  Type::registerSerDe();

  // Initialize session properties and config.
  connectorSessionProperties_ = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>(), true);

  connectorConfig_ =
      std::make_shared<HiveConfig>(std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));

  setupMemoryPools("IcebergTestBase");

  // Initialize vector fuzzer for test data generation.
  fuzzerOptions_.vectorSize = 100;
  fuzzerOptions_.nullRatio = 0.1;
  fuzzer_ = std::make_unique<VectorFuzzer>(fuzzerOptions_, opPool_.get());

  vectorMaker_ =
      std::make_unique<facebook::velox::test::VectorMaker>(opPool_.get());
}

void IcebergTestBase::TearDown() {
  vectorMaker_.reset();
  fuzzer_.reset();
  connectorQueryCtx_.reset();
  connectorPool_.reset();
  opPool_.reset();
  root_.reset();
  HiveConnectorTestBase::TearDown();
}

void IcebergTestBase::setupMemoryPools(const std::string& name) {
  root_.reset();
  opPool_.reset();
  connectorPool_.reset();
  connectorQueryCtx_.reset();

  root_ = memory::memoryManager()->addRootPool(
      name, 1L << 30, exec::MemoryReclaimer::create());
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
      "query" + name,
      "task" + name,
      "planNodeId" + name,
      0,
      "");
}

std::vector<RowVectorPtr> IcebergTestBase::createTestData(
    int32_t numBatches,
    vector_size_t rowsPerBatch,
    double nullRatio) {
  std::vector<RowVectorPtr> vectors;
  vectors.reserve(numBatches);

  fuzzerOptions_.nullRatio = nullRatio;
  fuzzerOptions_.allowDictionaryVector = false;
  fuzzer_->setOptions(fuzzerOptions_);

  for (auto i = 0; i < numBatches; ++i) {
    vectors.push_back(fuzzer_->fuzzRow(rowType_, rowsPerBatch, false));
  }

  return vectors;
}

std::shared_ptr<IcebergPartitionSpec> IcebergTestBase::createPartitionSpec(
    const std::vector<PartitionField>& partitionFields,
    const RowTypePtr& rowType) {
  std::vector<IcebergPartitionSpec::Field> fields;
  for (const auto& partitionField : partitionFields) {
    fields.push_back(IcebergPartitionSpec::Field(
        rowType->nameOf(partitionField.id),
        rowType->childAt(partitionField.id),
        partitionField.type,
        partitionField.parameter));
  }

  return std::make_shared<IcebergPartitionSpec>(1, fields);
}

void addColumnHandles(
    const RowTypePtr& rowType,
    const std::vector<PartitionField>& partitionFields,
    std::vector<std::shared_ptr<const HiveColumnHandle>>& columnHandles) {
  std::unordered_set<int32_t> partitionColumnIds;
  for (const auto& field : partitionFields) {
    partitionColumnIds.insert(field.id);
  }
  HiveColumnHandle::ColumnParseParameters columnParseParameters;

  std::function<IcebergNestedField(const TypePtr&, int32_t&)>
      collectNestedField = [&](const TypePtr& type,
                               int32_t& columnOrdinal) -> IcebergNestedField {
    int32_t currentId = columnOrdinal++;
    std::vector<IcebergNestedField> children;
    if (type->isRow()) {
      auto rowType = asRowType(type);
      for (auto i = 0; i < rowType->size(); ++i) {
        children.push_back(
            collectNestedField(rowType->childAt(i), columnOrdinal));
      }
    } else if (type->isArray()) {
      auto arrayType = std::dynamic_pointer_cast<const ArrayType>(type);
      for (auto i = 0; i < arrayType->size(); ++i) {
        children.push_back(
            collectNestedField(arrayType->childAt(i), columnOrdinal));
      }
    } else if (type->isMap()) {
      auto mapType = std::dynamic_pointer_cast<const MapType>(type);
      for (auto i = 0; i < mapType->size(); ++i) {
        children.push_back(
            collectNestedField(mapType->childAt(i), columnOrdinal));
      }
    }

    return IcebergNestedField{currentId, children};
  };

  int32_t startIndex = 1;
  for (auto i = 0; i < rowType->size(); ++i) {
    auto columnName = rowType->nameOf(i);
    auto type = rowType->childAt(i);
    auto field = collectNestedField(type, startIndex);
    columnHandles.push_back(std::make_shared<IcebergColumnHandle>(
        columnName,
        partitionColumnIds.count(i) > 0
            ? HiveColumnHandle::ColumnType::kPartitionKey
            : HiveColumnHandle::ColumnType::kRegular,
        type,
        type,
        field,
        std::vector<common::Subfield>{},
        columnParseParameters));
  }
}

std::shared_ptr<IcebergInsertTableHandle>
IcebergTestBase::createIcebergInsertTableHandle(
    const RowTypePtr& rowType,
    const std::string& outputDirectoryPath,
    const std::vector<PartitionField>& partitionFields,
    const std::vector<std::string>& sortedBy) {
  std::vector<std::shared_ptr<const HiveColumnHandle>> columnHandles;
  addColumnHandles(rowType, partitionFields, columnHandles);

  auto locationHandle = std::make_shared<LocationHandle>(
      outputDirectoryPath,
      outputDirectoryPath,
      LocationHandle::TableType::kNew);

  auto partitionSpec = createPartitionSpec(partitionFields, rowType);

  // Create sorting columns if specified
  std::vector<IcebergSortingColumn> sortingColumns;
  for (const auto& sortExpr : sortedBy) {
    std::string columnName;
    bool isAscending = true;
    bool isNullsFirst = true;

    // Parse sort expression
    std::istringstream iss(sortExpr);
    iss >> columnName;

    std::string token;
    if (iss >> token) {
      if (token == "DESC") {
        isAscending = false;
      } else if (token != "ASC") {
        // If not ASC, put it back (might be NULLS)
        iss.seekg(-(int)token.length(), std::ios_base::cur);
      }

      if (iss >> token && token == "NULLS") {
        if (iss >> token && token == "LAST") {
          isNullsFirst = false;
        }
      }
    }

    core::SortOrder sortOrder(isAscending, isNullsFirst);
    IcebergSortingColumn(columnName, sortOrder);
    sortingColumns.push_back(IcebergSortingColumn(columnName, sortOrder));
  }

  return std::make_shared<IcebergInsertTableHandle>(
      columnHandles,
      locationHandle,
      partitionSpec,
      opPool_.get(),
      fileFormat_,
      sortingColumns,
      common::CompressionKind::CompressionKind_ZSTD);
}

std::shared_ptr<IcebergDataSink> IcebergTestBase::createIcebergDataSink(
    const RowTypePtr& rowType,
    const std::string& outputDirectoryPath,
    const std::vector<PartitionField>& partitionFields,
    const std::vector<std::string>& sortedBy) {
  auto tableHandle = createIcebergInsertTableHandle(
      rowType, outputDirectoryPath, partitionFields, sortedBy);
  return std::make_shared<IcebergDataSink>(
      rowType,
      tableHandle,
      connectorQueryCtx_.get(),
      connector::CommitStrategy::kNoCommit,
      connectorConfig_);
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

std::vector<std::shared_ptr<ConnectorSplit>>
IcebergTestBase::createSplitsForDirectory(const std::string& directory) {
  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  std::unordered_map<std::string, std::string> customSplitInfo;
  customSplitInfo["table_format"] = "hive-iceberg";

  auto files = listFiles(directory);
  for (const auto& filePath : files) {
    std::unordered_map<std::string, std::optional<std::string>> partitionKeys;

    // Extract partition keys from path if any.
    std::vector<std::string> pathComponents;
    folly::split("/", filePath, pathComponents);
    for (const auto& component : pathComponents) {
      if (component.find('=') != std::string::npos) {
        std::vector<std::string> keys;
        folly::split('=', component, keys);
        if (keys.size() == 2) {
          partitionKeys[keys[0]] = keys[1];
          if (keys[1] == "null") {
            partitionKeys[keys[0]] = std::nullopt;
          }
        }
      }
    }

    const auto file = filesystems::getFileSystem(filePath, nullptr)
                          ->openFileForRead(filePath);
    const auto fileSize = file->size();

    splits.push_back(std::make_shared<HiveIcebergSplit>(
        kHiveConnectorId,
        filePath,
        fileFormat_,
        0,
        fileSize,
        partitionKeys,
        std::nullopt,
        customSplitInfo,
        nullptr,
        true,
        std::vector<IcebergDeleteFile>()));
  }

  return splits;
}

} // namespace facebook::velox::connector::hive::iceberg::test
