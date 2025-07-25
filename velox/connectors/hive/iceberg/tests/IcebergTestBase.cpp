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
    const std::vector<std::string>& transformSpecs) {
  std::vector<IcebergPartitionSpec::Field> fields;
  static const std::regex identityRegex(R"(([a-z_][a-z0-9_]*))");

  for (const auto& spec : transformSpecs) {
    TransformType transformType = TransformType::kIdentity;
    std::string name;
    std::smatch matches;

    if (std::regex_match(spec, matches, identityRegex)) {
      transformType = TransformType::kIdentity;
      name = matches[1];
    } else {
      VELOX_FAIL("Unsupported transform specification: {}", spec);
    }

    fields.push_back(
        IcebergPartitionSpec::Field(name, transformType, std::nullopt));
  }

  return std::make_shared<IcebergPartitionSpec>(1, fields);
}

std::shared_ptr<IcebergInsertTableHandle>
IcebergTestBase::createIcebergInsertTableHandle(
    const RowTypePtr& rowType,
    const std::string& outputDirectoryPath,
    const std::vector<std::string>& partitionTransforms) {
  std::vector<std::shared_ptr<const HiveColumnHandle>> columnHandles;
  for (auto i = 0; i < rowType->size(); ++i) {
    auto columnName = rowType->nameOf(i);
    auto columnType = HiveColumnHandle::ColumnType::kRegular;
    for (auto transform : partitionTransforms) {
      if (columnName == transform) {
        columnType = HiveColumnHandle::ColumnType::kPartitionKey;
        break;
      }
    }
    columnHandles.push_back(std::make_shared<HiveColumnHandle>(
        columnName, columnType, rowType->childAt(i), rowType->childAt(i)));
  }

  auto locationHandle = std::make_shared<LocationHandle>(
      outputDirectoryPath,
      outputDirectoryPath,
      LocationHandle::TableType::kNew);

  auto partitionSpec = createPartitionSpec(partitionTransforms);

  return std::make_shared<IcebergInsertTableHandle>(
      columnHandles,
      locationHandle,
      partitionSpec,
      fileFormat_,
      nullptr,
      common::CompressionKind::CompressionKind_ZSTD);
}

std::shared_ptr<IcebergDataSink> IcebergTestBase::createIcebergDataSink(
    const RowTypePtr& rowType,
    const std::string& outputDirectoryPath,
    const std::vector<std::string>& partitionTransforms) {
  auto tableHandle = createIcebergInsertTableHandle(
      rowType, outputDirectoryPath, partitionTransforms);
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
        /*cacheable=*/true,
        std::vector<IcebergDeleteFile>()));
  }

  return splits;
}

} // namespace facebook::velox::connector::hive::iceberg::test
