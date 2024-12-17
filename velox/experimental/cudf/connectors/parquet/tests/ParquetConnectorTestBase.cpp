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

#include "velox/exec/tests/utils/ParquetConnectorTestBase.h"

#include "velox/common/file/FileSystems.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/dwio/dwrf/writer/FlushPolicy.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"

namespace facebook::velox::cudf_velox::exec::test {

ParquetConnectorTestBase::ParquetConnectorTestBase() {
  filesystems::registerLocalFileSystem();
  tests::utils::registerFaultyFileSystem();
}

void ParquetConnectorTestBase::SetUp() {
  OperatorTestBase::SetUp();
  connector::registerConnectorFactory(
      std::make_shared<connector::parquet::ParquetConnectorFactory>());
  auto parquetConnector =
      connector::getConnectorFactory(
          connector::parquet::ParquetConnectorFactory::kParquetConnectorName)
          ->newConnector(
              kParquetConnectorId,
              std::make_shared<config::ConfigBase>(
                  std::unordered_map<std::string, std::string>()),
              ioExecutor_.get());
  connector::registerConnector(parquetConnector);
  // TODO: Using Velox's Parquet writer for testing until we have a DataSink in
  // ParquetConnector
  parquet::registerParquetWriterFactory();
}

void ParquetConnectorTestBase::TearDown() {
  // Make sure all pending loads are finished or cancelled before unregister
  // connector.
  ioExecutor_.reset();
  connector::unregisterConnector(kParquetConnectorId);
  connector::unregisterConnectorFactory(
      connector::parquet::ParquetConnectorFactory::kParquetConnectorName);
  // TODO: Using Velox's Parquet writer for testing until we have a DataSink in
  // ParquetConnector
  parquet::unregisterParquetWriterFactory();
  OperatorTestBase::TearDown();
}

void ParquetConnectorTestBase::resetParquetConnector(
    const std::shared_ptr<const config::ConfigBase>& config) {
  connector::unregisterConnector(kParquetConnectorId);
  auto parquetConnector =
      connector::getConnectorFactory(
          connector::parquet::ParquetConnectorFactory::kParquetConnectorName)
          ->newConnector(kParquetConnectorId, config, ioExecutor_.get());
  connector::registerConnector(parquetConnector);
}

std::vector<RowVectorPtr> ParquetConnectorTestBase::makeVectors(
    const RowTypePtr& rowType,
    int32_t numVectors,
    int32_t rowsPerVector) {
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < numVectors; ++i) {
    auto vector = std::dynamic_pointer_cast<RowVector>(
        velox::test::BatchMaker::createBatch(rowType, rowsPerVector, *pool_));
    vectors.push_back(vector);
  }
  return vectors;
}

std::shared_ptr<exec::Task> ParquetConnectorTestBase::assertQuery(
    const core::PlanNodePtr& plan,
    const std::vector<std::shared_ptr<TempFilePath>>& filePaths,
    const std::string& duckDbSql) {
  return OperatorTestBase::assertQuery(
      plan, makeParquetConnectorSplits(filePaths), duckDbSql);
}

std::shared_ptr<Task> ParquetConnectorTestBase::assertQuery(
    const core::PlanNodePtr& plan,
    const std::vector<std::shared_ptr<connector::ConnectorSplit>>& splits,
    const std::string& duckDbSql,
    const int32_t numPrefetchSplit) {
  return AssertQueryBuilder(plan, duckDbQueryRunner_)
      .config(
          core::QueryConfig::kMaxSplitPreloadPerDriver,
          std::to_string(numPrefetchSplit))
      .splits(splits)
      .assertResults(duckDbSql);
}

std::vector<std::shared_ptr<TempFilePath>>
ParquetConnectorTestBase::makeFilePaths(int count) {
  std::vector<std::shared_ptr<TempFilePath>> filePaths;

  filePaths.reserve(count);
  for (auto i = 0; i < count; ++i) {
    filePaths.emplace_back(TempFilePath::create());
  }
  return filePaths;
}

std::unique_ptr<connector::parquet::ParquetColumnHandle>
ParquetConnectorTestBase::makeColumnHandle(
    const std::string& name,
    const TypePtr& type,
    const std::vector<std::string>& requiredSubfields) {
  return makeColumnHandle(name, type, type, requiredSubfields);
}

std::unique_ptr<connector::parquet::ParquetColumnHandle>
ParquetConnectorTestBase::makeColumnHandle(
    const std::string& name,
    const TypePtr& dataType,
    const TypePtr& parquetType,
    const std::vector<std::string>& requiredSubfields,
    connector::parquet::ParquetColumnHandle::ColumnType columnType) {
  std::vector<common::Subfield> subfields;
  subfields.reserve(requiredSubfields.size());
  for (auto& path : requiredSubfields) {
    subfields.emplace_back(path);
  }

  return std::make_unique<connector::parquet::ParquetColumnHandle>(
      name, columnType, dataType, parquetType, std::move(subfields));
}

std::vector<std::shared_ptr<connector::ConnectorSplit>>
ParquetConnectorTestBase::makeParquetConnectorSplits(
    const std::vector<std::shared_ptr<TempFilePath>>& filePaths) {
  std::vector<std::shared_ptr<connector::ConnectorSplit>> splits;
  for (auto filePath : filePaths) {
    splits.push_back(makeParquetConnectorSplit(filePath->getPath()));
  }
  return splits;
}

std::shared_ptr<connector::parquet::ParquetConnectorSplit>
ParquetConnectorTestBase::makeParquetConnectorSplit(
    const std::string& filePath,
    int64_t splitWeight) {
  return ParquetConnectorSplitBuilder(filePath)
      .splitWeight(splitWeight)
      .build();
}

// static
std::shared_ptr<connector::parquet::ParquetInsertTableHandle>
ParquetConnectorTestBase::makeParquetInsertTableHandle(
    const std::vector<std::string>& tableColumnNames,
    const std::vector<TypePtr>& tableColumnTypes,
    const std::vector<std::string>& partitionedBy,
    std::shared_ptr<connector::parquet::LocationHandle> locationHandle,
    const dwio::common::FileFormat tableStorageFormat,
    const std::optional<common::CompressionKind> compressionKind,
    const std::shared_ptr<dwio::common::WriterOptions>& writerOptions) {
  return makeParquetInsertTableHandle(
      tableColumnNames,
      tableColumnTypes,
      partitionedBy,
      nullptr,
      std::move(locationHandle),
      tableStorageFormat,
      compressionKind,
      {},
      writerOptions);
}

// static
std::shared_ptr<connector::parquet::ParquetInsertTableHandle>
ParquetConnectorTestBase::makeParquetInsertTableHandle(
    const std::vector<std::string>& tableColumnNames,
    const std::vector<TypePtr>& tableColumnTypes,
    const std::vector<std::string>& partitionedBy,
    std::shared_ptr<connector::parquet::ParquetBucketProperty> bucketProperty,
    std::shared_ptr<connector::parquet::LocationHandle> locationHandle,
    const dwio::common::FileFormat tableStorageFormat,
    const std::optional<common::CompressionKind> compressionKind,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    const std::shared_ptr<dwio::common::WriterOptions>& writerOptions) {
  std::vector<std::shared_ptr<const connector::parquet::ParquetColumnHandle>>
      columnHandles;
  std::vector<std::string> bucketedBy;
  std::vector<TypePtr> bucketedTypes;
  std::vector<std::shared_ptr<const connector::parquet::ParquetSortingColumn>>
      sortedBy;
  if (bucketProperty != nullptr) {
    bucketedBy = bucketProperty->bucketedBy();
    bucketedTypes = bucketProperty->bucketedTypes();
    sortedBy = bucketProperty->sortedBy();
  }
  int32_t numPartitionColumns{0};
  int32_t numSortingColumns{0};
  int32_t numBucketColumns{0};
  for (int i = 0; i < tableColumnNames.size(); ++i) {
    for (int j = 0; j < bucketedBy.size(); ++j) {
      if (bucketedBy[j] == tableColumnNames[i]) {
        ++numBucketColumns;
      }
    }
    for (int j = 0; j < sortedBy.size(); ++j) {
      if (sortedBy[j]->sortColumn() == tableColumnNames[i]) {
        ++numSortingColumns;
      }
    }
    if (std::find(
            partitionedBy.cbegin(),
            partitionedBy.cend(),
            tableColumnNames.at(i)) != partitionedBy.cend()) {
      ++numPartitionColumns;
      columnHandles.push_back(std::make_shared<
                              connector::parquet::ParquetColumnHandle>(
          tableColumnNames.at(i),
          connector::parquet::ParquetColumnHandle::ColumnType::kPartitionKey,
          tableColumnTypes.at(i),
          tableColumnTypes.at(i)));
    } else {
      columnHandles.push_back(
          std::make_shared<connector::parquet::ParquetColumnHandle>(
              tableColumnNames.at(i),
              connector::parquet::ParquetColumnHandle::ColumnType::kRegular,
              tableColumnTypes.at(i),
              tableColumnTypes.at(i)));
    }
  }
  VELOX_CHECK_EQ(numPartitionColumns, partitionedBy.size());
  VELOX_CHECK_EQ(numBucketColumns, bucketedBy.size());
  VELOX_CHECK_EQ(numSortingColumns, sortedBy.size());

  return std::make_shared<connector::parquet::ParquetInsertTableHandle>(
      columnHandles,
      locationHandle,
      tableStorageFormat,
      bucketProperty,
      compressionKind,
      serdeParameters,
      writerOptions);
}

std::shared_ptr<connector::parquet::ParquetColumnHandle>
ParquetConnectorTestBase::regularColumn(
    const std::string& name,
    const TypePtr& type) {
  return std::make_shared<connector::parquet::ParquetColumnHandle>(
      name,
      connector::parquet::ParquetColumnHandle::ColumnType::kRegular,
      type,
      type);
}

std::shared_ptr<connector::parquet::ParquetColumnHandle>
ParquetConnectorTestBase::synthesizedColumn(
    const std::string& name,
    const TypePtr& type) {
  return std::make_shared<connector::parquet::ParquetColumnHandle>(
      name,
      connector::parquet::ParquetColumnHandle::ColumnType::kSynthesized,
      type,
      type);
}

} // namespace facebook::velox::cudf_velox::exec::test
