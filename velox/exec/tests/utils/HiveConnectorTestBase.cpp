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

#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

namespace facebook::velox::exec::test {

HiveConnectorTestBase::HiveConnectorTestBase() {
  filesystems::registerLocalFileSystem();
}

void HiveConnectorTestBase::SetUp() {
  OperatorTestBase::SetUp();
  auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(kHiveConnectorId, nullptr, ioExecutor_.get());
  connector::registerConnector(hiveConnector);
  dwrf::registerDwrfReaderFactory();
}

void HiveConnectorTestBase::TearDown() {
  // Make sure all pending loads are finished or cancelled before unregister
  // connector.
  ioExecutor_.reset();
  dwrf::unregisterDwrfReaderFactory();
  connector::unregisterConnector(kHiveConnectorId);
  OperatorTestBase::TearDown();
}

void HiveConnectorTestBase::writeToFile(
    const std::string& filePath,
    RowVectorPtr vector) {
  writeToFile(filePath, std::vector{vector});
}

void HiveConnectorTestBase::writeToFile(
    const std::string& filePath,
    const std::vector<RowVectorPtr>& vectors,
    std::shared_ptr<dwrf::Config> config) {
  facebook::velox::dwrf::WriterOptions options;
  options.config = config;
  options.schema = vectors[0]->type();
  auto sink =
      std::make_unique<facebook::velox::dwio::common::LocalFileSink>(filePath);
  auto childPool = rootPool_->addChild(
      "HiveConnectorTestBase.Writer", memory::MemoryPool::Kind::kAggregate);
  facebook::velox::dwrf::Writer writer{options, std::move(sink), *childPool};
  for (size_t i = 0; i < vectors.size(); ++i) {
    writer.write(vectors[i]);
  }
  writer.close();
}

std::vector<RowVectorPtr> HiveConnectorTestBase::makeVectors(
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

std::shared_ptr<exec::Task> HiveConnectorTestBase::assertQuery(
    const core::PlanNodePtr& plan,
    const std::vector<std::shared_ptr<TempFilePath>>& filePaths,
    const std::string& duckDbSql) {
  return OperatorTestBase::assertQuery(
      plan, makeHiveConnectorSplits(filePaths), duckDbSql);
}

std::vector<std::shared_ptr<TempFilePath>> HiveConnectorTestBase::makeFilePaths(
    int count) {
  std::vector<std::shared_ptr<TempFilePath>> filePaths;

  filePaths.reserve(count);
  for (auto i = 0; i < count; ++i) {
    filePaths.emplace_back(TempFilePath::create());
  }
  return filePaths;
}

std::vector<std::shared_ptr<connector::hive::HiveConnectorSplit>>
HiveConnectorTestBase::makeHiveConnectorSplits(
    const std::string& filePath,
    uint32_t splitCount,
    dwio::common::FileFormat format) {
  const int fileSize = fs::file_size(filePath);
  // Take the upper bound.
  const int splitSize = std::ceil((fileSize) / splitCount);
  std::vector<std::shared_ptr<connector::hive::HiveConnectorSplit>> splits;

  // Add all the splits.
  for (int i = 0; i < splitCount; i++) {
    auto split = HiveConnectorSplitBuilder(filePath)
                     .fileFormat(format)
                     .start(i * splitSize)
                     .length(splitSize)
                     .build();
    splits.push_back(std::move(split));
  }
  return splits;
}

std::vector<std::shared_ptr<connector::ConnectorSplit>>
HiveConnectorTestBase::makeHiveConnectorSplits(
    const std::vector<std::shared_ptr<TempFilePath>>& filePaths) {
  std::vector<std::shared_ptr<connector::ConnectorSplit>> splits;
  for (auto filePath : filePaths) {
    splits.push_back(makeHiveConnectorSplit(filePath->path));
  }
  return splits;
}

std::shared_ptr<connector::ConnectorSplit>
HiveConnectorTestBase::makeHiveConnectorSplit(
    const std::string& filePath,
    uint64_t start,
    uint64_t length) {
  return HiveConnectorSplitBuilder(filePath)
      .start(start)
      .length(length)
      .build();
}

// static
std::shared_ptr<connector::hive::HiveInsertTableHandle>
HiveConnectorTestBase::makeHiveInsertTableHandle(
    const std::vector<std::string>& tableColumnNames,
    const std::vector<TypePtr>& tableColumnTypes,
    const std::vector<std::string>& partitionedBy,
    std::shared_ptr<connector::hive::LocationHandle> locationHandle) {
  std::vector<std::shared_ptr<const connector::hive::HiveColumnHandle>>
      columnHandles;
  for (int i = 0; i < tableColumnNames.size(); ++i) {
    if (std::find(
            partitionedBy.cbegin(),
            partitionedBy.cend(),
            tableColumnNames.at(i)) != partitionedBy.cend()) {
      columnHandles.push_back(
          std::make_shared<connector::hive::HiveColumnHandle>(
              tableColumnNames.at(i),
              connector::hive::HiveColumnHandle::ColumnType::kPartitionKey,
              tableColumnTypes.at(i)));
    } else {
      columnHandles.push_back(
          std::make_shared<connector::hive::HiveColumnHandle>(
              tableColumnNames.at(i),
              connector::hive::HiveColumnHandle::ColumnType::kRegular,
              tableColumnTypes.at(i)));
    }
  }

  return std::make_shared<connector::hive::HiveInsertTableHandle>(
      columnHandles, locationHandle);
}

std::shared_ptr<connector::hive::HiveColumnHandle>
HiveConnectorTestBase::regularColumn(
    const std::string& name,
    const TypePtr& type) {
  return std::make_shared<connector::hive::HiveColumnHandle>(
      name, connector::hive::HiveColumnHandle::ColumnType::kRegular, type);
}

std::shared_ptr<connector::hive::HiveColumnHandle>
HiveConnectorTestBase::synthesizedColumn(
    const std::string& name,
    const TypePtr& type) {
  return std::make_shared<connector::hive::HiveColumnHandle>(
      name, connector::hive::HiveColumnHandle::ColumnType::kSynthesized, type);
}

std::shared_ptr<connector::hive::HiveColumnHandle>
HiveConnectorTestBase::partitionKey(
    const std::string& name,
    const TypePtr& type) {
  return std::make_shared<connector::hive::HiveColumnHandle>(
      name, connector::hive::HiveColumnHandle::ColumnType::kPartitionKey, type);
}

} // namespace facebook::velox::exec::test
