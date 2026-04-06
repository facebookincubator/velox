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

#include "velox/experimental/cudf/connectors/hive/iceberg/tests/CudfIcebergTestBase.h"

#include "velox/common/file/FileSystems.h"
#include "velox/dwio/dwrf/RegisterDwrfReader.h"
#include "velox/dwio/dwrf/RegisterDwrfWriter.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::cudf_velox::exec::test {

using namespace facebook::velox::connector::hive::iceberg;
using namespace facebook::velox::exec::test;

void CudfIcebergTestBase::SetUp() {
  CudfHiveConnectorTestBase::SetUp();

  // Register DWRF reader/writer factories so that the upstream
  // PositionalDeleteFileReader and EqualityDeleteFileReader can read DWRF
  // delete files.
  dwrf::registerDwrfReaderFactory();
  dwrf::registerDwrfWriterFactory();

  connector::hive::iceberg::CudfIcebergConnectorFactory factory;
  auto icebergConnector = factory.newConnector(
      kCudfIcebergConnectorId,
      std::make_shared<facebook::velox::config::ConfigBase>(
          std::unordered_map<std::string, std::string>()),
      ioExecutor_.get());
  facebook::velox::connector::registerConnector(icebergConnector);
}

void CudfIcebergTestBase::TearDown() {
  facebook::velox::connector::unregisterConnector(kCudfIcebergConnectorId);
  dwrf::unregisterDwrfReaderFactory();
  dwrf::unregisterDwrfWriterFactory();
  CudfHiveConnectorTestBase::TearDown();
}

std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
CudfIcebergTestBase::makeIcebergSplits(
    const std::string& dataFilePath,
    const std::vector<IcebergDeleteFile>& deleteFiles,
    const std::unordered_map<std::string, std::optional<std::string>>&
        partitionKeys,
    uint32_t splitCount,
    int64_t dataSequenceNumber) {
  auto file = filesystems::getFileSystem(dataFilePath, nullptr)
                  ->openFileForRead(dataFilePath);
  const int64_t fileSize = file->size();
  const uint64_t splitSize =
      static_cast<uint64_t>(std::floor(fileSize / splitCount));

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
      splits;
  splits.reserve(splitCount);

  for (uint32_t i = 0; i < splitCount; ++i) {
    splits.emplace_back(
        std::make_shared<HiveIcebergSplit>(
            kCudfIcebergConnectorId,
            dataFilePath,
            dwio::common::FileFormat::PARQUET,
            i * splitSize,
            splitSize,
            partitionKeys,
            std::nullopt,
            std::unordered_map<std::string, std::string>{},
            nullptr,
            /*cacheable=*/true,
            deleteFiles,
            std::unordered_map<std::string, std::string>{},
            std::nullopt,
            dataSequenceNumber));
  }

  return splits;
}

void CudfIcebergTestBase::writeDeleteFile(
    const std::string& filePath,
    const std::vector<RowVectorPtr>& vectors) {
  // Uses the upstream velox::dwrf::Writer — same as
  // HiveConnectorTestBase::writeToFile.
  velox::dwrf::WriterOptions options;
  options.config = std::make_shared<facebook::velox::dwrf::Config>();
  options.schema = vectors[0]->type();
  auto fs = filesystems::getFileSystem(filePath, {});
  auto writeFile = fs->openFileForWrite(
      filePath,
      {.shouldCreateParentDirectories = true,
       .shouldThrowOnFileAlreadyExists = false});
  auto sink = std::make_unique<dwio::common::WriteFileSink>(
      std::move(writeFile), filePath);
  auto childPool =
      rootPool_->addAggregateChild("CudfIcebergTestBase.DwrfWriter");
  options.memoryPool = childPool.get();

  facebook::velox::dwrf::Writer writer{std::move(sink), options};
  for (const auto& vector : vectors) {
    writer.write(vector);
  }
  writer.close();
}

uint64_t CudfIcebergTestBase::getFileSize(const std::string& path) {
  return filesystems::getFileSystem(path, nullptr)
      ->openFileForRead(path)
      ->size();
}

core::PlanNodePtr CudfIcebergTestBase::makeTableScanPlan(
    const RowTypePtr& rowType) {
  return PlanBuilder()
      .startTableScan()
      .connectorId(kCudfIcebergConnectorId)
      .outputType(rowType)
      .dataColumns(rowType)
      .endTableScan()
      .planNode();
}

} // namespace facebook::velox::cudf_velox::exec::test
