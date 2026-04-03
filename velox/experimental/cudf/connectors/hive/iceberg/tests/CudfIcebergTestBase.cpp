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

namespace facebook::velox::cudf_velox::exec::test {

using namespace facebook::velox::connector::hive::iceberg;

void CudfIcebergTestBase::SetUp() {
  CudfHiveConnectorTestBase::SetUp();

  // Register cudf Iceberg connector.
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
  CudfHiveConnectorTestBase::TearDown();
}

std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
CudfIcebergTestBase::makeIcebergSplits(
    const std::string& dataFilePath,
    const std::vector<IcebergDeleteFile>& deleteFiles,
    const std::unordered_map<std::string, std::optional<std::string>>&
        partitionKeys,
    uint32_t splitCount) {
  auto file = filesystems::getFileSystem(dataFilePath, nullptr)
                  ->openFileForRead(dataFilePath);
  const int64_t fileSize = file->size();
  const uint64_t splitSize =
      static_cast<uint64_t>(std::floor(fileSize / splitCount));

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
      splits;
  splits.reserve(splitCount);

  for (uint32_t i = 0; i < splitCount; ++i) {
    splits.emplace_back(std::make_shared<HiveIcebergSplit>(
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
        deleteFiles));
  }

  return splits;
}

} // namespace facebook::velox::cudf_velox::exec::test
