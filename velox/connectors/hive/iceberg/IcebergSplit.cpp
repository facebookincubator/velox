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

#include "velox/connectors/hive/iceberg/IcebergSplit.h"

#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"

namespace facebook::velox::connector::hive::iceberg {

HiveIcebergSplit::HiveIcebergSplit(
    const std::string& connectorId,
    const std::string& filePath,
    dwio::common::FileFormat fileFormat,
    uint64_t start,
    uint64_t length,
    const std::unordered_map<std::string, std::optional<std::string>>&
        partitionKeys,
    std::optional<int32_t> tableBucketNumber,
    const std::unordered_map<std::string, std::string>& customSplitInfo,
    const std::shared_ptr<std::string>& extraFileInfo,
    bool cacheable,
    const std::unordered_map<std::string, std::string>& infoColumns,
    std::optional<FileProperties> properties,
    int64_t dataSequenceNumber,
    std::shared_ptr<ChangelogSplitInfo> changelogInfo)
    : HiveConnectorSplit(
          connectorId,
          filePath,
          fileFormat,
          start,
          length,
          partitionKeys,
          tableBucketNumber,
          customSplitInfo,
          extraFileInfo,
          /*serdeParameters=*/{},
          /*splitWeight=*/0,
          cacheable,
          infoColumns,
          properties,
          std::nullopt,
          std::nullopt),
      dataSequenceNumber(dataSequenceNumber),
      changelogSplitInfo(std::move(changelogInfo)) {
  // TODO: Deserialize _extraFileInfo to get deleteFiles;
}

// For tests only
HiveIcebergSplit::HiveIcebergSplit(
    const std::string& connectorId,
    const std::string& filePath,
    dwio::common::FileFormat fileFormat,
    uint64_t start,
    uint64_t length,
    const std::unordered_map<std::string, std::optional<std::string>>&
        partitionKeys,
    std::optional<int32_t> tableBucketNumber,
    const std::unordered_map<std::string, std::string>& customSplitInfo,
    const std::shared_ptr<std::string>& extraFileInfo,
    bool cacheable,
    std::vector<IcebergDeleteFile> deletes,
    const std::unordered_map<std::string, std::string>& infoColumns,
    std::optional<FileProperties> properties,
    int64_t dataSequenceNumber,
    std::shared_ptr<ChangelogSplitInfo> changelogInfo)
    : HiveConnectorSplit(
          connectorId,
          filePath,
          fileFormat,
          start,
          length,
          partitionKeys,
          tableBucketNumber,
          customSplitInfo,
          extraFileInfo,
          /*serdeParameters=*/{},
          0,
          cacheable,
          infoColumns,
          properties,
          std::nullopt,
          std::nullopt),
      deleteFiles(std::move(deletes)),
      dataSequenceNumber(dataSequenceNumber),
      changelogSplitInfo(std::move(changelogInfo)) {}

std::shared_ptr<HiveIcebergSplit> IcebergSplitBuilder::build() const {
  return std::make_shared<HiveIcebergSplit>(
      connectorId_,
      filePath_,
      fileFormat_,
      start_,
      length_,
      partitionKeys_,
      std::nullopt,
      std::unordered_map<std::string, std::string>{},
      nullptr,
      /*cacheable=*/true,
      deleteFiles_,
      infoColumns_,
      std::nullopt,
      dataSequenceNumber_,
      changelogSplitInfo_);
}
} // namespace facebook::velox::connector::hive::iceberg
