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

#include <utility>

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
    int64_t dataSequenceNumber)
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
      dataSequenceNumber(dataSequenceNumber) {
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
    int64_t dataSequenceNumber)
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
      dataSequenceNumber(dataSequenceNumber) {}

HiveIcebergSplitBuilder::HiveIcebergSplitBuilder(std::string filePath)
    : filePath_{std::move(filePath)} {
  infoColumns_["$path"] = filePath_;
}

HiveIcebergSplitBuilder& HiveIcebergSplitBuilder::connectorId(
    std::string connectorId) {
  connectorId_ = std::move(connectorId);
  return *this;
}

HiveIcebergSplitBuilder& HiveIcebergSplitBuilder::fileFormat(
    dwio::common::FileFormat fileFormat) {
  fileFormat_ = fileFormat;
  return *this;
}

HiveIcebergSplitBuilder& HiveIcebergSplitBuilder::start(uint64_t start) {
  start_ = start;
  return *this;
}

HiveIcebergSplitBuilder& HiveIcebergSplitBuilder::length(uint64_t length) {
  length_ = length;
  return *this;
}

HiveIcebergSplitBuilder& HiveIcebergSplitBuilder::partitionKeys(
    const std::unordered_map<std::string, std::optional<std::string>>&
        partitionKeys) {
  partitionKeys_ = partitionKeys;
  return *this;
}

HiveIcebergSplitBuilder& HiveIcebergSplitBuilder::infoColumns(
    const std::unordered_map<std::string, std::string>& infoColumns) {
  for (const auto& [name, value] : infoColumns) {
    infoColumns_[name] = value;
  }
  return *this;
}

HiveIcebergSplitBuilder& HiveIcebergSplitBuilder::deleteFiles(
    std::vector<IcebergDeleteFile> deleteFiles) {
  deleteFiles_ = std::move(deleteFiles);
  return *this;
}

HiveIcebergSplitBuilder& HiveIcebergSplitBuilder::dataSequenceNumber(
    int64_t dataSequenceNumber) {
  dataSequenceNumber_ = dataSequenceNumber;
  return *this;
}

std::shared_ptr<HiveIcebergSplit> HiveIcebergSplitBuilder::build() const {
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
      dataSequenceNumber_);
}
} // namespace facebook::velox::connector::hive::iceberg
