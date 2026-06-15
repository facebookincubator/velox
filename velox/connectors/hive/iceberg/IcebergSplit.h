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
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"

namespace facebook::velox::connector::hive::iceberg {

struct HiveIcebergSplit : public connector::hive::HiveConnectorSplit {
  std::vector<IcebergDeleteFile> deleteFiles;

  /// Data sequence number of the base data file in this split. Per the Iceberg
  /// spec (V2+), an equality delete file should only apply to data files whose
  /// data sequence number is strictly less than the delete file's sequence
  /// number. A value of 0 means "unassigned" (legacy V1 tables) and disables
  /// sequence number filtering.
  int64_t dataSequenceNumber{0};

  HiveIcebergSplit(
      const std::string& connectorId,
      const std::string& filePath,
      dwio::common::FileFormat fileFormat,
      uint64_t start = 0,
      uint64_t length = std::numeric_limits<uint64_t>::max(),
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {},
      std::optional<int32_t> tableBucketNumber = std::nullopt,
      const std::unordered_map<std::string, std::string>& customSplitInfo = {},
      const std::shared_ptr<std::string>& extraFileInfo = {},
      bool cacheable = true,
      const std::unordered_map<std::string, std::string>& infoColumns = {},
      std::optional<FileProperties> fileProperties = std::nullopt,
      int64_t dataSequenceNumber = 0);

  // For tests only
  HiveIcebergSplit(
      const std::string& connectorId,
      const std::string& filePath,
      dwio::common::FileFormat fileFormat,
      uint64_t start = 0,
      uint64_t length = std::numeric_limits<uint64_t>::max(),
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {},
      std::optional<int32_t> tableBucketNumber = std::nullopt,
      const std::unordered_map<std::string, std::string>& customSplitInfo = {},
      const std::shared_ptr<std::string>& extraFileInfo = {},
      bool cacheable = true,
      std::vector<IcebergDeleteFile> deletes = {},
      const std::unordered_map<std::string, std::string>& infoColumns = {},
      std::optional<FileProperties> fileProperties = std::nullopt,
      int64_t dataSequenceNumber = 0);
};

/// Builds HiveIcebergSplit instances with named parameters.
class HiveIcebergSplitBuilder {
 public:
  explicit HiveIcebergSplitBuilder(std::string filePath);

  HiveIcebergSplitBuilder& connectorId(std::string connectorId);

  HiveIcebergSplitBuilder& fileFormat(dwio::common::FileFormat fileFormat);

  HiveIcebergSplitBuilder& start(uint64_t start);

  HiveIcebergSplitBuilder& length(uint64_t length);

  HiveIcebergSplitBuilder& partitionKeys(
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys);

  HiveIcebergSplitBuilder& infoColumns(
      const std::unordered_map<std::string, std::string>& infoColumns);

  HiveIcebergSplitBuilder& deleteFiles(
      std::vector<IcebergDeleteFile> deleteFiles);

  HiveIcebergSplitBuilder& dataSequenceNumber(int64_t dataSequenceNumber);

  std::shared_ptr<HiveIcebergSplit> build() const;

 private:
  const std::string filePath_;
  std::string connectorId_;
  dwio::common::FileFormat fileFormat_{dwio::common::FileFormat::DWRF};
  uint64_t start_{0};
  uint64_t length_{std::numeric_limits<uint64_t>::max()};
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys_;
  std::unordered_map<std::string, std::string> infoColumns_;
  std::vector<IcebergDeleteFile> deleteFiles_;
  int64_t dataSequenceNumber_{0};
};

} // namespace facebook::velox::connector::hive::iceberg
