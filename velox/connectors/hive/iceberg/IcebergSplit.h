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
#include <utility>
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

  /// Partition values keyed by the Iceberg *source* column field ID, for
  /// partition fields the serialized spec explicitly marks as the 'identity'
  /// transform. Only an identity value equals the source column's value, so
  /// only these entries may be substituted for a read of the source column.
  ///
  /// Transformed fields (bucket, truncate, day, void) are never present, even
  /// when their derived partition-field name happens to collide with a source
  /// column name. An empty map means no identity provenance was available and
  /// callers must read source columns from the data file.
  ///
  /// Distinct from the inherited name-keyed 'partitionKeys', which carries
  /// every partition field under its derived 'PartitionField.name()' and
  /// therefore cannot prove a transform is identity.
  std::unordered_map<int32_t, std::optional<std::string>> identityPartitionKeys;

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
      int64_t dataSequenceNumber = 0,
      const std::unordered_map<int32_t, std::optional<std::string>>&
          identityPartitionKeys = {},
      std::optional<dwio::common::ColumnMappingMode> columnMappingMode =
          std::nullopt);

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
      int64_t dataSequenceNumber = 0,
      const std::unordered_map<int32_t, std::optional<std::string>>&
          identityPartitionKeys = {},
      std::optional<dwio::common::ColumnMappingMode> columnMappingMode =
          std::nullopt);
};

/// Builds Iceberg splits with named parameters.
class IcebergSplitBuilder {
 public:
  explicit IcebergSplitBuilder(std::string filePath)
      : filePath_{std::move(filePath)} {
    infoColumns_["$path"] = filePath_;
  }

  IcebergSplitBuilder& connectorId(std::string id) {
    connectorId_ = std::move(id);
    return *this;
  }

  IcebergSplitBuilder& fileFormat(dwio::common::FileFormat format) {
    fileFormat_ = format;
    return *this;
  }

  IcebergSplitBuilder& start(uint64_t start) {
    start_ = start;
    return *this;
  }

  IcebergSplitBuilder& length(uint64_t length) {
    length_ = length;
    return *this;
  }

  IcebergSplitBuilder& partitionKeys(
      const std::unordered_map<std::string, std::optional<std::string>>& keys) {
    partitionKeys_ = keys;
    return *this;
  }

  IcebergSplitBuilder& infoColumns(
      const std::unordered_map<std::string, std::string>& columns) {
    for (const auto& [name, value] : columns) {
      infoColumns_[name] = value;
    }
    return *this;
  }

  IcebergSplitBuilder& deleteFiles(std::vector<IcebergDeleteFile> files) {
    deleteFiles_ = std::move(files);
    return *this;
  }

  IcebergSplitBuilder& dataSequenceNumber(int64_t sequenceNumber) {
    dataSequenceNumber_ = sequenceNumber;
    return *this;
  }

  /// Sets identity-transform partition values keyed by source field ID. See
  /// 'HiveIcebergSplit::identityPartitionKeys'.
  IcebergSplitBuilder& identityPartitionKeys(
      const std::unordered_map<int32_t, std::optional<std::string>>& keys) {
    identityPartitionKeys_ = keys;
    return *this;
  }

  IcebergSplitBuilder& columnMappingMode(dwio::common::ColumnMappingMode mode) {
    columnMappingMode_ = mode;
    return *this;
  }

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
  std::unordered_map<int32_t, std::optional<std::string>>
      identityPartitionKeys_;
  std::optional<dwio::common::ColumnMappingMode> columnMappingMode_;
};

} // namespace facebook::velox::connector::hive::iceberg
