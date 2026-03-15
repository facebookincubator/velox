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

#include <string>

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

  /// Positional update files attached to this split. Each update file contains
  /// (file_path, pos, <all output columns>) for merge-on-read full-row updates.
  /// Following the standard Iceberg MoR pattern, update files carry complete
  /// rows — all output columns are present and replace the base row values at
  /// matching positions.
  ///
  /// NOTE: "Positional updates" are a Velox-side merge-on-read extension, not
  /// part of the Iceberg V2 spec (which achieves updates via delete + insert).
  ///
  /// Reuses IcebergDeleteFile because the metadata shape (path, format, record
  /// count, bounds) is identical; the content field distinguishes the file type
  /// via FileContent::kPositionalUpdates.
  std::vector<IcebergDeleteFile> updateFiles;

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
      std::optional<FileProperties> fileProperties = std::nullopt);

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
      std::vector<IcebergDeleteFile> updates = {},
      int64_t dataSequenceNumber = 0);
};

} // namespace facebook::velox::connector::hive::iceberg
