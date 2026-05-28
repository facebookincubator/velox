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

namespace facebook::velox::connector::hive::delta {

/// Delta Lake split for reading data files.
/// Extends HiveConnectorSplit to leverage existing Hive file reading infrastructure.
struct HiveDeltaSplit : public connector::hive::HiveConnectorSplit {
  /// Constructor for Delta Lake splits
  /// @param connectorId Connector identifier
  /// @param filePath Path to the data file
  /// @param fileFormat File format (Parquet, ORC, etc.)
  /// @param start Starting byte offset in the file
  /// @param length Number of bytes to read
  /// @param partitionKeys Map of partition column names to values
  /// @param tableBucketNumber Optional bucket number for bucketed tables
  /// @param customSplitInfo Custom split metadata (includes table_format=hive-delta)
  /// @param extraFileInfo Additional file information
  /// @param cacheable Whether the split data can be cached
  /// @param infoColumns Additional metadata columns (e.g., file path, modification time)
  /// @param fileProperties Optional file properties (row count, file size, etc.)
  HiveDeltaSplit(
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
};

} // namespace facebook::velox::connector::hive::delta

