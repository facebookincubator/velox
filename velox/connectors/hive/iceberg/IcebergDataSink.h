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

#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"

namespace facebook::velox::connector::hive::iceberg {

/// Represents a request for Iceberg write.
class IcebergInsertTableHandle final : public HiveInsertTableHandle {
 public:
  /// @param inputColumns Columns from the table schema to write.
  /// The input RowVector must have the same number of columns and matching
  /// types in the same order.
  /// Column names in the RowVector may differ from those in inputColumns,
  /// only position and type must align. All columns present in the input
  /// data must be included, mismatches can lead to write failure.
  /// @param locationHandle Contains the target location information including:
  /// - Base directory path where data files will be written.
  /// - File naming scheme and temporary directory paths.
  /// @param tableStorageFormat File format to use for writing data files.
  /// @param partitionSpec Optional partition specification defining how to
  /// partition the data. If nullptr, the table is unpartitioned and all data
  /// is written to a single directory.
  /// @param compressionKind Optional compression to apply to data files.
  /// @param serdeParameters Additional serialization/deserialization parameters
  /// for the file format.
  IcebergInsertTableHandle(
      std::vector<HiveColumnHandlePtr> inputColumns,
      LocationHandlePtr locationHandle,
      dwio::common::FileFormat tableStorageFormat,
      IcebergPartitionSpecPtr partitionSpec,
      std::optional<common::CompressionKind> compressionKind = {},
      const std::unordered_map<std::string, std::string>& serdeParameters = {});

#ifdef VELOX_ENABLE_BACKWARD_COMPATIBILITY
  IcebergInsertTableHandle(
      std::vector<HiveColumnHandlePtr> inputColumns,
      LocationHandlePtr locationHandle,
      dwio::common::FileFormat tableStorageFormat,
      std::optional<common::CompressionKind> compressionKind = {},
      const std::unordered_map<std::string, std::string>& serdeParameters = {})
      : IcebergInsertTableHandle(
            inputColumns,
            locationHandle,
            tableStorageFormat,
            nullptr,
            compressionKind,
            serdeParameters) {}
#endif

  /// Returns the Iceberg partition specification that defines how the table
  /// is partitioned.
  const IcebergPartitionSpecPtr& partitionSpec() const {
    return partitionSpec_;
  }

 private:
  const IcebergPartitionSpecPtr partitionSpec_;
};

using IcebergInsertTableHandlePtr =
    std::shared_ptr<const IcebergInsertTableHandle>;

class IcebergDataSink : public HiveDataSink {
 public:
  IcebergDataSink(
      RowTypePtr inputType,
      IcebergInsertTableHandlePtr insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const HiveConfig>& hiveConfig);

  /// Generates Iceberg-specific commit messages for all writers containing
  /// metadata about written files. Creates a JSON object for each writer
  /// in the format expected by Presto and Spark for Iceberg tables.
  ///
  /// Each commit message contains:
  /// - path: full file path where data was written.
  /// - fileSizeInBytes: raw bytes written to disk.
  /// - metrics: object with recordCount (number of rows written).
  /// - partitionSpecJson: partition specification.
  /// - fileFormat: storage format (e.g., "PARQUET").
  /// - content: file content type ("DATA" for data files).
  ///
  /// See
  /// https://github.com/prestodb/presto/blob/master/presto-iceberg/src/main/java/com/facebook/presto/iceberg/CommitTaskData.java
  ///
  /// Note: Complete Iceberg metrics are not yet implemented, which results in
  /// incomplete manifest files that may lead to suboptimal query planning.
  ///
  /// @return Vector of JSON strings, one per writer, formatted according to
  /// Presto and Spark Iceberg commit protocol.
  std::vector<std::string> commitMessage() const override;
};

} // namespace facebook::velox::connector::hive::iceberg
