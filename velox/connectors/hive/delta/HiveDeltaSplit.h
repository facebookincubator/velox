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

/// Delta Lake column-mapping mode
/// (https://docs.delta.io/latest/delta-column-mapping.html).
///
/// Records the source table's column-mapping mode; the field is metadata
/// carried alongside the split, not a directive that changes how the worker
/// reads today. The Presto coordinator resolves each column to its physical
/// name on the HiveColumnHandle before the split is sent (see
/// DeltaPrestoToVeloxConnector::sourceName() on the coordinator side), so
/// the worker reads by name in every mode. This matches the spec's read
/// requirement for kName ("resolve by physicalName") and works in practice
/// for kId against delta.io writers (they stamp physicalName as the Parquet
/// column name alongside field_id). It is not strictly spec-compliant for
/// kId against a spec-only-field_id writer; see the TODO in
/// DeltaSplitReader::prepareSplit for the follow-up that routes kId through
/// dwio::common::ColumnMappingMode::kParquetFieldId once field_ids are on
/// the wire.
enum class DeltaColumnMappingMode {
  /// No column mapping. Logical column names match Parquet physical names.
  kNone,

  /// Column mapping by name. Physical Parquet column names are opaque IDs
  /// (col-<uuid>). The coordinator provides those physical names on
  /// HiveColumnHandle; the worker reads by name.
  kName,

  /// Column mapping by id. Same coordinator-side name resolution as kName;
  /// spec-strict field_id resolution is a follow-up (see enum doc).
  kId,
};

/// Represents a Delta Lake data file to read. Extends HiveConnectorSplit to
/// reuse the existing Hive file reading infrastructure.
struct HiveDeltaSplit : public connector::hive::HiveConnectorSplit {
  /// @param connectorId Connector identifier.
  /// @param filePath Path to the data file.
  /// @param fileFormat File format of the data file.
  /// @param start Starting byte offset in the file.
  /// @param length Number of bytes to read.
  /// @param partitionKeys Partition column names to their values.
  /// @param tableBucketNumber Bucket number for bucketed tables.
  /// @param customSplitInfo Custom split metadata, includes
  /// table_format=hive-delta.
  /// @param extraFileInfo Additional file information.
  /// @param cacheable Whether the split data can be cached.
  /// @param infoColumns Synthesized metadata columns, e.g. $path and
  /// $file_size.
  /// @param fileProperties File properties such as row count and file size.
  /// @param hasDeletionVector Whether the file has a deletion vector marking
  /// some rows as logically deleted. Reading such a file is not yet
  /// supported: the split reader rejects it rather than silently returning
  /// logically deleted rows. When support is added, the descriptor itself
  /// (path, offset, cardinality, ...) will be added alongside this flag.
  /// @param columnMappingMode Delta column-mapping mode of the source table.
  /// Informational only today: the worker reads by name in every mode
  /// because the coordinator resolves physical names on the column
  /// handles. See the enum doc for the follow-up that will make id mode
  /// spec-strict.
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
      std::optional<FileProperties> fileProperties = std::nullopt,
      bool hasDeletionVector = false,
      DeltaColumnMappingMode columnMappingMode = DeltaColumnMappingMode::kNone);

  folly::dynamic serialize() const override;

  static std::shared_ptr<HiveDeltaSplit> create(const folly::dynamic& obj);

  static void registerSerDe();

  /// Whether this file has a deletion vector. Currently rejected by the
  /// reader; see the constructor doc.
  bool hasDeletionVector{false};

  /// Delta column-mapping mode of the source table. See the enum doc; only
  /// kNone is accepted by the reader today.
  DeltaColumnMappingMode columnMappingMode{DeltaColumnMappingMode::kNone};
};

/// Serializes a DeltaColumnMappingMode to its canonical string form
/// ("none", "name", "id").
std::string_view toString(DeltaColumnMappingMode mode);

/// Parses a canonical DeltaColumnMappingMode string. Throws
/// VeloxUserError on any unknown value.
DeltaColumnMappingMode deltaColumnMappingModeFromString(std::string_view name);

} // namespace facebook::velox::connector::hive::delta
