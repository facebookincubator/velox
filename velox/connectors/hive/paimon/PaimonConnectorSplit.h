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

#include <fmt/format.h>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <folly/CPortability.h>

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/paimon/PaimonDataFileMeta.h"
#include "velox/dwio/common/Options.h"

/// @file
/// Paimon logical split representation.
///
/// PaimonConnectorSplit represents a planner-selected Paimon DataSplit: a
/// logical read unit with one or more files from the same partition/bucket.
/// Unlike a single-file split, it can group multiple physical files. A bucket
/// may produce several splits; a split need not contain every file in it.
///
/// The planner supplies the file set and its snapshot identity. Files may
/// span LSM levels; when logical merging is needed, the necessary files must
/// remain together in the split. The split does not discover files from
/// storage or select snapshots.
///
/// partitionKeys carries this split's partition values, including nulls,
/// rather than the table's partition column declaration. dataFiles carries
/// file paths and per-file metadata. PaimonTableHandle separately carries
/// the table definition and scan conditions.
///
/// rawConvertible indicates that the planner-selected files can be read
/// without logical merging. It does not imply support for every file feature:
/// deletion vectors and changelog files require their own read semantics.

namespace facebook::velox::connector::hive::paimon {

/// Classifies a Paimon table by whether it declares primary keys.
enum class PaimonTableType {
  /// Table without a primary key.
  kAppendOnly,
  /// Table with a primary key.
  kPrimaryKey,
};

/// Returns the string name of the table type (e.g., "APPEND_ONLY").
std::string paimonTableTypeString(PaimonTableType type);

/// Parses a table type from its string name.
PaimonTableType paimonTableTypeFromString(const std::string& str);

FOLLY_ALWAYS_INLINE std::ostream& operator<<(
    std::ostream& os,
    PaimonTableType type) {
  os << paimonTableTypeString(type);
  return os;
}

} // namespace facebook::velox::connector::hive::paimon

template <>
struct fmt::formatter<facebook::velox::connector::hive::paimon::PaimonTableType>
    : formatter<std::string> {
  auto format(
      facebook::velox::connector::hive::paimon::PaimonTableType type,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::connector::hive::paimon::paimonTableTypeString(type),
        ctx);
  }
};

namespace facebook::velox::connector::hive::paimon {

/// Logical read unit carrying planner-selected files, partition values,
/// snapshot identity and raw-read eligibility. See the file-level design above.
class PaimonConnectorSplit : public connector::ConnectorSplit {
 public:
  /// @param connectorId Connector identifier.
  /// @param snapshotId Snapshot ID associated with the planner-selected files.
  /// @param tableType Whether this is an append-only or primary-key table.
  /// @param fileFormat File format of the data files (e.g., ORC, Parquet).
  /// @param dataFiles Non-empty collection of files in this logical split.
  /// @param partitionKeys Partition key-value pairs. Keys map to partition
  ///        column names; values are nullopt for null partitions.
  /// @param tableBucketNumber Optional bucket identifier for this split.
  /// @param rawConvertible Planner-provided flag indicating that this split
  ///        can be read directly without logical merging.
  PaimonConnectorSplit(
      const std::string& connectorId,
      int64_t snapshotId,
      PaimonTableType tableType,
      dwio::common::FileFormat fileFormat,
      const std::vector<PaimonDataFile>& dataFiles,
      std::unordered_map<std::string, std::optional<std::string>> partitionKeys,
      std::optional<int32_t> tableBucketNumber,
      bool rawConvertible = true);

  int64_t snapshotId() const {
    return snapshotId_;
  }

  PaimonTableType tableType() const {
    return tableType_;
  }

  const std::vector<PaimonDataFile>& dataFiles() const {
    return dataFiles_;
  }

  const std::unordered_map<std::string, std::optional<std::string>>&
  partitionKeys() const {
    return partitionKeys_;
  }

  std::optional<int32_t> tableBucketNumber() const {
    return tableBucketNumber_;
  }

  bool rawConvertible() const {
    return rawConvertible_;
  }

  dwio::common::FileFormat fileFormat() const {
    return fileFormat_;
  }

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static std::shared_ptr<PaimonConnectorSplit> create(
      const folly::dynamic& obj);

  static void registerSerDe();

 private:
  const int64_t snapshotId_;
  const PaimonTableType tableType_;
  const dwio::common::FileFormat fileFormat_;
  const std::vector<PaimonDataFile> dataFiles_;
  const std::unordered_map<std::string, std::optional<std::string>>
      partitionKeys_;
  const std::optional<int32_t> tableBucketNumber_;
  const bool rawConvertible_;
};

/// Builder for PaimonConnectorSplit construction.
class PaimonConnectorSplitBuilder {
 public:
  PaimonConnectorSplitBuilder(
      std::string connectorId,
      int64_t snapshotId,
      PaimonTableType tableType,
      dwio::common::FileFormat fileFormat)
      : connectorId_(std::move(connectorId)),
        snapshotId_(snapshotId),
        tableType_(tableType),
        fileFormat_(fileFormat) {}

  PaimonConnectorSplitBuilder&
  addFile(std::string filePath, uint64_t fileSize, int32_t level = 0);

  PaimonConnectorSplitBuilder& partitionKey(
      std::string name,
      std::optional<std::string> value);

  PaimonConnectorSplitBuilder& tableBucketNumber(int32_t bucketId);

  PaimonConnectorSplitBuilder& rawConvertible(bool value);

  std::shared_ptr<PaimonConnectorSplit> build();

 private:
  const std::string connectorId_;
  const int64_t snapshotId_;
  const PaimonTableType tableType_;
  const dwio::common::FileFormat fileFormat_;
  std::vector<PaimonDataFile> dataFiles_;
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys_;
  std::optional<int32_t> tableBucketNumber_;
  bool rawConvertible_{true};
};

} // namespace facebook::velox::connector::hive::paimon
