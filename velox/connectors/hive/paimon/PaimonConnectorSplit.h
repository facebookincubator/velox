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
#include "velox/type/Variant.h"

namespace facebook::velox::connector::hive::paimon {

using PaimonPartition = std::unordered_map<int32_t, variant>;

/// Table semantics are explicit in both the table handle and the split.
/// Split planning and snapshot selection belong to the caller.
enum class PaimonTableType {
  /// No primary key. Actual read capability also depends on layout and options
  /// such as Data Evolution and deletion vectors.
  kAppendOnly,
  /// Primary-key table whose versions may require merge-on-read.
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

/// One complete logical input in a partition / bucket. For primary-key MOR,
/// overlapping versions must stay in the same split. File format is a physical
/// property; table semantics and schema identities are independent of it.
class PaimonConnectorSplit : public connector::ConnectorSplit {
 public:
  /// @param connectorId Connector identifier.
  /// @param snapshotId Paimon table snapshot version this split was generated
  ///        from.
  /// @param tableType Whether this is an append-only or primary-key table.
  /// @param fileFormat File format of the data files (e.g., ORC, Parquet).
  /// @param dataFiles Data files in this split, each representing a physical
  ///        file in the LSM-tree.
  /// @param partitionKeys Partition key-value pairs. Keys map to partition
  ///        column names; values are nullopt for null partitions.
  /// @param tableBucketNumber Split bucket ID, not the table's configured
  ///        bucket count. Optional for legacy append input.
  /// @param rawConvertible Planner hint that the split permits raw reading.
  ///        The executor also validates schema, options and deletion metadata.
  /// @param partitionValues Typed partition constants keyed by field ID;
  ///        mutually exclusive with legacy string partitionKeys.
  PaimonConnectorSplit(
      const std::string& connectorId,
      int64_t snapshotId,
      PaimonTableType tableType,
      dwio::common::FileFormat fileFormat,
      const std::vector<PaimonDataFile>& dataFiles,
      std::unordered_map<std::string, std::optional<std::string>> partitionKeys,
      std::optional<int32_t> tableBucketNumber,
      bool rawConvertible = true,
      bool cacheable = true,
      PaimonPartition partitionValues = {},
      std::string readMode = "SNAPSHOT",
      int32_t wireVersion = 1);

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

  const PaimonPartition& partitionValues() const {
    return partitionValues_;
  }

  int32_t wireVersion() const {
    return wireVersion_;
  }

  const std::string& readMode() const {
    return readMode_;
  }

  uint64_t size() const override {
    return size_;
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
  const PaimonPartition partitionValues_;
  const std::string readMode_;
  const int32_t wireVersion_;
  uint64_t size_{0};
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
