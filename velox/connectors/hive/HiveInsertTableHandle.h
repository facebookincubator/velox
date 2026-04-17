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
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Portability.h"
#include "velox/common/compression/Compression.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/common/Options.h"

namespace facebook::velox::connector::hive {

class HiveConfig;
class LocationHandle;
using LocationHandlePtr = std::shared_ptr<const LocationHandle>;

/// Location related properties of the Hive table to be written.
class LocationHandle : public ISerializable {
 public:
  enum class TableType {
    /// Write to a new table to be created.
    kNew,
    /// Write to an existing table.
    kExisting,
  };

  LocationHandle(
      std::string targetPath,
      std::string writePath,
      TableType tableType,
      std::string targetFileName = "")
      : targetPath_(std::move(targetPath)),
        targetFileName_(std::move(targetFileName)),
        writePath_(std::move(writePath)),
        tableType_(tableType) {}

  const std::string& targetPath() const {
    return targetPath_;
  }

  const std::string& targetFileName() const {
    return targetFileName_;
  }

  const std::string& writePath() const {
    return writePath_;
  }

  TableType tableType() const {
    return tableType_;
  }

  std::string toString() const;

  static void registerSerDe();

  folly::dynamic serialize() const override;

  static LocationHandlePtr create(const folly::dynamic& obj);

  static const std::string tableTypeName(LocationHandle::TableType type);

  static LocationHandle::TableType tableTypeFromName(const std::string& name);

 private:
  // Target directory path.
  const std::string targetPath_;
  // If non-empty, use this name instead of generating our own.
  const std::string targetFileName_;
  // Staging directory path.
  const std::string writePath_;
  // Whether the table to be written is new, already existing or temporary.
  const TableType tableType_;
};

class HiveSortingColumn : public ISerializable {
 public:
  HiveSortingColumn(
      const std::string& sortColumn,
      const core::SortOrder& sortOrder);

  const std::string& sortColumn() const {
    return sortColumn_;
  }

  core::SortOrder sortOrder() const {
    return sortOrder_;
  }

  folly::dynamic serialize() const override;

  static std::shared_ptr<HiveSortingColumn> deserialize(
      const folly::dynamic& obj,
      void* context);

  std::string toString() const;

  static void registerSerDe();

 private:
  const std::string sortColumn_;
  const core::SortOrder sortOrder_;
};

class HiveBucketProperty : public ISerializable {
 public:
  enum class Kind { kHiveCompatible, kPrestoNative };

  HiveBucketProperty(
      Kind kind,
      int32_t bucketCount,
      const std::vector<std::string>& bucketedBy,
      const std::vector<TypePtr>& bucketedTypes,
      const std::vector<std::shared_ptr<const HiveSortingColumn>>& sortedBy);

  Kind kind() const {
    return kind_;
  }

  static std::string kindString(Kind kind);

  /// Returns the number of bucket count.
  int32_t bucketCount() const {
    return bucketCount_;
  }

  /// Returns the bucketed by column names.
  const std::vector<std::string>& bucketedBy() const {
    return bucketedBy_;
  }

  /// Returns the bucketed by column types.
  const std::vector<TypePtr>& bucketedTypes() const {
    return bucketTypes_;
  }

  /// Returns the hive sorting columns if not empty.
  const std::vector<std::shared_ptr<const HiveSortingColumn>>& sortedBy()
      const {
    return sortedBy_;
  }

  folly::dynamic serialize() const override;

  static std::shared_ptr<HiveBucketProperty> deserialize(
      const folly::dynamic& obj,
      void* context);

  bool operator==(const HiveBucketProperty& other) const;

  static void registerSerDe();

  std::string toString() const;

 private:
  void validate() const;

  const Kind kind_;
  const int32_t bucketCount_;
  const std::vector<std::string> bucketedBy_;
  const std::vector<TypePtr> bucketTypes_;
  const std::vector<std::shared_ptr<const HiveSortingColumn>> sortedBy_;
};

FOLLY_ALWAYS_INLINE std::ostream& operator<<(
    std::ostream& os,
    HiveBucketProperty::Kind kind) {
  os << HiveBucketProperty::kindString(kind);
  return os;
}

class HiveInsertTableHandle;
using HiveInsertTableHandlePtr = std::shared_ptr<HiveInsertTableHandle>;

class FileNameGenerator : public ISerializable {
 public:
  virtual ~FileNameGenerator() = default;

  virtual std::pair<std::string, std::string> gen(
      std::optional<uint32_t> bucketId,
      const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx& connectorQueryCtx,
      bool commitRequired) const = 0;

  virtual std::string toString() const = 0;
};

class HiveInsertFileNameGenerator : public FileNameGenerator {
 public:
  HiveInsertFileNameGenerator() {}

  std::pair<std::string, std::string> gen(
      std::optional<uint32_t> bucketId,
      const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx& connectorQueryCtx,
      bool commitRequired) const override;

  /// Version of file generation that takes hiveConfig into account when
  /// generating file names.
  std::pair<std::string, std::string> gen(
      std::optional<uint32_t> bucketId,
      const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx& connectorQueryCtx,
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      bool commitRequired) const;

  static void registerSerDe();

  folly::dynamic serialize() const override;

  static std::shared_ptr<HiveInsertFileNameGenerator> deserialize(
      const folly::dynamic& obj,
      void* context);

  std::string toString() const override;

  /// Replaces potentially unsafe characters in a file name with underscores.
  static void sanitizeFileName(std::string& name);
};

/// Represents a request for Hive write.
class HiveInsertTableHandle : public ConnectorInsertTableHandle {
 public:
  HiveInsertTableHandle(
      std::vector<std::shared_ptr<const HiveColumnHandle>> inputColumns,
      std::shared_ptr<const LocationHandle> locationHandle,
      dwio::common::FileFormat storageFormat = dwio::common::FileFormat::DWRF,
      std::shared_ptr<const HiveBucketProperty> bucketProperty = nullptr,
      std::optional<common::CompressionKind> compressionKind = {},
      const std::unordered_map<std::string, std::string>& serdeParameters = {},
      const std::shared_ptr<dwio::common::WriterOptions>& writerOptions =
          nullptr,
      // When this option is set the HiveDataSink will always write a file even
      // if there's no data. This is useful when the table is bucketed, but the
      // engine handles ensuring a 1 to 1 mapping from task to bucket.
      bool ensureFiles = false,
      std::shared_ptr<const FileNameGenerator> fileNameGenerator =
          std::make_shared<const HiveInsertFileNameGenerator>(),
      const std::unordered_map<std::string, std::string>& storageParameters =
          {});

  virtual ~HiveInsertTableHandle() = default;

  const std::vector<std::shared_ptr<const HiveColumnHandle>>& inputColumns()
      const {
    return inputColumns_;
  }

  const std::shared_ptr<const LocationHandle>& locationHandle() const {
    return locationHandle_;
  }

  std::optional<common::CompressionKind> compressionKind() const {
    return compressionKind_;
  }

  dwio::common::FileFormat storageFormat() const {
    return storageFormat_;
  }

  const std::unordered_map<std::string, std::string>& serdeParameters() const {
    return serdeParameters_;
  }

  /// Returns storage specific options.
  const std::unordered_map<std::string, std::string>& storageParameters()
      const {
    return storageParameters_;
  }

  const std::shared_ptr<dwio::common::WriterOptions>& writerOptions() const {
    return writerOptions_;
  }

  bool ensureFiles() const {
    return ensureFiles_;
  }

  const std::shared_ptr<const FileNameGenerator>& fileNameGenerator() const {
    return fileNameGenerator_;
  }

  bool supportsMultiThreading() const override {
    return true;
  }

  bool isPartitioned() const;

  bool isBucketed() const;

  const HiveBucketProperty* bucketProperty() const;

  bool isExistingTable() const;

  /// Returns a subset of column indices corresponding to partition keys.
  const std::vector<column_index_t>& partitionChannels() const {
    return partitionChannels_;
  }

  /// Returns the column indices of non-partition data columns.
  const std::vector<column_index_t>& nonPartitionChannels() const {
    return nonPartitionChannels_;
  }

  folly::dynamic serialize() const override;

  static HiveInsertTableHandlePtr create(const folly::dynamic& obj);

  static void registerSerDe();

  std::string toString() const override;

 protected:
  const std::vector<std::shared_ptr<const HiveColumnHandle>> inputColumns_;
  const std::shared_ptr<const LocationHandle> locationHandle_;

 private:
  const dwio::common::FileFormat storageFormat_;
  const std::shared_ptr<const HiveBucketProperty> bucketProperty_;
  const std::optional<common::CompressionKind> compressionKind_;
  const std::unordered_map<std::string, std::string> serdeParameters_;
  const std::shared_ptr<dwio::common::WriterOptions> writerOptions_;
  const bool ensureFiles_;
  const std::shared_ptr<const FileNameGenerator> fileNameGenerator_;
  const std::unordered_map<std::string, std::string> storageParameters_;
  const std::vector<column_index_t> partitionChannels_;
  const std::vector<column_index_t> nonPartitionChannels_;
};

} // namespace facebook::velox::connector::hive

template <>
struct fmt::formatter<
    facebook::velox::connector::hive::LocationHandle::TableType>
    : formatter<int> {
  auto format(
      facebook::velox::connector::hive::LocationHandle::TableType s,
      format_context& ctx) const {
    return formatter<int>::format(static_cast<int>(s), ctx);
  }
};
