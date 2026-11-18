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

#include "velox/common/EnumDeclare.h"
#include "velox/common/compression/Compression.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/HiveBucketProperty.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Options.h"

namespace facebook::velox::connector::hive {

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

  VELOX_DECLARE_EMBEDDED_ENUM_NAME(TableType);

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

class FileNameGenerator;
using FileNameGeneratorPtr = std::shared_ptr<const FileNameGenerator>;

class HiveInsertTableHandle;
using HiveInsertTableHandlePtr = std::shared_ptr<const HiveInsertTableHandle>;

/// Names the files a HiveDataSink writes. A HiveInsertTableHandle carries one,
/// so naming is part of the plan. Connectors with their own naming scheme
/// supply a subclass, e.g. Iceberg and Prism deltas.
class FileNameGenerator : public ISerializable {
 public:
  virtual ~FileNameGenerator() = default;

  /// Returns the names of one output file: the target name, which the file has
  /// in the target directory once committed, and the write name, under which
  /// it is written to the staging directory.
  /// @param bucketId The bucket the file holds, if the table is bucketed.
  /// @param maxNumBuckets Upper bound on the number of buckets, used to
  /// zero-pad the bucket id in bucketed file names to a fixed width. Resolved
  /// from the configurable hive.max-bucket-count. Generators that do not
  /// produce bucketed names may ignore it.
  /// @param commitRequired Whether the file is committed by a rename. If not,
  /// the write name is the target name.
  virtual std::pair<std::string, std::string> gen(
      std::optional<uint32_t> bucketId,
      const HiveInsertTableHandlePtr& insertTableHandle,
      const ConnectorQueryCtx& connectorQueryCtx,
      uint32_t maxNumBuckets,
      bool commitRequired) const = 0;

  virtual std::string toString() const = 0;
};

/// The default FileNameGenerator. The target name is:
/// - the LocationHandle's target file name, if set, used as is. A bucketed
///   write must not set one;
/// - for a bucketed write, '0<bucket>_0_<query id>', with the bucket id
///   zero-padded to the width of the largest bucket id;
/// - otherwise '<task id>_<driver id>_<plan node id>_<uuid>', unique across
///   the writers of a task and across task retries.
/// Characters other than letters, digits, '.', '_' and '-' in a generated name
/// become '_'. If a commit is required, the write name is
/// '.tmp.velox.<target name>_<uuid>'; otherwise it is the target name. A
/// generated name of a Parquet file ends in '.parquet'.
class HiveInsertFileNameGenerator : public FileNameGenerator {
 public:
  std::pair<std::string, std::string> gen(
      std::optional<uint32_t> bucketId,
      const HiveInsertTableHandlePtr& insertTableHandle,
      const ConnectorQueryCtx& connectorQueryCtx,
      uint32_t maxNumBuckets,
      bool commitRequired) const override;

  static void registerSerDe();

  folly::dynamic serialize() const override;

  static std::shared_ptr<HiveInsertFileNameGenerator> deserialize(
      const folly::dynamic& obj,
      void* context);

  std::string toString() const override;
};

/// Represents a request for Hive write.
class HiveInsertTableHandle : public ConnectorInsertTableHandle {
 public:
  /// @param insertExistingPartitionsBehavior What a write to an existing
  /// partition of an existing partitioned table does. The writers report
  /// update mode NEW for kError, so committing an existing partition fails,
  /// and OVERWRITE for kOverwrite, so the commit replaces the partition.
  /// @param writeEmptyFile Whether the sink writes a file even if it receives
  /// no rows. Not supported for a bucketed or partitioned write. An engine
  /// that assigns each task one bucket uses it, without a bucket property, so
  /// that every bucket has a file.
  HiveInsertTableHandle(
      std::vector<HiveColumnHandlePtr> inputColumns,
      std::shared_ptr<const LocationHandle> locationHandle,
      HiveConfig::InsertExistingPartitionsBehavior
          insertExistingPartitionsBehavior,
      dwio::common::FileFormat storageFormat = dwio::common::FileFormat::DWRF,
      HiveBucketPropertyPtr bucketProperty = nullptr,
      std::optional<common::CompressionKind> compressionKind = {},
      const std::unordered_map<std::string, std::string>& serdeParameters = {},
      bool writeEmptyFile = false,
      FileNameGeneratorPtr fileNameGenerator =
          std::make_shared<const HiveInsertFileNameGenerator>(),
      const std::unordered_map<std::string, std::string>& storageParameters =
          {});

#ifdef VELOX_ENABLE_BACKWARD_COMPATIBILITY
  /// Leaves the behavior for existing partitions unset: the writers read it
  /// from the session property 'insert_existing_partitions_behavior'.
  /// 'writerOptions' must be null.
  HiveInsertTableHandle(
      std::vector<HiveColumnHandlePtr> inputColumns,
      std::shared_ptr<const LocationHandle> locationHandle,
      dwio::common::FileFormat storageFormat = dwio::common::FileFormat::DWRF,
      HiveBucketPropertyPtr bucketProperty = nullptr,
      std::optional<common::CompressionKind> compressionKind = {},
      const std::unordered_map<std::string, std::string>& serdeParameters = {},
      const std::shared_ptr<dwio::common::WriterOptions>& writerOptions =
          nullptr,
      const bool ensureFiles = false,
      FileNameGeneratorPtr fileNameGenerator =
          std::make_shared<const HiveInsertFileNameGenerator>(),
      const std::unordered_map<std::string, std::string>& storageParameters =
          {})
      : HiveInsertTableHandle(
            std::move(inputColumns),
            std::move(locationHandle),
            std::nullopt,
            storageFormat,
            std::move(bucketProperty),
            compressionKind,
            serdeParameters,
            ensureFiles,
            std::move(fileNameGenerator),
            storageParameters) {
    VELOX_CHECK_NULL(writerOptions);
  }
#endif

  virtual ~HiveInsertTableHandle() = default;

  const std::vector<HiveColumnHandlePtr>& inputColumns() const {
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

  /// The table's Hive SerDe properties, e.g. 'field.delim' of a TEXT table.
  /// The writers receive them as dwio::common::WriterOptions::serdeParameters.
  const std::unordered_map<std::string, std::string>& serdeParameters() const {
    return serdeParameters_;
  }

  /// Options for the file system the files are written to, passed to each
  /// file sink as dwio::common::FileSink::Options::storageParameters. The file
  /// system defines which keys it reads and what they mean.
  const std::unordered_map<std::string, std::string>& storageParameters()
      const {
    return storageParameters_;
  }

  bool writeEmptyFile() const {
    return writeEmptyFile_;
  }

  /// Unset for a handle built by the backward-compatible constructor and for
  /// a serialized handle without it. The writers then read the session
  /// property 'insert_existing_partitions_behavior'.
  std::optional<HiveConfig::InsertExistingPartitionsBehavior>
  insertExistingPartitionsBehavior() const {
    return insertExistingPartitionsBehavior_;
  }

  const FileNameGeneratorPtr& fileNameGenerator() const {
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
  const std::vector<HiveColumnHandlePtr> inputColumns_;
  const std::shared_ptr<const LocationHandle> locationHandle_;

 private:
  HiveInsertTableHandle(
      std::vector<HiveColumnHandlePtr> inputColumns,
      std::shared_ptr<const LocationHandle> locationHandle,
      std::optional<HiveConfig::InsertExistingPartitionsBehavior>
          insertExistingPartitionsBehavior,
      dwio::common::FileFormat storageFormat,
      HiveBucketPropertyPtr bucketProperty,
      std::optional<common::CompressionKind> compressionKind,
      const std::unordered_map<std::string, std::string>& serdeParameters,
      bool writeEmptyFile,
      FileNameGeneratorPtr fileNameGenerator,
      const std::unordered_map<std::string, std::string>& storageParameters);

  const std::optional<HiveConfig::InsertExistingPartitionsBehavior>
      insertExistingPartitionsBehavior_;
  const dwio::common::FileFormat storageFormat_;
  const HiveBucketPropertyPtr bucketProperty_;
  const std::optional<common::CompressionKind> compressionKind_;
  const std::unordered_map<std::string, std::string> serdeParameters_;
  const bool writeEmptyFile_;
  const FileNameGeneratorPtr fileNameGenerator_;
  const std::unordered_map<std::string, std::string> storageParameters_;
  const std::vector<column_index_t> partitionChannels_;
  const std::vector<column_index_t> nonPartitionChannels_;
};

} // namespace facebook::velox::connector::hive

VELOX_EMBEDDED_ENUM_FORMATTER(
    facebook::velox::connector::hive::LocationHandle,
    TableType);
