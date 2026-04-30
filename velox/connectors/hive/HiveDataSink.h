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

#include "velox/common/compression/Compression.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileDataSink.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HivePartitionName.h"
#include "velox/connectors/hive/PartitionIdGenerator.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Writer.h"
#include "velox/dwio/common/WriterFactory.h"
#include "velox/exec/MemoryReclaimer.h"

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

  bool operator==(const HiveBucketProperty& other) const {
    return true;
  }

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
  /// generating file names
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

  /// Replaces potentially unsafe characters in a file name with underscores
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
      const bool ensureFiles = false,
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

  /// Format specific options.
  const std::unordered_map<std::string, std::string>& serdeParameters() const {
    return serdeParameters_;
  }

  /// Storage specific options.
  const std::unordered_map<std::string, std::string>& storageParameters()
      const {
    return storageParameters_;
  }

  /// Avoid this in future usages. Format specific change should go through
  /// serdeParameters.
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

/// JSON field names for the partition update object produced by each writer
/// and consumed by the Presto coordinator to finalize files and update the
/// metastore.
///
/// JSON structure:
/// {
///   "name":                        "<partition key, e.g. ds=2024-01-01>",
///   "updateMode":                  "NEW" | "APPEND" | "OVERWRITE",
///   "writePath":                   "<staging directory>",
///   "targetPath":                  "<final directory>",
///   "fileWriteInfos": [
///     {
///       "writeFileName":           "<temp filename in writePath>",
///       "targetFileName":          "<final filename in targetPath>",
///       "fileSize":                <bytes>
///     }
///   ],
///   "rowCount":                    <total rows>,
///   "inMemoryDataSizeInBytes":     <uncompressed bytes>,
///   "onDiskDataSizeInBytes":       <compressed bytes on disk>,
///   "containsNumberedFileNames":   true | false
/// }
struct HiveCommitMessage {
  /// Partition directory name in Hive format (e.g., "ds=2024-01-01/region=us").
  /// Empty string for unpartitioned tables.
  static constexpr const char* kName = "name";
  /// Write mode: "NEW", "APPEND", or "OVERWRITE". Controls how the committer
  /// handles metastore updates and existing file conflicts.
  static constexpr const char* kUpdateMode = "updateMode";
  /// Staging directory where files were written during execution.
  static constexpr const char* kWritePath = "writePath";
  /// Final destination directory. Files are renamed from writePath to
  /// targetPath during commit.
  static constexpr const char* kTargetPath = "targetPath";
  /// Array of per-file metadata objects. One entry per file written, including
  /// rotated files.
  static constexpr const char* kFileWriteInfos = "fileWriteInfos";
  /// Temporary filename used during writing (in the staging directory).
  static constexpr const char* kWriteFileName = "writeFileName";
  /// Final filename after commit (in the target directory).
  static constexpr const char* kTargetFileName = "targetFileName";
  /// Size of individual file in bytes.
  static constexpr const char* kFileSize = "fileSize";
  /// Total rows written to this partition across all files.
  static constexpr const char* kRowCount = "rowCount";
  /// Uncompressed input data size in bytes.
  static constexpr const char* kInMemoryDataSizeInBytes =
      "inMemoryDataSizeInBytes";
  /// Compressed bytes written to disk.
  static constexpr const char* kOnDiskDataSizeInBytes = "onDiskDataSizeInBytes";
  /// Whether filenames follow a numbered sequence from file rotation.
  static constexpr const char* kContainsNumberedFileNames =
      "containsNumberedFileNames";
};

class HiveDataSink : public FileDataSink {
 public:
  /// The list of runtime stats reported by hive data sink
  static constexpr const char* kEarlyFlushedRawBytes = "earlyFlushedRawBytes";

  /// Creates a HiveDataSink for writing data to Hive table files.
  ///
  /// @param inputType The schema of input data rows to be written.
  /// @param insertTableHandle Metadata about the table write operation,
  /// including storage format, compression, bucketing, and partitioning
  /// configuration.
  /// @param connectorQueryCtx Query context with session properties, memory
  /// pools, and spill configuration.
  /// @param commitStrategy Strategy for committing written data (kNoCommit or
  /// kTaskCommit).
  /// @param hiveConfig Hive connector configuration.
  HiveDataSink(
      RowTypePtr inputType,
      std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const HiveConfig>& hiveConfig);

  /// Constructor with explicit bucketing and partitioning parameters.
  ///
  /// @param inputType The schema of input data rows to be written.
  /// @param insertTableHandle Metadata about the table write operation,
  /// including storage format, compression, location, and serialization
  /// parameters.
  /// @param connectorQueryCtx Query context with session properties, memory
  /// pools, and spill configuration.
  /// @param commitStrategy Strategy for committing written data (kNoCommit or
  /// kTaskCommit). Determines whether temporary files need to be renamed on
  /// commit.
  /// @param hiveConfig Hive connector configuration with settings for max
  /// partitions, bucketing limits etc.
  /// @param bucketCount Number of buckets for bucketed tables (0 if not
  /// bucketed). Must be less than the configured max bucket count.
  /// @param bucketFunction Function to compute bucket IDs from row data
  /// (nullptr if not bucketed). Used to distribute rows across buckets.
  /// @param partitionChannels Column indices used for partitioning (empty if
  /// not partitioned). These columns are extracted to determine partition
  /// directories.
  /// @param dataChannels Column indices for the actual data columns to be
  /// written.
  /// @param partitionIdGenerator Generates partition IDs from partition column
  /// values (nullptr if not partitioned). Compute partition key combinations to
  /// unique IDs.
  HiveDataSink(
      RowTypePtr inputType,
      std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      uint32_t bucketCount,
      std::unique_ptr<core::PartitionFunction> bucketFunction,
      const std::vector<column_index_t>& partitionChannels,
      const std::vector<column_index_t>& dataChannels,
      std::unique_ptr<PartitionIdGenerator> partitionIdGenerator);

  bool canReclaim() const;

 protected:
  std::vector<std::string> commitMessage() const override;

  class WriterReclaimer : public exec::MemoryReclaimer {
   public:
    static std::unique_ptr<memory::MemoryReclaimer> create(
        HiveDataSink* dataSink,
        WriterInfo* writerInfo,
        io::IoStatistics* ioStats);

    bool reclaimableBytes(
        const memory::MemoryPool& pool,
        uint64_t& reclaimableBytes) const override;

    uint64_t reclaim(
        memory::MemoryPool* pool,
        uint64_t targetBytes,
        uint64_t maxWaitMs,
        memory::MemoryReclaimer::Stats& stats) override;

   private:
    WriterReclaimer(
        HiveDataSink* dataSink,
        WriterInfo* writerInfo,
        io::IoStatistics* ioStats)
        : exec::MemoryReclaimer(0),
          dataSink_(dataSink),
          writerInfo_(writerInfo),
          ioStats_(ioStats) {
      VELOX_CHECK_NOT_NULL(dataSink_);
      VELOX_CHECK_NOT_NULL(writerInfo_);
      VELOX_CHECK_NOT_NULL(ioStats_);
    }

    HiveDataSink* const dataSink_;
    WriterInfo* const writerInfo_;
    io::IoStatistics* const ioStats_;
  };

  void setMemoryReclaimers(WriterInfo* writerInfo, io::IoStatistics* ioStats)
      override;

  // Compute the partition id and bucket id for each row in 'input'.
  void computePartitionAndBucketIds(const RowVectorPtr& input) override;

  std::unique_ptr<facebook::velox::dwio::common::Writer> createWriterForIndex(
      size_t writerIndex) override;

  // Creates and configures WriterOptions based on file format.
  std::shared_ptr<dwio::common::WriterOptions> createWriterOptions()
      const override;

  virtual std::shared_ptr<dwio::common::WriterOptions> createWriterOptions(
      size_t writerIndex) const override;

  // Returns the Hive partition directory name for the given partition ID.
  virtual std::string getPartitionName(uint32_t partitionId) const override;

  std::unique_ptr<facebook::velox::dwio::common::Writer>
  maybeCreateBucketSortWriter(
      size_t writerIndex,
      std::unique_ptr<facebook::velox::dwio::common::Writer> writer);

  WriterParameters getWriterParameters(
      const std::optional<std::string>& partition,
      std::optional<uint32_t> bucketId) const override;

  // Gets write and target file names for a writer.
  std::pair<std::string, std::string> getWriterFileNames(
      std::optional<uint32_t> bucketId) const;

  WriterParameters::UpdateMode getUpdateMode() const;

  const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle_;
  const std::shared_ptr<const HiveConfig> hiveConfig_;
  const WriterParameters::UpdateMode updateMode_;

  std::vector<column_index_t> sortColumnIndices_;
  std::vector<CompareFlags> sortCompareFlags_;

  // Strategy for naming writer files
  std::shared_ptr<const FileNameGenerator> fileNameGenerator_;
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
