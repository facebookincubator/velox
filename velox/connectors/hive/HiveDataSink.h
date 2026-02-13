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
          std::make_shared<const HiveInsertFileNameGenerator>());

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
  const std::vector<column_index_t> partitionChannels_;
  const std::vector<column_index_t> nonPartitionChannels_;
};

/// Parameters for Hive writers.
class HiveWriterParameters {
 public:
  enum class UpdateMode {
    kNew, // Write files to a new directory.
    kOverwrite, // Overwrite an existing directory.
    // Append mode is currently only supported for unpartitioned tables.
    kAppend, // Append to an unpartitioned table.
  };

  /// @param updateMode Write the files to a new directory, or append to an
  /// existing directory or overwrite an existing directory.
  /// @param partitionName Partition name in the typical Hive style, which is
  /// also the partition subdirectory part of the partition path.
  /// @param targetFileName The final name of a file after committing.
  /// @param targetDirectory The final directory that a file should be in after
  /// committing.
  /// @param writeFileName The temporary name of the file that a running writer
  /// writes to. If a running writer writes directory to the target file, set
  /// writeFileName to targetFileName by default.
  /// @param writeDirectory The temporary directory that a running writer writes
  /// to. If a running writer writes directory to the target directory, set
  /// writeDirectory to targetDirectory by default.
  HiveWriterParameters(
      UpdateMode updateMode,
      std::optional<std::string> partitionName,
      std::string targetFileName,
      std::string targetDirectory,
      std::optional<std::string> writeFileName = std::nullopt,
      std::optional<std::string> writeDirectory = std::nullopt)
      : updateMode_(updateMode),
        partitionName_(std::move(partitionName)),
        targetFileName_(std::move(targetFileName)),
        targetDirectory_(std::move(targetDirectory)),
        writeFileName_(writeFileName.value_or(targetFileName_)),
        writeDirectory_(writeDirectory.value_or(targetDirectory_)) {}

  UpdateMode updateMode() const {
    return updateMode_;
  }

  static std::string updateModeToString(UpdateMode updateMode) {
    switch (updateMode) {
      case UpdateMode::kNew:
        return "NEW";
      case UpdateMode::kOverwrite:
        return "OVERWRITE";
      case UpdateMode::kAppend:
        return "APPEND";
      default:
        VELOX_UNSUPPORTED("Unsupported update mode.");
    }
  }

  const std::optional<std::string>& partitionName() const {
    return partitionName_;
  }

  const std::string& targetFileName() const {
    return targetFileName_;
  }

  const std::string& writeFileName() const {
    return writeFileName_;
  }

  const std::string& targetDirectory() const {
    return targetDirectory_;
  }

  const std::string& writeDirectory() const {
    return writeDirectory_;
  }

 private:
  const UpdateMode updateMode_;
  const std::optional<std::string> partitionName_;
  const std::string targetFileName_;
  const std::string targetDirectory_;
  const std::string writeFileName_;
  const std::string writeDirectory_;
};

/// Information about a single file written as part of a writer's output.
/// When file rotation occurs, multiple HiveFileInfo entries are created.
struct HiveFileInfo {
  /// The temporary file name used during writing (in the staging directory).
  std::string writeFileName;
  /// The final file name after commit (in the target directory).
  std::string targetFileName;
  /// Size of the file in bytes.
  uint64_t fileSize{0};
};

struct HiveWriterInfo {
  HiveWriterInfo(
      HiveWriterParameters parameters,
      std::shared_ptr<memory::MemoryPool> _writerPool,
      std::shared_ptr<memory::MemoryPool> _sinkPool,
      std::shared_ptr<memory::MemoryPool> _sortPool)
      : writerParameters(std::move(parameters)),
        nonReclaimableSectionHolder(new tsan_atomic<bool>(false)),
        spillStats(std::make_unique<exec::SpillStats>()),
        writerPool(std::move(_writerPool)),
        sinkPool(std::move(_sinkPool)),
        sortPool(std::move(_sortPool)) {}

  const HiveWriterParameters writerParameters;
  const std::unique_ptr<tsan_atomic<bool>> nonReclaimableSectionHolder;
  /// Collects the spill stats from sort writer if the spilling has been
  /// triggered.
  const std::unique_ptr<exec::SpillStats> spillStats;
  const std::shared_ptr<memory::MemoryPool> writerPool;
  const std::shared_ptr<memory::MemoryPool> sinkPool;
  const std::shared_ptr<memory::MemoryPool> sortPool;
  int64_t numWrittenRows = 0;
  int64_t inputSizeInBytes = 0;
  /// File sequence number for tracking multiple files written due to size-based
  /// splitting. Incremented each time the writer rotates to a new file.
  /// Used to generate sequenced file names (e.g., file_1.orc, file_2.orc).
  /// Invariant during write: fileSequenceNumber == writtenFiles.size()
  /// After close: fileSequenceNumber + 1 == writtenFiles.size() (final file
  /// added)
  uint32_t fileSequenceNumber{0};
  /// Tracks all files written by this writer.
  /// During write: contains only rotated (completed) files.
  /// After close: contains all files including the final one (via
  /// finalizeWriterFile).
  std::vector<HiveFileInfo> writtenFiles;
  /// Snapshot of total bytes written at the start of the current file.
  /// Used as baseline to calculate current file size: rawBytesWritten() - this.
  /// Updated to ioStats->rawBytesWritten() after each rotation.
  uint64_t cumulativeWrittenBytes{0};
  /// Current file's write filename (set when file is created/rotated).
  /// This avoids recomputing makeSequencedFileName() in commitMessage().
  std::string currentWriteFileName;
  /// Current file's target filename (set when file is created/rotated).
  std::string currentTargetFileName;
};

/// Identifies a hive writer.
struct HiveWriterId {
  std::optional<uint32_t> partitionId{std::nullopt};
  std::optional<uint32_t> bucketId{std::nullopt};

  HiveWriterId() = default;

  HiveWriterId(
      std::optional<uint32_t> _partitionId,
      std::optional<uint32_t> _bucketId = std::nullopt)
      : partitionId(_partitionId), bucketId(_bucketId) {}

  /// Returns the special writer id for the un-partitioned (and non-bucketed)
  /// table.
  static const HiveWriterId& unpartitionedId();

  std::string toString() const;

  bool operator==(const HiveWriterId& other) const {
    return std::tie(partitionId, bucketId) ==
        std::tie(other.partitionId, other.bucketId);
  }
};

struct HiveWriterIdHasher {
  std::size_t operator()(const HiveWriterId& id) const {
    return bits::hashMix(
        id.partitionId.value_or(std::numeric_limits<uint32_t>::max()),
        id.bucketId.value_or(std::numeric_limits<uint32_t>::max()));
  }
};

struct HiveWriterIdEq {
  bool operator()(const HiveWriterId& lhs, const HiveWriterId& rhs) const {
    return lhs == rhs;
  }
};

class HiveDataSink : public DataSink {
 public:
  /// The list of runtime stats reported by hive data sink
  static constexpr const char* kEarlyFlushedRawBytes = "earlyFlushedRawBytes";

  /// Defines the execution states of a hive data sink running internally.
  enum class State {
    /// The data sink accepts new append data in this state.
    kRunning = 0,
    /// The data sink flushes any buffered data to the underlying file writer
    /// but no more data can be appended.
    kFinishing = 1,
    /// The data sink is aborted on error and no more data can be appended.
    kAborted = 2,
    /// The data sink is closed on error and no more data can be appended.
    kClosed = 3
  };
  static std::string stateString(State state);

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

  void appendData(RowVectorPtr input) override;

  bool finish() override;

  Stats stats() const override;

  std::unordered_map<std::string, RuntimeCounter> runtimeStats() const override;

  std::vector<std::string> close() override;

  void abort() override;

  bool canReclaim() const;

 protected:
  // Validates the state transition from 'oldState' to 'newState'.
  void checkStateTransition(State oldState, State newState);
  void setState(State newState);

  // Generates commit messages for all writers containing metadata about written
  // files. Creates a JSON object for each writer with partition name,
  // file paths, file names, data sizes, and row counts. This metadata is used
  // by the coordinator to commit the transaction and update the metastore.
  //
  // @return Vector of JSON strings, one per writer.
  virtual std::vector<std::string> commitMessage() const;

  class WriterReclaimer : public exec::MemoryReclaimer {
   public:
    static std::unique_ptr<memory::MemoryReclaimer> create(
        HiveDataSink* dataSink,
        HiveWriterInfo* writerInfo,
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
        HiveWriterInfo* writerInfo,
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
    HiveWriterInfo* const writerInfo_;
    io::IoStatistics* const ioStats_;
  };

  FOLLY_ALWAYS_INLINE bool sortWrite() const {
    return !sortColumnIndices_.empty();
  }

  // Returns true if the table is partitioned.
  FOLLY_ALWAYS_INLINE bool isPartitioned() const {
    return partitionIdGenerator_ != nullptr;
  }

  // Returns true if the table is bucketed.
  FOLLY_ALWAYS_INLINE bool isBucketed() const {
    return bucketCount_ != 0;
  }

  FOLLY_ALWAYS_INLINE bool isCommitRequired() const {
    return commitStrategy_ != CommitStrategy::kNoCommit;
  }

  std::shared_ptr<memory::MemoryPool> createWriterPool(
      const HiveWriterId& writerId);

  void setMemoryReclaimers(
      HiveWriterInfo* writerInfo,
      io::IoStatistics* ioStats);

  // Returns the bytes written to the current file for the specified writer.
  // This is calculated as total bytes minus cumulative bytes from rotated
  // files. Use this instead of rawBytesWritten() when you need current file
  // size.
  uint64_t getCurrentFileBytes(size_t writerIndex) const;

  // Compute the partition id and bucket id for each row in 'input'.
  virtual void computePartitionAndBucketIds(const RowVectorPtr& input);

  // Get the HiveWriter corresponding to the row
  // from partitionIds and bucketIds.
  HiveWriterId getWriterId(size_t row) const;

  // Computes the number of input rows as well as the actual input row indices
  // to each corresponding (bucketed) partition based on the partition and
  // bucket ids calculated by 'computePartitionAndBucketIds'. The function also
  // ensures that there is a writer created for each (bucketed) partition.
  void splitInputRowsAndEnsureWriters();

  // Makes sure to create one writer for the given writer id. The function
  // returns the corresponding index in 'writers_'.
  virtual uint32_t ensureWriter(const HiveWriterId& id);

  // Appends a new writer for the given 'id'. The function returns the index of
  // the newly created writer in 'writers_'.
  uint32_t appendWriter(const HiveWriterId& id);

  // Creates and configures WriterOptions based on file format.
  // Sets up compression, schema, and other writer configuration based on the
  // insert table handle and connector settings.
  // The no-argument overload uses the last writer's info (for appendWriter).
  virtual std::shared_ptr<dwio::common::WriterOptions> createWriterOptions()
      const;

  // Creates WriterOptions for a specific writer index. Use this overload
  // during writer rotation to ensure the correct writer's memory pool and
  // nonReclaimableSection are used.
  std::shared_ptr<dwio::common::WriterOptions> createWriterOptions(
      size_t writerIndex) const;

  // Returns the Hive partition directory name for the given partition ID.
  // Converts the partition values associated with the partition ID into a
  // Hive-formatted directory path. Returns std::nullopt if the table is
  // unpartitioned. Should be called only when writing to a partitioned table.
  virtual std::string getPartitionName(uint32_t partitionId) const;

  std::unique_ptr<facebook::velox::dwio::common::Writer>
  maybeCreateBucketSortWriter(
      std::unique_ptr<facebook::velox::dwio::common::Writer> writer);

  // Records a row index for a specific partition. This method maintains the
  // mapping of which input rows belong to which partition by storing row
  // indices in partition-specific buffers. If the buffer for the partition
  // doesn't exist or is too small, it allocates/reallocates the buffer to
  // accommodate all rows.
  void
  updatePartitionRows(uint32_t index, vector_size_t numRows, vector_size_t row);

  HiveWriterParameters getWriterParameters(
      const std::optional<std::string>& partition,
      std::optional<uint32_t> bucketId) const;

  // Gets write and target file names for a writer based on the table commit
  // strategy as well as table partitioned type. If commit is not required, the
  // write file and target file has the same name. If not, add a temp file
  // prefix to the target file for write file name. The coordinator (or driver
  // for Presto on spark) will rename the write file to target file to commit
  // the table write when update the metadata store. If it is a bucketed table,
  // the file name encodes the corresponding bucket id.
  std::pair<std::string, std::string> getWriterFileNames(
      std::optional<uint32_t> bucketId) const;

  HiveWriterParameters::UpdateMode getUpdateMode() const;

  FOLLY_ALWAYS_INLINE void checkRunning() const {
    VELOX_CHECK_EQ(state_, State::kRunning, "Hive data sink is not running");
  }

  // Invoked to write 'input' to the specified file writer.
  void write(size_t index, RowVectorPtr input);

  /// Rotates the writer at the given index to a new file. This is called when
  /// the current file exceeds maxTargetFileBytes_. The old writer is closed
  /// and a new writer is created for the same partition/bucket.
  void rotateWriter(size_t index);

  /// Finalizes the current file for the writer at the given index.
  /// Captures file stats and adds the file info to writtenFiles.
  /// Called by rotateWriter() and closeInternal().
  void finalizeWriterFile(size_t index);

  virtual void closeInternal();

  // IMPORTANT NOTE: these are passed to writers as raw pointers. HiveDataSink
  // owns the lifetime of these objects, and therefore must destroy them last.
  // Additionally, we must assume that no objects which hold a reference to
  // these stats will outlive the HiveDataSink instance. This is a reasonable
  // assumption given the semantics of these stats objects.
  std::vector<std::unique_ptr<io::IoStatistics>> ioStats_;
  // Generic filesystem stats, exposed as RuntimeStats
  std::unique_ptr<IoStats> fileSystemStats_;

  const RowTypePtr inputType_;
  const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle_;
  const ConnectorQueryCtx* const connectorQueryCtx_;
  const CommitStrategy commitStrategy_;
  const std::shared_ptr<const HiveConfig> hiveConfig_;
  const HiveWriterParameters::UpdateMode updateMode_;
  const uint32_t maxOpenWriters_;
  const std::vector<column_index_t> partitionChannels_;
  const std::unique_ptr<PartitionIdGenerator> partitionIdGenerator_;
  // Indices of dataChannel are stored in ascending order
  const std::vector<column_index_t> dataChannels_;
  const int32_t bucketCount_{0};
  const std::unique_ptr<core::PartitionFunction> bucketFunction_;
  const std::shared_ptr<dwio::common::WriterFactory> writerFactory_;
  const common::SpillConfig* const spillConfig_;
  const uint64_t sortWriterFinishTimeSliceLimitMs_{0};
  const uint64_t maxTargetFileBytes_{0};
  const bool partitionKeyAsLowerCase_;

  std::vector<column_index_t> sortColumnIndices_;
  std::vector<CompareFlags> sortCompareFlags_;

  State state_{State::kRunning};

  tsan_atomic<bool> nonReclaimableSection_{false};

  // The map from writer id to the writer index in 'writers_' and 'writerInfo_'.
  folly::F14FastMap<HiveWriterId, uint32_t, HiveWriterIdHasher, HiveWriterIdEq>
      writerIndexMap_;

  // Below are structures for partitions from all inputs. writerInfo_ and
  // writers_ are both indexed by partitionId.
  std::vector<std::shared_ptr<HiveWriterInfo>> writerInfo_;
  std::vector<std::unique_ptr<dwio::common::Writer>> writers_;

  // Below are structures updated when processing current input. partitionIds_
  // are indexed by the row of input_. partitionRows_, rawPartitionRows_ and
  // partitionSizes_ are indexed by partitionId.
  raw_vector<uint64_t> partitionIds_;
  std::vector<BufferPtr> partitionRows_;
  std::vector<vector_size_t*> rawPartitionRows_;
  std::vector<vector_size_t> partitionSizes_;

  // Reusable buffers for bucket id calculations.
  std::vector<uint32_t> bucketIds_;

  // Strategy for naming writer files
  std::shared_ptr<const FileNameGenerator> fileNameGenerator_;
};

FOLLY_ALWAYS_INLINE std::ostream& operator<<(
    std::ostream& os,
    HiveDataSink::State state) {
  os << HiveDataSink::stateString(state);
  return os;
}
} // namespace facebook::velox::connector::hive

template <>
struct fmt::formatter<facebook::velox::connector::hive::HiveDataSink::State>
    : formatter<std::string> {
  auto format(
      facebook::velox::connector::hive::HiveDataSink::State s,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::connector::hive::HiveDataSink::stateString(s), ctx);
  }
};

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
