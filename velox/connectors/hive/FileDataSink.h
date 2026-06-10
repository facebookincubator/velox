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

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/PartitionIdGenerator.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Writer.h"
#include "velox/dwio/common/WriterFactory.h"
#include "velox/exec/MemoryReclaimer.h"

namespace facebook::velox::connector::hive {

/// Parameters for file writers.
class WriterParameters {
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
  WriterParameters(
      UpdateMode updateMode,
      std::optional<std::string> partitionName,
      std::string targetFileName,
      std::string targetDirectory,
      const std::optional<std::string>& writeFileName = std::nullopt,
      const std::optional<std::string>& writeDirectory = std::nullopt)
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
/// When file rotation occurs, multiple FileInfo entries are created.
struct FileInfo {
  /// The temporary file name used during writing (in the staging directory).
  std::string writeFileName;
  /// The final file name after commit (in the target directory).
  std::string targetFileName;
  /// Size of the file in bytes.
  uint64_t fileSize{0};
  /// Number of rows in the file.
  uint64_t numRows{0};
};

struct WriterInfo {
  WriterInfo(
      WriterParameters parameters,
      std::shared_ptr<memory::MemoryPool> _writerPool,
      std::shared_ptr<memory::MemoryPool> _sinkPool,
      std::shared_ptr<memory::MemoryPool> _sortPool)
      : writerParameters(std::move(parameters)),
        nonReclaimableSectionHolder(new tsan_atomic<bool>(false)),
        spillStats(std::make_unique<exec::SpillStats>()),
        writerPool(std::move(_writerPool)),
        sinkPool(std::move(_sinkPool)),
        sortPool(std::move(_sortPool)) {}

  // Writer configuration: update mode, partition, file paths.
  const WriterParameters writerParameters;
  // Guards non-reclaimable sections during write operations.
  const std::unique_ptr<tsan_atomic<bool>> nonReclaimableSectionHolder;
  // Collects the spill stats from sort writer if the spilling has been
  // triggered.
  const std::unique_ptr<exec::SpillStats> spillStats;
  // Memory pool for the writer itself.
  const std::shared_ptr<memory::MemoryPool> writerPool;
  // Memory pool for the file sink (serialization layer).
  const std::shared_ptr<memory::MemoryPool> sinkPool;
  // Memory pool for sort buffers (nullptr if not a sorted write).
  const std::shared_ptr<memory::MemoryPool> sortPool;
  // Total rows written by this writer across all files.
  uint64_t numWrittenRows{0};
  // Rows written to the current file; reset to 0 when the file is finalized.
  uint64_t currentFileWrittenRows{0};
  uint64_t inputSizeInBytes{0};
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
  std::vector<FileInfo> writtenFiles;
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

/// Identifies a writer by partition and bucket.
struct WriterId {
  std::optional<uint32_t> partitionId{std::nullopt};
  std::optional<uint32_t> bucketId{std::nullopt};

  WriterId() = default;

  explicit WriterId(
      std::optional<uint32_t> _partitionId,
      std::optional<uint32_t> _bucketId = std::nullopt)
      : partitionId(_partitionId), bucketId(_bucketId) {}

  /// Returns the special writer id for the un-partitioned (and non-bucketed)
  /// table.
  static const WriterId& unpartitionedId();

  std::string toString() const;

  bool operator==(const WriterId& other) const {
    return std::tie(partitionId, bucketId) ==
        std::tie(other.partitionId, other.bucketId);
  }
};

struct WriterIdHasher {
  std::size_t operator()(const WriterId& id) const {
    return bits::hashMix(
        id.partitionId.value_or(std::numeric_limits<uint32_t>::max()),
        id.bucketId.value_or(std::numeric_limits<uint32_t>::max()));
  }
};

struct WriterIdEq {
  bool operator()(const WriterId& lhs, const WriterId& rhs) const {
    return lhs == rhs;
  }
};

/// Base class for file-based data sinks that write data to columnar file
/// formats (ORC, Parquet, etc.). Provides the generic write pipeline: state
/// machine, row routing to writers, file rotation, and stats aggregation.
///
/// Connector-specific data sinks (Hive, Paimon, etc.) extend this class to
/// add format-specific behavior like commit protocols, partition naming,
/// writer creation, and memory reclamation.
class FileDataSink : public DataSink {
 public:
  /// Defines the execution states of a file data sink.
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

  FileDataSink(
      RowTypePtr inputType,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      dwio::common::FileFormat storageFormat,
      uint32_t maxOpenWriters,
      std::vector<column_index_t> partitionChannels,
      std::vector<column_index_t> dataChannels,
      int32_t bucketCount,
      std::unique_ptr<core::PartitionFunction> bucketFunction,
      std::unique_ptr<PartitionIdGenerator> partitionIdGenerator,
      std::shared_ptr<dwio::common::WriterFactory> writerFactory,
      uint64_t maxTargetFileBytes,
      bool partitionKeyAsLowerCase,
      const common::SpillConfig* spillConfig,
      uint64_t sortWriterFinishTimeSliceLimitMs);

  void appendData(RowVectorPtr input) override;

  bool finish() override;

  Stats stats() const override;

  std::unordered_map<std::string, RuntimeCounter> runtimeStats() const override;

  std::vector<std::string> close() override;

  void abort() override;

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
  virtual std::vector<std::string> commitMessage() const = 0;

  // Returns the type of non-partition data columns.
  static RowTypePtr getNonPartitionTypes(
      const std::vector<column_index_t>& dataCols,
      const RowTypePtr& inputType);

  // Filters out partition columns from the input, returning only data columns.
  static RowVectorPtr makeDataInput(
      const std::vector<column_index_t>& dataCols,
      const RowVectorPtr& input);

  FOLLY_ALWAYS_INLINE bool sortWrite() const {
    return sortWrite_;
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
      const WriterId& writerId);

  // Sets up memory reclaimers for writer pools. Override to install
  // format-specific reclaimers. Default is a no-op.
  virtual void setMemoryReclaimers(
      WriterInfo* writerInfo,
      io::IoStatistics* ioStats);

  // Returns the bytes written to the current file for the specified writer.
  uint64_t getCurrentFileBytes(size_t writerIndex) const;

  // Compute the partition id and bucket id for each row in 'input'.
  virtual void computePartitionAndBucketIds(const RowVectorPtr& input) = 0;

  // Get the HiveWriter corresponding to the row
  // from partitionIds and bucketIds.
  WriterId getWriterId(vector_size_t row) const;

  // Computes the number of input rows as well as the actual input row indices
  // to each corresponding (bucketed) partition based on the partition and
  // bucket ids calculated by 'computePartitionAndBucketIds'.
  void splitInputRowsAndEnsureWriters();

  // Makes sure to create one writer for the given writer id. The function
  // returns the corresponding index in 'writers_'.
  virtual uint32_t ensureWriter(const WriterId& id);

  // Appends a new writer for the given 'id'. The function returns the index of
  // the newly created writer in 'writers_'.
  uint32_t appendWriter(const WriterId& id);

  // Creates a writer for the given index using the current file sequence.
  virtual std::unique_ptr<facebook::velox::dwio::common::Writer>
  createWriterForIndex(size_t writerIndex) = 0;

  // Creates and configures WriterOptions based on file format.
  virtual std::shared_ptr<dwio::common::WriterOptions> createWriterOptions()
      const = 0;

  virtual std::shared_ptr<dwio::common::WriterOptions> createWriterOptions(
      size_t writerIndex) const = 0;

  // Returns the partition directory name for the given partition ID.
  virtual std::string getPartitionName(uint32_t partitionId) const = 0;

  // Returns writer parameters for the given partition and bucket.
  virtual WriterParameters getWriterParameters(
      const std::optional<std::string>& partition,
      std::optional<uint32_t> bucketId) const = 0;

  // Records a row index for a specific partition.
  void
  updatePartitionRows(uint32_t index, vector_size_t numRows, vector_size_t row);

  FOLLY_ALWAYS_INLINE void checkRunning() const {
    VELOX_CHECK_EQ(state_, State::kRunning, "File data sink is not running");
  }

  // Invoked to write 'input' to the specified file writer.
  void write(size_t index, RowVectorPtr input);

  // Rotates the writer at the given index to a new file.
  virtual void rotateWriter(size_t index);

  // Finalizes the current file for the writer at the given index.
  void finalizeWriterFile(size_t index);

  virtual void closeInternal();

  // IMPORTANT NOTE: these are passed to writers as raw pointers. FileDataSink
  // owns the lifetime of these objects, and therefore must destroy them last.
  std::vector<std::unique_ptr<io::IoStatistics>> ioStats_;
  // Generic filesystem stats, exposed as RuntimeStats
  std::unique_ptr<IoStats> fileSystemStats_;

  const RowTypePtr inputType_;
  const ConnectorQueryCtx* const connectorQueryCtx_;
  const CommitStrategy commitStrategy_;
  const dwio::common::FileFormat storageFormat_;
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

  /// Whether this sink uses sorted writes. Set by subclass after computing
  /// sort columns in its constructor.
  bool sortWrite_{false};

  State state_{State::kRunning};

  tsan_atomic<bool> nonReclaimableSection_{false};

  // The map from writer id to the writer index in 'writers_' and 'writerInfo_'.
  folly::F14FastMap<WriterId, uint32_t, WriterIdHasher, WriterIdEq>
      writerIndexMap_;

  // Below are structures for partitions from all inputs. writerInfo_ and
  // writers_ are both indexed by partitionId.
  std::vector<std::shared_ptr<WriterInfo>> writerInfo_;
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
};

FOLLY_ALWAYS_INLINE std::ostream& operator<<(
    std::ostream& os,
    FileDataSink::State state) {
  os << FileDataSink::stateString(state);
  return os;
}
} // namespace facebook::velox::connector::hive

template <>
struct fmt::formatter<facebook::velox::connector::hive::FileDataSink::State>
    : formatter<std::string> {
  auto format(
      facebook::velox::connector::hive::FileDataSink::State s,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::connector::hive::FileDataSink::stateString(s), ctx);
  }
};
