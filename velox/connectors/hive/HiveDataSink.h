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
#include "velox/connectors/hive/HiveCommitMessage.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveWriterTypes.h"
#include "velox/connectors/hive/PartitionIdGenerator.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/WriterFactory.h"
#include "velox/exec/MemoryReclaimer.h"

namespace facebook::velox::connector::hive {

class FileNameGenerator;
class HiveInsertTableHandle;
class PartitionWriterInterface;

class BucketSortingWriter;
class PartitionWriter;

class HiveDataSink : public DataSink {
 public:
  ~HiveDataSink() override;
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

  // Compute the partition id and bucket id for each row in 'input'.
  virtual void computePartitionAndBucketIds(const RowVectorPtr& input);

  // Creates a partition writer for the given writer ID and index. Called by
  // the PartitionWriter's WriterFactory to create new writers on demand.
  std::unique_ptr<PartitionWriterInterface> createRotationWriter(
      const HiveWriterId& id,
      uint32_t writerIndex);

  // Creates a format writer (optionally wrapped in SortingWriter) for the
  // given writer info and IO stats. Updates file names based on the current
  // file sequence number.
  std::unique_ptr<facebook::velox::dwio::common::Writer> createFormatWriter(
      HiveWriterInfo* writerInfo,
      io::IoStatistics* ioStats);

  // Creates and configures WriterOptions for a specific writer.
  virtual std::shared_ptr<dwio::common::WriterOptions> createWriterOptions(
      const HiveWriterInfo* writerInfo) const;

  // Returns the Hive partition directory name for the given partition ID.
  // Converts the partition values associated with the partition ID into a
  // Hive-formatted directory path. Returns std::nullopt if the table is
  // unpartitioned. Should be called only when writing to a partitioned table.
  virtual std::string getPartitionName(uint32_t partitionId) const;

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

  virtual void closeInternal();

  // Generic filesystem stats, exposed as RuntimeStats.
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
  const uint64_t maxTargetFileBytes_{0};
  const bool partitionKeyAsLowerCase_;

  State state_{State::kRunning};

  // Wraps format writers with SortingWriter when bucket sorting is enabled.
  std::unique_ptr<BucketSortingWriter> bucketSortingWriter_;

  // Routes rows to writers by partition/bucket and owns partition writers.
  std::unique_ptr<PartitionWriter> partitionWriter_;

  // Below are structures updated when processing current input. partitionIds_
  // are indexed by the row of input_.
  raw_vector<uint64_t> partitionIds_;

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
