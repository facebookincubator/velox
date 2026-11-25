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
#include "velox/connectors/hive/HiveBucketProperty.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveInsertTableHandle.h"
#include "velox/connectors/hive/HivePartitionName.h"
#include "velox/connectors/hive/PartitionIdGenerator.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Writer.h"
#include "velox/dwio/common/WriterFactory.h"
#include "velox/exec/MemoryReclaimer.h"

namespace facebook::velox::connector::hive {

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
  // Written-column channels of the unique key; empty keeps every row.
  std::vector<column_index_t> dedupeKeyChannels_;

  // Strategy for naming writer files
  FileNameGeneratorPtr fileNameGenerator_;
};

} // namespace facebook::velox::connector::hive
