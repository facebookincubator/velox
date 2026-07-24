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

#include "velox/connectors/hive/FileDataSink.h"
#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/paimon/PaimonConfig.h"

namespace facebook::velox::connector::hive::paimon {

/// JSON field names for the commit message produced by PaimonDataSink.
///
/// Each writer (one per partition × bucket) produces a JSON object consumed by
/// the Paimon coordinator to create manifest entries and a new snapshot.
///
/// JSON structure:
/// {
///   "partitionValues":           {"dt": "2024-01-01", "hour": "10"},
///   "bucketNumber":              0,
///   "writePath":                 "<staging directory>",
///   "targetPath":                "<final directory>",
///   "fileWriteInfos": [
///     {
///       "writeFileName":         "<temp filename in writePath>",
///       "targetFileName":        "<final filename in targetPath>",
///       "fileSize":              <bytes>,
///       "rowCount":              <rows in this file>
///     }
///   ],
///   "totalRowCount":             <total rows across all files>,
///   "inMemoryDataSizeInBytes":   <uncompressed bytes>,
///   "onDiskDataSizeInBytes":     <compressed bytes on disk>
/// }
struct CommitMessage {
  /// Partition key-value pairs. Empty object for unpartitioned tables.
  static constexpr const char* kPartitionValues = "partitionValues";
  /// Bucket number within the partition. -1 for unbucketed tables.
  static constexpr const char* kBucketNumber = "bucketNumber";
  /// Staging directory where files were written.
  static constexpr const char* kWritePath = "writePath";
  /// Final directory after commit.
  static constexpr const char* kTargetPath = "targetPath";
  /// Per-file metadata array.
  static constexpr const char* kFileWriteInfos = "fileWriteInfos";
  /// Temporary filename in the staging directory.
  static constexpr const char* kWriteFileName = "writeFileName";
  /// Final filename after commit.
  static constexpr const char* kTargetFileName = "targetFileName";
  /// Size of individual file in bytes.
  static constexpr const char* kFileSize = "fileSize";
  /// Number of rows in individual file.
  static constexpr const char* kFileRowCount = "rowCount";
  /// Total rows written across all files for this partition/bucket.
  static constexpr const char* kTotalRowCount = "totalRowCount";
  /// Uncompressed input data size in bytes.
  static constexpr const char* kInMemoryDataSizeInBytes =
      "inMemoryDataSizeInBytes";
  /// Compressed bytes written to disk.
  static constexpr const char* kOnDiskDataSizeInBytes = "onDiskDataSizeInBytes";
};

/// Paimon-specific data sink for writing data files to Paimon tables.
///
/// Extends FileDataSink with Paimon's directory layout (bucket-N/ directories),
/// file naming (data-{uuid}.{format}), and commit message format.
///
/// Supports append-only writes for unpartitioned, partitioned, and bucketed
/// tables. Primary-key table writes, sorted writes, and memory reclamation
/// are not yet supported.
///
/// Directory layout:
///   {tableRoot}/{partition=value}/bucket-{N}/data-{uuid}.orc
///
/// All Paimon tables have a bucket directory — unbucketed tables use bucket-0.
class PaimonDataSink : public FileDataSink {
 public:
  PaimonDataSink(
      RowTypePtr inputType,
      std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const PaimonConfig>& paimonConfig);

 protected:
  /// Generates Paimon-specific commit messages for the coordinator.
  std::vector<std::string> commitMessage() const override;

  /// Computes partition and bucket IDs for each row.
  void computePartitionAndBucketIds(const RowVectorPtr& input) override;

  /// Creates a file writer for the given writer index.
  std::unique_ptr<facebook::velox::dwio::common::Writer> createWriterForIndex(
      size_t writerIndex) override;

  /// Creates writer options with Paimon-specific settings.
  std::shared_ptr<dwio::common::WriterOptions> createWriterOptions()
      const override;

  std::shared_ptr<dwio::common::WriterOptions> createWriterOptions(
      size_t writerIndex) const override;

  /// Returns Hive-style partition directory name (key=value).
  std::string getPartitionName(uint32_t partitionId) const override;

  /// Returns writer parameters with Paimon directory layout (bucket-N/).
  WriterParameters getWriterParameters(
      const std::optional<std::string>& partition,
      std::optional<uint32_t> bucketId) const override;

 private:
  const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle_;
  const std::shared_ptr<const PaimonConfig> paimonConfig_;
};

} // namespace facebook::velox::connector::hive::paimon
