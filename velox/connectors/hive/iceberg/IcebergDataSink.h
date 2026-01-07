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

#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/iceberg/DataFileStatsCollector.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/TransformFactory.h"
#include "velox/connectors/hive/iceberg/Transforms.h"

namespace facebook::velox::connector::hive::iceberg {

class IcebergSortingColumn : public ISerializable {
 public:
  IcebergSortingColumn(
      const std::string& sortColumn,
      const core::SortOrder& sortOrder);

  const std::string& sortColumn() const;

  const core::SortOrder& sortOrder() const;

  folly::dynamic serialize() const override;

 private:
  const std::string sortColumn_;
  const core::SortOrder sortOrder_;
};

class IcebergFileNameGenerator : public FileNameGenerator {
public:
  IcebergFileNameGenerator() {}

  std::pair<std::string, std::string> gen(
      std::optional<uint32_t> bucketId,
      const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx& connectorQueryCtx,
      bool commitRequired) const override;

  folly::dynamic serialize() const override;

  std::string toString() const override;
};

// Represents a request for Iceberg write.
class IcebergInsertTableHandle final : public HiveInsertTableHandle {
 public:
  /// @param inputColumns Columns from the table schema to write.
  /// The input RowVector must have the same number of columns and matching
  /// types in the same order.
  /// Column names in the RowVector may differ from those in inputColumns,
  /// only position and type must align. All columns present in the input
  /// data must be included, mismatches can lead to write failure.
  /// @param locationHandle Contains the target location information including:
  /// - Base directory path where data files will be written.
  /// - File naming scheme and temporary directory paths.
  /// @param tableStorageFormat File format to use for writing data files.
  /// @param partitionSpec Optional partition specification defining how to
  /// partition the data. If nullptr, the table is unpartitioned and all data
  /// is written to a single directory.
  /// @param compressionKind Optional compression to apply to data files.
  /// @param serdeParameters Additional serialization/deserialization parameters
  /// for the file format.
  /// @param fileNameGenerator File name generator for generating unique file
  /// names for data files. If nullptr, will use IcebergFileNameGenerator.
  IcebergInsertTableHandle(
      std::vector<IcebergColumnHandlePtr> inputColumns,
      LocationHandlePtr locationHandle,
      std::shared_ptr<const IcebergPartitionSpec> partitionSpec,
      memory::MemoryPool* pool,
      dwio::common::FileFormat tableStorageFormat =
          dwio::common::FileFormat::PARQUET,
      const std::vector<IcebergSortingColumn>& sortedBy = {},
      std::optional<common::CompressionKind> compressionKind = {},
      const std::unordered_map<std::string, std::string>& serdeParameters = {},
      std::shared_ptr<const FileNameGenerator> fileNameGenerator =
          std::make_shared<const IcebergFileNameGenerator>());

  std::shared_ptr<const IcebergPartitionSpec> partitionSpec() const {
    return partitionSpec_;
  }

  const std::vector<std::shared_ptr<Transform>>& columnTransforms() const {
    return columnTransforms_;
  }

  const std::vector<IcebergSortingColumn>& sortedBy() const {
    return sortedBy_;
  }

 private:
  const std::shared_ptr<const IcebergPartitionSpec> partitionSpec_;
  const std::vector<std::shared_ptr<Transform>> columnTransforms_;
  const std::vector<IcebergSortingColumn> sortedBy_;
};

using IcebergInsertTableHandlePtr =
    std::shared_ptr<const IcebergInsertTableHandle>;

class IcebergDataSink : public HiveDataSink {
 public:
  IcebergDataSink(
      RowTypePtr inputType,
      IcebergInsertTableHandlePtr insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const HiveConfig>& hiveConfig);

  void appendData(RowVectorPtr input) override;

  const std::vector<std::shared_ptr<dwio::common::DataFileStatistics>>&
  dataFileStats() const {
    return dataFileStats_;
  }
  /// Generates Iceberg-specific commit messages for all writers containing
  /// metadata about written files. Creates a JSON object for each writer
  /// in the format expected by Presto and Spark for Iceberg tables.
  ///
  /// Each commit message contains:
  /// - path: full file path where data was written.
  /// - fileSizeInBytes: raw bytes written to disk.
  /// - metrics: object with recordCount (number of rows written).
  /// - partitionSpecJson: partition specification.
  /// - fileFormat: storage format (e.g., "PARQUET").
  /// - content: file content type ("DATA" for data files).
  ///
  /// See
  /// https://github.com/prestodb/presto/blob/master/presto-iceberg/src/main/java/com/facebook/presto/iceberg/CommitTaskData.java
  ///
  /// Note: Complete Iceberg metrics are not yet implemented, which results in
  /// incomplete manifest files that may lead to suboptimal query planning.
  ///
  /// @return Vector of JSON strings, one per writer, formatted according to
  /// Presto and Spark Iceberg commit protocol.
  std::vector<std::string> commitMessage() const override;

  bool finish() override;

 private:
  IcebergDataSink(
      RowTypePtr inputType,
      IcebergInsertTableHandlePtr insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      const std::vector<column_index_t>& partitionChannels,
      const std::vector<column_index_t>& dataChannels);

  void splitInputRowsAndEnsureWriters(RowVectorPtr input) override;

  void computePartition(const RowVectorPtr& input);

  HiveWriterId getIcebergWriterId(size_t row) const;

  // Creates writer options configured for Iceberg table writes. Extends the
  // base HiveDataSink writer options with Iceberg-specific settings:
  // - Sets timestamp timezone to nullopt (UTC) for Iceberg compliance.
  // - Sets timestamp precision to microseconds.
  std::shared_ptr<dwio::common::WriterOptions> createWriterOptions(
      size_t writerIndex) const override;
  // Creates writer options configured for Iceberg table writes. Extends the
  // base HiveDataSink writer options with Iceberg-specific settings:
  // - Sets timestamp timezone to nullopt (UTC) for Iceberg compliance.
  // - Sets timestamp precision to microseconds.
  std::shared_ptr<dwio::common::WriterOptions> createWriterOptions()
      const override;

  void rotateWriter(size_t index) override;

  void closeInternal() override;

  void closeWriter(int32_t index);

  bool finishWriter(int32_t index);

  std::optional<std::string> getPartitionName(
      const HiveWriterId& id) const override;

  std::unique_ptr<dwio::common::Writer> maybeCreateBucketSortWriter(
      std::unique_ptr<dwio::common::Writer> writer);

  void buildPartitionData(int32_t index);

  void clusteredWrite(RowVectorPtr input, int32_t writerIdx);

  const IcebergInsertTableHandlePtr icebergInsertTableHandle_;

  // TODO: Add IcebergParquetStatsCollector back later
  // #ifdef VELOX_ENABLE_PARQUET
  //   std::shared_ptr<IcebergParquetStatsCollector> parquetStatsCollector_;
  // #endif


  // Below are structures for partitions from all inputs. partitionData_
  // is indexed by partitionId.
  std::vector<std::vector<folly::dynamic>> partitionData_;

  std::vector<std::shared_ptr<dwio::common::DataFileStatistics>> dataFileStats_;
  std::shared_ptr<
      std::vector<std::unique_ptr<dwio::common::DataFileStatsSettings>>>
      statsSettings_;
  std::unique_ptr<DataFileStatsCollector> icebergStatsCollector_;

  // Below are structures for clustered mode writer.
  const bool fanoutEnabled_;
  uint32_t currentWriterId_;
  std::unordered_set<uint32_t> completedWriterIds_;
};

} // namespace facebook::velox::connector::hive::iceberg
