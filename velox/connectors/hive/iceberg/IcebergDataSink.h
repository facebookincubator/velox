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
#include <string>
#include <unordered_map>
#include <vector>

#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergDataFileStatistics.h"

#ifdef VELOX_ENABLE_PARQUET
#include "velox/connectors/hive/iceberg/IcebergParquetStatsCollector.h"
#endif

#include "velox/connectors/hive/iceberg/IcebergConfig.h"
#include "velox/connectors/hive/iceberg/IcebergPartitionName.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"
#include "velox/connectors/hive/iceberg/TransformEvaluator.h"
#include "velox/functions/iceberg/Register.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

IcebergColumnHandlePtr convertToIcebergColumnHandle(
    const HiveColumnHandlePtr& hiveColumn) {
  static int32_t fieldIdCounter = 1;

  std::function<parquet::ParquetFieldId(const TypePtr&, int32_t&)> makeField =
      [&makeField](
          const TypePtr& type, int32_t& fieldId) -> parquet::ParquetFieldId {
    const int32_t currentId = fieldId++;
    std::vector<parquet::ParquetFieldId> children;
    children.reserve(type->size());
    for (auto i = 0; i < type->size(); ++i) {
      children.push_back(makeField(type->childAt(i), fieldId));
    }
    return parquet::ParquetFieldId{currentId, children};
  };

  auto field = makeField(hiveColumn->dataType(), fieldIdCounter);

  return std::make_shared<const IcebergColumnHandle>(
      hiveColumn->name(),
      hiveColumn->columnType(),
      hiveColumn->dataType(),
      field);
}

} // namespace

/// Represents a request for Iceberg write.
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
  IcebergInsertTableHandle(
      std::vector<IcebergColumnHandlePtr> inputColumns,
      LocationHandlePtr locationHandle,
      dwio::common::FileFormat tableStorageFormat,
      IcebergPartitionSpecPtr partitionSpec,
      std::optional<common::CompressionKind> compressionKind = {},
      const std::unordered_map<std::string, std::string>& serdeParameters = {});

  /// Returns the Iceberg partition specification that defines how the table
  /// is partitioned.
  const IcebergPartitionSpecPtr& partitionSpec() const {
    return partitionSpec_;
  }

 private:
  const IcebergPartitionSpecPtr partitionSpec_;
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
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      const IcebergConfigPtr& icebergConfig);

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

 private:
  IcebergDataSink(
      RowTypePtr inputType,
      IcebergInsertTableHandlePtr insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      const std::vector<column_index_t>& partitionChannels,
      const std::vector<column_index_t>& dataChannels,
      RowTypePtr partitionRowType,
      const IcebergConfigPtr& icebergConfig);

  // Computes partition IDs for each row in the input batch by applying Iceberg
  // partition transforms and generating unique partition identifiers.
  //
  // Performs a two-step process:
  // 1. Applies Iceberg partition transforms (e.g., year, month, day, hour,
  //    bucket, truncate) to the input partition columns using
  //    transformEvaluator_ to produce transformed partition values.
  // 2. Wraps the transformed columns in a RowVector with partitionRowType_
  //    schema and passes it to partitionIdGenerator_ to compute partition IDs.
  //
  // The resulting partition IDs are stored in partitionIds_ buffer, where each
  // element corresponds to a row in the input. These IDs are used to:
  // - Route rows to the appropriate writer (one writer per unique partition).
  // - Generate partition directory names via getPartitionName().
  //
  // Note: Iceberg does not support bucketing, so this method only computes
  // partition IDs, not bucket IDs.
  //
  // @param input The input RowVector containing rows to be partitioned.
  void computePartitionAndBucketIds(const RowVectorPtr& input) override;

  // Returns the Iceberg partition directory name for the given partition ID.
  // Converts the transformed partition values associated with the partition ID
  // into an Iceberg compliant directory path
  // (e.g., "date_year=2023/id_bucket=5").
  std::string getPartitionName(uint32_t partitionId) const override;

  // Ensures a writer exists for the given writer ID and returns its index.
  // If the writer doesn't exist, creates it by calling appendWriter().
  // Additionally, extracts and stores the transformed partition values for
  // the writer in commitPartitionValue_ if not already set, which will be
  // included in the commit message as "partitionDataJson".
  uint32_t ensureWriter(const HiveWriterId& id) override;

  // Creates writer options configured for Iceberg table writes. Extends the
  // base HiveDataSink writer options with Iceberg-specific settings:
  // - Sets timestamp timezone to nullopt (UTC) for Iceberg compliance.
  // - Sets timestamp precision to microseconds.
  std::shared_ptr<dwio::common::WriterOptions> createWriterOptions()
      const override;

  // Extracts partition values for a specific writer to be included in the
  // commit message. Converts the transformed partition values from columnar
  // storage (partitionIdGenerator_->partitionValues() where each partition
  // field is a separate column) to row storage (a folly::dynamic array of
  // values for the given writer index) for JSON serialization.
  // Returns nullptr for null partition values.
  folly::dynamic makeCommitPartitionValue(uint32_t writerIndex) const;

  void closeInternal() override;

  // Iceberg partition specification defining how the table is partitioned.
  // Contains partition fields with source column names, transform types
  // (e.g., identity, year, month, day, hour, bucket, truncate), transform
  // parameters, and result types. Null if the table is unpartitioned.
  const IcebergPartitionSpecPtr partitionSpec_;

  // Evaluates Iceberg partition transforms on input rows to produce transformed
  // partition keys. Applies transforms defined in partitionSpec_ (e.g.,
  // year(date_col), bucket(id, 16)) to the corresponding input columns and
  // returns a vector of transformed columns. The transformed keys are then
  // wrapped in a RowVector and passed to IcebergPartitionIdGenerator.
  // Null if the table is unpartitioned.
  const std::unique_ptr<TransformEvaluator> transformEvaluator_;

  // Generates Iceberg compliant partition directory names from partition IDs.
  // Converts transformed partition values to human-readable strings based on
  // their transform types (e.g., year -> "2025", month -> "2025-11", hour ->
  // "2025-11-12-13") and constructs URL-encoded partition paths.
  // Null if the table is unpartitioned.
  const std::unique_ptr<IcebergPartitionName> icebergPartitionName_;

  // RowType schema for the transformed partition values RowVector.
  // Contains one column per partition field in partitionSpec, where each
  // column has:
  // - Type: The result type of the partition transform (e.g., INTEGER for year
  //   transform, DATE for day transform).
  // - Name: Source column name for identity transforms, or
  //   "columnName_transformName" for non-identity transforms (e.g.,
  //   "date_year").
  // Used to construct the RowVector that wraps the transformed partition
  // columns before passing them to IcebergPartitionIdGenerator for partition ID
  // generation and to IcebergPartitionNameGenerator for partition path name
  // generation.
  RowTypePtr partitionRowType_;

  // Stores the transformed partition values for each writer to be included in
  // the commit message sent to Presto. Indexed by writer index. Each entry
  // contains the transformed partition values (as a folly::dynamic array) for
  // that writer's partition, which are serialized to JSON as
  // "partitionDataJson" in the commit protocol. These values represent the same
  // transformed partition data as partitionIdGenerator_->partitionValues(), but
  // converted from columnar storage (where each partition field is a separate
  // column in the RowVector) to row storage (where each writer has a
  // folly::dynamic array of values across all partition fields), ready for JSON
  // serialization.
  std::vector<folly::dynamic> commitPartitionValue_;

  // Statistics for all data files written by this sink. These statistics
  // are populated during closeInternal(). These metrics are subsequently used
  // to construct Iceberg commit messages.
  // Multiple entries are saved because a single write session from
  // appendData can generate multiple data files if the input row batch spans
  // multiple partitions or if file rotation is triggered (e.g., reaching file
  // size limits). Each entry corresponds to one individual data file.
  std::vector<IcebergDataFileStatisticsPtr> dataFileStats_;

  const IcebergInsertTableHandlePtr icebergInsertTableHandle_;

#ifdef VELOX_ENABLE_PARQUET
  std::shared_ptr<IcebergParquetStatsCollector> parquetStatsCollector_;
#endif
};

} // namespace facebook::velox::connector::hive::iceberg
