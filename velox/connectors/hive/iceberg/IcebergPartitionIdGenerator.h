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

#include "velox/connectors/hive/PartitionIdGenerator.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"
#include "velox/connectors/hive/iceberg/TransformEvaluator.h"

namespace facebook::velox::connector::hive::iceberg {

/// Generates partition IDs for writing Iceberg table.
///
/// Computes partition keys by applying Iceberg partition transforms
/// (e.g., bucket, truncate, year, month, day, hour, identity) to source columns
/// and maps the transformed values to sequential partition IDs. Maintains a
/// mapping from partition values to partition IDs and generates
/// Iceberg-compliant partition path names for file organization.
///
/// Each unique combination of transformed partition values is assigned
/// a sequential ID (0, 1, 2, ...) up to a configurable maximum.
/// Uses TransformEvaluator for batch evaluation of transforms.
class IcebergPartitionIdGenerator : public PartitionIdGenerator {
 public:
  /// @param inputType RowType of the input data. Used to build transform
  /// expressions once during construction for reuse across multiple input
  /// batches.
  /// @param partitionChannels Column indices (0-based) in the input RowVector
  /// for source columns referenced by each partition field. Must have the same
  /// size as partitionSpec->fields. partitionChannels[i] specifies which input
  /// column to use for partitionSpec->fields[i]. The same channel index may
  /// appear multiple times if multiple transforms are applied to the same
  /// source column (e.g., both bucket(c1, 16) and truncate(c1, 10)).
  /// This separate mapping is necessary because partitionSpec field names refer
  /// to table schema column names, while the input RowVector may use different
  /// column names (e.g., generated names like c0, c1). Cannot use column names
  /// in partitionSpec to locate child vector in input RowVector.
  /// The partitionChannels provide the positional mapping from partition spec
  /// fields to input RowVector columns.
  /// @param partitionSpec Iceberg partition specification containing transform
  /// definitions for each partition field. Each field specifies a source column
  /// name (matching table schema, not input RowVector names), transform type
  /// (identity, bucket, truncate, year, etc.), and optional parameters.
  /// Multiple transforms on the same column are allowed with restrictions
  /// enforced by IcebergPartitionSpec validation.
  /// See IcebergPartitionSpec::checkCompatibility for detail.
  /// @param maxPartitions Maximum number of distinct partitions allowed.
  /// Exceeding this limit will cause a VeloxUserError.
  /// @param connectorQueryCtx Connector query context for expression evaluation
  /// and memory allocation.
  IcebergPartitionIdGenerator(
      const RowTypePtr& inputType,
      std::vector<column_index_t> partitionChannels,
      IcebergPartitionSpecPtr partitionSpec,
      uint32_t maxPartitions,
      const ConnectorQueryCtx* connectorQueryCtx);

  /// Generate sequential partition IDs for all rows in the input vector.
  ///
  /// For each row, applies the partition transforms, hashes the transformed
  /// values, and maps to a sequential partition ID. New partitions are created
  /// as needed (up to maxPartitions limit). Partition values are stored for
  /// later partition name generation.
  ///
  /// @param input Input RowVector containing all columns. Must have columns
  /// at indices specified by partitionChannels.
  /// @param result Output vector of partition IDs, one per input row.
  /// Resized to match input size. Values are 0-based sequential IDs.
  void run(const RowVectorPtr& input, raw_vector<uint64_t>& result) override;

  /// Generate the partition name for the given partition ID.
  ///
  /// Constructs an Iceberg-compliant partition path string in the format:
  /// "key1=value1/key2=value2/..." where keys are partition column names for
  /// identity transform or column_transform for non-identity transforms and
  /// values are the human-readable string representations of the transformed
  /// partition values. Special characters are URL-encoded per
  /// java.net.URLEncoder.encode().
  ///
  /// In typical usage (e.g., HiveDataSink::appendWriter), this is called
  /// once per partition ID when creating a new writer for that partition.
  /// It performs string formatting and URL encoding on each call, so callers
  /// should cache the result if the same partition name is needed
  /// multiple times.
  ///
  /// @param partitionId Sequential partition ID (0-based) returned by run().
  /// Must be less than the number of partitions created so far.
  /// @return Partition path string suitable for use in file paths.
  std::string partitionName(uint32_t partitionId) const override;

  /// Returns the RowVector containing transformed partition values.
  /// Each row in this vector corresponds to a partition ID (row index =
  /// partition ID). Each column contains the transformed values for that
  /// partition column.
  /// The vector grows dynamically as new partitions are discovered, up to
  /// maxPartitions_.
  /// Should be called after calling run() method.
  ///
  /// @return RowVector with one column per transformed column, columns in same
  /// order as IcebergPartitionSpec::fields.
  const RowVectorPtr& partitionKeys() const {
    return partitionValues_;
  }

 private:
  void savePartitionValues(
      uint32_t partitionId,
      const RowVectorPtr& input,
      vector_size_t row) override;

  std::vector<std::pair<std::string, std::string>> extractPartitionKeyValues(
      const RowVectorPtr& partitionsVector,
      vector_size_t row) const;

  const IcebergPartitionSpecPtr partitionSpec_;
  const ConnectorQueryCtx* connectorQueryCtx_;
  const std::unique_ptr<TransformEvaluator> transformEvaluator_;
  RowTypePtr rowType_;
};

} // namespace facebook::velox::connector::hive::iceberg
