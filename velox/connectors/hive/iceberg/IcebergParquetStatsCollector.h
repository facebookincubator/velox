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

#include <unordered_set>

#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/DataFileStatsCollector.h"
#include "velox/dwio/parquet/ParquetFieldId.h"
#include "velox/type/Type.h"

namespace facebook::velox::parquet::arrow {

class FileMetaData;

} // namespace facebook::velox::parquet::arrow

namespace facebook::velox::connector::hive::iceberg {

using ParquetFileMetadata = std::shared_ptr<parquet::arrow::FileMetaData>;

class IcebergParquetStatsCollector
    : public dwio::common::DataFileStatsCollector<ParquetFileMetadata> {
 public:
  explicit IcebergParquetStatsCollector(
      const std::vector<std::shared_ptr<const HiveColumnHandle>>& inputColumns);

  /// Returns the Parquet field IDs for all input columns.
  /// The field IDs are written to the Parquet data file's column metadata.
  const std::vector<parquet::ParquetFieldId>& parquetFieldIds() const {
    return parquetFieldIds_;
  }

  /// Aggregates Parquet file metadata into Iceberg data file statistics.
  /// Iterates through all row groups and columns to collect:
  /// - Record count, split offsets, value counts, column sizes, null counts.
  /// - Min/max bounds (base64-encoded) for columns not in skipBounds set.
  /// @param metadata The Parquet file metadata to aggregate.
  /// @return DataFileStatistics populated with aggregated metrics.
  std::shared_ptr<dwio::common::IcebergDataFileStatistics> aggregate(
      const ParquetFileMetadata& metadata) override;

  // TODO: Need to support this config property.
  // 16 is default value. See DEFAULT_WRITE_METRICS_MODE_DEFAULT in
  // org.apache.iceberg.TableProperties.
  constexpr static int32_t kDefaultTruncateLength{16};

 private:
  bool shouldStoreBounds(int32_t fieldId) const {
    return !skipBoundsFieldIds_.contains(fieldId);
  }

  /// Parquet field IDs for all input columns.
  std::vector<parquet::ParquetFieldId> parquetFieldIds_;

  /// Set of field IDs for which bounds collection should be skipped.
  /// This includes MAP and ARRAY types and all their descendants.
  std::unordered_set<int32_t> skipBoundsFieldIds_;
};

} // namespace facebook::velox::connector::hive::iceberg
