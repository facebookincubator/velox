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
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergDataFileStatistics.h"
#include "velox/dwio/common/FileMetadata.h"
#include "velox/dwio/parquet/ParquetFieldId.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

class IcebergParquetStatsCollector {
 public:
  explicit IcebergParquetStatsCollector(
      const std::vector<IcebergColumnHandlePtr>& inputColumns);

  /// Returns the Parquet field IDs for all input columns.
  /// The field IDs are written to the Parquet data file's column metadata.
  /// The return object describes a multi-column input.
  const parquet::ParquetFieldId& parquetFieldIds() const {
    return parquetFieldIds_;
  }

  /// Aggregates Parquet file metadata into Iceberg data file statistics.
  /// Iterates through all row groups and columns to collect:
  /// - Record count, split offsets, value counts, column sizes, null counts.
  /// - Min/max bounds (base64-encoded). Currently not collected for MAP and
  ///   ARRAY types and all their descendants.
  /// @param fileMetadata The Parquet file metadata to aggregate.
  IcebergDataFileStatisticsPtr aggregate(
      std::unique_ptr<dwio::common::FileMetadata> fileMetadata);

  /// TODO: Need to support this config property.
  /// 16 is default value. See DEFAULT_WRITE_METRICS_MODE_DEFAULT in
  /// org.apache.iceberg.TableProperties.
  constexpr static int32_t kDefaultTruncateLength{16};

 private:
  bool shouldStoreBounds(int32_t fieldId) const {
    return !skipBoundsFieldIds_.contains(fieldId);
  }

  // Hierarchical Parquet field IDs for all input columns. A single
  // ParquetFieldId can describe all the columns including their nested
  // children.
  parquet::ParquetFieldId parquetFieldIds_;

  // Set of field IDs for which bounds collection should be skipped.
  // This includes MAP and ARRAY types and all their descendants.
  std::unordered_set<int32_t> skipBoundsFieldIds_;
};

} // namespace facebook::velox::connector::hive::iceberg
