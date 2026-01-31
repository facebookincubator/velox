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

#include "velox/connectors/hive/iceberg/IcebergParquetStatsConfig.h"
#include "velox/dwio/common/DataFileStatsCollector.h"

namespace facebook::velox::connector::hive::iceberg {

class IcebergParquetStatsCollector
    : public dwio::common::DataFileStatsCollector {
 public:
  explicit IcebergParquetStatsCollector(
      std::vector<IcebergParquetStatsConfig> configs);

  /// Aggregates Parquet file metadata into Iceberg data file statistics.
  /// Iterates through all row groups and columns to collect:
  /// - Record count, split offsets, value counts, column sizes, null counts.
  /// - Min/max bounds (base64-encoded) for columns not in skipBounds set.
  /// @param metadata Pointer to shared_ptr<parquet::arrow::FileMetaData>.
  /// @return DataFileStatistics populated with aggregated metrics.
  std::shared_ptr<dwio::common::DataFileStatistics> aggregate(
      const void* metadata) override;

  // TODO: Need to support this config property.
  // 16 is default value. See DEFAULT_WRITE_METRICS_MODE_DEFAULT in
  // org.apache.iceberg.TableProperties.
  constexpr static int32_t kDefaultTruncateLength = 16;

 private:
  const std::vector<IcebergParquetStatsConfig> icebergParquetStatsConfig_;
};

} // namespace facebook::velox::connector::hive::iceberg
