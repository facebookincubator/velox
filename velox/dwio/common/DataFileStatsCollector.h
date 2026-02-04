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

#include "velox/dwio/common/DataFileStatistics.h"

namespace facebook::velox::dwio::common {

class DataFileStatsCollectorBase {
 public:
  DataFileStatsCollectorBase() = default;
  virtual ~DataFileStatsCollectorBase() = default;
};

/// Base class for collecting file statistics from format-specific metadata.
/// @tparam MetadataType The format-specific metadata type (e.g.,
/// std::shared_ptr<parquet::arrow::FileMetaData> for Parquet).
template <typename MetadataType>
class DataFileStatsCollector : public DataFileStatsCollectorBase {
 public:
  DataFileStatsCollector() = default;

  ~DataFileStatsCollector() override = default;

  /// Aggregates format-specific file metadata into DataFileStatistics.
  /// @param metadata The format-specific metadata to aggregate.
  /// @return DataFileStatistics populated with aggregated stats.
  virtual std::shared_ptr<IcebergDataFileStatistics> aggregate(
      const MetadataType& metadata) = 0;
};

} // namespace facebook::velox::dwio::common
