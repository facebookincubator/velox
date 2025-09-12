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

#include "velox/dwio/common/DataFileStatsCollector.h"

namespace facebook::velox::connector::hive::iceberg {

/// Settings for collecting Iceberg parquet data file statistics.
/// Holds the Iceberg source field id and whether to skip bounds
/// collection for this field. For nested field, it contains child fields.
struct IcebergDataFileStatsSettings
    : public dwio::common::DataFileStatsSettings {
  int32_t fieldId;
  bool skipBounds;
  std::vector<std::unique_ptr<IcebergDataFileStatsSettings>> children;

  IcebergDataFileStatsSettings(int32_t id, bool skip)
      : fieldId(id), skipBounds(skip), children() {}
};

class DataFileStatsCollector : public dwio::common::FileStatsCollector {
 public:
  explicit DataFileStatsCollector(
      std::shared_ptr<
          std::vector<std::unique_ptr<dwio::common::DataFileStatsSettings>>>
          settings);

  void collectStats(
      const void* metadata,
      const std::shared_ptr<dwio::common::DataFileStatistics>& fileStats)
      override;
};

} // namespace facebook::velox::connector::hive::iceberg
