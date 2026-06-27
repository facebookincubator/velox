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

#include "velox/connectors/hive/iceberg/IcebergStatsCollector.h"

#include "velox/connectors/hive/iceberg/IcebergDwrfStatsCollector.h"
#ifdef VELOX_ENABLE_PARQUET
#include "velox/connectors/hive/iceberg/IcebergParquetStatsCollector.h"
#endif

namespace facebook::velox::connector::hive::iceberg {

std::unique_ptr<IcebergStatsCollector> IcebergStatsCollector::create(
    dwio::common::FileFormat format,
    const std::vector<IcebergColumnHandlePtr>& inputColumns,
    const RowTypePtr& schema) {
  switch (format) {
    case dwio::common::FileFormat::DWRF:
    case dwio::common::FileFormat::ORC:
      // DWRF/ORC statistics are read from the writer footer; the collector maps
      // footer node ids to Iceberg field ids using the written row type.
      return std::make_unique<IcebergDwrfStatsCollector>(inputColumns, schema);
#ifdef VELOX_ENABLE_PARQUET
    case dwio::common::FileFormat::PARQUET:
      return std::make_unique<IcebergParquetStatsCollector>(inputColumns);
#endif
    default:
      return nullptr;
  }
}

} // namespace facebook::velox::connector::hive::iceberg
