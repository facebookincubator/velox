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
#include <vector>

#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergDataFileStatistics.h"
#include "velox/dwio/common/FileMetadata.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Writer.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

/// Collects per-file Iceberg column statistics (column sizes, value counts,
/// null counts, lower/upper bounds) from a closed file writer, and wires the
/// Iceberg field ids into the writer options.
///
/// Each file format extracts statistics from a different source: DWRF/ORC read
/// the writer's footer proto from the still-alive writer, while Parquet reads
/// the FileMetadata returned by Writer::close(). That format-specific logic is
/// encapsulated behind this interface so IcebergDataSink holds a single base
/// pointer and never branches on file format. Create one instance per
/// IcebergDataSink via create() and reuse it across all writers.
class IcebergStatsCollector {
 public:
  virtual ~IcebergStatsCollector() = default;

  /// Wires the Iceberg field ids for the input columns into the format-specific
  /// writer options (Parquet column metadata or DWRF/ORC "iceberg.id"
  /// attributes). A no-op when 'options' is not for this collector's format.
  virtual void configureWriterOptions(
      dwio::common::WriterOptions& options) const = 0;

  /// Aggregates per-file Iceberg column statistics from a just-closed writer.
  /// @param writer The live writer (DWRF/ORC read its footer proto).
  /// @param closeMetadata The metadata returned by Writer::close() (Parquet
  /// consumes it; DWRF/ORC return an empty placeholder and ignore it).
  /// @return The aggregated statistics, or nullptr when statistics are
  /// unavailable, in which case the caller derives a row-count-only estimate.
  virtual IcebergDataFileStatisticsPtr collect(
      dwio::common::Writer& writer,
      std::unique_ptr<dwio::common::FileMetadata>& closeMetadata) const = 0;

  /// Creates the statistics collector for 'format', or nullptr when the format
  /// has no Iceberg statistics support compiled in (e.g. Parquet without
  /// VELOX_ENABLE_PARQUET).
  /// @param inputColumns The Iceberg input column handles (carry the field-id
  /// trees).
  /// @param schema The written row type.
  static std::unique_ptr<IcebergStatsCollector> create(
      dwio::common::FileFormat format,
      const std::vector<IcebergColumnHandlePtr>& inputColumns,
      const RowTypePtr& schema);
};

} // namespace facebook::velox::connector::hive::iceberg
