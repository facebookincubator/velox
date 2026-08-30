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

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>

#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergDataFileStatistics.h"
#include "velox/connectors/hive/iceberg/IcebergStatsCollector.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/type/Type.h"

namespace facebook::velox::dwrf {
class FooterWrapper;
struct StatsContext;
} // namespace facebook::velox::dwrf

namespace facebook::velox::connector::hive::iceberg {

/// Aggregates per-file Iceberg column statistics (column sizes, value counts,
/// null counts, lower/upper bounds) from the DWRF/ORC writer footer.
///
/// Unlike Parquet (whose writer returns a self-describing FileMetadata),
/// dwrf::Writer::close() returns an empty placeholder. DWRF column statistics
/// live in the footer proto (one proto::ColumnStatistics per pre-order schema
/// node id), reachable from the still-alive writer via
/// dwrf::Writer::getFooter(). aggregate() therefore consumes a
/// dwrf::FooterWrapper read from the live writer rather than a FileMetadata.
///
/// Create one instance per IcebergDataSink and reuse it across all writers.
class IcebergDwrfStatsCollector : public IcebergStatsCollector {
 public:
  /// @param inputColumns The Iceberg input column handles. Each carries the
  /// hierarchical Iceberg field-id tree (ParquetFieldId) for one top-level
  /// column.
  /// @param schema The written row type. Walked together with the field-id
  /// trees to map pre-order DWRF footer node ids to Iceberg field ids.
  IcebergDwrfStatsCollector(
      const std::vector<IcebergColumnHandlePtr>& inputColumns,
      const RowTypePtr& schema);

  /// Aggregates DWRF footer statistics into Iceberg data file statistics.
  /// For each schema node that maps to an Iceberg field id, reads the footer's
  /// per-node ColumnStatistics and populates:
  /// - columnSize from the column's total stream length.
  /// - valueCount: numRecords for top-level flat columns, otherwise the
  ///   non-null value count from the stats (caveat: DWRF reports non-null
  ///   count, not total occurrences, for nested columns).
  /// - nullValueCount: numRecords - nonNullCount for top-level flat columns;
  ///   left unset for nested columns (the mapping is not 1:1 with row count).
  /// - lower/upper bounds (base64-encoded Iceberg single-value binary) for
  ///   types that expose scalar min/max in DWRF stats and are not MAP/ARRAY
  ///   descendants.
  /// @param footer The DWRF footer read from the live writer.
  /// @param statsContext Writer name/version used to interpret the proto stats.
  IcebergDataFileStatisticsPtr aggregate(
      const dwrf::FooterWrapper& footer,
      const dwrf::StatsContext& statsContext) const;

  void configureWriterOptions(
      dwio::common::WriterOptions& options) const override;

  IcebergDataFileStatisticsPtr collect(
      dwio::common::Writer& writer,
      std::unique_ptr<dwio::common::FileMetadata>& closeMetadata)
      const override;

  /// TODO: Need to support this config property.
  /// 16 is default value. See DEFAULT_WRITE_METRICS_MODE_DEFAULT in
  /// org.apache.iceberg.TableProperties.
  constexpr static int32_t kDefaultTruncateLength{16};

 private:
  // Per-node metadata captured during the schema walk.
  struct NodeInfo {
    // Iceberg field id for this node.
    int32_t fieldId{};
    // Velox logical type, used to choose the bound serializer (e.g. DATE vs
    // INTEGER, TIMESTAMP vs BIGINT, decimal precision/scale).
    TypePtr type;
    // True only for primitive columns that are direct children of the root
    // row. valueCount/nullValueCount are row-count-derived only for these.
    bool topLevelFlat{};
  };

  bool shouldStoreBounds(int32_t fieldId) const {
    return !skipBoundsFieldIds_.contains(fieldId);
  }

  // Recursively records NodeInfo for 'node' and its descendants, keyed by
  // pre-order schema node id (matching the DWRF footer node ids). 'field' is
  // the Iceberg field-id tree aligned to 'node'.
  void buildNodeInfo(
      const dwio::common::TypeWithId& node,
      const parquet::ParquetFieldId& field,
      bool topLevel);

  // Maps pre-order DWRF footer node id to its Iceberg field id and type.
  folly::F14FastMap<uint32_t, NodeInfo> nodeInfo_;

  // Iceberg field ids whose bounds collection is skipped: MAP and ARRAY types
  // and all of their descendants.
  folly::F14FastSet<int32_t> skipBoundsFieldIds_;

  // Iceberg input column handles, retained to build the DWRF/ORC "iceberg.id"
  // schema attributes in configureWriterOptions().
  std::vector<IcebergColumnHandlePtr> inputColumns_;
};

} // namespace facebook::velox::connector::hive::iceberg
