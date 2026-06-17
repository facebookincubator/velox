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

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/iceberg/IcebergConfig.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"
#include "velox/connectors/hive/iceberg/IcebergDeletionVectorSink.h"

namespace facebook::velox::connector::hive::iceberg {

/// Composite DataSink that fans an input RowVector tagged with a per-row
/// `operation` byte out to two underlying sinks:
///   - INSERT rows  → an inner `IcebergDataSink`              (kData)
///   - DELETE rows  → an inner `IcebergDeletionVectorSink`    (kDeletionVector)
///
/// Selected by `IcebergConnector::createDataSink` when the incoming
/// `IcebergInsertTableHandle` has `writeKind() == kMerge`. Mirrors the
/// connector-side behavior of OSS Java `IcebergAbstractMetadata.finishMerge` /
/// `finishUpdate` which absorbs a single mixed fragment stream into one
/// Iceberg snapshot (positional-delete files + new data files committed
/// atomically).
///
/// Upstream contract:
///   - The composite is fed by Layer 1's `IcebergMergeProcessor`, whose
///     output schema is `[target_cols…, operation TINYINT, row_id ROW,
///     insert_from_update TINYINT]`.
///   - UPDATE (3) and DEFAULT_CASE (-1) bytes are fanned out / dropped by
///     `IcebergMergeProcessor` before reaching this sink; only INSERT (1)
///     and DELETE (2) bytes are valid here. Any other value triggers
///     `VELOX_USER_FAIL`.
///   - The constructor takes explicit channel indices so the upstream
///     plan-node ordering is decoupled from the sink — Layer 3 wires these
///     up from the protocol payload.
///
/// Sub-sink wiring:
///   The composite clones the incoming `IcebergInsertTableHandle` into two
///   narrow handles — one with `WriteKind::kData` for the data sink and one
///   with `WriteKind::kDeletionVector` for the DV sink. The narrow handles
///   share the same `locationHandle`, `partitionSpec`, `inputColumns`,
///   `storageFormat`, `compressionKind`, and `serdeParameters` as the
///   original. Sub-sinks see their normal handle types; no invariant
///   relaxation is required.
class IcebergMergeSink : public DataSink {
 public:
  /// Operation byte values that may appear on the composite's input. Mirrors
  /// `IcebergMergeProcessor::k{Insert,Delete}OperationNumber`; UPDATE and
  /// DEFAULT_CASE are intentionally NOT accepted here (Layer 1 must have
  /// fanned them out / dropped them already).
  static constexpr int8_t kInsertOperationNumber = 1;
  static constexpr int8_t kDeleteOperationNumber = 2;

  /// @param inputType Schema of every RowVector passed to `appendData`. Must
  /// have at least `max(targetColumnChannels) + 1`,
  /// `operationChannel + 1`, and `rowIdChannel + 1` fields.
  /// @param insertTableHandle The original `kMerge` handle. Cloned twice
  /// internally into narrow `kData` / `kDeletionVector` handles.
  /// @param targetColumnChannels Indices of the target table data columns
  /// within `inputType`, in the same order as the target table schema. The
  /// inner data sink will see exactly these columns. The set must match the
  /// `inputColumns()` count on `insertTableHandle`.
  /// @param operationChannel Index of the TINYINT operation column.
  /// @param rowIdChannel Index of the row id ROW column (must be a ROW
  /// type whose first two fields are file_path VARCHAR and pos BIGINT —
  /// extra trailing fields are tolerated and ignored).
  IcebergMergeSink(
      RowTypePtr inputType,
      IcebergInsertTableHandlePtr insertTableHandle,
      const ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      const IcebergConfigPtr& icebergConfig,
      std::vector<column_index_t> targetColumnChannels,
      column_index_t operationChannel,
      column_index_t rowIdChannel);

  void appendData(RowVectorPtr input) override;

  bool finish() override;

  std::vector<std::string> close() override;

  void abort() override;

  Stats stats() const override;

 private:
  // Builds a narrow data-sub-sink batch (one row per insert-tagged input
  // row) by dictionary-wrapping the target column children using the
  // pre-computed `insertIndices` buffer. Returned RowVector has type
  // `dataInputType_`.
  RowVectorPtr makeInsertBatch(
      const RowVectorPtr& input,
      const BufferPtr& insertIndices,
      vector_size_t insertSize) const;

  // Builds a narrow DV-sub-sink batch (one row per delete-tagged input row)
  // by extracting the first two children of the row_id ROW field
  // (file_path, pos) and dictionary-wrapping each using the pre-computed
  // `deleteIndices` buffer. Returned RowVector has type
  // `deletionVectorInputType_`.
  RowVectorPtr makeDeleteBatch(
      const RowVectorPtr& input,
      const BufferPtr& deleteIndices,
      vector_size_t deleteSize) const;

  // Builds an `IcebergInsertTableHandle` identical to `original` except for
  // its `writeKind`. Used to construct narrow `kData` / `kDeletionVector`
  // handles for the inner sub-sinks.
  static IcebergInsertTableHandlePtr cloneHandleWithKind(
      const IcebergInsertTableHandle& original,
      IcebergInsertTableHandle::WriteKind kind);

  // Projects `inputType` to a RowType containing only the columns named by
  // `targetColumnChannels`, but using the iceberg-schema column names from
  // `insertTableHandle.inputColumns()` (one entry per channel, in order).
  // Deriving names from the handle rather than the source RowVector makes
  // `dataInputType_` independent of whatever upstream named the columns,
  // so the writer's name-based binding (TableWriter::setTypeMappings) always
  // sees the iceberg-correct names.
  static RowTypePtr projectDataInputType(
      const RowTypePtr& inputType,
      const std::vector<column_index_t>& targetColumnChannels,
      const IcebergInsertTableHandle& insertTableHandle);

  // The 2-column ROW type the DV sub-sink consumes: (file_path VARCHAR,
  // pos BIGINT). Validated against the first two children of the row_id
  // ROW at construction time.
  static RowTypePtr makeDeletionVectorInputType();

  const RowTypePtr inputType_;
  const IcebergInsertTableHandlePtr insertTableHandle_;
  const ConnectorQueryCtx* const connectorQueryCtx_;
  const std::vector<column_index_t> targetColumnChannels_;
  const column_index_t operationChannel_;
  const column_index_t rowIdChannel_;
  const RowTypePtr dataInputType_;
  const RowTypePtr deletionVectorInputType_;

  std::unique_ptr<IcebergDataSink> dataSink_;
  std::unique_ptr<IcebergDeletionVectorSink> deletionVectorSink_;

  // Sub-sink completion latches. `finish()` may yield by returning false
  // when the inner data sink is mid-flush; on the next call we must not
  // re-finish a sink that already returned true.
  bool dataSinkFinished_{false};
  bool deletionVectorSinkFinished_{false};
  bool finished_{false};
  bool aborted_{false};

  // close() latch. The first close() drains both sub-sinks and caches their
  // combined commit messages here; subsequent close() calls return the cached
  // vector unchanged, so close() is idempotent (sub-sink close() is not).
  bool closed_{false};
  std::vector<std::string> commitMessages_;
};

} // namespace facebook::velox::connector::hive::iceberg
