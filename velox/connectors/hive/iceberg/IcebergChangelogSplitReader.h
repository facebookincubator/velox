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

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/iceberg/IcebergChangelogSplitInfo.h"
#include "velox/connectors/hive/iceberg/IcebergSplitReader.h"

namespace facebook::velox::connector::hive::iceberg {

/// Immutable scan state shared across all splits of a single changelog query.
///
/// Built once in IcebergDataSource's constructor from the table handle so that
/// stats-based filter reordering and column adaptation accumulated in
/// dataScanSpec across splits are not discarded between splits.
struct ChangelogScanContext {
  /// Base-table column handles keyed by column name.
  std::shared_ptr<ColumnHandleMap> dataColumnHandles;

  /// Base-table projected schema (subset of dataColumns present in
  /// dataColumnHandles).
  RowTypePtr dataReaderOutputType;

  /// ScanSpec for the base-table scan.  Shared so that per-split adaptation
  /// (filter reordering, bloom-filter caches) accumulates across splits.
  std::shared_ptr<common::ScanSpec> dataScanSpec;
};

/// Split reader for Iceberg changelog table queries.
///
/// Inherits the full IcebergSplitReader file-reading pipeline and adds
/// changelog-specific behaviour:
///
///   - readerOutputType(): reports the changelog output schema
///     (operation, ordinal, snapshotid, rowdata) to FileDataSource so that
///     output_ is allocated with the correct shape.
///
///   - next(): reads a base-table batch via IcebergSplitReader::next() into a
///     private buffer, then transforms it into the changelog output schema
///     before writing into the caller-supplied output vector.
///
///   - prepareSplit(): evaluates the constant-column subfield filters
///     (operation / ordinal / snapshotid) from filters_ against the split's
///     ChangelogSplitInfo. Marks the split empty when any filter rejects the
///     constant, avoiding all file I/O.
///
/// The parent IcebergSplitReader is constructed with the base-table
/// readerOutputType so its row reader reads the correct data columns.
/// IcebergChangelogSplitReader keeps the changelog output type separately and
/// overrides readerOutputType() so the FileDataSource layer sees the changelog
/// schema for output allocation and remaining-filter evaluation.
class IcebergChangelogSplitReader : public IcebergSplitReader {
 public:
  IcebergChangelogSplitReader(
      const std::shared_ptr<const HiveIcebergSplit>& icebergSplit,
      const FileTableHandlePtr& tableHandle,
      const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const FileConfig>& fileConfig,
      const ChangelogScanContext& scanContext,
      const std::shared_ptr<io::IoStatistics>& dataIoStats,
      const std::shared_ptr<io::IoStatistics>& metadataIoStats,
      const std::shared_ptr<IoStats>& ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const RowTypePtr& changelogOutputType,
      ColumnHandleMap changelogColumnHandles,
      const common::SubfieldFilters* changelogFilters);

  /// Returns the changelog output schema so FileDataSource allocates output_
  /// with the right shape and evaluates the remaining filter against changelog
  /// column names.
  const RowTypePtr& readerOutputType() const override {
    return changelogOutputType_;
  }

  void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStats& runtimeStats,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {})
      override;

  uint64_t next(uint64_t size, VectorPtr& output) override;

 private:
  /// Evaluates the changelog constant-column subfield filters
  /// (operation/ordinal/snapshotid) against changelogSplitInfo_.
  /// Returns true if the split passes all filters (or there are none),
  /// false if any filter rejects the split's constant values.
  /// Requires changelogSplitInfo_ to be set before calling.
  bool applyChangelogFilters() const;

  /// Builds a single constant or rowdata column vector for the changelog
  /// output schema from a data batch produced by the base reader.
  VectorPtr buildChangelogColumn(
      const RowVectorPtr& dataOutput,
      const std::string& fieldName,
      column_index_t colIdx,
      vector_size_t positionCount) const;

  /// Changelog output schema: (operation VARCHAR, ordinal BIGINT,
  /// snapshotid BIGINT, rowdata ROW<...>).
  const RowTypePtr changelogOutputType_;

  /// Column handles for changelog output columns, keyed by output column name.
  const ColumnHandleMap changelogColumnHandles_;

  /// Changelog metadata for the current split, set in prepareSplit().
  /// Points into the split's changelogSplitInfo optional; valid for the
  /// lifetime of the split.
  const ChangelogSplitInfo* changelogSplitInfo_{nullptr};

  /// Pointer to FileDataSource::filters_, containing the changelog
  /// constant-column subfield filters (operation/ordinal/snapshotid).
  /// Not owned; lifetime is guaranteed by the owning FileDataSource.
  const common::SubfieldFilters* const changelogFilters_;

  /// Reusable buffer for the base-table batch produced by IcebergSplitReader.
  VectorPtr dataOutput_;
};

} // namespace facebook::velox::connector::hive::iceberg
