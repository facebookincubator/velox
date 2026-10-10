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

#include "velox/connectors/hive/iceberg/IcebergDataSource.h"

#include "velox/connectors/hive/FileScanState.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergConnector.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/IcebergSplitReader.h"
#include "velox/connectors/hive/iceberg/IcebergTableHandle.h"

namespace facebook::velox::connector::hive::iceberg {

IcebergDataSource::IcebergDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& assignments,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<HiveConfig>& hiveConfig)
    : HiveDataSource(
          outputType,
          tableHandle,
          assignments,
          fileHandleFactory,
          ioExecutor,
          connectorQueryCtx,
          hiveConfig),
      columnHandles_(std::make_shared<ColumnHandleMap>(assignments)) {
  auto* icebergTableHandle =
      dynamic_cast<const IcebergTableHandle*>(tableHandle_.get());
  if (!icebergTableHandle || !icebergTableHandle->isChangelogQuery()) {
    return;
  }

  // Validate post-extraction subfield filters.  The pre-construction check in
  // IcebergConnector::createDataSource validates handle.subfieldFilters()
  // before FileDataSource's constructor runs.  However, FileDataSource's
  // constructor calls extractFiltersFromRemainingFilter which can pull
  // additional entries into filters_  (e.g. "rowdata.id < 50" expressed as a
  // remainingFilter becomes Subfield("rowdata.id") in filters_).  Those
  // extracted entries bypass the pre-construction check, and
  // applyChangelogFilters only inspects operation/ordinal/snapshotid, so the
  // predicate silently disappears and extra rows are returned.  Re-running the
  // validation here against the fully-populated filters_ turns the silent miss
  // into a loud error.
  IcebergConnector::validateChangelogSubfieldFilters(filters_);

  // Changelog columns use a separate base-table schema. The physical scan
  // spec is rebuilt for each split while the column demands are retained.
  const auto& dataColumns = tableHandle_->dataColumns();
  VELOX_CHECK_NOT_NULL(
      dataColumns,
      "IcebergDataSource: changelog query requires tableHandle.dataColumns");

  const auto& rawDataHandles = icebergTableHandle->dataColumnHandles();
  auto dataColumnHandles = std::make_shared<ColumnHandleMap>(
      rawDataHandles.begin(), rawDataHandles.end());

  std::vector<std::string> dataNames;
  std::vector<TypePtr> dataTypes;
  for (uint32_t i = 0; i < dataColumns->size(); ++i) {
    const auto& name = dataColumns->nameOf(i);
    if (rawDataHandles.contains(name)) {
      dataNames.push_back(name);
      dataTypes.push_back(dataColumns->childAt(i));
    }
  }
  auto dataReaderOutputType = ROW(std::move(dataNames), std::move(dataTypes));

  // Changelog metadata filters (operation/ordinal/snapshotid) must NOT be
  // forwarded — those names don't exist in the base-table schema.
  auto dataScanSpec = makeScanSpec(
      dataReaderOutputType,
      /*outputSubfields=*/{},
      common::SubfieldFilters{},
      /*indexColumns=*/{},
      tableHandle_->dataColumns(),
      partitionKeys_,
      infoColumns_,
      specialColumns_,
      fileConfig_->readStatsBasedFilterReorderDisabled(
          connectorQueryCtx_->sessionProperties()),
      pool_);

  changelogScanContext_ = ChangelogScanContext{
      std::move(dataColumnHandles),
      std::move(dataReaderOutputType),
      std::move(dataScanSpec)};
}

void IcebergDataSource::addDynamicFilter(
    column_index_t outputChannel,
    const std::shared_ptr<common::Filter>& filter) {
  if (!changelogScanContext_.has_value()) {
    // Regular (non-changelog) query: delegate to the base implementation which
    // sets the filter on scanSpec_ for the row reader to consume.
    FileDataSource::addDynamicFilter(outputChannel, filter);
    return;
  }

  // Changelog query: translate the output channel to a column name and
  // accumulate into changelogDynamicFilters_ for split-level evaluation.
  //
  // rowdata is a ROW-typed column; HashProbe never produces a pushable filter
  // for ROW types (VectorHasher::getFilter returns null for them), so this
  // branch is unreachable in practice.  If it were somehow reached, delegating
  // to FileDataSource::addDynamicFilter would set the filter on scanSpec_
  // (changelog-space) rather than dataScanSpec (base-table), silently dropping
  // it and producing wrong results.  We therefore drop it explicitly here —
  // the join key comparison in HashProbe will still enforce correctness.
  const auto& colName = outputType()->nameOf(outputChannel);
  if (colName == kChangelogColRowdata) {
    return;
  }

  // Driver::pushdownFilters already merges all filters for a given channel
  // before calling addDynamicFilter, so the incoming filter is already the
  // intersection of all dynamic filters for this column.  Mirror what
  // ScanSpec::setFilter does: unconditional overwrite.
  auto [it, inserted] = changelogDynamicFilters_.emplace(
      common::Subfield(std::string(colName)), filter);
  if (!inserted) {
    it->second = filter;
  }
}

std::unique_ptr<FileSplitReader> IcebergDataSource::createSplitReader() {
  prepareSplit();
  auto icebergSplit = checkedPointerCast<const HiveIcebergSplit>(split_);

  if (changelogScanContext_.has_value()) {
    auto& context = *changelogScanContext_;
    auto dataScanSpec = makeScanSpec(
        context.dataReaderOutputType,
        /*outputSubfields=*/{},
        common::SubfieldFilters{},
        /*indexColumns=*/{},
        tableHandle_->dataColumns(),
        partitionKeys_,
        infoColumns_,
        specialColumns_,
        fileConfig_->readStatsBasedFilterReorderDisabled(
            connectorQueryCtx_->sessionProperties()),
        pool_);
    dataScanSpec->moveAdaptationFrom(*context.dataScanSpec);
    context.dataScanSpec = std::move(dataScanSpec);

    // Pass readerOutputType_ (not outputType()) so that columns referenced
    // only by the remainingFilter — which FileDataSource::constructor appended
    // to readerOutputType_ but omitted from outputType_ — are present in the
    // RowVector that evaluateRemainingFilter receives. FileDataSource::addSplit
    // gets readerOutputType_ through the default reader adapter after
    // createSplitReader() returns; passing the pre-overwrite value here ensures
    // the shape matches what the compiled ExprSet expects.
    return std::make_unique<IcebergChangelogSplitReader>(
        icebergSplit,
        tableHandle_,
        &fileScanSpec_->partitionKeys(),
        connectorQueryCtx_,
        fileConfig_,
        *changelogScanContext_,
        dataIoStats_,
        metadataIoStats_,
        ioStats_,
        fileHandleFactory_,
        ioExecutor_,
        readerOutputType_,
        *columnHandles_,
        &fileScanState_->filters,
        &changelogDynamicFilters_);
  }

  // Regular (non-changelog) Iceberg query.
  return std::make_unique<IcebergSplitReader>(
      icebergSplit,
      tableHandle_,
      &fileScanSpec_->partitionKeys(),
      connectorQueryCtx_,
      fileConfig_,
      readerOutputType_,
      dataIoStats_,
      metadataIoStats_,
      ioStats_,
      fileHandleFactory_,
      ioExecutor_,
      scanSpec_,
      columnHandles_);
}

} // namespace facebook::velox::connector::hive::iceberg
