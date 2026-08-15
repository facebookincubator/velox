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

#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergChangelogSplitReader.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/IcebergSplitReader.h"

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
      columnHandles_(std::make_shared<ColumnHandleMap>(assignments)) {}

std::unique_ptr<FileSplitReader> IcebergDataSource::createSplitReader() {
  prepareSplit();
  auto icebergSplit = checkedPointerCast<const HiveIcebergSplit>(split_);

  auto* hiveTableHandle =
      dynamic_cast<const HiveTableHandle*>(tableHandle_.get());

  if (hiveTableHandle && hiveTableHandle->isChangelogQuery()) {
    // Changelog query: build a base-table readerOutputType and scanSpec from
    // the data column handles stored in the table handle, then create an
    // IcebergChangelogSplitReader that reads the data columns and wraps each
    // batch into the changelog output schema.
    const auto& dataColumns = tableHandle_->dataColumns();
    VELOX_CHECK_NOT_NULL(
        dataColumns,
        "IcebergDataSource: changelog query requires tableHandle.dataColumns");

    const auto& rawDataHandles = hiveTableHandle->getDataColumnHandles();
    auto dataColumnHandles = std::make_shared<ColumnHandleMap>(rawDataHandles);

    // Build the base-table readerOutputType: include columns present in both
    // the table schema and the data column handles.
    std::vector<std::string> dataNames;
    std::vector<TypePtr> dataTypes;
    for (uint32_t i = 0; i < dataColumns->size(); ++i) {
      const auto& physName = dataColumns->nameOf(i);
      if (rawDataHandles.count(physName)) {
        dataNames.push_back(physName);
        dataTypes.push_back(dataColumns->childAt(i));
      }
    }
    auto dataReaderOutputType = ROW(std::move(dataNames), std::move(dataTypes));

    // Build a data-scoped scanSpec covering the projected base-table columns.
    // Pass empty filters: changelog metadata filters (operation/ordinal/
    // snapshotid) are constant per-split and evaluated in
    // IcebergChangelogSplitReader::prepareSplit(); they must NOT be forwarded
    // to makeScanSpec because those names do not exist in the base-table schema
    // and would cause "Field not found" errors.  Data-column predicates (e.g.
    // rowdata.id < 50) reference the nested "rowdata" namespace and are not
    // present as bare subfield filters here; any remaining predicate is
    // evaluated by the downstream Filter operator.
    auto dataScanSpec = makeScanSpec(
        dataReaderOutputType,
        subfields_,
        common::SubfieldFilters{},
        /*indexColumns=*/{},
        tableHandle_->dataColumns(),
        partitionKeys_,
        infoColumns_,
        specialColumns_,
        fileConfig_->readStatsBasedFilterReorderDisabled(
            connectorQueryCtx_->sessionProperties()),
        pool_);

    auto changelogOutputType = getOutputType();
    // columnHandles_ holds the changelog output column handles
    // (operation/ordinal/snapshotid/rowdata) — passed as
    // changelogColumnHandles.
    return std::make_unique<IcebergChangelogSplitReader>(
        icebergSplit,
        tableHandle_,
        &partitionKeys_,
        connectorQueryCtx_,
        fileConfig_,
        dataReaderOutputType,
        dataIoStats_,
        metadataIoStats_,
        ioStats_,
        fileHandleFactory_,
        ioExecutor_,
        dataScanSpec,
        dataColumnHandles,
        changelogOutputType, // changelog output type
        *columnHandles_, // changelog column handles
        &filters_);
  }

  // Regular (non-changelog) Iceberg query.
  return std::make_unique<IcebergSplitReader>(
      icebergSplit,
      tableHandle_,
      &partitionKeys_,
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
