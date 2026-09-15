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

  // For changelog queries, build the ChangelogScanContext once so it is
  // reused across all splits.  This lets stats-based filter reordering and
  // column adaptation accumulate rather than being discarded after each split.
  const auto& dataColumns = tableHandle_->dataColumns();
  VELOX_CHECK_NOT_NULL(
      dataColumns,
      "IcebergDataSource: changelog query requires tableHandle.dataColumns");

  const auto& rawDataHandles = icebergTableHandle->dataColumnHandles();
  auto dataColumnHandles = std::make_shared<ColumnHandleMap>();
  for (const auto& [name, handle] : rawDataHandles) {
    dataColumnHandles->emplace(name, handle);
  }

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

  // Changelog metadata filters (operation/ordinal/snapshotid) must NOT be
  // forwarded — those names don't exist in the base-table schema.
  // rowdata.* subfield filters are rejected in createSplitReader().
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

std::unique_ptr<FileSplitReader> IcebergDataSource::createSplitReader() {
  prepareSplit();
  auto icebergSplit = checkedPointerCast<const HiveIcebergSplit>(split_);
  auto* icebergTableHandle =
      dynamic_cast<const IcebergTableHandle*>(tableHandle_.get());

  if (icebergTableHandle && icebergTableHandle->isChangelogQuery()) {
    // Subfield filters on rowdata.* columns are silently dropped because the
    // split reader's row reader uses dataScanSpec (no filters), so the
    // predicate never reaches the data.  Reject such filters explicitly so the
    // caller receives a clear error rather than incorrect results.
    //
    // Predicates on rowdata columns must be expressed as a remainingFilter
    // (post-scan Filter operator) evaluated against the changelog output.
    for (const auto& [subfield, filter] : filters_) {
      const auto& path = subfield.path();
      if (path.empty()) {
        continue;
      }
      const auto* root = path[0]->as<common::Subfield::NestedField>();
      if (root == nullptr) {
        continue;
      }
      VELOX_USER_CHECK(
          root->name() != kChangelogColRowdata,
          "Subfield filter pushdown on rowdata columns is not supported for "
          "changelog queries. Predicates on rowdata columns must be expressed "
          "as a remainingFilter. Unsupported filter: {}",
          subfield.toString());
    }

    VELOX_CHECK(
        changelogScanContext_.has_value(),
        "ChangelogScanContext not initialised — this should not happen");
    return std::make_unique<IcebergChangelogSplitReader>(
        icebergSplit,
        tableHandle_,
        &partitionKeys_,
        connectorQueryCtx_,
        fileConfig_,
        *changelogScanContext_,
        dataIoStats_,
        metadataIoStats_,
        ioStats_,
        fileHandleFactory_,
        ioExecutor_,
        outputType(),
        *columnHandles_,
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
