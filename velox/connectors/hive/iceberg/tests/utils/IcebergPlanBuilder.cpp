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

#include "velox/connectors/hive/iceberg/tests/utils/IcebergPlanBuilder.h"

#include <unordered_set>

#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergTableHandle.h"

namespace facebook::velox::connector::hive::iceberg::test {

connector::ColumnHandlePtr IcebergTableScanBuilder::buildConnectorColumnHandle(
    const std::string& name,
    const TypePtr& type,
    uint32_t /*outputIndex*/) {
  int32_t fieldId = -1;
  if (!dataColumnFieldIds_.empty() && dataColumns_ != nullptr) {
    if (auto index = dataColumns_->getChildIdxIfExists(name)) {
      fieldId = dataColumnFieldIds_[*index];
    }
  }
  return std::make_shared<IcebergColumnHandle>(
      name,
      FileColumnHandle::ColumnType::kRegular,
      type,
      parquet::ParquetFieldId{fieldId, {}});
}

namespace {

// Recursively collects all field-access (input column) names from 'expr'.
void collectFieldNames(
    const core::TypedExprPtr& expr,
    std::unordered_set<std::string>& names) {
  if (!expr) {
    return;
  }
  if (auto* fieldAccess =
          dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get())) {
    names.insert(fieldAccess->name());
  }
  for (const auto& input : expr->inputs()) {
    collectFieldNames(input, names);
  }
}

} // namespace

connector::ConnectorTableHandlePtr
IcebergTableScanBuilder::buildConnectorTableHandle(
    common::SubfieldFilters subfieldFilters,
    const core::TypedExprPtr& remainingFilter) {
  // Collect names of all columns referenced by pushed-down filters.
  std::unordered_set<std::string> filterColumnNames;
  for (const auto& [subfield, _] : subfieldFilters) {
    filterColumnNames.insert(subfield.baseName());
  }
  if (remainingFilter) {
    collectFieldNames(remainingFilter, filterColumnNames);
  }

  // Auto-build filter-only handles for columns not already covered.
  if (!filterColumnNames.empty() && dataColumns_ != nullptr) {
    std::unordered_set<std::string> covered;
    for (const auto& handle : filterColumnHandles_) {
      covered.insert(handle->name());
    }
    for (uint32_t i = 0; i < dataColumns_->size(); ++i) {
      const auto& columnName = dataColumns_->nameOf(i);
      if (!filterColumnNames.count(columnName)) {
        continue;
      }
      if (assignments_.count(columnName) || covered.count(columnName)) {
        continue;
      }
      int32_t fieldId = -1;
      if (!dataColumnFieldIds_.empty()) {
        fieldId = dataColumnFieldIds_[i];
      }
      filterColumnHandles_.push_back(
          std::make_shared<IcebergColumnHandle>(
              columnName,
              FileColumnHandle::ColumnType::kRegular,
              dataColumns_->childAt(i),
              parquet::ParquetFieldId{fieldId, {}}));
    }
  }

  // Downcast every filterColumnHandle to IcebergColumnHandle.
  std::vector<IcebergColumnHandlePtr> icebergFilterHandles;
  icebergFilterHandles.reserve(filterColumnHandles_.size());
  for (const auto& handle : filterColumnHandles_) {
    auto icebergHandle =
        std::dynamic_pointer_cast<const IcebergColumnHandle>(handle);
    VELOX_CHECK_NOT_NULL(
        icebergHandle,
        "IcebergTableScanBuilder: filterColumnHandle '{}' is not an "
        "IcebergColumnHandle",
        handle->name());
    icebergFilterHandles.push_back(std::move(icebergHandle));
  }

  return std::make_shared<const IcebergTableHandle>(
      connectorId_,
      tableName_,
      std::move(subfieldFilters),
      remainingFilter,
      dataColumns_,
      indexColumns_,
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      std::move(icebergFilterHandles),
      sampleRate_,
      /*dbName=*/"",
      dataColumnFieldIds_);
}

IcebergTableScanBuilder& IcebergPlanBuilder::startTableScan(
    std::string connectorId) {
  icebergTableScanBuilder_ = std::make_shared<IcebergTableScanBuilder>(*this);
  icebergTableScanBuilder_->connectorId(std::move(connectorId));
  // Keep the base tableScanBuilder_ in sync so endTableScan() delegates
  // through the right object.
  tableScanBuilder_ = icebergTableScanBuilder_;
  return *icebergTableScanBuilder_;
}

} // namespace facebook::velox::connector::hive::iceberg::test
