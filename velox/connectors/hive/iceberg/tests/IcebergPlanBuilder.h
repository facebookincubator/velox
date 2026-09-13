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

#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergTableHandle.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg::test {

/// Default connector ID used by IcebergPlanBuilder and IcebergTestBase.
inline const std::string kIcebergConnectorId{"test-iceberg"};

/// A TableScanBuilder subclass that constructs an IcebergTableHandle instead
/// of a HiveTableHandle. All filter-string parsing (subfield filters, remaining
/// filter) is performed by the base class build() — zero duplication.
///
/// Assignments are auto-built as IcebergColumnHandles so that
/// IcebergSplitReader can resolve field IDs. Caller-supplied assignments
/// (via .assignments()) take precedence when provided.
class IcebergTableScanBuilder
    : public exec::test::PlanBuilder::TableScanBuilder {
 public:
  explicit IcebergTableScanBuilder(exec::test::PlanBuilder& planBuilder)
      : TableScanBuilder(planBuilder) {}

  /// Overrides the base to accept connector::ColumnHandlePtr, asserting that
  /// each handle is an IcebergColumnHandle. Enables chaining on the derived
  /// type via covariant return.
  IcebergTableScanBuilder& filterColumnHandles(
      std::vector<connector::ColumnHandlePtr> handles) override {
    filterColumnHandles_.clear();
    filterColumnHandles_.reserve(handles.size());
    for (auto& h : handles) {
      VELOX_CHECK_NOT_NULL(
          std::dynamic_pointer_cast<const IcebergColumnHandle>(h),
          "IcebergTableScanBuilder: filterColumnHandle '{}' is not an "
          "IcebergColumnHandle",
          h->name());
      filterColumnHandles_.push_back(std::move(h));
    }
    return *this;
  }

 protected:
  /// Overrides the base factory to always produce an IcebergColumnHandle.
  /// When dataColumnFieldIds_ is set, the handle carries the real Iceberg field
  /// ID (looked up by column name in dataColumns_). Otherwise the sentinel
  /// value -1 is used.
  connector::ColumnHandlePtr buildDefaultColumnHandle(
      const std::string& name,
      const TypePtr& type,
      uint32_t /*outputIndex*/) override {
    int32_t fieldId = -1;
    if (!dataColumnFieldIds_.empty() && dataColumns_ != nullptr) {
      if (auto idx = dataColumns_->getChildIdxIfExists(name)) {
        fieldId = dataColumnFieldIds_[*idx];
      }
    }
    return std::make_shared<IcebergColumnHandle>(
        name,
        FileColumnHandle::ColumnType::kRegular,
        type,
        parquet::ParquetFieldId{fieldId, {}});
  }

  /// Overrides the base factory to construct an IcebergTableHandle carrying
  /// all the already-parsed filter state, plus Iceberg-specific field IDs.
  connector::ConnectorTableHandlePtr buildConnectorTableHandle(
      common::SubfieldFilters subfieldFilters,
      const core::TypedExprPtr& remainingFilter) override {
    // filterColumnHandles_ entries were already asserted to be
    // IcebergColumnHandle in the filterColumnHandles() override above.
    std::vector<IcebergColumnHandlePtr> icebergFilterHandles;
    icebergFilterHandles.reserve(filterColumnHandles_.size());
    for (const auto& h : filterColumnHandles_) {
      icebergFilterHandles.push_back(
          std::static_pointer_cast<const IcebergColumnHandle>(h));
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
};

/// A PlanBuilder subclass whose startTableScan() returns an
/// IcebergTableScanBuilder so every scan node is backed by an
/// IcebergTableHandle. All fluent builder methods (outputType, dataColumns,
/// subfieldFilters, remainingFilter, assignments, filterColumnHandles,
/// dataColumnFieldIds, …) are inherited unchanged.
class IcebergPlanBuilder : public exec::test::PlanBuilder {
 public:
  using PlanBuilder::PlanBuilder;

  IcebergTableScanBuilder& startTableScan(
      std::string connectorId = kIcebergConnectorId) {
    icebergTableScanBuilder_ = std::make_shared<IcebergTableScanBuilder>(*this);
    icebergTableScanBuilder_->connectorId(std::move(connectorId));
    // Keep the base tableScanBuilder_ in sync so endTableScan() delegates
    // through the right object.
    tableScanBuilder_ = icebergTableScanBuilder_;
    return *icebergTableScanBuilder_;
  }

 private:
  std::shared_ptr<IcebergTableScanBuilder> icebergTableScanBuilder_;
};

} // namespace facebook::velox::connector::hive::iceberg::test
