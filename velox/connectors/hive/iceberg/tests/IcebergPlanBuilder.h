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

#include <unordered_set>

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
/// Assignments are auto-built as IcebergColumnHandles with sentinel field ID
/// (-1). Field-ID column mapping is only activated when the caller explicitly
/// provides real field IDs via dataColumnFieldIds() or assignments(), so scans
/// that do not require field-ID mapping work without any extra setup.
///
/// filterColumnHandles for filter-only columns (columns referenced in a
/// pushed-down filter but absent from the output projection) are auto-built
/// with sentinel field IDs from dataColumns(). Callers only need to supply
/// them explicitly when non-sentinel field IDs are required on a filter-only
/// column (e.g. explicit field-ID schema-evolution tests).
class IcebergTableScanBuilder
    : public exec::test::PlanBuilder::TableScanBuilder {
 public:
  explicit IcebergTableScanBuilder(exec::test::PlanBuilder& planBuilder)
      : TableScanBuilder(planBuilder) {}

 protected:
  /// Overrides the base factory to produce an IcebergColumnHandle.
  /// Uses the field ID from dataColumnFieldIds_ when set (looked up by column
  /// name in dataColumns_); otherwise uses sentinel -1, which keeps field-ID
  /// column mapping off unless the caller supplies real IDs via assignments().
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

  /// Overrides the base factory to construct an IcebergTableHandle.
  ///
  /// Auto-detects filter-only columns before building the handle: any column
  /// in dataColumns_ that is referenced by a pushed-down subfield filter or
  /// remaining-filter expression but is absent from the output assignments and
  /// not already in filterColumnHandles_ gets an auto-built IcebergColumnHandle
  /// (with sentinel field ID). Caller-supplied filterColumnHandles_ take
  /// precedence and are never replaced.
  connector::ConnectorTableHandlePtr buildConnectorTableHandle(
      common::SubfieldFilters subfieldFilters,
      const core::TypedExprPtr& remainingFilter) override {
    // Collect names of all columns referenced by pushed-down filters.
    std::unordered_set<std::string> filterColNames;
    for (const auto& [subfield, _] : subfieldFilters) {
      filterColNames.insert(subfield.baseName());
    }
    if (remainingFilter) {
      collectFieldNames(remainingFilter, filterColNames);
    }

    // Auto-build filter-only handles for columns not already covered.
    if (!filterColNames.empty() && dataColumns_ != nullptr) {
      std::unordered_set<std::string> covered;
      for (const auto& h : filterColumnHandles_) {
        covered.insert(h->name());
      }
      for (uint32_t i = 0; i < dataColumns_->size(); ++i) {
        const auto& colName = dataColumns_->nameOf(i);
        if (!filterColNames.count(colName)) {
          continue;
        }
        if (assignments_.count(colName) || covered.count(colName)) {
          continue;
        }
        int32_t fieldId = -1;
        if (!dataColumnFieldIds_.empty()) {
          fieldId = dataColumnFieldIds_[i];
        }
        filterColumnHandles_.push_back(
            std::make_shared<IcebergColumnHandle>(
                colName,
                FileColumnHandle::ColumnType::kRegular,
                dataColumns_->childAt(i),
                parquet::ParquetFieldId{fieldId, {}}));
      }
    }

    // Downcast every filterColumnHandle to IcebergColumnHandle.
    std::vector<IcebergColumnHandlePtr> icebergFilterHandles;
    icebergFilterHandles.reserve(filterColumnHandles_.size());
    for (const auto& h : filterColumnHandles_) {
      auto iceberg = std::dynamic_pointer_cast<const IcebergColumnHandle>(h);
      VELOX_CHECK_NOT_NULL(
          iceberg,
          "IcebergTableScanBuilder: filterColumnHandle '{}' is not an "
          "IcebergColumnHandle",
          h->name());
      icebergFilterHandles.push_back(std::move(iceberg));
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

 private:
  /// Recursively collects all field-access (input column) names from 'expr'.
  static void collectFieldNames(
      const core::TypedExprPtr& expr,
      std::unordered_set<std::string>& out) {
    if (!expr) {
      return;
    }
    if (auto* fa =
            dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get())) {
      out.insert(fa->name());
    }
    for (const auto& input : expr->inputs()) {
      collectFieldNames(input, out);
    }
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
      std::string connectorId = kIcebergConnectorId) override {
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
