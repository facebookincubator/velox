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

#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg::test {

/// Default connector ID used by IcebergPlanBuilder and IcebergTestBase.
inline const std::string kIcebergConnectorId{"test-iceberg"};

/// A TableScanBuilder subclass that constructs Iceberg-specific handles by
/// overriding the two connector factory methods defined in TableScanBuilder:
///
///   buildConnectorColumnHandle() — produces an IcebergColumnHandle for each
///     output column when no explicit assignments are provided. The Iceberg
///     field ID is resolved from dataColumnFieldIds_ when available; otherwise
///     a sentinel value of -1 is used, keeping field-ID column mapping off
///     unless the caller supplies real IDs via dataColumnFieldIds() or
///     assignments().
///
///   buildConnectorTableHandle() — produces an IcebergTableHandle after all
///     filter-string parsing (subfield filters, remaining filter) has been
///     performed by the base class build(). Filter-only columns (referenced
///     in a pushed-down filter but absent from the output projection) are
///     auto-built as IcebergColumnHandles with sentinel field IDs from
///     dataColumns(); caller-supplied filterColumnHandles_ take precedence
///     and are never replaced.
///
/// All shared logic in build() (filter parsing, alias resolution,
/// filtersAsNode handling) is inherited without duplication.
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
  connector::ColumnHandlePtr buildConnectorColumnHandle(
      const std::string& name,
      const TypePtr& type,
      uint32_t outputIndex) override;

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
      const core::TypedExprPtr& remainingFilter) override;
};

/// A PlanBuilder subclass whose startTableScan() returns an
/// IcebergTableScanBuilder so every scan node is backed by an
/// IcebergTableHandle. All fluent builder methods (outputType, dataColumns,
/// subfieldFilters, remainingFilter, assignments, filterColumnHandles,
/// dataColumnFieldIds, …) are inherited unchanged.
class IcebergPlanBuilder : public exec::test::PlanBuilder {
 public:
  using PlanBuilder::PlanBuilder;

  /// Starts a scan whose table and column handles are Iceberg handles.
  IcebergTableScanBuilder& startTableScan(
      std::string connectorId = kIcebergConnectorId) override;

 private:
  // Typed alias of the base tableScanBuilder_, which points to the same
  // object, so that startTableScan() can return the Iceberg builder.
  std::shared_ptr<IcebergTableScanBuilder> icebergTableScanBuilder_;
};

} // namespace facebook::velox::connector::hive::iceberg::test
