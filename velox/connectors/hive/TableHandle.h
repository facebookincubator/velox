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
#include "velox/connectors/hive/FileColumnHandle.h"
#include "velox/connectors/hive/FileTableHandle.h"
#include "velox/core/ITypedExpr.h"
#include "velox/type/Filter.h"
#include "velox/type/Subfield.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive {

class HiveColumnHandle : public FileColumnHandle {
 public:
  struct ColumnParseParameters {
    enum PartitionDateValueFormat {
      kISO8601,
      kDaysSinceEpoch,
    } partitionDateValueFormat;
  };

  /// NOTE: 'dataType' is the column type in target write table.  'hiveType' is
  /// converted type of the corresponding column in source table which might not
  /// be the same type, and the table scan needs to do data coercion if needs.
  /// The table writer also needs to respect the type difference when processing
  /// input data such as bucket id calculation.
  ///
  /// 'extractions' specifies named extraction chains.  When non-empty,
  /// 'requiredSubfields' must be empty (mutually exclusive).  When a single
  /// extraction is present, 'dataType' is that extraction's dataType.  When
  /// multiple extractions are present, 'dataType' is a ROW type whose fields
  /// are the outputNames with their corresponding dataTypes.
  HiveColumnHandle(
      const std::string& name,
      ColumnType columnType,
      TypePtr dataType,
      TypePtr hiveType,
      std::vector<common::Subfield> requiredSubfields = {},
      std::vector<NamedExtraction> extractions = {},
      ColumnParseParameters columnParseParameters = {},
      std::function<void(VectorPtr&)> postProcessor = {});

  /// Legacy constructor without extractions for backward compatibility.
  HiveColumnHandle(
      const std::string& name,
      ColumnType columnType,
      TypePtr dataType,
      TypePtr hiveType,
      std::vector<common::Subfield> requiredSubfields,
      ColumnParseParameters columnParseParameters,
      std::function<void(VectorPtr&)> postProcessor = {})
      : HiveColumnHandle(
            name,
            columnType,
            std::move(dataType),
            std::move(hiveType),
            std::move(requiredSubfields),
            /*extractions=*/{},
            columnParseParameters,
            std::move(postProcessor)) {}

  const std::string& name() const override {
    return name_;
  }

  ColumnType columnType() const override {
    return columnType_;
  }

  const TypePtr& dataType() const override {
    return dataType_;
  }

  /// The type of this column as stored in the Hive source table. May differ
  /// from dataType() when type coercion is needed for schema evolution.
  const TypePtr& hiveType() const {
    return hiveType_;
  }

  /// Applies to columns of complex types: arrays, maps and structs.  When a
  /// query uses only some of the subfields, the engine provides the complete
  /// list of required subfields and the connector is free to prune the rest.
  ///
  /// Examples:
  ///  - SELECT a[1], b['x'], x.y FROM t
  ///  - SELECT a FROM t WHERE b['y'] > 10
  ///
  /// Pruning a struct means populating some of the members with null values.
  ///
  /// Pruning a map means dropping keys not listed in the required subfields.
  ///
  /// Pruning arrays means dropping values with indices larger than maximum
  /// required index.
  const std::vector<common::Subfield>& requiredSubfields() const override {
    return requiredSubfields_;
  }

  /// Named extraction chains.  Empty means no extraction (current behavior).
  /// When a single entry is present, the column handle's dataType is that
  /// entry's dataType.  When multiple entries are present, the column
  /// handle's dataType is a ROW type whose fields are the outputNames with
  /// their corresponding dataTypes.
  /// Mutually exclusive with requiredSubfields — if extractions is non-empty,
  /// requiredSubfields must be empty.
  const std::vector<NamedExtraction>& extractions() const override {
    return extractions_;
  }

  bool isPartitionKey() const override {
    return columnType_ == ColumnType::kPartitionKey;
  }

  bool isPartitionDateValueDaysSinceEpoch() const override {
    return columnParseParameters_.partitionDateValueFormat ==
        ColumnParseParameters::kDaysSinceEpoch;
  }

  /// Apply some row-wise post processing to this column when it is present in
  /// output.
  ///
  /// It's not allowed to change the size of the vector in the processor.  The
  /// top level vector is guaranteed to be safe to change.  Any inner vectors
  /// and buffers need to check the reference count before doing any change in
  /// place, otherwise you need to allocate new vectors and buffers.
  ///
  /// For lazy vector, this will be applied after the lazy vector is loaded.
  /// This is only applied after all the filtering is done; the filters (both
  /// subfield filters and remaining filter) still apply to values before post
  /// processing.  ValueHook usage will be disabled if a post processor is
  /// present.
  const std::function<void(VectorPtr&)>& postProcessor() const override {
    return postProcessor_;
  }

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static ColumnHandlePtr create(const folly::dynamic& obj);

  static void registerSerDe();

 private:
  const std::string name_;
  const ColumnType columnType_;
  const TypePtr dataType_;
  const TypePtr hiveType_;
  const std::vector<common::Subfield> requiredSubfields_;
  const std::vector<NamedExtraction> extractions_;
  const ColumnParseParameters columnParseParameters_;
  const std::function<void(VectorPtr&)> postProcessor_;
};

using HiveColumnHandlePtr = std::shared_ptr<const HiveColumnHandle>;
using HiveColumnHandleMap =
    std::unordered_map<std::string, HiveColumnHandlePtr>;

class HiveTableHandle : public FileTableHandle {
 public:
  /// @param sampleRate Sampling rate in (0, 1] range. 0.1 means 10% sampling.
  /// 1.0 means no sampling. Default is no sampling.
  HiveTableHandle(
      std::string connectorId,
      const std::string& tableName,
      common::SubfieldFilters subfieldFilters,
      const core::TypedExprPtr& remainingFilter,
      const RowTypePtr& dataColumns = nullptr,
      std::vector<std::string> indexColumns = {},
      const std::unordered_map<std::string, std::string>& tableParameters = {},
      std::vector<HiveColumnHandlePtr> filterColumnHandles = {},
      double sampleRate = 1.0,
      std::string dbName = "");

  /// Legacy constructor without indexColumns parameter for backward
  /// compatibility.
  HiveTableHandle(
      std::string connectorId,
      const std::string& tableName,
      common::SubfieldFilters subfieldFilters,
      const core::TypedExprPtr& remainingFilter,
      const RowTypePtr& dataColumns,
      const std::unordered_map<std::string, std::string>& tableParameters,
      std::vector<HiveColumnHandlePtr> filterColumnHandles,
      double sampleRate = 1.0);

  const std::string& tableName() const {
    return tableName_;
  }

  const std::string& name() const override {
    return tableName();
  }

  bool supportsIndexLookup() const override {
    return !indexColumns_.empty();
  }

  bool needsIndexSplit() const override {
    return true;
  }

  const common::SubfieldFilters& subfieldFilters() const override {
    return subfieldFilters_;
  }

  const core::TypedExprPtr& remainingFilter() const override {
    return remainingFilter_;
  }

  double sampleRate() const override {
    return sampleRate_;
  }

  const RowTypePtr& dataColumns() const override {
    return dataColumns_;
  }

  /// Returns the names of the index columns for the table.
  const std::vector<std::string>& indexColumns() const {
    return indexColumns_;
  }

  const std::unordered_map<std::string, std::string>& tableParameters()
      const override {
    return tableParameters_;
  }

  /// Return filter column handles as FileColumnHandlePtr for the generic scan
  /// pipeline.
  std::vector<FileColumnHandlePtr> filterColumnHandles() const override {
    return {filterColumnHandles_.begin(), filterColumnHandles_.end()};
  }

  /// Return filter column handles with their concrete Hive type.
  const std::vector<HiveColumnHandlePtr>& hiveFilterColumnHandles() const {
    return filterColumnHandles_;
  }

  const std::string& dbName() const override {
    return dbName_;
  }

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static ConnectorTableHandlePtr create(
      const folly::dynamic& obj,
      void* context);

  static void registerSerDe();

 private:
  const std::string tableName_;
  const common::SubfieldFilters subfieldFilters_;
  const core::TypedExprPtr remainingFilter_;
  const double sampleRate_;
  const RowTypePtr dataColumns_;
  const std::vector<std::string> indexColumns_;
  const std::unordered_map<std::string, std::string> tableParameters_;
  const std::vector<HiveColumnHandlePtr> filterColumnHandles_;
  const std::string dbName_;
};

using HiveTableHandlePtr = std::shared_ptr<const HiveTableHandle>;

} // namespace facebook::velox::connector::hive

// The fmt::formatter for FileColumnHandle::ColumnType is defined in
// FileColumnHandle.h.
