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
#include "velox/connectors/common/FileColumnHandle.h"
#include "velox/connectors/common/FileTableHandle.h"
#include "velox/core/ITypedExpr.h"
#include "velox/type/Filter.h"
#include "velox/type/Subfield.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive {

class HiveColumnHandle : public file::FileColumnHandle {
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
      std::vector<file::NamedExtraction> extractions = {},
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

  /// The type of this column as stored in the Hive source table. May differ
  /// from dataType() when type coercion is needed for schema evolution.
  const TypePtr& hiveType() const {
    return schemaType();
  }

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static ColumnHandlePtr create(const folly::dynamic& obj);

  static void registerSerDe();

 protected:
  // Emits the common column fields (columnType, dataType, requiredSubfields,
  // extractions) without a class-name prefix. Subclasses use this to build
  // their own toString() with the correct class name in the header.
  std::string toStringFields() const;
};

using HiveColumnHandlePtr = std::shared_ptr<const HiveColumnHandle>;
using HiveColumnHandleMap =
    std::unordered_map<std::string, HiveColumnHandlePtr>;

class HiveTableHandle : public file::FileTableHandle {
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
      std::string dbName = "",
      std::vector<int32_t> dataColumnFieldIds = {});

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

  bool supportsIndexLookup() const override {
    return !indexColumns_.empty();
  }

  bool needsIndexSplit() const override {
    return true;
  }

  /// Returns Iceberg field IDs aligned positionally to dataColumns(). An empty
  /// vector means field IDs are unavailable.
  const std::vector<int32_t>& dataColumnFieldIds() const {
    return dataColumnFieldIds_;
  }

  /// Returns the names of the index columns for the table.
  const std::vector<std::string>& indexColumns() const {
    return indexColumns_;
  }

  /// Return filter column handles as FileColumnHandlePtr for the generic scan
  /// pipeline.
  std::vector<file::FileColumnHandlePtr> filterColumnHandles() const override {
    return {filterColumnHandles_.begin(), filterColumnHandles_.end()};
  }

  /// Return filter column handles with their concrete Hive type.
  const std::vector<HiveColumnHandlePtr>& hiveFilterColumnHandles() const {
    return filterColumnHandles_;
  }

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static ConnectorTableHandlePtr create(
      const folly::dynamic& obj,
      void* context);

  static void registerSerDe();

 protected:
  // Serializes the Hive-common fields (tableName, subfieldFilters,
  // remainingFilter, sampleRate, dataColumns, tableParameters,
  // filterColumnHandles, indexColumns, dbName) into an object whose "name"
  // key is set to @p typeName. Subclasses call this instead of duplicating
  // the field list, then append their own keys.
  folly::dynamic serializeHiveFields(std::string_view typeName) const;

  // Fills the common Hive fields from @p obj into the provided out-parameters.
  // Subclasses call this from their own create() and then parse only their
  // extra keys on top.
  static void deserializeHiveFields(
      const folly::dynamic& obj,
      void* context,
      std::string& connectorId,
      std::string& tableName,
      common::SubfieldFilters& subfieldFilters,
      core::TypedExprPtr& remainingFilter,
      double& sampleRate,
      RowTypePtr& dataColumns,
      std::unordered_map<std::string, std::string>& tableParameters,
      std::vector<HiveColumnHandlePtr>& filterColumnHandles,
      std::vector<std::string>& indexColumns,
      std::string& dbName,
      std::vector<int32_t>& dataColumnFieldIds);

 private:
  const std::vector<int32_t> dataColumnFieldIds_;
  const std::vector<std::string> indexColumns_;
  const std::vector<HiveColumnHandlePtr> filterColumnHandles_;
};

using HiveTableHandlePtr = std::shared_ptr<const HiveTableHandle>;

} // namespace facebook::velox::connector::hive

// The fmt::formatter for FileColumnHandle::ColumnType is defined in
// FileColumnHandle.h.
