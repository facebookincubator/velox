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

#include <unordered_map>

#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"

namespace facebook::velox::connector::hive::iceberg {

/// Iceberg-specific table handle that extends HiveTableHandle.
///
/// Carries all fields inherited from HiveTableHandle (subfield filters,
/// remaining filter, data columns, table parameters, etc.) plus
/// Iceberg-specific fields:
///  - isChangelogQuery_: whether this scan is a changelog (CDC) query.
///  - dataColumnHandles_: column handles keyed by column name, used for
///    Iceberg-specific column metadata (field IDs, default values, etc.).
class IcebergTableHandle : public HiveTableHandle {
 public:
  IcebergTableHandle(
      std::string connectorId,
      const std::string& tableName,
      common::SubfieldFilters subfieldFilters,
      const core::TypedExprPtr& remainingFilter,
      const RowTypePtr& dataColumns = nullptr,
      std::vector<std::string> indexColumns = {},
      const std::unordered_map<std::string, std::string>& tableParameters = {},
      std::vector<IcebergColumnHandlePtr> filterColumnHandles = {},
      double sampleRate = 1.0,
      std::string dbName = "",
      bool isChangelogQuery = false,
      std::unordered_map<std::string, IcebergColumnHandlePtr>
          dataColumnHandles = {});

  /// Whether this scan is a changelog (CDC) query over an Iceberg table.
  bool isChangelogQuery() const {
    return isChangelogQuery_;
  }

  /// Column handles keyed by column name, carrying Iceberg-specific metadata
  /// such as field IDs and initial-default values.
  const std::unordered_map<std::string, IcebergColumnHandlePtr>&
  dataColumnHandles() const {
    return dataColumnHandles_;
  }

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static ConnectorTableHandlePtr create(
      const folly::dynamic& obj,
      void* context);

  static void registerSerDe();

 private:
  const bool isChangelogQuery_;
  const std::unordered_map<std::string, IcebergColumnHandlePtr>
      dataColumnHandles_;
};

using IcebergTableHandlePtr = std::shared_ptr<const IcebergTableHandle>;

} // namespace facebook::velox::connector::hive::iceberg
