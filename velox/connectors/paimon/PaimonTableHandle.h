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

#include "velox/connectors/common/FileTableHandle.h"
#include "velox/connectors/paimon/PaimonConnectorSplit.h"

namespace facebook::velox::connector::paimon {

/// Table handle for Paimon scans. Stores the fixed table definition and scan
/// conditions, including a schema version ID and stable field IDs aligned with
/// the top-level columns of its Velox RowType.
class PaimonTableHandle final : public connector::FileTableHandle {
 public:
  static constexpr int32_t kSerdeVersion = 1;

  /// Caller-resolved historical table fields, in their declared order,
  /// including partition columns. Only top-level identities are represented.
  /// This order aligns names and field IDs, not physical file columns.
  /// Partition values come from the split, not their physical copies.
  struct TableSchema {
    RowTypePtr rowType;
    std::vector<int32_t> fieldIds;
  };
  using TableSchemas = std::unordered_map<int64_t, TableSchema>;

  /// rowType includes partition columns; fieldIds follow its column order.
  /// options are resolved by the caller.
  PaimonTableHandle(
      std::string connectorId,
      std::string tableName,
      int64_t schemaId,
      RowTypePtr rowType,
      std::vector<int32_t> fieldIds,
      PaimonTableType tableType,
      std::vector<std::string> partitionKeys = {},
      std::unordered_map<std::string, std::string> options = {},
      common::SubfieldFilters subfieldFilters = {},
      core::TypedExprPtr remainingFilter = nullptr,
      std::vector<connector::FileColumnHandlePtr> filterColumnHandles = {},
      double sampleRate = 1.0,
      std::string dbName = "",
      TableSchemas tableSchemas = {});

  /// Identifier of the fixed table schema, distinct from a snapshot ID.
  int64_t schemaId() const {
    return schemaId_;
  }

  /// Complete logical table schema, including partition columns.
  const RowTypePtr& rowType() const {
    return rowType_;
  }

  /// Stable Paimon field IDs aligned with rowType()'s top-level columns.
  /// IDs identify fields independently of their names and positions.
  const std::vector<int32_t>& fieldIds() const {
    return fieldIds_;
  }

  /// Historical definitions required by the files, excluding schemaId(),
  /// whose definition is already held in rowType() and fieldIds().
  const TableSchemas& tableSchemas() const {
    return tableSchemas_;
  }

  PaimonTableType tableType() const {
    return tableType_;
  }

  /// Ordered partition column names for the complete table.
  const std::vector<std::string>& partitionKeys() const {
    return partitionKeys_;
  }

  /// Native table options resolved by the caller, not file-reader parameters.
  const std::unordered_map<std::string, std::string>& options() const {
    return options_;
  }

  std::vector<connector::FileColumnHandlePtr> filterColumnHandles()
      const override {
    return filterColumnHandles_;
  }

  std::string toString() const override;
  folly::dynamic serialize() const override;
  static ConnectorTableHandlePtr create(
      const folly::dynamic& obj,
      void* context);
  static void registerSerDe();

 private:
  const int64_t schemaId_;
  const RowTypePtr rowType_;
  const std::vector<int32_t> fieldIds_;
  const TableSchemas tableSchemas_;
  const PaimonTableType tableType_;
  const std::vector<std::string> partitionKeys_;
  const std::unordered_map<std::string, std::string> options_;
  const std::vector<connector::FileColumnHandlePtr> filterColumnHandles_;
};

} // namespace facebook::velox::connector::paimon
