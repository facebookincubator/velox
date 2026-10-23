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

#include "velox/connectors/hive/FileTableHandle.h"
#include "velox/connectors/hive/paimon/PaimonConnectorSplit.h"

namespace facebook::velox::connector::hive::paimon {

/// Fixed table definition and scan conditions for a Paimon file scan.
/// Uses Velox RowType for names and types, with stable field IDs aligned to
/// its top-level columns, and exposes scan inputs through FileTableHandle.
class PaimonTableHandle final : public FileTableHandle {
 public:
  static constexpr int32_t kSerdeVersion = 1;

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
      std::vector<FileColumnHandlePtr> filterColumnHandles = {},
      double sampleRate = 1.0,
      std::string dbName = "");

  const std::string& name() const override {
    return tableName_;
  }

  const std::string& dbName() const override {
    return dbName_;
  }

  int64_t schemaId() const {
    return schemaId_;
  }

  const RowTypePtr& rowType() const {
    return rowType_;
  }

  const std::vector<int32_t>& fieldIds() const {
    return fieldIds_;
  }

  PaimonTableType tableType() const {
    return tableType_;
  }

  /// Ordered partition column names for the complete table.
  const std::vector<std::string>& partitionKeys() const {
    return partitionKeys_;
  }

  const std::unordered_map<std::string, std::string>& options() const {
    return options_;
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

  /// Non-partition table columns in schema order.
  const RowTypePtr& dataColumns() const override {
    return dataColumns_;
  }

  const std::unordered_map<std::string, std::string>& tableParameters()
      const override;

  std::vector<FileColumnHandlePtr> filterColumnHandles() const override {
    return filterColumnHandles_;
  }

  std::string toString() const override;
  folly::dynamic serialize() const override;
  static ConnectorTableHandlePtr create(
      const folly::dynamic& obj,
      void* context);
  static void registerSerDe();

 private:
  const std::string tableName_;
  const std::string dbName_;
  const int64_t schemaId_;
  const RowTypePtr rowType_;
  const std::vector<int32_t> fieldIds_;
  const PaimonTableType tableType_;
  const std::vector<std::string> partitionKeys_;
  const std::unordered_map<std::string, std::string> options_;
  const common::SubfieldFilters subfieldFilters_;
  const core::TypedExprPtr remainingFilter_;
  const std::vector<FileColumnHandlePtr> filterColumnHandles_;
  const double sampleRate_;
  const RowTypePtr dataColumns_;
};

} // namespace facebook::velox::connector::hive::paimon
