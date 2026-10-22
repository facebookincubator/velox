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
#include "velox/connectors/hive/paimon/PaimonSchema.h"

namespace facebook::velox::connector::hive::paimon {

/// Table definition and scan conditions for the existing Paimon file scan.
/// The fixed target schema owns field identities, key declarations and
/// schema-local options. Column handles retain FileColumnHandle semantics;
/// this handle does not bind query columns or physical files by field ID.
class PaimonTableHandle final : public FileTableHandle {
 public:
  static constexpr int32_t kSerdeVersion = 1;

  /// The caller guarantees every legacy file has the fixed target schema and
  /// ordinary append layout and belongs to the supplied batch snapshot.
  enum class ReadProfile { kSameSchemaAppendBatch };

  /// Stores the complete logical conditions separately from schema-local
  /// options. Execution requires explicit type=table, batch-scan-mode=none,
  /// deletion-vectors.enabled=false, data-evolution.enabled=false,
  /// data-file.thin-mode=false and boolean row-tracking.enabled options.
  /// Optional bucket and file.format are validated; other schema options and
  /// nonempty queryOptions are rejected by this profile. No defaults are added.
  /// These execution limits describe the current reader, not Paimon table
  /// validity; additional capabilities require their own semantic validation.
  PaimonTableHandle(
      std::string connectorId,
      std::string tableName,
      PaimonSchemaPtr schema,
      ReadProfile readProfile,
      common::SubfieldFilters subfieldFilters = {},
      core::TypedExprPtr remainingFilter = nullptr,
      std::vector<FileColumnHandlePtr> filterColumnHandles = {},
      std::unordered_map<std::string, std::string> queryOptions = {},
      double sampleRate = 1.0,
      std::string dbName = "");

  const std::string& name() const override {
    return tableName_;
  }

  const std::string& dbName() const override {
    return dbName_;
  }

  const PaimonSchemaPtr& schema() const {
    return schema_;
  }

  /// Derived from the schema's primary key declaration.
  PaimonTableType tableType() const;

  /// Compares table identity and the complete fixed schema, not scan
  /// conditions.
  bool hasSameTableDefinition(const PaimonTableHandle& other) const;

  /// Compares logical filters, sampling, profile and query options. Column
  /// assignments are checked by the data source.
  bool hasSameScanConditions(const PaimonTableHandle& other) const;

  const common::SubfieldFilters& subfieldFilters() const override {
    return subfieldFilters_;
  }

  const core::TypedExprPtr& remainingFilter() const override {
    return remainingFilter_;
  }

  double sampleRate() const override {
    return sampleRate_;
  }

  /// Schema columns excluding declared partition fields, in schema order.
  const RowTypePtr& dataColumns() const override {
    return dataColumns_;
  }

  const std::unordered_map<std::string, std::string>& tableParameters()
      const override {
    return queryOptions_;
  }

  std::vector<FileColumnHandlePtr> filterColumnHandles() const override {
    return filterColumnHandles_;
  }

  /// Checks connector, table type and the complete set of partition names.
  /// Partition values retain the existing string/NULL encoding. The current
  /// split has no file schema IDs, so this cannot verify file schema identity.
  /// Requires a table definition accepted by validateReadProfile().
  void validateSplit(const PaimonConnectorSplit& split) const;

  /// Checks whether the table definition fits the same-schema append read
  /// profile.
  void validateReadProfile() const;

  /// Checks a column against the fixed schema and partition declarations.
  void validateColumn(const FileColumnHandle& column) const;

  /// Checks the decoded footer's ordered names and types before adaptation.
  /// Physical partition columns may all be present or all omitted; optional
  /// trailing tracking columns are accepted only when row tracking is enabled.
  /// Requires a table definition accepted by validateReadProfile().
  void validateFileSchema(const RowTypePtr& fileType) const;

  /// Rejects filter kinds incompatible with the validated output channel type.
  void validateDynamicFilter(const TypePtr& type, const common::Filter& filter)
      const;

  std::string toString() const override;
  folly::dynamic serialize() const override;
  static ConnectorTableHandlePtr create(
      const folly::dynamic& obj,
      void* context);
  static void registerSerDe();

 private:
  const std::string tableName_;
  const std::string dbName_;
  const PaimonSchemaPtr schema_;
  const ReadProfile readProfile_;
  const common::SubfieldFilters subfieldFilters_;
  const core::TypedExprPtr remainingFilter_;
  const std::vector<FileColumnHandlePtr> filterColumnHandles_;
  const std::unordered_map<std::string, std::string> queryOptions_;
  const double sampleRate_;
  const RowTypePtr dataColumns_;
};

} // namespace facebook::velox::connector::hive::paimon
