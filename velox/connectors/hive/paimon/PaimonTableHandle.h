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

/// A normalized historical schema supplied by the planner. Field IDs are
/// Paimon user-field IDs, not DWIO structural IDs or physical column ordinals.
/// The first profile supports top-level scalar fields only.
struct PaimonSchema {
  int64_t id;
  RowTypePtr rowType;
  std::vector<int32_t> fieldIds;

  void validate() const;
  folly::dynamic serialize() const;
  static PaimonSchema create(const folly::dynamic& obj);
};

class PaimonColumnHandle final : public FileColumnHandle {
 public:
  PaimonColumnHandle(
      std::string name,
      int32_t fieldId,
      TypePtr type,
      ColumnType role = ColumnType::kRegular,
      std::vector<common::Subfield> requiredSubfields = {});

  const std::string& name() const override {
    return name_;
  }
  int32_t fieldId() const {
    return fieldId_;
  }
  ColumnType columnType() const override {
    return role_;
  }
  const TypePtr& schemaType() const override {
    return type_;
  }
  const std::vector<common::Subfield>& requiredSubfields() const override {
    return requiredSubfields_;
  }
  bool isPartitionDateValueDaysSinceEpoch() const override {
    return false;
  }
  const std::function<void(VectorPtr&)>& postProcessor() const override {
    static const std::function<void(VectorPtr&)> kNone;
    return kNone;
  }

  folly::dynamic serialize() const override;
  static ColumnHandlePtr create(const folly::dynamic& obj);
  static void registerSerDe();

 private:
  const std::string name_;
  const int32_t fieldId_;
  const TypePtr type_;
  const ColumnType role_;
  const std::vector<common::Subfield> requiredSubfields_;
};

/// Query-fixed execution semantics. The caller normalizes historical schema
/// JSON defaults and supplies every schema referenced by its complete splits.
/// This is not the Paimon schema JSON format; wireVersion is independent of
/// schemaId. Options contain only normalized execution options, not arbitrary
/// catalog or writer properties. Unknown options fail capability validation.
class PaimonTableHandle final : public FileTableHandle {
 public:
  PaimonTableHandle(
      std::string connectorId,
      std::string tableName,
      PaimonTableType tableType,
      int64_t targetSchemaId,
      std::vector<PaimonSchema> schemas,
      std::vector<int32_t> partitionFieldIds = {},
      std::vector<int32_t> primaryKeyFieldIds = {},
      common::SubfieldFilters filters = {},
      core::TypedExprPtr remainingFilter = nullptr,
      std::unordered_map<std::string, std::string> options = {});

  const std::string& name() const override {
    return tableName_;
  }
  const std::string& dbName() const override {
    static const std::string kEmpty;
    return kEmpty;
  }
  PaimonTableType tableType() const {
    return tableType_;
  }
  const PaimonSchema& targetSchema() const {
    return schema(targetSchemaId_);
  }
  const PaimonSchema& schema(int64_t id) const;
  const std::vector<int32_t>& partitionFieldIds() const {
    return partitionFieldIds_;
  }
  const std::vector<int32_t>& primaryKeyFieldIds() const {
    return primaryKeyFieldIds_;
  }
  const std::unordered_map<std::string, std::string>& options() const {
    return options_;
  }

  const common::SubfieldFilters& subfieldFilters() const override {
    return filters_;
  }
  const core::TypedExprPtr& remainingFilter() const override {
    return remainingFilter_;
  }
  double sampleRate() const override {
    return 1;
  }
  const RowTypePtr& dataColumns() const override {
    return targetSchema().rowType;
  }
  const std::unordered_map<std::string, std::string>& tableParameters()
      const override {
    static const std::unordered_map<std::string, std::string> kEmpty;
    return kEmpty;
  }
  std::vector<FileColumnHandlePtr> filterColumnHandles() const override;

  folly::dynamic serialize() const override;
  static ConnectorTableHandlePtr create(
      const folly::dynamic& obj,
      void* context);
  static void registerSerDe();

 private:
  const std::string tableName_;
  const PaimonTableType tableType_;
  const int64_t targetSchemaId_;
  const std::vector<PaimonSchema> schemas_;
  const std::vector<int32_t> partitionFieldIds_;
  const std::vector<int32_t> primaryKeyFieldIds_;
  const common::SubfieldFilters filters_;
  const core::TypedExprPtr remainingFilter_;
  const std::unordered_map<std::string, std::string> options_;
};

} // namespace facebook::velox::connector::hive::paimon
