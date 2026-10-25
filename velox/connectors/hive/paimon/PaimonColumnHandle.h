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

#include "velox/connectors/hive/FileColumnHandle.h"

namespace facebook::velox::connector::hive::paimon {

/// A query column identified by its stable Paimon field ID. The name and schema
/// type describe that field in the scan's fixed table schema, not an output
/// alias or a historical file schema. Subfield paths stay relative to this
/// schema; nested field-ID mapping is not represented here.
class PaimonColumnHandle final : public FileColumnHandle {
 public:
  static constexpr int32_t kSerdeVersion = 1;

  PaimonColumnHandle(
      std::string name,
      int32_t fieldId,
      ColumnType columnType,
      TypePtr schemaType,
      std::vector<common::Subfield> requiredSubfields = {},
      std::vector<NamedExtraction> extractions = {},
      bool partitionDateValueDaysSinceEpoch = false,
      std::function<void(VectorPtr&)> postProcessor = {});

  const std::string& name() const override {
    return name_;
  }

  int32_t fieldId() const {
    return fieldId_;
  }

  ColumnType columnType() const override {
    return columnType_;
  }

  const TypePtr& schemaType() const override {
    return schemaType_;
  }

  const TypePtr& dataType() const override {
    return dataType_;
  }

  const std::vector<common::Subfield>& requiredSubfields() const override {
    return requiredSubfields_;
  }

  const std::vector<NamedExtraction>& extractions() const override {
    return extractions_;
  }

  bool isPartitionDateValueDaysSinceEpoch() const override {
    return partitionDateValueDaysSinceEpoch_;
  }

  const std::function<void(VectorPtr&)>& postProcessor() const override {
    return postProcessor_;
  }

  std::string toString() const override;
  folly::dynamic serialize() const override;
  static ColumnHandlePtr create(const folly::dynamic& obj);
  static void registerSerDe();

 private:
  const std::string name_;
  const int32_t fieldId_;
  const ColumnType columnType_;
  const TypePtr schemaType_;
  const std::vector<common::Subfield> requiredSubfields_;
  const std::vector<NamedExtraction> extractions_;
  const bool partitionDateValueDaysSinceEpoch_;
  const std::function<void(VectorPtr&)> postProcessor_;
  const TypePtr dataType_;
};

using PaimonColumnHandlePtr = std::shared_ptr<const PaimonColumnHandle>;

} // namespace facebook::velox::connector::hive::paimon
