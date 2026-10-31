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

#include "velox/connectors/common/FileColumnHandle.h"

namespace facebook::velox::connector::paimon {

/// A query column identified by its stable Paimon field ID. The name and schema
/// type describe that field in the scan's fixed table schema, not an output
/// alias or a historical file schema. Subfield paths stay relative to this
/// schema; nested field-ID mapping is not represented here.
class PaimonColumnHandle final : public file::FileColumnHandle {
 public:
  static constexpr int32_t kSerdeVersion = 1;

  PaimonColumnHandle(
      std::string name,
      int32_t fieldId,
      ColumnType columnType,
      TypePtr schemaType,
      std::vector<common::Subfield> requiredSubfields = {},
      std::vector<file::NamedExtraction> extractions = {},
      bool partitionDateValueDaysSinceEpoch = false,
      std::function<void(VectorPtr&)> postProcessor = {});

  int32_t fieldId() const {
    return fieldId_;
  }

  std::string toString() const override;
  folly::dynamic serialize() const override;
  static ColumnHandlePtr create(const folly::dynamic& obj);
  static void registerSerDe();

 private:
  const int32_t fieldId_;
};

using PaimonColumnHandlePtr = std::shared_ptr<const PaimonColumnHandle>;

} // namespace facebook::velox::connector::paimon
