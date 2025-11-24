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

#include "velox/connectors/hive/TableHandle.h"

namespace facebook::velox::connector::hive::iceberg {

/// Represents the nested field structure in Iceberg schema.
struct IcebergField {
  /// The field id from Iceberg schema.
  int32_t id;

  /// Child fields of nested types (e.g., struct fields, list elements, map
  /// key/value). Empty for primitive types.
  std::vector<IcebergField> children;
};

class IcebergColumnHandle : public HiveColumnHandle {
 public:
  IcebergColumnHandle(
      const std::string& name,
      ColumnType columnType,
      TypePtr dataType,
      TypePtr hiveType,
      const IcebergField& nestedField,
      std::vector<common::Subfield> requiredSubfields = {});

  const IcebergField& field() const;

 private:
  const IcebergField field_;
};

using IcebergColumnHandlePtr = std::shared_ptr<const IcebergColumnHandle>;

} // namespace facebook::velox::connector::hive::iceberg
