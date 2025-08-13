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

struct IcebergNestedField {
  int32_t id;
  std::vector<IcebergNestedField> children;
};

class IcebergColumnHandle : public HiveColumnHandle {
 public:
  IcebergColumnHandle(
      const std::string& name,
      ColumnType columnType,
      TypePtr dataType,
      TypePtr hiveType,
      const IcebergNestedField& nestedField,
      std::vector<common::Subfield> requiredSubfields = {},
      ColumnParseParameters columnParseParameters = {});

  const IcebergNestedField& nestedField() const;

 private:
  const IcebergNestedField nestedField_;
};

} // namespace facebook::velox::connector::hive::iceberg
