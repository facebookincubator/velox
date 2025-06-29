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

#include "velox/connectors/hive/iceberg/ColumnTransform.h"

namespace facebook::velox::connector::hive::iceberg {

struct IcebergPartitionSpec {
  enum class TransformType {
    kIdentity,
    kYear,
    kMonth,
    kDay,
    kHour,
    kBucket,
    kTruncate
  };

  struct Schema {
    std::unordered_map<std::string, int32_t> columnNameToIdMapping;
    std::unordered_map<std::string, TypePtr> columnNameToTypeMapping;

    Schema(
        const std::unordered_map<std::string, int32_t>& _columnNameToIdMapping,
        const std::unordered_map<std::string, TypePtr>&
            _columnNameToTypeMapping)
        : columnNameToIdMapping(_columnNameToIdMapping),
          columnNameToTypeMapping(_columnNameToTypeMapping) {}
  };

  struct Field {
    // The column name of this partition field as it appears in the partition
    // spec.
    std::string name;

    // The transform type applied to the source field (e.g., kIdentity, kBucket,
    // kTruncate, etc.).
    TransformType transform;

    // Optional parameter for transforms that require configuration
    // (e.g., bucket count or truncate width).
    std::optional<int32_t> parameter;

    Field(
        const std::string& _name,
        TransformType _transform,
        std::optional<int32_t> _parameter)
        : name(_name), transform(_transform), parameter(_parameter) {}
  };

  const int32_t specId;
  const std::shared_ptr<const Schema> schema;
  std::shared_ptr<ColumnTransforms> columnTransforms;
  IcebergPartitionSpec(
      int32_t specId,
      const std::shared_ptr<const Schema>& schema,
      const std::vector<Field>& fields,
      memory::MemoryPool* pool);

  std::vector<ColumnTransform> getColumnTransforms() const;
};

} // namespace facebook::velox::connector::hive::iceberg
