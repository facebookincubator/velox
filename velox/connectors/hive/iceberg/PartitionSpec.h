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

#include <optional>
#include <string>
#include <vector>

namespace facebook::velox::connector::hive::iceberg {

enum class TransformType {
  kIdentity,
  kHour,
  kDay,
  kMonth,
  kYear,
  kBucket,
  kTruncate
};

struct IcebergPartitionSpec {
  struct Field {
    // The column name and type of this partition field as it appears in the
    // partition spec. The column can be a nested column in struct field.
    std::string name;

    // The transform type applied to the source field (e.g., kIdentity, kBucket,
    // kTruncate, etc.).
    TransformType transformType;

    // Optional parameter for transforms that require configuration
    // (e.g., bucket count or truncate width).
    std::optional<int32_t> parameter;

    Field(
        const std::string& _name,
        TransformType _transform,
        std::optional<int32_t> _parameter)
        : name(_name), transformType(_transform), parameter(_parameter) {}
  };

  const int32_t specId;
  const std::vector<Field> fields;

  IcebergPartitionSpec(int32_t _specId, const std::vector<Field>& _fields)
      : specId(_specId), fields(_fields) {}
};

} // namespace facebook::velox::connector::hive::iceberg
