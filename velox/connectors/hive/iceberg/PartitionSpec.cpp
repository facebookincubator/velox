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

#include "velox/connectors/hive/iceberg/PartitionSpec.h"

#include <unordered_map>

#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

enum class TransformCategory {
  kIdentity,
  kTemporal, // Year, Month, Day, Hour.
  kBucket,
  kTruncate,
};

TransformCategory getTransformCategory(TransformType transformType) {
  switch (transformType) {
    case TransformType::kIdentity:
      return TransformCategory::kIdentity;
    case TransformType::kYear:
    case TransformType::kMonth:
    case TransformType::kDay:
    case TransformType::kHour:
      return TransformCategory::kTemporal;
    case TransformType::kBucket:
      return TransformCategory::kBucket;
    case TransformType::kTruncate:
      return TransformCategory::kTruncate;
    default:
      VELOX_UNREACHABLE("Unknown transform type");
  }
}

bool isValidPartitionType(const TypePtr& type) {
  return !(
      type->isRow() || type->isArray() || type->isMap() || type->isDouble() ||
      type->isReal() || isTimestampWithTimeZoneType(type));
}

bool canTransform(TransformType transformType, const TypePtr& type) {
  switch (transformType) {
    case TransformType::kIdentity:
      return type->isTinyint() || type->isSmallint() || type->isInteger() ||
          type->isBigint() || type->isBoolean() || type->isDecimal() ||
          type->isDate() || type->isTimestamp() || type->isVarchar() ||
          type->isVarbinary();
    case TransformType::kYear:
    case TransformType::kMonth:
    case TransformType::kDay:
      return type->isDate() || type->isTimestamp();
    case TransformType::kHour:
      return type->isTimestamp();
    case TransformType::kBucket:
      return type->isInteger() || type->isBigint() || type->isDecimal() ||
          type->isVarchar() || type->isVarbinary() || type->isDate() ||
          type->isTimestamp();
    case TransformType::kTruncate:
      return type->isInteger() || type->isBigint() || type->isDecimal() ||
          type->isVarchar() || type->isVarbinary();
    default:
      VELOX_UNREACHABLE("Unsupported partition transform type.");
  }
}

const auto& transformTypeNames() {
  static const folly::F14FastMap<TransformType, std::string_view>
      kTransformNames = {
          {TransformType::kIdentity, "identity"},
          {TransformType::kHour, "hour"},
          {TransformType::kDay, "day"},
          {TransformType::kMonth, "month"},
          {TransformType::kYear, "year"},
          {TransformType::kBucket, "bucket"},
          {TransformType::kTruncate, "trunc"},
      };
  return kTransformNames;
}

} // namespace

VELOX_DEFINE_ENUM_NAME(TransformType, transformTypeNames);

void IcebergPartitionSpec::checkCompatibility() const {
  std::unordered_map<std::string, std::unordered_map<TransformCategory, int>>
      columnTransformCategories;

  for (const auto& field : fields) {
    const auto& type = field.type;
    const auto& name = field.name;
    if (!isValidPartitionType(type)) {
      VELOX_USER_FAIL(
          "Type is not supported as a partition column: {}", type->name());
    }

    if (!canTransform(field.transformType, type)) {
      VELOX_USER_FAIL(
          "Transform is not supported for partition field. Field: '{}', Type: '{}', Transform: '{}'.",
          name,
          type->toString(),
          TransformTypeName::toName(field.transformType));
    }

    auto category = getTransformCategory(field.transformType);
    columnTransformCategories[name][category]++;

    // Check if any transform category appears more than once.
    if (columnTransformCategories[name][category] > 1) {
      std::string categoryName;
      switch (category) {
        case TransformCategory::kIdentity:
          categoryName = "Identity";
          break;
        case TransformCategory::kTemporal:
          categoryName = "Temporal (Year/Month/Day/Hour)";
          break;
        case TransformCategory::kBucket:
          categoryName = "Bucket";
          break;
        case TransformCategory::kTruncate:
          categoryName = "Truncate";
          break;
      }
      VELOX_USER_FAIL(
          "Multiple transforms of the same category on a column are not allowed. "
          "Each transform category can appear at most once per column. Column: '{}', Category: {}.",
          name,
          categoryName);
    }
  }
}

} // namespace facebook::velox::connector::hive::iceberg
