/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

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

const auto& transformCategoryNames() {
  static const folly::F14FastMap<TransformCategory, std::string_view>
      kTransformCategoryNames = {
          {TransformCategory::kIdentity, "Identity"},
          {TransformCategory::kBucket, "Bucket"},
          {TransformCategory::kTruncate, "Truncate"},
          {TransformCategory::kTemporal, "Temporal"},
      };
  return kTransformCategoryNames;
}

} // namespace

VELOX_DEFINE_ENUM_NAME(TransformType, transformTypeNames);

VELOX_DEFINE_ENUM_NAME(TransformCategory, transformCategoryNames);

void IcebergPartitionSpec::checkCompatibility() const {
  folly::F14FastMap<std::string_view, std::vector<TransformType>>
      columnTransforms;

  for (const auto& field : fields) {
    const auto& type = field.type;
    const auto& name = field.name;
    VELOX_USER_CHECK(
        isValidPartitionType(type),
        "Type is not supported as a partition column: {}",
        type->name());

    VELOX_USER_CHECK(
        canTransform(field.transformType, type),
        "Transform is not supported for partition column. Column: '{}', Type: '{}', Transform: '{}'.",
        name,
        type->name(),
        TransformTypeName::toName(field.transformType));

    columnTransforms[name].emplace_back(field.transformType);
  }

  // Check for duplicate transform categories per column.
  std::vector<std::string> errors;
  for (const auto& [columnName, transforms] : columnTransforms) {
    folly::F14FastSet<TransformCategory> seenCategories;
    for (const auto& transform : transforms) {
      auto category = getTransformCategory(transform);
      if (!seenCategories.insert(category).second) {
        std::vector<std::string> transformNames;
        for (const auto& t : transforms) {
          transformNames.emplace_back(
              std::string(TransformTypeName::toName(t)));
        }
        errors.emplace_back(
            fmt::format(
                "Column: '{}', Category: {}, Transforms: [{}]",
                columnName,
                TransformCategoryName::toName(category),
                folly::join(", ", transformNames)));
        break;
      }
    }
  }

  VELOX_USER_CHECK(
      errors.empty(),
      "Multiple transforms of the same category on a column are not allowed. "
      "Each transform category can appear at most once per column. {}",
      folly::join("; ", errors));
}

} // namespace facebook::velox::connector::hive::iceberg
