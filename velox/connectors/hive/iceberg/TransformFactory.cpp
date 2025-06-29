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

#include "velox/connectors/hive/iceberg/ColumnTransform.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"
#include "velox/connectors/hive/iceberg/TransformFunction.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {
inline constexpr int32_t SECONDS_PER_DAY = 86400;

const std::string TRANSFORM_IDENTITY = "identity";
const std::string TRANSFORM_YEAR = "year";
const std::string TRANSFORM_MONTH = "month";
const std::string TRANSFORM_DAY = "day";
const std::string TRANSFORM_HOUR = "hour";
const std::string TRANSFORM_BUCKET = "bucket";
const std::string TRANSFORM_TRUNCATE = "trunc";

const std::tm getUTCTime(int64_t daysSinceEpoch) {
  const std::time_t seconds =
      static_cast<std::time_t>(daysSinceEpoch) * SECONDS_PER_DAY;
  std::tm timeInfo;
  if (!gmtime_r(&seconds, &timeInfo)) {
    VELOX_FAIL("Failed to convert days to UTC time: {}", daysSinceEpoch);
  }
  return timeInfo;
}

int32_t epochYear(int64_t daysSinceEpoch) {
  const std::tm timeInfo = getUTCTime(daysSinceEpoch);
  // tm_year is the number of years since 1900.
  return timeInfo.tm_year + 1900 - 1970;
}

int32_t epochMonth(int64_t daysSinceEpoch) {
  const std::tm timeInfo = getUTCTime(daysSinceEpoch);
  return (timeInfo.tm_year + 1900 - 1970) * 12 + timeInfo.tm_mon;
}

int32_t epochDay(int64_t daysSinceEpoch) {
  return daysSinceEpoch;
}

template <TypeKind Kind>
ColumnTransform createTemporalTransform(
    const std::string& transformName,
    std::function<int32_t(int64_t)> epochFunc,
    const std::string& columnName,
    const TypePtr& type,
    memory::MemoryPool* pool) {
  using NativeType = typename TypeTraits<Kind>::NativeType;
  auto transform =
      std::make_shared<TemporalTransform<NativeType>>(epochFunc, type, pool);

  if (type->isDate() || type->isTimestamp()) {
    return ColumnTransform(columnName, transformName, transform, std::nullopt);
  } else {
    VELOX_NYI(fmt::format(
        "Unsupported column type {} for transform {}",
        type->name(),
        transformName));
  }
}

template <TypeKind Kind>
ColumnTransform createIdentityTransform(
    const std::string& columnName,
    const std::string& transformName,
    const std::shared_ptr<const Type>& type,
    memory::MemoryPool* pool) {
  using NativeType = typename TypeTraits<Kind>::NativeType;
  auto transform = std::make_shared<IdentityTransform<NativeType>>(type, pool);
  return ColumnTransform(columnName, transformName, transform, std::nullopt);
}

template <TypeKind Kind>
ColumnTransform createBucketTransform(
    const std::string& columnName,
    const std::string& transformName,
    const std::shared_ptr<const Type>& type,
    int32_t count,
    memory::MemoryPool* pool) {
  VELOX_USER_CHECK_GT(count, 0, "Bucket count must be positive.");
  using NativeType = typename TypeTraits<Kind>::NativeType;
  auto transform =
      std::make_shared<BucketTransform<NativeType>>(count, type, pool);
  return ColumnTransform(columnName, transformName, transform, count);
}

template <TypeKind Kind>
ColumnTransform createTruncateTransform(
    const std::string& columnName,
    const std::string& transformName,
    const std::shared_ptr<const Type>& type,
    int32_t width,
    memory::MemoryPool* pool) {
  VELOX_USER_CHECK_GT(width, 0, "Truncate width must be positive.");
  using NativeType = typename TypeTraits<Kind>::NativeType;
  auto transform =
      std::make_shared<TruncateTransform<NativeType>>(width, type, pool);
  return ColumnTransform(columnName, transformName, transform, width);
}

ColumnTransform buildColumnTransform(
    const std::string& columnName,
    const std::string& transformName,
    const TypePtr& type,
    std::optional<int32_t> parameter,
    memory::MemoryPool* pool) {
  // Identity transform.
  if (transformName == TRANSFORM_IDENTITY) {
    if (type->isInteger()) {
      return createIdentityTransform<TypeKind::INTEGER>(
          columnName, transformName, type, pool);
    }
    if (type->isSmallint() || type->isTinyint()) {
      return createIdentityTransform<TypeKind::SMALLINT>(
          columnName, transformName, type, pool);
    }
    if (type->isBigint() || type->isShortDecimal()) {
      return createIdentityTransform<TypeKind::BIGINT>(
          columnName, transformName, type, pool);
    }
    if (type->isVarchar()) {
      return createIdentityTransform<TypeKind::VARCHAR>(
          columnName, transformName, type, pool);
    }
    if (type->isVarbinary()) {
      return createIdentityTransform<TypeKind::VARBINARY>(
          columnName, transformName, type, pool);
    }
    if (type->isBoolean()) {
      return createIdentityTransform<TypeKind::BOOLEAN>(
          columnName, transformName, type, pool);
    }
    if (type->isLongDecimal()) {
      return createIdentityTransform<TypeKind::HUGEINT>(
          columnName, transformName, type, pool);
    }
    VELOX_NYI(fmt::format(
        "Unsupported column type {} for transform {}",
        type->name(),
        transformName));
  }
  // Year transform.
  if (transformName == TRANSFORM_YEAR) {
    if (type->isDate()) {
      return createTemporalTransform<TypeKind::INTEGER>(
          std::string(TRANSFORM_YEAR),
          [](int32_t v) { return epochYear(v); },
          columnName,
          type,
          pool);
    }
    if (type->isTimestamp()) {
      return createTemporalTransform<TypeKind::BIGINT>(
          std::string(TRANSFORM_YEAR),
          [](int64_t v) { return epochYear(v); },
          columnName,
          type,
          pool);
    }
    VELOX_NYI(fmt::format(
        "Unsupported column type {} for transform {}",
        type->name(),
        transformName));
  }
  // Month transform.
  if (transformName == TRANSFORM_MONTH) {
    if (type->isDate()) {
      return createTemporalTransform<TypeKind::INTEGER>(
          std::string(TRANSFORM_MONTH),
          [](int32_t v) { return epochMonth(v); },
          columnName,
          type,
          pool);
    }
    if (type->isTimestamp()) {
      return createTemporalTransform<TypeKind::BIGINT>(
          std::string(TRANSFORM_MONTH),
          [](int64_t v) { return epochMonth(v); },
          columnName,
          type,
          pool);
    }
    VELOX_NYI(fmt::format(
        "Unsupported column type {} for transform {}",
        type->name(),
        transformName));
  }
  // Day transform.
  if (transformName == TRANSFORM_DAY) {
    if (type->isDate()) {
      return createTemporalTransform<TypeKind::INTEGER>(
          std::string(TRANSFORM_DAY),
          [](int32_t v) { return epochDay(v); },
          columnName,
          type,
          pool);
    }
    if (type->isTimestamp()) {
      return createTemporalTransform<TypeKind::BIGINT>(
          std::string(TRANSFORM_DAY),
          [](int64_t v) { return epochDay(v); },
          columnName,
          type,
          pool);
    }
    VELOX_NYI(fmt::format(
        "Unsupported column type {} for transform {}",
        type->name(),
        transformName));
  }
  // Hour transform.
  if (transformName == TRANSFORM_HOUR) {
    VELOX_NYI("Unsupported partition transform.");
  }
  // Bucket transform.
  if (transformName == TRANSFORM_BUCKET) {
    VELOX_USER_CHECK(
        parameter.has_value() && parameter.value() > 0,
        "Transform '{}' requires a positive parameter.",
        transformName);
    auto count = parameter.value();
    if (type->isInteger()) {
      return createBucketTransform<TypeKind::INTEGER>(
          columnName, transformName, type, count, pool);
    }
    if (type->isBigint() || type->isShortDecimal()) {
      return createBucketTransform<TypeKind::BIGINT>(
          columnName, transformName, type, count, pool);
    }
    if (type->isVarchar()) {
      return createBucketTransform<TypeKind::VARCHAR>(
          columnName, transformName, type, count, pool);
    }
    if (type->isVarbinary()) {
      return createBucketTransform<TypeKind::VARBINARY>(
          columnName, transformName, type, count, pool);
    }
    VELOX_NYI(fmt::format(
        "Unsupported column type {} for transform {}",
        type->name(),
        transformName));
  }

  // Truncate transform.
  if (transformName == TRANSFORM_TRUNCATE) {
    VELOX_USER_CHECK(
        parameter.has_value() && parameter.value() > 0,
        "Transform '{}' requires a positive parameter.",
        transformName);
    auto width = parameter.value();
    if (type->isInteger()) {
      return createTruncateTransform<TypeKind::INTEGER>(
          columnName, transformName, type, width, pool);
    }
    if (type->isBigint() || type->isShortDecimal()) {
      return createTruncateTransform<TypeKind::BIGINT>(
          columnName, transformName, type, width, pool);
    }
    if (type->isVarchar()) {
      return createTruncateTransform<TypeKind::VARCHAR>(
          columnName, transformName, type, width, pool);
    }
    if (type->isVarbinary()) {
      return createTruncateTransform<TypeKind::VARBINARY>(
          columnName, transformName, type, width, pool);
    }
    VELOX_NYI(fmt::format(
        "Unsupported column type {} for transform {}",
        type->name(),
        transformName));
  }
  VELOX_FAIL(fmt::format("Unsupported transform {}."), transformName);
}

TypePtr getColumnType(
    const std::string& name,
    const std::shared_ptr<const IcebergPartitionSpec::Schema>& schema) {
  return schema->columnNameToTypeMapping.at(name);
}

std::optional<ColumnTransform> parsePartitionTransform(
    const IcebergPartitionSpec::Field& field,
    const std::shared_ptr<const IcebergPartitionSpec::Schema>& schema,
    memory::MemoryPool* pool) {
  static const std::unordered_map<
      IcebergPartitionSpec::TransformType,
      std::string>
      transformTypeToName = {
          {IcebergPartitionSpec::TransformType::kIdentity, TRANSFORM_IDENTITY},
          {IcebergPartitionSpec::TransformType::kYear, TRANSFORM_YEAR},
          {IcebergPartitionSpec::TransformType::kMonth, TRANSFORM_MONTH},
          {IcebergPartitionSpec::TransformType::kDay, TRANSFORM_DAY},
          {IcebergPartitionSpec::TransformType::kHour, TRANSFORM_HOUR},
          {IcebergPartitionSpec::TransformType::kBucket, TRANSFORM_BUCKET},
          {IcebergPartitionSpec::TransformType::kTruncate, TRANSFORM_TRUNCATE}};

  auto type = getColumnType(field.name, schema);
  VELOX_CHECK(type, "Partition column '{}' not found in schema.", field.name);

  const auto& transformName = transformTypeToName.at(field.transform);
  if (field.transform == IcebergPartitionSpec::TransformType::kBucket ||
      field.transform == IcebergPartitionSpec::TransformType::kTruncate) {
    VELOX_CHECK(
        field.parameter.has_value(),
        "Transform '{}' requires a positive parameter.",
        transformName);
    return buildColumnTransform(
        field.name, transformName, type, field.parameter.value(), pool);
  }

  // identity, year, month, day, hour.
  return buildColumnTransform(
      field.name, transformName, type, std::nullopt, pool);
}

} // namespace

std::shared_ptr<ColumnTransforms> parsePartitionTransformSpecs(
    const std::vector<IcebergPartitionSpec::Field>& fields,
    const std::shared_ptr<const IcebergPartitionSpec::Schema>& schema,
    memory::MemoryPool* pool) {
  auto columnTransforms = std::make_shared<ColumnTransforms>();
  for (auto& field : fields) {
    auto transform = parsePartitionTransform(field, schema, pool);
    if (transform.has_value()) {
      columnTransforms->add(transform.value());
    }
  }
  return columnTransforms;
}

} // namespace facebook::velox::connector::hive::iceberg
