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

#include "velox/connectors/hive/iceberg/TransformFactory.h"

#include "velox/functions/lib/TimeUtils.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {
int32_t epochYear(int32_t daysSinceEpoch) {
  const std::tm tm = functions::getDateTime(daysSinceEpoch);
  // tm_year is the number of years since 1900.
  return tm.tm_year + 1900 - 1970;
}

int32_t epochYear(Timestamp ts) {
  return functions::getYear(functions::getDateTime(ts, nullptr)) - 1970;
}

int32_t epochMonth(int32_t daysSinceEpoch) {
  const std::tm tm = functions::getDateTime(daysSinceEpoch);
  return (tm.tm_year + 1900 - 1970) * 12 + tm.tm_mon;
}

int32_t epochMonth(Timestamp ts) {
  const std::tm tm = functions::getDateTime(ts, nullptr);
  return (tm.tm_year + 1900 - 1970) * 12 + tm.tm_mon;
}

int32_t epochDay(int32_t daysSinceEpoch) {
  return daysSinceEpoch;
}

int32_t epochDay(Timestamp ts) {
  const auto seconds = ts.getSeconds();
  return (seconds >= 0) ? seconds / Timestamp::kSecondsInDay
                        : ((seconds + 1) / Timestamp::kSecondsInDay) - 1;
}

int32_t epochHour(Timestamp ts) {
  const auto seconds = ts.getSeconds();
  return (seconds >= 0) ? seconds / 3600 : ((seconds + 1) / 3600) - 1;
}

bool isValidPartitionType(TypePtr type) {
  if (type->isRow() || type->isArray() || type->isMap() ||
      isTimestampWithTimeZoneType(type)) {
    return false;
  }
  return true;
}

template <TypeKind Kind>
std::shared_ptr<Transform> createDateTimeTransform(
    TransformType transformType,
    const IcebergPartitionSpec::Field& field,
    std::function<int32_t(typename TypeTraits<Kind>::NativeType)> epochFunc,
    memory::MemoryPool* pool) {
  using NativeType = typename TypeTraits<Kind>::NativeType;
  VELOX_DCHECK_EQ(
      true,
      field.type->isDate() || field.type->isTimestamp(),
      "Unsupported column type {} for transform {}",
      field.type->name(),
      TransformTypeName::toName(transformType));
  return std::make_shared<TemporalTransform<NativeType>>(
      field.type, transformType, field.name, pool, epochFunc);
}

template <TypeKind Kind>
std::shared_ptr<Transform> createIdentityTransform(
    const IcebergPartitionSpec::Field& field,
    memory::MemoryPool* pool) {
  using NativeType = typename TypeTraits<Kind>::NativeType;
  return std::make_shared<IdentityTransform<NativeType>>(
      field.type, field.name, pool);
}

template <TypeKind Kind>
std::shared_ptr<Transform> createBucketTransform(
    const IcebergPartitionSpec::Field& field,
    int32_t count,
    memory::MemoryPool* pool) {
  VELOX_USER_CHECK_GT(count, 0, "Bucket count must be positive.");
  using NativeType = typename TypeTraits<Kind>::NativeType;
  return std::make_shared<BucketTransform<NativeType>>(
      count, field.type, field.name, pool);
}

template <TypeKind Kind>
std::shared_ptr<Transform> createTruncateTransform(
    const IcebergPartitionSpec::Field& field,
    int32_t width,
    memory::MemoryPool* pool) {
  VELOX_USER_CHECK_GT(width, 0, "Truncate width must be positive.");
  using NativeType = typename TypeTraits<Kind>::NativeType;
  return std::make_shared<TruncateTransform<NativeType>>(
      width, field.type, field.name, pool);
}

std::shared_ptr<Transform> buildColumnTransform(
    const IcebergPartitionSpec::Field& field,
    memory::MemoryPool* pool) {
  if (!isValidPartitionType(field.type)) {
    VELOX_USER_FAIL(fmt::format(
        "Type not supported as partition column: {}.", field.type->name()));
  }
  switch (field.transformType) {
    // Identity transform.
    case TransformType::kIdentity: {
      return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createIdentityTransform, field.type->kind(), field, pool);
    }
    // Year transform.
    case TransformType::kYear: {
      if (field.type->isDate()) {
        return createDateTimeTransform<TypeKind::INTEGER>(
            TransformType::kYear,
            field,
            [](int32_t v) { return epochYear(v); },
            pool);
      }

      if (field.type->isTimestamp()) {
        return createDateTimeTransform<TypeKind::TIMESTAMP>(
            TransformType::kYear,
            field,
            [](Timestamp v) { return epochYear(v); },
            pool);
      }

      VELOX_UNREACHABLE(fmt::format(
          "Unsupported column type {} for transform year.",
          field.type->name()));
    }
    // Month transform.
    case TransformType::kMonth: {
      if (field.type->isDate()) {
        return createDateTimeTransform<TypeKind::INTEGER>(
            TransformType::kMonth,
            field,
            [](int32_t v) { return epochMonth(v); },
            pool);
      }

      if (field.type->isTimestamp()) {
        return createDateTimeTransform<TypeKind::TIMESTAMP>(
            TransformType::kMonth,
            field,
            [](Timestamp v) { return epochMonth(v); },
            pool);
      }

      VELOX_UNREACHABLE(fmt::format(
          "Unsupported column type {} for transform month.",
          field.type->name()));
    }
    // Day transform.
    case TransformType::kDay: {
      if (field.type->isDate()) {
        return createDateTimeTransform<TypeKind::INTEGER>(
            TransformType::kDay,
            field,
            [](int32_t v) { return epochDay(v); },
            pool);
      }

      if (field.type->isTimestamp()) {
        return createDateTimeTransform<TypeKind::TIMESTAMP>(
            TransformType::kDay,
            field,
            [](Timestamp v) { return epochDay(v); },
            pool);
      }

      VELOX_UNREACHABLE(fmt::format(
          "Unsupported column type {} for transform day.", field.type->name()));
    }
    // Hour transform.
    case TransformType::kHour: {
      if (field.type->isTimestamp()) {
        return createDateTimeTransform<TypeKind::TIMESTAMP>(
            TransformType::kHour,
            field,
            [](Timestamp v) { return epochHour(v); },
            pool);
      }

      VELOX_UNREACHABLE(fmt::format(
          "Unsupported column type {} for transform hour.",
          field.type->name()));
    }
    // Bucket transform.
    case TransformType::kBucket: {
      VELOX_USER_CHECK(
          field.parameter.has_value() && field.parameter.value() > 0,
          "Bucket transform requires a positive parameter.");
      auto numBuckets = field.parameter.value();

      if (field.type->isInteger() || field.type->isDate()) {
        return createBucketTransform<TypeKind::INTEGER>(
            field, numBuckets, pool);
      }
      if (field.type->isBigint() || field.type->isShortDecimal()) {
        return createBucketTransform<TypeKind::BIGINT>(field, numBuckets, pool);
      }
      if (field.type->isTimestamp()) {
        return createBucketTransform<TypeKind::TIMESTAMP>(
            field, numBuckets, pool);
      }
      if (field.type->isLongDecimal()) {
        return createBucketTransform<TypeKind::HUGEINT>(
            field, numBuckets, pool);
      }
      if (field.type->isVarchar()) {
        return createBucketTransform<TypeKind::VARCHAR>(
            field, numBuckets, pool);
      }
      if (field.type->isVarbinary()) {
        return createBucketTransform<TypeKind::VARBINARY>(
            field, numBuckets, pool);
      }
      VELOX_UNREACHABLE(fmt::format(
          "Unsupported column type {} for transform bucket.",
          field.type->name()));
    }
    // Truncate transform.
    case TransformType::kTruncate: {
      VELOX_USER_CHECK(
          field.parameter.has_value() && field.parameter.value() > 0,
          "Truncate transform requires a positive parameter.");
      auto width = field.parameter.value();
      if (field.type->isInteger()) {
        return createTruncateTransform<TypeKind::INTEGER>(field, width, pool);
      }
      if (field.type->isBigint() || field.type->isShortDecimal()) {
        return createTruncateTransform<TypeKind::BIGINT>(field, width, pool);
      }
      if (field.type->isVarchar()) {
        return createTruncateTransform<TypeKind::VARCHAR>(field, width, pool);
      }
      if (field.type->isVarbinary()) {
        return createTruncateTransform<TypeKind::VARBINARY>(field, width, pool);
      }
      VELOX_UNREACHABLE(fmt::format(
          "Unsupported column type {} for transform truncate.",
          field.type->name()));
    }
    default:
      VELOX_UNREACHABLE("Unsupported transform.");
  }
}

} // namespace

std::vector<std::shared_ptr<Transform>> parsePartitionTransformSpecs(
    const std::vector<IcebergPartitionSpec::Field>& fields,
    memory::MemoryPool* pool) {
  std::vector<std::shared_ptr<Transform>> transforms;
  transforms.reserve(fields.size());
  for (auto& field : fields) {
    transforms.emplace_back(buildColumnTransform(field, pool));
  }
  return transforms;
}

} // namespace facebook::velox::connector::hive::iceberg
