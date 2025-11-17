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
#include "velox/connectors/hive/iceberg/IcebergPartitionName.h"
#include "velox/common/encode/Base64.h"
#include "velox/dwio/catalog/fbhive/FileUtils.h"
#include "velox/functions/prestosql/URLFunctions.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

std::string escapePathName(const std::string& name) {
  std::string encoded;
  // Pre-allocate for worst case: every byte is invalid UTF-8.
  // urlEscape() writes directly into the pre-allocated buffer and
  // calls resize() at the end to shrink to the actual size used.
  encoded.resize(name.size() * 9);
  functions::detail::urlEscape(encoded, name);
  return encoded;
}

} // namespace

IcebergPartitionName::IcebergPartitionName(
    const IcebergPartitionSpecPtr& partitionSpec) {
  VELOX_CHECK_NOT_NULL(partitionSpec);
  transformTypes_.reserve(partitionSpec->fields.size());
  for (const auto& field : partitionSpec->fields) {
    transformTypes_.emplace_back(field.transformType);
  }
}

std::string IcebergPartitionName::partitionName(
    uint32_t partitionId,
    const RowVectorPtr& partitionValues,
    bool partitionKeyAsLowerCase) const {
  auto toPartitionName = [this](
                             auto value, const TypePtr& type, int columnIndex) {
    return IcebergPartitionName::toName(
        value, type, transformTypes_[columnIndex]);
  };

  return dwio::catalog::fbhive::FileUtils::makePartName(
      HivePartitionName::partitionKeyValues(
          partitionId,
          partitionValues,
          /*nullValueString=*/"null",
          toPartitionName),
      partitionKeyAsLowerCase,
      /*useDefaultPartitionValue=*/false,
      escapePathName);
}

std::string IcebergPartitionName::toName(
    int32_t value,
    const TypePtr& type,
    TransformType transformType) {
  constexpr int32_t kEpochYear = 1970;
  switch (transformType) {
    case TransformType::kIdentity: {
      if (type->isDate()) {
        return DateType::toIso8601(value);
      }
      return fmt::to_string(value);
    }
    case TransformType::kDay:
      return DATE()->toString(value);
    case TransformType::kYear:
      return fmt::format("{:04d}", kEpochYear + value);
    case TransformType::kMonth: {
      int32_t year = kEpochYear + value / 12;
      int32_t month = 1 + value % 12;
      if (month <= 0) {
        month += 12;
        year -= 1;
      }
      return fmt::format("{:04d}-{:02d}", year, month);
    }
    case TransformType::kHour: {
      int64_t seconds = static_cast<int64_t>(value) * 3600;
      std::tm tmValue;
      VELOX_USER_CHECK(
          Timestamp::epochToCalendarUtc(seconds, tmValue),
          "Failed to convert seconds to time: {}",
          seconds);
      return fmt::format(
          "{:04d}-{:02d}-{:02d}-{:02d}",
          tmValue.tm_year + 1900,
          tmValue.tm_mon + 1,
          tmValue.tm_mday,
          tmValue.tm_hour);
    }
    default:
      return fmt::to_string(value);
  }
}

std::string IcebergPartitionName::toName(
    Timestamp value,
    const TypePtr& type,
    TransformType transformType) {
  VELOX_CHECK(transformType == TransformType::kIdentity);
  TimestampToStringOptions options;
  options.precision = TimestampPrecision::kMilliseconds;
  options.zeroPaddingYear = true;
  options.skipTrailingZeros = true;
  options.leadingPositiveSign = true;
  return value.toString(options);
}

std::string IcebergPartitionName::toName(
    StringView value,
    const TypePtr& type,
    TransformType transformType) {
  VELOX_CHECK(
      transformType == TransformType::kIdentity ||
      transformType == TransformType::kTruncate);
  if (type->isVarbinary()) {
    return encoding::Base64::encode(value.data(), value.size());
  }
  return std::string(value);
}

} // namespace facebook::velox::connector::hive::iceberg
