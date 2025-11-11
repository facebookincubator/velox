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
#include "velox/connectors/hive/iceberg/IcebergPartitionPath.h"
#include "velox/common/encode/Base64.h"

namespace facebook::velox::connector::hive::iceberg {

std::string IcebergPartitionPath::toPartitionString(
    int32_t value,
    const TypePtr& type) const {
  constexpr int32_t kEpochYear = 1970;
  switch (transformType_) {
    case TransformType::kIdentity: {
      if (type->isDate()) {
        return DATE()->toString(value);
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

std::string IcebergPartitionPath::toPartitionString(
    Timestamp value,
    const TypePtr& type) const {
  VELOX_CHECK(transformType_ == TransformType::kIdentity);
  TimestampToStringOptions options;
  options.precision = TimestampPrecision::kMilliseconds;
  options.zeroPaddingYear = true;
  options.skipTrailingZeros = true;
  options.leadingPositiveSign = true;
  return value.toString(options);
}

std::string IcebergPartitionPath::toPartitionString(
    StringView value,
    const TypePtr& type) const {
  if (type->isVarbinary()) {
    return encoding::Base64::encode(value.data(), value.size());
  }
  return std::string(value);
}

} // namespace facebook::velox::connector::hive::iceberg
