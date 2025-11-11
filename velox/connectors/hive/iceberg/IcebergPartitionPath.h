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

#include "velox/connectors/hive/HivePartitionUtil.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"

namespace facebook::velox::connector::hive::iceberg {

/// Converts a partition value to its string representation for use in
/// partition directory path. The implementation follows the behavior of
/// the Apache Iceberg Java library for partition path name.
class IcebergPartitionPath : public HivePartitionUtil {
 public:
  explicit IcebergPartitionPath(TransformType transformType)
      : transformType_(transformType) {}

  using HivePartitionUtil::toPartitionString;

  /// Converts an int32_t partition key to its string representation based on
  /// the transform type:
  /// - kIdentity: For DATE type return "YYYY-MM-DD" format (e.g.,
  ///              "2025-11-07").
  ///              For other types return the value as-is (e.g., "-123").
  /// - kDay: Returns date in "YYYY-MM-DD" format (e.g., "2025-11-07").
  /// - kYear: Returns 4-digit year "YYYY" (e.g., "2025").
  /// - kMonth: Returns "YYYY-MM" format (e.g., "2025-01").
  /// - kHour: Returns "YYYY-MM-DD-HH" format (e.g., "2025-11-07-21").
  std::string toPartitionString(int32_t value, const TypePtr& type)
      const override;

  /// Returns timestamp formatted with milliseconds precision, zero-padded year,
  /// trailing zeros skipped, and leading positive sign for years >= 10000.
  /// Examples:
  /// - Timestamp(0, 0) -> "1970-01-01T00:00:00".
  /// - Timestamp(1609459200, 999000000) -> "2021-01-01T00:00:00.999".
  /// - Timestamp(1640995200, 500000000) -> "2022-01-01T00:00:00.5".
  /// - Timestamp(-1, 999000000) -> "1969-12-31T23:59:59.999".
  /// - Timestamp(253402300800, 100000000) -> "+10000-01-01T00:00:00.1".
  std::string toPartitionString(Timestamp value, const TypePtr& type)
      const override;

  /// Converts a StringView partition key to its string representation.
  /// - For VARBINARY type returns Base64-encoded string.
  /// - For VARCHAR type returns the string value as-is.
  std::string toPartitionString(StringView value, const TypePtr& type)
      const override;

 private:
  const TransformType transformType_;
};

using IcebergPartitionPathPtr = std::shared_ptr<const IcebergPartitionPath>;

} // namespace facebook::velox::connector::hive::iceberg
