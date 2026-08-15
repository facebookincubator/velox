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

#include "velox/connectors/hive/HivePartitionName.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"

namespace facebook::velox::connector::hive::iceberg {

/// Generates Iceberg-compliant partition path names.
/// Converts partition keys to human-readable strings based on their transform
/// types (e.g., year, month, day, hour, identity, truncate) and constructs
/// URL-encoded partition paths in the format "key1=value1/key2=value2/...".
class IcebergPartitionName {
 public:
  /// @param partitionSpec Iceberg partition specification containing transform
  /// definitions for each partition field. Used to get transform type and call
  /// different format functions to convert transformed partition values to
  /// human-readable strings.
  IcebergPartitionName(const IcebergPartitionSpecPtr& partitionSpec);

  /// Generates an Iceberg compliant partition path string for the given
  /// partition ID.
  ///
  /// Constructs a partition path in the format "key1=value1/key2=value2/..."
  /// where:
  /// - Keys are partition column names for identity transforms, or
  ///   "columnName_transformName" for non-identity transforms (e.g.,
  ///   "date_year")
  /// - Values are human-readable string representations of the transformed
  ///   partition keys, formatted according to their transform types
  /// - Both keys and values are URL-encoded per java.net.URLEncoder.encode()
  ///
  /// Example: "store_id=123/date_year=2025/address_bucket=1"
  ///
  /// Typically called once per partition ID when creating a new writer for that
  /// partition.
  ///
  /// @param partitionId Sequential partition ID (0-based) used as the row index
  /// into partitionValues. Must be less than partitionValues->size().
  /// @param partitionValues RowVector containing transformed partition keys
  /// for all partitions. Each row represents one unique partition, with
  /// columns corresponding to partition fields in partitionSpec. Row at
  /// partitionId contains the keys for this specific partition.
  /// @param partitionKeyAsLowerCase Whether to convert partition keys to
  /// lowercase in the generated partition path. When true, partition keys like
  /// "Year" become "year" in the path "year=2025/...".
  /// @return URL-encoded partition path string suitable for use in file paths.
  std::string partitionName(
      uint32_t partitionId,
      const RowVectorPtr& partitionValues,
      bool partitionKeyAsLowerCase) const;

  /// Generic template for formatting simple types that just need string
  /// conversion. Specialized for types that need special handling.
  template <typename T>
  FOLLY_ALWAYS_INLINE static std::string
  toName(T value, const TypePtr& type, TransformType transformType) {
    return HivePartitionName::toName(value, type);
  }

  /// Converts an int32_t partition key to its string representation based on
  /// the transform type:
  /// - kIdentity: For DATE type return "YYYY-MM-DD" format (e.g.,
  ///              "2025-11-07").
  ///              For other types return the value as-is (e.g., "-123").
  /// - kDay: Returns date in "YYYY-MM-DD" format (e.g., "2025-11-07").
  /// - kYear: Returns 4-digit year "YYYY" (e.g., "2025").
  /// - kMonth: Returns "YYYY-MM" format (e.g., "2025-01").
  /// - kHour: Returns "YYYY-MM-DD-HH" format (e.g., "2025-11-07-21").
  static std::string
  toName(int32_t value, const TypePtr& type, TransformType transformType);

  /// Returns timestamp formatted with milliseconds precision, zero-padded
  /// year, trailing zeros skipped, and leading positive sign for years >=
  /// 10000. Examples:
  /// - Timestamp(0, 0) -> "1970-01-01T00:00:00".
  /// - Timestamp(1609459200, 999000000) -> "2021-01-01T00:00:00.999".
  /// - Timestamp(1640995200, 500000000) -> "2022-01-01T00:00:00.5".
  /// - Timestamp(-1, 999000000) -> "1969-12-31T23:59:59.999".
  /// - Timestamp(253402300800, 100000000) -> "+10000-01-01T00:00:00.1".
  static std::string
  toName(Timestamp value, const TypePtr& type, TransformType transformType);

  /// Converts a StringView partition key to its string representation.
  /// - For VARBINARY type returns Base64-encoded string.
  /// - For VARCHAR type returns the string value as-is.
  static std::string
  toName(StringView value, const TypePtr& type, TransformType transformType);

 private:
  // Cached transform types, one per partition column. Created once in
  // constructor and reused for all formatting operations. Index corresponds to
  // column index in partitionSpec_->fields.
  std::vector<TransformType> transformTypes_;
};

using IcebergPartitionNamePtr = std::shared_ptr<const IcebergPartitionName>;

} // namespace facebook::velox::connector::hive::iceberg
