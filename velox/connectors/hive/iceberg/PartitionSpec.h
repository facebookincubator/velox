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

#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

/// Partition transform types supported by Iceberg.
/// Define how source column values are converted into partition values.
/// Each transform type corresponds to a specific partitioning strategy.
/// See https://iceberg.apache.org/spec/#partition-transforms.
enum class TransformType {
  /// Use the source value as-is (no transformation).
  kIdentity,
  /// Extract a timestamp hour, as hours from 1970-01-01 00:00:00.
  kHour,
  /// Extract a date or timestamp day, as days from 1970-01-01.
  kDay,
  /// Extract a date or timestamp month, as months from 1970-01.
  kMonth,
  /// Extract a date or timestamp year, as years from 1970.
  kYear,
  /// Hash the value into N buckets for even distribution.
  kBucket,
  /// Truncate strings or numbers to a specified width.
  kTruncate
};

VELOX_DECLARE_ENUM_NAME(TransformType);

/// Represents how to produce partition data for an Iceberg table.
///
/// This structure corresponds to the Iceberg Java PartitionSpec class but
/// contains only the necessary fields for Velox. Partition data is produced
/// by transforming columns in a table, where each column can have multiple
/// transforms applied in sequence.
///
/// The upstream engine processes this specification through the Iceberg Java
/// library to validate column types, detect duplicates, and generate the
/// partition spec that is passed to Velox.
///
/// IMPORTANT: Iceberg spec uses field IDs to identify source columns, but
/// Velox RowType only supports matching fields by name. Therefore, Velox uses
/// the partition field name to match against the table schema column names.
/// Callers must ensure that partition field names exactly match the column
/// names in the table schema.
///
/// Multiple transforms on the same column are allowed, but with restrictions:
/// - Transforms are organized into 4 categories: Identity, Temporal
///   (Year/Month/Day/Hour), Bucket, and Truncate.
/// - Each category can appear at most once per column.
/// - Example valid specs: ARRAY['truncate(a, 2)', 'bucket(a, 16)', 'a'] or
///   ARRAY['year(b)', 'bucket(b, 16)', 'b']
///
/// The partition spec defines:
/// - Unique ID for versioning and evolution.
/// - Which source columns in current table schema to use for partitioning
/// (identified by field name,
///   not field ID as in the Iceberg spec).
/// - What transforms to apply (identity, bucket, truncate etc.).
/// - Transform parameters (e.g., bucket count, truncate width).
struct IcebergPartitionSpec {
  struct Field {
    /// The field name of this partition field as it appears in the partition
    /// spec. This is the original Iceberg field name, not the transformed name
    /// from org.apache.iceberg.PartitionField which includes the transform as a
    /// suffix.
    const std::string name;

    /// The source column type.
    const TypePtr type;

    /// The transform type applied to the source field (e.g., kIdentity,
    /// kBucket, kTruncate, etc.). Callers must ensure the transform is
    /// compatible with the source column type.
    const TransformType transformType;

    /// Optional parameter for transforms that require configuration
    /// (e.g., bucket count or truncate width).
    const std::optional<int32_t> parameter;

    /// Returns the result type after applying this transform.
    TypePtr resultType() const {
      switch (transformType) {
        case TransformType::kBucket:
        case TransformType::kYear:
        case TransformType::kMonth:
        case TransformType::kHour:
          return INTEGER();
        case TransformType::kDay:
          return DATE();
        case TransformType::kIdentity:
        case TransformType::kTruncate:
        default:
          return type;
      }
    }
  };

  const int32_t specId;
  const std::vector<Field> fields;

  /// Constructor with validation that:
  /// - Each field's type is supported for partitioning.
  /// - Each field's transform type is compatible with its data type.
  /// - No transform category appears more than once per column (Identity,
  ///   Temporal, Bucket, and Truncate are separate categories).
  ///
  /// @param _specId Partition specification ID.
  /// @param _fields Vector of partition fields. When empty indicates no
  /// partition.
  /// @throws VELOX_USER_FAIL if validation fails.
  IcebergPartitionSpec(int32_t _specId, const std::vector<Field>& _fields)
      : specId(_specId), fields(_fields) {
    checkCompatibility();
  }

 private:
  /// Validates partition fields for correctness.
  /// Checks type/transform compatibility and transform combination rules.
  void checkCompatibility() const;
};

using IcebergPartitionSpecPtr = std::shared_ptr<const IcebergPartitionSpec>;

} // namespace facebook::velox::connector::hive::iceberg
