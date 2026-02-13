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

#pragma once

#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

/// Partition transform types.
/// Defines how source column values are converted into partition keys.
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
  /// Hash the value into N buckets for even distribution. Requires an integer
  /// parameter specifying the bucket count.
  kBucket,
  /// Truncate strings or numbers to a specified width. Requires an integer
  /// parameter specifying the truncate width.
  kTruncate
};

VELOX_DECLARE_ENUM_NAME(TransformType);

/// A single column can be used to produce multiple partition keys, but with
/// following restrictions:
/// - Transforms are organized into 4 categories: Identity, Temporal,
///   Bucket, and Truncate.
/// - Each category can appear at most once per column.
/// - Sample valid specs on same column: ['truncate(a,2)', 'bucket(a,16)', 'a']
///   or ['year(b)', 'bucket(b, 16)', 'b']
enum class TransformCategory {
  kIdentity,
  /// Year/Month/Day/Hour
  kTemporal,
  kBucket,
  kTruncate,
};

VELOX_DECLARE_ENUM_NAME(TransformCategory);

/// Represents how to produce partition data for an Iceberg table.
///
/// This structure corresponds to the Iceberg Java PartitionSpec class but
/// contains only the necessary fields for Velox. Partition keys are computed
/// by transforming columns in a table.
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
/// The partition spec contains:
/// - Unique ID for versioning and evolution.
/// - Which source columns in current table schema to use for partitioning
/// (identified by field name, not field ID as in the Iceberg spec).
/// - What transforms to apply (identity, bucket, truncate etc.).
/// - Transform parameters (e.g., bucket count, truncate width).
struct IcebergPartitionSpec {
  struct Field {
    /// Column name as defined in table schema. This column's value is used to
    /// compute partition key by applying 'transformType' transformation.
    const std::string name;

    /// Column type.
    const TypePtr type;

    /// Transform to apply. Callers must ensure the transform is compatible with
    /// the column type.
    const TransformType transformType;

    /// Optional parameter for transforms that require configuration.
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
          return type;
      }
      VELOX_UNREACHABLE("Unknown transform type");
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
  /// @throws VeloxUserError if validation fails.
  IcebergPartitionSpec(int32_t _specId, std::vector<Field> _fields)
      : specId(_specId), fields(std::move(_fields)) {
    checkCompatibility();
  }

 private:
  // Validates partition fields for correctness.
  // Checks type/transform compatibility and transform combination rules.
  void checkCompatibility() const;
};

using IcebergPartitionSpecPtr = std::shared_ptr<const IcebergPartitionSpec>;

} // namespace facebook::velox::connector::hive::iceberg
