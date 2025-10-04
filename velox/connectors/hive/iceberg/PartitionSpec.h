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

#include "velox/common/Enums.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

/// Partition transform types supported by Iceberg.
///
/// These transforms define how source column values are converted into
/// partition values. Each transform type corresponds to a specific
/// partitioning strategy:
/// - kIdentity: Use the source value as-is (no transformation).
/// - kHour/kDay/kMonth/kYear: Extract time components from timestamps/dates.
/// - kBucket: Hash the value into N buckets for even distribution.
/// - kTruncate: Truncate strings/numbers to a specified width.
/// See https://iceberg.apache.org/spec/#partition-transforms.
enum class TransformType {
  kIdentity,
  kHour,
  kDay,
  kMonth,
  kYear,
  kBucket,
  kTruncate
};

VELOX_DECLARE_ENUM_NAME(TransformType);

/// Represents how to produce partition data for an Iceberg table.
///
/// This structure corresponds to the Iceberg Java PartitionSpec class but
/// contains only the necessary fields for Velox. Partition data is produced
/// by transforming columns in a table, where each column transform is
/// represented by a named Field.
///
/// The partition spec defines:
/// - Unique partition spec ID for versioning and evolution.
/// - Which source columns to use for partitioning.
/// - What transforms to apply (identity, bucket, truncate etc.).
/// - Transform parameters (e.g., bucket count, truncate width).
struct IcebergPartitionSpec {
  struct Field {
    /// The field name of this partition field as it appears in the partition
    /// spec. This is the original Iceberg field name, not the transformed name
    /// from org.apache.iceberg.PartitionField which includes the transform as a
    /// suffix.
    std::string name;

    /// The source column type.
    TypePtr type;

    /// The transform type applied to the source field (e.g., kIdentity,
    /// kBucket, kTruncate, etc.).
    TransformType transformType;

    /// Optional parameter for transforms that require configuration
    /// (e.g., bucket count or truncate width).
    std::optional<int32_t> parameter;
  };

  const int32_t specId;
  const std::vector<Field> fields;

  IcebergPartitionSpec(int32_t _specId, const std::vector<Field> _fields)
      : specId(_specId), fields(std::move(_fields)) {}
};

using IcebergPartitionSpecPtr = std::shared_ptr<const IcebergPartitionSpec>;

} // namespace facebook::velox::connector::hive::iceberg
