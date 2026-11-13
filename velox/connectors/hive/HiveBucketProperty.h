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

#include "velox/common/EnumDeclare.h"
#include "velox/common/serialization/Serializable.h"
#include "velox/core/PlanNode.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive {

/// A column that rows within a bucket file are sorted on, and the sort order.
class HiveSortingColumn : public ISerializable {
 public:
  /// Throws if 'sortColumn' is empty, or if 'sortOrder' is other than
  /// ascending with nulls first or descending with nulls last, the two orders
  /// Hive supports.
  HiveSortingColumn(
      const std::string& sortColumn,
      const core::SortOrder& sortOrder);

  const std::string& sortColumn() const {
    return sortColumn_;
  }

  core::SortOrder sortOrder() const {
    return sortOrder_;
  }

  folly::dynamic serialize() const override;

  static std::shared_ptr<HiveSortingColumn> deserialize(
      const folly::dynamic& obj,
      void* context);

  /// Returns the column name and sort order, for debugging.
  std::string toString() const;

  /// Registers deserialize() with the serialization registry.
  static void registerSerDe();

 private:
  const std::string sortColumn_;
  const core::SortOrder sortOrder_;
};

using HiveSortingColumnPtr = std::shared_ptr<const HiveSortingColumn>;

/// Describes how a Hive table is bucketed: the hash function, the number of
/// buckets, the bucket columns, and the sort columns within each bucket file.
class HiveBucketProperty : public ISerializable {
 public:
  /// The hash function that assigns rows to buckets. Velox implements only
  /// kHiveCompatible.
  enum class Kind {
    /// Hive's hash function, implemented by HivePartitionFunction.
    kHiveCompatible,
    /// Presto's hash function: each column's value is hashed with the type's
    /// hash operator, and column hashes combine as 31 * h + hash. See
    /// getBucket in Presto's com.facebook.presto.hive.HiveBucketing.
    kPrestoNative,
  };

  VELOX_DECLARE_EMBEDDED_ENUM_NAME(Kind);

  /// Throws if 'bucketCount' is not positive, 'bucketedBy' is empty, or
  /// 'bucketedTypes' has a different size than 'bucketedBy'.
  HiveBucketProperty(
      Kind kind,
      int32_t bucketCount,
      const std::vector<std::string>& bucketedBy,
      const std::vector<TypePtr>& bucketedTypes,
      const std::vector<HiveSortingColumnPtr>& sortedBy);

  /// Returns the hash function that assigns rows to buckets.
  Kind kind() const {
    return kind_;
  }

  /// Returns the number of buckets.
  int32_t bucketCount() const {
    return bucketCount_;
  }

  /// Returns the names of the bucket columns, in the order they are hashed.
  const std::vector<std::string>& bucketedBy() const {
    return bucketedBy_;
  }

  /// Returns the types of the bucket columns, in the order of bucketedBy().
  const std::vector<TypePtr>& bucketedTypes() const {
    return bucketedTypes_;
  }

  /// Returns the sort columns within each bucket file. Empty when the files are
  /// unsorted.
  const std::vector<HiveSortingColumnPtr>& sortedBy() const {
    return sortedBy_;
  }

  folly::dynamic serialize() const override;

  static std::shared_ptr<HiveBucketProperty> deserialize(
      const folly::dynamic& obj,
      void* context);

  /// Registers deserialize() with the serialization registry.
  static void registerSerDe();

  /// Returns a multi-line description, for debugging.
  std::string toString() const;

 private:
  const Kind kind_;
  const int32_t bucketCount_;
  const std::vector<std::string> bucketedBy_;
  const std::vector<TypePtr> bucketedTypes_;
  const std::vector<HiveSortingColumnPtr> sortedBy_;
};

using HiveBucketPropertyPtr = std::shared_ptr<const HiveBucketProperty>;

} // namespace facebook::velox::connector::hive

VELOX_EMBEDDED_ENUM_FORMATTER(
    facebook::velox::connector::hive::HiveBucketProperty,
    Kind);
