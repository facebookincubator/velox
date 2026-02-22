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

#include <memory>
#include "velox/vector/ComplexVector.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::connector::hive {

/// Converting partition values to their string representations.
/// Provides template methods for formatting different data types according to
/// Hive partitioning conventions.
class HivePartitionName {
 public:
  /// Generic template for formatting partition values to strings using
  /// fmt::to_string. Specialized for types that need special handling
  /// (int32_t, int64_t, int128_t, Timestamp).
  template <typename T>
  FOLLY_ALWAYS_INLINE static std::string toName(T value, const TypePtr& type) {
    return fmt::to_string(value);
  }

  /// Format int32_t partition values. Specialized to handle DATE type which
  /// requires ISO-8601 formatting (YYYY-MM-DD) instead of raw integer value.
  static std::string toName(int32_t value, const TypePtr& type);

  /// Format int64_t partition values. Specialized to handle short DECIMAL type
  /// which requires decimal string formatting with proper precision and scale
  /// instead of raw integer value.
  static std::string toName(int64_t value, const TypePtr& type);

  /// Format int128_t partition values. Specialized to handle long DECIMAL type
  /// which requires decimal string formatting with proper precision and scale
  /// instead of raw integer value.
  static std::string toName(int128_t value, const TypePtr& type);

  /// Format Timestamp partition values. Specialized to:
  /// 1. Convert to default timezone
  /// 2. Use space as date-time separator (not 'T')
  /// 3. Use millisecond precision with trailing zeros skipped
  /// 4. Always keep at least ".0" for fractional seconds (Presto compatibility)
  static std::string toName(Timestamp value, const TypePtr& type);

  /// Build partition key-value pairs from partition values.
  /// Returns a vector of (key, value) pairs for all partition columns.
  /// @tparam F A callable that converts a value to a partition string.
  ///         Takes (value, type, columnIndex) and returns string.
  /// @param partitionId The partition ID (row index) to extract values from.
  /// @param partitionValues RowVector containing partition values.
  /// @param nullValueString The string to use for null values.
  /// @param toPartitionName Callable to convert a value to a string.
  template <typename F>
  static std::vector<std::pair<std::string, std::string>> partitionKeyValues(
      uint32_t partitionId,
      const RowVectorPtr& partitionValues,
      const std::string& nullValueString,
      const F& toPartitionName);

  /// Generate a Hive partition directory name from partition values for
  /// partitionId.
  ///
  /// @param partitionId The row index in partitionValues to extract values
  ///        from.
  /// @param partitionValues RowVector containing partition values. Each
  ///        child vector represents a partition column, and the row at
  ///        partitionId contains the values for this partition.
  /// @param partitionKeyAsLowerCase Controls whether partition column names
  ///        should be converted to lowercase in the output. When true, column
  ///        names are lowercased (e.g., "year=2025"); when false, original
  ///        casing is preserved (e.g., "Year=2025").
  /// @return A formatted partition directory name string. Null values are
  /// represented as __HIVE_DEFAULT_PARTITION__.
  static std::string partitionName(
      uint32_t partitionId,
      const RowVectorPtr& partitionValues,
      bool partitionKeyAsLowerCase);
};

namespace detail {

// Unified template function to extract partition key-value string from a
// vector. Used by both Hive and Iceberg partition name generators.
//
// @tparam Kind The TypeKind of the partition column.
// @tparam F A callable that converts a value to a partition string.
// @param partitionVector The vector containing partition values.
// @param row The row index to extract the value from.
// @param type The type of the partition column.
// @param columnIndex The column index in the partition values.
// @param toPartitionName Callable to convert a value to a partition string.
// @return A pair of (column_name, formatted_value).
template <TypeKind Kind, typename F>
std::string makePartitionKeyValueString(
    const BaseVector& partitionVector,
    vector_size_t row,
    const TypePtr& type,
    int columnIndex,
    const F& toPartitionName) {
  using T = typename TypeTraits<Kind>::NativeType;

  return toPartitionName(
      partitionVector.as<SimpleVector<T>>()->valueAt(row), type, columnIndex);
}

#define PARTITION_TYPE_DISPATCH(TEMPLATE_FUNC, typeKind, ...)                  \
  [&]() {                                                                      \
    switch (typeKind) {                                                        \
      case TypeKind::BOOLEAN:                                                  \
      case TypeKind::TINYINT:                                                  \
      case TypeKind::SMALLINT:                                                 \
      case TypeKind::INTEGER:                                                  \
      case TypeKind::BIGINT:                                                   \
      case TypeKind::VARCHAR:                                                  \
      case TypeKind::VARBINARY:                                                \
      case TypeKind::TIMESTAMP:                                                \
        return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(                             \
            TEMPLATE_FUNC, typeKind, __VA_ARGS__);                             \
      default:                                                                 \
        VELOX_UNSUPPORTED(                                                     \
            "Unsupported partition type: {}", TypeKindName::toName(typeKind)); \
    }                                                                          \
  }()

} // namespace detail

template <typename F>
std::vector<std::pair<std::string, std::string>>
HivePartitionName::partitionKeyValues(
    uint32_t partitionId,
    const RowVectorPtr& partitionValues,
    const std::string& nullValueString,
    const F& toPartitionName) {
  std::vector<std::pair<std::string, std::string>> partitionKeyValuePairs;
  for (auto i = 0; i < partitionValues->childrenSize(); i++) {
    const auto& child = partitionValues->childAt(i);
    const auto& name = partitionValues->rowType()->nameOf(i);
    if (child->isNullAt(partitionId)) {
      partitionKeyValuePairs.emplace_back(
          std::make_pair(name, nullValueString));
      continue;
    }

    partitionKeyValuePairs.emplace_back(
        std::make_pair(
            name,
            PARTITION_TYPE_DISPATCH(
                detail::makePartitionKeyValueString,
                child->typeKind(),
                *child->loadedVector(),
                partitionId,
                child->type(),
                i,
                toPartitionName)));
  }
  return partitionKeyValuePairs;
}

} // namespace facebook::velox::connector::hive
