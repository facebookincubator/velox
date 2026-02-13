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

#include <memory>
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::connector::hive {

FOLLY_ALWAYS_INLINE std::string formatDate(int32_t value) {
  return DATE()->toString(value);
}

/// Converting partition values to their string representations.
/// Provides virtual methods for formatting different data types
/// according to Hive partitioning conventions.
class HivePartitionUtil {
 public:
  virtual ~HivePartitionUtil() = default;

  /// Generic template for formatting simple types that just need string
  /// conversion. Specialized/overloaded for types that need special handling.
  template <typename T>
  FOLLY_ALWAYS_INLINE std::string toPartitionString(
      T value,
      const TypePtr& type) const {
    return folly::to<std::string>(value);
  }

  FOLLY_ALWAYS_INLINE std::string toPartitionString(
      bool value,
      const TypePtr& type) const {
    return value ? "true" : "false";
  }

  std::string toPartitionString(int64_t value, const TypePtr& type) const;

  std::string toPartitionString(int128_t value, const TypePtr& type) const;

  virtual std::string toPartitionString(StringView value, const TypePtr& type)
      const {
    return folly::to<std::string>(value);
  }

  virtual std::string toPartitionString(int32_t value, const TypePtr& type)
      const;

  virtual std::string toPartitionString(Timestamp value, const TypePtr& type)
      const;

  /// Extract partition key-value pairs from a row vector at a specific row.
  ///
  /// @param partitionsVector A row vector containing partition columns.
  /// @param row The row index to extract partition values from.
  /// @return A vector of (column_name, formatted_value) pairs.
  static std::vector<std::pair<std::string, std::string>>
  extractPartitionKeyValues(
      const RowVectorPtr& partitionsVector,
      vector_size_t row);
};

using HivePartitionUtilPtr = std::shared_ptr<const HivePartitionUtil>;

} // namespace facebook::velox::connector::hive
