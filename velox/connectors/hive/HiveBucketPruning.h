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

#include <optional>
#include <vector>

#include "velox/type/Filter.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive {

/// Given filters on the bucket columns of a Hive bucketed table, finds the
/// buckets a scan must read; it can skip the files of the others. Lists the
/// values each filter allows, then hashes every combination with Hive's hash,
/// as the writer does (see HivePartitionFunction).
///
/// It runs twice for a scan. Split generation runs it with the static filters
/// and makes no splits for the other buckets. A data source runs it again
/// with the dynamic filters, such as a join's filter on its keys, which exist
/// only once the query runs, and skips the splits of the other buckets
/// without opening their files.
///
/// Example, for a table bucketed 16 ways on (a BIGINT, b VARCHAR) and the
/// filters a IN (1, 2) and b = 'x':
///
///   auto buckets = HiveBucketPruning::buckets(
///       {filterOnA.get(), filterOnB.get()}, {BIGINT(), VARCHAR()}, 16);
///   // 'buckets' holds the buckets of (1, 'x') and (2, 'x').
class HiveBucketPruning {
 public:
  /// Bounds the cost of buckets(). Folding in a bucket column hashes each of
  /// its values with each distinct hash of the columns before it. buckets()
  /// gives up when that is more than kMaxCombinations hashes.
  static constexpr int64_t kMaxCombinations{10'000};

  /// Returns the buckets, of 'numBuckets', that can hold rows passing
  /// 'filters': distinct, in no particular order, and empty when no row can
  /// pass. Returns std::nullopt when the filters don't narrow the buckets
  /// down: a bucket column has no filter, a filter passes NULL, a filter's
  /// values can't be listed (e.g. a range of strings), or listing them takes
  /// more than kMaxCombinations hashes.
  ///
  /// Lists the values of filters on columns stored as TINYINT, SMALLINT,
  /// INTEGER or BIGINT (single values, value lists, and ranges), VARCHAR
  /// (single values and value lists) or BOOLEAN. This includes logical types
  /// stored as these, such as DATE and short DECIMAL: Hive's hash depends only
  /// on how a value is stored, so the buckets match HivePartitionFunction's
  /// for every such type.
  ///
  /// @param filters The filter on each bucket column, in bucketing order, or
  /// nullptr for a column with no filter.
  /// @param types The type of each bucket column, in bucketing order. A filter
  /// doesn't carry its column's type, and the bucket depends on it: Hive
  /// hashes -1 to -1 as an INTEGER but to 0 as a BIGINT. The type's range also
  /// narrows a filter's values: on a TINYINT column, x > 120 allows only 121
  /// to 127.
  static std::optional<std::vector<int32_t>> buckets(
      const std::vector<const common::Filter*>& filters,
      const std::vector<TypePtr>& types,
      int32_t numBuckets);
};

} // namespace facebook::velox::connector::hive
