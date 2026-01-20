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

#include <folly/json/dynamic.h>

namespace facebook::velox::dwio::common {

/// Statistics for an Iceberg parquet data file, corresponding to the data_file
/// struct in the Iceberg specification. All column-level statistics maps use
/// Iceberg field IDs (int32_t) as keys, which uniquely identify columns in the
/// Iceberg schema independent of column names or positions.
struct DataFileStatistics {
  int64_t numRecords;
  std::unordered_map<int32_t, int64_t> columnsSizes;
  std::unordered_map<int32_t, int64_t> valueCounts;
  std::unordered_map<int32_t, int64_t> nullValueCounts;
  std::unordered_map<int32_t, int64_t> nanValueCounts;
  std::unordered_map<int32_t, std::string> lowerBounds;
  std::unordered_map<int32_t, std::string> upperBounds;

  /// Split offsets for the data file. For example, all row
  /// group offsets in a Parquet file. Must be sorted ascending.
  std::vector<uint64_t> splitOffsets;

  folly::dynamic toJson() const;

  folly::dynamic splitOffsetsAsJson() const;
};

} // namespace facebook::velox::dwio::common
