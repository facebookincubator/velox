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

#include <cstdint>
#include <optional>
#include <string>

#include <folly/container/F14Map.h>
#include <folly/json/dynamic.h>

namespace facebook::velox::connector::hive::iceberg {

/// Statistics for an Iceberg data file, corresponding to the `data_file`
/// structure defined in the Iceberg specification:
/// https://iceberg.apache.org/spec/#data-file-fields.
///
/// All column-level statistics maps are keyed by Iceberg field IDs (`int32_t`),
/// which uniquely identify columns in the Iceberg schema independent of column
/// names or physical column positions.
struct IcebergDataFileStatistics {
  struct ColumnStats {
    int64_t columnSize{0};

    /// Total number of values for this field ID in the file, including null and
    /// NaN values.
    ///
    /// For primitive (flat) columns, this is equal to the number of rows in the
    /// file: numRows = valueCount = (nonNullValues + numNulls + numNaNs).
    ///
    /// For nested columns (e.g. elements inside an array), this represents the
    /// total occurrences of the field across all rows, which is not necessarily
    /// related to the top-level record count.
    int64_t valueCount{0};
    int64_t nullValueCount{0};
    std::optional<int64_t> nanValueCount;
    /// Base64 encoded lower bound.
    std::optional<std::string> lowerBound;
    /// Base64 encoded upper bound.
    std::optional<std::string> upperBound;
  };

  int64_t numRecords{0};
  folly::F14FastMap<int32_t, ColumnStats> columnStats;

  /// Returns a IcebergDataFileStatistics with all values set to zero/empty.
  /// Useful for empty data files that have no actual data.
  static IcebergDataFileStatistics empty() {
    return IcebergDataFileStatistics{.numRecords = 0, .columnStats = {}};
  }

  folly::dynamic toJson() const;
};

using IcebergDataFileStatisticsPtr = std::shared_ptr<IcebergDataFileStatistics>;

} // namespace facebook::velox::connector::hive::iceberg
