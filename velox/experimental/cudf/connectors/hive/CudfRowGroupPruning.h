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

#include "velox/type/Filter.h"
#include "velox/type/Type.h"

#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>

#include <optional>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// Row groups kept by the host statistics test of
/// pruneRowGroupsWithStatistics().
struct HostRowGroupPruning {
  /// Per source, the row groups whose statistics do not rule out every row,
  /// in the order they were given.
  std::vector<std::vector<cudf::size_type>> rowGroupIndices;

  /// Whether the statistics decided every filter on every row group. When
  /// false, some row group was kept because a filter could not be decided on
  /// it, so cuDF's statistics pass still has to run.
  bool complete{true};
};

/// The sorted values of an IN list the host test decides value by value.
struct InListValues {
  std::vector<int64_t> int64s;
  std::vector<int128_t> int128s;
  std::vector<std::string> strings;
};

/// The subfield filters a data source prepared for the host statistics test
/// of pruneRowGroupsWithStatistics(). The filters must outlive it.
struct HostRowGroupPruningFilters {
  struct Column {
    std::string name;
    TypePtr type;
    const common::Filter* filter;
    /// Set for the IN lists the test decides value by value.
    std::optional<InListValues> inList;
  };

  /// The filters on top-level primitive columns of the table.
  std::vector<Column> columns;

  /// Whether every filter applies to a top-level column of the table. When
  /// false, cuDF's statistics pass has to run for the others.
  bool complete{true};
};

/// Resolves 'filters' against 'tableType' for pruneRowGroupsWithStatistics():
/// the type of each filtered column and the sorted values of each IN list.
HostRowGroupPruningFilters prepareRowGroupPruningFilters(
    const common::SubfieldFilters& filters,
    const RowType& tableType);

/// Tests the prepared subfield filters against the Parquet column-chunk
/// statistics of 'files' on the host and returns the row groups of
/// 'rowGroupIndices' that may hold a passing row. File column names are
/// lower-cased before they are matched when 'caseInsensitiveColumnNames' is
/// set.
///
/// The decision is common::testFilter(), the row-group test of the CPU Parquet
/// reader, over the bounds and null counts cuDF's statistics pass reads, plus
/// a value test for decimals, booleans, IN lists and floating-point ranges,
/// which that test leaves open or decides more loosely than cuDF's pass. It
/// covers integer, DATE, REAL, DOUBLE, BOOLEAN, VARCHAR, VARBINARY and INT64
/// TIMESTAMP columns stored as their Velox kind reads them, decimals stored at
/// the table's scale, and IS NULL / IS NOT NULL on any column. A filtered
/// column the file lacks, a repeated or nested column and a row group without
/// statistics are kept; an empty row group is pruned. A row group the host
/// cannot decide (a column stored as another physical type, an unsigned
/// integer, a timestamp without a unit, a decimal at another scale or with a
/// bound only in the deprecated byte-array fields, a chunk with one bound or
/// a NaN bound, an IN list inside an OR, a filter without a range test for
/// the column's type) is kept and clears 'complete'.
HostRowGroupPruning pruneRowGroupsWithStatistics(
    const std::vector<cudf::io::parquet::FileMetaData>& files,
    const std::vector<std::vector<cudf::size_type>>& rowGroupIndices,
    const HostRowGroupPruningFilters& filters,
    bool caseInsensitiveColumnNames);

} // namespace facebook::velox::cudf_velox::connector::hive
