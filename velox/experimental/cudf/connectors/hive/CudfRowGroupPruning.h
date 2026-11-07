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
  /// it, so cuDF's statistics pass still has to run over the kept row groups.
  bool complete{true};
};

/// Tests the subfield filters in 'filters' against the Parquet column-chunk
/// statistics of 'files' on the host and returns the row groups of
/// 'rowGroupIndices' that may hold a passing row. 'tableType' gives the Velox
/// type of each filtered column; file column names are lower-cased before they
/// are matched when 'caseInsensitiveColumnNames' is set.
///
/// The decision is common::testFilter(), the row-group test of the CPU Parquet
/// reader, over the bounds and null counts cuDF's statistics pass reads, plus
/// a value test for decimals, booleans and IN lists, which that test leaves
/// open. It covers integer, DATE, REAL, DOUBLE, BOOLEAN, VARCHAR, VARBINARY
/// and INT64 TIMESTAMP columns stored as their Velox kind reads them, decimals
/// stored at the table's scale, and IS NULL / IS NOT NULL on any column. A
/// filtered column the file lacks, a repeated or nested column and a row group
/// without statistics are kept. A row group the host cannot decide (a column
/// stored as another physical type, an unsigned integer, a timestamp without
/// a unit, a decimal at another scale or with only deprecated byte-array
/// bounds) is kept and clears 'complete'.
HostRowGroupPruning pruneRowGroupsWithStatistics(
    const std::vector<cudf::io::parquet::FileMetaData>& files,
    const std::vector<std::vector<cudf::size_type>>& rowGroupIndices,
    const common::SubfieldFilters& filters,
    const RowType& tableType,
    bool caseInsensitiveColumnNames);

} // namespace facebook::velox::cudf_velox::connector::hive
