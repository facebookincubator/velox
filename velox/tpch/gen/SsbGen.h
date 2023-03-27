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

#include "velox/common/memory/Memory.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::ssb {

/// This file uses TPC-H DBGEN to generate data encoded using Velox Vectors.
///
/// The basic input for the API is the TPC-H table name (the Table enum), the
/// TPC-H scale factor, the maximum batch size, and the offset. The common usage
/// is to make successive calls to this API advancing the offset parameter,
/// until all records were read. Clients might also assign different slices of
/// the range "[0, getRowCount(Table, scaleFactor)[" to different threads in
/// order to generate datasets in parallel.
///
/// If not enough records are available given a particular scale factor and
/// offset, less than maxRows records might be returned.
///
/// Data is always returned in a RowVector.

enum class SSB_Table : uint8_t {
  TBL_SUPPLIER,
  TBL_CUSTOMER,
  TBL_PART,
  TBL_DATE,
  TBL_LINEORDER
};

/// Returns table name as a string.
std::string_view toTableName(SSB_Table table);

/// Returns the table enum value given a table name.
SSB_Table fromTableName(std::string_view tableName);

/// Returns the row count for a particular TPC-H table given a scale factor, as
/// defined in the spec available at:
///
///  https://www.tpc.org/tpch/
size_t getRowCount(SSB_Table table, double scaleFactor);

/// Returns the schema (RowType) for a particular TPC-H table.
RowTypePtr getTableSchema(SSB_Table table);

/// Returns the type of a particular table:column pair. Throws if `columnName`
/// does not exist in `table`.
TypePtr resolveSsbColumn(SSB_Table table, const std::string& columnName);

/// Returns a row vector containing at most `maxRows` rows of the "orders"
/// table, starting at `offset`, and given the scale factor. The row vector
/// returned has the following schema:
///
///  o_orderkey: BIGINT
///  o_custkey: BIGINT
///  o_orderstatus: VARCHAR
///  o_totalprice: DOUBLE
///  o_orderdate: DATE
///  o_orderpriority: VARCHAR
///  o_clerk: VARCHAR
///  o_shippriority: INTEGER
///  o_comment: VARCHAR
///

} // namespace facebook::velox::ssb
