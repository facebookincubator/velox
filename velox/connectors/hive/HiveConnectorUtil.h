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
#include <folly/Executor.h>
#include <folly/container/F14Map.h>

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"

namespace facebook::velox::exec {
class Expr;
class FieldReference;
} // namespace facebook::velox::exec

namespace facebook::velox::connector::hive {

class HiveColumnHandle;
class HiveTableHandle;
class HiveConfig;
struct HiveConnectorSplit;

const std::string& getColumnName(const common::Subfield& subfield);

void checkColumnNameLowerCase(const TypePtr& type);

void checkColumnNameLowerCase(
    const common::SubfieldFilters& filters,
    const std::unordered_map<
        std::string,
        std::shared_ptr<const HiveColumnHandle>>& infoColumns);

void checkColumnNameLowerCase(const core::TypedExprPtr& typeExpr);

struct SpecialColumnNames {
  std::optional<std::string> rowIndex;
  std::optional<std::string> rowId;
};

/// Checks that two HiveColumnHandle instances are consistent in terms of
/// column type, data type, and hive type. Throws if inconsistent.
void checkColumnHandleConsistent(
    const HiveColumnHandle& x,
    const HiveColumnHandle& y);

/// Creates a ScanSpec for reading data from a Hive table.
///
/// The ScanSpec describes which columns to read and what filters to apply.
/// It handles several types of columns:
/// - Regular data columns from the file
/// - Partition key columns (values from file path)
/// - Synthesized columns (e.g., $path, $bucket)
/// - Special columns (e.g., row index, row ID)
/// - Index columns for index lookup joins
///
/// @param rowType Schema of columns to be projected in the output.
/// @param outputSubfields Map of column names to subfields that need to be
///     read. Used for pruning nested structures.
/// @param subfieldFilters Map of subfields to filters to apply during scan.
/// @param indexColumns Column names used for index lookup joins. These columns
///     are added to the scan spec even if they are not in the output
///     projection, ensuring they are read from the file for join key matching.
/// @param dataColumns Full schema of all columns in the data file. Used to
///     look up column types when a column is referenced in filters or index
///     columns but not in the output projection.
/// @param partitionKeys Map of partition column names to their handles.
///     Partition columns are not read from the file.
/// @param infoColumns Map of synthesized column names (e.g., $path) to their
///     handles.
/// @param specialColumns Names of special columns like row index and row ID.
/// @param disableStatsBasedFilterReorder If true, disables reordering of
///     filters based on statistics.
/// @param pool Memory pool for allocations during scan spec construction.
/// @return A ScanSpec that can be used to configure a reader.
std::shared_ptr<common::ScanSpec> makeScanSpec(
    const RowTypePtr& rowType,
    const folly::F14FastMap<std::string, std::vector<const common::Subfield*>>&
        outputSubfields,
    const common::SubfieldFilters& subfieldFilters,
    const std::vector<std::string>& indexColumns,
    const RowTypePtr& dataColumns,
    const std::unordered_map<
        std::string,
        std::shared_ptr<const HiveColumnHandle>>& partitionKeys,
    const std::unordered_map<
        std::string,
        std::shared_ptr<const HiveColumnHandle>>& infoColumns,
    const SpecialColumnNames& specialColumns,
    bool disableStatsBasedFilterReorder,
    memory::MemoryPool* pool);

/// Overload without indexColumns for backward compatibility.
inline std::shared_ptr<common::ScanSpec> makeScanSpec(
    const RowTypePtr& rowType,
    const folly::F14FastMap<std::string, std::vector<const common::Subfield*>>&
        outputSubfields,
    const common::SubfieldFilters& subfieldFilters,
    const RowTypePtr& dataColumns,
    const std::unordered_map<
        std::string,
        std::shared_ptr<const HiveColumnHandle>>& partitionKeys,
    const std::unordered_map<
        std::string,
        std::shared_ptr<const HiveColumnHandle>>& infoColumns,
    const SpecialColumnNames& specialColumns,
    bool disableStatsBasedFilterReorder,
    memory::MemoryPool* pool) {
  return makeScanSpec(
      rowType,
      outputSubfields,
      subfieldFilters,
      /*indexColumns=*/{},
      dataColumns,
      partitionKeys,
      infoColumns,
      specialColumns,
      disableStatsBasedFilterReorder,
      pool);
}

void configureReaderOptions(
    const std::shared_ptr<const HiveConfig>& config,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const HiveTableHandle>& hiveTableHandle,
    const std::shared_ptr<const HiveConnectorSplit>& hiveSplit,
    dwio::common::ReaderOptions& readerOptions);

void configureReaderOptions(
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    const ConnectorQueryCtx* connectorQueryCtx,
    const RowTypePtr& fileSchema,
    const std::shared_ptr<const HiveConnectorSplit>& hiveSplit,
    const std::unordered_map<std::string, std::string>& tableParameters,
    dwio::common::ReaderOptions& readerOptions);

void configureRowReaderOptions(
    const std::unordered_map<std::string, std::string>& tableParameters,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    const RowTypePtr& rowType,
    const std::shared_ptr<const HiveConnectorSplit>& hiveSplit,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    const config::ConfigBase* sessionProperties,
    folly::Executor* ioExecutor,
    dwio::common::RowReaderOptions& rowReaderOptions);

bool testFilters(
    const common::ScanSpec* scanSpec,
    const dwio::common::Reader* reader,
    const std::string& filePath,
    const std::unordered_map<std::string, std::optional<std::string>>&
        partitionKey,
    const std::unordered_map<
        std::string,
        std::shared_ptr<const HiveColumnHandle>>& partitionKeysHandle,
    bool asLocalTime);

std::unique_ptr<dwio::common::BufferedInput> createBufferedInput(
    const FileHandle& fileHandle,
    const dwio::common::ReaderOptions& readerOpts,
    const ConnectorQueryCtx* connectorQueryCtx,
    std::shared_ptr<io::IoStatistics> ioStats,
    std::shared_ptr<filesystems::File::IoStats> fsStats,
    folly::Executor* executor,
    const folly::F14FastMap<std::string, std::string>& fileReadOps = {});

/// Given a boolean expression, breaks it up into conjuncts and sorts these into
/// single-column comparisons with constants (filters), rand() < sampleRate, and
/// the rest (return value).
///
/// Multiple rand() < K conjuncts are combined into a single sampleRate by
/// multiplying individual sample rates. rand() < 0.1 and rand() < 0.2 produces
/// sampleRate = 0.02.
///
/// Multiple single-column comparisons with constants that reference the same
/// column or subfield are combined into a single filter. Pre-existing entries
/// in 'filters' are preserved and combined with the ones extracted from the
/// 'expr'.
///
/// NOT(x OR y) is converted to (NOT x) AND (NOT y).
///
/// @param expr Boolean expression to break up.
/// @param evaluator Expression evaluator to use.
/// @param filters Mapping from a column or a subfield to comparison with
/// constant.
/// @param sampleRate Sample rate extracted from rand() < sampleRate conjuncts.
/// @return Expression with filters and rand() < sampleRate conjuncts removed.
///
/// Examples:
///   expr := a = 1 AND b > 0
///   filters := {a: eq(1), b: gt(0)}
//    sampleRate left unmodified
//    return value is nullptr
///
///   expr: not (a > 0 or b > 10)
///   filters := {a: le(0), b: le(10)}
///   sampleRate left unmodified
///   return value is nullptr
///
///   expr := a > 0 AND a < b AND rand() < 0.1
///   filters := {a: gt(0)}
///   sampleRate := 0.1
///   return value is a < b
core::TypedExprPtr extractFiltersFromRemainingFilter(
    const core::TypedExprPtr& expr,
    core::ExpressionEvaluator* evaluator,
    common::SubfieldFilters& filters,
    double& sampleRate);

/// Determines whether a field referenced in the remaining filter should be
/// eagerly materialized (loaded upfront) or can be lazily loaded.
///
/// Returns true (eager materialization needed) when:
/// 1. The remaining filter is NOT an AND expression (e.g., OR), because row
///    access patterns are unpredictable.
/// 2. The field is used within a conditional sub-expression (IF, CASE, nested
///    AND/OR) of an AND expression, because the conditional may access rows
///    unpredictably.
///
/// Returns false (lazy loading OK) when the remaining filter is an AND
/// expression and the field is only used in simple, non-conditional conjuncts.
///
/// @param remainingFilter The compiled remaining filter expression.
/// @param field The field reference to check.
/// @return true if the field should be eagerly materialized.
bool shouldEagerlyMaterialize(
    const exec::Expr& remainingFilter,
    const exec::FieldReference& field);

/// Creates a point lookup filter from a variant value.
/// Null values are not allowed.
/// Supports TINYINT, SMALLINT, INTEGER, BIGINT, REAL, DOUBLE, BOOLEAN,
/// VARCHAR, and VARBINARY types.
/// @param type The type of the value.
/// @param value The filter value (must not be null).
/// @return A filter for point lookup, or nullptr if type is not supported.
std::unique_ptr<common::Filter> createPointFilter(
    const TypePtr& type,
    const variant& value);

/// Creates a range filter from two variant values.
/// Both lower and upper bounds are inclusive. Null values are not allowed.
/// Supports TINYINT, SMALLINT, INTEGER, BIGINT, REAL, DOUBLE, VARCHAR,
/// and VARBINARY types.
/// @param type The type of the values.
/// @param lower The lower bound value.
/// @param upper The upper bound value.
/// @return A filter for range lookup, or nullptr if type is not supported.
std::unique_ptr<common::Filter> createRangeFilter(
    const TypePtr& type,
    const variant& lower,
    const variant& upper);

} // namespace facebook::velox::connector::hive
