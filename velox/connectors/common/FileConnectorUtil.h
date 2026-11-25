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

#include "velox/common/caching/FileHandle.h"
#include "velox/common/config/Config.h"
#include "velox/connectors/Connector.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"

namespace facebook::velox::connector {

class FileColumnHandle;
using FileColumnHandlePtr = std::shared_ptr<const FileColumnHandle>;
class FileTableHandle;
using FileTableHandlePtr = std::shared_ptr<const FileTableHandle>;
class FileConfig;
struct FileConnectorSplit;

struct FormatScopedConfigs {
  config::ConfigBase connectorConfig;
  config::ConfigBase sessionProperties;
};

FormatScopedConfigs makeFormatScopedConfigs(
    const FileConfig& fileConfig,
    const config::ConfigBase& sessionProperties,
    dwio::common::FileFormat fileFormat);

/// Configures reader options for reading a data file. This is the generic
/// version that does not apply serde (serialization/deserialization) options.
/// For Hive tables that need serde options, use the overload in
/// HiveConnectorUtil.h that accepts serdeParameters.
void configureReaderOptions(
    const std::shared_ptr<const FileConfig>& config,
    const ConnectorQueryCtx* connectorQueryCtx,
    const FileTableHandlePtr& tableHandle,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    dwio::common::ReaderOptions& readerOptions);

void configureReaderOptions(
    const std::shared_ptr<const FileConfig>& fileConfig,
    const ConnectorQueryCtx* connectorQueryCtx,
    const RowTypePtr& fileSchema,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    const std::unordered_map<std::string, std::string>& tableParameters,
    dwio::common::ReaderOptions& readerOptions);

/// Configures row reader options for reading rows from a data file. This is the
/// generic version that does not set serde parameters. For Hive tables that
/// need serde parameters, use the overload in HiveConnectorUtil.h.
void configureRowReaderOptions(
    const std::unordered_map<std::string, std::string>& tableParameters,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    const RowTypePtr& rowType,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const config::ConfigBase* sessionProperties,
    folly::Executor* ioExecutor,
    dwio::common::RowReaderOptions& rowReaderOptions);

struct SpecialColumnNames {
  std::optional<std::string> rowIndex;
  std::optional<std::string> rowId;
  std::optional<std::string> readRange;
};

/// Creates a ScanSpec for reading data from files.
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
    const RowTypePtr& dataColumns,
    const std::unordered_map<std::string, FileColumnHandlePtr>& partitionKeys,
    const std::unordered_map<std::string, FileColumnHandlePtr>& infoColumns,
    const SpecialColumnNames& specialColumns,
    bool disableStatsBasedFilterReorder,
    memory::MemoryPool* pool);

/// @deprecated Use the overload without indexColumns parameter instead.
/// This overload is kept for backward compatibility and will be removed in a
/// future release.
std::shared_ptr<common::ScanSpec> makeScanSpec(
    const RowTypePtr& rowType,
    const folly::F14FastMap<std::string, std::vector<const common::Subfield*>>&
        outputSubfields,
    const common::SubfieldFilters& subfieldFilters,
    const std::vector<std::string>& indexColumns,
    const RowTypePtr& dataColumns,
    const std::unordered_map<std::string, FileColumnHandlePtr>& partitionKeys,
    const std::unordered_map<std::string, FileColumnHandlePtr>& infoColumns,
    const SpecialColumnNames& specialColumns,
    bool disableStatsBasedFilterReorder,
    memory::MemoryPool* pool);

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
///
/// A null 'expr', a scan with no remaining filter, returns nullptr.
core::TypedExprPtr extractFiltersFromRemainingFilter(
    const core::TypedExprPtr& expr,
    core::ExpressionEvaluator* evaluator,
    common::SubfieldFilters& filters,
    double& sampleRate);

} // namespace facebook::velox::connector
