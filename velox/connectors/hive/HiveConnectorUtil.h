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

std::shared_ptr<common::ScanSpec> makeScanSpec(
    const RowTypePtr& rowType,
    const folly::F14FastMap<std::string, std::vector<const common::Subfield*>>&
        outputSubfields,
    const common::SubfieldFilters& filters,
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
    bool asLocalTime,
    dwio::common::FileFormat fileFormat);

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

} // namespace facebook::velox::connector::hive
