/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/common/io/IoStatistics.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveIndexReader.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/Expr.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::connector::hive {

class HiveConfig;

/// HiveIndexSource provides index lookup for Hive tables.
/// Currently supports Nimble file format with TabletIndex for efficient
/// key-based stripe lookups to enable IndexLookupJoin.
///
/// NOTE: This is a scaffold implementation that assumes a single Nimble file.
/// The user may need to optimize and extend this for production use cases.
class HiveIndexSource : public IndexSource,
                        public std::enable_shared_from_this<HiveIndexSource> {
 public:
  HiveIndexSource(
      const RowTypePtr& requestType,
      const std::vector<core::IndexLookupConditionPtr>& joinConditions,
      const RowTypePtr& outputType,
      HiveTableHandlePtr tableHandle,
      const ColumnHandleMap& columnHandles,
      FileHandleFactory* fileHandleFactory,
      ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<HiveConfig>& hiveConfig,
      folly::Executor* executor);

  ~HiveIndexSource() override = default;

  void addSplits(std::vector<std::shared_ptr<ConnectorSplit>> splits) override;

  std::shared_ptr<ResultIterator> lookup(const Request& request) override;

  std::unordered_map<std::string, RuntimeMetric> runtimeStats() override;

 private:
  friend class HiveLookupIterator;

  memory::MemoryPool* pool() const {
    return pool_;
  }

  const RowTypePtr& outputType() const {
    return outputType_;
  }

  const RowVectorPtr& getEmptyOutput() {
    if (!emptyOutput_) {
      emptyOutput_ = RowVector::createEmpty(outputType_, pool_);
    }
    return emptyOutput_;
  }

  // Evaluates remainingFilter on the specified vector. Returns number of rows
  // passed. Populates filterEvalCtx_.selectedIndices and selectedBits if only
  // some rows passed the filter. If none or all rows passed
  // filterEvalCtx_.selectedIndices and selectedBits are not updated.
  vector_size_t evaluateRemainingFilter(RowVectorPtr& rowVector);

  // Projects output from reader output to the final output type. Wraps child
  // vectors with remaining filter indices if provided.
  RowVectorPtr projectOutput(
      vector_size_t numRows,
      const BufferPtr& remainingIndices,
      const RowVectorPtr& rowVector);

  void init(
      const ColumnHandleMap& assignments,
      const std::vector<core::IndexLookupConditionPtr>& joinConditions);

  // Validates and initializes join conditions:
  // - Converts filter conditions (with constant values) to filters_.
  // - Non-filter conditions are stored in joinConditions_.
  // Returns a list of index column names used by joinConditions_.
  std::vector<std::string> initJoinConditions(
      const std::vector<core::IndexLookupConditionPtr>& joinConditions,
      const ColumnHandleMap& assignments);

  // Initializes the remaining filter:
  // - Compiles the remaining filter expression.
  // - Identifies columns to eagerly materialize.
  // - Adds columns referenced by remaining filter but not projected.
  // - Extracts and processes subfields from the remaining filter.
  void initRemainingFilter(
      std::vector<std::string>& readColumnNames,
      std::vector<TypePtr>& readColumnTypes);

  // Creates HiveIndexReader for the splits.
  void createHiveIndexReader(
      std::vector<std::shared_ptr<const HiveConnectorSplit>> splits);

  FileHandleFactory* const fileHandleFactory_;
  ConnectorQueryCtx* const connectorQueryCtx_;
  const std::shared_ptr<HiveConfig> hiveConfig_;
  memory::MemoryPool* const pool_;
  core::ExpressionEvaluator* const expressionEvaluator_;

  const HiveTableHandlePtr tableHandle_;
  const RowTypePtr requestType_;
  const RowTypePtr outputType_;
  folly::Executor* const executor_;

  // All join conditions including equal join keys converted to
  // EqualIndexLookupConditions and original non-filter join conditions.
  // This is passed to HiveIndexReader.
  std::vector<core::IndexLookupConditionPtr> joinConditions_;

  // Filters for pushdown. Includes subfield filters from table handle and
  // filters converted from constant join conditions.
  common::SubfieldFilters filters_;

  // Remaining filter expression set after filter pushdown.
  std::unique_ptr<exec::ExprSet> remainingFilterExprSet_;
  // Subfields referenced by the remaining filter.
  std::vector<common::Subfield> remainingFilterSubfields_;
  // Total time spent on evaluating remaining filter in nanoseconds.
  uint64_t remainingFilterTimeNs_{0};
  // Field indices referenced in both remaining filter and output type. These
  // columns need to be materialized eagerly to avoid missing values in output.
  std::vector<column_index_t> remainingEagerlyLoadFields_;

  // Filter evaluation state for remaining filter.
  SelectivityVector remainingFilterRows_;
  VectorPtr remainingFilterResult_;
  DecodedVector remainingFilterLazyDecoded_;
  SelectivityVector remainingFilterLazyBaseRows_;
  exec::FilterEvalCtx remainingFilterEvalCtx_;

  // Points to subfields from both output columns' required subfields and
  // remainingFilterSubfields_. Used to tell the reader which nested fields to
  // project out.
  folly::F14FastMap<std::string, std::vector<const common::Subfield*>>
      projectedSubfields_;

  // Scan spec for the index reader.
  std::shared_ptr<common::ScanSpec> scanSpec_;
  // Output type for the index reader.
  RowTypePtr readerOutputType_;
  /// Index reader created by addSplits().
  std::unique_ptr<HiveIndexReader> indexReader_;

  // Cached empty output vector.
  RowVectorPtr emptyOutput_;

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  std::shared_ptr<IoStats> ioStats_;
};

} // namespace facebook::velox::connector::hive
